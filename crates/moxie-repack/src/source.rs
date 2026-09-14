//! Resolving a selection against the shards that hold it.
//!
//! Every read here goes through `moxie-storage`: this module opens no file of
//! its own, decodes nothing, and owns no second index. What it owns is the
//! question "which shard holds this tensor, and what does its header say",
//! which task 0024 recorded as the gap a production caller would need and
//! deliberately did not build.
//!
//! One shard is kept open at a time. A module's four tensors need not share a
//! shard -- Qwen3.8-27B splits every one of its 256 modules -- so the cache is
//! keyed by file and reopening is a header parse, not a re-read of payload.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use moxie_format::safetensors::{Dtype, TensorEntry};
use moxie_storage::{ByteBudget, HeaderBudget, Shard};
use moxie_types::{Error, Result};

pub(crate) fn invalid(detail: String) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

/// What one source tensor is: where it lives and what its header declares.
#[derive(Debug, Clone)]
pub struct SourceTensor {
    pub file: String,
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub len: u64,
}

/// The shards a selection names, opened one at a time.
#[derive(Debug)]
pub struct Sources {
    root: PathBuf,
    header_budget: HeaderBudget,
    read_budget: ByteBudget,
    open: Option<(String, Shard)>,
    /// Bytes read through this cache, for honest reporting.
    bytes_read: u64,
    headers_parsed: u64,
}

impl Sources {
    pub fn new(root: &Path, header_budget: HeaderBudget, read_budget: ByteBudget) -> Result<Self> {
        let root = root.canonicalize().map_err(|e| {
            invalid(format!(
                "source root {} does not canonicalize: {e}",
                root.display()
            ))
        })?;
        if !root.is_dir() {
            return Err(invalid(format!(
                "source root {} is not a directory",
                root.display()
            )));
        }
        Ok(Self {
            root,
            header_budget,
            read_budget,
            open: None,
            bytes_read: 0,
            headers_parsed: 0,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub fn headers_parsed(&self) -> u64 {
        self.headers_parsed
    }

    /// Resolve one file name inside the root.
    ///
    /// The selection parser already refused traversal on the string; this is
    /// the filesystem half, which is the only one that can catch a symlink
    /// pointing out of the root. Both are needed and neither is sufficient.
    pub fn path_of(&self, file: &str) -> Result<PathBuf> {
        let joined = self.root.join(file);
        let canonical = joined.canonicalize().map_err(|e| {
            invalid(format!(
                "source file {} does not canonicalize: {e}",
                joined.display()
            ))
        })?;
        if !canonical.starts_with(&self.root) {
            return Err(invalid(format!(
                "source file '{file}' resolves outside {}: a symlink pointing outside, which no \
                 string check can catch",
                self.root.display()
            )));
        }
        if !canonical.is_file() {
            return Err(invalid(format!(
                "source file '{file}' is not a regular file"
            )));
        }
        Ok(canonical)
    }

    fn shard(&mut self, file: &str) -> Result<&Shard> {
        if self.open.as_ref().map(|(f, _)| f.as_str()) != Some(file) {
            let path = self.path_of(file)?;
            let shard = Shard::open_with_limits(&path, self.read_budget, self.header_budget)?;
            self.headers_parsed += 1;
            self.open = Some((file.to_string(), shard));
        }
        Ok(&self.open.as_ref().expect("just opened").1)
    }

    /// Whether a shard declares a tensor at all, without reading it.
    ///
    /// The cross-shard resolver needs this to see a companion the selection did
    /// **not** declare: a symmetric selection over a module that carries zero
    /// points is a disagreement, and the only way to notice is to look.
    pub fn declares(&mut self, file: &str, name: &str) -> Result<bool> {
        let shard = self.shard(file)?;
        Ok(shard.header().tensors().contains_key(name))
    }

    /// One tensor's full header entry, for the shared validation the
    /// single-header resolver applies.
    pub fn raw_entry(&mut self, file: &str, name: &str) -> Result<TensorEntry> {
        let shard = self.shard(file)?;
        Ok(shard.header().get(name)?.clone())
    }

    /// One tensor's header entry, with the file it came from.
    pub fn entry(&mut self, file: &str, name: &str) -> Result<SourceTensor> {
        let shard = self.shard(file)?;
        let entry: &TensorEntry = shard.header().get(name)?;
        Ok(SourceTensor {
            file: file.to_string(),
            name: name.to_string(),
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            len: entry.len(),
        })
    }

    /// Read one bounded range of one tensor into a caller's buffer.
    pub fn read_range(
        &mut self,
        file: &str,
        name: &str,
        offset: u64,
        into: &mut [u8],
    ) -> Result<()> {
        let shard = self.shard(file)?;
        shard.read_tensor_range(name, offset, into)?;
        self.bytes_read += into.len() as u64;
        Ok(())
    }

    /// Read one whole (small) tensor, refusing anything above `cap`.
    ///
    /// For `weight_shape`, which is sixteen bytes. The cap is here so that
    /// "read the whole thing" can never be reached by a tensor that is not
    /// tiny, whatever a corrupt header declares.
    pub fn read_small(&mut self, file: &str, name: &str, cap: u64) -> Result<Vec<u8>> {
        let entry = self.entry(file, name)?;
        if entry.len > cap {
            return Err(invalid(format!(
                "{name} in {file} is {} byte(s); this read is capped at {cap}",
                entry.len
            )));
        }
        let shard = self.shard(file)?;
        let bytes = shard.tensor_bytes(name)?;
        self.bytes_read += bytes.len() as u64;
        Ok(bytes)
    }

    /// Re-hash every file and refuse if any digest has moved.
    ///
    /// A digest taken at the start of a run is a statement about the file **at
    /// that moment**. An independent review changed a source immediately after
    /// its digest was computed and the run published the new bytes under the
    /// old digest. Hashing through a retained handle does not fix that -- an
    /// in-place write reaches every handle -- so the only honest check is to
    /// hash again after the conversion and before anything is exposed, and to
    /// say what that costs: a second full pass over every source file.
    pub fn verify_unchanged(
        &mut self,
        recorded: &[(String, String)],
        scratch: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<()> {
        for (file, digest) in recorded {
            let (now, _) = self.file_digest_cancellable(file, scratch, cancelled)?;
            if &now != digest {
                return Err(invalid(format!(
                    "source file '{file}' hashed {digest} when this run started and {now} now: \
                     it changed underneath the conversion, and publishing would record a digest \
                     that describes bytes nobody has"
                )));
            }
        }
        Ok(())
    }

    /// SHA-256 of a whole source file, streamed through `scratch`.
    ///
    /// The manifest's `source.files.sha256` has exactly one meaning, so this
    /// computes exactly that. A repack that recorded a digest of the ranges it
    /// happened to read, in a field that says "this file", would be publishing
    /// a false checksum -- and a later whole-artifact claim would inherit it.
    pub fn file_digest(&mut self, file: &str, scratch: &mut [u8]) -> Result<(String, u64)> {
        self.file_digest_cancellable(file, scratch, &|| false)
    }

    /// The same, checked against cancellation between slices.
    ///
    /// A whole-file hash of a 5 GB shard is minutes of reading, and a
    /// cancellation that is only observed after it is a cancellation nobody
    /// experiences. Bounded buffers bound memory, not latency.
    pub fn file_digest_cancellable(
        &mut self,
        file: &str,
        scratch: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(String, u64)> {
        use moxie_format::StreamingSha256;
        let path = self.path_of(file)?;
        // The shard cache holds a handle to this file; opening it again for a
        // sequential read is deliberate, so hashing never moves a shared
        // cursor.
        let mut handle = std::fs::File::open(&path)
            .map_err(|e| invalid(format!("cannot open {}: {e}", path.display())))?;
        let mut hasher = StreamingSha256::new();
        let mut total = 0u64;
        loop {
            use std::io::Read;
            let n = handle
                .read(scratch)
                .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
            if n == 0 {
                break;
            }
            if cancelled() {
                return Err(invalid(format!(
                    "cancelled while hashing {file} after {total} byte(s)"
                )));
            }
            hasher.update(&scratch[..n]);
            total += n as u64;
        }
        self.bytes_read += total;
        Ok((hasher.finalize_hex(), total))
    }
}

/// The file each of a module's tensors lives in, by suffix.
pub type ModuleFiles = BTreeMap<String, String>;
