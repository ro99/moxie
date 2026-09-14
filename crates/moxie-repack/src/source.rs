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
    /// The kernel identity of every source file this run has opened.
    ///
    /// One shard is cached at a time, so a file can be closed and reopened
    /// while a run is in progress. If what the name resolves to has changed in
    /// between, everything measured before it refers to a file that is no
    /// longer there. Independent review replaced a source between an inspection
    /// and the repack that followed, and the artifact carried the old file's
    /// values under the new file's digest. This makes the second open a
    /// refusal.
    identities: BTreeMap<String, (u64, u64)>,
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
            identities: BTreeMap::new(),
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
            // The identity of the file actually opened, from its handle. A run
            // reads one source under one identity or it stops: continuing over
            // a replacement would publish bytes from one file described by
            // another file's digest.
            let key = shard.file_key()?;
            match self.identities.get(file) {
                Some(seen) if *seen != key => {
                    return Err(invalid(format!(
                        "source file '{file}' has been replaced since this run first opened it: \
                         it was {seen:?} and is now {key:?}. A repack reads one set of bytes, and \
                         continuing would describe the bytes it converted with a digest of bytes \
                         it never saw"
                    )));
                }
                Some(_) => {}
                None => {
                    self.identities.insert(file.to_string(), key);
                }
            }
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
    /// `Ok(false)` means the caller cancelled part-way through.
    pub fn verify_unchanged(
        &mut self,
        recorded: &[(String, String)],
        scratch: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<bool> {
        // **Does the name still point at the file we read?** Hashing through
        // the retained handle is what keeps the conversion and the digest
        // describing the same bytes -- but it also means an atomic replacement
        // is invisible to it, because the old inode is still there and still
        // readable. The path is what the manifest's `source.files` names, so
        // the path is what has to still resolve here.
        self.confirm_identities()?;
        for (file, digest) in recorded {
            let Some((now, _)) = self.file_digest_cancellable(file, scratch, cancelled)? else {
                return Ok(false);
            };
            if &now != digest {
                return Err(invalid(format!(
                    "source file '{file}' hashed {digest} when this run started and {now} now: \
                     it changed underneath the conversion, and publishing would record a digest \
                     that describes bytes nobody has"
                )));
            }
        }
        Ok(true)
    }

    /// Every source file this run has opened still resolves to the file it
    /// opened.
    ///
    /// Independent review replaced a shard between an inspection and the repack
    /// that followed it. Two different failures hide in that: reading one file
    /// while hashing another, which the retained handle now prevents, and
    /// publishing a digest for a path that no longer holds those bytes, which
    /// only a fresh look at the path can catch.
    pub fn confirm_identities(&mut self) -> Result<()> {
        let files: Vec<String> = self.identities.keys().cloned().collect();
        for file in files {
            let path = self.path_of(&file)?;
            let now = path_key(&path)?;
            let seen = self.identities[&file];
            if now != seen {
                return Err(invalid(format!(
                    "source file '{file}' has been replaced since this run first opened it: it \
                     was {seen:?} and is now {now:?}. The digest this run would publish describes \
                     bytes that are no longer at that path"
                )));
            }
        }
        Ok(())
    }

    /// SHA-256 of a whole source file, read through the handle this run is
    /// already using for that file.
    ///
    /// The manifest's `source.files.sha256` has exactly one meaning, so this
    /// computes exactly that. A repack that recorded a digest of the ranges it
    /// happened to read, in a field that says "this file", would be publishing
    /// a false checksum -- and a later whole-artifact claim would inherit it.
    pub fn file_digest(&mut self, file: &str, scratch: &mut [u8]) -> Result<(String, u64)> {
        self.file_digest_cancellable(file, scratch, &|| false)?
            .ok_or_else(|| invalid("hashing was cancelled by a caller that cannot cancel".into()))
    }

    /// The same, checked against cancellation between slices.
    ///
    /// A whole-file hash of a 5 GB shard is minutes of reading, and a
    /// cancellation that is only observed after it is a cancellation nobody
    /// experiences. Bounded buffers bound memory, not latency.
    ///
    /// `Ok(None)` is that cancellation. It used to be an `InvalidArtifact`
    /// naming the file, which independent review received in place of
    /// `Outcome::Cancelled`: a stop the caller asked for is not a corrupt
    /// source, and a run that reports one as the other teaches its user to
    /// distrust the difference.
    pub fn file_digest_cancellable(
        &mut self,
        file: &str,
        scratch: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<(String, u64)>> {
        // Through the shard cache, so the bytes hashed are the bytes read: the
        // same open file description answers both questions, and a file swapped
        // under this run is refused by `shard` rather than silently hashed.
        let shard = self.shard(file)?;
        let digested = shard.digest_whole_file(scratch, cancelled)?;
        if let Some((_, total)) = &digested {
            self.bytes_read += *total;
        }
        Ok(digested)
    }
}

/// A path's identity, as the kernel knows it: the same pair `Shard::file_key`
/// reports for an open handle.
fn path_key(path: &Path) -> Result<(u64, u64)> {
    let meta = std::fs::metadata(path)
        .map_err(|e| invalid(format!("cannot stat {}: {e}", path.display())))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok((meta.dev(), meta.ino()))
    }
    #[cfg(not(unix))]
    {
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Ok((meta.len(), mtime))
    }
}

/// The file each of a module's tensors lives in, by suffix.
pub type ModuleFiles = BTreeMap<String, String>;
