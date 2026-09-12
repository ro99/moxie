//! Bounded artifact reads: the only crate here that touches the filesystem.
//!
//! Document 02: `moxie-format` owns the manifest schema and its validation;
//! this crate owns opening an artifact directory, bounded chunk reads and
//! checksum verification while reading. It must not interpret architecture
//! metadata, decode weights, or choose what to keep resident -- and it must
//! never name a model family. Those are model-adapter and `moxie-memory`
//! jobs; `arch-check` enforces the boundary.
//!
//! ## Bounded opening and bounded reads
//!
//! `open` reads and validates `manifest.toml` and stats the chunk files. It
//! does not read a payload byte, so opening a 400 GB artifact costs the
//! manifest. `read_tensor` writes into a caller-supplied buffer in fixed-size
//! slices capped by [`ByteBudget`], hashes while reading, and validates every
//! BF16 element on the way through.
//!
//! If the checksum mismatches, the destination buffer's contents are explicitly
//! **not to be trusted**: "it returned an error but also wrote something" is
//! how a checksum gets skipped in practice. The error says which tensor.
//!
//! There is a time-of-check/time-of-use race between the canonical-path check
//! and the read that this does not close. It is recorded rather than papered
//! over: the artifact directory is assumed not to be mutated by another writer
//! during an open, which is the same assumption document 03's atomic-publish
//! workflow already makes.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};

use moxie_format::StreamingSha256;
use moxie_format::bf16::is_finite_bf16_bits;
use moxie_format::manifest::{self, Manifest, TensorPrecision};
use moxie_format::safetensors::Header as SafeHeader;
use moxie_types::{Error, Result};

/// Cap on the reader's own scratch allocation, in bytes.
///
/// Chunk payloads are read in fixed-size slices no larger than this, directly
/// into the caller's buffer, so a tensor many times the budget's size never
/// makes the reader hold more than the budget. The budget covers reader-owned
/// scratch only -- not the caller's destination buffer, which the caller sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteBudget {
    bytes: usize,
}

impl ByteBudget {
    /// The default: 64 KiB of reader scratch. Large enough that slice overhead
    /// is negligible, small enough to be obviously bounded.
    pub const DEFAULT: Self = Self { bytes: 64 * 1024 };

    pub const fn new(bytes: usize) -> Option<Self> {
        if bytes == 0 || bytes > 256 * 1024 * 1024 {
            return None;
        }
        Some(Self { bytes })
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

impl Default for ByteBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// An opened artifact: validated manifest plus statted, confined chunk files.
#[derive(Debug)]
pub struct Artifact {
    manifest: Manifest,
    dir: PathBuf,
    chunks: BTreeMap<String, ChunkFile>,
    budget: ByteBudget,
}

#[derive(Debug)]
struct ChunkFile {
    path: PathBuf,
    /// Open once at `Artifact::open` and held for the artifact's lifetime.
    /// Reads use position-independent reads on this handle, so no pathname
    /// is opened -- and no pathname-length allocation made -- during a read.
    file: File,
    len: u64,
}

impl Artifact {
    /// Open with the default read budget. Manifest only; no payload byte read.
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_budget(dir, ByteBudget::DEFAULT)
    }

    /// Open with an explicit budget. Tests use a small budget to prove a large
    /// tensor reads correctly without the reader holding more than the budget.
    pub fn open_with_budget(dir: &Path, budget: ByteBudget) -> Result<Self> {
        let manifest_text = read_manifest_capped(dir)?;
        let manifest = manifest::parse(&manifest_text)?;
        Self::from_manifest(dir, manifest, budget)
    }

    fn from_manifest(dir: &Path, manifest: Manifest, budget: ByteBudget) -> Result<Self> {
        let canonical_dir = dir.canonicalize().map_err(|e| Error::InvalidArtifact {
            detail: format!(
                "artifact directory {} does not canonicalize: {e}",
                dir.display()
            ),
        })?;
        let mut lengths = BTreeMap::new();
        let mut chunks = BTreeMap::new();
        // One entry per chunk name referenced, so a manifest naming the same
        // chunk twice stats it once.
        let mut names: Vec<&String> = manifest.tensors.iter().map(|t| &t.chunk).collect();
        names.sort();
        names.dedup();
        for name in names {
            let resolved = resolve_chunk(&canonical_dir, dir, name)?;
            let file = File::open(&resolved).map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot open chunk '{}': {e}", resolved.display()),
            })?;
            let len = file
                .metadata()
                .map_err(|e| Error::InvalidArtifact {
                    detail: format!("cannot stat chunk '{}': {e}", resolved.display()),
                })?
                .len();
            lengths.insert(name.clone(), len);
            chunks.insert(
                name.clone(),
                ChunkFile {
                    path: resolved,
                    file,
                    len,
                },
            );
        }
        manifest::validate_chunks(&manifest, &lengths)?;
        Ok(Self {
            manifest,
            dir: canonical_dir,
            chunks,
            budget,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The canonicalized artifact directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Statted length of one chunk file, by chunk name.
    pub fn chunk_len(&self, chunk: &str) -> Option<u64> {
        self.chunks.get(chunk).map(|c| c.len)
    }

    /// Canonical path of one chunk file, by chunk name.
    pub fn chunk_path(&self, chunk: &str) -> Option<&Path> {
        self.chunks.get(chunk).map(|c| c.path.as_path())
    }

    pub fn read_budget(&self) -> ByteBudget {
        self.budget
    }

    /// Stable artifact identity, from the validated manifest.
    pub fn identity(&self) -> String {
        manifest::artifact_identity(&self.manifest)
    }

    /// Read one tensor into a caller-supplied buffer.
    ///
    /// Returns the tensor's byte length. If the buffer is smaller than the
    /// tensor, this is an error naming both sizes -- never a partial read
    /// reported as success. On any error, including a checksum mismatch, the
    /// destination buffer's contents are explicitly not to be trusted.
    pub fn read_tensor(&self, role: &str, into: &mut [u8]) -> Result<usize> {
        let t = self
            .manifest
            .tensors
            .iter()
            .find(|t| t.role == role)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("no tensor named '{role}'"),
            })?;
        if matches!(
            self.manifest.completeness,
            moxie_format::manifest::Completeness::Partial { .. }
        ) {
            return Err(Error::InvalidArtifact {
                detail: "artifact is partial: it opens for inspection and refuses every read; partial output is not a loadable model".into(),
            });
        }
        if t.precision.is_affine() {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "tensor '{role}' is {}: manifest v1 describes affine tensors and refuses to read them; decoding arrives in M3",
                    t.precision.name()
                ),
            });
        }
        let need: usize = t.length.try_into().map_err(|_| Error::InvalidArtifact {
            detail: format!("tensor '{role}' length {} does not fit in usize", t.length),
        })?;
        if into.len() < need {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "tensor '{role}' needs {need} bytes but the buffer holds {}: never a partial read reported as success",
                    into.len()
                ),
            });
        }
        let chunk = self
            .chunks
            .get(&t.chunk)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("chunk '{}' was validated but is not open", t.chunk),
            })?;
        let dest = &mut into[..need];
        let mut source = OpenChunk { file: &chunk.file };
        pump_range(
            &mut source,
            t.offset,
            dest,
            &t.sha256,
            t.precision,
            self.budget,
            &t.chunk,
        )
        .map_err(|e| match e {
            Error::InvalidArtifact { detail } => Error::InvalidArtifact {
                detail: format!("tensor '{role}': {detail}"),
            },
            other => other,
        })?;
        Ok(need)
    }
}

/// One safetensors shard, opened for bounded reads.
///
/// A *source* container, not a canonical artifact: there is no manifest, no
/// role mapping and no checksum in a safetensors file, so this deliberately
/// shares none of [`Artifact`]'s machinery. What it shares is the rule that
/// this crate is the only one that touches the filesystem --
/// [`moxie_format::safetensors`] validates the header and never opens a file.
///
/// Opening reads the header and stats the file. **No payload byte is read**, so
/// opening a 5 GB shard costs its header. Nothing is memory-mapped: a file that
/// changed under a mapping would make a validated header a lie.
#[derive(Debug)]
pub struct Shard {
    path: PathBuf,
    file: File,
    header: SafeHeader,
    len: u64,
    budget: ByteBudget,
}

impl Shard {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_budget(path, ByteBudget::DEFAULT)
    }

    pub fn open_with_budget(path: &Path, budget: ByteBudget) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot open {}: {e}", path.display()),
        })?;
        let len = file
            .metadata()
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot stat {}: {e}", path.display()),
            })?
            .len();
        // Read the eight-byte length, learn the bound, then read exactly that
        // much -- rather than reading a length the file itself chose.
        let mut prefix = [0u8; 8];
        let mut source = OpenChunk { file: &file };
        source
            .read_at(0, &mut prefix)
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot read the length prefix of {}: {e}", path.display()),
            })?;
        let prefix_len = SafeHeader::prefix_len(&prefix)?;
        let mut bytes = crate::try_vec::<u8>(usize::try_from(prefix_len).map_err(|_| {
            Error::InvalidArtifact {
                detail: "header length does not fit this platform".into(),
            }
        })?)?;
        bytes.resize(prefix_len as usize, 0);
        source
            .read_at(0, &mut bytes)
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot read the header of {}: {e}", path.display()),
            })?;
        let header = SafeHeader::parse(&bytes, len)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            header,
            len,
            budget,
        })
    }

    pub fn header(&self) -> &SafeHeader {
        &self.header
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read one named tensor's payload into a caller-sized buffer.
    ///
    /// The length comes from the validated header, and a buffer that is not
    /// exactly that size is refused: a partial read must never be reported as
    /// success. Reads are issued in slices capped by the budget, so the reader
    /// itself holds nothing beyond it.
    pub fn read_tensor(&self, name: &str, into: &mut [u8]) -> Result<()> {
        let entry = self.header.get(name)?;
        let need = usize::try_from(entry.len()).map_err(|_| Error::InvalidArtifact {
            detail: format!("tensor {name:?} does not fit this platform"),
        })?;
        if into.len() != need {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "tensor {name:?} is {need} byte(s) but the buffer holds {}",
                    into.len()
                ),
            });
        }
        let offset = entry.file_offset(&self.header);
        let mut source = OpenChunk { file: &self.file };
        let slice = self.budget.bytes().max(1);
        let mut done = 0usize;
        while done < need {
            let end = (done + slice).min(need);
            let at = offset
                .checked_add(done as u64)
                .ok_or_else(|| Error::InvalidArtifact {
                    detail: format!("tensor {name:?} offset overflows"),
                })?;
            source
                .read_at(at, &mut into[done..end])
                .map_err(|e| Error::InvalidArtifact {
                    detail: format!("reading {name:?} from {}: {e}", self.path.display()),
                })?;
            done = end;
        }
        Ok(())
    }

    /// Read a tensor into a freshly allocated, fallibly sized buffer.
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self.header.get(name)?;
        let need = usize::try_from(entry.len()).map_err(|_| Error::InvalidArtifact {
            detail: format!("tensor {name:?} does not fit this platform"),
        })?;
        let mut out = crate::try_vec::<u8>(need)?;
        out.resize(need, 0);
        self.read_tensor(name, &mut out)?;
        Ok(out)
    }
}

fn try_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
            requested_bytes: capacity.saturating_mul(size_of::<T>()) as u64,
            available_bytes: 0,
        })?;
    Ok(out)
}

/// A positioned byte source: one slice read at an absolute offset.
///
/// The pump below only ever hands it sub-slices of the caller's destination
/// buffer, so reader-driven I/O sizing stays within the budget by
/// construction. The memory gate itself is measured, not asserted from slice
/// sizes: see the allocation-counting test over `Artifact::read_tensor`.
trait RangeSource {
    /// Fill `buf` from `offset`, exactly, or fail with an I/O error.
    /// Implementations retry `Interrupted` internally (as the legacy
    /// `CheckpointShardSet::read` does) and report premature EOF as
    /// `UnexpectedEof`; the pump maps failures to truncation errors naming
    /// the chunk.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()>;
}

struct OpenChunk<'a> {
    file: &'a File,
}

impl RangeSource for OpenChunk<'_> {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            // `pread` via `read_exact_at`: no file offset is touched, so a
            // shared handle serves every read with no seek, no clone and no
            // pathname open -- and short reads, including `Interrupted`,
            // are retried inside `read_exact_at` rather than in another
            // hand-rolled exact-read loop here.
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset)
        }
        #[cfg(not(unix))]
        {
            // Best effort off the product platform (Linux-only): a duplicated
            // handle keeps the shared offset untouched. `read_exact` retries
            // `Interrupted` internally.
            let mut owned = self.file.try_clone()?;
            owned.seek(SeekFrom::Start(offset))?;
            owned.read_exact(buf)
        }
    }
}

/// Read `manifest.toml` through a capped reader that errors at the limit
/// rather than reading the file and then measuring it.
fn read_manifest_capped(dir: &Path) -> Result<String> {
    let path = dir.join("manifest.toml");
    let mut f = File::open(&path).map_err(|e| Error::InvalidArtifact {
        detail: format!("cannot open {}: {e}", path.display()),
    })?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let n = f.read(&mut chunk).map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot read {}: {e}", path.display()),
        })?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > moxie_format::manifest::MAX_MANIFEST_BYTES {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "manifest.toml exceeds the {} byte cap before parsing: it bounds the parse itself",
                    moxie_format::manifest::MAX_MANIFEST_BYTES
                ),
            });
        }
    }
    String::from_utf8(buf).map_err(|e| Error::InvalidArtifact {
        detail: format!("manifest.toml is not UTF-8: {e}"),
    })
}

/// Confine a chunk reference to the artifact directory.
///
/// The string check runs first, before any path is joined: it stops traversal
/// without touching the filesystem. The canonical check runs after opening:
/// it stops symlinks pointing outside, which no string check can catch. Either
/// alone is insufficient.
fn resolve_chunk(canonical_dir: &Path, dir: &Path, name: &str) -> Result<PathBuf> {
    manifest::validate_chunk_name(name).map_err(|d| Error::InvalidArtifact {
        detail: format!("chunk reference rejected on the string, before joining: {d}"),
    })?;
    let joined = dir.join(name);
    let canonical = joined.canonicalize().map_err(|e| Error::InvalidArtifact {
        detail: format!("chunk '{name}' does not canonicalize: {e}"),
    })?;
    if !canonical.starts_with(canonical_dir) {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "chunk '{name}' resolves outside the artifact directory: a symlink pointing outside, which no string check can catch"
            ),
        });
    }
    // A regular file: not a directory, not a device, not a dangling link that
    // somehow canonicalized.
    let meta = std::fs::metadata(&canonical).map_err(|e| Error::InvalidArtifact {
        detail: format!("cannot metadata chunk '{name}': {e}"),
    })?;
    if !meta.is_file() {
        return Err(Error::InvalidArtifact {
            detail: format!("chunk '{name}' is not a regular file"),
        });
    }
    Ok(canonical)
}

/// Read one byte range in budget-capped slices, hashing while reading and
/// validating BF16 finiteness on the way through.
///
/// No reader-owned buffer ever holds more than `budget` bytes: every slice is
/// a sub-slice of the caller's `dest`, the hasher holds one 64-byte block
/// plus eight words, the BF16 validator holds one carry byte, and checksum
/// verification finalizes to a stack array compared against a stack-decoded
/// expectation. The success path allocates no heap at all: slices are
/// sub-slices of the caller's buffer, hashing and hex run on stack arrays,
/// and the chunk handle was opened once at `Artifact::open`, so pathname
/// length cannot allocate during a read. The allocation gate tests measure
/// this, on short and long paths alike. Interrupted syscalls retry without
/// advancing; every other I/O failure is truncation naming the chunk.
fn pump_range<S: RangeSource>(
    source: &mut S,
    offset: u64,
    dest: &mut [u8],
    want_sha256: &str,
    precision: TensorPrecision,
    budget: ByteBudget,
    chunk: &str,
) -> Result<()> {
    let mut hasher = StreamingSha256::new();
    let mut bf16 = Bf16StreamValidator::new();
    let slice = budget.bytes().max(1);
    let total = dest.len();
    let mut done = 0usize;
    while done < total {
        let end = (done + slice).min(total);
        let buf = &mut dest[done..end];
        let pos = offset
            .checked_add(done as u64)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("tensor offset {offset} + {done} overflows"),
            })?;
        // An interrupted syscall is retried without advancing: the tensor
        // offset has not moved, so no byte is skipped or duplicated.
        // Anything else -- including premature EOF -- is truncation naming
        // the chunk, never a silent short read.
        match source.read_at(pos, buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "short read of chunk '{chunk}' at {pos} for {} bytes: truncation: {e}",
                        buf.len()
                    ),
                });
            }
        }
        hasher.update(buf);
        if matches!(precision, TensorPrecision::Bf16V1) {
            bf16.feed(buf)?;
        }
        done = end;
    }
    if matches!(precision, TensorPrecision::Bf16V1) {
        bf16.finish()?;
    }
    let got = hasher.finalize_bytes();
    let want = decode_hex32(want_sha256).map_err(|()| Error::InvalidArtifact {
        detail: "stored checksum is not 64 hex digits".into(),
    })?;
    if got != want {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "checksum mismatch: expected {want_sha256}, computed {}; the destination buffer's contents are explicitly not to be trusted",
                str::from_utf8(&hex_of(&got)).unwrap_or("?")
            ),
        });
    }
    Ok(())
}

/// Decode 64 hex digits to 32 bytes on the stack. Case-insensitive; the
/// manifest validator already pins lowercase, this stays liberal.
fn decode_hex32(s: &str) -> std::result::Result<[u8; 32], ()> {
    fn val(b: u8) -> std::result::Result<u8, ()> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err(()),
        }
    }
    let b = s.as_bytes();
    if b.len() != 64 {
        return Err(());
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = val(b[2 * i])? << 4 | val(b[2 * i + 1])?;
    }
    Ok(out)
}

/// 32 digest bytes to 64 lowercase hex bytes, on the stack.
fn hex_of(bytes: &[u8; 32]) -> [u8; 64] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 64];
    for (i, b) in bytes.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 15) as usize];
    }
    out
}

/// Incremental finite-BF16 validation across arbitrarily split slices.
///
/// A budget slice may start mid-element, so one carry byte pairs the tail of
/// the previous slice with the head of the next. Every 16-bit element is
/// checked exactly once, whatever the slice boundaries.
struct Bf16StreamValidator {
    carry: Option<u8>,
    elements: usize,
}

impl Bf16StreamValidator {
    fn new() -> Self {
        Self {
            carry: None,
            elements: 0,
        }
    }

    fn feed(&mut self, mut bytes: &[u8]) -> Result<()> {
        if let Some(lo) = self.carry.take() {
            let Some((&hi, rest)) = bytes.split_first() else {
                self.carry = Some(lo);
                return Ok(());
            };
            self.check(lo, hi)?;
            bytes = rest;
        }
        for (i, pair) in bytes.chunks_exact(2).enumerate() {
            let _ = i;
            self.check(pair[0], pair[1])?;
        }
        if !bytes.len().is_multiple_of(2) {
            self.carry = Some(bytes[bytes.len() - 1]);
        }
        Ok(())
    }

    fn check(&mut self, lo: u8, hi: u8) -> Result<()> {
        let bits = u16::from_le_bytes([lo, hi]);
        if !is_finite_bf16_bits(bits) {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "BF16 element {} is 0x{bits:04x}, a non-finite weight: caught at load, not at the first matmul",
                    self.elements
                ),
            });
        }
        self.elements += 1;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        if self.carry.is_some() {
            return Err(Error::InvalidArtifact {
                detail: "BF16 payload has odd length".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_format::sha256::sha256_hex;

    struct MemSource {
        data: Vec<u8>,
    }

    impl RangeSource for MemSource {
        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
            let start = offset as usize;
            let end = start + buf.len();
            if end > self.data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read: truncation",
                ));
            }
            buf.copy_from_slice(&self.data[start..end]);
            Ok(())
        }
    }

    /// Deterministic `EINTR`: fails the listed 1-based calls with
    /// `Interrupted`, then behaves like memory. Models the fault injector
    /// that made the first `pread64` return `EINTR`.
    struct FlakySource {
        data: Vec<u8>,
        interrupt_calls: Vec<u64>,
        calls: std::cell::Cell<u64>,
        fail_with: Option<std::io::ErrorKind>,
    }

    impl RangeSource for FlakySource {
        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            if self.interrupt_calls.contains(&call) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "injected EINTR",
                ));
            }
            if let Some(kind) = self.fail_with {
                return Err(std::io::Error::new(kind, "injected genuine error"));
            }
            let start = offset as usize;
            let end = start + buf.len();
            if end > self.data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read: truncation",
                ));
            }
            buf.copy_from_slice(&self.data[start..end]);
            Ok(())
        }
    }

    fn finite_bytes(seed: u64, n: usize) -> Vec<u8> {
        let mut b: Vec<u8> = (0..n)
            .map(|i| ((i * 37 + seed as usize) % 251) as u8)
            .collect();
        for pair in b.chunks_exact_mut(2) {
            let mut bits = u16::from_le_bytes([pair[0], pair[1]]);
            if (bits & 0x7F80) == 0x7F80 {
                bits &= 0x7F7F;
                pair.copy_from_slice(&bits.to_le_bytes());
            }
        }
        b
    }

    /// Odd budgets split BF16 elements mid-pair; the pump must still read
    /// exactly. (Slice sizes are capped by construction. The memory gate is
    /// the allocation-counting test over `Artifact::read_tensor`, not an
    /// I/O-request-size assertion.)
    #[test]
    fn odd_budgets_still_read_exactly() {
        for budget in [1usize, 3, 7, 1023, 1024, 4096, 8192] {
            let budget = ByteBudget::new(budget).unwrap();
            let data: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
            let mut finite = data.clone();
            for pair in finite.chunks_exact_mut(2) {
                let mut bits = u16::from_le_bytes([pair[0], pair[1]]);
                if (bits & 0x7F80) == 0x7F80 {
                    bits &= 0x7F7F;
                    pair.copy_from_slice(&bits.to_le_bytes());
                }
            }
            let sha = sha256_hex(&finite);
            let mut src = MemSource {
                data: finite.clone(),
            };
            let mut dest = vec![0u8; 8192];
            pump_range(
                &mut src,
                0,
                &mut dest,
                &sha,
                TensorPrecision::Bf16V1,
                budget,
                "c.bin",
            )
            .expect("valid payload reads");
            assert_eq!(dest, finite);
        }
    }

    #[test]
    fn hex_helpers_round_trip_on_the_stack() {
        let sha = sha256_hex(b"artifact");
        let bytes = decode_hex32(&sha).expect("our own hex decodes");
        let back: [u8; 64] = sha.as_bytes().try_into().unwrap();
        assert_eq!(hex_of(&bytes), back);
        assert!(decode_hex32(&sha.to_uppercase()).is_ok());
        assert!(decode_hex32("short").is_err());
        assert!(decode_hex32(&"zz".repeat(32)).is_err());
    }

    #[test]
    fn a_nonfinite_element_is_found_whatever_the_slice_boundary() {
        // NaN at element 100; odd budgets split it across slices in different
        // places. The carry logic must find it every time.
        for budget in [1usize, 2, 3, 5, 7, 64, 1023] {
            let budget = ByteBudget::new(budget).unwrap();
            let mut finite: Vec<u8> = (0..512).map(|i| (i % 251) as u8).collect();
            for pair in finite.chunks_exact_mut(2) {
                let mut bits = u16::from_le_bytes([pair[0], pair[1]]);
                if (bits & 0x7F80) == 0x7F80 {
                    bits &= 0x7F7F;
                    pair.copy_from_slice(&bits.to_le_bytes());
                }
            }
            finite[200..202].copy_from_slice(&0x7FC0u16.to_le_bytes());
            let sha = sha256_hex(&finite);
            let mut src = MemSource { data: finite };
            let mut dest = vec![0u8; 512];
            let e = pump_range(
                &mut src,
                0,
                &mut dest,
                &sha,
                TensorPrecision::Bf16V1,
                budget,
                "c.bin",
            )
            .unwrap_err();
            assert!(e.to_string().contains("non-finite"), "{e}");
        }
    }

    #[test]
    fn budget_boundaries_are_checked() {
        assert!(ByteBudget::new(0).is_none());
        assert!(ByteBudget::new(64 * 1024).is_some());
        assert_eq!(ByteBudget::default(), ByteBudget::DEFAULT);
    }

    /// An interrupted syscall retries without advancing: the tensor offset
    /// has not moved, so no byte is skipped or duplicated. Mirrors the
    /// fault injector that made the first `pread64` return `EINTR`.
    #[test]
    fn interruption_before_any_bytes_retries_to_success() {
        // Budget 8 over 32 bytes: four slices. The first two attempts fail
        // before a single byte is accepted.
        let budget = ByteBudget::new(8).unwrap();
        let finite = finite_bytes(41, 32);
        let sha = sha256_hex(&finite);
        let mut src = FlakySource {
            data: finite.clone(),
            interrupt_calls: vec![1, 2],
            calls: std::cell::Cell::new(0),
            fail_with: None,
        };
        let mut dest = vec![0u8; 32];
        pump_range(
            &mut src,
            0,
            &mut dest,
            &sha,
            TensorPrecision::Bf16V1,
            budget,
            "c.bin",
        )
        .expect("interruption retries");
        assert_eq!(dest, finite);
        assert_eq!(src.calls.get(), 6, "four slices plus two retries");
    }

    #[test]
    fn interruption_after_partial_progress_retries_to_success() {
        // Two slices (16 bytes) land, the third attempt is interrupted, the
        // retry re-reads the same slice: five slice reads plus one retry.
        let budget = ByteBudget::new(8).unwrap();
        let finite = finite_bytes(42, 48);
        let sha = sha256_hex(&finite);
        let mut src = FlakySource {
            data: finite.clone(),
            interrupt_calls: vec![3],
            calls: std::cell::Cell::new(0),
            fail_with: None,
        };
        let mut dest = vec![0u8; 48];
        pump_range(
            &mut src,
            0,
            &mut dest,
            &sha,
            TensorPrecision::Bf16V1,
            budget,
            "c.bin",
        )
        .expect("interruption retries");
        assert_eq!(dest, finite);
        assert_eq!(src.calls.get(), 7, "six slices plus one retry");
    }

    #[test]
    fn genuine_errors_are_retained_not_retried() {
        // A non-interruption failure is truncation naming the chunk, even
        // after partial progress -- never a silent short read.
        let budget = ByteBudget::new(8).unwrap();
        let finite = finite_bytes(43, 32);
        let sha = sha256_hex(&finite);
        let mut src = FlakySource {
            data: finite,
            interrupt_calls: vec![],
            calls: std::cell::Cell::new(0),
            fail_with: Some(std::io::ErrorKind::ConnectionReset),
        };
        let mut dest = vec![0u8; 32];
        let e = pump_range(
            &mut src,
            0,
            &mut dest,
            &sha,
            TensorPrecision::Bf16V1,
            budget,
            "chunk7.bin",
        )
        .unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("truncation"), "{msg}");
        assert!(msg.contains("chunk7.bin"), "{msg}");
    }

    #[test]
    fn premature_eof_is_truncation() {
        // The source ends mid-tensor: the pump reports truncation, not a
        // short buffer claimed as success.
        let budget = ByteBudget::new(1024).unwrap();
        let finite = finite_bytes(44, 16);
        let sha = sha256_hex(&finite);
        let mut src = MemSource { data: finite };
        let mut dest = vec![0u8; 32];
        let e = pump_range(
            &mut src,
            0,
            &mut dest,
            &sha,
            TensorPrecision::Bf16V1,
            budget,
            "c.bin",
        )
        .unwrap_err();
        assert!(e.to_string().contains("truncation"), "{e}");
    }
}
