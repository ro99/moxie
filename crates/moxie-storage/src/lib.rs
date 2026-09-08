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
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use moxie_format::StreamingSha256;
use moxie_format::bf16::is_finite_bf16_bits;
use moxie_format::manifest::{self, Manifest, TensorPrecision};
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
            let len = resolved
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

    /// Stattached length of one chunk file, by chunk name.
    pub fn chunk_len(&self, chunk: &str) -> Option<u64> {
        self.chunks.get(chunk).map(|c| c.len)
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
        let mut source = FileRange::open(&chunk.path)?;
        let mut peak = 0usize;
        pump_range(
            &mut source,
            t.offset,
            dest,
            &t.sha256,
            t.precision,
            self.budget,
            &mut peak,
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

/// A positioned byte source: one slice read at an absolute offset.
///
/// The pump below only ever hands it sub-slices of the caller's destination
/// buffer, so the peak slice length it requests -- recorded in `peak` -- is
/// the bound on reader-driven I/O sizing. Tests drive the same pump with an
/// in-memory source and assert the peak never exceeds the budget.
trait RangeSource {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()>;
}

struct FileRange(File);

impl FileRange {
    fn open(path: &Path) -> Result<Self> {
        File::open(path)
            .map(FileRange)
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot open chunk '{}': {e}", path.display()),
            })
    }
}

impl RangeSource for FileRange {
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.0
            .seek(SeekFrom::Start(offset))
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot seek to {offset}: {e}"),
            })?;
        self.0.read_exact(buf).map_err(|e| Error::InvalidArtifact {
            detail: format!(
                "short read at {offset} for {} bytes: truncation: {e}",
                buf.len()
            ),
        })?;
        Ok(())
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
/// plus eight words, and the BF16 validator holds one carry byte. `peak`
/// records the largest slice requested, so tests count rather than trust.
fn pump_range<S: RangeSource>(
    source: &mut S,
    offset: u64,
    dest: &mut [u8],
    want_sha256: &str,
    precision: TensorPrecision,
    budget: ByteBudget,
    peak: &mut usize,
) -> Result<()> {
    let mut hasher = StreamingSha256::new();
    let mut bf16 = Bf16StreamValidator::new();
    let slice = budget.bytes().max(1);
    let total = dest.len();
    let mut done = 0usize;
    while done < total {
        let end = (done + slice).min(total);
        let buf = &mut dest[done..end];
        *peak = (*peak).max(buf.len());
        let pos = offset
            .checked_add(done as u64)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("tensor offset {offset} + {done} overflows"),
            })?;
        source.read_at(pos, buf)?;
        hasher.update(buf);
        if matches!(precision, TensorPrecision::Bf16V1) {
            bf16.feed(buf)?;
        }
        done = end;
    }
    if matches!(precision, TensorPrecision::Bf16V1) {
        bf16.finish()?;
    }
    let got = hasher.finalize_hex();
    if got != want_sha256.to_ascii_lowercase() {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "checksum mismatch: expected {want_sha256}, computed {got}; the destination buffer's contents are explicitly not to be trusted"
            ),
        });
    }
    Ok(())
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
        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
            let start = offset as usize;
            let end = start + buf.len();
            if end > self.data.len() {
                return Err(Error::InvalidArtifact {
                    detail: "short read: truncation".into(),
                });
            }
            buf.copy_from_slice(&self.data[start..end]);
            Ok(())
        }
    }

    /// Peak reader-driven slice length never exceeds the budget, for a tensor
    /// many times the budget's size -- counted, not trusted. Uses the same
    /// pump `Artifact::read_tensor` runs.
    #[test]
    fn peak_slice_length_is_capped_by_the_budget() {
        // 8 KiB tensor, budgets from 1 byte to 8 KiB: every slice the pump
        // requests is at most the budget, including odd budgets that split
        // BF16 elements mid-pair.
        for budget in [1usize, 3, 7, 1023, 1024, 4096, 8192] {
            let budget = ByteBudget::new(budget).unwrap();
            let data: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
            // Force finite BF16 so the read succeeds and the peak is what is
            // measured, not an early validation failure.
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
            let mut peak = 0usize;
            pump_range(
                &mut src,
                0,
                &mut dest,
                &sha,
                TensorPrecision::Bf16V1,
                budget,
                &mut peak,
            )
            .expect("valid payload reads");
            assert_eq!(dest, finite);
            assert!(
                peak <= budget.bytes(),
                "budget {}: peak slice {peak}",
                budget.bytes()
            );
            assert!(peak >= 1);
        }
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
            let mut peak = 0usize;
            let e = pump_range(
                &mut src,
                0,
                &mut dest,
                &sha,
                TensorPrecision::Bf16V1,
                budget,
                &mut peak,
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
}
