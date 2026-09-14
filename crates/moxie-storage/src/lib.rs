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
    payloads: Payloads,
    budget: ByteBudget,
}

/// Where an opened artifact's bytes are.
///
/// Version 1 keeps raw chunk files and the reader it always had; version 2 is
/// safetensors shards ([ADR 0025]). Nothing converts between them and nothing
/// reinterprets one as the other.
///
/// [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md
#[derive(Debug)]
enum Payloads {
    Chunks(BTreeMap<String, ChunkFile>),
    Shards(BTreeMap<String, Shard>),
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

    /// Open an artifact whose manifest is **not yet published**.
    ///
    /// The publisher's validation step, and the reason it exists is that there
    /// must not be a second validator. A repack writes its chunk files under
    /// their final names and its manifest under a private name; publication is
    /// the rename that gives the manifest its real one. Validating before that
    /// rename means opening a directory whose manifest is somewhere else, and
    /// the alternative -- publish, then check -- is checking after the artifact
    /// is already visible.
    ///
    /// The manifest path is confined to `dir`: this opens a candidate manifest
    /// for the directory it belongs to, never one from elsewhere.
    ///
    /// Everything else is the published path's behaviour, byte for byte: the
    /// same parser, the same chunk resolution, the same validation.
    pub fn open_unpublished(dir: &Path, manifest_path: &Path, budget: ByteBudget) -> Result<Self> {
        let canonical_dir = dir.canonicalize().map_err(|e| Error::InvalidArtifact {
            detail: format!(
                "artifact directory {} does not canonicalize: {e}",
                dir.display()
            )
            .into(),
        })?;
        let canonical_manifest =
            manifest_path
                .canonicalize()
                .map_err(|e| Error::InvalidArtifact {
                    detail: format!(
                        "candidate manifest {} does not canonicalize: {e}",
                        manifest_path.display()
                    )
                    .into(),
                })?;
        if canonical_manifest.parent() != Some(canonical_dir.as_path()) {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "candidate manifest {} is not directly inside {}: a manifest describes the \
                     directory it lives in",
                    canonical_manifest.display(),
                    canonical_dir.display()
                )
                .into(),
            });
        }
        let text = read_file_capped(&canonical_manifest)?;
        let manifest = manifest::parse(&text)?;
        Self::from_manifest(dir, manifest, budget)
    }

    fn from_manifest(dir: &Path, manifest: Manifest, budget: ByteBudget) -> Result<Self> {
        let canonical_dir = dir.canonicalize().map_err(|e| Error::InvalidArtifact {
            detail: format!(
                "artifact directory {} does not canonicalize: {e}",
                dir.display()
            )
            .into(),
        })?;
        // Which payload shape this manifest describes is a property of its
        // rows, not a guess: a tensor either names a chunk range or names its
        // components, and the schema validator has already refused a manifest
        // that mixes them.
        let is_v1 = manifest
            .tensors
            .first()
            .map(|t| t.chunk_range().is_some())
            .unwrap_or(false);
        let payloads = if is_v1 {
            let mut lengths = BTreeMap::new();
            let mut chunks = BTreeMap::new();
            // One entry per chunk name referenced, so a manifest naming the
            // same chunk twice stats it once.
            let mut names: Vec<&str> = manifest
                .tensors
                .iter()
                .filter_map(|t| t.chunk_range().map(|(chunk, _, _, _)| chunk))
                .collect();
            names.sort();
            names.dedup();
            for name in names {
                let resolved = resolve_chunk(&canonical_dir, dir, name)?;
                let file = File::open(&resolved).map_err(|e| Error::InvalidArtifact {
                    detail: format!("cannot open chunk '{}': {e}", resolved.display()).into(),
                })?;
                let len = file
                    .metadata()
                    .map_err(|e| Error::InvalidArtifact {
                        detail: format!("cannot stat chunk '{}': {e}", resolved.display()).into(),
                    })?
                    .len();
                lengths.insert(name.to_string(), len);
                chunks.insert(
                    name.to_string(),
                    ChunkFile {
                        path: resolved,
                        file,
                        len,
                    },
                );
            }
            manifest::validate_chunks(&manifest, &lengths)?;
            Payloads::Chunks(chunks)
        } else {
            let mut names: Vec<&str> = manifest
                .tensors
                .iter()
                .flat_map(|t| t.components().unwrap_or(&[]))
                .map(|c| c.file.as_str())
                .collect();
            names.sort();
            names.dedup();
            let mut shards = BTreeMap::new();
            for name in names {
                let resolved = resolve_chunk(&canonical_dir, dir, name)?;
                let shard = Shard::open_with_limits(&resolved, budget, HeaderBudget::DEFAULT)?;
                shards.insert(name.to_string(), shard);
            }
            // A canonical shard is covered exactly. Checked before any
            // component is looked up, because a shard carrying bytes no
            // tensor claims is refused by the reference implementation and
            // must be refused here (ADR 0025).
            for (name, shard) in &shards {
                shard
                    .header()
                    .require_exact_coverage()
                    .map_err(|e| Error::InvalidArtifact {
                        detail: format!("shard '{name}': {e}").into(),
                    })?;
            }
            // Every component must be the tensor the **descriptor** implies:
            // present, and of that dtype, that shape and that byte length.
            // Checked at open, so a renamed, re-typed or re-shaped component
            // is found before a read rather than during one -- and derived
            // from the descriptor rather than from the manifest's own
            // component row, which is the claim under test.
            for t in &manifest.tensors {
                let expected = t
                    .expected_components()
                    .map_err(|e| Error::InvalidArtifact {
                        detail: format!(
                            "tensor '{}': its components cannot be derived: {e}",
                            t.role
                        )
                        .into(),
                    })?;
                for (c, want) in t.components().unwrap_or(&[]).iter().zip(expected.iter()) {
                    let shard =
                        shards
                            .get(c.file.as_str())
                            .ok_or_else(|| Error::InvalidArtifact {
                                detail: format!(
                                    "tensor '{}' names shard '{}', which is not open",
                                    t.role, c.file
                                )
                                .into(),
                            })?;
                    let entry = shard.header().get(&c.name).map_err(|e| {
                        Error::InvalidArtifact {
                        detail: format!(
                            "tensor '{}' component '{}': shard '{}' declares no tensor '{}': {e}",
                            t.role,
                            c.kind.name(),
                            c.file,
                            c.name
                        )
                        .into(),
                    }
                    })?;
                    if entry.dtype != want.dtype
                        || entry.shape != want.shape
                        || entry.len() != want.len
                    {
                        return Err(Error::InvalidArtifact {
                                detail: format!(
                                    "tensor '{}' component '{}': shard '{}' declares it {}                                      shape-{:?} of {} byte(s); the descriptor implies {}                                      shape-{:?} of {} byte(s)",
                                    t.role,
                                    c.kind.name(),
                                    c.file,
                                    entry.dtype.name(),
                                    entry.shape,
                                    entry.len(),
                                    want.dtype.name(),
                                    want.shape,
                                    want.len
                                )
                                .into(),
                            });
                    }
                }
            }
            Payloads::Shards(shards)
        };
        Ok(Self {
            manifest,
            dir: canonical_dir,
            payloads,
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
        match &self.payloads {
            Payloads::Chunks(chunks) => chunks.get(chunk).map(|c| c.len),
            Payloads::Shards(shards) => shards.get(chunk).map(|s| s.len()),
        }
    }

    /// Canonical path of one chunk file, by chunk name.
    pub fn chunk_path(&self, chunk: &str) -> Option<&Path> {
        match &self.payloads {
            Payloads::Chunks(chunks) => chunks.get(chunk).map(|c| c.path.as_path()),
            Payloads::Shards(shards) => shards.get(chunk).map(|s| s.path()),
        }
    }

    /// One component's payload length, from the shard that declares it.
    pub fn shard_entry(&self, file: &str, name: &str) -> Option<u64> {
        match &self.payloads {
            Payloads::Shards(shards) => shards
                .get(file)
                .and_then(|s| s.header().get(name).ok())
                .map(|e| e.len()),
            Payloads::Chunks(_) => None,
        }
    }

    /// One shard's header length: bytes that belong to no component.
    pub fn shard_header_len(&self, file: &str) -> Option<u64> {
        match &self.payloads {
            Payloads::Shards(shards) => shards.get(file).map(|s| s.header().payload_start),
            Payloads::Chunks(_) => None,
        }
    }

    pub fn read_budget(&self) -> ByteBudget {
        self.budget
    }

    /// Stable artifact identity, from the validated manifest.
    pub fn identity(&self) -> String {
        manifest::artifact_identity(&self.manifest)
    }

    /// Stream one tensor's payload through a caller-supplied sink, hashing it.
    ///
    /// This is the **verification** primitive, and it is deliberately not
    /// [`Artifact::read_tensor`]:
    ///
    /// * It works for **affine** tensors. Reading one as a weight needs a decoder
    ///   and a kernel, which is M3 item 3; checking that the bytes a repack
    ///   published are the bytes it hashed needs neither.
    /// * It works on a **partial** artifact. A partial artifact is not a loadable
    ///   model and `read_tensor` refuses it for that reason; refusing to check the
    ///   checksums of a partial artifact would mean the one command that can tell a
    ///   user their selection is intact does not work on selections.
    /// * It never holds the tensor. The caller supplies `scratch`, which is capped
    ///   by the artifact's read budget, and sees the payload as a sequence of
    ///   slices in file order.
    ///
    /// The sink must treat what it is handed as **provisional**: the checksum is
    /// verified after the last slice, so bytes are trustworthy only once this
    /// returns `Ok`. That is the same rule `read_tensor` states about its
    /// destination buffer, moved to where a streaming consumer can see it.
    ///
    /// Returns the number of bytes streamed, which is the tensor's manifest length.
    pub fn stream_tensor(
        &self,
        role: &str,
        scratch: &mut [u8],
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<u64> {
        let t = self
            .manifest
            .tensors
            .iter()
            .find(|t| t.role == role)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("no tensor named '{role}'").into(),
            })?;
        if scratch.is_empty() {
            return Err(Error::InvalidArtifact {
                detail: "a zero-byte scratch buffer cannot stream anything".into(),
            });
        }
        match (&self.payloads, &t.placement) {
            (
                Payloads::Chunks(chunks),
                manifest::Placement::Chunk {
                    chunk,
                    offset,
                    length,
                    sha256,
                    ..
                },
            ) => {
                let file = chunks
                    .get(chunk.as_str())
                    .ok_or_else(|| Error::InvalidArtifact {
                        detail: format!("chunk '{chunk}' was validated but is not open").into(),
                    })?;
                let mut source = OpenChunk { file: &file.file };
                self.pump_into(
                    role,
                    &mut source,
                    *offset,
                    *length,
                    sha256,
                    t.precision,
                    scratch,
                    sink,
                )
            }
            (Payloads::Shards(shards), manifest::Placement::Components(components)) => {
                // The components stream in canonical order -- codes, then
                // scales, then zero points -- so what a consumer sees is
                // exactly ADR 0023's payload, whichever container it came out
                // of. Each component is verified against **its own** checksum,
                // which is the scope ADR 0025 records.
                let mut total = 0u64;
                for c in components {
                    let shard =
                        shards
                            .get(c.file.as_str())
                            .ok_or_else(|| Error::InvalidArtifact {
                                detail: format!("shard '{}' was validated but is not open", c.file)
                                    .into(),
                            })?;
                    let entry = shard.header().get(&c.name)?;
                    let begin = entry.file_offset(shard.header());
                    let len = entry.len();
                    let mut source = OpenChunk { file: &shard.file };
                    self.pump_into(
                        // The component's own name, which is what the manifest
                        // records and what a safetensors reader shows: the
                        // role for a BF16 weight, `role.codes` and friends for
                        // an affine one.
                        &c.name,
                        &mut source,
                        begin,
                        len,
                        &c.sha256,
                        // BF16 finiteness is checked for a BF16 component;
                        // packed codes, scales and zero points are not BF16
                        // numbers and must not be read as any.
                        match c.kind {
                            moxie_format::canonical::ComponentKind::Weights => t.precision,
                            _ => TensorPrecision::AffineInt8V1,
                        },
                        scratch,
                        sink,
                    )?;
                    total += len;
                }
                Ok(total)
            }
            _ => Err(Error::InvalidArtifact {
                detail: format!(
                    "tensor '{role}' is placed one way and this artifact is opened the other; \
                     nothing here converts between the two schema versions"
                )
                .into(),
            }),
        }
    }

    /// Read one byte range in scratch-sized slices, hashing it, validating
    /// BF16 finiteness when the bytes are BF16, and handing each slice on.
    #[allow(clippy::too_many_arguments)]
    fn pump_into<S: RangeSource>(
        &self,
        what: &str,
        source: &mut S,
        offset: u64,
        length: u64,
        want_sha256: &str,
        precision: TensorPrecision,
        scratch: &mut [u8],
        sink: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<u64> {
        let slice = scratch.len().min(self.budget.bytes()).max(1);
        pump_stream(
            what,
            source,
            offset,
            length,
            want_sha256,
            precision,
            slice,
            scratch,
            sink,
        )?;
        Ok(length)
    }

    /// Verify one tensor's payload against its recorded checksum.
    ///
    /// [`Artifact::stream_tensor`] with a sink that keeps nothing: the bytes
    /// are read, hashed and dropped, so verifying a 400 GB artifact costs the
    /// scratch buffer.
    pub fn verify_tensor(&self, role: &str, scratch: &mut [u8]) -> Result<u64> {
        self.stream_tensor(role, scratch, &mut |_| Ok(()))
    }

    /// The same, asked about cancellation between slices.
    ///
    /// `Ok(None)` means the caller cancelled. A whole-tensor verification is a
    /// whole-tensor read: independent review pointed out that checking only
    /// between logical tensors leaves a large one running to completion after
    /// the user has asked to stop. The scratch buffer is the bound on how much
    /// I/O a cancellation can still be waiting on.
    pub fn verify_tensor_cancellable(
        &self,
        role: &str,
        scratch: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<u64>> {
        let stopped = std::cell::Cell::new(false);
        let result = self.stream_tensor(role, scratch, &mut |_| {
            if cancelled() {
                stopped.set(true);
                return Err(Error::InvalidArtifact {
                    detail: "cancelled".into(),
                });
            }
            Ok(())
        });
        match result {
            Ok(n) => Ok(Some(n)),
            // The flag, not the message: only this sink sets it, so a genuine
            // checksum failure can never be mistaken for a cancellation.
            Err(_) if stopped.get() => Ok(None),
            Err(e) => Err(e),
        }
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
                detail: format!("no tensor named '{role}'").into(),
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
                    "tensor '{role}' is {}: this reads weights, and decoding an affine tensor into weights arrives in M3 item 3. Its bytes can be checked now -- `verify_tensor` and `stream_tensor` hash them against the manifest without decoding anything",
                    t.precision.name()
                ).into(),
            });
        }
        // One BF16 tensor, whichever container holds it: a chunk range in
        // version 1, the single `weights` component in version 2.
        let need: usize = {
            let elements: u64 = t.shape.iter().product();
            usize::try_from(elements * 2).map_err(|_| Error::InvalidArtifact {
                detail: format!("tensor '{role}' does not fit this platform").into(),
            })?
        };
        if into.len() < need {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "tensor '{role}' needs {need} bytes but the buffer holds {}: never a partial read reported as success",
                    into.len()
                ).into(),
            });
        }
        let mut at = 0usize;
        let mut scratch = [0u8; 8192];
        let read = self.stream_tensor(role, &mut scratch, &mut |slice| {
            let end = at + slice.len();
            if end > need {
                return Err(Error::InvalidArtifact {
                    detail: format!("tensor '{role}' streamed more than its {need} bytes").into(),
                });
            }
            into[at..end].copy_from_slice(slice);
            at = end;
            Ok(())
        })?;
        if read as usize != need || at != need {
            return Err(Error::InvalidArtifact {
                detail: format!("tensor '{role}' streamed {read} of {need} bytes").into(),
            });
        }
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
///
/// Two budgets, because there are two resources. [`HeaderBudget`] caps the
/// header, which is read whole and retained; [`ByteBudget`] caps the reader's
/// payload slicing, which is transient. Conflating them was a real defect: the
/// first version honoured only the second and allocated whatever header the
/// file declared.
#[derive(Debug)]
pub struct Shard {
    path: PathBuf,
    file: File,
    header: SafeHeader,
    /// SHA-256 of the header bytes this shard was parsed from.
    ///
    /// Every tensor offset in use came out of those bytes. A source rewritten
    /// **in place** -- same inode, same length, different tensor locations --
    /// changes nothing a stat can see, and independent review used exactly that
    /// to make a repack read one tensor's bytes and publish them as another's.
    header_sha256: String,
    len: u64,
    budget: ByteBudget,
    header_budget: HeaderBudget,
}

/// What a shard may spend on its header: **peak heap while opening it**, not
/// the header's serialized length.
///
/// [`ByteBudget`] caps the reader's *payload slicing* -- how much it holds while
/// pumping a tensor into a caller's buffer -- and a header is a different
/// resource: it is read whole, because it has to be parsed whole, and part of
/// it is retained for the shard's life.
///
/// Two rounds of independent review shaped this. The first found the header
/// simply unbudgeted: a shard opened with a sixteen-byte `ByteBudget` allocated
/// 65,575 bytes for its header. The second found the fix still measuring the
/// wrong thing -- a 54,899-byte serialized bound admitted a header whose peak
/// heap was 353,105 bytes and whose retained heap was 163,275, because parsing
/// allocates an entry list, a key string and a shape vector per tensor, a
/// temporary span list, and the retained maps. **A budget on the input is not a
/// budget on the memory that input costs.**
///
/// So admission is against [`HeaderBudget::estimated_peak`], and a regression
/// measures the real peak against that estimate rather than trusting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderBudget {
    bytes: u64,
}

impl HeaderBudget {
    /// 8 MiB of peak heap. The Gemma 4 shards' headers are 17--64 KB
    /// serialized, so their estimated peaks are well under 1 MiB and this
    /// leaves an order of magnitude of margin.
    pub const DEFAULT: Self = Self { bytes: 8 << 20 };

    /// Peak heap per serialized byte, **derived** rather than fitted.
    ///
    /// A header spends its bytes on constructs, and the cost of any mix is
    /// linear in that split: if it spends `b_i` bytes on construct `i` with
    /// `sum b_i <= S`, the peak is `sum b_i r_i <= S * max r_i`. So the sound
    /// bound is the **largest per-construct ratio**, and the job is to
    /// enumerate the constructs rather than to sample header shapes.
    ///
    /// Sampling is what the first two attempts did, and independent review
    /// defeated both with shapes the samples had not covered. The marginal
    /// costs below were measured two points apart, so they are slopes and not
    /// whole-header averages:
    ///
    /// | Construct | Min serialized | Peak heap | Ratio |
    /// |---|---:|---:|---:|
    /// | Tensor entry | 55 B | 297 B | 5.4 |
    /// | `__metadata__` entry | 9 B | 85 B | **9.5** |
    /// | Shape dimension | 2 B | 24 B | 12.0 |
    /// | Tensor-name byte | 1 B | 2 B | 2.0 |
    ///
    /// The shape dimension is the largest, and it is the one
    /// [`moxie_format::safetensors::MAX_RANK`] exists to bound: capped at eight
    /// dimensions, a tensor's dimensions cost at most `8 * 24` peak against at
    /// least 69 serialized bytes, a ratio of 2.8. That is a structural limit
    /// rather than a larger multiplier, because no multiplier survives an
    /// unbounded rank.
    ///
    /// With the rank bounded, the largest ratio is the metadata entry at 9.5.
    /// The factor is **16**, roughly 1.7x above it, for allocator and layout
    /// variation.
    /// `a_header_costs_no_more_peak_heap_than_its_admitted_estimate` measures
    /// every construct at its own worst shape and both of the review's
    /// counterexamples against this bound. **If it fails, enumerate the
    /// construct that beat it and bound that -- do not simply raise this
    /// number.**
    const PEAK_FACTOR: u64 = 16;

    /// Fixed overhead independent of header size, for the small-header case
    /// where per-entry costs dominate the ratio (23x at sixty-one bytes).
    const PEAK_FIXED: u64 = 8 << 10;

    /// A conservative peak-heap estimate for a header of `serialized` bytes,
    /// including the input buffer, the parser's allocations, the temporary
    /// validation structures and the retained maps.
    ///
    /// Sound only together with the structural limits in
    /// [`moxie_format::safetensors`]: without `MAX_RANK` the shape-dimension
    /// ratio exceeds this factor, which is how the previous bound was defeated.
    ///
    /// `None` when the arithmetic overflows, which is itself a refusal.
    pub const fn estimated_peak(serialized: u64) -> Option<u64> {
        match serialized.checked_mul(Self::PEAK_FACTOR) {
            Some(v) => v.checked_add(Self::PEAK_FIXED),
            None => None,
        }
    }

    pub const fn new(bytes: u64) -> Option<Self> {
        // The ceiling is the peak a maximum-size header could cost, not the
        // serialized ceiling: this budget is denominated in heap.
        let Some(ceiling) = Self::estimated_peak(moxie_format::safetensors::MAX_HEADER_BYTES)
        else {
            return None;
        };
        if bytes == 0 || bytes > ceiling {
            return None;
        }
        Some(Self { bytes })
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// The largest serialized header this budget admits.
    pub const fn max_serialized(self) -> u64 {
        (self.bytes.saturating_sub(Self::PEAK_FIXED)) / Self::PEAK_FACTOR
    }
}

impl Default for HeaderBudget {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Shard {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_limits(path, ByteBudget::DEFAULT, HeaderBudget::DEFAULT)
    }

    /// Open with an explicit payload-read budget and the default header budget.
    pub fn open_with_budget(path: &Path, budget: ByteBudget) -> Result<Self> {
        Self::open_with_limits(path, budget, HeaderBudget::DEFAULT)
    }

    /// Open with both budgets stated.
    pub fn open_with_limits(
        path: &Path,
        budget: ByteBudget,
        header_budget: HeaderBudget,
    ) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot open {}: {e}", path.display()).into(),
        })?;
        let len = file
            .metadata()
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot stat {}: {e}", path.display()).into(),
            })?
            .len();
        // Read the eight-byte length, learn the bound, then read exactly that
        // much -- rather than reading a length the file itself chose.
        let mut prefix = [0u8; 8];
        let mut source = OpenChunk { file: &file };
        source
            .read_at(0, &mut prefix)
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot read the length prefix of {}: {e}", path.display()).into(),
            })?;
        let prefix_len = SafeHeader::prefix_len(&prefix)?;
        // Both bounds before the allocation, not after. A header longer than
        // the file cannot be read at all, and one larger than the budget is a
        // resource the caller declined to spend.
        if prefix_len > len {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{} declares a {prefix_len}-byte header in a {len}-byte file",
                    path.display()
                )
                .into(),
            });
        }
        // Against the estimated **peak heap**, not the serialized length: the
        // parse costs several times what the bytes weigh.
        let peak = HeaderBudget::estimated_peak(prefix_len).ok_or(Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
            requested_bytes: u64::MAX,
            available_bytes: header_budget.bytes(),
        })?;
        if peak > header_budget.bytes() {
            return Err(Error::CapacityExceeded {
                tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
                requested_bytes: peak,
                available_bytes: header_budget.bytes(),
            });
        }
        let mut bytes = crate::try_vec::<u8>(usize::try_from(prefix_len).map_err(|_| {
            Error::InvalidArtifact {
                detail: "header length does not fit this platform".into(),
            }
        })?)?;
        bytes.resize(prefix_len as usize, 0);
        source
            .read_at(0, &mut bytes)
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot read the header of {}: {e}", path.display()).into(),
            })?;
        let header = SafeHeader::parse(&bytes, len)?;
        // **After** the parse, and streamed rather than one-shot. `sha256_hex`
        // copies its input, which for a header is the one allocation
        // `HeaderBudget` exists to bound -- doing it here and eagerly doubled
        // the admitted peak and spent it on headers that were about to be
        // refused. The streaming hasher reads the buffer in place.
        let header_sha256 = {
            let mut h = moxie_format::StreamingSha256::new();
            h.update(&bytes);
            h.finalize_hex()
        };
        Ok(Self {
            path: path.to_path_buf(),
            file,
            header,
            header_sha256,
            len,
            budget,
            header_budget,
        })
    }

    /// SHA-256 of the header bytes this shard's offsets came from.
    pub fn header_sha256(&self) -> &str {
        &self.header_sha256
    }

    /// This open file's identity, as the kernel knows it.
    ///
    /// `(device, inode)` on unix, taken from the **handle** rather than the
    /// path: a path can be renamed over between two questions, and a reader
    /// that asks the path twice is asking about two different files without
    /// being told. Elsewhere, length and modification time, which is weaker and
    /// says so.
    pub fn file_key(&self) -> Result<(u64, u64)> {
        let meta = self.file.metadata().map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot stat the open {}: {e}", self.path.display()).into(),
        })?;
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

    /// SHA-256 of this shard's whole file, read through **this** handle.
    ///
    /// Not by reopening the path. Independent review replaced a source file
    /// between an inspection and the repack that followed it: the cached handle
    /// still held the old inode while a fresh open of the same name hashed the
    /// new one, so the artifact carried the old file's values under the new
    /// file's digest. One open file description answers both questions.
    ///
    /// `Ok(None)` means the caller cancelled: a cancellation is not a corrupt
    /// artifact and must not be reported as one.
    pub fn digest_whole_file(
        &self,
        scratch: &mut [u8],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<(String, u64)>> {
        if scratch.is_empty() {
            return Err(Error::InvalidArtifact {
                detail: "a zero-byte scratch buffer cannot hash anything".into(),
            });
        }
        // The length **now**, from this handle, not the one captured when the
        // shard was opened. A file appended to after its first hash would
        // otherwise be hashed twice to the same stale end, and the digest would
        // describe a prefix while claiming to describe the file.
        let now = self
            .file
            .metadata()
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot stat the open {}: {e}", self.path.display()).into(),
            })?
            .len();
        if now != self.len {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{} was {} byte(s) when this run opened it and is {now} now: its header's offsets describe the file that was there, not the one that is",
                    self.path.display(),
                    self.len
                )
                .into(),
            });
        }
        let mut hasher = moxie_format::StreamingSha256::new();
        let mut done = 0u64;
        let mut source = OpenChunk { file: &self.file };
        while done < self.len {
            if cancelled() {
                return Ok(None);
            }
            let want = core::cmp::min(scratch.len() as u64, self.len - done) as usize;
            let buf = &mut scratch[..want];
            source
                .read_at(done, buf)
                .map_err(|e| Error::InvalidArtifact {
                    detail: format!("cannot read {}: {e}", self.path.display()).into(),
                })?;
            hasher.update(buf);
            done += want as u64;
        }
        Ok(Some((hasher.finalize_hex(), done)))
    }

    /// The header bytes this shard was admitted for.
    pub fn header_budget(&self) -> HeaderBudget {
        self.header_budget
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
            detail: format!("tensor {name:?} does not fit this platform").into(),
        })?;
        if into.len() != need {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "tensor {name:?} is {need} byte(s) but the buffer holds {}",
                    into.len()
                )
                .into(),
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
                    detail: format!("tensor {name:?} offset overflows").into(),
                })?;
            source
                .read_at(at, &mut into[done..end])
                .map_err(|e| Error::InvalidArtifact {
                    detail: format!("reading {name:?} from {}: {e}", self.path.display()).into(),
                })?;
            done = end;
        }
        Ok(())
    }

    /// Read one **bounded byte range** of one named tensor.
    ///
    /// This is what a residency cache demands, and it is the only shape of read
    /// that makes a fused expert tensor usable. Task 0020's designated artifact
    /// stores all 128 of a layer's experts in one `[128, 1408, 2816]` tensor:
    /// serving expert `e` through [`Shard::read_tensor`] would read
    /// 1,015,021,568 bytes to use 7,929,856 of them, and again for `down_proj`.
    /// Here it reads exactly the range asked for.
    ///
    /// The range is checked against the *validated header*, not against the
    /// file: an offset past the tensor's end is a refusal naming both, never a
    /// read that wanders into the next tensor. The buffer must be exactly the
    /// range's length, for the same reason `read_tensor` insists on it -- a
    /// partial read reported as success is the defect, not the short buffer.
    pub fn read_tensor_range(&self, name: &str, offset_bytes: u64, into: &mut [u8]) -> Result<()> {
        let entry = self.header.get(name)?;
        let len = entry.len();
        let want = into.len() as u64;
        let end = offset_bytes
            .checked_add(want)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("range of tensor {name:?} overflows").into(),
            })?;
        if want == 0 {
            return Err(Error::InvalidArtifact {
                detail: format!("an empty range of tensor {name:?} is not a read").into(),
            });
        }
        if end > len {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "range {offset_bytes}..{end} of tensor {name:?} exceeds its {len} byte(s)"
                )
                .into(),
            });
        }
        let base = entry
            .file_offset(&self.header)
            .checked_add(offset_bytes)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("tensor {name:?} offset overflows").into(),
            })?;
        let mut source = OpenChunk { file: &self.file };
        let slice = self.budget.bytes().max(1);
        let need = into.len();
        let mut done = 0usize;
        while done < need {
            let stop = (done + slice).min(need);
            let at = base
                .checked_add(done as u64)
                .ok_or_else(|| Error::InvalidArtifact {
                    detail: format!("tensor {name:?} offset overflows").into(),
                })?;
            source
                .read_at(at, &mut into[done..stop])
                .map_err(|e| Error::InvalidArtifact {
                    detail: format!("reading {name:?} from {}: {e}", self.path.display()).into(),
                })?;
            done = stop;
        }
        Ok(())
    }

    /// Read a tensor into a freshly allocated, fallibly sized buffer.
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self.header.get(name)?;
        let need = usize::try_from(entry.len()).map_err(|_| Error::InvalidArtifact {
            detail: format!("tensor {name:?} does not fit this platform").into(),
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

/// Read one exact byte range of a file, in scratch-sized slices.
///
/// Public because this crate is where bounded file reads live, and a caller
/// that has to re-read bytes it wrote -- the offline repacker, rehashing a
/// staged unit before it trusts a journal entry -- would otherwise hand-roll
/// `pread` again. It did, in an earlier arrangement, and the copy was
/// character-identical to [`OpenChunk::read_at`].
///
/// The range is checked against the file's length before anything is read, and
/// `sink` sees the bytes in file order, never more than `scratch` at a time.
pub fn read_range(
    path: &Path,
    offset: u64,
    len: u64,
    scratch: &mut [u8],
    sink: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    if scratch.is_empty() {
        return Err(Error::InvalidArtifact {
            detail: "a zero-byte scratch buffer cannot read anything".into(),
        });
    }
    let file = File::open(path).map_err(|e| Error::InvalidArtifact {
        detail: format!("cannot open {}: {e}", path.display()).into(),
    })?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::InvalidArtifact {
            detail: format!("range {offset}..+{len} of {} overflows", path.display()).into(),
        })?;
    let have = file
        .metadata()
        .map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot stat {}: {e}", path.display()).into(),
        })?
        .len();
    if end > have {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "{} holds {have} byte(s); the range {offset}..{end} is past its end",
                path.display()
            )
            .into(),
        });
    }
    let mut source = OpenChunk { file: &file };
    let mut done = 0u64;
    while done < len {
        let want = usize::try_from((len - done).min(scratch.len() as u64)).map_err(|_| {
            Error::InvalidArtifact {
                detail: "slice length does not fit this platform".into(),
            }
        })?;
        let buf = &mut scratch[..want];
        source
            .read_at(offset + done, buf)
            .map_err(|e| Error::InvalidArtifact {
                detail: format!("cannot read {}: {e}", path.display()).into(),
            })?;
        sink(buf)?;
        done += want as u64;
    }
    Ok(())
}

/// Read a whole text file, refusing anything above `cap` **before** reading it.
///
/// The rule `manifest.toml` is read under, made available to the other small
/// text file this repository's tooling reads back: the repacker's private
/// restart journal.
pub fn read_text_capped(path: &Path, cap: usize) -> Result<String> {
    let bytes = read_bytes_capped(path, cap)?;
    String::from_utf8(bytes).map_err(|e| Error::InvalidArtifact {
        detail: format!("{} is not UTF-8: {e}", path.display()).into(),
    })
}

/// The same, returning bytes.
///
/// A file whose last record was interrupted mid-write can end inside a
/// multi-byte character, and the repacker's journal recovery exists precisely
/// to discard such a tail. Decoding the whole file as UTF-8 first would refuse
/// it before recovery ever saw it -- independent review reproduced that with a
/// record torn inside a Unicode string -- so the caller that knows where its
/// records end takes the bytes and decodes the part it has committed.
pub fn read_bytes_capped(path: &Path, cap: usize) -> Result<Vec<u8>> {
    let have = std::fs::metadata(path)
        .map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot stat {}: {e}", path.display()).into(),
        })?
        .len();
    if have > cap as u64 {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "{} is {have} byte(s), above the {cap} byte cap checked before reading",
                path.display()
            )
            .into(),
        });
    }
    read_file_capped_bytes(path, cap)
}

/// Read `manifest.toml` through a capped reader that errors at the limit
/// rather than reading the file and then measuring it.
fn read_manifest_capped(dir: &Path) -> Result<String> {
    read_file_capped(&dir.join("manifest.toml"))
}

/// The same cap, for a manifest that is not yet called `manifest.toml`.
fn read_file_capped(path: &Path) -> Result<String> {
    let bytes = read_file_capped_bytes(path, moxie_format::manifest::MAX_MANIFEST_BYTES)?;
    String::from_utf8(bytes).map_err(|e| Error::InvalidArtifact {
        detail: format!("{} is not UTF-8: {e}", path.display()).into(),
    })
}

/// Read a whole file, stopping at `cap` rather than reading it and measuring
/// afterwards.
///
/// The cap is the caller's. It used to be the manifest's in every case, so a
/// caller that admitted a larger journal was refused at a limit it had never
/// asked for: independent review produced a valid 4,354,889-byte journal that
/// a resume could not reopen, because the manifest's 4 MiB bound was applied to
/// it.
fn read_file_capped_bytes(path: &Path, cap: usize) -> Result<Vec<u8>> {
    let path = path.to_path_buf();
    let mut f = File::open(&path).map_err(|e| Error::InvalidArtifact {
        detail: format!("cannot open {}: {e}", path.display()).into(),
    })?;
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let n = f.read(&mut chunk).map_err(|e| Error::InvalidArtifact {
            detail: format!("cannot read {}: {e}", path.display()).into(),
        })?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > cap {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{} exceeds the {cap} byte cap before parsing: it bounds the parse itself",
                    path.display()
                )
                .into(),
            });
        }
    }
    Ok(buf)
}

/// Confine a chunk reference to the artifact directory.
///
/// The string check runs first, before any path is joined: it stops traversal
/// without touching the filesystem. The canonical check runs after opening:
/// it stops symlinks pointing outside, which no string check can catch. Either
/// alone is insufficient.
fn resolve_chunk(canonical_dir: &Path, dir: &Path, name: &str) -> Result<PathBuf> {
    manifest::validate_chunk_name(name).map_err(|d| Error::InvalidArtifact {
        detail: format!("chunk reference rejected on the string, before joining: {d}").into(),
    })?;
    let joined = dir.join(name);
    let canonical = joined.canonicalize().map_err(|e| Error::InvalidArtifact {
        detail: format!("chunk '{name}' does not canonicalize: {e}").into(),
    })?;
    if !canonical.starts_with(canonical_dir) {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "chunk '{name}' resolves outside the artifact directory: a symlink pointing outside, which no string check can catch"
            ).into(),
        });
    }
    // A regular file: not a directory, not a device, not a dangling link that
    // somehow canonicalized.
    let meta = std::fs::metadata(&canonical).map_err(|e| Error::InvalidArtifact {
        detail: format!("cannot metadata chunk '{name}': {e}").into(),
    })?;
    if !meta.is_file() {
        return Err(Error::InvalidArtifact {
            detail: format!("chunk '{name}' is not a regular file").into(),
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
/// Fill a caller's whole buffer from one range, through the shared pump.
///
/// Only the unit tests below reach this shape now -- the artifact reader
/// streams -- and they are the reason it stays: interruption retry, truncation
/// and the BF16 boundary are properties of [`pump_stream`], and these are the
/// cases that hold it to them.
#[cfg(test)]
fn pump_range<S: RangeSource>(
    source: &mut S,
    offset: u64,
    dest: &mut [u8],
    want_sha256: &str,
    precision: TensorPrecision,
    budget: ByteBudget,
    chunk: &str,
) -> Result<()> {
    let length = dest.len() as u64;
    let slice = budget.bytes().max(1);
    pump_stream(
        chunk,
        source,
        offset,
        length,
        want_sha256,
        precision,
        slice,
        dest,
        &mut |_| Ok(()),
    )
}

/// Read one byte range in bounded slices, hashing it, validating BF16
/// finiteness when the bytes are BF16, and handing each slice to `sink`.
///
/// The one pump in this crate. `dest` is where the bytes land -- the caller's
/// buffer when it wants them all, a scratch buffer when it only wants them
/// streamed -- and `sink` sees each slice as it arrives. No reader-owned
/// allocation exceeds `slice`.
///
/// The bytes are trustworthy only once this returns `Ok`: the checksum is
/// verified after the last slice, which is why the error says so in as many
/// words.
#[allow(clippy::too_many_arguments)]
fn pump_stream<S: RangeSource>(
    what: &str,
    source: &mut S,
    offset: u64,
    length: u64,
    want_sha256: &str,
    precision: TensorPrecision,
    slice: usize,
    dest: &mut [u8],
    sink: &mut dyn FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut hasher = StreamingSha256::new();
    let mut bf16 = Bf16StreamValidator::new();
    let streaming = (dest.len() as u64) < length;
    let mut done: u64 = 0;
    while done < length {
        let want = usize::try_from((length - done).min(slice as u64)).map_err(|_| {
            Error::InvalidArtifact {
                detail: format!("{what}: slice length does not fit this platform").into(),
            }
        })?;
        // Streaming reuses the front of the buffer; filling writes in place.
        let at = if streaming { 0 } else { done as usize };
        let buf = &mut dest[at..at + want];
        let pos = offset
            .checked_add(done)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("{what}: offset {offset} + {done} overflows").into(),
            })?;
        // An interrupted syscall is retried without advancing: the offset has
        // not moved, so no byte is skipped or duplicated. Anything else --
        // including premature EOF -- is truncation, never a silent short read.
        match source.read_at(pos, buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "short read of chunk '{what}' at {pos} for {} bytes: truncation: {e}",
                        buf.len()
                    )
                    .into(),
                });
            }
        }
        hasher.update(buf);
        if matches!(precision, TensorPrecision::Bf16V1) {
            bf16.feed(buf)?;
        }
        sink(buf)?;
        done += want as u64;
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
                "{what}: checksum mismatch: expected {want_sha256}, computed {}; everything the \
                 sink was handed is explicitly not to be trusted",
                str::from_utf8(&hex_of(&got)).unwrap_or("?")
            )
            .into(),
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
                ).into(),
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
