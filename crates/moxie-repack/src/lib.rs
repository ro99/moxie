//! `moxie-repack`: the offline inspector, repacker and verifier.
//!
//! [ADR 0021] says repack is a Moxie program the user runs offline rather than
//! an external script, and [ADR 0022] places it here: argument parsing, the
//! offline workflow and reporting, over the shared codec (`moxie-format`), the
//! shared reader (`moxie-storage`) and the shared writer
//! (its own `write` module, which calls the reader's bounded primitives rather
//! than repeating them). It contains no second decoder, hash
//! implementation, manifest validator or filesystem writer, and `arch-check`
//! is what keeps that true rather than this paragraph.
//!
//! ## What it does, and what it deliberately does not
//!
//! `inspect` reads headers and reports what a selection would cost. `repack`
//! converts it in bounded work units, resumes an interrupted run, validates the
//! result through the production reader and publishes a manifest-v1 directory
//! with one rename. `verify` re-reads a published artifact through that same
//! reader. There is no unchecked publish, because publication **is** the last
//! validated step of `repack`.
//!
//! It never quantizes ([ADR 0017]), never discovers work for itself, never
//! writes outside the destination it is given, and never touches a source file
//! except to read declared byte ranges of it. A selection is text a user wrote.
//!
//! ## What a repack is not evidence of
//!
//! A value-preserving repack is [ADR 0018]'s definition of v1 quality, and it
//! is a statement about **bytes**. It says nothing about what a model produces:
//! that needs paired output against the released model, which is O2 and is not
//! this program's business. A published selection is also not a model: a
//! selection of one module publishes a *partial* artifact, the reader refuses
//! to load one, and `inspect` says so in as many words.
//!
//! [ADR 0017]: ../../../docs/decisions/adr/0017-v1-catalog-and-no-quantizer.md
//! [ADR 0018]: ../../../docs/decisions/adr/0018-v1-quality-is-bit-identical-repack.md
//! [ADR 0021]: ../../../docs/decisions/adr/0021-repack-is-a-moxie-program.md
//! [ADR 0022]: ../../../docs/decisions/adr/0022-user-programs-and-canonical-write-authority.md

#![forbid(unsafe_code)]

pub mod discover;
pub mod source;
pub mod work;
pub mod write;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::write::{Faults, Outcome, OutputPlan, Run, Start, TensorRequest, WriteBudget};
use moxie_format::affine::Grouping;
use moxie_format::compressed_tensors::{
    DeclaredSource, PackQuantizedPlan, ZeroPointSource, decode_weight_shape, scale_dtype_of,
};
use moxie_format::journal::RunBinding;
use moxie_format::manifest::{
    self, AffineFields, Architecture, Completeness, Endianness, GroupRule, Identity, Manifest,
    OpaqueArchMetadata, Provenance, ScaleDtype as ManifestScaleDtype, Source, SourceFile, Tensor,
    TensorPrecision, ZeroPointMode,
};
use moxie_format::payload::{self, ZeroPointSection};
use moxie_format::safetensors::{Dtype, TensorEntry};
use moxie_format::scale::ScaleDtype;
use moxie_format::selection::{Selection, SelectionKind};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_storage::{Artifact, ByteBudget, HeaderBudget};
use moxie_types::{HostTier, Result, Scope, Tier};

use crate::source::{Sources, invalid};

/// The converter identity recorded in a run's journal binding.
///
/// A resumed run must have been started by the same converter: a different one
/// may produce different canonical bytes from the same source, and half a
/// tensor from each is not a tensor. It is the crate version and the canonical
/// schema together, because either changing is enough to invalidate a resume.
pub fn converter_identity() -> String {
    format!(
        "moxie-repack/{}+manifest-v{}",
        env!("CARGO_PKG_VERSION"),
        manifest::SCHEMA_VERSION
    )
}

/// Every budget a run is given. None of them has a default here: the program
/// takes them from its command line, and the command line requires them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budgets {
    /// Total admitted dynamic working memory, including metadata.
    pub total_bytes: u64,
    /// Peak heap admitted for one source header.
    pub header_bytes: u64,
    /// Payload scratch: source tile plus canonical tile.
    pub scratch_bytes: usize,
    /// Largest output chunk file. A disk-plan parameter, not a RAM allocation.
    pub chunk_file_bytes: u64,
    /// Total payload bytes this run may write.
    pub disk_bytes: u64,
}

impl Budgets {
    /// What this program allocates besides its payload tiles, bounded.
    ///
    /// One source header at the declared budget, the selection text, the
    /// manifest text and the journal buffer. Every one of them is capped by a
    /// constant this repository already declares, so the floor is arithmetic
    /// rather than a guess.
    pub fn metadata_floor_bytes(&self) -> u64 {
        // **Not the selection cap.** That constant rose to 64 MiB so a plan for
        // a real model would fit, and using it here charged every run for the
        // largest plan anyone could write -- a 72 MiB floor on a checkpoint
        // with three tensors. What a selection actually costs is
        // `metadata_bound`, which is proportional to the one in hand and is
        // what the ledger admits. This stays a floor: the two allocations whose
        // size is genuinely fixed by a cap rather than by the input.
        self.header_bytes + moxie_format::manifest::MAX_MANIFEST_BYTES as u64
    }

    /// Bytes each selected tensor can hold in **parsed** form, at once.
    ///
    /// The selection's own text is counted separately; this is the per-tensor
    /// cost of everything built from it that is live at the same time: the
    /// parsed selection entry, the resolved tensor, up to three canonical
    /// components, the manifest row, the plan's placed components, the shard
    /// header's JSON, the progress map with its hasher, and the report row.
    /// Every one of those is a `String`, a `Vec` or a map node with its own
    /// allocation header, which is why the number is this large for structures
    /// that look small.
    ///
    /// Measured, not guessed: `tests/admission.rs` runs an inspection under an
    /// instrumented allocator and fails if the peak live heap exceeds what this
    /// admits.
    const PER_TENSOR_METADATA_BYTES: u64 = 4096;
    /// How many times a role's own bytes are retained across those structures.
    const ROLE_RETENTIONS: u64 = 16;
    /// One parsed journal record, without its names: two 64-character digests,
    /// the numbers, the struct and its map node.
    const PER_JOURNAL_RECORD_BYTES: u64 = 384;

    /// What this program may hold besides its payload tiles, for **this**
    /// selection.
    ///
    /// A constant floor was not a bound. Independent review inspected 5,000
    /// tensors under a 9,438,720-byte total and measured about 19,969,672 bytes
    /// of peak live heap, because `header + MAX_SELECTION_BYTES +
    /// MAX_MANIFEST_BYTES` are three facts about serialized text and none about
    /// the structures parsed out of it. This is proportional to the two
    /// quantities that actually drive those structures: how many tensors were
    /// selected, and how many bytes of names the selection carries.
    pub fn metadata_bound(&self, selection_bytes: u64, tensors: u64) -> u64 {
        self.header_bytes
            + moxie_format::manifest::MAX_MANIFEST_BYTES as u64
            + selection_bytes.saturating_mul(Self::ROLE_RETENTIONS)
            + tensors.saturating_mul(Self::PER_TENSOR_METADATA_BYTES)
    }

    /// What **recovery** holds, which the selection says nothing about.
    ///
    /// Reading an interrupted run's journal holds its bytes and the records
    /// parsed out of them at the same time, and how many records there are is a
    /// function of the payload and the work-unit size -- not of how many
    /// tensors were selected or how long their names are. Independent review
    /// resumed an 8 MiB source under a 9,438,720-byte total and measured
    /// 12,942,600 bytes of peak live heap against a 5,257,792-byte reservation.
    ///
    /// Admitted separately, after the plan exists and before recovery runs,
    /// because that is the first moment the unit count is known.
    pub fn recovery_bound(journal_bytes: u64, units: u64, role_bytes: u64) -> u64 {
        journal_bytes.saturating_add(
            units.saturating_mul(role_bytes.saturating_add(Self::PER_JOURNAL_RECORD_BYTES)),
        )
    }

    /// Refuse a budget that cannot hold what the run will hold.
    ///
    /// An independent review published with `--total-bytes 2048` -- consumed
    /// entirely by the three admitted buffers, with every header, selection,
    /// plan and manifest allocation outside the ledger -- and inspected with a
    /// total of zero. A total that does not cover the working set is not an
    /// admission, it is a number.
    pub fn validate(&self) -> Result<()> {
        if self.scratch_bytes == 0 || self.header_bytes == 0 || self.chunk_file_bytes == 0 {
            return Err(invalid(
                "every budget must be positive: a zero budget admits nothing and refuses nothing"
                    .into(),
            ));
        }
        let tiles = 3u64 * self.scratch_bytes as u64 / 2;
        let need = tiles + self.metadata_floor_bytes();
        if self.total_bytes < need {
            return Err(invalid(format!(
                "--total-bytes {} cannot hold this run: {tiles} byte(s) of payload tiles plus at \
                 most {} of header, selection and manifest. Admit at least {need}",
                self.total_bytes,
                self.metadata_floor_bytes()
            )));
        }
        Ok(())
    }

    /// Bytes for one tile of the payload scratch. Two tiles are live at once --
    /// the source bytes and the canonical bytes they convert into -- so each is
    /// half of what was admitted.
    pub fn tile_bytes(&self) -> usize {
        (self.scratch_bytes / 2).max(1)
    }
}

/// What `inspect` found, and what `repack` would cost.
#[derive(Debug, Clone)]
pub struct InspectReport {
    pub model: String,
    pub revision: String,
    pub source_root: PathBuf,
    pub selection_digest: String,
    pub tensors: Vec<TensorReport>,
    pub files: Vec<String>,
    /// Bytes of canonical tensor data: the components, and nothing else.
    pub canonical_payload_bytes: u64,
    /// Bytes the shards occupy, which is the above plus each shard's JSON
    /// header. Reported separately because a header is not tensor data and a
    /// disk plan has to count both.
    pub shard_bytes: u64,
    pub chunk_files: usize,
    pub largest_chunk_bytes: u64,
    pub source_payload_bytes: u64,
    pub headers_parsed: u64,
    pub header_bytes_read: u64,
    pub completeness: Completeness,
    pub plan_digest: String,
    /// An upper bound on what the destination directory holds at its largest,
    /// with the pieces it is made of.
    ///
    /// The payload is exact -- it is the descriptors' own arithmetic. The other
    /// two are bounds rather than sizes, because the manifest's length depends
    /// on the text in it and the journal's on how many units a run takes; both
    /// are derived from counts this inspection already knows, and both are
    /// removed at publication.
    pub staging: StagingEstimate,
}

/// What the destination holds at its peak, in three named parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingEstimate {
    pub payload_bytes: u64,
    pub journal_bound_bytes: u64,
    pub manifest_bound_bytes: u64,
}

/// What one selected tensor contributes to the staging files.
///
/// Every field is a length this program already knows before it writes
/// anything, and the role's own byte length is one of them. A **constant** per
/// line is not a bound: independent review selected a 1,000-character role and
/// watched a 160,000-byte disk budget retain 166,026 bytes.
#[derive(Debug, Clone, Copy)]
pub struct TensorStaging {
    pub role_bytes: u64,
    pub units: u64,
    pub components: u64,
    pub group_index_entries: u64,
}

impl StagingEstimate {
    /// Everything in a journal unit line except the names: the keys, two
    /// 64-hex digests, the numbers and the newline.
    const JOURNAL_LINE_FIXED: u64 = 384;
    /// The version line and the binding line, which carry three digests and a
    /// converter identity and no role.
    const JOURNAL_HEADER_BOUND: u64 = 4096;
    /// A shard or chunk file name. This program generates them
    /// (`model-NNNNN-of-NNNNN.safetensors`, 31 bytes); the allowance is four
    /// times that so a longer scheme cannot quietly invalidate the bound.
    const FILE_NAME_BOUND: u64 = 128;
    /// A manifest tensor entry's keys, shape, precision and affine fields,
    /// without its role or its components.
    const MANIFEST_TENSOR_FIXED: u64 = 1024;
    /// One component row: its checksum, its keys, and room for the file name.
    const MANIFEST_COMPONENT_FIXED: u64 = 256;
    /// The keys and punctuation of a manifest's global sections, without the
    /// values, which come from the selection and are counted separately.
    const MANIFEST_FIXED_BOUND: u64 = 16 * 1024;
    /// How much larger the manifest's global half can be than the selection
    /// text it is built from: TOML re-serialization repeats table keys, quotes
    /// and escapes strings, and adds a 64-character digest per source file.
    const MANIFEST_GLOBAL_EXPANSION: u64 = 4;

    fn of(payload_bytes: u64, tensors: &[TensorStaging], selection_bytes: u64) -> Self {
        let mut journal = Self::JOURNAL_HEADER_BOUND;
        // The **global** half of the manifest -- architecture metadata,
        // provenance, the tokenizer and template identities, the excluded list,
        // the source-file digests -- is not a constant. Every one of those
        // values comes from the selection, so the selection's own length bounds
        // them. Independent review put a 32 KiB architecture-metadata string in
        // a valid selection and watched publication fail against a derived
        // allowance of 22,435 bytes with a 64 MiB disk budget, which no budget
        // could fix.
        let mut manifest = Self::MANIFEST_FIXED_BOUND
            + selection_bytes.saturating_mul(Self::MANIFEST_GLOBAL_EXPANSION);
        for t in tensors {
            // A unit names its component, which is the role plus the longest
            // suffix this schema has (`.zero_points`, 12 bytes).
            let component_name = t.role_bytes + 16;
            journal +=
                t.units * (Self::JOURNAL_LINE_FIXED + component_name + Self::FILE_NAME_BOUND);
            // The role appears once as the tensor's own, and once inside each
            // component's name.
            manifest += Self::MANIFEST_TENSOR_FIXED
                + t.role_bytes
                + t.components
                    * (Self::MANIFEST_COMPONENT_FIXED + component_name + Self::FILE_NAME_BOUND)
                // A group-index map is the one manifest field whose length is
                // a tensor's own dimension rather than a constant.
                + t.group_index_entries * 8;
        }
        Self {
            payload_bytes,
            journal_bound_bytes: journal,
            manifest_bound_bytes: manifest,
        }
    }

    /// What a journal of `units` records costs, when the longest role in the
    /// plan is `longest_role_bytes` long.
    ///
    /// The **same arithmetic** `of` uses above, exposed so the planner sizes
    /// work units by the rule the run will bound itself by rather than by an
    /// estimate of its own. Independent review found a planner whose separate
    /// estimate chose a scratch the run then rejected; one rule cannot disagree
    /// with itself. Using the longest role for every record is the upper bound,
    /// so a plan this admits is one the run admits.
    pub fn journal_bound_for(units: u64, longest_role_bytes: u64) -> u64 {
        let component_name = longest_role_bytes.saturating_add(16);
        Self::JOURNAL_HEADER_BOUND.saturating_add(
            units.saturating_mul(
                Self::JOURNAL_LINE_FIXED
                    .saturating_add(component_name)
                    .saturating_add(Self::FILE_NAME_BOUND),
            ),
        )
    }

    /// The destination's largest moment, which is what a disk budget has to
    /// cover: payload, the staged manifest, and **two** journals -- compaction
    /// writes its replacement beside the original before renaming over it.
    pub fn total(&self) -> u64 {
        self.payload_bytes + 2 * self.journal_bound_bytes + self.manifest_bound_bytes
    }
}

/// One selected tensor, as inspection sees it.
#[derive(Debug, Clone)]
pub struct TensorReport {
    pub role: String,
    pub profile: String,
    pub shape: Vec<u64>,
    pub canonical_bytes: u64,
    pub source_bytes: u64,
    pub chunk: String,
    pub offset: u64,
    pub units: usize,
}

/// A tensor resolved against its shards: everything needed to convert it.
#[derive(Debug)]
pub struct Resolved {
    pub role: String,
    pub alignment: u64,
    /// The physical tensors this becomes ([ADR 0025]).
    ///
    /// [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md
    pub components: Vec<moxie_format::canonical::Component>,
    pub shape: Vec<u64>,
    pub precision: TensorPrecision,
    pub affine: Option<AffineFields>,
    pub canonical_bytes: u64,
    pub source_bytes: u64,
    pub kind: ResolvedKind,
}

/// How a resolved tensor's canonical bytes are produced.
#[derive(Debug)]
pub enum ResolvedKind {
    /// Copied through unchanged.
    Bf16 {
        file: String,
        name: String,
        len: u64,
    },
    /// Converted from a `pack-quantized` module.
    PackQuantized {
        module: String,
        plan: PackQuantizedPlan,
        files: BTreeMap<String, String>,
        section: ZeroPointSection,
    },
}

impl Resolved {
    pub fn profile(&self) -> String {
        match &self.kind {
            ResolvedKind::Bf16 { .. } => "bf16 passthrough".into(),
            ResolvedKind::PackQuantized { plan, section, .. } => {
                let d = plan.descriptor();
                format!(
                    "pack-quantized {} {} {} scales, {}",
                    d.width.profile(),
                    match d.grouping {
                        Grouping::PerOutputChannel => "per-channel".to_string(),
                        Grouping::Contiguous { size } => format!("group-{size}"),
                    },
                    d.scale_dtype.name(),
                    match section {
                        ZeroPointSection::Absent => "symmetric",
                        ZeroPointSection::PerGroup => "asymmetric",
                    }
                )
            }
        }
    }
}

/// Resolve a selection against its shards, reading headers only.
pub fn resolve(selection: &Selection, sources: &mut Sources) -> Result<Vec<Resolved>> {
    let mut out = Vec::with_capacity(selection.tensors.len());
    for t in &selection.tensors {
        let resolved = match &t.kind {
            SelectionKind::Bf16 { name, file } => {
                let entry = sources.entry(file, name)?;
                if entry.dtype != Dtype::Bf16 {
                    return Err(invalid(format!(
                        "tensor '{}': {name} in {file} is {}; a bf16 selection copies BF16 bytes \
                         through unchanged and refuses anything else rather than converting it",
                        t.role,
                        entry.dtype.name()
                    )));
                }
                if entry.shape.is_empty() {
                    return Err(invalid(format!(
                        "tensor '{}': {name} declares an empty shape",
                        t.role
                    )));
                }
                Resolved {
                    role: t.role.clone(),
                    alignment: t.alignment,
                    components: moxie_format::canonical::bf16_components(&t.role, &entry.shape)?,
                    shape: entry.shape.clone(),
                    precision: TensorPrecision::Bf16V1,
                    affine: None,
                    canonical_bytes: entry.len,
                    source_bytes: entry.len,
                    kind: ResolvedKind::Bf16 {
                        file: file.clone(),
                        name: name.clone(),
                        len: entry.len,
                    },
                }
            }
            SelectionKind::PackQuantized {
                module,
                spec,
                files,
            } => {
                let named = |suffix: &str| -> Result<(String, String)> {
                    let file = files.get(suffix).cloned().ok_or_else(|| {
                        invalid(format!("tensor '{}': no shard named for {suffix}", t.role))
                    })?;
                    Ok((file, format!("{module}.{suffix}")))
                };
                let (shape_file, shape_name) = named("weight_shape")?;
                let shape_bytes = sources.read_small(&shape_file, &shape_name, 64)?;
                let logical = decode_weight_shape(&shape_bytes)?;

                let (packed_file, packed_name) = named("weight_packed")?;
                let packed = sources.entry(&packed_file, &packed_name)?;

                // The importer's own entry rules, applied to entries resolved
                // across shards. An independent review published a module whose
                // packed codes and zero points were declared F32 and whose
                // shape was F64, and published a symmetric selection over a
                // source carrying zero points -- because this path looked at
                // shapes and never at what the single-header resolver checks.
                // The zero-point entry passed in is the one the **source**
                // carries, found wherever the selection's files put it, not the
                // one the selection expected.
                let zero_point_name = format!("{module}.weight_zero_point");
                let declared_zero_point = match files.get("weight_zero_point") {
                    Some(file) => sources
                        .declares(file, &zero_point_name)?
                        .then(|| sources.raw_entry(file, &zero_point_name))
                        .transpose()?,
                    // A symmetric selection names no zero-point file, so every
                    // shard the selection declares **for this module** is asked.
                    // Looking only beside the codes was the first fix and it was
                    // not enough: independent review put the codes in one shard
                    // and the scales with nonzero zero points in another, both
                    // named by the selection, and the symmetric claim published.
                    // A module split across shards is the normal case here --
                    // Qwen3.8-27B splits all 256 of them -- so "beside the
                    // codes" is not where a companion has to be.
                    None => {
                        let mut searched: Vec<&str> = files.values().map(String::as_str).collect();
                        searched.sort_unstable();
                        searched.dedup();
                        let mut found: Option<(String, TensorEntry)> = None;
                        for file in searched {
                            if sources.declares(file, &zero_point_name)? {
                                let entry = sources.raw_entry(file, &zero_point_name)?;
                                if let Some((first, _)) = &found {
                                    return Err(invalid(format!(
                                        "tensor '{}': {zero_point_name} is declared by both                                          '{first}' and '{file}'; a module's zero points live in                                          one shard, and two copies is a source this repacker                                          will not guess about",
                                        t.role
                                    )));
                                }
                                found = Some((file.to_string(), entry));
                            }
                        }
                        found.map(|(_, entry)| entry)
                    }
                };
                let packed_entry = sources.raw_entry(&packed_file, &packed_name)?;
                let shape_entry = sources.raw_entry(&shape_file, &shape_name)?;
                moxie_format::compressed_tensors::validate_source_entries(
                    module,
                    &packed_entry,
                    &shape_entry,
                    declared_zero_point.as_ref(),
                    spec.zero_points,
                )?;
                let (scale_file, scale_name) = named("weight_scale")?;
                let scale = sources.entry(&scale_file, &scale_name)?;
                let scale_dtype = scale_dtype_of(scale.dtype)?;
                let zero_point = match spec.zero_points {
                    ZeroPointSource::Symmetric => None,
                    ZeroPointSource::PackedAlongOutput => {
                        let (file, name) = named("weight_zero_point")?;
                        Some(sources.entry(&file, &name)?)
                    }
                };
                let plan = PackQuantizedPlan::new(
                    spec,
                    DeclaredSource {
                        packed_shape: &packed.shape,
                        scale_shape: &scale.shape,
                        scale_dtype,
                        zero_point_shape: zero_point.as_ref().map(|z| z.shape.as_slice()),
                        logical,
                    },
                )?;
                // The declared payload lengths, checked here rather than when
                // the first tile is short: an inspection that says a selection
                // is convertible has to have checked the sizes it read.
                let (want_packed, want_scale, want_zero) = plan.source_lengths()?;
                for (what, got, want) in [
                    ("weight_packed", packed.len, want_packed as u64),
                    ("weight_scale", scale.len, want_scale as u64),
                ] {
                    if got != want {
                        return Err(invalid(format!(
                            "tensor '{}': {module}.{what} is {got} byte(s); its declared shape \
                             needs {want}",
                            t.role
                        )));
                    }
                }
                if let (Some(zp), Some(want)) = (&zero_point, want_zero)
                    && zp.len != want as u64
                {
                    return Err(invalid(format!(
                        "tensor '{}': {module}.weight_zero_point is {} byte(s); its declared \
                         shape needs {want}",
                        t.role, zp.len
                    )));
                }
                let section = match spec.zero_points {
                    ZeroPointSource::Symmetric => ZeroPointSection::Absent,
                    ZeroPointSource::PackedAlongOutput => ZeroPointSection::PerGroup,
                };
                let canonical_bytes = payload::length_of(plan.descriptor(), section)?;
                let source_bytes =
                    packed.len + scale.len + zero_point.as_ref().map(|z| z.len).unwrap_or(0);
                let d = plan.descriptor();
                Resolved {
                    role: t.role.clone(),
                    alignment: t.alignment,
                    components: moxie_format::canonical::affine_components(
                        &t.role,
                        plan.descriptor(),
                        section,
                    )?,
                    shape: vec![d.out_features as u64, d.in_features as u64],
                    precision: match d.width {
                        moxie_format::affine::IntWidth::Int4 => TensorPrecision::AffineInt4V1,
                        moxie_format::affine::IntWidth::Int8 => TensorPrecision::AffineInt8V1,
                    },
                    affine: Some(AffineFields {
                        group_rule: match d.grouping {
                            Grouping::PerOutputChannel => GroupRule::PerChannel,
                            Grouping::Contiguous { size: 32 } => GroupRule::Contiguous32,
                            Grouping::Contiguous { size: 128 } => GroupRule::Contiguous128,
                            Grouping::Contiguous { size } => {
                                return Err(invalid(format!(
                                    "tensor '{}': group size {size} has no manifest group_rule",
                                    t.role
                                )));
                            }
                        },
                        scale_dtype: match d.scale_dtype {
                            ScaleDtype::F16 => ManifestScaleDtype::F16,
                            ScaleDtype::Bf16 => ManifestScaleDtype::Bf16,
                            ScaleDtype::F32 => ManifestScaleDtype::F32,
                        },
                        zero_point: match section {
                            ZeroPointSection::Absent => ZeroPointMode::Symmetric,
                            ZeroPointSection::PerGroup => ZeroPointMode::PerGroup,
                        },
                        // Preserved rather than invented: the importer refuses
                        // a source that carries an activation-order map, so a
                        // canonical group index can only be absent here. When
                        // that lane exists it arrives through the descriptor,
                        // never by being defaulted away.
                        group_index: d.group_index.clone(),
                    }),
                    canonical_bytes,
                    source_bytes,
                    kind: ResolvedKind::PackQuantized {
                        module: module.clone(),
                        plan,
                        files: files.clone(),
                        section,
                    },
                }
            }
        };
        out.push(resolved);
    }
    Ok(out)
}

/// Build the output plan from resolved tensors.
///
/// `overhead_bytes` is the bound on what the run writes **besides** payload --
/// the journal and the staged manifest -- so that the disk budget covers the
/// whole plan rather than the part that is easy to count.
pub fn output_plan(
    resolved: &[Resolved],
    budget: &WriteBudget,
    overhead_bytes: u64,
) -> Result<OutputPlan> {
    let requests = resolved
        .iter()
        .map(|r| TensorRequest {
            role: r.role.clone(),
            shape: r.shape.clone(),
            precision: r.precision,
            affine: r.affine.clone(),
            components: r.components.clone(),
        })
        .collect();
    OutputPlan::build(requests, budget, overhead_bytes)
}

/// Refuse a destination that overlaps the source root, either way round.
///
/// The destination usually does not exist yet, so the nearest existing ancestor
/// is what gets canonicalized: a path under `/fast/models/...` that has not been
/// created is still a path under the checkpoint root.
pub fn separate_source_and_destination(source_root: &Path, destination: &Path) -> Result<PathBuf> {
    // The destination usually does not exist yet, so the nearest existing
    // ancestor is canonicalized and the missing components are put back on.
    // Comparing the ancestor itself would refuse every destination that merely
    // shares a parent directory with the source.
    let mut missing: Vec<std::ffi::OsString> = Vec::new();
    // A relative destination is relative to where the user is standing, and
    // `--out new-output` is the ordinary way to say "here". Walking a bare
    // relative path's ancestors reaches the empty path, which canonicalizes to
    // nothing -- independent review hit exactly that and was told no part of
    // `new-output` resolves to a real directory.
    let mut probe = if destination.is_absolute() {
        destination.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| invalid(format!("cannot read the current directory: {e}")))?
            .join(destination)
    };
    let mut dest = loop {
        if let Ok(canonical) = probe.canonicalize() {
            break canonical;
        }
        match (probe.file_name(), probe.parent()) {
            (Some(name), Some(parent)) if parent != probe => {
                missing.push(name.to_os_string());
                probe = parent.to_path_buf();
            }
            _ => {
                return Err(invalid(format!(
                    "no part of {} resolves to a real directory",
                    destination.display()
                )));
            }
        }
    };
    for name in missing.iter().rev() {
        dest.push(name);
    }
    if dest.starts_with(source_root) || source_root.starts_with(&dest) {
        return Err(invalid(format!(
            "the destination {} overlaps the source root {}: checkpoint roots are read-only \
             inputs, and a repack never writes inside the artifact it is reading",
            dest.display(),
            source_root.display()
        )));
    }
    // The absolute destination, which is what everything downstream must use:
    // a relative path reaches `Run::begin` as written, and syncing the parent
    // of a bare name asks the filesystem to open the empty path.
    Ok(dest)
}

/// The journal and staged manifest a run of this shape can write, bounded.
///
/// Both are proportional to counts this program already knows -- units and
/// tensors -- and both are removed at publication. The disk budget is checked
/// against payload **plus** this, because a budget that covers only the part
/// that is easy to count is not a budget.
pub fn overhead_bound(
    resolved: &[Resolved],
    budgets: &Budgets,
    selection_bytes: u64,
) -> Result<u64> {
    let estimate = StagingEstimate::of(0, &staging_shape(resolved, budgets)?, selection_bytes);
    // **Two journals, not one.** Compaction writes its replacement beside the
    // original and renames over it, so for the length of that write the
    // destination holds both. Independent review measured a 250,000-byte disk
    // budget retaining 326,868 bytes at the moment of the second write. The
    // plan reserves the peak, not the resting size.
    let total = 2 * estimate.journal_bound_bytes + estimate.manifest_bound_bytes;
    // The journal has a cap of its own, and a plan whose journal cannot be read
    // back is a plan that cannot be resumed. Independent review produced a
    // valid 4.35 MB journal from an 8 MiB source at a 1 KiB scratch; raising
    // the reader's limit without bounding the writer would leave the same run
    // unresumable at a larger size.
    if estimate.journal_bound_bytes > moxie_format::journal::MAX_JOURNAL_BYTES as u64 {
        return Err(invalid(format!(
            "this selection would write up to {} journal byte(s), above the              {} byte cap a resume can read back. Larger work units (a bigger --scratch-bytes)              mean fewer records",
            estimate.journal_bound_bytes,
            moxie_format::journal::MAX_JOURNAL_BYTES
        )));
    }
    Ok(total)
}

/// The per-tensor shape of the staging files, from what is already resolved.
fn staging_shape(resolved: &[Resolved], budgets: &Budgets) -> Result<Vec<TensorStaging>> {
    let mut out = Vec::with_capacity(resolved.len());
    for r in resolved {
        out.push(TensorStaging {
            // The **serialized** length: what a journal line and a manifest row
            // actually cost. A role is a user-supplied string and escaping can
            // multiply it sixfold.
            role_bytes: moxie_format::journal::escaped_len(&r.role) as u64,
            units: work::unit_count(r, budgets.tile_bytes())? as u64,
            components: r.components.len() as u64,
            group_index_entries: r
                .affine
                .as_ref()
                .and_then(|a| a.group_index.as_ref())
                .map(|m| m.len() as u64)
                .unwrap_or(0),
        });
    }
    Ok(out)
}

/// A write budget from the program's budgets.
pub fn write_budget(budgets: &Budgets) -> Result<WriteBudget> {
    WriteBudget::new(
        budgets.scratch_bytes,
        budgets.chunk_file_bytes,
        budgets.disk_bytes,
    )
}

/// A ledger admitting at most `total_bytes` of pageable host memory, against
/// what this machine actually has.
///
/// The cap is the run's declared budget and the capacity is the measurement:
/// asking for 128 MiB on a machine that has less is a refusal, not a smaller
/// run that pretends to be the same one.
pub fn ledger_for(budgets: &Budgets) -> Result<Ledger> {
    let host = moxie_host::read()?;
    let snapshot = CapacitySnapshot::measured_host(&host, 0)?
        .with_tier_cap(Tier::Host(HostTier::Pageable), budgets.total_bytes)?;
    Ledger::new([snapshot])
}

/// Inspect a selection: headers only, no output directory side effects.
pub fn inspect(
    selection: &Selection,
    sources: &mut Sources,
    budgets: &Budgets,
    ledger: &mut Ledger,
) -> Result<InspectReport> {
    budgets.validate()?;
    // **Admitted before it is built, not measured after.** Everything below
    // this line -- the resolved tensors, the plan, every shard header, the
    // report -- is the allocation this charge stands for.
    let charge = admit_metadata(ledger, selection, budgets)?;
    let report = inspect_inner(selection, sources, budgets);
    ledger
        .release(charge)
        .map_err(|e| invalid(format!("cannot release the metadata charge: {e:?}")))?;
    report
}

/// Admit what this selection's parsed form will occupy.
fn admit_metadata(
    ledger: &mut Ledger,
    selection: &Selection,
    budgets: &Budgets,
) -> Result<Reservation> {
    let bound = budgets.metadata_bound(selection.source_bytes(), selection.tensors.len() as u64);
    let mut plan = PlanRequest::new("repack metadata", ["live"])?;
    plan.buffer(BufferRequest::new(
        "parsed selection, plan, headers and report",
        Scope::Host,
        Tier::Host(HostTier::Pageable),
        bound,
        StageSpan::at(0),
    ))?;
    ledger.admit(&plan).map_err(moxie_types::Error::from)
}

fn inspect_inner(
    selection: &Selection,
    sources: &mut Sources,
    budgets: &Budgets,
) -> Result<InspectReport> {
    let write = write_budget(budgets)?;
    let resolved = resolve(selection, sources)?;
    let plan = output_plan(
        &resolved,
        &write,
        overhead_bound(&resolved, budgets, selection.source_bytes())?,
    )?;
    let mut tensors = Vec::with_capacity(resolved.len());
    for r in resolved.iter() {
        tensors.push(TensorReport {
            role: r.role.clone(),
            profile: r.profile(),
            shape: r.shape.clone(),
            canonical_bytes: r.canonical_bytes,
            source_bytes: r.source_bytes,
            chunk: plan
                .components_of(&r.role)
                .map(|c| c.file.clone())
                .next()
                .unwrap_or_default(),
            offset: plan
                .components_of(&r.role)
                .map(|c| c.file_offset)
                .next()
                .unwrap_or(0),
            units: work::unit_count(r, budgets.tile_bytes())?,
        });
    }
    Ok(InspectReport {
        staging: StagingEstimate::of(
            plan.payload_bytes(),
            &staging_shape(&resolved, budgets)?,
            selection.source_bytes(),
        ),
        model: selection.model.clone(),
        revision: selection.revision.clone(),
        source_root: sources.root().to_path_buf(),
        selection_digest: selection_digest(selection),
        plan_digest: plan.digest(),
        canonical_payload_bytes: resolved.iter().map(|r| r.canonical_bytes).sum(),
        shard_bytes: plan.payload_bytes(),
        chunk_files: plan.shards().len(),
        largest_chunk_bytes: plan
            .shards()
            .iter()
            .map(|s| s.file_len())
            .max()
            .unwrap_or(0),
        source_payload_bytes: resolved.iter().map(|r| r.source_bytes).sum(),
        headers_parsed: sources.headers_parsed(),
        header_bytes_read: sources.bytes_read(),
        completeness: match &selection.completeness {
            moxie_format::selection::Completeness::Complete => Completeness::Complete,
            moxie_format::selection::Completeness::Partial { missing } => Completeness::Partial {
                missing: missing.clone(),
            },
        },
        files: selection.files(),
        tensors,
    })
}

/// A digest over the selection's own text-level decisions.
///
/// Distinct from the output plan's digest: this identifies *what was asked
/// for*, the plan identifies *where it goes*, and a run binds both.
pub fn selection_digest(selection: &Selection) -> String {
    let mut h = moxie_format::StreamingSha256::new();
    let mut field = |b: &[u8]| {
        h.update(&(b.len() as u64).to_le_bytes());
        h.update(b);
    };
    field(b"selection-v1");
    field(selection.model.as_bytes());
    field(selection.revision.as_bytes());
    field(selection.license.as_bytes());
    for id in [&selection.tokenizer, &selection.template] {
        field(id.name.as_bytes());
        field(id.version.as_bytes());
        field(id.digest.as_bytes());
    }
    field(selection.architecture_name.as_bytes());
    field(selection.architecture_version.as_bytes());
    field(selection.architecture_metadata.to_string().as_bytes());
    field(selection.scale_convention.as_bytes());
    field(selection.quantizer.as_bytes());
    field(selection.calibration.as_bytes());
    match &selection.completeness {
        moxie_format::selection::Completeness::Complete => field(b"complete"),
        moxie_format::selection::Completeness::Partial { missing } => {
            field(b"partial");
            for m in missing {
                field(m.as_bytes());
            }
        }
    }
    for (role, reason) in &selection.excluded {
        field(role.as_bytes());
        field(reason.as_bytes());
    }
    for t in &selection.tensors {
        field(t.role.as_bytes());
        field(&t.alignment.to_le_bytes());
        match &t.kind {
            SelectionKind::Bf16 { name, file } => {
                field(b"bf16");
                field(name.as_bytes());
                field(file.as_bytes());
            }
            SelectionKind::PackQuantized {
                module,
                spec,
                files,
            } => {
                field(b"pack-quantized");
                field(module.as_bytes());
                field(format!("{spec:?}").as_bytes());
                for (suffix, file) in files {
                    field(suffix.as_bytes());
                    field(file.as_bytes());
                }
            }
        }
    }
    h.finalize_hex()
}

/// What a completed repack measured.
#[derive(Debug, Clone)]
pub struct RepackReport {
    pub outcome: Outcome,
    pub artifact_identity: String,
    pub units_written: usize,
    pub units_reused: usize,
    pub bytes_written: u64,
    pub source_bytes_read: u64,
    pub resumed: bool,
    pub resume_detail: Vec<String>,
    pub source_digests: Vec<(String, String)>,
}

/// The whole offline workflow: convert, validate, publish.
///
/// A wrapper around the workflow whose only job is that **no failure leaves a
/// reservation outstanding**. An independent review injected the first
/// chunk-write failure and found all three charges still held: the early `?`
/// returns walked past the release at the end. Cleanup that lives at the end of
/// a function only runs for the paths that reach it.
#[allow(clippy::too_many_arguments)]
pub fn repack(
    selection: &Selection,
    sources: &mut Sources,
    destination: &Path,
    budgets: &Budgets,
    options: &crate::write::Options,
    faults: &Faults,
    cancelled: &dyn Fn() -> bool,
    ledger: &mut Ledger,
    progress: &mut dyn FnMut(&str),
) -> Result<RepackReport> {
    budgets.validate()?;
    // What the parsed selection, the plan, the shard headers and the manifest
    // will hold -- admitted before any of them is built, and proportional to
    // this selection rather than to a constant.
    let metadata = budgets.metadata_bound(selection.source_bytes(), selection.tensors.len() as u64);
    let mut buffers = work::Buffers::admit(ledger, budgets, metadata)?;
    let mut run_slot: Option<Run> = None;
    let result = repack_inner(
        selection,
        sources,
        destination,
        budgets,
        options,
        faults,
        cancelled,
        ledger,
        progress,
        &mut buffers,
        &mut run_slot,
    );
    // Whatever happened, give back what was admitted. A run still in the slot
    // is one that neither published nor cancelled, so it is abandoned: its
    // staged state stays on disk and remains resumable, which is the whole
    // point of the journal.
    if let Some(run) = run_slot.take() {
        let _ = run.abandon(ledger);
    }
    let _ = buffers.release(ledger);
    result
}

#[allow(clippy::too_many_arguments)]
fn repack_inner(
    selection: &Selection,
    sources: &mut Sources,
    destination: &Path,
    budgets: &Budgets,
    options: &crate::write::Options,
    faults: &Faults,
    cancelled: &dyn Fn() -> bool,
    ledger: &mut Ledger,
    progress: &mut dyn FnMut(&str),
    buffers: &mut work::Buffers,
    run_slot: &mut Option<Run>,
) -> Result<RepackReport> {
    budgets.validate()?;
    // The destination may not be the source, or inside it, or hold it. A repack
    // that publishes beneath the checkpoint it is reading is writing into a
    // read-only input, and an independent review found that publishing beneath
    // the source root succeeded. Checked on canonical paths, so a symlink into
    // the source root is caught with the rest.
    let destination = &separate_source_and_destination(sources.root(), destination)?;
    let write = write_budget(budgets)?;
    let resolved = resolve(selection, sources)?;
    let plan = output_plan(
        &resolved,
        &write,
        overhead_bound(&resolved, budgets, selection.source_bytes())?,
    )?;

    // Source identity before anything is written: the whole-file digests the
    // manifest will record, which are also what binds a resume. Hashing a
    // whole file is deliberate -- `source.files.sha256` means "this file", and
    // a digest of the ranges this run happened to read is not that.
    let mut source_digests: Vec<(String, String)> = Vec::new();
    for file in selection.files() {
        // A cancellation here is a cancellation, not a corrupt source. It
        // arrives before any destination exists, so there is nothing to resume
        // and nothing to clean up but the admitted buffers the caller releases.
        let Some((digest, bytes)) =
            sources.file_digest_cancellable(&file, buffers.source_tile_mut(), cancelled)?
        else {
            return Ok(RepackReport {
                outcome: Outcome::Cancelled {
                    destination: destination.to_path_buf(),
                    bytes_done: 0,
                },
                artifact_identity: String::new(),
                units_written: 0,
                units_reused: 0,
                bytes_written: 0,
                source_bytes_read: 0,
                resumed: false,
                resume_detail: Vec::new(),
                source_digests,
            });
        };
        progress(&format!(
            "hashed source {file}: {bytes} byte(s) -> {digest}"
        ));
        source_digests.push((file, digest));
    }

    let binding = RunBinding {
        plan_digest: run_digest(
            &plan.digest(),
            &selection_digest(selection),
            &source_digests,
        ),
        converter: converter_identity(),
        schema_version: manifest::SCHEMA_VERSION,
    };

    // **What recovery will hold, admitted before it holds it.** The unit count
    // is the thing that drives it, and this is the first line at which it is
    // known: the plan exists and nothing has been read back yet.
    let shape = staging_shape(&resolved, budgets)?;
    let units: u64 = shape.iter().map(|t| t.units).sum();
    let widest_role = shape.iter().map(|t| t.role_bytes).max().unwrap_or(0);
    let journal_bytes =
        StagingEstimate::of(0, &shape, selection.source_bytes()).journal_bound_bytes;
    buffers.admit_recovery(
        ledger,
        Budgets::recovery_bound(journal_bytes, units, widest_role),
    )?;

    let start = Run::begin(
        destination,
        plan,
        binding,
        write,
        options,
        ledger,
        faults,
        cancelled,
    )?;
    let (resumed, units_reused, resume_detail) = match start {
        Start::AlreadyPublished { artifact } => {
            return Err(invalid(format!(
                "{} already holds a published manifest: this slice never overwrites a published \
                 artifact and never updates a model in place",
                artifact.display()
            )));
        }
        Start::Cancelled { destination } => {
            return Ok(RepackReport {
                outcome: Outcome::Cancelled {
                    destination,
                    bytes_done: 0,
                },
                artifact_identity: String::new(),
                units_written: 0,
                units_reused: 0,
                bytes_written: 0,
                source_bytes_read: 0,
                resumed: true,
                resume_detail: vec!["cancelled while recovering the interrupted run".into()],
                source_digests,
            });
        }
        Start::Fresh(run) => {
            *run_slot = Some(run);
            (false, 0, Vec::new())
        }
        Start::Resumed(run, report) => {
            let mut detail = vec![format!(
                "resumed: {} unit(s) and {} byte(s) reused; {} byte(s) past the journal \
                 truncated; torn journal tail {} byte(s); phase {}",
                report.reused_units,
                report.reused_bytes,
                report.truncated_bytes,
                report.torn_journal_bytes,
                report.phase.name()
            )];
            detail.extend(report.discarded.iter().map(|d| format!("discarded {d}")));
            for line in &detail {
                progress(line);
            }
            *run_slot = Some(run);
            (true, report.reused_units, detail)
        }
    };

    let mut units_written = 0usize;
    let mut bytes_written = 0u64;
    let before_read = sources.bytes_read();
    for r in &resolved {
        let mut covered = 0u64;
        for unit in work::units(r, budgets.tile_bytes())? {
            covered += unit.canonical_len as u64;
            if cancelled() {
                let outcome = run_slot.take().expect("a run").cancel(ledger)?;
                return Ok(RepackReport {
                    outcome,
                    artifact_identity: String::new(),
                    units_written,
                    units_reused,
                    bytes_written,
                    source_bytes_read: sources.bytes_read() - before_read,
                    resumed,
                    resume_detail,
                    source_digests,
                });
            }
            // Each unit belongs to one physical component, and progress is
            // per component now: a shard holds three tensors of one logical
            // weight, each with its own offsets and its own checksum.
            let component = unit.component().tensor_name(&r.role);
            let done = run_slot.as_ref().expect("a run").bytes_done(&component)?;
            if unit.component_offset + unit.canonical_len as u64 <= done {
                // Already durable and already rehashed by the resume.
                continue;
            }
            if unit.component_offset < done {
                // Units are whole or absent, so a resumed tensor always
                // restarts on a unit boundary. Landing inside one means the
                // plan changed under a resume the binding should have refused.
                return Err(invalid(format!(
                    "component '{component}': a unit starting at {} lands inside the {} durable \
                     byte(s): units are whole or absent",
                    unit.component_offset, done
                )));
            }
            let source_sha = work::convert_unit(r, &unit, sources, buffers, ledger)?;
            let bytes = buffers.canonical(unit.canonical_len);
            run_slot
                .as_mut()
                .expect("a run")
                .write_unit(&component, bytes, &source_sha, faults)?;
            units_written += 1;
            bytes_written += unit.canonical_len as u64;
            // One line per durable unit, which is also the boundary the
            // program's cancellation and simulated-crash options count.
            progress(&format!(
                "unit {units_written} of '{component}': {} byte(s) at {}",
                unit.canonical_len, unit.component_offset
            ));
        }
        // The units are generated one at a time now, so the coverage check the
        // materialized list used to make is made here instead: a tensor whose
        // units do not add up to its payload would otherwise be caught only by
        // the seal, after the work.
        if covered != r.canonical_bytes {
            return Err(invalid(format!(
                "tensor '{}': its units cover {covered} byte(s) of a {} byte payload",
                r.role, r.canonical_bytes
            )));
        }
    }

    // **Before anything is exposed, hash every source again.** The digests
    // recorded above describe the files as they were when the run started; a
    // source that changed since would otherwise be published under a digest
    // that describes bytes nobody has. This is a second full pass over every
    // source file, and that cost is the price of the claim.
    progress("re-hashing every source to confirm it did not change");
    if !sources.verify_unchanged(&source_digests, buffers.source_tile_mut(), cancelled)? {
        let outcome = run_slot.take().expect("a run").cancel(ledger)?;
        return Ok(RepackReport {
            outcome,
            artifact_identity: String::new(),
            units_written,
            units_reused,
            bytes_written,
            source_bytes_read: sources.bytes_read() - before_read,
            resumed,
            resume_detail,
            source_digests,
        });
    }

    let sealed = run_slot.as_mut().expect("a run").seal()?;
    let manifest = build_manifest(selection, &resolved, &sealed, &source_digests)?;
    let text = manifest::encode(&manifest)?;
    let artifact_identity = manifest::artifact_identity(&manifest);
    let outcome = run_slot
        .take()
        .expect("a run")
        .publish(&text, cancelled, faults, ledger)?;
    Ok(RepackReport {
        outcome,
        artifact_identity,
        units_written,
        units_reused,
        bytes_written,
        source_bytes_read: sources.bytes_read() - before_read,
        resumed,
        resume_detail,
        source_digests,
    })
}

/// The binding digest: output plan, selection and every source file's digest.
pub fn run_digest(plan: &str, selection: &str, sources: &[(String, String)]) -> String {
    let mut h = moxie_format::StreamingSha256::new();
    let mut field = |b: &[u8]| {
        h.update(&(b.len() as u64).to_le_bytes());
        h.update(b);
    };
    field(b"repack-run-v1");
    field(plan.as_bytes());
    field(selection.as_bytes());
    for (file, digest) in sources {
        field(file.as_bytes());
        field(digest.as_bytes());
    }
    h.finalize_hex()
}

/// Assemble the manifest from the selection and what the run sealed.
pub fn build_manifest(
    selection: &Selection,
    resolved: &[Resolved],
    sealed: &[crate::write::SealedTensor],
    source_digests: &[(String, String)],
) -> Result<Manifest> {
    let mut tensors = Vec::with_capacity(resolved.len());
    for (order, r) in resolved.iter().enumerate() {
        if !sealed.iter().any(|s| s.role == r.role) {
            return Err(invalid(format!("tensor '{}' was not sealed", r.role)));
        }
        tensors.push(Tensor {
            role: r.role.clone(),
            shape: r.shape.clone(),
            precision: r.precision,
            logical_order: order as u64,
            affine: r.affine.clone(),
            placement: moxie_format::manifest::Placement::Components(
                sealed
                    .iter()
                    .filter(|s| s.role == r.role)
                    .map(|s| moxie_format::manifest::Component {
                        kind: s.kind,
                        file: s.file.clone(),
                        name: s.name.clone(),
                        sha256: s.sha256.clone(),
                    })
                    .collect(),
            ),
        });
    }
    Ok(Manifest {
        // This writer publishes version 2 and says so, rather than leaving the
        // version to be guessed from a row.
        schema_version: manifest::SCHEMA_VERSION,
        required_features: Vec::new(),
        endianness: Endianness::Little,
        source: Source {
            model: selection.model.clone(),
            revision: selection.revision.clone(),
            license: selection.license.clone(),
            files: source_digests
                .iter()
                .map(|(path, sha256)| SourceFile {
                    path: path.clone(),
                    sha256: sha256.clone(),
                })
                .collect(),
        },
        tokenizer: Identity {
            name: selection.tokenizer.name.clone(),
            version: selection.tokenizer.version.clone(),
            digest: selection.tokenizer.digest.clone(),
        },
        template: Identity {
            name: selection.template.name.clone(),
            version: selection.template.version.clone(),
            digest: selection.template.digest.clone(),
        },
        architecture: Architecture {
            name: selection.architecture_name.clone(),
            version: selection.architecture_version.clone(),
            metadata: OpaqueArchMetadata(selection.architecture_metadata.clone()),
        },
        provenance: Provenance {
            scale_convention: selection.scale_convention.clone(),
            quantizer: selection.quantizer.clone(),
            calibration: selection.calibration.clone(),
        },
        tensors,
        excluded: selection
            .excluded
            .iter()
            .map(|(role, reason)| manifest::Excluded {
                role: role.clone(),
                reason: reason.clone(),
            })
            .collect(),
        completeness: match &selection.completeness {
            moxie_format::selection::Completeness::Complete => Completeness::Complete,
            moxie_format::selection::Completeness::Partial { missing } => Completeness::Partial {
                missing: missing.clone(),
            },
        },
    })
}

/// What `verify` measured.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub artifact: PathBuf,
    pub identity: String,
    pub tensors: usize,
    pub bytes_verified: u64,
    pub completeness: Completeness,
    /// Bytes in the chunk files that no tensor's range covers.
    ///
    /// Alignment padding between tensors is legitimate and lands here, so this
    /// is not an error on its own. It is reported because the alternative is a
    /// verifier that reads every byte it was told about and stays silent about
    /// the ones it was not.
    pub unclaimed_bytes: u64,
}

/// Re-read a published artifact through the production reader.
pub fn verify(artifact_dir: &Path, scratch: &mut [u8]) -> Result<VerifyReport> {
    let artifact = Artifact::open(artifact_dir)?;
    let mut bytes = 0u64;
    for t in &artifact.manifest().tensors {
        bytes += artifact.verify_tensor(&t.role, scratch)?;
    }
    // Bytes in the shards that no component claims. A safetensors shard has a
    // header, which belongs to no tensor, so the accounting subtracts it: what
    // is left over after the header and every component is a file holding
    // bytes nothing describes.
    let mut claimed: BTreeMap<&str, u64> = BTreeMap::new();
    for t in &artifact.manifest().tensors {
        for c in t.components().unwrap_or(&[]) {
            let entry = artifact
                .shard_entry(&c.file, &c.name)
                .ok_or_else(|| invalid(format!("component '{}' is not open", c.name)))?;
            *claimed.entry(c.file.as_str()).or_default() += entry;
        }
    }
    let mut unclaimed = 0u64;
    for (file, bytes) in claimed {
        let len = artifact
            .chunk_len(file)
            .ok_or_else(|| invalid(format!("shard '{file}' was validated but has no length")))?;
        let header = artifact
            .shard_header_len(file)
            .ok_or_else(|| invalid(format!("shard '{file}' has no header length")))?;
        unclaimed += len.saturating_sub(bytes + header);
    }
    Ok(VerifyReport {
        unclaimed_bytes: unclaimed,
        artifact: artifact.dir().to_path_buf(),
        identity: artifact.identity(),
        tensors: artifact.manifest().tensors.len(),
        bytes_verified: bytes,
        completeness: artifact.manifest().completeness.clone(),
    })
}

/// The header budget to open source shards with.
pub fn header_budget(budgets: &Budgets) -> Result<HeaderBudget> {
    HeaderBudget::new(budgets.header_bytes).ok_or_else(|| {
        invalid(format!(
            "{} is not a valid header budget",
            budgets.header_bytes
        ))
    })
}

/// The read budget source shards slice their payload reads with.
pub fn read_budget(budgets: &Budgets) -> Result<ByteBudget> {
    ByteBudget::new(budgets.tile_bytes()).ok_or_else(|| {
        invalid(format!(
            "{} is not a valid read budget",
            budgets.tile_bytes()
        ))
    })
}

/// Open the sources a selection names.
pub fn open_sources(root: &Path, budgets: &Budgets) -> Result<Sources> {
    Sources::new(root, header_budget(budgets)?, read_budget(budgets)?)
}

/// Read a selection file, capped before it is read.
pub fn read_selection(path: &Path) -> Result<Selection> {
    let meta = std::fs::metadata(path)
        .map_err(|e| invalid(format!("cannot stat {}: {e}", path.display())))?;
    if meta.len() > moxie_format::selection::MAX_SELECTION_BYTES as u64 {
        return Err(invalid(format!(
            "{} is {} byte(s), above the selection cap",
            path.display(),
            meta.len()
        )));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
    moxie_format::selection::parse(&text)
}

/// The scope every admitted byte here is charged to.
pub const SCOPE: Scope = Scope::Host;
