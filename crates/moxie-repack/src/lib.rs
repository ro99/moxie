//! `moxie-repack`: the offline inspector, repacker and verifier.
//!
//! [ADR 0021] says repack is a Moxie program the user runs offline rather than
//! an external script, and [ADR 0022] places it here: argument parsing, the
//! offline workflow and reporting, over the shared codec (`moxie-format`), the
//! shared reader (`moxie-storage`) and the shared writer
//! (`moxie-storage-write`). It contains no second decoder, hash
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

pub mod source;
pub mod work;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
use moxie_format::safetensors::Dtype;
use moxie_format::scale::ScaleDtype;
use moxie_format::selection::{Selection, SelectionKind};
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_storage::{Artifact, ByteBudget, HeaderBudget};
use moxie_storage_write::{Faults, Outcome, OutputPlan, Run, Start, TensorRequest, WriteBudget};
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
    pub canonical_payload_bytes: u64,
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

impl StagingEstimate {
    /// Bytes per journal line, generously: a role of any plausible length, two
    /// 64-hex digests and the numbers.
    const JOURNAL_LINE_BOUND: u64 = 1024;
    /// Bytes per manifest tensor entry, generously, plus a fixed header
    /// allowance for the identity, provenance and completeness sections.
    const MANIFEST_TENSOR_BOUND: u64 = 2048;
    const MANIFEST_FIXED_BOUND: u64 = 16 * 1024;

    fn of(payload_bytes: u64, units: u64, tensors: u64, group_index_entries: u64) -> Self {
        Self {
            payload_bytes,
            journal_bound_bytes: (units + 2) * Self::JOURNAL_LINE_BOUND,
            manifest_bound_bytes: Self::MANIFEST_FIXED_BOUND
                + tensors * Self::MANIFEST_TENSOR_BOUND
                // A group-index map is the one manifest field whose length is
                // a tensor's own dimension rather than a constant.
                + group_index_entries * 8,
        }
    }

    pub fn total(&self) -> u64 {
        self.payload_bytes + self.journal_bound_bytes + self.manifest_bound_bytes
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
pub fn output_plan(resolved: &[Resolved], budget: &WriteBudget) -> Result<OutputPlan> {
    let requests = resolved
        .iter()
        .map(|r| TensorRequest {
            role: r.role.clone(),
            shape: r.shape.clone(),
            precision: r.precision,
            affine: r.affine.clone(),
            length: r.canonical_bytes,
            alignment: r.alignment,
        })
        .collect();
    OutputPlan::build(requests, budget)
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
) -> Result<InspectReport> {
    let write = write_budget(budgets)?;
    let resolved = resolve(selection, sources)?;
    let plan = output_plan(&resolved, &write)?;
    let mut tensors = Vec::with_capacity(resolved.len());
    for (r, planned) in resolved.iter().zip(plan.tensors()) {
        tensors.push(TensorReport {
            role: r.role.clone(),
            profile: r.profile(),
            shape: r.shape.clone(),
            canonical_bytes: r.canonical_bytes,
            source_bytes: r.source_bytes,
            chunk: planned.chunk.clone(),
            offset: planned.offset,
            units: work::units_of(r, budgets.tile_bytes())?.len(),
        });
    }
    let units: u64 = tensors.iter().map(|t| t.units as u64).sum();
    let group_index_entries: u64 = resolved
        .iter()
        .map(|r| {
            r.affine
                .as_ref()
                .and_then(|a| a.group_index.as_ref())
                .map(|m| m.len() as u64)
                .unwrap_or(0)
        })
        .sum();
    Ok(InspectReport {
        staging: StagingEstimate::of(
            plan.payload_bytes(),
            units,
            resolved.len() as u64,
            group_index_entries,
        ),
        model: selection.model.clone(),
        revision: selection.revision.clone(),
        source_root: sources.root().to_path_buf(),
        selection_digest: selection_digest(selection),
        plan_digest: plan.digest(),
        canonical_payload_bytes: plan.payload_bytes(),
        chunk_files: plan.chunks().len(),
        largest_chunk_bytes: plan.chunks().iter().map(|(_, l)| *l).max().unwrap_or(0),
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
#[allow(clippy::too_many_arguments)]
pub fn repack(
    selection: &Selection,
    sources: &mut Sources,
    destination: &Path,
    budgets: &Budgets,
    options: &moxie_storage_write::Options,
    faults: &Faults,
    cancelled: &dyn Fn() -> bool,
    ledger: &mut Ledger,
    progress: &mut dyn FnMut(&str),
) -> Result<RepackReport> {
    let write = write_budget(budgets)?;
    let resolved = resolve(selection, sources)?;
    let plan = output_plan(&resolved, &write)?;

    // Source identity before anything is written: the whole-file digests the
    // manifest will record, which are also what binds a resume. Hashing a
    // whole file is deliberate -- `source.files.sha256` means "this file", and
    // a digest of the ranges this run happened to read is not that.
    let mut buffers = work::Buffers::admit(ledger, budgets)?;
    let mut source_digests: Vec<(String, String)> = Vec::new();
    for file in selection.files() {
        let (digest, bytes) = sources.file_digest(&file, buffers.source_tile_mut())?;
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

    let start = Run::begin(destination, plan, binding, write, options, ledger, faults)?;
    let (mut run, resumed, units_reused, resume_detail) = match start {
        Start::AlreadyPublished { artifact } => {
            buffers.release(ledger)?;
            return Err(invalid(format!(
                "{} already holds a published manifest: this slice never overwrites a published \
                 artifact and never updates a model in place",
                artifact.display()
            )));
        }
        Start::Fresh(run) => (run, false, 0, Vec::new()),
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
            (run, true, report.reused_units, detail)
        }
    };

    let mut units_written = 0usize;
    let mut bytes_written = 0u64;
    let before_read = sources.bytes_read();
    for r in &resolved {
        let done = run.bytes_done(&r.role)?;
        for unit in work::units_of(r, budgets.tile_bytes())? {
            if cancelled() {
                let outcome = run.cancel(ledger)?;
                buffers.release(ledger)?;
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
            if unit.canonical_offset + unit.canonical_len as u64 <= done {
                // Already durable and already rehashed by the resume.
                continue;
            }
            if unit.canonical_offset < done {
                // Units are whole or absent, so a resumed tensor always
                // restarts on a unit boundary. Landing inside one means the
                // plan changed under a resume the binding should have refused.
                return Err(invalid(format!(
                    "tensor '{}': a unit starting at {} lands inside the {} durable byte(s): \
                     units are whole or absent",
                    r.role, unit.canonical_offset, done
                )));
            }
            let source_sha = work::convert_unit(r, &unit, sources, &mut buffers, ledger)?;
            let bytes = buffers.canonical(unit.canonical_len);
            run.write_unit(&r.role, bytes, &source_sha, faults)?;
            units_written += 1;
            bytes_written += unit.canonical_len as u64;
            // One line per durable unit, which is also the boundary the
            // program's cancellation and simulated-crash options count.
            progress(&format!(
                "unit {units_written} of '{}': {} byte(s) at {}",
                r.role, unit.canonical_len, unit.canonical_offset
            ));
        }
    }

    let sealed = run.seal()?;
    let manifest = build_manifest(selection, &resolved, &sealed, &source_digests)?;
    let text = manifest::encode(&manifest)?;
    let artifact_identity = manifest::artifact_identity(&manifest);
    let outcome = run.publish(&text, cancelled, faults, ledger)?;
    buffers.release(ledger)?;
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
    sealed: &[moxie_storage_write::SealedTensor],
    source_digests: &[(String, String)],
) -> Result<Manifest> {
    let mut tensors = Vec::with_capacity(resolved.len());
    for (order, r) in resolved.iter().enumerate() {
        let s = sealed
            .iter()
            .find(|s| s.role == r.role)
            .ok_or_else(|| invalid(format!("tensor '{}' was not sealed", r.role)))?;
        tensors.push(Tensor {
            role: r.role.clone(),
            shape: r.shape.clone(),
            precision: r.precision,
            chunk: s.chunk.clone(),
            offset: s.offset,
            length: s.length,
            sha256: s.sha256.clone(),
            alignment: r.alignment,
            logical_order: order as u64,
            affine: r.affine.clone(),
        });
    }
    Ok(Manifest {
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
    let mut chunks: BTreeMap<&str, u64> = BTreeMap::new();
    for t in &artifact.manifest().tensors {
        *chunks.entry(t.chunk.as_str()).or_default() += t.length;
    }
    let mut unclaimed = 0u64;
    for (chunk, claimed) in chunks {
        let len = artifact
            .chunk_len(chunk)
            .ok_or_else(|| invalid(format!("chunk '{chunk}' was validated but has no length")))?;
        unclaimed += len.saturating_sub(claimed);
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
