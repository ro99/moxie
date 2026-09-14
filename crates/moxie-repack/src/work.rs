//! Bounded work units: what one read, one conversion and one append cover.
//!
//! The contract is the budget. A repack admits a payload scratch measured in
//! megabytes and converts tensors measured in gigabytes, so the unit of work is
//! a **row range of one section**, never a tensor. Two tiles are live at a
//! time: the source bytes a unit reads and the canonical bytes it produces.
//!
//! Unit boundaries are a pure function of the plan and the tile size, so a
//! resumed run recomputes exactly the same ones -- which is what lets a
//! journal record "unit 7 of this tensor" and mean something after a restart.

use moxie_format::StreamingSha256;
use moxie_format::canonical::ComponentKind;
use moxie_format::compressed_tensors::PackQuantizedPlan;
use moxie_format::payload::{self, ZeroPointSection};
use moxie_memory::{BufferRequest, HostBuffer, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_types::{HostTier, Result, Scope, Tier};

use crate::source::{Sources, invalid};
use crate::{Budgets, Resolved, ResolvedKind};

/// One bounded unit of work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// Where this unit's bytes start inside the tensor's canonical payload:
    /// the concatenation ADR 0023 describes, which is still what a consumer
    /// streaming the whole tensor sees.
    pub canonical_offset: u64,
    /// Where they start inside **their own component**, which is where they
    /// are written now that each component is a physical safetensors tensor.
    pub component_offset: u64,
    pub canonical_len: usize,
    pub source: UnitSource,
}

impl Unit {
    /// Which physical component these bytes belong to.
    pub fn component(&self) -> ComponentKind {
        match self.source {
            UnitSource::Bytes { .. } => ComponentKind::Weights,
            UnitSource::Codes { .. } => ComponentKind::Codes,
            UnitSource::Scales { .. } => ComponentKind::Scales,
            UnitSource::Zeros { .. } => ComponentKind::ZeroPoints,
        }
    }
}

/// What a unit reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitSource {
    /// A byte range of one source tensor, copied through unchanged.
    Bytes { offset: u64, len: usize },
    /// Output rows of `weight_packed`.
    Codes { start: usize, end: usize },
    /// Output rows of `weight_scale`.
    Scales { start: usize, end: usize },
    /// Output rows of `weight_zero_point`, word-aligned on the output axis.
    Zeros { start: usize, end: usize },
}

/// The smallest payload tile this tensor can be converted with.
///
/// A zero-point block must cover whole source words, because one word holds
/// several consecutive **output channels**; so the minimum is a word's worth of
/// canonical zero points, not a row's. An independent review passed
/// `--scratch-bytes 8` for an asymmetric INT4 `[8,8]` per-channel fixture and
/// got exit 101: the block was forced to a word and the slice that received it
/// held four bytes. A budget too small to do the work is a refusal with a
/// number in it, stated before anything is created -- not a panic partway
/// through.
pub fn minimum_tile_bytes(tensor: &Resolved) -> Result<usize> {
    Ok(match &tensor.kind {
        ResolvedKind::Bf16 { .. } => 1,
        ResolvedKind::PackQuantized { plan, section, .. } => {
            let codes = plan
                .source_code_row_bytes()
                .max(plan.canonical_code_row_bytes());
            let scales = plan.scale_row_bytes();
            let zeros = if *section == ZeroPointSection::PerGroup {
                let per_word = plan.zero_points_per_word();
                plan.canonical_zero_point_row_bytes()
                    .checked_mul(per_word)
                    .ok_or_else(|| invalid("zero-point block size overflows".into()))?
                    .max(plan.source_zero_point_word_row_bytes())
            } else {
                0
            };
            codes.max(scales).max(zeros).max(1)
        }
    })
}

/// How many units a tensor takes, **without building them**.
///
/// Inspection needs the count and the disk plan needs it; neither needs the
/// list. An independent review found `units_of` materializing every unit of
/// every tensor -- an allocation that grows with the conversion, in a program
/// whose budget is supposed to bound exactly that.
pub fn unit_count(tensor: &Resolved, tile: usize) -> Result<usize> {
    let need = minimum_tile_bytes(tensor)?;
    if tile < need {
        return Err(invalid(format!(
            "tensor '{}' needs at least {need} byte(s) of payload tile and this run admitted \
             {tile}: raise --scratch-bytes to at least {}",
            tensor.role,
            need * 2
        )));
    }
    Ok(match &tensor.kind {
        ResolvedKind::Bf16 { len, .. } => (*len as usize).div_ceil(tile.max(1)),
        ResolvedKind::PackQuantized { plan, section, .. } => {
            let rows = plan.out_features();
            let codes = rows.div_ceil(block_rows(
                tile,
                plan.source_code_row_bytes()
                    .max(plan.canonical_code_row_bytes()),
                1,
            ));
            let scales = rows.div_ceil(block_rows(tile, plan.scale_row_bytes(), 1));
            let zeros = if *section == ZeroPointSection::PerGroup {
                rows.div_ceil(block_rows(
                    tile,
                    plan.canonical_zero_point_row_bytes()
                        .max(plan.source_zero_point_word_row_bytes()),
                    plan.zero_points_per_word(),
                ))
            } else {
                0
            };
            codes + scales + zeros
        }
    })
}

/// Rows per block, rounded down to a multiple of `granularity`.
///
/// One definition, used by the counter and by the splitter, so the two can
/// never disagree about how many units there are.
fn block_rows(tile: usize, per_row: usize, granularity: usize) -> usize {
    let rows = (tile / per_row.max(1)).max(1);
    (rows / granularity.max(1)).max(1) * granularity.max(1)
}

/// One tensor's units, produced **one at a time**.
///
/// The conversion loop needs the next unit, never the list, and an independent
/// review found the list being built: an allocation proportional to the
/// conversion, in a program whose budget exists to bound exactly that. A 400 GB
/// tensor at half-megabyte tiles is 800,000 descriptors nobody reads twice.
///
/// The section order is ADR 0023's -- codes, scales, zero points -- so
/// concatenating what this yields is the canonical payload, and the arithmetic
/// is [`block_rows`], shared with [`unit_count`] so the two can never disagree.
#[derive(Debug)]
pub struct Units<'a> {
    tensor: &'a Resolved,
    tile: usize,
    /// 0 = codes (or the whole payload, for BF16), 1 = scales, 2 = zero
    /// points, 3 = finished.
    section: u8,
    /// Next output row, or next byte for a BF16 passthrough.
    row: usize,
    /// Where the next unit's bytes start in the canonical payload.
    at: u64,
    rows: usize,
    scales_start: u64,
    zeros_start: Option<u64>,
}

impl<'a> Units<'a> {
    fn new(tensor: &'a Resolved, tile: usize) -> Result<Self> {
        let (rows, scales_start, zeros_start) = match &tensor.kind {
            ResolvedKind::Bf16 { len, .. } => (*len as usize, 0, None),
            ResolvedKind::PackQuantized { plan, section, .. } => {
                let ext = payload::extents(plan.descriptor(), *section)?;
                (
                    plan.out_features(),
                    ext.scales().start as u64,
                    ext.zero_points().map(|z| z.start as u64),
                )
            }
        };
        Ok(Self {
            tensor,
            tile,
            section: 0,
            row: 0,
            at: 0,
            rows,
            scales_start,
            zeros_start,
        })
    }
}

impl Iterator for Units<'_> {
    type Item = Unit;

    fn next(&mut self) -> Option<Unit> {
        match &self.tensor.kind {
            ResolvedKind::Bf16 { len, .. } => {
                if self.at >= *len {
                    return None;
                }
                let take = ((*len - self.at) as usize).min(self.tile);
                let unit = Unit {
                    canonical_offset: self.at,
                    // One component, so the two offsets coincide.
                    component_offset: self.at,
                    canonical_len: take,
                    source: UnitSource::Bytes {
                        offset: self.at,
                        len: take,
                    },
                };
                self.at += take as u64;
                Some(unit)
            }
            ResolvedKind::PackQuantized { plan, .. } => loop {
                if self.section > 2 {
                    return None;
                }
                if self.row >= self.rows {
                    // Move to the next section, at its own start offset.
                    self.section += 1;
                    self.row = 0;
                    match self.section {
                        1 => self.at = self.scales_start,
                        2 => self.at = self.zeros_start?,
                        _ => return None,
                    }
                    continue;
                }
                let (per_row, granularity, canonical_row) = match self.section {
                    0 => (
                        plan.source_code_row_bytes()
                            .max(plan.canonical_code_row_bytes()),
                        1,
                        plan.canonical_code_row_bytes(),
                    ),
                    1 => (plan.scale_row_bytes(), 1, plan.scale_row_bytes()),
                    _ => (
                        plan.canonical_zero_point_row_bytes()
                            .max(plan.source_zero_point_word_row_bytes()),
                        plan.zero_points_per_word(),
                        plan.canonical_zero_point_row_bytes(),
                    ),
                };
                let block = block_rows(self.tile, per_row, granularity);
                let start = self.row;
                let end = (start + block).min(self.rows);
                let len = (end - start) * canonical_row;
                let source = match self.section {
                    0 => UnitSource::Codes { start, end },
                    1 => UnitSource::Scales { start, end },
                    _ => UnitSource::Zeros { start, end },
                };
                let section_start = match self.section {
                    0 => 0,
                    1 => self.scales_start,
                    _ => self.zeros_start.unwrap_or(self.at),
                };
                let unit = Unit {
                    canonical_offset: self.at,
                    component_offset: self.at - section_start,
                    canonical_len: len,
                    source,
                };
                self.at += len as u64;
                self.row = end;
                return Some(unit);
            },
        }
    }
}

/// One tensor's units, lazily. Refuses a tile too small to hold a block first,
/// so the refusal arrives before any output exists rather than as a panic
/// partway through.
pub fn units(tensor: &Resolved, tile: usize) -> Result<Units<'_>> {
    let need = minimum_tile_bytes(tensor)?;
    if tile < need {
        return Err(invalid(format!(
            "tensor '{}' needs at least {need} byte(s) of payload tile and this run admitted \
             {tile}: raise --scratch-bytes to at least {}",
            tensor.role,
            need * 2
        )));
    }
    Units::new(tensor, tile.max(1))
}

/// Split one resolved tensor into units no larger than `tile`.
///
/// The three sections are consecutive, in ADR 0023's order, so concatenating
/// every unit's bytes is exactly the canonical payload.
pub fn units_of(tensor: &Resolved, tile: usize) -> Result<Vec<Unit>> {
    let units: Vec<Unit> = units(tensor, tile)?.collect();
    if units.is_empty() {
        return Err(invalid(format!(
            "tensor '{}' produced no work units",
            tensor.role
        )));
    }
    let total: u64 = units.iter().map(|u| u.canonical_len as u64).sum();
    if total != tensor.canonical_bytes {
        return Err(invalid(format!(
            "tensor '{}': its units cover {total} byte(s) of a {} byte payload",
            tensor.role, tensor.canonical_bytes
        )));
    }
    Ok(units)
}

/// The two payload tiles and the column scratch, all admitted.
#[derive(Debug)]
pub struct Buffers {
    source: HostBuffer,
    canonical: HostBuffer,
    columns: Vec<i32>,
    columns_charge: Option<Reservation>,
    /// The whole-run metadata charge: headers, selection, manifest.
    metadata_charge: Option<Reservation>,
}

impl Buffers {
    /// Admit both tiles **and the metadata this run holds beside them**.
    ///
    /// Headers, the selection text, the plan and the manifest are allocations
    /// this program makes for the whole run, and an independent review found
    /// every one of them outside the ledger -- so a 2,048-byte total admitted
    /// three buffers and accounted for nothing else. They are charged here, at
    /// the caps the format crate declares, and released with the tiles.
    pub fn admit(ledger: &mut Ledger, budgets: &Budgets, metadata: u64) -> Result<Self> {
        let tile = budgets.tile_bytes();
        let mut plan = PlanRequest::new("repack metadata", ["live"])?;
        plan.buffer(BufferRequest::new(
            "headers, selection and manifest",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            metadata,
            StageSpan::at(0),
        ))?;
        let metadata_charge = ledger.admit(&plan).map_err(moxie_types::Error::from)?;
        let source =
            HostBuffer::allocate_in(ledger, "repack source tile", HostTier::Pageable, tile, 0)?;
        let canonical =
            HostBuffer::allocate_in(ledger, "repack canonical tile", HostTier::Pageable, tile, 0)?;
        Ok(Self {
            source,
            canonical,
            columns: Vec::new(),
            columns_charge: None,
            metadata_charge: Some(metadata_charge),
        })
    }

    pub fn source_tile_mut(&mut self) -> &mut [u8] {
        self.source.bytes_mut()
    }

    /// The canonical tile's first `len` bytes: what one unit produced.
    pub fn canonical(&self, len: usize) -> &[u8] {
        &self.canonical.bytes()[..len]
    }

    /// Make room for one row of signed codes, admitting the bytes first.
    ///
    /// Charged through the ledger rather than allocated quietly: it is small,
    /// it is proportional to a source's declared width, and "small" is how
    /// unaccounted allocations start.
    fn columns(&mut self, width: usize, ledger: &mut Ledger) -> Result<&mut [i32]> {
        if self.columns.len() < width {
            if let Some(previous) = self.columns_charge.take() {
                ledger.release(previous).map_err(|e| {
                    invalid(format!("cannot release the column scratch charge: {e:?}"))
                })?;
            }
            let bytes = width
                .checked_mul(size_of::<i32>())
                .ok_or_else(|| invalid("column scratch size overflows".into()))?;
            let mut plan = PlanRequest::new("repack column scratch", ["live"])?;
            // No `Scaling` declared: this scratch is one row of the widest
            // selected tensor. It does not shrink with a shorter context or
            // fewer branches, and declaring that it does would license a
            // refusal to suggest something that cannot help.
            plan.buffer(BufferRequest::new(
                "column scratch",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                bytes as u64,
                StageSpan::at(0),
            ))?;
            let charge = ledger.admit(&plan).map_err(moxie_types::Error::from)?;
            self.columns = Vec::new();
            self.columns
                .try_reserve_exact(width)
                .map_err(|_| invalid(format!("cannot reserve {width} column(s) of scratch")))?;
            self.columns.resize(width, 0);
            self.columns_charge = Some(charge);
        }
        Ok(&mut self.columns[..width])
    }

    /// Give every admitted byte back.
    ///
    /// Tolerant of being called twice: the failure path calls it after an
    /// error that may already have released, and a cleanup that panics on a
    /// second call is a cleanup nobody can put in an error path.
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        let _ = self.source.release(ledger);
        let _ = self.canonical.release(ledger);
        if let Some(charge) = self.metadata_charge.take() {
            ledger
                .release(charge)
                .map_err(|e| invalid(format!("cannot release the metadata charge: {e:?}")))?;
        }
        if let Some(charge) = self.columns_charge.take() {
            ledger
                .release(charge)
                .map_err(|e| invalid(format!("cannot release the column scratch charge: {e:?}")))?;
            self.columns = Vec::new();
        }
        Ok(())
    }
}

/// Read one unit's source bytes and convert them into canonical bytes.
///
/// Returns the SHA-256 of the **source** bytes it consumed, which the journal
/// records so that a resumed run can tell whether the source moved under it.
pub fn convert_unit(
    tensor: &Resolved,
    unit: &Unit,
    sources: &mut Sources,
    buffers: &mut Buffers,
    ledger: &mut Ledger,
) -> Result<String> {
    match (&tensor.kind, &unit.source) {
        (ResolvedKind::Bf16 { file, name, .. }, UnitSource::Bytes { offset, len }) => {
            let src = &mut buffers.source.bytes_mut()[..*len];
            sources.read_range(file, name, *offset, src)?;
            let digest = sha256_of(src);
            // A BF16 payload's canonical bytes are its source bytes: ADR
            // 0018's bit-identical repack in its simplest form. The copy is
            // through the named writer so that every canonical byte in this
            // program leaves the same module.
            let (source, canonical) = (&buffers.source, &mut buffers.canonical);
            payload::write_code_block(&source.bytes()[..*len], &mut canonical.bytes_mut()[..*len])?;
            Ok(digest)
        }
        (ResolvedKind::PackQuantized { plan, .. }, UnitSource::Codes { start, end }) => {
            let stride = plan.source_code_row_bytes();
            let src_len = (end - start) * stride;
            read_module_range(
                tensor,
                sources,
                buffers,
                "weight_packed",
                (*start * stride) as u64,
                src_len,
            )?;
            let digest = sha256_of(&buffers.source.bytes()[..src_len]);
            let width = plan.descriptor().in_features;
            let plan = plan.clone();
            let columns_len = width;
            // The three borrows are taken apart explicitly: the column scratch
            // has to be admitted (which needs the ledger) before the tiles are
            // borrowed for the conversion.
            buffers.columns(columns_len, ledger)?;
            let Buffers {
                source,
                canonical,
                columns,
                ..
            } = buffers;
            plan.convert_code_rows(
                *start..*end,
                &source.bytes()[..src_len],
                &mut columns[..columns_len],
                &mut canonical.bytes_mut()[..unit.canonical_len],
            )?;
            Ok(digest)
        }
        (ResolvedKind::PackQuantized { plan, .. }, UnitSource::Scales { start, end }) => {
            let stride = plan.scale_row_bytes();
            let src_len = (end - start) * stride;
            read_module_range(
                tensor,
                sources,
                buffers,
                "weight_scale",
                (*start * stride) as u64,
                src_len,
            )?;
            let digest = sha256_of(&buffers.source.bytes()[..src_len]);
            let Buffers {
                source, canonical, ..
            } = buffers;
            plan.convert_scale_rows(
                *start..*end,
                &source.bytes()[..src_len],
                &mut canonical.bytes_mut()[..unit.canonical_len],
            )?;
            Ok(digest)
        }
        (ResolvedKind::PackQuantized { plan, .. }, UnitSource::Zeros { start, end }) => {
            let words = plan.zero_point_word_rows(*start..*end)?;
            let stride = plan.source_zero_point_word_row_bytes();
            let src_len = (words.end - words.start) * stride;
            read_module_range(
                tensor,
                sources,
                buffers,
                "weight_zero_point",
                (words.start * stride) as u64,
                src_len,
            )?;
            let digest = sha256_of(&buffers.source.bytes()[..src_len]);
            let Buffers {
                source, canonical, ..
            } = buffers;
            plan.convert_zero_point_rows(
                *start..*end,
                &source.bytes()[..src_len],
                &mut canonical.bytes_mut()[..unit.canonical_len],
            )?;
            Ok(digest)
        }
        (kind, source) => Err(invalid(format!(
            "tensor '{}': a {source:?} unit does not belong to a {} tensor",
            tensor.role,
            match kind {
                ResolvedKind::Bf16 { .. } => "bf16",
                ResolvedKind::PackQuantized { .. } => "pack-quantized",
            }
        ))),
    }
}

/// Read one range of one of a module's four tensors into the source tile.
fn read_module_range(
    tensor: &Resolved,
    sources: &mut Sources,
    buffers: &mut Buffers,
    suffix: &str,
    offset: u64,
    len: usize,
) -> Result<()> {
    let ResolvedKind::PackQuantized { module, files, .. } = &tensor.kind else {
        return Err(invalid(format!(
            "tensor '{}' is not a pack-quantized module",
            tensor.role
        )));
    };
    let file = files.get(suffix).ok_or_else(|| {
        invalid(format!(
            "tensor '{}': no shard named for {suffix}",
            tensor.role
        ))
    })?;
    if len > buffers.source.bytes().len() {
        return Err(invalid(format!(
            "tensor '{}': a {len}-byte read is above the admitted source tile",
            tensor.role
        )));
    }
    let name = format!("{module}.{suffix}");
    sources.read_range(file, &name, offset, &mut buffers.source.bytes_mut()[..len])
}

fn sha256_of(bytes: &[u8]) -> String {
    let mut h = StreamingSha256::new();
    h.update(bytes);
    h.finalize_hex()
}

/// Re-exported so the workflow can name the plan type without importing the
/// format crate's module path twice.
pub type Plan = PackQuantizedPlan;
