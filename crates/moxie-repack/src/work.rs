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
use moxie_format::compressed_tensors::PackQuantizedPlan;
use moxie_format::payload::{self, ZeroPointSection};
use moxie_memory::{BufferRequest, HostBuffer, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_types::{HostTier, Result, Scope, Tier};

use crate::source::{Sources, invalid};
use crate::{Budgets, Resolved, ResolvedKind};

/// One bounded unit of work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// Where this unit's bytes start inside the tensor's canonical payload.
    pub canonical_offset: u64,
    pub canonical_len: usize,
    pub source: UnitSource,
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

/// Split one resolved tensor into units no larger than `tile`.
///
/// The three sections are consecutive, in ADR 0023's order, so concatenating
/// every unit's bytes is exactly the canonical payload.
pub fn units_of(tensor: &Resolved, tile: usize) -> Result<Vec<Unit>> {
    let tile = tile.max(1);
    let mut units = Vec::new();
    match &tensor.kind {
        ResolvedKind::Bf16 { len, .. } => {
            let mut at = 0u64;
            while at < *len {
                let take = ((*len - at) as usize).min(tile);
                units.push(Unit {
                    canonical_offset: at,
                    canonical_len: take,
                    source: UnitSource::Bytes {
                        offset: at,
                        len: take,
                    },
                });
                at += take as u64;
            }
        }
        ResolvedKind::PackQuantized { plan, section, .. } => {
            let ext = payload::extents(plan.descriptor(), *section)?;
            let rows = plan.out_features();

            // Codes. The tile has to hold one source row and one canonical
            // row, so the block is bounded by the larger of the two.
            let per_row = plan
                .source_code_row_bytes()
                .max(plan.canonical_code_row_bytes());
            let block = (tile / per_row.max(1)).max(1);
            let mut at = ext.codes().start as u64;
            let mut row = 0;
            while row < rows {
                let end = (row + block).min(rows);
                let len = (end - row) * plan.canonical_code_row_bytes();
                units.push(Unit {
                    canonical_offset: at,
                    canonical_len: len,
                    source: UnitSource::Codes { start: row, end },
                });
                at += len as u64;
                row = end;
            }

            // Scales.
            let per_row = plan.scale_row_bytes().max(1);
            let block = (tile / per_row).max(1);
            let mut at = ext.scales().start as u64;
            let mut row = 0;
            while row < rows {
                let end = (row + block).min(rows);
                let len = (end - row) * per_row;
                units.push(Unit {
                    canonical_offset: at,
                    canonical_len: len,
                    source: UnitSource::Scales { start: row, end },
                });
                at += len as u64;
                row = end;
            }

            // Zero points, in blocks that start on a source word boundary:
            // the lanes of one word are consecutive **output channels**, so a
            // block that began mid-word would need bytes the caller did not
            // read.
            if *section == ZeroPointSection::PerGroup {
                let per_word = plan.zero_points_per_word();
                let per_row = plan
                    .canonical_zero_point_row_bytes()
                    .max(plan.source_zero_point_word_row_bytes())
                    .max(1);
                let block = ((tile / per_row).max(1) / per_word).max(1) * per_word;
                let mut at = ext
                    .zero_points()
                    .expect("an asymmetric tensor has the section")
                    .start as u64;
                let mut row = 0;
                while row < rows {
                    let end = (row + block).min(rows);
                    let len = (end - row) * plan.canonical_zero_point_row_bytes();
                    units.push(Unit {
                        canonical_offset: at,
                        canonical_len: len,
                        source: UnitSource::Zeros { start: row, end },
                    });
                    at += len as u64;
                    row = end;
                }
            }
        }
    }
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
}

impl Buffers {
    /// Admit both tiles. The column scratch starts empty and grows once, to
    /// the widest row any tensor has, when the first conversion needs it.
    pub fn admit(ledger: &mut Ledger, budgets: &Budgets) -> Result<Self> {
        let tile = budgets.tile_bytes();
        let source =
            HostBuffer::allocate_in(ledger, "repack source tile", HostTier::Pageable, tile, 0)?;
        let canonical =
            HostBuffer::allocate_in(ledger, "repack canonical tile", HostTier::Pageable, tile, 0)?;
        Ok(Self {
            source,
            canonical,
            columns: Vec::new(),
            columns_charge: None,
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
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        self.source.release(ledger)?;
        self.canonical.release(ledger)?;
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
