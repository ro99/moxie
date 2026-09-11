//! Appendable host KV pages bound to the accepted sequence transaction journal.
//!
//! This is storage, not an attention executor. All pages belong exclusively to
//! one root branch. A fixed envelope is admitted before use; growth, COW and
//! device views require later capability qualification.

use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};

use moxie_memory::{HostBuffer, Ledger};
use moxie_types::{BranchId, DimError, Error, Precision, Result, StateTransactionId};

use crate::{Branch, Journal, PrefixLineage, ROOT, SequenceState, StateKind};

/// Uniform geometry across layers. K and V may have different widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvGeometry {
    pub layers: usize,
    pub kv_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub precision: Precision,
    pub page_tokens: usize,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    key_bytes: usize,
    value_bytes: usize,
    layer_bytes: usize,
    page_bytes: usize,
    table_bytes: usize,
    backing_bytes: usize,
    control_bytes: usize,
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(DimError::Overflow.into())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(DimError::Overflow.into())
}

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

impl KvGeometry {
    fn layout(self) -> Result<Layout> {
        if [
            self.layers,
            self.kv_heads,
            self.key_dim,
            self.value_dim,
            self.page_tokens,
            self.max_tokens,
        ]
        .contains(&0)
        {
            return Err(invalid("kv_geometry", "all dimensions must be positive"));
        }
        if !self.precision.is_legal_cache() {
            return Err(Error::Unsupported {
                capability: "paged cache precision",
                reason: "only BF16, FP16 and FP32 encoded state is supported".into(),
            });
        }
        let element = self.precision.bits() as usize / 8;
        let key_bytes = mul(mul(self.kv_heads, self.key_dim)?, element)?;
        let value_bytes = mul(mul(self.kv_heads, self.value_dim)?, element)?;
        let layer_bytes = mul(self.page_tokens, add(key_bytes, value_bytes)?)?;
        let page_bytes = mul(self.layers, layer_bytes)?;
        let pages = self.max_tokens.div_ceil(self.page_tokens);
        let table_bytes = mul(pages, size_of::<u64>())?;
        let backing_bytes = add(table_bytes, mul(pages, page_bytes)?)?;
        // The facade permits one branch, one open journal and no retained
        // results. Lineage is reserved exactly once. A conservative node bound
        // for the pinned Rust BTreeMap (11 entries, 12 edges) covers each map's
        // single node, including padding/header, even when the branch has only
        // one entry. No per-token map entries can accumulate here.
        let node_bound = |entry: usize| 11 * (entry + size_of::<usize>()) + 16 * size_of::<usize>();
        let control_bytes = add(
            mul(add(self.max_tokens, 1)?, size_of::<PrefixLineage>())?,
            size_of::<Self>()
                + size_of::<PagedSequence>()
                + size_of::<StateKind>()
                + node_bound(size_of::<(BranchId, Branch)>())
                + node_bound(size_of::<(StateTransactionId, Journal)>()),
        )?;
        if backing_bytes > isize::MAX as usize || control_bytes > isize::MAX as usize {
            return Err(DimError::Overflow.into());
        }
        Ok(Layout {
            key_bytes,
            value_bytes,
            layer_bytes,
            page_bytes,
            table_bytes,
            backing_bytes,
            control_bytes,
        })
    }
}

/// Already encoded, head-major bytes for one layer at one position. The store
/// performs no floating-point conversion, including for NaN payloads or -0.
#[derive(Debug, Clone, Copy)]
pub struct KvRow<'a> {
    pub key: &'a [u8],
    pub value: &'a [u8],
}

/// Logical occupancy versus the full admitted physical envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedUsage {
    pub rows: usize,
    pub live_pages: usize,
    pub live_page_bytes: usize,
    pub logical_kv_bytes: usize,
    pub backing_bytes: usize,
    pub page_table_bytes: usize,
    pub control_reserve_bytes: usize,
}

/// One sequence and its exclusively owned physical KV pages.
///
/// There is no mutable access to the inner `SequenceState`: executed positions
/// can advance only by writing all their physical rows. Transactions and all
/// frontier/lineage rules are delegated to the existing mechanism.
#[derive(Debug)]
pub struct PagedSequence {
    state: SequenceState,
    geometry: KvGeometry,
    layout: Layout,
    backing: HostBuffer,
    rows: usize,
}

/// A refused close retains the entire sequence, including release authority.
#[derive(Debug)]
pub struct PagedCloseRefused {
    pub sequence: PagedSequence,
    pub error: Error,
}

impl PagedSequence {
    pub fn new(ledger: &mut Ledger, geometry: KvGeometry) -> Result<Self> {
        let layout = geometry.layout()?;
        let mut backing = HostBuffer::allocate(
            ledger,
            "paged sequence",
            layout.backing_bytes,
            layout.control_bytes,
        )?;
        let mut state = SequenceState::new([StateKind::KvPages]);
        let lineage = &mut state.branches.get_mut(&ROOT).expect("root exists").lineage;
        let capacity = geometry.max_tokens + 1; // checked by layout
        if lineage.try_reserve_exact(capacity - lineage.len()).is_err()
            || lineage.capacity() != capacity
        {
            drop(state);
            backing.release(ledger).expect("admitting ledger");
            return Err(invalid(
                "lineage",
                "cannot allocate admitted lineage capacity",
            ));
        }
        // Immutable page table. Pages are statically partitioned in this first
        // exclusive-owner implementation; no per-row allocation or dense
        // history materialization is involved in addressing them.
        for page in 0..geometry.max_tokens.div_ceil(geometry.page_tokens) {
            let offset = layout.table_bytes + page * layout.page_bytes;
            backing.bytes_mut()[page * 8..page * 8 + 8]
                .copy_from_slice(&(offset as u64).to_le_bytes());
        }
        Ok(Self {
            state,
            geometry,
            layout,
            backing,
            rows: 0,
        })
    }

    pub fn state(&self) -> &SequenceState {
        &self.state
    }

    pub fn geometry(&self) -> KvGeometry {
        self.geometry
    }

    pub fn usage(&self) -> PagedUsage {
        let live_pages = self.rows.div_ceil(self.geometry.page_tokens);
        PagedUsage {
            rows: self.rows,
            live_pages,
            live_page_bytes: live_pages * self.layout.page_bytes,
            logical_kv_bytes: self.rows
                * self.geometry.layers
                * (self.layout.key_bytes + self.layout.value_bytes),
            backing_bytes: self.layout.backing_bytes,
            page_table_bytes: self.layout.table_bytes,
            control_reserve_bytes: self.layout.control_bytes,
        }
    }

    fn check_frontier(&self, base: u64, n: u64) -> Result<()> {
        let end = base.checked_add(n).ok_or(DimError::Overflow)?;
        if end > self.geometry.max_tokens as u64 {
            return Err(invalid(
                "context",
                "requested prefix exceeds admitted maximum tokens",
            ));
        }
        Ok(())
    }

    pub fn append_prompt(&mut self, n: u64) -> Result<()> {
        self.check_frontier(self.state.frontiers(ROOT)?.accepted, n)?;
        self.state.append_prompt(ROOT, n)
    }

    pub fn accept(&mut self, n: u64) -> Result<()> {
        self.check_frontier(self.state.frontiers(ROOT)?.accepted, n)?;
        self.state.accept(ROOT, n)
    }

    pub fn emit(&mut self, n: u64) -> Result<()> {
        self.state.emit(ROOT, n)
    }

    pub fn begin(&mut self) -> Result<StateTransactionId> {
        self.state.begin(ROOT)
    }

    /// Task 0004 semantics: accept n additional tokens, keep executed work,
    /// close. Verification truncates an unwanted executed suffix explicitly
    /// with `rollback_to` after resolving; a zero commit keeps materialization.
    pub fn commit_prefix(&mut self, txn: StateTransactionId, n: u64) -> Result<()> {
        self.check_transaction(txn)?;
        self.check_frontier(self.state.frontiers(ROOT)?.accepted, n)?;
        self.state.commit_prefix(txn, n)
    }

    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        self.state.abort(txn)?;
        self.truncate(self.state.frontiers(ROOT)?.executed as usize);
        Ok(())
    }

    pub fn rollback_to(&mut self, prefix: u64) -> Result<()> {
        self.state.rollback_to(ROOT, prefix, &[])?;
        self.truncate(self.state.frontiers(ROOT)?.executed as usize);
        Ok(())
    }

    pub fn fork(&mut self, _at: u64) -> Result<BranchId> {
        Err(Error::Unsupported {
            capability: "paged state fork",
            reason: "COW page sharing is not qualified; this pool owns one root branch".into(),
        })
    }

    fn check_transaction(&self, txn: StateTransactionId) -> Result<()> {
        if !self.state.open.contains_key(&txn) {
            return Err(invalid(
                "transaction",
                "no such open transaction on this sequence",
            ));
        }
        Ok(())
    }

    /// Append one complete token across all layers. A failure with a valid ID
    /// aborts the whole transaction, including earlier successful appends.
    /// Foreign/resolved IDs do not resolve or mutate any local transaction.
    pub fn append(
        &mut self,
        txn: StateTransactionId,
        position: u64,
        layers: &[KvRow<'_>],
        cancelled: &AtomicBool,
    ) -> Result<()> {
        self.append_checked(txn, position, layers, || {
            if cancelled.load(Ordering::Relaxed) {
                Err(Error::Cancelled { at: "paged append" })
            } else {
                Ok(())
            }
        })
    }

    // Private harness seam: production and deterministic fault tests traverse
    // the identical publication/error handler. No backend callback is public.
    fn append_checked(
        &mut self,
        txn: StateTransactionId,
        position: u64,
        layers: &[KvRow<'_>],
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.check_transaction(txn)?;
        let result = (|| {
            checkpoint()?;
            if position != self.rows as u64 {
                return Err(invalid(
                    "position",
                    "append must start at the executed frontier",
                ));
            }
            self.check_frontier(position, 1)?;
            if layers.len() != self.geometry.layers
                || layers.iter().any(|r| {
                    r.key.len() != self.layout.key_bytes || r.value.len() != self.layout.value_bytes
                })
            {
                return Err(invalid(
                    "kv_rows",
                    "every layer must supply exactly one complete K/V row",
                ));
            }
            let row = self.rows;
            self.rows += 1;
            for (layer, values) in layers.iter().enumerate() {
                let (key, value) = self.ranges(layer, row);
                self.backing.bytes_mut()[key].copy_from_slice(values.key);
                self.backing.bytes_mut()[value].copy_from_slice(values.value);
                checkpoint()?;
            }
            self.state.execute(ROOT, 1)?;
            checkpoint()?;
            Ok(())
        })();
        if result.is_err() {
            self.abort(txn).expect("the validated open transaction");
        }
        result
    }

    fn ranges(&self, layer: usize, row: usize) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        let page = row / self.geometry.page_tokens;
        let local = row % self.geometry.page_tokens;
        let table = &self.backing.bytes()[page * 8..page * 8 + 8];
        let base = u64::from_le_bytes(table.try_into().expect("eight-byte entry")) as usize
            + layer * self.layout.layer_bytes;
        let key = base + local * self.layout.key_bytes;
        let value = base
            + self.geometry.page_tokens * self.layout.key_bytes
            + local * self.layout.value_bytes;
        (
            key..key + self.layout.key_bytes,
            value..value + self.layout.value_bytes,
        )
    }

    pub fn row(&self, layer: usize, position: u64) -> Result<KvRow<'_>> {
        if layer >= self.geometry.layers || position >= self.rows as u64 {
            return Err(invalid(
                "kv_row",
                "layer or position is outside visible state",
            ));
        }
        let (key, value) = self.ranges(layer, position as usize);
        Ok(KvRow {
            key: &self.backing.bytes()[key],
            value: &self.backing.bytes()[value],
        })
    }

    fn truncate(&mut self, rows: usize) {
        for row in rows..self.rows {
            for layer in 0..self.geometry.layers {
                let (key, value) = self.ranges(layer, row);
                self.backing.bytes_mut()[key].fill(0);
                self.backing.bytes_mut()[value].fill(0);
            }
        }
        self.rows = rows;
    }

    /// Close even when a transaction was abandoned. This storage is entirely
    /// synchronous; there is no device work to drain. Wrong-ledger failure
    /// returns the live sequence so cleanup can be retried.
    pub fn close(mut self, ledger: &mut Ledger) -> std::result::Result<(), Box<PagedCloseRefused>> {
        if let Err(error) = self.backing.release(ledger) {
            return Err(Box::new(PagedCloseRefused {
                sequence: self,
                error,
            }));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_memory::CapacitySnapshot;
    use moxie_types::Scope;

    #[test]
    fn cancellation_at_every_publication_boundary_restores_physical_and_logical_state() {
        let geometry = KvGeometry {
            layers: 3,
            kv_heads: 1,
            key_dim: 2,
            value_dim: 1,
            precision: Precision::Bf16,
            page_tokens: 2,
            max_tokens: 8,
        };
        // Entry, each layer's K/V copy, and frontier publication. Exercise both
        // a partial existing page and the first row on a new page.
        for start in [1, 2, 3] {
            for fail_at in 0..geometry.layers + 2 {
                let mut ledger =
                    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()])
                        .unwrap();
                let mut sequence = PagedSequence::new(&mut ledger, geometry).unwrap();
                let rows = [KvRow {
                    key: &[0x80, 0x7f, 0, 0x80],
                    value: &[0xff, 0xff],
                }; 3];
                let txn = sequence.begin().unwrap();
                sequence.append_prompt(start).unwrap();
                for position in 0..start {
                    sequence
                        .append(txn, position, &rows, &AtomicBool::new(false))
                        .unwrap();
                }
                sequence.commit_prefix(txn, 0).unwrap();
                let bytes = sequence.backing.bytes().to_vec();
                let frontiers = sequence.state.frontiers(ROOT).unwrap();
                let lineage = sequence.state.branches[&ROOT].lineage.clone();
                let charge = ledger.scope_committed(Scope::Host);
                let txn = sequence.begin().unwrap();
                sequence.append_prompt(2).unwrap();
                sequence
                    .append(txn, start, &rows, &AtomicBool::new(false))
                    .unwrap();
                let mut boundary = 0;
                let result = sequence.append_checked(txn, start + 1, &rows, || {
                    let fail = boundary == fail_at;
                    boundary += 1;
                    if fail {
                        Err(Error::Cancelled {
                            at: "injected paged publication",
                        })
                    } else {
                        Ok(())
                    }
                });
                assert!(matches!(result, Err(Error::Cancelled { .. })));
                assert_eq!(sequence.backing.bytes(), bytes);
                assert_eq!(sequence.state.frontiers(ROOT).unwrap(), frontiers);
                assert_eq!(sequence.state.branches[&ROOT].lineage, lineage);
                assert_eq!(sequence.rows, start as usize);
                assert!(sequence.state.open_transactions().is_empty());
                assert_eq!(ledger.scope_committed(Scope::Host), charge);
                let retry = sequence.begin().unwrap();
                sequence
                    .append(retry, start, &rows, &AtomicBool::new(false))
                    .unwrap();
                sequence.commit_prefix(retry, 1).unwrap();
                sequence.close(&mut ledger).unwrap();
                assert_eq!(ledger.scope_committed(Scope::Host), 0);
            }
        }
    }
}
