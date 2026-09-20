//! Who owns a device-resident KV page, and what is true about it.
//!
//! [`crate::paged::PagedSequence`] binds this crate's transactions to admitted
//! **host** pages: it owns the decisions *and* the bytes. A device sequence
//! cannot work that way — this crate may not allocate device memory, and the
//! crate that can may not own retention or a frontier. So this module owns
//! exactly the decisions and publishes them as data: where a row is placed,
//! what the committed frontier is, which rows are still retained, and what a
//! page table contains. `moxie-executor` performs them.
//!
//! **It is the same mechanism, not a second one.** Transactions, frontiers and
//! lineage are [`crate::SequenceState`], the type `PagedSequence` uses;
//! retention is [`crate::paged::Retention`]; the geometry is
//! [`crate::paged::KvGeometry`]. What differs is only that the bytes live
//! somewhere this crate cannot address, so their placement is returned rather
//! than written.
//!
//! ## The ring is a page table
//!
//! `PagedSequence::ranges` places row `r` of a layer at
//! `page = (r / page_tokens) % pages`, `slot = r % page_tokens`. That is a page
//! table: logical page `L` is physical page `L % pages`. The device path does
//! not invent a second scheme, it writes that one down, and
//! `device_placement_agrees_with_the_host_ring` asserts the two agree at every
//! row across a wrap.
//!
//! ## Why a device page is evicted whole
//!
//! The host store's retained start is
//! `max(high_water − capacity, rows − window)`. The second term is row-granular
//! and the kernel handles it exactly, by masking on absolute positions. The
//! first is the ring's eviction boundary and is **not** page-aligned: at
//! capacity 64 and high water 100, rows 36..99 are live, and physical page 2
//! then holds rows 96..99 in its first four slots and rows 36..47 in its last
//! twelve. No `history_base` describes a page like that, because a page table
//! addresses whole pages.
//!
//! So a device page leaves the retained range as soon as **any** of its rows
//! would be overwritten, and the retained base is always a whole page. That
//! costs up to `page_tokens − 1` rows, which is why [`DeviceKvSequence::new`]
//! rounds the admitted capacity up by a page. Rows below the base are not
//! wrong, they are *gone*: reading one is [`Error::Reclaimed`], exactly as it is
//! on the host.

use std::ops::Range;

use moxie_types::{DimError, Error, PagePlacement, Result, StateTransactionId};

use crate::paged::{KvGeometry, Retention};
use crate::{ROOT, SequenceState, StateKind};

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

/// One layer's resolved device page layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLayerLayout {
    /// Physical pages admitted for this layer.
    pub pages: u64,
    /// Rows those pages hold: `pages * page_tokens`.
    pub capacity_rows: u64,
    /// Elements in one stored row: every key/value head, side by side.
    pub row_elements: u64,
}

/// One layer's logical-to-physical mapping, and the absolute row its first
/// logical page names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageView {
    pub base: u64,
    pub table: Vec<u32>,
}

/// Where one run of rows physically lands.
///
/// [`moxie_types::PagePlacement`], because it is the vocabulary that passes
/// between this authority and the crate that performs it; naming it twice would
/// be two descriptions of one placement.
pub type Placement = PagePlacement;

/// The authority for one sequence's device-resident KV pages.
///
/// It holds no bytes. Every method answers a question about placement,
/// retention or the frontier, and the answers are the only thing a performer is
/// permitted to act on.
/// Rows a transaction has reserved and not yet published.
///
/// Returned by [`DeviceKvSequence::stage`] and consumed by
/// [`DeviceKvSequence::publish`]. The positions and the count are one value, so
/// a caller cannot write one range and publish another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "staged rows must be published or the transaction aborted"]
pub struct StagedRows {
    transaction: StateTransactionId,
    first: u64,
    rows: u64,
}

impl StagedRows {
    /// The absolute position of the first staged row.
    pub const fn first(&self) -> u64 {
        self.first
    }

    /// How many rows are staged.
    pub const fn rows(&self) -> u64 {
        self.rows
    }
}

/// One open transaction's bookkeeping.
#[derive(Debug, Clone, Copy)]
struct OpenTransaction {
    id: StateTransactionId,
    /// The frontier when it began, which is what an abort restores.
    base: u64,
    /// Rows staged and not yet published. At most one batch at a time: two
    /// outstanding batches would let a caller publish them out of order and
    /// move the frontier over rows nothing wrote.
    pending: Option<(u64, u64)>,
}

#[derive(Debug)]
pub struct DeviceKvSequence {
    geometry: KvGeometry,
    layout: Vec<DeviceLayerLayout>,
    state: SequenceState,
    /// The published frontier, including rows a transaction has published but
    /// not yet committed. Tentative rows are legal state (document 04), and
    /// they are readable and attendable before the commit that accepts them.
    rows: u64,
    /// The frontier as of the last **commit**.
    ///
    /// Not updated by `publish`, and that is the fix for a defect worth naming:
    /// the retained base is derived from this plus the admitted undo headroom,
    /// so it cannot move while a transaction is open. When it was derived from
    /// the published frontier instead, a tentative append advanced the base and
    /// an abort left it advanced — rows the window still admitted became
    /// permanently unreadable because of a transaction that was rolled back.
    committed_high_water: u64,
    open: Option<OpenTransaction>,
}

impl DeviceKvSequence {
    /// Resolve a geometry into device page layouts.
    ///
    /// The admitted capacity is `window + tentative_rows` rounded up to whole
    /// pages **plus one page**, and the extra page is not slack: a page leaves
    /// the retained range as soon as any of its rows would be overwritten, so
    /// without it the oldest row the window still admits could be evicted with
    /// the page it shares. A layer that retains everything is admitted for
    /// `max_tokens` rounded up instead.
    pub fn new(geometry: KvGeometry) -> Result<Self> {
        if geometry.layers.is_empty() {
            return Err(invalid("layers", "a sequence with no layer"));
        }
        if geometry.page_tokens == 0 {
            return Err(invalid("page_tokens", "a page holding no row"));
        }
        if geometry.max_tokens == 0 {
            return Err(invalid("max_tokens", "a sequence admitting no row"));
        }
        // **The cache precision is a contract, not a byte width.** Every path
        // below this one reads two bytes per element as BF16, and an FP16 cache
        // has the same width and a different meaning, so accepting one here
        // would be the silent reinterpretation task 0037 refused by name and
        // this task inherited. It is `Unsupported` rather than invalid: the
        // request is coherent and this slice does not serve it.
        if geometry.precision != moxie_types::Precision::Bf16 {
            return Err(Error::Unsupported {
                capability: "device_kv_precision",
                reason: "device paged state is BF16 in this slice; an FP16 or integer \
                         cache is unsupported rather than reinterpreted"
                    .into(),
            });
        }
        let page_tokens = geometry.page_tokens as u64;
        let mut layout = Vec::new();
        layout
            .try_reserve_exact(geometry.layers.len())
            .map_err(|_| Error::Dim(DimError::Overflow))?;
        for layer in &geometry.layers {
            if layer.kv_heads == 0 || layer.key_dim == 0 || layer.value_dim == 0 {
                return Err(invalid("layer", "a layer with no head or no width"));
            }
            // This slice stores one payload width per row for keys and values
            // alike. A layer whose value width differs from its key width is
            // MLA-shaped, which task 0037 excluded by name and this task does
            // not widen.
            if layer.key_dim != layer.value_dim {
                return Err(Error::Unsupported {
                    capability: "device_kv_value_width",
                    reason: "a value width that differs from the key width is MLA-shaped \
                             and unsupported by this path"
                        .into(),
                });
            }
            if let Retention::Window { window } = layer.retention {
                // A window of zero sees nothing, including the query's own
                // position, and a reclaiming sequence with no undo headroom has
                // a transaction it can never abort. The host geometry refuses
                // both; so does this.
                if window == 0 {
                    return Err(invalid("window", "a sliding window retaining no row"));
                }
                if geometry.tentative_rows == 0 {
                    return Err(invalid(
                        "tentative_rows",
                        "a reclaiming sequence with no undo headroom cannot abort a \
                         transaction it has written",
                    ));
                }
            }
            let rows = match layer.retention {
                Retention::All => geometry.max_tokens as u64,
                Retention::Window { window } => (window as u64)
                    .checked_add(geometry.tentative_rows as u64)
                    .ok_or(Error::Dim(DimError::Overflow))?,
            };
            let mut pages = rows.div_ceil(page_tokens);
            if matches!(layer.retention, Retention::Window { .. }) {
                // The page that whole-page eviction costs.
                pages = pages.checked_add(1).ok_or(Error::Dim(DimError::Overflow))?;
            }
            let capacity_rows = pages
                .checked_mul(page_tokens)
                .ok_or(Error::Dim(DimError::Overflow))?;
            let row_elements = (layer.kv_heads as u64)
                .checked_mul(layer.key_dim as u64)
                .ok_or(Error::Dim(DimError::Overflow))?;
            layout.push(DeviceLayerLayout {
                pages,
                capacity_rows,
                row_elements,
            });
        }
        Ok(Self {
            geometry,
            layout,
            state: SequenceState::new([StateKind::KvPages, StateKind::PositionCounter]),
            rows: 0,
            committed_high_water: 0,
            open: None,
        })
    }

    pub fn geometry(&self) -> &KvGeometry {
        &self.geometry
    }

    pub fn layer_count(&self) -> usize {
        self.layout.len()
    }

    pub fn layout(&self, layer: usize) -> Result<DeviceLayerLayout> {
        self.layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))
    }

    /// The state machine underneath, for a caller that needs its frontiers.
    pub fn state(&self) -> &SequenceState {
        &self.state
    }

    /// Rows this authority has published. **Not** the admitted capacity and not
    /// the retained count; the three are different numbers and a cache that
    /// conflates them claims a context it does not hold.
    pub fn committed_rows(&self) -> u64 {
        self.rows
    }

    /// The rows one layer still holds, as absolute positions.
    ///
    /// The start is always a whole number of pages, because a page table
    /// addresses whole pages. The end is the committed frontier.
    pub fn retained(&self, layer: usize) -> Result<Range<u64>> {
        let layout = self.layout(layer)?;
        let page_tokens = self.geometry.page_tokens as u64;
        // The ring overwrites row `r` with row `r + capacity`, and a page is
        // gone once any of its rows has been. Two things make this the *stable*
        // base rather than the tightest one.
        //
        // It is computed from the **committed** frontier, not the published
        // one, and it adds the admitted undo headroom. So the highest row a
        // transaction could legally write — `committed + tentative_rows` — is
        // already accounted for before the transaction starts, the base cannot
        // move while it is open, and an abort cannot leave it advanced over
        // rows the window still admits. It costs the headroom in retained rows,
        // which is exactly what the headroom was admitted for.
        let reach = self
            .committed_high_water
            .saturating_add(self.headroom())
            .max(self.rows);
        let overwritten = reach.saturating_sub(layout.capacity_rows);
        let base = overwritten.div_ceil(page_tokens) * page_tokens;
        Ok(base.min(self.rows)..self.rows)
    }

    /// The undo headroom a transaction may use, zero when nothing reclaims.
    fn headroom(&self) -> u64 {
        if self.reclaims() {
            self.geometry.tentative_rows as u64
        } else {
            0
        }
    }

    /// Where row `position` of one layer physically sits.
    pub fn placement_of(&self, layer: usize, position: u64) -> Result<Placement> {
        let layout = self.layout(layer)?;
        if position >= self.rows {
            return Err(invalid("position", "position has not been committed"));
        }
        let retained = self.retained(layer)?;
        if position < retained.start {
            return Err(Error::Reclaimed {
                layer: layer as u32,
                position,
                retained_from: retained.start,
            });
        }
        Ok(self.place(layout, position, 1))
    }

    /// The page-aligned runs covering one staged batch, for one layer.
    ///
    /// What a performer copies: one call per stretch of rows that shares a
    /// page. The performer does not compute `position / page_tokens` itself,
    /// which is the whole point — two implementations of one mapping is what
    /// this module exists to prevent.
    ///
    /// The positions come from [`StagedRows`], not from an argument: they are
    /// the authority's to choose and they are always the frontier.
    pub fn placements(&self, staged: &StagedRows, layer: usize) -> Result<Vec<Placement>> {
        let layout = self.layout(layer)?;
        let open = self
            .open
            .ok_or_else(|| invalid("transaction", "no transaction is open"))?;
        if open.id != staged.transaction || open.pending != Some((staged.first, staged.rows)) {
            return Err(invalid(
                "staged",
                "these rows are not the batch this transaction staged",
            ));
        }
        let page_tokens = self.geometry.page_tokens as u64;
        let mut out = Vec::new();
        out.try_reserve_exact((staged.rows.div_ceil(page_tokens) + 1) as usize)
            .map_err(|_| Error::Dim(DimError::Overflow))?;
        let mut done = 0u64;
        while done < staged.rows {
            let position = staged.first + done;
            let slot = position % page_tokens;
            let run = (page_tokens - slot).min(staged.rows - done);
            out.push(self.place(layout, position, run));
            done += run;
        }
        Ok(out)
    }

    fn place(&self, layout: DeviceLayerLayout, position: u64, rows: u64) -> Placement {
        let page_tokens = self.geometry.page_tokens as u64;
        Placement {
            position,
            // The host ring, written as a mapping.
            physical_page: (position / page_tokens) % layout.pages,
            slot: position % page_tokens,
            rows,
        }
    }

    /// The mapping a launch reads: logical page zero is `base`.
    ///
    /// It covers the retained range and any rows this transaction has staged,
    /// because a performer must be able to write where it is about to write.
    pub fn page_view(&self, layer: usize) -> Result<PageView> {
        let layout = self.layout(layer)?;
        let retained = self.retained(layer)?;
        let staged_end = self.open.and_then(|o| o.pending).map_or(0, |(f, n)| f + n);
        let end = self.rows.max(staged_end);
        if end == 0 || retained.start >= end {
            return Err(invalid("retained", "this layer holds no row"));
        }
        let page_tokens = self.geometry.page_tokens as u64;
        let first = retained.start / page_tokens;
        let last = (end - 1) / page_tokens;
        let mut table = Vec::new();
        table
            .try_reserve_exact((last - first + 1) as usize)
            .map_err(|_| Error::Dim(DimError::Overflow))?;
        for absolute in first..=last {
            table.push(u32::try_from(absolute % layout.pages).map_err(|_| {
                invalid(
                    "page_table",
                    "a physical page identity exceeds this ABI's u32",
                )
            })?);
        }
        Ok(PageView {
            base: retained.start,
            table,
        })
    }

    /// Open a transaction. The frontier it starts at is what an abort restores.
    pub fn begin(&mut self) -> Result<StateTransactionId> {
        if self.open.is_some() {
            return Err(invalid("transaction", "a transaction is already open here"));
        }
        let id = self.state.begin(ROOT)?;
        self.open = Some(OpenTransaction {
            id,
            base: self.rows,
            pending: None,
        });
        Ok(id)
    }

    /// Reserve the next `rows` positions for this transaction to write.
    ///
    /// The positions are the authority's to choose, and they are always the
    /// frontier: a dense, ordered append is the only shape paged state has.
    /// What comes back is the only thing [`Self::placements`] will place and
    /// the only thing [`Self::publish`] will publish.
    ///
    /// One batch at a time. Two outstanding batches could be published out of
    /// order, which would move the frontier over rows nothing had written.
    pub fn stage(&mut self, txn: StateTransactionId, rows: u64) -> Result<StagedRows> {
        let open = self.open_transaction(txn)?;
        if open.pending.is_some() {
            return Err(invalid(
                "staged",
                "this transaction already has a staged batch; publish it first",
            ));
        }
        if rows == 0 {
            return Err(invalid("rows", "staging no row"));
        }
        let end = self
            .rows
            .checked_add(rows)
            .ok_or(Error::Dim(DimError::Overflow))?;
        if end > self.geometry.max_tokens as u64 {
            return Err(Error::CapacityExceeded {
                tier: None,
                requested_bytes: end,
                available_bytes: self.geometry.max_tokens as u64,
            });
        }
        // ADR 0014's headroom, checked before anything is written rather than
        // after: a transaction that appended more than the admitted undo
        // headroom would overwrite rows its own abort has to put back.
        if self.reclaims() && end - open.base > self.geometry.tentative_rows as u64 {
            return Err(invalid(
                "tentative_rows",
                "this transaction would append more rows than the admitted undo headroom",
            ));
        }
        let staged = StagedRows {
            transaction: txn,
            first: self.rows,
            rows,
        };
        self.open
            .as_mut()
            .expect("a validated open transaction")
            .pending = Some((staged.first, staged.rows));
        Ok(staged)
    }

    /// Publish a staged batch as history.
    ///
    /// Called **after** a performer has observed its copies complete. Before
    /// that the rows are staged bytes with no reader; after it they are state.
    /// There is no moment in between, which is why this is a separate call from
    /// [`Self::stage`] rather than something that call does for you.
    pub fn publish(&mut self, txn: StateTransactionId, staged: StagedRows) -> Result<()> {
        let open = self.open_transaction(txn)?;
        if staged.transaction != txn || open.pending != Some((staged.first, staged.rows)) {
            return Err(invalid(
                "staged",
                "these rows are not the batch this transaction staged",
            ));
        }
        if staged.first != self.rows {
            return Err(invalid(
                "staged",
                "the staged batch no longer starts at the frontier",
            ));
        }
        self.state.execute(ROOT, staged.rows)?;
        self.rows = staged.first + staged.rows;
        self.open
            .as_mut()
            .expect("a validated open transaction")
            .pending = None;
        Ok(())
    }

    /// Accept `accept` of the transaction's rows and close it.
    ///
    /// An unpublished staged batch is refused: its bytes may or may not have
    /// been written, and committing over that question is how a frontier ends
    /// up ahead of the rows behind it.
    pub fn commit(&mut self, txn: StateTransactionId, accept: u64) -> Result<()> {
        let open = self.open_transaction(txn)?;
        if open.pending.is_some() {
            return Err(invalid(
                "staged",
                "a staged batch is unpublished; publish it or abort the transaction",
            ));
        }
        self.state.commit_prefix(txn, accept)?;
        self.open = None;
        // The ring's reach only moves at a commit. Everything the retained base
        // is derived from is therefore stable for the whole of a transaction.
        self.committed_high_water = self.committed_high_water.max(self.rows);
        Ok(())
    }

    /// Discard the open transaction's rows.
    ///
    /// The frontier returns to where the transaction began **and so does the
    /// retained range**, because the base was never derived from anything a
    /// transaction can move. The bytes those rows overwrote are gone, which is
    /// why the admitted capacity carries the undo headroom: the rows the window
    /// still admits were never in the pages the transaction could reach.
    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        let open = self.open_transaction(txn)?;
        let base = open.base;
        self.state.abort(txn)?;
        self.rows = base;
        self.open = None;
        Ok(())
    }

    /// Drop every row at or after `prefix`.
    ///
    /// `StateKind::KvPages` is `RestoreCapability::Truncate`, and this is what
    /// that means for device pages: the suffix stops being addressable, and
    /// what remains is exactly what was there before those positions were
    /// written. Refused while a transaction is open — a truncation underneath
    /// one would move the frontier its abort is holding.
    ///
    /// Refused **below any layer's retained base**, for the reason the host
    /// store refuses the same rollback: those rows are not there. Accepting it
    /// would leave a frontier pointing at bytes the ring overwrote, and a
    /// re-append from that prefix would report rows reclaimed the moment they
    /// were written.
    pub fn truncate(&mut self, prefix: u64) -> Result<()> {
        if self.open.is_some() {
            return Err(invalid(
                "truncate",
                "a transaction is open; commit or abort before truncating",
            ));
        }
        if prefix > self.rows {
            return Err(invalid("prefix", "truncating past the committed frontier"));
        }
        for layer in 0..self.layout.len() {
            let retained = self.retained(layer)?;
            if prefix < retained.start {
                return Err(Error::Reclaimed {
                    layer: layer as u32,
                    position: prefix,
                    retained_from: retained.start,
                });
            }
        }
        self.state.rollback_to(ROOT, prefix, &[])?;
        self.rows = prefix;
        Ok(())
    }

    fn reclaims(&self) -> bool {
        self.geometry
            .layers
            .iter()
            .any(|l| matches!(l.retention, Retention::Window { .. }))
    }

    /// The open transaction, if it is this one.
    fn open_transaction(&self, txn: StateTransactionId) -> Result<OpenTransaction> {
        let open = self
            .open
            .ok_or_else(|| invalid("transaction", "no transaction is open"))?;
        if open.id != txn
            || !self
                .state
                .open_transactions()
                .iter()
                .any(|(id, _)| *id == txn)
        {
            return Err(invalid("transaction", "this transaction is not open here"));
        }
        Ok(open)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged::LayerKv;
    use moxie_types::Precision;

    fn windowed(window: usize, page_tokens: usize, tentative: usize) -> KvGeometry {
        KvGeometry {
            layers: vec![LayerKv {
                kv_heads: 2,
                key_dim: 64,
                value_dim: 64,
                retention: Retention::Window { window },
            }],
            precision: Precision::Bf16,
            page_tokens,
            max_tokens: 4096,
            tentative_rows: tentative,
        }
    }

    fn full(page_tokens: usize, max_tokens: usize) -> KvGeometry {
        KvGeometry {
            layers: vec![LayerKv {
                kv_heads: 2,
                key_dim: 64,
                value_dim: 64,
                retention: Retention::All,
            }],
            precision: Precision::Bf16,
            page_tokens,
            max_tokens,
            tentative_rows: max_tokens,
        }
    }

    /// Append `rows` rows through committed transactions.
    ///
    /// In chunks no larger than the admitted undo headroom, because that is the
    /// contract: a reclaiming sequence refuses a transaction longer than what
    /// its abort could put back (ADR 0014). A helper that ignored it would be
    /// testing a sequence nobody can drive.
    fn fill(sequence: &mut DeviceKvSequence, rows: u64) {
        let headroom = if sequence.reclaims() {
            sequence.geometry.tentative_rows as u64
        } else {
            rows.max(1)
        };
        let mut done = 0;
        while done < rows {
            let step = headroom.min(rows - done);
            let txn = sequence.begin().expect("a transaction");
            let staged = sequence.stage(txn, step).expect("stage");
            // A performer would write these placements here.
            let _ = sequence.placements(&staged, 0).expect("placements");
            sequence.publish(txn, staged).expect("publish");
            sequence.commit(txn, step).expect("commit");
            done += step;
        }
    }

    #[test]
    fn device_placement_agrees_with_the_host_ring() {
        // The claim this module rests on: the host store's ring and this page
        // table are one mapping. `PagedSequence::ranges` computes
        // `(row / page_tokens) % pages`; if these two ever disagree, a device
        // launch reads a different row than the host reference wrote.
        let page_tokens = 8u64;
        let geometry = windowed(24, page_tokens as usize, 8);
        let mut sequence = DeviceKvSequence::new(geometry).expect("a sequence");
        let layout = sequence.layout(0).expect("one layer");
        // 24 + 8 = 32 rows, four pages, plus the eviction page: five.
        assert_eq!(layout.pages, 5);
        assert_eq!(layout.capacity_rows, 40);

        fill(&mut sequence, 200);
        for position in 0..200u64 {
            let host_page = (position / page_tokens) % layout.pages;
            let host_slot = position % page_tokens;
            let placement = sequence.place(layout, position, 1);
            assert_eq!(
                (placement.physical_page, placement.slot),
                (host_page, host_slot),
                "row {position} is placed differently by the two mappings"
            );
        }
    }

    #[test]
    fn a_page_leaves_the_retained_range_whole() {
        // Capacity is 40 rows in five 8-row pages, and the base is derived
        // from the committed frontier **plus the admitted undo headroom** —
        // 108 — so the ring's reach is 68 and the first whole page at or above
        // it is 72. Pricing the headroom in is what keeps the base from moving
        // while a transaction is open, and it costs exactly the headroom.
        let mut sequence = DeviceKvSequence::new(windowed(24, 8, 8)).expect("a sequence");
        fill(&mut sequence, 100);
        let retained = sequence.retained(0).expect("a range");
        assert_eq!(retained, 72..100, "{retained:?}");
        assert_eq!(retained.start % 8, 0, "the base must be a whole page");
        assert!(
            retained.start <= 100 - 24,
            "the window admits row 76 and the base must be at or below it"
        );

        // One more row does **not** move the base: the reach is 109, the
        // ring has overwritten through row 68, and the first whole page at or
        // above that is still 72. A base that moved per row would be
        // describing a partially overwritten page.
        fill(&mut sequence, 1);
        assert_eq!(sequence.retained(0).expect("a range"), 72..101);

        // It moves when the page starting at 72 is itself reached.
        fill(&mut sequence, 8);
        assert_eq!(sequence.retained(0).expect("a range"), 80..109);
        // And the row just below a base is gone rather than wrong.
        assert!(sequence.placement_of(0, 79).is_err());
        sequence
            .placement_of(0, 80)
            .expect("the first retained row");
    }

    #[test]
    fn whole_page_eviction_never_drops_a_row_the_window_admits() {
        // The reason the admitted capacity rounds up by a page. At every
        // frontier, the oldest row the window still admits must still be
        // addressable — that is the property the extra page buys, and it is
        // checked at every step rather than argued.
        for window in [1usize, 7, 8, 9, 24, 31] {
            for page_tokens in [4usize, 8, 16] {
                let mut sequence =
                    DeviceKvSequence::new(windowed(window, page_tokens, 4)).expect("a sequence");
                for _ in 0..300 {
                    fill(&mut sequence, 1);
                    let rows = sequence.committed_rows();
                    let retained = sequence.retained(0).expect("a range");
                    let oldest_visible = rows.saturating_sub(window as u64);
                    assert!(
                        retained.start <= oldest_visible,
                        "window {window} pages of {page_tokens}: at frontier {rows} the \
                         window admits row {oldest_visible} and the retained range starts \
                         at {}",
                        retained.start
                    );
                }
            }
        }
    }

    #[test]
    fn a_retaining_layer_is_admitted_for_its_whole_context() {
        // No eviction page: nothing is ever overwritten, so nothing is ever
        // lost to rounding a page away.
        let sequence = DeviceKvSequence::new(full(16, 1024)).expect("a sequence");
        let layout = sequence.layout(0).expect("one layer");
        assert_eq!(layout.pages, 64);
        assert_eq!(layout.capacity_rows, 1024);
        let mut sequence = sequence;
        fill(&mut sequence, 1024);
        assert_eq!(sequence.retained(0).expect("a range"), 0..1024);
    }

    #[test]
    fn the_page_table_starts_at_the_retained_base() {
        let mut sequence = DeviceKvSequence::new(windowed(24, 8, 8)).expect("a sequence");
        fill(&mut sequence, 100);
        let retained = sequence.retained(0).expect("a range");
        assert_eq!(retained, 72..100);
        // Rows 72..99 are absolute pages 9..12, and page 9 is physical 9 % 5.
        let view = sequence.page_view(0).expect("a view");
        assert_eq!(view.base, retained.start);
        assert_eq!(view.table, vec![4, 0, 1, 2]);
        for position in retained.clone() {
            let logical = (position - view.base) / 8;
            let physical = u64::from(view.table[logical as usize]);
            assert_eq!(
                physical,
                sequence
                    .placement_of(0, position)
                    .expect("a placement")
                    .physical_page,
                "row {position}"
            );
        }
    }

    #[test]
    fn placements_break_at_page_boundaries_and_nowhere_else() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 5);
        // Six rows from the frontier at 5: 5..7 in one page, 8..10 in the next.
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 6).expect("stage");
        assert_eq!((staged.first(), staged.rows()), (5, 6));
        let runs = sequence.placements(&staged, 0).expect("placements");
        assert_eq!(runs.len(), 2);
        assert_eq!(
            runs[0],
            Placement {
                position: 5,
                physical_page: 0,
                slot: 5,
                rows: 3
            }
        );
        assert_eq!(
            runs[1],
            Placement {
                position: 8,
                physical_page: 1,
                slot: 0,
                rows: 3
            }
        );
        // A second batch before the first is published is refused: two
        // outstanding batches could be published out of order, moving the
        // frontier over rows nothing wrote.
        assert!(sequence.stage(txn, 1).is_err());
        // So is committing over an unpublished batch.
        assert!(sequence.commit(txn, 6).is_err());
        sequence.publish(txn, staged).expect("publish");
        // And publishing the same batch twice.
        assert!(sequence.publish(txn, staged).is_err());
        sequence.commit(txn, 6).expect("commit");

        // A run that starts on a boundary and covers exactly one page is one
        // placement, not two.
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 5).expect("stage");
        let runs = sequence.placements(&staged, 0).expect("placements");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].rows, 5);
        assert!(sequence.placements(&staged, 1).is_err(), "no such layer");
        sequence.abort(txn).expect("abort");
        // Staged rows from an aborted transaction place nothing.
        assert!(sequence.placements(&staged, 0).is_err());
        assert!(sequence.stage(txn, 1).is_err(), "the transaction is closed");
    }

    #[test]
    fn the_page_view_covers_rows_a_transaction_has_staged() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 8);
        assert_eq!(sequence.page_view(0).expect("a view").table.len(), 1);
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 8).expect("stage");
        let view = sequence.page_view(0).expect("a view");
        assert_eq!(view.base, 0);
        assert_eq!(view.table.len(), 2, "the staged page is not in the view");
        for placement in sequence.placements(&staged, 0).expect("placements") {
            let logical = (placement.position - view.base) / 8;
            assert_eq!(
                u64::from(view.table[logical as usize]),
                placement.physical_page
            );
        }
        sequence.abort(txn).expect("abort");
        assert_eq!(sequence.page_view(0).expect("a view").table.len(), 1);
    }

    #[test]
    fn a_staged_batch_cannot_name_positions_the_authority_did_not_choose() {
        // Staging starts at the frontier, and only a batch this transaction
        // staged can be placed or published.
        let mut sequence = DeviceKvSequence::new(full(8, 128)).expect("a sequence");
        fill(&mut sequence, 3);
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 2).expect("stage");
        assert_eq!(staged.first(), 3, "staging starts at the frontier");

        // A batch from another sequence's transaction, and one this transaction
        // did not stage, are both refused.
        let mut other = DeviceKvSequence::new(full(8, 128)).expect("another sequence");
        let other_txn = other.begin().expect("a transaction");
        let foreign = other.stage(other_txn, 2).expect("stage");
        assert!(sequence.placements(&foreign, 0).is_err());
        assert!(sequence.publish(txn, foreign).is_err());
        sequence.publish(txn, staged).expect("publish");
        sequence.commit(txn, 2).expect("commit");
        assert_eq!(sequence.committed_rows(), 5);
        other.abort(other_txn).expect("abort");
    }

    #[test]
    fn an_abort_restores_the_frontier_and_the_retained_range() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 10);
        let before = sequence.retained(0).expect("a range");
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 6).expect("stage");
        sequence.publish(txn, staged).expect("publish");
        assert_eq!(sequence.committed_rows(), 16);
        // Tentative rows are legal state and are addressable before the commit
        // that accepts them: document 04 forbids publishing candidates before
        // verification, so they have to be readable without it.
        sequence
            .placement_of(0, 15)
            .expect("a tentative row is placed");
        sequence.abort(txn).expect("abort");
        assert_eq!(sequence.committed_rows(), 10);
        assert_eq!(sequence.retained(0).expect("a range"), before);
        // And the rows it published are not addressable any more.
        assert!(sequence.placement_of(0, 10).is_err());
    }

    #[test]
    fn a_windowed_abort_leaves_the_retained_base_where_it_found_it() {
        // The retained base is derived from the committed frontier plus the
        // admitted headroom, so no transaction can move it.
        let mut sequence = DeviceKvSequence::new(windowed(24, 8, 8)).expect("a sequence");
        fill(&mut sequence, 100);
        let before = sequence.retained(0).expect("a range");
        assert_eq!(before, 72..100, "the headroom is priced into the base");

        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 8).expect("stage the whole headroom");
        sequence.publish(txn, staged).expect("publish");
        assert_eq!(
            sequence.retained(0).expect("a range").start,
            before.start,
            "a tentative append moved the retained base"
        );
        sequence.abort(txn).expect("abort");
        assert_eq!(
            sequence.retained(0).expect("a range"),
            before,
            "an abort left the retained base advanced"
        );
        // And the oldest row the window admits is still placeable, which is the
        // property the whole arrangement exists for.
        let oldest = sequence.committed_rows() - 24;
        sequence
            .placement_of(0, oldest)
            .expect("the oldest row the window admits");
    }

    #[test]
    fn truncation_leaves_exactly_the_earlier_prefix() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 20);
        let at_twelve: Vec<Placement> = (0..12)
            .map(|p| sequence.placement_of(0, p).expect("a placement"))
            .collect();
        sequence.truncate(12).expect("truncate");
        assert_eq!(sequence.committed_rows(), 12);
        assert!(sequence.placement_of(0, 12).is_err());
        let after: Vec<Placement> = (0..12)
            .map(|p| sequence.placement_of(0, p).expect("a placement"))
            .collect();
        assert_eq!(
            at_twelve, after,
            "truncation moved rows it was supposed to leave alone"
        );
        // Re-appending lands where the truncated rows were: the ring is a
        // function of the position, not of how often it has been written.
        fill(&mut sequence, 4);
        assert_eq!(
            sequence.placement_of(0, 12).expect("a placement"),
            Placement {
                position: 12,
                physical_page: 1,
                slot: 4,
                rows: 1
            }
        );
        // A truncation past the frontier, and one under an open transaction,
        // are both refused.
        assert!(sequence.truncate(100).is_err());
        let txn = sequence.begin().expect("a transaction");
        assert!(sequence.truncate(1).is_err());
        sequence.abort(txn).expect("abort");
    }

    #[test]
    fn truncating_below_the_retained_base_is_refused() {
        // The host store refuses a rollback whose target has been reclaimed,
        // and for the same reason: those rows are not there. Accepting it would
        // leave a frontier pointing at bytes the ring overwrote, and a
        // re-append from that prefix would report rows reclaimed the moment
        // they were written.
        let mut sequence = DeviceKvSequence::new(windowed(24, 8, 8)).expect("a sequence");
        fill(&mut sequence, 100);
        let retained = sequence.retained(0).expect("a range");
        assert_eq!(retained.start, 72);

        let refused = sequence
            .truncate(retained.start - 1)
            .expect_err("a truncation below the retained base was accepted");
        assert!(matches!(refused, Error::Reclaimed { .. }), "{refused:?}");
        assert_eq!(
            sequence.committed_rows(),
            100,
            "a refusal moved the frontier"
        );

        // The base itself is a legal target, and what remains after it is
        // exactly what was there.
        let at_base = sequence
            .placement_of(0, retained.start)
            .expect("the first retained row");
        sequence
            .truncate(retained.start)
            .expect("truncate to the base");
        assert_eq!(sequence.committed_rows(), retained.start);
        assert_eq!(
            sequence
                .placement_of(0, retained.start - 1)
                .expect_err("below the base is still gone")
                .kind(),
            "reclaimed"
        );
        // Re-appending from there lands where those rows were.
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 1).expect("stage");
        sequence.publish(txn, staged).expect("publish");
        sequence.commit(txn, 1).expect("commit");
        assert_eq!(
            sequence
                .placement_of(0, retained.start)
                .expect("the re-appended row"),
            at_base,
            "a re-append landed somewhere else"
        );
    }

    #[test]
    fn the_undo_headroom_is_refused_before_it_is_overwritten() {
        // ADR 0014: a transaction longer than the admitted headroom would
        // overwrite rows its own abort has to restore. Refused at **staging**,
        // before a performer is given anywhere to write, so nothing is
        // overwritten and abort never has to fail.
        let mut sequence = DeviceKvSequence::new(windowed(16, 8, 4)).expect("a sequence");
        fill(&mut sequence, 40);
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 4).expect("four rows fit the headroom");
        sequence.publish(txn, staged).expect("publish");
        let refused = sequence
            .stage(txn, 1)
            .expect_err("a fifth row exceeds the headroom");
        assert!(
            matches!(
                refused,
                Error::InvalidRequest {
                    field: "tentative_rows",
                    ..
                }
            ),
            "{refused:?}"
        );
        // And the transaction is still abortable, which is the point.
        sequence.abort(txn).expect("abort");
        assert_eq!(sequence.committed_rows(), 40);
    }

    #[test]
    fn a_reclaimed_row_is_a_typed_refusal_rather_than_another_rows_bytes() {
        let mut sequence = DeviceKvSequence::new(windowed(24, 8, 8)).expect("a sequence");
        fill(&mut sequence, 100);
        let retained = sequence.retained(0).expect("a range");
        let error = sequence
            .placement_of(0, retained.start - 1)
            .expect_err("a reclaimed row must not be placed");
        assert!(matches!(error, Error::Reclaimed { .. }), "{error:?}");
        assert!(sequence.placement_of(0, 100).is_err(), "past the frontier");
        sequence
            .placement_of(0, retained.start)
            .expect("the first retained row");
    }

    #[test]
    fn a_malformed_geometry_is_refused_at_construction() {
        let mut no_layers = full(8, 64);
        no_layers.layers.clear();
        assert!(DeviceKvSequence::new(no_layers).is_err());
        let mut no_page = full(0, 64);
        no_page.page_tokens = 0;
        assert!(DeviceKvSequence::new(no_page).is_err());
        let mut mla = full(8, 64);
        mla.layers[0].value_dim = 32;
        assert!(matches!(
            DeviceKvSequence::new(mla).unwrap_err(),
            Error::Unsupported {
                capability: "device_kv_value_width",
                ..
            }
        ));
        let mut empty = full(8, 64);
        empty.layers[0].kv_heads = 0;
        assert!(DeviceKvSequence::new(empty).is_err());

        // The cache precision is a contract. Everything below reads two bytes
        // per element as BF16, and an FP16 cache has the same width and a
        // different meaning, so it is refused rather than reinterpreted.
        for precision in [
            moxie_types::Precision::F16,
            moxie_types::Precision::F32,
            moxie_types::Precision::Int8,
        ] {
            let mut other = full(8, 64);
            other.precision = precision;
            assert!(
                matches!(
                    DeviceKvSequence::new(other).unwrap_err(),
                    Error::Unsupported {
                        capability: "device_kv_precision",
                        ..
                    }
                ),
                "{precision:?} was accepted as a device cache precision"
            );
        }

        // A window retaining nothing, and a reclaiming sequence with no undo
        // headroom, are both refused: the first sees nothing at all, the second
        // has a transaction it can never abort.
        let mut no_window = windowed(0, 8, 4);
        no_window.layers[0].retention = Retention::Window { window: 0 };
        assert!(DeviceKvSequence::new(no_window).is_err());
        assert!(DeviceKvSequence::new(windowed(8, 8, 0)).is_err());
    }

    #[test]
    fn a_transaction_is_required_and_checked() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        let mut other = DeviceKvSequence::new(full(8, 64)).expect("another sequence");
        let txn = sequence.begin().expect("a transaction");
        let foreign_txn = other.begin().expect("another transaction");
        assert!(
            sequence.stage(foreign_txn, 1).is_err(),
            "another sequence's transaction was accepted"
        );
        assert!(
            sequence.begin().is_err(),
            "a second transaction opened over an open one"
        );
        let staged = sequence.stage(txn, 1).expect("stage");
        sequence.publish(txn, staged).expect("publish");
        sequence.commit(txn, 1).expect("commit");
        other.abort(foreign_txn).expect("abort");
        assert!(
            sequence.stage(txn, 1).is_err(),
            "a committed transaction was reused"
        );
        let txn = sequence.begin().expect("a transaction");
        assert!(sequence.stage(txn, 0).is_err(), "a batch of no rows");
        sequence.abort(txn).expect("abort");
    }
}
