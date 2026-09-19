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
#[derive(Debug)]
pub struct DeviceKvSequence {
    geometry: KvGeometry,
    layout: Vec<DeviceLayerLayout>,
    state: SequenceState,
    /// The committed frontier: rows whose write this authority has published.
    rows: u64,
    /// The highest frontier ever reached, which is what the ring overwrites
    /// against. An aborted transaction lowers `rows` and not this.
    high_water: u64,
    /// The frontier when the open transaction began, so an abort restores it.
    tentative_base: Option<u64>,
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
            high_water: 0,
            tentative_base: None,
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
        // The ring overwrites row `r` with row `r + capacity`. A page is gone
        // once any of its rows has been: the first row still addressable is
        // `high_water - capacity` rounded **up** to a page boundary.
        let overwritten = self.high_water.saturating_sub(layout.capacity_rows);
        let base = overwritten.div_ceil(page_tokens) * page_tokens;
        Ok(base.min(self.rows)..self.rows)
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

    /// The page-aligned runs covering `rows` rows from `first`.
    ///
    /// What a performer copies: one call per stretch of rows that shares a
    /// page. The performer does not compute `position / page_tokens` itself,
    /// which is the whole point — two implementations of one mapping is what
    /// this module exists to prevent.
    ///
    /// Positions **at or beyond** the committed frontier are legal here and
    /// nowhere else: this is what a transaction asks before it writes.
    pub fn placements(&self, layer: usize, first: u64, rows: u64) -> Result<Vec<Placement>> {
        let layout = self.layout(layer)?;
        if rows == 0 {
            return Err(invalid("rows", "a placement of no rows"));
        }
        let end = first
            .checked_add(rows)
            .ok_or(Error::Dim(DimError::Overflow))?;
        if end > self.geometry.max_tokens as u64 {
            return Err(Error::CapacityExceeded {
                tier: None,
                requested_bytes: end,
                available_bytes: self.geometry.max_tokens as u64,
            });
        }
        let page_tokens = self.geometry.page_tokens as u64;
        let mut out = Vec::new();
        out.try_reserve_exact((rows.div_ceil(page_tokens) + 1) as usize)
            .map_err(|_| Error::Dim(DimError::Overflow))?;
        let mut done = 0u64;
        while done < rows {
            let position = first + done;
            let slot = position % page_tokens;
            let run = (page_tokens - slot).min(rows - done);
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

    /// The logical-to-physical page table a launch reads, for the retained
    /// range of one layer.
    ///
    /// Logical page zero is the first retained page, which is what the kernel's
    /// `history_base` names. An empty history has no table and says so.
    pub fn page_table(&self, layer: usize) -> Result<Vec<u32>> {
        let layout = self.layout(layer)?;
        let retained = self.retained(layer)?;
        if retained.is_empty() {
            return Err(invalid("retained", "this layer retains no row"));
        }
        let page_tokens = self.geometry.page_tokens as u64;
        let first = retained.start / page_tokens;
        let last = (retained.end - 1) / page_tokens;
        let mut out = Vec::new();
        out.try_reserve_exact((last - first + 1) as usize)
            .map_err(|_| Error::Dim(DimError::Overflow))?;
        for absolute in first..=last {
            let physical = absolute % layout.pages;
            out.push(u32::try_from(physical).map_err(|_| {
                invalid(
                    "page_table",
                    "a physical page identity exceeds this ABI's u32",
                )
            })?);
        }
        Ok(out)
    }

    /// Open a transaction. The frontier it starts at is what an abort restores.
    pub fn begin(&mut self) -> Result<StateTransactionId> {
        let txn = self.state.begin(ROOT)?;
        self.tentative_base = Some(self.rows);
        Ok(txn)
    }

    /// Publish `rows` rows appended from the current frontier.
    ///
    /// Called **after** a performer has observed its copies complete. Before
    /// that the rows are staged bytes with no reader; after it they are
    /// history. There is no state in between, which is why this is a separate
    /// call from [`Self::placements`] rather than a return value of it.
    pub fn publish(&mut self, txn: StateTransactionId, rows: u64) -> Result<()> {
        self.check_transaction(txn)?;
        if rows == 0 {
            return Err(invalid("rows", "publishing no row"));
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
        // ADR 0014's headroom: a transaction that appends more than the
        // admitted undo headroom would overwrite rows its own abort has to put
        // back. Refused before publishing, so abort never has to fail.
        let base = self.tentative_base.expect("a validated open transaction");
        if self.reclaims() && end - base > self.geometry.tentative_rows as u64 {
            return Err(invalid(
                "tentative_rows",
                "this transaction has appended more rows than the admitted undo headroom",
            ));
        }
        self.state.execute(ROOT, rows)?;
        self.rows = end;
        self.high_water = self.high_water.max(self.rows);
        Ok(())
    }

    /// Accept `accept` of the transaction's rows and close it.
    pub fn commit(&mut self, txn: StateTransactionId, accept: u64) -> Result<()> {
        self.check_transaction(txn)?;
        self.state.commit_prefix(txn, accept)?;
        self.tentative_base = None;
        Ok(())
    }

    /// Discard the open transaction's rows.
    ///
    /// The frontier returns to where the transaction began and the retained
    /// range with it. `high_water` does **not** move: the ring has already
    /// overwritten what it overwrote, and pretending otherwise would hand back
    /// rows whose bytes are gone.
    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        self.check_transaction(txn)?;
        let base = self.tentative_base.take().expect("an open transaction");
        self.state.abort(txn)?;
        self.rows = base;
        Ok(())
    }

    /// Drop every row at or after `prefix`.
    ///
    /// `StateKind::KvPages` is `RestoreCapability::Truncate`, and this is what
    /// that means for device pages: the suffix stops being addressable, and
    /// what remains is exactly what was there before those positions were
    /// written. Refused while a transaction is open — a truncation underneath
    /// one would move the frontier its abort is holding.
    pub fn truncate(&mut self, prefix: u64) -> Result<()> {
        if self.tentative_base.is_some() {
            return Err(invalid(
                "truncate",
                "a transaction is open; commit or abort before truncating",
            ));
        }
        if prefix > self.rows {
            return Err(invalid("prefix", "truncating past the committed frontier"));
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

    fn check_transaction(&self, txn: StateTransactionId) -> Result<()> {
        if self.tentative_base.is_none() {
            return Err(invalid("transaction", "no transaction is open"));
        }
        let open = self.state.open_transactions();
        if !open.iter().any(|(id, _)| *id == txn) {
            return Err(invalid("transaction", "this transaction is not open here"));
        }
        Ok(())
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
            sequence.publish(txn, step).expect("publish");
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
        // Capacity 40 rows in five 8-row pages. At frontier 100 the ring has
        // overwritten rows 0..=59, so row 60 is the first whose bytes survive
        // — and 60 is a page boundary here, so nothing is lost to rounding.
        let mut sequence = DeviceKvSequence::new(windowed(24, 8, 8)).expect("a sequence");
        fill(&mut sequence, 100);
        let retained = sequence.retained(0).expect("a range");
        assert_eq!(retained, 64..100, "{retained:?}");
        assert_eq!(retained.start % 8, 0, "the base must be a whole page");

        // One more row does **not** move the base: the ring has overwritten
        // through row 60, and the first whole page at or above 61 is still the
        // one starting at 64. A base that moved per row would be describing a
        // partially overwritten page.
        fill(&mut sequence, 1);
        assert_eq!(sequence.retained(0).expect("a range"), 64..101);

        // It moves when the page starting at 64 is itself reached: row 104
        // overwrites row 64's slot, so at frontier 109 the base is 72.
        fill(&mut sequence, 8);
        assert_eq!(sequence.retained(0).expect("a range"), 72..109);
        // And the row just below a base is gone rather than wrong.
        assert!(sequence.placement_of(0, 71).is_err());
        sequence
            .placement_of(0, 72)
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
        assert_eq!(retained, 64..100);
        // Rows 64..99 are absolute pages 8..12, and page 8 is physical 8 % 5.
        let table = sequence.page_table(0).expect("a table");
        assert_eq!(table, vec![3, 4, 0, 1, 2]);
        // The kernel's arithmetic, checked against the authority's own: logical
        // page zero is the retained base.
        for position in retained.clone() {
            let logical = (position - retained.start) / 8;
            let physical = u64::from(table[logical as usize]);
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
        fill(&mut sequence, 8);
        // Six rows from position 5: 5..7 in one page, 8..10 in the next.
        let runs = sequence.placements(0, 5, 6).expect("placements");
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
        // A run that starts on a boundary and covers exactly one page is one
        // placement, not two.
        let runs = sequence.placements(0, 8, 8).expect("placements");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].rows, 8);
        assert!(sequence.placements(0, 0, 0).is_err());
        assert!(sequence.placements(1, 0, 1).is_err());
        assert!(sequence.placements(0, 0, 65).is_err());
    }

    #[test]
    fn an_abort_restores_the_frontier_and_the_retained_range() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 10);
        let before = sequence.retained(0).expect("a range");
        let txn = sequence.begin().expect("a transaction");
        sequence.publish(txn, 6).expect("publish");
        assert_eq!(sequence.committed_rows(), 16);
        sequence.abort(txn).expect("abort");
        assert_eq!(sequence.committed_rows(), 10);
        assert_eq!(sequence.retained(0).expect("a range"), before);
        // And the rows it published are not addressable any more.
        assert!(sequence.placement_of(0, 10).is_err());
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
    fn the_undo_headroom_is_refused_before_it_is_overwritten() {
        // ADR 0014: a transaction longer than the admitted headroom would
        // overwrite rows its own abort has to restore. Refused at publication,
        // so abort never has to fail.
        let mut sequence = DeviceKvSequence::new(windowed(16, 8, 4)).expect("a sequence");
        fill(&mut sequence, 40);
        let txn = sequence.begin().expect("a transaction");
        sequence
            .publish(txn, 4)
            .expect("four rows fit the headroom");
        let refused = sequence
            .publish(txn, 1)
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
    }

    #[test]
    fn a_transaction_is_required_and_checked() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        let mut other = DeviceKvSequence::new(full(8, 64)).expect("another sequence");
        let txn = sequence.begin().expect("a transaction");
        let foreign = other.begin().expect("another transaction");
        assert!(
            sequence.publish(foreign, 1).is_err(),
            "another sequence's transaction was accepted"
        );
        sequence.publish(txn, 1).expect("publish");
        sequence.commit(txn, 1).expect("commit");
        assert!(
            sequence.publish(txn, 1).is_err(),
            "a committed transaction was reused"
        );
        assert!(sequence.publish(txn, 0).is_err());
    }
}
