//! Who owns a device-resident KV page, and what is true about it.
//!
//! [`crate::paged::PagedSequence`] binds this crate's transactions to admitted
//! **host** pages: it owns the decisions *and* the bytes. A device sequence
//! cannot work that way — this crate may not allocate device memory, and the
//! crate that can may not own retention or a frontier. So this module owns
//! exactly the decisions and publishes them as data: where a row is placed,
//! what the frontier is, which rows are still retained, and what a page table
//! contains. `moxie-executor` performs them, through [`PagedKvWriter`].
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

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{
    BatchId, BranchId, DimError, Error, HostTier, PagePlacement, PagedKvWriter, Result,
    StateTransactionId, Tier,
};

use crate::paged::{KvGeometry, Retention};
use crate::{ROOT, SequenceState, StateKind};

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: moxie_memory::fallible::text(format_args!("{detail}")).unwrap_or_default(),
    }
}

fn unsupported(capability: &'static str, reason: &'static str) -> Error {
    Error::Unsupported {
        capability,
        reason: moxie_memory::fallible::text(format_args!("{reason}")).unwrap_or_default(),
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

/// One layer's logical-to-physical page mapping, and the absolute row its
/// first logical page names.
///
/// Lives in `moxie-types`, beside [`PagePlacement`]: it is the vocabulary
/// that passes between the authority that decides a page mapping and the
/// performer that uploads it. Re-exported here so existing callers keep
/// working.
pub use moxie_types::PageView;

/// Rows a transaction has reserved and not yet published.
///
/// Private: [`DeviceKvSequence::append`] is the only producer and the only
/// consumer, both inside this module. Nothing outside it stages a batch
/// without also being the authority that publishes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "staged rows must be published or the transaction aborted"]
struct StagedRows {
    transaction: StateTransactionId,
    first: u64,
    rows: u64,
}

impl StagedRows {
    /// The identity [`DeviceKvSequence::append`] passes to
    /// [`PagedKvWriter::write_layer`] for this batch.
    const fn id(&self) -> BatchId {
        BatchId {
            transaction: self.transaction,
            first: self.first,
            rows: self.rows,
        }
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

/// The physical decisions for one logical branch. The root uses the same
/// shape as children; keeping the bookkeeping together prevents branch paths
/// from acquiring a second retention or transaction rule.
#[derive(Debug)]
struct DeviceBranchStorage {
    rows: u64,
    committed_high_water: u64,
    open: Option<OpenTransaction>,
    completed_layers: Vec<bool>,
    poisoned: bool,
    /// The parent's retained floor at fork time. A child cannot resurrect a
    /// device row the parent had already reclaimed when it was forked.
    retained_floor: Vec<u64>,
}

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
    /// Root and child branch decisions. Device bytes remain in executor runs.
    branches: BTreeMap<BranchId, DeviceBranchStorage>,
    /// This sequence's identity and a count every mutation advances, so a
    /// [`PreparedCommit`] can prove it was prepared against exactly this state.
    id: u64,
    generation: u64,
}

/// A root commit with every host-side refusal already taken: validation and
/// each page view the commit will publish. It changes nothing until
/// [`DeviceKvSequence::apply_commit`], which refuses it once the sequence it
/// was prepared on has changed, or on any other sequence.
#[derive(Debug)]
#[must_use = "a prepared commit changes nothing until it is applied"]
pub struct PreparedCommit {
    sequence: u64,
    generation: u64,
    txn: StateTransactionId,
    accept: u64,
    watermark_after: u64,
    updates: Vec<(usize, PageView)>,
}

/// A mutable view of one child branch's device-state decisions.
#[derive(Debug)]
pub struct DeviceBranch<'a> {
    sequence: &'a mut DeviceKvSequence,
    branch: BranchId,
}

impl DeviceKvSequence {
    /// Host bytes reserved for this device sequence's root lineage and its
    /// one possible open transaction-map node.
    pub fn root_host_metadata_bytes(lineage_entries: u64) -> Result<u64> {
        SequenceState::root_host_metadata_bytes(lineage_entries)
    }

    /// Host bytes reserved for one device branch fork.
    ///
    /// `lineage_entries` is the admitted maximum position count plus one. The
    /// charge includes the two per-layer vectors, `SequenceState`'s reserved
    /// lineage, its branch-map node and one possible transaction-map node,
    /// plus this device authority's branch-map node. Each map charge is a full
    /// internal-node upper bound.
    pub fn fork_host_metadata_bytes(lineage_entries: u64, layers: usize) -> Result<u64> {
        if lineage_entries == 0 || layers == 0 {
            return Err(invalid("fork_metadata", "a fork needs lineage and layers"));
        }
        let layers = u64::try_from(layers).map_err(|_| Error::Dim(DimError::Overflow))?;
        let per_layer = (core::mem::size_of::<u64>() as u64)
            .checked_add(core::mem::size_of::<bool>() as u64)
            .ok_or(Error::Dim(DimError::Overflow))?;
        let vectors = layers
            .checked_mul(per_layer)
            .ok_or(Error::Dim(DimError::Overflow))?;
        let logical = SequenceState::fork_host_metadata_bytes(lineage_entries)?;
        let device_node = super::btree_node_host_bytes::<BranchId, DeviceBranchStorage>()?;
        vectors
            .checked_add(logical)
            .and_then(|bytes| bytes.checked_add(device_node))
            .ok_or(Error::Dim(DimError::Overflow))
    }

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
            return Err(unsupported(
                "device_kv_precision",
                "device paged state is BF16 in this slice; an FP16 or integer cache is \
                 unsupported rather than reinterpreted",
            ));
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
                return Err(unsupported(
                    "device_kv_value_width",
                    "a value width that differs from the key width is MLA-shaped and \
                     unsupported by this path",
                ));
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
        let mut completed_layers = Vec::new();
        completed_layers
            .try_reserve_exact(layout.len())
            .map_err(|_| Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: layout.len() as u64,
                available_bytes: 0,
            })?;
        completed_layers.resize(layout.len(), false);
        let mut branches = BTreeMap::new();
        branches.insert(
            ROOT,
            DeviceBranchStorage {
                rows: 0,
                committed_high_water: 0,
                open: None,
                completed_layers,
                poisoned: false,
                retained_floor: vec![0; layout.len()],
            },
        );
        let lineage_capacity = geometry
            .max_tokens
            .checked_add(1)
            .ok_or(Error::Dim(DimError::Overflow))?;
        let mut state = SequenceState::new([StateKind::KvPages, StateKind::PositionCounter]);
        state.reserve_lineage_to(ROOT, lineage_capacity)?;
        Ok(Self {
            geometry,
            layout,
            state,
            branches,
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(0);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            generation: 0,
        })
    }

    /// Refuse if a previous commit's mapping transition failed partway.
    ///
    /// The device may already hold a page view no state here can vouch for —
    /// a prior layer in that commit's writer order may have published
    /// successfully before a later one refused — and there is no way to take
    /// that back. Every public method calls this first, so the authority
    /// stops trusting itself rather than guess at a mapping that might
    /// already be inconsistent.
    fn branch_storage(&self, branch: BranchId) -> Result<&DeviceBranchStorage> {
        self.branches.get(&branch).ok_or_else(|| {
            invalid(
                "branch",
                "the device sequence has no bookkeeping for this branch",
            )
        })
    }

    fn branch_storage_mut(&mut self, branch: BranchId) -> Result<&mut DeviceBranchStorage> {
        // Every mutation of branch bookkeeping comes through here.
        self.generation += 1;
        self.branches.get_mut(&branch).ok_or_else(|| {
            invalid(
                "branch",
                "the device sequence has no bookkeeping for this branch",
            )
        })
    }

    fn check_poisoned_on(&self, branch: BranchId) -> Result<()> {
        if self.branch_storage(branch)?.poisoned {
            return Err(invalid(
                "sequence",
                "a previous commit's mapping transition failed; this sequence can no longer \
                 be trusted",
            ));
        }
        Ok(())
    }

    fn check_poisoned(&self) -> Result<()> {
        self.check_poisoned_on(ROOT)
    }

    pub fn geometry(&self) -> Result<&KvGeometry> {
        self.check_poisoned()?;
        Ok(&self.geometry)
    }

    pub fn layer_count(&self) -> Result<usize> {
        self.check_poisoned()?;
        Ok(self.layout.len())
    }

    pub fn layout(&self, layer: usize) -> Result<DeviceLayerLayout> {
        self.check_poisoned()?;
        self.layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))
    }

    /// The state machine underneath, for a caller that needs its frontiers.
    pub fn state(&self) -> Result<&SequenceState> {
        self.check_poisoned()?;
        Ok(&self.state)
    }

    /// Rows this authority has published, **including** any a transaction has
    /// published but not yet committed. Not the admitted capacity, not the
    /// retained count and not [`Self::committed_rows`]; conflating any of
    /// these claims a context this authority does not hold.
    pub fn published_rows(&self) -> Result<u64> {
        self.published_rows_for(ROOT)
    }

    fn published_rows_for(&self, branch: BranchId) -> Result<u64> {
        self.check_poisoned_on(branch)?;
        Ok(self.branch_storage(branch)?.rows)
    }

    /// Rows published for one layer. A completed layer can be one staged batch
    /// ahead of the sequence frontier while later layers execute.
    pub fn layer_published_rows(&self, layer: usize) -> Result<u64> {
        self.layer_published_rows_for(ROOT, layer)
    }

    fn layer_published_rows_for(&self, branch: BranchId, layer: usize) -> Result<u64> {
        self.check_poisoned_on(branch)?;
        self.layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))?;
        let storage = self.branch_storage(branch)?;
        if storage.completed_layers[layer]
            && let Some((first, rows)) = storage.open.and_then(|open| open.pending)
        {
            return first
                .checked_add(rows)
                .ok_or(Error::Dim(DimError::Overflow));
        }
        Ok(storage.rows)
    }

    /// Rows a commit has accepted, from [`SequenceState`]'s own accepted
    /// count. Unlike [`Self::published_rows`], a tentative append does not
    /// move this until [`Self::commit`] accepts it, and an abort leaves it
    /// exactly where it was — `commit`'s `accept` argument is this counter's
    /// only writer, bounded to what the committing transaction published.
    ///
    /// Not the private retention watermark [`Self::retained`] uses either:
    /// that one moves on every commit regardless of `accept`, because it
    /// tracks physical overwrite, not acceptance.
    pub fn committed_rows(&self) -> Result<u64> {
        self.committed_rows_for(ROOT)
    }

    fn committed_rows_for(&self, branch: BranchId) -> Result<u64> {
        self.check_poisoned_on(branch)?;
        Ok(self.state.frontiers(branch)?.accepted)
    }

    /// The rows one layer still holds, as absolute positions.
    ///
    /// The start is always a whole number of pages, because a page table
    /// addresses whole pages. The end is the published frontier: tentative
    /// rows are retained too, because the window that would evict them has
    /// not moved past them yet.
    pub fn retained(&self, layer: usize) -> Result<Range<u64>> {
        self.retained_for(ROOT, layer)
    }

    fn retained_for(&self, branch: BranchId, layer: usize) -> Result<Range<u64>> {
        self.check_poisoned_on(branch)?;
        let storage = self.branch_storage(branch)?;
        self.retained_at(branch, layer, storage.committed_high_water, storage.rows)
    }

    /// Retained rows currently published for one layer, including its staged
    /// batch during an open transaction.
    pub fn layer_retained(&self, layer: usize) -> Result<Range<u64>> {
        self.layer_retained_for(ROOT, layer)
    }

    fn layer_retained_for(&self, branch: BranchId, layer: usize) -> Result<Range<u64>> {
        let rows = self.layer_published_rows_for(branch, layer)?;
        let storage = self.branch_storage(branch)?;
        self.retained_at(branch, layer, storage.committed_high_water, rows)
    }

    /// [`Self::retained`] against a caller-chosen watermark instead of the
    /// current one — what [`Self::commit`] uses to preview the retained
    /// range a candidate watermark would produce, before adopting it.
    fn retained_at(
        &self,
        branch: BranchId,
        layer: usize,
        watermark: u64,
        rows: u64,
    ) -> Result<Range<u64>> {
        let layout = self
            .layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))?;
        let page_tokens = self.geometry.page_tokens as u64;
        // The ring overwrites row `r` with row `r + capacity`, and a page is
        // gone once any of its rows has been. Two things make this the *stable*
        // base rather than the tightest one.
        //
        // It is computed from the private retention watermark
        // (`committed_high_water`, or a candidate replacement for it), not
        // the published frontier, and it adds the admitted undo headroom. So
        // the highest row a transaction could legally write —
        // `watermark + tentative_rows` — is already accounted for before the
        // transaction starts, the base cannot move while it is open, and an
        // abort cannot leave it advanced over rows the window still admits.
        // It costs the headroom in retained rows, which is exactly what the
        // headroom was admitted for.
        let reach = watermark.saturating_add(self.headroom()).max(rows);
        let overwritten = reach.saturating_sub(layout.capacity_rows);
        let base = overwritten.div_ceil(page_tokens) * page_tokens;
        let floor = self.branch_storage(branch)?.retained_floor[layer];
        Ok(base.max(floor).min(rows)..rows)
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
        self.placement_of_for(ROOT, layer, position)
    }

    fn placement_of_for(&self, branch: BranchId, layer: usize, position: u64) -> Result<Placement> {
        self.check_poisoned_on(branch)?;
        let layout = self
            .layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))?;
        if position >= self.branch_storage(branch)?.rows {
            return Err(invalid("position", "position has not been published"));
        }
        let retained = self.retained_for(branch, layer)?;
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
    /// What [`Self::append`] hands each layer's [`PagedKvWriter`]: one run per
    /// stretch of rows that shares a page. A writer does not compute
    /// `position / page_tokens` itself, which is the whole point — two
    /// implementations of one mapping is what this module exists to prevent.
    ///
    /// The positions come from [`StagedRows`], not from an argument: they are
    /// the authority's to choose and they are always the frontier.
    fn placements_for(
        &self,
        branch: BranchId,
        staged: &StagedRows,
        layer: usize,
    ) -> Result<Vec<Placement>> {
        let layout = self
            .layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))?;
        let open = self
            .branch_storage(branch)?
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
        // One run per page boundary the staged rows cross, plus one: a batch
        // need not start page-aligned, so its first run can end before a
        // whole page has passed, splitting off one more run than
        // `rows.div_ceil(page_tokens)` alone accounts for (e.g. 6 rows from
        // slot 5 of an 8-row page is two runs, not one).
        let capacity = staged
            .rows
            .div_ceil(page_tokens)
            .checked_add(1)
            .and_then(|runs| usize::try_from(runs).ok())
            .ok_or(Error::Dim(DimError::Overflow))?;
        out.try_reserve_exact(capacity)
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

    #[cfg(test)]
    fn placements(&self, staged: &StagedRows, layer: usize) -> Result<Vec<Placement>> {
        self.placements_for(ROOT, staged, layer)
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
        self.page_view_for_branch(ROOT, layer)
    }

    fn page_view_for_branch(&self, branch: BranchId, layer: usize) -> Result<PageView> {
        let retained = self.retained_for(branch, layer)?;
        self.page_view_for(branch, layer, retained)
    }

    /// [`Self::page_view`]'s table, for a retained range the caller already
    /// computed — real or a candidate from [`Self::retained_at`]. Split out
    /// so [`Self::commit`] can preview the view a prospective watermark would
    /// produce without first adopting it.
    fn page_view_for(
        &self,
        branch: BranchId,
        layer: usize,
        retained: Range<u64>,
    ) -> Result<PageView> {
        let layout = self
            .layout
            .get(layer)
            .copied()
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))?;
        let storage = self.branch_storage(branch)?;
        let staged_end = storage
            .open
            .and_then(|o| o.pending)
            .map_or(0, |(f, n)| f.saturating_add(n));
        let end = storage.rows.max(staged_end);
        if end == 0 || retained.start >= end {
            return Err(invalid("retained", "this layer holds no row"));
        }
        let page_tokens = self.geometry.page_tokens as u64;
        let first = retained.start / page_tokens;
        let last = (end - 1) / page_tokens;
        let mut table = Vec::new();
        let capacity = last
            .checked_sub(first)
            .and_then(|span| span.checked_add(1))
            .and_then(|pages| usize::try_from(pages).ok())
            .ok_or(Error::Dim(DimError::Overflow))?;
        table
            .try_reserve_exact(capacity)
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
        self.begin_for(ROOT)
    }

    fn begin_for(&mut self, branch: BranchId) -> Result<StateTransactionId> {
        self.check_poisoned_on(branch)?;
        if self.branch_storage(branch)?.open.is_some() {
            return Err(invalid("transaction", "a transaction is already open here"));
        }
        let id = self.state.begin(branch)?;
        let storage = self.branch_storage_mut(branch)?;
        storage.open = Some(OpenTransaction {
            id,
            base: storage.rows,
            pending: None,
        });
        storage.completed_layers.fill(false);
        Ok(id)
    }

    /// Stage `rows`, hand each layer's writer the page view and placements
    /// this authority chose for its own batch, and publish only if every
    /// writer returns `Ok`.
    ///
    /// This is the only public way to move the frontier: [`Self::stage`] and
    /// the publish step are private, so no value a caller can hold makes
    /// publication happen — the only way to reach it is to be one of
    /// `writers` and return success. A writer that returns `Ok` before its
    /// copy has completed is lying, and that is the writer's contract to
    /// keep; this method has no way to check it from here.
    ///
    /// The view passed is [`Self::page_view`] for that layer, computed after
    /// staging so it already covers this batch: what a page table contains,
    /// and when it slides, is this authority's decision, and a writer that
    /// derived its own would be a second one.
    ///
    /// `writers` must hold exactly one entry per layer, in layer order. On
    /// any writer's refusal the batch stays staged and unpublished, and the
    /// error is returned unchanged so the caller can abort the transaction.
    pub fn append(
        &mut self,
        txn: StateTransactionId,
        rows: u64,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        self.append_for(ROOT, txn, rows, writers)
    }

    fn append_for(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        rows: u64,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        self.check_poisoned_on(branch)?;
        if writers.len() != self.layout.len() {
            return Err(invalid(
                "writers",
                "append needs exactly one writer per layer, in layer order",
            ));
        }
        let staged = self.stage_for(branch, txn, rows)?;
        let batch = staged.id();
        for (layer, writer) in writers.iter_mut().enumerate() {
            let view = self.page_view_for_branch(branch, layer)?;
            let placements = self.placements_for(branch, &staged, layer)?;
            writer.write_layer(layer, batch, view, &placements)?;
            self.branch_storage_mut(branch)?.completed_layers[layer] = true;
        }
        self.publish_for(branch, txn, staged)
    }

    /// Append one layer of the staged batch.
    ///
    /// Layers execute in order, so an earlier layer must append and attend
    /// before a later layer has produced its K/V rows. The first call stages
    /// the batch. The sequence frontier advances only when every layer has
    /// completed that same batch.
    pub fn append_layer(
        &mut self,
        txn: StateTransactionId,
        layer: usize,
        rows: u64,
        writer: &mut dyn PagedKvWriter,
    ) -> Result<()> {
        self.append_layer_for(ROOT, txn, layer, rows, writer)
    }

    fn append_layer_for(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        layer: usize,
        rows: u64,
        writer: &mut dyn PagedKvWriter,
    ) -> Result<()> {
        self.check_poisoned_on(branch)?;
        if self.layout.get(layer).is_none() {
            return Err(invalid("layer", "layer is outside this sequence"));
        }
        let Some(next) = self
            .branch_storage(branch)?
            .completed_layers
            .iter()
            .position(|complete| !complete)
        else {
            return Err(invalid(
                "layer",
                "the staged batch already completed every layer",
            ));
        };
        if layer != next {
            return Err(invalid("layer", "layers must append in graph order"));
        }
        let staged = match self.open_transaction_for(branch, txn)?.pending {
            Some((first, pending_rows))
                if first == self.branch_storage(branch)?.rows && pending_rows == rows =>
            {
                StagedRows {
                    transaction: txn,
                    first,
                    rows,
                }
            }
            Some(_) => {
                return Err(invalid(
                    "rows",
                    "every layer must publish the same staged batch",
                ));
            }
            None => self.stage_for(branch, txn, rows)?,
        };
        let view = self.page_view_for_branch(branch, layer)?;
        let placements = self.placements_for(branch, &staged, layer)?;
        writer.write_layer(layer, staged.id(), view, &placements)?;
        let storage = self.branch_storage_mut(branch)?;
        storage.completed_layers[layer] = true;
        if storage.completed_layers.iter().all(|done| *done) {
            self.publish_for(branch, txn, staged)?;
        }
        Ok(())
    }

    /// Reserve the next `rows` positions for this transaction to write.
    ///
    /// The positions are the authority's to choose, and they are always the
    /// frontier: a dense, ordered append is the only shape paged state has.
    /// What comes back is the only thing [`Self::placements`] will place and
    /// the only thing the private publish step will publish.
    ///
    /// One batch at a time. Two outstanding batches could be published out of
    /// order, which would move the frontier over rows nothing had written.
    fn stage_for(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        rows: u64,
    ) -> Result<StagedRows> {
        let open = self.open_transaction_for(branch, txn)?;
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
            .branch_storage(branch)?
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
            first: self.branch_storage(branch)?.rows,
            rows,
        };
        self.branch_storage_mut(branch)?
            .open
            .as_mut()
            .expect("a validated open transaction")
            .pending = Some((staged.first, staged.rows));
        Ok(staged)
    }

    #[cfg(test)]
    fn stage(&mut self, txn: StateTransactionId, rows: u64) -> Result<StagedRows> {
        self.stage_for(ROOT, txn, rows)
    }

    /// Publish a staged batch as history.
    ///
    /// Private: the only path here is [`Self::append`], after every layer's
    /// writer has returned `Ok`. Before that the rows are staged bytes with
    /// no reader; after this call they are state. There is no moment in
    /// between, which is why this is a separate step from [`Self::stage`]
    /// rather than something that call does for you.
    fn publish_for(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        staged: StagedRows,
    ) -> Result<()> {
        let open = self.open_transaction_for(branch, txn)?;
        if staged.transaction != txn || open.pending != Some((staged.first, staged.rows)) {
            return Err(invalid(
                "staged",
                "these rows are not the batch this transaction staged",
            ));
        }
        if staged.first != self.branch_storage(branch)?.rows {
            return Err(invalid(
                "staged",
                "the staged batch no longer starts at the frontier",
            ));
        }
        self.state.execute(branch, staged.rows)?;
        let storage = self.branch_storage_mut(branch)?;
        storage.rows = staged.first + staged.rows;
        storage
            .open
            .as_mut()
            .expect("a validated open transaction")
            .pending = None;
        storage.completed_layers.fill(false);
        Ok(())
    }

    #[cfg(test)]
    fn publish(&mut self, txn: StateTransactionId, staged: StagedRows) -> Result<()> {
        self.publish_for(ROOT, txn, staged)
    }

    /// Accept `accept` of the transaction's rows, republish any layer whose
    /// page view the resulting retention change moves, and close the
    /// transaction.
    ///
    /// An unpublished staged batch is refused: its bytes may or may not have
    /// been written, and committing over that question is how a frontier ends
    /// up ahead of the rows behind it. `accept` is refused above what this
    /// transaction itself published: [`SequenceState::commit_prefix`] would
    /// otherwise accept rows this transaction never wrote, and possibly rows
    /// this authority has not published at all.
    ///
    /// This is the one place that knows the retention transition, so it is
    /// the one place that performs it: for each layer whose page view would
    /// change under the watermark this commit is about to adopt, the
    /// matching `writers` entry publishes it through
    /// [`PagedKvWriter::publish_view`] **before** anything is finalized. A
    /// layer whose view is unchanged is not republished.
    ///
    /// If every required publication succeeds, the commit finalizes. If any
    /// refuses, the commit does not finalize and this sequence is
    /// **poisoned**: a prior layer in `writers` order may already have
    /// published successfully, the device may already hold a partially
    /// updated mapping, and there is no way to take that back — see
    /// [`Self::check_poisoned`].
    ///
    /// `writers` must hold exactly one entry per layer, in layer order, same
    /// as [`Self::append`].
    pub fn commit(
        &mut self,
        txn: StateTransactionId,
        accept: u64,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        self.commit_for(ROOT, txn, accept, writers)
    }

    /// Take every refusal [`Self::commit`] could make before its first
    /// effect, and preview the page views it will publish, without changing
    /// anything. `commit` is exactly this followed by [`Self::apply_commit`].
    pub fn prepare_commit(&self, txn: StateTransactionId, accept: u64) -> Result<PreparedCommit> {
        self.prepare_for(ROOT, txn, accept)
    }

    /// Apply a commit prepared on this sequence, in this exact state. After
    /// the staleness check, the only refusals left are a writer count that is
    /// not one per layer, and a device publication failure, which poisons the
    /// sequence as [`Self::commit`] describes.
    pub fn apply_commit(
        &mut self,
        prepared: PreparedCommit,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        if prepared.sequence != self.id || prepared.generation != self.generation {
            return Err(invalid(
                "prepared_commit",
                "prepared on another sequence, or this one has changed since",
            ));
        }
        self.apply_for(ROOT, prepared, writers)
    }

    fn commit_for(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        accept: u64,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        let prepared = self.prepare_for(branch, txn, accept)?;
        self.apply_for(branch, prepared, writers)
    }

    fn prepare_for(
        &self,
        branch: BranchId,
        txn: StateTransactionId,
        accept: u64,
    ) -> Result<PreparedCommit> {
        self.check_poisoned_on(branch)?;
        let open = self.open_transaction_for(branch, txn)?;
        if open.pending.is_some() {
            return Err(invalid(
                "staged",
                "a staged batch is unpublished; publish it or abort the transaction",
            ));
        }
        let storage = self.branch_storage(branch)?;
        if accept > storage.rows - open.base {
            return Err(invalid(
                "accept",
                "cannot accept more rows than this transaction published",
            ));
        }
        self.state
            .frontiers(branch)?
            .accepted
            .checked_add(accept)
            .ok_or(Error::Dim(DimError::Overflow))?;

        // The watermark this commit is about to adopt. Publication and the
        // transition it enables are decided from the same number commit will
        // actually apply.
        let watermark_after = storage.committed_high_water.max(storage.rows);
        let mut updates = Vec::new();
        updates
            .try_reserve_exact(self.layout.len())
            .map_err(|_| Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: self
                    .layout
                    .len()
                    .saturating_mul(core::mem::size_of::<(usize, PageView)>())
                    as u64,
                available_bytes: 0,
            })?;
        for layer in 0..self.layout.len() {
            // A sequence that has never published a row has no view to
            // compare or republish for any layer.
            if storage.rows == 0 {
                continue;
            }
            let before = self.retained_for(branch, layer)?;
            let after_retained = self.retained_at(branch, layer, watermark_after, storage.rows)?;
            if before.start == after_retained.start {
                continue;
            }
            updates.push((layer, self.page_view_for(branch, layer, after_retained)?));
        }
        Ok(PreparedCommit {
            sequence: self.id,
            generation: self.generation,
            txn,
            accept,
            watermark_after,
            updates,
        })
    }

    fn apply_for(
        &mut self,
        branch: BranchId,
        prepared: PreparedCommit,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        if writers.len() != self.layout.len() {
            return Err(invalid(
                "writers",
                "commit needs exactly one writer per layer, in layer order",
            ));
        }
        let published_mapping = !prepared.updates.is_empty();
        for (layer, view) in prepared.updates {
            if let Err(error) = writers[layer].publish_view(layer, view) {
                self.branch_storage_mut(branch)?.poisoned = true;
                return Err(error);
            }
        }

        if let Err(error) = self.state.commit_prefix(prepared.txn, prepared.accept) {
            if published_mapping {
                self.branch_storage_mut(branch)?.poisoned = true;
            }
            return Err(error);
        }
        let storage = self.branch_storage_mut(branch)?;
        storage.open = None;
        // The ring's reach only moves at a commit. Everything the retained base
        // is derived from is therefore stable for the whole of a transaction.
        storage.committed_high_water = prepared.watermark_after;
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
        self.abort_for(ROOT, txn)
    }

    fn abort_for(&mut self, branch: BranchId, txn: StateTransactionId) -> Result<()> {
        self.check_poisoned_on(branch)?;
        let open = self.open_transaction_for(branch, txn)?;
        let base = open.base;
        self.state.abort(txn)?;
        let storage = self.branch_storage_mut(branch)?;
        storage.rows = base;
        storage.open = None;
        storage.completed_layers.fill(false);
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
        self.truncate_for(ROOT, prefix)
    }

    fn truncate_for(&mut self, branch: BranchId, prefix: u64) -> Result<()> {
        self.check_poisoned_on(branch)?;
        if self.branch_storage(branch)?.open.is_some() {
            return Err(invalid(
                "truncate",
                "a transaction is open; commit or abort before truncating",
            ));
        }
        if prefix > self.branch_storage(branch)?.rows {
            return Err(invalid("prefix", "truncating past the frontier"));
        }
        for layer in 0..self.layout.len() {
            let retained = self.retained_for(branch, layer)?;
            if prefix < retained.start {
                return Err(Error::Reclaimed {
                    layer: layer as u32,
                    position: prefix,
                    retained_from: retained.start,
                });
            }
        }
        self.state.rollback_to(branch, prefix, &[])?;
        let storage = self.branch_storage_mut(branch)?;
        storage.rows = prefix;
        storage.completed_layers.fill(false);
        Ok(())
    }

    fn reclaims(&self) -> bool {
        self.geometry
            .layers
            .iter()
            .any(|l| matches!(l.retention, Retention::Window { .. }))
    }

    /// The open transaction, if it is this one.
    fn open_transaction_for(
        &self,
        branch: BranchId,
        txn: StateTransactionId,
    ) -> Result<OpenTransaction> {
        let open = self
            .branch_storage(branch)?
            .open
            .ok_or_else(|| invalid("transaction", "no transaction is open"))?;
        if open.id != txn {
            return Err(invalid("transaction", "this transaction is not open here"));
        }
        Ok(open)
    }

    /// Fork the device decisions at an accepted root prefix and ask the
    /// executor-owned writers to eagerly copy the corresponding device pages.
    ///
    /// The logical child is created before the callbacks run. If a device copy
    /// refuses, both that child bookkeeping and the logical `SequenceState`
    /// branch are discarded, so a partially copied child cannot escape.
    pub fn fork(&mut self, at: u64, writers: &mut [&mut dyn PagedKvWriter]) -> Result<BranchId> {
        self.generation += 1;
        self.check_poisoned()?;
        if self.branches.len() > 1 {
            return Err(invalid(
                "branch",
                "only one live device child branch is supported",
            ));
        }
        if writers.len() != self.layout.len() {
            return Err(invalid(
                "writers",
                "fork needs exactly one writer per layer, in layer order",
            ));
        }
        if self.branch_storage(ROOT)?.open.is_some() {
            return Err(invalid(
                "branch",
                "fork requires a committed root prefix outside a transaction",
            ));
        }
        let parent_rows = self.branch_storage(ROOT)?.rows;
        let parent_high_water = self.branch_storage(ROOT)?.committed_high_water;
        let accepted = self.state.frontiers(ROOT)?.accepted;
        if at > accepted {
            return Err(invalid(
                "at",
                "a device branch cannot start past the parent's accepted frontier",
            ));
        }
        if at > parent_rows {
            return Err(invalid(
                "at",
                "a device branch cannot start past the parent's published rows",
            ));
        }
        let mut retained_floor = Vec::with_capacity(self.layout.len());
        for layer in 0..self.layout.len() {
            let retained = self.retained_for(ROOT, layer)?;
            if at < retained.start {
                return Err(Error::Reclaimed {
                    layer: layer as u32,
                    position: at,
                    retained_from: retained.start,
                });
            }
            retained_floor.push(retained.start);
        }

        let child = self.state.fork(ROOT, at)?;
        self.branches.insert(
            child,
            DeviceBranchStorage {
                rows: at,
                committed_high_water: parent_high_water.min(at),
                open: None,
                completed_layers: vec![false; self.layout.len()],
                poisoned: false,
                retained_floor,
            },
        );

        let copy_result = (|| {
            if at == 0 {
                return Ok(());
            }
            for (layer, writer) in writers.iter_mut().enumerate() {
                let view = self.page_view_for_branch(child, layer)?;
                writer.copy_branch(layer, view, at)?;
            }
            Ok(())
        })();
        if let Err(error) = copy_result {
            self.branches.remove(&child);
            // The logical fork was created before the device callbacks. This
            // is the post-fork cleanup path, not an early allocation refusal.
            let _ = self.state.discard_branch(child);
            return Err(error);
        }
        Ok(child)
    }

    /// Release the child decisions after its executor run has been closed.
    pub fn discard_branch(&mut self, branch: BranchId) -> Result<()> {
        self.generation += 1;
        self.check_poisoned_on(branch)?;
        if branch == ROOT {
            return Err(invalid("branch", "the root branch cannot be discarded"));
        }
        if self.branch_storage(branch)?.open.is_some() {
            return Err(invalid(
                "branch",
                "resolve the branch transaction before discarding it",
            ));
        }
        self.state.discard_branch(branch)?;
        self.branches.remove(&branch);
        Ok(())
    }

    /// Borrow one child branch's decisions for transaction and placement
    /// operations. The branch object is only a view; device bytes remain in
    /// the executor run that the caller supplies to the writer callbacks.
    pub fn branch(&mut self, branch: BranchId) -> Result<DeviceBranch<'_>> {
        if branch == ROOT {
            return Err(invalid(
                "branch",
                "the root branch is accessed through DeviceKvSequence",
            ));
        }
        self.check_poisoned_on(branch)?;
        Ok(DeviceBranch {
            sequence: self,
            branch,
        })
    }
}

impl DeviceBranch<'_> {
    pub fn id(&self) -> BranchId {
        self.branch
    }

    pub fn published_rows(&self) -> Result<u64> {
        self.sequence.published_rows_for(self.branch)
    }

    pub fn committed_rows(&self) -> Result<u64> {
        self.sequence.committed_rows_for(self.branch)
    }

    pub fn retained(&self, layer: usize) -> Result<Range<u64>> {
        self.sequence.retained_for(self.branch, layer)
    }

    pub fn layer_retained(&self, layer: usize) -> Result<Range<u64>> {
        self.sequence.layer_retained_for(self.branch, layer)
    }

    pub fn placement_of(&self, layer: usize, position: u64) -> Result<Placement> {
        self.sequence.placement_of_for(self.branch, layer, position)
    }

    pub fn page_view(&self, layer: usize) -> Result<PageView> {
        self.sequence.page_view_for_branch(self.branch, layer)
    }

    pub fn begin(&mut self) -> Result<StateTransactionId> {
        self.sequence.begin_for(self.branch)
    }

    pub fn append(
        &mut self,
        txn: StateTransactionId,
        rows: u64,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        self.sequence.append_for(self.branch, txn, rows, writers)
    }

    pub fn append_layer(
        &mut self,
        txn: StateTransactionId,
        layer: usize,
        rows: u64,
        writer: &mut dyn PagedKvWriter,
    ) -> Result<()> {
        self.sequence
            .append_layer_for(self.branch, txn, layer, rows, writer)
    }

    pub fn commit(
        &mut self,
        txn: StateTransactionId,
        accept: u64,
        writers: &mut [&mut dyn PagedKvWriter],
    ) -> Result<()> {
        self.sequence.commit_for(self.branch, txn, accept, writers)
    }

    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        self.sequence.abort_for(self.branch, txn)
    }

    pub fn truncate(&mut self, prefix: u64) -> Result<()> {
        self.sequence.truncate_for(self.branch, prefix)
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

    /// A writer that does no copy and returns `Ok`.
    ///
    /// These tests exercise the state authority's bookkeeping — staging,
    /// placement, frontiers, retention — not a device, so they have no copy
    /// to observe.
    struct NullWriter;

    impl PagedKvWriter for NullWriter {
        fn write_layer(
            &mut self,
            _layer: usize,
            _batch: BatchId,
            _view: PageView,
            _placements: &[PagePlacement],
        ) -> Result<()> {
            Ok(())
        }

        fn publish_view(&mut self, _layer: usize, _view: PageView) -> Result<()> {
            Ok(())
        }
    }

    /// A writer that counts `publish_view` calls and can be told to refuse
    /// them, for exercising [`DeviceKvSequence::commit`]'s own
    /// view-transition invariant rather than a device.
    #[derive(Default)]
    struct CountingWriter {
        publish_view_calls: usize,
        refuse_publish_view: bool,
    }

    impl PagedKvWriter for CountingWriter {
        fn write_layer(
            &mut self,
            _layer: usize,
            _batch: BatchId,
            _view: PageView,
            _placements: &[PagePlacement],
        ) -> Result<()> {
            Ok(())
        }

        fn publish_view(&mut self, _layer: usize, _view: PageView) -> Result<()> {
            self.publish_view_calls += 1;
            if self.refuse_publish_view {
                return Err(invalid("publish_view", "test refusal"));
            }
            Ok(())
        }
    }

    struct ForkWriter {
        copy_calls: usize,
        refuse_copy: bool,
    }

    impl PagedKvWriter for ForkWriter {
        fn write_layer(
            &mut self,
            _layer: usize,
            _batch: BatchId,
            _view: PageView,
            _placements: &[PagePlacement],
        ) -> Result<()> {
            Ok(())
        }

        fn publish_view(&mut self, _layer: usize, _view: PageView) -> Result<()> {
            Ok(())
        }

        fn copy_branch(&mut self, _layer: usize, _view: PageView, _rows: u64) -> Result<()> {
            self.copy_calls += 1;
            if self.refuse_copy {
                return Err(invalid("copy_branch", "injected post-fork copy refusal"));
            }
            Ok(())
        }
    }

    /// One [`NullWriter`] per layer, boxed as [`DeviceKvSequence::append`]
    /// and [`DeviceKvSequence::commit`] need them.
    fn null_writers(sequence: &DeviceKvSequence) -> Vec<NullWriter> {
        (0..sequence.layer_count().expect("not poisoned"))
            .map(|_| NullWriter)
            .collect()
    }

    fn writer_refs(writers: &mut [NullWriter]) -> Vec<&mut dyn PagedKvWriter> {
        writers
            .iter_mut()
            .map(|w| w as &mut dyn PagedKvWriter)
            .collect()
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
            let mut writers = null_writers(sequence);
            sequence
                .append(txn, step, &mut writer_refs(&mut writers))
                .expect("append");
            let mut commit_writers = null_writers(sequence);
            sequence
                .commit(txn, step, &mut writer_refs(&mut commit_writers))
                .expect("commit");
            done += step;
        }
    }

    #[test]
    fn device_fork_keeps_branch_bookkeeping_independent() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 8);
        let parent_frontier = sequence.committed_rows().expect("parent frontier");
        let parent_view = sequence.page_view(0).expect("parent view");
        let mut writer = ForkWriter {
            copy_calls: 0,
            refuse_copy: false,
        };
        let open = sequence.begin().expect("transaction");
        assert!(sequence.fork(4, &mut [&mut writer]).is_err());
        sequence.abort(open).expect("abort transaction");
        let child_id = sequence
            .fork(4, &mut [&mut writer])
            .expect("fork after the logical state exists");
        assert_eq!(writer.copy_calls, 1);
        assert_eq!(sequence.state().expect("state").branch_ids().len(), 2);
        let second_error = sequence
            .fork(4, &mut [&mut writer])
            .expect_err("a second live device child was admitted");
        assert!(matches!(
            second_error,
            Error::InvalidRequest {
                field: "branch",
                ..
            }
        ));
        assert_eq!(writer.copy_calls, 1);
        {
            let mut child = sequence.branch(child_id).expect("child view");
            assert_eq!(child.committed_rows().expect("child frontier"), 4);
            assert_eq!(child.published_rows().expect("child rows"), 4);
            assert_eq!(child.page_view(0).expect("child view"), parent_view);

            let txn = child.begin().expect("child transaction");
            let mut writers = [NullWriter];
            child
                .append(txn, 1, &mut writer_refs(&mut writers))
                .expect("child append");
            let mut commit_writers = [NullWriter];
            child
                .commit(txn, 1, &mut writer_refs(&mut commit_writers))
                .expect("child commit");
            assert_eq!(child.committed_rows().expect("child frontier"), 5);
            child.truncate(4).expect("child truncate");
            assert_eq!(child.committed_rows().expect("child frontier"), 4);
        }
        assert_eq!(
            sequence.committed_rows().expect("parent frontier"),
            parent_frontier
        );
        assert_eq!(sequence.page_view(0).expect("parent view"), parent_view);
        sequence
            .discard_branch(child_id)
            .expect("discard child decisions");
        assert_eq!(sequence.state().expect("state").branch_ids(), vec![ROOT]);
    }

    #[test]
    fn device_fork_copy_refusal_discards_post_fork_child() {
        let mut sequence = DeviceKvSequence::new(full(8, 64)).expect("a sequence");
        fill(&mut sequence, 8);
        let parent_frontier = sequence.committed_rows().expect("parent frontier");
        let parent_view = sequence.page_view(0).expect("parent view");
        let mut writer = ForkWriter {
            copy_calls: 0,
            refuse_copy: true,
        };
        assert!(sequence.fork(4, &mut [&mut writer]).is_err());
        assert_eq!(writer.copy_calls, 1);
        assert_eq!(sequence.state().expect("state").branch_ids(), vec![ROOT]);
        assert_eq!(
            sequence.committed_rows().expect("parent frontier"),
            parent_frontier
        );
        assert_eq!(sequence.page_view(0).expect("parent view"), parent_view);

        writer.refuse_copy = false;
        let fresh_child = sequence
            .fork(4, &mut [&mut writer])
            .expect("a fresh fork proves the child bookkeeping was removed");
        assert_eq!(writer.copy_calls, 2);
        sequence
            .discard_branch(fresh_child)
            .expect("discard the fresh child");
        assert_eq!(sequence.state().expect("state").branch_ids(), vec![ROOT]);
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
    fn commit_republishes_a_moved_view_and_poisons_on_refusal() {
        // Same geometry and fill pattern as `a_page_leaves_the_retained_range_whole`:
        // a hundred rows leaves the base at 72..100, one more row does not
        // move it, eight more moves it to 80..109.
        let mut geometry = windowed(24, 8, 8);
        geometry.layers.push(geometry.layers[0]);
        let mut sequence = DeviceKvSequence::new(geometry).expect("a sequence");
        fill(&mut sequence, 100);
        assert_eq!(sequence.retained(0).expect("a range"), 72..100);

        let append_one_batch = |sequence: &mut DeviceKvSequence, rows: u64| -> StateTransactionId {
            let txn = sequence.begin().expect("a transaction");
            let mut writers = null_writers(sequence);
            sequence
                .append(txn, rows, &mut writer_refs(&mut writers))
                .expect("append");
            txn
        };

        let mut first = CountingWriter::default();
        let mut second = CountingWriter::default();

        // A real forward publishes layers in order. The first layer can see
        // its staged row while the sequence frontier waits for the second.
        let txn = sequence.begin().expect("a transaction");
        sequence
            .append_layer(txn, 0, 1, &mut NullWriter)
            .expect("first layer");
        assert_eq!(sequence.published_rows().expect("frontier"), 100);
        assert_eq!(sequence.layer_published_rows(0).expect("layer 0"), 101);
        assert_eq!(sequence.layer_published_rows(1).expect("layer 1"), 100);
        sequence
            .append_layer(txn, 1, 1, &mut NullWriter)
            .expect("second layer");
        assert_eq!(sequence.published_rows().expect("frontier"), 101);

        // The base does not move: nothing is republished.
        sequence
            .commit(txn, 1, &mut [&mut first, &mut second])
            .expect("commit");
        assert_eq!(
            first.publish_view_calls, 0,
            "an unmoved view was republished"
        );
        assert_eq!(
            second.publish_view_calls, 0,
            "an unmoved view was republished"
        );
        assert_eq!(sequence.retained(0).expect("a range"), 72..101);

        // The base moves to 80: the writer sees exactly one republish.
        let txn = append_one_batch(&mut sequence, 8);
        sequence
            .commit(txn, 8, &mut [&mut first, &mut second])
            .expect("commit");
        assert_eq!(
            first.publish_view_calls, 1,
            "a moved view was not republished"
        );
        assert_eq!(
            second.publish_view_calls, 1,
            "a moved view was not republished"
        );
        assert_eq!(sequence.retained(0).expect("a range"), 80..109);

        // Force the base to move again and refuse the republish. Commit
        // itself refuses, and the sequence is poisoned: every public method
        // refuses from here on, including ones that only read.
        second.refuse_publish_view = true;
        let txn = append_one_batch(&mut sequence, 8);
        assert!(
            sequence
                .commit(txn, 8, &mut [&mut first, &mut second])
                .is_err()
        );
        assert_eq!(
            first.publish_view_calls, 2,
            "the first layer did not publish"
        );
        assert_eq!(
            second.publish_view_calls, 2,
            "the second layer was not attempted"
        );
        assert!(
            sequence.published_rows().is_err(),
            "poisoned but still answering"
        );
        assert!(
            sequence.layer_count().is_err(),
            "poisoned but still answering"
        );
        assert!(
            sequence.retained(0).is_err(),
            "poisoned but still answering"
        );
        assert!(
            sequence
                .append(txn, 1, &mut [&mut NullWriter as &mut dyn PagedKvWriter])
                .is_err(),
            "poisoned but still accepting writes"
        );
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
                    let rows = sequence.committed_rows().expect("frontiers");
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
        assert_eq!((staged.first, staged.rows), (5, 6));
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
        assert!(sequence.commit(txn, 6, &mut []).is_err());
        sequence.publish(txn, staged).expect("publish");
        // And publishing the same batch twice.
        assert!(sequence.publish(txn, staged).is_err());
        let mut writers = null_writers(&sequence);
        sequence
            .commit(txn, 6, &mut writer_refs(&mut writers))
            .expect("commit");

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
        assert_eq!(staged.first, 3, "staging starts at the frontier");

        // A batch from another sequence's transaction, and one this transaction
        // did not stage, are both refused.
        let mut other = DeviceKvSequence::new(full(8, 128)).expect("another sequence");
        let other_txn = other.begin().expect("a transaction");
        let foreign = other.stage(other_txn, 2).expect("stage");
        assert!(sequence.placements(&foreign, 0).is_err());
        assert!(sequence.publish(txn, foreign).is_err());
        sequence.publish(txn, staged).expect("publish");
        let mut writers = null_writers(&sequence);
        sequence
            .commit(txn, 2, &mut writer_refs(&mut writers))
            .expect("commit");
        assert_eq!(sequence.committed_rows().expect("frontiers"), 5);
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
        assert_eq!(sequence.published_rows().expect("not poisoned"), 16);
        // Tentative rows are legal state and are addressable before the commit
        // that accepts them: document 04 forbids publishing candidates before
        // verification, so they have to be readable without it.
        sequence
            .placement_of(0, 15)
            .expect("a tentative row is placed");
        sequence.abort(txn).expect("abort");
        assert_eq!(sequence.published_rows().expect("not poisoned"), 10);
        assert_eq!(sequence.retained(0).expect("a range"), before);
        // And the rows it published are not addressable any more.
        assert!(sequence.placement_of(0, 10).is_err());
    }

    #[test]
    fn a_prepared_commit_changes_nothing_and_binds_to_its_exact_state() {
        let mut sequence = DeviceKvSequence::new(windowed(8, 4, 4)).expect("a sequence");
        fill(&mut sequence, 20);
        let txn = sequence.begin().expect("a transaction");
        let staged = sequence.stage(txn, 3).expect("stage");
        sequence.publish(txn, staged).expect("publish");
        let observe = |s: &DeviceKvSequence| {
            (
                s.published_rows().unwrap(),
                s.committed_rows().unwrap(),
                s.retained(0).unwrap(),
                s.page_view(0).unwrap(),
            )
        };
        let before = observe(&sequence);
        let stale = sequence.prepare_commit(txn, 3).expect("prepare");
        assert!(!stale.updates.is_empty(), "this commit moves the window");
        assert_eq!(observe(&sequence), before, "preparing changes nothing");

        let mut other = DeviceKvSequence::new(windowed(8, 4, 4)).expect("a sequence");
        let other_txn = other.begin().expect("a transaction");
        let foreign = other.prepare_commit(other_txn, 0).expect("prepare");
        let mut writer = NullWriter;
        assert!(sequence.apply_commit(foreign, &mut [&mut writer]).is_err());
        // Any mutation after preparing makes the preparation stale.
        let staged = sequence.stage(txn, 1).expect("stage");
        sequence.publish(txn, staged).expect("publish");
        assert!(sequence.apply_commit(stale, &mut [&mut writer]).is_err());
        let fresh = sequence.prepare_commit(txn, 4).expect("prepare");
        sequence
            .apply_commit(fresh, &mut [&mut writer])
            .expect("a fresh preparation applies");
        assert_eq!(sequence.committed_rows().unwrap(), 24);
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
        let oldest = sequence.committed_rows().expect("frontiers") - 24;
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
        assert_eq!(sequence.published_rows().expect("not poisoned"), 12);
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
            sequence.published_rows().expect("not poisoned"),
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
        assert_eq!(
            sequence.published_rows().expect("not poisoned"),
            retained.start
        );
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
        let mut writers = null_writers(&sequence);
        sequence
            .commit(txn, 1, &mut writer_refs(&mut writers))
            .expect("commit");
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
        assert_eq!(sequence.published_rows().expect("not poisoned"), 40);
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
        let mut writers = null_writers(&sequence);
        sequence
            .commit(txn, 1, &mut writer_refs(&mut writers))
            .expect("commit");
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
