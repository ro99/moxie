//! Appendable host KV pages bound to the accepted sequence transaction journal.
//!
//! This is storage, not an attention executor. The root and its host branches
//! each own an admitted fixed envelope. Forks eagerly copy that envelope; lazy
//! COW and device views require later capability qualification.

use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};

use moxie_memory::{HostBuffer, Ledger};
use moxie_sampling::{Distribution, History, HistoryView, Layout as SamplingLayout};
use moxie_types::{
    BranchId, DimError, Error, HostTier, Precision, Result, StateTransactionId, Tier,
};

use crate::{Branch, Journal, PrefixLineage, ROOT, SequenceState, StateKind};

/// How much of its history one layer keeps.
///
/// A layer's retention is part of its state schema, not a policy the engine
/// applies afterwards, because it decides how many physical rows the layer is
/// admitted for. The declared window must equal the window in the graph's
/// `Visibility::SlidingWindow`; a store that retained less than its mask admits
/// would be silently wrong, and one that retained more would spend memory the
/// window was chosen to save.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// Every executed row, up to the admitted maximum.
    All,
    /// The `window` most recent rows, inclusive of the current position.
    /// Older rows are reclaimed and reading one is [`Error::Reclaimed`].
    Window { window: usize },
}

impl Retention {
    const fn window(self) -> Option<usize> {
        match self {
            Retention::All => None,
            Retention::Window { window } => Some(window),
        }
    }
}

/// One layer's key/value geometry and retention. K and V may have different
/// widths, and layers need not agree with each other: Gemma 4's sliding layers
/// are 16 heads of 256 and its global layers 4 of 512.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKv {
    pub kv_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub retention: Retention,
}

/// The whole sequence's page schema: one entry per layer, plus the properties
/// every layer shares.
///
/// Not `Copy`: the per-layer vector is allocated once at construction and
/// charged to the admitted control reserve, like every other byte here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvGeometry {
    pub layers: Vec<LayerKv>,
    pub precision: Precision,
    pub page_tokens: usize,
    pub max_tokens: usize,
    /// How many rows one transaction may append before it must commit.
    ///
    /// This is undo headroom, not context. A windowed layer is admitted for
    /// `window + tentative_rows` rows so that aborting the longest legal
    /// transaction still leaves every row its window can see; see
    /// [ADR 0014](../../../docs/decisions/adr/0014-bounded-tentative-undo-headroom.md).
    /// The headroom is never readable, so it cannot be mistaken for history.
    pub tentative_rows: usize,
}

impl KvGeometry {
    /// Every layer with the same width and full retention -- the shape every
    /// consumer had before per-layer geometry existed, and still the right
    /// description of a graph whose layers genuinely agree.
    pub fn uniform(
        layers: usize,
        kv_heads: usize,
        key_dim: usize,
        value_dim: usize,
        precision: Precision,
        page_tokens: usize,
        max_tokens: usize,
    ) -> Self {
        Self {
            layers: vec![
                LayerKv {
                    kv_heads,
                    key_dim,
                    value_dim,
                    retention: Retention::All,
                };
                layers
            ],
            precision,
            page_tokens,
            max_tokens,
            tentative_rows: max_tokens,
        }
    }
}

/// One layer's resolved byte layout. Pages belong to exactly one layer: layers
/// no longer agree on row width or on how many rows they keep, so they cannot
/// share a page.
#[derive(Debug, Clone, Copy)]
struct LayerLayout {
    key_bytes: usize,
    value_bytes: usize,
    /// Physical rows this layer is admitted for, always a whole number of
    /// pages and always at least `window + tentative_rows`.
    capacity: usize,
    pages: usize,
    page_bytes: usize,
    /// Index of this layer's first page-table entry.
    table_base: usize,
}

#[derive(Debug, Clone)]
struct Layout {
    layers: Vec<LayerLayout>,
    table_bytes: usize,
    backing_bytes: usize,
    control_bytes: usize,
}

fn row_ranges(
    layout: &Layout,
    geometry: &KvGeometry,
    backing: &HostBuffer,
    layer: usize,
    row: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let l = &layout.layers[layer];
    let page = (row / geometry.page_tokens) % l.pages;
    let local = row % geometry.page_tokens;
    let entry = (l.table_base + page) * 8;
    let table = &backing.bytes()[entry..entry + 8];
    let base = u64::from_le_bytes(table.try_into().expect("eight-byte entry")) as usize;
    let key = base + local * l.key_bytes;
    let value = base + geometry.page_tokens * l.key_bytes + local * l.value_bytes;
    (key..key + l.key_bytes, value..value + l.value_bytes)
}

fn truncate_storage(
    layout: &Layout,
    geometry: &KvGeometry,
    backing: &mut HostBuffer,
    current: usize,
    rows: usize,
) {
    for (index, layer) in layout.layers.iter().enumerate() {
        // Only the last `capacity` discarded rows have slots of their own;
        // anything older shares a slot with a row already zeroed here.
        // Zeroing a slot can clear the bytes of the row `capacity` below
        // it, which was already overwritten by the row being cleared and so
        // is below every retained start -- nothing readable is lost.
        for row in rows.max(current.saturating_sub(layer.capacity))..current {
            let (key, value) = row_ranges(layout, geometry, backing, index, row);
            backing.bytes_mut()[key].fill(0);
            backing.bytes_mut()[value].fill(0);
        }
    }
}

pub(super) fn btree_node_bound(entry: usize) -> usize {
    // The pinned Rust BTreeMap (11 entries, 12 edges) needs this conservative
    // bound for one node, including padding and the edge array.
    11 * (entry + size_of::<usize>()) + 16 * size_of::<usize>()
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
    fn layout(&self) -> Result<Layout> {
        if self.layers.is_empty() || self.page_tokens == 0 || self.max_tokens == 0 {
            return Err(invalid(
                "kv_geometry",
                "at least one layer, one page token and one context token",
            ));
        }
        if self.tentative_rows == 0 {
            return Err(invalid(
                "tentative_rows",
                "a transaction must be allowed to append at least one row",
            ));
        }
        if !self.precision.is_legal_cache() {
            return Err(Error::Unsupported {
                capability: "paged cache precision",
                reason: "only BF16, FP16 and FP32 encoded state is supported".into(),
            });
        }
        let element = self.precision.bits() as usize / 8;
        let mut layers = try_vec(self.layers.len())?;
        let mut entries = 0usize;
        let mut pool_bytes = 0usize;
        for layer in &self.layers {
            if [layer.kv_heads, layer.key_dim, layer.value_dim].contains(&0) {
                return Err(invalid("kv_geometry", "all dimensions must be positive"));
            }
            let key_bytes = mul(mul(layer.kv_heads, layer.key_dim)?, element)?;
            let value_bytes = mul(mul(layer.kv_heads, layer.value_dim)?, element)?;
            // A windowed layer is admitted for what it can see plus the undo
            // headroom, and never for more than the whole context -- a window
            // wider than the context is legal and simply means full retention.
            //
            // The clamp is why the abort argument survives it. Capacity is
            // `min(window + tentative_rows, max_tokens)` rounded up to a page,
            // so it can be *below* `window + tentative_rows` only when it is at
            // or above `max_tokens` -- and a layer with that much capacity
            // never reclaims, because `rows` cannot exceed `max_tokens`. Where
            // reclamation can happen at all, the headroom is fully present.
            let needed = match layer.retention {
                Retention::All => self.max_tokens,
                Retention::Window { window } => {
                    if window == 0 {
                        return Err(invalid(
                            "retention",
                            "a window must retain at least one row",
                        ));
                    }
                    add(window, self.tentative_rows)?.min(self.max_tokens)
                }
            };
            let pages = needed.div_ceil(self.page_tokens);
            let capacity = mul(pages, self.page_tokens)?;
            let page_bytes = mul(self.page_tokens, add(key_bytes, value_bytes)?)?;
            pool_bytes = add(pool_bytes, mul(pages, page_bytes)?)?;
            layers.push(LayerLayout {
                key_bytes,
                value_bytes,
                capacity,
                pages,
                page_bytes,
                table_base: entries,
            });
            entries = add(entries, pages)?;
        }
        let table_bytes = mul(entries, size_of::<u64>())?;
        let backing_bytes = add(table_bytes, pool_bytes)?;
        // The facade permits one root branch, one open journal and no retained
        // results. Lineage is reserved exactly once. No per-token map entries
        // can accumulate here.
        let control_bytes = add(
            add(
                mul(add(self.max_tokens, 1)?, size_of::<PrefixLineage>())?,
                // The two per-layer vectors, each allocated once: the caller's
                // declared geometry and this resolved layout.
                add(
                    mul(self.layers.len(), size_of::<LayerKv>())?,
                    mul(self.layers.len(), size_of::<LayerLayout>())?,
                )?,
            )?,
            size_of::<Self>()
                + size_of::<PagedSequence>()
                + size_of::<StateKind>()
                + btree_node_bound(size_of::<(BranchId, Branch)>())
                + btree_node_bound(size_of::<(StateTransactionId, Journal)>())
                + btree_node_bound(size_of::<(crate::ResultId, crate::LogitsHandle)>()),
        )?;
        if backing_bytes > isize::MAX as usize || control_bytes > isize::MAX as usize {
            return Err(DimError::Overflow.into());
        }
        Ok(Layout {
            layers,
            table_bytes,
            backing_bytes,
            control_bytes,
        })
    }
}

fn try_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(Tier::Host(HostTier::Pageable)),
            requested_bytes: (capacity.saturating_mul(size_of::<T>())) as u64,
            available_bytes: 0,
        })?;
    Ok(out)
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
    /// The executed frontier: how many rows have been written.
    pub rows: usize,
    /// Rows still readable, summed over layers. Below `rows * layers` exactly
    /// when a windowed layer has reclaimed something.
    pub retained_rows: usize,
    pub live_pages: usize,
    pub live_page_bytes: usize,
    /// Encoded bytes of the rows that are still readable.
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
    /// The greatest `rows` ever reached. It never decreases, because a ring
    /// slot that has been overwritten stays overwritten: truncating the logical
    /// frontier does not bring back the row that was physically replaced. This
    /// is what lets a rollback tell whether its target's window survives.
    high_water: usize,
    /// The executed frontier when the open transaction began, so an append can
    /// refuse before it writes the row that would make the transaction too long
    /// to undo. `None` outside a transaction.
    tentative_base: Option<usize>,
    sampler: Option<Sampler>,
    execution: Option<PagedExecutionBinding>,
    children: BTreeMap<BranchId, PagedBranchStorage>,
}

#[derive(Debug)]
struct PagedBranchStorage {
    backing: HostBuffer,
    rows: usize,
    high_water: usize,
    tentative_base: Option<usize>,
    retained_from: Vec<usize>,
}

struct PagedRead<'a> {
    backing: &'a HostBuffer,
    rows: usize,
    high_water: usize,
    retained_from: Option<&'a [usize]>,
}

/// A mutable view of one eagerly-copied host paged child branch.
///
/// The view borrows its parent sequence, so a child cannot be used after the
/// sequence is discarded or closed, and the parent cannot release the child's
/// bytes while this view exists. The root branch keeps the original API;
/// branch operations are explicit here so existing consumers remain unchanged.
#[derive(Debug)]
pub struct PagedBranch<'a> {
    sequence: &'a mut PagedSequence,
    branch: BranchId,
}

/// Opaque authority for the one immutable execution configuration bound to a
/// paged sequence. Only [`PagedSequence::claim_execution`] can mint one, and a
/// second claimant is refused even when its graph has compatible geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedExecutionBinding(u64);

#[derive(Debug)]
struct Sampler {
    history: History,
    seed: u64,
}

/// A prepared distribution holds the exclusive sequence borrow. Its prefix and
/// transaction cannot change between preparation and staging. Numerical copies
/// have no authority to publish a generated token.
///
/// ```compile_fail
/// use moxie_state::PagedSequence;
/// use moxie_types::StateTransactionId;
/// use std::sync::atomic::AtomicBool;
/// fn stale(s: &mut PagedSequence, txn: StateTransactionId) {
///     let prepared = s.prepare_sample(txn, 1, &[0.0; 3], None, 1.0).unwrap();
///     s.abort(txn).unwrap();
///     prepared.stage(&AtomicBool::new(false)).unwrap();
/// }
/// ```
#[derive(Debug)]
pub struct PreparedSample<'a> {
    sequence: &'a mut PagedSequence,
    txn: StateTransactionId,
    prefix: u64,
    greedy: bool,
}

impl PreparedSample<'_> {
    pub fn distribution(&self) -> Distribution<'_> {
        Distribution::from_bytes(self.sequence.workspace()).expect("prepared probabilities")
    }
    pub fn draw(&self) -> Result<u32> {
        let sampler = self.sequence.sampler.as_ref().expect("sampling session");
        self.distribution()
            .draw(sampler.seed, sampler.history.len() as u64, self.greedy)
    }
    pub fn stage(self, cancelled: &AtomicBool) -> Result<u32> {
        self.stage_checked(|| {
            if cancelled.load(Ordering::Relaxed) {
                Err(Error::Cancelled {
                    at: "sample publication",
                })
            } else {
                Ok(())
            }
        })
    }
    fn stage_checked(self, mut checkpoint: impl FnMut() -> Result<()>) -> Result<u32> {
        self.sequence.check_transaction(self.txn)?;
        let result = (|| {
            checkpoint()?;
            let token = self.draw()?;
            let start = self.sequence.layout.backing_bytes;
            let sampler = self.sequence.sampler.as_mut().expect("sampling session");
            let end = start + sampler.history.layout().state_bytes();
            sampler.history.append(
                &mut self.sequence.backing.bytes_mut()[start..end],
                self.prefix,
                token,
            )?;
            checkpoint()?;
            Ok(token)
        })();
        if result.is_err() {
            self.sequence
                .abort(self.txn)
                .expect("validated transaction");
        }
        result
    }
}

/// A refused close retains the entire sequence, including release authority.
#[derive(Debug)]
pub struct PagedCloseRefused {
    pub sequence: PagedSequence,
    pub error: Error,
}

impl PagedSequence {
    pub fn new(ledger: &mut Ledger, geometry: KvGeometry) -> Result<Self> {
        Self::construct(ledger, geometry, None)
    }

    /// One generation, with fixed vocabulary/history capacity and an explicit seed.
    pub fn with_sampling(
        ledger: &mut Ledger,
        geometry: KvGeometry,
        vocabulary: usize,
        history_capacity: usize,
        seed: u64,
    ) -> Result<Self> {
        if history_capacity > geometry.max_tokens {
            return Err(invalid(
                "history_capacity",
                "history exceeds sequence capacity",
            ));
        }
        Self::construct(
            ledger,
            geometry,
            Some((SamplingLayout::new(vocabulary, history_capacity)?, seed)),
        )
    }

    fn construct(
        ledger: &mut Ledger,
        mut geometry: KvGeometry,
        sampling: Option<(SamplingLayout, u64)>,
    ) -> Result<Self> {
        // Normalize the caller's vector to exactly its length, fallibly, before
        // anything is charged or admitted.
        //
        // The sequence takes ownership of this vector and holds it for its
        // whole life, so what it retains is the vector's *capacity* -- and a
        // caller is free to hand over one built with `Vec::with_capacity` far
        // larger than its length. Charging `len` for an allocation of `capacity`
        // would put megabytes outside the memory authority, which is precisely
        // the untracked residency this repository has one ledger to prevent.
        // Re-seating it makes the charge below exact by construction.
        let mut layers = try_vec(geometry.layers.len())?;
        layers.extend_from_slice(&geometry.layers);
        geometry.layers = layers;
        let layout = geometry.layout()?;
        let history_bytes = sampling.map_or(0, |(s, _)| s.state_bytes());
        let workspace_bytes = sampling.map_or(0, |(s, _)| s.workspace_bytes());
        let control_bytes = add(
            layout.control_bytes,
            if sampling.is_some() {
                size_of::<StateKind>()
            } else {
                0
            },
        )?;
        let mut backing = HostBuffer::allocate_with_workspace(
            ledger,
            "paged sequence",
            add(layout.backing_bytes, history_bytes)?,
            workspace_bytes,
            control_bytes,
        )?;
        let schema = [StateKind::KvPages, StateKind::SamplerHistory];
        let mut state = SequenceState::new(
            schema[..if sampling.is_some() { 2 } else { 1 }]
                .iter()
                .copied(),
        );
        let lineage = &mut state.branches.get_mut(&ROOT).expect("root exists").lineage;
        let capacity = geometry.max_tokens + 1; // checked by layout
        let lineage_bytes =
            u64::try_from(capacity * size_of::<PrefixLineage>()).map_err(|_| DimError::Overflow)?; // product checked by layout
        if lineage.try_reserve_exact(capacity - lineage.len()).is_err()
            || lineage.capacity() != capacity
        {
            drop(state);
            backing.release(ledger).expect("admitting ledger");
            return Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: lineage_bytes,
                available_bytes: 0,
            });
        }
        // Immutable page table, one contiguous pool per layer. Pages are
        // statically partitioned in this first exclusive-owner implementation;
        // no per-row allocation or dense history materialization is involved in
        // addressing them.
        let mut offset = layout.table_bytes;
        for layer in &layout.layers {
            for page in 0..layer.pages {
                let entry = (layer.table_base + page) * 8;
                backing.bytes_mut()[entry..entry + 8]
                    .copy_from_slice(&(offset as u64).to_le_bytes());
                offset += layer.page_bytes;
            }
        }
        let sampler = sampling.map(|(sampling, seed)| Sampler {
            history: History::new(
                sampling,
                &mut backing.bytes_mut()
                    [layout.backing_bytes..layout.backing_bytes + history_bytes],
            )
            .expect("checked layout"),
            seed,
        });
        Ok(Self {
            state,
            geometry,
            layout,
            backing,
            rows: 0,
            high_water: 0,
            tentative_base: None,
            sampler,
            execution: None,
            children: BTreeMap::new(),
        })
    }

    /// Bind this sequence to one immutable executor before any physical row is
    /// written. The prompt frontier may already be declared, but no second
    /// program can acquire the existing authority.
    pub fn claim_execution(&mut self) -> Result<PagedExecutionBinding> {
        if self.execution.is_some() {
            return Err(invalid(
                "execution_configuration",
                "paged sequence is already bound to an immutable program",
            ));
        }
        if self.rows != 0 || !self.state.open_transactions().is_empty() {
            return Err(invalid(
                "execution_configuration",
                "bind before physical execution and outside a transaction",
            ));
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| {
                invalid(
                    "execution_configuration",
                    "process execution binding identity space is exhausted",
                )
            })?;
        let binding = PagedExecutionBinding(id);
        self.execution = Some(binding);
        Ok(binding)
    }

    pub fn validate_execution(&self, binding: PagedExecutionBinding) -> Result<()> {
        if self.execution != Some(binding) {
            return Err(invalid(
                "execution_configuration",
                "program does not own this paged history",
            ));
        }
        Ok(())
    }

    pub fn state(&self) -> &SequenceState {
        &self.state
    }

    /// Validate authority before an executor reads or publishes this transaction.
    pub fn validate_transaction(&self, txn: StateTransactionId) -> Result<()> {
        self.check_transaction(txn)
    }

    /// Retain at most one result in this bounded facade. The executor must bind
    /// the returned identity to the immutable logits it actually computed.
    pub fn record_logits(&mut self, txn: StateTransactionId) -> Result<crate::LogitsHandle> {
        self.check_transaction(txn)?;
        if !self.state.live.is_empty() {
            return Err(invalid(
                "logits",
                "clear the previous result before beginning forward",
            ));
        }
        self.state.record_logits(ROOT, self.rows as u64)
    }

    /// Retire the previous forward result before executing another. Existing
    /// generation invalidation refuses an open transaction and stales all handles.
    pub fn clear_logits(&mut self) -> Result<()> {
        self.state.invalidate_generation()
    }

    pub fn geometry(&self) -> &KvGeometry {
        &self.geometry
    }

    pub fn layer_count(&self) -> usize {
        self.geometry.layers.len()
    }

    /// The absolute positions layer `layer` still holds, as `start..rows`.
    ///
    /// The start is the later of what the window admits and what the ring has
    /// already overwritten. The second term only exceeds the first after a
    /// rollback moved the frontier back below what was reclaimed, which is the
    /// case [`Self::rollback_to`] refuses outright.
    ///
    /// For a window of `w` the start is `rows - w`, which is one row wider than
    /// the *next* query strictly needs: a query at position `rows` sees keys
    /// from `rows - w + 1`. The extra row is the one a query at `rows - 1`
    /// would need, so the most recently executed position can be re-executed
    /// without re-prefilling -- which document 04's two-frontier rule makes an
    /// ordinary operation rather than a special case. It costs one row per
    /// windowed layer and is charged like every other row.
    pub fn retained_range(&self, layer: usize) -> Result<std::ops::Range<u64>> {
        if layer >= self.geometry.layers.len() {
            return Err(invalid("kv_row", "layer is outside this sequence"));
        }
        Ok(self.retained_start(layer) as u64..self.rows as u64)
    }

    fn retained_start(&self, layer: usize) -> usize {
        self.retained_start_for(layer, self.rows, self.high_water)
    }

    fn retained_start_for(&self, layer: usize, rows: usize, high_water: usize) -> usize {
        let evicted = high_water.saturating_sub(self.layout.layers[layer].capacity);
        match self.geometry.layers[layer].retention.window() {
            // `capacity >= max_tokens >= high_water`, so `evicted` is zero.
            None => evicted,
            Some(window) => rows.saturating_sub(window).max(evicted),
        }
    }

    pub fn usage(&self) -> PagedUsage {
        self.usage_for(self.rows, self.high_water)
    }

    fn usage_for(&self, rows: usize, high_water: usize) -> PagedUsage {
        self.usage_for_branch(rows, high_water, None)
    }

    fn usage_for_branch(
        &self,
        rows: usize,
        high_water: usize,
        retained_from: Option<&[usize]>,
    ) -> PagedUsage {
        let mut live_pages = 0;
        let mut live_page_bytes = 0;
        let mut retained_rows = 0;
        let mut logical_kv_bytes = 0;
        for (index, layer) in self.layout.layers.iter().enumerate() {
            let pages = rows.div_ceil(self.geometry.page_tokens).min(layer.pages);
            live_pages += pages;
            live_page_bytes += pages * layer.page_bytes;
            let start = self
                .retained_start_for(index, rows, high_water)
                .max(retained_from.map_or(0, |from| from[index]));
            let retained = rows.saturating_sub(start);
            retained_rows += retained;
            logical_kv_bytes += retained * (layer.key_bytes + layer.value_bytes);
        }
        PagedUsage {
            rows,
            retained_rows,
            live_pages,
            live_page_bytes,
            logical_kv_bytes,
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
        if self.sampler.as_ref().is_some_and(|s| !s.history.is_empty()) {
            return Err(invalid(
                "prompt",
                "one generation: prompt must precede generated history",
            ));
        }
        self.check_frontier(self.state.frontiers(ROOT)?.accepted, n)?;
        self.state.append_prompt(ROOT, n)
    }

    pub fn accept(&mut self, n: u64) -> Result<()> {
        if self.sampler.is_some() {
            return Err(invalid(
                "accept",
                "sampling sessions publish token identities through commit_prefix",
            ));
        }
        self.check_frontier(self.state.frontiers(ROOT)?.accepted, n)?;
        self.state.accept(ROOT, n)
    }

    pub fn emit(&mut self, n: u64) -> Result<()> {
        self.state.emit(ROOT, n)
    }

    pub fn begin(&mut self) -> Result<StateTransactionId> {
        let txn = self.state.begin(ROOT)?;
        self.state
            .open
            .get_mut(&txn)
            .expect("opened journal")
            .sampler_len = self.sampler.as_ref().map_or(0, |s| s.history.len());
        // The facade owns one branch, so at most one transaction is open and
        // one base suffices. It is what an append measures its length against.
        self.tentative_base = Some(self.rows);
        Ok(txn)
    }

    /// Whether any layer can reclaim. A sequence of full-retention layers is
    /// always exactly restorable, so the tentative bound does not apply to it
    /// and existing consumers keep their previous freedom.
    fn reclaims(&self) -> bool {
        self.geometry
            .layers
            .iter()
            .any(|l| l.retention.window().is_some())
    }

    /// Task 0004 semantics: accept n additional tokens, keep executed work,
    /// close. Verification truncates an unwanted executed suffix explicitly
    /// with `rollback_to` after resolving; a zero commit keeps materialization.
    pub fn commit_prefix(&mut self, txn: StateTransactionId, n: u64) -> Result<()> {
        self.commit_checked(txn, n, || Ok(()))
    }

    /// Observe cancellation through publication of both participants, before
    /// closing the shared journal. A later cancellation cannot retract emitted
    /// output; emission is a separate operation after successful commit.
    pub fn commit_prefix_cancellable(
        &mut self,
        txn: StateTransactionId,
        n: u64,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        self.commit_checked(txn, n, || {
            if cancelled.load(Ordering::Relaxed) {
                Err(Error::Cancelled {
                    at: "sample commit",
                })
            } else {
                Ok(())
            }
        })
    }

    fn commit_checked(
        &mut self,
        txn: StateTransactionId,
        n: u64,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.check_transaction(txn)?;
        self.check_frontier(self.state.frontiers(ROOT)?.accepted, n)?;
        let publish = if let Some(s) = &self.sampler {
            let n = usize::try_from(n).map_err(|_| DimError::Overflow)?;
            let publish = add(s.history.committed_len(), n)?;
            if publish > s.history.len() {
                return Err(invalid("accepted", "no matching staged token identities"));
            }
            Some(publish)
        } else {
            None
        };
        // Keep the existing journal open until both participants are published.
        // Its frontier and sampler marks undo either intermediate state.
        // Validation above is nonmutating; a failed validation remains abortable.
        let result = (|| {
            checkpoint()?;
            self.state.accept(ROOT, n)?;
            checkpoint()?;
            if let Some(len) = publish {
                self.update_history(|h, b| h.publish(b, len));
            }
            checkpoint()?;
            self.state.commit_prefix(txn, 0)
        })();
        if result.is_err() {
            self.abort(txn).expect("validated open publication journal");
        } else {
            self.tentative_base = None;
        }
        result
    }

    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        self.check_transaction(txn)?;
        let len = self.state.open[&txn].sampler_len;
        self.state.abort(txn)?;
        if self.sampler.is_some() {
            self.update_history(|h, b| h.truncate(b, len));
        }
        self.truncate(self.state.frontiers(ROOT)?.executed as usize);
        self.tentative_base = None;
        Ok(())
    }

    pub fn rollback_to(&mut self, prefix: u64) -> Result<()> {
        self.check_retained_at(prefix)?;
        if self.sampler.is_some() {
            // Prevalidate every destructive refusal before applying count undo.
            // The facade owns the schema and has no mutable raw-state escape.
            let f = self.state.frontiers(ROOT)?;
            if self.state.open_on(ROOT).is_some()
                || prefix > f.accepted
                || prefix < f.prompt
                || prefix < f.prompt + f.emitted
                || self.state.lineage_at(ROOT, prefix)?.is_none()
            {
                return Err(invalid(
                    "prefix",
                    "rollback target is open, unaccepted, inside prompt or already emitted",
                ));
            }
            let len = self
                .history(true)?
                .entries(0)
                .take_while(|(p, _)| *p < prefix)
                .count();
            self.update_history(|h, b| h.replay_prefix(b, len));
            let evidence = self.state.restore_evidence(
                StateKind::SamplerHistory,
                ROOT,
                crate::RestoreMethod::Replay {
                    from: f.prompt,
                    to: prefix,
                },
            )?;
            self.state
                .rollback_to(ROOT, prefix, &[evidence])
                .expect("validated exclusive schema and completed restoration");
        } else {
            self.state.rollback_to(ROOT, prefix, &[])?;
        }
        self.truncate(self.state.frontiers(ROOT)?.executed as usize);
        Ok(())
    }

    /// Refuse a rollback whose target window has already been reclaimed.
    ///
    /// Unlike an abort, which the tentative bound keeps exact, a rollback may
    /// reach back across any number of committed transactions, and the rows its
    /// target needs to see may be long overwritten. Document 04: "if history
    /// has been released, recompute or report that the requested operation
    /// needs re-prefill. Do not silently change results." Reported, not
    /// silently served, and checked before anything mutates.
    fn check_retained_at(&self, prefix: u64) -> Result<()> {
        self.check_retained_at_with_floor(self.high_water, prefix, None)
    }

    fn check_retained_at_with_floor(
        &self,
        high_water: usize,
        prefix: u64,
        retained_from: Option<&[usize]>,
    ) -> Result<()> {
        for (index, layer) in self.geometry.layers.iter().enumerate() {
            let Some(window) = layer.retention.window() else {
                continue;
            };
            let evicted = high_water.saturating_sub(self.layout.layers[index].capacity) as u64;
            let wanted = prefix.saturating_sub(window as u64);
            let retained_start =
                retained_from.map_or(evicted, |from| evicted.max(from[index] as u64));
            if wanted < retained_start {
                return Err(Error::Reclaimed {
                    layer: index as u32,
                    position: wanted,
                    retained_from: retained_start,
                });
            }
        }
        Ok(())
    }

    fn child_storage(&self, branch: BranchId) -> Result<&PagedBranchStorage> {
        self.children.get(&branch).ok_or_else(|| {
            invalid(
                "branch",
                format!("branch {branch} has no host paged storage"),
            )
        })
    }

    fn child_storage_mut(&mut self, branch: BranchId) -> Result<&mut PagedBranchStorage> {
        self.children.get_mut(&branch).ok_or_else(|| {
            invalid(
                "branch",
                format!("branch {branch} has no host paged storage"),
            )
        })
    }

    fn branch_control_bytes(&self) -> Result<usize> {
        add(
            add(
                mul(
                    add(self.geometry.max_tokens, 1)?,
                    size_of::<PrefixLineage>(),
                )?,
                mul(self.geometry.layers.len(), size_of::<usize>())?,
            )?,
            btree_node_bound(size_of::<(BranchId, Branch)>())
                + btree_node_bound(size_of::<(StateTransactionId, Journal)>())
                + btree_node_bound(size_of::<(BranchId, PagedBranchStorage)>()),
        )
    }

    /// Borrow one eagerly-copied host branch for mutation and reads.
    pub fn branch(&mut self, branch: BranchId) -> Result<PagedBranch<'_>> {
        if branch == ROOT {
            return Err(invalid(
                "branch",
                "the root branch is operated through PagedSequence",
            ));
        }
        self.child_storage(branch)?;
        self.state.frontiers(branch)?;
        Ok(PagedBranch {
            sequence: self,
            branch,
        })
    }

    /// Fork the committed root prefix into an independent host backing.
    ///
    /// The implementation deliberately chooses eager copying over lazy COW:
    /// the second branch has an ordinary `HostBuffer` reservation and therefore
    /// needs no new sharing, refcount, or write-barrier authority. The copy is
    /// made before `SequenceState` is changed, so allocation and copy failures
    /// leave the root branch untouched; a post-fork checkpoint unwinds both
    /// participants before reporting its failure.
    pub fn fork(&mut self, ledger: &mut Ledger, at: u64) -> Result<BranchId> {
        self.fork_checked(ledger, at, || Ok(()))
    }

    fn fork_checked(
        &mut self,
        ledger: &mut Ledger,
        at: u64,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<BranchId> {
        if self.sampler.is_some() {
            return Err(Error::Unsupported {
                capability: "paged state fork",
                reason: "sampler-history branching is outside the host KV fork primitive".into(),
            });
        }
        if self.state.open_on(ROOT).is_some() {
            return Err(invalid(
                "branch",
                "fork requires a committed root prefix outside a transaction",
            ));
        }
        let frontier = self.state.frontiers(ROOT)?;
        if at > frontier.accepted {
            return Err(invalid(
                "at",
                format!(
                    "cannot fork at {at}, root has only {} accepted",
                    frontier.accepted
                ),
            ));
        }
        if at > self.rows as u64 {
            return Err(invalid(
                "at",
                "cannot fork before the requested prefix has physical KV rows",
            ));
        }
        let mut retained_from = try_vec(self.geometry.layers.len())?;
        for layer in 0..self.geometry.layers.len() {
            retained_from.push(self.retained_start(layer));
        }
        self.check_retained_at_with_floor(self.high_water, at, Some(&retained_from))?;
        let at = usize::try_from(at).map_err(|_| DimError::Overflow)?;
        checkpoint()?;

        let branch_control = self.branch_control_bytes()?;
        let mut backing = HostBuffer::allocate_with_workspace(
            ledger,
            "paged branch",
            self.layout.backing_bytes,
            0,
            branch_control,
        )?;
        if let Err(error) = checkpoint() {
            backing.release(ledger).expect("the admitting ledger");
            return Err(error);
        }
        backing.bytes_mut().copy_from_slice(self.backing.bytes());
        if let Err(error) = checkpoint() {
            backing.release(ledger).expect("the admitting ledger");
            return Err(error);
        }

        let child = match self.state.fork(ROOT, at as u64) {
            Ok(child) => child,
            Err(error) => {
                backing.release(ledger).expect("the admitting ledger");
                return Err(error);
            }
        };
        if let Err(error) = checkpoint() {
            self.state
                .discard_branch(child)
                .expect("new logical branch is not open");
            backing.release(ledger).expect("the admitting ledger");
            return Err(error);
        }
        let lineage_capacity = add(self.geometry.max_tokens, 1)?;
        let lineage_ready = {
            let lineage = &mut self
                .state
                .branches
                .get_mut(&child)
                .expect("new logical branch")
                .lineage;
            lineage
                .try_reserve_exact(lineage_capacity - lineage.len())
                .is_ok()
                && lineage.capacity() == lineage_capacity
        };
        if !lineage_ready {
            self.state
                .discard_branch(child)
                .expect("new logical branch is not open");
            backing.release(ledger).expect("the admitting ledger");
            return Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: branch_control as u64,
                available_bytes: 0,
            });
        }
        let previous = self.children.insert(
            child,
            PagedBranchStorage {
                backing,
                rows: at,
                high_water: self.high_water,
                tentative_base: None,
                retained_from,
            },
        );
        debug_assert!(
            previous.is_none(),
            "SequenceState issued a duplicate branch id"
        );
        Ok(child)
    }

    /// Discard a child branch and release exactly its eager-copy reservation.
    pub fn discard_branch(&mut self, ledger: &mut Ledger, branch: BranchId) -> Result<()> {
        if branch == ROOT {
            return Err(invalid("branch", "the root branch cannot be discarded"));
        }
        self.child_storage(branch)?;
        self.state.frontiers(branch)?;
        if self.state.open_on(branch).is_some() {
            return Err(invalid(
                "branch",
                "resolve the child transaction before discarding its branch",
            ));
        }
        let mut storage = self
            .children
            .remove(&branch)
            .expect("validated child storage");
        if let Err(error) = storage.backing.release(ledger) {
            self.children.insert(branch, storage);
            return Err(error);
        }
        // All refusal conditions in SequenceState::discard_branch were
        // checked above; keeping this as its authority preserves its logical
        // branch/result cleanup instead of duplicating that bookkeeping here.
        self.state
            .discard_branch(branch)
            .expect("validated child branch");
        Ok(())
    }

    fn check_branch_transaction(&self, branch: BranchId, txn: StateTransactionId) -> Result<()> {
        self.check_transaction_on(branch, txn)
    }

    fn begin_branch(&mut self, branch: BranchId) -> Result<StateTransactionId> {
        self.child_storage(branch)?;
        self.state.frontiers(branch)?;
        let txn = self.state.begin(branch)?;
        self.state
            .open
            .get_mut(&txn)
            .expect("opened child journal")
            .sampler_len = 0;
        let rows = self.child_storage(branch)?.rows;
        self.child_storage_mut(branch)?.tentative_base = Some(rows);
        Ok(txn)
    }

    fn commit_branch(&mut self, branch: BranchId, txn: StateTransactionId, n: u64) -> Result<()> {
        self.check_branch_transaction(branch, txn)?;
        let accepted = self.state.frontiers(branch)?.accepted;
        self.check_frontier(accepted, n)?;
        let result = (|| {
            self.state.accept(branch, n)?;
            self.state.commit_prefix(txn, 0)
        })();
        if result.is_err() {
            self.abort_branch(branch, txn)
                .expect("validated child transaction");
        } else {
            self.child_storage_mut(branch)?.tentative_base = None;
        }
        result
    }

    fn abort_branch(&mut self, branch: BranchId, txn: StateTransactionId) -> Result<()> {
        self.check_branch_transaction(branch, txn)?;
        self.state.abort(txn)?;
        let rows = self.state.frontiers(branch)?.executed as usize;
        self.truncate_child(branch, rows);
        self.child_storage_mut(branch)?.tentative_base = None;
        Ok(())
    }

    fn rollback_branch(&mut self, branch: BranchId, prefix: u64) -> Result<()> {
        let child = self.child_storage(branch)?;
        self.check_retained_at_with_floor(child.high_water, prefix, Some(&child.retained_from))?;
        self.state.rollback_to(branch, prefix, &[])?;
        let rows = self.state.frontiers(branch)?.executed as usize;
        self.truncate_child(branch, rows);
        self.child_storage_mut(branch)?.tentative_base = None;
        Ok(())
    }

    fn truncate_child(&mut self, branch: BranchId, rows: usize) {
        let mut storage = self
            .children
            .remove(&branch)
            .expect("validated child storage");
        let current = storage.rows;
        truncate_storage(
            &self.layout,
            &self.geometry,
            &mut storage.backing,
            current,
            rows,
        );
        storage.rows = rows;
        self.children.insert(branch, storage);
    }

    fn append_child(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        position: u64,
        layers: &[KvRow<'_>],
        cancelled: &AtomicBool,
    ) -> Result<()> {
        self.append_child_checked(branch, txn, position, layers, || {
            if cancelled.load(Ordering::Relaxed) {
                Err(Error::Cancelled {
                    at: "paged branch append",
                })
            } else {
                Ok(())
            }
        })
    }

    fn append_child_checked(
        &mut self,
        branch: BranchId,
        txn: StateTransactionId,
        position: u64,
        layers: &[KvRow<'_>],
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.check_branch_transaction(branch, txn)?;
        let result = (|| {
            checkpoint()?;
            let (current, base) = {
                let child = self.child_storage(branch)?;
                (
                    child.rows,
                    child.tentative_base.expect("validated child transaction"),
                )
            };
            if position != current as u64 {
                return Err(invalid(
                    "position",
                    "append must start at the child executed frontier",
                ));
            }
            self.check_frontier(position, 1)?;
            if self.reclaims() && current + 1 - base > self.geometry.tentative_rows {
                return Err(invalid(
                    "tentative_rows",
                    format!(
                        "this child transaction has appended {} row(s) and a reclaiming \
                         sequence admits {}; commit before appending more",
                        current - base,
                        self.geometry.tentative_rows
                    ),
                ));
            }
            if layers.len() != self.layout.layers.len()
                || layers
                    .iter()
                    .zip(&self.layout.layers)
                    .any(|(r, l)| r.key.len() != l.key_bytes || r.value.len() != l.value_bytes)
            {
                return Err(invalid(
                    "kv_rows",
                    "every layer must supply exactly one complete K/V row of its own width",
                ));
            }
            let layout = &self.layout;
            let geometry = &self.geometry;
            let child = self
                .children
                .get_mut(&branch)
                .expect("validated child storage");
            let row = child.rows;
            child.rows += 1;
            child.high_water = child.high_water.max(child.rows);
            for (layer, values) in layers.iter().enumerate() {
                let (key, value) = row_ranges(layout, geometry, &child.backing, layer, row);
                child.backing.bytes_mut()[key].copy_from_slice(values.key);
                child.backing.bytes_mut()[value].copy_from_slice(values.value);
                checkpoint()?;
            }
            self.state.execute(branch, 1)?;
            checkpoint()?;
            Ok(())
        })();
        if result.is_err() {
            self.abort_branch(branch, txn)
                .expect("the validated child transaction");
        }
        result
    }

    fn check_transaction(&self, txn: StateTransactionId) -> Result<()> {
        self.check_transaction_on(ROOT, txn)
    }

    fn check_transaction_on(&self, branch: BranchId, txn: StateTransactionId) -> Result<()> {
        let Some(journal) = self.state.open.get(&txn) else {
            return Err(invalid(
                "transaction",
                "no such open transaction on this sequence",
            ));
        };
        if journal.branch != branch {
            return Err(invalid(
                "transaction",
                "transaction belongs to another paged branch",
            ));
        }
        Ok(())
    }

    fn update_history(&mut self, f: impl FnOnce(&mut History, &mut [u8]) -> Result<()>) {
        let s = self.sampler.as_mut().expect("sampling session");
        let start = self.layout.backing_bytes;
        let end = start + s.history.layout().state_bytes();
        f(&mut s.history, &mut self.backing.bytes_mut()[start..end])
            .expect("validated participant transition");
    }
    pub fn history(&self, committed: bool) -> Result<HistoryView<'_>> {
        let s = self
            .sampler
            .as_ref()
            .ok_or_else(|| invalid("sampler", "no sampling session"))?;
        let start = self.layout.backing_bytes;
        s.history.view(
            &self.backing.bytes()[start..start + s.history.layout().state_bytes()],
            committed,
        )
    }
    /// Additional bytes beyond the existing paged usage envelope.
    pub fn sampling_bytes(&self) -> (usize, usize, usize) {
        self.sampler.as_ref().map_or((0, 0, 0), |s| {
            (
                s.history.layout().state_bytes(),
                s.history.layout().workspace_bytes(),
                size_of::<StateKind>(),
            )
        })
    }
    fn workspace(&self) -> &[u8] {
        let s = self.sampler.as_ref().expect("sampling session");
        &self.backing.bytes()[self.layout.backing_bytes + s.history.layout().state_bytes()..]
    }
    /// Synthetic producers name the materialized prefix explicitly. This API
    /// checks storage frontiers; it does not certify model-output provenance.
    pub fn prepare_sample(
        &mut self,
        txn: StateTransactionId,
        prefix: u64,
        logits: &[f32],
        legal: Option<&[bool]>,
        temperature: f64,
    ) -> Result<PreparedSample<'_>> {
        self.check_transaction(txn)?;
        let s = self
            .sampler
            .as_ref()
            .ok_or_else(|| invalid("sampler", "no sampling session"))?;
        let f = self.state.frontiers(ROOT)?;
        let logical = f
            .accepted
            .checked_add((s.history.len() - s.history.committed_len()) as u64)
            .ok_or(DimError::Overflow)?;
        if prefix == 0
            || prefix != logical
            || prefix != f.executed
            || logits.len() != s.history.layout().vocabulary()
        {
            return Err(invalid(
                "sampling_prefix",
                "logits must name the current materialized logical prefix and vocabulary",
            ));
        }
        self.check_frontier(prefix, 1)?;
        let start = self.layout.backing_bytes + s.history.layout().state_bytes();
        moxie_sampling::distribution(
            logits,
            legal,
            temperature,
            &mut self.backing.bytes_mut()[start..],
        )?;
        Ok(PreparedSample {
            sequence: self,
            txn,
            prefix,
            greedy: temperature == 0.0,
        })
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
            // Refuse before writing, so abort never has to fail. A transaction
            // longer than the admitted undo headroom would overwrite rows its
            // own abort has to put back; ADR 0014 is why this is a refusal
            // rather than a snapshot.
            let base = self.tentative_base.expect("validated open transaction");
            if self.reclaims() && self.rows + 1 - base > self.geometry.tentative_rows {
                return Err(invalid(
                    "tentative_rows",
                    format!(
                        "this transaction has appended {} row(s) and a reclaiming \
                         sequence admits {}; commit before appending more",
                        self.rows - base,
                        self.geometry.tentative_rows
                    ),
                ));
            }
            // Per layer, not one shared width: layers may disagree on both
            // head count and head dimension, and a row of the wrong layer's
            // width would otherwise be copied into the right number of bytes.
            if layers.len() != self.layout.layers.len()
                || layers
                    .iter()
                    .zip(&self.layout.layers)
                    .any(|(r, l)| r.key.len() != l.key_bytes || r.value.len() != l.value_bytes)
            {
                return Err(invalid(
                    "kv_rows",
                    "every layer must supply exactly one complete K/V row of its own width",
                ));
            }
            let row = self.rows;
            self.rows += 1;
            self.high_water = self.high_water.max(self.rows);
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

    /// Byte ranges of one row in one layer's own page pool.
    ///
    /// The page index wraps at the layer's capacity: writing row `r` reuses the
    /// slot of row `r - capacity`, which *is* the reclamation. There is no
    /// separate eviction pass, no memmove and no page free -- the pinned legacy
    /// host path erases from the front of a vector
    /// (`src/models/gemma4/gemma4_runtime.cpp:914`) while its device path
    /// already used the ring (`:1066`), and the ring is the one that costs
    /// nothing per token.
    fn ranges(&self, layer: usize, row: usize) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        self.ranges_in(&self.backing, layer, row)
    }

    fn ranges_in(
        &self,
        backing: &HostBuffer,
        layer: usize,
        row: usize,
    ) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        row_ranges(&self.layout, &self.geometry, backing, layer, row)
    }

    /// One retained row.
    ///
    /// The three outcomes are deliberately distinct. A position at or beyond
    /// the frontier has not been executed; a position below the layer's
    /// retained start was reclaimed by its window and needs re-prefill; a bad
    /// layer index is a malformed request. Collapsing the middle case into
    /// either of the others is exactly the silent wrong answer document 04
    /// forbids.
    pub fn row(&self, layer: usize, position: u64) -> Result<KvRow<'_>> {
        let source = PagedRead {
            backing: &self.backing,
            rows: self.rows,
            high_water: self.high_water,
            retained_from: None,
        };
        self.row_in(&source, layer, position)
    }

    fn row_in<'a>(&self, source: &PagedRead<'a>, layer: usize, position: u64) -> Result<KvRow<'a>> {
        if layer >= self.geometry.layers.len() {
            return Err(invalid("kv_row", "layer is outside this sequence"));
        }
        if position >= source.rows as u64 {
            return Err(invalid("kv_row", "position has not been executed"));
        }
        let start = self
            .retained_start_for(layer, source.rows, source.high_water)
            .max(source.retained_from.map_or(0, |from| from[layer]));
        if position < start as u64 {
            return Err(Error::Reclaimed {
                layer: layer as u32,
                position,
                retained_from: start as u64,
            });
        }
        let (key, value) = self.ranges_in(source.backing, layer, position as usize);
        Ok(KvRow {
            key: &source.backing.bytes()[key],
            value: &source.backing.bytes()[value],
        })
    }

    /// Copy one bounded logical block in row order for an explicit host-backed
    /// consumer. The ring remains the authority for placement and retention;
    /// this only materializes the requested rows without exposing its physical
    /// page layout to an executor.
    pub fn read_block(&self, layer: usize, first: u64, rows: u64) -> Result<(Vec<u8>, Vec<u8>)> {
        let source = PagedRead {
            backing: &self.backing,
            rows: self.rows,
            high_water: self.high_water,
            retained_from: None,
        };
        self.read_block_in(&source, layer, first, rows)
    }

    fn read_block_in(
        &self,
        source: &PagedRead<'_>,
        layer: usize,
        first: u64,
        rows: u64,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        if rows == 0 || rows > self.geometry.page_tokens as u64 {
            return Err(invalid(
                "rows",
                "a host-backed block must contain one to one page of rows",
            ));
        }
        let end = first.checked_add(rows).ok_or(DimError::Overflow)?;
        if end > source.rows as u64 {
            return Err(invalid(
                "rows",
                "the requested block exceeds the executed frontier",
            ));
        }
        let layer_layout = self
            .layout
            .layers
            .get(layer)
            .ok_or_else(|| invalid("layer", "layer is outside this sequence"))?;
        let rows_usize = usize::try_from(rows).map_err(|_| DimError::Overflow)?;
        let key_len = layer_layout
            .key_bytes
            .checked_mul(rows_usize)
            .ok_or(DimError::Overflow)?;
        let value_len = layer_layout
            .value_bytes
            .checked_mul(rows_usize)
            .ok_or(DimError::Overflow)?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(key_len)
            .map_err(|_| Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: key_len as u64,
                available_bytes: 0,
            })?;
        values
            .try_reserve_exact(value_len)
            .map_err(|_| Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: value_len as u64,
                available_bytes: 0,
            })?;
        for position in first..end {
            let row = self.row_in(source, layer, position)?;
            keys.extend_from_slice(row.key);
            values.extend_from_slice(row.value);
        }
        Ok((keys, values))
    }

    fn truncate(&mut self, rows: usize) {
        let current = self.rows;
        truncate_storage(
            &self.layout,
            &self.geometry,
            &mut self.backing,
            current,
            rows,
        );
        self.rows = rows;
    }

    /// Close even when a transaction was abandoned. This storage is entirely
    /// synchronous; there is no device work to drain. Wrong-ledger failure
    /// returns the live sequence so cleanup can be retried.
    pub fn close(mut self, ledger: &mut Ledger) -> std::result::Result<(), Box<PagedCloseRefused>> {
        if !self.children.is_empty() {
            return Err(Box::new(PagedCloseRefused {
                sequence: self,
                error: invalid(
                    "branch",
                    "discard child branches before closing the paged sequence",
                ),
            }));
        }
        if let Err(error) = self.backing.release(ledger) {
            return Err(Box::new(PagedCloseRefused {
                sequence: self,
                error,
            }));
        }
        Ok(())
    }
}

impl<'a> PagedBranch<'a> {
    pub fn begin(&mut self) -> Result<StateTransactionId> {
        self.sequence.begin_branch(self.branch)
    }

    pub fn commit_prefix(&mut self, txn: StateTransactionId, n: u64) -> Result<()> {
        self.sequence.commit_branch(self.branch, txn, n)
    }

    pub fn abort(&mut self, txn: StateTransactionId) -> Result<()> {
        self.sequence.abort_branch(self.branch, txn)
    }

    pub fn rollback_to(&mut self, prefix: u64) -> Result<()> {
        self.sequence.rollback_branch(self.branch, prefix)
    }

    pub fn append(
        &mut self,
        txn: StateTransactionId,
        position: u64,
        layers: &[KvRow<'_>],
        cancelled: &AtomicBool,
    ) -> Result<()> {
        self.sequence
            .append_child(self.branch, txn, position, layers, cancelled)
    }

    pub fn retained_range(&self, layer: usize) -> Result<std::ops::Range<u64>> {
        if layer >= self.sequence.geometry.layers.len() {
            return Err(invalid("kv_row", "layer is outside this sequence"));
        }
        let child = self.sequence.child_storage(self.branch)?;
        Ok(self
            .sequence
            .retained_start_for(layer, child.rows, child.high_water)
            .max(child.retained_from[layer]) as u64..child.rows as u64)
    }

    pub fn usage(&self) -> Result<PagedUsage> {
        let child = self.sequence.child_storage(self.branch)?;
        Ok(self
            .sequence
            .usage_for_branch(child.rows, child.high_water, Some(&child.retained_from)))
    }

    pub fn row(&self, layer: usize, position: u64) -> Result<KvRow<'_>> {
        let child = self.sequence.child_storage(self.branch)?;
        let source = PagedRead {
            backing: &child.backing,
            rows: child.rows,
            high_water: child.high_water,
            retained_from: Some(&child.retained_from),
        };
        self.sequence.row_in(&source, layer, position)
    }

    pub fn read_block(&self, layer: usize, first: u64, rows: u64) -> Result<(Vec<u8>, Vec<u8>)> {
        let child = self.sequence.child_storage(self.branch)?;
        let source = PagedRead {
            backing: &child.backing,
            rows: child.rows,
            high_water: child.high_water,
            retained_from: Some(&child.retained_from),
        };
        self.sequence.read_block_in(&source, layer, first, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_memory::CapacitySnapshot;
    use moxie_types::{BatchId, PagePlacement, PageView, PagedKvWriter, Scope};

    /// A writer that records the placements it was handed instead of copying
    /// anything: these cross-check tests compare the device authority's
    /// placement mapping against the host ring's, they do not drive a device.
    #[derive(Default)]
    struct RecordingWriter {
        placements: Vec<PagePlacement>,
    }

    impl PagedKvWriter for RecordingWriter {
        fn write_layer(
            &mut self,
            _layer: usize,
            _batch: BatchId,
            _view: PageView,
            placements: &[PagePlacement],
        ) -> Result<()> {
            self.placements = placements.to_vec();
            Ok(())
        }

        fn publish_view(&mut self, _layer: usize, _view: PageView) -> Result<()> {
            Ok(())
        }
    }

    fn fork_ledger() -> Ledger {
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 14).unwrap()]).unwrap()
    }

    fn append_test_row(
        sequence: &mut PagedSequence,
        txn: StateTransactionId,
        position: u64,
        seed: u8,
    ) {
        let key = [seed, seed.wrapping_add(1)];
        let value = [seed.wrapping_add(2), seed.wrapping_add(3)];
        sequence
            .append(
                txn,
                position,
                &[KvRow {
                    key: &key,
                    value: &value,
                }],
                &AtomicBool::new(false),
            )
            .unwrap();
    }

    #[test]
    fn eager_fork_isolates_parent_bytes_across_child_divergence_and_releases_child() {
        let geometry = KvGeometry::uniform(1, 1, 1, 1, Precision::Bf16, 2, 16);
        let mut ledger = fork_ledger();
        let mut sequence = PagedSequence::new(&mut ledger, geometry).unwrap();
        sequence.append_prompt(4).unwrap();
        let txn = sequence.begin().unwrap();
        for position in 0..4 {
            append_test_row(&mut sequence, txn, position, 10 + position as u8);
        }
        sequence.commit_prefix(txn, 0).unwrap();

        let parent_before: Vec<_> = (0..4)
            .map(|position| {
                let row = sequence.row(0, position).unwrap();
                (row.key.to_vec(), row.value.to_vec())
            })
            .collect();
        let frontiers_before = sequence.state.frontiers(ROOT).unwrap();
        let lineage_before = sequence.state.branches[&ROOT].lineage.clone();
        assert!(matches!(
            sequence.fork(&mut ledger, 5),
            Err(Error::InvalidRequest { field: "at", .. })
        ));
        let charge_before = ledger.scope_committed(Scope::Host);
        let usage = sequence.usage();
        let branch_control = sequence.branch_control_bytes().unwrap();
        let branch = sequence.fork(&mut ledger, 4).unwrap();
        assert_eq!(
            ledger.scope_committed(Scope::Host),
            charge_before + (usage.backing_bytes + branch_control) as u64
        );

        {
            let child = sequence.branch(branch).unwrap();
            let child_prefix: Vec<_> = (0..4)
                .map(|position| {
                    let row = child.row(0, position).unwrap();
                    (row.key.to_vec(), row.value.to_vec())
                })
                .collect();
            assert_eq!(child_prefix, parent_before);
        }

        {
            let mut child = sequence.branch(branch).unwrap();
            let txn = child.begin().unwrap();
            let key = [90, 91];
            let value = [92, 93];
            child
                .append(
                    txn,
                    4,
                    &[KvRow {
                        key: &key,
                        value: &value,
                    }],
                    &AtomicBool::new(false),
                )
                .unwrap();
            child.commit_prefix(txn, 0).unwrap();
            assert_eq!(child.row(0, 4).unwrap().key, &key);

            // Truncate the first child continuation, then diverge from the
            // same committed prefix with different physical bytes.
            child.rollback_to(4).unwrap();
            let txn = child.begin().unwrap();
            let key = [120, 121];
            let value = [122, 123];
            child
                .append(
                    txn,
                    4,
                    &[KvRow {
                        key: &key,
                        value: &value,
                    }],
                    &AtomicBool::new(false),
                )
                .unwrap();
            child.commit_prefix(txn, 0).unwrap();
            assert_eq!(child.row(0, 4).unwrap().key, &key);

            let child_prefix: Vec<_> = (0..4)
                .map(|position| {
                    let row = child.row(0, position).unwrap();
                    (row.key.to_vec(), row.value.to_vec())
                })
                .collect();
            assert_eq!(child_prefix, parent_before);
        }

        let parent_after_child: Vec<_> = (0..4)
            .map(|position| {
                let row = sequence.row(0, position).unwrap();
                (row.key.to_vec(), row.value.to_vec())
            })
            .collect();
        assert_eq!(parent_after_child, parent_before);
        assert_eq!(sequence.state.frontiers(ROOT).unwrap(), frontiers_before);
        assert_eq!(sequence.state.branches[&ROOT].lineage, lineage_before);

        // The root may continue independently after the child diverges.
        let txn = sequence.begin().unwrap();
        append_test_row(&mut sequence, txn, 4, 200);
        sequence.commit_prefix(txn, 0).unwrap();
        let parent_prefix_after_continue: Vec<_> = (0..4)
            .map(|position| {
                let row = sequence.row(0, position).unwrap();
                (row.key.to_vec(), row.value.to_vec())
            })
            .collect();
        assert_eq!(parent_prefix_after_continue, parent_before);
        assert_eq!(
            sequence.state.branches[&ROOT].lineage[..=4],
            lineage_before[..=4]
        );

        sequence.discard_branch(&mut ledger, branch).unwrap();
        assert_eq!(ledger.scope_committed(Scope::Host), charge_before);
        assert_eq!(sequence.state.branch_ids(), vec![ROOT]);
        sequence.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }

    #[test]
    fn fork_refuses_a_reclaimed_window_but_accepts_its_retained_boundary() {
        let geometry = KvGeometry {
            layers: vec![LayerKv {
                kv_heads: 1,
                key_dim: 1,
                value_dim: 1,
                retention: Retention::Window { window: 2 },
            }],
            precision: Precision::Bf16,
            page_tokens: 2,
            max_tokens: 8,
            tentative_rows: 2,
        };
        let mut ledger = fork_ledger();
        let mut sequence = PagedSequence::new(&mut ledger, geometry).unwrap();
        sequence.append_prompt(8).unwrap();
        for start in (0..8).step_by(2) {
            let txn = sequence.begin().unwrap();
            append_test_row(&mut sequence, txn, start, start as u8);
            append_test_row(&mut sequence, txn, start + 1, start as u8 + 1);
            sequence.commit_prefix(txn, 0).unwrap();
        }
        assert_eq!(sequence.retained_range(0).unwrap().start, 6);
        assert!(matches!(
            sequence.fork(&mut ledger, 5),
            Err(Error::Reclaimed { .. })
        ));
        let branch = sequence.fork(&mut ledger, 8).unwrap();
        {
            let child = sequence.branch(branch).unwrap();
            assert_eq!(child.retained_range(0).unwrap(), 6..8);
            assert_eq!(child.row(0, 6).unwrap().key, &[6, 7]);
            assert!(matches!(child.row(0, 5), Err(Error::Reclaimed { .. })));
        }
        sequence.discard_branch(&mut ledger, branch).unwrap();
        sequence.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }

    #[test]
    fn injected_mid_fork_failure_restores_parent_bytes_state_and_charge() {
        let geometry = KvGeometry::uniform(1, 1, 1, 1, Precision::Bf16, 2, 16);
        let mut ledger = fork_ledger();
        let mut sequence = PagedSequence::new(&mut ledger, geometry).unwrap();
        sequence.append_prompt(4).unwrap();
        let txn = sequence.begin().unwrap();
        for position in 0..4 {
            append_test_row(&mut sequence, txn, position, 30 + position as u8);
        }
        sequence.commit_prefix(txn, 0).unwrap();
        let parent_before: Vec<_> = (0..4)
            .map(|position| {
                let row = sequence.row(0, position).unwrap();
                (row.key.to_vec(), row.value.to_vec())
            })
            .collect();
        let frontiers_before = sequence.state.frontiers(ROOT).unwrap();
        let lineage_before = sequence.state.branches[&ROOT].lineage.clone();
        let charge_before = ledger.scope_committed(Scope::Host);
        let mut boundary = 0;
        let result = sequence.fork_checked(&mut ledger, 4, || {
            let fail = boundary == 3;
            boundary += 1;
            if fail {
                Err(Error::Cancelled {
                    at: "injected post-logical-fork cleanup",
                })
            } else {
                Ok(())
            }
        });
        assert!(result.is_err());
        assert_eq!(boundary, 4);
        assert!(sequence.children.is_empty());
        assert_eq!(sequence.state.branch_ids(), vec![ROOT]);
        assert_eq!(ledger.scope_committed(Scope::Host), charge_before);
        assert_eq!(sequence.state.frontiers(ROOT).unwrap(), frontiers_before);
        assert_eq!(sequence.state.branches[&ROOT].lineage, lineage_before);
        let parent_after: Vec<_> = (0..4)
            .map(|position| {
                let row = sequence.row(0, position).unwrap();
                (row.key.to_vec(), row.value.to_vec())
            })
            .collect();
        assert_eq!(parent_after, parent_before);
        sequence.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }

    #[test]
    fn cancelled_commit_restores_frontiers_counts_and_rows_at_every_boundary() {
        for accept in 0..=3 {
            for fail_at in 0..3 {
                let mut ledger =
                    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()])
                        .unwrap();
                let g = KvGeometry::uniform(1, 1, 1, 1, Precision::Bf16, 2, 16);
                let mut s = PagedSequence::with_sampling(&mut ledger, g, 3, 8, 77).unwrap();
                let rows = [KvRow {
                    key: &[1, 2],
                    value: &[3, 4],
                }];
                let cancel = AtomicBool::new(false);
                let txn = s.begin().unwrap();
                s.append_prompt(1).unwrap();
                s.append(txn, 0, &rows, &cancel).unwrap();
                // Include one pre-existing generated count in the restore target.
                s.prepare_sample(txn, 1, &[0.; 3], None, 1.)
                    .unwrap()
                    .stage(&cancel)
                    .unwrap();
                s.append(txn, 1, &rows, &cancel).unwrap();
                s.commit_prefix(txn, 1).unwrap();
                let end = s.layout.backing_bytes
                    + s.sampler.as_ref().unwrap().history.layout().state_bytes();
                let bytes = s.backing.bytes()[..end].to_vec();
                let frontiers = s.state.frontiers(ROOT).unwrap();
                let lineage = s.state.branches[&ROOT].lineage.clone();
                let txn = s.begin().unwrap();
                for prefix in 2..5 {
                    s.prepare_sample(txn, prefix, &[0.; 3], None, 1.)
                        .unwrap()
                        .stage(&cancel)
                        .unwrap();
                    s.append(txn, prefix, &rows, &cancel).unwrap();
                }
                let mut boundary = 0;
                let result = s.commit_checked(txn, accept, || {
                    let fail = boundary == fail_at;
                    boundary += 1;
                    if fail {
                        Err(Error::Cancelled {
                            at: "injected commit boundary",
                        })
                    } else {
                        Ok(())
                    }
                });
                assert!(result.is_err());
                assert_eq!(s.state.frontiers(ROOT).unwrap(), frontiers);
                assert_eq!(s.state.branches[&ROOT].lineage, lineage);
                assert_eq!(&s.backing.bytes()[..end], &bytes);
                assert_eq!(s.history(true).unwrap().len(), 1);
                assert_eq!(s.history(false).unwrap().len(), 1);
                assert!(s.state.open_transactions().is_empty());
                s.close(&mut ledger).unwrap();
                assert!(ledger.outstanding().is_empty());
            }
        }
    }

    #[test]
    fn sampling_faults_after_each_mutation_restore_all_participants() {
        for v in [3, 7] {
            for start in [2, 3] {
                // Entry, every layer copy and frontier update; then separately
                // entry/after count+history mutation in the sampling path.
                for sample_fault in [false, true] {
                    let boundaries = if sample_fault { 2 } else { 5 };
                    for fail_at in 0..boundaries {
                        let mut ledger =
                            Ledger::new([
                                CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()
                            ])
                            .unwrap();
                        let g = KvGeometry::uniform(3, 1, 2, 1, Precision::Bf16, 2, 32);
                        let mut s =
                            PagedSequence::with_sampling(&mut ledger, g, v, 16, 77).unwrap();
                        let rows = [KvRow {
                            key: &[1, 2, 3, 4],
                            value: &[5, 6],
                        }; 3];
                        let cancel = AtomicBool::new(false);
                        let txn = s.begin().unwrap();
                        s.append_prompt(start).unwrap();
                        for p in 0..start {
                            s.append(txn, p, &rows, &cancel).unwrap();
                        }
                        s.commit_prefix(txn, 0).unwrap();
                        let end = s.layout.backing_bytes
                            + s.sampler.as_ref().unwrap().history.layout().state_bytes();
                        let bytes = s.backing.bytes()[..end].to_vec();
                        let f = s.state.frontiers(ROOT).unwrap();
                        let lineage = s.state.branches[&ROOT].lineage.clone();
                        let charge = ledger.scope_committed(Scope::Host);
                        let txn = s.begin().unwrap();
                        s.prepare_sample(txn, start, &vec![0.; v], None, 1.)
                            .unwrap()
                            .stage(&cancel)
                            .unwrap();
                        s.append(txn, start, &rows, &cancel).unwrap();
                        let mut boundary = 0;
                        let checkpoint = || {
                            let fail = boundary == fail_at;
                            boundary += 1;
                            if fail {
                                Err(Error::Cancelled {
                                    at: "injected sampler/paged mutation",
                                })
                            } else {
                                Ok(())
                            }
                        };
                        if sample_fault {
                            assert!(
                                s.prepare_sample(txn, start + 1, &vec![0.; v], None, 1.)
                                    .unwrap()
                                    .stage_checked(checkpoint)
                                    .is_err()
                            );
                        } else {
                            s.prepare_sample(txn, start + 1, &vec![0.; v], None, 1.)
                                .unwrap()
                                .stage(&cancel)
                                .unwrap();
                            assert!(s.append_checked(txn, start + 1, &rows, checkpoint).is_err());
                        }
                        assert_eq!(&s.backing.bytes()[..end], &bytes);
                        assert_eq!(s.state.frontiers(ROOT).unwrap(), f);
                        assert_eq!(s.state.branches[&ROOT].lineage, lineage);
                        assert!(s.state.open_transactions().is_empty());
                        assert!(s.history(false).unwrap().is_empty());
                        assert!(s.history(true).unwrap().is_empty());
                        assert_eq!(ledger.scope_committed(Scope::Host), charge);
                        s.close(&mut ledger).unwrap();
                        assert!(ledger.outstanding().is_empty());
                    }
                }
            }
        }
    }

    /// Task 0038: the device authority places a row where this store does.
    ///
    /// The device path claims that its page table **is** this ring, written
    /// down. That claim is checked against this store's own table bytes rather
    /// than against a restatement of its formula: the placement's physical page
    /// is used to read this sequence's page-table entry, and the byte offset
    /// that produces must be the one `ranges` returns. Two implementations of
    /// one mapping is the failure this asserts away.
    #[test]
    fn the_device_authority_places_rows_where_this_store_does() {
        use crate::device::DeviceKvSequence;

        // Retain-all, so both sides admit the same page count: a windowed layer
        // deliberately differs, and that difference is asserted below.
        let geometry = KvGeometry::uniform(2, 1, 2, 2, Precision::Bf16, 4, 32);
        let mut ledger =
            Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()]).unwrap();
        let mut host = PagedSequence::new(&mut ledger, geometry.clone()).unwrap();
        let mut device = DeviceKvSequence::new(geometry.clone()).unwrap();

        let rows = [KvRow {
            key: &[1, 2, 3, 4],
            value: &[5, 6, 7, 8],
        }; 2];
        // Twenty-four rows of an admitted thirty-two: six whole pages and a
        // wrap of neither, which is what makes the comparison about placement
        // rather than about the boundary.
        let txn = host.begin().unwrap();
        host.append_prompt(24).unwrap();
        let device_txn = device.begin().unwrap();
        for position in 0..24u64 {
            host.append(txn, position, &rows, &AtomicBool::new(false))
                .unwrap();
        }
        // A performer would copy through `write_layer`; this test only needs
        // the mapping each layer's writer was handed.
        let mut writers: Vec<RecordingWriter> = (0..geometry.layers.len())
            .map(|_| RecordingWriter::default())
            .collect();
        let mut writer_refs: Vec<&mut dyn PagedKvWriter> = writers
            .iter_mut()
            .map(|w| w as &mut dyn PagedKvWriter)
            .collect();
        device.append(device_txn, 24, &mut writer_refs).unwrap();
        // Zero, because prompt tokens are accepted when they are appended:
        // accepting them again would count the same context twice.
        host.commit_prefix(txn, 0).unwrap();
        let mut writer_refs: Vec<&mut dyn PagedKvWriter> = writers
            .iter_mut()
            .map(|w| w as &mut dyn PagedKvWriter)
            .collect();
        device.commit(device_txn, 24, &mut writer_refs).unwrap();
        let device_placements: Vec<Vec<PagePlacement>> =
            writers.into_iter().map(|w| w.placements).collect();

        for (layer, placements) in device_placements.iter().enumerate() {
            let l = &host.layout.layers[layer];
            for position in 0..24u64 {
                let (key, value) = host.ranges(layer, position as usize);
                let placement = device.placement_of(layer, position).unwrap();
                // The same row through the staged batch's own placements: one
                // mapping, asked two ways.
                let staged_run = placements
                    .iter()
                    .find(|p| p.position <= position && position < p.position + p.rows)
                    .expect("every staged row is in a run");
                assert_eq!(
                    placement.physical_page, staged_run.physical_page,
                    "layer {layer} row {position}: placement and staged run disagree"
                );
                // Through this store's *own* page table, not through a second
                // copy of its arithmetic.
                let entry = (l.table_base + placement.physical_page as usize) * 8;
                let table = &host.backing.bytes()[entry..entry + 8];
                let base = u64::from_le_bytes(table.try_into().unwrap()) as usize;
                assert_eq!(
                    key.start,
                    base + placement.slot as usize * l.key_bytes,
                    "layer {layer} row {position}: the two mappings disagree on the key"
                );
                assert_eq!(
                    value.start,
                    base + geometry.page_tokens * l.key_bytes
                        + placement.slot as usize * l.value_bytes,
                    "layer {layer} row {position}: the two mappings disagree on the value"
                );
            }
        }

        // The one place they differ, on purpose: a windowed layer's device
        // capacity carries an extra page, because a device page is evicted
        // whole and the oldest row the window admits must survive that.
        let mut windowed = geometry.clone();
        windowed.layers[0].retention = Retention::Window { window: 8 };
        windowed.tentative_rows = 4;
        let device = DeviceKvSequence::new(windowed.clone()).unwrap();
        let mut ledger =
            Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()]).unwrap();
        let host = PagedSequence::new(&mut ledger, windowed).unwrap();
        assert_eq!(
            device.layout(0).unwrap().pages,
            host.layout.layers[0].pages as u64 + 1,
            "the device layer must admit exactly one page more than the host ring"
        );
    }

    /// The same comparison **across a wrap**, which is where a ring can differ.
    ///
    /// The retain-all case above cannot wrap: its capacity is its context. So
    /// this one gives both stores four pages — the host by windowing 24 rows
    /// with 8 of headroom, the device by windowing 16 with the same 8 and the
    /// eviction page it adds — and appends a hundred rows through a 32-row
    /// ring. Equal page counts mean one modulus, and every row after the first
    /// wrap is a row whose physical page is *not* its logical one.
    #[test]
    fn the_two_mappings_still_agree_after_the_ring_has_wrapped() {
        use crate::device::DeviceKvSequence;

        const ROWS: u64 = 100;
        let layer = |window: usize| LayerKv {
            kv_heads: 1,
            key_dim: 2,
            value_dim: 2,
            retention: Retention::Window { window },
        };
        let make = |window: usize| KvGeometry {
            layers: vec![layer(window)],
            precision: Precision::Bf16,
            page_tokens: 8,
            max_tokens: 4096,
            tentative_rows: 8,
        };
        // 24 + 8 is four pages for the host; 16 + 8 is three plus the eviction
        // page for the device. Both admit 32 rows in four 8-row pages.
        let mut ledger =
            Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()]).unwrap();
        let mut host = PagedSequence::new(&mut ledger, make(24)).unwrap();
        let mut device = DeviceKvSequence::new(make(16)).unwrap();
        assert_eq!(host.layout.layers[0].pages, 4);
        assert_eq!(device.layout(0).unwrap().pages, 4);
        assert_eq!(host.layout.layers[0].capacity, 32);

        let rows = [KvRow {
            key: &[1, 2, 3, 4],
            value: &[5, 6, 7, 8],
        }];
        let mut written = 0u64;
        while written < ROWS {
            let step = 8.min(ROWS - written);
            let txn = host.begin().unwrap();
            host.append_prompt(step).unwrap();
            for offset in 0..step {
                host.append(txn, written + offset, &rows, &AtomicBool::new(false))
                    .unwrap();
            }
            host.commit_prefix(txn, 0).unwrap();
            let device_txn = device.begin().unwrap();
            let mut writer = RecordingWriter::default();
            device
                .append(
                    device_txn,
                    step,
                    &mut [&mut writer as &mut dyn PagedKvWriter],
                )
                .unwrap();
            device
                .commit(
                    device_txn,
                    step,
                    &mut [&mut writer as &mut dyn PagedKvWriter],
                )
                .unwrap();
            written += step;
        }
        assert_eq!(device.committed_rows().unwrap(), ROWS);

        // Every row, including the ones whose page has been reused twelve
        // times. The comparison is against this store's own table bytes, so a
        // device mapping that had drifted would land on different bytes here.
        let l = &host.layout.layers[0];
        let mut wrapped = 0usize;
        for position in 0..ROWS {
            let (key, value) = host.ranges(0, position as usize);
            let logical_page = position / 8;
            let physical = (position / 8) % 4;
            if logical_page != physical {
                wrapped += 1;
            }
            let entry = (l.table_base + physical as usize) * 8;
            let table = &host.backing.bytes()[entry..entry + 8];
            let base = u64::from_le_bytes(table.try_into().unwrap()) as usize;
            let slot = (position % 8) as usize;
            assert_eq!(
                key.start,
                base + slot * l.key_bytes,
                "row {position}: the two mappings disagree on the key after a wrap"
            );
            assert_eq!(
                value.start,
                base + 8 * l.key_bytes + slot * l.value_bytes,
                "row {position}: the two mappings disagree on the value after a wrap"
            );
        }
        assert!(
            wrapped > 60,
            "only {wrapped} rows were on a reused page; this fixture is not wrapping"
        );

        // And the device's own placements agree with that, for every row it
        // still retains.
        let retained = device.retained(0).unwrap();
        assert!(
            retained.start > 0,
            "the device ring was supposed to reclaim"
        );
        for position in retained {
            let placement = device.placement_of(0, position).unwrap();
            assert_eq!(placement.physical_page, (position / 8) % 4);
            assert_eq!(placement.slot, position % 8);
        }
    }

    #[test]
    fn cancellation_at_every_publication_boundary_restores_physical_and_logical_state() {
        let geometry = KvGeometry::uniform(3, 1, 2, 1, Precision::Bf16, 2, 8);
        // Entry, each layer's K/V copy, and frontier publication. Exercise both
        // a partial existing page and the first row on a new page.
        for start in [1, 2, 3] {
            for fail_at in 0..geometry.layers.len() + 2 {
                let mut ledger =
                    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap()])
                        .unwrap();
                let mut sequence = PagedSequence::new(&mut ledger, geometry.clone()).unwrap();
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

    /// Which publishing operation the injected fault interrupts.
    #[derive(Debug, Clone, Copy)]
    enum Fault {
        Append,
        Sample,
        Commit,
    }

    /// The contract's abort gate, on a ring that has actually wrapped.
    ///
    /// The three tests above inject a fault at every publication boundary but
    /// all use full retention, where no byte is ever overwritten; the window
    /// tests exercise a wrapped ring but abort only after completed appends.
    /// Neither on its own covers the case the tentative bound exists for:
    /// a fault part-way through publishing a row whose slot has already been
    /// reused, on a sequence whose layers disagree about width and retention,
    /// with sampler history and lineage to restore alongside the pages.
    ///
    /// Whole-buffer equality is deliberately not the assertion here. A wrapped
    /// ring's headroom legitimately holds bytes from rows nothing can read any
    /// more, so the claim is over the **readable** range of every layer.
    #[test]
    fn faults_at_every_boundary_of_a_wrapped_mixed_geometry_restore_all_participants() {
        let geometry = KvGeometry {
            layers: vec![
                LayerKv {
                    kv_heads: 2,
                    key_dim: 2,
                    value_dim: 1,
                    retention: Retention::All,
                },
                LayerKv {
                    kv_heads: 1,
                    key_dim: 3,
                    value_dim: 2,
                    retention: Retention::Window { window: 3 },
                },
            ],
            precision: Precision::Bf16,
            page_tokens: 2,
            max_tokens: 40,
            tentative_rows: 2,
        };
        let row = |position: usize, layer: usize, key: bool, width: usize| -> Vec<u8> {
            (0..width * 2)
                .map(|c| ((position * 31 + layer * 17 + c * 5 + usize::from(key) * 97) % 256) as u8)
                .collect()
        };
        let readable = |s: &PagedSequence| -> Vec<(usize, u64, Vec<u8>, Vec<u8>)> {
            let mut out = Vec::new();
            for layer in 0..s.layer_count() {
                for position in s.retained_range(layer).unwrap() {
                    let r = s.row(layer, position).unwrap();
                    out.push((layer, position, r.key.to_vec(), r.value.to_vec()));
                }
            }
            out
        };

        // Three fault sites, because three separate operations publish on a
        // wrapped ring and each has to undo the others' partial work:
        //
        //   Append: entry, each layer's two copies, the frontier.
        //   Sample: entry, and after the count and history mutation.
        //   Commit: entry, after the accepted frontier moves, and after the
        //           committed sampler history is published.
        //
        // Commit is exercised here against a windowed geometry, not only a
        // full-retention one: it is the only operation that advances the
        // *accepted* frontier and publishes sampler history, and a
        // full-retention geometry never overwrites a row, so it alone cannot
        // show a fault there unwinding correctly.
        for fault in [Fault::Append, Fault::Sample, Fault::Commit] {
            let boundaries = match fault {
                Fault::Append => 4,
                Fault::Sample => 2,
                Fault::Commit => 3,
            };
            for fail_at in 0..boundaries {
                let mut ledger =
                    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 14).unwrap()])
                        .unwrap();
                let mut s =
                    PagedSequence::with_sampling(&mut ledger, geometry.clone(), 3, 20, 11).unwrap();
                let cancel = AtomicBool::new(false);
                let append = |s: &mut PagedSequence, txn, position: usize| {
                    let k0 = row(position, 0, true, 4);
                    let v0 = row(position, 0, false, 2);
                    let k1 = row(position, 1, true, 3);
                    let v1 = row(position, 1, false, 2);
                    s.append(
                        txn,
                        position as u64,
                        &[
                            KvRow {
                                key: &k0,
                                value: &v0,
                            },
                            KvRow {
                                key: &k1,
                                value: &v1,
                            },
                        ],
                        &AtomicBool::new(false),
                    )
                    .unwrap();
                };
                // One prompt token, then generated tokens two at a time -- the
                // full admitted headroom -- until the windowed layer's ring of
                // ceil((3 + 2)/2)*2 = 6 rows has wrapped more than twice.
                let txn = s.begin().unwrap();
                s.append_prompt(1).unwrap();
                append(&mut s, txn, 0);
                s.commit_prefix(txn, 0).unwrap();
                let mut position = 1;
                while position < 15 {
                    let txn = s.begin().unwrap();
                    for _ in 0..2 {
                        s.prepare_sample(txn, position as u64, &[0.25, 0.5, 0.25], None, 1.)
                            .unwrap()
                            .stage(&cancel)
                            .unwrap();
                        append(&mut s, txn, position);
                        position += 1;
                    }
                    s.commit_prefix(txn, 2).unwrap();
                }
                assert!(s.retained_range(1).unwrap().start > 0, "the ring must wrap");
                assert_eq!(s.retained_range(0).unwrap().start, 0);

                let before_rows = readable(&s);
                let frontiers = s.state.frontiers(ROOT).unwrap();
                let lineage = s.state.lineage_at(ROOT, position as u64).unwrap();
                let history: Vec<_> = s.history(true).unwrap().entries(0).collect();
                let charge = ledger.scope_committed(Scope::Host);

                let txn = s.begin().unwrap();
                s.prepare_sample(txn, position as u64, &[0.25, 0.5, 0.25], None, 1.)
                    .unwrap()
                    .stage(&cancel)
                    .unwrap();
                append(&mut s, txn, position);
                let mut boundary = 0;
                let mut checkpoint = || {
                    let fail = boundary == fail_at;
                    boundary += 1;
                    if fail {
                        Err(Error::Cancelled {
                            at: "injected wrapped-ring boundary",
                        })
                    } else {
                        Ok(())
                    }
                };
                match fault {
                    Fault::Sample => {
                        assert!(
                            s.prepare_sample(
                                txn,
                                position as u64 + 1,
                                &[0.25, 0.5, 0.25],
                                None,
                                1.
                            )
                            .unwrap()
                            .stage_checked(&mut checkpoint)
                            .is_err()
                        );
                    }
                    Fault::Append => {
                        s.prepare_sample(txn, position as u64 + 1, &[0.25, 0.5, 0.25], None, 1.)
                            .unwrap()
                            .stage(&cancel)
                            .unwrap();
                        let k0 = row(position + 1, 0, true, 4);
                        let v0 = row(position + 1, 0, false, 2);
                        let k1 = row(position + 1, 1, true, 3);
                        let v1 = row(position + 1, 1, false, 2);
                        assert!(
                            s.append_checked(
                                txn,
                                position as u64 + 1,
                                &[
                                    KvRow {
                                        key: &k0,
                                        value: &v0
                                    },
                                    KvRow {
                                        key: &k1,
                                        value: &v1
                                    },
                                ],
                                &mut checkpoint,
                            )
                            .is_err()
                        );
                    }
                    Fault::Commit => {
                        // A full, legal transaction -- two staged tokens and
                        // two appended rows, exactly the admitted headroom --
                        // failing while it publishes. This is the only path
                        // that moves the accepted frontier and the committed
                        // sampler history, and it has to undo both plus every
                        // row the appends overwrote.
                        s.prepare_sample(txn, position as u64 + 1, &[0.25, 0.5, 0.25], None, 1.)
                            .unwrap()
                            .stage(&cancel)
                            .unwrap();
                        append(&mut s, txn, position + 1);
                        assert!(s.commit_checked(txn, 2, &mut checkpoint).is_err());
                    }
                }

                // Every readable row on both layers, both frontiers, the
                // lineage, the committed sampler history and the ledger.
                assert_eq!(readable(&s), before_rows, "{fault:?} fault at {fail_at}");
                assert_eq!(s.state.frontiers(ROOT).unwrap(), frontiers);
                assert_eq!(s.state.lineage_at(ROOT, position as u64).unwrap(), lineage);
                assert_eq!(
                    s.history(true).unwrap().entries(0).collect::<Vec<_>>(),
                    history
                );
                assert!(s.history(false).unwrap().len() == history.len());
                assert!(s.state.open_transactions().is_empty());
                assert_eq!(ledger.scope_committed(Scope::Host), charge);

                // And it is still usable: the same append succeeds on retry.
                let retry = s.begin().unwrap();
                append(&mut s, retry, position);
                s.commit_prefix(retry, 0).unwrap();
                s.close(&mut ledger).unwrap();
                assert!(ledger.outstanding().is_empty());
            }
        }
    }
}
