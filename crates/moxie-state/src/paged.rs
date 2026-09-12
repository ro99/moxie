//! Appendable host KV pages bound to the accepted sequence transaction journal.
//!
//! This is storage, not an attention executor. All pages belong exclusively to
//! one root branch. A fixed envelope is admitted before use; growth, COW and
//! device views require later capability qualification.

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
        // The facade permits one branch, one open journal and no retained
        // results. Lineage is reserved exactly once. A conservative node bound
        // for the pinned Rust BTreeMap (11 entries, 12 edges) covers each map's
        // single node, including padding/header, even when the branch has only
        // one entry. No per-token map entries can accumulate here.
        let node_bound = |entry: usize| 11 * (entry + size_of::<usize>()) + 16 * size_of::<usize>();
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
                + node_bound(size_of::<(BranchId, Branch)>())
                + node_bound(size_of::<(StateTransactionId, Journal)>())
                + node_bound(size_of::<(crate::ResultId, crate::LogitsHandle)>()),
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
        geometry: KvGeometry,
        sampling: Option<(SamplingLayout, u64)>,
    ) -> Result<Self> {
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
        let evicted = self
            .high_water
            .saturating_sub(self.layout.layers[layer].capacity);
        match self.geometry.layers[layer].retention.window() {
            // `capacity >= max_tokens >= high_water`, so `evicted` is zero.
            None => evicted,
            Some(window) => self.rows.saturating_sub(window).max(evicted),
        }
    }

    pub fn usage(&self) -> PagedUsage {
        let mut live_pages = 0;
        let mut live_page_bytes = 0;
        let mut retained_rows = 0;
        let mut logical_kv_bytes = 0;
        for (index, layer) in self.layout.layers.iter().enumerate() {
            let pages = self
                .rows
                .div_ceil(self.geometry.page_tokens)
                .min(layer.pages);
            live_pages += pages;
            live_page_bytes += pages * layer.page_bytes;
            let retained = self.rows - self.retained_start(index);
            retained_rows += retained;
            logical_kv_bytes += retained * (layer.key_bytes + layer.value_bytes);
        }
        PagedUsage {
            rows: self.rows,
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
        for (index, layer) in self.geometry.layers.iter().enumerate() {
            let Some(window) = layer.retention.window() else {
                continue;
            };
            let evicted =
                self.high_water
                    .saturating_sub(self.layout.layers[index].capacity) as u64;
            let wanted = prefix.saturating_sub(window as u64);
            if wanted < evicted {
                return Err(Error::Reclaimed {
                    layer: index as u32,
                    position: wanted,
                    retained_from: evicted,
                });
            }
        }
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
        let l = &self.layout.layers[layer];
        let page = (row / self.geometry.page_tokens) % l.pages;
        let local = row % self.geometry.page_tokens;
        let entry = (l.table_base + page) * 8;
        let table = &self.backing.bytes()[entry..entry + 8];
        let base = u64::from_le_bytes(table.try_into().expect("eight-byte entry")) as usize;
        let key = base + local * l.key_bytes;
        let value = base + self.geometry.page_tokens * l.key_bytes + local * l.value_bytes;
        (key..key + l.key_bytes, value..value + l.value_bytes)
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
        if layer >= self.geometry.layers.len() {
            return Err(invalid("kv_row", "layer is outside this sequence"));
        }
        if position >= self.rows as u64 {
            return Err(invalid("kv_row", "position has not been executed"));
        }
        let start = self.retained_start(layer);
        if position < start as u64 {
            return Err(Error::Reclaimed {
                layer: layer as u32,
                position,
                retained_from: start as u64,
            });
        }
        let (key, value) = self.ranges(layer, position as usize);
        Ok(KvRow {
            key: &self.backing.bytes()[key],
            value: &self.backing.bytes()[value],
        })
    }

    fn truncate(&mut self, rows: usize) {
        for layer in 0..self.geometry.layers.len() {
            // Only the last `capacity` discarded rows have slots of their own;
            // anything older shares a slot with a row already zeroed here.
            // Zeroing a slot can clear the bytes of the row `capacity` below
            // it, which was already overwritten by the row being cleared and so
            // is below every retained start -- nothing readable is lost.
            let capacity = self.layout.layers[layer].capacity;
            for row in rows.max(self.rows.saturating_sub(capacity))..self.rows {
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
}
