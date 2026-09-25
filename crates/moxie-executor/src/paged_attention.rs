//! Binding and launch for the common BF16 paged attention.
//!
//! Task 0037. The kernel and its catalogue identities belong to
//! `moxie-kernels`; what is here is what only this crate may do — check that a
//! launch's geometry, page mapping and visibility are the ones a descriptor
//! serves, turn admitted ranges into addresses, and wait on the event that says
//! the answer exists.
//!
//! **What this module is and is not.** `PagedAttentionRun` does hold pages
//! between launches and does track how many rows it has observed **written** —
//! that is what "persistent device state" means physically, and it is a fact
//! about copied bytes, not a claim about history. What it does not do is
//! *decide* anything: no retention rule, no transaction, no branch, no
//! truncation, no lineage, no publication. Those are `moxie-state`'s: it calls
//! through [`moxie_types::PagedKvWriter`], handing a writer the batch and the
//! placements it chose for one layer, and treats that layer published only
//! when the writer's `write_layer` returns `Ok` — which this module's writer
//! implementation may say only once the copy it drives has been observed
//! complete. There is no value this module hands back that would let a caller
//! make publication happen some other way. Admission is `moxie-memory`'s
//! throughout: runtime payload and metadata buffers are charged before they
//! are allocated. The short-lived vectors used to construct an admission are
//! control bookkeeping, not run resources.
//!
//! **The device-handle path document 04 requires.** `attend_into` takes device
//! query and output ranges and launches directly, with no host round trip;
//! `attend` is the explicit separate host-staged implementation built on top of
//! it, for a caller with host bytes rather than a plan's own activation arena.
//! There is one launch, in `attend_into`; the two entry points differ in what
//! happens before and after it, not in how the kernel is invoked.
//!
//! **Why a separate module from `affine_linear.rs`.** Different operation,
//! different operand set — a paged payload plus a page table rather than a
//! weight's three components — and, more to the point, a different *shape of
//! error*: the questions this file has to answer are about positions,
//! visibility and page identity, which the quantized linear has none of.
//! Folding them together would produce one file whose only shared property is
//! the word "launch".

use moxie_plan::Visibility;
use moxie_types::{
    ActivationPrecision, DeviceCapability, DimError, Error, KernelCatalogue, KernelOperand,
    Precision, Result, SemanticKernelDescriptor, SemanticKernelOp,
};

/// BF16 is two bytes, and this slice serves BF16 only.
const PAYLOAD_BYTES: u64 = 2;
/// A page-table entry is one `u32`.
const PAGE_ENTRY_BYTES: u64 = 4;

/// Compose prose fallibly. An empty detail is a shorter true statement, not an
/// abort: the variant and the `&'static str` field are what a caller branches
/// on either way. `format!` **aborts** when an allocation fails, and the
/// context that produces a refusal is exactly the context most likely to
/// coincide with memory pressure, so growth goes through
/// [`moxie_memory::fallible::text`] -- the crate's one fallible formatter --
/// rather than a second copy of it.
fn fallible(args: core::fmt::Arguments<'_>) -> String {
    moxie_memory::fallible::text(args).unwrap_or_default()
}

/// [`Error::InvalidRequest`] with a fixed message, still composed **fallibly**.
///
/// `detail.into()` on a `&str` allocates, and it allocates infallibly: a literal
/// message is not free just because it is constant. Every refusal in this module
/// goes through the sink above, whether or not its prose has anything to
/// interpolate — which is the difference between "no `format!` calls" and "no
/// allocation that can abort".
fn invalid(field: &'static str, detail: &str) -> Error {
    invalid_fmt(field, format_args!("{detail}"))
}

/// [`Error::InvalidRequest`] whose prose is composed **fallibly**.
fn invalid_fmt(field: &'static str, detail: core::fmt::Arguments<'_>) -> Error {
    Error::InvalidRequest {
        field,
        detail: fallible(detail),
    }
}

/// [`Error::Unsupported`] whose prose is composed **fallibly**.
fn unsupported_fmt(capability: &'static str, reason: core::fmt::Arguments<'_>) -> Error {
    Error::Unsupported {
        capability,
        reason: fallible(reason),
    }
}

/// [`Error::UnsupportedKernel`] whose prose is composed **fallibly**.
fn unsupported_kernel_fmt(operation: &'static str, detail: core::fmt::Arguments<'_>) -> Error {
    Error::UnsupportedKernel {
        operation,
        detail: fallible(detail),
    }
}

/// A zeroed buffer whose allocation can fail rather than abort.
#[cfg_attr(not(feature = "driver"), allow(dead_code))]
fn try_zeroed(len: usize) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::Pageable)),
            requested_bytes: len as u64,
            available_bytes: 0,
        })?;
    out.resize(len, 0);
    Ok(out)
}

/// One layer's paged key/value geometry, as a launch sees it.
///
/// This describes bytes that already exist, not a policy for making them: how
/// many physical pages were admitted, how wide a page is, and how a row is laid
/// out inside one. The *logical* history — where it starts, how long it is —
/// belongs to the step, because it changes with every append while this does
/// not.
///
/// Key and value rows share `head_dim` in this slice. A layer whose value width
/// differs from its key width is MLA-shaped, which task 0037 excludes; the
/// refusal is explicit rather than a silent reinterpretation of the value
/// payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageGeometry {
    pub kv_heads: u64,
    pub head_dim: u64,
    /// Rows per page. Storage, not scheduling: the kernel's online-softmax tile
    /// is [`moxie_kernels::PAGED_ATTENTION_TILE`] and the two are deliberately
    /// independent.
    pub page_tokens: u64,
    /// Physical pages this layer was admitted for.
    pub pages: u64,
}

impl PageGeometry {
    pub fn check(&self) -> Result<()> {
        if self.kv_heads == 0 || self.head_dim == 0 || self.page_tokens == 0 || self.pages == 0 {
            return Err(invalid_fmt(
                "page_geometry",
                format_args!(
                    "{} kv head(s) of {} in pages of {} row(s), {} page(s): none may be zero",
                    self.kv_heads, self.head_dim, self.page_tokens, self.pages
                ),
            ));
        }
        if self.head_dim > moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM {
            return Err(unsupported_fmt(
                "attention_head_dim",
                format_args!(
                    "head dimension {} exceeds the {} this image serves",
                    self.head_dim,
                    moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM
                ),
            ));
        }
        Ok(())
    }

    /// Elements in one stored row: every key/value head, side by side.
    pub fn row_elements(&self) -> Result<u64> {
        self.kv_heads
            .checked_mul(self.head_dim)
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// Bytes in one page of keys, which is also one page of values.
    pub fn page_bytes(&self) -> Result<u64> {
        self.row_elements()?
            .checked_mul(self.page_tokens)
            .and_then(|e| e.checked_mul(PAYLOAD_BYTES))
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// Bytes for the whole admitted key payload, which is also the value one.
    ///
    /// Two separate allocations of this size, not one: keys and values are
    /// distinct operands with distinct addresses, and interleaving them would
    /// make a value read depend on a key stride.
    pub fn payload_bytes(&self) -> Result<u64> {
        self.page_bytes()?
            .checked_mul(self.pages)
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// Bytes for the reusable little-endian page-table upload buffer.
    ///
    /// Page-table validation is performed in place, so the upload buffer is
    /// the only host workspace this run needs for publication. It is allocated
    /// during admission and reused for every publication.
    pub fn page_table_upload_bytes(&self) -> Result<u64> {
        self.pages
            .checked_mul(PAGE_ENTRY_BYTES)
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// The physical rows these pages hold.
    pub fn capacity_rows(&self) -> Result<u64> {
        self.page_tokens
            .checked_mul(self.pages)
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// The element offset of one logical row's key or value block.
    ///
    /// The host arithmetic the kernel performs per row, exposed so a test can
    /// check the two agree rather than assume it. `physical` comes from the
    /// page table; this function does not consult one, because resolving a page
    /// identity is the state owner's job and reproducing it here would be the
    /// second page table.
    pub fn row_offset(&self, physical_page: u64, slot: u64) -> Result<u64> {
        if slot >= self.page_tokens {
            return Err(invalid_fmt(
                "slot",
                format_args!("slot {slot} in a page of {} row(s)", self.page_tokens),
            ));
        }
        if physical_page >= self.pages {
            return Err(invalid_fmt(
                "physical_page",
                format_args!(
                    "physical page {physical_page} of {} admitted page(s)",
                    self.pages
                ),
            ));
        }
        physical_page
            .checked_mul(self.page_tokens)
            .and_then(|r| r.checked_add(slot))
            .and_then(|r| r.checked_mul(self.row_elements().ok()?))
            .ok_or(Error::Dim(DimError::Overflow))
    }
}

/// One layer's attention semantics: what does not change between launches.
///
/// Head geometry, the declared score scale and the visibility rule belong to
/// the layer; how many query rows a launch carries and where they sit belong to
/// the launch. Separating them is what lets a decode row and a prefill chunk be
/// built from one description without restating the parameters that must not
/// differ between them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionLayer {
    pub geometry: PageGeometry,
    /// Query heads. Equal to `geometry.kv_heads` for multi-head attention, a
    /// multiple of it for grouped-query attention.
    pub heads: u64,
    /// The layer's **declared** score scale, not a derived one. Gemma-style
    /// layers normalize queries and keys per head and declare exactly 1.0.
    pub scale: f32,
    pub visibility: Visibility,
}

/// One launch of the attention kernel: query rows against a visible history.
///
/// Whole prefill, one prefill chunk and a single decode row are the same value
/// with a different row count, which is the point of having one kernel.
///
/// **The fields are private and there is no way to build one except through
/// [`PagedAttentionLaunch::new`] or [`PagedAttentionLaunch::at`].** A value of
/// this type *is* the evidence that its geometry, positions and ABI widths were
/// checked, exactly as `AffineLaunch` is for the quantized linear. Public
/// fields on a checked struct mean the check happened once, to a value nobody
/// has to keep: an instance edited after `check` — or built by a caller who
/// never called it — would carry positions that `allows` adds without
/// overflowing and that the kernel indexes with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedAttentionLaunch {
    layer: AttentionLayer,
    rows: u64,
    first_position: u64,
    history_base: u64,
    history_rows: u64,
    /// Explicitly admitted two-block streaming metadata. A normal launch may
    /// never declare more history than its device geometry holds; this bit is
    /// set only by [`Self::two_block_stream`], whose executor splits that
    /// history before touching the single-shot kernel.
    two_block_stream: bool,
    /// Number of staged blocks in a generalized host-backed launch. Zero is a
    /// normal device-resident launch; the value is checked against the
    /// history geometry at construction.
    staged_blocks: u64,
}

impl PagedAttentionLaunch {
    /// One launch, checked. Every other constructor goes through this.
    pub fn new(
        layer: AttentionLayer,
        rows: u64,
        first_position: u64,
        history_base: u64,
        history_rows: u64,
    ) -> Result<Self> {
        let launch = Self {
            layer,
            rows,
            first_position,
            history_base,
            history_rows,
            two_block_stream: false,
            staged_blocks: 0,
        };
        launch.check()?;
        Ok(launch)
    }

    /// One explicit two-block host-backed launch.
    ///
    /// The resident block is exactly one device page and the staged block is a
    /// nonempty tail page. The returned value describes the complete logical
    /// history even when that history is larger than the run's device
    /// geometry; only [`device::PagedAttentionRun::attend_two_block`] accepts
    /// this marker, and it launches the separate partial ABI once per block.
    pub fn two_block_stream(
        layer: AttentionLayer,
        rows: u64,
        first_position: u64,
        history_base: u64,
        staged_rows: u64,
    ) -> Result<Self> {
        let resident_rows = layer.geometry.page_tokens;
        if staged_rows == 0 || staged_rows > layer.geometry.page_tokens {
            return Err(invalid(
                "staged_rows",
                "the staged block must contain one to one page of rows",
            ));
        }
        let history_rows = resident_rows
            .checked_add(staged_rows)
            .ok_or(Error::Dim(DimError::Overflow))?;
        let launch = Self {
            layer,
            rows,
            first_position,
            history_base,
            history_rows,
            two_block_stream: true,
            staged_blocks: 1,
        };
        launch.check()?;
        Ok(launch)
    }

    /// One bounded host-backed launch with one resident page and a nonempty
    /// sequence of staged pages. The executor reuses one staging buffer for
    /// the derived number of blocks; no block count is multiplied into the
    /// device allocation.
    pub fn n_block_stream(
        layer: AttentionLayer,
        rows: u64,
        first_position: u64,
        history_base: u64,
        history_rows: u64,
    ) -> Result<Self> {
        layer.geometry.check()?;
        if history_rows <= layer.geometry.page_tokens {
            return Err(invalid(
                "history_rows",
                "an N-block stream must contain one resident page and at least one staged row",
            ));
        }
        let staged_blocks =
            (history_rows - layer.geometry.page_tokens).div_ceil(layer.geometry.page_tokens);
        let launch = Self {
            layer,
            rows,
            first_position,
            history_base,
            history_rows,
            two_block_stream: false,
            staged_blocks,
        };
        launch.check()?;
        Ok(launch)
    }

    /// The same layer and the same history, a different query block.
    ///
    /// What a chunked prefill needs, and the reason it cannot be a field
    /// assignment: moving the query block changes which keys are visible, so it
    /// is re-checked rather than edited in place.
    pub fn at(&self, rows: u64, first_position: u64) -> Result<Self> {
        if self.is_host_stream() {
            return Err(invalid(
                "launch",
                "a host-backed stream has one fixed full-history launch",
            ));
        }
        Self::new(
            self.layer,
            rows,
            first_position,
            self.history_base,
            self.history_rows,
        )
    }

    /// The same launch over a different committed history.
    pub fn over(&self, history_base: u64, history_rows: u64) -> Result<Self> {
        if self.is_host_stream() {
            return Err(invalid(
                "launch",
                "a host-backed stream has one fixed full-history launch",
            ));
        }
        Self::new(
            self.layer,
            self.rows,
            self.first_position,
            history_base,
            history_rows,
        )
    }

    pub const fn layer(&self) -> &AttentionLayer {
        &self.layer
    }

    pub const fn geometry(&self) -> &PageGeometry {
        &self.layer.geometry
    }

    pub const fn heads(&self) -> u64 {
        self.layer.heads
    }

    pub const fn scale(&self) -> f32 {
        self.layer.scale
    }

    pub const fn visibility(&self) -> Visibility {
        self.layer.visibility
    }

    #[cfg(feature = "driver")]
    pub(crate) const fn is_two_block_stream(&self) -> bool {
        self.two_block_stream
    }

    pub(crate) const fn is_host_stream(&self) -> bool {
        self.staged_blocks != 0
    }

    pub const fn staged_blocks(&self) -> u64 {
        self.staged_blocks
    }

    pub const fn rows(&self) -> u64 {
        self.rows
    }

    pub const fn first_position(&self) -> u64 {
        self.first_position
    }

    pub const fn history_base(&self) -> u64 {
        self.history_base
    }

    pub const fn history_rows(&self) -> u64 {
        self.history_rows
    }

    /// Check every geometric, positional and ABI precondition this launch has.
    ///
    /// Private, because the only values of this type are ones that passed it.
    /// Admission re-derives the scalars it needs rather than re-checking, for
    /// the reason `affine_linear::descriptor_serves` exists: a check applied to
    /// a value nobody keeps is a check that happened to something else.
    fn check(&self) -> Result<()> {
        self.layer.geometry.check()?;
        if self.layer.heads == 0
            || !self
                .layer
                .heads
                .is_multiple_of(self.layer.geometry.kv_heads)
        {
            // The same grouping rule the graph validates and the oracle
            // enforces. A ratio that does not divide gives one group more query
            // heads than another, which is not a layout any released checkpoint
            // uses and not one this launch will invent.
            return Err(invalid_fmt(
                "heads",
                format_args!(
                    "{} query head(s) must be a nonzero multiple of {} key/value head(s)",
                    self.layer.heads, self.layer.geometry.kv_heads
                ),
            ));
        }
        if !(self.layer.scale.is_finite() && self.layer.scale > 0.0) {
            return Err(invalid_fmt(
                "attention_scale",
                format_args!(
                    "score scale must be finite and positive, got {}",
                    self.layer.scale
                ),
            ));
        }
        if let Visibility::SlidingWindow { window } = self.layer.visibility {
            if window == 0 {
                return Err(invalid(
                    "visibility",
                    "a sliding window of zero sees nothing, including the query's own position",
                ));
            }
            // Every ABI width is checked where the value is constructed, so a
            // launch that exists can always be expressed to the kernel.
            if u32::try_from(window).is_err() {
                return Err(unsupported_fmt(
                    "attention_window",
                    format_args!("a window of {window} rows exceeds this ABI's u32"),
                ));
            }
        }
        if self.rows == 0 {
            return Err(invalid("rows", "a launch with no query row"));
        }
        if !self
            .history_base
            .is_multiple_of(self.layer.geometry.page_tokens)
        {
            return Err(invalid_fmt(
                "history_base",
                format_args!(
                    "history base {} is not a whole number of {}-row pages; a partially \
                     reclaimed page has no stable slot for its rows",
                    self.history_base, self.layer.geometry.page_tokens
                ),
            ));
        }
        if self.history_rows == 0 {
            return Err(invalid(
                "history_rows",
                "attending over an empty history: the caller appends this row's key and \
                 value before attending, because a causal query attends to itself",
            ));
        }
        if !self.is_host_stream() && self.history_rows > self.layer.geometry.capacity_rows()? {
            return Err(Error::CapacityExceeded {
                tier: None,
                requested_bytes: self.history_rows,
                available_bytes: self.layer.geometry.capacity_rows()?,
            });
        }
        if self.is_host_stream() {
            let expected_blocks = (self.history_rows - self.layer.geometry.page_tokens)
                .div_ceil(self.layer.geometry.page_tokens);
            if self.staged_blocks != expected_blocks {
                return Err(invalid(
                    "history_rows",
                    "the host-backed block count must match the staged history tail",
                ));
            }
        }
        // Every other scalar the launch hands the kernel, at the same width the
        // ABI declares. `grid` and `window` cannot fail after this.
        u32::try_from(self.rows).map_err(|_| {
            invalid_fmt(
                "rows",
                format_args!("{} query rows exceed a u32 launch grid", self.rows),
            )
        })?;
        for (value, field) in [
            (self.layer.heads, "heads"),
            (self.layer.geometry.kv_heads, "kv_heads"),
            (self.layer.geometry.head_dim, "head_dim"),
            (self.layer.geometry.page_tokens, "page_tokens"),
        ] {
            u32::try_from(value).map_err(|_| {
                invalid_fmt(
                    field,
                    format_args!("{value} exceeds this ABI's u32 {field}"),
                )
            })?;
        }
        let history_end = self
            .history_base
            .checked_add(self.history_rows)
            .ok_or(Error::Dim(DimError::Overflow))?;
        let last_query = self
            .first_position
            .checked_add(self.rows)
            .ok_or(Error::Dim(DimError::Overflow))?
            .checked_sub(1)
            .expect("rows is nonzero");
        if self.first_position < self.history_base {
            return Err(invalid_fmt(
                "first_position",
                format_args!(
                    "query position {} precedes the retained history, which starts at {}",
                    self.first_position, self.history_base
                ),
            ));
        }
        if last_query >= history_end {
            // The append-before-attend rule, and the gap check with it: a
            // causal query attends to itself, so its own key must already be
            // stored. A query beyond the frontier would silently attend to a
            // shorter history rather than to nothing.
            return Err(invalid_fmt(
                "first_position",
                format_args!(
                    "query positions {}..={last_query} run past a history holding [{}, \
                     {history_end}); append before attending",
                    self.first_position, self.history_base
                ),
            ));
        }
        Ok(())
    }

    /// Logical pages the page table must describe.
    pub fn logical_pages(&self) -> Result<u64> {
        Ok(self.history_rows.div_ceil(self.layer.geometry.page_tokens))
    }

    /// Bytes of page table this launch reads.
    pub fn page_table_bytes(&self) -> Result<u64> {
        self.logical_pages()?
            .checked_mul(PAGE_ENTRY_BYTES)
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// Bytes of query, which is also the byte count of the output.
    pub fn query_bytes(&self) -> Result<u64> {
        self.rows
            .checked_mul(self.layer.heads)
            .and_then(|r| r.checked_mul(self.layer.geometry.head_dim))
            .and_then(|r| r.checked_mul(PAYLOAD_BYTES))
            .ok_or(Error::Dim(DimError::Overflow))
    }

    /// Bytes of output, which is the query's count: one value row per query row
    /// per head.
    pub fn output_bytes(&self) -> Result<u64> {
        self.query_bytes()
    }

    /// The window parameter the kernel takes: zero for full causal visibility.
    ///
    /// Zero is not "no window" by convention alone — a window of zero is
    /// refused at construction precisely so the value can carry this meaning.
    pub fn window(&self) -> Result<u32> {
        match self.layer.visibility {
            Visibility::Causal => Ok(0),
            Visibility::SlidingWindow { window } => u32::try_from(window).map_err(|_| {
                unsupported_fmt(
                    "attention_window",
                    format_args!("a window of {window} rows exceeds this ABI's u32"),
                )
            }),
        }
    }

    /// Whether query row `row` may attend to the stored row at absolute
    /// position `key`.
    ///
    /// The host's copy of the kernel's decision, over **absolute** positions,
    /// so a test can compare the two rather than trust one. It defers to
    /// `Visibility::allows` instead of re-deriving the rule: R21 is what
    /// happens when two implementations of the same mask exist.
    pub fn allows(&self, row: u64, key: u64) -> bool {
        // A row outside this launch has no position, and saturating is not an
        // answer: `check` has already proved that `first_position + rows` and
        // `history_base + history_rows` are representable, so the only way to
        // reach an overflow here is to ask about a row this launch does not
        // have. That question has one true answer and it is `false`.
        if row >= self.rows {
            return false;
        }
        let position = self.first_position + row;
        let end = self.history_base + self.history_rows;
        key >= self.history_base && key < end && self.layer.visibility.allows(position, key)
    }

    /// The launch grid: one block per query row and head.
    pub fn grid(&self) -> Result<(u32, u32, u32)> {
        let x = u32::try_from(self.rows).map_err(|_| {
            invalid_fmt(
                "rows",
                format_args!("{} query rows exceed a u32 launch grid", self.rows),
            )
        })?;
        let y = u32::try_from(self.layer.heads).map_err(|_| {
            invalid_fmt(
                "heads",
                format_args!("{} heads exceed a u32 launch grid", self.layer.heads),
            )
        })?;
        Ok((x, y, 1))
    }
}

/// Choose the paged attention descriptor for this device and launch.
///
/// Selection is by semantic capability, operand roles, architecture and shape —
/// never by which model asked. One descriptor must match: two would mean the
/// catalogue had grown a second implementation of the same operation for the
/// same hardware, which is the thing selection exists to prevent.
pub fn select_paged_attention_kernel(
    catalogue: &KernelCatalogue,
    capability: &DeviceCapability,
    launch: &PagedAttentionLaunch,
) -> Result<SemanticKernelDescriptor> {
    launch.check()?;
    let mut chosen: Option<&SemanticKernelDescriptor> = None;
    let mut matched = 0usize;
    let mut first_rejection: Option<(&'static str, &SemanticKernelDescriptor)> = None;
    for descriptor in catalogue.descriptors() {
        if descriptor.operation != SemanticKernelOp::PagedAttention
            || descriptor.sm.major != capability.compute_major
            || descriptor.sm.minor != capability.compute_minor
        {
            continue;
        }
        match descriptor_mismatch(descriptor, launch) {
            None => {
                matched += 1;
                if chosen.is_none() {
                    chosen = Some(descriptor);
                }
            }
            Some(reason) => {
                if first_rejection.is_none() {
                    first_rejection = Some((reason, descriptor));
                }
            }
        }
    }
    if matched == 0
        && let Some((reason, descriptor)) = first_rejection
    {
        return Err(unsupported_kernel_fmt(
            "paged_attention",
            format_args!(
                "{} does not serve this launch: {reason}",
                descriptor.id.0.as_str()
            ),
        ));
    }
    let Some(descriptor) = chosen.filter(|_| matched == 1) else {
        return Err(unsupported_kernel_fmt(
            "paged_attention",
            format_args!(
                "expected exactly one paged attention descriptor for {} row(s) of {} head(s) \
                 by {} on sm_{}{}; found {matched}",
                launch.rows,
                launch.heads(),
                launch.geometry().head_dim,
                capability.compute_major,
                capability.compute_minor,
            ),
        ));
    };
    descriptor.try_clone()
}

/// Whether one descriptor serves this launch. Selection and admission apply
/// this same predicate.
pub fn descriptor_serves(
    descriptor: &SemanticKernelDescriptor,
    launch: &PagedAttentionLaunch,
) -> Result<()> {
    launch.check()?;
    match descriptor_mismatch(descriptor, launch) {
        None => Ok(()),
        Some(reason) => Err(unsupported_kernel_fmt(
            "paged_attention",
            format_args!("{} does not serve this launch: {reason}", descriptor.id.0),
        )),
    }
}

fn descriptor_mismatch(
    descriptor: &SemanticKernelDescriptor,
    launch: &PagedAttentionLaunch,
) -> Option<&'static str> {
    if descriptor.operation != SemanticKernelOp::PagedAttention {
        return Some("it is not a paged attention descriptor");
    }
    if descriptor.abi_version != moxie_kernels::PAGED_ATTENTION_ABI {
        return Some("its ABI version is not the one this binding speaks");
    }
    let bf16 = KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16));
    // Query, keys, values, then the page table. The cache precision is part of
    // the operand list rather than a flag: an FP16 cache is unsupported by this
    // task, and "unsupported" has to be a failure to match rather than a
    // reinterpretation of the bytes.
    let expected = [bf16, bf16, bf16, KernelOperand::PageIndex];
    if descriptor.inputs.as_slice() != expected.as_slice() {
        return Some("its operand roles are not query, keys, values and a page table in BF16");
    }
    if descriptor.output != ActivationPrecision::expect(Precision::Bf16) {
        return Some("its output precision is not BF16");
    }
    if descriptor.shape.max_input < launch.geometry().head_dim
        || descriptor.shape.max_output < launch.geometry().head_dim
    {
        return Some("the head dimension exceeds the shape bounds it declares");
    }
    if descriptor.shape.max_rows < launch.rows() {
        return Some("the query row count exceeds the shape bounds it declares");
    }
    if descriptor.symbols.len() != 1 || descriptor.symbols[0].0 != moxie_kernels::PAGED_ATTENTION {
        return Some("it does not name the one paged attention symbol");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> PageGeometry {
        PageGeometry {
            kv_heads: 2,
            head_dim: 64,
            page_tokens: 32,
            pages: 8,
        }
    }

    fn layer() -> AttentionLayer {
        AttentionLayer {
            geometry: geometry(),
            heads: 8,
            scale: moxie_plan::reciprocal_sqrt_scale(64),
            visibility: Visibility::Causal,
        }
    }

    /// Four query rows ending at position 63 of a 64-row history.
    fn launch() -> PagedAttentionLaunch {
        PagedAttentionLaunch::new(layer(), 4, 60, 0, 64).expect("a legal launch")
    }

    #[test]
    fn a_launch_reports_the_bytes_its_operands_occupy() {
        let l = launch();
        // 2 kv heads of 64, 32 rows to a page, two bytes each.
        assert_eq!(l.geometry().row_elements().unwrap(), 128);
        assert_eq!(l.geometry().page_bytes().unwrap(), 128 * 32 * 2);
        assert_eq!(l.geometry().payload_bytes().unwrap(), 128 * 32 * 2 * 8);
        assert_eq!(l.geometry().capacity_rows().unwrap(), 256);
        // 64 rows of history is exactly two 32-row pages.
        assert_eq!(l.logical_pages().unwrap(), 2);
        assert_eq!(l.page_table_bytes().unwrap(), 8);
        // Four query rows of eight heads by 64, two bytes each.
        assert_eq!(l.query_bytes().unwrap(), 4 * 8 * 64 * 2);
        assert_eq!(l.output_bytes().unwrap(), l.query_bytes().unwrap());
        assert_eq!(l.grid().unwrap(), (4, 8, 1));
    }

    #[test]
    fn a_partial_final_page_is_counted_as_a_whole_page() {
        // The page tail: 65 rows is three pages, the last holding one row. A
        // page table sized by division rather than ceiling would leave the
        // final row unaddressable, which is the classic off-by-one a paged
        // cache has.
        let l = PagedAttentionLaunch::new(layer(), 1, 64, 0, 65).expect("a legal launch");
        assert_eq!(l.logical_pages().unwrap(), 3);
        assert_eq!(l.page_table_bytes().unwrap(), 12);
    }

    #[test]
    fn an_n_block_stream_counts_the_resident_page_and_staged_tail() {
        let l = PagedAttentionLaunch::n_block_stream(layer(), 1, 127, 0, 128)
            .expect("one resident page plus three staged pages");
        assert_eq!(l.staged_blocks(), 3);
        assert_eq!(l.logical_pages().unwrap(), 4);
        assert!(l.at(1, 127).is_err());
        assert!(l.over(0, 128).is_err());

        let tail = PagedAttentionLaunch::n_block_stream(layer(), 1, 96, 0, 97)
            .expect("a final partial staged page");
        assert_eq!(tail.staged_blocks(), 3);
        assert_eq!(tail.logical_pages().unwrap(), 4);
    }

    #[test]
    fn the_row_offset_is_the_arithmetic_the_kernel_performs() {
        let g = geometry();
        // Physical page 3, slot 5: (3 * 32 + 5) rows of 128 elements each.
        assert_eq!(g.row_offset(3, 5).unwrap(), (3 * 32 + 5) * 128);
        assert_eq!(g.row_offset(0, 0).unwrap(), 0);
        // Past the page, and past the admitted pages: typed refusals, not
        // addresses into another layer's bytes.
        assert!(g.row_offset(0, 32).is_err());
        assert!(g.row_offset(8, 0).is_err());
    }

    #[test]
    fn a_launch_cannot_exist_without_having_been_checked() {
        // The point of the private fields. Every refusal below is a value that
        // simply does not come into being, so no later code can be handed one
        // and trust a check that happened to something else.
        let mut illegal = layer();
        illegal.heads = 1; // fewer query heads than key/value heads
        assert!(PagedAttentionLaunch::new(illegal, 4, 60, 0, 64).is_err());
        let mut illegal = layer();
        illegal.heads = 6;
        illegal.geometry.kv_heads = 4; // a ratio that does not divide
        assert!(PagedAttentionLaunch::new(illegal, 4, 60, 0, 64).is_err());
        let mut illegal = layer();
        illegal.heads = 0;
        assert!(PagedAttentionLaunch::new(illegal, 4, 60, 0, 64).is_err());

        // Multi-head is the same value with the ratio at one.
        let mut mha = layer();
        mha.heads = 2;
        PagedAttentionLaunch::new(mha, 4, 60, 0, 64).expect("multi-head is grouped with one");
    }

    #[test]
    fn the_declared_scale_must_be_finite_and_positive() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut illegal = layer();
            illegal.scale = bad;
            assert!(
                PagedAttentionLaunch::new(illegal, 4, 60, 0, 64).is_err(),
                "scale {bad} was accepted"
            );
        }
        // 1.0 is an ordinary declared scale, not a suspicious one: a
        // Gemma-style layer normalizes its queries and keys and declares it.
        let mut gemma = layer();
        gemma.scale = 1.0;
        PagedAttentionLaunch::new(gemma, 4, 60, 0, 64).expect("a declared scale of one");
    }

    #[test]
    fn a_query_beyond_the_frontier_is_refused_rather_than_shortened() {
        // The append-before-attend rule. Attending at position 64 against a
        // 64-row history means the query's own key was never written, and the
        // kernel would then attend over 64 keys and return a confident wrong
        // answer instead of nothing.
        assert!(PagedAttentionLaunch::new(layer(), 1, 64, 0, 64).is_err());
        // The last legal position is the frontier's last row.
        let last = PagedAttentionLaunch::new(layer(), 1, 63, 0, 64).expect("the last row");
        // A chunk that starts legally and runs past the end is refused too.
        assert!(last.at(4, 62).is_err());
        assert!(last.at(2, 62).is_ok());
    }

    #[test]
    fn a_query_below_the_retained_base_is_refused() {
        // A sliding layer has reclaimed everything below `history_base`. A
        // query there cannot see its own key, and the answer is a refusal
        // rather than attention over whatever the pages now hold.
        assert!(PagedAttentionLaunch::new(layer(), 1, 31, 32, 32).is_err());
        PagedAttentionLaunch::new(layer(), 1, 32, 32, 32).expect("the first retained row");
    }

    #[test]
    fn a_history_base_inside_a_page_is_refused() {
        assert!(matches!(
            PagedAttentionLaunch::new(layer(), 1, 70, 16, 64).unwrap_err(),
            Error::InvalidRequest {
                field: "history_base",
                ..
            }
        ));
        // A whole page of reclamation is fine.
        PagedAttentionLaunch::new(layer(), 4, 60, 32, 32).expect("whole-page reclamation");
    }

    #[test]
    fn a_history_longer_than_the_admitted_pages_is_a_capacity_refusal() {
        // Not an invalid request: the geometry is legal and the bytes are not
        // there. The distinction is what lets a caller admit more pages and
        // retry rather than rewrite its request.
        assert!(matches!(
            PagedAttentionLaunch::new(layer(), 1, 256, 0, 257).unwrap_err(),
            Error::CapacityExceeded { .. }
        ));
    }

    #[test]
    fn an_empty_history_and_an_empty_launch_are_typed_errors() {
        assert!(PagedAttentionLaunch::new(layer(), 4, 60, 0, 0).is_err());
        assert!(PagedAttentionLaunch::new(layer(), 0, 60, 0, 64).is_err());
        let mut no_page = layer();
        no_page.geometry.page_tokens = 0;
        assert!(PagedAttentionLaunch::new(no_page, 4, 60, 0, 64).is_err());
        let mut no_pages = layer();
        no_pages.geometry.pages = 0;
        assert!(PagedAttentionLaunch::new(no_pages, 4, 60, 0, 64).is_err());
    }

    #[test]
    fn a_head_dimension_wider_than_the_image_serves_is_unsupported() {
        // Unsupported, not invalid: the request is coherent and this build
        // cannot serve it. A named refusal is the contract; silently reading
        // past the end of a row is what it prevents.
        let mut wide = geometry();
        wide.head_dim = moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM + 1;
        assert!(matches!(
            wide.check().unwrap_err(),
            Error::Unsupported {
                capability: "attention_head_dim",
                ..
            }
        ));
    }

    #[test]
    fn overflowing_geometry_is_refused_before_it_becomes_an_address() {
        let mut g = geometry();
        g.kv_heads = u64::MAX;
        assert!(matches!(g.row_elements(), Err(Error::Dim(_))));
        let mut g = geometry();
        g.pages = u64::MAX;
        assert!(matches!(g.payload_bytes(), Err(Error::Dim(_))));
        assert!(PagedAttentionLaunch::new(layer(), 4, u64::MAX, 0, 64).is_err());
        assert!(PagedAttentionLaunch::new(layer(), 4, 60, 0, u64::MAX - 1).is_err());
    }

    #[test]
    fn every_abi_width_is_refused_at_construction_not_at_the_launch() {
        // A window wider than the kernel's `u32` must be refused here, at
        // construction -- discovering it only inside the launch, after the
        // query copy had already been submitted, would turn a knowable refusal
        // into an unknown submission and a quarantined run. Nothing that
        // reaches a launch can fail a width conversion, so `window` and `grid`
        // are total on a value of this type.
        let mut wide = layer();
        wide.visibility = Visibility::SlidingWindow {
            window: u64::from(u32::MAX) + 1,
        };
        assert!(matches!(
            PagedAttentionLaunch::new(wide, 4, 60, 0, 64).unwrap_err(),
            Error::Unsupported {
                capability: "attention_window",
                ..
            }
        ));
        // A window of zero would be indistinguishable from causal in the ABI,
        // so it is refused at construction rather than encoded.
        let mut zero = layer();
        zero.visibility = Visibility::SlidingWindow { window: 0 };
        assert!(PagedAttentionLaunch::new(zero, 4, 60, 0, 64).is_err());

        let mut narrow = layer();
        narrow.visibility = Visibility::SlidingWindow { window: 16 };
        let l = PagedAttentionLaunch::new(narrow, 4, 60, 0, 64).expect("a legal window");
        assert_eq!(l.window().unwrap(), 16);
        assert_eq!(
            launch().window().unwrap(),
            0,
            "causal is zero and only that"
        );
        assert_eq!(l.grid().unwrap(), (4, 8, 1));
    }

    #[test]
    fn visibility_is_decided_on_absolute_positions() {
        // R21: the same layer, one chunk starting at zero and one starting
        // later, must mask on the true position rather than on the row index.
        let first = PagedAttentionLaunch::new(layer(), 4, 0, 0, 64).expect("the first chunk");
        assert!(first.allows(0, 0));
        assert!(!first.allows(0, 1), "row zero must not see the future");
        assert!(first.allows(3, 3) && first.allows(3, 0));

        let later = launch();
        assert!(later.allows(0, 60) && later.allows(0, 59));
        assert!(!later.allows(0, 61), "position 60 must not see 61");

        let mut windowed = layer();
        windowed.visibility = Visibility::SlidingWindow { window: 4 };
        let w = PagedAttentionLaunch::new(windowed, 4, 60, 0, 64).expect("a windowed launch");
        assert!(w.allows(0, 57) && !w.allows(0, 56));
        // Reclaimed rows are invisible whatever the window says.
        let w = PagedAttentionLaunch::new(windowed, 4, 60, 32, 32).expect("after reclamation");
        assert!(!w.allows(0, 31));
        // A row this launch does not have has no position, and asking is false
        // rather than an addition that overflows.
        assert!(!w.allows(4, 60));
        assert!(!w.allows(u64::MAX, 60));
    }
}

/// Admission, persistent paged operands and launch, against a real driver.
///
/// The bytes a launch reads are admitted here and stay admitted across appends
/// and decodes, because that is what "persistent device state" means
/// physically: the pages are not restaged per step. What this module does *not*
/// do is decide what those pages mean. The **written** high-water mark it
/// tracks is the count of rows whose copy has been enqueued — a
/// fact about bytes, not a retention policy, a transaction or a branch.
/// `moxie-state` owns those, and it drives this module through
/// [`moxie_types::PagedKvWriter`] rather than through a second authority grown
/// here: `PagedKvWriterAdapter` is the bridge, `write_rows` the mechanism it
/// calls for host-backed rows; the dense device writer defers observation until
/// a later run entry point.
#[cfg(feature = "driver")]
pub mod device {
    #[cfg(feature = "paged-attention-test-hooks")]
    use core::sync::atomic::{AtomicBool, Ordering};
    use core::{ffi::c_void, mem::ManuallyDrop};

    use moxie_cuda::{
        Event, Module, ModuleImage, PinnedHostBuffer, RankContext, ResolvedModule, Stream,
        TrustedImage,
    };
    use moxie_memory::{
        BufferRequest, Ledger, LedgerId, PlanRequest, Rejection, Reservation, StageSpan,
    };
    #[cfg(feature = "paged-attention-binding")]
    use moxie_state::{DeviceBranch, DeviceKvSequence};
    use moxie_types::PageView;
    #[cfg(feature = "paged-attention-binding")]
    use moxie_types::{BatchId, BranchId, PagedKvWriter};
    use moxie_types::{
        DeviceTier, DimError, Error, HostTier, PagePlacement, Result, Scope,
        SemanticKernelDescriptor, Tier,
    };

    use super::{
        PAGE_ENTRY_BYTES, PAYLOAD_BYTES, PageGeometry, PagedAttentionLaunch, invalid, invalid_fmt,
        unsupported_kernel_fmt,
    };
    use crate::arena::{DeviceArena, DeviceRange};

    /// 256-byte alignment, as every other device range in this crate uses.
    const ALIGNMENT: u64 = 256;

    #[cfg(feature = "paged-attention-test-hooks")]
    static PREFETCH_GATE_RELEASED: AtomicBool = AtomicBool::new(false);

    #[cfg(feature = "paged-attention-test-hooks")]
    unsafe extern "C" fn wait_for_prefetch_release(_: *mut c_void) {
        let start = std::time::Instant::now();
        while !PREFETCH_GATE_RELEASED.load(Ordering::Acquire)
            && start.elapsed() < std::time::Duration::from_secs(10)
        {
            std::thread::yield_now();
        }
    }

    /// Whether this run admits the buffers the host-staged path needs.
    ///
    /// The pages and the page table are persistent state and are always
    /// admitted. The query and output ranges, and the host readback that goes
    /// with them, exist **only** for [`PagedAttentionRun::attend`] — and a
    /// caller using [`PagedAttentionRun::attend_into`] brings its own, from its
    /// plan's activation arena.
    ///
    /// Admitting them unconditionally charged a direct-device caller three
    /// times: once in its own arena, once again here, and once more for a host
    /// readback nothing would read. On a device whose memory is nearly spoken
    /// for, that difference is an admissible plan being refused.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Staging {
        /// Admit query, output and readback buffers. `attend` works.
        Host,
        /// Admit only the persistent pages and table. `attend` refuses and
        /// names `attend_into`, which is the path.
        DeviceHandles,
        /// Admit one bounded host-sourced page and one reusable FP32 partial
        /// output buffer for the exact two-block streaming proof.
        TwoBlock,
        /// Admit one bounded host-sourced page and one reusable FP32 partial
        /// output buffer for a bounded sequence of staged blocks. The page and
        /// partial ranges are reused sequentially; the count is an execution
        /// bound, not a request for simultaneous device pages.
        HostBacked { max_staged_blocks: u64 },
        /// Host-backed streaming with two pinned bounce pages and two device
        /// staging pages, allowing one page to copy while the other is folded.
        HostBackedReadAhead { max_staged_blocks: u64 },
    }

    impl Staging {
        const fn partial_buffers(self) -> bool {
            matches!(
                self,
                Self::TwoBlock | Self::HostBacked { .. } | Self::HostBackedReadAhead { .. }
            )
        }

        const fn max_staged_blocks(self) -> u64 {
            match self {
                Self::TwoBlock => 1,
                Self::HostBacked { max_staged_blocks }
                | Self::HostBackedReadAhead { max_staged_blocks } => max_staged_blocks,
                Self::Host | Self::DeviceHandles => 0,
            }
        }
    }

    /// A refusal from admission, or from the unwind it runs on a partial
    /// failure.
    ///
    /// Not only "before anything was allocated": `unwind` builds one after
    /// pages, a table and query/output ranges already exist, when a later
    /// step -- the image load, module resolution, the page-table vector --
    /// fails. `error` is always admission's *own* reason for refusing, never
    /// overwritten by a failure while unwinding; that failure goes in
    /// `cleanup` instead. Whatever cleanup could not give back travels with
    /// the refusal rather than being dropped: the arena in `arena`, and every
    /// range a failed `release` did not reach -- including the one its own
    /// failure was holding -- in `ranges`.
    #[derive(Debug)]
    pub struct PagedAdmitRefused<'ctx> {
        pub error: Error,
        /// Returned held only when giving it back *also* failed.
        pub reservation: Option<Reservation>,
        pub rejection: Option<Rejection>,
        /// The arena, when a failed `close` could not hand it back to the
        /// ledger.
        pub arena: Option<DeviceArena<'ctx>>,
        /// Ranges a failed `release` could not return.
        pub ranges: Vec<DeviceRange<'ctx>>,
        /// A cleanup failure, distinct from `error`.
        pub cleanup: Option<Error>,
    }

    /// What a refusal hands back, **in the allocations the caller gave it**.
    ///
    /// Not one concatenated buffer. Joining an append's key and value rows to
    /// return them reallocates — `Vec::append` on an exact-capacity vector
    /// always does — and that allocation is infallible, on the path whose whole
    /// purpose is to report a refusal.
    #[derive(Debug)]
    pub enum RefusedSource {
        /// An append's rows, each still in the vector it arrived in.
        Rows { keys: Vec<u8>, values: Vec<u8> },
        /// A page mapping, still the `u32` entries it arrived as.
        PageTable(Vec<u32>),
        /// One launch's query rows.
        Query(Vec<u8>),
        /// The run's own admitted page-table upload buffer, held across its
        /// copy. Never handed to a caller: it returns to the run once the copy
        /// is observed complete.
        PageTableUpload(Vec<u8>),
        /// A host-backed stream's query, current host K/V page and page-table entry.
        Stream {
            query: Vec<u8>,
            keys: Vec<u8>,
            values: Vec<u8>,
            table: [u8; 4],
        },
    }

    /// A refused append, mapping or attend.
    ///
    /// `source` is `Some` when the refusal happened **before** anything was
    /// enqueued, and `None` when the run retained it: submitted work whose
    /// completion is unknown may still be reading those bytes.
    #[derive(Debug)]
    pub struct PagedRunRefused {
        pub error: Error,
        pub source: Option<RefusedSource>,
    }

    /// One raw FP32 partial produced by the separately-qualified device entry
    /// point. The composition/qualification root widens these fields and calls
    /// `moxie_oracles::online_softmax::Partial::merge`; the executor owns the
    /// device launch, not a second copy of the merge algebra.
    #[derive(Debug)]
    pub struct DevicePartial {
        pub max: f32,
        pub sum: f32,
        pub weighted: Vec<f32>,
    }

    /// The two raw per-block results from one bounded stream. The merge is
    /// deliberately performed by the caller that owns the shared oracle.
    /// `host_to_device_bytes` excludes the query upload and names the staged
    /// block's K/V bytes plus its one page-table entry.
    #[derive(Debug)]
    pub struct TwoBlockAttention {
        pub resident: Vec<DevicePartial>,
        pub staged: Vec<DevicePartial>,
        pub host_to_device_bytes: u64,
    }

    /// An incremental host-backed stream. The caller supplies exactly one
    /// host block to [`Self::stage_next`], folds the returned partial, and may
    /// then read the next block. The run owns only the current block while a
    /// device operation is in flight.
    #[derive(Debug)]
    pub struct NBlockStream<'run, 'ctx> {
        run: &'run mut PagedAttentionRun<'ctx>,
        stream: &'run Stream<'ctx>,
        launch: PagedAttentionLaunch,
        next_base: u64,
        remaining_rows: u64,
        remaining_blocks: u64,
        host_to_device_bytes: u64,
        prefetched_total: u64,
        folded_total: u64,
        outstanding_rows: u64,
        page_rows: [u64; 2],
    }

    impl NBlockStream<'_, '_> {
        /// Staged blocks still required before this stream is complete.
        pub const fn remaining_blocks(&self) -> u64 {
            self.remaining_blocks
        }

        /// Host-to-device bytes completed by this stream, excluding the query.
        pub const fn host_to_device_bytes(&self) -> u64 {
            self.host_to_device_bytes
        }
    }

    impl PagedRunRefused {
        /// Whether the run kept the source because its completion is unknown.
        pub const fn retained_source(&self) -> bool {
            self.source.is_none()
        }
    }

    #[derive(Debug)]
    pub struct PagedCloseRefused<'ctx> {
        pub run: PagedAttentionRun<'ctx>,
        pub error: Error,
    }

    /// A refused `attend_into`, **in the ranges the caller gave it**.
    ///
    /// `ranges` is `Some` when the refusal happened before anything was
    /// enqueued -- the caller's query and output are handed straight back --
    /// and `None` when the launch was enqueued with its completion
    /// unobserved: the run keeps them, because the caller releasing or
    /// reusing them while the kernel may still be reading or writing them
    /// would be a use-after-free the type system cannot see. The run is
    /// quarantined exactly when this is `None`.
    #[derive(Debug)]
    pub struct PagedAttendRefused<'ctx> {
        pub error: Error,
        pub ranges: Option<(DeviceRange<'ctx>, DeviceRange<'ctx>)>,
    }

    impl PagedAttendRefused<'_> {
        /// Whether the run kept the ranges because their completion is
        /// unknown.
        pub const fn retained_ranges(&self) -> bool {
            self.ranges.is_none()
        }
    }

    /// One layer's admitted paged key/value state, plus the per-launch query and
    /// output ranges and the resolved attention symbol.
    #[derive(Debug)]
    #[must_use = "an unclosed run keeps its arena and its reservation"]
    pub struct PagedAttentionRun<'ctx> {
        module: ManuallyDrop<ResolvedModule<'ctx>>,
        descriptor: SemanticKernelDescriptor,
        geometry: PageGeometry,
        heads: u64,
        max_rows: u64,
        arena: Option<DeviceArena<'ctx>>,
        keys: Option<DeviceRange<'ctx>>,
        values: Option<DeviceRange<'ctx>>,
        table: Option<DeviceRange<'ctx>>,
        query: Option<DeviceRange<'ctx>>,
        output: Option<DeviceRange<'ctx>>,
        /// Host-sourced staging pages and reusable FP32 partial outputs.
        staged_keys: [Option<DeviceRange<'ctx>>; 2],
        staged_values: [Option<DeviceRange<'ctx>>; 2],
        staged_table: [Option<DeviceRange<'ctx>>; 2],
        partial_max: Option<DeviceRange<'ctx>>,
        partial_sum: Option<DeviceRange<'ctx>>,
        partial_weighted: Option<DeviceRange<'ctx>>,
        pinned_bounce: Option<PinnedHostBuffer<'ctx>>,
        pinned_page_bytes: usize,
        copy_stream: Option<Stream<'ctx>>,
        copied: [Option<Event<'ctx>>; 2],
        copied_pending: [bool; 2],
        read_ahead_stream_active: bool,
        prefetched_unused: u64,
        /// Prefix-lineage entries the run reserves for one branch fork; zero
        /// means this run was not admitted as a fork destination. The matching
        /// request also charges SequenceState's branch and transaction nodes.
        #[cfg(feature = "paged-attention-binding")]
        fork_lineage_capacity: u64,
        /// The published mapping, and the absolute row its logical page zero
        /// names.
        ///
        /// **A table without its base is not a mapping.** The base makes it a
        /// view of *absolute* rows: every write is checked against it, every
        /// launch must name it, and a republication must agree with it wherever
        /// both describe a page that holds written rows.
        page_table: Vec<u32>,
        page_table_base: u64,
        /// Reusable little-endian page-table bytes. Allocated during admission
        /// so publishing never creates an unaccounted host buffer.
        page_table_upload: Vec<u8>,
        staging: Staging,
        /// Rows whose copy into the pages this run has enqueued.
        ///
        /// A physical high-water mark, not a frontier. What is history is the
        /// state authority's to say, and it says it by publishing rows this run
        /// has already returned successfully for. The two are checked against
        /// each other rather than one standing in for the other: this refuses a
        /// launch reading bytes it never wrote, and the authority refuses one
        /// reading rows it never committed. With no pending event, these copies
        /// have also been observed complete.
        written: u64,
        arena_bytes: u64,
        ledger: LedgerId,
        ctx: &'ctx RankContext,
        /// What this run is holding across work whose completion it has not
        /// observed. An append holds two vectors and hands both back unjoined.
        held: Option<RefusedSource>,
        /// The event recorded after this run's last deferred operation. Every
        /// later deferred operation waits on it before enqueueing, preserving
        /// order when operations use different streams. Until it is observed,
        /// `held` may still be read by the device, and `written` counts rows
        /// whose copies are enqueued rather than observed.
        pending: Option<Event<'ctx>>,
        /// The query and output ranges of an `attend`/`attend_into` whose
        /// completion is unobserved.
        ///
        /// Fixed two-slot storage, not a `Vec`: exactly one launch is ever in
        /// flight, so there is nothing to reserve and nothing to allocate on
        /// this path. Populated only alongside `quarantined`, and **never**
        /// read back out again -- not by `close`, which refuses outright while
        /// quarantined, and not by anything else. An ordinary drop is what
        /// disposes of it; a caller reusing a range the kernel may still be
        /// reading or writing is exactly the bug this field exists to
        /// prevent, and never handing the pair back is what prevents it.
        held_ranges: Option<(DeviceRange<'ctx>, DeviceRange<'ctx>)>,
        quarantined: bool,
        partial_symbol: Option<usize>,
        #[cfg(feature = "paged-attention-test-hooks")]
        staging_failure_after: Option<u64>,
        #[cfg(feature = "paged-attention-test-hooks")]
        branch_copy_failure_after: Option<u64>,
        #[cfg(feature = "paged-attention-test-hooks")]
        gate_next_prefetch: bool,
    }

    impl<'ctx> PagedAttentionRun<'ctx> {
        /// Neither quarantined nor holding a refused source or range: no
        /// operation of this run is outstanding.
        #[cfg(feature = "paged-attention-binding")]
        pub(crate) fn is_idle(&self) -> bool {
            !self.quarantined
                && !self.read_ahead_stream_active
                && self.held.is_none()
                && self.held_ranges.is_none()
                && self.pending.is_none()
        }

        /// Observe deferred work and return any held page-table upload buffer.
        /// Unfolded copy events remain asynchronous while a stream pipelines.
        pub(crate) fn observe_pending(&mut self) -> Result<()> {
            if let Some(pending) = self.pending.as_ref() {
                if let Err(error) = pending.synchronize() {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
                self.pending = None;
                if let Some(held) = self.held.take() {
                    match held {
                        RefusedSource::PageTableUpload(bytes) => self.page_table_upload = bytes,
                        other => self.held = Some(other),
                    }
                }
            }
            if !self.read_ahead_stream_active {
                for page in 0..2 {
                    if !self.copied_pending[page] {
                        continue;
                    }
                    if let Err(error) = self.copied[page]
                        .as_ref()
                        .expect("a pending copy has its completion event")
                        .synchronize()
                    {
                        self.quarantined = true;
                        return Err(self.attribute(error));
                    }
                    self.copied_pending[page] = false;
                }
            }
            Ok(())
        }

        fn order_after_pending(&self, stream: &Stream<'ctx>) -> Result<()> {
            if let Some(pending) = self.pending.as_ref() {
                stream.wait_event(pending)?;
            }
            Ok(())
        }

        /// Clear a quarantine this caller caused and has since observed
        /// drained. Only the tensor-parallel workers call this, from cleanup
        /// after their rank streams drain, and only for runs they accepted idle
        /// ([`Self::is_idle`]) and then used on the drained rank stream alone.
        /// So everything held here was created by that step's own work on
        /// that stream. Every other owner keeps the quarantine, and `close`'s
        /// refusal of it (task 0038).
        ///
        /// The held source is resolved by what it is. The run's own admitted
        /// page-table upload buffer is restored, since publishing reuses it.
        /// An append's rows are that step's own copies and are dropped. A host
        /// query or stream source no tensor-parallel step creates, so it
        /// stays held, the run stays quarantined and the refusal says so. The
        /// held query and output ranges go back to the caller, which owns
        /// them.
        #[cfg(feature = "paged-attention-binding")]
        pub(crate) fn reclaim_drained(
            &mut self,
        ) -> Result<Option<(DeviceRange<'ctx>, DeviceRange<'ctx>)>> {
            self.observe_pending()?;
            match self.held.take() {
                None | Some(RefusedSource::Rows { .. }) => {}
                Some(RefusedSource::PageTableUpload(bytes)) => self.page_table_upload = bytes,
                Some(other) => {
                    self.held = Some(other);
                    return Err(invalid(
                        "run",
                        "this run holds a source a tensor-parallel step does not create",
                    ));
                }
            }
            self.quarantined = false;
            Ok(self.held_ranges.take())
        }

        /// Charge the pages, the table and one launch's query and output;
        /// materialize them; resolve the symbol.
        #[allow(clippy::result_large_err)]
        pub fn admit(
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
            descriptor: SemanticKernelDescriptor,
            geometry: PageGeometry,
            heads: u64,
            max_rows: u64,
            staging: Staging,
        ) -> std::result::Result<Self, PagedAdmitRefused<'ctx>> {
            Self::admit_with_lineage_capacities(
                ledger, ctx, descriptor, geometry, heads, max_rows, 0, 0, staging,
            )
        }

        /// Admit the root lineage and transaction slot owned by a
        /// `DeviceKvSequence`. Raw paged runs without that state authority keep
        /// using [`Self::admit`] and do not charge or impose a lineage limit.
        #[allow(clippy::result_large_err)]
        #[allow(clippy::too_many_arguments)]
        pub fn admit_for_sequence(
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
            descriptor: SemanticKernelDescriptor,
            geometry: PageGeometry,
            heads: u64,
            max_rows: u64,
            root_lineage_capacity: u64,
            staging: Staging,
        ) -> std::result::Result<Self, PagedAdmitRefused<'ctx>> {
            if root_lineage_capacity == 0 {
                return Err(PagedAdmitRefused {
                    error: invalid("root_lineage_capacity", "a sequence needs lineage entries"),
                    reservation: None,
                    rejection: None,
                    arena: None,
                    ranges: Vec::new(),
                    cleanup: None,
                });
            }
            Self::admit_with_lineage_capacities(
                ledger,
                ctx,
                descriptor,
                geometry,
                heads,
                max_rows,
                root_lineage_capacity,
                0,
                staging,
            )
        }

        /// Admit this run as the destination of one branch fork, reserving the
        /// full logical lineage capacity separately from physical page count.
        #[allow(clippy::result_large_err)]
        #[allow(clippy::too_many_arguments)]
        pub fn admit_for_fork(
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
            descriptor: SemanticKernelDescriptor,
            geometry: PageGeometry,
            heads: u64,
            max_rows: u64,
            fork_lineage_capacity: u64,
            staging: Staging,
        ) -> std::result::Result<Self, PagedAdmitRefused<'ctx>> {
            if fork_lineage_capacity == 0 {
                return Err(PagedAdmitRefused {
                    error: invalid("fork_lineage_capacity", "a fork needs lineage entries"),
                    reservation: None,
                    rejection: None,
                    arena: None,
                    ranges: Vec::new(),
                    cleanup: None,
                });
            }
            Self::admit_with_lineage_capacities(
                ledger,
                ctx,
                descriptor,
                geometry,
                heads,
                max_rows,
                0,
                fork_lineage_capacity,
                staging,
            )
        }

        #[allow(clippy::result_large_err)]
        #[allow(clippy::too_many_arguments)]
        fn admit_with_lineage_capacities(
            ledger: &mut Ledger,
            ctx: &'ctx RankContext,
            descriptor: SemanticKernelDescriptor,
            geometry: PageGeometry,
            heads: u64,
            max_rows: u64,
            root_lineage_capacity: u64,
            fork_lineage_capacity: u64,
            staging: Staging,
        ) -> std::result::Result<Self, PagedAdmitRefused<'ctx>> {
            let fail = |error| PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
                arena: None,
                ranges: Vec::new(),
                cleanup: None,
            };
            if descriptor.sm.major != ctx.capability().compute_major
                || descriptor.sm.minor != ctx.capability().compute_minor
            {
                return Err(fail(super::unsupported_kernel_fmt(
                    "paged_attention",
                    format_args!(
                        "descriptor {} is qualified for {} and this device is sm_{}{}",
                        descriptor.id.0,
                        descriptor.sm.name(),
                        ctx.capability().compute_major,
                        ctx.capability().compute_minor
                    ),
                )));
            }
            if let Err(error) = geometry.check() {
                return Err(fail(error));
            }
            if max_rows == 0 || heads == 0 || !heads.is_multiple_of(geometry.kv_heads) {
                return Err(fail(invalid_fmt(
                    "admission",
                    format_args!(
                        "{heads} query head(s) over {} key/value head(s) and {max_rows} row(s) \
                         per launch",
                        geometry.kv_heads
                    ),
                )));
            }
            if let Staging::HostBacked { max_staged_blocks }
            | Staging::HostBackedReadAhead { max_staged_blocks } = staging
                && (max_staged_blocks == 0
                    || max_staged_blocks > moxie_memory::HostBackedPlan::MAX_STAGED_BLOCKS)
            {
                return Err(fail(invalid_fmt(
                    "max_staged_blocks",
                    format_args!(
                        "{} is outside the admitted host-backed range 1..={}",
                        max_staged_blocks,
                        moxie_memory::HostBackedPlan::MAX_STAGED_BLOCKS
                    ),
                )));
            }
            if matches!(staging, Staging::HostBackedReadAhead { .. })
                && ledger
                    .snapshot(Scope::Host)
                    .and_then(|snapshot| snapshot.tier_cap(Tier::Host(HostTier::Pinned)))
                    .is_none()
            {
                return Err(fail(invalid("pinned", "no pinned cap is declared")));
            }
            // Admission re-applies selection's predicate, on the widest launch
            // this run can serve. A descriptor that cannot serve that launch
            // must be refused now rather than at the first attend.
            let widest = match PagedAttentionLaunch::new(
                super::AttentionLayer {
                    geometry,
                    heads,
                    scale: 1.0,
                    visibility: moxie_plan::Visibility::Causal,
                },
                max_rows,
                0,
                0,
                max_rows,
            ) {
                Ok(launch) => launch,
                Err(error) => return Err(fail(error)),
            };
            if let Err(error) = super::descriptor_serves(&descriptor, &widest) {
                return Err(fail(error));
            }
            // **Identity before loading, and whole identity.** Everything above
            // checks fields one at a time, and admission then loads *this
            // build's* fatbin unconditionally — so a descriptor declaring a
            // different layout, accumulation policy, rounding profile,
            // workspace expression or image would be executed by the local
            // image anyway, with its own declaration silently ignored. A
            // descriptor whose operation, ABI and image hash match but whose
            // symbol names a different kernel body is exactly that shape: every
            // field-by-field check above passes while every GPU computes the
            // wrong activation for a plan that looks valid.
            //
            // The descriptor must therefore **be** one the built-in package
            // declares. That binds operation, ABI, operand roles and
            // precisions, output, accumulation, rounding, layout, shape bounds,
            // SM, workspace, image digest and symbols together, which is the
            // only form of this check that cannot be half-satisfied. Selection
            // still runs against whatever catalogue it is given, and this is
            // the boundary where a selected descriptor becomes a launch.
            //
            // Asked of a predicate rather than of a rebuilt catalogue:
            // `paged_attention_catalogue()` allocates a `Vec`, two `String`s per
            // descriptor and a `format!` for each id, and this runs on the path
            // that must produce a typed refusal under memory pressure. The
            // predicate compares every field the catalogue sets, allocates
            // nothing, and is pinned to the catalogue by
            // `the_package_predicate_and_the_catalogue_agree`.
            if !moxie_kernels::paged_attention_declares(&descriptor) {
                return Err(fail(unsupported_kernel_fmt(
                    "paged_attention",
                    format_args!(
                        "descriptor {} is not one this build's paged attention package \
                         declares, so its layout, accumulation, rounding, workspace and \
                         image are not bound to the code that would run",
                        descriptor.id.0
                    ),
                )));
            }

            // The device's own grid limits, and this is why they are queried
            // rather than assumed: `y` and `z` stop at 65,535 while `x` reaches
            // `2^31 - 1`, so a head count that fits a `u32` can still be
            // unlaunchable. Refusing here means a geometry this device cannot
            // launch never gets pages admitted for it; `attend` re-applies the
            // same check for the same reason it re-applies `descriptor_serves`.
            if let Err(error) = check_grid(&widest, ctx) {
                return Err(fail(error));
            }
            let extents = match Extents::derive(
                &geometry,
                heads,
                max_rows,
                root_lineage_capacity,
                fork_lineage_capacity,
                staging,
            ) {
                Ok(extents) => extents,
                Err(error) => return Err(fail(error)),
            };
            let request = match resource_request_with_lineage_capacities(
                &geometry,
                heads,
                max_rows,
                root_lineage_capacity,
                fork_lineage_capacity,
                staging,
                ctx,
            ) {
                Ok(request) => request,
                Err(error) => return Err(fail(error)),
            };
            let reservation = match ledger.admit(&request) {
                Ok(reservation) => reservation,
                Err(moxie_memory::AdmitError::Invalid(error)) => return Err(fail(error)),
                Err(moxie_memory::AdmitError::Rejected(rejection)) => {
                    return Err(PagedAdmitRefused {
                        error: Error::CapacityExceeded {
                            tier: None,
                            requested_bytes: rejection.shortfall_bytes,
                            available_bytes: 0,
                        },
                        reservation: None,
                        rejection: Some(rejection),
                        arena: None,
                        ranges: Vec::new(),
                        cleanup: None,
                    });
                }
            };
            let label = match moxie_memory::fallible::text(format_args!(
                "paged-attention-{}",
                descriptor.id.0
            )) {
                Ok(label) => label,
                Err(error) => return Err(give_back(ledger, reservation, error)),
            };
            // Only the regions that carry bytes. A direct-device run holds no
            // per-step buffers at all, and a zero-byte region is not a
            // partition of anything.
            let mut regions: Vec<(DeviceTier, u64)> = Vec::new();
            if regions
                .try_reserve_exact(if staging.partial_buffers() { 3 } else { 2 })
                .is_err()
            {
                return Err(give_back(
                    ledger,
                    reservation,
                    Error::CapacityExceeded {
                        tier: None,
                        requested_bytes: 0,
                        available_bytes: 0,
                    },
                ));
            }
            regions.push((DeviceTier::KvStatePages, extents.persistent));
            if extents.per_step != 0 {
                regions.push((DeviceTier::Activations, extents.per_step));
            }
            if extents.transfer != 0 {
                regions.push((DeviceTier::TransferStaging, extents.transfer));
            }
            let mut arena = match DeviceArena::create_partitioned(
                ledger,
                reservation,
                ctx,
                // The pages are persistent state; the query and output are
                // per-step activations. One physical allocation, two declared
                // tiers, because the ledger's totals are about what the bytes
                // are for and not only about how many there are.
                &regions,
                extents.total,
                label,
            ) {
                Ok(arena) => arena,
                Err(refused) => {
                    return Err(give_back(ledger, refused.reservation, refused.error));
                }
            };
            let hold_count = match staging {
                Staging::Host => 5,
                Staging::DeviceHandles => 3,
                Staging::TwoBlock | Staging::HostBacked { .. } => 10,
                Staging::HostBackedReadAhead { .. } => 13,
            };
            let mut hold: Vec<DeviceRange<'ctx>> = Vec::new();
            if hold.try_reserve_exact(hold_count).is_err() {
                return Err(unwind(
                    arena,
                    Vec::new(),
                    ledger,
                    Error::CapacityExceeded {
                        tier: None,
                        requested_bytes: 0,
                        available_bytes: 0,
                    },
                ));
            }
            let allocate = |arena: &mut DeviceArena<'ctx>, bytes, label: &str| {
                let owned = moxie_memory::fallible::text(format_args!("{label}"))?;
                arena
                    .allocate(bytes, ALIGNMENT, owned)
                    .map_err(|refused| refused.error)
            };
            let mut wanted: Vec<(u64, &str)> = vec![
                (extents.payload, "attention-keys"),
                (extents.payload, "attention-values"),
                (extents.table, "attention-page-table"),
            ];
            if staging == Staging::Host {
                wanted.push((extents.query, "attention-query"));
                wanted.push((extents.query, "attention-output"));
            } else if staging.partial_buffers() {
                wanted.push((extents.query, "attention-stream-query"));
                wanted.push((extents.staged_payload, "attention-stream-staged-keys"));
                wanted.push((extents.staged_payload, "attention-stream-staged-values"));
                wanted.push((extents.staged_table, "attention-stream-staged-table"));
                if matches!(staging, Staging::HostBackedReadAhead { .. }) {
                    wanted.push((extents.staged_payload, "attention-stream-staged-keys-next"));
                    wanted.push((
                        extents.staged_payload,
                        "attention-stream-staged-values-next",
                    ));
                    wanted.push((extents.staged_table, "attention-stream-staged-table-next"));
                }
                wanted.push((extents.partial_scalars, "attention-stream-partial-max"));
                wanted.push((extents.partial_scalars, "attention-stream-partial-sum"));
                wanted.push((
                    extents.partial_weighted,
                    "attention-stream-partial-weighted",
                ));
            }
            for (bytes, name) in wanted {
                match allocate(&mut arena, bytes, name) {
                    Ok(range) => hold.push(range),
                    Err(error) => return Err(unwind(arena, hold, ledger, error)),
                }
            }
            // SAFETY: the bytes are this build's own nvcc output, embedded by
            // `include_bytes!`, and the descriptor's image digest is the one the
            // built-in catalogue published for them.
            let image = match unsafe {
                TrustedImage::from_build_output(moxie_kernels::PAGED_ATTENTION_FATBIN)
            } {
                Ok(image) => image,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            let symbols = {
                let mut symbols: Vec<String> = Vec::new();
                let needed = descriptor.symbols.len() + usize::from(staging.partial_buffers());
                let mut room = symbols.try_reserve_exact(needed).is_ok();
                if room {
                    for symbol in &descriptor.symbols {
                        match moxie_memory::fallible::text(format_args!("{}", symbol.0)) {
                            Ok(name) => symbols.push(name),
                            Err(_) => {
                                room = false;
                                break;
                            }
                        }
                    }
                }
                if !room {
                    return Err(unwind(
                        arena,
                        hold,
                        ledger,
                        Error::CapacityExceeded {
                            tier: None,
                            requested_bytes: 0,
                            available_bytes: 0,
                        },
                    ));
                }
                let partial_symbol = if staging.partial_buffers() {
                    match moxie_memory::fallible::text(format_args!(
                        "{}",
                        moxie_kernels::PAGED_ATTENTION_PARTIAL
                    )) {
                        Ok(name) => {
                            symbols.push(name);
                            Some(symbols.len() - 1)
                        }
                        Err(_) => {
                            return Err(unwind(
                                arena,
                                hold,
                                ledger,
                                Error::CapacityExceeded {
                                    tier: None,
                                    requested_bytes: 0,
                                    available_bytes: 0,
                                },
                            ));
                        }
                    }
                } else {
                    None
                };
                (symbols, partial_symbol)
            };
            let (symbols, partial_symbol) = symbols;
            let module = match Module::load(ctx, ModuleImage::Binary(image))
                .and_then(|module| module.resolve_all(&symbols))
            {
                Ok(module) => module,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            let mut page_table: Vec<u32> = Vec::new();
            if page_table.try_reserve_exact(extents.pages_usize).is_err() {
                return Err(unwind(
                    arena,
                    hold,
                    ledger,
                    Error::CapacityExceeded {
                        tier: None,
                        requested_bytes: extents.table,
                        available_bytes: 0,
                    },
                ));
            }
            let page_table_upload_len = match usize::try_from(extents.page_table_upload) {
                Ok(bytes) => bytes,
                Err(_) => {
                    return Err(unwind(
                        arena,
                        hold,
                        ledger,
                        invalid(
                            "page_table",
                            "the upload buffer exceeds host addressability",
                        ),
                    ));
                }
            };
            let page_table_upload = match super::try_zeroed(page_table_upload_len) {
                Ok(bytes) => bytes,
                Err(error) => return Err(unwind(arena, hold, ledger, error)),
            };
            let (copy_stream, copied, pinned_bounce, pinned_page_bytes) =
                if matches!(staging, Staging::HostBackedReadAhead { .. }) {
                    let copy_stream = match Stream::new(ctx) {
                        Ok(stream) => stream,
                        Err(error) => return Err(unwind(arena, hold, ledger, error)),
                    };
                    let first = match Event::new(ctx) {
                        Ok(event) => event,
                        Err(error) => return Err(unwind(arena, hold, ledger, error)),
                    };
                    let second = match Event::new(ctx) {
                        Ok(event) => event,
                        Err(error) => return Err(unwind(arena, hold, ledger, error)),
                    };
                    let pinned_page_bytes = match usize::try_from(extents.pinned_bounce / 2) {
                        Ok(bytes) if bytes != 0 => bytes,
                        _ => {
                            return Err(unwind(
                                arena,
                                hold,
                                ledger,
                                invalid("pinned", "bounce page size exceeds host addressability"),
                            ));
                        }
                    };
                    let pinned_bytes = match usize::try_from(extents.pinned_bounce) {
                        Ok(bytes) if bytes != 0 => bytes,
                        _ => {
                            return Err(unwind(
                                arena,
                                hold,
                                ledger,
                                invalid("pinned", "bounce allocation exceeds host addressability"),
                            ));
                        }
                    };
                    let pinned_bounce = match PinnedHostBuffer::alloc(ctx, pinned_bytes) {
                        Ok(buffer) => buffer,
                        Err(error) => return Err(unwind(arena, hold, ledger, error)),
                    };
                    (
                        Some(copy_stream),
                        [Some(first), Some(second)],
                        Some(pinned_bounce),
                        pinned_page_bytes,
                    )
                } else {
                    (None, [None, None], None, 0)
                };
            let (
                query,
                output,
                staged_keys,
                staged_values,
                staged_table,
                partial_max,
                partial_sum,
                partial_weighted,
            ) = match staging {
                Staging::Host => {
                    let output = hold.pop().expect("output range");
                    let query = hold.pop().expect("query range");
                    (
                        Some(query),
                        Some(output),
                        [None, None],
                        [None, None],
                        [None, None],
                        None,
                        None,
                        None,
                    )
                }
                Staging::TwoBlock | Staging::HostBacked { .. } => {
                    let partial_weighted = hold.pop().expect("partial weighted range");
                    let partial_sum = hold.pop().expect("partial sum range");
                    let partial_max = hold.pop().expect("partial max range");
                    let staged_table = hold.pop().expect("staged table range");
                    let staged_values = hold.pop().expect("staged values range");
                    let staged_keys = hold.pop().expect("staged keys range");
                    let query = hold.pop().expect("query range");
                    (
                        Some(query),
                        None,
                        [Some(staged_keys), None],
                        [Some(staged_values), None],
                        [Some(staged_table), None],
                        Some(partial_max),
                        Some(partial_sum),
                        Some(partial_weighted),
                    )
                }
                Staging::HostBackedReadAhead { .. } => {
                    let partial_weighted = hold.pop().expect("partial weighted range");
                    let partial_sum = hold.pop().expect("partial sum range");
                    let partial_max = hold.pop().expect("partial max range");
                    let staged_table_next = hold.pop().expect("second staged table range");
                    let staged_values_next = hold.pop().expect("second staged values range");
                    let staged_keys_next = hold.pop().expect("second staged keys range");
                    let staged_table = hold.pop().expect("first staged table range");
                    let staged_values = hold.pop().expect("first staged values range");
                    let staged_keys = hold.pop().expect("first staged keys range");
                    let query = hold.pop().expect("query range");
                    (
                        Some(query),
                        None,
                        [Some(staged_keys), Some(staged_keys_next)],
                        [Some(staged_values), Some(staged_values_next)],
                        [Some(staged_table), Some(staged_table_next)],
                        Some(partial_max),
                        Some(partial_sum),
                        Some(partial_weighted),
                    )
                }
                Staging::DeviceHandles => (
                    None,
                    None,
                    [None, None],
                    [None, None],
                    [None, None],
                    None,
                    None,
                    None,
                ),
            };
            let table = hold.pop().expect("page table range");
            let values = hold.pop().expect("value range");
            let keys = hold.pop().expect("key range");
            Ok(Self {
                module: ManuallyDrop::new(module),
                descriptor,
                geometry,
                heads,
                max_rows,
                arena: Some(arena),
                keys: Some(keys),
                values: Some(values),
                table: Some(table),
                query,
                output,
                staged_keys,
                staged_values,
                staged_table,
                partial_max,
                partial_sum,
                partial_weighted,
                pinned_bounce,
                pinned_page_bytes,
                copy_stream,
                copied,
                #[cfg(feature = "paged-attention-binding")]
                fork_lineage_capacity,
                staging,
                page_table,
                page_table_base: 0,
                page_table_upload,
                written: 0,
                arena_bytes: extents.total,
                ledger: ledger.id(),
                ctx,
                held: None,
                pending: None,
                copied_pending: [false, false],
                read_ahead_stream_active: false,
                prefetched_unused: 0,
                held_ranges: None,
                quarantined: false,
                partial_symbol,
                #[cfg(feature = "paged-attention-test-hooks")]
                staging_failure_after: None,
                #[cfg(feature = "paged-attention-test-hooks")]
                branch_copy_failure_after: None,
                #[cfg(feature = "paged-attention-test-hooks")]
                gate_next_prefetch: false,
            })
        }

        /// Device bytes this run holds: persistent pages, table, query, output.
        ///
        /// A memory bound and **not** a speed measurement: O6 and O7 are open
        /// and nothing here is timed.
        pub const fn arena_bytes(&self) -> u64 {
            self.arena_bytes
        }

        /// Blocks prefetched by read-ahead streams and never folded.
        pub const fn prefetched_unused(&self) -> u64 {
            self.prefetched_unused
        }

        /// Rows whose copies are enqueued on this run's stream order, not observed.
        ///
        /// Reported separately from the admitted capacity and from what a launch
        /// declares visible, because the three are different numbers and
        /// conflating them is how a cache claims a context it does not hold. It
        /// is **not** the committed frontier: `moxie_state::DeviceKvSequence`
        /// owns that, and this number only bounds what the bytes can support.
        pub const fn written_rows(&self) -> u64 {
            self.written
        }

        /// Rows the admitted pages can physically hold.
        pub fn capacity_rows(&self) -> Result<u64> {
            self.geometry.capacity_rows()
        }

        /// Check the append size against the geometry this run admitted.
        /// This must run before handing the call to `DeviceKvSequence`, which
        /// allocates its page view and placement list from the requested rows.
        #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
        pub(super) fn check_append_rows(&self, rows: u64) -> Result<()> {
            if rows > self.max_rows {
                return Err(invalid_fmt(
                    "rows",
                    format_args!(
                        "append of {rows} rows exceeds this run's admitted maximum of {}",
                        self.max_rows
                    ),
                ));
            }
            Ok(())
        }

        pub const fn geometry(&self) -> &PageGeometry {
            &self.geometry
        }

        pub fn descriptor(&self) -> &SemanticKernelDescriptor {
            &self.descriptor
        }

        /// Cause the next host-block staging operation to fail after its key
        /// copy is enqueued. Qualification uses this to prove a partial copy
        /// quarantines the run instead of launching over an incomplete page.
        #[cfg(feature = "paged-attention-test-hooks")]
        pub fn inject_staging_failure(&mut self) {
            self.staging_failure_after = Some(0);
        }

        /// Gate one prefetch copy for the deterministic ordering test.
        #[cfg(feature = "paged-attention-test-hooks")]
        pub fn gate_next_prefetch_for_test(&mut self) {
            PREFETCH_GATE_RELEASED.store(false, Ordering::Release);
            self.gate_next_prefetch = true;
        }

        /// Release a gated prefetch from the test's helper thread.
        #[cfg(feature = "paged-attention-test-hooks")]
        pub fn release_prefetch_gate_for_test() {
            PREFETCH_GATE_RELEASED.store(true, Ordering::Release);
        }

        /// Cause the staging operation after `successful_blocks` completed
        /// blocks to fail. This keeps the original next-block hook intact and
        /// lets the N-block qualification exercise a middle iteration.
        #[cfg(feature = "paged-attention-test-hooks")]
        pub fn inject_staging_failure_after(&mut self, successful_blocks: u64) {
            self.staging_failure_after = Some(successful_blocks);
        }

        /// Cause an eager child copy to fail after this many physical pages
        /// have completed. The callback is invoked after the logical state
        /// fork, so this exercises cleanup of a real partial child.
        #[cfg(feature = "paged-attention-test-hooks")]
        pub fn inject_branch_copy_failure_after(&mut self, successful_pages: u64) {
            self.branch_copy_failure_after = Some(successful_pages);
        }

        /// Publish the logical-to-physical page mapping this run will use.
        ///
        /// Republishing is legal -- the state authority's retained range slides
        /// as its ring wraps, and a mapping that starts later is how this run
        /// learns that -- but a new table must agree with the one it replaces
        /// wherever both describe a page holding rows this run has written:
        /// those rows already live at the addresses the old table resolved, and
        /// sending one elsewhere would leave it unreadable while every other
        /// check still passed. The table is then read by the kernel and by this
        /// host code for exactly the same addresses; a second host-side mapping
        /// would be the page table that disagrees with the page table.
        ///
        /// `pub(crate)`, not `pub`: publishing a table is one half of the one
        /// authorized operation [`PagedKvWriterAdapter::write_layer`] performs
        /// from the state authority's own [`PageView`], and a caller reaching
        /// this directly could publish a mapping the authority never decided.
        /// `RawPagedFixture` is the one named exception, for a gate that has no
        /// authority to begin with.
        #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
        #[allow(clippy::result_large_err)]
        pub(crate) fn publish_page_table(
            &mut self,
            stream: &Stream<'ctx>,
            base: u64,
            table: Vec<u32>,
        ) -> std::result::Result<(), PagedRunRefused> {
            self.publish_page_table_with_mode(stream, base, table, false)
        }

        /// Publish a mapping on the stream and defer observing its completion.
        #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
        #[allow(clippy::result_large_err)]
        pub(crate) fn publish_page_table_deferred(
            &mut self,
            stream: &Stream<'ctx>,
            base: u64,
            table: Vec<u32>,
        ) -> std::result::Result<(), PagedRunRefused> {
            self.publish_page_table_with_mode(stream, base, table, true)
        }

        #[allow(clippy::result_large_err)]
        fn publish_page_table_with_mode(
            &mut self,
            stream: &Stream<'ctx>,
            base: u64,
            table: Vec<u32>,
            defer: bool,
        ) -> std::result::Result<(), PagedRunRefused> {
            // The table goes back as the entries it arrived as. Re-encoding it
            // into bytes to report a refusal is an allocation on the refusal
            // path, which is where allocations fail.
            let give_back = |error, table: Vec<u32>| PagedRunRefused {
                error,
                source: Some(RefusedSource::PageTable(table)),
            };
            if let Err(error) = self.observe_pending() {
                return Err(give_back(error, table));
            }
            if defer && let Err(error) = self.order_after_pending(stream) {
                return Err(give_back(error, table));
            }
            if self.quarantined {
                return Err(give_back(invalid("run", "this run is quarantined"), table));
            }
            if let Err(error) = self.same_device(stream) {
                return Err(give_back(error, table));
            }
            // Republishing is legal: the authority's retained range slides as
            // its ring wraps. What must hold is the mapping's shape and its
            // agreement with the one it replaces, both checked below.
            let pages = self.geometry.pages;
            if !base.is_multiple_of(self.geometry.page_tokens) {
                return Err(give_back(
                    invalid(
                        "base",
                        "a mapping's first logical page must start on a page boundary",
                    ),
                    table,
                ));
            }
            if base > self.written {
                return Err(give_back(
                    invalid(
                        "base",
                        "a mapping cannot start beyond the rows this run has written",
                    ),
                    table,
                ));
            }
            if table.is_empty() || table.len() as u64 > pages {
                return Err(give_back(
                    invalid_fmt(
                        "page_table",
                        format_args!(
                            "{} logical page(s) over {pages} admitted page(s)",
                            table.len()
                        ),
                    ),
                    table,
                ));
            }
            if table.len() > self.page_table.capacity() {
                return Err(give_back(
                    invalid(
                        "page_table",
                        "the mapping exceeds the admitted host table capacity",
                    ),
                    table,
                ));
            }
            // Every physical identity must exist, and no two logical pages may
            // name the same one: aliasing pages would make an append overwrite
            // history that is still visible, which no later check could detect.
            // Reuse the admitted upload buffer as a bitset, then overwrite it
            // with the encoded table below. The buffer is cleared on every
            // publication because a successful publication left encoded bytes
            // in it and a refused one may have left partial bits.
            let bitset_bytes = match pages
                .checked_add(7)
                .and_then(|pages| pages.checked_div(8))
                .and_then(|bytes| usize::try_from(bytes).ok())
            {
                Some(bytes) => bytes,
                None => return Err(give_back(Error::Dim(DimError::Overflow), table)),
            };
            if bitset_bytes > self.page_table_upload.len() {
                return Err(give_back(
                    invalid("page_table", "the admitted upload buffer is too small"),
                    table,
                ));
            }
            {
                let bitset = &mut self.page_table_upload[..bitset_bytes];
                bitset.fill(0);
                for (logical, physical_entry) in table.iter().copied().enumerate() {
                    let physical = u64::from(physical_entry);
                    if physical >= pages {
                        return Err(give_back(
                            invalid_fmt(
                                "page_table",
                                format_args!(
                                    "logical page {logical} names physical page {physical} of \
                                     {pages} admitted"
                                ),
                            ),
                            table,
                        ));
                    }
                    let byte = match usize::try_from(physical / 8) {
                        Ok(byte) => byte,
                        Err(_) => {
                            return Err(give_back(
                                invalid("page_table", "a physical page is not addressable"),
                                table,
                            ));
                        }
                    };
                    let mask = 1u8 << (physical % 8);
                    if bitset[byte] & mask != 0 {
                        return Err(give_back(
                            invalid_fmt(
                                "page_table",
                                format_args!("physical page {physical} is named twice"),
                            ),
                            table,
                        ));
                    }
                    bitset[byte] |= mask;
                }
            }

            // **Agreement with the mapping it replaces**, wherever both describe
            // the same absolute page. Rows already written live at addresses the
            // old table resolved; a new table that sent one of them somewhere
            // else would leave every one of those rows unreadable while every
            // individual check still passed. This is the other half of what
            // makes a write, a table and a launch one coherent view.
            if !self.page_table.is_empty() {
                let page_tokens = self.geometry.page_tokens;
                let old_first = self.page_table_base / page_tokens;
                let new_first = base / page_tokens;
                for (index, physical) in table.iter().enumerate() {
                    let absolute = new_first + index as u64;
                    let Some(old_index) = absolute.checked_sub(old_first) else {
                        continue;
                    };
                    let Some(previous) = self.page_table.get(old_index as usize) else {
                        continue;
                    };
                    // Only pages that actually hold written rows are bound: a
                    // page beyond the frontier has nothing to be inconsistent
                    // with.
                    if absolute * page_tokens >= self.written {
                        continue;
                    }
                    if previous != physical {
                        return Err(give_back(
                            invalid_fmt(
                                "page_table",
                                format_args!(
                                    "absolute page {absolute} holds written rows at physical \
                                     page {previous} and the new mapping sends it to \
                                     {physical}"
                                ),
                            ),
                            table,
                        ));
                    }
                }
            }

            let encoded_len = match table.len().checked_mul(core::mem::size_of::<u32>()) {
                Some(bytes) => bytes,
                None => {
                    return Err(give_back(Error::Dim(DimError::Overflow), table));
                }
            };
            let mut bytes = core::mem::take(&mut self.page_table_upload);
            if encoded_len > bytes.len() {
                self.page_table_upload = bytes;
                return Err(give_back(
                    invalid("page_table", "the admitted upload buffer is too small"),
                    table,
                ));
            }
            for (entry, destination) in table
                .iter()
                .zip(bytes[..encoded_len].chunks_exact_mut(core::mem::size_of::<u32>()))
            {
                destination.copy_from_slice(&entry.to_le_bytes());
            }
            let range = self.table.as_ref().expect("live page table range");
            // Owned by `self.held` before the first enqueue, exactly as
            // `attend` owns its query: a copy call that itself refuses may
            // still have submitted work the driver has not unwound, so the
            // encoded bytes must already be something other than this local
            // variable before that call runs, not after.
            self.held = Some(RefusedSource::PageTableUpload(bytes));
            let Some(RefusedSource::PageTableUpload(bytes)) = self.held.as_ref() else {
                unreachable!("just assigned")
            };
            // SAFETY: the source is owned by `self.held` until completion is
            // observed, and the destination is this run's own admitted range.
            if let Err(error) = unsafe { range.copy_from_host_async(&bytes[..encoded_len], stream) }
            {
                self.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            let completion = if defer {
                self.defer(Ok(()), stream)
            } else {
                self.settle(Ok(()), stream)
            };
            if let Err(error) = completion {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            if !defer {
                let Some(RefusedSource::PageTableUpload(bytes)) = self.held.take() else {
                    unreachable!("page-table upload source is still held after settlement")
                };
                self.page_table_upload = bytes;
            }
            self.page_table.clear();
            self.page_table.extend_from_slice(&table);
            self.page_table_base = base;
            Ok(())
        }

        /// Eagerly copy one parent's complete admitted page storage into this
        /// child run. The state authority has already created the child and
        /// supplied its page view through [`PagedKvWriter::copy_branch`]; this
        /// method only performs the device effect and records the copied
        /// frontier.
        #[cfg(feature = "paged-attention-binding")]
        pub(crate) fn copy_branch_from(
            &mut self,
            stream: &Stream<'ctx>,
            source: &Self,
            view: PageView,
            rows: u64,
        ) -> Result<()> {
            self.observe_pending()?;
            if self.quarantined || source.quarantined {
                return Err(invalid("run", "a branch copy names a quarantined run"));
            }
            self.same_device(stream)?;
            source.same_device(stream)?;
            if self.geometry != source.geometry
                || self.heads != source.heads
                || self.max_rows != source.max_rows
            {
                return Err(invalid(
                    "branch",
                    "parent and child runs do not have identical admitted geometry",
                ));
            }
            if rows == 0 {
                return Ok(());
            }
            if rows > source.written {
                return Err(invalid(
                    "rows",
                    "the parent run has not observed the fork prefix copied",
                ));
            }
            if self.written != 0 || !self.page_table.is_empty() {
                return Err(invalid(
                    "branch",
                    "the child run already contains published device state",
                ));
            }
            if view.base != source.page_table_base
                || view.table.is_empty()
                || view.table.len() > source.page_table.len()
                || source.page_table[..view.table.len()] != view.table
            {
                return Err(invalid(
                    "page_table",
                    "the child fork view does not match the parent's published mapping",
                ));
            }
            source.order_after_pending(stream)?;
            let page_bytes = self.geometry.page_bytes()?;
            for physical_page in 0..self.geometry.pages {
                let within = physical_page
                    .checked_mul(page_bytes)
                    .ok_or(Error::Dim(DimError::Overflow))?;
                let copied = (|| -> Result<()> {
                    let target_keys = self.keys.as_ref().expect("live child key range");
                    let source_keys = source.keys.as_ref().expect("live parent key range");
                    // SAFETY: both runs retain their ranges through the
                    // synchronous settle below, and the arena method checks
                    // both extents and the device identity.
                    unsafe {
                        target_keys.copy_from_device_async_at(
                            within,
                            source_keys,
                            within,
                            page_bytes,
                            stream,
                        )?
                    };
                    let target_values = self.values.as_ref().expect("live child value range");
                    let source_values = source.values.as_ref().expect("live parent value range");
                    // SAFETY: as for the key page above.
                    unsafe {
                        target_values.copy_from_device_async_at(
                            within,
                            source_values,
                            within,
                            page_bytes,
                            stream,
                        )
                    }
                })();
                if let Err(error) = copied {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
                self.settle(Ok(()), stream)?;
                #[cfg(feature = "paged-attention-test-hooks")]
                if let Some(remaining) = self.branch_copy_failure_after {
                    if remaining <= 1 {
                        self.branch_copy_failure_after = None;
                        return Err(Error::Cancelled {
                            at: "injected device branch copy",
                        });
                    }
                    self.branch_copy_failure_after = Some(remaining - 1);
                }
            }

            let table_bytes = view
                .table
                .len()
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(Error::Dim(DimError::Overflow))?;
            let source_table = source.table.as_ref().expect("live parent table range");
            let target_table = self.table.as_ref().expect("live child table range");
            // SAFETY: the table extent is checked by the arena wrapper and the
            // source table contains the same entries validated above.
            unsafe {
                target_table.copy_from_device_async_at(
                    0,
                    source_table,
                    0,
                    table_bytes as u64,
                    stream,
                )?
            };
            self.settle(Ok(()), stream)?;
            self.page_table.clear();
            self.page_table.extend_from_slice(&view.table);
            self.page_table_base = view.base;
            self.written = rows;
            Ok(())
        }

        /// Whether this stream belongs to the device this run was admitted on.
        ///
        /// An offset resolved inside another device's allocation names the
        /// wrong bytes, and enqueuing on a foreign stream would order the copy
        /// against work this run never sees -- silently, since a stream from a
        /// different GPU is otherwise a value of the right type.
        fn same_device(&self, stream: &Stream<'ctx>) -> Result<()> {
            if stream.device_uuid() != self.ctx.uuid() {
                return Err(invalid(
                    "stream",
                    "this stream and this run do not name one device",
                ));
            }
            Ok(())
        }

        /// Where the published mapping puts the page holding absolute `row`.
        ///
        /// The one place this run turns a row into a physical page, used by
        /// both the write path and the readback so neither can drift from the
        /// table the kernel reads.
        fn resolves_to(&self, row: u64) -> Result<u64> {
            if self.page_table.is_empty() {
                return Err(invalid("page_table", "no page mapping has been published"));
            }
            let page_tokens = self.geometry.page_tokens;
            let Some(offset) = row.checked_sub(self.page_table_base) else {
                return Err(invalid_fmt(
                    "placements",
                    format_args!(
                        "row {row} is below the published mapping, which starts at {}",
                        self.page_table_base
                    ),
                ));
            };
            let logical = offset / page_tokens;
            self.page_table
                .get(logical as usize)
                .map(|physical| u64::from(*physical))
                .ok_or_else(|| {
                    invalid_fmt(
                        "placements",
                        format_args!(
                            "row {row} is beyond the published mapping's {} page(s)",
                            self.page_table.len()
                        ),
                    )
                })
        }

        /// Validate the authority's placements and this run's physical extents
        /// before either write path enqueues any payload bytes.
        fn check_write_placements(
            &self,
            stream: &Stream<'ctx>,
            placements: &[PagePlacement],
        ) -> Result<(u64, u64, u64)> {
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            self.same_device(stream)?;
            if placements.is_empty() {
                return Err(invalid("placements", "a write with no placement"));
            }
            let row_bytes = self
                .geometry
                .row_elements()?
                .checked_mul(2)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;

            // The placements are checked, not trusted: the authority owns the
            // mapping, and this run owns the extents. A page identity outside
            // the admitted pages, a slot outside a page, a gap or a step
            // backwards would each write somewhere nothing asked for.
            let mut rows = 0u64;
            let mut position = placements[0].position;
            for placement in placements {
                if placement.position != position {
                    return Err(invalid(
                        "placements",
                        "the runs are not contiguous and ascending from the first",
                    ));
                }
                if placement.rows == 0 {
                    return Err(invalid("placements", "a run of no rows"));
                }
                // **Against the published mapping, not merely in range.** A
                // placement names an absolute row, the table says where that
                // row's page is, and a write that disagreed with it would put
                // bytes somewhere no launch will ever look — while every
                // bounds check passed. This is what binds the three operations
                // into one view: writes go where the table says, launches read
                // what the table says, and a republication may not contradict
                // either.
                match self.resolves_to(placement.position) {
                    Ok(physical) if physical == placement.physical_page => {}
                    Ok(physical) => {
                        return Err(invalid_fmt(
                            "placements",
                            format_args!(
                                "row {} is placed on physical page {} and the published \
                                 mapping sends it to {physical}",
                                placement.position, placement.physical_page
                            ),
                        ));
                    }
                    Err(error) => return Err(error),
                }
                if placement.slot != placement.position % self.geometry.page_tokens {
                    return Err(invalid_fmt(
                        "placements",
                        format_args!(
                            "row {} is placed at slot {} and its page holds it at {}",
                            placement.position,
                            placement.slot,
                            placement.position % self.geometry.page_tokens
                        ),
                    ));
                }
                let slot_end = placement.slot.checked_add(placement.rows);
                if slot_end.is_none_or(|end| end > self.geometry.page_tokens) {
                    return Err(invalid(
                        "placements",
                        "a run crosses the end of the page it is placed on",
                    ));
                }
                match placement.end() {
                    Some(end) => position = end,
                    None => return Err(Error::Dim(moxie_types::DimError::Overflow)),
                }
                rows += placement.rows;
            }
            rows.checked_mul(row_bytes)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            Ok((row_bytes, rows, position))
        }

        /// Write host-backed dense K/V rows at placements from the state
        /// authority, returning the sources on a refusal before enqueue.
        #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
        #[allow(clippy::result_large_err)]
        pub(crate) fn write_rows(
            &mut self,
            stream: &Stream<'ctx>,
            placements: &[PagePlacement],
            keys: Vec<u8>,
            values: Vec<u8>,
        ) -> std::result::Result<(), PagedRunRefused> {
            let give_back = |error, keys: Vec<u8>, values: Vec<u8>| PagedRunRefused {
                error,
                source: Some(RefusedSource::Rows { keys, values }),
            };
            if let Err(error) = self.observe_pending() {
                return Err(give_back(error, keys, values));
            }
            let (row_bytes, rows, position_end) =
                match self.check_write_placements(stream, placements) {
                    Ok(checked) => checked,
                    Err(error) => return Err(give_back(error, keys, values)),
                };
            let want = rows * row_bytes;
            if keys.len() as u64 != want || values.len() as u64 != want {
                return Err(give_back(
                    invalid_fmt(
                        "rows",
                        format_args!(
                            "{} key byte(s) and {} value byte(s) for {rows} row(s) of \
                             {row_bytes}",
                            keys.len(),
                            values.len()
                        ),
                    ),
                    keys,
                    values,
                ));
            }

            // Everything above refuses without touching the device. From here
            // on the run holds both sources, unmoved and unjoined, until
            // completion is observed.
            self.held = Some(RefusedSource::Rows { keys, values });
            match self.enqueue_writes(stream, placements, row_bytes) {
                Ok(()) => {}
                Err(error) => {
                    return Err(PagedRunRefused {
                        error,
                        source: None,
                    });
                }
            }
            self.held = None;
            // Last, after the event said the bytes are there. This is the
            // physical high-water mark, not a frontier: a `PagedKvWriter`
            // publishes history through the authority, and only for a layer
            // whose write returned `Ok`.
            self.written = self.written.max(position_end);
            Ok(())
        }

        #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
        pub(crate) fn write_rows_from_device(
            &mut self,
            stream: &Stream<'ctx>,
            placements: &[PagePlacement],
            keys: &DeviceRange<'ctx>,
            values: &DeviceRange<'ctx>,
        ) -> Result<()> {
            let (row_bytes, rows, position_end) =
                self.check_write_placements(stream, placements)?;
            let want = rows
                .checked_mul(row_bytes)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            if keys.bytes() < want || values.bytes() < want {
                return Err(invalid(
                    "rows",
                    "a device key or value range is shorter than the append",
                ));
            }

            self.order_after_pending(stream)?;

            let page_bytes = self.geometry.page_bytes()?;
            let mut done = 0u64;
            for placement in placements {
                let within = placement.physical_page * page_bytes + placement.slot * row_bytes;
                let source_within = done * row_bytes;
                let bytes = placement.rows * row_bytes;
                for (range, source) in [
                    (self.keys.as_ref().expect("live key range"), keys),
                    (self.values.as_ref().expect("live value range"), values),
                ] {
                    // SAFETY: the dense operation lease retains each source
                    // range through this method's completion event, and the
                    // destination extent is checked against this run's range.
                    let copied = unsafe {
                        range.copy_from_device_async_at(
                            within,
                            source,
                            source_within,
                            bytes,
                            stream,
                        )
                    };
                    if let Err(error) = copied {
                        self.quarantined = true;
                        return Err(self.attribute(error));
                    }
                }
                done += placement.rows;
            }
            self.defer(Ok(()), stream)?;
            self.written = self.written.max(position_end);
            Ok(())
        }

        #[cfg_attr(not(feature = "paged-attention-binding"), allow(dead_code))]
        fn enqueue_writes(
            &mut self,
            stream: &Stream<'ctx>,
            placements: &[PagePlacement],
            row_bytes: u64,
        ) -> Result<()> {
            let page_bytes = self.geometry.page_bytes()?;
            let Some(RefusedSource::Rows { keys, values }) = self.held.as_ref() else {
                return Err(invalid("write", "the write is not holding its rows"));
            };
            // One copy per placement, per payload. The loop is over the
            // authority's runs, which is where the storage discontinuities are.
            let mut done = 0u64;
            for placement in placements {
                let within = placement.physical_page * page_bytes + placement.slot * row_bytes;
                let start = (done * row_bytes) as usize;
                let len = (placement.rows * row_bytes) as usize;
                for (range, source) in [
                    (self.keys.as_ref().expect("live key range"), keys),
                    (self.values.as_ref().expect("live value range"), values),
                ] {
                    let slice = &source[start..start + len];
                    // SAFETY: the source is held by this run until completion is
                    // observed, and the destination extent is checked against
                    // this run's own admitted range.
                    let copied = unsafe { range.copy_from_host_async_at(within, slice, stream) };
                    if let Err(error) = copied {
                        self.quarantined = true;
                        return Err(self.attribute(error));
                    }
                }
                done += placement.rows;
            }
            self.settle(Ok(()), stream)
        }

        /// Read rows back out of the pages, at placements the authority gave.
        ///
        /// Evidence, not a data path: it is how a gate checks that an aborted
        /// write left every prior byte exactly as it was. It takes placements
        /// for the same reason the write does — the mapping is not this run's
        /// to reconstruct, and a readback that computed its own would be able
        /// to agree with a write that was wrong.
        pub fn read_rows(&self, placements: &[PagePlacement]) -> Result<Vec<u8>> {
            if self.pending.is_some() {
                return Err(invalid("run", "this run has unobserved device work"));
            }
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            if placements.is_empty() {
                return Err(invalid("placements", "a readback of no rows"));
            }
            let row_bytes = self
                .geometry
                .row_elements()?
                .checked_mul(2)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let page_bytes = self.geometry.page_bytes()?;
            let mut rows = 0u64;
            for placement in placements {
                if self.resolves_to(placement.position)? != placement.physical_page
                    || placement.slot != placement.position % self.geometry.page_tokens
                    || placement
                        .slot
                        .checked_add(placement.rows)
                        .is_none_or(|end| end > self.geometry.page_tokens)
                {
                    return Err(invalid(
                        "placements",
                        "a run that the published mapping does not put there",
                    ));
                }
                let end = placement
                    .end()
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                if end > self.written {
                    return Err(invalid_fmt(
                        "placements",
                        format_args!(
                            "row {} was never written; {} row(s) have been",
                            end - 1,
                            self.written
                        ),
                    ));
                }
                rows += placement.rows;
            }
            let len = usize::try_from(rows * row_bytes * 2)
                .map_err(|_| invalid("read_rows", "the readback exceeds this host's usize"))?;
            let mut out = super::try_zeroed(len)?;
            let half = (rows * row_bytes) as usize;
            let mut done = 0u64;
            for placement in placements {
                let within = placement.physical_page * page_bytes + placement.slot * row_bytes;
                let start = (done * row_bytes) as usize;
                let take = (placement.rows * row_bytes) as usize;
                self.keys
                    .as_ref()
                    .expect("live key range")
                    .copy_to_host_at(within, &mut out[start..start + take])?;
                self.values
                    .as_ref()
                    .expect("live value range")
                    .copy_to_host_at(within, &mut out[half + start..half + start + take])?;
                done += placement.rows;
            }
            Ok(out)
        }

        /// Attend from a device query range into a device output range.
        ///
        /// **This is the contract document 04 states**: "attention consumes
        /// device tensor handles and page-table/state handles", and host
        /// reference paths are "explicit separate implementations, not
        /// compulsory staging interfaces". Nothing crosses the host boundary
        /// here — the query is already on the device, the answer stays there,
        /// and a graph whose activation arena holds both never pays for a round
        /// trip per step.
        ///
        /// The ranges must belong to this run's device and be at least the
        /// launch's extents. They are **not** this run's own: a caller passes
        /// the arena slots its plan bound, which is what makes this the path a
        /// lowered graph can use.
        ///
        /// **Taken and returned by value, never borrowed.** Launch is
        /// asynchronous, and the settled form's completion is known once it
        /// returns. If event creation, recording or synchronization fails
        /// after the kernel was enqueued, whether it is still reading these
        /// ranges is unknowable. A borrowed `&DeviceRange` cannot stop the
        /// caller from releasing or reusing them while that is true; owning
        /// them can. A refusal before anything is enqueued hands them straight
        /// back (`PagedAttendRefused::ranges` is `Some`); a refusal after
        /// enqueue with completion unobserved keeps them here instead, and the
        /// run is quarantined exactly then.
        #[allow(clippy::result_large_err)]
        pub fn attend_into(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: DeviceRange<'ctx>,
            output: DeviceRange<'ctx>,
        ) -> std::result::Result<(DeviceRange<'ctx>, DeviceRange<'ctx>), PagedAttendRefused<'ctx>>
        {
            self.attend_into_with_mode(stream, launch, query, output, false)
        }

        /// Enqueue attention and defer observing its completion. The caller
        /// retains `query` and `output` until it observes a completion
        /// recorded after this call on the same stream.
        #[allow(clippy::result_large_err)]
        pub(crate) fn attend_into_deferred(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: DeviceRange<'ctx>,
            output: DeviceRange<'ctx>,
        ) -> std::result::Result<(DeviceRange<'ctx>, DeviceRange<'ctx>), PagedAttendRefused<'ctx>>
        {
            self.attend_into_with_mode(stream, launch, query, output, true)
        }

        #[allow(clippy::result_large_err)]
        fn attend_into_with_mode(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: DeviceRange<'ctx>,
            output: DeviceRange<'ctx>,
            defer: bool,
        ) -> std::result::Result<(DeviceRange<'ctx>, DeviceRange<'ctx>), PagedAttendRefused<'ctx>>
        {
            macro_rules! refuse {
                ($error:expr) => {
                    return Err(PagedAttendRefused {
                        error: $error,
                        ranges: Some((query, output)),
                    })
                };
            }
            if !defer && let Err(error) = self.observe_pending() {
                refuse!(error);
            }
            if defer && let Err(error) = self.order_after_pending(stream) {
                refuse!(error);
            }
            if let Err(error) = self.check_attend(stream, launch) {
                refuse!(error);
            }
            let want = match launch.query_bytes() {
                Ok(want) => want,
                Err(error) => refuse!(error),
            };
            // Extents and identity, read into owned values before any check can
            // move `query`/`output` into a refusal: a loop over `&query`/
            // `&output` would hold them borrowed for the loop's own lifetime,
            // which conflicts with handing them back on the first line refused.
            for (uuid, bytes, what) in [
                (query.device_uuid(), query.bytes(), "query"),
                (output.device_uuid(), output.bytes(), "output"),
            ] {
                if uuid != self.ctx.uuid() {
                    refuse!(invalid(what, "this range belongs to another device"));
                }
                if bytes < want {
                    refuse!(invalid_fmt(
                        what,
                        format_args!("{bytes} byte(s) for a launch needing {want}"),
                    ));
                }
            }
            let addresses = match (|| -> Result<[u64; 5]> {
                Ok([
                    query.device_address()?,
                    self.keys
                        .as_ref()
                        .expect("live key range")
                        .device_address()?,
                    self.values
                        .as_ref()
                        .expect("live value range")
                        .device_address()?,
                    self.table
                        .as_ref()
                        .expect("live page table range")
                        .device_address()?,
                    output.device_address()?,
                ])
            })() {
                Ok(addresses) => addresses,
                Err(error) => refuse!(error),
            };
            // The kernel declares query and output `__restrict__`
            // (`paged_attention.cu`): the compiler is trusted to assume a store
            // through one is never observable through the other. An in-place
            // launch -- the same range for both -- would violate that, so it is
            // refused here rather than left to whatever the alias happens to
            // produce.
            if ranges_overlap(addresses[0], query.bytes(), addresses[4], output.bytes()) {
                refuse!(invalid(
                    "output",
                    "the query and output ranges overlap, which this kernel's __restrict__ \
                     query and output pointers forbid"
                ));
            }
            let scalars = match Self::abi_scalars(launch) {
                Ok(scalars) => scalars,
                Err(error) => refuse!(error),
            };
            match self.launch_with(stream, scalars, addresses, defer) {
                Ok(()) => Ok((query, output)),
                Err(error) => {
                    self.held_ranges = Some((query, output));
                    Err(PagedAttendRefused {
                        error,
                        ranges: None,
                    })
                }
            }
        }

        /// Everything both entry points check before either touches the device.
        fn check_attend(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
        ) -> Result<()> {
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            self.same_device(stream)?;
            super::descriptor_serves(&self.descriptor, launch)?;
            check_grid(launch, self.ctx)?;
            if *launch.geometry() != self.geometry
                || launch.heads() != self.heads
                || launch.rows() > self.max_rows
            {
                return Err(invalid(
                    "launch",
                    "this launch's geometry is not the one this run was admitted for",
                ));
            }
            if launch.history_base() + launch.history_rows() > self.written {
                return Err(invalid_fmt(
                    "history_rows",
                    format_args!(
                        "a launch declaring [{}, {}) against {} written row(s)",
                        launch.history_base(),
                        launch.history_base() + launch.history_rows(),
                        self.written
                    ),
                ));
            }
            // The launch's own view of the history must be the published one:
            // its `history_base` is what the kernel treats as logical page
            // zero, so a launch naming a different base would read the table
            // through an offset nothing wrote against.
            if launch.history_base() != self.page_table_base {
                return Err(invalid_fmt(
                    "history_base",
                    format_args!(
                        "this launch starts its history at {} and the published mapping \
                         starts at {}",
                        launch.history_base(),
                        self.page_table_base
                    ),
                ));
            }
            if launch.logical_pages().unwrap_or(u64::MAX) > self.page_table.len() as u64 {
                return Err(invalid(
                    "page_table",
                    "the published mapping is shorter than the history",
                ));
            }
            Ok(())
        }

        /// Check one block of a full launch for the partial-producing symbol.
        ///
        /// A `PagedAttentionLaunch` describes the complete visible history and
        /// therefore includes the query row. The partial kernel sees one block
        /// of that history at a time, so its block range is checked separately:
        /// the resident block must be in this run's published pages, while the
        /// staged block is exactly one bounded page supplied by the caller.
        fn check_partial(
            &self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            block_base: u64,
            block_rows: u64,
            staged: bool,
        ) -> Result<()> {
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            self.same_device(stream)?;
            if self.partial_symbol.is_none() {
                return Err(unsupported_kernel_fmt(
                    "paged_attention_partial",
                    format_args!(
                        "the run was admitted without the separately-qualified partial symbol"
                    ),
                ));
            }
            super::descriptor_serves(&self.descriptor, launch)?;
            check_grid(launch, self.ctx)?;
            if *launch.geometry() != self.geometry
                || launch.heads() != self.heads
                || launch.rows() > self.max_rows
            {
                return Err(invalid(
                    "launch",
                    "this partial launch's geometry is not the one this run was admitted for",
                ));
            }
            if block_rows == 0 || block_rows > self.geometry.page_tokens {
                return Err(invalid(
                    "block_rows",
                    "a partial launch must cover one nonempty page or less",
                ));
            }
            if !block_base.is_multiple_of(self.geometry.page_tokens) {
                return Err(invalid(
                    "block_base",
                    "a partial block must start on a page boundary",
                ));
            }
            let block_end = block_base
                .checked_add(block_rows)
                .ok_or(Error::Dim(DimError::Overflow))?;
            let history_end = launch
                .history_base()
                .checked_add(launch.history_rows())
                .ok_or(Error::Dim(DimError::Overflow))?;
            if block_base < launch.history_base() || block_end > history_end {
                return Err(invalid(
                    "block",
                    "the partial block is outside the launch's visible history",
                ));
            }
            if staged {
                return Ok(());
            }
            if block_base != self.page_table_base {
                return Err(invalid_fmt(
                    "block_base",
                    format_args!(
                        "the resident partial starts at {block_base}, but the published mapping \
                         starts at {}",
                        self.page_table_base
                    ),
                ));
            }
            if block_end > self.written {
                return Err(invalid_fmt(
                    "block_rows",
                    format_args!(
                        "the resident partial reads through {block_end}, but {} row(s) were \
                         copied",
                        self.written
                    ),
                ));
            }
            if block_rows.div_ceil(self.geometry.page_tokens) > self.page_table.len() as u64 {
                return Err(invalid(
                    "page_table",
                    "the published mapping is shorter than the resident partial",
                ));
            }
            Ok(())
        }

        /// Stage the current host block into the one bounded device page
        /// admitted for a host-backed stream.
        fn stage_stream(&mut self, stream: &Stream<'ctx>) -> Result<()> {
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            self.same_device(stream)?;
            let (keys, values, table) = match self.held.as_ref() {
                Some(RefusedSource::Stream {
                    keys,
                    values,
                    table,
                    ..
                }) => (keys.as_slice(), values.as_slice(), *table),
                _ => return Err(invalid("stream", "no host block is retained for staging")),
            };
            let key_range = self
                .staged_keys
                .first()
                .and_then(Option::as_ref)
                .expect("two-block admission has staged keys");
            let value_range = self
                .staged_values
                .first()
                .and_then(Option::as_ref)
                .expect("two-block admission has staged values");
            let table_range = self
                .staged_table
                .first()
                .and_then(Option::as_ref)
                .expect("two-block admission has a staged table");
            if keys.len() as u64 > key_range.bytes() || values.len() as u64 > value_range.bytes() {
                return Err(invalid(
                    "stream",
                    "the host block exceeds its bounded device staging page",
                ));
            }
            // SAFETY: `self.held` owns `keys` until this settled copy completes,
            // and `key_range` is this run's admitted device staging range.
            if let Err(error) = unsafe { key_range.copy_from_host_async(keys, stream) } {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            #[cfg(feature = "paged-attention-test-hooks")]
            let injected_failure = match self.staging_failure_after {
                Some(0) => {
                    self.staging_failure_after = None;
                    true
                }
                Some(remaining) => {
                    self.staging_failure_after = Some(remaining - 1);
                    false
                }
                None => false,
            };
            #[cfg(feature = "paged-attention-test-hooks")]
            if injected_failure {
                self.quarantined = true;
                return Err(invalid(
                    "staging",
                    "injected failure after the staged key copy",
                ));
            }
            // SAFETY: the source remains in `self.held` and the destination is
            // the run's admitted value staging range until `settle` returns.
            if let Err(error) = unsafe { value_range.copy_from_host_async(values, stream) } {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            // SAFETY: the table bytes are copied from the retained source into
            // this run's admitted one-entry staging range.
            if let Err(error) = unsafe { table_range.copy_from_host_async(&table, stream) } {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            self.settle(Ok(()), stream)
        }

        /// Launch the separate partial-producing ABI over one block.
        fn launch_partial(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            block_base: u64,
            block_rows: u64,
            staging_page: Option<usize>,
        ) -> Result<()> {
            let query_address = self
                .query
                .as_ref()
                .expect("two-block admission has a query range")
                .device_address()?;
            let key_address = if let Some(page) = staging_page {
                self.staged_keys
                    .get(page)
                    .and_then(Option::as_ref)
                    .expect("two-block admission has staged keys")
                    .device_address()?
            } else {
                self.keys
                    .as_ref()
                    .expect("live key range")
                    .device_address()?
            };
            let value_address = if let Some(page) = staging_page {
                self.staged_values
                    .get(page)
                    .and_then(Option::as_ref)
                    .expect("two-block admission has staged values")
                    .device_address()?
            } else {
                self.values
                    .as_ref()
                    .expect("live value range")
                    .device_address()?
            };
            let table_address = if let Some(page) = staging_page {
                self.staged_table
                    .get(page)
                    .and_then(Option::as_ref)
                    .expect("two-block admission has a staged table")
                    .device_address()?
            } else {
                self.table
                    .as_ref()
                    .expect("live page table range")
                    .device_address()?
            };
            let partial_max = self
                .partial_max
                .as_ref()
                .expect("two-block admission has a partial max range")
                .device_address()?;
            let partial_sum = self
                .partial_sum
                .as_ref()
                .expect("two-block admission has a partial sum range")
                .device_address()?;
            let partial_weighted = self
                .partial_weighted
                .as_ref()
                .expect("two-block admission has weighted partials")
                .device_address()?;
            let scalars = Self::abi_scalars(launch)?;
            let AbiScalars {
                rows,
                first_position,
                heads,
                kv_heads,
                head_dim,
                page_tokens,
                window,
                scale,
                grid,
                ..
            } = scalars;
            let mut query_address = query_address;
            let mut key_address = key_address;
            let mut value_address = value_address;
            let mut table_address = table_address;
            let mut partial_max = partial_max;
            let mut partial_sum = partial_sum;
            let mut partial_weighted = partial_weighted;
            let mut rows = rows;
            let mut first_position = first_position;
            let mut block_base = block_base;
            let mut block_rows = block_rows;
            let mut heads = heads;
            let mut kv_heads = kv_heads;
            let mut head_dim = head_dim;
            let mut page_tokens = page_tokens;
            let mut window = window;
            let mut scale = scale;
            let mut params: [*mut c_void; 17] = [
                (&raw mut query_address).cast(),
                (&raw mut key_address).cast(),
                (&raw mut value_address).cast(),
                (&raw mut table_address).cast(),
                (&raw mut partial_max).cast(),
                (&raw mut partial_sum).cast(),
                (&raw mut partial_weighted).cast(),
                (&raw mut rows).cast(),
                (&raw mut first_position).cast(),
                (&raw mut block_base).cast(),
                (&raw mut block_rows).cast(),
                (&raw mut heads).cast(),
                (&raw mut kv_heads).cast(),
                (&raw mut head_dim).cast(),
                (&raw mut page_tokens).cast(),
                (&raw mut window).cast(),
                (&raw mut scale).cast(),
            ];
            let symbol = self
                .partial_symbol
                .expect("check_partial validated the partial symbol");
            // SAFETY: the separately-qualified symbol's ABI is declared beside
            // the kernel. Every address names an admitted range and every
            // scalar was checked before this launch.
            let launched = unsafe {
                self.module.launch_async(
                    symbol,
                    stream,
                    grid,
                    (moxie_kernels::PAGED_ATTENTION_THREADS, 1, 1),
                    0,
                    &mut params,
                )
            };
            self.settle(launched, stream)
        }

        /// Read the FP32 partial buffers after the producing launch settled.
        fn read_partials(&self, rows: u64) -> Result<Vec<DevicePartial>> {
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            let count = rows
                .checked_mul(self.heads)
                .ok_or(Error::Dim(DimError::Overflow))?;
            let scalar_bytes = count.checked_mul(4).ok_or(Error::Dim(DimError::Overflow))?;
            let weighted_bytes = count
                .checked_mul(self.geometry.head_dim)
                .and_then(|v| v.checked_mul(4))
                .ok_or(Error::Dim(DimError::Overflow))?;
            let scalar_len = usize::try_from(scalar_bytes)
                .map_err(|_| invalid("partial", "scalar readback exceeds host addressability"))?;
            let weighted_len = usize::try_from(weighted_bytes)
                .map_err(|_| invalid("partial", "weighted readback exceeds host addressability"))?;
            let mut max_bytes = super::try_zeroed(scalar_len)?;
            let mut sum_bytes = super::try_zeroed(scalar_len)?;
            let mut weighted_bytes_host = super::try_zeroed(weighted_len)?;
            self.partial_max
                .as_ref()
                .expect("two-block admission has a partial max range")
                .copy_to_host(&mut max_bytes)?;
            self.partial_sum
                .as_ref()
                .expect("two-block admission has a partial sum range")
                .copy_to_host(&mut sum_bytes)?;
            self.partial_weighted
                .as_ref()
                .expect("two-block admission has weighted partials")
                .copy_to_host(&mut weighted_bytes_host)?;
            let heads = usize::try_from(self.heads)
                .map_err(|_| invalid("heads", "head count exceeds host addressability"))?;
            let head_dim = usize::try_from(self.geometry.head_dim)
                .map_err(|_| invalid("head_dim", "head dimension exceeds host addressability"))?;
            let count = usize::try_from(count)
                .map_err(|_| invalid("partial", "partial count exceeds host addressability"))?;
            let mut partials = Vec::new();
            partials
                .try_reserve_exact(count)
                .map_err(|_| Error::CapacityExceeded {
                    tier: Some(Tier::Host(HostTier::Pageable)),
                    requested_bytes: scalar_bytes,
                    available_bytes: 0,
                })?;
            for index in 0..count {
                let scalar_offset = index * 4;
                let max = f32::from_le_bytes(
                    max_bytes[scalar_offset..scalar_offset + 4]
                        .try_into()
                        .expect("scalar readback is four-byte aligned"),
                );
                let sum = f32::from_le_bytes(
                    sum_bytes[scalar_offset..scalar_offset + 4]
                        .try_into()
                        .expect("scalar readback is four-byte aligned"),
                );
                let mut weighted = Vec::new();
                weighted
                    .try_reserve_exact(head_dim)
                    .map_err(|_| Error::CapacityExceeded {
                        tier: Some(Tier::Host(HostTier::Pageable)),
                        requested_bytes: self.geometry.head_dim * 8,
                        available_bytes: 0,
                    })?;
                weighted.resize(head_dim, 0.0);
                let partial = if sum == 0.0 {
                    if max != f32::NEG_INFINITY {
                        return Err(Error::Numerical {
                            detail: super::fallible(format_args!(
                                "empty device partial {index} did not carry negative infinity"
                            )),
                        });
                    }
                    DevicePartial { max, sum, weighted }
                } else {
                    if !(max.is_finite() && sum.is_finite() && sum > 0.0) {
                        return Err(Error::Numerical {
                            detail: super::fallible(format_args!(
                                "device partial {index} has max {max} and sum {sum}"
                            )),
                        });
                    }
                    let weighted_offset = index
                        .checked_mul(head_dim)
                        .and_then(|v| v.checked_mul(4))
                        .ok_or(Error::Dim(DimError::Overflow))?;
                    for (component, slot) in weighted.iter_mut().enumerate() {
                        let offset = weighted_offset + component * 4;
                        let value = f32::from_le_bytes(
                            weighted_bytes_host[offset..offset + 4]
                                .try_into()
                                .expect("weighted readback is four-byte aligned"),
                        );
                        if !value.is_finite() {
                            return Err(Error::Numerical {
                                detail: super::fallible(format_args!(
                                    "device partial {index} has nonfinite weighted value {value}"
                                )),
                            });
                        }
                        *slot = value;
                    }
                    DevicePartial { max, sum, weighted }
                };
                partials.push(partial);
            }
            debug_assert_eq!(partials.len(), rows as usize * heads);
            Ok(partials)
        }

        /// Attend `launch.rows` query rows against the rows this run has
        /// physically written.
        ///
        /// The launch's declared history must be one this run actually holds:
        /// a launch that claimed more rows than were written would attend over
        /// uninitialized pages and return a confident wrong answer. This is a
        /// fact about copied bytes, not the state authority's committed
        /// frontier -- `self.written` and `moxie_state`'s frontier are checked
        /// against each other elsewhere, not conflated here.
        #[allow(clippy::result_large_err)]
        pub fn attend(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: Vec<u8>,
        ) -> std::result::Result<Vec<u8>, PagedRunRefused> {
            let give_back = |error, query| PagedRunRefused {
                error,
                source: Some(RefusedSource::Query(query)),
            };
            if let Err(error) = self.observe_pending() {
                return Err(give_back(error, query));
            }
            // A run that admitted no staging buffers has nowhere to put a host
            // query, and that is the point of admitting none: a direct-device
            // caller is not charged for a path it does not use.
            if self.staging != Staging::Host {
                return Err(give_back(
                    invalid(
                        "staging",
                        "this run admitted no staging buffers; attend_into takes the \
                         caller's own device ranges",
                    ),
                    query,
                ));
            }
            // The same preconditions `attend_into` applies, because there is
            // one operation: the difference between the two entry points is
            // where the query already is, not what makes a launch legal. The
            // physical half of the history check lives in there — the authority
            // refuses a launch reading rows it never committed, this refuses one
            // reading bytes this run never wrote, and neither subsumes the
            // other.
            if let Err(error) = self.check_attend(stream, launch) {
                return Err(give_back(error, query));
            }
            match launch.query_bytes() {
                Ok(want) if query.len() as u64 == want => {}
                Ok(want) => {
                    return Err(give_back(
                        invalid_fmt(
                            "query",
                            format_args!(
                                "{} query byte(s) where the launch needs {want}",
                                query.len()
                            ),
                        ),
                        query,
                    ));
                }
                Err(error) => return Err(give_back(error, query)),
            }
            let out_len = match launch
                .output_bytes()
                .and_then(|b| usize::try_from(b).map_err(|_| invalid("output", "extent")))
                .and_then(super::try_zeroed)
            {
                Ok(out) => out,
                Err(error) => return Err(give_back(error, query)),
            };

            // --- from here on, work is in flight -------------------------
            // The host source is owned by `self.held`, not by this frame, from
            // before the first enqueue: a `Drop` while quarantined forgets it
            // rather than freeing bytes a DMA transfer may still be reading,
            // and nothing below clears it until completion is observed.
            self.held = Some(RefusedSource::Query(query));
            let Some(RefusedSource::Query(query_bytes)) = self.held.as_ref() else {
                unreachable!("just assigned")
            };
            // The run's own ranges, taken out of their slots so the shared
            // path below owns them exactly as a caller's own ranges would.
            // There is one launch operation, in `attend_into`; staging a host
            // query differs only in what happens before and after it.
            let query_range = self
                .query
                .take()
                .expect("a host-staged run has a query range");
            let output_range = self
                .output
                .take()
                .expect("a host-staged run has an output range");
            // SAFETY: the source is owned by `self.held` until completion is
            // observed below, and the destination is this run's own admitted
            // range.
            if let Err(error) = unsafe { query_range.copy_from_host_async(query_bytes, stream) } {
                self.quarantined = true;
                self.held_ranges = Some((query_range, output_range));
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            // The copy above is already enqueued on `stream`, so from here on
            // this run's completion is unknown regardless of what
            // `attend_into` enqueued for its own kernel launch: a "before
            // anything was enqueued" refusal from its perspective is not one
            // from this call's, and its ranges must not be handed back.
            let (query_range, output_range) =
                match self.attend_into(stream, launch, query_range, output_range) {
                    Ok(ranges) => ranges,
                    Err(refused) => {
                        self.quarantined = true;
                        if let Some(ranges) = refused.ranges {
                            self.held_ranges = Some(ranges);
                        }
                        // else: `attend_into` already moved them into `held_ranges`.
                        return Err(PagedRunRefused {
                            error: refused.error,
                            source: None,
                        });
                    }
                };
            let mut out = out_len;
            if let Err(error) = output_range.copy_to_host_at(0, &mut out) {
                self.quarantined = true;
                self.held_ranges = Some((query_range, output_range));
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            // Completion is observed: the host source and the device ranges
            // may be reclaimed and reused.
            self.held = None;
            self.query = Some(query_range);
            self.output = Some(output_range);
            Ok(out)
        }

        /// Attend one query over one resident page and one host-sourced page.
        ///
        /// `launch` describes the resident block. The host block must be the
        /// immediately following page and is staged into the one bounded page
        /// pair admitted by [`Staging::TwoBlock`]. Each block uses the new
        /// partial-producing symbol once; the raw partials are returned to the
        /// caller, which deliberately widens and merges them.
        #[allow(clippy::result_large_err)]
        pub fn attend_two_block(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: Vec<u8>,
            host_keys: Vec<u8>,
            host_values: Vec<u8>,
        ) -> std::result::Result<TwoBlockAttention, PagedRunRefused> {
            let table = [0u8; 4];
            let give_back = |error, query, keys, values| PagedRunRefused {
                error,
                source: Some(RefusedSource::Stream {
                    query,
                    keys,
                    values,
                    table,
                }),
            };
            if let Err(error) = self.observe_pending() {
                return Err(give_back(error, query, host_keys, host_values));
            }
            if !launch.is_two_block_stream() {
                return Err(give_back(
                    invalid(
                        "launch",
                        "a two-block launch must come from two_block_stream",
                    ),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            if self.staging != Staging::TwoBlock {
                return Err(give_back(
                    invalid(
                        "staging",
                        "a two-block launch requires Staging::TwoBlock admission",
                    ),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            if launch.rows() != 1 {
                return Err(give_back(
                    invalid("rows", "the bounded streaming slice serves one query row"),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            let resident_rows = self.geometry.page_tokens;
            if launch.history_rows() <= resident_rows {
                return Err(give_back(
                    invalid(
                        "history_rows",
                        "the total history must contain a resident and a staged block",
                    ),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            if !launch
                .history_base()
                .is_multiple_of(self.geometry.page_tokens)
            {
                return Err(give_back(
                    invalid_fmt(
                        "history_base",
                        format_args!("{} is not page aligned", launch.history_base()),
                    ),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            let row_bytes = match self.geometry.row_elements().and_then(|elements| {
                elements
                    .checked_mul(PAYLOAD_BYTES)
                    .ok_or(Error::Dim(DimError::Overflow))
            }) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, query, host_keys, host_values)),
            };
            let host_rows = match u64::try_from(host_keys.len())
                .ok()
                .and_then(|len| len.checked_div(row_bytes))
            {
                Some(rows) if rows > 0 && rows <= self.geometry.page_tokens => rows,
                _ => {
                    return Err(give_back(
                        invalid(
                            "host_block",
                            "the staged block must contain one to one page of complete rows",
                        ),
                        query,
                        host_keys,
                        host_values,
                    ));
                }
            };
            if host_keys.len() as u64 != host_rows * row_bytes
                || host_values.len() as u64 != host_rows * row_bytes
            {
                return Err(give_back(
                    invalid(
                        "host_block",
                        "host key and value blocks must have equal row width",
                    ),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            let total_rows = match resident_rows.checked_add(host_rows) {
                Some(rows) if rows == launch.history_rows() => rows,
                _ => {
                    return Err(give_back(
                        invalid_fmt(
                            "host_block",
                            format_args!(
                                "resident block plus staged block must equal the {}-row history",
                                launch.history_rows()
                            ),
                        ),
                        query,
                        host_keys,
                        host_values,
                    ));
                }
            };
            let host_base = match launch.history_base().checked_add(resident_rows) {
                Some(base) => base,
                None => {
                    return Err(give_back(
                        Error::Dim(DimError::Overflow),
                        query,
                        host_keys,
                        host_values,
                    ));
                }
            };
            debug_assert_eq!(total_rows, launch.history_rows());
            let query_bytes = match launch.query_bytes() {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, query, host_keys, host_values)),
            };
            if query.len() as u64 != query_bytes {
                return Err(give_back(
                    invalid_fmt(
                        "query",
                        format_args!("{} byte(s) supplied, expected {query_bytes}", query.len()),
                    ),
                    query,
                    host_keys,
                    host_values,
                ));
            }
            if let Err(error) =
                self.check_partial(stream, launch, launch.history_base(), resident_rows, false)
            {
                return Err(give_back(error, query, host_keys, host_values));
            }
            if let Err(error) = self.check_partial(stream, launch, host_base, host_rows, true) {
                return Err(give_back(error, query, host_keys, host_values));
            }
            let host_to_device_bytes = (host_keys.len() as u64)
                .checked_add(host_values.len() as u64)
                .and_then(|bytes| bytes.checked_add(table.len() as u64));
            let Some(host_to_device_bytes) = host_to_device_bytes else {
                return Err(give_back(
                    Error::Dim(DimError::Overflow),
                    query,
                    host_keys,
                    host_values,
                ));
            };

            self.held = Some(RefusedSource::Stream {
                query,
                keys: host_keys,
                values: host_values,
                table,
            });
            let Some(RefusedSource::Stream { query, .. }) = self.held.as_ref() else {
                unreachable!("just assigned")
            };
            // SAFETY: `self.held` owns the query until the resident and staged
            // partial launches have settled, and the destination is admitted
            // exclusively to this run.
            if let Err(error) = unsafe {
                self.query
                    .as_ref()
                    .expect("two-block admission has a query range")
                    .copy_from_host_async(query, stream)
            } {
                self.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            if let Err(error) =
                self.launch_partial(stream, launch, launch.history_base(), resident_rows, None)
            {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            let resident = match self.read_partials(launch.rows()) {
                Ok(partials) => partials,
                Err(error) => {
                    self.quarantined = true;
                    return Err(PagedRunRefused {
                        error: self.attribute(error),
                        source: None,
                    });
                }
            };
            if let Err(error) = self.stage_stream(stream) {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            if let Err(error) = self.launch_partial(stream, launch, host_base, host_rows, Some(0)) {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            let staged = match self.read_partials(launch.rows()) {
                Ok(partials) => partials,
                Err(error) => {
                    self.quarantined = true;
                    return Err(PagedRunRefused {
                        error: self.attribute(error),
                        source: None,
                    });
                }
            };
            self.held = None;
            Ok(TwoBlockAttention {
                resident,
                staged,
                host_to_device_bytes,
            })
        }

        /// Start one query over one resident page and a bounded sequence of
        /// host-sourced pages. The resident partial settles before this
        /// returns; the caller then supplies each staged page to
        /// [`NBlockStream::stage_next`] and folds it before reading the next.
        #[allow(clippy::result_large_err)]
        pub fn start_n_block<'run>(
            &'run mut self,
            stream: &'run Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: Vec<u8>,
        ) -> std::result::Result<(NBlockStream<'run, 'ctx>, Vec<DevicePartial>), PagedRunRefused>
        {
            let give_back = |error, query| PagedRunRefused {
                error,
                source: Some(RefusedSource::Stream {
                    query,
                    keys: Vec::new(),
                    values: Vec::new(),
                    table: [0u8; 4],
                }),
            };
            if let Err(error) = self.observe_pending() {
                return Err(give_back(error, query));
            }
            if !launch.is_host_stream() || launch.is_two_block_stream() {
                return Err(give_back(
                    invalid("launch", "an N-block launch must come from n_block_stream"),
                    query,
                ));
            }
            if !self.staging.partial_buffers()
                || self.staging.max_staged_blocks() < launch.staged_blocks()
            {
                return Err(give_back(
                    invalid_fmt(
                        "staging",
                        format_args!(
                            "the admitted host-backed bound is {} staged block(s), but the \
                             launch needs {}",
                            self.staging.max_staged_blocks(),
                            launch.staged_blocks()
                        ),
                    ),
                    query,
                ));
            }
            if launch.rows() != 1 {
                return Err(give_back(
                    invalid("rows", "the bounded streaming slice serves one query row"),
                    query,
                ));
            }
            let staged_count = launch.staged_blocks();
            let resident_rows = self.geometry.page_tokens;
            if !launch
                .history_base()
                .is_multiple_of(self.geometry.page_tokens)
            {
                return Err(give_back(
                    invalid_fmt(
                        "history_base",
                        format_args!("{} is not page aligned", launch.history_base()),
                    ),
                    query,
                ));
            }
            let query_bytes = match launch.query_bytes() {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, query)),
            };
            if query.len() as u64 != query_bytes {
                return Err(give_back(
                    invalid_fmt(
                        "query",
                        format_args!("{} byte(s) supplied, expected {query_bytes}", query.len()),
                    ),
                    query,
                ));
            }
            if let Err(error) =
                self.check_partial(stream, launch, launch.history_base(), resident_rows, false)
            {
                return Err(give_back(error, query));
            }
            let block_base = match launch.history_base().checked_add(resident_rows) {
                Some(base) => base,
                None => {
                    return Err(give_back(Error::Dim(DimError::Overflow), query));
                }
            };

            self.held = Some(RefusedSource::Stream {
                query,
                keys: Vec::new(),
                values: Vec::new(),
                table: [0u8; 4],
            });
            let Some(RefusedSource::Stream { query, .. }) = self.held.as_ref() else {
                unreachable!("just assigned")
            };
            // SAFETY: the source is owned by `self.held` until the resident
            // partial has settled, and the destination is this run's own
            // admitted query range.
            if let Err(error) = unsafe {
                self.query
                    .as_ref()
                    .expect("host-backed admission has a query range")
                    .copy_from_host_async(query, stream)
            } {
                self.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            if let Err(error) =
                self.launch_partial(stream, launch, launch.history_base(), resident_rows, None)
            {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            let resident = match self.read_partials(launch.rows()) {
                Ok(partials) => partials,
                Err(error) => {
                    self.quarantined = true;
                    return Err(PagedRunRefused {
                        error: self.attribute(error),
                        source: None,
                    });
                }
            };
            self.held = None;
            let remaining_rows = launch
                .history_rows()
                .checked_sub(resident_rows)
                .expect("host stream has a resident page");
            let read_ahead = matches!(self.staging, Staging::HostBackedReadAhead { .. });
            self.read_ahead_stream_active = read_ahead;
            Ok((
                NBlockStream {
                    run: self,
                    stream,
                    launch: *launch,
                    next_base: block_base,
                    remaining_rows,
                    remaining_blocks: staged_count,
                    host_to_device_bytes: 0,
                    prefetched_total: 0,
                    folded_total: 0,
                    outstanding_rows: 0,
                    page_rows: [0, 0],
                },
                resident,
            ))
        }

        /// Every scalar this ABI needs, derived **before** anything is
        /// enqueued.
        ///
        /// `PagedAttentionLaunch::new` has already refused any value that does
        /// not fit, so none of these conversions can fail on a launch that
        /// exists. They are still fallible and they still happen here, because
        /// the alternative is what this code did before: discover an unfitting
        /// window *after* submitting the query copy, and turn a refusal that was
        /// knowable up front into an unknown submission and a quarantined run.
        fn abi_scalars(launch: &PagedAttentionLaunch) -> Result<AbiScalars> {
            Ok(AbiScalars {
                rows: launch.rows(),
                first_position: launch.first_position(),
                history_base: launch.history_base(),
                history_rows: launch.history_rows(),
                heads: u32::try_from(launch.heads())
                    .map_err(|_| invalid("heads", "the head count exceeds a u32"))?,
                kv_heads: u32::try_from(launch.geometry().kv_heads)
                    .map_err(|_| invalid("kv_heads", "the key/value head count exceeds a u32"))?,
                head_dim: u32::try_from(launch.geometry().head_dim)
                    .map_err(|_| invalid("head_dim", "the head dimension exceeds a u32"))?,
                page_tokens: u32::try_from(launch.geometry().page_tokens)
                    .map_err(|_| invalid("page_tokens", "the page width exceeds a u32"))?,
                window: launch.window()?,
                scale: launch.scale(),
                grid: launch.grid()?,
            })
        }

        /// The launch itself: five addresses, the ABI's scalars, one grid.
        ///
        /// Shared by both entry points, because there is one kernel and one ABI
        /// and the only difference between staging through this run's ranges
        /// and attending into a caller's is which addresses arrive here.
        fn launch_with(
            &mut self,
            stream: &Stream<'ctx>,
            scalars: AbiScalars,
            addresses: [u64; 5],
            defer: bool,
        ) -> Result<()> {
            let [
                mut query_address,
                mut key_address,
                mut value_address,
                mut table_address,
                mut output_address,
            ] = addresses;
            let AbiScalars {
                mut rows,
                mut first_position,
                mut history_base,
                mut history_rows,
                mut heads,
                mut kv_heads,
                mut head_dim,
                mut page_tokens,
                mut window,
                mut scale,
                grid,
            } = scalars;
            let mut params: [*mut c_void; 15] = [
                (&raw mut query_address).cast(),
                (&raw mut key_address).cast(),
                (&raw mut value_address).cast(),
                (&raw mut table_address).cast(),
                (&raw mut output_address).cast(),
                (&raw mut rows).cast(),
                (&raw mut first_position).cast(),
                (&raw mut history_base).cast(),
                (&raw mut history_rows).cast(),
                (&raw mut heads).cast(),
                (&raw mut kv_heads).cast(),
                (&raw mut head_dim).cast(),
                (&raw mut page_tokens).cast(),
                (&raw mut window).cast(),
                (&raw mut scale).cast(),
            ];
            // SAFETY: the symbol's ABI is the one declared in
            // `paged_attention.cu`; every pointer names a live admitted range
            // whose extent was checked above, and the grid is one block per
            // query row and head.
            let launched = unsafe {
                self.module.launch_async(
                    0,
                    stream,
                    grid,
                    (moxie_kernels::PAGED_ATTENTION_THREADS, 1, 1),
                    0,
                    &mut params,
                )
            };
            if defer {
                self.defer(launched, stream)
            } else {
                self.settle(launched, stream)
            }
        }

        fn settle(&mut self, launched: Result<()>, stream: &Stream<'ctx>) -> Result<()> {
            if let Err(error) = launched {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            let event = match Event::new(self.ctx) {
                Ok(event) => event,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            if let Err(error) = event.record(stream) {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            if let Err(error) = event.synchronize() {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            Ok(())
        }

        fn defer(&mut self, launched: Result<()>, stream: &Stream<'ctx>) -> Result<()> {
            if let Err(error) = launched {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            let event = match Event::new(self.ctx) {
                Ok(event) => event,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            if let Err(error) = event.record(stream) {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            self.pending = Some(event);
            Ok(())
        }

        fn attribute(&self, error: Error) -> Error {
            match error {
                // Fallible: this runs where a launch has already failed, which
                // is where memory pressure is most likely, and `format!` there
                // turns a typed device error into `SIGABRT`.
                Error::DeviceLost { detail, .. } => Error::DeviceLost {
                    device: self.ctx.ordinal(),
                    detail: super::fallible(format_args!(
                        "kernel {}: {detail}",
                        self.descriptor.id.0
                    )),
                },
                other => other,
            }
        }

        /// Release the pages, the per-step ranges and their charge.
        ///
        /// Refuses while quarantined: work whose completion is unknown may
        /// still be reading these ranges, including whatever `held_ranges`
        /// holds -- which stays inside the returned run rather than being
        /// exposed, because handing it back would let a caller reuse exactly
        /// the ranges this refusal says may still be in flight. `Drop` is what
        /// disposes of it if the caller gives up on retrying.
        #[allow(clippy::result_large_err)]
        pub fn close(
            mut self,
            ledger: &mut Ledger,
        ) -> std::result::Result<(), PagedCloseRefused<'ctx>> {
            let read_ahead = matches!(self.staging, Staging::HostBackedReadAhead { .. });
            if let Err(error) = self.observe_pending() {
                if !read_ahead || self.ctx.synchronize().is_err() {
                    return Err(PagedCloseRefused { run: self, error });
                }
                self.pending = None;
                self.copied_pending = [false, false];
                self.held = None;
                self.quarantined = false;
            }
            if self.quarantined {
                if read_ahead {
                    if let Err(error) = self.ctx.synchronize() {
                        return Err(PagedCloseRefused { run: self, error });
                    }
                    self.pending = None;
                    self.copied_pending = [false, false];
                    self.held = None;
                    self.quarantined = false;
                } else {
                    let error = invalid(
                        "close",
                        "this run is quarantined; its ranges may still be in flight",
                    );
                    return Err(PagedCloseRefused { run: self, error });
                }
            }
            if ledger.id() != self.ledger {
                let error = invalid("ledger", "this run belongs to another ledger");
                return Err(PagedCloseRefused { run: self, error });
            }
            if let Some(pinned) = self.pinned_bounce.take()
                && let Err((pinned, error)) = pinned.free()
            {
                self.pinned_bounce = Some(pinned);
                return Err(PagedCloseRefused { run: self, error });
            }
            for slot in 0..14 {
                let taken = match slot {
                    0 => self.output.take(),
                    1 => self.query.take(),
                    2 => self.table.take(),
                    3 => self.values.take(),
                    4 => self.keys.take(),
                    5 => self.staged_table[0].take(),
                    6 => self.staged_values[0].take(),
                    7 => self.staged_keys[0].take(),
                    8 => self.partial_weighted.take(),
                    9 => self.partial_sum.take(),
                    10 => self.partial_max.take(),
                    11 => self.staged_table[1].take(),
                    12 => self.staged_values[1].take(),
                    _ => self.staged_keys[1].take(),
                };
                let Some(range) = taken else { continue };
                let arena = self.arena.as_mut().expect("an open run has its arena");
                if let Err(refused) = arena.release(range) {
                    // Put the range back in the slot it came from. The arena
                    // still records this allocation as live either way; losing
                    // the handle here would make that permanent, because
                    // nothing else names this allocation to retry it with.
                    match slot {
                        0 => self.output = Some(refused.range),
                        1 => self.query = Some(refused.range),
                        2 => self.table = Some(refused.range),
                        3 => self.values = Some(refused.range),
                        4 => self.keys = Some(refused.range),
                        5 => self.staged_table[0] = Some(refused.range),
                        6 => self.staged_values[0] = Some(refused.range),
                        7 => self.staged_keys[0] = Some(refused.range),
                        8 => self.partial_weighted = Some(refused.range),
                        9 => self.partial_sum = Some(refused.range),
                        10 => self.partial_max = Some(refused.range),
                        11 => self.staged_table[1] = Some(refused.range),
                        12 => self.staged_values[1] = Some(refused.range),
                        _ => self.staged_keys[1] = Some(refused.range),
                    }
                    let error = refused.error;
                    return Err(PagedCloseRefused { run: self, error });
                }
            }
            let arena = self.arena.take().expect("an open run has its arena");
            match arena.close(ledger) {
                Ok(()) => Ok(()),
                Err(refused) => {
                    self.arena = Some(refused.arena);
                    let error = refused.error;
                    Err(PagedCloseRefused { run: self, error })
                }
            }
        }
    }

    impl<'run, 'ctx> NBlockStream<'run, 'ctx> {
        fn refused_current(&mut self, error: Error) -> PagedRunRefused {
            let source = if self.run.quarantined {
                None
            } else {
                self.run.held.take()
            };
            PagedRunRefused { error, source }
        }

        /// Stage exactly one host page, launch its partial, and read the
        /// result before returning. The caller can therefore fold the result
        /// and drop the page before obtaining the next page from its store.
        #[allow(clippy::result_large_err)]
        pub fn stage_next(
            &mut self,
            keys: Vec<u8>,
            values: Vec<u8>,
        ) -> std::result::Result<Vec<DevicePartial>, PagedRunRefused> {
            let give_back = |error, keys, values| PagedRunRefused {
                error,
                source: Some(RefusedSource::Stream {
                    query: Vec::new(),
                    keys,
                    values,
                    table: [0u8; 4],
                }),
            };
            if matches!(self.run.staging, Staging::HostBackedReadAhead { .. }) {
                return Err(give_back(
                    invalid("stream", "this stream was admitted for read-ahead prefetch"),
                    keys,
                    values,
                ));
            }
            if let Err(error) = self.run.observe_pending() {
                return Err(give_back(error, keys, values));
            }
            if self.run.quarantined {
                return Err(give_back(
                    invalid("run", "this run is quarantined"),
                    keys,
                    values,
                ));
            }
            if self.remaining_blocks == 0 || self.remaining_rows == 0 {
                return Err(give_back(
                    invalid("stream", "all host-backed blocks have already been staged"),
                    keys,
                    values,
                ));
            }
            let row_bytes = match self.run.geometry.row_elements().and_then(|elements| {
                elements
                    .checked_mul(PAYLOAD_BYTES)
                    .ok_or(Error::Dim(DimError::Overflow))
            }) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, keys, values)),
            };
            let expected_rows = self.remaining_rows.min(self.run.geometry.page_tokens);
            let rows = match u64::try_from(keys.len())
                .ok()
                .and_then(|len| len.checked_div(row_bytes))
            {
                Some(rows) if rows == expected_rows => rows,
                _ => {
                    return Err(give_back(
                        invalid(
                            "host_block",
                            "each staged block must contain the next complete page or tail",
                        ),
                        keys,
                        values,
                    ));
                }
            };
            let expected_bytes = match rows.checked_mul(row_bytes) {
                Some(bytes) => bytes,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            if keys.len() as u64 != expected_bytes || values.len() as u64 != expected_bytes {
                return Err(give_back(
                    invalid(
                        "host_block",
                        "host key and value blocks must have equal complete row widths",
                    ),
                    keys,
                    values,
                ));
            }
            let transfer = match expected_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(PAGE_ENTRY_BYTES))
            {
                Some(bytes) => bytes,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let total_transfer = match self.host_to_device_bytes.checked_add(transfer) {
                Some(total) => total,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            if let Err(error) =
                self.run
                    .check_partial(self.stream, &self.launch, self.next_base, rows, true)
            {
                return Err(give_back(error, keys, values));
            }

            self.run.held = Some(RefusedSource::Stream {
                query: Vec::new(),
                keys,
                values,
                table: [0u8; 4],
            });
            if let Err(error) = self.run.stage_stream(self.stream) {
                return Err(self.refused_current(error));
            }
            if let Err(error) =
                self.run
                    .launch_partial(self.stream, &self.launch, self.next_base, rows, Some(0))
            {
                return Err(self.refused_current(error));
            }
            let partials = match self.run.read_partials(self.launch.rows()) {
                Ok(partials) => partials,
                Err(error) => {
                    let error = self.run.attribute(error);
                    self.run.quarantined = true;
                    return Err(PagedRunRefused {
                        error,
                        source: None,
                    });
                }
            };
            self.run.held = None;
            self.next_base = self
                .next_base
                .checked_add(rows)
                .expect("validated host block range cannot overflow");
            self.remaining_rows -= rows;
            self.remaining_blocks -= 1;
            self.host_to_device_bytes = total_transfer;
            Ok(partials)
        }

        /// Copy one host page into a pinned bounce slot and enqueue its device
        /// copies without waiting for the compute stream.
        #[allow(clippy::result_large_err)]
        pub fn prefetch(
            &mut self,
            keys: Vec<u8>,
            values: Vec<u8>,
        ) -> std::result::Result<(), PagedRunRefused> {
            let give_back = |error, keys, values| PagedRunRefused {
                error,
                source: Some(RefusedSource::Stream {
                    query: Vec::new(),
                    keys,
                    values,
                    table: [0; 4],
                }),
            };
            if !matches!(self.run.staging, Staging::HostBackedReadAhead { .. }) {
                return Err(give_back(
                    invalid("stream", "this stream was not admitted for read-ahead"),
                    keys,
                    values,
                ));
            }
            let outstanding = self.prefetched_total - self.folded_total;
            if self.remaining_blocks == 0
                || self.remaining_rows <= self.outstanding_rows
                || outstanding >= 2
            {
                return Err(give_back(
                    invalid("stream", "no read-ahead slot is available"),
                    keys,
                    values,
                ));
            }
            if let Err(error) = self.run.observe_pending() {
                return Err(give_back(error, keys, values));
            }
            if self.run.quarantined {
                return Err(give_back(
                    invalid("run", "this run is quarantined"),
                    keys,
                    values,
                ));
            }
            let row_bytes = match self.run.geometry.row_elements().and_then(|elements| {
                elements
                    .checked_mul(PAYLOAD_BYTES)
                    .ok_or(Error::Dim(DimError::Overflow))
            }) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, keys, values)),
            };
            let expected_rows = self.remaining_rows - self.outstanding_rows;
            let expected_rows = expected_rows.min(self.run.geometry.page_tokens);
            let rows = match u64::try_from(keys.len())
                .ok()
                .and_then(|len| len.checked_div(row_bytes))
            {
                Some(rows) if rows == expected_rows => rows,
                _ => {
                    return Err(give_back(
                        invalid(
                            "host_block",
                            "each prefetched block must contain the next complete page or tail",
                        ),
                        keys,
                        values,
                    ));
                }
            };
            let expected_bytes = match rows.checked_mul(row_bytes) {
                Some(bytes) => bytes,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            if keys.len() as u64 != expected_bytes || values.len() as u64 != expected_bytes {
                return Err(give_back(
                    invalid(
                        "host_block",
                        "host key and value blocks must have equal complete row widths",
                    ),
                    keys,
                    values,
                ));
            }
            let transfer = match expected_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(PAGE_ENTRY_BYTES))
            {
                Some(bytes) => bytes,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let outstanding_bytes = match transfer.checked_mul(outstanding + 1) {
                Some(bytes) => bytes,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            if self
                .host_to_device_bytes
                .checked_add(outstanding_bytes)
                .is_none()
            {
                return Err(give_back(Error::Dim(DimError::Overflow), keys, values));
            }
            let next_prefetched = match self.prefetched_total.checked_add(1) {
                Some(total) => total,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let next_outstanding_rows = match self.outstanding_rows.checked_add(rows) {
                Some(total) => total,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let block_base = match self.next_base.checked_add(self.outstanding_rows) {
                Some(base) => base,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let copy_stream = self
                .run
                .copy_stream
                .as_ref()
                .expect("read-ahead admission owns a copy stream");
            if let Err(error) =
                self.run
                    .check_partial(copy_stream, &self.launch, block_base, rows, true)
            {
                return Err(give_back(error, keys, values));
            }
            let page =
                usize::try_from(self.prefetched_total % 2).expect("two-page index fits usize");
            let page_bytes = self.run.pinned_page_bytes;
            let payload_bytes = match self.run.geometry.page_bytes().and_then(|bytes| {
                usize::try_from(bytes)
                    .map_err(|_| invalid("host_block", "page size exceeds host addressability"))
            }) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, keys, values)),
            };
            let key_offset = match page.checked_mul(page_bytes) {
                Some(offset) => offset,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let value_offset = match key_offset.checked_add(payload_bytes) {
                Some(offset) => offset,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let table_offset = match value_offset.checked_add(payload_bytes) {
                Some(offset) => offset,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let slot_end = match key_offset.checked_add(page_bytes) {
                Some(end) => end,
                None => return Err(give_back(Error::Dim(DimError::Overflow), keys, values)),
            };
            let key_end = match key_offset.checked_add(keys.len()) {
                Some(end) if end <= value_offset => end,
                _ => {
                    return Err(give_back(
                        invalid("host_block", "key page exceeds its bounce slot"),
                        keys,
                        values,
                    ));
                }
            };
            let value_end = match value_offset.checked_add(values.len()) {
                Some(end) if end <= table_offset => end,
                _ => {
                    return Err(give_back(
                        invalid("host_block", "value page exceeds its bounce slot"),
                        keys,
                        values,
                    ));
                }
            };
            let table_end = match table_offset.checked_add(4) {
                Some(end) if end <= slot_end => end,
                _ => {
                    return Err(give_back(
                        invalid("host_block", "page-table entry exceeds its bounce slot"),
                        keys,
                        values,
                    ));
                }
            };
            let bounce = self
                .run
                .pinned_bounce
                .as_mut()
                .expect("read-ahead admission owns pinned bounce pages")
                .as_mut_slice();
            bounce[key_offset..key_end].copy_from_slice(&keys);
            bounce[value_offset..value_end].copy_from_slice(&values);
            bounce[table_offset..table_end].copy_from_slice(&[0; 4]);

            #[cfg(feature = "paged-attention-test-hooks")]
            let mut gate_enqueued = false;
            #[cfg(feature = "paged-attention-test-hooks")]
            if self.run.gate_next_prefetch {
                // SAFETY: the callback is static, takes no borrowed data, and
                // only reads the static release flag until its fixed deadline.
                if let Err(error) = unsafe {
                    copy_stream.launch_host_func(wait_for_prefetch_release, core::ptr::null_mut())
                } {
                    return Err(give_back(error, keys, values));
                }
                self.run.gate_next_prefetch = false;
                gate_enqueued = true;
            }

            // SAFETY: the host slice is from the run-owned pinned bounce slot;
            // it remains unchanged through the copy event and following fold.
            let key_result = unsafe {
                self.run.staged_keys[page]
                    .as_ref()
                    .expect("read-ahead admission owns both device key pages")
                    .copy_from_host_async(
                        &self
                            .run
                            .pinned_bounce
                            .as_ref()
                            .expect("pinned pages")
                            .as_slice()[key_offset..key_end],
                        copy_stream,
                    )
            };
            if let Err(error) = key_result {
                #[cfg(feature = "paged-attention-test-hooks")]
                if gate_enqueued && copy_stream.synchronize().is_err() {
                    self.run.quarantined = true;
                }
                return Err(give_back(self.run.attribute(error), keys, values));
            }
            self.prefetched_total = next_prefetched;
            self.outstanding_rows = next_outstanding_rows;
            self.page_rows[page] = rows;
            drop(keys);
            drop(values);

            // SAFETY: the host slice is from the run-owned pinned bounce slot;
            // it remains unchanged through the copy event and following fold.
            let value_result = unsafe {
                self.run.staged_values[page]
                    .as_ref()
                    .expect("read-ahead admission owns both device value pages")
                    .copy_from_host_async(
                        &self
                            .run
                            .pinned_bounce
                            .as_ref()
                            .expect("pinned pages")
                            .as_slice()[value_offset..value_end],
                        copy_stream,
                    )
            };
            if let Err(error) = value_result {
                self.run.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.run.attribute(error),
                    source: None,
                });
            }
            // SAFETY: the host slice is from the run-owned pinned bounce slot;
            // it remains unchanged through the copy event and following fold.
            let table_result = unsafe {
                self.run.staged_table[page]
                    .as_ref()
                    .expect("read-ahead admission owns both device table pages")
                    .copy_from_host_async(
                        &self
                            .run
                            .pinned_bounce
                            .as_ref()
                            .expect("pinned pages")
                            .as_slice()[table_offset..table_end],
                        copy_stream,
                    )
            };
            if let Err(error) = table_result {
                self.run.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.run.attribute(error),
                    source: None,
                });
            }
            let copied = self.run.copied[page]
                .as_ref()
                .expect("read-ahead admission owns copy events")
                .record(copy_stream);
            if let Err(error) = copied {
                self.run.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.run.attribute(error),
                    source: None,
                });
            }
            self.run.copied_pending[page] = true;
            Ok(())
        }

        /// Wait for the oldest prefetched page, fold it, and free its slots.
        #[allow(clippy::result_large_err)]
        pub fn fold_next(&mut self) -> std::result::Result<Vec<DevicePartial>, PagedRunRefused> {
            let refuse = |error| PagedRunRefused {
                error,
                source: None,
            };
            if !matches!(self.run.staging, Staging::HostBackedReadAhead { .. }) {
                return Err(refuse(invalid(
                    "stream",
                    "this stream was not admitted for read-ahead",
                )));
            }
            if let Err(error) = self.run.observe_pending() {
                return Err(refuse(error));
            }
            if self.run.quarantined {
                return Err(refuse(invalid("run", "this run is quarantined")));
            }
            if self.folded_total == self.prefetched_total {
                return Err(refuse(invalid(
                    "stream",
                    "no prefetched block is ready to fold",
                )));
            }
            let page = usize::try_from(self.folded_total % 2).expect("two-page index fits usize");
            let rows = self.page_rows[page];
            if rows == 0 || !self.run.copied_pending[page] {
                return Err(refuse(invalid(
                    "stream",
                    "the next prefetched page is incomplete",
                )));
            }
            let row_bytes = match self.run.geometry.row_elements().and_then(|elements| {
                elements
                    .checked_mul(PAYLOAD_BYTES)
                    .ok_or(Error::Dim(DimError::Overflow))
            }) {
                Ok(bytes) => bytes,
                Err(error) => return Err(refuse(error)),
            };
            let transfer = match rows
                .checked_mul(row_bytes)
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| bytes.checked_add(PAGE_ENTRY_BYTES))
            {
                Some(bytes) => bytes,
                None => return Err(refuse(Error::Dim(DimError::Overflow))),
            };
            let total_transfer = match self.host_to_device_bytes.checked_add(transfer) {
                Some(total) => total,
                None => return Err(refuse(Error::Dim(DimError::Overflow))),
            };
            let next_base = match self.next_base.checked_add(rows) {
                Some(base) => base,
                None => return Err(refuse(Error::Dim(DimError::Overflow))),
            };
            let next_folded = match self.folded_total.checked_add(1) {
                Some(total) => total,
                None => return Err(refuse(Error::Dim(DimError::Overflow))),
            };
            let next_outstanding_rows = match self.outstanding_rows.checked_sub(rows) {
                Some(total) => total,
                None => return Err(refuse(invalid("stream", "prefetched row count is invalid"))),
            };
            let next_remaining_rows = match self.remaining_rows.checked_sub(rows) {
                Some(total) => total,
                None => return Err(refuse(invalid("stream", "remaining row count is invalid"))),
            };
            let next_remaining_blocks = match self.remaining_blocks.checked_sub(1) {
                Some(total) => total,
                None => {
                    return Err(refuse(invalid(
                        "stream",
                        "remaining block count is invalid",
                    )));
                }
            };
            if let Err(error) =
                self.run
                    .check_partial(self.stream, &self.launch, self.next_base, rows, true)
            {
                return Err(refuse(error));
            }
            let event = self.run.copied[page]
                .as_ref()
                .expect("read-ahead admission owns copy events");
            if let Err(error) = self.stream.wait_event(event) {
                self.run.quarantined = true;
                return Err(refuse(self.run.attribute(error)));
            }
            if let Err(error) =
                self.run
                    .launch_partial(self.stream, &self.launch, self.next_base, rows, Some(page))
            {
                return Err(refuse(error));
            }
            let partials = match self.run.read_partials(self.launch.rows()) {
                Ok(partials) => partials,
                Err(error) => {
                    self.run.quarantined = true;
                    return Err(refuse(self.run.attribute(error)));
                }
            };
            self.run.copied_pending[page] = false;
            self.page_rows[page] = 0;
            self.folded_total = next_folded;
            self.outstanding_rows = next_outstanding_rows;
            self.next_base = next_base;
            self.remaining_rows = next_remaining_rows;
            self.remaining_blocks = next_remaining_blocks;
            self.host_to_device_bytes = total_transfer;
            Ok(partials)
        }
    }

    impl Drop for NBlockStream<'_, '_> {
        fn drop(&mut self) {
            self.run.prefetched_unused = self
                .run
                .prefetched_unused
                .saturating_add(self.prefetched_total.saturating_sub(self.folded_total));
            if matches!(self.run.staging, Staging::HostBackedReadAhead { .. }) {
                self.run.read_ahead_stream_active = false;
            }
        }
    }

    impl Drop for PagedAttentionRun<'_> {
        fn drop(&mut self) {
            let read_ahead = matches!(self.staging, Staging::HostBackedReadAhead { .. });
            let mut failed_context_drain = false;
            if read_ahead {
                if self.quarantined || self.observe_pending().is_err() {
                    if self.ctx.synchronize().is_err() {
                        self.quarantined = true;
                        failed_context_drain = true;
                    } else {
                        self.pending = None;
                        self.copied_pending = [false, false];
                        self.held = None;
                        self.quarantined = false;
                    }
                }
            } else if self
                .pending
                .as_ref()
                .is_some_and(|pending| pending.synchronize().is_err())
            {
                self.quarantined = true;
            }
            if failed_context_drain {
                if let Some(held) = self.held.take() {
                    match held {
                        RefusedSource::Rows { keys, values } => {
                            core::mem::forget(keys);
                            core::mem::forget(values);
                        }
                        RefusedSource::PageTable(table) => core::mem::forget(table),
                        RefusedSource::Query(bytes) | RefusedSource::PageTableUpload(bytes) => {
                            core::mem::forget(bytes)
                        }
                        RefusedSource::Stream {
                            query,
                            keys,
                            values,
                            table: _,
                        } => {
                            core::mem::forget(query);
                            core::mem::forget(keys);
                            core::mem::forget(values);
                        }
                    }
                }
                if let Some(pinned) = self.pinned_bounce.take() {
                    core::mem::forget(pinned);
                }
                if let Some(stream) = self.copy_stream.take() {
                    core::mem::forget(stream);
                }
                if let Some(pending) = self.pending.take() {
                    core::mem::forget(pending);
                }
                let copied = core::mem::take(&mut self.copied);
                core::mem::forget(copied);
                if let Some(ranges) = self.held_ranges.take() {
                    core::mem::forget(ranges);
                }
                for range in [
                    &mut self.keys,
                    &mut self.values,
                    &mut self.table,
                    &mut self.query,
                    &mut self.output,
                    &mut self.partial_max,
                    &mut self.partial_sum,
                    &mut self.partial_weighted,
                ] {
                    if let Some(range) = range.take() {
                        core::mem::forget(range);
                    }
                }
                for page in 0..2 {
                    for range in [
                        &mut self.staged_keys[page],
                        &mut self.staged_values[page],
                        &mut self.staged_table[page],
                    ] {
                        if let Some(range) = range.take() {
                            core::mem::forget(range);
                        }
                    }
                }
                if let Some(arena) = self.arena.take() {
                    core::mem::forget(arena);
                }
                return;
            }
            if self.quarantined {
                // Host bytes a still-enqueued copy may be reading. Forgetting
                // -- not dropping -- prevents reuse while the device may read.
                if let Some(held) = self.held.take() {
                    match held {
                        RefusedSource::Rows { keys, values } => {
                            core::mem::forget(keys);
                            core::mem::forget(values);
                        }
                        RefusedSource::PageTable(table) => core::mem::forget(table),
                        RefusedSource::Query(bytes) | RefusedSource::PageTableUpload(bytes) => {
                            core::mem::forget(bytes)
                        }
                        RefusedSource::Stream {
                            query,
                            keys,
                            values,
                            table: _,
                        } => {
                            core::mem::forget(query);
                            core::mem::forget(keys);
                            core::mem::forget(values);
                        }
                    }
                }
            } else {
                // A pending event was observed above, so unloading the module
                // cannot race work from this run.
                // SAFETY: the field is created once and otherwise never
                // dropped; all normal-path work is settled or observed here.
                unsafe { ManuallyDrop::drop(&mut self.module) };
            }
            if read_ahead
                && let Some(pinned) = self.pinned_bounce.take()
                && let Err((pinned, _)) = pinned.free()
            {
                core::mem::forget(pinned);
            }
            // `held_ranges` needs no such rescue: a `DeviceRange` has no
            // `Drop` of its own, so an ordinary drop here does not release its
            // suballocation back to the arena's free list -- no later
            // allocation can be handed the same bytes. That is a narrower
            // claim than permanent withholding: it says nothing about the
            // arena's own physical buffer, which is a fact about
            // `DeviceArena`'s lifetime, not this run's.
        }
    }

    /// Performs one state authority's page writes for one layer.
    #[derive(Debug)]
    #[cfg(feature = "paged-attention-binding")]
    pub(crate) struct PagedKvWriterAdapter<'run, 'ctx> {
        layer: usize,
        run: &'run mut PagedAttentionRun<'ctx>,
        stream: &'run Stream<'ctx>,
        keys: Vec<u8>,
        values: Vec<u8>,
        retained: bool,
        source: Option<&'run PagedAttentionRun<'ctx>>,
        device_rows: Option<(&'run DeviceRange<'ctx>, &'run DeviceRange<'ctx>)>,
    }

    #[cfg(feature = "paged-attention-binding")]
    impl<'run, 'ctx> PagedKvWriterAdapter<'run, 'ctx> {
        pub(crate) fn new(
            layer: usize,
            run: &'run mut PagedAttentionRun<'ctx>,
            stream: &'run Stream<'ctx>,
            keys: Vec<u8>,
            values: Vec<u8>,
        ) -> Self {
            Self {
                layer,
                run,
                stream,
                keys,
                values,
                retained: false,
                source: None,
                device_rows: None,
            }
        }

        pub(crate) fn from_device(
            layer: usize,
            run: &'run mut PagedAttentionRun<'ctx>,
            stream: &'run Stream<'ctx>,
            keys: &'run DeviceRange<'ctx>,
            values: &'run DeviceRange<'ctx>,
        ) -> Self {
            Self {
                layer,
                run,
                stream,
                keys: Vec::new(),
                values: Vec::new(),
                retained: false,
                source: None,
                device_rows: Some((keys, values)),
            }
        }

        fn for_branch(
            layer: usize,
            source: &'run PagedAttentionRun<'ctx>,
            run: &'run mut PagedAttentionRun<'ctx>,
            stream: &'run Stream<'ctx>,
        ) -> Self {
            Self {
                layer,
                run,
                stream,
                keys: Vec::new(),
                values: Vec::new(),
                retained: false,
                source: Some(source),
                device_rows: None,
            }
        }

        /// Returns rows only when no device operation retained them.
        fn into_rows(self) -> Option<(Vec<u8>, Vec<u8>)> {
            (!self.retained).then_some((self.keys, self.values))
        }

        fn publish(&mut self, layer: usize, view: PageView) -> moxie_types::Result<()> {
            if layer != self.layer {
                return Err(invalid_fmt(
                    "layer",
                    format_args!("this writer serves layer {}, not {layer}", self.layer),
                ));
            }
            let result = if self.device_rows.is_some() {
                self.run
                    .publish_page_table_deferred(self.stream, view.base, view.table)
            } else {
                self.run
                    .publish_page_table(self.stream, view.base, view.table)
            };
            result.map_err(|refused| refused.error)
        }
    }

    #[cfg(feature = "paged-attention-binding")]
    impl PagedKvWriter for PagedKvWriterAdapter<'_, '_> {
        fn write_layer(
            &mut self,
            layer: usize,
            _batch: BatchId,
            view: PageView,
            placements: &[PagePlacement],
        ) -> moxie_types::Result<()> {
            self.publish(layer, view)?;
            if let Some((keys, values)) = self.device_rows {
                return self
                    .run
                    .write_rows_from_device(self.stream, placements, keys, values);
            }
            let keys = core::mem::take(&mut self.keys);
            let values = core::mem::take(&mut self.values);
            match self.run.write_rows(self.stream, placements, keys, values) {
                Ok(()) => Ok(()),
                Err(refused) => {
                    if let Some(RefusedSource::Rows { keys, values }) = refused.source {
                        self.keys = keys;
                        self.values = values;
                    } else {
                        self.retained = true;
                    }
                    Err(refused.error)
                }
            }
        }

        fn publish_view(&mut self, layer: usize, view: PageView) -> moxie_types::Result<()> {
            self.publish(layer, view)
        }

        fn copy_branch(
            &mut self,
            layer: usize,
            view: PageView,
            rows: u64,
        ) -> moxie_types::Result<()> {
            if layer != self.layer {
                return Err(invalid_fmt(
                    "layer",
                    format_args!("this writer serves layer {}, not {layer}", self.layer),
                ));
            }
            let source = self.source.ok_or_else(|| {
                invalid(
                    "branch",
                    "this writer was not constructed for a device branch copy",
                )
            })?;
            self.run.copy_branch_from(self.stream, source, view, rows)
        }
    }

    /// Refused authority-driven append. `source` is present only when no
    /// in-flight device work retained the rows.
    #[derive(Debug)]
    #[cfg(feature = "paged-attention-binding")]
    pub struct PagedKvRows {
        pub keys: Vec<u8>,
        pub values: Vec<u8>,
    }

    #[derive(Debug)]
    #[cfg(feature = "paged-attention-binding")]
    pub struct PagedStateAppendRefused {
        pub error: Error,
        pub source: Option<PagedKvRows>,
    }

    /// Append one layer through the state authority.
    #[cfg(feature = "paged-attention-binding")]
    #[allow(clippy::result_large_err)]
    pub fn append_paged_layer<'ctx>(
        state: &mut DeviceKvSequence,
        txn: moxie_types::StateTransactionId,
        layer: usize,
        count: u64,
        run: &mut PagedAttentionRun<'ctx>,
        stream: &Stream<'ctx>,
        source: PagedKvRows,
    ) -> std::result::Result<(), PagedStateAppendRefused> {
        if let Err(error) = run.check_append_rows(count) {
            return Err(PagedStateAppendRefused {
                error,
                source: Some(source),
            });
        }
        let mut writer = PagedKvWriterAdapter::new(layer, run, stream, source.keys, source.values);
        match state.append_layer(txn, layer, count, &mut writer) {
            Ok(()) => Ok(()),
            Err(error) => Err(PagedStateAppendRefused {
                error,
                source: writer
                    .into_rows()
                    .map(|(keys, values)| PagedKvRows { keys, values }),
            }),
        }
    }

    /// Append device-resident K/V rows through the state authority.
    #[cfg(feature = "paged-attention-binding")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn append_paged_layer_from_device<'ctx>(
        state: &mut DeviceKvSequence,
        txn: moxie_types::StateTransactionId,
        layer: usize,
        count: u64,
        run: &mut PagedAttentionRun<'ctx>,
        stream: &Stream<'ctx>,
        keys: &DeviceRange<'ctx>,
        values: &DeviceRange<'ctx>,
    ) -> Result<()> {
        run.check_append_rows(count)?;
        state.append_layer(
            txn,
            layer,
            count,
            &mut PagedKvWriterAdapter::from_device(layer, run, stream, keys, values),
        )
    }

    /// Fork one layer through the existing state-to-executor writer callback.
    /// The child run is allocated by the caller and is closed by the caller if
    /// the post-logical-fork copy refuses.
    #[cfg(feature = "paged-attention-binding")]
    pub fn fork_paged_layer<'ctx>(
        state: &mut DeviceKvSequence,
        at: u64,
        parent: &PagedAttentionRun<'ctx>,
        child: &mut PagedAttentionRun<'ctx>,
        stream: &Stream<'ctx>,
    ) -> Result<BranchId> {
        if state.layer_count()? != 1 {
            return Err(invalid(
                "layers",
                "single-layer device fork requires a one-layer state authority",
            ));
        }
        let lineage_entries = at
            .checked_add(1)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
        if lineage_entries > child.fork_lineage_capacity {
            return Err(invalid_fmt(
                "fork_lineage_capacity",
                format_args!(
                    "fork at {at} needs {lineage_entries} lineage entries; the child run admitted {}",
                    child.fork_lineage_capacity
                ),
            ));
        }
        let mut writer = PagedKvWriterAdapter::for_branch(0, parent, child, stream);
        state.fork(at, &mut [&mut writer])
    }

    /// Append one layer on a child branch through the same writer callback as
    /// the root path.
    #[cfg(feature = "paged-attention-binding")]
    #[allow(clippy::result_large_err)]
    pub fn append_paged_branch<'ctx>(
        branch: &mut DeviceBranch<'_>,
        txn: moxie_types::StateTransactionId,
        layer: usize,
        count: u64,
        run: &mut PagedAttentionRun<'ctx>,
        stream: &Stream<'ctx>,
        source: PagedKvRows,
    ) -> std::result::Result<(), PagedStateAppendRefused> {
        if let Err(error) = run.check_append_rows(count) {
            return Err(PagedStateAppendRefused {
                error,
                source: Some(source),
            });
        }
        let mut writer = PagedKvWriterAdapter::new(layer, run, stream, source.keys, source.values);
        match branch.append_layer(txn, layer, count, &mut writer) {
            Ok(()) => Ok(()),
            Err(error) => Err(PagedStateAppendRefused {
                error,
                source: writer
                    .into_rows()
                    .map(|(keys, values)| PagedKvRows { keys, values }),
            }),
        }
    }

    /// Commit one child branch and publish any retained-page transition.
    #[cfg(feature = "paged-attention-binding")]
    pub fn commit_paged_branch<'ctx>(
        branch: &mut DeviceBranch<'_>,
        txn: moxie_types::StateTransactionId,
        accept: u64,
        run: &mut PagedAttentionRun<'ctx>,
        stream: &Stream<'ctx>,
    ) -> Result<()> {
        let mut writer = PagedKvWriterAdapter::new(0, run, stream, Vec::new(), Vec::new());
        branch.commit(txn, accept, &mut [&mut writer])
    }

    #[cfg(feature = "paged-attention-binding")]
    pub(crate) fn with_paged_writers<'ctx, R>(
        runs: &mut [PagedAttentionRun<'ctx>],
        stream: &Stream<'ctx>,
        f: impl FnOnce(&mut [&mut dyn PagedKvWriter]) -> Result<R>,
    ) -> Result<R> {
        let mut adapters = moxie_memory::fallible::with_capacity(runs.len())?;
        for (layer, run) in runs.iter_mut().enumerate() {
            adapters.push(PagedKvWriterAdapter::new(
                layer,
                run,
                stream,
                Vec::new(),
                Vec::new(),
            ));
        }
        let mut writers = moxie_memory::fallible::with_capacity(adapters.len())?;
        writers.extend(
            adapters
                .iter_mut()
                .map(|writer| writer as &mut dyn PagedKvWriter),
        );
        f(&mut writers)
    }

    /// Commit a completed batch and publish page-table transitions through the
    /// matching run for every layer.
    #[cfg(feature = "paged-attention-binding")]
    pub fn commit_paged_state<'ctx>(
        state: &mut DeviceKvSequence,
        txn: moxie_types::StateTransactionId,
        accept: u64,
        runs: &mut [PagedAttentionRun<'ctx>],
        stream: &Stream<'ctx>,
    ) -> Result<()> {
        if state.layer_count()? != runs.len() {
            return Err(invalid(
                "runs",
                "commit needs exactly one device run per state layer",
            ));
        }
        with_paged_writers(runs, stream, |writers| state.commit(txn, accept, writers))
    }

    /// Single-layer form used by a standalone attention plan.
    #[cfg(feature = "paged-attention-binding")]
    pub fn commit_paged_layer<'ctx>(
        state: &mut DeviceKvSequence,
        txn: moxie_types::StateTransactionId,
        accept: u64,
        run: &mut PagedAttentionRun<'ctx>,
        stream: &Stream<'ctx>,
    ) -> Result<()> {
        if state.layer_count()? != 1 {
            return Err(invalid(
                "runs",
                "single-layer commit requires a one-layer state authority",
            ));
        }
        let mut writer = PagedKvWriterAdapter::new(0, run, stream, Vec::new(), Vec::new());
        state.commit(txn, accept, &mut [&mut writer])
    }

    /// Test-only access to raw page mutation for kernel qualification and
    /// injected-fault cases.
    #[derive(Debug)]
    #[must_use = "an unclosed run keeps its arena and its reservation"]
    #[cfg(feature = "paged-attention-test-hooks")]
    pub struct RawPagedFixture<'ctx> {
        run: PagedAttentionRun<'ctx>,
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    impl<'ctx> RawPagedFixture<'ctx> {
        pub fn new(run: PagedAttentionRun<'ctx>) -> Self {
            Self { run }
        }

        #[allow(clippy::result_large_err)]
        pub fn publish_page_table(
            &mut self,
            stream: &Stream<'ctx>,
            base: u64,
            table: Vec<u32>,
        ) -> std::result::Result<(), PagedRunRefused> {
            self.run.publish_page_table(stream, base, table)
        }

        #[allow(clippy::result_large_err)]
        pub fn write_rows(
            &mut self,
            stream: &Stream<'ctx>,
            placements: &[PagePlacement],
            keys: Vec<u8>,
            values: Vec<u8>,
        ) -> std::result::Result<(), PagedRunRefused> {
            self.run.write_rows(stream, placements, keys, values)
        }

        pub fn run(&self) -> &PagedAttentionRun<'ctx> {
            &self.run
        }

        pub fn into_inner(self) -> PagedAttentionRun<'ctx> {
            self.run
        }
    }

    /// Whether this device will launch this launch's grid.
    ///
    /// A launch is a host value and cannot know a device's limits; a run is
    /// bound to one device and can. One block per query row and head means the
    /// head count lands on grid `y`, whose limit is 65,535 and not `u32::MAX` —
    /// the distinction that would otherwise be discovered by `cuLaunchKernel`
    /// after the query had already been copied.
    fn check_grid(launch: &PagedAttentionLaunch, ctx: &RankContext) -> Result<()> {
        let (x, y, z) = launch.grid()?;
        let (max_x, max_y, max_z) = ctx.capability().max_grid;
        if x > max_x || y > max_y || z > max_z {
            return Err(unsupported_kernel_fmt(
                "paged_attention",
                format_args!(
                    "a grid of {x}x{y}x{z} blocks exceeds this device's {max_x}x{max_y}x{max_z}: \
                     {} query row(s) of {} head(s) cannot be launched here",
                    launch.rows(),
                    launch.heads()
                ),
            ));
        }
        Ok(())
    }

    /// Everything the kernel's ABI takes that is not an address.
    ///
    /// Derived from a checked launch before any device work starts, so a
    /// launch that cannot be expressed in this ABI is refused with nothing
    /// enqueued and nothing withheld.
    struct AbiScalars {
        rows: u64,
        first_position: u64,
        history_base: u64,
        history_rows: u64,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page_tokens: u32,
        window: u32,
        scale: f32,
        grid: (u32, u32, u32),
    }

    /// Host bytes used by one layer's commit vectors. The state `updates`
    /// vector, executor adapter vector and writer-reference vector all reserve
    /// one entry per admitted layer.
    pub(super) fn commit_host_metadata_bytes() -> u64 {
        let state_updates = core::mem::size_of::<(usize, PageView)>() as u64;
        #[cfg(feature = "paged-attention-binding")]
        let wrappers = core::mem::size_of::<PagedKvWriterAdapter<'static, 'static>>() as u64
            + core::mem::size_of::<&mut dyn PagedKvWriter>() as u64;
        #[cfg(not(feature = "paged-attention-binding"))]
        let wrappers = 0;
        state_updates + wrappers
    }

    /// The aligned extents one run admits.
    struct Extents {
        payload: u64,
        table: u64,
        page_table_upload: u64,
        page_table_host: u64,
        page_view_host: u64,
        placements_host: u64,
        commit_host: u64,
        root_lineage_host: u64,
        fork_host: u64,
        partials_host: u64,
        pinned_bounce: u64,
        query: u64,
        persistent: u64,
        per_step: u64,
        transfer: u64,
        staged_payload: u64,
        staged_table: u64,
        partial_scalars: u64,
        partial_weighted: u64,
        total: u64,
        pages_usize: usize,
    }

    impl Extents {
        fn derive(
            geometry: &PageGeometry,
            heads: u64,
            max_rows: u64,
            root_lineage_capacity: u64,
            fork_lineage_capacity: u64,
            staging: Staging,
        ) -> Result<Self> {
            geometry.check()?;
            let payload = align_up(geometry.payload_bytes()?)?;
            let table = align_up(
                geometry
                    .pages
                    .checked_mul(4)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
            )?;
            let page_table_upload = geometry.page_table_upload_bytes()?;
            let page_table_host = page_table_upload;
            let page_view_host = page_table_upload;
            let placement_capacity = max_rows
                .div_ceil(geometry.page_tokens)
                .checked_add(1)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let placements_host = placement_capacity
                .checked_mul(core::mem::size_of::<PagePlacement>() as u64)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let commit_host = commit_host_metadata_bytes();
            #[cfg(feature = "paged-attention-binding")]
            let root_lineage_host = if root_lineage_capacity == 0 {
                0
            } else {
                moxie_state::DeviceKvSequence::root_host_metadata_bytes(root_lineage_capacity)?
            };
            #[cfg(not(feature = "paged-attention-binding"))]
            let root_lineage_host = if root_lineage_capacity == 0 {
                0
            } else {
                return Err(Error::Unsupported {
                    capability: "paged_attention_sequence_metadata",
                    reason: "sequence metadata admission requires the paged-attention binding"
                        .into(),
                });
            };
            #[cfg(feature = "paged-attention-binding")]
            let fork_host = if fork_lineage_capacity == 0 {
                0
            } else {
                moxie_state::DeviceKvSequence::fork_host_metadata_bytes(fork_lineage_capacity, 1)?
            };
            #[cfg(not(feature = "paged-attention-binding"))]
            let fork_host = if fork_lineage_capacity == 0 {
                0
            } else {
                return Err(Error::Unsupported {
                    capability: "paged_attention_fork",
                    reason: "fork metadata admission requires the paged-attention binding".into(),
                });
            };
            let partials_host = if staging.partial_buffers() {
                let partial_count = max_rows
                    .checked_mul(heads)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                let scalar_bytes = partial_count
                    .checked_mul(core::mem::size_of::<f32>() as u64)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                let weighted_bytes = partial_count
                    .checked_mul(geometry.head_dim)
                    .and_then(|bytes| bytes.checked_mul(core::mem::size_of::<f32>() as u64))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                let partial_vector_bytes = partial_count
                    .checked_mul(core::mem::size_of::<DevicePartial>() as u64)
                    .and_then(|bytes| bytes.checked_add(weighted_bytes))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                partial_vector_bytes
                    .checked_mul(
                        staging
                            .max_staged_blocks()
                            .checked_add(1)
                            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
                    )
                    .and_then(|bytes| bytes.checked_add(scalar_bytes.checked_mul(2)?))
                    .and_then(|bytes| bytes.checked_add(weighted_bytes))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?
            } else {
                0
            };
            let pinned_bounce = if matches!(staging, Staging::HostBackedReadAhead { .. }) {
                geometry
                    .page_bytes()?
                    .checked_mul(2)
                    .and_then(|bytes| bytes.checked_add(PAGE_ENTRY_BYTES))
                    .and_then(|bytes| bytes.checked_mul(2))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?
            } else {
                0
            };
            let query = align_up(
                max_rows
                    .checked_mul(heads)
                    .and_then(|v| v.checked_mul(geometry.head_dim))
                    .and_then(|v| v.checked_mul(2))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
            )?;
            let persistent = payload
                .checked_mul(2)
                .and_then(|v| v.checked_add(table))
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let (
                transfer,
                per_step,
                staged_payload,
                staged_table,
                partial_scalars,
                partial_weighted,
            ) = if staging.partial_buffers() {
                let staged_payload = align_up(geometry.page_bytes()?)?;
                let staged_table = align_up(PAGE_ENTRY_BYTES)?;
                let partial_scalars = align_up(
                    max_rows
                        .checked_mul(heads)
                        .and_then(|v| v.checked_mul(4))
                        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
                )?;
                let partial_weighted = align_up(
                    max_rows
                        .checked_mul(heads)
                        .and_then(|v| v.checked_mul(geometry.head_dim))
                        .and_then(|v| v.checked_mul(4))
                        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
                )?;
                let single_page = staged_payload
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(staged_table))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                let staging_pages = if matches!(staging, Staging::HostBackedReadAhead { .. }) {
                    2
                } else {
                    1
                };
                let transfer = single_page
                    .checked_mul(staging_pages)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                let per_step = query
                    .checked_add(
                        partial_scalars
                            .checked_mul(2)
                            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
                    )
                    .and_then(|v| v.checked_add(partial_weighted))
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                (
                    transfer,
                    per_step,
                    staged_payload,
                    staged_table,
                    partial_scalars,
                    partial_weighted,
                )
            } else if staging == Staging::Host {
                // Zero for a direct-device run: the caller's ranges are the
                // query and the output, and charging for a second pair here is
                // what makes an admissible plan refusable.
                (
                    0,
                    query
                        .checked_mul(2)
                        .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
                    0,
                    0,
                    0,
                    0,
                )
            } else {
                (0, 0, 0, 0, 0, 0)
            };
            let total = persistent
                .checked_add(per_step)
                .and_then(|v| v.checked_add(transfer))
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let pages_usize = usize::try_from(geometry.pages)
                .map_err(|_| invalid("pages", "the page count exceeds this host's usize"))?;
            Ok(Self {
                payload,
                table,
                page_table_upload,
                page_table_host,
                page_view_host,
                placements_host,
                commit_host,
                root_lineage_host,
                fork_host,
                partials_host,
                pinned_bounce,
                query,
                persistent,
                per_step,
                transfer,
                staged_payload,
                staged_table,
                partial_scalars,
                partial_weighted,
                total,
                pages_usize,
            })
        }
    }

    /// The admission request one paged attention run makes.
    ///
    /// Three device buffers always: keys, values and the page table, each
    /// named for what it is and charged as `KvStatePages`. Host metadata is
    /// charged by phase: the reused upload and run mapping persist, append
    /// admits one state view and placement list, and commit admits its state
    /// views plus bounded adapter/update vectors. A fork destination also
    /// admits its explicit logical-lineage capacity and branch metadata.
    /// `Staging::Host` adds two device buffers (query and output, charged as
    /// `Activations`) and one host readback. Partial-stream modes also admit
    /// their bounded host readback results. `Staging::TwoBlock` adds one bounded page pair in
    /// `TransferStaging`, the query and FP32 partials in `Activations`, and no
    /// unbounded history buffer.
    pub fn resource_request(
        geometry: &PageGeometry,
        heads: u64,
        max_rows: u64,
        staging: Staging,
        ctx: &RankContext,
    ) -> Result<PlanRequest> {
        resource_request_with_lineage_capacities(geometry, heads, max_rows, 0, 0, staging, ctx)
    }

    /// The resource request for a root `DeviceKvSequence` run. The logical
    /// lineage capacity and possible transaction node are independent of the
    /// run's physical page count and append-row bound.
    pub fn resource_request_for_sequence(
        geometry: &PageGeometry,
        heads: u64,
        max_rows: u64,
        root_lineage_capacity: u64,
        staging: Staging,
        ctx: &RankContext,
    ) -> Result<PlanRequest> {
        if root_lineage_capacity == 0 {
            return Err(invalid(
                "root_lineage_capacity",
                "a sequence needs lineage entries",
            ));
        }
        resource_request_with_lineage_capacities(
            geometry,
            heads,
            max_rows,
            root_lineage_capacity,
            0,
            staging,
            ctx,
        )
    }

    /// The resource request for a run admitted to receive a branch fork.
    /// `fork_lineage_capacity` is the reserved number of logical prefix entries,
    /// independent of the run's physical page count. It charges the child
    /// lineage, both state-map nodes, and per-layer device branch vectors.
    pub fn resource_request_for_fork(
        geometry: &PageGeometry,
        heads: u64,
        max_rows: u64,
        fork_lineage_capacity: u64,
        staging: Staging,
        ctx: &RankContext,
    ) -> Result<PlanRequest> {
        if fork_lineage_capacity == 0 {
            return Err(invalid(
                "fork_lineage_capacity",
                "a fork needs lineage entries",
            ));
        }
        resource_request_with_lineage_capacities(
            geometry,
            heads,
            max_rows,
            0,
            fork_lineage_capacity,
            staging,
            ctx,
        )
    }

    fn resource_request_with_lineage_capacities(
        geometry: &PageGeometry,
        heads: u64,
        max_rows: u64,
        root_lineage_capacity: u64,
        fork_lineage_capacity: u64,
        staging: Staging,
        ctx: &RankContext,
    ) -> Result<PlanRequest> {
        let extents = Extents::derive(
            geometry,
            heads,
            max_rows,
            root_lineage_capacity,
            fork_lineage_capacity,
            staging,
        )?;
        let mut request = PlanRequest::new(
            moxie_memory::fallible::text(format_args!(
                "paged-attention-{}x{}",
                geometry.kv_heads, geometry.head_dim
            ))?,
            ["fork", "append", "launch", "read", "commit"],
        )?;
        if let Staging::HostBacked { max_staged_blocks }
        | Staging::HostBackedReadAhead { max_staged_blocks } = staging
        {
            request.host_backed_blocks(max_staged_blocks)?;
        }
        let scope = Scope::Device(ctx.uuid());
        request.buffer(BufferRequest::new(
            "kv-pages-keys",
            scope,
            Tier::Device(DeviceTier::KvStatePages),
            extents.payload,
            StageSpan::inclusive(0, 4),
        ))?;
        request.buffer(BufferRequest::new(
            "kv-pages-values",
            scope,
            Tier::Device(DeviceTier::KvStatePages),
            extents.payload,
            StageSpan::inclusive(0, 4),
        ))?;
        request.buffer(BufferRequest::new(
            "kv-page-table",
            scope,
            Tier::Device(DeviceTier::KvStatePages),
            extents.table,
            StageSpan::inclusive(0, 4),
        ))?;
        request.buffer(BufferRequest::new(
            "page-table-upload-workspace",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.page_table_upload,
            StageSpan::inclusive(0, 4),
        ))?;
        request.buffer(BufferRequest::new(
            "paged-run-page-table-host",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.page_table_host,
            StageSpan::inclusive(0, 4),
        ))?;
        if extents.root_lineage_host != 0 {
            request.buffer(BufferRequest::new(
                "device-kv-root-host-metadata",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                extents.root_lineage_host,
                StageSpan::inclusive(0, 4),
            ))?;
        }
        if extents.fork_host != 0 {
            request.buffer(BufferRequest::new(
                "device-kv-fork-host-metadata",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                extents.fork_host,
                StageSpan::inclusive(0, 4),
            ))?;
        }
        if extents.pinned_bounce != 0 {
            request.buffer(BufferRequest::new(
                "attention-stream-pinned-bounce-pages",
                Scope::Host,
                Tier::Host(HostTier::Pinned),
                extents.pinned_bounce,
                StageSpan::at(2),
            ))?;
        }
        request.buffer(BufferRequest::new(
            "device-kv-page-view-host",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.page_view_host,
            StageSpan::at(1),
        ))?;
        request.buffer(BufferRequest::new(
            "device-kv-commit-page-view-host",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.page_view_host,
            StageSpan::at(4),
        ))?;
        request.buffer(BufferRequest::new(
            "device-kv-placements-host",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.placements_host,
            StageSpan::at(1),
        ))?;
        request.buffer(BufferRequest::new(
            "device-kv-commit-host-metadata",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.commit_host,
            StageSpan::at(4),
        ))?;
        // Host staging only: a direct-device run's query and output are the
        // caller's own arena slots, and charging this ledger for a device
        // activation pair and a host readback it will never hold is what lets
        // an admissible direct-device plan be refused for bytes it does not
        // use.
        if staging == Staging::Host {
            request.buffer(BufferRequest::new(
                "attention-query",
                scope,
                Tier::Device(DeviceTier::Activations),
                extents.query,
                StageSpan::inclusive(2, 3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-output",
                scope,
                Tier::Device(DeviceTier::Activations),
                extents.query,
                StageSpan::inclusive(2, 3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-output-readback",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                extents.query,
                StageSpan::at(3),
            ))?;
        } else if staging.partial_buffers() {
            request.buffer(BufferRequest::new(
                "attention-stream-partial-readback-host",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                extents.partials_host,
                StageSpan::at(3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-query",
                scope,
                Tier::Device(DeviceTier::Activations),
                extents.query,
                StageSpan::inclusive(2, 3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-partial-max",
                scope,
                Tier::Device(DeviceTier::Activations),
                extents.partial_scalars,
                // The run owns one contiguous physical arena. Keep the
                // reusable partial slots live across staging as well as
                // readback so ledger admission covers that real allocation,
                // not only the non-overlapping kernel lifetimes.
                StageSpan::inclusive(2, 3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-partial-sum",
                scope,
                Tier::Device(DeviceTier::Activations),
                extents.partial_scalars,
                StageSpan::inclusive(2, 3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-partial-weighted",
                scope,
                Tier::Device(DeviceTier::Activations),
                extents.partial_weighted,
                StageSpan::inclusive(2, 3),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-staged-keys",
                scope,
                Tier::Device(DeviceTier::TransferStaging),
                extents.staged_payload,
                StageSpan::at(2),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-staged-values",
                scope,
                Tier::Device(DeviceTier::TransferStaging),
                extents.staged_payload,
                StageSpan::at(2),
            ))?;
            request.buffer(BufferRequest::new(
                "attention-stream-staged-table",
                scope,
                Tier::Device(DeviceTier::TransferStaging),
                extents.staged_table,
                StageSpan::at(2),
            ))?;
            if matches!(staging, Staging::HostBackedReadAhead { .. }) {
                request.buffer(BufferRequest::new(
                    "attention-stream-staged-keys-next",
                    scope,
                    Tier::Device(DeviceTier::TransferStaging),
                    extents.staged_payload,
                    StageSpan::at(2),
                ))?;
                request.buffer(BufferRequest::new(
                    "attention-stream-staged-values-next",
                    scope,
                    Tier::Device(DeviceTier::TransferStaging),
                    extents.staged_payload,
                    StageSpan::at(2),
                ))?;
                request.buffer(BufferRequest::new(
                    "attention-stream-staged-table-next",
                    scope,
                    Tier::Device(DeviceTier::TransferStaging),
                    extents.staged_table,
                    StageSpan::at(2),
                ))?;
            }
        }
        Ok(request)
    }

    fn align_up(bytes: u64) -> Result<u64> {
        bytes
            .checked_add(ALIGNMENT - 1)
            .map(|v| v / ALIGNMENT * ALIGNMENT)
            .ok_or_else(|| invalid("align", "the aligned extent overflows"))
    }

    /// Whether `[a, a + a_len)` and `[b, b + b_len)` share a byte.
    fn ranges_overlap(a: u64, a_len: u64, b: u64, b_len: u64) -> bool {
        a < b.saturating_add(b_len) && b < a.saturating_add(a_len)
    }

    fn give_back<'ctx>(
        ledger: &mut Ledger,
        reservation: Reservation,
        error: Error,
    ) -> PagedAdmitRefused<'ctx> {
        match ledger.release(reservation) {
            Ok(()) => PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
                arena: None,
                ranges: Vec::new(),
                cleanup: None,
            },
            Err(refused) => PagedAdmitRefused {
                error,
                reservation: Some(refused.reservation),
                rejection: None,
                arena: None,
                ranges: Vec::new(),
                cleanup: Some(refused.error),
            },
        }
    }

    /// Release everything admission had built when a later step refused.
    ///
    /// Stops at the first failed `release` or `close` rather than trying the
    /// rest: `error` stays admission's own reason for refusing throughout,
    /// and whatever cleanup could not give back -- the arena, and every range
    /// `hold` still names, including the one the failed `release` handed back
    /// -- travels with the refusal in `arena`/`ranges` instead of being
    /// dropped. `ranges` is `hold` itself, reused rather than reallocated on
    /// this refusal path.
    fn unwind<'ctx>(
        mut arena: DeviceArena<'ctx>,
        mut ranges: Vec<DeviceRange<'ctx>>,
        ledger: &mut Ledger,
        error: Error,
    ) -> PagedAdmitRefused<'ctx> {
        while let Some(range) = ranges.pop() {
            if let Err(refused) = arena.release(range) {
                ranges.push(refused.range);
                return PagedAdmitRefused {
                    error,
                    reservation: None,
                    rejection: None,
                    arena: Some(arena),
                    ranges,
                    cleanup: Some(refused.error),
                };
            }
        }
        match arena.close(ledger) {
            Ok(()) => PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
                arena: None,
                ranges,
                cleanup: None,
            },
            Err(refused) => PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
                arena: Some(refused.arena),
                ranges,
                cleanup: Some(refused.error),
            },
        }
    }

    #[cfg(test)]
    mod overlap_tests {
        use super::ranges_overlap;

        /// No hardware needed: this is the arithmetic `attend_into` refuses an
        /// aliased query/output on, checked directly rather than only through
        /// two live `DeviceRange`s -- which, once `attend_into` takes them by
        /// value, cannot be made to alias through the safe API at all.
        #[test]
        fn identical_and_nested_ranges_overlap_disjoint_ones_do_not() {
            assert!(ranges_overlap(0, 16, 0, 16), "identical ranges");
            assert!(ranges_overlap(0, 16, 8, 16), "overlapping tails");
            assert!(ranges_overlap(8, 16, 0, 16), "overlapping tails, swapped");
            assert!(ranges_overlap(4, 4, 0, 16), "nested inside a wider range");
            assert!(!ranges_overlap(0, 16, 16, 16), "adjacent, not overlapping");
            assert!(!ranges_overlap(0, 8, 100, 8), "far apart");
            assert!(
                !ranges_overlap(u64::MAX - 4, 8, 0, 4),
                "a saturating length must not wrap into a false overlap"
            );
        }
    }
}

#[cfg(all(test, feature = "driver"))]
mod device_tests {
    //! The device-handle entry point, which has no external caller yet.
    //!
    //! `attend_into` takes `DeviceRange`s and reads their addresses, and both
    //! of those are crate-internal — deliberately, because a range's address is
    //! only meaningful inside the crate that owns its arena. So the test that
    //! proves the two entry points compute the same answer lives here, beside
    //! them, and it needs real hardware.

    use moxie_cuda::{RankContext, Stream};
    use moxie_memory::{CapacitySnapshot, Ledger};
    use moxie_plan::Visibility;
    #[cfg(feature = "paged-attention-binding")]
    use moxie_types::Error;
    use moxie_types::{PagePlacement, RankId, Scope};

    use super::device::{PagedAttentionRun, Staging};
    #[cfg(feature = "paged-attention-binding")]
    use super::device::{PagedKvRows, append_paged_layer};
    use super::{AttentionLayer, PageGeometry, PagedAttentionLaunch};
    use crate::arena::DeviceArena;

    const HEADS: u64 = 4;
    const HEAD_DIM: u64 = 64;

    fn geometry() -> PageGeometry {
        PageGeometry {
            kv_heads: 2,
            head_dim: HEAD_DIM,
            page_tokens: 8,
            pages: 2,
        }
    }

    fn bytes(count: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(count * 2);
        for _ in 0..count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = ((state >> 41) as f32) / ((1u32 << 22) as f32) - 1.0;
            out.extend_from_slice(&moxie_kernels::cpu_expert::to_bf16_bits(value).to_le_bytes());
        }
        out
    }

    /// A direct-device run is charged for its complete host metadata peak, but
    /// not for query/output staging it never uses.
    ///
    /// Proved through the ledger's own admission, not the physical arena's
    /// byte counter: a ledger with exactly the direct path's host-metadata
    /// capacity must admit `Staging::DeviceHandles`, and the ledger's charged
    /// host bytes must cover the page table, page view, and placements live at
    /// append peak. It must refuse `Staging::Host` because that request adds
    /// query/output staging. A byte-count comparison of two admitted arenas
    /// cannot tell "the ledger was never asked" from "it was asked and
    /// happened to fit."
    #[test]
    fn a_direct_device_run_is_not_charged_for_staging_it_never_uses() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if moxie_cuda::device_count().expect("enumerate") == 0 {
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let ctx = RankContext::acquire(RankId(38_002), 0).expect("a rank context");
        let stream = Stream::new(&ctx).expect("a stream");
        let capability = moxie_cuda::query_device(0).expect("query device 0");
        let layer = AttentionLayer {
            geometry: geometry(),
            heads: HEADS,
            scale: moxie_plan::reciprocal_sqrt_scale(HEAD_DIM),
            visibility: Visibility::Causal,
        };
        let descriptor = || {
            super::select_paged_attention_kernel(
                &moxie_kernels::paged_attention_catalogue(),
                &capability,
                &PagedAttentionLaunch::new(layer, 1, 0, 0, 1).expect("a launch"),
            )
            .expect("a descriptor")
        };

        let measurement = ctx.measure().expect("measure");
        let geometry = geometry();
        let max_rows = 1u64;
        let page_table_upload = geometry
            .page_table_upload_bytes()
            .expect("page-table upload extent");
        let table_bytes = page_table_upload
            .checked_mul(2)
            .expect("run and state page-table extents");
        let placements = max_rows
            .div_ceil(geometry.page_tokens)
            .checked_add(1)
            .and_then(|count| count.checked_mul(core::mem::size_of::<PagePlacement>() as u64))
            .expect("placement extent");
        let append_metadata = table_bytes
            .checked_add(placements)
            .expect("append metadata extent");
        let commit_metadata = table_bytes
            .checked_add(super::device::commit_host_metadata_bytes())
            .expect("commit metadata extent");
        let metadata_bytes = append_metadata.max(commit_metadata);
        let host_charge = page_table_upload
            .checked_add(metadata_bytes)
            .expect("host admission extent");
        let host_capacity = host_charge.checked_add(1).expect("host capacity");
        let mut ledger = Ledger::new([
            CapacitySnapshot::measured(&measurement, 1 << 20).expect("device capacity"),
            CapacitySnapshot::new(Scope::Host, host_capacity, 1)
                .expect("exact paged host metadata capacity"),
        ])
        .expect("a ledger with device and exact paged host metadata capacity");

        let mut direct = PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor(),
            geometry,
            HEADS,
            max_rows,
            Staging::DeviceHandles,
        )
        .map_err(|r| r.error)
        .expect("a direct-device run needs its host metadata capacity to admit");
        let charged_host = ledger.scope_committed(Scope::Host);
        assert_eq!(charged_host, host_charge);
        assert!(charged_host >= metadata_bytes);
        #[cfg(feature = "paged-attention-binding")]
        {
            let mut state = moxie_state::DeviceKvSequence::new(moxie_state::KvGeometry {
                layers: vec![moxie_state::LayerKv {
                    kv_heads: geometry.kv_heads as usize,
                    key_dim: geometry.head_dim as usize,
                    value_dim: geometry.head_dim as usize,
                    retention: moxie_state::Retention::All,
                }],
                precision: moxie_types::Precision::Bf16,
                page_tokens: geometry.page_tokens as usize,
                max_tokens: 64,
                tentative_rows: 64,
            })
            .expect("a state whose own capacity can hold the oversized append");
            let txn = state.begin().expect("a transaction");
            let refused = append_paged_layer(
                &mut state,
                txn,
                0,
                17,
                &mut direct,
                &stream,
                PagedKvRows {
                    keys: Vec::new(),
                    values: Vec::new(),
                },
            )
            .expect_err("an append past the run's admitted maximum was accepted");
            assert!(matches!(
                refused.error,
                Error::InvalidRequest { field: "rows", .. }
            ));
            state.abort(txn).expect("abort the untouched transaction");
        }
        // `direct` itself is outstanding from here on -- the assertion below
        // is about whether the *failed* admission changes that count, not
        // about the ledger being empty, which it never is while `direct`
        // lives.
        let outstanding_before = ledger.outstanding_count();

        let staged_error = PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor(),
            geometry,
            HEADS,
            max_rows,
            Staging::Host,
        )
        .expect_err("a host-staged run exceeded the metadata-only host capacity")
        .error;
        assert!(
            matches!(staged_error, moxie_types::Error::CapacityExceeded { .. }),
            "{staged_error:?}"
        );
        assert_eq!(
            ledger.outstanding_count(),
            outstanding_before,
            "the failed admission changed what the ledger holds outstanding"
        );

        // And the host-staged entry point is refused on the direct run rather
        // than silently reaching for ranges that are not there.
        direct
            .publish_page_table(&stream, 0, vec![0, 1])
            .map_err(|r| r.error)
            .expect("a mapping");
        let refused = direct
            .attend(
                &stream,
                &PagedAttentionLaunch::new(layer, 1, 0, 0, 1).expect("a launch"),
                bytes((HEADS * HEAD_DIM) as usize, 1),
            )
            .expect_err("a direct-device run staged a host query");
        assert!(!refused.retained_source());
        direct
            .close(&mut ledger)
            .map_err(|r| r.error)
            .expect("close");
        assert!(ledger.outstanding().is_empty());
    }

    /// The two entry points are one operation.
    ///
    /// Document 04 requires attention to consume device handles, with host
    /// paths as "explicit separate implementations, not compulsory staging
    /// interfaces". That is only true if the separate implementation computes
    /// the same thing, so this asserts **byte equality** between a launch whose
    /// query was staged through the run's own ranges and one that read a
    /// caller's device range and wrote a caller's device range.
    #[test]
    fn device_handles_and_host_staging_give_the_same_bytes() {
        let _guard = crate::DRIVER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if moxie_cuda::device_count().expect("enumerate") == 0 {
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let ctx = RankContext::acquire(RankId(38_001), 0).expect("a rank context");
        let stream = Stream::new(&ctx).expect("a stream");
        let capability = moxie_cuda::query_device(0).expect("query device 0");
        let layer = AttentionLayer {
            geometry: geometry(),
            heads: HEADS,
            scale: moxie_plan::reciprocal_sqrt_scale(HEAD_DIM),
            visibility: Visibility::Causal,
        };
        let descriptor = super::select_paged_attention_kernel(
            &moxie_kernels::paged_attention_catalogue(),
            &capability,
            &PagedAttentionLaunch::new(layer, 1, 0, 0, 1).expect("a launch"),
        )
        .expect("a descriptor");

        let measurement = ctx.measure().expect("measure");
        let mut ledger = Ledger::new([
            CapacitySnapshot::measured(&measurement, 1 << 20).expect("device capacity"),
            CapacitySnapshot::new(moxie_types::Scope::Host, 1 << 28, 1 << 20).expect("host"),
        ])
        .expect("one ledger");
        let mut run = PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor,
            geometry(),
            HEADS,
            1,
            Staging::Host,
        )
        .map_err(|r| r.error)
        .expect("admission fits");
        run.publish_page_table(&stream, 0, vec![1, 0])
            .map_err(|r| r.error)
            .expect("a reversed mapping");

        let rows = 6u64;
        let row = (geometry().kv_heads * HEAD_DIM) as usize;
        let placements = vec![PagePlacement {
            position: 0,
            physical_page: 1,
            slot: 0,
            rows,
        }];
        run.write_rows(
            &stream,
            &placements,
            bytes(rows as usize * row, 0x3801),
            bytes(rows as usize * row, 0x3802),
        )
        .map_err(|r| r.error)
        .expect("the write fits");

        let launch = PagedAttentionLaunch::new(layer, 1, rows - 1, 0, rows).expect("a launch");
        let query = bytes((HEADS * HEAD_DIM) as usize, 0x3803);
        let staged = run
            .attend(&stream, &launch, query.clone())
            .map_err(|r| r.error)
            .expect("the staged decode runs");

        // The caller's own ranges, from the caller's own arena -- which is what
        // a lowered graph's activation slots are.
        let want = launch.query_bytes().expect("extent");
        let mut request =
            moxie_memory::PlanRequest::new("attention handles", ["bind"]).expect("a request");
        request
            .buffer(moxie_memory::BufferRequest::new(
                "caller activations",
                moxie_types::Scope::Device(ctx.uuid()),
                moxie_types::Tier::Device(moxie_types::DeviceTier::Activations),
                3 * want,
                moxie_memory::StageSpan { first: 0, last: 0 },
            ))
            .expect("one buffer");
        let reservation = ledger.admit(&request).expect("admission fits");
        let mut arena = DeviceArena::create(
            &ledger,
            reservation,
            &ctx,
            moxie_types::DeviceTier::Activations,
            3 * want,
            "caller arena",
        )
        .map_err(|r| r.error)
        .expect("an arena");
        let query_range = arena
            .allocate(want, 256, "query")
            .map_err(|r| r.error)
            .expect("a query range");
        let output_range = arena
            .allocate(want, 256, "output")
            .map_err(|r| r.error)
            .expect("an output range");
        // SAFETY: `query` outlives the synchronize below, and the destination
        // is a live range of this test's own arena on this device.
        unsafe { query_range.copy_from_host_async(&query, &stream) }.expect("upload the query");
        stream.synchronize().expect("the upload completes");

        let (query_range, output_range) = run
            .attend_into(&stream, &launch, query_range, output_range)
            .map_err(|r| r.error)
            .expect("the device-handle decode runs");
        let mut direct = vec![0u8; want as usize];
        output_range
            .copy_to_host(&mut direct)
            .expect("read the caller's output back");

        assert_eq!(
            staged, direct,
            "staging the query through the run and reading it from a device handle \
             gave different answers"
        );
        assert!(
            direct.iter().any(|b| *b != 0),
            "the decode produced nothing"
        );

        // Ownership by value already makes aliasing `f(&r, &r)` inexpressible
        // here: a caller cannot hand the same `DeviceRange` to both parameters
        // without a double move. `ranges_overlap` (unit tested below) is the
        // runtime guard for the case ownership cannot rule out on its own --
        // two distinct ranges whose bytes happen to coincide -- which this
        // kernel's `__restrict__` query and output pointers forbid either way.
        //
        // A query range too small for the launch is refused rather than read,
        // and it is handed straight back: nothing was enqueued.
        let short = arena
            .allocate(256, 256, "short")
            .map_err(|r| r.error)
            .expect("a short range");
        let refused = run
            .attend_into(&stream, &launch, short, output_range)
            .expect_err("a query range too small for the launch was accepted");
        assert!(!refused.retained_ranges());
        let (short, output_range) = refused.ranges.expect("a pre-launch refusal keeps nothing");

        arena.release(short).map_err(|r| r.error).expect("release");
        arena
            .release(output_range)
            .map_err(|r| r.error)
            .expect("release");
        arena
            .release(query_range)
            .map_err(|r| r.error)
            .expect("release");
        arena
            .close(&mut ledger)
            .map_err(|r| r.error)
            .expect("close");
        run.close(&mut ledger).map_err(|r| r.error).expect("close");
        assert!(ledger.outstanding().is_empty());
    }
}
