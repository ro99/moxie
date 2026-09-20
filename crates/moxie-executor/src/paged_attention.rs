//! Binding and launch for the common BF16 paged attention.
//!
//! Task 0037. The kernel and its catalogue identities belong to
//! `moxie-kernels`; what is here is what only this crate may do — check that a
//! launch's geometry, page mapping and visibility are the ones a descriptor
//! serves, turn admitted ranges into addresses, and wait on the event that says
//! the answer exists.
//!
//! **What this module is and is not.** `PagedAttentionRun` does hold pages
//! between launches and does track how many rows it has observed committed —
//! that is what "persistent device state" means physically. What it does not do
//! is *decide* anything about them: no retention rule, no transaction, no
//! branch, no truncation, no lineage. Those are `moxie-state`'s, and the
//! frontier here is a fact about copied bytes rather than a journal entry.
//! Until that binding exists, these mechanics are provisional and belong to the
//! task that will replace them; a second copy of the state authority's
//! decisions here is the thing task 0037 forbids by name. Admission is
//! `moxie-memory`'s throughout: every byte this module names was charged before
//! it was allocated.
//!
//! **This is not yet the common execution path.** Document 04 requires attention
//! to consume device tensor and page-table handles, with host reference paths as
//! explicit separate implementations rather than compulsory staging. The
//! `attend` below takes host query bytes and returns host output bytes, which is
//! a bring-up interface: it is how a gate drives the kernel, not how a graph
//! will. `moxie-plan` still refuses every stateful graph, and no
//! `OpParams::Attention` node lowers to this yet. Both are named work in task
//! 0037's record, not properties of the design.
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

/// A `String` that grows only through `try_reserve`.
///
/// The same device `moxie-format` and the affine binding use, for the reason
/// task 0024's review established: `format!` **aborts** when an allocation
/// fails, and the context that produces a refusal is exactly the context most
/// likely to coincide with memory pressure.
struct FallibleString(String);

impl core::fmt::Write for FallibleString {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0.try_reserve(s.len()).map_err(|_| core::fmt::Error)?;
        self.0.push_str(s);
        Ok(())
    }
}

/// Compose prose fallibly. An empty detail is a shorter true statement, not an
/// abort: the variant and the `&'static str` field are what a caller branches
/// on either way.
fn fallible(args: core::fmt::Arguments<'_>) -> String {
    use core::fmt::Write;
    let mut sink = FallibleString(String::new());
    match sink.write_fmt(args) {
        Ok(()) => sink.0,
        Err(_) => String::new(),
    }
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
        if self.history_rows > self.layer.geometry.capacity_rows()? {
            return Err(Error::CapacityExceeded {
                tier: None,
                requested_bytes: self.history_rows,
                available_bytes: self.layer.geometry.capacity_rows()?,
            });
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
        // The finding this closes: a window wider than the kernel's `u32` used
        // to be discovered inside the launch, **after** the query copy had been
        // submitted, turning a knowable refusal into an unknown submission and
        // a quarantined run. Nothing that reaches a launch can fail a width
        // conversion any more, so `window` and `grid` are total on a value of
        // this type.
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
/// do is decide what those pages mean. The committed frontier it tracks is the
/// count of rows whose copy has been observed to complete — a fact about bytes,
/// not a retention policy, a transaction or a branch. `moxie-state` owns those,
/// and binding this run to its journal is the next task rather than a second
/// authority grown here.
#[cfg(feature = "driver")]
pub mod device {
    use core::ffi::c_void;

    use moxie_cuda::{
        Event, Module, ModuleImage, RankContext, ResolvedModule, Stream, TrustedImage,
    };
    use moxie_memory::{
        BufferRequest, Ledger, LedgerId, PlanRequest, Rejection, Reservation, StageSpan,
    };
    use moxie_types::{
        DeviceTier, Error, HostTier, PagePlacement, Result, Scope, SemanticKernelDescriptor, Tier,
    };

    use super::{PageGeometry, PagedAttentionLaunch, invalid, invalid_fmt, unsupported_kernel_fmt};
    use crate::arena::{DeviceArena, DeviceRange};

    /// 256-byte alignment, as every other device range in this crate uses.
    const ALIGNMENT: u64 = 256;

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
    }

    /// A refusal that happened before anything was allocated.
    #[derive(Debug)]
    pub struct PagedAdmitRefused {
        pub error: Error,
        /// Returned held only when giving it back *also* failed.
        pub reservation: Option<Reservation>,
        pub rejection: Option<Rejection>,
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
        /// One launch's query rows, or the encoded table bytes in flight.
        Query(Vec<u8>),
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

    /// One layer's admitted paged key/value state, plus the per-launch query and
    /// output ranges and the resolved attention symbol.
    #[derive(Debug)]
    #[must_use = "an unclosed run keeps its arena and its reservation"]
    pub struct PagedAttentionRun<'ctx> {
        module: ResolvedModule<'ctx>,
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
        /// The published mapping, and the absolute row its logical page zero
        /// names.
        ///
        /// **A table without its base is not a mapping.** The base makes it a
        /// view of *absolute* rows: every write is checked against it, every
        /// launch must name it, and a republication must agree with it wherever
        /// both describe a page that holds written rows.
        page_table: Vec<u32>,
        page_table_base: u64,
        staging: Staging,
        /// Rows whose copy into the pages this run has **observed** complete.
        ///
        /// A physical high-water mark, not a frontier. What is history is the
        /// state authority's to say, and it says it by publishing rows this run
        /// has already returned successfully for. The two are checked against
        /// each other rather than one standing in for the other: this refuses a
        /// launch reading bytes it never wrote, and the authority refuses one
        /// reading rows it never committed.
        written: u64,
        arena_bytes: u64,
        ledger: LedgerId,
        ctx: &'ctx RankContext,
        /// What this run is holding across work whose completion it has not
        /// observed. An append holds two vectors and hands both back unjoined.
        held: Option<RefusedSource>,
        quarantined: bool,
    }

    impl<'ctx> PagedAttentionRun<'ctx> {
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
        ) -> std::result::Result<Self, PagedAdmitRefused> {
            let fail = |error| PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
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
            // image anyway, with its own declaration silently ignored. Task
            // 0021's review produced exactly that shape one package over:
            // operation, ABI and hash matched, the symbol did not, and every
            // GPU computed the wrong activation for a valid-looking plan.
            //
            // The descriptor must therefore **be** one the built-in package
            // declares. That binds operation, ABI, operand roles and
            // precisions, output, accumulation, rounding, layout, shape bounds,
            // SM, workspace, image digest and symbols together, which is the
            // only form of this check that cannot be half-satisfied. Selection
            // still runs against whatever catalogue it is given — that is task
            // 0012's design — and this is the boundary where a selected
            // descriptor becomes a launch.
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
            let extents = match Extents::derive(&geometry, heads, max_rows, staging) {
                Ok(extents) => extents,
                Err(error) => return Err(fail(error)),
            };
            let request = match resource_request(&geometry, heads, max_rows, staging, ctx) {
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
            if regions.try_reserve_exact(2).is_err() {
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
            let mut hold: Vec<DeviceRange<'ctx>> = Vec::new();
            if hold.try_reserve_exact(5).is_err() {
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
                let mut room = symbols.try_reserve_exact(descriptor.symbols.len()).is_ok();
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
                symbols
            };
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
            let (query, output) = if staging == Staging::Host {
                let output = hold.pop().expect("output range");
                let query = hold.pop().expect("query range");
                (Some(query), Some(output))
            } else {
                (None, None)
            };
            let table = hold.pop().expect("page table range");
            let values = hold.pop().expect("value range");
            let keys = hold.pop().expect("key range");
            Ok(Self {
                module,
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
                staging,
                page_table,
                page_table_base: 0,
                written: 0,
                arena_bytes: extents.total,
                ledger: ledger.id(),
                ctx,
                held: None,
                quarantined: false,
            })
        }

        /// Device bytes this run holds: persistent pages, table, query, output.
        ///
        /// A memory bound and **not** a speed measurement: O6 and O7 are open
        /// and nothing here is timed.
        pub const fn arena_bytes(&self) -> u64 {
            self.arena_bytes
        }

        /// Rows this run has observed copied into its pages.
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

        pub const fn geometry(&self) -> &PageGeometry {
            &self.geometry
        }

        pub fn descriptor(&self) -> &SemanticKernelDescriptor {
            &self.descriptor
        }

        /// Publish the logical-to-physical page mapping this run will use.
        ///
        /// One table, uploaded once and then read by the kernel and by this
        /// host code for exactly the same addresses. A second host-side mapping
        /// would be the page table that disagrees with the page table.
        ///
        /// Refused once rows are committed: remapping pages under a live
        /// history would move rows that something has already attended to.
        pub fn publish_page_table(
            &mut self,
            stream: &Stream<'ctx>,
            base: u64,
            table: Vec<u32>,
        ) -> std::result::Result<(), PagedRunRefused> {
            // The table goes back as the entries it arrived as. Re-encoding it
            // into bytes to report a refusal is an allocation on the refusal
            // path, which is where allocations fail.
            let give_back = |error, table: Vec<u32>| PagedRunRefused {
                error,
                source: Some(RefusedSource::PageTable(table)),
            };
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
            // Every physical identity must exist, and no two logical pages may
            // name the same one: aliasing pages would make an append overwrite
            // history that is still visible, which no later check could detect.
            let mut seen: Vec<bool> = Vec::new();
            if seen.try_reserve_exact(pages as usize).is_err() {
                return Err(give_back(
                    Error::CapacityExceeded {
                        tier: Some(Tier::Host(HostTier::CpuWorkspace)),
                        requested_bytes: pages,
                        available_bytes: 0,
                    },
                    table,
                ));
            }
            seen.resize(pages as usize, false);
            for (logical, physical) in table.iter().enumerate() {
                let physical = u64::from(*physical);
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
                if seen[physical as usize] {
                    return Err(give_back(
                        invalid_fmt(
                            "page_table",
                            format_args!("physical page {physical} is named twice"),
                        ),
                        table,
                    ));
                }
                seen[physical as usize] = true;
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

            let mut bytes: Vec<u8> = Vec::new();
            if bytes.try_reserve_exact(table.len() * 4).is_err() {
                return Err(give_back(
                    Error::CapacityExceeded {
                        tier: Some(Tier::Host(HostTier::Pageable)),
                        requested_bytes: (table.len() * 4) as u64,
                        available_bytes: 0,
                    },
                    table,
                ));
            }
            for entry in &table {
                bytes.extend_from_slice(&entry.to_le_bytes());
            }
            let range = self.table.as_ref().expect("live page table range");
            // SAFETY: the source is held by this run until completion is
            // observed, and the destination is this run's own admitted range.
            if let Err(error) = unsafe { range.copy_from_host_async(&bytes, stream) } {
                self.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            self.held = Some(RefusedSource::Query(bytes));
            if let Err(error) = self.settle(Ok(()), stream) {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            self.held = None;
            self.page_table = table;
            self.page_table_base = base;
            Ok(())
        }

        /// Whether this stream belongs to the device this run was admitted on.
        ///
        /// An offset resolved inside another device's allocation names the
        /// wrong bytes, and enqueuing on a foreign stream would order the copy
        /// against work this run never sees. The affine linear binding learned
        /// this from a review that drove 3090 leases through a 5060 Ti backing
        /// and got a confident, different answer.
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

        /// Write `rows` dense rows of keys and values where the **state
        /// authority** says they go.
        ///
        /// `placements` comes from `moxie_state::DeviceKvSequence`, which owns
        /// retention, the frontier and the ring. This run performs them: it
        /// does not compute `position / page_tokens`, does not decide which
        /// page may be overwritten, and does not publish anything as history.
        /// Two implementations of one mapping is the failure that split is for.
        ///
        /// What it *does* own is a physical fact: which rows it has observed
        /// copied. A refusal before anything is enqueued leaves that and every
        /// prior byte untouched and hands the sources back; a failure after
        /// enqueue quarantines the run, keeps the sources, and still does not
        /// advance it. There is no state in between: a partially copied write
        /// is never reported as written, and the authority only publishes rows
        /// this call has returned successfully for.
        #[allow(clippy::result_large_err)]
        pub fn write_rows(
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
            if self.quarantined {
                return Err(give_back(
                    invalid("run", "this run is quarantined"),
                    keys,
                    values,
                ));
            }
            if let Err(error) = self.same_device(stream) {
                return Err(give_back(error, keys, values));
            }
            if placements.is_empty() {
                return Err(give_back(
                    invalid("placements", "a write with no placement"),
                    keys,
                    values,
                ));
            }
            let row_bytes = match self.geometry.row_elements().and_then(|e| {
                e.checked_mul(2)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))
            }) {
                Ok(bytes) => bytes,
                Err(error) => return Err(give_back(error, keys, values)),
            };

            // The placements are checked, not trusted: the authority owns the
            // mapping, and this run owns the extents. A page identity outside
            // the admitted pages, a slot outside a page, a gap or a step
            // backwards would each write somewhere nothing asked for.
            let mut rows = 0u64;
            let mut position = placements[0].position;
            for placement in placements {
                if placement.position != position {
                    return Err(give_back(
                        invalid(
                            "placements",
                            "the runs are not contiguous and ascending from the first",
                        ),
                        keys,
                        values,
                    ));
                }
                if placement.rows == 0 {
                    return Err(give_back(
                        invalid("placements", "a run of no rows"),
                        keys,
                        values,
                    ));
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
                        return Err(give_back(
                            invalid_fmt(
                                "placements",
                                format_args!(
                                    "row {} is placed on physical page {} and the published \
                                     mapping sends it to {physical}",
                                    placement.position, placement.physical_page
                                ),
                            ),
                            keys,
                            values,
                        ));
                    }
                    Err(error) => return Err(give_back(error, keys, values)),
                }
                if placement.slot != placement.position % self.geometry.page_tokens {
                    return Err(give_back(
                        invalid_fmt(
                            "placements",
                            format_args!(
                                "row {} is placed at slot {} and its page holds it at {}",
                                placement.position,
                                placement.slot,
                                placement.position % self.geometry.page_tokens
                            ),
                        ),
                        keys,
                        values,
                    ));
                }
                let slot_end = placement.slot.checked_add(placement.rows);
                if slot_end.is_none_or(|end| end > self.geometry.page_tokens) {
                    return Err(give_back(
                        invalid(
                            "placements",
                            "a run crosses the end of the page it is placed on",
                        ),
                        keys,
                        values,
                    ));
                }
                match placement.end() {
                    Some(end) => position = end,
                    None => {
                        return Err(give_back(
                            Error::Dim(moxie_types::DimError::Overflow),
                            keys,
                            values,
                        ));
                    }
                }
                rows += placement.rows;
            }
            let want = match rows.checked_mul(row_bytes) {
                Some(want) => want,
                None => {
                    return Err(give_back(
                        Error::Dim(moxie_types::DimError::Overflow),
                        keys,
                        values,
                    ));
                }
            };
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
            // physical high-water mark, not a frontier: the authority publishes
            // history, and it does so only for a call that returned `Ok`.
            self.written = self.written.max(position);
            Ok(())
        }

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
        /// Nothing is retained on refusal because nothing of the caller's was
        /// taken: the query is already device-resident and this call copies no
        /// host bytes. A failure after the launch still quarantines the run,
        /// because the ranges may still be read.
        pub fn attend_into(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
            query: &DeviceRange<'ctx>,
            output: &DeviceRange<'ctx>,
        ) -> Result<()> {
            self.check_attend(stream, launch)?;
            let want = launch.query_bytes()?;
            for (range, what) in [(query, "query"), (output, "output")] {
                if range.device_uuid() != self.ctx.uuid() {
                    return Err(invalid(what, "this range belongs to another device"));
                }
                if range.bytes() < want {
                    return Err(invalid_fmt(
                        what,
                        format_args!("{} byte(s) for a launch needing {want}", range.bytes()),
                    ));
                }
            }
            let scalars = Self::abi_scalars(launch)?;
            let addresses = [
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
            ];
            self.launch_with(stream, scalars, addresses)
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

        /// Attend `launch.rows` query rows against the committed history.
        ///
        /// The launch's declared history must be one this run actually holds:
        /// a launch that claimed more rows than were committed would attend
        /// over uninitialized pages and return a confident wrong answer.
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

            // The last thing that can be known without the device: every ABI
            // width this launch needs. A failure here is an ordinary refusal
            // with the query handed back.
            let scalars = match Self::abi_scalars(launch) {
                Ok(scalars) => scalars,
                Err(error) => return Err(give_back(error, query)),
            };

            // --- from here on, work is in flight -------------------------
            self.held = Some(RefusedSource::Query(query));
            if let Err(error) = self.enqueue_attend(stream, scalars) {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            let mut out = out_len;
            if let Err(error) = self
                .output
                .as_ref()
                .expect("live output range")
                .copy_to_host_at(0, &mut out)
            {
                self.quarantined = true;
                return Err(PagedRunRefused {
                    error: self.attribute(error),
                    source: None,
                });
            }
            self.held = None;
            Ok(out)
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

        fn enqueue_attend(&mut self, stream: &Stream<'ctx>, scalars: AbiScalars) -> Result<()> {
            let Some(RefusedSource::Query(query_source)) = self.held.as_ref() else {
                return Err(invalid("attend", "the attend is not holding its query"));
            };
            let query_range = self.query.as_ref().expect("live query range");
            // SAFETY: the source is held until completion is observed and the
            // destination is this run's own admitted range.
            if let Err(error) = unsafe { query_range.copy_from_host_async(query_source, stream) } {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            let mut addresses = [0u64; 5];
            for (slot, range) in [
                query_range,
                self.keys.as_ref().expect("live key range"),
                self.values.as_ref().expect("live value range"),
                self.table.as_ref().expect("live page table range"),
                self.output.as_ref().expect("live output range"),
            ]
            .into_iter()
            .enumerate()
            {
                match range.device_address() {
                    Ok(address) => addresses[slot] = address,
                    Err(error) => {
                        self.quarantined = true;
                        return Err(self.attribute(error));
                    }
                }
            }
            self.launch_with(stream, scalars, addresses)
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
            self.settle(launched, stream)
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
        /// still be reading these ranges.
        #[allow(clippy::result_large_err)]
        pub fn close(
            mut self,
            ledger: &mut Ledger,
        ) -> std::result::Result<(), PagedCloseRefused<'ctx>> {
            if self.quarantined {
                let error = invalid(
                    "close",
                    "this run is quarantined; its ranges may still be in flight",
                );
                return Err(PagedCloseRefused { run: self, error });
            }
            if ledger.id() != self.ledger {
                let error = invalid("ledger", "this run belongs to another ledger");
                return Err(PagedCloseRefused { run: self, error });
            }
            for slot in 0..5 {
                let taken = match slot {
                    0 => self.output.take(),
                    1 => self.query.take(),
                    2 => self.table.take(),
                    3 => self.values.take(),
                    _ => self.keys.take(),
                };
                let Some(range) = taken else { continue };
                let arena = self.arena.as_mut().expect("an open run has its arena");
                if let Err(refused) = arena.release(range) {
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

    /// The aligned extents one run admits.
    struct Extents {
        payload: u64,
        table: u64,
        query: u64,
        persistent: u64,
        per_step: u64,
        total: u64,
        pages_usize: usize,
    }

    impl Extents {
        fn derive(
            geometry: &PageGeometry,
            heads: u64,
            max_rows: u64,
            staging: Staging,
        ) -> Result<Self> {
            let payload = align_up(geometry.payload_bytes()?)?;
            let table = align_up(
                geometry
                    .pages
                    .checked_mul(4)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?,
            )?;
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
            // Zero for a direct-device run: the caller's ranges are the query
            // and the output, and charging for a second pair here is what makes
            // an admissible plan refusable.
            let per_step = if staging == Staging::Host {
                query
                    .checked_mul(2)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?
            } else {
                0
            };
            let total = persistent
                .checked_add(per_step)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let pages_usize = usize::try_from(geometry.pages)
                .map_err(|_| invalid("pages", "the page count exceeds this host's usize"))?;
            Ok(Self {
                payload,
                table,
                query,
                persistent,
                per_step,
                total,
                pages_usize,
            })
        }
    }

    /// The admission request one paged attention run makes.
    ///
    /// Four device buffers and one host readback, each named for what it is.
    /// The pages are `PersistentState` and the query and output are
    /// `Activations`, because the ledger's report is about what memory is *for*
    /// and a cache charged as scratch is a cache nobody can see growing.
    pub fn resource_request(
        geometry: &PageGeometry,
        heads: u64,
        max_rows: u64,
        staging: Staging,
        ctx: &RankContext,
    ) -> Result<PlanRequest> {
        let extents = Extents::derive(geometry, heads, max_rows, staging)?;
        let mut request = PlanRequest::new(
            moxie_memory::fallible::text(format_args!(
                "paged-attention-{}x{}",
                geometry.kv_heads, geometry.head_dim
            ))?,
            ["append", "launch", "read"],
        )?;
        let scope = Scope::Device(ctx.uuid());
        request.buffer(BufferRequest::new(
            "kv-pages-keys",
            scope,
            Tier::Device(DeviceTier::KvStatePages),
            extents.payload,
            StageSpan::inclusive(0, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "kv-pages-values",
            scope,
            Tier::Device(DeviceTier::KvStatePages),
            extents.payload,
            StageSpan::inclusive(0, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "kv-page-table",
            scope,
            Tier::Device(DeviceTier::KvStatePages),
            extents.table,
            StageSpan::inclusive(0, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "attention-query",
            scope,
            Tier::Device(DeviceTier::Activations),
            extents.query,
            StageSpan::inclusive(1, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "attention-output",
            scope,
            Tier::Device(DeviceTier::Activations),
            extents.query,
            StageSpan::inclusive(1, 2),
        ))?;
        request.buffer(BufferRequest::new(
            "attention-output-readback",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            extents.query,
            StageSpan::at(2),
        ))?;
        Ok(request)
    }

    fn align_up(bytes: u64) -> Result<u64> {
        bytes
            .checked_add(ALIGNMENT - 1)
            .map(|v| v / ALIGNMENT * ALIGNMENT)
            .ok_or_else(|| invalid("align", "the aligned extent overflows"))
    }

    fn give_back(ledger: &mut Ledger, reservation: Reservation, error: Error) -> PagedAdmitRefused {
        match ledger.release(reservation) {
            Ok(()) => PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
            },
            Err(refused) => PagedAdmitRefused {
                error,
                reservation: Some(refused.reservation),
                rejection: None,
            },
        }
    }

    fn unwind<'ctx>(
        mut arena: DeviceArena<'ctx>,
        mut ranges: Vec<DeviceRange<'ctx>>,
        ledger: &mut Ledger,
        error: Error,
    ) -> PagedAdmitRefused {
        while let Some(range) = ranges.pop() {
            if let Err(refused) = arena.release(range) {
                return PagedAdmitRefused {
                    error: refused.error,
                    reservation: None,
                    rejection: None,
                };
            }
        }
        match arena.close(ledger) {
            Ok(()) => PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
            },
            Err(refused) => PagedAdmitRefused {
                error: refused.error,
                reservation: None,
                rejection: None,
            },
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
    use moxie_types::{PagePlacement, RankId};

    use super::device::{PagedAttentionRun, Staging};
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

    /// A direct-device run is not charged for staging it never uses.
    #[test]
    fn a_direct_device_run_admits_no_staging_buffers() {
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

        // The same geometry, admitted both ways, against the same ledger.
        let measurement = ctx.measure().expect("measure");
        let mut ledger = Ledger::new([
            CapacitySnapshot::measured(&measurement, 1 << 20).expect("device capacity"),
            CapacitySnapshot::new(moxie_types::Scope::Host, 1 << 28, 1 << 20).expect("host"),
        ])
        .expect("one ledger");
        let staged = PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor(),
            geometry(),
            HEADS,
            4,
            Staging::Host,
        )
        .map_err(|r| r.error)
        .expect("admission fits");
        let staged_bytes = staged.arena_bytes();
        staged
            .close(&mut ledger)
            .map_err(|r| r.error)
            .expect("close");

        let mut direct = PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor(),
            geometry(),
            HEADS,
            4,
            Staging::DeviceHandles,
        )
        .map_err(|r| r.error)
        .expect("admission fits");
        let direct_bytes = direct.arena_bytes();

        // Four query rows of four heads by 64 in BF16, twice over: the query
        // and the output the direct run does not hold.
        let per_step = 2 * 4 * HEADS * HEAD_DIM * 2;
        assert_eq!(
            staged_bytes - direct_bytes,
            per_step,
            "a direct-device run was charged for staging buffers"
        );
        // And the host-staged entry point is refused rather than silently
        // reaching for ranges that are not there.
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

        run.attend_into(&stream, &launch, &query_range, &output_range)
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

        // A range from another device, and one too small for the launch, are
        // refused rather than read.
        assert!(
            run.attend_into(&stream, &launch, &query_range, &query_range)
                .is_ok(),
            "a caller may aim the output at any range of the right size"
        );
        let short = arena
            .allocate(256, 256, "short")
            .map_err(|r| r.error)
            .expect("a short range");
        assert!(
            run.attend_into(&stream, &launch, &short, &output_range)
                .is_err(),
            "a query range too small for the launch was accepted"
        );

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
