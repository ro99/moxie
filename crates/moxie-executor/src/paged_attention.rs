//! Binding and launch for the common BF16 paged attention.
//!
//! Task 0037. The kernel and its catalogue identities belong to
//! `moxie-kernels`; what is here is what only this crate may do — check that a
//! launch's geometry, page mapping and visibility are the ones a descriptor
//! serves, turn admitted ranges into addresses, and wait on the event that says
//! the answer exists.
//!
//! **What this module is not.** It is not a state owner. It holds no pages
//! between launches, publishes no frontier and decides no retention: those are
//! `moxie-state`'s, and a second copy of them here is the "second state
//! authority" task 0037 forbids by name. A launch is handed ranges that were
//! already admitted and already filled, and it reads them. It is not an
//! admission authority either — every byte it names was charged by
//! `moxie-memory` before this module saw it.
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

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
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

fn unsupported_kernel(operation: &'static str, detail: impl Into<String>) -> Error {
    Error::UnsupportedKernel {
        operation,
        detail: detail.into(),
    }
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
            return Err(invalid(
                "page_geometry",
                format!(
                    "{} kv head(s) of {} in pages of {} row(s), {} page(s): none may be zero",
                    self.kv_heads, self.head_dim, self.page_tokens, self.pages
                ),
            ));
        }
        if self.head_dim > moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM {
            return Err(Error::Unsupported {
                capability: "attention_head_dim",
                reason: format!(
                    "head dimension {} exceeds the {} this image serves",
                    self.head_dim,
                    moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM
                ),
            });
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
            return Err(invalid(
                "slot",
                format!("slot {slot} in a page of {} row(s)", self.page_tokens),
            ));
        }
        if physical_page >= self.pages {
            return Err(invalid(
                "physical_page",
                format!(
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

/// One launch of the attention kernel: query rows against a visible history.
///
/// Whole prefill, one prefill chunk and a single decode row are the same value
/// with a different `rows`, which is the point of having one kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedAttentionLaunch {
    pub geometry: PageGeometry,
    /// Query heads. Equal to `geometry.kv_heads` for multi-head attention, a
    /// multiple of it for grouped-query attention.
    pub heads: u64,
    /// The layer's **declared** score scale, not a derived one. Gemma-style
    /// layers normalize queries and keys per head and declare exactly 1.0.
    pub scale: f32,
    pub visibility: Visibility,
    /// Query rows in this launch.
    pub rows: u64,
    /// The absolute position of query row zero.
    pub first_position: u64,
    /// The absolute position of logical row zero of the history.
    ///
    /// Above zero for a layer that has reclaimed what its window can no longer
    /// see. It must be a whole number of pages: a partially reclaimed page
    /// would put a row at a slot that depends on the eviction history, which is
    /// the state a page table exists to avoid.
    pub history_base: u64,
    /// Rows the history holds, from `history_base`.
    pub history_rows: u64,
}

impl PagedAttentionLaunch {
    /// Check every geometric and positional precondition this launch has.
    ///
    /// Called by construction *and* re-applied at admission, for the reason
    /// `affine_linear::descriptor_serves` exists: a public entry point that
    /// takes a checked value and trusts it is a check that happened to
    /// something nobody kept.
    pub fn check(&self) -> Result<()> {
        self.geometry.check()?;
        if self.heads == 0 || !self.heads.is_multiple_of(self.geometry.kv_heads) {
            // The same grouping rule the graph validates and the oracle
            // enforces. A ratio that does not divide gives one group more query
            // heads than another, which is not a layout any released checkpoint
            // uses and not one this launch will invent.
            return Err(invalid(
                "heads",
                format!(
                    "{} query head(s) must be a nonzero multiple of {} key/value head(s)",
                    self.heads, self.geometry.kv_heads
                ),
            ));
        }
        if !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(invalid(
                "attention_scale",
                format!(
                    "score scale must be finite and positive, got {}",
                    self.scale
                ),
            ));
        }
        if let Visibility::SlidingWindow { window } = self.visibility
            && window == 0
        {
            return Err(invalid(
                "visibility",
                "a sliding window of zero sees nothing, including the query's own position",
            ));
        }
        if self.rows == 0 {
            return Err(invalid("rows", "a launch with no query row"));
        }
        if !self.history_base.is_multiple_of(self.geometry.page_tokens) {
            return Err(invalid(
                "history_base",
                format!(
                    "history base {} is not a whole number of {}-row pages; a partially \
                     reclaimed page has no stable slot for its rows",
                    self.history_base, self.geometry.page_tokens
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
        if self.history_rows > self.geometry.capacity_rows()? {
            return Err(Error::CapacityExceeded {
                tier: None,
                requested_bytes: self.history_rows,
                available_bytes: self.geometry.capacity_rows()?,
            });
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
            return Err(invalid(
                "first_position",
                format!(
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
            return Err(invalid(
                "first_position",
                format!(
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
        Ok(self.history_rows.div_ceil(self.geometry.page_tokens))
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
            .checked_mul(self.heads)
            .and_then(|r| r.checked_mul(self.geometry.head_dim))
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
        match self.visibility {
            Visibility::Causal => Ok(0),
            Visibility::SlidingWindow { window } => {
                u32::try_from(window).map_err(|_| Error::Unsupported {
                    capability: "attention_window",
                    reason: format!("a window of {window} rows exceeds this ABI's u32"),
                })
            }
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
        let position = self.first_position + row;
        key >= self.history_base
            && key < self.history_base + self.history_rows
            && self.visibility.allows(position, key)
    }

    /// The launch grid: one block per query row and head.
    pub fn grid(&self) -> Result<(u32, u32, u32)> {
        let x = u32::try_from(self.rows).map_err(|_| {
            invalid(
                "rows",
                format!("{} query rows exceed a u32 launch grid", self.rows),
            )
        })?;
        let y = u32::try_from(self.heads).map_err(|_| {
            invalid(
                "heads",
                format!("{} heads exceed a u32 launch grid", self.heads),
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
        return Err(unsupported_kernel(
            "paged_attention",
            format!(
                "{} does not serve this launch: {reason}",
                descriptor.id.0.as_str()
            ),
        ));
    }
    let Some(descriptor) = chosen.filter(|_| matched == 1) else {
        return Err(unsupported_kernel(
            "paged_attention",
            format!(
                "expected exactly one paged attention descriptor for {} row(s) of {} head(s) \
                 by {} on sm_{}{}; found {matched}",
                launch.rows,
                launch.heads,
                launch.geometry.head_dim,
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
        Some(reason) => Err(unsupported_kernel(
            "paged_attention",
            format!("{} does not serve this launch: {reason}", descriptor.id.0),
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
    if descriptor.shape.max_input < launch.geometry.head_dim
        || descriptor.shape.max_output < launch.geometry.head_dim
    {
        return Some("the head dimension exceeds the shape bounds it declares");
    }
    if descriptor.shape.max_rows < launch.rows {
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

    fn launch() -> PagedAttentionLaunch {
        PagedAttentionLaunch {
            geometry: geometry(),
            heads: 8,
            scale: moxie_plan::reciprocal_sqrt_scale(64),
            visibility: Visibility::Causal,
            rows: 4,
            first_position: 60,
            history_base: 0,
            history_rows: 64,
        }
    }

    #[test]
    fn a_launch_reports_the_bytes_its_operands_occupy() {
        let l = launch();
        // 2 kv heads of 64, 32 rows to a page, two bytes each.
        assert_eq!(l.geometry.row_elements().unwrap(), 128);
        assert_eq!(l.geometry.page_bytes().unwrap(), 128 * 32 * 2);
        assert_eq!(l.geometry.payload_bytes().unwrap(), 128 * 32 * 2 * 8);
        assert_eq!(l.geometry.capacity_rows().unwrap(), 256);
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
        let mut l = launch();
        l.history_rows = 65;
        l.first_position = 64;
        l.rows = 1;
        assert_eq!(l.logical_pages().unwrap(), 3);
        assert_eq!(l.page_table_bytes().unwrap(), 12);
        l.check().unwrap();
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
    fn grouped_and_multi_head_ratios_are_checked_the_way_the_graph_checks_them() {
        let mut l = launch();
        // 8 query heads over 2 key/value heads: four queries per group.
        l.check().unwrap();
        // Multi-head is the same value with the ratio at one.
        l.heads = 2;
        l.check().unwrap();
        // Fewer query heads than key/value heads, and a ratio that does not
        // divide, are both refused.
        l.heads = 1;
        assert!(l.check().is_err());
        l.heads = 6;
        l.geometry.kv_heads = 4;
        assert!(l.check().is_err());
        l.heads = 0;
        assert!(l.check().is_err());
    }

    #[test]
    fn the_declared_scale_must_be_finite_and_positive() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut l = launch();
            l.scale = bad;
            assert!(l.check().is_err(), "scale {bad} was accepted");
        }
        // 1.0 is an ordinary declared scale, not a suspicious one: a
        // Gemma-style layer normalizes its queries and keys and declares it.
        let mut l = launch();
        l.scale = 1.0;
        l.check().unwrap();
    }

    #[test]
    fn a_query_beyond_the_frontier_is_refused_rather_than_shortened() {
        // The append-before-attend rule. Attending at position 64 against a
        // 64-row history means the query's own key was never written, and the
        // kernel would then attend over 64 keys and return a confident wrong
        // answer instead of nothing.
        let mut l = launch();
        l.first_position = 64;
        l.rows = 1;
        assert!(l.check().is_err());
        // The last legal position is the frontier's last row.
        l.first_position = 63;
        l.check().unwrap();
        // A chunk that starts legally and runs past the end is refused too.
        l.first_position = 62;
        l.rows = 4;
        assert!(l.check().is_err());
    }

    #[test]
    fn a_query_below_the_retained_base_is_refused() {
        // A sliding layer has reclaimed everything below `history_base`. A
        // query there cannot see its own key, and the answer is a refusal
        // rather than attention over whatever the pages now hold.
        let mut l = launch();
        l.history_base = 32;
        l.first_position = 31;
        l.rows = 1;
        l.history_rows = 32;
        assert!(l.check().is_err());
        l.first_position = 32;
        l.check().unwrap();
    }

    #[test]
    fn a_history_base_inside_a_page_is_refused() {
        let mut l = launch();
        l.history_base = 16;
        l.first_position = 70;
        assert!(matches!(
            l.check().unwrap_err(),
            Error::InvalidRequest {
                field: "history_base",
                ..
            }
        ));
        // A whole page of reclamation is fine.
        l.history_base = 32;
        l.check().unwrap();
    }

    #[test]
    fn a_history_longer_than_the_admitted_pages_is_a_capacity_refusal() {
        // Not an invalid request: the geometry is legal and the bytes are not
        // there. The distinction is what lets a caller admit more pages and
        // retry rather than rewrite its request.
        let mut l = launch();
        l.history_rows = 257;
        l.first_position = 256;
        l.rows = 1;
        assert!(matches!(
            l.check().unwrap_err(),
            Error::CapacityExceeded { .. }
        ));
    }

    #[test]
    fn an_empty_history_and_an_empty_launch_are_typed_errors() {
        let mut l = launch();
        l.history_rows = 0;
        assert!(l.check().is_err());
        let mut l = launch();
        l.rows = 0;
        assert!(l.check().is_err());
        let mut l = launch();
        l.geometry.page_tokens = 0;
        assert!(l.check().is_err());
        let mut l = launch();
        l.geometry.pages = 0;
        assert!(l.check().is_err());
    }

    #[test]
    fn a_head_dimension_wider_than_the_image_serves_is_unsupported() {
        // Unsupported, not invalid: the request is coherent and this build
        // cannot serve it. A named refusal is the contract; silently reading
        // past the end of a row is what it prevents.
        let mut l = launch();
        l.geometry.head_dim = moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM + 1;
        assert!(matches!(
            l.geometry.check().unwrap_err(),
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
        let mut l = launch();
        l.first_position = u64::MAX;
        assert!(l.check().is_err());
        let mut l = launch();
        l.history_base = 0;
        l.history_rows = u64::MAX - 1;
        assert!(l.check().is_err());
    }

    #[test]
    fn the_window_parameter_says_causal_with_zero_and_nothing_else_does() {
        let mut l = launch();
        assert_eq!(l.window().unwrap(), 0);
        l.visibility = Visibility::SlidingWindow { window: 16 };
        assert_eq!(l.window().unwrap(), 16);
        // A window of zero would be indistinguishable from causal in the ABI,
        // so it is refused at construction rather than encoded.
        l.visibility = Visibility::SlidingWindow { window: 0 };
        assert!(l.check().is_err());
        l.visibility = Visibility::SlidingWindow {
            window: u64::from(u32::MAX) + 1,
        };
        assert!(matches!(l.window(), Err(Error::Unsupported { .. })));
    }

    #[test]
    fn visibility_is_decided_on_absolute_positions() {
        // R21: the same launch, one chunk starting at zero and one starting
        // later, must mask on the true position rather than on the row index.
        let mut l = launch();
        l.first_position = 0;
        l.rows = 4;
        assert!(l.allows(0, 0));
        assert!(!l.allows(0, 1), "row zero must not see the future");
        assert!(l.allows(3, 3) && l.allows(3, 0));

        l.first_position = 60;
        assert!(l.allows(0, 60) && l.allows(0, 59));
        assert!(!l.allows(0, 61), "position 60 must not see 61");

        l.visibility = Visibility::SlidingWindow { window: 4 };
        assert!(l.allows(0, 57) && !l.allows(0, 56));
        // Reclaimed rows are invisible whatever the window says.
        l.history_base = 32;
        l.history_rows = 32;
        assert!(!l.allows(0, 31));
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
    use moxie_types::{DeviceTier, Error, HostTier, Result, Scope, SemanticKernelDescriptor, Tier};

    use super::{PageGeometry, PagedAttentionLaunch, invalid};
    use crate::arena::{DeviceArena, DeviceRange};

    /// 256-byte alignment, as every other device range in this crate uses.
    const ALIGNMENT: u64 = 256;

    /// A refusal that happened before anything was allocated.
    #[derive(Debug)]
    pub struct PagedAdmitRefused {
        pub error: Error,
        /// Returned held only when giving it back *also* failed.
        pub reservation: Option<Reservation>,
        pub rejection: Option<Rejection>,
    }

    /// A refused append or attend.
    ///
    /// `source` is `Some` when the refusal happened **before** anything was
    /// enqueued, and `None` when the run retained it: submitted work whose
    /// completion is unknown may still be reading those bytes.
    #[derive(Debug)]
    pub struct PagedRunRefused {
        pub error: Error,
        pub source: Option<Vec<u8>>,
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
        /// The host's copy of the page table, which is what resolves a logical
        /// page to an offset when rows are written. One table, read by the host
        /// for addresses and by the kernel for the same addresses.
        page_table: Vec<u32>,
        /// Rows whose copy into the pages has been **observed** to complete.
        committed_rows: u64,
        arena_bytes: u64,
        ledger: LedgerId,
        ctx: &'ctx RankContext,
        held: Option<Vec<u8>>,
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
        ) -> std::result::Result<Self, PagedAdmitRefused> {
            let fail = |error| PagedAdmitRefused {
                error,
                reservation: None,
                rejection: None,
            };
            if descriptor.sm.major != ctx.capability().compute_major
                || descriptor.sm.minor != ctx.capability().compute_minor
            {
                return Err(fail(super::unsupported_kernel(
                    "paged_attention",
                    format!(
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
                return Err(fail(invalid(
                    "admission",
                    format!(
                        "{heads} query head(s) over {} key/value head(s) and {max_rows} row(s) \
                         per launch",
                        geometry.kv_heads
                    ),
                )));
            }
            // Admission re-applies selection's predicate, on the widest launch
            // this run can serve. A descriptor that cannot serve that launch
            // must be refused now rather than at the first attend.
            let widest = PagedAttentionLaunch {
                geometry,
                heads,
                scale: 1.0,
                visibility: moxie_plan::Visibility::Causal,
                rows: max_rows,
                first_position: 0,
                history_base: 0,
                history_rows: max_rows,
            };
            if let Err(error) = super::descriptor_serves(&descriptor, &widest) {
                return Err(fail(error));
            }

            let extents = match Extents::derive(&geometry, heads, max_rows) {
                Ok(extents) => extents,
                Err(error) => return Err(fail(error)),
            };
            let request = match resource_request(&geometry, heads, max_rows, ctx) {
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
            let mut arena = match DeviceArena::create_partitioned(
                ledger,
                reservation,
                ctx,
                // The pages are persistent state; the query and output are
                // per-step activations. One physical allocation, two declared
                // tiers, because the ledger's totals are about what the bytes
                // are for and not only about how many there are.
                &[
                    (DeviceTier::KvStatePages, extents.persistent),
                    (DeviceTier::Activations, extents.per_step),
                ],
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
            for (bytes, name) in [
                (extents.payload, "attention-keys"),
                (extents.payload, "attention-values"),
                (extents.table, "attention-page-table"),
                (extents.query, "attention-query"),
                (extents.query, "attention-output"),
            ] {
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
            let output = hold.pop().expect("output range");
            let query = hold.pop().expect("query range");
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
                query: Some(query),
                output: Some(output),
                page_table,
                committed_rows: 0,
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

        /// Rows whose copy into the pages has been observed to complete.
        ///
        /// Reported separately from the admitted capacity and from what a
        /// launch declares visible, because the three are different numbers and
        /// conflating them is how a cache claims a context it does not hold.
        pub const fn committed_rows(&self) -> u64 {
            self.committed_rows
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
            table: Vec<u32>,
        ) -> std::result::Result<(), PagedRunRefused> {
            let give_back = |error, table: Vec<u32>| PagedRunRefused {
                error,
                source: Some(table.iter().flat_map(|v| v.to_le_bytes()).collect()),
            };
            if self.quarantined {
                return Err(give_back(invalid("run", "this run is quarantined"), table));
            }
            if let Err(error) = self.same_device(stream) {
                return Err(give_back(error, table));
            }
            if self.committed_rows != 0 {
                return Err(give_back(
                    invalid(
                        "page_table",
                        "the mapping cannot change once rows are committed",
                    ),
                    table,
                ));
            }
            let pages = self.geometry.pages;
            if table.is_empty() || table.len() as u64 > pages {
                return Err(give_back(
                    invalid(
                        "page_table",
                        format!(
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
                        invalid(
                            "page_table",
                            format!(
                                "logical page {logical} names physical page {physical} of \
                                 {pages} admitted"
                            ),
                        ),
                        table,
                    ));
                }
                if seen[physical as usize] {
                    return Err(give_back(
                        invalid(
                            "page_table",
                            format!("physical page {physical} is named twice"),
                        ),
                        table,
                    ));
                }
                seen[physical as usize] = true;
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
            self.held = Some(bytes);
            if let Err(error) = self.settle(Ok(()), stream) {
                return Err(PagedRunRefused {
                    error,
                    source: None,
                });
            }
            self.held = None;
            self.page_table = table;
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

        /// Append `rows` dense rows of keys and values at the committed
        /// frontier.
        ///
        /// The frontier advances **only** after the copy's completion event is
        /// observed. A refusal before anything is enqueued leaves it and every
        /// prior byte untouched and hands the source back; a failure after
        /// enqueue quarantines the run, keeps the source, and still does not
        /// advance the frontier. There is no state in between: a partially
        /// copied append is never published as history.
        #[allow(clippy::result_large_err)]
        pub fn append(
            &mut self,
            stream: &Stream<'ctx>,
            rows: u64,
            keys: Vec<u8>,
            values: Vec<u8>,
        ) -> std::result::Result<(), PagedRunRefused> {
            let give_back = |error, keys: Vec<u8>, mut values: Vec<u8>| {
                let mut source = keys;
                source.append(&mut values);
                PagedRunRefused {
                    error,
                    source: Some(source),
                }
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
            if self.page_table.is_empty() {
                return Err(give_back(
                    invalid("page_table", "no page mapping has been published"),
                    keys,
                    values,
                ));
            }
            if rows == 0 {
                return Err(give_back(
                    invalid("rows", "an append of no rows"),
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
                    invalid(
                        "append",
                        format!(
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
            let end = match self.committed_rows.checked_add(rows) {
                Some(end) => end,
                None => {
                    return Err(give_back(
                        Error::Dim(moxie_types::DimError::Overflow),
                        keys,
                        values,
                    ));
                }
            };
            let capacity = match self.geometry.capacity_rows() {
                Ok(capacity) => capacity,
                Err(error) => return Err(give_back(error, keys, values)),
            };
            let mapped = self.page_table.len() as u64 * self.geometry.page_tokens;
            if end > capacity || end > mapped {
                return Err(give_back(
                    Error::CapacityExceeded {
                        tier: Some(Tier::Device(DeviceTier::KvStatePages)),
                        requested_bytes: end,
                        available_bytes: capacity.min(mapped),
                    },
                    keys,
                    values,
                ));
            }

            // Everything above refuses without touching the device. From here
            // on the run holds both sources until completion is observed.
            let mut source = keys;
            let mut values = values;
            source.append(&mut values);
            self.held = Some(source);
            match self.enqueue_append(stream, rows, row_bytes, want) {
                Ok(()) => {}
                Err(error) => {
                    return Err(PagedRunRefused {
                        error,
                        source: None,
                    });
                }
            }
            self.held = None;
            // The frontier moves last, after the event said the bytes are there.
            self.committed_rows = end;
            Ok(())
        }

        fn enqueue_append(
            &mut self,
            stream: &Stream<'ctx>,
            rows: u64,
            row_bytes: u64,
            half: u64,
        ) -> Result<()> {
            let page_tokens = self.geometry.page_tokens;
            let page_bytes = self.geometry.page_bytes()?;
            let source = self.held.as_ref().expect("the append holds its source");
            let first_row = self.committed_rows;
            // One copy per page-aligned run: a dense append crosses page
            // boundaries, and the physical pages it lands on need not be
            // adjacent. The loop is over *storage discontinuities*, not over
            // rows.
            let mut done = 0u64;
            while done < rows {
                let row = first_row + done;
                let logical_page = row / page_tokens;
                let slot = row % page_tokens;
                let run = (page_tokens - slot).min(rows - done);
                let physical = u64::from(self.page_table[logical_page as usize]);
                let within = physical * page_bytes + slot * row_bytes;
                let start = (done * row_bytes) as usize;
                let len = (run * row_bytes) as usize;
                for (range, offset) in [
                    (self.keys.as_ref().expect("live key range"), 0usize),
                    (
                        self.values.as_ref().expect("live value range"),
                        half as usize,
                    ),
                ] {
                    let slice = &source[offset + start..offset + start + len];
                    // SAFETY: the source is held by this run until completion is
                    // observed, and the destination extent is checked against
                    // this run's own admitted range.
                    let copied = unsafe { range.copy_from_host_async_at(within, slice, stream) };
                    if let Err(error) = copied {
                        self.quarantined = true;
                        return Err(self.attribute(error));
                    }
                }
                done += run;
            }
            self.settle(Ok(()), stream)
        }

        /// Read committed rows back out of the pages.
        ///
        /// Evidence, not a data path: it is how a gate checks that an aborted
        /// append left every prior byte exactly as it was.
        pub fn read_rows(&self, first_row: u64, rows: u64) -> Result<Vec<u8>> {
            if self.quarantined {
                return Err(invalid("run", "this run is quarantined"));
            }
            let end = first_row
                .checked_add(rows)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            if end > self.committed_rows {
                return Err(invalid(
                    "read_rows",
                    format!(
                        "rows {first_row}..{end} are not committed; the frontier is {}",
                        self.committed_rows
                    ),
                ));
            }
            let row_bytes = self
                .geometry
                .row_elements()?
                .checked_mul(2)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
            let page_tokens = self.geometry.page_tokens;
            let page_bytes = self.geometry.page_bytes()?;
            let len = usize::try_from(rows * row_bytes * 2)
                .map_err(|_| invalid("read_rows", "the readback exceeds this host's usize"))?;
            let mut out = super::try_zeroed(len)?;
            let half = (rows * row_bytes) as usize;
            let mut done = 0u64;
            while done < rows {
                let row = first_row + done;
                let logical_page = row / page_tokens;
                let slot = row % page_tokens;
                let run = (page_tokens - slot).min(rows - done);
                let physical = u64::from(self.page_table[logical_page as usize]);
                let within = physical * page_bytes + slot * row_bytes;
                let start = (done * row_bytes) as usize;
                let take = (run * row_bytes) as usize;
                self.keys
                    .as_ref()
                    .expect("live key range")
                    .copy_to_host_at(within, &mut out[start..start + take])?;
                self.values
                    .as_ref()
                    .expect("live value range")
                    .copy_to_host_at(within, &mut out[half + start..half + start + take])?;
                done += run;
            }
            Ok(out)
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
                source: Some(query),
            };
            if self.quarantined {
                return Err(give_back(invalid("run", "this run is quarantined"), query));
            }
            if let Err(error) = self.same_device(stream) {
                return Err(give_back(error, query));
            }
            if let Err(error) = super::descriptor_serves(&self.descriptor, launch) {
                return Err(give_back(error, query));
            }
            if launch.geometry != self.geometry
                || launch.heads != self.heads
                || launch.rows > self.max_rows
            {
                return Err(give_back(
                    invalid(
                        "launch",
                        "this launch's geometry is not the one this run was admitted for",
                    ),
                    query,
                ));
            }
            if launch.history_base + launch.history_rows > self.committed_rows {
                return Err(give_back(
                    invalid(
                        "history_rows",
                        format!(
                            "a launch declaring [{}, {}) against a frontier of {}",
                            launch.history_base,
                            launch.history_base + launch.history_rows,
                            self.committed_rows
                        ),
                    ),
                    query,
                ));
            }
            if launch.logical_pages().unwrap_or(u64::MAX) > self.page_table.len() as u64 {
                return Err(give_back(
                    invalid(
                        "page_table",
                        "the published mapping is shorter than the history",
                    ),
                    query,
                ));
            }
            match launch.query_bytes() {
                Ok(want) if query.len() as u64 == want => {}
                Ok(want) => {
                    return Err(give_back(
                        invalid(
                            "query",
                            format!(
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
            self.held = Some(query);
            if let Err(error) = self.enqueue_attend(stream, launch) {
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

        fn enqueue_attend(
            &mut self,
            stream: &Stream<'ctx>,
            launch: &PagedAttentionLaunch,
        ) -> Result<()> {
            let query_source = self.held.as_ref().expect("the attend holds its query");
            let query_range = self.query.as_ref().expect("live query range");
            // SAFETY: the source is held until completion is observed and the
            // destination is this run's own admitted range.
            if let Err(error) = unsafe { query_range.copy_from_host_async(query_source, stream) } {
                self.quarantined = true;
                return Err(self.attribute(error));
            }
            let mut query_address = match query_range.device_address() {
                Ok(address) => address,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            let mut key_address = match self.keys.as_ref().expect("live key range").device_address()
            {
                Ok(address) => address,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            let mut value_address = match self
                .values
                .as_ref()
                .expect("live value range")
                .device_address()
            {
                Ok(address) => address,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            let mut table_address = match self
                .table
                .as_ref()
                .expect("live page table range")
                .device_address()
            {
                Ok(address) => address,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            let mut output_address = match self
                .output
                .as_ref()
                .expect("live output range")
                .device_address()
            {
                Ok(address) => address,
                Err(error) => {
                    self.quarantined = true;
                    return Err(self.attribute(error));
                }
            };
            let mut rows = launch.rows;
            let mut first_position = launch.first_position;
            let mut history_base = launch.history_base;
            let mut history_rows = launch.history_rows;
            let mut heads = match u32::try_from(launch.heads) {
                Ok(heads) => heads,
                Err(_) => {
                    self.quarantined = true;
                    return Err(invalid("heads", "the head count exceeds a u32"));
                }
            };
            let mut kv_heads = match u32::try_from(launch.geometry.kv_heads) {
                Ok(kv) => kv,
                Err(_) => {
                    self.quarantined = true;
                    return Err(invalid(
                        "kv_heads",
                        "the key/value head count exceeds a u32",
                    ));
                }
            };
            let mut head_dim = match u32::try_from(launch.geometry.head_dim) {
                Ok(dim) => dim,
                Err(_) => {
                    self.quarantined = true;
                    return Err(invalid("head_dim", "the head dimension exceeds a u32"));
                }
            };
            let mut page_tokens = match u32::try_from(launch.geometry.page_tokens) {
                Ok(tokens) => tokens,
                Err(_) => {
                    self.quarantined = true;
                    return Err(invalid("page_tokens", "the page width exceeds a u32"));
                }
            };
            let mut window = match launch.window() {
                Ok(window) => window,
                Err(error) => {
                    self.quarantined = true;
                    return Err(error);
                }
            };
            let mut scale = launch.scale;
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
            let grid = match launch.grid() {
                Ok(grid) => grid,
                Err(error) => {
                    self.quarantined = true;
                    return Err(error);
                }
            };
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
                Error::DeviceLost { detail, .. } => Error::DeviceLost {
                    device: self.ctx.ordinal(),
                    detail: format!("kernel {}: {detail}", self.descriptor.id.0),
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
        fn derive(geometry: &PageGeometry, heads: u64, max_rows: u64) -> Result<Self> {
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
            let per_step = query
                .checked_mul(2)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
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
        ctx: &RankContext,
    ) -> Result<PlanRequest> {
        let extents = Extents::derive(geometry, heads, max_rows)?;
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
