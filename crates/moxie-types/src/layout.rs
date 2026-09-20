//! Closed physical tensor layouts shared below planning and execution.
//!
//! A layout is semantic metadata, not a caller-selected integer. Prepared
//! weight layouts continue to use [`crate::LayoutId`] because their identity
//! includes a chunk, device capability and kernel layout version. This enum is
//! the smaller vocabulary needed before any prepared weight exists.

/// Physical layouts whose byte interpretation is fixed by the common API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorLayout {
    /// Dense logical axes in row-major order, with no padding between elements.
    ContiguousRowMajorV1,
}

impl TensorLayout {
    pub const fn name(self) -> &'static str {
        match self {
            TensorLayout::ContiguousRowMajorV1 => "contiguous-row-major-v1",
        }
    }
}

impl core::fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Where one run of paged rows physically sits.
///
/// The vocabulary the state authority speaks and the executor performs. It
/// lives here, in the crate both already depend on, because it is neither a
/// state decision nor an execution effect but the *description* that passes
/// between them — and because the alternative is a dependency edge from the
/// executor to `moxie-state`, which would put retention and frontiers inside
/// the crate that launches kernels.
///
/// A run rather than a row: a dense append crosses page boundaries, and the
/// physical pages it lands on need not be adjacent, so the unit that matters is
/// the stretch of rows sharing one page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagePlacement {
    /// The absolute position of this run's first row.
    pub position: u64,
    /// The physical page it lands on.
    pub physical_page: u64,
    /// The slot within that page where it starts.
    pub slot: u64,
    /// How many rows this run covers.
    pub rows: u64,
}

impl PagePlacement {
    /// One past this run's last position.
    pub fn end(&self) -> Option<u64> {
        self.position.checked_add(self.rows)
    }
}

/// Which staged batch a write belongs to.
///
/// The three numbers that already identify a batch inside one sequence: the
/// transaction that staged it and the rows it reserved. Carried as one value so
/// that a performer names the batch it copied rather than describing it again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchId {
    pub transaction: crate::StateTransactionId,
    pub first: u64,
    pub rows: u64,
}

/// One layer's logical-to-physical page mapping, and the absolute row its first
/// logical page names.
///
/// A table without its base is not a mapping: the base is what makes the entries
/// describe *absolute* rows, so a write checked against one base and a launch
/// reading against another would disagree about the same page. They travel as one
/// value for that reason.
///
/// It lives here for the reason [`PagePlacement`] does. What a page table contains,
/// and when it slides, is the state authority's decision; uploading it is the
/// performer's effect. This is the description that passes between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageView {
    /// The absolute row logical page zero names.
    pub base: u64,
    /// `table[logical] = physical`.
    pub table: Vec<u32>,
}

/// The performer a sequence drives to put one layer's staged rows on a device.
///
/// The authority owns staging, placement and publication; the performer owns the
/// copy and the completion it observes. Neither crate may name the other, so the
/// authority calls *through* this trait: it stages, it hands each layer's writer
/// the placements it chose, and it publishes only if every writer returned `Ok`.
/// There is no value a caller can hold that makes publication happen — the only
/// way to reach it is to be the writer and to return success.
pub trait PagedKvWriter {
    /// Copy this batch's rows into `placements` and observe the copy complete.
    ///
    /// `Ok` is a statement that the bytes are on the device and that the copy is
    /// no longer in flight. Returning it before completion is observed is the one
    /// way an implementation can lie, and it is the implementation's contract to
    /// keep — a sequence has no other evidence.
    ///
    /// `view` is the authority's own mapping for this layer, not one the performer
    /// derives: what the table contains and when it slides are state decisions, so a
    /// performer that computed its own would be the second authority this boundary
    /// exists to prevent.
    ///
    /// The rows themselves are the writer's: it was constructed with them, so a
    /// refusal returns them or retains them by its own rules rather than handing
    /// a half-copied buffer back through this signature.
    fn write_layer(
        &mut self,
        layer: usize,
        batch: BatchId,
        view: &PageView,
        placements: &[PagePlacement],
    ) -> crate::Result<()>;
}
