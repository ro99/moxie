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
