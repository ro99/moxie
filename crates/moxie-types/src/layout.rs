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
