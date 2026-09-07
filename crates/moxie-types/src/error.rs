//! Typed errors.
//!
//! Document 02: errors are typed, "not classified by matching CUDA error-message
//! strings". A caller decides what to do from the variant, never from the text.
//! `Unsupported` in particular is a first-class outcome: `required` mode returns
//! it rather than silently falling back, and a test may assert it.

use core::fmt;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A resource request exceeds a tier's physical capacity or admitted budget.
    /// Carries the breakdown so the caller can report legal alternatives rather
    /// than silently shrinking context (document 03).
    CapacityExceeded {
        tier: &'static str,
        requested_bytes: u64,
        available_bytes: u64,
    },
    /// No kernel implements this semantic operation for the given shape, layout,
    /// precision and hardware. Never downgrade silently; `auto` may pick another
    /// candidate, `required` propagates this.
    UnsupportedKernel {
        operation: &'static str,
        detail: String,
    },
    /// A capability was requested in `required` mode and is not available.
    Unsupported {
        capability: &'static str,
        reason: String,
    },
    /// An artifact failed validation: checksum, overlap, truncation, NaN scale,
    /// incompatible dimensions or an unknown required feature (document 03).
    InvalidArtifact { detail: String },
    /// The device was lost or its context is unusable. Not recoverable in place.
    DeviceLost { device: u32, detail: String },
    /// Work was cancelled. Resources must still be released at a safe boundary
    /// even though no next token arrives (R08).
    Cancelled { at: &'static str },
    /// A checked dimension computation overflowed or was inconsistent.
    Dim(crate::dim::DimError),
    /// A numerical execution failure: NaN logits, all-illegal candidate set,
    /// unexpected +inf. Document 05 forbids inheriting the old silent fallback.
    Numerical { detail: String },
    /// A request was malformed. Distinct from `Numerical`, which is an execution
    /// failure (document 05).
    InvalidRequest { field: &'static str, detail: String },
}

impl Error {
    /// Stable machine-readable discriminant, for logs, protocol mapping and
    /// tests. Adding a variant without extending this is a compile error.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::CapacityExceeded { .. } => "capacity_exceeded",
            Error::UnsupportedKernel { .. } => "unsupported_kernel",
            Error::Unsupported { .. } => "unsupported",
            Error::InvalidArtifact { .. } => "invalid_artifact",
            Error::DeviceLost { .. } => "device_lost",
            Error::Cancelled { .. } => "cancelled",
            Error::Dim(_) => "dim",
            Error::Numerical { .. } => "numerical",
            Error::InvalidRequest { .. } => "invalid_request",
        }
    }

    /// Whether retrying the identical request could plausibly succeed.
    /// `CapacityExceeded` is not retryable: the caller must change the request.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Error::Cancelled { .. })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::CapacityExceeded {
                tier,
                requested_bytes,
                available_bytes,
            } => write!(
                f,
                "capacity exceeded on {tier}: requested {requested_bytes} B, {available_bytes} B available"
            ),
            Error::UnsupportedKernel { operation, detail } => {
                write!(f, "no kernel for operation {operation}: {detail}")
            }
            Error::Unsupported { capability, reason } => {
                write!(f, "capability {capability} unavailable: {reason}")
            }
            Error::InvalidArtifact { detail } => write!(f, "invalid artifact: {detail}"),
            Error::DeviceLost { device, detail } => {
                write!(f, "device {device} lost: {detail}")
            }
            Error::Cancelled { at } => write!(f, "cancelled at {at}"),
            Error::Dim(e) => write!(f, "dimension error: {e}"),
            Error::Numerical { detail } => write!(f, "numerical failure: {detail}"),
            Error::InvalidRequest { field, detail } => {
                write!(f, "invalid request field {field}: {detail}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::dim::DimError> for Error {
    fn from(e: crate::dim::DimError) -> Self {
        Error::Dim(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_is_stable_and_distinct() {
        let all = [
            Error::CapacityExceeded {
                tier: "device",
                requested_bytes: 1,
                available_bytes: 0,
            },
            Error::UnsupportedKernel {
                operation: "linear",
                detail: String::new(),
            },
            Error::Unsupported {
                capability: "tp",
                reason: String::new(),
            },
            Error::InvalidArtifact {
                detail: String::new(),
            },
            Error::DeviceLost {
                device: 0,
                detail: String::new(),
            },
            Error::Cancelled { at: "prefill" },
            Error::Dim(crate::dim::DimError::Overflow),
            Error::Numerical {
                detail: String::new(),
            },
            Error::InvalidRequest {
                field: "top_k",
                detail: String::new(),
            },
        ];
        let mut kinds: Vec<_> = all.iter().map(|e| e.kind()).collect();
        kinds.sort_unstable();
        let before = kinds.len();
        kinds.dedup();
        assert_eq!(before, kinds.len(), "error kinds must be distinct");
    }

    #[test]
    fn capacity_exceeded_is_not_retryable() {
        // Retrying an over-budget request unchanged just fails again. The caller
        // has to lower context, branches or precision -- document 03.
        let e = Error::CapacityExceeded {
            tier: "device:1",
            requested_bytes: 1 << 40,
            available_bytes: 1 << 30,
        };
        assert!(!e.is_retryable());
        assert!(Error::Cancelled { at: "decode" }.is_retryable());
    }
}
