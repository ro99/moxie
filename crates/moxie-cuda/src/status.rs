//! Driver status classification, with no link dependency on `libcuda`.
//!
//! Document 02: "Errors are typed ... not classified by matching CUDA
//! error-message strings." The mapping from a numeric `CUresult` to a typed
//! error is pure arithmetic over an ABI-stable code space, so it lives here,
//! outside the `driver` feature.
//!
//! Document 07 requires host CI to run "without a checkpoint, NVIDIA driver
//! library or CUDA toolkit". Keeping this module driver-free is what lets the
//! error-mapping tests run on a machine that has no `libcuda` at all: the
//! earlier version called `cuGetErrorString` to build the message, so every
//! test of the classification linked the driver.

use moxie_types::{Error, Result};

/// The driver's result code type. Declared here rather than in `ffi` so that the
/// classification, and its tests, compile without the driver feature.
pub type CUresult = core::ffi::c_int;

pub const CUDA_SUCCESS: CUresult = 0;

/// Map a driver result code to a typed error.
///
/// `detail` is human-facing context only. The variant is chosen from `code`
/// alone: no branch below inspects the text, and none may.
///
/// Codes are from `cuda.h` for the toolkit pinned in
/// `docs/evidence/toolchain.md`, grouped by what a caller must do about them.
pub fn classify(code: CUresult, detail: String) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    match code {
        // CUDA_ERROR_OUT_OF_MEMORY. The byte counts and the tier are filled in
        // by the allocator, which knows what it asked for and what for. This
        // classifier sees a driver code and nothing else, so it attributes
        // nothing rather than guessing a tier.
        2 => Err(Error::CapacityExceeded {
            tier: None,
            requested_bytes: 0,
            available_bytes: 0,
        }),
        // DEINITIALIZED, LAUNCH_FAILED, ILLEGAL_ADDRESS, CONTEXT_IS_DESTROYED,
        // ECC_UNCORRECTABLE, HARDWARE_STACK_ERROR: the context is unusable.
        4 | 700 | 709 | 719 | 214 | 714 => Err(Error::DeviceLost {
            device: u32::MAX,
            detail,
        }),
        // NOT_INITIALIZED. This is our defect, not the machine's: the driver
        // API was called before `cuInit`. Kept distinct so it can never be
        // misreported as absent hardware.
        3 => Err(Error::InvalidRequest {
            field: "cuda_init",
            detail,
        }),
        // NO_DEVICE, INVALID_DEVICE
        100 | 101 => Err(Error::Unsupported {
            capability: "cuda_device",
            reason: detail,
        }),
        // NOT_FOUND, INVALID_PTX, UNSUPPORTED_PTX_VERSION, NO_BINARY_FOR_GPU,
        // INVALID_IMAGE, INVALID_SOURCE. The last two are what a truncated or
        // non-image byte buffer produces, and they belong with the other
        // module-load failures rather than in the numerical catch-all.
        500 | 218 | 222 | 209 | 200 | 300 => Err(Error::UnsupportedKernel {
            operation: "module",
            detail,
        }),
        _ => Err(Error::Numerical { detail }),
    }
}

/// Format a device UUID as the canonical `GPU-...` string.
///
/// Pure, and tested without a device: the driver hands back sixteen raw bytes
/// and the grouping below is ours, so a transposition here would silently
/// rename every GPU in every evidence record.
pub fn format_uuid(b: &[u8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        b[8],
        b[9],
        b[10],
        b[11],
        b[12],
        b[13],
        b[14],
        b[15]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // These run on the host lane, with no driver present. Real device behaviour
    // is `cargo xtask-cuda test-gpu`.

    #[test]
    fn success_is_not_an_error() {
        assert!(classify(CUDA_SUCCESS, "ctx".into()).is_ok());
    }

    #[test]
    fn out_of_memory_maps_to_capacity_exceeded() {
        let e = classify(2, "cuMemAlloc".into()).unwrap_err();
        assert_eq!(e.kind(), "capacity_exceeded");
        assert!(!e.is_retryable());
    }

    #[test]
    fn missing_binary_for_gpu_maps_to_unsupported_kernel() {
        // NO_BINARY_FOR_GPU is what a fatbin returns when it has no image for
        // the current architecture -- directly an M0 exit-gate case.
        for code in [209, 218, 222, 500] {
            assert_eq!(
                classify(code, "cuModuleLoadData".into())
                    .unwrap_err()
                    .kind(),
                "unsupported_kernel",
                "code {code}"
            );
        }
    }

    #[test]
    fn a_rejected_image_is_a_module_error_not_a_numerical_one() {
        // INVALID_IMAGE / INVALID_SOURCE are what the driver returns for a byte
        // buffer that is not a supported image. A caller that loads an image
        // must see the same error kind whichever way the driver refuses it.
        for code in [200, 300] {
            assert_eq!(
                classify(code, "cuModuleLoadData".into())
                    .unwrap_err()
                    .kind(),
                "unsupported_kernel",
                "code {code}"
            );
        }
    }

    #[test]
    fn fatal_device_codes_map_to_device_lost() {
        for code in [4, 700, 709, 719, 214, 714] {
            assert_eq!(
                classify(code, "ctx".into()).unwrap_err().kind(),
                "device_lost",
                "code {code}"
            );
        }
    }

    #[test]
    fn uninitialised_driver_is_not_reported_as_missing_hardware() {
        // A wrapper defect and an empty machine are different problems and must
        // not share an error kind.
        let e = classify(3, "cuDeviceGetCount".into()).unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        assert_ne!(e.kind(), classify(100, "cuInit".into()).unwrap_err().kind());
    }

    #[test]
    fn no_device_maps_to_unsupported_not_to_a_panic() {
        assert_eq!(
            classify(100, "cuInit".into()).unwrap_err().kind(),
            "unsupported"
        );
        assert_eq!(
            classify(101, "cuDeviceGet".into()).unwrap_err().kind(),
            "unsupported"
        );
    }

    #[test]
    fn variant_comes_from_the_code_not_the_message() {
        match classify(2, "alloc".into()).unwrap_err() {
            Error::CapacityExceeded { tier, .. } => assert_eq!(
                tier, None,
                "the driver code carries no tier; only the allocator can attribute one"
            ),
            other => panic!("wrong variant: {other:?}"),
        }
        // Same text, different code, different variant: proof the message is
        // not consulted.
        assert_ne!(
            classify(2, "alloc".into()).unwrap_err().kind(),
            classify(700, "alloc".into()).unwrap_err().kind()
        );
    }

    #[test]
    fn uuid_formats_as_the_canonical_gpu_string() {
        let bytes = [
            0x97u8, 0xfe, 0x48, 0x89, 0x48, 0x74, 0xa3, 0x78, 0x19, 0x8e, 0x95, 0x5d, 0x2e, 0x72,
            0xc3, 0xa3,
        ];
        assert_eq!(
            format_uuid(&bytes),
            "GPU-97fe4889-4874-a378-198e-955d2e72c3a3"
        );
    }
}
