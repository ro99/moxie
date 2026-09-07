//! Device images and the kernel catalogue.
//!
//! Document 02: this crate owns "audited unsafe ABI and kernel implementations".
//! It must not perform model registration, placement policy or checkpoint
//! discovery. It exposes *images and descriptors*; loading and launching belong
//! to `moxie-cuda` and, later, to the executor.
//!
//! M0 scope: the smoke kernels only. Real operation kernels arrive with the
//! operations that define them.

#![forbid(unsafe_code)]

use moxie_types::KernelCapability;

/// Fatbin containing every architecture this build targets.
pub const M0_SMOKE_FATBIN: &[u8] = include_bytes!(env!("MOXIE_M0_FATBIN"));

/// Fatbin containing SM86 only.
///
/// Used to prove that loading an image with no binary for the current device
/// fails with a typed `UnsupportedKernel`. Do not "fix" a failure to load this
/// on an SM120 device -- that failure is the assertion.
pub const M0_SMOKE_FATBIN_SM86_ONLY: &[u8] = include_bytes!(env!("MOXIE_M0_FATBIN_SM86"));

/// Compute capabilities compiled into `M0_SMOKE_FATBIN`, as `["86", "120"]`.
pub const KERNEL_ARCHS: &str = env!("MOXIE_KERNEL_ARCHS");

pub const AXPY_F32: &str = "moxie_m0_axpy_f32";
pub const F32_TO_BF16_BITS: &str = "moxie_m0_f32_to_bf16_bits";

/// The architectures this build qualified, as `sm_NN` capability strings.
pub fn qualified_sm() -> Vec<String> {
    KERNEL_ARCHS
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|a| format!("sm_{a}"))
        .collect()
}

/// Descriptor for the smoke kernel, in the same shape a real operation kernel
/// will use: matched on capability and shape, never on a model name.
pub fn axpy_capability() -> KernelCapability {
    KernelCapability {
        operation: AXPY_F32,
        qualified_sm: qualified_sm(),
        workspace_upper_bound_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatbins_are_present_and_non_trivial() {
        // A zero-length image would load as a silent no-op on some drivers.
        assert!(M0_SMOKE_FATBIN.len() > 1024, "fatbin looks empty");
        assert!(M0_SMOKE_FATBIN_SM86_ONLY.len() > 512);
        // The multi-arch image must be the larger of the two: it carries strictly
        // more code. If this ever inverts, the build lost an architecture.
        assert!(
            M0_SMOKE_FATBIN.len() > M0_SMOKE_FATBIN_SM86_ONLY.len(),
            "multi-arch fatbin ({}) is not larger than the sm86-only one ({}); \
             an architecture was probably dropped from the build",
            M0_SMOKE_FATBIN.len(),
            M0_SMOKE_FATBIN_SM86_ONLY.len()
        );
    }

    #[test]
    fn both_product_architectures_are_compiled() {
        // The product claims SM86 (3090) and SM120 (5060 Ti). Losing either from
        // the build must break the host lane, not wait for a GPU run.
        let sm = qualified_sm();
        assert!(sm.contains(&"sm_86".to_string()), "got {sm:?}");
        assert!(sm.contains(&"sm_120".to_string()), "got {sm:?}");
    }

    #[test]
    fn capability_matches_only_qualified_architectures() {
        let cap = axpy_capability();
        assert!(cap.qualified_sm.contains(&"sm_86".to_string()));
        // Nothing claims sm_90: we have no Hopper and never qualified one.
        assert!(!cap.qualified_sm.contains(&"sm_90".to_string()));
    }
}
