//! Device images and the kernel catalogue.
//!
//! Document 02: this crate owns "audited unsafe ABI and kernel implementations".
//! It must not perform model registration, placement policy or checkpoint
//! discovery. It exposes *images and descriptors*; loading and launching belong
//! to `moxie-cuda` and, later, to the executor.
//!
//! Scope: toolchain smoke kernels plus the narrowly qualified BF16 Linear,
//! RMSNorm and Residual package. More operations arrive only with their own
//! semantic and numerical contracts.
//!
//! The images and their build identity are behind the **`fatbin`** feature,
//! which is off by default: document 07 requires the host lane to build with no
//! CUDA toolkit. What stays available without it is the *declared* target list,
//! which is a support-matrix fact rather than a build artifact.
//!
//! Naming: identifiers here describe what the code *is*, never which milestone
//! produced it. `smoke` says "toolchain and launch probe, not a product kernel"
//! and stays true forever; a milestone tag stops meaning anything the moment the
//! milestone closes, and `moxie_smoke_axpy_f32` is a CUDA symbol baked into the
//! fatbin ABI, so renaming it later breaks every lookup by name.

#![forbid(unsafe_code)]

pub mod cpu_expert;

/// Compute capabilities this build *targets*, as `["86", "120"]`.
///
/// Declared, not measured. Targeting an architecture is not compiling for it
/// (that needs the `fatbin` feature) and compiling for it is not qualifying it
/// (that needs a passing `test-gpu` run on real hardware). Three different
/// claims; document 07 keeps them separate and so does this crate.
pub const TARGET_ARCHS: &str = env!("MOXIE_KERNEL_TARGET_ARCHS");

/// `sm_NN` strings for [`TARGET_ARCHS`].
pub fn target_sm() -> Vec<String> {
    split_sm(TARGET_ARCHS)
}

fn split_sm(list: &str) -> Vec<String> {
    list.split(',')
        .filter(|s| !s.is_empty())
        .map(|a| format!("sm_{a}"))
        .collect()
}

pub const AXPY_F32: &str = "moxie_smoke_axpy_f32";
pub const F32_TO_BF16_BITS: &str = "moxie_smoke_f32_to_bf16_bits";
pub const BF16_LINEAR: &str = "moxie_bf16_linear_v1";
pub const BF16_RMS_SUM: &str = "moxie_bf16_rms_sum_v1";
pub const BF16_RMS_APPLY: &str = "moxie_bf16_rms_apply_v1";
pub const BF16_RESIDUAL: &str = "moxie_bf16_residual_v1";
pub const BF16_CHAIN_ABI: u32 = 1;

/// Task 0021's grouped expert package. Two projection symbols, one per gate
/// transform, because they are two operations rather than one with a flag.
pub const BF16_EXPERT_PROJECT_GELU: &str = "moxie_bf16_expert_project_gelu_v1";
pub const BF16_EXPERT_PROJECT_SILU: &str = "moxie_bf16_expert_project_silu_v1";
pub const BF16_EXPERT_DOWN: &str = "moxie_bf16_expert_down_v1";
pub const BF16_EXPERT_ABI: u32 = 1;

#[cfg(feature = "fatbin")]
mod images {
    use moxie_types::{
        AccumulationPolicy, ActivationPrecision, GateTransform, KernelCapability, KernelCatalogue,
        KernelId, KernelOperand, KernelShapeBounds, KernelSymbol, Precision, RoundingProfile,
        SemanticKernelDescriptor, SemanticKernelOp, SmVersion, TensorLayout, WeightPrecision,
        WorkspaceExpression,
    };

    /// Fatbin containing every architecture this build compiled.
    ///
    /// These bytes are `nvcc` output embedded at build time. That provenance is
    /// what a caller asserts when it wraps them in `moxie_cuda::TrustedImage`;
    /// `cuModuleLoadData` receives no length and cannot check them.
    pub const SMOKE_FATBIN: &[u8] = include_bytes!(env!("MOXIE_SMOKE_FATBIN"));

    /// Fatbin containing SM86 only.
    ///
    /// Used to prove that loading an image with no binary for the current device
    /// fails with a typed `UnsupportedKernel`. Do not "fix" a failure to load
    /// this on an SM120 device -- that failure is the assertion. It is SASS-only
    /// for the same reason: embedded PTX would let the driver JIT it anywhere.
    pub const SMOKE_FATBIN_SM86_ONLY: &[u8] = include_bytes!(env!("MOXIE_SMOKE_FATBIN_SM86"));
    pub const BF16_CHAIN_FATBIN: &[u8] = include_bytes!(env!("MOXIE_BF16_CHAIN_FATBIN"));
    pub const EXPERT_MLP_FATBIN: &[u8] = include_bytes!(env!("MOXIE_EXPERT_MLP_FATBIN"));

    /// Compute capabilities actually compiled into [`SMOKE_FATBIN`].
    pub const KERNEL_ARCHS: &str = env!("MOXIE_KERNEL_ARCHS");

    /// Build identity, for the benchmark manifest document 07 requires.
    pub const SMOKE_FATBIN_SHA256: &str = env!("MOXIE_SMOKE_FATBIN_SHA256");
    pub const SMOKE_FATBIN_SM86_SHA256: &str = env!("MOXIE_SMOKE_FATBIN_SM86_SHA256");
    pub const BF16_CHAIN_FATBIN_SHA256: &str = env!("MOXIE_BF16_CHAIN_FATBIN_SHA256");
    pub const EXPERT_MLP_FATBIN_SHA256: &str = env!("MOXIE_EXPERT_MLP_FATBIN_SHA256");
    /// What `nvcc --version` reported, verified against the pin in `build.rs`.
    pub const NVCC_VERSION: &str = env!("MOXIE_NVCC_VERSION");
    /// The host compiler nvcc drove. Recorded, not pinned.
    pub const HOST_COMPILER_VERSION: &str = env!("MOXIE_HOST_COMPILER_VERSION");

    /// The architectures compiled into the image, as `sm_NN` strings.
    ///
    /// Compiled, **not qualified**. A capability is qualified by a passing
    /// `test-gpu` case on a real device of that architecture; this function only
    /// reports what nvcc emitted.
    pub fn compiled_sm() -> Vec<String> {
        super::split_sm(KERNEL_ARCHS)
    }

    /// Descriptor for the smoke kernel, in the same shape a real operation
    /// kernel will use: matched on capability and shape, never on a model name.
    ///
    /// `qualified_sm` is populated from the compiled list because the smoke
    /// kernel is architecture-agnostic C. A real kernel's descriptor must be
    /// populated from the gate IDs that actually passed (document 07), not from
    /// what the compiler accepted.
    pub fn axpy_capability() -> KernelCapability {
        KernelCapability {
            operation: super::AXPY_F32,
            qualified_sm: compiled_sm(),
            workspace_upper_bound_bytes: 0,
        }
    }

    /// The closed task-0012 package. Each SM has a distinct identity so
    /// qualification cannot leak from Ampere to Blackwell or vice versa.
    pub fn bf16_chain_catalogue() -> KernelCatalogue {
        let hash = parse_sha256(BF16_CHAIN_FATBIN_SHA256);
        let mut descriptors = Vec::new();
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            let suffix = sm.name();
            descriptors.push(descriptor(
                format!("bf16-linear-v1-{suffix}"),
                SemanticKernelOp::Linear,
                vec![
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
                ],
                sm,
                WorkspaceExpression::Zero,
                hash,
                &[super::BF16_LINEAR],
            ));
            descriptors.push(descriptor(
                format!("bf16-rms-norm-v1-{suffix}"),
                SemanticKernelOp::RmsNorm,
                vec![
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
                ],
                sm,
                WorkspaceExpression::RowsTimesF32,
                hash,
                &[super::BF16_RMS_SUM, super::BF16_RMS_APPLY],
            ));
            descriptors.push(descriptor(
                format!("bf16-residual-v1-{suffix}"),
                SemanticKernelOp::Residual,
                vec![
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                ],
                sm,
                WorkspaceExpression::Zero,
                hash,
                &[super::BF16_RESIDUAL],
            ));
        }
        KernelCatalogue::new(descriptors).expect("built-in descriptors are unique")
    }

    /// Task 0021's grouped expert package, one descriptor per (gate transform,
    /// SM). Separate identities per SM so qualification cannot leak from Ampere
    /// to Blackwell, exactly as the chain package does.
    ///
    /// The shape bounds are the operation's, not the chain's: an expert's
    /// reduction runs over `hidden` and its output is `hidden` wide, and the
    /// designated artifact's layer is 2,816 by 704.
    pub fn expert_mlp_catalogue() -> KernelCatalogue {
        let hash = parse_sha256(EXPERT_MLP_FATBIN_SHA256);
        let mut descriptors = Vec::new();
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            for (gate, project) in [
                (GateTransform::GeluTanh, super::BF16_EXPERT_PROJECT_GELU),
                (GateTransform::Silu, super::BF16_EXPERT_PROJECT_SILU),
            ] {
                descriptors.push(SemanticKernelDescriptor {
                    id: KernelId(format!("bf16-expert-mlp-v1-{}-{}", gate.name(), sm.name())),
                    abi_version: super::BF16_EXPERT_ABI,
                    operation: SemanticKernelOp::ExpertMlp(gate),
                    inputs: vec![
                        KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                        KernelOperand::RouteIndex,
                        KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
                        KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
                    ],
                    output: ActivationPrecision::expect(Precision::Bf16),
                    accumulation: AccumulationPolicy::Bf16InF32Acc,
                    rounding: RoundingProfile::FinalBf16Rne,
                    layout: TensorLayout::ContiguousRowMajorV1,
                    shape: KernelShapeBounds {
                        max_rows: 65_536,
                        max_input: 16_384,
                        max_output: 16_384,
                    },
                    sm,
                    workspace: WorkspaceExpression::RowsTimesIntermediateF32,
                    image_sha256: hash,
                    symbols: vec![
                        KernelSymbol(project.to_string()),
                        KernelSymbol(super::BF16_EXPERT_DOWN.to_string()),
                    ],
                });
            }
        }
        KernelCatalogue::new(descriptors).expect("built-in descriptors are unique")
    }

    fn descriptor(
        id: String,
        operation: SemanticKernelOp,
        inputs: Vec<KernelOperand>,
        sm: SmVersion,
        workspace: WorkspaceExpression,
        image_sha256: [u8; 32],
        symbols: &[&str],
    ) -> SemanticKernelDescriptor {
        SemanticKernelDescriptor {
            id: KernelId(id),
            abi_version: super::BF16_CHAIN_ABI,
            operation,
            inputs,
            output: ActivationPrecision::expect(Precision::Bf16),
            accumulation: AccumulationPolicy::Bf16InF32Acc,
            rounding: RoundingProfile::FinalBf16Rne,
            layout: TensorLayout::ContiguousRowMajorV1,
            shape: KernelShapeBounds {
                max_rows: 64,
                max_input: 1024,
                max_output: 1024,
            },
            sm,
            workspace,
            image_sha256,
            symbols: symbols
                .iter()
                .map(|name| KernelSymbol((*name).to_string()))
                .collect(),
        }
    }

    fn parse_sha256(value: &str) -> [u8; 32] {
        assert_eq!(value.len(), 64, "build emitted malformed SHA-256");
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[i * 2..i * 2 + 2], 16)
                .expect("build emitted non-hex SHA-256");
        }
        out
    }
}

#[cfg(feature = "fatbin")]
pub use images::{
    BF16_CHAIN_FATBIN, BF16_CHAIN_FATBIN_SHA256, EXPERT_MLP_FATBIN, EXPERT_MLP_FATBIN_SHA256,
    HOST_COMPILER_VERSION, KERNEL_ARCHS, NVCC_VERSION, SMOKE_FATBIN, SMOKE_FATBIN_SHA256,
    SMOKE_FATBIN_SM86_ONLY, SMOKE_FATBIN_SM86_SHA256, axpy_capability, bf16_chain_catalogue,
    compiled_sm, expert_mlp_catalogue,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_product_architectures_are_targeted() {
        // The product claims SM86 (3090) and SM120 (5060 Ti). Losing either from
        // the target list must break the host lane, not wait for a GPU run.
        let sm = target_sm();
        assert!(sm.contains(&"sm_86".to_string()), "got {sm:?}");
        assert!(sm.contains(&"sm_120".to_string()), "got {sm:?}");
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn fatbins_are_present_and_non_trivial() {
        // A zero-length image would load as a silent no-op on some drivers.
        assert!(SMOKE_FATBIN.len() > 1024, "fatbin looks empty");
        assert!(SMOKE_FATBIN_SM86_ONLY.len() > 512);
        assert!(BF16_CHAIN_FATBIN.len() > 1024);
        // The multi-arch image must be the larger of the two: it carries strictly
        // more code. If this ever inverts, the build lost an architecture.
        assert!(
            SMOKE_FATBIN.len() > SMOKE_FATBIN_SM86_ONLY.len(),
            "multi-arch fatbin ({}) is not larger than the sm86-only one ({}); \
             an architecture was probably dropped from the build",
            SMOKE_FATBIN.len(),
            SMOKE_FATBIN_SM86_ONLY.len()
        );
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn every_targeted_architecture_was_compiled() {
        // The declared target list and what nvcc actually emitted must agree.
        // A silent divergence is how a support-matrix claim outlives its build.
        assert_eq!(compiled_sm(), target_sm());
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn build_identity_is_recorded() {
        // Document 07 requires a recorded executable/image identity.
        assert_eq!(SMOKE_FATBIN_SHA256.len(), 64, "{SMOKE_FATBIN_SHA256}");
        assert_eq!(SMOKE_FATBIN_SM86_SHA256.len(), 64);
        assert_ne!(SMOKE_FATBIN_SHA256, SMOKE_FATBIN_SM86_SHA256);
        assert_eq!(BF16_CHAIN_FATBIN_SHA256.len(), 64);
        assert!(NVCC_VERSION.contains("13.0"), "{NVCC_VERSION}");
        assert!(!HOST_COMPILER_VERSION.is_empty());
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn capability_matches_only_compiled_architectures() {
        let cap = axpy_capability();
        assert!(cap.qualified_sm.contains(&"sm_86".to_string()));
        // Nothing claims sm_90: we have no Hopper and never qualified one.
        assert!(!cap.qualified_sm.contains(&"sm_90".to_string()));
    }
}
