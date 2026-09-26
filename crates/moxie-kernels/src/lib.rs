//! Device images and the kernel catalogue.
//!
//! Document 02: this crate owns "audited unsafe ABI and kernel implementations".
//! It must not perform model registration, placement policy or checkpoint
//! discovery. It exposes *images and descriptors*; loading and launching belong
//! to `moxie-cuda` and, later, to the executor.
//!
//! Scope: toolchain smoke kernels plus the qualified BF16 operation packages.
//! Each package has its own semantic and numerical contract.
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

pub mod affine;
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
pub const DENSE_LINEAR_SPLIT: &str = "moxie_dense_linear_split_v1";
pub const DENSE_LINEAR_PARTIAL: &str = "moxie_dense_linear_partial_v1";
pub const TP_REDUCE_F32: &str = "moxie_tp_reduce_f32_v1";
pub const TP_F32_TO_BF16: &str = "moxie_tp_f32_to_bf16_v1";
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

/// Task 0028's shared quantized dense linear. **One symbol for both widths.**
///
/// W4A16 and W8A16 are two catalogue identities over the same code, because the
/// only thing that differs between them is how a code is read out of a packed
/// row. The group rule, the zero-point section and the scale encoding are
/// runtime parameters. Two symbols here would drift into two implementations,
/// and ADR 0003 is explicit that "W4A16 is an execution profile, AWQ/AutoRound
/// are methods": neither is a licence for a kernel per source.
pub const AFFINE_LINEAR: &str = "moxie_affine_linear_v2";
pub const AFFINE_LINEAR_ABI: u32 = 2;
/// The output tile the kernel's warp computes, and the tensor-core `k` step.
///
/// Public because it is a *contract*, not an implementation detail: the host
/// checks that a group boundary never falls inside a `k` tile before it
/// launches, and that check needs this number.
pub const AFFINE_LINEAR_TILE: u64 = 16;

/// Task 0037's common paged attention. **One symbol for every calling pattern.**
///
/// Whole prefill, a prefill chunk and a single decode row are the same
/// operation over different row counts, and multi-head attention is grouped
/// attention with one query head per group. Splitting any of those into its own
/// symbol would make the shapes that share a kernel today diverge tomorrow,
/// which is the failure document 02 names. Visibility, head counts, head
/// dimension, page width and the score scale are runtime parameters.
pub const PAGED_ATTENTION: &str = "moxie_bf16_paged_attention_v1";
pub const PAGED_ATTENTION_INDIRECT: &str = "moxie_bf16_paged_attention_indirect_v1";
pub const KV_APPEND_INDIRECT: &str = "moxie_kv_append_indirect_v1";
pub const PAGED_ATTENTION_ABI: u32 = 1;
/// The separately qualified partial-producing entry point used by the bounded
/// two-block host-backed path. Its ABI is intentionally not the single-shot
/// output ABI above.
pub const PAGED_ATTENTION_PARTIAL: &str = "moxie_bf16_paged_attention_partial_v1";
/// Keys the kernel scores, maximizes and folds into its running partial before
/// it looks at the next ones.
///
/// Public because it is a *contract* rather than an implementation detail: it
/// is the block width of the online softmax the host oracle pins, and a host
/// checking that a tile spans a page boundary needs the number. It is
/// deliberately not the page width — pages are storage, tiles are scheduling,
/// and a tile that matched the page would never exercise a tail.
pub const PAGED_ATTENTION_TILE: u64 = 128;
/// Threads per attention block, which is also the number of output components
/// one block writes per pass.
pub const PAGED_ATTENTION_THREADS: u32 = 128;
/// The widest head dimension the image serves. The host refuses more rather
/// than letting the kernel read past a row. Task 0105: raised to Gemma 31B's
/// global-layer head dimension; the paged-attention catalogue, the host
/// admission guard and the refusal tests all read this constant rather than
/// restating 256, so this is the only edit their bound needed.
pub const PAGED_ATTENTION_MAX_HEAD_DIM: u64 = 512;

pub const DENSE_EMBEDDING: &str = "moxie_dense_embedding_v1";
pub const DENSE_GROUPED_RMS: &str = "moxie_dense_grouped_rms_v1";
pub const DENSE_ROPE: &str = "moxie_dense_rope_v1";
pub const DENSE_GEGLU: &str = "moxie_dense_geglu_v1";
pub const DENSE_RESIDUAL_SCALED: &str = "moxie_dense_residual_scaled_v1";
pub const DENSE_VOCAB_PROJECTION: &str = "moxie_dense_vocab_projection_v1";
pub const DENSE_ROUTE: &str = "moxie_dense_route_v1";
pub const DENSE_EXPERT_PROJECT_GELU: &str = "moxie_dense_expert_project_gelu_v1";
pub const DENSE_EXPERT_DOWN: &str = "moxie_dense_expert_down_v1";
pub const DENSE_COMBINE: &str = "moxie_dense_combine_v1";
pub const DENSE_COMBINE_PARTIAL: &str = "moxie_dense_combine_partial_v1";
pub const DENSE_GRAPH_ABI: u32 = 1;

/// The catalogue identity this build publishes for one architecture.
///
/// A `&'static str` per SM rather than a formatted name, because
/// [`paged_attention_declares`] must answer without allocating. Gated with the
/// images it identifies: without them there is no package to be a member of.
#[cfg(feature = "fatbin")]
const fn paged_attention_id(sm: moxie_types::SmVersion) -> Option<&'static str> {
    match (sm.major, sm.minor) {
        (8, 6) => Some("bf16-paged-attention-v1-sm_86"),
        (12, 0) => Some("bf16-paged-attention-v1-sm_120"),
        _ => None,
    }
}

/// Whether this descriptor is one this build's paged attention package declares.
///
/// **Allocation-free, and that is the point.** The binding asks this question at
/// admission, on the path that must produce a typed refusal under memory
/// pressure; building a catalogue to answer it would allocate a `Vec`, two
/// `String`s per descriptor and a `format!` for each id, any of which can abort
/// instead of refusing (task 0019's rule). Every field the catalogue sets is
/// compared here — including the ones nothing else checks, which is the whole
/// reason the question is asked: a descriptor declaring another layout,
/// accumulation policy, rounding profile, workspace or image would otherwise be
/// executed by this build's own code with its declaration ignored.
///
/// `the_package_predicate_and_the_catalogue_agree` pins the two together, so
/// this cannot drift from what [`images::paged_attention_catalogue`] builds.
#[cfg(feature = "fatbin")]
pub fn paged_attention_declares(descriptor: &moxie_types::SemanticKernelDescriptor) -> bool {
    use moxie_types::{
        AccumulationPolicy, ActivationPrecision, KernelOperand, Precision, RoundingProfile,
        SemanticKernelOp, TensorLayout, WorkspaceExpression,
    };
    let Some(id) = paged_attention_id(descriptor.sm) else {
        return false;
    };
    let bf16 = KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16));
    descriptor.id.0 == id
        && descriptor.abi_version == PAGED_ATTENTION_ABI
        && descriptor.operation == SemanticKernelOp::PagedAttention
        && descriptor.inputs.as_slice() == [bf16, bf16, bf16, KernelOperand::PageIndex]
        && descriptor.output == ActivationPrecision::expect(Precision::Bf16)
        && descriptor.accumulation == AccumulationPolicy::Bf16InF32Acc
        && descriptor.rounding == RoundingProfile::FinalBf16Rne
        && descriptor.layout == TensorLayout::ContiguousRowMajorV1
        && descriptor.shape.max_rows == 65_536
        && descriptor.shape.max_input == PAGED_ATTENTION_MAX_HEAD_DIM
        && descriptor.shape.max_output == PAGED_ATTENTION_MAX_HEAD_DIM
        && descriptor.workspace == WorkspaceExpression::Zero
        && descriptor.image_sha256 == images::paged_attention_sha256()
        && descriptor.symbols.len() == 1
        && descriptor.symbols[0].0 == PAGED_ATTENTION
}

#[cfg(feature = "fatbin")]
mod images {
    use super::{
        BF16_LINEAR, BF16_RESIDUAL, BF16_RMS_APPLY, BF16_RMS_SUM, DENSE_COMBINE,
        DENSE_COMBINE_PARTIAL, DENSE_EMBEDDING, DENSE_EXPERT_DOWN, DENSE_EXPERT_PROJECT_GELU,
        DENSE_GEGLU, DENSE_GRAPH_ABI, DENSE_GROUPED_RMS, DENSE_LINEAR_PARTIAL, DENSE_LINEAR_SPLIT,
        DENSE_RESIDUAL_SCALED, DENSE_ROPE, DENSE_ROUTE, DENSE_VOCAB_PROJECTION, TP_REDUCE_F32,
    };
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
    pub const AFFINE_LINEAR_FATBIN: &[u8] = include_bytes!(env!("MOXIE_AFFINE_LINEAR_FATBIN"));
    pub const PAGED_ATTENTION_FATBIN: &[u8] = include_bytes!(env!("MOXIE_PAGED_ATTENTION_FATBIN"));
    pub const DENSE_GRAPH_FATBIN: &[u8] = include_bytes!(env!("MOXIE_DENSE_GRAPH_FATBIN"));

    /// Compute capabilities actually compiled into [`SMOKE_FATBIN`].
    pub const KERNEL_ARCHS: &str = env!("MOXIE_KERNEL_ARCHS");

    /// Build identity, for the benchmark manifest document 07 requires.
    pub const SMOKE_FATBIN_SHA256: &str = env!("MOXIE_SMOKE_FATBIN_SHA256");
    pub const SMOKE_FATBIN_SM86_SHA256: &str = env!("MOXIE_SMOKE_FATBIN_SM86_SHA256");
    pub const BF16_CHAIN_FATBIN_SHA256: &str = env!("MOXIE_BF16_CHAIN_FATBIN_SHA256");
    pub const EXPERT_MLP_FATBIN_SHA256: &str = env!("MOXIE_EXPERT_MLP_FATBIN_SHA256");
    pub const AFFINE_LINEAR_FATBIN_SHA256: &str = env!("MOXIE_AFFINE_LINEAR_FATBIN_SHA256");
    pub const PAGED_ATTENTION_FATBIN_SHA256: &str = env!("MOXIE_PAGED_ATTENTION_FATBIN_SHA256");
    pub const DENSE_GRAPH_FATBIN_SHA256: &str = env!("MOXIE_DENSE_GRAPH_FATBIN_SHA256");
    #[cfg(feature = "cublas")]
    pub const CUBLAS_SO_SHA256: &str = env!("MOXIE_CUBLAS_SO_SHA256");
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
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            for gate_width in [Precision::Bf16, Precision::Int4, Precision::Int8] {
                for down_width in [Precision::Bf16, Precision::Int4, Precision::Int8] {
                    if gate_width == Precision::Bf16 && down_width == Precision::Bf16 {
                        continue;
                    }
                    for (gate, project) in [
                        (
                            GateTransform::GeluTanh,
                            "moxie_affine_expert_project_gelu_v1",
                        ),
                        (GateTransform::Silu, "moxie_affine_expert_project_silu_v1"),
                    ] {
                        descriptors.push(SemanticKernelDescriptor {
                            id: KernelId(format!(
                                "affine-expert-{}-{}-{}-{}",
                                gate_width.name(),
                                down_width.name(),
                                gate.name(),
                                sm.name()
                            )),
                            abi_version: super::BF16_EXPERT_ABI,
                            operation: SemanticKernelOp::ExpertMlp(gate),
                            inputs: vec![
                                KernelOperand::Activation(ActivationPrecision::expect(
                                    Precision::Bf16,
                                )),
                                KernelOperand::RouteIndex,
                                KernelOperand::Weight(WeightPrecision::expect(gate_width)),
                                KernelOperand::Weight(WeightPrecision::expect(down_width)),
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
                                KernelSymbol("moxie_affine_expert_down_v1".to_string()),
                            ],
                        });
                    }
                }
            }
        }
        KernelCatalogue::new(descriptors).expect("built-in descriptors are unique")
    }

    /// Task 0028's W4A16 / W8A16 dense linear, one descriptor per (width, SM).
    ///
    /// The **weight operand precision** is what separates a W4A16 entry from a
    /// W8A16 one and from the BF16 linear in the chain package: selection is by
    /// semantic capability and operand precision, never by which checkpoint the
    /// tensor came from (document 02). All four name the same symbol, so a
    /// catalogue split can never become an implementation split.
    ///
    /// Qualification does not leak between architectures: SM86 and SM120 are
    /// separate identities, and a passing SM86 run says nothing about SM120.
    pub fn affine_linear_catalogue() -> KernelCatalogue {
        let hash = parse_sha256(AFFINE_LINEAR_FATBIN_SHA256);
        let mut descriptors = Vec::new();
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            for width in [Precision::Int4, Precision::Int8] {
                descriptors.push(SemanticKernelDescriptor {
                    id: KernelId(format!(
                        "{}-linear-v1-{}",
                        super::profile_name(width),
                        sm.name()
                    )),
                    abi_version: super::AFFINE_LINEAR_ABI,
                    operation: SemanticKernelOp::Linear,
                    inputs: vec![
                        KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                        KernelOperand::Weight(WeightPrecision::expect(width)),
                    ],
                    output: ActivationPrecision::expect(Precision::Bf16),
                    accumulation: AccumulationPolicy::Bf16InF32Acc,
                    rounding: RoundingProfile::FinalBf16Rne,
                    layout: TensorLayout::ContiguousRowMajorV1,
                    shape: KernelShapeBounds {
                        max_rows: 65_536,
                        // Task 0105: Gemma 31B's dense MLP declares 21,504
                        // (`down_proj`'s input, `gate_proj`/`up_proj`'s
                        // output). `max_output` stays 65,536 -- it was
                        // already wide enough -- and `expert_mlp_catalogue`
                        // is untouched: Gemma 31B is dense, and nothing here
                        // qualifies a routed shape at this depth.
                        max_input: 21_504,
                        max_output: 65_536,
                    },
                    sm,
                    // Zero, and that is the claim worth reading twice: the
                    // dequantized weight lives in a 16x16 shared-memory tile
                    // inside the launch, so there is no workspace for a BF16
                    // copy of the weight because no such copy exists.
                    workspace: WorkspaceExpression::Zero,
                    image_sha256: hash,
                    symbols: vec![KernelSymbol(super::AFFINE_LINEAR.to_string())],
                });
            }
        }
        KernelCatalogue::new(descriptors).expect("built-in descriptors are unique")
    }

    /// Task 0037's paged attention, one descriptor per SM.
    ///
    /// Per architecture and not per shape: head dimension, head counts, page
    /// width and visibility are runtime parameters of the one symbol, so a
    /// descriptor per mask or per head geometry would be a catalogue that grows
    /// with every checkpoint rather than with every kernel. Qualification does
    /// not leak between architectures — SM86 and SM120 are separate identities,
    /// and a passing Ampere run says nothing about Blackwell.
    ///
    /// The shape bounds are this operation's: `max_input` is the head
    /// dimension the image serves, and `max_rows` is the query rows one launch
    /// may carry. Neither is the history length, which is bounded by what the
    /// state authority admitted rather than by the kernel.
    pub fn paged_attention_catalogue() -> KernelCatalogue {
        let hash = parse_sha256(PAGED_ATTENTION_FATBIN_SHA256);
        let mut descriptors = Vec::new();
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(
                    super::paged_attention_id(sm)
                        .expect("this build declares an id for every targeted architecture")
                        .to_string(),
                ),
                abi_version: super::PAGED_ATTENTION_ABI,
                operation: SemanticKernelOp::PagedAttention,
                inputs: vec![
                    // Query rows, then the two paged payloads, then the table
                    // that says where a logical page physically is.
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
                    KernelOperand::PageIndex,
                ],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape: KernelShapeBounds {
                    max_rows: 65_536,
                    max_input: super::PAGED_ATTENTION_MAX_HEAD_DIM,
                    max_output: super::PAGED_ATTENTION_MAX_HEAD_DIM,
                },
                sm,
                // Zero, and it is worth reading twice: the running maximum, the
                // running denominator and the running value sum live in
                // registers and shared memory for the whole launch. There is no
                // workspace because there is no materialized score matrix —
                // which is the property that lets a 32,768-row history be
                // attended over without a buffer that grows with it.
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(super::PAGED_ATTENTION.to_string())],
            });
        }
        KernelCatalogue::new(descriptors).expect("built-in descriptors are unique")
    }

    /// The shared dense graph package. Existing Linear, whole-row RMSNorm,
    /// routed expert and unit Residual symbols are included in this image so
    /// one module can run the full selected graph. Paged attention keeps the
    /// already-qualified descriptor and image identity; its state run owns the
    /// separate module load.
    pub fn dense_graph_catalogue() -> KernelCatalogue {
        let hash = parse_sha256(DENSE_GRAPH_FATBIN_SHA256);
        let bf16 = KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16));
        let weight = KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16));
        let index = KernelOperand::PositionIndex;
        let shape = KernelShapeBounds {
            max_rows: 65_536,
            max_input: 65_536,
            max_output: 65_536,
        };
        let mut descriptors = Vec::new();
        for sm in [SmVersion::SM86, SmVersion::SM120] {
            let suffix = sm.name();
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-embedding-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::Embedding,
                inputs: vec![KernelOperand::TokenIndex, weight],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_EMBEDDING.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-linear-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::Linear,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(BF16_LINEAR.to_string())],
            });
            for width in [Precision::Int4, Precision::Int8] {
                descriptors.push(SemanticKernelDescriptor {
                    id: KernelId(format!(
                        "dense-affine-linear-{}-v1-{suffix}",
                        super::profile_name(width)
                    )),
                    abi_version: DENSE_GRAPH_ABI,
                    operation: SemanticKernelOp::Linear,
                    inputs: vec![bf16, KernelOperand::Weight(WeightPrecision::expect(width))],
                    output: ActivationPrecision::expect(Precision::Bf16),
                    accumulation: AccumulationPolicy::Bf16InF32Acc,
                    rounding: RoundingProfile::FinalBf16Rne,
                    layout: TensorLayout::ContiguousRowMajorV1,
                    shape: KernelShapeBounds {
                        max_rows: 65_536,
                        // Same bound and the same reason as
                        // `affine_linear_catalogue`: the dense graph's own
                        // affine linear is the same kernel and symbol.
                        max_input: 21_504,
                        max_output: 65_536,
                    },
                    sm,
                    workspace: WorkspaceExpression::Zero,
                    image_sha256: hash,
                    symbols: vec![KernelSymbol(super::AFFINE_LINEAR.to_string())],
                });
            }
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-linear-split-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::LinearSplit,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_LINEAR_SPLIT.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-linear-partial-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::LinearPartial,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(Precision::F32),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::Unrounded,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_LINEAR_PARTIAL.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-rms-norm-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::RmsNorm,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::RowsTimesF32,
                image_sha256: hash,
                symbols: vec![
                    KernelSymbol(BF16_RMS_SUM.to_string()),
                    KernelSymbol(BF16_RMS_APPLY.to_string()),
                ],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-grouped-rms-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::GroupedRmsNorm,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_GROUPED_RMS.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-rope-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::Rope,
                inputs: vec![bf16, index],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::RowsTimesRopeAnglesF32,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_ROPE.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-geglu-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::GeGlu,
                inputs: vec![bf16, bf16],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_GEGLU.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-residual-scaled-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::ScaledResidual,
                inputs: vec![bf16, bf16],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_RESIDUAL_SCALED.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-vocab-projection-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::VocabProjection,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(Precision::F32),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::Unrounded,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_VOCAB_PROJECTION.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-route-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::Route,
                inputs: vec![bf16, weight, weight, weight],
                output: ActivationPrecision::expect(Precision::F32),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::Unrounded,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_ROUTE.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-expert-mlp-gelu-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::ExpertMlp(GateTransform::GeluTanh),
                inputs: vec![bf16, KernelOperand::RouteIndex, weight, weight],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::RowsTimesIntermediateF32,
                image_sha256: hash,
                symbols: vec![
                    KernelSymbol(DENSE_EXPERT_PROJECT_GELU.to_string()),
                    KernelSymbol(DENSE_EXPERT_DOWN.to_string()),
                ],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-combine-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::Combine,
                inputs: vec![KernelOperand::RouteIndex, bf16],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_COMBINE.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-combine-partial-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::CombinePartial,
                inputs: vec![KernelOperand::RouteIndex, bf16],
                output: ActivationPrecision::expect(Precision::F32),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::Unrounded,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(DENSE_COMBINE_PARTIAL.to_string())],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-combine-host-join-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::CombineHostJoin,
                inputs: vec![KernelOperand::RouteIndex, bf16],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::RowsTimesHiddenTimesTwoF32,
                image_sha256: hash,
                symbols: vec![
                    KernelSymbol(DENSE_COMBINE_PARTIAL.to_string()),
                    KernelSymbol(TP_REDUCE_F32.to_string()),
                ],
            });
            descriptors.push(SemanticKernelDescriptor {
                id: KernelId(format!("dense-residual-v1-{suffix}")),
                abi_version: DENSE_GRAPH_ABI,
                operation: SemanticKernelOp::Residual,
                inputs: vec![bf16, bf16],
                output: ActivationPrecision::expect(Precision::Bf16),
                accumulation: AccumulationPolicy::Bf16InF32Acc,
                rounding: RoundingProfile::FinalBf16Rne,
                layout: TensorLayout::ContiguousRowMajorV1,
                shape,
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: hash,
                symbols: vec![KernelSymbol(BF16_RESIDUAL.to_string())],
            });
        }
        descriptors.extend(paged_attention_catalogue().descriptors().iter().cloned());
        KernelCatalogue::new(descriptors).expect("built-in dense descriptors are unique")
    }

    /// The dense package with this architecture's BF16 Linear backed by cuBLAS.
    #[cfg(feature = "cublas")]
    pub fn dense_graph_catalogue_unordered(sm: SmVersion) -> KernelCatalogue {
        let bf16 = KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16));
        let weight = KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16));
        let mut descriptors = dense_graph_catalogue().descriptors().to_vec();
        let selected = descriptors
            .iter_mut()
            .find(|descriptor| {
                descriptor.operation == SemanticKernelOp::Linear
                    && descriptor.sm == sm
                    && descriptor.inputs.as_slice() == [bf16, weight]
            })
            .expect("built-in dense catalogue has one BF16 Linear for each target SM");
        *selected = SemanticKernelDescriptor {
            id: KernelId(format!("bf16-linear-cublas-v1-{}", sm.name())),
            abi_version: DENSE_GRAPH_ABI,
            operation: SemanticKernelOp::Linear,
            inputs: vec![bf16, weight],
            output: ActivationPrecision::expect(Precision::Bf16),
            accumulation: AccumulationPolicy::Bf16InF32AccUnordered,
            rounding: RoundingProfile::FinalBf16Rne,
            layout: TensorLayout::ContiguousRowMajorV1,
            shape: KernelShapeBounds {
                max_rows: 65_536,
                max_input: 65_536,
                max_output: 262_144,
            },
            sm,
            workspace: WorkspaceExpression::Zero,
            image_sha256: cublas_sha256(),
            symbols: vec![KernelSymbol("cublas:gemm_ex".to_string())],
        };
        for (operation, id, symbols) in [
            (
                SemanticKernelOp::LinearPartial,
                format!("bf16-linear-partial-cublas-v1-{}", sm.name()),
                vec![KernelSymbol("cublas:gemm_ex_partial".to_string())],
            ),
            (
                SemanticKernelOp::LinearSplit,
                format!("bf16-linear-split-cublas-v1-{}", sm.name()),
                vec![
                    KernelSymbol("cublas:gemm_ex_split".to_string()),
                    KernelSymbol(TP_REDUCE_F32.to_string()),
                ],
            ),
        ] {
            let selected = descriptors
                .iter_mut()
                .find(|descriptor| {
                    descriptor.operation == operation
                        && descriptor.sm == sm
                        && descriptor.inputs.as_slice() == [bf16, weight]
                })
                .expect("built-in dense catalogue has one row-linear descriptor per SM");
            *selected = SemanticKernelDescriptor {
                id: KernelId(id),
                abi_version: DENSE_GRAPH_ABI,
                operation,
                inputs: vec![bf16, weight],
                output: ActivationPrecision::expect(
                    if operation == SemanticKernelOp::LinearPartial {
                        Precision::F32
                    } else {
                        Precision::Bf16
                    },
                ),
                accumulation: AccumulationPolicy::Bf16InF32AccUnordered,
                rounding: if operation == SemanticKernelOp::LinearPartial {
                    RoundingProfile::Unrounded
                } else {
                    RoundingProfile::FinalBf16Rne
                },
                layout: TensorLayout::ContiguousRowMajorV1,
                shape: KernelShapeBounds {
                    max_rows: 65_536,
                    max_input: 65_536,
                    max_output: 262_144,
                },
                sm,
                workspace: WorkspaceExpression::Zero,
                image_sha256: cublas_sha256(),
                symbols,
            };
        }
        KernelCatalogue::new(descriptors).expect("built-in unordered dense descriptors are unique")
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

    /// This build's paged attention image digest, parsed without allocating.
    pub fn paged_attention_sha256() -> [u8; 32] {
        parse_sha256(PAGED_ATTENTION_FATBIN_SHA256)
    }

    #[cfg(feature = "cublas")]
    pub fn cublas_sha256() -> [u8; 32] {
        parse_sha256(CUBLAS_SO_SHA256)
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

/// The execution profile an integer weight width names, as document 03 spells
/// it. Kept beside the symbol rather than inside the image module so the host
/// lane can name a profile without a CUDA toolkit.
pub const fn profile_name(width: moxie_types::Precision) -> &'static str {
    match width {
        moxie_types::Precision::Int4 => "w4a16",
        moxie_types::Precision::Int8 => "w8a16",
        _ => "bf16",
    }
}

#[cfg(feature = "fatbin")]
pub use images::{
    AFFINE_LINEAR_FATBIN, AFFINE_LINEAR_FATBIN_SHA256, BF16_CHAIN_FATBIN, BF16_CHAIN_FATBIN_SHA256,
    DENSE_GRAPH_FATBIN, DENSE_GRAPH_FATBIN_SHA256, EXPERT_MLP_FATBIN, EXPERT_MLP_FATBIN_SHA256,
    HOST_COMPILER_VERSION, KERNEL_ARCHS, NVCC_VERSION, PAGED_ATTENTION_FATBIN,
    PAGED_ATTENTION_FATBIN_SHA256, SMOKE_FATBIN, SMOKE_FATBIN_SHA256, SMOKE_FATBIN_SM86_ONLY,
    SMOKE_FATBIN_SM86_SHA256, affine_linear_catalogue, axpy_capability, bf16_chain_catalogue,
    compiled_sm, dense_graph_catalogue, expert_mlp_catalogue, paged_attention_catalogue,
};

#[cfg(all(feature = "fatbin", feature = "cublas"))]
pub use images::{CUBLAS_SO_SHA256, cublas_sha256, dense_graph_catalogue_unordered};

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
        assert!(DENSE_GRAPH_FATBIN.len() > 1024);
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
        assert_eq!(DENSE_GRAPH_FATBIN_SHA256.len(), 64);
        assert!(NVCC_VERSION.contains("13.0"), "{NVCC_VERSION}");
        assert!(!HOST_COMPILER_VERSION.is_empty());
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn the_package_predicate_and_the_catalogue_agree() {
        // Two statements of one package's identity, pinned to each other. The
        // predicate exists because admission cannot afford to build the
        // catalogue; it is only worth having if it says the same thing.
        let catalogue = paged_attention_catalogue();
        assert_eq!(catalogue.descriptors().len(), 2);
        for descriptor in catalogue.descriptors() {
            assert!(
                paged_attention_declares(descriptor),
                "the predicate rejected {}, which this build declares",
                descriptor.id.0
            );
        }

        // And every field it compares actually changes the answer. A predicate
        // that returned `true` for everything would pass the loop above.
        let base = catalogue.descriptors()[0].try_clone().expect("clone");
        /// One named change to a descriptor, applied alone.
        type Change = Box<dyn Fn(&mut moxie_types::SemanticKernelDescriptor)>;
        let mutations: Vec<(&str, Change)> = vec![
            (
                "id",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.id = moxie_types::KernelId("invented".into());
                }),
            ),
            (
                "abi",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| d.abi_version += 1),
            ),
            (
                "operation",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.operation = moxie_types::SemanticKernelOp::Linear;
                }),
            ),
            (
                "operands",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.inputs.pop();
                }),
            ),
            (
                "accumulation",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.accumulation = moxie_types::AccumulationPolicy::F32;
                }),
            ),
            (
                "workspace",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.workspace = moxie_types::WorkspaceExpression::RowsTimesF32;
                }),
            ),
            (
                "shape bounds",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.shape.max_rows = u64::MAX;
                }),
            ),
            (
                "image digest",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| d.image_sha256 = [0; 32]),
            ),
            (
                "symbol",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.symbols[0] = moxie_types::KernelSymbol(AFFINE_LINEAR.to_string());
                }),
            ),
            (
                "architecture",
                Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                    d.sm = moxie_types::SmVersion { major: 9, minor: 0 };
                }),
            ),
        ];
        for (what, change) in mutations {
            let mut descriptor = base.try_clone().expect("clone");
            change(&mut descriptor);
            assert!(
                !paged_attention_declares(&descriptor),
                "a descriptor with a changed {what} was accepted as this package's"
            );
        }

        // `layout` and `rounding` are compared too, and cannot be mutated here:
        // `TensorLayout` and `RoundingProfile` each have exactly one variant
        // today. When either gains a second, this list gains a case.
        assert_eq!(base.layout, moxie_types::TensorLayout::ContiguousRowMajorV1);
        assert_eq!(base.rounding, moxie_types::RoundingProfile::FinalBf16Rne);
    }

    #[test]
    fn a_profile_name_follows_the_weight_width() {
        // Document 03's names. `w4a16` is a weight/activation profile, so the
        // activation half is fixed and only the weight width may move it.
        assert_eq!(profile_name(moxie_types::Precision::Int4), "w4a16");
        assert_eq!(profile_name(moxie_types::Precision::Int8), "w8a16");
        assert_eq!(profile_name(moxie_types::Precision::Bf16), "bf16");
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn the_two_quantized_profiles_are_one_symbol_and_four_identities() {
        // The shared-ness has to be structural. If these ever name different
        // symbols, "shared W4A16/W8A16 path" has quietly become two kernels.
        let catalogue = affine_linear_catalogue();
        assert_eq!(catalogue.descriptors().len(), 4);
        for descriptor in catalogue.descriptors() {
            assert_eq!(descriptor.symbols.len(), 1);
            assert_eq!(descriptor.symbols[0].0, AFFINE_LINEAR);
            assert_eq!(
                descriptor.workspace,
                moxie_types::WorkspaceExpression::Zero,
                "a workspace would be where a dequantized weight could hide"
            );
        }
        let ids: Vec<&str> = catalogue
            .descriptors()
            .iter()
            .map(|d| d.id.0.as_str())
            .collect();
        for want in [
            "w4a16-linear-v1-sm_86",
            "w8a16-linear-v1-sm_86",
            "w4a16-linear-v1-sm_120",
            "w8a16-linear-v1-sm_120",
        ] {
            assert!(ids.contains(&want), "{want} missing from {ids:?}");
        }
    }

    #[cfg(feature = "fatbin")]
    #[test]
    fn the_quantized_package_is_a_separate_image_from_the_bf16_one() {
        // Task 0012's chain and task 0028's linear are different fatbins, so a
        // descriptor that cited the wrong image would fail its own load rather
        // than silently running the other package's code.
        assert_ne!(AFFINE_LINEAR_FATBIN_SHA256, BF16_CHAIN_FATBIN_SHA256);
        assert_ne!(AFFINE_LINEAR_FATBIN_SHA256, EXPERT_MLP_FATBIN_SHA256);
        assert!(AFFINE_LINEAR_FATBIN.len() > 1024);
        assert_ne!(
            affine_linear_catalogue().digest(),
            bf16_chain_catalogue().digest()
        );
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
