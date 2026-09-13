//! Laguna: metadata, tensor roles and the routed block. Nothing else.
//!
//! Document 09 §B lists what a model definition may contain -- "checkpoint
//! architecture metadata interpretation and logical tensor-role mapping",
//! "graph composition and mathematical constants/options specific to that
//! family" -- and what it may not. As in [`crate::gemma4`], the enforcement is
//! the containing crate's three-entry dependency list, which no module can
//! widen.
//!
//! # What this is, and the two things that stop it being more
//!
//! [`RoutedBlocks`] composes **Laguna's routed feed-forward block** over
//! weights the caller invents. It is a contract fixture. It is not Laguna, it
//! does not read the checkpoint, and its output is not model output.
//!
//! It is the *block* and not the tower because a faithful Laguna decoder layer
//! needs two operations the shared catalogue does not have, and inventing
//! either is forbidden:
//!
//! 1. **Attention output gating.** `LagunaAttention.forward` computes
//!    `softplus(g_proj(x))` in FP32, per head, and multiplies it into the
//!    attention output *before* `o_proj`. The equation is pinned in the
//!    artifact's own `modeling_laguna.py`, so it is known; what does not exist
//!    is a shared operation for it. `gating_types` is `per_head` on all 48
//!    layers, so this blocks every layer.
//! 2. **The yarn RoPE ramp** on the twelve `full_attention` layers. The
//!    artifact declares `rope_type: "yarn"` with its parameters, and its own
//!    file implements **only** `compute_default_rope_parameters`, delegating
//!    every other type to a `transformers` function it does not ship. The
//!    artifact declares `transformers` 5.14.1; the copy installed on this
//!    machine is 5.5.3, and `truncate` -- a parameter of that function -- is
//!    not declared in the config at all. One mismatched copy is not a pinned
//!    exporter.
//!
//! [`ArtifactGeometry::tower_gaps`] returns exactly that list, computed from
//! the declared geometry rather than hard-coded, so a configuration without
//! them reports none.
//!
//! # What the routed block does preserve
//!
//! From `LagunaTopKRouter.forward`, `LagunaExperts.forward` and
//! `LagunaSparseMoeBlock.forward`, read and **never executed**:
//!
//! - a **sigmoid** router over a row the block has already normalized, with no
//!   normalization or gain of its own;
//! - `e_score_correction_bias` added to the **selection** scores while the
//!   coefficients are gathered from the **unbiased** ones;
//! - renormalization over the selected experts (`norm_topk_prob`);
//! - a SwiGLU expert feed-forward at `moe_intermediate_size`, fused across
//!   experts;
//! - accumulation in ascending expert id (`index_add_` over `expert_hit`);
//! - the routed scaling factor applied to the **combined** row;
//! - a shared expert reading the **same** normalized tensor the router does,
//!   added after the scale.

use moxie_graph::{
    CombineOrder, ExpertActivation, Graph, GraphBuilder, IndexEncoding, OpParams, OracleRegistry,
    RouteCoefficient, RouteScore, RouterInput, TensorSpec, ValueId, ValueRole,
};
use moxie_model_api::{
    GraphRequirements, ModelDefinition, ModelMetadata, TensorRequirement, TensorRole,
};
use moxie_types::{Dim, Error, Precision, Result, SymbolId, WeightPrecision};

/// A routed-expert block's declared geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeGeometry {
    pub experts: u64,
    pub top_k: u64,
    /// One routed expert's intermediate width, `moe_intermediate_size`.
    pub moe_intermediate: u64,
    /// The shared expert's intermediate width, `shared_expert_intermediate_size`.
    ///
    /// A separate field even though the artifact's two happen to be equal:
    /// `LagunaSparseMoeBlock` constructs the shared expert with
    /// `intermediate_size=config.shared_expert_intermediate_size`, so they are
    /// two parameters, and a family that set them apart would be described
    /// wrongly by one.
    pub shared_intermediate: u64,
    /// `moe_routed_scaling_factor`, applied to the **combined** routed row
    /// before the shared expert is added.
    pub routed_scaling_factor: f32,
    /// `moe_router_logit_softcapping`. Zero means disabled, which is the only
    /// value this graph can compose; see [`BlockConfig::check`].
    pub router_logit_softcap: f32,
}

/// One attention layer's output gate, as the artifact declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionGate {
    /// `gating: false`. No `g_proj`.
    None,
    /// `gating: "per-head"`. One `softplus` gate per head, broadcast across the
    /// head dimension.
    PerHead,
    /// `gating: true` / `"per-element"`. One gate per `(head, channel)`.
    PerElement,
}

/// One layer type's rotary parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeKind {
    /// `rope_type: "default"` -- `compute_default_rope_parameters`, which the
    /// artifact's own file implements.
    Default {
        theta: f32,
        /// `partial_rotary_factor`. The inverse-frequency denominator is the
        /// **rotated** width here, `dim = int(head_dim * partial)`, unlike
        /// Gemma 4's global layers, which divide by the full head dimension.
        partial_rotary: Fraction,
    },
    /// `rope_type: "yarn"` -- delegated to a `transformers` function the
    /// artifact does not ship. Recorded, never composed.
    Yarn,
}

/// An exact rational fraction of a head, so a rotary fraction is never a float
/// that nearly divides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fraction {
    pub numerator: u64,
    pub denominator: u64,
}

impl Fraction {
    pub const WHOLE: Self = Self {
        numerator: 1,
        denominator: 1,
    };
    /// The artifact's `partial_rotary_factor` of 0.5 on its full-attention
    /// layers.
    pub const HALF: Self = Self {
        numerator: 1,
        denominator: 2,
    };
}

/// An operation a Laguna decoder layer needs that the shared catalogue has not
/// got.
///
/// Data rather than a refusal site: [`ArtifactGeometry::tower_gaps`] computes
/// this list from the declared geometry, so a geometry without them reports an
/// empty list and the same function answers both cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gap {
    /// `softplus(g_proj(x))` applied per head before `o_proj`.
    AttentionOutputGate,
    /// The yarn inverse-frequency ramp and its attention scaling.
    YarnRope,
}

impl Gap {
    /// What is missing and why it may not be guessed.
    pub const fn detail(self) -> &'static str {
        match self {
            Gap::AttentionOutputGate => {
                "attention output gating: softplus(g_proj(x)) applied per head before o_proj has \
                 no shared semantic operation, and no existing activation is softplus"
            }
            Gap::YarnRope => {
                "yarn rotary scaling: modeling_laguna.py implements only \
                 compute_default_rope_parameters and delegates every other rope_type to a \
                 transformers function the artifact does not ship"
            }
        }
    }
}

/// The artifact's declared geometry, for reference and refusal.
///
/// Nothing executes this. It is here so that a caller can compare a reduced
/// configuration against the real one, and so that the numbers in
/// `docs/models/laguna.md` have an executable counterpart that a test can check
/// against `config.json` if the artifact is ever re-inspected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArtifactGeometry {
    pub hidden: u64,
    pub layers: u32,
    pub head_dim: u64,
    /// Query heads on a **full-attention** layer.
    pub full_heads: u64,
    /// Query heads on a **sliding-attention** layer. The artifact's two differ:
    /// 48 and 72, so one field could not describe it.
    pub sliding_heads: u64,
    pub kv_heads: u64,
    /// The dense MLP's intermediate width, used by the layers in
    /// [`ArtifactGeometry::dense_layers`].
    pub intermediate: u64,
    pub vocab: u64,
    /// Layer `l` is full-attention when `l % full_stride == 0`. The artifact's
    /// stride is 4, giving twelve full-attention layers.
    pub full_stride: u32,
    pub sliding_window: u64,
    pub rms_eps: f32,
    pub full_rope: RopeKind,
    pub sliding_rope: RopeKind,
    pub attention_gate: AttentionGate,
    /// The layers whose feed-forward is dense rather than routed,
    /// `mlp_only_layers`. The artifact's is `[0]`.
    pub dense_layers: &'static [u32],
    /// `tie_word_embeddings`. False here: `lm_head` is its own tensor.
    pub tied_embeddings: bool,
    /// Routed layers whose experts the quantizer left in **BF16**.
    ///
    /// `quantization_config.ignore` names every module of layers 0, 46 and 47,
    /// and layer 0 has no routed experts, so these two are the routed layers
    /// that cost the BF16 figure rather than the packed one. Recorded as data
    /// because it is a quantizer's sensitivity choice, not a family rule.
    ///
    /// This field exists because a record got the arithmetic wrong without it:
    /// multiplying the packed per-layer cost by all 47 sparse layers understated
    /// the expert working set by **6.87 GB**, in the same document that recorded
    /// the exception three paragraphs earlier.
    pub bf16_expert_layers: &'static [u32],
    /// One routed expert as the artifact stores it: packed INT4 codes, BF16
    /// scales, packed zero points and the shape vector.
    ///
    /// Measured from shard headers, not derived: the zero points are packed
    /// along the **output** axis while the codes are packed along the input
    /// axis, and a derivation that assumed one convention for both would be
    /// wrong by a factor.
    pub stored_expert_bytes: u64,
    /// The artifact's `total_size`.
    pub artifact_bytes: u64,
    pub max_trained_position: u64,
    pub moe: MoeGeometry,
}

/// `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`, revision
/// `bc59f497520b23759ce61cc5164ca28bcc4f53bc`, read from its own `config.json`
/// and safetensors headers on 2026-09-13.
///
/// The quantized variant of `poolside/Laguna-S-2.1`. Its weights are
/// compressed-tensors `pack-quantized` **asymmetric INT4 at group 32**, which
/// no importer in this workspace accepts; nothing here reads a tensor payload.
pub const ARTIFACT: ArtifactGeometry = ArtifactGeometry {
    hidden: 3072,
    layers: 48,
    head_dim: 128,
    full_heads: 48,
    sliding_heads: 72,
    kv_heads: 8,
    intermediate: 12288,
    vocab: 100_352,
    full_stride: 4,
    sliding_window: 512,
    rms_eps: 1e-6,
    full_rope: RopeKind::Yarn,
    sliding_rope: RopeKind::Default {
        theta: 10_000.0,
        partial_rotary: Fraction::WHOLE,
    },
    attention_gate: AttentionGate::PerHead,
    dense_layers: &[0],
    tied_embeddings: false,
    bf16_expert_layers: &[46, 47],
    stored_expert_bytes: 5_455_920,
    artifact_bytes: 76_813_095_232,
    max_trained_position: 1_048_576,
    moe: MoeGeometry {
        experts: 256,
        top_k: 10,
        moe_intermediate: 1024,
        shared_intermediate: 1024,
        routed_scaling_factor: 2.5,
        router_logit_softcap: 0.0,
    },
};

impl ArtifactGeometry {
    /// Whether layer `l` uses full causal attention rather than a window.
    pub const fn full_attention_layer(&self, layer: u32) -> bool {
        layer < self.layers && self.full_stride != 0 && layer.is_multiple_of(self.full_stride)
    }

    /// Whether layer `l`'s feed-forward is the dense MLP rather than a routed
    /// block.
    pub fn dense_layer(&self, layer: u32) -> bool {
        self.dense_layers.contains(&layer)
    }

    /// One routed expert's BF16 bytes at this geometry.
    ///
    /// `3 · intermediate · hidden · 2`: the fused gate and up halves plus the
    /// down projection. **Not** the artifact's own per-expert cost, which is
    /// smaller because its experts are INT4 -- see
    /// `docs/models/laguna.md`. This is the figure a BF16 fixture at this
    /// geometry demands, and the one a restricted budget is a ratio of.
    pub const fn expert_bf16_bytes(&self) -> u64 {
        3 * self.moe.moe_intermediate * self.hidden * 2
    }

    /// Whether layer `l`'s routed experts are stored in BF16 rather than packed.
    pub fn bf16_expert_layer(&self, layer: u32) -> bool {
        self.bf16_expert_layers.contains(&layer)
    }

    /// One routed layer's experts, as the artifact stores them.
    ///
    /// `None` for a layer that has no routed experts.
    pub fn layer_expert_bytes(&self, layer: u32) -> Option<u64> {
        if layer >= self.layers || self.dense_layer(layer) {
            return None;
        }
        Some(
            self.moe.experts
                * if self.bf16_expert_layer(layer) {
                    self.expert_bf16_bytes()
                } else {
                    self.stored_expert_bytes
                },
        )
    }

    /// Every routed layer's experts, as the artifact stores them.
    ///
    /// Summed over the layers rather than multiplied by their count, because
    /// they do not all cost the same and a multiplication cannot say so.
    pub fn expert_bytes_total(&self) -> u64 {
        (0..self.layers)
            .filter_map(|l| self.layer_expert_bytes(l))
            .sum()
    }

    /// The operations a faithful decoder layer of this geometry needs and the
    /// shared catalogue has not got.
    ///
    /// Empty for a geometry without an attention gate whose layer types all use
    /// a rotary this workspace can compute. Both outcomes are reachable, which
    /// is what keeps this a computation rather than a constant with a comment.
    pub fn tower_gaps(&self) -> Vec<Gap> {
        let mut gaps = Vec::new();
        if self.attention_gate != AttentionGate::None {
            gaps.push(Gap::AttentionOutputGate);
        }
        if matches!(self.full_rope, RopeKind::Yarn) || matches!(self.sliding_rope, RopeKind::Yarn) {
            gaps.push(Gap::YarnRope);
        }
        gaps
    }
}

/// The constants a routed block needs, at whatever size the caller asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockConfig {
    pub hidden: u64,
    /// How many routed blocks the fixture stacks.
    pub blocks: u32,
    pub vocab: u64,
    pub rms_eps: f32,
    /// The longest position this configuration was built for.
    ///
    /// R19: width, admitted context and actual visible tokens are three
    /// different numbers. A fixture states its own small one, so the artifact's
    /// 1,048,576 cannot be read out of it and mistaken for a supported context.
    pub max_trained_position: u64,
    pub moe: MoeGeometry,
}

impl BlockConfig {
    /// A small configuration with the artifact's routing *shape* and none of
    /// its scale.
    pub fn reduced() -> Self {
        Self {
            hidden: 12,
            blocks: 2,
            vocab: 11,
            rms_eps: ARTIFACT.rms_eps,
            max_trained_position: 64,
            moe: MoeGeometry {
                experts: 6,
                top_k: 3,
                moe_intermediate: 5,
                shared_intermediate: 4,
                routed_scaling_factor: ARTIFACT.moe.routed_scaling_factor,
                router_logit_softcap: ARTIFACT.moe.router_logit_softcap,
            },
        }
    }

    /// Everything this graph cannot express, refused rather than approximated.
    pub fn check(&self) -> Result<()> {
        if self.hidden == 0 || self.blocks == 0 || self.vocab == 0 {
            return Err(Error::InvalidRequest {
                field: "laguna_block",
                detail: format!(
                    "hidden {}, blocks {}, vocab {}",
                    self.hidden, self.blocks, self.vocab
                ),
            });
        }
        if self.moe.top_k > self.moe.experts || self.moe.top_k == 0 {
            return Err(Error::InvalidRequest {
                field: "top_k",
                detail: format!("top_k {} outside 1..={}", self.moe.top_k, self.moe.experts),
            });
        }
        // Refused, not carried as a dead parameter. `Route` has no softcap
        // field precisely because no consumer needs one: the artifact declares
        // 0.0, and a branch nothing exercises is a stub. A checkpoint that
        // declares a nonzero value stops here, where the gap is visible.
        if self.moe.router_logit_softcap != 0.0 {
            return Err(Error::UnsupportedKernel {
                operation: "route",
                detail: format!(
                    "moe_router_logit_softcapping is {}; Route has no logit softcap parameter, \
                     because no artifact inspected so far declares one. Adding it needs a \
                     checkpoint that uses it and a fixture that distinguishes it.",
                    self.moe.router_logit_softcap
                ),
            });
        }
        if !self.moe.routed_scaling_factor.is_finite() {
            return Err(Error::InvalidRequest {
                field: "moe_routed_scaling_factor",
                detail: format!("{} is not finite", self.moe.routed_scaling_factor),
            });
        }
        Ok(())
    }
}

/// What [`RoutedBlocks`] gives up.
///
/// Wider than [`crate::gemma4::Reduction`], and the difference is the point:
/// Gemma's reduced graph drops *scale*, and this one also drops *mathematics*
/// it is not allowed to invent. A fixture that could not say so would be the
/// "synthetic output described as model support" document 06 forbids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduction {
    /// Weights are whatever the caller supplied. The artifact is asymmetric
    /// INT4 `pack-quantized` at group 32 and has no importer.
    pub synthetic_weights: bool,
    /// **No attention at all**: no projections, no rotary, no mask, no KV. The
    /// two operations that block it are [`ArtifactGeometry::tower_gaps`].
    pub no_attention: bool,
    /// No dense-MLP layer, no tokenizer, no chat template, no reasoning or tool
    /// parser, and no `dflash` speculative head.
    pub routed_block_only: bool,
}

impl Reduction {
    pub const fn all() -> Self {
        Self::ALL
    }

    const ALL: Self = Self {
        synthetic_weights: true,
        no_attention: true,
        routed_block_only: true,
    };
}

/// A logical tensor role and the graph value that carries it.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundRole {
    pub role: TensorRole,
    pub value: ValueId,
}

/// A composed graph and everything a caller needs to fill it.
#[derive(Debug)]
pub struct Composition {
    pub graph: Graph,
    pub tokens: ValueId,
    pub weights: Vec<BoundRole>,
}

/// Laguna's routed feed-forward blocks as a model definition.
#[derive(Debug, Clone)]
pub struct RoutedBlocks {
    config: BlockConfig,
    metadata: ModelMetadata,
    tensors: Vec<TensorRequirement>,
}

fn role(name: &str, layer: Option<u32>) -> TensorRole {
    TensorRole {
        name: name.to_string(),
        layer,
        expert: None,
    }
}

fn required(name: &str, layer: Option<u32>) -> Result<TensorRequirement> {
    Ok(TensorRequirement {
        role: role(name, layer),
        // BF16 only, and that is a statement about this graph rather than about
        // the family: the artifact's routed experts are INT4. Task 0024's
        // importer now reads them into canonical affine form, but **nothing
        // executes a canonical INT4 tensor** -- W4A16 is M3 item 3 and no
        // kernel exists. Listing INT4 here would advertise a path that is not
        // there.
        allowed: vec![WeightPrecision::new(Precision::Bf16)?],
        required: true,
    })
}

fn weight(
    g: &mut GraphBuilder,
    bound: &mut Vec<BoundRole>,
    name: &str,
    layer: Option<u32>,
    shape: Vec<u64>,
) -> Result<ValueId> {
    let label = match layer {
        Some(l) => format!("{name}.{l}"),
        None => name.to_string(),
    };
    let id = g.weight(
        &label,
        TensorSpec::new(
            ValueRole::Weight(WeightPrecision::new(Precision::Bf16)?),
            shape.into_iter().map(Dim::constant).collect(),
        ),
    )?;
    bound.push(BoundRole {
        role: role(name, layer),
        value: id,
    });
    Ok(id)
}

/// Checked `a * b`, because these products become tensor extents.
fn width(what: &'static str, a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(Error::InvalidRequest {
        field: "laguna_block",
        detail: format!("{what} overflows: {a} x {b}"),
    })
}

impl RoutedBlocks {
    /// A reduced stack of Laguna-shaped routed blocks.
    ///
    /// `revision` names what the caller is modelling; it is recorded in the
    /// metadata and never interpreted. Passing the artifact's revision here
    /// does not make this the artifact, and the reduction is reported alongside
    /// it.
    pub fn reduced(config: BlockConfig, revision: &str) -> Result<Self> {
        config.check()?;
        let metadata = ModelMetadata {
            family: "laguna-routed-block-reduced".into(),
            revision: revision.into(),
            max_trained_position: config.max_trained_position,
            vocab_size: u32::try_from(config.vocab).map_err(|_| Error::InvalidArtifact {
                detail: format!("vocabulary {} does not fit a u32", config.vocab),
            })?,
        };
        let mut tensors = vec![
            required("embedding", None)?,
            required("final_norm", None)?,
            // `tie_word_embeddings` is false in this family, so the projection
            // is its own tensor. A graph that tied them would be describing a
            // different checkpoint.
            required("lm_head", None)?,
        ];
        for layer in 0..config.blocks {
            for name in [
                "ffn_norm",
                "router_proj",
                "router_selection_bias",
                "experts_gate_up",
                "experts_down",
                "shared_gate",
                "shared_up",
                "shared_down",
            ] {
                tensors.push(required(name, Some(layer))?);
            }
        }
        Ok(Self {
            config,
            metadata,
            tensors,
        })
    }

    pub fn config(&self) -> &BlockConfig {
        &self.config
    }

    /// What this graph is not. See [`Reduction`].
    pub const fn reduction(&self) -> Reduction {
        Reduction::ALL
    }

    /// Compose the graph.
    pub fn compose(&self, oracles: &OracleRegistry, rows: SymbolId) -> Result<Composition> {
        let c = &self.config;
        width("vocab * hidden", c.vocab, c.hidden)?;
        // The doubling is checked **first**. `2 * moe_intermediate` used to be
        // evaluated inline and handed to `width`, so an intermediate of
        // `1 << 63` overflowed before the checked multiplication ever ran and
        // panicked in a debug build -- reached through a public constructor
        // that had already accepted the configuration.
        let gate_up_rows = width("2 * moe_intermediate", 2, c.moe.moe_intermediate)?;
        width(
            "experts * 2 * moe_intermediate * hidden",
            c.moe.experts,
            width("2 * moe_intermediate * hidden", gate_up_rows, c.hidden)?,
        )?;
        width(
            "shared_intermediate * hidden",
            c.moe.shared_intermediate,
            c.hidden,
        )?;

        let mut g = GraphBuilder::new(moxie_graph::OracleId("moxie_oracles::host_reference"), rows);
        let tokens = g.input(
            "tokens",
            TensorSpec::new(
                ValueRole::Index(IndexEncoding::U64),
                vec![Dim::symbol(rows)],
            ),
        );
        let mut bound = Vec::new();

        let embedding = weight(
            &mut g,
            &mut bound,
            "embedding",
            None,
            vec![c.vocab, c.hidden],
        )?;
        let final_norm = weight(&mut g, &mut bound, "final_norm", None, vec![c.hidden])?;
        let lm_head = weight(&mut g, &mut bound, "lm_head", None, vec![c.vocab, c.hidden])?;

        // No embedding scale: `LagunaModel.forward` does not multiply the
        // looked-up rows by anything, unlike Gemma 4.
        let mut stream = g.node(
            OpParams::Embedding {
                vocab: c.vocab,
                hidden: c.hidden,
                scale: 1.0,
            },
            &[tokens, embedding],
        )?;

        for layer in 0..c.blocks {
            stream = routed_block(&mut g, &mut bound, c, layer, stream, gate_up_rows)?;
        }

        let normed = g.node(
            OpParams::RmsNorm {
                hidden: c.hidden,
                group: 1,
                eps: c.rms_eps,
            },
            &[stream, final_norm],
        )?;
        // No logit softcap: the artifact declares none.
        let logits = g.node(
            OpParams::VocabProjection {
                hidden: c.hidden,
                vocab: c.vocab,
                softcap: None,
            },
            &[normed, lm_head],
        )?;
        let graph = g.finish(logits, oracles)?;
        Ok(Composition {
            graph,
            tokens,
            weights: bound,
        })
    }
}

/// One routed block, transcribed from `LagunaDecoderLayer.forward` and
/// `LagunaSparseMoeBlock.forward`:
///
/// ```text
/// n = post_attention_layernorm(r)
/// s = shared_expert(n)                      SwiGLU at shared_expert_intermediate_size
/// y = Combine(Route(n), ExpertMlp(n, Route(n))) · moe_routed_scaling_factor
/// r'= r + (y + s)
/// ```
///
/// Three orderings are worth stating because a plausible graph gets them wrong.
///
/// **The router and the shared expert read the same tensor.** Both take `n`,
/// the block's single `post_attention_layernorm` output. That is the opposite
/// of Gemma 4, whose experts read their own `pre_feedforward_layernorm_2` and
/// whose router reads the *un-normalized* residual, and a graph that carried
/// Gemma's habit here would route every row on a different vector while still
/// producing a well-shaped answer.
///
/// **The scale is on the combined routed row only.** `expert_output` is
/// multiplied before `shared_expert_output` is added, so it is
/// [`OpParams::Combine`]'s `output_scale` and not the residual's.
///
/// **The shared expert takes no routing coefficient.** It is added outside
/// routing, exactly as in Gemma 4 -- one of the few things the two families
/// agree about.
fn routed_block(
    g: &mut GraphBuilder,
    bound: &mut Vec<BoundRole>,
    c: &BlockConfig,
    layer: u32,
    residual: ValueId,
    // `2 * moe_intermediate`, checked once by the caller and reused here rather
    // than recomputed. One question, answered once -- and recomputing it is how
    // the checked extent and the extent a tensor is actually given come apart.
    gate_up_rows: u64,
) -> Result<ValueId> {
    let moe = c.moe;
    let ffn_norm = weight(g, bound, "ffn_norm", Some(layer), vec![c.hidden])?;
    let normed = g.node(
        OpParams::RmsNorm {
            hidden: c.hidden,
            group: 1,
            eps: c.rms_eps,
        },
        &[residual, ffn_norm],
    )?;

    // The shared expert: an ordinary SwiGLU MLP at its own width.
    let shared_gate = weight(
        g,
        bound,
        "shared_gate",
        Some(layer),
        vec![moe.shared_intermediate, c.hidden],
    )?;
    let shared_up = weight(
        g,
        bound,
        "shared_up",
        Some(layer),
        vec![moe.shared_intermediate, c.hidden],
    )?;
    let shared_down = weight(
        g,
        bound,
        "shared_down",
        Some(layer),
        vec![c.hidden, moe.shared_intermediate],
    )?;
    let gate = linear(g, normed, shared_gate, c.hidden, moe.shared_intermediate)?;
    let up = linear(g, normed, shared_up, c.hidden, moe.shared_intermediate)?;
    let activated = g.node(
        OpParams::SwiGlu {
            width: moe.shared_intermediate,
        },
        &[gate, up],
    )?;
    let shared = linear(g, activated, shared_down, moe.shared_intermediate, c.hidden)?;

    // The router. Two tensors, and neither is a gain: this router does not
    // normalize its own input.
    let router_proj = weight(
        g,
        bound,
        "router_proj",
        Some(layer),
        vec![moe.experts, c.hidden],
    )?;
    let selection_bias = weight(
        g,
        bound,
        "router_selection_bias",
        Some(layer),
        vec![moe.experts],
    )?;
    let route = g.node(
        OpParams::Route {
            hidden: c.hidden,
            experts: moe.experts,
            top_k: moe.top_k,
            input: RouterInput::Raw,
            score: RouteScore::Sigmoid,
            per_expert_scale: false,
            selection_bias: true,
            // `routing_weights = routing_weights.to(hidden_states.dtype)`, the
            // last statement of `LagunaTopKRouter.forward`. The model dtype is
            // BF16, and this narrowing reaches every combined row.
            coefficient: RouteCoefficient::Bf16,
        },
        &[normed, router_proj, selection_bias],
    )?;

    // Fused across experts, as `LagunaExperts` declares them. The **checkpoint**
    // stores one tensor per expert per projection instead, and the artifact's
    // `_checkpoint_conversion_mapping` does not cover that rewrite: it remaps
    // only `e_score_correction_bias`. Whoever writes the importer resolves it;
    // this graph states the shape the pinned model computes with.
    let gate_up = weight(
        g,
        bound,
        "experts_gate_up",
        Some(layer),
        vec![moe.experts, gate_up_rows, c.hidden],
    )?;
    let expert_down = weight(
        g,
        bound,
        "experts_down",
        Some(layer),
        vec![moe.experts, c.hidden, moe.moe_intermediate],
    )?;
    let slots = g.node(
        OpParams::ExpertMlp {
            hidden: c.hidden,
            intermediate: moe.moe_intermediate,
            experts: moe.experts,
            top_k: moe.top_k,
            // `hidden_act` is `silu`, and `LagunaExperts` uses
            // `ACT2FN[config.hidden_act]` -- the same function the dense and
            // shared MLPs use.
            activation: ExpertActivation::SwiGlu,
        },
        &[normed, route, gate_up, expert_down],
    )?;
    let combined = g.node(
        OpParams::Combine {
            hidden: c.hidden,
            top_k: moe.top_k,
            // `index_add_` over `expert_hit`, which is expert-major.
            order: CombineOrder::AscendingExpertId,
            output_scale: moe.routed_scaling_factor,
        },
        &[route, slots],
    )?;

    // `expert_output + shared_expert_output`, then the layer residual. Two
    // unscaled sums: the scale that exists in this block is already on
    // `combined`.
    let branch = g.node(OpParams::Residual { scale: 1.0 }, &[combined, shared])?;
    g.node(OpParams::Residual { scale: 1.0 }, &[residual, branch])
}

fn linear(
    g: &mut GraphBuilder,
    x: ValueId,
    w: ValueId,
    in_features: u64,
    out_features: u64,
) -> Result<ValueId> {
    g.node(
        OpParams::Linear {
            in_features,
            out_features,
            // `attention_bias: false`, and none of the feed-forward linears in
            // the index carries a bias tensor either.
            bias: false,
        },
        &[x, w],
    )
}

impl ModelDefinition for RoutedBlocks {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensors(&self) -> &[TensorRequirement] {
        &self.tensors
    }

    fn graph_requirements(&self) -> GraphRequirements {
        GraphRequirements {
            ops: vec![
                moxie_graph::Op::Embedding,
                moxie_graph::Op::RmsNorm,
                moxie_graph::Op::Linear,
                moxie_graph::Op::SwiGlu,
                moxie_graph::Op::Route,
                moxie_graph::Op::ExpertMlp,
                moxie_graph::Op::Combine,
                moxie_graph::Op::Residual,
                moxie_graph::Op::VocabProjection,
            ],
        }
    }
}
