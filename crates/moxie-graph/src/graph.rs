//! Values, edges, shapes and graph construction.
//!
//! Document 02: a model definition is "data and graph construction", and every
//! operation must define "equations, input/output shapes, valid dtypes ... state
//! effects, partition legality". The M0 draft had the operation catalogue and
//! the per-operation rules but no graph: no edges, no shapes, nothing to walk.
//! Task 0003 adds them, because an interpreter is the first thing that needs to
//! know what feeds what.
//!
//! Two things are deliberately *not* here. There is no `Custom` node and no
//! callback: a graph is a list of closed operations with closed parameters, so a
//! planner can reason about it without executing model code. And there is no
//! execution: this crate builds and validates, and `moxie-interp` walks.
//!
//! ## Shapes
//!
//! Shapes are [`Dim`] expressions. The model's own dimensions -- hidden width,
//! vocabulary, head count -- are constants known when the graph is built. The
//! **row count is not**: a prefill step has many rows and a decode step has one,
//! so `rows` stays an unbound symbol and shape agreement between operations is
//! structural equality of the expressions, not of evaluated numbers.
//!
//! Anything that must divide exactly -- hidden by head count, rotary dimension
//! by two -- is checked at construction with `Dim::DivExact`, which returns
//! `DimError::NotDivisible` rather than truncating. Document 04: "Non-divisible
//! dimensions use checked padding or explicit unsupported combinations."

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{
    AccumulationPolicy, ActivationPrecision, Dim, Error, Precision, Result, SymbolId, SymbolTable,
    WeightPrecision,
};

use crate::{
    AttentionOutputReduction, KvHeadPartition, MlaAttentionDescriptor, Op, OpContract, OracleId,
    OracleRegistry, PartitionRule, StateEffect, Visibility,
};

/// Process-unique identity of one immutable validated graph.
///
/// The field is private: callers can retain and compare an identity, but only a
/// successful [`GraphBuilder::finish`] can create one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphId(u64);

impl GraphId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for GraphId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GraphId({})", self.0)
    }
}

#[derive(Debug)]
struct GraphIdAllocator {
    next: AtomicU64,
}

impl GraphIdAllocator {
    const fn new(next: u64) -> Self {
        Self {
            next: AtomicU64::new(next),
        }
    }

    fn allocate(&self) -> Result<GraphId> {
        let mut current = self.next.load(Ordering::Relaxed);
        loop {
            if current == 0 || current == u64::MAX {
                return Err(Error::InvalidRequest {
                    field: "graph_id",
                    detail: "process graph identity space is exhausted".into(),
                });
            }
            match self.next.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(GraphId(current)),
                Err(observed) => current = observed,
            }
        }
    }
}

static GRAPH_IDS: GraphIdAllocator = GraphIdAllocator::new(1);

/// A value flowing through the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// One operation node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// The stored integer representation of an index-role value.
///
/// This is separate from floating activation and quantized-weight precision.
/// The shared host reference stores indices as `u64`, which is the one encoding
/// supported until a kernel contract adds and validates another representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexEncoding {
    /// Unsigned 64-bit indices, matching the shared host reference value.
    ///
    /// Token ids, positions and page indices. The one encoding a *step input*
    /// may use, because `Value::Index` stores `u64` and a caller must not be
    /// able to declare a narrower one than the interpreter reads.
    U64,
    /// Unsigned 32-bit indices.
    ///
    /// Added for expert ids, which is where a declared encoding first stopped
    /// matching a stored one: a route table stores `u32` ids, so declaring
    /// `U64` made the resource plan charge twelve bytes per entry for eight.
    /// That over-charges rather than under-charges, which is why it is a
    /// contract defect rather than an overflow -- and the contract is what
    /// residency and transfer code will consume, so it has to be true now.
    U32,
}

impl IndexEncoding {
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::U64 => 8,
            Self::U32 => 4,
        }
    }

    /// Whether a value a caller supplies per step may declare this encoding.
    ///
    /// Only `U64`, because `moxie_interp::Value::Index` stores `u64` and a
    /// declaration narrower than the storage makes every byte count downstream
    /// too small. This is a rule, not a note: the first version of `U32` said
    /// "only `U64` for step inputs" in a comment and enforced nothing, so a
    /// graph declaring three `U32` token ids lowered to a 12-byte requirement
    /// for 24 bytes of storage -- trading the route table's four-byte
    /// *over*-count for an eight-byte *under*-count, which is the direction
    /// that actually corrupts.
    pub const fn is_legal_external_input(self) -> bool {
        matches!(self, Self::U64)
    }
}

/// What kind of thing a value is.
///
/// Document 02: "Token IDs, positions, page/group/sparse indices and masks also
/// need integer/boolean descriptor roles; they are not quantized weights or
/// floating activations. Capability validation checks each operand's role, shape
/// and dtype, not one universal precision allowlist."
///
/// The three roles carry three different precision types, so the checks cannot
/// be confused for one another: a weight dtype does not typecheck where an
/// activation dtype is required, and neither typechecks where an index is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueRole {
    /// A stored model parameter.
    Weight(WeightPrecision),
    /// An activation flowing between operations.
    Activation(ActivationPrecision),
    /// Token ids, positions, page or expert indices. Integer, never quantised.
    Index(IndexEncoding),
    /// One row's routing decision: the expert ids it selected and the
    /// coefficient each was selected with.
    ///
    /// Deliberately **not** two values and deliberately not one float tensor.
    /// Document 02 requires index roles to be separate from activations
    /// precisely so that "page/group/sparse indices" cannot be validated by a
    /// precision rule meant for floats; an expert id is such an index, and a
    /// route table that stored its ids as BF16 would silently alias expert 257
    /// onto expert 256. The two halves therefore carry their own descriptors
    /// and travel together, because a coefficient without the id it belongs to
    /// is not a routing decision.
    Route {
        index: IndexEncoding,
        coefficient: ActivationPrecision,
    },
}

impl ValueRole {
    pub fn is_index(self) -> bool {
        matches!(self, ValueRole::Index(_))
    }

    /// Whether this value is a routing decision rather than a tensor.
    pub fn is_route(self) -> bool {
        matches!(self, ValueRole::Route { .. })
    }

    /// The stored element encoding, for a float role.
    ///
    /// `None` for a route table as well as for an index: a route has *two*
    /// encodings, and returning either one of them here would let a caller
    /// validate a route against a single precision and believe it had checked
    /// the whole value.
    pub fn precision(self) -> Option<Precision> {
        match self {
            ValueRole::Weight(w) => Some(w.get()),
            ValueRole::Activation(a) => Some(a.get()),
            ValueRole::Index(_) | ValueRole::Route { .. } => None,
        }
    }
}

/// A value's role and shape.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub role: ValueRole,
    pub shape: Vec<Dim>,
}

impl TensorSpec {
    pub fn new(role: ValueRole, shape: Vec<Dim>) -> Self {
        Self { role, shape }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Evaluate the shape. `rows` and any other unbound symbol must be in
    /// `bindings`; a missing one is `DimError::Unbound`, never a default.
    pub fn extent(&self, bindings: &SymbolTable) -> Result<Vec<u64>> {
        self.shape
            .iter()
            .map(|d| d.eval(bindings).map_err(Error::from))
            .collect()
    }
}

/// The closed parameter set of an operation.
///
/// Closed on purpose: document 02 forbids "an opaque whole-model custom op or
/// arbitrary backend callback", and a parameter that is a function pointer is
/// the same escape hatch wearing a different hat.
#[derive(Debug, Clone, PartialEq)]
pub enum OpParams {
    Embedding {
        vocab: u64,
        hidden: u64,
        /// The factor every looked-up row is multiplied by, at the BF16
        /// boundary. Gemma scales by `bf16(sqrt(hidden))`; most families use
        /// 1.0. It is explicit for the same reason `eps` is: a default here is
        /// a silent numerical difference between two checkpoints.
        scale: f32,
    },
    Linear {
        in_features: u64,
        out_features: u64,
        bias: bool,
    },
    /// `eps` is explicit and stored. Document 02 requires "explicit epsilon,
    /// axes, pre/post scaling"; a defaulted epsilon is a silent numerical
    /// difference between two checkpoints that declared different ones.
    RmsNorm {
        /// The full width of the tensor's last axis.
        hidden: u64,
        /// How many independent normalizations each row is divided into.
        ///
        /// `1` is the ordinary whole-row norm. `heads` is per-head Q/K/V
        /// normalization: each contiguous `hidden / group` lanes are normalized
        /// on their own and share one gain of that width, which is what
        /// `src/models/gemma4/gemma4_runtime.cpp:798` does by looping heads.
        ///
        /// This is not the same operation with a different shape. Reducing over
        /// the whole width instead would mix every head's magnitude into every
        /// other head's scale factor -- an ordinary-looking graph that is wrong
        /// at every layer.
        group: u64,
        eps: f32,
    },
    SwiGlu {
        width: u64,
    },
    /// Gated GELU with the tanh approximation, `bf16(bf16(gelu_tanh(g)) * u)`.
    ///
    /// Distinct from [`OpParams::SwiGlu`] rather than a parameter on it:
    /// document 02 lists them as separate activations, and the gate transform
    /// is the whole of the difference.
    GeGlu {
        width: u64,
    },
    Rope {
        heads: u64,
        head_dim: u64,
        /// How many of a head's elements rotate. May be smaller than
        /// `head_dim`; the remainder passes through unchanged.
        rotary_dim: u64,
        /// The denominator of the inverse-frequency exponent,
        /// `base^(-2j / frequency_dim)`.
        ///
        /// Separate from `rotary_dim` because the two conventions disagree when
        /// rotation is partial: task 0003's fixtures divide by the rotated
        /// width, and Gemma 4's global layers divide by the **full** head
        /// dimension while rotating only a quarter of it. Collapsing them
        /// would silently change every angle on a partial-rotary layer.
        frequency_dim: u64,
        base: f32,
        layout: RopeLayout,
    },
    Attention {
        heads: u64,
        /// Key/value heads. Equal to `heads` for multi-head attention; a
        /// divisor of it for grouped-query attention, where query head `h`
        /// reads key/value head `h * kv_heads / heads`.
        kv_heads: u64,
        head_dim: u64,
        /// The factor applied to the raw score dot product.
        ///
        /// Usually `1/sqrt(head_dim)`, but not always: Gemma 4 normalizes its
        /// queries and keys per head and then attends with a scale of exactly
        /// 1.0. Baking the reciprocal square root in would make every such
        /// layer quietly wrong, so the scale is stated rather than derived.
        scale: f32,
        visibility: Visibility,
        /// Which KV store this node reads and appends to.
        layer: u32,
    },
    /// The unabsorbed MLA reference composition.
    ///
    /// The operands are `(hidden, positions, q_a_proj, q_a_layernorm,
    /// q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj)`.
    /// The descriptor keeps the projection, RoPE and latent-cache geometry on
    /// the node; the host interpreter dispatches each stage to the shared MLA
    /// oracle rather than inventing a second algebra or an absorbed path.
    MlaAttention {
        descriptor: MlaAttentionDescriptor,
    },
    Residual {
        /// The factor applied after the sum, at the BF16 boundary:
        /// `bf16(bf16(a + b) * scale)`.
        ///
        /// Per-residual, not per-layer: Gemma 4 scales its MLP residual by a
        /// checkpoint scalar and leaves its attention residual alone, so one
        /// shared value would be wrong on half the residuals in the graph.
        scale: f32,
    },
    VocabProjection {
        vocab: u64,
        hidden: u64,
        /// Logit soft capping, `bf16(bf16(tanh(bf16(bf16(x)/c))) * c)`.
        ///
        /// `None` means no cap. It is an option rather than a sentinel value
        /// because there is no cap magnitude that means "uncapped": a very
        /// large `c` still rounds and still costs a tanh.
        softcap: Option<f32>,
    },
    /// Which experts a row is sent to, and with what coefficients.
    ///
    /// Inputs are `(rows, router projection[, router gain][, per-expert
    /// scale][, selection bias])` and the output is a route table of
    /// `[rows, top_k]`. The optional operands appear in exactly that order and
    /// **two of them have the same shape**: `per_expert_scale` and
    /// `selection_bias` are both `[experts]`, so binding them the wrong way
    /// round is not caught by shape validation. `graph::route_operands` is the
    /// one place that order is written down, and
    /// `swapping_the_two_expert_vectors_changes_the_answer` in the interpreter's
    /// reference graphs is the substitution test that makes the distinction
    /// load-bearing rather than documented. The whole score
    /// transformation lives here rather than being composed from a norm and a
    /// linear, because document 02's `RouteSpec` makes "router score
    /// transformation, top-k/group selection, renormalization ... biases and
    /// scaling location" parameters of *routing*: the transform decides which
    /// expert weights the step needs, so it is a residency decision as much as
    /// a numerical one and cannot be left implicit in whatever a graph author
    /// happened to wire in front of it. The unfused stages stay separately
    /// callable in `moxie_oracles::route`, which is what document 09 requires
    /// of anything that could have been fused.
    ///
    /// The tie rule is not a parameter: **the lower expert id always wins**.
    /// Two ranks that broke a tie differently would route one row to two
    /// different experts, which is a residency divergence as well as a
    /// numerical one.
    Route {
        hidden: u64,
        experts: u64,
        top_k: u64,
        /// What the router does to the row before projecting it.
        ///
        /// A parameter rather than a fixed prologue because the two families
        /// that consume this operation disagree: Gemma 4's router owns a
        /// scale-free normalization, a trained gain and a scalar; Laguna's
        /// projects the block's `post_attention_layernorm` output unchanged.
        /// Composing Gemma's prologue out of separate nodes would move a
        /// residency decision into whatever a graph author happened to wire in
        /// front of the router, which is what this operation exists to prevent.
        input: RouterInput,
        /// How logits become scores.
        ///
        /// Softmax couples the experts; sigmoid does not. They are different
        /// functions rather than one function with a flag, and they select
        /// differently once a bias is added, because a bias shifts a sigmoid
        /// score by a different amount at every logit.
        score: RouteScore,
        /// Whether a per-expert coefficient scale is bound as an operand.
        ///
        /// Applied **after** renormalization and never renormalized away, so
        /// the coefficients of a scaled router do not sum to one. A combine
        /// that normalized them again would delete a trained parameter.
        per_expert_scale: bool,
        /// Whether a per-expert additive bias is bound as an operand.
        ///
        /// It moves **selection only**: the coefficients are gathered from the
        /// unbiased scores. Laguna's `e_score_correction_bias` is the
        /// auxiliary-loss-free load balancing of arXiv:2408.15664, and a router
        /// that gathered the biased score would still produce a well-formed
        /// distribution -- which is why this is a parameter with its own
        /// fixture rather than an implementation detail.
        selection_bias: bool,
        /// The precision the coefficients are narrowed to **as they leave the
        /// router**.
        ///
        /// The two pinned references differ, and the difference is the last
        /// statement of each router's `forward`:
        /// `Gemma4TextRouter` returns `top_k_weights` as the softmax and the
        /// per-expert scale produced it, while `LagunaTopKRouter` ends with
        /// `routing_weights = routing_weights.to(hidden_states.dtype)`, a BF16
        /// narrowing. On the logits `[0, 1]` that is `[0.59375, 0.40625]`
        /// against `[0.5938455, 0.4061545]`, and every combined row downstream
        /// carries the difference.
        ///
        /// **This is a statement about values, not about storage.** A route
        /// table holds FP32 coefficients either way and
        /// [`OpParams::output_role`] still says `F32`, because that role is
        /// what the byte trace M2's exit gate reconciles against the ledger --
        /// the same reason the index encoding there says `U32`. Narrowing the
        /// value and narrowing the buffer are different claims, and only the
        /// first one is the reference's.
        coefficient: RouteCoefficient,
    },
    /// The gated expert feed-forward, evaluated per selected slot.
    ///
    /// Inputs are `(rows, route, fused gate/up, fused down)`; the output is
    /// `[rows * top_k, hidden]`, slot-major, with slot `j` of row `r` at index
    /// `r * top_k + j`. Emitting per-slot outputs instead of an already
    /// combined row is what keeps [`OpParams::Combine`] independently testable.
    ///
    /// The expert tensors are **fused across experts**: `[experts, 2 * intermediate,
    /// hidden]` and `[experts, hidden, intermediate]`, which is how the
    /// designated artifact stores all 128 experts of a layer in two tensors.
    /// Within the gate/up tensor the output axis is the whole gate block
    /// followed by the whole up block -- `chunk(2, dim=-1)` in the pinned
    /// reference -- not interleaved pairs.
    ExpertMlp {
        hidden: u64,
        /// One expert's intermediate width. Distinct from a dense MLP's: the
        /// designated artifact's are 704 and 2112, and a layer carries both.
        intermediate: u64,
        experts: u64,
        top_k: u64,
        activation: ExpertActivation,
    },
    /// The weighted sum of a row's selected expert outputs.
    ///
    /// Inputs are `(route, slots)`; the output is `[rows, hidden]`.
    Combine {
        hidden: u64,
        top_k: u64,
        /// The order the `top_k` terms are added in.
        ///
        /// A parameter rather than a scheduling detail because floating-point
        /// addition is not associative: two orders give two different answers
        /// at any precision, so an executor that reduced in completion order
        /// would produce a result no oracle predicts.
        order: CombineOrder,
        /// A scalar applied to the **combined** row, `bf16(Σ terms · scale)`.
        ///
        /// Laguna's `moe_routed_scaling_factor` of 2.5 multiplies the summed
        /// routed output before the shared expert is added to it.
        /// `Residual { scale }` cannot express that -- it is
        /// `bf16(bf16(a + b) · scale)`, which scales the shared expert too --
        /// and scaling each term before summing is a different rounding
        /// pattern. A router without one passes 1.0.
        output_scale: f32,
    },
}

/// The gate transform of an expert's gated feed-forward.
///
/// The same distinction [`OpParams::SwiGlu`] and [`OpParams::GeGlu`] draw for
/// dense activations, carried into routed experts because the two consumers of
/// this operation genuinely differ: the Gemma 4 family declares
/// `gelu_pytorch_tanh`, and a family with a SiLU gate is not the same function
/// with a flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertActivation {
    /// `gelu_tanh(gate) * up`.
    GeGlu,
    /// `silu(gate) * up`.
    SwiGlu,
}

/// A router's operand list: at most five, held inline.
///
/// Not a `Vec`. `OpParams::route_operands` is called by the **interpreter**,
/// once per `Route` node per step, and an infallible heap allocation inside a
/// generation step aborts the process on failure instead of returning a typed
/// error the transaction can roll back. This module's `softmax` carried that
/// defect once and had it fixed; a review found it again in
/// `narrow_coefficients`; applying the review's own question to the rest of the
/// same change found it here. The list is bounded by construction, so the fix
/// is to have nothing to allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteOperands {
    operands: [RouteOperand; Self::MAX],
    len: usize,
}

impl RouteOperands {
    /// Rows, projection, gain, per-expert scale, selection bias.
    pub const MAX: usize = 5;

    fn new() -> Self {
        Self {
            operands: [RouteOperand::Rows; Self::MAX],
            len: 0,
        }
    }

    fn push(&mut self, operand: RouteOperand) {
        // Unreachable by construction: there are five distinct operands and
        // each is pushed at most once. A `debug_assert` rather than a silent
        // truncation, because a truncated operand list would be a wrong arity
        // that every shape check would then agree with.
        debug_assert!(
            self.len < Self::MAX,
            "a router has at most {} operands",
            Self::MAX
        );
        if self.len < Self::MAX {
            self.operands[self.len] = operand;
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[RouteOperand] {
        &self.operands[..self.len]
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The precision a router narrows its coefficients to before emitting them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteCoefficient {
    /// Emitted as computed. `Gemma4TextRouter.forward` has no cast.
    Fp32,
    /// Narrowed to BF16. `LagunaTopKRouter.forward`'s last statement is
    /// `routing_weights.to(hidden_states.dtype)`, and the model dtype is BF16.
    Bf16,
}

/// One operand of [`OpParams::Route`], in the order a router takes them.
///
/// This exists because two of them -- [`RouteOperand::PerExpertScale`] and
/// [`RouteOperand::SelectionBias`] -- are both `[experts]`, so a graph that
/// bound them the wrong way round would pass every shape check and return a
/// confident, different answer. The fourth review of task 0021 found the same
/// shape of defect in a kernel descriptor: a check that is half an identity is
/// a check of something else. The answer there was to make the identity a
/// binding, and the answer here is that there is exactly **one** statement of
/// the order and every other part of the crate derives from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteOperand {
    /// The rows to route, `[rows, hidden]`.
    Rows,
    /// The expert projection, `[experts, hidden]`.
    Projection,
    /// The router's own gain, `[hidden]`. Only for a `Normalized` router.
    Gain,
    /// The per-expert coefficient scale, `[experts]`.
    PerExpertScale,
    /// The per-expert selection bias, `[experts]`.
    SelectionBias,
}

/// What a router does to a row before projecting it to expert logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RouterInput {
    /// `bf16(bf16(bf16(x / rms(x)) · gain) · input_scale)`, with the trained
    /// gain bound as an operand.
    ///
    /// Gemma 4's router: `Gemma4RMSNorm(..., with_scale=False)` followed by
    /// `router.scale` and `scalar_root_size`. The three BF16 boundaries are
    /// part of the equation; dropping them changes the *selected experts* on
    /// roughly one row in 270.
    Normalized {
        /// Added to the mean square before the reciprocal square root.
        eps: f32,
        /// The scalar multiplying the normalized, gained row. Gemma 4 uses
        /// `hidden^(-1/2)`; it is stated rather than derived because it is a
        /// family choice and not an identity.
        input_scale: f32,
    },
    /// The row is projected as given, with no gain operand.
    ///
    /// Laguna's router: `F.linear(hidden_states, self.weight)` over the block's
    /// already-normalized stream. Passing `Normalized` with a unit gain would
    /// not be the same function -- it would normalize a second time.
    Raw,
}

/// How a router turns logits into per-expert scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteScore {
    /// `softmax(logits)` over the whole expert set, in FP32.
    Softmax,
    /// `sigmoid(logit)` per expert, independently, in FP32.
    ///
    /// Not softmax with a flag: the scores do not sum to one before
    /// renormalization, and adding a selection bias moves each score by an
    /// amount that depends on its own logit.
    Sigmoid,
}

/// The order a routed row's expert contributions are summed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombineOrder {
    /// Ascending expert id, regardless of the order the row selected them.
    ///
    /// What the pinned Gemma 4 reference does: `Gemma4TextExperts.forward`
    /// iterates `expert_hit`, which is `nonzero()` over an expert-major mask,
    /// and accumulates into the output with `index_add_`.
    AscendingExpertId,
    /// The row's own selection order, highest score first.
    SelectionOrder,
}

/// Which elements of a head RoPE pairs together.
///
/// Both conventions appear in released checkpoints and they are not
/// interchangeable. R06 is the standing instruction against erasing exactly
/// this kind of difference behind a common default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeLayout {
    /// Adjacent pairs, `(2j, 2j+1)`.
    Interleaved,
    /// Halves paired across the head, `(j, j + head_dim/2)`.
    HalfSplit,
}

impl OpParams {
    pub fn op(&self) -> Op {
        match self {
            OpParams::Embedding { .. } => Op::Embedding,
            OpParams::Linear { .. } => Op::Linear,
            OpParams::RmsNorm { .. } => Op::RmsNorm,
            OpParams::SwiGlu { .. } => Op::SwiGlu,
            OpParams::GeGlu { .. } => Op::GeGlu,
            OpParams::Rope { .. } => Op::Rope,
            OpParams::Attention { .. } => Op::Attention,
            OpParams::MlaAttention { .. } => Op::MlaAttention,
            OpParams::Residual { .. } => Op::Residual,
            OpParams::VocabProjection { .. } => Op::VocabProjection,
            OpParams::Route { .. } => Op::Route,
            OpParams::ExpertMlp { .. } => Op::ExpertMlp,
            OpParams::Combine { .. } => Op::Combine,
        }
    }

    /// The partition rule from task 0003's contract table.
    ///
    /// Encoded here rather than left to each caller, because document 04 makes
    /// partition legality a property of the operation. `NotDetermined` fails
    /// closed for TP lowering, which is what makes M5 a lowering rather than a
    /// rewrite.
    pub fn partition_rule(&self) -> PartitionRule {
        match self {
            // Elementwise or output-channel sharded.
            OpParams::Linear { .. }
            | OpParams::SwiGlu { .. }
            | OpParams::GeGlu { .. }
            | OpParams::VocabProjection { .. } => PartitionRule::ColumnShardable,
            // Ordinary norms reduce over the whole hidden axis and therefore
            // run once. Grouped norms reduce each head independently and are
            // legal only when the lowering owns whole heads.
            OpParams::Embedding { .. } | OpParams::Residual { .. } => PartitionRule::Replicated,
            OpParams::RmsNorm { group, .. } => {
                if *group == 1 {
                    PartitionRule::Replicated
                } else {
                    PartitionRule::HeadAligned
                }
            }
            // Rotation is independent per head, but its layout is head-sized,
            // not an arbitrary output-column split.
            OpParams::Rope { .. } => PartitionRule::HeadAligned,
            // Plain attention owns query heads and concatenates their outputs;
            // its separate output Linear keeps its existing contract.
            OpParams::Attention { .. } => PartitionRule::HeadShardable {
                kv: KvHeadPartition::GqaReplicateWhenOversubscribed,
                output: AttentionOutputReduction::ConcatenateHeads,
            },
            // MLA owns query heads too, but its latent/positional cache is
            // shared rather than per-head and its descriptor includes o_proj,
            // so the operation boundary includes the global output reduction.
            OpParams::MlaAttention { .. } => PartitionRule::HeadShardable {
                kv: KvHeadPartition::SharedLatentReplicated,
                output: AttentionOutputReduction::GlobalReduction,
            },
            // Replicated, and that is a correctness requirement rather than a
            // cost choice. Every rank must reach the same selection from the
            // same row: a router sharded over its expert axis would reduce
            // partial logits in a rank-dependent order, and two ranks that
            // disagree about which expert a row needs disagree about which
            // weights have to be resident. The router is three small tensors,
            // so replicating them costs almost nothing.
            OpParams::Route { .. } => PartitionRule::Replicated,
            // Whole experts are assigned to one rank. The lowering owns the
            // coupled ExpertMlp/Combine stage and declares its ordered sum.
            OpParams::ExpertMlp { .. } | OpParams::Combine { .. } => {
                PartitionRule::ExpertOwnerShardable
            }
        }
    }

    /// What this operation does to sequence state.
    pub fn state_effect(&self) -> StateEffect {
        match self {
            OpParams::Attention { .. } | OpParams::MlaAttention { .. } => StateEffect::Appends,
            _ => StateEffect::None,
        }
    }

    /// The precision of this operation's output.
    ///
    /// BF16 everywhere except the vocabulary projection, whose logits stay FP32:
    /// document 05 builds the sampler's pre-truncation normalizer from them and
    /// exact speculative verification consumes that distribution, so rounding
    /// them first would change it.
    pub fn output_precision(&self) -> ActivationPrecision {
        match self {
            OpParams::VocabProjection { .. } => ActivationPrecision::expect(Precision::F32),
            _ => ActivationPrecision::expect(Precision::Bf16),
        }
    }

    /// The role of this operation's output value.
    ///
    /// Everything produces an activation except [`OpParams::Route`], whose
    /// output is a routing decision: integer expert ids beside float
    /// coefficients. Typing it as an activation would make the ids floats.
    pub fn output_role(&self) -> ValueRole {
        match self {
            OpParams::Route { .. } => ValueRole::Route {
                // `u32`, because that is what `RouteTable` stores. Declaring
                // `U64` here and storing `u32` there made the two disagree by
                // four bytes an entry in every byte trace M2's exit gate asks
                // to reconcile against the ledger.
                index: IndexEncoding::U32,
                // The coefficients stay FP32 for the same reason the vocabulary
                // projection's logits do: they are a distribution's tail, and
                // rounding eight renormalized probabilities to BF16 before they
                // weight anything loses more than the expert outputs they
                // multiply ever recover.
                coefficient: ActivationPrecision::expect(Precision::F32),
            },
            _ => ValueRole::Activation(self.output_precision()),
        }
    }

    /// The operands this router takes, in order.
    ///
    /// The single statement of [`OpParams::Route`]'s input list: the arity, the
    /// shape validation and every consumer read it rather than repeating it.
    /// Empty for every other operation.
    ///
    /// Allocates nothing -- see [`RouteOperands`] for why that is a
    /// requirement here and not a micro-optimisation.
    pub fn route_operands(&self) -> RouteOperands {
        let mut operands = RouteOperands::new();
        let OpParams::Route {
            input,
            per_expert_scale,
            selection_bias,
            ..
        } = self
        else {
            return operands;
        };
        operands.push(RouteOperand::Rows);
        operands.push(RouteOperand::Projection);
        if matches!(input, RouterInput::Normalized { .. }) {
            operands.push(RouteOperand::Gain);
        }
        if *per_expert_scale {
            operands.push(RouteOperand::PerExpertScale);
        }
        if *selection_bias {
            operands.push(RouteOperand::SelectionBias);
        }
        operands
    }

    /// How many value inputs this operation takes.
    pub fn arity(&self) -> usize {
        match self {
            OpParams::Embedding { .. } => 2, // tokens, table
            OpParams::Linear { bias, .. } => 2 + usize::from(*bias),
            OpParams::RmsNorm { .. } => 2,      // x, gain
            OpParams::SwiGlu { .. } => 2,       // gate, up
            OpParams::GeGlu { .. } => 2,        // gate, up
            OpParams::Rope { .. } => 2,         // x, positions
            OpParams::Attention { .. } => 4,    // q, k, v, positions
            OpParams::MlaAttention { .. } => 9, // hidden, positions, seven MLA weights
            OpParams::Residual { .. } => 2,
            OpParams::VocabProjection { .. } => 2, // hidden, table
            // `route_operands` is the one statement of which operands a
            // router takes and in what order; the arity is derived from it so
            // the two cannot drift apart.
            OpParams::Route { .. } => self.route_operands().len(),
            OpParams::ExpertMlp { .. } => 4, // rows, route, gate/up, down
            OpParams::Combine { .. } => 2,   // route, slots
        }
    }

    /// Structural checks that need no inputs: the divisibilities and bounds this
    /// operation's own parameters must satisfy.
    ///
    /// Uses `Dim::DivExact`, so a head count that does not divide the hidden
    /// width is `NotDivisible` at construction rather than a truncation that
    /// surfaces as a wrong answer.
    pub fn check_params(&self) -> Result<()> {
        let empty = SymbolTable::new();
        match *self {
            OpParams::Rope {
                heads,
                head_dim,
                rotary_dim,
                frequency_dim,
                base,
                layout,
            } => {
                if heads == 0 || head_dim == 0 {
                    return Err(Error::InvalidRequest {
                        field: "rope",
                        detail: format!("{heads} heads of dimension {head_dim}"),
                    });
                }
                // The rotated width participates in a shape; a product that
                // wraps becomes a buffer size.
                (Dim::constant(heads) * Dim::constant(head_dim)).eval(&empty)?;
                if rotary_dim > head_dim {
                    return Err(Error::InvalidRequest {
                        field: "rotary_dim",
                        detail: format!("rotary_dim {rotary_dim} exceeds head_dim {head_dim}"),
                    });
                }
                // Rotation is over pairs, so a shard that split a pair would
                // rotate half of it. Exact division, not a floor.
                Dim::constant(rotary_dim).div_exact(2).eval(&empty)?;
                if frequency_dim == 0 {
                    return Err(Error::InvalidRequest {
                        field: "frequency_dim",
                        detail: "the inverse-frequency denominator cannot be zero".into(),
                    });
                }
                // Half-split pairing reads `x[j + head_dim/2]`, so a head with
                // an odd dimension has no partner for its middle element. The
                // interleaved form does not care, which is why this check is
                // here and not beside the `rotary_dim` one.
                if layout == RopeLayout::HalfSplit {
                    Dim::constant(head_dim).div_exact(2).eval(&empty)?;
                }
                if !(base.is_finite() && base > 1.0) {
                    return Err(Error::InvalidRequest {
                        field: "rope_base",
                        detail: format!("base must be finite and > 1, got {base}"),
                    });
                }
                Ok(())
            }
            OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                scale,
                ..
            } => {
                if heads == 0 || head_dim == 0 {
                    return Err(Error::InvalidRequest {
                        field: "attention",
                        detail: format!("{heads} heads of dimension {head_dim}"),
                    });
                }
                // Grouped query attention maps query head `h` onto key/value
                // head `h * kv_heads / heads`. A ratio that does not divide
                // gives some group one more query head than another, which is
                // not a layout any released checkpoint uses and not one this
                // reference will silently invent.
                if kv_heads == 0 || kv_heads > heads || !heads.is_multiple_of(kv_heads) {
                    return Err(Error::InvalidRequest {
                        field: "kv_heads",
                        detail: format!(
                            "{kv_heads} key/value heads must be a nonzero divisor of \
                             {heads} query heads"
                        ),
                    });
                }
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(Error::InvalidRequest {
                        field: "attention_scale",
                        detail: format!("score scale must be finite and positive, got {scale}"),
                    });
                }
                Ok(())
            }
            OpParams::MlaAttention { descriptor } => descriptor.validate(),
            OpParams::Embedding { scale, .. } => {
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(Error::InvalidRequest {
                        field: "embedding_scale",
                        detail: format!("embedding scale must be finite and positive, got {scale}"),
                    });
                }
                Ok(())
            }
            OpParams::Residual { scale } => {
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(Error::InvalidRequest {
                        field: "residual_scale",
                        detail: format!("residual scale must be finite and positive, got {scale}"),
                    });
                }
                Ok(())
            }
            OpParams::VocabProjection {
                softcap: Some(cap), ..
            } => {
                if !(cap.is_finite() && cap > 0.0) {
                    return Err(Error::InvalidRequest {
                        field: "logit_softcap",
                        detail: format!("logit softcap must be finite and positive, got {cap}"),
                    });
                }
                Ok(())
            }
            OpParams::RmsNorm { eps, hidden, group } => {
                if !(eps.is_finite() && eps > 0.0) {
                    return Err(Error::InvalidRequest {
                        field: "eps",
                        detail: format!("epsilon must be finite and positive, got {eps}"),
                    });
                }
                if hidden == 0 {
                    return Err(Error::InvalidRequest {
                        field: "hidden",
                        detail: "a norm over zero features".into(),
                    });
                }
                if group == 0 {
                    return Err(Error::InvalidRequest {
                        field: "group",
                        detail: "a norm over zero groups".into(),
                    });
                }
                // A group count that does not divide the width would leave a
                // remainder of lanes normalized against nothing.
                Dim::constant(hidden).div_exact(group).eval(&empty)?;
                Ok(())
            }
            OpParams::Route {
                hidden,
                experts,
                top_k,
                input,
                ..
            } => {
                if hidden == 0 {
                    return Err(Error::InvalidRequest {
                        field: "hidden",
                        detail: "a router over zero features".into(),
                    });
                }
                if experts == 0 {
                    return Err(Error::InvalidRequest {
                        field: "experts",
                        detail: "a router over zero experts".into(),
                    });
                }
                // `top_k == experts` is legal and is a real case: it keeps the
                // whole distribution and makes the renormalisation a no-op, so
                // it is the fixture that proves the selection did not silently
                // drop anything. `top_k > experts` is not legal anywhere.
                if top_k == 0 || top_k > experts {
                    return Err(Error::InvalidRequest {
                        field: "top_k",
                        detail: format!("top_k {top_k} outside 1..={experts}"),
                    });
                }
                // The route table's extent is `rows * top_k`, and the slot
                // tensor's is `rows * top_k * hidden`.
                (Dim::constant(top_k) * Dim::constant(hidden)).eval(&empty)?;
                (Dim::constant(experts) * Dim::constant(hidden)).eval(&empty)?;
                if let RouterInput::Normalized { eps, input_scale } = input {
                    if !(eps.is_finite() && eps > 0.0) {
                        return Err(Error::InvalidRequest {
                            field: "eps",
                            detail: format!("epsilon must be finite and positive, got {eps}"),
                        });
                    }
                    if !(input_scale.is_finite() && input_scale > 0.0) {
                        return Err(Error::InvalidRequest {
                            field: "router_input_scale",
                            detail: format!(
                                "router input scale must be finite and positive, got {input_scale}"
                            ),
                        });
                    }
                }
                Ok(())
            }
            OpParams::ExpertMlp {
                hidden,
                intermediate,
                experts,
                top_k,
                ..
            } => {
                if hidden == 0 || intermediate == 0 {
                    return Err(Error::InvalidRequest {
                        field: "expert_mlp",
                        detail: format!("hidden {hidden}, intermediate {intermediate}"),
                    });
                }
                if experts == 0 || top_k == 0 || top_k > experts {
                    return Err(Error::InvalidRequest {
                        field: "top_k",
                        detail: format!("top_k {top_k} outside 1..={experts}"),
                    });
                }
                // Both fused extents, checked here because they become buffer
                // sizes: `experts * 2 * intermediate * hidden` is the largest
                // tensor in any routed layer and is the one a caller's own
                // arithmetic overflows first.
                (Dim::constant(experts)
                    * Dim::constant(2)
                    * Dim::constant(intermediate)
                    * Dim::constant(hidden))
                .eval(&empty)?;
                (Dim::constant(experts) * Dim::constant(hidden) * Dim::constant(intermediate))
                    .eval(&empty)?;
                (Dim::constant(top_k) * Dim::constant(hidden)).eval(&empty)?;
                Ok(())
            }
            OpParams::Combine {
                hidden,
                top_k,
                output_scale,
                ..
            } => {
                if hidden == 0 || top_k == 0 {
                    return Err(Error::InvalidRequest {
                        field: "combine",
                        detail: format!("hidden {hidden}, top_k {top_k}"),
                    });
                }
                // Finite, and that is the whole rule. Zero and negative are
                // legal: this is a checkpoint scalar, not a probability, and
                // the same reasoning `apply_per_expert_scale` records applies.
                if !output_scale.is_finite() {
                    return Err(Error::InvalidRequest {
                        field: "combine_output_scale",
                        detail: format!("combine output scale must be finite, got {output_scale}"),
                    });
                }
                (Dim::constant(top_k) * Dim::constant(hidden)).eval(&empty)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Which input, if any, an operation reads positions from.
fn position_operand(params: &OpParams) -> Option<usize> {
    match params {
        OpParams::Rope { .. } => Some(1),
        OpParams::Attention { .. } => Some(3),
        OpParams::MlaAttention { .. } => Some(1),
        _ => None,
    }
}

/// One node: an operation, its parameters, its inputs and its output.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub params: OpParams,
    pub inputs: Vec<ValueId>,
    pub output: ValueId,
    pub contract: OpContract,
}

/// A validated graph.
///
/// Construction is the only way to get one, and construction validates, so a
/// `Graph` in hand is evidence that its shapes agree, its roles are right, its
/// state effects are declared and every operation has a registered oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    id: GraphId,
    values: Vec<TensorSpec>,
    names: BTreeMap<ValueId, String>,
    nodes: Vec<Node>,
    inputs: Vec<ValueId>,
    weights: Vec<ValueId>,
    output: ValueId,
    rows: SymbolId,
    positions: Option<ValueId>,
}

impl Graph {
    pub const fn id(&self) -> GraphId {
        self.id
    }

    /// Exact graph structure excluding process identity.
    ///
    /// A plan compares this together with [`GraphId`]. Identity catches an
    /// accidental graph substitution cheaply; structure prevents a bare ID
    /// from becoming authority if a future deserializer is defective.
    pub fn signature(&self) -> GraphSignature {
        GraphSignature {
            values: self.values.clone(),
            names: self.names.clone(),
            nodes: self.nodes.clone(),
            inputs: self.inputs.clone(),
            weights: self.weights.clone(),
            output: self.output,
            rows: self.rows,
            positions: self.positions,
        }
    }

    pub fn values(&self) -> &[TensorSpec] {
        &self.values
    }
    /// The single value every position-consuming node reads.
    ///
    /// One binding, enforced when the graph is built. Two nodes reading
    /// different position vectors is not a graph an executor can validate
    /// cheaply -- and the fourth review showed that validating only the first
    /// one lets the rest say anything they like.
    pub fn positions(&self) -> Option<ValueId> {
        self.positions
    }

    /// The symbol standing for this step's row count.
    ///
    /// Declared rather than inferred, so an executor can bind it and check the
    /// values it was handed against the shapes the graph declares. Without it
    /// the shapes would be structural decoration that nothing enforces at run
    /// time.
    pub fn rows_symbol(&self) -> SymbolId {
        self.rows
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn spec(&self, v: ValueId) -> Option<&TensorSpec> {
        self.values.get(v.0 as usize)
    }
    pub fn name(&self, v: ValueId) -> Option<&str> {
        self.names.get(&v).map(String::as_str)
    }
    /// Values supplied per step: tokens, positions.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }
    /// Values supplied once: the model's parameters.
    pub fn weights(&self) -> &[ValueId] {
        &self.weights
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// The KV layers this graph's attention nodes use, ascending.
    ///
    /// `finish` guarantees these are exactly `0..n`, so an executor can check a
    /// cache covers the graph by comparing one count.
    pub fn attention_layers(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .nodes
            .iter()
            .filter_map(|n| match n.params {
                OpParams::Attention { layer, .. } => Some(layer),
                OpParams::MlaAttention { descriptor } => Some(descriptor.layer),
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Every state kind this graph touches, so a caller can build the schema its
    /// sequence state needs rather than guessing.
    pub fn state_effects(&self) -> Vec<(NodeId, StateEffect)> {
        self.nodes
            .iter()
            .filter(|n| n.params.state_effect() != StateEffect::None)
            .map(|n| (n.id, n.params.state_effect()))
            .collect()
    }
}

/// Exact immutable graph summary retained by a resource plan.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSignature {
    values: Vec<TensorSpec>,
    names: BTreeMap<ValueId, String>,
    nodes: Vec<Node>,
    inputs: Vec<ValueId>,
    weights: Vec<ValueId>,
    output: ValueId,
    rows: SymbolId,
    positions: Option<ValueId>,
}

/// Builds and validates a graph.
#[derive(Debug)]
pub struct GraphBuilder {
    weight_precisions: BTreeMap<String, WeightPrecision>,
    values: Vec<TensorSpec>,
    names: BTreeMap<ValueId, String>,
    nodes: Vec<Node>,
    inputs: Vec<ValueId>,
    weights: Vec<ValueId>,
    oracle: OracleId,
    rows: SymbolId,
    positions: Option<ValueId>,
}

impl GraphBuilder {
    /// `oracle` is the reference every node in this graph will be validated
    /// against; it must be registered when `finish` is called. `rows` is the
    /// symbol standing for the row count, which is not known until a step runs.
    pub fn new(oracle: OracleId, rows: SymbolId) -> Self {
        Self {
            weight_precisions: BTreeMap::new(),
            values: Vec::new(),
            names: BTreeMap::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            weights: Vec::new(),
            oracle,
            rows,
            positions: None,
        }
    }

    /// Bind source-declared weight precisions by semantic role. Unknown roles
    /// are rejected at finish and consumers still enforce their own contracts.
    pub fn with_weight_precisions(mut self, precisions: BTreeMap<String, WeightPrecision>) -> Self {
        self.weight_precisions = precisions;
        self
    }

    fn add_value(&mut self, name: &str, spec: TensorSpec) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(spec);
        self.names.insert(id, name.to_string());
        id
    }

    /// A value supplied per step.
    ///
    /// Infallible, because the role it is handed is checked where a graph
    /// becomes valid rather than where a value is named: `finish` refuses a step
    /// input whose declared index encoding is narrower than the `u64` the host
    /// reference stores, and refuses a route table supplied from outside.
    pub fn input(&mut self, name: &str, spec: TensorSpec) -> ValueId {
        let id = self.add_value(name, spec);
        self.inputs.push(id);
        id
    }

    /// A model parameter, supplied once.
    pub fn weight(&mut self, name: &str, mut spec: TensorSpec) -> Result<ValueId> {
        if !matches!(spec.role, ValueRole::Weight(_)) {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{name} is declared as a weight but its role is {:?}",
                    spec.role
                )
                .into(),
            });
        }
        if let Some(precision) = self.weight_precisions.remove(name) {
            spec.role = ValueRole::Weight(precision);
        }
        let id = self.add_value(name, spec);
        self.weights.push(id);
        Ok(id)
    }

    fn spec(&self, v: ValueId) -> Result<&TensorSpec> {
        self.values.get(v.0 as usize).ok_or(Error::InvalidRequest {
            field: "value",
            detail: format!("value {} is not defined", v.0),
        })
    }

    /// Append an operation. Validates arity, roles and shapes before accepting.
    pub fn node(&mut self, params: OpParams, inputs: &[ValueId]) -> Result<ValueId> {
        params.check_params()?;
        if inputs.len() != params.arity() {
            return Err(Error::InvalidRequest {
                field: "inputs",
                detail: format!(
                    "{} takes {} input(s), got {}",
                    params.op().name(),
                    params.arity(),
                    inputs.len()
                ),
            });
        }
        for v in inputs {
            self.spec(*v)?;
        }
        // One attention node per KV layer. Two nodes sharing a layer index would
        // both append this step's keys to the same history, so the second would
        // attend over a history containing the first's rows twice.
        let state_layer = match params {
            OpParams::Attention { layer, .. } => Some(layer),
            OpParams::MlaAttention { descriptor } => Some(descriptor.layer),
            _ => None,
        };
        if let Some(layer) = state_layer
            && let Some(clash) = self.nodes.iter().find(|n| {
                matches!(
                    n.params,
                    OpParams::Attention { layer: l, .. } if l == layer
                ) || matches!(
                    n.params,
                    OpParams::MlaAttention { descriptor } if descriptor.layer == layer
                )
            })
        {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "attention layer {layer} is already used by node {}; two nodes sharing \
                     a KV layer would append the same positions twice",
                    clash.id.0
                )
                .into(),
            });
        }
        // Every operation that consumes positions must consume the *same* ones.
        // An executor validates that vector against the branch's frontier once;
        // if two nodes could read different vectors, only one of them would be
        // checked.
        if let Some(i) = position_operand(&params) {
            let v = inputs[i];
            match self.positions {
                Some(existing) if existing != v => {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "{} reads positions from value {} but this graph already uses \
                             value {}; every position-consuming operation must read one \
                             shared binding",
                            params.op().name(),
                            v.0,
                            existing.0
                        )
                        .into(),
                    });
                }
                Some(_) => {}
                None => self.positions = Some(v),
            }
        }
        let out_shape = self.check_shapes(&params, inputs)?;
        let out_spec = TensorSpec::new(params.output_role(), out_shape);
        let id = NodeId(self.nodes.len() as u32);
        let output = self.add_value(&format!("{}#{}", params.op().name(), id.0), out_spec);
        let contract = OpContract {
            op: params.op(),
            weights: if matches!(params.op(), Op::Linear | Op::ExpertMlp) {
                [Precision::Int4, Precision::Int8, Precision::Bf16]
                    .into_iter()
                    .map(WeightPrecision::expect)
                    .collect()
            } else {
                vec![WeightPrecision::expect(Precision::Bf16)]
            },
            activations: vec![ActivationPrecision::expect(Precision::Bf16)],
            output: params.output_precision(),
            accumulation: AccumulationPolicy::Bf16InF32Acc,
            workspace_upper_bound: Dim::constant(0),
            oracle: self.oracle,
            partition: params.partition_rule(),
            state_effect: params.state_effect(),
        };
        // An operand's declared precision must be one the operation's contract
        // accepts. Agreement between a binding and its declaration is not the
        // same as agreement with the node that consumes it: the fifth review
        // declared an input FP32 and fed it to a RoPE whose contract permits
        // only BF16 activations, and both construction and execution succeeded.
        for (i, v) in inputs.iter().enumerate() {
            let spec = self.spec(*v).expect("checked above");
            let allowed = match spec.role {
                // Both are validated structurally by `check_shapes`, which
                // knows which operand of which operation may be one. Neither
                // has a single precision for the contract's allowlist to check.
                ValueRole::Index(_) | ValueRole::Route { .. } => continue,
                ValueRole::Weight(w) => contract.weights.iter().any(|a| a.get() == w.get()),
                ValueRole::Activation(a) => contract.activations.iter().any(|x| x.get() == a.get()),
            };
            if !allowed {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "{} input {i} is declared {:?} but the operation's contract accepts \
                         weights {:?} and activations {:?}",
                        contract.op.name(),
                        spec.role,
                        contract
                            .weights
                            .iter()
                            .map(|w| w.get().name())
                            .collect::<Vec<_>>(),
                        contract
                            .activations
                            .iter()
                            .map(|a| a.get().name())
                            .collect::<Vec<_>>(),
                    )
                    .into(),
                });
            }
        }
        self.nodes.push(Node {
            id,
            params,
            inputs: inputs.to_vec(),
            output,
            contract,
        });
        Ok(output)
    }

    /// Per-operation shape and role rules. Returns the output shape.
    ///
    /// Shape agreement is structural equality of `Dim` expressions, because the
    /// row count is a symbol that is not bound until a step runs.
    fn check_shapes(&self, params: &OpParams, inputs: &[ValueId]) -> Result<Vec<Dim>> {
        let s = |i: usize| self.spec(inputs[i]).expect("checked above");
        let bad = |detail: String| -> Error {
            Error::InvalidArtifact {
                detail: format!("{}: {detail}", params.op().name()).into(),
            }
        };
        let want_index = |i: usize| -> Result<()> {
            if !s(i).role.is_index() {
                return Err(bad(format!(
                    "input {i} must be an index role, got {:?}",
                    s(i).role
                )));
            }
            Ok(())
        };
        let want_float = |i: usize| -> Result<()> {
            // A route table is not a float tensor even though half of it is
            // floating: it has no single precision, and letting it through here
            // would let a routed value be consumed as an activation.
            if s(i).role.is_index() || s(i).role.is_route() {
                return Err(bad(format!(
                    "input {i} must be a float role, got {:?}",
                    s(i).role
                )));
            }
            Ok(())
        };
        let want_weight = |i: usize| -> Result<()> {
            if !matches!(s(i).role, ValueRole::Weight(_)) {
                return Err(bad(format!(
                    "input {i} must be a weight role, got {:?}",
                    s(i).role
                )));
            }
            Ok(())
        };
        // Stricter than `want_float`: a weight passes that check too (Linear's
        // second operand, RmsNorm's gain and Embedding's table are legitimately
        // weights in a "float" slot), but query/key/value feed a per-step state
        // append, not a stored parameter, so only an activation belongs there.
        let want_activation = |i: usize| -> Result<()> {
            if !matches!(s(i).role, ValueRole::Activation(_)) {
                return Err(bad(format!(
                    "input {i} must be an activation role, got {:?}",
                    s(i).role
                )));
            }
            Ok(())
        };
        let want_route = |i: usize| -> Result<()> {
            if !s(i).role.is_route() {
                return Err(bad(format!(
                    "input {i} must be a route table, got {:?}",
                    s(i).role
                )));
            }
            Ok(())
        };
        let rank = |i: usize, r: usize| -> Result<()> {
            if s(i).rank() != r {
                return Err(bad(format!(
                    "input {i} must have rank {r}, got {}",
                    s(i).rank()
                )));
            }
            Ok(())
        };
        let dim_is = |i: usize, axis: usize, want: u64| -> Result<()> {
            let got = &s(i).shape[axis];
            if *got != Dim::constant(want) {
                return Err(bad(format!(
                    "input {i} axis {axis} must be {want}, got {got:?}"
                )));
            }
            Ok(())
        };

        Ok(match *params {
            OpParams::Embedding { vocab, hidden, .. } => {
                want_index(0)?;
                rank(0, 1)?;
                want_float(1)?;
                rank(1, 2)?;
                dim_is(1, 0, vocab)?;
                dim_is(1, 1, hidden)?;
                vec![s(0).shape[0].clone(), Dim::constant(hidden)]
            }
            OpParams::Linear {
                in_features,
                out_features,
                bias,
            } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, in_features)?;
                want_float(1)?;
                rank(1, 2)?;
                dim_is(1, 0, out_features)?;
                dim_is(1, 1, in_features)?;
                if bias {
                    want_float(2)?;
                    rank(2, 1)?;
                    dim_is(2, 0, out_features)?;
                }
                vec![s(0).shape[0].clone(), Dim::constant(out_features)]
            }
            OpParams::RmsNorm { hidden, group, .. } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, hidden)?;
                want_float(1)?;
                rank(1, 1)?;
                // One gain per group lane, shared by every group -- not one
                // gain per element of the row.
                dim_is(
                    1,
                    0,
                    Dim::constant(hidden)
                        .div_exact(group)
                        .eval(&SymbolTable::new())?,
                )?;
                s(0).shape.clone()
            }
            OpParams::SwiGlu { width } | OpParams::GeGlu { width } => {
                want_float(0)?;
                want_float(1)?;
                rank(0, 2)?;
                rank(1, 2)?;
                dim_is(0, 1, width)?;
                dim_is(1, 1, width)?;
                if s(0).shape != s(1).shape {
                    return Err(bad("gate and up must have the same shape".into()));
                }
                s(0).shape.clone()
            }
            OpParams::Rope {
                heads, head_dim, ..
            } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, heads * head_dim)?;
                want_index(1)?;
                rank(1, 1)?;
                if s(1).shape[0] != s(0).shape[0] {
                    return Err(bad("one position per row is required".into()));
                }
                s(0).shape.clone()
            }
            OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                ..
            } => {
                // Queries are as wide as the query heads; keys and values are
                // as wide as the key/value heads. Under MHA those coincide,
                // which is why the earlier single-width rule went unnoticed --
                // and why a GQA graph built against it would have declared
                // keys `heads * head_dim` wide and stored mostly padding.
                let query_width =
                    (Dim::constant(heads) * Dim::constant(head_dim)).eval(&SymbolTable::new())?;
                let kv_width =
                    (Dim::constant(kv_heads) * Dim::constant(head_dim)).eval(&SymbolTable::new())?;
                for (i, want) in [query_width, kv_width, kv_width].into_iter().enumerate() {
                    want_activation(i)?;
                    rank(i, 2)?;
                    dim_is(i, 1, want)?;
                }
                if s(0).shape[0] != s(1).shape[0] || s(1).shape[0] != s(2).shape[0] {
                    return Err(bad("q, k and v must have the same row count".into()));
                }
                want_index(3)?;
                rank(3, 1)?;
                if s(3).shape[0] != s(0).shape[0] {
                    return Err(bad("one position per row is required".into()));
                }
                s(0).shape.clone()
            }
            OpParams::MlaAttention { descriptor } => {
                descriptor.validate()?;
                want_activation(0)?;
                rank(0, 2)?;
                dim_is(0, 1, descriptor.hidden)?;
                want_index(1)?;
                rank(1, 1)?;
                if s(1).shape[0] != s(0).shape[0] {
                    return Err(bad("one position per row is required".into()));
                }

                let (q_a_rows, q_a_cols) = descriptor.q_a_proj_shape()?;
                let (q_b_rows, q_b_cols) = descriptor.q_b_proj_shape()?;
                let (kv_a_rows, kv_a_cols) = descriptor.kv_a_proj_shape()?;
                let (kv_b_rows, kv_b_cols) = descriptor.kv_b_proj_shape()?;
                let (o_rows, o_cols) = descriptor.o_proj_shape()?;
                for (i, (rows, cols)) in [
                    (q_a_rows, q_a_cols),
                    (q_b_rows, q_b_cols),
                    (kv_a_rows, kv_a_cols),
                    (kv_b_rows, kv_b_cols),
                    (o_rows, o_cols),
                ]
                .into_iter()
                .enumerate()
                {
                    let input = [2, 4, 5, 7, 8][i];
                    want_weight(input)?;
                    rank(input, 2)?;
                    dim_is(input, 0, rows)?;
                    dim_is(input, 1, cols)?;
                }
                for input in [3, 6] {
                    want_weight(input)?;
                    rank(input, 1)?;
                    dim_is(
                        input,
                        0,
                        if input == 3 {
                            descriptor.q_lora_rank
                        } else {
                            descriptor.kv_lora_rank
                        },
                    )?;
                }
                vec![s(0).shape[0].clone(), Dim::constant(descriptor.hidden)]
            }
            OpParams::Residual { .. } => {
                want_float(0)?;
                want_float(1)?;
                if s(0).shape != s(1).shape {
                    return Err(bad(format!(
                        "operands must have the same shape, got {:?} and {:?}",
                        s(0).shape,
                        s(1).shape
                    )));
                }
                s(0).shape.clone()
            }
            OpParams::VocabProjection { vocab, hidden, .. } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, hidden)?;
                want_float(1)?;
                rank(1, 2)?;
                dim_is(1, 0, vocab)?;
                dim_is(1, 1, hidden)?;
                vec![s(0).shape[0].clone(), Dim::constant(vocab)]
            }
            OpParams::Route {
                hidden,
                experts,
                top_k,
                ..
            } => {
                // Validated from `route_operands` rather than from a second
                // hand-written list. Two enumerations of "which operands does a
                // router take" is how they come to disagree, and this operation
                // has two same-shaped optional operands whose order is the only
                // thing separating them.
                for (index, operand) in params.route_operands().as_slice().iter().enumerate() {
                    want_float(index)?;
                    match operand {
                        RouteOperand::Rows => {
                            rank(index, 2)?;
                            dim_is(index, 1, hidden)?;
                        }
                        RouteOperand::Projection => {
                            rank(index, 2)?;
                            dim_is(index, 0, experts)?;
                            dim_is(index, 1, hidden)?;
                        }
                        // The router's own gain, one element per hidden
                        // channel. Present only for a `Normalized` router: a
                        // family without one does not bind ones, it declares
                        // `Raw`, because normalizing an already-normalized
                        // stream with a unit gain is a different function.
                        RouteOperand::Gain => {
                            rank(index, 1)?;
                            dim_is(index, 0, hidden)?;
                        }
                        RouteOperand::PerExpertScale | RouteOperand::SelectionBias => {
                            rank(index, 1)?;
                            dim_is(index, 0, experts)?;
                        }
                    }
                }
                vec![s(0).shape[0].clone(), Dim::constant(top_k)]
            }
            OpParams::ExpertMlp {
                hidden,
                intermediate,
                experts,
                top_k,
                ..
            } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, hidden)?;
                want_route(1)?;
                rank(1, 2)?;
                dim_is(1, 1, top_k)?;
                if s(1).shape[0] != s(0).shape[0] {
                    return Err(bad(
                        "the route table must have one entry per input row".into()
                    ));
                }
                // Fused across experts, and rank 3 rather than a flattened
                // rank-2 tensor so that the expert axis is visible to whatever
                // later decides which slice has to be resident.
                want_float(2)?;
                rank(2, 3)?;
                dim_is(2, 0, experts)?;
                dim_is(
                    2,
                    1,
                    (Dim::constant(2) * Dim::constant(intermediate)).eval(&SymbolTable::new())?,
                )?;
                dim_is(2, 2, hidden)?;
                want_float(3)?;
                rank(3, 3)?;
                dim_is(3, 0, experts)?;
                dim_is(3, 1, hidden)?;
                dim_is(3, 2, intermediate)?;
                vec![
                    s(0).shape[0].clone() * Dim::constant(top_k),
                    Dim::constant(hidden),
                ]
            }
            OpParams::Combine { hidden, top_k, .. } => {
                want_route(0)?;
                rank(0, 2)?;
                dim_is(0, 1, top_k)?;
                want_float(1)?;
                rank(1, 2)?;
                dim_is(1, 1, hidden)?;
                // Structural equality of the `rows * top_k` expression, not of
                // an evaluated number: `rows` is unbound until a step runs, so
                // a slot tensor built for a different route would differ here
                // rather than at execution.
                let slots = s(0).shape[0].clone() * Dim::constant(top_k);
                if s(1).shape[0] != slots {
                    return Err(bad(format!(
                        "the slot tensor has {:?} rows but this route needs {slots:?}",
                        s(1).shape[0]
                    )));
                }
                vec![s(0).shape[0].clone(), Dim::constant(hidden)]
            }
        })
    }

    /// Finish, checking that every operation has a registered reference.
    ///
    /// This is where F6's rule becomes enforced behaviour rather than a
    /// declaration: an operation with no oracle cannot reach an interpreter,
    /// because it cannot reach a `Graph`.
    pub fn finish(self, output: ValueId, oracles: &OracleRegistry) -> Result<Graph> {
        self.finish_with_ids(output, oracles, &GRAPH_IDS)
    }

    fn finish_with_ids(
        self,
        output: ValueId,
        oracles: &OracleRegistry,
        ids: &GraphIdAllocator,
    ) -> Result<Graph> {
        if !self.weight_precisions.is_empty() {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "weight precision declarations name absent roles: {:?}",
                    self.weight_precisions.keys()
                )
                .into(),
            });
        }
        if output.0 as usize >= self.values.len() {
            return Err(Error::InvalidRequest {
                field: "output",
                detail: format!("value {} is not defined", output.0),
            });
        }
        if self.nodes.is_empty() {
            return Err(Error::InvalidArtifact {
                detail: "a graph with no operations computes nothing".into(),
            });
        }
        // Step inputs, before anything else: a declaration narrower than the
        // storage makes every byte count derived from it too small, and the
        // first `U32` encoding shipped with that rule in a comment and nothing
        // enforcing it. A graph declaring three `U32` token ids lowered to a
        // 12-byte requirement for 24 bytes of storage.
        for id in &self.inputs {
            let spec = &self.values[id.0 as usize];
            let name = self.names.get(id).map(String::as_str).unwrap_or("?");
            match spec.role {
                ValueRole::Index(encoding) if !encoding.is_legal_external_input() => {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "step input {name} is declared {encoding:?}, which is                              narrower than the u64 it is stored as"
                        ).into(),
                    });
                }
                ValueRole::Route { .. } => {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "step input {name} is declared a route table; a route is                              produced by a Route operation over this step's own rows,                              never supplied"
                        ).into(),
                    });
                }
                _ => {}
            }
        }
        for n in &self.nodes {
            n.contract.check_lowerable(oracles)?;
        }
        // KV layers must be numbered densely from zero. `node` already refuses
        // two attention nodes on one layer; this is the other half. A graph that
        // used layer 1 and not layer 0 would leave a cache layer permanently
        // empty, so the cache's length would disagree with the frontier forever
        // -- which the sixth review hit, after the state had already advanced.
        let mut layers: Vec<u32> = self
            .nodes
            .iter()
            .filter_map(|n| match n.params {
                OpParams::Attention { layer, .. } => Some(layer),
                OpParams::MlaAttention { descriptor } => Some(descriptor.layer),
                _ => None,
            })
            .collect();
        layers.sort_unstable();
        for (i, l) in layers.iter().enumerate() {
            if *l != i as u32 {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "attention layers must be numbered from zero without gaps; got {layers:?}"
                    )
                    .into(),
                });
            }
        }
        let id = ids.allocate()?;
        Ok(Graph {
            id,
            values: self.values,
            names: self.names,
            nodes: self.nodes,
            inputs: self.inputs,
            weights: self.weights,
            output,
            rows: self.rows,
            positions: self.positions,
        })
    }
}

/// Values supplied to one execution: inputs and weights, by id.
#[derive(Debug, Default, Clone)]
pub struct Bindings<T> {
    entries: BTreeMap<ValueId, T>,
}

impl<T> Bindings<T> {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn set(&mut self, v: ValueId, value: T) -> Option<T> {
        self.entries.insert(v, value)
    }

    pub fn get(&self, v: ValueId) -> Option<&T> {
        self.entries.get(&v)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::OracleEvidence;
    use moxie_types::{ActivationPrecision, Precision};

    fn embedding_registry() -> OracleRegistry {
        let mut registry = OracleRegistry::new();
        registry
            .register(
                Op::Embedding,
                OracleId("test"),
                OracleEvidence {
                    implementation: "test",
                    test_module: "test",
                },
            )
            .unwrap();
        registry
    }

    #[test]
    fn a_step_input_narrower_than_its_storage_is_refused() {
        // `IndexEncoding::U32` exists for route tables, whose ids really are
        // `u32`. It must not become a way to declare token ids or positions
        // narrower than the `u64` they are stored as: the route table's defect
        // was a four-byte over-count, and this would be an eight-byte
        // *under*-count, which is the direction that corrupts. The first
        // version of `U32` stated this rule in a comment and enforced nothing,
        // and an independent review lowered a three-token graph to a 12-byte
        // requirement for 24 bytes of storage.
        let registry = embedding_registry();
        let rows = SymbolId(0);
        let build = |encoding: IndexEncoding| {
            let mut g = GraphBuilder::new(OracleId("test"), rows);
            let tokens = g.input(
                "tokens",
                TensorSpec::new(ValueRole::Index(encoding), vec![Dim::symbol(rows)]),
            );
            let table = g
                .weight(
                    "embedding",
                    TensorSpec::new(
                        ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                        vec![Dim::constant(4), Dim::constant(2)],
                    ),
                )
                .unwrap();
            let out = g
                .node(
                    OpParams::Embedding {
                        vocab: 4,
                        hidden: 2,
                        scale: 1.0,
                    },
                    &[tokens, table],
                )
                .unwrap();
            g.finish(out, &registry)
        };
        assert!(build(IndexEncoding::U64).is_ok());
        let error = build(IndexEncoding::U32).unwrap_err();
        assert_eq!(error.kind(), "invalid_artifact", "{error}");
        assert!(format!("{error}").contains("narrower"), "{error}");
    }

    #[test]
    fn attention_refuses_a_weight_in_a_query_key_or_value_slot() {
        // `want_float` alone would accept this: Linear's second operand,
        // RmsNorm's gain and Embedding's table are legitimately weights in a
        // "float" slot. But query/key/value feed a per-step state append, not
        // a stored parameter, and a BF16 weight there passes the node's own
        // precision contract too -- so nothing downstream of `want_float`
        // would have refused it either.
        let rows = SymbolId(0);
        let activation = |rows: SymbolId| {
            TensorSpec::new(
                ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
                vec![Dim::symbol(rows), Dim::constant(4)],
            )
        };
        let disguised_weight = TensorSpec::new(
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            vec![Dim::symbol(rows), Dim::constant(4)],
        );
        let build = |query: TensorSpec| {
            let mut g = GraphBuilder::new(OracleId("test"), rows);
            let q = g.input("q", query);
            let k = g.input("k", activation(rows));
            let v = g.input("v", activation(rows));
            let positions = g.input(
                "positions",
                TensorSpec::new(
                    ValueRole::Index(IndexEncoding::U64),
                    vec![Dim::symbol(rows)],
                ),
            );
            g.node(
                OpParams::Attention {
                    heads: 1,
                    kv_heads: 1,
                    head_dim: 4,
                    scale: 1.0,
                    visibility: Visibility::Causal,
                    layer: 0,
                },
                &[q, k, v, positions],
            )
        };
        assert!(build(activation(rows)).is_ok());
        let error = build(disguised_weight).unwrap_err();
        assert_eq!(error.kind(), "invalid_artifact", "{error}");
        assert!(format!("{error}").contains("activation role"), "{error}");
    }

    #[test]
    fn a_route_table_cannot_be_declared_as_a_step_input() {
        // A supplied route would let a caller choose which experts a step
        // demands without the router ever running -- a residency decision made
        // by the wrong owner.
        let registry = embedding_registry();
        let rows = SymbolId(0);
        let mut g = GraphBuilder::new(OracleId("test"), rows);
        let tokens = g.input(
            "tokens",
            TensorSpec::new(
                ValueRole::Index(IndexEncoding::U64),
                vec![Dim::symbol(rows)],
            ),
        );
        let _smuggled = g.input(
            "smuggled route",
            TensorSpec::new(
                ValueRole::Route {
                    index: IndexEncoding::U32,
                    coefficient: ActivationPrecision::expect(Precision::F32),
                },
                vec![Dim::symbol(rows), Dim::constant(2)],
            ),
        );
        let table = g
            .weight(
                "embedding",
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                    vec![Dim::constant(4), Dim::constant(2)],
                ),
            )
            .unwrap();
        let out = g
            .node(
                OpParams::Embedding {
                    vocab: 4,
                    hidden: 2,
                    scale: 1.0,
                },
                &[tokens, table],
            )
            .unwrap();
        let error = g.finish(out, &registry).unwrap_err();
        assert_eq!(error.kind(), "invalid_artifact", "{error}");
    }

    const ORACLE: OracleId = OracleId("graph-id-test");
    const ROWS: SymbolId = SymbolId(91);

    fn registry() -> OracleRegistry {
        let mut out = OracleRegistry::new();
        out.register(
            Op::Residual,
            ORACLE,
            OracleEvidence {
                implementation: "identity_tests",
                test_module: "identity_tests",
            },
        )
        .unwrap();
        out
    }

    fn builder() -> (GraphBuilder, ValueId) {
        let mut builder = GraphBuilder::new(ORACLE, ROWS);
        let spec = TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![Dim::symbol(ROWS), Dim::constant(2)],
        );
        let left = builder.input("left", spec.clone());
        let right = builder.input("right", spec);
        let output = builder
            .node(OpParams::Residual { scale: 1.0 }, &[left, right])
            .unwrap();
        (builder, output)
    }

    #[test]
    fn identity_is_assigned_only_after_validation() {
        let ids = GraphIdAllocator::new(7);
        let (bad, output) = builder();
        assert!(
            bad.finish_with_ids(output, &OracleRegistry::new(), &ids)
                .is_err()
        );

        let (good, output) = builder();
        assert_eq!(
            good.finish_with_ids(output, &registry(), &ids)
                .unwrap()
                .id(),
            GraphId(7)
        );
    }

    #[test]
    fn identity_exhaustion_refuses_instead_of_wrapping() {
        let ids = GraphIdAllocator::new(u64::MAX - 1);
        let (last, output) = builder();
        assert_eq!(
            last.finish_with_ids(output, &registry(), &ids)
                .unwrap()
                .id(),
            GraphId(u64::MAX - 1)
        );
        let (overflow, output) = builder();
        let error = overflow
            .finish_with_ids(output, &registry(), &ids)
            .unwrap_err();
        assert_eq!(error.kind(), "invalid_request");
    }

    #[test]
    fn clones_share_identity_and_separate_graphs_do_not() {
        let (first, output) = builder();
        let first = first.finish(output, &registry()).unwrap();
        assert_eq!(first.id(), first.clone().id());
        assert_eq!(first.signature(), first.clone().signature());

        let (second, output) = builder();
        let second = second.finish(output, &registry()).unwrap();
        assert_ne!(first.id(), second.id());
    }
}
