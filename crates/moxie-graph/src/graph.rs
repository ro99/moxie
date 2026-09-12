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

use crate::{Op, OpContract, OracleId, OracleRegistry, PartitionRule, StateEffect, Visibility};

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
    /// Inputs are `(rows, router gain, router projection[, per-expert scale])`
    /// and the output is a route table of `[rows, top_k]`. The whole score
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
        /// Epsilon of the router's own RMS normalization, which is scale-free:
        /// the gain is the bound `router.scale` tensor, applied after it.
        eps: f32,
        /// The scalar multiplying the normalized, gained row before projection.
        ///
        /// Gemma 4 uses `hidden^(-1/2)` (`Gemma4TextRouter.scalar_root_size`).
        /// It is stated rather than derived from `hidden` because it is a
        /// family choice, not an identity: a router without one passes 1.0, and
        /// deriving it would silently impose Gemma's on every other family.
        input_scale: f32,
        /// Whether a per-expert coefficient scale is bound as input 3.
        ///
        /// Applied **after** renormalization and never renormalized away, so
        /// the coefficients of a scaled router do not sum to one. A combine
        /// that normalized them again would delete a trained parameter.
        per_expert_scale: bool,
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
            | OpParams::Rope { .. }
            | OpParams::VocabProjection { .. } => PartitionRule::ColumnShardable,
            // Applied exactly once; the norm reduces over the whole hidden axis.
            OpParams::Embedding { .. } | OpParams::RmsNorm { .. } | OpParams::Residual { .. } => {
                PartitionRule::Replicated
            }
            // Head ownership, GQA KV replication and the output reduction are
            // document 04's M5 work. Undetermined until then, deliberately.
            OpParams::Attention { .. } => PartitionRule::NotDetermined,
            // Replicated, and that is a correctness requirement rather than a
            // cost choice. Every rank must reach the same selection from the
            // same row: a router sharded over its expert axis would reduce
            // partial logits in a rank-dependent order, and two ranks that
            // disagree about which expert a row needs disagree about which
            // weights have to be resident. The router is three small tensors,
            // so replicating them costs almost nothing.
            OpParams::Route { .. } => PartitionRule::Replicated,
            // Expert partitioning is M5. Failing closed here is what keeps that
            // a lowering decision rather than something this task pre-empted:
            // sharding experts also decides where `Combine`'s reduction happens.
            OpParams::ExpertMlp { .. } | OpParams::Combine { .. } => PartitionRule::NotDetermined,
        }
    }

    /// What this operation does to sequence state.
    pub fn state_effect(&self) -> StateEffect {
        match self {
            OpParams::Attention { .. } => StateEffect::Appends,
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

    /// How many value inputs this operation takes.
    pub fn arity(&self) -> usize {
        match self {
            OpParams::Embedding { .. } => 2, // tokens, table
            OpParams::Linear { bias, .. } => 2 + usize::from(*bias),
            OpParams::RmsNorm { .. } => 2,   // x, gain
            OpParams::SwiGlu { .. } => 2,    // gate, up
            OpParams::GeGlu { .. } => 2,     // gate, up
            OpParams::Rope { .. } => 2,      // x, positions
            OpParams::Attention { .. } => 4, // q, k, v, positions
            OpParams::Residual { .. } => 2,
            OpParams::VocabProjection { .. } => 2, // hidden, table
            // rows, router gain, router projection, and the per-expert scale
            // when the family has one.
            OpParams::Route {
                per_expert_scale, ..
            } => 3 + usize::from(*per_expert_scale),
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
                eps,
                input_scale,
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
            OpParams::Combine { hidden, top_k, .. } => {
                if hidden == 0 || top_k == 0 {
                    return Err(Error::InvalidRequest {
                        field: "combine",
                        detail: format!("hidden {hidden}, top_k {top_k}"),
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

    fn add_value(&mut self, name: &str, spec: TensorSpec) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(spec);
        self.names.insert(id, name.to_string());
        id
    }

    /// A value supplied per step.
    pub fn input(&mut self, name: &str, spec: TensorSpec) -> ValueId {
        let id = self.add_value(name, spec);
        self.inputs.push(id);
        id
    }

    /// A model parameter, supplied once.
    pub fn weight(&mut self, name: &str, spec: TensorSpec) -> Result<ValueId> {
        if !matches!(spec.role, ValueRole::Weight(_)) {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{name} is declared as a weight but its role is {:?}",
                    spec.role
                ),
            });
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
        if let OpParams::Attention { layer, .. } = params
            && let Some(clash) = self
                .nodes
                .iter()
                .find(|n| matches!(n.params, OpParams::Attention { layer: l, .. } if l == layer))
        {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "attention layer {layer} is already used by node {}; two nodes sharing \
                     a KV layer would append the same positions twice",
                    clash.id.0
                ),
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
                        ),
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
            weights: vec![WeightPrecision::expect(Precision::Bf16)],
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
                    ),
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
                detail: format!("{}: {detail}", params.op().name()),
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
                    want_float(i)?;
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
                per_expert_scale,
                ..
            } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, hidden)?;
                // The router's own gain, one element per hidden channel. It is
                // a bound weight rather than a folded constant so that a family
                // whose router has no `scale` tensor binds ones and says so,
                // the way the reduced Gemma graph already binds a unit gain for
                // its value normalization.
                want_float(1)?;
                rank(1, 1)?;
                dim_is(1, 0, hidden)?;
                want_float(2)?;
                rank(2, 2)?;
                dim_is(2, 0, experts)?;
                dim_is(2, 1, hidden)?;
                if per_expert_scale {
                    want_float(3)?;
                    rank(3, 1)?;
                    dim_is(3, 0, experts)?;
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
                _ => None,
            })
            .collect();
        layers.sort_unstable();
        for (i, l) in layers.iter().enumerate() {
            if *l != i as u32 {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "attention layers must be numbered from zero without gaps; got {layers:?}"
                    ),
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
