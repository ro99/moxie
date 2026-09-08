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

use moxie_types::{
    AccumulationPolicy, ActivationPrecision, Dim, Error, Precision, Result, SymbolId, SymbolTable,
    WeightPrecision,
};

use crate::{Op, OpContract, OracleId, OracleRegistry, PartitionRule, StateEffect, Visibility};

/// A value flowing through the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

/// One operation node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

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
    Index,
}

impl ValueRole {
    pub fn is_index(self) -> bool {
        matches!(self, ValueRole::Index)
    }

    /// The stored element encoding, for a float role.
    pub fn precision(self) -> Option<Precision> {
        match self {
            ValueRole::Weight(w) => Some(w.get()),
            ValueRole::Activation(a) => Some(a.get()),
            ValueRole::Index => None,
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
        hidden: u64,
        eps: f32,
    },
    SwiGlu {
        width: u64,
    },
    Rope {
        heads: u64,
        head_dim: u64,
        rotary_dim: u64,
        base: f32,
    },
    Attention {
        heads: u64,
        head_dim: u64,
        visibility: Visibility,
        /// Which KV store this node reads and appends to.
        layer: u32,
    },
    Residual,
    VocabProjection {
        vocab: u64,
        hidden: u64,
    },
}

impl OpParams {
    pub fn op(&self) -> Op {
        match self {
            OpParams::Embedding { .. } => Op::Embedding,
            OpParams::Linear { .. } => Op::Linear,
            OpParams::RmsNorm { .. } => Op::RmsNorm,
            OpParams::SwiGlu { .. } => Op::SwiGlu,
            OpParams::Rope { .. } => Op::Rope,
            OpParams::Attention { .. } => Op::Attention,
            OpParams::Residual => Op::Residual,
            OpParams::VocabProjection { .. } => Op::VocabProjection,
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
            | OpParams::Rope { .. }
            | OpParams::VocabProjection { .. } => PartitionRule::ColumnShardable,
            // Applied exactly once; the norm reduces over the whole hidden axis.
            OpParams::Embedding { .. } | OpParams::RmsNorm { .. } | OpParams::Residual => {
                PartitionRule::Replicated
            }
            // Head ownership, GQA KV replication and the output reduction are
            // document 04's M5 work. Undetermined until then, deliberately.
            OpParams::Attention { .. } => PartitionRule::NotDetermined,
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

    /// How many value inputs this operation takes.
    pub fn arity(&self) -> usize {
        match self {
            OpParams::Embedding { .. } => 2, // tokens, table
            OpParams::Linear { bias, .. } => 2 + usize::from(*bias),
            OpParams::RmsNorm { .. } => 2,   // x, gain
            OpParams::SwiGlu { .. } => 2,    // gate, up
            OpParams::Rope { .. } => 2,      // x, positions
            OpParams::Attention { .. } => 4, // q, k, v, positions
            OpParams::Residual => 2,
            OpParams::VocabProjection { .. } => 2, // hidden, table
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
                base,
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
                if !(base.is_finite() && base > 1.0) {
                    return Err(Error::InvalidRequest {
                        field: "rope_base",
                        detail: format!("base must be finite and > 1, got {base}"),
                    });
                }
                Ok(())
            }
            OpParams::Attention {
                heads, head_dim, ..
            } => {
                if heads == 0 || head_dim == 0 {
                    return Err(Error::InvalidRequest {
                        field: "attention",
                        detail: format!("{heads} heads of dimension {head_dim}"),
                    });
                }
                Ok(())
            }
            OpParams::RmsNorm { eps, hidden } => {
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct Graph {
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
        let out_spec = TensorSpec::new(ValueRole::Activation(params.output_precision()), out_shape);
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
            if s(i).role.is_index() {
                return Err(bad(format!("input {i} must be a float role, got an index")));
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
            OpParams::Embedding { vocab, hidden } => {
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
            OpParams::RmsNorm { hidden, .. } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, hidden)?;
                want_float(1)?;
                rank(1, 1)?;
                dim_is(1, 0, hidden)?;
                s(0).shape.clone()
            }
            OpParams::SwiGlu { width } => {
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
                heads, head_dim, ..
            } => {
                let width = heads * head_dim;
                for i in 0..3 {
                    want_float(i)?;
                    rank(i, 2)?;
                    dim_is(i, 1, width)?;
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
            OpParams::Residual => {
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
            OpParams::VocabProjection { vocab, hidden } => {
                want_float(0)?;
                rank(0, 2)?;
                dim_is(0, 1, hidden)?;
                want_float(1)?;
                rank(1, 2)?;
                dim_is(1, 0, vocab)?;
                dim_is(1, 1, hidden)?;
                vec![s(0).shape[0].clone(), Dim::constant(vocab)]
            }
        })
    }

    /// Finish, checking that every operation has a registered reference.
    ///
    /// This is where F6's rule becomes enforced behaviour rather than a
    /// declaration: an operation with no oracle cannot reach an interpreter,
    /// because it cannot reach a `Graph`.
    pub fn finish(self, output: ValueId, oracles: &OracleRegistry) -> Result<Graph> {
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
        Ok(Graph {
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
