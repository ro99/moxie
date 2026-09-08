//! The host reference interpreter.
//!
//! Document 02: "Host reference evaluation is a shared graph interpreter, with
//! separate primitive-level fixtures where needed." This is that interpreter. It
//! walks a validated [`Graph`], evaluates each node against the registered
//! reference in `moxie-oracles`, advances sequence state through `moxie-state`,
//! and returns logits.
//!
//! ## What it is not
//!
//! Not a fast path. Task 0003: "this is the thing other paths are compared
//! against, and a 'reference' that has been optimised is not one." It allocates
//! per row, copies freely, and does the arithmetic in the order the contract
//! names. A CUDA kernel that wants to reassociate owes document 07's measured
//! error and quality evidence, and this is what that evidence is produced
//! against.
//!
//! Not a second implementation of the mathematics either. It **dispatches** to
//! `moxie-oracles`; there is one implementation of each equation, and pretending
//! to two would be theatre. The independence document 07 requires lives in the
//! tests, which transcribe the equations separately in FP64.
//!
//! ## What it enforces
//!
//! The three M0 contracts that had no consumer until now:
//!
//! * an operation whose oracle is not registered cannot be built into a graph,
//!   so it cannot reach execution;
//! * every forward pass advances `executed` and records a provenance-qualified
//!   result through `moxie-state`, so logit validity is a retained result rather
//!   than a counter coincidence;
//! * positions are absolute, checked against the branch's own frontier, so a
//!   chunk-local index cannot masquerade as a sequence position (R21).

#![forbid(unsafe_code)]

pub mod kv;
pub mod tensor;

use moxie_graph::{Bindings, Graph, Node, OpParams};
use moxie_oracles::{activation, attention, linear, norm, residual, rope};
use moxie_state::{LogitsHandle, SequenceState};
use moxie_types::BranchId;
use moxie_types::{Error, Result};

pub use kv::KvCache;
pub use tensor::{HostTensor, Value};

/// A cancellation token checked at every operation boundary.
///
/// R08: "Memory leases are released at turn boundaries and cancellation even if
/// no next token arrives." At this scale there are no leases to release, but the
/// same rule applies to state: a cancelled step must leave the sequence exactly
/// as it found it. [`Interpreter::run`] stages its KV appends and commits them
/// only on success, so a cancelled step has nothing to roll back.
///
/// The budget form is deliberate: a boolean flipped by another thread cannot be
/// tested deterministically, and a test that cancels "somewhere" proves less
/// than one that cancels at a named operation.
#[derive(Debug, Clone)]
pub struct Cancel {
    remaining: std::cell::Cell<u64>,
}

impl Cancel {
    /// Never cancels.
    pub fn never() -> Self {
        Self {
            remaining: std::cell::Cell::new(u64::MAX),
        }
    }

    /// Cancels once `n` operation boundaries have been passed.
    pub fn after(n: u64) -> Self {
        Self {
            remaining: std::cell::Cell::new(n),
        }
    }

    fn check(&self, at: &'static str) -> Result<()> {
        let left = self.remaining.get();
        if left == 0 {
            return Err(Error::Cancelled { at });
        }
        if left != u64::MAX {
            self.remaining.set(left - 1);
        }
        Ok(())
    }
}

/// What one step produced.
#[derive(Debug, Clone)]
pub struct StepOutput {
    /// One row of logits per executed row, FP32 and unrounded.
    ///
    /// Task 0003: document 05 builds the sampler's pre-truncation normalizer
    /// from these and exact speculative verification consumes that
    /// distribution, so rounding them to BF16 first would change it.
    pub logits: HostTensor,
    /// The retained forward result, qualified by branch, prefix and lineage.
    pub retained: LogitsHandle,
    /// The absolute prefix the logits predict the token after.
    pub prefix: u64,
}

/// Walks a graph.
#[derive(Debug)]
pub struct Interpreter;

impl Interpreter {
    pub fn new() -> Self {
        Self
    }

    /// Execute one step.
    ///
    /// `bindings` supplies every graph input and weight. `positions` inside the
    /// bindings must be the branch's next `rows` absolute positions, contiguous
    /// and ascending: R21's failure is a chunk-local index reaching an operation
    /// that needed a sequence position, and this is where that is refused rather
    /// than silently computed.
    ///
    /// On success: the staged KV appends are committed, `executed` advances by
    /// the row count, and the result is recorded so `next_logits_valid` holds.
    /// On any failure, including cancellation, nothing is committed and the
    /// state is untouched.
    pub fn run(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
        state: &mut SequenceState,
        branch: BranchId,
        kv: &mut KvCache,
        cancel: &Cancel,
    ) -> Result<StepOutput> {
        let before = state.frontiers(branch)?;
        let mut values: Vec<Option<Value>> = vec![None; graph.value_count()];

        // Bind inputs and weights.
        for v in graph.inputs().iter().chain(graph.weights()) {
            let bound = bindings.get(*v).ok_or_else(|| Error::InvalidRequest {
                field: "bindings",
                detail: format!(
                    "value {} ({}) is not bound",
                    v.0,
                    graph.name(*v).unwrap_or("?")
                ),
            })?;
            values[v.0 as usize] = Some(bound.clone());
        }

        // Every step needs positions, because every step writes state at them.
        let positions = self.positions_of(graph, &values)?;
        let rows = positions.len();
        if rows == 0 {
            return Err(Error::InvalidRequest {
                field: "rows",
                detail: "a step with no rows".into(),
            });
        }
        for (i, p) in positions.iter().enumerate() {
            let want = before.executed + i as u64;
            if *p != want {
                return Err(Error::InvalidRequest {
                    field: "positions",
                    detail: format!(
                        "row {i} claims absolute position {p}, but this branch has executed \
                         {} token(s), so it must be {want}. Positions are absolute over the \
                         sequence, not indices within a chunk",
                        before.executed
                    ),
                });
            }
        }

        // The declared shapes are checked against the values actually supplied.
        // Without this the graph's shapes would be structural decoration that
        // nothing enforces once real data arrives.
        let mut symbols = moxie_types::SymbolTable::new();
        symbols.declare(graph.rows_symbol(), "rows");
        symbols.bind(graph.rows_symbol(), rows as u64);
        for v in graph.inputs().iter().chain(graph.weights()) {
            let spec = graph.spec(*v).expect("bound value has a spec");
            let want = spec.extent(&symbols)?;
            let bound = values[v.0 as usize].as_ref().expect("bound above");
            let got: Vec<u64> = match bound {
                Value::Float(t) => t.shape().iter().map(|d| *d as u64).collect(),
                Value::Index(i) => vec![i.len() as u64],
            };
            if got != want {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "{} has shape {got:?} but the graph declares {want:?}",
                        graph.name(*v).unwrap_or("?")
                    ),
                });
            }
        }

        // Staged, not written through: a failed or cancelled step never wrote.
        let mut staged: Vec<StagedAppend> = Vec::new();

        for node in graph.nodes() {
            cancel.check(node.params.op().name())?;
            let out = self.eval(node, &values, kv, &mut staged, &positions)?;
            values[node.output.0 as usize] = Some(out);
        }

        let logits = values[graph.output().0 as usize]
            .as_ref()
            .ok_or_else(|| Error::InvalidArtifact {
                detail: "the graph output was never produced".into(),
            })?
            .as_float()?
            .clone();
        if logits.rows() != rows {
            return Err(Error::InvalidArtifact {
                detail: format!("{} logit rows for {rows} input row(s)", logits.rows()),
            });
        }

        // Commit. Everything above either succeeded or returned without writing.
        for a in staged.drain(..) {
            kv.append(a.layer, a.position, a.key, a.value)?;
        }
        state.execute(branch, rows as u64)?;
        let prefix = state.frontiers(branch)?.executed;
        let retained = state.record_logits(branch, prefix)?;

        Ok(StepOutput {
            logits,
            retained,
            prefix,
        })
    }

    /// The positions any state-touching node was given.
    fn positions_of(&self, graph: &Graph, values: &[Option<Value>]) -> Result<Vec<u64>> {
        for node in graph.nodes() {
            if let OpParams::Attention { .. } = node.params {
                let v = values[node.inputs[3].0 as usize].as_ref().ok_or_else(|| {
                    Error::InvalidRequest {
                        field: "positions",
                        detail: "attention positions are not bound".into(),
                    }
                })?;
                return Ok(v.as_index()?.to_vec());
            }
            if let OpParams::Rope { .. } = node.params {
                let v = values[node.inputs[1].0 as usize].as_ref().ok_or_else(|| {
                    Error::InvalidRequest {
                        field: "positions",
                        detail: "rope positions are not bound".into(),
                    }
                })?;
                return Ok(v.as_index()?.to_vec());
            }
        }
        Err(Error::InvalidArtifact {
            detail: "no operation in this graph consumes positions, so the step has no \
                     place in the sequence"
                .into(),
        })
    }

    fn eval(
        &self,
        node: &Node,
        values: &[Option<Value>],
        kv: &KvCache,
        staged: &mut Vec<StagedAppend>,
        positions: &[u64],
    ) -> Result<Value> {
        let input = |i: usize| -> Result<&Value> {
            values[node.inputs[i].0 as usize]
                .as_ref()
                .ok_or_else(|| Error::InvalidArtifact {
                    detail: format!(
                        "{} input {i} was used before it was produced",
                        node.params.op().name()
                    ),
                })
        };

        let out = match node.params {
            OpParams::Embedding { vocab, hidden } => {
                let tokens = input(0)?.as_index()?;
                let table = input(1)?.as_float()?;
                let mut out = Vec::with_capacity(tokens.len() * hidden as usize);
                for t in tokens {
                    let id = u32::try_from(*t).map_err(|_| Error::InvalidRequest {
                        field: "token",
                        detail: format!("token {t} does not fit a u32"),
                    })?;
                    out.extend(linear::embedding_row(
                        id,
                        table.data(),
                        vocab as usize,
                        hidden as usize,
                    )?);
                }
                // A copy: the stored value passes through unchanged, so this is
                // `bf16` rather than `round_to_bf16`, and it would fail loudly
                // if the table were not BF16-valued.
                Value::Float(HostTensor::bf16(out, vec![tokens.len(), hidden as usize])?)
            }
            OpParams::Linear {
                out_features, bias, ..
            } => {
                let x = input(0)?.as_float()?;
                let w = input(1)?.as_float()?;
                let b = if bias {
                    Some(input(2)?.as_float()?.data().to_vec())
                } else {
                    None
                };
                let mut out = Vec::with_capacity(x.rows() * out_features as usize);
                for r in 0..x.rows() {
                    out.extend(linear::linear_row(
                        x.row(r)?,
                        w.data(),
                        out_features as usize,
                        b.as_deref(),
                    )?);
                }
                Value::Float(HostTensor::round_to_bf16(
                    out,
                    vec![x.rows(), out_features as usize],
                )?)
            }
            OpParams::RmsNorm { eps, .. } => {
                let x = input(0)?.as_float()?;
                let g = input(1)?.as_float()?;
                let mut out = Vec::with_capacity(x.data().len());
                for r in 0..x.rows() {
                    out.extend(norm::rms_norm_row(x.row(r)?, g.data(), eps)?);
                }
                Value::Float(HostTensor::round_to_bf16(out, x.shape().to_vec())?)
            }
            OpParams::SwiGlu { .. } => {
                let gate = input(0)?.as_float()?;
                let up = input(1)?.as_float()?;
                let mut out = Vec::with_capacity(gate.data().len());
                for r in 0..gate.rows() {
                    out.extend(activation::swiglu_row(gate.row(r)?, up.row(r)?)?);
                }
                Value::Float(HostTensor::round_to_bf16(out, gate.shape().to_vec())?)
            }
            OpParams::Rope {
                heads,
                head_dim,
                rotary_dim,
                base,
            } => {
                let x = input(0)?.as_float()?;
                let pos = input(1)?.as_index()?;
                if pos.len() != x.rows() {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{} positions for {} rows", pos.len(), x.rows()),
                    });
                }
                let mut out = Vec::with_capacity(x.data().len());
                for (r, p) in pos.iter().enumerate().take(x.rows()) {
                    out.extend(rope::rope_row(
                        x.row(r)?,
                        *p,
                        base,
                        heads as usize,
                        head_dim as usize,
                        rotary_dim as usize,
                    )?);
                }
                Value::Float(HostTensor::round_to_bf16(out, x.shape().to_vec())?)
            }
            OpParams::Attention {
                heads,
                head_dim,
                visibility,
                layer,
            } => {
                let q = input(0)?.as_float()?;
                let k = input(1)?.as_float()?;
                let v = input(2)?.as_float()?;
                // Attention is the only state-touching operation here. It reads
                // the committed history plus this step's staged appends, which
                // is what lets a multi-row prefill attend to its own earlier
                // rows without those rows having been committed yet.
                let mut history = kv.history(layer)?.clone();
                for a in staged.iter().filter(|a| a.layer == layer) {
                    history.append(a.position, a.key.clone(), a.value.clone())?;
                }
                let mut out = Vec::with_capacity(q.data().len());
                for (r, position) in positions.iter().enumerate().take(q.rows()) {
                    let position = *position;
                    let (key, value) = (k.row(r)?.to_vec(), v.row(r)?.to_vec());
                    // A causal query attends to itself, so this row's key and
                    // value join the history before it is read. They also join
                    // the staged list, so a later row of the same prefill sees
                    // them and a failed step never writes them.
                    history.append(position, key.clone(), value.clone())?;
                    staged.push(StagedAppend {
                        layer,
                        position,
                        key,
                        value,
                    });
                    out.extend(attention::attend_multi_head(
                        q.row(r)?,
                        &history,
                        position,
                        heads as usize,
                        head_dim as usize,
                        visibility,
                    )?);
                }
                Value::Float(HostTensor::round_to_bf16(out, q.shape().to_vec())?)
            }
            OpParams::Residual => {
                let a = input(0)?.as_float()?;
                let b = input(1)?.as_float()?;
                let mut out = Vec::with_capacity(a.data().len());
                for r in 0..a.rows() {
                    out.extend(residual::residual_row(a.row(r)?, b.row(r)?)?);
                }
                Value::Float(HostTensor::round_to_bf16(out, a.shape().to_vec())?)
            }
            OpParams::VocabProjection { vocab, .. } => {
                let h = input(0)?.as_float()?;
                let w = input(1)?.as_float()?;
                let mut out = Vec::with_capacity(h.rows() * vocab as usize);
                for r in 0..h.rows() {
                    out.extend(linear::linear_row(
                        h.row(r)?,
                        w.data(),
                        vocab as usize,
                        None,
                    )?);
                }
                // Not rounded. See `StepOutput::logits`.
                Value::Float(HostTensor::f32(out, vec![h.rows(), vocab as usize])?)
            }
        };
        Ok(out)
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// A KV append held until the whole step succeeds.
#[derive(Debug, Clone)]
struct StagedAppend {
    layer: u32,
    position: u64,
    key: Vec<f32>,
    value: Vec<f32>,
}
