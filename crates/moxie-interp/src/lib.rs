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
pub mod paged;
pub mod tensor;

use moxie_graph::{Bindings, Graph, Node, OpParams, ValueId};
use moxie_oracles::{activation, attention, linear, norm, residual, rope};
use moxie_state::{LogitsHandle, SequenceState};
use moxie_types::BranchId;
use moxie_types::{Error, Result};

pub use kv::{CacheId, CacheJournal, KvCache};
pub use tensor::{HostTensor, Value};

pub(crate) fn try_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let requested_bytes = capacity
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(moxie_types::DimError::Overflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::CpuWorkspace)),
            requested_bytes: requested_bytes as u64,
            available_bytes: 0,
        })?;
    Ok(out)
}

pub(crate) fn try_clone_slice<T: Clone>(values: &[T]) -> Result<Vec<T>> {
    let mut out = try_vec(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}

fn try_shape2(first: usize, second: usize) -> Result<Vec<usize>> {
    let mut shape = try_vec(2)?;
    shape.extend([first, second]);
    Ok(shape)
}

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
#[derive(Debug)]
pub struct Cancel {
    remaining: std::cell::Cell<u64>,
    signal: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    local: std::sync::atomic::AtomicBool,
}

impl Clone for Cancel {
    fn clone(&self) -> Self {
        Self {
            remaining: self.remaining.clone(),
            signal: self.signal.clone(),
            local: std::sync::atomic::AtomicBool::new(
                self.local.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

impl Cancel {
    /// Never cancels.
    pub fn never() -> Self {
        Self {
            remaining: std::cell::Cell::new(u64::MAX),
            signal: None,
            local: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Cancels once `n` operation boundaries have been passed.
    pub fn after(n: u64) -> Self {
        Self {
            remaining: std::cell::Cell::new(n),
            signal: None,
            local: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Boundaries still allowed before this token cancels.
    ///
    /// Exposed so a test can *count* a step's boundaries rather than hardcode
    /// how many there are: a test that cancels at "depth 7" stops meaning
    /// anything the moment a node is added.
    pub fn remaining(&self) -> u64 {
        self.remaining.get()
    }

    pub fn with_signal(signal: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        Self {
            remaining: std::cell::Cell::new(u64::MAX),
            signal: Some(signal),
            local: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn signal(&self) -> &std::sync::atomic::AtomicBool {
        self.signal.as_deref().unwrap_or(&self.local)
    }

    pub fn check(&self, at: &'static str) -> Result<()> {
        let left = self.remaining.get();
        if left == 0 || self.signal().load(std::sync::atomic::Ordering::Relaxed) {
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

/// Every node result from one state-free graph walk, plus its final value.
/// Values are retained in graph order so device qualification can compare
/// semantic boundaries without reimplementing the interpreter dispatcher.
#[derive(Debug, Clone)]
pub struct StatelessTrace {
    node_outputs: Vec<(ValueId, Value)>,
    output: Value,
}

impl StatelessTrace {
    pub fn node_outputs(&self) -> &[(ValueId, Value)] {
        &self.node_outputs
    }

    pub fn node_output(&self, value: ValueId) -> Option<&Value> {
        self.node_outputs
            .iter()
            .find_map(|(candidate, output)| (*candidate == value).then_some(output))
    }

    pub fn output(&self) -> &Value {
        &self.output
    }
}

impl Interpreter {
    pub fn new() -> Self {
        Self
    }

    /// Walk a graph that has no state effects, using the same primitive oracle
    /// dispatch and BF16 boundaries as [`Self::run`] but without publishing a
    /// sequence transaction. This is the reference consumer for qualification
    /// of small stateless device subgraphs.
    pub fn run_stateless(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
    ) -> Result<StatelessTrace> {
        if !graph.state_effects().is_empty() {
            return Err(Error::InvalidRequest {
                field: "graph",
                detail: "run_stateless refuses a graph with state effects".into(),
            });
        }
        if bindings.len() != graph.inputs().len() + graph.weights().len() {
            return Err(Error::InvalidRequest {
                field: "bindings",
                detail: "stateless bindings must name exactly every input and weight".into(),
            });
        }
        let first = *graph
            .inputs()
            .first()
            .ok_or_else(|| Error::InvalidRequest {
                field: "graph",
                detail: "a stateless graph needs an input".into(),
            })?;
        let rows = bindings
            .get(first)
            .ok_or_else(|| Error::InvalidRequest {
                field: "bindings",
                detail: "the first graph input is not bound".into(),
            })?
            .rows();
        let mut symbols = moxie_types::SymbolTable::new();
        symbols.declare(graph.rows_symbol(), "rows");
        symbols.bind(graph.rows_symbol(), rows as u64);
        let mut values: Vec<Option<Value>> = vec![None; graph.value_count()];
        for value in graph.inputs().iter().chain(graph.weights()) {
            let bound = bindings.get(*value).ok_or_else(|| Error::InvalidRequest {
                field: "bindings",
                detail: format!("value {} is not bound", value.0),
            })?;
            let spec = graph.spec(*value).expect("validated graph value");
            let expected = spec.extent(&symbols)?;
            let actual: Vec<u64> = match bound {
                Value::Float(tensor) => tensor.shape().iter().map(|dim| *dim as u64).collect(),
                Value::Index(index) => vec![index.len() as u64],
            };
            if actual != expected || bound_precision(bound) != spec.role.precision() {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "binding {} shape or precision differs from the graph",
                        value.0
                    ),
                });
            }
            if let Value::Float(tensor) = bound
                && tensor.data().iter().any(|element| !element.is_finite())
            {
                return Err(Error::InvalidArtifact {
                    detail: format!("binding {} contains a nonfinite value", value.0),
                });
            }
            values[value.0 as usize] = Some(bound.clone());
        }

        // Stateless nodes never consult either object, but sharing `eval`
        // avoids a second graph dispatcher or a second copy of any equation.
        let state = SequenceState::new([]);
        let kv = KvCache::for_branch(0, &state, moxie_state::ROOT)?;
        let mut staged = Vec::new();
        let mut node_outputs = Vec::with_capacity(graph.nodes().len());
        for node in graph.nodes() {
            let output = self.eval(node, &values, &kv, &mut staged, &[])?;
            if let Value::Float(tensor) = &output
                && tensor.data().iter().any(|element| !element.is_finite())
            {
                return Err(Error::Numerical {
                    detail: format!("{} produced a nonfinite value", node.params.op().name()),
                });
            }
            node_outputs.push((node.output, output.clone()));
            values[node.output.0 as usize] = Some(output);
        }
        let output = values[graph.output().0 as usize]
            .as_ref()
            .ok_or_else(|| Error::InvalidArtifact {
                detail: "the stateless graph output was never produced".into(),
            })?
            .clone();
        Ok(StatelessTrace {
            node_outputs,
            output,
        })
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
        // The cache must be this branch's, at this prefix, holding this version
        // of it. Without that check the provenance rules guard a counter while
        // the bytes come from anywhere.
        kv.check_owner(state, branch)?;
        // ... and it must cover exactly the layers this graph writes. A graph
        // whose attention sits on a layer the cache does not have, or a cache
        // with a layer no attention writes, means the layers advance at
        // different rates and the cache can never agree with the frontier again.
        // The sixth review found that discovered *after* the state had advanced;
        // it is a property of the graph and the cache, so it is checked here,
        // before anything is written.
        let graph_layers = graph.attention_layers().len();
        // A step publishes sequence state, and a graph that touches no state has
        // none to publish. The sixth review's second pass reached the same
        // atomicity defect through a `RoPE -> VocabProjection` graph with a
        // zero-layer cache: the counts matched at zero, so the coverage check
        // below passed, and then `executed` advanced past a cache that could
        // never hold anything.
        //
        // Attention-free execution is a coherent thing to want -- it just does
        // not advance `executed`, so it is a different operation with different
        // publication rules. Refused here rather than half-supported; document
        // 06 M1.4's transaction API is where it would belong.
        if graph_layers == 0 {
            return Err(Error::InvalidArtifact {
                detail: "this graph has no attention, so it touches no sequence state and \
                         cannot be executed as a step; a stateless graph would not advance \
                         the executed frontier and needs its own publication rules"
                    .into(),
            });
        }
        if kv.layers() != graph_layers {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "the cache has {} layer(s) but the graph writes {graph_layers}; every \
                     layer must advance together or the cache cannot track the frontier",
                    kv.layers()
                ),
            });
        }
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
            let name = graph.name(*v).unwrap_or("?");
            let got: Vec<u64> = match bound {
                Value::Float(t) => t.shape().iter().map(|d| *d as u64).collect(),
                Value::Index(i) => vec![i.len() as u64],
            };
            if got != want {
                return Err(Error::InvalidArtifact {
                    detail: format!("{name} has shape {got:?} but the graph declares {want:?}"),
                });
            }
            // Shape agreement is not dtype agreement. The fourth review bound an
            // FP32 tensor holding a value BF16 cannot represent to a BF16-declared
            // input and execution accepted it, so every error bound downstream
            // was resting on an invariant nothing checked. A checked constructor
            // does not help when the caller can pick a different one.
            match (bound, spec.role.precision()) {
                (Value::Float(t), Some(p)) if t.precision() != p => {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{name} is {} but the graph declares {p}", t.precision()),
                    });
                }
                (Value::Float(_), None) => {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{name} is declared an index but a tensor was bound"),
                    });
                }
                (Value::Index(_), Some(p)) => {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{name} is declared {p} but an index was bound"),
                    });
                }
                _ => {}
            }
            // A non-finite weight produces non-finite everything, and document
            // 05's rule that "NaN logits ... produce typed errors" is worth
            // nothing if the NaN is admitted at the boundary and only noticed
            // after the state has advanced.
            if let Value::Float(t) = bound
                && let Some(i) = t.data().iter().position(|x| !x.is_finite())
            {
                return Err(Error::InvalidArtifact {
                    detail: format!("{name} element {i} is {}", t.data()[i]),
                });
            }
        }

        // Staged, not written through: a failed or cancelled step never wrote.
        let mut staged: Vec<StagedAppend> = Vec::new();

        for node in graph.nodes() {
            cancel.check(node.params.op().name())?;
            let out = self.eval(node, &values, kv, &mut staged, &positions)?;
            // Checked per node, not only at the output: attributing a NaN to the
            // operation that produced it is the difference between a defect
            // report and a puzzle.
            if let Value::Float(t) = &out
                && let Some(i) = t.data().iter().position(|x| !x.is_finite())
            {
                return Err(Error::Numerical {
                    detail: format!(
                        "{} produced {} at element {i}",
                        node.params.op().name(),
                        t.data()[i]
                    ),
                });
            }
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

        // Publication, as one transaction with two participants.
        //
        // `moxie-state` journals the branch and `KvCache` journals its layers;
        // this is the composition point that opens both and resolves both the
        // same way. Task 0003 shipped without this, and four review passes each
        // found a different way for a failure after `state.execute` to leave the
        // sequence advanced with no way to retry -- because there was no
        // `unexecute`, correctness depended on having enumerated every way the
        // remaining calls could fail. Now a failure aborts.
        let cache_journal = kv.begin()?;
        let txn = match state.begin(branch) {
            Ok(t) => t,
            Err(e) => {
                kv.abort(cache_journal)
                    .expect("the journal was opened on this cache above");
                return Err(e);
            }
        };
        match self.publish(state, branch, kv, &mut staged, rows, cancel) {
            Ok(prefix_and_handle) => {
                state.commit_prefix(txn, 0)?;
                kv.commit(cache_journal)
                    .expect("the journal was opened on this cache above");
                let (prefix, retained) = prefix_and_handle;
                Ok(StepOutput {
                    logits,
                    retained,
                    prefix,
                })
            }
            Err(e) => {
                // Both halves, unconditionally, and neither can fail: `abort` is
                // assignment and truncation on an id this function just opened.
                state.abort(txn).expect("the transaction was opened above");
                kv.abort(cache_journal)
                    .expect("the journal was opened on this cache above");
                Err(e)
            }
        }
    }

    /// The mutating half of a step, inside an open transaction.
    ///
    /// Every failure here is undone by the caller, so this can be written in the
    /// order the work happens rather than in an order chosen to put the
    /// irreversible step last. That is the whole benefit: the pre-`execute`
    /// precondition checks that existed only because publication could not be
    /// undone are gone.
    ///
    /// `cancel` is checked **after** each of the four mutations, not only during
    /// node evaluation. Two reasons, and the fifth review found both:
    ///
    /// - R08 says a cancelled step leaves the sequence untouched. Checking only
    ///   before publication satisfies that by never having started, which is a
    ///   weaker claim than the one the transaction exists to make. Cancellation
    ///   is now an abort, and the record can say so truthfully.
    /// - It is the one fault a test can inject at each publication boundary
    ///   through the real `run`. That is deliberate: once the preconditions
    ///   moved inside the transaction, no *malformed input* can make publication
    ///   fail halfway any more, so a test that only feeds bad input can no
    ///   longer reach the error handler it is supposed to be checking.
    fn publish(
        &self,
        state: &mut SequenceState,
        branch: BranchId,
        kv: &mut KvCache,
        staged: &mut Vec<StagedAppend>,
        rows: usize,
        cancel: &Cancel,
    ) -> Result<(u64, LogitsHandle)> {
        for a in staged.drain(..) {
            kv.append(a.layer, a.position, a.key, a.value)?;
        }
        cancel.check("publish/append")?;
        state.execute(branch, rows as u64)?;
        cancel.check("publish/execute")?;
        let prefix = state.frontiers(branch)?.executed;
        // The cache now describes a longer prefix, and records the lineage of
        // each prefix it gained at the moment it gained it.
        kv.stamp(state, branch)?;
        cancel.check("publish/stamp")?;
        let retained = state.record_logits(branch, prefix)?;
        cancel.check("publish/record_logits")?;
        Ok((prefix, retained))
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
                return try_clone_slice(v.as_index()?);
            }
            if let OpParams::Rope { .. } = node.params {
                let v = values[node.inputs[1].0 as usize].as_ref().ok_or_else(|| {
                    Error::InvalidRequest {
                        field: "positions",
                        detail: "rope positions are not bound".into(),
                    }
                })?;
                return try_clone_slice(v.as_index()?);
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
        kv: &impl paged::HistorySource,
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
                let mut out = try_vec(tokens.len() * hidden as usize)?;
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
                Value::Float(HostTensor::bf16(
                    out,
                    try_shape2(tokens.len(), hidden as usize)?,
                )?)
            }
            OpParams::Linear {
                out_features, bias, ..
            } => {
                let x = input(0)?.as_float()?;
                let w = input(1)?.as_float()?;
                let b = if bias {
                    Some(try_clone_slice(input(2)?.as_float()?.data())?)
                } else {
                    None
                };
                let mut out = try_vec(x.rows() * out_features as usize)?;
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
                    try_shape2(x.rows(), out_features as usize)?,
                )?)
            }
            OpParams::RmsNorm { eps, .. } => {
                let x = input(0)?.as_float()?;
                let g = input(1)?.as_float()?;
                let mut out = try_vec(x.data().len())?;
                for r in 0..x.rows() {
                    out.extend(norm::rms_norm_row(x.row(r)?, g.data(), eps)?);
                }
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(x.shape())?)?)
            }
            OpParams::SwiGlu { .. } => {
                let gate = input(0)?.as_float()?;
                let up = input(1)?.as_float()?;
                let mut out = try_vec(gate.data().len())?;
                for r in 0..gate.rows() {
                    out.extend(activation::swiglu_row(gate.row(r)?, up.row(r)?)?);
                }
                Value::Float(HostTensor::round_to_bf16(
                    out,
                    try_clone_slice(gate.shape())?,
                )?)
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
                let mut out = try_vec(x.data().len())?;
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
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(x.shape())?)?)
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
                let mut history = kv.read_history(layer)?;
                for a in staged.iter().filter(|a| a.layer == layer) {
                    history.append(
                        a.position,
                        try_clone_slice(&a.key)?,
                        try_clone_slice(&a.value)?,
                    )?;
                }
                let mut out = try_vec(q.data().len())?;
                for (r, position) in positions.iter().enumerate().take(q.rows()) {
                    let position = *position;
                    let (key, value) = (try_clone_slice(k.row(r)?)?, try_clone_slice(v.row(r)?)?);
                    // A causal query attends to itself, so this row's key and
                    // value join the history before it is read. They also join
                    // the staged list, so a later row of the same prefill sees
                    // them and a failed step never writes them.
                    history.append(position, try_clone_slice(&key)?, try_clone_slice(&value)?)?;
                    staged.try_reserve(1).map_err(|_| Error::CapacityExceeded {
                        tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::CpuWorkspace)),
                        requested_bytes: std::mem::size_of::<StagedAppend>() as u64,
                        available_bytes: 0,
                    })?;
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
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(q.shape())?)?)
            }
            OpParams::Residual => {
                let a = input(0)?.as_float()?;
                let b = input(1)?.as_float()?;
                let mut out = try_vec(a.data().len())?;
                for r in 0..a.rows() {
                    out.extend(residual::residual_row(a.row(r)?, b.row(r)?)?);
                }
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(a.shape())?)?)
            }
            OpParams::VocabProjection { vocab, .. } => {
                let h = input(0)?.as_float()?;
                let w = input(1)?.as_float()?;
                let mut out = try_vec(h.rows() * vocab as usize)?;
                for r in 0..h.rows() {
                    out.extend(linear::linear_row(
                        h.row(r)?,
                        w.data(),
                        vocab as usize,
                        None,
                    )?);
                }
                // Not rounded. See `StepOutput::logits`.
                Value::Float(HostTensor::f32(out, try_shape2(h.rows(), vocab as usize)?)?)
            }
        };
        Ok(out)
    }
}

fn bound_precision(value: &Value) -> Option<moxie_types::Precision> {
    match value {
        Value::Float(tensor) => Some(tensor.precision()),
        Value::Index(_) => None,
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
