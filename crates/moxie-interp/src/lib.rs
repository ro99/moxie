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

use std::collections::BTreeMap;

pub mod kv;
pub mod paged;
pub mod tensor;

use moxie_graph::{
    Bindings, CombineReductionOrder, ExpertOwnership, Graph, LinearInputSlice,
    LinearReductionOrder, MlaAttentionDescriptor, Node, NodeId, OpParams, ValueId,
};
use moxie_oracles::{activation, attention, linear, mla, norm, residual, rope, route};
use moxie_state::{LogitsHandle, MlaLatentDescriptor, SequenceState};
use moxie_types::BranchId;
use moxie_types::{Error, Result};

pub use kv::{CacheId, CacheJournal, KvCache, MlaCacheRow};
pub use tensor::{HostTensor, RouteTable, Value};

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

fn validate_linear_slice(
    slice: LinearInputSlice,
    local_width: u64,
    x: &HostTensor,
    w: &HostTensor,
    blocks: u32,
) -> Result<()> {
    // A slice is one whole declared block: aligned to its own width.
    if blocks == 0
        || slice.width == 0
        || slice.width != local_width
        || slice.full_width == 0
        || !slice.full_width.is_multiple_of(u64::from(blocks))
        || slice.width != slice.full_width / u64::from(blocks)
        || slice.first > slice.full_width - slice.width
        || !slice.first.is_multiple_of(slice.width)
        || x.cols() as u64 != slice.width
        || w.cols() as u64 != slice.width
    {
        return Err(Error::InvalidArtifact {
            detail: "the linear input slice does not match its compact operands".into(),
        });
    }
    Ok(())
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
        self.run_stateless_with_linear_orders(graph, bindings, &BTreeMap::new())
    }

    /// Stateless execution with a plan-declared input-axis reduction order.
    ///
    /// The ordinary entry point remains the S=1 path. This additive entry point
    /// lets a plan make the reference's block order explicit without changing
    /// model-owned `OpParams::Linear` values.
    pub fn run_stateless_with_linear_orders(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
        orders: &BTreeMap<NodeId, LinearReductionOrder>,
    ) -> Result<StatelessTrace> {
        self.run_stateless_with_partition_orders(
            graph,
            bindings,
            orders,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
    }

    /// Stateless execution with plan-declared linear, combine, and expert
    /// partition orders. Missing entries preserve the ordinary operation.
    pub fn run_stateless_with_partition_orders(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
        linear_orders: &BTreeMap<NodeId, LinearReductionOrder>,
        combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
        expert_ownership: &BTreeMap<NodeId, ExpertOwnership>,
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
                Value::Route(_) => {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "value {} was bound a route table; a route is produced by a \
                             Route operation over this step's own rows, never supplied",
                            value.0
                        )
                        .into(),
                    });
                }
            };
            if actual != expected || bound_precision(bound) != spec.role.precision() {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "binding {} shape or precision differs from the graph",
                        value.0
                    )
                    .into(),
                });
            }
            if let Value::Float(tensor) = bound
                && tensor.data().iter().any(|element| !element.is_finite())
            {
                return Err(Error::InvalidArtifact {
                    detail: format!("binding {} contains a nonfinite value", value.0).into(),
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
            let output = self.eval(
                graph,
                node,
                &values,
                &kv,
                &mut staged,
                &[],
                linear_orders,
                combine_orders,
                expert_ownership,
            )?;
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
        self.run_with_linear_orders(graph, bindings, state, branch, kv, cancel, &BTreeMap::new())
    }

    /// Execute one step with a plan-declared linear reduction order.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_linear_orders(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
        state: &mut SequenceState,
        branch: BranchId,
        kv: &mut KvCache,
        cancel: &Cancel,
        orders: &BTreeMap<NodeId, LinearReductionOrder>,
    ) -> Result<StepOutput> {
        self.run_with_partition_orders(
            graph,
            bindings,
            state,
            branch,
            kv,
            cancel,
            orders,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
    }

    /// Execute one step with plan-declared linear, combine, and expert
    /// partition orders. Missing entries preserve the ordinary operation.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_partition_orders(
        &self,
        graph: &Graph,
        bindings: &Bindings<Value>,
        state: &mut SequenceState,
        branch: BranchId,
        kv: &mut KvCache,
        cancel: &Cancel,
        linear_orders: &BTreeMap<NodeId, LinearReductionOrder>,
        combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
        expert_ownership: &BTreeMap<NodeId, ExpertOwnership>,
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
        let mut mla_descriptor = None;
        let mut has_kv_attention = false;
        for node in graph.nodes() {
            match node.params {
                OpParams::Attention { .. } => has_kv_attention = true,
                OpParams::MlaAttention { descriptor } => {
                    let descriptor = mla_state_descriptor(descriptor)?;
                    if let Some(previous) = mla_descriptor
                        && previous != descriptor
                    {
                        return Err(Error::InvalidArtifact {
                            detail: "all MLA layers in one host cache must use the same latent \
                                     descriptor"
                                .into(),
                        });
                    }
                    mla_descriptor = Some(descriptor);
                }
                _ => {}
            }
        }
        if has_kv_attention && mla_descriptor.is_some() {
            return Err(Error::InvalidArtifact {
                detail: "a graph cannot mix conventional KV attention and MLA latent state".into(),
            });
        }
        if let Some(descriptor) = mla_descriptor {
            kv.check_mla_descriptor(descriptor)?;
        } else {
            kv.require_kv_pages()?;
        }
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
                )
                .into(),
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
                // Unreachable through a validated graph -- `moxie-plan` refuses
                // a route-role external input and no graph input is declared
                // one -- but shaped rather than panicked, because "unreachable"
                // is a claim about today's callers.
                Value::Route(t) => vec![t.rows() as u64, t.top_k() as u64],
            };
            if got != want {
                return Err(Error::InvalidArtifact {
                    detail: format!("{name} has shape {got:?} but the graph declares {want:?}")
                        .into(),
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
                        detail: format!("{name} is {} but the graph declares {p}", t.precision())
                            .into(),
                    });
                }
                (Value::Float(_), None) => {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{name} is declared an index but a tensor was bound")
                            .into(),
                    });
                }
                (Value::Index(_), Some(p)) => {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{name} is declared {p} but an index was bound").into(),
                    });
                }
                (Value::Route(_), _) => {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "{name} was bound a route table; a route is produced by a \
                             Route operation over this step's own rows, never supplied"
                        )
                        .into(),
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
                    detail: format!("{name} element {i} is {}", t.data()[i]).into(),
                });
            }
        }

        // Staged, not written through: a failed or cancelled step never wrote.
        let mut staged: Vec<StagedAppend> = Vec::new();

        for node in graph.nodes() {
            cancel.check(node.params.op().name())?;
            let out = self.eval(
                graph,
                node,
                &values,
                kv,
                &mut staged,
                &positions,
                linear_orders,
                combine_orders,
                expert_ownership,
            )?;
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
                detail: format!("{} logit rows for {rows} input row(s)", logits.rows()).into(),
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
            match a {
                StagedAppend::Kv {
                    layer,
                    position,
                    key,
                    value,
                } => kv.append(layer, position, key, value)?,
                StagedAppend::Mla {
                    layer,
                    position,
                    latent,
                    rope,
                } => kv.append_mla(
                    layer,
                    MlaCacheRow {
                        position,
                        latent,
                        rope,
                    },
                )?,
            }
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
            if let OpParams::MlaAttention { .. } = node.params {
                let v = values[node.inputs[1].0 as usize].as_ref().ok_or_else(|| {
                    Error::InvalidRequest {
                        field: "positions",
                        detail: "MLA positions are not bound".into(),
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

    #[allow(clippy::too_many_arguments)]
    fn eval(
        &self,
        graph: &Graph,
        node: &Node,
        values: &[Option<Value>],
        kv: &impl paged::HistorySource,
        staged: &mut Vec<StagedAppend>,
        positions: &[u64],
        linear_orders: &BTreeMap<NodeId, LinearReductionOrder>,
        combine_orders: &BTreeMap<NodeId, CombineReductionOrder>,
        expert_ownership: &BTreeMap<NodeId, ExpertOwnership>,
    ) -> Result<Value> {
        let input = |i: usize| -> Result<&Value> {
            values[node.inputs[i].0 as usize]
                .as_ref()
                .ok_or_else(|| Error::InvalidArtifact {
                    detail: format!(
                        "{} input {i} was used before it was produced",
                        node.params.op().name()
                    )
                    .into(),
                })
        };

        let out = match node.params {
            OpParams::Embedding {
                vocab,
                hidden,
                scale,
            } => {
                let tokens = input(0)?.as_index()?;
                let table = input(1)?.as_float()?;
                let mut out = try_vec(tokens.len() * hidden as usize)?;
                for t in tokens {
                    let id = u32::try_from(*t).map_err(|_| Error::InvalidRequest {
                        field: "token",
                        detail: format!("token {t} does not fit a u32"),
                    })?;
                    out.extend(linear::embedding_row_scaled(
                        id,
                        table.data(),
                        vocab as usize,
                        hidden as usize,
                        scale,
                    )?);
                }
                let shape = try_shape2(tokens.len(), hidden as usize)?;
                if scale == 1.0 {
                    // A copy: the stored value passes through unchanged, so
                    // this is `bf16` rather than `round_to_bf16`, and it would
                    // fail loudly if the table were not BF16-valued.
                    Value::Float(HostTensor::bf16(out, shape)?)
                } else {
                    // A scaled row is no longer the stored value, so the
                    // node-output boundary applies. Keeping `bf16` here would
                    // reject every scaled embedding as a non-BF16 payload.
                    Value::Float(HostTensor::round_to_bf16(out, shape)?)
                }
            }
            OpParams::Linear {
                in_features,
                out_features,
                bias,
            } => {
                let x = input(0)?.as_float()?;
                let w = input(1)?.as_float()?;
                let b = if bias {
                    Some(try_clone_slice(input(2)?.as_float()?.data())?)
                } else {
                    None
                };
                let mut out = try_vec(x.rows() * out_features as usize)?;
                let order = linear_orders.get(&node.id).copied();
                if let Some(LinearReductionOrder {
                    blocks,
                    slice: Some(slice),
                }) = order
                {
                    validate_linear_slice(slice, in_features, x, w, blocks)?;
                }
                if let Some(order) = order
                    && order.slice.is_some()
                    && bias
                {
                    return Err(Error::Unsupported {
                        capability: "biased_input_axis_linear",
                        reason: "a rank-local reduction cannot apply a bias more than once".into(),
                    });
                }
                for r in 0..x.rows() {
                    let row = x.row(r)?;
                    let values = match order {
                        Some(LinearReductionOrder {
                            blocks,
                            slice: None,
                        }) => linear::linear_row_ordered(
                            row,
                            w.data(),
                            out_features as usize,
                            b.as_deref(),
                            blocks,
                        )?,
                        Some(LinearReductionOrder {
                            blocks: _,
                            slice: Some(_slice),
                        }) => linear::linear_row_partial(row, w.data(), out_features as usize)?,
                        None => {
                            linear::linear_row(row, w.data(), out_features as usize, b.as_deref())?
                        }
                    };
                    out.extend(values);
                }
                if let Some(LinearReductionOrder {
                    blocks,
                    slice: Some(_),
                }) = order
                {
                    if blocks == 0 {
                        return Err(Error::InvalidRequest {
                            field: "linear_blocks",
                            detail: "the declared reduction block count must be positive".into(),
                        });
                    }
                    return Ok(Value::Float(HostTensor::f32(
                        out,
                        try_shape2(x.rows(), out_features as usize)?,
                    )?));
                }
                Value::Float(HostTensor::round_to_bf16(
                    out,
                    try_shape2(x.rows(), out_features as usize)?,
                )?)
            }
            OpParams::RmsNorm { eps, group, .. } => {
                let x = input(0)?.as_float()?;
                let g = input(1)?.as_float()?;
                let mut out = try_vec(x.data().len())?;
                for r in 0..x.rows() {
                    out.extend(norm::rms_norm_row_grouped(
                        x.row(r)?,
                        g.data(),
                        group as usize,
                        eps,
                    )?);
                }
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(x.shape())?)?)
            }
            OpParams::SwiGlu { .. } | OpParams::GeGlu { .. } => {
                let geglu = matches!(node.params, OpParams::GeGlu { .. });
                let gate = input(0)?.as_float()?;
                let up = input(1)?.as_float()?;
                let mut out = try_vec(gate.data().len())?;
                for r in 0..gate.rows() {
                    let (g, u) = (gate.row(r)?, up.row(r)?);
                    out.extend(if geglu {
                        activation::geglu_row(g, u)?
                    } else {
                        activation::swiglu_row(g, u)?
                    });
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
                frequency_dim,
                base,
                layout,
            } => {
                let x = input(0)?.as_float()?;
                let pos = input(1)?.as_index()?;
                if pos.len() != x.rows() {
                    return Err(Error::InvalidArtifact {
                        detail: format!("{} positions for {} rows", pos.len(), x.rows()).into(),
                    });
                }
                let mut out = try_vec(x.data().len())?;
                for (r, p) in pos.iter().enumerate().take(x.rows()) {
                    out.extend(rope::rope_row(
                        x.row(r)?,
                        *p,
                        heads as usize,
                        head_dim as usize,
                        rope::Rotation {
                            base,
                            rotary_dim: rotary_dim as usize,
                            frequency_dim: frequency_dim as usize,
                            layout,
                        },
                    )?);
                }
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(x.shape())?)?)
            }
            OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                scale,
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
                for a in staged.iter() {
                    let StagedAppend::Kv {
                        layer: staged_layer,
                        position,
                        key,
                        value,
                    } = a
                    else {
                        continue;
                    };
                    if *staged_layer != layer {
                        continue;
                    }
                    history.append(*position, try_clone_slice(key)?, try_clone_slice(value)?)?;
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
                    staged.push(StagedAppend::Kv {
                        layer,
                        position,
                        key,
                        value,
                    });
                    out.extend(attention::attend_multi_head(
                        q.row(r)?,
                        &history,
                        position,
                        attention::Heads {
                            query: heads as usize,
                            key_value: kv_heads as usize,
                            head_dim: head_dim as usize,
                            scale,
                        },
                        visibility,
                    )?);
                }
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(q.shape())?)?)
            }
            OpParams::MlaAttention { descriptor } => {
                if let Some(order) = combine_orders.get(&node.id)
                    && (order.groups == 0
                        || match order.owned {
                            None => !descriptor.heads.is_multiple_of(u64::from(order.groups)),
                            Some(owner) => owner >= order.groups,
                        })
                {
                    return Err(Error::InvalidRequest {
                        field: "mla_order",
                        detail: "groups or owned head group do not match the MLA geometry".into(),
                    });
                }
                let hidden = input(0)?.as_float()?;
                let q_a_proj = widen(input(2)?.as_float()?.data())?;
                let q_a_layernorm = widen(input(3)?.as_float()?.data())?;
                let q_b_proj = widen(input(4)?.as_float()?.data())?;
                let kv_a_proj_with_mqa = widen(input(5)?.as_float()?.data())?;
                let kv_a_layernorm = widen(input(6)?.as_float()?.data())?;
                let kv_b_proj = widen(input(7)?.as_float()?.data())?;
                let o_proj = widen(input(8)?.as_float()?.data())?;
                let weights = mla::MlaWeights {
                    q_a_proj: &q_a_proj,
                    q_a_layernorm: &q_a_layernorm,
                    q_b_proj: &q_b_proj,
                    kv_a_proj_with_mqa: &kv_a_proj_with_mqa,
                    kv_a_layernorm: &kv_a_layernorm,
                    kv_b_proj: &kv_b_proj,
                    o_proj: &o_proj,
                };
                let mut history = kv.read_mla_history(descriptor.layer)?;
                for staged_row in staged.iter() {
                    let StagedAppend::Mla {
                        layer,
                        position,
                        latent,
                        rope,
                    } = staged_row
                    else {
                        continue;
                    };
                    if *layer == descriptor.layer {
                        push_mla_history(&mut history, *position, latent, rope)?;
                    }
                }
                let mut out = try_vec(hidden.rows() * descriptor.hidden as usize)?;
                for (row, position) in positions.iter().enumerate().take(hidden.rows()) {
                    let input = widen(hidden.row(row)?)?;
                    let projection = mla::project(descriptor, &input, weights, *position)?;
                    let token = projection.cached_token()?;
                    let latent = round_bf16(&token.latent)?;
                    let rope = round_bf16(&token.rope)?;
                    push_mla_history(&mut history, *position, &latent, &rope)?;
                    staged.try_reserve(1).map_err(|_| Error::CapacityExceeded {
                        tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::CpuWorkspace)),
                        requested_bytes: std::mem::size_of::<StagedAppend>() as u64,
                        available_bytes: 0,
                    })?;
                    staged.push(StagedAppend::Mla {
                        layer: descriptor.layer,
                        position: *position,
                        latent,
                        rope,
                    });
                    let result =
                        mla::attend(descriptor, &projection, &history, &kv_b_proj, &o_proj)?;
                    if let Some(order) = combine_orders.get(&node.id) {
                        let groups = if order.owned.is_some() {
                            1
                        } else {
                            order.groups as usize
                        };
                        out.extend(mla::o_proj_grouped(
                            &o_proj,
                            descriptor.hidden as usize,
                            descriptor.heads as usize * descriptor.v_head_dim as usize,
                            &result.head_output,
                            groups,
                        )?);
                    } else {
                        out.extend(result.output.iter().map(|value| *value as f32));
                    }
                }
                let shape = try_shape2(hidden.rows(), descriptor.hidden as usize)?;
                match combine_orders.get(&node.id) {
                    Some(order) if order.owned.is_some() => {
                        Value::Float(HostTensor::f32(out, shape)?)
                    }
                    _ => Value::Float(HostTensor::round_to_bf16(out, shape)?),
                }
            }
            OpParams::Residual { scale } => {
                let a = input(0)?.as_float()?;
                let b = input(1)?.as_float()?;
                let mut out = try_vec(a.data().len())?;
                for r in 0..a.rows() {
                    out.extend(residual::residual_row_scaled(a.row(r)?, b.row(r)?, scale)?);
                }
                Value::Float(HostTensor::round_to_bf16(out, try_clone_slice(a.shape())?)?)
            }
            OpParams::VocabProjection { vocab, softcap, .. } => {
                let h = input(0)?.as_float()?;
                let w = input(1)?.as_float()?;
                let mut out = try_vec(h.rows() * vocab as usize)?;
                for r in 0..h.rows() {
                    let row = linear::linear_row(h.row(r)?, w.data(), vocab as usize, None)?;
                    // The cap's own BF16 boundaries are internal to it and live
                    // in the oracle. The logits themselves stay FP32 either
                    // way: capping bends the distribution the sampler sees, it
                    // does not change what that distribution is stored as.
                    out.extend(match softcap {
                        Some(cap) => activation::softcap_row(&row, cap)?,
                        None => row,
                    });
                }
                // Not rounded. See `StepOutput::logits`.
                Value::Float(HostTensor::f32(out, try_shape2(h.rows(), vocab as usize)?)?)
            }
            OpParams::Route {
                hidden,
                experts,
                top_k,
                input: router_input,
                score,
                per_expert_scale,
                selection_bias,
                coefficient,
            } => {
                // Resolved from the operation's own operand list rather than
                // from positions written out here. Two of the optional operands
                // are `[experts]` floats, so a second enumeration that drifted
                // by one would bind a selection bias where a coefficient scale
                // belongs and every shape check would still pass.
                let mut rows = None;
                let mut gain = None;
                let mut proj = None;
                let mut scale = None;
                let mut bias = None;
                let operands = node.params.route_operands();
                for (i, operand) in operands.as_slice().iter().enumerate() {
                    let slot = match operand {
                        moxie_graph::RouteOperand::Rows => &mut rows,
                        moxie_graph::RouteOperand::Projection => &mut proj,
                        moxie_graph::RouteOperand::Gain => &mut gain,
                        moxie_graph::RouteOperand::PerExpertScale => &mut scale,
                        moxie_graph::RouteOperand::SelectionBias => &mut bias,
                    };
                    *slot = Some(input(i)?.as_float()?);
                }
                let (Some(x), Some(proj)) = (rows, proj) else {
                    return Err(Error::InvalidArtifact {
                        detail: "a router without rows or a projection".into(),
                    });
                };
                debug_assert_eq!(per_expert_scale, scale.is_some());
                debug_assert_eq!(selection_bias, bias.is_some());
                // `hidden` is checked structurally when the node is built; the
                // oracle reads the row's own width.
                debug_assert_eq!(x.row(0).map(<[f32]>::len).unwrap_or(0), hidden as usize);
                let mut ids = try_vec(x.rows() * top_k as usize)?;
                let mut weights = try_vec(x.rows() * top_k as usize)?;
                for r in 0..x.rows() {
                    let route = route::router_route_row(
                        x.row(r)?,
                        gain.map(|g| g.data()),
                        proj.data(),
                        scale.map(|s| s.data()),
                        bias.map(|b| b.data()),
                        route::RouterSpec {
                            experts: experts as usize,
                            top_k: top_k as usize,
                            input: router_input,
                            score,
                            coefficient,
                        },
                    )?;
                    ids.extend(route.experts);
                    weights.extend(route.weights);
                }
                // Whether the coefficients were narrowed to BF16 is the
                // router's own `coefficient` parameter, applied by
                // `router_route_row` as the reference's last statement. The
                // route table still *stores* FP32 either way, and the declared
                // output role still says FP32, because that role is what a byte
                // trace reconciles against the ledger -- narrowing a value and
                // narrowing a buffer are different claims.
                Value::Route(RouteTable::new(top_k as usize, ids, weights)?)
            }
            OpParams::ExpertMlp {
                hidden,
                intermediate,
                experts,
                top_k,
                activation,
            } => {
                let x = input(0)?.as_float()?;
                let table = input(1)?.as_route()?;
                let gate_up = input(2)?.as_float()?;
                let down = input(3)?.as_float()?;
                if table.rows() != x.rows() {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "the route table covers {} rows but {} were supplied",
                            table.rows(),
                            x.rows()
                        )
                        .into(),
                    });
                }
                let spec = route::ExpertSpec {
                    experts: experts as usize,
                    hidden: hidden as usize,
                    intermediate: intermediate as usize,
                    activation,
                };
                if let Some(ownership) = expert_ownership.get(&node.id).copied() {
                    if ownership.groups == 0 || ownership.owned >= ownership.groups {
                        return Err(Error::InvalidRequest {
                            field: "expert_ownership",
                            detail: "the owned group must be inside a positive group count".into(),
                        });
                    }
                    let total = experts
                        .checked_mul(u64::from(ownership.groups))
                        .ok_or(moxie_types::DimError::Overflow)?;
                    let first = experts
                        .checked_mul(u64::from(ownership.owned))
                        .ok_or(moxie_types::DimError::Overflow)?;
                    let total =
                        usize::try_from(total).map_err(|_| moxie_types::DimError::Overflow)?;
                    let first =
                        usize::try_from(first).map_err(|_| moxie_types::DimError::Overflow)?;
                    if table
                        .experts()
                        .iter()
                        .any(|expert| *expert as usize >= total)
                    {
                        return Err(Error::InvalidArtifact {
                            detail: "the route selected an expert outside the declared ownership"
                                .into(),
                        });
                    }
                    let width = hidden as usize;
                    let top_k = top_k as usize;
                    let length = x
                        .rows()
                        .checked_mul(top_k)
                        .and_then(|n| n.checked_mul(width))
                        .ok_or(moxie_types::DimError::Overflow)?;
                    let mut out = try_vec(length)?;
                    out.resize(length, 0.0);
                    for r in 0..x.rows() {
                        for (slot, &expert) in table.row_experts(r)?.iter().enumerate() {
                            let expert = expert as usize;
                            if (first..first + spec.experts).contains(&expert) {
                                let value = route::expert_row(
                                    x.row(r)?,
                                    gate_up.data(),
                                    down.data(),
                                    (expert - first) as u32,
                                    spec,
                                )?;
                                let start = (r * top_k + slot) * width;
                                out[start..start + width].copy_from_slice(&value);
                            }
                        }
                    }
                    Value::Float(HostTensor::round_to_bf16(
                        out,
                        try_shape2(x.rows() * top_k, width)?,
                    )?)
                } else {
                    let mut out = try_vec(x.rows() * top_k as usize * hidden as usize)?;
                    for r in 0..x.rows() {
                        // Slot-major, in the row's own selection order. A grouped
                        // kernel is free to visit the same work expert-major -- that
                        // is exactly what `route::dispatch` describes -- but it owes
                        // this scatter, because slot `j` belongs to the expert the
                        // route chose at position `j`.
                        for e in table.row_experts(r)? {
                            out.extend(route::expert_row(
                                x.row(r)?,
                                gate_up.data(),
                                down.data(),
                                *e,
                                spec,
                            )?);
                        }
                    }
                    Value::Float(HostTensor::round_to_bf16(
                        out,
                        try_shape2(x.rows() * top_k as usize, hidden as usize)?,
                    )?)
                }
            }
            OpParams::Combine {
                hidden,
                top_k,
                order,
                output_scale,
            } => {
                let table = input(0)?.as_route()?;
                let slots = input(1)?.as_float()?;
                if slots.rows() != table.rows() * top_k as usize {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "the slot tensor has {} rows but this route needs {}",
                            slots.rows(),
                            table.rows() * top_k as usize
                        )
                        .into(),
                    });
                }
                let width = hidden as usize;
                let mut out = try_vec(table.rows() * width)?;
                let reduction = combine_orders.get(&node.id).copied();
                let experts_total = if let Some(reduction) = reduction {
                    let producer = graph
                        .nodes()
                        .iter()
                        .find(|candidate| candidate.output == node.inputs[1])
                        .ok_or_else(|| Error::InvalidArtifact {
                            detail: "combine slots have no producing node".into(),
                        })?;
                    let OpParams::ExpertMlp { experts, .. } = producer.params else {
                        return Err(Error::InvalidArtifact {
                            detail: "combine slots are not produced by ExpertMlp".into(),
                        });
                    };
                    let ownership = expert_ownership.get(&producer.id).copied();
                    if ownership.is_some_and(|owner| {
                        owner.groups != reduction.groups || reduction.owned != Some(owner.owned)
                    }) {
                        return Err(Error::InvalidRequest {
                            field: "combine_groups",
                            detail: "ExpertMlp ownership and Combine order disagree".into(),
                        });
                    }
                    let groups = ownership.map_or(1, |owner| owner.groups);
                    let total = experts
                        .checked_mul(u64::from(groups))
                        .ok_or(moxie_types::DimError::Overflow)?;
                    usize::try_from(total).map_err(|_| moxie_types::DimError::Overflow)?
                } else {
                    0
                };
                for r in 0..table.rows() {
                    let base = r * top_k as usize * width;
                    let row_experts = table.row_experts(r)?;
                    let row_weights = table.row_weights(r)?;
                    let row_slots = &slots.data()[base..base + top_k as usize * width];
                    if let Some(reduction) = reduction {
                        out.extend(route::combine_row_ordered(
                            row_experts,
                            row_weights,
                            row_slots,
                            width,
                            order,
                            output_scale,
                            experts_total,
                            reduction.groups,
                            reduction.owned,
                        )?);
                    } else {
                        out.extend(route::combine_row(
                            row_experts,
                            row_weights,
                            row_slots,
                            width,
                            order,
                            output_scale,
                        )?);
                    }
                }
                let shape = try_shape2(table.rows(), width)?;
                if reduction.is_some_and(|order| order.owned.is_some()) {
                    Value::Float(HostTensor::f32(out, shape)?)
                } else {
                    Value::Float(HostTensor::round_to_bf16(out, shape)?)
                }
            }
        };
        Ok(out)
    }
}

fn bound_precision(value: &Value) -> Option<moxie_types::Precision> {
    match value {
        Value::Float(tensor) => Some(tensor.precision()),
        // Neither has a single precision. A route carries two, and reporting
        // either one of them would let a caller believe it had checked both.
        Value::Index(_) | Value::Route(_) => None,
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

fn mla_state_descriptor(descriptor: MlaAttentionDescriptor) -> Result<MlaLatentDescriptor> {
    let kv_lora_rank =
        usize::try_from(descriptor.kv_lora_rank).map_err(|_| moxie_types::DimError::Overflow)?;
    let qk_rope_head_dim = usize::try_from(descriptor.qk_rope_head_dim)
        .map_err(|_| moxie_types::DimError::Overflow)?;
    MlaLatentDescriptor::new(kv_lora_rank, qk_rope_head_dim, descriptor.cache_precision)
}

fn widen(values: &[f32]) -> Result<Vec<f64>> {
    let mut output = try_vec(values.len())?;
    output.extend(values.iter().map(|value| *value as f64));
    Ok(output)
}

fn round_bf16(values: &[f64]) -> Result<Vec<f32>> {
    let mut output = try_vec(values.len())?;
    output.extend(values.iter().map(|value| tensor::to_bf16(*value as f32)));
    Ok(output)
}

fn push_mla_history(
    history: &mut Vec<mla::MlaCachedToken>,
    position: u64,
    latent: &[f32],
    rope: &[f32],
) -> Result<()> {
    history
        .try_reserve(1)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(moxie_types::Tier::Host(moxie_types::HostTier::CpuWorkspace)),
            requested_bytes: std::mem::size_of::<mla::MlaCachedToken>() as u64,
            available_bytes: 0,
        })?;
    history.push(mla::MlaCachedToken {
        position,
        latent: widen(latent)?,
        rope: widen(rope)?,
    });
    Ok(())
}

/// A state append held until the whole step succeeds.
#[derive(Debug, Clone)]
enum StagedAppend {
    Kv {
        layer: u32,
        position: u64,
        key: Vec<f32>,
        value: Vec<f32>,
    },
    Mla {
        layer: u32,
        position: u64,
        latent: Vec<f32>,
        rope: Vec<f32>,
    },
}
