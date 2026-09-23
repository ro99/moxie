//! One tensor-parallel dense step across a two-rank group (task 0060, M5.2).
//!
//! The step runs task 0058's lowering stage by stage. Each rank runs its own
//! stage graph as one admitted dense plan. A `Local` stage's two outputs meet
//! in [`RankGroup::gather`] or the exact [`RankGroup::reduce`]; a `Replicated`
//! stage runs on both ranks. Every value a later stage reads stays in one
//! ledger-admitted boundary arena per rank for the length of the step, and is
//! copied into the next stage plan's own input range on that rank's stream.
//!
//! Atomicity follows Strata's rank step (`dsv4-rank-local-architecture.md`,
//! "Failure and rollback"): each rank's KV appends for the whole step ride one
//! transaction. Every failure path, and the end of every step, goes through
//! one [`settle`]: both streams are observed drained within the group's
//! deadline, then retained leases are reclaimed, plans and boundaries closed
//! and the transactions aborted, and the group stays usable. If a rank cannot
//! be observed, a commit's publication fails or a cleanup cannot complete,
//! everything is withheld and the group is lost. [`DenseStep::commit`] prepares both ranks
//! before applying either; dropping an uncommitted step aborts both.

#![cfg(all(feature = "driver", feature = "paged-attention-binding"))]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;

use moxie_cuda::{RankContext, Stream};
use moxie_graph::{Graph, LinearInputSlice, OracleId, OracleRegistry, ValueId, ValueRole};
use moxie_memory::{BufferRequest, Ledger, PlanRequest, StageSpan};
use moxie_plan::{
    Join, Phase, ResourceWorkload, Stage, StageGraph, TensorParallelLowering, build_stage_graph,
    lower_selected_ordered,
};
use moxie_state::DeviceKvSequence;
use moxie_types::{
    DeviceCapability, DeviceTier, Dim, Error, KernelCatalogue, Result, Scope, StateTransactionId,
    Tier,
};

use crate::arena::{DeviceArena, DeviceRange, OperationLease};
use crate::chain::{OwnedBinding, SelectedAdmitRefused, SelectedCompletion, SelectedReservedPlan};
use crate::dense::{DenseGraphStep, DenseOperation};
use crate::paged_attention::device::{PagedAttentionRun, PairCommitRefused, commit_paged_pair};
use crate::tensor_parallel::{GatherDeclaration, RankGroup};

const ALIGNMENT: u64 = 256;

/// One rank's device and state authorities for a step.
#[derive(Debug)]
pub struct DenseRank<'r, 'ctx> {
    pub ctx: &'ctx RankContext,
    pub capability: &'r DeviceCapability,
    pub stream: &'r Stream<'ctx>,
    pub ledger: &'r mut Ledger,
    pub state: &'r mut DeviceKvSequence,
}

/// One step of a lowered dense graph on both ranks.
pub struct DenseTensorParallelStep<'s> {
    pub graph: &'s Graph,
    pub lowering: &'s TensorParallelLowering,
    pub oracle: OracleId,
    pub oracles: &'s OracleRegistry,
    pub catalogue: &'s KernelCatalogue,
    pub rows: u64,
    pub visible_tokens: u64,
    /// Host bindings for rank `r`'s stage graph: its weights, compacted as the
    /// graph's `weights` say, and any original graph input it reads.
    pub bindings: &'s mut dyn FnMut(usize, &StageGraph) -> Result<Vec<OwnedBinding>>,
    pub cancel: &'s AtomicBool,
}

impl core::fmt::Debug for DenseTensorParallelStep<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DenseTensorParallelStep")
            .field("rows", &self.rows)
            .field("visible_tokens", &self.visible_tokens)
            .finish_non_exhaustive()
    }
}

/// An executed step whose transactions are still open. Acceptance is the
/// caller's decision after sampling, so the logits are readable here, and a
/// commit is the only way to keep the step; dropping it aborts both ranks.
#[derive(Debug)]
#[must_use = "dropping an uncommitted step aborts it"]
pub struct DenseStep<'g, 'r, 'ctx> {
    group: &'g mut RankGroup<'ctx>,
    ranks: [DenseRank<'r, 'ctx>; 2],
    transactions: [StateTransactionId; 2],
    logits: Vec<u8>,
    settled: bool,
}

impl<'ctx> DenseStep<'_, '_, 'ctx> {
    /// Rank 0's gathered FP32 logits, to sample from before committing.
    pub fn logits(&self) -> &[u8] {
        &self.logits
    }

    /// Commit both ranks or neither, then return rank 0's FP32 logits.
    ///
    /// Both ranks prepare (every fallible host step) before either applies.
    /// A prepare refusal settles and aborts both. An apply failure is a
    /// device publication failure after rank 0 may have committed: the group
    /// is lost and the reason recorded.
    pub fn commit(mut self, runs: [&mut [PagedAttentionRun<'ctx>]; 2]) -> Result<Vec<u8>> {
        self.settled = true;
        let [first, second] = &mut self.ranks;
        let streams = [first.stream, second.stream];
        match commit_paged_pair(
            [&mut *first.state, &mut *second.state],
            self.transactions,
            runs,
            streams,
        ) {
            Ok(()) => Ok(core::mem::take(&mut self.logits)),
            Err(PairCommitRefused::Prepare(error)) => {
                settle(
                    self.group,
                    &mut self.ranks,
                    [&mut [], &mut []],
                    Held::none(),
                    Some(self.transactions),
                )?;
                Err(error)
            }
            Err(PairCommitRefused::Apply(error)) => Err(self.group.lose(format!(
                "a rank's commit publication failed after both prepared ({error}); the \
                 ranks may have diverged"
            ))),
        }
    }
}

impl Drop for DenseStep<'_, '_, '_> {
    fn drop(&mut self) {
        if !self.settled {
            let _ = settle(
                self.group,
                &mut self.ranks,
                [&mut [], &mut []],
                Held::none(),
                Some(self.transactions),
            );
        }
    }
}

/// The step's per-rank boundary values and the arena that holds them.
#[derive(Debug)]
struct Boundary<'ctx> {
    arena: DeviceArena<'ctx>,
    values: BTreeMap<ValueId, DeviceRange<'ctx>>,
}

/// Everything one rank holds until the step settles.
#[derive(Debug, Default)]
struct Held<'ctx> {
    plans: Vec<SelectedReservedPlan<'ctx>>,
    /// Stage leases that refused; retained here, never dropped, until the
    /// drain lets them retire.
    leases: Vec<OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>>,
    boundary: Option<Boundary<'ctx>>,
}

impl Held<'_> {
    fn none() -> [Self; 2] {
        [Self::default(), Self::default()]
    }
}

/// The one settle routine of a tensor-parallel step, and the rule every
/// failure path follows. Nothing is released until [`RankGroup::drain`] has
/// observed both streams complete. Then every retained lease is reclaimed,
/// every plan and boundary closes, `abort`'s transactions are aborted, and
/// the group stays usable. The group is lost instead, with everything
/// withheld (dropped unreleased) and no transaction touched, only if the
/// drain fails or a close cannot complete.
fn settle<'ctx>(
    group: &mut RankGroup<'ctx>,
    ranks: &mut [DenseRank<'_, 'ctx>; 2],
    runs: [&mut [PagedAttentionRun<'ctx>]; 2],
    held: [Held<'ctx>; 2],
    abort: Option<[StateTransactionId; 2]>,
) -> Result<()> {
    group.drain([ranks[0].stream, ranks[1].stream])?;
    for ((rank, runs), held) in ranks.iter_mut().zip(runs).zip(held) {
        let Held {
            mut plans,
            leases,
            boundary,
        } = held;
        // Drained: a refused stage's lease, lost or not, and a run this step
        // quarantined (it entered idle, `execute_dense` checks that, and was
        // used on this rank's stream alone) hold nothing a stream can still
        // touch. The commit and drop paths pass no runs.
        plans.extend(
            leases
                .into_iter()
                .filter_map(|lease| lease.reclaim_drained().plan),
        );
        for run in runs.iter_mut() {
            match run.reclaim_drained() {
                Ok(Some((query, output))) => {
                    for range in [query, output] {
                        return_to_plans(group, &mut plans, range)?;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    return Err(group.lose(format!(
                        "an attention run cannot be reclaimed ({error}); it stays quarantined"
                    )));
                }
            }
        }
        close_plans(group, rank.ledger, &mut plans)?;
        if let Some(boundary) = boundary {
            close_boundary(group, rank.ledger, boundary)?;
        }
    }
    if let Some(transactions) = abort {
        for (rank, transaction) in ranks.iter_mut().zip(transactions) {
            let _ = rank.state.abort(transaction);
        }
    }
    Ok(())
}

impl<'ctx> RankGroup<'ctx> {
    /// Execute one step. On success both transactions stay open in the
    /// returned [`DenseStep`]; on failure both are settled and aborted.
    pub fn execute_dense<'g, 'r>(
        &'g mut self,
        ranks: [DenseRank<'r, 'ctx>; 2],
        runs: [&mut [PagedAttentionRun<'ctx>]; 2],
        mut step: DenseTensorParallelStep<'_>,
    ) -> Result<DenseStep<'g, 'r, 'ctx>> {
        if let Some(lost) = self.lost() {
            return Err(lost.clone());
        }
        // Every boundary size is checked before either transaction begins,
        // so this refusal leaves nothing to settle.
        let bytes = boundary_bytes(step.graph, step.rows)?;
        // A step accepts only idle runs. It then uses each run on its rank's
        // stream alone, so a run `settle` finds quarantined was quarantined by
        // this step, on a stream `settle` has observed drained.
        if runs
            .iter()
            .flat_map(|runs| runs.iter())
            .any(|run| !run.is_idle())
        {
            return Err(invalid(
                "runs",
                "a supplied attention run is quarantined or holds a refused source; a step \
                 accepts only idle runs",
            ));
        }
        let mut ranks = ranks;
        let [runs0, runs1] = runs;
        let first = ranks[0].state.begin()?;
        let transactions = match ranks[1].state.begin() {
            Ok(second) => [first, second],
            Err(error) => {
                let _ = ranks[0].state.abort(first);
                return Err(error);
            }
        };
        let mut held = Held::none();
        let result = (|| {
            for (rank, held) in ranks.iter_mut().zip(held.iter_mut()) {
                held.boundary = Some(open_boundary(self, rank, bytes)?);
            }
            let runs = [&mut *runs0, &mut *runs1];
            self.run_stages(&mut ranks, runs, &mut step, transactions, &mut held)
        })();
        let runs = [runs0, runs1];
        match result {
            Ok(logits) => {
                // Every settle error has already lost the group.
                settle(self, &mut ranks, runs, held, None)?;
                Ok(DenseStep {
                    group: self,
                    ranks,
                    transactions,
                    logits,
                    settled: false,
                })
            }
            Err(error) => Err(settle(self, &mut ranks, runs, held, Some(transactions))
                .err()
                .unwrap_or(error)),
        }
    }

    /// Run every stage. Each stage plan or refused lease is pushed to `held`
    /// as soon as it exists, so a failure leaves it to [`settle`].
    fn run_stages(
        &mut self,
        ranks: &mut [DenseRank<'_, 'ctx>; 2],
        runs: [&mut [PagedAttentionRun<'ctx>]; 2],
        step: &mut DenseTensorParallelStep<'_>,
        transactions: [StateTransactionId; 2],
        held: &mut [Held<'ctx>; 2],
    ) -> Result<Vec<u8>> {
        let (graph, lowering) = (step.graph, step.lowering);
        let streams = [ranks[0].stream, ranks[1].stream];
        for declared in &lowering.stages {
            match declared {
                // ponytail: one plan per replicated node, so every value a
                // later stage reads is some plan's output; cut only at
                // live-outs if per-node admission ever costs.
                Stage::Replicated(nodes) => {
                    for node in nodes.clone() {
                        let stage = build_stage_graph(
                            graph,
                            None,
                            node..node + 1,
                            None,
                            step.oracle,
                            step.oracles,
                        )?;
                        let original = graph.nodes()[node].output;
                        for (r, rank) in ranks.iter_mut().enumerate() {
                            run_stage(
                                self,
                                r,
                                rank,
                                runs[r],
                                step,
                                &stage,
                                transactions[r],
                                &mut held[r],
                            )?;
                            let held = &mut held[r];
                            let plan = held.plans.last().expect("the stage plan is open");
                            let boundary = held.boundary.as_mut().expect("the boundary is open");
                            keep(plan, stage.graph.output(), original, boundary, rank.stream)?;
                        }
                        // The copies read the plans' ranges: settle before closing them.
                        self.drain(streams)?;
                        for (rank, held) in ranks.iter_mut().zip(held.iter_mut()) {
                            close_plans(self, rank.ledger, &mut held.plans)?;
                        }
                    }
                }
                Stage::Local { nodes, join } => {
                    let (Join::Gather { output } | Join::Reduce { output }) = *join;
                    let mut declarations = Vec::new();
                    for (r, rank) in ranks.iter_mut().enumerate() {
                        let stage = build_stage_graph(
                            graph,
                            Some(&lowering.ranks[r]),
                            nodes.clone(),
                            Some(output),
                            step.oracle,
                            step.oracles,
                        )?;
                        run_stage(
                            self,
                            r,
                            rank,
                            runs[r],
                            step,
                            &stage,
                            transactions[r],
                            &mut held[r],
                        )?;
                        let plan = held[r].plans.last().expect("the stage plan is open");
                        declarations.push((
                            declare(self.next_sequence(), plan, stage.graph.output())?,
                            stage.graph.output(),
                        ));
                    }
                    let [(d0, v0), (d1, v1)] = [declarations[0], declarations[1]];
                    let [h0, h1] = held;
                    let sources = [
                        h0.plans
                            .last()
                            .expect("rank 0 stage plan")
                            .range_for_value(v0)?,
                        h1.plans
                            .last()
                            .expect("rank 1 stage plan")
                            .range_for_value(v1)?,
                    ];
                    let [b0, b1] = [&mut h0.boundary, &mut h1.boundary]
                        .map(|boundary| boundary.as_mut().expect("the boundary is open"));
                    let arenas = [&mut b0.arena, &mut b1.arena];
                    let joined = match join {
                        Join::Gather { .. } => {
                            self.gather([d0, d1], sources, arenas, streams, step.cancel)
                        }
                        Join::Reduce { .. } => {
                            self.reduce([d0, d1], sources, arenas, streams, step.cancel)
                        }
                    }
                    .map_err(|refused| refused.errors[0].clone())?;
                    let [j0, j1] = joined;
                    b0.values.insert(output, j0);
                    b1.values.insert(output, j1);
                    for (rank, held) in ranks.iter_mut().zip(held.iter_mut()) {
                        close_plans(self, rank.ledger, &mut held.plans)?;
                    }
                }
            }
        }
        let logits = held[0]
            .boundary
            .as_ref()
            .expect("the boundary is open")
            .values
            .get(&graph.output())
            .ok_or_else(|| invalid("logits", "the lowering did not gather the graph output"))?;
        let mut bytes = vec![0; value_bytes(graph, graph.output(), step.rows, 4)? as usize];
        logits.copy_to_host(&mut bytes)?;
        Ok(bytes)
    }
}

/// Admit one rank's stage plan, fill its inputs and run it to completion.
#[allow(clippy::too_many_arguments)]
fn run_stage<'ctx>(
    group: &mut RankGroup<'ctx>,
    r: usize,
    rank: &mut DenseRank<'_, 'ctx>,
    runs: &mut [PagedAttentionRun<'ctx>],
    step: &mut DenseTensorParallelStep<'_>,
    stage: &StageGraph,
    transaction: StateTransactionId,
    held: &mut Held<'ctx>,
) -> Result<()> {
    let boundary = held.boundary.as_ref().expect("the boundary is open");
    // Attention runs are indexed by the layer each stage-local attention
    // node had in the full graph; a stage without attention takes none.
    let runs = if stage.state_layers.is_empty() {
        &mut []
    } else {
        runs
    };
    let workload = ResourceWorkload {
        phase: if step.rows == 1 {
            Phase::Decode
        } else {
            Phase::Prefill
        },
        rows: step.rows,
        visible_tokens: step.visible_tokens,
        branch_rows: step.rows,
        output: stage.graph.output(),
        device: rank.capability.uuid,
        paged_state_capacity: None,
    };
    let candidate = lower_selected_ordered(
        &stage.graph,
        workload,
        rank.capability,
        step.catalogue,
        &stage.linear_orders,
    )?;
    let plan = SelectedReservedPlan::admit(
        candidate,
        &stage.graph,
        rank.capability,
        step.catalogue,
        rank.ledger,
        rank.ctx,
    )
    .map_err(|refused| match refused {
        SelectedAdmitRefused::Invalid { error, .. } => error,
        SelectedAdmitRefused::Rejected { rejection, .. } => rejection.into(),
        // Admission could not undo its own reservation: withheld, and so
        // the group cannot be reused.
        SelectedAdmitRefused::Held { error, .. } => group.lose(format!(
            "rank {r}: a refused stage admission could not release what it held ({error})"
        )),
    })?;
    let mut resident = BTreeSet::new();
    let bindings = stage
        .reads
        .iter()
        .try_for_each(|read| {
            let Some(source) = boundary.values.get(&read.original) else {
                return Ok(());
            };
            resident.insert(read.local);
            copy_boundary(
                &plan,
                read.local,
                source,
                read.slice,
                step.rows,
                rank.stream,
            )
        })
        .and_then(|()| (step.bindings)(r, stage));
    let bindings = match bindings {
        Ok(bindings) => bindings,
        Err(error) => {
            held.plans.push(plan);
            return Err(error);
        }
    };
    let lease = plan.execute_dense_stage(
        DenseGraphStep {
            graph: &stage.graph,
            capability: rank.capability,
            catalogue: step.catalogue,
            ctx: rank.ctx,
            stream: rank.stream,
            state: rank.state,
            transaction,
            runs,
            bindings,
        },
        &resident,
        &stage.state_layers,
    );
    let lease = match lease {
        Ok(lease) => lease,
        Err(refused) => {
            held.plans.extend(refused.plan);
            held.leases.extend(refused.held);
            return Err(refused.error);
        }
    };
    match lease.finish() {
        Ok(result) => held.plans.push(result.plan),
        Err(refused) => {
            held.leases.push(refused.lease);
            return Err(refused.error);
        }
    }
    Ok(())
}

/// Copy a boundary value, or its declared column slice, into `local`'s range
/// in `plan`, on the stage's stream.
fn copy_boundary<'ctx>(
    plan: &SelectedReservedPlan<'ctx>,
    local: ValueId,
    source: &DeviceRange<'ctx>,
    slice: Option<LinearInputSlice>,
    rows: u64,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let bytes = plan
        .candidate()
        .value(local)
        .ok_or_else(|| invalid("boundary", "a stage input is absent from its plan"))?
        .logical_bytes;
    let destination = plan.range_for_value(local)?;
    let Some(slice) = slice else {
        // SAFETY: both ranges stay live until the stage's completion is
        // observed, or are withheld with the step.
        return unsafe { destination.copy_from_device_async_at(0, source, 0, bytes, stream) };
    };
    let row = bytes / rows;
    let element = row / slice.width;
    for r in 0..rows {
        // SAFETY: as above; the extents are the planned input's own rows.
        unsafe {
            destination.copy_from_device_async_at(
                r * row,
                source,
                (r * slice.full_width + slice.first) * element,
                row,
                stream,
            )?;
        }
    }
    Ok(())
}

/// Copy a replicated stage's output into the boundary as `original`.
fn keep<'ctx>(
    plan: &SelectedReservedPlan<'ctx>,
    local: ValueId,
    original: ValueId,
    boundary: &mut Boundary<'ctx>,
    stream: &Stream<'ctx>,
) -> Result<()> {
    let bytes = plan
        .candidate()
        .value(local)
        .ok_or_else(|| invalid("boundary", "a stage output is absent from its plan"))?
        .logical_bytes;
    let range = boundary
        .arena
        .allocate(bytes, ALIGNMENT, "stage boundary")
        .map_err(|refused| refused.error)?;
    let range = boundary
        .values
        .entry(original)
        .insert_entry(range)
        .into_mut();
    // SAFETY: the caller synchronizes `stream` before the plan closes, and
    // the boundary holds `range` for the rest of the step.
    unsafe { range.copy_from_device_async_at(0, plan.range_for_value(local)?, 0, bytes, stream) }
}

/// A rank's declaration of its stage output for the next collective.
fn declare(
    sequence: u64,
    plan: &SelectedReservedPlan<'_>,
    local: ValueId,
) -> Result<GatherDeclaration> {
    let planned = plan
        .candidate()
        .value(local)
        .ok_or_else(|| invalid("collective", "a stage output is absent from its plan"))?;
    let (ValueRole::Activation(precision), &[rows, columns]) =
        (planned.role, planned.shape.as_slice())
    else {
        return Err(invalid(
            "collective",
            "a stage output is not a 2D activation",
        ));
    };
    Ok(GatherDeclaration {
        sequence,
        rows,
        columns,
        precision: precision.get(),
    })
}

/// Return a range a quarantined run held to the stage plan whose arena
/// allocated it. One that no held plan owns is withheld, and the group lost.
fn return_to_plans<'ctx>(
    group: &mut RankGroup<'ctx>,
    plans: &mut [SelectedReservedPlan<'ctx>],
    range: DeviceRange<'ctx>,
) -> Result<()> {
    let mut range = range;
    for plan in plans {
        match plan.release_reclaimed(range) {
            Ok(()) => return Ok(()),
            Err(refused) => range = refused.range,
        }
    }
    Err(group.lose(format!(
        "a range reclaimed from a quarantined attention run belongs to no held plan \
         ({} bytes); it is withheld",
        range.bytes()
    )))
}

/// Close every plan. One that cannot close is withheld with the rest of the
/// step, and the group is lost.
fn close_plans(
    group: &mut RankGroup<'_>,
    ledger: &mut Ledger,
    plans: &mut Vec<SelectedReservedPlan<'_>>,
) -> Result<()> {
    while let Some(plan) = plans.pop() {
        if let Err(refused) = plan.close(ledger) {
            return Err(group.lose(format!(
                "a stage plan could not close ({}); it is withheld",
                refused.error
            )));
        }
    }
    Ok(())
}

fn value_bytes(graph: &Graph, value: ValueId, rows: u64, element: u64) -> Result<u64> {
    let spec = graph
        .spec(value)
        .ok_or_else(|| invalid("boundary", "a value has no tensor spec"))?;
    spec.shape.iter().try_fold(element, |bytes, dim| {
        let extent = match dim {
            Dim::Const(extent) => *extent,
            Dim::Symbol(_) => rows,
            _ => return Err(invalid("boundary", "unresolved dimension")),
        };
        bytes
            .checked_mul(extent)
            .ok_or_else(|| invalid("boundary", "boundary extent overflowed"))
    })
}

/// Every graph value at FP32 width, plus the largest again for a reduce's
/// peer partial, which is released after each reduce.
// ponytail: an upper bound over all values; size it from the lowering's live
// boundaries if activations ever dominate the ledger.
fn boundary_bytes(graph: &Graph, rows: u64) -> Result<u64> {
    let mut total = 0u64;
    let mut largest = 0u64;
    for node in graph.nodes() {
        let bytes = value_bytes(graph, node.output, rows, 4)?
            .checked_next_multiple_of(ALIGNMENT)
            .ok_or_else(|| invalid("boundary", "boundary extent overflowed"))?;
        largest = largest.max(bytes);
        total = total
            .checked_add(bytes)
            .ok_or_else(|| invalid("boundary", "boundary extent overflowed"))?;
    }
    total
        .checked_add(largest)
        .ok_or_else(|| invalid("boundary", "boundary extent overflowed"))
}

fn open_boundary<'ctx>(
    group: &mut RankGroup<'ctx>,
    rank: &mut DenseRank<'_, 'ctx>,
    bytes: u64,
) -> Result<Boundary<'ctx>> {
    let scope = Scope::Device(rank.ctx.uuid());
    let mut request = PlanRequest::new("tensor-parallel step boundaries", ["step"])?;
    request.buffer(BufferRequest::try_new(
        "stage boundaries",
        scope,
        Tier::Device(DeviceTier::Activations),
        bytes,
        StageSpan { first: 0, last: 0 },
    )?)?;
    let reservation = rank.ledger.admit(&request)?;
    match DeviceArena::create(
        rank.ledger,
        reservation,
        rank.ctx,
        DeviceTier::Activations,
        bytes,
        "tensor-parallel step boundaries",
    ) {
        Ok(arena) => Ok(Boundary {
            arena,
            values: BTreeMap::new(),
        }),
        Err(refused) => match rank.ledger.release(refused.reservation) {
            Ok(()) => Err(refused.error),
            Err(held) => Err(group.lose(format!(
                "a boundary reservation could not be released ({})",
                held.error
            ))),
        },
    }
}

/// Release every boundary value and close the arena. Anything that cannot
/// be released is withheld, and the group is lost.
fn close_boundary(
    group: &mut RankGroup<'_>,
    ledger: &mut Ledger,
    boundary: Boundary<'_>,
) -> Result<()> {
    let Boundary { mut arena, values } = boundary;
    for (_, range) in values {
        if let Err(refused) = arena.release(range) {
            return Err(group.lose(format!(
                "a boundary value could not be released ({})",
                refused.error
            )));
        }
    }
    arena.close(ledger).map_err(|refused| {
        group.lose(format!(
            "the boundary arena could not close ({})",
            refused.error
        ))
    })
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}
