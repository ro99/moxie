#![cfg(feature = "paged-attention-binding")]

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use moxie_graph::{Graph, OracleRegistry, ValueId, ValueRole};
use moxie_plan::{
    PipelineLowering, StageGraph, build_stage_graph, lower_pipeline, lower_tensor_parallel,
    value_bytes as plan_value_bytes, wavefront,
};
use moxie_types::{Error, KernelCatalogue, Result, SymbolTable, TensorLayout};

use crate::{DenseRankWorkers, DenseWorkerStep, OwnedBinding, SoloRankWorker};

#[derive(Debug)]
pub enum PipelineStageWorker {
    Solo(SoloRankWorker),
    Pair(DenseRankWorkers),
}

#[derive(Debug)]
pub struct PipelineWorkers {
    pair: Option<DenseRankWorkers>,
    solos: Vec<SoloRankWorker>,
    lost: Option<Error>,
}

#[derive(Debug)]
pub struct StageBindings<'a> {
    pub stage: usize,
    pub graph: &'a StageGraph,
    pub rank: Option<(usize, &'a StageGraph)>,
    pub rows: Range<u64>,
}

#[derive(Debug)]
pub struct PipelineStep<'w> {
    pair_step: Option<DenseWorkerStep<'w>>,
    solos: &'w mut [SoloRankWorker],
    lost: &'w mut Option<Error>,
    logits: Vec<u8>,
    handoff_bytes: Vec<u64>,
    open: bool,
}

impl PipelineWorkers {
    /// Stage `i` runs on the corresponding worker; a pair is allowed only at stage 0.
    pub fn new(stages: Vec<PipelineStageWorker>) -> Result<Self> {
        if stages.len() < 2 {
            return Err(invalid(
                "pipeline",
                "at least two stage workers are required",
            ));
        }
        if stages
            .iter()
            .enumerate()
            .any(|(index, stage)| index != 0 && matches!(stage, PipelineStageWorker::Pair(_)))
        {
            return Err(invalid("pipeline", "a pair is supported only at stage 0"));
        }
        let mut stages = stages.into_iter();
        let (pair, first_solo) = match stages.next().expect("two or more stages") {
            PipelineStageWorker::Pair(pair) => (Some(pair), None),
            PipelineStageWorker::Solo(solo) => (None, Some(solo)),
        };
        let mut solos = Vec::with_capacity(stages.len() + usize::from(first_solo.is_some()));
        if let Some(solo) = first_solo {
            solos.push(solo);
        }
        solos.extend(stages.map(|stage| match stage {
            PipelineStageWorker::Solo(solo) => solo,
            PipelineStageWorker::Pair(_) => unreachable!("pair position was validated"),
        }));
        Ok(Self {
            pair,
            solos,
            lost: None,
        })
    }

    fn solo_index(&self, stage: usize) -> Result<usize> {
        stage
            .checked_sub(usize::from(self.pair.is_some()))
            .filter(|index| *index < self.solos.len())
            .ok_or_else(|| invalid("stage", "pipeline stage has no solo worker"))
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn fail_next_step(&mut self, stage: usize) -> Result<()> {
        let index = self.solo_index(stage)?;
        self.solos[index].fail_next_step();
        Ok(())
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn refuse_next_prepare(&mut self, stage: usize) -> Result<()> {
        let index = self.solo_index(stage)?;
        self.solos[index].refuse_next_prepare();
        Ok(())
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn refuse_next_pair_commit(&mut self, rank: usize) -> Result<()> {
        self.pair
            .as_mut()
            .ok_or_else(|| invalid("pipeline", "pipeline has no pair stage"))?
            .refuse_next_commit_prepare(rank);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn execute(
        &mut self,
        graph: &Graph,
        lowering: &PipelineLowering,
        oracle: moxie_graph::OracleId,
        oracles: &OracleRegistry,
        catalogue: &KernelCatalogue,
        microbatches: &[Range<u64>],
        bindings: &mut dyn FnMut(StageBindings<'_>) -> Result<Vec<OwnedBinding>>,
        cancel: &AtomicBool,
    ) -> Result<PipelineStep<'_>> {
        if let Some(error) = &self.lost {
            return Err(error.clone());
        }
        let cuts: Vec<_> = lowering
            .stages()
            .iter()
            .skip(1)
            .map(|stage| stage.start)
            .collect();
        if !lower_pipeline(graph, &cuts).is_ok_and(|derived| derived == *lowering) {
            return Err(invalid(
                "pipeline",
                "lowering does not match the source graph",
            ));
        }
        let stage_count = self.solos.len() + usize::from(self.pair.is_some());
        if lowering.stages().len() != stage_count || lowering.handoffs().len() + 1 != stage_count {
            return Err(invalid(
                "pipeline",
                "lowered stage and worker counts do not match",
            ));
        }
        if microbatches.is_empty()
            || microbatches.iter().any(|rows| rows.start >= rows.end)
            || microbatches
                .windows(2)
                .any(|pair| pair[0].end != pair[1].start || pair[0].start >= pair[1].start)
        {
            return Err(invalid(
                "microbatches",
                "ranges must be nonempty, contiguous, and strictly increasing",
            ));
        }
        if self.pair.is_some() && microbatches.len() != 1 {
            return Err(invalid(
                "microbatches",
                "a pair stage requires exactly one microbatch",
            ));
        }

        let mut stage_graphs = Vec::with_capacity(stage_count);
        for (stage, nodes) in lowering.stages().iter().enumerate() {
            let output = lowering
                .handoffs()
                .get(stage)
                .copied()
                .unwrap_or_else(|| graph.output());
            stage_graphs.push(build_stage_graph(
                graph,
                None,
                nodes.clone(),
                Some(output),
                oracle,
                oracles,
            )?);
        }
        let tp_lowering = if self.pair.is_some() {
            Some(
                lower_tensor_parallel(&stage_graphs[0].graph, 2)
                    .map_err(|_| invalid("pipeline", "stage 0 refuses TP2 lowering"))?,
            )
        } else {
            None
        };

        let mut intermediate: Vec<Option<Vec<u8>>> =
            (0..microbatches.len()).map(|_| None).collect();
        let mut logits = Vec::new();
        let mut handoff_bytes = vec![0u64; stage_count - 1];
        let PipelineWorkers { pair, solos, lost } = self;
        let has_pair = pair.is_some();
        let mut pair_step = None;

        if let (Some(pair), Some(tp)) = (pair.as_mut(), tp_lowering.as_ref()) {
            if cancel.load(Ordering::SeqCst) {
                return Err(abort_pipeline(
                    &mut pair_step,
                    solos,
                    lost,
                    Error::Cancelled {
                        at: "pipeline stage",
                    },
                ));
            }
            let rows = microbatches[0].clone();
            let count = rows.end - rows.start;
            let mut pair_bindings = |rank, sub_stage: &StageGraph| {
                bindings(StageBindings {
                    stage: 0,
                    graph: &stage_graphs[0],
                    rank: Some((rank, sub_stage)),
                    rows: rows.clone(),
                })
            };
            let step = match pair.execute_dense(
                &stage_graphs[0].graph,
                tp,
                oracle,
                oracles,
                catalogue,
                count,
                rows.end,
                &mut pair_bindings,
                cancel,
            ) {
                Ok(step) => step,
                Err(error) => return Err(abort_pipeline(&mut pair_step, solos, lost, error)),
            };
            intermediate[0] = Some(step.logits().to_vec());
            pair_step = Some(step);
        }

        // ponytail: sequential wavefront keeps tentative KV causal. Overlap needs a split send/receive on SoloRankWorker.
        let execution = (|| {
            for (stage, microbatch) in wavefront(stage_count, microbatches.len()) {
                if cancel.load(Ordering::SeqCst) {
                    return Err(Error::Cancelled {
                        at: "pipeline stage",
                    });
                }
                let rows = microbatches[microbatch].clone();
                let count = rows.end - rows.start;
                let output = if has_pair && stage == 0 {
                    intermediate[microbatch]
                        .take()
                        .ok_or_else(|| invalid("handoff", "pair stage has no output"))?
                } else {
                    solo_stage(
                        solos,
                        has_pair,
                        stage,
                        &stage_graphs[stage],
                        &rows,
                        microbatch,
                        &mut intermediate,
                        lowering,
                        bindings,
                        count,
                        catalogue,
                    )?
                };
                let (_, _, expected_bytes) = value_extent(
                    &stage_graphs[stage].graph,
                    stage_graphs[stage].graph.output(),
                    count,
                )?;
                if u64::try_from(output.len()).ok() != Some(expected_bytes) {
                    return Err(invalid(
                        "pipeline output",
                        "stage output does not match its declared role and shape",
                    ));
                }
                if stage + 1 == stage_count {
                    logits.extend(output);
                    continue;
                }
                let next = &stage_graphs[stage + 1];
                let handoff = lowering.handoffs()[stage];
                let read = next
                    .reads
                    .iter()
                    .find(|read| read.original == handoff)
                    .ok_or_else(|| {
                        invalid("handoff", "stage graph has no declared handoff read")
                    })?;
                let (_, _, expected) = value_extent(&next.graph, read.local, count)?;
                if expected != expected_bytes {
                    return Err(invalid("handoff", "adjacent stage extents do not match"));
                }
                handoff_bytes[stage] = handoff_bytes[stage]
                    .checked_add(expected_bytes)
                    .ok_or_else(|| invalid("handoff", "cumulative byte count overflowed"))?;
                intermediate[microbatch] = Some(output);
            }
            if cancel.load(Ordering::SeqCst) {
                return Err(Error::Cancelled {
                    at: "pipeline stage",
                });
            }
            Ok(())
        })();
        if let Err(error) = execution {
            return Err(abort_pipeline(&mut pair_step, solos, lost, error));
        }

        Ok(PipelineStep {
            pair_step,
            solos,
            lost,
            logits,
            handoff_bytes,
            open: true,
        })
    }

    /// One entry per solo stage: published rows, committed rows, reservations.
    pub fn stats(&mut self) -> Result<Vec<(u64, u64, usize)>> {
        if let Some(error) = &self.lost {
            return Err(error.clone());
        }
        let mut stats = Vec::with_capacity(self.solos.len());
        let mut first_error = None;
        for worker in &mut self.solos {
            match worker.stats() {
                Ok(value) => stats.push(value),
                Err(error) => {
                    if matches!(error, Error::DeviceLost { .. }) && self.lost.is_none() {
                        self.lost = Some(error.clone());
                    }
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        first_error.map_or(Ok(stats), Err)
    }

    pub fn pair_frontiers(&self) -> Option<([u64; 2], [u64; 2])> {
        self.pair
            .as_ref()
            .map(|pair| (pair.published_frontiers(), pair.committed_frontiers()))
    }

    pub fn close(mut self) -> Result<()> {
        let mut first_error = self.lost.take();
        if let Some(pair) = self.pair.take()
            && let Err(error) = pair.close()
            && first_error.is_none()
        {
            first_error = Some(error);
        }
        for worker in self.solos.drain(..) {
            if let Err(error) = worker.close()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

#[allow(clippy::too_many_arguments)]
fn solo_stage(
    solos: &mut [SoloRankWorker],
    has_pair: bool,
    stage: usize,
    stage_graph: &StageGraph,
    rows: &Range<u64>,
    microbatch: usize,
    intermediate: &mut [Option<Vec<u8>>],
    lowering: &PipelineLowering,
    bindings: &mut dyn FnMut(StageBindings<'_>) -> Result<Vec<OwnedBinding>>,
    count: u64,
    catalogue: &KernelCatalogue,
) -> Result<Vec<u8>> {
    let mut stage_bindings = bindings(StageBindings {
        stage,
        graph: stage_graph,
        rank: None,
        rows: rows.clone(),
    })?;
    if stage > 0 {
        let handoff = lowering.handoffs()[stage - 1];
        let read = stage_graph
            .reads
            .iter()
            .find(|read| read.original == handoff)
            .ok_or_else(|| invalid("handoff", "stage graph has no declared handoff read"))?;
        if stage_bindings
            .iter()
            .any(|binding| binding.value == read.local)
        {
            return Err(invalid(
                "handoff",
                "caller supplied the pipeline-owned handoff binding",
            ));
        }
        let bytes = intermediate[microbatch]
            .take()
            .ok_or_else(|| invalid("handoff", "preceding stage has no output"))?;
        let (role, shape, expected_bytes) = value_extent(&stage_graph.graph, read.local, count)?;
        if u64::try_from(bytes.len()).ok() != Some(expected_bytes) {
            return Err(invalid(
                "handoff",
                "preceding stage output does not match the declared read extent",
            ));
        }
        let device = stage_bindings
            .first()
            .map(|binding| binding.device)
            .ok_or_else(|| invalid("handoff", "caller bindings do not identify a device"))?;
        stage_bindings.push(OwnedBinding {
            value: read.local,
            role,
            shape,
            layout: TensorLayout::ContiguousRowMajorV1,
            device,
            bytes,
        });
    }
    let solo_index = stage
        .checked_sub(usize::from(has_pair))
        .ok_or_else(|| invalid("stage", "pipeline stage has no solo worker"))?;
    solos[solo_index].step(
        stage_graph.graph.clone(),
        catalogue.clone(),
        stage_bindings,
        count,
        rows.end,
    )
}

fn abort_pipeline(
    pair_step: &mut Option<DenseWorkerStep<'_>>,
    solos: &mut [SoloRankWorker],
    lost: &mut Option<Error>,
    error: Error,
) -> Error {
    drop(pair_step.take());
    abort_solos(solos, lost, error)
}

fn abort_solos(solos: &mut [SoloRankWorker], lost: &mut Option<Error>, error: Error) -> Error {
    if matches!(error, Error::DeviceLost { .. }) && lost.is_none() {
        *lost = Some(error.clone());
    }
    for solo in solos {
        if let Err(abort_error) = solo.abort()
            && matches!(abort_error, Error::DeviceLost { .. })
            && lost.is_none()
        {
            *lost = Some(abort_error);
        }
    }
    lost.clone().unwrap_or(error)
}

impl PipelineStep<'_> {
    pub fn logits(&self) -> &[u8] {
        &self.logits
    }

    /// Host-staged byte count for each boundary, summed over microbatches.
    pub fn handoff_bytes(&self) -> &[u64] {
        &self.handoff_bytes
    }

    pub fn commit(mut self) -> Result<Vec<u8>> {
        for stage in 0..self.solos.len() {
            if let Err(error) = self.solos[stage].prepare_commit() {
                let error = abort_pipeline(&mut self.pair_step, self.solos, self.lost, error);
                self.open = false;
                return Err(error);
            }
        }
        let pair_committed = if let Some(step) = self.pair_step.take() {
            if let Err(error) = step.commit() {
                let error = abort_solos(self.solos, self.lost, error);
                self.open = false;
                return Err(error);
            }
            true
        } else {
            false
        };
        let mut applied = false;
        for (index, solo) in self.solos.iter_mut().enumerate() {
            if let Err(error) = solo.apply_commit() {
                let stage = index + usize::from(pair_committed);
                let error = if pair_committed || applied {
                    Error::DeviceLost {
                        device: stage as u32,
                        detail: format!(
                            "pipeline stage {stage} failed after an earlier stage committed ({error})"
                        ),
                    }
                } else {
                    error
                };
                let error = abort_solos(self.solos, self.lost, error);
                self.open = false;
                return Err(error);
            }
            applied = true;
        }
        self.open = false;
        Ok(std::mem::take(&mut self.logits))
    }
}

impl Drop for PipelineStep<'_> {
    fn drop(&mut self) {
        if self.open {
            let error = abort_pipeline(
                &mut self.pair_step,
                self.solos,
                self.lost,
                Error::Cancelled {
                    at: "dropped pipeline step",
                },
            );
            let _ = error;
            self.open = false;
        }
    }
}

fn value_extent(graph: &Graph, value: ValueId, rows: u64) -> Result<(ValueRole, Vec<u64>, u64)> {
    let spec = graph
        .spec(value)
        .ok_or_else(|| invalid("pipeline value", "value has no declared tensor spec"))?;
    let ValueRole::Activation(_) = spec.role else {
        return Err(invalid(
            "pipeline value",
            "pipeline output is not a declared activation",
        ));
    };
    let mut symbols = SymbolTable::new();
    symbols.bind(graph.rows_symbol(), rows);
    let shape = spec
        .extent(&symbols)
        .map_err(|_| invalid("pipeline value", "unresolved dimension"))?;
    let bytes = plan_value_bytes(graph, value, rows)?;
    Ok((spec.role, shape, bytes))
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}
