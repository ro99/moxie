#![cfg(feature = "paged-attention-binding")]

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use moxie_graph::{Graph, OracleRegistry, ValueId, ValueRole};
use moxie_plan::{PipelineLowering, StageGraph, build_stage_graph, lower_pipeline, wavefront};
use moxie_types::{Error, KernelCatalogue, Result, SymbolTable, TensorLayout};

use crate::{OwnedBinding, SoloRankWorker};

#[derive(Debug)]
pub struct PipelineWorkers {
    workers: Vec<SoloRankWorker>,
    lost: Option<Error>,
}

#[derive(Debug)]
pub struct PipelineStep<'w> {
    workers: &'w mut PipelineWorkers,
    logits: Vec<u8>,
    handoff_bytes: Vec<u64>,
    open: bool,
}

impl PipelineWorkers {
    /// Stage `i` runs on `workers[i]`.
    pub fn new(workers: Vec<SoloRankWorker>) -> Result<Self> {
        if workers.len() < 2 {
            return Err(invalid(
                "pipeline",
                "at least two stage workers are required",
            ));
        }
        Ok(Self {
            workers,
            lost: None,
        })
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn fail_next_step(&mut self, stage: usize) -> Result<()> {
        self.workers
            .get_mut(stage)
            .ok_or_else(|| invalid("stage", "pipeline stage index is out of range"))?
            .fail_next_step();
        Ok(())
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn refuse_next_prepare(&mut self, stage: usize) -> Result<()> {
        self.workers
            .get_mut(stage)
            .ok_or_else(|| invalid("stage", "pipeline stage index is out of range"))?
            .refuse_next_prepare();
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
        bindings: &mut dyn FnMut(usize, &StageGraph, Range<u64>) -> Result<Vec<OwnedBinding>>,
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
        let stage_count = self.workers.len();
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

        let mut intermediate: Vec<Option<Vec<u8>>> =
            (0..microbatches.len()).map(|_| None).collect();
        let mut logits = Vec::new();
        let mut handoff_bytes = vec![0u64; stage_count - 1];

        // ponytail: sequential wavefront keeps tentative KV causal. Overlap needs a split send/receive on SoloRankWorker.
        for (stage, microbatch) in wavefront(stage_count, microbatches.len()) {
            if cancel.load(Ordering::SeqCst) {
                let error = Error::Cancelled {
                    at: "pipeline stage",
                };
                return Err(self.abort_after(error));
            }
            let rows = microbatches[microbatch].clone();
            let count = rows.end - rows.start;
            let result = (|| {
                let mut stage_bindings = bindings(stage, &stage_graphs[stage], rows.clone())?;
                if stage > 0 {
                    let handoff = lowering.handoffs()[stage - 1];
                    let read = stage_graphs[stage]
                        .reads
                        .iter()
                        .find(|read| read.original == handoff)
                        .ok_or_else(|| {
                            invalid("handoff", "stage graph has no declared handoff read")
                        })?;
                    if stage_bindings
                        .iter()
                        .any(|binding| binding.value == read.local)
                    {
                        return Err(invalid(
                            "handoff",
                            "caller supplied the pipeline-owned handoff binding",
                        ));
                    }
                    let bytes = intermediate[microbatch].take().ok_or_else(|| {
                        invalid(
                            "handoff",
                            "preceding stage has no output for this microbatch",
                        )
                    })?;
                    let (role, shape, expected_bytes) =
                        value_extent(&stage_graphs[stage].graph, read.local, count)?;
                    if u64::try_from(bytes.len()).ok() != Some(expected_bytes) {
                        return Err(invalid(
                            "handoff",
                            "preceding stage output does not match the declared read extent",
                        ));
                    }
                    let device = stage_bindings
                        .first()
                        .map(|binding| binding.device)
                        .ok_or_else(|| {
                            invalid("handoff", "caller bindings do not identify a device")
                        })?;
                    stage_bindings.push(OwnedBinding {
                        value: read.local,
                        role,
                        shape,
                        layout: TensorLayout::ContiguousRowMajorV1,
                        device,
                        bytes,
                    });
                    handoff_bytes[stage - 1] = handoff_bytes[stage - 1]
                        .checked_add(expected_bytes)
                        .ok_or_else(|| invalid("handoff", "cumulative byte count overflowed"))?;
                }

                let output = self.workers[stage].step(
                    stage_graphs[stage].graph.clone(),
                    catalogue.clone(),
                    stage_bindings,
                    count,
                    rows.end,
                )?;
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
                Ok(output)
            })();

            let output = match result {
                Ok(output) => output,
                Err(error) => return Err(self.abort_after(error)),
            };
            if stage + 1 == stage_count {
                logits.extend(output);
            } else {
                intermediate[microbatch] = Some(output);
            }
        }

        if cancel.load(Ordering::SeqCst) {
            return Err(self.abort_after(Error::Cancelled {
                at: "pipeline stage",
            }));
        }

        Ok(PipelineStep {
            workers: self,
            logits,
            handoff_bytes,
            open: true,
        })
    }

    /// One entry per stage: published rows, committed rows, reservations.
    pub fn stats(&mut self) -> Result<Vec<(u64, u64, usize)>> {
        if let Some(error) = &self.lost {
            return Err(error.clone());
        }
        let mut stats = Vec::with_capacity(self.workers.len());
        let mut first_error = None;
        for worker in &mut self.workers {
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

    pub fn close(mut self) -> Result<()> {
        let mut first_error = self.lost.take();
        for worker in self.workers.drain(..) {
            if let Err(error) = worker.close()
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn abort_after(&mut self, error: Error) -> Error {
        if matches!(error, Error::DeviceLost { .. }) && self.lost.is_none() {
            self.lost = Some(error.clone());
        }
        let _ = self.abort_all();
        self.lost.clone().unwrap_or(error)
    }

    fn abort_all(&mut self) -> Option<Error> {
        let mut first_error = None;
        for worker in &mut self.workers {
            if let Err(error) = worker.abort() {
                if matches!(error, Error::DeviceLost { .. }) && self.lost.is_none() {
                    self.lost = Some(error.clone());
                }
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        first_error
    }
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
        for stage in 0..self.workers.workers.len() {
            if let Err(error) = self.workers.workers[stage].prepare_commit() {
                let error = self.workers.abort_after(error);
                self.open = false;
                return Err(error);
            }
        }
        let mut applied = false;
        for stage in 0..self.workers.workers.len() {
            if let Err(error) = self.workers.workers[stage].apply_commit() {
                let error = if applied {
                    Error::DeviceLost {
                        device: stage as u32,
                        detail: format!(
                            "pipeline stage {stage} failed after an earlier stage committed ({error})"
                        ),
                    }
                } else {
                    error
                };
                let error = self.workers.abort_after(error);
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
            let _ = self.workers.abort_all();
            self.open = false;
        }
    }
}

fn value_extent(graph: &Graph, value: ValueId, rows: u64) -> Result<(ValueRole, Vec<u64>, u64)> {
    let spec = graph
        .spec(value)
        .ok_or_else(|| invalid("pipeline value", "value has no declared tensor spec"))?;
    let ValueRole::Activation(precision) = spec.role else {
        return Err(invalid(
            "pipeline value",
            "pipeline output is not a declared activation",
        ));
    };
    let mut symbols = SymbolTable::new();
    symbols.bind(graph.rows_symbol(), rows);
    let shape = spec
        .shape
        .iter()
        .map(|dim| {
            dim.eval(&symbols)
                .map_err(|_| invalid("pipeline value", "unresolved dimension"))
        })
        .collect::<Result<Vec<_>>>()?;
    let elements = shape.iter().try_fold(1u64, |total, dim| {
        total
            .checked_mul(*dim)
            .ok_or_else(|| invalid("pipeline value", "declared extent overflowed"))
    })?;
    let bytes_per_element = u64::from(precision.get().bits() / 8);
    let bytes = elements
        .checked_mul(bytes_per_element)
        .ok_or_else(|| invalid("pipeline value", "declared byte extent overflowed"))?;
    Ok((spec.role, shape, bytes))
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}
