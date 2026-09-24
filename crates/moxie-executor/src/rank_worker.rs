#![cfg(feature = "paged-attention-binding")]

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use moxie_cuda::{RankContext, Stream};
use moxie_graph::Graph;
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::{Phase, ResourceWorkload, lower_selected};
use moxie_state::{DeviceKvSequence, KvGeometry, PreparedCommit};
use moxie_types::{
    DeviceCapability, Error, KernelCatalogue, PagedKvWriter, RankId, Result, StateTransactionId,
};

use crate::chain::{OwnedBinding, SelectedAdmitRefused, SelectedReservedPlan};
use crate::dense::{DenseGraphStep, DensePlanRunRefused};
use crate::dense_tp::workers::{
    admit_worker_runs, commit_capacity_error, device_lost, park_lost, recv_until,
};
use crate::paged_attention::device::{PagedAttentionRun, PagedKvWriterAdapter};

#[derive(Debug, Clone)]
pub struct SoloRankWorkerConfig {
    pub rank: RankId,
    pub ordinal: u32,
    pub geometry: KvGeometry,
    pub heads: u64,
    pub max_rows: u64,
    pub host_capacity: CapacitySnapshot,
    pub deadline: Duration,
}

#[derive(Debug)]
enum WorkerCommand {
    Step {
        graph: Box<Graph>,
        catalogue: KernelCatalogue,
        bindings: Vec<OwnedBinding>,
        rows: u64,
        visible_tokens: u64,
        reply: Sender<Result<Vec<u8>>>,
    },
    PrepareCommit(Sender<Result<()>>),
    ApplyCommit(Sender<Result<()>>),
    Abort(Sender<Result<()>>),
    Stats(Sender<Result<(u64, u64, usize)>>),
    Shutdown(Sender<Result<()>>),
}

#[derive(Debug)]
pub struct SoloRankWorker {
    commands: Sender<WorkerCommand>,
    thread: Option<JoinHandle<()>>,
    ordinal: u32,
    deadline: Duration,
    lost: Option<Error>,
    closed: bool,
    #[cfg(feature = "paged-attention-test-hooks")]
    fail_next_step: bool,
}

impl SoloRankWorker {
    pub fn spawn(config: SoloRankWorkerConfig) -> Result<Self> {
        let ordinal = config.ordinal;
        let deadline = config.deadline;
        let (commands, receiver) = mpsc::channel();
        let (ready, ready_rx) = mpsc::channel();
        let thread = thread::Builder::new()
            .name(format!("moxie-solo-rank-{ordinal}"))
            .spawn(move || solo_rank_worker(config, receiver, ready))
            .map_err(|error| Error::InvalidRequest {
                field: "rank_worker",
                detail: format!("could not start rank thread: {error}"),
            })?;
        let ready = recv_until(&ready_rx, Instant::now() + deadline, "solo rank startup").map_err(
            |error| match error {
                Error::DeviceLost { detail, .. } => device_lost(ordinal, detail),
                other => other,
            },
        )?;
        ready?;
        Ok(Self {
            commands,
            thread: Some(thread),
            ordinal,
            deadline,
            lost: None,
            closed: false,
            #[cfg(feature = "paged-attention-test-hooks")]
            fail_next_step: false,
        })
    }

    /// Run one graph step inside the open transaction, beginning one if needed.
    pub fn step(
        &mut self,
        graph: Graph,
        catalogue: KernelCatalogue,
        bindings: Vec<OwnedBinding>,
        rows: u64,
        visible_tokens: u64,
    ) -> Result<Vec<u8>> {
        self.check_live()?;
        #[cfg(feature = "paged-attention-test-hooks")]
        if std::mem::take(&mut self.fail_next_step) {
            return Err(Error::InvalidRequest {
                field: "fault",
                detail: "injected step refusal before admission".into(),
            });
        }
        self.request(|reply| WorkerCommand::Step {
            graph: Box::new(graph),
            catalogue,
            bindings,
            rows,
            visible_tokens,
            reply,
        })
    }

    pub fn prepare_commit(&mut self) -> Result<()> {
        self.request(WorkerCommand::PrepareCommit)
    }

    pub fn apply_commit(&mut self) -> Result<()> {
        self.request(WorkerCommand::ApplyCommit)
    }

    /// Roll back the open transaction, dropping a prepared handle first.
    pub fn abort(&mut self) -> Result<()> {
        self.request(WorkerCommand::Abort)
    }

    /// Published rows, committed rows, and outstanding ledger reservations.
    pub fn stats(&mut self) -> Result<(u64, u64, usize)> {
        self.request(WorkerCommand::Stats)
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn fail_next_step(&mut self) {
        self.fail_next_step = true;
    }

    pub fn close(mut self) -> Result<()> {
        if let Some(error) = self.lost.clone() {
            return Err(error);
        }
        let result = self.request(WorkerCommand::Shutdown);
        if let Some(error) = self.lost.clone() {
            return Err(error);
        }
        let thread = self.thread.take();
        self.closed = true;
        match thread.map(|handle| join_until(handle, Instant::now() + self.deadline)) {
            None | Some(JoinOutcome::Joined) => result,
            Some(JoinOutcome::Panicked) => Err(device_lost(
                self.ordinal,
                "solo rank thread panicked during shutdown".into(),
            )),
            Some(JoinOutcome::TimedOut) => Err(device_lost(
                self.ordinal,
                "solo rank thread did not stop before its deadline".into(),
            )),
        }
    }

    fn request<T>(
        &mut self,
        command: impl FnOnce(Sender<Result<T>>) -> WorkerCommand,
    ) -> Result<T> {
        self.check_live()?;
        let (reply, receiver) = mpsc::channel();
        self.commands.send(command(reply)).map_err(|_| {
            let error = device_lost(
                self.ordinal,
                "rank worker stopped accepting commands".into(),
            );
            self.lost = Some(error.clone());
            error
        })?;
        let response = recv_until(
            &receiver,
            Instant::now() + self.deadline,
            "solo rank command",
        );
        let response = match response {
            Ok(response) => response,
            Err(Error::DeviceLost { detail, .. }) => {
                let error = device_lost(self.ordinal, detail);
                self.lost = Some(error.clone());
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        match response {
            Err(error @ Error::DeviceLost { .. }) => {
                self.lost = Some(error.clone());
                Err(error)
            }
            result => result,
        }
    }

    fn check_live(&self) -> Result<()> {
        self.lost
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }
}

impl Drop for SoloRankWorker {
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        if self.lost.is_some() {
            self.thread.take();
            self.closed = true;
            return;
        }
        let deadline = Instant::now() + self.deadline;
        let (reply, receiver) = mpsc::channel();
        let result = if self.commands.send(WorkerCommand::Shutdown(reply)).is_ok() {
            recv_until(&receiver, deadline, "solo rank shutdown").and_then(|result| result)
        } else {
            Err(device_lost(
                self.ordinal,
                "rank worker stopped accepting commands".into(),
            ))
        };
        if let Err(error @ Error::DeviceLost { .. }) = result {
            self.lost = Some(error);
        }
        if self.lost.is_some() {
            self.thread.take();
        } else if let Some(handle) = self.thread.take() {
            let _ = join_until(handle, deadline);
        }
        self.closed = true;
    }
}

enum JoinOutcome {
    Joined,
    Panicked,
    TimedOut,
}

fn join_until(handle: JoinHandle<()>, deadline: Instant) -> JoinOutcome {
    while !handle.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    if handle.is_finished() {
        match handle.join() {
            Ok(()) => JoinOutcome::Joined,
            Err(_) => JoinOutcome::Panicked,
        }
    } else {
        drop(handle);
        JoinOutcome::TimedOut
    }
}

struct WorkerState<'ctx> {
    context: &'ctx RankContext,
    capability: DeviceCapability,
    stream: &'ctx Stream<'ctx>,
    ledger: Ledger,
    state: DeviceKvSequence,
    runs: Vec<PagedAttentionRun<'ctx>>,
    transaction: Option<StateTransactionId>,
    prepared: Option<PreparedCommit>,
}

impl WorkerState<'_> {
    fn step(
        &mut self,
        graph: Graph,
        catalogue: KernelCatalogue,
        bindings: Vec<OwnedBinding>,
        rows: u64,
        visible_tokens: u64,
    ) -> Result<Vec<u8>> {
        if self.prepared.is_some() {
            return Err(invalid(
                "transaction",
                "the open transaction is already prepared",
            ));
        }
        let workload = ResourceWorkload {
            phase: if rows == 1 {
                Phase::Decode
            } else {
                Phase::Prefill
            },
            rows,
            visible_tokens,
            branch_rows: rows,
            output: graph.output(),
            device: self.capability.uuid,
            paged_state_capacity: None,
        };
        let candidate = lower_selected(&graph, workload, &self.capability, &catalogue)?;
        let plan = SelectedReservedPlan::admit(
            candidate,
            &graph,
            &self.capability,
            &catalogue,
            &mut self.ledger,
            self.context,
        )
        .map_err(|refused| match refused {
            SelectedAdmitRefused::Invalid { error, .. } => error,
            SelectedAdmitRefused::Rejected { rejection, .. } => rejection.into(),
            SelectedAdmitRefused::Held {
                resource, error, ..
            } => {
                std::mem::forget(resource);
                device_lost(
                    self.context.ordinal(),
                    format!("plan admission retained a resource ({error})"),
                )
            }
        })?;
        let transaction = match self.transaction {
            Some(transaction) => transaction,
            None => {
                let transaction = match self.state.begin() {
                    Ok(transaction) => transaction,
                    Err(error) => {
                        self.close_plan(plan)?;
                        return Err(error);
                    }
                };
                self.transaction = Some(transaction);
                transaction
            }
        };
        let lease = match plan.execute_dense(DenseGraphStep {
            graph: &graph,
            capability: &self.capability,
            catalogue: &catalogue,
            ctx: self.context,
            stream: self.stream,
            state: &mut self.state,
            transaction,
            runs: &mut self.runs,
            bindings,
            host_experts: &[],
        }) {
            Ok(lease) => lease,
            Err(DensePlanRunRefused {
                plan, error, held, ..
            }) => {
                if let Some(lease) = held {
                    std::mem::forget(lease);
                    return Err(device_lost(
                        self.context.ordinal(),
                        format!("dense launch retained its lease ({error})"),
                    ));
                }
                if let Some(plan) = plan {
                    self.close_plan(plan)?;
                }
                return Err(error);
            }
        };
        let result = lease.finish().map_err(|refused| {
            let error = device_lost(
                self.context.ordinal(),
                format!("dense completion retained its lease ({})", refused.error),
            );
            std::mem::forget(refused.lease);
            error
        })?;
        let crate::dense::DenseGraphResult { plan, output, .. } = result;
        self.close_plan(plan)?;
        Ok(output)
    }

    fn close_plan(&mut self, plan: SelectedReservedPlan<'_>) -> Result<()> {
        plan.close(&mut self.ledger).map_err(|refused| {
            let error = device_lost(
                self.context.ordinal(),
                format!("dense plan could not close ({})", refused.error),
            );
            std::mem::forget(refused.plan);
            error
        })
    }

    fn prepare_commit(&mut self) -> Result<()> {
        let transaction = self
            .transaction
            .ok_or_else(|| invalid("transaction", "there is no open transaction to prepare"))?;
        if self.prepared.is_some() {
            return Err(invalid(
                "prepared_commit",
                "the transaction is already prepared",
            ));
        }
        if self.state.layer_count()? != self.runs.len() {
            return Err(invalid(
                "runs",
                "commit needs one device run per state layer",
            ));
        }
        self.prepared = Some(self.state.prepare_commit(transaction, 0)?);
        Ok(())
    }

    fn apply_commit(&mut self) -> Result<()> {
        if self.transaction.is_none() || self.prepared.is_none() {
            return Err(invalid(
                "prepared_commit",
                "apply needs an open, prepared transaction",
            ));
        }
        let mut adapters = Vec::<PagedKvWriterAdapter<'_, '_>>::new();
        adapters
            .try_reserve_exact(self.runs.len())
            .map_err(|_| commit_capacity_error(self.runs.len()))?;
        for (layer, run) in self.runs.iter_mut().enumerate() {
            adapters.push(PagedKvWriterAdapter::new(
                layer,
                run,
                self.stream,
                Vec::new(),
                Vec::new(),
            ));
        }
        let mut writers = Vec::<&mut dyn PagedKvWriter>::new();
        writers
            .try_reserve_exact(adapters.len())
            .map_err(|_| commit_capacity_error(adapters.len()))?;
        writers.extend(
            adapters
                .iter_mut()
                .map(|adapter| adapter as &mut dyn PagedKvWriter),
        );
        let prepared = self.prepared.take().expect("checked prepared commit");
        self.state
            .apply_commit(prepared, &mut writers)
            .map_err(|error| {
                device_lost(
                    self.context.ordinal(),
                    format!("prepared commit publication failed ({error}); state may diverge"),
                )
            })?;
        self.transaction = None;
        Ok(())
    }

    fn abort(&mut self) -> Result<()> {
        self.prepared.take();
        if let Some(transaction) = self.transaction {
            self.state.abort(transaction)?;
            self.transaction = None;
        }
        Ok(())
    }

    fn stats(&self) -> Result<(u64, u64, usize)> {
        Ok((
            self.state.published_rows()?,
            self.state.committed_rows()?,
            self.ledger.outstanding().len(),
        ))
    }

    fn close(&mut self) -> Result<()> {
        self.abort().map_err(|error| {
            device_lost(
                self.context.ordinal(),
                format!("open transaction could not abort during close ({error})"),
            )
        })?;
        while let Some(run) = self.runs.pop() {
            if let Err(refused) = run.close(&mut self.ledger) {
                let error = device_lost(
                    self.context.ordinal(),
                    format!("attention run could not close ({})", refused.error),
                );
                std::mem::forget(refused.run);
                return Err(error);
            }
        }
        if !self.ledger.outstanding().is_empty() {
            return Err(device_lost(
                self.context.ordinal(),
                "worker shutdown left charged reservations".into(),
            ));
        }
        Ok(())
    }
}

fn solo_rank_worker(
    config: SoloRankWorkerConfig,
    commands: Receiver<WorkerCommand>,
    ready: Sender<Result<()>>,
) {
    let context = match RankContext::acquire(config.rank, config.ordinal) {
        Ok(context) => context,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let initialized: Result<_> = (|| {
        let stream = Stream::new(&context)?;
        let capability = context.capability().clone();
        let device = CapacitySnapshot::measured(&context.measure()?, 1 << 20)?;
        let ledger = Ledger::new([device, config.host_capacity.clone()])?;
        let state = DeviceKvSequence::new(config.geometry.clone())?;
        let (ledger, runs) = admit_worker_runs(
            &context,
            ledger,
            &state,
            &config.geometry,
            config.heads,
            config.max_rows,
        )?;
        Ok((stream, capability, state, ledger, runs))
    })();
    let (stream, capability, state, ledger, runs) = match initialized {
        Ok(initialized) => initialized,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let mut worker = WorkerState {
        context: &context,
        capability,
        stream: &stream,
        ledger,
        state,
        runs,
        transaction: None,
        prepared: None,
    };
    if ready.send(Ok(())).is_err() {
        park_lost();
    }
    loop {
        let command = match commands.recv() {
            Ok(command) => command,
            Err(_) => {
                if worker.close().is_err() {
                    park_lost();
                }
                return;
            }
        };
        let lost = match command {
            WorkerCommand::Step {
                graph,
                catalogue,
                bindings,
                rows,
                visible_tokens,
                reply,
            } => send_reply(
                reply,
                worker.step(*graph, catalogue, bindings, rows, visible_tokens),
            ),
            WorkerCommand::PrepareCommit(reply) => send_reply(reply, worker.prepare_commit()),
            WorkerCommand::ApplyCommit(reply) => send_reply(reply, worker.apply_commit()),
            WorkerCommand::Abort(reply) => send_reply(reply, worker.abort()),
            WorkerCommand::Stats(reply) => send_reply(reply, worker.stats()),
            WorkerCommand::Shutdown(reply) => {
                if send_reply(reply, worker.close()) {
                    park_lost();
                }
                return;
            }
        };
        if lost {
            park_lost();
        }
    }
}

fn send_reply<T>(reply: Sender<Result<T>>, result: Result<T>) -> bool {
    let lost = matches!(&result, Err(Error::DeviceLost { .. }));
    let _ = reply.send(result);
    lost
}

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}
