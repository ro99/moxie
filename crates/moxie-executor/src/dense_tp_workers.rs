//! Persistent, rank-owned workers for dense TP execution (task 0062).

use core::ffi::c_void;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use moxie_cuda::{Event, Module, ModuleImage, PeerContextToken, RankContext, Stream, TrustedImage};
use moxie_graph::{Graph, LinearInputSlice, OracleRegistry, ValueId, ValueRole};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_plan::{
    Join, Stage, StageGraph, TensorParallelLowering, build_stage_graph,
    value_bytes as plan_value_bytes,
};
use moxie_state::{DeviceKvSequence, KvGeometry, PreparedCommit};
use moxie_types::{
    DeviceCapability, DeviceTier, Error, KernelCatalogue, PagedKvWriter, Precision, RankId, Result,
    Scope, StateTransactionId, SymbolTable, Tier,
};

use moxie_cuda::nccl::{CommState, Communicator, DataType as NcclDataType, NcclId};

use crate::arena::{DeviceArena, DeviceRange, OperationLease};
use crate::chain::{OwnedBinding, SelectedAdmitRefused, SelectedCompletion, SelectedReservedPlan};
use crate::dense::{DenseGraphStep, DenseOperation};
use crate::paged_attention::device::{PagedAttentionRun, Staging, with_paged_writers};
#[cfg(feature = "paged-attention-binding")]
use crate::tensor_parallel::{RankDeclaration, RankRendezvous};

const ALIGNMENT: u64 = 256;
// Per-rank delta: 146,800,640 bytes for NCCL init plus a 1 MiB all-reduce on
// both 3090s (25,013,125,120 → 24,866,324,480 free bytes). The largest
// admitted Gather was 5×128 BF16 values (2,560-byte collective range); after
// the first-join 27,262,976-byte lazy allocation, it added 0 bytes. The 1 MiB
// probe already includes that lazy allocation. 146,800,640 rounded up to the
// next 16 MiB is 150,994,944 bytes.
const NCCL_DEVICE_RESERVE_BYTES: u64 = 144 * 1024 * 1024;
const JOIN_STATUS_BYTES: u64 = 4;

/// Per-rank boundary values and their admitted device arena.
#[derive(Debug)]
struct Boundary<'ctx> {
    arena: DeviceArena<'ctx>,
    values: BTreeMap<ValueId, DeviceRange<'ctx>>,
}

/// Rank-local resources retained until streams drain and cleanup completes.
#[derive(Debug, Default)]
struct Held<'ctx> {
    plans: Vec<SelectedReservedPlan<'ctx>>,
    leases: Vec<OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>>,
    boundary: Option<Boundary<'ctx>>,
}

const OP_BEGIN: u16 = 1;
const OP_STAGE: u16 = 2;
const OP_DRAIN: u16 = 3;
const OP_CLOSE_PLANS: u16 = 4;
const OP_JOIN_PREPARE: u16 = 5;
const OP_JOIN_COPY: u16 = 6;
const OP_CLEANUP: u16 = 7;
const OP_ABORT: u16 = 8;
const OP_FINALIZE: u16 = 9;
const OP_COMMIT_PREPARE: u16 = 10;
const OP_COMMIT_APPLY: u16 = 11;
const OP_SHUTDOWN: u16 = 12;
const OP_STATS: u16 = 13;
const OP_JOIN_DRAIN: u16 = 14;
const OP_CLOSE_PREPARE: u16 = 15;
const OP_CLOSE_DESTROY: u16 = 16;
const OP_CLOSE_RELEASE: u16 = 17;

/// Inputs used to initialize both workers. Device capacity is measured after
/// each worker has acquired its own context; the shared host snapshot is only
/// copied as a numeric budget.
#[derive(Debug, Clone)]
pub struct DenseRankWorkerConfig {
    pub ranks: [RankId; 2],
    pub ordinals: [u32; 2],
    pub geometry: KvGeometry,
    pub heads: u64,
    pub max_rows: u64,
    pub host_capacity: CapacitySnapshot,
    pub deadline: Duration,
}

#[derive(Debug)]
struct WorkerEnvelope {
    command: WorkerCommand,
    declaration: RankDeclaration,
    stall: bool,
    deadline: Instant,
    reply: Sender<WorkerReply>,
}

#[derive(Debug)]
enum WorkerCommand {
    Begin {
        boundary_bytes: u64,
    },
    Stage {
        stage: Box<StageGraph>,
        catalogue: KernelCatalogue,
        bindings: Result<Vec<OwnedBinding>>,
        rows: u64,
        visible_tokens: u64,
        transaction: StateTransactionId,
        keep: Option<ValueId>,
    },
    Drain,
    ClosePlans,
    JoinPrepare {
        join: Join,
        declaration: crate::GatherDeclaration,
        source: ValueId,
    },
    JoinCopy {
        join: Join,
        declaration: crate::GatherDeclaration,
        fail_after_prepare: bool,
    },
    JoinDrain {
        join: Join,
        declaration: crate::GatherDeclaration,
    },
    Cleanup,
    Abort(Option<StateTransactionId>),
    ReadOutput {
        value: ValueId,
        bytes: u64,
    },
    Commit {
        transaction: StateTransactionId,
        sequence: u64,
        fail_prepare: bool,
    },
    Stats,
    ClosePrepare,
    CloseDestroy,
    CloseRelease,
    Shutdown,
}

enum StartupIdChannel {
    Generate(Sender<Result<NcclId>>),
    Receive(Receiver<Result<NcclId>>),
}

#[derive(Debug)]
enum WorkerValue {
    Unit,
    Transaction(StateTransactionId),
    Bytes(Vec<u8>),
    Stage(Option<crate::GatherDeclaration>),
    Stats {
        published: u64,
        committed: u64,
        reservations: usize,
    },
    Closed,
}

#[derive(Debug)]
struct WorkerReply {
    result: Result<WorkerValue>,
    rounds: u8,
}

#[derive(Debug)]
struct StartupContext {
    rank: usize,
    generation: u64,
    token: PeerContextToken,
}

#[derive(Debug)]
struct StartupReport {
    rank: usize,
    result: Result<()>,
}

#[derive(Debug)]
struct StartupDisabled {
    rank: usize,
    result: Result<()>,
}

#[derive(Debug, Clone, Copy)]
enum StartupOutcome {
    Proceed,
    Abort,
}

#[derive(Debug, Clone, Copy)]
enum StartupExit {
    Exit,
}

#[derive(Debug)]
struct CollectiveTemp<'ctx> {
    source: ValueId,
    arena: Option<DeviceArena<'ctx>>,
    status: Option<DeviceRange<'ctx>>,
    data: Option<DeviceRange<'ctx>>,
    dummy: Option<DeviceRange<'ctx>>,
    joined_output: Option<DeviceRange<'ctx>>,
    local_source: Option<crate::DeviceRange<'ctx>>,
    module: Option<moxie_cuda::ResolvedModule<'ctx>>,
}

#[derive(Debug)]
struct WorkerState<'ctx> {
    ctx: &'ctx RankContext,
    capability: DeviceCapability,
    stream: &'ctx Stream<'ctx>,
    ledger: Ledger,
    state: DeviceKvSequence,
    runs: Vec<PagedAttentionRun<'ctx>>,
    held: Held<'ctx>,
    transaction: Option<StateTransactionId>,
    temp: Option<CollectiveTemp<'ctx>>,
    communicator: Option<Communicator<'ctx>>,
    nccl_reservation: Option<Reservation>,
    preserve_communicator_on_loss: bool,
}

/// Owns the rank threads and their command channels. CUDA values never leave
/// the worker stack; only owned graph data and host byte vectors cross here.
#[derive(Debug)]
pub struct DenseRankWorkers {
    commands: [Sender<WorkerEnvelope>; 2],
    threads: [Option<JoinHandle<()>>; 2],
    rendezvous: Arc<RankRendezvous>,
    committed_frontiers: Arc<[AtomicU64; 2]>,
    published_frontiers: Arc<[AtomicU64; 2]>,
    deadline: Duration,
    sequence: u64,
    lost: Option<Error>,
    closed: bool,
    #[cfg(feature = "paged-attention-test-hooks")]
    mismatch_next: bool,
    #[cfg(feature = "paged-attention-test-hooks")]
    stall_next: Option<usize>,
    #[cfg(feature = "paged-attention-test-hooks")]
    fail_commit_next: Option<usize>,
    #[cfg(feature = "paged-attention-test-hooks")]
    fail_join_next: Option<usize>,
}

impl DenseRankWorkers {
    /// Acquire and pin one context on each persistent owner thread.
    pub fn spawn(config: DenseRankWorkerConfig) -> Result<Self> {
        if config.ranks[0] == config.ranks[1] || config.ordinals[0] == config.ordinals[1] {
            return Err(invalid(
                "ranks",
                "a TP2 worker pair needs distinct ranks and GPUs",
            ));
        }
        let rendezvous = Arc::new(RankRendezvous::new(config.ordinals[0]));
        let committed_frontiers = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let published_frontiers = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let (commands0, receiver0) = mpsc::channel();
        let (commands1, receiver1) = mpsc::channel();
        let (startup_id_tx, startup_id_rx) = mpsc::channel();
        let (startup_context_tx0, startup_context_rx0) = mpsc::channel();
        let (startup_context_tx1, startup_context_rx1) = mpsc::channel();
        let (peer_context_tx0, peer_context_rx0) = mpsc::channel();
        let (peer_context_tx1, peer_context_rx1) = mpsc::channel();
        let (startup_report_tx0, startup_report_rx0) = mpsc::channel();
        let (startup_report_tx1, startup_report_rx1) = mpsc::channel();
        let (startup_outcome_tx0, startup_outcome_rx0) = mpsc::channel();
        let (startup_outcome_tx1, startup_outcome_rx1) = mpsc::channel();
        let (startup_disabled_tx0, startup_disabled_rx0) = mpsc::channel();
        let (startup_disabled_tx1, startup_disabled_rx1) = mpsc::channel();
        let (startup_exit_tx0, startup_exit_rx0) = mpsc::channel();
        let (startup_exit_tx1, startup_exit_rx1) = mpsc::channel();
        let (ready0, ready_rx0) = mpsc::channel();
        let (ready1, ready_rx1) = mpsc::channel();
        let deadline = Instant::now() + config.deadline;

        let make_worker = |rank: usize,
                           receiver: Receiver<WorkerEnvelope>,
                           startup_context: Sender<StartupContext>,
                           peer_context: Receiver<PeerContextToken>,
                           startup_report: Sender<StartupReport>,
                           startup_outcome: Receiver<StartupOutcome>,
                           startup_disabled: Sender<StartupDisabled>,
                           startup_exit: Receiver<StartupExit>,
                           startup_id: StartupIdChannel,
                           ready: Sender<Result<()>>,
                           startup_deadline: Instant| {
            let worker_config = config.clone();
            let shared = Arc::clone(&rendezvous);
            let frontiers = Arc::clone(&committed_frontiers);
            let published = Arc::clone(&published_frontiers);
            thread::Builder::new()
                .name(format!("moxie-tp-rank-{rank}"))
                .spawn(move || {
                    rank_worker(
                        rank,
                        worker_config,
                        receiver,
                        startup_context,
                        peer_context,
                        startup_report,
                        startup_outcome,
                        startup_disabled,
                        startup_exit,
                        startup_id,
                        ready,
                        startup_deadline,
                        shared,
                        frontiers,
                        published,
                    )
                })
                .map_err(|error| Error::InvalidRequest {
                    field: "rank_worker",
                    detail: format!("could not start rank thread: {error}"),
                })
        };

        let first = make_worker(
            0,
            receiver0,
            startup_context_tx0,
            peer_context_rx0,
            startup_report_tx0,
            startup_outcome_rx0,
            startup_disabled_tx0,
            startup_exit_rx0,
            StartupIdChannel::Generate(startup_id_tx),
            ready0,
            deadline,
        )?;
        let second = match make_worker(
            1,
            receiver1,
            startup_context_tx1,
            peer_context_rx1,
            startup_report_tx1,
            startup_outcome_rx1,
            startup_disabled_tx1,
            startup_exit_rx1,
            StartupIdChannel::Receive(startup_id_rx),
            ready1,
            deadline,
        ) {
            Ok(thread) => thread,
            Err(error) => {
                drop(commands0);
                drop(first);
                return Err(error);
            }
        };
        let lost_startup =
            |detail: String| rendezvous.lose(format!("rank worker startup failed: {detail}"));
        let contexts = [
            recv_until(&startup_context_rx0, deadline, "rank 0 context token")
                .map_err(|error| lost_startup(error.to_string()))?,
            recv_until(&startup_context_rx1, deadline, "rank 1 context token")
                .map_err(|error| lost_startup(error.to_string()))?,
        ];
        if contexts[0].rank != 0
            || contexts[1].rank != 1
            || contexts
                .iter()
                .any(|entry| entry.generation != entry.token.generation())
        {
            return Err(lost_startup("rank context token identity disagrees".into()));
        }
        peer_context_tx0
            .send(contexts[1].token.clone())
            .map_err(|_| lost_startup("rank 0 did not accept its peer token".into()))?;
        peer_context_tx1
            .send(contexts[0].token.clone())
            .map_err(|_| lost_startup("rank 1 did not accept its peer token".into()))?;
        let reports = [
            recv_until(&startup_report_rx0, deadline, "rank 0 peer grant")
                .map_err(|error| lost_startup(error.to_string()))?,
            recv_until(&startup_report_rx1, deadline, "rank 1 peer grant")
                .map_err(|error| lost_startup(error.to_string()))?,
        ];
        if reports[0].rank != 0 || reports[1].rank != 1 {
            return Err(lost_startup("rank peer-grant reports disagree".into()));
        }
        let proceed = reports.iter().all(|report| report.result.is_ok());
        let outcome = if proceed {
            StartupOutcome::Proceed
        } else {
            StartupOutcome::Abort
        };
        startup_outcome_tx0
            .send(outcome)
            .map_err(|_| lost_startup("rank 0 missed its startup outcome".into()))?;
        startup_outcome_tx1
            .send(outcome)
            .map_err(|_| lost_startup("rank 1 missed its startup outcome".into()))?;
        if !proceed {
            let disabled_receivers = [startup_disabled_rx0, startup_disabled_rx1];
            let mut disable_error = None;
            for (rank, report) in reports.iter().enumerate() {
                if report.result.is_ok() {
                    let what = if rank == 0 {
                        "rank 0 peer-disable acknowledgement"
                    } else {
                        "rank 1 peer-disable acknowledgement"
                    };
                    let disabled = recv_until(&disabled_receivers[rank], deadline, what)
                        .map_err(|error| lost_startup(error.to_string()))?;
                    if disabled.rank != rank {
                        return Err(lost_startup(
                            "rank peer-disable acknowledgements disagree".into(),
                        ));
                    }
                    if let Err(error) = disabled.result {
                        disable_error.get_or_insert(error);
                    }
                }
            }
            if let Some(error) = disable_error {
                return Err(lost_startup(format!("peer grant disable failed: {error}")));
            }
        }
        startup_exit_tx0
            .send(StartupExit::Exit)
            .map_err(|_| lost_startup("rank 0 missed its startup exit".into()))?;
        startup_exit_tx1
            .send(StartupExit::Exit)
            .map_err(|_| lost_startup("rank 1 missed its startup exit".into()))?;
        let ready = [
            recv_until(&ready_rx0, deadline, "rank 0 startup outcome")
                .map_err(|error| lost_startup(error.to_string()))?,
            recv_until(&ready_rx1, deadline, "rank 1 startup outcome")
                .map_err(|error| lost_startup(error.to_string()))?,
        ];
        if let Some(error) = ready.iter().find_map(|result| result.as_ref().err()) {
            return Err(lost_startup(error.to_string()));
        }
        if !proceed {
            let error = reports
                .into_iter()
                .find_map(|report| report.result.err())
                .expect("Abort follows a failed peer grant");
            return Err(error);
        }
        Ok(Self {
            commands: [commands0, commands1],
            threads: [Some(first), Some(second)],
            rendezvous,
            committed_frontiers,
            published_frontiers,
            deadline: config.deadline,
            sequence: 0,
            lost: None,
            closed: false,
            #[cfg(feature = "paged-attention-test-hooks")]
            mismatch_next: false,
            #[cfg(feature = "paged-attention-test-hooks")]
            stall_next: None,
            #[cfg(feature = "paged-attention-test-hooks")]
            fail_commit_next: None,
            #[cfg(feature = "paged-attention-test-hooks")]
            fail_join_next: None,
        })
    }

    /// A test-only declaration mismatch. Both workers still enter the round.
    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn inject_collective_mismatch_once(&mut self) {
        self.mismatch_next = true;
    }

    /// A test-only worker stall immediately before its next rendezvous.
    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn stall_before_next_rendezvous(&mut self, rank: usize) {
        self.stall_next = Some(rank);
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn set_deadline_for_test(&mut self, deadline: Duration) {
        self.deadline = deadline;
    }

    /// Last committed frontier each worker published to the host. This is a
    /// read-only diagnostic mirror and remains available after sticky loss.
    pub fn committed_frontiers(&self) -> [u64; 2] {
        self.committed_frontiers
            .each_ref()
            .map(|frontier| frontier.load(Ordering::Acquire))
    }

    /// Last published KV row frontier from each owner, readable after loss.
    pub fn published_frontiers(&self) -> [u64; 2] {
        self.published_frontiers
            .each_ref()
            .map(|frontier| frontier.load(Ordering::Acquire))
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn refuse_next_commit_prepare(&mut self, rank: usize) {
        self.fail_commit_next = Some(rank);
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    pub fn fail_next_join_after_prepare(&mut self, rank: usize) {
        self.fail_join_next = Some(rank);
    }

    #[allow(clippy::too_many_arguments)]
    /// Cancellation is checked at stage boundaries, where both ranks have drained.
    pub fn execute_dense(
        &mut self,
        graph: &Graph,
        lowering: &TensorParallelLowering,
        oracle: moxie_graph::OracleId,
        oracles: &OracleRegistry,
        catalogue: &KernelCatalogue,
        rows: u64,
        visible_tokens: u64,
        bindings: &mut dyn FnMut(usize, &StageGraph) -> Result<Vec<OwnedBinding>>,
        cancel: &AtomicBool,
    ) -> Result<DenseWorkerStep<'_>> {
        self.check_live()?;
        let bytes = boundary_bytes(graph, rows)?;
        let begin = self.pair_command(
            [
                WorkerCommand::Begin {
                    boundary_bytes: bytes,
                },
                WorkerCommand::Begin {
                    boundary_bytes: bytes,
                },
            ],
            OP_BEGIN,
            [([bytes, rows, 0, 0], Precision::F32); 2],
        );
        let begin = match begin {
            Ok(begin) => begin,
            Err(error) => {
                if self.lost.is_none() {
                    let _ = self.settle(None, true);
                }
                return Err(error);
            }
        };
        let transactions = [transaction_reply(&begin[0])?, transaction_reply(&begin[1])?];
        let execution = (|| {
            for declared in &lowering.stages {
                if cancel.load(Ordering::Acquire) {
                    return Err(Error::Cancelled {
                        at: "tensor-parallel stage",
                    });
                }
                match declared {
                    Stage::Replicated(nodes) => {
                        for node in nodes.clone() {
                            let stage = build_stage_graph(
                                graph,
                                None,
                                node..node + 1,
                                None,
                                oracle,
                                oracles,
                            )?;
                            let original = graph.nodes()[node].output;
                            let commands = [0, 1].map(|rank| WorkerCommand::Stage {
                                stage: Box::new(stage.clone()),
                                catalogue: catalogue.clone(),
                                bindings: bindings(rank, &stage),
                                rows,
                                visible_tokens,
                                transaction: transactions[rank],
                                keep: Some(original),
                            });
                            self.pair_command(
                                commands,
                                OP_STAGE,
                                [([rows, node as u64, 1, 0], stage_precision(&stage)?); 2],
                            )?;
                            self.pair_command(
                                [WorkerCommand::Drain, WorkerCommand::Drain],
                                OP_DRAIN,
                                [([rows, node as u64, 0, 0], Precision::F32); 2],
                            )?;
                            self.pair_command(
                                [WorkerCommand::ClosePlans, WorkerCommand::ClosePlans],
                                OP_CLOSE_PLANS,
                                [([rows, node as u64, 0, 0], Precision::F32); 2],
                            )?;
                        }
                    }
                    Stage::Local { nodes, join } => {
                        let (Join::Gather { output } | Join::Reduce { output }) = *join;
                        let stages = [0, 1].map(|rank| {
                            build_stage_graph(
                                graph,
                                Some(&lowering.ranks[rank]),
                                nodes.clone(),
                                Some(output),
                                oracle,
                                oracles,
                            )
                        });
                        let [stage0, stage1] = stages;
                        let [stage0, stage1] = [stage0?, stage1?];
                        let commands = [
                            WorkerCommand::Stage {
                                bindings: bindings(0, &stage0),
                                stage: Box::new(stage0.clone()),
                                catalogue: catalogue.clone(),
                                rows,
                                visible_tokens,
                                transaction: transactions[0],
                                keep: None,
                            },
                            WorkerCommand::Stage {
                                bindings: bindings(1, &stage1),
                                stage: Box::new(stage1.clone()),
                                catalogue: catalogue.clone(),
                                rows,
                                visible_tokens,
                                transaction: transactions[1],
                                keep: None,
                            },
                        ];
                        let stage_results = self.pair_command(
                            commands,
                            OP_STAGE,
                            [(
                                [rows, nodes.start as u64, nodes.end as u64, 0],
                                stage_precision(&stage0)?,
                            ); 2],
                        )?;
                        let mut d0 = stage_reply_declaration(&stage_results[0])?;
                        let mut d1 = stage_reply_declaration(&stage_results[1])?;
                        d0.sequence = self.sequence;
                        d1.sequence = self.sequence;
                        #[cfg(feature = "paged-attention-test-hooks")]
                        let d1 = self.maybe_inject_shape_mismatch(d1)?;
                        let (v0, v1) = (stage0.graph.output(), stage1.graph.output());
                        let join_declarations = [
                            ([d0.rows, d0.columns, 0, 0], d0.precision),
                            ([d1.rows, d1.columns, 0, 0], d1.precision),
                        ];
                        self.pair_command(
                            [
                                WorkerCommand::JoinPrepare {
                                    join: *join,
                                    declaration: d0,
                                    source: v0,
                                },
                                WorkerCommand::JoinPrepare {
                                    join: *join,
                                    declaration: d1,
                                    source: v1,
                                },
                            ],
                            OP_JOIN_PREPARE,
                            join_declarations,
                        )?;
                        let fail_after_prepare = [self.take_join_fault(0), self.take_join_fault(1)];
                        self.pair_command(
                            [
                                WorkerCommand::JoinCopy {
                                    join: *join,
                                    declaration: d0,
                                    fail_after_prepare: fail_after_prepare[0],
                                },
                                WorkerCommand::JoinCopy {
                                    join: *join,
                                    declaration: d1,
                                    fail_after_prepare: fail_after_prepare[1],
                                },
                            ],
                            OP_JOIN_COPY,
                            join_declarations,
                        )?;
                        self.pair_command(
                            [
                                WorkerCommand::JoinDrain {
                                    join: *join,
                                    declaration: d0,
                                },
                                WorkerCommand::JoinDrain {
                                    join: *join,
                                    declaration: d1,
                                },
                            ],
                            OP_JOIN_DRAIN,
                            join_declarations,
                        )?;
                    }
                }
            }
            let output_spec = graph
                .spec(graph.output())
                .ok_or_else(|| invalid("logits", "the output has no tensor spec"))?;
            let bytes = match output_spec.role {
                ValueRole::Activation(_) => plan_value_bytes(graph, graph.output(), rows)?,
                _ => value_bytes(graph, graph.output(), rows, 4)?,
            };
            let outputs = self.pair_command(
                [
                    WorkerCommand::ReadOutput {
                        value: graph.output(),
                        bytes,
                    },
                    WorkerCommand::ReadOutput {
                        value: graph.output(),
                        bytes,
                    },
                ],
                OP_FINALIZE,
                [([rows, bytes, 0, 0], Precision::F32); 2],
            )?;
            let logits = bytes_reply(&outputs[0])?;
            self.settle(Some(transactions), false)?;
            Ok(logits)
        })();
        match execution {
            Ok(logits) => Ok(DenseWorkerStep {
                group: self,
                transactions,
                logits,
                settled: false,
            }),
            Err(error) => {
                if self.lost.is_none() {
                    let _ = self.settle(Some(transactions), true);
                }
                Err(error)
            }
        }
    }

    fn settle(&mut self, transactions: Option<[StateTransactionId; 2]>, abort: bool) -> Result<()> {
        self.pair_command(
            [WorkerCommand::Drain, WorkerCommand::Drain],
            OP_DRAIN,
            [([0; 4], Precision::F32); 2],
        )?;
        self.pair_command(
            [WorkerCommand::Cleanup, WorkerCommand::Cleanup],
            OP_CLEANUP,
            [([0; 4], Precision::F32); 2],
        )?;
        if abort {
            let aborts = transactions.map_or(
                [WorkerCommand::Abort(None), WorkerCommand::Abort(None)],
                |transactions| {
                    [
                        WorkerCommand::Abort(Some(transactions[0])),
                        WorkerCommand::Abort(Some(transactions[1])),
                    ]
                },
            );
            self.pair_command(aborts, OP_ABORT, [([0; 4], Precision::F32); 2])?;
        }
        Ok(())
    }

    fn pair_command(
        &mut self,
        commands: [WorkerCommand; 2],
        operation: u16,
        declarations: [([u64; 4], Precision); 2],
    ) -> Result<[WorkerValue; 2]> {
        self.check_live()?;
        let deadline = Instant::now() + self.deadline;
        let mut receivers: [Option<Receiver<WorkerReply>>; 2] = [None, None];
        for (rank, command) in commands.into_iter().enumerate() {
            let (reply, receiver) = mpsc::channel();
            let (shape, precision) = declarations[rank];
            let declaration = RankDeclaration {
                sequence: self.sequence,
                operation,
                rank: rank as u8,
                shape,
                precision,
            };
            let stall = self.take_stall(rank, operation);
            if self.commands[rank]
                .send(WorkerEnvelope {
                    command,
                    declaration,
                    stall,
                    deadline,
                    reply,
                })
                .is_err()
            {
                let error = self
                    .rendezvous
                    .lose(format!("rank {rank} worker stopped accepting commands"));
                self.lost = Some(error.clone());
                return Err(error);
            }
            receivers[rank] = Some(receiver);
        }
        let mut values: [Option<WorkerValue>; 2] = [None, None];
        let mut failures = [None, None];
        for (rank, receiver) in receivers.iter().enumerate() {
            let received = recv_until(
                receiver.as_ref().expect("response receiver exists"),
                deadline,
                "rank command",
            );
            let response = match received {
                Ok(response) => response,
                Err(error) => {
                    let error = self
                        .rendezvous
                        .lost()
                        .unwrap_or_else(|| self.rendezvous.lose(error.to_string()));
                    self.lost = Some(error.clone());
                    return Err(error);
                }
            };
            match response.result {
                Ok(value) => values[rank] = Some(value),
                Err(error @ Error::DeviceLost { .. }) => {
                    self.lost = Some(error.clone());
                    return Err(error);
                }
                Err(error) => failures[rank] = Some(error),
            }
        }
        self.sequence = self.sequence.checked_add(1).ok_or_else(|| {
            let error = self.rendezvous.lose("host collective sequence overflowed");
            self.lost = Some(error.clone());
            error
        })?;
        match (&failures[0], &failures[1]) {
            (None, None) => {}
            (Some(first), Some(second)) if first == second => return Err(first.clone()),
            _ => {
                let error = self
                    .rendezvous
                    .lose("rank workers left a status rendezvous with different outcomes");
                self.lost = Some(error.clone());
                return Err(error);
            }
        }
        Ok([
            values[0].take().expect("rank 0 replied"),
            values[1].take().expect("rank 1 replied"),
        ])
    }

    fn check_live(&mut self) -> Result<()> {
        if let Some(error) = self.lost.clone().or_else(|| self.rendezvous.lost()) {
            self.lost = Some(error.clone());
            return Err(error);
        }
        Ok(())
    }

    /// Published KV frontiers and live reservation counts, returned as owned
    /// host values from the rank owners.
    pub fn stats(&mut self) -> Result<[(u64, u64, usize); 2]> {
        let values = self.pair_command(
            [WorkerCommand::Stats, WorkerCommand::Stats],
            OP_STATS,
            [([0; 4], Precision::F32); 2],
        )?;
        let [
            WorkerValue::Stats {
                published: published0,
                committed: committed0,
                reservations: reservations0,
            },
            WorkerValue::Stats {
                published: published1,
                committed: committed1,
                reservations: reservations1,
            },
        ] = values
        else {
            return Err(invalid("rank_worker", "stats returned the wrong result"));
        };
        Ok([
            (published0, committed0, reservations0),
            (published1, committed1, reservations1),
        ])
    }

    /// Stop both workers on their owner threads after returning every run and
    /// reservation. A lost group remains parked with its contexts and grants.
    pub fn close(mut self) -> Result<()> {
        match self.shutdown_inner() {
            Ok(()) => {
                self.closed = true;
                Ok(())
            }
            Err(error) => {
                // A refused close retains rank contexts, communicators and
                // their charges; do not let Drop retry or release the group.
                core::mem::forget(self);
                Err(error)
            }
        }
    }

    fn shutdown_inner(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.pair_command(
            [WorkerCommand::ClosePrepare, WorkerCommand::ClosePrepare],
            OP_CLOSE_PREPARE,
            [([0; 4], Precision::F32); 2],
        )?;
        self.pair_command(
            [WorkerCommand::CloseDestroy, WorkerCommand::CloseDestroy],
            OP_CLOSE_DESTROY,
            [([0; 4], Precision::F32); 2],
        )?;
        self.pair_command(
            [WorkerCommand::CloseRelease, WorkerCommand::CloseRelease],
            OP_CLOSE_RELEASE,
            [([0; 4], Precision::F32); 2],
        )?;
        let values = self.pair_command(
            [WorkerCommand::Shutdown, WorkerCommand::Shutdown],
            OP_SHUTDOWN,
            [([0; 4], Precision::F32); 2],
        )?;
        if values
            .iter()
            .any(|value| !matches!(value, WorkerValue::Closed))
        {
            return Err(invalid("rank_worker", "shutdown returned the wrong result"));
        }
        for handle in &mut self.threads {
            if let Some(handle) = handle.take()
                && handle.join().is_err()
            {
                let error = self
                    .rendezvous
                    .lose("rank owner thread panicked during shutdown");
                self.lost = Some(error.clone());
                return Err(error);
            }
        }
        self.closed = true;
        Ok(())
    }

    fn send_commit(&mut self, transactions: [StateTransactionId; 2]) -> Result<()> {
        self.check_live()?;
        let sequence = self.sequence;
        let deadline = Instant::now() + self.deadline;
        let mut receivers: [Option<Receiver<WorkerReply>>; 2] = [None, None];
        for rank in 0..2 {
            let (reply, receiver) = mpsc::channel();
            let fail_prepare = self.take_commit_fault(rank);
            self.commands[rank]
                .send(WorkerEnvelope {
                    command: WorkerCommand::Commit {
                        transaction: transactions[rank],
                        sequence,
                        fail_prepare,
                    },
                    declaration: RankDeclaration {
                        sequence,
                        operation: OP_COMMIT_PREPARE,
                        rank: rank as u8,
                        shape: [0; 4],
                        precision: Precision::F32,
                    },
                    stall: false,
                    deadline,
                    reply,
                })
                .map_err(|_| {
                    let error = self
                        .rendezvous
                        .lose("rank worker stopped before commit prepare");
                    self.lost = Some(error.clone());
                    error
                })?;
            receivers[rank] = Some(receiver);
        }
        let mut rounds = [0u8; 2];
        let mut errors = [None, None];
        for rank in 0..2 {
            let reply = match recv_until(
                receivers[rank].as_ref().expect("commit receiver exists"),
                deadline,
                "rank commit",
            ) {
                Ok(reply) => reply,
                Err(error) => {
                    let error = self
                        .rendezvous
                        .lost()
                        .unwrap_or_else(|| self.rendezvous.lose(error.to_string()));
                    self.lost = Some(error.clone());
                    return Err(error);
                }
            };
            rounds[rank] = reply.rounds;
            if let Err(error) = reply.result {
                errors[rank] = Some(error);
            }
        }
        if rounds[0] != rounds[1] || rounds[0] == 0 {
            let error = self
                .rendezvous
                .lose("rank workers disagreed on commit phases");
            self.lost = Some(error.clone());
            return Err(error);
        }
        self.sequence = self
            .sequence
            .checked_add(rounds[0] as u64)
            .ok_or_else(|| self.rendezvous.lose("commit sequence overflowed"))?;
        if let Some(error) = errors.into_iter().flatten().next() {
            if matches!(error, Error::DeviceLost { .. }) {
                self.lost = Some(error.clone());
            }
            return Err(error);
        }
        Ok(())
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    fn maybe_inject_shape_mismatch(
        &mut self,
        mut declaration: crate::GatherDeclaration,
    ) -> Result<crate::GatherDeclaration> {
        if self.mismatch_next {
            self.mismatch_next = false;
            let elements = declaration
                .rows
                .checked_mul(declaration.columns)
                .filter(|&elements| elements > 1)
                .ok_or_else(|| invalid("collective", "test mismatch needs multiple elements"))?;
            if declaration.rows == elements {
                declaration.rows = 1;
                declaration.columns = elements;
            } else {
                declaration.rows = elements;
                declaration.columns = 1;
            }
        }
        Ok(declaration)
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    fn take_stall(&mut self, rank: usize, operation: u16) -> bool {
        if self.stall_next == Some(rank) && operation == OP_STAGE {
            self.stall_next = None;
            true
        } else {
            false
        }
    }

    #[cfg(not(feature = "paged-attention-test-hooks"))]
    fn take_stall(&mut self, _rank: usize, _operation: u16) -> bool {
        false
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    fn take_commit_fault(&mut self, rank: usize) -> bool {
        if self.fail_commit_next == Some(rank) {
            self.fail_commit_next = None;
            true
        } else {
            false
        }
    }

    #[cfg(not(feature = "paged-attention-test-hooks"))]
    fn take_commit_fault(&mut self, _rank: usize) -> bool {
        false
    }

    #[cfg(feature = "paged-attention-test-hooks")]
    fn take_join_fault(&mut self, rank: usize) -> bool {
        if self.fail_join_next == Some(rank) {
            self.fail_join_next = None;
            true
        } else {
            false
        }
    }

    #[cfg(not(feature = "paged-attention-test-hooks"))]
    fn take_join_fault(&mut self, _rank: usize) -> bool {
        false
    }
}

impl Drop for DenseRankWorkers {
    fn drop(&mut self) {
        if !self.closed && self.lost.is_none() && self.rendezvous.lost().is_none() {
            let _ = self.shutdown_inner();
        }
    }
}

/// A sampled TP result whose two rank transactions remain unpublished until
/// the caller commits or drops it.
#[derive(Debug)]
#[must_use = "dropping an uncommitted step aborts both rank transactions"]
pub struct DenseWorkerStep<'g> {
    group: &'g mut DenseRankWorkers,
    transactions: [StateTransactionId; 2],
    logits: Vec<u8>,
    settled: bool,
}

impl DenseWorkerStep<'_> {
    pub fn logits(&self) -> &[u8] {
        &self.logits
    }

    pub fn commit(mut self) -> Result<Vec<u8>> {
        self.settled = true;
        let commits = self.group.send_commit(self.transactions);
        match commits {
            Ok(()) => Ok(std::mem::take(&mut self.logits)),
            Err(error) => {
                if self.group.lost.is_none() {
                    let _ = self.group.settle(Some(self.transactions), true);
                }
                Err(error)
            }
        }
    }
}

impl Drop for DenseWorkerStep<'_> {
    fn drop(&mut self) {
        if !self.settled && self.group.lost.is_none() {
            let _ = self.group.settle(Some(self.transactions), true);
        }
    }
}

pub(crate) fn recv_until<T>(receiver: &Receiver<T>, deadline: Instant, what: &str) -> Result<T> {
    receiver
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|_| Error::DeviceLost {
            device: 0,
            detail: format!("{what} did not complete before the deadline"),
        })
}

fn transaction_reply(reply: &WorkerValue) -> Result<StateTransactionId> {
    match reply {
        WorkerValue::Transaction(transaction) => Ok(*transaction),
        _ => Err(invalid("rank_worker", "begin returned the wrong result")),
    }
}

fn bytes_reply(reply: &WorkerValue) -> Result<Vec<u8>> {
    match reply {
        WorkerValue::Bytes(bytes) => Ok(bytes.clone()),
        _ => Err(invalid("rank_worker", "readback returned the wrong result")),
    }
}

fn stage_precision(stage: &StageGraph) -> Result<Precision> {
    let role = stage
        .graph
        .spec(stage.graph.output())
        .ok_or_else(|| invalid("stage", "stage output has no tensor spec"))?
        .role;
    match role {
        ValueRole::Activation(precision) => Ok(precision.get()),
        _ => Err(invalid("stage", "stage output is not an activation")),
    }
}

fn stage_reply_declaration(reply: &WorkerValue) -> Result<crate::GatherDeclaration> {
    match reply {
        WorkerValue::Stage(Some(declaration)) => Ok(*declaration),
        _ => Err(invalid(
            "rank_worker",
            "local stage returned no join declaration",
        )),
    }
}

fn declare_selected_plan(
    plan: &SelectedReservedPlan<'_>,
    local: ValueId,
) -> Result<crate::GatherDeclaration> {
    let planned = plan
        .candidate()
        .value(local)
        .ok_or_else(|| invalid("collective", "stage output is absent from its plan"))?;
    let (ValueRole::Activation(precision), &[rows, columns]) =
        (planned.role, planned.shape.as_slice())
    else {
        return Err(invalid(
            "collective",
            "a joined stage output must be a two-dimensional activation",
        ));
    };
    Ok(crate::GatherDeclaration {
        sequence: 0,
        rows,
        columns,
        precision: precision.get(),
    })
}

#[allow(clippy::too_many_arguments)]
fn rank_worker(
    rank: usize,
    config: DenseRankWorkerConfig,
    commands: Receiver<WorkerEnvelope>,
    startup_context: Sender<StartupContext>,
    peer_context: Receiver<PeerContextToken>,
    startup_report: Sender<StartupReport>,
    startup_outcome: Receiver<StartupOutcome>,
    startup_disabled: Sender<StartupDisabled>,
    startup_exit: Receiver<StartupExit>,
    startup_id: StartupIdChannel,
    ready: Sender<Result<()>>,
    startup_deadline: Instant,
    rendezvous: Arc<RankRendezvous>,
    committed_frontiers: Arc<[AtomicU64; 2]>,
    published_frontiers: Arc<[AtomicU64; 2]>,
) {
    let context = match RankContext::acquire(config.ranks[rank], config.ordinals[rank]) {
        Ok(context) => context,
        Err(error) => {
            let _ = startup_report.send(StartupReport {
                rank,
                result: Err(error),
            });
            return;
        }
    };
    let token = context.peer_context_token();
    if startup_context
        .send(StartupContext {
            rank,
            generation: token.generation(),
            token,
        })
        .is_err()
    {
        std::mem::forget(context);
        return;
    }
    let peer_context = match peer_context
        .recv_timeout(startup_deadline.saturating_duration_since(Instant::now()))
    {
        Ok(peer_context) => peer_context,
        Err(_) => {
            std::mem::forget(context);
            return;
        }
    };
    let grant = context.enable_peer_access_to(&peer_context);
    if startup_report
        .send(StartupReport {
            rank,
            result: grant.as_ref().copied().map_err(Clone::clone),
        })
        .is_err()
    {
        std::mem::forget(context);
        return;
    }
    let outcome = match startup_outcome
        .recv_timeout(startup_deadline.saturating_duration_since(Instant::now()))
    {
        Ok(outcome) => outcome,
        Err(_) => {
            std::mem::forget(context);
            return;
        }
    };
    match outcome {
        StartupOutcome::Abort => {
            if grant.is_ok() {
                let disabled = context.disable_peer_access_to(&peer_context);
                let _ = startup_disabled.send(StartupDisabled {
                    rank,
                    result: disabled,
                });
            }
            match startup_exit
                .recv_timeout(startup_deadline.saturating_duration_since(Instant::now()))
            {
                Ok(StartupExit::Exit) => {
                    let _ = ready.send(Ok(()));
                    return;
                }
                Err(_) => {
                    std::mem::forget(context);
                    return;
                }
            }
        }
        StartupOutcome::Proceed if grant.is_ok() => {}
        StartupOutcome::Proceed => park_lost(),
    }

    let nccl_id = match startup_id {
        StartupIdChannel::Generate(sender) => match NcclId::generate(&context) {
            Ok(id) => {
                if sender.send(Ok(id)).is_err() {
                    let lost = rendezvous.lose("rank 1 did not receive the NCCL unique id");
                    let _ = ready.send(Err(lost));
                    park_lost();
                }
                id
            }
            Err(error) => {
                let _ = sender.send(Err(error.clone()));
                let lost =
                    rendezvous.lose(format!("rank {rank} could not create an NCCL id: {error}"));
                let _ = ready.send(Err(lost));
                park_lost();
            }
        },
        StartupIdChannel::Receive(receiver) => match receiver
            .recv_timeout(startup_deadline.saturating_duration_since(Instant::now()))
        {
            Ok(Ok(id)) => id,
            Ok(Err(error)) => {
                let lost = rendezvous.lose(format!("rank 0 could not create an NCCL id: {error}"));
                let _ = ready.send(Err(lost));
                park_lost();
            }
            Err(_) => {
                let lost = rendezvous
                    .lose("rank 1 did not receive the NCCL unique id before startup deadline");
                let _ = ready.send(Err(lost));
                park_lost();
            }
        },
    };

    let stream = match Stream::new(&context) {
        Ok(stream) => stream,
        Err(error) => {
            let lost = rendezvous.lose(format!(
                "rank {rank} startup could not create a stream: {error}"
            ));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let capability = context.capability().clone();
    let measurement = match context.measure() {
        Ok(measurement) => measurement,
        Err(error) => {
            let lost = rendezvous.lose(format!("rank {rank} startup measurement failed: {error}"));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let device = match CapacitySnapshot::measured(&measurement, 1 << 20) {
        Ok(device) => device,
        Err(error) => {
            let lost = rendezvous.lose(format!("rank {rank} startup capacity failed: {error}"));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let ledger = match Ledger::new([device, config.host_capacity.clone()]) {
        Ok(ledger) => ledger,
        Err(error) => {
            let lost = rendezvous.lose(format!("rank {rank} startup ledger failed: {error}"));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let state = match DeviceKvSequence::new(config.geometry.clone()) {
        Ok(state) => state,
        Err(error) => {
            let lost = rendezvous.lose(format!("rank {rank} startup state failed: {error}"));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let (mut ledger, runs) = match admit_worker_runs(
        &context,
        ledger,
        &state,
        &config.geometry,
        config.heads,
        config.max_rows,
    ) {
        Ok((ledger, runs)) => (ledger, runs),
        Err(error) => {
            let lost =
                rendezvous.lose(format!("rank {rank} startup run admission failed: {error}"));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let nccl_reservation = match admit_nccl_reserve(&mut ledger, context.uuid()) {
        Ok(reservation) => reservation,
        Err(error) => {
            for run in runs {
                if let Err(refused) = run.close(&mut ledger) {
                    let _ = refused;
                    break;
                }
            }
            let lost = rendezvous.lose(format!(
                "rank {rank} NCCL reserve admission failed: {error}"
            ));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    };
    let mut worker = WorkerState {
        ctx: &context,
        capability,
        stream: &stream,
        ledger,
        state,
        runs,
        held: Held::default(),
        transaction: None,
        temp: None,
        communicator: None,
        nccl_reservation: Some(nccl_reservation),
        preserve_communicator_on_loss: false,
    };
    match Communicator::init(&context, 2, nccl_id, rank as i32, startup_deadline) {
        Ok(communicator) => worker.communicator = Some(communicator),
        Err(error) => {
            let lost = rendezvous.lose(format!("rank {rank} NCCL initialization failed: {error}"));
            let _ = ready.send(Err(lost));
            park_lost();
        }
    }
    if ready.send(Ok(())).is_err() {
        worker.abort_communicator();
        park_lost();
    }
    worker_loop(
        rank,
        commands,
        &mut worker,
        &rendezvous,
        committed_frontiers,
        published_frontiers,
    );
}

pub(crate) fn admit_worker_runs<'ctx>(
    context: &'ctx RankContext,
    mut ledger: Ledger,
    state: &DeviceKvSequence,
    geometry: &KvGeometry,
    heads: u64,
    max_rows: u64,
) -> Result<(Ledger, Vec<PagedAttentionRun<'ctx>>)> {
    let descriptor = moxie_kernels::paged_attention_catalogue()
        .descriptors()
        .iter()
        .find(|descriptor| {
            descriptor.sm.major == context.capability().compute_major
                && descriptor.sm.minor == context.capability().compute_minor
        })
        .cloned()
        .ok_or_else(|| Error::Unsupported {
            capability: "paged_attention",
            reason: "the device has no qualified paged-attention descriptor".into(),
        })?;
    let lineage = u64::try_from(geometry.max_tokens)
        .ok()
        .and_then(|rows| rows.checked_add(1))
        .ok_or_else(|| invalid("geometry", "maximum sequence length overflowed"))?;
    let mut runs = Vec::new();
    runs.try_reserve_exact(geometry.layers.len())
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(Tier::Host(moxie_types::HostTier::Pageable)),
            requested_bytes: (geometry.layers.len()
                * std::mem::size_of::<PagedAttentionRun<'static>>())
                as u64,
            available_bytes: 0,
        })?;
    for (layer, declared) in geometry.layers.iter().enumerate() {
        let layout = state.layout(layer)?;
        let page_geometry = crate::PageGeometry {
            kv_heads: declared.kv_heads as u64,
            head_dim: declared.key_dim as u64,
            page_tokens: geometry.page_tokens as u64,
            pages: layout.pages,
        };
        let run = if layer == 0 {
            PagedAttentionRun::admit_for_sequence(
                &mut ledger,
                context,
                descriptor.clone(),
                page_geometry,
                heads,
                max_rows,
                lineage,
                Staging::DeviceHandles,
            )
        } else {
            PagedAttentionRun::admit(
                &mut ledger,
                context,
                descriptor.clone(),
                page_geometry,
                heads,
                max_rows,
                Staging::DeviceHandles,
            )
        }
        .map_err(|refused| refused.error)?;
        runs.push(run);
    }
    Ok((ledger, runs))
}

fn admit_nccl_reserve(ledger: &mut Ledger, device: moxie_types::DeviceUuid) -> Result<Reservation> {
    let mut request = PlanRequest::new("NCCL communicator reserve", ["startup"])?;
    request.buffer(BufferRequest::try_new(
        "NCCL communicator device reserve",
        Scope::Device(device),
        Tier::Device(DeviceTier::CollectiveBuffers),
        NCCL_DEVICE_RESERVE_BYTES,
        StageSpan::at(0),
    )?)?;
    ledger.admit(&request).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn worker_loop<'ctx>(
    rank: usize,
    commands: Receiver<WorkerEnvelope>,
    worker: &mut WorkerState<'ctx>,
    rendezvous: &RankRendezvous,
    committed_frontiers: Arc<[AtomicU64; 2]>,
    published_frontiers: Arc<[AtomicU64; 2]>,
) {
    loop {
        let envelope = match commands.recv() {
            Ok(envelope) => envelope,
            Err(_) => {
                rendezvous.lose(format!(
                    "rank {rank} command channel closed before shutdown"
                ));
                worker.abort_communicator();
                park_lost();
            }
        };
        if let Some(error) = rendezvous.lost() {
            let _ = envelope.reply.send(WorkerReply {
                result: Err(error),
                rounds: 0,
            });
            worker.abort_communicator();
            park_lost();
        }
        if envelope.stall {
            park_lost();
        }
        match envelope.command {
            WorkerCommand::Commit {
                transaction,
                sequence,
                fail_prepare,
            } => {
                let (result, rounds) = worker_commit(
                    rank,
                    worker,
                    transaction,
                    sequence,
                    fail_prepare,
                    rendezvous,
                    envelope.deadline,
                    &committed_frontiers[rank],
                );
                if let Ok(rows) = worker.state.published_rows() {
                    published_frontiers[rank].store(rows, Ordering::Release);
                }
                let _ = envelope.reply.send(WorkerReply {
                    result: result.map(|()| WorkerValue::Unit),
                    rounds,
                });
                if rendezvous.lost().is_some() {
                    worker.abort_communicator();
                    park_lost();
                }
            }
            command => {
                let shutdown = matches!(&command, WorkerCommand::Shutdown);
                let deadline = envelope.deadline;
                let local = worker.execute(command, deadline);
                if let Ok(rows) = worker.state.published_rows() {
                    published_frontiers[rank].store(rows, Ordering::Release);
                }
                let status = local.as_ref().map(|_| ()).map_err(Clone::clone);
                let agreed = rendezvous.enter(rank, envelope.declaration, status, deadline);
                let result = agreed.map(|()| local).and_then(core::convert::identity);
                let _ = envelope.reply.send(WorkerReply { result, rounds: 1 });
                if rendezvous.lost().is_some() {
                    worker.abort_communicator();
                    park_lost();
                }
                if shutdown {
                    return;
                }
            }
        }
    }
}

pub(crate) fn park_lost() -> ! {
    loop {
        thread::park();
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_commit<'ctx>(
    rank: usize,
    worker: &mut WorkerState<'ctx>,
    transaction: StateTransactionId,
    sequence: u64,
    fail_prepare: bool,
    rendezvous: &RankRendezvous,
    deadline: Instant,
    committed_frontier: &AtomicU64,
) -> (Result<()>, u8) {
    let mut preparation_error = None;
    let mut prepared: Option<PreparedCommit> = None;
    if fail_prepare {
        preparation_error = Some(Error::InvalidRequest {
            field: "fault",
            detail: "injected commit preparation refusal".into(),
        });
    }
    match worker.state.layer_count() {
        Ok(layers) if layers == worker.runs.len() && preparation_error.is_none() => {
            match worker.state.prepare_commit(transaction, 0) {
                Ok(commit) => prepared = Some(commit),
                Err(error) => preparation_error = Some(error),
            }
        }
        Ok(_) if preparation_error.is_none() => {
            preparation_error = Some(invalid(
                "runs",
                "commit needs exactly one device run per state layer",
            ));
        }
        Err(error) if preparation_error.is_none() => preparation_error = Some(error),
        _ => {}
    }

    let prepare_declaration = RankDeclaration {
        sequence,
        operation: OP_COMMIT_PREPARE,
        rank: rank as u8,
        shape: [0; 4],
        precision: Precision::F32,
    };
    if let Some(error) = preparation_error {
        return (
            rendezvous.enter(rank, prepare_declaration, Err(error), deadline),
            1,
        );
    }

    let apply_declaration = RankDeclaration {
        sequence: sequence.saturating_add(1),
        operation: OP_COMMIT_APPLY,
        rank: rank as u8,
        shape: [0; 4],
        precision: Precision::F32,
    };
    let run_count = worker.runs.len();
    let stream = worker.stream;
    let ordinal = worker.ctx.ordinal();
    let state = &mut worker.state;
    let transaction_slot = &mut worker.transaction;
    let mut callback_entered = false;
    let result = with_paged_writers(&mut worker.runs, stream, |writers| {
        callback_entered = true;
        if let Err(error) = rendezvous.enter(rank, prepare_declaration, Ok(()), deadline) {
            return Ok((Err(error), 1));
        }
        let Some(prepared) = prepared else {
            return Ok((
                Err(invalid(
                    "prepared_commit",
                    "commit prepare returned no handle",
                )),
                1,
            ));
        };
        let applied = state.apply_commit(prepared, writers).map_err(|error| {
            device_lost(
                ordinal,
                format!("prepared commit publication failed ({error}); rank state may diverge"),
            )
        });
        if applied.is_ok() {
            *transaction_slot = None;
            if let Ok(rows) = state.committed_rows() {
                committed_frontier.store(rows, Ordering::Release);
            }
        }
        Ok((
            rendezvous.enter(rank, apply_declaration, applied, deadline),
            2,
        ))
    });
    if !callback_entered {
        return (
            rendezvous.enter(
                rank,
                prepare_declaration,
                Err(commit_capacity_error(run_count)),
                deadline,
            ),
            1,
        );
    }
    result.expect("paged-writer callback returns its rendezvous result")
}

pub(crate) fn commit_capacity_error(count: usize) -> Error {
    Error::CapacityExceeded {
        tier: Some(Tier::Host(moxie_types::HostTier::Pageable)),
        requested_bytes: count.saturating_mul(std::mem::size_of::<&mut dyn PagedKvWriter>()) as u64,
        available_bytes: 0,
    }
}

impl<'ctx> WorkerState<'ctx> {
    fn abort_communicator(&mut self) {
        if self.preserve_communicator_on_loss {
            return;
        }
        if let Some(communicator) = self.communicator.take() {
            let _ = communicator.abort();
        }
    }

    fn drain(&mut self, deadline: Instant) -> Result<()> {
        if self.communicator.is_none() {
            return Err(device_lost(
                self.ctx.ordinal(),
                "rank stream drain has no NCCL communicator".into(),
            ));
        }
        loop {
            let poll = self
                .communicator
                .as_ref()
                .expect("NCCL communicator remains until drain settles")
                .poll();
            match poll {
                Ok(CommState::Ready) if Instant::now() < deadline => break,
                Ok(CommState::InProgress) if Instant::now() < deadline => thread::yield_now(),
                Ok(CommState::Ready | CommState::InProgress) => {
                    self.abort_communicator();
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        "NCCL work missed the group deadline before stream drain".into(),
                    ));
                }
                Err(error) => {
                    self.abort_communicator();
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        format!("NCCL asynchronous operation failed ({error})"),
                    ));
                }
            }
        }

        // NCCL may enqueue stream work after group_end returns InProgress.
        let event = match Event::new(self.ctx) {
            Ok(event) => event,
            Err(error) => {
                self.abort_communicator();
                return Err(device_lost(
                    self.ctx.ordinal(),
                    format!("NCCL stream completion event could not be created ({error})"),
                ));
            }
        };
        if let Err(error) = event.record(self.stream) {
            self.abort_communicator();
            return Err(device_lost(
                self.ctx.ordinal(),
                format!("NCCL stream completion event could not be recorded ({error})"),
            ));
        }
        loop {
            match event.is_complete() {
                Ok(true) if Instant::now() < deadline => return Ok(()),
                Ok(true) | Ok(false) => {}
                Err(error) => {
                    self.abort_communicator();
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        format!("NCCL stream completion could not be queried ({error})"),
                    ));
                }
            }
            if Instant::now() >= deadline {
                self.abort_communicator();
                return Err(device_lost(
                    self.ctx.ordinal(),
                    "rank stream carrying NCCL work missed the group deadline".into(),
                ));
            }
            thread::yield_now();
        }
    }

    fn prepare_close(&mut self, deadline: Instant) -> Result<()> {
        self.drain(deadline)?;
        self.close_runs()?;
        let Some(communicator) = self.communicator.as_mut() else {
            return Err(device_lost(
                self.ctx.ordinal(),
                "worker close found no NCCL communicator".into(),
            ));
        };
        if let Err(error) = communicator.finalize(deadline) {
            self.abort_communicator();
            return Err(device_lost(
                self.ctx.ordinal(),
                format!("NCCL communicator could not finalize ({error})"),
            ));
        }
        Ok(())
    }

    fn destroy_communicator(&mut self) -> Result<()> {
        let Some(communicator) = self.communicator.take() else {
            return Err(device_lost(
                self.ctx.ordinal(),
                "worker close found no NCCL communicator to destroy".into(),
            ));
        };
        // SAFETY: ClosePrepare drained the stream and completed communicator
        // finalization before the paired destroy command.
        match unsafe { communicator.destroy() } {
            Ok(()) => Ok(()),
            Err((communicator, error)) => {
                self.communicator = Some(communicator);
                self.preserve_communicator_on_loss = true;
                Err(error)
            }
        }
    }

    fn release_nccl_reservation(&mut self) -> Result<()> {
        let Some(reservation) = self.nccl_reservation.take() else {
            return Err(device_lost(
                self.ctx.ordinal(),
                "worker close found no NCCL reservation".into(),
            ));
        };
        if let Err(refused) = self.ledger.release(reservation) {
            self.nccl_reservation = Some(refused.reservation);
            return Err(device_lost(
                self.ctx.ordinal(),
                format!("NCCL reservation could not be released ({})", refused.error),
            ));
        }
        Ok(())
    }

    fn execute(&mut self, command: WorkerCommand, deadline: Instant) -> Result<WorkerValue> {
        match command {
            WorkerCommand::Begin { boundary_bytes } => {
                let mut all_runs_observed = true;
                for run in &mut self.runs {
                    if run.observe_pending().is_err() {
                        all_runs_observed = false;
                    }
                }
                if !all_runs_observed || self.runs.iter().any(|run| !run.is_idle()) {
                    return Err(invalid(
                        "runs",
                        "a dense step accepts only idle attention runs",
                    ));
                }
                if self.transaction.is_some() || self.held.boundary.is_some() {
                    return Err(invalid(
                        "transaction",
                        "the worker already has an open step",
                    ));
                }
                let transaction = self.state.begin()?;
                self.transaction = Some(transaction);
                self.held.boundary = Some(self.open_boundary(boundary_bytes)?);
                Ok(WorkerValue::Transaction(transaction))
            }
            WorkerCommand::Stage {
                stage,
                catalogue,
                bindings,
                rows,
                visible_tokens,
                transaction,
                keep: original,
            } => {
                let local_output = stage.graph.output();
                let declaration = self.run_stage(
                    *stage,
                    catalogue,
                    bindings,
                    rows,
                    visible_tokens,
                    transaction,
                    original.is_none(),
                )?;
                if let Some(original) = original {
                    let plan = self.held.plans.last().expect("stage plan is open");
                    let boundary = self.held.boundary.as_mut().expect("step boundary is open");
                    keep(plan, local_output, original, boundary, self.stream)?;
                }
                Ok(WorkerValue::Stage(declaration))
            }
            WorkerCommand::Drain => {
                self.drain(deadline)?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::ClosePlans => {
                close_plans(self.ctx, &mut self.ledger, &mut self.held.plans)?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::JoinPrepare {
                join,
                declaration,
                source,
            } => {
                self.prepare_join(join, declaration, source)?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::JoinCopy {
                join,
                declaration,
                fail_after_prepare,
            } => {
                self.copy_join(join, declaration, fail_after_prepare)?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::JoinDrain { join, declaration } => {
                self.drain_join(join, declaration, deadline)?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::Cleanup => {
                self.cleanup()?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::Abort(transaction) => {
                if let Some(current) = self.transaction {
                    if transaction.is_some_and(|expected| expected != current) {
                        return Err(invalid(
                            "transaction",
                            "abort names a different transaction",
                        ));
                    }
                    self.state.abort(current)?;
                    self.transaction = None;
                }
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::ReadOutput { value, bytes } => {
                let range = self
                    .held
                    .boundary
                    .as_ref()
                    .and_then(|boundary| boundary.values.get(&value))
                    .ok_or_else(|| invalid("logits", "the output is absent from the boundary"))?;
                let count = usize::try_from(bytes)
                    .map_err(|_| invalid("logits", "output size is not addressable"))?;
                let mut host = Vec::new();
                host.try_reserve_exact(count)
                    .map_err(|_| Error::CapacityExceeded {
                        tier: Some(Tier::Host(moxie_types::HostTier::Pageable)),
                        requested_bytes: bytes,
                        available_bytes: 0,
                    })?;
                host.resize(count, 0);
                range.copy_to_host(&mut host)?;
                Ok(WorkerValue::Bytes(host))
            }
            WorkerCommand::Stats => {
                self.stats()
                    .map(|(published, committed, reservations)| WorkerValue::Stats {
                        published,
                        committed,
                        reservations,
                    })
            }
            WorkerCommand::Commit { .. } => unreachable!("commit handled by worker loop"),
            WorkerCommand::ClosePrepare => {
                self.prepare_close(deadline)?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::CloseDestroy => {
                self.destroy_communicator()?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::CloseRelease => {
                self.release_nccl_reservation()?;
                Ok(WorkerValue::Unit)
            }
            WorkerCommand::Shutdown => {
                if self.communicator.is_some()
                    || self.nccl_reservation.is_some()
                    || !self.ledger.outstanding().is_empty()
                {
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        "worker shutdown retains an NCCL resource".into(),
                    ));
                }
                Ok(WorkerValue::Closed)
            }
        }
    }

    fn open_boundary(&mut self, bytes: u64) -> Result<Boundary<'ctx>> {
        let scope = Scope::Device(self.ctx.uuid());
        let mut request = PlanRequest::new("tensor-parallel step boundaries", ["step"])?;
        request.buffer(BufferRequest::try_new(
            "stage boundaries",
            scope,
            Tier::Device(DeviceTier::Activations),
            bytes,
            StageSpan { first: 0, last: 0 },
        )?)?;
        let reservation = self.ledger.admit(&request)?;
        match crate::DeviceArena::create(
            &self.ledger,
            reservation,
            self.ctx,
            DeviceTier::Activations,
            bytes,
            "tensor-parallel step boundaries",
        ) {
            Ok(arena) => Ok(Boundary {
                arena,
                values: std::collections::BTreeMap::new(),
            }),
            Err(refused) => match self.ledger.release(refused.reservation) {
                Ok(()) => Err(refused.error),
                Err(held) => Err(device_lost(
                    self.ctx.ordinal(),
                    format!(
                        "boundary reservation could not be released ({})",
                        held.error
                    ),
                )),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_stage(
        &mut self,
        stage: StageGraph,
        catalogue: KernelCatalogue,
        bindings: Result<Vec<OwnedBinding>>,
        rows: u64,
        visible_tokens: u64,
        transaction: StateTransactionId,
        declare_collective: bool,
    ) -> Result<Option<crate::GatherDeclaration>> {
        if self.transaction != Some(transaction) {
            return Err(invalid(
                "transaction",
                "stage names a different worker transaction",
            ));
        }
        let boundary = self.held.boundary.as_ref().expect("step boundary is open");
        let workload = moxie_plan::ResourceWorkload {
            phase: if rows == 1 {
                moxie_plan::Phase::Decode
            } else {
                moxie_plan::Phase::Prefill
            },
            rows,
            visible_tokens,
            branch_rows: rows,
            output: stage.graph.output(),
            device: self.capability.uuid,
            paged_state_capacity: None,
        };
        let candidate = moxie_plan::lower_selected_ordered(
            &stage.graph,
            workload,
            &self.capability,
            &catalogue,
            &stage.linear_orders,
            &stage.combine_orders,
            &stage.expert_ownership,
        )?;
        let plan = SelectedReservedPlan::admit(
            candidate,
            &stage.graph,
            &self.capability,
            &catalogue,
            &mut self.ledger,
            self.ctx,
        )
        .map_err(|refused| match refused {
            SelectedAdmitRefused::Invalid { error, .. } => error,
            SelectedAdmitRefused::Rejected { rejection, .. } => rejection.into(),
            SelectedAdmitRefused::Held { error, .. } => device_lost(
                self.ctx.ordinal(),
                format!("a refused stage admission could not release its reservation ({error})"),
            ),
        })?;
        let declaration = if declare_collective {
            Some(declare_selected_plan(&plan, stage.graph.output())?)
        } else {
            None
        };
        let mut resident = std::collections::BTreeSet::new();
        let copied = stage.reads.iter().try_for_each(|read| {
            let Some(source) = boundary.values.get(&read.original) else {
                return Ok(());
            };
            resident.insert(read.local);
            copy_boundary(&plan, read.local, source, read.slice, rows, self.stream)
        });
        let bindings = match copied.and(bindings) {
            Ok(bindings) => bindings,
            Err(error) => {
                self.held.plans.push(plan);
                return Err(error);
            }
        };
        let runs: &mut [PagedAttentionRun<'ctx>] = if stage.state_layers.is_empty() {
            &mut []
        } else {
            self.runs.as_mut_slice()
        };
        let lease = plan.execute_dense_stage(
            DenseGraphStep {
                graph: &stage.graph,
                capability: &self.capability,
                catalogue: &catalogue,
                ctx: self.ctx,
                stream: self.stream,
                state: &mut self.state,
                transaction,
                runs,
                bindings,
                host_experts: &[],
            },
            &resident,
            &stage.state_layers,
        );
        let lease = match lease {
            Ok(lease) => lease,
            Err(refused) => {
                self.held.plans.extend(refused.plan);
                self.held.leases.extend(refused.held);
                return Err(refused.error);
            }
        };
        match lease.finish() {
            Ok(result) => self.held.plans.push(result.plan),
            Err(refused) => {
                self.held.leases.push(refused.lease);
                return Err(refused.error);
            }
        }
        Ok(declaration)
    }

    fn prepare_join(
        &mut self,
        join: Join,
        declaration: crate::GatherDeclaration,
        source: ValueId,
    ) -> Result<()> {
        if self.temp.is_some() {
            return Err(invalid(
                "collective",
                "a prior join still owns temporary ranges",
            ));
        }
        let element = match (join, declaration.precision) {
            (Join::Gather { .. }, Precision::Bf16) => 2u64,
            (Join::Gather { .. }, Precision::F32) => 4u64,
            (Join::Reduce { .. }, Precision::F32) => 4u64,
            _ => {
                return Err(invalid(
                    "collective",
                    &format!("unsupported {join:?} precision {:?}", declaration.precision),
                ));
            }
        };
        let elements = declaration
            .rows
            .checked_mul(declaration.columns)
            .ok_or_else(|| invalid("collective", "join element count overflowed"))?;
        let part_bytes = elements
            .checked_mul(element)
            .ok_or_else(|| invalid("collective", "join byte count overflowed"))?;
        let collective_bytes = match join {
            Join::Gather { .. } => part_bytes
                .checked_mul(2)
                .ok_or_else(|| invalid("collective", "gather buffer size overflowed"))?,
            Join::Reduce { .. } => elements
                .checked_mul(4)
                .ok_or_else(|| invalid("collective", "reduce buffer size overflowed"))?,
        };
        let output_bytes = match join {
            Join::Gather { .. } => part_bytes
                .checked_mul(2)
                .ok_or_else(|| invalid("collective", "gather output size overflowed"))?,
            Join::Reduce { .. } => elements
                .checked_mul(2)
                .ok_or_else(|| invalid("collective", "reduce output size overflowed"))?,
        };
        let capacity = [JOIN_STATUS_BYTES, collective_bytes, part_bytes]
            .into_iter()
            .try_fold(0u64, |total, bytes| {
                let aligned = bytes
                    .checked_add(ALIGNMENT - 1)
                    .map(|value| value / ALIGNMENT * ALIGNMENT)
                    .ok_or_else(|| invalid("collective", "temporary size overflowed"))?;
                total
                    .checked_add(aligned)
                    .ok_or_else(|| invalid("collective", "temporary size overflowed"))
            })?;
        let mut request = PlanRequest::new("tensor-parallel join buffers", ["join"])?;
        request.buffer(BufferRequest::try_new(
            "status, collective and dummy buffers",
            Scope::Device(self.ctx.uuid()),
            Tier::Device(DeviceTier::CollectiveBuffers),
            capacity,
            StageSpan::at(0),
        )?)?;
        let reservation = self.ledger.admit(&request)?;
        let arena = match DeviceArena::create(
            &self.ledger,
            reservation,
            self.ctx,
            DeviceTier::CollectiveBuffers,
            capacity,
            "tensor-parallel join buffers",
        ) {
            Ok(arena) => arena,
            Err(refused) => {
                return match self.ledger.release(refused.reservation) {
                    Ok(()) => Err(refused.error),
                    Err(held) => Err(device_lost(
                        self.ctx.ordinal(),
                        format!("join reservation could not be released ({})", held.error),
                    )),
                };
            }
        };
        self.temp = Some(CollectiveTemp {
            source,
            arena: Some(arena),
            status: None,
            data: None,
            dummy: None,
            joined_output: None,
            local_source: None,
            module: None,
        });
        let boundary = self.held.boundary.as_mut().expect("step boundary is open");
        let output = boundary
            .arena
            .allocate(output_bytes, ALIGNMENT, "collective output")
            .map_err(|refused| refused.error)?;
        self.temp
            .as_mut()
            .expect("temporary is installed")
            .joined_output = Some(output);
        let temp = self.temp.as_mut().expect("temporary is installed");
        let arena = temp.arena.as_mut().expect("temporary arena is open");
        temp.status = Some(
            arena
                .allocate(JOIN_STATUS_BYTES, ALIGNMENT, "join status")
                .map_err(|refused| refused.error)?,
        );
        temp.data = Some(
            arena
                .allocate(collective_bytes, ALIGNMENT, "join collective")
                .map_err(|refused| refused.error)?,
        );
        temp.dummy = Some(
            arena
                .allocate(part_bytes, ALIGNMENT, "join dummy source")
                .map_err(|refused| refused.error)?,
        );
        if matches!(join, Join::Reduce { .. }) {
            // SAFETY: the image is this build's pinned nvcc output.
            let image =
                unsafe { TrustedImage::from_build_output(moxie_kernels::DENSE_GRAPH_FATBIN) }?;
            let module = Module::load(self.ctx, ModuleImage::Binary(image))?;
            let module = module.resolve_all(&[moxie_kernels::TP_F32_TO_BF16.to_string()])?;
            self.temp.as_mut().expect("temporary is installed").module = Some(module);
        }
        Ok(())
    }

    fn copy_join(
        &mut self,
        join: Join,
        declaration: crate::GatherDeclaration,
        fail_after_prepare: bool,
    ) -> Result<()> {
        let stream = self.stream;
        let mut failed = fail_after_prepare;
        let elements = declaration
            .rows
            .checked_mul(declaration.columns)
            .ok_or_else(|| invalid("collective", "join element count overflowed"))?;
        let element_bytes = match declaration.precision {
            Precision::Bf16 => 2,
            Precision::F32 => 4,
            _ => return Err(invalid("collective", "join precision is unsupported")),
        };
        let part_bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid("collective", "join source size overflowed"))?;
        let source_value = self
            .temp
            .as_ref()
            .ok_or_else(|| invalid("collective", "join was not prepared"))?
            .source;
        if !failed {
            let source = self
                .held
                .plans
                .last_mut()
                .ok_or_else(|| invalid("collective", "join has no live stage plan"))?
                .take_range_for_value(source_value);
            match source {
                Ok((_, source)) => {
                    if source.bytes() < part_bytes {
                        failed = true;
                    }
                    self.temp
                        .as_mut()
                        .expect("join temporary remains")
                        .local_source = Some(source);
                }
                Err(_) => failed = true,
            }
        }
        let temp = self.temp.as_mut().expect("join temporary remains");
        let dummy = temp.dummy.as_ref().expect("dummy source is allocated");
        let source_address = if failed {
            dummy.device_address()?
        } else {
            match temp
                .local_source
                .as_ref()
                .expect("successful join retains its local source")
                .device_address()
            {
                Ok(address) => address,
                Err(_) => {
                    failed = true;
                    dummy.device_address()?
                }
            }
        };
        let status_address = temp
            .status
            .as_ref()
            .expect("join status is allocated")
            .device_address()?;
        let data_address = temp
            .data
            .as_ref()
            .expect("join data is allocated")
            .device_address()?;
        let communicator = self.communicator.as_ref().ok_or_else(|| {
            device_lost(
                self.ctx.ordinal(),
                "join has no live NCCL communicator".into(),
            )
        })?;
        // SAFETY: the status word is an admitted device range retained through
        // JoinDrain, and this rank's stream owns the ordered write.
        unsafe { communicator.set_u32_async(status_address, u32::from(failed), stream)? };
        communicator.group_start()?;
        let enqueued = (|| {
            // SAFETY: all ranges are admitted, large enough for these counts,
            // and retained through the paired JoinDrain.
            unsafe {
                communicator.all_reduce_u32_max(status_address, status_address, 1, stream)?;
                match join {
                    Join::Reduce { .. } => communicator.all_reduce_f32_sum(
                        source_address,
                        data_address,
                        usize::try_from(elements).map_err(|_| {
                            invalid("collective", "join element count is too large")
                        })?,
                        stream,
                    ),
                    Join::Gather { .. } => communicator.all_gather(
                        source_address,
                        data_address,
                        usize::try_from(elements).map_err(|_| {
                            invalid("collective", "join element count is too large")
                        })?,
                        match declaration.precision {
                            Precision::Bf16 => NcclDataType::Bf16,
                            Precision::F32 => NcclDataType::F32,
                            _ => unreachable!("prepare_join validates the join precision"),
                        },
                        stream,
                    ),
                }
            }
        })();
        let ended = communicator.group_end();
        enqueued?;
        ended?;
        Ok(())
    }

    fn drain_join(
        &mut self,
        join: Join,
        _declaration: crate::GatherDeclaration,
        deadline: Instant,
    ) -> Result<()> {
        self.drain(deadline)?;
        let status = self
            .temp
            .as_ref()
            .and_then(|temp| temp.status.as_ref())
            .ok_or_else(|| invalid("collective", "join status is absent"))?;
        let mut status_bytes = [0u8; 4];
        status.copy_to_host(&mut status_bytes).map_err(|error| {
            device_lost(
                self.ctx.ordinal(),
                format!("join status could not be observed ({error})"),
            )
        })?;
        let failed = u32::from_le_bytes(status_bytes) != 0;
        if !failed {
            let stream = self.stream;
            let temp = self.temp.as_ref().expect("join temporary remains");
            match join {
                Join::Reduce { .. } => {
                    let module = temp.module.as_ref().expect("conversion kernel is loaded");
                    let input = temp
                        .data
                        .as_ref()
                        .expect("reduce output is allocated")
                        .device_address()?;
                    let mut input = input;
                    let mut output = temp
                        .joined_output
                        .as_ref()
                        .expect("joined output is allocated")
                        .device_address()?;
                    let mut count = _declaration
                        .rows
                        .checked_mul(_declaration.columns)
                        .ok_or_else(|| invalid("collective", "join element count overflowed"))?;
                    let mut params: [*mut c_void; 3] = [
                        (&raw mut input).cast(),
                        (&raw mut output).cast(),
                        (&raw mut count).cast(),
                    ];
                    let grid = u32::try_from(count.div_ceil(256))
                        .map_err(|_| invalid("collective", "reduce grid is too large"))?;
                    // SAFETY: the one-input conversion ABI matches the admitted
                    // FP32 sum and BF16 output ranges, retained through JoinDrain.
                    unsafe {
                        module.launch_async(
                            0,
                            stream,
                            (grid, 1, 1),
                            (256, 1, 1),
                            0,
                            &mut params,
                        )?;
                    }
                }
                Join::Gather { .. } => {
                    let gathered = temp.data.as_ref().expect("gather output is allocated");
                    let output = temp
                        .joined_output
                        .as_ref()
                        .expect("joined output is allocated");
                    let element_bytes = match _declaration.precision {
                        Precision::Bf16 => 2,
                        Precision::F32 => 4,
                        _ => return Err(invalid("collective", "join precision is unsupported")),
                    };
                    let source_row_bytes = _declaration
                        .columns
                        .checked_mul(element_bytes)
                        .ok_or_else(|| invalid("collective", "gather row size overflowed"))?;
                    let output_row_bytes = source_row_bytes
                        .checked_mul(2)
                        .ok_or_else(|| invalid("collective", "gather output row overflowed"))?;
                    let gathered_rank_bytes = _declaration
                        .rows
                        .checked_mul(source_row_bytes)
                        .ok_or_else(|| invalid("collective", "gather source size overflowed"))?;
                    for row in 0.._declaration.rows {
                        let source_row = row
                            .checked_mul(source_row_bytes)
                            .ok_or_else(|| invalid("collective", "gather row offset overflowed"))?;
                        let output_row = row.checked_mul(output_row_bytes).ok_or_else(|| {
                            invalid("collective", "gather output offset overflowed")
                        })?;
                        let second_output =
                            output_row.checked_add(source_row_bytes).ok_or_else(|| {
                                invalid("collective", "gather output offset overflowed")
                            })?;
                        let second_source =
                            gathered_rank_bytes.checked_add(source_row).ok_or_else(|| {
                                invalid("collective", "gather source offset overflowed")
                            })?;
                        // SAFETY: all-gather is Ready and its stream completion
                        // was observed; all ranges remain admitted through the
                        // second drain below.
                        unsafe {
                            output.copy_from_device_async_at(
                                output_row,
                                gathered,
                                source_row,
                                source_row_bytes,
                                stream,
                            )?;
                            output.copy_from_device_async_at(
                                second_output,
                                gathered,
                                second_source,
                                source_row_bytes,
                                stream,
                            )?;
                        }
                    }
                }
            }
            self.drain(deadline)?;
        }
        let mut temp = self.temp.take().expect("join temporary remains");
        if let Some(source) = temp.local_source.take() {
            release_join_source(&mut self.held.plans, source, self.ctx.ordinal())?;
        }
        let output_id = match join {
            Join::Gather { output } | Join::Reduce { output } => output,
        };
        let output = temp
            .joined_output
            .take()
            .expect("joined output is allocated");
        let boundary = self.held.boundary.as_mut().expect("step boundary is open");
        if failed {
            boundary.arena.release(output).map_err(|refused| {
                device_lost(
                    self.ctx.ordinal(),
                    format!(
                        "failed join output could not be released ({})",
                        refused.error
                    ),
                )
            })?;
        } else {
            boundary.values.insert(output_id, output);
        }
        for index in 0..3 {
            let range = match index {
                0 => temp.status.take(),
                1 => temp.data.take(),
                _ => temp.dummy.take(),
            };
            if let Some(range) = range {
                let result = temp
                    .arena
                    .as_mut()
                    .expect("join arena is open")
                    .release(range);
                if let Err(refused) = result {
                    match index {
                        0 => temp.status = Some(refused.range),
                        1 => temp.data = Some(refused.range),
                        _ => temp.dummy = Some(refused.range),
                    }
                    self.temp = Some(temp);
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        format!("join temporary could not be released ({})", refused.error),
                    ));
                }
            }
        }
        temp.module.take();
        if let Some(arena) = temp.arena.take()
            && let Err(refused) = arena.close(&mut self.ledger)
        {
            temp.arena = Some(refused.arena);
            self.temp = Some(temp);
            return Err(device_lost(
                self.ctx.ordinal(),
                format!("join arena could not close ({})", refused.error),
            ));
        }
        close_plans(self.ctx, &mut self.ledger, &mut self.held.plans)?;
        if failed {
            return Err(invalid(
                "collective",
                "a rank failed after join preparation",
            ));
        }
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        let Held {
            mut plans,
            leases,
            mut boundary,
        } = std::mem::take(&mut self.held);
        plans.extend(
            leases
                .into_iter()
                .filter_map(|lease| lease.reclaim_drained().plan),
        );
        for run in &mut self.runs {
            match run.reclaim_drained() {
                Ok(Some((query, output))) => {
                    for range in [query, output] {
                        return_worker_range(self.ctx, &mut plans, range)?;
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    self.held.plans = plans;
                    self.held.boundary = boundary;
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        format!("attention run cannot be reclaimed ({error})"),
                    ));
                }
            }
        }
        if let Some(temp) = self.temp.take() {
            let mut temp = temp;
            if let Some(source) = temp.local_source.take() {
                return_worker_range(self.ctx, &mut plans, source)?;
            }
            if let Some(output) = temp.joined_output.take() {
                let released = boundary
                    .as_mut()
                    .expect("step boundary is open")
                    .arena
                    .release(output);
                if let Err(refused) = released {
                    temp.joined_output = Some(refused.range);
                    self.temp = Some(temp);
                    self.held.plans = plans;
                    self.held.boundary = boundary;
                    return Err(device_lost(
                        self.ctx.ordinal(),
                        format!(
                            "collective output could not be released ({})",
                            refused.error
                        ),
                    ));
                }
            }
            for index in 0..3 {
                let range = match index {
                    0 => temp.status.take(),
                    1 => temp.data.take(),
                    _ => temp.dummy.take(),
                };
                if let Some(range) = range {
                    let result = temp
                        .arena
                        .as_mut()
                        .expect("temporary range has an arena")
                        .release(range);
                    if let Err(refused) = result {
                        match index {
                            0 => temp.status = Some(refused.range),
                            1 => temp.data = Some(refused.range),
                            _ => temp.dummy = Some(refused.range),
                        }
                        self.temp = Some(temp);
                        self.held.plans = plans;
                        self.held.boundary = boundary;
                        return Err(device_lost(
                            self.ctx.ordinal(),
                            format!("collective range could not be released ({})", refused.error),
                        ));
                    }
                }
            }
            temp.module.take();
            if let Some(arena) = temp.arena.take()
                && let Err(refused) = arena.close(&mut self.ledger)
            {
                temp.arena = Some(refused.arena);
                self.temp = Some(temp);
                self.held.plans = plans;
                self.held.boundary = boundary;
                return Err(device_lost(
                    self.ctx.ordinal(),
                    format!("collective arena could not close ({})", refused.error),
                ));
            }
        }
        close_plans(self.ctx, &mut self.ledger, &mut plans)?;
        if let Some(boundary) = boundary {
            close_boundary_worker(self.ctx, &mut self.ledger, boundary)?;
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

    fn close_runs(&mut self) -> Result<()> {
        if self.transaction.is_some() || self.held.boundary.is_some() {
            return Err(device_lost(
                self.ctx.ordinal(),
                "worker shutdown found an unsettled step".into(),
            ));
        }
        while let Some(run) = self.runs.pop() {
            if let Err(refused) = run.close(&mut self.ledger) {
                let error = device_lost(
                    self.ctx.ordinal(),
                    format!("attention run could not close ({})", refused.error),
                );
                std::mem::forget(refused.run);
                return Err(error);
            }
        }
        if !self.ledger.outstanding().is_empty() {
            if self.ledger.outstanding().len() == usize::from(self.nccl_reservation.is_some()) {
                return Ok(());
            }
            return Err(device_lost(
                self.ctx.ordinal(),
                "worker shutdown left charged reservations".into(),
            ));
        }
        Ok(())
    }
}

fn return_worker_range<'ctx>(
    context: &RankContext,
    plans: &mut [SelectedReservedPlan<'ctx>],
    mut range: crate::DeviceRange<'ctx>,
) -> Result<()> {
    for plan in plans {
        match plan.release_reclaimed(range) {
            Ok(()) => return Ok(()),
            Err(refused) => range = refused.range,
        }
    }
    Err(device_lost(
        context.ordinal(),
        format!(
            "reclaimed attention range of {} bytes has no owning plan",
            range.bytes()
        ),
    ))
}

fn release_join_source<'ctx>(
    plans: &mut [SelectedReservedPlan<'ctx>],
    source: crate::DeviceRange<'ctx>,
    ordinal: u32,
) -> Result<()> {
    let plan = plans
        .last_mut()
        .expect("source plan remains held through the join");
    if let Err(refused) = plan.release_reclaimed(source) {
        core::mem::forget(refused.range);
        return Err(device_lost(
            ordinal,
            format!(
                "peer source range could not return to plan ({})",
                refused.error
            ),
        ));
    }
    Ok(())
}

fn close_plans<'ctx>(
    context: &RankContext,
    ledger: &mut Ledger,
    plans: &mut Vec<SelectedReservedPlan<'ctx>>,
) -> Result<()> {
    while let Some(plan) = plans.pop() {
        plan.close(ledger).map_err(|refused| {
            device_lost(
                context.ordinal(),
                format!("stage plan could not close ({})", refused.error),
            )
        })?;
    }
    Ok(())
}

fn close_boundary_worker(
    context: &RankContext,
    ledger: &mut Ledger,
    boundary: Boundary<'_>,
) -> Result<()> {
    let Boundary { mut arena, values } = boundary;
    for (_, range) in values {
        arena.release(range).map_err(|refused| {
            device_lost(
                context.ordinal(),
                format!("boundary value could not be released ({})", refused.error),
            )
        })?;
    }
    arena.close(ledger).map_err(|refused| {
        device_lost(
            context.ordinal(),
            format!("boundary arena could not close ({})", refused.error),
        )
    })
}

pub(crate) fn device_lost(device: u32, detail: String) -> Error {
    Error::DeviceLost { device, detail }
}

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
        // SAFETY: the planned input extent determines each row's byte range.
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
    // SAFETY: the boundary retains the destination through stream completion.
    unsafe { range.copy_from_device_async_at(0, plan.range_for_value(local)?, 0, bytes, stream) }
}

fn value_bytes(graph: &Graph, value: ValueId, rows: u64, element: u64) -> Result<u64> {
    let mut table = SymbolTable::new();
    table.bind(graph.rows_symbol(), rows);
    let spec = graph
        .spec(value)
        .ok_or_else(|| invalid("boundary", "a value has no tensor spec"))?;
    spec.shape.iter().try_fold(element, |bytes, dim| {
        let extent = dim
            .eval(&table)
            .map_err(|_| invalid("boundary", "unresolved dimension"))?;
        bytes
            .checked_mul(extent)
            .ok_or_else(|| invalid("boundary", "boundary extent overflowed"))
    })
}

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

fn invalid(field: &'static str, detail: &str) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

#[cfg(feature = "paged-attention-test-hooks")]
#[cfg(test)]
mod close_tests {
    use super::*;

    #[test]
    fn quarantined_run_close_keeps_unprocessed_runs() {
        let ordinal = 0;
        let context = RankContext::acquire(RankId(1), ordinal).expect("rank context");
        let capability = moxie_cuda::query_device(ordinal).expect("device capability");
        let stream = Stream::new(&context).expect("stream");
        let device =
            CapacitySnapshot::measured(&context.measure().expect("device measure"), 1 << 20)
                .expect("device capacity");
        let host =
            CapacitySnapshot::measured_host(&moxie_host::read().expect("host measure"), 1 << 20)
                .expect("host capacity");
        let mut ledger = Ledger::new([device, host]).expect("ledger");
        let geometry = crate::paged_attention::PageGeometry {
            kv_heads: 2,
            head_dim: 64,
            page_tokens: 8,
            pages: 1,
        };
        let layer = crate::paged_attention::AttentionLayer {
            geometry,
            heads: 4,
            scale: moxie_plan::reciprocal_sqrt_scale(64),
            visibility: moxie_graph::Visibility::Causal,
        };
        let probe = crate::paged_attention::PagedAttentionLaunch::new(layer, 1, 0, 0, 1)
            .expect("kernel probe");
        let catalogue = moxie_kernels::paged_attention_catalogue();
        let descriptor = crate::select_paged_attention_kernel(&catalogue, &capability, &probe)
            .expect("paged attention descriptor");
        let mut state = DeviceKvSequence::new(KvGeometry {
            layers: vec![moxie_state::LayerKv {
                kv_heads: 2,
                key_dim: 64,
                value_dim: 64,
                retention: moxie_state::Retention::All,
            }],
            precision: moxie_types::Precision::Bf16,
            page_tokens: 8,
            max_tokens: 8,
            tentative_rows: 8,
        })
        .expect("device state");
        let mut quarantined = PagedAttentionRun::admit_for_sequence(
            &mut ledger,
            &context,
            descriptor.clone(),
            geometry,
            4,
            8,
            9,
            Staging::TwoBlock,
        )
        .map_err(|refused| refused.error)
        .expect("quarantine test run");
        let normal = PagedAttentionRun::admit(
            &mut ledger,
            &context,
            descriptor,
            geometry,
            4,
            8,
            Staging::DeviceHandles,
        )
        .map_err(|refused| refused.error)
        .expect("remaining run");
        let transaction = state.begin().expect("append transaction");
        let row_bytes = 2 * 64 * 2;
        crate::paged_attention::device::append_paged_layer(
            &mut state,
            transaction,
            0,
            8,
            &mut quarantined,
            &stream,
            crate::paged_attention::device::PagedKvRows {
                keys: vec![0; 8 * row_bytes],
                values: vec![0; 8 * row_bytes],
            },
        )
        .map_err(|refused| refused.error)
        .expect("publish resident rows");
        crate::paged_attention::device::commit_paged_layer(
            &mut state,
            transaction,
            8,
            &mut quarantined,
            &stream,
        )
        .expect("commit resident rows");
        quarantined.inject_staging_failure();
        let launch =
            crate::paged_attention::PagedAttentionLaunch::two_block_stream(layer, 1, 15, 0, 8)
                .expect("two-block launch");
        let refused = quarantined
            .attend_two_block(
                &stream,
                &launch,
                vec![0; 4 * 64 * 2],
                vec![0; 8 * row_bytes],
                vec![0; 8 * row_bytes],
            )
            .expect_err("injected staging failure was accepted");
        assert!(refused.retained_source());

        let mut worker = WorkerState {
            ctx: &context,
            capability,
            stream: &stream,
            ledger,
            state,
            runs: vec![normal, quarantined],
            held: Held::default(),
            transaction: None,
            temp: None,
            communicator: None,
            nccl_reservation: None,
            preserve_communicator_on_loss: false,
        };
        assert!(matches!(worker.close_runs(), Err(Error::DeviceLost { .. })));
        assert_eq!(worker.runs.len(), 1);
        worker
            .runs
            .pop()
            .expect("unprocessed run remains held")
            .close(&mut worker.ledger)
            .unwrap();
        assert!(!worker.ledger.outstanding().is_empty());

        std::mem::forget(worker);
        std::mem::forget(stream);
        std::mem::forget(context);
    }
}
