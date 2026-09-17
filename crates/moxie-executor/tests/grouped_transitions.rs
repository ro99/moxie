//! The run's product, enumerated -- not sampled.
//!
//! Task 0020 established the method and AGENTS.md carries it: a state machine's
//! tests should enumerate its product, the sweep should print what it covered,
//! and its strength should be measured by mutation rather than asserted. Task
//! 0021 applied that to the **planner** and not to the **run**, and both rounds
//! of independent review found defects the run's own product would have
//! enumerated. Round one's were checks missing on a neighbouring path; round
//! two's were the same path one step later -- quarantine set correctly at the
//! moment of failure and ignored by `close`, a failure made terminal for a group
//! and not for a load.
//!
//! So: every combination of candidate, failure point, cancellation and close
//! ordering, with [`GroupedRun::check_invariants`] called after **every**
//! operation and the authority's own lease count reconciled against the run's
//! after every one of them too. That second check is external to the run and is
//! the one that catches a lease the run thinks it gave back.

use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_executor::grouped::{ExpertRoles, GroupedRun, Progress};
use moxie_executor::residency::{ChunkSource, ShardSource};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::to_bf16_bits;
use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger,
    ResidencyAuthority, ResidencyRequest, TurnId, UseClass,
};
use moxie_plan::expert::{
    Candidate, ExpertBudget, ExpertKernels, ExpertPlan, ExpertPolicy, ResidentChunks,
    compile_experts,
};
use moxie_storage::Shard;
use moxie_types::{DeviceUuid, Error, Result, Scope, StrategyControl};

/// `3 * intermediate * hidden * 2` must be a whole cache alignment, so these
/// are not arbitrary: 1,536 bytes per expert.
const BUS: &str = "0000:82:00.0";

/// Which family's routing shape a case is swept at.
///
/// Task 0022's second consumer goes **through** this sweep rather than beside
/// it. The axes it changes are the ones this product already enumerates: a
/// different top-k changes the slot arithmetic every group writes into, and a
/// different expert count changes how often a restricted cache has to evict,
/// which is what the queue's backpressure path exists for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Profile {
    /// The shape task 0021 gated: five distinct experts over four rows of two.
    GemmaLike,
    /// Laguna's shape at fixture scale: **eight** distinct experts over four
    /// rows of **three**, with a SwiGLU gate and a routed scaling factor.
    LagunaLike,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Geometry {
    hidden: u64,
    intermediate: u64,
    experts: u64,
    top_k: u64,
    rows: u64,
    activation: ExpertActivation,
    output_scale: f32,
    route: &'static [u32],
    /// A stable name for the shard this profile's weights are written to.
    shard: &'static str,
}

impl Geometry {
    /// `3 * intermediate * hidden * 2` must be a whole cache alignment, so
    /// these are not arbitrary: 1,536 bytes per expert in both profiles.
    const fn chunk(&self) -> u64 {
        3 * self.intermediate * self.hidden * 2
    }

    /// How many groups a plan over this route has: one per **distinct** expert.
    ///
    /// Computed rather than written down, because it is the number the run's
    /// completeness is judged against and the two profiles disagree about it --
    /// five against eight.
    fn groups(&self) -> usize {
        let mut seen: Vec<u32> = self.route.to_vec();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }
}

impl Profile {
    const ALL: [Profile; 2] = [Profile::GemmaLike, Profile::LagunaLike];

    const fn geometry(self) -> Geometry {
        match self {
            Profile::GemmaLike => Geometry {
                hidden: 32,
                intermediate: 8,
                experts: 6,
                top_k: 2,
                rows: 4,
                activation: ExpertActivation::GeGlu,
                output_scale: 1.0,
                // Five distinct experts over four rows, so a queue of one and a
                // queue of four behave differently.
                route: &[0, 1, 0, 2, 1, 3, 0, 4],
                shard: "gemma-like",
            },
            Profile::LagunaLike => Geometry {
                hidden: 32,
                intermediate: 8,
                experts: 8,
                top_k: 3,
                rows: 4,
                activation: ExpertActivation::SwiGlu,
                output_scale: 2.5,
                // Eight distinct experts over four rows of three: half again as
                // many chunks to admit, into the same cache.
                route: &[5, 1, 0, 7, 5, 2, 5, 3, 1, 6, 4, 5],
                shard: "laguna-like",
            },
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Profile::GemmaLike => "gemma-like",
            Profile::LagunaLike => "laguna-like",
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len * 2);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
        out.extend_from_slice(&to_bf16_bits(unit).to_le_bytes());
    }
    out
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("moxie-transitions-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_shard(dir: &Path, g: Geometry) -> PathBuf {
    let (experts, hidden, intermediate) = (g.experts, g.hidden, g.intermediate);
    let gate_up = bytes(0xa1, (experts * 2 * intermediate * hidden) as usize);
    let down = bytes(0xb2, (experts * hidden * intermediate) as usize);
    let split = gate_up.len();
    let end = split + down.len();
    let header = format!(
        "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{experts},{},{hidden}],\
         \"data_offsets\":[0,{split}]}},\
         \"experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{experts},{hidden},{intermediate}],\
         \"data_offsets\":[{split},{end}]}}}}",
        2 * intermediate
    );
    let path = dir.join(format!("{}.safetensors", g.shard));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&gate_up).unwrap();
    f.write_all(&down).unwrap();
    path
}

fn uuid() -> DeviceUuid {
    DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap()
}

fn roles(g: Geometry) -> ExpertRoles {
    ExpertRoles {
        per_expert: None,
        // A distinct artifact per profile: two profiles sharing one identity
        // would let one profile's chunk satisfy the other's demand, which is a
        // residency answer no route asked for.
        artifact: ArtifactId::new(format!("transitions-fixture-v1-{}", g.shard)).unwrap(),
        gate_up_role: "experts_gate_up".into(),
        down_role: "experts_down".into(),
        format_version: 1,
    }
}

fn shard_source(path: &Path, g: Geometry) -> ShardSource {
    ShardSource::new(roles(g).artifact, vec![Shard::open(path).unwrap()])
        .role("experts_gate_up", 0, "experts.gate_up_proj")
        .unwrap()
        .role("experts_down", 0, "experts.down_proj")
        .unwrap()
}

/// A source that fails the `nth` distinct chunk it is asked for, once.
struct Fallible {
    inner: ShardSource,
    seen: std::collections::BTreeSet<String>,
    fail_at: Option<usize>,
}

impl ChunkSource for Fallible {
    fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> Result<()> {
        self.seen.insert(format!("{chunk}"));
        if self.fail_at == Some(self.seen.len()) {
            self.fail_at = None;
            return Err(Error::InvalidArtifact {
                detail: "an injected short read".into(),
            });
        }
        self.inner.read_chunk(chunk, into)
    }
}

/// Kernel launches one device group submits: one per symbol of the selected
/// descriptor, which for the expert kernel is a projection and a reduction.
const KERNELS_PER_GROUP: u32 = 2;

/// A lane that fails on demand, at a chosen point, with a chosen submission
/// state.
#[derive(Debug)]
struct Lane {
    fail_load: Option<bool>,
    fail_launch: Option<bool>,
    quarantined: bool,
}

impl moxie_executor::grouped::ExpertDeviceLane for Lane {
    fn load_activations(
        &mut self,
        _x: &[u8],
    ) -> std::result::Result<(), moxie_executor::grouped::LaunchRefused> {
        match self.fail_load.take() {
            None => Ok(()),
            Some(unknown) => {
                self.quarantined |= unknown;
                Err(moxie_executor::grouped::LaunchRefused {
                    error: Error::DeviceLost {
                        device: 0,
                        detail: "injected activation copy failure".into(),
                    },
                    submission_unknown: unknown,
                    // An activation copy is not a kernel launch.
                    submitted: 0,
                })
            }
        }
    }

    fn perform_upload(
        &mut self,
        authority: &mut ResidencyAuthority,
        order: &moxie_memory::WorkOrder,
    ) -> Result<()> {
        authority.complete_upload(order.ticket(), moxie_memory::Outcome::Completed)
    }

    fn run_group(
        &mut self,
        _authority: &ResidencyAuthority,
        _group: &moxie_plan::expert::ExpertGroup,
        _gate_up: &moxie_memory::ResidencyLease,
        _down: &moxie_memory::ResidencyLease,
        _staging: moxie_executor::grouped::ExpertStaging<'_>,
        _host_slots: &mut [u8],
    ) -> std::result::Result<u32, moxie_executor::grouped::LaunchRefused> {
        match self.fail_launch.take() {
            // What the production lane submits: one launch per symbol of the
            // descriptor it selected. A double that reported a different number
            // would make the launch equality mean something else here than it
            // means on a card.
            None => Ok(KERNELS_PER_GROUP),
            Some(unknown) => {
                self.quarantined |= unknown;
                Err(moxie_executor::grouped::LaunchRefused {
                    error: Error::DeviceLost {
                        device: 0,
                        detail: "injected launch failure".into(),
                    },
                    submission_unknown: unknown,
                    // Failed between the two symbols: the first is on the
                    // device and the count has to say so.
                    submitted: 1,
                })
            }
        }
    }

    fn close(&mut self, _ledger: &mut Ledger) -> Result<()> {
        Ok(())
    }

    fn is_quarantined(&self) -> bool {
        self.quarantined
    }
}

// ---------------------------------------------------------------------------
// The product
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    /// The cache cannot admit the next expert: recoverable, and the only thing
    /// that is.
    AcquireCapacity,
    /// A read that fails: not recoverable, however full the queue is.
    AcquireRead,
    /// A chunk another ticket is still reading, which this executor cannot wait
    /// for.
    Readiness,
    LoadPlain,
    LoadUnknown,
    LaunchPlain,
    LaunchUnknown,
}

impl Failure {
    const ALL: [Failure; 8] = [
        Failure::None,
        Failure::AcquireCapacity,
        Failure::AcquireRead,
        Failure::Readiness,
        Failure::LoadPlain,
        Failure::LoadUnknown,
        Failure::LaunchPlain,
        Failure::LaunchUnknown,
    ];

    const fn needs_lane(self) -> bool {
        matches!(
            self,
            Failure::LoadPlain
                | Failure::LoadUnknown
                | Failure::LaunchPlain
                | Failure::LaunchUnknown
        )
    }

    const fn name(self) -> &'static str {
        match self {
            Failure::None => "none",
            Failure::AcquireCapacity => "acquire-capacity",
            Failure::AcquireRead => "acquire-read",
            Failure::Readiness => "readiness",
            Failure::LoadPlain => "load-plain",
            Failure::LoadUnknown => "load-unknown",
            Failure::LaunchPlain => "launch-plain",
            Failure::LaunchUnknown => "launch-unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cancel {
    Never,
    BeforeStart,
    AfterFirstStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Case {
    profile: Profile,
    candidate: Candidate,
    failure: Failure,
    cancel: Cancel,
    /// Close as soon as the run stops, or run to the end first.
    close_early: bool,
    queue_capacity: u32,
}

fn plan_for(case: Case) -> ExpertPlan {
    let g = case.profile.geometry();
    let catalogue =
        moxie_types::KernelCatalogue::new(vec![moxie_types::SemanticKernelDescriptor {
            id: moxie_types::KernelId("transitions-expert".into()),
            abi_version: moxie_plan::expert::EXPERT_ABI_VERSION,
            operation: moxie_types::SemanticKernelOp::ExpertMlp(match g.activation {
                ExpertActivation::GeGlu => moxie_types::GateTransform::GeluTanh,
                ExpertActivation::SwiGlu => moxie_types::GateTransform::Silu,
            }),
            inputs: vec![
                moxie_types::KernelOperand::Activation(moxie_types::ActivationPrecision::expect(
                    moxie_types::Precision::Bf16,
                )),
                moxie_types::KernelOperand::RouteIndex,
                moxie_types::KernelOperand::Weight(moxie_types::WeightPrecision::expect(
                    moxie_types::Precision::Bf16,
                )),
                moxie_types::KernelOperand::Weight(moxie_types::WeightPrecision::expect(
                    moxie_types::Precision::Bf16,
                )),
            ],
            output: moxie_types::ActivationPrecision::expect(moxie_types::Precision::Bf16),
            accumulation: moxie_types::AccumulationPolicy::Bf16InF32Acc,
            rounding: moxie_types::RoundingProfile::FinalBf16Rne,
            layout: moxie_types::TensorLayout::ContiguousRowMajorV1,
            shape: moxie_types::KernelShapeBounds {
                max_rows: 1024,
                max_input: 1024,
                max_output: 1024,
            },
            sm: moxie_types::SmVersion::SM86,
            workspace: moxie_types::WorkspaceExpression::RowsTimesIntermediateF32,
            image_sha256: [9; 32],
            symbols: vec![
                moxie_types::KernelSymbol("project".into()),
                moxie_types::KernelSymbol("down".into()),
            ],
        }])
        .unwrap();
    let capability = moxie_types::DeviceCapability {
        ordinal: 1,
        uuid: uuid(),
        name: "fixture".into(),
        compute_major: 8,
        compute_minor: 6,
        total_memory_bytes: 24 << 30,
        multiprocessor_count: 82,
        pci_bus_id: BUS.into(),
        peer_access: Vec::new(),
    };
    let on_device = case.candidate == Candidate::Device;
    let budget = ExpertBudget {
        device: uuid(),
        device_pci_bus_id: BUS.into(),
        device_cache_cap_bytes: if on_device { 64 * g.chunk() } else { 0 },
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: u64::MAX,
        device_arena_free_bytes: if on_device { 1 << 20 } else { 0 },
        host_workspace_bytes: 1 << 20,
        host_buffer_bytes: 1 << 20,
        host_cache_cap_bytes: 1 << 30,
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: u64::MAX,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident: ResidentChunks::none(),
    };
    let policy = ExpertPolicy {
        device: if on_device {
            StrategyControl::Required
        } else {
            StrategyControl::Off
        },
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        max_inflight_orders: case.queue_capacity,
        ..ExpertPolicy::default()
    };
    compile_experts(
        &OpParams::ExpertMlp {
            hidden: g.hidden,
            intermediate: g.intermediate,
            experts: g.experts,
            top_k: g.top_k,
            activation: g.activation,
        },
        &OpParams::Combine {
            hidden: g.hidden,
            top_k: g.top_k,
            order: CombineOrder::AscendingExpertId,
            output_scale: g.output_scale,
        },
        g.route,
        &budget,
        &policy,
        None,
        on_device.then_some(ExpertKernels {
            capability: &capability,
            catalogue: &catalogue,
        }),
    )
    .expect("a plan")
}

/// Everything the run and the authority must agree on, checked after **every**
/// operation.
fn agree(
    run: &GroupedRun<'_>,
    authority: &ResidencyAuthority,
    baseline: usize,
    where_: &str,
    case: Case,
) {
    run.check_invariants()
        .unwrap_or_else(|e| panic!("{case:?} after {where_}: {e}"));
    // The run's own account of what it holds, against the authority's. This is
    // the check that is **external** to the run, and it is the one that catches
    // a lease the run believes it gave back. `baseline` is what a case holds
    // outside the run, which only the readiness case does.
    let expected = 2 * run.queue().len() + run.withheld_leases();
    assert_eq!(
        authority.live_lease_count() - baseline,
        expected,
        "{case:?} after {where_}: the authority holds {} lease(s) beyond the baseline, the run \
         accounts for {expected}",
        authority.live_lease_count() - baseline
    );
}

#[test]
fn the_run_lifecycle_product_holds_its_invariants_after_every_operation() {
    let dir = scratch("sweep");

    let mut cases = 0usize;
    let mut completed = 0usize;
    let mut failed = 0usize;
    let mut cancelled = 0usize;
    let mut closed = 0usize;
    let mut close_refused = 0usize;
    let mut withheld_runs = 0usize;
    let mut drained = 0usize;
    let mut reached: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();

    let mut per_profile: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for profile in Profile::ALL {
        let g = profile.geometry();
        let path = write_shard(&dir, g);
        let x = bytes(0xc3, (g.rows * g.hidden) as usize);
        let coefficients = vec![1.0f32; (g.rows * g.top_k) as usize];
        for candidate in [Candidate::Host, Candidate::Device] {
            for failure in Failure::ALL {
                for cancel in [Cancel::Never, Cancel::BeforeStart, Cancel::AfterFirstStep] {
                    for close_early in [true, false] {
                        for queue_capacity in [1u32, 4] {
                            let case = Case {
                                profile,
                                candidate,
                                failure,
                                cancel,
                                close_early,
                                queue_capacity,
                            };
                            if failure.needs_lane() && candidate != Candidate::Device {
                                continue;
                            }
                            cases += 1;
                            *per_profile.entry(profile.name()).or_default() += 1;

                            let plan = plan_for(case);
                            assert!(plan.uses(candidate));
                            let host_cap = 1u64 << 25;
                            let mut ledger = Ledger::new([
                                CapacitySnapshot::new(Scope::Host, host_cap, 1 << 14).unwrap(),
                                CapacitySnapshot::new(Scope::Device(uuid()), 1 << 25, 1 << 14)
                                    .unwrap(),
                            ])
                            .unwrap();
                            // A cache of two experts when the case wants a capacity
                            // refusal, and room for all five otherwise.
                            // Capacity has two outcomes and the sweep needs both: a
                            // cache that holds one expert drains and recovers, and
                            // one that holds none fails with an empty queue. The
                            // second is the only shape in which a capacity refusal
                            // is fatal.
                            let cache = match (failure, case.queue_capacity) {
                                (Failure::AcquireCapacity, 1) => g.chunk() - 256,
                                (Failure::AcquireCapacity, _) => g.chunk(),
                                _ => 64 * g.chunk(),
                            };
                            // Small bounds: the control envelope is charged up
                            // front, and 4,096 placements would dominate a sweep
                            // whose plans hold five or eight experts.
                            let mut request = ResidencyRequest::new("sweep", cache)
                                .max_placements(64)
                                .max_leases(64)
                                .prefetch_queue_capacity(8);
                            if candidate == Candidate::Device {
                                request = request.device(uuid(), cache);
                            }
                            let mut authority =
                                ResidencyAuthority::open(&mut ledger, &request).unwrap();
                            let mut src = Fallible {
                                inner: shard_source(&path, g),
                                seen: std::collections::BTreeSet::new(),
                                fail_at: (failure == Failure::AcquireRead).then_some(3),
                            };

                            // A chunk left mid-read, for the readiness case.
                            let held = if failure == Failure::Readiness {
                                let (gate_up, _) = roles(g).chunks(0, plan.shape()).unwrap();
                                let destination = match candidate {
                                    Candidate::Host => Scope::Host,
                                    Candidate::Device => Scope::Device(uuid()),
                                };
                                match authority
                                    .acquire(AcquireRequest {
                                        chunk: &gate_up,
                                        destination,
                                        now: 0,
                                        deadline: u64::MAX,
                                        class: UseClass::demand(Content::Expert),
                                        turn: TurnId::new(9),
                                    })
                                    .unwrap()
                                {
                                    Acquired::Pending { lease, work, .. } => Some((lease, work)),
                                    Acquired::Ready(lease) => {
                                        Some((lease, moxie_memory::PendingWork::Coalesced))
                                    }
                                }
                            } else {
                                None
                            };

                            let baseline = authority.live_lease_count();
                            let mut run =
                                GroupedRun::admit(&mut ledger, plan, roles(g), None).unwrap();
                            if candidate == Candidate::Device {
                                run.install_lane(Box::new(Lane {
                                    fail_load: match failure {
                                        Failure::LoadPlain => Some(false),
                                        Failure::LoadUnknown => Some(true),
                                        _ => None,
                                    },
                                    fail_launch: match failure {
                                        Failure::LaunchPlain => Some(false),
                                        Failure::LaunchUnknown => Some(true),
                                        _ => None,
                                    },
                                    quarantined: false,
                                }))
                                .unwrap();
                            }

                            agree(&run, &authority, baseline, "admit", case);
                            // Before anything is loaded, there is nothing to run:
                            // a step here would compute over an unwritten buffer.
                            assert!(
                                run.step(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
                                    .is_err(),
                                "{case:?}: stepped before any activations were loaded"
                            );
                            agree(&run, &authority, baseline, "step-before-load", case);
                            if cancel == Cancel::BeforeStart {
                                run.cancel(&mut authority);
                                reached
                                    .entry("cancel-before-start")
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);
                            }
                            agree(&run, &authority, baseline, "cancel-before-start", case);

                            let loaded = run.load_activations(&x);
                            agree(&run, &authority, baseline, "load", case);
                            match (&loaded, failure) {
                                (Err(_), Failure::LoadPlain | Failure::LoadUnknown) => {
                                    reached
                                        .entry(failure.name())
                                        .and_modify(|n| *n += 1)
                                        .or_insert(1);
                                }
                                (Err(_), _) if cancel == Cancel::BeforeStart => {}
                                (Ok(()), Failure::LoadPlain | Failure::LoadUnknown) => {
                                    panic!("{case:?}: the injected load failure never happened")
                                }
                                _ => {}
                            }

                            let mut steps = 0usize;
                            if loaded.is_ok() {
                                loop {
                                    let progress = run.step(
                                        &mut authority,
                                        &mut src,
                                        TurnId::new(1),
                                        0,
                                        u64::MAX,
                                    );
                                    agree(&run, &authority, baseline, "step", case);
                                    match progress {
                                        Ok(Progress::Done) => break,
                                        Ok(Progress::Ran { .. }) => {
                                            steps += 1;
                                            if steps == 1 {
                                                // Mid-run, with groups left: a
                                                // reduction would be over slots
                                                // nobody has written, and a reload
                                                // would leave one row's slots
                                                // computed from two inputs. Both
                                                // must refuse, and neither is
                                                // reachable once the run has failed
                                                // or been cancelled -- which is why
                                                // it is checked **here**.
                                                assert!(
                                                    run.reduce(&coefficients).is_err(),
                                                    "{case:?}: reduced with groups left"
                                                );
                                                assert!(
                                                    run.load_activations(&x).is_err(),
                                                    "{case:?}: took new activations mid-run"
                                                );
                                                agree(&run, &authority, baseline, "mid-run", case);
                                                reached
                                                    .entry("mid-run-refusals")
                                                    .and_modify(|n| *n += 1)
                                                    .or_insert(1);
                                            }
                                            if cancel == Cancel::AfterFirstStep && steps == 1 {
                                                run.cancel(&mut authority);
                                                agree(
                                                    &run,
                                                    &authority,
                                                    baseline,
                                                    "cancel-mid",
                                                    case,
                                                );
                                                reached
                                                    .entry("cancel-after-step")
                                                    .and_modify(|n| *n += 1)
                                                    .or_insert(1);
                                                break;
                                            }
                                        }
                                        Err(_) => {
                                            reached
                                                .entry(failure.name())
                                                .and_modify(|n| *n += 1)
                                                .or_insert(1);
                                            break;
                                        }
                                    }
                                }
                            }
                            if run.stats().backpressure_drains > 0 {
                                drained += 1;
                                reached
                                    .entry("acquire-capacity-drained")
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);
                            }

                            if !close_early {
                                // Reducing is legal only from a complete, unfailed,
                                // uncancelled run; every other case must refuse.
                                let complete = run.failure().is_none()
                                    && !run.is_cancelled()
                                    && run.enqueued() == g.groups()
                                    && run.queue().is_empty();
                                let reduced = run.reduce(&coefficients).is_ok();
                                agree(&run, &authority, baseline, "reduce", case);
                                assert_eq!(
                                    reduced, complete,
                                    "{case:?}: reduce disagreed with the run's own state"
                                );
                            }

                            // Each axis value must produce the outcome it names.
                            // Counting that a failure "was reached" is not the same
                            // as checking it did what it is for, and three mutants
                            // survived on exactly that difference.
                            // A cancellation can pre-empt the injected failure, so
                            // the axis is only checked where it is the only thing
                            // that can have happened.
                            if cancel == Cancel::Never {
                                match failure {
                                    Failure::AcquireRead => {
                                        assert!(
                                            run.failure().is_some(),
                                            "{case:?}: a corrupt read did not fail the run"
                                        );
                                        assert_eq!(
                                            run.stats().backpressure_drains,
                                            0,
                                            "{case:?}: a corrupt read was counted as backpressure"
                                        );
                                    }
                                    Failure::Readiness => {
                                        assert!(
                                            run.failure().is_some(),
                                            "{case:?}: an unreadable chunk did not fail the run"
                                        );
                                    }
                                    Failure::LoadUnknown | Failure::LaunchUnknown => {
                                        assert!(
                                            run.is_withholding(),
                                            "{case:?}: an unknown submission withheld nothing"
                                        );
                                        // And the **host buffers** specifically: the
                                        // copy's source is one of them, and the lane
                                        // quarantining its own device ranges says
                                        // nothing about the pages being read from.
                                        assert!(
                                            run.buffers().is_quarantined(),
                                            "{case:?}: the source of an unconfirmed copy was not \
                                         quarantined"
                                        );
                                    }
                                    Failure::LoadPlain | Failure::LaunchPlain => {
                                        assert!(
                                            !run.is_withholding(),
                                            "{case:?}: a known submission failure withheld something"
                                        );
                                        assert!(run.failure().is_some());
                                    }
                                    Failure::None | Failure::AcquireCapacity => {}
                                }
                                // What a failed group **submitted** before it
                                // failed. A review found the run counting zero
                                // launches for a group that had already put one
                                // kernel on the device, and a mutation that
                                // discarded the lane's report then survived the
                                // whole battery, because nothing compared this
                                // number to anything.
                                match case.failure {
                                    Failure::LaunchPlain | Failure::LaunchUnknown => {
                                        let completed_groups = run.stats().device_groups
                                            * u64::from(KERNELS_PER_GROUP);
                                        assert_eq!(
                                            run.stats().launches,
                                            completed_groups + 1,
                                            "{case:?}: a group that failed between its two \
                                             symbols must still count the one it submitted"
                                        );
                                    }
                                    _ => {}
                                }
                            }
                            match run.state_name() {
                                "failed" => failed += 1,
                                "cancelled" => cancelled += 1,
                                _ => {}
                            }
                            if run.failure().is_none() && !run.is_cancelled() && steps == g.groups()
                            {
                                completed += 1;
                            }
                            if run.is_withholding() {
                                withheld_runs += 1;
                                reached
                                    .entry("withheld")
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);
                            }
                            if run.withheld_leases() > 0 {
                                reached
                                    .entry("withheld-leases")
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);
                            }
                            if run.buffers().is_quarantined() {
                                reached
                                    .entry("quarantined-buffers")
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);
                            }

                            // Closing **with work still queued** is its own
                            // transition and it used to be unreachable here: both
                            // branches drained first, so removing the guard changed
                            // nothing and all 144 combinations still passed. It is
                            // exercised now, and the refusal has to leave everything
                            // exactly where it was.
                            let mut run = run;
                            if !run.queue().is_empty() {
                                let queued = run.queue().len();
                                let held = authority.live_lease_count();
                                let charged = ledger.scope_committed(Scope::Host);
                                let refused = run
                                    .close(&mut ledger)
                                    .expect_err("a run still holding leases may not close");
                                assert!(
                                    format!("{refused}").contains("still hold residency leases"),
                                    "{case:?}: {refused}"
                                );
                                run = refused.run;
                                assert_eq!(
                                    run.queue().len(),
                                    queued,
                                    "{case:?}: a refused close changed the queue"
                                );
                                assert_eq!(
                                    authority.live_lease_count(),
                                    held,
                                    "{case:?}: a refused close moved a lease"
                                );
                                assert_eq!(
                                    ledger.scope_committed(Scope::Host),
                                    charged,
                                    "{case:?}: a refused close moved the charge"
                                );
                                agree(&run, &authority, baseline, "close-with-queued-work", case);
                                reached
                                    .entry("close-with-queued-work")
                                    .and_modify(|n| *n += 1)
                                    .or_insert(1);

                                // Then drain and let the real close proceed, which
                                // is what a caller must do.
                                run.cancel(&mut authority);
                                agree(&run, &authority, baseline, "cancel-drain", case);
                            }
                            agree(&run, &authority, baseline, "before close", case);
                            let withholding = run.is_withholding();
                            let host_before = ledger.scope_committed(Scope::Host);
                            match run.close(&mut ledger) {
                                Ok(()) => {
                                    closed += 1;
                                    assert!(
                                        !withholding,
                                        "{case:?}: a withholding run released its envelope"
                                    );
                                    assert!(ledger.scope_committed(Scope::Host) < host_before);
                                }
                                Err(refused) => {
                                    close_refused += 1;
                                    assert!(
                                        withholding,
                                        "{case:?}: close refused without withholding: {refused}"
                                    );
                                    // The charge stays with the memory.
                                    drop(refused);
                                    assert_eq!(
                                        ledger.scope_committed(Scope::Host),
                                        host_before,
                                        "{case:?}: a refused close moved the charge"
                                    );
                                }
                            }

                            if let Some((lease, work)) = held {
                                let _ = moxie_executor::residency::drain_reads(
                                    &mut authority,
                                    &mut src,
                                    work,
                                );
                                let _ = authority.release(lease);
                            }
                            authority.end_turn(TurnId::new(1));
                            authority.end_turn(TurnId::new(9));
                            let _ = authority.close(&mut ledger);
                        }
                    }
                }
            }
        }
    }

    println!(
        "run-lifecycle sweep: {cases} combination(s); {completed} completed, {failed} failed, \
         {cancelled} cancelled; {closed} closed, {close_refused} close(s) refused; \
         {withheld_runs} withholding, {drained} with backpressure"
    );
    for (profile, count) in &per_profile {
        println!("  profile {profile}: {count}");
    }
    let mut names: Vec<_> = reached.iter().collect();
    names.sort();
    for (what, count) in &names {
        println!("  reached {what}: {count}");
    }

    // Every axis value is exercised, and the counts above are the evidence
    // rather than this sentence.
    assert_eq!(cases, 2 * (2 * 4 * 3 * 2 * 2 + 4 * 3 * 2 * 2));
    assert_eq!(per_profile.len(), 2, "a profile was never swept");
    assert!(
        per_profile.values().all(|c| *c == cases / 2),
        "the profiles did not sweep the same product: {per_profile:?}"
    );
    for required in [
        "acquire-capacity",
        "acquire-capacity-drained",
        "acquire-read",
        "readiness",
        "load-plain",
        "load-unknown",
        "launch-plain",
        "launch-unknown",
        "cancel-before-start",
        "cancel-after-step",
        "withheld",
        "withheld-leases",
        "quarantined-buffers",
        "mid-run-refusals",
        "close-with-queued-work",
    ] {
        assert!(
            reached.get(required).copied().unwrap_or(0) > 0,
            "{required} was never reached; the sweep does not cover what it claims"
        );
    }
    assert!(completed > 0 && closed > 0 && close_refused > 0 && drained > 0);
    let _ = std::fs::remove_dir_all(&dir);
}
