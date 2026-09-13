//! The expert chooser's product, swept rather than sampled -- and the sweep
//! prints what it exercised.
//!
//! AGENTS.md records three task-0020 claims that were wrong the same way: a
//! property of the tests was asserted instead of measured. The two habits that
//! came out of that are applied here from the start. The space is a **product**
//! -- two controls, two plan-wide admissibilities, a per-expert cache
//! admissibility, the amortisation comparison, residency and the reduction
//! order -- so it is enumerated; and what the enumeration covered is **printed**
//! by the test rather than described in a record.
//!
//! Each case is checked against a second, independent statement of the decision
//! rule (`expected_choice` below). A sweep that only re-ran the implementation
//! would prove the implementation is deterministic, which was never in doubt.

use std::collections::BTreeMap;

use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_plan::expert::{
    Candidate, ExpertBudget, ExpertKernels, ExpertPolicy, Placement, RejectionReason,
    compile_experts,
};
use moxie_types::{
    AccumulationPolicy, ActivationPrecision, DeviceCapability, DeviceTier, DeviceUuid,
    GateTransform, HostPlacement, HostTier, KernelCatalogue, KernelId, KernelOperand,
    KernelShapeBounds, KernelSymbol, NumaNode, NumaNodeId, NumaTopology, Precision,
    RoundingProfile, SemanticKernelDescriptor, SemanticKernelOp, SmVersion, StrategyControl,
    TensorLayout, WeightPrecision, WorkspaceExpression,
};

const HIDDEN: u64 = 8;
const INTERMEDIATE: u64 = 4;
const EXPERTS: u64 = 6;
const TOP_K: u64 = 2;
/// `3 * intermediate * hidden * 2`.
const CHUNK: u64 = 3 * INTERMEDIATE * HIDDEN * 2;
const BUS: &str = "0000:82:00.0";

/// Four rows over five experts, with reuse counts 3, 2, 1, 1, 1 -- so one plan
/// contains both an expert whose transfer is amortised over three rows and
/// experts that pay it for one.
///
/// Every row selects its experts in **descending** id order. That is not
/// decoration: with ascending rows the two `CombineOrder` values agree, and
/// mutation testing showed the sweep passing while the plan ignored the
/// parameter entirely.
const ROUTE: [u32; 8] = [1, 0, 2, 0, 3, 1, 4, 0];
const ROWS: u64 = 4;

fn uuid() -> DeviceUuid {
    DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").expect("uuid")
}

fn mlp() -> OpParams {
    OpParams::ExpertMlp {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        activation: ExpertActivation::GeGlu,
    }
}

fn combine(order: CombineOrder) -> OpParams {
    OpParams::Combine {
        hidden: HIDDEN,
        top_k: TOP_K,
        order,
    }
}

fn topology() -> NumaTopology {
    NumaTopology::new(
        vec![
            NumaNode {
                id: NumaNodeId::new(0),
                cpus: vec![0, 1],
                total_bytes: 1 << 36,
                free_bytes: 1 << 35,
            },
            NumaNode {
                id: NumaNodeId::new(1),
                cpus: vec![2, 3],
                total_bytes: 1 << 36,
                free_bytes: 1 << 35,
            },
        ],
        vec![(BUS.to_string(), Some(NumaNodeId::new(1)))],
    )
    .expect("topology")
}

/// How generous the declared amortisation threshold is.
///
/// Three levels rather than two, because two cannot produce a **mixed** plan:
/// the route's reuse counts are 3, 2 and 1, so a threshold that admits one
/// admits all of them or none. `Partial` sits between `CHUNK/2` and `CHUNK/1`,
/// which is the only way this sweep reaches a plan with one expert on each
/// candidate -- the case the whole interface exists for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Amortisation {
    All,
    Partial,
    None,
}

impl Amortisation {
    fn limit(self) -> u64 {
        match self {
            Self::All => CHUNK,
            Self::Partial => CHUNK / 2 + 1,
            Self::None => 1,
        }
    }
}

/// Whether a kernel exists for this device, and the two ways it can be absent.
///
/// `WrongSm` is not the same case as `Absent`: a catalogue that has a descriptor
/// for the *other* architecture is what a real mismatch looks like, and a
/// selector that matched on operation alone would pass with `Absent` and fail
/// here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernels {
    Absent,
    Matching,
    WrongSm,
}

fn capability() -> DeviceCapability {
    DeviceCapability {
        ordinal: 1,
        uuid: uuid(),
        name: "fixture".into(),
        compute_major: 8,
        compute_minor: 6,
        total_memory_bytes: 24 << 30,
        multiprocessor_count: 82,
        pci_bus_id: BUS.into(),
        peer_access: Vec::new(),
    }
}

fn catalogue(sm: SmVersion) -> KernelCatalogue {
    KernelCatalogue::new(vec![SemanticKernelDescriptor {
        id: KernelId(format!("fixture-expert-{}", sm.name())),
        abi_version: 1,
        operation: SemanticKernelOp::ExpertMlp(GateTransform::GeluTanh),
        inputs: vec![
            KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
            KernelOperand::RouteIndex,
            KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
            KernelOperand::Weight(WeightPrecision::expect(Precision::Bf16)),
        ],
        output: ActivationPrecision::expect(Precision::Bf16),
        accumulation: AccumulationPolicy::Bf16InF32Acc,
        rounding: RoundingProfile::FinalBf16Rne,
        layout: TensorLayout::ContiguousRowMajorV1,
        shape: KernelShapeBounds {
            max_rows: 1024,
            max_input: 1024,
            max_output: 1024,
        },
        sm,
        workspace: WorkspaceExpression::RowsTimesIntermediateF32,
        image_sha256: [3; 32],
        symbols: vec![KernelSymbol("project".into()), KernelSymbol("down".into())],
    }])
    .expect("one descriptor")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Case {
    device_control: StrategyControl,
    host_control: StrategyControl,
    arena_fits: bool,
    host_workspace_fits: bool,
    cache_fits: bool,
    amortised: Amortisation,
    resident: bool,
    order: CombineOrder,
    with_topology: bool,
    kernels: Kernels,
}

/// Generous enough that the only binding constraint is the one the case turns
/// off. Computed the same way the planner does, so a case that means "the arena
/// fits" cannot accidentally mean "the arena is one byte short".
fn generous_arena() -> u64 {
    let activations = ROWS * HIDDEN * 2;
    let slots = ROWS * TOP_K * HIDDEN * 2;
    let workspace = 4 * ROWS * INTERMEDIATE * 4;
    activations + slots + workspace
}

fn generous_host_workspace() -> u64 {
    // queue capacity 4, one tile of 64 lanes clamped to the intermediate.
    4 * (INTERMEDIATE + 2 * INTERMEDIATE) * 4
}

fn generous_host_buffers() -> u64 {
    ROWS * HIDDEN * 2 + ROWS * TOP_K * HIDDEN * 2 + ROWS * HIDDEN * 2
}

impl Case {
    fn budget(self) -> ExpertBudget {
        ExpertBudget {
            device: uuid(),
            device_pci_bus_id: BUS.to_string(),
            device_cache_cap_bytes: if self.cache_fits {
                CHUNK * 8
            } else {
                CHUNK - 1
            },
            device_cache_leased_bytes: 0,
            device_arena_free_bytes: if self.arena_fits { generous_arena() } else { 0 },
            host_workspace_bytes: if self.host_workspace_fits {
                generous_host_workspace()
            } else {
                0
            },
            host_buffer_bytes: generous_host_buffers(),
            resident_experts: if self.resident {
                vec![0, 1, 2, 3, 4]
            } else {
                Vec::new()
            },
        }
    }

    fn policy(self) -> ExpertPolicy {
        ExpertPolicy {
            device: self.device_control,
            host: self.host_control,
            // `CHUNK` over one row is the largest per-row transfer this route
            // produces, so the threshold either admits every group or no group
            // that is not already resident.
            max_transfer_bytes_per_row: self.amortised.limit(),
            max_inflight_orders: 4,
            cpu_tile_lanes: 64,
            host_placement: StrategyControl::Auto,
        }
    }

    /// The decision rule, restated independently of the planner.
    ///
    /// Returns the candidate every group is expected on, per expert, or `None`
    /// when the plan is expected to be refused.
    fn expected_choice(self, reuse: u64) -> Option<Candidate> {
        if self.device_control == StrategyControl::Required
            && self.host_control == StrategyControl::Required
        {
            return None;
        }
        let device_open = self.device_control.may_select()
            && self.host_control != StrategyControl::Required
            && self.kernels == Kernels::Matching
            && self.arena_fits;
        let host_open = self.host_control.may_select()
            && self.device_control != StrategyControl::Required
            && self.host_workspace_fits;
        if !device_open && self.device_control == StrategyControl::Required {
            return None;
        }
        if !host_open && self.host_control == StrategyControl::Required {
            return None;
        }
        if !device_open && !host_open {
            return None;
        }
        let transfer = if self.resident { 0 } else { CHUNK };
        let group_on_device = device_open
            && transfer
                <= (if self.cache_fits {
                    CHUNK * 8
                } else {
                    CHUNK - 1
                })
            && {
                let per_row = transfer.div_ceil(reuse);
                per_row <= self.amortised.limit()
            };
        if group_on_device {
            Some(Candidate::Device)
        } else if host_open {
            Some(Candidate::Host)
        } else {
            None
        }
    }
}

/// Which error a refusal must carry, stated independently of the planner.
///
/// The sweep asserted only *that* a case was refused until mutation testing
/// showed two mutants surviving on that alone: one that let a `required`
/// candidate fall back with a capacity error instead of an actionable one, and
/// one that accepted two `required` candidates at once. A refusal's reason is
/// part of its contract, so it is part of what is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedError {
    /// Both candidates are `required`.
    StrategyContradiction,
    /// A `required` candidate is not admissible.
    RequiredUnavailable,
    /// Neither candidate fits, and neither was required.
    Capacity,
}

impl Case {
    fn expected_error(self) -> ExpectedError {
        if self.device_control == StrategyControl::Required
            && self.host_control == StrategyControl::Required
        {
            ExpectedError::StrategyContradiction
        } else if self.device_control == StrategyControl::Required
            || self.host_control == StrategyControl::Required
        {
            ExpectedError::RequiredUnavailable
        } else {
            ExpectedError::Capacity
        }
    }
}

fn classify(error: &moxie_types::Error) -> ExpectedError {
    match error {
        moxie_types::Error::CapacityExceeded { .. } => ExpectedError::Capacity,
        moxie_types::Error::InvalidRequest {
            field: "strategy",
            detail,
        } => {
            if detail.contains("both candidates are `required`") {
                ExpectedError::StrategyContradiction
            } else {
                ExpectedError::RequiredUnavailable
            }
        }
        other => panic!("a refusal carried an unexpected error: {other}"),
    }
}

fn reuse_counts() -> BTreeMap<u32, u64> {
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    for expert in ROUTE {
        *counts.entry(expert).or_default() += 1;
    }
    counts
}

#[test]
fn the_chooser_agrees_with_an_independent_statement_of_its_rule_across_the_product() {
    let controls = [
        StrategyControl::Off,
        StrategyControl::Auto,
        StrategyControl::Required,
    ];
    let counts = reuse_counts();
    let topology = topology();

    let mut cases = 0usize;
    let mut planned = 0usize;
    let mut refused = 0usize;
    let mut mixed = 0usize;
    let mut all_device = 0usize;
    let mut all_host = 0usize;
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();

    for device_control in controls {
        for host_control in controls {
            for arena_fits in [true, false] {
                for host_workspace_fits in [true, false] {
                    for cache_fits in [true, false] {
                        for amortised in
                            [Amortisation::All, Amortisation::Partial, Amortisation::None]
                        {
                            for resident in [true, false] {
                                for order in [
                                    CombineOrder::AscendingExpertId,
                                    CombineOrder::SelectionOrder,
                                ] {
                                    for with_topology in [true, false] {
                                        for kernels in
                                            [Kernels::Absent, Kernels::Matching, Kernels::WrongSm]
                                        {
                                            let case = Case {
                                                device_control,
                                                host_control,
                                                arena_fits,
                                                host_workspace_fits,
                                                cache_fits,
                                                amortised,
                                                resident,
                                                order,
                                                with_topology,
                                                kernels,
                                            };
                                            cases += 1;
                                            let cat = match kernels {
                                                Kernels::Absent => None,
                                                Kernels::Matching => {
                                                    Some(catalogue(SmVersion::SM86))
                                                }
                                                Kernels::WrongSm => {
                                                    Some(catalogue(SmVersion::SM120))
                                                }
                                            };
                                            let cap = capability();
                                            let injected = cat.as_ref().map(|c| ExpertKernels {
                                                capability: &cap,
                                                catalogue: c,
                                            });
                                            let result = compile_experts(
                                                &mlp(),
                                                &combine(order),
                                                &ROUTE,
                                                &case.budget(),
                                                &case.policy(),
                                                with_topology.then_some(&topology),
                                                injected,
                                            );
                                            // Every expert's expectation. A plan
                                            // exists only when every group has one.
                                            let expected: Vec<Option<Candidate>> = counts
                                                .values()
                                                .map(|reuse| case.expected_choice(*reuse))
                                                .collect();
                                            match result {
                                                Err(refusal) => {
                                                    refused += 1;
                                                    assert!(
                                                        expected.iter().any(Option::is_none),
                                                        "{case:?} was refused but every group had a \
                                                     candidate: {refusal}"
                                                    );
                                                    assert!(
                                                        refusal.device.is_some()
                                                            || refusal.host.is_some(),
                                                        "{case:?}: a refusal with no reason"
                                                    );
                                                    assert_eq!(
                                                        classify(&refusal.error),
                                                        case.expected_error(),
                                                        "{case:?}: {refusal}"
                                                    );
                                                    for reason in [refusal.device, refusal.host]
                                                        .into_iter()
                                                        .flatten()
                                                    {
                                                        *reasons
                                                            .entry(reason_name(reason).to_string())
                                                            .or_default() += 1;
                                                    }
                                                }
                                                Ok(plan) => {
                                                    planned += 1;
                                                    plan.check_invariants().expect("invariants");
                                                    assert!(
                                                        expected.iter().all(Option::is_some),
                                                        "{case:?} produced a plan where a group had \
                                                     no candidate"
                                                    );
                                                    for (group, want) in
                                                        plan.groups().iter().zip(&expected)
                                                    {
                                                        assert_eq!(
                                                            Some(group.placement.candidate()),
                                                            *want,
                                                            "{case:?}: expert {}",
                                                            group.expert
                                                        );
                                                        *reasons
                                                            .entry(
                                                                reason_name(group.decision.reason)
                                                                    .to_string(),
                                                            )
                                                            .or_default() += 1;
                                                    }
                                                    let d = plan.uses(Candidate::Device);
                                                    let h = plan.uses(Candidate::Host);
                                                    match (d, h) {
                                                        (true, true) => mixed += 1,
                                                        (true, false) => all_device += 1,
                                                        (false, true) => all_host += 1,
                                                        (false, false) => {
                                                            panic!("a plan with no group")
                                                        }
                                                    }
                                                    check_envelope(&plan);
                                                    check_placement(&plan, case);
                                                    check_reduction_order(&plan, order);
                                                    // A device plan carries the
                                                    // descriptor it selected; a
                                                    // host-only plan carries none.
                                                    assert_eq!(
                                                        plan.kernel().is_some(),
                                                        plan.uses(Candidate::Device),
                                                        "{case:?}"
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "expert-plan sweep: {cases} combination(s); {planned} planned ({all_device} all-device, \
         {all_host} all-host, {mixed} mixed), {refused} refused"
    );
    let mut names: Vec<_> = reasons.iter().collect();
    names.sort();
    for (name, count) in &names {
        println!("  reason {name}: {count}");
    }

    assert_eq!(cases, 3 * 3 * 2 * 2 * 2 * 3 * 2 * 2 * 2 * 3);
    assert_eq!(planned + refused, cases);
    // Every arm of the decision is exercised, and the counts are printed above
    // rather than asserted as a sentence somewhere else.
    for required in [
        "candidate-off",
        "transfer-not-amortised",
        "device-cache-too-small",
        "device-arena-too-small",
        "host-workspace-too-small",
        "other-candidate-required",
        "no-qualified-kernel",
        "not-preferred",
    ] {
        assert!(
            reasons.get(required).copied().unwrap_or(0) > 0,
            "{required} was never exercised; the sweep does not cover what it claims"
        );
    }
    assert!(mixed > 0, "no case put one expert on each candidate");
    assert!(all_device > 0 && all_host > 0);
}

fn reason_name(reason: RejectionReason) -> &'static str {
    match reason {
        RejectionReason::CandidateOff => "candidate-off",
        RejectionReason::TransferNotAmortised { .. } => "transfer-not-amortised",
        RejectionReason::DeviceCacheTooSmall { .. } => "device-cache-too-small",
        RejectionReason::DeviceArenaTooSmall { .. } => "device-arena-too-small",
        RejectionReason::HostWorkspaceTooSmall { .. } => "host-workspace-too-small",
        RejectionReason::HostBuffersTooSmall { .. } => "host-buffers-too-small",
        RejectionReason::OtherCandidateRequired => "other-candidate-required",
        RejectionReason::NoQualifiedKernel => "no-qualified-kernel",
        RejectionReason::NotPreferred => "not-preferred",
    }
}

fn check_envelope(plan: &moxie_plan::expert::ExpertPlan) {
    let envelope = plan.envelope();
    let activations = ROWS * HIDDEN * 2;
    let slots = ROWS * TOP_K * HIDDEN * 2;
    assert_eq!(plan.activation_bytes(), activations);
    assert_eq!(plan.slot_bytes(), slots);
    // Device regions are charged in whole 256-byte ranges, because that is what
    // the executor will allocate. Charging the logical extent and allocating the
    // aligned one is how a plan comes to need more memory than it reserved.
    let align = |bytes: u64| bytes.div_ceil(256) * 256;
    if plan.uses(Candidate::Device) {
        assert_eq!(
            envelope.device_bytes(DeviceTier::Activations),
            align(activations) + align(slots)
        );
        assert_eq!(
            envelope.device_bytes(DeviceTier::KernelWorkspace),
            align(u64::from(plan.queue_capacity()) * ROWS * INTERMEDIATE * 4)
        );
        // Two `RouteIndex` operands, each sized for the largest group this
        // shape can produce.
        assert_eq!(
            envelope.device_bytes(DeviceTier::TransferStaging),
            align(ROWS * TOP_K * 4) * 2
        );
    } else {
        assert_eq!(envelope.total_device_bytes(), 0);
    }
    if plan.uses(Candidate::Host) {
        assert_eq!(
            envelope.host_bytes(HostTier::CpuWorkspace),
            u64::from(plan.queue_capacity()) * (INTERMEDIATE + 2 * INTERMEDIATE) * 4
        );
    } else {
        assert_eq!(envelope.host_bytes(HostTier::CpuWorkspace), 0);
    }
    // The residency demand is reported and is exactly the non-resident device
    // groups' chunks -- it is not in either tier list, because the residency
    // authority admits those bytes and charging them here would charge twice.
    let demanded: u64 = plan
        .groups_on(Candidate::Device)
        .filter(|g| !g.decision.already_resident)
        .map(|g| g.chunk_bytes)
        .sum();
    assert_eq!(envelope.residency_demand_bytes, demanded);
}

/// The reduction permutation, recomputed independently of the planner.
///
/// Mutation testing found that the sweep passed when the plan ignored
/// `CombineOrder` entirely -- `check_invariants` only asks whether each row is
/// *a* permutation, and the identity is one.
fn check_reduction_order(plan: &moxie_plan::expert::ExpertPlan, order: CombineOrder) {
    let top_k = TOP_K as usize;
    let mut want = Vec::new();
    for row in ROUTE.chunks_exact(top_k) {
        let mut positions: Vec<u32> = (0..top_k as u32).collect();
        if order == CombineOrder::AscendingExpertId {
            positions.sort_by_key(|j| (row[*j as usize], *j));
        }
        want.extend(positions);
    }
    assert_eq!(plan.reduction_order(), want.as_slice(), "{order:?}");
}

fn check_placement(plan: &moxie_plan::expert::ExpertPlan, case: Case) {
    let expected = if case.with_topology {
        HostPlacement::Node(NumaNodeId::new(1))
    } else {
        HostPlacement::Unspecified
    };
    assert_eq!(plan.host_placement(), expected);
    for group in plan.groups_on(Candidate::Host) {
        assert_eq!(group.placement, Placement::Host(expected));
    }
}

#[test]
fn a_required_placement_with_no_topology_is_an_error_rather_than_an_unplaced_buffer() {
    let case = Case {
        device_control: StrategyControl::Auto,
        host_control: StrategyControl::Auto,
        arena_fits: true,
        host_workspace_fits: true,
        cache_fits: true,
        amortised: Amortisation::All,
        resident: false,
        order: CombineOrder::AscendingExpertId,
        with_topology: false,
        kernels: Kernels::Matching,
    };
    let mut policy = case.policy();
    policy.host_placement = StrategyControl::Required;
    let refusal = compile_experts(
        &mlp(),
        &combine(case.order),
        &ROUTE,
        &case.budget(),
        &policy,
        None,
        None,
    )
    .expect_err("required placement with no topology");
    assert!(
        format!("{refusal}").contains("placement is required"),
        "{refusal}"
    );
}

#[test]
fn placement_off_leaves_host_buffers_unplaced_even_with_a_topology() {
    let case = Case {
        device_control: StrategyControl::Off,
        host_control: StrategyControl::Auto,
        arena_fits: true,
        host_workspace_fits: true,
        cache_fits: true,
        amortised: Amortisation::All,
        resident: false,
        order: CombineOrder::AscendingExpertId,
        with_topology: true,
        kernels: Kernels::Matching,
    };
    let mut policy = case.policy();
    policy.host_placement = StrategyControl::Off;
    let plan = compile_experts(
        &mlp(),
        &combine(case.order),
        &ROUTE,
        &case.budget(),
        &policy,
        Some(&topology()),
        None,
    )
    .expect("plan");
    assert_eq!(plan.host_placement(), HostPlacement::Unspecified);
}

#[test]
fn a_route_that_sends_one_row_to_one_expert_twice_is_refused() {
    let case = Case {
        device_control: StrategyControl::Auto,
        host_control: StrategyControl::Auto,
        arena_fits: true,
        host_workspace_fits: true,
        cache_fits: true,
        amortised: Amortisation::All,
        resident: false,
        order: CombineOrder::AscendingExpertId,
        with_topology: true,
        kernels: Kernels::Matching,
    };
    let route = [0u32, 0, 1, 2, 3, 4, 5, 1];
    let refusal = compile_experts(
        &mlp(),
        &combine(case.order),
        &route,
        &case.budget(),
        &case.policy(),
        Some(&topology()),
        None,
    )
    .expect_err("a duplicated selection");
    assert!(format!("{refusal}").contains("twice"), "{refusal}");
}

#[test]
fn the_reduction_order_is_a_permutation_per_row_and_follows_the_declared_order() {
    let case = Case {
        device_control: StrategyControl::Auto,
        host_control: StrategyControl::Auto,
        arena_fits: true,
        host_workspace_fits: true,
        cache_fits: true,
        amortised: Amortisation::All,
        resident: false,
        order: CombineOrder::AscendingExpertId,
        with_topology: true,
        kernels: Kernels::Matching,
    };
    let ascending = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &case.budget(),
        &case.policy(),
        Some(&topology()),
        None,
    )
    .expect("plan");
    let selection = compile_experts(
        &mlp(),
        &combine(CombineOrder::SelectionOrder),
        &ROUTE,
        &case.budget(),
        &case.policy(),
        Some(&topology()),
        None,
    )
    .expect("plan");

    assert_eq!(selection.reduction_order(), &[0, 1, 0, 1, 0, 1, 0, 1]);
    // `ROUTE`'s rows are descending by expert id, so ascending order reverses
    // every one of them. The two parameters must therefore disagree here, and
    // asserting that they do is what keeps a chooser that ignores the parameter
    // from passing.
    assert_eq!(ascending.reduction_order(), &[1, 0, 1, 0, 1, 0, 1, 0]);
    assert_ne!(ascending.reduction_order(), selection.reduction_order());

    // And the converse: a route whose rows are already ascending leaves the two
    // orders agreeing, which is why the sweep's own route is not that one.
    let ascending_rows = [0u32, 1, 0, 2, 1, 3, 0, 4];
    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ascending_rows,
        &case.budget(),
        &case.policy(),
        Some(&topology()),
        None,
    )
    .expect("plan");
    assert_eq!(plan.reduction_order(), &[0, 1, 0, 1, 0, 1, 0, 1]);
}
