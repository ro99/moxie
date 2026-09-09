//! Acceptance tests for task 0006: the resource ledger and admission.
//!
//! Each test names the rule from the task contract that it proves. Nothing here
//! allocates, measures or touches a device: every byte is declared by the test.

use moxie_memory::{
    AdmitError, BindingKind, BufferRequest, CapacitySnapshot, DerivedReserve, Ledger,
    LegalAlternative, PlanRequest, ReserveRule, Scaling, StageSpan,
};
use moxie_types::{DeviceTier, DeviceUuid, HostTier, Scope, Tier};

const KV: Tier = Tier::Device(DeviceTier::KvStatePages);
const ACT: Tier = Tier::Device(DeviceTier::Activations);
const WORK: Tier = Tier::Device(DeviceTier::KernelWorkspace);
const WEIGHTS: Tier = Tier::Device(DeviceTier::PackedResidentWeights);
const EXPERTS: Tier = Tier::Device(DeviceTier::ExpertCache);
const STAGING: Tier = Tier::Device(DeviceTier::TransferStaging);
const BRANCHES: Tier = Tier::Device(DeviceTier::EntropyBranches);
const SPILL: Tier = Tier::Host(HostTier::StateSpill);
const PINNED: Tier = Tier::Host(HostTier::Pinned);
const MAPPED: Tier = Tier::Host(HostTier::MappedResident);

fn gpu(n: u8) -> Scope {
    let mut bytes = [0u8; 16];
    bytes[15] = n;
    Scope::Device(DeviceUuid::from_bytes(bytes))
}

fn device_snapshot(scope: Scope, physical: u64) -> CapacitySnapshot {
    CapacitySnapshot::new(scope, physical, 0).unwrap()
}

fn host_snapshot(physical: u64, headroom: u64) -> CapacitySnapshot {
    CapacitySnapshot::new(Scope::Host, physical, headroom).unwrap()
}

/// Every committed counter in the ledger -- per tier and per scope -- so a
/// before/after comparison cannot miss one.
fn counters(ledger: &Ledger) -> Vec<(Scope, Option<Tier>, u64)> {
    let mut out = Vec::new();
    for scope in ledger.scopes().collect::<Vec<_>>() {
        out.push((scope, None, ledger.scope_committed(scope)));
        for tier in Tier::valid_in(scope.kind()) {
            out.push((scope, Some(tier), ledger.committed(scope, tier)));
        }
    }
    out
}

fn rejection(e: &AdmitError) -> &moxie_memory::Rejection {
    e.as_rejection()
        .unwrap_or_else(|| panic!("expected a refusal, got {e}"))
}

#[test]
fn the_peak_is_the_stage_maximum_and_not_the_sum() {
    // Two buffers of 100 B that are never live together cost 100 B, not 200 B.
    // Document 03: reserve the peak overlapping live set, "not the sum of
    // mutually exclusive buffers".
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 150)]).unwrap();

    let mut disjoint = PlanRequest::new("disjoint", ["prefill", "decode"]).unwrap();
    disjoint
        .buffer(BufferRequest::new("act", d, ACT, 100, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("work", d, WORK, 100, StageSpan::at(1)))
        .unwrap();
    let reserved = ledger.admit(&disjoint).expect("100 B peak fits in 150 B");
    assert_eq!(ledger.scope_committed(d), 100);
    ledger.release(reserved).unwrap();

    let mut overlapping = PlanRequest::new("overlapping", ["prefill", "decode"]).unwrap();
    overlapping
        .buffer(BufferRequest::new(
            "act",
            d,
            ACT,
            100,
            StageSpan::inclusive(0, 1),
        ))
        .unwrap()
        .buffer(BufferRequest::new(
            "work",
            d,
            WORK,
            100,
            StageSpan::inclusive(0, 1),
        ))
        .unwrap();
    let e = ledger.admit(&overlapping).unwrap_err();
    let r = rejection(&e);
    assert_eq!(r.worst().needed_bytes, 200);
    assert_eq!(r.worst().available_bytes, 150);
    assert_eq!(r.shortfall_bytes, 50);
}

#[test]
fn a_barrier_where_both_phases_are_live_is_the_peak() {
    // Document 03: "Admission must reflect temporary coexistence during a
    // transition." The barrier stage is where prefill and decode buffers
    // overlap, and that stage is what must be reserved.
    let d = gpu(1);
    let ledger = Ledger::new([device_snapshot(d, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("transition", ["prefill", "barrier", "decode"]).unwrap();
    plan.buffer(BufferRequest::new(
        "prefill-act",
        d,
        ACT,
        300,
        StageSpan::inclusive(0, 1),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "decode-work",
        d,
        WORK,
        200,
        StageSpan::inclusive(1, 2),
    ))
    .unwrap();

    let report = ledger.preview(&plan).unwrap();
    let scope = report.scope(d).unwrap();
    assert_eq!(scope.request_peak_bytes, 500);
    assert_eq!(scope.peak_stage, 1);
    assert_eq!(report.stages[scope.peak_stage as usize], "barrier");
    // Each tier's own peak is at its own stage.
    assert_eq!(scope.tier(ACT).unwrap().peak_stage, 0);
    assert_eq!(scope.tier(WORK).unwrap().peak_stage, 1);
}

#[test]
fn a_tier_cap_binds_independently_of_the_scope_total() {
    let d = gpu(1);
    let snapshot = device_snapshot(d, 100_000)
        .with_tier_cap(EXPERTS, 500)
        .unwrap();
    let mut ledger = Ledger::new([snapshot]).unwrap();
    let mut plan = PlanRequest::new("cache", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new(
        "experts",
        d,
        EXPERTS,
        900,
        StageSpan::at(0),
    ))
    .unwrap();

    let e = ledger.admit(&plan).unwrap_err();
    let r = rejection(&e);
    assert_eq!(
        r.binding.len(),
        1,
        "the scope has room; only the tier cap binds"
    );
    assert_eq!(r.binding[0].kind, BindingKind::TierCap);
    assert_eq!(r.binding[0].tier, EXPERTS);
    assert_eq!(r.binding[0].available_bytes, 500);
    // The scope row still shows room, which is the point of reporting both.
    assert!(r.report.scope(d).unwrap().remaining_headroom_bytes > 0);
}

#[test]
fn a_refusal_leaves_every_counter_exactly_as_it_was() {
    // Atomicity: the tiers that would have fit are not committed either.
    let d = gpu(1);
    let mut ledger = Ledger::new([
        device_snapshot(d, 1_000).with_tier_cap(WORK, 10).unwrap(),
        host_snapshot(1_000, 100),
    ])
    .unwrap();

    let mut first = PlanRequest::new("first", ["decode"]).unwrap();
    first
        .buffer(BufferRequest::new("kv", d, KV, 100, StageSpan::at(0)))
        .unwrap();
    let held = ledger.admit(&first).unwrap();

    let before = counters(&ledger);
    let outstanding_before = ledger.outstanding().len();

    let mut second = PlanRequest::new("second", ["decode"]).unwrap();
    second
        // This one fits.
        .buffer(BufferRequest::new(
            "weights",
            d,
            WEIGHTS,
            100,
            StageSpan::at(0),
        ))
        .unwrap()
        // This one does not: the tier cap is 10 B.
        .buffer(BufferRequest::new("work", d, WORK, 800, StageSpan::at(0)))
        .unwrap();
    let e = ledger.admit(&second).unwrap_err();
    assert!(e.as_rejection().is_some());

    assert_eq!(counters(&ledger), before, "a refusal commits nothing");
    assert_eq!(ledger.outstanding().len(), outstanding_before);
    assert_eq!(
        ledger.committed(d, WEIGHTS),
        0,
        "the tier that fit is not charged"
    );
    ledger.release(held).unwrap();
    assert_eq!(ledger.scope_committed(d), 0);
}

#[test]
fn every_failing_constraint_is_reported_not_only_the_first() {
    let d = gpu(1);
    let snapshot = device_snapshot(d, 500)
        .with_tier_cap(EXPERTS, 10)
        .unwrap()
        .with_tier_cap(STAGING, 10)
        .unwrap();
    let mut ledger = Ledger::new([snapshot]).unwrap();
    let mut plan = PlanRequest::new("everything", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new(
        "experts",
        d,
        EXPERTS,
        400,
        StageSpan::at(0),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "staging",
        d,
        STAGING,
        400,
        StageSpan::at(0),
    ))
    .unwrap();

    let e = ledger.admit(&plan).unwrap_err();
    let r = rejection(&e);
    let kinds: Vec<_> = r.binding.iter().map(|c| (c.tier, c.kind)).collect();
    assert!(
        kinds.contains(&(EXPERTS, BindingKind::TierCap)),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&(STAGING, BindingKind::TierCap)),
        "{kinds:?}"
    );
    assert!(
        r.binding.iter().any(|c| c.kind == BindingKind::ScopeBudget),
        "the scope budget binds too: {kinds:?}"
    );
    // Three constraints, three different numbers: each tier is 390 B over its
    // 10 B cap, and the scope is 300 B over its 500 B budget. The summary takes
    // the worst, and ties keep the first in report order.
    assert_eq!(r.shortfall_bytes, 390);
    assert_eq!(r.worst().kind, BindingKind::TierCap);
    assert_eq!(r.worst().tier, EXPERTS);
    let scope_bound = r
        .binding
        .iter()
        .find(|c| c.kind == BindingKind::ScopeBudget)
        .unwrap();
    assert_eq!(scope_bound.needed_bytes, 800);
    assert_eq!(scope_bound.available_bytes, 500);
}

#[test]
fn lower_context_is_offered_only_when_a_binding_buffer_says_it_scales_with_context() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();

    let mut declared = PlanRequest::new("declared", ["decode"]).unwrap();
    declared
        .buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)).scaling(Scaling::Context))
        .unwrap();
    let e = ledger.admit(&declared).unwrap_err();
    assert!(
        rejection(&e)
            .alternatives
            .contains(&LegalAlternative::LowerContext)
    );

    let mut undeclared = PlanRequest::new("undeclared", ["decode"]).unwrap();
    undeclared
        .buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)))
        .unwrap();
    let e = ledger.admit(&undeclared).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::LowerContext),
        "a suggestion the request cannot act on is worse than none"
    );
}

#[test]
fn fewer_branches_is_not_offered_when_nothing_scales_with_branches() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();

    let mut no_branches = PlanRequest::new("no-branches", ["decode"]).unwrap();
    no_branches
        .buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)).scaling(Scaling::Context))
        .unwrap();
    let e = ledger.admit(&no_branches).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::FewerBranches)
    );

    let mut branches = PlanRequest::new("branches", ["decode"]).unwrap();
    branches
        .buffer(
            BufferRequest::new("entropy", d, BRANCHES, 500, StageSpan::at(0))
                .scaling(Scaling::Branches),
        )
        .unwrap();
    let e = ledger.admit(&branches).unwrap_err();
    assert!(
        rejection(&e)
            .alternatives
            .contains(&LegalAlternative::FewerBranches)
    );
}

#[test]
fn another_precision_is_offered_only_when_a_weight_tier_binds() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();

    let mut weights = PlanRequest::new("weights", ["decode"]).unwrap();
    weights
        .buffer(BufferRequest::new("w", d, WEIGHTS, 500, StageSpan::at(0)))
        .unwrap();
    let e = ledger.admit(&weights).unwrap_err();
    assert!(
        rejection(&e)
            .alternatives
            .contains(&LegalAlternative::OtherWeightPrecisionOrArtifact)
    );

    let mut logits = PlanRequest::new("logits", ["decode"]).unwrap();
    logits
        .buffer(BufferRequest::new(
            "logits",
            d,
            Tier::Device(DeviceTier::Logits),
            500,
            StageSpan::at(0),
        ))
        .unwrap();
    let e = ledger.admit(&logits).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::OtherWeightPrecisionOrArtifact),
        "a vocabulary-sized logits buffer does not shrink with weight precision"
    );
}

#[test]
fn different_topology_is_offered_only_when_more_than_one_device_is_declared() {
    let (a, b) = (gpu(1), gpu(2));
    let mut ledger = Ledger::new([device_snapshot(a, 100), device_snapshot(b, 100_000)]).unwrap();

    let mut one = PlanRequest::new("one-device", ["decode"]).unwrap();
    one.buffer(BufferRequest::new("w", a, WEIGHTS, 500, StageSpan::at(0)))
        .unwrap();
    let e = ledger.admit(&one).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::DifferentTopology)
    );

    let mut two = PlanRequest::new("two-devices", ["decode"]).unwrap();
    two.buffer(BufferRequest::new("w-a", a, WEIGHTS, 500, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("w-b", b, WEIGHTS, 500, StageSpan::at(0)))
        .unwrap();
    let e = ledger.admit(&two).unwrap_err();
    assert!(
        rejection(&e)
            .alternatives
            .contains(&LegalAlternative::DifferentTopology)
    );
}

#[test]
fn host_backed_execution_is_offered_only_when_the_host_can_absorb_the_shortfall() {
    let d = gpu(1);

    // A host with room for the 400 B shortfall.
    let mut roomy = Ledger::new([device_snapshot(d, 100), host_snapshot(10_000, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("spill", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)))
        .unwrap();
    let e = roomy.admit(&plan).unwrap_err();
    assert!(
        rejection(&e)
            .alternatives
            .contains(&LegalAlternative::HostBackedExecution)
    );

    // A host with 10 B free cannot absorb it, and is not offered.
    let mut tight = Ledger::new([device_snapshot(d, 100), host_snapshot(20, 10)]).unwrap();
    let e = tight.admit(&plan).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::HostBackedExecution)
    );

    // And with no host scope at all there is nothing to fall back to.
    let mut alone = Ledger::new([device_snapshot(d, 100)]).unwrap();
    let e = alone.admit(&plan).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::HostBackedExecution)
    );
}

#[test]
fn a_rejection_applies_nothing_and_returns_no_reservation() {
    // Document 03 forbids automatically shortening context or lowering weights.
    // The type system carries half of that (no reservation is returned) and this
    // test carries the other half (nothing was changed to make one possible).
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();
    let before = counters(&ledger);
    let mut plan = PlanRequest::new("too-big", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)).scaling(Scaling::Context))
        .unwrap();
    assert!(ledger.admit(&plan).is_err());
    assert_eq!(counters(&ledger), before);
    assert!(ledger.outstanding().is_empty());
    // The same request refused twice gives the same answer: nothing adapted.
    let e = ledger.admit(&plan).unwrap_err();
    assert_eq!(rejection(&e).shortfall_bytes, 400);
}

#[test]
fn an_incoming_item_reserve_is_derived_from_the_request_not_supplied() {
    // R03: Inkling's worst-case incoming-expert reserve, as a derived quantity.
    let d = gpu(1);
    let ledger = Ledger::new([device_snapshot(d, 100_000)]).unwrap();

    let build = |largest: u64| {
        let mut plan = PlanRequest::new("moe", ["decode"]).unwrap();
        plan.buffer(BufferRequest::new(
            "e0",
            d,
            EXPERTS,
            largest,
            StageSpan::at(0),
        ))
        .unwrap()
        .buffer(BufferRequest::new("e1", d, EXPERTS, 80, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("e2", d, EXPERTS, 60, StageSpan::at(0)))
        .unwrap()
        .reserve(DerivedReserve::new(
            "incoming",
            d,
            EXPERTS,
            ReserveRule::LargestBufferOfTier,
            StageSpan::at(0),
        ))
        .unwrap();
        plan
    };

    let peak = |largest: u64| {
        ledger
            .preview(&build(largest))
            .unwrap()
            .scope(d)
            .unwrap()
            .tier(EXPERTS)
            .unwrap()
            .request_peak_bytes
    };

    assert_eq!(peak(100), 100 + 80 + 60 + 100);
    // Changing the largest buffer changes the reserve. A constant would not
    // move, which is exactly the legacy failure this rule replaces.
    assert_eq!(peak(400), 400 + 80 + 60 + 400);
}

#[test]
fn the_two_largest_linears_reserve_is_the_sum_of_the_two_largest() {
    // R03's other constant: GLM-5.3 reserved its two largest linears.
    let d = gpu(1);
    let ledger = Ledger::new([device_snapshot(d, 100_000)]).unwrap();
    let mut plan = PlanRequest::new("dense", ["prefill"]).unwrap();
    plan.buffer(BufferRequest::new("l0", d, WEIGHTS, 90, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("l1", d, WEIGHTS, 70, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("l2", d, WEIGHTS, 50, StageSpan::at(0)))
        .unwrap()
        .reserve(DerivedReserve::new(
            "two-largest",
            d,
            WEIGHTS,
            ReserveRule::NLargestBuffersOfTier(2),
            StageSpan::at(0),
        ))
        .unwrap();
    let peak = ledger
        .preview(&plan)
        .unwrap()
        .scope(d)
        .unwrap()
        .tier(WEIGHTS)
        .unwrap()
        .request_peak_bytes;
    assert_eq!(peak, 90 + 70 + 50 + (90 + 70));
}

#[test]
fn a_reserve_over_more_buffers_than_exist_is_an_error_not_a_clamp() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100_000)]).unwrap();
    let mut plan = PlanRequest::new("moe", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("e0", d, EXPERTS, 10, StageSpan::at(0)))
        .unwrap()
        .reserve(DerivedReserve::new(
            "incoming",
            d,
            EXPERTS,
            ReserveRule::NLargestBuffersOfTier(3),
            StageSpan::at(0),
        ))
        .unwrap();
    match ledger.admit(&plan).unwrap_err() {
        AdmitError::Invalid(e) => assert_eq!(e.kind(), "invalid_request"),
        other => panic!("expected an invalid request, got {other}"),
    }

    // The same is true of a reserve over a tier with no buffers at all.
    let mut empty = PlanRequest::new("empty", ["decode"]).unwrap();
    empty
        .reserve(DerivedReserve::new(
            "incoming",
            d,
            EXPERTS,
            ReserveRule::LargestBufferOfTier,
            StageSpan::at(0),
        ))
        .unwrap();
    assert!(matches!(
        ledger.admit(&empty).unwrap_err(),
        AdmitError::Invalid(_)
    ));
}

#[test]
fn a_mappings_virtual_extent_is_reported_and_charged_to_nothing() {
    // Document 03: "Mapped virtual bytes do not equal committed host RAM;
    // neither is free." The resident pages are charged like every other byte
    // (see `resident_mapped_pages_share_the_physical_host_budget`); the address
    // range they sit in is reported beside them and charged to nothing.
    let mut ledger = Ledger::new([host_snapshot(1_000, 100)]).unwrap();
    let mut plan = PlanRequest::new("mapped", ["prefill"]).unwrap();
    plan.buffer(BufferRequest::new(
        "arena",
        Scope::Host,
        PINNED,
        800,
        StageSpan::at(0),
    ))
    .unwrap()
    .buffer(
        BufferRequest::new("mapping", Scope::Host, MAPPED, 50, StageSpan::at(0))
            .virtual_extent(100_000),
    )
    .unwrap();

    let reserved = ledger
        .admit(&plan)
        .expect("850 B resident fits; the 100 kB of address space is not memory");
    let report = ledger
        .preview(&PlanRequest::new("empty", ["prefill"]).unwrap())
        .unwrap();
    let host = report.scope(Scope::Host).unwrap();
    assert_eq!(host.committed_bytes, 850, "both resident sets are charged");
    assert_eq!(ledger.committed(Scope::Host, MAPPED), 50);

    let mapped_row = ledger
        .preview(&plan)
        .unwrap()
        .scope(Scope::Host)
        .unwrap()
        .tier(MAPPED)
        .unwrap()
        .clone();
    assert_eq!(mapped_row.request_peak_bytes, 50);
    assert_eq!(mapped_row.virtual_extent_bytes, 100_000);

    ledger.release(reserved).unwrap();
    assert_eq!(ledger.committed(Scope::Host, MAPPED), 0);
}

#[test]
fn a_virtual_extent_is_refused_where_it_would_be_meaningless_or_impossible() {
    let mut plan = PlanRequest::new("mapped", ["prefill"]).unwrap();
    // Only a mapping has an extent distinct from its bytes.
    let e = plan
        .buffer(
            BufferRequest::new("pinned", Scope::Host, PINNED, 10, StageSpan::at(0))
                .virtual_extent(100),
        )
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    // And a resident set cannot exceed the mapping it is resident in.
    let e = plan
        .buffer(
            BufferRequest::new("mapping", Scope::Host, MAPPED, 100, StageSpan::at(0))
                .virtual_extent(10),
        )
        .unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(plan.buffers().is_empty());
}

#[test]
fn a_mapping_is_still_bounded_by_its_own_cap() {
    let snapshot = host_snapshot(1_000, 100)
        .with_tier_cap(MAPPED, 500)
        .unwrap();
    let mut ledger = Ledger::new([snapshot]).unwrap();
    let mut plan = PlanRequest::new("mapped", ["prefill"]).unwrap();
    plan.buffer(BufferRequest::new(
        "mapping",
        Scope::Host,
        MAPPED,
        600,
        StageSpan::at(0),
    ))
    .unwrap();
    let e = ledger.admit(&plan).unwrap_err();
    let r = rejection(&e);
    assert_eq!(r.binding.len(), 1);
    assert_eq!(r.binding[0].tier, MAPPED);
    assert_eq!(r.binding[0].kind, BindingKind::TierCap);
}

#[test]
fn a_device_is_its_uuid_and_two_devices_hold_separate_budgets() {
    let (a, b) = (gpu(1), gpu(2));
    let mut ledger = Ledger::new([device_snapshot(a, 1_000), device_snapshot(b, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("on-b", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", b, KV, 400, StageSpan::at(0)))
        .unwrap();
    let held = ledger.admit(&plan).unwrap();

    assert_eq!(ledger.scope_committed(b), 400);
    assert_eq!(
        ledger.scope_committed(a),
        0,
        "the other device is untouched"
    );
    // Identity is the UUID's bytes, not the order the snapshots were given in
    // and not an ordinal: the same UUID names the same scope.
    let same_b =
        Scope::Device(DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000002").unwrap());
    assert_eq!(same_b, b);
    assert_eq!(ledger.scope_committed(same_b), 400);
    ledger.release(held).unwrap();
}

#[test]
fn a_scope_with_no_snapshot_is_an_invalid_request_not_an_unlimited_one() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("elsewhere", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", gpu(9), KV, 1, StageSpan::at(0)))
        .unwrap();
    match ledger.admit(&plan).unwrap_err() {
        AdmitError::Invalid(e) => assert_eq!(e.kind(), "invalid_request"),
        other => panic!("an unknown device must not be treated as free: {other}"),
    }
}

#[test]
fn release_returns_the_bytes_and_only_to_its_own_ledger() {
    let d = gpu(1);
    let mut first = Ledger::new([device_snapshot(d, 1_000)]).unwrap();
    let mut second = Ledger::new([device_snapshot(d, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("held", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", d, KV, 400, StageSpan::at(0)))
        .unwrap();

    let held = first.admit(&plan).unwrap();
    assert_eq!(first.outstanding().len(), 1);
    assert_eq!(first.outstanding()[0].label, "held");

    let before_first = counters(&first);
    let before_second = counters(&second);
    let refused = second
        .release(held)
        .expect_err("a reservation must not be releasable against another ledger");
    assert_eq!(refused.error.kind(), "invalid_request");
    // Nothing moved in either ledger, and the handle came back rather than
    // being destroyed by the error path.
    assert_eq!(counters(&first), before_first);
    assert_eq!(counters(&second), before_second);
    assert_eq!(first.committed(d, KV), 400);

    first.release(refused.reservation).unwrap();
    assert_eq!(first.committed(d, KV), 0);
    assert!(first.outstanding().is_empty());
}

#[test]
fn a_turn_that_ends_with_no_next_token_still_has_to_release() {
    // R08: the legacy leak released a lease on "next token", so a turn that
    // ended leaked. Here the release is explicit and the leak is visible.
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("turn", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", d, KV, 400, StageSpan::at(0)))
        .unwrap();

    let held = ledger.admit(&plan).unwrap();
    // The turn ends here. No next token arrives, so nothing implicit happens.
    drop(held);
    assert_eq!(
        ledger.scope_committed(d),
        400,
        "dropping a reservation must not free it: Drop alone cannot be the release path"
    );
    let leaked = ledger.outstanding();
    assert_eq!(leaked.len(), 1);
    assert_eq!(leaked[0].label, "turn", "the leak names itself");
    assert_eq!(leaked[0].charges, vec![(d, KV, 400)]);
    assert_eq!(leaked[0].scope_charges, vec![(d, 400)]);

    // A second turn is still admitted against the truth, not against a
    // counter that silently forgot.
    let mut big = PlanRequest::new("second-turn", ["decode"]).unwrap();
    big.buffer(BufferRequest::new("kv", d, KV, 700, StageSpan::at(0)))
        .unwrap();
    assert!(ledger.admit(&big).is_err());
}

#[test]
fn overflow_is_an_error_not_a_wrap() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, u64::MAX)]).unwrap();
    let mut plan = PlanRequest::new("overflow", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("a", d, KV, u64::MAX, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("b", d, KV, 1, StageSpan::at(0)))
        .unwrap();
    match ledger.admit(&plan).unwrap_err() {
        AdmitError::Invalid(e) => assert_eq!(e.kind(), "dim"),
        other => panic!("a wrapped total would admit a plan that cannot fit: {other}"),
    }

    // The same across two tiers, where the scope total is what overflows.
    let mut across = PlanRequest::new("across", ["decode"]).unwrap();
    across
        .buffer(BufferRequest::new("a", d, KV, u64::MAX, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("b", d, ACT, 1, StageSpan::at(0)))
        .unwrap();
    match ledger.admit(&across).unwrap_err() {
        AdmitError::Invalid(e) => assert_eq!(e.kind(), "dim"),
        other => panic!("expected an overflow error, got {other}"),
    }
}

#[test]
fn preview_commits_nothing() {
    let d = gpu(1);
    let ledger = Ledger::new([device_snapshot(d, 1_000)]).unwrap();
    let mut plan = PlanRequest::new("preview", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", d, KV, 400, StageSpan::at(0)))
        .unwrap();
    let before = counters(&ledger);
    let report = ledger.preview(&plan).unwrap();
    assert_eq!(report.scope(d).unwrap().request_peak_bytes, 400);
    assert_eq!(counters(&ledger), before);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_rejection_converts_to_the_typed_error_naming_the_worst_tier() {
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();
    let mut plan = PlanRequest::new("big", ["decode"]).unwrap();
    plan.buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)))
        .unwrap();
    let e = ledger.admit(&plan).unwrap_err();
    let worst_tier = rejection(&e).worst().tier;
    let typed: moxie_types::Error = e.into();
    assert_eq!(typed.kind(), "capacity_exceeded");
    match typed {
        moxie_types::Error::CapacityExceeded { tier, .. } => assert_eq!(tier, Some(worst_tier)),
        other => panic!("wrong variant: {other:?}"),
    }
}

// --- two consumers, different shapes and different tier profiles -------------

#[test]
fn a_dense_single_device_chain_admits_and_then_binds_on_its_own_tier() {
    // Consumer one: a dense BF16 layer chain on one device. Weights are
    // persistent, activations and workspace are per stage, logits appear once.
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 4_000)]).unwrap();
    let mut plan = PlanRequest::new("dense", ["prefill", "barrier", "decode"]).unwrap();
    plan.buffer(BufferRequest::new(
        "weights",
        d,
        WEIGHTS,
        2_000,
        StageSpan::inclusive(0, 2),
    ))
    .unwrap()
    .buffer(
        BufferRequest::new("kv", d, KV, 600, StageSpan::inclusive(0, 2)).scaling(Scaling::Context),
    )
    .unwrap()
    .buffer(BufferRequest::new(
        "prefill-act",
        d,
        ACT,
        900,
        StageSpan::inclusive(0, 1),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "decode-act",
        d,
        ACT,
        100,
        StageSpan::inclusive(1, 2),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "logits",
        d,
        Tier::Device(DeviceTier::Logits),
        300,
        StageSpan::at(2),
    ))
    .unwrap();

    let held = ledger.admit(&plan).expect("3600 B peak fits in 4000 B");
    // Peak is the barrier: 2000 weights + 600 kv + 900 + 100 activations.
    assert_eq!(ledger.scope_committed(d), 3_600);

    // A second identical plan cannot fit beside the first.
    let e = ledger.admit(&plan).unwrap_err();
    let r = rejection(&e);
    assert_eq!(r.worst().kind, BindingKind::ScopeBudget);
    assert!(
        r.alternatives.contains(&LegalAlternative::LowerContext),
        "the KV buffer declared that it follows context"
    );
    ledger.release(held).unwrap();
    assert_eq!(ledger.scope_committed(d), 0);
    assert!(ledger.admit(&plan).is_ok(), "the bytes really came back");
}

#[test]
fn an_oversized_moe_workload_admits_and_then_binds_on_its_own_tier() {
    // Consumer two: different shapes, different tiers. Experts stream through a
    // capped cache with an incoming reserve, staging is transient, and state
    // spills to the host. Document 06 M2 is the workload this shape anticipates;
    // the ledger has to be able to express it before then.
    let (a, b) = (gpu(1), gpu(2));
    let mut ledger = Ledger::new([
        device_snapshot(a, 10_000)
            .with_tier_cap(EXPERTS, 3_000)
            .unwrap(),
        device_snapshot(b, 6_000),
        host_snapshot(100_000, 20_000),
    ])
    .unwrap();

    let mut plan = PlanRequest::new("moe", ["prefill", "barrier", "decode"]).unwrap();
    plan.buffer(BufferRequest::new(
        "spine-a",
        a,
        WEIGHTS,
        1_500,
        StageSpan::inclusive(0, 2),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "spine-b",
        b,
        WEIGHTS,
        1_500,
        StageSpan::inclusive(0, 2),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "expert-0",
        a,
        EXPERTS,
        800,
        StageSpan::inclusive(0, 2),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "expert-1",
        a,
        EXPERTS,
        700,
        StageSpan::inclusive(0, 2),
    ))
    .unwrap()
    .reserve(DerivedReserve::new(
        "incoming-expert",
        a,
        EXPERTS,
        ReserveRule::LargestBufferOfTier,
        StageSpan::inclusive(0, 2),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "staging",
        a,
        STAGING,
        400,
        StageSpan::inclusive(0, 1),
    ))
    .unwrap()
    .buffer(
        BufferRequest::new(
            "spill",
            Scope::Host,
            SPILL,
            5_000,
            StageSpan::inclusive(1, 2),
        )
        .scaling(Scaling::Context),
    )
    .unwrap()
    .buffer(BufferRequest::new(
        "pinned-staging",
        Scope::Host,
        PINNED,
        2_000,
        StageSpan::at(1),
    ))
    .unwrap();

    let held = ledger.admit(&plan).expect("the shaped workload fits");
    // Expert cache: 800 + 700 + an 800 B incoming reserve, under its 3000 B cap.
    assert_eq!(ledger.committed(a, EXPERTS), 2_300);
    assert_eq!(ledger.committed(Scope::Host, SPILL), 5_000);

    // Growing the cache past its cap binds on the expert tier, and the
    // alternatives name the artifact, not the context.
    let mut bigger = PlanRequest::new("moe-bigger", ["decode"]).unwrap();
    bigger
        .buffer(BufferRequest::new(
            "expert-2",
            a,
            EXPERTS,
            2_000,
            StageSpan::at(0),
        ))
        .unwrap()
        .reserve(DerivedReserve::new(
            "incoming-expert",
            a,
            EXPERTS,
            ReserveRule::LargestBufferOfTier,
            StageSpan::at(0),
        ))
        .unwrap();
    let e = ledger.admit(&bigger).unwrap_err();
    let r = rejection(&e);
    assert_eq!(r.binding.len(), 1);
    assert_eq!(r.binding[0].tier, EXPERTS);
    assert_eq!(r.binding[0].kind, BindingKind::TierCap);
    assert_eq!(r.binding[0].needed_bytes, 2_300 + 4_000);
    assert!(
        r.alternatives
            .contains(&LegalAlternative::OtherWeightPrecisionOrArtifact)
    );
    assert!(!r.alternatives.contains(&LegalAlternative::LowerContext));

    ledger.release(held).unwrap();
    assert_eq!(ledger.scope_committed(a), 0);
    assert_eq!(ledger.scope_committed(Scope::Host), 0);
}

// --- review corrections, 2026-09-08 -----------------------------------------

#[test]
fn resident_mapped_pages_share_the_physical_host_budget() {
    // Review finding 1. Resident pages of a mapping are physical RAM. The
    // distinction document 03 draws is between a mapping's *virtual extent* and
    // its resident pages; excluding the whole tier from the budget did not
    // prevent double counting, it permitted under-counting.
    let mut ledger = Ledger::new([host_snapshot(1_000, 100)]).unwrap();
    let mut plan = PlanRequest::new("mapped", ["run"]).unwrap();
    plan.buffer(BufferRequest::new(
        "pinned",
        Scope::Host,
        PINNED,
        800,
        StageSpan::at(0),
    ))
    .unwrap()
    .buffer(BufferRequest::new(
        "resident",
        Scope::Host,
        MAPPED,
        800,
        StageSpan::at(0),
    ))
    .unwrap();
    let e = ledger
        .admit(&plan)
        .expect_err("1600 B resident cannot fit a 900 B budget");
    let r = rejection(&e);
    assert_eq!(r.worst().needed_bytes, 1_600);
    assert_eq!(r.worst().available_bytes, 900);
}

#[test]
fn host_backed_execution_accounts_for_this_requests_own_host_working_set() {
    // Review finding 2. The host headroom subtracted earlier commitments but
    // ignored what this very request already asks the host for.
    let d = gpu(1);
    let mut ledger = Ledger::new([host_snapshot(1_000, 100), device_snapshot(d, 100)]).unwrap();
    let mut plan = PlanRequest::new("spill", ["run"]).unwrap();
    plan.buffer(BufferRequest::new(
        "workspace",
        Scope::Host,
        Tier::Host(HostTier::CpuWorkspace),
        900,
        StageSpan::at(0),
    ))
    .unwrap()
    .buffer(BufferRequest::new("kv", d, KV, 500, StageSpan::at(0)))
    .unwrap();
    let e = ledger.admit(&plan).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::HostBackedExecution),
        "the host is already fully occupied by this same request"
    );
}

#[test]
fn a_context_alternative_must_be_able_to_move_the_binding_peak() {
    // Review finding 3. A scaling buffer that is not live at the binding stage
    // contributes nothing to the peak that failed, so lowering context cannot
    // help however much of it is removed.
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();
    let mut plan = PlanRequest::new("stages", ["prefill", "decode"]).unwrap();
    plan.buffer(BufferRequest::new("fixed", d, WORK, 200, StageSpan::at(0)))
        .unwrap()
        .buffer(BufferRequest::new("kv", d, KV, 50, StageSpan::at(1)).scaling(Scaling::Context))
        .unwrap();
    let e = ledger.admit(&plan).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::LowerContext),
        "removing every KV byte leaves the 200 B prefill peak exactly where it is"
    );

    // The same buffer live at the binding stage does qualify, so this is a test
    // of the liveness rule and not of the alternative being unreachable.
    let mut overlapping = PlanRequest::new("overlapping", ["prefill", "decode"]).unwrap();
    overlapping
        .buffer(BufferRequest::new("fixed", d, WORK, 200, StageSpan::at(0)))
        .unwrap()
        .buffer(
            BufferRequest::new("kv", d, KV, 50, StageSpan::inclusive(0, 1))
                .scaling(Scaling::Context),
        )
        .unwrap();
    let e = ledger.admit(&overlapping).unwrap_err();
    assert!(
        rejection(&e)
            .alternatives
            .contains(&LegalAlternative::LowerContext)
    );
}

#[test]
fn a_branch_alternative_must_be_able_to_move_the_binding_peak() {
    // Review finding 3, the other half: the branch rule shares the code path.
    let d = gpu(1);
    let mut ledger = Ledger::new([device_snapshot(d, 100)]).unwrap();
    let mut plan = PlanRequest::new("stages", ["prefill", "decode"]).unwrap();
    plan.buffer(BufferRequest::new("fixed", d, WORK, 200, StageSpan::at(0)))
        .unwrap()
        .buffer(
            BufferRequest::new("entropy", d, BRANCHES, 50, StageSpan::at(1))
                .scaling(Scaling::Branches),
        )
        .unwrap();
    let e = ledger.admit(&plan).unwrap_err();
    assert!(
        !rejection(&e)
            .alternatives
            .contains(&LegalAlternative::FewerBranches)
    );
}
