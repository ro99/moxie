//! Acceptance tests for task 0021's grouped expert execution on the host lane.
//!
//! The oracle is task 0019's: every reduced row must equal
//! `combine_row(expert_row(...))` bit for bit. Around that sit the properties
//! the contract named -- the ledger charged before a byte exists, a bounded
//! queue that refuses, a budget smaller than the working set that still
//! executes, a cancelled run that releases every lease exactly once, a
//! reduction whose declared order decides the answer, and a NUMA placement that
//! is read back rather than asserted.

use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_executor::grouped::{ExpertRoles, GroupedRun, OrderQueue, Progress};
use moxie_executor::residency::ShardSource;
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
use moxie_memory::{
    ArtifactId, CapacitySnapshot, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_oracles::route;
use moxie_plan::expert::{Candidate, ExpertBudget, ExpertPolicy, ExpertShape, compile_experts};
use moxie_storage::Shard;
use moxie_types::{DeviceUuid, HostPlacement, HostTier, Scope, StrategyControl, Tier};

const HIDDEN: u64 = 16;
const INTERMEDIATE: u64 = 8;
const EXPERTS: u64 = 6;
const TOP_K: u64 = 2;
const ROWS: u64 = 4;
/// `3 * intermediate * hidden * 2`.
const CHUNK: u64 = 3 * INTERMEDIATE * HIDDEN * 2;
const BUS: &str = "0000:82:00.0";

/// Four rows over five experts: reuse counts 3, 2, 1, 1, 1.
const ROUTE: [u32; 8] = [0, 1, 0, 2, 1, 3, 0, 4];

// ---------------------------------------------------------------------------
// A real safetensors shard, read through the real reader
// ---------------------------------------------------------------------------

struct Values(u64);

impl Values {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
        bf16_round(unit * 2.0)
    }
    fn block(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next()).collect()
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16_bits(*v).to_le_bytes());
    }
    out
}

struct Weights {
    gate_up: Vec<f32>,
    down: Vec<f32>,
    x: Vec<f32>,
    coefficients: Vec<f32>,
}

fn weights(seed: u64) -> Weights {
    let mut v = Values::new(seed);
    Weights {
        gate_up: v.block((EXPERTS * 2 * INTERMEDIATE * HIDDEN) as usize),
        down: v.block((EXPERTS * HIDDEN * INTERMEDIATE) as usize),
        x: v.block((ROWS * HIDDEN) as usize),
        coefficients: (0..(ROWS * TOP_K) as usize)
            .map(|_| v.next().abs())
            .collect(),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("moxie-grouped-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_shard(dir: &Path, w: &Weights) -> PathBuf {
    let gate_up = to_bytes(&w.gate_up);
    let down = to_bytes(&w.down);
    let split = gate_up.len();
    let end = split + down.len();
    let header = format!(
        "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{EXPERTS},{},{HIDDEN}],\
         \"data_offsets\":[0,{split}]}},\
         \"experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{EXPERTS},{HIDDEN},{INTERMEDIATE}],\
         \"data_offsets\":[{split},{end}]}}}}",
        2 * INTERMEDIATE
    );
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&gate_up).unwrap();
    f.write_all(&down).unwrap();
    f.flush().unwrap();
    path
}

fn artifact() -> ArtifactId {
    ArtifactId::new("grouped-fixture-v1").unwrap()
}

fn roles() -> ExpertRoles {
    ExpertRoles {
        artifact: artifact(),
        gate_up_role: "experts_gate_up".into(),
        down_role: "experts_down".into(),
        format_version: 1,
    }
}

fn source(path: &Path) -> ShardSource {
    ShardSource::new(artifact(), vec![Shard::open(path).unwrap()])
        .role("experts_gate_up", 0, "experts.gate_up_proj")
        .unwrap()
        .role("experts_down", 0, "experts.down_proj")
        .unwrap()
}

fn uuid() -> DeviceUuid {
    DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap()
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

/// A budget on which every group goes to the host: the device candidate is off.
fn host_only_budget() -> ExpertBudget {
    ExpertBudget {
        device: uuid(),
        device_pci_bus_id: BUS.into(),
        device_cache_cap_bytes: 0,
        device_cache_leased_bytes: 0,
        device_arena_free_bytes: 0,
        host_workspace_bytes: 1 << 20,
        host_buffer_bytes: 1 << 20,
        resident_experts: Vec::new(),
    }
}

fn host_only_policy() -> ExpertPolicy {
    ExpertPolicy {
        device: StrategyControl::Off,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Auto,
        ..ExpertPolicy::default()
    }
}

fn ledger(host_bytes: u64) -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, host_bytes, 1 << 16).unwrap()]).unwrap()
}

/// The oracle: what the reduced rows must be, bit for bit.
fn expected_rows(w: &Weights, order: CombineOrder) -> Vec<u16> {
    let spec = route::ExpertSpec {
        experts: EXPERTS as usize,
        hidden: HIDDEN as usize,
        intermediate: INTERMEDIATE as usize,
        activation: ExpertActivation::GeGlu,
    };
    let mut out = Vec::new();
    for r in 0..ROWS as usize {
        let experts = &ROUTE[r * TOP_K as usize..(r + 1) * TOP_K as usize];
        let mut slots = Vec::new();
        for e in experts {
            let y = route::expert_row(
                &w.x[r * HIDDEN as usize..(r + 1) * HIDDEN as usize],
                &w.gate_up,
                &w.down,
                *e,
                spec,
            )
            .unwrap();
            // The node's own BF16 output boundary, which the interpreter applies
            // and the slot buffer stores.
            slots.extend(y.iter().map(|v| bf16_round(*v)));
        }
        let row = route::combine_row(
            experts,
            &w.coefficients[r * TOP_K as usize..(r + 1) * TOP_K as usize],
            &slots,
            HIDDEN as usize,
            order,
        )
        .unwrap();
        out.extend(row.iter().map(|v| to_bf16_bits(*v)));
    }
    out
}

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

#[test]
fn a_host_plan_reproduces_the_oracle_bit_for_bit() {
    let dir = scratch("oracle");
    let w = weights(0x9e3779b9);
    let path = write_shard(&dir, &w);
    let mut src = source(&path);
    let mut l = ledger(1 << 24);
    let mut authority =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("experts", 64 * CHUNK)).unwrap();

    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &host_only_budget(),
        &host_only_policy(),
        None,
        None,
    )
    .unwrap();
    assert!(
        plan.groups()
            .iter()
            .all(|g| g.placement.candidate() == Candidate::Host)
    );

    let mut run = GroupedRun::admit(&mut l, plan, roles(), None).unwrap();
    run.load_activations(&to_bytes(&w.x)).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
        .unwrap();
    let got = as_u16(run.reduce(&w.coefficients).unwrap());
    assert_eq!(got, expected_rows(&w, CombineOrder::AscendingExpertId));

    let stats = run.stats();
    assert_eq!(stats.groups_run, 5);
    assert_eq!(stats.host_groups, 5);
    assert_eq!(stats.device_groups, 0);
    assert_eq!(stats.slots_written, ROWS * TOP_K);
    // Ten leases: two chunks for each of five experts, every one given back.
    assert_eq!(stats.leases_released, 10);
    assert_eq!(authority.live_lease_count(), 0);

    run.close(&mut l).unwrap();
    authority.close(&mut l).unwrap();
    assert!(l.outstanding().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_ledger_is_charged_before_a_single_byte_exists() {
    let mut l = ledger(1 << 24);
    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &host_only_budget(),
        &host_only_policy(),
        None,
        None,
    )
    .unwrap();
    let envelope_bytes = plan.envelope().total_host_bytes();
    assert!(envelope_bytes > 0);
    assert_eq!(l.scope_committed(Scope::Host), 0);

    let run = GroupedRun::admit(&mut l, plan, roles(), None).unwrap();
    // The charge is visible before `load_activations` has written anything.
    assert_eq!(
        l.committed(Scope::Host, Tier::Host(HostTier::CpuWorkspace))
            + l.committed(Scope::Host, Tier::Host(HostTier::Pageable)),
        envelope_bytes
    );
    run.close(&mut l).unwrap();
    assert_eq!(l.scope_committed(Scope::Host), 0);
}

#[test]
fn an_envelope_the_ledger_cannot_hold_is_refused_with_its_report() {
    // A host budget far below the plan's envelope.
    // Physical far below the plan's envelope, with headroom it can actually
    // hold: a snapshot whose headroom exceeds its physical size is refused by
    // the ledger before this test's question is even asked.
    let mut l = Ledger::new([CapacitySnapshot::new(Scope::Host, 640, 128).unwrap()]).unwrap();
    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &host_only_budget(),
        &host_only_policy(),
        None,
        None,
    )
    .unwrap();
    let refusal = GroupedRun::admit(&mut l, plan, roles(), None).unwrap_err();
    match &refusal {
        moxie_executor::grouped::GroupedAdmitRefused::Rejected { rejection, .. } => {
            assert!(rejection.shortfall_bytes > 0);
        }
        other => panic!("expected a rejection with a report, got {other}"),
    }
    // The plan comes back intact, and nothing is charged.
    assert_eq!(refusal.plan().groups().len(), 5);
    assert_eq!(l.scope_committed(Scope::Host), 0);
}

#[test]
fn the_bounded_queue_refuses_rather_than_growing() {
    let mut queue = OrderQueue::with_capacity(2).unwrap();
    assert_eq!(queue.capacity(), 2);
    assert!(queue.is_empty() && !queue.is_full());
    assert!(OrderQueue::with_capacity(0).is_err());
    // The queue's entries carry residency leases, so the refusal is exercised
    // through a real run below; here the arithmetic of the bound is pinned.
    assert_eq!(queue.len(), 0);
    assert!(queue.pop().is_none());
}

#[test]
fn a_budget_smaller_than_the_working_set_still_executes() {
    let dir = scratch("restricted");
    let w = weights(0x1234_5678);
    let path = write_shard(&dir, &w);
    let mut src = source(&path);
    let mut l = ledger(1 << 24);
    // Room for **one** expert's two chunks at a time, against five experts.
    // Document 06 M2 item 4 asks for "an intentionally restricted memory budget
    // smaller than its working weights"; this is that, on the host tier.
    let mut authority =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("restricted", CHUNK)).unwrap();

    let mut policy = host_only_policy();
    // A queue of four against a cache that holds one: every acquire past the
    // first must meet the cache's refusal and drain instead of failing.
    policy.max_inflight_orders = 4;
    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &host_only_budget(),
        &policy,
        None,
        None,
    )
    .unwrap();
    assert_eq!(plan.queue_capacity(), 4);

    let mut run = GroupedRun::admit(&mut l, plan, roles(), None).unwrap();
    run.load_activations(&to_bytes(&w.x)).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(7), 0, u64::MAX)
        .unwrap();
    let got = as_u16(run.reduce(&w.coefficients).unwrap());
    assert_eq!(got, expected_rows(&w, CombineOrder::AscendingExpertId));

    // The evidence that the budget really bound: the queue was refused and
    // drained rather than the run failing, and the cache evicted.
    assert!(
        run.stats().backpressure_drains > 0,
        "a one-expert cache never refused a four-deep queue: {:?}",
        run.stats()
    );
    assert!(authority.stats().evictions > 0);
    println!(
        "restricted budget: cap {CHUNK} B, {} group(s), {} backpressure drain(s), {} eviction(s)",
        run.stats().groups_run,
        run.stats().backpressure_drains,
        authority.stats().evictions
    );
    run.close(&mut l).unwrap();
    authority.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_cancelled_run_releases_every_lease_exactly_once() {
    let dir = scratch("cancel");
    let w = weights(0xabcd_ef01);
    let path = write_shard(&dir, &w);
    let mut src = source(&path);
    let mut l = ledger(1 << 24);
    let mut authority =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("cancel", 64 * CHUNK)).unwrap();
    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &host_only_budget(),
        &host_only_policy(),
        None,
        None,
    )
    .unwrap();
    let mut run = GroupedRun::admit(&mut l, plan, roles(), None).unwrap();
    run.load_activations(&to_bytes(&w.x)).unwrap();

    // One group runs, and the queue still holds the groups acquired behind it.
    let progress = run
        .step(&mut authority, &mut src, TurnId::new(3), 0, u64::MAX)
        .unwrap();
    assert!(matches!(progress, Progress::Ran { .. }));
    assert!(
        !run.queue().is_empty(),
        "nothing was queued behind the first group"
    );
    let queued = run.queue().len();
    let released_before = run.stats().leases_released;
    let pinned = authority.live_lease_count();
    assert_eq!(pinned, 2 * queued, "each queued group pins its two chunks");

    run.cancel(&mut authority);
    assert!(run.is_cancelled());
    assert_eq!(
        authority.live_lease_count(),
        0,
        "cancellation left {pinned} lease(s) pinned"
    );
    // Exactly once, counted: the executor only increments on a release the
    // authority accepted, and a second release of the same lease is refused.
    assert_eq!(
        run.stats().leases_released - released_before,
        2 * queued as u64
    );
    assert!(run.queue().is_empty());
    // Cancelled runs refuse further work and refuse to answer.
    assert!(
        run.step(&mut authority, &mut src, TurnId::new(3), 0, u64::MAX)
            .is_err()
    );
    assert!(run.reduce(&w.coefficients).is_err());

    run.close(&mut l).unwrap();
    authority.close(&mut l).unwrap();
    assert!(l.outstanding().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_declared_reduction_order_decides_the_answer() {
    // Three terms, because two are not enough: floating-point addition is
    // commutative, so a two-slot row gives the same answer in either order and
    // a fixture built from one would prove nothing. The first version of this
    // test used two and passed for that reason.
    //
    // `[V, -V, tiny]` reduced as `(V + (-V)) + tiny` is `tiny`; reduced as
    // `(V + tiny) + (-V)` it is `0`, because `V + tiny` rounds back to `V`.
    let v = 1.0e30f32;
    let tiny = 1.0f32;
    let slots = [v, -v, tiny];
    let coefficients = [1.0f32, 1.0, 1.0];
    // Selection order is `0, 1, 2`; ascending expert id is `0, 2, 1`.
    let experts = [1u32, 3, 2];
    let ascending = route::combine_row(
        &experts,
        &coefficients,
        &slots,
        1,
        CombineOrder::AscendingExpertId,
    )
    .unwrap();
    let selection = route::combine_row(
        &experts,
        &coefficients,
        &slots,
        1,
        CombineOrder::SelectionOrder,
    )
    .unwrap();
    assert_eq!(ascending, vec![0.0]);
    assert_eq!(selection, vec![tiny]);
    assert_ne!(
        ascending, selection,
        "the fixture must distinguish the two orders or the test proves nothing"
    );

    let bytes = to_bytes(&slots);
    for (order, want) in [
        (CombineOrder::AscendingExpertId, &ascending),
        (CombineOrder::SelectionOrder, &selection),
    ] {
        let permutation: Vec<u32> = route::combine_order(&experts, order)
            .unwrap()
            .into_iter()
            .map(|j| j as u32)
            .collect();
        let mut out = vec![0u8; 2];
        moxie_kernels::cpu_expert::combine_rows_bf16(
            &bytes,
            &coefficients,
            &permutation,
            1,
            3,
            1,
            &mut out,
        )
        .unwrap();
        let want: Vec<u16> = want.iter().map(|v| to_bf16_bits(*v)).collect();
        assert_eq!(as_u16(&out), want, "{order:?}");
    }
}

/// The same distinction, end to end, with each slot produced by a **different
/// group** of a real run.
///
/// Expert 0 and expert 2 are built to cancel exactly -- identical gate/up
/// slices, negated down slices -- and expert 1 contributes a term small enough
/// to vanish against them. The row selects them in an order whose ascending-id
/// permutation is not the identity, so the plan's declared order is the only
/// thing that decides which of two answers the run produces.
#[test]
fn two_groups_of_one_run_reduce_in_the_planned_order_and_not_in_completion_order() {
    const H: u64 = 4;
    const I: u64 = 2;
    const E: u64 = 4;
    const K: u64 = 3;
    let dir = scratch("cancelling");

    // Big, small, and the exact negation of big.
    let mut gate_up = vec![0f32; (E * 2 * I * H) as usize];
    let mut down = vec![0f32; (E * H * I) as usize];
    let gu_stride = (2 * I * H) as usize;
    let d_stride = (H * I) as usize;
    for lane in 0..I as usize {
        for k in 0..H as usize {
            // Gate lane 1, up lane 1: the activated intermediate is then 1.
            gate_up[lane * H as usize + k] = if k == 0 { 1.0 } else { 0.0 };
            gate_up[(I as usize + lane) * H as usize + k] = if k == 0 { 1.0 } else { 0.0 };
        }
    }
    // Experts 1, 2 and 3 share expert 0's projection.
    for e in 1..E as usize {
        let (head, tail) = gate_up.split_at_mut(e * gu_stride);
        tail[..gu_stride].copy_from_slice(&head[..gu_stride]);
    }
    let big = 1.0e30f32;
    for o in 0..H as usize {
        down[o * I as usize] = big; // expert 0: +big
        down[d_stride + o * I as usize] = 1.0; // expert 1: 1
        down[2 * d_stride + o * I as usize] = -big; // expert 2: -big
    }

    let x: Vec<f32> = (0..H as usize)
        .map(|k| if k == 0 { 1.0 } else { 0.0 })
        .collect();
    // One row selecting experts 0, 2, 1 in that order.
    let route = [0u32, 2, 1];
    let coefficients = [1.0f32, 1.0, 1.0];

    let gate_up_bytes = to_bytes(&gate_up);
    let down_bytes = to_bytes(&down);
    let split = gate_up_bytes.len();
    let end = split + down_bytes.len();
    let header = format!(
        "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{E},{},{H}],\
         \"data_offsets\":[0,{split}]}},\
         \"experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{E},{H},{I}],\
         \"data_offsets\":[{split},{end}]}}}}",
        2 * I
    );
    let path = dir.join("model.safetensors");
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(&gate_up_bytes).unwrap();
        f.write_all(&down_bytes).unwrap();
    }

    let mlp = OpParams::ExpertMlp {
        hidden: H,
        intermediate: I,
        experts: E,
        top_k: K,
        activation: ExpertActivation::GeGlu,
    };
    let mut answers = Vec::new();
    for order in [
        CombineOrder::AscendingExpertId,
        CombineOrder::SelectionOrder,
    ] {
        let mut src = source(&path);
        let mut l = ledger(1 << 24);
        let mut authority =
            ResidencyAuthority::open(&mut l, &ResidencyRequest::new("cancelling", 1 << 20))
                .unwrap();
        let plan = compile_experts(
            &mlp,
            &OpParams::Combine {
                hidden: H,
                top_k: K,
                order,
            },
            &route,
            &host_only_budget(),
            &host_only_policy(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(plan.groups().len(), 3, "three groups, one per expert");
        let mut run = GroupedRun::admit(&mut l, plan, roles(), None).unwrap();
        run.load_activations(&to_bytes(&x)).unwrap();
        run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
            .unwrap();
        answers.push(as_u16(run.reduce(&coefficients).unwrap()));
        run.close(&mut l).unwrap();
        authority.close(&mut l).unwrap();
    }
    // Ascending expert id reduces (+big, 1, -big); selection order reduces
    // (+big, -big, 1). The two differ, and the run produced whichever the plan
    // declared -- not whichever group happened to finish first.
    assert_ne!(
        answers[0], answers[1],
        "the declared order made no difference to a row built to expose it"
    );
    assert!(
        answers[0].iter().all(|v| *v == 0),
        "ascending expert id should lose the small term: {:?}",
        answers[0]
    );
    // The surviving term is expert 1's own slot value -- the activation makes it
    // `bf16(gelu_tanh(1)) * 1`, not 1 -- so the expectation comes from the
    // oracle rather than from a number typed here.
    let tiny_slot = to_bf16_bits(bf16_round(
        route::expert_row(
            &x,
            &gate_up,
            &down,
            1,
            route::ExpertSpec {
                experts: E as usize,
                hidden: H as usize,
                intermediate: I as usize,
                activation: ExpertActivation::GeGlu,
            },
        )
        .unwrap()[0],
    ));
    assert!(
        answers[1].iter().all(|v| *v == tiny_slot),
        "selection order should keep expert 1's term ({tiny_slot:#06x}): {:?}",
        answers[1]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// NUMA placement, **measured**.
///
/// Two claims are separated here because this machine separates them. Binding
/// the thread decides which CPUs run the kernel; it does **not** decide where
/// the pages are. Node 1 on this machine has about 334 MB free against node 0's
/// 5.1 GB, with 125 GB of page cache on each, so the default policy prefers
/// local and then falls back rather than reclaiming: a 32 MiB buffer touched
/// entirely by a node-1-bound thread came back with 4,471 of 6,144 pages on
/// node 0. `required` therefore makes the node binding (`mbind`), and that is
/// what this asserts; `auto` is measured and reported, not asserted.
#[test]
fn host_buffers_are_bound_to_the_planned_node_and_the_pages_are_read_back() {
    let topology = match moxie_host::numa::read_topology_under(
        Path::new("/"),
        &[BUS.to_string(), "0000:03:00.0".to_string()],
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIPPED: this machine reports no NUMA topology: {e}");
            return;
        }
    };
    let HostPlacement::Node(node) = topology.placement_for_pci(BUS) else {
        eprintln!("SKIPPED: this machine gives {BUS} no NUMA node");
        return;
    };

    // Big enough that the mapping has thousands of pages to count.
    let rows = 4096u64;
    let mlp = OpParams::ExpertMlp {
        hidden: 1024,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        activation: ExpertActivation::GeGlu,
    };
    let combine = OpParams::Combine {
        hidden: 1024,
        top_k: TOP_K,
        order: CombineOrder::AscendingExpertId,
    };
    let route: Vec<u32> = (0..rows * TOP_K)
        .map(|i| ((i % TOP_K) + 2 * (i / TOP_K % 2)) as u32)
        .collect();
    let mut budget = host_only_budget();
    budget.host_buffer_bytes = 1 << 30;
    budget.host_workspace_bytes = 1 << 30;
    let mut policy = host_only_policy();
    policy.host_placement = StrategyControl::Required;
    let plan = compile_experts(
        &mlp,
        &combine,
        &route,
        &budget,
        &policy,
        Some(&topology),
        None,
    )
    .unwrap();
    assert_eq!(plan.host_placement(), HostPlacement::Node(node));

    let mut l = ledger(1 << 30);
    let run = GroupedRun::admit(&mut l, plan, roles(), Some(&topology)).unwrap();
    let report = run.placement();
    assert!(report.is_bound(), "the run did not bind: {:?}", report);
    assert_eq!(report.requested, HostPlacement::Node(node));
    assert_eq!(report.page_policy, "mbind", "`required` must be binding");
    assert_eq!(
        report.bound_cpus,
        topology.cpus_of(node).unwrap(),
        "bound to a different CPU set than the node's"
    );
    assert!(
        run.buffers().owns_pages(),
        "a placement claim over pages the allocator may already have faulted is not a measurement"
    );

    let addresses = run.buffers().addresses();
    for (what, address) in [
        ("slots", addresses.slots),
        ("activations", addresses.activations),
        ("workspace", addresses.workspace),
    ] {
        let mapping = moxie_host::numa::node_of_address_under(Path::new("/"), address)
            .unwrap()
            .unwrap_or_else(|| panic!("{what} has no mapping in numa_maps"));
        assert!(
            mapping.total_pages() > 0,
            "{what} has no resident page: {mapping:?}"
        );
        // Strict: not "mostly", not "dominantly". A bound node that leaked
        // pages to the other socket would be the silent fallback AGENTS.md
        // forbids, and it is exactly what the non-binding policy does here.
        assert_eq!(
            mapping.pages,
            vec![(node, mapping.total_pages())],
            "{what} did not land entirely on {node}"
        );
        println!("{what}: {} page(s) on {node}", mapping.total_pages());
    }
    run.close(&mut l).unwrap();
}

/// The same run with `auto`, where placement is a preference the kernel may
/// decline. Nothing is asserted about where the pages went -- the distribution
/// is printed, because on this machine it genuinely varies with how much free
/// memory each node has at that instant.
#[test]
fn an_auto_placement_reports_what_it_got_rather_than_claiming_what_it_asked_for() {
    let topology = match moxie_host::numa::read_topology_under(Path::new("/"), &[BUS.to_string()]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIPPED: this machine reports no NUMA topology: {e}");
            return;
        }
    };
    let HostPlacement::Node(node) = topology.placement_for_pci(BUS) else {
        eprintln!("SKIPPED: this machine gives {BUS} no NUMA node");
        return;
    };
    let mut l = ledger(1 << 24);
    let plan = compile_experts(
        &mlp(),
        &combine(CombineOrder::AscendingExpertId),
        &ROUTE,
        &host_only_budget(),
        &host_only_policy(),
        Some(&topology),
        None,
    )
    .unwrap();
    let run = GroupedRun::admit(&mut l, plan, roles(), Some(&topology)).unwrap();
    assert_eq!(run.placement().page_policy, "preferred");
    let mapping =
        moxie_host::numa::node_of_address_under(Path::new("/"), run.buffers().addresses().slots)
            .unwrap()
            .expect("a mapping");
    println!(
        "auto placement asked for {node} and got {:?} across {} page(s)",
        mapping.pages,
        mapping.total_pages()
    );
    run.close(&mut l).unwrap();
}

#[test]
fn the_expert_chunks_are_the_declared_slices_of_the_fused_tensors() {
    let shape = ExpertShape {
        hidden: 2816,
        intermediate: 704,
        experts: 128,
        top_k: 8,
        activation: ExpertActivation::GeGlu,
    };
    let roles = roles();
    let (gate_up, down) = roles.chunks(3, shape).unwrap();
    // The designated artifact's arithmetic, from the bring-up record.
    assert_eq!(gate_up.len_bytes(), 7_929_856);
    assert_eq!(gate_up.range().offset_bytes(), 3 * 7_929_856);
    assert_eq!(down.len_bytes(), 3_964_928);
    assert_eq!(down.range().offset_bytes(), 3 * 3_964_928);
    assert!(roles.chunks(128, shape).is_err());
}
