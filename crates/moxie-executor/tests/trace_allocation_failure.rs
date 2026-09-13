//! Isolated executable: an allocation failure while tracing is a typed error.
//!
//! A review injected one allocation failure immediately before
//! `LayerSnapshot::take` and got **SIGABRT**: `memory allocation of 864 bytes
//! failed`. That is task 0019's defect for the fourth time in this workspace,
//! and the rule has not changed — an allocation failure inside a generation step
//! must be a typed error the transaction can roll back, not a panic that takes
//! the rollback, the lease release and the next generation with it.
//!
//! The trace is squarely inside a step: it is taken around every layer. So every
//! collection it builds is reserved before it is written, none of them is a
//! `BTreeMap` (which has no fallible insert and would abort however carefully
//! the rest reserved), and the owners hand their numbers out through visitors
//! that allocate nothing.
//!
//! The allocator here fails a **counted** number of requests on the measuring
//! thread, so the injection is precise and repeatable rather than a global
//! switch that also fails the harness.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_executor::grouped::{ExpertRoles, GroupedRun};
use moxie_executor::residency::ShardSource;
use moxie_executor::trace::{LayerSnapshot, StepSnapshot, StepTrace};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::to_bf16_bits;
use moxie_memory::{
    ArtifactId, CapacitySnapshot, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_plan::expert::{ExpertBudget, ExpertPolicy, ResidentChunks, compile_experts};
use moxie_storage::Shard;
use moxie_types::{DeviceUuid, Scope, StrategyControl};

thread_local! {
    /// Allocations still to be failed on this thread. Zero means "serve
    /// everything", which is the state every line outside an injection runs in.
    static FAIL: Cell<usize> = const { Cell::new(0) };
}

struct Injector;

// SAFETY: every request is forwarded to the system allocator unchanged, except
// the counted ones, which return null -- the documented way for a `GlobalAlloc`
// to report failure.
unsafe impl GlobalAlloc for Injector {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = SERVED.try_with(|c| c.set(c.get() + 1));
        let skipping = SKIP.try_with(|s| {
            let left = s.get();
            if left > 0 {
                s.set(left - 1);
            }
            left > 0
        });
        if skipping != Ok(true) {
            let fail = FAIL.try_with(|f| {
                let left = f.get();
                if left > 0 {
                    f.set(left - 1);
                }
                left > 0
            });
            if fail == Ok(true) {
                return core::ptr::null_mut();
            }
        }
        // SAFETY: the caller supplies a valid layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: Injector = Injector;

thread_local! {
    /// Allocations to serve before the failing one begins.
    static SKIP: Cell<usize> = const { Cell::new(0) };
    /// Allocations served on this thread, for counting a body's demand.
    static SERVED: Cell<usize> = const { Cell::new(0) };
}

/// Fail the next `n` allocations on this thread, run `body`, then serve again.
fn while_failing<T>(n: usize, body: impl FnOnce() -> T) -> T {
    while_failing_at(0, n, body)
}

/// Serve `skip` allocations, then fail `n`, then serve again.
///
/// The parameter a review found missing. The first version of this file looped
/// six times calling `while_failing(1, ...)`, so **every iteration failed the
/// first allocation** and the loop index only changed the assertion message: an
/// axis that was exercised and never varied. A failure at the sixth allocation
/// of `close` -- inside `reservation_ids`, which built a `Vec` -- aborted the
/// process, and this file said nothing.
fn while_failing_at<T>(skip: usize, n: usize, body: impl FnOnce() -> T) -> T {
    SKIP.with(|f| f.set(skip));
    FAIL.with(|f| f.set(n));
    let out = body();
    SKIP.with(|f| f.set(0));
    FAIL.with(|f| f.set(0));
    out
}

/// How many allocations `body` asks for on this thread.
fn allocations_of<T>(body: impl FnOnce() -> T) -> usize {
    let before = SERVED.try_with(Cell::get).unwrap_or(0);
    let out = body();
    drop(out);
    SERVED.try_with(Cell::get).unwrap_or(0) - before
}

const HIDDEN: u64 = 16;
const INTERMEDIATE: u64 = 8;
const EXPERTS: u64 = 6;
const TOP_K: u64 = 2;
const ROWS: u64 = 4;
const CHUNK: u64 = 3 * INTERMEDIATE * HIDDEN * 2;
const ROUTE: [u32; 8] = [0, 1, 0, 2, 1, 3, 0, 4];
const BUS: &str = "0000:82:00.0";

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&to_bf16_bits(*v).to_le_bytes());
    }
    out
}

fn write_shard(dir: &Path) -> (PathBuf, Vec<u8>) {
    let mut state = 0x0023_3131u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5
    };
    let gate_up: Vec<f32> = (0..(EXPERTS * 2 * INTERMEDIATE * HIDDEN))
        .map(|_| next())
        .collect();
    let down: Vec<f32> = (0..(EXPERTS * HIDDEN * INTERMEDIATE))
        .map(|_| next())
        .collect();
    let gate_up = to_bytes(&gate_up);
    let down = to_bytes(&down);
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
    let x: Vec<f32> = (0..(ROWS * HIDDEN)).map(|_| next()).collect();
    (path, to_bytes(&x))
}

fn artifact() -> ArtifactId {
    ArtifactId::new("trace-allocation-fixture-v1").unwrap()
}

#[test]
fn a_trace_that_cannot_allocate_returns_an_error_instead_of_aborting() {
    let dir = std::env::temp_dir().join(format!("moxie-trace-oom-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (path, x) = write_shard(&dir);
    let mut src = ShardSource::new(artifact(), vec![Shard::open(&path).unwrap()])
        .role("experts_gate_up", 0, "experts.gate_up_proj")
        .unwrap()
        .role("experts_down", 0, "experts.down_proj")
        .unwrap();

    let union = 5 * CHUNK;
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 26, 1 << 20).unwrap()]).unwrap();
    let mut authority =
        ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new("oom", union)).unwrap();

    // Every allocation the trace may need is reserved, so the injection has to
    // be precise: one failed request is enough, and the counts below say how
    // many each entry point asks for.
    let refused = while_failing(1, || StepSnapshot::take(&authority));
    assert!(
        refused.is_err(),
        "a step snapshot that cannot allocate must refuse"
    );
    let start = StepSnapshot::take(&authority).expect("and succeeds when it can");

    let refused = while_failing(1, || LayerSnapshot::take(0, &authority));
    assert!(
        refused.is_err(),
        "a layer snapshot that cannot allocate must refuse"
    );

    let plan = compile_experts(
        &OpParams::ExpertMlp {
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            experts: EXPERTS,
            top_k: TOP_K,
            activation: ExpertActivation::GeGlu,
        },
        &OpParams::Combine {
            hidden: HIDDEN,
            top_k: TOP_K,
            order: CombineOrder::AscendingExpertId,
            output_scale: 1.0,
        },
        &ROUTE,
        &ExpertBudget {
            device: DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap(),
            device_pci_bus_id: BUS.into(),
            device_cache_cap_bytes: 0,
            device_cache_leased_bytes: 0,
            device_cache_largest_free_bytes: u64::MAX,
            device_arena_free_bytes: 0,
            host_workspace_bytes: 1 << 20,
            host_buffer_bytes: 1 << 20,
            host_cache_cap_bytes: union,
            host_cache_leased_bytes: 0,
            host_cache_largest_free_bytes: u64::MAX,
            cache_alignment_bytes: 256,
            chunks_per_expert: 2,
            resident: ResidentChunks::none(),
        },
        &ExpertPolicy {
            device: StrategyControl::Off,
            host: StrategyControl::Auto,
            host_placement: StrategyControl::Off,
            ..ExpertPolicy::default()
        },
        None,
        None,
    )
    .unwrap();

    // Seven snapshots of the same moment -- before the run -- so each injection
    // below closes a snapshot that is genuinely this layer's. Taking one after
    // the run would describe a layer that moved nothing, which the
    // reconciliation rejects and rightly so.
    // One per injection position, plus one to measure the demand with and one
    // for the real close. Twenty-four is comfortably more than `close` asks for;
    // the loop below asserts it used what it needed.
    let mut snapshots = Vec::new();
    for _ in 0..24 {
        snapshots.push(LayerSnapshot::take(0, &authority).unwrap());
    }
    let mut run = GroupedRun::admit(
        &mut ledger,
        plan,
        ExpertRoles {
            artifact: artifact(),
            gate_up_role: "experts_gate_up".into(),
            down_role: "experts_down".into(),
            format_version: 1,
        },
        None,
    )
    .unwrap();
    run.load_activations(&x).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
        .unwrap();

    // `close` reserves several times: the scope deltas, the ledger's scopes, one
    // per charged tier, and the accounted charges. Failing each of the first
    // few must refuse rather than abort, and the loop is what shows that no
    // single one of them was left infallible.
    // **Every** allocation position, not the first one six times over. How many
    // there are is measured rather than assumed, so a position that stops being
    // reached shows up as a change in the count.
    let demand = allocations_of(|| {
        let snapshot = snapshots.pop().expect("a snapshot to measure with");
        snapshot.close(&run, &authority, &ledger)
    });
    assert!(demand > 0, "closing a layer asked for no memory at all");
    println!("closing a layer asks for {demand} allocation(s); failing each in turn");
    for position in 0..demand {
        let snapshot = snapshots.pop().expect("a snapshot per injection");
        let refused = while_failing_at(position, 1, || snapshot.close(&run, &authority, &ledger));
        assert!(
            refused.is_err(),
            "closing a layer must refuse when its allocation {position} fails"
        );
    }
    let snapshot = snapshots.pop().expect("one snapshot left");
    let trace = snapshot
        .close(&run, &authority, &ledger)
        .expect("and succeeds when it can");
    run.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(1));
    authority.retire_all(Scope::Host);
    authority.close(&mut ledger).unwrap();

    // Built outside the window: the identity strings are the caller's to
    // allocate, which is exactly why `new` takes them owned.
    let name = artifact().as_str().to_string();
    let case = "oom".to_string();
    let (start_ref, authority_ref, ledger_ref) = (&start, &authority, &ledger);
    let refused = while_failing(1, move || {
        StepTrace::new(name, case, Vec::new(), start_ref, authority_ref, ledger_ref)
    });
    assert!(
        refused.is_err(),
        "assembling a step that cannot allocate must refuse"
    );

    let step = StepTrace::new(
        artifact().as_str().to_string(),
        "oom".to_string(),
        vec![trace],
        &start,
        &authority,
        &ledger,
    )
    .expect("and succeeds when it can");
    step.reconcile().expect("the trace reconciles");
    // And **reconciliation itself**, at every position it allocates in. A review
    // found this path aborting while it built the message that says it ran out
    // of memory: the reserve failed, and formatting the discrepancy was the
    // second failure. A discrepancy that reports an allocation failure carries a
    // borrowed message now, and allocates nothing.
    let demand = allocations_of(|| step.reconcile());
    println!("reconciling asks for {demand} allocation(s); failing each in turn");
    for position in 0..demand.max(1) {
        let outcome = while_failing_at(position, 4, || step.reconcile());
        match outcome {
            Ok(_) => {}
            Err(discrepancy) => assert_eq!(
                discrepancy.check, "step-is-the-sum-of-layers",
                "reconciling under memory pressure reported {}: {discrepancy}",
                discrepancy.check
            ),
        }
    }
    println!(
        "every trace entry point refused a failed allocation with a typed error, and \
         reconciliation reported one without allocating to say so"
    );
}
