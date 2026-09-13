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

/// Fail the next `n` allocations on this thread, run `body`, then serve again.
fn while_failing<T>(n: usize, body: impl FnOnce() -> T) -> T {
    FAIL.with(|f| f.set(n));
    let out = body();
    FAIL.with(|f| f.set(0));
    out
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
            device_cache_resident_bytes: 0,
            device_arena_free_bytes: 0,
            host_workspace_bytes: 1 << 20,
            host_buffer_bytes: 1 << 20,
            host_cache_cap_bytes: union,
            host_cache_leased_bytes: 0,
            host_cache_resident_bytes: 0,
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
    let mut snapshots = Vec::new();
    for _ in 0..7 {
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
    for budget in 0..6 {
        // One snapshot per injection: `close` consumes it, and a
        // `LayerSnapshot` is deliberately not `Clone` -- cloning one would be an
        // infallible allocation inside a step, which is the defect this file is
        // about.
        let snapshot = snapshots.pop().expect("a snapshot per injection");
        let refused = while_failing(1, || snapshot.close(&run, &authority, &ledger));
        assert!(
            refused.is_err(),
            "closing a layer that cannot allocate must refuse (injection {budget})"
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
    println!(
        "every trace entry point refused a failed allocation with a typed error; \
         the step then assembled and reconciled normally"
    );
}
