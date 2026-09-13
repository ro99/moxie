//! Isolated executable: what a traced layer costs the heap, layer after layer.
//!
//! M2's exit asks that a real out-of-device-memory working set execute "without
//! OOM or **hidden allocations**". The gate here is the strongest proxy this
//! workspace has for that sentence, and it is a number rather than an absence:
//! **a layer's allocation count does not grow with the layers before it.**
//!
//! That is the property worth holding. A per-layer cost that is constant means
//! the step's heap is bounded by one layer's record however long the step runs;
//! a cost that grows by one allocation per layer is a leak that a thirty-layer
//! test would pass and a three-hundred-layer session would not. The absolute
//! number is printed, because a count is a measurement and a measurement belongs
//! in the output rather than in a sentence.
//!
//! The counter is thread-local for the reason task 0022's third review
//! established: a process-wide one measures whatever else `libtest` is running
//! in parallel, and requiring `--test-threads=1` fixes the flake by making the
//! gate everyone actually runs unreliable.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_executor::grouped::{ExpertRoles, GroupedRun};
use moxie_executor::residency::ShardSource;
use moxie_executor::trace::{LayerSnapshot, LayerTrace, StepTrace};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::to_bf16_bits;
use moxie_memory::{
    ArtifactId, CapacitySnapshot, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_plan::expert::{ExpertBudget, ExpertPolicy, ResidentChunks, compile_experts};
use moxie_storage::Shard;
use moxie_types::{DeviceUuid, Scope, StrategyControl};

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

fn calls() -> usize {
    CALLS.try_with(Cell::get).unwrap_or(0)
}

struct Counter;
// SAFETY: allocation operations are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the allocator caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            // `try_with` and no allocation of its own: this runs inside the
            // allocator, and a counter that allocated would recurse.
            let _ = CALLS.try_with(|c| c.set(c.get() + 1));
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

const HIDDEN: u64 = 16;
const INTERMEDIATE: u64 = 8;
const EXPERTS: u64 = 6;
const TOP_K: u64 = 2;
const ROWS: u64 = 4;
const CHUNK: u64 = 3 * INTERMEDIATE * HIDDEN * 2;
const LAYERS: u32 = 8;
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
    let mut state = 0x0023_7777u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5
    };
    let mut payload = Vec::new();
    let mut entries = Vec::new();
    for layer in 0..LAYERS {
        for (name, len, shape) in [
            (
                "gate_up",
                (EXPERTS * 2 * INTERMEDIATE * HIDDEN) as usize,
                format!("[{EXPERTS},{},{HIDDEN}]", 2 * INTERMEDIATE),
            ),
            (
                "down",
                (EXPERTS * HIDDEN * INTERMEDIATE) as usize,
                format!("[{EXPERTS},{HIDDEN},{INTERMEDIATE}]"),
            ),
        ] {
            let block: Vec<f32> = (0..len).map(|_| next()).collect();
            let bytes = to_bytes(&block);
            let start = payload.len();
            payload.extend_from_slice(&bytes);
            entries.push(format!(
                "\"layers.{layer}.{name}_proj\":{{\"dtype\":\"BF16\",\"shape\":{shape},\
                 \"data_offsets\":[{start},{}]}}",
                payload.len()
            ));
        }
    }
    let header = format!("{{{}}}", entries.join(","));
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&payload).unwrap();
    f.flush().unwrap();
    let x: Vec<f32> = (0..(ROWS * HIDDEN)).map(|_| next()).collect();
    (path, to_bytes(&x))
}

fn artifact() -> ArtifactId {
    ArtifactId::new("step-allocation-fixture-v1").unwrap()
}

fn source(path: &Path) -> ShardSource {
    let mut src = ShardSource::new(artifact(), vec![Shard::open(path).unwrap()]);
    for layer in 0..LAYERS {
        src = src
            .role(
                format!("experts_gate_up.{layer}"),
                0,
                format!("layers.{layer}.gate_up_proj"),
            )
            .unwrap()
            .role(
                format!("experts_down.{layer}"),
                0,
                format!("layers.{layer}.down_proj"),
            )
            .unwrap();
    }
    src
}

/// A whole working set of `LAYERS` identical layers, measuring each one.
#[test]
fn a_traced_layer_costs_the_same_however_many_came_before_it() {
    let dir = std::env::temp_dir().join(format!("moxie-step-alloc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (path, x) = write_shard(&dir);
    let mut src = source(&path);

    let union = {
        let mut all = ROUTE.to_vec();
        all.sort_unstable();
        all.dedup();
        all.len() as u64 * CHUNK
    };
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 26, 1 << 20).unwrap()]).unwrap();
    let mut authority =
        ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new("step", union)).unwrap();

    let mlp = OpParams::ExpertMlp {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        activation: ExpertActivation::GeGlu,
    };
    let combine = OpParams::Combine {
        hidden: HIDDEN,
        top_k: TOP_K,
        order: CombineOrder::AscendingExpertId,
        output_scale: 1.0,
    };
    let policy = ExpertPolicy {
        device: StrategyControl::Off,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        ..ExpertPolicy::default()
    };

    let mut per_layer = Vec::new();
    let mut traces: Vec<LayerTrace> = Vec::new();
    for layer in 0..LAYERS {
        let budget = ExpertBudget {
            device: DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap(),
            device_pci_bus_id: BUS.into(),
            device_cache_cap_bytes: 0,
            device_cache_leased_bytes: 0,
            device_arena_free_bytes: 0,
            host_workspace_bytes: 1 << 20,
            host_buffer_bytes: 1 << 20,
            host_cache_cap_bytes: union,
            host_cache_leased_bytes: 0,
            resident: ResidentChunks::none(),
        };
        let roles = ExpertRoles {
            artifact: artifact(),
            gate_up_role: format!("experts_gate_up.{layer}"),
            down_role: format!("experts_down.{layer}"),
            format_version: 1,
        };

        let before = calls();
        let plan = compile_experts(&mlp, &combine, &ROUTE, &budget, &policy, None, None).unwrap();
        let snapshot = LayerSnapshot::take(layer, &authority);
        let mut run = GroupedRun::admit(&mut ledger, plan, roles, None).unwrap();
        run.load_activations(&x).unwrap();
        run.run_to_completion(
            &mut authority,
            &mut src,
            TurnId::new(u64::from(layer) + 1),
            0,
            u64::MAX,
        )
        .unwrap();
        let trace = snapshot.close(&run, &authority, &ledger);
        run.close(&mut ledger).unwrap();
        authority.end_turn(TurnId::new(u64::from(layer) + 1));
        let after = calls();
        per_layer.push(after - before);
        traces.push(trace);
    }

    println!("traced layer heap requests, layer by layer: {per_layer:?}");

    // The first two layers are the warm-up: layer 0 admits into an empty cache
    // and layer 1 is the first to evict. From layer 2 the work is identical, so
    // the cost must be too.
    let steady = &per_layer[2..];
    let first = steady[0];
    assert!(
        steady.iter().all(|n| *n == first),
        "a traced layer's heap cost changes with the layers before it: {per_layer:?}"
    );
    println!(
        "steady state: {first} heap request(s) per traced layer, identical across layers 2..{}",
        LAYERS
    );

    // And the step's own assembly, measured separately, because a trace that was
    // cheap per layer and expensive to close would still be an unbounded step.
    authority.retire_all(Scope::Host);
    authority.close(&mut ledger).unwrap();
    let before = calls();
    let step = StepTrace::new(
        artifact().as_str(),
        "allocation probe",
        traces,
        &authority,
        &ledger,
    );
    let assembled = calls() - before;
    let report = step.reconcile().expect("the probe's trace reconciles");
    println!(
        "step assembly: {assembled} heap request(s) for {} layer(s); reconcile checked {} \
         equalities",
        step.layers.len(),
        report.equalities_checked
    );
    assert!(
        assembled < 8 * LAYERS as usize,
        "assembling a step trace cost {assembled} allocation(s) for {LAYERS} layer(s)"
    );
}
