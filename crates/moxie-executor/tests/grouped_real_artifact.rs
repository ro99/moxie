//! One layer's routed expert block of the designated artifact, executed.
//!
//! **This is not model support and no quality claim follows from it.** The
//! activations are synthetic and the route is written here, so what this shows
//! is that the machinery -- demand residency, a restricted device budget, a
//! bounded queue, the grouped kernel and the declared reduction -- carries real
//! expert weights from disk to a computed answer. Output quality is O2 and needs
//! paired output against the released model.
//!
//! The oracle is the **host candidate over the same bytes**: the CPU kernel is
//! bit-identical to task 0019's reference, proven separately, so running the
//! same plan on both candidates and comparing the slot buffers checks the device
//! path against a reference without paying for an FP64 oracle over 2,816 by 704
//! tensors in a debug build.
//!
//! Skipped, loudly, when the artifact is not on this machine.
#![cfg(feature = "driver")]

use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use moxie_cuda::RankContext;
use moxie_executor::grouped::{ExpertRoles, GroupedRun};
use moxie_executor::residency::{DeviceResidency, ShardSource};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_memory::{
    ArtifactId, CapacitySnapshot, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_plan::expert::{Candidate, ExpertBudget, ExpertKernels, ExpertPolicy, compile_experts};
use moxie_storage::Shard;
use moxie_types::{RankId, Scope, StrategyControl};

/// The designated BF16 MoE. A read-only input: nothing here writes, copies,
/// converts or downloads anything.
const ARTIFACT: &str = "/fast/models/google/gemma-4-26B-A4B-it";
const GATE_UP: &str = "model.language_model.layers.0.experts.gate_up_proj";
const DOWN: &str = "model.language_model.layers.0.experts.down_proj";

const HIDDEN: u64 = 2816;
const INTERMEDIATE: u64 = 704;
const EXPERTS: u64 = 128;
const TOP_K: u64 = 8;
const ROWS: u64 = 2;
const GATE_UP_PER_EXPERT: u64 = 1408 * HIDDEN * 2;
const DOWN_PER_EXPERT: u64 = HIDDEN * INTERMEDIATE * 2;
const EXPERT_TOTAL: u64 = GATE_UP_PER_EXPERT + DOWN_PER_EXPERT;
const MIB: u64 = 1024 * 1024;

/// A `RankContext` is exclusive per device and `cargo test` runs a binary's
/// tests in parallel threads. Serialising them is not a workaround: the
/// exclusivity is the property task 0007 established deliberately.
static DEVICE: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Two rows of top-k 8 sharing six experts: ten distinct experts, not sixteen.
const ROUTE: [u32; 16] = [
    3, 17, 42, 91, 5, 60, 118, 7, //
    3, 17, 42, 99, 5, 60, 118, 11,
];

fn artifact_id() -> ArtifactId {
    // The artifact's own index digest, recorded in the bring-up record. Not a
    // path: the authority may not learn what one is.
    ArtifactId::new("sha256:907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7")
        .unwrap()
}

fn roles() -> ExpertRoles {
    ExpertRoles {
        artifact: artifact_id(),
        gate_up_role: "experts_gate_up".into(),
        down_role: "experts_down".into(),
        format_version: 1,
    }
}

fn mlp() -> OpParams {
    OpParams::ExpertMlp {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        // The artifact declares `gelu_pytorch_tanh`.
        activation: ExpertActivation::GeGlu,
    }
}

fn combine() -> OpParams {
    OpParams::Combine {
        hidden: HIDDEN,
        top_k: TOP_K,
        order: CombineOrder::AscendingExpertId,
    }
}

fn source(shard_path: &Path) -> ShardSource {
    ShardSource::new(artifact_id(), vec![Shard::open(shard_path).unwrap()])
        .role("experts_gate_up", 0, GATE_UP)
        .unwrap()
        .role("experts_down", 0, DOWN)
        .unwrap()
}

/// Deterministic BF16 activations. Synthetic, and that is the whole caveat.
fn activations() -> Vec<u8> {
    let mut state = 0x0021_2026_u64;
    let mut out = Vec::with_capacity((ROWS * HIDDEN * 2) as usize);
    for _ in 0..(ROWS * HIDDEN) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
        out.extend_from_slice(&moxie_kernels::cpu_expert::to_bf16_bits(unit).to_le_bytes());
    }
    out
}

/// The declared amortisation threshold has to be raised for this layer, and
/// that is a fact worth stating rather than a knob worth hiding.
///
/// One expert of this artifact is 11,894,784 B. Two rows share six of the ten
/// experts, so the best reuse here is two rows -- 5,947,392 B per row, against
/// the declared default of 1 MiB. At decode batch sizes this artifact's experts
/// do **not** amortise their transfer at the default policy, and
/// `the_default_policy_sends_this_layer_to_the_cpu` asserts exactly that.
fn device_policy() -> ExpertPolicy {
    ExpertPolicy {
        device: StrategyControl::Required,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        max_transfer_bytes_per_row: EXPERT_TOTAL,
        ..ExpertPolicy::default()
    }
}

fn budget(device: moxie_types::DeviceUuid, bus: String, cache: u64, arena: u64) -> ExpertBudget {
    ExpertBudget {
        device,
        device_pci_bus_id: bus,
        device_cache_cap_bytes: cache,
        device_cache_leased_bytes: 0,
        device_arena_free_bytes: arena,
        host_workspace_bytes: MIB,
        host_buffer_bytes: 4 * MIB,
        resident_experts: Vec::new(),
    }
}

/// What the declared policy actually decides for this artifact at this batch
/// size, asserted rather than assumed.
#[test]
fn the_default_policy_sends_this_layers_experts_to_the_cpu() {
    let _serial = one_at_a_time();
    let dir = Path::new(ARTIFACT);
    if !dir.is_dir() {
        println!("SKIPPED: {ARTIFACT} is not present on this machine");
        return;
    }
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ctx = RankContext::acquire(RankId(0), 0).unwrap();
    let catalogue = moxie_kernels::expert_mlp_catalogue();
    let plan = compile_experts(
        &mlp(),
        &combine(),
        &ROUTE,
        &budget(
            ctx.uuid(),
            ctx.capability().pci_bus_id.clone(),
            8 * EXPERT_TOTAL,
            64 * MIB,
        ),
        &ExpertPolicy {
            host_placement: StrategyControl::Off,
            ..ExpertPolicy::default()
        },
        None,
        Some(ExpertKernels {
            capability: ctx.capability(),
            catalogue: &catalogue,
        }),
    )
    .unwrap();
    assert!(
        plan.groups()
            .iter()
            .all(|g| g.placement().candidate() == Candidate::Host),
        "the default 1 MiB/row threshold should refuse an 11.9 MB expert over two rows"
    );
    let worst = plan
        .groups()
        .iter()
        .map(|g| g.decision().bytes_per_row)
        .max()
        .unwrap();
    println!(
        "the designated artifact's layer 0 at {ROWS} row(s): up to {worst} B/row transferred,          against the declared default of {} B/row -- every expert goes to the CPU candidate",
        ExpertPolicy::default().max_transfer_bytes_per_row
    );
}

#[test]
fn a_real_layers_experts_execute_on_the_gpu_and_agree_with_the_cpu_candidate() {
    let _serial = one_at_a_time();
    let dir = Path::new(ARTIFACT);
    if !dir.is_dir() {
        println!(
            "SKIPPED: {ARTIFACT} is not present on this machine; the grouped expert path is \
             exercised against the synthetic fixtures in grouped_device.rs"
        );
        return;
    }
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let shard_path = dir.join("model-00001-of-00002.safetensors");
    assert!(shard_path.is_file(), "{} is missing", shard_path.display());
    {
        let shard = Shard::open(&shard_path).unwrap();
        assert_eq!(
            shard.header().get(GATE_UP).unwrap().len(),
            EXPERTS * GATE_UP_PER_EXPERT
        );
        assert_eq!(
            shard.header().get(DOWN).unwrap().len(),
            EXPERTS * DOWN_PER_EXPERT
        );
    }

    let ctx = RankContext::acquire(RankId(0), 0).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();
    let x = activations();

    // **Two** experts on the device against the ten this layer demands, with a
    // queue four deep. Eviction runs throughout and the queue is refused, which
    // is document 06 M2 item 4's "intentionally restricted memory budget
    // smaller than its working weights" met by the two mechanisms that are
    // supposed to meet it. A four-expert cache also works and never refuses the
    // queue, which is why it is not what this case uses.
    let device_cache = 2 * EXPERT_TOTAL;
    let host_cache = 6 * EXPERT_TOTAL;

    // --- the device candidate -------------------------------------------------
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 512 * MIB, 16 * MIB).unwrap(),
        CapacitySnapshot::new(scope, 512 * MIB, 16 * MIB).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("real experts", host_cache).device(ctx.uuid(), device_cache),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();
    let mut src = source(&shard_path);

    let plan = compile_experts(
        &mlp(),
        &combine(),
        &ROUTE,
        &budget(
            ctx.uuid(),
            ctx.capability().pci_bus_id.clone(),
            device_cache,
            64 * MIB,
        ),
        &device_policy(),
        None,
        Some(ExpertKernels {
            capability: ctx.capability(),
            catalogue: &catalogue,
        }),
    )
    .unwrap();
    assert_eq!(
        plan.groups().len(),
        10,
        "ten distinct experts over two rows"
    );
    assert!(
        plan.groups()
            .iter()
            .all(|g| g.placement().candidate() == Candidate::Device)
    );
    assert_eq!(plan.envelope().residency_demand_bytes, 10 * EXPERT_TOTAL);

    let started = Instant::now();
    let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
    run.attach_device(
        &mut ledger,
        &ctx,
        &mut residency,
        moxie_kernels::EXPERT_MLP_FATBIN,
    )
    .unwrap();
    run.load_activations(&x).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
        .unwrap();
    let device_slots = run.buffers().slots().to_vec();
    let device_stats = run.stats();
    let residency_stats = authority.stats();
    let device_elapsed = started.elapsed();
    run.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(1));
    authority.retire_all(scope);
    residency.close(&mut authority).unwrap();
    authority.close(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());

    // --- the host candidate, over the same bytes ------------------------------
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 512 * MIB, 16 * MIB).unwrap()]).unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("real experts on the host", host_cache),
    )
    .unwrap();
    let mut src = source(&shard_path);
    let plan = compile_experts(
        &mlp(),
        &combine(),
        &ROUTE,
        &budget(ctx.uuid(), ctx.capability().pci_bus_id.clone(), 0, 0),
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
    let started = Instant::now();
    let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
    run.load_activations(&x).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(2), 0, u64::MAX)
        .unwrap();
    let host_slots = run.buffers().slots().to_vec();
    let host_elapsed = started.elapsed();
    run.close(&mut ledger).unwrap();
    authority.close(&mut ledger).unwrap();

    assert_eq!(
        device_slots, host_slots,
        "the two candidates disagree on the same real expert bytes"
    );
    assert_eq!(device_stats.device_groups, 10);
    assert!(
        residency_stats.evictions > 0,
        "a two-expert device cache served ten experts without evicting: {residency_stats:?}"
    );
    assert!(
        device_stats.backpressure_drains > 0,
        "a two-expert cache never refused a four-deep queue: {device_stats:?}"
    );

    // Counts, not a performance claim. There is no baseline on this machine to
    // compare either figure against, both are **debug builds**, the host figure
    // includes reading 118 MB from disk, and neither was repeated. They are
    // printed because document 03 requires reads, evictions and demand-eviction
    // counts to be recorded -- not because either number means anything about
    // throughput.
    println!(
        "real layer 0 experts: {} distinct expert(s), {} B demanded, {} eviction(s), \
         {} backpressure drain(s); {} BF16 slot components equal between the two candidates",
        device_stats.device_groups,
        10 * EXPERT_TOTAL,
        residency_stats.evictions,
        device_stats.backpressure_drains,
        device_slots.len() / 2
    );
    println!(
        "wall clock, debug builds, one run each, NOT a benchmark: device {device_elapsed:.2?}, \
         host {host_elapsed:.2?}"
    );
    println!(
        "NOT model support: the activations are synthetic and the route is written by this test. \
         Output quality is O2."
    );
}
