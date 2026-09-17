//! Real-device proof for task 0021's grouped expert path.
//!
//! The declared gate is **bitwise** equality with task 0019's oracle over the
//! qualification fixtures, and the count of components compared is printed
//! rather than described. The one operation that cannot be bitwise by
//! construction is the gate transform's transcendental -- CUDA's device
//! `tanh`/`exp` in FP64 may differ from the host libm's by under an ulp -- and
//! the contract's honest statement is that no fixture here constructs a case
//! where such a difference straddles a BF16 tie point, so that case is
//! **untested rather than shown absent**.
//!
//! No checkpoint, no model. Weight-shaped bytes this test invents, run through
//! the same residency authority and the same plan a real layer would use.
#![cfg(feature = "driver")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use moxie_cuda::RankContext;
use moxie_executor::grouped::{ExpertRoles, GroupedRun};
use moxie_executor::residency::{DeviceResidency, ShardSource};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
use moxie_memory::{
    ArtifactId, CapacitySnapshot, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_oracles::route;
use moxie_plan::expert::{
    Candidate, ExpertBudget, ExpertKernels, ExpertPolicy, ResidentChunks, compile_experts,
};
use moxie_storage::Shard;
use moxie_types::{RankId, Scope, StrategyControl};

const HIDDEN: u64 = 128;
const INTERMEDIATE: u64 = 32;
const EXPERTS: u64 = 5;
const TOP_K: u64 = 2;
const ROWS: u64 = 16;
/// `3 * intermediate * hidden * 2`.
const CHUNK: u64 = 3 * INTERMEDIATE * HIDDEN * 2;
const MIB: u64 = 1024 * 1024;

/// Sixteen rows over five experts, with reuse counts 15, 14, 1, 1, 1.
///
/// The spread is deliberate: at a threshold between `CHUNK / 2` and `CHUNK`,
/// the two reused experts amortise their transfer and the three singletons do
/// not, which is the mixed plan the interface exists for.
fn route() -> Vec<u32> {
    let mut out = Vec::with_capacity((ROWS * TOP_K) as usize);
    for row in 0..ROWS {
        let pair = match row {
            13 => (0, 2),
            14 => (1, 3),
            15 => (0, 4),
            _ => (0, 1),
        };
        out.push(pair.0);
        out.push(pair.1);
    }
    out
}

/// A `RankContext` is exclusive per device, and `cargo test` runs a binary's
/// tests in parallel threads. Serialising them here is not a workaround for a
/// defect: the exclusivity is the property task 0007 established deliberately,
/// and two tests racing for one card would be testing the harness.
static DEVICE: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

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

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
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
    ArtifactId::new("grouped-device-fixture-v1").unwrap()
}

fn roles() -> ExpertRoles {
    ExpertRoles {
        per_expert: None,
        artifact: artifact(),
        gate_up_role: "experts_gate_up".into(),
        down_role: "experts_down".into(),
        format_version: 1,
    }
}

/// The same source, opened again: a run consumes its reader, and this test
/// opens one per device.
fn source_at(path: &Path) -> ShardSource {
    source(path)
}

fn source(path: &Path) -> ShardSource {
    ShardSource::new(artifact(), vec![Shard::open(path).unwrap()])
        .role("experts_gate_up", 0, "experts.gate_up_proj")
        .unwrap()
        .role("experts_down", 0, "experts.down_proj")
        .unwrap()
}

fn mlp(activation: ExpertActivation) -> OpParams {
    OpParams::ExpertMlp {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        experts: EXPERTS,
        top_k: TOP_K,
        activation,
    }
}

fn combine() -> OpParams {
    OpParams::Combine {
        hidden: HIDDEN,
        top_k: TOP_K,
        order: CombineOrder::AscendingExpertId,
        output_scale: 1.0,
    }
}

/// What every slot must be, bit for bit.
fn oracle_slots(w: &Weights, activation: ExpertActivation) -> Vec<u16> {
    let spec = route::ExpertSpec {
        experts: EXPERTS as usize,
        hidden: HIDDEN as usize,
        intermediate: INTERMEDIATE as usize,
        activation,
    };
    let mut out = Vec::new();
    let route = route();
    for (index, expert) in route.iter().enumerate() {
        let row = index / TOP_K as usize;
        let y = route::expert_row(
            &w.x[row * HIDDEN as usize..(row + 1) * HIDDEN as usize],
            &w.gate_up,
            &w.down,
            *expert,
            spec,
        )
        .unwrap();
        out.extend(y.iter().map(|v| to_bf16_bits(*v)));
    }
    out
}

fn oracle_rows(w: &Weights, activation: ExpertActivation) -> Vec<u16> {
    let slots = oracle_slots(w, activation);
    let mut out = Vec::new();
    for r in 0..ROWS as usize {
        let route = route();
        let experts = &route[r * TOP_K as usize..(r + 1) * TOP_K as usize];
        let values: Vec<f32> = slots
            [r * TOP_K as usize * HIDDEN as usize..(r + 1) * TOP_K as usize * HIDDEN as usize]
            .iter()
            .map(|bits| f32::from_bits((*bits as u32) << 16))
            .collect();
        let row = route::combine_row(
            experts,
            &w.coefficients[r * TOP_K as usize..(r + 1) * TOP_K as usize],
            &values,
            HIDDEN as usize,
            CombineOrder::AscendingExpertId,
            1.0,
        )
        .unwrap();
        out.extend(row.iter().map(|v| to_bf16_bits(*v)));
    }
    out
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "moxie-grouped-device-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every expert on the device, on every installed card, for both gate
/// transforms.
#[test]
fn every_device_reproduces_the_oracle_bit_for_bit_for_both_gate_transforms() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let dir = scratch("oracle");
    let mut compared = 0usize;

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let catalogue = moxie_kernels::expert_mlp_catalogue();

        for activation in [ExpertActivation::GeGlu, ExpertActivation::SwiGlu] {
            let w = weights(0x0021_0019 + ordinal as u64);
            let path = write_shard(&dir, &w);
            let mut src = source(&path);
            let mut ledger = Ledger::new([
                CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
                CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
            ])
            .unwrap();
            let mut authority = ResidencyAuthority::open(
                &mut ledger,
                &ResidencyRequest::new("device experts", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
            )
            .unwrap();
            let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

            let budget = ExpertBudget {
                device: ctx.uuid(),
                device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
                device_cache_cap_bytes: 8 * CHUNK,
                device_cache_leased_bytes: 0,
                device_cache_largest_free_bytes: u64::MAX,
                device_arena_free_bytes: 16 * MIB,
                host_workspace_bytes: MIB,
                host_buffer_bytes: MIB,
                host_cache_cap_bytes: 1 << 30,
                host_cache_leased_bytes: 0,
                host_cache_largest_free_bytes: u64::MAX,
                cache_alignment_bytes: 256,
                chunks_per_expert: 2,
                resident: ResidentChunks::none(),
            };
            let policy = ExpertPolicy {
                device: StrategyControl::Required,
                host: StrategyControl::Auto,
                host_placement: StrategyControl::Off,
                ..ExpertPolicy::default()
            };
            let plan = compile_experts(
                &mlp(activation),
                &combine(),
                &route(),
                &budget,
                &policy,
                None,
                Some(ExpertKernels {
                    capability: ctx.capability(),
                    catalogue: &catalogue,
                }),
            )
            .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));
            assert!(
                plan.groups()
                    .iter()
                    .all(|g| g.placement().candidate() == Candidate::Device),
                "`required` must put every group on the device"
            );
            assert!(plan.kernel().is_some());

            let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None)
                .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));
            run.attach_device(&mut ledger, &ctx, &mut residency)
                .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));

            let x = to_bytes(&w.x);
            run.load_activations(&x).unwrap();
            run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
                .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));

            // Slots first: a wrong reduction over right slots and a right
            // reduction over wrong slots are different defects.
            let want_slots = oracle_slots(&w, activation);
            assert_eq!(
                as_u16(run.buffers().slots()),
                want_slots,
                "device {ordinal} {activation:?}: slots"
            );
            let got = as_u16(run.reduce(&w.coefficients).unwrap());
            assert_eq!(
                got,
                oracle_rows(&w, activation),
                "device {ordinal} {activation:?}: rows"
            );
            compared += want_slots.len() + got.len();
            assert_eq!(run.stats().device_groups, 5);
            assert_eq!(run.stats().host_groups, 0);

            run.close(&mut ledger).unwrap();
            // Every lease is back, but the chunks are still resident. Retiring
            // them is what frees the ranges that live in the one real device
            // allocation, and the residency refuses to close until they are.
            authority.end_turn(TurnId::new(1));
            authority.retire_all(scope);
            residency.close(&mut authority).unwrap();
            authority.close(&mut ledger).unwrap();
            assert!(ledger.outstanding().is_empty());
        }
    }
    println!(
        "grouped expert device gate: {compared} BF16 components bit-identical to the oracle \
         across {count} device(s) and both gate transforms"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// One plan, both candidates. The slot buffer is written by a CPU kernel and by
/// a GPU kernel, and the reduction is one declared order over both.
#[test]
fn a_mixed_plan_reduces_cpu_and_gpu_slots_in_one_declared_order() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ordinal = 0;
    let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();
    let dir = scratch("mixed");
    let w = weights(0x5ec0_11de);
    let path = write_shard(&dir, &w);
    let mut src = source(&path);

    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("mixed experts", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

    let budget = ExpertBudget {
        device: ctx.uuid(),
        device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
        device_cache_cap_bytes: 8 * CHUNK,
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: u64::MAX,
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        host_cache_cap_bytes: 1 << 30,
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: u64::MAX,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident: ResidentChunks::none(),
    };
    // Between `CHUNK / 2` and `CHUNK`: experts reused by two or three rows
    // amortise their transfer, the singletons do not. That is the crossover the
    // interface exists for, expressed with the declared parameter.
    let policy = ExpertPolicy {
        device: StrategyControl::Auto,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        max_transfer_bytes_per_row: CHUNK / 2 + 1,
        ..ExpertPolicy::default()
    };
    let plan = compile_experts(
        &mlp(ExpertActivation::GeGlu),
        &combine(),
        &route(),
        &budget,
        &policy,
        None,
        Some(ExpertKernels {
            capability: ctx.capability(),
            catalogue: &catalogue,
        }),
    )
    .unwrap();
    assert!(
        plan.uses(Candidate::Device) && plan.uses(Candidate::Host),
        "not a mixed plan"
    );
    let on_device: Vec<u32> = plan
        .groups_on(Candidate::Device)
        .map(|g| g.expert())
        .collect();
    let on_host: Vec<u32> = plan
        .groups_on(Candidate::Host)
        .map(|g| g.expert())
        .collect();
    assert_eq!(on_device, vec![0, 1], "reuse 15 and 14 amortise");
    assert_eq!(on_host, vec![2, 3, 4], "reuse 1 does not");

    let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
    run.attach_device(&mut ledger, &ctx, &mut residency)
        .unwrap();

    let x = to_bytes(&w.x);
    run.load_activations(&x).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(2), 0, u64::MAX)
        .unwrap();
    assert_eq!(run.stats().device_groups, 2);
    assert_eq!(run.stats().host_groups, 3);
    assert_eq!(run.stats().slots_written, ROWS * TOP_K);

    assert_eq!(
        as_u16(run.buffers().slots()),
        oracle_slots(&w, ExpertActivation::GeGlu),
        "a slot written by one candidate differs from the other's"
    );
    assert_eq!(
        as_u16(run.reduce(&w.coefficients).unwrap()),
        oracle_rows(&w, ExpertActivation::GeGlu)
    );

    run.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(2));
    authority.retire_all(scope);
    residency.close(&mut authority).unwrap();
    authority.close(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A device cache that holds one expert against a five-expert layer, with a
/// queue four deep: the authority refuses, the run drains, and the answer is
/// still the oracle's.
#[test]
fn a_device_cache_smaller_than_the_working_set_executes_under_backpressure() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ordinal = 0;
    let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();
    let dir = scratch("restricted");
    let w = weights(0x0bad_cafe);
    let path = write_shard(&dir, &w);
    let mut src = source(&path);

    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    // One expert's two chunks, and no more.
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("restricted device", 64 * CHUNK).device(ctx.uuid(), CHUNK),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

    let budget = ExpertBudget {
        device: ctx.uuid(),
        device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
        device_cache_cap_bytes: CHUNK,
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: u64::MAX,
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        host_cache_cap_bytes: 1 << 30,
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: u64::MAX,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident: ResidentChunks::none(),
    };
    let policy = ExpertPolicy {
        device: StrategyControl::Required,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        max_inflight_orders: 4,
        ..ExpertPolicy::default()
    };
    let plan = compile_experts(
        &mlp(ExpertActivation::GeGlu),
        &combine(),
        &route(),
        &budget,
        &policy,
        None,
        Some(ExpertKernels {
            capability: ctx.capability(),
            catalogue: &catalogue,
        }),
    )
    .unwrap();
    assert_eq!(plan.queue_capacity(), 4);

    let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
    run.attach_device(&mut ledger, &ctx, &mut residency)
        .unwrap();
    let x = to_bytes(&w.x);
    run.load_activations(&x).unwrap();
    run.run_to_completion(&mut authority, &mut src, TurnId::new(3), 0, u64::MAX)
        .unwrap();
    assert_eq!(
        as_u16(run.reduce(&w.coefficients).unwrap()),
        oracle_rows(&w, ExpertActivation::GeGlu)
    );
    assert!(
        run.stats().backpressure_drains > 0,
        "a one-expert device cache never refused a four-deep queue: {:?}",
        run.stats()
    );
    assert!(authority.stats().evictions > 0);
    println!(
        "device budget {CHUNK} B against {} experts: {} drain(s), {} eviction(s)",
        EXPERTS,
        run.stats().backpressure_drains,
        authority.stats().evictions
    );

    run.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(3));
    authority.retire_all(scope);
    residency.close(&mut authority).unwrap();
    authority.close(&mut ledger).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The attachment's failure paths give the envelope back.
///
/// Two of them, because they are different: a plan with no selected kernel
/// fails **before** an arena takes the reservation and hands it straight back,
/// while a bad image fails **after**, with ranges already allocated — the arena
/// must then release every one of them and close, or the charge and the device
/// memory are both stranded behind a tidy-looking error return.
#[test]
fn a_refused_attachment_leaves_nothing_charged() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ordinal = 0;
    let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();

    let budget = ExpertBudget {
        device: ctx.uuid(),
        device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
        device_cache_cap_bytes: 8 * CHUNK,
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: u64::MAX,
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        host_cache_cap_bytes: 1 << 30,
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: u64::MAX,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident: ResidentChunks::none(),
    };
    let device_policy = ExpertPolicy {
        device: StrategyControl::Required,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        ..ExpertPolicy::default()
    };

    // After the arena: the ranges are allocated and the image is refused.
    {
        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        // A catalogue whose descriptor names a package this build does not
        // have. It is the review's own counterexample: an all-zero image hash
        // used to load the real fatbin anyway.
        let mut stranger = catalogue.descriptors()[0].clone();
        stranger.image_sha256 = [0; 32];
        let stranger = moxie_types::KernelCatalogue::new(vec![stranger]).unwrap();
        let plan = compile_experts(
            &mlp(ExpertActivation::GeGlu),
            &combine(),
            &route(),
            &budget,
            &device_policy,
            None,
            Some(ExpertKernels {
                capability: ctx.capability(),
                catalogue: &stranger,
            }),
        )
        .unwrap();
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("refused", 8 * CHUNK).device(ctx.uuid(), CHUNK),
        )
        .unwrap();
        let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();
        let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
        assert!(ledger.scope_committed(scope) > 0);
        let error = run
            .attach_device(&mut ledger, &ctx, &mut residency)
            .expect_err("the plan's descriptor does not name this build's package");
        assert!(
            format!("{error}").contains("does not identify this build's"),
            "{error}"
        );
        // The arena took the reservation and released it on the way out, so the
        // run gave its host buffers back at the same moment. Nothing is charged
        // and nothing is allocated against a charge of zero.
        assert!(run.buffers().slots().is_empty());
        assert!(run.failure().is_some());
        run.close(&mut ledger).unwrap();
        residency.close(&mut authority).unwrap();
        authority.close(&mut ledger).unwrap();
        assert!(
            ledger.outstanding().is_empty(),
            "a refused attachment left {:?} charged",
            ledger.outstanding()
        );
    }

    // Before the arena: a host-only plan selects no device kernel.
    {
        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let plan = compile_experts(
            &mlp(ExpertActivation::GeGlu),
            &combine(),
            &route(),
            &budget,
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
        assert!(plan.kernel().is_none());
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("refused", 8 * CHUNK).device(ctx.uuid(), CHUNK),
        )
        .unwrap();
        let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();
        let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
        run.attach_device(&mut ledger, &ctx, &mut residency)
            .expect_err("a host-only plan has no device kernel");
        // No arena took the reservation, so it went back into the run: the host
        // buffers and their charge stay together and one `close` gives both
        // back.
        assert!(!run.buffers().slots().is_empty());
        run.close(&mut ledger).unwrap();
        residency.close(&mut authority).unwrap();
        authority.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }
}

/// A close against the wrong ledger changes nothing and can be retried.
///
/// A second review closed against another ledger, lost the attachment to the
/// refusal, and could not then close against the right one: the retry reported
/// "already been closed" while the original reservation stayed outstanding.
#[test]
fn a_close_against_the_wrong_ledger_is_recoverable() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ctx = RankContext::acquire(RankId(0), 0).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();

    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut other = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("retry", 8 * CHUNK).device(ctx.uuid(), CHUNK),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

    let budget = ExpertBudget {
        device: ctx.uuid(),
        device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
        device_cache_cap_bytes: 8 * CHUNK,
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: u64::MAX,
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        host_cache_cap_bytes: 1 << 30,
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: u64::MAX,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident: ResidentChunks::none(),
    };
    let plan = compile_experts(
        &mlp(ExpertActivation::GeGlu),
        &combine(),
        &route(),
        &budget,
        &ExpertPolicy {
            device: StrategyControl::Required,
            host: StrategyControl::Auto,
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
    let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
    run.attach_device(&mut ledger, &ctx, &mut residency)
        .unwrap();
    let charged = ledger.scope_committed(scope);
    assert!(charged > 0);

    let run = {
        let refused = run
            .close(&mut other)
            .expect_err("this run belongs to another ledger");
        // The run-level check, by its own words: nothing was touched. Letting the
        // arena refuse instead would mean the ranges had already been released
        // into it before the ledger was even looked at.
        assert!(
            format!("{refused}").contains("nothing was released"),
            "{refused}"
        );
        // Nothing moved: the charge is where it was, and the attachment
        // survives the refusal intact.
        assert_eq!(ledger.scope_committed(scope), charged);
        assert_eq!(other.scope_committed(scope), 0);
        let run = refused.run;
        assert!(run.has_device());
        run
    };

    // And the retry against the right ledger works. What is left charged
    // afterwards is the residency cache's own capacity, not this run's.
    run.close(&mut ledger).unwrap();
    assert_eq!(ledger.scope_committed(scope), CHUNK);
    residency.close(&mut authority).unwrap();
    authority.close(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

// ---------------------------------------------------------------------------
// Task 0022: the second consumer's shape, against a budget stated as a ratio
// ---------------------------------------------------------------------------

/// The second consumer's shape, taken from the model definition rather than
/// repeated here.
///
/// `moxie_models::laguna::ARTIFACT` is what
/// `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`'s own `config.json` declares,
/// checked against it by `the_declared_geometry_matches_the_artifact_config`.
/// Writing 3,072 and 1,024 again here would be a second answer to one
/// question, which is how the two come to disagree.
///
/// The artifact stores its experts as asymmetric INT4 at group 32, which costs
/// **5,455,920 B** and which no importer accepts; `docs/models/laguna.md`
/// records both figures. These are weight-shaped bytes this test invents at the
/// artifact's declared *shape*, not its weights, and no quality claim follows.
const LAGUNA: moxie_models::laguna::ArtifactGeometry = moxie_models::laguna::ARTIFACT;
const LAGUNA_HIDDEN: u64 = LAGUNA.hidden;
const LAGUNA_INTERMEDIATE: u64 = LAGUNA.moe.moe_intermediate;
const LAGUNA_CHUNK: u64 = LAGUNA.expert_bf16_bytes();
/// `num_experts_per_tok`.
const LAGUNA_TOP_K: u64 = LAGUNA.moe.top_k;
/// Twelve of the artifact's 256, which is what fits a test's disk and time
/// while still leaving a route that does not select everything.
const LAGUNA_EXPERTS: u64 = 12;
const LAGUNA_ROWS: u64 = 2;

/// The declared restriction: the device may hold **one eighth** of what the
/// route demands.
///
/// A ratio rather than a constant, because a constant stops meaning
/// "restricted" the moment a shape changes, and `the_restricted_budget_is_a_ratio`
/// asserts the relation rather than the number.
const RESTRICTION: u64 = 8;

fn laguna_route() -> Vec<u32> {
    // Row 0 takes experts 0..10, row 1 takes 2..12: every expert is demanded,
    // eight of them by both rows and four by one. Neither row is in ascending
    // order, so the two `CombineOrder` values do not agree.
    let mut route: Vec<u32> = (0..LAGUNA_TOP_K as u32).rev().collect();
    route.extend((2..2 + LAGUNA_TOP_K as u32).rev());
    route
}

fn laguna_working_set_bytes(route: &[u32]) -> u64 {
    let mut distinct: Vec<u32> = route.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    distinct.len() as u64 * LAGUNA_CHUNK
}

fn laguna_weights(seed: u64) -> Weights {
    let mut v = Values::new(seed);
    Weights {
        gate_up: v.block((LAGUNA_EXPERTS * 2 * LAGUNA_INTERMEDIATE * LAGUNA_HIDDEN) as usize),
        down: v.block((LAGUNA_EXPERTS * LAGUNA_HIDDEN * LAGUNA_INTERMEDIATE) as usize),
        x: v.block((LAGUNA_ROWS * LAGUNA_HIDDEN) as usize),
        coefficients: (0..(LAGUNA_ROWS * LAGUNA_TOP_K) as usize)
            .map(|_| v.next().abs())
            .collect(),
    }
}

fn laguna_write_shard(dir: &Path, w: &Weights) -> PathBuf {
    let gate_up = to_bytes(&w.gate_up);
    let down = to_bytes(&w.down);
    let split = gate_up.len();
    let end = split + down.len();
    let header = format!(
        "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{LAGUNA_EXPERTS},{},\
         {LAGUNA_HIDDEN}],\"data_offsets\":[0,{split}]}},\
         \"experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{LAGUNA_EXPERTS},{LAGUNA_HIDDEN},\
         {LAGUNA_INTERMEDIATE}],\"data_offsets\":[{split},{end}]}}}}",
        2 * LAGUNA_INTERMEDIATE
    );
    let path = dir.join("laguna-shaped.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(&gate_up).unwrap();
    f.write_all(&down).unwrap();
    f.flush().unwrap();
    path
}

fn laguna_mlp() -> OpParams {
    OpParams::ExpertMlp {
        hidden: LAGUNA_HIDDEN,
        intermediate: LAGUNA_INTERMEDIATE,
        experts: LAGUNA_EXPERTS,
        top_k: LAGUNA_TOP_K,
        // `hidden_act` is `silu`.
        activation: ExpertActivation::SwiGlu,
    }
}

fn laguna_combine() -> OpParams {
    OpParams::Combine {
        hidden: LAGUNA_HIDDEN,
        top_k: LAGUNA_TOP_K,
        order: CombineOrder::AscendingExpertId,
        // `moe_routed_scaling_factor`.
        output_scale: 2.5,
    }
}

/// The restriction is a relation, not a number.
///
/// Checked on its own so that a shape change which quietly made the budget
/// larger than the working set fails here, naming the reason, rather than
/// turning a "restricted" case into an unrestricted one that still passes.
#[test]
fn the_restricted_budget_is_a_ratio_of_the_working_set() {
    let route = laguna_route();
    let working = laguna_working_set_bytes(&route);
    let budget = working.div_ceil(RESTRICTION);
    assert_eq!(working, 12 * LAGUNA_CHUNK);
    assert!(budget < working, "the budget is not a restriction");
    assert!(
        budget >= LAGUNA_CHUNK,
        "a budget below one expert refuses rather than evicts, which is a different case"
    );
    assert!(
        budget < 2 * LAGUNA_CHUNK,
        "at 1/{RESTRICTION} of a {}-expert working set the cache must hold exactly one expert",
        working / LAGUNA_CHUNK
    );
    println!(
        "laguna-shaped working set {working} B over {} expert(s); budget {budget} B at 1/{RESTRICTION}",
        working / LAGUNA_CHUNK
    );
}

/// The second consumer's shape, on **every** installed device, against a budget
/// that is one eighth of what its route demands.
///
/// The check is that the two candidates agree: the same plan shape run all on
/// the device and all on the host produces the same bytes. That is the gate
/// task 0021's real-artifact case used, and it is the one that survives a
/// shape this large -- an FP64 oracle over 61,440 components at this width is
/// already covered bit for bit by
/// `every_device_reproduces_the_oracle_bit_for_bit_for_both_gate_transforms`
/// at a size a test can afford to run for both gate transforms on three cards.
#[test]
fn the_second_consumers_shape_executes_on_every_device_under_a_restricted_budget() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");

    let dir = scratch("laguna-restricted");
    let w = laguna_weights(0x1a90_0220);
    let path = laguna_write_shard(&dir, &w);
    let route = laguna_route();
    let working = laguna_working_set_bytes(&route);
    let budget_bytes = working.div_ceil(RESTRICTION);
    let x = to_bytes(&w.x);

    // The host candidate's answer, computed once: this is what every device
    // must agree with.
    let host = {
        let mut ledger =
            Ledger::new([CapacitySnapshot::new(Scope::Host, 4 << 30, 64 * MIB).unwrap()]).unwrap();
        let mut authority =
            ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new("laguna host", working))
                .unwrap();
        let mut src = source_at(&path);
        let budget = ExpertBudget {
            device: moxie_types::DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9")
                .unwrap(),
            device_pci_bus_id: "0000:82:00.0".into(),
            device_cache_cap_bytes: 0,
            device_cache_leased_bytes: 0,
            device_cache_largest_free_bytes: u64::MAX,
            device_arena_free_bytes: 0,
            host_workspace_bytes: 64 * MIB,
            host_buffer_bytes: 64 * MIB,
            host_cache_cap_bytes: 1 << 30,
            host_cache_leased_bytes: 0,
            host_cache_largest_free_bytes: u64::MAX,
            cache_alignment_bytes: 256,
            chunks_per_expert: 2,
            resident: ResidentChunks::none(),
        };
        let policy = ExpertPolicy {
            device: StrategyControl::Off,
            host: StrategyControl::Required,
            host_placement: StrategyControl::Off,
            max_inflight_orders: 4,
            ..ExpertPolicy::default()
        };
        let plan = compile_experts(
            &laguna_mlp(),
            &laguna_combine(),
            &route,
            &budget,
            &policy,
            None,
            None,
        )
        .unwrap();
        assert!(plan.uses(Candidate::Host) && !plan.uses(Candidate::Device));
        let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
        run.load_activations(&x).unwrap();
        run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
            .unwrap();
        let out = as_u16(run.reduce(&w.coefficients).unwrap());
        run.close(&mut ledger).unwrap();
        authority.end_turn(TurnId::new(1));
        authority.retire_all(Scope::Host);
        authority.close(&mut ledger).unwrap();
        out
    };
    assert_eq!(host.len(), (LAGUNA_ROWS * LAGUNA_HIDDEN) as usize);

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let catalogue = moxie_kernels::expert_mlp_catalogue();
        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 4 << 30, 64 * MIB).unwrap(),
            CapacitySnapshot::new(scope, 2 << 30, 64 * MIB).unwrap(),
        ])
        .unwrap();
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("laguna restricted", working).device(ctx.uuid(), budget_bytes),
        )
        .unwrap();
        let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();
        let mut src = source_at(&path);

        let budget = ExpertBudget {
            device: ctx.uuid(),
            device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
            device_cache_cap_bytes: budget_bytes,
            device_cache_leased_bytes: 0,
            device_cache_largest_free_bytes: u64::MAX,
            device_arena_free_bytes: 256 * MIB,
            host_workspace_bytes: 64 * MIB,
            host_buffer_bytes: 64 * MIB,
            host_cache_cap_bytes: 1 << 30,
            host_cache_leased_bytes: 0,
            host_cache_largest_free_bytes: u64::MAX,
            cache_alignment_bytes: 256,
            chunks_per_expert: 2,
            resident: ResidentChunks::none(),
        };
        let policy = ExpertPolicy {
            device: StrategyControl::Required,
            host: StrategyControl::Auto,
            host_placement: StrategyControl::Off,
            max_inflight_orders: 4,
            // Declared, because the default would refuse every group here --
            // see `the_default_amortisation_threshold_sends_every_laguna_expert_to_the_cpu`.
            // This case is about what residency does under a restricted budget,
            // and the amortisation crossover is M6's to measure.
            max_transfer_bytes_per_row: LAGUNA_CHUNK,
            ..ExpertPolicy::default()
        };
        let plan = compile_experts(
            &laguna_mlp(),
            &laguna_combine(),
            &route,
            &budget,
            &policy,
            None,
            Some(ExpertKernels {
                capability: ctx.capability(),
                catalogue: &catalogue,
            }),
        )
        .unwrap();
        assert!(plan.uses(Candidate::Device));

        let mut run = GroupedRun::admit(&mut ledger, plan, roles(), None).unwrap();
        run.attach_device(&mut ledger, &ctx, &mut residency)
            .unwrap();
        run.load_activations(&x).unwrap();
        run.run_to_completion(&mut authority, &mut src, TurnId::new(2), 0, u64::MAX)
            .unwrap();
        let device = as_u16(run.reduce(&w.coefficients).unwrap());
        assert_eq!(
            device,
            host,
            "device {ordinal} disagreed with the host candidate on {} component(s)",
            device.iter().zip(&host).filter(|(a, b)| a != b).count()
        );
        assert!(
            run.stats().backpressure_drains > 0,
            "a one-expert cache never refused a four-deep queue on device {ordinal}: {:?}",
            run.stats()
        );
        assert!(authority.stats().evictions > 0);
        println!(
            "device {ordinal} ({}): laguna-shaped top-{LAGUNA_TOP_K} over {} expert(s), \
             working set {working} B, budget {budget_bytes} B (1/{RESTRICTION}); {} drain(s), \
             {} eviction(s); {} BF16 component(s) agree with the CPU candidate",
            ctx.uuid(),
            working / LAGUNA_CHUNK,
            run.stats().backpressure_drains,
            authority.stats().evictions,
            host.len()
        );

        run.close(&mut ledger).unwrap();
        authority.end_turn(TurnId::new(2));
        authority.retire_all(scope);
        residency.close(&mut authority).unwrap();
        authority.close(&mut ledger).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same arithmetic AGENTS.md records for the Gemma artifact, at Laguna's
/// numbers: **the declared default sends every expert to the CPU.**
///
/// One expert is 18,874,368 B in BF16, and the best reuse a two-row batch can
/// offer is both rows, so 9,437,184 B per row against a declared default of
/// 1 MiB. Asserted here rather than left to be rediscovered. Whether it is the
/// *right* decision is a measurement, and the measurement is M6's.
#[test]
fn the_default_amortisation_threshold_sends_every_laguna_expert_to_the_cpu() {
    let route = laguna_route();
    let budget = ExpertBudget {
        device: moxie_types::DeviceUuid::parse("GPU-3032cfa3-19df-028f-5ebd-43314911e0b9").unwrap(),
        device_pci_bus_id: "0000:82:00.0".into(),
        device_cache_cap_bytes: 64 * LAGUNA_CHUNK,
        device_cache_leased_bytes: 0,
        device_cache_largest_free_bytes: u64::MAX,
        device_arena_free_bytes: 1 << 30,
        host_workspace_bytes: 64 * MIB,
        host_buffer_bytes: 64 * MIB,
        host_cache_cap_bytes: 1 << 30,
        host_cache_leased_bytes: 0,
        host_cache_largest_free_bytes: u64::MAX,
        cache_alignment_bytes: 256,
        chunks_per_expert: 2,
        resident: ResidentChunks::none(),
    };
    let policy = ExpertPolicy {
        device: StrategyControl::Auto,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        max_inflight_orders: 4,
        ..ExpertPolicy::default()
    };
    assert_eq!(
        policy.max_transfer_bytes_per_row,
        1024 * 1024,
        "the declared default moved; this test's arithmetic is about that number"
    );
    let plan = compile_experts(
        &laguna_mlp(),
        &laguna_combine(),
        &route,
        &budget,
        &policy,
        None,
        None,
    )
    .unwrap();
    assert!(
        !plan.uses(Candidate::Device) && plan.uses(Candidate::Host),
        "a group reached the device at 1 MiB/row"
    );
    for group in plan.groups() {
        let decision = group.decision();
        assert!(
            decision.bytes_per_row >= LAGUNA_CHUNK / 2,
            "expert {} amortised to {} B/row, which this fixture did not intend",
            group.expert(),
            decision.bytes_per_row
        );
    }
    println!(
        "laguna-shaped: {} expert(s), best reuse {} row(s), {} B/row against a 1 MiB default",
        plan.groups().len(),
        plan.groups()
            .iter()
            .map(|g| g.decision().reuse_rows)
            .max()
            .unwrap(),
        plan.groups()
            .iter()
            .map(|g| g.decision().bytes_per_row)
            .min()
            .unwrap()
    );
}

struct IntegerSource(std::collections::BTreeMap<String, Vec<u8>>);
impl moxie_executor::residency::ChunkSource for IntegerSource {
    fn read_chunk(
        &mut self,
        chunk: &moxie_memory::ChunkId,
        into: &mut [u8],
    ) -> moxie_types::Result<()> {
        assert_eq!(chunk.artifact(), &artifact());
        assert_eq!(chunk.range().offset_bytes(), 0);
        let bytes = &self.0[chunk.slot().role()];
        assert_eq!(bytes.len(), into.len());
        into.copy_from_slice(bytes);
        Ok(())
    }
}

fn integer_cases() -> Vec<([moxie_plan::expert::ExpertWeightFormat; 2], bool)> {
    use moxie_plan::expert::ExpertWeightFormat::{Affine, Bf16};
    use moxie_types::Precision;
    let a = Affine {
        width: Precision::Int4,
        group: 32,
        scale: Precision::F16,
        zeros: true,
    };
    let b = Affine {
        width: Precision::Int8,
        group: 128,
        scale: Precision::F32,
        zeros: true,
    };
    let c = Affine {
        width: Precision::Int8,
        group: 32,
        scale: Precision::Bf16,
        zeros: false,
    };
    [[a, b], [b, a], [Bf16, c], [c, Bf16]]
        .into_iter()
        .flat_map(|formats| [false, true].map(|device| (formats, device)))
        .collect()
}

fn integer_weights(
    formats: [moxie_plan::expert::ExpertWeightFormat; 2],
) -> (Weights, IntegerSource, Vec<(String, String)>) {
    use moxie_plan::expert::ExpertWeightFormat;
    use moxie_types::Precision;
    let mut w = weights(0x0035);
    w.gate_up.clear();
    w.down.clear();
    let mut sources = std::collections::BTreeMap::new();
    let mut pairs = Vec::new();
    for e in 0..EXPERTS {
        let names = (format!("expert.{e}.gate_up"), format!("expert.{e}.down"));
        for (projection, (outputs, inputs, name)) in [
            (2 * INTERMEDIATE, HIDDEN, &names.0),
            (HIDDEN, INTERMEDIATE, &names.1),
        ]
        .into_iter()
        .enumerate()
        {
            let (outputs, inputs) = (outputs as usize, inputs as usize);
            let (bits, group, scale, zeros) = match formats[projection] {
                ExpertWeightFormat::Bf16 => (16, 32, Precision::Bf16, false),
                ExpertWeightFormat::Affine {
                    width,
                    group,
                    scale,
                    zeros,
                } => (width.bits() as usize, group as usize, scale, zeros),
            };
            let groups = inputs.div_ceil(group);
            let stride = if bits == 16 {
                inputs * 2
            } else {
                inputs.div_ceil(8 / bits)
            };
            let mut bytes = vec![0; outputs * stride];
            let values = if projection == 0 {
                &mut w.gate_up
            } else {
                &mut w.down
            };
            for o in 0..outputs {
                for k in 0..inputs {
                    let g = k / group;
                    let zero = if zeros { ((o + g) % 3) as i32 - 1 } else { 0 };
                    let sign = if (o + g).is_multiple_of(2) { 1.0 } else { -1.0 };
                    let width = bits.min(8);
                    let q =
                        ((o * 11 + k * 7 + e as usize) % (1 << width)) as i32 - (1 << (width - 1));
                    let value = (q - zero) as f32 / 64.0 * sign;
                    values.push(value);
                    match bits {
                        4 => bytes[o * stride + k / 2] |= (q as u8 & 15) << ((k % 2) * 4),
                        8 => bytes[o * stride + k] = q as u8,
                        _ => bytes[o * stride + k * 2..o * stride + k * 2 + 2]
                            .copy_from_slice(&to_bf16_bits(value).to_le_bytes()),
                    }
                }
            }
            if bits != 16 {
                for o in 0..outputs {
                    for g in 0..groups {
                        let negative = !(o + g).is_multiple_of(2);
                        match scale {
                            Precision::F16 => bytes.extend_from_slice(
                                &(if negative { 0xa400u16 } else { 0x2400 }).to_le_bytes(),
                            ),
                            Precision::Bf16 => bytes.extend_from_slice(
                                &to_bf16_bits(if negative { -1.0 / 64.0 } else { 1.0 / 64.0 })
                                    .to_le_bytes(),
                            ),
                            _ => bytes.extend_from_slice(
                                &(if negative { -1.0f32 / 64.0 } else { 1.0 / 64.0 }).to_le_bytes(),
                            ),
                        }
                    }
                }
                if zeros {
                    for o in 0..outputs {
                        for g in 0..groups {
                            bytes.extend_from_slice(&(((o + g) % 3) as i16 - 1).to_le_bytes());
                        }
                    }
                }
            }
            sources.insert(name.clone(), bytes);
        }
        pairs.push(names);
    }
    (w, IntegerSource(sources), pairs)
}

#[test]
fn canonical_integer_experts_use_the_shared_residency_and_reduction_on_every_device() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let mut compared = 0usize;

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let catalogue = moxie_kernels::expert_mlp_catalogue();

        for activation in [ExpertActivation::GeGlu, ExpertActivation::SwiGlu] {
            for (formats, on_device) in integer_cases() {
                let (w, mut src, role_pairs) = integer_weights(formats);

                let mut ledger = Ledger::new([
                    CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
                    CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
                ])
                .unwrap();
                let mut authority = ResidencyAuthority::open(
                    &mut ledger,
                    &ResidencyRequest::new("device experts", 64 * CHUNK)
                        .device(ctx.uuid(), 8 * CHUNK),
                )
                .unwrap();
                let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

                let budget = ExpertBudget {
                    device: ctx.uuid(),
                    device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
                    device_cache_cap_bytes: 8 * CHUNK,
                    device_cache_leased_bytes: 0,
                    device_cache_largest_free_bytes: u64::MAX,
                    device_arena_free_bytes: 16 * MIB,
                    host_workspace_bytes: MIB,
                    host_buffer_bytes: MIB,
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
                    host: if on_device {
                        StrategyControl::Auto
                    } else {
                        StrategyControl::Required
                    },
                    host_placement: StrategyControl::Off,
                    ..ExpertPolicy::default()
                };
                let plan = moxie_plan::expert::compile_experts_with_weights(
                    &mlp(activation),
                    &combine(),
                    &route(),
                    &budget,
                    &policy,
                    None,
                    Some(ExpertKernels {
                        capability: ctx.capability(),
                        catalogue: &catalogue,
                    }),
                    formats,
                )
                .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));
                assert!(
                    plan.groups().iter().all(|g| g.placement().candidate()
                        == if on_device {
                            Candidate::Device
                        } else {
                            Candidate::Host
                        }),
                    "`required` must put every group on the device"
                );
                assert_eq!(plan.kernel().is_some(), on_device);

                let mut expert_roles = roles();
                expert_roles.per_expert = Some(role_pairs);
                let mut run = GroupedRun::admit(&mut ledger, plan, expert_roles, None)
                    .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));
                if on_device {
                    run.attach_device(&mut ledger, &ctx, &mut residency)
                        .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));
                }

                let x = to_bytes(&w.x);
                run.load_activations(&x).unwrap();
                run.run_to_completion(&mut authority, &mut src, TurnId::new(1), 0, u64::MAX)
                    .unwrap_or_else(|e| panic!("device {ordinal}: {e}"));

                // Slots first: a wrong reduction over right slots and a right
                // reduction over wrong slots are different defects.
                let want_slots = oracle_slots(&w, activation);
                assert_eq!(
                    as_u16(run.buffers().slots()),
                    want_slots,
                    "device {ordinal} {activation:?}: slots"
                );
                let got = as_u16(run.reduce(&w.coefficients).unwrap());
                assert_eq!(
                    got,
                    oracle_rows(&w, activation),
                    "device {ordinal} {activation:?}: rows"
                );
                compared += want_slots.len() + got.len();
                assert_eq!(run.stats().device_groups, if on_device { 5 } else { 0 });
                assert_eq!(run.stats().host_groups, if on_device { 0 } else { 5 });

                run.close(&mut ledger).unwrap();
                // Every lease is back, but the chunks are still resident. Retiring
                // them is what frees the ranges that live in the one real device
                // allocation, and the residency refuses to close until they are.
                authority.end_turn(TurnId::new(1));
                authority.retire_all(scope);
                residency.close(&mut authority).unwrap();
                authority.close(&mut ledger).unwrap();
                assert!(ledger.outstanding().is_empty());
                eprintln!(
                    "quantized expert UUID={} gate={activation:?} device={on_device} formats={formats:?} passed",
                    ctx.uuid()
                );
            }
        }
    }
    println!(
        "grouped expert device gate: {compared} BF16 components bit-identical to the oracle \
         across {count} device(s) and both gate transforms"
    );
}
