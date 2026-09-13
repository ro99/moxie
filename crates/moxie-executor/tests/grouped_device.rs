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
use moxie_executor::grouped_device::DeviceExperts;
use moxie_executor::residency::{DeviceResidency, ShardSource};
use moxie_graph::{CombineOrder, ExpertActivation, OpParams};
use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
use moxie_memory::{
    ArtifactId, CapacitySnapshot, Ledger, ResidencyAuthority, ResidencyRequest, TurnId,
};
use moxie_oracles::route;
use moxie_plan::expert::{Candidate, ExpertBudget, ExpertKernels, ExpertPolicy, compile_experts};
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
                device_arena_free_bytes: 16 * MIB,
                host_workspace_bytes: MIB,
                host_buffer_bytes: MIB,
                resident_experts: Vec::new(),
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
            run.attach_device(
                &mut ledger,
                &ctx,
                &mut residency,
                moxie_kernels::EXPERT_MLP_FATBIN,
            )
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
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        resident_experts: Vec::new(),
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
    run.attach_device(
        &mut ledger,
        &ctx,
        &mut residency,
        moxie_kernels::EXPERT_MLP_FATBIN,
    )
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
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        resident_experts: Vec::new(),
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
    run.attach_device(
        &mut ledger,
        &ctx,
        &mut residency,
        moxie_kernels::EXPERT_MLP_FATBIN,
    )
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
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        resident_experts: Vec::new(),
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
        let plan = compile_experts(
            &mlp(ExpertActivation::GeGlu),
            &combine(),
            &route(),
            &budget,
            &device_policy,
            None,
            Some(ExpertKernels {
                capability: ctx.capability(),
                catalogue: &catalogue,
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
        const NOT_AN_IMAGE: &[u8] = &[0u8; 64];
        let error = run
            .attach_device(&mut ledger, &ctx, &mut residency, NOT_AN_IMAGE)
            .expect_err("64 zero bytes are not a fatbin");
        assert!(!format!("{error}").is_empty());
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
        run.attach_device(
            &mut ledger,
            &ctx,
            &mut residency,
            moxie_kernels::EXPERT_MLP_FATBIN,
        )
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

/// A lease resolved through another authority's backing is refused.
///
/// A lease is an **offset**, and an offset means nothing without the allocation
/// it belongs to. An independent review drove authority A's leases through
/// authority B's backing on a real GPU and got a confident, different answer:
/// `perform_upload` had checked this since task 0020 and the launch path had
/// not. Both halves are checked here, and the same call with the right backing
/// succeeds -- so this is the check biting, not the fixture failing.
#[test]
fn a_launch_refuses_leases_that_belong_to_another_authority() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ctx = RankContext::acquire(RankId(0), 0).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();
    let dir = scratch("foreign-backing");
    let w = weights(0x0f0f_0f0f);
    let path = write_shard(&dir, &w);

    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 128 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(scope, 128 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut a = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("a", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
    )
    .unwrap();
    let mut b = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("b", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
    )
    .unwrap();
    let mut ra = DeviceResidency::create(&ctx, &mut a).unwrap();
    let rb = DeviceResidency::create(&ctx, &mut b).unwrap();

    let budget = ExpertBudget {
        device: ctx.uuid(),
        device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
        device_cache_cap_bytes: 8 * CHUNK,
        device_cache_leased_bytes: 0,
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        resident_experts: Vec::new(),
    };
    let policy = ExpertPolicy {
        device: StrategyControl::Required,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
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
    let group = plan.groups()[0].clone();

    let request = GroupedRun::request_for(&plan).unwrap();
    let reservation = ledger.admit(&request).unwrap();
    let mut experts = DeviceExperts::attach(
        &mut ledger,
        reservation,
        &ctx,
        &plan,
        moxie_kernels::EXPERT_MLP_FATBIN,
    )
    .unwrap();
    experts.load_activations(&to_bytes(&w.x)).unwrap();

    // Authority A's leases, uploaded through A's own backing.
    let mut src = source(&path);
    let (gate_up_chunk, down_chunk) = roles().chunks(group.expert(), plan.shape()).unwrap();
    let mut leases = Vec::new();
    for chunk in [&gate_up_chunk, &down_chunk] {
        let acquired = a
            .acquire(moxie_memory::AcquireRequest {
                chunk,
                destination: scope,
                now: 0,
                deadline: u64::MAX,
                class: moxie_memory::UseClass::demand(moxie_memory::Content::Expert),
                turn: TurnId::new(1),
            })
            .unwrap();
        match acquired {
            moxie_memory::Acquired::Ready(lease) => leases.push(lease),
            moxie_memory::Acquired::Pending { lease, work, .. } => {
                for order in moxie_executor::residency::drain_reads(&mut a, &mut src, work).unwrap()
                {
                    ra.perform_upload(&mut a, experts.stream(), &order).unwrap();
                }
                leases.push(lease);
            }
        }
    }

    let mut staging_rows = vec![0u8; 256];
    let mut staging_slots = vec![0u8; 256];
    let slot_bytes = (plan.slot_count() * HIDDEN * 2) as usize;
    let mut through_b = vec![0u8; slot_bytes];
    let refused = experts
        .run_group(
            &a,
            &group,
            &leases[0],
            &leases[1],
            &rb,
            moxie_executor::grouped::ExpertStaging {
                rows: &mut staging_rows,
                slots: &mut staging_slots,
            },
            &mut through_b,
        )
        .expect_err("authority A's leases must not resolve inside B's allocation");
    assert!(
        !refused.submission_unknown,
        "the refusal must come before anything is enqueued"
    );
    assert!(
        format!("{}", refused.error).contains("backing"),
        "the refusal must name the backing: {}",
        refused.error
    );
    assert!(
        through_b.iter().all(|byte| *byte == 0),
        "a refused launch must write nothing"
    );

    // The same call through A's own backing succeeds, so the check is what
    // refused and not the fixture.
    let mut through_a = vec![0u8; slot_bytes];
    experts
        .run_group(
            &a,
            &group,
            &leases[0],
            &leases[1],
            &ra,
            moxie_executor::grouped::ExpertStaging {
                rows: &mut staging_rows,
                slots: &mut staging_slots,
            },
            &mut through_a,
        )
        .expect("the right backing resolves the same leases");
    assert!(through_a.iter().any(|byte| *byte != 0));

    for lease in leases {
        a.release(lease).unwrap();
    }
    experts.close(&mut ledger).unwrap();
    a.end_turn(TurnId::new(1));
    a.retire_all(scope);
    ra.close(&mut a).unwrap();
    rb.close(&mut b).unwrap();
    a.close(&mut ledger).unwrap();
    b.close(&mut ledger).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// A launch index outside this attachment's own batch is refused before
/// anything is enqueued.
///
/// `ExpertGroup`'s indices are private now, so no caller can change one -- but
/// an attachment is built for one plan and `run_group` takes a group, so a group
/// from a *different* plan is the shape that survives that. A row index past the
/// batch reads outside the activation block; a slot index past it writes outside
/// the output range, and the host-slot check used to happen after execution.
#[test]
fn a_group_from_another_plan_is_refused_before_anything_is_enqueued() {
    let _serial = one_at_a_time();
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "the device lane requires real hardware");
    let ctx = RankContext::acquire(RankId(0), 0).unwrap();
    let scope = Scope::Device(ctx.uuid());
    let catalogue = moxie_kernels::expert_mlp_catalogue();
    let dir = scratch("stranger-group");
    let w = weights(0x0a0a_0a0a);
    let path = write_shard(&dir, &w);

    let budget = ExpertBudget {
        device: ctx.uuid(),
        device_pci_bus_id: ctx.capability().pci_bus_id.clone(),
        device_cache_cap_bytes: 8 * CHUNK,
        device_cache_leased_bytes: 0,
        device_arena_free_bytes: 16 * MIB,
        host_workspace_bytes: MIB,
        host_buffer_bytes: MIB,
        resident_experts: Vec::new(),
    };
    let policy = ExpertPolicy {
        device: StrategyControl::Required,
        host: StrategyControl::Auto,
        host_placement: StrategyControl::Off,
        ..ExpertPolicy::default()
    };
    let kernels = Some(ExpertKernels {
        capability: ctx.capability(),
        catalogue: &catalogue,
    });

    // The wide plan's first group names rows this attachment will not have.
    let wide = compile_experts(
        &mlp(ExpertActivation::GeGlu),
        &combine(),
        &route(),
        &budget,
        &policy,
        None,
        kernels,
    )
    .unwrap();
    let stranger = wide.groups()[0].clone();
    assert!(
        stranger.rows().iter().any(|row| *row >= 2),
        "the stranger must name a row the narrow attachment does not have"
    );

    // A two-row attachment with **eight** slots per row, admitted directly:
    // this case is about the attachment's own checks, so it needs no run around
    // it. The wide slot count is deliberate -- it leaves the assignment-count
    // check satisfied so the **row** bound is the one that has to bite, and a
    // fixture that trips three checks at once pins none of them.
    let narrow_route: Vec<u32> = (0..16).collect();
    let narrow_mlp = OpParams::ExpertMlp {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        // Sixteen experts at top-k 8 over two rows: sixteen slots, so the
        // assignment-count check is satisfied and the row bound is the one that
        // has to bite.
        experts: 16,
        top_k: 8,
        activation: ExpertActivation::GeGlu,
    };
    let narrow_combine = OpParams::Combine {
        hidden: HIDDEN,
        top_k: 8,
        order: CombineOrder::AscendingExpertId,
    };
    let narrow = compile_experts(
        &narrow_mlp,
        &narrow_combine,
        &narrow_route,
        &budget,
        &policy,
        None,
        kernels,
    )
    .unwrap();
    assert_eq!(narrow.rows(), 2);
    assert!(
        stranger.rows().len() as u64 <= narrow.slot_count(),
        "the assignment-count check must not fire first"
    );

    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("narrow", 64 * CHUNK).device(ctx.uuid(), 8 * CHUNK),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();
    let request = GroupedRun::request_for(&narrow).unwrap();
    let reservation = ledger.admit(&request).unwrap();
    let mut experts = DeviceExperts::attach(
        &mut ledger,
        reservation,
        &ctx,
        &narrow,
        moxie_kernels::EXPERT_MLP_FATBIN,
    )
    .unwrap();
    experts
        .load_activations(&to_bytes(&w.x[..(2 * HIDDEN) as usize]))
        .unwrap();

    // Real leases for the stranger's expert, so the refusal is about the
    // indices and not about an absent weight.
    let mut src = source(&path);
    let (gate_up_chunk, down_chunk) = roles().chunks(stranger.expert(), wide.shape()).unwrap();
    let mut leases = Vec::new();
    for chunk in [&gate_up_chunk, &down_chunk] {
        let acquired = authority
            .acquire(moxie_memory::AcquireRequest {
                chunk,
                destination: scope,
                now: 0,
                deadline: u64::MAX,
                class: moxie_memory::UseClass::demand(moxie_memory::Content::Expert),
                turn: TurnId::new(1),
            })
            .unwrap();
        match acquired {
            moxie_memory::Acquired::Ready(lease) => leases.push(lease),
            moxie_memory::Acquired::Pending { lease, work, .. } => {
                for order in
                    moxie_executor::residency::drain_reads(&mut authority, &mut src, work).unwrap()
                {
                    residency
                        .perform_upload(&mut authority, experts.stream(), &order)
                        .unwrap();
                }
                leases.push(lease);
            }
        }
    }

    let mut staging_rows = vec![0u8; 64];
    let mut staging_slots = vec![0u8; 64];
    let mut host_slots = vec![0u8; (narrow.slot_count() * HIDDEN * 2) as usize];
    let refused = experts
        .run_group(
            &authority,
            &stranger,
            &leases[0],
            &leases[1],
            &residency,
            moxie_executor::grouped::ExpertStaging {
                rows: &mut staging_rows,
                slots: &mut staging_slots,
            },
            &mut host_slots,
        )
        .expect_err("a group from another plan names indices this attachment does not have");
    assert!(
        !refused.submission_unknown,
        "the refusal must come before anything is enqueued"
    );
    let message = format!("{}", refused.error);
    assert!(
        message.contains("row ") && message.contains("batches"),
        "the refusal must name the row index it rejected: {message}"
    );
    // Nothing was written.
    assert!(host_slots.iter().all(|byte| *byte == 0));

    for lease in leases {
        authority.release(lease).unwrap();
    }
    experts.close(&mut ledger).unwrap();
    authority.end_turn(TurnId::new(1));
    authority.retire_all(scope);
    residency.close(&mut authority).unwrap();
    authority.close(&mut ledger).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
