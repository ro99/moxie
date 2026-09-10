//! Actual CUDA boundary regressions. This test executable interposes only its
//! own driver calls; production crates contain no injection switches. Each
//! unfaulted call forwards to libcuda, so buffers/events are real on every card.
#![cfg(feature = "driver")]

use moxie_cuda::{Event, RankContext, Stream};
use moxie_executor::{DeviceArena, Lease, LeaseState, PlanAdmitRefused, ReservedPlan};
use moxie_graph::{
    Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry, TensorSpec,
    ValueRole,
};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
use moxie_plan::{Phase, ResourceWorkload, lower};
use moxie_types::{
    ActivationPrecision, DeviceTier, Dim, HostTier, Precision, RankId, Scope, SymbolId, Tier,
    WeightPrecision,
};
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering::SeqCst};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COPIES: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static SYNCS: AtomicUsize = AtomicUsize::new(0);
static COPY_ERROR: AtomicI32 = AtomicI32::new(0);
static RECORD_ERROR: AtomicI32 = AtomicI32::new(0);
static FREE_ERROR: AtomicI32 = AtomicI32::new(0);
static ALLOC_ERROR: AtomicI32 = AtomicI32::new(0);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

macro_rules! forward {
    ($name:literal, $signature:ty, $($arg:expr),*) => {{
        // SAFETY: RTLD_NEXT resolves the real driver definition after this
        // executable. Every signature below matches the audited CUDA ABI.
        let function: $signature = unsafe {
            let symbol = dlsym((-1isize) as *mut c_void, concat!($name, "\0").as_ptr().cast());
            assert!(!symbol.is_null(), "missing real driver symbol");
            std::mem::transmute(symbol)
        };
        // SAFETY: pointers and handles are passed unchanged from the driver
        // wrapper; this test adds no lifetime or size conversions.
        unsafe { function($($arg),*) }
    }};
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int {
    ALLOCS.fetch_add(1, SeqCst);
    let error = ALLOC_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!(
        "cuMemAlloc_v2",
        unsafe extern "C" fn(*mut u64, usize) -> c_int,
        ptr,
        bytes
    )
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyHtoDAsync_v2(
    ptr: u64,
    source: *const c_void,
    bytes: usize,
    stream: *mut c_void,
) -> c_int {
    COPIES.fetch_add(1, SeqCst);
    let error = COPY_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!(
        "cuMemcpyHtoDAsync_v2",
        unsafe extern "C" fn(u64, *const c_void, usize, *mut c_void) -> c_int,
        ptr,
        source,
        bytes,
        stream
    )
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int {
    let error = RECORD_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!(
        "cuEventRecord",
        unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
        event,
        stream
    )
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemFree_v2(ptr: u64) -> c_int {
    FREES.fetch_add(1, SeqCst);
    let error = FREE_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!("cuMemFree_v2", unsafe extern "C" fn(u64) -> c_int, ptr)
}
#[unsafe(no_mangle)]
unsafe extern "C" fn cuCtxSynchronize() -> c_int {
    SYNCS.fetch_add(1, SeqCst);
    forward!("cuCtxSynchronize", unsafe extern "C" fn() -> c_int,)
}

fn test_ledger(ctx: &RankContext) -> Ledger {
    Ledger::new([
        CapacitySnapshot::new(Scope::Device(ctx.uuid()), 1 << 20, 1024).unwrap(),
        CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).unwrap(),
    ])
    .unwrap()
}
fn acquire<'ctx>(
    ledger: &mut Ledger,
    ctx: &RankContext,
    device: u64,
    host: u64,
    wrong_tier: bool,
) -> Lease<Event<'ctx>> {
    let mut request = PlanRequest::new("driver regression", ["copy"]).unwrap();
    for (scope, tier, bytes) in [
        (
            Scope::Device(ctx.uuid()),
            Tier::Device(if wrong_tier {
                DeviceTier::PackedResidentWeights
            } else {
                DeviceTier::TransferStaging
            }),
            device,
        ),
        (Scope::Host, Tier::Host(HostTier::Pageable), host),
    ] {
        request
            .buffer(BufferRequest::new(
                format!("{scope}"),
                scope,
                tier,
                bytes,
                StageSpan { first: 0, last: 0 },
            ))
            .unwrap();
    }
    let reservation = ledger.admit(&request).unwrap();
    Lease::acquire(ledger, reservation, "driver regression").unwrap()
}

fn arena_reservation(
    ledger: &mut Ledger,
    ctx: &RankContext,
    device: u64,
    host: u64,
) -> moxie_memory::Reservation {
    let mut request = PlanRequest::new("arena driver regression", ["resident"]).unwrap();
    request
        .buffer(BufferRequest::new(
            "arena",
            Scope::Device(ctx.uuid()),
            Tier::Device(DeviceTier::PackedResidentWeights),
            device,
            StageSpan { first: 0, last: 0 },
        ))
        .unwrap();
    request
        .buffer(BufferRequest::new(
            "arena source",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            host,
            StageSpan { first: 0, last: 0 },
        ))
        .unwrap();
    ledger.admit(&request).unwrap()
}

const PLAN_ROWS: SymbolId = SymbolId(93);
const PLAN_ORACLE: OracleId = OracleId("driver-plan-fault");

fn resource_graph() -> Graph {
    let mut registry = OracleRegistry::new();
    registry
        .register(
            Op::Linear,
            PLAN_ORACLE,
            OracleEvidence {
                implementation: "driver_faults",
                test_module: "driver_faults",
            },
        )
        .unwrap();
    let mut builder = GraphBuilder::new(PLAN_ORACLE, PLAN_ROWS);
    let input = builder.input(
        "x",
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![Dim::symbol(PLAN_ROWS), Dim::constant(8)],
        ),
    );
    let weight = builder
        .weight(
            "w",
            TensorSpec::new(
                ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
                vec![Dim::constant(8), Dim::constant(8)],
            ),
        )
        .unwrap();
    let output = builder
        .node(
            OpParams::Linear {
                in_features: 8,
                out_features: 8,
                bias: false,
            },
            &[input, weight],
        )
        .unwrap();
    builder.finish(output, &registry).unwrap()
}

fn resource_workload(graph: &Graph, ctx: &RankContext) -> ResourceWorkload {
    ResourceWorkload {
        phase: Phase::Prefill,
        rows: 2,
        visible_tokens: 32_768,
        branch_rows: 2,
        output: graph.output(),
        device: ctx.uuid(),
    }
}

#[test]
fn real_driver_admission_submission_and_cleanup_are_fail_closed() {
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "driver lane needs real hardware");
    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let stream = Stream::new(&ctx).unwrap();
        let mut ledger = test_ledger(&ctx);

        // A graph/UUID validation refusal occurs before the resource-plan path
        // can call cuMemAlloc.
        let graph = resource_graph();
        let other = resource_graph();
        let candidate = lower(&graph, resource_workload(&graph, &ctx)).unwrap();
        let allocations = ALLOCS.load(SeqCst);
        assert!(matches!(
            ReservedPlan::admit(candidate, &other, &mut ledger, &ctx),
            Err(PlanAdmitRefused::Invalid { .. })
        ));
        assert_eq!(ALLOCS.load(SeqCst), allocations);
        assert!(ledger.outstanding().is_empty());

        // A physical activation-arena allocation failure returns the candidate
        // and releases the just-admitted reservation.
        let candidate = lower(&graph, resource_workload(&graph, &ctx)).unwrap();
        ALLOC_ERROR.store(2, SeqCst);
        let allocations = ALLOCS.load(SeqCst);
        assert!(matches!(
            ReservedPlan::admit(candidate, &graph, &mut ledger, &ctx),
            Err(PlanAdmitRefused::Invalid { .. })
        ));
        ALLOC_ERROR.store(0, SeqCst);
        assert_eq!(ALLOCS.load(SeqCst), allocations + 1);
        assert!(ledger.outstanding().is_empty());

        // Final plan free is checked before releasing the full envelope. A
        // failed free quarantines the arena and a second close cannot retry it.
        let mut plan_fault_ledger = test_ledger(&ctx);
        let candidate = lower(&graph, resource_workload(&graph, &ctx)).unwrap();
        let plan = ReservedPlan::admit(candidate, &graph, &mut plan_fault_ledger, &ctx).unwrap();
        FREE_ERROR.store(1, SeqCst);
        let frees = FREES.load(SeqCst);
        let refused = plan.close(&mut plan_fault_ledger).unwrap_err();
        FREE_ERROR.store(0, SeqCst);
        assert_eq!(FREES.load(SeqCst), frees + 1);
        assert_eq!(plan_fault_ledger.outstanding().len(), 1);
        let refused = refused.plan.close(&mut plan_fault_ledger).unwrap_err();
        assert_eq!(refused.error.kind(), "device_lost");
        assert_eq!(FREES.load(SeqCst), frees + 1);
        drop(refused);

        // A separate happy plan closes with one free and no context-wide sync.
        let mut happy_ledger = test_ledger(&ctx);
        let candidate = lower(&graph, resource_workload(&graph, &ctx)).unwrap();
        let allocations = ALLOCS.load(SeqCst);
        let plan = ReservedPlan::admit(candidate, &graph, &mut happy_ledger, &ctx).unwrap();
        assert_eq!(ALLOCS.load(SeqCst), allocations + 1);
        let frees = FREES.load(SeqCst);
        let syncs = SYNCS.load(SeqCst);
        plan.close(&mut happy_ledger).unwrap();
        assert_eq!(FREES.load(SeqCst), frees + 1);
        assert_eq!(SYNCS.load(SeqCst), syncs);
        assert!(happy_ledger.outstanding().is_empty());

        // The real cuMemAlloc entry point must not run before refusal. Capacity
        // exceeds length in the final variant: account allocated host RAM too.
        for (device, host, wrong_tier, capacity) in [
            (512, 512, false, 4096),
            (0, 4096, false, 4096),
            (4096, 0, false, 4096),
            (4096, 4096, true, 4096),
        ] {
            let lease = acquire(&mut ledger, &ctx, device, host, wrong_tier);
            let mut source = Vec::with_capacity(capacity);
            source.resize(4096, 7);
            let allocations = ALLOCS.load(SeqCst);
            let refused = lease.prepare_upload(&ctx, source).unwrap_err();
            assert_eq!(ALLOCS.load(SeqCst), allocations);
            assert_eq!(refused.source.len(), 4096);
            refused.lease.retire(&mut ledger).unwrap();
        }
        let lease = acquire(&mut ledger, &ctx, 512, 512, false);
        let mut source = Vec::with_capacity(4096);
        source.resize(512, 7);
        let allocations = ALLOCS.load(SeqCst);
        let refused = lease.prepare_upload(&ctx, source).unwrap_err();
        assert_eq!(ALLOCS.load(SeqCst), allocations);
        refused.lease.retire(&mut ledger).unwrap();

        // Happy path proves interposition forwards to real allocation/copy/free.
        let lease = acquire(&mut ledger, &ctx, 512, 512, false);
        let allocations = ALLOCS.load(SeqCst);
        let mut lease = lease.prepare_upload(&ctx, vec![7; 512]).unwrap();
        assert_eq!(
            ALLOCS.load(SeqCst),
            allocations + 1,
            "allocation interposition must be live"
        );
        let copies = COPIES.load(SeqCst);
        lease.submit(&stream, Event::new(&ctx).unwrap()).unwrap();
        assert_eq!(
            COPIES.load(SeqCst),
            copies + 1,
            "copy interposition must be live"
        );
        let mut readback = vec![0; 512];
        lease.readback(&mut readback).unwrap();
        assert_eq!(readback, vec![7; 512]);
        let frees = FREES.load(SeqCst);
        let syncs = SYNCS.load(SeqCst);
        let (_, source) = lease.retire(&mut ledger).unwrap();
        assert_eq!(source, vec![7; 512]);
        assert_eq!(FREES.load(SeqCst), frees + 1);
        assert_eq!(
            SYNCS.load(SeqCst),
            syncs,
            "retirement must not synchronize the context"
        );
        assert!(ledger.outstanding().is_empty());

        for (copy_error, record_error, free_error) in [
            (700, 0, 0),
            (1, 0, 0),
            (0, 700, 0),
            (0, 400, 0),
            (0, 0, 700),
            (0, 0, 1),
        ] {
            let lease = acquire(&mut ledger, &ctx, 512, 512, false);
            let mut lease = lease.prepare_upload(&ctx, vec![7; 512]).unwrap();
            let event = Event::new(&ctx).unwrap();
            COPY_ERROR.store(copy_error, SeqCst);
            RECORD_ERROR.store(record_error, SeqCst);
            let submission = lease.submit(&stream, event);
            COPY_ERROR.store(0, SeqCst);
            RECORD_ERROR.store(0, SeqCst);
            if free_error == 0 {
                assert!(submission.is_err());
                assert_eq!(lease.state(), LeaseState::Lost);
            } else {
                submission.unwrap();
                lease.synchronize().unwrap();
            }
            FREE_ERROR.store(free_error, SeqCst);
            let held_before = ledger.outstanding().len();
            let refused = lease.retire(&mut ledger).unwrap_err();
            FREE_ERROR.store(0, SeqCst);
            assert_eq!(ledger.outstanding().len(), held_before);
            assert_eq!(refused.lease.state(), LeaseState::Lost);
            let frees = FREES.load(SeqCst);
            let refused = refused.lease.retire(&mut ledger).unwrap_err();
            assert_eq!(refused.error.kind(), "device_lost");
            drop(refused);
            assert_eq!(
                FREES.load(SeqCst),
                frees,
                "quarantine must not retry free in Drop"
            );
        }

        // The arena path uses the same real boundary but one allocation serves
        // every range. Refusal must precede allocation.
        let foreign_uuid =
            moxie_types::DeviceUuid::parse("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff").unwrap();
        let mut cross_device_ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Device(ctx.uuid()), 4096, 0).unwrap(),
            CapacitySnapshot::new(Scope::Device(foreign_uuid), 4096, 0).unwrap(),
        ])
        .unwrap();
        let mut cross_device = PlanRequest::new("cross-device arena", ["resident"]).unwrap();
        for (label, scope) in [
            ("selected device", Scope::Device(ctx.uuid())),
            ("foreign device", Scope::Device(foreign_uuid)),
        ] {
            cross_device
                .buffer(BufferRequest::new(
                    label,
                    scope,
                    Tier::Device(DeviceTier::PackedResidentWeights),
                    4096,
                    StageSpan { first: 0, last: 0 },
                ))
                .unwrap();
        }
        let reservation = cross_device_ledger.admit(&cross_device).unwrap();
        let allocations = ALLOCS.load(SeqCst);
        let refused = DeviceArena::create(
            &cross_device_ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            4096,
            "cross-device arena",
        )
        .unwrap_err();
        assert_eq!(refused.error.kind(), "invalid_request");
        assert_eq!(ALLOCS.load(SeqCst), allocations);
        cross_device_ledger.release(refused.reservation).unwrap();

        for capacity in [0, 1, 257] {
            let reservation = arena_reservation(&mut ledger, &ctx, 4096, 4096);
            let allocations = ALLOCS.load(SeqCst);
            let refused = DeviceArena::create(
                &ledger,
                reservation,
                &ctx,
                DeviceTier::PackedResidentWeights,
                capacity,
                "invalid capacity",
            )
            .unwrap_err();
            assert_eq!(refused.error.kind(), "invalid_request");
            assert_eq!(ALLOCS.load(SeqCst), allocations);
            ledger.release(refused.reservation).unwrap();
        }

        let reservation = arena_reservation(&mut ledger, &ctx, 512, 512);
        let allocations = ALLOCS.load(SeqCst);
        let refused = DeviceArena::create(
            &ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            4096,
            "oversized arena",
        )
        .unwrap_err();
        assert_eq!(ALLOCS.load(SeqCst), allocations);
        ledger.release(refused.reservation).unwrap();

        // A real allocator failure returns the only reservation authority.
        let reservation = arena_reservation(&mut ledger, &ctx, 4096, 4096);
        ALLOC_ERROR.store(2, SeqCst);
        let refused = DeviceArena::create(
            &ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            4096,
            "failed arena",
        )
        .unwrap_err();
        ALLOC_ERROR.store(0, SeqCst);
        assert_eq!(refused.error.kind(), "capacity_exceeded");
        ledger.release(refused.reservation).unwrap();

        // Copy and record failures occur after the range is retained. They
        // quarantine the range and parent arena rather than advertising reuse.
        for (copy_error, record_error) in [(700, 0), (1, 0), (0, 700), (0, 400)] {
            let mut fault_ledger = test_ledger(&ctx);
            let reservation = arena_reservation(&mut fault_ledger, &ctx, 4096, 4096);
            let mut arena = DeviceArena::create(
                &fault_ledger,
                reservation,
                &ctx,
                DeviceTier::PackedResidentWeights,
                4096,
                "fault arena",
            )
            .unwrap();
            let range = arena.allocate(4096, 256, "fault range").unwrap();
            let mut lease = range.prepare_upload(vec![7; 4096], "fault upload").unwrap();
            COPY_ERROR.store(copy_error, SeqCst);
            RECORD_ERROR.store(record_error, SeqCst);
            assert!(lease.submit(&stream, Event::new(&ctx).unwrap()).is_err());
            COPY_ERROR.store(0, SeqCst);
            RECORD_ERROR.store(0, SeqCst);
            assert_eq!(lease.state(), LeaseState::Lost);
            drop(lease);
            assert_eq!(arena.outstanding().len(), 1);
            assert_eq!(fault_ledger.outstanding().len(), 1);
            drop(arena); // named quarantine: physical bytes and charge survive
        }

        // Final free is checked before ledger release and never retried after
        // an ambiguous failure.
        let mut fault_ledger = test_ledger(&ctx);
        let reservation = arena_reservation(&mut fault_ledger, &ctx, 4096, 4096);
        let arena = DeviceArena::create(
            &fault_ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            4096,
            "free fault arena",
        )
        .unwrap();
        FREE_ERROR.store(1, SeqCst);
        let frees = FREES.load(SeqCst);
        let refused = arena.close(&mut fault_ledger).unwrap_err();
        FREE_ERROR.store(0, SeqCst);
        assert_eq!(FREES.load(SeqCst), frees + 1);
        assert_eq!(fault_ledger.outstanding().len(), 1);
        let mut refused = refused;
        assert!(refused.arena.allocate(256, 256, "quarantined").is_err());
        let refused = refused.arena.close(&mut fault_ledger).unwrap_err();
        assert_eq!(refused.error.kind(), "device_lost");
        assert_eq!(
            FREES.load(SeqCst),
            frees + 1,
            "quarantine cannot retry free"
        );
        drop(refused);

        // Happy close: one allocation, one free, no context-wide synchronize,
        // then and only then the parent charge disappears.
        let mut close_ledger = test_ledger(&ctx);
        let reservation = arena_reservation(&mut close_ledger, &ctx, 4096, 4096);
        let arena = DeviceArena::create(
            &close_ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            4096,
            "close arena",
        )
        .unwrap();
        let allocations = ALLOCS.load(SeqCst);
        let mut arena = arena;
        let a = arena.allocate(1024, 256, "a").unwrap();
        let b = arena.allocate(3072, 256, "b").unwrap();
        assert_eq!(
            ALLOCS.load(SeqCst),
            allocations,
            "suballocation must not call cuMemAlloc"
        );
        arena.release(a).unwrap();
        arena.release(b).unwrap();
        let frees = FREES.load(SeqCst);
        let syncs = SYNCS.load(SeqCst);
        arena.close(&mut close_ledger).unwrap();
        assert_eq!(FREES.load(SeqCst), frees + 1);
        assert_eq!(SYNCS.load(SeqCst), syncs);
        assert!(close_ledger.outstanding().is_empty());

        // These are intentional, named quarantines. Stream drain only protects
        // test teardown; it must not make the leases reclaimable again.
        stream.synchronize().unwrap();
        assert_eq!(ledger.outstanding().len(), 6);
        eprintln!("PASS driver boundary faults on {}", ctx.uuid());
    }
}
