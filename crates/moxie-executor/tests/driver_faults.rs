//! Actual CUDA boundary regressions. This test executable interposes only its
//! own driver calls; production crates contain no injection switches. Each
//! unfaulted call forwards to libcuda, so buffers/events are real on every card.
#![cfg(feature = "driver")]

use moxie_cuda::{Event, RankContext, Stream, query_device};
use moxie_executor::{
    DeviceArena, Lease, LeaseState, OperationTurn, OwnedBinding, PlanAdmitRefused, ReservedPlan,
    SelectedAdmitRefused, SelectedReservedPlan, selected_resource_request,
};
use moxie_graph::{
    Graph, GraphBuilder, Op, OpParams, OracleEvidence, OracleId, OracleRegistry, TensorSpec,
    ValueRole,
};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
use moxie_plan::{Phase, ResourceWorkload, lower, lower_selected};
use moxie_types::{
    ActivationPrecision, DeviceTier, Dim, HostTier, Precision, RankId, Scope, SymbolId,
    TensorLayout, Tier, WeightPrecision,
};
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering::SeqCst};
use std::time::{Duration, Instant};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COPIES: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static SYNCS: AtomicUsize = AtomicUsize::new(0);
static COPY_ERROR: AtomicI32 = AtomicI32::new(0);
static RECORD_ERROR: AtomicI32 = AtomicI32::new(0);
static FREE_ERROR: AtomicI32 = AtomicI32::new(0);
static ALLOC_ERROR: AtomicI32 = AtomicI32::new(0);
static LAUNCHES: AtomicUsize = AtomicUsize::new(0);
static RECORDS: AtomicUsize = AtomicUsize::new(0);
static EVENT_SYNCS: AtomicUsize = AtomicUsize::new(0);
static STREAM_SYNCS: AtomicUsize = AtomicUsize::new(0);
static READBACKS: AtomicUsize = AtomicUsize::new(0);
static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
static BLOCK_NEXT_RECORD: AtomicBool = AtomicBool::new(false);
static RELEASE_GATE: AtomicBool = AtomicBool::new(true);
static GATE_TIMED_OUT: AtomicBool = AtomicBool::new(false);
static LAUNCH_ERROR: AtomicI32 = AtomicI32::new(0);
static LAUNCH_FAIL_COUNTDOWN: AtomicUsize = AtomicUsize::new(0);
static EVENT_SYNC_ERROR: AtomicI32 = AtomicI32::new(0);
static EVENT_QUERY_ERROR: AtomicI32 = AtomicI32::new(0);
static READBACK_ERROR: AtomicI32 = AtomicI32::new(0);

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
    RECORDS.fetch_add(1, SeqCst);
    let error = RECORD_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    if BLOCK_NEXT_RECORD.swap(false, SeqCst) {
        let result = forward!(
            "cuLaunchHostFunc",
            unsafe extern "C" fn(
                *mut c_void,
                unsafe extern "C" fn(*mut c_void),
                *mut c_void,
            ) -> c_int,
            stream,
            pending_chain_work,
            std::ptr::null_mut()
        );
        if result != 0 {
            return result;
        }
    }
    forward!(
        "cuEventRecord",
        unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
        event,
        stream
    )
}

unsafe extern "C" fn pending_chain_work(_: *mut c_void) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !RELEASE_GATE.load(SeqCst) {
        if Instant::now() >= deadline {
            GATE_TIMED_OUT.store(true, SeqCst);
            break;
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

struct PendingChainGate;
impl PendingChainGate {
    fn arm() -> Self {
        RELEASE_GATE.store(false, SeqCst);
        GATE_TIMED_OUT.store(false, SeqCst);
        BLOCK_NEXT_RECORD.store(true, SeqCst);
        Self
    }
}
impl Drop for PendingChainGate {
    fn drop(&mut self) {
        BLOCK_NEXT_RECORD.store(false, SeqCst);
        RELEASE_GATE.store(true, SeqCst);
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventSynchronize(event: *mut c_void) -> c_int {
    EVENT_SYNCS.fetch_add(1, SeqCst);
    let error = EVENT_SYNC_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!(
        "cuEventSynchronize",
        unsafe extern "C" fn(*mut c_void) -> c_int,
        event
    )
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventQuery(event: *mut c_void) -> c_int {
    let error = EVENT_QUERY_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!(
        "cuEventQuery",
        unsafe extern "C" fn(*mut c_void) -> c_int,
        event
    )
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuStreamSynchronize(stream: *mut c_void) -> c_int {
    STREAM_SYNCS.fetch_add(1, SeqCst);
    forward!(
        "cuStreamSynchronize",
        unsafe extern "C" fn(*mut c_void) -> c_int,
        stream
    )
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuMemcpyDtoH_v2(destination: *mut c_void, source: u64, bytes: usize) -> c_int {
    READBACKS.fetch_add(1, SeqCst);
    let error = READBACK_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
    forward!(
        "cuMemcpyDtoH_v2",
        unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int,
        destination,
        source,
        bytes
    )
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuModuleGetFunction(
    function: *mut *mut c_void,
    module: *mut c_void,
    name: *const c_char,
) -> c_int {
    LOOKUPS.fetch_add(1, SeqCst);
    forward!(
        "cuModuleGetFunction",
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
        function,
        module,
        name
    )
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuLaunchKernel(
    function: *mut c_void,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
    shared: u32,
    stream: *mut c_void,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> c_int {
    LAUNCHES.fetch_add(1, SeqCst);
    let countdown = LAUNCH_FAIL_COUNTDOWN.load(SeqCst);
    if countdown != 0 && LAUNCH_FAIL_COUNTDOWN.fetch_sub(1, SeqCst) == 1 {
        return LAUNCH_ERROR.load(SeqCst);
    }
    forward!(
        "cuLaunchKernel",
        unsafe extern "C" fn(
            *mut c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut c_void,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> c_int,
        function,
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared,
        stream,
        params,
        extra
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

fn selected_graph() -> (
    Graph,
    moxie_graph::ValueId,
    moxie_graph::ValueId,
    moxie_graph::ValueId,
) {
    let mut registry = OracleRegistry::new();
    for op in [Op::Linear, Op::RmsNorm, Op::Residual] {
        registry
            .register(
                op,
                PLAN_ORACLE,
                OracleEvidence {
                    implementation: "driver_faults",
                    test_module: "selected chain census",
                },
            )
            .unwrap();
    }
    let activation = |shape| {
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            shape,
        )
    };
    let weight_spec = |shape| {
        TensorSpec::new(
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            shape,
        )
    };
    let mut builder = GraphBuilder::new(PLAN_ORACLE, PLAN_ROWS);
    let x = builder.input(
        "x",
        activation(vec![Dim::symbol(PLAN_ROWS), Dim::constant(8)]),
    );
    let weight = builder
        .weight(
            "weight",
            weight_spec(vec![Dim::constant(8), Dim::constant(8)]),
        )
        .unwrap();
    let linear = builder
        .node(
            OpParams::Linear {
                in_features: 8,
                out_features: 8,
                bias: false,
            },
            &[x, weight],
        )
        .unwrap();
    let gain = builder
        .weight("gain", weight_spec(vec![Dim::constant(8)]))
        .unwrap();
    let norm = builder
        .node(
            OpParams::RmsNorm {
                hidden: 8,
                eps: 3.5,
            },
            &[linear, gain],
        )
        .unwrap();
    let output = builder.node(OpParams::Residual, &[x, norm]).unwrap();
    (builder.finish(output, &registry).unwrap(), x, weight, gain)
}

fn selected_workload(graph: &Graph, ctx: &RankContext) -> ResourceWorkload {
    ResourceWorkload {
        phase: Phase::Decode,
        rows: 1,
        visible_tokens: 1,
        branch_rows: 1,
        output: graph.output(),
        device: ctx.uuid(),
    }
}

fn bf16_words(words: impl IntoIterator<Item = u16>) -> Vec<u8> {
    let words = words.into_iter();
    let (lower, _) = words.size_hint();
    let mut bytes = Vec::with_capacity(lower * 2);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn selected_bindings(
    ctx: &RankContext,
    x: moxie_graph::ValueId,
    weight: moxie_graph::ValueId,
    gain: moxie_graph::ValueId,
    include_weights: bool,
) -> Vec<OwnedBinding> {
    let mut bindings = vec![OwnedBinding {
        value: x,
        role: ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
        shape: vec![1, 8],
        layout: TensorLayout::ContiguousRowMajorV1,
        device: ctx.uuid(),
        bytes: bf16_words((0..8).map(|i| if i % 2 == 0 { 0x4040 } else { 0x4080 })),
    }];
    if include_weights {
        bindings.push(OwnedBinding {
            value: weight,
            role: ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            shape: vec![8, 8],
            layout: TensorLayout::ContiguousRowMajorV1,
            device: ctx.uuid(),
            bytes: bf16_words((0..64).map(|i| if i / 8 == i % 8 { 0x3f80 } else { 0 })),
        });
        bindings.push(OwnedBinding {
            value: gain,
            role: ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            shape: vec![8],
            layout: TensorLayout::ContiguousRowMajorV1,
            device: ctx.uuid(),
            bytes: bf16_words([0x3f80; 8]),
        });
    }
    bindings
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

        // Task 0012 call census: one admitted physical allocation, three
        // retained uploads, four ordered kernel launches (RMS is two symbols),
        // one event, no stream/context synchronization, then exactly one final
        // event synchronization and D2H readback. All four symbols resolve
        // before the first launch.
        let capability = query_device(ordinal).unwrap();
        let catalogue = moxie_kernels::bf16_chain_catalogue();
        let (selected, x, weight, gain) = selected_graph();

        // Binding errors are checked before package lookup, copy or launch and
        // return the still-owned plan for explicit close.
        for mutation in 0..7 {
            let candidate = lower_selected(
                &selected,
                selected_workload(&selected, &ctx),
                &capability,
                &catalogue,
            )
            .unwrap();
            let mut refusal_ledger = test_ledger(&ctx);
            let plan = SelectedReservedPlan::admit(
                candidate,
                &selected,
                &capability,
                &catalogue,
                &mut refusal_ledger,
                &ctx,
            )
            .unwrap();
            let mut bindings = selected_bindings(&ctx, x, weight, gain, true);
            match mutation {
                0 => {
                    bindings.pop(); // missing gain
                }
                1 => {
                    bindings[1].value = x; // duplicate x and missing weight
                }
                2 => {
                    bindings[0].device =
                        moxie_types::DeviceUuid::parse("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff")
                            .unwrap();
                }
                3 => bindings[0].shape = vec![8, 1],
                4 => bindings[0].bytes[0..2].copy_from_slice(&0x7f80u16.to_le_bytes()),
                5 => {
                    let mut overallocated = Vec::with_capacity(bindings[0].bytes.len() + 8);
                    overallocated.extend_from_slice(&bindings[0].bytes);
                    bindings[0].bytes = overallocated;
                }
                6 => bindings[0].value = selected.nodes()[0].output,
                _ => unreachable!(),
            }
            let copies = COPIES.load(SeqCst);
            let launches = LAUNCHES.load(SeqCst);
            let lookups = LOOKUPS.load(SeqCst);
            let refused = plan
                .launch(&selected, &capability, &catalogue, &ctx, &stream, bindings)
                .unwrap_err();
            assert_eq!(refused.error.kind(), "invalid_request");
            assert!(refused.held.is_none());
            assert_eq!(COPIES.load(SeqCst), copies);
            assert_eq!(LAUNCHES.load(SeqCst), launches);
            assert_eq!(LOOKUPS.load(SeqCst), lookups);
            refused.plan.unwrap().close(&mut refusal_ledger).unwrap();
            assert!(refusal_ledger.outstanding().is_empty());
        }

        if ordinal == 0 {
            // A failure on the second kernel proves no later RMS-apply or
            // residual launch occurs and the already-submitted plan is lost,
            // complete with ranges and upload sources.
            let candidate = lower_selected(
                &selected,
                selected_workload(&selected, &ctx),
                &capability,
                &catalogue,
            )
            .unwrap();
            let mut fault_ledger = test_ledger(&ctx);
            let plan = SelectedReservedPlan::admit(
                candidate,
                &selected,
                &capability,
                &catalogue,
                &mut fault_ledger,
                &ctx,
            )
            .unwrap();
            let launches = LAUNCHES.load(SeqCst);
            let records = RECORDS.load(SeqCst);
            LAUNCH_ERROR.store(1, SeqCst);
            LAUNCH_FAIL_COUNTDOWN.store(2, SeqCst);
            let refused = plan
                .launch(
                    &selected,
                    &capability,
                    &catalogue,
                    &ctx,
                    &stream,
                    selected_bindings(&ctx, x, weight, gain, true),
                )
                .unwrap_err();
            LAUNCH_FAIL_COUNTDOWN.store(0, SeqCst);
            LAUNCH_ERROR.store(0, SeqCst);
            assert_eq!(LAUNCHES.load(SeqCst), launches + 2);
            assert_eq!(RECORDS.load(SeqCst), records);
            assert!(
                refused
                    .error
                    .to_string()
                    .contains("node 1 kernel bf16-rms-norm-v1"),
                "{}",
                refused.error
            );
            let held = refused.held.unwrap();
            assert_eq!(held.state(), LeaseState::Lost);
            assert!(held.resource().has_plan());
            assert_eq!(held.resource().retained_source_count(), 3);
            drop(held); // intentional named quarantine after ambiguous launch

            // Event-record ambiguity likewise returns the whole lost resource.
            let candidate = lower_selected(
                &selected,
                selected_workload(&selected, &ctx),
                &capability,
                &catalogue,
            )
            .unwrap();
            let plan = SelectedReservedPlan::admit(
                candidate,
                &selected,
                &capability,
                &catalogue,
                &mut fault_ledger,
                &ctx,
            )
            .unwrap();
            RECORD_ERROR.store(1, SeqCst);
            let refused = plan
                .launch(
                    &selected,
                    &capability,
                    &catalogue,
                    &ctx,
                    &stream,
                    selected_bindings(&ctx, x, weight, gain, true),
                )
                .unwrap_err();
            RECORD_ERROR.store(0, SeqCst);
            assert_eq!(refused.held.as_ref().unwrap().state(), LeaseState::Lost);
            assert!(
                refused
                    .error
                    .to_string()
                    .contains("completion event record")
            );
            drop(refused);

            // Query, synchronize and readback refusals each return the lease.
            // Nonfatal injected status can then be cleared and retried without
            // losing the sole ownership token.
            let candidate = lower_selected(
                &selected,
                selected_workload(&selected, &ctx),
                &capability,
                &catalogue,
            )
            .unwrap();
            let mut recovery_ledger = test_ledger(&ctx);
            let plan = SelectedReservedPlan::admit(
                candidate,
                &selected,
                &capability,
                &catalogue,
                &mut recovery_ledger,
                &ctx,
            )
            .unwrap();
            let lease = plan
                .launch(
                    &selected,
                    &capability,
                    &catalogue,
                    &ctx,
                    &stream,
                    selected_bindings(&ctx, x, weight, gain, true),
                )
                .unwrap();
            EVENT_QUERY_ERROR.store(1, SeqCst);
            let mut query_turn = OperationTurn::new("query refusal").unwrap();
            query_turn.hold(lease);
            let query_report = query_turn.release_turn();
            EVENT_QUERY_ERROR.store(0, SeqCst);
            assert_eq!(query_report.held.len(), 1);
            let query_held = query_report.held.into_iter().next().unwrap();
            assert!(
                query_held.reason.contains("completion event for [node"),
                "{}",
                query_held.reason
            );
            assert!(
                query_held.reason.contains("kernel "),
                "{}",
                query_held.reason
            );
            let lease = query_held.lease;
            assert_eq!(lease.state(), LeaseState::InFlight);

            EVENT_SYNC_ERROR.store(1, SeqCst);
            let refused = lease.finish().unwrap_err();
            EVENT_SYNC_ERROR.store(0, SeqCst);
            assert!(refused.error.to_string().contains("completion event"));
            assert_eq!(refused.lease.state(), LeaseState::InFlight);
            READBACK_ERROR.store(1, SeqCst);
            let refused = refused.lease.finish().unwrap_err();
            READBACK_ERROR.store(0, SeqCst);
            assert!(refused.error.to_string().contains("final output readback"));
            let result = refused.lease.finish().unwrap();
            result.plan.close(&mut recovery_ledger).unwrap();
            assert!(recovery_ledger.outstanding().is_empty());

            // A failed physical free returns the selected plan quarantined and
            // leaves its reservation visible; no Drop retry can manufacture a
            // clean close.
            let candidate = lower_selected(
                &selected,
                selected_workload(&selected, &ctx),
                &capability,
                &catalogue,
            )
            .unwrap();
            let mut free_ledger = test_ledger(&ctx);
            let plan = SelectedReservedPlan::admit(
                candidate,
                &selected,
                &capability,
                &catalogue,
                &mut free_ledger,
                &ctx,
            )
            .unwrap();
            FREE_ERROR.store(1, SeqCst);
            let refused = plan.close(&mut free_ledger).unwrap_err();
            FREE_ERROR.store(0, SeqCst);
            assert_eq!(free_ledger.outstanding().len(), 1);
            let refused = refused.plan.close(&mut free_ledger).unwrap_err();
            assert_eq!(refused.error.kind(), "device_lost");
            drop(refused);
        }

        let candidate = lower_selected(
            &selected,
            selected_workload(&selected, &ctx),
            &capability,
            &catalogue,
        )
        .unwrap();
        let request = selected_resource_request(&candidate).unwrap();
        assert_eq!(
            request.stages(),
            [
                "linear",
                "rms-reduce",
                "rms-apply",
                "residual",
                "terminal-output"
            ]
        );
        let expected_request = [
            (
                "weights",
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::PackedResidentWeights),
                512,
                StageSpan::inclusive(0, 4),
            ),
            (
                "activations",
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::Activations),
                768,
                StageSpan::inclusive(0, 4),
            ),
            (
                "workspace",
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::KernelWorkspace),
                256,
                StageSpan::inclusive(1, 2),
            ),
            (
                "retained-upload-sources",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                160,
                StageSpan::inclusive(0, 4),
            ),
            (
                "final-output-readback",
                Scope::Host,
                Tier::Host(HostTier::Pageable),
                16,
                StageSpan::at(4),
            ),
        ];
        assert_eq!(request.buffers().len(), expected_request.len());
        for (actual, expected) in request.buffers().iter().zip(expected_request) {
            assert_eq!(
                (
                    actual.label.as_str(),
                    actual.scope,
                    actual.tier,
                    actual.bytes,
                    actual.live,
                ),
                expected
            );
        }

        // The output is allocated while all upload sources are still owned by
        // the operation. Exactly 160 usable host bytes therefore rejects this
        // 176-byte peak before the one physical device allocation is attempted.
        let tight_candidate = lower_selected(
            &selected,
            selected_workload(&selected, &ctx),
            &capability,
            &catalogue,
        )
        .unwrap();
        let mut tight_ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Device(ctx.uuid()), 1 << 20, 1024).unwrap(),
            CapacitySnapshot::new(Scope::Host, 161, 1).unwrap(),
        ])
        .unwrap();
        let allocations_before_tight_refusal = ALLOCS.load(SeqCst);
        let refusal = SelectedReservedPlan::admit(
            tight_candidate,
            &selected,
            &capability,
            &catalogue,
            &mut tight_ledger,
            &ctx,
        )
        .unwrap_err();
        let SelectedAdmitRefused::Rejected { rejection, .. } = refusal else {
            panic!("the exact host peak must be an admission rejection")
        };
        let host_refusal = rejection.report.scope(Scope::Host).unwrap();
        assert_eq!(host_refusal.admissible_bytes, 160);
        assert_eq!(host_refusal.request_peak_bytes, 176);
        assert_eq!(ALLOCS.load(SeqCst), allocations_before_tight_refusal);
        assert!(tight_ledger.outstanding().is_empty());

        let mut selected_ledger = test_ledger(&ctx);
        let counts = (
            ALLOCS.load(SeqCst),
            COPIES.load(SeqCst),
            LAUNCHES.load(SeqCst),
            RECORDS.load(SeqCst),
            EVENT_SYNCS.load(SeqCst),
            STREAM_SYNCS.load(SeqCst),
            SYNCS.load(SeqCst),
            READBACKS.load(SeqCst),
            LOOKUPS.load(SeqCst),
        );
        let plan = SelectedReservedPlan::admit(
            candidate,
            &selected,
            &capability,
            &catalogue,
            &mut selected_ledger,
            &ctx,
        )
        .unwrap();
        let outstanding = selected_ledger.outstanding();
        assert_eq!(outstanding.len(), 1);
        let record = &outstanding[0];
        assert_eq!(record.scope_charges.len(), 2);
        assert!(record.scope_charges.contains(&(Scope::Host, 176)));
        assert!(
            record
                .scope_charges
                .contains(&(Scope::Device(ctx.uuid()), 1536))
        );
        assert_eq!(record.charges.len(), 4);
        for charge in [
            (Scope::Host, Tier::Host(HostTier::Pageable), 176),
            (
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::PackedResidentWeights),
                512,
            ),
            (
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::Activations),
                768,
            ),
            (
                Scope::Device(ctx.uuid()),
                Tier::Device(DeviceTier::KernelWorkspace),
                256,
            ),
        ] {
            assert!(record.charges.contains(&charge));
        }
        assert_eq!(ALLOCS.load(SeqCst), counts.0 + 1);
        let lease = plan
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, true),
            )
            .unwrap();
        assert_eq!(COPIES.load(SeqCst), counts.1 + 3);
        assert_eq!(LAUNCHES.load(SeqCst), counts.2 + 4);
        assert_eq!(RECORDS.load(SeqCst), counts.3 + 1);
        assert_eq!(EVENT_SYNCS.load(SeqCst), counts.4);
        assert_eq!(STREAM_SYNCS.load(SeqCst), counts.5);
        assert_eq!(SYNCS.load(SeqCst), counts.6);
        assert_eq!(READBACKS.load(SeqCst), counts.7);
        assert_eq!(LOOKUPS.load(SeqCst), counts.8 + 4);
        let first = lease.finish().unwrap();
        assert_eq!(EVENT_SYNCS.load(SeqCst), counts.4 + 1);
        assert_eq!(READBACKS.load(SeqCst), counts.7 + 1);
        assert_eq!(first.plan.bound_weight_count(), 2);

        // A second execution resolves its complete package again but uploads
        // only x; neither immutable weight has a second H2D copy.
        let second_copies = COPIES.load(SeqCst);
        let second_launches = LAUNCHES.load(SeqCst);
        let refused = first
            .plan
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, true),
            )
            .unwrap_err();
        assert_eq!(refused.error.kind(), "invalid_request");
        assert_eq!(COPIES.load(SeqCst), second_copies);
        assert_eq!(LAUNCHES.load(SeqCst), second_launches);
        let second = refused
            .plan
            .unwrap()
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, false),
            )
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(COPIES.load(SeqCst), second_copies + 1);
        second.plan.close(&mut selected_ledger).unwrap();
        assert!(selected_ledger.outstanding().is_empty());

        // The same real chain is held before its completion event. The first
        // no-next-token sweep returns the whole cancelled lease; after the
        // bounded gate opens, the second sweep returns the complete plan and
        // all three upload sources for explicit cleanup.
        let (selected, x, weight, gain) = selected_graph();
        let candidate = lower_selected(
            &selected,
            selected_workload(&selected, &ctx),
            &capability,
            &catalogue,
        )
        .unwrap();
        let mut cancellation_ledger = test_ledger(&ctx);
        let plan = SelectedReservedPlan::admit(
            candidate,
            &selected,
            &capability,
            &catalogue,
            &mut cancellation_ledger,
            &ctx,
        )
        .unwrap();
        let gate = PendingChainGate::arm();
        let mut lease = plan
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, true),
            )
            .unwrap();
        assert!(!BLOCK_NEXT_RECORD.load(SeqCst));
        lease.cancel();
        let mut first_sweep = OperationTurn::new("selected cancellation first sweep").unwrap();
        first_sweep.hold(lease);
        let first_report = first_sweep.release_turn();
        assert!(first_report.retired.is_empty());
        assert_eq!(first_report.held.len(), 1);
        let held = first_report.held.into_iter().next().unwrap();
        assert_eq!(held.lease.state(), LeaseState::Cancelled);
        assert!(held.lease.resource().has_plan());
        assert_eq!(held.lease.resource().retained_source_count(), 3);
        let mut second_sweep = OperationTurn::new("selected cancellation second sweep").unwrap();
        second_sweep.hold(held.lease);
        drop(gate);
        second_sweep.synchronize().unwrap();
        assert!(!GATE_TIMED_OUT.load(SeqCst));
        let mut second_report = second_sweep.release_turn();
        assert!(second_report.is_clean());
        assert_eq!(second_report.retired.len(), 1);
        let operation = second_report.retired.pop().unwrap().resource;
        assert!(operation.has_plan());
        assert_eq!(operation.retained_source_count(), 3);
        let (plan, sources) = operation.into_parts();
        assert_eq!(sources.len(), 1);
        assert_eq!(plan.bound_weight_count(), 2);

        // Sweep recovery has the same immutable-weight handoff as `finish`:
        // rebinding is refused, while an input-only execution reuses them.
        let copies_before_rebind = COPIES.load(SeqCst);
        let launches_before_rebind = LAUNCHES.load(SeqCst);
        let refused = plan
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, true),
            )
            .unwrap_err();
        assert_eq!(refused.error.kind(), "invalid_request");
        assert_eq!(COPIES.load(SeqCst), copies_before_rebind);
        assert_eq!(LAUNCHES.load(SeqCst), launches_before_rebind);
        let recovered = refused
            .plan
            .unwrap()
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, false),
            )
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(recovered.plan.bound_weight_count(), 2);
        recovered.plan.close(&mut cancellation_ledger).unwrap();
        assert!(cancellation_ledger.outstanding().is_empty());
        eprintln!(
            "PASS selected chain census and controlled cancellation on {}",
            ctx.uuid()
        );

        // These are intentional, named quarantines. Stream drain only protects
        // test teardown; it must not make the leases reclaimable again.
        stream.synchronize().unwrap();
        assert_eq!(ledger.outstanding().len(), 6);
        eprintln!("PASS driver boundary faults on {}", ctx.uuid());
    }
}
