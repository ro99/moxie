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

/// A `RankContext` is exclusive per device and `cargo test` runs a binary's
/// tests in parallel threads. Serialising them is not a workaround: the
/// exclusivity is the property task 0007 established deliberately, and this
/// executable's injected faults are process-wide besides.
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

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
static CTX_SYNC_ERROR: AtomicI32 = AtomicI32::new(0);

/// One host allocation watched by address, and whether it was ever freed.
///
/// A byte counter cannot answer this question: the claim is about **these**
/// bytes, the activations an asynchronous `cuMemcpyHtoDAsync` reads, and a
/// total that happens to stay level proves nothing about one buffer. A pointer
/// comparison is exact and needs no serialisation beyond what this executable
/// already has.
static WATCHED_HOST_PTR: AtomicUsize = AtomicUsize::new(0);
static WATCHED_HOST_FREED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Task 0029's trap: how many more allocations on this thread succeed
    /// before one fails. `None` is disarmed. Per **thread**, because this
    /// executable's tests share a process and a global flag lands in an
    /// unrelated test's `format!`.
    static ARMED: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Whether the last arming reached its allocation.
    static FIRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn take_arming() -> bool {
    ARMED
        .try_with(|armed| match armed.get() {
            None => false,
            Some(0) => {
                armed.set(None);
                let _ = FIRED.try_with(|fired| fired.set(true));
                true
            }
            Some(n) => {
                armed.set(Some(n - 1));
                false
            }
        })
        .unwrap_or(false)
}

/// Run `body` with the `skip`-th allocation after arming failing, and report
/// whether that allocation was reached.
fn with_failure_at<T>(skip: usize, body: impl FnOnce() -> T) -> (T, bool) {
    /// Disarms however the scope ends. A panic past a plain assignment leaks
    /// the trap onto the next test on this thread.
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            let _ = ARMED.try_with(|armed| armed.set(None));
        }
    }

    let _ = FIRED.try_with(|fired| fired.set(false));
    let guard = Disarm;
    let _ = ARMED.try_with(|armed| armed.set(Some(skip)));
    let out = body();
    drop(guard);
    let fired = FIRED.try_with(std::cell::Cell::get).unwrap_or(false);
    (out, fired)
}

struct WatchingAllocator;

// SAFETY: every method forwards the caller's unmodified pointer and layout to
// the system allocator. The only addition is one address comparison.
unsafe impl std::alloc::GlobalAlloc for WatchingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if take_arming() {
            return core::ptr::null_mut();
        }
        // SAFETY: forwards the caller's unmodified layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        if ptr as usize == WATCHED_HOST_PTR.load(SeqCst) && ptr as usize != 0 {
            WATCHED_HOST_FREED.store(true, SeqCst);
        }
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { std::alloc::System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        if ptr as usize == WATCHED_HOST_PTR.load(SeqCst) && ptr as usize != 0 {
            WATCHED_HOST_FREED.store(true, SeqCst);
        }
        if take_arming() {
            return core::ptr::null_mut();
        }
        // SAFETY: forwards the caller's own pointer, layout and size.
        unsafe { std::alloc::System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static HOST_ALLOCATOR: WatchingAllocator = WatchingAllocator;

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
    let error = CTX_SYNC_ERROR.load(SeqCst);
    if error != 0 {
        return error;
    }
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
        paged_state_capacity: None,
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
                group: 1,
            },
            &[linear, gain],
        )
        .unwrap();
    let output = builder
        .node(OpParams::Residual { scale: 1.0 }, &[x, norm])
        .unwrap();
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
        paged_state_capacity: None,
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
    let _serial = one_at_a_time();
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

        // A fatal asynchronous status first surfaced by final readback is a
        // permanent loss even though the completion event was already ready.
        // Retry, direct retirement and turn sweeping must all keep the one
        // plan and its reservation quarantined. The nonfatal retry above
        // remains the contrasting recoverable path.
        let fatal_candidate = lower_selected(
            &selected,
            selected_workload(&selected, &ctx),
            &capability,
            &catalogue,
        )
        .unwrap();
        let mut fatal_ledger = test_ledger(&ctx);
        let fatal_plan = SelectedReservedPlan::admit(
            fatal_candidate,
            &selected,
            &capability,
            &catalogue,
            &mut fatal_ledger,
            &ctx,
        )
        .unwrap();
        let fatal_lease = fatal_plan
            .launch(
                &selected,
                &capability,
                &catalogue,
                &ctx,
                &stream,
                selected_bindings(&ctx, x, weight, gain, true),
            )
            .unwrap();
        READBACK_ERROR.store(700, SeqCst);
        let fatal = fatal_lease.finish().unwrap_err();
        READBACK_ERROR.store(0, SeqCst);
        assert_eq!(fatal.error.kind(), "device_lost");
        assert!(fatal.error.to_string().contains("final output readback"));
        assert_eq!(fatal.lease.state(), LeaseState::Lost);
        assert_eq!(fatal_ledger.outstanding().len(), 1);

        let readbacks_after_loss = READBACKS.load(SeqCst);
        let event_syncs_after_loss = EVENT_SYNCS.load(SeqCst);
        let fatal = fatal.lease.finish().unwrap_err();
        assert_eq!(fatal.error.kind(), "device_lost");
        assert_eq!(fatal.lease.state(), LeaseState::Lost);
        assert_eq!(READBACKS.load(SeqCst), readbacks_after_loss);
        assert_eq!(EVENT_SYNCS.load(SeqCst), event_syncs_after_loss);
        assert_eq!(fatal_ledger.outstanding().len(), 1);

        let fatal = fatal.lease.retire().unwrap_err();
        assert_eq!(fatal.error.kind(), "device_lost");
        assert_eq!(fatal.lease.state(), LeaseState::Lost);
        assert_eq!(fatal_ledger.outstanding().len(), 1);

        let frees_before_fatal_sweep = FREES.load(SeqCst);
        let mut fatal_turn = OperationTurn::new("fatal readback quarantine").unwrap();
        fatal_turn.hold(fatal.lease);
        let fatal_report = fatal_turn.release_turn();
        assert!(fatal_report.retired.is_empty());
        assert_eq!(fatal_report.held.len(), 1);
        assert_eq!(fatal_report.held[0].lease.state(), LeaseState::Lost);
        assert!(
            fatal_report.held[0]
                .reason
                .contains("final output readback")
        );
        drop(fatal_report);
        assert_eq!(FREES.load(SeqCst), frees_before_fatal_sweep);
        assert_eq!(fatal_ledger.outstanding().len(), 1);

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
                    actual.label.as_ref(),
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

/// Task 0028's review, finding 2: a launch that cannot establish its own
/// completion must keep **every** operand, not just its arena.
///
/// The first version took the weight leases and the activation source by
/// reference. After a failure between the first copy and an observed
/// completion, the caller still owned both: it could free the activation source
/// or release the leases and let the authority evict the weights, while
/// submitted work was still reading those bytes. Quarantining this run's own
/// activation/output arena protected none of it.
///
/// `cuEventRecord` is failed here for real, on real hardware, after the copy
/// and the launch have been submitted -- which is exactly the window.
#[test]
fn a_quantized_launch_that_cannot_prove_completion_keeps_its_operands() {
    let _serial = one_at_a_time();
    use moxie_executor::affine_linear::{AffineLaunch, AffineLinearRun, ResidentAffineWeight};
    use moxie_executor::residency::DeviceResidency;
    use moxie_executor::{ChunkSource, drain_reads, select_affine_linear_kernel};
    use moxie_format::affine::{AffineDescriptor, Grouping, IntWidth};
    use moxie_format::payload::ZeroPointSection;
    use moxie_format::scale::ScaleDtype;
    use moxie_memory::{
        AcquireRequest, Acquired, ArtifactId, ChunkId, Content, LogicalRange, ResidencyAuthority,
        ResidencyRequest, TensorSlot, TurnId, UseClass,
    };

    const OUT: usize = 32;
    const IN: usize = 64;
    const ROWS: u64 = 3;

    /// A symmetric per-channel INT8 tensor: two components, no zero points, so
    /// the fixture is as small as the contract allows.
    struct Fixture {
        artifact: ArtifactId,
        codes: Vec<u8>,
        scales: Vec<u8>,
    }

    impl ChunkSource for Fixture {
        fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> moxie_types::Result<()> {
            assert_eq!(chunk.artifact(), &self.artifact);
            let whole: &[u8] = match chunk.slot().role() {
                "codes" => &self.codes,
                "scales" => &self.scales,
                other => panic!("no component is bound to role {other:?}"),
            };
            let start = chunk.range().offset_bytes() as usize;
            into.copy_from_slice(&whole[start..start + into.len()]);
            Ok(())
        }
    }

    let ctx = RankContext::acquire(RankId(28_900), 0).unwrap();
    let stream = Stream::new(&ctx).unwrap();
    let capability = query_device(0).unwrap();
    let scope = Scope::Device(ctx.uuid());

    let descriptor = AffineDescriptor {
        width: IntWidth::Int8,
        out_features: OUT,
        in_features: IN,
        grouping: Grouping::PerOutputChannel,
        group_index: None,
        scale_dtype: ScaleDtype::F32,
    };
    let launch = AffineLaunch::derive(&descriptor, ZeroPointSection::Absent, ROWS).unwrap();
    let kernel = select_affine_linear_kernel(
        &moxie_kernels::affine_linear_catalogue(),
        &capability,
        WeightPrecision::expect(Precision::Int8),
        &launch,
    )
    .unwrap();

    let code_bytes = launch.code_bytes().unwrap();
    let scale_bytes = launch.scale_bytes().unwrap();
    let mut source = Fixture {
        artifact: ArtifactId::new("sha256:affine-linear-fault-fixture").unwrap(),
        codes: vec![1u8; code_bytes as usize],
        scales: (0..OUT).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
    };
    let align = |bytes: u64| bytes.div_ceil(256) * 256;
    let cap = align(code_bytes) + align(scale_bytes);

    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 << 20, 1 << 20).unwrap(),
        CapacitySnapshot::new(scope, 64 << 20, 1 << 20).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("fault fixture", cap).device(ctx.uuid(), cap),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

    fn resident<'ctx>(
        authority: &mut ResidencyAuthority,
        residency: &mut DeviceResidency<'ctx>,
        stream: &Stream<'ctx>,
        source: &mut Fixture,
        scope: Scope,
        role: &str,
        len: u64,
    ) -> moxie_memory::ResidencyLease {
        let chunk = ChunkId::new(
            source.artifact.clone(),
            TensorSlot::tensor(role).unwrap(),
            LogicalRange::new(0, len).unwrap(),
            1,
        );
        match authority
            .acquire(AcquireRequest {
                chunk: &chunk,
                destination: scope,
                now: 0,
                deadline: u64::MAX,
                class: UseClass::demand(Content::DenseSpine),
                turn: TurnId::new(1),
            })
            .unwrap_or_else(|e| panic!("acquiring {role}: {}", e.error))
        {
            Acquired::Ready(lease) => lease,
            Acquired::Pending { lease, work, .. } => {
                for order in &drain_reads(authority, source, work).unwrap() {
                    residency.perform_upload(authority, stream, order).unwrap();
                }
                lease
            }
        }
    }
    let codes = resident(
        &mut authority,
        &mut residency,
        &stream,
        &mut source,
        scope,
        "codes",
        code_bytes,
    );
    let scales = resident(
        &mut authority,
        &mut residency,
        &stream,
        &mut source,
        scope,
        "scales",
        scale_bytes,
    );

    let mut run = AffineLinearRun::admit(&mut ledger, &ctx, kernel, launch)
        .unwrap_or_else(|refused| panic!("admission: {}", refused.error));

    // **The activations are watched by address.** These are the host bytes a
    // `cuMemcpyHtoDAsync` reads, and the original finding was that dropping the
    // quarantined run ran `Vec::drop` on them while the copy might still be in
    // flight. Nothing observed that; the first regression checked the ledger and
    // the leases and then dropped the run.
    let activations = vec![0u8; launch.activation_bytes().unwrap() as usize];
    WATCHED_HOST_FREED.store(false, SeqCst);
    WATCHED_HOST_PTR.store(activations.as_ptr() as usize, SeqCst);

    // The window: the copy and the launch are submitted for real, and the
    // event that would prove they finished is refused.
    let records = RECORDS.load(SeqCst);
    RECORD_ERROR.store(1, SeqCst);
    let refused = run
        .run(
            &stream,
            &authority,
            &residency,
            ResidentAffineWeight {
                codes,
                scales,
                zero_points: None,
                group_index: None,
            },
            activations,
        )
        .expect_err("a refused event record must refuse the run");
    RECORD_ERROR.store(0, SeqCst);
    assert!(RECORDS.load(SeqCst) > records, "the record was attempted");

    // The operands are **not** handed back. Before this fix they were never
    // taken in the first place, so the caller could release these leases while
    // the launch was still reading them.
    assert!(refused.retained_operands());
    assert!(refused.weight.is_none() && refused.activations.is_none());

    // And the consequences of holding them are real, not advisory: the run
    // refuses to release its ranges, the ledger stays charged, and the
    // authority cannot close because its leases are still live.
    let held = run
        .close(&mut ledger)
        .expect_err("a quarantined run must not release ranges in flight");
    assert!(!ledger.outstanding().is_empty());
    assert!(
        authority.close(&mut ledger).is_err(),
        "the authority closed while a launch may still be reading its cache"
    );
    // Dropped, not released: a dropped quarantined run keeps its allocation and
    // its charge, exactly as `DeviceArena` already does. Withholding is the
    // point -- a context that cannot prove completion withholds forever rather
    // than advertising memory nothing can recover.
    assert!(
        !WATCHED_HOST_FREED.load(SeqCst),
        "the activations were freed before the run was even dropped"
    );
    drop(held);
    // **The claim, observed.** Dropping a quarantined run must not return the
    // host bytes an in-flight copy may be reading.
    assert!(
        !WATCHED_HOST_FREED.load(SeqCst),
        "dropping the quarantined run deallocated the activations the copy may \
         still be reading"
    );
    WATCHED_HOST_PTR.store(0, SeqCst);
}

/// A device allocation whose completion cannot be established is **not** freed.
///
/// `DeviceBuffer::drop` synchronised the context and then called
/// `cuMemFree_v2` whatever the synchronise returned. A failed
/// `cuCtxSynchronize` is the one answer that means "whether anything is still
/// reading this allocation is unknown", and freeing on it hands live pages back
/// to the driver -- the physical half of the same lifetime finding. The
/// allocation is permanently withheld instead.
#[test]
fn a_backing_whose_synchronize_fails_is_withheld_rather_than_freed() {
    let _guard = one_at_a_time();
    let ctx = RankContext::acquire(RankId(28_901), 0).expect("a rank context");

    // A real device allocation, dropped with the context synchronise made to
    // fail -- which is how a device in a bad state answers, and the state in
    // which "is anything still reading this?" has no answer.
    let buffer = moxie_cuda::DeviceBuffer::alloc(&ctx, 4096).expect("a device allocation");
    let frees = FREES.load(SeqCst);
    let syncs = SYNCS.load(SeqCst);
    CTX_SYNC_ERROR.store(4, SeqCst);
    drop(buffer);
    CTX_SYNC_ERROR.store(0, SeqCst);
    assert!(
        SYNCS.load(SeqCst) > syncs,
        "the drop did not synchronise the context at all, so it never asked"
    );
    assert_eq!(
        FREES.load(SeqCst),
        frees,
        "the allocation was freed even though the synchronise that would prove \
         nothing is reading it failed"
    );

    // The control, and it is load-bearing: the same drop with a working
    // synchronise **does** free. Without it, a drop that had stopped freeing
    // altogether would pass the assertion above.
    let buffer = moxie_cuda::DeviceBuffer::alloc(&ctx, 4096).expect("a device allocation");
    let frees = FREES.load(SeqCst);
    drop(buffer);
    assert_eq!(
        FREES.load(SeqCst),
        frees + 1,
        "a buffer whose synchronise succeeded was not freed, so the withholding \
         above proves nothing"
    );
}

/// Sweep the actual admission through its first non-firing position. Each
/// refusal must return all charges and physical device allocations; the first
/// successful run must carry the same launch, descriptor and arena as baseline.
#[allow(clippy::result_large_err)]
#[test]
fn every_allocation_in_a_quantized_admission_refuses_rather_than_aborting() {
    let _serial = one_at_a_time();
    use moxie_executor::affine_linear::{AffineLaunch, AffineLinearRun};
    use moxie_executor::select_affine_linear_kernel;
    use moxie_format::affine::{AffineDescriptor, Grouping, IntWidth};
    use moxie_format::payload::ZeroPointSection;
    use moxie_format::scale::ScaleDtype;

    let count = moxie_cuda::device_count().expect("device count");
    assert!(count > 0, "the hardware gate cannot pass without a device");
    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(28_902 + ordinal), ordinal).expect("rank context");
        let descriptor = AffineDescriptor {
            width: IntWidth::Int4,
            out_features: 32,
            in_features: 64,
            grouping: Grouping::Contiguous { size: 32 },
            group_index: None,
            scale_dtype: ScaleDtype::Bf16,
        };
        let launch = AffineLaunch::derive(&descriptor, ZeroPointSection::PerGroup, 4).unwrap();
        let capability = query_device(ordinal).unwrap();
        let catalogue = moxie_kernels::affine_linear_catalogue();
        let kernel = select_affine_linear_kernel(
            &catalogue,
            &capability,
            moxie_types::WeightPrecision::expect(moxie_types::Precision::Int4),
            &launch,
        )
        .unwrap();
        let mut baseline_ledger = test_ledger(&ctx);
        let baseline = AffineLinearRun::admit(
            &mut baseline_ledger,
            &ctx,
            kernel.try_clone().unwrap(),
            launch,
        )
        .unwrap();
        let expected = (
            baseline.arena_bytes(),
            format!("{:?}", baseline.launch()),
            format!("{:?}", baseline.descriptor()),
        );
        baseline.close(&mut baseline_ledger).unwrap();
        let mut completed = false;
        let mut fired_positions = 0;
        for skip in 0..512 {
            let mut ledger = test_ledger(&ctx);
            let kernel = kernel.try_clone().unwrap();
            let allocations = ALLOCS.load(SeqCst);
            let frees = FREES.load(SeqCst);
            let (result, fired) = with_failure_at(skip, || {
                AffineLinearRun::admit(&mut ledger, &ctx, kernel, launch)
            });
            if fired {
                fired_positions += 1;
                let refused = result.expect_err("a fired failure must refuse admission");
                assert_eq!(
                    refused.error.kind(),
                    "capacity_exceeded",
                    "position {skip}: {}",
                    refused.error
                );
                assert!(
                    refused.reservation.is_none(),
                    "reservation was not returned at {skip}"
                );
                drop(refused);
            } else {
                let run = result.expect("non-firing admission must succeed");
                assert_eq!(
                    (
                        run.arena_bytes(),
                        format!("{:?}", run.launch()),
                        format!("{:?}", run.descriptor())
                    ),
                    expected
                );
                run.close(&mut ledger).unwrap();
                completed = true;
            }
            assert_eq!(
                ALLOCS.load(SeqCst) - allocations,
                FREES.load(SeqCst) - frees,
                "physical allocation leaked at {skip}"
            );
            assert!(
                ledger.outstanding().is_empty(),
                "reservation leaked at {skip}"
            );
            for scope in ledger.scopes() {
                assert_eq!(ledger.scope_committed(scope), 0);
                for tier in Tier::valid_in(scope.kind()) {
                    assert_eq!(ledger.committed(scope, tier), 0);
                }
            }
            if completed {
                break;
            }
        }
        assert!(
            completed && fired_positions > 0,
            "sweep must reach a non-firing admission"
        );
        eprintln!(
            "{}: all {fired_positions} admission allocation positions refused; first non-firing run equals baseline",
            ctx.uuid()
        );
    }
}

/// A paged attention launch whose completion cannot be established keeps its
/// query, its pages and its frontier.
///
/// Task 0037. The same lifetime rule the quantized launch above proves, at the
/// one place where it also has to protect *history*: the committed frontier is
/// what later launches attend over, so a launch or an event record that fails
/// must leave it exactly where it was. A frontier advanced on an unproven copy
/// would make every later decode attend over bytes nothing wrote.
#[test]
#[cfg(feature = "paged-attention-test-hooks")]
fn a_paged_attention_failure_keeps_its_query_and_its_frontier() {
    let _serial = one_at_a_time();
    use moxie_executor::paged_attention::device::RawPagedFixture;
    use moxie_executor::{PageGeometry, PagedAttentionLaunch, PagedAttentionRun, Staging};
    use moxie_plan::Visibility;
    use moxie_types::PagePlacement;

    /// The placements a state authority would give for `rows` rows from
    /// `first`, in this fixture's geometry.
    ///
    /// Written out rather than driven through `moxie_state` because this file
    /// interposes the CUDA driver process-wide and its fixtures stay as small
    /// as the fault window allows; `paged_attention_device.rs` is where the two
    /// authorities meet for real.
    fn placements(first: u64, rows: u64, page_tokens: u64, table: &[u32]) -> Vec<PagePlacement> {
        let mut out = Vec::new();
        let mut done = 0;
        while done < rows {
            let position = first + done;
            let slot = position % page_tokens;
            let run = (page_tokens - slot).min(rows - done);
            out.push(PagePlacement {
                position,
                // Through the published mapping, because that is what the run
                // checks a write against: a placement computed from a formula
                // that disagreed with the table would be refused, which is the
                // binding working.
                physical_page: u64::from(table[(position / page_tokens) as usize]),
                slot,
                rows: run,
            });
            done += run;
        }
        out
    }

    const HEADS: u64 = 2;
    let geometry = PageGeometry {
        kv_heads: 1,
        head_dim: 64,
        page_tokens: 8,
        pages: 2,
    };
    let row_elements = (geometry.kv_heads * geometry.head_dim) as usize;
    let rows_bytes = |rows: usize, seed: u8| vec![seed; rows * row_elements * 2];

    let ctx = RankContext::acquire(RankId(37_001), 0).expect("a rank context");
    let stream = Stream::new(&ctx).expect("a stream");
    let capability = query_device(0).expect("query device 0");
    let layer = moxie_executor::AttentionLayer {
        geometry,
        heads: HEADS,
        scale: moxie_plan::reciprocal_sqrt_scale(64),
        visibility: Visibility::Causal,
    };
    let launch = |rows: u64, first_position: u64, history_rows: u64| {
        PagedAttentionLaunch::new(layer, rows, first_position, 0, history_rows)
            .expect("the fixture's launches are legal")
    };
    let descriptor = moxie_executor::select_paged_attention_kernel(
        &moxie_kernels::paged_attention_catalogue(),
        &capability,
        &launch(1, 0, 1),
    )
    .expect("this build declares a paged attention descriptor for this device");

    let mut ledger = test_ledger(&ctx);
    let run = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor,
        geometry,
        HEADS,
        1,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("admission fits this ledger");
    // No `moxie_state` authority anywhere in this file (it interposes the
    // CUDA driver process-wide, so its fixtures stay as small as the fault
    // window allows): every run here is driven directly through
    // `RawPagedFixture`.
    let mut run = RawPagedFixture::new(run);
    let table = vec![1u32, 0];
    run.publish_page_table(&stream, 0, table.clone())
        .map_err(|r| r.error)
        .expect("a reversed mapping is a mapping");
    let first_four = placements(0, 4, geometry.page_tokens, &table);
    run.write_rows(
        &stream,
        &first_four,
        rows_bytes(4, 0x21),
        rows_bytes(4, 0x22),
    )
    .map_err(|r| r.error)
    .expect("four rows fit");
    assert_eq!(run.run().written_rows(), 4);
    let committed = run
        .run()
        .read_rows(&first_four)
        .expect("read the rows back");

    // An append whose event record fails. The copies were submitted for real,
    // so their completion is unknown and the rows cannot be published.
    let fifth = placements(4, 1, geometry.page_tokens, &table);
    let fifth_keys = rows_bytes(1, 0x31);
    let fifth_values = rows_bytes(1, 0x32);
    // Watched by address, not by a byte total: the claim is about **this**
    // allocation, the one an in-flight `cuMemcpyHtoDAsync` may still read.
    WATCHED_HOST_FREED.store(false, SeqCst);
    WATCHED_HOST_PTR.store(fifth_keys.as_ptr() as usize, SeqCst);
    let records = RECORDS.load(SeqCst);
    RECORD_ERROR.store(1, SeqCst);
    let refused = run
        .write_rows(&stream, &fifth, fifth_keys, fifth_values)
        .expect_err("a write whose record failed must refuse");
    RECORD_ERROR.store(0, SeqCst);
    assert!(RECORDS.load(SeqCst) > records, "the record was attempted");
    assert!(
        refused.retained_source(),
        "a refusal after enqueue must keep the rows the copy may be reading"
    );
    assert_eq!(
        run.run().written_rows(),
        4,
        "the high-water mark advanced on a copy nothing proved"
    );
    // The run is quarantined, so it will not release ranges that may be in
    // flight, will not read its own pages back, and stays charged.
    assert!(
        run.run().read_rows(&first_four).is_err(),
        "a quarantined run read back"
    );
    let held = run
        .into_inner()
        .close(&mut ledger)
        .expect_err("a quarantined run must not release ranges in flight");
    assert!(!ledger.outstanding().is_empty());
    assert!(
        !WATCHED_HOST_FREED.load(SeqCst),
        "the row source was freed before the quarantined run was even dropped"
    );
    drop(held);
    assert!(
        !WATCHED_HOST_FREED.load(SeqCst),
        "dropping the quarantined run deallocated the row source the copy may \
         still be reading"
    );
    WATCHED_HOST_PTR.store(0, SeqCst);

    // The same window on the launch itself, on a fresh run: the copy is real,
    // the launch is refused, and the query is retained rather than handed back.
    let mut ledger = test_ledger(&ctx);
    let descriptor = moxie_executor::select_paged_attention_kernel(
        &moxie_kernels::paged_attention_catalogue(),
        &capability,
        &launch(1, 0, 1),
    )
    .expect("a descriptor");
    let run = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor,
        geometry,
        HEADS,
        1,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("admission fits this ledger");
    let mut run = RawPagedFixture::new(run);
    run.publish_page_table(&stream, 0, table.clone())
        .map_err(|r| r.error)
        .expect("a mapping");
    run.write_rows(
        &stream,
        &first_four,
        rows_bytes(4, 0x21),
        rows_bytes(4, 0x22),
    )
    .map_err(|r| r.error)
    .expect("four rows fit");
    assert_eq!(
        run.run().read_rows(&first_four).expect("read back"),
        committed,
        "the same rows through the same mapping are the same bytes"
    );
    // Done writing directly: hand the run back to attend through its own
    // public API, which never needed narrowing.
    let mut run = run.into_inner();

    let query = vec![0x3Cu8; (HEADS * geometry.head_dim) as usize * 2];
    WATCHED_HOST_FREED.store(false, SeqCst);
    WATCHED_HOST_PTR.store(query.as_ptr() as usize, SeqCst);
    let launches = LAUNCHES.load(SeqCst);
    LAUNCH_ERROR.store(1, SeqCst);
    LAUNCH_FAIL_COUNTDOWN.store(1, SeqCst);
    let refused = run
        .attend(&stream, &launch(1, 3, 4), query)
        .expect_err("a refused launch must refuse the attend");
    LAUNCH_FAIL_COUNTDOWN.store(0, SeqCst);
    LAUNCH_ERROR.store(0, SeqCst);
    assert!(LAUNCHES.load(SeqCst) > launches, "the launch was attempted");
    assert!(
        refused.retained_source(),
        "the query copy was submitted, so the query must be retained"
    );
    assert_eq!(run.written_rows(), 4, "a failed launch moved the frontier");
    let held = run
        .close(&mut ledger)
        .expect_err("a quarantined run released its ranges");
    assert!(
        !WATCHED_HOST_FREED.load(SeqCst),
        "the query was freed before the quarantined run was even dropped"
    );
    drop(held);
    assert!(
        !WATCHED_HOST_FREED.load(SeqCst),
        "dropping the quarantined run deallocated the query the copy may still \
         be reading"
    );
    WATCHED_HOST_PTR.store(0, SeqCst);

    // The page-table upload has the same ordering requirement: the encoded
    // bytes must be owned by `self.held` before the copy call, not after, so
    // a copy that refuses synchronously still leaves them retained rather
    // than dropping a local the function never reached. `publish_page_table`
    // builds that buffer from `table` internally, so unlike the two windows
    // above there is no host pointer this test ever held to watch -- this is
    // the same `None`-convention behavior the earlier windows also check, not
    // a second proof of the deallocation property.
    let mut ledger = test_ledger(&ctx);
    let descriptor = moxie_executor::select_paged_attention_kernel(
        &moxie_kernels::paged_attention_catalogue(),
        &capability,
        &launch(1, 0, 1),
    )
    .expect("a descriptor");
    let run = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor,
        geometry,
        HEADS,
        1,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("admission fits this ledger");
    let mut run = RawPagedFixture::new(run);
    let copies = COPIES.load(SeqCst);
    COPY_ERROR.store(1, SeqCst);
    let refused = run
        .publish_page_table(&stream, 0, table.clone())
        .expect_err("a page-table upload whose copy call failed must refuse");
    COPY_ERROR.store(0, SeqCst);
    assert!(COPIES.load(SeqCst) > copies, "the copy was attempted");
    assert!(
        refused.retained_source(),
        "a refusal after the copy call must keep the encoded table the driver \
         may still be reading"
    );
    assert!(
        run.into_inner().close(&mut ledger).is_err(),
        "a quarantined run released its ranges"
    );
}
