//! Actual CUDA boundary regressions. This test executable interposes only its
//! own driver calls; production crates contain no injection switches. Each
//! unfaulted call forwards to libcuda, so buffers/events are real on every card.
#![cfg(feature = "driver")]

use moxie_cuda::{Event, RankContext, Stream};
use moxie_executor::{Lease, LeaseState};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
use moxie_types::{DeviceTier, HostTier, RankId, Scope, Tier};
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering::SeqCst};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COPIES: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static SYNCS: AtomicUsize = AtomicUsize::new(0);
static COPY_ERROR: AtomicI32 = AtomicI32::new(0);
static RECORD_ERROR: AtomicI32 = AtomicI32::new(0);
static FREE_ERROR: AtomicI32 = AtomicI32::new(0);

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

fn ledger(ctx: &RankContext) -> Ledger {
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

#[test]
fn real_driver_admission_submission_and_cleanup_are_fail_closed() {
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "driver lane needs real hardware");
    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let stream = Stream::new(&ctx).unwrap();
        let mut ledger = ledger(&ctx);
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
        // These are intentional, named quarantines. Stream drain only protects
        // test teardown; it must not make the leases reclaimable again.
        stream.synchronize().unwrap();
        assert_eq!(ledger.outstanding().len(), 6);
        eprintln!("PASS driver boundary faults on {}", ctx.uuid());
    }
}
