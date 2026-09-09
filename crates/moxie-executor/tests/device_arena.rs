//! Real-device proof for the admitted suballocator. No model or kernel.
#![cfg(feature = "driver")]

use moxie_cuda::{Event, RankContext, Stream};
use moxie_executor::{
    ArenaUpload, DeviceArena, DeviceRange, LeaseState, OperationLease, OperationTurn,
};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_types::{DeviceTier, HostTier, RankId, Scope, Tier};

const MIB: u64 = 1024 * 1024;
const ARENA_BYTES: u64 = 16 * MIB;

// Test-only real stream gate: enqueue a bounded host function immediately before
// the actual event record. Copies remain real; the completion cannot race past
// the first sweep. No production API or fabricated event status is involved.
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::time::{Duration, Instant};
static BLOCK_NEXT: AtomicBool = AtomicBool::new(false);
static RELEASE: AtomicBool = AtomicBool::new(true);
static TIMED_OUT: AtomicBool = AtomicBool::new(false);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

unsafe extern "C" fn pending_work(_: *mut c_void) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !RELEASE.load(SeqCst) {
        if Instant::now() >= deadline {
            TIMED_OUT.store(true, SeqCst);
            break;
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int {
    // SAFETY: RTLD_NEXT finds the real CUDA ABI symbols after this executable.
    let (launch, record) = unsafe {
        let launch = dlsym((-1isize) as *mut c_void, c"cuLaunchHostFunc".as_ptr());
        let record = dlsym((-1isize) as *mut c_void, c"cuEventRecord".as_ptr());
        if launch.is_null() || record.is_null() {
            return 1;
        }
        (
            std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(
                    *mut c_void,
                    unsafe extern "C" fn(*mut c_void),
                    *mut c_void,
                ) -> c_int,
            >(launch),
            std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
            >(record),
        )
    };
    if BLOCK_NEXT.swap(false, SeqCst) {
        // SAFETY: stream comes unchanged from CUDA; callback uses static state,
        // calls no CUDA API and exits within ten seconds even on test failure.
        let result = unsafe { launch(stream, pending_work, std::ptr::null_mut()) };
        if result != 0 {
            return result;
        }
    }
    // SAFETY: event and stream are forwarded unchanged to the real driver.
    unsafe { record(event, stream) }
}

struct PendingGate;
impl PendingGate {
    fn arm() -> Self {
        RELEASE.store(false, SeqCst);
        TIMED_OUT.store(false, SeqCst);
        BLOCK_NEXT.store(true, SeqCst);
        Self
    }
}
impl Drop for PendingGate {
    fn drop(&mut self) {
        BLOCK_NEXT.store(false, SeqCst);
        RELEASE.store(true, SeqCst);
    }
}

fn admitted(ctx: &RankContext) -> (Ledger, Reservation) {
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Device(ctx.uuid()), 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut request = PlanRequest::new("device arena", ["resident"]).unwrap();
    request
        .buffer(BufferRequest::new(
            "physical weights",
            Scope::Device(ctx.uuid()),
            Tier::Device(DeviceTier::PackedResidentWeights),
            ARENA_BYTES,
            StageSpan { first: 0, last: 0 },
        ))
        .unwrap();
    request
        .buffer(BufferRequest::new(
            "one retained upload source",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            8 * MIB,
            StageSpan { first: 0, last: 0 },
        ))
        .unwrap();
    let reservation = ledger.admit(&request).unwrap();
    (ledger, reservation)
}

fn finish_upload<'ctx>(
    mut lease: OperationLease<Event<'ctx>, ArenaUpload<'ctx>>,
    expected: &[u8],
) -> DeviceRange<'ctx> {
    let mut readback = vec![0; expected.len()];
    lease.readback(&mut readback).unwrap();
    assert_eq!(readback, expected);
    let (_, upload) = lease.retire().unwrap();
    let (range, source) = upload.finish();
    assert_eq!(source, expected);
    range
}

#[test]
fn admitted_device_arena_reuses_only_completed_ranges_on_every_device() {
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "device lane requires real hardware");

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let stream = Stream::new(&ctx).unwrap();
        let before = ctx.memory_info().unwrap().0;
        let (mut ledger, reservation) = admitted(&ctx);
        let mut arena = DeviceArena::create(
            &ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            ARENA_BYTES,
            format!("weights on {}", ctx.uuid()),
        )
        .unwrap();
        let during = ctx.memory_info().unwrap().0;
        assert!(during < before, "physical arena must spend device memory");

        let first = arena.allocate(8 * MIB, 256, "importer").unwrap();
        let mut foreign_ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Device(ctx.uuid()), 64 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let refused = arena.close(&mut foreign_ledger).unwrap_err();
        assert_eq!(refused.error.kind(), "invalid_request");
        arena = refused.arena;
        let refused = arena.close(&mut ledger).unwrap_err();
        assert!(refused.error.to_string().contains("remain live"));
        arena = refused.arena;
        let first_generation = first.key().generation;
        let second = arena.allocate(3 * MIB, 128, "workspace").unwrap();
        let third = arena.allocate(5 * MIB, 64, "persistent").unwrap();
        let refused = arena.allocate(256, 256, "overflow").unwrap_err();
        assert_eq!(refused.occupancy.free_bytes, 0);
        assert_eq!(refused.occupancy.live_allocations, 3);

        let first_source = vec![0x19; 8 * MIB as usize];
        let mut first_use = first
            .prepare_upload(first_source.clone(), "first range upload")
            .unwrap();
        let gate = PendingGate::arm();
        first_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        assert!(!BLOCK_NEXT.load(SeqCst), "real event hook must have run");
        first_use.cancel();
        let mut turn = OperationTurn::new("cancelled first sweep").unwrap();
        turn.hold(first_use);
        let report = turn.release_turn();
        assert!(
            report.retired.is_empty(),
            "pending range retired on {}",
            ctx.uuid()
        );
        assert_eq!(report.held.len(), 1);
        let held = report.held.into_iter().next().unwrap();
        assert_eq!(held.lease.state(), LeaseState::Cancelled);
        assert_eq!(held.lease.resource().source(), first_source);
        assert_eq!(
            arena
                .allocate(1, 1, "pending reuse")
                .unwrap_err()
                .occupancy
                .free_bytes,
            0
        );
        let mut turn = OperationTurn::new("second sweep without next token").unwrap();
        turn.hold(held.lease);
        drop(gate);
        turn.synchronize().unwrap();
        assert!(!TIMED_OUT.load(SeqCst), "pending gate timed out");
        let mut report = turn.release_turn();
        assert!(report.is_clean());
        assert_eq!(report.retired.len(), 1);
        let (first, source) = report.retired.pop().unwrap().resource.finish();
        assert_eq!(source, first_source);
        eprintln!(
            "PASS controlled pending/cancel/first+second sweep on {}",
            ctx.uuid()
        );
        // A completed upload is safe to use again; verify the exact offset bytes.
        let mut first_use = first
            .prepare_upload(first_source.clone(), "first range readback")
            .unwrap();
        first_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        let first = finish_upload(first_use, &first_source);
        let first_key = first.key();
        let first = arena.transfer(first, "executor").unwrap();
        assert_eq!(first.key(), first_key);
        assert_eq!(first.owner(), "executor");

        let second_source = vec![0x2a; 3 * MIB as usize];
        let mut second_use = second
            .prepare_upload(second_source.clone(), "cancelled range upload")
            .unwrap();
        second_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        second_use.cancel();
        let second = finish_upload(second_use, &second_source);

        let third_source = vec![0x3b; 5 * MIB as usize];
        let mut third_use = third
            .prepare_upload(third_source.clone(), "persistent range upload")
            .unwrap();
        third_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        let third = finish_upload(third_use, &third_source);

        arena.release(second).unwrap();
        arena.release(first).unwrap();
        arena.release(third).unwrap();
        assert_eq!(arena.occupancy().largest_free_bytes, ARENA_BYTES);
        let whole = arena.allocate(ARENA_BYTES, 256, "coalesced owner").unwrap();
        assert_eq!(whole.offset(), 0);
        assert!(whole.key().generation > first_generation);
        arena.release(whole).unwrap();
        assert!(arena.outstanding().is_empty());

        arena.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
        let after = ctx.memory_info().unwrap().0;
        assert!(
            after >= during.saturating_add(ARENA_BYTES),
            "checked close must return the physical allocation"
        );
        eprintln!("PASS admitted device arena on {}", ctx.uuid());
    }
}
