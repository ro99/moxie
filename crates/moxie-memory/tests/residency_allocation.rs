//! Separate executable: measure the residency authority's heap, with no
//! concurrent test perturbing the counter.
//!
//! Two numbers this task predeclares and this file measures rather than
//! asserts from inspection:
//!
//! * **A resident-chunk acquire allocates nothing.** It is the hot path of a
//!   routed step -- one `acquire` per expert per layer per token -- and an
//!   allocation there is a per-token allocation. The measurement is also what
//!   forced the index to be nested by scope: a `(Scope, ChunkId)` key cannot be
//!   looked up from a borrow, so every hit would clone two `String`s to ask
//!   whether it was already resident.
//! * **A long demand/evict history retains nothing.** The same shape as the
//!   arena's own history gate: capacity that grows with the number of chunks
//!   ever seen, rather than with the number resident, is a leak that only shows
//!   up after a long session.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering::SeqCst};

use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger, LogicalRange,
    Outcome, PendingWork, ResidencyAuthority, ResidencyRequest, TensorSlot, TurnId, UseClass,
    WorkOrder,
};
use moxie_types::Scope;

struct Counter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: all allocation and deallocation are forwarded to System unchanged.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size() as isize, SeqCst);
            CALLS.fetch_add(1, SeqCst);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, SeqCst);
        // SAFETY: pointer/layout are the original allocation supplied by caller.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: Counter = Counter;

/// The counter is process-global, so two probes running at once measure each
/// other. Every test in this file takes this lock for its whole body.
static PROBE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn probe() -> std::sync::MutexGuard<'static, ()> {
    PROBE.lock().unwrap_or_else(|e| e.into_inner())
}

const CHUNK: u64 = 1_024;

fn chunk(index: u32) -> ChunkId {
    ChunkId::new(
        ArtifactId::new("allocation-probe").unwrap(),
        TensorSlot::expert("experts_gate_up", index).unwrap(),
        LogicalRange::new(u64::from(index) * CHUNK, CHUNK).unwrap(),
        1,
    )
}

fn request(chunk: &ChunkId, now: u64) -> AcquireRequest<'_> {
    AcquireRequest {
        chunk,
        destination: Scope::Host,
        now,
        deadline: u64::MAX,
        class: UseClass::demand(Content::Expert),
        turn: TurnId::new(1),
    }
}

fn open(ledger: &mut Ledger, cap: u64) -> ResidencyAuthority {
    ResidencyAuthority::open(ledger, &ResidencyRequest::new("probe", cap)).unwrap()
}

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 30, 1 << 20).unwrap()]).unwrap()
}

fn load(authority: &mut ResidencyAuthority, id: &ChunkId, now: u64) {
    let Acquired::Pending {
        lease,
        ticket,
        work,
        ..
    } = authority.acquire(request(id, now)).unwrap()
    else {
        panic!("absent");
    };
    assert!(matches!(work, PendingWork::Issued(WorkOrder::Read { .. })));
    authority.read_destination(ticket).unwrap().fill(0x42);
    authority.complete_read(ticket, Outcome::Completed).unwrap();
    authority.release(lease).unwrap();
}

#[test]
fn acquiring_a_resident_chunk_allocates_nothing() {
    let _probe = probe();
    let mut l = ledger();
    let mut a = open(&mut l, 16 * CHUNK);
    let id = chunk(0);
    load(&mut a, &id, 0);

    // Warm anything the first hit's machinery might size once.
    let Acquired::Ready(warm) = a.acquire(request(&id, 1)).unwrap() else {
        panic!("resident")
    };
    a.release(warm).unwrap();

    let mut leases = Vec::with_capacity(1_000);
    let before_loop = CALLS.load(SeqCst);
    for tick in 2..1_002u64 {
        let Acquired::Ready(lease) = a.acquire(request(&id, tick)).unwrap() else {
            panic!("a resident chunk is a hit")
        };
        leases.push(lease);
    }
    let acquired_calls = CALLS.load(SeqCst) - before_loop;
    assert_eq!(
        acquired_calls, 0,
        "1,000 hits allocated {acquired_calls} time(s); the hot path of a routed \
         step must not allocate"
    );
    let released = CALLS.load(SeqCst);
    for lease in leases.drain(..) {
        a.release(lease).unwrap();
    }
    assert_eq!(
        CALLS.load(SeqCst) - released,
        0,
        "giving a lease back must not allocate either"
    );
    a.close(&mut l).unwrap();
}

#[test]
fn a_long_demand_history_retains_no_heap() {
    let _probe = probe();
    let mut l = ledger();
    // Four resident at a time, ten thousand distinct chunks through the cache.
    let mut a = open(&mut l, 4 * CHUNK);
    for i in 0..64u32 {
        load(&mut a, &chunk(i), u64::from(i));
    }
    let baseline = LIVE.load(SeqCst);

    for i in 0..10_000u32 {
        let id = chunk(i % 512);
        match a.acquire(request(&id, u64::from(i) + 1_000)).unwrap() {
            Acquired::Ready(lease) => a.release(lease).unwrap(),
            Acquired::Pending { lease, ticket, .. } => {
                a.read_destination(ticket).unwrap().fill(0x11);
                a.complete_read(ticket, Outcome::Completed).unwrap();
                a.release(lease).unwrap();
            }
        }
        assert!(a.committed_bytes(Scope::Host).unwrap() <= 4 * CHUNK);
    }

    let retained = LIVE.load(SeqCst) - baseline;
    assert_eq!(
        retained, 0,
        "a cache that has seen 512 chunks and holds 4 retained {retained} byte(s)"
    );
    assert_eq!(a.outstanding().len(), 4);
    assert_eq!(a.live_lease_count(), 0);
    assert_eq!(a.in_flight_count(), 0);
    std::hint::black_box(&a);
    a.close(&mut l).unwrap();
    assert_eq!(l.scope_committed(Scope::Host), 0);
}

#[test]
fn ten_thousand_failed_reads_retain_nothing_and_charge_nothing() {
    let _probe = probe();
    let mut l = ledger();
    let mut a = open(&mut l, 4 * CHUNK);
    // Warm the maps at the size the loop will use. The baseline is taken after
    // a hundred iterations of the same cycle, so what is measured is growth
    // that tracks *history* -- a leak -- rather than a one-off capacity step
    // the first few iterations pay for and then stop paying.
    load(&mut a, &chunk(0), 0);
    let cycle = |a: &mut ResidencyAuthority, i: u32| {
        let id = chunk(1 + i % 3);
        let Acquired::Pending { lease, ticket, .. } =
            a.acquire(request(&id, u64::from(i) + 1)).unwrap()
        else {
            panic!("a failed read must leave the chunk absent")
        };
        a.complete_read(
            ticket,
            Outcome::Failed(moxie_types::Error::InvalidArtifact {
                detail: "injected".into(),
            }),
        )
        .unwrap();
        a.release(lease).unwrap();
        assert_eq!(a.committed_bytes(Scope::Host).unwrap(), CHUNK);
    };
    for i in 0..100u32 {
        cycle(&mut a, i);
    }
    let baseline = LIVE.load(SeqCst);

    for i in 100..10_000u32 {
        cycle(&mut a, i);
    }

    let retained = LIVE.load(SeqCst) - baseline;
    assert_eq!(
        retained, 0,
        "9,900 further failed reads retained {retained} byte(s)"
    );
    assert_eq!(a.stats().read_failures, 10_000);
    std::hint::black_box(&a);
    a.close(&mut l).unwrap();
}
