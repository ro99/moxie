//! Isolated executable: inject failure into the lineage allocation without
//! affecting other tests or asking the operating system for excessive memory.
use std::alloc::{GlobalAlloc, Layout, System};
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};

use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, PagedSequence, PrefixLineage};
use moxie_types::{Error, HostTier, Precision, Scope, Tier};

struct FailingAllocator;
static ARMED: AtomicBool = AtomicBool::new(false);
static FAIL_BYTES: AtomicUsize = AtomicUsize::new(0);
static FAILURES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: successful operations are forwarded unchanged to the system
// allocator. The single injected null result is permitted by `GlobalAlloc` and
// is consumed by `Vec::try_reserve_exact` as an allocation error.
unsafe impl GlobalAlloc for FailingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(SeqCst)
            && layout.size() == FAIL_BYTES.load(SeqCst)
            && ARMED.compare_exchange(true, false, SeqCst, SeqCst).is_ok()
        {
            FAILURES.fetch_add(1, SeqCst);
            return std::ptr::null_mut();
        }
        // SAFETY: the allocator caller supplies a valid layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe an allocation returned above.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: FailingAllocator = FailingAllocator;

#[test]
fn lineage_allocation_failure_is_capacity_exhaustion_and_releases_the_reservation() {
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).unwrap()]).unwrap();
    let geometry = KvGeometry::uniform(1, 1, 2, 2, Precision::Bf16, 16, 1_000);
    let lineage_bytes = (geometry.max_tokens + 1) * size_of::<PrefixLineage>();
    assert_eq!(
        lineage_bytes, 8_008,
        "the injected allocation stays precise"
    );

    FAIL_BYTES.store(lineage_bytes, SeqCst);
    ARMED.store(true, SeqCst);
    let error = PagedSequence::new(&mut ledger, geometry).unwrap_err();

    assert_eq!(FAILURES.load(SeqCst), 1, "the intended allocation failed");
    assert!(!ARMED.load(SeqCst));
    assert_eq!(error.kind(), "capacity_exceeded");
    assert_eq!(
        error,
        Error::CapacityExceeded {
            tier: Some(Tier::Host(HostTier::Pageable)),
            requested_bytes: lineage_bytes as u64,
            available_bytes: 0,
        }
    );
    assert!(ledger.outstanding().is_empty());
    assert_eq!(ledger.scope_committed(Scope::Host), 0);
    assert_eq!(
        ledger.committed(Scope::Host, Tier::Host(HostTier::StateSpill)),
        0
    );
    assert_eq!(
        ledger.committed(Scope::Host, Tier::Host(HostTier::Pageable)),
        0
    );
}
