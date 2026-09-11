//! Isolated requested-heap accounting and post-admission fault injection.
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, KvRow, PagedSequence, PrefixLineage};
use moxie_types::{Error, HostTier, Precision, Scope, Tier};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::SeqCst};

struct Counter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static FAIL: AtomicUsize = AtomicUsize::new(0);
static FAILURES: AtomicUsize = AtomicUsize::new(0);
// SAFETY: calls are forwarded to System unchanged, except a permitted one-shot
// null return consumed by the constructor's fallible allocation path.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if FAIL.load(SeqCst) == l.size()
            && FAIL.compare_exchange(l.size(), 0, SeqCst, SeqCst).is_ok()
        {
            FAILURES.fetch_add(1, SeqCst);
            return std::ptr::null_mut();
        }
        // SAFETY: the caller supplies a valid allocator layout.
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size() as isize, SeqCst);
            CALLS.fetch_add(1, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, SeqCst);
        // SAFETY: p and l identify the live allocation returned by System.
        unsafe { System.dealloc(p, l) };
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

#[test]
fn admitted_history_has_no_token_allocations_and_complete_fault_cleanup() {
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 16 << 20, 1 << 20).unwrap()]).unwrap();
    let g = KvGeometry {
        layers: 2,
        kv_heads: 1,
        key_dim: 2,
        value_dim: 1,
        precision: Precision::Bf16,
        page_tokens: 127,
        max_tokens: 32770,
    };
    let warm = PagedSequence::with_sampling(&mut ledger, g, 7, 32769, 33377335).unwrap();
    let usage = warm.usage();
    let (history, workspace, control) = warm.sampling_bytes();
    let backing = usage.backing_bytes + history + workspace;
    let total = backing + usage.control_reserve_bytes + control;
    assert_eq!(history, 16 * (32769 + 7));
    assert_eq!(workspace, 8 * 7);
    assert_eq!(
        ledger.committed(Scope::Host, Tier::Host(HostTier::StateSpill)),
        (usage.backing_bytes + history) as u64
    );
    assert_eq!(
        ledger.committed(Scope::Host, Tier::Host(HostTier::CpuWorkspace)),
        workspace as u64
    );
    warm.close(&mut ledger).unwrap();
    for (bytes, tier) in [
        (backing, None),
        (
            (g.max_tokens + 1) * size_of::<PrefixLineage>(),
            Some(Tier::Host(HostTier::Pageable)),
        ),
    ] {
        let before = LIVE.load(SeqCst);
        let failures = FAILURES.load(SeqCst);
        FAIL.store(bytes, SeqCst);
        let error = PagedSequence::with_sampling(&mut ledger, g, 7, 32769, 33377335).unwrap_err();
        assert_eq!(
            error,
            Error::CapacityExceeded {
                tier,
                requested_bytes: bytes as u64,
                available_bytes: 0
            }
        );
        assert_eq!(FAIL.load(SeqCst), 0);
        assert_eq!(FAILURES.load(SeqCst), failures + 1);
        assert_eq!(LIVE.load(SeqCst), before);
        assert!(ledger.outstanding().is_empty());
        for t in HostTier::ALL {
            assert_eq!(ledger.committed(Scope::Host, Tier::Host(*t)), 0);
        }
    }
    let before = LIVE.load(SeqCst);
    let mut s = PagedSequence::with_sampling(&mut ledger, g, 7, 32769, 33377335).unwrap();
    assert!(LIVE.load(SeqCst) - before <= total as isize);
    let cancel = AtomicBool::new(false);
    let rows = [KvRow {
        key: &[1, 2, 3, 4],
        value: &[5, 6],
    }; 2];
    let txn = s.begin().unwrap();
    s.append_prompt(1).unwrap();
    s.append(txn, 0, &rows, &cancel).unwrap();
    s.commit_prefix(txn, 0).unwrap();
    let txn = s.begin().unwrap();
    let calls = CALLS.load(SeqCst);
    for p in 1..=32768 {
        let prepared = s.prepare_sample(txn, p, &[0.; 7], None, 1.).unwrap();
        prepared.stage(&cancel).unwrap();
        s.append(txn, p, &rows, &cancel).unwrap();
    }
    assert_eq!(
        CALLS.load(SeqCst),
        calls,
        "distribution/draw/history/KV append allocate no heap"
    );
    assert!(LIVE.load(SeqCst) - before <= total as isize);
    s.commit_prefix(txn, 32768).unwrap();
    assert_eq!(s.history(true).unwrap().len(), 32768);
    assert_eq!(
        (0..7)
            .map(|i| s.history(true).unwrap().count(i).unwrap())
            .sum::<u64>(),
        32768
    );
    let retained = LIVE.load(SeqCst);
    for _ in 0..10000 {
        let txn = s.begin().unwrap();
        s.prepare_sample(txn, 32769, &[0.; 7], None, 1.)
            .unwrap()
            .stage(&cancel)
            .unwrap();
        s.append(txn, 32769, &rows, &cancel).unwrap();
        s.abort(txn).unwrap();
    }
    assert_eq!(LIVE.load(SeqCst), retained);
    s.close(&mut ledger).unwrap();
    assert_eq!(LIVE.load(SeqCst), before);
    assert!(ledger.outstanding().is_empty());
    assert_eq!(ledger.scope_committed(Scope::Host), 0);
    eprintln!(
        "task0014 actual_history=32768 kv_rows=32769 history_bytes={history} workspace_bytes={workspace} backing_bytes={backing} admitted_bytes={total} token_allocations=0 abort_cycles=10000 retained_growth=0 close_live_delta=0 allocation_failures=2"
    );
}
