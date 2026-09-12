//! Isolated executable: actual 32,768 stored rows, counted requested heap.
//! Storage/transaction evidence only; no model or attention executes here.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::SeqCst};

use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, KvRow, PagedSequence, ROOT};
use moxie_types::{Precision, Scope};

struct Counter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
// SAFETY: allocation operations are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the allocator caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size() as isize, SeqCst);
            CALLS.fetch_add(1, SeqCst);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, SeqCst);
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

#[test]
fn actual_32768_rows_have_no_append_allocations_and_repeated_abort_retains_no_history() {
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 16 << 20, 1 << 20).unwrap()]).unwrap();
    let geometry = KvGeometry::uniform(2, 1, 2, 1, Precision::Bf16, 127, 32_769);
    // Warm the ledger's persistent maps; their bookkeeping is separate from
    // this consumer's admitted envelope.
    PagedSequence::new(&mut ledger, geometry.clone())
        .unwrap()
        .close(&mut ledger)
        .unwrap();
    let before_new = LIVE.load(SeqCst);
    let mut sequence = PagedSequence::new(&mut ledger, geometry.clone()).unwrap();
    let usage = sequence.usage();
    let total = (usage.backing_bytes + usage.control_reserve_bytes) as isize;
    assert!(LIVE.load(SeqCst) - before_new <= total);
    let cancelled = AtomicBool::new(false);
    // Odd chunk and page lengths ensure many partial and later-page appends.
    for start in (0..32_768).step_by(113) {
        let end = (start + 113).min(32_768);
        let txn = sequence.begin().unwrap();
        sequence.append_prompt((end - start) as u64).unwrap();
        let calls = CALLS.load(SeqCst);
        for position in start..end {
            let key = (position as u32).to_le_bytes();
            let value = (position as u16 ^ 0x8000).to_le_bytes();
            let rows = [KvRow {
                key: &key,
                value: &value,
            }; 2];
            sequence
                .append(txn, position as u64, &rows, &cancelled)
                .unwrap();
        }
        assert_eq!(
            CALLS.load(SeqCst),
            calls,
            "appending rows must allocate no heap"
        );
        assert!(
            LIVE.load(SeqCst) - before_new <= total,
            "live transaction fits control reserve"
        );
        sequence.commit_prefix(txn, 0).unwrap();
    }
    for position in 0..32_768u64 {
        for layer in 0..2 {
            let row = sequence.row(layer, position).unwrap();
            assert_eq!(row.key, (position as u32).to_le_bytes());
            assert_eq!(row.value, (position as u16 ^ 0x8000).to_le_bytes());
        }
    }
    assert_eq!(sequence.state().frontiers(ROOT).unwrap().executed, 32_768);
    // 259 pages per layer, and the layers no longer share them: a page belongs
    // to one layer now, because layers need not agree on row width or capacity.
    assert_eq!(sequence.usage().live_pages, 2 * 259);
    assert_eq!(sequence.usage().retained_rows, 2 * 32_768);
    let retained = LIVE.load(SeqCst);
    let rows = [KvRow {
        key: &[1, 2, 3, 4],
        value: &[0, 0x80],
    }; 2];
    for _ in 0..10_000 {
        let txn = sequence.begin().unwrap();
        sequence.append(txn, 32_768, &rows, &cancelled).unwrap();
        sequence.abort(txn).unwrap();
    }
    assert_eq!(
        LIVE.load(SeqCst),
        retained,
        "aborted attempts cannot accumulate metadata"
    );
    assert!(sequence.row(0, 32_768).is_err());
    sequence.close(&mut ledger).unwrap();
    assert_eq!(ledger.scope_committed(Scope::Host), 0);
    assert_eq!(
        LIVE.load(SeqCst),
        before_new,
        "close frees payload and control storage"
    );
    eprintln!(
        "task0017 storage evidence (full retention): actual_rows=32768 max_tokens={} \
         page_tokens={} backing_bytes={} table_bytes={} control_reserve_bytes={} \
         append_allocations=0 abort_cycles=10000 retained_growth=0 close_live_delta=0",
        geometry.max_tokens,
        geometry.page_tokens,
        usage.backing_bytes,
        usage.page_table_bytes,
        usage.control_reserve_bytes
    );
}
