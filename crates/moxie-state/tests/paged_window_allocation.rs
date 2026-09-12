//! Isolated executable: 32,768 actual stored rows with one windowed layer.
//!
//! Separate from `paged_allocation` on purpose. The counter is a **global**
//! allocator, so two tests in one executable race on it and neither result
//! means anything; task 0013 gave each counted measurement its own binary for
//! exactly that reason.
//!
//! Storage/transaction evidence only; no model or attention executes here.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::SeqCst};

use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_state::{KvGeometry, KvRow, PagedSequence, ROOT, Retention};
use moxie_types::{Error, Precision, Scope};

struct Counter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's
// unmodified pointer and layout; the counters are relaxed atomics and do not
// affect the allocation itself.
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

/// The same 32,768 actual stored rows, with one layer windowed.
///
/// **Storage capacity evidence only.** No attention runs here, nothing is a
/// context claim and nothing is model support. What it shows is that a windowed
/// layer's admitted envelope follows its window rather than the context, and
/// that reclamation costs no allocation per row.
#[test]
fn a_windowed_layer_is_admitted_for_its_window_not_for_the_context() {
    let mut ledger =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 16 << 20, 1 << 20).unwrap()]).unwrap();
    let window = 1024;
    let tentative = 113;
    let mut geometry = KvGeometry::uniform(2, 1, 2, 1, Precision::Bf16, 127, 32_769);
    geometry.tentative_rows = tentative;
    geometry.layers[1].retention = Retention::Window { window };
    // Warm both geometries through the ledger before measuring: a first
    // construction grows the ledger's own persistent bookkeeping, which is not
    // part of this consumer's admitted envelope.
    let full = KvGeometry::uniform(2, 1, 2, 1, Precision::Bf16, 127, 32_769);
    let full_backing = {
        let s = PagedSequence::new(&mut ledger, full).unwrap();
        let b = s.usage().backing_bytes;
        s.close(&mut ledger).unwrap();
        b
    };
    PagedSequence::new(&mut ledger, geometry.clone())
        .unwrap()
        .close(&mut ledger)
        .unwrap();
    let before_new = LIVE.load(SeqCst);
    let mut sequence = PagedSequence::new(&mut ledger, geometry.clone()).unwrap();
    let usage = sequence.usage();
    assert!(
        usage.backing_bytes < full_backing,
        "windowing must shrink the admitted envelope"
    );
    let cancelled = AtomicBool::new(false);
    for start in (0..32_768).step_by(tentative) {
        let end = (start + tentative).min(32_768);
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
            "reclaiming appends must allocate no heap either"
        );
        sequence.commit_prefix(txn, 0).unwrap();
    }
    assert_eq!(sequence.state().frontiers(ROOT).unwrap().executed, 32_768);
    // Layer 0 keeps everything; layer 1 keeps exactly its window.
    assert_eq!(sequence.retained_range(0).unwrap(), 0..32_768);
    assert_eq!(
        sequence.retained_range(1).unwrap(),
        (32_768 - window) as u64..32_768
    );
    assert_eq!(usage.retained_rows, 0);
    assert_eq!(sequence.usage().retained_rows, 32_768 + window);
    for position in 0..32_768u64 {
        let row = sequence.row(0, position).unwrap();
        assert_eq!(row.key, (position as u32).to_le_bytes());
        let windowed = sequence.row(1, position);
        if position < (32_768 - window) as u64 {
            assert!(matches!(
                windowed,
                Err(Error::Reclaimed {
                    layer: 1,
                    retained_from,
                    ..
                }) if retained_from == (32_768 - window) as u64
            ));
        } else {
            assert_eq!(windowed.unwrap().key, (position as u32).to_le_bytes());
        }
    }
    sequence.close(&mut ledger).unwrap();
    assert_eq!(ledger.scope_committed(Scope::Host), 0);
    // Before the report, not after: with the harness capturing output,
    // `eprintln!` retains a buffer that this counter would see as a leak.
    assert_eq!(
        LIVE.load(SeqCst),
        before_new,
        "close frees payload and control storage"
    );
    eprintln!(
        "task0017 storage evidence (windowed): actual_rows=32768 max_tokens={} window={window} \
         tentative_rows={tentative} page_tokens={} backing_bytes={} full_retention_backing={} \
         table_bytes={} control_reserve_bytes={} append_allocations=0",
        geometry.max_tokens,
        geometry.page_tokens,
        usage.backing_bytes,
        full_backing,
        usage.page_table_bytes,
        usage.control_reserve_bytes
    );
}
