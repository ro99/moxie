//! Separate executable: count retained requested heap with no concurrent tests.
use moxie_memory::Arena;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering::SeqCst};

struct Counter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
// SAFETY: all allocation and deallocation are forwarded to System unchanged.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE.fetch_add(layout.size() as isize, SeqCst);
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

#[test]
fn variable_offsets_do_not_retain_history_for_a_bounded_live_set() {
    let mut arena = Arena::new("probe", 1_048_576, 256).unwrap();
    // Warm the free-list capacity for the fixed two-allocation live set.
    let warm = arena.allocate(1, 1, "warm").unwrap();
    arena.release(warm).unwrap();
    let baseline = LIVE.load(SeqCst);
    let mut previous_generation = 0;
    for offset in 1..=100_000 {
        let prefix = arena.allocate(offset, 1, "prefix").unwrap();
        let tail = arena.allocate(1, 1, "tail").unwrap();
        assert_eq!(tail.offset(), offset);
        assert!(tail.key().generation > previous_generation);
        previous_generation = tail.key().generation;
        arena.release(tail).unwrap();
        arena.release(prefix).unwrap();
    }
    let retained = LIVE.load(SeqCst) - baseline;
    assert_eq!(retained, 0, "historical offsets must retain no heap");
    assert_eq!(arena.occupancy().live_allocations, 0);
    assert_eq!(arena.occupancy().largest_free_bytes, arena.capacity());
    std::hint::black_box(&arena);
}
