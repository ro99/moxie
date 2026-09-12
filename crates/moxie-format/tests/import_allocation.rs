//! Isolated executable: the importer's allocation count.
//!
//! Separate binary because the counter is a **global** allocator: two tests in
//! one executable race on it and neither result means anything.
//!
//! Independent review reproduced a `SIGABRT` on task 0018's first version by
//! injecting failure into the seventeen-byte row allocation that
//! `affine::pack_row` made per row through the infallible allocator. The fix
//! packs into a destination reserved once, so the allocation it aborted on no
//! longer exists. This counts them to prove it, rather than asserting it in a
//! comment.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use moxie_format::affine::IntWidth;
use moxie_format::compressed_tensors::{Granularity, PackQuantizedSpec, TensorTriple, import};
use moxie_format::scale::ScaleDtype;

struct Counter;
static CALLS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's
// unmodified pointer and layout; the counter is a relaxed atomic and does not
// affect the allocation itself.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller's unmodified layout to the system
        // allocator; the counter is a relaxed atomic and changes nothing.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            CALLS.fetch_add(1, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(p, layout) };
    }
}

#[global_allocator]
static A: Counter = Counter;

#[test]
fn importing_allocates_a_bounded_number_of_times_regardless_of_row_count() {
    // The row count is what used to drive the per-row allocation, so the test
    // is that allocations do not grow with it.
    let mut counts = Vec::new();
    for rows in [4usize, 64, 512] {
        let columns: usize = 66; // a partial group and a partial packed word
        let spec = PackQuantizedSpec {
            width: IntWidth::Int4,
            granularity: Granularity::Group { size: 32 },
            symmetric: true,
        };
        let per_word = spec.values_per_word();
        let packed_columns = columns.div_ceil(per_word);
        let packed = vec![0x11u8; rows * packed_columns * 4];
        let groups = columns.div_ceil(32);
        let scale: Vec<u8> = (0..rows * groups)
            .flat_map(|_| ((1.0f32.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let packed_shape = [rows as u64, packed_columns as u64];
        let scale_shape = [rows as u64, groups as u64];
        let triple = TensorTriple {
            packed: &packed,
            packed_shape: &packed_shape,
            scale: &scale,
            scale_shape: &scale_shape,
            scale_dtype: ScaleDtype::Bf16,
            logical: (rows, columns),
        };
        let before = CALLS.load(SeqCst);
        let tensor = import(&spec, triple).unwrap();
        let used = CALLS.load(SeqCst) - before;
        assert_eq!(tensor.descriptor().out_features, rows);
        counts.push((rows, used));
    }
    eprintln!("task0018 import allocations: {counts:?}");
    // Same count at every row count: nothing is allocated per row.
    let first = counts[0].1;
    for (rows, used) in &counts {
        assert_eq!(
            *used, first,
            "importing {rows} rows allocated {used} times against {first} for \
             {} rows: the importer must not allocate per row",
            counts[0].0
        );
    }
    // And the fixed cost is small: the code destination, the column scratch
    // and the scale vector. A double-digit count would mean something else
    // crept onto the path.
    assert!(
        first <= 8,
        "{first} allocations is more than the fixed cost"
    );
}
