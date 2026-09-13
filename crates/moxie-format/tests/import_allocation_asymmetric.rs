//! Isolated executable: the **asymmetric** importer's allocation count.
//!
//! A second binary, not a second test in [`import_allocation`]'s. That file's
//! own docstring says why -- the counter is a global allocator, and two tests
//! in one executable race on it -- and adding this test beside it proved the
//! point immediately: the symmetric case's fixed cost of 3 read as 3, 13 and 25
//! as the two tests interleaved. A measurement taken while something else
//! allocates is not a measurement, which is task 0022's lesson about flaky
//! substitutions in the place it was already written down.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use moxie_format::affine::IntWidth;
use moxie_format::compressed_tensors::{
    Granularity, PackQuantizedSpec, PackedZeroPoints, SourceTensors, ZeroPointSource, import,
};
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

/// Pack zero points along the output axis, the pinned compressor's way.
fn pack_zero_points(rows: usize, groups: usize) -> Vec<u8> {
    let words = rows.div_ceil(8);
    let mut out = vec![0u8; words * groups * 4];
    for o in 0..rows {
        for g in 0..groups {
            let at = ((o / 8) * groups + g) * 4;
            let mut word = u32::from_le_bytes(out[at..at + 4].try_into().unwrap());
            // A biased-unsigned 4-bit zero point; the value does not matter.
            word |= ((o % 16) as u32) << ((o % 8) as u32 * 4);
            out[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    out
}

/// The same, for an asymmetric source.
///
/// The zero-point vector is a new collection on an old path, and the recurring
/// defect in this workspace is an import path that aborts instead of returning
/// `CapacityExceeded`. Task 0024's addition must not allocate per row either,
/// and it must add exactly one fixed allocation -- the reserved zero-point
/// vector -- over the symmetric case.
#[test]
fn an_asymmetric_import_adds_one_fixed_allocation_and_none_per_row() {
    let columns: usize = 66;
    let groups = columns.div_ceil(32);
    let spec = PackQuantizedSpec {
        width: IntWidth::Int4,
        granularity: Granularity::Group { size: 32 },
        zero_points: ZeroPointSource::PackedAlongOutput,
    };
    let per_word = spec.values_per_word();
    let packed_columns = columns.div_ceil(per_word);
    let mut counts = Vec::new();
    let mut symmetric_cost = 0usize;
    for rows in [8usize, 64, 512] {
        let packed = vec![0x11u8; rows * packed_columns * 4];
        let scale: Vec<u8> = (0..rows * groups)
            .flat_map(|_| ((1.0f32.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let zp = pack_zero_points(rows, groups);
        let packed_shape = [rows as u64, packed_columns as u64];
        let scale_shape = [rows as u64, groups as u64];
        let zp_shape = [rows.div_ceil(per_word) as u64, groups as u64];
        let symmetric = SourceTensors {
            packed: &packed,
            packed_shape: &packed_shape,
            scale: &scale,
            scale_shape: &scale_shape,
            scale_dtype: ScaleDtype::Bf16,
            zero_point: None,
            logical: (rows, columns),
        };
        let asymmetric = SourceTensors {
            zero_point: Some(PackedZeroPoints {
                payload: &zp,
                shape: &zp_shape,
            }),
            ..symmetric
        };
        let sym_spec = PackQuantizedSpec {
            zero_points: ZeroPointSource::Symmetric,
            ..spec
        };
        let before = CALLS.load(SeqCst);
        let baseline = import(&sym_spec, symmetric).unwrap();
        let sym_used = CALLS.load(SeqCst) - before;

        let before = CALLS.load(SeqCst);
        let tensor = import(&spec, asymmetric).unwrap();
        let used = CALLS.load(SeqCst) - before;
        assert_eq!(tensor.descriptor().out_features, rows);
        assert_ne!(
            tensor.zero_points(),
            baseline.zero_points(),
            "the asymmetric import must actually carry zero points"
        );
        if symmetric_cost == 0 {
            symmetric_cost = sym_used;
        }
        assert_eq!(
            sym_used, symmetric_cost,
            "symmetric cost moved at {rows} rows"
        );
        counts.push((rows, used));
    }
    eprintln!("task0024 asymmetric import allocations: {counts:?} (symmetric {symmetric_cost})");
    let first = counts[0].1;
    for (rows, used) in &counts {
        assert_eq!(
            *used, first,
            "importing {rows} rows allocated {used} times against {first} for {} rows",
            counts[0].0
        );
    }
    assert_eq!(
        first,
        symmetric_cost + 1,
        "the zero-point vector is one reservation; {first} against {symmetric_cost} symmetric"
    );
}
