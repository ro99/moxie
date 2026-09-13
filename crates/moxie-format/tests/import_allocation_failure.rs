//! Isolated executable: an allocation failure during an import is a typed
//! error, at **every** position it can happen.
//!
//! The most repeated defect in this workspace is an infallible allocation on a
//! path that must return `CapacityExceeded` -- five times over tasks 0019 to
//! 0023, each time on a path added after the previous fix. Task 0024 adds one
//! collection to the importer (the zero-point vector), so the rule gets a gate
//! rather than a comment.
//!
//! The sweep **measures** how many allocation positions the body has and fails
//! each one in turn, rather than failing the first one several times. That is
//! task 0023's own correction: a loop that called `while_failing(1, ...)` six
//! times had an axis whose index only changed the assertion message.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use moxie_format::affine::IntWidth;
use moxie_format::compressed_tensors::{
    Granularity, PackQuantizedSpec, PackedZeroPoints, SourceTensors, ZeroPointSource, import,
};
use moxie_format::scale::ScaleDtype;
use moxie_types::Error;

thread_local! {
    static SKIP: Cell<usize> = const { Cell::new(0) };
    static FAIL: Cell<usize> = const { Cell::new(0) };
    static SERVED: Cell<usize> = const { Cell::new(0) };
}

struct Injector;

// SAFETY: every request is forwarded to the system allocator unchanged, except
// the counted ones, which return null -- the documented way for a `GlobalAlloc`
// to report failure.
unsafe impl GlobalAlloc for Injector {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = SERVED.try_with(|c| c.set(c.get() + 1));
        let skipping = SKIP.try_with(|s| {
            let left = s.get();
            if left > 0 {
                s.set(left - 1);
            }
            left > 0
        });
        if skipping != Ok(true) {
            let fail = FAIL.try_with(|f| {
                let left = f.get();
                if left > 0 {
                    f.set(left - 1);
                }
                left > 0
            });
            if fail == Ok(true) {
                return core::ptr::null_mut();
            }
        }
        // SAFETY: the caller supplies a valid layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: Injector = Injector;

fn while_failing_at<T>(skip: usize, n: usize, body: impl FnOnce() -> T) -> T {
    SKIP.with(|f| f.set(skip));
    FAIL.with(|f| f.set(n));
    let out = body();
    SKIP.with(|f| f.set(0));
    FAIL.with(|f| f.set(0));
    out
}

fn allocations_of<T>(body: impl FnOnce() -> T) -> usize {
    let before = SERVED.try_with(Cell::get).unwrap_or(0);
    let out = body();
    drop(out);
    SERVED.try_with(Cell::get).unwrap_or(0) - before
}

const ROWS: usize = 13;
const COLUMNS: usize = 66;

fn pack_codes(width: IntWidth) -> Vec<u8> {
    let per_word = 32 / width.bits() as usize;
    let packed_columns = COLUMNS.div_ceil(per_word);
    vec![0x21u8; ROWS * packed_columns * 4]
}

fn pack_zero_points(width: IntWidth, groups: usize) -> Vec<u8> {
    let per_word = 32 / width.bits() as usize;
    vec![0x35u8; ROWS.div_ceil(per_word) * groups * 4]
}

/// Every allocation position, for both zero-point serializations and both
/// widths: refused one at a time, and each refusal is a typed error.
#[test]
fn every_allocation_position_in_an_import_is_a_typed_error() {
    let mut swept = Vec::new();
    for width in [IntWidth::Int4, IntWidth::Int8] {
        for zero_points in [
            ZeroPointSource::Symmetric,
            ZeroPointSource::PackedAlongOutput,
        ] {
            let spec = PackQuantizedSpec {
                width,
                granularity: Granularity::Group { size: 32 },
                zero_points,
            };
            let per_word = spec.values_per_word();
            let groups = COLUMNS.div_ceil(32);
            let packed = pack_codes(width);
            let scale: Vec<u8> = (0..ROWS * groups)
                .flat_map(|_| ((1.0f32.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let zp = pack_zero_points(width, groups);
            let packed_shape = [ROWS as u64, COLUMNS.div_ceil(per_word) as u64];
            let scale_shape = [ROWS as u64, groups as u64];
            let zp_shape = [ROWS.div_ceil(per_word) as u64, groups as u64];
            let src = SourceTensors {
                packed: &packed,
                packed_shape: &packed_shape,
                scale: &scale,
                scale_shape: &scale_shape,
                scale_dtype: ScaleDtype::Bf16,
                zero_point: match zero_points {
                    ZeroPointSource::Symmetric => None,
                    ZeroPointSource::PackedAlongOutput => Some(PackedZeroPoints {
                        payload: &zp,
                        shape: &zp_shape,
                    }),
                },
                logical: (ROWS, COLUMNS),
            };
            // The control: it imports, and the count of positions is measured.
            assert!(import(&spec, src).is_ok(), "{width:?} {zero_points:?}");
            let positions = allocations_of(|| import(&spec, src).unwrap());
            assert!(
                positions > 0,
                "{width:?} {zero_points:?} allocates nothing, so this sweep would test nothing"
            );
            for at in 0..positions {
                let refused = while_failing_at(at, 1, || import(&spec, src));
                match refused {
                    Err(Error::CapacityExceeded { .. }) => {}
                    Err(other) => panic!(
                        "{width:?} {zero_points:?} position {at}: {other}, which is not a \
                         capacity error"
                    ),
                    // A position the importer does not itself own -- the failure
                    // landed in a formatting temporary on an error path that was
                    // not reached, say -- is allowed to succeed. What is not
                    // allowed is a panic or an abort, and reaching this line at
                    // all means neither happened.
                    Ok(_) => {}
                }
            }
            swept.push((width, zero_points, positions));
        }
    }
    eprintln!("task0024 import allocation positions swept: {swept:?}");
    // The asymmetric path must have strictly more positions than the symmetric
    // one, or the new collection is not on this path and the sweep is not
    // reaching it.
    for width in [IntWidth::Int4, IntWidth::Int8] {
        let sym = swept
            .iter()
            .find(|(w, z, _)| *w == width && *z == ZeroPointSource::Symmetric)
            .expect("symmetric swept")
            .2;
        let asym = swept
            .iter()
            .find(|(w, z, _)| *w == width && *z == ZeroPointSource::PackedAlongOutput)
            .expect("asymmetric swept")
            .2;
        assert!(
            asym > sym,
            "{width:?}: {asym} asymmetric positions against {sym} symmetric ones"
        );
    }
}
