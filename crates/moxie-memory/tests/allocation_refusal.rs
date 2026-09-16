//! Task 0029: admission refuses a failed allocation instead of aborting.
//!
//! An isolated executable with a per-thread one-shot failing allocator. A
//! process-wide flag is wrong here for the reason `budget.rs` and task 0028's
//! first attempt both found: the harness runs tests concurrently, so a global
//! trap lands in an unrelated test's `format!` and kills the binary from a
//! place nothing was testing.
//!
//! Three properties, and the weakest of them on its own is what task 0028's
//! review rejected:
//!
//! 1. every position that fires returns a typed refusal, never a value built
//!    from a reservation that failed;
//! 2. the first position that does not fire returns a value **equal** to the
//!    unarmed one;
//! 3. exhausting the loop bound is a failure, because a sweep that never
//!    reached the end never established (2).
//!
//! And two properties no top-level sweep can see, which is why they are swept
//! here rather than through `admit`: a refusal must leave the **ledger's
//! counters** where they were, and it must leave the **arena's free list**
//! where it was.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use moxie_memory::arena::Arena;
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
use moxie_types::{HostTier, Scope, Tier};

thread_local! {
    /// How many more allocations on this thread succeed before one fails.
    /// `None` is disarmed; `Some(n)` lets `n` through and fails the next.
    static ARMED: Cell<Option<usize>> = const { Cell::new(None) };
    /// Whether the last arming reached its allocation. A sweep past the end of
    /// a call would otherwise look identical to one still finding positions.
    static FIRED: Cell<bool> = const { Cell::new(false) };
}

fn take_arming() -> bool {
    ARMED
        .try_with(|armed| match armed.get() {
            None => false,
            Some(0) => {
                armed.set(None);
                let _ = FIRED.try_with(|fired| fired.set(true));
                true
            }
            Some(n) => {
                armed.set(Some(n - 1));
                false
            }
        })
        .unwrap_or(false)
}

struct OneShotFailure;

// SAFETY: every method forwards the caller's unmodified pointer and layout to
// the system allocator. The only departure is returning null for one `alloc`,
// which `GlobalAlloc::alloc` is explicitly allowed to return.
unsafe impl GlobalAlloc for OneShotFailure {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if take_arming() {
            return core::ptr::null_mut();
        }
        // SAFETY: forwards the caller's unmodified layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if take_arming() {
            return core::ptr::null_mut();
        }
        // SAFETY: forwards the caller's own pointer, layout and size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static A: OneShotFailure = OneShotFailure;

/// Run `body` with the `skip`-th allocation after arming failing, and report
/// whether that allocation was reached.
fn with_failure_at<T>(skip: usize, body: impl FnOnce() -> T) -> (T, bool) {
    /// Disarms on the way out **however** the scope ends.
    ///
    /// A plain assignment after the call leaks the trap when `body` panics --
    /// and a panic is exactly what a bad mutation produces -- so the next test
    /// on this thread runs under a live trap it never asked for. Independent
    /// review found the harness without this.
    struct Disarm;
    impl Drop for Disarm {
        fn drop(&mut self) {
            let _ = ARMED.try_with(|armed| armed.set(None));
        }
    }

    let _ = FIRED.try_with(|fired| fired.set(false));
    let guard = Disarm;
    let _ = ARMED.try_with(|armed| armed.set(Some(skip)));
    let out = body();
    drop(guard);
    let fired = FIRED.try_with(Cell::get).unwrap_or(false);
    (out, fired)
}

const LIMIT: usize = 256;

fn plan() -> moxie_types::Result<PlanRequest> {
    let mut request = PlanRequest::new("sweep", ["bind", "launch", "read"])?;
    request.buffer(BufferRequest::new(
        "activations",
        Scope::Host,
        Tier::Host(HostTier::CpuWorkspace),
        4096,
        StageSpan::inclusive(0, 2),
    ))?;
    request.buffer(BufferRequest::new(
        "output",
        Scope::Host,
        Tier::Host(HostTier::CpuWorkspace),
        4096,
        StageSpan::inclusive(1, 2),
    ))?;
    // A second scope, so a refusal partway through the per-scope preparation is
    // reachable rather than theoretical.
    request.buffer(BufferRequest::new(
        "device-activations",
        Scope::Device(moxie_types::DeviceUuid::from_bytes([9u8; 16])),
        Tier::Device(moxie_types::DeviceTier::Activations),
        4096,
        StageSpan::inclusive(0, 2),
    ))?;
    Ok(request)
}

/// Building a request refuses rather than aborting, at every position.
#[test]
fn every_allocation_in_a_plan_request_degrades_rather_than_aborting() {
    let expected = plan().expect("the unarmed request builds");

    let mut fired_positions = 0usize;
    let mut ran_past_the_end = false;
    for skip in 0..LIMIT {
        let (result, fired) = with_failure_at(skip, plan);
        if fired {
            fired_positions += 1;
            match result {
                Err(error) => assert_eq!(
                    error.kind(),
                    "capacity_exceeded",
                    "position {skip} failed an allocation and refused with the wrong kind: {error}"
                ),
                Ok(request) => panic!(
                    "position {skip} failed an allocation and still returned a request labelled {:?}",
                    request.label()
                ),
            }
            continue;
        }
        let request = result
            .unwrap_or_else(|e| panic!("position {skip} failed nothing and still refused: {e}"));
        assert_eq!(
            request, expected,
            "position {skip} failed nothing and returned a request unequal to the unarmed one"
        );
        ran_past_the_end = true;
        break;
    }

    assert!(
        ran_past_the_end,
        "the sweep exhausted its {LIMIT}-position bound without running past the end of the call"
    );
    assert!(
        fired_positions > 0,
        "no allocation was reached while building a request, so this sweep proves nothing"
    );
}

/// **Two scopes**, because that is what the affine request touches and because
/// a single-scope ledger cannot expose a partial multi-scope preparation:
/// `Ledger::admit` installs missing tier entries across every scope it charges.
fn device_scope() -> Scope {
    Scope::Device(moxie_types::DeviceUuid::from_bytes([9u8; 16]))
}

fn ledger() -> Ledger {
    Ledger::new([
        CapacitySnapshot::new(Scope::Host, 1 << 20, 1024).expect("a snapshot"),
        CapacitySnapshot::new(device_scope(), 1 << 20, 1024).expect("a snapshot"),
    ])
    .expect("a ledger")
}

/// Every tier of every scope, plus the scope totals.
///
/// An earlier version of this checked one host tier and the outstanding count,
/// which is not "every counter" -- independent review said so. `Tier::valid_in`
/// enumerates the tiers a scope can hold, and `scope_committed` is the total
/// the per-tier figures do not add up to.
fn counters(ledger: &Ledger) -> Vec<(Scope, Option<Tier>, u64)> {
    let mut out = Vec::new();
    for scope in [Scope::Host, device_scope()] {
        out.push((scope, None, ledger.scope_committed(scope)));
        for tier in Tier::valid_in(scope.kind()) {
            out.push((scope, Some(tier), ledger.committed(scope, tier)));
        }
    }
    out
}

/// **Admission's counters.** A refusal must leave every tier and scope counter
/// exactly where it was.
///
/// `Ledger::admit` used to add every committed counter and *then* allocate the
/// record naming them, so a failure between the two left capacity charged
/// against a reservation nobody could see and no release could undo.
/// "Nothing outstanding" passes in that state, which is why this compares the
/// counters and not the outstanding list.
#[test]
fn a_refused_admission_leaves_every_counter_where_it_was() {
    let request = plan().expect("a request");

    let mut fired_positions = 0usize;
    let mut ran_past_the_end = false;
    for skip in 0..LIMIT {
        let mut ledger = ledger();
        let before = counters(&ledger);
        let before_count = ledger.outstanding_count();

        let (result, fired) = with_failure_at(skip, || ledger.admit(&request));

        if fired {
            fired_positions += 1;
            assert!(
                result.is_err(),
                "position {skip} failed an allocation and still admitted"
            );
            assert_eq!(
                counters(&ledger),
                before,
                "position {skip} refused and left a counter changed"
            );
            assert_eq!(
                ledger.outstanding_count(),
                before_count,
                "position {skip} refused and left a reservation behind"
            );
            continue;
        }

        let reservation = result
            .unwrap_or_else(|e| panic!("position {skip} failed nothing and still refused: {e:?}"));
        assert_ne!(
            counters(&ledger),
            before,
            "an admitted reservation charged nothing"
        );
        ledger.release(reservation).expect("release");
        ran_past_the_end = true;
        break;
    }

    assert!(
        ran_past_the_end,
        "the sweep exhausted its {LIMIT}-position bound without admitting once"
    );
    assert!(
        fired_positions > 0,
        "no allocation was reached during admission, so this sweep proves nothing"
    );
}

/// **The arena's free list.** A refusal must leave it byte-for-byte as it was.
///
/// `Arena::allocate` mutated the free list and *then* allocated the record
/// describing the range, so a failure between the two left the arena believing
/// bytes were handed out that nothing owned. `AffineLinearRun::admit` owns its
/// arena and drops it on refusal, so no sweep through `admit` can tell a
/// rolled-back free list from a discarded one: this keeps the arena.
#[test]
fn a_refused_arena_allocation_leaves_the_free_list_where_it_was() {
    let mut fired_positions = 0usize;
    let mut ran_past_the_end = false;
    for skip in 0..LIMIT {
        let mut arena = Arena::new("sweep", 1 << 16, 256).expect("an arena");
        // **The first allocation is the swept one.** A later one finds spare
        // capacity in the record map and allocates nothing, so arming it
        // measures an empty window -- the sweep's own `fired_positions` check
        // caught that.
        // **The exact state**, not the aggregate. `ArenaOccupancy` is byte and
        // range *counts*; it cannot see free-range offsets or order, live
        // records, owners or generations. `Arena` derives `Debug`, so its own
        // rendering is the whole structure -- and building the string here is
        // outside the armed window, so it measures the arena rather than this
        // file.
        let before = format!("{arena:?}");

        // **An owned label.** `AffineLinearRun::admit` builds its labels with
        // its own fallible formatter, so what reaches an arena is
        // `Cow::Owned` -- and cloning one copies bytes, where cloning a literal
        // copies a pointer. A sweep that only passes `"first"` exercises the
        // free half and misses the allocation independent review found.
        // Built **outside** the armed window: the copy the test makes is the
        // test's own allocation, and failing it would measure this file rather
        // than the arena.
        let owner = moxie_memory::request::Label::Owned(String::from("owned-label-for-the-sweep"));
        let arena_ref = &mut arena;
        let (result, fired) = with_failure_at(skip, move || arena_ref.allocate(4096, 256, owner));

        if fired {
            fired_positions += 1;
            assert!(
                result.is_err(),
                "position {skip} failed an allocation and still handed out a range"
            );
            assert_eq!(
                format!("{arena:?}"),
                before,
                "position {skip} refused and left the arena changed"
            );
            continue;
        }

        let first = result.unwrap_or_else(|e| {
            panic!(
                "position {skip} failed nothing and still refused: {}",
                e.error
            )
        });
        assert_ne!(
            format!("{arena:?}"),
            before,
            "an accepted allocation changed nothing"
        );
        arena.release(first).expect("release");
        ran_past_the_end = true;
        break;
    }

    assert!(
        ran_past_the_end,
        "the sweep exhausted its {LIMIT}-position bound without allocating once"
    );
    assert!(
        fired_positions > 0,
        "no allocation was reached inside the arena, so this sweep proves nothing"
    );
}
