//! Task 0028, finding 3 and its re-review: a refusal must survive a **failed
//! allocation**, not just a large one.
//!
//! The first repair routed the refusal prose through a `try_reserve` sink and
//! proved it with an allocation so large that `Vec` rejects it on layout and
//! capacity grounds, before the allocator is ever asked. That demonstrates the
//! wrong thing: `format!` aborts when the allocator **returns null**, and no
//! oversized request reaches that path.
//!
//! So this file owns a global allocator that can be made to fail exactly one
//! allocation, and arms it immediately before a refusal is composed. With the
//! fallible sink the call returns a typed error; with `format!` the process
//! receives `SIGABRT` and this whole binary dies, which is what a regression
//! for an abort has to look like.
//!
//! Isolated executable: the allocator below is global, so any other test in the
//! same process would be running under an armed failure it did not ask for.
//!
//! The trap is armed **per thread**, not per process. A global flag was the
//! first attempt and it is wrong for the same reason `budget.rs` was wrong:
//! the harness runs tests concurrently, so one test's armed failure lands in
//! another test's `format!` and aborts the binary from a place nothing was
//! testing. A thread-local flag can only fail an allocation made by the call
//! it was armed around.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use moxie_executor::{AffineLaunch, descriptor_serves, select_affine_linear_kernel};
use moxie_format::affine::{AffineDescriptor, Grouping, IntWidth};
use moxie_format::payload::ZeroPointSection;
use moxie_format::scale::ScaleDtype;
use moxie_types::{
    AccumulationPolicy, ActivationPrecision, DeviceCapability, DeviceUuid, KernelCatalogue,
    KernelId, KernelOperand, KernelShapeBounds, KernelSymbol, Precision, RoundingProfile,
    SemanticKernelDescriptor, SemanticKernelOp, SmVersion, TensorLayout, WeightPrecision,
    WorkspaceExpression,
};

thread_local! {
    /// How many more allocations on this thread succeed before one fails.
    ///
    /// `None` is disarmed. `Some(n)` lets `n` through and fails the next one,
    /// then disarms, so exactly one request returns null per arming. `const`
    /// initialised, so reading it inside the allocator cannot itself allocate.
    static ARMED: Cell<Option<usize>> = const { Cell::new(None) };
    /// Whether the last arming actually reached its allocation. A sweep that has
    /// run past the end of a function would otherwise look identical to one
    /// still finding new positions.
    static FIRED: Cell<bool> = const { Cell::new(false) };
}

/// Count down, and report whether this allocation is the one to fail.
///
/// `try_with` because a thread tearing down its locals must not panic inside
/// the allocator.
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

fn set_arming(value: Option<usize>) {
    let _ = ARMED.try_with(|armed| armed.set(value));
}

struct OneShotFailure;

// SAFETY: every method forwards the caller's unmodified pointer and layout to
// the system allocator. The only departure is returning null for one `alloc`,
// which is a value `GlobalAlloc::alloc` is explicitly allowed to return.
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

/// Run `body` with the next single allocation failing.
///
/// Nothing is asserted inside: an assertion allocates, and the point is to keep
/// the armed window as narrow as the call being tested.
fn with_one_failed_allocation<T>(body: impl FnOnce() -> T) -> T {
    with_failure_at(0, body).0
}

/// Run `body` with the `skip`-th allocation after arming failing, and report
/// whether that allocation was ever reached.
fn with_failure_at<T>(skip: usize, body: impl FnOnce() -> T) -> (T, bool) {
    let _ = FIRED.try_with(|fired| fired.set(false));
    set_arming(Some(skip));
    let out = body();
    // Disarm whether or not the allocation happened, so a path that stops
    // allocating cannot leave this thread running under a live trap.
    set_arming(None);
    let fired = FIRED.try_with(Cell::get).unwrap_or(false);
    (out, fired)
}

fn descriptor(width: IntWidth, in_features: usize, out_features: usize) -> AffineDescriptor {
    AffineDescriptor {
        width,
        out_features,
        in_features,
        grouping: Grouping::Contiguous { size: 32 },
        group_index: None,
        scale_dtype: ScaleDtype::Bf16,
    }
}

fn catalogue_entry(weight: Precision, sm: SmVersion) -> SemanticKernelDescriptor {
    SemanticKernelDescriptor {
        id: KernelId(format!(
            "{}-linear-v1-{}",
            moxie_kernels::profile_name(weight),
            sm.name()
        )),
        abi_version: moxie_kernels::AFFINE_LINEAR_ABI,
        operation: SemanticKernelOp::Linear,
        inputs: vec![
            KernelOperand::Activation(ActivationPrecision::expect(Precision::Bf16)),
            KernelOperand::Weight(WeightPrecision::expect(weight)),
        ],
        output: ActivationPrecision::expect(Precision::Bf16),
        accumulation: AccumulationPolicy::Bf16InF32Acc,
        rounding: RoundingProfile::FinalBf16Rne,
        layout: TensorLayout::ContiguousRowMajorV1,
        shape: KernelShapeBounds {
            max_rows: 64,
            max_input: 1024,
            max_output: 1024,
        },
        sm,
        workspace: WorkspaceExpression::Zero,
        image_sha256: [7u8; 32],
        symbols: vec![KernelSymbol(moxie_kernels::AFFINE_LINEAR.to_string())],
    }
}

fn capability(major: u32, minor: u32) -> DeviceCapability {
    DeviceCapability {
        ordinal: 0,
        uuid: DeviceUuid::from_bytes([0u8; 16]),
        name: "fixture".into(),
        compute_major: major,
        compute_minor: minor,
        total_memory_bytes: 1 << 30,
        multiprocessor_count: 1,
        pci_bus_id: "0000:00:00.0".into(),
        peer_access: Vec::new(),
    }
}

/// A descriptor mismatch refused while one allocation fails.
///
/// This is the exact path review aborted: `descriptor_serves` built its prose
/// with a `to_string()` evaluated **before** the fallible sink saw the
/// arguments, so the failure landed in `format!`'s infallible growth.
#[test]
fn a_descriptor_mismatch_is_refused_when_its_prose_cannot_be_allocated() {
    let d = descriptor(IntWidth::Int4, 64, 32);
    let launch = AffineLaunch::derive(&d, ZeroPointSection::PerGroup, 4).expect("a launch");
    let int4 = WeightPrecision::expect(Precision::Int4);
    // A descriptor for the other width: the mismatch that produces a wrong
    // answer rather than a failed lookup, and the first refusal in the function.
    let wrong = catalogue_entry(Precision::Int8, SmVersion::SM86);

    let result = with_one_failed_allocation(|| descriptor_serves(&wrong, int4, &launch));

    let error = result.expect_err("a width mismatch is not servable");
    assert_eq!(
        error.kind(),
        "unsupported_kernel",
        "the refusal changed kind under allocation failure: {error}"
    );
    // The prose is allowed to degrade to nothing -- that is the documented
    // trade -- but the typed refusal has to arrive.
    let _ = error.to_string();
}

/// The same, one level up: selection's "expected exactly one descriptor".
#[test]
fn a_selection_refusal_survives_a_failed_allocation() {
    let catalogue = KernelCatalogue::new(vec![catalogue_entry(Precision::Int4, SmVersion::SM86)])
        .expect("one descriptor");
    let d = descriptor(IntWidth::Int4, 64, 32);
    let launch = AffineLaunch::derive(&d, ZeroPointSection::PerGroup, 4).expect("a launch");
    let int4 = WeightPrecision::expect(Precision::Int4);
    // No descriptor for this architecture, so selection ends at the count
    // refusal, which built its detail with `format!`.
    let cap = capability(12, 0);

    let result =
        with_one_failed_allocation(|| select_affine_linear_kernel(&catalogue, &cap, int4, &launch));

    let error = result.expect_err("sm_120 has no descriptor here");
    assert_eq!(error.kind(), "unsupported_kernel", "{error}");
}

/// A control: with nothing armed, the same call produces its full prose.
///
/// Without this, a test that passed because the refusal path never ran would
/// look identical to one that passed because the path degraded correctly.
#[test]
fn the_same_refusal_carries_its_prose_when_allocation_succeeds() {
    let d = descriptor(IntWidth::Int4, 64, 32);
    let launch = AffineLaunch::derive(&d, ZeroPointSection::PerGroup, 4).expect("a launch");
    let int4 = WeightPrecision::expect(Precision::Int4);
    let wrong = catalogue_entry(Precision::Int8, SmVersion::SM86);

    let error = descriptor_serves(&wrong, int4, &launch).expect_err("a width mismatch");
    let text = error.to_string();
    assert!(
        text.contains("cannot bind"),
        "the unarmed refusal lost its prose: {text}"
    );
}

/// **Every** allocation position in a *successful* selection, swept.
///
/// The first regression only exercised a refusal, where both temporary vectors
/// stay empty and nothing is allocated before the formatter runs. Review armed
/// the trap around a selection that **succeeds** on sm_86 and got `memory
/// allocation of 32 bytes failed` and `SIGABRT`: the two `collect()` calls and
/// the winner's `clone()` were all infallible, on the one path with no refusal
/// to fall back to.
///
/// One position proves nothing here, because which position aborts depends on
/// how the function is written. So this walks every position until the arming
/// stops firing, which is where the call has stopped allocating at all.
/// Surviving the loop **is** the assertion: an abort takes the process, and no
/// assertion inside it would run.
#[test]
fn every_allocation_in_a_successful_selection_degrades_rather_than_aborting() {
    let catalogue = KernelCatalogue::new(vec![
        catalogue_entry(Precision::Int4, SmVersion::SM86),
        catalogue_entry(Precision::Int8, SmVersion::SM86),
        catalogue_entry(Precision::Int4, SmVersion::SM120),
    ])
    .expect("three descriptors");
    let d = descriptor(IntWidth::Int4, 64, 32);
    let launch = AffineLaunch::derive(&d, ZeroPointSection::PerGroup, 4).expect("a launch");
    let int4 = WeightPrecision::expect(Precision::Int4);
    let cap = capability(8, 6);

    // The control: unarmed, this selection succeeds and finds the INT4 sm_86
    // entry. Without it, a sweep over a selection that always failed would pass
    // by never reaching an allocation at all.
    let chosen = select_affine_linear_kernel(&catalogue, &cap, int4, &launch)
        .expect("an sm_86 INT4 descriptor is in the catalogue");
    assert!(chosen.id.0.contains("sm_86"), "{}", chosen.id.0);

    let mut positions_reached = 0usize;
    let mut refusals = 0usize;
    for skip in 0..64 {
        let (result, fired) = with_failure_at(skip, || {
            select_affine_linear_kernel(&catalogue, &cap, int4, &launch)
        });
        if result.is_err() {
            refusals += 1;
        }
        if !fired {
            break;
        }
        positions_reached += 1;
    }

    // At least one position must have been reachable, or this sweep measured
    // nothing -- the same way the first regression measured nothing.
    assert!(
        positions_reached > 0,
        "no allocation was reached during a successful selection, so this sweep proves nothing"
    );
    assert!(
        refusals > 0,
        "every armed position still returned Ok, so the trap never reached the call it was \
         armed around"
    );
}

/// The same sweep over the geometry derivation a caller runs first.
#[test]
fn every_allocation_in_a_launch_derivation_degrades_rather_than_aborting() {
    let d = descriptor(IntWidth::Int4, 64, 32);
    for skip in 0..64 {
        let (result, fired) = with_failure_at(skip, || {
            AffineLaunch::derive(&d, ZeroPointSection::PerGroup, 4)
        });
        // Either outcome is legal; an abort is not, and an abort ends the
        // process rather than this loop.
        let _ = result;
        if !fired {
            break;
        }
    }
}

/// What this sweep does **not** cover, stated where it is measured.
///
/// Review asked for the sweep to cover successful **admission** too.
/// `AffineLinearRun::admit`'s own allocations are fallible -- its labels go
/// through `try_label` -- but the first thing it calls is
/// `moxie_memory::PlanRequest::new`, and that aborts: armed at position 4, a
/// plan request built exactly as `resource_request` builds one dies with
/// `memory allocation of 5 bytes failed`.
///
/// That is **not** task 0028's code. `PlanRequest` and `BufferRequest` are the
/// shared admission vocabulary: the BF16 chain, the expert plans and the
/// residency authority all build them, their labels are `impl Into<String>`
/// evaluated at every call site, and their refusals use `format!`. Making that
/// path fallible is a change to a shared owner with its own consumers, and
/// smuggling it into this correction round would be the kind of scope drift
/// this repository's task contracts exist to stop.
///
/// So it is recorded rather than half-done, and the admission half of the sweep
/// is **unmeasured**, not passing. The task record names it as the next bounded
/// task.
#[test]
fn the_admission_sweep_is_unmeasured_and_this_says_so() {
    // Deliberately empty of assertions. It exists so the gap has a name in the
    // same file as the sweeps that *are* measured, rather than only in a
    // document nobody runs.
}
