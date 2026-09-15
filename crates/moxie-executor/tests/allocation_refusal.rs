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
    /// Armed by [`with_one_failed_allocation`] and cleared by the allocation it
    /// fails, so exactly one request on this thread returns null and everything
    /// after it succeeds. `const` initialised, so reading it inside the
    /// allocator cannot itself allocate.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Take the arming, if this thread has any. `try_with` because a thread tearing
/// down its locals must not panic inside the allocator.
fn take_arming() -> bool {
    ARMED
        .try_with(|armed| armed.replace(false))
        .unwrap_or(false)
}

fn set_arming(value: bool) {
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
    set_arming(true);
    let out = body();
    // Disarm whether or not the allocation happened, so a path that stops
    // allocating cannot leave this thread running under a live trap.
    set_arming(false);
    out
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
