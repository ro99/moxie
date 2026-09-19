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
        // The real limits on every NVIDIA architecture to date.
        max_grid: (2_147_483_647, 65_535, 65_535),
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
/// Three things are required of each position, and the first version required
/// only the weakest of them. Review mutated `try_string` to return `Ok("")` on
/// a failed reservation -- which yields a **corrupted descriptor**, not a
/// refusal -- and the sweep still passed, because it counted refusals globally
/// and two positions elsewhere supplied them. So:
///
/// 1. every position that actually fires must return the typed refusal, not a
///    value built from a reservation that failed;
/// 2. the first position that does **not** fire -- the sweep has run past the
///    end of the call -- must return a descriptor equal to the catalogue's,
///    which is what catches a silently truncated one;
/// 3. exhausting the loop bound is a failure, because a sweep that never
///    reached the end never established (2).
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
    let expected = select_affine_linear_kernel(&catalogue, &cap, int4, &launch)
        .expect("an sm_86 INT4 descriptor is in the catalogue");
    assert!(expected.id.0.contains("sm_86"), "{}", expected.id.0);

    const LIMIT: usize = 64;
    let mut fired_positions = 0usize;
    let mut ran_past_the_end = false;
    for skip in 0..LIMIT {
        let (result, fired) = with_failure_at(skip, || {
            select_affine_linear_kernel(&catalogue, &cap, int4, &launch)
        });
        if fired {
            fired_positions += 1;
            // (1) A failed reservation is a refusal. A descriptor returned here
            // was built from an allocation that did not happen.
            match result {
                Err(error) => assert_eq!(
                    error.kind(),
                    "capacity_exceeded",
                    "position {skip} failed an allocation and refused with the wrong kind: {error}"
                ),
                Ok(chosen) => panic!(
                    "position {skip} failed an allocation and still returned a descriptor: {:?}",
                    chosen.id
                ),
            }
            continue;
        }
        // (2) Past the end of the call: nothing was failed, so this must be the
        // whole descriptor, field for field.
        let chosen = result
            .unwrap_or_else(|e| panic!("position {skip} failed nothing and still refused: {e}"));
        assert_eq!(
            chosen, expected,
            "position {skip} failed nothing and returned a descriptor unequal to the catalogue's"
        );
        ran_past_the_end = true;
        break;
    }

    // (3) The sweep has to have reached the end, or (2) was never established.
    assert!(
        ran_past_the_end,
        "the sweep exhausted its {LIMIT}-position bound without running past the end of the \
         call, so it never checked a complete result"
    );
    assert!(
        fired_positions > 0,
        "no allocation was reached during a successful selection, so this sweep proves nothing"
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

/// Task 0037: a paged attention launch refuses under a failed allocation.
///
/// The same requirement, for the newer binding. Its checks compose prose for
/// every refusal — head ratios, scales, positions, page alignment, ABI widths —
/// and every one of them runs on the path a caller takes when it is short of
/// memory. Sweeping the positions is what proves the sink is fallible
/// everywhere rather than at the first refusal somebody happened to test.
#[test]
fn a_paged_attention_launch_refuses_when_its_prose_cannot_be_allocated() {
    use moxie_executor::{AttentionLayer, PageGeometry, PagedAttentionLaunch};
    use moxie_plan::Visibility;

    let layer = AttentionLayer {
        geometry: PageGeometry {
            kv_heads: 2,
            head_dim: 64,
            page_tokens: 16,
            pages: 4,
        },
        heads: 8,
        scale: 1.0,
        visibility: Visibility::Causal,
    };
    // Four refusals, each from a different check, each composing its own prose.
    /// One named way to build a launch that must be refused.
    type Refusal = Box<dyn Fn() -> moxie_types::Result<PagedAttentionLaunch>>;
    let refusals: Vec<(&str, Refusal)> = vec![
        (
            "an indivisible head ratio",
            Box::new(move || {
                let mut bad = layer;
                bad.heads = 3;
                PagedAttentionLaunch::new(bad, 1, 0, 0, 1)
            }),
        ),
        (
            "a nonpositive declared scale",
            Box::new(move || {
                let mut bad = layer;
                bad.scale = 0.0;
                PagedAttentionLaunch::new(bad, 1, 0, 0, 1)
            }),
        ),
        (
            "a query past the frontier",
            Box::new(move || PagedAttentionLaunch::new(layer, 1, 64, 0, 64)),
        ),
        (
            "a history base inside a page",
            Box::new(move || PagedAttentionLaunch::new(layer, 1, 40, 8, 40)),
        ),
        (
            "a window wider than the ABI",
            Box::new(move || {
                let mut bad = layer;
                bad.visibility = Visibility::SlidingWindow {
                    window: u64::from(u32::MAX) + 1,
                };
                PagedAttentionLaunch::new(bad, 1, 0, 0, 1)
            }),
        ),
    ];
    for (what, refuse) in refusals {
        // Position zero is the first allocation the refusal makes; the sweep
        // walks forward until nothing allocates any more. Any position that
        // aborts takes the whole binary with it, which is the regression.
        for skip in 0..8 {
            let (result, fired) = with_failure_at(skip, &refuse);
            assert!(result.is_err(), "{what} was accepted at position {skip}");
            if !fired {
                break;
            }
        }
    }
}

/// Task 0037: selecting a paged attention kernel refuses the same way.
#[test]
fn paged_attention_selection_refuses_when_its_prose_cannot_be_allocated() {
    use moxie_executor::{AttentionLayer, PageGeometry, PagedAttentionLaunch};
    use moxie_plan::Visibility;

    let layer = AttentionLayer {
        geometry: PageGeometry {
            kv_heads: 2,
            head_dim: 64,
            page_tokens: 16,
            pages: 4,
        },
        heads: 8,
        scale: 1.0,
        visibility: Visibility::Causal,
    };
    let launch = PagedAttentionLaunch::new(layer, 1, 0, 0, 1).expect("a legal launch");
    // An empty catalogue: the "found none" refusal, which is the one that
    // composes the longest prose.
    let empty = KernelCatalogue::new(Vec::new()).expect("an empty catalogue");
    let capability = capability(8, 6);
    for skip in 0..8 {
        let (result, fired) = with_failure_at(skip, || {
            moxie_executor::select_paged_attention_kernel(&empty, &capability, &launch)
        });
        assert!(
            result.is_err(),
            "an empty catalogue selected something at position {skip}"
        );
        if !fired {
            break;
        }
    }
}

// What this file does **not** cover, and why there is no test here for it.
//
// Review asked the sweep to cover successful **admission** too. It is not
// covered, and the first attempt to say so was an empty `#[test]` that reported
// `ok` — which is worse than silence: it counted as a passing test, and the
// behaviour it named is not unmeasured but **measured to abort**. Armed at
// position 4, a plan request built exactly as `resource_request` builds one
// dies with `memory allocation of 5 bytes failed`.
//
// The status is therefore **failing**, and it is recorded where failures are
// recorded — the task record, the support matrix, and
// `docs/tasks/0029-allocation-fallible-admission-vocabulary.md`, which is the
// bounded task that fixes it. A comment cannot pass, which is the point.
