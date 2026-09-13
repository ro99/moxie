//! Isolated executable: counted heap requests on the routed path.
//!
//! This module already learned this once. `softmax` used a plain `collect()`
//! while routing was an unreached M0 fixture, and it became a process abort the
//! moment task 0019 put it under the interpreter: an allocation failure inside a
//! generation step must be a typed error the transaction can roll back, not a
//! panic that takes the rollback, the lease release and the next generation with
//! it. Every allocation on this path has been fallible since.
//!
//! Task 0022 reintroduced the defect in `narrow_coefficients`, and an
//! independent review found it. The fix is stronger than a fallible allocation:
//! the function **owns** the route, so it rounds in place and allocates nothing.
//! This test is what keeps that true, and it counts allocator *calls* rather
//! than injecting a failure, because "requires no allocation" is the property
//! and a call count states it directly.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use moxie_graph::{RouteCoefficient, RouteScore, RouterInput};
use moxie_oracles::route::{self, Route, RouterSpec};

struct Counter;
static CALLS: AtomicUsize = AtomicUsize::new(0);
// SAFETY: allocation operations are forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the allocator caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            CALLS.fetch_add(1, SeqCst);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(pointer, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;

#[test]
fn narrowing_a_routes_coefficients_requests_no_heap() {
    let route = Route {
        experts: vec![3, 1],
        weights: vec![0.593_845_5, 0.406_154_5],
    };
    // The vectors above are already allocated; what is counted is what
    // narrowing adds.
    let before = CALLS.load(SeqCst);
    let narrowed = route::narrow_coefficients(route, RouteCoefficient::Bf16);
    let after = CALLS.load(SeqCst);
    assert_eq!(
        after - before,
        0,
        "narrowing requested {} allocation(s); it owns the route and must round in place",
        after - before
    );
    assert_eq!(narrowed.weights, vec![0.593_75, 0.406_25]);
    assert_eq!(narrowed.experts, vec![3, 1]);

    // And the identity arm, which never had the defect, still has no
    // allocation either -- so the two arms are the same shape rather than one
    // being accidentally cheap.
    let route = Route {
        experts: vec![3, 1],
        weights: vec![0.25, 0.75],
    };
    let before = CALLS.load(SeqCst);
    let same = route::narrow_coefficients(route, RouteCoefficient::Fp32);
    assert_eq!(CALLS.load(SeqCst) - before, 0);
    assert_eq!(same.weights, vec![0.25, 0.75]);
}

#[test]
fn the_whole_router_allocates_only_what_it_must() {
    // The narrowing arm must not cost a router a single extra allocation over
    // the unnarrowed one. A regression that only counted `narrow_coefficients`
    // in isolation would pass if the caller had started cloning the route to
    // hand it over.
    let hidden = 16;
    let experts = 8;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 - 7.0) / 4.0).collect();
    let proj: Vec<f32> = (0..experts * hidden)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 32.0)
        .collect();
    let bias: Vec<f32> = (0..experts).map(|e| (e as f32 - 4.0) / 64.0).collect();
    let spec = |coefficient| RouterSpec {
        experts,
        top_k: 3,
        input: RouterInput::Raw,
        score: RouteScore::Sigmoid,
        coefficient,
    };

    // Warm anything lazily initialised behind the first call.
    let _ = route::router_route_row(
        &x,
        None,
        &proj,
        None,
        Some(&bias),
        spec(RouteCoefficient::Fp32),
    );

    let before = CALLS.load(SeqCst);
    let plain = route::router_route_row(
        &x,
        None,
        &proj,
        None,
        Some(&bias),
        spec(RouteCoefficient::Fp32),
    )
    .unwrap();
    let plain_calls = CALLS.load(SeqCst) - before;

    let before = CALLS.load(SeqCst);
    let narrowed = route::router_route_row(
        &x,
        None,
        &proj,
        None,
        Some(&bias),
        spec(RouteCoefficient::Bf16),
    )
    .unwrap();
    let narrowed_calls = CALLS.load(SeqCst) - before;

    assert_eq!(
        narrowed_calls, plain_calls,
        "narrowing cost {narrowed_calls} allocation(s) against {plain_calls} without it"
    );
    // And it did something, so this is not two identical cheap paths.
    assert_ne!(narrowed.weights, plain.weights);
    assert_eq!(narrowed.experts, plain.experts);
}

/// A router's operand list is read once per `Route` node per step, and costs no
/// heap either.
///
/// Found by applying the review's own question to the rest of the same change
/// rather than by the review: `route_operands` returned a `Vec`, and the
/// interpreter calls it inside a generation step. The list is bounded by
/// construction -- five operands, each pushed at most once -- so it is held
/// inline.
#[test]
fn a_routers_operand_list_requests_no_heap() {
    use moxie_graph::{OpParams, RouteOperand};

    let params = OpParams::Route {
        hidden: 8,
        experts: 4,
        top_k: 2,
        input: RouterInput::Normalized {
            eps: 1e-6,
            input_scale: 1.0,
        },
        score: RouteScore::Softmax,
        per_expert_scale: true,
        selection_bias: true,
        coefficient: RouteCoefficient::Fp32,
    };
    // Warm anything the first call touches.
    let _ = params.route_operands();

    let before = CALLS.load(SeqCst);
    let operands = params.route_operands();
    let arity = params.arity();
    assert_eq!(
        CALLS.load(SeqCst) - before,
        0,
        "reading a router's operand list allocated"
    );

    // The widest list, in the documented order, and the arity derived from it.
    assert_eq!(
        operands.as_slice(),
        &[
            RouteOperand::Rows,
            RouteOperand::Projection,
            RouteOperand::Gain,
            RouteOperand::PerExpertScale,
            RouteOperand::SelectionBias,
        ]
    );
    assert_eq!(arity, operands.len());
    assert_eq!(operands.len(), moxie_graph::RouteOperands::MAX);

    // The narrowest list is the two required operands, not an empty one.
    let minimal = OpParams::Route {
        hidden: 8,
        experts: 4,
        top_k: 2,
        input: RouterInput::Raw,
        score: RouteScore::Sigmoid,
        per_expert_scale: false,
        selection_bias: false,
        coefficient: RouteCoefficient::Bf16,
    };
    assert_eq!(
        minimal.route_operands().as_slice(),
        &[RouteOperand::Rows, RouteOperand::Projection]
    );

    // And an operation that is not a router has none.
    assert!(
        OpParams::Residual { scale: 1.0 }
            .route_operands()
            .is_empty()
    );
}
