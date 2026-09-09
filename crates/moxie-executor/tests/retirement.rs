//! Retirement rules over an injected completion source.
//!
//! Every transition the contract promises, with no GPU present: the driver
//! event is one implementation of `Completion`, and the machine cannot tell it
//! from the manual source used here.

use moxie_executor::{Lease, LeaseState, ManualCompletion, Turn};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_types::{HostTier, Scope, Tier};

fn host_ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 30, 1 << 20).unwrap()]).unwrap()
}

fn admit(ledger: &mut Ledger, label: &str, bytes: u64) -> Reservation {
    let mut req = PlanRequest::new(label, ["run"]).unwrap();
    req.buffer(BufferRequest::new(
        format!("{label}-buf"),
        Scope::Host,
        Tier::Host(HostTier::CpuWorkspace),
        bytes,
        StageSpan { first: 0, last: 0 },
    ))
    .unwrap();
    ledger.admit(&req).unwrap()
}

fn lease(ledger: &mut Ledger, label: &str) -> (Lease, ManualCompletion) {
    let reservation = admit(ledger, label, 1 << 20);
    let completion = ManualCompletion::new();
    let mut lease = Lease::<ManualCompletion>::acquire(reservation, label).unwrap();
    lease.track(completion.clone()).unwrap();
    (lease, completion)
}
fn outstanding_ids(ledger: &Ledger) -> Vec<moxie_memory::ReservationId> {
    ledger.outstanding().iter().map(|o| o.id).collect()
}

#[test]
fn acquire_use_retire_releases_both_sides() {
    let mut ledger = host_ledger();
    let (lease, completion) = lease(&mut ledger, "upload");
    assert_eq!(lease.state(), LeaseState::InFlight);
    let id = lease.id();
    let rid = lease.reservation_id().unwrap();
    assert!(outstanding_ids(&ledger).contains(&rid));

    // Still in flight: retirement refuses and hands the lease back.
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "invalid_request");
    assert!(refused.error.to_string().contains("not observed"));

    // Observed complete: both sides release.
    completion.complete();
    assert_eq!(refused.lease.retire(&mut ledger).unwrap(), id);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn synchronize_is_an_explicit_wait_not_a_backdoor() {
    // Waiting changes nothing observable: retirement still refuses until the
    // source itself reports complete. On the driver this is the visible wait
    // before retire; here it must be a silent no-op that refuses the same way.
    let mut ledger = host_ledger();
    let (lease, completion) = lease(&mut ledger, "wait");
    lease.synchronize().unwrap();
    let refused = lease.retire(&mut ledger).unwrap_err();
    completion.complete();
    refused.lease.synchronize().unwrap();
    refused.lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_refused_lease_stays_usable() {
    // The bite check: retire without querying completion must fail this test.
    // Deleting the event query turns the refusal below into a release.
    let mut ledger = host_ledger();
    let (lease, completion) = lease(&mut ledger, "use");
    let id = lease.id();
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.lease.id(), id);
    assert_eq!(refused.lease.state(), LeaseState::InFlight);
    completion.complete();
    assert_eq!(refused.lease.retire(&mut ledger).unwrap(), id);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_dropped_lease_stays_charged_and_visible() {
    let mut ledger = host_ledger();
    let (lease, _completion) = lease(&mut ledger, "dropped");
    let rid = lease.reservation_id().unwrap();
    drop(lease);
    let ids = outstanding_ids(&ledger);
    assert_eq!(
        ids,
        vec![rid],
        "a dropped lease must stay visible, not leak"
    );
}

#[test]
fn retiring_an_untracked_lease_releases_without_an_event() {
    // The abandon path: acquired, nothing enqueued, nothing in flight.
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "abandon", 1 << 20);
    let lease = Lease::<ManualCompletion>::acquire(reservation, "abandon").unwrap();
    assert_eq!(lease.state(), LeaseState::Live);
    lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn tracking_twice_is_refused() {
    let mut ledger = host_ledger();
    let (mut lease, _) = lease(&mut ledger, "double-track");
    let e = lease.track(ManualCompletion::new()).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("InFlight"));
}

#[test]
fn an_empty_label_is_refused_at_acquire_and_at_turn() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "labelled", 1 << 20);
    let e = Lease::<ManualCompletion>::acquire(reservation, "").unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    let e = Turn::<ManualCompletion>::new("").unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
}

#[test]
fn a_turn_sweep_retires_the_completed_and_names_the_held() {
    let mut ledger = host_ledger();
    let (done, done_flag) = lease(&mut ledger, "done");
    let (held, _) = lease(&mut ledger, "held");
    let done_id = done.id();
    let held_id = held.id();
    let held_rid = held.reservation_id().unwrap();
    done_flag.complete();

    let mut turn = Turn::new("decode").unwrap();
    turn.hold(done);
    turn.hold(held);
    let report = turn.release_turn(&mut ledger);
    assert_eq!(report.retired, vec![done_id]);
    assert_eq!(report.held.len(), 1);
    assert_eq!(report.held[0].id, held_id);
    assert_eq!(report.held[0].label, "held");
    assert!(!report.is_clean());
    // The held lease was dropped by the sweep and stays charged.
    assert_eq!(outstanding_ids(&ledger), vec![held_rid]);
}

#[test]
fn a_turn_with_no_next_token_releases_everything_completable() {
    // R08's shape: the turn ends, nothing arrives to trigger per-lease
    // release, and the sweep must still retire all of it.
    let mut ledger = host_ledger();
    let mut turn = Turn::new("prefill").unwrap();
    let mut ids = Vec::new();
    for i in 0..3 {
        let (lease, flag) = lease(&mut ledger, &format!("buf-{i}"));
        ids.push(lease.id());
        flag.complete();
        turn.hold(lease);
    }
    let report = turn.release_turn(&mut ledger);
    assert!(report.is_clean(), "held: {:?}", report.held);
    assert_eq!(report.retired.len(), 3);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_cancelled_lease_withholds_bytes_until_completion() {
    let mut ledger = host_ledger();
    let (mut lease, completion) = lease(&mut ledger, "cancelled");
    lease.cancel();
    assert_eq!(lease.state(), LeaseState::Cancelled);
    // Intent retired, bytes not: still in flight, still refused.
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert!(refused.error.to_string().contains("not observed"));
    // Observed complete: now it releases.
    completion.complete();
    refused.lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_cancelled_untracked_lease_releases_at_once() {
    // Cancelled before anything was enqueued: nothing is in flight, so there
    // is nothing to withhold.
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "cancel-early", 1 << 20);
    let mut lease = Lease::<ManualCompletion>::acquire(reservation, "cancel-early").unwrap();
    lease.cancel();
    lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_lost_context_withholds_forever() {
    let mut ledger = host_ledger();
    let (mut lease, completion) = lease(&mut ledger, "lost");
    let rid = lease.reservation_id().unwrap();
    lease.mark_lost(0, "transport error");
    assert_eq!(lease.state(), LeaseState::Lost);
    // Even observed completion must not free into a lost context.
    completion.complete();
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "device_lost");
    assert!(refused.error.to_string().contains("withheld"));
    assert_eq!(outstanding_ids(&ledger), vec![rid]);
}

#[test]
fn cancellation_does_not_unlose_a_lease() {
    // Withholding wins over cancellation, in that order too.
    let mut ledger = host_ledger();
    let (mut lease, _) = lease(&mut ledger, "lost-then-cancel");
    lease.mark_lost(1, "transport error");
    lease.cancel();
    assert_eq!(lease.state(), LeaseState::Lost);
    assert_eq!(
        lease.retire(&mut ledger).unwrap_err().error.kind(),
        "device_lost"
    );
}

#[test]
fn retiring_into_the_wrong_ledger_hands_the_lease_back() {
    let mut ledger = host_ledger();
    let mut other = host_ledger();
    let (lease, completion) = lease(&mut ledger, "wrong-ledger");
    completion.complete();
    let refused = lease.retire(&mut other).unwrap_err();
    assert!(outstanding_ids(&ledger).len() == 1);
    assert!(outstanding_ids(&other).is_empty());
    // And the handed-back lease still retires where it belongs.
    refused.lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}
