//! Retirement rules over injected completion sources.
//!
//! Every transition the contract promises, with no GPU present: the driver
//! event is one implementation of `Completion`, and the machine cannot tell it
//! from the doubles used here.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use moxie_executor::{
    Lease, LeaseState, ManualCompletion, Script, ScriptedCompletion, SettledResource, Turn,
    check_fit,
};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_types::{DeviceUuid, HostTier, Scope, Tier};

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

fn tracked_lease(ledger: &mut Ledger, label: &str) -> (Lease, ManualCompletion) {
    let reservation = admit(ledger, label, 1 << 20);
    let lease = Lease::<ManualCompletion>::acquire(ledger, reservation, label).unwrap();
    let completion = ManualCompletion::new();
    let mut lease = lease;
    lease.track_manual(completion.clone()).unwrap();
    (lease, completion)
}

fn outstanding_ids(ledger: &Ledger) -> Vec<moxie_memory::ReservationId> {
    ledger.outstanding().iter().map(|o| o.id).collect()
}

/// A resource that counts its own destructions: the only way to observe
/// whether a dropped lease freed early or withheld.
#[derive(Debug)]
struct CountedDrop {
    drops: Arc<AtomicUsize>,
}

impl CountedDrop {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let drops = Arc::new(AtomicUsize::new(0));
        (
            CountedDrop {
                drops: drops.clone(),
            },
            drops,
        )
    }
}

impl Drop for CountedDrop {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl SettledResource for CountedDrop {
    type Settled = ();
    fn settle(self) {}
}

#[test]
fn acquire_binds_the_admitted_scope_and_bytes() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "bound", 1 << 20);
    let lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "bound").unwrap();
    assert_eq!(lease.admitted(), &[(Scope::Host, 1 << 20)]);
    assert_eq!(lease.admitted_bytes(), 1 << 20);
}

#[test]
fn check_fit_refuses_wrong_scope_and_over_budget() {
    let admitted = vec![(Scope::Host, 512u64)];
    assert!(check_fit(&admitted, Scope::Host, 512).is_ok());
    assert!(
        check_fit(&admitted, Scope::Host, 4096)
            .unwrap_err()
            .to_string()
            .contains("4096")
    );
    let dev = Scope::Device(DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000001").unwrap());
    let e = check_fit(&admitted, dev, 8).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    let e = check_fit(&admitted, Scope::Host, 4096).unwrap_err();
    assert_eq!(e.kind(), "capacity_exceeded");
}

#[test]
fn acquire_refuses_an_empty_label_and_returns_the_reservation() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "labelled", 1 << 20);
    let rid = reservation.id();
    let refused = Lease::<ManualCompletion>::acquire(&ledger, reservation, "").unwrap_err();
    assert_eq!(refused.error.kind(), "invalid_request");
    // The reservation comes back usable: the refusal stranded nothing.
    let lease =
        Lease::<ManualCompletion>::acquire(&ledger, refused.reservation, "labelled").unwrap();
    assert_eq!(lease.reservation_id(), Some(rid));
}

#[test]
fn acquire_refuses_a_reservation_from_another_ledger() {
    let mut ledger = host_ledger();
    let other = host_ledger();
    let reservation = admit(&mut ledger, "foreign", 1 << 20);
    let refused = Lease::<ManualCompletion>::acquire(&other, reservation, "foreign").unwrap_err();
    assert!(refused.error.to_string().contains("not outstanding"));
    // Still charged where it belongs, untouched elsewhere — and reusable.
    assert_eq!(outstanding_ids(&ledger).len(), 1);
    assert!(outstanding_ids(&other).is_empty());
    let _reacquired =
        Lease::<ManualCompletion>::acquire(&ledger, refused.reservation, "foreign").unwrap();
}

#[test]
fn acquire_use_retire_releases_both_sides_and_settles_the_resource() {
    let mut ledger = host_ledger();
    let (lease, completion) = tracked_lease(&mut ledger, "upload");
    assert_eq!(lease.state(), LeaseState::InFlight);
    let id = lease.id();
    let rid = lease.reservation_id().unwrap();
    assert!(outstanding_ids(&ledger).contains(&rid));

    // Still in flight: retirement refuses and hands the lease back.
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "invalid_request");
    assert!(refused.error.to_string().contains("not observed"));

    // Observed complete: both sides release, resource settled.
    completion.complete();
    let (retired, ()) = refused.lease.retire(&mut ledger).unwrap();
    assert_eq!(retired, id);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_retained_source_comes_back_only_at_retirement() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "src", 1 << 20);
    let lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "src").unwrap();
    let completion = ManualCompletion::new();
    let mut lease = lease.retain(vec![7u8; 8]);
    lease.track_manual(completion.clone()).unwrap();
    // From here the caller names no source: it moved. Retirement returns it.
    completion.complete();
    let (_, src) = lease.retire(&mut ledger).unwrap();
    assert_eq!(src, vec![7u8; 8]);
}

#[test]
fn a_refused_lease_stays_usable() {
    // The bite check: retire without querying completion must fail this test.
    // Deleting the event query turns the refusal below into a release.
    let mut ledger = host_ledger();
    let (lease, completion) = tracked_lease(&mut ledger, "use");
    let id = lease.id();
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.lease.id(), id);
    assert_eq!(refused.lease.state(), LeaseState::InFlight);
    completion.complete();
    assert_eq!(refused.lease.retire(&mut ledger).unwrap().0, id);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn synchronize_is_an_explicit_wait_not_a_backdoor() {
    // Waiting changes nothing observable: retirement still refuses until the
    // source itself reports complete. On the driver this is the visible wait
    // before retire; here it must be a silent no-op that refuses the same way.
    let mut ledger = host_ledger();
    let (mut lease, completion) = tracked_lease(&mut ledger, "wait");
    lease.synchronize().unwrap();
    let mut refused = lease.retire(&mut ledger).unwrap_err();
    completion.complete();
    refused.lease.synchronize().unwrap();
    refused.lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_dropped_live_lease_frees_its_unsubmitted_resource() {
    // No completion was ever tracked, so nothing was submitted: normal drop.
    // This is the only droppable shape, and the abandon path relies on it.
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "live-drop", 1 << 20);
    let (resource, drops) = CountedDrop::new();
    let lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "live-drop")
        .unwrap()
        .retain(resource);
    drop(lease);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn a_dropped_inflight_lease_withholds_its_resource() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "inflight-drop", 1 << 20);
    let (resource, drops) = CountedDrop::new();
    let completion = ManualCompletion::new();
    let mut lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "inflight-drop")
        .unwrap()
        .retain(resource);
    lease.track_manual(completion).unwrap();
    let rid = lease.reservation_id().unwrap();
    drop(lease);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "an in-flight resource must survive the lease drop"
    );
    assert_eq!(outstanding_ids(&ledger), vec![rid]);
}

#[test]
fn a_dropped_cancelled_lease_withholds_its_resource() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "cancelled-drop", 1 << 20);
    let (resource, drops) = CountedDrop::new();
    let completion = ManualCompletion::new();
    let mut lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "cancelled-drop")
        .unwrap()
        .retain(resource);
    lease.track_manual(completion).unwrap();
    lease.cancel();
    drop(lease);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "a cancelled-but-unobserved resource must survive the lease drop"
    );
    assert_eq!(outstanding_ids(&ledger).len(), 1);
}

#[test]
fn a_dropped_lost_lease_withholds_its_resource() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "lost-drop", 1 << 20);
    let (resource, drops) = CountedDrop::new();
    let completion = ManualCompletion::new();
    let mut lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "lost-drop")
        .unwrap()
        .retain(resource);
    lease.track_manual(completion).unwrap();
    lease.mark_lost(0, "transport error");
    drop(lease);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "a lost resource must survive the lease drop"
    );
    assert_eq!(outstanding_ids(&ledger).len(), 1);
}

#[test]
fn a_dropped_lease_stays_charged_and_visible() {
    let mut ledger = host_ledger();
    let (lease, _completion) = tracked_lease(&mut ledger, "dropped");
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
    let lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "abandon").unwrap();
    assert_eq!(lease.state(), LeaseState::Live);
    lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn tracking_twice_is_refused() {
    let mut ledger = host_ledger();
    let (mut lease, _) = tracked_lease(&mut ledger, "double-track");
    let e = lease.track_manual(ManualCompletion::new()).unwrap_err();
    assert_eq!(e.kind(), "invalid_request");
    assert!(e.to_string().contains("InFlight"));
}

#[test]
fn a_turn_sweep_retires_the_completed_and_returns_the_rest() {
    let mut ledger = host_ledger();
    let (done, done_flag) = tracked_lease(&mut ledger, "done");
    let (held, _) = tracked_lease(&mut ledger, "held");
    let done_id = done.id();
    let held_id = held.id();
    done_flag.complete();

    let mut turn = Turn::new("decode").unwrap();
    turn.hold(done);
    turn.hold(held);
    let report = turn.release_turn(&mut ledger);
    assert_eq!(report.retired.len(), 1);
    assert_eq!(report.retired[0].id, done_id);
    assert_eq!(report.held.len(), 1);
    assert_eq!(report.held[0].id, held_id);
    assert_eq!(report.held[0].label, "held");
    assert!(!report.is_clean());
}

#[test]
fn a_sweep_returns_retained_resources_with_the_release() {
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "kept", 1 << 20);
    let lease = Lease::<ManualCompletion>::acquire(&ledger, reservation, "kept").unwrap();
    let completion = ManualCompletion::new();
    let mut lease = lease.retain(vec![9u8; 4]);
    lease.track_manual(completion.clone()).unwrap();
    completion.complete();
    let mut turn = Turn::new("sweep-kept").unwrap();
    turn.hold(lease);
    let report = turn.release_turn(&mut ledger);
    assert!(report.is_clean());
    assert_eq!(report.retired.len(), 1);
    assert_eq!(report.retired[0].resource, vec![9u8; 4]);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_second_sweep_after_completion_releases_with_no_next_token() {
    // P1's repro shape: the first sweep names the hold and returns the
    // handle; the event completes; a second sweep with no new work releases.
    let mut ledger = host_ledger();
    let (lease, completion) = tracked_lease(&mut ledger, "late");
    let id = lease.id();
    let mut first = Turn::new("turn-1").unwrap();
    first.hold(lease);
    let report = first.release_turn(&mut ledger);
    assert!(!report.is_clean());
    assert_eq!(outstanding_ids(&ledger).len(), 1);

    completion.complete();
    let mut second = Turn::new("turn-2").unwrap();
    second.hold(report.held.into_iter().next().unwrap().lease);
    let report = second.release_turn(&mut ledger);
    assert!(
        report.is_clean(),
        "held: {:?}",
        report.held.iter().map(|h| &h.label).collect::<Vec<_>>()
    );
    assert_eq!(report.retired.len(), 1);
    assert_eq!(report.retired[0].id, id);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_turn_with_no_next_token_releases_everything_completable() {
    // R08's shape: the turn ends, nothing arrives to trigger per-lease
    // release, and the sweep must still retire all of it.
    let mut ledger = host_ledger();
    let mut turn = Turn::new("prefill").unwrap();
    for i in 0..3 {
        let (lease, flag) = tracked_lease(&mut ledger, &format!("buf-{i}"));
        flag.complete();
        turn.hold(lease);
    }
    let report = turn.release_turn(&mut ledger);
    assert!(
        report.is_clean(),
        "held: {:?}",
        report.held.iter().map(|h| &h.label).collect::<Vec<_>>()
    );
    assert_eq!(report.retired.len(), 3);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_cancelled_lease_withholds_bytes_until_completion() {
    let mut ledger = host_ledger();
    let (mut lease, completion) = tracked_lease(&mut ledger, "cancelled");
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
fn cancellation_then_completion_then_sweep_releases() {
    // The reviewer-named sequence: cancel, complete, sweep again — no next
    // token arrives in between, and nothing may strand.
    let mut ledger = host_ledger();
    let (mut lease, completion) = tracked_lease(&mut ledger, "cancel-sweep");
    lease.cancel();
    let mut turn = Turn::new("turn-1").unwrap();
    turn.hold(lease);
    let report = turn.release_turn(&mut ledger);
    assert!(!report.is_clean());
    completion.complete();
    let mut turn = Turn::new("turn-2").unwrap();
    turn.hold(report.held.into_iter().next().unwrap().lease);
    let report = turn.release_turn(&mut ledger);
    assert!(report.is_clean());
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_cancelled_untracked_lease_releases_at_once() {
    // Cancelled before anything was enqueued: nothing is in flight, so there
    // is nothing to withhold.
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "cancel-early", 1 << 20);
    let mut lease =
        Lease::<ManualCompletion>::acquire(&ledger, reservation, "cancel-early").unwrap();
    lease.cancel();
    lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn observed_loss_persists_past_a_racing_completion() {
    // P2's repro: the source reports loss, then completion. The lease must
    // stay withheld; the returned handle must stay unusable.
    let mut ledger = host_ledger();
    let script = ScriptedCompletion::new([
        Script::Lost(0, "transport error".into()),
        Script::Ready(true),
    ]);
    let reservation = admit(&mut ledger, "race", 1 << 20);
    let mut lease = Lease::<ScriptedCompletion>::acquire(&ledger, reservation, "race").unwrap();
    lease.track_manual(script).unwrap();
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "device_lost");
    let refused = refused.lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "device_lost");
    assert!(refused.error.to_string().contains("withheld"));
    assert_eq!(outstanding_ids(&ledger).len(), 1);
}

#[test]
fn observed_loss_through_synchronize_persists_too() {
    // A wait whose handle reports loss must withhold exactly like a query
    // that does: the lease below never completes, it only loses.
    #[derive(Debug)]
    struct LossyWait;
    impl moxie_executor::Completion for LossyWait {
        fn query_complete(&self) -> Result<bool, moxie_types::Error> {
            Ok(false)
        }
        fn synchronize(&self) -> Result<(), moxie_types::Error> {
            Err(moxie_types::Error::DeviceLost {
                device: 1,
                detail: "wait observed the loss".into(),
            })
        }
        fn describe(&self) -> String {
            "lossy wait".into()
        }
    }
    impl moxie_executor::TrackableManual for LossyWait {
        fn into_tracked(self) -> Self {
            self
        }
    }
    let mut ledger = host_ledger();
    let reservation = admit(&mut ledger, "race-sync", 1 << 20);
    let mut lease = Lease::<LossyWait>::acquire(&ledger, reservation, "race-sync").unwrap();
    lease.track_manual(LossyWait).unwrap();
    let e = lease.synchronize().unwrap_err();
    assert_eq!(e.kind(), "device_lost");
    assert_eq!(lease.state(), LeaseState::Lost);
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "device_lost");
}

#[test]
fn a_transient_source_failure_does_not_withhold() {
    // Only device loss persists. A broken source refuses this observation and
    // stays usable for the next one.
    let mut ledger = host_ledger();
    let script = ScriptedCompletion::new([
        Script::Broken("query transport hiccup".into()),
        Script::Ready(true),
    ]);
    let reservation = admit(&mut ledger, "flaky", 1 << 20);
    let mut lease = Lease::<ScriptedCompletion>::acquire(&ledger, reservation, "flaky").unwrap();
    lease.track_manual(script).unwrap();
    let id = lease.id();
    let refused = lease.retire(&mut ledger).unwrap_err();
    assert_eq!(refused.error.kind(), "invalid_request");
    let (retired, ()) = refused.lease.retire(&mut ledger).unwrap();
    assert_eq!(retired, id);
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_lost_context_withholds_forever() {
    let mut ledger = host_ledger();
    let (mut lease, completion) = tracked_lease(&mut ledger, "lost");
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
    let (mut lease, _) = tracked_lease(&mut ledger, "lost-then-cancel");
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
    let (lease, completion) = tracked_lease(&mut ledger, "wrong-ledger");
    completion.complete();
    let refused = lease.retire(&mut other).unwrap_err();
    assert_eq!(outstanding_ids(&ledger).len(), 1);
    assert!(outstanding_ids(&other).is_empty());
    // And the handed-back lease still retires where it belongs.
    refused.lease.retire(&mut ledger).unwrap();
    assert!(ledger.outstanding().is_empty());
}
