//! An exhaustive sweep of the residency lifecycle's transition combinations.
//!
//! **Why this file exists.** Three rounds of independent review found sixteen
//! defects between them, and the third round's closing note named the pattern
//! rather than another case: "These cases continue to expose gaps between
//! individually passing regressions." Every defect had the same shape — a
//! transition that was individually reasonable left the structure inconsistent
//! in a combination nobody had hand-written a test for, and the damage surfaced
//! one or two operations later, usually as a panic.
//!
//! Point regressions caught each case and missed the next, because the space is
//! a product and the tests were points in it. So this enumerates the product:
//! every combination of destination, urgency, joiner and terminal outcome, with
//! and without a surviving lease, and calls
//! [`ResidencyAuthority::check_invariants`] after **every** operation rather
//! than asserting a hand-picked consequence at the end.
//!
//! What it proves is narrow and worth stating exactly: that no reachable
//! sequence in this space leaves the authority structurally inconsistent, panics,
//! or fails to drain. It does not prove the *policy* is right — the named tests
//! in `residency.rs` do that, and they stay.

use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger, LogicalRange,
    Outcome, PendingWork, ResidencyAuthority, ResidencyLease, ResidencyRequest, TensorSlot,
    TicketId, TurnId, Urgency, UseClass, WorkOrder,
};
use moxie_types::{DeviceUuid, Error, Scope};

const CHUNK: u64 = 4_096;

fn gpu() -> DeviceUuid {
    let mut bytes = [0u8; 16];
    bytes[15] = 7;
    DeviceUuid::from_bytes(bytes)
}

fn chunk(index: u32) -> ChunkId {
    ChunkId::new(
        ArtifactId::new("sweep").unwrap(),
        TensorSlot::expert("experts", index).unwrap(),
        LogicalRange::new(u64::from(index) * CHUNK, CHUNK).unwrap(),
        1,
    )
}

fn ledger() -> Ledger {
    Ledger::new([
        CapacitySnapshot::new(Scope::Host, 1 << 30, 1 << 20).unwrap(),
        CapacitySnapshot::new(Scope::Device(gpu()), 1 << 30, 0).unwrap(),
    ])
    .unwrap()
}

/// Every axis of the space, named so a failure reads as a scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dest {
    Host,
    Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Joiner {
    None,
    HostDemand,
    HostPrefetch,
    DeviceDemand,
    DevicePrefetch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    Completed,
    Failed,
    SubmissionUnknown,
    CancelThenCompleted,
    Expire,
}

#[derive(Debug, Clone, Copy)]
struct Case {
    first: Dest,
    urgency: Urgency,
    joiner: Joiner,
    ending: Ending,
    hold_lease: bool,
}

impl Case {
    fn scope(dest: Dest) -> Scope {
        match dest {
            Dest::Host => Scope::Host,
            Dest::Device => Scope::Device(gpu()),
        }
    }
}

struct Harness {
    authority: ResidencyAuthority,
    ledger: Ledger,
    case: Case,
    step: usize,
}

impl Harness {
    fn open(case: Case) -> Self {
        let mut ledger = ledger();
        let authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("sweep", 8 * CHUNK).device(gpu(), 8 * CHUNK),
        )
        .unwrap();
        Harness {
            authority,
            ledger,
            case,
            step: 0,
        }
    }

    /// Every operation goes through here, so no step can skip the check.
    fn checked(&mut self, what: &str) {
        self.step += 1;
        if let Err(e) = self.authority.check_invariants() {
            panic!(
                "invariant broken after step {} ({what}) in {:?}: {e}",
                self.step, self.case
            );
        }
    }

    fn request<'a>(
        &self,
        id: &'a ChunkId,
        dest: Scope,
        urgency: Urgency,
        now: u64,
        deadline: u64,
    ) -> AcquireRequest<'a> {
        AcquireRequest {
            chunk: id,
            destination: dest,
            now,
            deadline,
            class: UseClass {
                urgency,
                content: Content::Expert,
            },
            turn: TurnId::new(1),
        }
    }

    /// Perform whatever the authority offered, filling reads, and return the
    /// orders that came back. Never assumes a shape.
    fn perform(&mut self, work: PendingWork) -> Vec<WorkOrder> {
        let mut pending = match work {
            PendingWork::Issued(order) => vec![order],
            PendingWork::Coalesced | PendingWork::Queued => Vec::new(),
        };
        let mut done = Vec::new();
        while let Some(order) = pending.pop() {
            match &order {
                WorkOrder::Read { ticket, .. } => {
                    if self.authority.read_destination(*ticket).is_ok() {
                        self.authority.read_destination(*ticket).unwrap().fill(0xAB);
                        let released = self
                            .authority
                            .complete_read(*ticket, Outcome::Completed)
                            .unwrap();
                        self.checked("complete a released read");
                        pending.extend(released);
                    }
                }
                WorkOrder::Upload { ticket, .. } => {
                    if self.authority.upload_source(*ticket).is_ok() {
                        self.authority
                            .complete_upload(*ticket, Outcome::Completed)
                            .unwrap();
                        self.checked("complete a released upload");
                    }
                }
            }
            done.push(order);
        }
        done
    }

    /// Drain everything: prefetch queue, outstanding tickets, leases. The
    /// authority must reach zero from any reachable state.
    fn drain(&mut self, leases: Vec<ResidencyLease>) {
        for lease in leases {
            // A lease whose placement is gone is still a live lease.
            let _ = self.authority.release(lease);
            self.checked("release a lease");
        }

        // Settle anything still in flight, whatever stage it is in.
        for _ in 0..64 {
            let tickets: Vec<TicketId> = self
                .authority
                .outstanding()
                .iter()
                .filter(|c| c.state.is_in_flight())
                .filter_map(|c| self.authority.ticket_of(c.scope, &c.chunk))
                .collect();
            if tickets.is_empty() {
                break;
            }
            for ticket in tickets {
                if self.authority.read_destination(ticket).is_ok() {
                    let _ = self.authority.complete_read(ticket, Outcome::Completed);
                } else if self.authority.upload_source(ticket).is_ok() {
                    let _ = self.authority.complete_upload(ticket, Outcome::Completed);
                } else {
                    break;
                }
                self.checked("drain a ticket");
            }
        }
        // Anything withheld is settled explicitly, which is the only way out.
        for c in self.authority.outstanding() {
            if c.state == moxie_memory::ChunkState::Quarantined && c.leases == 0 {
                let _ = self.authority.settle_quarantined(c.scope, &c.chunk);
                self.checked("settle a quarantined placement");
            }
        }
        while let Some(order) = self.authority.next_prefetch() {
            self.perform(PendingWork::Issued(order));
        }
        self.checked("drain the prefetch queue");
    }
}

/// The sweep. Every combination, every step checked.
#[test]
fn every_transition_combination_keeps_the_authority_consistent() {
    let mut ran = 0usize;
    for first in [Dest::Host, Dest::Device] {
        for urgency in [Urgency::Demand, Urgency::Prefetch] {
            for joiner in [
                Joiner::None,
                Joiner::HostDemand,
                Joiner::HostPrefetch,
                Joiner::DeviceDemand,
                Joiner::DevicePrefetch,
            ] {
                for ending in [
                    Ending::Completed,
                    Ending::Failed,
                    Ending::SubmissionUnknown,
                    Ending::CancelThenCompleted,
                    Ending::Expire,
                ] {
                    for hold_lease in [false, true] {
                        run(Case {
                            first,
                            urgency,
                            joiner,
                            ending,
                            hold_lease,
                        });
                        ran += 1;
                    }
                }
            }
        }
    }
    assert_eq!(ran, 2 * 2 * 5 * 5 * 2, "the sweep must cover the product");
    println!("{ran} transition combinations, every step invariant-checked");
}

fn run(case: Case) {
    let mut h = Harness::open(case);
    h.checked("open");
    let id = chunk(0);
    let mut leases = Vec::new();

    // The first acquire.
    let first = h
        .authority
        .acquire(h.request(&id, Case::scope(case.first), case.urgency, 0, 100));
    h.checked("first acquire");
    let Ok(Acquired::Pending {
        lease,
        ticket,
        work,
        ..
    }) = first
    else {
        // A refusal is a legal outcome; nothing is outstanding, so the case is
        // finished and must still drain to nothing.
        h.drain(leases);
        h.authority.close(&mut h.ledger).unwrap();
        assert_eq!(h.ledger.scope_committed(Scope::Host), 0);
        return;
    };
    leases.push(lease);

    // The joiner, if this case has one.
    if case.joiner != Joiner::None {
        let (dest, urgency) = match case.joiner {
            Joiner::HostDemand => (Scope::Host, Urgency::Demand),
            Joiner::HostPrefetch => (Scope::Host, Urgency::Prefetch),
            Joiner::DeviceDemand => (Scope::Device(gpu()), Urgency::Demand),
            Joiner::DevicePrefetch => (Scope::Device(gpu()), Urgency::Prefetch),
            Joiner::None => unreachable!(),
        };
        let joined = h.authority.acquire(h.request(&id, dest, urgency, 1, 50));
        h.checked("joining acquire");
        if let Ok(Acquired::Pending { lease, work, .. }) = joined {
            leases.push(lease);
            h.perform(work);
            h.checked("perform the joiner's work");
        } else if let Ok(Acquired::Ready(lease)) = joined {
            leases.push(lease);
        }
    }

    // The first acquire's ending.
    match case.ending {
        Ending::Completed => {
            h.perform(work);
            h.checked("complete");
        }
        Ending::Failed => {
            if h.authority.read_destination(ticket).is_ok() {
                let _ = h.authority.complete_read(
                    ticket,
                    Outcome::Failed(Error::InvalidArtifact {
                        detail: "injected".into(),
                    }),
                );
            } else if h.authority.upload_source(ticket).is_ok() {
                let _ = h.authority.complete_upload(
                    ticket,
                    Outcome::Failed(Error::InvalidArtifact {
                        detail: "injected".into(),
                    }),
                );
            }
            h.checked("fail");
        }
        Ending::SubmissionUnknown => {
            let lost = Error::DeviceLost {
                device: 0,
                detail: "unknown".into(),
            };
            if h.authority.read_destination(ticket).is_ok() {
                let _ = h
                    .authority
                    .complete_read(ticket, Outcome::SubmissionUnknown(lost));
            } else if h.authority.upload_source(ticket).is_ok() {
                let _ = h
                    .authority
                    .complete_upload(ticket, Outcome::SubmissionUnknown(lost));
            }
            h.checked("submission unknown");
        }
        Ending::CancelThenCompleted => {
            h.authority.cancel(ticket).unwrap();
            h.checked("cancel");
            h.authority.cancel(ticket).unwrap();
            h.checked("cancel again");
            h.perform(work);
            h.checked("complete after cancellation");
        }
        Ending::Expire => {
            h.authority.expire(1_000);
            h.checked("expire");
        }
    }

    // Anything the authority still offers, and then a second acquire of the
    // same chunk -- the operation that panicked in two separate review rounds.
    while let Some(order) = h.authority.next_prefetch() {
        h.perform(PendingWork::Issued(order));
        h.checked("released prefetch");
    }
    let again =
        h.authority
            .acquire(h.request(&id, Case::scope(case.first), Urgency::Demand, 2, 1_000));
    h.checked("re-acquire after the ending");
    match again {
        Ok(Acquired::Ready(lease)) => leases.push(lease),
        Ok(Acquired::Pending { lease, work, .. }) => {
            leases.push(lease);
            h.perform(work);
            h.checked("perform the re-acquire's work");
        }
        Err(_) => {}
    }

    if !case.hold_lease {
        let held = std::mem::take(&mut leases);
        h.drain(held);
    }

    // A turn end is the other way leases go, and it must reach the same place.
    let cleanup = h.authority.end_turn(TurnId::new(1));
    h.checked("end the turn");
    drop(cleanup);
    let held = std::mem::take(&mut leases);
    h.drain(held);

    // Everything must come back.
    for c in h.authority.outstanding() {
        assert!(
            !c.state.is_in_flight() || c.state == moxie_memory::ChunkState::Quarantined,
            "{:?} left {} {}",
            case,
            c.chunk,
            c.state.name()
        );
    }
    assert_eq!(
        h.authority.live_lease_count(),
        0,
        "{:?} left leases behind",
        case
    );

    // Close, or account for why not: a withheld placement is the one legal
    // reason, and it must be visible rather than silent.
    let quarantined = h
        .authority
        .outstanding()
        .iter()
        .filter(|c| c.state == moxie_memory::ChunkState::Quarantined)
        .count();
    match h.authority.close(&mut h.ledger) {
        Ok(()) => {
            assert_eq!(h.ledger.scope_committed(Scope::Host), 0, "{case:?}");
            assert_eq!(
                h.ledger.scope_committed(Scope::Device(gpu())),
                0,
                "{case:?}"
            );
        }
        Err(e) => {
            assert!(
                quarantined > 0,
                "{case:?} refused to close with nothing withheld: {e}"
            );
        }
    }
}
