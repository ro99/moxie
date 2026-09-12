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
//! **The harness is a faithful executor, and the first version was not.** A
//! fourth review mutated `promote_ticket` to discard every order it produced and
//! all 200 combinations still passed, while a named regression caught it in one.
//! The reason: the harness discovered work through `ticket_of` and completed it
//! directly, so it was testing whether the authority *can be poked* into a
//! consistent state, not whether the scheduler ever hands out the work. A sweep
//! that completes work the scheduler lost cannot make a claim about progress.
//!
//! So the harness now tracks only the orders it was actually given — from an
//! acquire, from a completion's follow-ons, from `next_prefetch` — completes
//! nothing else, and **fails when a ticket is left in flight that it never
//! received an order for**. That is the property worth having: every outstanding
//! transfer is one somebody was told to perform.
//!
//! What it proves is narrow and worth stating exactly: that across this space no
//! sequence leaves the authority structurally inconsistent, panics, strands work
//! the executor was never given, or fails to drain. It does not prove the
//! *policy* is right — the named tests in `residency.rs` do that, and they stay.

use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger, LogicalRange,
    Outcome, PendingWork, ResidencyAuthority, ResidencyLease, ResidencyRequest, TensorSlot, TurnId,
    Urgency, UseClass, WorkOrder,
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
    /// A cache too small to hold a second chunk, so admitting one has to
    /// displace something -- with a lease held over the first.
    ///
    /// Without this axis the sweep never ran eviction at all: a mutation that
    /// made eviction ignore leases entirely survived all 200 combinations.
    pressure: bool,
    /// Retire the chunk once it is ready, so `Retiring` is entered and has to
    /// finalise. A mutation that stopped retirement finalising on an internal
    /// pin release survived until this axis existed.
    retire: bool,
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
    /// Exactly the orders the authority handed out, and the only work this
    /// harness may perform.
    orders: Vec<WorkOrder>,
}

impl Harness {
    fn open(case: Case) -> Self {
        let mut ledger = ledger();
        // Under pressure the cache holds exactly one chunk.
        let cap = if case.pressure { CHUNK } else { 8 * CHUNK };
        let authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("sweep", cap).device(gpu(), cap),
        )
        .unwrap();
        Harness {
            authority,
            ledger,
            case,
            step: 0,
            orders: Vec::new(),
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

    /// Record an order the authority handed out. **Only** these may be
    /// performed; nothing is discovered.
    fn receive(&mut self, work: PendingWork) {
        if let PendingWork::Issued(order) = work {
            self.orders.push(order);
        }
    }

    /// Perform every order this harness has been given, and every follow-on
    /// those completions release. Nothing else is touched.
    fn perform_received(&mut self) {
        while let Some(order) = self.orders.pop() {
            match &order {
                WorkOrder::Read { ticket, .. } => {
                    // An order whose ticket has since settled is not an error:
                    // a cancellation can retire it before the executor gets to
                    // it. It is simply no longer performable.
                    if self.authority.read_destination(*ticket).is_ok() {
                        self.authority.read_destination(*ticket).unwrap().fill(0xAB);
                        let released = self
                            .authority
                            .complete_read(*ticket, Outcome::Completed)
                            .unwrap();
                        self.checked("complete a received read");
                        self.orders.extend(released);
                    }
                }
                WorkOrder::Upload { ticket, .. } => {
                    if self.authority.upload_source(*ticket).is_ok() {
                        self.authority
                            .complete_upload(*ticket, Outcome::Completed)
                            .unwrap();
                        self.checked("complete a received upload");
                    }
                }
            }
        }
    }

    /// Release leases, then run the scheduler to exhaustion: everything it
    /// offers is performed, and nothing it did not offer is.
    fn drain(&mut self, leases: Vec<ResidencyLease>) {
        for lease in leases {
            // A lease whose placement is gone is still a live lease.
            let _ = self.authority.release(lease);
            self.checked("release a lease");
        }
        self.perform_received();
        while let Some(order) = self.authority.next_prefetch() {
            self.orders.push(order);
            self.perform_received();
            self.checked("drain the prefetch queue");
        }
        // Anything withheld is settled explicitly, which is the only way out.
        for c in self.authority.outstanding() {
            if c.state == moxie_memory::ChunkState::Quarantined && c.leases == 0 {
                let _ = self.authority.settle_quarantined(c.scope, &c.chunk);
                self.checked("settle a quarantined placement");
            }
        }
    }

    /// The progress property the first harness could not state: after the
    /// scheduler has been run to exhaustion, nothing may still be in flight.
    ///
    /// A placement left `Reading` or `Uploading` here is work the authority is
    /// waiting on and never handed to anybody -- which is exactly what the
    /// mutated `promote_ticket` produced, and what the old harness hid by
    /// completing it anyway.
    fn assert_no_stranded_work(&self) {
        let stranded: Vec<String> = self
            .authority
            .outstanding()
            .iter()
            .filter(|c| c.state.is_in_flight() && c.state != moxie_memory::ChunkState::Quarantined)
            .map(|c| format!("{} is {}", c.chunk, c.state.name()))
            .collect();
        assert!(
            stranded.is_empty(),
            "{:?} left work nobody was told to perform: {}",
            self.case,
            stranded.join(", ")
        );
    }
}

/// The sweep. Every combination, every step checked.
#[test]
fn every_transition_combination_keeps_the_authority_consistent() {
    let mut ran = 0usize;
    let mut endings_applied = 0usize;
    let mut short_circuited = 0usize;
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
                        for pressure in [false, true] {
                            for retire in [false, true] {
                                let covered = run(Case {
                                    first,
                                    urgency,
                                    joiner,
                                    ending,
                                    hold_lease,
                                    pressure,
                                    retire,
                                });
                                ran += 1;
                                if covered.short_circuited {
                                    short_circuited += 1;
                                } else if covered.ending_applied {
                                    endings_applied += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        ran,
        2 * 2 * 5 * 5 * 2 * 2 * 2,
        "the sweep must cover the product"
    );
    // Coverage is reported, not assumed. A previous version of this sweep let
    // 120 cases reach their ending and do nothing while the record claimed
    // every ending applied; the number is printed now so the claim and the
    // measurement are the same thing.
    assert_eq!(
        endings_applied + short_circuited,
        ran,
        "a case neither applied its ending nor short-circuited"
    );
    println!(
        "{ran} transition combinations, every step invariant-checked: \
         {endings_applied} applied their ending, {short_circuited} were \
         short-circuited by a hit or a refusal before reaching one"
    );
}

/// What one case actually exercised, so the sweep can report coverage rather
/// than only a pass.
#[derive(Debug, Default, Clone, Copy)]
struct Covered {
    ending_applied: bool,
    /// The case ended before its ending could be reached: the first acquire was
    /// a hit or a refusal. These are legal and are counted separately rather
    /// than excused inside the assertion.
    short_circuited: bool,
}

fn run(case: Case) -> Covered {
    let mut h = Harness::open(case);
    h.checked("open");
    let id = chunk(0);
    let mut leases = Vec::new();

    // When this case retires, the chunk is warmed to `HostReady` first and its
    // lease released. That is what lets a later device acquire pin a *settled*
    // source, so retirement can happen while an upload -- not a consumer --
    // holds the last thing keeping the placement alive. Retiring only after
    // everything had completed never reached that, and a mutation that stopped
    // retirement finalising on an internal pin release survived because of it.
    // Only when the first acquire targets a device: that is the shape where an
    // upload's pin can be the last thing holding the source. Warming before a
    // *host* acquire would just turn it into a hit and skip the case entirely,
    // which is 100 combinations that would exercise nothing.
    if case.retire && case.first == Dest::Device {
        let warm = h
            .authority
            .acquire(h.request(&id, Scope::Host, Urgency::Demand, 0, 1_000));
        h.checked("warm acquire");
        match warm {
            Ok(Acquired::Pending { lease, work, .. }) => {
                h.receive(work);
                h.perform_received();
                h.checked("warm read");
                h.authority.release(lease).unwrap();
                h.checked("release the warm lease");
            }
            Ok(Acquired::Ready(lease)) => {
                h.authority.release(lease).unwrap();
                h.checked("release the warm lease");
            }
            Err(_) => {}
        }
    }

    // The first acquire.
    let first = h
        .authority
        .acquire(h.request(&id, Case::scope(case.first), case.urgency, 0, 100));
    h.checked("first acquire");
    let (lease, ticket, work) = match first {
        Ok(Acquired::Pending {
            lease,
            ticket,
            work,
            ..
        }) => (lease, ticket, work),
        // A hit -- the warm preamble already made it resident -- and a refusal
        // are both legal outcomes with nothing outstanding. The case is over,
        // and it must still drain to nothing.
        other => {
            if let Ok(Acquired::Ready(lease)) = other {
                leases.push(lease);
            }
            h.drain(leases);
            h.assert_no_stranded_work();
            h.authority.close(&mut h.ledger).unwrap();
            assert_eq!(h.ledger.scope_committed(Scope::Host), 0);
            return Covered {
                ending_applied: false,
                short_circuited: true,
            };
        }
    };
    leases.push(lease);

    h.receive(work);

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
            h.receive(work);
        } else if let Ok(Acquired::Ready(lease)) = joined {
            leases.push(lease);
        }
    }

    // The first acquire's ending.
    // Retirement, if this case has it, happens **while a transfer is still
    // outstanding**: stop serving the chunk and let its last holder free it.
    // Whatever that holder is -- a consumer's lease or an upload's internal pin
    // -- releasing it must finish the job, or the placement is stranded:
    // charged, unservable and unevictable.
    if case.retire {
        let _ = h.authority.retire(Scope::Host, &id);
        h.checked("retire the host chunk");
        let _ = h.authority.retire(Scope::Device(gpu()), &id);
        h.checked("retire the device chunk");
    }

    // Release queued work **without performing it**, while the first acquire and
    // its joiner are both still outstanding. This is where a dependency ordering
    // error shows: the entry with the earliest deadline may be the one waiting
    // on another's read. Performing here is what previously let an ending find
    // nothing to act on.
    while let Some(order) = h.authority.next_prefetch() {
        h.orders.push(order);
        h.checked("early prefetch release");
    }

    // Every ending must actually happen. A branch that silently did nothing --
    // because an earlier step had already settled the ticket -- would make the
    // case a duplicate of `Completed` while claiming to test a failure path.
    let ending_applied;
    match case.ending {
        Ending::Completed => {
            h.perform_received();
            h.checked("complete");
            ending_applied = true;
        }
        Ending::Failed => {
            let injected = Error::InvalidArtifact {
                detail: "injected".into(),
            };
            if h.authority.read_destination(ticket).is_ok() {
                h.authority
                    .complete_read(ticket, Outcome::Failed(injected))
                    .unwrap();
                ending_applied = true;
            } else if h.authority.upload_source(ticket).is_ok() {
                h.authority
                    .complete_upload(ticket, Outcome::Failed(injected))
                    .unwrap();
                ending_applied = true;
            } else {
                ending_applied = false;
            }
            h.checked("fail");
        }
        Ending::SubmissionUnknown => {
            let lost = Error::DeviceLost {
                device: 0,
                detail: "unknown".into(),
            };
            if h.authority.read_destination(ticket).is_ok() {
                h.authority
                    .complete_read(ticket, Outcome::SubmissionUnknown(lost))
                    .unwrap();
                ending_applied = true;
            } else if h.authority.upload_source(ticket).is_ok() {
                h.authority
                    .complete_upload(ticket, Outcome::SubmissionUnknown(lost))
                    .unwrap();
                ending_applied = true;
            } else {
                ending_applied = false;
            }
            h.checked("submission unknown");
        }
        Ending::CancelThenCompleted => {
            h.authority.cancel(ticket).unwrap();
            h.checked("cancel");
            h.authority.cancel(ticket).unwrap();
            h.checked("cancel again");
            h.perform_received();
            h.checked("complete after cancellation");
            ending_applied = true;
        }
        Ending::Expire => {
            let expired = h.authority.expire(1_000);
            h.checked("expire");
            ending_applied = !expired.is_empty();
        }
    }

    // **Every ending must actually happen.** Nothing is performed before this
    // point, so the first acquire's ticket is still outstanding and there is
    // always something for a failure, a loss or an expiry to act on. A case
    // whose ending silently did nothing is a duplicate of `Completed` wearing
    // another name, and an earlier version of this sweep had 120 of them while
    // the record claimed otherwise.
    assert!(
        ending_applied,
        "{:?} claimed an ending it never applied",
        case
    );

    // Anything the authority still offers, and then a second acquire of the
    // same chunk -- the operation that panicked in two separate review rounds.
    while let Some(order) = h.authority.next_prefetch() {
        h.orders.push(order);
        h.perform_received();
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
            h.receive(work);
            h.perform_received();
            h.checked("perform the re-acquire's work");
        }
        Err(_) => {}
    }

    // Cache pressure, with a lease still held. Admitting a second chunk into a
    // one-chunk cache must displace something -- and it may not displace what a
    // consumer is holding. That is a *functional* property no structural
    // invariant can see, so it is asserted directly: whatever a live lease
    // could read before the pressure, it can still read after.
    if case.pressure {
        let readable: Vec<(usize, Vec<u8>)> = leases
            .iter()
            .enumerate()
            .filter_map(|(i, l)| h.authority.chunk_bytes(l).ok().map(|b| (i, b.to_vec())))
            .collect();
        let other = chunk(1);
        let squeeze =
            h.authority
                .acquire(h.request(&other, Scope::Host, Urgency::Demand, 3, 1_000));
        h.checked("acquire under pressure");
        if let Ok(Acquired::Pending { lease, work, .. }) = squeeze {
            leases.push(lease);
            h.receive(work);
            h.perform_received();
            h.checked("perform under pressure");
        } else if let Ok(Acquired::Ready(lease)) = squeeze {
            leases.push(lease);
        }
        for (i, before) in readable {
            let after = h.authority.chunk_bytes(&leases[i]).unwrap_or_else(|e| {
                panic!("{:?} lost a held chunk to eviction: {e}", case);
            });
            // A retired chunk still serves the leases it already had, which is
            // the whole point of `Retiring`; eviction must not take it either.
            assert_eq!(
                after,
                &before[..],
                "{:?} served a held chunk different bytes after eviction",
                case
            );
        }
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

    // Everything must come back, and nothing may be left waiting on a transfer
    // that was never handed out.
    h.assert_no_stranded_work();
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

    Covered {
        ending_applied,
        short_circuited: false,
    }
}
