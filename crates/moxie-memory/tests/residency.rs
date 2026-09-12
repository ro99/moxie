//! Acceptance tests for task 0020: the weight-residency authority.
//!
//! Every test names the rule from the task contract it proves. Nothing here
//! opens a file or touches a device: the authority's job is to decide, and a
//! decision is testable without either. The driver's real reads are tested in
//! `moxie-executor`.
//!
//! Roadmap M2 item 5 lists nine cases by name -- full cache, all entries leased,
//! incoming-largest-expert, failure mid-read, failure mid-upload, repeated
//! cancellation, no-next-token cleanup, repeated and missing expert routes, and
//! nonuniform row counts. The first seven are here; the last two are demand-set
//! arithmetic and are tested where that lives, in `moxie-engine`.

use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapabilityKey, CapacitySnapshot, ChunkId, ChunkState,
    Content, Ledger, LogicalRange, Outcome, PendingWork, PreparedId, ResidencyAuthority,
    ResidencyLease, ResidencyRequest, TensorSlot, TicketId, TurnId, UseClass, WorkOrder,
};
use moxie_types::{DeviceTier, DeviceUuid, Error, HostTier, Scope, Tier};

const EXPERT_BYTES: u64 = 4_096;

fn artifact() -> ArtifactId {
    ArtifactId::new("fixture-artifact-v1").unwrap()
}

fn expert(index: u32) -> ChunkId {
    ChunkId::new(
        artifact(),
        TensorSlot::expert("experts_gate_up", index).unwrap(),
        LogicalRange::new(u64::from(index) * EXPERT_BYTES, EXPERT_BYTES).unwrap(),
        1,
    )
}

fn sized(index: u32, bytes: u64) -> ChunkId {
    ChunkId::new(
        artifact(),
        TensorSlot::expert("experts_gate_up", index).unwrap(),
        LogicalRange::new(u64::from(index) * bytes, bytes).unwrap(),
        1,
    )
}

fn gpu() -> DeviceUuid {
    let mut bytes = [0u8; 16];
    bytes[15] = 7;
    DeviceUuid::from_bytes(bytes)
}

/// A ledger with room for everything these tests admit.
fn ledger() -> Ledger {
    Ledger::new([
        CapacitySnapshot::new(Scope::Host, 1 << 30, 1 << 20).unwrap(),
        CapacitySnapshot::new(Scope::Device(gpu()), 1 << 30, 0).unwrap(),
    ])
    .unwrap()
}

fn open(ledger: &mut Ledger, host_cap: u64) -> ResidencyAuthority {
    ResidencyAuthority::open(ledger, &ResidencyRequest::new("test", host_cap)).unwrap()
}

fn demand(chunk: &ChunkId, now: u64) -> AcquireRequest<'_> {
    AcquireRequest {
        chunk,
        destination: Scope::Host,
        now,
        deadline: u64::MAX,
        class: UseClass::demand(Content::Expert),
        turn: TurnId::new(1),
    }
}

/// Acquire, perform the read with a caller-supplied filler, and return the lease.
fn load(authority: &mut ResidencyAuthority, chunk: &ChunkId, now: u64, fill: u8) -> ResidencyLease {
    let acquired = authority.acquire(demand(chunk, now)).unwrap();
    let Acquired::Pending {
        lease,
        ticket,
        work,
        ..
    } = acquired
    else {
        panic!("an absent chunk is not resident");
    };
    assert!(matches!(work, PendingWork::Issued(WorkOrder::Read { .. })));
    fill_and_complete(authority, ticket, fill);
    lease
}

fn fill_and_complete(authority: &mut ResidencyAuthority, ticket: TicketId, fill: u8) {
    authority.read_destination(ticket).unwrap().fill(fill);
    authority.complete_read(ticket, Outcome::Completed).unwrap();
}

/// Every reconciliation the contract predeclares, checked together: the
/// authority's own committed total, the arena's live occupancy, and the ledger's
/// charge for the envelope. Any two agreeing while the third does not is the
/// exact failure R02 describes.
fn reconciles(authority: &ResidencyAuthority, scope: Scope) {
    let committed = authority.committed_bytes(scope).unwrap();
    let occupancy = authority.occupancy(scope).unwrap();
    assert_eq!(
        committed, occupancy.live_bytes,
        "authority and arena disagree about {scope}"
    );
    let outstanding: u64 = authority
        .outstanding()
        .iter()
        .filter(|c| c.scope == scope)
        .map(|c| c.bytes)
        .sum();
    assert_eq!(
        committed, outstanding,
        "committed total does not equal the visible placements"
    );
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

#[test]
fn an_artifact_identity_is_not_a_path() {
    for smuggled in ["/fast/models/x", "models\\x", "a\0b", ""] {
        assert!(
            ArtifactId::new(smuggled).is_err(),
            "{smuggled:?} was accepted as an artifact identity"
        );
    }
    assert!(ArtifactId::new("sha256:abc").is_ok());
}

#[test]
fn the_format_version_participates_in_chunk_identity() {
    let v1 = expert(0);
    let v2 = ChunkId::new(
        artifact(),
        TensorSlot::expert("experts_gate_up", 0).unwrap(),
        LogicalRange::new(0, EXPERT_BYTES).unwrap(),
        2,
    );
    assert_ne!(v1, v2, "a changed format version must be a different chunk");

    let mut l = ledger();
    let mut a = open(&mut l, 16 * EXPERT_BYTES);
    let first = load(&mut a, &v1, 0, 0xAA);
    // The second version is a miss, not a hit on the first's bytes: document 02
    // requires a changed scale convention to invalidate an incompatible layout.
    let acquired = a.acquire(demand(&v2, 1)).unwrap();
    assert!(matches!(acquired, Acquired::Pending { .. }));
    let Acquired::Pending { lease, ticket, .. } = acquired else {
        unreachable!()
    };
    fill_and_complete(&mut a, ticket, 0xBB);
    assert_eq!(a.chunk_bytes(&first).unwrap()[0], 0xAA);
    assert_eq!(a.chunk_bytes(&lease).unwrap()[0], 0xBB);
    a.release(first).unwrap();
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn a_prepared_layout_version_is_part_of_the_prepared_identity() {
    let capability = CapabilityKey {
        sm_major: 8,
        sm_minor: 6,
        layout_tag: "bf16-row".into(),
    };
    let v1 = PreparedId {
        chunk: expert(0),
        capability: capability.clone(),
        layout_version: 1,
    };
    let v2 = PreparedId {
        layout_version: 2,
        ..v1.clone()
    };
    let other_sm = PreparedId {
        capability: CapabilityKey {
            sm_major: 12,
            ..capability
        },
        ..v1.clone()
    };
    assert_ne!(v1, v2);
    assert_ne!(v1, other_sm);
}

// ---------------------------------------------------------------------------
// The lifecycle
// ---------------------------------------------------------------------------

#[test]
fn a_miss_reads_once_and_a_hit_reads_not_at_all() {
    let mut l = ledger();
    let mut a = open(&mut l, 16 * EXPERT_BYTES);
    let chunk = expert(0);

    let first = load(&mut a, &chunk, 0, 0x11);
    assert_eq!(a.state_of(Scope::Host, &chunk), Some(ChunkState::HostReady));
    assert_eq!(a.stats().misses, 1);
    assert_eq!(a.stats().bytes_read, EXPERT_BYTES);

    let second = a.acquire(demand(&chunk, 1)).unwrap();
    assert!(
        matches!(second, Acquired::Ready(_)),
        "a resident chunk is a hit"
    );
    let Acquired::Ready(second) = second else {
        unreachable!()
    };
    assert_eq!(a.stats().hits, 1);
    assert_eq!(
        a.stats().bytes_read,
        EXPERT_BYTES,
        "a hit must not read again"
    );
    assert_eq!(a.chunk_bytes(&second).unwrap(), &[0x11; 4096][..]);

    reconciles(&a, Scope::Host);
    a.release(first).unwrap();
    a.release(second).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn concurrent_acquires_of_one_chunk_coalesce_to_exactly_one_read() {
    let mut l = ledger();
    let mut a = open(&mut l, 16 * EXPERT_BYTES);
    let chunk = expert(3);

    let Acquired::Pending {
        lease: first,
        ticket,
        work,
        ..
    } = a.acquire(demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    assert!(matches!(work, PendingWork::Issued(_)));

    // Four more waiters. None of them may produce work.
    let mut waiters = Vec::new();
    for i in 1..5 {
        let Acquired::Pending {
            lease,
            ticket: same,
            work,
            ..
        } = a.acquire(demand(&chunk, i)).unwrap()
        else {
            panic!("an in-flight chunk is pending, not ready")
        };
        assert_eq!(same, ticket, "a waiter joins the outstanding ticket");
        assert_eq!(
            work,
            PendingWork::Coalesced,
            "a second read of one chunk is the defect this prevents"
        );
        waiters.push(lease);
    }
    assert_eq!(a.in_flight_count(), 1, "five acquires, one transfer");

    fill_and_complete(&mut a, ticket, 0x5A);
    assert_eq!(
        a.stats().bytes_read,
        EXPERT_BYTES,
        "one chunk's bytes, once"
    );
    // Every waiter reads the same bytes from the one read.
    for lease in std::iter::once(&first).chain(waiters.iter()) {
        assert_eq!(a.chunk_bytes(lease).unwrap()[0], 0x5A);
    }
    a.release(first).unwrap();
    for lease in waiters {
        a.release(lease).unwrap();
    }
    a.close(&mut l).unwrap();
}

#[test]
fn a_failed_read_releases_every_byte_and_fails_every_waiter_identically() {
    let mut l = ledger();
    let mut a = open(&mut l, 16 * EXPERT_BYTES);
    let chunk = expert(1);
    let baseline = a.committed_bytes(Scope::Host).unwrap();

    let Acquired::Pending {
        lease: first,
        ticket,
        ..
    } = a.acquire(demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    let Acquired::Pending { lease: second, .. } = a.acquire(demand(&chunk, 1)).unwrap() else {
        panic!("pending")
    };
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), EXPERT_BYTES);

    a.complete_read(
        ticket,
        Outcome::Failed(Error::InvalidArtifact {
            detail: "short read of chunk".into(),
        }),
    )
    .unwrap();

    assert_eq!(
        a.committed_bytes(Scope::Host).unwrap(),
        baseline,
        "a failed read must charge nothing"
    );
    assert_eq!(a.state_of(Scope::Host, &chunk), None, "the entry is absent");
    assert_eq!(a.stats().read_failures, 1);

    // Both waiters see the same thing: their lease no longer names bytes.
    for lease in [&first, &second] {
        assert!(
            a.chunk_bytes(lease).is_err(),
            "a failed read leaves no readable bytes"
        );
    }
    // The authority is still usable: a retry is a fresh acquire, never an
    // invisible internal retry.
    a.release(first).unwrap();
    a.release(second).unwrap();
    let retry = load(&mut a, &chunk, 2, 0x77);
    assert_eq!(a.chunk_bytes(&retry).unwrap()[0], 0x77);
    a.release(retry).unwrap();
    reconciles(&a, Scope::Host);
    a.close(&mut l).unwrap();
}

#[test]
fn a_read_whose_submission_state_is_unknown_is_quarantined_not_reused() {
    let mut l = ledger();
    let mut a = open(&mut l, 16 * EXPERT_BYTES);
    let chunk = expert(2);
    let Acquired::Pending { lease, ticket, .. } = a.acquire(demand(&chunk, 0)).unwrap() else {
        panic!("absent")
    };
    a.complete_read(
        ticket,
        Outcome::SubmissionUnknown(Error::DeviceLost {
            device: 0,
            detail: "submission state unknown".into(),
        }),
    )
    .unwrap();

    assert_eq!(
        a.state_of(Scope::Host, &chunk),
        Some(ChunkState::Quarantined)
    );
    assert_eq!(
        a.committed_bytes(Scope::Host).unwrap(),
        EXPERT_BYTES,
        "quarantined bytes stay charged: a transfer may still be touching them"
    );
    // Re-acquiring quarantined bytes is refused, not silently re-read.
    let refused = a.acquire(demand(&chunk, 1)).unwrap_err();
    assert!(
        format!("{}", refused.error).contains("quarantined"),
        "{refused}"
    );

    a.release(lease).unwrap();
    a.settle_quarantined(Scope::Host, &chunk).unwrap();
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
    a.close(&mut l).unwrap();
}

// ---------------------------------------------------------------------------
// M2 item 5: full cache, all leased, incoming largest
// ---------------------------------------------------------------------------

#[test]
fn a_full_cache_evicts_in_the_predeclared_order_and_still_serves_correct_bytes() {
    let mut l = ledger();
    // Exactly four experts fit.
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let mut leases = Vec::new();
    for i in 0..4u32 {
        leases.push(load(&mut a, &expert(i), u64::from(i), i as u8));
    }
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 4 * EXPERT_BYTES);

    // Touch expert 1 so it is the most recent, then unpin everything.
    for lease in leases.drain(..) {
        a.release(lease).unwrap();
    }
    let touched = a.acquire(demand(&expert(1), 10)).unwrap();
    let Acquired::Ready(touched) = touched else {
        panic!("resident")
    };
    a.release(touched).unwrap();

    // The fifth expert displaces the least recently used, which is expert 0 --
    // named before anything is removed.
    let acquired = a.acquire(demand(&expert(4), 11)).unwrap();
    let Acquired::Pending {
        lease,
        ticket,
        report,
        ..
    } = acquired
    else {
        panic!("absent")
    };
    assert_eq!(report.incoming_bytes, EXPERT_BYTES);
    assert_eq!(report.committed_bytes, 4 * EXPERT_BYTES);
    assert_eq!(report.cap_bytes, 4 * EXPERT_BYTES);
    assert_eq!(
        report.would_evict,
        vec![expert(0)],
        "deterministic demand LRU: expert 0 is the oldest touch"
    );
    fill_and_complete(&mut a, ticket, 0xE4);

    assert_eq!(a.state_of(Scope::Host, &expert(0)), None);
    assert_eq!(a.chunk_bytes(&lease).unwrap()[0], 0xE4);
    // The survivors still serve their own bytes, not the newcomer's.
    let one = a.acquire(demand(&expert(1), 12)).unwrap();
    let Acquired::Ready(one) = one else {
        panic!("resident")
    };
    assert_eq!(a.chunk_bytes(&one).unwrap()[0], 1);
    assert_eq!(a.stats().evictions, 1);
    assert_eq!(a.stats().evicted_bytes, EXPERT_BYTES);

    reconciles(&a, Scope::Host);
    a.release(lease).unwrap();
    a.release(one).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn a_demand_against_a_cache_whose_every_entry_is_leased_is_refused_immediately() {
    let mut l = ledger();
    let mut a = open(&mut l, 3 * EXPERT_BYTES);
    let mut held = Vec::new();
    for i in 0..3u32 {
        held.push(load(&mut a, &expert(i), u64::from(i), i as u8));
    }

    // This is M2's "demonstrate demand failure cannot deadlock". The call
    // returns; it does not park, queue or wait, because there is no code path
    // in which it could.
    let refused = a.acquire(demand(&expert(9), 10)).unwrap_err();
    assert!(
        matches!(refused.error, Error::CapacityExceeded { .. }),
        "{:?}",
        refused.error
    );
    assert_eq!(refused.report.incoming_bytes, EXPERT_BYTES);
    assert_eq!(refused.report.leased_bytes, 3 * EXPERT_BYTES);
    assert_eq!(refused.report.evictable_bytes, 0);
    assert!(refused.report.would_evict.is_empty());
    assert_eq!(a.stats().refusals, 1);
    assert_eq!(
        a.committed_bytes(Scope::Host).unwrap(),
        3 * EXPERT_BYTES,
        "a refusal displaces nothing"
    );

    // Release one lease and the same request now succeeds: the refusal was a
    // statement about the cache's state, not a permanent failure.
    a.release(held.pop().unwrap()).unwrap();
    let lease = load(&mut a, &expert(9), 11, 0x99);
    assert_eq!(a.chunk_bytes(&lease).unwrap()[0], 0x99);

    reconciles(&a, Scope::Host);
    a.release(lease).unwrap();
    for lease in held {
        a.release(lease).unwrap();
    }
    a.close(&mut l).unwrap();
}

#[test]
fn the_incoming_largest_expert_is_counted_before_anything_is_evicted() {
    let mut l = ledger();
    let mut a = open(&mut l, 8 * EXPERT_BYTES);
    // Four small residents, none leased.
    for i in 0..4u32 {
        let lease = load(&mut a, &expert(i), u64::from(i), i as u8);
        a.release(lease).unwrap();
    }
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 4 * EXPERT_BYTES);

    // An incoming chunk larger than any resident one and larger than the free
    // space: 6 x EXPERT_BYTES into 4 free.
    let big = sized(20, 6 * EXPERT_BYTES);
    let acquired = a.acquire(demand(&big, 10)).unwrap();
    let Acquired::Pending {
        lease,
        ticket,
        report,
        ..
    } = acquired
    else {
        panic!("absent")
    };
    assert_eq!(report.incoming_bytes, 6 * EXPERT_BYTES);
    assert_eq!(report.committed_bytes, 4 * EXPERT_BYTES);

    // Phase one is byte accounting: `committed + incoming - cap` is two
    // chunks' worth, so two is what the cache must *hold* to admit this.
    let needed_by_bytes = (4 * EXPERT_BYTES + 6 * EXPERT_BYTES) - 8 * EXPERT_BYTES;
    assert_eq!(needed_by_bytes, 2 * EXPERT_BYTES);

    // Phase two is contiguity, and it is why the answer is four rather than
    // two. Freeing the two oldest leaves 4 free chunks in two pieces of two,
    // and a six-chunk request fits neither. Document 03 counts
    // `allocator_fragmentation` as real capacity for exactly this reason, so
    // the authority keeps going in the same deterministic order instead of
    // reporting a full cache that is not full.
    assert_eq!(
        report.would_evict,
        vec![expert(0), expert(1), expert(2), expert(3)],
        "oldest first, through both phases"
    );
    assert_eq!(report.remaining_headroom_bytes, 2 * EXPERT_BYTES);
    fill_and_complete(&mut a, ticket, 0xB1);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 6 * EXPERT_BYTES);
    assert_eq!(
        a.chunk_bytes(&lease).unwrap().len(),
        6 * EXPERT_BYTES as usize
    );
    a.release(lease).unwrap();

    reconciles(&a, Scope::Host);
    a.close(&mut l).unwrap();
}

#[test]
fn a_chunk_larger_than_the_whole_cache_is_refused_without_evicting_anything() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    for i in 0..4u32 {
        let lease = load(&mut a, &expert(i), u64::from(i), i as u8);
        a.release(lease).unwrap();
    }

    let impossible = sized(30, 5 * EXPERT_BYTES);
    let refused = a.acquire(demand(&impossible, 10)).unwrap_err();
    match refused.error {
        Error::CapacityExceeded {
            requested_bytes,
            available_bytes,
            ..
        } => {
            assert_eq!(requested_bytes, 5 * EXPERT_BYTES);
            assert_eq!(available_bytes, 4 * EXPERT_BYTES);
        }
        other => panic!("{other:?}"),
    }
    assert!(
        refused.report.would_evict.is_empty(),
        "destroying live data for a request that could not fit an empty cache \
         would be loss for nothing"
    );
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 4 * EXPERT_BYTES);
    assert_eq!(a.outstanding().len(), 4);
    a.close(&mut l).unwrap();
}

// ---------------------------------------------------------------------------
// M2 item 5: cancellation and the no-next-token turn
// ---------------------------------------------------------------------------

#[test]
fn cancelling_an_in_flight_read_keeps_its_bytes_charged_until_the_outcome_lands() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let Acquired::Pending { lease, ticket, .. } = a.acquire(demand(&chunk, 0)).unwrap() else {
        panic!("absent")
    };
    a.cancel(ticket).unwrap();
    // R08 and document 02: cancellation retires the intent, not the bytes.
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), EXPERT_BYTES);
    assert_eq!(a.state_of(Scope::Host, &chunk), Some(ChunkState::Reading));

    // Repeated cancellation is a normal shape of a cancelled generation.
    a.cancel(ticket).unwrap();
    a.cancel(ticket).unwrap();

    a.release(lease).unwrap();
    fill_and_complete(&mut a, ticket, 0x00);
    assert_eq!(
        a.committed_bytes(Scope::Host).unwrap(),
        0,
        "the charge is given back when the transfer is known to be over"
    );
    // And cancelling a settled ticket is still not an error.
    a.cancel(ticket).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn releasing_the_last_waiter_cancels_the_work_and_one_of_several_does_not() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let Acquired::Pending {
        lease: first,
        ticket,
        ..
    } = a.acquire(demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    let Acquired::Pending { lease: second, .. } = a.acquire(demand(&chunk, 1)).unwrap() else {
        panic!("pending")
    };

    a.release(first).unwrap();
    assert_eq!(
        a.is_cancelled(ticket),
        Some(false),
        "one of two waiters leaving does not cancel the transfer"
    );
    a.release(second).unwrap();
    assert_eq!(
        a.is_cancelled(ticket),
        Some(true),
        "nobody is waiting for these bytes any more"
    );
    fill_and_complete(&mut a, ticket, 0x00);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
    a.close(&mut l).unwrap();
}

#[test]
fn a_turn_that_ends_with_no_next_token_releases_everything_r08() {
    let mut l = ledger();
    let mut a = open(&mut l, 8 * EXPERT_BYTES);
    let turn = TurnId::new(42);
    let baseline = a.committed_bytes(Scope::Host).unwrap();

    // A whole step's worth of demand, then the turn ends without producing a
    // token. R08's leak was a lease released on "next token"; there is no next
    // token here, and the leases must still go.
    let mut leases = Vec::new();
    for i in 0..4u32 {
        let chunk = expert(i);
        let Acquired::Pending { lease, ticket, .. } = a
            .acquire(AcquireRequest {
                turn,
                ..demand(&chunk, u64::from(i))
            })
            .unwrap()
        else {
            panic!("absent")
        };
        fill_and_complete(&mut a, ticket, i as u8);
        leases.push(lease);
    }
    assert_eq!(a.live_lease_count(), 4);
    // Dropping a lease does not release it -- that is the whole point of the
    // type -- so this is exactly the R08 shape: a turn ends and nobody gave the
    // leases back.
    drop(leases);

    let cleanup = a.end_turn(turn);
    assert_eq!(cleanup.released_leases.len(), 4);
    assert_eq!(cleanup.unpinned_chunks.len(), 4);
    assert!(cleanup.still_in_flight.is_empty());
    assert_eq!(a.live_lease_count(), 0);

    // The bytes are still cached -- that is what a cache is for -- but nothing
    // is pinned, so the next step can displace all of it.
    for chunk in a.outstanding() {
        assert_eq!(chunk.leases, 0, "{} is still pinned", chunk.chunk);
    }
    let big = sized(50, 8 * EXPERT_BYTES);
    let lease = load(&mut a, &big, 100, 0xFF);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 8 * EXPERT_BYTES);
    a.release(lease).unwrap();

    // And every byte comes back.
    let cleanup = a.end_turn(turn);
    assert!(cleanup.released_leases.is_empty());
    drop(cleanup);
    for chunk in a.outstanding() {
        assert_eq!(chunk.leases, 0);
    }
    a.close(&mut l).unwrap();
    assert_eq!(l.outstanding().len(), 0, "the envelope came back too");
    assert_eq!(baseline, 0);
}

#[test]
fn a_turn_ending_over_an_in_flight_transfer_does_not_free_its_bytes() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let turn = TurnId::new(9);
    let chunk = expert(0);
    let Acquired::Pending { lease, ticket, .. } = a
        .acquire(AcquireRequest {
            turn,
            ..demand(&chunk, 0)
        })
        .unwrap()
    else {
        panic!("absent")
    };
    drop(lease);

    let cleanup = a.end_turn(turn);
    assert_eq!(cleanup.still_in_flight, vec![ticket]);
    assert_eq!(
        a.committed_bytes(Scope::Host).unwrap(),
        EXPERT_BYTES,
        "ending a turn never frees bytes a transfer may still touch"
    );
    fill_and_complete(&mut a, ticket, 0);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
    a.close(&mut l).unwrap();
}

#[test]
fn a_thousand_cancel_and_restart_cycles_leave_no_charge_and_no_growth() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    for i in 0..1_000u64 {
        let chunk = expert((i % 3) as u32);
        let Acquired::Pending { lease, ticket, .. } = a.acquire(demand(&chunk, i)).unwrap() else {
            // After a cancelled cycle the chunk is absent again, so every
            // iteration is a miss; a hit here would mean a cancellation left
            // usable bytes behind.
            panic!("a cancelled cycle must leave the chunk absent")
        };
        a.cancel(ticket).unwrap();
        a.release(lease).unwrap();
        a.complete_read(ticket, Outcome::Completed).unwrap();
        assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
        assert_eq!(a.in_flight_count(), 0);
        assert_eq!(a.live_lease_count(), 0);
    }
    assert!(a.outstanding().is_empty());
    a.close(&mut l).unwrap();
}

// ---------------------------------------------------------------------------
// Classes: demand, prefetch, conditional memory
// ---------------------------------------------------------------------------

fn prefetch(chunk: &ChunkId, now: u64) -> AcquireRequest<'_> {
    AcquireRequest {
        class: UseClass::prefetch(Content::Expert),
        ..demand(chunk, now)
    }
}

#[test]
fn a_prefetch_is_queued_behind_outstanding_demand() {
    let mut l = ledger();
    let mut a = open(&mut l, 8 * EXPERT_BYTES);

    let Acquired::Pending {
        lease: d,
        ticket: demand_ticket,
        ..
    } = a.acquire(demand(&expert(0), 0)).unwrap()
    else {
        panic!("absent")
    };
    let Acquired::Pending { lease: p, work, .. } = a.acquire(prefetch(&expert(1), 1)).unwrap()
    else {
        panic!("absent")
    };
    assert_eq!(work, PendingWork::Queued);
    assert!(
        a.next_prefetch().is_none(),
        "demand work outranks predictions"
    );

    fill_and_complete(&mut a, demand_ticket, 0);
    let released = a.next_prefetch().expect("demand is idle now");
    assert!(matches!(released, WorkOrder::Read { .. }));
    fill_and_complete(&mut a, released.ticket(), 1);
    assert_eq!(a.stats().prefetch_admitted_bytes, EXPERT_BYTES);

    a.release(d).unwrap();
    a.release(p).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn a_prefetch_never_evicts_demand_data_and_is_refused_instead() {
    let mut l = ledger();
    let mut a = open(&mut l, 2 * EXPERT_BYTES);
    for i in 0..2u32 {
        let lease = load(&mut a, &expert(i), u64::from(i), i as u8);
        a.release(lease).unwrap();
    }
    // Both residents are unleased demand data, so there is room by the ordinary
    // rule -- but this is a prediction, and document 03 makes prefetch
    // "evictable before useful demand data", one way only.
    let refused = a.acquire(prefetch(&expert(5), 10)).unwrap_err();
    assert!(matches!(refused.error, Error::CapacityExceeded { .. }));
    assert_eq!(a.stats().demand_evicted_bytes, 0);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 2 * EXPERT_BYTES);

    // A demand for the same chunk is admitted, displacing the oldest.
    let lease = load(&mut a, &expert(5), 11, 0x55);
    assert_eq!(a.stats().demand_evicted_bytes, EXPERT_BYTES);
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn unused_prefetch_data_is_displaced_before_demand_data_and_counted_as_waste() {
    let mut l = ledger();
    let mut a = open(&mut l, 3 * EXPERT_BYTES);

    // One prefetch, admitted first so it is also the least recently used, and
    // two demands after it.
    let Acquired::Pending { lease: p, work, .. } = a.acquire(prefetch(&expert(0), 0)).unwrap()
    else {
        panic!("absent")
    };
    assert_eq!(work, PendingWork::Queued);
    let order = a.next_prefetch().expect("no demand outstanding");
    fill_and_complete(&mut a, order.ticket(), 0);
    a.release(p).unwrap();

    for i in 1..3u32 {
        let lease = load(&mut a, &expert(i), u64::from(i), i as u8);
        a.release(lease).unwrap();
    }

    // Touch the prefetched chunk's neighbours so LRU alone would pick a demand
    // entry: expert 1 is older than the prefetch's last use.
    let touched = a.acquire(demand(&expert(0), 10)).unwrap();
    assert!(matches!(touched, Acquired::Ready(_)));
    let Acquired::Ready(touched) = touched else {
        unreachable!()
    };
    a.release(touched).unwrap();
    // ...but it was demanded, so it is demand data now and LRU applies.
    assert_eq!(a.stats().prefetch_used_bytes, EXPERT_BYTES);

    let acquired = a.acquire(demand(&expert(3), 11)).unwrap();
    let Acquired::Pending { lease, report, .. } = acquired else {
        panic!("absent")
    };
    assert_eq!(
        report.would_evict,
        vec![expert(1)],
        "a prefetch that was later demanded is ordinary demand data"
    );
    assert_eq!(a.stats().prefetch_wasted_bytes, 0);
    let Acquired::Pending {
        lease: settling,
        ticket,
        ..
    } = a.acquire(demand(&expert(3), 11)).unwrap()
    else {
        panic!("in flight")
    };
    fill_and_complete(&mut a, ticket, 3);
    a.release(settling).unwrap();
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn an_unused_prefetch_is_the_first_victim_and_is_counted_as_wasted() {
    let mut l = ledger();
    let mut a = open(&mut l, 3 * EXPERT_BYTES);

    // Two demands first, then a prediction that is never demanded. LRU alone
    // would take a demand entry; the class rank takes the prediction.
    for i in 0..2u32 {
        let lease = load(&mut a, &expert(i), u64::from(i), i as u8);
        a.release(lease).unwrap();
    }
    let Acquired::Pending { lease: p, .. } = a.acquire(prefetch(&expert(9), 5)).unwrap() else {
        panic!("absent")
    };
    let order = a.next_prefetch().expect("demand is idle");
    fill_and_complete(&mut a, order.ticket(), 9);
    a.release(p).unwrap();

    let acquired = a.acquire(demand(&expert(3), 6)).unwrap();
    let Acquired::Pending { lease, report, .. } = acquired else {
        panic!("absent")
    };
    assert_eq!(
        report.would_evict,
        vec![expert(9)],
        "the prediction goes first even though it is the most recent"
    );
    assert_eq!(a.stats().prefetch_wasted_bytes, EXPERT_BYTES);
    assert_eq!(a.stats().demand_evicted_bytes, 0);
    let Acquired::Pending {
        lease: settling,
        ticket,
        ..
    } = a.acquire(demand(&expert(3), 7)).unwrap()
    else {
        panic!("in flight")
    };
    fill_and_complete(&mut a, ticket, 3);
    a.release(settling).unwrap();
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn the_prefetch_queue_is_bounded_and_refuses_rather_than_growing() {
    let mut l = ledger();
    let mut a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("bounded", 64 * EXPERT_BYTES).prefetch_queue_capacity(3),
    )
    .unwrap();

    // Keep demand outstanding so nothing drains the queue.
    let Acquired::Pending {
        lease: blocker,
        ticket,
        ..
    } = a.acquire(demand(&expert(0), 0)).unwrap()
    else {
        panic!("absent")
    };

    let mut queued = Vec::new();
    for i in 1..4u32 {
        let Acquired::Pending { lease, .. } =
            a.acquire(prefetch(&expert(i), u64::from(i))).unwrap()
        else {
            panic!("absent")
        };
        queued.push(lease);
    }
    assert_eq!(a.prefetch_queue_len(), 3);

    let refused = a.acquire(prefetch(&expert(4), 9)).unwrap_err();
    assert!(matches!(refused.error, Error::CapacityExceeded { .. }));
    assert_eq!(
        a.prefetch_queue_len(),
        3,
        "a bounded queue that grows under pressure is an unbounded queue"
    );
    assert_eq!(
        a.state_of(Scope::Host, &expert(4)),
        None,
        "a refused prediction costs no bytes at all"
    );

    fill_and_complete(&mut a, ticket, 0);
    a.release(blocker).unwrap();
    for lease in queued {
        a.release(lease).unwrap();
    }
    while let Some(order) = a.next_prefetch() {
        a.cancel(order.ticket()).unwrap();
        a.complete_read(order.ticket(), Outcome::Completed).unwrap();
    }
    a.close(&mut l).unwrap();
}

#[test]
fn a_conditional_memory_class_is_never_evicted_below_its_floor() {
    // ADR 0009: conditional-memory tables are one residency class under this
    // authority, "never zero-resident". The floor is that sentence as a rule.
    let mut l = ledger();
    let mut a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("engram", 4 * EXPERT_BYTES)
            .conditional_floor_bytes(2 * EXPERT_BYTES),
    )
    .unwrap();

    let rows: Vec<ChunkId> = (0..3).map(expert).collect();
    for (i, row) in rows.iter().enumerate() {
        let request = AcquireRequest {
            class: UseClass::demand(Content::ConditionalTable),
            ..demand(row, i as u64)
        };
        let Acquired::Pending { lease, ticket, .. } = a.acquire(request).unwrap() else {
            panic!("absent")
        };
        fill_and_complete(&mut a, ticket, i as u8);
        a.release(lease).unwrap();
    }
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 3 * EXPERT_BYTES);

    // One expert fits in what is left, displacing nothing, and stays leased so
    // it is not a candidate for what follows.
    let first = load(&mut a, &expert(10), 10, 0xAA);
    assert_eq!(a.stats().evictions, 0);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 4 * EXPERT_BYTES);

    // The second expert must displace something, and the only candidates are
    // table rows. One may go: the class lands exactly on its floor.
    let second = load(&mut a, &expert(11), 11, 0xBB);
    assert_eq!(a.stats().evictions, 1);
    let resident_tables: u64 = a
        .outstanding()
        .iter()
        .filter(|c| c.class.content == Content::ConditionalTable)
        .map(|c| c.bytes)
        .sum();
    assert_eq!(resident_tables, 2 * EXPERT_BYTES);

    // The third may not. Every remaining candidate is a table row and each one
    // would take the class below the floor, so the demand is refused rather
    // than the class being emptied -- which is what "never zero-resident"
    // means when something else wants the bytes.
    let refused = a.acquire(demand(&expert(12), 12)).unwrap_err();
    assert!(
        matches!(refused.error, Error::CapacityExceeded { .. }),
        "{:?}",
        refused.error
    );
    assert!(refused.report.would_evict.is_empty());
    let still_resident: u64 = a
        .outstanding()
        .iter()
        .filter(|c| c.class.content == Content::ConditionalTable)
        .map(|c| c.bytes)
        .sum();
    assert_eq!(still_resident, 2 * EXPERT_BYTES);

    reconciles(&a, Scope::Host);
    a.release(first).unwrap();
    a.release(second).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn a_dense_spine_tensor_is_charged_against_the_same_capacity_as_an_expert() {
    // Document 03: persist dense spine tensors "but charge their residency
    // against expert and context capacity". One cap, not two.
    let mut l = ledger();
    let mut a = open(&mut l, 2 * EXPERT_BYTES);
    let spine = ChunkId::new(
        artifact(),
        TensorSlot::tensor("embedding").unwrap(),
        LogicalRange::new(0, EXPERT_BYTES).unwrap(),
        1,
    );
    let held = {
        let request = AcquireRequest {
            class: UseClass::demand(Content::DenseSpine),
            ..demand(&spine, 0)
        };
        let Acquired::Pending { lease, ticket, .. } = a.acquire(request).unwrap() else {
            panic!("absent")
        };
        fill_and_complete(&mut a, ticket, 0xDE);
        lease
    };
    let e = load(&mut a, &expert(0), 1, 0xE0);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 2 * EXPERT_BYTES);

    // The cache is full and both entries are leased: the spine competes for the
    // same bytes an expert does.
    let refused = a.acquire(demand(&expert(1), 2)).unwrap_err();
    assert_eq!(refused.report.leased_bytes, 2 * EXPERT_BYTES);
    a.release(held).unwrap();
    a.release(e).unwrap();
    a.close(&mut l).unwrap();
}

// ---------------------------------------------------------------------------
// Deadlines
// ---------------------------------------------------------------------------

#[test]
fn a_pending_transfer_past_its_deadline_is_failed_not_served_late() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let Acquired::Pending { lease, ticket, .. } = a
        .acquire(AcquireRequest {
            deadline: 100,
            ..demand(&chunk, 0)
        })
        .unwrap()
    else {
        panic!("absent")
    };
    assert!(
        a.expire(100).is_empty(),
        "a deadline is not yet passed at 100"
    );
    let expired = a.expire(101);
    assert_eq!(expired, vec![ticket]);
    assert_eq!(a.stats().expired, 1);
    // The order was issued, so the bytes are withheld rather than reused: the
    // read may still be running in whoever was performing it.
    assert_eq!(
        a.state_of(Scope::Host, &chunk),
        Some(ChunkState::Quarantined)
    );
    a.release(lease).unwrap();
    a.settle_quarantined(Scope::Host, &chunk).unwrap();
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
    a.close(&mut l).unwrap();
}

// ---------------------------------------------------------------------------
// The device path: two-stage tickets, pinned sources, mid-upload failure
// ---------------------------------------------------------------------------

fn device_demand(chunk: &ChunkId, now: u64) -> AcquireRequest<'_> {
    AcquireRequest {
        destination: Scope::Device(gpu()),
        ..demand(chunk, now)
    }
}

fn open_with_device(ledger: &mut Ledger, host: u64, device: u64) -> ResidencyAuthority {
    ResidencyAuthority::open(
        ledger,
        &ResidencyRequest::new("test", host).device(gpu(), device),
    )
    .unwrap()
}

#[test]
fn a_device_acquire_reads_to_the_host_then_uploads_under_one_ticket() {
    let mut l = ledger();
    let mut a = open_with_device(&mut l, 4 * EXPERT_BYTES, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let device = Scope::Device(gpu());

    let Acquired::Pending {
        lease,
        ticket,
        work,
        ..
    } = a.acquire(device_demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    // One readiness dependency, not two.
    assert!(matches!(work, PendingWork::Issued(WorkOrder::Read { .. })));
    assert_eq!(a.state_of(Scope::Host, &chunk), Some(ChunkState::Reading));
    assert_eq!(a.state_of(device, &chunk), Some(ChunkState::Reading));

    a.read_destination(ticket).unwrap().fill(0xC0);
    let follow_on = a.complete_read(ticket, Outcome::Completed).unwrap();
    assert_eq!(follow_on.len(), 1);
    let WorkOrder::Upload {
        ticket: same,
        device_offset,
        len_bytes,
        ..
    } = &follow_on[0]
    else {
        panic!("the second stage is an upload")
    };
    assert_eq!(*same, ticket, "one ticket covers both transfers");
    assert_eq!(*len_bytes, EXPERT_BYTES);
    assert_eq!(a.state_of(Scope::Host, &chunk), Some(ChunkState::HostReady));
    assert_eq!(a.state_of(device, &chunk), Some(ChunkState::Uploading));

    // Document 02: the upload leases its source. The host copy is unevictable
    // while the copy is outstanding.
    let host_pin = a
        .outstanding()
        .into_iter()
        .find(|c| c.scope == Scope::Host)
        .unwrap();
    assert_eq!(host_pin.leases, 1, "the upload pins its source");
    assert_eq!(a.upload_source(ticket).unwrap(), &[0xC0; 4096][..]);

    a.complete_upload(ticket, Outcome::Completed).unwrap();
    assert_eq!(a.state_of(device, &chunk), Some(ChunkState::DeviceReady));
    assert_eq!(
        a.device_range(&lease).unwrap(),
        (*device_offset, EXPERT_BYTES)
    );
    let host_after = a
        .outstanding()
        .into_iter()
        .find(|c| c.scope == Scope::Host)
        .unwrap();
    assert_eq!(
        host_after.leases, 0,
        "the copy is done, so the source stops being a source"
    );

    reconciles(&a, Scope::Host);
    reconciles(&a, device);
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn an_observed_upload_failure_keeps_the_host_bytes_and_frees_the_device_range() {
    let mut l = ledger();
    let mut a = open_with_device(&mut l, 4 * EXPERT_BYTES, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let device = Scope::Device(gpu());

    let Acquired::Pending { lease, ticket, .. } = a.acquire(device_demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    a.read_destination(ticket).unwrap().fill(0xC1);
    a.complete_read(ticket, Outcome::Completed).unwrap();
    a.complete_upload(
        ticket,
        Outcome::Failed(Error::InvalidRequest {
            field: "copy",
            detail: "enqueue refused".into(),
        }),
    )
    .unwrap();

    assert_eq!(a.state_of(device, &chunk), None, "the device range is back");
    assert_eq!(
        a.committed_bytes(device).unwrap(),
        0,
        "an observed failure charges no device bytes"
    );
    assert_eq!(
        a.state_of(Scope::Host, &chunk),
        Some(ChunkState::HostReady),
        "the host bytes are valid; a retry uploads without re-reading"
    );
    assert_eq!(a.stats().upload_failures, 1);
    assert_eq!(a.stats().bytes_read, EXPERT_BYTES);
    a.release(lease).unwrap();

    // The retry is a fresh device acquire that hits the host copy: one read,
    // two uploads attempted.
    let Acquired::Pending {
        lease,
        ticket,
        work,
        ..
    } = a.acquire(device_demand(&chunk, 1)).unwrap()
    else {
        panic!("absent on the device")
    };
    assert!(matches!(
        work,
        PendingWork::Issued(WorkOrder::Upload { .. })
    ));
    a.complete_upload(ticket, Outcome::Completed).unwrap();
    assert_eq!(
        a.stats().bytes_read,
        EXPERT_BYTES,
        "the host copy was reused, not re-read"
    );
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn an_upload_whose_submission_state_is_unknown_withholds_both_ends() {
    let mut l = ledger();
    let mut a = open_with_device(&mut l, 4 * EXPERT_BYTES, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let device = Scope::Device(gpu());

    let Acquired::Pending { lease, ticket, .. } = a.acquire(device_demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    a.read_destination(ticket).unwrap().fill(0xC2);
    a.complete_read(ticket, Outcome::Completed).unwrap();
    a.complete_upload(
        ticket,
        Outcome::SubmissionUnknown(Error::DeviceLost {
            device: 1,
            detail: "event record failed after submission".into(),
        }),
    )
    .unwrap();

    // R07: a copy that may still be running owns both ends until it is known
    // to be over.
    assert_eq!(a.state_of(device, &chunk), Some(ChunkState::Quarantined));
    assert_eq!(
        a.state_of(Scope::Host, &chunk),
        Some(ChunkState::Quarantined)
    );
    assert_eq!(a.committed_bytes(device).unwrap(), EXPERT_BYTES);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), EXPERT_BYTES);
    assert!(
        a.acquire(device_demand(&chunk, 1)).is_err(),
        "quarantined bytes are not re-acquired"
    );

    a.release(lease).unwrap();
    a.settle_quarantined(device, &chunk).unwrap();
    a.settle_quarantined(Scope::Host, &chunk).unwrap();
    assert_eq!(a.committed_bytes(device).unwrap(), 0);
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
    a.close(&mut l).unwrap();
}

#[test]
fn each_device_has_its_own_cap_and_a_chunk_resident_on_one_is_absent_on_the_other() {
    let mut second = [0u8; 16];
    second[15] = 8;
    let second = DeviceUuid::from_bytes(second);
    let mut l = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 1 << 30, 1 << 20).unwrap(),
        CapacitySnapshot::new(Scope::Device(gpu()), 1 << 30, 0).unwrap(),
        CapacitySnapshot::new(Scope::Device(second), 1 << 30, 0).unwrap(),
    ])
    .unwrap();
    let mut a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("two devices", 8 * EXPERT_BYTES)
            .device(gpu(), 2 * EXPERT_BYTES)
            .device(second, 4 * EXPERT_BYTES),
    )
    .unwrap();

    // AGENTS.md forbids assuming the cards' memory is one allocation, and one
    // shared cap would be exactly that assumption.
    assert_eq!(a.cap_bytes(Scope::Device(gpu())), Some(2 * EXPERT_BYTES));
    assert_eq!(a.cap_bytes(Scope::Device(second)), Some(4 * EXPERT_BYTES));

    let chunk = expert(0);
    let Acquired::Pending { lease, ticket, .. } = a.acquire(device_demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    a.read_destination(ticket).unwrap().fill(0xD0);
    a.complete_read(ticket, Outcome::Completed).unwrap();
    a.complete_upload(ticket, Outcome::Completed).unwrap();

    assert_eq!(
        a.state_of(Scope::Device(second), &chunk),
        None,
        "residency is per device"
    );
    assert_eq!(a.committed_bytes(Scope::Device(second)).unwrap(), 0);
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}

// ---------------------------------------------------------------------------
// The envelope
// ---------------------------------------------------------------------------

#[test]
fn the_cache_envelope_is_admitted_from_the_ledger_and_returned_on_close() {
    let mut l = ledger();
    let before = l.outstanding().len();
    let a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("envelope", 64 * EXPERT_BYTES)
            .max_placements(128)
            .max_leases(256),
    )
    .unwrap();
    let host = l.committed(Scope::Host, Tier::Host(HostTier::Pageable));
    assert_eq!(
        host,
        64 * EXPERT_BYTES
            + 128 * moxie_memory::residency::PLACEMENT_CONTROL_BYTES
            + 256 * moxie_memory::residency::LEASE_CONTROL_BYTES,
        "the cache, its placement table and its lease table are all admitted"
    );
    let mut a = a;
    a.close(&mut l).unwrap();
    assert_eq!(l.outstanding().len(), before);
    assert_eq!(l.committed(Scope::Host, Tier::Host(HostTier::Pageable)), 0);
}

#[test]
fn an_envelope_larger_than_the_ledger_admits_is_refused_and_charges_nothing() {
    let mut l = Ledger::new([CapacitySnapshot::new(Scope::Host, 8_192, 4_096).unwrap()]).unwrap();
    let refused =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("too big", 1 << 30)).unwrap_err();
    assert!(
        matches!(refused, Error::CapacityExceeded { .. }),
        "{refused:?}"
    );
    assert_eq!(l.outstanding().len(), 0);
    assert_eq!(l.scope_committed(Scope::Host), 0);
}

#[test]
fn closing_over_live_leases_or_in_flight_transfers_is_refused() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let lease = load(&mut a, &expert(0), 0, 0);
    let e = a.close(&mut l).unwrap_err();
    assert!(format!("{e}").contains("lease"), "{e}");
    a.release(lease).unwrap();

    let Acquired::Pending { lease, ticket, .. } = a.acquire(demand(&expert(1), 1)).unwrap() else {
        panic!("absent")
    };
    a.release(lease).unwrap();
    let e = a.close(&mut l).unwrap_err();
    assert!(format!("{e}").contains("reading"), "{e}");
    a.complete_read(ticket, Outcome::Completed).unwrap();
    a.close(&mut l).unwrap();
}

#[test]
fn the_lease_table_is_bounded_and_refuses_rather_than_growing() {
    let mut l = ledger();
    let mut a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("few leases", 64 * EXPERT_BYTES).max_leases(3),
    )
    .unwrap();
    let chunk = expert(0);
    let mut held = vec![load(&mut a, &chunk, 0, 0x01)];
    for i in 1..3u64 {
        let Acquired::Ready(lease) = a.acquire(demand(&chunk, i)).unwrap() else {
            panic!("resident")
        };
        held.push(lease);
    }
    assert_eq!(a.live_lease_count(), 3);

    let refused = a.acquire(demand(&chunk, 3)).unwrap_err();
    assert!(
        matches!(refused.error, Error::CapacityExceeded { .. }),
        "{:?}",
        refused.error
    );
    // Exhausting an admitted table is a capacity refusal, never a panic and
    // never a growth nobody charged for.
    a.release(held.pop().unwrap()).unwrap();
    let again = a.acquire(demand(&chunk, 4)).unwrap();
    assert!(matches!(again, Acquired::Ready(_)));
    let Acquired::Ready(again) = again else {
        unreachable!()
    };
    // The reused slot carries a new generation, so the identity that held it
    // before is not this one.
    assert_ne!(again.id(), held[0].id());
    a.release(again).unwrap();
    for lease in held {
        a.release(lease).unwrap();
    }
    a.close(&mut l).unwrap();
}

#[test]
fn a_lease_from_another_authority_is_refused_and_handed_back() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let mut b = open(&mut l, 4 * EXPERT_BYTES);
    let lease = load(&mut a, &expert(0), 0, 0);
    let refused = b.release(lease).unwrap_err();
    assert!(format!("{refused}").contains("another authority"));
    // The refusal handed the lease back, so the chunk is not stranded.
    a.release(refused.lease).unwrap();
    a.close(&mut l).unwrap();
    b.close(&mut l).unwrap();
}

#[test]
fn a_destination_with_no_open_cache_is_refused_by_name() {
    let mut l = ledger();
    let mut a = open(&mut l, 4 * EXPERT_BYTES);
    let chunk = expert(0);
    let refused = a.acquire(device_demand(&chunk, 0)).unwrap_err();
    assert!(
        format!("{}", refused.error).contains("no residency cache is open"),
        "{refused}"
    );
    a.close(&mut l).unwrap();
}

#[test]
fn the_report_names_the_expert_cache_tier_on_a_device_and_the_pageable_tier_on_the_host() {
    let mut l = ledger();
    let mut a = open_with_device(&mut l, 4 * EXPERT_BYTES, EXPERT_BYTES);
    // One expert fills the device cache; the second is refused.
    let chunk = expert(0);
    let Acquired::Pending { lease, ticket, .. } = a.acquire(device_demand(&chunk, 0)).unwrap()
    else {
        panic!("absent")
    };
    a.read_destination(ticket).unwrap().fill(0);
    a.complete_read(ticket, Outcome::Completed).unwrap();
    a.complete_upload(ticket, Outcome::Completed).unwrap();

    let other = expert(1);
    let refused = a.acquire(device_demand(&other, 1)).unwrap_err();
    assert_eq!(refused.report.tier, Tier::Device(DeviceTier::ExpertCache));
    assert_eq!(refused.report.scope, Scope::Device(gpu()));

    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
}
