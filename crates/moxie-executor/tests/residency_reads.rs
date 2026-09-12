//! Acceptance tests for task 0020's driver: the authority's work orders,
//! performed as real bounded reads.
//!
//! The oracle here is **byte identity against an independent reader**. Every
//! chunk the cache serves -- on a hit, on a reload after eviction, and to a
//! coalesced waiter -- must equal the same range read straight from the file
//! with no cache in the way. A cache whose hits differ from its misses is the
//! defect this checks for, and nothing else it does matters if that is false.
//!
//! The last test in this file reads the designated artifact. It is skipped,
//! loudly, when that artifact is not on this machine.

use std::io::Write;
use std::path::{Path, PathBuf};

use moxie_executor::residency::{ChunkSource, ShardSource, drain_reads, perform_read};
use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger, LogicalRange,
    PendingWork, ResidencyAuthority, ResidencyRequest, TensorSlot, TurnId, UseClass,
    WorkOrder,
};
use moxie_storage::Shard;
use moxie_types::{Error, Result, Scope};

// ---------------------------------------------------------------------------
// A safetensors shard written by the test, so the source is real
// ---------------------------------------------------------------------------

/// Write a minimal safetensors file with one fused BF16 expert tensor.
///
/// Real bytes in a real file, read through the real reader. A hand-rolled
/// in-memory double would test the authority against a fiction; the point of
/// this file is the other half.
fn write_fused_shard(dir: &Path, experts: u32, bytes_per_expert: u64) -> PathBuf {
    let total = u64::from(experts) * bytes_per_expert;
    let rows = bytes_per_expert / 2;
    let header = format!(
        "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{experts},{rows}],\
         \"data_offsets\":[0,{total}]}}}}"
    );
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    // Every expert's bytes are its own index, so a chunk served from the wrong
    // place is visible rather than plausible.
    for e in 0..experts {
        let pattern = vec![(e & 0x7F) as u8; bytes_per_expert as usize];
        f.write_all(&pattern).unwrap();
    }
    f.flush().unwrap();
    path
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("moxie-residency-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const EXPERT_BYTES: u64 = 2_048;
const EXPERTS: u32 = 8;

fn artifact() -> ArtifactId {
    ArtifactId::new("fused-fixture-v1").unwrap()
}

fn chunk(expert: u32) -> ChunkId {
    ChunkId::new(
        artifact(),
        TensorSlot::expert("experts_gate_up", expert).unwrap(),
        LogicalRange::new(u64::from(expert) * EXPERT_BYTES, EXPERT_BYTES).unwrap(),
        1,
    )
}

fn source(path: &Path) -> ShardSource {
    ShardSource::new(artifact(), vec![Shard::open(path).unwrap()])
        .role("experts_gate_up", 0, "experts.gate_up_proj")
        .unwrap()
}

fn ledger() -> Ledger {
    Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 30, 1 << 20).unwrap()]).unwrap()
}

fn request(chunk: &ChunkId, now: u64) -> AcquireRequest<'_> {
    AcquireRequest {
        chunk,
        destination: Scope::Host,
        now,
        deadline: u64::MAX,
        class: UseClass::demand(Content::Expert),
        turn: TurnId::new(1),
    }
}

/// Read one chunk's range straight from the file, with no cache involved.
fn independently(path: &Path, id: &ChunkId) -> Vec<u8> {
    let shard = Shard::open(path).unwrap();
    let mut out = vec![0u8; id.len_bytes() as usize];
    shard
        .read_tensor_range("experts.gate_up_proj", id.range().offset_bytes(), &mut out)
        .unwrap();
    out
}

// ---------------------------------------------------------------------------
// The oracle
// ---------------------------------------------------------------------------

#[test]
fn served_bytes_equal_an_independent_read_on_hits_misses_and_reloads() {
    let dir = scratch("oracle");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let mut src = source(&path);
    let mut l = ledger();
    // Two experts fit, so reloads actually happen.
    let mut a =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("oracle", 2 * EXPERT_BYTES))
            .unwrap();

    // A miss, a hit, then an eviction and a reload -- each checked against the
    // file rather than against the last thing the cache said.
    for (tick, expert) in [0u32, 0, 1, 2, 0, 1].iter().copied().enumerate() {
        let id = chunk(expert);
        let expected = independently(&path, &id);
        let acquired = a.acquire(request(&id, tick as u64)).unwrap();
        let lease = match acquired {
            Acquired::Ready(lease) => lease,
            Acquired::Pending { lease, work, .. } => {
                assert!(drain_reads(&mut a, &mut src, work).unwrap().is_empty());
                lease
            }
        };
        assert_eq!(
            a.chunk_bytes(&lease).unwrap(),
            &expected[..],
            "expert {expert} at tick {tick} was not served its own bytes"
        );
        assert_eq!(expected[0], (expert & 0x7F) as u8);
        a.release(lease).unwrap();
    }

    // Six acquires over three distinct experts in a two-expert cache: some of
    // them missed and some hit, and every one of them was correct.
    assert!(a.stats().hits > 0 && a.stats().misses > 0);
    assert!(a.stats().evictions > 0);
    a.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_coalesced_waiter_is_served_the_same_bytes_from_the_one_read() {
    let dir = scratch("coalesce");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let mut src = source(&path);
    let mut l = ledger();
    let mut a =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("coalesce", 4 * EXPERT_BYTES))
            .unwrap();

    let id = chunk(5);
    let Acquired::Pending {
        lease: first, work, ..
    } = a.acquire(request(&id, 0)).unwrap()
    else {
        panic!("absent")
    };
    let Acquired::Pending { lease: second, .. } = a.acquire(request(&id, 1)).unwrap() else {
        panic!("pending")
    };
    drain_reads(&mut a, &mut src, work).unwrap();

    let expected = independently(&path, &id);
    assert_eq!(a.chunk_bytes(&first).unwrap(), &expected[..]);
    assert_eq!(a.chunk_bytes(&second).unwrap(), &expected[..]);
    assert_eq!(
        a.stats().bytes_read,
        EXPERT_BYTES,
        "two acquires, one chunk's bytes read once"
    );
    a.release(first).unwrap();
    a.release(second).unwrap();
    a.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_ranged_read_reads_the_range_and_not_the_tensor() {
    let dir = scratch("bounded");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let shard = Shard::open(&path).unwrap();

    // The whole fused tensor is eight experts; one expert is an eighth of it.
    let mut one = vec![0u8; EXPERT_BYTES as usize];
    shard
        .read_tensor_range("experts.gate_up_proj", 3 * EXPERT_BYTES, &mut one)
        .unwrap();
    assert!(one.iter().all(|b| *b == 3));

    // Out of range is a refusal naming both sizes, never a read that wanders
    // into whatever follows.
    let mut past = vec![0u8; EXPERT_BYTES as usize];
    let e = shard
        .read_tensor_range(
            "experts.gate_up_proj",
            u64::from(EXPERTS) * EXPERT_BYTES,
            &mut past,
        )
        .unwrap_err();
    assert!(format!("{e}").contains("exceeds its"), "{e}");

    // A zero-length range is not a read.
    let mut empty: [u8; 0] = [];
    assert!(
        shard
            .read_tensor_range("experts.gate_up_proj", 0, &mut empty)
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Failure injection: a source that fails part way through
// ---------------------------------------------------------------------------

/// A source that fails after serving `ok_before` chunks. No real filesystem
/// will fail on request, and the authority's failure path is the half that
/// matters most.
#[derive(Debug)]
struct FailingSource {
    inner: ShardSource,
    ok_before: usize,
    served: usize,
}

impl ChunkSource for FailingSource {
    fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> Result<()> {
        if self.served >= self.ok_before {
            // Half-filled, exactly as a truncated read leaves a buffer. The
            // authority must not serve these bytes to anyone.
            let half = into.len() / 2;
            into[..half].fill(0xFF);
            return Err(Error::InvalidArtifact {
                detail: format!("short read of chunk '{}': truncation", chunk.slot().role()),
            });
        }
        self.served += 1;
        self.inner.read_chunk(chunk, into)
    }
}

#[test]
fn a_read_that_fails_part_way_serves_nobody_and_leaves_the_cache_usable() {
    let dir = scratch("failing");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let mut src = FailingSource {
        inner: source(&path),
        ok_before: 1,
        served: 0,
    };
    let mut l = ledger();
    let mut a =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("failing", 4 * EXPERT_BYTES))
            .unwrap();

    // The first read succeeds.
    let good = chunk(0);
    let Acquired::Pending { lease, work, .. } = a.acquire(request(&good, 0)).unwrap() else {
        panic!("absent")
    };
    drain_reads(&mut a, &mut src, work).unwrap();
    let expected = independently(&path, &good);
    assert_eq!(a.chunk_bytes(&lease).unwrap(), &expected[..]);
    a.release(lease).unwrap();

    // The second fails part way, with two waiters on it.
    let bad = chunk(1);
    let Acquired::Pending {
        lease: first, work, ..
    } = a.acquire(request(&bad, 1)).unwrap()
    else {
        panic!("absent")
    };
    let Acquired::Pending { lease: second, .. } = a.acquire(request(&bad, 2)).unwrap() else {
        panic!("pending")
    };
    let error = drain_reads(&mut a, &mut src, work).unwrap_err();
    assert!(format!("{error}").contains("truncation"), "{error}");

    // Both waiters see the same thing, and it is not the half-written buffer.
    for lease in [&first, &second] {
        assert!(a.chunk_bytes(lease).is_err());
    }
    assert_eq!(a.state_of(Scope::Host, &bad), None);
    assert_eq!(
        a.committed_bytes(Scope::Host).unwrap(),
        EXPERT_BYTES,
        "only the chunk that actually arrived is charged"
    );
    a.release(first).unwrap();
    a.release(second).unwrap();

    // Still usable. A retry is a fresh acquire, and this source now succeeds
    // again because the test moved its failure point.
    src.ok_before = 10;
    let Acquired::Pending { lease, work, .. } = a.acquire(request(&bad, 3)).unwrap() else {
        panic!("absent")
    };
    drain_reads(&mut a, &mut src, work).unwrap();
    assert_eq!(
        a.chunk_bytes(&lease).unwrap(),
        &independently(&path, &bad)[..]
    );
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_chunk_the_source_cannot_name_is_refused_rather_than_guessed() {
    let dir = scratch("missing");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let mut src = source(&path);
    let mut l = ledger();
    let mut a =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("missing", 4 * EXPERT_BYTES))
            .unwrap();

    // M2 item 5's "missing expert route": the demand names a role this source
    // has no tensor for.
    let unknown = ChunkId::new(
        artifact(),
        TensorSlot::expert("experts_down", 0).unwrap(),
        LogicalRange::new(0, EXPERT_BYTES).unwrap(),
        1,
    );
    let Acquired::Pending { lease, work, .. } = a.acquire(request(&unknown, 0)).unwrap() else {
        panic!("absent")
    };
    let error = drain_reads(&mut a, &mut src, work).unwrap_err();
    assert!(
        format!("{error}").contains("experts_down"),
        "the refusal must name the chunk: {error}"
    );
    assert_eq!(a.committed_bytes(Scope::Host).unwrap(), 0);
    a.release(lease).unwrap();

    // And a chunk belonging to a different artifact is refused by the source,
    // not silently served from this one.
    let foreign = ChunkId::new(
        ArtifactId::new("some-other-artifact").unwrap(),
        TensorSlot::expert("experts_gate_up", 0).unwrap(),
        LogicalRange::new(0, EXPERT_BYTES).unwrap(),
        1,
    );
    let Acquired::Pending { lease, work, .. } = a.acquire(request(&foreign, 1)).unwrap() else {
        panic!("absent")
    };
    let error = drain_reads(&mut a, &mut src, work).unwrap_err();
    assert!(
        format!("{error}").contains("some-other-artifact"),
        "{error}"
    );
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_driver_performs_no_second_read_for_a_coalesced_acquire() {
    let dir = scratch("no-second");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let mut src = FailingSource {
        inner: source(&path),
        // Exactly one read is permitted. A second would fail the test by
        // failing the read, which is a louder signal than a counter.
        ok_before: 1,
        served: 0,
    };
    let mut l = ledger();
    let mut a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("no-second", 4 * EXPERT_BYTES),
    )
    .unwrap();

    let id = chunk(2);
    let Acquired::Pending {
        lease: first, work, ..
    } = a.acquire(request(&id, 0)).unwrap()
    else {
        panic!("absent")
    };
    let Acquired::Pending {
        lease: second,
        work: second_work,
        ..
    } = a.acquire(request(&id, 1)).unwrap()
    else {
        panic!("pending")
    };
    assert_eq!(second_work, PendingWork::Coalesced);
    // Draining the waiter's work must perform nothing at all.
    assert!(
        drain_reads(&mut a, &mut src, second_work)
            .unwrap()
            .is_empty()
    );
    assert_eq!(src.served, 0);

    drain_reads(&mut a, &mut src, work).unwrap();
    assert_eq!(src.served, 1);
    a.release(first).unwrap();
    a.release(second).unwrap();
    a.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_order_of_the_wrong_kind_is_refused_by_the_driver() {
    let dir = scratch("wrong-order");
    let path = write_fused_shard(&dir, EXPERTS, EXPERT_BYTES);
    let mut src = source(&path);
    let mut l = ledger();
    let mut a = ResidencyAuthority::open(
        &mut l,
        &ResidencyRequest::new("wrong-order", 4 * EXPERT_BYTES),
    )
    .unwrap();
    let id = chunk(0);
    let Acquired::Pending { lease, ticket, .. } = a.acquire(request(&id, 0)).unwrap() else {
        panic!("absent")
    };
    let fabricated = WorkOrder::Upload {
        ticket,
        chunk: id.clone(),
        scope: Scope::Host,
        host_offset: 0,
        device_offset: 0,
        len_bytes: EXPERT_BYTES,
    };
    let e = perform_read(&mut a, &mut src, &fabricated).unwrap_err();
    assert!(format!("{e}").contains("not a read order"), "{e}");

    let real = WorkOrder::Read {
        ticket,
        chunk: id.clone(),
        host_offset: 0,
        len_bytes: EXPERT_BYTES,
    };
    perform_read(&mut a, &mut src, &real).unwrap();
    a.release(lease).unwrap();
    a.close(&mut l).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// The designated artifact, read-only, bounded, and nothing computed with it
// ---------------------------------------------------------------------------

/// The designated BF16 MoE. A read-only input; nothing here writes, copies,
/// converts or executes anything.
const ARTIFACT: &str = "/fast/models/google/gemma-4-26B-A4B-it";

/// Layer 0's fused expert tensors, and one expert's slice of each.
/// `[128, 1408, 2816]` and `[128, 2816, 704]` in BF16, read from the artifact's
/// own safetensors header on 2026-09-12.
const GATE_UP: &str = "model.language_model.layers.0.experts.gate_up_proj";
const DOWN: &str = "model.language_model.layers.0.experts.down_proj";
const GATE_UP_PER_EXPERT: u64 = 1408 * 2816 * 2;
const DOWN_PER_EXPERT: u64 = 2816 * 704 * 2;
const EXPERT_TOTAL: u64 = GATE_UP_PER_EXPERT + DOWN_PER_EXPERT;

#[test]
fn the_designated_artifacts_experts_are_demand_loaded_through_the_authority() {
    let dir = Path::new(ARTIFACT);
    if !dir.is_dir() {
        // Reported as skipped, with its reason, never as passed.
        println!(
            "SKIPPED: {ARTIFACT} is not present on this machine; the residency path is \
             exercised against the synthetic fused shard above"
        );
        return;
    }
    let shard_path = dir.join("model-00001-of-00002.safetensors");
    assert!(shard_path.is_file(), "{} is missing", shard_path.display());

    // The identity is the artifact's own index digest, recorded in the bring-up
    // record. It is not a path: the authority may not learn what one is.
    let artifact =
        ArtifactId::new("sha256:907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7")
            .unwrap();
    let shard = Shard::open(&shard_path).unwrap();
    assert_eq!(
        shard.header().get(GATE_UP).unwrap().len(),
        128 * GATE_UP_PER_EXPERT
    );
    assert_eq!(
        shard.header().get(DOWN).unwrap().len(),
        128 * DOWN_PER_EXPERT
    );
    let mut src = ShardSource::new(artifact.clone(), vec![shard])
        .role("experts_gate_up", 0, GATE_UP)
        .unwrap()
        .role("experts_down", 0, DOWN)
        .unwrap();

    // A demand set from routes a real router could produce: eight experts,
    // overlapping across rows, exactly as top-k 8 gives.
    let roles = vec![
        moxie_engine::ExpertRole::new("experts_gate_up", GATE_UP_PER_EXPERT).unwrap(),
        moxie_engine::ExpertRole::new("experts_down", DOWN_PER_EXPERT).unwrap(),
    ];
    let routes: Vec<&[u32]> = vec![
        &[3, 17, 42, 91, 5, 60, 118, 7],
        &[3, 17, 42, 91, 5, 60, 118, 7],
        &[3, 17, 42, 99, 5, 60, 118, 7],
    ];
    let demand = moxie_engine::expert_demand(&artifact, &roles, &routes, 128, 1).unwrap();
    // Nine distinct experts across three rows of eight, not twenty-four.
    assert_eq!(demand.len(), 18);
    assert_eq!(moxie_engine::demand_bytes(&demand), 9 * EXPERT_TOTAL);
    assert_eq!(EXPERT_TOTAL, 11_894_784);

    // A cache deliberately smaller than the demand, so eviction actually runs.
    let cap = 4 * EXPERT_TOTAL;
    let mut l =
        Ledger::new([CapacitySnapshot::new(Scope::Host, 4 << 30, 1 << 30).unwrap()]).unwrap();
    let mut a =
        ResidencyAuthority::open(&mut l, &ResidencyRequest::new("gemma4-moe", cap)).unwrap();

    let mut served = 0u64;
    for (tick, id) in demand.iter().enumerate() {
        let acquired = a.acquire(request(id, tick as u64)).unwrap();
        let lease = match acquired {
            Acquired::Ready(lease) => lease,
            Acquired::Pending { lease, work, .. } => {
                assert!(drain_reads(&mut a, &mut src, work).unwrap().is_empty());
                lease
            }
        };
        let bytes = a.chunk_bytes(&lease).unwrap();
        assert_eq!(bytes.len() as u64, id.len_bytes());
        served += bytes.len() as u64;
        a.release(lease).unwrap();
        assert!(a.committed_bytes(Scope::Host).unwrap() <= cap);
    }

    // Exactly the union's bytes were read, once each -- not the 1,522,532,352
    // of the two fused tensors, and not `rows x top_k` copies of anything.
    assert_eq!(served, 9 * EXPERT_TOTAL);
    assert_eq!(a.stats().bytes_read, 9 * EXPERT_TOTAL);
    assert_eq!(a.stats().misses, 18);
    assert_eq!(a.stats().hits, 0);
    assert!(
        a.stats().evictions > 0,
        "a cache of four experts serving nine must have displaced something"
    );

    // The oracle: one of the served ranges, re-read independently.
    let probe = &demand[0];
    let mut expected = vec![0u8; probe.len_bytes() as usize];
    Shard::open(&shard_path)
        .unwrap()
        .read_tensor_range(GATE_UP, probe.range().offset_bytes(), &mut expected)
        .unwrap();
    let acquired = a.acquire(request(probe, 1_000)).unwrap();
    let lease = match acquired {
        Acquired::Ready(lease) => lease,
        Acquired::Pending { lease, work, .. } => {
            drain_reads(&mut a, &mut src, work).unwrap();
            lease
        }
    };
    assert_eq!(a.chunk_bytes(&lease).unwrap(), &expected[..]);
    a.release(lease).unwrap();

    a.close(&mut l).unwrap();
    assert_eq!(l.scope_committed(Scope::Host), 0);

    println!(
        "read {served} B of real expert weights through the residency authority \
         ({} experts, {cap} B cache); nothing was computed with them and nothing was written",
        9
    );
}
