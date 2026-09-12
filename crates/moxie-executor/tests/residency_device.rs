//! Real-device proof for task 0020's upload path: the authority's ranges,
//! backed by one real allocation per device, filled by real copies whose
//! readiness is a real event.
//!
//! This is the half a host lane cannot argue. The authority's rule is that a
//! device chunk is readable only after its completion is *observed* -- document
//! 02: "An upload owns or leases its source bytes through a completion event. A
//! scratch source cannot be overwritten merely because the enqueue function
//! returned" -- and the only way to show that holds is to enqueue a real copy
//! and read the bytes back off the card.
//!
//! No model, no kernel, no checkpoint. Weight-shaped bytes the test invents.
#![cfg(feature = "driver")]

use moxie_cuda::{RankContext, Stream};
use moxie_executor::residency::{DeviceResidency, ShardSource, drain_reads};
use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, ChunkState, Content, Ledger,
    LogicalRange, ResidencyAuthority, ResidencyRequest, TensorSlot, TurnId, UseClass, WorkOrder,
};
use moxie_storage::Shard;
use moxie_types::{DeviceTier, RankId, Scope, Tier};

use std::io::Write;
use std::path::{Path, PathBuf};

const MIB: u64 = 1024 * 1024;
/// Four experts of a megabyte each: small enough to run on every card here,
/// large enough that a copy is a copy.
const EXPERT_BYTES: u64 = MIB;
const EXPERTS: u32 = 8;
const DEVICE_CAP: u64 = 2 * EXPERT_BYTES;

fn artifact() -> ArtifactId {
    ArtifactId::new("device-fixture-v1").unwrap()
}

fn chunk(expert: u32) -> ChunkId {
    ChunkId::new(
        artifact(),
        TensorSlot::expert("experts_gate_up", expert).unwrap(),
        LogicalRange::new(u64::from(expert) * EXPERT_BYTES, EXPERT_BYTES).unwrap(),
        1,
    )
}

fn write_fused_shard(dir: &Path) -> PathBuf {
    let total = u64::from(EXPERTS) * EXPERT_BYTES;
    let rows = EXPERT_BYTES / 2;
    let header = format!(
        "{{\"experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{EXPERTS},{rows}],\
         \"data_offsets\":[0,{total}]}}}}"
    );
    let path = dir.join("model.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for e in 0..EXPERTS {
        f.write_all(&vec![(0x40 + e) as u8; EXPERT_BYTES as usize])
            .unwrap();
    }
    f.flush().unwrap();
    path
}

fn source(path: &Path) -> ShardSource {
    ShardSource::new(artifact(), vec![Shard::open(path).unwrap()])
        .role("experts_gate_up", 0, "experts.gate_up_proj")
        .unwrap()
}

fn request(chunk: &ChunkId, scope: Scope, now: u64) -> AcquireRequest<'_> {
    AcquireRequest {
        chunk,
        destination: scope,
        now,
        deadline: u64::MAX,
        class: UseClass::demand(Content::Expert),
        turn: TurnId::new(1),
    }
}

#[test]
fn a_device_chunk_is_readable_only_after_its_copy_is_observed_on_every_device() {
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "device lane requires real hardware");

    let dir = std::env::temp_dir().join(format!("moxie-residency-device-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = write_fused_shard(&dir);

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let stream = Stream::new(&ctx).unwrap();
        let scope = Scope::Device(ctx.uuid());
        let mut src = source(&path);

        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 256 * MIB, 16 * MIB).unwrap(),
            CapacitySnapshot::new(scope, 256 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new(format!("residency on {}", ctx.uuid()), 4 * EXPERT_BYTES)
                .device(ctx.uuid(), DEVICE_CAP),
        )
        .unwrap();
        assert_eq!(
            ledger.committed(scope, Tier::Device(DeviceTier::ExpertCache)),
            DEVICE_CAP,
            "the device cache is admitted before a byte of it exists"
        );

        let before = ctx.memory_info().unwrap().0;
        let mut device = DeviceResidency::create(&ctx, &authority).unwrap();
        let during = ctx.memory_info().unwrap().0;
        assert!(
            during < before,
            "the device cache must spend real device memory on {}",
            ctx.uuid()
        );
        assert_eq!(device.capacity(), DEVICE_CAP);

        // Two experts, filling the cache exactly.
        let mut leases = Vec::new();
        for expert in 0..2u32 {
            let id = chunk(expert);
            let Acquired::Pending { lease, work, .. } = authority
                .acquire(request(&id, scope, u64::from(expert)))
                .unwrap()
            else {
                panic!("absent on {}", ctx.uuid())
            };
            // The read runs first; the upload comes back as the second stage of
            // the same ticket.
            let uploads = drain_reads(&mut authority, &mut src, work).unwrap();
            assert_eq!(uploads.len(), 1);
            let WorkOrder::Upload { ticket, .. } = &uploads[0] else {
                panic!("the second stage is an upload")
            };

            // Before the copy is observed, the bytes are not readable and the
            // host source is pinned.
            assert_eq!(authority.state_of(scope, &id), Some(ChunkState::Uploading));
            assert!(authority.device_range(&lease).is_err());
            let host_pin = authority
                .outstanding()
                .into_iter()
                .find(|c| c.scope == Scope::Host && c.chunk == id)
                .unwrap();
            assert_eq!(host_pin.leases, 1, "the upload leases its source");
            assert_eq!(
                authority.upload_source(*ticket).unwrap().len() as u64,
                EXPERT_BYTES
            );

            device
                .perform_upload(&mut authority, &stream, &uploads[0])
                .unwrap();

            assert_eq!(
                authority.state_of(scope, &id),
                Some(ChunkState::DeviceReady)
            );
            let (offset, len) = authority.device_range(&lease).unwrap();
            assert_eq!(len, EXPERT_BYTES);

            // The oracle: what is on the card equals what the file holds.
            let mut arrived = vec![0u8; len as usize];
            device.read_back(offset, &mut arrived).unwrap();
            assert!(
                arrived.iter().all(|b| *b == 0x40 + expert as u8),
                "expert {expert} arrived wrong on {}",
                ctx.uuid()
            );
            leases.push(lease);
        }
        assert_eq!(authority.committed_bytes(scope).unwrap(), DEVICE_CAP);
        assert_eq!(authority.stats().bytes_uploaded, 2 * EXPERT_BYTES);

        // The cache is full and both entries are leased: a third demand is
        // refused immediately, with the report saying why. This is M2's
        // "demonstrate demand failure cannot deadlock", on real hardware.
        let third = chunk(2);
        let refused = authority.acquire(request(&third, scope, 10)).unwrap_err();
        assert_eq!(refused.report.leased_bytes, DEVICE_CAP);
        assert_eq!(refused.report.evictable_bytes, 0);
        assert_eq!(refused.report.incoming_bytes, EXPERT_BYTES);
        assert_eq!(refused.report.scope, scope);
        assert_eq!(refused.report.tier, Tier::Device(DeviceTier::ExpertCache));

        // Release one, and the same request now evicts and succeeds, serving
        // its own bytes off the card.
        authority.release(leases.remove(0)).unwrap();
        let Acquired::Pending { lease, work, .. } =
            authority.acquire(request(&third, scope, 11)).unwrap()
        else {
            panic!("absent")
        };
        let uploads = drain_reads(&mut authority, &mut src, work).unwrap();
        assert_eq!(uploads.len(), 1);
        device
            .perform_upload(&mut authority, &stream, &uploads[0])
            .unwrap();
        let (offset, len) = authority.device_range(&lease).unwrap();
        let mut arrived = vec![0u8; len as usize];
        device.read_back(offset, &mut arrived).unwrap();
        assert!(arrived.iter().all(|b| *b == 0x42));
        assert!(authority.stats().evictions > 0);
        leases.push(lease);

        // A turn that ends with no next token releases every lease, and the
        // charge comes back to exactly nothing after the caches close. R08.
        for lease in leases.drain(..) {
            std::mem::drop(lease);
        }
        let cleanup = authority.end_turn(TurnId::new(1));
        assert!(!cleanup.released_leases.is_empty());
        assert!(cleanup.still_in_flight.is_empty());
        assert_eq!(authority.live_lease_count(), 0);

        let uploaded = authority.stats().bytes_uploaded;
        let evictions = authority.stats().evictions;
        drop(device);
        authority.close(&mut ledger).unwrap();
        assert_eq!(ledger.scope_committed(scope), 0);
        assert_eq!(ledger.scope_committed(Scope::Host), 0);
        println!(
            "{}: {uploaded} B uploaded into a {DEVICE_CAP} B admitted expert cache,              {evictions} eviction(s), every range verified by readback",
            ctx.uuid()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
