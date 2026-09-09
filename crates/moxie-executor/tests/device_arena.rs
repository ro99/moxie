//! Real-device proof for the admitted suballocator. No model or kernel.
#![cfg(feature = "driver")]

use moxie_cuda::{Event, RankContext, Stream};
use moxie_executor::{ArenaUpload, DeviceArena, DeviceRange, OperationLease};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, Reservation, StageSpan};
use moxie_types::{DeviceTier, HostTier, RankId, Scope, Tier};

const MIB: u64 = 1024 * 1024;
const ARENA_BYTES: u64 = 16 * MIB;

fn admitted(ctx: &RankContext) -> (Ledger, Reservation) {
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Device(ctx.uuid()), 64 * MIB, MIB).unwrap(),
        CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
    ])
    .unwrap();
    let mut request = PlanRequest::new("device arena", ["resident"]).unwrap();
    request
        .buffer(BufferRequest::new(
            "physical weights",
            Scope::Device(ctx.uuid()),
            Tier::Device(DeviceTier::PackedResidentWeights),
            ARENA_BYTES,
            StageSpan { first: 0, last: 0 },
        ))
        .unwrap();
    request
        .buffer(BufferRequest::new(
            "one retained upload source",
            Scope::Host,
            Tier::Host(HostTier::Pageable),
            8 * MIB,
            StageSpan { first: 0, last: 0 },
        ))
        .unwrap();
    let reservation = ledger.admit(&request).unwrap();
    (ledger, reservation)
}

fn finish_upload<'ctx>(
    mut lease: OperationLease<Event<'ctx>, ArenaUpload<'ctx>>,
    expected: &[u8],
) -> DeviceRange<'ctx> {
    let mut readback = vec![0; expected.len()];
    lease.readback(&mut readback).unwrap();
    assert_eq!(readback, expected);
    let (_, upload) = lease.retire().unwrap();
    let (range, source) = upload.finish();
    assert_eq!(source, expected);
    range
}

#[test]
fn admitted_device_arena_reuses_only_completed_ranges_on_every_device() {
    let count = moxie_cuda::device_count().unwrap();
    assert!(count > 0, "device lane requires real hardware");
    let mut observed_in_flight = false;

    for ordinal in 0..count {
        let ctx = RankContext::acquire(RankId(ordinal), ordinal).unwrap();
        let stream = Stream::new(&ctx).unwrap();
        let before = ctx.memory_info().unwrap().0;
        let (mut ledger, reservation) = admitted(&ctx);
        let mut arena = DeviceArena::create(
            &ledger,
            reservation,
            &ctx,
            DeviceTier::PackedResidentWeights,
            ARENA_BYTES,
            format!("weights on {}", ctx.uuid()),
        )
        .unwrap();
        let during = ctx.memory_info().unwrap().0;
        assert!(during < before, "physical arena must spend device memory");

        let first = arena.allocate(8 * MIB, 256, "importer").unwrap();
        let mut foreign_ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Device(ctx.uuid()), 64 * MIB, MIB).unwrap(),
            CapacitySnapshot::new(Scope::Host, 64 * MIB, MIB).unwrap(),
        ])
        .unwrap();
        let refused = arena.close(&mut foreign_ledger).unwrap_err();
        assert_eq!(refused.error.kind(), "invalid_request");
        arena = refused.arena;
        let refused = arena.close(&mut ledger).unwrap_err();
        assert!(refused.error.to_string().contains("remain live"));
        arena = refused.arena;
        let first_generation = first.key().generation;
        let second = arena.allocate(3 * MIB, 128, "workspace").unwrap();
        let third = arena.allocate(5 * MIB, 64, "persistent").unwrap();
        let refused = arena.allocate(256, 256, "overflow").unwrap_err();
        assert_eq!(refused.occupancy.free_bytes, 0);
        assert_eq!(refused.occupancy.live_allocations, 3);

        let first_source = vec![0x19; 8 * MIB as usize];
        let mut first_use = first
            .prepare_upload(first_source.clone(), "first range upload")
            .unwrap();
        first_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        let first = match first_use.retire() {
            Ok((_, upload)) => upload.finish().0,
            Err(mut held) => {
                observed_in_flight = true;
                assert!(held.error.to_string().contains("not observed"));
                held.lease.synchronize().unwrap();
                held.lease.retire().unwrap().1.finish().0
            }
        };
        // A completed upload is safe to use again; verify the exact offset bytes.
        let mut first_use = first
            .prepare_upload(first_source.clone(), "first range readback")
            .unwrap();
        first_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        let first = finish_upload(first_use, &first_source);
        let first_key = first.key();
        let first = arena.transfer(first, "executor").unwrap();
        assert_eq!(first.key(), first_key);
        assert_eq!(first.owner(), "executor");

        let second_source = vec![0x2a; 3 * MIB as usize];
        let mut second_use = second
            .prepare_upload(second_source.clone(), "cancelled range upload")
            .unwrap();
        second_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        second_use.cancel();
        let second = finish_upload(second_use, &second_source);

        let third_source = vec![0x3b; 5 * MIB as usize];
        let mut third_use = third
            .prepare_upload(third_source.clone(), "persistent range upload")
            .unwrap();
        third_use
            .submit(&stream, Event::new(&ctx).unwrap())
            .unwrap();
        let third = finish_upload(third_use, &third_source);

        arena.release(second).unwrap();
        arena.release(first).unwrap();
        arena.release(third).unwrap();
        assert_eq!(arena.occupancy().largest_free_bytes, ARENA_BYTES);
        let whole = arena.allocate(ARENA_BYTES, 256, "coalesced owner").unwrap();
        assert_eq!(whole.offset(), 0);
        assert!(whole.key().generation > first_generation);
        arena.release(whole).unwrap();
        assert!(arena.outstanding().is_empty());

        arena.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
        let after = ctx.memory_info().unwrap().0;
        assert!(
            after >= during.saturating_add(ARENA_BYTES),
            "checked close must return the physical allocation"
        );
        eprintln!("PASS admitted device arena on {}", ctx.uuid());
    }

    assert!(
        observed_in_flight,
        "at least one real async copy must expose the pre-completion refusal"
    );
}
