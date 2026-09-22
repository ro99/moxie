//! Task 0057: a two-rank group on the peer-connected 3090 pair and its ordered
//! all-gather, proven by a column-sharded BF16 linear. Real hardware only.
//!
//! Devices are chosen by what the driver grants, not by ordinal: the pair is
//! the two devices with mutual peer access, and the refused device is one with
//! none to them. Their UUIDs are printed; run with `--nocapture` to see them.
#![cfg(feature = "driver")]

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use moxie_cuda::{DeviceBuffer, RankContext, Stream, query_device};
use moxie_executor::{ColumnLinear, DeviceArena, LinearShape, RankGroup, RankShard};
use moxie_memory::{BufferRequest, CapacitySnapshot, Ledger, PlanRequest, StageSpan};
use moxie_types::{DeviceTier, Error, RankId, Scope, Tier};

const MIB: u64 = 1024 * 1024;
const ARENA_BYTES: u64 = 16 * MIB;
const DEADLINE: Duration = Duration::from_secs(10);
/// Even over two ranks, and large enough that the gather is still running
/// when a caller that skipped the completion wait would read it.
const EVEN: LinearShape = LinearShape {
    rows: 256,
    in_features: 1024,
    out_features: 2048,
};

// Test-only real stream gate: enqueue a bounded host function immediately before
// the actual event record. Copies remain real; the completion cannot race past
// the first sweep. No production API or fabricated event status is involved.
use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
static BLOCK_NEXT: AtomicBool = AtomicBool::new(false);
static RELEASE: AtomicBool = AtomicBool::new(true);
static TIMED_OUT: AtomicBool = AtomicBool::new(false);

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

unsafe extern "C" fn pending_work(_: *mut c_void) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !RELEASE.load(SeqCst) {
        if Instant::now() >= deadline {
            TIMED_OUT.store(true, SeqCst);
            break;
        }
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int {
    // SAFETY: RTLD_NEXT finds the real CUDA ABI symbols after this executable.
    let (launch, record) = unsafe {
        let launch = dlsym((-1isize) as *mut c_void, c"cuLaunchHostFunc".as_ptr());
        let record = dlsym((-1isize) as *mut c_void, c"cuEventRecord".as_ptr());
        if launch.is_null() || record.is_null() {
            return 1;
        }
        (
            std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(
                    *mut c_void,
                    unsafe extern "C" fn(*mut c_void),
                    *mut c_void,
                ) -> c_int,
            >(launch),
            std::mem::transmute::<
                *mut c_void,
                unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
            >(record),
        )
    };
    if BLOCK_NEXT.swap(false, SeqCst) {
        // SAFETY: stream comes unchanged from CUDA; callback uses static state,
        // calls no CUDA API and exits within ten seconds even on test failure.
        let result = unsafe { launch(stream, pending_work, std::ptr::null_mut()) };
        if result != 0 {
            return result;
        }
    }
    // SAFETY: event and stream are forwarded unchanged to the real driver.
    unsafe { record(event, stream) }
}

struct PendingGate;
impl PendingGate {
    fn arm() -> Self {
        RELEASE.store(false, SeqCst);
        TIMED_OUT.store(false, SeqCst);
        BLOCK_NEXT.store(true, SeqCst);
        Self
    }
}
impl Drop for PendingGate {
    fn drop(&mut self) {
        BLOCK_NEXT.store(false, SeqCst);
        RELEASE.store(true, SeqCst);
    }
}

static DEVICE: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The ordinals of the peer-connected pair, and of a device with no peer
/// access to it, if the machine has one.
fn devices() -> ([u32; 2], Option<u32>) {
    let caps: Vec<_> = (0..moxie_cuda::device_count().unwrap())
        .map(|ordinal| query_device(ordinal).unwrap())
        .collect();
    let pair = caps
        .iter()
        .flat_map(|a| caps.iter().map(move |b| (a, b)))
        .find(|(a, b)| {
            a.ordinal < b.ordinal && a.can_access_peer(b.ordinal) && b.can_access_peer(a.ordinal)
        })
        .map(|(a, b)| [a.ordinal, b.ordinal])
        .expect("the TP2 lane needs a peer-connected pair");
    let outsider = caps
        .iter()
        .find(|c| !pair.contains(&c.ordinal) && !c.can_access_peer(pair[0]))
        .map(|c| c.ordinal);
    (pair, outsider)
}

/// One rank's admitted arena, stream and kernel.
struct Rank<'c> {
    ledger: Ledger,
    arena: DeviceArena<'c>,
    stream: Stream<'c>,
    linear: ColumnLinear<'c>,
}

impl<'c> Rank<'c> {
    fn new(ctx: &'c RankContext) -> Self {
        let scope = Scope::Device(ctx.uuid());
        let mut ledger =
            Ledger::new([CapacitySnapshot::new(scope, 64 * MIB, MIB).unwrap()]).unwrap();
        let mut request = PlanRequest::new("tp2 rank", ["step"]).unwrap();
        request
            .buffer(BufferRequest::new(
                "shard, local output and gathered output",
                scope,
                Tier::Device(DeviceTier::Activations),
                ARENA_BYTES,
                StageSpan { first: 0, last: 0 },
            ))
            .unwrap();
        let reservation = ledger.admit(&request).unwrap();
        let arena = DeviceArena::create(
            &ledger,
            reservation,
            ctx,
            DeviceTier::Activations,
            ARENA_BYTES,
            "tp2 rank",
        )
        .unwrap();
        Self {
            ledger,
            arena,
            stream: Stream::new(ctx).unwrap(),
            linear: ColumnLinear::load(ctx).unwrap(),
        }
    }

    fn launch(
        &mut self,
        shape: LinearShape,
        rank: u32,
        ranks: u32,
        sequence: u64,
        cancel: &AtomicBool,
    ) -> moxie_types::Result<RankShard<'c>> {
        let (input, weight) = operands(shape);
        self.linear.launch(
            &mut self.arena,
            &self.stream,
            shape,
            input,
            &weight,
            rank,
            ranks,
            sequence,
            cancel,
        )
    }

    fn live(&self) -> usize {
        self.arena.occupancy().live_allocations
    }

    fn close(self) {
        let mut ledger = self.ledger;
        self.arena.close(&mut ledger).unwrap();
        assert!(ledger.outstanding().is_empty());
    }
}

/// Deterministic BF16 bytes: small integers over 64, all exact in BF16.
fn operands(shape: LinearShape) -> (Vec<u8>, Vec<u8>) {
    let bf16 = |n: u64, salt: u64| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                let v = ((i * 7919 + salt) % 255) as f32 - 127.0;
                (((v / 64.0).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect()
    };
    (
        bf16(shape.rows * shape.in_features, 3),
        bf16(shape.out_features * shape.in_features, 11),
    )
}

#[test]
fn a_column_sharded_linear_gathers_bit_identically_to_one_gpu() {
    let _guard = one_at_a_time();
    let ([a, b], _) = devices();
    let contexts = [
        RankContext::acquire(RankId(57_000), a).unwrap(),
        RankContext::acquire(RankId(57_001), b).unwrap(),
    ];
    let mut group = RankGroup::form([&contexts[0], &contexts[1]], DEADLINE).unwrap();
    println!("TP2 pair: {} + {}", group.uuids()[0], group.uuids()[1]);
    let [mut r0, mut r1] = [Rank::new(&contexts[0]), Rank::new(&contexts[1])];
    let never = AtomicBool::new(false);

    // Refused by `shard_weight_ranges` before anything is allocated.
    let odd = LinearShape {
        out_features: 2047,
        ..EVEN
    };
    let refused = r0.launch(odd, 0, 2, 0, &never).unwrap_err();
    assert!(matches!(refused, Error::Dim(_)), "{refused}");
    assert_eq!(r0.live(), 0);

    let sequence = group.next_sequence();
    let shards = [
        r0.launch(EVEN, 0, 2, sequence, &never),
        r1.launch(EVEN, 1, 2, sequence, &never),
    ];
    let gathered = group
        .all_gather(
            shards,
            [&mut r0.arena, &mut r1.arena],
            [&r0.stream, &r1.stream],
            &never,
        )
        .unwrap();
    let bytes = (EVEN.rows * EVEN.out_features * 2) as usize;
    let mut out = [vec![0u8; bytes], vec![0u8; bytes]];
    for (g, o) in gathered.iter().zip(&mut out) {
        g.read(o).unwrap();
    }
    let [g0, g1] = gathered;
    g0.release(&mut r0.arena).unwrap();
    g1.release(&mut r1.arena).unwrap();

    // The oracle runs after the gather, so no range it wrote can be mistaken
    // for a gathered one.
    let reference = r0
        .launch(EVEN, 0, 1, 0, &never)
        .unwrap()
        .finish(&mut r0.arena, DEADLINE)
        .unwrap();
    assert!(out[0] == reference, "rank 0's gather differs from one GPU");
    assert!(out[1] == reference, "rank 1's gather differs from one GPU");
    r0.close();
    r1.close();
}

#[test]
fn injected_failures_are_typed_bounded_and_release_everything() {
    let _guard = one_at_a_time();
    let ([a, b], _) = devices();
    let contexts = [
        RankContext::acquire(RankId(57_010), a).unwrap(),
        RankContext::acquire(RankId(57_011), b).unwrap(),
    ];
    let mut group = RankGroup::form([&contexts[0], &contexts[1]], DEADLINE).unwrap();
    let [mut r0, mut r1] = [Rank::new(&contexts[0]), Rank::new(&contexts[1])];
    let other_rows = LinearShape { rows: 128, ..EVEN };
    let odd = LinearShape {
        out_features: 2047,
        ..EVEN
    };
    for (what, kind) in [
        ("rank 1's own work fails before the collective", "dim"),
        ("rank 1 declares the next sequence", "invalid_request"),
        ("rank 1 declares another shape", "invalid_request"),
        ("cancelled between the rank launches", "cancelled"),
    ] {
        let cancel = AtomicBool::new(false);
        let sequence = group.next_sequence();
        let first = r0.launch(EVEN, 0, 2, sequence, &cancel);
        let second = match what {
            "rank 1's own work fails before the collective" => {
                r1.launch(odd, 1, 2, sequence, &cancel)
            }
            "rank 1 declares the next sequence" => r1.launch(EVEN, 1, 2, sequence + 1, &cancel),
            "rank 1 declares another shape" => r1.launch(other_rows, 1, 2, sequence, &cancel),
            _ => {
                cancel.store(true, SeqCst);
                r1.launch(EVEN, 1, 2, sequence, &cancel)
            }
        };
        let started = Instant::now();
        let refused = group
            .all_gather(
                [first, second],
                [&mut r0.arena, &mut r1.arena],
                [&r0.stream, &r1.stream],
                &cancel,
            )
            .map(|_| ())
            .unwrap_err();
        // Early, not at the deadline: nothing waits on the failed rank.
        assert!(started.elapsed() < DEADLINE / 10, "{what}: not early");
        for error in &refused.errors {
            assert_eq!(error.kind(), kind, "{what}: {error}");
        }
        assert_eq!(
            (r0.live(), r1.live()),
            (0, 0),
            "{what}: a range is still held"
        );
    }

    let never = AtomicBool::new(false);
    let sequence = group.next_sequence();
    let shards = [
        r0.launch(EVEN, 0, 2, sequence, &never),
        r1.launch(EVEN, 1, 2, sequence, &never),
    ];
    let [g0, g1] = group
        .all_gather(
            shards,
            [&mut r0.arena, &mut r1.arena],
            [&r0.stream, &r1.stream],
            &never,
        )
        .expect("a clean collective after the failures");
    g0.release(&mut r0.arena).unwrap();
    g1.release(&mut r1.arena).unwrap();
    r0.close();
    r1.close();
}

#[test]
fn a_drain_that_misses_the_deadline_withholds_every_range() {
    let _guard = one_at_a_time();
    let ([a, b], _) = devices();
    let contexts = [
        RankContext::acquire(RankId(57_030), a).unwrap(),
        RankContext::acquire(RankId(57_031), b).unwrap(),
    ];
    let mut group = RankGroup::form([&contexts[0], &contexts[1]], Duration::ZERO).unwrap();
    let [mut r0, mut r1] = [Rank::new(&contexts[0]), Rank::new(&contexts[1])];
    let never = AtomicBool::new(false);
    let shards = [
        r0.launch(EVEN, 0, 2, 0, &never),
        r1.launch(EVEN, 1, 2, 0, &never),
    ];
    // Hold rank 0's stream before its drain event, so the zero deadline is
    // missed while rank 0 may still be reading both ranks' sources.
    let gate = PendingGate::arm();
    let refused = group
        .all_gather(
            shards,
            [&mut r0.arena, &mut r1.arena],
            [&r0.stream, &r1.stream],
            &never,
        )
        .map(|_| ())
        .unwrap_err();
    for error in &refused.errors {
        assert_eq!(error.kind(), "device_lost", "{error}");
    }
    // Input, weight shard, local output and gathered output, on both ranks:
    // sources included, because a consumer of each is not known complete.
    assert_eq!((r0.live(), r1.live()), (4, 4), "{refused}");
    drop(gate);
    r0.stream.synchronize().unwrap();
    assert!(!TIMED_OUT.load(SeqCst), "the gate released itself");
}

#[test]
fn a_device_without_peer_access_is_refused_and_never_staged() {
    let _guard = one_at_a_time();
    let ([a, b], outsider) = devices();
    let Some(outsider) = outsider else {
        panic!("this lane needs a device without peer access to the pair");
    };
    let inside = RankContext::acquire(RankId(57_020), a).unwrap();
    let outside = RankContext::acquire(RankId(57_021), outsider).unwrap();
    println!("refused pairing: {} + {}", inside.uuid(), outside.uuid());
    let refused = |result: moxie_types::Result<()>| {
        assert!(
            matches!(
                result,
                Err(Error::Unsupported {
                    capability: "peer_access",
                    ..
                })
            ),
            "{result:?}"
        );
    };
    refused(RankGroup::form([&inside, &outside], DEADLINE).map(|_| ()));

    // Without a grant the driver would stage this copy through the host.
    let source = DeviceBuffer::alloc(&inside, 256).unwrap();
    let destination = DeviceBuffer::alloc(&outside, 256).unwrap();
    let stream = Stream::new(&outside).unwrap();
    // SAFETY: both buffers and the stream outlive the synchronize below.
    let copied = unsafe { destination.copy_from_peer_async_at(0, &source, 0, 256, &stream) };
    stream.synchronize().unwrap();
    refused(copied);

    // A grant names the peer's acquisition, not its device: once the peer
    // context is released and reacquired, the old grant authorizes nothing.
    let peer = RankContext::acquire(RankId(57_022), b).unwrap();
    inside.enable_peer_access(&peer).unwrap();
    drop(peer);
    let peer = RankContext::acquire(RankId(57_022), b).unwrap();
    let source = DeviceBuffer::alloc(&peer, 256).unwrap();
    let destination = DeviceBuffer::alloc(&inside, 256).unwrap();
    let stream = Stream::new(&inside).unwrap();
    // SAFETY: as above.
    let copied = unsafe { destination.copy_from_peer_async_at(0, &source, 0, 256, &stream) };
    stream.synchronize().unwrap();
    refused(copied);
}
