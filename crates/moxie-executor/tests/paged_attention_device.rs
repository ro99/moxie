//! Task 0037 acceptance 4: what happens when a paged attention run is refused.
//!
//! The device numerics are qualified by `cargo xtask-cuda test-gpu`, which
//! compares every component against the FP64 oracle on all three GPUs. This
//! file is about the other half — the paths where nothing should happen:
//! admission that cannot fit, a page mapping that is not a mapping, an append
//! whose bytes do not match its rows, a stream from another device, and a
//! decode loop that must not grow anything.
//!
//! **Refusal has a shape here.** Before anything is enqueued, a refusal hands
//! the caller's bytes back and leaves the committed frontier and every prior
//! byte exactly as they were. After something is enqueued and its completion is
//! unknown, the run keeps the bytes instead, forever, because submitted work
//! may still be reading them. Those are different outcomes and the tests below
//! distinguish them rather than checking `is_err()`.
#![cfg(feature = "driver")]

use std::sync::{Mutex, MutexGuard};

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_executor::{
    AttentionLayer, PageGeometry, PagedAttentionLaunch, PagedAttentionRun, Staging,
};
use moxie_kernels::cpu_expert::to_bf16_bits;
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::Visibility;
use moxie_state::{DeviceKvSequence, KvGeometry, LayerKv, Retention};
use moxie_types::{Error, PagePlacement, Precision, RankId, Scope};

/// A `RankContext` is exclusive per device and `cargo test` runs a binary's
/// tests in parallel threads. Serialising them is the property task 0007
/// established, not a workaround for it.
static DEVICE: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

fn geometry() -> PageGeometry {
    PageGeometry {
        kv_heads: 2,
        head_dim: 64,
        page_tokens: 16,
        pages: 4,
    }
}

/// The same layer, described to the state authority.
///
/// One geometry, two vocabularies: the authority speaks rows, retention and
/// transactions, the run speaks bytes and pages. `authority()` below asserts
/// they resolve to the same page count, which is what keeps them one layer.
fn kv_geometry() -> KvGeometry {
    KvGeometry {
        layers: vec![LayerKv {
            kv_heads: geometry().kv_heads as usize,
            key_dim: geometry().head_dim as usize,
            value_dim: geometry().head_dim as usize,
            retention: Retention::All,
        }],
        precision: Precision::Bf16,
        page_tokens: geometry().page_tokens as usize,
        max_tokens: (geometry().pages * geometry().page_tokens) as usize,
        tentative_rows: (geometry().pages * geometry().page_tokens) as usize,
    }
}

fn authority() -> DeviceKvSequence {
    let sequence = DeviceKvSequence::new(kv_geometry()).expect("a device sequence");
    assert_eq!(
        sequence.layout(0).expect("one layer").pages,
        geometry().pages,
        "the authority and the run must describe the same pages"
    );
    sequence
}

/// One decode over whatever the authority currently retains.
fn decode_at<'ctx>(
    run: &mut PagedAttentionRun<'ctx>,
    stream: &Stream<'ctx>,
    sequence: &DeviceKvSequence,
    position: u64,
) -> Vec<u8> {
    let retained = sequence.retained(0).expect("a range");
    let launch = PagedAttentionLaunch::new(
        layer(),
        1,
        position,
        retained.start,
        retained.end - retained.start,
    )
    .expect("a launch over the retained range");
    run.attend(
        stream,
        &launch,
        bf16_bytes((HEADS * geometry().head_dim) as usize, 0x4242),
    )
    .map_err(|r| r.error)
    .expect("the decode runs")
}

/// Publish `rows` rows through the authority and write them through the run.
///
/// The order is the contract: placements first, bytes second, publication last
/// and only when the write returned. A run that wrote nothing must never leave
/// the authority claiming history.
fn append_through_authority<'ctx>(
    sequence: &mut DeviceKvSequence,
    run: &mut PagedAttentionRun<'ctx>,
    stream: &Stream<'ctx>,
    rows: u64,
    seed: u64,
) -> Vec<PagePlacement> {
    let txn = sequence.begin().expect("a transaction");
    let staged = sequence.stage(txn, rows).expect("stage");
    let placements = sequence
        .placements(&staged, 0)
        .expect("the authority places the rows");
    // The mapping **before** the write, covering the rows about to land. A
    // write is checked against the published view, so a view that stopped at
    // the retained range would refuse the very rows it is about to gain. It
    // grows as pages are filled and slides as the ring reclaims; republishing
    // it is how the run learns both.
    let view = sequence.page_view(0).expect("a view");
    run.publish_page_table(stream, view.base, view.table)
        .map_err(|r| r.error)
        .expect("the authority's mapping");
    let keys = bf16_bytes(rows as usize * row_bytes(), seed);
    let values = bf16_bytes(rows as usize * row_bytes(), seed + 1);
    run.write_rows(stream, &placements, keys, values)
        .map_err(|r| r.error)
        .expect("the write fits");
    sequence.publish(txn, staged).expect("publish");
    sequence.commit(txn, rows).expect("commit");
    placements
}

const HEADS: u64 = 4;
const MAX_ROWS: u64 = 2;

fn layer() -> AttentionLayer {
    AttentionLayer {
        geometry: geometry(),
        heads: HEADS,
        scale: moxie_plan::reciprocal_sqrt_scale(64),
        visibility: Visibility::Causal,
    }
}

fn launch(rows: u64, first_position: u64, history_rows: u64) -> PagedAttentionLaunch {
    PagedAttentionLaunch::new(layer(), rows, first_position, 0, history_rows)
        .expect("the fixture's launches are legal")
}

/// Deterministic BF16 bytes: `rows` stored rows, or one query block.
fn bf16_bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count * 2);
    for _ in 0..count {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = ((state >> 41) as f32) / ((1u32 << 22) as f32) - 1.0;
        out.extend_from_slice(&to_bf16_bits(value).to_le_bytes());
    }
    out
}

fn row_bytes() -> usize {
    (geometry().kv_heads * geometry().head_dim) as usize
}

fn measured_ledger(ctx: &RankContext) -> Ledger {
    let measurement = ctx.measure().expect("measure the device");
    let device = CapacitySnapshot::measured(&measurement, 1 << 20).expect("device capacity");
    let host = moxie_host::read().expect("host capacity");
    let host = CapacitySnapshot::measured_host(&host, 1 << 20).expect("host capacity");
    Ledger::new([device, host]).expect("one ledger")
}

fn descriptor_for(ctx: &RankContext) -> moxie_types::SemanticKernelDescriptor {
    let capability = query_device(ctx.ordinal()).expect("query the device");
    moxie_executor::select_paged_attention_kernel(
        &moxie_kernels::paged_attention_catalogue(),
        &capability,
        &launch(1, 0, 1),
    )
    .expect("this build declares a paged attention descriptor for this device")
}

/// One admitted run with its mapping published and `rows` rows committed.
fn prepared<'ctx>(
    ledger: &mut Ledger,
    ctx: &'ctx RankContext,
    stream: &Stream<'ctx>,
    rows: u64,
) -> (PagedAttentionRun<'ctx>, DeviceKvSequence) {
    let mut run = PagedAttentionRun::admit(
        ledger,
        ctx,
        descriptor_for(ctx),
        geometry(),
        HEADS,
        MAX_ROWS,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("admission fits a measured device");
    let mut sequence = authority();
    // The mapping the authority publishes for its own retained range, not one
    // this test invented. With a retain-all layer it is the identity until the
    // ring wraps, and the wrap is what `a_reclaimed_base` exercises.
    if rows > 0 {
        append_through_authority(&mut sequence, &mut run, stream, rows, 0x37_0001);
    } else {
        run.publish_page_table(stream, 0, vec![0, 1, 2, 3])
            .map_err(|r| r.error)
            .expect("an identity mapping");
    }
    (run, sequence)
}

#[test]
fn admission_refuses_before_it_allocates_and_strands_nothing() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let scope = Scope::Device(ctx.uuid());

    // A ledger that cannot hold the pages. The refusal must carry the ledger's
    // own rejection -- summarising it into a byte count throws away what a
    // caller needs to act -- and must leave nothing charged.
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(scope, 4096, 1024).expect("tiny device snapshot"),
        CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).expect("host snapshot"),
    ])
    .expect("one ledger");
    let refused = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor_for(&ctx),
        geometry(),
        HEADS,
        MAX_ROWS,
        Staging::Host,
    )
    .expect_err("32 KiB of pages do not fit in 4 KiB");
    assert!(
        matches!(refused.error, Error::CapacityExceeded { .. }),
        "{:?}",
        refused.error
    );
    assert!(
        refused.rejection.is_some(),
        "a capacity refusal must carry the ledger's breakdown"
    );
    assert!(
        refused.reservation.is_none(),
        "nothing was charged, so nothing should be handed back"
    );
    assert!(ledger.outstanding().is_empty(), "a refusal stranded bytes");
}

#[test]
fn a_descriptor_that_does_not_serve_the_geometry_is_refused_at_admission() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let mut ledger = measured_ledger(&ctx);

    // Admission re-applies selection's own predicate. These are descriptors
    // selection would never have chosen, handed to a public entry point.
    let narrowed = {
        let mut d = descriptor_for(&ctx);
        d.shape.max_input = geometry().head_dim - 1;
        d
    };
    assert!(
        PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            narrowed,
            geometry(),
            HEADS,
            MAX_ROWS,
            Staging::Host
        )
        .is_err(),
        "a descriptor whose shape bounds exclude this head dimension was accepted"
    );
    let renamed = {
        let mut d = descriptor_for(&ctx);
        d.symbols[0] = moxie_types::KernelSymbol(moxie_kernels::AFFINE_LINEAR.to_string());
        d
    };
    assert!(
        PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            renamed,
            geometry(),
            HEADS,
            MAX_ROWS,
            Staging::Host
        )
        .is_err(),
        "a descriptor naming another package's symbol was accepted"
    );
    let integer_cache = {
        let mut d = descriptor_for(&ctx);
        d.inputs[1] = moxie_types::KernelOperand::Weight(moxie_types::WeightPrecision::expect(
            moxie_types::Precision::Int4,
        ));
        d
    };
    assert!(
        PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            integer_cache,
            geometry(),
            HEADS,
            MAX_ROWS,
            Staging::Host,
        )
        .is_err(),
        "a non-BF16 cache operand was accepted; unsupported must fail to match"
    );
    // **Whole identity, not field by field.** Each of these keeps operation,
    // ABI, operands, output, symbol and shape bounds intact and changes one
    // field that admission would otherwise never look at -- and admission loads
    // this build's image regardless, so a descriptor declaring a different
    // layout, accumulation, rounding, workspace or image would have been
    // executed by code that declares something else. The binding requires the
    // descriptor to *be* one the built-in package declares, which is the only
    // form of the check that cannot be half-satisfied.
    /// One named change to a descriptor, applied alone.
    type Change = Box<dyn Fn(&mut moxie_types::SemanticKernelDescriptor)>;
    let identity: Vec<(&str, Change)> = vec![
        (
            "a different accumulation policy",
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                d.accumulation = moxie_types::AccumulationPolicy::F32;
            }),
        ),
        (
            "a workspace this kernel does not take",
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                d.workspace = moxie_types::WorkspaceExpression::RowsTimesF32;
            }),
        ),
        (
            "a zeroed image digest",
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                d.image_sha256 = [0; 32];
            }),
        ),
        (
            "an ABI version this binding does not speak",
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                d.abi_version += 1;
            }),
        ),
        (
            "an identity that is not in the package",
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                d.id = moxie_types::KernelId("bf16-paged-attention-v1-invented".to_string());
            }),
        ),
        (
            "a rounding profile or layout this package does not declare",
            // Nothing to write: `RoundingProfile` and `TensorLayout` each have
            // exactly one variant today, so neither field can be given a wrong
            // value. Membership compares them regardless, and
            // `moxie-kernels`' own fixture asserts that. When either enum gains
            // a second variant this case stops being a no-op and this comment
            // stops being true.
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                assert_eq!(d.rounding, moxie_types::RoundingProfile::FinalBf16Rne);
                assert_eq!(d.layout, moxie_types::TensorLayout::ContiguousRowMajorV1);
                // Change something that *is* variable, so this case still
                // asserts a refusal rather than passing vacuously.
                d.output = moxie_types::ActivationPrecision::expect(moxie_types::Precision::F32);
            }),
        ),
        (
            "widened shape bounds",
            Box::new(|d: &mut moxie_types::SemanticKernelDescriptor| {
                d.shape.max_rows = u64::MAX;
            }),
        ),
    ];
    for (what, change) in identity {
        let mut descriptor = descriptor_for(&ctx);
        change(&mut descriptor);
        assert!(
            PagedAttentionRun::admit(
                &mut ledger,
                &ctx,
                descriptor,
                geometry(),
                HEADS,
                MAX_ROWS,
                Staging::Host
            )
            .is_err(),
            "a descriptor with {what} was accepted for a launch"
        );
    }
    // The control, and it is load-bearing: the package's own descriptor still
    // admits, so the refusals above are the check biting rather than the
    // fixture failing.
    let run = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor_for(&ctx),
        geometry(),
        HEADS,
        MAX_ROWS,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("the package's own descriptor admits");
    run.close(&mut ledger).map_err(|r| r.error).expect("close");

    // A head count that fits a `u32` but not the device's grid. CUDA's `y` and
    // `z` limits are 65,535 while `x` reaches `2^31 - 1`, so this launch is
    // legal as a value and unlaunchable on this hardware. It must be refused at
    // **admission**, with nothing charged — not discovered by `cuLaunchKernel`
    // after a query has already been copied, which is what happened before the
    // limits were queried and checked.
    let mut one_kv = geometry();
    one_kv.kv_heads = 1;
    let too_many_heads = u64::from(query_device(0).expect("query").max_grid.1) + 1;
    let refused = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor_for(&ctx),
        one_kv,
        too_many_heads,
        1,
        Staging::Host,
    )
    .err()
    .map(|r| r.error)
    .expect("a grid of 65,536 blocks in y was accepted");
    assert!(
        matches!(refused, Error::UnsupportedKernel { .. }),
        "{refused:?}"
    );
    assert!(ledger.outstanding().is_empty(), "a refusal stranded bytes");
    // One below the limit is admissible, so the refusal is the limit biting
    // rather than the head count being large.
    PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor_for(&ctx),
        one_kv,
        too_many_heads - 1,
        1,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("the largest launchable head count admits")
    .close(&mut ledger)
    .map_err(|r| r.error)
    .expect("close");

    // A head ratio that does not divide is refused before any of that.
    let mut odd = geometry();
    odd.kv_heads = 3;
    assert!(
        PagedAttentionRun::admit(
            &mut ledger,
            &ctx,
            descriptor_for(&ctx),
            odd,
            HEADS,
            MAX_ROWS,
            Staging::Host,
        )
        .is_err(),
        "four query heads over three key/value heads was accepted"
    );
    assert!(ledger.outstanding().is_empty(), "a refusal stranded bytes");
}

#[test]
fn a_mapping_that_is_not_a_mapping_is_refused() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let stream = Stream::new(&ctx).expect("a stream");
    let mut ledger = measured_ledger(&ctx);
    let mut run = PagedAttentionRun::admit(
        &mut ledger,
        &ctx,
        descriptor_for(&ctx),
        geometry(),
        HEADS,
        MAX_ROWS,
        Staging::Host,
    )
    .map_err(|r| r.error)
    .expect("admission fits");

    // A physical page that does not exist, and one named twice. The second is
    // the dangerous one: aliased pages make an append overwrite history that is
    // still visible, and no later check could tell.
    assert!(
        run.publish_page_table(&stream, 0, vec![0, 1, 99, 3])
            .is_err(),
        "a physical page outside the admitted set was accepted"
    );
    assert!(
        run.publish_page_table(&stream, 0, vec![0, 1, 1, 3])
            .is_err(),
        "an aliased physical page was accepted"
    );
    assert!(
        run.publish_page_table(&stream, 0, Vec::new()).is_err(),
        "an empty mapping was accepted"
    );
    assert!(
        run.publish_page_table(&stream, 0, vec![0, 1, 2, 3, 0])
            .is_err(),
        "a mapping longer than the admitted pages was accepted"
    );
    assert!(
        run.publish_page_table(&stream, 4, vec![0, 1, 2, 3])
            .is_err(),
        "a mapping starting beyond the written rows was accepted"
    );
    assert!(
        run.publish_page_table(&stream, 1, vec![0, 1, 2, 3])
            .is_err(),
        "a mapping starting inside a page was accepted"
    );
    run.publish_page_table(&stream, 0, vec![3, 2, 1, 0])
        .map_err(|r| r.error)
        .expect("a reversed mapping is a mapping");
    let placement = PagePlacement {
        position: 0,
        physical_page: 3,
        slot: 0,
        rows: 1,
    };
    let keys = bf16_bytes(row_bytes(), 1);
    let values = bf16_bytes(row_bytes(), 2);
    run.write_rows(&stream, &[placement], keys, values)
        .map_err(|r| r.error)
        .expect("the write fits");

    // A write must agree with the published mapping: row 0 lives on physical
    // page 3 under this table.
    let disagreeing = PagePlacement {
        position: 1,
        physical_page: 0,
        slot: 1,
        rows: 1,
    };
    let refused = run
        .write_rows(
            &stream,
            &[disagreeing],
            bf16_bytes(row_bytes(), 5),
            bf16_bytes(row_bytes(), 6),
        )
        .expect_err("a placement contradicting the mapping was accepted");
    assert!(!refused.retained_source());

    // Republishing is legal — the retained range slides — but it may not
    // contradict the mapping rows were written through.
    let refused = run
        .publish_page_table(&stream, 0, vec![0, 1, 2, 3])
        .expect_err("a mapping contradicting written rows was accepted");
    assert!(!refused.retained_source());
    // A mapping that agrees where it overlaps is accepted, including one that
    // starts later: the rows below the new base are simply not described.
    run.publish_page_table(&stream, 0, vec![3, 2, 1, 0])
        .map_err(|r| r.error)
        .expect("the same mapping republishes");
    // A base past the written rows is refused: there is nothing there. A base
    // that has really moved is exercised by the wrapped-ring case below.
    assert!(
        run.publish_page_table(&stream, 16, vec![2, 1, 0]).is_err(),
        "a mapping starting past the written rows was accepted"
    );
    // A launch naming the wrong base is checked in the wrapped-ring case,
    // where a base other than zero exists.
    // A placement outside the admitted pages, one that runs off the end of its
    // page, and a set that is not contiguous are each refused before any copy.
    for (what, placements) in [
        (
            "a page that does not exist",
            vec![PagePlacement {
                position: 1,
                physical_page: 9,
                slot: 0,
                rows: 1,
            }],
        ),
        (
            "a run off the end of its page",
            vec![PagePlacement {
                position: 1,
                physical_page: 0,
                slot: 15,
                rows: 2,
            }],
        ),
        (
            "a gap between runs",
            vec![
                PagePlacement {
                    position: 1,
                    physical_page: 0,
                    slot: 1,
                    rows: 1,
                },
                PagePlacement {
                    position: 3,
                    physical_page: 0,
                    slot: 3,
                    rows: 1,
                },
            ],
        ),
    ] {
        let rows: u64 = placements.iter().map(|p| p.rows).sum();
        let refused = run
            .write_rows(
                &stream,
                &placements,
                bf16_bytes(rows as usize * row_bytes(), 3),
                bf16_bytes(rows as usize * row_bytes(), 4),
            )
            .expect_err(what);
        assert!(
            !refused.retained_source(),
            "{what}: a pre-enqueue refusal kept bytes"
        );
    }
    run.close(&mut ledger).map_err(|r| r.error).expect("close");
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_refused_append_moves_nothing_and_hands_the_rows_back() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let stream = Stream::new(&ctx).expect("a stream");
    let mut ledger = measured_ledger(&ctx);
    let (mut run, mut sequence) = prepared(&mut ledger, &ctx, &stream, 20);
    // The rows already written, placed again: the authority's mapping is a
    // function of the position, so asking twice gives the same answer.
    let written: Vec<PagePlacement> = (0..20)
        .map(|row| sequence.placement_of(0, row).expect("a placement"))
        .collect();
    let before = run.read_rows(&written).expect("written rows read back");
    let capacity = run.capacity_rows().expect("capacity");
    assert_eq!(capacity, 64, "four pages of sixteen rows");
    let txn = sequence.begin().expect("a transaction");
    let staged = sequence.stage(txn, 2).expect("stage");
    let good = sequence.placements(&staged, 0).expect("two more rows");

    let cases: Vec<(&str, Vec<PagePlacement>, usize, usize)> = vec![
        ("no placement at all", Vec::new(), 0, 0),
        (
            "keys short of their row count",
            good.clone(),
            row_bytes(),
            2 * row_bytes(),
        ),
        (
            "values short of their row count",
            good.clone(),
            2 * row_bytes(),
            row_bytes(),
        ),
    ];
    for (label, placements, keys, values) in cases {
        let refused = run
            .write_rows(
                &stream,
                &placements,
                bf16_bytes(keys, 7),
                bf16_bytes(values, 8),
            )
            .err()
            .unwrap_or_else(|| panic!("{label} was accepted"));
        assert!(
            !refused.retained_source(),
            "{label}: a refusal before enqueue must hand the rows back"
        );
        assert_eq!(run.written_rows(), 20, "{label}: the high-water mark moved");
        assert_eq!(
            run.read_rows(&written).expect("read back"),
            before,
            "{label}: written bytes changed"
        );
        assert_eq!(
            sequence.committed_rows(),
            20,
            "{label}: the authority published rows the run refused"
        );
    }
    // The authority refuses to stage rows past its own admitted context, which
    // is the other half: the run bounds bytes, the authority bounds history.
    sequence.abort(txn).expect("abort");
    let txn = sequence.begin().expect("a transaction");
    assert!(sequence.stage(txn, capacity + 1).is_err());
    sequence.abort(txn).expect("abort");

    // And a launch that claims history nothing wrote.
    let query = bf16_bytes((HEADS * geometry().head_dim) as usize, 9);
    let refused = run
        .attend(&stream, &launch(1, 19, 21), query)
        .expect_err("a launch past the written rows was accepted");
    assert!(!refused.retained_source());
    assert_eq!(run.written_rows(), 20);
    run.close(&mut ledger).map_err(|r| r.error).expect("close");
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn a_stream_from_another_device_is_refused() {
    let _guard = one_at_a_time();
    let count = device_count().expect("enumerate");
    if count < 2 {
        eprintln!("SKIPPED: this case needs two devices, {count} visible");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let other = RankContext::acquire(RankId(1), 1).expect("acquire device 1");
    let stream = Stream::new(&ctx).expect("a stream");
    let foreign = Stream::new(&other).expect("a stream on the other device");
    let mut ledger = measured_ledger(&ctx);
    let (mut run, sequence) = prepared(&mut ledger, &ctx, &stream, 4);

    let mut sequence = sequence;
    let txn = sequence.begin().expect("a transaction");
    let staged = sequence.stage(txn, 1).expect("stage");
    let placements = sequence.placements(&staged, 0).expect("one more row");
    let keys = bf16_bytes(row_bytes(), 3);
    let values = bf16_bytes(row_bytes(), 4);
    let refused = run
        .write_rows(&foreign, &placements, keys, values)
        .expect_err("a write on a foreign stream was accepted");
    assert!(!refused.retained_source());
    assert_eq!(run.written_rows(), 4, "the high-water mark moved");

    let query = bf16_bytes((HEADS * geometry().head_dim) as usize, 5);
    assert!(
        run.attend(&foreign, &launch(1, 3, 4), query).is_err(),
        "a launch on a foreign stream was accepted"
    );
    run.close(&mut ledger).map_err(|r| r.error).expect("close");
    assert!(ledger.outstanding().is_empty());
}

#[test]
fn repeated_decode_admits_nothing_further() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let stream = Stream::new(&ctx).expect("a stream");
    let mut ledger = measured_ledger(&ctx);
    let (mut run, mut sequence) = prepared(&mut ledger, &ctx, &stream, 1);

    // The claim is narrow and exact: decoding does not grow what was admitted.
    // The pages, the table, the query and the output are charged once at
    // admission, and every step after that reuses them. It is *not* a claim
    // about host memory the allocator may hold, which this test cannot see.
    let charged = ledger.outstanding().len();
    let bytes = run.arena_bytes();
    let mut previous: Option<Vec<u8>> = None;
    for step in 1..=32u64 {
        append_through_authority(&mut sequence, &mut run, &stream, 1, 0x1000 + step);
        let query = bf16_bytes((HEADS * geometry().head_dim) as usize, 0x3000 + step);
        let out = run
            .attend(&stream, &launch(1, step, step + 1), query)
            .map_err(|r| r.error)
            .expect("one decode per step");
        assert!(
            out.chunks_exact(2)
                .all(|w| u16::from_le_bytes([w[0], w[1]]) & 0x7F80 != 0x7F80),
            "step {step} produced a nonfinite BF16 component"
        );
        // Each step attends over a different history, so the answers must
        // differ: an unchanging output would mean the append was not read.
        if let Some(earlier) = previous.replace(out.clone()) {
            assert_ne!(earlier, out, "step {step} repeated the previous answer");
        }
        assert_eq!(run.written_rows(), step + 1);
        assert_eq!(
            sequence.committed_rows(),
            step + 1,
            "the authority and the run disagree about step {step}"
        );
        assert_eq!(run.arena_bytes(), bytes, "step {step} grew the arena");
        assert_eq!(
            ledger.outstanding().len(),
            charged,
            "step {step} admitted something new"
        );
    }
    run.close(&mut ledger).map_err(|r| r.error).expect("close");
    assert!(ledger.outstanding().is_empty());
}

/// Task 0038 acceptance 2: **the wrap is invisible to the answer.**
///
/// The case task 0037 could not reach. A windowed layer's ring wraps, the
/// authority's retained base leaves zero, and the launch attends over
/// `history_base > 0` — physical pages in a different order than the logical
/// rows, with the oldest pages holding the newest rows.
///
/// The check is an equality rather than a tolerance: a second run with enough
/// pages that nothing is ever overwritten holds the same rows at the same
/// absolute positions, and a query at the same position with the same window
/// sees the same keys and values. Two different physical layouts, one logical
/// history, and the device must not be able to tell the difference. A tolerance
/// would have accepted a kernel that read the wrong page and happened to be
/// close.
#[test]
fn a_wrapped_ring_answers_exactly_as_an_unwrapped_one() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    const ROWS: u64 = 100;
    const WINDOW: u64 = 40;
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let stream = Stream::new(&ctx).expect("a stream");

    // Two geometries over one layer. The first wraps: four 16-row pages hold 64
    // rows and the history is 100 long. The second cannot: eight pages hold
    // 128, so every row stays where it was written.
    let run_case = |pages: u64, ledger: &mut Ledger| -> (Vec<u8>, u64, u64) {
        let geometry = PageGeometry {
            kv_heads: 2,
            head_dim: 64,
            page_tokens: 16,
            pages,
        };
        let kv = KvGeometry {
            layers: vec![LayerKv {
                kv_heads: 2,
                key_dim: 64,
                value_dim: 64,
                retention: if pages == 4 {
                    // 40 + 8 is three pages, plus the eviction page: four.
                    Retention::Window {
                        window: WINDOW as usize,
                    }
                } else {
                    Retention::All
                },
            }],
            precision: Precision::Bf16,
            page_tokens: 16,
            // Context, not capacity. A windowed layer's whole purpose is a
            // context longer than the rows it keeps, so the four-page case
            // admits a 4,096-position context over 64 physical rows; the
            // retain-all case must physically hold everything it admits.
            max_tokens: if pages == 4 {
                4096
            } else {
                (pages * 16) as usize
            },
            tentative_rows: 8,
        };
        let mut sequence = DeviceKvSequence::new(kv).expect("a device sequence");
        assert_eq!(
            sequence.layout(0).expect("one layer").pages,
            pages,
            "the authority and the run must describe the same pages"
        );
        let capability = query_device(ctx.ordinal()).expect("query the device");
        let layer = AttentionLayer {
            geometry,
            heads: HEADS,
            scale: moxie_plan::reciprocal_sqrt_scale(64),
            visibility: Visibility::SlidingWindow { window: WINDOW },
        };
        let descriptor = moxie_executor::select_paged_attention_kernel(
            &moxie_kernels::paged_attention_catalogue(),
            &capability,
            &PagedAttentionLaunch::new(layer, 1, 0, 0, 1).expect("a launch"),
        )
        .expect("a descriptor");
        let mut run =
            PagedAttentionRun::admit(ledger, &ctx, descriptor, geometry, HEADS, 1, Staging::Host)
                .map_err(|r| r.error)
                .expect("admission fits");

        // The same bytes at the same positions in both cases: the row's seed is
        // its absolute position, so nothing about the physical layout can
        // change what a row contains.
        let mut written = 0u64;
        while written < ROWS {
            let step = 8.min(ROWS - written);
            let txn = sequence.begin().expect("a transaction");
            let staged = sequence.stage(txn, step).expect("stage");
            let placements = sequence
                .placements(&staged, 0)
                .expect("the authority places the rows");
            // The mapping before the write: a write is checked against the
            // published view, and the view has to cover the rows about to land.
            let view = sequence.page_view(0).expect("a view");
            run.publish_page_table(&stream, view.base, view.table)
                .map_err(|r| r.error)
                .expect("the authority's mapping");
            let row = (2 * 64) as usize;
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for offset in 0..step {
                keys.extend_from_slice(&bf16_bytes(row, 0x5000 + written + offset));
                values.extend_from_slice(&bf16_bytes(row, 0x9000 + written + offset));
            }
            run.write_rows(&stream, &placements, keys, values)
                .map_err(|r| r.error)
                .expect("the write fits");
            sequence.publish(txn, staged).expect("publish");
            sequence.commit(txn, step).expect("commit");
            written += step;
        }

        let retained = sequence.retained(0).expect("a range");
        let view = sequence.page_view(0).expect("a view");
        run.publish_page_table(&stream, view.base, view.table)
            .map_err(|r| r.error)
            .expect("the authority's mapping for the retained range");
        let launch = PagedAttentionLaunch::new(
            layer,
            1,
            ROWS - 1,
            retained.start,
            retained.end - retained.start,
        )
        .expect("a launch over the retained range");
        // **A launch must name the base the mapping describes.** The kernel
        // treats `history_base` as logical page zero, so a launch declaring a
        // different base would read the published table through an offset
        // nothing wrote against. With a wrapped ring the two differ, which is
        // why this is checked here.
        if retained.start > 0 {
            // A narrower history than the published one, page-aligned and
            // legal on its own terms: base 64 over 36 rows still covers
            // position 99. It is refused because it is not the base the
            // mapping describes, not because it is out of range.
            let mismatched =
                PagedAttentionLaunch::new(layer, 1, ROWS - 1, 64, ROWS - 64).expect("a launch");
            let query = bf16_bytes((HEADS * 64) as usize, 0x7778);
            assert!(
                run.attend(&stream, &mismatched, query).is_err(),
                "a launch whose history base is not the published one was accepted"
            );
        }
        let query = bf16_bytes((HEADS * 64) as usize, 0x7777);
        let out = run
            .attend(&stream, &launch, query)
            .map_err(|r| r.error)
            .expect("the decode runs");
        run.close(ledger).map_err(|r| r.error).expect("close");
        (out, retained.start, sequence.committed_rows())
    };

    let mut ledger = measured_ledger(&ctx);
    let (wrapped, wrapped_base, wrapped_rows) = run_case(4, &mut ledger);
    let (flat, flat_base, flat_rows) = run_case(8, &mut ledger);

    // The premise: one of these really did wrap and the other really did not.
    assert_eq!(
        (wrapped_base, wrapped_rows),
        (48, ROWS),
        "the four-page case was supposed to reclaim"
    );
    assert_eq!(
        (flat_base, flat_rows),
        (0, ROWS),
        "the eight-page case was not supposed to reclaim"
    );
    assert!(
        wrapped.iter().any(|b| *b != 0),
        "the wrapped decode produced nothing"
    );
    assert_eq!(
        wrapped, flat,
        "the same logical history gave different answers on two physical layouts"
    );
    assert!(ledger.outstanding().is_empty());
}

/// Task 0038 acceptance 2: abort, truncate and re-append, on hardware.
///
/// The transaction shapes the authority owns, exercised where the bytes really
/// are. Three properties, each checked by what attention *answers* rather than
/// by what a counter says:
///
/// 1. An aborted transaction leaves the committed history untouched — the same
///    decode gives the same bytes before and after, and the rows it staged are
///    not addressable.
/// 2. A truncation drops exactly its suffix: a decode over the truncated prefix
///    equals the decode that prefix gave before the suffix ever existed.
/// 3. A re-append after truncation lands where the truncated rows were, and the
///    answer follows the new rows rather than the old ones.
#[test]
fn abort_truncate_and_reappend_hold_on_device() {
    let _guard = one_at_a_time();
    if device_count().expect("enumerate") == 0 {
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    let ctx = RankContext::acquire(RankId(0), 0).expect("acquire device 0");
    let stream = Stream::new(&ctx).expect("a stream");
    let mut ledger = measured_ledger(&ctx);
    let (mut run, mut sequence) = prepared(&mut ledger, &ctx, &stream, 12);

    let at_twelve = decode_at(&mut run, &stream, &sequence, 11);

    // (1) An aborted transaction changes nothing a decode can see.
    let txn = sequence.begin().expect("a transaction");
    let staged = sequence.stage(txn, 4).expect("stage");
    let placements = sequence.placements(&staged, 0).expect("placements");
    let view = sequence.page_view(0).expect("a view");
    run.publish_page_table(&stream, view.base, view.table)
        .map_err(|r| r.error)
        .expect("a mapping covering the staged rows");
    run.write_rows(
        &stream,
        &placements,
        bf16_bytes(4 * row_bytes(), 0x9001),
        bf16_bytes(4 * row_bytes(), 0x9002),
    )
    .map_err(|r| r.error)
    .expect("the write fits");
    sequence.abort(txn).expect("abort");
    assert_eq!(
        sequence.committed_rows(),
        12,
        "the abort moved the frontier"
    );
    assert!(
        sequence.placement_of(0, 12).is_err(),
        "a row the abort discarded is still addressable"
    );
    assert_eq!(
        decode_at(&mut run, &stream, &sequence, 11),
        at_twelve,
        "an aborted transaction changed what the committed history answers"
    );

    // (2) A truncation drops exactly its suffix.
    let at_eight = decode_at(&mut run, &stream, &sequence, 7);
    sequence.truncate(8).expect("truncate to eight rows");
    assert_eq!(sequence.committed_rows(), 8);
    assert_eq!(
        decode_at(&mut run, &stream, &sequence, 7),
        at_eight,
        "a truncation changed the prefix it was supposed to keep"
    );
    // The rows above the prefix are gone as far as the authority is concerned,
    // and a launch that reached for them would be refused by the run as well.
    assert!(sequence.placement_of(0, 8).is_err());

    // (3) Re-appending lands where the truncated rows were, and the answer
    // follows the new bytes. Different seed, so an answer that had not changed
    // would mean the old rows were still being read.
    append_through_authority(&mut sequence, &mut run, &stream, 4, 0x9101);
    assert_eq!(sequence.committed_rows(), 12);
    let after = decode_at(&mut run, &stream, &sequence, 11);
    assert_ne!(
        after, at_twelve,
        "the decode still answers with the rows the truncation dropped"
    );
    assert!(
        after.iter().any(|b| *b != 0),
        "the re-appended decode produced nothing"
    );

    run.close(&mut ledger).map_err(|r| r.error).expect("close");
    assert!(ledger.outstanding().is_empty());
}
