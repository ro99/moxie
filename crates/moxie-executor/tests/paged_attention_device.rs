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
use moxie_executor::{AttentionLayer, PageGeometry, PagedAttentionLaunch, PagedAttentionRun};
use moxie_kernels::cpu_expert::to_bf16_bits;
use moxie_memory::{CapacitySnapshot, Ledger};
use moxie_plan::Visibility;
use moxie_types::{Error, RankId, Scope};

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
) -> PagedAttentionRun<'ctx> {
    let mut run = PagedAttentionRun::admit(
        ledger,
        ctx,
        descriptor_for(ctx),
        geometry(),
        HEADS,
        MAX_ROWS,
    )
    .map_err(|r| r.error)
    .expect("admission fits a measured device");
    run.publish_page_table(
        stream,
        (0..geometry().pages).rev().map(|p| p as u32).collect(),
    )
    .map_err(|r| r.error)
    .expect("a reversed mapping is a mapping");
    if rows > 0 {
        let keys = bf16_bytes(rows as usize * row_bytes(), 0x37_0001);
        let values = bf16_bytes(rows as usize * row_bytes(), 0x37_0002);
        run.append(stream, rows, keys, values)
            .map_err(|r| r.error)
            .expect("the append fits");
    }
    run
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
        PagedAttentionRun::admit(&mut ledger, &ctx, narrowed, geometry(), HEADS, MAX_ROWS).is_err(),
        "a descriptor whose shape bounds exclude this head dimension was accepted"
    );
    let renamed = {
        let mut d = descriptor_for(&ctx);
        d.symbols[0] = moxie_types::KernelSymbol(moxie_kernels::AFFINE_LINEAR.to_string());
        d
    };
    assert!(
        PagedAttentionRun::admit(&mut ledger, &ctx, renamed, geometry(), HEADS, MAX_ROWS).is_err(),
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
            MAX_ROWS
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
            PagedAttentionRun::admit(&mut ledger, &ctx, descriptor, geometry(), HEADS, MAX_ROWS)
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
            MAX_ROWS
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
    )
    .map_err(|r| r.error)
    .expect("admission fits");

    // A physical page that does not exist, and one named twice. The second is
    // the dangerous one: aliased pages make an append overwrite history that is
    // still visible, and no later check could tell.
    assert!(
        run.publish_page_table(&stream, vec![0, 1, 99, 3]).is_err(),
        "a physical page outside the admitted set was accepted"
    );
    assert!(
        run.publish_page_table(&stream, vec![0, 1, 1, 3]).is_err(),
        "an aliased physical page was accepted"
    );
    assert!(
        run.publish_page_table(&stream, Vec::new()).is_err(),
        "an empty mapping was accepted"
    );
    assert!(
        run.publish_page_table(&stream, vec![0, 1, 2, 3, 0])
            .is_err(),
        "a mapping longer than the admitted pages was accepted"
    );
    // An append before any mapping exists has nowhere to write.
    let keys = bf16_bytes(row_bytes(), 1);
    let values = bf16_bytes(row_bytes(), 2);
    let refused = run
        .append(&stream, 1, keys, values)
        .expect_err("an append with no mapping was accepted");
    assert!(
        !refused.retained_source(),
        "a pre-enqueue refusal kept bytes"
    );
    assert_eq!(run.committed_rows(), 0);

    run.publish_page_table(&stream, vec![3, 2, 1, 0])
        .map_err(|r| r.error)
        .expect("a reversed mapping is a mapping");
    let keys = bf16_bytes(row_bytes(), 1);
    let values = bf16_bytes(row_bytes(), 2);
    run.append(&stream, 1, keys, values)
        .map_err(|r| r.error)
        .expect("the append fits");
    // Remapping under a live history would move rows something has attended to.
    assert!(
        run.publish_page_table(&stream, vec![0, 1, 2, 3]).is_err(),
        "the mapping changed under committed rows"
    );
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
    let mut run = prepared(&mut ledger, &ctx, &stream, 20);
    let before = run.read_rows(0, 20).expect("committed rows read back");
    let capacity = run.capacity_rows().expect("capacity");
    assert_eq!(capacity, 64, "four pages of sixteen rows");

    let cases: Vec<(&str, u64, usize, usize)> = vec![
        // label, rows, key element count, value element count
        ("no rows at all", 0, 0, 0),
        ("more rows than the pages hold", capacity + 1, 0, 0),
        (
            "keys short of their row count",
            2,
            row_bytes(),
            2 * row_bytes(),
        ),
        (
            "values short of their row count",
            2,
            2 * row_bytes(),
            row_bytes(),
        ),
    ];
    for (label, rows, keys, values) in cases {
        let refused = run
            .append(&stream, rows, bf16_bytes(keys, 7), bf16_bytes(values, 8))
            .err()
            .unwrap_or_else(|| panic!("{label} was accepted"));
        assert!(
            !refused.retained_source(),
            "{label}: a refusal before enqueue must hand the rows back"
        );
        assert_eq!(run.committed_rows(), 20, "{label}: the frontier moved");
        assert_eq!(
            run.read_rows(0, 20).expect("read back"),
            before,
            "{label}: committed bytes changed"
        );
    }

    // And a launch that claims history the frontier does not have.
    let query = bf16_bytes((HEADS * geometry().head_dim) as usize, 9);
    let refused = run
        .attend(&stream, &launch(1, 19, 21), query)
        .expect_err("a launch past the frontier was accepted");
    assert!(!refused.retained_source());
    assert_eq!(run.committed_rows(), 20);
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
    let mut run = prepared(&mut ledger, &ctx, &stream, 4);

    let keys = bf16_bytes(row_bytes(), 3);
    let values = bf16_bytes(row_bytes(), 4);
    let refused = run
        .append(&foreign, 1, keys, values)
        .expect_err("an append on a foreign stream was accepted");
    assert!(!refused.retained_source());
    assert_eq!(run.committed_rows(), 4, "the frontier moved");

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
    let mut run = prepared(&mut ledger, &ctx, &stream, 1);

    // The claim is narrow and exact: decoding does not grow what was admitted.
    // The pages, the table, the query and the output are charged once at
    // admission, and every step after that reuses them. It is *not* a claim
    // about host memory the allocator may hold, which this test cannot see.
    let charged = ledger.outstanding().len();
    let bytes = run.arena_bytes();
    let mut previous: Option<Vec<u8>> = None;
    for step in 1..=32u64 {
        let keys = bf16_bytes(row_bytes(), 0x1000 + step);
        let values = bf16_bytes(row_bytes(), 0x2000 + step);
        run.append(&stream, 1, keys, values)
            .map_err(|r| r.error)
            .expect("one row per step");
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
        assert_eq!(run.committed_rows(), step + 1);
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
