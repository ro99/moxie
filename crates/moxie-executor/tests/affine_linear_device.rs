//! Task 0028 acceptance 2-4: a canonical affine weight executes on real
//! hardware and matches an independent host oracle.
//!
//! **This is not model support and no quality claim follows from it.** The
//! weights and activations here are written by this file. What it shows is that
//! the shared W4A16/W8A16 path carries a canonical INT4 or INT8 tensor from
//! three admitted device ranges to a computed answer that agrees with task
//! 0024's decoder. Output quality is O2 and needs paired output against the
//! released model.
//!
//! **The oracle shares no code path with the kernel.** It reconstructs the
//! whole weight on the host through `AffineTensor::reconstruct`, rounds each
//! value to BF16, and multiplies in ascending order with FP32 accumulation. The
//! kernel never materializes a whole weight at all. A comparison whose two
//! sides come from one implementation is the shape this repository has produced
//! repeatedly; these two do not meet anywhere.
//!
//! **The tolerance.** The predeclared gate is 2 ULP of BF16 at the oracle's
//! magnitude. The owner added a second clause on 2026-09-14 after this file
//! measured the first one: an element also passes when the difference is within
//! `2^-8 * sum|x_k * W_k|`, the reduction's own resolution. That clause fires
//! only where the result has cancelled below what a BF16 output can express --
//! at 33x1024x3072 the worst element's result is 2.2e-5 against a term sum of
//! 44.8, and no reordered FP32 reduction can meet the first clause there. Both
//! bounds are computed and both are reported.
#![cfg(feature = "driver")]

use std::sync::{Mutex, MutexGuard};

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_executor::affine_linear::{AffineLaunch, AffineLinearRun, ResidentAffineWeight};
use moxie_executor::residency::DeviceResidency;
use moxie_executor::{ChunkSource, drain_reads, select_affine_linear_kernel};
use moxie_format::affine::{
    AffineDescriptor, AffineTensor, Grouping, IntWidth, ZeroPoints, pack_row,
};
use moxie_format::scale::{ScaleDtype, ScaleValues};
use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger, LogicalRange,
    ResidencyAuthority, ResidencyLease, ResidencyRequest, TensorSlot, TurnId, UseClass,
};
use moxie_types::{RankId, Scope, WeightPrecision};

/// A `RankContext` is exclusive per device and `cargo test` runs a binary's
/// tests in parallel threads. Serialising them is not a workaround: the
/// exclusivity is the property task 0007 established deliberately.
static DEVICE: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The three canonical components, as the bytes a device reads.
struct Components {
    codes: Vec<u8>,
    scales: Vec<u8>,
    zero_points: Option<Vec<u8>>,
    group_index: Option<Vec<u8>>,
}

/// Serves the three components out of memory, in the authority's own chunk
/// vocabulary. A real artifact reaches the same authority through `ShardSource`;
/// what differs is where the bytes come from, which is the point of the trait.
struct Fixture {
    artifact: ArtifactId,
    components: Components,
}

impl ChunkSource for Fixture {
    fn read_chunk(&mut self, chunk: &ChunkId, into: &mut [u8]) -> moxie_types::Result<()> {
        assert_eq!(chunk.artifact(), &self.artifact);
        let whole: &[u8] = match chunk.slot().role() {
            "codes" => &self.components.codes,
            "scales" => &self.components.scales,
            "zero_points" => self
                .components
                .zero_points
                .as_deref()
                .expect("a symmetric tensor has no zero-point component"),
            "group_index" => self
                .components
                .group_index
                .as_deref()
                .expect("a contiguous tensor has no group-index component"),
            other => panic!("no component is bound to role {other:?}"),
        };
        let start = chunk.range().offset_bytes() as usize;
        into.copy_from_slice(&whole[start..start + into.len()]);
        Ok(())
    }
}

/// Encode the scale table in the source's own scalar encoding.
fn scale_bytes(values: &ScaleValues) -> Vec<u8> {
    match values {
        ScaleValues::F16(v) | ScaleValues::Bf16(v) => {
            v.iter().flat_map(|s| s.to_le_bytes()).collect()
        }
        ScaleValues::F32(v) => v.iter().flat_map(|s| s.to_le_bytes()).collect(),
    }
}

fn components(tensor: &AffineTensor) -> Components {
    Components {
        codes: tensor.codes().to_vec(),
        scales: scale_bytes(tensor.scales()),
        zero_points: match tensor.zero_points() {
            ZeroPoints::Symmetric => None,
            ZeroPoints::PerGroup(z) => Some(z.iter().flat_map(|v| v.to_le_bytes()).collect()),
        },
        group_index: tensor
            .descriptor()
            .group_index
            .as_ref()
            .map(|map| map.iter().flat_map(|group| group.to_le_bytes()).collect()),
    }
}

/// A deterministic unit sequence. Written here so the fixture is a fact rather
/// than a seed nobody can reproduce.
struct Sequence(u64);

impl Sequence {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

/// Build one canonical tensor whose codes cover the width's **whole** signed
/// range and whose zero points, when present, span both signs.
///
/// M3's exit gate requires whole-range coverage and a random fixture does not
/// guarantee it. Every third position of the flattened tensor walks the code
/// range in order; since three is coprime to both 16 and 256, those positions
/// cycle through every code of either width, spread across rows and columns
/// rather than bunched at the start of row zero. The rest are drawn. The
/// coverage is then *asserted* from the tensor, never from this comment.
#[allow(clippy::too_many_arguments)]
fn tensor(
    width: IntWidth,
    out_features: usize,
    in_features: usize,
    grouping: Grouping,
    asymmetric: bool,
    scale_dtype: ScaleDtype,
    mapped: bool,
    seed: u64,
) -> AffineTensor {
    let mut descriptor = AffineDescriptor {
        width,
        out_features,
        in_features,
        grouping,
        group_index: None,
        scale_dtype,
    };
    if mapped {
        let Grouping::Contiguous { size } = grouping else {
            panic!("mapped fixture requires grouped scales");
        };
        let groups = in_features.div_ceil(size as usize);
        descriptor.group_index = Some(
            (0..in_features)
                .map(|k| (groups - 1 - k / size as usize) as u32)
                .collect(),
        );
    }
    let (lo, hi) = width.code_range();
    let span = (hi - lo + 1) as usize;
    let mut rng = Sequence(seed);
    let mut codes = Vec::with_capacity(descriptor.code_bytes().unwrap());
    for o in 0..out_features {
        let row: Vec<i32> = (0..in_features)
            .map(|k| {
                let flat = o * in_features + k;
                if flat.is_multiple_of(3) {
                    lo + (flat % span) as i32
                } else {
                    lo + (rng.next() * span as f32) as i32 % span as i32
                }
            })
            .collect();
        codes.extend_from_slice(&pack_row(width, &row).unwrap());
    }
    let entries = descriptor.group_entries().unwrap();
    let raw: Vec<f32> = (0..entries)
        .map(|i| (0.01 + rng.next() * 0.05) * if i.is_multiple_of(2) { 1.0 } else { -1.0 })
        .collect();
    let scales = match scale_dtype {
        ScaleDtype::F32 => ScaleValues::F32(raw),
        ScaleDtype::Bf16 => ScaleValues::Bf16(raw.iter().map(|s| to_bf16_bits(*s)).collect()),
        ScaleDtype::F16 => ScaleValues::F16(raw.iter().map(|s| half_bits(*s)).collect()),
    };
    let zero_points = if asymmetric {
        ZeroPoints::PerGroup(
            (0..entries)
                .map(|i| (lo + ((i * 7) % span) as i32) as i16)
                .collect(),
        )
    } else {
        ZeroPoints::Symmetric
    };
    AffineTensor::new(descriptor, codes, scales, zero_points).expect("a valid canonical tensor")
}

/// Round to IEEE binary16, for the F16 scale lane. Only ever used on the small
/// finite values of either sign this fixture generates, and the result is checked by
/// `ScaleValues::validate` before it can reach a device.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    assert!(
        (1..=30).contains(&exponent),
        "{value} is outside normal f16"
    );
    let mantissa = bits & 0x007F_FFFF;
    let mut half = sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16);
    // Round to nearest even on the dropped 13 bits.
    let dropped = mantissa & 0x1FFF;
    if dropped > 0x1000 || (dropped == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    half
}

/// The whole reconstruction, then a BF16 matmul with FP32 accumulation.
///
/// Returns the FP32 result and the sum of the absolute terms, which is the
/// reduction's own scale and the denominator of the owner's second clause.
fn oracle(tensor: &AffineTensor, x: &[f32], rows: usize) -> (Vec<f32>, Vec<f32>) {
    let descriptor = tensor.descriptor();
    let (out_features, in_features) = (descriptor.out_features, descriptor.in_features);
    let weight = tensor.reconstruct().expect("the tensor reconstructs");
    let weight: Vec<f32> = weight.iter().map(|w| bf16_round(*w)).collect();
    let mut y = vec![0f32; rows * out_features];
    let mut terms = vec![0f32; rows * out_features];
    for m in 0..rows {
        for n in 0..out_features {
            let mut acc = 0f32;
            let mut abs = 0f32;
            for k in 0..in_features {
                let term = x[m * in_features + k] * weight[n * in_features + k];
                acc += term;
                abs += term.abs();
            }
            y[m * out_features + n] = acc;
            terms[m * out_features + n] = abs;
        }
    }
    (y, terms)
}

/// One ULP of BF16 at `value`'s magnitude.
fn bf16_ulp(value: f32) -> f32 {
    let exponent = (value.abs().to_bits() >> 23) & 0xFF;
    if exponent <= 7 {
        // Subnormal territory for the 8-bit significand: the smallest step.
        return f32::from_bits(1);
    }
    f32::from_bits((exponent - 7) << 23)
}

#[derive(Debug)]
struct Measured {
    worst_ulp: f64,
    /// Elements the first clause alone did not cover.
    cancelled: usize,
    checked: usize,
}

/// The owner's two-clause gate, applied element by element.
///
/// Returns the failure rather than panicking, so the gate itself can be tested
/// in both directions instead of only in the direction the fixtures happen to
/// take.
fn compare(got: &[u8], want: &[f32], terms: &[f32]) -> Result<Measured, String> {
    assert_eq!(got.len(), want.len() * 2);
    let mut worst_ulp = 0f64;
    let mut cancelled = 0usize;
    for (i, expected) in want.iter().enumerate() {
        let device =
            f32::from_bits(u32::from(u16::from_le_bytes([got[i * 2], got[i * 2 + 1]])) << 16);
        let reference = f32::from_bits(u32::from(to_bf16_bits(*expected)) << 16);
        let difference = (device - reference).abs();
        let in_ulp = f64::from(difference) / f64::from(bf16_ulp(reference));
        worst_ulp = worst_ulp.max(in_ulp);
        if in_ulp <= 2.0 {
            continue;
        }
        // The owner's second clause: the reduction's own resolution. BF16
        // carries eight significand bits, so `sum|terms| / 256` is the smallest
        // difference a BF16 output of this reduction could express at all.
        let resolution = terms[i] / 256.0;
        if difference > resolution {
            return Err(format!(
                "element {i}: |{device} - {reference}| = {difference} exceeds both 2 ULP \
                 ({}) and the reduction's resolution ({resolution}); sum|terms| = {}",
                2.0 * bf16_ulp(reference),
                terms[i]
            ));
        }
        cancelled += 1;
    }
    Ok(Measured {
        worst_ulp,
        cancelled,
        checked: want.len(),
    })
}

/// Acquire one component into the device cache and hold its lease.
fn resident<'ctx>(
    authority: &mut ResidencyAuthority,
    source: &mut Fixture,
    residency: &mut DeviceResidency<'ctx>,
    stream: &Stream<'ctx>,
    scope: Scope,
    role: &str,
    len: u64,
) -> ResidencyLease {
    let chunk = ChunkId::new(
        source.artifact.clone(),
        TensorSlot::tensor(role).unwrap(),
        LogicalRange::new(0, len).unwrap(),
        1,
    );
    let acquired = authority
        .acquire(AcquireRequest {
            chunk: &chunk,
            destination: scope,
            now: 0,
            deadline: u64::MAX,
            class: UseClass::demand(Content::DenseSpine),
            turn: TurnId::new(1),
        })
        .unwrap_or_else(|refused| panic!("acquiring {role}: {}", refused.error));
    match acquired {
        Acquired::Ready(lease) => lease,
        Acquired::Pending { lease, work, .. } => {
            let uploads = drain_reads(authority, source, work).expect("the component reads");
            for order in &uploads {
                residency
                    .perform_upload(authority, stream, order)
                    .unwrap_or_else(|e| panic!("uploading {role}: {e}"));
            }
            lease
        }
    }
}

struct Case {
    name: &'static str,
    width: IntWidth,
    grouping: Grouping,
    asymmetric: bool,
    scale_dtype: ScaleDtype,
    mapped: bool,
    out_features: usize,
    in_features: usize,
    rows: usize,
}

/// The contract's two required tensors, plus the crossed pair that makes the
/// kernel's sharing more than a name.
///
/// (a) and (b) are acceptance 2's. (c) and (d) swap the group rule and the
/// zero-point section across the two widths: if the group rule or the
/// zero-point handling were secretly tied to the code width, one of those two
/// fails. The row counts are 5, 3, 17 and 7 -- none a multiple of the 16-wide
/// tile -- and 100 and 300 input features leave a short final group.
const CASES: &[Case] = &[
    Case {
        name: "a: int4 group-32 asymmetric, bf16 scales",
        width: IntWidth::Int4,
        grouping: Grouping::Contiguous { size: 32 },
        asymmetric: true,
        scale_dtype: ScaleDtype::Bf16,
        mapped: false,
        out_features: 96,
        in_features: 100,
        rows: 5,
    },
    Case {
        name: "b: int8 group-128 symmetric, f32 scales",
        width: IntWidth::Int8,
        grouping: Grouping::Contiguous { size: 128 },
        asymmetric: false,
        scale_dtype: ScaleDtype::F32,
        mapped: false,
        out_features: 64,
        in_features: 300,
        rows: 3,
    },
    Case {
        name: "c: int4 group-128 symmetric, f16 scales",
        width: IntWidth::Int4,
        grouping: Grouping::Contiguous { size: 128 },
        asymmetric: false,
        scale_dtype: ScaleDtype::F16,
        mapped: false,
        out_features: 48,
        in_features: 300,
        rows: 17,
    },
    // The shape that made the owner's second clause necessary, and the same
    // 3,072 by 1,024 geometry as the real module task 0025 published. A
    // 1,024-deep reduction over 101,376 outputs produces a handful whose result
    // has cancelled to six orders of magnitude below the sum of its own terms,
    // and no reordered FP32 reduction can land within 2 ULP of *that*. The case
    // is here so the clause is exercised by the gate rather than by a note --
    // at 33 by 1,024 by 1,024 it never fires, which is how few elements reach
    // the condition.
    Case {
        name: "e: int4 group-32 asymmetric, 3072x1024 at reduction depth 1024",
        width: IntWidth::Int4,
        grouping: Grouping::Contiguous { size: 32 },
        asymmetric: true,
        scale_dtype: ScaleDtype::Bf16,
        mapped: false,
        out_features: 3072,
        in_features: 1024,
        rows: 33,
    },
    Case {
        name: "d: int8 per-channel asymmetric, bf16 scales",
        width: IntWidth::Int8,
        grouping: Grouping::PerOutputChannel,
        asymmetric: true,
        scale_dtype: ScaleDtype::Bf16,
        mapped: false,
        out_features: 33,
        in_features: 100,
        rows: 7,
    },
    Case {
        name: "f: int4 group-32 asymmetric with activation-order map",
        width: IntWidth::Int4,
        grouping: Grouping::Contiguous { size: 32 },
        asymmetric: true,
        scale_dtype: ScaleDtype::F16,
        mapped: true,
        out_features: 48,
        in_features: 100,
        rows: 5,
    },
];

#[test]
fn a_canonical_affine_weight_executes_on_every_visible_device() {
    let _serial = one_at_a_time();
    let count = device_count().expect("the device lane enumerates devices");
    assert!(count > 0, "the device lane requires real hardware");
    let mut exercised = Vec::new();
    for ordinal in 0..count {
        let capability = query_device(ordinal).expect("a device capability");
        let ctx = RankContext::acquire(RankId(28_000 + ordinal), ordinal)
            .expect("an exclusive rank context");
        let stream = Stream::new(&ctx).expect("a stream");
        for case in CASES {
            run_case(&ctx, &stream, &capability, case);
        }
        exercised.push(format!("{} ({})", capability.uuid, capability.sm()));
    }
    eprintln!(
        "task0028: {} case(s) on {} device(s): {}. Synthetic weights and activations; \
         this is execution, not model support.",
        CASES.len(),
        count,
        exercised.join(", ")
    );
}

fn run_case<'ctx>(
    ctx: &'ctx RankContext,
    stream: &Stream<'ctx>,
    capability: &moxie_types::DeviceCapability,
    case: &Case,
) {
    let tensor = tensor(
        case.width,
        case.out_features,
        case.in_features,
        case.grouping,
        case.asymmetric,
        case.scale_dtype,
        case.mapped,
        0x0028_2026,
    );
    assert_code_coverage(&tensor, case);

    let launch = AffineLaunch::for_tensor(&tensor, case.rows as u64).expect("the launch derives");
    let catalogue = moxie_kernels::affine_linear_catalogue();
    let descriptor = select_affine_linear_kernel(
        &catalogue,
        capability,
        WeightPrecision::expect(case.width.precision()),
        &launch,
    )
    .unwrap_or_else(|e| panic!("{}: {e}", case.name));
    assert_eq!(
        descriptor.id.0,
        format!(
            "{}-linear-v1-{}",
            moxie_kernels::profile_name(case.width.precision()),
            capability.sm()
        )
    );

    let mut rng = Sequence(0x5eed_0028);
    let x: Vec<f32> = (0..case.rows * case.in_features)
        .map(|_| bf16_round(rng.next() - 0.5))
        .collect();
    let x_bytes: Vec<u8> = x
        .iter()
        .flat_map(|v| to_bf16_bits(*v).to_le_bytes())
        .collect();

    let scope = Scope::Device(ctx.uuid());
    let code_bytes = launch.code_bytes().unwrap();
    let scale_bytes_len = launch.scale_bytes().unwrap();
    let zero_bytes = launch.zero_point_bytes().unwrap();
    let group_index_bytes = launch.group_index_bytes().unwrap();
    // The cache is capped at the three components, each rounded up to the
    // cache's own 256-byte alignment and nothing more, so `capacity()` is a
    // measured bound rather than a generous one.
    let align = |bytes: u64| bytes.div_ceil(256) * 256;
    let weight_bytes = align(code_bytes)
        + align(scale_bytes_len)
        + align(zero_bytes.unwrap_or(0))
        + align(group_index_bytes.unwrap_or(0));
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 << 20, 1 << 20).unwrap(),
        CapacitySnapshot::new(scope, 64 << 20, 1 << 20).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("affine linear components", weight_bytes)
            .device(ctx.uuid(), weight_bytes),
    )
    .unwrap();
    let mut residency = DeviceResidency::create(ctx, &mut authority).unwrap();
    let mut source = Fixture {
        artifact: ArtifactId::new("sha256:task0028-synthetic-affine-fixture").unwrap(),
        components: components(&tensor),
    };

    let codes = resident(
        &mut authority,
        &mut source,
        &mut residency,
        stream,
        scope,
        "codes",
        code_bytes,
    );
    let scales = resident(
        &mut authority,
        &mut source,
        &mut residency,
        stream,
        scope,
        "scales",
        scale_bytes_len,
    );
    let zero_points = zero_bytes.map(|bytes| {
        resident(
            &mut authority,
            &mut source,
            &mut residency,
            stream,
            scope,
            "zero_points",
            bytes,
        )
    });
    let group_index = group_index_bytes.map(|bytes| {
        resident(
            &mut authority,
            &mut source,
            &mut residency,
            stream,
            scope,
            "group_index",
            bytes,
        )
    });

    let mut run = AffineLinearRun::admit(&mut ledger, ctx, descriptor, launch)
        .unwrap_or_else(|refused| panic!("{}: {}", case.name, refused.error));

    // Acceptance 4: the launch's whole device footprint -- the three resident
    // components plus the per-step arena -- against what one BF16 copy of this
    // weight would need. A memory bound, not a speed measurement: O6 and O7 are
    // open and nothing here is timed.
    let footprint = residency.capacity() + run.arena_bytes();
    let dequantized = launch.dequantized_weight_bytes().unwrap();
    assert!(
        footprint < dequantized,
        "{}: the launch holds {footprint} device byte(s) and a BF16 copy of the weight \
         alone would need {dequantized}",
        case.name
    );

    let (weight, x_bytes) = if launch.mapped() {
        let refused = run
            .run(
                stream,
                &authority,
                &residency,
                ResidentAffineWeight {
                    codes,
                    scales,
                    zero_points,
                    group_index: None,
                },
                x_bytes,
            )
            .expect_err("a mapped launch must not run without its admitted map");
        assert!(!refused.retained_operands());
        assert!(refused.error.to_string().contains("group-index component"));
        let returned = refused
            .weight
            .expect("the pre-launch refusal returns weight");
        (
            ResidentAffineWeight {
                codes: returned.codes,
                scales: returned.scales,
                zero_points: returned.zero_points,
                group_index,
            },
            refused
                .activations
                .expect("the pre-launch refusal returns activations"),
        )
    } else {
        (
            ResidentAffineWeight {
                codes,
                scales,
                zero_points,
                group_index,
            },
            x_bytes,
        )
    };
    let completed = run
        .run(stream, &authority, &residency, weight, x_bytes)
        .unwrap_or_else(|refused| panic!("{}: {}", case.name, refused.error));
    let got = completed.output;
    let weight = completed.weight;

    let (want, terms) = oracle(&tensor, &x, case.rows);
    let measured =
        compare(&got, &want, &terms).unwrap_or_else(|why| panic!("{}: {why}", case.name));
    eprintln!(
        "task0028 {} on {} {}: {} element(s), worst {:.3} ULP, {} covered by the \
         cancellation clause; {} device byte(s) against {} for a BF16 copy",
        case.name,
        capability.uuid,
        capability.sm(),
        measured.checked,
        measured.worst_ulp,
        measured.cancelled,
        footprint,
        dequantized
    );

    run.close(&mut ledger).expect("the run closes");
    for lease in [
        Some(weight.codes),
        Some(weight.scales),
        weight.zero_points,
        weight.group_index,
    ]
    .into_iter()
    .flatten()
    {
        authority
            .release(lease)
            .expect("the component lease retires");
    }
    authority.end_turn(TurnId::new(1));
    authority.retire_all(scope);
    residency
        .close(&mut authority)
        .expect("the backing returns");
    authority.close(&mut ledger).expect("the authority closes");
    assert!(
        ledger.outstanding().is_empty(),
        "{}: {:?} byte(s) stayed charged",
        case.name,
        ledger.outstanding()
    );
}

/// Acceptance 3, asserted from the tensor rather than from the generator.
fn assert_code_coverage(tensor: &AffineTensor, case: &Case) {
    let descriptor = tensor.descriptor();
    let (lo, hi) = case.width.code_range();
    let mut seen = vec![false; (hi - lo + 1) as usize];
    for o in 0..descriptor.out_features {
        for k in 0..descriptor.in_features {
            let code = tensor.code(o, k).unwrap();
            assert!((lo..=hi).contains(&code));
            seen[(code - lo) as usize] = true;
        }
    }
    if case.width == IntWidth::Int4 {
        // M3's exit gate: every one of the sixteen INT4 codes, not a sample.
        assert!(
            seen.iter().all(|s| *s),
            "{}: the INT4 codes do not span the whole range",
            case.name
        );
    } else {
        assert!(
            seen[0] && seen[seen.len() - 1],
            "{}: INT8 -128 and 127",
            case.name
        );
        assert!(
            seen.iter().filter(|s| **s).count() > 200,
            "{}: the INT8 codes barely move",
            case.name
        );
    }
    if let ZeroPoints::PerGroup(z) = tensor.zero_points() {
        assert!(
            z.iter().any(|v| *v < 0) && z.iter().any(|v| *v > 0),
            "{}: the zero points do not span both signs",
            case.name
        );
    }
}

/// Task 0028's review, finding 4: admission must re-apply selection's predicate.
///
/// `admit` is a public entry point that takes a descriptor, and it trusted
/// whatever it was handed. A W8A16 descriptor bound to an INT4 geometry would
/// have launched the shared kernel with `code_bits = 4` against component sizes
/// computed for eight-bit codes.
#[test]
fn admission_refuses_a_descriptor_that_does_not_serve_the_geometry() {
    let _serial = one_at_a_time();
    assert!(device_count().expect("devices") > 0, "real hardware");
    let capability = query_device(0).expect("a capability");
    let ctx = RankContext::acquire(RankId(28_700), 0).expect("a rank context");
    let case = &CASES[0];
    let tensor = tensor(
        case.width,
        case.out_features,
        case.in_features,
        case.grouping,
        case.asymmetric,
        case.scale_dtype,
        case.mapped,
        0x0028_2026,
    );
    let launch = AffineLaunch::for_tensor(&tensor, case.rows as u64).expect("the launch derives");
    let catalogue = moxie_kernels::affine_linear_catalogue();
    // The *other* width's descriptor, for this device. A real catalogue entry,
    // correctly qualified, that simply does not serve this geometry.
    let wrong = catalogue
        .descriptors()
        .iter()
        .find(|d| d.id.0 == format!("w8a16-linear-v1-{}", capability.sm()))
        .expect("the w8a16 entry")
        .clone();
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 8 << 20, 1 << 20).unwrap(),
        CapacitySnapshot::new(Scope::Device(ctx.uuid()), 8 << 20, 1 << 20).unwrap(),
    ])
    .unwrap();
    let refused = AffineLinearRun::admit(&mut ledger, &ctx, wrong, launch)
        .expect_err("admission must refuse a descriptor that does not serve this geometry");
    assert_eq!(refused.error.kind(), "unsupported_kernel");
    assert!(
        refused.error.to_string().contains("cannot bind"),
        "{}",
        refused.error
    );
    // Refused before anything was allocated, and nothing stayed charged.
    assert!(refused.reservation.is_none());
    assert!(
        ledger.outstanding().is_empty(),
        "{:?}",
        ledger.outstanding()
    );
}

/// Task 0028's review, finding 1: a lease resident on another GPU must refuse.
///
/// One authority can hold a cache on **every** device, and each cache has its
/// own offsets. `component()` checked the authority and the byte length and
/// never the device, so a 5060 Ti launch accepted 3090 leases, resolved their
/// offsets inside the 5060 Ti's allocation and returned a confident wrong
/// answer over whatever happened to live there. Task 0021's review found the
/// same shape one layer down, which is why this one is a test rather than a
/// comment.
#[test]
fn a_component_resident_on_another_device_is_refused() {
    let _serial = one_at_a_time();
    let count = device_count().expect("devices");
    if count < 2 {
        eprintln!(
            "SKIP the cross-device refusal: this machine has {count} GPU(s) and the case \
             needs two. A missing device is a blocked lane, never acceptance."
        );
        return;
    }
    let case = &CASES[0];
    let tensor = tensor(
        case.width,
        case.out_features,
        case.in_features,
        case.grouping,
        case.asymmetric,
        case.scale_dtype,
        case.mapped,
        0x0028_2026,
    );
    let launch = AffineLaunch::for_tensor(&tensor, case.rows as u64).expect("the launch derives");

    // Two contexts, two residency backings, **one** authority -- which is the
    // configuration the check exists for.
    let host = RankContext::acquire(RankId(28_800), 0).expect("the first device");
    let other = RankContext::acquire(RankId(28_801), 1).expect("the second device");
    let host_stream = Stream::new(&host).expect("a stream");
    let other_stream = Stream::new(&other).expect("a stream");

    let code_bytes = launch.code_bytes().unwrap();
    let scale_bytes_len = launch.scale_bytes().unwrap();
    let zero_bytes = launch.zero_point_bytes().unwrap();
    let align = |bytes: u64| bytes.div_ceil(256) * 256;
    let cap = align(code_bytes) + align(scale_bytes_len) + align(zero_bytes.unwrap_or(0));
    let mut ledger = Ledger::new([
        CapacitySnapshot::new(Scope::Host, 64 << 20, 1 << 20).unwrap(),
        CapacitySnapshot::new(Scope::Device(host.uuid()), 64 << 20, 1 << 20).unwrap(),
        CapacitySnapshot::new(Scope::Device(other.uuid()), 64 << 20, 1 << 20).unwrap(),
    ])
    .unwrap();
    let mut authority = ResidencyAuthority::open(
        &mut ledger,
        &ResidencyRequest::new("two device caches", cap)
            .device(host.uuid(), cap)
            .device(other.uuid(), cap),
    )
    .unwrap();
    let host_residency = DeviceResidency::create(&host, &mut authority).unwrap();
    let mut other_residency = DeviceResidency::create(&other, &mut authority).unwrap();
    let mut source = Fixture {
        artifact: ArtifactId::new("sha256:task0028-cross-device-fixture").unwrap(),
        components: components(&tensor),
    };

    // Every component made resident on the **second** device.
    let elsewhere = Scope::Device(other.uuid());
    let codes = resident(
        &mut authority,
        &mut source,
        &mut other_residency,
        &other_stream,
        elsewhere,
        "codes",
        code_bytes,
    );
    let scales = resident(
        &mut authority,
        &mut source,
        &mut other_residency,
        &other_stream,
        elsewhere,
        "scales",
        scale_bytes_len,
    );
    let zero_points = zero_bytes.map(|bytes| {
        resident(
            &mut authority,
            &mut source,
            &mut other_residency,
            &other_stream,
            elsewhere,
            "zero_points",
            bytes,
        )
    });

    let capability = query_device(0).expect("a capability");
    let kernel = select_affine_linear_kernel(
        &moxie_kernels::affine_linear_catalogue(),
        &capability,
        WeightPrecision::expect(case.width.precision()),
        &launch,
    )
    .expect("a descriptor");
    let mut run = AffineLinearRun::admit(&mut ledger, &host, kernel, launch)
        .unwrap_or_else(|refused| panic!("admission: {}", refused.error));

    let refused = run
        .run(
            &host_stream,
            &authority,
            &host_residency,
            ResidentAffineWeight {
                codes,
                scales,
                zero_points,
                group_index: None,
            },
            vec![0u8; launch.activation_bytes().unwrap() as usize],
        )
        .expect_err("a launch must not read another device's leases");
    assert_eq!(refused.error.kind(), "invalid_request");
    assert!(
        refused
            .error
            .to_string()
            .contains("another allocation's bytes"),
        "{}",
        refused.error
    );
    // Nothing was enqueued, so the operands come straight back and the caller
    // can retry them where they actually live.
    assert!(!refused.retained_operands());
    let weight = refused.weight.expect("the operands are handed back");

    run.close(&mut ledger).expect("the run closes");
    for lease in [Some(weight.codes), Some(weight.scales), weight.zero_points]
        .into_iter()
        .flatten()
    {
        authority.release(lease).expect("the lease retires");
    }
    authority.end_turn(TurnId::new(1));
    authority.retire_all(elsewhere);
    other_residency
        .close(&mut authority)
        .expect("the second backing returns");
    host_residency
        .close(&mut authority)
        .expect("the first backing returns");
    authority.close(&mut ledger).expect("the authority closes");
    assert!(
        ledger.outstanding().is_empty(),
        "{:?}",
        ledger.outstanding()
    );
    eprintln!(
        "task0028: a component resident on {} was refused by a launch on {}",
        other.uuid(),
        host.uuid()
    );
}

/// The owner's second clause, tested in both directions and needing no device.
///
/// This is here because the acceptance fixtures above never reach it: the
/// declared 2 ULP clause covers every element of all five, worst 2.000. A guard
/// nothing exercises is the shape this repository keeps producing, so the guard
/// is driven directly, with the numbers that prompted the ruling.
///
/// Those numbers are a real measurement, not an invention: at 33 by 1,024 by
/// 3,072 with uniformly drawn codes and zero points, the worst element's oracle
/// result was 2.21729279e-5 against a term sum of 44.8413914, and the kernel
/// returned 2.28881836e-5 -- 6 ULP apart at that magnitude, and 1.6e-8 of the
/// reduction's own scale, which is below one FP32 epsilon. Both are written as
/// the BF16 bit patterns they round to, so the fixture cannot drift by a
/// decimal digit.
#[test]
fn the_cancellation_clause_covers_a_cancelled_element_and_nothing_else() {
    let reference = f32::from_bits(0x37_BA_00_00);
    let device = f32::from_bits(0x37_C0_00_00);
    let got: Vec<u8> = to_bf16_bits(device).to_le_bytes().to_vec();
    let difference = (bf16_round(device) - bf16_round(reference)).abs();
    assert!(
        f64::from(difference) / f64::from(bf16_ulp(bf16_round(reference))) > 2.0,
        "the fixture no longer misses the first clause, so it tests nothing"
    );

    // Cancelled: the result is six orders of magnitude below the sum of the
    // terms that produced it, and the difference is far inside what a BF16
    // output of that sum can express.
    let measured = compare(&got, &[reference], &[44.841_39]).expect("the second clause covers it");
    assert_eq!(measured.cancelled, 1);
    assert_eq!(measured.checked, 1);
    assert!(measured.worst_ulp > 2.0);

    // The same difference on a well-conditioned reduction is a real failure:
    // the clause must not become a blanket tolerance. `256 * difference` is the
    // exact boundary, so a term sum just below it has to be refused.
    let boundary = 256.0 * difference;
    assert!(compare(&got, &[reference], &[boundary]).is_ok());
    let refused = compare(&got, &[reference], &[boundary * 0.999]).unwrap_err();
    assert!(refused.contains("exceeds both 2 ULP"), "{refused}");
}
