//! Task 0028 acceptance 5: a module of a published canonical artifact executes.
//!
//! **Read-only, and this is not model support.** One module of one inventoried
//! local checkpoint is repacked into a private temporary directory that is
//! removed when the test ends, reopened through the production reader, made
//! resident by the weight-residency authority, and multiplied by activations
//! this file writes. Nothing is downloaded, nothing is converted in bulk, and
//! nothing under `/fast/models` is written.
//!
//! What it adds to the synthetic device case is that the codes, scales and zero
//! points are a real quantizer's output rather than this repository's: 3,072 by
//! 1,024, group-32 asymmetric INT4, with the scale encoding the source actually
//! used. Task 0025 published these bytes and nothing executed them; this is the
//! first time a canonical INT4 tensor reaches a kernel.
//!
//! **No output-quality claim follows.** The activations are synthetic, the
//! module is one projection of one layer, and O2 needs paired output against
//! the released model.
//!
//! The artifact skips with a printed message when absent, so a fresh clone is
//! green and a missing checkpoint is never read as a passing gate.
#![cfg(feature = "driver")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use moxie_cuda::{RankContext, Stream, device_count, query_device};
use moxie_executor::affine_linear::{AffineLaunch, AffineLinearRun, ResidentAffineWeight};
use moxie_executor::residency::DeviceResidency;
use moxie_executor::{ShardSource, drain_reads, select_affine_linear_kernel};
use moxie_format::canonical::ComponentKind;
use moxie_format::payload::ZeroPointSection;
use moxie_kernels::cpu_expert::{bf16_round, to_bf16_bits};
use moxie_memory::{
    AcquireRequest, Acquired, ArtifactId, CapacitySnapshot, ChunkId, Content, Ledger, LogicalRange,
    ResidencyAuthority, ResidencyLease, ResidencyRequest, TensorSlot, TurnId, UseClass,
};
use moxie_storage::{Artifact, Shard};
use moxie_types::{RankId, Scope, WeightPrecision};

/// The same module task 0025 published, at the revision its record names.
const ROOT: &str = "/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4";
const REVISION: &str = "bc59f497520b23759ce61cc5164ca28bcc4f53bc";
const SHARD: &str = "model-00001-of-00015.safetensors";
const MODULE: &str = "model.layers.1.mlp.experts.0.down_proj";
const ROLE: &str = "model.layers.1.mlp.experts.0.down_proj.weight";
const OUT_FEATURES: usize = 3_072;
const IN_FEATURES: usize = 1_024;
/// Not a multiple of the 16-wide tile, on purpose.
const ROWS: usize = 3;

static DEVICE: Mutex<()> = Mutex::new(());

fn one_at_a_time() -> MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A private destination, removed when the test ends however it ends.
struct Destination {
    path: PathBuf,
}

impl Destination {
    fn new() -> Self {
        Self {
            path: std::env::temp_dir().join(format!(
                "moxie-task0028-real-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("a clock after 1970")
                    .as_nanos()
            )),
        }
    }
}

impl Drop for Destination {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The revision this artifact records for itself, as the hub wrote it.
fn recorded_revision(root: &Path) -> Option<String> {
    let path = root
        .join(".cache/huggingface/download")
        .join(format!("{SHARD}.metadata"));
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .next()
        .map(|line| line.trim().to_string())
}

fn selection_text() -> String {
    const ABSENT: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    format!(
        "version = 1\n\n[source]\nmodel = \"Laguna-S-2.1-AWQ-INT4\"\nrevision = \"{REVISION}\"\n\
         license = \"see LICENSE.md in the source revision\"\n\n\
         [tokenizer]\nname = \"none\"\nversion = \"not-selected\"\ndigest = \"{ABSENT}\"\n\n\
         [template]\nname = \"none\"\nversion = \"not-selected\"\ndigest = \"{ABSENT}\"\n\n\
         [architecture]\nname = \"laguna\"\nversion = \"1\"\n[architecture.metadata]\n\
         note = \"opaque here; this selection is one module and not a model\"\n\n\
         [provenance]\nscale_convention = \"affine-v1\"\nquantizer = \"none\"\n\
         calibration = \"none\"\n\n[completeness]\nstatus = \"partial\"\n\
         missing = [\"every tensor of this revision except {MODULE}\"]\n\n\
         [[tensor]]\nrole = \"{ROLE}\"\nkind = \"pack-quantized\"\nmodule = \"{MODULE}\"\n\
         width = \"int4\"\ngroup = 32\nzero_points = \"packed-along-output\"\n\
         [tensor.files]\nweight_packed = \"{SHARD}\"\nweight_scale = \"{SHARD}\"\n\
         weight_shape = \"{SHARD}\"\nweight_zero_point = \"{SHARD}\"\n"
    )
}

/// Publish the one module, through the converter's own library entry point,
/// and return the artifact identity it computed.
fn publish(destination: &Path) -> String {
    let selection =
        moxie_format::selection::parse(&selection_text()).expect("the selection parses");
    let budgets = moxie_repack::Budgets {
        total_bytes: 128 << 20,
        header_bytes: 64 << 20,
        scratch_bytes: 1 << 20,
        chunk_file_bytes: 4 << 20,
        disk_bytes: 64 << 20,
    };
    let mut sources =
        moxie_repack::open_sources(Path::new(ROOT), &budgets).expect("the source root opens");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a converter ledger");
    let report = moxie_repack::repack(
        &selection,
        &mut sources,
        destination,
        &budgets,
        &moxie_repack::write::run::Options {
            take_over_interrupted_run: false,
        },
        &moxie_repack::write::fault::Faults::default(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect("the module publishes");
    assert!(
        matches!(
            report.outcome,
            moxie_repack::write::run::Outcome::Published { .. }
        ),
        "{report:?}"
    );
    // The bytes task 0025's record names for this module, from the header
    // rather than from this test's arithmetic.
    assert_eq!(report.bytes_written, 1_966_080, "{report:?}");
    report.artifact_identity
}

/// Acquire one published component into the device cache.
#[allow(clippy::too_many_arguments)]
fn resident<'ctx>(
    authority: &mut ResidencyAuthority,
    source: &mut ShardSource,
    residency: &mut DeviceResidency<'ctx>,
    stream: &Stream<'ctx>,
    artifact: &ArtifactId,
    scope: Scope,
    role: &str,
    len: u64,
) -> ResidencyLease {
    let chunk = ChunkId::new(
        artifact.clone(),
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

#[test]
fn a_published_canonical_int4_module_executes_and_matches_its_decoder() {
    let _serial = one_at_a_time();
    let root = Path::new(ROOT);
    if !root.join(SHARD).exists() {
        eprintln!(
            "SKIP task0028's real-module case: {} is not present on this machine. \
             A missing source is a blocked real lane, never acceptance.",
            root.join(SHARD).display()
        );
        return;
    }
    assert_eq!(
        recorded_revision(root).as_deref(),
        Some(REVISION),
        "this artifact records a different revision than the task contract names"
    );
    assert!(device_count().expect("devices") > 0, "real hardware");

    let destination = Destination::new();
    let started = std::time::Instant::now();
    let artifact_identity = publish(&destination.path);
    let published_in = started.elapsed();

    // --- what the artifact says it is ---------------------------------------
    let artifact = Artifact::open(&destination.path).expect("the published artifact opens");
    let entry = artifact
        .manifest()
        .tensors
        .iter()
        .find(|t| t.role == ROLE)
        .expect("the role is in the manifest")
        .clone();
    // The manifest, not `__metadata__`, defines what the tensors mean (owner
    // ruling of 2026-09-14), so the descriptor the kernel runs against is read
    // from it rather than restated here.
    let descriptor = entry.affine_descriptor().expect("an affine descriptor");
    assert_eq!(
        (descriptor.out_features, descriptor.in_features),
        (OUT_FEATURES, IN_FEATURES)
    );
    let section = if entry
        .components()
        .expect("a version 2 tensor")
        .iter()
        .any(|c| c.kind == ComponentKind::ZeroPoints)
    {
        ZeroPointSection::PerGroup
    } else {
        ZeroPointSection::Absent
    };
    assert_eq!(section, ZeroPointSection::PerGroup, "an asymmetric module");
    let launch =
        AffineLaunch::derive(&descriptor, section, ROWS as u64).expect("the launch derives");

    // --- the host oracle, over the published bytes --------------------------
    // Streamed through the production reader, which verifies each component
    // against its own checksum, then decoded by task 0024's decoder. It shares
    // no code with the kernel: it reconstructs the whole weight and multiplies;
    // the kernel never materializes one.
    let mut payload = Vec::new();
    let mut scratch = vec![0u8; 64 * 1024];
    artifact
        .stream_tensor(ROLE, &mut scratch, &mut |slice| {
            payload.extend_from_slice(slice);
            Ok(())
        })
        .expect("the payload verifies against its checksums");
    let decoded = moxie_format::payload::decode(descriptor.clone(), section, &payload)
        .expect("the published payload decodes");
    let weight: Vec<f32> = decoded
        .reconstruct()
        .expect("every value reconstructs")
        .iter()
        .map(|w| bf16_round(*w))
        .collect();

    let mut state = 0x0028_0025u64;
    let x: Vec<f32> = (0..ROWS * IN_FEATURES)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            bf16_round(((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5)
        })
        .collect();
    let x_bytes: Vec<u8> = x
        .iter()
        .flat_map(|v| to_bf16_bits(*v).to_le_bytes())
        .collect();

    let mut want = vec![0f32; ROWS * OUT_FEATURES];
    let mut terms = vec![0f32; ROWS * OUT_FEATURES];
    for m in 0..ROWS {
        for n in 0..OUT_FEATURES {
            let (mut acc, mut abs) = (0f32, 0f32);
            for k in 0..IN_FEATURES {
                let term = x[m * IN_FEATURES + k] * weight[n * IN_FEATURES + k];
                acc += term;
                abs += term.abs();
            }
            want[m * OUT_FEATURES + n] = acc;
            terms[m * OUT_FEATURES + n] = abs;
        }
    }

    // --- the same bytes, through residency and the kernel -------------------
    // The authority's artifact identity is data supplied by whoever opened the
    // artifact; the source revision is the honest thing to key it on here.
    let artifact_id =
        ArtifactId::new(format!("sha256:{artifact_identity}")).expect("an artifact identity");
    let mut shards = Vec::new();
    let mut lengths = Vec::new();
    for component in entry.components().expect("a version 2 tensor") {
        let bytes = artifact
            .shard_entry(&component.file, &component.name)
            .expect("the shard declares the component");
        shards.push(Shard::open(&destination.path.join(&component.file)).expect("a shard"));
        lengths.push((component.kind, component.name.clone(), bytes));
    }
    let mut source = ShardSource::new(artifact_id.clone(), shards);
    for (index, (kind, name, _)) in lengths.iter().enumerate() {
        source = source
            .role(role_of(*kind), index, name.clone())
            .expect("a role");
    }
    let length_of = |kind: ComponentKind| {
        lengths
            .iter()
            .find(|(k, _, _)| *k == kind)
            .map(|(_, _, bytes)| *bytes)
            .expect("the component is published")
    };
    assert_eq!(
        length_of(ComponentKind::Codes),
        launch.code_bytes().unwrap()
    );
    assert_eq!(
        length_of(ComponentKind::Scales),
        launch.scale_bytes().unwrap()
    );
    assert_eq!(
        Some(length_of(ComponentKind::ZeroPoints)),
        launch.zero_point_bytes().unwrap()
    );

    // Every visible device, because a pass on one architecture is not evidence
    // for the other (document 07).
    let count = device_count().expect("devices");
    for ordinal in 0..count {
        let capability = query_device(ordinal).expect("a device capability");
        let ctx = RankContext::acquire(RankId(28_500 + ordinal), ordinal).expect("a rank context");
        let stream = Stream::new(&ctx).expect("a stream");
        let scope = Scope::Device(ctx.uuid());
        let align = |bytes: u64| bytes.div_ceil(256) * 256;
        let weight_cap = align(length_of(ComponentKind::Codes))
            + align(length_of(ComponentKind::Scales))
            + align(length_of(ComponentKind::ZeroPoints));
        let mut ledger = Ledger::new([
            CapacitySnapshot::new(Scope::Host, 256 << 20, 1 << 20).unwrap(),
            CapacitySnapshot::new(scope, 256 << 20, 1 << 20).unwrap(),
        ])
        .unwrap();
        let mut authority = ResidencyAuthority::open(
            &mut ledger,
            &ResidencyRequest::new("published int4 module", weight_cap)
                .device(ctx.uuid(), weight_cap),
        )
        .unwrap();
        let mut residency = DeviceResidency::create(&ctx, &mut authority).unwrap();

        let codes = resident(
            &mut authority,
            &mut source,
            &mut residency,
            &stream,
            &artifact_id,
            scope,
            "codes",
            length_of(ComponentKind::Codes),
        );
        let scales = resident(
            &mut authority,
            &mut source,
            &mut residency,
            &stream,
            &artifact_id,
            scope,
            "scales",
            length_of(ComponentKind::Scales),
        );
        let zero_points = resident(
            &mut authority,
            &mut source,
            &mut residency,
            &stream,
            &artifact_id,
            scope,
            "zero_points",
            length_of(ComponentKind::ZeroPoints),
        );

        let kernel = select_affine_linear_kernel(
            &moxie_kernels::affine_linear_catalogue(),
            &capability,
            WeightPrecision::expect(descriptor.width.precision()),
            &launch,
        )
        .expect("a descriptor for this device");
        let mut run = AffineLinearRun::admit(&mut ledger, &ctx, kernel, launch)
            .unwrap_or_else(|refused| panic!("admission: {}", refused.error));
        let footprint = residency.capacity() + run.arena_bytes();
        let dequantized = launch.dequantized_weight_bytes().unwrap();
        assert!(
            footprint < dequantized,
            "the launch holds {footprint} device byte(s); a BF16 copy of this weight alone \
             would need {dequantized}"
        );

        let got = run
            .run(
                &stream,
                &authority,
                &residency,
                ResidentAffineWeight {
                    codes: &codes,
                    scales: &scales,
                    zero_points: Some(&zero_points),
                },
                &x_bytes,
            )
            .expect("the module executes");

        let (mut worst_ulp, mut cancelled) = (0f64, 0usize);
        for (i, expected) in want.iter().enumerate() {
            let device =
                f32::from_bits(u32::from(u16::from_le_bytes([got[i * 2], got[i * 2 + 1]])) << 16);
            let reference = bf16_round(*expected);
            let difference = (device - reference).abs();
            let exponent = (reference.abs().to_bits() >> 23) & 0xFF;
            let ulp = if exponent <= 7 {
                f32::from_bits(1)
            } else {
                f32::from_bits((exponent - 7) << 23)
            };
            let in_ulp = f64::from(difference) / f64::from(ulp);
            worst_ulp = worst_ulp.max(in_ulp);
            if in_ulp <= 2.0 {
                continue;
            }
            // The owner's cancellation clause of 2026-09-14.
            assert!(
                difference <= terms[i] / 256.0,
                "element {i}: |{device} - {reference}| = {difference} exceeds both 2 ULP \
                 ({}) and the reduction's resolution ({}); sum|terms| = {}",
                2.0 * ulp,
                terms[i] / 256.0,
                terms[i]
            );
            cancelled += 1;
        }

        eprintln!(
            "task0028 real module {MODULE} of {REVISION} on {} {}: {} output element(s) from \
             a {OUT_FEATURES}x{IN_FEATURES} group-32 asymmetric INT4 weight, worst \
             {worst_ulp:.3} ULP, {cancelled} covered by the cancellation clause; {footprint} \
             device byte(s) against {dequantized} for a BF16 copy. Synthetic activations: \
             this is execution, not model support.",
            capability.uuid,
            capability.sm(),
            want.len()
        );

        run.close(&mut ledger).expect("the run closes");
        for lease in [codes, scales, zero_points] {
            authority.release(lease).expect("the lease retires");
        }
        authority.end_turn(TurnId::new(1));
        authority.retire_all(scope);
        residency
            .close(&mut authority)
            .expect("the backing returns");
        authority.close(&mut ledger).expect("the authority closes");
        assert!(
            ledger.outstanding().is_empty(),
            "{:?}",
            ledger.outstanding()
        );
    }
    eprintln!(
        "task0028 real module: published in {:.1}s and executed on {count} device(s).",
        published_in.as_secs_f64()
    );
}

fn role_of(kind: ComponentKind) -> &'static str {
    match kind {
        ComponentKind::Codes => "codes",
        ComponentKind::Scales => "scales",
        ComponentKind::ZeroPoints => "zero_points",
        other => panic!("a quantized module has no {other:?} component"),
    }
}
