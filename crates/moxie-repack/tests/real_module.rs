//! Task 0025 acceptance 8: the real module, published and read back.
//!
//! **Read-only, and nothing here is model support.** One module of one
//! inventoried local artifact is repacked into a private temporary directory,
//! reopened through the production reader, and every one of its 3,145,728
//! values is compared with two quantities that are not the same one:
//!
//! * the **canonical** equation of document 03, `W = (q - z) * s`, evaluated in
//!   FP32 over the *source's* bytes; and
//! * the **source's own** arithmetic, with the BF16 rounding boundary its
//!   reference applies.
//!
//! Task 0024's review is why both are here: its first version computed the FP32
//! quantity and called it "the source's own declared arithmetic", and the two
//! differ on a quarter of the values. Neither is wrong; one of them was called
//! by the other's name.
//!
//! What this proves is that the bytes this program published reconstruct to the
//! values the source's bytes do. It is [ADR 0018]'s v1 quality definition --
//! a bit-identical repack -- and it is **not** evidence about what any model
//! produces (O2), not a whole-artifact claim, and not model support: nothing in
//! this repository executes a canonical INT4 tensor.
//!
//! The artifact skips with a printed message when absent, so a fresh clone is
//! green and a missing checkpoint is never read as a regression.
//!
//! [ADR 0018]: ../../../docs/decisions/adr/0018-v1-quality-is-bit-identical-repack.md

mod common;

use std::path::{Path, PathBuf};

use moxie_format::affine::{AffineDescriptor, Grouping, IntWidth, ZeroPoints};
use moxie_format::payload::{self, ZeroPointSection};
use moxie_format::scale::ScaleDtype;
use moxie_storage::{Artifact, ByteBudget, HeaderBudget, Shard};

const ROOT: &str = "/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4";
const REVISION: &str = "bc59f497520b23759ce61cc5164ca28bcc4f53bc";
const SHARD: &str = "model-00001-of-00015.safetensors";
const MODULE: &str = "model.layers.1.mlp.experts.0.down_proj";
const ROLE: &str = "model.layers.1.mlp.experts.0.down_proj.weight";
/// The contract's header-derived expectation, checked against what runs.
const EXPECTED_CANONICAL_BYTES: usize = 1_966_080;
const EXPECTED_VALUES: usize = 3_072 * 1_024;

/// The revision this artifact was downloaded at, as the hub recorded it.
///
/// Not checksum evidence and not treated as any: it is the artifact's own
/// claim about which revision it is, and the test refuses to proceed if it
/// disagrees with the one the task contract names. The checksum evidence is
/// the digest the repack computes over the whole file.
fn recorded_revision(root: &Path) -> Option<String> {
    let path = root
        .join(".cache/huggingface/download")
        .join(format!("{SHARD}.metadata"));
    let text = std::fs::read_to_string(path).ok()?;
    text.lines().next().map(|l| l.trim().to_string())
}

/// The file digest the hub recorded, for comparing with the one this run
/// computes. An independent statement of the same fact.
fn recorded_digest(root: &Path) -> Option<String> {
    let path = root
        .join(".cache/huggingface/download")
        .join(format!("{SHARD}.metadata"));
    let text = std::fs::read_to_string(path).ok()?;
    text.lines().nth(1).map(|l| l.trim().to_string())
}

/// A private destination under `/tmp`, as the contract requires, removed when
/// the test ends however it ends.
struct Destination {
    path: PathBuf,
}

impl Destination {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "moxie-task0025-real-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("a clock after 1970")
                .as_nanos()
        ));
        Self { path }
    }
}

impl Drop for Destination {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The source's payloads, read once through the bounded reader for the oracle.
struct Source {
    packed: Vec<u8>,
    scale: Vec<u8>,
    zero_point: Vec<u8>,
    rows: usize,
    columns: usize,
    groups: usize,
}

impl Source {
    fn read(root: &Path) -> Self {
        let shard = Shard::open_with_limits(
            &root.join(SHARD),
            ByteBudget::new(1 << 20).expect("a read budget"),
            HeaderBudget::new(64 << 20).expect("a header budget"),
        )
        .expect("the shard opens");
        let shape = shard
            .tensor_bytes(&format!("{MODULE}.weight_shape"))
            .expect("weight_shape");
        let (rows, columns) =
            moxie_format::compressed_tensors::decode_weight_shape(&shape).expect("a logical shape");
        Self {
            packed: shard
                .tensor_bytes(&format!("{MODULE}.weight_packed"))
                .expect("weight_packed"),
            scale: shard
                .tensor_bytes(&format!("{MODULE}.weight_scale"))
                .expect("weight_scale"),
            zero_point: shard
                .tensor_bytes(&format!("{MODULE}.weight_zero_point"))
                .expect("weight_zero_point"),
            rows,
            columns,
            groups: columns.div_ceil(32),
        }
    }

    /// The source's own code at `(o, k)`, read straight from `weight_packed`:
    /// eight per I32 word along the **input** axis.
    fn code(&self, o: usize, k: usize) -> i32 {
        let packed_columns = self.columns.div_ceil(8);
        let at = (o * packed_columns + k / 8) * 4;
        let word = u32::from_le_bytes(self.packed[at..at + 4].try_into().expect("four bytes"));
        (((word >> ((k % 8) as u32 * 4)) & 0xF) as i32) - 8
    }

    /// The source's own zero point, from `weight_zero_point`: eight per I32
    /// word along the **output** axis, which is the convention task 0024
    /// pinned and measured.
    fn zero(&self, o: usize, g: usize) -> i32 {
        let at = ((o / 8) * self.groups + g) * 4;
        let word = u32::from_le_bytes(self.zero_point[at..at + 4].try_into().expect("four bytes"));
        (((word >> ((o % 8) as u32 * 4)) & 0xF) as i32) - 8
    }

    fn scale(&self, o: usize, g: usize) -> f32 {
        let at = (o * self.groups + g) * 2;
        moxie_format::bf16::bf16_bits_to_f32(u16::from_le_bytes(
            self.scale[at..at + 2].try_into().expect("two bytes"),
        ))
    }
}

#[test]
fn the_real_module_repacks_reopens_and_reconstructs_every_value() {
    let root = Path::new(ROOT);
    if !root.join(SHARD).exists() {
        eprintln!(
            "SKIP the real-module proof: {} is not present on this machine. A missing source is \
             a blocked real lane, never acceptance.",
            root.join(SHARD).display()
        );
        return;
    }
    let recorded = recorded_revision(root);
    assert_eq!(
        recorded.as_deref(),
        Some(REVISION),
        "this artifact records a different revision than the contract names"
    );

    // 1. Repack the selection through the real program, with the contract's
    //    budgets and nothing raised to make it pass.
    let destination = Destination::new();
    let selection_path = destination.path.with_extension("selection.toml");
    std::fs::write(&selection_path, selection_text()).expect("writing the selection");
    let started = std::time::Instant::now();
    let result = common::run(&[
        "repack",
        "--selection",
        selection_path.to_str().expect("a path"),
        "--source-root",
        ROOT,
        "--out",
        destination.path.to_str().expect("a path"),
        "--total-bytes",
        "128MiB",
        "--header-bytes",
        "64MiB",
        "--scratch-bytes",
        "1MiB",
        "--chunk-file-bytes",
        "4MiB",
        "--disk-bytes",
        "64MiB",
    ]);
    let elapsed = started.elapsed();
    let _ = std::fs::remove_file(&selection_path);
    assert_eq!(
        result.outcome(),
        "published",
        "{}{}",
        result.stdout,
        result.stderr
    );
    // The digest this run computed over the whole 5.37 GB shard, against the
    // one the hub recorded when it was downloaded. Two independent statements
    // of the same fact; a revision string alone would be neither.
    let computed = result
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("source-digest: "))
        .and_then(|l| l.split_whitespace().nth(1))
        .expect("a source digest");
    if let Some(recorded) = recorded_digest(root) {
        assert_eq!(
            computed, recorded,
            "the digest this run computed differs from the one recorded at download"
        );
    }

    // 2. Reopen through the production reader and stream the payload back.
    let artifact = Artifact::open(&destination.path).expect("the published artifact opens");
    let tensor = artifact
        .manifest()
        .tensors
        .iter()
        .find(|t| t.role == ROLE)
        .expect("the role is in the manifest");
    // The canonical payload is now three physical tensors, and their lengths
    // add up to what one byte range used to hold: the bytes did not change,
    // their container did (ADR 0025).
    let components = tensor.components().expect("a version 2 tensor");
    assert_eq!(components.len(), 3, "codes, scales and zero points");
    let published: u64 = components
        .iter()
        .map(|c| {
            artifact
                .shard_entry(&c.file, &c.name)
                .expect("the shard declares it")
        })
        .sum();
    assert_eq!(published as usize, EXPECTED_CANONICAL_BYTES);
    let mut payload = Vec::with_capacity(EXPECTED_CANONICAL_BYTES);
    let mut scratch = vec![0u8; 64 * 1024];
    let read = artifact
        .stream_tensor(ROLE, &mut scratch, &mut |slice| {
            payload.extend_from_slice(slice);
            Ok(())
        })
        .expect("the payload verifies against its checksum");
    assert_eq!(read as usize, EXPECTED_CANONICAL_BYTES);
    assert_eq!(payload.len(), EXPECTED_CANONICAL_BYTES);

    // 3. Decode the canonical payload and reconstruct every value.
    let source = Source::read(root);
    assert_eq!((source.rows, source.columns), (3072, 1024));
    let descriptor = AffineDescriptor {
        width: IntWidth::Int4,
        out_features: source.rows,
        in_features: source.columns,
        grouping: Grouping::Contiguous { size: 32 },
        group_index: None,
        scale_dtype: ScaleDtype::Bf16,
    };
    let decoded = payload::decode(descriptor, ZeroPointSection::PerGroup, &payload)
        .expect("the published payload decodes");
    match decoded.zero_points() {
        ZeroPoints::PerGroup(z) => assert_eq!(z.len(), source.rows * source.groups),
        other => panic!("a symmetric tensor came back: {other:?}"),
    }

    let mut row = vec![0f32; source.columns];
    let mut checked = 0usize;
    let mut narrowed = 0usize;
    let (mut low, mut high) = (i32::MAX, i32::MIN);
    let (mut zlow, mut zhigh) = (i32::MAX, i32::MIN);
    for o in 0..source.rows {
        decoded
            .reconstruct_row_into(o, &mut row)
            .expect("every row reconstructs");
        for (k, value) in row.iter().enumerate() {
            let q = source.code(o, k);
            let z = source.zero(o, k / 32);
            let s = source.scale(o, k / 32);
            // (a) The canonical equation, in FP32, over the source's bytes.
            let canonical = (q - z) as f32 * s;
            assert_eq!(
                value.to_bits(),
                canonical.to_bits(),
                "({o},{k}): the published value {value} is not the canonical {canonical}"
            );
            // (b) The source's own arithmetic, at the BF16 boundary its
            // reference applies. The canonical FP32 value rounds to it
            // exactly; a repack that changed one code, zero point or scale
            // byte could not.
            let source_bf16 = moxie_format::bf16::f32_to_bf16_bits(canonical);
            assert_eq!(
                moxie_format::bf16::f32_to_bf16_bits(*value),
                source_bf16,
                "({o},{k}): the published value does not round to the source's own result"
            );
            if canonical.to_bits() != moxie_format::bf16::bf16_bits_to_f32(source_bf16).to_bits() {
                narrowed += 1;
            }
            low = low.min(q);
            high = high.max(q);
            zlow = zlow.min(z);
            zhigh = zhigh.max(z);
            checked += 1;
        }
    }
    assert_eq!(checked, EXPECTED_VALUES, "every value of the module");
    assert_eq!((low, high), (-8, 7), "the codes span the whole INT4 range");
    assert!(
        zlow < 0 && zhigh > 0,
        "the zero points span [{zlow}, {zhigh}], so the asymmetric lane is exercised"
    );
    assert!(
        narrowed > 0,
        "the source's BF16 boundary never fired across {checked} value(s), so checking it \
         proved nothing"
    );

    // 4. Verify every published payload through the program, as a user would.
    let verify = common::run(&[
        "verify",
        "--artifact",
        destination.path.to_str().expect("a path"),
        "--scratch-bytes",
        "1MiB",
    ]);
    assert_eq!(verify.outcome(), "verified", "{}", verify.stdout);
    assert_eq!(
        verify.field("bytes-verified"),
        Some(EXPECTED_CANONICAL_BYTES.to_string().as_str())
    );
    assert_eq!(verify.field("completeness"), Some("partial"));

    let staged: u64 = std::fs::read_dir(&destination.path)
        .expect("the artifact")
        .filter_map(|e| e.ok())
        .map(|e| e.metadata().expect("a file").len())
        .sum();
    eprintln!(
        "task0025 real module {MODULE} of {REVISION}: published {EXPECTED_CANONICAL_BYTES} \
         canonical byte(s) in {staged} byte(s) of directory, checked {checked} reconstructed \
         value(s) against the canonical FP32 equation and against the source's own BF16 \
         arithmetic; the source's rounding boundary moves {narrowed} of them; codes span \
         [{low}, {high}] and zero points span [{zlow}, {zhigh}]; the whole repack took \
         {:.1}s, nearly all of it hashing the 5.37 GB source shard. This is a repack, not \
         model support: nothing here executes a canonical INT4 tensor.",
        elapsed.as_secs_f64()
    );
    assert!(
        staged < 64 << 20,
        "{staged} byte(s) is above the task's scratch-disk cap"
    );
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
