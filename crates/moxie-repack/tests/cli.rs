//! Task 0025 acceptance 3: the **real binary**, through its command line.
//!
//! Every test here runs `moxie-repack` as a process and reads what it printed
//! and what it exited with. A harness that called the library would not check
//! the part a user meets: the exit status, the refusals, and whether the four
//! outcomes -- failed, cancelled, published, published-durability-unconfirmed
//! -- are distinguishable without reading prose.

mod common;

use std::collections::BTreeMap;

use common::{Entry, Module, Scratch, SelectionBuilder, budgets, run, write_shard};

/// One small asymmetric INT4 module plus a BF16 tensor, in one shard.
struct Fixture {
    scratch: Scratch,
    module: Module,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let scratch = Scratch::new(label);
        let src = scratch.join("src");
        std::fs::create_dir_all(&src).expect("a source directory");
        let (rows, columns) = (16usize, 64usize);
        let groups = columns / 32;
        let module = Module {
            rows,
            columns,
            group: 32,
            bits: 4,
            codes: (0..rows)
                .map(|o| (0..columns).map(|k| ((o + k) % 16) as u32).collect())
                .collect(),
            scales: (0..rows)
                .map(|o| (0..groups).map(|g| 0.5 + 0.25 * ((o + g) % 3) as f32).collect())
                .collect(),
            zeros: (0..rows)
                .map(|o| (0..groups).map(|g| ((o * 3 + g) % 16) as u32).collect())
                .collect(),
        };
        let name = "model.layers.0.mlp.down_proj";
        let norm: Vec<u8> = (0..8u16).flat_map(|v| (0x3F80 + v).to_le_bytes()).collect();
        write_shard(
            &src.join("shard-a.safetensors"),
            &[
                Entry::new(
                    &format!("{name}.weight_packed"),
                    "I32",
                    module.packed_shape(),
                    module.packed(),
                ),
                Entry::new(
                    &format!("{name}.weight_scale"),
                    "BF16",
                    module.scale_shape(),
                    module.scale_payload("BF16"),
                ),
                Entry::new(
                    &format!("{name}.weight_shape"),
                    "I64",
                    vec![2],
                    module.weight_shape(),
                ),
                Entry::new(
                    &format!("{name}.weight_zero_point"),
                    "I32",
                    module.zero_point_shape(),
                    module.zero_point(),
                ),
                Entry::new("model.norm.weight", "BF16", vec![8], norm),
            ],
        );
        let files = BTreeMap::from([
            ("weight_packed", "shard-a.safetensors"),
            ("weight_scale", "shard-a.safetensors"),
            ("weight_shape", "shard-a.safetensors"),
            ("weight_zero_point", "shard-a.safetensors"),
        ]);
        SelectionBuilder::new("synthetic")
            .pack_quantized(
                "model.layers.0.mlp.down_proj.weight",
                name,
                "int4",
                "32",
                "packed-along-output",
                &files,
            )
            .bf16("model.norm.weight", "model.norm.weight", "shard-a.safetensors")
            .write(&scratch.join("selection.toml"));
        Self { scratch, module }
    }

    fn path(&self, name: &str) -> String {
        self.scratch
            .join(name)
            .to_str()
            .expect("a UTF-8 path")
            .to_string()
    }

    fn repack_args(&self, out: &str, extra: &[&str]) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "repack".into(),
            "--selection".into(),
            self.path("selection.toml"),
            "--source-root".into(),
            self.path("src"),
            "--out".into(),
            out.to_string(),
        ];
        args.extend(budgets().into_iter().map(str::to_string));
        args.extend(extra.iter().map(|s| (*s).to_string()));
        args
    }

    fn repack(&self, out: &str, extra: &[&str]) -> common::Run {
        let args = self.repack_args(out, extra);
        let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
        run(&borrowed)
    }
}

#[test]
fn inspect_reports_the_selection_and_writes_nothing() {
    let f = Fixture::new("cli-inspect");
    let before = f.scratch.bytes_used();
    let r = run(&[
        "inspect",
        "--selection",
        &f.path("selection.toml"),
        "--source-root",
        &f.path("src"),
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
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.outcome(), "inspected");
    // What it must report.
    for field in [
        "model",
        "revision",
        "selection-digest",
        "plan-digest",
        "canonical-payload-bytes",
        "source-payload-bytes",
        "staging-disk-peak-bytes",
        "admitted-ram-bytes",
        "headers-parsed",
        "completeness",
        "estimated-time",
    ] {
        assert!(r.field(field).is_some(), "inspect reports no {field}:\n{}", r.stdout);
    }
    // The canonical size is the descriptor's arithmetic, not a guess.
    let expected = f.module.expected_canonical("BF16").len() + 16;
    assert_eq!(
        r.field("canonical-payload-bytes"),
        Some(expected.to_string().as_str()),
        "{}",
        r.stdout
    );
    // Time is unmeasured rather than invented.
    assert!(r.says("unmeasured"), "{}", r.stdout);
    // A partial selection says so, in as many words.
    assert_eq!(r.field("completeness"), Some("partial"));
    assert!(r.says("not a complete model"), "{}", r.stdout);
    // And nothing on disk moved.
    assert_eq!(f.scratch.bytes_used(), before, "inspect wrote something");
    // It also refuses an argument that would suggest otherwise.
    let r = run(&[
        "inspect",
        "--selection",
        &f.path("selection.toml"),
        "--source-root",
        &f.path("src"),
        "--out",
        &f.path("artifact"),
    ]);
    assert_eq!(r.status, 1);
    assert!(r.says("never writes anything"), "{}{}", r.stdout, r.stderr);
}

/// The budgets are required. A repack states what it may spend.
#[test]
fn a_run_without_its_budgets_is_refused_before_it_starts() {
    let f = Fixture::new("cli-budgets");
    for missing in [
        "--total-bytes",
        "--header-bytes",
        "--scratch-bytes",
        "--chunk-file-bytes",
        "--disk-bytes",
    ] {
        let args = f.repack_args(&f.path("artifact"), &[]);
        let mut kept: Vec<&str> = Vec::new();
        let mut skip = false;
        for a in args.iter().map(String::as_str) {
            if skip {
                skip = false;
                continue;
            }
            if a == missing {
                skip = true;
                continue;
            }
            kept.push(a);
        }
        let r = run(&kept);
        assert_eq!(r.status, 1, "{missing}: {}{}", r.stdout, r.stderr);
        assert!(r.says(missing), "{missing}: {}", r.stderr);
        assert!(!f.scratch.join("artifact").exists(), "{missing} wrote a destination");
    }
}

/// A budget too small to do the work is a refusal, and the message says which
/// budget and by how much.
#[test]
fn an_insufficient_budget_is_refused_before_any_payload_is_created() {
    let f = Fixture::new("cli-small-budget");
    let out = f.path("artifact");
    let mut args = vec![
        "repack".to_string(),
        "--selection".into(),
        f.path("selection.toml"),
        "--source-root".into(),
        f.path("src"),
        "--out".into(),
        out.clone(),
        "--total-bytes".into(),
        "128MiB".into(),
        "--header-bytes".into(),
        "64MiB".into(),
        "--scratch-bytes".into(),
        "1MiB".into(),
        "--chunk-file-bytes".into(),
        // Smaller than the first tensor's canonical payload.
        "64".into(),
        "--disk-bytes".into(),
        "64MiB".into(),
    ];
    let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
    let r = run(&borrowed);
    assert_eq!(r.status, 2, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.outcome(), "failed");
    assert!(
        r.says("larger admitted file limit"),
        "the refusal names the limit: {}",
        r.stdout
    );
    assert!(
        !std::path::Path::new(&out).join("chunk0.bin").exists(),
        "a refused plan created a payload"
    );
    // And the same for the disk budget: a chunk limit that fits, and a disk
    // budget that does not.
    let chunk_at = args
        .iter()
        .position(|a| a == "--chunk-file-bytes")
        .expect("the flag is there")
        + 1;
    args[chunk_at] = "4MiB".into();
    let disk_at = args
        .iter()
        .position(|a| a == "--disk-bytes")
        .expect("the flag is there")
        + 1;
    args[disk_at] = "4MiB".into();
    // One tensor of 656 canonical bytes against a 512-byte allowance.
    args[disk_at] = "512".into();
    args[chunk_at] = "512".into();
    let borrowed: Vec<&str> = args.iter().map(String::as_str).collect();
    let r = run(&borrowed);
    assert_eq!(r.status, 2, "{}{}", r.stdout, r.stderr);
    assert!(
        r.says("larger admitted file limit") || r.says("disk budget"),
        "a selection above both disk limits is refused, naming one: {}",
        r.stdout
    );
}

#[test]
fn a_bounded_repack_publishes_and_verifies_and_reports_what_it_did() {
    let f = Fixture::new("cli-repack");
    let out = f.path("artifact");
    let r = f.repack(&out, &[]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.outcome(), "published");
    assert_eq!(r.field("resumed"), Some("false"));
    assert!(r.field("artifact-identity").is_some());
    assert!(r.field("source-digest").is_some(), "{}", r.stdout);
    let v = run(&["verify", "--artifact", &out, "--scratch-bytes", "4096"]);
    assert_eq!(v.status, 0, "{}{}", v.stdout, v.stderr);
    assert_eq!(v.outcome(), "verified");
    assert_eq!(v.field("artifact-identity"), r.field("artifact-identity"));
    assert!(
        v.says("not a statement about what any model produces"),
        "{}",
        v.stdout
    );
}

#[test]
fn a_published_destination_is_refused_rather_than_overwritten() {
    let f = Fixture::new("cli-existing");
    let out = f.path("artifact");
    assert_eq!(f.repack(&out, &[]).outcome(), "published");
    let before = std::fs::read(std::path::Path::new(&out).join("chunk0.bin")).expect("a chunk");
    let again = f.repack(&out, &[]);
    assert_eq!(again.status, 2, "{}{}", again.stdout, again.stderr);
    assert!(again.says("already holds a published manifest"), "{}", again.stdout);
    assert!(again.says("never updates a model in place"), "{}", again.stdout);
    assert_eq!(
        std::fs::read(std::path::Path::new(&out).join("chunk0.bin")).expect("a chunk"),
        before,
        "the published artifact was touched"
    );
}

#[test]
fn writing_into_the_source_is_refused() {
    let f = Fixture::new("cli-collision");
    // The destination is the source directory itself, which holds shards this
    // run must not write beside.
    let r = f.repack(&f.path("src"), &[]);
    assert_eq!(r.status, 2, "{}{}", r.stdout, r.stderr);
    assert!(
        r.says("refusing to write into a directory this run did not create"),
        "{}",
        r.stdout
    );
    assert!(
        f.scratch.join("src").join("shard-a.safetensors").exists(),
        "the source shard survived"
    );
}

#[test]
fn a_missing_or_unsupported_source_is_refused_by_name() {
    let f = Fixture::new("cli-missing");
    // A source root that is not there.
    let r = run(&[
        "inspect",
        "--selection",
        &f.path("selection.toml"),
        "--source-root",
        &f.path("nowhere"),
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
    assert_eq!(r.status, 2, "{}{}", r.stdout, r.stderr);
    assert!(r.says("does not canonicalize"), "{}", r.stdout);

    // A selection naming a packing this importer does not support.
    let text = std::fs::read_to_string(f.scratch.join("selection.toml")).expect("a selection");
    let unsupported = text.replace("kind = \"pack-quantized\"", "kind = \"auto-gptq\"");
    std::fs::write(f.scratch.join("unsupported.toml"), unsupported).expect("writing it");
    let r = run(&[
        "inspect",
        "--selection",
        &f.path("unsupported.toml"),
        "--source-root",
        &f.path("src"),
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
    assert_eq!(r.status, 2);
    assert!(r.says("refused by name rather than guessed at"), "{}", r.stdout);

    // A selection naming a tensor that is not in the shard.
    let absent = text.replace("model.norm.weight", "model.absent.weight");
    std::fs::write(f.scratch.join("absent.toml"), absent).expect("writing it");
    let r = run(&[
        "inspect",
        "--selection",
        &f.path("absent.toml"),
        "--source-root",
        &f.path("src"),
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
    assert_eq!(r.status, 2);
    assert!(r.says("model.absent.weight"), "{}", r.stdout);
}

/// The four outcomes are distinguishable by exit status alone.
#[test]
fn failed_cancelled_published_and_durability_unconfirmed_are_distinct() {
    let f = Fixture::new("cli-outcomes");

    // Cancelled: exit 3, a resumable destination, no manifest.
    let cancelled = f.path("cancelled");
    let r = f.repack(&cancelled, &["--test-cancel-after-units", "1"]);
    assert_eq!(r.status, 3, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.outcome(), "cancelled");
    assert!(!std::path::Path::new(&cancelled).join("manifest.toml").exists());
    // And it resumes to a published artifact.
    let r = f.repack(&cancelled, &["--take-over-interrupted-run"]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.field("resumed"), Some("true"));

    // Failed: exit 2, an injected failure at a durable boundary.
    let failed = f.path("failed");
    let r = f.repack(&failed, &["--test-fail-at", "chunk-sync:1"]);
    assert_eq!(r.status, 2, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.outcome(), "failed");
    assert!(r.says("resumable private state"), "{}", r.stdout);
    let r = f.repack(&failed, &["--take-over-interrupted-run"]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);

    // Published, durability unconfirmed: exit 4, and the artifact exists.
    let unconfirmed = f.path("unconfirmed");
    let r = f.repack(&unconfirmed, &["--test-fail-at", "publish-durability:1"]);
    assert_eq!(r.status, 4, "{}{}", r.stdout, r.stderr);
    assert_eq!(r.outcome(), "published-durability-unconfirmed");
    assert!(std::path::Path::new(&unconfirmed).join("manifest.toml").exists());
    let v = run(&[
        "verify",
        "--artifact",
        &unconfirmed,
        "--scratch-bytes",
        "4096",
    ]);
    assert_eq!(v.status, 0, "an unconfirmed publish is still readable");

    // And an unknown failure site is a command-line error, not a silent
    // no-op: a gate that asked for a boundary that does not exist measured
    // nothing.
    let r = f.repack(&f.path("typo"), &["--test-fail-at", "chunk-fsync:1"]);
    assert_eq!(r.status, 1, "{}{}", r.stdout, r.stderr);
    assert!(r.says("unknown failure site"), "{}", r.stderr);
}

/// An actual process interruption, not an in-process fault: the program aborts
/// mid-run, and a restart finishes the job and publishes the same artifact.
#[test]
fn an_aborted_process_restarts_and_publishes_an_identical_artifact() {
    let f = Fixture::new("cli-abort");
    let reference = f.path("reference");
    let r = f.repack(&reference, &[]);
    assert_eq!(r.outcome(), "published");
    let reference_identity = r.field("artifact-identity").expect("an identity").to_string();
    let reference_chunk =
        std::fs::read(std::path::Path::new(&reference).join("chunk0.bin")).expect("a chunk");

    for after in 1..=3 {
        let out = f.path(&format!("aborted-{after}"));
        let r = f.repack(&out, &["--test-abort-after-units", &after.to_string()]);
        // SIGABRT: the shell reports 134, and the important part is that it is
        // not a clean exit.
        assert_ne!(r.status, 0, "the process did not abort: {}", r.stdout);
        assert!(
            !std::path::Path::new(&out).join("manifest.toml").exists(),
            "an aborted run left a readable artifact"
        );
        assert!(
            std::path::Path::new(&out).join(".moxie-repack-journal").exists(),
            "an aborted run left no journal to resume from"
        );
        let r = f.repack(&out, &["--take-over-interrupted-run"]);
        assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
        assert_eq!(r.field("resumed"), Some("true"));
        assert_eq!(
            r.field("artifact-identity"),
            Some(reference_identity.as_str()),
            "a restarted run published a different artifact"
        );
        assert_eq!(
            std::fs::read(std::path::Path::new(&out).join("chunk0.bin")).expect("a chunk"),
            reference_chunk
        );
    }
}

/// A destination another run owns is refused, and taking it over is explicit.
#[test]
fn an_interrupted_runs_lock_is_not_taken_over_silently() {
    let f = Fixture::new("cli-lock");
    let out = f.path("locked");
    let r = f.repack(&out, &["--test-abort-after-units", "1"]);
    assert_ne!(r.status, 0);
    assert!(std::path::Path::new(&out).join(".moxie-repack-lock").exists());
    let r = f.repack(&out, &[]);
    assert_eq!(r.status, 2, "{}{}", r.stdout, r.stderr);
    assert!(r.says("owns this destination"), "{}", r.stdout);
    let r = f.repack(&out, &["--take-over-interrupted-run"]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
}

#[test]
fn verify_refuses_a_corrupted_artifact_and_says_which_tensor() {
    let f = Fixture::new("cli-corrupt");
    let out = f.path("artifact");
    assert_eq!(f.repack(&out, &[]).outcome(), "published");
    let chunk = std::path::Path::new(&out).join("chunk0.bin");
    let mut bytes = std::fs::read(&chunk).expect("a chunk");
    bytes[7] ^= 0xFF;
    std::fs::write(&chunk, bytes).expect("corrupting it");
    let v = run(&["verify", "--artifact", &out, "--scratch-bytes", "4096"]);
    assert_eq!(v.status, 2, "{}{}", v.stdout, v.stderr);
    assert_eq!(v.outcome(), "failed");
    assert!(v.says("checksum mismatch"), "{}", v.stdout);
    assert!(
        v.says("model.layers.0.mlp.down_proj.weight"),
        "the refusal names the tensor: {}",
        v.stdout
    );
}

#[test]
fn the_usage_text_is_available_and_an_unknown_command_is_refused() {
    let r = run(&["--help"]);
    assert_eq!(r.status, 0);
    assert!(r.stdout.contains("moxie-repack"), "{}", r.stdout);
    let r = run(&["convert"]);
    assert_eq!(r.status, 1);
    assert!(r.says("unknown command 'convert'"), "{}", r.stderr);
    let r = run(&[]);
    assert_eq!(r.status, 1);
}
