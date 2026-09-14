//! Task 0026 round 2: what a published artifact must refuse, and what a run
//! must refuse to touch.
//!
//! Every case here is a reproduction from the second independent review. They
//! share one shape: something about the artifact or the destination disagrees
//! with what the descriptor, the schema or the filesystem says, and the
//! previous code accepted it because it compared a claim against a copy of
//! itself.

mod common;

use std::collections::BTreeMap;

use common::{Entry, Module, Scratch, SelectionBuilder, bf16_bytes, budgets, run, write_shard};

/// One asymmetric INT4 module plus a BF16 tensor, and the pieces to repack it.
struct Fixture {
    scratch: Scratch,
    src: std::path::PathBuf,
    selection: std::path::PathBuf,
    out: std::path::PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let scratch = Scratch::new(label);
        let src = scratch.join("src");
        std::fs::create_dir_all(&src).expect("a source directory");
        let name = "model.layers.0.mlp.down_proj";
        let rows = 4usize;
        let columns = 64usize;
        let groups = columns / 32;
        let m = Module {
            rows,
            columns,
            group: 32,
            bits: 4,
            codes: (0..rows)
                .map(|o| {
                    (0..columns)
                        .map(|k| ((o * 7 + k * 3) % 16) as u32)
                        .collect()
                })
                .collect(),
            scales: (0..rows)
                .map(|o| (0..groups).map(|g| 0.5 + ((o + g) % 3) as f32).collect())
                .collect(),
            zeros: (0..rows)
                .map(|o| {
                    (0..groups)
                        .map(|g| ((o * 5 + g * 11) % 16) as u32)
                        .collect()
                })
                .collect(),
        };
        let entries = vec![
            Entry::new(
                &format!("{name}.weight_packed"),
                "I32",
                m.packed_shape(),
                m.packed(),
            ),
            Entry::new(
                &format!("{name}.weight_scale"),
                "BF16",
                m.scale_shape(),
                m.scale_payload("BF16"),
            ),
            Entry::new(
                &format!("{name}.weight_shape"),
                "I64",
                vec![2],
                m.weight_shape(),
            ),
            Entry::new(
                &format!("{name}.weight_zero_point"),
                "I32",
                m.zero_point_shape(),
                m.zero_point(),
            ),
            Entry::new(
                "model.norm.weight",
                "BF16",
                vec![8],
                bf16_bytes(1.0).repeat(8),
            ),
        ];
        write_shard(&src.join("shard.safetensors"), &entries);

        let selection = SelectionBuilder::new("malformed")
            .pack_quantized(
                &format!("{name}.weight"),
                name,
                "int4",
                "32",
                "packed-along-output",
                &BTreeMap::from([
                    ("weight_packed", "shard.safetensors"),
                    ("weight_scale", "shard.safetensors"),
                    ("weight_shape", "shard.safetensors"),
                    ("weight_zero_point", "shard.safetensors"),
                ]),
            )
            .bf16(
                "model.norm.weight",
                "model.norm.weight",
                "shard.safetensors",
            );
        let selection_path = scratch.join("selection.toml");
        selection.write(&selection_path);
        let out = scratch.join("artifact");
        Self {
            scratch,
            src,
            selection: selection_path,
            out,
        }
    }

    fn repack(&self, extra: &[&str]) -> common::Run {
        let mut args = vec![
            "repack".to_string(),
            "--selection".into(),
            self.selection.to_string_lossy().into_owned(),
            "--source-root".into(),
            self.src.to_string_lossy().into_owned(),
            "--out".into(),
            self.out.to_string_lossy().into_owned(),
        ];
        args.extend(budgets().iter().map(|s| s.to_string()));
        args.extend(extra.iter().map(|s| s.to_string()));
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        run(&argv)
    }
}

/// Publish, and return the fixture with its artifact.
fn publish(label: &str) -> (Fixture, std::path::PathBuf) {
    let f = Fixture::new(label);
    let r = f.repack(&[]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    let out = f.out.clone();
    (f, out)
}

fn verify(artifact: &std::path::Path) -> common::Run {
    run(&[
        "verify",
        "--artifact",
        artifact.to_str().expect("utf-8"),
        "--scratch-bytes",
        "1MiB",
    ])
}

fn shard_of(artifact: &std::path::Path) -> std::path::PathBuf {
    let mut found = None;
    for entry in std::fs::read_dir(artifact).expect("the artifact reads") {
        let path = entry.expect("an entry").path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            found = Some(path);
        }
    }
    found.expect("a shard")
}

/// Rewrite the shard's JSON header in place, keeping its length.
///
/// The header is padded with spaces to an eight-byte boundary, so a
/// same-length substitution leaves every offset and the declared header length
/// untouched: what changes is only the claim under test.
fn patch_header(shard: &std::path::Path, from: &str, to: &str) {
    let mut bytes = std::fs::read(shard).expect("the shard reads");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
    let header = String::from_utf8(bytes[8..8 + header_len].to_vec()).expect("utf-8 header");
    let mut patched = header.replacen(from, to, 1);
    assert_ne!(patched, header, "the substitution matched nothing");
    // The header is space-padded to an eight-byte boundary, so a shorter or
    // longer substitution is absorbed there: the declared header length and
    // every payload offset stay exactly as published, and the only thing the
    // reader sees change is the claim under test.
    match patched.len().cmp(&header_len) {
        std::cmp::Ordering::Less => {
            patched.push_str(&" ".repeat(header_len - patched.len()));
        }
        std::cmp::Ordering::Greater => {
            let over = patched.len() - header_len;
            let trimmed = patched.trim_end_matches(' ').len();
            assert!(
                patched.len() - trimmed >= over,
                "the header has no padding to absorb a longer substitution"
            );
            patched.truncate(patched.len() - over);
        }
        std::cmp::Ordering::Equal => {}
    }
    bytes[8..8 + header_len].copy_from_slice(patched.as_bytes());
    std::fs::write(shard, bytes).expect("the shard writes");
}

/// A component whose declared dtype is not the one the descriptor implies.
///
/// Review changed a published BF16 component's header to `F16` and `verify`
/// still succeeded: opening checked that the component **existed**, and nothing
/// else about it.
#[test]
fn a_component_with_the_wrong_dtype_is_refused() {
    let (_scratch, artifact) = publish("dtype");
    assert_eq!(verify(&artifact).status, 0, "it verifies before the change");
    patch_header(&shard_of(&artifact), "\"BF16\"", "\"F16\"");
    let r = verify(&artifact);
    assert_ne!(r.status, 0, "a re-typed component verified: {}", r.stdout);
    assert!(
        r.stdout.contains("descriptor implies") || r.stderr.contains("descriptor implies"),
        "the refusal does not say what was expected: {}{}",
        r.stdout,
        r.stderr
    );
}

/// A logical shape that no longer matches the components on disk.
///
/// Review changed a manifest's shape from `[2048]` to `[2047]` and verification
/// still succeeded, because the shape was only ever compared against itself.
#[test]
fn a_logical_shape_that_disagrees_with_the_components_is_refused() {
    let (_scratch, artifact) = publish("shape");
    let manifest_path = artifact.join("manifest.toml");
    let text = std::fs::read_to_string(&manifest_path).expect("the manifest reads");
    let patched = text.replacen("shape = [8]", "shape = [7]", 1);
    assert_ne!(patched, text, "the BF16 tensor's shape was not found");
    std::fs::write(&manifest_path, patched).expect("the manifest writes");
    let r = verify(&artifact);
    assert_ne!(r.status, 0, "a wrong logical shape verified: {}", r.stdout);
}

/// Components in the wrong order are a different byte stream.
///
/// `stream_tensor` concatenates them as listed and promises codes, then scales,
/// then zero points. Validation compared a **sorted** set of kinds, so a
/// reversed list passed and then streamed reversed.
#[test]
fn components_out_of_canonical_order_are_refused() {
    let (_scratch, artifact) = publish("order");
    let manifest_path = artifact.join("manifest.toml");
    let text = std::fs::read_to_string(&manifest_path).expect("the manifest reads");
    // Swap the kinds of the first two component rows of the affine tensor.
    let patched = text
        .replacen("kind = \"codes\"", "kind = \"TEMP\"", 1)
        .replacen("kind = \"scales\"", "kind = \"codes\"", 1)
        .replacen("kind = \"TEMP\"", "kind = \"scales\"", 1);
    assert_ne!(patched, text, "the component rows were not found");
    std::fs::write(&manifest_path, patched).expect("the manifest writes");
    let r = verify(&artifact);
    assert_ne!(
        r.status, 0,
        "a reordered component list verified: {}",
        r.stdout
    );
}

/// Bytes no tensor claims are bytes the reference implementation refuses.
///
/// Review appended garbage to a published shard; `verify` reported it verified
/// while `safetensors` 0.7.0 called the same file "file not fully covered".
#[test]
fn a_shard_with_bytes_no_tensor_claims_is_refused() {
    let (_scratch, artifact) = publish("coverage");
    let shard = shard_of(&artifact);
    let mut bytes = std::fs::read(&shard).expect("the shard reads");
    bytes.extend_from_slice(b"garbage");
    std::fs::write(&shard, bytes).expect("the shard writes");
    let r = verify(&artifact);
    assert_ne!(r.status, 0, "a shard with slack verified: {}", r.stdout);
    assert!(
        r.stdout.contains("covered") || r.stderr.contains("covered"),
        "the refusal does not name the coverage rule: {}{}",
        r.stdout,
        r.stderr
    );
}

/// A private name that is a hard link to something else is not this run's file.
///
/// `O_NOFOLLOW` cannot see a hard link: it is not a link to follow, it is a
/// second name for the same inode. Review cancelled a run, linked an unrelated
/// file over `.moxie-repack-manifest` and resumed, and publication overwrote
/// the unrelated file.
#[test]
#[cfg(unix)]
fn a_hard_linked_private_file_is_refused_and_its_target_is_untouched() {
    let f = Fixture::new("hardlink");
    // Cancel part-way, so the destination is a resumable private state.
    let r = f.repack(&["--test-cancel-after-units", "1"]);
    assert_eq!(r.status, 3, "{}{}", r.stdout, r.stderr);

    let outsider = f.scratch.join("outsider.txt");
    std::fs::write(&outsider, b"someone else's bytes").expect("the outsider writes");
    let staged = f.out.join(".moxie-repack-manifest");
    let _ = std::fs::remove_file(&staged);
    std::fs::hard_link(&outsider, &staged).expect("the link is made");

    let r = f.repack(&["--take-over-interrupted-run"]);
    assert_ne!(
        r.status, 0,
        "a run published through a hard link: {}{}",
        r.stdout, r.stderr
    );
    assert!(
        r.stdout.contains("names") || r.stderr.contains("names"),
        "the refusal does not name the aliasing: {}{}",
        r.stdout,
        r.stderr
    );
    assert_eq!(
        std::fs::read(&outsider).expect("the outsider reads"),
        b"someone else's bytes",
        "the link's target was written through"
    );
    assert!(!f.out.join("manifest.toml").exists(), "it published anyway");
}

/// A resume whose plan differs is refused **before** the destination changes.
///
/// Review cancelled a run, changed the selected role and resumed: the refusal
/// was correct, and the existing shard had already been rewritten by the header
/// pass that ran ahead of the binding check.
#[test]
fn a_refused_resume_leaves_the_destination_exactly_as_it_was() {
    let f = Fixture::new("refused-resume");
    let r = f.repack(&["--test-cancel-after-units", "1"]);
    assert_eq!(r.status, 3, "{}{}", r.stdout, r.stderr);

    let before: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&f.out)
        .expect("the destination reads")
        .map(|e| {
            let p = e.expect("an entry").path();
            (
                p.file_name()
                    .expect("a name")
                    .to_string_lossy()
                    .into_owned(),
                std::fs::read(&p).expect("a file"),
            )
        })
        .collect();

    // A different selection: same destination, different plan.
    let other = f.scratch.join("other-selection.toml");
    let text = std::fs::read_to_string(&f.selection).expect("the selection reads");
    std::fs::write(
        &other,
        text.replace("model.norm.weight", "model.norm.other"),
    )
    .expect("written");
    let mut args = vec![
        "repack".to_string(),
        "--selection".into(),
        other.to_string_lossy().into_owned(),
        "--source-root".into(),
        f.src.to_string_lossy().into_owned(),
        "--out".into(),
        f.out.to_string_lossy().into_owned(),
        "--take-over-interrupted-run".into(),
    ];
    args.extend(budgets().iter().map(|s| s.to_string()));
    let argv: Vec<&str> = args.iter().map(String::as_str).collect();
    let r = run(&argv);
    assert_ne!(
        r.status, 0,
        "a different plan resumed: {}{}",
        r.stdout, r.stderr
    );

    let after: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&f.out)
        .expect("the destination reads")
        .map(|e| {
            let p = e.expect("an entry").path();
            (
                p.file_name()
                    .expect("a name")
                    .to_string_lossy()
                    .into_owned(),
                std::fs::read(&p).expect("a file"),
            )
        })
        .collect();
    assert_eq!(
        before.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>(),
        "a refused resume changed which files the destination holds"
    );
    for (name, bytes) in &before {
        assert_eq!(
            bytes,
            after.get(name).expect("the same file"),
            "a refused resume rewrote '{name}'"
        );
    }
}

/// An ordinary relative `--out` is an ordinary destination.
#[test]
fn a_relative_destination_is_resolved_against_the_current_directory() {
    let f = Fixture::new("relative-out");
    let here = f.scratch.join("cwd");
    std::fs::create_dir_all(&here).expect("a working directory");
    let mut args = vec![
        "repack".to_string(),
        "--selection".into(),
        f.selection.to_string_lossy().into_owned(),
        "--source-root".into(),
        f.src.to_string_lossy().into_owned(),
        "--out".into(),
        "new-output".into(),
    ];
    args.extend(budgets().iter().map(|s| s.to_string()));
    let out = std::process::Command::new(common::binary())
        .args(&args)
        .current_dir(&here)
        .output()
        .expect("the repack binary runs");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert_eq!(
        out.status.code(),
        Some(0),
        "a relative --out was refused: {stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        here.join("new-output").join("manifest.toml").exists(),
        "nothing was published where the user was standing"
    );
}
