//! Task 0027: the path a normal user takes.
//!
//! ```text
//! moxie-repack plan   --source-root <checkpoint> --out-plan <file>
//! moxie-repack repack --plan <file> --out <dir>
//! ```
//!
//! No hand-edited TOML, no expert flags, no budgets. On **fixtures**: this
//! instruction authorizes no bulk conversion of a real checkpoint, and a
//! fixture is what can assert exact bytes anyway.

mod common;

use common::{Entry, Module, Scratch, bf16_bytes, binary, run, write_shard};

/// A checkpoint: `config.json`, an index, and shards.
fn checkpoint(scratch: &Scratch, symmetric: bool) -> std::path::PathBuf {
    checkpoint_with_group(scratch, symmetric, 32)
}

fn checkpoint_with_group(scratch: &Scratch, symmetric: bool, group: usize) -> std::path::PathBuf {
    let root = scratch.join("checkpoint");
    std::fs::create_dir_all(&root).expect("a checkpoint directory");

    let rows = 4usize;
    let columns = group * 2;
    let groups = columns / group;
    let m = Module {
        rows,
        columns,
        group,
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
        zeros: if symmetric {
            Vec::new()
        } else {
            (0..rows)
                .map(|o| {
                    (0..groups)
                        .map(|g| ((o * 5 + g * 11) % 16) as u32)
                        .collect()
                })
                .collect()
        },
    };
    let module = "model.layers.0.mlp.down_proj";
    let mut entries = vec![
        Entry::new(
            &format!("{module}.weight_packed"),
            "I32",
            m.packed_shape(),
            m.packed(),
        ),
        Entry::new(
            &format!("{module}.weight_scale"),
            "BF16",
            m.scale_shape(),
            m.scale_payload("BF16"),
        ),
        Entry::new(
            &format!("{module}.weight_shape"),
            "I64",
            vec![2],
            m.weight_shape(),
        ),
    ];
    if !symmetric {
        entries.push(Entry::new(
            &format!("{module}.weight_zero_point"),
            "I32",
            m.zero_point_shape(),
            m.zero_point(),
        ));
    }
    entries.push(Entry::new(
        "model.norm.weight",
        "BF16",
        vec![8],
        bf16_bytes(1.0).repeat(8),
    ));
    write_shard(&root.join("model-00001-of-00001.safetensors"), &entries);

    let mut weight_map = String::from("{\n \"metadata\": {},\n \"weight_map\": {\n");
    for (i, e) in entries.iter().enumerate() {
        weight_map.push_str(&format!(
            "  \"{}\": \"model-00001-of-00001.safetensors\"{}\n",
            e.name,
            if i + 1 == entries.len() { "" } else { "," }
        ));
    }
    weight_map.push_str(" }\n}\n");
    std::fs::write(root.join("model.safetensors.index.json"), weight_map).expect("an index");

    let sym = if symmetric { "true" } else { "false" };
    std::fs::write(
        root.join("config.json"),
        format!(
            r#"{{
  "model_type": "fixture",
  "architectures": ["FixtureForCausalLM"],
  "quantization_config": {{
    "format": "pack-quantized",
    "quant_method": "compressed-tensors",
    "ignore": [],
    "config_groups": {{
      "group_0": {{
        "weights": {{
          "num_bits": 4, "group_size": {group}, "symmetric": {sym},
          "strategy": "group", "type": "int", "actorder": null
        }}
      }}
    }}
  }}
}}"#
        ),
    )
    .expect("a config");
    root
}

/// The whole path, with nothing a user has to know.
#[test]
fn plan_then_repack_needs_no_flags_and_no_hand_editing() {
    let scratch = Scratch::new("two-command");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    let out = scratch.join("artifact");

    // One: plan. No budgets, no selection, nothing about packing.
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "plan failed: {}{}", r.stdout, r.stderr);
    assert!(r.stdout.contains("outcome: planned"), "{}", r.stdout);
    assert!(plan.is_file(), "no plan was written");
    // What it tells the user to run next is the **two-command** form. It used
    // to print the advanced invocation -- `--selection`, `--source-root` and
    // five budgets -- which is exactly what this path exists to remove.
    assert!(
        r.stdout.contains("repack --plan ") && r.stdout.contains(" --out <dir>"),
        "the next command it printed is not the two-command form: {}",
        r.stdout
    );
    assert!(
        !r.stdout.contains("<budgets>"),
        "it still asks the user for budgets: {}",
        r.stdout
    );

    // The checkpoint is untouched: a source root is a read-only input.
    let names: Vec<String> = std::fs::read_dir(&root)
        .expect("the checkpoint reads")
        .map(|e| {
            e.expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names.len(),
        3,
        "planning wrote into the checkpoint: {names:?}"
    );

    // Two: repack. The plan carries the source root and the budgets.
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "repack failed: {}{}", r.stdout, r.stderr);
    assert!(r.stdout.contains("outcome: published"), "{}", r.stdout);
    // The settings it used are reported, not hidden.
    assert!(r.stdout.contains("settings: total="), "{}", r.stdout);
    assert!(out.join("manifest.toml").is_file(), "nothing was published");

    // And it verifies through the production reader.
    let r = run(&[
        "verify",
        "--artifact",
        out.to_str().expect("utf-8"),
        "--scratch-bytes",
        "1MiB",
    ]);
    assert_eq!(r.status, 0, "verify failed: {}{}", r.stdout, r.stderr);
}

/// A symmetric checkpoint plans with three tensors per module, not four.
#[test]
fn a_symmetric_checkpoint_plans_and_repacks() {
    let scratch = Scratch::new("two-command-symmetric");
    let root = checkpoint(&scratch, true);
    let plan = scratch.join("model.plan.toml");
    let out = scratch.join("artifact");

    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    let text = std::fs::read_to_string(&plan).expect("the plan reads");
    assert!(
        text.contains("zero_points = \"symmetric\""),
        "the plan did not record symmetry:\n{text}"
    );
    assert!(
        !text.contains("weight_zero_point"),
        "a symmetric plan named a zero-point tensor:\n{text}"
    );

    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    assert!(out.join("manifest.toml").is_file());
}

/// An existing plan is not silently replaced.
#[test]
fn planning_does_not_overwrite_an_existing_plan() {
    let scratch = Scratch::new("two-command-force");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    std::fs::write(&plan, "someone else's file").expect("a file in the way");

    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_ne!(r.status, 0, "it overwrote the file: {}", r.stdout);
    assert!(
        r.stderr.contains("--force"),
        "the refusal does not say how to proceed: {}",
        r.stderr
    );
    assert_eq!(
        std::fs::read_to_string(&plan).expect("it reads"),
        "someone else's file",
        "the existing file was replaced anyway"
    );

    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
        "--force",
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    assert!(
        std::fs::read_to_string(&plan)
            .expect("it reads")
            .contains("version = 2")
    );
}

/// A partial plan is refused by the normal path and converted only on request.
#[test]
fn a_partial_plan_is_refused_unless_it_is_asked_for() {
    let scratch = Scratch::new("two-command-partial");
    let root = checkpoint(&scratch, false);
    // An F16 tensor the importer has no passthrough for, named by the index:
    // the plan must mark the model partial rather than omit it.
    let shard = root.join("model-00002-of-00002.safetensors");
    write_shard(
        &shard,
        &[Entry::new(
            "model.extra.weight",
            "F16",
            vec![4],
            vec![0u8; 8],
        )],
    );
    let index = root.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index).expect("the index reads");
    let patched = text.replace(
        " }\n}",
        ",\n  \"model.extra.weight\": \"model-00002-of-00002.safetensors\"\n }\n}",
    );
    assert_ne!(patched, text, "the index was not extended");
    std::fs::write(&index, patched).expect("the index writes");

    let plan = scratch.join("model.plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    let text = std::fs::read_to_string(&plan).expect("the plan reads");
    assert!(
        text.contains("status = \"partial\""),
        "an uncovered tensor did not make the plan partial:\n{text}"
    );

    let out = scratch.join("artifact");
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_ne!(
        r.status, 0,
        "a partial plan converted silently: {}",
        r.stdout
    );
    assert!(
        r.stdout.contains("--allow-partial") || r.stderr.contains("--allow-partial"),
        "the refusal does not say how to proceed: {}{}",
        r.stdout,
        r.stderr
    );
    assert!(!out.join("manifest.toml").exists());

    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
        "--allow-partial",
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    assert!(out.join("manifest.toml").is_file());
}

/// A checkpoint whose packing this repository has not measured is refused, by
/// name, rather than partially converted.
#[test]
fn an_unmeasured_packing_is_refused_by_name() {
    let scratch = Scratch::new("two-command-autoround");
    let root = checkpoint(&scratch, false);
    let config = root.join("config.json");
    let text = std::fs::read_to_string(&config).expect("it reads");
    std::fs::write(
        &config,
        text.replace("\"format\": \"pack-quantized\",", "\"format\": \"\",")
            .replace("\"compressed-tensors\"", "\"auto-round\""),
    )
    .expect("it writes");

    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        scratch.join("model.plan.toml").to_str().expect("utf-8"),
    ]);
    assert_ne!(r.status, 0, "an unmeasured packing planned: {}", r.stdout);
    assert!(
        r.stderr.contains("auto-round"),
        "the refusal does not name what it found: {}",
        r.stderr
    );
    assert!(
        !scratch.join("model.plan.toml").exists(),
        "a refused plan was written anyway"
    );
    let _ = binary();
}

/// A module whose tensors span shards is expanded, not dropped.
///
/// `to_plan_toml` writes such a module **only** under `[[weights.split]]`, and
/// expansion iterated `weights.modules` alone. The plan said `complete`, the
/// expansion produced one fewer tensor, and the artifact published without the
/// quantized module entirely.
#[test]
fn a_split_module_survives_the_round_trip() {
    let scratch = Scratch::new("split-module");
    let root = scratch.join("checkpoint");
    std::fs::create_dir_all(&root).expect("a checkpoint directory");

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
        zeros: Vec::new(),
    };
    let module = "model.layers.0.mlp.down_proj";
    // packed and shape in one shard, scale in another: the measured exception.
    write_shard(
        &root.join("a.safetensors"),
        &[
            Entry::new(
                &format!("{module}.weight_packed"),
                "I32",
                m.packed_shape(),
                m.packed(),
            ),
            Entry::new(
                &format!("{module}.weight_shape"),
                "I64",
                vec![2],
                m.weight_shape(),
            ),
            Entry::new(
                "model.norm.weight",
                "BF16",
                vec![8],
                bf16_bytes(1.0).repeat(8),
            ),
        ],
    );
    write_shard(
        &root.join("b.safetensors"),
        &[Entry::new(
            &format!("{module}.weight_scale"),
            "BF16",
            m.scale_shape(),
            m.scale_payload("BF16"),
        )],
    );
    std::fs::write(
        root.join("model.safetensors.index.json"),
        format!(
            r#"{{"metadata":{{}},"weight_map":{{
 "{module}.weight_packed":"a.safetensors",
 "{module}.weight_shape":"a.safetensors",
 "{module}.weight_scale":"b.safetensors",
 "model.norm.weight":"a.safetensors"
}}}}"#
        ),
    )
    .expect("an index");
    std::fs::write(
        root.join("config.json"),
        r#"{"model_type":"fixture","architectures":["FixtureForCausalLM"],
 "quantization_config":{"format":"pack-quantized","quant_method":"compressed-tensors",
 "ignore":[],"config_groups":{"group_0":{"weights":{"num_bits":4,"group_size":32,
 "symmetric":true,"strategy":"group","type":"int","actorder":null}}}}}"#,
    )
    .expect("a config");

    let plan = scratch.join("model.plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    let text = std::fs::read_to_string(&plan).expect("the plan reads");
    assert!(
        text.contains("[[weights.split]]"),
        "the fixture did not produce a split module:\n{text}"
    );
    assert!(text.contains("status = \"complete\""), "{text}");

    let out = scratch.join("artifact");
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);

    // The quantized module is **in** the artifact, not silently missing.
    let manifest = std::fs::read_to_string(out.join("manifest.toml")).expect("a manifest");
    assert!(
        manifest.contains(&format!("{module}.weight")),
        "the split module was dropped from the artifact:\n{manifest}"
    );
    assert!(manifest.contains("model.norm.weight"));
}

/// A staging path that is a symbolic link is refused, and its target untouched.
///
/// `fs::write` follows a link. Independent review pointed `<plan>.toml.partial`
/// at a checkpoint's `config.json` and watched planning overwrite it, before
/// the atomic rename ever came into it.
#[test]
#[cfg(unix)]
fn a_symlinked_staging_path_is_refused_and_its_target_untouched() {
    let scratch = Scratch::new("symlink-staging");
    let root = checkpoint(&scratch, false);
    let victim = root.join("config.json");
    let before = std::fs::read(&victim).expect("the config reads");

    let plan = scratch.join("victim.toml");
    let staging = scratch.join("victim.toml.partial");
    std::os::unix::fs::symlink(&victim, &staging).expect("the link is made");

    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_ne!(r.status, 0, "it wrote through the link: {}", r.stdout);
    assert_eq!(
        std::fs::read(&victim).expect("the config reads"),
        before,
        "the link's target was overwritten"
    );
    assert!(!plan.exists(), "a plan was published anyway");
}

/// A plan is refused once its checkpoint has gained a tensor.
///
/// Every shard the plan named was byte-identical; the model was not the same
/// model. A source digest alone cannot see that.
#[test]
fn a_plan_whose_checkpoint_changed_is_refused() {
    let scratch = Scratch::new("stale-plan");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);

    // A new indexed tensor in a new shard: the plan still describes the old set.
    write_shard(
        &root.join("model-00002-of-00002.safetensors"),
        &[Entry::new(
            "model.added.weight",
            "BF16",
            vec![4],
            bf16_bytes(1.0).repeat(4),
        )],
    );
    let index = root.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index).expect("the index reads");
    let patched = text.replace(
        " }\n}",
        ",\n  \"model.added.weight\": \"model-00002-of-00002.safetensors\"\n }\n}",
    );
    assert_ne!(patched, text, "the index was not extended");
    std::fs::write(&index, patched).expect("the index writes");

    let out = scratch.join("artifact");
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_ne!(
        r.status, 0,
        "a stale plan published the old subset as complete: {}",
        r.stdout
    );
    assert!(
        r.stdout.contains("has changed") || r.stderr.contains("has changed"),
        "the refusal does not name the change: {}{}",
        r.stdout,
        r.stderr
    );
    assert!(!out.join("manifest.toml").exists());
}

/// One override is honoured, and the rest still come from the plan.
#[test]
fn a_single_override_does_not_discard_the_others() {
    let scratch = Scratch::new("override-precedence");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);

    // One flag, deliberately too small to convert with. It must be the value
    // the run uses, not silently replaced by the plan's.
    let out = scratch.join("artifact");
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
        "--disk-bytes",
        "1",
    ]);
    assert!(
        r.stdout.contains("disk=1"),
        "the override was discarded: {}{}",
        r.stdout,
        r.stderr
    );
    assert_ne!(r.status, 0, "a 1-byte disk budget converted: {}", r.stdout);
}

/// A plan made with a relative root is usable from another directory.
#[test]
fn a_plan_records_an_absolute_source_root() {
    let scratch = Scratch::new("relative-root");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");

    // Generated with a *relative* --source-root, from the checkpoint's parent.
    let out = std::process::Command::new(common::binary())
        .args([
            "plan",
            "--source-root",
            "checkpoint",
            "--out-plan",
            plan.to_str().expect("utf-8"),
        ])
        .current_dir(root.parent().expect("a parent"))
        .output()
        .expect("the binary runs");
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&plan).expect("the plan reads");
    assert!(
        text.contains(&format!("root = \"{}\"", root.display())),
        "the plan did not record an absolute root:\n{}",
        text.lines().find(|l| l.starts_with("root =")).unwrap_or("")
    );

    // And it converts from an unrelated directory.
    let artifact = scratch.join("artifact");
    let out = std::process::Command::new(common::binary())
        .args([
            "repack",
            "--plan",
            plan.to_str().expect("utf-8"),
            "--out",
            artifact.to_str().expect("utf-8"),
        ])
        .current_dir("/tmp")
        .output()
        .expect("the binary runs");
    assert!(
        out.status.success(),
        "a plan made with a relative root failed elsewhere: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A BF16-only checkpoint of `count` tensors, for the sizing rules.
///
/// Every tensor is eight bytes. What is being measured is the **count**: one
/// journal record per work unit, and a unit never spans two tensors.
fn bf16_checkpoint(scratch: &Scratch, name: &str, count: usize) -> std::path::PathBuf {
    let root = scratch.join(name);
    std::fs::create_dir_all(&root).expect("a checkpoint directory");
    let shard = "model-00001-of-00001.safetensors";

    let entries: Vec<Entry> = (0..count)
        .map(|i| {
            Entry::new(
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                "BF16",
                vec![4],
                bf16_bytes(1.0).repeat(4),
            )
        })
        .collect();
    write_shard(&root.join(shard), &entries);

    let mut index = String::from("{\n \"metadata\": {},\n \"weight_map\": {\n");
    for (i, e) in entries.iter().enumerate() {
        index.push_str(&format!(
            "  \"{}\": \"{shard}\"{}\n",
            e.name,
            if i + 1 == entries.len() { "" } else { "," }
        ));
    }
    index.push_str(" }\n}\n");
    std::fs::write(root.join("model.safetensors.index.json"), index).expect("an index");
    std::fs::write(
        root.join("config.json"),
        "{\n  \"model_type\": \"fixture\",\n  \"architectures\": [\"FixtureForCausalLM\"]\n}\n",
    )
    .expect("a config");
    root
}

/// The `[binding]` digests describe the checkpoint the plan was **built from**,
/// even when it changes before the plan is written.
///
/// The digests used to be fields a caller filled in by reading `config.json`
/// and the index a second time. Independent review changed the index between
/// the two reads: the plan's selection then described the old checkpoint and
/// its binding described the new one, so `repack` confirmed the binding and
/// published the old subset under `completeness = "complete"`.
///
/// The test changes the index in exactly that window -- after `discover`
/// returns, before `to_plan_toml` is called -- and nothing in it can supply a
/// digest, because `SourceBinding` no longer has the fields.
#[test]
fn a_plan_is_bound_to_the_checkpoint_it_was_built_from() {
    let scratch = Scratch::new("binding-window");
    let root = checkpoint(&scratch, false);
    let index = root.join("model.safetensors.index.json");
    let as_discovered = std::fs::read(&index).expect("the index reads");

    let budgets = moxie_repack::Budgets {
        total_bytes: 2 << 30,
        header_bytes: 64 << 20,
        scratch_bytes: 1 << 20,
        chunk_file_bytes: 1 << 30,
        disk_bytes: 1 << 30,
    };
    let mut sources = moxie_repack::open_sources(&root, &budgets).expect("the sources");
    let discovery = moxie_repack::discover::discover(&root, &mut sources).expect("a discovery");

    // **The window.** The checkpoint gains a tensor between being read and
    // being described.
    write_shard(
        &root.join("model-00002-of-00002.safetensors"),
        &[Entry::new(
            "model.added.weight",
            "BF16",
            vec![4],
            bf16_bytes(1.0).repeat(4),
        )],
    );
    let text = std::fs::read_to_string(&index).expect("the index reads");
    let patched = text.replace(
        " }\n}",
        ",\n  \"model.added.weight\": \"model-00002-of-00002.safetensors\"\n }\n}",
    );
    assert_ne!(patched, text, "the index was not extended");
    std::fs::write(&index, patched).expect("the index writes");

    let binding = moxie_repack::discover::SourceBinding {
        root: root.to_string_lossy().into_owned(),
        model: "fixture/binding-window".into(),
        revision: None,
        license: "fixture".into(),
        quantizer: "declared-by-the-checkpoint".into(),
        tokenizer: moxie_repack::discover::asset_identity(&root, "tokenizer.json"),
        template: moxie_repack::discover::asset_identity(&root, "chat_template.jinja"),
        recorded_digests: Vec::new(),
        options: moxie_repack::discover::automatic_budgets(&discovery).expect("budgets"),
    };
    let plan_text =
        moxie_repack::discover::to_plan_toml(&discovery, &binding).expect("a plan document");

    // Bound to what it describes: the index as discovery parsed it.
    let expected = moxie_format::sha256_hex(&as_discovered);
    assert!(
        plan_text.contains(&format!("index_sha256 = \"{expected}\"")),
        "the plan is bound to a checkpoint it did not describe:\n{}",
        plan_text
            .lines()
            .find(|l| l.starts_with("index_sha256"))
            .unwrap_or("(no index_sha256 line)")
    );

    // And so the second command refuses it, rather than publishing the old
    // subset of a model that has changed.
    let plan = scratch.join("window.plan.toml");
    std::fs::write(&plan, &plan_text).expect("the plan writes");
    let out = scratch.join("artifact");
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_ne!(
        r.status, 0,
        "a plan bound to a different moment converted: {}",
        r.stdout
    );
    assert!(!out.join("manifest.toml").exists());
}

/// A checkpoint with more tensors than a resume journal can record is refused
/// while planning, not handed a scratch size the reader rejects.
///
/// Every work unit appends one journal line and a unit never spans two tensors,
/// so the record count has a floor no tile size lowers. The sizing loop raised
/// the scratch against that floor until it hit 1 GiB, wrote the plan, and
/// `repack` then failed on the planner's own number: `536870912 is not a valid
/// read budget`.
#[test]
fn a_checkpoint_the_journal_cannot_record_is_refused_while_planning() {
    let scratch = Scratch::new("journal-floor");
    // Independent review's own fixture size.
    let root = bf16_checkpoint(&scratch, "many", 32_000);
    let plan = scratch.join("many.plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_ne!(
        r.status, 0,
        "a plan its own defaults reject was emitted: {}",
        r.stdout
    );
    assert!(
        r.stderr.contains("journal"),
        "the refusal does not say what cannot fit: {}{}",
        r.stdout,
        r.stderr
    );
    assert!(!plan.exists(), "a refused plan was left behind");

    // The whole contract, so a future sizing rule cannot satisfy this by
    // emitting an unusable plan instead: whatever is written has to be usable.
    if plan.exists() {
        let text = std::fs::read_to_string(&plan).expect("the plan reads");
        let scratch_bytes: u64 = text
            .lines()
            .find_map(|l| l.strip_prefix("scratch_bytes = "))
            .and_then(|v| v.trim().parse().ok())
            .expect("the plan records a scratch size");
        assert!(
            scratch_bytes / 2 <= 256 * 1024 * 1024,
            "a {scratch_bytes} byte scratch implies a tile the reader refuses"
        );
    }
}

/// Whatever the tensor count, a plan that is emitted carries a payload tile the
/// reader admits -- which is what `repack` opens its sources with.
#[test]
fn an_emitted_plan_carries_a_tile_the_reader_admits() {
    let scratch = Scratch::new("tile-limit");
    let root = bf16_checkpoint(&scratch, "moderate", 2_000);
    let plan = scratch.join("moderate.plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        plan.to_str().expect("utf-8"),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);

    let text = std::fs::read_to_string(&plan).expect("the plan reads");
    let scratch_bytes: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("scratch_bytes = "))
        .and_then(|v| v.trim().parse().ok())
        .expect("the plan records a scratch size");
    // `ByteBudget` admits at most 256 MiB, and two tiles come out of the
    // scratch. A plan above this is one `open_sources` refuses.
    assert!(
        scratch_bytes / 2 <= 256 * 1024 * 1024,
        "a {scratch_bytes} byte scratch implies a tile the reader refuses"
    );

    // Proved by using it, not by reading it.
    let out = scratch.join("artifact");
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().expect("utf-8"),
        "--out",
        out.to_str().expect("utf-8"),
    ]);
    assert_eq!(
        r.status, 0,
        "the planner's own budgets were refused by repack: {}{}",
        r.stdout, r.stderr
    );
}

/// Planning never writes inside the checkpoint it is reading.
///
/// The default output resolves against the current directory, so running the
/// command from inside a checkpoint put the plan -- and the staging file it is
/// renamed from -- among the source files. An explicit `--out-plan` pointing in
/// there had the same gap.
#[test]
fn planning_refuses_to_write_inside_the_checkpoint() {
    let scratch = Scratch::new("write-inside");
    let root = checkpoint(&scratch, false);
    let before: std::collections::BTreeSet<String> = std::fs::read_dir(&root)
        .expect("the checkpoint lists")
        .map(|e| {
            e.expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    // The default path, from inside the checkpoint.
    let out = std::process::Command::new(binary())
        .args(["plan", "--source-root", "."])
        .current_dir(&root)
        .output()
        .expect("the binary runs");
    assert!(
        !out.status.success(),
        "planning wrote into the checkpoint it was reading: {}",
        String::from_utf8_lossy(&out.stdout)
    );

    // And an explicit path inside it.
    let inside = root.join("plan.toml");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().expect("utf-8"),
        "--out-plan",
        inside.to_str().expect("utf-8"),
    ]);
    assert_ne!(
        r.status, 0,
        "an explicit path inside the checkpoint was accepted: {}",
        r.stdout
    );

    let after: std::collections::BTreeSet<String> = std::fs::read_dir(&root)
        .expect("the checkpoint lists")
        .map(|e| {
            e.expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        before,
        after,
        "the checkpoint gained files: {:?}",
        after.difference(&before).collect::<Vec<_>>()
    );
}

/// Task 0028's review, finding 5: an edited plan cannot publish a subset as
/// complete.
///
/// The guard downstream read the plan's own `completeness` field, and the
/// binding check asked a different question -- has the *checkpoint* changed --
/// which an edit to the plan leaves answered "no". The review deleted one
/// tensor entry from a generated plan, ran the ordinary `repack --plan`, and
/// got a one-tensor artifact marked `complete` with a success exit.
#[test]
fn a_plan_edited_to_cover_less_cannot_publish_as_complete() {
    let scratch = Scratch::new("edited-plan");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    let out = scratch.join("out");

    let planned = common::run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
    ]);
    assert_eq!(planned.outcome(), "planned", "{}", planned.stdout);
    let text = std::fs::read_to_string(&plan).expect("the plan reads");
    assert!(
        text.contains("status = \"complete\""),
        "the fixture plan is not complete to begin with:\n{text}"
    );

    // Delete one entry from the `[bf16]` tensor list, leaving `completeness`
    // untouched -- which is exactly what an editor does. The plan is ADR 0026's
    // compact form, so an entry is a line in a list rather than a table.
    let at = text.find("\n[bf16]\ntensors = [\n").expect("a bf16 list");
    let first = at + "\n[bf16]\ntensors = [\n".len();
    let line_end = text[first..].find('\n').expect("an entry line") + first + 1;
    let edited = format!("{}{}", &text[..first], &text[line_end..]);
    assert!(
        edited.len() < text.len() && edited.contains("status = \"complete\""),
        "the edit did not remove an entry, or removed the completeness line"
    );
    std::fs::write(&plan, &edited).expect("the edited plan writes");

    let refused = common::run(&[
        "repack",
        "--plan",
        plan.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    // Non-zero and nothing readable, the same shape
    // `a_plan_whose_checkpoint_changed_is_refused` asserts: which outcome word
    // this lands on is the program's existing classification and not what the
    // guard is for.
    assert_ne!(
        refused.status, 0,
        "an edited plan published:\n{}{}",
        refused.stdout, refused.stderr
    );
    assert!(
        refused.says("says it is complete"),
        "the refusal does not say what is wrong:\n{}{}",
        refused.stdout,
        refused.stderr
    );
    // Nothing readable was produced. A partial artifact that opens is the most
    // expensive kind of wrong, and this one must not exist at all.
    assert!(
        !out.join("manifest.toml").exists(),
        "a manifest was published for a plan that was refused"
    );
}

/// The re-review of the same finding: a substitution that keeps the **count**.
///
/// The first repair counted what the plan accounted for and compared the total
/// with the bound `index_tensors`. Review then replaced one entry with a
/// different tensor the index already named -- same total, same digests -- and
/// published a `complete` artifact carrying one tensor twice and the other not
/// at all. A total is a shadow of a set, and two different sets cast the same
/// one, so the exact `(tensor name, shard)` pairs are what is compared now.
#[test]
fn a_plan_edited_to_swap_an_entry_cannot_publish_as_complete() {
    let scratch = Scratch::new("swapped-plan");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    let out = scratch.join("out");

    let planned = common::run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
    ]);
    assert_eq!(planned.outcome(), "planned", "{}", planned.stdout);
    let text = std::fs::read_to_string(&plan).expect("the plan reads");

    // Replace the standalone BF16 tensor with a tensor the index **already**
    // names -- one of the quantized module's own companions. The entry count
    // does not move, the completeness line does not move, and the checkpoint is
    // untouched, so every digest still matches.
    let victim = "model.norm.weight";
    let duplicate = "model.layers.0.mlp.down_proj.weight_scale";
    assert!(
        text.contains(victim),
        "the fixture plan does not name {victim}:\n{text}"
    );
    let edited = text.replace(victim, duplicate);
    assert_ne!(edited, text, "the substitution changed nothing");
    assert_eq!(
        edited.matches('\n').count(),
        text.matches('\n').count(),
        "the substitution changed the line count, so it is not a same-count edit"
    );
    assert!(
        edited.contains("status = \"complete\""),
        "the substitution removed the completeness line"
    );
    std::fs::write(&plan, &edited).expect("the edited plan writes");

    let refused = common::run(&[
        "repack",
        "--plan",
        plan.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_ne!(
        refused.status, 0,
        "a same-count substitution published:\n{}{}",
        refused.stdout, refused.stderr
    );
    assert!(
        refused.says("more than once") || refused.says("does not describe this checkpoint"),
        "the refusal does not name the substitution:\n{}{}",
        refused.stdout,
        refused.stderr
    );
    assert!(
        !out.join("manifest.toml").exists(),
        "a manifest was published for a plan that was refused"
    );
}

/// A `--plan` document with its `[binding]` block removed is refused.
///
/// The exact-set comparison, the config digest and the index digest are all
/// measured **against the binding**, and `confirm_binding` returned success the
/// moment the section was absent -- which the parser allows, because a
/// hand-written selection has none. Review deleted the block, deleted a tensor,
/// and published a `complete` artifact missing it. The optional section belongs
/// to `--selection`; `--plan` says this program wrote the document.
#[test]
fn a_plan_without_its_binding_is_refused_rather_than_unchecked() {
    let scratch = Scratch::new("unbound-plan");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("model.plan.toml");
    let out = scratch.join("out");

    let planned = common::run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
    ]);
    assert_eq!(planned.outcome(), "planned", "{}", planned.stdout);
    let text = std::fs::read_to_string(&plan).expect("the plan reads");

    // Remove the whole `[binding]` block, and one tensor with it -- the exact
    // sequence review reproduced.
    let at = text.find("\n[binding]\n").expect("a binding section");
    let after = text[at + 1..]
        .find("\n[")
        .map(|i| at + 1 + i)
        .expect("a section after the binding");
    let unbound = format!("{}{}", &text[..at], &text[after..]);
    assert!(
        !unbound.contains("[binding]") && unbound.contains("status = \"complete\""),
        "the edit did not remove the binding, or removed the completeness line"
    );
    let victim = "model.norm.weight";
    let at = unbound.find(victim).expect("the bf16 tensor is named");
    let line_start = unbound[..at].rfind('\n').expect("a line start") + 1;
    let line_end = unbound[at..].find('\n').expect("a line end") + at + 1;
    let edited = format!("{}{}", &unbound[..line_start], &unbound[line_end..]);
    assert!(
        !edited.contains(victim),
        "the tensor was not removed:\n{edited}"
    );
    std::fs::write(&plan, &edited).expect("the edited plan writes");

    let refused = common::run(&[
        "repack",
        "--plan",
        plan.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_ne!(
        refused.status, 0,
        "an unbound plan published a subset as complete:\n{}{}",
        refused.stdout, refused.stderr
    );
    assert!(
        refused.says("[binding]"),
        "the refusal does not name what is missing:\n{}{}",
        refused.stdout,
        refused.stderr
    );
    assert!(
        !out.join("manifest.toml").exists(),
        "a manifest was published for a plan that was refused"
    );
}

/// Task 0028's review, finding 7: a file this program did not create is not
/// this program's to delete.
///
/// Any regular file at the predictable staging path was treated as an
/// interrupted plan's leftover and removed. The review put unrelated data
/// there, ran an ordinary `plan`, and had it deleted with a success exit.
#[test]
fn planning_refuses_to_delete_a_staging_path_it_did_not_create() {
    let scratch = Scratch::new("staging-ownership");
    let root = checkpoint(&scratch, false);
    let plan = scratch.join("result.toml");
    let staging = plan.with_extension("toml.partial");
    let precious = b"someone else's bytes";
    std::fs::write(&staging, precious).expect("the other file writes");

    let refused = common::run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
    ]);
    assert_eq!(
        refused.outcome(),
        "refused",
        "planning proceeded over a file it did not create:\n{}{}",
        refused.stdout,
        refused.stderr
    );
    assert_eq!(
        std::fs::read(&staging).expect("the other file survives"),
        precious,
        "planning deleted a file it did not create"
    );
    assert!(refused.says("--force"), "the refusal names the way out");

    // `--force` already means "replace an existing plan", so it is where the
    // user says the leftover is theirs to clear. Then it proceeds.
    let forced = common::run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
        "--force",
    ]);
    assert_eq!(
        forced.outcome(),
        "planned",
        "{}{}",
        forced.stdout,
        forced.stderr
    );
    assert!(plan.exists(), "the plan was not written");
    assert!(!staging.exists(), "the staging file outlived the rename");
}

/// Task 0028's review, finding 6: the size cap applies to the first read.
///
/// `repack --plan` read the document with `read_to_string` and only the later
/// `read_selection` applied the cap, so an arbitrarily large file was already
/// resident by the time anything checked -- an unbounded allocation in a
/// program whose whole point is bounded memory, before any of it was admitted.
#[test]
fn an_oversized_plan_is_refused_by_the_read_that_first_touches_it() {
    let scratch = Scratch::new("oversized-plan");
    let plan = scratch.join("huge.plan.toml");
    let mut text = String::from("version = 1\n");
    // One byte over the cap, built from a comment so the document would
    // otherwise be parseable.
    text.push_str("# ");
    while text.len() <= moxie_format::selection::MAX_SELECTION_BYTES {
        text.push('x');
    }
    text.push('\n');
    std::fs::write(&plan, &text).expect("the oversized plan writes");

    let refused = common::run(&[
        "repack",
        "--plan",
        plan.to_str().unwrap(),
        "--out",
        scratch.join("out").to_str().unwrap(),
    ]);
    assert_ne!(refused.status, 0, "an oversized plan was accepted");
    assert!(
        refused.says("cannot read") || refused.says("byte"),
        "the refusal does not name the size:\n{}{}",
        refused.stdout,
        refused.stderr
    );
}

#[test]
fn static_group128_repack_preserves_all_codes_and_group_scales() {
    use moxie_format::affine::{AffineDescriptor, Grouping, IntWidth};
    use moxie_format::payload::{self, ZeroPointSection};
    use moxie_format::scale::ScaleDtype;
    let scratch = Scratch::new("static-group128");
    let root = checkpoint_with_group(&scratch, true, 128);
    let config = root.join("config.json");
    let text = std::fs::read_to_string(&config)
        .unwrap()
        .replace("\"actorder\": null", "\"actorder\": \"static\"");
    std::fs::write(&config, text).unwrap();
    let plan = scratch.join("plan.toml");
    let out = scratch.join("artifact");
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        plan.to_str().unwrap(),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    let r = run(&[
        "repack",
        "--plan",
        plan.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(r.status, 0, "{}{}", r.stdout, r.stderr);
    let descriptor = AffineDescriptor {
        width: IntWidth::Int4,
        out_features: 4,
        in_features: 256,
        grouping: Grouping::Contiguous { size: 128 },
        group_index: None,
        scale_dtype: ScaleDtype::Bf16,
    };
    let expected_len = payload::length_of(&descriptor, ZeroPointSection::Absent).unwrap() as usize;
    let mut bytes = Vec::with_capacity(expected_len);
    let artifact = moxie_storage::Artifact::open(&out).unwrap();
    artifact
        .stream_tensor(
            "model.layers.0.mlp.down_proj.weight",
            &mut [0; 512],
            &mut |slice| {
                bytes.extend_from_slice(slice);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(bytes.len(), expected_len);
    let tensor = payload::decode(descriptor, ZeroPointSection::Absent, &bytes).unwrap();
    for o in 0..4 {
        for (k, value) in tensor.reconstruct_row(o).unwrap().iter().enumerate() {
            let signed = ((o * 7 + k * 3) % 16) as i32 - 8;
            let scale = 0.5 + ((o + k / 128) % 3) as f32;
            assert_eq!(value.to_bits(), (signed as f32 * scale).to_bits());
        }
    }
}

#[test]
fn planning_cannot_treat_a_group_map_as_an_unrelated_skipped_tensor() {
    let scratch = Scratch::new("static-map-contradiction");
    let root = checkpoint(&scratch, true);
    let index_path = root.join("model.safetensors.index.json");
    let index = std::fs::read_to_string(&index_path).unwrap();
    let name = "model.layers.0.mlp.down_proj.weight_g_idx";
    write_shard(
        &root.join("map.safetensors"),
        &[Entry::new(name, "I32", vec![64], vec![0; 256])],
    );
    let index = index.replace(
        "\"weight_map\": {",
        &format!("\"weight_map\": {{\n \"{name}\": \"map.safetensors\","),
    );
    std::fs::write(index_path, index).unwrap();
    let r = run(&[
        "plan",
        "--source-root",
        root.to_str().unwrap(),
        "--out-plan",
        scratch.join("plan.toml").to_str().unwrap(),
    ]);
    assert_ne!(r.status, 0);
    assert!(format!("{}{}", r.stdout, r.stderr).contains("weight_g_idx"));
}
