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
          "num_bits": 4, "group_size": 32, "symmetric": {sym},
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
