//! Regressions for the review's findings that the suites above could not see:
//! malformed source declarations, a source that moves under a running
//! conversion, failure cleanup at the workflow boundary, and a budget too small
//! to do the work.
//!
//! These drive the **library** rather than the binary, because each one needs to
//! observe or interfere with the run while it is happening -- change a file
//! between two callbacks, inject a failure at a named boundary, or read the
//! ledger afterwards.

mod common;

use std::collections::BTreeMap;

use common::{Entry, Module, Scratch, SelectionBuilder, write_shard};
use moxie_repack::Budgets;
use moxie_repack::write::{Faults, Options, Site};

fn budgets() -> Budgets {
    Budgets {
        total_bytes: 128 << 20,
        header_bytes: 64 << 20,
        scratch_bytes: 64 * 1024,
        chunk_file_bytes: 4 << 20,
        disk_bytes: 64 << 20,
    }
}

fn module(rows: usize, columns: usize, symmetric: bool) -> Module {
    let groups = columns.div_ceil(32);
    Module {
        rows,
        columns,
        group: 32,
        bits: 4,
        codes: (0..rows)
            .map(|o| (0..columns).map(|k| ((o + k) % 16) as u32).collect())
            .collect(),
        scales: (0..rows)
            .map(|o| {
                (0..groups)
                    .map(|g| 0.5 + 0.25 * ((o + g) % 3) as f32)
                    .collect()
            })
            .collect(),
        zeros: if symmetric {
            Vec::new()
        } else {
            (0..rows)
                .map(|o| (0..groups).map(|g| ((o * 3 + g) % 16) as u32).collect())
                .collect()
        },
    }
}

/// Write a one-module shard, with each tensor's dtype overridable so that a
/// malformed declaration can be built deliberately.
fn shard_with(
    scratch: &Scratch,
    module: &Module,
    packed_dtype: &'static str,
    shape_dtype: &'static str,
    zero_point: bool,
) -> std::path::PathBuf {
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let name = "model.layers.0.mlp.down_proj";
    let mut entries = vec![
        Entry::new(
            &format!("{name}.weight_packed"),
            packed_dtype,
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
            shape_dtype,
            vec![2],
            module.weight_shape(),
        ),
    ];
    if zero_point {
        entries.push(Entry::new(
            &format!("{name}.weight_zero_point"),
            "I32",
            module.zero_point_shape(),
            module.zero_point(),
        ));
    }
    write_shard(&src.join("shard-a.safetensors"), &entries);
    src
}

fn selection_for(scratch: &Scratch, symmetric: bool) -> std::path::PathBuf {
    let name = "model.layers.0.mlp.down_proj";
    let mut files = BTreeMap::from([
        ("weight_packed", "shard-a.safetensors"),
        ("weight_scale", "shard-a.safetensors"),
        ("weight_shape", "shard-a.safetensors"),
    ]);
    if !symmetric {
        files.insert("weight_zero_point", "shard-a.safetensors");
    }
    let path = scratch.join("selection.toml");
    SelectionBuilder::new("synthetic")
        .pack_quantized(
            "model.layers.0.mlp.down_proj.weight",
            name,
            "int4",
            "32",
            if symmetric {
                "symmetric"
            } else {
                "packed-along-output"
            },
            &files,
        )
        .write(&path);
    path
}

/// The importer's entry rules, applied to entries resolved across shards.
///
/// A review published a module whose packed codes and zero points were declared
/// `F32` and whose shape was `F64`: the cross-shard resolver looked at shapes
/// and never at what the single-header resolver checks.
#[test]
fn a_module_whose_entries_are_the_wrong_dtype_is_refused() {
    for (packed, shape, needle) in [
        ("F32", "I64", "weight_packed is F32"),
        ("I32", "F64", "weight_shape must be I64[2]"),
    ] {
        let scratch = Scratch::new("dtype");
        let m = module(16, 64, false);
        let src = shard_with(&scratch, &m, packed, shape, true);
        let selection_path = selection_for(&scratch, false);
        let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
        let budgets = budgets();
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
        let e = moxie_repack::inspect(
            &selection,
            &mut sources,
            &budgets,
            &mut moxie_repack::ledger_for(&budgets).expect("a ledger"),
        )
        .unwrap_err();
        assert!(e.to_string().contains(needle), "{packed}/{shape}: {e}");
    }
}

/// A symmetric selection over a source that carries zero points is a
/// disagreement between two independently obtained facts, not a symmetric
/// module with a spare tensor. A review published one, silently dropping them.
#[test]
fn a_symmetric_selection_over_an_asymmetric_source_is_refused() {
    let scratch = Scratch::new("symmetric-over-asymmetric");
    let m = module(16, 64, false);
    // The shard carries zero points; the selection declares symmetric and
    // names no zero-point file at all.
    let src = shard_with(&scratch, &m, "I32", "I64", true);
    let selection_path = selection_for(&scratch, true);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
    let budgets = budgets();
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
    let e = moxie_repack::inspect(
        &selection,
        &mut sources,
        &budgets,
        &mut moxie_repack::ledger_for(&budgets).expect("a ledger"),
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("declared symmetric but serializes"),
        "{e}"
    );
}

/// A source that changes after its digest is taken must not be published under
/// that digest.
///
/// The review changed a synthetic source immediately after its digest was
/// computed, through the progress callback, and the run published the new bytes
/// under the old digest. The same interference is reproduced here.
#[test]
fn a_source_that_changes_after_it_is_hashed_is_refused() {
    let scratch = Scratch::new("source-moved");
    let m = module(16, 64, false);
    let src = shard_with(&scratch, &m, "I32", "I64", true);
    let selection_path = selection_for(&scratch, false);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
    let budgets = budgets();
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let out = scratch.join("artifact");

    let shard = src.join("shard-a.safetensors");
    let original = std::fs::read(&shard).expect("the shard");
    let e = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &Options::default(),
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |line: &str| {
            // The moment the digest is recorded, change a payload byte that no
            // declared range's length depends on.
            if line.starts_with("hashed source") {
                let mut bytes = original.clone();
                let at = bytes.len() - 1;
                bytes[at] ^= 0xFF;
                std::fs::write(&shard, &bytes).expect("moving the source");
            }
        },
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("changed underneath the conversion"),
        "{e}"
    );
    assert!(
        !out.join("manifest.toml").exists(),
        "it published a manifest describing bytes nobody has"
    );
    assert!(
        ledger.outstanding().is_empty(),
        "a refused run left a charge: {:?}",
        ledger.outstanding()
    );
}

/// No failure path may leave a reservation outstanding.
///
/// The review injected the first chunk-write failure and found all three
/// charges still held -- source tile, canonical tile and read-back scratch --
/// because the early `?` returns walked past the release at the end. Every
/// named boundary is checked here, through the public entry point.
#[test]
fn every_injected_failure_leaves_the_ledger_empty() {
    let scratch = Scratch::new("ledger");
    let m = module(16, 64, false);
    let src = shard_with(&scratch, &m, "I32", "I64", true);
    let selection_path = selection_for(&scratch, false);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
    let budgets = budgets();

    let mut checked = 0usize;
    for site in Site::ALL.iter().copied() {
        let out = scratch.join(&format!("artifact-{}", site.name()));
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        let faults = Faults::none().fail_at(site, 1);
        let outcome = moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
            &Options::default(),
            &faults,
            &|| false,
            &mut ledger,
            &mut |_| {},
        );
        assert!(
            ledger.outstanding().is_empty(),
            "failing at {} left {:?} charged",
            site.name(),
            ledger.outstanding()
        );
        // A site this scenario never reaches cannot fail it, and that is worth
        // counting rather than assuming.
        if outcome.is_err() {
            checked += 1;
        }
    }
    eprintln!(
        "task0025 ledger cleanup: {checked} of {} boundaries failed the run and released everything",
        Site::ALL.len()
    );
    assert!(
        checked >= 8,
        "only {checked} boundaries were reached; this measured almost nothing"
    );
}

/// A scratch too small to hold one zero-point word is refused **before** any
/// output exists, with the number it needs.
///
/// The review passed `--scratch-bytes 8` for an asymmetric INT4 `[8,8]`
/// per-channel fixture and got exit 101: the planner forced a word-sized block
/// and the tile that received it held four bytes.
#[test]
fn a_scratch_too_small_for_a_zero_point_word_is_refused_before_any_output() {
    let scratch = Scratch::new("tiny-scratch");
    let m = module(16, 64, false);
    let src = shard_with(&scratch, &m, "I32", "I64", true);
    let selection_path = selection_for(&scratch, false);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
    let mut budgets = budgets();
    budgets.scratch_bytes = 8;
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
    let e = moxie_repack::inspect(
        &selection,
        &mut sources,
        &budgets,
        &mut moxie_repack::ledger_for(&budgets).expect("a ledger"),
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("needs at least"),
        "the refusal does not say how much is needed: {e}"
    );

    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let out = scratch.join("artifact");
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
    let e = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &Options::default(),
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(e.to_string().contains("needs at least"), "{e}");
    assert!(
        !out.join("chunk0.bin").exists(),
        "a refused run created a payload"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A total that cannot hold the working set is not an admission.
#[test]
fn a_total_budget_below_the_working_set_is_refused() {
    let scratch = Scratch::new("small-total");
    let m = module(16, 64, false);
    let src = shard_with(&scratch, &m, "I32", "I64", true);
    let selection_path = selection_for(&scratch, false);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
    for total in [0u64, 2048, 64 << 20] {
        let mut budgets = budgets();
        budgets.total_bytes = total;
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("the sources");
        let e = moxie_repack::inspect(
            &selection,
            &mut sources,
            &budgets,
            &mut moxie_repack::ledger_for(&budgets).expect("a ledger"),
        )
        .unwrap_err();
        assert!(
            e.to_string().contains("cannot hold this run") || e.to_string().contains("positive"),
            "total {total}: {e}"
        );
    }
}
