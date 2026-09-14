//! Task 0026 round 2: the source, the journal and cancellation.
//!
//! The companions to `malformed.rs`. Those cases are about an artifact that
//! disagrees with its schema; these are about a run whose inputs move, whose
//! private files grow past what a resume can read, or whose user asks it to
//! stop.

mod common;

use std::collections::BTreeMap;

use common::{Entry, Module, Scratch, SelectionBuilder, bf16_bytes, write_shard};
use moxie_repack::write::{Faults, Options, Outcome};

fn budgets(scratch_bytes: usize) -> moxie_repack::Budgets {
    moxie_repack::Budgets {
        total_bytes: 256 << 20,
        header_bytes: 16 << 20,
        scratch_bytes,
        chunk_file_bytes: 64 << 20,
        disk_bytes: 64 << 20,
    }
}

fn module(rows: usize, columns: usize, symmetric: bool) -> Module {
    let groups = columns / 32;
    Module {
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
    }
}

/// A source replaced between two reads is a source this run cannot describe.
///
/// Review inspected a selection, atomically replaced the shard, then repacked
/// through the same `Sources`: the artifact held the old file's values and the
/// manifest recorded the new file's digest. Both hash passes agreed, because
/// both hashed the new file while the conversion read the old one.
#[test]
fn a_source_replaced_under_a_run_is_refused() {
    let scratch = Scratch::new("replacement");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let entries = vec![Entry::new(
        "model.norm.weight",
        "BF16",
        vec![8],
        bf16_bytes(1.0).repeat(8),
    )];
    write_shard(&src.join("s.safetensors"), &entries);

    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("replacement")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(64 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");

    // Read it once, exactly as an inspection would.
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    moxie_repack::inspect(&selection, &mut sources, &budgets, &mut ledger).expect("it inspects");

    // Now replace the file underneath, atomically, so the name resolves to a
    // different inode with different values.
    let replacement = vec![Entry::new(
        "model.norm.weight",
        "BF16",
        vec![8],
        bf16_bytes(2.0).repeat(8),
    )];
    let staged = scratch.join("replacement.safetensors");
    write_shard(&staged, &replacement);
    std::fs::rename(&staged, src.join("s.safetensors")).expect("the replacement lands");

    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let e = moxie_repack::repack(
        &selection,
        &mut sources,
        &scratch.join("out"),
        &budgets,
        &Options::default(),
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect_err("a replaced source is not a source this run can describe");
    assert!(
        e.to_string().contains("replaced"),
        "the refusal does not name the replacement: {e}"
    );
    assert!(ledger.outstanding().is_empty());
}

/// Zero points in a shard that is not the codes' shard are still zero points.
///
/// Review put the codes in one shard and the scales plus nonzero zero points in
/// another -- both named by the selection -- and a symmetric selection
/// published. That silently drops `Z` from `W = (Q - Z) * S`.
#[test]
fn a_symmetric_selection_over_a_split_module_with_zero_points_is_refused() {
    let scratch = Scratch::new("split-symmetric");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let name = "model.layers.0.mlp.down_proj";
    let m = module(4, 64, false);

    write_shard(
        &src.join("codes.safetensors"),
        &[
            Entry::new(
                &format!("{name}.weight_packed"),
                "I32",
                m.packed_shape(),
                m.packed(),
            ),
            Entry::new(
                &format!("{name}.weight_shape"),
                "I64",
                vec![2],
                m.weight_shape(),
            ),
        ],
    );
    // The companion lives here, in a shard the selection names for the scales.
    write_shard(
        &src.join("scales.safetensors"),
        &[
            Entry::new(
                &format!("{name}.weight_scale"),
                "BF16",
                m.scale_shape(),
                m.scale_payload("BF16"),
            ),
            Entry::new(
                &format!("{name}.weight_zero_point"),
                "I32",
                m.zero_point_shape(),
                m.zero_point(),
            ),
        ],
    );

    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("split-symmetric")
        .pack_quantized(
            &format!("{name}.weight"),
            name,
            "int4",
            "32",
            "symmetric",
            &BTreeMap::from([
                ("weight_packed", "codes.safetensors"),
                ("weight_scale", "scales.safetensors"),
                ("weight_shape", "codes.safetensors"),
            ]),
        )
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(64 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let e = moxie_repack::repack(
        &selection,
        &mut sources,
        &scratch.join("out"),
        &budgets,
        &Options::default(),
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect_err("a symmetric claim over a source with zero points is a disagreement");
    assert!(
        e.to_string().contains("zero point"),
        "the refusal does not name the companion: {e}"
    );
}

/// Cancellation while hashing sources is cancellation.
///
/// Review's early-cancellation probe received `InvalidArtifact` naming the
/// file, which is a corrupt-source report for a stop the caller asked for.
#[test]
fn cancellation_while_hashing_is_reported_as_cancellation() {
    let scratch = Scratch::new("cancel-hash");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    write_shard(
        &src.join("s.safetensors"),
        &[Entry::new(
            "model.norm.weight",
            "BF16",
            vec![8],
            bf16_bytes(1.0).repeat(8),
        )],
    );
    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("cancel-hash")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(64 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let out = scratch.join("out");
    let report = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &Options::default(),
        &Faults::none(),
        &|| true,
        &mut ledger,
        &mut |_| {},
    )
    .expect("cancellation is not an error");
    assert!(
        matches!(report.outcome, Outcome::Cancelled { .. }),
        "cancelling during the source hash gave {:?}",
        report.outcome
    );
    assert!(
        !out.join("manifest.toml").exists(),
        "it published while cancelled"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A plan whose journal could not be read back is refused before it starts.
///
/// Review produced a valid 4,354,889-byte journal from an 8 MiB source at a
/// 1 KiB scratch, and the resume could not reopen it. Raising the reader's
/// limit alone would leave the writer able to exceed the journal's own cap;
/// this refuses the plan instead, and says what to change.
#[test]
fn a_plan_whose_journal_would_exceed_the_cap_is_refused() {
    let scratch = Scratch::new("journal-cap");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    // 16 MiB of BF16 at a 1 KiB scratch is ~32,000 units, each a journal record.
    let values = 8 << 20;
    write_shard(
        &src.join("s.safetensors"),
        &[Entry::new(
            "model.norm.weight",
            "BF16",
            vec![values as u64],
            bf16_bytes(1.0).repeat(values),
        )],
    );
    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("journal-cap")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(1 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let e = moxie_repack::repack(
        &selection,
        &mut sources,
        &scratch.join("out"),
        &budgets,
        &Options::default(),
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect_err("a journal nobody can read back is not a plan");
    let text = e.to_string();
    assert!(
        text.contains("journal byte") && text.contains("scratch-bytes"),
        "the refusal does not say what to change: {text}"
    );
    assert!(ledger.outstanding().is_empty());
}
