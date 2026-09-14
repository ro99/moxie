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
        e.to_string().contains("a different file"),
        "the refusal does not name the replacement: {e}"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A source **appended to** after its first hash is not the file that was hashed.
///
/// Round 2 bound the inode; round 3 found that both hash passes then stopped at
/// the length captured when the shard was opened, so a file that grew was
/// hashed twice to the same stale end and published under a digest describing a
/// prefix.
#[test]
fn a_source_appended_to_under_a_run_is_refused() {
    let scratch = Scratch::new("appended");
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
    SelectionBuilder::new("appended")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(64 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    moxie_repack::inspect(&selection, &mut sources, &budgets, &mut ledger).expect("it inspects");

    // Grow the file in place: same inode, more bytes.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(src.join("s.safetensors"))
            .expect("the source opens");
        f.write_all(b"appended").expect("the append lands");
    }

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
    .expect_err("a file that grew is not the file that was hashed");
    // Caught at the hash itself: the live length from the handle disagrees with
    // the one the header was parsed against.
    assert!(
        e.to_string().contains("byte(s) when this run opened it")
            || e.to_string().contains("byte(s) became"),
        "the refusal does not name the length change: {e}"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A source whose **header** was rewritten in place is not the file that was
/// resolved.
///
/// The inode is the same and so is the length; every tensor offset this run
/// resolved came out of header bytes that are gone. Independent review produced
/// an artifact holding BF16 zeros for a tensor whose source held ones.
#[test]
fn a_source_whose_header_was_rewritten_in_place_is_refused() {
    let scratch = Scratch::new("rewritten-header");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let path = src.join("s.safetensors");
    // Two tensors of identical shape and dtype: swapping their names in the
    // header changes which bytes each one means, and changes nothing else.
    write_shard(
        &path,
        &[
            Entry::new("a.weight", "BF16", vec![8], bf16_bytes(1.0).repeat(8)),
            Entry::new("b.weight", "BF16", vec![8], bf16_bytes(0.0).repeat(8)),
        ],
    );
    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("rewritten-header")
        .bf16("a.weight", "a.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(64 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    moxie_repack::inspect(&selection, &mut sources, &budgets, &mut ledger).expect("it inspects");

    // Swap the two names inside the header, in place. Same inode, same length.
    let mut bytes = std::fs::read(&path).expect("the shard reads");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
    let header = String::from_utf8(bytes[8..8 + header_len].to_vec()).expect("utf-8");
    let swapped = header
        .replacen("\"a.weight\"", "\"TEMP\"", 1)
        .replacen("\"b.weight\"", "\"a.weight\"", 1)
        .replacen("\"TEMP\"", "\"b.weight\"", 1);
    assert_eq!(swapped.len(), header.len(), "the swap changes no length");
    assert_ne!(swapped, header, "the swap matched nothing");
    bytes[8..8 + header_len].copy_from_slice(swapped.as_bytes());
    std::fs::write(&path, &bytes).expect("the rewrite lands");

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
    .expect_err("a rewritten header means the resolved offsets describe nothing");
    assert!(
        e.to_string().contains("rewritten header"),
        "the refusal does not name the rewrite: {e}"
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

/// A budget that admits the metadata and refuses a tile leaves nothing charged.
///
/// `Buffers::admit` reserved the metadata, then allocated two tiles with bare
/// `?`. A failure on the second admission returned without a `Buffers` for the
/// caller to release, so independent review watched 10,857,840 bytes stay
/// charged for the life of the process.
///
/// Getting here at all takes care: `Budgets::validate` refuses a total below
/// its constant floor before admission is reached, so the first version of this
/// test measured that refusal instead and the targeted mutation survived. The
/// budget below passes the floor and fails at the **ledger**, which is where
/// the leak was.
#[test]
fn a_partly_admitted_plan_leaves_nothing_charged() {
    let scratch = Scratch::new("partial-admission");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    // Many tensors: `metadata_bound` is proportional to the selection, so it
    // can exceed the total while the constant floor stays inside it.
    const N: usize = 4_000;
    let mut entries = Vec::with_capacity(N);
    let mut sel = SelectionBuilder::new("partial-admission");
    for i in 0..N {
        let role = format!("model.layers.{i:05}.norm.weight");
        entries.push(Entry::new(
            &role,
            "BF16",
            vec![2],
            bf16_bytes(1.0).repeat(2),
        ));
        sel = sel.bf16(&role, &role, "s.safetensors");
    }
    write_shard(&src.join("s.safetensors"), &entries);
    let selection_path = scratch.join("selection.toml");
    sel.write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");

    let base = moxie_repack::Budgets {
        total_bytes: 0,
        header_bytes: 1 << 20,
        scratch_bytes: 1 << 10,
        chunk_file_bytes: 64 << 20,
        disk_bytes: 64 << 20,
    };
    let metadata = base.metadata_bound(selection.source_bytes(), selection.tensors.len() as u64);
    let floor = base.metadata_floor_bytes() + 3 * base.scratch_bytes as u64 / 2;
    assert!(
        metadata > floor,
        "this selection does not reproduce the finding: the metadata bound {metadata} is inside \
         the constant floor {floor}, so `validate` refuses before admission"
    );
    // Both tiles, one at a time. The tile is half the scratch, so a total that
    // leaves less than one tile fails on the **first** and a total that leaves
    // exactly one fails on the **second** -- and the two failures unwind
    // different amounts. Covering only the second was how the mutation on the
    // first survived.
    let tile = base.scratch_bytes as u64 / 2;
    for (slack, which) in [(tile - 1, "the first tile"), (tile, "the second tile")] {
        let budgets = moxie_repack::Budgets {
            total_bytes: metadata + slack,
            ..base
        };
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        let e = moxie_repack::repack(
            &selection,
            &mut sources,
            &scratch.join(&format!("out-{slack}")),
            &budgets,
            &Options::default(),
            &Faults::none(),
            &|| false,
            &mut ledger,
            &mut |_| {},
        )
        .unwrap_err();
        assert!(
            matches!(e, moxie_types::Error::CapacityExceeded { .. }),
            "refusing at {which} is not an admission failure: {e}"
        );
        assert!(
            ledger.outstanding().is_empty(),
            "refusing at {which} left {:?} charged",
            ledger.outstanding()
        );
    }
}

/// Cancelling while an interrupted run is being recovered is cancellation.
///
/// Rehashing an interrupted run's staged units re-reads everything it had
/// written, so it is exactly where a user asks to stop. It used to come back as
/// `InvalidArtifact: cancelled while rehashing the staged units of an
/// interrupted run` -- a corrupt-artifact report for a stop the caller asked
/// for.
#[test]
fn cancellation_while_recovering_is_reported_as_cancellation() {
    let scratch = Scratch::new("cancel-recovery");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    // Big enough that the first run writes several units for the resume to
    // rehash, and small enough to stay quick.
    let values = 65_536usize;
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
    SelectionBuilder::new("cancel-recovery")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(1 << 10);
    let out = scratch.join("out");

    // A first run that stops part-way, leaving units in the journal.
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    // Stop after real **units**, so the destination is left with a journal to
    // recover. Counting cancellation questions stopped inside the first run's
    // own source hash, which leaves nothing behind -- and then the second run
    // started fresh and never recovered anything, so this test passed with the
    // protection removed.
    let units = std::cell::Cell::new(0usize);
    let stop_after_a_few = || units.get() > 8;
    let first = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &Options::default(),
        &Faults::none(),
        &stop_after_a_few,
        &mut ledger,
        &mut |line: &str| {
            if line.starts_with("unit ") {
                units.set(units.get() + 1);
            }
        },
    )
    .expect("cancellation is not an error");
    assert!(
        matches!(first.outcome, Outcome::Cancelled { .. }),
        "the first run did not stop: {:?}",
        first.outcome
    );
    assert!(
        out.join(".moxie-repack-journal").exists(),
        "the first run left no journal, so there is nothing to recover"
    );

    // Resume, and cancel once the source hashes are done and recovery starts.
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    // Cancel **during recovery**, which means: not before the source hash has
    // finished. Counting questions cancelled inside the hash instead, where the
    // outcome is the same either way -- so the test passed without ever
    // reaching the code it names.
    let hashed = std::cell::Cell::new(false);
    let cancel_during_recovery = || hashed.get();
    let second = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &Options {
            take_over_interrupted_run: true,
        },
        &Faults::none(),
        &cancel_during_recovery,
        &mut ledger,
        &mut |line: &str| {
            if line.starts_with("hashed source") {
                hashed.set(true);
            }
        },
    )
    .expect("cancelling a recovery is not an artifact failure");
    assert!(
        second.resumed,
        "the second run did not resume, so it never recovered anything: {:?}",
        second.resume_detail
    );
    assert!(
        matches!(second.outcome, Outcome::Cancelled { .. }),
        "cancelling during recovery gave {:?}",
        second.outcome
    );
    assert!(
        !out.join("manifest.toml").exists(),
        "it published while cancelled"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A total that cannot hold recovery refuses the resume.
///
/// Asserting that a measured peak is inside a bound the **test** computes says
/// nothing about whether the program admitted it: the targeted mutation removed
/// the admission and the test still passed. What distinguishes them is the
/// ledger. With a total sized for the metadata and the tiles but not for the
/// journal records a resume must hold, admitting recovery is the difference
/// between a refusal and a run that quietly exceeds its declared budget.
#[test]
fn a_total_that_cannot_hold_recovery_refuses_the_resume() {
    let scratch = Scratch::new("recovery-admission");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    // Enough units that what recovery holds dominates what the selection does.
    let values = 5 << 20;
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
    SelectionBuilder::new("recovery-admission")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let out = scratch.join("out");

    // Roomy enough to write, so there is a journal to recover. A small header
    // budget keeps `Budgets::validate`'s constant floor out of the way: the
    // constraint under test is the ledger, not that floor.
    let writing = moxie_repack::Budgets {
        total_bytes: 256 << 20,
        header_bytes: 1 << 20,
        scratch_bytes: 1 << 10,
        chunk_file_bytes: 64 << 20,
        disk_bytes: 64 << 20,
    };
    let mut sources = moxie_repack::open_sources(&src, &writing).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&writing).expect("a ledger");
    let units = std::cell::Cell::new(0usize);
    let stop = || units.get() > 4_000;
    let first = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &writing,
        &Options::default(),
        &Faults::none(),
        &stop,
        &mut ledger,
        &mut |line: &str| {
            if line.starts_with("unit ") {
                units.set(units.get() + 1);
            }
        },
    )
    .expect("cancellation is not an error");
    assert!(
        matches!(first.outcome, Outcome::Cancelled { .. }),
        "the first run did not stop: {:?}",
        first.outcome
    );

    // Now a total that covers the metadata and the tiles, and not the 2,000-odd
    // journal records a resume has to hold.
    let metadata = writing.metadata_bound(selection.source_bytes(), selection.tensors.len() as u64);
    let tight = moxie_repack::Budgets {
        total_bytes: metadata + 3 * writing.scratch_bytes as u64 / 2 + (4 << 20),
        ..writing
    };
    let mut sources = moxie_repack::open_sources(&src, &tight).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&tight).expect("a ledger");
    let e = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &tight,
        &Options {
            take_over_interrupted_run: true,
        },
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect_err("a total that cannot hold recovery is not a total this run can use");
    assert!(
        matches!(e, moxie_types::Error::CapacityExceeded { .. }),
        "the refusal is not an admission failure: {e}"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A large-but-valid architecture-metadata string publishes.
///
/// The manifest allowance reserved a flat 16 KiB for everything that is not a
/// tensor row, so a selection carrying a 32 KiB architecture-metadata string
/// failed publication against a derived allowance smaller than the manifest it
/// was about to write -- with a 64 MiB disk budget, which no increase could fix,
/// because the allowance was not derived from the disk budget.
#[test]
fn a_large_architecture_metadata_string_still_publishes() {
    let scratch = Scratch::new("big-arch-metadata");
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
    let text = SelectionBuilder::new("big-arch-metadata")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .text()
        .replace(
            "note = \"opaque to every shared crate\"",
            &format!("note = \"{}\"", "m".repeat(32 * 1024)),
        );
    assert!(text.len() > 32 * 1024, "the fixture is not large");
    std::fs::write(&selection_path, &text).expect("the selection writes");

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
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect("a valid selection with large metadata is publishable");
    assert!(
        matches!(report.outcome, Outcome::Published { .. }),
        "it did not publish: {:?}",
        report.outcome
    );
    assert!(out.join("manifest.toml").exists());
    assert!(ledger.outstanding().is_empty());
}

/// Repeated recovery does not grow the journal.
///
/// `overhead_used` started at zero on every resume, and recovery kept the
/// records it had just discarded while recomputation appended replacements.
/// Independent review cancelled, corrupted, resumed and cancelled repeatedly
/// against a 240,000-byte disk budget and watched the destination retain
/// 166,026 bytes, then 326,698, then 487,370.
///
/// The payload legitimately grows while a run makes progress, so this corrupts
/// the staged bytes between rounds: recovery then discards every unit and
/// rewrites the same ones, and **only** the history is left to grow.
#[test]
fn repeated_recovery_does_not_grow_the_journal() {
    let scratch = Scratch::new("recovery-growth");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let values = 65_536usize;
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
    SelectionBuilder::new("recovery-growth")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(1 << 10);
    let out = scratch.join("out");
    let journal = out.join(".moxie-repack-journal");

    let mut sizes = Vec::new();
    for round in 0..3 {
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        // Count **units**, not cancellation questions: the source hash asks
        // first, and a counter over every question stops before anything is
        // written -- a cancelled run that retains nothing proves nothing here.
        let units = std::cell::Cell::new(0usize);
        let stop = || units.get() > 8;
        let report = moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
            &Options {
                take_over_interrupted_run: round > 0,
            },
            &Faults::none(),
            &stop,
            &mut ledger,
            &mut |line: &str| {
                if line.starts_with("unit ") {
                    units.set(units.get() + 1);
                }
            },
        )
        .expect("cancellation is not an error");
        assert!(
            matches!(report.outcome, Outcome::Cancelled { .. }),
            "round {round} gave {:?}",
            report.outcome
        );
        assert!(ledger.outstanding().is_empty());
        sizes.push(std::fs::metadata(&journal).expect("a journal").len());

        // Corrupt the first staged payload byte, so the next recovery discards
        // every unit and rewrites the same ones: no forward progress, and any
        // growth is history rather than work.
        let shard = out.join(moxie_repack::write::shard_name(1, 1));
        let mut bytes = std::fs::read(&shard).expect("a staged shard");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes")) as usize;
        bytes[8 + header_len] ^= 0xFF;
        std::fs::write(&shard, &bytes).expect("corrupting it");
    }

    eprintln!("task0026 recovery growth: journal bytes per round {sizes:?}");
    assert!(
        sizes[2] <= sizes[0],
        "repeated recovery grew the journal: {sizes:?}"
    );
}

/// Hashing a source that grew refuses, rather than returning a prefix digest.
///
/// This targets the guard directly, at `Sources`, because the publication path
/// has a second net: `verify_unchanged` re-opens every source before anything is
/// exposed and would catch the same file there. That later check does not make
/// this one redundant — between them sits every digest this run *reports* and
/// writes into its run binding, and a digest of the first N bytes labelled as
/// the digest of the file is wrong at the moment it is produced.
#[test]
fn hashing_a_source_that_grew_refuses_instead_of_hashing_a_prefix() {
    let scratch = Scratch::new("grew-during-hash");
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
    SelectionBuilder::new("grew-during-hash")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(64 << 10);
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    // Open and parse the shard, which is what fixes the length the digest would
    // otherwise run to.
    moxie_repack::inspect(&selection, &mut sources, &budgets, &mut ledger).expect("it inspects");

    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(src.join("s.safetensors"))
            .expect("the source opens");
        f.write_all(b"grown").expect("the append lands");
    }

    let mut scratch_buf = vec![0u8; 4096];
    let e = sources
        .file_digest("s.safetensors", &mut scratch_buf)
        .expect_err("a digest of a prefix is not a digest of the file");
    assert!(
        e.to_string().contains("byte(s) when this run opened it"),
        "the refusal does not name the length it would have hashed to: {e}"
    );
}
