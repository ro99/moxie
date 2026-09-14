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

/// An interruption during journal compaction leaves a resumable destination.
///
/// Compaction used to truncate the only journal and rewrite it in place, so a
/// failure between the two left staged shards with no binding to say whose they
/// were: `begin` then refused the destination as one it had not created, with
/// takeover enabled or not. Independent review reproduced both reachable
/// states. The journal is a replacement and a rename now, so at every instant
/// it is one of two complete documents.
#[test]
fn an_interrupted_compaction_still_resumes() {
    for site in [
        moxie_repack::write::Site::JournalCompactWrite,
        moxie_repack::write::Site::JournalCompactSync,
        moxie_repack::write::Site::JournalCompactPublish,
    ] {
        let scratch = Scratch::new(&format!("compaction-{}", site.name()));
        let src = scratch.join("src");
        std::fs::create_dir_all(&src).expect("a source directory");
        let values = 4096usize;
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
        SelectionBuilder::new("compaction")
            .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
            .write(&selection_path);
        let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
        let budgets = budgets(1 << 10);
        let out = scratch.join("out");

        // Stop part-way, so there is a journal to compact.
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        let units = std::cell::Cell::new(0usize);
        let stop = || units.get() > 4;
        moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
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

        // Fail inside compaction, on the resume.
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        let faults = Faults::none().fail_at(site, 1);
        let interrupted = moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
            &Options {
                take_over_interrupted_run: true,
            },
            &faults,
            &|| false,
            &mut ledger,
            &mut |_| {},
        );
        assert!(
            interrupted.is_err(),
            "failing at {} did not fail the run",
            site.name()
        );
        assert!(faults.all_fired(), "{} never fired", site.name());
        assert!(ledger.outstanding().is_empty());

        // **And the destination is inside its disk budget at that moment.**
        // Compaction holds the original and its replacement at once;
        // independent review measured 326,868 retained bytes against a 250,000
        // byte budget, because the plan reserved one journal and the run wrote
        // two.
        let retained: u64 = std::fs::read_dir(&out)
            .expect("the destination reads")
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();
        assert!(
            retained <= budgets.disk_bytes,
            "failing at {} left {retained} byte(s) against a {}-byte disk budget",
            site.name(),
            budgets.disk_bytes
        );

        // And the destination is still ours to finish.
        let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        let report = moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
            &Options {
                take_over_interrupted_run: true,
            },
            &Faults::none(),
            &|| false,
            &mut ledger,
            &mut |_| {},
        )
        .unwrap_or_else(|e| {
            panic!(
                "a destination interrupted at {} cannot be resumed: {e}",
                site.name()
            )
        });
        assert!(
            matches!(report.outcome, Outcome::Published { .. }),
            "resuming after {} gave {:?}",
            site.name(),
            report.outcome
        );
        assert!(
            !out.join(".moxie-repack-journal-new").exists(),
            "a leftover journal replacement survived the run that followed {}",
            site.name()
        );
    }
}

/// A **fresh** run clears a leftover replacement it finds.
///
/// This is where the cleanup is observable. A resuming run compacts, and
/// compaction truncates and rewrites the replacement anyway, so not removing it
/// first changes nothing there. A run that never compacts is the case that
/// exposes it -- and a leftover left lying in a destination counts against the
/// disk budget the next plan is checked against.
#[test]
fn a_fresh_run_clears_a_leftover_journal_replacement() {
    let scratch = Scratch::new("fresh-leftover");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    write_shard(
        &src.join("s.safetensors"),
        &[Entry::new(
            "model.norm.weight",
            "BF16",
            vec![512],
            bf16_bytes(1.0).repeat(512),
        )],
    );
    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("fresh-leftover")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(1 << 10);
    let out = scratch.join("out");
    std::fs::create_dir_all(&out).expect("the destination");

    // What an interruption between the write and the rename leaves behind, in a
    // destination with no journal: nothing here is resuming anything.
    let leftover = out.join(".moxie-repack-journal-new");
    std::fs::write(&leftover, vec![b'x'; 64 * 1024]).expect("the leftover writes");

    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
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
    .expect("a leftover replacement does not block a fresh run");
    assert!(
        matches!(report.outcome, Outcome::Published { .. }),
        "{:?}",
        report.outcome
    );
    assert!(
        !leftover.exists(),
        "a fresh run left someone's interrupted replacement in the destination"
    );
    assert!(ledger.outstanding().is_empty());
}

/// A leftover replacement from an interrupted compaction is cleared, and does
/// not count against the destination twice.
///
/// The file accounts for nothing -- the journal beside it is authoritative
/// whichever side of the rename the interruption fell on -- so a run that finds
/// one removes it before planning around the disk it holds.
#[test]
fn a_leftover_journal_replacement_is_cleared_before_planning() {
    let scratch = Scratch::new("leftover-replacement");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let values = 4096usize;
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
    SelectionBuilder::new("leftover")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(1 << 10);
    let out = scratch.join("out");

    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let units = std::cell::Cell::new(0usize);
    let stop = || units.get() > 4;
    moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
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

    // What an interruption between the write and the rename leaves behind.
    let leftover = out.join(".moxie-repack-journal-new");
    std::fs::write(&leftover, vec![b'x'; 128 * 1024]).expect("the leftover writes");

    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let report = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &Options {
            take_over_interrupted_run: true,
        },
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect("a leftover replacement does not block a resume");
    assert!(
        matches!(report.outcome, Outcome::Published { .. }),
        "{:?}",
        report.outcome
    );
    assert!(!leftover.exists(), "the leftover replacement survived");
    assert!(ledger.outstanding().is_empty());
}

/// A **refused** resume does not clear the leftover either.
///
/// Cleanup used to run before the journal's binding was checked, so a resume
/// with a different plan was correctly refused **after** deleting a file that
/// was not its to delete. Ownership first, mutation second -- the same rule the
/// header pass and the tear repair already follow.
#[test]
fn a_refused_resume_leaves_the_leftover_replacement_alone() {
    let scratch = Scratch::new("refused-leftover");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    write_shard(
        &src.join("s.safetensors"),
        &[Entry::new(
            "model.norm.weight",
            "BF16",
            vec![8192],
            bf16_bytes(1.0).repeat(8192),
        )],
    );
    let selection_path = scratch.join("selection.toml");
    SelectionBuilder::new("refused-leftover")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = budgets(1 << 10);
    let out = scratch.join("out");

    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let units = std::cell::Cell::new(0usize);
    let stop = || units.get() > 2;
    moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
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
        !out.join("manifest.toml").exists(),
        "the first run published, so there is nothing to resume"
    );

    let leftover = out.join(".moxie-repack-journal-new");
    std::fs::write(&leftover, b"not this run's").expect("the leftover writes");
    let before: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&out)
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

    // A different output plan, with the source still resolvable.
    let other = scratch.join("other.toml");
    let text = std::fs::read_to_string(&selection_path).expect("it reads");
    let patched = text.replacen(
        "role = \"model.norm.weight\"",
        "role = \"model.norm.renamed\"",
        1,
    );
    assert_ne!(patched, text, "the role line was not found");
    std::fs::write(&other, patched).expect("written");
    let other_selection = moxie_repack::read_selection(&other).expect("it parses");

    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
    let e = moxie_repack::repack(
        &other_selection,
        &mut sources,
        &out,
        &budgets,
        &Options {
            take_over_interrupted_run: true,
        },
        &Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect_err("a different plan is a refusal");
    assert!(
        e.to_string().contains("bound to a different plan"),
        "the refusal is not the binding check: {e}"
    );

    let after: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&out)
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
    assert!(ledger.outstanding().is_empty());
}

/// A role whose escaped form is larger than the role is **refused at planning**.
///
/// Journal sizing used the decoded role length while the journal stores the
/// escaped one. Independent review used a role of 1,000 escaped NULs to write a
/// 17,517,998-byte journal inside an admitted run, which the next invocation
/// could not read back.
///
/// The arithmetic itself is checked in `moxie-format`, against `unit_line`
/// directly. This is the end-to-end half, and it requires a **specific**
/// outcome: a refusal naming the journal, before any payload work. Accepting
/// "refused or published" was how the targeted mutation survived -- with the
/// sizing wrong the run publishes, and a test that tolerates both sees nothing.
#[test]
fn a_role_whose_escaped_form_overruns_the_journal_is_refused_at_planning() {
    let scratch = Scratch::new("escaped-role");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    // A backslash is one byte in a name and two in the journal. With enough
    // units, that difference is the difference between a plan that fits under
    // the journal's cap and one that does not.
    let role: String = std::iter::repeat_n('\\', 900).collect();
    assert_eq!(
        moxie_format::journal::escaped_len(&role),
        1_800,
        "this role does not reproduce the finding"
    );
    // Sized so the **decoded** estimate fits under the journal's cap and the
    // escaped one does not -- 13.5 MB against 22.1 MB, either side of 16.8 MB.
    // Without that separation both spellings refuse and the test cannot tell
    // which arithmetic produced the refusal.
    let values = 2_426_112usize;
    write_shard(
        &src.join("s.safetensors"),
        &[Entry::new(
            "plain.weight",
            "BF16",
            vec![values as u64],
            bf16_bytes(1.0).repeat(values),
        )],
    );
    let selection_path = scratch.join("selection.toml");
    let placeholder = "ROLE-PLACEHOLDER";
    let text = SelectionBuilder::new("escaped-role")
        .bf16(placeholder, "plain.weight", "s.safetensors")
        .text()
        .replace(placeholder, &"\\\\".repeat(900));
    std::fs::write(&selection_path, &text).expect("the selection writes");
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");

    let budgets = moxie_repack::Budgets {
        total_bytes: 512 << 20,
        header_bytes: 4 << 20,
        scratch_bytes: 1 << 10,
        chunk_file_bytes: 256 << 20,
        disk_bytes: 256 << 20,
    };
    let out = scratch.join("out");
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
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
    .expect_err("a journal nobody can read back is not a plan");
    assert!(
        e.to_string().contains("journal byte"),
        "the refusal does not name the journal cap: {e}"
    );
    assert!(
        !out.join("manifest.toml").exists(),
        "it published despite a journal it could not read back"
    );
    assert!(ledger.outstanding().is_empty());
}
