//! Task 0026 round 2, finding 5: does the declared total actually cover what
//! an inspection allocates?
//!
//! Independent review ran an allocator-instrumented `inspect` over 5,000
//! selected tensors under a 9,438,720-byte total and measured roughly
//! 19,969,672 bytes of additional peak live heap. The floor it was checked
//! against was `header + MAX_SELECTION_BYTES + MAX_MANIFEST_BYTES`: three
//! constants about **serialized text**, which say nothing about the parsed
//! selection, the resolved tensors, the plan, the shard headers or the report.
//!
//! This measures the same thing on purpose, so the bound is a measurement
//! rather than a hope.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{Entry, Scratch, SelectionBuilder, bf16_bytes, write_shard};

/// Live bytes, and the high-water mark.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to the system allocator with the same layout and
// pointer it was given; the counters only observe sizes and change no
// allocation contract.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the same layout, forwarded unchanged.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Saturating: a measurement that zeroes the counter mid-process would
        // otherwise underflow here and overflow on the next allocation.
        let _ = LIVE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.saturating_sub(layout.size()))
        });
        // SAFETY: the pointer and layout pair a matching `alloc` handed out.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the pointer, its original layout and the new size, forwarded
        // unchanged.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                let grew = new_size - layout.size();
                let now = LIVE.fetch_add(grew, Ordering::Relaxed) + grew;
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                let shrank = layout.size() - new_size;
                let _ = LIVE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                    Some(n.saturating_sub(shrank))
                });
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// A selection of `n` BF16 tensors whose roles are `role_len` bytes each.
fn build(scratch: &Scratch, n: usize, role_len: usize) -> (std::path::PathBuf, String) {
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("source root");
    // Spread across shards, so the per-shard header stays inside the header
    // budget: this test is about the **parsed selection**, not about how large
    // a single source header may be, which `--header-bytes` already bounds and
    // `HeaderBudget` already enforces.
    const PER_SHARD: usize = 250;
    let mut sel = SelectionBuilder::new("admission").complete();
    let mut entries: Vec<Entry> = Vec::new();
    let mut shard = 0usize;
    for i in 0..n {
        // A long role is a valid role, and its bytes are retained many times
        // over: in the selection, the resolved tensor, the component name, the
        // manifest row, the plan, the shard header and every journal line.
        let role = format!("{}{i:06}", "r".repeat(role_len.saturating_sub(6)));
        let file = format!("shard-{shard:05}.safetensors");
        entries.push(Entry::new(
            &role,
            "BF16",
            vec![4],
            bf16_bytes(1.0).repeat(4),
        ));
        sel = sel.bf16(&role, &role, &file);
        if entries.len() == PER_SHARD {
            write_shard(&src.join(&file), &entries);
            entries.clear();
            shard += 1;
        }
    }
    if !entries.is_empty() {
        write_shard(&src.join(format!("shard-{shard:05}.safetensors")), &entries);
    }
    (src, sel.text())
}

/// The admitted total is an upper bound on what an inspection actually holds.
#[test]
fn an_inspection_stays_inside_the_total_it_was_admitted_for() {
    let scratch = Scratch::new("admission");
    let tensors = 5_000usize;
    let role_len = 40usize;
    let (src, selection_text) = build(&scratch, tensors, role_len);
    let selection_path = scratch.join("selection.toml");
    std::fs::write(&selection_path, &selection_text).expect("selection");

    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = moxie_repack::Budgets {
        total_bytes: 0, // filled in below, from the bound this selection implies
        header_bytes: 1 << 20,
        scratch_bytes: 64 << 10,
        chunk_file_bytes: 64 << 20,
        disk_bytes: 64 << 20,
    };
    let bound = budgets.metadata_bound(selection_text.len() as u64, selection.tensors.len() as u64);
    let budgets = moxie_repack::Budgets {
        total_bytes: bound + 3 * budgets.scratch_bytes as u64 / 2,
        ..budgets
    };

    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("ledger");

    // Relative to what is already live: zeroing a process-wide counter while
    // allocations are outstanding measures the wrong thing and corrupts it.
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let report = moxie_repack::inspect(&selection, &mut sources, &budgets, &mut ledger)
        .expect("it inspects");
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    assert_eq!(report.tensors.len(), tensors);

    let old_floor = budgets.metadata_floor_bytes();
    eprintln!(
        "task0026 admission: {tensors} tensor(s) at {role_len}-byte roles, selection {} byte(s); \
         peak live heap {peak}; the constant floor this replaced was {old_floor}; admitted \
         metadata bound {bound}",
        selection_text.len()
    );
    // And the floor this replaced is **not** a bound: three constants about
    // serialized text, which is the shape of the defect rather than its size.
    assert!(
        peak as u64 > old_floor,
        "this selection no longer reproduces the finding: peak {peak} is inside the old \
         {old_floor}-byte floor, so the test would pass with the bug present"
    );
    assert!(
        peak as u64 <= bound,
        "an inspection peaked at {peak} byte(s) of live heap against a {bound}-byte metadata \
         admission: the bound is not a bound"
    );
    assert!(ledger.outstanding().is_empty(), "inspection left a charge");
}

/// A **resume** stays inside its admission too.
///
/// The bound was derived from selection size and tensor count, and recovery's
/// allocations are driven by neither: reading an interrupted run's journal
/// holds its bytes and the records parsed out of them together, and how many
/// records there are follows from the payload and the work-unit size.
/// Independent review resumed an 8 MiB source under a 9,438,720-byte total and
/// measured 12,942,600 bytes of peak live heap against a 5,257,792-byte
/// reservation.
#[test]
fn a_resume_stays_inside_the_total_it_was_admitted_for() {
    let scratch = Scratch::new("admission-resume");
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("source root");
    // One tensor, many small units: the journal, not the selection, is what
    // grows here.
    // 8 MiB of payload at a 1 KiB scratch: the shape of the review's
    // reproduction, whose journal reached 4.35 MB.
    let values = 4 << 20;
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
    SelectionBuilder::new("admission-resume")
        .bf16("model.norm.weight", "model.norm.weight", "s.safetensors")
        .write(&selection_path);
    let selection = moxie_repack::read_selection(&selection_path).expect("it parses");
    let budgets = moxie_repack::Budgets {
        total_bytes: 256 << 20,
        header_bytes: 1 << 20,
        scratch_bytes: 1 << 10,
        chunk_file_bytes: 64 << 20,
        disk_bytes: 64 << 20,
    };
    let out = scratch.join("out");

    // A first run that stops part-way, leaving a journal to recover.
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("ledger");
    let units = std::cell::Cell::new(0usize);
    let stop = || units.get() > 12_000;
    let first = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &moxie_repack::write::Options::default(),
        &moxie_repack::write::Faults::none(),
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
        matches!(
            first.outcome,
            moxie_repack::write::Outcome::Cancelled { .. }
        ),
        "the first run did not stop: {:?}",
        first.outcome
    );

    // Now resume, and measure what recovery holds.
    let mut sources = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("ledger");
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let second = moxie_repack::repack(
        &selection,
        &mut sources,
        &out,
        &budgets,
        &moxie_repack::write::Options {
            take_over_interrupted_run: true,
        },
        &moxie_repack::write::Faults::none(),
        &|| false,
        &mut ledger,
        &mut |_| {},
    )
    .expect("the resume finishes");
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let _ = second;

    // What the run was admitted for, by the same arithmetic the run uses, from
    // the public report rather than a copy of the formula.
    let mut probe = moxie_repack::open_sources(&src, &budgets).expect("sources");
    let mut probe_ledger = moxie_repack::ledger_for(&budgets).expect("ledger");
    let report = moxie_repack::inspect(&selection, &mut probe, &budgets, &mut probe_ledger)
        .expect("inspect");
    let units: u64 = report.tensors.iter().map(|t| t.units as u64).sum();
    let widest_role = report
        .tensors
        .iter()
        .map(|t| t.role.len() as u64)
        .max()
        .unwrap_or(0);
    let metadata = budgets.metadata_bound(selection.source_bytes(), selection.tensors.len() as u64);
    let recovery = moxie_repack::Budgets::recovery_bound(
        report.staging.journal_bound_bytes,
        units,
        widest_role,
    );
    let tiles = 3 * budgets.scratch_bytes as u64 / 2;
    let admitted = metadata + recovery + tiles;
    eprintln!(
        "task0026 resume admission: {units} journal record(s); peak live heap {peak}; admitted \
         {admitted} = metadata {metadata} + recovery {recovery} + tiles {tiles}"
    );
    assert!(
        peak as u64 <= admitted,
        "a resume peaked at {peak} byte(s) against a {admitted}-byte admission: metadata \
         {metadata} + recovery {recovery} + tiles {tiles}"
    );
    // And the metadata reservation alone does **not** cover it, which is the
    // finding: a bound derived from the selection says nothing about how many
    // journal records a resume has to hold.
    assert!(
        peak as u64 > metadata,
        "this resume no longer reproduces the finding: {peak} is inside the {metadata}-byte \
         metadata reservation, so the test would pass with the recovery admission removed"
    );
    assert!(ledger.outstanding().is_empty(), "the resume left a charge");
}
