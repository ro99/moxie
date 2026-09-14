//! Task 0025 acceptance 7: what the run actually held, against what it
//! admitted.
//!
//! Isolated executable, because the counter below is a **global** allocator:
//! two tests measuring it in one process race and neither number means
//! anything. That is task 0024's lesson, in the file that would repeat it.
//!
//! What is measured here is peak **live** heap across a whole repack, not the
//! number of allocations: a bounded converter is one whose working set does
//! not grow with the tensor, and a tensor many times the scratch is the case
//! that shows it.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use common::{Entry, Module, Scratch, SelectionBuilder, write_shard};
use moxie_repack::Budgets;
use moxie_storage_write::{Faults, Options, Outcome};

struct Counter;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's
// unmodified pointer and layout; the counters are relaxed atomics and do not
// affect the allocation itself.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the caller's unmodified layout.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size(), SeqCst) + layout.size();
            PEAK.fetch_max(live, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), SeqCst);
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwards the caller's own pointer, layout and size.
        let out = unsafe { System.realloc(p, layout, new_size) };
        if !out.is_null() {
            LIVE.fetch_sub(layout.size(), SeqCst);
            let live = LIVE.fetch_add(new_size, SeqCst) + new_size;
            PEAK.fetch_max(live, SeqCst);
        }
        out
    }
}

#[global_allocator]
static A: Counter = Counter;

/// One BF16 tensor many times the payload scratch, plus a quantized module.
fn fixture(scratch: &Scratch, elements: usize) -> (Module, std::path::PathBuf) {
    let src = scratch.join("src");
    std::fs::create_dir_all(&src).expect("a source directory");
    let (rows, columns) = (64usize, 256usize);
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
    let big: Vec<u8> = (0..elements)
        .flat_map(|v| (((v % 0x7F00) as u16) & 0x7F7F).to_le_bytes())
        .collect();
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
            Entry::new("model.big.weight", "BF16", vec![elements as u64], big),
        ],
    );
    let files = BTreeMap::from([
        ("weight_packed", "shard-a.safetensors"),
        ("weight_scale", "shard-a.safetensors"),
        ("weight_shape", "shard-a.safetensors"),
        ("weight_zero_point", "shard-a.safetensors"),
    ]);
    let selection = scratch.join("selection.toml");
    SelectionBuilder::new("synthetic")
        .pack_quantized(
            "model.layers.0.mlp.down_proj.weight",
            name,
            "int4",
            "32",
            "packed-along-output",
            &files,
        )
        .bf16("model.big.weight", "model.big.weight", "shard-a.safetensors")
        .write(&selection);
    (module, selection)
}

fn budgets(scratch_bytes: usize) -> Budgets {
    Budgets {
        total_bytes: 128 << 20,
        header_bytes: 64 << 20,
        scratch_bytes,
        chunk_file_bytes: 4 << 20,
        disk_bytes: 64 << 20,
    }
}

#[test]
fn a_repack_holds_its_budget_and_gives_every_admitted_byte_back() {
    let scratch = Scratch::new("budget");
    // Two mebibytes of BF16 against a 64 KiB payload scratch: 32 units for
    // this tensor alone, and a working set that must not grow with it.
    let elements = 1024 * 1024;
    let (module, selection_path) = fixture(&scratch, elements);
    let out = scratch.join("artifact");
    let budgets = budgets(64 * 1024);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");
    let mut sources =
        moxie_repack::open_sources(&scratch.join("src"), &budgets).expect("the sources");
    let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");

    let before = LIVE.load(SeqCst);
    PEAK.store(before, SeqCst);
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
    .expect("it publishes");
    let peak = PEAK.load(SeqCst) - before;
    let after = LIVE.load(SeqCst);

    assert!(
        matches!(report.outcome, Outcome::Published { .. }),
        "{:?}",
        report.outcome
    );
    // The big tensor alone is 32 units at this scratch size.
    assert!(
        report.units_written >= 32,
        "{} unit(s) for a tensor 32 times the scratch",
        report.units_written
    );

    // Peak live heap against what a run of this shape may hold: three tiles of
    // the payload scratch (source, canonical and the writer's read-back), the
    // column scratch, the shard's parsed header, the manifest text and the
    // program's own bookkeeping. The admitted total is the contract; this is
    // the measurement against it.
    let admitted = budgets.total_bytes;
    eprintln!(
        "task0025 budget: peak live heap {peak} B, admitted {admitted} B, \
         payload scratch {} B, units {}, published {} B",
        budgets.scratch_bytes,
        report.units_written,
        report.bytes_written
    );
    assert!(
        (peak as u64) < admitted,
        "peak live heap {peak} B is above the admitted {admitted} B"
    );
    // And much smaller than the artifact it produced: the point of the budget.
    assert!(
        peak < 8 * 1024 * 1024,
        "peak live heap {peak} B is not bounded by the tiles"
    );
    assert!(
        report.bytes_written > 2 * 1024 * 1024,
        "{} byte(s) published",
        report.bytes_written
    );

    // Every admitted byte returned, and the process is back where it started.
    assert!(
        ledger.outstanding().is_empty(),
        "a finished run left a charge: {:?}",
        ledger.outstanding()
    );
    assert!(
        after <= before + 512 * 1024,
        "after the run {after} B is live against {before} B before it"
    );

    // Disk: what the plan said, and nothing else.
    let published: u64 = std::fs::read_dir(&out)
        .expect("the artifact")
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("chunk"))
        .map(|e| e.metadata().expect("a file").len())
        .sum();
    let expected = module.expected_canonical("BF16").len() as u64;
    let big_bytes = (elements * 2) as u64;
    assert!(
        published >= expected + big_bytes,
        "{published} B published against {expected} + {big_bytes} B of payload"
    );
    assert!(
        published <= budgets.disk_bytes,
        "{published} B is above the admitted disk budget"
    );
    // No private file survives a publish.
    for private in [".moxie-repack-journal", ".moxie-repack-lock", ".moxie-repack-manifest"] {
        assert!(!out.join(private).exists(), "{private} survived the publish");
    }
}

/// Repeated cancellation and resume must not grow what is retained, in memory
/// or on disk.
#[test]
fn repeated_cancellation_and_resume_grow_neither_heap_nor_disk() {
    let scratch = Scratch::new("budget-resume");
    let (_, selection_path) = fixture(&scratch, 64 * 1024);
    let out = scratch.join("artifact");
    let budgets = budgets(16 * 1024);
    let selection = moxie_repack::read_selection(&selection_path).expect("a selection");

    let mut peaks = Vec::new();
    let mut disks = Vec::new();
    for attempt in 0..4 {
        let mut sources =
            moxie_repack::open_sources(&scratch.join("src"), &budgets).expect("the sources");
        let mut ledger = moxie_repack::ledger_for(&budgets).expect("a ledger");
        let stop_after = 2usize;
        let seen = std::cell::Cell::new(0usize);
        let cancel = |_: &str| {};
        let cancelled = || seen.get() >= stop_after;
        let before = LIVE.load(SeqCst);
        PEAK.store(before, SeqCst);
        let report = moxie_repack::repack(
            &selection,
            &mut sources,
            &out,
            &budgets,
            &Options {
                take_over_interrupted_run: true,
            },
            &Faults::none(),
            &cancelled,
            &mut ledger,
            &mut |line: &str| {
                if line.starts_with("unit ") {
                    seen.set(seen.get() + 1);
                }
                cancel(line);
            },
        );
        let report = report.expect("each attempt either cancels or publishes");
        peaks.push(PEAK.load(SeqCst) - before);
        disks.push(scratch.bytes_used());
        assert!(
            ledger.outstanding().is_empty(),
            "attempt {attempt} left a charge: {:?}",
            ledger.outstanding()
        );
        if matches!(report.outcome, Outcome::Published { .. }) {
            break;
        }
    }
    eprintln!("task0025 resume peaks: {peaks:?} B; disk after each attempt: {disks:?} B");
    // No attempt holds materially more than the first: a resume that grew the
    // working set would be one that kept what it recovered.
    let first = peaks[0];
    for (i, p) in peaks.iter().enumerate() {
        assert!(
            *p <= first * 2,
            "attempt {i} peaked at {p} B against {first} B for the first"
        );
    }
    // Disk grows, and every byte of the growth is accounted: each attempt
    // converts two more units of at most one tile each, plus the journal lines
    // that record them. What must not happen is an attempt leaving a second
    // copy of what it recovered.
    let tile = budgets.tile_bytes() as u64;
    let allowance = 2 * tile + 4096;
    for window in disks.windows(2) {
        let growth = window[1] - window[0];
        assert!(
            growth <= allowance,
            "an attempt added {growth} B against two units of {tile} B plus its journal"
        );
    }
    let last = *disks.last().expect("at least one attempt");
    assert!(
        last <= budgets.disk_bytes,
        "{last} B is above the admitted disk budget"
    );
}
