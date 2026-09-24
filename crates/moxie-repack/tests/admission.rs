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
//!
//! **One measurement per test binary.** The counter below is process-wide and
//! each measurement resets its high-water mark, so two of them in one binary
//! race: cargo runs tests concurrently by default, and either can erase the
//! other's peak while its own allocations distort the window. The storage
//! crate's header-budget test is one-per-executable for the same reason, and
//! independent review found this file was not.

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
        "admission: {tensors} tensor(s) at {role_len}-byte roles, selection {} byte(s); \
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
