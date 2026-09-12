//! Isolated executable: what a safetensors header actually costs to open.
//!
//! Separate binary because the counter is a **global** allocator; two tests in
//! one executable race on it and neither result means anything.
//!
//! Two rounds of independent review produced this. The first found the header
//! unbudgeted. The second found the budget measuring the wrong thing: a
//! 54,899-byte *serialized* bound admitted a header whose peak heap was 353,105
//! bytes and whose retained heap was 163,275, because parsing allocates an
//! entry list, a key string and a shape vector per tensor, a temporary span
//! list, and the retained maps.
//!
//! So the budget is denominated in peak heap, and this measures the real peak
//! against the admitted estimate instead of asserting the estimate is right.
//!
//! **Exactly one test lives here.** The counter is global, so a second test in
//! this binary would run concurrently with this one and both measurements would
//! be meaningless. The admission rules, which need no allocator, are in
//! `tests/artifact.rs`.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicIsize, Ordering::SeqCst};

use moxie_storage::{HeaderBudget, Shard};

struct Counter;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's
// unmodified pointer and layout; the counters are relaxed atomics and do not
// affect the allocation itself.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid layout.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let live = LIVE.fetch_add(layout.size() as isize, SeqCst) + layout.size() as isize;
            PEAK.fetch_max(live, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, SeqCst);
        // SAFETY: pointer and layout describe the original live allocation.
        unsafe { System.dealloc(p, layout) };
    }
}

#[global_allocator]
static A: Counter = Counter;

fn tempdir() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "moxie-header-budget-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// `tensors` entries with `name_len`-character names: short names pack the most
/// entries into a byte of header, which is the worst case for the ratio.
fn shard(dir: &Path, tag: &str, tensors: usize, name_len: usize) -> (PathBuf, u64) {
    let mut json = String::from("{");
    for i in 0..tensors {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "\"{:0width$}\":{{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[{i},{}]}}",
            i,
            i + 1,
            width = name_len
        ));
    }
    json.push('}');
    let path = dir.join(format!("{tag}.safetensors"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(json.as_bytes()).unwrap();
    f.write_all(&vec![0u8; tensors]).unwrap();
    (path, json.len() as u64 + 8)
}

/// The admitted estimate must cover the peak heap opening actually costs, over
/// header shapes from one tiny entry to five thousand minimal ones.
#[test]
fn a_header_costs_no_more_peak_heap_than_its_admitted_estimate() {
    let dir = tempdir();
    let mut rows = Vec::new();
    for (tensors, name_len) in [(1usize, 1usize), (100, 1), (1000, 1), (1000, 64), (5000, 1)] {
        let (path, serialized) = shard(&dir, &format!("s{tensors}_{name_len}"), tensors, name_len);
        let estimate = HeaderBudget::estimated_peak(serialized).unwrap();

        // Warm: the first open of any shard pulls in one-off machinery.
        Shard::open(&path).unwrap();
        let before = LIVE.load(SeqCst);
        PEAK.store(before, SeqCst);
        let s = Shard::open(&path).unwrap();
        let peak = (PEAK.load(SeqCst) - before) as u64;
        let retained = (LIVE.load(SeqCst) - before) as u64;
        assert_eq!(s.header().tensors().len(), tensors);
        drop(s);

        assert!(
            peak <= estimate,
            "{tensors} tensors x {name_len}: peak {peak} B exceeds the admitted \
             estimate {estimate} B for a {serialized}-byte header. The estimate is \
             what a caller's budget was checked against, so this is the number \
             that must move -- see HeaderBudget::PEAK_FACTOR"
        );
        rows.push((tensors, name_len, serialized, peak, retained, estimate));
    }
    for (t, n, serialized, peak, retained, estimate) in &rows {
        eprintln!(
            "task0018 header cost: tensors={t} name_len={n} serialized={serialized} \
             peak={peak} retained={retained} admitted_estimate={estimate} \
             peak_ratio={:.2}",
            *peak as f64 / *serialized as f64
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
