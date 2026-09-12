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
//! A third round defeated the peak-heap budget too, with two shapes the five
//! sampled ones did not cover: many tiny `__metadata__` entries, and a single
//! tensor of 65,537 dimensions. **Sampling shapes cannot establish a bound.**
//!
//! So the bound is now derived from the cost of each *construct* a header can
//! contain, and this file measures each construct at its own worst shape --
//! which is what the derivation rests on -- plus both counterexamples. The
//! unbounded one, rank, is answered by a structural limit rather than a larger
//! multiplier, because no multiplier survives it.
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

/// One header built from a named construct, at the worst shape for that
/// construct: the most of it a byte of header can buy.
fn write_shard(dir: &Path, tag: &str, json: String, payload: usize) -> (PathBuf, u64) {
    let path = dir.join(format!("{tag}.safetensors"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(json.as_bytes()).unwrap();
    f.write_all(&vec![0u8; payload]).unwrap();
    (path, json.len() as u64 + 8)
}

/// `tensors` minimal entries with `name_len`-character names.
fn tensor_json(tensors: usize, name_len: usize) -> String {
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
    json
}

/// `n` distinct `__metadata__` entries with three-character keys and empty
/// values -- the cheapest construct per serialized byte after the rank limit.
fn metadata_json(n: usize) -> String {
    let alpha: Vec<char> = ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
    let mut json = String::from("{\"__metadata__\":{");
    let mut c = 0;
    'outer: for a in &alpha {
        for b in &alpha {
            for d in &alpha {
                if c == n {
                    break 'outer;
                }
                if c > 0 {
                    json.push(',');
                }
                json.push_str(&format!("\"{a}{b}{d}\":\"\""));
                c += 1;
            }
        }
    }
    assert_eq!(c, n, "the fixture must produce distinct keys");
    json.push_str("},\"t\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}");
    json
}

/// The admitted estimate must cover the peak heap that opening actually costs,
/// at **each construct's own worst shape** -- which is what the derivation in
/// `HeaderBudget::PEAK_FACTOR` rests on -- and at both counterexamples an
/// earlier bound failed.
#[test]
fn a_header_costs_no_more_peak_heap_than_its_admitted_estimate() {
    let dir = tempdir();
    let mut cases: Vec<(String, String, usize)> = Vec::new();
    // Whole-header shapes.
    for (t, n) in [(1usize, 1usize), (100, 1), (1000, 1), (1000, 64), (5000, 1)] {
        cases.push((format!("tensors-{t}x{n}"), tensor_json(t, n), t));
    }
    // Each construct at its worst shape.
    //
    // Tensor entries: the most entries a byte can buy.
    cases.push(("worst-tensor-entries".into(), tensor_json(8000, 1), 8000));
    // Metadata entries: cheaper per byte than a tensor entry, and the first
    // counterexample. 20,000 of them, well past the review's 5,000.
    cases.push(("worst-metadata".into(), metadata_json(20_000), 1));
    // Tensor-name bytes.
    cases.push(("worst-name-bytes".into(), tensor_json(500, 900), 500));
    // Shape dimensions, at the largest rank the limit admits.
    let dims = format!(
        "[1{}]",
        ",1".repeat(moxie_format::safetensors::MAX_RANK - 1)
    );
    let mut ranked = String::from("{");
    for i in 0..4000 {
        if i > 0 {
            ranked.push(',');
        }
        ranked.push_str(&format!(
            "\"{i}\":{{\"dtype\":\"U8\",\"shape\":{dims},\"data_offsets\":[{i},{}]}}",
            i + 1
        ));
    }
    ranked.push('}');
    cases.push(("worst-rank".into(), ranked, 4000));

    let mut rows = Vec::new();
    for (tag, json, payload) in &cases {
        let (path, serialized) = write_shard(&dir, tag, json.clone(), *payload);
        let estimate = HeaderBudget::estimated_peak(serialized).unwrap();

        // Warm: the first open of any shard pulls in one-off machinery.
        Shard::open(&path).unwrap();
        let before = LIVE.load(SeqCst);
        PEAK.store(before, SeqCst);
        let s = Shard::open(&path).unwrap();
        let peak = (PEAK.load(SeqCst) - before) as u64;
        let retained = (LIVE.load(SeqCst) - before) as u64;
        drop(s);

        assert!(
            peak <= estimate,
            "{tag}: peak {peak} B exceeds the admitted estimate {estimate} B for a \
             {serialized}-byte header. The estimate is what a caller's budget was \
             checked against. Enumerate the construct that beat it and bound that \
             -- raising HeaderBudget::PEAK_FACTOR is what failed three times."
        );
        rows.push((tag.clone(), serialized, peak, retained, estimate));
        std::fs::remove_file(&path).ok();
    }
    // The rank counterexample, which no bound derived from the other
    // constructs could absorb: refused outright rather than budgeted for.
    //
    // In the same test, because the counter is a global allocator and a second
    // test in this binary would run concurrently with this one.

    let rank = moxie_format::safetensors::MAX_RANK;
    let json = |n: usize| {
        let dims = format!("1{}", ",1".repeat(n - 1));
        format!("{{\"t\":{{\"dtype\":\"U8\",\"shape\":[{dims}],\"data_offsets\":[0,1]}}}}")
    };
    // At the limit it parses.
    let (ok, _) = write_shard(&dir, "rank-ok", json(rank), 1);
    assert_eq!(
        Shard::open(&ok)
            .unwrap()
            .header()
            .get("t")
            .unwrap()
            .shape
            .len(),
        rank
    );
    // One past it, and the review's 65,537, are both refused -- and the refusal
    // costs a fraction of what accepting would have.
    for (tag, n) in [("rank-over", rank + 1), ("rank-65537", 65_537)] {
        let (path, serialized) = write_shard(&dir, tag, json(n), 1);
        let before = LIVE.load(SeqCst);
        PEAK.store(before, SeqCst);
        assert!(Shard::open(&path).is_err(), "{tag} must be refused");
        let peak = (PEAK.load(SeqCst) - before) as u64;
        let estimate = HeaderBudget::estimated_peak(serialized).unwrap();
        assert!(
            peak <= estimate,
            "{tag}: refusing cost {peak} B against {estimate} B"
        );
        eprintln!(
            "task0018 rank refusal: {tag} dims={n} serialized={serialized} refusal_peak={peak}"
        );
    }

    for (tag, serialized, peak, retained, estimate) in &rows {
        eprintln!(
            "task0018 header cost: {tag} serialized={serialized} peak={peak} \
             retained={retained} admitted_estimate={estimate} peak_ratio={:.2}",
            *peak as f64 / *serialized as f64
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}
