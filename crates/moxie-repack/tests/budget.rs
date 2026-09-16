//! Task 0025 acceptance 7: what the run actually held, against what it
//! admitted. Task 0030 acceptance 1: measured so the number means something.
//!
//! Isolated executable, because the counter below is a **global** allocator:
//! two tests measuring it in one process race and neither number means
//! anything. That is task 0024's lesson, in the file that would repeat it.
//!
//! One executable was not enough, and neither was a lock around the run.
//!
//! * The first version let two tests reset each other's peak, so a run could
//!   read a peak *below* the live bytes it started from and subtract past zero.
//! * The second took a lock around the measured call only. Everything else --
//!   building a two-mebibyte fixture, opening the sources, reading `LIVE`
//!   afterwards, destroying the scratch directory -- still ran outside it, in
//!   parallel with the other test's window. Those allocations are counted by a
//!   *process-wide* counter, so they landed in a peak that names a repack.
//!   Two baseline runs of this lane on an identical clean tree reported "fails"
//!   and "disagrees with itself", and the mutation battery had to serialise the
//!   executable to stay runnable.
//!
//! A measurement is now a [`Session`]: the lock is taken before the fixture
//! exists and released after it is destroyed, and every counter read happens
//! inside it. Nothing in this executable allocates measurably outside a
//! session, so a window contains one test's work and no other's. The negative
//! control at the bottom is what that claim is worth: it allocates thirty-two
//! mebibytes against a gate whose bound is eight, and under a session that
//! burst cannot land in another test's window.
//!
//! The reset the second version could suffer produced a peak **below** the live
//! bytes the window opened with, and the subtraction saturated to zero -- which
//! passes every bound here. That path is now a panic rather than a zero, so a
//! future edit that breaks the isolation fails loudly instead of quietly
//! passing.
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
use moxie_repack::write::{Faults, Options, Outcome};

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

/// The bound on peak live heap that this file's real measurement applies, and
/// the bound its negative controls are measured against.
///
/// **One constant, named once.** Review found the number written twice -- once
/// in the repack gate and once in the control that is supposed to prove the
/// gate can fail -- and two copies of a threshold are two thresholds. Raise
/// this and the controls move with it; that is the whole point of them.
const TILE_BOUND: usize = 8 * 1024 * 1024;

/// Held for the whole of one measurement -- fixture, windows, counter reads and
/// destruction -- so no other test allocates into this test's numbers.
static MEASURING: std::sync::Mutex<()> = std::sync::Mutex::new(());

thread_local! {
    /// Whether **this thread** holds a session.
    ///
    /// The session is a convention until something checks it. This is the
    /// check: [`fixture`] refuses to build megabytes of shard unless the thread
    /// building them holds one, because every byte of that would otherwise be
    /// counted into whatever window is open. Move `Session::open()` below the
    /// fixture in any test here -- which is exactly where it used to be -- and
    /// that test fails, deterministically, naming the ordering rather than
    /// reporting a strange peak.
    ///
    /// **Per thread, not a global count.** A global "is any session open" is
    /// true whenever the *other* test holds one, so it passes precisely when
    /// two tests are interleaved, which is the case it exists to catch. That
    /// version was written first and caught nothing; the tests it was supposed
    /// to protect failed later, elsewhere, on a strange number -- which is the
    /// symptom, not the ordering.
    static SESSION_HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Exclusive use of the process-wide counter.
///
/// Open it **before** the fixture and keep it alive **past** the fixture's
/// destruction: declare it first in the test body and every later local is
/// dropped before it. A test that allocates outside a session is the bug this
/// type exists to prevent, not an optimisation.
struct Session {
    // A poisoned lock means another test panicked, which is already a failure
    // being reported; it does not make this measurement wrong.
    _guard: std::sync::MutexGuard<'static, ()>,
}

/// One measured window: what `body` returned, and what the heap did while it
/// ran.
struct Window<T> {
    out: T,
    /// Peak live heap above the live bytes the window started from.
    peak: usize,
    /// Live bytes when the window opened.
    before: usize,
    /// Live bytes when it closed, with the fixture still alive.
    after: usize,
}

impl Drop for Session {
    fn drop(&mut self) {
        SESSION_HELD.with(|held| held.set(false));
    }
}

impl Session {
    fn open() -> Self {
        let session = Session {
            _guard: MEASURING.lock().unwrap_or_else(|e| e.into_inner()),
        };
        // After the lock, so the flag is only ever set by the thread that holds
        // it.
        SESSION_HELD.with(|held| {
            assert!(!held.get(), "this thread already holds a session");
            held.set(true);
        });
        session
    }

    /// Run `body` as the only measured work in this process.
    ///
    /// The peak window opens *after* the fixture is built, because what is
    /// being measured is a repack and not the bytes a test wrote to give it
    /// something to repack. Isolation is the session's job; scope is this
    /// method's.
    fn window<T>(&self, body: impl FnOnce() -> T) -> Window<T> {
        let before = LIVE.load(SeqCst);
        PEAK.store(before, SeqCst);
        let out = body();
        let peak = PEAK.load(SeqCst);
        let after = LIVE.load(SeqCst);
        // `PEAK` is stored once here and only ever raised by `fetch_max`
        // afterwards, so inside a session it cannot come back below `before`.
        // It could before sessions existed -- another test's `store` landed
        // mid-window -- and the subtraction then saturated to **zero**, which
        // every bound in this file passes. A gate that reports success when its
        // instrument has been reset under it is worse than no gate, so the
        // arithmetic that used to hide that now names it.
        let peak = peak.checked_sub(before).unwrap_or_else(|| {
            panic!(
                "a window's peak is {peak} B, below the {before} B it opened with: the counter \
                 was reset by something outside this session, and nothing measured here means \
                 anything"
            )
        });
        Window {
            out,
            peak,
            before,
            after,
        }
    }
}

/// One BF16 tensor many times the payload scratch, plus a quantized module.
fn fixture(scratch: &Scratch, elements: usize) -> (Module, std::path::PathBuf) {
    assert!(
        SESSION_HELD.with(|held| held.get()),
        "this fixture writes megabytes and is being built outside a session, so \
         every byte of it lands in whichever window is open. `let session = \
         Session::open();` belongs ABOVE the fixture, not below it -- below it \
         is the bug this file was repaired for"
    );
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
            .map(|o| {
                (0..groups)
                    .map(|g| 0.5 + 0.25 * ((o + g) % 3) as f32)
                    .collect()
            })
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
        .bf16(
            "model.big.weight",
            "model.big.weight",
            "shard-a.safetensors",
        )
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
    // First, so that everything below -- the fixture, the run, the counter
    // reads and the destruction of the scratch directory -- is inside it.
    let session = Session::open();
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

    let measured = session.window(|| {
        moxie_repack::repack(
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
        .expect("it publishes")
    });
    let Window {
        out: report,
        peak,
        before,
        after,
    } = measured;

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
        budgets.scratch_bytes, report.units_written, report.bytes_written
    );
    assert!(
        (peak as u64) < admitted,
        "peak live heap {peak} B is above the admitted {admitted} B"
    );
    // And much smaller than the artifact it produced: the point of the budget.
    assert!(
        peak < TILE_BOUND,
        "peak live heap {peak} B is not bounded by the tiles ({TILE_BOUND} B)"
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
        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
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
    for private in [
        ".moxie-repack-journal",
        ".moxie-repack-lock",
        ".moxie-repack-manifest",
    ] {
        assert!(
            !out.join(private).exists(),
            "{private} survived the publish"
        );
    }
}

/// Repeated cancellation and resume must not grow what is retained, in memory
/// or on disk.
#[test]
fn repeated_cancellation_and_resume_grow_neither_heap_nor_disk() {
    let session = Session::open();
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
        let measured = session.window(|| {
            moxie_repack::repack(
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
            )
        });
        let report = measured
            .out
            .expect("each attempt either cancels or publishes");
        peaks.push(measured.peak);
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

/// The second control: what a session is actually preventing.
///
/// The witness in [`fixture`] refuses to build a fixture outside a session.
/// This is why that refusal is worth having. It allocates **outside** any
/// session, on another thread, while a window is open -- the exact shape of
/// every fixture in this file before task 0030 -- and asserts that those bytes
/// land in the window and carry it past the gate's own bound.
///
/// It is deterministic: the two channels order the allocation strictly inside
/// the window rather than hoping a race lands there. Without the ordering this
/// was a coin flip, and a coin flip recorded as a measurement is what the
/// battery refused to build verdicts on in the first place.
///
/// The allocation is deliberately the thing the witness forbids. It is safe
/// here only because this test holds the session itself, so the one window it
/// can pollute is its own.
#[test]
fn an_allocation_outside_a_session_lands_in_whichever_window_is_open() {
    let session = Session::open();
    let (allocated_tx, allocated_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();

    let outsider = std::thread::spawn(move || {
        // One byte past the gate, so what the window reads is unambiguous.
        let unsessioned: Vec<u8> = vec![0x5A; TILE_BOUND + 1];
        allocated_tx
            .send(unsessioned.len())
            .expect("the window listens");
        release_rx.recv().expect("the window closes");
        std::hint::black_box(&unsessioned);
    });

    let measured = session.window(|| {
        allocated_rx
            .recv()
            .expect("the allocation outside the session happens")
    });
    release_tx.send(()).expect("the outsider is waiting");
    outsider.join().expect("the outsider finishes");

    assert_eq!(measured.out, TILE_BOUND + 1);
    assert!(
        measured.peak > TILE_BOUND,
        "{} B of an allocation made outside a session did not reach the window \
         it overlapped: if that were true, this file would not have needed a \
         session at all",
        measured.peak
    );
}

/// The negative control: what the two gates above are worth.
///
/// A bound that cannot fail is not a measurement, and a bound measured through
/// a process-wide counter fails for the wrong reason as easily as the right
/// one. This allocates thirty-two mebibytes -- four times the bound the repack
/// gate applies -- holds it long enough to overlap any concurrent test, and
/// asserts two things about the instrument rather than about the repacker:
///
/// 1. **It can see.** The window reports the whole burst, so a peak that comes
///    back small above is a small peak and not a blind counter.
/// 2. **It cannot leak.** The burst happens inside a session, so it is ordered
///    against every other measurement in this executable rather than landing
///    in one. Move this allocation outside the session -- which is where the
///    fixtures used to be built -- and the repack gate above reads it and
///    fails on a buffer that has nothing to do with a repack.
#[test]
fn an_allocation_far_above_the_bound_is_seen_and_reaches_no_other_measurement() {
    const BURST: usize = 4 * TILE_BOUND;

    let session = Session::open();
    let measured = session.window(|| {
        // Allocated and freed repeatedly rather than held: a buffer that is
        // already live when a window opens is part of what that window starts
        // from, and only a burst *inside* a window moves its peak. That is the
        // shape this controls for.
        //
        // The deadline is checked at the *end* of the body, so a thread that
        // loses two seconds to a loaded machine still bursts once. Checking it
        // first made this control fail under load with `largest == 0`, having
        // measured nothing and reported it as a broken meter.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(2000);
        let mut largest = 0usize;
        loop {
            let unbounded: Vec<u8> = vec![0xA5; BURST];
            largest = largest.max(std::hint::black_box(unbounded.len()));
            if std::time::Instant::now() >= deadline {
                break largest;
            }
        }
    });
    // What the harness itself does while a session is held: a few hundred
    // bytes of its own bookkeeping, freed on another thread. A window reports
    // the peak *above the live bytes it opened with*, so bytes freed elsewhere
    // during the window make the reading conservative by that much. Under
    // 56-way load this control read 33,554,044 B of a 33,554,432 B burst --
    // 388 B short. That is the instrument's resolution, and it is named here
    // rather than rounded away.
    //
    // How much headroom that leaves, divided rather than asserted: the tile
    // bound is 8,388,608 / 388 = **4.33 orders of magnitude** above it, and the
    // admitted total 134,217,728 / 388 = **5.54**. An earlier version of this
    // comment said "six orders" of both, which is a number nobody divided.
    const HARNESS_NOISE: usize = 64 * 1024;

    assert_eq!(measured.out, BURST);
    assert!(
        measured.peak + HARNESS_NOISE >= BURST,
        "the meter saw {} B of a {BURST} B allocation: it cannot see what the \
         gates above ask it to",
        measured.peak
    );
    // Stated the way the gates state it, so the control fails if the bound is
    // ever raised above what it is controlling.
    assert!(
        measured.peak >= TILE_BOUND,
        "a {BURST} B allocation did not exceed the {TILE_BOUND} B bound"
    );
    // And it gave every byte back, so the burst cannot be what a later test
    // starts from.
    assert!(
        measured.after <= measured.before + 64 * 1024,
        "the control left {} B live against {} B before it",
        measured.after,
        measured.before
    );
}
