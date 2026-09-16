//! `cargo xtask mutation-check` -- can the gates actually fail?
//!
//! A suite that passes says nothing on its own: every checksum it compares it
//! also computed, and a resume that recomputes everything produces the right
//! answer whatever the resume logic did. This applies one substitution at a
//! time to tracked source, runs the lanes, and records which of them noticed.
//! Only `caught` counts.
//!
//! Each substitution is committed here rather than described in prose, for the
//! reason experiment 0005 records: the mutation *names* are not the
//! measurement, these exact edits are.
//!
//! # Independence controls
//!
//! A mutation may declare `Expect::Survivor`. Removing *that* protection must
//! leave the battery green, because the protection under test still holds. A
//! control that is caught means the two are entangled, which is a different
//! fact from a mutant that survived, and is reported separately.
//!
//! # It edits tracked source
//!
//! Run it on a clean tree; `git status` afterwards is part of the evidence. The
//! original is written to `target/mutation-check/` **before** the substitution
//! and removed after it is put back, so a run killed at any moment -- including
//! `SIGKILL`, which no handler can catch -- leaves a marker that the next run
//! finds and restores. A mutation left in the working tree is worse than a
//! failed battery: the next `cargo test` measures it silently. That has
//! happened here, which is why the recovery is a file rather than a handler.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// How many times each verdict is taken, in both directions.
///
/// A substitution read once from a flaky lane is a coin flip recorded as a
/// measurement.
const REPEATS: usize = 3;

/// Where the pre-substitution original is parked while a mutation is applied.
const RESTORE_DIR: &str = "target/mutation-check";

struct Lane {
    name: &'static str,
    argv: &'static [&'static str],
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Expect {
    /// A mutant: some lane must fail.
    Caught,
    /// An independence control or an equivalent mutant: no lane may fail.
    Survivor,
}

struct Mutation {
    name: &'static str,
    file: &'static str,
    from: &'static str,
    to: &'static str,
    expect: Expect,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Verdict {
    Caught,
    Survivor,
    Unstable,
    InvalidControl,
    ControlHeld,
    ControlBroken,
}

impl Verdict {
    fn name(self) -> &'static str {
        match self {
            Verdict::Caught => "caught",
            Verdict::Survivor => "survivor",
            Verdict::Unstable => "unstable",
            Verdict::InvalidControl => "invalid-control",
            Verdict::ControlHeld => "control-held",
            Verdict::ControlBroken => "control-broken",
        }
    }
}

/// Puts the file back, however this scope is left.
///
/// Covers a normal return, an early `?` and a panic. It does **not** cover a
/// signal, which is what the restore marker beside it is for.
/// How far a guard has got through putting the tree back.
///
/// **Three steps, not two booleans.** Each one can fail on its own and each
/// failure means something different, so a pair of flags kept losing one of the
/// combinations: with `armed`/`restored`, a marker removal that succeeded and a
/// parked-original removal that failed left the guard claiming both files were
/// still there, retrying the removed marker and never retrying the copy.
/// Independent review found that, after finding the previous ordering bug in
/// the same place. A state a step cannot half-leave is the fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// The source file still carries the mutant.
    Mutated,
    /// The source is back. The marker and the parked original are both on disk,
    /// which is the recoverable pair.
    Restored,
    /// The marker is gone. Only the parked original is left, and it is now
    /// inert: nothing looks for it without a marker.
    MarkerCleared,
    /// Nothing left.
    Clean,
}

struct Restore {
    path: PathBuf,
    original: String,
    marker: PathBuf,
    stage: Stage,
}

impl Restore {
    fn arm(root: &Path, file: &str, original: &str) -> Result<Self, String> {
        let dir = root.join(RESTORE_DIR);
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let marker = dir.join("in-progress");
        // The original **before** the substitution: if this process dies
        // between here and the restore, the next run has everything it needs.
        std::fs::write(dir.join("original"), original)
            .map_err(|e| format!("cannot park the original: {e}"))?;
        std::fs::write(&marker, file).map_err(|e| format!("cannot write the marker: {e}"))?;
        Ok(Self {
            path: root.join(file),
            original: original.to_string(),
            marker,
            stage: Stage::Mutated,
        })
    }

    fn parked(&self) -> PathBuf {
        self.marker.with_file_name("original")
    }

    /// Advance one step, or report why it could not.
    ///
    /// The order is the whole point: **source, then marker, then the parked
    /// copy.** The marker is what tells the next run there is something to
    /// recover and the copy is what it recovers *from*, so a marker outliving
    /// its copy is the one combination nothing can recover from -- which is why
    /// the copy goes last and only after the marker is confirmed gone.
    fn step(&mut self) -> Result<(), String> {
        match self.stage {
            Stage::Mutated => {
                std::fs::write(&self.path, &self.original).map_err(|e| {
                    format!(
                        "cannot restore {}: {e}. The mutant is still installed; the original is \
                         parked at {} and the marker names the file, so the next run recovers it",
                        self.path.display(),
                        self.parked().display()
                    )
                })?;
                self.stage = Stage::Restored;
            }
            Stage::Restored => {
                std::fs::remove_file(&self.marker).map_err(|e| {
                    format!(
                        "{} was restored but its marker could not be cleared: {e}. The marker and \
                         its parked original are both left in place, so the next run sees a \
                         recoverable pair rather than a marker with nothing behind it",
                        self.path.display()
                    )
                })?;
                self.stage = Stage::MarkerCleared;
            }
            Stage::MarkerCleared => {
                std::fs::remove_file(self.parked()).map_err(|e| {
                    format!(
                        "{} was restored and its marker cleared, but the parked original at {} \
                         could not be removed: {e}. Nothing looks for it without a marker, so \
                         this is untidy rather than unsafe",
                        self.path.display(),
                        self.parked().display()
                    )
                })?;
                self.stage = Stage::Clean;
            }
            Stage::Clean => {}
        }
        Ok(())
    }

    /// Put the file back and clear the evidence, one confirmed step at a time.
    ///
    /// The ordinary path. `Drop` is the emergency one, for a panic or a `?` on
    /// the way out, and it cannot report anything -- which is exactly why the
    /// cleanup must not live only there: an early version ignored the result of
    /// the rewrite and then deleted the marker and the parked copy
    /// unconditionally, so a filesystem error left the mutant installed with
    /// its only backup gone.
    fn disarm(&mut self) -> Result<(), String> {
        while self.stage != Stage::Clean {
            self.step()?;
        }
        Ok(())
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        // The emergency path: something is unwinding, so nothing here can be
        // returned. It runs the same steps and stops at the first that fails,
        // leaving whatever that step was protecting -- a recovery that deletes
        // its own backup is worse than one that leaves a marker behind.
        while self.stage != Stage::Clean {
            if let Err(e) = self.step() {
                eprintln!("mutation-check: {e}");
                return;
            }
        }
    }
}

/// Put back whatever a killed run left behind, before measuring anything.
fn recover_interrupted(root: &Path) -> Result<(), String> {
    let dir = root.join(RESTORE_DIR);
    let marker = dir.join("in-progress");
    if !marker.exists() {
        return Ok(());
    }
    let file = std::fs::read_to_string(&marker)
        .map_err(|e| format!("cannot read {}: {e}", marker.display()))?;
    let original = std::fs::read_to_string(dir.join("original"))
        .map_err(|e| format!("cannot read the parked original: {e}"))?;
    let path = root.join(file.trim());
    std::fs::write(&path, &original)
        .map_err(|e| format!("cannot restore {}: {e}", path.display()))?;
    std::fs::remove_file(&marker).map_err(|e| format!("cannot clear the marker: {e}"))?;
    let _ = std::fs::remove_file(dir.join("original"));
    println!(
        "restored {} from an interrupted run before measuring anything",
        file.trim()
    );
    Ok(())
}

/// The longest any single lane run may take.
///
/// **A mutation can hang rather than fail.** A substitution that removes a
/// reservation can leave a later `insert` panicking inside a thread the harness
/// then waits on forever, and the tree stays mutated for as long as that lasts:
/// an independent review found `T0029` stuck for over twenty-five minutes on a
/// futex, with the source file still carrying the mutant. A lane that does not
/// finish is a lane that failed, and saying so is what lets the run move on and
/// put the file back.
const LANE_TIMEOUT: Duration = Duration::from_secs(600);

fn lane_passes(root: &Path, lane: &Lane) -> bool {
    // `timeout --kill-after` rather than a wait loop: the child is a `cargo`
    // that spawns a test binary, and killing the group is what actually stops
    // a wedged test process.
    let mut command = Command::new("timeout");
    command
        .arg("--kill-after=30s")
        .arg(format!("{}s", LANE_TIMEOUT.as_secs()))
        .arg("cargo")
        .args(lane.argv)
        .current_dir(root);
    command
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn builds(root: &Path, groups: &[&[&str]]) -> bool {
    groups.iter().all(|extra| {
        let mut c = Command::new("cargo");
        c.args(["build", "--offline", "--locked"]).args(*extra);
        c.current_dir(root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Run one lane `REPEATS` times; `None` if it disagrees with itself.
fn repeated(root: &Path, lane: &Lane, expect: bool) -> Option<bool> {
    let first = lane_passes(root, lane);
    for _ in 1..REPEATS {
        if lane_passes(root, lane) != first {
            return None;
        }
    }
    Some(first == expect)
}

/// What a mutation's run actually established.
///
/// Experiment 0005's second review is why only `Caught` counts: its first
/// driver appended unstable and failing-control verdicts to a list and then
/// added them to the total, so a run whose restored control never passed
/// printed "1 of 1 caught".
fn classify(
    caught_by: &[&str],
    mutant_ok: Option<bool>,
    control_ok: Option<bool>,
    expect: Expect,
) -> Verdict {
    if expect == Expect::Survivor {
        return if caught_by.is_empty() {
            Verdict::ControlHeld
        } else {
            Verdict::ControlBroken
        };
    }
    if caught_by.is_empty() {
        return Verdict::Survivor;
    }
    match (mutant_ok, control_ok) {
        (Some(true), Some(true)) => Verdict::Caught,
        (Some(true), Some(false)) => Verdict::InvalidControl,
        _ => Verdict::Unstable,
    }
}

struct Battery {
    tag: &'static str,
    lanes: &'static [Lane],
    mutations: &'static [Mutation],
    builds: &'static [&'static [&'static str]],
}

include!("mutationtable.rs");

/// The batteries this command can run.
///
/// Experiment 0005's battery is **not** here. Thirteen of its twenty-four
/// anchors already matched nothing at `ccfd7fa`: the importer it measured was
/// rewritten by tasks 0025 and 0026, and nobody noticed because nobody ran it.
/// A mutation that does not apply is not evidence -- that is this command's own
/// rule -- so carrying an unrunnable battery would be carrying a number that
/// cannot be reproduced. Its record keeps what it measured, and says when the
/// driver was retired.
const BATTERIES: &[Battery] = &[
    Battery {
        tag: "0006",
        lanes: LANES_T0006,
        mutations: BATTERY_T0006,
        builds: BUILDS_T0006,
    },
    // Task 0028's battery needs a GPU and the CUDA toolkit, because its
    // mutations are wrong *answers* rather than wrong control flow. Running it
    // on a machine with neither reports every lane as stably failing, which the
    // baseline already refuses to build verdicts on.
    Battery {
        tag: "0028",
        lanes: LANES_T0028,
        mutations: BATTERY_T0028,
        builds: BUILDS_T0028,
    },
    // Task 0029's battery needs a GPU for its device lane, and both of its
    // shapes are wrong *answers* rather than crashes: a failed reservation that
    // yields a value, and a refusal returned after something has moved.
    Battery {
        tag: "0029",
        lanes: LANES_T0029,
        mutations: BATTERY_T0029,
        builds: BUILDS_T0029,
    },
];

/// What every lane does on the **clean** tree, measured once.
///
/// `Some(true)` stably passes, `Some(false)` stably fails, `None` disagrees with
/// itself. This is the control side of every verdict below, and it is one fact
/// about the tree rather than fifty: re-proving it after each substitution cost
/// an extra rebuild and `REPEATS` more lane runs per mutation, which is most of
/// why a full battery took four hours.
///
/// The trade is stated rather than hidden: a tree that breaks *during* the run
/// is noticed at the end, when the baseline is taken again, instead of at the
/// mutation that broke it.
fn baseline(root: &Path, lanes: &[Lane]) -> Vec<(&'static str, Option<bool>)> {
    lanes
        .iter()
        .map(|l| {
            let first = lane_passes(root, l);
            let mut stable = true;
            for _ in 1..REPEATS {
                if lane_passes(root, l) != first {
                    stable = false;
                }
            }
            (l.name, if stable { Some(first) } else { None })
        })
        .collect()
}

/// Split requested names into (chosen, unknown), keeping battery order.
fn select<'a>(requested: &[String], battery: &'a [Mutation]) -> (Vec<&'a Mutation>, Vec<String>) {
    let unknown: Vec<String> = requested
        .iter()
        .filter(|r| !battery.iter().any(|m| m.name == r.as_str()))
        .cloned()
        .collect();
    let chosen: Vec<&Mutation> = battery
        .iter()
        .filter(|m| requested.iter().any(|r| r == m.name))
        .collect();
    (chosen, unknown)
}

/// Deterministic checks of the verdict rule and the selector.
fn self_test() -> i32 {
    let f = |lanes: &[&str], m: Option<bool>, c: Option<bool>, e: Expect| classify(lanes, m, c, e);
    let cases: &[(Verdict, Verdict)] = &[
        (
            f(&[], Some(true), Some(true), Expect::Caught),
            Verdict::Survivor,
        ),
        (f(&[], None, None, Expect::Caught), Verdict::Survivor),
        (
            f(&["format"], Some(true), Some(true), Expect::Caught),
            Verdict::Caught,
        ),
        (
            f(&["format"], None, Some(true), Expect::Caught),
            Verdict::Unstable,
        ),
        (
            f(&["format"], Some(true), None, Expect::Caught),
            Verdict::Unstable,
        ),
        (
            f(&["format"], Some(false), Some(true), Expect::Caught),
            Verdict::Unstable,
        ),
        (
            f(&["format"], Some(true), Some(false), Expect::Caught),
            Verdict::InvalidControl,
        ),
        (
            f(&["format"], Some(false), Some(false), Expect::Caught),
            Verdict::Unstable,
        ),
        (f(&[], None, None, Expect::Survivor), Verdict::ControlHeld),
        (
            f(&["arch"], Some(true), Some(true), Expect::Survivor),
            Verdict::ControlBroken,
        ),
    ];
    let mut bad = 0usize;
    for (got, want) in cases {
        if got != want {
            bad += 1;
            println!("SELF-TEST FAIL: want {want:?}, got {got:?}");
        }
    }

    let fake = &[
        Mutation {
            name: "a",
            file: "",
            from: "",
            to: "",
            expect: Expect::Caught,
        },
        Mutation {
            name: "b",
            file: "",
            from: "",
            to: "",
            expect: Expect::Caught,
        },
    ];
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let selector_cases: &[(Vec<String>, Vec<&str>, Vec<&str>)] = &[
        (s(&["a"]), vec!["a"], vec![]),
        (s(&["a", "b"]), vec!["a", "b"], vec![]),
        (s(&["typo"]), vec![], vec!["typo"]),
        (s(&["a", "typo"]), vec!["a"], vec!["typo"]),
        (s(&[]), vec![], vec![]),
    ];
    let mut sbad = 0usize;
    for (requested, want_chosen, want_unknown) in selector_cases {
        let (chosen, unknown) = select(requested, fake);
        let got_chosen: Vec<&str> = chosen.iter().map(|m| m.name).collect();
        if &got_chosen != want_chosen || &unknown != want_unknown {
            sbad += 1;
            println!("SELF-TEST FAIL select({requested:?}): got {got_chosen:?} / {unknown:?}");
        }
    }

    // Every anchor must occur exactly once, or the battery would silently skip
    // it. Checked here so a rewrite that moves code is a self-test failure
    // rather than a skipped line in a four-hour run.
    let root = super::workspace_root();
    let mut anchors = 0usize;
    let mut stale = 0usize;
    for b in BATTERIES {
        for m in b.mutations {
            anchors += 1;
            let path = root.join(m.file);
            match std::fs::read_to_string(&path) {
                Ok(text) => {
                    let n = text.matches(m.from).count();
                    if n != 1 {
                        stale += 1;
                        println!("SELF-TEST FAIL anchor {}: occurs {n} time(s)", m.name);
                    }
                }
                Err(e) => {
                    stale += 1;
                    println!("SELF-TEST FAIL anchor {}: {} ({e})", m.name, path.display());
                }
            }
        }
    }

    let total = cases.len() + selector_cases.len() + anchors;
    let wrong = bad + sbad + stale;
    println!(
        "self-test: {} of {total} case(s) correct ({} verdict, {} selector, {anchors} anchor)",
        total - wrong,
        cases.len(),
        selector_cases.len()
    );
    if wrong == 0 { 0 } else { 1 }
}

pub fn run(args: &[String]) -> i32 {
    let root = super::workspace_root();
    if let Err(e) = recover_interrupted(&root) {
        eprintln!("mutation-check: {e}");
        return 2;
    }
    if args.iter().any(|a| a == "--self-test") {
        return self_test();
    }
    let tag = args
        .iter()
        .position(|a| a == "--battery")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("0006");
    let Some(battery) = BATTERIES.iter().find(|b| b.tag == tag) else {
        eprintln!(
            "mutation-check: no battery '{tag}'. Known: {}",
            BATTERIES
                .iter()
                .map(|b| b.tag)
                .collect::<Vec<_>>()
                .join(", ")
        );
        return 2;
    };

    let requested: Vec<String> = {
        let mut out = Vec::new();
        let mut skip_next = false;
        for a in args {
            if skip_next {
                skip_next = false;
                continue;
            }
            if a == "--battery" {
                skip_next = true;
                continue;
            }
            if !a.starts_with("--") {
                out.push(a.clone());
            }
        }
        out
    };
    let chosen: Vec<&Mutation> = if requested.is_empty() {
        battery.mutations.iter().collect()
    } else {
        let (chosen, unknown) = select(&requested, battery.mutations);
        if !unknown.is_empty() {
            for name in &unknown {
                println!("UNKNOWN MUTATION {name}");
            }
            println!("refusing to run: {} unknown selector(s)", unknown.len());
            return 2;
        }
        if chosen.is_empty() {
            println!("refusing to run: the selected battery is empty");
            return 2;
        }
        chosen
    };

    // The control side of every verdict, measured once, before anything is
    // edited. A lane that cannot pass on a clean tree cannot tell us anything
    // about a mutant, and the battery says so rather than reporting fifty
    // invalid controls one at a time.
    println!(
        "measuring the clean-tree baseline over {} lane(s)...",
        battery.lanes.len()
    );
    if !builds(&root, battery.builds) {
        eprintln!("mutation-check: the clean tree does not build; nothing was measured");
        return 2;
    }
    let base = baseline(&root, battery.lanes);
    let bad: Vec<&str> = base
        .iter()
        .filter(|(_, ok)| *ok != Some(true))
        .map(|(n, _)| *n)
        .collect();
    if !bad.is_empty() {
        for (name, ok) in &base {
            if *ok != Some(true) {
                let what = match ok {
                    None => "disagrees with itself",
                    Some(false) => "fails",
                    Some(true) => unreachable!(),
                };
                println!("  BASELINE {name}: {what} on the clean tree");
            }
        }
        eprintln!(
            "mutation-check: {} lane(s) cannot pass on a clean tree, so no verdict below would \
             mean anything. Nothing was measured.",
            bad.len()
        );
        return 2;
    }
    println!(
        "baseline: {} lane(s) stably pass, {REPEATS} repetition(s) each",
        base.len()
    );

    let mut outcomes: Vec<(&str, Verdict, Vec<&str>, Expect)> = Vec::new();
    let mut skipped: Vec<(&str, String)> = Vec::new();

    for m in chosen {
        let path = root.join(m.file);
        let Ok(original) = std::fs::read_to_string(&path) else {
            skipped.push((m.name, format!("cannot read {}", path.display())));
            println!("SKIP {}: cannot read {}", m.name, path.display());
            continue;
        };
        let n = original.matches(m.from).count();
        if n != 1 {
            skipped.push((m.name, format!("anchor occurs {n} time(s)")));
            println!("SKIP {}: anchor occurs {n} time(s)", m.name);
            continue;
        }

        let mut caught_by: Vec<&str> = Vec::new();
        let mut mutant_ok = None;
        {
            let mut guard = match Restore::arm(&root, m.file, &original) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("mutation-check: {e}");
                    return 2;
                }
            };
            // **Every `continue` below leaves through `Drop`.** That is the
            // emergency path, which cannot report a failed restore, so each one
            // disarms explicitly first and stops the whole run if the file did
            // not go back.
            if std::fs::write(&path, original.replacen(m.from, m.to, 1)).is_err() {
                if let Err(e) = guard.disarm() {
                    eprintln!("mutation-check: {e}");
                    return 2;
                }
                skipped.push((m.name, "cannot apply".into()));
                println!("SKIP {}: cannot apply", m.name);
                continue;
            }
            if !builds(&root, battery.builds) {
                if let Err(e) = guard.disarm() {
                    eprintln!("mutation-check: {e}");
                    return 2;
                }
                skipped.push((m.name, "does not compile".into()));
                println!("SKIP {}: does not compile", m.name);
                continue;
            }
            match m.expect {
                // A mutant needs **one** failing lane. Stopping there rather
                // than running the other eleven to fill in a column is most of
                // the remaining cost.
                Expect::Caught => {
                    for lane in battery.lanes {
                        if !lane_passes(&root, lane) {
                            caught_by.push(lane.name);
                            mutant_ok = repeated(&root, lane, false);
                            break;
                        }
                    }
                }
                // An independence control claims **no** lane fails, so every
                // lane has to run, and the claim is repeated. The second review
                // found controls certified from a single pass while the summary
                // said three.
                Expect::Survivor => {
                    for lane in battery.lanes {
                        let mut failed = false;
                        for _ in 0..REPEATS {
                            if !lane_passes(&root, lane) {
                                failed = true;
                            }
                        }
                        if failed {
                            caught_by.push(lane.name);
                        }
                    }
                }
            }
            // **Restored here, not by `Drop`.** A failure to put the file back
            // is the one thing this run must not swallow: it stops, with the
            // marker and the parked copy still on disk for the next run.
            if let Err(e) = guard.disarm() {
                eprintln!("mutation-check: {e}");
                return 2;
            }
        }

        // The control side is the baseline above, taken once on the clean tree.
        let control_ok = caught_by
            .first()
            .and_then(|n| base.iter().find(|(name, _)| name == n))
            .map(|(_, ok)| *ok)
            .unwrap_or(Some(true));

        let verdict = classify(&caught_by, mutant_ok, control_ok, m.expect);
        let detail = if caught_by.is_empty() {
            "-".to_string()
        } else {
            caught_by.join(",")
        };
        println!("{:52} {:15} {detail}", m.name, verdict.name());
        outcomes.push((m.name, verdict, caught_by, m.expect));
    }

    // Put the tree back the way a build expects it, and confirm the baseline
    // still holds: the one thing the per-mutation control used to cover.
    if !builds(&root, battery.builds) {
        eprintln!("mutation-check: the tree does not build after the battery");
        return 2;
    }
    let after = baseline(&root, battery.lanes);
    let drifted: Vec<&str> = after
        .iter()
        .filter(|(_, ok)| *ok != Some(true))
        .map(|(n, _)| *n)
        .collect();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (_, v, _, _) in &outcomes {
        *counts.entry(v.name()).or_default() += 1;
    }
    let mutants = outcomes
        .iter()
        .filter(|(_, _, _, e)| *e == Expect::Caught)
        .count();
    let controls = outcomes.len() - mutants;
    let caught = *counts.get("caught").unwrap_or(&0);
    let held = *counts.get("control-held").unwrap_or(&0);
    println!(
        "\n{caught} of {mutants} mutant(s) caught, {held} of {controls} expected survivor(s) held \
         (independence controls and equivalent mutants); {} survivor(s), {} unstable, {} invalid \
         control(s), {} broken control(s), {} skipped",
        counts.get("survivor").unwrap_or(&0),
        counts.get("unstable").unwrap_or(&0),
        counts.get("invalid-control").unwrap_or(&0),
        counts.get("control-broken").unwrap_or(&0),
        skipped.len(),
    );
    // What was actually measured, rather than a sentence about repetitions that
    // some verdicts never received.
    println!(
        "measured: the clean-tree baseline over all {} lane(s), {REPEATS} repetition(s) each, \
         before and after; each caught mutant's deciding lane repeated {REPEATS} time(s) under \
         the mutation; each expected survivor's full lane set repeated {REPEATS} time(s)",
        base.len()
    );
    if !drifted.is_empty() {
        println!(
            "  BASELINE DRIFT after the battery: {drifted:?} no longer pass on the restored tree"
        );
    }
    for (name, verdict, _, _) in &outcomes {
        if !matches!(verdict, Verdict::Caught | Verdict::ControlHeld) {
            println!("  {} {name}", verdict.name().to_uppercase());
        }
    }
    for (name, why) in &skipped {
        println!("  SKIPPED {name}: {why}");
    }
    // An incomplete or invalid battery is not a measurement.
    let ok = caught == mutants && held == controls && skipped.is_empty() && drifted.is_empty();
    if ok { 0 } else { 1 }
}

/// `cargo xtask reference-check --artifact <dir>` -- does the **reference**
/// safetensors implementation accept what we published?
///
/// It runs a foreign implementation on purpose, and that is why the comparison
/// itself lives in a small script this workspace does not link: adding
/// `safetensors` to `Cargo.toml` would both breach task 0026's "no new
/// third-party dependency" and weaken the claim, because our reader agreeing
/// with a crate we vendored is a weaker statement than our bytes being accepted
/// by an implementation that has never heard of us.
///
/// The entry point is here so that every gate in this repository is a
/// `cargo xtask` command, whatever it has to invoke underneath.
pub fn reference_check(args: &[String]) -> i32 {
    let Some(artifact) = args
        .iter()
        .position(|a| a == "--artifact")
        .and_then(|i| args.get(i + 1))
    else {
        eprintln!("reference-check: --artifact <dir> is required");
        return 2;
    };
    let root = super::workspace_root();
    let driver = root.join("tools/experiments/0007-reference-reader.py");
    if !driver.exists() {
        eprintln!("reference-check: {} is missing", driver.display());
        return 2;
    }
    match Command::new("python3")
        .arg(&driver)
        .arg(artifact)
        .current_dir(&root)
        .status()
    {
        Ok(s) => s.code().unwrap_or(1),
        Err(e) => {
            eprintln!(
                "reference-check: cannot run the reference reader: {e}\n\
                 It needs `python3` with the `safetensors` package, which is the reference \
                 implementation's own binding. Nothing was measured, so nothing passed."
            );
            2
        }
    }
}

#[cfg(test)]
mod restore_tests {
    use super::{Restore, Stage};

    /// A scratch directory that removes itself.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "moxie-restore-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0)
            ));
            std::fs::create_dir_all(&dir).expect("a scratch directory");
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// An armed guard over a real file, plus the root it lives in.
    fn armed(scratch: &Scratch, body: &str) -> Restore {
        std::fs::write(scratch.0.join("subject.rs"), body).expect("the subject writes");
        Restore::arm(&scratch.0, "subject.rs", body).expect("an armed guard")
    }

    #[test]
    fn the_ordinary_path_restores_and_clears_both_artifacts() {
        let scratch = Scratch::new("ordinary");
        let mut guard = armed(&scratch, "original\n");
        std::fs::write(scratch.0.join("subject.rs"), "mutant\n").expect("the mutant writes");

        guard.disarm().expect("disarm");

        assert_eq!(guard.stage, Stage::Clean);
        assert_eq!(
            std::fs::read_to_string(scratch.0.join("subject.rs")).expect("the subject reads"),
            "original\n"
        );
        assert!(!guard.marker.exists(), "the marker survived a clean disarm");
        assert!(
            !guard.parked().exists(),
            "the parked original survived a clean disarm"
        );
    }

    /// **The rewrite fails.** Both artifacts must survive: the mutant is still
    /// installed and the parked copy is the only way back.
    #[test]
    fn a_failed_rewrite_keeps_the_marker_and_the_parked_original() {
        let scratch = Scratch::new("rewrite");
        let mut guard = armed(&scratch, "original\n");
        // A directory cannot be overwritten by a file write.
        std::fs::remove_file(scratch.0.join("subject.rs")).expect("remove the file");
        std::fs::create_dir(scratch.0.join("subject.rs")).expect("a directory in its place");

        let error = guard
            .disarm()
            .expect_err("a rewrite over a directory fails");

        assert!(error.contains("cannot restore"), "{error}");
        assert_eq!(guard.stage, Stage::Mutated);
        assert!(guard.marker.exists(), "the marker was cleared anyway");
        assert!(
            guard.parked().exists(),
            "the parked original was removed while the mutant was still installed"
        );
        guard.stage = Stage::Clean;
    }

    /// **The marker removal fails.** Both artifacts must survive, because a
    /// marker without its copy is the one unrecoverable pair.
    #[test]
    fn a_failed_marker_removal_keeps_the_parked_original() {
        let scratch = Scratch::new("marker");
        let mut guard = armed(&scratch, "original\n");
        std::fs::remove_file(&guard.marker).expect("remove the marker");
        // A directory at the marker's path: `remove_file` refuses it.
        std::fs::create_dir(&guard.marker).expect("a directory in its place");

        let error = guard
            .disarm()
            .expect_err("removing a directory as a file fails");

        assert!(error.contains("marker could not be cleared"), "{error}");
        assert_eq!(guard.stage, Stage::Restored);
        assert!(
            guard.parked().exists(),
            "the parked original was removed while its marker survived -- the one \
             combination nothing can recover from"
        );
        guard.stage = Stage::Clean;
    }

    /// **The parked-original removal fails.** The marker is already gone, so
    /// nothing looks for the copy: untidy, not unsafe, and the state says so.
    #[test]
    fn a_failed_parked_removal_stops_at_the_last_step() {
        let scratch = Scratch::new("parked");
        let mut guard = armed(&scratch, "original\n");
        std::fs::remove_file(guard.parked()).expect("remove the parked original");
        std::fs::create_dir(guard.parked()).expect("a directory in its place");

        let error = guard
            .disarm()
            .expect_err("removing a directory as a file fails");

        assert!(error.contains("parked original"), "{error}");
        assert_eq!(
            guard.stage,
            Stage::MarkerCleared,
            "the guard did not record that the marker was already cleared, so a retry \
             would attempt a removal that has already happened"
        );
        assert!(!guard.marker.exists(), "the marker was not cleared");
        guard.stage = Stage::Clean;
    }

    /// A retry after a failed cleanup resumes where it stopped rather than
    /// repeating a step that already succeeded.
    #[test]
    fn a_retry_resumes_from_the_step_that_failed() {
        let scratch = Scratch::new("retry");
        let mut guard = armed(&scratch, "original\n");
        std::fs::remove_file(guard.parked()).expect("remove the parked original");
        std::fs::create_dir(guard.parked()).expect("a directory in its place");
        guard.disarm().expect_err("the last step fails");
        assert_eq!(guard.stage, Stage::MarkerCleared);

        // Clear the obstruction and retry: it must not try the marker again.
        std::fs::remove_dir(guard.parked()).expect("clear the obstruction");
        std::fs::write(guard.parked(), "original\n").expect("a real parked copy");

        guard.disarm().expect("the retry completes");
        assert_eq!(guard.stage, Stage::Clean);
    }
}
