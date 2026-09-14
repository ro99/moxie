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
struct Restore {
    path: PathBuf,
    original: String,
    marker: PathBuf,
    armed: bool,
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
            armed: true,
        })
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let _ = std::fs::write(&self.path, &self.original);
        let _ = std::fs::remove_file(&self.marker);
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
    println!(
        "restored {} from an interrupted run before measuring anything",
        file.trim()
    );
    Ok(())
}

fn lane_passes(root: &Path, lane: &Lane) -> bool {
    Command::new("cargo")
        .args(lane.argv)
        .current_dir(root)
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
const BATTERIES: &[Battery] = &[Battery {
    tag: "0006",
    lanes: LANES_T0006,
    mutations: BATTERY_T0006,
    builds: BUILDS_T0006,
}];

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

        let caught_by: Vec<&str>;
        let mut mutant_ok = None;
        {
            let guard = match Restore::arm(&root, m.file, &original) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("mutation-check: {e}");
                    return 2;
                }
            };
            let _ = &guard;
            if std::fs::write(&path, original.replacen(m.from, m.to, 1)).is_err() {
                skipped.push((m.name, "cannot apply".into()));
                println!("SKIP {}: cannot apply", m.name);
                continue;
            }
            if !builds(&root, battery.builds) {
                skipped.push((m.name, "does not compile".into()));
                println!("SKIP {}: does not compile", m.name);
                continue;
            }
            caught_by = battery
                .lanes
                .iter()
                .filter(|l| !lane_passes(&root, l))
                .map(|l| l.name)
                .collect();
            if !caught_by.is_empty() && m.expect == Expect::Caught {
                let lane = battery
                    .lanes
                    .iter()
                    .find(|l| l.name == caught_by[0])
                    .expect("a lane that just ran");
                mutant_ok = repeated(&root, lane, false);
            }
            // `guard` drops here: the file is back before the control runs.
        }

        let mut control_ok = None;
        if !caught_by.is_empty() && m.expect == Expect::Caught {
            if !builds(&root, battery.builds) {
                skipped.push((m.name, "the tree did not rebuild after restore".into()));
                println!("SKIP {}: no rebuild after restore", m.name);
                continue;
            }
            let lane = battery
                .lanes
                .iter()
                .find(|l| l.name == caught_by[0])
                .expect("a lane that just ran");
            control_ok = repeated(&root, lane, true);
        } else {
            builds(&root, battery.builds);
        }

        let verdict = classify(&caught_by, mutant_ok, control_ok, m.expect);
        let detail = if caught_by.is_empty() {
            "-".to_string()
        } else {
            caught_by.join(",")
        };
        println!("{:52} {:15} {detail}", m.name, verdict.name());
        outcomes.push((m.name, verdict, caught_by, m.expect));
    }

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
         control(s), {} broken control(s), {} skipped; {REPEATS} repetition(s) of each verdict in \
         both directions",
        counts.get("survivor").unwrap_or(&0),
        counts.get("unstable").unwrap_or(&0),
        counts.get("invalid-control").unwrap_or(&0),
        counts.get("control-broken").unwrap_or(&0),
        skipped.len(),
    );
    for (name, verdict, _, _) in &outcomes {
        if !matches!(verdict, Verdict::Caught | Verdict::ControlHeld) {
            println!("  {} {name}", verdict.name().to_uppercase());
        }
    }
    for (name, why) in &skipped {
        println!("  SKIPPED {name}: {why}");
    }
    // An incomplete or invalid battery is not a measurement.
    let ok = caught == mutants && held == controls && skipped.is_empty();
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
