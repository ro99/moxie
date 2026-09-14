#!/usr/bin/env python3
"""Mutation battery for task 0025 — experiment 0006's driver.

Committed rather than described, for the reason experiment 0005 records: the
mutation *names* are not the measurement, these exact substitutions are.

Method, unchanged from experiments 0002–0005: each mutation is a single edit
that changes behaviour and still compiles. For each one — apply, build, run
every lane, record which lanes caught it, revert — with every verdict repeated
REPEATS times in both directions, because a substitution read once from a flaky
lane is a coin flip recorded as a measurement.

One thing is new here, and task 0025's contract is why: it asks for
"substitutions that remove each protection", and for the architecture rule
that includes proving the rule is **not** the allowlist in disguise. So a
mutation may declare `expect=SURVIVOR`: an independence control, where removing
some *other* protection must leave the battery green because the protection
under test still holds. A control that is caught is as interesting as a mutant
that is not, and both are reported.

Usage:  python3 docs/evidence/experiments/drivers/0006-mutations.py [name ...]
        python3 docs/evidence/experiments/drivers/0006-mutations.py --self-test

It edits tracked source in place and restores it in a `finally`. Run it on a
clean tree; `git status` afterwards is part of the evidence.
"""

import collections
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

RUN = os.path.join(ROOT, "crates/moxie-storage-write/src/run.rs")
PLAN = os.path.join(ROOT, "crates/moxie-storage-write/src/plan.rs")
WLIB = os.path.join(ROOT, "crates/moxie-storage-write/src/lib.rs")
PAY = os.path.join(ROOT, "crates/moxie-format/src/payload.rs")
JOU = os.path.join(ROOT, "crates/moxie-format/src/journal.rs")
MAN = os.path.join(ROOT, "crates/moxie-format/src/manifest.rs")
SEL = os.path.join(ROOT, "crates/moxie-format/src/selection.rs")
RLIB = os.path.join(ROOT, "crates/moxie-repack/src/lib.rs")
WORK = os.path.join(ROOT, "crates/moxie-repack/src/work.rs")
ARCH = os.path.join(ROOT, "xtask/src/archcheck.rs")

REPEATS = 3

LANES = {
    "format": ["cargo", "test", "-p", "moxie-format", "--lib", "--offline", "--locked"],
    "manifest": ["cargo", "test", "-p", "moxie-format", "--offline", "--locked",
                 "--test", "manifest_v1"],
    "publication": ["cargo", "test", "-p", "moxie-storage-write", "--offline", "--locked"],
    "roundtrip": ["cargo", "test", "-p", "moxie-repack", "--offline", "--locked",
                  "--test", "round_trip"],
    "cli": ["cargo", "test", "-p", "moxie-repack", "--offline", "--locked", "--test", "cli"],
    "budget": ["cargo", "test", "-p", "moxie-repack", "--offline", "--locked",
               "--test", "budget"],
    "arch": ["cargo", "run", "--offline", "--locked", "-q", "--bin", "xtask", "--",
             "arch-check"],
}

CAUGHT = "caught"
SURVIVOR = "survivor"
UNSTABLE = "unstable"
INVALID_CONTROL = "invalid-control"
CONTROL_HELD = "control-held"
CONTROL_BROKEN = "control-broken"

# name -> (file, old, new, expect). Every `old` must occur exactly once.
# `expect` is CAUGHT for a mutant and SURVIVOR for an independence control.
MUTATIONS = [
    # --- checksums: what the gates are actually comparing -------------------
    ("unit-checksum-not-compared", RUN,
     "        if got != unit.sha256 {",
     "        if false && got != unit.sha256 {", CAUGHT),
    ("tensor-hash-never-sees-the-bytes", RUN,
     "        progress.hasher.update(bytes);",
     "        let _ = &mut progress.hasher;", CAUGHT),
    ("published-validation-skipped", RUN,
     """        faults.check(Site::Validate)?;
        let artifact = Artifact::open_unpublished(&self.dest, &staged, ByteBudget::default())?;
        for t in &sealed {""",
     """        let artifact = Artifact::open_unpublished(&self.dest, &staged, ByteBudget::default())?;
        for t in sealed.iter().take(0) {""", CAUGHT),
    ("source-digest-is-a-constant", RLIB,
     '        let (digest, bytes) = sources.file_digest(&file, buffers.source_tile_mut())?;',
     '        let (digest, bytes) = ("0".repeat(64), 0u64);\n        let _ = buffers.source_tile_mut();',
     CAUGHT),
    ("unit-source-digest-is-recorded-not-checked", WORK,
     "fn sha256_of(bytes: &[u8]) -> String {\n    let mut h = StreamingSha256::new();\n    h.update(bytes);\n    h.finalize_hex()\n}",
     "fn sha256_of(bytes: &[u8]) -> String {\n    let _ = bytes;\n    \"0\".repeat(64)\n}",
     SURVIVOR),

    # --- source, plan and version binding ------------------------------------
    ("resume-ignores-its-binding", RUN,
     "        if recorded != self.binding {",
     "        if false && recorded != self.binding {", CAUGHT),
    ("run-binding-omits-the-source-digests", RLIB,
     """    field(b"repack-run-v1");
    field(plan.as_bytes());
    field(selection.as_bytes());
    for (file, digest) in sources {
        field(file.as_bytes());
        field(digest.as_bytes());
    }""",
     """    field(b"repack-run-v1");
    field(plan.as_bytes());
    field(selection.as_bytes());
    let _ = sources;""", CAUGHT),
    ("journal-version-not-checked", JOU,
     "                if v != JOURNAL_VERSION {",
     "                if false && v != JOURNAL_VERSION {", CAUGHT),
    ("selection-version-not-checked", SEL,
     "    if v.version != SELECTION_VERSION {",
     "    if false && v.version != SELECTION_VERSION {", CAUGHT),
    ("manifest-schema-version-not-checked", MAN,
     "    if v.schema_version != SCHEMA_VERSION {",
     "    if false && v.schema_version != SCHEMA_VERSION {", CAUGHT),

    # --- premature publication and false completion --------------------------
    ("manifest-staged-under-its-final-name", RUN,
     "        let staged = self.dest.join(STAGED_MANIFEST_FILE);",
     "        let staged = self.dest.join(MANIFEST_FILE);", CAUGHT),
    ("incomplete-tensors-can-be-sealed", RUN,
     "            if p.done != t.request.length {",
     "            if false && p.done != t.request.length {", CAUGHT),
    ("journal-recorded-before-the-bytes-are-durable", RUN,
     """        write_at(&mut file, offset, bytes, faults)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;
        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;

        let unit = CompletedUnit {
            tensor: role.to_string(),
            index,
            chunk: planned.chunk.clone(),
            offset,
            len: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            source_sha256: source_sha256.to_string(),
        };
        append_durably(&mut self.journal, &journal::unit_line(&unit), faults)?;""",
     """        let unit = CompletedUnit {
            tensor: role.to_string(),
            index,
            chunk: planned.chunk.clone(),
            offset,
            len: bytes.len() as u64,
            sha256: sha256_hex(bytes),
            source_sha256: source_sha256.to_string(),
        };
        append_durably(&mut self.journal, &journal::unit_line(&unit), faults)?;
        write_at(&mut file, offset, bytes, faults)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;
        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;""",
     CAUGHT),
    ("resume-trusts-the-journals-offsets", RUN,
     "        if unit.chunk != planned.chunk || unit.offset != planned.offset + progress.done {",
     "        if false && (unit.chunk != planned.chunk || unit.offset != planned.offset + progress.done) {",
     SURVIVOR),

    # --- syncs and error handling -------------------------------------------
    ("chunk-sync-skipped", RUN,
     """        faults.check(Site::ChunkSync)?;
        file.sync_all()
            .map_err(|e| invalid(format!("cannot sync {}: {e}", path.display())))?;""",
     """        let _ = &file;""", CAUGHT),
    ("directory-sync-skipped", RUN,
     """        faults.check(Site::DirectorySync)?;
        let dir = File::open(&self.dest).map_err(|e| {""",
     """        if true {
            return Ok(());
        }
        let dir = File::open(&self.dest).map_err(|e| {""", CAUGHT),
    ("write-errors-are-swallowed", RUN,
     """        write_at(&mut file, offset, bytes, faults)
            .map_err(|e| invalid(format!("cannot write {}: {e}", path.display())))?;""",
     """        let _ = write_at(&mut file, offset, bytes, faults);""", CAUGHT),
    ("journal-sync-skipped", RUN,
     """    faults.check(Site::JournalSync)?;
    file.sync_data()
        .map_err(|e| invalid(format!("cannot sync the journal: {e}")))""",
     """    Ok(())""", CAUGHT),

    # --- budgets -------------------------------------------------------------
    ("unit-size-not-checked-against-the-scratch", RUN,
     "        if bytes.len() > self.budget.scratch_bytes() {",
     "        if false && bytes.len() > self.budget.scratch_bytes() {", CAUGHT),
    # **Equivalent, with the argument written out.** `OutputPlan::build` refuses
    # a plan whose chunk lengths exceed the disk budget, and every `write_unit`
    # is bounded by its tensor's planned range, so `disk_used` can never exceed
    # the planned total. The runtime check is defence against a future
    # disagreement between the plan and the run; today it cannot fire, and a
    # test that made it fire would have to break that invariant first. Expected
    # to survive for that reason, not because a gate is missing.
    ("disk-budget-not-checked-while-writing", RUN,
     "        if new_disk > self.budget.disk_bytes() {",
     "        if false && new_disk > self.budget.disk_bytes() {", SURVIVOR),
    ("chunk-file-limit-ignored-by-the-plan", PLAN,
     "            if alone > budget.chunk_file_bytes() {",
     "            if false && alone > budget.chunk_file_bytes() {", CAUGHT),
    ("plan-ignores-the-disk-budget", PLAN,
     "        if total > budget.disk_bytes() {",
     "        if false && total > budget.disk_bytes() {", CAUGHT),
    ("tile-is-the-whole-scratch", RLIB,
     "        (self.scratch_bytes / 2).max(1)",
     "        (4usize << 20).max(self.scratch_bytes)", CAUGHT),
    ("header-budget-flag-ignored", RLIB,
     "    HeaderBudget::new(budgets.header_bytes).ok_or_else(|| {",
     "    Some(HeaderBudget::DEFAULT).ok_or_else(|| {", CAUGHT),

    # --- cancellation ordering ----------------------------------------------
    ("cancellation-not-checked-between-units", RLIB,
     "            if cancelled() {\n                let outcome = run.cancel(ledger)?;",
     "            if false && cancelled() {\n                let outcome = run.cancel(ledger)?;", CAUGHT),
    ("cancellation-not-checked-before-publish", RUN,
     """        if cancelled() {
            // The last point at which cancellation can be honoured: after this
            // the artifact exists.
            let bytes = self.progress.values().map(|p| p.done).sum();
            return self.cancel_with(bytes, ledger);
        }""",
     """        let _ = &cancelled;""", CAUGHT),

    # --- the canonical payload ----------------------------------------------
    ("payload-sections-in-the-wrong-order", PAY,
     """    Ok(PayloadExtents {
        codes: 0..code_bytes,
        scales: code_bytes..scales_end,
        zero_points,
    })""",
     """    Ok(PayloadExtents {
        codes: (scales_end - code_bytes)..scales_end,
        scales: 0..scale_bytes,
        zero_points,
    })""", CAUGHT),
    ("zero-point-section-dropped", PAY,
     "            Some(scales_end..end)",
     "            let _ = end;\n            None", CAUGHT),
    ("scale-block-not-validated", PAY,
     "        if !v.is_finite() || v <= 0.0 {",
     "        if false && (!v.is_finite() || v <= 0.0) {", CAUGHT),
    # `if need != length {` occurs twice in this file -- the BF16 byte-count
    # rule is the other one -- so the anchor carries the line that follows it.
    ("affine-length-rule-deleted", MAN,
     """            if need != length {
                return Err(invalid(format_args!(
                    "tensor '{role}': a {} {out_features}x{in_features} tensor with {} \\""",
     """            if false && need != length {
                return Err(invalid(format_args!(
                    "tensor '{role}': a {} {out_features}x{in_features} tensor with {} \\""",
     CAUGHT),

    # --- the architecture rule, and its independence from the allowlist -----
    ("write-authority-rule-disabled", ARCH,
     "    out.extend(check_write_authority(root, workspace.as_ref())?);",
     "    let _ = check_write_authority(root, workspace.as_ref())?;", CAUGHT),
    ("write-authority-consumers-widened", ARCH,
     'const WRITE_AUTHORITY_CONSUMERS: &[&str] = &["moxie-repack"];',
     'const WRITE_AUTHORITY_CONSUMERS: &[&str] = &["moxie-repack", "moxie-engine", "moxie-cli"];',
     CAUGHT),
    ("allowlist-permits-the-writer-in-the-engine", ARCH,
     """        (
            "moxie-engine",
            Allowed {
                // ADR 0011: explicit bounded host-reference generation.
                workspace: &[
                    "moxie-types",
                    "moxie-graph",
                    "moxie-interp",
                    "moxie-state",
                    "moxie-memory",
                ],""",
     """        (
            "moxie-engine",
            Allowed {
                // ADR 0011: explicit bounded host-reference generation.
                workspace: &[
                    "moxie-types",
                    "moxie-graph",
                    "moxie-interp",
                    "moxie-state",
                    "moxie-memory",
                    "moxie-storage-write",
                ],""",
     SURVIVOR),
]


def lane(name):
    r = subprocess.run(LANES[name], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


def build():
    for extra in (["-p", "moxie-format"], ["-p", "moxie-storage-write", "--tests"],
                  ["-p", "moxie-repack", "--tests"], ["-p", "xtask"]):
        r = subprocess.run(["cargo", "build", "--offline", "--locked"] + extra,
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            return False
    return True


def repeated(name, expect):
    """Run one lane REPEATS times; None if it disagrees with itself."""
    seen = {lane(name) for _ in range(REPEATS)}
    if len(seen) != 1:
        return None
    return seen.pop() == expect


def classify(caught_by, mutant_ok, control_ok, expect=CAUGHT):
    """What a mutation's run actually established.

    Experiment 0005's second review is why only `caught` counts: its first
    driver appended unstable and failing-control verdicts to a list and then
    added them to the total, so a run whose restored control never passed
    printed "1 of 1 caught".

    `mutant_ok` / `control_ok` are `True` (stably as expected), `False` (stably
    the opposite) or `None` (the lane disagreed with itself).

    An independence control (`expect=SURVIVOR`) inverts the question: it must
    **not** be caught, because the protection it removes is not the one the
    battery is measuring. A control that is caught means the two protections
    are entangled, which is worth knowing and is reported as `control-broken`.
    """
    if expect == SURVIVOR:
        if caught_by:
            return CONTROL_BROKEN
        return CONTROL_HELD
    if not caught_by:
        return SURVIVOR
    if mutant_ok is None or control_ok is None:
        return UNSTABLE
    if mutant_ok is not True:
        return UNSTABLE
    if control_ok is not True:
        return INVALID_CONTROL
    return CAUGHT


def self_test():
    """Deterministic checks of the verdict rule, run with `--self-test`."""
    cases = [
        (([], True, True, CAUGHT), SURVIVOR),
        (([], None, None, CAUGHT), SURVIVOR),
        ((["format"], True, True, CAUGHT), CAUGHT),
        ((["format"], None, True, CAUGHT), UNSTABLE),
        ((["format"], True, None, CAUGHT), UNSTABLE),
        ((["format"], False, True, CAUGHT), UNSTABLE),
        ((["format"], True, False, CAUGHT), INVALID_CONTROL),
        ((["format"], False, False, CAUGHT), UNSTABLE),
        # Independence controls: not caught is the passing verdict.
        (([], None, None, SURVIVOR), CONTROL_HELD),
        ((["arch"], True, True, SURVIVOR), CONTROL_BROKEN),
    ]
    bad = [(a, want, classify(*a)) for a, want in cases if classify(*a) != want]
    for a, want, got in bad:
        print(f"SELF-TEST FAIL {a}: want {want}, got {got}")

    fake = [("a", None, None, None, CAUGHT), ("b", None, None, None, CAUGHT)]
    selector_cases = [
        (["a"], (["a"], [])),
        (["a", "b"], (["a", "b"], [])),
        (["typo"], ([], ["typo"])),
        (["a", "typo"], (["a"], ["typo"])),
        ([], ([], [])),
    ]
    sbad = []
    for requested, (want_chosen, want_unknown) in selector_cases:
        chosen, unknown = select(requested, fake)
        got = ([c[0] for c in chosen], unknown)
        if got != (want_chosen, want_unknown):
            sbad.append((requested, (want_chosen, want_unknown), got))
            print(f"SELF-TEST FAIL select({requested}): want "
                  f"{(want_chosen, want_unknown)}, got {got}")

    total = len(cases) + len(selector_cases)
    print(f"self-test: {total - len(bad) - len(sbad)} of {total} cases correct "
          f"({len(cases)} verdict, {len(selector_cases)} selector)")
    return 0 if not bad and not sbad else 1


def select(requested, mutations):
    """Split requested names into (chosen, unknown), preserving battery order."""
    known = [m[0] for m in mutations]
    unknown = [r for r in requested if r not in known]
    chosen = [m for m in mutations if m[0] in requested]
    return chosen, unknown


def main():
    if "--self-test" in sys.argv[1:]:
        return self_test()
    requested = [a for a in sys.argv[1:] if not a.startswith("--")]
    if requested:
        battery, unknown = select(requested, MUTATIONS)
        if unknown:
            for name in unknown:
                print(f"UNKNOWN MUTATION {name}", flush=True)
            print(f"refusing to run: {len(unknown)} unknown selector(s)")
            return 2
        if not battery:
            print("refusing to run: the selected battery is empty")
            return 2
    else:
        battery = MUTATIONS
    outcomes = collections.OrderedDict()
    skipped = []
    for name, path, old, new, expect in battery:
        src = open(path).read()
        if src.count(old) != 1:
            skipped.append((name, f"anchor occurs {src.count(old)} time(s)"))
            print(f"SKIP {name}: anchor occurs {src.count(old)} time(s)", flush=True)
            continue
        open(path, "w").write(src.replace(old, new, 1))
        mutant_ok, caught_by = None, []
        try:
            if not build():
                skipped.append((name, "does not compile"))
                print(f"SKIP {name}: does not compile", flush=True)
                continue
            caught_by = [n for n in LANES if not lane(n)]
            if caught_by and expect == CAUGHT:
                mutant_ok = repeated(caught_by[0], False)
        finally:
            open(path, "w").write(src)
        control_ok = None
        if caught_by and expect == CAUGHT:
            if not build():
                skipped.append((name, "the tree did not rebuild after restore"))
                print(f"SKIP {name}: no rebuild after restore", flush=True)
                continue
            control_ok = repeated(caught_by[0], True)
        else:
            build()
        verdict = classify(caught_by, mutant_ok, control_ok, expect)
        outcomes[name] = (verdict, caught_by, expect)
        detail = ",".join(caught_by) if caught_by else "-"
        print(f"{name:52s} {verdict:15s} {detail}", flush=True)

    counts = collections.Counter(v for v, _, _ in outcomes.values())
    mutants = [n for n, (_, _, e) in outcomes.items() if e == CAUGHT]
    controls = [n for n, (_, _, e) in outcomes.items() if e == SURVIVOR]
    total = len(outcomes) + len(skipped)
    print(f"\n{counts[CAUGHT]} of {len(mutants)} mutant(s) caught, "
          f"{counts[CONTROL_HELD]} of {len(controls)} expected survivor(s) held "
          f"(independence controls and equivalent mutants); "
          f"{counts[SURVIVOR]} survivor(s), {counts[UNSTABLE]} unstable, "
          f"{counts[INVALID_CONTROL]} invalid control(s), "
          f"{counts[CONTROL_BROKEN]} broken control(s), {len(skipped)} skipped; "
          f"{REPEATS} repetition(s) of each verdict in both directions")
    for name, (verdict, _, _) in outcomes.items():
        if verdict not in (CAUGHT, CONTROL_HELD):
            print(f"  {verdict.upper()} {name}")
    for name, why in skipped:
        print(f"  SKIPPED {name}: {why}")
    # An incomplete or invalid battery is not a measurement.
    ok = (counts[CAUGHT] == len(mutants)
          and counts[CONTROL_HELD] == len(controls)
          and not skipped)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
