#!/usr/bin/env python3
"""Mutation battery for task 0024 — experiment 0005's driver.

Committed rather than described. The first version of experiment 0005 said the
driver was "reproduced in the task record's result section"; it was not, in
either commit, and an independent review called that what it was — a
reproducibility gap. The mutation *names* are not the measurement; these exact
substitutions are.

Method, unchanged from experiments 0002–0004: each mutation is a single edit
that changes behaviour and still compiles. For each one — apply, build, run
every lane, record which lanes caught it, revert.

Added here, because task 0024's contract promised it and the first run did not
do it: **every verdict is repeated**. Task 0022's record is the reason — a
substitution read once from a flaky test is a coin flip recorded as a
measurement. The control is run REPEATS times against the unmutated tree and
the mutant REPEATS times against the mutated one, on the first lane that
catches, and a lane that does not give the same answer every time is reported
as nondeterministic rather than counted.

Usage:  python3 docs/evidence/experiments/drivers/0005-mutations.py [name ...]

It edits tracked source in place and restores it in a `finally`. Run it on a
clean tree; `git status` afterwards is part of the evidence.
"""

import collections
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
CT = os.path.join(ROOT, "crates/moxie-format/src/compressed_tensors.rs")
AF = os.path.join(ROOT, "crates/moxie-format/src/affine.rs")
BF = os.path.join(ROOT, "crates/moxie-format/src/bf16.rs")
IMP = os.path.join(ROOT, "crates/moxie-storage/tests/asymmetric_int4_import.rs")

REPEATS = 3

LANES = {
    "unit": ["cargo", "test", "-p", "moxie-format", "--lib", "--offline", "--locked"],
    "allocfail": ["cargo", "test", "-p", "moxie-format", "--offline", "--locked",
                  "--test", "import_allocation_failure"],
    "alloccount": ["cargo", "test", "-p", "moxie-format", "--offline", "--locked",
                   "--test", "import_allocation_asymmetric"],
    "artifact": ["cargo", "test", "-p", "moxie-storage", "--offline", "--locked",
                 "--test", "asymmetric_int4_import"],
    "symmetric": ["cargo", "test", "-p", "moxie-storage", "--offline", "--locked",
                  "--test", "gemma4_import"],
}

# name -> (file, old, new). Every `old` must occur exactly once.
MUTATIONS = [
    # --- the decode itself -------------------------------------------------
    ("zp-sign-added-not-subtracted", AF,
     "let w = (q - z) as f32 * scale;",
     "let w = (q + z) as f32 * scale;"),
    ("zp-lane-pinned-to-zero", CT,
     "let lane = (o % per_word) as u32;\n        for g in 0..groups_per_row {",
     "let lane = 0u32;\n        for g in 0..groups_per_row {"),
    ("zp-lane-reversed", CT,
     "let lane = (o % per_word) as u32;\n        for g in 0..groups_per_row {",
     "let lane = (per_word - 1 - o % per_word) as u32;\n        for g in 0..groups_per_row {"),
    ("zp-word-row-is-block-major", CT,
     "let row = o / per_word;\n        let lane = (o % per_word) as u32;",
     "let row = o % zp_rows;\n        let lane = (o / zp_rows) as u32;"),
    ("zp-rebias-dropped", CT,
     "values.push(rebias_zero_point(spec.width, raw)?);",
     "values.push(raw as i16);"),
    ("zero-points-dropped-entirely", CT,
     "    Ok(ZeroPoints::PerGroup(values))",
     "    let _ = values;\n    Ok(ZeroPoints::Symmetric)"),
    ("zp-reads-the-padding-lanes", CT,
     "    for o in 0..out_features {\n        let row = o / per_word;",
     "    for o in 0..zp_rows * per_word {\n        let row = o / per_word;"),
    # --- the validation ----------------------------------------------------
    ("zp-shape-check-deleted", CT,
     "    if zp.shape != want {",
     "    if false && zp.shape != want {"),
    ("zp-length-check-deleted", CT,
     "    if zp.payload.len() != need {",
     "    if false && zp.payload.len() != need {"),
    ("spec-payload-disagreement-ignored", CT,
     """        (ZeroPointSource::Symmetric, true) => {
            return Err(invalid_static(
                "a symmetric pack-quantized source carries no weight_zero_point, but one was \\
                 supplied",
            ));
        }""",
     """        (ZeroPointSource::Symmetric, true) => {}"""),
    ("spec-payload-missing-ignored", CT,
     """        (ZeroPointSource::PackedAlongOutput, false) => {
            return Err(invalid_static(
                "an asymmetric pack-quantized source requires its weight_zero_point payload",
            ));
        }""",
     """        (ZeroPointSource::PackedAlongOutput, false) => {}"""),
    ("index-disagreement-symmetric-ignored", CT,
     """        (ZeroPointSource::Symmetric, Some(_)) => {
            return Err(invalid(format_args!(
                "{module} is declared symmetric but serializes {name}; the config and the \\
                 tensor index disagree about this module's zero points"
            )));
        }""",
     """        (ZeroPointSource::Symmetric, Some(_)) => None,"""),
    ("index-missing-zero-point-ignored", CT,
     """        (ZeroPointSource::PackedAlongOutput, None) => {
            return Err(invalid(format_args!(
                "{module} is declared asymmetric but has no {name}"
            )));
        }""",
     """        (ZeroPointSource::PackedAlongOutput, None) => None,"""),
    ("zero-point-dtype-unchecked", CT,
     """            if e.dtype != Dtype::I32 {
                return Err(invalid(format_args!(
                    "{name} is {}; a packed zero point is I32",
                    e.dtype.name()
                )));
            }""",
     """            if false {
                return Err(invalid_static("unreachable"));
            }"""),
    # --- allocation discipline --------------------------------------------
    ("zp-vec-allocated-infallibly", CT,
     "let mut values = crate::try_vec::<i16>(entries)?;",
     "let mut values = Vec::<i16>::with_capacity(entries);"),
    # Added after the review's P1: the refusal prose itself must not abort.
    ("refusal-prose-allocates-infallibly", CT,
     '''fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed compressed-tensors pack-quantized source (detail unavailable: out of memory)",
        detail,
    )
}''',
     '''fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    Error::InvalidArtifact {
        detail: std::borrow::Cow::Owned(std::fmt::format(detail)),
    }
}'''),
    # --- the source's own rounding boundary (the review's finding 2) -------
    ("bf16-rounding-truncates", BF,
     "pub fn f32_to_bf16_bits(v: f32) -> u16 {",
     "pub fn f32_to_bf16_bits(v: f32) -> u16 {\n    return f32_to_bf16_bits_truncating(v);\n    #[allow(unreachable_code)]"),
    ("boundary-count-never-increments", IMP,
     "                        narrowed += 1;",
     "                        let _ = &mut narrowed;"),
    # --- the inventory audit (the review's finding 3) ----------------------
    ("missing-companions-always-empty", IMP,
     """    SUFFIXES
        .iter()
        .copied()
        .filter(|s| !located(&format!("{module}.{s}")))
        .collect()""",
     """    let _ = (module, located);
    Vec::new()"""),
    # --- the measurement itself, not the product ---------------------------
    ("measurement-pinned-becomes-lane-reversed", IMP,
     """                (
                    "pinned o = 8j + l",
                    Box::new(|o: usize, g: usize| {
                        ((word(o / 8, g) >> (4 * (o % 8) as u32)) & 0xF) as i32 - 8
                    }),
                ),""",
     """                (
                    "pinned o = 8j + l",
                    Box::new(|o: usize, g: usize| {
                        ((word(o / 8, g) >> (4 * (7 - o % 8) as u32)) & 0xF) as i32 - 8
                    }),
                ),"""),
    # --- the second review's findings ------------------------------------
    # Its P1: the lookup names on the way to a refusal, not the refusal.
    ("source-entries-names-allocate-infallibly", CT,
     '''    let packed = header.get(&crate::join_name(module, "weight_packed")?)?;
    let scale = header.get(&crate::join_name(module, "weight_scale")?)?;
    let shape = header.get(&crate::join_name(module, "weight_shape")?)?;''',
     '''    let packed = header.get(&format!("{module}.weight_packed"))?;
    let scale = header.get(&format!("{module}.weight_scale"))?;
    let shape = header.get(&format!("{module}.weight_shape"))?;'''),
    ("source-entries-zero-point-name-allocates-infallibly", CT,
     '    let name = crate::join_name(module, "weight_zero_point")?;',
     '    let name = format!("{module}.weight_zero_point");'),
    # Its finding 2: the original population filter, restored.
    ("inventory-filters-incomplete-modules", IMP,
     "        let mut modules: Vec<(String, u64)> = sizes.into_iter().collect();",
     '''        let mut modules: Vec<(String, u64)> = sizes
            .into_iter()
            .filter(|(module, _)| {
                SUFFIXES
                    .iter()
                    .all(|s| located.contains_key(&format!("{module}.{s}")))
            })
            .collect();'''),
    # Added after the review's finding 5: the sign candidate is load-bearing.
    ("measurement-sign-candidate-equals-pinned", IMP,
     """                (
                    "sign-flipped z -> -z",
                    Box::new(|o: usize, g: usize| {
                        -(((word(o / 8, g) >> (4 * (o % 8) as u32)) & 0xF) as i32 - 8)
                    }),
                ),""",
     """                (
                    "sign-flipped z -> -z",
                    Box::new(|o: usize, g: usize| {
                        ((word(o / 8, g) >> (4 * (o % 8) as u32)) & 0xF) as i32 - 8
                    }),
                ),"""),
]


def lane(name):
    r = subprocess.run(LANES[name], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0


def build():
    for extra in (["-p", "moxie-format"], ["-p", "moxie-storage", "--tests"]):
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


CAUGHT = "caught"
SURVIVOR = "survivor"
UNSTABLE = "unstable"
INVALID_CONTROL = "invalid-control"


def classify(caught_by, mutant_ok, control_ok):
    """What a mutation's run actually established.

    The second review's finding: the first version appended unstable and
    failing-control verdicts to `nondet` **and then counted them as caught**,
    so a run whose restored control never passed printed "1 of 1 caught". A
    mutation is caught only when the mutant fails the lane every time and the
    restored tree passes it every time; anything else is an outcome to report,
    not a number to add.

    `mutant_ok` / `control_ok` are `True` (stably as expected), `False` (stably
    the opposite) or `None` (the lane disagreed with itself).
    """
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
    """Deterministic checks of the verdict rule, run with `--self-test`.

    The rule is what the battery's headline number means, so it is tested
    rather than read.
    """
    cases = [
        (([], True, True), SURVIVOR),
        (([], None, None), SURVIVOR),
        ((["unit"], True, True), CAUGHT),
        ((["unit"], None, True), UNSTABLE),     # mutant repeats disagreed
        ((["unit"], True, None), UNSTABLE),     # control repeats disagreed
        ((["unit"], False, True), UNSTABLE),    # failed once, then passed
        ((["unit"], True, False), INVALID_CONTROL),  # restored tree still fails
        ((["unit"], False, False), UNSTABLE),
    ]
    bad = [(a, want, classify(*a)) for a, want in cases if classify(*a) != want]
    for a, want, got in bad:
        print(f"SELF-TEST FAIL {a}: want {want}, got {got}")

    # Selector validation, the third review's P3. A mistyped name must not
    # quietly become an empty battery that exits successfully.
    fake = [("a", None, None, None), ("b", None, None, None)]
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
    """Split requested names into (chosen, unknown), preserving battery order.

    The third review's P3: a mistyped selector ran nothing and exited 0 with
    "0 of 0 caught". A battery that silently omits the work it was asked for
    reports a number about a different battery.
    """
    known = [name for name, _, _, _ in mutations]
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
    for name, path, old, new in battery:
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
            if caught_by:
                mutant_ok = repeated(caught_by[0], False)
        finally:
            open(path, "w").write(src)
        control_ok = None
        if caught_by:
            if not build():
                skipped.append((name, "the tree did not rebuild after restore"))
                print(f"SKIP {name}: no rebuild after restore", flush=True)
                continue
            control_ok = repeated(caught_by[0], True)
        verdict = classify(caught_by, mutant_ok, control_ok)
        outcomes[name] = (verdict, caught_by)
        detail = ",".join(caught_by) if caught_by else "-"
        print(f"{name:52s} {verdict:15s} {detail}", flush=True)

    counts = collections.Counter(v for v, _ in outcomes.values())
    total = len(outcomes) + len(skipped)
    print(f"\n{counts[CAUGHT]} of {total} caught, "
          f"{counts[SURVIVOR]} survivor(s), {counts[UNSTABLE]} unstable, "
          f"{counts[INVALID_CONTROL]} invalid control(s), {len(skipped)} skipped; "
          f"{REPEATS} repetition(s) of each verdict in both directions")
    for name, (verdict, _) in outcomes.items():
        if verdict != CAUGHT:
            print(f"  {verdict.upper()} {name}")
    for name, why in skipped:
        print(f"  SKIPPED {name}: {why}")
    # An incomplete or invalid battery is not a measurement.
    return 0 if counts[CAUGHT] == total else 1


if __name__ == "__main__":
    sys.exit(main())
