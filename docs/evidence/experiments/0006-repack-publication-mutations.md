# 0006 — Can the repacker's gates fail? Measured by mutation, with an independence control

Date opened: 2026-09-13. Milestone: M3 item 1,
[task 0025](../../tasks/0025-m3-offline-repack-publication.md), then
[task 0026](../../tasks/0026-m3-canonical-safetensors-publication.md), then
[task 0030](../../tasks/0030-m3-repack-correctness-package.md).
Status: **measured three times; last re-run 2026-09-16 on task 0030's tree.**
The run that describes the current tree is the last section, not the first:
the `ccfd7fa` figures below are kept because the differences between the
runs are the evidence, and **none of them describes this tree**.

## Why this exists

Task 0025's acceptance asks for gates that "mutation-measure ... omitted
checksum/source/version checks, premature manifest publish, false journal
completion, missing sync/error handling, budget bypass and cancellation
ordering", and adds the sentence that shapes this record:

> Fixtures must check each axis's effect, not just count visits.

A publication state machine is exactly the kind of code whose tests can all pass
while checking nothing: every checksum it compares it also computed, every
journal entry it trusts it also wrote, and a resume that recomputes everything
produces the right answer whatever the resume logic did. The battery is what
separates "the gates pass" from "the gates can fail".

It is a measurement of the **tests**, not of the artifact. Nothing here executes
a canonical tensor, and a repack is a statement about bytes rather than about
model output ([ADR 0018](../../decisions/adr/0018-v1-quality-is-bit-identical-repack.md)).

## Method

Unchanged from experiments 0002–0005: each mutation is a single edit that
changes behaviour and still compiles. Apply, build, run every lane, record which
lanes caught it, revert. Every verdict is repeated three times in both
directions — the mutant against the mutated tree and the restored control
against the clean one — and a lane that disagrees with itself is reported as
unstable rather than counted. Only `caught` counts.

The driver is committed, at
[`cargo xtask mutation-check`](../../../xtask/src/mutationcheck.rs), and its
verdict rule is self-tested (`--self-test`): **65 of 65** cases — ten of the
verdict rule, five of the selector, because a mistyped selector that silently
runs nothing would report a number about a different battery, and **fifty
anchor checks**, because a substitution that no longer matches its line is a
gate nobody measured.

It became an `xtask` command on 2026-09-14, from a Python script under `docs/`.
The owner's question was the right one — tooling belongs in the tooling crate —
and the port paid for itself immediately: the anchor half of the self-test,
which the Python driver only performed while running, found that **thirteen of
experiment 0005's twenty-four anchors had matched nothing since before
`ccfd7fa`**. That battery was retired rather than carried; [its record](0005-asymmetric-int4-zero-point-assignment.md)
says so.

The substitution table is now type-checked, the driver is covered by `fmt`,
`clippy` and the workspace suite like any other code here, and a run stopped by
a signal — including `SIGKILL`, which no handler can catch — leaves a marker
that the next run finds and restores. That last one is not hypothetical: a
battery killed mid-substitution left `if false && got != unit.sha256` in the
resume path of this repository's writer, and the gates were re-run against it
before anyone noticed.

### What is new here: the independence control

Task 0025's acceptance also asks, of the architecture rule, for "substitutions
that remove each protection", noting that "a fixture rejected for an unrelated
rule is not evidence this rule works". That is a real hazard for the new
write-authority rule: five of its six negative fixtures would *also* be rejected
by the dependency allowlist, so the fixtures alone say nothing about the rule.

So a mutation may declare `expect=SURVIVOR` — an **independence control**, where
removing some *other* protection must leave the battery green because the
protection under test still holds. A control that is caught means the two
protections are entangled; it is reported as `control-broken`, separately from
survivors, because it is a different fact.

## Lanes

Twelve, as the battery runs them. The table drifted from the driver twice; it is
now written from `LANES_T0006` rather than from memory.

| Lane | Command |
|---|---|
| `format` | `cargo test -p moxie-format --lib` |
| `manifest` | `cargo test -p moxie-format --test manifest_v1` |
| `publication` | `cargo test -p moxie-repack --test publication` |
| `workflow` | `cargo test -p moxie-repack --test workflow` (added 2026-09-14) |
| `roundtrip` | `cargo test -p moxie-repack --test round_trip` |
| `cli` | `cargo test -p moxie-repack --test cli` |
| `budget` | `cargo test -p moxie-repack --test budget` |
| `malformed` | `cargo test -p moxie-repack --test malformed` |
| `round2` | `cargo test -p moxie-repack --test round2` |
| `admission` | `cargo test -p moxie-repack --test admission` |
| `storage` | `cargo test -p moxie-storage --test artifact` |
| `arch` | `cargo xtask arch-check` |

The `budget` lane ran with `--test-threads=1` from 2026-09-14 until task 0030
took it off again. Why it was there, and why taking it off is a repair rather
than a relaxation, is the [task 0030 section](#re-run-for-task-0030-on-the-repaired-budget-lane-2026-09-16).

The real-artifact lane is deliberately **not** in the battery: it needs a
checkpoint this machine happens to have, and thirty seconds of SHA-256 per
mutation would buy coverage the synthetic fixtures already provide. What it
proves — that the published bytes reconstruct to the source's values — is
proved once, in the task record.

## Result

**29 of 29 mutant(s) caught, 4 of 4 expected survivor(s) held (independence controls and equivalent mutants); 0 survivor(s), 0 unstable, 0 invalid control(s), 0 broken control(s), 0 skipped; 3 repetition(s) of each verdict in both directions**

| Axis | Mutation | Verdict | Caught by |
|---|---|---|---|
| Checksums: what the gates compare | `unit-checksum-not-compared` | caught | `publication` |
|  | `tensor-hash-never-sees-the-bytes` | caught | `publication`, `roundtrip`, `cli`, `budget` |
|  | `published-validation-skipped` | caught | `publication` |
|  | `source-digest-is-a-constant` | caught | `cli` |
|  | `unit-source-digest-is-recorded-not-checked` | control-held | — |
| Source, plan and version binding | `resume-ignores-its-binding` | caught | `publication`, `cli` |
|  | `run-binding-omits-the-source-digests` | caught | `cli` |
|  | `journal-version-not-checked` | caught | `format` |
|  | `selection-version-not-checked` | caught | `format` |
|  | `manifest-schema-version-not-checked` | caught | `manifest` |
| Premature publication and false completion | `manifest-staged-under-its-final-name` | caught | `publication` |
|  | `incomplete-tensors-can-be-sealed` | caught | `publication` |
|  | `journal-recorded-before-the-bytes-are-durable` | caught | `publication` |
|  | `resume-trusts-the-journals-offsets` | control-held | — |
| Syncs and error handling | `chunk-sync-skipped` | caught | `publication`, `cli` |
|  | `directory-sync-skipped` | caught | `publication` |
|  | `write-errors-are-swallowed` | caught | `publication` |
|  | `journal-sync-skipped` | caught | `publication` |
| Budgets | `unit-size-not-checked-against-the-scratch` | caught | `publication` |
|  | `disk-budget-not-checked-while-writing` | control-held | — |
|  | `chunk-file-limit-ignored-by-the-plan` | caught | `cli` |
|  | `plan-ignores-the-disk-budget` | caught | `cli` |
|  | `tile-is-the-whole-scratch` | caught | `roundtrip`, `budget` |
|  | `header-budget-flag-ignored` | caught | `cli` |
| Cancellation ordering | `cancellation-not-checked-between-units` | caught | `budget` |
|  | `cancellation-not-checked-before-publish` | caught | `publication` |
| The canonical payload | `payload-sections-in-the-wrong-order` | caught | `format`, `manifest`, `roundtrip`, `cli` |
|  | `zero-point-section-dropped` | caught | `format`, `manifest`, `roundtrip`, `cli`, `budget` |
|  | `scale-block-not-validated` | caught | `format` |
|  | `affine-length-rule-deleted` | caught | `manifest` |
| The architecture rule, and its independence from the allowlist | `write-authority-rule-disabled` | caught | `arch` |
|  | `write-authority-consumers-widened` | caught | `arch` |
|  | `allowlist-permits-the-writer-in-the-engine` | control-held | — |

### What the first run found, and why that is the interesting part

The battery was run twice. The **first** run, against the tree as it stood when
the implementation was finished, caught 24 of 28 mutants and held 3 of 3
controls — and left **four survivors and two skips**, each of which was a real
gap rather than a scoring accident:

| Survivor | What it exposed |
|---|---|
| `write-errors-are-swallowed` | Ignoring the error from a payload write survived **every** gate, because the checksum and the resume machinery repair the damage on the next attempt. Repairing a failure is not reporting it, and a run that silently redoes work it was told had failed is one nobody can debug. A test now requires an injected write failure to fail the run **naming the boundary it happened at**. |
| `plan-ignores-the-disk-budget` | The CLI case that was supposed to cover it set the chunk-file limit and the disk budget low at the same time, and the chunk-file rule fired first. Two rules that can both fire need a case each; the new one sets the chunk-file limit large and the disk budget one byte short, with the arithmetic written out. |
| `cancellation-not-checked-before-publish` | With the per-unit check and the pre-staging check both in place, removing the check *between validation and the rename* changed nothing any test could see — and that is the check that matters most, because it is the last point at which stopping is still free. A test now cancels during validation, not before it. |
| `disk-budget-not-checked-while-writing` | **Equivalent, and the argument is written down**: `OutputPlan::build` refuses a plan whose chunk lengths exceed the disk budget, and every `write_unit` is bounded by its tensor's planned range, so the runtime counter can never exceed the planned total. The check stays as defence against a future disagreement between the plan and the run; it is now declared an expected survivor rather than counted as a caught one. |

The two skips were the battery failing to measure rather than the product
failing: one anchor had been reformatted by `cargo fmt` and the other matched
two places in one file — `if need != length {` is also the BF16 byte-count
rule. A mutation that does not apply is not evidence, which is why the driver
reports skips separately and refuses to exit zero with any.

**Three tests and two anchors later, the second run is the measurement above.**

## Re-run for task 0026, against the safetensors writer (2026-09-14)

Task 0026 replaced the container the publication path writes, so this battery
was re-run against it. **It did not agree with itself on the first attempt, and
that is the whole value of re-running it.**

| What the re-run said | What it was |
|---|---|
| 9 mutations `SKIPPED`, "anchor occurs 0 time(s)" | The rewrite moved the code under them: `t.request.length` became `c.len`, `planned.chunk` became `planned.file`, `file_digest` gained a cancellation argument, one schema version became a supported set, and the affine length rule became a `Placement::Chunk` guard. Re-anchored. **A mutation that does not apply is not evidence**, which is why the driver refuses to exit zero with any. |
| `chunk-sync-skipped` and `write-errors-are-swallowed` survived | The safetensors header pass reused `chunk-create`, `chunk-write` and `chunk-sync`, so every fault injected for a payload unit fired at a *header* first and the suites kept failing for an unrelated reason. The header pass now owns `shard-create`, `shard-header-write` and `shard-header-sync`, and `write_at` takes its boundary as a parameter. |
| `cancellation-not-checked-before-publish` survived | Task 0025's finding-9 correction added a cancellation check inside the validation loop, and the existing test's closure stopped there rather than at the last gate before the rename. A new test answers *no* to every boundary but the last, and asserts how many there are. |
| The `unit-checksum-not-compared` scare before all this | The corruption fixtures flipped `bytes[3]`, which after the rewrite lands in the shard header that every `begin` rewrites. They now compute `payload_start = 8 + header_len`. |

Adding the `workflow` lane then produced 7 invalid controls and 3 broken
controls at once -- the driver's way of saying **a lane is red on the clean
tree**. It was: `every_injected_failure_leaves_the_ledger_empty` had been
failing since `d719f67`, because `Run::begin` admits the read-back scratch and
the new header pass returned with a bare `?`, leaking it. That is review finding
7 at an error path that postdates finding 7's fix.
`header-failure-leaks-the-admission` is now a mutation, and `workflow` is the
only lane that catches it.

**Result on the corrected tree: 30 of 30 mutant(s) caught, 3 of 3 expected
survivor(s) held; 0 survivor(s), 0 unstable, 0 invalid control(s), 0 broken
control(s), 0 skipped; 3 repetition(s) of each verdict in both directions.**

| Axis | Mutation | Verdict | Caught by |
|---|---|---|---|
| Checksums | `unit-checksum-not-compared` | caught | `publication` |
|  | `tensor-hash-never-sees-the-bytes` | caught | `publication`, `roundtrip`, `cli`, `budget` |
|  | `published-validation-skipped` | caught | `publication` |
|  | `source-digest-is-a-constant` | caught | `roundtrip`, `cli`, `budget` |
|  | `unit-source-digest-is-recorded-not-checked` | control-held | — |
| Source, plan and version binding | `resume-ignores-its-binding` | caught | `publication`, `cli` |
|  | `run-binding-omits-the-source-digests` | caught | `cli` |
|  | `journal-version-not-checked` | caught | `format` |
|  | `selection-version-not-checked` | caught | `format` |
|  | `manifest-schema-version-not-checked` | caught | `manifest` |
| Premature publication and false completion | `manifest-staged-under-its-final-name` | caught | `publication`, `roundtrip`, `cli`, `budget` |
|  | `incomplete-tensors-can-be-sealed` | caught | `publication` |
|  | `journal-recorded-before-the-bytes-are-durable` | caught | `publication` |
|  | `resume-trusts-the-journals-offsets` | control-held | — |
| Syncs, error handling and the admission | `chunk-sync-skipped` | caught | `publication`, `cli` |
|  | `directory-sync-skipped` | caught | `publication` |
|  | `write-errors-are-swallowed` | caught | `publication` |
|  | `header-failure-leaks-the-admission` | caught | `workflow` |
|  | `shard-header-sync-skipped` | caught | `publication` |
|  | `shard-header-never-written` | caught | `publication`, `roundtrip`, `cli`, `budget` |
|  | `journal-sync-skipped` | caught | `publication` |
| Budgets | `unit-size-not-checked-against-the-scratch` | caught | `publication` |
|  | `disk-budget-not-checked-while-writing` | control-held | — |
|  | `chunk-file-limit-ignored-by-the-plan` | caught | `cli` |
|  | `plan-ignores-the-disk-budget` | caught | `cli` |
|  | `tile-is-the-whole-scratch` | caught | `workflow`, `roundtrip`, `budget` |
|  | `header-budget-flag-ignored` | caught | `cli` |
| Cancellation ordering | `cancellation-not-checked-between-units` | caught | `cli`, `budget` |
|  | `cancellation-not-checked-before-publish` | caught | `publication` |
| The canonical payload | `payload-sections-in-the-wrong-order` | caught | `format`, `manifest`, `roundtrip` |
|  | `zero-point-section-dropped` | caught | `format`, `manifest`, `roundtrip`, `cli`, `budget` |
|  | `scale-block-not-validated` | caught | `format` |
|  | `affine-length-rule-deleted` | caught | `manifest` |

The three architecture-rule mutations from the first run are gone with the rule
they policed: [ADR 0024](../../decisions/adr/0024-one-storage-crate-and-a-write-module.md)
folded the writer into the only program that writes, and nothing outside a
program can name a module inside it. The module boundary is what replaced the
rule, and no substitution in the driver can weaken it.

## What the battery does not establish

- **It measures the tests, not the artifact.** A caught mutation says a gate can
  fail; it says nothing about whether the gate is checking the right thing. The
  expected-bytes fixtures are what address that, and they are built from the
  values the test wrote rather than from the encoder.
- **It is not exhaustive.** Thirty-odd single edits over a state machine with
  twenty named boundaries is a sample. The enumeration in
  `publication.rs` is the systematic half; this is the adversarial one.
- **No quality, execution or performance claim follows.** Nothing here runs a
  model, a kernel or a benchmark.
