# 0006 — Can the repacker's gates fail? Measured by mutation, with an independence control

Date: 2026-09-13. Milestone: M3 item 1,
[task 0025](../../tasks/0025-m3-offline-repack-publication.md).
Status: **measured; the task is implemented and not reviewed.**

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
[drivers/0006-mutations.py](drivers/0006-mutations.py), and its verdict rule is
self-tested (`--self-test`): **15 of 15** cases, ten of the verdict rule and five
of the selector, because a mistyped selector that silently runs nothing would
report a number about a different battery.

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

| Lane | Command |
|---|---|
| `format` | `cargo test -p moxie-format --lib` |
| `manifest` | `cargo test -p moxie-format --test manifest_v1` |
| `publication` | `cargo test -p moxie-storage-write` |
| `roundtrip` | `cargo test -p moxie-repack --test round_trip` |
| `cli` | `cargo test -p moxie-repack --test cli` |
| `budget` | `cargo test -p moxie-repack --test budget` |
| `arch` | `cargo xtask arch-check` |

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

## What the battery does not establish

- **It measures the tests, not the artifact.** A caught mutation says a gate can
  fail; it says nothing about whether the gate is checking the right thing. The
  expected-bytes fixtures are what address that, and they are built from the
  values the test wrote rather than from the encoder.
- **It is not exhaustive.** Thirty-odd single edits over a state machine with
  seventeen named boundaries is a sample. The enumeration in
  `publication.rs` is the systematic half; this is the adversarial one.
- **No quality, execution or performance claim follows.** Nothing here runs a
  model, a kernel or a benchmark.
