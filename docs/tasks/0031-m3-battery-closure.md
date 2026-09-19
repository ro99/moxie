# Task 0031 — M3.1 battery closure: validation survivor, full T0006, T0028 regression

Status: **accepted by the owner as part of M3 closure, 2026-09-19**; final T0006
and T0028 passed on the same candidate.
The 2026-09-16 correction below supersedes the inherited acceptance premise and
blocked CUDA status; the historical Result remains as provenance.

## Identity and authority

- Task ID 0031; **M3 item 1 battery closure**. Owner/reviewer: the repository owner under the existing review process.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `4ccb3b9`, tree clean at authoring.
- Read-only: `/models` and `/fast/models`; legacy root `/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: task 0030 acceptance 3 (full T0006 on the final tree) and acceptance 8 (T0028 regression). Carries finding 1: `published-validation-skipped` survived on the 0030 tree where experiment 0006 records it caught at `ccfd7fa`.
- Required reading: [task 0030](0030-m3-repack-correctness-package.md) (Result, findings 1 and 5, acceptance 3/8), [experiment 0006](../evidence/experiments/0006-repack-publication-mutations.md) (task 0030 section), `xtask/src/mutationtable.rs` (`published-validation-skipped` at line 54), `crates/moxie-repack/src/write/run.rs` (`publish_inner` validation at lines 1000–1030), `crates/moxie-repack/tests/publication.rs` (`run_to_end`, enumeration), `docs/spec/09-agent-playbooks.md` §E.
- Owner gates already resolved: O1/O2/O3/O4/O5. O6/O7 stay open: no timing here is a claim. Experiment 0007 stays pending.

## Bounded deliverable

- **One concrete outcome**: the survivor is repaired with a regression that fails without it, full T0006 runs end to end on the final tree with 0 survivors, then full T0028 runs as shared-code regression — all recorded in this Result and experiment 0006.
- **Sole owning shared component**: `moxie-repack` plus its `write` module. Battery table edits confined to the survivor's anchor.
- **Allowed production files**: `crates/moxie-repack/src/write/run.rs` (validation path only), `crates/moxie-repack/src/write/fault.rs` only if the `Validate` fault point needs repair.
- **Allowed test files**: `crates/moxie-repack/tests/publication.rs` (one regression for the survivor), `xtask/src/mutationtable.rs` (re-anchor only if the repair moves the anchor), plus this record, experiment 0006, support matrix `G-REPACK`, engineering log.
- **Explicit non-goals**: retention decision; whole-model workflow revival; expert MoE; any engine/kernel/attention/sampler change; any bulk conversion, download, or write under checkpoint roots; any performance claim; the `Label`/E0521 owner decision (test-gpu stays blocked until it is made).
- **Forbidden shortcuts**: counting the stopped 10-of-66 pass as evidence; relabelling `ccfd7fa` figures; weakening any assertion; editing reference docs 01–09.
- **Second proof**: reference safetensors 0.7.0 stays the independent reader; the Laguna real-module lane stays the second read path.
- **Temporary paths**: parked mutation originals cleared; tree clean afterwards.

## Contract before implementation

- **Mechanism**: `publish_inner` validates the staged artifact through the production reader (`Artifact::open_unpublished` plus `verify_tensor_cancellable` per sealed role) before the manifest rename. The mutant removes the `Site::Validate` fault check and empties the role list (`take(0)`), so no payload byte is verified. A gate that stays green under that substitution is not guarding validation.
- **Repair shape**: a regression that corrupts staged payload bytes (or removes the validation loop's effect) and requires publication to fail — failing on the current tree, passing with the fix. Mutation, not assertion-only.
- **Predeclared thresholds**: T0006 66 substitutions over 12 lanes, all caught, expected survivors held, 0 unstable, 0 skipped; T0028 full battery caught with survivors held; enumeration still 48/48; suites unchanged except the new regression.

## Acceptance

1. Regression for `published-validation-skipped` fails on the current tree and passes with the fix; the mutant is caught by the `publication` lane.
2. Full `cargo xtask mutation-check --battery 0006` end to end on the final tree: 66 substitutions, all caught, expected survivors held, 0 unstable, 0 skipped, tree clean and no parked original afterwards.
3. Full `cargo xtask mutation-check --battery 0028` end to end on the final tree as shared-code regression; result recorded here, not as publication evidence.
4. `cargo xtask-cuda test-gpu` recorded as blocked (E0521) or run if the owner decision lands elsewhere; never counted as passing.
5. Gates: fmt; host clippy plus driver clippy lane; spec-check; arch-check; self-test 109/109; host plus device-feature suites with passed/failed/skipped separately.
6. Records: this Result, experiment 0006 (new run section, old figures kept), `G-REPACK` figures from this tree, engineering-log shape entry.
7. Exact condition requiring owner direction: any tolerance change; any retention claim; accepting the survivor as an expected survivor (would need the equivalence argument written down, and this contract predeclares it must be caught, not held).

## Recovery correction — 2026-09-16

Owner's instruction to implement the M3 recovery authorizes this correction.
Acceptance 1 confused a missing test with broken production code. Its corrected
criterion is: the new staged-corruption regression passes on production and
fails under `published-validation-skipped`; the unchanged production reader must
refuse corrupted staged payloads. Introducing a production defect merely to
obtain a red test is forbidden. The earlier targeted mutation supplies that
contrast; the full final-tree battery remains required. This corrects a false
premise and does not weaken the publication guarantee.

Task 0032 restores borrowed-label input and the CUDA build. Its
`cargo xtask-cuda test-gpu` run passed 45 cases, zero failed or skipped, on SM86
and SM120. That replaces the blocked alternative in acceptance 4. It does not
replace either mutation battery. Tasks 0031/0032 are now being executed directly
by the owner-assigned recovery agent; no coordinator order is pending.

## Historical result before the final recovery runs

Status at that checkpoint: **implemented and self-measured; batteries not run,
by the coordinator's instruction.** The survivor is closed and the mutation that
survived task 0030's battery is caught. Acceptance 1 is **partially met** with
its literal before/after clause open; 4 is satisfied via its blocked
alternative; **2, 3, 5 and 6 are outstanding**.
One acceptance premise turned out to be wrong, and a second defect was found,
raised, and repaired under a coordinator ruling — both below.

Tree: base `4ccb3b9`, branch `main`. Product files changed: **none** —
`crates/moxie-repack/src/write/run.rs` and `fault.rs` are untouched, and the
reason is acceptance 1 below. Test files changed:
`crates/moxie-repack/tests/publication.rs` — one test added, one assertion
tightened.

### Acceptance 1 — the survivor's regression: **partially met, literal clause open**

**The predeclared before/after criterion is not met as written, and this record
does not claim it is.** Acceptance 1 asks for a regression that "fails on the
current tree and passes with the fix". It does not fail on the current tree. It
passes on unchanged production code, first run, unmodified. The clause is
**open**, and no wording here converts a passing test into the failing one the
contract predicted.

| Acceptance 1's clause | State |
|---|---|
| Regression **fails** on the current tree | **not met** — it passes on unchanged production code |
| Regression **passes with the fix** | vacuous: there is no fix, because there is no defect |
| Mutant **caught by the `publication` lane** | **met** — measured, and recorded below as the discrimination evidence |

**Why the first clause cannot be met without manufacturing a defect.** The
clause presumes `publish_inner`'s validation pass is broken. It is not: the
production path is **sound but untested**. The regression was written exactly to
the contract's repair shape — corrupt staged payload bytes, require publication
to fail — and the path refused the artifact the first time it was asked to. The
only way to make the regression fail first would have been to damage `run.rs`
and then repair it, which is a defect invented to satisfy a sentence. That is
the "manufacture completion with a private model path" AGENTS.md forbids, and
the reason `crates/moxie-repack/src/write/run.rs` and `fault.rs` are untouched
and absent from this task's diff.

**What stands in for the before/after in this case.** A before/after on
production code is one way to show a test discriminates; a mutation is the
other, and it is the one this repository already trusts. The targeted run of
`published-validation-skipped` is therefore the discrimination evidence: the
substitution that **survived** task 0030's full battery is **caught** by the
`publication` lane with this regression present. Same lane, same substitution,
opposite verdict — which is the fact acceptance 1 was reaching for.

What was actually wrong is narrower and worth stating precisely: **the gate was
never given the input it exists for.** Every other test in `publication.rs`
publishes bytes the writer itself just wrote and hashed, and skipping a read-back
changes nothing when nothing on disk has changed since it was written. A
validation pass can only be shown to work against bytes that went wrong *after*
the writer finished with them, and nothing produced such bytes.

**The regression**, `a_staged_payload_corrupted_after_sealing_is_refused_before_publication`:

* Writes every unit, seals, then flips **one bit** in the staged shard's
  payload, then publishes.
* The window is chosen, not convenient. `seal()` finalises the streaming hasher
  accumulated as the bytes were written — it never re-reads the file — and
  `Site::ChunkReadBack` is reached only on a resume or a rehash. Between `seal`
  and `publish`, the validation loop is the only thing that touches the payload
  on disk. Corrupting anywhere else would be caught by a neighbouring gate and
  would prove nothing about this one.
* **One bit in a mantissa's low byte**, not a whole byte and not the high byte:
  flipping an exponent would also trip the BF16 finiteness check, and the test
  would then pass for a reason unrelated to checksums. That is the "two rules
  that can both fire need a case each" shape experiment 0006 already recorded
  once, avoided here by construction.
* It asserts four things, not one: publication returns an error naming a
  `checksum mismatch`; no `manifest.toml` is exposed; the destination does not
  open as an artifact; and the ledger comes back empty on the failed path.

**Measured, through the driver rather than by hand.** The substitution was run
as `cargo xtask mutation-check --battery 0006 published-validation-skipped`,
which applies the committed substitution itself, takes the clean-tree baseline
over all 12 lanes before and after, and restores the file from a parked original
on exit. A hand-applied edit would have been a second transcription of the
mutation and would not have been the thing the battery runs:

```
published-validation-skipped                         caught          publication

1 of 1 mutant(s) caught, 0 of 0 expected survivor(s) held (independence controls
and equivalent mutants); 0 survivor(s), 0 unstable, 0 invalid control(s),
0 broken control(s), 0 skipped
```

Baseline: `12 lane(s) stably pass, 3 repetition(s) each`, before and after; the
deciding lane repeated 3 times under the mutation. Afterwards `git diff` reported
`crates/moxie-repack/src/write/run.rs` identical to `HEAD` and
`target/mutation-check/` empty — no marker, no parked original.

**The mutation that survived task 0030's battery is now caught, by the
`publication` lane, which is the lane acceptance 1 names.** One substitution is
not the battery, and acceptance 2 is still owed.

### The second defect: a coverage allowance that outlived its gap

The mutation has **two** halves, and the regression above closes one of them.
The other deletes `faults.check(Site::Validate)?`. That half should have been
caught by the enumeration's own unreached-boundary assertion, and it was not.
Measured on this tree:

```
boundaries not reached by this scenario: []
```

against an assertion of `unvisited.len() <= 1`. **Zero unreached, one allowed.**
Deleting any single `faults.check(Site::X)` for a once-visited boundary moves
the count from 0 to 1 and still passes. The allowance was written when one
boundary was genuinely unreachable by the scenario; the scenario later grew a
resume, `chunk-read-back` now has 2 visits, and the allowance was never
tightened behind it. It is stale slack, sized exactly to hide a removed check.

**Raised rather than decided, then repaired under a ruling.** Acceptance 7
names any tolerance change as an owner decision, so this task recorded the
measurement and recommended the change instead of making it. The **coordinator
ruled on 2026-09-16** that tightening it is a compliant strengthening and not a
tolerance weakening, on the evidence that the coverage table measures zero
unreached boundaries and `chunk-read-back` now has 2 visits. It is now
`unvisited.is_empty()`, and the reasoning is written beside the assertion rather
than only here — the next person to read that line gets the history that makes
the allowance's absence deliberate.

What the tightened form costs a future change, stated so it is not a surprise: a
boundary that becomes genuinely unreachable again has to be argued for in
writing rather than absorbed by a spare slot. That is the intended cost.

This is the same shape as the survivor itself: **a gate whose slack nobody
re-measured after the thing it allowed for went away.**

### Historical outstanding ledger before the final recovery runs

| # | Acceptance | State | What is missing |
|---|---|---|---|
| 1 | Survivor regression | **partially met** | the literal "fails on the current tree" clause, open above with its reason |
| 2 | Full `T0006`, 66 substitutions over 12 lanes, 0 survivors | **not run** | the whole run — awaiting the coordinator's order |
| 3 | Full `T0028` as shared-code regression | **not run** | the whole run — same |
| 4 | `test-gpu` run **or recorded as blocked**, never counted as passing | **satisfied, via its blocked alternative** | nothing |
| 5 | Gate set: fmt, host clippy, driver clippy, spec-check, arch-check, self-test, host + device-feature suites, passed/failed/skipped separately | **met** — table below | nothing |
| 6 | Records | **outstanding** | experiment 0006's new run section; `G-REPACK` figures from this tree; the engineering-log entry. This Result is written |

**Acceptance 4, satisfied by the clause's own second branch.** It reads "run if
the owner decision lands elsewhere" **or** "recorded as blocked (E0521) …never
counted as passing". The decision has not landed, so the second branch applies
and the blockage is recorded here in full:

```
error[E0521]: borrowed data escapes outside of function
   --> xtask/src/gpu.rs:595:19
    |
592 |     label: &str,
    |     -----  - let's call the lifetime of this reference `'1`
595 |     let mut req = PlanRequest::new(label, ["run"])?;
    |                   ^^^^ `label` escapes the function body here
    |                   argument requires that `'1` must outlive `'static`
```

It is task 0029's `Label` narrowing, an **open owner decision** on AGENTS.md's
gate list, present at base `0d7fe63` before task 0030 touched anything, and
named a non-goal by this contract. `cargo xtask-cuda test-gpu` is therefore
**unmeasured, which is never passing** — satisfied as a record, not as a run.

### Acceptance 5 — gates, as run on this tree

Host lane, no GPUs, no batteries. Run in the order given.

| # | Command | passed | failed | skipped / ignored |
|---|---|---|---|---|
| 1 | `cargo fmt --all -- --check` | clean (exit 0) | 0 | — |
| 2 | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | clean (exit 0) | 0 | — |
| 3 | `cargo clippy --workspace --all-targets --locked --offline --features moxie-executor/driver -- -D warnings` | clean (exit 0) | 0 | — |
| 4 | `cargo xtask spec-check` | 10 document(s) present and unchanged | 0 | 0 |
| 5 | `cargo xtask arch-check` | 79 rejected fixture(s), 21 accepted, 13 rule(s) | 0 | 0 |
| 6 | `cargo xtask mutation-check --self-test` | 109 of 109 (10 verdict, 5 selector, 94 anchor) | 0 | 0 |
| 7 | `cargo test --workspace --locked --offline` | **1,109**, 101 suites | 0 | 0 ignored, 0 filtered |
| 8 | `cargo test --workspace --features moxie-executor/driver --locked --offline` | **1,154**, 101 suites | 0 | 0 ignored, 0 skipped |

Host is 1,108 on task 0030's tree plus **one**: the corruption regression. The
device-feature figure moves by the same one, for the same reason.

**One run in this set was discarded rather than reported.** The device-feature
suite was first invoked under a 585-second command timeout and was killed at
**81 of 101 suites** with 1,084 tests passed. That is an incomplete run, not a
result, and 1,084 is not a figure about this tree — a partial suite reported as
a total is the same error as a partial battery reported as a measurement. It was
re-run without a timeout, and row 8 is the complete run.

**Not run here, and not counted anywhere:** `cargo xtask-cuda test-gpu`
(E0521, acceptance 4), the full `T0006` battery (acceptance 2) and the full
`T0028` battery (acceptance 3).

**Acceptances 5 and 6 are the remaining work**, and neither can be closed before
acceptance 2: a gate report and a `G-REPACK` figure written before the battery
runs would be a figure about a tree whose battery is unmeasured, which is the
mistake task 0030 spent a round correcting. Until the batteries run, acceptance
1's verdict stands on the single-mutation run above and nothing else, and **a
single-mutation run is not the battery**.


## Recovery battery, 2026-09-17

The original /tmp run was interrupted by a reboot and is not completion evidence.
The test operator monitored the restarted full T0006 run in the persistent
`/home/rodrigo/Developer/moxie-m3-publication-battery` worktree at
`356f9655a8f7ca1add336a04a8dbf72b9b3f1ad1`, detached and clean after the driver.
Logs: `results/m3-recovery/t0006.log`, exit status file `t0006.exit` (exit 1).

Twelve baseline lanes, three repetitions, stable before and after. All 66
substitutions ran: 58/63 caught, 3/3 expected controls held, **five unexpected
survivors**, zero unstable/invalid/broken/skipped. Acceptance2 remains open.
Survivors: journal-recorded-before-the-bytes-are-durable-tail,
source-identity-not-rebound, cancelled-hashing-is-an-error-again,
compaction-peak-not-budgeted, plan-budgets-for-one-journal.
The operator is investigating regression tests on that isolated snapshot;
expectations remain unchanged. This result does not qualify subsequent builder
changes. Targeted checks of the five survivors come before any repeat full run.


## Targeted survivor repairs, 2026-09-17

Five added regressions catch all five survivors in an isolated source snapshot
of 783f91a plus the test patch. Each test passed three baseline repetitions,
failed three repetitions with its exact declared mutant, then passed three
restored repetitions. No expectation changed. Evidence:
`results/m3-recovery/survivor-targeted-summary.log` and `survivor-targeted/`.
This is targeted evidence, not the full 66-substitution battery; acceptance2
remains open until the final-tree battery passes.

The durable-tail mutant adds a duplicate write/sync after journal recording;
it does not delete the original preceding write. Its regression checks duplicate
I/O/failure boundaries, not a demonstrated loss of durability. Other regressions
exercise source reopening after identity change, cancellation during hashing,
actual compaction peak rejection and consistency of public staging estimates.


## Final-tree T0006, 2026-09-19

Acceptance 2 is met on the reviewed candidate
`0672d24f716a9293ec77de68b31e26e39fa64159`. A clean detached worktree ran the
full `cargo xtask mutation-check --battery 0006` end to end:

- all 12 clean lanes passed stably, three repetitions each, before and after;
- **63/63 declared mutants caught**;
- **3/3 expected survivors held** (the two independence controls and the
  equivalent runtime disk-budget check);
- **zero** unexpected survivors, unstable verdicts, invalid controls, broken
  controls or skipped substitutions; and
- exit status 0, unchanged detached HEAD and a clean tree with no parked
  original afterward.

The five survivors from the `356f965` recovery run are among the 63 caught
mutants. This is the complete battery that the targeted repairs did not claim
to replace. Durable local evidence:
`results/m3-recovery/t0006-final-0672d24.log`, SHA-256
`b5e61d449ad87661f230b6b3147b5d4045a3d584d3f86a680722dde3e719fc1c`.
Acceptance 3 is also met on the same identity. The full T0028 battery caught
**24/24 declared mutants**, held **1/1 expected control**, and reported zero
survivors, unstable verdicts, invalid controls, broken controls or skips. All
six clean lanes were stable for three repetitions before and after. The run
exited zero with unchanged detached HEAD, a clean tree and no parked original.
Evidence: `results/m3-recovery/t0028-final-0672d24.log`, SHA-256
`adb5a19f543026d9197b446419e0f28b8e8cd427fcd06e25ddb4d73596a3ce9b`.
