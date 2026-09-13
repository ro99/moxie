# 0002 — Measuring task 0021's sweeps, by mutation

Date: 2026-09-12. Milestone: M2, [task 0021](../../tasks/0021-m2-expert-execution-plans.md).
Status: **accepted**; the sweep catches every mutation in the battery, after two rounds of
strengthening that the first measurement forced.

## Why this exists

AGENTS.md records three task-0020 claims that were wrong in the same way: a property of the tests
was asserted instead of measured. "The sweep covers the chooser" is exactly that kind of claim, so
task 0021's contract required it to be measured rather than stated. This record is the measurement,
the battery it was taken with, and — more usefully — the two gaps it found.

## Method

Sixteen deliberate mutations of `moxie_plan::expert`, each a single edit that changes the decision
the planner makes while leaving it compiling. For each: apply, run
`cargo test -p moxie-plan --test expert_plan_matrix -- the_chooser_agrees`, then (if the sweep
passed) run the rest of `moxie-plan`'s and `moxie-executor`'s host tests to see whether anything
else caught it, then revert. The driver script is reproduced at the bottom so the number can be
re-taken rather than believed.

The mutations, by what they break:

| Mutation | What it changes |
|---|---|
| `residency-ignored` | charges transfer for an already-resident expert |
| `whole-transfer-not-per-row` | compares the whole chunk against the per-row threshold |
| `amortisation-strict` | `>=` instead of `>` at the threshold |
| `cache-check-dropped` | never refuses on device-cache capacity |
| `arena-check-dropped` | never refuses on device-arena capacity |
| `required-device-falls-back-silently` | reports a capacity error where a `required` candidate's own reason belongs |
| `required-host-not-excluded` | lets the device candidate be considered while the host one is `required` |
| `both-required-allowed` | accepts two `required` candidates at once |
| `order-ignored` | ignores `CombineOrder` and always reduces in selection order |
| `envelope-unaligned` | charges logical rather than aligned device regions |
| `queue-unbounded` | ignores `max_inflight_orders` |
| `slot-is-row` | writes a group's results to the row index instead of the slot index |
| `kernel-sm-ignored` | selects a kernel without matching the SM |
| `placement-defaults-to-node-zero` | places host buffers on node 0 when the topology gives none |
| `demand-bytes-doubled` | double-counts the residency demand it reports |
| `group-rows-unsorted` | builds each group's row list in reverse |

## Result

**16 of 16 caught by the sweep**, 0 by a named test, 0 survivors — *after* the two corrections
below. The first measurement was **13 of 16 by the sweep, 1 by a named test, 2 survivors**, and the
survivors are the interesting part.

### The first gap: a refusal's reason is part of its contract

`required-device-falls-back-silently` and `both-required-allowed` both survived because the sweep
asserted only **that** a case was refused. Both mutants still refuse; they refuse with the wrong
error. A caller told `CapacityExceeded` when the actual cause is "you asked for a candidate that is
not admissible" cannot act on it — which is precisely what document 01's `required` rule exists to
prevent. The sweep now classifies every refusal's error against an independently written
expectation.

The same defect was present in the product code, not only in the test, and the real-artifact case
surfaced it independently: a plan reported `11,894,784 B` exceeding `47,579,136 B` available,
because the *other* candidate had been excluded by a control rather than by capacity.

### The second gap: a fixture that cannot tell two answers apart

`order-ignored` survived the sweep and was caught only by a named test. The reason was the sweep's
own route: every row selected its experts in ascending id order, so `AscendingExpertId` and
`SelectionOrder` produce the same permutation and a planner that ignores the parameter is
indistinguishable from one that honours it. The sweep's route now selects in **descending** order,
and the sweep checks the permutation against an independently computed one.

That is the more general lesson and it is not new: a fixture on which two behaviours agree tests
neither. It is worth stating because the fixture looked entirely reasonable — a route is a route —
and nothing but a mutation would have shown it.

### A dead branch, found by a mutation that changed nothing

An earlier run's survivor was a mutation of a branch handling "the device candidate is `required`
and this group fell back to the host". Disabling it changed no test because the branch is
**unreachable**: when the device candidate is `required`, the host candidate is excluded for the
whole plan, so a group can never fall back. AGENTS.md counts an unreachable path as a stub, so the
branch was deleted rather than given a test. The both-refused arm above it already reports the
required candidate's reason.

## What this does not establish, demonstrated the hard way

A mutation battery measures a sweep against the mutations someone thought of. Sixteen is not a proof
of adequacy, and the two gaps above are evidence for exactly that: both were invisible until a
mutation aimed at them. The number belongs beside the sweep's own printed coverage — 5,184
combinations, 1,348 planned, every rejection reason exercised — and neither replaces a named
regression for a specific defect.

**An independent review then found ten defects this battery could not have reached**, seven of them
P1. Nine were a check present on one path and missing on the neighbouring one — the upload path
validated a backing and the launch path did not; feasibility used a smaller envelope than admission
reserved; a run could be cancelled but not fail. Mutating the chooser cannot find a defect in the
executor, and enumerating one state machine's product cannot find two paths that were never compared
to each other. The battery measures what it measures.

## The same method, applied to the review's own regressions

The twelve regressions written for those findings were put through a **substitution** battery of the
same shape: remove the check, run only that regression, and require it to fail.

**Three of the twelve did not fail.** Each asserted the *symptom* rather than the check — a run that
fails either way, a fixture tripping three bounds at once so none of them is pinned, a budget that
happens to be generous enough that alignment does not matter. They were rewritten: one now asserts
the failure comes from the acquire by its own message, one uses a fixture where only the row bound
can fire, and one pins the device budget at the exact byte from both sides. It is **12 of 12** now.

A regression is not load-bearing because it was written for a defect. It is load-bearing when a
substitution says so, and a quarter of these were not.

**A second review round then found five more defects**, three P1, and two of them were the *other
half* of findings the first round's corrections had claimed to close: an unknown submission was
reported on the launch path and not on the activation-upload path, and the buffers were quarantined
while the charge was still released. The battery grew to eighteen checks and **four of the six new
ones survived their first substitution** — three because the mutation was aimed at the wrong one of
two identical lines, and one because the check it removed was genuinely redundant. That last line
was **deleted**: when a substitution says a check cannot be observed, the honest answer is to remove
it, not to write a test that reaches it by another route.

It is **eighteen of eighteen** now. One check is deliberately **not** in the battery and is not
claimed as covered: the attachment-level quarantine in `DeviceExperts::close` sits behind a
run-level check that is covered, and reaching it directly needs a CUDA fault injected after an
enqueue, which this task does not do.

## The battery that was missing, and now exists

The first two review rounds found their defects in the **executor's** transitions while the only
enumerated product in task 0021 was the **planner's**. The third round said plainly that naming that
gap for a later task is not evidence this one is finished. It was right, and the sweep is built.

`tests/grouped_transitions.rs` enumerates **144 combinations** — candidate × failure point ×
cancellation × close ordering × queue depth — calling `GroupedRun::check_invariants` after every
operation and reconciling the residency authority's own live-lease count against the run's account
after every one of them. That second check is **external** to the run, and it is the one that
catches a lease the run believes it returned.

Coverage, printed by the test: 144 combinations, 12 completed, 68 failed, 64 cancelled, 128 closed
and **16 closes refused**, 16 withholding, 8 with backpressure, every failure point reached and both
capacity outcomes — the recoverable drain and the fatal refusal — separately.

Strength, measured: **14 of 14** deliberate mutations of the run machinery, two of which are the
second review's own P1s. Four of them are caught by this sweep and by nothing else.

### It took four rounds of strengthening, and every one was the measurement working

The first measurement was 8 of 14 by the sweep, 4 by named tests and **2 survivors**, and the
reasons were all instructive rather than mechanical:

- One survivor was an **equivalent mutant** I had written — a change that could not alter behaviour.
  Replaced, not counted as a gap, which is the rule task 0020 established.
- One survivor was a check the sweep could not reach because an earlier check always fired first:
  `reduce` refusing a non-empty queue is unreachable while `usable()` rejects every failed or
  cancelled run. The sweep now attempts a reduction and a reload **mid-run**, where neither has
  happened, which is a real transition it was not enumerating at all.
- Three mutants survived on a subtler point: the sweep **recorded that an axis was reached** without
  checking what it caused. A corrupt read reached "acquire-read" whether or not it failed the run;
  an unknown submission reached "load-unknown" whether or not it withheld anything. Each axis now
  asserts its own outcome, including that the **host buffers** specifically are quarantined when the
  copy's source is one of them.

Counting that a failure happened is not the same as checking what it did, and a sweep that only
counts is a sweep that measures its own itinerary.

### And a fourth round found an axis that was not there at all

The sweep advertised "close ordering" as an axis. It was not one: the flag toggled only whether a
reduction was attempted, and **both** branches drained the queue before closing, so `close`'s
queued-work refusal was unreachable. A fourth review demonstrated it in the only way that settles
such a question — delete the guard, run the sweep, watch all 144 combinations pass.

Sixteen combinations now close **with work still queued** and require the refusal to leave the
queue, the authority's live-lease count and the ledger charge exactly where they were, then recover
the returned run and close it properly. The guard is in the battery: **15 of 15**.

## The number that matters most in this record

**Three separate coverage claims in one task were softer than they looked.** Regressions that
asserted the symptom rather than the check. Sweep axes that recorded being reached without checking
what they caused. And an advertised axis that was unreachable. Every one was found by asking what a
mutation would survive; **none** by reading the test, and none by the three independent review
rounds that read the same code.

That is the argument for keeping these batteries, and it is stronger than "the sweep is thorough."
An axis that is exercised is not an axis that is checked, and only a mutation can tell the two
apart.

## Reproducing it

The driver is a short script; it is reproduced here rather than committed as a tool, because it
edits source in place and a tool that does that should not be one command away.

```python
# For each (name, old, new): write the mutated source, run the sweep, record,
# restore. `SRC` is crates/moxie-plan/src/expert.rs.
for name, old, new in MUTANTS:
    SRC.write_text(original.replace(old, new, 1))
    swept = run("cargo test -p moxie-plan --test expert_plan_matrix -- the_chooser_agrees")
    if swept:            # the sweep missed it; did anything else catch it?
        elsewhere = run("cargo test -p moxie-plan") or run("cargo test -p moxie-executor")
    SRC.write_text(original)
```
