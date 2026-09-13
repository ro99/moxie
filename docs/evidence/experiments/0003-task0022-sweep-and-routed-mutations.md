# 0003 — Measuring task 0022's extended sweeps and its new routed parameters, by mutation

Date: 2026-09-13. Milestone: M2, [task 0022](../../tasks/0022-m2-laguna-metadata-and-second-consumer.md).
Status: **accepted**; **37 of 37** mutations caught, 0 survivors, after an independent review's six
findings were fixed and four of its checks were added to the batteries. The measurement before the
review was 33 of 33, after two corrections the first measurement forced.

Companion to [experiment 0002](0002-expert-plan-sweep-mutations.md), which measured the same two
sweeps before task 0022 extended them.

## Why this exists

The task 0021 handover asked that the second MoE consumer go **through** the existing sweeps rather
than beside them, on the grounds that "the measurement is already there to say whether the extension
helped." That is a claim about test strength, and AGENTS.md's standing rule is that such a claim is
measured or it is not known. This record is the measurement.

## What changed in the sweeps

Both sweeps gained a **profile** axis, `{GemmaLike, LagunaLike}`, over their whole existing product.

| Sweep | Before | After |
|---|---|---|
| `moxie-plan/tests/expert_plan_matrix.rs` | 5,184 combinations | **10,368**, 5,184 per profile |
| `moxie-executor/tests/grouped_transitions.rs` | 144 combinations | **288**, 144 per profile |

The Laguna-like profile differs on every axis the routed contract has: SwiGLU instead of GeGLU, a
top-k of 3 instead of 2 (so every slot, staging and workspace figure changes), more experts than the
Gemma-like route uses (8 against 6, and in the run sweep 8 distinct experts against 5, so a
restricted cache evicts more often), different hidden and intermediate widths, a different reuse
distribution, and a routed scaling factor of 2.5 against 1.0.

One thing had to change in the fixtures for the axis to mean anything: the run sweep's kernel
catalogue and the planner sweep's both declare a descriptor for **the gate transform the profile
under test asks for**, not a catalogue holding both. A catalogue answering every gate transform
would make `Kernels::Matching` mean "some kernel exists" rather than "a kernel for this operation
exists", and the second profile would stop testing the selector at all.

## Method

The same method as experiment 0002. Each mutation is a single edit that changes behaviour while
leaving the source compiling; for each: apply, run the sweep it should be caught by, then — if the
sweep passed — run the rest of the affected crates' host tests to see whether anything else caught
it, then revert. Two drivers, reproduced at the bottom.

## The planner battery: 18 mutations

Task 0021's sixteen, re-derived against the current source and re-run against the **extended**
sweep, plus two of task 0022's own.

| Mutation | Caught by |
|---|---|
| `residency-ignored` | sweep |
| `whole-transfer-not-per-row` | sweep |
| `amortisation-strict` | sweep |
| `cache-check-dropped` | sweep |
| `arena-check-dropped` | sweep |
| `required-device-falls-back-silently` | sweep |
| `required-host-not-excluded` | sweep |
| `order-ignored` | sweep |
| `envelope-unaligned` | sweep |
| `queue-unbounded` | sweep |
| `slot-is-row` | sweep |
| `kernel-sm-ignored` | sweep |
| `placement-defaults-to-node-zero` | sweep |
| `demand-bytes-doubled` | sweep |
| `group-rows-unsorted` | sweep |
| `duplicate-route-accepted` | a named test |
| `output-scale-dropped-from-plan` | sweep — **after the correction below** |
| `combine-spec-scale-ignored` | sweep — **after the correction below** |

**17 of 18 by the sweep, 1 by a named test, 0 survivors.** The first measurement was **15 by the
sweep, 1 by a named test, 2 survivors**, and both survivors were the same defect.

### The gap: a parameter no fixture ever varies is a parameter no test checks

`output-scale-dropped-from-plan` returns 1.0 from `ExpertPlan::output_scale`, and
`combine-spec-scale-ignored` drops the scale on the way out of `shape_of`. Both survived the whole
10,368-combination sweep **and** every executor test.

The reason is not subtle once seen and was invisible before: `Combine::output_scale` is new, and
every fixture in the workspace used **1.0**, because until Laguna arrived no family had a routed
scaling factor. The sweep checked the reduction *order* against an independently computed
permutation and said nothing at all about the scale beside it.

Two corrections, and they are different in kind:

- the sweep now asserts `plan.output_scale()` against the profile's declared value, which is what
  makes the parameter part of the enumerated product rather than a field that happens to exist;
- `the_combine_output_scale_reaches_the_reduced_rows` is a new executor regression: it reduces the
  same fixture at 2.5 and at 1.0, requires the first to equal an oracle computed **with** the scale,
  and requires the two to differ. Without that second assertion it would be a fixture on which two
  behaviours agree.

This is the fourth review of task 0021's lesson arriving in a new place: **a gate only fires on
inputs something actually hands it.** There was no weak assertion here. There was a strong
assertion over an input space in which the parameter was always 1.

## The run and routed-parameter battery: 15 mutations

Six of the run machinery, re-run against the **extended** run sweep, and nine of task 0022's own
additions across four crates.

| Mutation | Crate | Caught by |
|---|---|---|
| `close-accepts-queued-work` | executor | run sweep |
| `close-ignores-withholding` | executor | run sweep |
| `unknown-launch-releases-leases` | executor | run sweep |
| `unknown-load-does-not-quarantine` | executor | run sweep |
| `backpressure-never-counted` | executor | run sweep |
| `invariant-accepts-unaccounted-leases` | executor | run sweep |
| `raw-router-accepts-a-gain` | oracles | run sweep |
| `kernel-output-scale-ignored` | kernels | a named test |
| `kernel-scale-per-term` | kernels | a named test |
| `reduce-ignores-the-plans-scale` | executor | a named test |
| `oracle-combine-scale-ignored` | oracles | a named test |
| `bias-gathers-the-biased-score` | oracles | a named test |
| `bias-ignored` | oracles | a named test |
| `sigmoid-is-softmax` | oracles | a named test |
| `route-operands-swapped` | graph | a named test — **after the correction below** |

**7 by the run sweep, 8 by named tests, 0 survivors.** The first measurement had **1 survivor**.

### The second gap: a test that could only compare a thing to itself

`route-operands-swapped` reverses the order of the two `[experts]` operands in
`OpParams::route_operands` — `per_expert_scale` and `selection_bias`, which have the **same shape**,
so no shape check can tell them apart. The test written for exactly this hazard,
`swapping_the_two_expert_vectors_changes_the_answer`, built the graph twice with the two bindings
exchanged and required the results to differ.

It survived. The reason is worth stating plainly: `route_operands` is the **single** statement of
the order, so both the shape validation and the interpreter's resolution read it, and reversing it
relabels both consistently. The two runs still produced two different answers; they were simply each
other's. An assertion that compares a thing to itself is not an assertion about what the thing
means.

The correction is an oracle: the test now composes the expected answer from
`moxie_oracles::route`'s accepted stages with each operand in the role its **name** says, and
requires the declared binding to equal it. The "swapping changes the answer" assertion is kept
beside it, because it catches a different failure — an implementation that ignored one of the two
entirely.

That is the same shape as task 0021's swapped kernel symbol, one level up: the check existed, the
fixture was reasonable, and nothing in it was anchored to an independent statement of the meaning.

## What this does not establish

Thirty-three mutations are thirty-three mutations somebody thought of, and the two gaps above are
the evidence for why that matters: both were invisible until a mutation aimed at them, and neither
was found by reading the tests.

Specifically **not** covered:

- The mutations were applied to the planner, the run machinery, the routed oracle, the reduction
  kernel and the operation catalogue. Nothing here measures the **device** path: the grouped kernel,
  the arena, the upload lane. Those have their own bitwise gate and their own device cases.
- The two gaps that block Laguna's attention tower are **not** mutations of anything, because there
  is nothing to mutate. A gap is measured by whether a task refuses, and
  `the_artifact_declares_the_two_gaps_that_block_its_tower` is that check.
- Whether the **profile axis itself** found anything new. It did not find a defect in the planner or
  the run machinery: all six re-derived run mutants and all fifteen re-derived planner mutants were
  already caught before the extension, and no mutation was caught by the Laguna-like profile alone.
  That is worth reporting as a null result rather than leaving unsaid. What the extension did
  produce is the *fixture pressure* that made the `output_scale` gap visible at all — the second
  profile is the first fixture in the workspace with a scale that is not 1.

## Re-measured after the independent review

The review that requested changes on task 0022 reported that it had **not** re-run the mutation
campaign, so it was re-run against the corrected code, with a mutation added for each check the
review's findings produced:

| Added mutation | Crate | Caught by |
|---|---|---|
| `coefficient-narrowing-dropped` | oracles | a named test |
| `planner-accepts-a-nonfinite-scale` | plan | a named test |
| `scaled-store-overflow-unchecked` | kernels | a named test |
| `doubling-unchecked` | models | a named test |

Totals after the corrections: planner battery **18 mutations, 17 by the sweep, 1 by a named test, 0
survivors**; run and routed battery **19 mutations, 7 by the run sweep, 12 by named tests, 0
survivors**. All six of the review's regressions were separately put through the substitution
battery — remove the check, run only that regression, require it to fail — at **6 of 6
load-bearing**.

### What this campaign could not have found, demonstrated

Two of the review's six findings were **outside every battery here**, and saying so is more useful
than the totals above. One was a number in prose: the expert inventory multiplied a per-layer cost
by all 47 routed layers in a document that had already recorded that two of them cost something
else, understating the working set by 6.87 GB. The other was an omission shared by an
implementation and its "independent" FP64 transcription, written from the same source by the same
reader — the two agreed, and a bitwise gate passed on their agreement.

A mutation battery measures a test suite against mutations of the **code**. It cannot reach an
arithmetic claim that lives in a record, and it cannot reach a boundary that the oracle and the
implementation are both missing, because mutating either one moves them apart and the test fails
for the right reason by accident. The structural answers are elsewhere: make the record's arithmetic
executable, and check a transcription against a quantity computed from the source's stated equation
rather than against the implementation it is supposed to be independent of.

## Reproducing it

Two drivers, reproduced rather than committed as tools, because they edit source in place and a tool
that does that should not be one command away.

```python
# Planner battery. SRC is crates/moxie-plan/src/expert.rs.
SWEEP = "cargo test -p moxie-plan --test expert_plan_matrix --offline -- the_chooser_agrees"
REST  = "cargo test -p moxie-plan --offline"
for name, old, new in MUTANTS:
    assert original.count(old) == 1          # an anchor that is not unique is not a mutation
    SRC.write_text(original.replace(old, new, 1))
    where = "sweep" if not run(SWEEP) else ("elsewhere" if not run(REST) else "SURVIVED")
    SRC.write_text(original)
```

```python
# Run and routed battery: the same shape, over four crates.
SWEEP = "cargo test -p moxie-executor --test grouped_transitions --offline -- the_run_lifecycle"
REST  = ("cargo test -p moxie-executor --offline --test grouped_experts && "
         "cargo test -p moxie-kernels --offline && cargo test -p moxie-oracles --offline && "
         "cargo test -p moxie-interp --offline && cargo test -p moxie-cli --offline && "
         "cargo test -p moxie-plan --offline")
for path, name, old, new in MUTANTS:
    ...                                       # same loop, per-file originals restored in `finally`
```
