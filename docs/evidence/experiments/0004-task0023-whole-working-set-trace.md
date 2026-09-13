# 0004 — Measuring task 0023's byte accounting and its reconciliation, by mutation

Date: 2026-09-13. Milestone: M2's exit, [task 0023](../../tasks/0023-m2-whole-working-set-trace.md).
Status: **accepted**; **30 of 30** mutations caught, 0 survivors. The first
measurement was 27 of 30, and all three survivors were **defects** -- one missing
equality, one unchecked counter and one property with no fixture -- rather than
opinions about test strength.

Companion to [experiment 0003](0003-task0022-sweep-and-routed-mutations.md), which measured the
sweeps this task extends rather than replaces.

## Why this exists

Task 0023 adds a **reconciliation**: a set of named equalities between three independently
maintained descriptions of the same bytes — the residency authority's per-scope byte flow, the
resource ledger's charges, and the planner's prediction. Two different questions follow, and they
need two different measurements.

The first is whether each equality can **fail**, and is answered by a violating fixture per
equality: a trace that reconciles, mutated in one number, must be rejected by the equality that
number belongs to and named as that one. That battery lives in the test itself
(`every_equality_can_fail_and_names_itself`) because a check nobody has ever violated is
indistinguishable from one that was never written.

The second is whether the **accounting** the equalities are computed from is itself checked, and that
is what this record measures. An equality between two numbers that are both wrong in the same way
holds perfectly, and that failure mode is not reachable from the test side at all: it needs a
mutation of the code that produces the numbers.

## Method

Each mutation replaces one piece of production source with a plausible defect, rebuilds, and runs
the host-lane tests of `moxie-memory`, `moxie-plan` and `moxie-executor`. A mutation that does not
compile is **caught** (the type system is part of the suite); a mutation that compiles and leaves
every test passing is a **survivor** and is either a gap or an equivalent mutant, reported as
whichever it is.

The harness, the exact substitutions and the raw output are in the task's working directory and are
not tracked; what is tracked is this record and the tests the measurement produced.

## Results

**30 mutations, 30 caught, 0 survivors** — after the three fixes the first
measurement forced. The first measurement was **27 of 30 with three survivors**,
and all three were defects rather than opinions about test strength:

| Survivor | What it was | What it produced |
|---|---|---|
| Setting the residency high-water mark to the current level at every admission | Nothing ever compared **two readings** of `peak_resident_bytes`. `check_invariants` asks whether the level is below the mark, which stays true when the mark follows the level down | The sweep now requires a scope's peak to be non-decreasing from layer to layer |
| Deleting the device launch counter's increment | Nothing compared `launches` to anything at all. Document 07 lists launch count among the per-phase counters, and a count nobody checks is a count nobody can trust | A **fourteenth equality**, `launches-match-device-groups`, with its own violating fixture |
| Deleting the filter that keeps only **this step's** reservations | In every fixture the only reservations the ledger held *were* this step's, so filtering and not filtering gave the same answer. "No third charger" was the property with no fixture at all | `a_charge_nobody_can_name_is_reported`: an unrelated consumer's reservation, live across a layer, in a tier this step also uses |

The third is the shape AGENTS.md already records: **a fixture on which two
behaviours agree tests neither.** It is worth noticing that the property it left
untested is the one the equality exists for — the others check that the numbers
agree, and this one checks that nobody else is spending.

### The mutations

| Mutation | What it breaks | Caught by |
|---|---|---|
| `M01` requested uncharged | an acquire is not counted as a request | `residency` unit tests |
| `M02` hit uncharged | a cache hit moves no bytes into the account | the 800-combination transition sweep |
| `M03` coalesce uncharged | joining a transfer in flight counts as nothing | `residency` unit tests |
| `M04` refusal uncharged | a refusal is not counted | the transition sweep |
| `M05` origin swapped | an admission's origin is decided backwards | `residency` unit tests |
| `M06` source origin swapped | creating a host source and reusing one are exchanged | the transition sweep |
| `M07` read uncharged | a completed read moves no bytes | `residency` unit tests |
| `M08` upload uncharged | a completed upload moves no bytes | the transition sweep |
| `M09` unfinished uncharged | a placement dropped in flight reports nothing | the transition sweep |
| `M10` departure mislabelled | an eviction is recorded as a retirement | the transition sweep |
| `M11` outcome never recorded | a transfer outcome is never noted | the transition sweep |
| `M12` peak not maintained | the high-water mark follows the level down | the trace sweep (**after** the first measurement) |
| `M13` rollback uncharged | a rolled-back host source is not recorded | the trace sweep and the violation battery |
| `M14` device hit unpredicted | the planner predicts no device hits | the 31,104-combination planner sweep |
| `M15` read predicted whole | a reused host source is predicted to be read anyway | the planner sweep |
| `M16` always exact | every prediction claims to be an equality | the planner sweep |
| `M17` exactness ignores residency | exactness ignores what the layer is counting on | the planner sweep |
| `M18` reuse ignores device | a chunk on the device also counts as a reusable source | the trace sweep and the battery |
| `M19` host residency ignored | a host group never hits | the planner sweep |
| `M20` retry not counted | a refused acquire is not counted as an attempt | the trace sweep and the battery |
| `M21` launches uncounted | device launches are not counted | the new equality (**after** the first measurement) |
| `M22` delta is not a delta | a layer's flow is the whole run's | the allocation gate |
| `M23` sum drops a term | summing two flows keeps only the first | the trace sweep and the battery |
| `M24` any reservation accounted | every charge is accounted for, whoever made it | the third-charger case (**after** the first measurement) |
| `M25` bound never fails | a declared lower bound cannot be violated | the battery's lower-bound arm |
| `M26` equality never fails | an equality cannot be violated | the battery and the third-charger case |
| `M27` tiers narrowed | only the first tier of a scope is compared | the allocation gate |
| `M28` always complete | a layer that did not finish is traced as one that did | the trace sweep |
| `M29` withheld ignored | a withheld lease is counted as lost | the trace sweep |
| `M30` failed bound reversed | a failed layer's group bound is checked backwards | the battery's interrupted-layer arm |

## What the measurement found, beyond the three survivors

**Every mutation of the accounting was caught by a sweep that already existed.**
`M02`, `M04`, `M06`–`M11` are all caught by task 0020's transition sweep, without
a new harness, because the conservation identities went into
`ResidencyAuthority::check_invariants` rather than into a test of their own. That
is the whole argument for putting them there: the sweep calls it after **every**
operation over 800 combinations, and an identity that holds there holds under
interleavings no test would have thought to write.

**A null result, reported as one.** The trace sweep's `fail_read_at` axis caught
no mutation that the other axes did not already catch. What it produced instead
is the incomplete-layer branch of the reconciliation — a layer that did not
finish still has to account for its bytes — and the three violating fixtures that
branch needed. The axis earns its place through the code it forced, not through a
mutation it was the only one to catch.
