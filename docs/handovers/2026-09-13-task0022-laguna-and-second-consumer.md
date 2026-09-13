# Handover — task 0022 implemented; M2 item 5's remainder is next

**Task 0022 is implemented and awaits independent review and owner acceptance.**
It delivers roadmap **M2 item 4** in the only shape the evidence allows, and it
says so in its own contract rather than afterwards. **It does not close M2.**

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Contract `78c493d`, written and committed before implementation; the
  implementation is the commits between it and the one this handover
  accompanies. Base before both was `551b7cd`.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, re-verified; its untracked `.pi/`
  and `tests/p2p/` remain untouched. **The frozen legacy tree has no Laguna
  adapter**, so it was not a reference for anything here.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was copied, converted, deleted, downloaded or
  modified.** Two metadata files of one artifact were read — `config.json` and
  the safetensors index — plus the headers of its shards. **No Laguna tensor
  payload was read.**

## Completed facts

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy | passed |
| `cargo test --workspace --locked --offline` | **908 passed, 0 failed** (883 at task 0021) |
| Device-feature workspace tests | **941 passed, 0 failed** (913 at task 0021) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures**, 78 rejected fixtures, 21 accepted, 13 rules |

Nothing failed and nothing was skipped. The 883 baseline was **re-measured** in
an isolated worktree at `551b7cd` rather than quoted from the previous handover;
the first attempt at that comparison was taken on a half-built tree and read
874, which is why it was taken again properly.

**A routed layer at a second artifact's declared expert shape executed on all
three GPUs.** Top-k 10, hidden 3,072, `moe_intermediate` 1,024 — so
**18,874,368 B** per expert — over twelve synthetic experts, against a device
cache of **28,311,552 B**, one eighth of the **226,492,416 B** its route
demands. Eleven backpressure drains and twenty-two evictions on each card, and
all **6,144** BF16 components agreeing with the CPU candidate.

**That is weight-shaped bytes at a real artifact's declared shape, not its
weights.** Laguna's experts are asymmetric INT4 at group 32 and no importer in
this workspace accepts them. Nothing here is Laguna support and no quality claim
follows.

## Decisions

**Laguna's attention tower is not composable, and the deliverable is the routed
block.** Two operations are missing and neither may be guessed:

1. **Attention output gating** — `softplus(g_proj(x))` per head, before
   `o_proj`. The equation is pinned in the artifact's own `modeling_laguna.py`,
   so it is *known*; what does not exist is a shared operation, its oracle and a
   second consumer. `gating_types` is `per_head` on **all 48 layers**.
2. **The yarn rotary ramp** on the twelve `full_attention` layers.
   `modeling_laguna.py` implements only `compute_default_rope_parameters` and
   delegates every other `rope_type` to a `transformers` function it does not
   ship. The artifact declares `transformers` **5.14.1**; the copy installed
   here is **5.5.3**, and `truncate` — a parameter of that function — is not
   declared in this config at all.

The gemma4 record's own practice was to compare three copies of a reference
before pinning a rope reading. One mismatched copy is not a pinned exporter, and
a wrong ramp puts a wrong angle on every position of twelve layers.
`ArtifactGeometry::tower_gaps` computes the list from the declared geometry, so
a configuration without them reports none — both outcomes reachable, both
tested.

**Two routing parameters were deliberately not added.** Router logit softcapping
(this artifact declares 0.0) and `norm_topk_prob = false` (both families declare
true) have no consumer, and an unreachable branch is a stub. A configuration
that needs the softcap is **refused by name**, where the gap is visible.

**Sigmoid and softmax select the same experts.** Both are monotone in the logit,
so on their own they differ only in the coefficients; it is
`e_score_correction_bias` that makes the transform decide the *selection*, and
only because a bias shifts a sigmoid score and a softmax probability by
different relative amounts. This is worth carrying because the obvious claim —
"a different score transform routes differently" — is **false**, and would have
become a comment if a fixture had not been written to check it.

**The output scale multiplies the sum, not the terms.** `Combine::output_scale`
multiplies the FP32 accumulator once, before the single BF16 store. The fixture
that separates the two application points is a cancellation at `2^24` with
Laguna's own 2.5: sum-then-scale gives 2.5, scale-then-sum gives 4.0.

**The restricted budget is a ratio, not a constant.** `ceil(working_set /
ratio)`, where the working set is the **union** of experts the route demands —
document 03's "do not multiply active experts by batch rows when routes
overlap". A separate case asserts the *relation*, so a shape change that quietly
made the budget larger than the working set fails by name instead of turning a
restricted case into an unrestricted one that still passes.

### Two coverage gaps, found by mutation and not by reading

The two sweeps gained a profile axis over their whole product — 10,368 and 288
combinations — and were re-measured: **33 of 33 mutations caught, 0 survivors**
([experiment 0003](../evidence/experiments/0003-task0022-sweep-and-routed-mutations.md)).
The **first** measurement had three survivors, in two defects:

**A parameter no fixture ever varies is a parameter no test checks.** Two
mutants that dropped `Combine::output_scale` on the way to the plan survived the
entire 10,368-combination sweep *and* every executor test. Nothing was weakly
asserted; the assertions were strong over an input space in which the parameter
was always 1. This is the fourth review of task 0021's lesson — **a gate only
fires on inputs something actually hands it** — arriving somewhere new.

**A test that compares a thing to itself is not a test.** The operand-swap test
built the graph with `per_expert_scale` and `selection_bias` exchanged and
required different answers. Reversing `OpParams::route_operands` relabels the
validation and the interpretation *consistently*, so the two runs were still
different — they were each other's. It is now checked against an oracle composed
with each operand in the role its name says.

**A null result, reported as one:** the profile axis caught no mutation the
single-profile sweeps did not already catch. What it produced was the fixture
pressure — the first routed scale in the workspace that is not 1 — which is how
the `output_scale` gap became visible at all.

## Remaining hypotheses and blockers

- **M2 is not closed.** Its exit still asks that a real out-of-device-memory
  working set execute and that byte/cost traces reconcile with the resource
  ledger across a **whole** working set rather than one layer.
- **Neither designated artifact can execute as a model.** Gemma 4 26B-A4B is
  51.6 GB BF16 against a 24 GiB largest card; Laguna is 76.8 GB and needs an
  importer that does not exist.
- **The Laguna importer** is M3's: asymmetric INT4 at group 32, and — a second
  convention the existing reader has never seen — **zero points packed along the
  output axis** while the codes are packed along the input axis.
- **The fused/per-expert expert mapping is an open question.** The pinned model
  declares fused expert tensors; the checkpoint stores one tensor per expert per
  projection; the artifact's own `_checkpoint_conversion_mapping` covers only
  `e_score_correction_bias`.
- **No performance claim.** Both lanes are debug builds; there is no baseline on
  this machine. The amortisation threshold is still a declared policy parameter
  and **not** a measured crossover — and at Laguna's expert size the default
  sends every expert to the CPU, exactly as it does at Gemma's. M6 owns
  measuring one.
- **Quality is O2** for both families. O1 and O5 are open.
- **The `dflash` draft model** Laguna declares is not on this machine and has
  not been inspected. M9's.

## Next task

Task 0023 is **M2 item 5's remainder and M2's exit gate**: byte and cost traces
reconciled with the resource ledger across a **whole working set**, not one
layer.

- **Owning components:** `moxie-memory` for what it already owns —
  `ResidencyAuthority`'s own accounting is the source of the byte trace, and a
  second one anywhere is a failed task; `moxie-executor` for the run's cost
  record; `moxie-plan` for what the envelope predicted. **`moxie-models` gains
  nothing**, and no model-owned path may appear.
- **Required reading before the contract:** document 03's resource-ledger and
  admission section in full, especially "an admission report must show physical
  capacity, already committed resources, reserved peak, and remaining headroom
  for every tier"; document 07's benchmark and evidence rules, because a trace
  is evidence and has a schema; M2's exit paragraph; and task 0020's and 0021's
  accounting, which is what a trace must reconcile *against*.
- **The contract must state, before implementation:** what a "whole working set"
  is at a scale this machine can actually run — the designated artifact's real
  layer count is 30 and its experts are 51.6 GB, so the honest unit has to be
  named and defended rather than assumed; which tiers the trace covers and at
  what granularity; what "reconciled" means as an equality rather than as a
  comparison of two approximations; and what the trace is **not** (it is not a
  benchmark, and a debug build's wall clock is not a cost).
- **Stop conditions:** a second accounting owner; a trace assembled from
  estimates rather than from what the authority and the ledger actually
  recorded; any performance claim; any quality claim (**O2**); any bulk write
  (**O5**); and executing remote model code.
- **Everything task 0022 added must keep passing unchanged**, including both
  sweeps at their extended products and the restricted-budget device case on all
  three cards.

**Three habits from this task should be applied there rather than rediscovered.**

First, **ask what the input space is, not only what the assertion says.** The
`output_scale` gap was not a weak test. It was a strong test over fixtures that
all used the same value, and no amount of assertion strength would have reached
it. Before claiming a parameter is covered, ask which fixture varies it.

Second, **a test that compares two runs of the same code is not a test of what
the code means.** The operand-swap check survived its own mutation because both
halves read the same single statement of the order. When a property is about
*meaning*, the comparison has to be against something written independently.

Third — and this is the contract's own doing rather than the measurement's —
**write the narrowing into the contract, before implementing.** Laguna's tower
was going to be impossible whichever order the work was done in. Discovering it
halfway would have produced either a guessed yarn ramp or a task that quietly
delivered less than its roadmap item says. Stating it first turned it into a
scoping decision with evidence attached, and left two named gap tasks behind
instead of a silence.
