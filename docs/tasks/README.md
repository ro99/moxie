# Task contracts

## Current assignment and milestone handoff

**[Task 0037](0037-m4-paged-device-attention.md) is the active contract:** the
first bounded M4 slice, common BF16 paged device attention with persistent
admitted KV, prefill/append/decode, full/sliding MHA/GQA and an actual 32,768-row
gate on SM86 and SM120. It is a contract before implementation and claims no
device-attention result yet.

**M3 is accepted and complete** (owner, 2026-09-19) through [task
0036](0036-m3-final-closure.md). The [M3→M4
handover](../handovers/2026-09-19-m3-closure-to-m4.md) records the frozen
implementation, final mutation totals and the exact boundary that remains.
M4 is authorized. M1 is complete; M2's five items are accepted and its formal
milestone closure statement remains the owner's to make.

O6/O7 remain open, so task0037 cannot turn correctness into a performance
claim. Checkpoint roots and the legacy repository remain read-only. Any bulk
artifact operation still needs a task naming artifact, revision, size and
retention.

## Historical accepted-task register

Accepted through M1: tasks 0003–0015 (shared tensors and interpreter, sequence
transactions, manifest and bounded reads, resource ledger and admission,
event-backed leases, device arena, admitted graph resource plan, selected BF16
device chain, appendable paged state, transactional sampler history, generation
service and diagnostic CLI), [0016](0016-m1-gemma-reduced-graph.md) (reduced
Gemma graph), [0017](0017-m4-per-layer-kv-geometry-and-window-eviction.md)
(per-layer paged geometry and window reclamation, accepted 2026-09-12) and
[0018](0018-m3-compressed-tensors-int8-importer.md) (compressed-tensors
importer, accepted 2026-09-12 within its import-only scope).

[Task 0019](0019-m2-routed-expert-semantics.md) is **accepted, 2026-09-12**,
after three rounds of independent review: M2 item 1's
routed-expert mathematics — router score transform with its tie rule,
selection, renormalization and per-expert coefficient scale; grouped compute
over a fused expert tensor; and an ordered combination — with FP64 oracles and
two independent consumers. It deliberately excludes M2 item 2's residency
authority, which is task 0020, because the authority admits against a demand set
the routing equation defines. **Nothing executes a checkpoint**, and a routed
synthetic graph is not MoE model support. The acceptance closes task 0019 only,
not M2.

**[Task 0020](0020-m2-weight-residency-authority.md) is accepted, 2026-09-12**,
after seven rounds of independent review — twenty-seven findings, all
reproduced, all fixed, none disputed; the seventh recommended acceptance with no
new blocking findings. **The acceptance closes task 0020 only, not M2.** It is
M2 item 2's residency authority — one production weight-residency
owner connected to bounded storage reads, a host cache, real device uploads,
leases, eviction and demand/prefetch classes, with all nine of M2 item 5's cases
tested. Its contract was authored and committed at `d6e9170` before
implementation. This artifact's own expert bytes were demand-read through it,
107,053,056 B of `/fast/models/google/gemma-4-26B-A4B-it` layer 0, verified
against an independent read. **Reading is not executing**: nothing computes with
those bytes and no checkpoint runs. **It does not close M2**, which also needs
items 3–5 and a real working set that executes.

**[Task 0021](0021-m2-expert-execution-plans.md) is accepted by the owner on
2026-09-13**, after four rounds of independent review — twenty-one findings,
fourteen P1, all reproduced and fixed, none disputed. It was implemented on
2026-09-12. **The acceptance closes task 0021 only, not M2.** It is M2 item 3: CPU
expert fallback and GPU grouped candidate plans under one interface, with an
admitted envelope, a bounded queue that refuses rather than waits, NUMA-placed
host buffers whose pages are **read back** rather than asserted, and a reduction
whose order is an explicit permutation the plan computes. Its contract was
authored and committed at `cdda4f4` before implementation. The grouped kernel is
**bitwise** equal to task 0019's oracle on all three GPUs for both gate
transforms, and **one layer's routed expert block of
`/fast/models/google/gemma-4-26B-A4B-it` executed** — 118,947,840 B into a
two-expert device cache — over **synthetic activations and a route the test
writes**. **That is not model support and no quality claim follows.** **It does
not close M2**, which also needs item 4 and byte/cost traces reconciled with the
ledger across a whole working set.

**[Task 0022](0022-m2-laguna-metadata-and-second-consumer.md) is accepted by the
owner on 2026-09-13**, after three rounds of independent review; it was
implemented the same day. **The acceptance closes task 0022 only, not M2.** The third round recommended acceptance with no new
blocking findings, within this task's documented routed-block scope. The rounds found **eight** issues, two
P1, all reproduced and fixed, none disputed — the second round's P1 was
introduced by the first round's own fix, asking its question of the rest of the
change found a second instance of the same class, and the third round found the
regressions written for it **flaky**, which made the previous round's
substitution evidence untrustworthy whichever way it had come out. It is M2
item 4: Laguna's metadata and its **routed block**, a second routed consumer
through task 0021's interface, and a restricted budget expressed as a ratio of
working weights. Its contract was authored and committed at `78c493d` before
implementation. A routed layer at Laguna's **declared expert width** —
18,874,368 B per expert — executed on all three GPUs against a device cache of
one eighth of what its route demands, agreeing with the CPU candidate on every
component. The P1 was a **missing BF16 boundary** — the router's own
`routing_weights.to(hidden_states.dtype)` — which survived a *bitwise* gate
because the FP64 transcription was written by the same reader and omitted the
same cast: **a transcription is independent of the implementation, not of the
reader.** Two other findings were a record and a claim that each contradicted a
fact their own document already contained, and neither was reachable by any
mutation battery. **Those are weight-shaped bytes at a real artifact's declared
shape, not its weights**: Laguna's are asymmetric INT4 at group 32 and no importer
accepts them. **It does not close M2**, whose exit still needs byte/cost traces
reconciled with the ledger across a whole working set.

It records one narrowing in the contract rather than discovering it later.
**Laguna's attention tower is not composable from today's operation
catalogue**: `softplus` attention output gating has no shared operation, and the
yarn RoPE ramp on its twelve `full_attention` layers is implemented nowhere in
the artifact — `modeling_laguna.py` delegates it to a `transformers` function
the artifact does not ship, and the locally installed copy is 5.5.3 against the
artifact's declared 5.14.1. Both are named gap tasks. The deliverable is the
**routed block**, which is composable in full from pinned sources, and which is
what M2's exit gate needs.

**[Task 0023](0023-m2-whole-working-set-trace.md) is accepted by the owner on
2026-09-13**, at `ad69a0a`, after three rounds of independent review — ten
findings, four P1, all reproduced and fixed, none disputed; the third reported no
new blocking findings and recommended acceptance "within its declared M2
engineering scope". Its contract was authored and committed at `840b0e3` before
implementation. It is M2 item 5's remainder and M2's exit clause on traces:
**byte and cost traces reconciled with the resource ledger across a whole
working set**, not one layer — a step trace whose layers are reconciled as
**seventeen named equalities** across the planner's prediction, the residency
authority's accounting and the ledger's charges, each with a violating fixture,
and every discrepancy constructible without allocating. All 30 routed layers of
`/fast/models/google/gemma-4-26B-A4B-it` ran on each of the three GPUs over
**synthetic activations and routes the test writes**, one route constructed so
its union is the whole expert set. **The acceptance closes task 0023 only, not
M2**, which is the owner's separate decision, and **no model-quality claim
follows** (**O2**).

**[Task 0024](0024-m3-asymmetric-int4-pack-quantized-import.md) is accepted by
the owner on 2026-09-13**, after three rounds of independent review — eight
findings, two P1, all reproduced, all fixed, none disputed; the third reported
no new blocking findings and recommended acceptance within the task's declared
importer-only scope. **The acceptance closes task 0024 only, not M3 item 2.**
Its contract was authored and committed at `e122de3` before implementation. It is M3 item 2's asymmetric half: the importer
reads compressed-tensors `pack-quantized` **asymmetric INT4 at group 32**, whose
`weight_zero_point` is packed along the **output** axis while the codes are
packed along the input axis — two conventions inside one tensor group,
distinguished by nothing in either name, which is R16 in its exact form. Six
modules of `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` and
`/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4` import, with **112,640
reconstructed values** checked bitwise against document 03's canonical FP32
equation over the source's own bytes **and** against the source's own
arithmetic including the BF16 rounding its reference applies — two quantities
that differ on **27,501** of those values, which the first version of this task
conflated. The lane assignment inside a zero-point word
is **measured** against the artifacts' own codes rather than taken from the
pinned library, because a zero-point word's lanes are different output channels
— the check a *code* word can never support, and a check that mattered because
both artifacts declare untagged development compressor versions. **Import is not
execution**: nothing consumes a canonical INT4 tensor, W4A16 is M3 item 3, and
Laguna's graph still declares BF16. A bit-identical repack is ADR 0018's v1
quality definition, **not** evidence about model output. Mutation-measured **24 of
24**, 0 survivors, 0 unstable, 0 invalid controls, 0 skipped, with the driver
committed, every verdict repeated three times in both directions, and its
verdict rule self-tested. The
first review's P1 was the workspace's most repeated defect for the **sixth**
time — a refusal that aborted under an allocation failure, on a sweep that only
ever imported valid inputs — and it is fixed at the shared error type rather
than in the six lines this task added. **The second review's three findings are
all the same shape: a first-round correction that was narrower than it looked.**
A refusal made safe while the lookup names reaching it still aborted; a
regression that tested the helper the buggy filter ran *before*; and a mutation
driver that counted an invalid verdict as a caught one. Fixing a finding and
guarding the fix are two jobs.

Reopen accepted tasks only for a demonstrated defect in accepted scope.

One file per bounded assignment, `NNNN-slug.md`. Use [TASK.md](../spec/templates/TASK.md).

A task produces one primitive, one planner capability, one state transition, one importer rule, one
integration slice, or one controlled experiment. It is not "rewrite GLM". Fill in the Result section
after the work, keeping passed, failed and skipped separate.
