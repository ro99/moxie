# Task contracts

## Current assignment and milestone handoff

**M1 complete (M1.5 closed 2026-09-12); M2 active.** See
[the closure handover](../handovers/2026-09-12-m1-closure-to-m2.md) for M1's
exit evidence gate by gate, and
[the owner-gate register](../decisions/owner-gates.md) for the partial O1 ruling
that designated `/fast/models/google/gemma-4-26B-A4B-it` as M2's BF16 MoE.

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

**[Task 0020](0020-m2-weight-residency-authority.md) is active**: M2 item 2's
residency authority, specified in
[the task 0019 handover](../handovers/2026-09-12-task0019-routed-expert-semantics.md#next-task).
Its contract is authored and committed before implementation. It delivers one
production weight-residency owner connected to bounded storage reads, a host
cache, upload readiness, leases, eviction and demand/prefetch classes. **It does
not close M2**, which also needs items 3–5 and a real working set that executes.

Reopen accepted tasks only for a demonstrated defect in accepted scope.

One file per bounded assignment, `NNNN-slug.md`. Use [TASK.md](../spec/templates/TASK.md).

A task produces one primitive, one planner capability, one state transition, one importer rule, one
integration slice, or one controlled experiment. It is not "rewrite GLM". Fill in the Result section
after the work, keeping passed, failed and skipped separate.
