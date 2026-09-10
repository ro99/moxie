# Handover — task 0011 admitted graph/resource plan

## Workspace identity

Writable root `/home/rodrigo/Developer/moxie`, branch `main`, based on `9e14eeb` with contract
`96f20fe`, implementation `a95fa3a` and final slot-unwind evidence `1e87bb4`. All changes belong to
task 0011. Local commits only; no push. Read-only legacy remains
`/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its existing
untracked `.pi/` and `tests/p2p/` paths were untouched. No checkpoint or generated model binary was
read, copied or transformed.

## Completed facts

[Task 0011](../tasks/0011-m1-admitted-graph-resource-plan.md) records the fixed contract, source
references, implementation and validation. `moxie-graph` issues process-local identities only after
successful validation and exposes an exact immutable signature. Pure `moxie-plan` binds one
stateless graph to an explicit phase/rows/visible-tokens/branch-rows/output/device bucket, derives
checked tensor extents and bytes, computes inclusive liveness, and assigns deterministic aligned
first-fit slots. It depends only on `moxie-types` and `moxie-graph`; architecture fixtures enforce
that it cannot reach memory or CUDA and models cannot reach it.

`moxie-executor` turns that candidate into one exact shared-ledger request, materializes one
activation arena, and exposes graph-bound tensor metadata without a raw pointer. Refusals return the
candidate or an explicitly held resource. Successful close frees physical memory before releasing
the charge; forced slot exhaustion and final-free ambiguity retain the required ownership. The plan
has no `execute` method, and execution validation returns `UnsupportedKernel`.

All declared host, device, fault, architecture, specification, no-driver, visibility and capacity
gates pass. The support matrix adds `G-RESOURCE-PLAN` and keeps the status scoped to synthetic
resource planning. No production path was deleted except replacement of the formerly free
`moxie-types::GraphId` vocabulary with graph-owned construction.

## Decisions

No owner gate, ADR or reference-document amendment was needed. Rows and visible history remain
separate fields. Verify, entropy and every stateful graph fail closed. BF16 and F32 extents are
exact; integer weight plans fail closed until affine metadata participates. Logical activation
intervals are diagnostics, while the one physical arena is charged once for the whole plan. Weight
requirements reserve their canonical BF16 bytes but do not claim a resident allocation.

## Remaining hypotheses and blockers

Independent review is pending. No semantic kernel is selected or launched, no weight is loaded or
resident, and no workspace contract has been bound to a qualified implementation. State, paged KV,
sampling, generation, checkpoint import, residency/eviction, multi-device execution, model quality,
actual context and performance remain unavailable or unmeasured. Quarantined resources recover at
process teardown; this task adds no recovery protocol.

## Next task

First review task 0011 against its contract, especially graph/signature identity, strict liveness
reuse, physical-versus-logical charging, candidate return on refusal, forced unwind and final-free
quarantine. After acceptance, write a separate bounded M1.3 contract for registry-backed semantic
kernel selection and one real device layer chain. The shared planner must own capability/shape/
layout/hardware selection and exact workspace additions; the executor must launch through task
0009/0010 operation leases and refuse any missing node before execution. Use independent host
oracles and real `sm_86`/`sm_120` evidence. Stop before model loading, weight residency policy,
paged state, generation, multi-device work, quality or performance claims.
