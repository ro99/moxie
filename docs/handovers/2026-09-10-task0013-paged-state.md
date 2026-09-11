# Handover — task 0013 appendable paged state

## Workspace identity

Writable root `/home/rodrigo/Developer/moxie`, branch `main`; clean base `b0f06fc`.
Contract `a80ff2c` preceded implementation `c14e32a`; first review correction
`c9a4b33` fixes the sole reported issue. This record and the task result follow
those local commits; no push was performed. Confirm current HEAD
and dirty paths before continuing. All changes in this assignment belong to
task 0013; no unrelated writable-tree changes were present.

Read-only legacy `/home/rodrigo/Developer/strata` remains at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and `tests/p2p/`
were preserved. No checkpoint was read or changed.

## Completed facts

[Task 0013](../tasks/0013-m1-appendable-paged-state.md) implements a fixed,
admitted host KV pool in `moxie-memory` and page addressing/transaction composition
in `moxie-state`. No separate state journal or sequence counter was introduced.
The existing transaction IDs are now unique across sequences; other accepted
state semantics are preserved. The dense interpreter cache stays as an oracle.

Whole/chunked/later storage append, two geometries, three >=16-bit encodings,
partial acceptance/rollback, pending-token materialization, cancellation at every
publication boundary, exact ledger accounting, capacity refusal and no-next-token
cleanup pass. The allocation gate stored 32,768 actual rows using 662,725 B of
admitted backing/control capacity, with zero allocations inside append and no
retained growth after 10,000 abort/retry cycles. This is storage evidence only.

Host workspace after correction: 570 tests + 8 doctests passed. Architecture: 61 rejecting and
16 accepted fixtures, 12 rules. Host clippy/format/specification passed. A fresh
build with CUDA tooling disabled links no libcuda. Deliberately broken identity
and physical abort controls fail; restored focused tests pass. The device-feature
workspace passes **586 tests + 11 doctests** and device clippy. Aggregate real GPU
qualification passes **39/39** on the two 3090 UUIDs and 5060 Ti UUID. There are no
failed or ignored normal tests. The task result records UUIDs, exact commands,
binary hashes and the raw-log manifest under `results/task0013/` (retain through
review and M1 closure).

Independent review found one P2 error-classification issue and otherwise found
the bounded contract satisfied. Valid lineage allocation exhaustion was reported
as `InvalidRequest`. Correction `c9a4b33` returns exact
`CapacityExceeded(HostTier::Pageable)` and adds a one-shot 8,008-byte allocation
failure regression. It proves the outstanding set and total/StateSpill/Pageable
charges all return to zero. No architectural redesign or contract change was
needed.

## Decisions

M1.3 remains complete under the owner's task 0012 acceptance. **M1.4 is active**;
task 0013 implementation is not owner acceptance. O1–O7 remain open, and this work
requires no ruling on them. No default numerical bound, strategy or surface changed.

Paging currently permits one root branch and one fixed, fully charged host pool.
Truncation returns suffix capacity inside that pool; close returns its entire
ledger charge. BF16/FP16/FP32 are encoded bytes without conversion. COW, dynamic
pool growth, sliding reclamation and device/attention integration are explicit
future capabilities. The read-only sequence view prevents counters advancing
without physical rows. Async integration must consume the accepted event leases.

## Remaining hypotheses and blockers

Owner review of task 0013 remains. No actual attention at 32K, model execution,
quality, speed, COW/prefix reuse, sampler history, service or CLI is established.
The support matrix keeps these limits separate. New GPU sanitizer/topology/model
and paired performance lanes are unmeasured for this host storage slice.

## Next task

Review task 0013 against its frozen contract and acceptance evidence first. After
acceptance, define one bounded **M1.4 sampler-history and deterministic
greedy/temperature distribution** task before implementation. Shared sampling
owns distributions/history, and its rollback must consume the existing state
mechanism. Read documents 04/05/07, task 0004, this paging contract, frozen
`sampling.hpp`, `sampling.cpp` and `test_sampling.cpp`, plus existing sampler
oracles. Validate generated-only history, ties, seeded distribution semantics,
abort/replay and pending-token frontiers with tiny independent vocabularies.
Do not invent a model-owned sampler or silently enable unqualified processors.
Generation service and diagnostic CLI remain following bounded M1.4 work.
