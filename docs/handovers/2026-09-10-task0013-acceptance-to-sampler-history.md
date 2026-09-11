# Handover — task 0013 acceptance to sampler history

Superseded by [task 0014's implementation handover](2026-09-11-task0014-sampler-history.md).

## Workspace identity

Writable root `/home/rodrigo/Developer/moxie`, branch `main`. Task 0013 contract
`a80ff2c` preceded implementation `c14e32a`; validation record `29547c7`, correction
`c9a4b33` and final review evidence `0f39cf5` follow it. The tree was clean before
this acceptance record. The owner authorized documenting acceptance and pushing the
complete sequence to `origin/main`.

Read-only legacy `/home/rodrigo/Developer/strata` remains at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its pre-existing untracked `.pi/` and
`tests/p2p/` remain untouched. No checkpoint was read or changed.

## Completed facts

The owner accepted [task 0013](../tasks/0013-m1-appendable-paged-state.md) on
2026-09-10 after independent review through `0f39cf5`. The bounded implementation
provides one fixed, admitted host KV pool in `moxie-memory` and page
addressing/transaction composition in `moxie-state`. It uses task 0004's existing
transaction mechanism and one process-unique transaction-ID source.

The sole review issue was valid lineage allocation exhaustion being classified as
an invalid request. Correction `c9a4b33` reports `CapacityExceeded` for the exact
pageable bytes and releases backing, reservations and both tier charges before
returning. The reviewer proved the regression fails against the previous
implementation specifically on `invalid_request` versus `capacity_exceeded` and
passes after correction. No paging corruption, transaction-isolation defect,
competing resource owner or remaining code-review blocker was found.

Independent final review passed state/memory tests, the allocation regression under
device features, formatting, host clippy, architecture and specification checks.
The retained full-workspace and aggregate GPU logs and hashes matched their recorded
values; those full runs were not repeated in the final review. The accepted recorded
results remain 570 host tests + 8 doctests, 586 device-feature tests + 11 doctests,
39/39 aggregate real-GPU cases, and the 32,768-row allocation gate with complete
cleanup. No normal test failed or was skipped.

## Decisions

M1.3 remains complete. Task 0013 is closed and **M1.4 remains active**. O1–O7 remain
open and did not block this synthetic host-state slice. Fixed capacity, one root
branch, >=16-bit storage, and storage-only 32,768-row evidence remain its accepted
limits. It establishes no device attention, model context, quality or performance
claim.

The next task owns sampler history and deterministic greedy/temperature distribution.
Shared sampling owns the distribution and history. History publication, cancellation
and rollback must compose with task 0004's accepted transaction mechanism; no sampler
may create a competing journal or advance state independently.

## Remaining hypotheses and blockers

The sampler-history contract is now recorded in
[task 0014](../tasks/0014-m1-transactional-sampler-history.md) (2026-09-11),
before implementation. Existing
`moxie-oracles::sampler` coverage is partial: legality, top-k, top-p, min-p,
temperature, ties, extreme-temperature stability and typed failures exist, while
penalties, DRY, n-gram bans, logit bias, typical-p and XTC remain unimplemented. The
next task must keep unsupported processors explicit and must not claim service, CLI,
device attention, model generation, quality or performance integration.

No owner ruling is currently required for the bounded synthetic sampler task. Stop
before any conclusion that depends on O1–O7, changes a public generation default,
silently enables an unqualified processor or introduces model-owned sampling.

## Next task

Implement [task 0014](../tasks/0014-m1-transactional-sampler-history.md) after its
contract commit: bounded **M1.4 sampler history and deterministic
greedy/temperature distribution**. Read documents 04,
05 and 07; tasks 0004 and 0013; the current sampling oracle; and frozen legacy
`sampling.hpp`, `sampling.cpp` and `test_sampling.cpp`. Name the shared owner,
history/resource/cancellation contract, consumers, allowed files, deletion plan and
fixed acceptance thresholds.

The minimum independent consumers use tiny vocabularies to cover generated-only
history, stable ties, zero-temperature greedy selection, seeded positive-temperature
distribution, invalid/non-finite logits, pending-token frontiers, abort/replay and
partial acceptance. Require exact history/frontier restoration through the existing
transaction ID. Stop at the task gate if deterministic distribution semantics cannot
be fixed from the normative contracts and source evidence without an owner decision.
Generation service, diagnostic CLI and device attention follow as separate bounded
M1.4 work.
