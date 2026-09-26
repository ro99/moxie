# Task 0106 — rank chain: one stream wait per step

Status: **proposed** (coordinator, 2026-09-26). It runs after
[task 0102](0102-m6-persistent-tp-rank-chain.md) is accepted. It is the
"0102b" half of the reviewer's second design review of 0102
([record](../evidence/task-0102-design-review-2.md)). Builder Claude Sonnet
`builder`; reviewer Claude Opus `reviewer`. M6 roadmap **M6.1**.

Same rules as 0102 (root, carried files, explicit staging, PCI bus order,
no task numbers in identifiers, `DECISION` on a conflict). Allowed files:
`crates/moxie-executor/src/dense_tp_workers.rs`, the two TP2 test files, and
this task's Result.

## Changes


Changes only the `ChainRun` schedule and the counters:
- `drain` split (M-E).
- Per join: after `group_end`, run `wait_ready(deadline)` and queue the
  conversion or interleave **unconditionally**, into the persistent output.
  No status read, no per-join `wait_stream`.
- After the last stage: one `wait_ready` + `wait_stream`, then read **all**
  status words, then the 0102a tail.
- Test (c): per decode per rank, `ready_waits` delta == joins + 1 and
  `stream_waits` delta == 1, and `command_sequence` delta == 2.
- Rerun (a), (b), (d), (e) unchanged.

Why the split is safe: 0102a is a complete, reviewable ownership change,
covering persistent resources, close, the escape inventory and byte identity,
with 0100's failure semantics untouched. 0102b changes only when the status
is read and has no resource-lifetime change. The per-join lease/range
lifetimes are already "until the final drain" in 0102a.


## Acceptance

The same gates as task 0102.

## Result, filled after work
