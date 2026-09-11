# Task contracts

## Current assignment and milestone handoff

**M1.3 complete; M1.4 active.** The owner accepted
[task 0012](0012-m1-selected-bf16-device-chain.md) on 2026-09-10, including corrections through
`1138a2a` and their evidence at `ce7548d`.

[Task 0013](0013-m1-appendable-paged-state.md) implements **M1.4 appendable host
paged state bound to task 0004's accepted transactions**, with owner review pending.
Contract `a80ff2c` preceded implementation `c14e32a`. After review/acceptance,
sampler history, generation service and diagnostic CLI remain within M1.4.
Reopen M1.3 only for a demonstrated defect in its accepted scope.

One file per bounded assignment, `NNNN-slug.md`. Use [TASK.md](../spec/templates/TASK.md).

A task produces one primitive, one planner capability, one state transition, one importer rule, one
integration slice, or one controlled experiment. It is not "rewrite GLM". Fill in the Result section
after the work, keeping passed, failed and skipped separate.
