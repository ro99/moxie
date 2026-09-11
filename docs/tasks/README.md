# Task contracts

## Current assignment and milestone handoff

**M1.3 complete; M1.4 active.** The owner accepted
[task 0012](0012-m1-selected-bf16-device-chain.md) on 2026-09-10, including corrections through
`1138a2a` and their evidence at `ce7548d`.

[Task 0013](0013-m1-appendable-paged-state.md) is accepted after independent review
through `0f39cf5`. Contract `a80ff2c` preceded implementation `c14e32a`; correction
`c9a4b33` fixes the sole reported error-classification issue.

The active bounded M1.4 assignment is
[task 0014](0014-m1-transactional-sampler-history.md), implemented after contract
`4054dd7` and awaiting independent review: sampler history and deterministic greedy/temperature distribution,
using task 0004's accepted transaction mechanism for
history rollback. Generation service, diagnostic CLI and device attention remain later
scope. Reopen M1.3 or task 0013 only for a demonstrated defect in accepted scope.

One file per bounded assignment, `NNNN-slug.md`. Use [TASK.md](../spec/templates/TASK.md).

A task produces one primitive, one planner capability, one state transition, one importer rule, one
integration slice, or one controlled experiment. It is not "rewrite GLM". Fill in the Result section
after the work, keeping passed, failed and skipped separate.
