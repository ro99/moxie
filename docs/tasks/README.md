# Task contracts

## Current assignment and milestone handoff

[Task 0012](0012-m1-selected-bf16-device-chain.md) is the **final planned M1.3 assignment**. Its
implementation is at `6305f9d`; all three owner-review findings are corrected at `eaf8846` with the
affected evidence rerun. Owner re-review/acceptance remains open. Corrections remain part of task
0012 rather than becoming another preparatory M1.3 task.

When task 0012 is accepted, record **M1.3 complete**. The next bounded assignment must be **M1.4:
appendable paged state bound to task 0004's accepted transaction mechanism**. Subsequent sampler,
generation-service and diagnostic-CLI assignments remain within M1.4. Reopen M1.3 only for a
demonstrated defect in its accepted scope.

One file per bounded assignment, `NNNN-slug.md`. Use [TASK.md](../spec/templates/TASK.md).

A task produces one primitive, one planner capability, one state transition, one importer rule, one
integration slice, or one controlled experiment. It is not "rewrite GLM". Fill in the Result section
after the work, keeping passed, failed and skipped separate.
