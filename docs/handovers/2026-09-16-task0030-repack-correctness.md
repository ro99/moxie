# Handover — task 0030: M3.1 correctness package, reviewed, battery unmeasured

## Workspace identity

- `/home/rodrigo/Developer/moxie`, branch `main`, commit `fd3ef46` pushed to `origin/main` (verified equal rev-parse).
- Base was `0d7fe63`. Tree clean at handover.
- Legacy root `/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, untouched.
- `/models` and `/fast/models` read-only; one Laguna module published into scratch and removed.
- GPUs identified by UUID where used; no GPU launch was made on this tree by this task.

## Completed facts

- Task 0030 implemented and independently reviewed once plus a focused re-review of four findings; all six reviewer findings closed (P1 survivor recorded as product-gate gap, P1 control, P1 acceptance wording, P2 header scope, P1 T0028 path, P3 arithmetic).
- `budget.rs` isolation repaired: `Session` holds the process-wide counter from before the fixture to after destruction; `TILE_BOUND` single constant; per-thread `SESSION_HELD` witness; second outside-session control. Measured 20/20 parallel plus 12/12 under 56-way load at identical 209,256 B peak; pre-repair shape reproduced 15/20.
- `G-REPACK`, experiment 0006, engineering log reconciled; 0025/0026 headers corrected to implemented-not-accepted; 0027 deferral preserved; experiment 0007 pending.
- Gates on final tree: fmt clean; both clippy lanes clean; spec-check 10; arch-check 79/21/13; self-test 109/109 over 94 anchors; host 1,108 plus device-feature 1,153 across 101 suites, 0 failed/ignored/skipped. xtask cuda clippy lane fails to compile (pre-existing E0521 at `xtask/src/gpu.rs:595`, task 0029 `Label` narrowing).
- **Not run**: full T0006 end to end on the final tree (diagnostic pass stopped at 15/66 by coordinator order; `published-validation-skipped` survived where `ccfd7fa` records it caught); full T0028; `cargo xtask-cuda test-gpu` (blocked on the compile above). All recorded as unmeasured, never passing.

## Decisions

- Owner ruling adopted: M3.1 correctness before expert-MoE authorization; expert contract paused; T0006 re-run does not gate retention (experiment 0007 still pending).
- `coordinator.md` pick-next rule corrected per ruling: NEXT from Results plus matrix plus handovers plus roadmap/ADRs/rulings; progressive default with recorded owner exceptions (0027 stands); coordinator owns the recommendation.
- Compliant mechanisms delegated; requirement weakening (residual abort, API narrowing, whole-model as completion, retention before 0007) stays owner-reserved.

## Remaining hypotheses and blockers

- T0006 full re-run (~4 h) on `fd3ef46` plus a regression for `published-validation-skipped`, which survived on this tree but was caught at `ccfd7fa`.
- T0028 full battery (runnable, lanes avoid the xtask device build) plus `test-gpu` after the `Label`/E0521 owner decision.
- Owner acceptance of 0025/0026 within the demonstrated scope; disposition text is a recommendation.
- Task 0029 clauses 1/2/7, M3.2 group-128 symmetric plus AutoRound/AutoGPTQ, and the `Rc`/`Label` decisions are untouched by this task.

## Next task

- Bounded: full T0006 re-run on `fd3ef46` with the survivor regression, then T0028 plus affected-consumer gates. Owner `moxie-repack`, allowed files `crates/moxie-repack/**`, `xtask/src/mutationtable.rs` for anchors only, plus experiment 0006 and task 0030 Result. Oracle: battery 66 substitutions all caught, 0 unstable, tree clean. Stop at any tolerance or retention question.
