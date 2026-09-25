# Task 0078 — fixed-plan timing of the single-GPU dense step

Status: **active** (coordinator, 2026-09-24). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0078, M6 slice 3 (benchmark harness), moved ahead of slice 1's
  remaining work by the coordinator (ledger, 2026-09-24): the remaining
  slice-1 candidates (per-operation `settle` in `PagedAttentionRun`, the
  per-step `Module::load` in `execute_dense_stage`) need timing evidence to
  be ranked. Serves roadmap **M6.1** and the M6 exit's paired-benchmark
  method.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. If it
  conflicts with the code, stop and send `DECISION`; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.** No `git stash`, no `git worktree`.
- **GPUs are free**; the builder is the only GPU user and the only writer in
  the tree. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels or strings.
- **O6 is open.** The output is fixture-scale evidence for choosing the next
  task, not a performance claim. The evidence file must say so.

## Facts established before writing (coordinator, 2026-09-24)

- `dense_gemma_device.rs` `run_prefill_decode` (about 242) shows the whole
  step cycle: `SelectedReservedPlan::admit`, `state.begin()`,
  `plan.execute_dense(DenseGraphStep { … })`, `.finish()`, then commit and
  `result.plan.close(…)`. `finish` returns `DenseGraphResult { plan,
  returned_inputs, … }`: the admitted plan and the non-weight bindings come
  back and can be executed again.
- The main test builds Shape A's prompt, positions, workloads and bindings
  (about 780–880); `stage_bindings` (135) and `admit_runs` (625) are the
  helpers.
- `nsys` is installed (`/usr/local/bin/nsys`).

## Bounded deliverable

- **Outcome:** one ignored test that times a fixed, admitted dense plan's
  `execute_dense` + `finish`, for prefill and for decode, on one 3090; and an
  evidence file with base-versus-candidate numbers and a CUDA API breakdown.
- **Allowed files:** `crates/moxie-executor/tests/dense_gemma_device.rs`
  (one new test, **appended at the end of the file**, reusing existing
  helpers; no helper changes), new `docs/evidence/dense-step-timing.md`, this
  task's Result.
- **Non-goals:** no production code change; no xtask command; no new
  helper module; no other GPU; no change to any existing test.

## The test

`#[test] #[ignore = "timing harness; run explicitly"] fn dense_step_timing()`:

1. `one_at_a_time()`. Select the 3090 whose UUID is
   `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` by scanning `query_device(i)`
   for `i in 0..device_count()`; panic if absent.
2. Shape A, with the same prompt, positions, workloads and bindings the main
   test builds for Shape A (5 prompt rows; decode 1 row at position 5).
   Admit the runs, the prefill plan and the decode plan **once**.
3. **Prefill:** repeat `W = 5` warm-up plus `R = 50` measured times:
   `state.begin()`; start an `Instant` immediately before `execute_dense`;
   stop it after `finish()` returns; `state.abort(txn)`; keep
   `result.plan` and `result.returned_inputs` for the next repetition.
4. Run one prefill and **commit** it (as `run_prefill_decode` does). Then
   **decode:** the same `W + R` loop on the decode plan, aborting each
   transaction so every repetition decodes the same position.
5. For each phase print one line with `eprintln!`:
   `dense-step-timing phase=<prefill|decode> gpu=<uuid> warmup=5 reps=50
   median_us=<> min_us=<> max_us=<>`.
6. Close the plans and runs; assert `ledger.outstanding().is_empty()`.
   No output assertion: correctness is the other tests' job.

If `abort` after `finish` or plan reuse is refused, stop and send
`DECISION` with the error.

## Measurements

Command (release build, the GPU gate's features):
`CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test --release -p moxie-executor
--features <as the GPU gate> --test dense_gemma_device dense_step_timing --
--ignored --nocapture`.

1. **Candidate:** run it three times at the task's commit.
2. **Paired bases:** commit the test first. Then, for `6b8239c` (before task
   0076) and `97a7281` (after 0076, before 0077): `git checkout --detach
   <commit>`; apply only the new test with `git diff <parent>..<test commit>
   -- crates/moxie-executor/tests/dense_gemma_device.rs | git apply`; run it
   three times; `git checkout -- crates/moxie-executor/tests/dense_gemma_device.rs`;
   `git checkout main`. Before and after, `git status --short` must show only
   the carried files. If the apply fails, stop and report.
3. **Breakdown:** at the candidate, one `nsys profile -t cuda --stats=true`
   run of the same command, output under `/tmp`. Record the top CUDA driver
   API calls by total time (at least `cuModuleLoadData*`, `cuEventSynchronize`,
   `cuStreamSynchronize`, `cuMemcpy*`, `cuLaunchKernel`) with call counts, per
   step if nsys makes that derivable.

`docs/evidence/dense-step-timing.md`: GPU UUID, driver version, commits,
command, the table (commit × phase × median/min/max for each of the three
runs), the nsys table, and a first line stating "fixture-scale, O6 open, not a
performance claim".

## Acceptance

- Host gates: `cargo fmt --all -- --check`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; the executor driver-feature clippy
  from task 0077; `cargo test --workspace --locked`.
- `dense_gemma_device` full suite passes (the new test is ignored there).
- The evidence file exists with every item above; the tree is back on
  `main` with only the carried files dirty.
- No stop conditions beyond those named above.

## Result, filled after work

(pending)
