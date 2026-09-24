# Task 0075 — M5 milestone-end audit cleanup

Status: **open** (coordinator, 2026-09-24). The owner chose "cleanup task,
then package" after the milestone-end `/ponytail:ponytail-audit`. Builder
Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0075, M5 plan slice 6: the audit's cleanup, before the acceptance
  package.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs.
- **Every change is behaviour-preserving.** No assertion is removed or
  weakened. If a change cannot be made without changing behaviour, skip it
  and say why in the Result.
- **Naming rule** (the owner's): no task numbers in code identifiers,
  labels, fixture ids or strings. Task numbers belong in `docs/tasks`, the
  ledger and commit messages.

## Facts established before writing (coordinator, 2026-09-24; each signature checked)

- `moxie_graph::TensorSpec::extent(&self, bindings: &SymbolTable) ->
  Result<Vec<u64>>` exists (`graph.rs:229`).
- `moxie-plan`'s private `fn tensor_bytes(role: ValueRole, shape: &[u64]) ->
  Result<u64>` (`lib.rs:785`) sizes every role: BF16/F16 are 2, F32 is 4,
  plus index and route entries, and it refuses integer weights.
- Hand-written copies of that sizing, added in M5:
  - `compare.rs`: `weight_bytes`, `eval_dim` and `last_width`;
  - the executor's `pipeline.rs`: `value_extent`;
  - `dense_tp_workers.rs`: the role-sized output readback at about line 657.
    The FP32 boundary sizing (`value_bytes(..., 4)` in `boundary_bytes`) is
    deliberately role-independent and **stays**.
- Hand-written producer lookups (`graph.nodes().iter().find(|n| n.output ==
  v)`) appear 10 times, in `moxie-executor/src/dense.rs`,
  `moxie-plan/src/selected.rs`, `moxie-interp`, `moxie-engine` and
  `moxie-models/src/gemma4.rs`. `Graph` has no producer accessor.
- The paged-KV writer assembly (adapters, then `&mut dyn` writers, both
  reserved fallibly) appears three times:
  - `paged_attention.rs` `commit_paged_state` (about 4550–4568, using
    `moxie_memory::fallible::with_capacity`);
  - `dense_tp_workers.rs` `worker_commit` (about 1469–1495);
  - `rank_worker.rs` `apply_commit` (about 446–465).
- `lower_pipeline`'s input and weight membership checks
  (`moxie-plan/src/pipeline.rs:83–85`) are dead, because `producers` holds
  only node outputs.
- `compare.rs`'s closed-form decode sum (`sum_stage_decode`, `rank_line`,
  `sum_stage_segment`, `sum_max_pair` and `sum_line`, about 671–742) exists
  only because task 0074's contract demanded a closed form. A loop over the
  generated steps is simpler and fast enough.
- **Left out on purpose:**
  - `dense.rs`'s `HostJoinExtents` sizes from the kernel's
    `ExpertShape::workspace_f32`, while the planner restates the formula,
    because `moxie-plan` may not depend on `moxie-kernels`. Merging them
    would remove the consumer-side check.
  - `rank_worker.rs` and `dense_tp_workers.rs` close their runs
    differently: the solo worker forgets a refused run, and the TP worker
    drops it. Merging them would change behaviour. It is routed to review,
    not cut here.

## Numbered changes

1. **One byte-sizing helper.**
   - In `crates/moxie-plan/src/lib.rs`, add and re-export
     `pub fn value_bytes(graph: &Graph, value: ValueId, rows: u64) ->
     Result<u64>`. It binds `graph.rows_symbol()` to `rows`, calls
     `spec.extent`, then `tensor_bytes(spec.role, &shape)`. A missing spec
     is `InvalidRequest`.
   - Use it for `compare.rs`'s `weight_bytes` (`rows = 1`; weights carry no
     rows dimension).
   - Replace `eval_dim` and `last_width` with `spec.extent(...)` at their
     call sites.
   - Use it in the executor's `pipeline.rs` `value_extent`, for the byte
     count, keeping its activation-role check and returned shape.
   - Use it for `dense_tp_workers.rs`'s role-sized output readback, replacing
     the manual `bits() / 8` of task 0072's change 4.
   - Keep the FP32 `boundary_bytes` path as it is.

2. **`compare.rs` decode total as a loop.** `total = prefill +
   Σ_{t=0}^{generated−1} decode(prompt + t)`, computed with `(0..generated)`
   and the existing per-step stage and transfer functions. Delete the
   closed-form helpers. Mark it `ponytail:` (linear in `generated`).

3. **`Graph::producer`.**
   - In `crates/moxie-graph/src/graph.rs`, add `pub fn producer(&self,
     value: ValueId) -> Option<&Node>`, returning the node whose output is
     `value`.
   - Replace all 10 hand-written lookups listed above with it.

4. **One paged-writer assembly.** In `paged_attention.rs`'s device module,
   add
   `pub(crate) fn with_paged_writers<'ctx, R>(runs: &mut [PagedAttentionRun<'ctx>],
   stream: &Stream<'ctx>, f: impl FnOnce(&mut [&mut dyn PagedKvWriter]) -> Result<R>) -> Result<R>`.
   - It builds the adapters and writers with
     `moxie_memory::fallible::with_capacity` and calls `f`.
   - Use it in `commit_paged_state`, `worker_commit` and `rank_worker.rs`
     `apply_commit`.
   - Where a caller maps a reservation failure to `commit_capacity_error`
     today, keep that mapping at the call site.

5. **Delete the dead membership checks** in `lower_pipeline`
   (`graph.inputs().contains(value) || graph.weights().contains(value)`).

6. **`dense_gemma_device.rs` duplication** (from task 0068's list):
   - `run_prefill_decode` (about 243–359) and the step lifecycle at about
     642–793 share one step helper;
   - the host-weight row encoding (about 1066–1083) reuses
     `stage_bindings`' row encoding (about 210–225).

   Re-locate the line numbers first; they have moved since task 0068.

7. **Renames (naming rule):**
   - `xtask/src/gpu.rs` `mod task_0012_negative_fixtures` becomes a name
     saying what it tests (for example `nonfinite_metric_fixtures`);
   - `xtask/src/archcheck.rs` fixture path `results/task0020/probe` becomes
     `results/example/probe`;
   - every `taskNNNN` / `task_NNNN` / `TASK NNNN` in string labels, `eprintln!`
     and `println!` text and fixture ids (`ArtifactId`, temp-dir prefixes)
     under `crates/` and `xtask/` is reworded to describe the case. Find
     them with `git grep -n -i -E "task_?0?0[0-9]{2}|task[0-9]{2,4}|TASK
     [0-9]{4}" -- 'crates' 'xtask'`.
   - Comments that cite a task for provenance may stay. Code, identifiers
     and runtime strings may not.
   - Report the before and after grep counts.

## Allowed files

- `crates/moxie-plan/src/{lib.rs,compare.rs,pipeline.rs,selected.rs}`
- `crates/moxie-graph/src/graph.rs`
- `crates/moxie-executor/src/{pipeline.rs,dense_tp_workers.rs,rank_worker.rs,paged_attention.rs,dense.rs}`
- `crates/moxie-interp/src/lib.rs`, `crates/moxie-engine/src/lib.rs`,
  `crates/moxie-models/src/gemma4.rs` (producer lookups only)
- `crates/moxie-executor/tests/dense_gemma_device.rs`
- Any file under `crates/` or `xtask/` for change 7's string renames only
- This task's Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**Equivalence:** rerun the four `compare-plans` evidence commands and
compare them with `docs/evidence/plan-comparison.md`.
- The rankings must be identical.
- Printed figures may differ only in the last printed digit, from the change
  in summation order. Otherwise stop.
- If any figure changes, update the evidence tables and say so.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), with the full suites:
- `dense_gemma_device`;
- `dense_tp2_device`;
- `cargo xtask-cuda test-gpu`.

**Line accounting:** the Result gives per-change net lines and the total
(the audit estimated about −250 to −350 for this subset).

**Stop conditions:**
- Any change needs a behaviour change or a weakened assertion (skip it and
  report).
- The rankings change.
- A GPU suite fails.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Pending.
