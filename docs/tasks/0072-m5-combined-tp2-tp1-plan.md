# Task 0072 — a combined TP2 + TP1 pipeline plan, and its report

Status: **accepted** (coordinator, 2026-09-24, under the owner's auto-mode
delegation), after sol's review rounds R1 and R2. Builder Codex `luna`;
reviewer Codex `sol`.
- Change 4 was added after the builder's DECISION: the TP output readback
  is sized from the output's role.
- R1 (HIGH, test harness): the TP2 gate's deliberate stall leaves lost pair
  workers parked, and their context claims stay pinned for the rest of the
  process (correct fail-closed behaviour). A combined test that runs after it
  cannot claim the 3090s. Fixed by running the combined case first inside the
  file's single `#[test]`.
- **M5.3 verdict, at fixture scale:**
  - **capacity improves**: the peak per-GPU weight bytes fall from 134,544
    to 71,472;
  - **latency worsens**: decode goes from about 35 ms to 184 ms, about 5×.
- **Size, carried to the milestone-end ponytail audit:**
  `dense_tp2_device.rs` grew by +717/−18 lines for one test. The R1
  deduplication removed only about 11 lines net.

## Identity and authority

- Task0072, M5 plan slice 5, the last of four tasks (0069 lowering, 0070
  solo worker, 0071 pipeline, **0072 this**).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `048ac8b` (task
  0071's acceptance). Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs.
- **Clauses served:**
  - M5 exit: "at least one **combined TP/PP plan** … correctness, resource
    and cancellation tests";
  - M5 exit: "no … hidden peer-to-host fallback". The pair's collectives
    stay peer, and the pipeline handoff is declared and its bytes reported;
  - M5.3: "support stage-local rank groups, then test a TP2 stage plus TP1
    stage on the mixed hardware where feasible. **Report whether it improves
    capacity, latency, both or neither.**"

## Facts established before writing (coordinator, 2026-09-24)

- `DenseRankWorkers::execute_dense(graph, &TensorParallelLowering, oracle,
  oracles, catalogue, rows, visible_tokens, bindings: FnMut(rank,
  &StageGraph), cancel)` runs a whole TP2 transaction on the pair and returns
  a `DenseWorkerStep` with `logits()` and `commit()`. `commit` is the pair's
  own all-or-nothing two-phase commit. Dropping an uncommitted step aborts
  both ranks. The pair checks cancellation at its stage boundaries itself.
  The TP worker has no separate prepare and apply, and this task adds none.
- `lower_tensor_parallel(graph, 2)` takes any graph. It does not require an
  embedding or a head. The TP worker lowers each TP stage with
  `lower_selected_ordered`, which does not require a complete graph (the
  same fix as task 0071's amendment). Its KV state uses `state_layers`,
  indexed by the input graph's own dense local layers. For a pipeline stage
  graph, that is the stage's own layers.
- A **pipeline stage graph can be the TP worker's input graph.** The TP
  sub-stage reads' `original` ids are then pipeline-stage-local ids. The
  bindings caller maps them to the full graph through the pipeline stage
  graph's `reads` and `weights`. That mapping stays in the test; production
  code never slices weights.
- `PipelineWorkers` (task 0071) owns `SoloRankWorker`s, dispatches
  sequentially in wavefront order, and commits all-or-nothing through
  `prepare_commit`/`apply_commit`.

## Bounded deliverable

- `PipelineWorkers` accepts a **TP2 pair as stage 0**, followed by solo
  stages. The combined plan is dense Gemma Shape A, with layers 0–2 on the
  3090 pair (TP2) and layers 3–5 plus the head on the 5060 Ti (TP1). It
  runs within host tolerance, with an all-or-nothing commit and fault,
  cancellation and resource tests.
- A measured capacity and latency report, recorded in the Result, answering
  M5.3's question.
- **Non-goals:**
  - a TP stage anywhere but first;
  - microbatches when a pair stage is present (one microbatch is required,
    and anything else is refused);
  - routed Gemma in the combined plan (one combined plan meets the exit
    clause; expert-owner TP is already gated by task 0066);
  - any change to `DenseRankWorkers` or `SoloRankWorker` beyond change 4;
  - overlap;
  - speed claims.

## Numbered changes

1. **`crates/moxie-executor/src/pipeline.rs`:**
   - **Stages:**
     ```rust
     pub enum PipelineStageWorker { Solo(SoloRankWorker), Pair(DenseRankWorkers) }
     ```
     `PipelineWorkers::new(stages: Vec<PipelineStageWorker>)` refuses
     (`InvalidRequest`) a `Pair` anywhere but index 0, and fewer than two
     stages. Store it as `pair: Option<DenseRankWorkers>` plus
     `solos: Vec<SoloRankWorker>`, so that `PipelineStep` can hold the
     pair's `DenseWorkerStep<'w>` and `&'w mut` solos through disjoint field
     borrows.
   - **Bindings:** the callback becomes
     `&mut dyn FnMut(StageBindings<'_>) -> Result<Vec<OwnedBinding>>` with
     ```rust
     pub struct StageBindings<'a> {
         pub stage: usize,
         pub graph: &'a StageGraph,                 // the pipeline stage graph
         pub rank: Option<(usize, &'a StageGraph)>, // pair stage: (rank, TP sub-stage graph)
         pub rows: Range<u64>,
     }
     ```
     Update task 0071's test call sites mechanically (`rank: None`).
   - **Pair stage dispatch:**
     1. With a pair present, `microbatches.len() == 1` or `InvalidRequest`.
     2. Derive the TP lowering **inside `execute`** with
        `lower_tensor_parallel(&stage_graphs[0].graph, 2)`. A refusal
        becomes `InvalidRequest { field: "pipeline", .. }`. No TP lowering
        is ever accepted from the caller.
     3. Call the pair's `execute_dense(&stage_graphs[0].graph, &tp, oracle,
        oracles, catalogue, rows, rows.end, adapter, cancel)`. The adapter
        forwards to the callback with `rank: Some((rank, sub_stage))`.
     4. `logits()` is stage 0's output. It feeds stage 1's handoff exactly as
        a solo stage's output does (same extent check, same `handoff_bytes`).
     5. Keep the `DenseWorkerStep` in the returned `PipelineStep`.
     - On any later error or cancellation, drop the pair step (which aborts
       the pair) and abort every solo stage.
   - **Commit, with a pair present:**
     1. `prepare_commit` on every solo stage. On a refusal, drop the pair
        step and abort every solo stage.
     2. `pair_step.commit()`. On a failure, abort every solo stage (the
        prepared ones included) and return the error.
     3. `apply_commit` on every solo stage. A failure is a sticky pipeline
        `DeviceLost`, because the pair has already committed.
   - `Drop` for an open step: drop the pair step and abort every solo stage.
   - `stats()` covers the solo stages only. Add `pub fn
     pair_frontiers(&self) -> Option<([u64; 2], [u64; 2])>`, forwarding the
     pair's `published_frontiers()` and `committed_frontiers()`.
   - **Test hooks:** add `refuse_next_pair_commit(&mut self, rank: usize)`,
     forwarding to `refuse_next_commit_prepare`. Solo-stage indices in the
     existing hooks count stages, so with a pair present stage 1 is
     `solos[0]`.
   - **Size:** about +120–180 lines.

2. **`crates/moxie-executor/tests/dense_tp2_device.rs`:** add **one** test,
   `combined_tp2_tp1_pipeline_matches_host_and_reports_capacity_latency`,
   reusing the file's helpers (`local_config`, `geometry`,
   `stage_weight_value`, `host_logits`, `bf16_ulp`, `pair_ordinals`). Skip
   with a printed reason unless the 3090 pair and the 5060 Ti are present.
   - **Plan:** dense Gemma Shape A, with one cut at the first node of layer 3
     (a local `first_node_of_layer` helper). Stage 0 runs on the pair
     (`DenseRankWorkerConfig` with a stage-local, head-halved geometry for
     layers 0–2). Stage 1 runs on the 5060 Ti (a `SoloRankWorker` with the
     geometry for layers 3–5).
   - **Bindings:** map a TP sub-stage read or weight → the pipeline stage
     graph's read or weight (by local id) → the fixture's original, then
     apply the TP sub-stage's `rows` or `slice` with `stage_weight_value`'s
     rule.
   - **Correctness:** prefill of 5 rows (one microbatch), commit; decode
     row 5, commit; decode row 6, commit. The logits are within the file's
     ULP gate of the host reference. `handoff_bytes()` equals rows × hidden ×
     the dtype size.
   - **Failures**, each during the prefill, then a clean retry whose logits
     are bytewise equal to the first clean prefill's:
     - (a) `fail_next_step(1)`, after the pair ran;
     - (b) cancellation set by the callback at rank 1's layer-1 TP sub-stage
       (the file's existing trigger);
     - (c) `refuse_next_pair_commit(1)`, after the solo stage prepared;
     - (d) `refuse_next_prepare(1)`, so the pair step is dropped
       uncommitted.

     After each one, the solo `stats()` and `pair_frontiers()` equal their
     values before the failed step.
   - **Refusals:** `new` with a pair at index 1 is `InvalidRequest`, and
     `execute` with two microbatches and a pair is `InvalidRequest`, with no
     command sent (the stats are unchanged).
   - **Resources:** after every commit, the solo stage's outstanding count
     equals its spawn value. `close()` returns `Ok`.
   - **Report (measured, printed, not asserted):**
     - **Capacity:** per GPU, the sum of the weight-role binding bytes the
       plan places there, for (i) one 3090 running the whole graph through a
       `SoloRankWorker`, (ii) the combined plan, and (iii) task 0071's
       three-stage PP layout, computed from its stage graphs, with no
       execution needed.
     - **Latency:** the median of 8 decode steps (wall clock, each including
       its commit) for (i) and (ii).
     - Print one table.

3. **The task Result** records the table and a one-paragraph verdict:
   capacity, latency, both or neither. State that this is fixture scale and
   makes no speed guarantee (the M5 exit forbids a TP3 speed claim).

4. **Amended 2026-09-24, after the builder's DECISION:**
   **`crates/moxie-executor/src/dense_tp_workers.rs`**, `execute_dense`'s
   output readback only.
   - The problem: `value_bytes(graph, graph.output(), rows, 4)` assumes FP32
     logits. A headless stage outputs a BF16 activation, and its admitted
     output range is 2 bytes per element, so the readback is refused.
   - The fix: take the element size from the output spec. It is
     `ActivationPrecision::get().bits() / 8` for an `Activation` role, and 4
     for any other role (so a full graph's output is unchanged).
   - No other change to this file. The finalize declaration stays as it is.
   - Stop if the existing TP2 gate's bytes change.

## Allowed files

- `crates/moxie-executor/src/pipeline.rs`
- `crates/moxie-executor/src/lib.rs` (re-exports)
- `crates/moxie-executor/tests/dense_tp2_device.rs`
- `crates/moxie-executor/src/dense_tp_workers.rs` (change 4 only)
- `crates/moxie-executor/tests/dense_gemma_device.rs` (the mechanical
  `StageBindings` call-site update only)
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

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), with the full suites:
- `dense_tp2_device` (the new test plus the existing TP2 gate);
- `dense_gemma_device` (task 0071's pipeline test, after the call-site
  update);
- `cargo xtask-cuda test-gpu`.

**Mutations**, each applied, run, shown failing and restored:
1. On a pair-commit refusal, do not abort the prepared solo stage. The
   stats check after (c) fails.
2. On a solo prepare refusal, commit the pair step instead of dropping it.
   `pair_frontiers()` after (d) differs.
3. Accept a pair at index 1. The refusal check fails.

**Stop conditions:**
- `lower_tensor_parallel` refuses the stage-0 graph, or the TP worker
  refuses to run a graph without a head. Report the refusal; do not change
  `moxie-plan` or `DenseRankWorkers`.
- Combined-plan logits fall outside the ULP gate.
- A mutation survives.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

The combined Shape A plan ran layers 0–2 on the 3090 pair and layers 3–5
plus the head on the 5060 Ti. Prefill and decode matched the host reference
within the declared ULP gate; the declared handoff was 240 bytes per decode
row set. The existing TP2 gate remained byte-exact after change 4.

Capacity and latency report (fixture-scale weight-role binding bytes; GPU
identity is by UUID):

| GPU | One 3090, whole graph | Combined TP2+TP1 | Task 0071 three-stage PP |
|---|---:|---:|---:|
| RTX 3090, `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 134,544 | 32,544 | 21,600 |
| RTX 3090, `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 0 | 32,544 | 63,072 |
| RTX 5060 Ti, `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 0 | 71,472 | 50,448 |

The two R1 reruns measured these medians for eight decode steps including
commit: **38.024 ms** and **35.880 ms** on one 3090; **186.418 ms** and
**184.975 ms** for combined TP2+TP1. This fixture shows a lower peak per-GPU
weight-binding footprint (134,544 to 71,472 bytes), so capacity improves;
decode latency is about 5× slower. The measurements are fixture scale, make
no speed guarantee, and do not support a TP3 speed claim.

Acceptance:

- Host: `cargo fmt --all -- --check`, workspace clippy, driver-feature
  executor clippy, `cargo test --workspace --locked`, `cargo xtask arch-check`
  (79 rejected fixtures, 21 accepted fixtures, 13 rules), and
  `cargo xtask spec-check` all passed.
- GPU (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_tp2_device` passed twice
  with the default runner (one test, containing the combined case before the
  two TP2 gates); `dense_gemma_device` passed 6/6 tests across all three GPUs; and
  `cargo xtask-cuda test-gpu` passed all 63 cases on SM86 and SM120. An
  earlier default-runner failure from R1 was resolved by running the combined
  case before the sticky-stall gate. The existing TP2 gate's bytes remain
  unchanged.
- Mutations were applied, observed failing, and restored before the final
  gates: skipping solo abort after pair-commit refusal failed the solo stats
  invariant; committing the pair after solo-prepare refusal changed pair
  frontiers; and allowing a pair at stage 1 failed the placement refusal.

### R1 fixes (2026-09-24)

- Moved the combined scenario into a plain function called first by the sole
  `dense_tp2_device` test. The routed stall gate runs last because it pins the
  pair's device claims for the process.
- Folded TP sub-stage input and weight ID mapping into `stage_host_bindings`,
  removed the one-use `CombinedRun`, and summed report bytes from the
  weight-role bindings constructed by the same callback and stage helper.
  The report rows now use one UUID-keyed loop. The two 3090 combined-plan
  values are 32,544 bytes each after measuring those bindings.
- Host gates passed again: format check, both Clippy gates and all workspace
  tests. The full `dense_tp2_device` binary passed twice with default test
  runner settings. No other GPU suites were rerun in R1.
- R1 line-count delta: `dense_tp2_device.rs` **+251/-262** (net −11 lines).
  Total task diff for that file is **+717/-18** from the task base.
