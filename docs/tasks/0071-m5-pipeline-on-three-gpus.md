# Task 0071 — pipeline execution on the three GPUs

Status: **open** (coordinator, 2026-09-24, under the owner's auto-mode
delegation). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0071, M5 plan slice 5, third of four tasks (0069 lowering, 0070
  solo worker, **0071 this**, 0072 combined TP2 + TP1 and its report).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `352332f` (task
  0070's acceptance). Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs.
- **Clauses served** (M5 exit gate):
  - "same dense and MoE graph definitions execute … **PP** without model
    edits" — on devices;
  - "no … hidden peer-to-host fallback" — the host-staged handoff is
    declared by the lowering and its bytes are reported;
  - the pipeline half of the combined TP/PP plan (task 0072).

## Facts established before writing (coordinator, 2026-09-24)

- `moxie_plan::lower_pipeline(graph, cuts)` gives `PipelineLowering`, with
  `stages()` (node ranges) and `handoffs()` (one activation per boundary).
  `wavefront(S, M)` gives the `(stage, microbatch)` order. The host test
  `pipeline_stages_are_bit_identical_with_microbatched_prefill`
  (`moxie-cli/tests/tensor_parallel.rs`) is the executable reference for the
  step protocol.
- `moxie_plan::build_stage_graph(graph, None, nodes, output, oracle,
  oracles)` builds a stage. With `part: None`, attention layers are
  renumbered densely and `state_layers` maps each local attention node to its
  original layer. **Each stage owns only its own layers' KV** (document 04:
  explicit state ownership), so each stage worker's `KvGeometry` lists
  exactly its stage's original layers, in order.
- `SoloRankWorker` (task 0070) runs one graph per `step`, shares an open
  transaction across steps, and splits the commit into `prepare_commit` and
  `apply_commit`. `step` blocks for its reply.
- **No overlap.** Strata runs "a contiguous pipeline schedule" on PCIe
  systems and has not implemented the wavefront. Document 04 allows prefill
  microbatch overlap only when causality is preserved. This task dispatches
  **sequentially in wavefront order**, which is correct and causal. Overlap
  is a later performance item, marked `ponytail:` in code.
- Only the 3090 pair has peer access. The 5060 Ti handoff has no peer path,
  so the handoff is staged through the host: the stage's output bytes (the
  `step` reply) become the next stage's input binding. That is the declared
  path, not a fallback.

## Bounded deliverable

- `PipelineWorkers` in `moxie-executor` runs a `PipelineLowering` over one
  `SoloRankWorker` per stage, with microbatched prefill, declared
  host-staged handoffs and an all-or-nothing commit across stages.
- Dense Gemma Shape A and routed Gemma Shape C run on the three GPUs with
  uneven stages. Neither the model nor the graph is edited.
- **Non-goals:**
  - no TP stage (task 0072);
  - no overlap;
  - no change to `DenseRankWorkers`;
  - no model edits.

## Numbered changes

1. **New `crates/moxie-executor/src/pipeline.rs`**, behind
   `#![cfg(feature = "paged-attention-binding")]`, declared and re-exported
   in `lib.rs` beside `SoloRankWorker`:
   ```rust
   pub struct PipelineWorkers { workers: Vec<SoloRankWorker>, lost: Option<Error> }
   pub struct PipelineStep<'w> { /* &'w mut PipelineWorkers, logits: Vec<u8>, handoff_bytes: Vec<u64>, open: bool */ }

   impl PipelineWorkers {
       /// Stage i runs on workers[i]. Refuses fewer than two workers.
       pub fn new(workers: Vec<SoloRankWorker>) -> Result<Self>;
       #[allow(clippy::too_many_arguments)]
       pub fn execute(
           &mut self,
           graph: &Graph,
           lowering: &PipelineLowering,
           oracle: OracleId,
           oracles: &OracleRegistry,
           catalogue: &KernelCatalogue,
           microbatches: &[Range<u64>],
           bindings: &mut dyn FnMut(usize, &StageGraph, Range<u64>) -> Result<Vec<OwnedBinding>>,
           cancel: &AtomicBool,
       ) -> Result<PipelineStep<'_>>;
       /// One entry per stage, each `SoloRankWorker::stats`.
       pub fn stats(&mut self) -> Result<Vec<(u64, u64, usize)>>;
       pub fn close(self) -> Result<()>;
   }
   impl PipelineStep<'_> {
       pub fn logits(&self) -> &[u8];
       /// Host-staged bytes per boundary, summed over microbatches.
       pub fn handoff_bytes(&self) -> &[u64];
       pub fn commit(self) -> Result<Vec<u8>>;
   }
   // Drop for PipelineStep: if still open, abort every stage (errors ignored).
   ```
   - **Checks before any work** (`InvalidRequest`, nothing sent):
     - `lowering.stages().len() == workers.len()`;
     - `microbatches` is nonempty, contiguous, strictly increasing and
       nonempty per range;
     - a sticky `lost` returns that error.
   - **Stage graphs:** build each stage once per `execute` with
     `build_stage_graph(graph, None, stages[i].clone(), Some(output_i),
     oracle, oracles)`. `output_i` is `handoffs()[i]` for every stage except
     the last, and `graph.output()` for the last. The output is **derived
     from the lowering, never inferred** from the stage's last node.
   - **Dispatch:** iterate `wavefront(stages, microbatches.len())`. For each
     `(s, m)`:
     1. If `cancel` is set, abort every stage and return
        `Error::Cancelled { at: "pipeline stage" }`.
     2. Take `bindings(s, &stage_graphs[s], microbatches[m].clone())`.
     3. For `s > 0`, append the handoff binding: the stage read whose
        `original == handoffs()[s - 1]` gets that stage-microbatch's host
        bytes, taken from stage `s - 1`'s output **for microbatch `m`**. The
        role and shape come from that read's spec in the stage graph, with
        the rows symbol evaluated to the microbatch's row count (as
        `value_bytes` does with `Dim::eval`). The layout is
        `ContiguousRowMajorV1`, and the device is the caller's device from
        the other bindings. Add the byte count to `handoff_bytes[s - 1]`.
        A missing read is `InvalidRequest`.
     4. Call `workers[s].step(stage_graph.graph.clone(), catalogue.clone(),
        bindings, rows, visible_tokens = microbatches[m].end)`.
     5. The last stage's output bytes are appended to `logits`. The other
        stages' outputs are kept per microbatch until the next stage
        consumes them, then dropped.
     - On any error from steps 2–4, abort every stage and return the error.
       A worker's own sticky loss makes the pipeline's `lost` sticky too.
     - Mark the sequential dispatch with a `ponytail:` comment: no overlap;
       overlap needs a split send/receive on `SoloRankWorker`.
   - **`commit`:** `prepare_commit` on every stage in order. If any stage
     refuses, abort **every** stage, prepared ones included, and return the
     error. Then `apply_commit` on every stage in order. An apply failure
     after the first apply is a sticky `DeviceLost` for the pipeline, because
     stages may have diverged. Return `logits` and mark the step closed.
   - **`close`:** close every worker, returning the first error after
     trying all of them.
   - **Size:** about 200–300 lines.

2. **`crates/moxie-executor/src/rank_worker.rs`:** under
   `paged-attention-test-hooks`, add `pub fn refuse_next_prepare(&mut self)`.
   The next `prepare_commit` returns `InvalidRequest { field: "fault", .. }`
   without sending a command, exactly as `fail_next_step` works. Add
   `pub fn fail_next_step(&mut self, stage: usize)` and
   `pub fn refuse_next_prepare(&mut self, stage: usize)` on
   `PipelineWorkers` under the same feature, forwarding to that stage's
   worker. There is no other change to `rank_worker.rs`.

3. **`crates/moxie-executor/tests/dense_gemma_device.rs`:** add **one**
   test, `pipeline_runs_dense_and_routed_gemma_on_three_gpus`, reusing the
   file's helpers (`geometry`, `stage_bindings`, `assert_logits`,
   `host_step` or their pieces). Skip it with a printed reason unless the
   3090 pair and the 5060 Ti are all present. For **dense Shape A and
   routed Shape C**:
   - **Lowering:** cuts at the first node of layers 1 and 4 (the host test's
     `first_node_of_layer` rule, copied as a local helper). The stages are
     uneven: layer 0; layers 1–3; layers 4–5 plus the head.
   - **Placement:** stage 0 on 3090 #0, stage 1 on 3090 #1, stage 2 on the
     5060 Ti. Each worker's geometry lists its stage's original layers, via
     `state_layers`, with `tentative_rows` of at least 5.
   - **Steps:** prefill microbatches `[0..2, 2..5]`, commit; decode row 5,
     commit; decode row 6, commit.
     - Logits are within the file's existing ULP tolerance of the host
       reference (`assert_logits`). The comparison is to the host, because
       the stages run on two architectures.
     - `handoff_bytes()` equals, per boundary, rows × hidden × the handoff
       dtype size, summed over the microbatches.
   - **Bit-identity on one architecture:** a two-stage pipeline on the 3090
     pair (cut at layer 3), with the same steps, is **bytewise equal** to one
     `SoloRankWorker` on 3090 #0 running the whole graph with the same
     microbatch steps (0..2 and 2..5 in one transaction, then commit, then
     the decodes).
   - **Failures**, on the three-GPU pipeline, each during the prefill:
     - (a) `fail_next_step(1)`;
     - (b) cancellation set by the bindings closure at `(stage 1,
       microbatch 0)`;
     - (c) `refuse_next_prepare(2)`, so stages 0 and 1 have already
       prepared.

     After each one, the error is the expected kind, and `stats()` equals
     its value before the failed step **on every stage**. The same prefill
     then runs cleanly, and its logits are bytewise equal to the first clean
     run's.
   - **Resources:** after every commit, each stage's outstanding-reservation
     count equals its value right after spawn. `close()` returns `Ok`.
   - Print one `PASS` line per fixture with the handoff bytes.

## Allowed files

- `crates/moxie-executor/src/pipeline.rs` (new)
- `crates/moxie-executor/src/lib.rs` (the module declaration and re-export)
- `crates/moxie-executor/src/rank_worker.rs` (change 2 only)
- `crates/moxie-executor/tests/dense_gemma_device.rs`
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

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):
- `cargo test -p moxie-executor --features
  driver,paged-attention-binding,paged-attention-test-hooks --test
  dense_gemma_device`. The new test plus every existing case must pass.
- `cargo xtask-cuda test-gpu` (no regression).

**Mutations**, each applied, run, shown failing and restored:
1. On a step failure, abort only the failing stage. The every-stage
   stats-unchanged check fails after failure (a).
2. On a prepare refusal, abort only the stages not yet prepared. The
   stats-unchanged check fails after failure (c).
3. Feed stage `s` the handoff of microbatch 0 for every microbatch. The
   bit-identity or tolerance check fails, or the step is refused.

**Stop conditions:**
- A second prefill microbatch in the same open transaction does not see the
  first microbatch's tentative KV rows on the device (the pair pipeline is
  not bit-identical to the one-GPU microbatched reference, and the one-GPU
  microbatched reference differs from the one-step prefill beyond
  tolerance). Report it with the numbers; do not change `moxie-state` or
  the kernels.
- A stage graph's handoff read or output cannot be bound or returned with
  the declared role and shape.
- A mutation survives.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Pending.
