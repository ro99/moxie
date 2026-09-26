# Task 0107 — persistent pipeline: stage plan sets, resident weights (incl. INT8/INT4), NCCL hand-offs

Status: **proposed** (coordinator, 2026-09-26). Starts **after task 0102 is
accepted**, because it reuses 0102's `Ledger::admissions`,
`validate_embedding_token_sources` and in-memory `ChunkSource`. Builder
Claude Sonnet `builder`; reviewer Claude Opus `reviewer`. The reviewer
drafted it ([record](../evidence/persistent-pipeline-research-and-draft.md));
it gets a design check against 0102's accepted code before assignment.

## Identity and authority

- M6 roadmap **M6.1**, the multi-GPU line after 0102. It is on the M6 exit's
  critical path: both Gemma placements and Qwen's are pipelines (ledger,
  "Gemma exit decisions", "Qwen decisions").
- **P1 (owner, 2026-09-26):** NCCL carries pipeline hand-offs. ADR 0039 is
  amended to match.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** Always `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
  GPUs are named by UUID. No task numbers in code identifiers. On a
  conflict with the code, report `DECISION`; do not redesign.

## Outcome
After one `load_persistent`, a 2- or 3-stage **solo** pipeline decode or
prompt step:
- makes **zero ledger admissions and zero weight uploads**;
- hands activations between stages over NCCL send/recv on the stage streams,
  **with no host `Vec<u8>` handoff**;
- has exactly **two** coordinator command rounds per step (`PrepareStep`,
  `RunStep`);
- has **one CUDA stream wait per stage worker**, plus one NCCL Ready poll per
  NCCL group;
- returns logits **byte-identical** to the single-GPU run and to the M5
  pipeline path.

It also runs **affine INT8/INT4 weights resident** in a stage. KV commit is
unchanged.

**Not in scope:**
- a TP pair as a pipeline stage (stays on M5's path);
- multiple microbatches per step (M6.3);
- capture (0103);
- timing.

## Allowed files
- `crates/moxie-cuda/src/nccl.rs`: `send`/`recv` and their non-`nccl`
  stubs;
- `crates/moxie-plan/src/selected.rs`: `lower_stage_with_formats`;
- `crates/moxie-executor/src/dense_set.rs`: drop the formats refusal; add
  `step_stage`;
- `crates/moxie-executor/src/chain.rs`: extract `validate_affine_scales`;
- `crates/moxie-executor/src/residency.rs`: `WeightImages`, moved from
  0102;
- `crates/moxie-executor/src/dense_tp_workers.rs`: only the move out and
  `admit_nccl_reserve` to `pub(crate)`;
- `crates/moxie-executor/src/{rank_worker.rs,pipeline.rs}`;
- `crates/moxie-executor/tests/dense_gemma_device.rs` (new tests);
- this task's Result.

## Numbered changes

1. **NCCL point-to-point** (`nccl.rs`, beside `all_gather`, 363):
   - `pub unsafe fn send(&self, address: u64, count: usize, dtype: DataType, peer: u32, stream: &Stream<'ctx>) -> Result<()>`
     and a matching `recv`, wrapping `ncclSend`/`ncclRecv`;
   - `validate_buffer` as `all_gather` does;
   - the same `check_collective` in-group rule;
   - stubs in the non-`nccl` module (489-600) returning the same
     "NCCL feature disabled" error the others do.
2. **Stage lowering with formats** (`selected.rs`):
   `pub fn lower_stage_with_formats(graph, workload, capability, catalogue, formats: &BTreeMap<ValueId, WeightFormat>) -> Result<SelectedPlanCandidate, Error>`.
   It is identical to `lower_selected_with_formats` except that it passes
   `require_complete_graph = false`.
3. **`DensePlanSet` formats and stage step** (`dense_set.rs`).
   - Delete `|| !candidate.weight_formats().is_empty()` (175). The refusal
     text becomes "resident plan sets require matching dense candidates
     without host expert joins".
   - Add a stage step:
     ```rust
     pub(crate) fn step_stage(
         &mut self,
         rows: u64,
         step: DenseSetStep<'_, 'ctx>,
         resident: &BTreeSet<ValueId>,
         before_launch: &mut dyn FnMut(&SelectedReservedPlan<'ctx>, &Stream<'ctx>) -> Result<()>,
         after_launch: &mut dyn FnMut(&SelectedReservedPlan<'ctx>, &Stream<'ctx>) -> Result<()>,
     ) -> Result<DenseSetOutput>
     ```
     Body, in order:
     - the same lease revalidation as `step` (309-326);
     - `before_launch(&plan, stream)`;
     - `plan.execute_dense_stage(step, resident, &BTreeMap::new())`;
     - `after_launch(lease.resource().plan.as_ref().unwrap(), stream)`;
     - `finish()`.
   - Error handling exactly as `step`: a pre-launch error puts the plan back;
     a held lease goes to `self.lost`.
   - An `after_launch` error is a post-enqueue error: store the lease in
     `self.lost` and return the error.
4. **Scale validation for resident formats** (`chain.rs`). Move the affine
   scale check at 1536-1580 into
   `pub(crate) fn validate_affine_scales(format: &WeightFormat, shape: &[u64], bytes: &[u8]) -> Result<()>`,
   and call it from its old place. `WeightImages::new` (change 5) calls it
   for every formatted image.
5. **`WeightImages`** (`residency.rs`): 0102's in-memory source, moved, with
   - `pub(crate) fn new(artifact: ArtifactId, images: BTreeMap<String, Vec<u8>>, formats: &[(String, WeightFormat, Vec<u64>)]) -> Result<Self>`;
   - `impl ChunkSource` serving whole chunks only.
6. **Stage worker loaded mode** (`rank_worker.rs`).
   New commands:
   - `LoadStage(Box<LoadStage>, Sender<Result<Vec<GatherDeclaration>>>)`;
   - `PrepareStep { rows, visible_tokens, inputs: Vec<OwnedBinding>, reply }`;
   - `RunStep { stall: bool, reply: Sender<Result<Vec<u8>>> }`;
   - `UnloadStage(Sender<Result<()>>)`;
   - `StageStats(Sender<Result<StageCounters>>)`.

   ```rust
   pub struct StageWeights {
       pub source: Box<dyn ChunkSource + Send>,
       pub roles: BTreeMap<ValueId, String>,           // stage-local weight -> chunk role
       pub formats: BTreeMap<ValueId, WeightFormat>,   // stage-local
   }
   struct LoadStage {
       stage: StageGraph, catalogue: KernelCatalogue,
       buckets: Vec<(u64, u64)>, weights: StageWeights,
       pipeline_rank: u32, stages: u32,
       nccl_id: StageId,   // enum { Generate(Vec<Sender<Result<NcclId>>>), Receive(Receiver<Result<NcclId>>) }
   }
   pub struct StageCounters { pub admissions: u64, pub bytes_uploaded: u64, pub ready_waits: u64, pub stream_waits: u64 }
   ```

   On `LoadStage`, the outer loop calls
   `fn serve_loaded(worker: &mut WorkerState, commands: &Receiver<WorkerCommand>, load: LoadStage) -> Result<()>`.
   Inside it, as locals in this order:
   1. `NcclId`: generate or receive (the `StartupIdChannel` pattern,
      `dense_tp_workers.rs:147-151`);
   2. `Communicator::init(pipeline_rank, stages)`, then `admit_nccl_reserve`;
   3. `ResidencyAuthority::open(&mut worker.ledger, &ResidencyRequest::new("pipeline stage weights", 4 << 30).device_weights(uuid, total))`,
      where `total` = Σ planned `physical_bytes` of the stage's weights from
      the first bucket's candidate;
   4. `DeviceResidency::create`;
   5. for each `(value, role)` in `roles`: acquire → `drain_reads(source)`
      → `perform_upload` (the loop of `dense_gemma_device.rs:5811-5842`,
      `TurnId::new(1)`, `UseClass::demand(Content::DenseSpine)`);
   6. per bucket, `lower_stage_with_formats` with the stage workload, then
      `DensePlanSet::admit`;
   7. reply `Ok(declarations)`, where each declaration is the stage output's
      `(rows, columns, precision)` per bucket, via `declare_selected_plan`.

   Then loop on `commands`:
   - **`PrepareStep`**, with no enqueue:
     - bucket lookup, and `visible_tokens` ≤ the admitted value;
     - `validate_bindings_except(plan, graph, &inputs, &resident)`, where
       `resident = {handoff read local}` for stages > 0;
     - `validate_embedding_token_sources` for each Embedding node;
     - `state.begin()` if there is no transaction.

     Store the inputs. On error, abort the transaction if this command began
     it, and reply the error.
   - **`RunStep`**: if `stall`, `park_lost()` (test hook). Otherwise
     `set.step_stage` with:
     - `before_launch` (stages > 0): `group_start; recv(input range of the
       handoff local, rows × width, Bf16, peer = rank−1); group_end;
       wait_ready`;
     - `after_launch` (all stages but the last): `group_start; send(output
       range, …, peer = rank+1); group_end; wait_ready`.

     `wait_ready` is the Ready-poll half of 0102's `drain`, counted in
     `ready_waits`. `finish` is the one stream wait, counted in
     `stream_waits`. Reply the output bytes. **Any error here aborts the
     communicator and returns `DeviceLost`** (0102's post-enqueue rule).
   - **`PrepareCommit` / `ApplyCommit` / `Abort` / `Stats`**: as today.
   - **`StageStats`**: counters (`admissions` from `ledger.admissions()`,
     `bytes_uploaded` from `authority.stats()`).
   - **`UnloadStage`**, in order:
     - `set.close(&mut ledger, &mut authority)`;
     - `end_turn(TurnId::new(1))`;
     - `retire_all(scope)`;
     - `residency.close`;
     - `authority.close`;
     - `communicator.finalize(deadline)`, then `destroy`;
     - release the NCCL reserve.

     Reply `Ok` and return to the outer loop. On any refusal, reply
     `DeviceLost` and `park_lost()`, keeping every local alive.
   - **`Shutdown` while loaded:** unload first, then shut down.
   - The M5 `Step` while loaded is refused with `invalid("pipeline", "a
     persistent stage is loaded")`.
7. **Coordinator API** (`pipeline.rs`):
   - `pub fn load_persistent(&mut self, graph, lowering, oracle, oracles, catalogue, buckets: &[(u64, u64)], weights: &mut dyn FnMut(usize, &StageGraph) -> Result<StageWeights>) -> Result<()>`:
     - refuse if a pair stage exists or something is already loaded;
     - build the stage graphs as `execute` does (165-180);
     - send every `LoadStage` before collecting any reply (stage 0
       generates the id);
     - check that stage *i*'s output declaration equals stage *i+1*'s
       handoff read extent per bucket;
     - on any failure, `unload_persistent` the stages that loaded, then
       return the first error.
   - `pub fn step_persistent(&mut self, rows, visible_tokens, inputs: &mut dyn FnMut(StageBindings<'_>) -> Result<Vec<OwnedBinding>>) -> Result<PipelineStep<'_>>`:
     - round 1: `PrepareStep` to all; collect all replies. Any error means
       `Abort` to all, returned (recoverable);
     - round 2: `RunStep` to all (send all, then collect all); logits come
       from the last stage;
     - returns the existing `PipelineStep`, so commit and drop are
       unchanged;
     - `handoff_bytes` stays all zeros, because no host-staged bytes.
   - `pub fn unload_persistent(&mut self) -> Result<()>`;
     `pub fn stage_counters(&mut self) -> Result<Vec<StageCounters>>`;
     `pub fn command_rounds(&self) -> u64` (incremented once per round sent).
   - `execute` refuses while loaded. `close` unloads first.
8. **Test hooks** (`paged-attention-test-hooks`): `stall_next_run(stage)`
   sets `RunStep.stall`. The existing `fail_next_step(stage)` now fails the
   next `PrepareStep`.

## Escape inventory

| Resource | Owner | Success | PrepareStep refusal | Post-enqueue error / stall | Unload |
|---|---|---|---|---|---|
| Stage plans (per bucket) | `DensePlanSet` in `serve_loaded` | back in the set after `finish` | untouched | the lease sits in `set.lost`; the worker parks with every local | `set.close` first |
| Weight leases, `DeviceResidency`, authority | `serve_loaded` locals | untouched | untouched | parked | leases → `end_turn` → `retire_all` → residency → authority |
| Communicator and NCCL reserve | `serve_loaded` locals | persistent | untouched | aborted; the reserve stays charged, parked | finalize → destroy → release, **after** every stream use is drained |
| Handoff input/output ranges | the bucket plan | reused every step | untouched | held by the lost lease | with the plan |
| Stored step inputs | `serve_loaded` | moved into the lease, returned by `settle_sources` | dropped | held in the lease | none may exist |
| Attention runs, KV transaction | `WorkerState` | unchanged M5 rules | the transaction is aborted if this step began it | parked | `close_runs` at shutdown |
| `WeightImages` / `CheckpointSource` | `serve_loaded` | dropped after upload | n/a | dropped | n/a |

## Tests (`dense_gemma_device.rs`; driver, paged-attention-binding, test-hooks, nccl)

Fixtures and device selection are those of
`pipeline_runs_dense_and_routed_gemma_on_three_gpus` (2656-2750). Rows:
prompt 5 (one step), then 3 decodes. Buckets `{(5, 8), (1, 8)}`.

- **(a) `persistent_pipeline_matches_single_gpu_and_m5`**, dense Shape A and
  routed Shape C, on 3 stages (5060 Ti, 3090, 3090; cuts at layers 1 and 4)
  and on 2 stages (the 3090 pair; cut at layer 3). At every step:
  - persistent logits == single-GPU `DensePlanSet` logits on one 3090, byte
    for byte;
  - persistent logits == M5 `execute` logits with microbatches `[0..5]` then
    `[5..6]`, … on a separately spawned worker set, byte for byte.
- **(b) Counters.** After `load_persistent` and after each decode:
  - every stage's `admissions` and `bytes_uploaded` are unchanged;
  - `stream_waits` rises by exactly 1 per stage;
  - `ready_waits` rises by exactly 1 (first and last stage) or 2 (middle);
  - `command_rounds` rises by exactly 2.
- **(c) Formatted weights,
  `persistent_pipeline_runs_affine_int8_stages`.** Shape A with its linears
  formatted as task 0081's affine test formats them (INT8, group 32), on the
  2-stage 3090 pair. Logits == the single-GPU uploaded-binding run of the
  same formatted graph (the 0081 path), byte for byte, on prompt and 3
  decodes.
- **(d) Recoverable refusal.** `fail_next_step(1)` before decode 1: the step
  errors, every stage's `stats()` equals its pre-step value, and the retry
  is byte-identical to (a)'s decode 1.
- **(e) Lost group.** `stall_next_run(1)` before decode 2: `step_persistent`
  returns `DeviceLost` within `2 × deadline`, and later calls return the same
  lost error.
- **Mutant** (run only (a), 3-stage dense): in `before_launch`, receive into
  the stage **output** range instead of the handoff input range. (a) must
  fail.

## Gates
- **Host:** fmt; workspace clippy; xtask CUDA clippy; executor clippy with
  `driver,paged-attention-binding,paged-attention-test-hooks,nccl,cublas`
  and without; `cargo test --workspace --locked`; arch-check; spec-check.
- **GPU, once:**
  - the new tests;
  - the existing pipeline tests: `pipeline_runs_dense_and_routed_gemma_on_three_gpus`,
    and `combined_tp2_tp1_pipeline_matches_host_and_reports_capacity_latency`
    in `dense_tp2_device.rs`, because `pipeline.rs` and `rank_worker.rs`
    change.
- No `test-gpu` (no kernel source changes). No timing.

## Stop conditions
- P1 is not approved.
- 0102 is not accepted.
- NCCL cannot form a communicator across the 5060 Ti (SM120) and a 3090
  (SM86). Report the error; the fallback is pinned host staging, which needs
  a new design.
- Byte identity fails.
- `DensePlanSet` cannot hold formatted weights without a change beyond
  change 3.
- A file outside the list is needed.

---

## Result, filled after work
