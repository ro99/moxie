# Task 0102 — persistent TP2 rank chain with one stream wait per step

Status: **active, revision 3** (coordinator, 2026-09-26). Builder Claude
Sonnet `builder`; reviewer Claude Opus `reviewer`. Asynchronous-ownership
work.

This contract **is** the "0102a" text of the reviewer's second design review
([record](../evidence/task-0102-design-review-2.md)). That review also
rechecked sol's first design review, adopting parts of it and cutting
others. Its "0102b" half is **merged back in here** as the section "Final schedule:
one stream wait per step", on the owner's pace ruling (2026-09-26). Build
0102a's changes first, pass test (a), then apply that section, in the same
task.
The earlier revisions are kept below as history. **Implement only the
numbered changes below.** Where the history disagrees with them, the
numbered changes win.

## Identity and authority

- Task0102, M6 roadmap **M6.1** (device-resident execution), on the M6 exit
  critical path: all three exit checkpoints need more than one GPU. The
  finding and its lifecycle evidence are in
  [multi-gpu-lifecycle-before-0102.md](../evidence/multi-gpu-lifecycle-before-0102.md).
  One Shape A TP2 decode step today makes 114 plan admissions and
  re-uploads every weight.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files (`.gitignore`, `docs/evidence/specification-version.md`, ADRs 0034
  and 0035). **Stage explicit paths only.** No `git stash` or worktree.
  GPUs free; always `CUDA_DEVICE_ORDER=PCI_BUS_ID`. No task numbers in code
  identifiers.
- On any conflict with the code, stop and report `DECISION` to the
  coordinator; do not redesign.

## Allowed files

- `crates/moxie-executor/src/dense_tp_workers.rs`;
- `crates/moxie-memory/src/ledger.rs` (the `admissions` counter only);
- `crates/moxie-executor/src/dense.rs` (only the token-check split in
  change 6);
- `crates/moxie-executor/tests/dense_tp2_device.rs` and
  `crates/moxie-executor/tests/dense_tp2_cublas_device.rs`;
- this task's Result.

## Outcome


Outcome: after `load_chain`, a TP2 decode or prompt step makes zero ledger
admissions and zero weight uploads, has exactly two paired commands, and is
byte-identical to the single-GPU split reference and to M5. Joins still drain
one at a time **inside** the worker (same code as `drain_join`), so failure
semantics equal 0100's.

1. **Types (`dense_tp_workers.rs`).**
   ```rust
   pub struct ChainBucket { pub rows: u64, pub visible_tokens: u64 }
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub struct ChainCounters { pub admissions: u64, pub bytes_uploaded: u64, pub ready_waits: u64, pub stream_waits: u64 }
   enum ChainStage { Compute(Box<StageGraph>), Join { join: Join, source: ValueId } } // source = the Local stage graph's output
   type WeightKey = (ValueId, Option<(u64, u64)>, Option<(u64, u64, u64)>); // original, rows, slice(first,width,full_width)
   struct ChainJoinBuffers<'ctx> { arena: DeviceArena<'ctx>, status: DeviceRange<'ctx>, data: DeviceRange<'ctx>, dummy: DeviceRange<'ctx>, declaration: GatherDeclaration }
   struct ChainBucketSet<'ctx> { plans: Vec<Option<SelectedReservedPlan<'ctx>>> /* index = compute stage */, joins: Vec<ChainJoinBuffers<'ctx>> /* index = join */, boundary: Boundary<'ctx> /* every range allocated at load */, admitted_visible: u64 }
   struct RankChain<'ctx> { stages: Vec<ChainStage>, weight_keys: Vec<BTreeMap<ValueId, WeightKey>> /* per compute stage: local weight → key */, leases: BTreeMap<WeightKey, ResidencyLease>, authority: ResidencyAuthority, residency: DeviceResidency<'ctx>, module: ResolvedModule<'ctx>, buckets: BTreeMap<u64, ChainBucketSet<'ctx>>, catalogue: KernelCatalogue }
   ```
   `WorkerState` gains `chain: Option<RankChain<'ctx>>`, `ready_waits: u64` and
   `stream_waits: u64`. `DenseRankWorkers` gains `chain_stages: Option<[Vec<ChainStage>; 2]>`
   (the coordinator's copy, used to call the input callback).
2. **Chain builder:** private `fn rank_chain_stages(graph, lowering, rank, oracle, oracles) -> Result<Vec<ChainStage>>`,
   the loop body of `execute_dense` (`dense_tp_workers.rs:541-695`) without the
   commands. After the loop, check that every `StageRead.original` of every
   compute stage is a `graph.inputs()` member or the output of an earlier
   replicated node or join. Otherwise return `invalid("chain", ..)`.
3. **Commands:**
   - `LoadChain { stages, catalogue, buckets: Vec<ChainBucket>, weights: Vec<Vec<OwnedBinding>> /* per compute stage */ }`
     replies `WorkerValue::ChainJoins(Vec<(u64 bucket_rows, GatherDeclaration)>)`.
   - `ChainPrepare { rows, visible_tokens, inputs: Vec<Vec<OwnedBinding>> }` replies `Transaction`.
   - `ChainRun { transaction, fail_first_join: bool }` replies `Bytes` (logits).
   - `CloseChain` replies `Unit`.
   - `ChainStats` replies `WorkerValue::Chain(ChainCounters)`.

   Declarations: LoadChain `[stages.len(), buckets.len(), 0, 0]`,
   ChainPrepare/ChainRun `[rows, visible_tokens, 0, 0]`, F32 precision.
4. **`load_chain` worker order.** On any error, return the partially built
   `RankChain` into `self.chain` and fail; `CloseChain` is the only way out.
   1. Refuse if `self.chain`, `self.transaction` or `self.held.boundary` is set.
   2. Collect weight images from `weights` in stage order into `BTreeMap<WeightKey, Vec<u8>>`. A repeated key with different bytes is refused.
   3. `ResidencyAuthority::open(&mut self.ledger, &ResidencyRequest::new("tensor-parallel rank chain weights", total).device_weights(uuid, total))`,
      with `total` = Σ over images of `len` rounded up to 256.
      Then `DeviceResidency::create(ctx, &mut authority)`.
   4. For each image in key order, with role `format!("w{index}")` and artifact
      `ArtifactId::new("tp-rank-chain")`, run the acquire → `drain_reads` →
      `perform_upload` sequence of `dense_gemma_device.rs:5811-5842`.
      `AcquireRequest { now: 0, deadline: u64::MAX, class: UseClass::demand(Content::DenseSpine), turn: TurnId::new(1) }`.
      An `Acquired::Ready` result is refused. The in-memory `ChunkSource` is a
      private struct over `BTreeMap<String, Vec<u8>>` that copies
      `[offset, offset+len)`.
   5. `module` = the `TP_F32_TO_BF16` module, loaded as `prepare_join` does (`dense_tp_workers.rs:2306-2313`), once.
   6. For each bucket, for each compute stage:
      - lower with `lower_selected_ordered` (the `run_stage` workload, with the bucket's rows/visible_tokens);
      - refuse a candidate with nonempty `weight_formats()` or `host_expert_joins()`;
      - build `addresses: BTreeMap<local weight, u64>` from `residency.device_address(authority.device_range(lease)?)`, checking `len == planned.logical_bytes` and `region == StorageRegion::Weights`;
      - `admit_with_resident_weights`.

      For each join: `declare_selected_plan` on its Local stage's plan, then a
      `CollectiveBuffers` arena and three ranges sized as `prepare_join` sizes
      them (`2201-2283`). Allocate the boundary arena of `boundary_bytes(graph,
      rows)` and one range per replicated node output and per join output
      (the join output sized as `prepare_join`'s `output_bytes`).
5. **`step_chain` (coordinator):** `check_live`, then refuse if no chain is
   loaded or `rows` is not a bucket. Build `inputs` by calling the callback per
   compute stage per rank. Then `pair_command(ChainPrepare)`; on error,
   `settle(Some(tx), true)` as `execute_dense` does. Then
   `pair_command(ChainRun{fail_first_join: take_join_fault(rank)})`. Return
   `DenseWorkerStep` with rank 0's bytes; commit and drop are unchanged.
6. **`ChainPrepare` (worker)**, before any enqueue:
   - `Begin`'s run check;
   - `state.begin()`;
   - for every compute stage, the `dense_set.rs:309-326` lease/address check against that stage's plan;
   - `validate_bindings_except(plan, &stage.graph, &inputs[i], &resident_i)`, where `resident_i` is the set of the stage's `StageRead.local` whose `original` is in the boundary;
   - the embedding token check on the embedding stage's token binding. Today
     `validate_embedding_tokens` (`dense.rs:3544`) takes a `DenseOperation`
     and runs inside the enqueue, **after** `upload_sources` has marked the
     operation submitted, so a bad token loses the group. Split it: add
     `pub(crate) fn validate_embedding_token_sources(sources: &[OwnedBinding],
     value: ValueId, rows: u64, vocab: u64) -> Result<()>` holding the current
     body (via `index_values_from_sources`), and make `validate_embedding_tokens`
     call it with `&operation.sources`. `ChainPrepare` calls it for every
     `OpParams::Embedding { vocab, .. }` node of every compute stage, with
     `node.inputs[0]`;
   - `rows` equals the bucket's and `visible_tokens` ≤ `admitted_visible`.

   On failure, abort the transaction and return the error.
7. **`ChainRun` (worker), 0102a schedule.** For each chain stage in order:
   - **Compute:** `copy_boundary` for its boundary reads, then
     `plan.execute_dense_stage(..)`, then `finish` is **not** called: store the
     lease. Replicated: copy the plan output into its preallocated boundary
     range (a `keep` variant that takes the range instead of allocating).
   - **Join:** the `copy_join` body with the source address borrowed from the
     stored lease (M4), and the status set to `fail_first_join && join == 0`.
     Then `drain(deadline)`, read status, and on success the conversion or
     interleave into the preallocated output range, then `drain` again (0102a
     only).

   After the last stage: `drain`, reclaim runs (H6), then retire every lease:
   `retire()` → `plan = op.plan.take()`, `plan.settle_sources(op.sources)`, put
   the plan back in its slot. Read logits from the graph output's boundary
   range. If any status was nonzero, abort the transaction and return
   `invalid("collective", "a rank failed after join preparation")` (same on
   both ranks). Any other error after the first enqueue:
   `abort_communicator`, keep everything in `self.chain`, return `DeviceLost`.
8. **`close_chain` / `close` / `Drop`:** H5 order. `DenseRankWorkers::close`
   runs `CloseChain` first when `chain_stages` is `Some`.
9. **Counters:** `pub fn chain_counters(&mut self) -> Result<[ChainCounters; 2]>`
   (`admissions` from `ledger.admissions()`, `bytes_uploaded` from
   `authority.stats().bytes_uploaded`, 0 if no chain) and
   `pub fn command_sequence(&self) -> u64 { self.sequence }`.
10. **Tests (`dense_tp2_device.rs`), non-routed `order_sensitive_fixture(false)`.**
    Prompt `[1,4,7,2,9]` at positions 0..4, then 8 decodes with tokens
    `6,7,8,9,10,11,12,13` at positions 5..12. Buckets `{(5,13),(1,13)}`.
    - (a) Reference bytes from `reference_step` for all 9 steps. M5 bytes from
      `tp_worker_step` on a first `DenseRankWorkers`, then close it. Chain on a
      second group. Assert all three equal byte for byte at every step. The
      cuBLAS twin repeats it with the cuBLAS catalogue.
    - (b) `chain_counters` after `load_chain` and after each decode: `admissions`
      and `bytes_uploaded` unchanged on both ranks across all 8 decodes.
    - (d) `fail_next_join_after_prepare(1)` on decode 1: `step_chain` returns the
      collective error, `stats()` equals its value before the step, and the retry
      of the same token is byte-identical to the reference.
    - (e) `stall_before_next_rendezvous(1)` before a `step_chain`: the group is
      lost, and the elapsed time is under the bound the existing stall test uses
      (`dense_tp2_device.rs:1625-1661`).
    - Command count: `command_sequence` delta over one decode `step_chain` == 2.
11. **Mutant:** M-F.

Stop conditions:
- byte identity fails;
- a Shape A candidate has weight formats;
- the residency cap `total` is refused;
- a file outside the list is needed.


## Final schedule: one stream wait per step (formerly 0102b)

Apply this after 0102a's numbered changes pass test (a), in this same task.



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



## Ledger admission counter (review M-A)

Add to `Ledger` (`crates/moxie-memory/src/ledger.rs`):
- a field `admissions: u64`, incremented once per successful `admit`,
  after the reservation is committed;
- the accessor `pub const fn admissions(&self) -> u64`.

## Escape inventory


| Resource | Owner | Success | Pre-enqueue refusal (ChainPrepare) | Status failure | Other post-enqueue error / lost | Close |
|---|---|---|---|---|---|---|
| Chain plans | `RankChain.buckets[..].plans` | back in slot after `retire` | untouched | back in slot after drain + retire | parked in chain (plan inside un-retired lease) | H5 step 1 |
| Stage leases (in flight) | `ChainRun` local `Vec` | retired after the final drain | none exist | retired after drain | kept in `self.chain`, never dropped | none may exist |
| Weight leases / residency / authority | `RankChain` | untouched | untouched | untouched | parked | H5 steps 4–8 |
| Join arenas and ranges | `ChainBucketSet.joins` | persistent | untouched | persistent | parked | H5 step 2 |
| Boundary arena and ranges | `ChainBucketSet.boundary` | persistent, overwritten only after the prior drain | untouched | persistent | parked | H5 step 2 |
| Conversion module | `RankChain.module` | persistent | untouched | persistent | parked | H5 step 3, after the drains |
| Attention runs | `WorkerState.runs` | reclaimed after drain (H6) | idle check only | reclaimed after drain | park with group | existing `close_runs` |
| Transaction | `WorkerState.transaction` | commit via `DenseWorkerStep` | aborted in ChainPrepare | aborted after drain | group lost | n/a |
| Step host bindings | each `DenseOperation.sources` | returned by `settle_sources`, dropped | returned by the refusal, dropped | as success | held in the lease | n/a |

## Acceptance

Host gates:
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks,nccl,cublas
  --locked -- -D warnings`;
- `cargo test --workspace --locked`;
- `cargo xtask arch-check`;
- `cargo xtask spec-check`.

GPU gates, once, with `nccl,cublas`:
- the full `dense_tp2_device`;
- `dense_tp2_cublas_device`.

No `test-gpu` (no kernel source change), and no timing. Report the candidate
commit early, so the reviewer can start while the suites run.

## History (superseded; kept for provenance)

## Facts established before writing (sol's inventory, base `50464e9`, plus the coordinator)

- `DenseRankWorkers` keeps these across calls, per rank:
  - its context, stream and ledger;
  - its `DeviceKvSequence` and `PagedAttentionRun`s;
  - its NCCL communicator and 144 MiB `CollectiveBuffers` reserve.

  All of them live in `WorkerState` (`dense_tp_workers.rs` about 203–228,
  1233–1265, 1406–1537).
- Per `execute_dense` call:
  - `Begin` opens a transaction and a boundary arena (about 528–550).
  - Every **replicated node** becomes its own one-node `StageGraph`, with
    `Stage`, `Drain` and `ClosePlans` each (about 552–594).
  - Every `Stage::Local` runs `Stage`, `JoinPrepare`, `JoinCopy`,
    `JoinDrain` (about 596–695).
  - `Stage` lowers with `lower_selected_ordered` and **admits** a
    `SelectedReservedPlan`, with its weights uploaded from the caller's
    `OwnedBinding` host bytes (about 2072–2183; `dense.rs` about
    2493–2519).
  - `JoinPrepare` admits a fresh temp arena and loads a conversion module
    per join (about 2186–2315).
  - `ReadOutput`, `Drain` and `Cleanup` close the step; `Commit` later
    publishes KV (about 699–763, 1627–1732).
- Stage boundaries pass activations through a per-step boundary arena, with
  device-to-device copies (about 2132–2139, 2855–2885).
- `DensePlanSet` (`dense_set.rs`) admits bucket plans with
  `SelectedReservedPlan::admit_with_resident_weights` (`chain.rs` about
  276) against residency leases, and steps them.
- `build_stage_graph` and `StageGraph` (`moxie-plan/src/tensor_parallel.rs`
  about 86–100) build rank-local stage graphs from `lowering.ranks[rank]`.

## Bounded deliverable

- **Outcome:** `DenseRankWorkers` gains a **persistent rank chain** mode.
  After one `LoadChain`, a TP2 decode or prefill-bucket step:
  - makes **zero plan admissions and zero weight uploads**;
  - queues every stage and every NCCL join of the step on each rank's
    stream **without a host wait between stages**;
  - ends with **one** drain per rank that reads the join status words;
  - returns logits **byte-identical** to the single-GPU split reference and
    to the existing M5 path.

  KV commit is unchanged. The M5 per-step path stays for now: PP's pair
  stage uses it until the PP task.
- **Allowed files:**
  - `crates/moxie-plan/src/tensor_parallel.rs` (the chain builder), and
    `crates/moxie-plan/src/selected.rs` if slot liveness for escaping values
    needs it (sol H2);
  - `crates/moxie-memory/src/residency.rs` (a test-only cumulative upload
    counter, if one is missing; sol M7);
  - `crates/moxie-executor/src/{dense_tp_workers.rs,dense_set.rs,chain.rs,dense.rs}`;
  - `crates/moxie-executor/tests/dense_tp2_device.rs` (new tests);
  - this task's Result.
- **Non-goals:**
  - PP (a following task);
  - capture (task 0103);
  - removing the M5 path;
  - more than two ranks;
  - timing.

## Numbered changes

1. **Rank chain (`moxie-plan`).** Add `pub fn build_rank_chain(graph,
   lowering, rank, oracle, oracles) -> Result<RankChain>`, where
   `RankChain { stages: Vec<ChainStage> }`, and each `ChainStage` is one of:
   - `Compute(StageGraph)`: each maximal `Stage::Replicated` range becomes
     **one** stage graph (not one per node), and each `Stage::Local`
     becomes the rank's local stage graph;
   - `Join { kind: Reduce | Gather, declaration, source: ValueId }`,
     immediately after its `Local` compute stage.

   Order is exactly `lowering.stages`. Validate that each stage's reads are
   produced by an earlier stage or are graph inputs. If merging a
   replicated range into one stage graph conflicts with how
   `build_stage_graph` handles outputs, stop and send `DECISION`.
2. **Chain plan set (`dense_set.rs` or a sibling type in the same file).**
   Add `RankChainSet` per rank, per rows bucket. For each `Compute` stage it
   holds a `SelectedReservedPlan` admitted **once** with resident weights
   (`admit_with_resident_weights`). For each `Join` it holds a persistent
   collective temp (status word, data ranges) and the conversion module,
   admitted once.
   - **Weights.** The rank's weights are uploaded **once** at `LoadChain`
     into a per-rank residency authority (`device_weights`, task 0091) from
     the caller's bindings. Each stage's resident-weight map points into
     it. A weight used by several stages (for example a tied embedding) is
     resident once.
   - **Boundaries.** The chain admits one persistent boundary arena, sized
     by the largest bucket. Stage outputs copy device-to-device into it, as
     today.
3. **Worker commands (`dense_tp_workers.rs`).**
   - `LoadChain { chain, rows_buckets, bindings }`, paired, once, builds
     the rank's `RankChainSet`s.
   - `StepChain { rows, visible_tokens, token/position bindings }`, paired.
     Per rank it:
     - begins the transaction;
     - enqueues every compute stage (execute without finish, holding the
       leases) and every join's `JoinCopy` (the status and data
       collectives, the 0100 order rules) on the rank stream;
     - then performs **one** drain: `poll` Ready, record an event, wait
       under the group deadline;
     - reads every join's status word. Any nonzero status aborts the step
       on both ranks, as task 0100 does;
     - finishes every lease and reads the output.
   - `Commit`, `Abort` and close are as today.
   - `DenseRankWorkers::load_chain(...)` and `step_chain(...)` are public.
     The existing `execute_dense` stays.
4. **Stage execution without an inner host wait (`dense.rs`/`chain.rs`).**
   If `execute_dense_stage` cannot enqueue a plan whose inputs come from the
   boundary arena, and leave its lease pending while the next stage
   enqueues on the same stream, add the minimal `pub(crate)` enqueue/finish
   split. Stream order is the dependency, so no events are needed between
   same-stream stages. If a stage requires a host read mid-step (for
   example the embedding token check), it is done **before** the first
   enqueue for all stages, as task 0101's prepare does.
5. **Failure.** A failure before any enqueue returns the step refused, with
   the chain usable. A failure after enqueue (an enqueue error, a nonzero
   status, a poll error or a deadline) follows 0100: on a status error,
   both ranks abort the transaction and the chain stays usable; on a
   communicator error, the rank aborts its own communicator and the group
   is lost. A chain step never releases a plan, temp or weight.
6. **Close.** `close` drains, then closes every chain set (plans, temps,
   modules, boundary arena), then releases the weight leases and the
   residency authority, then does 0100's communicator close. A refused
   close keeps every surviving resource.
7. **Escape inventory.**

   | Resource | Owner | Rule |
   |---|---|---|
   | Chain plans | Rank's `RankChainSet` | Never released by a step. Closed only at chain close after an observed drain. On a lost group, parked and charged. |
   | Resident weights | Rank residency authority | Leases held by the set. Released after every plan is closed. |
   | Join temps and conversion modules | Chain | Persistent. A pending collective keeps them (0100 rules). |
   | Boundary arena | Chain | Persistent. Stage outputs are overwritten each step only after the previous step's drain. |
   | Leases of a step in flight | Rank | Kept until the single drain. A post-enqueue failure withholds them with the chain. |
8. **Tests (`dense_tp2_device.rs`).**
   - (a) Shape A: prompt, then 8 decode steps, on the chain path. Logits
     are byte-identical at every step to the single-GPU split reference and
     to the M5 `execute_dense` path. Run with the ordered catalogue and with
     the cuBLAS one.
   - (b) **Zero admissions and uploads per step:** the rank ledger's plan
     admission count and weight upload byte count do not change across the
     8 decode steps (a test-hook or `stats` counter).
   - (c) **One host wait per step per rank:** the paired rendezvous count
     per `StepChain` equals the fixed count (begin, step, output), and does
     not scale with stages.
   - (d) The status-word failure on rank 1 aborts both ranks' step, and the
     next step succeeds byte-identically.
   - (e) The stall test produces a lost group within the deadline on the
     chain path.
9. **Coverage check (one mutant, reverted after; run only test (a)).**
   Point one stage's resident weight at the wrong lease offset. Test (a)
   must fail.

## Acceptance

Host gates:
- fmt;
- workspace clippy;
- executor clippy with `nccl,cublas` and without;
- `cargo test --workspace --locked`;
- arch-check;
- spec-check.

GPU gates, once, with `nccl,cublas`:
- the full `dense_tp2_device`;
- `dense_tp2_cublas_device`.

No `test-gpu` (no kernel changes), and no timing. The reviewer may start at
the candidate commit.

**Stop conditions:**
- a replicated range cannot become one stage graph;
- a stage cannot be enqueued without a host wait;
- byte-identity fails;
- a file outside the allowed list is needed.

## Design review (sol, 2026-09-25) — adopted verbatim; overrides the numbered changes

- **H1 (outcome, change 3, test c).** Between joins, poll Ready under the
  group deadline whenever 0100 requires it, before the next NCCL or
  dependent enqueue. The single final CUDA drain remains. Any intermediate
  readiness wait is counted and reported. If one host wait per step is
  mandatory, stop for `DECISION`. Do not use the rendezvous count as a
  proxy for host waits.
- **H2 (changes 1, 2).** Compute records every produced value used by a
  later chain stage or by the final graph output. Preserve each such range
  until its boundary copy, or copy it immediately after its producer,
  before reuse. Validate that its planned slot cannot alias before
  copying. Preallocate and retain all corresponding boundary ranges. If
  planner liveness needs a file outside the allowlist, invoke the stated
  `DECISION` stop.
- **M1 (change 2).** Key resident images by original weight plus the exact
  rank-local row or column slice and packed layout. Create bytes matching
  `StageWeight`, deduplicate identical images, and validate each
  stage-local length and address against its lease before every step.
  Share a full tied embedding and a row shard only through a checked
  subrange when their bytes and layout permit it.
- **M2 (change 1).** `Join` stores only static kind, source and output
  metadata. At `LoadChain`, derive rows, columns and precision from the
  admitted candidate for each bucket. At each `StepChain`, insert the
  current rendezvous sequence and verify that both ranks agree.
- **M3 (changes 3, 5).** Preinitialize a bounded dummy source. Always queue
  the join conversion or gather and publish its boundary destination, even
  when the local status is set, and continue the fixed collective schedule
  on both ranks. After the final drain, read all status words, abort both
  transactions, and restore every plan and temp to a reusable idle set. Any
  unknown completion parks the entire set.
- **M4 (changes 2, 4).** `JoinCopy` borrows the source range from the held
  stage lease; no `take_range_for_value` or `release_join_source`. After the
  one observed drain, retire the intermediate leases without host output
  readback, settle their sources and return the plans to the set. Read only
  the final logits. A refusal preserves each lease and its named ranges.
- **M5 (change 6, inventory).** Close order:
  1. plans, join resources and boundary resources;
  2. release the leases;
  3. `DeviceResidency::close`, returning the backing;
  4. `ResidencyAuthority::close` against the rank ledger;
  5. only then finalize or destroy the communicator and release its
     reserve.

  On refusal, keep the exact surviving owner, including partial
  `LoadChain` uploads and admissions and in-flight leases.
- **M6 (changes 2, 7).** At `LoadChain`, allocate one persistent range per
  escaping original value per bucket (or prove a safe shared slot
  schedule). Charge it once, reuse it only after the previous step's
  observed drain, and keep all old ranges through a refusal or loss.
- **M7 (tests b, c).**
  - Expose a test-only cumulative admission count and the rank-local
    `ResidencyStats.bytes_uploaded`. Snapshot both after `LoadChain` and
    after each decode.
  - Count actual Ready waits and CUDA synchronizations inside `StepChain`
    separately, and assert the intended bound.
  - If checking only coordinator dispatches, assert exactly one
    `StepChain` pair and label it a command count.

## Result, filled after work
