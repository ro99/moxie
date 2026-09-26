# Task 0102 — persistent TP2 rank chains: plans admitted once, weights resident once, one host wait per step

Status: **active** (coordinator, 2026-09-25). Sol's design review (2 high, 7
medium) is adopted verbatim in "Design review" below. Where it conflicts
with the numbered changes, **it overrides them**.
Builder Codex `luna`; reviewer Codex `sol`. Asynchronous-ownership work:
change 7 is the escape inventory.

## Identity and authority

- Task0102, M6 roadmap **M6.1** ("device-resident layer chains … output-
  minimizing transfers"), on the critical path of the **M6 exit**: all
  three exit checkpoints need more than one GPU.
- Ledger finding (2026-09-25). Sol's inventory
  ([multi-gpu-lifecycle-before-0102.md](../evidence/multi-gpu-lifecycle-before-0102.md)) shows that one
  Shape A TP2 decode step makes **114 plan admissions** (57 per rank). It
  **re-uploads every weight** from host bindings into each new plan, and
  has **57 paired host barriers**.
- Design basis:
  - Spec 04: "TP is a graph partition/lowering, not a separate model
    implementation."
  - Spec 02 line 116: collective order is **plan** content. Model crates
    may not contain collectives, so there is no new graph operation here.
  - Strata's rank-local executor: persistent per-rank plans with inline
    NCCL collectives (`strata/docs/dsv4-rank-local-architecture.md`).
  - Moxie's `DensePlanSet` (task 0091): bucket plans admitted once against
    one resident weight copy.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**; on a conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** GPUs free;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Naming rule applies.

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
