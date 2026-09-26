# Task 0102 design review 2 (reviewer, read-only, /ponytail:ponytail-review)

Contract: docs/tasks/0102-m6-persistent-tp-rank-chain.md at 2957a15, with sol's section.
Findings: **6 HIGH, 6 MEDIUM, 3 LOW**, and a proposed split into **0102a / 0102b**.

## 0. Top-line recommendation

Do not hand Sonnet the numbered changes plus an override section plus this
review. That is three layers, and reconciling them is design work. Replace
"Numbered changes" and "Design review" with **one** set of numbered changes
per sub-task (§3 below is written to be pasted in). Keep sol's section as a
history note only.

The ponytail result: the task gets much smaller.
- **No `moxie-plan` change** (no merged replicated stages; see H2).
- **No `selected.rs`, `chain.rs`, `dense_set.rs` or `residency.rs` change, and one
  6-line `dense.rs` change** (the token check, see H4). Every item the chain needs is already `pub(crate)` or `pub` in
  `moxie-executor`: `admit_with_resident_weights` (`chain.rs:276`),
  `execute_dense_stage` (`dense.rs:146`), `DenseOperation.plan` / `.sources`
  (`dense.rs:81-89`), `OperationLease::retire` (`arena.rs:136`),
  `settle_sources` (`chain.rs:763`), `range_for_value` (`chain.rs:1007`),
  `value_address` (`chain.rs:544`), `DeviceResidency::device_address`
  (`residency.rs:562`), `drain_reads` (`residency.rs:381`) and
  `ResidencyAuthority::stats().bytes_uploaded` (`moxie-memory residency.rs:2048`).
- **One small addition outside the executor:** a cumulative admission counter on
  `Ledger` (M-A below).

Allowed files become:
- `crates/moxie-executor/src/dense_tp_workers.rs`;
- `crates/moxie-memory/src/ledger.rs` (the counter only);
- `crates/moxie-executor/src/dense.rs` (only the token-check split in H4);
- `crates/moxie-executor/tests/dense_tp2_device.rs`;
- the cuBLAS twin test file;
- this task's Result.

## 1. Sol's findings, checked against the code

| Sol | Verdict | Evidence / correction |
|---|---|---|
| H1 inter-join Ready poll | **Sound, and it must be decided now, not left as a DECISION stop.** | The communicator is non-blocking (`moxie-cuda/src/nccl.rs:164` `config.blocking = 0`). `group_end` only enqueues (`nccl.rs:249-256`). `drain` itself says NCCL may enqueue stream work after `InProgress` (`dense_tp_workers.rs:1790`). So a Ready poll is required before the next NCCL call and before a dependent enqueue (the Reduce conversion kernel, the Gather interleave copies). "One host wait per step" is therefore unachievable, so restate the outcome (H1 below). |
| H2 escaping values of a merged replicated range | **Sound. Avoid it rather than solve it.** | Slots are reused by liveness (`selected.rs:1509-1521`, `first`/`last` from `tensor.live`), and a `StageGraph` preserves only `graph.output()` (`tensor_parallel.rs:260-280`). Merging is a performance choice; the task's goal is zero admissions and uploads, and timing is a non-goal. Keep one stage graph per replicated node, exactly as `execute_dense` builds it (`dense_tp_workers.rs:552-579`). Every stage output is then its graph output, which the plan keeps to the end, and M5 byte-identity already proves the boundary set. |
| M1 image key | Sound; **drop the subrange-sharing clause.** | Key = `(original, rows, slice)` from `StageWeight` (`tensor_parallel.rs:74-80`). Equal key means one lease. No subrange sharing (the tied embedding costs one extra shard copy). That removes a design choice. |
| M2 static join metadata | Sound. | `declare_selected_plan` (`dense_tp_workers.rs:1192`) derives rows/columns/precision from an admitted plan. Call it at load per bucket and compare the two ranks' lists on the coordinator. |
| M3 fixed schedule on status failure | Sound; **drop "preinitialize a bounded dummy source".** | The dummy is only read on a failed join whose step is aborted and whose outputs are never read. Its content is irrelevant, and preinitializing it is code with no consumer. |
| M4 borrow the join source | Sound. | `take_range_for_value` + `release_join_source` frees the range into the plan's arena (`release_reclaimed`, `chain.rs:1108`) rather than restoring it, which leaves a persistent plan unusable. Borrow the address from the in-flight lease instead: `lease.resource().plan.as_ref().expect(..).range_for_value(v)?.device_address()?`. |
| M5 close order | **Incomplete, so it's HIGH (H5).** | `DeviceResidency::close` is refused while the scope holds placements (`moxie-executor/src/residency.rs:503-521`). The released leases leave placements cached, so `end_turn` + `retire_all` must come first, as in `affine_linear_device.rs` run_case. |
| M6 persistent boundary per escaping value per bucket | Sound, and simple once H2 is adopted. | The escaping values are every replicated node output plus every join output. |
| M7 counters | Half sound. | `bytes_uploaded` already exists (`ResidencyAuthority::stats()`), so no `residency.rs` edit. There is **no** cumulative admission count anywhere (`Ledger` has only `outstanding_count`, `ledger.rs:334`, which admit+release in one step leaves unchanged). See M-A. |

## 2. Findings

### HIGH

**H1 — the outcome's "one host wait" and the DECISION stop contradict the NCCL mode.**
Replace the outcome bullets 2–3 with:
> - queues every stage and every join on each rank's stream with **no paired
>   rendezvous and no CUDA stream/event wait between stages**. After each join's
>   `group_end`, the rank polls the communicator to Ready (a host spin on
>   `ncclCommGetAsyncError`, not a stream wait) before its next enqueue;
> - ends with **one** CUDA event wait per rank, after which it reads every
>   join status word.

Delete the H1 DECISION stop.

**H2 — merged replicated ranges: cut.** Replace change 1 with: "No `moxie-plan`
change. `Stage::Replicated(r)` yields one compute stage per node, built exactly
as `execute_dense` builds it today (`build_stage_graph(graph, None, n..n+1,
None, ..)`). `Stage::Local` yields the rank's local stage graph and one join."
Remove the stop condition "a replicated range cannot become one stage graph".
Remove `moxie-plan/*` from Allowed files.

**H3 — `DensePlanSet` cannot live in `WorkerState`.** `DensePlanSet<'r,'ctx>`
borrows `&'r Graph` and `&'r DeviceResidency<'ctx>` (`dense_set.rs:87-95`).
`WorkerState` would own both the residency and the set, which makes a
self-referential struct, so a builder will fight the borrow checker or
reinvent it. Specify a new private owned type in `dense_tp_workers.rs` (§3,
`RankChain`) that owns its stage graphs, authority, residency and leases. It
reuses `DensePlanSet`'s per-step lease check verbatim (`dense_set.rs:309-326`).
`dense_set.rs` stays untouched.

**H4 — an asymmetric pre-enqueue failure strands the peer in NCCL, so "refused,
chain usable" is false as written.** If rank 1 refuses a binding before its
first enqueue while rank 0 proceeds, rank 0's first join has no peer, and the
group is lost at the deadline. Task 0101's lesson applies: every host check
the enqueue path makes must run on both ranks, and their outcomes must be
agreed, **before** the first enqueue. The existing rendezvous already merges
one-sided failures into an agreed error (`tensor_parallel.rs:110-175`), which
is how M5's `Fault::Rank` recovers. So `StepChain` is **two** paired
commands: `ChainPrepare` (all host checks, begin transaction) then `ChainRun`
(enqueue, drain, status). Test (c)'s command count is therefore 2. Rule for
`ChainRun`: any error after its first enqueue, other than a nonzero join
status, makes the rank abort its communicator and return `DeviceLost`, so the
group is lost and every chain resource stays parked and charged. Only the
status-word path (test d) is recoverable after enqueue.

**H5 — close order is missing the residency drain steps.** Replace sol M5 with this exact order:
1. every chain plan `close(&mut ledger)`;
2. the per-bucket join arenas and the boundary arenas: release ranges, then
   `close(&mut ledger)`;
3. drop the conversion module;
4. `authority.release(lease)` for every weight lease;
5. `authority.end_turn(TurnId::new(1))`;
6. `authority.retire_all(Scope::Device(uuid))`;
7. `residency.close(&mut authority)`;
8. `authority.close(&mut ledger)`.

On any refusal, put the refused value back in the `RankChain` and return
`DeviceLost`, keeping the chain. `DenseRankWorkers::close` sends `CloseChain`
**before** `ClosePrepare`, because `close_runs` refuses any outstanding
reservation except NCCL's (`dense_tp_workers.rs:2759-2767`).

**H6 — the attention runs are missing from the escape inventory.** A chain
step enqueues deferred attention into the persistent `PagedAttentionRun`s
(`dense.rs:2347` `attend_into_deferred`). `Begin` requires idle runs
(`dense_tp_workers.rs:1884-1897`), and `cleanup` returns any held run range
to a plan (`dense_tp_workers.rs:2642-2658`). Add an inventory row and these steps:
> Runs: `ChainPrepare` performs `Begin`'s idle check. After `ChainRun`'s drain,
> each run is `reclaim_drained()`. Any returned range goes back through
> `return_worker_range` over that bucket's chain plans **before** the leases
> retire. On a lost group the runs park with it.

### MEDIUM

**M-A — no admission counter exists.** Add to `Ledger` (`ledger.rs`): a field
`admissions: u64`, incremented once per successful `admit` (after the
reservation is committed), with accessor `pub const fn admissions(&self) ->
u64`. This counts every admission on the rank ledger: plans, join arenas,
boundary arenas and residency. That is the property, measured where it is
owned. A counter at call sites misses the next new call site.

**M-B — the inputs/weights API is undefined.** Spell it out as in §3: two
callbacks with the existing `FnMut(usize, &StageGraph) -> Result<Vec<OwnedBinding>>`
shape. `load_chain`'s callback returns only `ValueRole::Weight` bindings.
`step_chain`'s callback returns only non-weight bindings (tokens, positions).
Any other role is refused with `invalid("bindings", ..)`. The test adapts
`stage_host_bindings` by filtering on role.

**M-C — buckets need an admitted `visible_tokens`.** `execute_attention`
refuses when `visible > admitted_visible` (`dense.rs:2327-2331`). Per-step
lowering used the real value (`dense_tp_workers.rs:2089-2095`). Buckets are
`ChainBucket { rows, visible_tokens }`. A step's rows must equal a bucket's
rows, and its `visible_tokens` must be ≤ that bucket's.

**M-D — chain and M5 path on one worker group.** Both use the same
`DeviceKvSequence`. Add: `load_chain` refuses while a transaction is open;
`execute_dense` refuses while a chain is loaded. Test (a) needs the M5 bytes,
so it runs M5 in its own `DenseRankWorkers`, closes it, then spawns a second
group for the chain (§3 test a).

**M-E — test (c) counters must say what is counted and where.** Split
`WorkerState::drain` (`dense_tp_workers.rs:1753`) into `wait_ready(deadline)`
(lines 1760-1786 unchanged) and `wait_stream(deadline)` (1788-1824
unchanged); `drain` calls both, so M5 is unchanged. Each increments
`WorkerState.ready_waits` / `.stream_waits` (`u64`). Expected per decode
`step_chain` per rank: `ready_waits` delta = joins + 1 (Shape A: 19 + 1 = 20),
`stream_waits` delta = 1, and `DenseRankWorkers::command_sequence()` delta =
2 (`ChainPrepare`, `ChainRun`).

**M-F — mutant 9 has no exact substitution.** It should read:
> in `load_chain`'s address map, `addresses.insert(local, address)` →
> `addresses.insert(local, address + 256)` for the first weight of the first
> compute stage only (guard `index == 0`). Test (a) must fail.

The per-step lease check (H3) is expected to catch it as a refusal. Record
which check fired.

### LOW

**L-1** — drop sol M3's dummy preinitialization (see table).
**L-2** — the header still names Codex `luna`/`sol`; the team is Sonnet `builder` / Opus `reviewer` (f2cc331).
**L-3** — one envelope deadline now covers a whole `ChainRun` (Shape A 5-row prefill). Keep the test's `DEADLINE`. If a clean prefill run exceeds it, that is a stop condition, not a reason to raise it silently.

## 3. Proposed split and the exact text

### 0102a — persistent rank chain (synchronous joins inside one worker command)

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

### 0102b — one stream wait per step

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

## 4. Escape inventory, replacing change 7

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
