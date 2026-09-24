# Task 0070 — a single-GPU rank worker (one execution thread per GPU)

Status: **accepted** (coordinator, 2026-09-24, under the owner's auto-mode
delegation), after sol's review rounds R1 and R2. Builder Codex `luna`;
reviewer Codex `sol`.
- R1 (MEDIUM): teardown joined without a deadline. Fixed with a
  deadline-bounded `join_until`; a sticky loss skips shutdown.
- R2 (MEDIUM, from the coordinator's R1 design): the bounded join discarded
  a worker panic. Fixed with a three-way `JoinOutcome`. The coordinator
  checked the fix directly; a third review round was not needed.
- **Size, carried to the milestone-end ponytail audit:** `rank_worker.rs` is
  569 lines, against an estimate of 250–350. Sol counted about 40–65 lines
  duplicated with `dense_tp_workers.rs`:
  - workload construction and admission, around 260–294 (vs 1726–1763);
  - writer-adapter assembly, around 389–410 (vs 1456–1496);
  - stats and run close, around 433–464 (vs 2140–2169).
  Removing them means editing accepted TP code, which this task did not
  allow.

## Identity and authority

- Task0070, M5 plan slice 5, second task.
- Slice 5 is now **four** tasks, one more than planned:
  - **0069:** host pipeline lowering (accepted);
  - **0070:** this worker;
  - **0071:** pipeline execution on the three GPUs;
  - **0072:** the combined TP2 + TP1 plan and its report.

  The extra task exists because document 01 requires one execution thread
  per GPU, and the only worker today is the two-rank TP pair.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `f27c99c`.
  Preserve the carried `.gitignore`, `docs/evidence/specification-version.md`
  and ADRs 0034 and 0035. **Stage explicit paths only.**
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs.
- **Clauses served:**
  - document 01: "one rank execution thread per GPU";
  - the prerequisite for the M5 exit's "**PP** … without model edits" on
    devices (task 0071), and for the combined TP/PP plan (task 0072).

## Facts established before writing (coordinator, 2026-09-24)

- **Why not generalize `DenseRankWorkers`** (`dense_tp_workers.rs`, 2,362
  lines, accepted in task 0062): its startup peer-grant handshake,
  `RankRendezvous` (`submissions: [_; 2]`, `arrived == 2`) and every
  command array are pair-specific. Generalizing them would reopen task
  0062's accepted failure-propagation code. A pipeline stage on one GPU
  needs no collective and no rendezvous, so a small separate worker carries
  less risk. Task 0072 uses the pair worker unchanged for its TP2 stage.
- **The single-GPU step protocol** already exists and is accepted. It is
  used in `dense_gemma_device.rs::run_prefill_decode` (around line 243):
  - `DeviceKvSequence::new(geometry)`, a measured `Ledger`, and admitted
    `PagedAttentionRun`s;
  - per step: `SelectedReservedPlan::admit`, `state.begin()` (once per
    transaction), `plan.execute_dense(DenseGraphStep { … })`, `.finish()`
    giving `DenseGraphResult { output, plan, … }`, then
    `result.plan.close(&mut ledger)`.
- **The two-phase commit** exists in `worker_commit`
  (`dense_tp_workers.rs`, around lines 1426–1536): `state.prepare_commit(txn,
  0)`, then `PagedKvWriterAdapter` writers built per run, then
  `state.apply_commit(prepared, &mut writers)`. An apply failure is
  `DeviceLost`, because state may diverge. `state.abort(txn)` rolls back.
- **Several steps may share one transaction.** The TP workers run many
  `Stage` commands between one `Begin` and one `Commit`. A pipeline stage
  needs this, because microbatch 0 and microbatch 1 both append to the same
  stage's KV before one commit.
- **Worker startup** is in `rank_worker` (around lines 1122–1268). After
  the peer grant it does `Stream::new`,
  `CapacitySnapshot::measured(&context.measure()?, 1 << 20)`,
  `Ledger::new([device, host])`, `DeviceKvSequence::new(geometry)` and
  `admit_worker_runs(…)` (around line 1270, a private fn). The solo worker
  needs the same steps, minus the peer grant.
- **An in-flight refusal.** `DensePlanRunRefused { held: Some(lease), … }`
  means device work may still be running on ranges the lease owns. The
  lease must never be dropped. The TP worker's answer is a sticky loss plus
  `park_lost()`.

## Bounded deliverable

- `SoloRankWorker` owns one GPU on its own thread. It runs whole graphs or
  stage graphs through the accepted single-GPU step path. Several steps
  share one open transaction, and the commit is split into
  `prepare_commit` and `apply_commit` so a coordinator can commit many
  workers all-or-nothing.
- On every GPU, a dense Gemma prefill plus decode through the worker is
  **bit-identical** to the same steps through the direct path.
- **Non-goals:**
  - no pipeline coordinator (task 0071);
  - no TP;
  - no change to `DenseRankWorkers`' behaviour;
  - no model edits.

## Numbered changes

1. **`crates/moxie-executor/src/dense_tp_workers.rs`:** make
   `admit_worker_runs` `pub(crate)`. There is no other change to this
   file.

2. **New `crates/moxie-executor/src/rank_worker.rs`**, behind
   `#![cfg(feature = "paged-attention-binding")]`, and declared and
   re-exported in `lib.rs` beside `DenseRankWorkers`:
   ```rust
   #[derive(Debug, Clone)]
   pub struct SoloRankWorkerConfig {
       pub rank: RankId, pub ordinal: u32, pub geometry: KvGeometry,
       pub heads: u64, pub max_rows: u64,
       pub host_capacity: CapacitySnapshot, pub deadline: Duration,
   }
   pub struct SoloRankWorker { /* command Sender, JoinHandle, deadline, lost: Option<Error> */ }
   impl SoloRankWorker {
       pub fn spawn(config: SoloRankWorkerConfig) -> Result<Self>;
       /// Run one graph step inside the open transaction (beginning one if
       /// none is open). Returns the graph output's bytes.
       pub fn step(&mut self, graph: Graph, catalogue: KernelCatalogue,
                   bindings: Vec<OwnedBinding>, rows: u64, visible_tokens: u64) -> Result<Vec<u8>>;
       pub fn prepare_commit(&mut self) -> Result<()>;
       pub fn apply_commit(&mut self) -> Result<()>;
       /// Roll back the open transaction, if any. A prepared commit is dropped first.
       pub fn abort(&mut self) -> Result<()>;
       /// (published rows, committed rows, outstanding ledger reservations)
       pub fn stats(&mut self) -> Result<(u64, u64, usize)>;
       pub fn close(self) -> Result<()>;
   }
   ```
   - **Thread:** `spawn` starts one thread named
     `moxie-solo-rank-{ordinal}`. It acquires
     `RankContext::acquire(rank, ordinal)` and initializes exactly as
     `rank_worker` does after its peer grant (the steps in the Facts), then
     replies ready or its error. Every CUDA value stays on that thread;
     commands carry only owned data.
   - **`step`:**
     1. Lower with `lower_selected` on the worker's own capability, with a
        `ResourceWorkload` built as `run_stage` builds it (the phase from
        `rows`, `branch_rows = rows`, `output = graph.output()`,
        `paged_state_capacity: None`).
     2. `SelectedReservedPlan::admit`.
     3. `state.begin()` if no transaction is open.
     4. `execute_dense` with `host_experts: &[]`, then `finish`, then
        close the plan into the ledger, then return `output`.
     - If the refusal holds a lease (`held.is_some()`): record a sticky
       `DeviceLost` naming the ordinal, `std::mem::forget` the lease, reply
       with the error, and `park_lost()`, exactly as the TP worker does.
     - Otherwise close the returned plan and reply with the error. The
       transaction stays open for the caller to abort.
   - **`prepare_commit`:** `state.prepare_commit(txn, 0)`. Store the
     `PreparedCommit` in the thread's state. Refuse if there is no open
     transaction or a commit is already prepared.
   - **`apply_commit`:** build the writers as `worker_commit` does, then
     `apply_commit(prepared, …)`. A failure is a sticky `DeviceLost`.
     Clear the transaction on success.
   - **`abort`:** drop any prepared commit, then `state.abort(txn)`. With
     no open transaction it is a no-op.
   - **Deadlines:** every reply is awaited with the existing `recv_until`
     at `deadline`. A timeout is a sticky `DeviceLost`. Every later call
     then returns that error without sending a command.
   - **`close`:** abort any open transaction, close every run into the
     ledger, and assert that `ledger.outstanding()` is empty (refuse
     otherwise). Then join the thread. `Drop` without `close` sends a
     shutdown and joins, and never blocks past the deadline.
   - **Test hook:** under `paged-attention-test-hooks`, add
     `pub fn fail_next_step(&mut self)`. The next `step` returns
     `InvalidRequest { field: "fault", .. }` before admitting anything.
   - **Size:** about 250–350 lines. Reuse `recv_until`, `park_lost`,
     `device_lost` and `commit_capacity_error` from `dense_tp_workers.rs`
     (make them `pub(crate)` if needed; that is allowed by change 1's
     spirit, so list each in the Result). Do not copy them.

3. **`crates/moxie-executor/tests/dense_gemma_device.rs`:** add **one**
   test, `solo_rank_worker_matches_the_direct_path_on_every_gpu`, reusing
   the file's helpers. For each GPU and dense Shape A:
   - **Reference:** the direct path's prefill (5 rows) and decode (1 row)
     output bytes. Reuse `run_prefill_decode` or its pieces.
   - **Worker run:** `step` (prefill), then `prepare_commit` and
     `apply_commit`; then `step` (decode), then prepare and apply again.
     Both outputs must be **bytewise equal** to the reference.
   - **Abort:** `step` the next decode row, then `abort`. `stats()` must
     equal its value before that step. Then the same decode row through
     `step` + prepare + apply must equal a direct-path third step computed
     in the reference.
   - **Fault:** `fail_next_step`, then `step` returns `InvalidRequest`.
     `abort`, then a clean step succeeds.
   - **Close:** `close()` returns `Ok`, which proves no outstanding
     reservation.

## Allowed files

- `crates/moxie-executor/src/rank_worker.rs` (new)
- `crates/moxie-executor/src/lib.rs` (the module declaration and
  re-export)
- `crates/moxie-executor/src/dense_tp_workers.rs` (visibility changes
  only)
- `crates/moxie-executor/src/dense_tp.rs`: visibility only, `mod workers` becomes `pub(crate) mod workers` (amended 2026-09-24 after the builder's DECISION)
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
  dense_gemma_device` on all three GPUs. The new test plus every existing
  case must pass.
- `cargo test -p moxie-executor --features
  driver,paged-attention-binding,paged-attention-test-hooks --test
  dense_tp2_device`, to check the TP workers are unaffected.
- `cargo xtask-cuda test-gpu` (no regression).

**Mutations**, each applied, run, shown failing and restored:
1. `step` begins a **new** transaction on every call, instead of only when
   none is open. The abort check then fails: the earlier step's rows are
   committed or lost.
2. `abort` does not call `state.abort`. The stats-unchanged assertion
   fails.

**Stop conditions:**
- Aborting after `prepare_commit` is not supported by `moxie-state`
  (dropping a `PreparedCommit` and then aborting fails). Report it; do not
  change `moxie-state`.
- The worker output is not bytewise equal to the direct path.
- A mutation survives.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Implemented `SoloRankWorker` as one named worker thread per GPU. Context,
  stream, capability, ledger, device state, admitted runs, plans, prepared
  commits and leases stay on that thread; commands carry owned host values.
  Steps share an open transaction until apply or abort. Replies use bounded
  waits and make `DeviceLost` sticky; a held resource/lease is forgotten and
  the worker parks. `close` aborts, closes runs, checks the ledger and joins.
- Reused `admit_worker_runs`, `recv_until`, `park_lost`, `device_lost` and
  `commit_capacity_error` from `dense_tp_workers.rs`. All five helper
  visibility changes are crate-private; `dense_tp.rs::workers` is also
  crate-private under the authorized amendment. No TP worker behavior changed.
- The prepared-then-abort stop condition is supported by `moxie-state`:
  `DeviceKvSequence::prepare_commit` takes `&self` and builds a read-only
  `PreparedCommit`; dropping it leaves the open transaction for `abort`.
- `solo_rank_worker_matches_the_direct_path_on_every_gpu` passes prefill,
  decode and retried-decode byte comparisons on all three GPUs. It also checks
  two steps in one aborted transaction, unchanged stats after abort, injected
  pre-admission refusal followed by a clean step, and close with no outstanding
  reservations.

### Gates

- Passed `cargo fmt --all -- --check` and `git diff --check`.
- Passed `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- Passed `cargo clippy -p moxie-executor --all-targets --features driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D warnings`.
- Passed `cargo test --workspace --locked`.
- Passed `cargo xtask arch-check` (79 rejected fixtures, 21 accepted,
  13 rules) and `cargo xtask spec-check` (10 documents unchanged).
- With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, passed the unfiltered
  `dense_gemma_device` suite (5 tests), `dense_tp2_device` (1 test), and
  `cargo xtask-cuda test-gpu` (63/63 cases on sm_120 and sm_86; all three
  physical GPUs qualified).

### Mutations

1. Changed `WorkerState::step` to call `state.begin()` on every step and store
   that transaction. The focused worker test failed on the second step of the
   abort transaction with `InvalidRequest: a transaction is already open here`.
   Restored transaction reuse.
2. Removed `self.state.abort(transaction)?` from `WorkerState::abort`. The
   focused worker test failed the stats check: published rows were 8 instead
   of the pre-abort value 6. Restored the abort call.

### Review map

| File | Added | Removed | Change |
|---|---:|---:|---|
| `crates/moxie-executor/src/rank_worker.rs` | 597 | 0 | New solo worker implementation, including the R1/R2 bounded join. |
| `crates/moxie-executor/src/dense_tp_workers.rs` | 5 | 5 | Made five shared helpers `pub(crate)`. |
| `crates/moxie-executor/src/dense_tp.rs` | 1 | 1 | Authorized visibility change for `workers`. |
| `crates/moxie-executor/src/lib.rs` | 4 | 0 | Module declaration and public re-exports. |
| `crates/moxie-executor/tests/dense_gemma_device.rs` | 153 | 5 | One worker test; direct-reference helper returns an optional third decode. |
| **Total** | **760** | **11** | The worker source is 597 lines, above the 250–350 line estimate; the added lines cover typed command replies, worker-owned CUDA lifetimes, bounded waits and held-resource failure paths. |

The carried `.gitignore`, `docs/evidence/specification-version.md`, and ADRs
0034/0035 were not staged. No single-thread path was changed or removed.

### R1 fixes

- Added `join_until`, which polls `JoinHandle::is_finished()` at 1 ms intervals
  until the deadline, then joins if finished or detaches on timeout. `close`
  returns an ordinal-specific `DeviceLost` if the worker misses that deadline;
  `Drop` ignores the join result.
- When loss is already sticky, `close` returns that error and `Drop` detaches
  without sending `Shutdown` or waiting. A shutdown reply that reports loss also
  causes `Drop` to detach immediately.
- Inlined the former `StepCommand` fields into `WorkerCommand::Step`; its
  `Graph` payload is boxed to keep the enum compact under clippy. Added no tests.
- R1 gates passed: fmt, both clippy gates, `cargo test --workspace --locked`,
  and the unfiltered `dense_gemma_device` suite on all three GPUs (5/5). The
  full `xtask-cuda test-gpu` was not rerun in this review round, as directed.

### R2 fixes

- `join_until` now distinguishes a clean join, a panicked thread, and deadline
  expiry with detachment. `close` maps panic and timeout to distinct
  ordinal-specific `DeviceLost` details; `Drop` ignores each outcome.
- R2 gates passed: fmt, both clippy gates, and `cargo test --workspace --locked`.
  GPU suites were skipped because this teardown change is host-only.
