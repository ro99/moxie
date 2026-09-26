# Task 0100 — NCCL communicators and TP joins

Status: **active** (coordinator, 2026-09-25). Sol's design review (4 high,
4 medium, 1 low) is adopted verbatim in "Design review" below. Where it
conflicts with the numbered changes, **it overrides them**.
Builder Codex `luna`; reviewer Codex `sol`. Asynchronous-ownership work:
change 7 is the escape inventory.

## Identity and authority

- Task0100, M6 roadmap **M6.1** (device-resident execution) and the
  prerequisite of M6.2's TP capture (task 0103). It implements
  [ADR 0039](../decisions/adr/0039-nccl-for-tensor-parallel-joins.md) (owner
  ruling: NCCL for TP joins, following Strata). Read the ADR first; it is
  binding.
- Ledger finding (2026-09-25): the TP join goes through host handshakes on
  every join (`dense_tp_workers.rs` `copy_join`, about 1900–2066). This
  task moves the join onto NCCL, on the rank streams. The per-step
  orchestration (admit, drain and close per stage) stays; task 0102
  replaces it.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**; on a conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** GPUs free;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Naming rule applies.

## Facts established before writing (coordinator, 2026-09-25)

- NCCL 2.31.2 is installed: `libnccl2` and `libnccl-dev`,
  `/usr/lib/x86_64-linux-gnu/libnccl.so.2`, `/usr/include/nccl.h`.
- `rank_worker` (`dense_tp_workers.rs` about 1131–1348) runs one thread per
  rank:
  - it acquires a `RankContext`, exchanges context tokens and enables peer
    access, then waits for a startup outcome;
  - rendezvous channels carry the startup messages;
  - `startup_deadline` bounds startup.
- `copy_join` (about 1900–2066) does the following:
  1. exports a peer-read handle and sends it to the peer;
  2. records a producer event and spins on it with the group deadline;
  3. receives the peer handle, peer-copies into a scratch buffer, and waits
     for `owner.finish(deadline)`;
  4. reduces (`TP_REDUCE_F32`) or interleaves (gather).

  `prepare_join` (about 1833) sizes the temporary ranges in
  `CollectiveTemp`.
- Group failure (about 700–880): a lost group (`self.lost`) parks with its
  contexts; `DenseRankWorkers::drop` skips shutdown when lost.
- The fault hooks exercised by `dense_tp2_device.rs`:
  `inject_collective_mismatch_once`, `stall_before_next_rendezvous`,
  `set_deadline_for_test` and `refuse_next_commit_prepare`.
- Strata's reference: `strata/docs/dsv4-rank-local-architecture.md`,
  "Collectives" and "Failure and rollback", and
  `kernels/cuda/deepseek_rank_local_layer_executor.cu` (about 731–763,
  945).

## Bounded deliverable

- **Outcome:** every TP2 `Join::Reduce` and `Join::Gather` runs as NCCL
  collectives on the rank streams, preceded by a status collective. The
  host peer-read handshake is removed from joins. TP2 stays
  **byte-identical** to the single-GPU split reference, with the ordered
  and the cuBLAS catalogues. Every M5 fault test still passes. NCCL device
  memory is admitted.
- **Allowed files:**
  - `crates/moxie-cuda/{build.rs,Cargo.toml,src/lib.rs,src/ffi.rs}`, plus a
    new `src/nccl.rs`;
  - `crates/moxie-kernels/{cuda/dense_ops.cu,src/lib.rs}` (the one-input
    FP32 → BF16 conversion kernel, sol H4);
  - `xtask/src/gpu.rs` (one new `test-gpu` case qualifying that kernel;
    authorized by the coordinator answering luna's DECISION, 2026-09-25);
  - **deletions only:** `crates/moxie-executor/src/arena.rs`
    (`export_peer_read_at` and `copy_from_peer_read_at` wrappers), plus
    `crates/moxie-cuda/src/driver.rs` and `src/lib.rs` (`PeerReadHandle`,
    `PeerReadOwner` and their methods, re-exports). Nothing else uses the
    peer-read mechanism once joins move to NCCL (coordinator, answering
    luna's DECISION; checked by grep);
  - `crates/moxie-executor/{Cargo.toml,src/dense_tp_workers.rs}`, and
    `src/rank_worker.rs` if the rendezvous lives there;
  - `crates/moxie-executor/tests/dense_tp2_device.rs` (one new test) and
    `dense_tp2_cublas_device.rs` (feature line only);
  - this task's Result.
- **Non-goals:**
  - pipeline handoffs (`pipeline.rs`) and persistent plans (task 0102);
  - capture (task 0103);
  - more than two ranks;
  - timing.

## Numbered changes

1. **Binding (`moxie-cuda`).**
   - Feature `nccl = ["driver"]`. `build.rs`: when it is on, link
     `dylib=nccl` from `/usr/lib/x86_64-linux-gnu`, and panic loudly if it
     is absent. Parse `NCCL_MAJOR`, `NCCL_MINOR` and `NCCL_PATCH` from
     `/usr/include/nccl.h` into an env var.
   - `ffi.rs`:
     - `ncclGetVersion`, `ncclGetUniqueId`;
     - `ncclCommInitRankConfig` (with `ncclConfig_t` and `blocking = 0`);
     - `ncclCommGetAsyncError`, `ncclCommAbort`, `ncclCommDestroy`;
     - `ncclAllReduce`, `ncclAllGather`, `ncclGetErrorString`;
     - the needed enums.
2. **`nccl.rs`.**
   - `pub struct NcclId([u8; 128])` is `Copy + Send`.
   - `pub struct Communicator<'ctx>` is bound to one `RankContext`, not
     `Clone`, not `Send`. Methods:
     - `pub fn init(ctx, ranks: i32, id, rank: i32, deadline) ->
       Result<Self>`, a non-blocking init, polling `ncclCommGetAsyncError`
       until `ncclSuccess` or the deadline. On timeout or error, it calls
       `ncclCommAbort` and returns the error;
     - `pub unsafe fn all_reduce_f32_sum(&self, send: u64, recv: u64, count,
       stream)`;
     - `all_reduce_u32_max(...)`;
     - `all_gather(&self, send, recv, count_per_rank, dtype: Bf16|F32,
       stream)`;
     - `pub fn async_error(&self) -> Result<()>`;
     - `pub fn abort(self)`;
     - `pub unsafe fn destroy(self) -> Result<(), (Self, Error)>` (the
       checked pattern from tasks 0092 and 0096).
   - Every call makes the context current. `init` checks the loaded
     `ncclGetVersion` against the header version, and refuses a mismatch
     naming both.
   - `Drop` does not destroy: an undestroyed communicator leaks
     (quarantine), like a live module.
3. **Admission.**
   - Measure once on this machine: the device free-memory delta across
     `Communicator::init` plus one 1 MiB all-reduce, per rank.
   - Record it as `NCCL_DEVICE_RESERVE_BYTES`, rounded up to the next
     16 MiB, with the measurement in a comment.
   - Each rank charges it in its ledger before `init`, in the device tier
     the memory authority uses for collective or scratch memory. If none
     fits, send `DECISION` naming the options.
   - The charge is released only after a successful `destroy`.
4. **Startup.** After the peer-access outcome succeeds (the existing
   rendezvous):
   - rank 0 calls `ncclGetUniqueId` and sends the `NcclId` to rank 1 over a
     new startup channel;
   - both ranks call `Communicator::init(ctx, 2, id, rank,
     startup_deadline)` concurrently;
   - a failure on either rank makes startup fail as today (both report),
     and a partly initialised communicator is aborted.
5. **Joins (`copy_join` replaced).** Per join, on the rank stream:
   1. a status word in `CollectiveTemp` (4 bytes, admitted with the temps)
      is set to 0, or 1 if this rank's stage for the join failed after
      `prepare_join`;
   2. `all_reduce_u32_max(status → status_result)`;
   3. the data collective:
      - `Join::Reduce`: `all_reduce_f32_sum(partial → reduced)`, then the
        existing one-rounding BF16 conversion (`TP_REDUCE_F32`'s rounding
        path, or its single-input equivalent). Note: `a + b` in NCCL, then
        one rounding, is what the split reference computes;
      - `Join::Gather`: `all_gather(part → gathered)`, then the existing
        row interleave.

   The host does not wait inside a join. `status_result` is read at the
   step's existing drain. If it is nonzero, both ranks fail the step with
   one typed error, the transactions abort as today, and the group stays
   usable. **Every rank enqueues every join's collectives**, including a
   rank whose stage failed (with the admitted temps; their contents are
   ignored).

   Remove the join use of `PeerReadHandle` and its channels. Keep peer
   access grants.
6. **Deadlines and loss.**
   - Every host wait on a rank stream that carries NCCL work (drain,
     finish, close) polls completion and `async_error()` against the group
     deadline.
   - On expiry or an async error, call `Communicator::abort` on **both**
     ranks, which unblocks NCCL kernels. The group becomes lost under the
     existing rule, with contexts parked.
   - `stall_before_next_rendezvous` must still produce a lost group within
     the deadline, not a hang.
7. **Escape inventory.**

   | Resource | Rule |
   |---|---|
   | Communicator | Created at startup. Destroyed in `close` after both streams are drained, before the context is released; a refused destroy keeps it and the group. On a lost group it is aborted, not destroyed, and the reserve stays charged. `Drop` of an unclosed group leaks it. |
   | Collective temps (status word, reduced/gathered buffers) | Follow the existing `CollectiveTemp` rules. They are released only after the step's drain observes completion. A collective that is still enqueued keeps them. |
   | Startup failure after one rank's `init` | That communicator is aborted. |
8. **Tests.**
   - (a) The whole `dense_tp2_device` suite passes unchanged on the NCCL
     path, including its byte-identical TP-versus-single comparison and
     every fault test. `dense_tp2_cublas_device` passes too.
   - (b) New, `nccl_status_poisons_both_ranks`: inject a failure on rank 1
     after `prepare_join` (a test hook). Both ranks return the same typed
     error, neither hangs, the transactions abort, and the next step
     succeeds and is byte-identical.
9. **Coverage check (one mutant, reverted after; run only test (b)).** Make
   a failing rank skip the collectives instead of entering them. Test (b)
   must fail (by deadline and loss, not by success).

## Acceptance

Host gates:
- fmt;
- workspace clippy;
- executor driver clippy, with and without `nccl` and `cublas`;
- `cargo test --workspace --locked`;
- arch-check (it must name NCCL if it polices native links);
- spec-check.

GPU gates, once, with `nccl,cublas`:
- the full `dense_tp2_device`;
- `dense_tp2_cublas_device`.

No `test-gpu`, and no timing. The reviewer may start at the candidate
commit.

**Stop conditions:**
- NCCL cannot use Moxie's explicitly created contexts (as tested for
  cuBLAS in task 0096);
- the NCCL reduce followed by one rounding is not byte-identical to the
  split reference;
- abort does not unblock a stalled peer within the deadline;
- a file outside the allowed list is needed.

## Design review (sol, 2026-09-25) — adopted verbatim; overrides the numbered changes

- **H1 (change 5).** Stage and `JoinPrepare` failures use the existing
  paired rendezvous and settle; no NCCL join starts. After both
  preparations succeed, each `JoinCopy` enqueues both collectives despite a
  local post-prepare failure, using a valid admitted dummy source if
  necessary.
- **H2 (changes 5, 7).** After each paired `JoinCopy`, pair a bounded
  `JoinDrain`. It observes both streams and the status, then retires the
  source, the temp and the plan before scheduling the next `Stage`. A
  nonzero status aborts both transactions before later stages. Task 0102
  may remove this host barrier.
- **H3 (changes 6, 7).** Each responsive rank aborts its own communicator
  from its deadline or async-error polling path before parking. The
  coordinator marks the group lost and keeps unreachable ranks parked and
  charged. There is no cross-thread abort.
- **H4 (change 5).** Add a one-input round-to-nearest FP32 → BF16
  conversion kernel (`dense_ops.cu`, and its constant in `lib.rs`), and use
  it after the NCCL sum. `TP_REDUCE_F32` is not reused, because NCCL has
  already summed. It must match the split reference's single rounding,
  including signed zero. A `test-gpu` case qualifies it (kernel source
  changes, so `test-gpu` is **required** for this task).
- **M1 (change 8a).** The existing peer-copy fault injection no longer
  fires once peer-copy joins are deleted. Replace only that injection with
  an NCCL enqueue or async failure, preserving its recovery assertions.
  Keep the host declaration-mismatch guard.
  **Amended (coordinator, answering luna's DECISION):** replace the
  peer-copy injection with the **post-prepare status-word failure**. It
  still enters the NCCL collectives and keeps the same-group recovery
  assertions. A real NCCL failure loses the communicator (H3), and the
  existing stall/deadline test covers it as a lost group, not a hang.
- **M2 (changes 1, 2, 7).**
  - Use `NCCL_CONFIG_INITIALIZER` (size, magic, version and the UNDEF
    fields) before setting `blocking = 0`.
  - Handle `ncclInProgress` during init and finalize.
  - Destroy only after finalization has completed. On refusal, return
    `Self` only while its handle is still live.
- **M3 (changes 3, 5, 7).**
  - The admission tier is `CollectiveBuffers` (`moxie-memory` `tier.rs`
    about 35–37).
  - The status is one in-place device word, written with a stream-ordered
    `cuMemsetD32Async` and kept through `JoinDrain`. That is L1: an
    in-place `ncclAllReduce` on one 4-byte word, with no separate
    `status_result`.
  - Charge all temporary data ranges plus the NCCL reserve. Measure the
    reserve against the largest admitted join as well as a 1 MiB one.
- **M4 (ADR 0039, change 4).** Communicators are created per rank with
  `ncclCommInitRankConfig`, which is equivalent to `ncclCommInitAll` for
  Moxie's explicit per-thread contexts. ADR 0039 decision 2 is amended to
  match.

## Result, filled after work
