# ADR 0039 — NCCL carries tensor-parallel joins

- **ID / date / author / status:** 0039 / 2026-09-25 / recorded by the coordinator on the owner's ruling / **accepted**
- **Classification:** **owner requirement.** It adopts a third-party library ("yes, go with NCCL", 2026-09-25).
- **Scope and owning shared component:** the TP2 join (`Join::Reduce`, `Join::Gather`) in `moxie-executor/src/dense_tp_workers.rs`, a new NCCL binding in `moxie-cuda`, and the admission of NCCL buffers in the memory authority.
- **Relation to ADR 0036:** consistent with it, and does not supersede it. ADR 0036 clause 3 requires the combine order to be fixed. With two ranks, `a + b == b + a` exactly, so an NCCL FP32 sum is exact. A third-party collective with more than two ranks may carry data reductions only under a declared order. This ADR admits NCCL for **two-rank** data reductions and for bit-copy gathers.

## Problem and mechanism

M5's TP join goes through the host at every join of every step. Each rank's
thread records an event and spins until the GPU finishes. The two threads
exchange peer-read handles over a channel. Each thread waits for the peer's
acknowledgement, then peer-copies and reduces
(`dense_tp_workers.rs`, `copy_join`, about 1935–1990). A CUDA graph cannot
contain those host waits, so TP decode cannot be captured (M6.2), and the host
round trips cost time at every layer.

## What Strata did

`strata/docs/dsv4-rank-local-architecture.md`, "Collectives" and "Failure
and rollback", and `kernels/cuda/deepseek_rank_local_layer_executor.cu`:
- one persistent communicator per rank, created with `ncclCommInitAll` in
  one process;
- a data all-reduce of FP32 partials with `ncclSum` ("exact-order for the
  declared contract");
- a one-word status all-reduce with `ncclMax`. A failing rank writes a
  nonzero word, and **every rank enters every collective**, so no rank is
  left waiting on a peer that already returned;
- a BF16 all-gather for the vocabulary head;
- NCCL buffers admitted as a named reserve, 64 MiB per device.

## Decision

1. **The binding.** `moxie-cuda` binds the system `libnccl.so.2` (2.31.2,
   `/usr/lib/x86_64-linux-gnu`, header `/usr/include/nccl.h`) behind an
   `nccl` feature. The loaded version is checked against the build-time
   header version at communicator creation, as ADR 0037 does for cuBLAS. No
   other crate links it.
2. **Communicators.** One persistent communicator per rank, created once
   at worker spawn and destroyed at close, bound to the rank's context and
   stream. Each rank thread creates its own communicator with
   `ncclCommInitRankConfig`, sharing one unique id. That is equivalent to
   Strata's `ncclCommInitAll`, and required by Moxie's explicit per-thread
   contexts (amended after sol's design review of task 0100).
3. **Joins.**
   - `Join::Reduce`: `ncclAllReduce` of the FP32 partials with `ncclSum`,
     then the existing single rounding to BF16 on each rank.
   - `Join::Gather`: `ncclAllGather` of the BF16 or FP32 parts, then the
     existing row interleave.

   Both run on the rank's stream, so no host thread waits inside a step.
4. **Status and failure.**
   - Each join is preceded by a one-word `ncclMax` status all-reduce. A rank
     that fails before a join writes a nonzero word and still enters that
     join's collectives.
   - A nonzero result poisons the step on every rank, and the step is
     aborted.
   - A deadline or asynchronous error (`ncclCommGetAsyncError`) aborts the
     communicator (`ncclCommAbort`), and the group becomes lost, as the M5
     lost-group rule does today.
5. **Admission.** Each rank's NCCL buffers are charged in the memory
   authority before the communicator is created, using a measured bound
   recorded with its measurement.
6. **Exactness.** TP2 on the unordered (cuBLAS, ADR 0037) and ordered
   catalogues stays byte-identical to the single-GPU split reference (ADR
   0036).

7. **Pipeline hand-offs (owner, 2026-09-26, "yes, use NCCL").** NCCL
   point-to-point `ncclSend`/`ncclRecv` also carries activations between
   pipeline stage GPUs. The 5060 Ti has no peer access to the 3090s. Each
   stage pair gets its own communicator, created and admitted as in
   decisions 2 and 5, with the failure rules of decision 4. Bit copies
   only; no reduction.

## Consequences and costs

- A new dependency with its own threads and memory, managed through
  admission and abort.
- M5's fault-injection guarantees (rank failure, collective mismatch,
  stalls) must be re-qualified on the NCCL path.
- It is the prerequisite for capturing TP decode (M6.2) and for persistent
  multi-GPU execution (M6.1).

## Enforcement and removal

- Data reductions over NCCL are restricted to two ranks until a declared
  order for more is specified (ADR 0036 clause 3).
- The TP bit-identity test stays a gate.
- Revisit if NCCL cannot meet the failure guarantees on this hardware.
