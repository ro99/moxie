# Task 0062 — one execution thread per GPU, with group-wide failure propagation

Status: **proposed**.

## Identity and authority

- Task0062, second half of M5 plan slice 3 ("Robust rank execution").
  Builder: the Claude Opus session `builder`, which takes complex tasks by
  owner direction of 2026-09-22 (`/ponytail:ponytail`). Reviewer: Codex
  `sol` (read-only, `/ponytail:ponytail-review`). Coordinator: Claude Opus
  `coordinator`. Accepted by the coordinator under the owner's auto-mode
  delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Base: the task 0061
  acceptance commit. Preserve the unrelated carried work
  (`docs/evidence/specification-version.md` and ADRs 0034 and 0035).
- Requirements:
  - Document 01, line 35: "Local single-process control with one rank
    execution thread per GPU is the initial topology. Isolate unsafe CUDA
    state behind rank-owned contexts."
  - Document 04, line 55: rank groups need a monotonically ordered collective
    schedule, consistent sizes and types, timeouts, and group-wide failure
    propagation. "Cancellation cannot leave one rank waiting forever. A
    failed rank invalidates the in-flight transaction."
  - M5 exit: "Inject rank failure and collective mismatch safely in the
    harness."
  - M5 ledger slice 3. Task 0057 recorded that document 01's thread-per-GPU
    requirement is **not yet met**, because one host thread drives both
    ranks.
- O6/O7 are open: no timing. Threads are required by the spec for isolation,
  not introduced as an optimization.

## Facts established before writing (coordinator, 2026-09-23)

**Moxie:**
- `moxie-cuda` `RankContext` (`src/driver.rs` around lines 170–205) is
  deliberately `!Send`/`!Sync`, with a `compile_fail` doctest. A context
  stays on the thread that acquired it, and a device held by another rank is
  refused. Peer grants are recorded by acquisition generation (task 0057).
  The type is already built for one thread per rank.
- Task 0057's `RankGroup` and task 0060's `RankGroup::execute_dense`,
  `drain`, `settle` and two-phase commit (`moxie-executor/src/tensor_parallel.rs`
  and `dense_tp.rs`) hold both `RankContext`s on **one** host thread. A peer
  copy needs the peer's device range on the issuing side, and
  `DeviceRange`/`DeviceBuffer` are `!Send`.
- Task 0060's written rule for reusable versus lost, and its failure-path
  inventory, are the correctness baseline. Each rank thread must preserve
  them.

**Strata** (read-only):
- `docs/dsv4-rank-local-architecture.md`, "GPU and rank": "Each rank owns one
  persistent CUDA context … Ranks are symmetric: neither is a coordinator."
  Strata drove both ranks from **one** host thread through grouped NCCL calls
  (`kernels/cuda/deepseek_rank_local_layer_executor.cu`, the rank loop
  inside `ncclGroupStart`/`ncclGroupEnd`). Moxie's spec asks for more than
  Strata did. Strata's collective **protocol** still carries over.
- "Collectives", status semantics: a failing rank sets its status word, and
  a MAX reduction delivers any nonzero status to every rank. "The failure
  path still enters both collectives — it does not skip them — so the state
  machine closes symmetrically instead of deadlocking one rank against a peer
  that already returned."
- "Failure and rollback": both rank chains abort, token-local mutations are
  truncated, outputs are withheld, the failed command is drained by a single
  owner, and post-failure reuse is exact and tested.

## Bounded deliverable

- **Outcome:** each 3090 rank runs on its **own host thread**, which acquires
  and exclusively owns its `RankContext`, streams, arenas, leases and
  `DeviceKvSequence`. The reduced dense Gemma TP2 step from task 0060 runs
  with that topology and stays **bit-identical** to the one-3090 S=2
  reference, over prefill plus decode.
- **Cross-rank exchange:** a peer copy needs a peer's range. Define the
  smallest `Send` handle that carries only what the destination needs: a
  device address, a length, the owner's acquisition generation, and a way to
  wait on the producer's readiness. It must not let the destination free or
  retain the source. The **owner** frees a range only after the consumer's
  copy is known complete, through a completion signal back across threads.
  Unsafe code stays in `moxie-cuda` behind a safe API. This is the only new
  unsafe surface; justify it.
- **Status protocol,** following Strata: every collective is a rendezvous in
  which **each rank enters with its status**, including a failing rank, and
  leaves with the group status (the MAX). Nonzero means every rank aborts
  and settles under task 0060's rule. The rendezvous has a deadline: a rank
  that never arrives becomes `DeviceLost` for the group and never causes an
  indefinite wait. The collective sequence stays monotonic, and declarations
  (shape, dtype) are agreed inside the rendezvous.
- **Commit:** task 0060's prepare-both-then-apply-both becomes a rendezvous.
  Apply runs only if every rank prepared.
- **Fault injection at graph level (M5 exit):**
  - a rank failure mid-step (its thread returns an error before a
    collective);
  - a collective mismatch (ranks declare different shapes or sequences);
  - a rank thread that stalls past the deadline.

  All three go in the existing TP2 test file. For each: both ranks abort,
  KV is unchanged, nothing is published, the group is reusable (or lost for
  the stall, per the rule), and a later clean step is exact.
- **Non-goals:**
  - no NCCL;
  - no process-per-rank;
  - no 5060 Ti;
  - no PP, MoE or MLA;
  - no timing claims;
  - no change to the reusable-versus-lost rule, except where threads force
    it; if they do, report it as a DECISION.

## Phase 1 — design proposal before code

Send a `DECISION` report of 50 lines or fewer covering:

1. **The thread model.** Who spawns the rank threads, how work reaches them,
   how their results return, and how `RankContext` stays on its thread.
2. **The cross-rank range handle.** Its fields, how it is `Send`, how
   producer readiness and consumer completion cross threads, and how the
   owner is prevented from freeing too early.
3. **The rendezvous and status protocol.** Its primitive (for example a
   mutex and condvar with a deadline), the MAX status, the declaration
   agreement, and how a failing rank still enters.
4. **How task 0060's settle, commit and failure inventory map onto per-rank
   threads,** and which parts change.
5. **Whether this fits one task,** and the files involved.

## Acceptance

- Host, driver and GPU lanes pass: `fmt`, workspace `clippy`, driver-only
  `clippy`, workspace tests, `arch-check`, `spec-check`, the TP GPU tests on
  the 3090 pair, and the full `cargo xtask test-gpu`.
- **Bit-identity:** TP2 matches the one-3090 S=2 reference over prefill plus
  decode on task 0060's order-sensitive fixture.
- The three injections behave as specified above.
- **Mutations,** each run and restored:
  - A failing rank skips the rendezvous (must deadlock-free fail, and be
    caught).
  - The status reduce uses MIN.
  - An owner frees its range before the consumer's completion.
  - Commit applies without a full prepare rendezvous.
- One test per invariant. The Result keeps a per-file **Review map** and an
  updated **failure-path inventory** per rank thread.
- **Stop conditions:**
  - A lease, ledger or `SequenceState` semantic change is needed.
  - An unsafe design cannot be justified.
  - Bit-identity fails for a reason that is not a defect.

## Result, filled after work

- Design decision (phase 1) and coordinator answer:
- Changed owners; source commit:
- Commands, GPU UUIDs; passed / failed / skipped:
- Mutation results and restoration:
- Review map:
- Failure-path inventory:
- Remaining obligations:
