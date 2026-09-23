# Task 0062 — one execution thread per GPU, with group-wide failure propagation

Status: **accepted** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Built by Codex `luna`; the coordinator supplied exact fix lists
after sol's early review and after round 1. Reviewed by Codex `sol`: an early
review, round 1, then round 2 ACCEPT with no new findings. The builder ran
the full test-gpu; the coordinator re-ran `fmt`, workspace and driver
`clippy`, `arch-check`, `spec-check` and `dense_tp2_device`.

**Amendment, 2026-09-23 (coordinator, coordinator.md §3).**
- *Old premise:* all three fault injections end with both ranks aborting, KV
  unchanged, and a later exact clean step.
- *Evidence (builder, phase 1):* a stalled rank cannot enter an abort, and
  task 0060's rule already makes a missed drain `DeviceLost`, with resources
  withheld and transactions untouched.
- *Replacement criterion:* the rank-error and collective-mismatch injections
  prove a bilateral abort, reuse of the same group, and an exact next step.
  The stall injection proves that published KV and frontiers are unchanged,
  that no output is published, and that the loss is sticky: the next call on
  the same group returns `DeviceLost`. It does not require an abort or a
  clean step on the lost group. "KV unchanged" means published state; pages
  withheld after a drain timeout are not inspected.
- *Authority:* coordinator. The rule is unchanged from task 0060.

## Identity and authority

- Task0062, second half of M5 plan slice 3 ("Robust rank execution").
  Builder: Codex `luna` (max, `/ponytail:ponytail`). The Claude Opus
  `builder` would normally take a complex task, but the shared Claude account
  was at 86% of its weekly limit on 2026-09-23, and the coordinator runs on
  the same account. The builder stays on standby for a rescue, as on
  task 0060. Reviewer: Codex
  `sol` (read-only, `/ponytail:ponytail-review`). Coordinator: Claude Opus
  `coordinator`. Accepted by the coordinator under the owner's auto-mode
  delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `d6470b7` (after
  tasks 0061 and 0063). Preserve the unrelated carried work
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

- **Design and owners:** the approved persistent rank-owned worker design is
  implemented with owned host values crossing threads, one monotonic
  mutex/condvar rendezvous, MAX status, declaration agreement, deadlines and
  sticky `DeviceLost`. `moxie-cuda` owns peer-read and peer-context grant
  primitives; `moxie-executor` owns workers, rendezvous, settlement and
  commit. `moxie-state` and `moxie-plan` semantics are unchanged. The task
  remains uncommitted on source commit `dcd3733`.
- **Peer-read safety (A1):** `export_peer_read_at` is unsafe and documents
  that its source owns `[address, address+len)` for the entire hold. The
  `DeviceRange` wrapper contains the required safety comment. The range moves
  into `PeerReadOwner`; `signal_ready(&mut self, producer)` verifies the event
  completed, `finish(self, deadline)` returns it only after peer acknowledgement,
  and `copy_join` hands the returned range to the existing plan release path.
  Dropping an unfinished owner forgets the range.
- **Startup safety (A2):** each worker acquires its `RankContext`, sends its
  rank, generation and token, then waits for its peer token. It installs the
  grant on its own thread and reports the result. Control sends `Proceed`
  only after both grants succeed; otherwise it sends `Abort`. On Abort every
  worker waits for control-side `Exit`, including a worker whose own grant
  failed. A worker with a grant disables it on its owner thread, removes its
  generation-keyed grant entry and reports `Disabled`; control waits to the
  group deadline for each such report before sending `Exit` to both. On any
  handshake timeout or disconnect the worker forgets its context and exits.

  `cuMemcpyPeerAsync` takes explicit `dstContext` and `srcContext` arguments
  and a destination-context stream. The destination worker makes only its own
  context current; it passes the source token as the explicit source-context
  argument and never makes the source context current. The group pins both
  owner contexts until peer acknowledgements arrive or the group is lost.
  Startup grants are installed only while the destination context is current
  on its owner thread. The generation-keyed grant table stores a context
  identity, not an owned context moved across threads. A lost worker keeps its
  context and grant table parked or leaked through process teardown; it never
  drops a context while a peer could still use its token. See NVIDIA's
  [CUDA Driver API, `cuMemcpyPeerAsync`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2).
- **Round 2 execution changes:** each rank submits its own actual
  `GatherDeclaration` shape and precision to the rendezvous. Stats uses the
  ordinary `pair_command` status path; Begin performs idle preflight with no
  CheckIdle round; one `settle`, `drain_rank` and `close_plans` path serves
  both callers. Worker rendezvous imports follow `paged-attention-binding`,
  including for the binding-only Gemma lane. The quarantine test path,
  `Noop` command and special stats channel were deleted. Only task 0057's
  `RankGroup::all_gather` remains single-threaded because
  `tensor_parallel_device.rs` still exercises it; no second dense execution
  path remains.
- **Fault assertions:** the mismatch injects unequal rank declarations with
  equal byte products and expects the rendezvous refusal. The mutation that
  sent rank 0's declaration to both ranks was caught. Rank refusal and
  mismatch each have an immediately following clean, bit-exact step. The
  stall fires at `OP_STAGE`, the first stage rendezvous after Begin; it returns
  no `DenseWorkerStep` or logits, leaves published and committed frontier
  mirrors unchanged, and makes the next `stats()` call return the same sticky
  `DeviceLost`. Stats timeout/disconnect sets both `self.lost` and rendezvous
  loss before returning. The published frontier mirror is a read-only host
  diagnostic updated by each owner before its status rendezvous, so it remains
  observable after group loss without issuing another command.
- **Validation:**

  | Gate | Result |
  |---|---|
  | `cargo fmt --all -- --check` | Passed |
  | `cargo clippy --workspace --all-targets --locked -- -D warnings` | Passed |
  | Workspace clippy with `moxie-executor/driver` | Passed |
  | Driver-only clippy (`moxie-cuda` and `moxie-executor`, `driver`) | Passed |
  | Executor all-target clippy with `driver,paged-attention-binding,paged-attention-test-hooks` | Passed |
  | `cargo test --workspace --locked` | Passed, including doctests |
  | `cargo test -p moxie-cuda --lib --locked --features driver` | Passed, 20/20 |
  | `cargo xtask arch-check` | Passed: 79 rejected, 21 accepted, 13 rules |
  | `cargo xtask spec-check` | Passed: 10 documents |
  | `dense_tp2_device` (driver, binding and test hooks) | Passed, 1/1 |
  | `tensor_parallel_device` (driver) | Passed, 4/4 on 3090 UUIDs `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and `GPU-81fe4578-59b2-37c4-421e-287cdac78704` |
  | `dense_gemma_device` (driver and binding) | Passed, 2/2; prefill/decode matched at 0 ULP on all three devices |
  | `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda test-gpu` | Passed: 63/63, 0 skipped/unmeasured; SM86 and SM120 qualified |

  The TP2 order-sensitive prefill and decode match the one-3090 S=2 reference
  bit-for-bit. The full CUDA gate also exercised the 5060 Ti UUID
  `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`.
- **Mutation results and restoration:** each listed mutation was applied,
  run, then restored before the final gates.

  | Mutation | Evidence it was caught |
  |---|---|
  | A failing rank skips its status rendezvous | TP2 gate timed out the peer, marked the group `DeviceLost`, and failed the recoverable-stats assertion. |
  | Rendezvous status changes from MAX to MIN | TP2 gate detected unequal rank outcomes and failed the recoverable-stats assertion. |
  | Owner returns/frees the peer source before consumer completion, including dropping `PeerReadOwner` early | `dropping_an_unfinished_owner_leaks_its_source_range` failed because the probe was dropped. |
  | Commit ignores the full prepare result before applying | TP2 gate observed the missing apply rendezvous as sticky `DeviceLost`; the recoverable-stats assertion failed. |
  | B1 swaps error priority so rank 0's ordinary error wins over rank 1 `DeviceLost` | `rank_one_device_loss_dominates_rank_zero_ordinary_error` failed its expected-`DeviceLost` assertion. |
  | Step 10 sends rank 0's declaration to both ranks | The unequal-declaration case unexpectedly succeeded and failed its typed-refusal assertion. |

- **Review map** (added/deleted lines; carried files excluded from the code
  subtotal):

  | File | +/- | Change |
  |---|---:|---|
  | `crates/moxie-cuda/src/driver.rs` | +390/-5 | Unsafe span contract, peer owner completion, context tokens and grants |
  | `crates/moxie-cuda/src/ffi.rs` | +1/-0 | `cuCtxDisablePeerAccess` binding |
  | `crates/moxie-cuda/src/lib.rs` | +3/-2 | Peer-read API exports |
  | `crates/moxie-executor/src/arena.rs` | +62/-1 | Owned range export/copy wrappers and safety justification |
  | `crates/moxie-executor/src/dense_tp.rs` | +3/-768 | Worker module export; deleted superseded dense path |
  | `crates/moxie-executor/src/dense_tp_workers.rs` | +2362/-0 | Persistent workers, startup barrier, rank declarations, shared command path and frontier mirrors |
  | `crates/moxie-executor/src/lib.rs` | +1/-1 | Export only worker dense-step types |
  | `crates/moxie-executor/src/paged_attention.rs` | +4/-69 | Removed pair-commit helper used only by the deleted path |
  | `crates/moxie-executor/src/tensor_parallel.rs` | +304/-321 | Worker rendezvous and DeviceLost priority; removed unused same-thread helpers |
  | `crates/moxie-executor/tests/dense_tp2_device.rs` | +249/-292 | Exact recovery, equal-byte declaration mismatch and in-flight stall assertions |
  | `docs/tasks/0062-m5-thread-per-rank-execution.md` | +135/-7 | Result and failure-path inventory |

  The code subtotal is `+3379/-1459`; the worker source has 2,362 lines.
  That is 83 fewer lines than round 1's recorded 2,445, below the review's
  rough 180-line estimate because the required startup barrier and published
  frontier diagnostic add code. The listed duplicate channels, commands and
  implementations are removed. No second dense execution path remains.
- **Per-worker failure-path inventory:**

  | Rank-local event | Both workers / control side | Resource and disposition |
  |---|---|---|
  | Context acquisition or token exchange fails | Control waits to the shared startup deadline; acquisition failure or channel closure loses startup. | A worker with a context forgets it on timeout/disconnect; none is dropped while a peer may reference its token. |
  | Peer grant fails | Control collects both reports and sends `Abort`; every worker waits for `Exit`, and every successful grant is disabled and reported first. | Control waits for `Disabled` from each granted rank. Disable failure or timeout loses the group; workers forget their contexts. |
  | Startup token, grant-report, outcome, disable-report or exit wait times out/disconnects | Control or worker reports startup `DeviceLost`; all waits use the group deadline. | Each worker forgets its context before exit; the grant table cannot outlive a freed context. |
  | Worker initialization after `Proceed` fails | The failing worker records group loss; control receives the error or times out. | Worker context remains pinned by the lost group; resources are not reused. |
  | Begin idle preflight refuses | Begin checks idle runs and no open transaction before state changes; both ranks enter the Begin status round. | No separate CheckIdle operation or rendezvous remains. |
  | Begin, stage validation, append, attention, launch or local rank work refuses | The failed rank enters the same declaration/status rendezvous; MAX status and the first non-`DeviceLost` error reach both ranks. Shared `settle` drains, cleans up and aborts both owners. | A completed drain permits reclamation and reuse. Unobserved drain or cleanup loses the group and withholds uncertain resources. |
  | Shape, dtype, operation or sequence differs; a worker misses a rendezvous | Each rank sends its own declaration at the monotonic sequence. A missing arrival reaches the deadline and sets sticky `DeviceLost`; either rank's `DeviceLost` dominates ordinary errors. | No logits publish after loss; both owner contexts remain pinned. |
  | Rank 1 stalls at the first `OP_STAGE` after Begin | Rank 0 reaches the same rendezvous; the deadline makes loss sticky. | Published and committed frontier mirrors stay unchanged, no step/logits return, and the next stats call reports sticky `DeviceLost`. |
  | Peer readiness/copy/event/ack fails | Destination reports failure through the step status path; no source is returned before copy completion acknowledgement. | Successful ack returns the source to its arena. Missing ack leaks/withholds it and loses the group on timeout. |
  | Commit preparation fails | Both ranks enter prepare status; apply is skipped, then both transactions abort after successful settle. | Published frontiers and logits remain unchanged; group reusable if drain/abort complete. |
  | Commit apply or apply-result rendezvous fails | A possibly asymmetric apply becomes sticky `DeviceLost`; publish follows only a successful apply-result MAX rendezvous. | No later reuse of possibly divergent state; worker-owned resources remain pinned. |
  | Stats rendezvous times out or disconnects | `stats()` uses `pair_command`; timeout/disconnect sets both `self.lost` and rendezvous loss before returning `DeviceLost`. | Subsequent calls return the sticky error; the separate stats reply channel is gone. |
  | Worker command channel closes, worker panics, or shutdown join fails | Control marks rendezvous lost and refuses subsequent work. | Worker-owned CUDA resources are never handed to another thread. |

- **Unrelated carried work preserved:** `docs/evidence/specification-version.md`
  remains modified as supplied. ADRs 0034 and 0035 remain present and
  untracked as supplied. Neither is part of this task's code subtotal. No
  moxie-plan change was adapted or edited.
- **Remaining obligations:** none within this task's exit gate. No commit was
  created.
