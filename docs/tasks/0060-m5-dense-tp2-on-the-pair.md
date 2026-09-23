# Task 0060 — the reduced dense Gemma graph runs TP2 on the 3090 pair

Status: **accepted** (coordinator, 2026-09-23, under the owner's auto-mode
delegation; the owner waived a docs-only review round). Started by Codex
`luna`, then rescued and completed by the Claude Opus session `builder`
after luna looped on a fixture defect. Reviewed by Codex `sol` over six
rounds, the first file by file at the owner's request; every code finding
was closed. Coordinator verification: the full `cargo xtask test-gpu` passed
63/63 (sm_86 and sm_120 qualified); `dense_tp2_device` 1/1,
`tensor_parallel_device` 4/4 and `dense_gemma_device` 2/2 passed on the 3090
pair; driver-only `clippy`, `fmt`, `arch-check` and `spec-check` passed.

## Identity and authority

- Task0060, second half of M5 plan slice 2. Builder Codex `luna` (max,
  `/ponytail:ponytail`); reviewer Codex `sol` (read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus `coordinator`.
  Accepted by the coordinator under the owner's auto-mode delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `9b27041` (task 0059
  acceptance). Preserve the unrelated carried work
  (`docs/evidence/specification-version.md` and ADRs 0034 and 0035).
- Requirements:
  - Roadmap M5.2: "TP2 on the 3090 pair: column/row linears, head/KV
    ownership, global … vocabulary operations, collective ordering and
    failure handling. Compare to single-rank reference."
  - [ADR 0036](../decisions/adr/0036-tp-reductions-are-exact-by-declared-order.md):
    exact reductions; the single-rank path runs the same declared order.
  - Document 04: a failed rank invalidates the in-flight transaction.
  - The M5 ledger's slice 2 obligations: per-rank device KV (M5.1-f), a
    whole rank step as one transaction, and an order-sensitive fixture.
- O6/O7 are open: no timing.

## Facts established before writing (coordinator, 2026-09-22)

**Moxie:**
- Task 0058's `moxie_plan::lower_tensor_parallel` produces
  `Local { nodes, join }` stages, where join is a Gather or an FP32 Reduce,
  together with per-rank params, row ranges and input slices. The per-rank
  **stage subgraph** is built only by the test harness today
  (`moxie-cli/tests/tensor_parallel.rs` `stage_graph`). A device consumer
  needs that construction in production code.
- Task 0059's `SelectedReservedPlan::execute_dense`
  (`moxie-executor/src/dense.rs`) runs one selected graph for one step on
  one GPU, with paged attention and admitted host uploads.
- Task 0057's `RankGroup` (`moxie-executor/src/tensor_parallel.rs`) runs a
  sequenced, peer-checked BF16 all-gather on the 3090 pair. Failures are
  typed and bounded, and a missed drain withholds every range. It has **no
  reduce**, and there is no FP32-output partial `Linear` kernel.
  `moxie_bf16_linear_v1` rounds to BF16.
- `moxie-state` has `SequenceState` transactions with `begin` / `abort` /
  `commit_prefix`.

**Strata** (read-only):
- `docs/dsv4-rank-local-architecture.md`, "Failure and rollback": on failure
  "both rank chains abort; every token-local KV … mutation is truncated; all
  outputs are withheld and zeroed — no partial token is ever published; the
  failed command is drained by a single owner; … Post-failure reuse is exact,
  and is tested." This is the model for rank-step atomicity.
- "Collectives": the data reduce is an FP32 sum of rank partials, with one
  BF16 rounding after.

## Bounded deliverable

- **Outcome:** the reduced dense Gemma 4 graph (unchanged model code),
  lowered by task 0058 for R = 2, executes across the two 3090s:
  - prefill, then decode;
  - per-rank stage graphs on each GPU;
  - task 0057's gather for the head output and the vocabulary;
  - a new exact FP32 reduce for `down_proj` and `o_proj`.

  Logits must be **bit-identical** to the same graph on one 3090 running the
  declared split S = 2 (ADR 0036).
- **New pieces:**
  - An FP32-output partial `Linear`, which is `moxie_bf16_linear_v1`'s loop
    without the final rounding.
  - An exact reduce collective on `RankGroup`: exchange the FP32 partials by
    peer copy, add them in ascending rank order in FP32, round to BF16 once.
    It keeps 0057's sequence and shape agreement, bounded settle, and
    drain-or-withhold.
  - The single-GPU device path honours a declared split S, so it can serve as
    the reference.
  - Production construction of the per-rank stage graph (in `moxie-plan`, or
    justify another owner). The harness helper is then deleted or made to
    call it.
- **Rank-step atomicity, Strata's model:** each rank's step runs inside one
  `SequenceState` transaction covering every layer's KV append. If any rank
  or collective fails, both ranks abort, no logits are published, and the
  group stays reusable. The next clean step then matches the reference
  exactly.
- **Order-sensitive fixture:** choose weights or inputs for which S = 2 and
  S = 1 produce *different* logits bits. Assert that difference once, so the
  bit-identity test can detect a wrong combine order.
- **Non-goals:** no thread-per-rank (slice 3); no status collective
  (slice 3); no 5060 Ti; no MLA or routed ops; no timing; no model edits.

## Phase 1 — design proposal before code

Send a `DECISION` report of 40 lines or fewer covering:

1. **Stage-graph construction:** its owner, and how it relates to the
   harness helper.
2. **How a TP step drives the per-rank stage graphs on the device**
   (`execute_dense` per stage, or another entry), and where the gather and
   reduce sit between them.
3. **The reduce collective:** buffers, order, rounding, and reuse of
   task 0057's settle and withhold paths.
4. **Atomicity:** how one transaction per rank spans the stage runs, and
   what abort does.
5. **The order-sensitive fixture.**
6. **Whether this fits one task,** with files and any new crate edge.

## Acceptance

- Host, driver and GPU lanes pass: fmt, clippy, workspace tests,
  `arch-check`, `spec-check`, and the GPU tests with the correct features on
  the 3090 pair (UUIDs in the output).
- **Bit-identity:** TP2 logits equal the one-3090 S = 2 reference, over
  prefill plus decode, on the order-sensitive fixture.
- **Failure:** inject one rank failure and one collective mismatch
  mid-step. Both ranks abort, KV is unchanged, no logits are published, and
  the next clean step is bit-identical.
- **Mutations,** each run and restored:
  - Reduce in descending order.
  - Round a partial before the reduce.
  - Reference ignores S.
  - Abort only the failing rank.
  - One rank keeps its KV append after a failure.
- One test per invariant. Stop conditions: a lease, ledger or `SequenceState`
  semantic change; bit-identity failing for a reason that is not a defect;
  a model edit.

## Result, filled after work

Builder `builder` (Claude Opus), rescue assignments `task-0060-rescue-1`,
`task-0060-rescue-2` to `task-0060-rescue-6` (the repairs of review
rounds 1 to 5). Base
`98321c4`; no commit made.

- **Design decision (phase 1) and coordinator answers:** the approved design
  is a pure `moxie-plan` stage builder that the harness also calls; an
  additive `execute_dense_stage`; an exact `RankGroup` reduce; one
  transaction per rank per step; and an order-sensitive fixture. It also
  carries C1 (an independent split-aware reference) and C2 (a symmetric
  commit). The round-2 answer to F4 approved an additive, two-phase
  `DeviceKvSequence` commit, `prepare_commit` plus `apply_commit`, with
  `commit()` now defined as prepare followed by apply. A `PreparedCommit` is
  bound to its sequence's identity and to a generation counter that every
  mutation advances. An apply failure is terminal.
- **Rescue diagnosis (round 1):** luna's test never reached a GPU. The
  fixture search could not produce an S=1/S=2 difference of one BF16 ulp or
  more; one exact-cancellation construction replaced it. The audit then
  found:
  - collectives that returned the ranks in the wrong order and copied
    unstrided;
  - collective outputs placed in exactly-sized stage arenas;
  - orchestration in the test that dropped the residual live-out;
  - no abort on stage failure;
  - a truncate-based C2 that `SequenceState` cannot honour at `accept = 0`;
  - atomicity asserts on `committed_rows`, which do not move.
- **Round 2 repairs:**

  | Finding | Repair |
  |---|---|
  | F1 | Every collective byte product and the reduce grid are checked before any allocation, copy or launch. |
  | F2, F5, F7 | Fixed as one class; see the failure-path inventory below. |
  | F3 | `lower_selected_ordered` validates every declared order before selection. The node must exist and be an unbiased `Linear`; blocks must be greater than 0; an unsliced order must divide the input axis; a slice must be one whole declared block of it, matching the local width. |
  | F4 | Two-phase commit: `commit_paged_pair` prepares both ranks (`prepare_commit`, the run-count check, the adapter and reserved writer Vecs) before either applies. |
  | F6 | A host binding that names a resident input is refused. |
  | F8 | `gather`/`reduce` are crate-private, and cfg-gated to the step that has observed its sources complete. |
  | F9 | Fixture row 1 now carries a within-block `+Q, -Q, ≈1` cancellation. |
  | F10 | The three dense-only `chain.rs` helpers are feature-gated. |
  | F11 | `DenseStep` releases logits only through `commit`; dropping it uncommitted settles and aborts both ranks. |

- **Round 3 repairs:**

  | Finding | Repair |
  |---|---|
  | F3 | Both slice validators, `lower_selected_ordered` and `moxie-interp`'s `validate_linear_slice`, now also require `first % width == 0`, so a slice is one whole declared block. Each existing test gained the straddling case. |
  | N1 | A plan or boundary that cannot close, a boundary reservation that cannot be released, or an admission `Held` refusal now marks the group lost instead of being dropped from a reusable group. |
  | N2 | A retained stage lease, Lost or not, is reclaimed after `settle`'s drain has been observed (`OperationLease::reclaim_drained`, crate-private and used only there). The single-GPU lease contract is unchanged. |
  | N3 | `DenseStep::logits()` is a read-only view before commit. |

- **Round 4 repairs:**

  | Finding | Repair |
  |---|---|
  | R3-1 | Boundary sizing uses checked products, `checked_next_multiple_of` and a checked sum, and runs before either transaction begins. The reported repro (Shape A, rows = 192153584101141162) is now a typed refusal with nothing to settle. |
  | R3-2 | Recoverable, following the coordinator's Strata-precedent decision. After `settle`'s observed drain, each rank's quarantined `PagedAttentionRun` is reclaimed (`reclaim_drained`, crate-private and feature-gated). Its held query and output ranges return to the owning stage plan's arena (`SelectedReservedPlan::release_reclaimed`). Task 0038's single-GPU quarantine and `close` refusal are unchanged. |
  | R3-3 | Inventory rows fixed: rank-0 boundary admission before a rank-1 refusal, and `enqueue_reduce`'s peer copy and launch. The unreachable abort-if-not-lost branch was deleted. |

- **Round 5 repairs (R3-2 narrowed):**
  - A step accepts only idle runs. Any supplied run that is quarantined,
    or holds a refused source or range, is refused with a typed error before
    either transaction begins, and is left untouched.
  - Every run the step accepted is then used only on its rank's stream, so
    the reclaim in `settle` applies only to a run this step quarantined, on
    a stream `settle` observed drained. It drops only host sources that
    step created.
  - Task 0038's standalone quarantine and `close` refusal are unchanged.

- **Round 6 repair (R5-1):**
  - The run's admitted page-table upload buffer now has its own
    `RefusedSource::PageTableUpload` variant, so it is told apart from a
    caller's query by type, not size.
  - After the observed drain, `reclaim_drained` restores it to
    `page_table_upload` instead of dropping it, and drops only the step's
    own append rows.
  - Any other held variant, which no TP step creates, is refused. That
    keeps the run quarantined and loses the group.
  - Task 0038's standalone paths are unchanged: the variant only renames
    what `publish_page_table` holds, and its `Drop` still forgets it.

- **The rule** (every site below is classified against it). If a
  failure's outstanding GPU work is **observed complete**
  (`RankGroup::drain` succeeded) and nothing irreversible happened (no
  apply or publish), then every resource is reclaimed and returned: plans
  closed, ranges released, reservations freed, retained leases reclaimed.
  Both transactions are aborted and the group stays **reusable**. The
  group becomes **lost**, with resources withheld (dropped unreleased,
  still charged) and `DeviceLost` returned, only if completion was not
  observed within the deadline, apply or publish failed, or cleanup itself
  could not complete. `drain` records an event on each rank's stream and
  polls each independently against the group deadline. Step-level
  failures reach it through `settle` (drain, reclaim leases, close plans
  and boundaries, abort).
- **Failure-path inventory.** It was derived mechanically: `grep` for `?`,
  `return Err`, `map_err`, `Refused` and `let _ =` over `dense_tp.rs`,
  `join`/`drain`/`release_all` in `tensor_parallel.rs`,
  `execute_dense_stage`/`finish` in `dense.rs`, and `commit_paged_pair`.
  Line numbers are from the final tree.
  - "Pure": a check before anything is allocated or enqueued; nothing is
    outstanding.
  - "→ settle": the error propagates to `execute_dense`, which calls
    `settle(abort)`: reusable after an observed drain, otherwise lost.

  | Site | Outstanding there | Class |
  |---|---|---|
  | `dense_tp.rs:239` group already lost | None | Refused with the recorded reason |
  | `dense_tp.rs:243`, `666-700` `boundary_bytes`/`value_bytes` (checked products, `checked_next_multiple_of`, checked sum; R3-1) | None: **before either `begin`** | Pure refusal; nothing to settle |
  | `dense_tp.rs:244-258` a supplied run is not idle: quarantined, or holding a refused source or range from earlier work on any stream (R3-2 narrowed) | Whatever that earlier work left; untouched | Refused with a typed error **before either `begin`**; the run keeps its quarantine and held bytes (`is_idle`, `paged_attention.rs:1389`) |
  | `dense_tp.rs:260-265` `begin` | Rank 0's transaction if rank 1 refuses | Rank 0 aborted (host) → reusable |
  | `dense_tp.rs:271`, `703-717` boundary request or admission | Rank 0's boundary if rank 1 refuses | → settle |
  | `dense_tp.rs:718-738` boundary arena create refused | Its reservation | Released → settle; a release failure → **lost** |
  | `dense_tp.rs:321`, `357` `build_stage_graph` | Earlier plans and boundaries in `Held` | → settle |
  | `dense_tp.rs:333`, `367` `run_stage` | Its plan, lease or quarantined run (below) | → settle |
  | `dense_tp.rs:337`, `575-590` `keep` allocate or copy | The boundary (range inserted before the copy) | → settle |
  | `dense_tp.rs:340` mid-step replicated drain | Plans being copied from | Not observed → **lost** |
  | `dense_tp.rs:342`, `402`, `650-663` `close_plans` refused | The refused plan | Cleanup cannot complete → **lost**, withheld |
  | `dense_tp.rs:397` collective refused | Stage plans and boundaries; `join` settled its own ranges | → settle |
  | `dense_tp.rs:413-415` logits missing or readback failed | Boundaries | → settle |
  | `dense_tp.rs:459` `lower_selected_ordered` | None for this stage | → settle (earlier stages) |
  | `dense_tp.rs:468-470` admission Invalid or Rejected | None (candidate only) | → settle |
  | `dense_tp.rs:473` admission `Held` | Admission's un-undone reservation or arena | Cleanup cannot complete → **lost** |
  | `dense_tp.rs:500` boundary copy or bindings refused | The admitted plan (kept in `Held`) | → settle |
  | `dense_tp.rs:523` `execute_dense_stage` refused | Plan (unsubmitted) or Lost lease (`held`), kept | → settle; lease reclaimed after the drain |
  | `dense_tp.rs:530` `finish` refused | The lease, kept | → settle; reclaimed |
  | `dense_tp.rs:538-568` `copy_boundary` | Plan (the caller keeps it) | → settle |
  | `dense_tp.rs:602-614` `declare` | Plan in `Held` | → settle |
  | `dense_tp.rs:185` settle's drain | Everything held | Not observed → **lost**, withheld |
  | `dense_tp.rs:197-199` settle: retained leases | Refused stage leases | `OperationLease::reclaim_drained` after the drain → reusable |
  | `dense_tp.rs:201-214`, `629-648` settle: **nested owner**. This is a `PagedAttentionRun` that entered the step idle (the `is_idle` check before `begin`) and was quarantined **during this step on its rank stream**: `attend_into`'s launch (`paged_attention.rs:2743`), an append's row upload (`2508`), or its page-table publication (`2122`). | Only what that step's own work created: the stage's query and output ranges in `held_ranges`, and one `held` source (per-variant table below) | After the drain has observed that rank stream complete, `PagedAttentionRun::reclaim_drained` (`paged_attention.rs:1409`, crate-private) resolves `held` by variant: it **restores** the run's admitted page-table upload buffer, drops the step's own append rows, and refuses any other variant. It then clears the quarantine and hands the ranges to `release_reclaimed` in the owning plan's arena → reusable. A refused variant, or a range no held plan owns → **lost**, run kept quarantined. A run quarantined **before** the step never gets here (the `is_idle` entry row above). |
  | `dense_tp.rs:216` settle `close_plans` | The refused plan | **lost** |
  | `dense_tp.rs:218`, `742-761` `close_boundary` release or close refused | The boundary arena | **lost** |
  | `dense_tp.rs:223` abort (`let _`) | None (host) | After the drain; an abort refusal means no open transaction |
  | `dense_tp.rs:280` success-path settle | Plans, boundaries | Every settle error has already lost the group; the dead abort branch was removed (R3-3) |
  | `dense_tp.rs:289` failure-path settle | Everything held | `settle(abort)` |
  | `dense_tp.rs:116-123` commit prepare refused | Open transactions | `settle(abort)` → reusable |
  | `dense_tp.rs:126` commit apply failed | A rank may have committed | Irreversible → **lost** |
  | `dense_tp.rs:137` `DenseStep` dropped uncommitted | Open transactions | `settle(abort)` → reusable |
  | `dense.rs:142-173` pre-submission rejects (identity, bindings including the F6 overlap, image, module) | The plan, returned in `refused.plan` | → settle |
  | `dense.rs:188-219` `enqueue_dense` (including `attend_into`'s launch, which quarantines the run), event create or record refused | Lease marked Lost (single-GPU contract); a nested run may be quarantined | Kept → settle → lease and run reclaimed after the drain |
  | `dense.rs:265`, `305`, `316`, `323` finish: sync, readback, nonfinite, retire | Lease, kept | → settle → reclaimed |
  | `tensor_parallel.rs:335` group lost | None | Refused |
  | `tensor_parallel.rs:340-365` agreement, element type, checked extents and grid, source size | None | Pure |
  | `tensor_parallel.rs:376-383` reduce image or module load | None | Pure |
  | `tensor_parallel.rs:414-415` allocation refused | Ranges allocated so far, **on rank 0 and possibly rank 1**; nothing enqueued | `release_all` → reusable; a release failure → **lost** |
  | `tensor_parallel.rs:548` `enqueue_reduce` peer copy | Earlier enqueues | → join's drain, then `release_all` |
  | `tensor_parallel.rs:551-553` `enqueue_reduce` addresses | Earlier enqueues, including the peer copy | → join's drain, then `release_all` |
  | `tensor_parallel.rs:563` `enqueue_reduce` launch | The peer copy | → join's drain, then `release_all` |
  | `tensor_parallel.rs:438` join's drain | Copies and launches on both streams | Not observed → **lost**, withheld |
  | `tensor_parallel.rs:440-441` enqueue refused after the first copy (F2) | Drained | `release_all` → reusable; a release failure → **lost** |
  | `tensor_parallel.rs:449` success: scratch release | Drained | `release_all`; a failure → **lost** |
  | `tensor_parallel.rs:466-475` drain event create, record or query | — | Any failure on a rank → **lost** |
  | `tensor_parallel.rs:506` `release_all` refused | The range | **lost** |
  | `paged_attention.rs:4392-4417` `commit_paged_pair` prepare | Nothing changed on either rank | `PairCommitRefused::Prepare` → `settle(abort)` → reusable |
  | `paged_attention.rs:4423-4426` apply | Publication | `PairCommitRefused::Apply` → **lost** |

  **`RefusedSource` held by a run at reclaim** (`paged_attention.rs`, enum at 1197):

  | Variant | Who creates it, and where | Is it an admitted, reusable buffer of the run? | Handling after the observed drain |
  |---|---|---|---|
  | `PageTableUpload(Vec<u8>)` (new; was `Query`) | `publish_page_table` moves `page_table_upload` in at `2122`, both in the step's append and at commit | **Yes**: admitted with the run, reused by every publication | **Restored** to `page_table_upload`. The next publication re-uploads the whole table before any launch reads it. |
  | `Rows { keys, values }` | `write_rows` at `2508`; in the step, the per-step K/V copies `dense.rs` allocates | No: the step's own copies | Dropped |
  | `Query(Vec<u8>)` | `attend` (host staging) at `3293`, from the caller's query | No: the caller's | Never created by a TP step (it uses `attend_into` on `Staging::DeviceHandles` runs). If present: refused, run kept quarantined, group **lost** |
  | `Stream { .. }` | Host-backed streaming at `3541`, `3701` and the stream stage | No: the caller's | Never created by a TP step. If present: refused, run kept quarantined, group **lost** |
  | `PageTable(Vec<u32>)` | Only returned in a refusal (`give_back`), never placed in `held` | — | Not reachable in `held` |

- **Changed owners and consumers:**
  - `moxie-plan` owns stage construction (`build_stage_graph`, used by the
    CLI harness and the executor) and declared-order selection
    (`lower_selected_ordered`).
  - `moxie-state` owns the two-phase commit (`prepare_commit` and
    `apply_commit`); `commit` is unchanged for every existing caller.
  - `moxie-executor` owns the collectives, the step (`RankGroup::execute_dense`
    and `DenseStep::commit`) and `commit_paged_pair`.
  - No model, ledger or lease semantics changed.
- **Commands, GPU UUIDs; passed / failed / skipped:**
  - PASS: `cargo fmt --all -- --check`.
  - PASS: `cargo clippy --workspace --all-targets --locked -- -D warnings`.
  - PASS: `cargo clippy -p moxie-executor --all-targets --locked --features
    driver -- -D warnings` (this lane now passes, F10).
  - PASS: the same with `driver,paged-attention-binding`, and with
    `paged-attention-test-hooks`.
  - PASS: `cargo test --workspace --locked`, including the moxie-state
    prepare/apply test, the moxie-plan order test and the moxie-interp
    straddling-slice case.
  - PASS: `cargo test -p moxie-state -p moxie-interp --locked`.
  - PASS: `cargo xtask arch-check`.
  - PASS: `cargo xtask spec-check`.
  - PASS: `cargo test -p moxie-executor --locked --features
    driver,paged-attention-binding --test dense_tp2_device --test
    tensor_parallel_device --test dense_gemma_device`, on
    `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and
    `GPU-81fe4578-59b2-37c4-421e-287cdac78704`. 0059's test is still
    0.000 BF16 ULP on all three GPUs.
  - PASS: the extra `--features paged-attention-test-hooks --test
    paged_attention_device` lane (7 tests), because `commit` is now prepare
    plus apply.
  - No timing was taken.
- **What the one GPU test asserts:**
  - Host S=1 and S=2 differ.
  - C1: the one-3090 S=2 reference is within 1 BF16 ULP of the task 0058
    host interpreter at S=2.
  - TP2 prefill is bit-identical to the reference.
  - Each of these leaves both ranks' published KV unchanged and returns
    every ledger reservation (the outstanding count equals its pre-step
    value); the group stays reusable:
    - a mid-step rank-1 stage failure;
    - a mid-step collective refusal;
    - a collective whose second peer copy fails (F2, interposed
      `cuMemcpyPeerAsync`);
    - rank 1's first stage kernel launch refused after its uploads (N2,
      interposed `cuLaunchKernel`; the lease is Lost and is reclaimed after
      the drain);
    - rank 1's paged-attention kernel launch refused (R3-2, identified by
      the driver's `cuFuncGetName`), which quarantines the run holding the
      stage's query and output ranges, then reclaimed after the drain;
    - rank 1's **in-step** page-table publication failing (R5-1; the
      attention stage's key readback, an interposed `cuMemcpyDtoH_v2`,
      arms a one-shot `cuMemcpyHtoDAsync_v2` failure, and the append
      publishes the table before its rows). The run's upload buffer is
      restored, and the clean decode after the loop, on the same pair,
      succeeds and matches the reference exactly;
    - a run pre-quarantined by a failed page-table upload on a **side
      stream** (interposed `cuMemcpyHtoDAsync_v2`), on a fresh group and
      pair. The step refuses before `begin`: frontiers and reservations are
      unchanged, both ranks can still `begin`, and the run still refuses
      `close` as quarantined. So its quarantine and held upload bytes were
      untouched;
    - a row count whose aligned boundary size overflows (R3-1), refused
      before `begin`; the next step's successful `begin` proves nothing was
      left open;
    - a rank-1 prepare refusal after rank 0 prepared (C2);
    - a dropped uncommitted step (F11).
  - Every step reads `DenseStep::logits()` before commit and checks that
    commit returns the same bytes (N3).
  - The next clean decode is bit-identical.
  - An unobservable rank-0 drain (F5, interposed `cuEventQuery`) returns
    DeviceLost. The group records it, and the next step refuses with the
    same error.
  - On a fresh pair and group, a plan whose `cuMemFree` fails after the
    drain (N1, interposed `cuMemFree_v2`) loses the group. Its reservation
    stays outstanding (withheld), as the rule requires.
  - Probes confirmed each fault hit its intended path. The F5 detail names
    only rank 0, so rank 1 was observed independently.
- **Mutation results and restoration** (final tree; each applied, run and
  restored; clean rerun passes):

  | Mutation | Result |
  |---|---|
  | Round a partial before the reduce | Caught: prefill bits |
  | Reference ignores S | Caught by the C1 cross-check |
  | Reverse `k` inside a partial (F9) | **Caught now**: prefill bits |
  | Abort only one rank in `settle` | Caught: the next step's `begin` refuses |
  | Skip rank 1's prepare check, so its refusal moves to apply (C2) | Caught: frontiers `[6, 6]` vs `[5, 5]` |
  | Drop does not abort (F11) | Caught: `[6, 6]` vs `[5, 5]` |
  | Drop a lost lease instead of reclaiming it (N2) | Caught: reservations unbalanced |
  | A close refusal keeps the group (N1) | Caught: `DeviceLost` expected |
  | No `first % width` check in `lower_selected_ordered` (F3) | Caught by the moxie-plan order test |
  | No `first % width` check in `validate_linear_slice` (F3) | Caught by the moxie-interp split test |
  | Skip the quarantined-run reclaim in `settle` (R3-2) | Caught: reservations unbalanced after the attention fault |
  | Unchecked `next_multiple_of` in boundary sizing (R3-1) | Caught: the oversized step panics |
  | Accept a non-idle run at step entry (R3-2 narrowed) | Caught: the pre-quarantined run is not refused |
  | Drop the page-table upload buffer on reclaim (R5-1) | Caught: the next clean decode fails |
  | Reduce in descending order (swap the kernel operands) | Survives: an equivalent mutant at R=2 (ADR 0036 §3, `a + b == b + a`) |

- **Review map** (lines +/-):

  **Part 1: kernels and FFI**

  | File | +/- | Purpose | Clause |
  |---|---|---|---|
  | `crates/moxie-kernels/cuda/dense_ops.cu` | +65/-0 | The split reference kernel, the FP32 partial kernel and the exact two-rank reduce kernel | ADR 0036; C1 |
  | `crates/moxie-kernels/src/lib.rs` | +35/-2 | Symbols and catalogue descriptors | New pieces |
  | `crates/moxie-types/src/capability.rs` | +10/-0 | `LinearSplit` and `LinearPartial` ops | C1 |

  **Part 2: `RankGroup::reduce`**

  | File | +/- | Purpose | Clause |
  |---|---|---|---|
  | `crates/moxie-executor/src/tensor_parallel.rs` | +353/-31 | Crate-private `gather`/`reduce` over one `join` (0057's `agree`, checked extents, a shared `copy_columns`); the one `drain`; `lose`/`lost` | Exact reduce; F1, F2, F5, F8 |

  **Part 3: the `moxie-plan` stage builder**

  | File | +/- | Purpose | Clause |
  |---|---|---|---|
  | `crates/moxie-plan/src/tensor_parallel.rs` | +176/-1 | `build_stage_graph` | Production stage construction |
  | `crates/moxie-plan/src/selected.rs` | +217/-27 | `lower_selected_ordered` with order validation, including slice alignment, plus one test | C1; FP32 partials; F3 |
  | `crates/moxie-plan/src/lib.rs` | +3/-2 | Exports | — |
  | `crates/moxie-interp/src/lib.rs` | +2/-0 | `validate_linear_slice` alignment | F3 |
  | `crates/moxie-interp/tests/reference_graphs.rs` | +21/-2 | The straddling-slice case in the existing split test | F3 |
  | `crates/moxie-cli/tests/tensor_parallel.rs` | +37/-94 | The harness calls `build_stage_graph` | "Harness helper calls it" |

  **Part 4: executor orchestration and atomicity**

  | File | +/- | Purpose | Clause |
  |---|---|---|---|
  | `crates/moxie-executor/src/dense_tp.rs` | +769/-0 (new) | `RankGroup::execute_dense`, `settle` (the rule), `Held`, the lose-on-cleanup-failure helpers, and `DenseStep` (`logits` view, commit, abort on drop) | Rank-step atomicity; C2; F2, F5, F7, F11, N1, N3 |
  | `crates/moxie-executor/src/paged_attention.rs` | +117/-5 | `commit_paged_pair` (prepare both, then apply both); the crate-private `PagedAttentionRun::is_idle` and `reclaim_drained` (restore-or-refuse by variant); the `RefusedSource::PageTableUpload` variant | F4; R3-2 |
  | `crates/moxie-state/src/device.rs` | +130/-10 | `prepare_commit`/`apply_commit`/`PreparedCommit` (identity and generation), plus one focused test | F4 (approved) |
  | `crates/moxie-state/src/lib.rs` | +3/-1 | Export | — |
  | `crates/moxie-executor/src/dense.rs` | +81/-28 | `execute_dense_stage` | Additive stage entry |
  | `crates/moxie-executor/src/arena.rs` | +12/-0 | `OperationLease::reclaim_drained`, crate-private and used only by `settle` | N2 |
  | `crates/moxie-executor/src/chain.rs` | +38/-1 | `validate_bindings_except` with the overlap refusal; feature gates; `release_reclaimed` | F6, F10, R3-2 |
  | `crates/moxie-executor/src/lib.rs` | +4/-0 | Exports | — |

  **Part 5: tests and fixture**

  | File | +/- | Purpose | Clause |
  |---|---|---|---|
  | `crates/moxie-executor/tests/dense_tp2_device.rs` | +1171/-0 (new) | The one GPU test, the two-row order-sensitive fixture and four interposed driver symbols (plus `cuFuncGetName` to identify the attention kernel). The copied 0059 setup is accepted as temporary. | Bit-identity; failure; C1; C2; F2; F5; F9; F11; N1; N2; N3 |

  **Split.** Parts 1 to 3 (the C1 reference half) review independently of
  parts 2, 4 and 5 (the TP half, with the `moxie-state` two-phase commit).
  Only one GPU test exists, so their evidence is shared.
- **Remaining obligations:**
  - A declared-shape collective mismatch cannot arise within one driver;
    0057 owns it.
  - Replicated stages run one plan per node, and the boundary arena is
    sized as an upper bound (`ponytail:` notes).
  - The reduce image is loaded per call.
  - Thread-per-rank and the status collective remain slice 3.
