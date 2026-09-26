# Task 0101 — full-step decode capture: step-state buffer, prepare/apply, one graph

Status: **active, revision 3** (coordinator, 2026-09-25). The second design
review (3 high, 1 medium) is adopted verbatim in "Design review 2" below.
Where it conflicts with the numbered changes, **it overrides them**.
Earlier revision history: This was
drafted as 0099. Sol's design review found 5 high and 4 medium, and all nine
are now folded into the numbered changes. The state authority's token was
split out as task 0099 (`prepare_append`/`apply_append`, `19eea35`). It is
awaiting a second design review.
It builds on task 0098's kernels. Builder Codex `luna`; reviewer Codex
`sol`. Asynchronous-ownership work: change 8 is the escape inventory.

## Identity and authority

- Task0101, M6 roadmap **M6.2**, "stable decode … graph capture with
  piecewise fallback" (owner ruling 2026-09-25: build every roadmap
  feature). Task 0098's step-indirect kernels exist and are bit-identical to
  the direct ones. This task captures a whole single-GPU decode step,
  attention included, as one graph. Task 0085's piecewise capture remains
  the fallback.
- Every point of sol's design review of the 0099 outline (recorded in task
  0098: H1–H4, M1, M2, M5, L1) is part of this contract, cited where used.
- **H1 decision (coordinator):** a graph names specific runs' addresses. A
  new set of runs (a new prompt) **recaptures**. Once recaptured, the graph
  replays for every decode token of that turn. Extending the kernel ABI with
  indirect run pointers is not done here.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. On a conflict with the code,
  stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** GPUs free;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Naming rule applies.

## Facts established before writing (coordinator, 2026-09-25)

- **Task 0099** (`19eea35`) provides `DeviceKvSequence::prepare_append(&mut
  self, txn, rows) -> Result<PreparedAppend>`. It changes no observable
  state, and its accessors return per layer `page_view(layer)` and
  `placements(layer)`. `apply_append(&mut self, PreparedAppend) ->
  Result<()>` is non-allocating, and poisons the sequence on any failure.
- `dense_graph.cu` `#include`s `paged_attention.cu`. The plan-owned dense
  module therefore contains `PAGED_ATTENTION_INDIRECT` and
  `KV_APPEND_INDIRECT` (sol H3).
- `DeviceKvSequence::append_layer(txn, layer, rows, writer)`
  (`moxie-state/src/device.rs` about 788–850):
  - it stages the batch on layer 0;
  - it computes `page_view_for_branch` and `placements_for`;
  - it calls `writer.write_layer(layer, batch, view, &placements)`, and on
    `Ok` it marks the layer complete;
  - after the last layer, `publish_for` advances the frontier.

  `place` (about 650) is a pure ring mapping: `physical_page = (position /
  page_tokens) % layout.pages`. No page is allocated during a step.
- The executor's writer adapter (`append_paged_layer_from_device`,
  `paged_attention.rs` about 5400) publishes the page table
  (`publish_page_table_deferred`, about 2305). It then enqueues one D2D copy
  per placement (`write_rows_from_device`, about 2878), inside the
  authority's callback.
- Attention: `execute_attention` (`dense.rs` about 1461–1600) builds a
  `PagedAttentionLaunch` and calls `run.attend_into_deferred`. That records
  a pending completion on the run (task 0082), so closing a run observes
  outstanding work.
- Piecewise capture: `close_segment` and the `Eager`/`Capture`/`Replay`
  modes (`dense.rs` about 1317–1360); `GraphPools` admission constants
  (task 0086): 8 KiB per kernel node, 128 KiB per graph.
- Task 0098's symbols: `PAGED_ATTENTION_INDIRECT` (`step[0..4] = rows,
  first_position, history_base, history_rows`) and `KV_APPEND_INDIRECT`
  (`step[0] = rows`, `offsets[2r], offsets[2r+1]` = key and value
  destination byte offsets). Both are in the paged-attention fatbin.

## Bounded deliverable

- **Outcome:** a single-GPU dense **decode** plan (rows bucket 1) with
  full-step capture enabled, and eligible (no host joins, orders, ownership
  or host-backed staging), runs its whole step as **one graph launch** after
  a first captured step. Outputs are bit-identical to eager steps, over many
  decode tokens. A new run set recaptures. Everything else falls back to
  task 0085's segments.
- **Allowed files:**
  - `crates/moxie-executor/src/{dense.rs,chain.rs,paged_attention.rs}`;
  - `crates/moxie-plan/src/selected.rs` (workspace placement of the step
    buffer only);
  - `crates/moxie-kernels/src/lib.rs` (constants only);
  - `crates/moxie-executor/tests/dense_gemma_device.rs` (new tests);
  - this task's Result.
- **Non-goals:**
  - prefill buckets (they stay piecewise);
  - TP and ordered plans (task 0100);
  - host-backed streaming;
  - any kernel source change;
  - in-place recovery of a quarantined direct run (H5);
  - timing.

## Numbered changes

1. **Enabling full-step capture (M2, M4).** Add `pub fn
   set_full_step_capture(&mut self, enabled, ledger)` on the dense plan (or
   the `DensePlanSet` rows entry, mirroring `set_segment_capture`). With the
   `Ledger` available, enabling it:
   - (a) checks eligibility, and refuses with a typed error otherwise;
   - (b) admits the full-step pool bound. That is 8 KiB times the full
     graph's kernel-node count: every node, each indirect append and
     attention, and task 0096's cuBLAS node bound per linear. Add 128 KiB
     for the one graph. Any piecewise reservation is replaced, never
     stacked: destroy the old graphs, then release their reservation, and
     keep it if the release refuses;
   - (c) charges `Host(Pinned)`, then allocates the pinned step mirror.

   A refusal at (b) or (c) leaves the plan unchanged and still eligible for
   eager or piecewise execution. Disabling it destroys the graph (the plan
   is idle), then frees the mirror with the checked free, then releases the
   pool and pinned charges.
2. **Step buffer (`selected.rs`).** For a decode candidate with attention,
   add one 256-byte-aligned range after the existing workspace ranges (after
   0096's BLAS range and 0097's split slots). It holds `layers × (4 +
   2·max_rows)` `u64`: per layer, the four scalars, then the key and value
   offsets per row. Use checked offsets, and include its end in the region,
   the arena ranges and the ledger request. It is charged whether or not
   full-step is enabled, because it is small (for decode, `6 × layers × 8`
   bytes). The mirror is the only full-step-only resource.
3. **Symbols from the plan module (H3).** Full-step launches
   `PAGED_ATTENTION_INDIRECT` and `KV_APPEND_INDIRECT` from the plan-owned
   dense module, resolved through 0096's node-to-module map as extra entries
   of each attention node. They are never taken from a run's module, so a
   run closing after its pending work cannot invalidate the plan's graph.
4. **Prepare (host, before any enqueue; H2, M1, M3).** Per step:
   - `token = state.prepare_append(txn, rows)`;
   - per layer, reuse the existing predicates against the token's view and
     placements:
     - table publication: table base, physical identities, uniqueness and
       agreement with already-written pages;
     - `check_write_placements`: order, slot, table identity, geometry and
       max rows;
     - source extents and device, query/output aliasing, and the attention
       descriptor and grid.

     Extract them into callable checks where needed; do not duplicate them.
   - The attention scalars come from the token's **projected** view:
     `history_base` from its base, and `history_rows` from its projected end.
     Run `PagedAttentionLaunch`'s checks against that projected range (H2),
     and check that the view length is ≤ the run's `geometry.pages` (M1).
   - Expand each placement into per-row key and value byte offsets
     (`physical_page·page_bytes + slot·row_bytes`, checked), each within the
     run's page range.
   - Write the scalars and offsets into the mirror, and each run's table
     bytes into its host table buffer.

   Any failure here refuses the step with the plan and bindings returned
   unchanged. The token is dropped, and nothing was enqueued.
5. **Upload-only table path (`paged_attention.rs`, M3).**
   `upload_table_for(view, stream)` validates (change 4's predicates), then
   enqueues the H2D copy into `self.table`. It updates the run's host
   `page_table` and base **only after** the copy is enqueued and tracked.
   The captured path never calls `publish_page_table_deferred` or
   `write_rows_from_device`.
6. **Replay order (H4).** Per step, on the plan stream:
   1. token and position binding uploads and RoPE-table uploads (as today);
   2. mirror → step buffer;
   3. each run's `upload_table_for`;
   4. on the first step or after a recapture: begin capture, enqueue the
      whole step with indirect append then indirect attention per layer,
      end, instantiate, launch. Otherwise, check the identity (change 7)
      and launch the instantiated graph;
   5. record the dense completion event.
7. **Identity (M5, H1).** The captured graph stores the ordered run ids and
   every launch pointer and fixed scalar:
   - the key, value, table and step addresses;
   - geometry, heads, kv_heads, head_dim, page_tokens, window and scale;
   - symbols, the rows bucket, and the BLAS workspace address.

   It is compared **before** step 1 of change 6, so a mismatch is found
   before any enqueue. On a mismatch, destroy the graph (the plan is idle)
   and recapture within the reserved pool bound.
8. **Apply (after tracked submission; H1, H2).**
   - Only after the graph launch and the event record both succeed:
     1. `state.apply_append(token)`;
     2. per participating run, atomically with it, advance `written` to the
        projected end, and record the dense completion as the run's pending
        work (task 0082).
   - If `apply_append` fails, the sequence is already poisoned by 0099. The
     plan and every run are quarantined, and never returned as reusable.
9. **Escape inventory (H4, H5).**

   | Path | Rule |
   |---|---|
   | Failure before the first enqueue of a step | Returns the plan and bindings unchanged. |
   | Failure after any upload of steps 1–3 may have been submitted, or a failed launch or event record | Plan, mirror, table sources and runs are retained. Plan withheld, runs quarantined. The token is dropped without applying, so the frontier is unchanged. No in-place recovery (H5): the direct runs stay quarantined and charged. |
   | Pinned mirror on close | Checked free; a refused free returns the mirror in the close refusal. The `Host(Pinned)` charge is released only after a successful free. |
   | Pinned mirror on unclosed `Drop` | Forgotten (leaked), like the module and graphs. |
   | Graph | Destroyed before `Blas`, the module and the arena. Destroyed on an identity mismatch only while the plan is idle. |
   | Runs | May close after their pending work. The plan module, not the run's, supplies the graph's functions (change 3). |
10. **Tests (`dense_gemma_device.rs`).**
    - (a) Shape A: prompt, then 8 decode steps, run three ways (eager,
      piecewise, full-step). The logits are **byte-identical** at every
      step, with the ordered catalogue and with the unordered one.
    - (b) A second prompt with fresh runs recaptures (a test-hook counter)
      and stays byte-identical.
    - (c) Sliding-window layers across a page boundary, with a window not
      aligned to pages, stay byte-identical.
    - (d) An injected launch failure: the frontier is unchanged, the plan is
      withheld, the runs are quarantined, and their charge is **still
      held** (H5).
    - (e) A run closed after `execute_dense` returns but before `finish`:
      the close observes the pending completion. The plan's next step
      recaptures for a new run; the graph never calls the closed run's
      module.
    - (f) Enabling full-step with a `Host(Pinned)` cap too small for the
      mirror is refused, and the plan still runs piecewise (M4).
11. **Coverage check (one mutant, reverted after; run only test (a)'s
    full-step case with `--exact`).** Skip the mirror upload on replay
    steps. Test (a) must fail.

## Acceptance

Host gates:
- fmt;
- workspace clippy;
- executor driver clippy, with and without `cublas`;
- `cargo test --workspace --locked`;
- arch-check;
- spec-check.

GPU gates, once, with `cublas`:
- the full `dense_gemma_device`;
- the full `paged_attention_device`.

No `test-gpu`, and no timing. The reviewer may start at the candidate
commit while the final gates run.

**Stop conditions:**
- an existing predicate cannot be called against a projected view without
  changing its semantics;
- capture of any node fails;
- a file outside the allowed list is needed.

## Design review (sol, 2026-09-25) — binding amendments, to be folded into the changes before implementation

- **H1 → task 0099.** The apply step is a prepared, non-allocating token,
  not a `VerifyOnlyWriter` over `append_layer`. If a post-submission
  invariant still fails, the state is poisoned and the plan and runs are
  quarantined, never returned as reusable.
- **H2.** Prepare derives `history_base` and `history_rows` from the
  previewed (projected) view and runs the launch checks against that
  projected range. After tracked submission, it updates each run's
  `written` high-water mark and pending event atomically with the token's
  apply.
- **H3.** Captured indirect symbols come from the **plan-owned dense
  module** (`dense_graph.cu` includes `paged_attention.cu`), not from a
  run's module. A run may close after its pending work without invalidating
  the plan. Run-address identity and recapture for fresh runs stay.
- **H4.** Only failures before the first enqueue return the plan and
  bindings. After any upload that may have been submitted, retain or
  quarantine the plan, the mirror, the table sources and the affected runs
  until a recorded completion or a drain. An unclosed plan's `Drop` forgets
  its pinned mirror. A checked free on close returns the mirror to a
  refused plan, and the `Host(Pinned)` charge is released only after a
  successful free.
- **H5 (coordinator decision).** No in-place recovery. After a full-step
  launch or event failure, the direct run stays quarantined and charged,
  as every other failure path does today. Test (d) expects a withheld plan,
  quarantined runs, and the charge still held. In-place reclaim of a direct
  run is not built here.
- **M1 → task 0099** (shared projected view and placement calculation).
- **M2.** When capture is enabled, with the `Ledger` available, admit the
  maximum pool bound for the selected mode before any graph is made. A
  transition or fallback must not double-charge. Old graphs are destroyed
  before their reservation is released, and the reservation is retained if
  the release refuses.
- **M3.** Prepare reuses the existing table-publication and
  `check_write_placements` predicates, plus the source-extent, device,
  aliasing and descriptor/grid checks, against the projected frontier.
  `upload_table_for` updates the run's host table and base only after
  validation and a tracked copy.
- **M4.** The pinned mirror is admitted and allocated only when full-step
  capture is enabled. A pinned-cap refusal leaves the plan eligible for
  eager or piecewise execution. Charge before allocating.

## Design review 2 (sol, 2026-09-25) — adopted verbatim; overrides the numbered changes

- **H1 (change 1, transitions).** From piecewise mode:
  1. First admit the pinned charge and allocate the mirror while the old
     piecewise graph and reservation remain. If this fails, keep piecewise
     unchanged.
  2. Then destroy the old graphs and release the old pool reservation. If
     the release refuses, retain that reservation and return a transition
     refusal with the plan held.
  3. Admit the new pool bound before capture, with no simultaneous pool
     reservations.

  A later admit or cleanup refusal may leave the plan in eager mode with
  retained charges, and must not claim "unchanged". Reject
  `set_segment_capture(true)` while full-step is active. A failed
  graph-pool or pinned-charge cleanup cannot return a clean, unchanged plan.
  Test (f) is read under this rule.
- **H2 (changes 4, 5, 9, table sources).**
  - Before writing any run's table bytes in prepare, observe its prior
    pending completion and recover `page_table_upload`. On an observation
    failure, quarantine and withhold the run and the plan.
  - Keep the encoded source owned by the run from before the H2D call,
    including a refused call, until a completion event or a drain. Only
    then may the next prepare rewrite it.
  - The escape inventory's pre-enqueue row splits in two: a prior-completion
    failure quarantines; any other pre-enqueue failure returns the plan
    unchanged.
- **H3 (change 8, apply atomicity).**
  - First, observe and retire the prior run events.
  - After the graph launch, and **before** `state.apply_append`, create and
    record one completion event per participating run, plus the lease's
    completion event. Handle every creation or record failure as a
    post-submission quarantine.
  - Then apply the token, and transfer the already-recorded events to the
    runs with infallible assignments of `pending` and `written`.
  - If apply refuses, keep every event with the quarantined runs and the
    withheld plan.

  Each run owns its own `Event` (events are not `Clone`); the lease owns the
  dense one.
- **M1 (capture errors).** On every error after `begin_capture` and before a
  graph is installed:
  - attempt `end_capture` exactly once and destroy any returned graph;
  - clear the open-capture flag;
  - withhold the plan and runs, and keep the upload sources, if completion
    is unknown.

  An `end_capture` or `instantiate` refusal follows the same
  post-enqueue rule. This also covers the stop condition where a node
  cannot be captured.

## Result, filled after work

- Implemented the plan-owned step buffer and pinned mirror, eligibility and
  pool transitions, projected append/table preflight, indirect append plus
  attention capture/replay, run-specific completion events, recapture identity,
  and quarantine/retention paths. Added tests (a)–(f) under the contract's
  feature gates.
- The replay-mirror mutant was caught: omitting the mirror upload made test
  (a) fail on the second decode step. The mutant was reverted.
- Targeted GPU checks passed on all three devices for tests (a), (b)/(e)/(f).
- Host gates: fmt, workspace clippy, executor driver clippy with and without
  cublas, arch-check, and spec-check pass. The final workspace test run is
  running. Full `dense_gemma_device` and `paged_attention_device` suites remain
  pending.
