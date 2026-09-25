# Task 0101 — full-step decode capture: step-state buffer, prepare/apply, one graph

Status: **proposed** (coordinator, 2026-09-25). This was drafted as 0099.
Sol's design review found 5 high and 4 medium, recorded as binding below.
The state authority's prepared-append token was split out as task 0099. This
contract is revised against that token and re-reviewed before
implementation.
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
  capture enabled, and eligible (no host joins, orders, ownership or
  host-backed staging), runs its whole step as **one graph launch** after a
  first captured step. Outputs are bit-identical to eager steps, over many
  decode tokens. A new run set recaptures. Everything else falls back to
  task 0085's segments.
- **Allowed files:**
  - `crates/moxie-state/src/device.rs` (the preview in change 1);
  - `crates/moxie-executor/src/{dense.rs,chain.rs,paged_attention.rs}`;
  - `crates/moxie-plan/src/selected.rs` (workspace placement of the step
    buffer only);
  - `crates/moxie-kernels/src/lib.rs` (constants only, if needed);
  - `crates/moxie-executor/tests/dense_gemma_device.rs` (new tests);
  - this task's Result.
- **Non-goals:** prefill buckets (they stay piecewise); TP and ordered
  plans (task 0100); host-backed streaming; any kernel source change;
  timing.

## Numbered changes

1. **Preview (`moxie-state`, H2).** Add `pub fn preview_append(&self, txn,
   rows) -> Result<Vec<(PageView, Vec<Placement>)>>`:
   - it returns, per layer in order, exactly the view and placements that
     `append_layer` would pass its writer for this batch, and changes no
     state;
   - it applies every check that `stage_for` and `placements_for` apply
     (open transaction, no pending batch, capacity, undo headroom);
   - one test asserts that it equals what a recording writer receives from
     `append_layer`, for layouts with and without a sliding window, and
     across a page boundary.
2. **Step-state buffer (`selected.rs` placement, H3, L1).**
   - A decode candidate with attention gets one 256-byte-aligned range in
     its workspace region. It holds `layers × (4 + 2·max_rows)` `u64`: the
     four scalars, then the offsets, per layer. Its end is included in the
     region, the arena ranges and the ledger request.
   - The plan owns a pinned host mirror of the same size (`PinnedHostBuffer`,
     charged `Host(Pinned)`), created lazily with the `Blas` handle's
     lifetime rules.
   - The mirror is rewritten only at the start of a step. The previous
     step's dense lease completion has already been observed by then (the
     plan is returned only after it), so no separate event is needed (L1).
3. **Upload-only table path (`paged_attention.rs`, H2, M1).**
   - Split the table upload out of `publish_page_table_deferred` into
     `upload_table_for(view, stream)`. It checks `view.len() ≤
     geometry.pages`, the run's admitted table extent (M1), and enqueues
     the H2D copy into `self.table`, with the run's existing retained-source
     and deferred-completion rules.
   - The captured path never calls `publish_page_table_deferred` or
     `write_rows_from_device` (H2).
4. **Indirect launches (`dense.rs`).** When the step mode is full-step
   (change 6), an `Attention` node enqueues, in this order:
   - `KV_APPEND_INDIRECT`: sources are the K and V activation ranges;
     destinations are the run's key and value page ranges; pointers are
     `step + layer_base` and `offsets`; grid `(max_rows)`, 128 threads;
   - `PAGED_ATTENTION_INDIRECT`, with the run's query, pages, table and
     output, the step pointer, and the plan-constant scalars.

   Resolve both symbols in the run's module at run admission (same fatbin
   and SHA). The run's selected descriptor is unchanged; task 0098's
   `test-gpu` case is the evidence that they are equivalent.
5. **Prepare (host, before any enqueue, H3, M1).** For each step:
   - call `preview_append`;
   - per layer, expand each placement's rows into per-row key and value
     byte offsets `physical_page·page_bytes + slot·row_bytes`, with checked
     arithmetic. Check each offset plus `row_bytes` against the run's page
     range, and the view length against `geometry.pages`;
   - write the four scalars (from the same `PagedAttentionLaunch` checks
     `execute_attention` applies today) and the offsets into the mirror;
   - write each run's table bytes.

   Any check failing refuses the step before any enqueue, with the plan
   returned unchanged.
6. **Full-step mode and replay order (H4, M2, M5).**
   - `CaptureMode` gains full-step for eligible decode plans with capture
     enabled.
   - Per step, on the plan stream, in this order:
     1. the existing token and position binding uploads and RoPE-table
        uploads (task 0083);
     2. the mirror → step buffer H2D;
     3. each run's `upload_table_for`;
     4. then, on the first step or after a recapture, begin capture, run
        the whole `enqueue_dense` with indirect attention, end, instantiate
        and launch. Otherwise, launch the instantiated graph.
   - **Identity (M5, H1):** the captured graph stores the ordered run ids
     and every launch pointer and fixed scalar:
     - the key, value, table and step addresses;
     - geometry, heads, kv_heads, head_dim, page_tokens, window and scale;
     - symbols, the rows bucket, and the BLAS workspace address.

     Before any launch, the step's values are compared with the stored
     identity. On a mismatch, the graph is destroyed (the plan is idle, so
     no work is outstanding) and the step recaptures.
   - **Pools (M2):** before capture, compute the full-step bound: 8 KiB per
     kernel node (every node, including each append and attention, plus
     task 0096's cuBLAS node bound per linear) and 128 KiB for the one
     graph. Reserve it in `GraphPools`. A refused reservation falls back to
     segments for this plan, typed and reported, never silently.
7. **Apply (after submission, H2, H3).** Only after the graph launch and
   the completion-event record both succeed:
   - call `append_layer` for each layer, with a `VerifyOnlyWriter` that
     checks the authority's view and placements equal the previewed ones
     and enqueues nothing;
   - then record the dense completion on each participating run as its
     pending work (the task 0082 mechanism), so that `close` or `drop` of a
     run observes it.

   If the launch or the event record fails:
   - `append_layer` is not called, and the frontier is unchanged. Rows the
     graph may have written lie beyond the frontier, unpublished;
   - the plan is withheld and quarantined under task 0096's rules, and the
     runs are quarantined.
8. **Escape inventory.**

   | Resource | Owner | Rule |
   |---|---|---|
   | Step buffer | Range in the plan's arena | Lives and dies with the plan's workspace |
   | Pinned mirror | Plan | Freed in plan teardown, after observed completion (checked free, task 0092); leaked on quarantine |
   | Captured graph | Plan | Destroyed before `Blas`, module and arena; on identity mismatch only while the plan is idle |
   | Runs named by a graph | Caller | Never launched against without a matching identity; close/drop observe the pending completion from change 7 |

   Pre-launch refusal returns the whole plan unchanged. A post-submission or
   event failure withholds the plan. A close refusal keeps every surviving
   resource.
9. **Tests (`dense_gemma_device.rs`).**
   - (a) Shape A: prompt, then 8 decode steps, run three ways (eager,
     piecewise, full-step). The logits are **byte-identical** at every
     step. Do this with the ordered catalogue and with the unordered one.
   - (b) A second prompt with fresh runs recaptures (a test-hook counter)
     and stays byte-identical to eager.
   - (c) Sliding-window layers across a page boundary during decode, with
     a window not aligned to pages (M1), stay byte-identical.
   - (d) An injected launch failure (test hook): the frontier is unchanged,
     the plan is withheld, the runs are quarantined, and the ledger is
     recovered after the quarantine drain.
   - (e) A run closed after `execute_dense` returns but before `finish`:
     the close observes the pending completion, with no use-after-free.
   - (f) `preview_append`'s equality test from change 1.
10. **Coverage check (one mutant, reverted after; run only test (a)'s
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
- the full `paged_attention_device`, since the run changes.

No `test-gpu` (no kernel source changes), and no timing. The reviewer may
start at the candidate commit while the final gates run.

**Stop conditions:**
- `preview_append` cannot equal what `append_layer` passes without changing
  `append_layer`;
- a check in change 5 needs run state the prepare step cannot read without
  a borrow conflict;
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

## Result, filled after work
