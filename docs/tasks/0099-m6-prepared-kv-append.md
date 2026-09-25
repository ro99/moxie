# Task 0099 — a prepared KV append: preview, then a non-allocating apply

Status: **accepted** (coordinator, 2026-09-25), sol's review R1 clean
(0/0/0). Implementation `19eea35`. Builder Codex `luna`; reviewer Codex
`sol`. Host-only (`moxie-state`). It is the foundation of task 0101
(full-step decode capture).

## Identity and authority

- Task0099, M6 roadmap **M6.2**. Split out of the first 0099 draft (now
  [task 0101](0101-m6-full-step-decode-capture.md)) after sol's design
  review, findings **H1** and **M1**:
  - a captured graph writes the KV rows on the device, so the state
    authority must hand the executor every row's placement **before** the
    graph is recorded or launched;
  - after the launch, the authority must record the rows as published by a
    transition that **cannot allocate or fail on a normal path**. Otherwise
    submitted device work could be left with half-advanced bookkeeping.
- Today `append_layer` does both at once: stage, compute the view and
  placements, call the writer, mark the layer, and publish after the last
  layer.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**; on a conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** Naming rule applies. No GPU.

## Facts established before writing (coordinator, 2026-09-25)

`moxie-state/src/device.rs`:
- `append_layer_for` (about 796–850): layers in order; staging on the first
  layer, through `stage_for` (about 862–909). That checks the open
  transaction, that nothing is pending, `rows > 0`, `max_tokens` capacity
  and the ADR 0014 undo headroom, then sets `open.pending`.
- `page_view_for_branch` and `retained_for` (about 484–488, 653–705) read
  `open.pending` to include the staged end.
- `placements_for` (about 582–628) refuses unless `open.pending` matches.
  It splits rows at page boundaries through `place` (a pure ring mapping).
- `publish_for` (about 924–952) calls `SequenceState::execute(branch,
  rows)` (`moxie-state/src/lib.rs` about 775–781). That calls
  `check_lineage_limit`, then `extend_lineage`. It then sets the rows,
  clears `pending` and resets `completed_layers`.
- The sequence has a `poisoned` flag per branch (about 144, 381–393).

## Bounded deliverable

- **Outcome:** `prepare_append` returns an opaque token holding, per layer,
  exactly the view and placements that `append_layer` would give its writer
  for this batch. `apply_append` publishes the batch from that token with
  no allocation. Its only failures are invariant violations, and those
  poison the sequence. `append_layer` behaves exactly as before, through
  the same shared calculation.
- **Allowed files:** `crates/moxie-state/src/device.rs`,
  `crates/moxie-state/src/lib.rs` (export, and a non-allocating
  lineage-extension split if change 3 needs it), this task's Result.
- **Non-goals:** any executor change (task 0101); branches other than
  `ROOT` (refuse); changing `append_layer`'s observable behaviour.

## Numbered changes

1. **One shared projection (M1).** Extract the pure calculation that
   `page_view_for_branch` and `placements_for` perform into one private
   function, `project(branch, layer, staged: &StagedRows) -> Result<(PageView,
   Vec<Placement>)>`. It takes an explicit, checked projected batch instead
   of reading `open.pending`, and uses the same retained base (committed
   watermark) and ring mapping.
   - `append_layer_for` calls it with the staged batch.
   - `page_view_for_branch` and `placements_for` keep their public
     behaviour.
2. **Prepare.** Add `pub fn prepare_append(&mut self, txn, rows) ->
   Result<PreparedAppend>`. It:
   - runs every check `stage_for` runs, without setting `pending`;
   - refuses if any layer of a previous batch is partly complete;
   - runs `SequenceState`'s lineage-limit check for `rows`, and reserves
     (with `try_reserve`) any capacity `extend_lineage` would need;
   - computes `project` for every layer.

   `PreparedAppend` is opaque (fields private, not `Clone`). It holds:
   - `txn` and `first` (the frontier);
   - `rows`;
   - per layer, the `PageView`, the placements and the retained range
     used;
   - the sequence's identity.

   Accessors expose the per-layer views and placements read-only.

   `&mut self` exists only to reserve capacity. Prepare changes no
   observable state: a dropped token leaves the sequence exactly as it
   was.
3. **Apply.** Add `pub fn apply_append(&mut self, token: PreparedAppend) ->
   Result<()>`. It:
   - rechecks, without allocating: the same sequence, the same open `txn`,
     no `pending`, `rows == token.first`, and every layer's retained range
     equal to the token's;
   - then advances exactly as `publish_for` does: executes the lineage
     using the reserved capacity, sets `rows`, and leaves
     `completed_layers` all false.

   **Any** failure in `apply_append` sets the branch's `poisoned` flag and
   returns the error. The caller has already submitted device work, so a
   refused apply cannot be retried as if nothing happened (H1).

   If `extend_lineage` cannot run without allocating even after the
   reservation in change 2, stop and send `DECISION` naming the allocation.
4. **Tests (in `device.rs`'s test module).**
   - (a) Recording-writer equality: `prepare_append`'s per-layer views and
     placements equal what a recording writer receives from `append_layer`
     for the same batch. Cover the first row, a batch crossing a page
     boundary, a sliding-window layer whose base moves, and a batch after a
     commit that advanced the watermark.
   - (b) Equivalence: two sequences built identically; one appends via
     `append_layer`, the other via `prepare_append` + `apply_append`.
     Afterwards `rows`, every layer's `layer_retained` and `page_view`,
     and the `SequenceState` frontiers are equal. Repeat for 20 steps
     across page boundaries.
   - (c) A stale token (another append happened after prepare) makes
     `apply_append` refuse and poison the sequence.
   - (d) A dropped token leaves the sequence unchanged, and a following
     `append_layer` works.
5. **Coverage check (one mutant, reverted after; run only test (b)).** In
   `apply_append`, skip the lineage execution. Test (b) must fail.

## Acceptance

Host gates:
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- `cargo test --workspace --locked`;
- `cargo xtask arch-check`;
- `cargo xtask spec-check`.

No GPU gates; `moxie-state` has no device code.

## Result, filled after work

- Added opaque `PreparedAppend` tokens with read-only per-layer page-view and
  placement accessors. `prepare_append` performs staging checks without setting
  pending state, rejects a partly completed batch, and captures each layer's
  retained range, view and placements. Dropping a token leaves the sequence
  usable.
- Extracted the shared projected-batch calculation into `project`; ordinary
  append and per-layer append use it. `apply_append` checks the token against
  the open transaction and current ranges, publishes the lineage through
  pre-reserved capacity, and poisons the sequence on any refusal.
- Added tests for recording-writer equality at the first row, a page crossing,
  and after a sliding-window base moved; 20-step equivalence; stale-token
  poisoning; partial-batch refusal; and dropped-token state preservation.
- Mutation check: skipping lineage execution made the 20-step equivalence test
  fail on the first frontier comparison (expected executed 1, observed 0);
  reverted the mutant.
- Host gates passed: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` (79 rejected
  fixtures, 21 accepted, 13 rules exercised), and `cargo xtask spec-check`.
