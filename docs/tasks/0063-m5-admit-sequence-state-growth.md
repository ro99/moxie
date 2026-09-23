# Task 0063 — admit SequenceState transaction and lineage growth on the device step

Status: **proposed**.

## Identity and authority

- Task0063, M5 plan slice 3, split from task 0061 by amendment on
  2026-09-23. Builder Codex `luna` (max, `/ponytail:ponytail`); reviewer
  Codex `sol` (read-only, `/ponytail:ponytail-review`); coordinator Claude
  Opus `coordinator`. Accepted by the coordinator under the owner's auto-mode
  delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `6f5a261`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035).
- Requirement: AGENTS.md, "One real memory authority admits all
  persistent/transient/branch/draft resources." Task 0061's round-3 review
  found `SequenceState` growth on every device step outside every request.
- Authority for the semantic change: task 0061 listed a `SequenceState`
  semantic change as its stop condition. Under coordinator.md §3 the
  coordinator authorizes one here, because it follows an existing precedent
  and serves an unchanged requirement: a reserved, charged lineage capacity
  with a typed refusal beyond it. No owner gate is involved.
- O6/O7 are open: no timing.

## Facts established before writing (coordinator, 2026-09-23)

**Growth sites (`moxie-state/src/lib.rs`), from task 0061's inventory:**
- `SequenceState::begin` inserts an open-transaction `BTreeMap` node
  (around line 966).
- `execute` and `accept` call `Branch::extend_lineage` (around lines
  491–497, 700–715), which pushes one `PrefixLineage` per newly occupied
  position and reallocates as the lineage grows.
- `commit_prefix` also extends the lineage (around lines 978–995).
- In task 0038's 32K gate, the root lineage grows to 32K entries. A forked
  child starts from a charged clone of `at + 1` entries, then grows past it.

**Precedent (`moxie-state/src/paged.rs`):**
- `PagedSequence` construction (around lines 538–552) pre-reserves the root
  lineage to `max_tokens + 1` with `try_reserve_exact`, confirms the exact
  capacity, and refuses with `CapacityExceeded` otherwise. The reservation is
  charged.
- Its control bytes charge the branch and transaction maps with
  `btree_node_bound` (around lines 270–282).
- `extend_lineage` never reallocates within that reservation.

**`execute` is deliberately unbounded above the accepted prefix**
(speculation executes before acceptance; lib.rs doc comment around
line 695). A cap on the *total* occupied positions does not conflict with
that; a cap *below* the accepted prefix would.

## Bounded deliverable

- **Outcome:** on the `DeviceKvSequence` step path, `SequenceState`
  lineage and transaction-map growth are admitted in advance.
  - Every lineage (root, and each forked child) is pre-reserved to its
    admitted maximum position count and charged.
  - `extend_lineage` never reallocates within the reservation.
  - A step that would exceed the reservation is refused with a typed error
    before any state changes.
  - Transaction-map nodes are charged under the existing `btree_node_bound`
    precedent.
- **Fork:** a child's lineage is reserved to the child's admitted maximum,
  not to `at + 1`. This replaces task 0061's `at + 1` clone charge.
- **Non-goals:**
  - no change to acceptance or speculation semantics below the cap;
  - no change to `PagedSequence`;
  - no thread-per-rank work (task 0062);
  - no timing.

## Phase 1 — design proposal before code

Send a `DECISION` report of 40 lines or fewer covering:

1. **The admitted maximum position count for a `DeviceKvSequence`.** Where
   does it come from today? For windowed or ring state, positions exceed the
   physical pages, so is there an admitted context maximum? If none exists,
   propose the smallest place to declare one, and say whether any caller has
   to supply it.
2. **Where the reservation and charge live.** At construction, and at fork.
   Say how they reach the ledger request that the paged run and task 0060's
   step already use.
3. **The refusal:** the exact check, where it sits relative to state
   mutation, and its error type.
4. **The transaction-map charge.**
5. **Files, and which existing tests change** (for example the 32K gate's
   request assertion).

The coordinator answers before implementation starts. If a fact above is
wrong, say so.

## Acceptance

- Host lanes pass: `fmt`, workspace `clippy`, driver-only `clippy`,
  workspace tests including `moxie-state`, `arch-check` and `spec-check`.
- GPU lanes pass: `paged_attention_device` (with test-hooks),
  `dense_gemma_device`, `dense_tp2_device` and `tensor_parallel_device`,
  plus the full `cargo xtask-cuda test-gpu` at 63/63.
- **Proof, extending existing assertions:**
  - The 32K gate's charged host bytes cover the root lineage and the child
    lineage at their full reserved capacity.
  - The lineage `capacity()` stays unchanged across the whole 32K run, which
    means it never reallocated.
- **Mutations,** each run and restored:
  - Drop the lineage reservation.
  - Reserve the child only to `at + 1`.
  - Remove the refusal.

  Each must fail a test.
- One test per invariant, plus a **Review map** and an updated allocation
  inventory in the Result.
- **Stop condition:** the admitted maximum cannot be defined without an owner
  decision (for example, an unbounded-context product requirement). In that
  case report the evidence and the smallest decision needed.

## Result, filled after work

- Design decision (phase 1) and coordinator answer:
- Changed owners; source commit:
- Commands, GPU UUIDs; passed / failed / skipped:
- Mutation results and restoration:
- Review map and allocation inventory:
- Remaining obligations:
