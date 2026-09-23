# Task 0063 — admit SequenceState transaction and lineage growth on the device step

Status: **accepted** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Built by Codex `luna`; reviewed by Codex `sol`, round 1 ACCEPT.
The builder ran the full test-gpu at 63/63; the coordinator re-ran `fmt`,
driver-only `clippy`, `arch-check` and `spec-check`.

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

- **Design decision and coordinator answer:** Approved. `DeviceKvSequence` takes
  its admitted maximum position count from `KvGeometry.max_tokens`, already
  enforced in `moxie-state/src/device.rs:883`. Strata makes the same split:
  `src/models/deepseek/deepseek_admission.cpp:113` validates
  `maximum_context_tokens`, while
  `src/models/deepseek/detail/runtime_public.inc.cpp:503` sets the separate
  `sliding_window_rows` retention bound. Every device root and
  child reserves `L = checked(max_tokens + 1)` lineage entries. The optional
  limit is per branch and is set only by `DeviceKvSequence`; `None` keeps
  existing `SequenceState` consumers on their historical unbounded path. A
  preflight in `append_prompt`, `execute`, `accept` and `commit_prefix` rejects
  over-capacity growth as `Error::CapacityExceeded` at
  `Host(Pageable)`, before changing a frontier, lineage or transaction. One
  transaction-map node is charged for each admitted open branch.
- **Changed owners; source commit:** `moxie-state` owns branch-local lineage
  limits, reservation, typed preflight refusal and metadata formulas;
  `moxie-executor::PagedAttentionRun` attaches root/fork metadata to the
  existing ledger requests; GPU fixtures and `xtask` pass geometry-derived
  capacities and extend the existing 32K proof. The accepted code baseline
  was `6f5a261`; work began at HEAD `7d2bf3d` (the task-opening commit). No
  commit created. Carried `specification-version.md` and ADRs 0034/0035 are
  preserved.
- **Commands, devices; passed / failed / skipped:** Final restored-tree gates:
  ```text
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --locked -- -D warnings
  cargo clippy -p moxie-executor --all-targets --features driver --locked -- -D warnings
  cargo test --workspace --locked
  cargo xtask arch-check
  cargo xtask spec-check
  CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --test paged_attention_device --features driver,paged-attention-binding,paged-attention-test-hooks --locked
  CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --test dense_gemma_device --features driver,paged-attention-binding,paged-attention-test-hooks --locked
  CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --test dense_tp2_device --features driver,paged-attention-binding,paged-attention-test-hooks --locked
  CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --test tensor_parallel_device --features driver,paged-attention-binding,paged-attention-test-hooks --locked
  CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda test-gpu
  ```
  All passed. Focused lanes were `paged_attention_device` (7/7),
  `dense_gemma_device` (2/2), `dense_tp2_device` (1/1), and
  `tensor_parallel_device` (4/4). The full GPU matrix passed 63/63, zero
  skipped. Devices were SM120 `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` and
  SM86 `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`,
  `GPU-81fe4578-59b2-37c4-421e-287cdac78704`. Existing `moxie-state`,
  `moxie-interp` and engine tests pass unmodified; the one new focused state
  unit test is in `moxie-state/src/lib.rs`. Mutation runs below intentionally
  failed their target assertions; no gate was skipped.
- **Mutation results and restoration:** Each mutation was restored before the
  final gates. Removing root reservation failed the 32K root-capacity
  assertion on all three GPUs. Reserving a child only to `at + 1` failed the
  32K full-child-capacity assertion on all three GPUs. Removing the refusal
  failed `admitted_lineage_limit_is_reserved_inherited_and_preflighted` at the
  root over-capacity check. The expected failures were observed and the
  production checks restored.
- **Review map:**
  - `docs/tasks/0063-m5-admit-sequence-state-growth.md`: records the approved
    maximum, optional per-branch compatibility rule, test and mutation results,
    allocation formulas and review map.
  - `crates/moxie-state/src/lib.rs`: optional per-branch limit; checked exact
    reservation; preflight for prompt, execute, accept and commit; bounded
    fork inheritance/full-capacity clone; root/fork formulas; one focused unit
    test covers unbounded compatibility, reservation/inheritance, typed
    refusal, unchanged frontiers and an abortable refused commit.
  - `crates/moxie-state/src/device.rs`: `DeviceKvSequence::new` reserves root
    to `max_tokens + 1`; helpers expose the root and full-child host charges.
  - `crates/moxie-executor/src/paged_attention.rs`: root/fork admission and
    preview requests include the pageable lineage and transaction metadata.
    The shared lineage buffer request is retained for the run's phases.
  - `crates/moxie-executor/tests/paged_attention_device.rs`,
    `dense_gemma_device.rs`, and `dense_tp2_device.rs`: existing setup paths
    use the admitted sequence request; no new per-method GPU tests.
  - `xtask/src/gpu.rs`: production-shaped run fixtures use the geometry bound;
    the existing 32K gate checks exact root/fork request deltas and verifies
    root, whole-child and chunked-child capacities before and after long
    appends.
- **Allocation inventory:** Let `M = KvGeometry.max_tokens`, `L = checked(M +
  1)`, `S = size_of::<PrefixLineage>()`, and `T(E) =
  btree_node_bound(E) = 11 * (E + size_of::<usize>()) + 16 *
  size_of::<usize>()`. Capacity products and totals use checked arithmetic;
  `T` reuses the existing bound for fixed tuple sizes.
  - `DeviceKvSequence` root lineage `Vec`: charged in the root request as
    `L*S`; pre-reserved exactly to `L` before append growth.
  - Root `SequenceState::open` BTreeMap node from `begin`: charged once as
    `T(size_of::<(StateTransactionId, Journal)>())` in the same root request.
    At most one transaction is open per branch.
  - `SequenceState::fork` lineage `Vec`: charged and reserved as `L*S`, not
    the current prefix length; the child inherits the same optional limit.
  - Fork `SequenceState::branches` node: charged as
    `T(size_of::<(BranchId, Branch)>())`; fork `open` transaction node:
    charged as `T(size_of::<(StateTransactionId, Journal)>())` for that
    branch's possible open transaction.
  - Fork `DeviceKvSequence::branches` node and its `retained_floor` and
    `completed_layers` vectors: charged as
    `T(size_of::<(BranchId, DeviceBranchStorage)>()) + layers *
    (size_of::<u64>() + size_of::<bool>())`.
  - `execute` / `accept` lineage pushes, `commit_prefix` acceptance and
    `append_prompt` lineage pushes: no further allocation while within `L`;
    checked preflight refuses before mutation when the requested high-water
    position needs more than `L` entries.
  - Existing consumers leave this optional limit absent: the host interpreter
    (`moxie-interp`), host/engine consumers and direct tests construct
    `SequenceState` without `reserve_lineage_to`; they retain its unbounded
    behavior. `PagedSequence` in `moxie-state/src/paged.rs` also leaves this
    limit absent and keeps its own existing geometry reservation and append
    bound. The new unit test explicitly exercises the absent-limit behavior;
    existing `moxie-state`, interpreter and engine tests pass unchanged.
  - **(c) Carried ledger follow-up:** `SequenceState::new`'s schema/root branch
    scaffolding (`moxie-state/src/lib.rs:623-638`) and
    `DeviceKvSequence::new`'s layout, root branch map, and per-layer vectors
    (`moxie-state/src/device.rs:307-339`) are allocated during construction,
    before a paged-run request exists; the task 0061 ledger records this
    follow-up. Standalone `DeviceKvSequence` consumers have no paged-run
    request to attach their existing append/commit/fork charges to and remain
    the same ledger follow-up. The new root lineage and transaction charges
    are covered by the root request described above.
- **Remaining obligations:** Those construction-scaffolding and standalone
  consumer ledger follow-ups remain open. No product limit, lease, or other
  `SequenceState` semantics changed.
