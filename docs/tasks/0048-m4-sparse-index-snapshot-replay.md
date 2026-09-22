# Task 0048 — M4.4e a model-independent snapshot/replay store for a maintained sparse index

Status: **accepted** (owner, 2026-09-21). Built
by Codex `luna`, independently reviewed by Codex `sol`. This closes M4.4's
five-task arc (0044–0048). Round 1's value-dependent-eviction demonstration
and all resource/lifecycle accounting were confirmed correct on first read
— the only finding was a subtle test-rigor gap: the failed-replay test's
pre-attempt state happened to coincidentally equal the post-mutation
failure scratch (a downstream no-op in the original candidate sequence),
so a mutant that leaked scratch into published state on failure would
still have passed every assertion. Round 2 changed the candidate sequence
so the two states genuinely diverge and asserted that divergence
explicitly; re-review traced the exact byte values and confirmed the
repair would catch the described mutant. Production code was correct
throughout both rounds — this was purely a test-strengthening fix.

## Identity and authority

- Task0048, fifth and final bounded M4.4 task; roadmap deliverable 4 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.4`).
  Builder Codex `luna` (max, `/ponytail:ponytail`); independent reviewer
  Codex `sol` (high, read-only, `/ponytail:ponytail-review`); coordinator
  Claude Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `ad840e7` (task 0047's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- **Read [task 0046](0046-m4-recurrent-state-snapshot-replay.md) and [task
  0047](0047-m4-convolution-history-snapshot-replay.md) first, in full,
  including their review outcomes.** Task 0047 was this session's first
  clean round-1 accept because both of task 0046's defect classes (test
  rigor; exact, traced resource accounting with non-consuming
  release-on-failure) were applied proactively. Do the same here from the
  start, for a third time — there is no excuse left for either class
  reappearing in this task.
- **What this task is not**, same framing as tasks 0046/0047: not
  "implement GLM-5.3's DSA indexer." R20 requires the *constraint*, not a
  family-specific implementation. GLM's actual indexer math is M7's
  model-bring-up-gated scope for that family.
- Requirement repaired: `StateKind::SparseIndex` exists as a typed schema
  entry with `RestoreCapability::Explicit`
  (`crates/moxie-state/src/lib.rs:63-72,148-168`), with the same comment
  already in the codebase explaining why: "a model-defined index (GLM-5.3's
  DSA indexer) is **maintained incrementally, not appended**." No physical
  store exists — same gap shape as tasks 0046/0047, same check (zero hits
  for `SparseIndex` in `moxie-interp`/`moxie-oracles`). This is M4.4's
  **last** unopened piece; accepting it closes the roadmap bullet's
  "recurrent/convolution/index-state snapshot/replay" clause in full
  (COW forks were tasks 0044/0045).
- **Why this is a third, genuinely distinct shape, not a relabeling of
  0046 or 0047:** `RecurrentAccumulator`'s irreversibility is decay/mixing
  (every step overwrites everything). `ConvolutionHistory`'s is
  position-based eviction: exactly one oldest entry leaves the window every
  single step, deterministically, regardless of values. A maintained sparse
  index is different again: it is a **bounded top-scoring selection**, and
  eviction is **value-dependent, not position-dependent** — a new candidate
  is inserted only if it outscores the current minimum, and *when* an
  eviction happens (and which entry it removes) depends on the data, not a
  fixed schedule. Some steps evict nothing at all. This means step count
  alone cannot even tell you how many evictions have occurred, which is a
  strictly harder property to reconstruct than either prior store's.
- Cited for rationale only, not to be implemented — the concrete shape:
  `src/models/glm52/glm52_ops.cpp:116-174` (`glm_index_topk_f32`): scores
  every candidate token by a ReLU-weighted multi-head dot product, then
  `std::partial_sort` selects the top-`k` positions (ties broken by lower
  position). The function itself is stateless per call — it is cited only
  to establish that the underlying model mechanism is a genuine bounded
  top-scoring selection, which is the property this task's synthetic
  version must share. Hand-verified by the coordinator (rationale only, not
  a fixture): a bounded top-2 selection buffer, inserting
  `(score, position)` candidates `(5, 0), (3, 1), (8, 2), (1, 3)` in order:
  step 1 inserts `(5,0)` (buffer `[(5,0)]`, not yet full); step 2 inserts
  `(3,1)` (buffer `[(5,0),(3,1)]`, now full); step 3's candidate `(8,2)`
  beats the current minimum `(3,1)`, which is evicted — buffer
  `[(5,0),(8,2)]`; step 4's candidate `(1,3)` does **not** beat the current
  minimum `(5,0)`, so nothing changes — buffer stays `[(5,0),(8,2)]`. **The
  property that matters:** after step 4, the buffer's bytes cannot tell you
  whether an eviction happened at step 3, step 4, both, or neither, and
  `(3,1)`'s value is nowhere in the current bytes — a "pop the newest
  insertion" truncation scheme would not even know there was nothing to pop
  at step 4, and could not reconstruct `(3,1)` for step 2 regardless.
- Required source reading: `crates/moxie-state/src/convolution.rs` and
  `crates/moxie-state/src/accumulator.rs` in full — the two direct
  precedents, especially `convolution.rs`'s bound/control-accounting
  pattern and non-consuming release, which should be reused verbatim in
  shape. `crates/moxie-state/src/lib.rs` `restore_evidence`/`RestoreMethod`/
  `Restore` (unchanged, called not duplicated).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim.

## Bounded deliverable

- **One concrete outcome:** a host-only, model-independent bounded
  top-scoring selection store for `StateKind::SparseIndex` that (a) refuses
  truncation outright, typed, (b) supports `snapshot()`/`replay()` through
  the existing `restore_evidence` machinery, and (c) proves the
  **value-dependent eviction** irreversibility specifically: construct a
  synthetic bounded top-K structure (capacity, comparison rule and
  candidate shape are the builder's choice — need not mirror GLM's
  ReLU-weighted dot product, only the structural property: insertion
  compares against a current minimum and conditionally evicts), demonstrate
  that step count alone does not determine how many evictions occurred (a
  genuinely different demonstration from tasks 0046/0047's — this one
  shows *non-uniform, data-dependent* information loss, not decay or a
  fixed schedule), then prove `snapshot()`/`replay()` correctly reconstructs
  an earlier selection state by replaying insertions from a retained
  prefix, matched against an independent re-run.
- **Sole owning shared component:** `moxie-state` — a fourth independent
  physical store alongside `paged.rs`, `device.rs`, `accumulator.rs` and
  `convolution.rs`.
- **Allowed production and test files/modules:** a new sibling module
  (e.g. `crates/moxie-state/src/sparse_index.rs`) plus its test module, and
  `crates/moxie-state/src/lib.rs` only for re-exports. Do not touch any of
  the three existing store files.
- **Explicit non-goals and forbidden shortcuts:** no GLM-5.3 DSA math, no
  `glm_index_topk_f32` port, no model crate, no real checkpoint weights
  read (config metadata already used in task 0039/0040's citations, not
  reused here); no interpreter/executor/graph wiring; no performance claim;
  no fork/COW semantics; this task closes M4.4's index-state obligation —
  do not open M4.5 or any new milestone item as a side effect.
- **Existing consumers and second-consumer/shape proof:** none yet. The
  proof is the data-dependent-eviction counterexample plus the
  snapshot/replay reconstruction.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none from this task — the synthetic selection rule exists
  only to have the value-dependent-eviction property.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  opaque bytes/candidate representation, builder's choice. No numerical
  requirement.
- **Partition and hardware capabilities:** host only.
- **Peak memory and transfer dependencies; source/lease lifetime:** exact,
  traced control-memory accounting from the first attempt — mirror
  `convolution.rs`'s pattern precisely (lineage capacity, `btree_node_bound`
  for the branch/journal maps), do not repeat `accumulator.rs` round 1's
  `size_of`-only mistake a third time.
- **Cancellation, failure and rollback behavior:** a `replay()` failure
  partway leaves exact pre-attempt state. A `release`/consume path must
  never lose a value on a recoverable failure — build the
  wrong-then-right retry test in from the start, as task 0047 did.
- **Independent oracle; predeclared numerical metrics/thresholds:** none
  numerical — byte-identity against an independently re-run selection
  sequence is the oracle.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Host tests: truncation refused, typed. The value-dependent-eviction
  property is empirically demonstrated (a sequence where a later step does
  *not* evict, following one that did, and the current bytes alone cannot
  distinguish that history from a different insertion sequence reaching the
  same selection set). Snapshot/replay reconstruct an earlier selection
  state matching an independent re-run, checked directly. A replay failure
  partway leaves exact pre-attempt state. A release/consume failure
  preserves the value for retry, proven with a real wrong-then-right test.
  The declared bound is refused beyond, with exact traced control-memory
  accounting.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane.
- Support-matrix entries: `SparseIndex` moves from "logical bookkeeping
  only" to "model-independent snapshot/replay proven, value-dependent
  eviction shape" — precise wording, no model-support claim. Note in the
  same entry (or a nearby summary) that this closes M4.4's
  recurrent/convolution/index-state snapshot/replay obligation across
  tasks 0044–0048, without claiming M4.4 as a whole roadmap deliverable is
  "done" beyond what these five tasks actually proved (N-branch COW
  generalization was named out of scope in tasks 0044/0045 and remains
  open).
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** same as
  tasks 0046/0047 — if proving the property genuinely requires real model
  math, stop and report (an R20-applicability question). If
  `restore_evidence`'s contract needs to change, stop and report.

## Result, filled after work

- Changed shared owners and consumers; source commit: added the independent
  host-only `moxie-state::SparseIndex` store and its re-export, plus the
  support-matrix gate/capability entries. The working tree remains based on
  `ad840e7`; no commit was created by the builder. The store uses fixed-width
  opaque selection bytes, an explicit maximum prefix, pre-reserved lineage
  capacity and exact control accounting for the store, schema, lineage and
  bounded `SequenceState` maps. `snapshot`/`replay` call the existing
  `restore_evidence` machinery; truncation is a typed refusal; replay and
  snapshot release failures preserve state for retry.
- Commands and result IDs; passed / failed / skipped separately:
  - Passed: `cargo test -p moxie-state --lib --locked` — 84 passed, 0
    failed; `cargo test --workspace --locked` — passed; both workspace
    clippy lanes and the CUDA-feature `xtask` clippy lane — passed;
    `cargo xtask arch-check` — 79 rejected and 21 accepted fixtures, 13
    rules exercised; `cargo xtask spec-check` — 10 documents unchanged;
    `cargo fmt --all -- --check`; `git diff --check` (tracked and new files).
  - Failed: none.
  - Skipped: GPU/driver lanes, real GLM-5.3/DSA math, model
    graph/interpreter/executor wiring and performance measurement, all outside
    this host-only contract.
- Measured effect and uncertainty: the synthetic canonical top-2 byte
  selection proves value-dependent eviction directly. Two four-step histories
  have identical final bytes and prefix: one evicts a lower-scoring candidate
  and then performs a no-op, while the other performs no evictions. Independent
  replay reproduces captured selections byte-for-byte, including snapshot and
  start sources; the failed-replay fixture distinguishes pre-attempt
  `[9@3,8@2]` bytes from the post-mutation failure scratch `[8@2,5@0]`
  before asserting preservation of bytes, frontiers and charge.
  The declared bound admits the traced control envelope before growth. No
  model quality or performance claim is made.
- Deleted/replaced paths: none; no existing state store or restore contract
  was modified.
- Remaining blockers and next bounded task: no local blocker. Real DSA/indexer
  math and consumers remain M7/follow-up work; N-branch COW generalization
  remains separately open as specified.

Do not fill acceptance with "sparse attention supported" or "GLM-5.3
supported." Real model math (M7 scope), interpreter/executor/graph wiring,
and N-branch COW generalization all remain separate, later work even after
this task is accepted.
