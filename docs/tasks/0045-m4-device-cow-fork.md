# Task 0045 — M4.4b copy-on-write branch forking for device-resident KV

Status: **accepted** (owner, 2026-09-21). Built
by Codex `luna` (eager copy, choice (b)), independently reviewed by Codex
`sol` across three rounds — round 1 found the implementation silently
permitted N simultaneous device children despite the task's own single-child
scope, plus two proof-coverage gaps (discard not checked around the actual
`discard_branch` call; the fault-path assertions never confirmed the
surviving branch was genuinely `ROOT`). Round 2's repair refused a second
live child and closed both coverage gaps, but round 2's review then found a
subtler remaining gap: none of the fault-path assertions could detect a
mutant that cleaned up `SequenceState` correctly but left
`DeviceKvSequence`'s own internal branch map stale — every check inspected
the wrong layer. Round 3 closed it with the only observable proof available:
a retry fork must succeed after the fault, which only happens if the
internal map was genuinely cleaned. Accepted with no remaining finding.

## Identity and authority

- Task0045, second bounded M4.4 task; roadmap deliverable 4 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.4`:
  "Implement COW forks... These are prerequisites for the next families and
  samplers, not optional cleanup."). Builder Codex `luna` (max,
  `/ponytail:ponytail`); independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `4b675cc` (task 0044's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: task 0044 qualified `PagedSequence::fork` for host
  paged KV, where the state authority owns the bytes directly. Task 0038's
  `DeviceKvSequence` is architecturally different by its own module doc
  comment (`crates/moxie-state/src/device.rs:1-16`): "this crate may not
  allocate device memory... this module owns exactly the decisions and
  publishes them as data — where a row is placed, what the frontier is,
  which rows are still retained, and what a page table contains.
  `moxie-executor` performs them, through `PagedKvWriter`." So device-side
  COW forking is not the same code as task 0044's — it needs (a)
  `DeviceKvSequence` to hold a second branch's placement/frontier/retention
  bookkeeping (today it is hardcoded to `ROOT` throughout — `:359`, `:542`,
  `:718`, `:885` all pass `ROOT` explicitly, exactly the single-branch shape
  `PagedSequence` had before task 0044) and (b) an actual device-to-device
  page copy performed by the executor through the same `PagedKvWriter`
  mechanism task 0038 already established, not a new write path.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 31 ("`fork` creates copy-on-write branch state"), line 37 ("Treat
  shared pages as immutable until exclusive ownership is established. Port
  GLM-5.3's COW idea and Kimi's recurrent rollback constraint as common
  tests, not family-specific transaction implementations (R20)" — task 0044
  already did this generically for host state; this task extends the same
  generic property to device state, still no model-family code).
- Required source reading: [task 0044](0044-m4-cow-paged-fork.md) in full —
  the exact parent/child isolation proof pattern this task mirrors at the
  device layer (byte-level comparison, not counter equality; an injected
  mid-fork failure at a *genuine* post-logical-fork checkpoint, not an
  early trivial exit — round 1 of task 0044 got both of these wrong before
  round 2 fixed them; do not repeat either mistake here). `crates/moxie-state/src/device.rs`
  in full, especially its module doc comment (the "same mechanism, not a
  second one" architecture, and "the ring is a page table" — `page_table[L]
  = L % pages` — the physical layout a device fork's page table must
  extend, not reinvent) and `SequenceState::fork` (`crates/moxie-state/src/lib.rs:1274`,
  the logical bookkeeping both `PagedSequence` and `DeviceKvSequence` share
  — task 0044 called it unmodified; this task must too).
  `moxie_types::PagedKvWriter` (`crates/moxie-types/src/layout.rs:101`) and
  `PagedKvWriterAdapter` (`crates/moxie-executor/src/paged_attention.rs:3901`)
  — the existing executor-side write mechanism a fork's device copy must go
  through, not a new one. Task 0038's record in full for how the
  authority/executor boundary was established and why ("no trait object, no
  callback into the executor" was the original rule, amended by the owner
  to "the callback stays" — read that amendment before assuming either
  extreme is available to you).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim. Mirror task
  0044's choice: eager copy is the expected default for the same reason
  (correctness before optimization is this session's established pattern),
  unless true device-side lazy sharing is genuinely no more work to get
  right — state which and why in the Result, same as task 0044 did.

## Bounded deliverable

- **One concrete outcome:** `DeviceKvSequence` admits a child branch at a
  committed prefix. The executor copies the parent's committed device pages
  for that branch (through `PagedKvWriter`, eager by default) into a new,
  separately admitted device allocation; the child can append, truncate and
  diverge independently; the parent's committed device bytes, frontier and
  retained range are unchanged by anything the child does — proven by
  reading back both branches' device memory and comparing bytes directly on
  real hardware, not by frontier/counter equality alone (task 0044's round-1
  lesson, applied here from the start).
- **Sole owning shared component:** `moxie-state` for the decisions (a
  second branch's placement, frontier, retention), exactly as it already
  owns them for `ROOT`; `moxie-executor` for the actual device copy and
  admission of the child's pages, through the existing `PagedKvWriter`
  mechanism. No new authority, no third crate.
- **Allowed production and test files/modules:** `crates/moxie-state/src/device.rs`
  (branch-aware placement/frontier/retention — extending, not replacing,
  the `ROOT`-only shape), `crates/moxie-executor/src/paged_attention.rs`
  (the device copy and child admission, through `PagedKvWriter`/
  `PagedKvWriterAdapter`), plus corresponding tests. Do not touch
  `crates/moxie-state/src/paged.rs` (task 0044's host path is done and
  accepted) or `crates/moxie-plan`/`crates/moxie-memory` unless genuinely
  required for a second branch's device-lease accounting — name exactly why
  if so, do not silently expand scope.
- **Explicit non-goals and forbidden shortcuts:** no recurrent/convolution/
  index-state snapshot/replay (document 04's *other* M4.4 obligation, a
  distinct follow-up); no speculation/future-entropy consumer wiring (this
  proves the primitive, same as task 0044); no MLA device forking (MLA has
  no device execution path yet at all — task 0040 is host-only by design,
  forking it is unreachable until that exists); no N-branch generalization
  beyond parent+one child (task 0044's own scope was exactly this bar, and
  N-branch is this task's own natural follow-up, not owed here); no
  performance claim.
- **Existing consumers and second-consumer/shape proof:** none yet, same
  pattern as every task this milestone. The proof is parent-continues /
  child-diverges on real device memory, matching task 0044's structure at
  the device layer.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none — memory-ownership/isolation property, not
  mathematics.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  unchanged — a forked device branch uses the same `KvGeometry`, cache
  precision and page-table mapping (`page_table[L] = L % pages`) as the
  parent.
- **Partition and hardware capabilities:** both SM86 GPUs and SM120 — this
  is real device memory, unlike task 0044.
- **Peak memory and transfer dependencies; source/lease lifetime:** the
  child's copied pages must be admitted through the same lease discipline
  every device allocation in this repository uses (task 0037's event-backed
  completion discipline applies to the copy itself). Report the device
  bytes transferred by the fork copy, honestly — this is diagnostic, not a
  speed claim.
- **Cancellation, failure and rollback behavior:** an injected fault during
  the device copy (matching task 0037/0038/0041's established pattern) must
  leave the parent's device pages and `DeviceKvSequence` state completely
  untouched, with the child's partial allocation released — proven at a
  genuine post-logical-fork checkpoint, not an early trivial exit (task
  0044 round 1's exact mistake).
- **Independent oracle; predeclared numerical metrics/thresholds:** none
  numerical — byte-identity is the oracle, checked by reading back device
  memory through the existing device-readback path both prior GPU tasks
  already use.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Device tests on both SM86 UUIDs and SM120: fork at a committed prefix;
  child appends/diverges independently; parent's device bytes (read back
  and compared directly, not inferred), frontier and retained range are
  unchanged after the child's execution, including after the child is
  discarded. An injected mid-copy fault leaves the parent untouched and
  releases the child's partial allocation — at a real post-logical-fork
  checkpoint. A windowed/evicted parent still forks correctly and the child
  cannot see reclaimed rows (task 0044's equivalent gate, at the device
  layer).
- `cargo xtask-cuda test-gpu` passes on all three devices; host suites,
  both clippy lanes, CUDA-feature clippy, `arch-check`, `spec-check`, fmt
  all pass.
- Support-matrix entries: device-resident paged KV COW moves from "not
  qualified" to exactly this task's scope (parent+one child, both SM86 and
  SM120, mechanism named) — do not write "branching supported" or claim
  speculation readiness.
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** if
  giving `DeviceKvSequence` branch-awareness requires changing
  `PagedKvWriter`'s existing contract in a way that could affect task
  0037/0038's already-accepted single-branch gates, stop and report rather
  than risking a regression. If the executor/authority boundary task 0038
  established (owner-amended: "the callback stays") cannot express a second
  branch's copy without inventing a second callback mechanism, stop and
  report — that boundary is an owner-settled shape, not this task's to
  redesign.

## Result, filled after work

- Changed shared owners and consumers; source commit: working tree based on
  `4b675cc` (no commit made here): `moxie-state::DeviceKvSequence` now keeps
  root and one child branch's frontiers, placements, retained floors and
  transaction bookkeeping, exposed through a borrowed `DeviceBranch` view.
  `moxie-types::PagedKvWriter` gained one defaulted `copy_branch` callback so
  the existing authority/executor boundary remains the only callback shape.
  `moxie-executor::PagedAttentionRun` implements that callback with an eager
  same-device D2D copy of the complete key/value page envelope and the child
  page table; the minimal D2D driver/range plumbing is in
  `moxie-cuda`/`moxie-executor::arena`. `xtask` adds the real-device proof and
  `support-matrix.md` records the bounded capability.
- Commands and result IDs; passed / failed / skipped separately: passed
  `cargo test --workspace --locked` (all host tests and doctests),
  `cargo test -p moxie-state --lib --locked`, host clippy
  (`cargo clippy --workspace --all-targets --locked -- -D warnings`), driver
  clippy (`--features moxie-executor/driver`), CUDA-feature clippy
  (`cargo clippy -p xtask --all-targets --locked --features cuda -- -D warnings`),
  `cargo xtask arch-check` (79 rejected, 21 accepted, 13 rules),
  `cargo xtask spec-check` (10 documents), `cargo fmt --all -- --check`, and
  `git diff --check`. `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda test-gpu`
  passed 63/63 cases, 0 failed and 0 skipped/unmeasured, on both SM86 UUIDs
  and SM120; the new `paged_attention_device_cow` case passed on all three.
  No failed or skipped result was recorded.
- Measured effect and uncertainty: the selected mechanism is (b), eager copy.
  On the synthetic BF16 fixture, the normal fork copied 8,200 device bytes
  (four key/value pages plus the two-entry page table); the windowed fork
  copied 6,152 bytes (three pages plus its two-entry table). The injected
  failure occurred after 2,048 device bytes (one key/value page), after the
  logical child existed, and the child allocation/branch were both released.
  Direct device readback compared all eight inherited rows before and after
  child append, truncate/reappend, parent continuation and child discard;
  the windowed child refused row 0 below its retained 4-row floor. These are
  correctness/byte-accounting measurements, not performance claims.
- Deleted/replaced paths: the `DeviceKvSequence` root-only bookkeeping shape
  and its typed single-branch execution path were replaced by the branch-local
  decision storage; no second state authority, write path, lazy-sharing path,
  N-branch generalization, MLA path, speculation consumer or snapshot/replay
  path was added.
- Remaining blockers and next bounded task: none within this contract. N-branch
  device COW, MLA/device execution, recurrent/convolution/index-state
  snapshot/replay, prefix reuse and speculation/entropy integration remain
  separate follow-up work.

Do not fill acceptance with "branching works" or "speculation ready."
N-branch generalization, recurrent/index-state snapshot/replay, and any
speculation/entropy integration all remain separate, later work even after
this task is accepted.
