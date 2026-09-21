# Task 0044 — M4.4a copy-on-write branch forking for host paged KV

Status: **accepted** (owner, 2026-09-21). Built by Codex `luna` (eager copy,
choice (b)), independently reviewed and re-reviewed by Codex `sol`; two
rounds — round 1 found two blocking (P1) proof gaps in the same class as
this session's recurring lesson (the production mechanism was correct, but
the tests didn't prove what they claimed): the main isolation test never
compared the child's *inherited* rows against the parent, only its
newly-appended row, so a corrupting eager copy would have passed; and the
injected-fault test failed before the copy or the logical fork even ran,
proving only trivial early-exit cleanup. Round 2's repair added the missing
inherited-row byte comparison and moved the injected failure to a genuine
post-logical-fork checkpoint with real unwind; re-review confirmed both
traces hold. Device-side COW and recurrent/index-state snapshot/replay
remain separate, unopened M4.4 work.

## Identity and authority

- Task0044, first bounded M4.4 task; roadmap deliverable 4 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.4`:
  "Implement COW forks, recurrent/convolution/index-state snapshot/replay
  contracts and model-independent rollback tests. These are prerequisites
  for the next families and samplers, not optional cleanup."). Builder
  Codex `luna` (max, `/ponytail:ponytail`); independent reviewer Codex `sol`
  (high, read-only, `/ponytail:ponytail-review`); coordinator Claude Opus.
  Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `887026f` (task 0043's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: this is not new scope invented by the coordinator —
  it is an existing, explicit, named refusal.
  `crates/moxie-state/src/paged.rs:857-861`:
  ```rust
  pub fn fork(&mut self, _at: u64) -> Result<BranchId> {
      Err(Error::Unsupported {
          capability: "paged state fork",
          reason: "COW page sharing is not qualified; this pool owns one root branch".into(),
      })
  }
  ```
  and the module's own doc comment (`:1-5`): "All pages belong exclusively
  to one root branch. A fixed envelope is admitted before use; growth, COW
  and device views require later capability qualification." M4.1/M4.2
  already qualified device views; this task qualifies COW. Device-side COW
  forking (`moxie-state::device::DeviceKvSequence`) is named explicitly out
  of scope below — a separate, later task, the same way M4.2 split host
  reference execution from device execution.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 31 ("`fork` creates copy-on-write branch state"), line 35 ("Prefix
  reuse keys and transaction commits include this distinction... Restore
  recurrent/index state through its schema's declared snapshot/replay
  capability, not a blanket `StateKind` boolean" — the *other* M4.4
  obligation, explicitly **not** this task's scope, see below), line 37
  ("Treat shared pages as immutable until exclusive ownership is
  established. Port GLM-5.3's COW idea and Kimi's recurrent rollback
  constraint as common tests, not family-specific transaction
  implementations (R20)").
- Required source reading: `crates/moxie-state/src/lib.rs`
  `SequenceState::fork` (`:1274`, the existing **logical** bookkeeping —
  frontiers, lineage, epoch; already qualified, do not modify its contract)
  and `discard_branch` (the doc comment states the correctness property
  this task must prove for physical bytes too: "document 04 requires the
  parent to be unchanged by [a discarded branch's] execution"). Task 0038's
  record (device page-table indirection is the existing precedent for
  "logical row maps to a physical slot," which a COW scheme reuses the
  vocabulary of, though this task is host-only). `crates/moxie-state/src/paged.rs`
  in full (`PagedSequence`'s single `HostBuffer`/`SequenceState` structure,
  the ring/page mapping, `RestoreCapability::Truncate`'s existing
  windowed-eviction logic — a forked branch's window/reclamation must
  remain sound after this task, not just its fork).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim. Whichever
  physical sharing mechanism this task chooses (see Bounded deliverable),
  it is chosen for correctness, not speed.

## Bounded deliverable

- **One concrete outcome:** `PagedSequence::fork` stops refusing and
  produces a child branch whose paged KV history is correct and provably
  isolated from its parent: the child can append, truncate and diverge
  independently, and the parent's committed bytes and logical state are
  unchanged by anything the child does, tested with a real counterexample
  (write past the fork point on the child, assert the parent's view at
  every retained row is bit-identical to before the fork), not merely
  asserted.
- **Sole owning shared component:** `moxie-state` — `PagedSequence` owns
  this alone, mirroring how it already owns truncation/rollback for a
  single branch. No new crate, no second state authority, no
  `moxie-memory`/`moxie-executor` involvement unless genuinely required for
  lease accounting of a second branch's bytes (see below).
- **Allowed production and test files/modules:** `crates/moxie-state/src/paged.rs`
  and its test module only, unless implementing physical sharing
  genuinely requires a lease-accounting change in `moxie-memory` for a
  second branch's admitted bytes — if so, that file may be touched too,
  but name exactly why in the Result rather than silently expanding scope.
  Do not touch `crates/moxie-state/src/device.rs` — device-side COW is
  explicitly out of scope (see below).
- **Explicit non-goals and forbidden shortcuts:** no device-side COW
  (`DeviceKvSequence` stays exactly as task 0038 left it — a follow-up
  task's scope, not this one's); no recurrent/convolution/index-state
  snapshot/replay (document 04's *other* M4.4 obligation — a distinct
  follow-up task, not folded into this one merely because it shares a
  roadmap line); no speculation/future-entropy integration (forking exists
  for their eventual use, per document 04 — this task proves the primitive
  correct, it does not wire a consumer); no performance claim or framing
  that lazy sharing is "faster" than eager copy — pick the mechanism for
  correctness and boundedness, state which and why.
- **A genuine open implementation choice, not a shortcut to avoid:**
  physical COW can mean either (a) true lazy sharing — parent and child
  reference the same physical page bytes until one of them writes past the
  fork point, at which point that write forces an exclusive copy of just
  the affected page(s) (classic copy-on-write), or (b) eager copy at fork
  time — the child gets its own independent copy of every retained page
  immediately, and no sharing exists afterward. Document 04's requirement
  ("treat shared pages as immutable until exclusive ownership is
  established") is satisfied by either: (b) trivially, because there is no
  sharing to protect; (a) by construction, because a write always
  triggers ownership transfer first. Given O6/O7 remain open and this
  session's pattern has been to prove correctness before optimizing
  (task 0041/0042 explicitly deferred prefetch/overlap the same way),
  **(b) eager copy is the expected default** unless (a) is genuinely no
  more work to implement correctly — the builder decides and states which,
  with rationale, in the Result. Do not silently pick (a) and leave its
  exclusive-ownership transition partially implemented; an incomplete lazy
  scheme is worse than a complete eager one.
- **Existing consumers and second-consumer/shape proof:** none yet — same
  pattern as every M4 task so far. The proof is two independently-executed
  branches (parent continues after fork; child diverges) producing correct,
  isolated results against the existing paged-store test fixtures, not a
  speculation/entropy consumer.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none — this is a memory-ownership/isolation property, not
  new mathematics.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  unchanged — a forked branch's pages use the same `KvGeometry`,
  `HostTier`/precision and ring mapping as the parent's.
- **Partition and hardware capabilities:** host only, this task.
- **Peak memory and transfer dependencies; source/lease lifetime:** a
  child branch's admitted bytes must be tracked through the same lease
  discipline every other admission in this repository uses — no untracked
  allocation, whichever mechanism ((a) or (b)) is chosen. Discarding a
  child branch (`discard_branch`, already qualified logically) must release
  exactly the bytes that branch owns, not the parent's.
- **Cancellation, failure and rollback behavior:** a fork that fails
  partway (a bounded, injectable fault, matching the pattern tasks
  0037/0038/0041 already established) must leave the parent completely
  untouched — no partial child state visible, no parent corruption.
- **Independent oracle; predeclared numerical metrics/thresholds:** none
  numerical — the oracle here is byte-identity: the parent's retained rows
  before and after a child's independent execution must be bit-identical,
  checked directly, not inferred from a counter.
- **Application compatibility and sampler implications:** none in this
  task — `Sampler`/`History` forking, if it needs its own contract, is out
  of scope here (paged KV bytes only).

## Acceptance

- Host tests: fork at a committed prefix produces a child that can append
  independently; the parent's retained rows are bit-identical before and
  after the child appends, truncates and is eventually discarded — checked
  by direct byte comparison, not by counter equality alone (this session's
  recurring lesson: a passing counter is not proof of correct bytes).
  Forking past the parent's accepted frontier is refused, matching
  `SequenceState::fork`'s existing logical check. A windowed/evicted parent
  (some rows already reclaimed by ring overwrite) still forks correctly —
  the child cannot see rows the parent no longer has. An injected
  mid-fork fault leaves the parent untouched. Discarding the child releases
  exactly its own bytes.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane — this task is host-only.
- Support-matrix entries: paged KV COW moves from "not qualified" to
  exactly this task's scope (host only, the chosen mechanism named) — do
  not write "branching supported" or claim speculation/entropy readiness.
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** if
  proving parent isolation for the eager-copy default requires changing
  `SequenceState::fork`'s existing logical contract (already qualified),
  stop and report rather than risking a regression there. If lease
  accounting for a second branch's bytes cannot be expressed through
  `moxie-memory`'s existing ledger without a new capacity concept, stop and
  report.

## Result, filled after work

- Changed shared owners and consumers; source commit: `moxie-state::PagedSequence`
  now admits a host-only child branch through the existing `SequenceState` fork
  and `HostBuffer`/`Ledger` ownership path. The implementation chooses (b), an
  eager copy: fork allocates and copies the complete fixed page envelope before
  publishing the logical child, then exposes only a borrowed `PagedBranch` view
  for append, rollback, row and block access. Child lineage, retained-floor
  metadata, branch/journal map nodes and page bytes are bounded in the child
  reservation. `discard_branch` releases that reservation before delegating
  logical cleanup to the existing `SequenceState::discard_branch`; the accepted
  `SequenceState::fork` contract and `device.rs` are unchanged. The directly
  affected legacy refusal fixture now exercises fork admission and discard.
- Commands and result IDs; passed / failed / skipped separately: passed
  `cargo test --workspace --locked`, `cargo test -p moxie-state --lib --locked`,
  `cargo test -p moxie-state --locked`, host clippy
  (`cargo clippy --workspace --all-targets --locked -- -D warnings`), driver
  clippy (`--features moxie-executor/driver`), CUDA-feature clippy
  (`--features cuda`), `cargo xtask arch-check`, `cargo xtask spec-check`,
  `cargo fmt --all -- --check`, and `git diff --check`. No GPU lane is in this
  host-only contract. Failed: none. Skipped/unmeasured: GPU/device COW,
  sampler-history branching, recurrent/convolution/index snapshots and
  speculation/entropy consumers, all explicitly outside this task.
- Measured effect and uncertainty: the successful fixture asserts the child
  ledger delta is exactly its copied backing plus its bounded branch-control
  envelope, and that discard returns the host charge to the pre-fork value.
  Parent retained K/V bytes and root frontier/lineage are compared directly
  before and after child append, rollback and divergence. The injected
  post-logical-fork checkpoint fault discards the logical child, releases its
  copied buffer, and restores the exact charge.
  The eager choice intentionally duplicates the fixed host envelope; no timing
  or performance claim is made.
- Deleted/replaced paths: the explicit `PagedSequence::fork` typed refusal is
  replaced by the eager host branch path. No device path, sampler path or
  speculative consumer was added.
- Remaining blockers and next bounded task: none within this contract. Device
  COW and the other M4.4 recurrent/convolution/index-state snapshot/replay
  obligation remain separate follow-up work; token-content prefix reuse and
  speculation/entropy integration are also not claimed.

Do not fill acceptance with "branching works" or "speculation ready." Device
COW, recurrent/index-state snapshot/replay, and any speculation/entropy
integration all remain separate, later work even after this task is
accepted.
