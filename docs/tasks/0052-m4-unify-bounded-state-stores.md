# Task 0052 — unify the three duplicated bounded/explicit-restore state stores

Status: **accepted** (owner, 2026-09-22). Built
by Codex `luna`, independently reviewed by Codex `sol` across two rounds.
Every original test's assertions, resource accounting and `StateKind`
wiring were confirmed byte-for-byte preserved on the first pass — but round
1 caught two real regressions the test suite itself didn't exercise: Rust's
default `#[derive]` added an implicit `K: Clone + Copy` bound to the
generic replay-source type, so all three public aliases silently lost
`Clone`/`Copy` they'd had unconditionally before; and the generic
`StateStoreKind` trait was left public and unsealed, letting a downstream
crate define an illegitimate fourth marker and reach an internal
`unreachable!()` panic — a new extension capability the task's own contract
explicitly forbade. Round 2 added the missing derives with a compile-time
assertion guarding against silent regression, and sealed the trait to the
three built-in markers. Re-review didn't just read the seal — it compiled a
hypothetical external implementation against the built crate and confirmed
rustc genuinely rejects it (E0277).

## Identity and authority

- Task0052, M4's audit follow-up, opened immediately as the session's final
  item rather than left dangling. Not a new roadmap deliverable — a
  maintenance/duplication repair the milestone-end
  `/ponytail:ponytail-audit` found and the owner chose to act on now rather
  than defer. Builder Codex `luna` (max, `/ponytail:ponytail`); independent
  reviewer Codex `sol` (high, read-only, `/ponytail:ponytail-review`);
  coordinator Claude Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `fddadd8` (M4 closure commit). Confirm `git status` clean and `HEAD`
  unmoved before starting; report if not. The tree may carry unrelated
  dirty work from a separate session (an ADR touching the roadmap file,
  observed 2026-09-21) — preserve it, do not stage or revert it.
- Requirement repaired: the audit's exact finding — `crates/moxie-state/src/accumulator.rs`
  (731 lines, task 0046), `convolution.rs` (706 lines, task 0047) and
  `sparse_index.rs` (947 lines, task 0048) have byte-for-byte identical
  public APIs (`prefix`, `evidence`, `bytes`, `release`, `new(ledger,
  initial, max_prefix)`, `sequence`, `advance<F>`, `snapshot`, `replay<F>`,
  `rollback_to`, `truncate`, `release`) and identical construction/
  lineage-capacity/control-byte accounting. Checked precisely: the **only**
  genuine difference across all three files is the literal `StateKind`
  value at exactly four call sites each (`SequenceState::new([StateKind::X])`,
  two places in `snapshot`/`replay` that check `evidence.kind() != StateKind::X`,
  and the corresponding `rollback_to`/`replay` publication) — everything
  else, including every byte of the resource-accounting arithmetic, is
  identical. None of the three ever inspects the opaque state's actual
  content; all three are already generic over a caller-supplied
  deterministic step-function closure.
- Required documents: none new — this is not a spec-driven task, it is a
  behavior-preserving internal refactor. `docs/spec/06-implementation-roadmap.md`'s
  M4 section is unaffected; this task neither reopens nor extends it.
- Required source reading: all three files in full, side by side. The
  existing test modules in each (tasks 0046/0047/0048's own adversarial
  tests — non-invertibility/window-loss/value-dependent-eviction
  demonstrations, resource-bound refusal, wrong-ledger-then-correct-ledger
  retry, mid-replay failure preservation) are this task's **regression
  oracle**: every one of them must still pass, unchanged in what they
  assert, against whatever the unified implementation becomes.
- O1–O5 resolved; O6/O7 open — no timing, no performance claim; this task
  is not a performance optimization and must not be framed as one.

## Bounded deliverable

- **One concrete outcome:** the three files collapse to one generic
  implementation (a single struct/impl parameterized by `StateKind`,
  builder's choice of exact mechanism — a const/runtime parameter on
  construction is simplest given `StateKind` is already a plain `Copy`
  enum, no const-generics machinery needed) plus, if any call site outside
  `moxie-state` names `RecurrentAccumulator`/`ConvolutionHistory`/
  `SparseIndex` as distinct types, either thin type aliases preserving
  those names or updated call sites — builder's judgment on which keeps
  the public API cleanest, name the choice in the Result. Every existing
  test from all three files is preserved and still passes, proving the
  unification changed no behavior.
- **Sole owning shared component:** `moxie-state`, unchanged ownership —
  this task does not move the capability anywhere, it removes duplication
  within the crate that already owns it.
- **Allowed production and test files/modules:** `crates/moxie-state/src/accumulator.rs`,
  `convolution.rs`, `sparse_index.rs`, and `lib.rs` (re-exports only). Do
  not touch `paged.rs`, `device.rs`, `prefix_reuse.rs` — those are
  structurally different (own physical bytes directly / publish decisions
  as data / pure decision function) and are not part of this duplication.
- **Explicit non-goals and forbidden shortcuts:** no behavior change of
  any kind — same refusals, same bounds, same accounting, same snapshot/
  replay semantics; no new public API beyond what unification mechanically
  requires; no touching `RestoreCapability`, `restore_evidence`, or any
  other `SequenceState` machinery these stores call; no attempt to also
  unify `paged.rs`/`device.rs` into this — they are genuinely different
  (byte-owning vs. decision-publishing) and forcing them into the same
  shape would be the over-engineering this session has avoided throughout;
  no performance claim.
- **Existing consumers and second-consumer/shape proof:** the three
  original call sites (still nothing outside `moxie-state` consumes any of
  them — unchanged from tasks 0046–0048) are this task's own proof that
  unification didn't silently change what each `StateKind` produces: every
  test tagged for `RecurrentAccumulator`, `ConvolutionHistory` and
  `SparseIndex` must still exercise its own `StateKind` through the shared
  implementation and still pass.
- **Temporary paths to delete or bridge expiry:** `accumulator.rs`,
  `convolution.rs`, `sparse_index.rs` as separate full implementations are
  expected to be deleted or reduced to thin wrappers by this task's own
  completion — that is the point, not leftover scope.

## Contract before implementation

- **Equations:** none — no algebra anywhere in this task.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  unchanged from all three original files.
- **Partition and hardware capabilities:** host only, unchanged.
- **Peak memory and transfer dependencies; source/lease lifetime:**
  unchanged — same `HostBuffer`/`Ledger` accounting, now written once
  instead of three times.
- **Cancellation, failure and rollback behavior:** unchanged — every
  fault-injection and wrong-ledger-retry test from all three files must
  keep passing against the unified implementation.
- **Independent oracle; predeclared numerical metrics/thresholds:** not
  applicable — no numerical claims in this task.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Every test currently in `accumulator.rs`, `convolution.rs` and
  `sparse_index.rs`'s test modules passes unchanged in what it asserts,
  now against the unified implementation, for its own `StateKind`. No test
  is deleted to make the unification easier — if a test genuinely becomes
  redundant across the three (e.g. the same refusal proven three times
  with only the `StateKind` differing), that is fine to note, but the
  *properties* — resource bound refusal, non-invertibility/window-loss/
  value-dependent-eviction demonstrations, wrong-ledger retry, mid-replay
  failure preservation — must still be proven for all three `StateKind`
  values somewhere.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane — this was always host-only.
- Support-matrix entries: update file references only if the module
  layout changed in a way the matrix cites; no capability claim changes —
  this task does not alter what any `StateKind` can do.
- Deletion and documentation gates: state exactly what was deleted (which
  files, or how much of each shrank) in the Result, with a line count
  comparison — that comparison is the actual evidence this task achieved
  its purpose.
- **Exact condition requiring owner direction or task rejection:** if
  unification turns out to require behavior changes to any of the three
  `StateKind`s' semantics (not just code structure), stop and report —
  that would mean the three were not actually identical, contradicting
  this task's own premise, and is an owner question about whether the
  original divergence was intentional.

## Result, filled after work

- Changed shared owners and consumers; source commit: `fddadd8` remained the
  base. `accumulator.rs` now owns one generic `BoundedStateStore<K>`, snapshot
  and replay implementation, selected by zero-sized `StateStoreKind` markers
  carrying each `StateKind`; the trait is sealed to those three markers.
  `convolution.rs` and `sparse_index.rs` are thin aliases, so the existing
  public names and `lib.rs` re-exports remain unchanged. The marker types
  derive `Clone + Copy`, and a compile-time assertion covers all three public
  replay-source aliases. No consumer outside `moxie-state` needed an update.
- Commands and result IDs; passed / failed / skipped separately:
  - `cargo test -p moxie-state --lib --locked --offline`: passed, 88/88;
    all three original store test modules remain unchanged and pass for their
    own `StateKind`.
  - `cargo clippy -p moxie-state --all-targets --locked --offline -- -D
    warnings`: passed.
  - `cargo test --workspace --locked --offline`: passed; no failures or
    skips.
  - `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`:
    passed.
  - `cargo clippy --workspace --all-targets --locked --offline --features
    moxie-executor/driver -- -D warnings`: passed.
  - `cargo xtask arch-check`: passed (79 rejected fixtures, 21 accepted
    fixtures, 13 rules).
  - `cargo xtask spec-check`: passed (10 documents present and unchanged).
  - `cargo fmt --all -- --check` and `git diff --check`: passed.
  - GPU/driver execution: not run; the task contract explicitly excludes
    that lane.
- Measured effect and uncertainty: the three files measured 2,384 lines before
  (`731 + 706 + 947`) and 1,618 after (`846 + 259 + 513`), a reduction of
  766 lines (32.1%). `PhantomData` keeps the generic marker out of the
  admitted struct and snapshot sizes; the existing accounting and behavior
  are covered by the unchanged tests. This is a structural reduction, not a
  performance claim.
- Deleted/replaced paths: the duplicated production implementations in
  `convolution.rs` and `sparse_index.rs` were replaced by aliases (471 and 458
  changed lines respectively); `accumulator.rs` was generalized in place.
  No tests, `lib.rs` re-exports, or unrelated dirty files were deleted or
  changed.
- Remaining blockers and next bounded task: no technical blocker. The
  milestone-end duplication finding is addressed; owner acceptance remains
  pending. No new capability or follow-up task is opened by this refactor.

Do not fill acceptance with "code is cleaner." Line-count evidence and every
preserved test passing for all three `StateKind` values is the acceptance
bar, not a subjective impression.
