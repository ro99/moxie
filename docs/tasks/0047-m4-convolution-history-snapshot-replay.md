# Task 0047 — M4.4d a model-independent snapshot/replay store for convolution history

Status: **accepted** (owner, 2026-09-21). Built
by Codex `luna`, independently reviewed by Codex `sol` — **accepted on the
first round**, this session's first clean pass. Both failure classes task
0046 caught in separate rounds were applied proactively from the start:
exact, traced control-memory accounting mirroring `PagedSequence`'s
precedent (not `accumulator.rs` round 1's `size_of`-only mistake), and a
non-consuming `release` with a wrong-ledger-then-correct-ledger retry test
built in rather than added after a finding. The window-loss demonstration
is a stronger design than task 0046's: two independently-constructed
histories with different early values converge to byte-identical state once
the divergence shifts out of the window — two distinct pasts, one physical
present.

## Identity and authority

- Task0047, fourth bounded M4.4 task; roadmap deliverable 4 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.4`).
  Builder Codex `luna` (max, `/ponytail:ponytail`); independent reviewer
  Codex `sol` (high, read-only, `/ponytail:ponytail-review`); coordinator
  Claude Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `48e37d8` (task 0046's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- **Read [task 0046](0046-m4-recurrent-state-snapshot-replay.md) first, in
  full, including its two review rounds.** This task is structurally the
  same shape — a physical store for a `StateKind` currently backed only by
  logical bookkeeping — and every requirement below assumes you have that
  task's pattern and its two caught defect classes in mind: (1) test rigor
  (non-invertibility demonstrated empirically, replay checked against an
  independent re-run, not the store's own bookkeeping) and (2) resource
  lifecycle (an explicit bound with exact, traced control-memory
  accounting mirroring `PagedSequence`'s precedent; a `release`/consume
  path that never silently loses a value on a recoverable failure). Do the
  adversarial self-review pass for **both** classes this time, not just
  the first — task 0046's round 1 caught the first class cleanly and
  missed the second entirely.
- **What this task is not**, same framing as task 0046: not "implement
  Kimi K3's short convolution." R20 (document 04, line 37) requires the
  *constraint*, not a family-specific implementation. Kimi's actual
  convolution math is M7's model-bring-up-gated scope.
- Requirement repaired: `StateKind::ConvolutionHistory` exists as a typed
  schema entry with `RestoreCapability::Explicit`
  (`crates/moxie-state/src/lib.rs:63-72,148-168`) and inherits the same
  logical `restore_evidence`/`RestoreMethod`/`Restore` machinery task 0046
  proved out. No physical store exists for it — same gap shape as task
  0046, checked the same way (zero hits for `ConvolutionHistory` in
  `moxie-interp`/`moxie-oracles`).
- **Why this is not redundant with task 0046, stated explicitly because it
  is the natural question:** `RecurrentAccumulator`'s irreversibility comes
  from *decay/mixing* — the transform destroys information every step, so
  no representation of the state, however retained, lets you recover an
  earlier one without replay. `ConvolutionHistory`'s irreversibility is
  different in kind: a bounded sliding window of **raw, unmixed** past
  inputs. Nothing is destroyed by the transform itself — if you kept every
  input, you could reconstruct any past window — but the *physical store*
  only retains the last `kernel - 1` raw values, so once a value shifts out
  of the window, the store alone cannot recover it, even though the
  information was never mathematically destroyed. This is a genuinely
  different failure mode for the same `RestoreCapability::Explicit`
  classification, and proving both is what makes the contract actually
  general rather than accidentally shaped around one example.
- Cited for rationale only, not to be implemented — the concrete shape:
  `src/models/kimi_k3/kimi_k3_ops.cpp:78-112` (`kimi_short_conv_step`): per
  channel, a `kernel`-wide causal depthwise convolution over the current
  input plus a `kernel - 1`-wide history buffer of past raw inputs (oldest
  to newest), `SiLU` activation (`x * sigmoid(x)`), then the history shifts
  left and the current input becomes the newest retained value. Hand-verified
  by the coordinator (rationale only, not a fixture to ship): with
  `kernel = 2` (so history width 1), taps `[w0, w1] = [0.5, 1.0]`, initial
  history `h = 0`, inputs `x1 = 2, x2 = 3`: step one computes
  `sum = 1.0·2 + 0.5·0 = 2.0`, output `2.0 · sigmoid(2.0) ≈ 1.7615941559557646`,
  history becomes `[2]`; step two computes `sum = 1.0·3 + 0.5·2 = 4.0`,
  output `4.0 · sigmoid(4.0) ≈ 3.928055160151634`, history becomes `[3]`.
  `/data/kimi-k3/config.json`'s `text_config.linear_attn_config.short_conv_kernel_size`
  is `4` — cited only to show this is a real checkpoint's actual shape.
- Required source reading: `crates/moxie-state/src/accumulator.rs` in full
  — the direct precedent this task mirrors, including its `max_prefix`
  bound, exact control reservation and `&mut self` release-retry pattern.
  `crates/moxie-state/src/lib.rs` `restore_evidence`/`RestoreMethod`/
  `Restore` (unchanged, called not duplicated, same as task 0046).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim.

## Bounded deliverable

- **One concrete outcome:** a host-only, model-independent bounded-window
  history store for `StateKind::ConvolutionHistory` that (a) refuses
  truncation outright, typed, same as task 0046, (b) supports `snapshot()`
  and `replay()` through the existing `restore_evidence` machinery, and (c)
  proves the *window-loss* irreversibility specifically: construct a
  synthetic bounded-window accumulator (window width and per-step transform
  are the builder's choice, need not mirror Kimi's SiLU/convolution
  algebra, only its structural shape: a fixed-width raw sliding buffer that
  shifts each step), demonstrate that a value once shifted out of the
  window cannot be recovered from the current physical bytes alone (a
  genuinely different demonstration from task 0046's decay/mixing
  counterexample — this one shows *information geometrically absent from
  the current bytes*, not information present-but-mixed), then prove
  `snapshot()`/`replay()` correctly reconstructs an earlier window by
  replaying from a retained prefix, matched against an independent
  re-run.
- **Sole owning shared component:** `moxie-state` — a third independent
  physical store alongside `paged.rs`, `device.rs` and `accumulator.rs`.
- **Allowed production and test files/modules:** a new sibling module
  (e.g. `crates/moxie-state/src/convolution.rs`) plus its test module, and
  `crates/moxie-state/src/lib.rs` only for re-exports. Do not touch
  `accumulator.rs`, `paged.rs` or `device.rs`.
- **Explicit non-goals and forbidden shortcuts:** no Kimi K3 math, no
  `kimi_short_conv_step` port, no model crate, no real checkpoint weights
  read (config metadata only, already cited above); no `SparseIndex` store
  (a further, separately-scoped follow-up — name it in the Result, don't
  fold it in); no interpreter/executor/graph wiring; no performance claim;
  no fork/COW semantics (unrelated to this contract, same reminder as task
  0046).
- **Existing consumers and second-consumer/shape proof:** none yet. The
  proof is the window-loss counterexample plus the snapshot/replay
  reconstruction, mirroring task 0046's structure.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none from this task — the synthetic window transform
  exists only to have the window-loss property, same framing as task
  0046's synthetic accumulator.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  opaque bytes, builder's choice of width and window size. No numerical
  requirement.
- **Partition and hardware capabilities:** host only.
- **Peak memory and transfer dependencies; source/lease lifetime:** same
  bar task 0046's round 2 established — an explicit bound (e.g. maximum
  retained prefix or maximum replay depth), exact control-memory
  reservation traced against what `SequenceState::execute`'s lineage
  growth actually allocates, not an approximation. Do not repeat task
  0046 round 1's exact mistake of charging only `size_of` the struct
  itself.
- **Cancellation, failure and rollback behavior:** a `replay()` failure
  partway leaves the store at its pre-attempt bytes/prefix/frontier,
  proven directly, same as task 0046. A `release`/consume path must never
  lose a value on a recoverable failure — same lesson from task 0046
  round 1's second defect, apply it from the start this time rather than
  waiting for review to find it again.
- **Independent oracle; predeclared numerical metrics/thresholds:** none
  numerical — byte-identity against an independently re-run window
  transform is the oracle.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Host tests: truncation is refused, typed. A value shifted out of the
  window is demonstrated unrecoverable from current bytes alone (the
  window-loss counterexample). Snapshot/replay reconstruct an earlier
  window matching an independent re-run, checked directly. A replay
  failure partway leaves pre-attempt state exactly. A release/consume
  failure preserves the value for retry, with a real wrong-then-right
  retry test using the same value (task 0046's exact repair pattern,
  applied proactively). The declared bound is refused beyond, with exact
  control-memory accounting traced, not approximated.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane.
- Support-matrix entries: `ConvolutionHistory` moves from "logical
  bookkeeping only" to "model-independent snapshot/replay proven,
  window-loss shape" — precise wording, no model-support claim.
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** same
  as task 0046 — if proving the window-loss property genuinely requires
  real model math, stop and report (an R20-applicability question, not a
  local design choice). If `restore_evidence`'s contract needs to change,
  stop and report rather than risk the tests already built against it.

## Result, filled after work

- Changed shared owners and consumers; source commit: added the independent
  host-only `moxie-state::convolution` store and its re-export, plus the
  support-matrix gate/capability entries. The working tree remains based on
  `48e37d8`; no commit was created by the builder. The store uses a fixed raw
  sliding window, an explicit maximum prefix, pre-reserved lineage capacity
  and an exact control reservation for the store, schema, lineage and bounded
  `SequenceState` maps. `snapshot`/`replay` call the existing
  `restore_evidence` machinery; truncation is a typed refusal; replay and
  snapshot release failures preserve state for retry.
- Commands and result IDs; passed / failed / skipped separately:
  - Passed: `cargo test -p moxie-state --lib --locked` — 79 passed, 0
    failed; `cargo test --workspace --locked` — passed; both workspace
    clippy lanes and the CUDA-feature `xtask` clippy lane — passed;
    `cargo xtask arch-check` — 79 rejected and 21 accepted fixtures, 13
    rules exercised; `cargo xtask spec-check` — 10 documents unchanged;
    `cargo fmt --all -- --check`; `git diff --check`.
  - Failed: none.
  - Skipped: GPU/driver lanes, real model math, model graph/interpreter/
    executor wiring and `SparseIndex`, all outside this host-only contract.
- Measured effect and uncertainty: the synthetic raw window proves a value
  becomes absent and ambiguous after it shifts out; independent replay
  reproduces captured windows byte-for-byte, including a failed-replay
  rollback and wrong-ledger-then-correct-ledger snapshot release. The bound
  prevents prefix growth beyond the pre-reserved control budget. No model
  quality or performance claim is made.
- Deleted/replaced paths: none; no existing state store was modified.
- Remaining blockers and next bounded task: no local blocker. `SparseIndex`
  physical storage, real convolution/Kimi math, and any consumer wiring
  remain separate follow-up work.

Do not fill acceptance with "convolution models supported." `SparseIndex`'s
own store, real model math (M7 scope), and any interpreter/executor/graph
wiring all remain separate, later work even after this task is accepted.
