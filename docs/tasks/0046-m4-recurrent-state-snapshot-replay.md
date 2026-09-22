# Task 0046 — M4.4c a model-independent snapshot/replay store for accumulated state

Status: **accepted** (owner, 2026-09-21). Built
by Codex `luna`, independently reviewed by Codex `sol` across two rounds —
round 1's test coverage was strong (the non-invertibility property was
empirically demonstrated, not just asserted, and replay was checked against
an independently re-run oracle rather than trusting the store's own
bookkeeping), but review still found two blocking resource-lifecycle
defects of a different shape than this milestone's usual pattern:
unbounded lineage growth with no declared maximum or exact control
reservation (unlike `PagedSequence`'s precedent), and a snapshot `release`
that consumed the snapshot even on a wrong-ledger refusal, permanently
losing it with no retry path. Round 2 added an explicit `max_prefix` bound
with an exact, traced control-memory reservation and changed `release` to
`&mut self` so a refused release preserves the snapshot for retry;
re-review traced the exact allocation accounting and confirmed both hold.

## Identity and authority

- Task0046, third bounded M4.4 task; roadmap deliverable 4 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.4`:
  "recurrent/convolution/index-state snapshot/replay contracts and
  model-independent rollback tests"). Builder Codex `luna` (max,
  `/ponytail:ponytail`); independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `77d36cd` (task 0045's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. `/data/kimi-k3` is a read-only
  checkpoint root, used below for metadata/dimensions only — no weights are
  read and no model math is built by this task.
- **What this task is not, stated up front because it is the easy mistake
  to make here:** this is not "implement Kimi K3's recurrent/KDA state." R20
  (document 04) is explicit: "Port GLM-5.3's COW idea and Kimi's recurrent
  rollback constraint as common tests, **not family-specific transaction
  implementations**." Kimi's actual recurrent math is document 06's M7 row
  for that family ("recurrent/KDA updates, convolution... bounded host arena
  lessons") — a later, model-bring-up-gated task. This task borrows only the
  *shape of the constraint* Kimi's math demonstrates — accumulated state
  that genuinely cannot be recovered by truncation — and builds the generic
  mechanism against a synthetic accumulator, exactly as `PagedSequence`
  (task 0044) and `DeviceKvSequence` (task 0045) were built and proven
  before any model consumed them.
- Requirement repaired: `StateKind::RecurrentAccumulator` and
  `ConvolutionHistory` already exist as typed schema entries with
  `RestoreCapability::Explicit` (`crates/moxie-state/src/lib.rs:63-72,148-168`),
  and the **logical** evidence/provenance machinery for `Explicit` restore
  already exists and is extensively tested — `restore_evidence`,
  `RestoreMethod`, `Restore` (`crates/moxie-state/src/lib.rs:1026-1060` and
  its ~30 covering tests from `:1692` onward, task 0004-era work). What does
  **not** exist anywhere in the tree — checked directly, zero hits in
  `moxie-interp` and `moxie-oracles` for either `StateKind` name — is any
  physical store: no type holds actual accumulated bytes, refuses
  truncation, and produces the snapshot/replay evidence the logical layer
  already knows how to check. This is the same shape of gap task 0044 closed
  for `PagedSequence::fork`: the logical primitive exists and is qualified,
  the physical one does not exist at all.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 35 ("Recurrent state cannot generally recover an earlier state by
  decrementing a position counter... Retain bounded prefix snapshots or
  recompute the accepted prefix from a saved state; account for that cost."),
  line 37 (R20, quoted above).
- Cited for rationale only, not to be implemented — the concrete shape of
  the constraint this task's synthetic fixture must reproduce:
  `src/models/kimi_k3/kimi_k3_ops.cpp:136-169` (`kimi_kda_step`): a
  `[value_dim, key_dim]` state matrix, each step decays every element then
  applies a delta-rule update — `row[k] *= decay[k]` then
  `delta = (value[v] - Σ row[k]·key[k]) · beta` then
  `row[k] += delta · key[k]`, output `Σ row[k]·query[k]`. Hand-verified by
  the coordinator (not shipped as a fixture — the synthetic accumulator
  below need not reproduce this exact algebra, only its *irreversibility*
  property): with `key=[1,2]`, `value=[3]`, `decay=[0.5,0.5]`, `beta=0.5`,
  `query=[1,1]`, starting from a zero state, step one produces state
  `[1.5, 3.0]` and output `4.5`; step two (same inputs) produces state
  `[0.375, 0.75]` and output `1.125`. **The property that matters for this
  task:** step two's state cannot be recovered from step one's state by any
  truncation — the decay has already mixed old and new information
  irreversibly, which is exactly what makes `RestoreCapability::Explicit`
  necessary instead of `Truncate`. `/data/kimi-k3/config.json`'s
  `text_config.linear_attn_config` (`head_dim: 128`, `num_heads: 96`,
  `kda_layers`: 69 of 93 total) is cited only to show this is a real,
  currently-loadable checkpoint's actual shape, not a hypothetical — no
  tensor from it is read by this task.
- Required source reading: `crates/moxie-state/src/lib.rs`
  `restore_evidence`, `RestoreMethod`, `Restore` in full, and their test
  coverage from `:1692` (do not duplicate this logical machinery — call it).
  [task 0044](0044-m4-cow-paged-fork.md) and
  [task 0045](0045-m4-device-cow-fork.md) in full, as the direct structural
  precedent: a physical store built against an already-qualified logical
  primitive, proven by a real counterexample (there: byte isolation across
  divergence; here: that a snapshot/replay actually restores the exact
  accumulator state a naive truncation would get wrong).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim.

## Bounded deliverable

- **One concrete outcome:** a host-only, model-independent accumulator
  store for `StateKind::RecurrentAccumulator` that (a) refuses truncation
  outright — not silently ignores it, a typed refusal — (b) supports
  `snapshot()`, capturing the current opaque state bytes and stamping
  `Restore` evidence through the existing `restore_evidence`, and (c)
  supports `replay()`, reconstructing state at a target prefix by re-running
  a caller-supplied deterministic step function from a snapshot (or from the
  start) forward to that prefix, also stamped through `restore_evidence`.
  The proof: construct a synthetic accumulator whose step function is
  **provably non-invertible by truncation** (any function where step N's
  state is not recoverable from step N+1's state by dropping bytes — a
  running XOR-fold or a decay-and-mix transform in the spirit of the cited
  KDA shape both qualify; pick one, state which and why), snapshot it at a
  prefix, advance it further, then prove `replay()` to the earlier prefix
  produces byte-identical state to what the accumulator actually held there
  — checked directly, not inferred, exactly as tasks 0044/0045 checked KV
  bytes directly rather than trusting a counter.
- **Sole owning shared component:** `moxie-state` — mirrors how
  `PagedSequence` owns host paged KV. No new crate. `moxie-interp`/
  `moxie-oracles`/`moxie-executor` are not touched by this task — there is
  no consumer yet, same as every primitive this milestone has built before
  a model needed it.
- **Allowed production and test files/modules:** a new sibling module in
  `crates/moxie-state/src/` (e.g. `accumulator.rs`, builder's naming
  choice) plus its test module, and `crates/moxie-state/src/lib.rs` only
  for re-exports. Do not touch `paged.rs` or `device.rs` — this is a third,
  independent physical store, not an extension of either.
- **Explicit non-goals and forbidden shortcuts:** no Kimi K3 math, no
  `kimi_kda_step` port, no GLM-5.3 convolution/index math, no model crate,
  no real checkpoint weights read (config metadata only, already used above
  for the citation); no `ConvolutionHistory` implementation in this task —
  name it as a natural, structurally similar follow-up in the Result, don't
  fold it in; no interpreter/executor wiring or graph consumer (this proves
  the primitive, as every prior task in this milestone has); no performance
  claim; no COW/fork semantics on this store (recurrent state's isolation
  problem is snapshot/replay, not branching — do not import task
  0044/0045's fork API here just because it is nearby).
- **Existing consumers and second-consumer/shape proof:** none yet. The
  proof is the non-invertibility counterexample itself: a store that could
  not actually distinguish "restored via replay" from "accidentally
  truncated" would be indistinguishable from the `RestoreCapability::Truncate`
  case this task exists to be different from — the test must show truncation
  is refused *and* that naive byte-truncation of the accumulator (if
  attempted directly, bypassing the API) would have produced the wrong
  state, which is what proves `Explicit` restoration is actually necessary
  here and not a formality.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none from this task — the synthetic accumulator's own
  step function is this task's only "algebra," and it exists solely to
  have the non-invertibility property, not to model anything real.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  opaque bytes, builder's choice of width — this is a memory/restoration
  contract, not a numerical one. No BF16/FP32 requirement.
- **Partition and hardware capabilities:** host only.
- **Peak memory and transfer dependencies; source/lease lifetime:** a
  snapshot's bytes must be bounded and accounted the way every other host
  allocation in this repository is (reuse `moxie-memory`'s existing host
  ledger/tier accounting if a snapshot allocates — do not invent a new
  concept; if the accumulator is small enough to avoid the ledger entirely,
  state that choice and why).
- **Cancellation, failure and rollback behavior:** a `replay()` that fails
  partway (the step function returns an error, or a bounded fault is
  injected, matching this session's established pattern) must leave the
  store at its state *before* the replay attempt, not a partially-replayed
  state.
- **Independent oracle; predeclared numerical metrics/thresholds:** none
  numerical — byte-identity between replayed and originally-computed state
  is the oracle, computed independently by the test (run the step function
  a second, separate time to the same prefix, compare) rather than trusting
  the store's own internal bookkeeping.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Host tests: truncation is refused with a typed error, not silently
  ignored or silently converted to a snapshot. A snapshot at prefix N,
  followed by further steps, followed by `replay()` back to N, produces
  state byte-identical to an independently-recomputed run to N — checked
  directly. The chosen synthetic accumulator's non-invertibility is itself
  demonstrated: naive truncation of its raw bytes at prefix N (if the test
  deliberately does this, bypassing the API, to make the point) does *not*
  equal the state actually held at N, which is the concrete evidence that
  `Explicit` restoration is doing real work here. A `replay()` failure
  partway leaves the store at its pre-replay state, not a partial one.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane — host-only.
- Support-matrix entries: `RecurrentAccumulator` moves from "logical
  bookkeeping only, no physical store" to "model-independent snapshot/
  replay proven" — precise wording; do not write "recurrent models
  supported" or reference Kimi/GLM-5.3 as consumers, since neither is one.
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** if
  proving the non-invertibility property genuinely requires real model math
  rather than a synthetic function, stop and report — that would mean this
  task's premise (a model-independent proof is possible, per R20) is wrong,
  and that is an owner-level question about R20's applicability, not a
  local design choice. If `restore_evidence`'s existing contract needs to
  change to accommodate this store, stop and report rather than risking the
  ~30 existing tests built against it.

## Result, filled after work

- Changed shared owners and consumers; source commit: working tree based on
  `77d36cd` (no commit made here). `moxie-state::RecurrentAccumulator` and
  `RecurrentSnapshot` are the new host-only physical owner. They use the
  existing `HostBuffer`/`Ledger` authority for start, current, scratch and
  snapshot bytes, and delegate identity, lineage, evidence and logical
  rollback to the existing `SequenceState` machinery. The new
  `ReplaySource` accepts either the admitted prefix-zero source or a snapshot;
  replay runs into isolated scratch and publishes only after all steps and
  `restore_evidence` succeed. The support matrix records this as the bounded
  model-independent primitive only.
- Commands and result IDs; passed / failed / skipped separately: passed
  `cargo test --workspace --locked --offline` (all host tests and doctests),
  `cargo test -p moxie-state --lib --locked --offline`, host clippy
  (`cargo clippy --workspace --all-targets --locked --offline -- -D warnings`),
  driver clippy (`cargo clippy --workspace --all-targets --locked --offline
  --features moxie-executor/driver -- -D warnings`), CUDA-feature clippy
  (`cargo clippy -p xtask --all-targets --locked --offline --features cuda --
  -D warnings`), `cargo xtask arch-check` (79 rejected, 21 accepted, 13
  rules), `cargo xtask spec-check` (10 documents), `cargo fmt --all --
  --check`, and `git diff --check`. Failed: none. Skipped/unmeasured: GPU
  lanes, real recurrent/KDA math, `ConvolutionHistory`, and interpreter,
  executor or graph consumers, all explicitly outside this task.
- Measured effect and uncertainty: the synthetic proof uses a fixed 32-byte
  opaque four-lane decay-and-mix accumulator. Direct bytes show a snapshot at
  prefix 2, replay from that snapshot to prefix 5, and replay from start back
  to prefix 2 all match an independently rerun step sequence. Dropping the
  newest raw lane after the next step does not recover the prior lanes. A
  callback failure after three mutating replay steps leaves the original
  bytes, physical prefix and logical frontiers unchanged. State and snapshot
  payloads are bounded by the caller's byte width and admitted through the
  existing host ledger; no timing or performance claim is made.
- Deleted/replaced paths: none. The existing `RestoreMethod`, `Restore` and
  `SequenceState::restore_evidence`/`rollback_to` contract remains unchanged;
  the former lack of a physical `RecurrentAccumulator` store is filled by the
  new sibling module and `moxie-state` re-export.
- Remaining blockers and next bounded task: none within this contract.
  `ConvolutionHistory` needs its own structurally similar store, and real
  recurrent/KDA model mathematics remains the model-bring-up-gated M7 task;
  neither is claimed here.

Do not fill acceptance with "recurrent state works" or "Kimi/GLM-5.3
supported." `ConvolutionHistory`'s own store, any real model's recurrent
math (M7 scope), and any interpreter/executor/graph wiring all remain
separate, later work even after this task is accepted.
