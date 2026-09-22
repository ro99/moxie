# Task 0049 — M4.5a the prefix-reuse identity key and full-hit/boundary decision

Status: **accepted** (owner, 2026-09-21). Built
by Codex `luna`, independently reviewed by Codex `sol` across three rounds
— this task's deepest scrutiny this session. Round 1 confirmed a real
correctness defect the coordinator flagged before handoff: `BoundaryRecompute`
used the branch's single cached `LogitsHandle` as the recompute boundary and
fell back to `from: 0` whenever no handle was retained, discarding
perfectly valid materialized state — document 04's "committed history" vs.
"materialized forward/state position" distinction, violated by conflating
an ephemeral logits cache with the actual execution frontier. Round 2 fixed
it using `Frontiers::executed`, an existing signal exactly matching that
distinction, and honestly narrowed the boundary's documented guarantee to
logical materialization, not physical KV retention. Round 2's own review
then found the repair's tests never actually exercised both sides of the
resulting `min()` — every case had `claimed_prefix == executed`, so a
mutant deleting `executed` from the formula entirely would still have
passed. Round 3 closed it with a fixture where the two genuinely diverge,
pinning both directions.

## Identity and authority

- Task0049, first bounded M4.5 task; roadmap deliverable 5 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.5`:
  "Add same-user prefix reuse and continuation with correct checkpoint/config
  identity; ensure a full prefix hit still produces valid next-token logits
  or recomputes the required boundary step."). Builder Codex `luna` (max,
  `/ponytail:ponytail`); independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `6b989ae` (task 0048's acceptance commit, M4.4 closed in full).
  Confirm `git status` clean and `HEAD` unmoved before starting; report if
  not. The tree may carry unrelated dirty work from a separate session (an
  ADR touching the roadmap file, observed 2026-09-21) — preserve it, do not
  stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: this is not new scope invented by the coordinator —
  it is an existing, explicit, named gap in the codebase's own doc comment.
  `crates/moxie-state/src/lib.rs:340-342` (`PrefixLineage`'s doc comment):
  "It is a lineage, **not** a content digest... Document 04's prefix-reuse
  key — checkpoint, tokenizer/template, configuration and token IDs — is a
  separate identity that composes with this one and **is not implemented
  here**." Everything the *decision* needs once identity is established
  already exists and is qualified: `SequenceState::next_logits_valid`
  (`:840`) and `restore_logits` (`:855`) are exactly "the 'or use a saved
  equivalent forward result' half of document 04," already checking branch,
  prefix, generation and lineage together — "equality of counters proves
  nothing" is already enforced there. What's missing is the piece upstream
  of that: given a *new* request claiming to continue an *existing*
  sequence's prefix, is that claim actually admissible, and if the full
  prefix isn't a hit, what is the exact boundary to recompute from.
- **A genuine open design question, stated up front, not a shortcut to
  avoid:** `moxie-state` has no concept of "checkpoint," "tokenizer" or
  "token ID" today — those are `moxie-storage`/`moxie-model-api`/model-crate
  vocabulary, and `PrefixLineage` is deliberately *not* a content digest
  (chain of epoch+position, not bytes). Actually comparing two token
  sequences for equality is therefore **not this crate's job** and this
  task does not attempt to give it one. The bounded scope: `moxie-state`
  gains a `PrefixReuseKey` type that composes **opaque, caller-supplied**
  checkpoint/tokenizer-template/config identifiers (the caller — whatever
  layer actually holds tokenizer and checkpoint identity, out of scope
  here — is responsible for having already determined that its opaque
  identifiers match and that its own token content matches up to a claimed
  length), and a decision function that takes that key plus a **claimed**
  matched-prefix length and tells the caller: full hit (a live, restorable
  `LogitsHandle` exists exactly there), boundary recompute (name the exact
  position the caller must recompute from), or refused (the claim exceeds
  what is actually committed, or the config identifiers do not match the
  sequence's current generation). If this bound turns out to be unworkable
  — if a real caller cannot supply what this task assumes — that is the
  named stop condition below, not something to quietly redesign around.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 35 (quoted above in full elsewhere in this repo — the committed vs.
  materialized-forward distinction, the four things that make two
  computations different, "changing prefix/config invalidates outputs
  unless an exact saved result is restored"), line 39 ("Long conversations
  and same-user prefix reuse consume this same mechanism... if history has
  been released, recompute or report that the requested operation needs
  re-prefill. Do not silently change results.").
- Required source reading: `crates/moxie-state/src/lib.rs` in full for
  `StateGeneration`, `PrefixLineage`, `LogitsHandle`,
  `next_logits_valid`, `restore_logits`, `invalidate_generation` and
  `handle_is_live` (the private helper both public methods call) —
  everything this task's decision function must call, not duplicate.
  Tasks 0044–0048's provenance pattern (`CacheOwner`, `Restore`) is the
  same discipline applied again: an identity check that cannot be spoofed
  by two things that merely look equal.
- O1–O5 resolved; O6/O7 open — no timing, no performance claim. Prefix
  reuse is a correctness/admission feature in this task, not a speed
  optimization — do not frame it as one.

## Bounded deliverable

- **One concrete outcome:** a `PrefixReuseKey` type in `moxie-state`
  composing opaque `u64` (or similarly opaque, builder's choice) checkpoint,
  tokenizer/template and graph/precision/position-config identifiers, and a
  decision function — given a key, a target `SequenceState`, a branch and a
  caller-claimed matched-prefix length — that returns one of exactly three
  outcomes: **FullHit** (the claimed length equals a live, restorable
  `LogitsHandle`'s prefix on that branch, under the key's declared
  generation — the caller should call the existing `restore_logits`, this
  function does not call it on the caller's behalf, since restoring is a
  mutation and deciding is not), **BoundaryRecompute** (names the exact
  position to recompute from: the largest position that is both within the
  caller's claim and within the sequence's actual committed frontier at an
  unchanged lineage), or **Refused** (the key's config identifiers do not
  match the sequence's current generation, or the claimed length exceeds
  the committed frontier — both are typed, distinct reasons, not one
  generic refusal).
- **Sole owning shared component:** `moxie-state` — this is the same crate
  that already owns `StateGeneration`/`PrefixLineage`/`LogitsHandle`; this
  task composes a new identity type with them, it does not create a second
  identity system.
- **Allowed production and test files/modules:** a new sibling module
  (e.g. `crates/moxie-state/src/prefix_reuse.rs`) plus its test module, and
  `crates/moxie-state/src/lib.rs` only for re-exports. Do not modify
  `next_logits_valid`, `restore_logits`, `PrefixLineage` or
  `StateGeneration`'s existing contracts — call them, do not change their
  behavior. Do not touch `paged.rs`, `device.rs`, `accumulator.rs`,
  `convolution.rs` or `sparse_index.rs`.
- **Explicit non-goals and forbidden shortcuts:** no tokenizer, checkpoint
  or model-crate integration of any kind — every identifier this task
  handles is opaque; no actual token-content comparison (named out of scope
  above, a caller responsibility); no "same-user" session/user-identity
  concept — that is a service-layer (M8) concern, this task only answers
  "given a claimed match, is it admissible and what follows"; no automatic
  recompute execution (this task *names* the boundary, it does not run a
  forward pass — that is the caller's job, through the existing
  interpreter/executor path); no performance claim, no "this makes
  regeneration fast" framing — prefix reuse here is a correctness gate, and
  a `Refused`/`BoundaryRecompute` outcome falling back to full re-prefill
  must remain completely legal and unsurprising.
- **Existing consumers and second-consumer/shape proof:** none yet — same
  pattern as every M4 task this session. The proof is exercising all three
  outcomes (`FullHit`, `BoundaryRecompute`, `Refused` — both refusal
  reasons) against real `SequenceState` fixtures built the same way
  `next_logits_valid`'s own existing tests are (`crates/moxie-state/src/lib.rs:1429`,
  `:1551`), not a new fixture style invented for this task alone.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none — this is an identity/admission decision, not
  mathematics.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  not applicable — opaque identifiers and position integers only.
- **Partition and hardware capabilities:** host only, this task never
  touches a device.
- **Peak memory and transfer dependencies; source/lease lifetime:** none —
  no allocation beyond the key/decision types themselves, which are plain
  value types, no ledger involvement.
- **Cancellation, failure and rollback behavior:** not applicable — the
  decision function is pure and side-effect-free (it reads `SequenceState`,
  it does not mutate it; mutation is the caller's subsequent call to
  `restore_logits` or a fresh forward pass).
- **Independent oracle; predeclared numerical metrics/thresholds:** none
  numerical. The "oracle" is document 04's own stated rule set (quoted
  above) — each outcome's test should cite which sentence of it the test
  proves, the same discipline task 0038's record already modeled.
- **Application compatibility and sampler implications:** none directly —
  this task does not touch `moxie-sampling`.

## Acceptance

- Host tests, one per outcome and per refusal reason, against real
  `SequenceState` fixtures:
  - `FullHit`: a sequence with a live `LogitsHandle` at exactly the claimed
    prefix, matching generation and key — decision is `FullHit`, and a
    counterexample proves it is *not* returned when the lineage at that
    prefix has changed (re-executed) since the handle was minted, even
    though the prefix number is unchanged — "equality of counters proves
    nothing," tested directly, mirroring `next_logits_valid`'s own
    existing discipline.
  - `BoundaryRecompute`: a sequence committed further than any live logits
    handle, or with a claimed length longer than any valid handle but
    within the committed frontier — decision names the correct boundary
    position, checked against what the sequence actually has committed at
    an unchanged lineage, not just the smaller of two numbers.
  - `Refused` (config mismatch): a key whose checkpoint/tokenizer/config
    identifiers do not match the sequence's current `StateGeneration` —
    refused, with that specific reason, even when the claimed length would
    otherwise be a plausible match.
  - `Refused` (claim exceeds committed): a claimed length beyond the
    sequence's actual committed frontier — refused, with that specific
    reason, never silently clipped to the frontier and treated as a hit.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane.
- Support-matrix entries: prefix reuse moves from "not implemented, no
  identity type" to "admission decision proven, opaque identifiers,
  host-only" — precise wording. Do not write "prefix reuse supported" or
  imply any caller (CLI, service, sampler) actually uses this yet.
- Deletion and documentation gates: update `PrefixLineage`'s doc comment
  (`:340-342`) once this lands — it currently states the prefix-reuse key
  "is not implemented here"; correct that statement to reflect what this
  task actually built and where.
- **Exact condition requiring owner direction or task rejection:** if the
  bounded scope above (opaque caller-supplied identifiers, no token-content
  comparison, no execution of the recompute) turns out to be unworkable —
  if a real caller genuinely cannot supply what this task assumes without
  `moxie-state` gaining tokenizer/checkpoint-aware vocabulary it does not
  have today — stop and report rather than quietly expanding this crate's
  ownership. That would be an architecture-boundary question, not a local
  design choice.

## Result, filled after work

- Changed shared owners and consumers; source commit: `crates/moxie-state` now
  owns `PrefixReuseKey`, `PrefixReuseDecision` and the two typed refusal reasons
  in `prefix_reuse.rs`; `lib.rs` re-exports them and corrects `PrefixLineage`'s
  documentation. The support matrix records the host-only admission gate.
  Round 2 repair: `BoundaryRecompute` now derives solely from
  `min(claimed_prefix, Frontiers::executed)`; tests cover a lower retained
  logits handle, no handle, and the changed-lineage case. The boundary is
  documented as logical materialization only; physical KV retention remains a
  separate owner check.
  Round 3 proof repair: the boundary fixture has accepted 10 and executed 6,
  and asserts claim 10 recomputes from 6 while claim 5 recomputes from 5,
  pinning both sides of the minimum.
  Source commit: working tree based on `6b989ae` (owner commit pending).
- Commands and result IDs; passed / failed / skipped separately:
  `cargo test -p moxie-state --lib --locked` — PASS, 88 tests;
  `cargo test --workspace --locked --offline` — PASS;
  `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` —
  PASS;
  `cargo clippy --workspace --all-targets --locked --offline --features
  moxie-executor/driver -- -D warnings` — PASS;
  `cargo clippy -p xtask --all-targets --locked --offline --features cuda --
  -D warnings` — PASS;
  `cargo xtask arch-check` — PASS, 79 rejected fixtures, 21 accepted fixtures,
  13 rules;
  `cargo xtask spec-check` — PASS, 10 documents;
  `cargo fmt --all -- --check` and `git diff --check` — PASS. No failed or
  skipped gates.
- Measured effect and uncertainty: none measured; this is a host-only,
  side-effect-free identity/admission decision, not a performance or model
  support claim. GPU, tokenizer/checkpoint integration, token-content
  comparison and caller recompute execution were not measured or implemented.
- Deleted/replaced paths: none. The old "not implemented here" statement in
  `PrefixLineage` was replaced with the actual `PrefixReuseKey` ownership and
  caller-content boundary.
- Remaining blockers and next bounded task: no technical blocker. A later
  caller-layer task may compare real checkpoint/tokenizer/template/config and
  token identities, then invoke the returned handle or recompute boundary;
  this task deliberately does not add that consumer.

Do not fill acceptance with "prefix reuse works" or "regeneration is fast."
Actual recompute execution, sampler/service-layer integration, real
tokenizer/checkpoint identity, and any performance claim all remain
separate, later work even after this task is accepted.
