# Task 0050 — M4 closure: 32,768-token prefill/decode with later-turn continuation

Status: **accepted** (owner, 2026-09-22). Built
by Codex `luna`, independently reviewed by Codex `sol` across two rounds —
the deepest and most consequential review of this session, given this
task's purpose. Round 1 found a serious gap that went to the task's own
central claim: the reported "decode after" the second turn was not a
genuine post-prefill append and decode — it re-queried the same row the
255→256-row prefill had already computed, so a mutant unable to actually
append or attend after the second turn would still have passed. Repaired
by prefilling 255 rows instead of 256, then appending and decoding row
33,023 as a genuinely separate step on both the whole and chunked
children, matching the original single-turn gate's own two-part structure
(multi-row parity, then a real decode-after). sol also caught the parent's
resource-admission envelope having been silently widened during
development and required both children's actual device arenas to be
reported, not just the parent's readback bytes. Re-review traced the exact
row-33,023 append/decode call sequence on both children and confirmed the
fix is load-bearing — the previous defect could not survive it.

## Identity and authority

- Task0050, the milestone-closing integration task for M4. Not a new
  roadmap sub-item — it is what makes M4's own exit gate checkable at all.
  Builder Codex `luna` (max, `/ponytail:ponytail`); independent reviewer
  Codex `sol` (high, read-only, `/ponytail:ponytail-review`); coordinator
  Claude Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `cd5c3d2` (task 0049's acceptance commit). Confirm `git status`
  clean and `HEAD` unmoved before starting; report if not. The tree may
  carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- **Why this task exists, stated precisely:** M4's exit gate
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md)):
  "supported graph runs at **32,768 actual context tokens**, including
  decode after long prefill **and later-turn continuation**, within the
  declared cache precision. Compare whole/chunked/multiturn results; test
  page/ring/tail boundaries." Tasks 0037/0038 already proved 32,768 actual
  rows with whole-vs-chunked prefill parity, decode after, and page/tail
  boundaries — **but as one continuous single-turn sequence**, never
  interrupted and resumed. Tasks 0044/0045 proved COW forking is correct
  in isolation, at small synthetic scale. Task 0049 proved a prefix-reuse
  admission decision in isolation, with no execution consumer at all. **No
  task has combined any of them, and none has been run at 32K scale with a
  genuine second turn.** The roadmap's own words — "later-turn
  continuation," "multiturn results" — name exactly the gap between "five
  accepted primitives" and "the milestone's exit gate is actually met."
  This task closes that gap using the mechanisms already accepted; it adds
  no new algebra, no new kernel, no new state authority.
- Required documents: the M4 exit-gate text above, quoted in full so this
  task cannot silently narrow it. [Document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 39: "Long conversations and same-user prefix reuse consume this same
  mechanism... if history has been released, recompute or report that the
  requested operation needs re-prefill. Do not silently change results."
- Required source reading: `xtask/src/gpu.rs`'s `paged_attention_32k`
  (`:4232` onward) — the existing single-turn 32,768-row gate this task
  extends with a second turn, not replaces; read its exact geometry
  (`kv_heads: 2, head_dim: 128, page_tokens: 256, pages: CONTEXT/256 + 1`,
  `heads: 8`) and its whole/chunked/decode/tail-boundary comparisons in
  full — this task's second turn must be held to the same rigor, not a
  weaker one. [Task 0045](0045-m4-device-cow-fork.md) in full — its
  `fork_paged_layer` entry point (`crates/moxie-executor/src/paged_attention.rs`,
  cited exactly in task 0045's own Result) is the mechanism this task uses
  to start the second turn from the first turn's committed 32,768-row
  state. [Task 0049](0049-m4-prefix-reuse-key-and-boundary-decision.md) in
  full — its `PrefixReuseKey::decide` is a candidate for how the second
  turn is *admitted* (a `FullHit`/`BoundaryRecompute` decision before the
  fork/continuation executes), though using it is this task's design
  choice to make, not a mandate (see below).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim anywhere.
  This task proves a capability exists and is correct at scale; it does
  not measure how fast it is.

## Bounded deliverable

- **One concrete outcome:** starting from task 0037/0038's existing
  32,768-actual-row single-turn gate (prefill to full history, decode one
  row), a **second turn** continues from that exact committed state —
  built either as a COW fork of the 32K-committed branch (task 0045's
  mechanism) or as an admitted continuation gated through the prefix-reuse
  decision (task 0049's mechanism) or both, builder's informed choice,
  justified in the Result — and the **same rigor the first turn already
  passed** is repeated on the second turn: whole-vs-chunked prefill parity,
  decode after, page/ring/tail boundary correctness, all checked against
  the FP64 oracle under `attention_error_bound`, on both SM86 GPUs and
  SM120. The first turn's committed bytes must be provably unchanged by
  the second turn's existence (the same parent-isolation proof tasks
  0044/0045 already established, now exercised at 32K scale instead of a
  small synthetic one).
- **Sole owning shared component:** none new. This task composes
  `moxie-state` (`DeviceKvSequence`, `fork_paged_layer`, and/or
  `PrefixReuseKey::decide`), `moxie-executor` (`PagedAttentionRun`, the
  existing single-shot and N-block/fork launch paths) and `xtask` (the GPU
  qualification harness) — exactly the crates tasks 0037/0038/0041–0045
  already own. No file outside those may gain new production logic;
  bringing new capability into existence is out of scope, this task proves
  existing capability composes correctly at the declared scale.
- **Allowed production and test files/modules:** `xtask/src/gpu.rs`
  (a new gate case, or an extension of `paged_attention_32k`) is the
  primary allowed change. `crates/moxie-state`/`crates/moxie-executor`
  may be touched **only** if composing the existing mechanisms at 32K
  scale reveals a genuine defect in one of them (in which case: stop,
  report the defect precisely against the task that built it, and treat
  the fix as a repair to that task's contract, not new scope here — do not
  silently patch around a defect in already-accepted work without naming
  it).
- **Explicit non-goals and forbidden shortcuts:** no new kernel, no new
  algebra, no MLA at this scale (MLA has no device execution path at all —
  task 0040 is host-only by design, out of reach here), no streaming at
  this scale beyond what task 0043 already qualified (N-block streaming
  was proven for a small bounded N, not required to also run at 32K in
  this task unless the builder judges it directly relevant to "later-turn
  continuation" — if so, justify that scope addition explicitly, don't
  silently fold it in); no recurrent/convolution/sparse-index store
  involvement (KV-paged attention only, matching every prior 32K gate); no
  real checkpoint, no real tokens, no model-quality claim — this is the
  same synthetic-fixture discipline every M4 task has used; no timing, no
  "this is how fast multiturn continuation is" framing.
- **Existing consumers and second-consumer/shape proof:** this task *is*
  the second-consumer proof the isolated primitives (0044/0045/0049) have
  been missing — the first time any of them is exercised at the scale and
  shape the milestone's own exit gate actually names, rather than a small
  hand-picked fixture.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none new — `attention_error_bound` (ADR 0028), unchanged,
  is the only numerical gate, exactly as in every prior paged-attention
  task.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  match `paged_attention_32k`'s existing geometry exactly, so the first
  and second turn are directly comparable and no new shape class is
  introduced.
- **Partition and hardware capabilities:** both SM86 GPUs and SM120 — same
  bar every M4 device task has held.
- **Peak memory and transfer dependencies; source/lease lifetime:** the
  second turn's admitted state (forked or continued) must be tracked
  through the same lease/ledger discipline every prior task used — report
  the second turn's device bytes honestly, no performance framing.
- **Cancellation, failure and rollback behavior:** not a new requirement —
  reuse task 0045's already-qualified fault-injection pattern if the
  second turn is built via fork; do not weaken it to make this task
  smaller.
- **Independent oracle; predeclared numerical metrics/thresholds:**
  `attention_error_bound`, applied identically to both turns.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Device tests on both SM86 UUIDs and SM120: the first turn reaches
  32,768 actual committed rows exactly as tasks 0037/0038 already proved
  (this is a regression check, not new evidence — it must still pass
  unchanged). A second turn, continuing from that exact committed state,
  independently passes whole-vs-chunked prefill parity, decode-after, and
  page/ring/tail boundary checks at the same or a comparably large scale
  — state the exact second-turn row count used and why it is a genuine
  test of "later-turn continuation" and not a token-sized formality. The
  first turn's committed bytes are proven unchanged by the second turn's
  existence, by direct byte comparison, at 32K scale.
- `cargo xtask-cuda test-gpu` passes on all three devices; host suites,
  both clippy lanes, CUDA-feature clippy, `arch-check`, `spec-check`, fmt
  all pass.
- Support-matrix and M4 ledger entries: record exactly what this task
  proved — "32,768-actual-row single-turn gate plus a qualified second
  turn via [fork|prefix-reuse|both]" — and do not write "M4 complete" in
  any record without the owner's explicit acceptance of this task first.
- Deletion and documentation gates: this task's Result is the evidence the
  owner needs to assess M4's exit gate as a whole; it does not itself
  declare the milestone closed — that is the owner's call, on the package
  this task (and the accepted tasks it composes) provides.
- **Exact condition requiring owner direction or task rejection:** if
  composing the existing mechanisms at 32K scale reveals that one of
  tasks 0037/0038/0044/0045/0049's own accepted contracts cannot actually
  support "later-turn continuation" as document 04 describes it — not a
  bug, but a genuine capability gap in what was scoped — stop and report
  precisely which contract is insufficient and why, rather than inventing
  new mechanism here to paper over it. That would be a real finding for
  the owner, not a defect to quietly patch.

## Result, filled after work

- Changed shared owners and consumers; source is the uncommitted working tree
  based on `cd5c3d2`; `xtask/src/gpu.rs` is the only code path changed,
  extending the existing `paged_attention_32k` composition-root gate. The path uses task
  0045's existing `fork_paged_layer` eager device-COW mechanism, not task
  0049's `PrefixReuseKey::decide`, because the latter is an admission
  primitive with no execution consumer. No moxie-state, moxie-executor,
  kernel or oracle production code changed. This task also updates the
  support matrix and this M4 ledger row with owner-pending status.
- Commands and result IDs; passed / failed / skipped separately:
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda test-gpu`: final **passed**,
    63/63 cases on both SM86 UUIDs and SM120; no skipped/unmeasured devices.
    An intermediate run **failed** all three new cases at a pre-publication
    boundary assertion (`position has not been published`); moving that
    assertion after the child append/commit produced the final pass.
  - `cargo test -p moxie-state --lib --locked`: **passed**, 88/88.
  - `cargo test --workspace --locked --offline`: **passed**, all workspace
    suites and doc tests; no skipped result affected this task.
  - Both workspace clippy lanes, with and without
    `moxie-executor/driver`, and `cargo clippy -p xtask --all-targets
    --locked --offline --features cuda -- -D warnings`: **passed**.
  - `cargo xtask arch-check`: **passed**, 79 rejected and 21 accepted
    fixtures, 13 rules covered. `cargo xtask spec-check`: **passed**, all 10
    normative documents unchanged.
  - `cargo fmt --all -- --check`, `git diff --check`, and
    `cargo check -p xtask --features cuda --locked --offline`: **passed**.
  - No final host command failed or was skipped.
- Measured effect and uncertainty: the second turn prefills **255 actual BF16
  rows**, then appends its 256th row separately as a genuine decode-after
  step. The page-tail decode is position 33,023, yielding 33,024 rows in each
  child. Whole append and chunked `[1, 63, 64, 127]` append outputs are
  byte-identical across the 255-row prefill, and both children produce
  byte-identical decode output for the separately appended tail row. The
  first prefill row, prefill tail row 33,022 and decode-after row 33,023 each
  pass the existing FP64 `attention_error_bound` gate; reported maxima are
  `2.953e-5`, `2.993e-5` and `3.039e-5` respectively. The parent prefix is
  read back directly as 33,554,432 bytes before fork, while each child exists
  and after discard; parent committed frontier and retained range remain
  unchanged. The parent and both child runs retain the same bounded admitted
  arena, 33,915,648 B, while the parent prefix readback remains the separate
  33,554,432 B evidence. The whole child then rewrites inherited row 32,767
  with a distinct payload: its first 32,767 rows still match the parent and
  the rewritten row does not. The existing page-ring lifecycle case also
  passes. Inputs are synthetic BF16 keys, values and queries; there is no
  checkpoint, model-quality or performance claim.
- Deleted/replaced paths: none. No new kernel, algebra, state authority,
  prefix-reuse consumer, MLA path or ring implementation was added.
- Remaining blockers and next bounded task: no technical blocker surfaced;
  owner acceptance of this evidence is required before any M4 closure
  statement. Prefix-reuse execution, device MLA, and larger-context tiers
  remain outside this task.

Do not fill acceptance with "M4 is done" or "long context works." This task
provides the evidence for the owner to assess M4's exit gate; it does not
itself close the milestone. 100k/200k/1m tiers, MLA/streaming at this scale,
and any performance claim remain explicitly out of scope.
