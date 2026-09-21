# Task 0041 — M4.3a a two-block host-backed streaming vertical slice

Status: **accepted** (owner, 2026-09-21). Built by Codex `luna`, independently
reviewed and re-reviewed by Codex `sol`; two rounds — round 1 found a genuine
architectural stop condition (the existing qualified kernel could not
produce partial online-softmax state), resolved by the coordinator
authorizing an additive, separately-qualified kernel variant, then a
blocking (P1) numerical-gate defect (the streamed-vs-single-shot comparison
was never actually asserted, and its tolerance was masking an FP64/BF16
representation mismatch rather than checking anything real) and one
ponytail finding; round 2's repair was re-reviewed and accepted, confirmed
load-bearing (would have failed round 1's own defect). This closes M4.3's
first bounded slice; N-block generalization, prefetch/overlap and any
long-context product claim remain separate, unopened work.

## Identity and authority

- Task0041, first bounded M4.3 task; roadmap deliverable 3 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.3`:
  "Add bounded host-backed state/page streaming with exact online-softmax
  merge and honest transfer diagnostics. Resource pressure must produce a
  legal plan or clear admission rejection."). Builder Codex `luna` (max,
  `/ponytail:ponytail`); independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `060eeac` (task 0040's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. Note: the tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: today, **every** device-resident paged attention
  launch requires the *entire* visible history to already be device pages —
  `moxie-state`'s device authority (task 0038) admits a **capped** capacity
  (`window + tentative_rows` rounded to whole pages, plus one page) and
  refuses any launch declaring more history than it has committed. There is
  no path for "the logical context is longer than what fit in admitted
  device pages, so attend the extra history from host-staged pages." That
  path — bounded, exact, with an honest transfer cost — is what M4.3 adds.
  Document 04: "For context that does not fit device KV, implement a shared
  streamed-page exact fallback with bounded staging and online-softmax
  merge... This bounds working memory but still transfers/scans required
  history; report the bandwidth cost." Document 03: "explicit host-backed
  execution" is a named legal alternative when a request cannot otherwise be
  admitted — today `moxie-memory::LegalAlternative::HostBackedExecution`
  (`crates/moxie-memory/src/report.rs:188`) is only a suggestion string;
  nothing executes it for KV/page streaming.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 23 (the merge equations: disjoint key blocks with partial states
  `(m_i, l_i, o_i)`, `m = max(m_i)`, rescale by `exp(m_i - m)`, sum, divide),
  [document 03](../spec/03-memory-formats-and-cuda.md) (reject impossible
  requests with legal alternatives; do not silently shorten context or run
  an unbounded fallback).
- Required source reading: `crates/moxie-oracles/src/online_softmax.rs` in
  full — **the merge algebra already exists and is already proven** (`Partial`,
  `Partial::block`, `Partial::merge`, `attend_row_blocked`; task 0037's own
  comment: "the transfer diagnostics of host-backed streaming... is later M4
  work" — this task is that work; do not re-derive or re-prove the algebra,
  call it). `crates/moxie-executor/src/paged_attention.rs` in full (today's
  single-shot device launch — every block this task adds must still route
  through this same kernel/launch, once per block, not a new kernel).
  `crates/moxie-state/src/device.rs` (`DeviceKvSequence`, admitted capacity,
  `RetentionCapability` — the boundary this task must recognize: when
  declared history exceeds what this authority can admit as device pages).
  `crates/moxie-state/src/paged.rs` (`PagedSequence`, the host ring store —
  the source a streamed block stages *from*). `crates/moxie-memory/src/ledger.rs`
  and `report.rs` (`LegalAlternative::HostBackedExecution`,
  `alternatives()` — the admission-report contract this task's rejection/plan
  path should extend rather than duplicate). Task 0037 and 0038's records for
  precedent: both split "prove the mechanism on a bounded case" from
  "generalize ownership," which this task does too.
- O1–O5 resolved; O6/O7 open — report transfer bytes honestly; no speed
  claim, no "this is fast" framing anywhere.

## Bounded deliverable

- **One concrete outcome:** one query attends correctly over a declared
  history split across **exactly two blocks** — one already device-resident
  (admitted the normal way, task 0038's path) and one staged on demand from
  the host paged store into a bounded device staging buffer — with the two
  blocks' partial attention results merged via `moxie_oracles::online_softmax::Partial::merge`
  to produce the same answer (within `attention_error_bound`) as attending
  the same total history in one shot when it *does* fit entirely in device
  residency. This is the smallest slice that proves the mechanism: stage,
  launch-per-block, merge, transfer-diagnostic. Generalizing to N blocks,
  prefetch/overlap and repeated multi-step streaming are named follow-up
  work, not this task's scope.
- **Sole owning shared component:** `moxie-state` for recognizing "declared
  history exceeds device-admitted capacity" and for owning the host-source
  boundary (it already owns the host ring and the device authority
  separately; this task connects them, it does not create a third
  authority). `moxie-executor` for staging the extra block to a bounded
  device buffer and issuing the second kernel launch. `moxie-memory` for the
  admission decision (legal streaming plan vs. clear rejection) — extend
  `alternatives()`/`LegalAlternative`, do not invent a parallel admission
  path. A model crate owns none of this.
- **Allowed production and test files/modules:** `crates/moxie-state/src/device.rs`
  and/or a new sibling module for the host-source boundary,
  `crates/moxie-executor/src/paged_attention.rs` (block-staged launch path),
  `crates/moxie-memory/src/ledger.rs` and `report.rs` (the admission
  decision), plus corresponding test modules. Do not touch
  `crates/moxie-oracles/src/online_softmax.rs`'s algebra — call it, extend
  its tests only if a genuine gap in the merge oracle itself is found (stop
  and report if so, that would be a task 0037-scope finding surfacing late).

  **Amended by the coordinator, 2026-09-21**, after round 1's source trace
  correctly hit this task's own named stop condition:
  `crates/moxie-kernels/cuda/paged_attention.cu` discards its running
  `(max, sum, weighted)` state and writes only the finished normalized BF16
  output (`:155-163` init, `:231-278` fold, `:289-295` write), and
  `moxie_oracles::online_softmax::Partial`'s `weighted` field is private, so
  neither side of the merge can be produced from the existing qualified
  kernel/launch without changing its contract. **Authorized:** a new,
  separately-qualified kernel entry point and launch path that exposes the
  per-block partial `(max, sum, weighted)` state — added alongside, not
  replacing, the existing single-shot kernel, which keeps its own launch
  contract, ABI and qualified gates completely unchanged. `crates/moxie-kernels/cuda/paged_attention.cu`
  and its CUDA-feature launch binding in `crates/moxie-executor/src/paged_attention.rs`
  are added to this task's allowed files for that purpose only. This is a
  routine technical choice under existing requirements (AGENTS.md: "Resolve
  normal technical choices through source, tests and measured ADRs; do not
  ask the owner to design kernels"), not a numerical-gate change, a
  compatibility sacrifice, or operational expansion — the new variant is
  qualified against the same `attention_error_bound` on both SM86 GPUs and
  SM120 like every other kernel in this repository. If a private field must
  become crate-visible for the merge to consume it, that is an ordinary
  visibility change, not an algebra change; the algebra itself in
  `online_softmax.rs` stays untouched as this task already required.
- **Explicit non-goals and forbidden shortcuts:** no prefetch, read-ahead or
  transfer/compute overlap (M6's scope by name); no N-block generalization
  beyond the two-block proof; no absorbed/MLA streaming (M4.2 built the MLA
  reference path host-only, on purpose; device MLA is separate, unopened
  work); no COW forks or prefix reuse (M4.4/M4.5); no recurrent/index state;
  no tensor cores; no performance claim of any kind — "report the bandwidth
  cost" means measure and state it, not optimize it. Do not silently shrink
  the requested context, silently drop to host-only execution, or run an
  unbounded streaming loop with no admitted staging-buffer size.
- **Existing consumers and second-consumer/shape proof:** none yet — same
  pattern as tasks 0037/0038/0039/0040, this is shared mechanism proven on a
  bounded synthetic case, not a model. The consistency check against the
  existing single-shot whole-history device path (same total history, no
  streaming needed) *is* this task's second-consumer-equivalent proof: two
  independent code paths must agree.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** document 04's merge, already implemented and proven in
  `online_softmax.rs` — call `Partial::block` for each block's local
  attention result and `Partial::merge` to combine them, then `Partial::finish`.
  This task's own contribution is producing each block's `Partial` from a
  real device kernel launch (today's kernel produces a finished, normalized
  result — determine whether it can expose the pre-`finish` partial state,
  or whether the merge instead operates on two finished per-block results
  computed against a shared running maximum; either is legitimate, but the
  choice and why belongs in this task's Result, not silently assumed).
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  BF16 cache and activations throughout, matching every existing paged
  attention gate. The merge itself runs in whatever precision
  `online_softmax.rs` already establishes as correct against
  `attention_error_bound` — do not invent a new tolerance.
- **Partition and hardware capabilities:** both SM86 GPUs and SM120 — this
  is a real device change (a second launch, a staged transfer), so it needs
  the same real-hardware qualification tasks 0037/0038 required, not a
  host-only proof.
- **Peak memory and transfer dependencies; source/lease lifetime:** the
  staging buffer for the host-sourced block must be an explicit, bounded
  allocation admitted the normal way (no untracked scratch); its lease
  lifetime follows the same enqueue/completion-event discipline task 0037
  established. Report the exact bytes transferred host-to-device for the
  staged block.
- **Cancellation, failure and rollback behavior:** if staging the host block
  fails partway (a bounded, injectable fault, matching task 0038's pattern),
  the query must not read a partially-staged buffer; refuse rather than
  silently attend over garbage or a short history.
- **Independent oracle; predeclared numerical metrics/thresholds:**
  `attention_error_bound` (ADR 0028), unchanged. The cross-check that
  two-block streamed attention equals single-shot whole-history attention on
  the same GPUs is this task's own correctness proof, exact within that same
  bound.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Device tests on both SM86 UUIDs and SM120: a query attends over a history
  split as one device-resident block plus one host-staged block; the merged
  result matches the same history attended in one shot via the existing
  single-shot device path (task 0037/0038's launch), within
  `attention_error_bound`. A bounded injected staging failure is refused,
  not silently short-read. Transfer bytes for the staged block are reported
  and match the block's actual admitted size.
- Admission (host test): a request whose declared history exceeds
  device-admitted capacity either produces a legal two-block streaming plan
  naming the staging-buffer size, or a clear typed rejection when even
  bounded staging cannot close the shortfall — never a silent unbounded
  loop or a silently shortened context.
- `cargo xtask-cuda test-gpu` passes on all three devices; host suites,
  both clippy lanes, CUDA-feature clippy, `arch-check`, `spec-check`, fmt
  all pass.
- Support-matrix entries: "streamed-page attention" moves from "not
  implemented" to the two-block proof's exact scope — name it precisely
  (two blocks, no prefetch, no N-block generalization), not "host-backed
  streaming supported."
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** if
  producing a device-kernel partial (rather than only a finished per-block
  result) requires changing `paged_attention`'s existing kernel/launch
  contract in a way that could affect today's single-shot path's qualified
  behavior, stop and report rather than risking a regression on already-
  accepted gates. If the host/device capacity boundary this task needs does
  not fit cleanly into `moxie-state`'s existing device-authority ownership
  (task 0038) without a second authority, stop and report — that is an
  architectural decision, not a routine integration choice.

## Result, filled after work

- Changed shared owners and consumers; source commit: base `060eeac` (the
  working tree is intentionally uncommitted). `moxie-state` now exposes one
  logical host block, `moxie-memory` reports a bounded
  `HostBackedPlan`/typed rejection, `moxie-kernels` adds the separately
  qualified partial-state entry point, and `moxie-executor` stages and launches
  the resident and host blocks. The merge is performed at the `xtask`
  composition root by calling `moxie_oracles::online_softmax::Partial::merge`.
  The existing single-shot kernel, ABI and launch path are unchanged.
- Commands and result IDs; passed / failed / skipped separately:
  `cargo test --workspace --locked`, both workspace clippy lanes, CUDA-feature
  clippy, `cargo fmt --all -- --check`, `cargo xtask arch-check`, and
  `cargo xtask spec-check` all passed with zero failures. The targeted state
  suite passed 10/10 and the admission suite passed 43/43. `cargo
  xtask-cuda test-gpu` passed 57/57 on both SM86 GPUs and SM120, with zero
  failures and zero skipped/unmeasured cases.
- Measured effect and uncertainty: the two-block fixture transferred exactly
  4,100 B for one BF16 K/V page and its table entry. Streamed-vs-FP64-oracle
  maximum error was `4.149595328462041e-8`. The streamed result is now
  explicitly narrowed to BF16 before the asserted cross-path comparison;
  maximum pairwise BF16 error against the existing single-shot path was `0`,
  with each lane checked against its `attention_error_bound` and no extra ULP
  allowance. The earlier `1.5891507770836588e-3` value was the invalid
  FP64-to-BF16 representation mismatch and is not a cross-path result. An
  injected fault after staged-key publication was refused and retained the
  source. This is a synthetic one-query, one-resident-plus-one-staged-page
  qualification: no prefetch/overlap, N-block generalization, timing, or
  model-quality claim is made.
- Deleted/replaced paths: none. The old single-shot device path remains
  intact; the new partial kernel and launch path are additive. The executor
  returns raw device partials so the composition root can invoke the existing
  oracle merge without creating a production executor-to-oracle dependency.
- Remaining blockers and next bounded task: none within this contract. N-block
  streaming, prefetch/overlap, and product-scale long-context/MLA streaming
  remain separate follow-up work.

Do not fill acceptance with "streaming works" or "long context supported."
N-block generalization, prefetch/overlap and a real long-context product
claim all remain separate, later work even after this task is accepted.
