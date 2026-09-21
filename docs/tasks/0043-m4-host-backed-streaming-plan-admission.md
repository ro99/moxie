# Task 0043 — M4.3c wire host-backed streaming into graph admission

Status: **accepted** (owner, 2026-09-21). Built by Codex `luna`,
independently reviewed and re-reviewed by Codex `sol`; two rounds — round 1
found a blocking (P1) plan/executor geometry mismatch (the planner accepted
an arbitrary whole-page resident capacity, but task 0042's qualified
executor is scoped to exactly one resident page — a concrete counterexample
showed the planner could admit a plan the executor would then have to
refuse) plus a host-coverage gap on the malformed-capacity branches; round
2's repair constrained admission to match the executor's actual scope by
construction and added the missing coverage, re-reviewed and accepted with
the counterexample traced closed. This substantively completes M4.3's
roadmap text for conventional attention.

## Identity and authority

- Task0043, third bounded M4.3 task; roadmap deliverable 3 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.3`).
  Builder Codex `luna` (max, `/ponytail:ponytail`); independent reviewer
  Codex `sol` (high, read-only, `/ponytail:ponytail-review`); coordinator
  Claude Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `d7aacf5` (task 0042's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: tasks 0041/0042 proved a correct, bounded,
  N-block host-backed streaming *mechanism* at the `moxie-executor`/
  `moxie-memory`/`moxie-state` layer — but nothing in `moxie-plan` knows it
  exists. `stateful_resource_plan`'s servable check
  (`crates/moxie-plan/src/lib.rs:659-663`) recognizes only
  `OpParams::Attention` and `OpParams::MlaAttention`, and every attention
  node is planned as if the whole declared history must already be device
  pages. There is no path from "a real graph declares more history than
  device residency admits" to "the plan reports a streaming requirement and
  the executor's already-qualified N-block path runs it." This is the same
  shape of gap task 0038 closed for paged attention (graph lowering) and
  task 0040 closed for MLA (plan admission) — this task closes it for
  streaming. Coordinator's own M4 ledger names this as M4.3's remaining
  piece, and it is opened now to bring M4.3 to the same completion bar as
  M4.1 and M4.2 before M4.4 starts.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  (attention consumes device tensor handles and page-table/state handles
  from an admitted plan — streaming is not a special case of that contract,
  it is the same contract with a different resident-page count).
- Required source reading: [task 0038](0038-m4-device-kv-state-authority.md)'s
  "second deliverable" (how `OpParams::Attention` first got a plan-admission
  path — the pattern this task follows for streaming), [task 0040](0040-m4-mla-reference-plan-admission.md)'s
  `OpParams::MlaAttention` admission (the second, more recent precedent),
  [task 0041](0041-m4-host-backed-page-streaming.md) and
  [task 0042](0042-m4-n-block-host-backed-streaming.md) in full (the
  mechanism this task connects — do not modify it, call it). Exact API
  surface to reuse: `PagedAttentionLaunch::n_block_stream`
  (`crates/moxie-executor/src/paged_attention.rs:335`),
  `PagedAttentionRun::start_n_block` (`:3371`) and `NBlockStream::stage_next`
  (`:3736`) for execution; `moxie_memory::HostBackedPlan` and its
  `MAX_STAGED_BLOCKS` constant (`crates/moxie-memory/src/report.rs:76`) for
  the admitted bound. `crates/moxie-plan/src/lib.rs` `stateful_resource_plan`
  and `lower_with_ids` in full (where `OpParams::Attention`/`MlaAttention`
  currently become a `StateRequirement` — this task adds the case where
  that requirement cannot be met by device residency alone).
  `crates/moxie-state/src/device.rs` (`DeviceKvSequence`'s admitted
  capacity — the boundary this task's plan-time check reads).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim; a streaming
  plan is legal, not fast, and must not be framed otherwise.

## Bounded deliverable

- **One concrete outcome:** a graph containing a conventional `Op::Attention`
  node whose declared history exceeds what `moxie-state`'s device authority
  can admit as resident pages — but fits within `HostBackedPlan::MAX_STAGED_BLOCKS`
  staged blocks — lowers through `moxie-plan` to an admitted **streaming**
  requirement (not a refusal), and executes end to end via the existing
  `n_block_stream`/`start_n_block`/`stage_next` path, matching the existing
  single-shot device result within `attention_error_bound` on both SM86 GPUs
  and SM120. A graph whose history exceeds even the bounded staging maximum
  still refuses, clearly, exactly as today.
- **Sole owning shared component:** `moxie-plan` for recognizing the
  device-capacity shortfall and reporting a streaming requirement;
  `moxie-executor` for executing an admitted streaming plan through the
  already-built N-block path (no new execution mechanism — task 0042's is
  the only one); `moxie-state` remains the sole capacity authority this task
  reads from, not a second one. A model crate owns none of this.
- **Allowed production and test files/modules:** `crates/moxie-plan/src/lib.rs`
  (the admission/lowering extension — likely a new field on
  `StateRequirement` or a sibling type naming "this requirement needs N
  staged blocks," mirroring how `MlaAttentionDescriptor` was carried),
  `crates/moxie-executor/src/paged_attention.rs` only if the admitted plan
  needs a thin binding to invoke the existing N-block path (prefer reusing
  `n_block_stream`/`start_n_block` unchanged), plus corresponding tests. Do
  not touch `moxie-memory`'s `HostBackedPlan` type or the N-block execution
  internals from tasks 0041/0042 — call them.
- **Explicit non-goals and forbidden shortcuts:** no MLA streaming (task
  0040's MLA path is host-only by design; device MLA execution is separate,
  unopened work, and streaming it is further still); no prefetch/overlap
  (M6.3, unchanged from tasks 0041/0042); no automatic *preference* for
  streaming when device residency alone would suffice — streaming is
  admitted only when device capacity genuinely cannot cover the declared
  history, never chosen for its own sake; no performance claim; no COW,
  prefix reuse, or recurrent/index state.
- **Existing consumers and second-consumer/shape proof:** the existing
  single-shot device path (task 0037/0038) must remain the plan's choice
  whenever device residency alone suffices — this task's own regression
  proof that adding the streaming case did not silently prefer it or change
  behavior when it is not needed.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** none new — this task wires existing, already-proven
  mechanisms together. No algebra changes anywhere.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  unchanged from tasks 0037/0038/0041/0042. The plan-time decision is
  purely "does declared history fit device residency, and if not, does it
  fit within the bounded staging maximum" — a capacity comparison, not a
  numerical one.
- **Partition and hardware capabilities:** both SM86 GPUs and SM120 — this
  is real device execution, reusing an already-qualified path.
- **Peak memory and transfer dependencies; source/lease lifetime:**
  unchanged from task 0042 — one reused staging buffer, bounded N.
- **Cancellation, failure and rollback behavior:** unchanged from task
  0042's mid-sequence fault handling — this task must not weaken it by
  adding a second call path that skips the existing refusal/retention
  logic.
- **Independent oracle; predeclared numerical metrics/thresholds:**
  `attention_error_bound`, unchanged, reused from the already-accepted
  N-block gate.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Host tests: a graph whose declared history exceeds device-admitted
  capacity but fits the bounded staging maximum lowers to a streaming
  requirement rather than a refusal; a graph whose history exceeds even
  that maximum still refuses with the same clarity as today; a graph whose
  history fits device residency plans the existing single-shot path,
  unchanged (regression proof).
- Device tests on both SM86 UUIDs and SM120: the admitted streaming plan
  executes through `n_block_stream`/`start_n_block`/`stage_next` end to
  end and matches the existing single-shot device result within
  `attention_error_bound`, reusing task 0042's own asserted, representation-
  consistent comparison pattern for any new cross-path check this task
  introduces.
- `cargo xtask-cuda test-gpu` passes on all three devices; host suites,
  both clippy lanes, CUDA-feature clippy, `arch-check`, `spec-check`, fmt
  all pass.
- Support-matrix entries: "streamed-page attention" moves from "mechanism
  proven, not reachable from a graph" to "reachable from a real graph under
  resource pressure, bounded N, no overlap" — precise wording, not "long
  context supported."
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** if
  admitting a streaming requirement genuinely requires changing
  `StateRequirement`'s existing shape in a way that could affect the
  already-accepted `Op::Attention`/`Op::MlaAttention` admission paths, stop
  and report rather than risking a regression on tasks 0038/0040's accepted
  gates. If device capacity and staging capacity turn out to need a second
  authority instead of one plan-time comparison against `moxie-state`'s
  existing reports, stop and report.

## Result, filled after work

- Changed shared owners and consumers; source commit: `moxie-plan` now accepts one optional `PagedStateCapacity` report on `ResourceWorkload`, lowers conventional `Op::Attention` to `StateRequirementKind::KvPagesHostBacked { staged_blocks }` only when declared history exceeds resident capacity, and preserves `KvPages` for resident-fit history. The report is composed from `moxie-state`'s existing layout and `moxie-memory::HostBackedPlan::MAX_STAGED_BLOCKS`; it adds no capacity authority. Admission requires exactly one resident page, matching the qualified executor geometry, and rejects zero, non-page-aligned, and multi-page reports. The existing selected graph is lowered and admitted before the unchanged `n_block_stream`/`start_n_block`/`stage_next` execution path runs; the composition-root proof compares its planned staged count with the launch-derived count. Source base: `d7aacf5` (working tree changes are uncommitted).
- Commands and result IDs; passed / failed / skipped separately: `cargo test --workspace --locked`, `cargo test -p moxie-plan --lib --locked`, `cargo test -p moxie-executor --lib --locked`, all three warning-as-error clippy lanes, `cargo xtask arch-check`, `cargo xtask spec-check`, `cargo fmt --all -- --check`, and `git diff --check` passed. `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda test-gpu` passed on both SM86 GPUs and SM120: 60 passed, 0 failed, 0 skipped/unmeasured. No failed or skipped result was recorded.
- Measured effect and uncertainty: the plan fixture admits 3 staged blocks for 8 resident rows, 8-row pages, and 32 visible rows; 8 visible rows remains resident-only, while 40 visible rows is a typed bounded-staging refusal. The GPU N=3 execution used one reused 10,752-byte staging arena and 12,300 bytes total transfer (4,100 bytes per staged block) on all three devices; the existing RNE/numerical gates passed with maximum pairwise error 0. This is a synthetic correctness fixture, not a performance or model-quality claim; prefetch/overlap and MLA streaming remain out of scope.
- Deleted/replaced paths: none. The accepted N=1/single-shot path and bounded N-block mechanism remain unchanged; only plan admission and its composition-root proof were added.
- Remaining blockers and next bounded task: none within this contract. Prefetch/overlap remains the separately gated M6.3 work, and MLA host-backed streaming is not admitted by this task.

Do not fill acceptance with "streaming works" or "long context supported."
Prefetch/overlap (M6.3), MLA streaming and a real long-context product claim
all remain separate, later work even after this task is accepted. This task
completing does not by itself close M4.3 as a roadmap item unless the
Result explicitly says every M4.3 obligation is met.
