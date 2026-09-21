# Task 0040 — M4.2b a reference MLA graph executes on the host interpreter

Status: proposed

## Identity and authority

- Task0040, second bounded M4.2 task; roadmap deliverable 2 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.2`,
  second half: "Implement absorbed/layout-specific paths only after mask,
  projection and rounding validation"). Builder Codex `luna` (max,
  `/ponytail:ponytail`); independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`.
  Confirm `git status` clean and current `HEAD` (task 0039's acceptance
  commit) before starting; report if not.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. `/models`/`/fast/models`
  read-only; no weight reads beyond what task 0039 already cites.
- Requirement repaired: task 0039 built the descriptor and an independent FP64
  oracle but deliberately admitted no execution — `stateful_resource_plan`
  still refuses `Op::MlaAttention` outright. This task is what "mask,
  projection and rounding validation" in the roadmap's own text means before
  any absorbed/fused path may be attempted: a **reference (unabsorbed)** MLA
  graph must actually execute, end to end, on real state, and agree with task
  0039's oracle within rounding.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  ("Plan absorption/fusion only when algebra and required rounding boundaries
  permit it" — this task is what establishes that permission exists, not
  what uses it), [document 02](../spec/02-architecture-and-common-api.md)
  (shared semantic operations, one state-transaction mechanism).
- Required source reading: [task 0039](0039-m4-mla-state-descriptors-and-glm52-fixtures.md)
  in full (the descriptor and oracle this task consumes — do not re-derive
  the algebra, call `moxie_oracles::mla`'s functions as the reference),
  `crates/moxie-interp/src/lib.rs` `Interpreter::run` in full (the stateful
  host step: binds inputs, checks `kv.check_owner`, requires
  `kv.layers() == graph.attention_layers().len()`, publishes frontiers —
  every one of these checks is `Op::Attention`/`KvCache`-specific today and
  is what must generalize or gain a parallel path for `Op::MlaAttention`),
  `crates/moxie-interp/src/kv.rs` in full (`KvCache`, `CacheOwner`,
  `KvHistory`-backed, "M1 reference form: a dense `Vec` per layer" — the
  pattern a host-side MLA cache follows, built on `moxie_oracles::mla`
  instead of `moxie_oracles::attention`), `crates/moxie-plan/src/lib.rs`
  `stateful_resource_plan` (the refusal to narrow, mirroring how task 0038's
  predecessor narrowed it for `Op::Attention`/`StateKind::KvPages`),
  `crates/moxie-state/src/lib.rs` `StateKind::MlaLatent`,
  `MlaLatentDescriptor`, `SequenceState`, `RestoreCapability::Truncate`.
- O1–O5 resolved; O6/O7 open — no timing, no performance claim.

## Bounded deliverable

- **One concrete outcome:** a graph containing `Op::MlaAttention`, composed
  from **existing shared ops only** (`Op::Linear`, `Op::RmsNorm`, `Op::Rope`,
  and — after decompression — `Op::Attention` over the reconstructed
  per-head keys/values) lowers through `moxie-plan`, admits, and executes
  through `Interpreter::run` on the **host only**, against a host-side
  `MlaLatentDescriptor`-shaped cache, across a multi-step sequence (prefill
  then at least one decode step, so the cache is actually read back, not
  only written once). Output matches `moxie_oracles::mla`'s FP64 oracle
  within F32/BF16 rounding at a predeclared bound.
- **Sole owning shared component:** `moxie-plan` for admission,
  `moxie-interp` for host execution and the host-side cache,
  `moxie-state`/`moxie-graph` unchanged from task 0039 except where this
  task's integration genuinely requires widening a signature (e.g. if
  `Interpreter::run`'s single `KvCache` parameter must become "the state
  this graph's layers actually need," that is this task's call to make, not
  a second interpreter).
- **Allowed production and test files/modules:** `crates/moxie-plan/src/lib.rs`
  (narrow `stateful_resource_plan`'s refusal for `Op::MlaAttention` the same
  way it is already narrowed for `Op::Attention`), `crates/moxie-interp/src/lib.rs`
  and `crates/moxie-interp/src/kv.rs` (or a new sibling module for the MLA
  host cache — builder's call which is cleaner; do not fork `Interpreter`
  into two types), `crates/moxie-graph/src/lib.rs` only if `OpParams` needs
  an `MlaAttention` variant to carry the descriptor onto a node (task 0039
  deliberately left this as a bare descriptor type, not a node parameter —
  this task is where that gap closes, if it must). Corresponding tests in
  each.
- **Explicit non-goals and forbidden shortcuts:** no GPU, no CUDA, no device
  kernel, no `moxie-executor` change — this is the host interpreter only,
  mirroring how M1 built a host reference before M1.3's device chain; no
  absorbed/fused attention math of any kind; no DSA/sparse index selection
  (`Op::SparseIndexSelect` stays untouched); no real checkpoint weights, no
  GLM-5.2 model crate; no second state-transaction mechanism — if the
  existing `SequenceState`/`KvCache` plumbing cannot be extended to cover
  `StateKind::MlaLatent` without duplicating what `SequenceState` already
  owns (frontiers, retention, rollback), stop and report rather than
  building a parallel authority; no performance claim.
- **Existing consumers and second-consumer/shape proof:** none yet — same as
  task 0039, this is vocabulary and a reference path, not a model. The
  multi-step (prefill + decode + rollback) test *is* this task's proof that
  the mechanism is real rather than a single-call demo.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** exactly task 0039's — this task adds no new algebra, only
  wiring. Any discrepancy discovered between the oracle and what the graph
  composition can express with existing ops is a stop-and-report condition,
  not a license to add a new op.
- **Shapes, precision, accumulation/rounding, logical/physical layout:** the
  reference path runs in the graph's normal activation precision (BF16
  matching every other shared op); the cache stores `MlaLatentDescriptor`'s
  BF16 latent+rope rows, matching task 0039. Rounding tolerance against the
  FP64 oracle must be predeclared before the comparison, not fitted after
  seeing the result (document 07's rule, already the convention in
  `attention_error_bound`).
- **Partition and hardware capabilities:** host only, this task.
- **Peak memory and transfer dependencies; source/lease lifetime:** host
  `Vec`-backed cache, no allocator, no device lease — same bound as
  `KvCache`'s existing M1 scope.
- **Cancellation, failure and rollback behavior:** the cache must honor
  `RestoreCapability::Truncate` the same way `KvCache::rollback_to` does for
  `StateKind::KvPages` — an aborted or truncated step leaves prior latent
  rows unchanged. Test this explicitly; it is the property `StateKind::MlaLatent`
  already declares and nothing yet exercises.
- **Independent oracle; predeclared numerical metrics/thresholds:**
  `moxie_oracles::mla`'s FP64 functions from task 0039, called directly, not
  reimplemented. State the tolerance before running the comparison.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Host tests only: a small MLA graph (composed from `Linear`/`RmsNorm`/`Rope`/
  `Attention`) lowers through `moxie-plan` without hitting
  `stateful_resource_plan`'s refusal; every other still-unsupported state
  effect (recurrent, sparse index, everything task 0038 already refuses)
  remains refused — the narrowing must be as specific as the one it mirrors.
  A multi-step run (prefill of several tokens, then one decode step reading
  the cache back) through `Interpreter::run` produces output matching the
  FP64 oracle within the predeclared bound. An aborted/truncated step leaves
  the latent cache exactly as it was. Malformed pairing (a cache with the
  wrong `MlaLatentDescriptor` for the graph's layer) is a typed refusal, not
  silently accepted.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU/driver lane touched by this task.
- Support-matrix entries: MLA moves from "no admission contract" to "host
  reference execution only" — state exactly that, not "supported." Absorbed
  execution and device execution both remain explicitly unsupported.
- Deletion and documentation gates: none beyond keeping task 0039's citations
  accurate if this task's integration reveals they were incomplete.
- **Exact condition requiring owner direction or task rejection:** if
  admitting `Op::MlaAttention` genuinely requires a second state-transaction
  mechanism, a second cache-ownership scheme, or a change to
  `Op::Attention`'s existing KV-paged contract to make room for it, stop and
  report rather than building around it locally — that is an architectural
  decision AGENTS.md reserves ("one state-transaction mechanism"), not a
  routine integration choice.

## Result, filled after work

- Changed shared owners and consumers; source commit:
- Commands and result IDs; passed / failed / skipped separately:
- Measured effect and uncertainty:
- Deleted/replaced paths:
- Remaining blockers and next bounded task:

Do not fill acceptance with "MLA works" or "GLM-5.2 supported." Missing
oracle, shape or ownership evidence is not an accepted task. Device execution
and absorbed/fused attention remain separate, later tasks even after this one
is accepted.
