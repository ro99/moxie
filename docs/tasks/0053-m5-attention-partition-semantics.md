# Task 0053 — attention head-ownership, GQA KV-replication and output-reduction partition semantics

Status: **proposed**.

## Identity and authority

- Task0053, M5.1's first bounded slice. Builder Codex `luna` (max,
  `/ponytail:ponytail`); independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `178c10f` (task 0052 acceptance commit). Confirm `git status` clean
  save for the unrelated carried dirty work (`docs/evidence/specification-
  version.md`, `docs/decisions/adr/0034-tokenizers-crate-named-at-m8.md`) and
  `HEAD` unmoved before starting; preserve that dirty work, do not stage or
  revert it, do not include it in this task's commit.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Not expected to be needed —
  this task defines new semantics, it does not port legacy TP code.
- Requirement repaired: roadmap M5.1, "Define legal partition semantics for
  all implemented ops," for the two operations the codebase itself already
  marks as the open part of that gap.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md),
  "Tensor parallelism (TP)" (lines 49–57) — row/column linear semantics,
  activation replication/sharding, attention head ownership, output
  reduction, GQA KV-head replication when ranks exceed KV heads, bias/residual
  applied exactly once, non-divisible dimensions use checked padding or
  explicit unsupported combinations. [Document 02](../spec/02-architecture-and-common-api.md),
  "Planning contract" (per-device/TP/PP placement is a plan property) and the
  crate ownership table (partition legality is a graph/op-contract property,
  not an executor or model concern).
- Required source reading: `crates/moxie-graph/src/graph.rs`'s
  `OpParams::partition_rule()` and its surrounding comments (the exact
  boundary this task moves); `crates/moxie-graph/src/lib.rs`'s `PartitionRule`
  enum and `check_partitionable()`; `crates/moxie-interp/tests/reference_graphs.rs`'s
  `the_routing_operations_declare_their_partition_and_state_contracts` (the
  existing test shape to extend, not replace, for `Route`/`ExpertMlp`/
  `Combine` which stay `NotDetermined` — this task does not touch them);
  `crates/moxie-state/src/device.rs`'s `DeviceKvSequence` and its page-table
  authority (task 0038) — read to confirm this task's semantics are
  expressible against the existing device KV ownership model, not to change
  it; task 0039/0040 (`crates/moxie-oracles`, `moxie-plan`) for how
  `MlaAttention`'s latent/positional state differs from plain multi-head
  attention's per-head KV, since MLA's shared latent projection may need a
  different legal partition than GQA's per-head split.
- Owner gates already resolved: O1–O5. O6/O7 remain open — no timing, no
  performance claim; this task is a correctness/legality contract, not an
  optimization and must not be framed as one. No owner gate is reached or
  reopened by this task; it is ordinary engineering within document 04's
  already-approved TP section.

## Bounded deliverable

- One concrete outcome: `OpParams::Attention` and `OpParams::MlaAttention`
  return a defined, non-`NotDetermined` partition rule (or rules — the two
  operations may legitimately need different legal shapes) from
  `partition_rule()`, capturing document 04's three requirements: which axis
  ranks split heads along, whether/how GQA KV heads replicate when rank count
  exceeds KV-head count, and where the output reduction (concatenation versus
  a global reduction) legally happens. `check_partitionable()` must accept
  both operations afterward.
- Sole owning shared component: `crates/moxie-graph` (the op contract lives
  there; `PartitionRule` may need a new variant or associated data if the
  existing four-variant enum cannot express head-sharding plus conditional KV
  replication — that is this task's design decision to make, not assumed in
  advance).
- Allowed production and test files/modules: `crates/moxie-graph/src/graph.rs`,
  `crates/moxie-graph/src/lib.rs`, and their own test modules;
  `crates/moxie-interp/tests/reference_graphs.rs` to extend the existing
  partition-contract test with `Attention`/`MlaAttention` assertions. Touch
  `crates/moxie-plan` or `crates/moxie-executor` only if the compiler
  requires a matching update for an exhaustive match on `PartitionRule` — do
  not add new planning or execution behavior there; this task does not lower
  anything to TP.
- Explicit non-goals and forbidden shortcuts: no TP2 execution, no rank-group
  type, no collective/NCCL plumbing, no actual multi-rank test (single-rank
  lowering must remain exactly as it behaves today — the existing
  `partition_semantics_fail_closed_until_defined`-style test proves
  `NotDetermined` still fails closed for whatever *is* still undetermined).
  Do not touch `OpParams::Route`/`ExpertMlp`/`Combine` — those stay
  `NotDetermined`, reserved for M5.4, and the existing routing test asserting
  that must still pass unchanged. Do not revisit `Linear`'s existing
  blanket `ColumnShardable` rule (task 0003's contract table; a row-parallel
  O-projection/down-projection optimization is M6 performance scope, not a
  legality gap). Do not invent a rank count, topology or hardware assumption
  — the semantics must hold for any rank count ≥ 1, including the existing
  single-rank case.
- Existing consumers and second-consumer/shape proof: `moxie-interp`'s
  reference-graph fixtures already build both plain multi-head/GQA attention
  and MLA graphs (task 0039/0040) — extend those existing fixtures rather
  than inventing a new synthetic graph, and prove the rule against at least
  one GQA shape (fewer KV heads than a hypothetical rank count) and one MLA
  shape (shared latent projection), since those are the two cases document
  04 calls out by name.
- Temporary paths to delete or bridge expiry: none.

## Contract before implementation

- Equations and semantic input/output/state effects: unchanged — this task
  defines partition *legality*, not new attention mathematics. `state_effect()`
  (`Appends` for both ops) is untouched.
- Shapes, precision, accumulation/rounding, logical/physical layout: head
  ownership must respect existing head-dimension shape contracts; do not
  introduce padding or reshaping behavior — if a head count is not evenly
  divisible by a hypothetical rank count, the rule must say so is illegal
  (checked, not silently truncated), per document 04's "non-divisible
  dimensions use checked padding or explicit unsupported combinations."
- Partition and hardware capabilities: the rule is hardware-agnostic — it
  describes what is *legal*, independent of which/how-many devices exist.
  Do not read `RankId`/`RankContext` or any device count in this task.
- Peak memory and transfer dependencies; source/lease lifetime: none — no
  execution path changes, so no new allocation, transfer or lease.
- Cancellation, failure and rollback behavior: unchanged.
- Independent oracle; predeclared numerical metrics/thresholds: none
  numerical — this is a type/contract-level property, proven by tests
  asserting the returned `PartitionRule` value(s) and by
  `check_partitionable()` accepting/rejecting the right cases, mirroring the
  existing `partition_semantics_fail_closed_until_defined` test's shape.
- Application compatibility and sampler implications: none — no
  service/CLI-visible behavior changes.

## Acceptance

- Architecture / host / GPU / distributed / state tests and required
  hardware: host only (`cargo test --workspace` plus `moxie-interp`'s
  reference-graph suite); `cargo xtask arch-check` and `cargo xtask
  spec-check` clean. No GPU or distributed test required — nothing executes
  differently.
- Actual-context cases and prefill/decode/quality/memory gates: not
  applicable — no execution behavior changes.
- Benchmark manifest, baseline identity, variance/repetition plan: not
  applicable — no performance claim.
- Support-matrix entries to update: none required (no new capability is
  executed or claimed); note in the Result section if the coordinator judges
  a matrix note appropriate.
- Deletion and documentation gates: none to delete. Update this task's own
  Result section; the coordinator updates the M5 milestone ledger
  (`docs/handovers/2026-09-22-m4-closure-to-m5.md`) on acceptance.
- Exact condition requiring owner direction or task rejection: if document
  04's text cannot express a single legal rule for MLA's shared-latent
  projection without contradicting its own GQA head-ownership text (i.e. the
  spec itself is ambiguous or incomplete for MLA partitioning), stop and
  report the specific ambiguity rather than guessing — that is an owner or
  ADR-level clarification, not a task-level inference. If expressing
  conditional KV replication requires a partition-rule shape so different
  from the existing four-variant enum that it would force every other
  operation's match arm to change meaning, stop and report before
  implementing — that is a design decision worth a second opinion, not a
  silent enum rewrite.

## Result, filled after work

- Changed shared owners and consumers; source commit:
- Commands and result IDs; passed / failed / skipped separately:
- Measured effect and uncertainty:
- Deleted/replaced paths:
- Remaining blockers and next bounded task:
