# Task 0054 — weight-shard byte-range addressing through the canonical format and residency authority

Status: **proposed**.

**Amendment, 2026-09-22 (coordinator, coordinator.md §3).** Old premise:
both `ColumnShardable` and `RowShardable` map to one `LogicalRange` per rank,
with disjoint ranges covering the tensor. Evidence (builder, round 1, before
any edit): canonical BF16 and affine tensors are row-major `[out, in]`.
An output-channel (`ColumnShardable`) shard is a contiguous run of rows. An
input-channel (`RowShardable`) shard takes a slice from every row, so it is
strided and one `LogicalRange` cannot express it. Replacement criterion:
`ColumnShardable` maps to one `LogicalRange` per component tensor (packed
codes, scales, zero points where present), and those ranges are disjoint and
cover each component. `RowShardable` returns a typed refusal that names the
strided-layout reason. A test proves that refusal. No implemented op produces
`RowShardable` today (`graph.rs` assigns `ColumnShardable`, `Replicated` or
`HeadShardable`), so no current consumer loses anything. Row-sharded weight
addressing becomes a named M5.2 obligation. Its options are a strided-range
primitive in `moxie-memory` or a TP-specific pre-sharded layout. The second
changes the canonical format, so it needs an ADR. Authority: coordinator.
The builder's round-1 evidence disproved the premise; the required behavior
and its oracle are otherwise unchanged.

**Second amendment, 2026-09-22 (coordinator).** Old premise: a quantized
shard needs a "group-aligned accepted / group-misaligned refused" pair.
Evidence (builder, before any edit): `AffineDescriptor::group_of(k)` and
`groups_per_row()` group the input `k` axis, while a `ColumnShardable` shard
splits whole output rows. A column boundary therefore cannot split a
quantization group, and every `RowShardable` request is refused before group
checking applies. The misaligned case cannot occur. Replacement criterion:
(1) a quantized `ColumnShardable` shard is accepted, and each component's
range (codes, scales, zero points) is derived from its own per-row width, is
disjoint from the other ranks' ranges, and covers the component; (2) an output
dimension that the rank count does not divide is refused; (3) `RowShardable`
is refused, for both BF16 and quantized tensors. Group-boundary checking moves
to M5.2 together with row-sharded addressing, where it can actually fail. The
second-consumer bullet and oracle (c) below are read under this amendment.
Authority: coordinator.

## Identity and authority

- Task0054, M5.1's second bounded slice (the first, task 0053, closed the
  op-contract half). Builder Codex `luna` (max, `/ponytail:ponytail`);
  independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `f7023bb` (task 0053 acceptance commit). Confirm `git status` clean
  save for the unrelated carried dirty work (`docs/evidence/specification-
  version.md`, `docs/decisions/adr/0034-tokenizers-crate-named-at-m8.md`) and
  `HEAD` unmoved before starting; preserve that dirty work, do not stage or
  revert it, do not include it in this task's commit.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Not expected to be needed.
- Requirement repaired: roadmap M5.1's second clause, "sharded weights and
  state use the canonical format and common residency authority," for
  weights. State (KV) shard addressing is a distinct, unopened obligation —
  see non-goals.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md),
  "Tensor parallelism (TP)" (lines 49–57) — "Check quantization-group
  boundaries and packed-layout alignment when sharding. Non-divisible
  dimensions use checked padding or explicit unsupported combinations."
  [Document 03](../spec/03-memory-formats-and-cuda.md)'s affine-integer
  descriptor and group semantics. [Document 02](../spec/02-architecture-and-common-api.md)'s
  crate ownership table — `moxie-format`/`moxie-storage` own the canonical
  manifest and chunk validation; `moxie-memory` owns the real resource
  authority; neither owns model-name or partition-specific branches.
- Required source reading: `crates/moxie-graph/src/lib.rs`'s `PartitionRule`
  (task 0053's `ColumnShardable`, `RowShardable`, `HeadShardable`, and the
  plain `Linear` rule already assigned in `crates/moxie-graph/src/graph.rs`);
  `crates/moxie-memory/src/residency.rs`'s `LogicalRange` ("a byte range
  **inside one tensor**, never inside a file" — this task's target output
  type) and `TensorSlot`/`ChunkId`/`ArtifactId`; `crates/moxie-format/src/affine.rs`'s
  `AffineDescriptor` (`groups_per_row()`, `group_of(k)`, `validate()`) —
  the existing group-boundary primitive this task must respect, not
  reinvent; `crates/moxie-executor/src/grouped.rs`'s existing
  `ChunkId::new(..., LogicalRange::new(0, len)?)` call sites — today every
  consumer requests the **whole** tensor as one chunk; this task is the
  first thing that ever requests less than that.
- Owner gates already resolved: O1–O5. O6/O7 remain open — no timing, no
  performance claim. No owner gate is reached or reopened; this is ordinary
  engineering within document 04's already-approved TP section.

## Bounded deliverable

- One concrete outcome: a pure function (no I/O, no residency admission)
  that, given a weight tensor's shape/dtype (BF16 or an `AffineDescriptor`
  for INT4/INT8), a `PartitionRule` (`ColumnShardable` or `RowShardable` —
  `HeadShardable`/`Replicated`/`NotDetermined` are out of scope, see
  non-goals), a rank count and a rank index, returns the `LogicalRange`
  that rank legally owns, or a typed refusal when the shard boundary is
  illegal: does not divide the sharded dimension evenly, or — for a
  quantized tensor — does not land on a quantization group boundary per
  `AffineDescriptor::group_of`. No silent padding or truncation, per
  document 04.
- Sole owning shared component: resolve by existing dependency direction,
  not assumption. `moxie-format` and `moxie-memory` do not currently depend
  on each other (checked: neither crate's `Cargo.toml` names the other).
  Do not introduce that edge if avoidable — `moxie-executor` already
  depends on both (see `grouped.rs`'s combined `AffineTensor`/`ChunkId`
  usage) and is a legal place for a function that consumes one crate's
  descriptor type and produces the other's range type. If, after reading
  the actual code, a new crate edge is genuinely the right shape, name and
  justify it explicitly (task 0038's precedent: an undeclared-but-legal
  dependency direction is an ordinary `arch-check` allowlist addition, not
  an owner decision) rather than silently picking a placement.
- Allowed production and test files/modules: the one crate identified above
  and its own test module; `crates/moxie-format/src/affine.rs` only to read
  `group_of`/`groups_per_row`, not to change their behavior;
  `crates/moxie-memory/src/residency.rs` only to read `LogicalRange`, not to
  change its constructor or invariants. Update `xtask/src/archcheck.rs`'s
  allowlist only if a new edge is genuinely needed, and say so in the
  Result section.
- Explicit non-goals and forbidden shortcuts: no residency admission,
  lease, chunk read or upload — this task computes an address, it does not
  fetch or place bytes. No rank-group type, no collective/NCCL plumbing, no
  TP2 execution (M5.2). No `HeadShardable` (attention/MLA) or `Replicated`
  shard addressing — those either need no range (replicated = whole tensor)
  or need the state-side (KV) equivalent this task does not cover; do not
  guess at either. No non-divisible-dimension padding or reshaping — reject,
  do not round. Do not touch `PartitionRule` itself or any op's
  `partition_rule()` — those are settled (tasks 0003, 0053).
- Existing consumers and second-consumer/shape proof: prove the function
  against at least one BF16 (ungrouped) shape and at least one quantized
  shape with a real `AffineDescriptor` (reuse an existing test fixture's
  descriptor rather than inventing a new one, per this repository's
  existing-fixture-reuse pattern), covering both a group-aligned shard
  boundary (accepted) and a group-misaligned one (refused).
- Temporary paths to delete or bridge expiry: none.

## Contract before implementation

- Equations and semantic input/output/state effects: none — this is address
  arithmetic, not a compute operation. No new equation, no state effect.
- Shapes, precision, accumulation/rounding, logical/physical layout: the
  function must reason in the tensor's **logical** column/row space and
  convert to byte offsets using its existing physical layout/dtype width —
  do not invent a new physical layout. Quantized code/scale/zero-point
  regions each have their own byte layout (see `AffineDescriptor`); decide
  explicitly whether this task addresses the packed-code region only or
  all regions a shard needs, and state that decision's evidence in the
  Result section rather than leaving it implicit.
- Partition and hardware capabilities: hardware-agnostic — legal for any
  rank count ≥ 1, including the existing single-rank (whole-tensor) case,
  which must still produce the same range as today's `LogicalRange::new(0,
  len)` call sites when rank count is 1.
- Peak memory and transfer dependencies; source/lease lifetime: none — no
  allocation, transfer or lease; the returned `LogicalRange` is not itself
  a lease.
- Cancellation, failure and rollback behavior: none — a pure, synchronous
  computation; illegal input returns a typed error, never a panic.
- Independent oracle; predeclared numerical metrics/thresholds: none
  numerical. Correctness is proven by: (a) rank-count 1 reproduces today's
  whole-tensor range exactly; (b) shard ranges for a given rank count are
  disjoint and their union is the whole tensor's range, for both the
  aligned and group-boundary-respecting cases; (c) a deliberately
  misaligned request is refused, not rounded.
- Application compatibility and sampler implications: none — no
  service/CLI-visible behavior changes; nothing calls this function yet.

## Acceptance

- Architecture / host / GPU / distributed / state tests and required
  hardware: host only (`cargo test --workspace`); `cargo xtask arch-check`
  and `cargo xtask spec-check` clean, including any new allowlist entry if
  one proved necessary. No GPU or distributed test required.
- Actual-context cases and prefill/decode/quality/memory gates: not
  applicable.
- Benchmark manifest, baseline identity, variance/repetition plan: not
  applicable — no performance claim.
- Support-matrix entries to update: none required (no new executed
  capability); note in the Result section if the coordinator judges a
  matrix note appropriate.
- Deletion and documentation gates: none to delete. Update this task's own
  Result section; the coordinator updates the M5 milestone ledger
  (`docs/handovers/2026-09-22-m4-closure-to-m5.md`) on acceptance.
- Exact condition requiring owner direction or task rejection: if the
  packed-code/scale/zero-point region question above cannot be resolved
  without changing `AffineDescriptor`'s existing public contract, stop and
  report rather than modifying a settled type. If no placement avoids a new
  crate edge that `arch-check` would need to treat as a genuinely new
  ownership direction (not just a missing declaration of an already-legal
  direction), stop and report before adding the edge — that is a second
  opinion worth having, not a silent allowlist expansion.

## Result, filled after work

- Changed shared owners and consumers; source commit:
- Commands and result IDs; passed / failed / skipped separately:
- Measured effect and uncertainty:
- Deleted/replaced paths:
- Remaining blockers and next bounded task:
