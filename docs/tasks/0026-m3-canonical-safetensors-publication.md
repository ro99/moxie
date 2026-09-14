# Task 0026 — M3 item 1: publish canonical artifacts as safetensors shards

Status: **active**; contract written before implementation.

## Identity and authority

- Task ID 0026; **M3 item 1**, the bounded follow-up
  [ADR 0022's 2026-09-14 amendment](../decisions/adr/0022-user-programs-and-canonical-write-authority.md)
  requires. Reviewer/acceptance: repository owner.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `a5929da`
  with task 0025's review corrections in place.
- Legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, read-only. `/models` and
  `/fast/models` remain read-only inputs.
- The schema is fixed **before this code** by
  [ADR 0025](../decisions/adr/0025-canonical-safetensors-schema.md): component
  naming, physical shapes, shard mapping, checksum scope and migration.
- **This is an addendum, not a replacement.** Task 0025's correctness
  corrections — confined writes, shared entry validation, source re-hashing,
  torn-journal repair, admitted metadata, the complete disk plan, failure
  cleanup, scratch refusal, cancellation and parent-directory durability —
  stay, and every one of them applies to the new writer.

## Bounded deliverable

`moxie-repack` publishes a **manifest-v2 directory**: one or more conforming
`.safetensors` shards plus `manifest.toml`. The manifest-v1 raw-chunk **writer**
is deleted; the v1 **reader** stays, refusing nothing it accepted before.

Allowed: `moxie-format` (manifest v2, safetensors header construction, the
canonical component mapping), `moxie-storage` (opening and verifying a v2
artifact), `moxie-repack` (planning components, writing shards, reporting),
their tests, and the records. No model graph, CUDA, sampler, cache or kernel
change. No new third-party dependency.

Not in scope: Hub upload, bulk conversion, a tokenizer, AutoGPTQ/AutoRound
packing, activation-order interpretation, catalog integration, and any claim
that a published artifact is loadable by Transformers or executable here.

## Contract before implementation

- **The mathematics does not move.** `W = (Q - Z) * S`, the INT4 nibble order,
  row byte-alignment, grouping, the source's own scale dtype and the i16
  zero-point convention are exactly as document 03 and ADR 0023 state them. The
  bytes that used to be three sections of one range become three tensors; no
  byte changes value.
- Physical shapes, dtypes, names, shard mapping, alignment and checksum scope
  are ADR 0025's, not this task's to reinterpret.
- Bounded throughout: header and payload access stay within the admitted
  budgets, and the shard-size budget is explicit like every other.
- Source identity, restart, cancellation, confined writes and durable
  publication behave exactly as task 0025's corrected run does. The private
  journal never becomes part of the published artifact.
- Checksum and numerical validation remain Moxie's: safetensors permits NaN and
  Inf and enforces none of document 03's scale rules.

## Acceptance

1. `cargo fmt --all -- --check`, host clippy with warnings denied, both CUDA
   clippy lanes, `cargo xtask spec-check`, `cargo xtask arch-check`, and the
   workspace host and device-feature suites.
2. Every published shard opens with the **reference safetensors
   implementation**, and every tensor's dtype, shape and bytes match what
   Moxie's reader returns. Recorded as evidence with the reader's version.
3. Every canonical component compared against an independent source oracle, as
   task 0025 does for values: all 3,145,728 values of the real Laguna module,
   bitwise against the canonical FP32 equation and against the source's own
   BF16 boundary, from the **reopened** artifact.
4. Tiny fixtures cover both widths, symmetric and asymmetric, every scale dtype,
   group tails, mixed BF16, several shards, and a component landing at a shard
   boundary — each against bytes the test builds itself.
5. Task 0025's whole publication enumeration still passes against the new
   writer: every named durable boundary failed in turn, no readable artifact
   before the rename, a resumable destination after, byte-identical republish.
6. The mutation battery is re-run and reported.
7. Records updated: this result, the handover, the support matrix, the README
   and the engineering log. The v1 reader's transitional expiry is stated.

Stop and report if the reference reader disagrees with ours about any byte,
dtype or shape; if a bounded budget cannot be honoured; or if honouring the
schema would require regrouping, requantizing or rounding anything.

## Result, filled after work

Not yet filled.
