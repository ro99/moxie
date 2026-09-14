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

Status: **implemented; not reviewed and not accepted.**

### What changed

| Component | What it owns now |
|---|---|
| `moxie-format::safetensors` | A **writer** beside the reader: it builds a shard's header and fixes every offset before a byte is written, and re-parses its own output before returning it |
| `moxie-format::canonical` | The one place that says which physical tensors a logical one becomes |
| `moxie-format::manifest` | Version 2 rows carry components; version 1 rows keep chunk ranges, one validator, two versions |
| `moxie-storage::Artifact` | Opens either shape. Version 2 opens each shard through the existing `Shard` reader and verifies each component against **its own** checksum |
| `moxie-repack::write::plan` | Groups components into shards, names them `model-NNNNN-of-NNNNN.safetensors`, and refuses a component larger than the admitted shard size rather than splitting it |

### The measurement that changed the schema

ADR 0025 first aligned each component's payload to eight bytes, for a future
mapping consumer. **The reference implementation refuses a gap**, measured with
`safetensors` 0.7.0 before any shard was written:

```text
gap=False: LOADED
gap=True:  SafetensorError: Error while deserializing header:
           invalid offset for tensor `w.scales`
```

Reading the reference source afterwards confirmed the rule in
`Metadata::validate` -- `s != start` -- along with two more this writer already
satisfied: the declared length must equal shape times dtype size, and the file
must end exactly at the last tensor (`buffer_end + N_LEN + n != buffer_len`).
`validate()` sorts by offset first, so header key order is free. Payloads are
contiguous now and the ADR carries the measurement.

### Gates

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | **passed** |
| Host clippy, warnings denied | **passed** |
| Device-lane clippy (`moxie-executor/driver`) | **passed** |
| `xtask` device clippy (`--features cuda`) | **passed** |
| `cargo xtask spec-check` | **passed**, 10 documents |
| `cargo xtask arch-check` | **passed**: 79 rejected fixtures, 21 accepted, 13 rules |
| Host tests | **1,028 passed, 0 failed, 0 ignored, 0 skipped**, 92 suites |
| Device-feature tests | **1,063 passed, 0 failed, 0 ignored**, 92 suites |
| Reference reader | every shard accepted by `safetensors` 0.7.0; every component's dtype, shape and SHA-256 matched the manifest ([driver](../evidence/experiments/drivers/0007-reference-reader.py)) |
| Mutation battery | **30 of 30 caught, 3 of 3 controls held, 0 survivors, 0 unstable, 0 skipped** |

**Nothing failed and nothing was skipped.** The real-artifact lane ran rather
than skipping -- 63.5 s of it -- and prints `SKIP` with a reason when its source
is absent.

**This table was wrong when it was first written, and the correction is the
point of keeping it.** It claimed 1,097 host tests, which is task 0025's figure
copied rather than measured: the safetensors rewrite deletes the v1 writer and
its chunk-specific tests, and this tree has 1,028. Worse, the suite was not
green at all. The mutation battery's new `workflow` lane refused to agree with
itself on the clean tree, which is how
`every_injected_failure_leaves_the_ledger_empty` was found failing at `792743a`
-- for a product reason, described next.

### Review finding 7, reintroduced by the header pass

`Run::begin` admits the read-back scratch, then the safetensors writer's new
pass writes and syncs every shard header. That pass returned with a bare `?`:

```rust
run.write_shard_headers(faults)?;     // leaks the admission
```

`begin` never hands back a `Run` it failed to build, so nothing above it could
release the charge on its behalf, and `repack`'s own cleanup only knows about
the buffers it owns. Failing at that boundary left 65,536 admitted bytes
reserved for the life of the process -- exactly the defect the review's finding
7 named, at an error path that did not exist when finding 7 was fixed. It now
goes through `run.abandon(ledger)`, the same release the journal handle beside
it has always used.

The gate that catches this -- `every_injected_failure_leaves_the_ledger_empty`,
which fails a run at every named boundary in turn and asks who still holds an
admission -- was already written, already correct, and had been failing since
`d719f67`. Nothing was wrong with the test. What was wrong was a gate table that
said the suite was green without the suite having been run, which is why the
correction above matters more than the leak does.

The same enumeration is also why the boundary was reachable at all, and that is
the other half of the story:

### Three boundaries that were hiding behind three others

The header pass reused `chunk-create`, `chunk-write` and `chunk-sync`, because
they are the same three syscalls. Every fault injected for a payload unit
therefore fired at a header first, and two mutations that had been caught --
`chunk-sync-skipped` and `write-errors-are-swallowed` -- survived, because the
suites kept failing for a reason unrelated to the code under test. The header
pass now owns `shard-create`, `shard-header-write` and `shard-header-sync`, and
`write_at` takes the boundary it writes at as a parameter. `Site::ALL` is 20
boundaries; the enumeration and the ledger walk both picked up the new cases
with no edit.

A third mutation, `cancellation-not-checked-before-publish`, survived for its
own reason: task 0025's finding-9 correction added a cancellation check *inside*
the validation loop, and the existing test's "cancel from the second question
onwards" closure stopped there without ever reaching the last gate before the
rename. `cancellation_at_the_final_gate_still_stops_before_the_rename` answers
*no* to every boundary but the last and asserts how many boundaries there are,
so a change in that count fails the test instead of silently moving where it
cancels.

### The real module, read by the reference implementation

`model.layers.1.mlp.experts.0.down_proj` of `Laguna-S-2.1-AWQ-INT4` revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`, published to a temporary directory
and opened by `safetensors.deserialize`:

| Component | dtype | shape | bytes | checksum |
|---|---|---|---:|---|
| `….weight.codes` | `U8` | `[3072, 512]` | 1,572,864 | matches the manifest |
| `….weight.scales` | `BF16` | `[3072, 32]` | 196,608 | matches the manifest |
| `….weight.zero_points` | `I16` | `[3072, 32]` | 196,608 | matches the manifest |

The shard is 1,966,504 bytes: 1,966,080 of canonical payload -- exactly what the
raw-chunk artifact held -- plus a 424-byte header. Reconstructing all
**3,145,728** values **from the reference reader's own bytes** gives finite
weights, codes spanning the whole INT4 range and zero points spanning
`[-6, 5]`, the same figures task 0025 measured. The in-repository lane
separately compares every value bitwise against the canonical FP32 equation and
against the source's own BF16 boundary.

Whole run: ~61 s, nearly all of it hashing the 5.37 GB source shard **twice** --
once at the start and once before publication, which is task 0025's
source-identity correction and is what it costs.

### The enumeration, and the mutation battery

The publication enumeration is **38 cases across 38 boundary/visit pairs**, up
from 35: the three boundaries the header pass stopped sharing. Measured rather
than assumed -- the test counts each boundary's visits in a clean run and
enumerates that many cases, so a boundary that gains a visit gains cases:

```text
destination-create 1   journal-append 6   shard-create       1   chunk-create 4
lock-acquire       1   journal-sync   6   shard-header-write 1   chunk-write  4
journal-create     1   allocate       1   shard-header-sync  1   chunk-sync   4
manifest-write 1  manifest-sync 1  validate 1  publish 1  publish-durability 1
directory-sync 1  journal-remove 1        chunk-read-back 0 (not reached)
```

Every case: the injected failure fired, no manifest existed before the rename,
the destination resumed, and the republished artifact matched the reference
byte for byte. Three cases failed at or after the publication boundary and
published anyway, which is the documented outcome there.

The battery ([experiment 0006](../evidence/experiments/0006-repack-publication-mutations.md))
was re-run against the safetensors writer: **30 of 30 mutants caught, 3 of 3
expected survivors held, 0 survivors, 0 unstable, 0 invalid controls, 0 broken
controls, 0 skipped**, three repetitions of every verdict in both directions.
Getting there took re-anchoring nine mutations the rewrite had moved, adding
the `workflow` lane, and the two product corrections above. The experiment
record carries what each attempt said.

### Limits

- **Nothing executes a canonical tensor.** A safetensors file is not an
  executable model and this changes nothing about M3 item 3.
- **Not loadable by Transformers**, and no claim that it is: the manifest
  defines what these tensors mean, and no `config.json`, tokenizer or
  index file is written.
- **No Hub upload, no bulk conversion, no source mutation.** One module of one
  artifact was read; nothing under a checkpoint root was written.
- Manifest v1 keeps its reader and its meaning. The transitional path expires
  at M11 item 4, or earlier if a task establishes no v1 artifact exists outside
  tests.
