# Handover — task 0026: canonical artifacts are safetensors shards

**Implemented and not reviewed.** M3 item 1's publication now writes what the
owner's 2026-09-14 amendment to
[ADR 0022](../decisions/adr/0022-user-programs-and-canonical-write-authority.md)
requires: a directory holding a TOML manifest and one or more conforming
`.safetensors` shards. The custom container is gone and M11 item 4 no longer
selects one.

**This is an addendum to task 0025, not a replacement.** Every correction that
task's independent review produced — confined writes, shared entry validation,
source re-hashing before publication, torn-journal repair, admitted metadata,
the complete disk plan, failure cleanup, the scratch refusal, cancellation in
long reads, parent-directory durability — is in place and applies to the shard
writer. Its publication enumeration runs against it unchanged in intent, over a
boundary set that safetensors grew from 17 named durable boundaries to 20.

## Workspace identity

- `/home/rodrigo/Developer/moxie`, branch `main`, built on `a5929da` (task
  0025's corrections) through `792743a`.
- [ADR 0025](../decisions/adr/0025-canonical-safetensors-schema.md) and the
  [task 0026 contract](../tasks/0026-m3-canonical-safetensors-publication.md)
  were committed at `e769780`, **before** any implementation commit, because
  the amendment requires the schema fixed first.
- `/models` and `/fast/models` remain read-only inputs. One shard of
  `Laguna-S-2.1-AWQ-INT4` was read; **nothing under either root was written**.

## Completed facts

- The reference implementation opens what we publish. `safetensors` 0.7.0
  accepted the real module's 1,966,504-byte shard: `U8 [3072, 512]` codes,
  `BF16 [3072, 32]` scales, `I16 [3072, 32]` zero points, every component's
  SHA-256 matching the manifest. Reconstructing from **its** bytes gives
  3,145,728 finite values, codes spanning the whole INT4 range, zero points
  spanning `[-6, 5]` — the figures task 0025 measured from raw chunks.
- All eight gates pass: fmt, three clippy lanes, spec-check, arch-check (79
  rejected fixtures, 21 accepted, 13 rules), **1,028** host tests across 92
  suites and the device-feature lane, with nothing failed, ignored or skipped.
  The real-artifact lane ran rather than skipping.
- **Two defects were found after the implementation commits, by re-running the
  mutation battery, and both are fixed.** The safetensors header pass leaked an
  admitted reservation when it failed (review finding 7, at an error path that
  postdates finding 7's fix), and it had absorbed the three payload fault
  boundaries into itself, which made two previously-caught mutations survive.
  The header pass now owns `shard-create`, `shard-header-write` and
  `shard-header-sync`. The task record describes both.
- An earlier draft of the task record claimed 1,097 host tests with nothing
  failing. That was task 0025's figure copied rather than measured, and the
  suite was not green: `every_injected_failure_leaves_the_ledger_empty` failed
  at `792743a`. Both are corrected there.

- The mutation battery was re-run on the final tree: **30 of 30 mutants caught,
  3 of 3 expected survivors held, 0 survivors, 0 unstable, 0 invalid controls,
  0 broken controls, 0 skipped**, three repetitions of each verdict in both
  directions. The publication enumeration is 38 cases across 20 named
  boundaries.
- The real module was republished on the final tree and re-checked: a
  1,966,504-byte shard, accepted by `safetensors` 0.7.0, three components whose
  checksums match the manifest, artifact identity
  `8285637189fdf76c2f8d9dae7b9ff721a80a349cb9e2301794c04dcc55ada29b`.

## Decisions

- **[ADR 0025](../decisions/adr/0025-canonical-safetensors-schema.md)** fixes
  the schema the amendment asked for: component naming, physical shapes, shard
  mapping, checksum scope (per component payload, not per file) and migration.
- **The reference implementation refuses a gap between tensors**, measured
  before a shard was written. ADR 0025 originally aligned each component to
  eight bytes; that would have produced files the reference reader rejects.
  Payloads are contiguous and the ADR records the measurement.
- **`__metadata__` is a courtesy, not an authority.** The manifest is the single
  source of truth; no reader here depends on the shard's metadata map.
- **Manifest v1 keeps its reader and its meaning.** The v1 writer is deleted,
  nothing is converted, and the transitional read path expires at M11 item 4 or
  earlier if a task shows no v1 artifact exists outside tests.

## Remaining hypotheses and blockers

- **Not reviewed.** Task 0025 needed ten corrections after one review; this
  changed the container underneath all of them.
- **A shard is not a model.** No `config.json`, tokenizer or `model.safetensors.index.json`
  is written, and nothing claims Transformers can load the result. Whether to
  write those is a catalog/packaging question nobody has been assigned.
- **Nothing executes a canonical tensor** — M3 item 3, untouched, still the
  largest gap in the milestone.
- **The whole-file source digest is computed twice per run**, once at the start
  and once before publication. On the 5.37 GB Laguna shard that is most of the
  61-second wall clock. It is the honest cost of task 0025's finding 3, and it
  is the obvious thing to make cheaper if a run ever spans a whole model.
- **Shard-size policy is a budget, not a convention.** `--chunk-file-bytes` is
  the shard limit; nothing yet matches the ecosystem's usual few-gigabyte
  shards or writes an index file.

## Next task

**M3 item 3 — shared W4A16/W8A16 dense and expert paths**, unchanged from task
0025's handover, and now with a canonical artifact in a format the ecosystem
reads to execute from. Document 03 bounds it before it starts: weight-only
paths, BF16 preferred with FP32 accumulation, SM86 first and SM120 qualified
separately, and bounded reference dequantization is not an acceptable final
fast path by assertion.
