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
- All eight gates pass on this tree: fmt, three clippy lanes, spec-check,
  arch-check (79 rejected fixtures, 21 accepted, 13 rules), **1,045** host tests
  across 95 suites and **1,080** device-feature tests across 95, with nothing
  failed, ignored or skipped. The real-artifact lane ran rather than skipping.
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

- **A second independent review found fourteen more, six of them P1, and all
  fourteen are corrected** with a regression each. The pattern across them: a
  claim was compared against a copy of itself. A component was checked to exist
  but not to be the dtype, shape and length its descriptor implies; a component
  list was compared as a set while it was streamed as a sequence; a source was
  hashed by reopening a path while it was read through a cached handle; a
  memory bound counted serialized text while the run held parsed structures; a
  shard was held to our parser while the ruling names the reference
  implementation. The task record carries the table.
- Two of those corrections needed a second attempt, and my own tests caught
  both: binding the source to one handle made the run self-consistent while
  hiding the replacement it was supposed to reject, and the final-gate
  cancellation test went self-fulfilling once validation began asking per slice.
- **The mutation batteries are `cargo xtask mutation-check` now**, not Python
  under `docs/`. The port's self-test checks every anchor, which immediately
  found that thirteen of experiment 0005's twenty-four had matched nothing since
  before `ccfd7fa`; that battery was retired rather than carried. The restore is
  a marker file written before each substitution, so a run killed by `SIGKILL`
  leaves nothing behind -- which had already happened once here, leaving
  `if false && got != unit.sha256` in the writer while gates ran against it.

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

## Status correction, 2026-09-14: this work is provisional

**Offline repacking is provisional** ([ADR 0027](../decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)).
The owner's requirement is that it brings **worthwhile inference performance
improvements**. No such measurement exists, and none is runnable until shared
execution does.

Nothing in this handover — the schema, the reference-reader conformance, the
byte guarantees, the enumeration — is evidence about inference speed. All of it
is evidence about bytes and structure. Retention of the step is decided by
[experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md),
whose criterion is predeclared and whose outcomes are fixed in advance.

The correctness obligations here are **unchanged** by that status: provisional
is not a waiver, and the outstanding review corrections are part of the bounded
scope in [task 0027](../tasks/0027-m3-generated-plans-and-automatic-budgets.md).

## What is not measured

**The mutation battery has not been run against the second review's
corrections.** Started, stopped after 7 of 50; a full run is about four hours.
The sixteen substitutions covering the round-2 protections have not been shown
to be caught by any lane. Each has a regression that was watched to fail before
its fix went in — weaker evidence than the battery, and the evidence that
exists. `cargo xtask mutation-check` runs it; `--self-test` verifies the verdict
rule, the selector and all 50 anchors in seconds.

The 30-of-30 battery result in the records describes `ccfd7fa`. The eight gates
below describe this tree.

## Remaining hypotheses and blockers

- **Reviewed twice, not accepted.** Ten findings in the first round and
  fourteen in the second, all corrected. The second round's shape -- six P1
  findings about validation comparing a claim against itself -- is the reason
  to expect a third round rather than to assume none.
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

**M3 shared W4A16/W8A16 execution**, after task 0027's bounded CLI scope is
validated and handed over — a sequencing recommendation, not an owner
requirement, and not a licence to close M3 or begin unlimited runtime work. It
is also what makes [experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md)
runnable, which is what decides whether the work in this handover is retained at
all. Document 03 bounds it before it starts: weight-only
paths, BF16 preferred with FP32 accumulation, SM86 first and SM120 qualified
separately, and bounded reference dequantization is not an acceptable final
fast path by assertion.
