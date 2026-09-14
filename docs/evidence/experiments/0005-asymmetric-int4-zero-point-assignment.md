# 0005 — Which output channel does a packed zero point belong to? Measured, and the importer's tests measured by mutation

Date: 2026-09-13. Milestone: M3 item 2,
[task 0024](../../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md).
Status: **corrected after two independent reviews, awaiting a third.** The
pinned assignment is separated from every alternative the same bytes permit —
including the **sign**, which the first version of this record wrongly claimed
was not separable — on four tensors across two artifacts. The mutation battery
is measured again, with the driver committed and every verdict repeated.

## Why this exists

[Task 0018](../../tasks/0018-m3-compressed-tensors-int8-importer.md) recorded a
thing it could not close:

> All `values_per_word` lanes of a word fall inside one scale group, so the
> artifact's own data cannot distinguish the lane order: reversing it
> reconstructs a different but equally plausible weight.

Task 0024 imports **asymmetric** sources, whose zero points are packed along a
different axis — the **output** axis — and that changes what the artifact can
say. A zero-point word's eight lanes are eight different **output channels**,
whose codes have different statistics. So the lane assignment is a measurable
property of the file rather than a convention taken on a library's word.

It matters that it is measurable, because the library is not the artifacts'
own. The reading is pinned to `compressed-tensors` **0.17.0**, resolved locally;
the four asymmetric artifacts on this machine declare compressor versions
`0.1.dev534+gb269f2e` and `0.1.dev535+gdc9611a`, which are untagged development
builds and pin nothing. The repository's standing rule — from the yarn RoPE gap,
where one mismatched `transformers` copy was refused as a pinned exporter — is
that a single mismatched copy is not authority. This measurement is what makes
the reading evidence rather than an assumption.

**It is corroboration of a reading, not a quality claim.** O2 is repack-only and
needs paired output against the released model; nothing here executes anything.

## The candidates

`weight_zero_point` is `I32[ceil(out/8), groups]` — a shape only reachable by
packing eight 4-bit values along the output axis. What the shape does **not**
fix is which output channel each lane holds. Four readings are consistent with
it:

| Name | Assignment |
|---|---|
| **pinned** | `o = 8j + l` — word row `j = o / 8`, lane `l = o % 8` |
| lane-reversed | `o = 8j + (7 - l)` |
| block-major | `o = j + rows * l` — the transpose the packer performs, not undone |
| block reversed | `o = j + rows * (7 - l)` |

The pinned one is what `pack_to_int32(..., packed_dim=0)` and
`unpack_from_int32(..., packed_dim=0)` do, step by step: transpose, pack along
what is then the column axis, transpose back; and
`unpacked[l::pack_factor, :] = value >> (bits * l)` to invert it.

## The statistic, declared before it was run

A group's codes are `q = round(w/s) + z` clipped to `[-8, 7]`, and a weight
group is roughly centred on zero, so `mean(q) ≈ z`. Under the correct pairing
`mean(q) - z` is therefore tight; under a wrong one it is two independent
quantities subtracted, and its spread is the quadrature sum of both.

The statistic is `mean |mean_code(o,g) - z(o,g)|`, in codes. The thresholds were
written into [the task contract](../../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md)
**before** the measurement: the pinned assignment below **1.0**, every
alternative above **1.2**. The sign candidate was added after a review and is
held to the same thresholds; it was not chosen to fit the number it produced. The test also requires every alternative to be at
least twice the pinned value, so a shrinking margin fails rather than degrading
quietly.

`mean_code` is built from `weight_packed` alone. No zero point enters it, so the
two sides of the comparison do not share a source.

## Result

Measured by
`crates/moxie-storage/tests/asymmetric_int4_import.rs::the_pinned_zero_point_lane_assignment_is_the_one_the_artifacts_bytes_support`,
on the two smallest complete modules of each artifact.

| Artifact | Module | pinned | lane-reversed | block-major | block reversed | sign-flipped |
|---|---|---:|---:|---:|---:|---:|
| Laguna-S-2.1-AWQ-INT4 | `layers.1.mlp.experts.0.down_proj` `[3072, 1024]` | **0.5236** | 1.4900 | 1.4853 | 1.4817 | 2.1797 |
| Laguna-S-2.1-AWQ-INT4 | `layers.1.mlp.experts.0.gate_proj` `[1024, 3072]` | **0.5183** | 1.5661 | 1.5652 | 1.5670 | 2.2912 |
| Qwen3.8-27B-AWQ-BF16-INT4 | `layers.11.self_attn.k_proj` `[1024, 5120]` | **0.5163** | 1.7629 | 1.7809 | 1.7839 | 2.5735 |
| Qwen3.8-27B-AWQ-BF16-INT4 | `layers.11.self_attn.v_proj` `[1024, 5120]` | **0.5300** | 1.5751 | 1.5741 | 1.5776 | 2.2888 |

The pinned reading is **2.8 to 4.9 times** tighter than every alternative, on
every tensor, on two artifacts from two different compressor builds. The three
misassignments land on each other, which is what "independent" looks like and is
itself a check that the statistic is measuring the pairing rather than something
about the codes; the sign lands further out still, for the reason below.

**A margin of 0.52 against 1.48 is corroboration, not proof.** What it rules out
is the three specific alternatives the same bytes permit. It does not establish
the *code* lane order, which remains exactly where task 0018 left it, and it is
not evidence about model output.

### The sign, and a claim this record got wrong

The first version of this record said the zero-point **sign** was not separable
by this statistic, because "negating `z` changes `mean(q) - z` into
`mean(q) + z`, which for a distribution centred near zero is indistinguishable
in mean absolute deviation."

**That is false, and it is false for a reason stated two paragraphs earlier in
this same document.** The argument holds only for quantities that are
independent. `mean(q)` and `z` are *correlated* — that correlation is the entire
premise of the measurement above. An independent review measured the difference
on the Laguna down-projection and it is larger than any misassignment's:
**0.523622194925944** for subtraction against **2.1797049840291343** for
addition. This workspace's own test now prints the same figures.

A record contradicting a fact it already contains is the third instance of that
class here; the previous two were arithmetic and this one is an argument, which
is if anything easier to wave through. The sign is a fifth candidate in the
table above now, measured like the rest.

### What is still not measured

The **code** word's lane order, exactly as task 0018 left it: all eight lanes of
a packed code word fall inside one scale group, so no statistic over this
artifact separates them. Closing that needs paired output against the released
model, which is O2's.

## The mutation battery: 24 of 24 caught, 0 survivors

The method is experiments 0002–0004's: one edit that changes behaviour and
still compiles; apply, run every lane, record which caught it, revert. Three
things are different here, and every one of them came from a review.

**The driver was committed** — `tools/experiments/0005-mutations.py`
— with every substitution verbatim, because the names below are not the
measurement.

**Every verdict is repeated three times in both directions**: the mutant against
the mutated tree and the control against the restored one, on the first lane
that catches.

**And a verdict that is not valid is not counted.** The first version appended
unstable repeats and failing restored controls to a `nondet` list *and then
counted them as caught*; the second review drove the driver's own `main` with a
control that never passed and it printed "1 of 1 caught, 0 survivors". The
verdict is a pure function now — `classify(caught_by, mutant_ok, control_ok)`
→ `caught` / `survivor` / `unstable` / `invalid-control` — only `caught` counts,
skips are named, and **the process exits nonzero unless every mutation is
caught**. `--self-test` checks the rule over eight cases, including both of the
review's, because the rule is what the headline number means.

| Mutation | Verdict | Caught by |
|---|---|---|
| `zp-sign-added-not-subtracted` | caught | unit, artifact |
| `zp-lane-pinned-to-zero` | caught | unit, artifact |
| `zp-lane-reversed` | caught | unit, artifact |
| `zp-word-row-is-block-major` | caught | unit, artifact |
| `zp-rebias-dropped` | caught | unit, artifact |
| `zero-points-dropped-entirely` | caught | unit, alloccount, artifact |
| `zp-reads-the-padding-lanes` | caught | unit, allocfail |
| `zp-shape-check-deleted` | caught | unit, allocfail |
| `zp-length-check-deleted` | caught | unit, allocfail |
| `spec-payload-disagreement-ignored` | caught | unit, allocfail |
| `spec-payload-missing-ignored` | caught | unit, allocfail |
| `index-disagreement-symmetric-ignored` | caught | unit, allocfail, artifact |
| `index-missing-zero-point-ignored` | caught | unit, allocfail |
| `zero-point-dtype-unchecked` | caught | unit |
| `zp-vec-allocated-infallibly` | caught | **allocfail only** |
| `refusal-prose-allocates-infallibly` | caught | **allocfail only** |
| `bf16-rounding-truncates` | caught | unit |
| `boundary-count-never-increments` | caught | **artifact only** |
| `missing-companions-always-empty` | caught | **artifact only** |
| `measurement-pinned-becomes-lane-reversed` | caught | **artifact only** |
| `source-entries-names-allocate-infallibly` | caught | **allocfail only** |
| `source-entries-zero-point-name-allocates-infallibly` | caught | **allocfail only** |
| `inventory-filters-incomplete-modules` | caught | **artifact only** |
| `measurement-sign-candidate-equals-pinned` | caught | **artifact only** |

**24 of 24 caught, 0 survivors, 0 unstable, 0 invalid controls, 0 skipped.**

Lanes: `unit` is `cargo test -p moxie-format --lib`; `allocfail` is
`--test import_allocation_failure`; `alloccount` is
`--test import_allocation_asymmetric`; `artifact` is
`-p moxie-storage --test asymmetric_int4_import`; `symmetric` is
`--test gemma4_import`, which caught none — correctly, since it is the
regression lane for the path this task did not change.

### The eight the reviews forced, and what each one is for

The first battery was 16 of 16 and it was measuring a suite with five holes in
it. **Eight mutations were added after a review** — five after the first, three
after the second — taking the battery from 16 to 24. **Every one of the eight is
caught by exactly one lane**, and seven of the eight by a check that a review's
finding added. The eighth, `bf16-rounding-truncates`, is caught by
`moxie-format`'s **pre-existing** BF16 rounding tests: an existing lane
acquiring a new regression is not a lane a review created, and the third review
was right to separate the two.

Eleven of the twenty-four are caught by exactly one lane; the eight added after
a review are all of them plus three that were single-lane from the start.

| Mutation | The hole it stands in for |
|---|---|
| `refusal-prose-allocates-infallibly` | the first sweep only imported **valid** inputs, so no refusal was ever constructed |
| `source-entries-names-allocate-infallibly` | …and the second sweep only called `import`, so a second public entry point's four `format!` names were never reached |
| `source-entries-zero-point-name-allocates-infallibly` | the same, on the name task 0024 itself added |
| `inventory-filters-incomplete-modules` | **the original defect**, which the first regression could not detect because it tested the helper the filter ran before |
| `missing-companions-always-empty` | the audit's decision, over a case no real artifact supplies |
| `boundary-count-never-increments` | a rounding-boundary check on a sample where the boundary never fires |
| `bf16-rounding-truncates` | the rounding the source-arithmetic comparison depends on |
| `measurement-sign-candidate-equals-pinned` | the sign candidate, added after the impossibility claim turned out to be false |

**A battery that is complete against the suite it was written for says nothing
about the suite's holes.** 16 of 16 (first run) and 21 of 21 (after the first
review) were both true when measured and both described a suite a reviewer then
walked straight through. Only a finding from outside can add the case; what the
battery does is stop it coming back.

## Two facts this measurement's plumbing turned up

**A module's four tensors need not live in one shard.** Laguna keeps them
together for 34,739 of its 34,740 asymmetric modules; Qwen3.8-27B keeps them
together for **none** of its 256 — codes and zero points in shards 1–2, every
scale elsewhere. `source_entries` takes one `Header` by design and cannot see
across that split, so a caller with a sharded artifact needs an index resolver.
That is M3 item 1's manifest work; this task records it rather than building it,
and its own artifact lane resolves each tensor to its own shard.

**Laguna's shard headers exceed the default header budget.** Its index holds
140,989 tensors, so one shard's header is about a megabyte serialized and its
admitted peak estimate is 16.5 MB, against `HeaderBudget::DEFAULT`'s 8 MiB — a
default whose own docstring is calibrated on Gemma 4's 17–64 KB headers. The
default refusing it is the budget working as designed. The artifact lane states
a 64 MiB budget for a read-only inspection; **the default is unchanged**, and
any production path that opens this artifact will have to state one too.

## Reproduction

```text
cargo test -p moxie-storage --locked --offline --test asymmetric_int4_import -- --nocapture
cargo test -p moxie-format  --locked --offline --lib
cargo test -p moxie-format  --locked --offline --test import_allocation_failure -- --nocapture
cargo test -p moxie-format  --locked --offline --test import_allocation_asymmetric -- --nocapture
```

The mutation driver is committed, with every substitution verbatim:

```text
python3 tools/experiments/0005-mutations.py   # as it was run, at task 0024
```

The first version of this record said it was "reproduced in the task record's
result section". It was not, in either commit, and an independent review called
that what it is — a reproducibility gap. **The mutation names are not the
measurement; the exact substitutions are.** It edits tracked source in place and
restores it in a `finally`, so a clean `git status` afterwards is part of the
evidence.

Artifacts read, read-only, nothing written (**O5**):
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`, and
`/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4`. Pinned library:
`compressed-tensors` 0.17.0 at
`/home/rodrigo/.cache/uv/archive-v0/uqs9z2Tvizx6-8cq0I0rd/compressed_tensors`,
file hashes in
[the task contract](../../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md#the-serialization).

## The driver was retired on 2026-09-14

**This record's numbers stand; its driver no longer runs.** When the mutation
batteries moved into `cargo xtask mutation-check`, the port's self-test checks
that every anchor still matches exactly once — and found that **thirteen of
this battery's twenty-four** matched nothing, already, at `ccfd7fa`. Tasks 0025
and 0026 rewrote the importer paths these substitutions were anchored to, and
nothing noticed because nothing re-ran them.

A mutation that does not apply is not evidence, which is this battery's own
rule, so the driver was removed rather than carried as a number nobody can
reproduce. What it measured is above, and it was measured against the tree at
task 0024. Re-anchoring it to today's code would be a different measurement of
different code, and that is a task's decision rather than a rename.
