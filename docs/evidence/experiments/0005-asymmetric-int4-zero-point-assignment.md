# 0005 — Which output channel does a packed zero point belong to? Measured, and the importer's tests measured by mutation

Date: 2026-09-13. Milestone: M3 item 2,
[task 0024](../../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md).
Status: **accepted**. The pinned assignment is separated from every alternative
the same bytes permit, on four tensors across two artifacts; and the tests
written for it caught **16 of 16** mutations, 0 survivors.

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
alternative above **1.2**. The test also requires every alternative to be at
least twice the pinned value, so a shrinking margin fails rather than degrading
quietly.

`mean_code` is built from `weight_packed` alone. No zero point enters it, so the
two sides of the comparison do not share a source.

## Result

Measured by
`crates/moxie-storage/tests/asymmetric_int4_import.rs::the_pinned_zero_point_lane_assignment_is_the_one_the_artifacts_bytes_support`,
on the two smallest complete modules of each artifact.

| Artifact | Module | pinned | lane-reversed | block-major | block reversed |
|---|---|---:|---:|---:|---:|
| Laguna-S-2.1-AWQ-INT4 | `layers.1.mlp.experts.0.down_proj` `[3072, 1024]` | **0.5236** | 1.4900 | 1.4853 | 1.4817 |
| Laguna-S-2.1-AWQ-INT4 | `layers.1.mlp.experts.0.gate_proj` `[1024, 3072]` | **0.5183** | 1.5661 | 1.5652 | 1.5670 |
| Qwen3.8-27B-AWQ-BF16-INT4 | `layers.11.self_attn.k_proj` `[1024, 5120]` | **0.5163** | 1.7629 | 1.7809 | 1.7839 |
| Qwen3.8-27B-AWQ-BF16-INT4 | `layers.11.self_attn.v_proj` `[1024, 5120]` | **0.5300** | 1.5751 | 1.5741 | 1.5776 |

The pinned reading is **2.8 to 3.5 times** tighter than every alternative, on
every tensor, on two artifacts from two different compressor builds. The three
alternatives land on each other, which is what "independent" looks like and is
itself a check that the statistic is measuring the pairing rather than something
about the codes.

**A margin of 0.52 against 1.48 is corroboration, not proof.** What it rules out
is the three specific alternatives the same bytes permit. It does not establish
the *code* lane order, which remains exactly where task 0018 left it, and it is
not evidence about model output.

### What was not measured

The zero-point **sign** is taken from the pinned library's `_dequantize`
(`(x_q - zero_point) * scale`) and is not separable by this statistic: negating
`z` changes `mean(q) - z` into `mean(q) + z`, which for a distribution centred
near zero is indistinguishable in mean absolute deviation. It is the same
equation document 03 already declares, from two independent sources, which is
why it is stated rather than measured.

## The mutation battery: 16 of 16 caught, 0 survivors

The method is experiments 0002–0004's: one edit that changes behaviour and
still compiles; apply, run every lane, record which caught it, revert. The
driver is at the bottom.

| Mutation | Caught by |
|---|---|
| `zp-sign-added-not-subtracted` | unit, artifact |
| `zp-lane-pinned-to-zero` | unit, artifact |
| `zp-lane-reversed` | unit, artifact |
| `zp-word-row-is-block-major` | unit, artifact |
| `zp-rebias-dropped` | unit, artifact |
| `zp-shape-check-deleted` | unit |
| `zp-length-check-deleted` | unit |
| `zp-vec-allocated-infallibly` | **allocfail only** |
| `zp-reads-the-padding-lanes` | unit, allocfail |
| `spec-payload-disagreement-ignored` | unit |
| `spec-payload-missing-ignored` | unit |
| `index-disagreement-symmetric-ignored` | unit, artifact |
| `index-missing-zero-point-ignored` | unit |
| `zero-point-dtype-unchecked` | unit |
| `zero-points-dropped-entirely` | unit, alloccount, artifact |
| `measurement-pinned-becomes-lane-reversed` | **artifact only** |

Lanes: `unit` is `cargo test -p moxie-format --lib`; `allocfail` is
`--test import_allocation_failure`; `alloccount` is
`--test import_allocation_asymmetric`; `artifact` is
`-p moxie-storage --test asymmetric_int4_import`; `symmetric` is
`--test gemma4_import`, which caught none — correctly, since it is the
regression lane for the path this task did not change.

**Two mutations are caught by exactly one lane each, and both are the point of
that lane.** `zp-vec-allocated-infallibly` swaps the fallible reservation for
`Vec::with_capacity`: every other lane passes, because the allocation succeeds
whenever memory is available. Only the injected-failure sweep sees it, and what
it sees is an abort — task 0019's rule for the sixth time in this workspace, on
a path added after the previous five. And
`measurement-pinned-becomes-lane-reversed` mutates the **measurement**, not the
product: it asks whether the declared thresholds would have accepted the wrong
reading. They would not.

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

The mutation driver is a Python script that applies each edit, runs the five
lanes, and reverts; it is reproduced in the task record's result section rather
than committed, because it is a one-shot measurement tool and the mutations are
the evidence.

Artifacts read, read-only, nothing written (**O5**):
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`, and
`/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4`. Pinned library:
`compressed-tensors` 0.17.0 at
`/home/rodrigo/.cache/uv/archive-v0/uqs9z2Tvizx6-8cq0I0rd/compressed_tensors`,
file hashes in
[the task contract](../../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md#the-serialization).
