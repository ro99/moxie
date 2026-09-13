# Handover — task 0024 implemented: asymmetric INT4 import, and a convention that could be measured

**Task 0024 is implemented on 2026-09-13 and is not accepted.** It has had no
independent review. Everything below is a claim awaiting one.

It is **M3 item 2's asymmetric half**: the compressed-tensors `pack-quantized`
importer now reads **asymmetric INT4 at group 32**, whose `weight_zero_point` is
packed along the **output** axis while the codes are packed along the input
axis. Two packing conventions inside one tensor group, distinguished by nothing
in either name — R16 in its exact form.

**It does not close M3 item 2**, which also wants group-128 symmetric INT4 and
the AutoRound/AutoGPTQ packing, and it establishes nothing about execution or
quality.

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Contract `e122de3`, written and committed **before** implementation; the
  implementation is the commits between it and the one this handover
  accompanies. Base before both was `02b74bd`.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, re-verified; its untracked `.pi/`
  and `tests/p2p/` remain untouched. R16 is the source lesson this task is
  against. The legacy tree has **no** asymmetric zero-point decode at all — its
  `src/platform/compressed_tensors.cpp:204` refuses asymmetric `pack-quantized`
  outright — so it could not be the pinned exporter here, and that is why the
  reading is pinned elsewhere and measured.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was copied, converted, deleted, downloaded or
  modified.** Headers of 21 shards were parsed and **14,549,088 B** of tensor
  payload were read through the accepted bounded reader.

## Completed facts

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy (`--features moxie-executor/driver`) | passed |
| CUDA-lane clippy (`--features cuda`) | passed |
| `cargo test --workspace --locked --offline` | **943 passed, 0 failed** (baseline **930**, re-measured at `e122de3` before implementation) |
| Device-feature workspace tests | **978 passed, 0 failed** (task 0023 recorded 965; +13 is exactly this task's new tests) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | zero failures, 79 rejected fixtures, 21 accepted, 13 rules |

**Nothing failed and nothing was skipped.** Both real artifacts are present on
this machine, so every artifact lane ran. Each of them prints `SKIP` with a
reason when its checkpoint is absent; that path was exercised while writing them
and is not what ran here.

**Mutation measurement: 16 of 16 caught, 0 survivors**, first measurement, no
corrections forced
([experiment 0005](../evidence/experiments/0005-asymmetric-int4-zero-point-assignment.md)).

### What was read, and what it is not

Six modules across two artifacts — three of
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` revision `bc59f497…` and three of
`/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4` — import to canonical affine
form, and **112,640 reconstructed values are bitwise equal** to the source's own
`(q − z) · s` computed independently from the raw bytes. All **34,996** of the
two artifacts' asymmetric modules had their declared zero-point shapes checked.

**Nothing executes a canonical INT4 tensor.** W4A16 is M3 item 3 and no kernel
exists; Laguna's graph still declares BF16 tensor requirements, because listing
INT4 would advertise a path that is not there. A bit-identical repack is
**ADR 0018's v1 quality definition** — it is not evidence about what any model
produces, and no paired output against a released model exists (**O2**).

## Decisions

**A named convention instead of a boolean.** `PackQuantizedSpec.symmetric: bool`
became `zero_points: ZeroPointSource`, with `Symmetric` and `PackedAlongOutput`.
The boolean said only that *some* zero points exist; the importer has to know
**where**, and this is the module whose whole job is to know that. `TensorTriple`
became `SourceTensors` for the same reason: a "triple" that can be four tensors
is a name that has stopped being true.

**The spec and the payload must agree, in both directions, in two places.** A
declaration of zero points with no payload is refused, and so is a payload with
no declaration; likewise `source_entries` refuses a module whose tensor index
disagrees with the declared serialization either way. These are two
independently obtained facts about one tensor, and reading one while ignoring
the other is how a whole tensor comes out shifted with every shape agreeing.

**Exactly one zero-point shape is accepted.** The scale rule also accepts a
one-dimensional `[out]`, because a writer was seen to emit it. No writer has been
seen to emit a one-dimensional zero point, so that is refused: accepting a shape
nothing produces is a branch no fixture can justify.

**What a writer padded with is not this reader's business.** The output axis is
padded to a whole word when it is not a multiple of `values_per_word`. The test
does not assert the padding is zero — that would be a claim about a writer. It
imports the same tensor twice with **different** padding and requires an
identical result, which is a property of this reader.

### The convention that could be measured, and why it mattered

Task 0018 recorded that a **code** word's lane order cannot be checked against an
artifact, because all its lanes fall inside one scale group: reversing them
reconstructs a different but equally plausible weight. A **zero-point** word is
different. Its lanes are different **output channels**, whose codes have
different statistics, so the assignment is a property of the file.

The statistic and its thresholds were written into the contract **before** the
measurement: `mean |mean_code − z|`, pinned below 1.0 and every alternative
above 1.2, on the reasoning that an asymmetric group's codes are centred near
its zero point.

| Artifact | Module | pinned | lane-reversed | block-major | block reversed |
|---|---|---:|---:|---:|---:|
| Laguna | `layers.1.mlp.experts.0.down_proj` | **0.5236** | 1.4900 | 1.4853 | 1.4817 |
| Laguna | `layers.1.mlp.experts.0.gate_proj` | **0.5183** | 1.5661 | 1.5652 | 1.5670 |
| Qwen3.8-27B | `layers.11.self_attn.k_proj` | **0.5163** | 1.7629 | 1.7809 | 1.7839 |
| Qwen3.8-27B | `layers.11.self_attn.v_proj` | **0.5300** | 1.5751 | 1.5741 | 1.5776 |

This is corroboration of a **reading**, not proof and not a quality claim. It
mattered rather than being a flourish: the reading is pinned to
`compressed-tensors` **0.17.0** resolved locally, while both artifacts declare
compressor versions `0.1.dev534+gb269f2e` and `0.1.dev535+gdc9611a` — untagged
development builds that pin nothing. This repository has already refused one
mismatched `transformers` copy as a pinned exporter, for the yarn RoPE ramp.
**When a convention cannot be checked against the artifact, say so and pin it;
when it can, measure it.**

The battery includes a mutation of the **measurement** rather than the product:
replacing the pinned candidate with the lane-reversed one. The declared
thresholds reject it. A threshold nobody has watched fail is a threshold nobody
has measured.

## Remaining hypotheses and blockers

- **The task is not reviewed and not accepted.** Everything above is a claim.
- **Nothing executes a canonical INT4 tensor.** That is the largest remaining
  gap in M3 and it is item 3's: W4A16/W8A16 dense and expert paths for SM86,
  with SM120 qualified separately. No capability row moved out of "not
  implemented" for execution.
- **The code word's lane order is still unchecked**, exactly where task 0018
  left it. The zero-point **sign** is taken from two independent statements of
  `(q − z) · s` — document 03 and the pinned library — rather than measured; a
  mean absolute deviation cannot separate `+z` from `−z` on a distribution
  centred near zero, and this is stated rather than quietly counted as measured.
- **Group-128 symmetric INT4 with `actorder: "static"`** is item 2's remainder.
  Both local candidates (`canada-quant/glm-5.3-w4a16-mtp`,
  `canada-quant/hy3-w4a16-mtp`) declare it, and **what `static` means for
  logical column identity must be read from the pinned exporter before anything
  consumes them.** A permutation folded into the stored weights changes which
  activation column each canonical column means; document 03 forbids ignoring
  one. Do not assume it is the same as `actorder: null`.
- **Cross-shard module resolution does not exist.** `source_entries` takes one
  `Header`. Qwen3.8-27B splits **every** one of its 256 modules — codes and zero
  points in shards 1–2, scales elsewhere — so a production caller needs an index
  resolver. That is M3 item 1's manifest work. This task's artifact lane
  resolves per tensor in the test and records the gap rather than building it.
- **Laguna's shard headers exceed `HeaderBudget::DEFAULT`.** 140,989 tensors,
  about a megabyte of header per shard, a 16.5 MB admitted peak against 8 MiB.
  The default refusing it is the budget working; the default is unchanged, and
  any production path opening this artifact must state one.
- **Laguna's attention tower is unchanged**: `softplus` output gating and the
  yarn rotary ramp are still gaps, and **no Laguna layer is composable.** An
  importer that reads its weights does not change that.
- **Two more local artifacts declare the same packing and are not tested here**:
  `Muse-Glimmer-30B-AWQ-INT4` (which also declares a general `dtype=float16`,
  so its scale dtype should be read rather than assumed) and
  `Inkling-Small-AWQ-INT4`.
- **The fused/per-expert role mapping** stays open — the Laguna record's
  question 1. This task imports at tensor granularity and the canonical form
  needs no fused layout, so it did not have to answer it and did not.

## Next task

**M3 item 3 — shared W4A16/W8A16 dense and expert paths — is the one that turns
every imported tensor into something.** It is the largest remaining gap in the
milestone and the reason nothing in this repository executes a quantized weight.
Document 03 bounds it before it starts: weight-only paths with BF16 preferred
and FP32 accumulation, **not** INT4×INT4 or INT8×INT8 MMA; SM86 first with SM120
qualified **separately**; and "bounded reference dequantization is not an
acceptable final fast path by assertion" — shared bounded dequantization plus a
BF16 GEMM is the correctness fallback, not a claimed result. Dense GEMM
qualification does not qualify routed or grouped MoE.

The alternative, if a smaller step is wanted first, is **item 2's remainder**:
the group-128 symmetric INT4 import, whose real content is the `actorder:
"static"` question above rather than the packing, which this importer already
handles.

Either way the standing bound is unchanged: **Moxie never quantizes** (ADR
0017), v1 quality is a bit-identical repack (ADR 0018), and **no
agent-initiated bulk download, copy or conversion** may start without a task
naming artifact, revision, expected size and retention (ADR 0020).
