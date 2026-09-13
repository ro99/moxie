# Handover — task 0024 implemented and corrected: asymmetric INT4 import, and a convention that could be measured

**Task 0024 is accepted by the owner on 2026-09-13**, after three rounds of
independent review. The rounds made **eight findings, two P1**; all eight were
reproduced, all eight are fixed, **none is disputed**. The third round reported
no new blocking findings and recommended acceptance "within task 0024's declared
importer-only M3 item 2 scope", with two nonblocking P3 corrections that are
applied.

**The acceptance closes task 0024 only.** It does not close M3 item 2, and it
establishes no W4A16 execution, no model-output quality and no cross-shard
production resolution.

**The second round's three findings share one shape: each is a first-round
correction that was narrower than it looked.** A refusal made allocation-safe
while the lookup names on the way to it still aborted; a regression written for
a filtering bug that tested the helper the filter ran *before*; and a driver
that reported an invalid verdict as a caught one. Fixing a finding and guarding
the fix are two jobs, and the second round is what happens when only the first
is done.

The corrections are summarised below and written up in full in
[the task record](../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md#corrections-after-the-first-independent-review).

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
| `cargo test --workspace --locked --offline` | **947 passed, 0 failed** (baseline **930**, re-measured at `e122de3` before implementation) |
| Device-feature workspace tests | **982 passed, 0 failed** (task 0023 recorded 965; +17 is exactly this task's new tests) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | zero failures, 79 rejected fixtures, 21 accepted, 13 rules |

**Nothing failed and nothing was skipped.** Both real artifacts are present on
this machine, so every artifact lane ran. Each of them prints `SKIP` with a
reason when its checkpoint is absent; that path was exercised while writing them
and is not what ran here.

**Mutation measurement: 24 of 24 caught, 0 survivors, 0 unstable, 0 invalid
controls, 0 skipped**, every verdict repeated three times in both directions,
**driver committed** and its verdict rule self-tested
([experiment 0005](../evidence/experiments/0005-asymmetric-int4-zero-point-assignment.md),
[its driver](../evidence/experiments/drivers/0005-mutations.py)). The first
measurement was 16 of 16 against a suite with **five** holes in it, and the
second 21 of 21 against a suite with three. **Eight mutations were added after a
review — five then three — and every one is caught by exactly one lane**; seven
by a check a finding added, and `bf16-rounding-truncates` by a pre-existing BF16
test, because an existing lane acquiring a new regression is not a lane a review
created.

### What was read, and what it is not

Six modules across two artifacts — three of
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` revision `bc59f497…` and three of
`/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4` — import to canonical affine
form, and **112,640 reconstructed values are checked against two quantities that
are not the same one**: bitwise against document 03's canonical FP32
`(q − z) · s` over the source's own bytes, and against the source's own
arithmetic **with the BF16 boundary its reference applies**, which the canonical
value rounds to exactly. The boundary moves **27,501** of them, a count the test
prints and requires to be nonzero. All **34,996** of the two artifacts'
asymmetric modules had their declared zero-point shapes checked, and a module
missing any of its four tensors is now reported rather than dropped from the
audit's own population.

**Nothing executes a canonical INT4 tensor.** W4A16 is M3 item 3 and no kernel
exists; Laguna's graph still declares BF16 tensor requirements, because listing
INT4 would advertise a path that is not there. A bit-identical repack is
**ADR 0018's v1 quality definition** — it is not evidence about what any model
produces, and no paired output against a released model exists (**O2**).

## What the second review found, and what changed

**Three findings, one P1, none disputed.**

1. **P1 — `source_entries` aborted before reaching its now-safe refusal.** It
   builds four lookup names with `format!`; eight of its eleven allocation
   positions took the process down, two of them on task 0024's own zero-point
   name. My eight-case refusal sweep calls only `import`, so it could not see
   any of them. `join_name` reserves each name fallibly, the shape comparison
   no longer builds a `Vec`, and a new sweep refuses **29** positions across
   five `source_entries` cases with the headers parsed beforehand.
2. **The regression for the first round's finding 3 could not detect it.** It
   called `missing_companions` directly; the defect was a filter that ran
   *before* that helper, so restoring the filter left all five artifact tests
   passing. The fixture now writes a real four-module shard with one zero point
   missing and goes through `Inventory::build`; with the filter restored it
   fails on `left: 3, right: 4`, the review's symptom exactly.
3. **The mutation driver counted invalid verdicts as caught.** An unstable
   repeat or a restored control that still failed was reported *and* added to
   the total. The verdict is a pure `classify` function now, only `caught`
   counts, skips are named, the process exits nonzero unless every mutation is
   caught, and `--self-test` checks the rule over eight cases including both of
   the review's.

## What the first review found, and what changed

**Five findings, one P1, none disputed.** The short version; the task record has
the full account.

1. **P1 — a refusal under memory pressure aborted the process.** Every refusal
   in `moxie-format` built its prose with `format!`, so refusing one allocation
   while the importer rejected a malformed artifact gave `SIGABRT`. My
   allocation sweep swept every position of an import that **succeeds**, and a
   refusal has no positions in it at all — **task 0023's third-round lesson word
   for word, in a file written by someone who had just read it.**
   `Error::InvalidArtifact` carries a `Cow<'static, str>` now, this crate's
   refusals compose their prose into a `try_reserve`d buffer and fall back to a
   borrowed static detail, and eight refusals are constructed with every
   allocation position refused in turn.
2. **The comparison was not the comparison it was advertised as.** The pinned
   `_dequantize` casts to `scale.dtype` before subtracting and multiplying, and
   that is **BF16** for both artifacts; the test computed FP32 and the records
   called it "the source's own declared arithmetic". Neither quantity is wrong —
   document 03 fixes the canonical reconstruction at FP32 — but one was called
   by the other's name, which is **task 0022's BF16-boundary lesson one task
   later.** There are two comparisons now, and the count of values the boundary
   moves is asserted nonzero so the new check cannot be vacuous.
3. **The inventory audit could not fail**, because it filtered incomplete
   modules out of its own population first. A filter applied to the population
   being audited removes exactly the rows the audit exists to see — task 0023's
   omitted layer, in a different file. Every packed module is kept now,
   `importable()` is the separately named subset the import tests use, and the
   audit's decision is a pure function tested against a table containing the
   case the real artifacts do not supply.
4. **The mutation driver was described and not committed**, and the contract's
   promise of repeated substitutions was not kept and not reported. Both fixed:
   the driver is tracked, every verdict is repeated three times in both
   directions, and a lane that disagrees with itself is reported rather than
   counted.
5. **The recorded reason for not measuring the zero-point sign was false.** It
   assumed independence between a group's code mean and its own zero point —
   the correlation that is the measurement's entire premise, two paragraphs
   earlier. The sign is a fifth candidate now and separates further than any
   misassignment: 2.18–2.57 against the pinned 0.52.

**A scope extension, stated because it exceeded the contract.** Fixing finding 1
properly meant widening a field of `moxie_types::Error`, which touches 41 files
and 150 construction sites rather than the two files the contract named. The
reason is that the defect is not reachable from the importer alone —
`desc.validate()`, `AffineTensor::new` and `scales.validate()` are all on the
import path and all aborted the same way — and "pre-existing" is how a standing
failure becomes background noise. Every one of the 150 edits is mechanical, and
each was located by rustc's own spans rather than by a regex over source.

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

**Task 0025 — M3 item 1's bounded offline repack and canonical publication.**
Its contract is authored and committed at `74e0709` by a concurrent agent and
its status is *proposed*.

**This section originally named M3 item 3 and that was wrong.** It was written
before [ADR 0021](../decisions/adr/0021-repack-is-a-moxie-program.md),
[ADR 0022](../decisions/adr/0022-user-programs-and-canonical-write-authority.md)
and [task 0025](../tasks/0025-m3-offline-repack-publication.md) existed. Three
things decide it the other way:

1. **Roadmap order.** Repack is item **1**; the importers this task extended are
   item 2, delivered ahead of it. Item 3 is later still.
2. **ADR 0021 assigns it explicitly** — repack "is built under M3 item 1" — and
   says M3's exit clause, *"lossless (repack) claims have source-oracle
   evidence"*, **cannot close without the program that publishes what the
   evidence is about.**
3. **Nothing persists what this task imports.** The importer produces canonical
   `AffineTensor`s in memory that no manifest holds and no reader round-trips.
   A kernel built before that has nothing to execute from but a test fixture.

**Two of task 0025's stated preconditions have since moved**, and its own
contract asks that they be re-recorded before it is activated: its base is
`645e759` with a dirty tree, and it says "task 0024 remains unaccepted ... do
not implicitly accept it by using it." Task 0024 **is** accepted, at
`adf47cb`, on a clean tree. That note has been updated in place; nothing else
of that contract was touched, because it is another agent's and its design
decisions are not this task's to revise.

**M3 item 3 — shared W4A16/W8A16 dense and expert paths — is what follows**, and
it is still the largest gap in the milestone: nothing in this repository
executes a quantized weight. Document 03 bounds it before it starts: weight-only
paths with BF16 preferred and FP32 accumulation, **not** INT4×INT4 or INT8×INT8
MMA; SM86 first with SM120 qualified **separately**; and "bounded reference
dequantization is not an acceptable final fast path by assertion" — shared
bounded dequantization plus a BF16 GEMM is the correctness fallback, not a
claimed result. Dense GEMM qualification does not qualify routed or grouped MoE.

**Item 2's own remainder is also open** and is smaller than either: the
group-128 symmetric INT4 import, whose real content is the `actorder: "static"`
question above rather than the packing, which this importer already handles.

Either way the standing bound is unchanged: **Moxie never quantizes** (ADR
0017), v1 quality is a bit-identical repack (ADR 0018), and **no
agent-initiated bulk download, copy or conversion** may start without a task
naming artifact, revision, expected size and retention (ADRs 0020–0021).
