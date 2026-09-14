# Task 0024 — M3 item 2: asymmetric INT4 `pack-quantized` import, zero points along the output axis

Status: **accepted by the owner on 2026-09-13**, after three rounds of
independent review — eight findings, two P1, all reproduced, all fixed, none
disputed; the third reported no new blocking findings and recommended acceptance
"within task 0024's declared importer-only M3 item 2 scope". Two nonblocking P3
corrections from that round are applied.

**The acceptance closes task 0024 only.** It does not close M3 item 2, which
also wants group-128 symmetric INT4 and the AutoRound/AutoGPTQ packing, and it
establishes **no** W4A16 execution, **no** model-output quality, **no**
cross-shard production resolution.

## Identity and authority

- Task ID 0024. Milestone **M3**, roadmap item 2 ("Import pinned
  compressed-tensors group-128 symmetric and **group-32 asymmetric INT4**").
  Owner/reviewer: the repository owner.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`, base
  commit `02b74bd`, working tree clean at authoring time.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and
  `tests/p2p/` are not touched.
- Requirement / finding IDs: **R16** — "Format names hide incompatible
  scale/layout conventions". This task is R16 in its exact form: two packing
  conventions inside one tensor group, distinguished by nothing in either name.
- Required documents: [03 §affine integer
  v1](../spec/03-memory-formats-and-cuda.md), [09 §A/§E](../spec/09-agent-playbooks.md),
  [ADR 0003](../decisions/adr/0003-int4-int8-bf16-weight-family.md),
  [ADR 0017](../decisions/adr/0017-v1-catalog-and-no-quantizer.md),
  [ADR 0018](../decisions/adr/0018-v1-quality-is-bit-identical-repack.md),
  [ADR 0020](../decisions/adr/0020-user-managed-storage-and-canonical-materialization.md),
  [task 0018](0018-m3-compressed-tensors-int8-importer.md) (the symmetric
  importer this extends), [the Laguna bring-up record](../models/laguna.md).
- **Owner gates already resolved, and they bound this task rather than opening
  it.** O1 pinned the v1 catalog (Laguna is number 8). O2 is repack-only: a
  lossless claim needs source-oracle evidence and **Moxie never quantizes**. O5
  is user-managed storage: `/models` and `/fast/models` are read-only inputs and
  **nothing may be written, copied, converted or downloaded**. O6 and O7 stay
  open, which is why no timing here is a claim.
- **Stop point.** If the pinned zero-point serialization cannot be established
  from a source outside this repository, the refusal stays and the task reports
  the evidence rather than guessing an offset convention.

## Bounded deliverable

**One concrete outcome:** `moxie-format`'s compressed-tensors importer accepts
**asymmetric** `pack-quantized` sources whose `weight_zero_point` is packed
along the **output** axis, producing the canonical `AffineTensor` with
`ZeroPoints::PerGroup`, and the reading is checked against the artifacts' own
bytes rather than asserted.

- **Sole owning shared component:** `moxie_format::compressed_tensors`. No
  second decoder appears anywhere; `moxie_format::affine` already carries the
  canonical zero-point representation and is not changed except where a
  measurement says it must be.
- **Allowed production files:** `crates/moxie-format/src/compressed_tensors.rs`,
  and `crates/moxie-format/src/affine.rs` only if a defect is found there.
- **Allowed test files:** `crates/moxie-format/src/compressed_tensors.rs`'s own
  test module, `crates/moxie-format/tests/import_allocation.rs`,
  `crates/moxie-storage/tests/gemma4_import.rs` (the existing symmetric
  real-artifact lane, updated for the renamed source-tensor type), and one new
  `crates/moxie-storage/tests/asymmetric_int4_import.rs`.
- **Explicit non-goals and forbidden shortcuts:**
  - **No execution and no kernel.** W4A16 is M3 item 3. A tensor that imports is
    a tensor nothing runs.
  - **No quality claim of any kind.** O2 is repack-only and paired-output
    evidence does not exist for any artifact here.
  - **No manifest, no canonical publication, no repack output.** M3 item 1 owns
    the inspector/repacker, and under ADR 0020 the repack is a user-run external
    script. **This task writes no bytes under any checkpoint root.**
  - **No activation ordering.** The two local group-128 symmetric INT4
    artifacts declare `actorder: "static"`, which is a different question with
    its own evidence; it is inspected and recorded as the next task, not
    implemented here.
  - **No fused/per-expert tensor-role mapping.** The Laguna record's open
    question 1 (per-expert on disk, fused in the model) is a role-mapping
    question, not a packing one, and stays open.
  - **No model graph change that advertises an INT4 execution path.** Laguna's
    `TensorRequirement`s stay BF16, because no kernel consumes canonical INT4.
- **Existing consumers and second-consumer proof:** the symmetric INT8 path
  (task 0018, Gemma 4 31B) must still pass unchanged; the new asymmetric path is
  exercised on **two independent real artifacts** with different shapes,
  observers and compressor version strings —
  `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` and
  `/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4` — plus synthetic fixtures at
  widths and tails neither artifact contains.
- **Temporary paths to delete:** none. The asymmetric refusal is replaced, not
  bridged.

## Contract before implementation

### The equation

Unchanged, and that is the point. Document 03:

```text
W[o,k] = (Q[o,k] - Z[o,group(k)]) * decode_scale(S[o,group(k)])
```

The pinned exporter's own dequantization is
`(x_q - zero_point) * scale` (`quantization/lifecycle/forward_helpers.py`,
`_dequantize`), which is the same equation with the same sign. A zero point that
were *added* rather than subtracted would reconstruct a different tensor with no
shape disagreeing, so the sign is named here before any code is written.

### The serialization

Source of truth: the `compressed-tensors` library, **version 0.17.0**, resolved
on this machine at
`/home/rodrigo/.cache/uv/archive-v0/uqs9z2Tvizx6-8cq0I0rd/compressed_tensors`:

| File | sha256 |
|---|---|
| `compressors/pack_quantized/helpers.py` | `8619308666eba5e8a442d1647c34b6b5f9716b13ed2051ed17fb1ea683b0db72` |
| `compressors/pack_quantized/base.py` | `a6a532a0b2ae19b7ebfb425d73776653ee8d117b3f69778f05e44e67175cfc9d` |
| `quantization/lifecycle/forward_helpers.py` | `8b3399fda143cc249c231e291d938c987a7da4793eeae35b3136b3390bbde6c8` |

`PackedQuantizationCompressor.compress` packs the zero point as
`pack_to_int32(zero_point, num_bits, packed_dim=0)` when the scheme is
asymmetric and its strategy is `GROUP` or `CHANNEL`, and `decompress` unpacks it
into `(*original_shape[:-1], scale.shape[-1])`, i.e. `[out_features, groups]`.
With `packed_dim=0`, `pack_to_int32` transposes, packs along what is now the
column axis, and transposes back; `unpack_from_int32` writes
`unpacked[l::pack_factor, :] = value >> (bits * l)`. Both give one mapping:

```text
values_per_word = 32 / bits
zp_rows         = ceil(out_features / values_per_word)
word_index      = (o / values_per_word) * groups + g
lane            = o % values_per_word
raw             = (word >> (lane * bits)) & ((1 << bits) - 1)
z               = raw - (1 << (bits - 1))
```

**Two packing conventions in one tensor group.** `weight_packed` packs
`values_per_word` codes along the **input** axis; `weight_zero_point` packs
`values_per_word` zero points along the **output** axis. The existing importer
has only ever seen the first. Nothing in either tensor's name says which it is;
the shapes do, and they are validated rather than assumed.

**What the artifact establishes and the symmetric case could not.** Task 0018
recorded that the lane order inside a packed *code* word cannot be checked
against the artifact, because all lanes of a word fall inside one scale group,
so reversing them reconstructs a different but equally plausible weight. The
zero-point word is different: its lanes are **different output channels**, whose
statistics differ. That makes the lane assignment measurable, and this task
measures it (below) instead of resting on the library alone — which matters,
because the four asymmetric artifacts declare compressor versions
(`0.1.dev534+gb269f2e`, `0.1.dev535+gdc9611a`) that are **not** the locally
resolved 0.17.0.

### Shapes, precision, rounding, layout

- `weight_packed`: `I32`, `[out_features, ceil(in_features / values_per_word)]`.
  Unchanged.
- `weight_scale`: dtype read from **its own header entry**, `[out_features,
  groups]`, or `[out_features]` when `groups == 1`. Unchanged.
- `weight_zero_point`: `I32`, `[ceil(out_features / values_per_word), groups]`.
  Its group count must equal the scale's; a disagreement is an
  `InvalidArtifact`, not a reshape.
- `weight_shape`: `I64[2]`, the authority for the logical shape. Unchanged.
- Output-axis **padding**: `pack_to_int32` pads the packed axis to a multiple of
  `values_per_word` with zeros, which rebias to the most negative code. Padding
  lanes are never read. `out_features` divisible by `values_per_word` is not
  assumed — no local artifact has such a tail, so synthetic fixtures supply one.
- Canonical output: codes logical row-major and byte-aligned per row (unchanged),
  scales preserved as **source bytes and source dtype**, zero points as `i16`
  per `(output channel, group)` — wider than the code range on purpose, and not
  clipped.
- No rounding happens on this path. An import that rounded would be a precision
  conversion, which ADR 0017 forbids in Moxie.

### Partition and hardware capabilities

None. This is a host-side importer; it touches no device, no stream and no
kernel. Partition legality belongs to the consuming operation and no consumer
exists yet.

### Peak memory, transfer dependencies, lifetimes

Bounded by one tensor: the packed code destination (`code_bytes`), one
row-of-columns scratch, the scale vector and the zero-point vector. **Every one
is reserved fallibly** — `try_reserve_exact`, `CapacityExceeded` on failure. The
recurring lesson in this workspace is that an infallible allocation on an import
path is an abort, and it has been found five times; the zero-point vector is a
new collection on an old path and gets the same treatment. Nothing is allocated
per row. Source bytes are borrowed slices the caller owns; the importer retains
none of them.

### Cancellation, failure and rollback

An import is a pure function of borrowed bytes: it either returns an
`AffineTensor` or a typed error, and there is nothing to roll back. Every
refusal is `Error::InvalidArtifact` or `Error::Unsupported`, never a panic and
never a string match.

### Independent oracle and predeclared metrics

1. **Decode oracle.** The module's existing test oracle — document 03's equation
   written over the *source* bytes without calling `import` or `AffineTensor` —
   is extended with the zero-point term, derived from the pinned library's
   `unpack_from_int32(packed_dim=0)` rather than from the importer.
   **Predeclared gate: bitwise equality** of every reconstructed `f32` against
   the oracle, over every fixture.
2. **Exhaustive codes.** All 16 INT4 codes × all 16 zero points = **256**
   combinations reconstruct exactly, in every lane of a word, for both widths
   (INT8 gives 256 × 256 sampled over its full code range in every lane).
3. **The measured lane assignment, on real bytes.** For a real tensor, compute
   the mean code of each `(output channel, group)` and compare it with the zero
   point assigned by the pinned mapping and by each alternative the same shapes
   permit (lane-reversed; block-major `o = j + rows*l`; block-major reversed).
   The statistic is `mean |mean_code - z|`. **Predeclared threshold: the pinned
   assignment's statistic must be below 1.0 and every alternative's above 1.2**,
   on every tensor tested, on both artifacts. The rationale is stated before the
   measurement: an asymmetric group's codes are centred near its zero point, so
   a correct pairing is tight and a wrong one is two independent quantities
   subtracted. **This is corroboration, not proof, and it is reported as
   corroboration** — it cannot become a quality claim, which is O2's.
4. **Repack faithfulness.** For a real tensor, reconstruct through the canonical
   decoder and compare against the source's own declared arithmetic computed
   independently from the raw bytes. **Predeclared gate: bitwise equality**,
   which is ADR 0018's "bit-identical repack `W=(Q-Z)*S`" at tensor scale — and
   is *not* a model-quality claim.

### Application compatibility and sampler implications

None. No CLI surface, protocol field, sampler or generation path changes. The
CLI's Laguna inspection test asserts that the importer refuses this artifact;
that assertion becomes its opposite and the comment explaining why is corrected
rather than deleted.

## Acceptance

### Tests and required hardware

Host only. No GPU is required and none is claimed.

| Gate | Command |
|---|---|
| Format | `cargo fmt --all -- --check` |
| Host clippy | `cargo clippy --workspace --all-targets --locked -- -D warnings` |
| Device-lane clippy | `cargo clippy --workspace --all-targets --locked --features moxie-executor/driver -- -D warnings` |
| CUDA-lane clippy | `cargo clippy --workspace --all-targets --features cuda -- -D warnings` |
| Host tests | `cargo test --workspace --locked --offline` |
| Allocation | `cargo test -p moxie-format --test import_allocation --locked --offline -- --nocapture` |
| Real artifacts | `cargo test -p moxie-storage --test asymmetric_int4_import --locked --offline -- --nocapture` |
| Symmetric regression | `cargo test -p moxie-storage --test gemma4_import --locked --offline -- --nocapture` |
| Architecture | `cargo xtask arch-check` |
| Specification | `cargo xtask spec-check` |

Baselines are **re-measured at `02b74bd`**, not quoted from the previous
handover, because task 0022's record says what happens when a count is taken on
a half-built tree.

Every real-artifact test **skips with a printed message** when its artifact is
absent, so a fresh clone is green and a missing checkpoint is never mistaken for
a regression. Skipped is reported separately from passed.

### Test-strength measurement

**Mutation testing, reported as a number the run prints.** Every new check gets
a mutation that should defeat it: the zero-point sign, the lane mapping, the
word index, the rebias, the padding truncation, the group-count agreement, the
spec/payload agreement, the fallible reservations, and each shape validation. A
survivor is a defect or a missing fixture, never a test-strength opinion. The
result goes in a new `docs/evidence/experiments/0005-*.md` with its driver.

A substitution is repeated in both directions, because task 0022's record shows
a flaky substitution turns a measurement into a coin flip.

### Support-matrix entries to update

`G-CT-IMPORT`'s row, and the affine/import rows that currently say asymmetric is
refused. The Laguna row's blocker list loses the importer and keeps everything
else. **The "not implemented" verdict for W4A16 execution does not move.**

### Deletion and documentation gates

- The `Unsupported` refusal for asymmetric import is deleted, and the negative
  fixture that asserted it is replaced by fixtures asserting the new refusals.
- [The Laguna bring-up record](../models/laguna.md) loses blocker 2 and gains
  what the importer established and what it did not.
- [The quantization candidates record](../evidence/quantization-candidates.md)
  gains the four asymmetric artifacts' verified packing parameters and the two
  `actorder: "static"` artifacts as the recorded next question.
- A handover naming the next bounded task.

### Exact condition requiring owner direction or task rejection

- If the measured lane assignment does **not** separate the pinned mapping from
  every alternative at the predeclared threshold, the import stays refused and
  the task reports the measurement. A statistic that fails is evidence, not a
  threshold to move.
- If any step would require writing, copying or converting bytes under `/models`
  or `/fast/models`, stop: ADR 0020 requires a task naming artifact, revision,
  expected size and retention, and this task names none because it needs none.

## Result, filled after work

### Corrections after the third independent review

Two findings, both **P3**, both nonblocking; the round recommended acceptance.

**The driver accepted unknown mutation selectors.** `nonexistent-mutation`
exited 0 with "0 of 0 caught", and a mixed valid/unknown selection silently
dropped the unknown one. A battery that quietly omits the work it was asked for
reports a number about a different battery. `select()` is a pure function now,
unknown names and an empty selection are refused with exit 2, and five selector
cases joined the three-case self-test — 13 of 13.

**The evidence prose contradicted its own table.** The battery grew 16 → 21 →
24, which is **eight** additions; the prose said "nine of the eleven", conflating
"added after a review" with "caught by exactly one lane" (eleven mutations are
single-lane, of which eight were added). The task index also still headlined 21
of 21. Corrected in all four places, and the counts are now derived from the
recorded table rather than restated from memory. The review also drew a
distinction I had blurred: **an existing lane acquiring a new regression is not
a lane the review created** — `bf16-rounding-truncates` is caught by
`moxie-format`'s pre-existing BF16 tests, so it is seven of eight, not eight.

Historical measurements stay as they were, dated: 16 of 16 on the first run, 21
of 21 after the first review.

### Corrections after the second independent review

Three findings, one P1. **All three reproduced, all three fixed, none
disputed.** Every one of them is about a correction from the first round being
narrower than it looked.

#### 1 (P1). The refusal was made safe and the path to it was not

`source_entries` builds four lookup names with `format!` and compared a shape
against `vec![2]`. Making the error it *returns* fallible did nothing for the
allocations on the way there: the review measured **eleven** allocation
positions in one call and **eight of them aborted**, including the two that
build task 0024's own zero-point name, after the other three entries had
already resolved.

**That is task 0023's sentence one layer further out** — "reserving a
destination says nothing about a temporary the callee builds" — and my
eight-case refusal sweep could not see it, because **every case in it calls
`import`**. A sweep that names one function has tested one function.

`crate::join_name` builds each name with one `try_reserve_exact` and returns
`CapacityExceeded`; the shape comparison is `as_slice() != [2]` and allocates
nothing. `every_allocation_position_in_source_entries_is_a_typed_error` parses
its headers **before** the injection starts, so what is swept is the helper and
not the parser, and it refuses **29** positions across five cases. Two mutations
put the `format!` calls back.

#### 2 (P2). The regression for finding 3 could not detect finding 3

The test I wrote called `missing_companions` directly. The defect was a filter
applied *before* that helper ran, so **a correct helper cannot detect a module
that was removed before it was called** — the review restored the original
filter and all five artifact tests passed.

This is the same shape as task 0023's own "the test I wrote for a review's
finding had a fake axis", and it is worse in one way: I had the reviewer's
reproduction in front of me — a synthetic shard with a module missing its zero
point — and wrote a table-driven test of the helper instead of building the
shard.

The fixture builds one now. `Source` owns its root so a synthetic artifact in a
temporary directory is the same kind of thing as a real one;
`the_inventory_reports_a_module_missing_a_companion_rather_than_dropping_it`
writes a four-module shard with `m3.weight_zero_point` omitted, constructs the
inventory **through `Inventory::build`**, and asserts both halves the filter
breaks: the independent packed-module population (4, not 3) and the module the
audit names. With the original filter restored it fails on `left: 3, right: 4`,
which is the review's symptom exactly. `inventory-filters-incomplete-modules` is
in the battery.

#### 3 (P2). The driver counted invalid verdicts as caught

Unstable mutant repeats and a restored control that still failed were appended
to `nondet` **and then counted in the total anyway**. The review drove the
driver's own `main` with a consistently failing control and it printed "1 of 1
caught, 0 survivors".

**A measurement tool that cannot report an invalid measurement is not a
measurement tool.** The verdict is a pure function now — `classify(caught_by,
mutant_ok, control_ok)` — returning `caught`, `survivor`, `unstable` or
`invalid-control`; only `caught` counts, skips are tracked and named, and the
process exits nonzero unless every mutation is caught. `--self-test` checks the
rule over eight cases including both of the review's, because the rule is what
the headline number means.

### Corrections after the first independent review

Five findings, one P1. **All five reproduced, all five fixed, none disputed.**
Two of them reproduce to the digit: the review's 27,501 divergent values and its
`2.1797049840291343` are what this workspace's own tests now print.

#### 1 (P1). A refusal under memory pressure aborted the process

Every refusal in this crate built its prose with `format!`, so refusing one
allocation while the importer rejected a malformed artifact gave **SIGABRT**. I
reproduced it on the first try, and on a path task 0018 wrote rather than one of
mine — the defect is the crate's, not the six lines this task added.

**That is task 0023's third-round lesson, word for word, in a file written by
someone who had just read it**: "the sweep written for the previous P1
reconciled only a *valid* trace, so it never constructed a diagnostic and could
not have caught this." My allocation sweep sweeps every allocation position of
an import that *succeeds*. A refusal has no positions in it at all.

The fix is the one task 0023 arrived at, moved to where it belongs — the shared
error type. `Error::InvalidArtifact` carries a `Cow<'static, str>`, so the
variant can be built **without allocating**; `moxie-format`'s refusals compose
their prose into a buffer grown only through `try_reserve` and fall back to a
borrowed static detail when that fails. The variant is what a caller branches
on (document 02), so the refusal survives and only its prose degrades.

**This exceeded the contract's declared file list**, which named
`compressed_tensors.rs` and `affine.rs`. Widening a field of `moxie_types::Error`
touches 41 files and 150 construction sites. I did it anyway, and the reason is
the repository's own rule rather than convenience: the defect is not reachable
from the importer alone — `desc.validate()`, `AffineTensor::new` and
`scales.validate()` are all on the import path and all aborted the same way — and
"pre-existing" is how a standing failure becomes background noise. Every one of
the 150 edits is mechanical (`.into()`), driven by rustc's own spans rather than
by a regex over source, so no site was guessed and none was missed.

The gate that was missing now exists:
`every_refusal_is_a_typed_error_with_every_allocation_refused` constructs **eight**
refusals with **every** allocation position refused in turn, and a mutation
(`refusal-prose-allocates-infallibly`) puts the old `format!` back to prove the
gate fails on it.

#### 2 (P2). The comparison was not the comparison it was advertised as

The test computed `(q - z) * s` in FP32 and the record called it agreement with
"the source's own declared arithmetic". The pinned `_dequantize` casts to
`scale.dtype` first, and for both artifacts that is **BF16**, so the source's
value is a BF16 number. The review ran the pinned helpers and measured **27,501
of 112,640** sampled values where the two differ; this workspace's own test now
prints the same number.

**Neither quantity is wrong and that is the point.** Document 03 fixes the
canonical host reconstruction at FP32, and the repack is faithful: codes, zero
points and scale bytes are preserved exactly. What was wrong was calling one
quantity by the other's name — **task 0022's lesson about a BF16 boundary,
one task later.**

There are two comparisons now. The canonical FP32 equation over the source's own
bytes, bitwise; and the source's own arithmetic *with its rounding boundary*,
which the canonical value must round to exactly. The second is sound because it
rounds once and not twice: `q - z` is an integer in `[-15, 15]` and the scale is
a BF16 value, so the FP32 product is **exact** and the only rounding is the
single round-to-nearest-even into BF16. And the boundary is shown to fire —
`narrowed` counts the values it moves and the test **requires that count to be
nonzero**, because a boundary check on a sample where the boundary never fires
proves nothing.

#### 3 (P2). The audit could not fail, because its population was filtered first

`Inventory::build` dropped every module missing one of its four tensors, and
then `every_asymmetric_module_declares_a_packed_zero_point` iterated what was
left. The review built a shard with four packed modules, one without its
`weight_zero_point`; the test passed and reported three.

**A filter applied to the population being audited removes exactly the rows the
audit exists to see.** That is task 0023's omitted layer — "totals were derived
from the records supplied and then re-summed from the same records" — in a
different file.

The inventory keeps **every** discovered packed module now, `incomplete()` is the
finding rather than a silent drop, and `importable()` is the separate, explicitly
named subset the import tests use. `missing_companions` is a pure function so the
audit's decision can be tested against a table that **contains** the case the
real artifacts do not supply, and a mutation
(`missing-companions-always-empty`) proves that test load-bearing.

#### 4 (P2). The driver was described and not committed

Experiment 0005 said the driver was "reproduced in the task record's result
section". It was not, in either commit. **The mutation names are not the
measurement; the exact substitutions are.** And the contract promised
substitutions "repeated in both directions" — the first run did each once, and
neither the record nor I said so.

The driver is tracked now, at
`tools/experiments/0005-mutations.py` (retired 2026-09-14),
with every substitution verbatim. Each verdict is repeated **three times in both
directions** — mutant against the mutated tree, control against the restored one
— and a lane that disagrees with itself is reported as nondeterministic rather
than counted.

#### 5 (P2). The stated reason for not measuring the sign was false

The record said a distribution centred near zero makes `mean|mean(q) - z|` and
`mean|mean(q) + z|` indistinguishable. That ignores the correlation between a
group's code mean and its own zero point — **which is the premise of the
measurement two paragraphs earlier.** A record contradicting a fact it already
contains is the third instance of that class in this workspace.

The sign is a fifth candidate in the measurement now, and it separates further
than any misassignment does: **2.18 to 2.57** against the pinned **0.52**,
measured on all four tensors.

### Changed shared owners and consumers

`moxie_format::compressed_tensors` is the sole owner and the only production
module whose behaviour changed. `moxie-format/src/affine.rs` was **not** changed:
the canonical descriptor already carried `ZeroPoints::PerGroup`, which is what
"what is missing is a verified source contract, not a canonical capability"
meant when task 0018 wrote the refusal.

| Change | Why |
|---|---|
| `PackQuantizedSpec::symmetric: bool` → `zero_points: ZeroPointSource` | The boolean said only that *some* zero points exist; the importer has to know **where**. `Symmetric` and `PackedAlongOutput` name conventions. |
| `TensorTriple` → `SourceTensors`, with `zero_point: Option<PackedZeroPoints>` | A "triple" that can be four tensors is a name that has stopped being true. |
| `triple_entries` → `source_entries(header, module, zero_points)` → `SourceEntries` | It now resolves a fourth entry, validates its dtype, and **requires the declared serialization and the tensor index to agree in both directions**. |
| `decode_zero_points` | The new decode. Refuses before any code is unpacked, so a malformed zero point does not cost a whole tensor's work first. |
| The `Error::Unsupported` refusal of asymmetric sources | Deleted. |

Consumers updated: `moxie-storage/tests/gemma4_import.rs` (the symmetric
real-artifact lane, unchanged in behaviour), `moxie-format/tests/import_allocation.rs`.
Two comments that recorded the old refusal as a *reason* were corrected rather
than deleted — `moxie-models/src/laguna.rs` and `moxie-cli/tests/laguna.rs` —
because the reason changed and the conclusion did not: **Laguna's graph still
declares BF16 tensor requirements, since nothing executes a canonical INT4
tensor.**

Base `e122de3` (this contract). Implementation is the commits after it.

### Commands and results — passed, failed and skipped separately

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | **passed** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **passed** |
| Device-lane clippy (`--features moxie-executor/driver`) | **passed** |
| CUDA-lane clippy (`--features cuda`) | **passed** |
| `cargo test --workspace --locked --offline` | **947 passed, 0 failed**, against a baseline of **930** re-measured at `e122de3` before implementation (943 after implementation, 945 after the first review) |
| `cargo test --workspace --locked --offline --features moxie-executor/driver` | **982 passed, 0 failed** (task 0023 recorded 965; +17 is exactly this task's new tests) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified. Re-run after the corrections, because widening `Error::InvalidArtifact` touches every crate including the device lane |
| `cargo xtask spec-check` | **passed**, 10 documents |
| `cargo xtask arch-check` | **zero failures**, 79 rejected fixtures, 21 accepted, 13 rules |

**Nothing failed.** **Nothing was skipped**: both real artifacts are present on
this machine and every artifact lane ran. The four artifact tests each print
`SKIP` with a reason when their checkpoint is absent, so a fresh clone stays
green — that path was exercised while the tests were being written and is not
what ran here.

Seventeen new tests: seven in `moxie-format`'s own module, one
`import_allocation_asymmetric`, **three** `import_allocation_failure`, **six** in
`moxie-storage/tests/asymmetric_int4_import.rs`.

**Mutation measurement: 24 of 24 caught, 0 survivors, 0 unstable, 0 invalid
controls, 0 skipped**, each verdict repeated **three times in both directions**,
driver committed and its verdict rule self-tested
([experiment 0005](../evidence/experiments/0005-asymmetric-int4-zero-point-assignment.md),
its driver (retired 2026-09-14)). The first
measurement was 16 of 16 against a suite with **five** holes in it, and the
second 21 of 21 against a suite with three. **Eight mutations were added after a
review — five then three — and every one of them is caught by exactly one
lane**; seven by a check a finding added, and `bf16-rounding-truncates` by a
pre-existing BF16 test, because an existing lane acquiring a new regression is
not a lane a review created. A battery that is complete against the suite it was
written for says nothing about the suite's holes.

### Measured effect and uncertainty

**The zero-point lane assignment is measured, not assumed.** `mean |mean_code −
z|` in codes, on four tensors across the two artifacts:

| Artifact | Module | pinned | lane-reversed | block-major | block reversed | sign-flipped |
|---|---|---:|---:|---:|---:|---:|
| Laguna | `layers.1.mlp.experts.0.down_proj` | **0.5236** | 1.4900 | 1.4853 | 1.4817 | 2.1797 |
| Laguna | `layers.1.mlp.experts.0.gate_proj` | **0.5183** | 1.5661 | 1.5652 | 1.5670 | 2.2912 |
| Qwen3.8-27B | `layers.11.self_attn.k_proj` | **0.5163** | 1.7629 | 1.7809 | 1.7839 | 2.5735 |
| Qwen3.8-27B | `layers.11.self_attn.v_proj` | **0.5300** | 1.5751 | 1.5741 | 1.5776 | 2.2888 |

Declared thresholds were 1.0 and 1.2, written into this contract before the
measurement; the margin is 2.8–4.9×. The **sign** column was added after the
review and is held to the same thresholds — this contract's original claim that
the statistic could not separate a sign was wrong, and wrong for a reason the
measurement's own premise contradicts.

**Uncertainty, stated:** this rules out the four alternatives the same bytes
permit. It does not establish the **code** word's lane order, which stays exactly
where task 0018 left it, because all eight lanes of a code word fall inside one
scale group.

**112,640 reconstructed values are checked against two quantities**, because
there are two and they are not the same one. Bitwise against document 03's
canonical FP32 `(q - z) * s` over the source's own bytes; and against the
source's own arithmetic **with the BF16 boundary its reference applies**, which
the canonical value rounds to exactly. The boundary moves **27,501** of the
112,640, a count the test prints and requires to be nonzero. The zero points
themselves are checked against a transcription that follows the library's
*unpack procedure* rather than this importer's closed-form index.

That is ADR 0018's bit-identical repack at tensor scale: codes, zero points and
scale bytes preserved exactly. **It is not a quality claim** and no paired output
against any released model exists.

**No performance claim.** Nothing was timed and the import path has no
duration field.

### Four things the work turned up that were not in the contract

1. **A module's four tensors need not share a shard.** Laguna keeps them
   together for 34,739 of 34,740 modules; Qwen3.8-27B for **none** of its 256.
   `source_entries` takes one `Header` by design and cannot see across that
   split. An index resolver is M3 item 1's manifest work; this task records the
   gap and its artifact lane resolves per tensor in the test.
2. **Laguna's shard headers exceed the default header budget.** 140,989 tensors
   means about a megabyte of header per shard and a 16.5 MB admitted peak
   estimate against `HeaderBudget::DEFAULT`'s 8 MiB — a default calibrated on
   Gemma 4's 17–64 KB headers. The default refusing it is the budget working.
   The artifact lane states 64 MiB for a read-only inspection; **the default is
   unchanged**, and any production path opening this artifact must state one.
3. **The existing allocation test's own docstring caught a mistake immediately.**
   Adding the asymmetric allocation count as a second test in
   `import_allocation.rs` made the symmetric fixed cost read 3, 13 and 25 as the
   two tests interleaved on the global counter — which that file's first
   paragraph says will happen. It is its own executable now. A measurement taken
   while something else allocates is not a measurement.
4. **A refusal is not reachable from a sweep of successful imports**, and the
   whole crate's refusal paths aborted under memory pressure. Found by the
   review, fixed at the shared error type. The full account is in the
   corrections above; it is listed here because it is the single largest thing
   this task learned and it was not in the contract.

### Deleted and replaced paths

- The `Error::Unsupported { capability: "asymmetric pack-quantized import" }`
  refusal and the negative fixture asserting it. Replaced by fixtures asserting
  the refusals that remain: a spec and a payload that disagree, an index and a
  declaration that disagree, a zero point of the wrong dtype, and every
  mis-declared zero-point shape including the **unpacked** one.
- No bridge, no temporary path, nothing deferred behind a flag.

### Remaining blockers and the next bounded task

- **Nothing executes a canonical INT4 tensor.** W4A16/W8A16 is M3 item 3 and no
  kernel exists. This is the largest remaining gap in M3 and the reason no
  capability row moved out of "not implemented" for execution.
- **Group-128 symmetric INT4 with `actorder: "static"`** is M3 item 2's
  remainder. Both local candidates declare it, and what `static` means for
  logical column identity must be read from the pinned exporter first — document
  03 forbids ignoring a permutation, and a permutation folded into the weights
  changes which activation column each canonical column means.
- **AutoRound / AutoGPTQ packing** is the second serialization, still not
  implemented.
- **Laguna's attention tower** is unchanged: `softplus` output gating and the
  yarn rotary ramp are still gaps, and no Laguna layer is composable.
- **The fused/per-expert role mapping** the Laguna record records as open
  question 1 is untouched: this task imports at tensor granularity and the
  canonical form needs no fused layout.
- **O2 and O5 bound what any of this may be called.** A bit-identical repack is
  v1's quality definition, not evidence about output, and nothing may be written
  under a checkpoint root without a task naming artifact, revision, expected
  size and retention. **This task wrote nothing.**
- **The rest of the workspace's refusal paths still allocate their prose.**
  `Error::InvalidArtifact` can now be built without allocating and
  `moxie-format`'s import path does; the other 40 files still call `format!`,
  which is correct wherever an allocation failure is not the context — and is
  an open question wherever it is. The three sibling variants
  (`Unsupported`, `Numerical`, `InvalidRequest`) still carry `String`. Deciding
  how far that goes is a bounded task of its own, not something to widen into
  this one after the fact.
