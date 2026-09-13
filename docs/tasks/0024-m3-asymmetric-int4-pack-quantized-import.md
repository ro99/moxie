# Task 0024 — M3 item 2: asymmetric INT4 `pack-quantized` import, zero points along the output axis

Status: **proposed**.

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

*(empty until the work is done)*
