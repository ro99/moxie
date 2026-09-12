# Task 0018 — M3 compressed-tensors pack-quantized importer

Status: **proposed**.

## Identity and authority

- Task ID / milestone / owner: 0018 / **M3 canonical integer import** / implementation
  agent; acceptance belongs to the owner. **This task does not close M3, does not
  close M1.5, and does not make any checkpoint executable.**
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `22f5733`
  (task 0017 corrected and pushed). Working tree clean at authoring.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and
  `tests/p2p/` are preserved and are not source evidence.
- Selected as the next bounded task by
  [the task 0017 handover](../handovers/2026-09-12-task0017-per-layer-kv-retention.md),
  which recorded that shape is no longer what blocks the Gemma 4 artifact — weights are.
- Repairs the first M3 deliverable: document 03's "Import pinned compressed-tensors
  group-128 symmetric and group-32 asymmetric INT4, group-32 INT8". R25.
- Required documents read: AGENTS.md, README, 01–09, the owner-gate register,
  `docs/README.md`, the TASK/ADR/HANDOVER templates,
  [the bring-up record](../models/gemma4.md),
  [the inventory](../evidence/checkpoint-inventory.md) and
  [the candidates](../evidence/quantization-candidates.md). Document 03's
  "Affine integer v1", "Import priority and quality boundaries" and "Converter
  and prepared layouts" sections are the normative text.
- Sources inspected at the frozen legacy commit:
  `src/platform/compressed_tensors.cpp:186` (pack-quantized layout validation —
  `weight_packed` must be I32, `weight_scale` BF16, `weight_shape` two I64),
  `:208` (`values_per_word = 32 / bits`, packed columns are
  `ceil(logical_columns / values_per_word)`), `:232` (`weight_shape` payload
  decode), and `:286`–`:300` (**the decode itself**: `word_index = row *
  packed_columns + column / values_per_word`, `lane = column % values_per_word`,
  `raw = (word >> (lane * bits)) & mask`, `quantized = raw - (1 << (bits - 1))`).
- Owner gates: O1–O7 remain open. **O5 governs this task directly** and the stop
  point is defined below. O2 is untouched: nothing here changes a quantized value.

## Bounded deliverable

**One concrete outcome:** a bounded, strict reader that turns a compressed-tensors
`pack-quantized` tensor triple — `weight_packed`, `weight_scale`, `weight_shape` —
plus its `quantization_config` parameters into the **existing canonical**
`moxie_format::affine::AffineTensor`, with nothing written to disk.

**Sole owning shared component:** `moxie-format` owns the safetensors header
schema and the pack-quantized decode. `moxie-storage` owns reading bounded byte
ranges out of a shard file. There is no second decoder and no runtime decode
path: the output is the canonical tensor the accepted `affine` module already
reconstructs.

**Allowed production files:**

- `crates/moxie-format/src/safetensors.rs` (new): header length, JSON header,
  per-tensor dtype/shape/offsets, and whole-file consistency validation.
- `crates/moxie-format/src/compressed_tensors.rs` (new): the pack-quantized
  layout contract and the import into `AffineTensor`.
- `crates/moxie-format/src/affine.rs`, `scale.rs`: only if the import needs a
  constructor or accessor that does not exist. **No change to the reconstruction
  equation, the descriptor fields or any existing test's expectation.**
- `crates/moxie-format/src/lib.rs`, `crates/moxie-format/Cargo.toml`,
  `Cargo.lock`: the module wiring and the `serde_json` dependency (ADR 0015).
- `crates/moxie-storage/src/lib.rs`: bounded range reads.
- Their tests and fixtures, architecture allowlist/negative fixtures, and tracked
  task/ADR/evidence/handover records.

**Explicit non-goals and forbidden shortcuts.** No W8A16 or W4A16 kernel and no
execution path — document 03 puts those in the same milestone but they are not
this task. No canonical manifest **writing**, no repack, no conversion, no
publish, no prepared layout. No model graph binding, no tensor-role mapping to
`moxie-models`, no vision. No AutoRound/AutoGPTQ packing adapter — document 03
orders compressed-tensors first and "add another serialization only after actual
metadata requires it". No `mmap`, no remote code execution, no Python. No
dequantization fallback presented as an execution path. **Importing a tensor is
not model support and must not be described as one.**

**Asymmetric sources are refused, explicitly.** The pinned reader
(`compressed_tensors.cpp:204`) rejects asymmetric pack-quantized, no local
artifact is asymmetric pack-quantized, and document 03 forbids guessing a zero
offset "from a suffix". The importer therefore accepts symmetric and returns a
typed `Unsupported` for an asymmetric source naming what evidence would be
needed. The canonical descriptor already carries `ZeroPoints::PerGroup`, so this
is a missing *verified source contract*, not a missing canonical capability.

**Existing consumers and second-shape proof.** `AffineTensor::reconstruct` and
its exhaustive code tests are the existing consumer and must keep passing
unchanged. The importer gets two independent shapes: synthetic fixtures covering
both widths and both groupings, and **real tensors from the local artifact**.

## Contract before implementation

### Safetensors container

`u64` little-endian header length `n`, then `n` bytes of JSON, then the payload
at `8 + n`. Each entry is `{"dtype", "shape", "data_offsets": [begin, end)}`,
with offsets **relative to the payload start**. `__metadata__` is a
string-to-string map and is not a tensor.

Validated before any payload byte is read, with checked arithmetic throughout:
`8 + n` does not exceed the file length; every `begin <= end`; every `end` is
within the payload; `end - begin` equals `product(shape) * dtype_bytes` exactly;
no two tensors overlap; the payload is exactly covered or the unused bytes are
reported; the header parses as an object whose every value has the three required
fields. A header length larger than a declared bound is refused **before**
allocating for it. Unknown dtypes are a typed error, not a skip.

This is untrusted input from a downloaded artifact. Nothing is inferred from a
name; every field is read and checked.

### Pack-quantized layout

From the pinned reader, with `bits` from `quantization_config`:

```text
values_per_word = 32 / bits                       (4 for INT8, 8 for INT4)
packed_columns  = ceil(in_features / values_per_word)
word_index      = o * packed_columns + k / values_per_word
lane            = k % values_per_word
raw             = (word >> (lane * bits)) & ((1 << bits) - 1)
q               = raw - (1 << (bits - 1))
```

- `weight_packed` is `I32`, shape `[out_features, packed_columns]`.
- `weight_scale` matches the declared granularity: `[out_features, 1]` per
  channel, `[out_features, ceil(in_features / group_size)]` per group.
- `weight_shape` is `I64[2]` holding logical `[out_features, in_features]`, and
  **it is the authority** for the logical shape; the packed shape is checked
  against it rather than used to derive it.
- The **scale dtype comes from the tensor header**, never from the config.
  Document 03 requires this and the artifact is exactly why: it declares
  `scale_dtype: null` while its headers say BF16.
- `q = raw - (1 << (bits - 1))` is document 03's rebias, not requantization, and
  reuses the accepted `affine::rebias_code`.

**Verified against the artifact before this contract was written.** In
`model-00002-of-00007.safetensors`,
`model.language_model.layers.12.self_attn.q_proj` has `weight_packed` `I32`
`[8192, 1344]`, `weight_scale` `BF16` `[8192, 168]`, `weight_shape` `I64[2]`
holding `(8192, 5376)` — `5376 / 4 = 1344` and `5376 / 32 = 168`, as the
equations require. A byte histogram of one packed row is dense in `64..191` and
sparse at both ends, mean raw byte **127.4**: the codes are stored
**biased-unsigned**, which is what `raw - 128` decodes. Two's-complement bytes
would have produced the opposite, edge-heavy histogram.

**A declared risk, because no local check can close it.** All four lanes of a
word fall inside one 32-column scale group, so the artifact's own data cannot
distinguish the lane order: a reversed order would reconstruct a different but
equally plausible weight. The order above is taken from the pinned reader and is
**not** inferred. Closing it needs either a second pinned exporter implementation
or paired output against the released model, which is O2's evidence and M3's
later kernel work. The task must state this rather than imply the import is
verified end to end.

### Resource contract

Every read is bounded and charged. One tensor at a time; a caller asks for a
named tensor and gets its canonical form, and nothing retains a whole shard.
Allocation is fallible on every payload-sized path — a header that declares a
huge tensor must produce `CapacityExceeded`, never an abort. The existing
`ByteBudget` governs shard reads. No `mmap`: a file that changes under a mapping
would make a validated header a lie.

### Cancellation, failure and rollback

Import is pure and synchronous: it reads bytes and returns a value, holding no
transaction and mutating no state. A failure leaves nothing to roll back and no
partial tensor is returned. There is no device work, no lease and no transfer.

### Oracle and predeclared numerical criterion

The oracle is an **independent** decoder written from document 03's equation and
the pinned source, not a call into the reader under test, applied to fixtures
whose codes and scales are chosen rather than read back.

**Exact equality, bit for bit.** The importer performs no arithmetic on scales
and only a rebias on codes, so there is no rounding to bound and no tolerance to
declare. Coverage is exhaustive rather than statistical:

- all **256** INT8 codes and all **16** INT4 codes, each in each lane of a word;
- group tails where `in_features` is not a multiple of the group size;
- per-channel and per-group granularity;
- BF16, FP16 and FP32 scale dtypes read from the header;
- a nonfinite, zero and negative scale each rejected;
- a truncated, overlapping, mis-sized and unknown-dtype header each rejected.

### Application compatibility

Nothing user-visible changes. No sampler, no context target, no cache precision,
no capability claim. The support matrix gains one gate and one row that says
**import**, not support.

## Acceptance

- Full host workspace tests; `cargo fmt --all -- --check`; clippy with
  `-D warnings`; `cargo xtask arch-check`; `cargo xtask spec-check`;
  `git diff --check`. `moxie-format` must still not touch the filesystem —
  that rule exists and `moxie-storage` reads the bytes for it. Add the
  architecture fixture for the new edge if one is introduced.
- The exhaustive code, tail, granularity, scale-dtype and rejection cases above,
  each against the independent oracle.
- **Real-artifact import, read-only.** Import a bounded selection of tensors
  from `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` — at minimum one
  `q_proj`, one `o_proj` whose logical shape is transposed relative to it, and
  one `down_proj` — and assert the declared shapes from the bring-up record, the
  header-derived scale dtype, the packed-column arithmetic, and that every
  reconstructed value is finite. Report the exact bytes read. This test must
  **skip with a clear message**, not fail, when the artifact is absent, because
  the artifact is not in git.
- Confirm the seven-shard header consistency the inventory recorded still holds,
  as a check on the reader rather than on the artifact.
- Allocation: a header declaring an implausibly large tensor returns
  `CapacityExceeded`; no import path aborts the process. Reconcile bytes read
  against the declared budget.
- Device-feature workspace build/tests, device clippy and
  `cargo xtask-cuda test-gpu` on the two 3090 UUIDs and the 5060 Ti UUID with
  `PCI_BUS_ID` ordering. **No device behaviour is added**, so an unchanged GPU
  result is expected and is not evidence for this task.
- Report failed, skipped and unmeasured separately. Topology, sanitizer,
  **model quality** and paired performance gates are **not measured**. No quality
  claim of any kind follows from a successful import.
- Support matrix: add the import gate and a row stating that compressed-tensors
  pack-quantized INT8/INT4 **imports to canonical form**, with execution still
  **not implemented**. Do not amend the Gemma row to suggest the artifact runs.
- Do not mark M3 complete, M1.5 complete, or this task accepted without owner
  review.

**O5 stop point, stated exactly.** This task **reads** bytes from the local
checkpoint root, which
[artifact-roots](../evidence/artifact-roots.md) already authorizes as
inspection. It **writes nothing** under `/models` or `/fast/models`, creates no
converted artifact anywhere, and downloads nothing. The moment the work would
need to persist a canonical tensor, a manifest or a prepared layout, it stops and
reports — that is the bulk-write O5 governs and it is not authorized.

**Other conditions requiring owner direction or rejection.** Stop and report if
the work would need: a second decoder or a model-owned loader; a
method-specific runtime path; a dequantization fallback presented as execution;
a change to the affine reconstruction equation or any existing numerical
expectation; an asymmetric zero-point convention that no pinned source
establishes; or any O2 quality claim.

## Result, filled after work

Not started.
