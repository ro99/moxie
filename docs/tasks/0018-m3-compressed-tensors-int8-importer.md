# Task 0018 — M3 compressed-tensors pack-quantized importer

Status: **accepted by the owner, 2026-09-12, within its declared import-only
scope.** Contract and
[ADR 0015](../decisions/adr/0015-serde-json-for-safetensors-headers.md)
committed at `db9529e`, before implementation.

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

Implementation follows contract `db9529e`. No CUDA, kernel, graph, model or
device execution path changed, and **no byte was written under any checkpoint
root**. The shared owners are:

- **`moxie-format::safetensors`** owns the container: an eight-byte
  little-endian length, a bounded JSON header, and per-tensor
  `{dtype, shape, data_offsets}`. `Header::prefix_len` reads and bounds the
  declared length **before** the parser sees anything, so a hostile `u64` cannot
  cause a large allocation ahead of validation. Every entry's declared range
  must equal `product(shape) * dtype_bytes` exactly — a tensor that merely
  *fits* is refused, because every later read derives its length here. Overlap
  is checked in one pass over start-sorted spans; unknown dtypes are a typed
  error rather than a skipped tensor. It never opens a file.
- **`moxie-format::compressed_tensors`** owns the `pack-quantized` decode and
  produces the **existing** canonical `AffineTensor`. The rebias reuses the
  accepted `affine::rebias_code`; the reconstruction equation, the descriptor
  and every existing expectation are untouched.
- **`moxie-storage::Shard`** opens a shard, reads its header, and serves
  positioned bounded reads under the existing `ByteBudget`. No `mmap`. Opening a
  5 GB shard costs its header.
- `moxie-format` gained `serde_json` ([ADR 0015](../decisions/adr/0015-serde-json-for-safetensors-headers.md)),
  declared in the `arch-check` allowlist. A new negative fixture proves a second
  crate taking it is rejected: the header contract has one owner.

### What the artifact established, and what it could not

The packing was verified **before** the contract was authored and again by the
implementation. In `model-00002-of-00007.safetensors`,
`model.language_model.layers.12.self_attn.q_proj` carries `weight_packed` `I32`
`[8192, 1344]`, `weight_scale` `BF16` `[8192, 168]` and `weight_shape` `I64[2]`
holding `(8192, 5376)`: `5376 / 4` and `5376 / 32`, as the equations require. A
byte histogram of one packed row is dense in `64..191` and sparse at both ends,
**mean raw byte 127.4** — the codes are biased-unsigned, which `raw - 128`
decodes, and two's-complement bytes would have been edge-heavy instead.

**The lane order is cited, not verified.** All four lanes of a word fall inside
one 32-column scale group, so the artifact's own data cannot distinguish it: a
reversed order reconstructs a different but equally plausible weight. The order
comes from the pinned reader (`compressed_tensors.cpp:286`). Closing it needs
paired output against the released model, which is O2 evidence and M3's later
kernel work. **Nothing here is a quality claim, and a successful import is not
one.**

### Real-artifact evidence, read-only

Re-derived by this reader rather than restated from the inventory:

| Measure | Value |
|---|---|
| Shards parsed | 7, each covering its payload exactly |
| Tensors | **2008** — 1188 `BF16`, 410 `I32`, 410 `I64` |
| Total tensor payload | **35,089,877,112 B** |
| Modules imported | `layers.0.self_attn.q_proj` `(8192, 5376)`, `layers.0.self_attn.o_proj` `(5376, 8192)`, `layers.0.mlp.down_proj` `(5376, 21504)` |
| Bytes read | **216,416,304** |
| Codes observed | the full `[-128, 127]`, including `-128`, which document 03 forbids a decoder from rejecting |
| Scale dtype | BF16, **from the tensor header** — the config declares `scale_dtype: null` |

Every reconstructed value on the sampled rows is finite, and the codes span the
range rather than being a constant. These tests **skip with a message** when the
artifact is absent, so a fresh clone stays green.

### Verification

| Gate | Exact command / result |
|---|---|
| Host workspace | `cargo test --workspace --locked --offline`: **689 unit/integration + 9 doctests passed**, 0 failed, 0 ignored (681 + 9 as first submitted) |
| Structural limits | `cargo test -p moxie-format --locked --offline safetensors`: rank, name length, tensor and metadata counts each refused while parsing |
| Header cost | `cargo test -p moxie-storage --test header_budget --locked --offline -- --nocapture`: **1 passed**; measured peaks within their admitted estimates, table above |
| Importer | `cargo test -p moxie-format --locked --offline`: **102 passed**, including the exhaustive code/lane pairs, tails, granularities, scale dtypes, axis mismatches and header rejections |
| Import allocation | `cargo test -p moxie-format --test import_allocation --locked --offline -- --nocapture`: **1 passed**; 3 allocations at 4, 64 and 512 rows |
| Real artifact | `cargo test -p moxie-storage --test gemma4_import --locked --offline -- --nocapture`: **3 passed**, figures above |
| Host clippy | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`: passed |
| Format / diff | `cargo fmt --all -- --check`; `git diff --check`: passed |
| Specification | `cargo xtask spec-check`: passed, 10 documents unchanged |
| Architecture | `cargo xtask arch-check`: **74 rejecting + 21 accepted fixtures**, 12 rules. The undeclared `serde_json` edge was **rejected before it was declared**, which is the allowlist working; the new `shared-takes-serde-json` fixture keeps a second crate from taking it |
| Device workspace | the host command with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda`: **705 + 12 doctests passed**, 0 failed |
| Device clippy | the clippy command with the same features: passed |
| Real GPU | `cargo xtask-cuda test-gpu`: **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |

| Hardware | UUID |
|---|---|
| RTX 5060 Ti / SM120 | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` |
| RTX 3090 / SM86 | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` |
| RTX 3090 / SM86 | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` |

**The GPU result is unchanged, which is expected: this task adds no device
behaviour.** It is regression evidence for accepted work, not evidence for this
task's mechanism.

**Unmeasured / not run:** no topology, Compute Sanitizer, **model quality** or
paired performance gate. No W8A16 or W4A16 kernel, no execution, no repack, no
manifest write, no prepared layout, no AutoRound/AutoGPTQ adapter, no vision.
O1–O7 remain open and none was resolved or relied on.

**O5 was respected exactly as the contract stated it.** The work opened files
under `/fast/models` for positioned reads and wrote nothing anywhere: no
converted artifact, no manifest, no prepared layout, no download. Nothing under
either checkpoint root was created, modified or deleted.

### Asymmetric sources

Refused with a typed `Unsupported` naming what evidence would close it. The
pinned reader rejects asymmetric pack-quantized
(`compressed_tensors.cpp:204`), no local artifact is one, and document 03
forbids guessing a zero offset "from a suffix". The canonical descriptor already
carries `ZeroPoints::PerGroup`, so what is missing is a **verified source
contract**, not a canonical capability. A test asserts the refusal is
`Unsupported` and not `InvalidArtifact` — the source is fine; we cannot read it.

### Independent review corrections

Five rounds, and the last four were all the same defect wearing different
clothes: **a resource bound asserted rather than established.** The first round
found five defects plus a documentation contradiction. The second found the
header budget measuring the wrong quantity. The third found the replacement
bound fitted to sampled shapes and defeated it twice. The fourth found the
derivation's minimum-cost term invalid — the parser accepted a cheaper encoding
than the derivation assumed — and its slopes blind to collection-capacity
boundaries. The fifth found a rejection path that spent memory on its way to
refusing. **Every finding was reproduced before any change and all are fixed.**
Every round confirmed the ownership split, the deferral of execution and
quality, and the lane-order disclosure.

**P1 — the importer accepted incompatible axes.** `import` took byte slices, so
declared shapes were invisible to it, and `triple_entries` never cross-checked
them against the logical shape. Reproduced: a logical `[2, 64]` imported
successfully from a packed tensor declared `[16, 2]` instead of `[2, 16]` and a
scale declared `[1, 4]` instead of `[2, 2]` — equal byte counts concealed
transposed axes and the weights were silently wrong. `TensorTriple` now carries
`packed_shape` and `scale_shape`, and `import` checks both against the logical
shape before decoding.
`declared_shapes_that_are_byte_compatible_but_axis_incompatible_are_refused`
covers four wrong packed shapes and four wrong scale shapes, each with exactly
the right byte count, plus a control that imports and the two scale shapes a
per-channel source may legitimately use (`[out]` and `[out, 1]`).

**P1 — a payload-sized allocation still aborted the process.** The importer's
own reservations were fallible, but it then called `affine::pack_row`, which
allocated a row through the infallible allocator. The review injected failure
into a seventeen-byte row allocation and got `SIGABRT`. `pack_row` is now
fallible, and the importer does not call it: `pack_row_into` writes into the
destination reserved once for the whole tensor, so the allocation that aborted
no longer exists. `import_allocation` — its own executable, because the counter
is a global allocator — measures **3 allocations at 4, 64 and 512 rows**. Its
control, restoring the per-row call, measures **7, 67 and 515**.

**P2 — the header allocation bypassed the reader budget, and the first fix
budgeted the wrong quantity.** `open_with_budget` originally allocated and read
the whole header without consulting `ByteBudget`; the first review opened a
shard with a sixteen-byte budget while observing a 65,575-byte allocation.
`ByteBudget` caps *payload slicing* and a header is a different resource — read
whole because it must be parsed whole, and partly retained for the shard's life
— so it gained its own `HeaderBudget`, checked along with the file length
**before** anything is allocated.

**A second review round found that still measuring the wrong thing**, and was
right: the budget capped the *serialized* length, while parsing allocates an
entry list, a key string and a shape vector per tensor, a temporary span list,
and the retained maps. Reproduced: a 54,899-byte serialized bound admitted a
header costing 353,105 bytes of peak heap and 163,275 retained.

`HeaderBudget` became denominated in **peak heap**, with admission against an
estimate. **A third round defeated that too**, with two shapes the five sampled
ones did not cover: many tiny `__metadata__` entries, and one tensor of 65,537
dimensions. Both reproduced. That is the same mistake twice — a factor fitted to
sampled shapes is not a bound, and can always be beaten by a shape that was not
sampled.

**The bound is derived now, not fitted.** A header spends its bytes on
constructs, and the peak is linear in that split: spending `b_i` bytes on
construct `i` with `Σ b_i ≤ S` costs `Σ b_i·r_i ≤ S · max r_i`. So the sound
bound is the **largest per-construct ratio**, and the work is enumerating the
constructs. Marginal costs, measured two points apart so they are slopes rather
than whole-header averages:

| Construct | Min serialized | Peak heap | Ratio |
|---|---:|---:|---:|
| Tensor entry | 55 B | 297 B | 5.4 |
| `__metadata__` entry | 9 B | 85 B | **9.5** |
| Shape dimension | 2 B | 24 B | **12.0** |
| Tensor-name byte | 1 B | 2 B | 2.0 |

The shape dimension is the largest and is **unbounded per tensor**, which is
exactly why no multiplier could survive it. It gets a **structural limit**
instead: `MAX_RANK = 8`, refused *while parsing* so the dimensions are never
allocated. Capped, a tensor's dimensions cost at most `8 × 24` peak against at
least 69 serialized bytes — a ratio of 2.8. `MAX_NAME_BYTES`, `MAX_TENSORS` and
`MAX_METADATA_ENTRIES` join it as explicit contract rather than arithmetic
accident.

With the rank bounded the largest ratio is the metadata entry at 9.5, and the
factor is **16** — about 1.7× above it for allocator and layout variation.

`a_header_costs_no_more_peak_heap_than_its_admitted_estimate` measures **each
construct at its own worst shape**, which is what the derivation rests on, plus
the earlier five shapes and the review's counterexamples:

| Header | Serialized | Peak | Retained | Admitted | Ratio |
|---|---:|---:|---:|---:|---:|
| 1 tensor | 61 | 1,417 | 916 | 9,168 | 23.23 |
| 1,000 tensors | 57,682 | 353,888 | 162,320 | 931,104 | 6.14 |
| 5,000 tensors | 301,682 | 2,231,602 | 817,344 | 4,835,104 | 7.40 |
| 8,000 minimal tensors | 484,682 | 2,869,664 | 1,305,598 | 7,763,104 | 5.92 |
| **20,000 metadata entries** | 180,078 | 1,900,826 | 1,720,311 | 2,889,440 | **10.56** |
| 500 × 900-byte names | 477,291 | 1,526,141 | 533,392 | 7,644,848 | 3.20 |
| 4,000 tensors at `MAX_RANK` | 296,682 | 1,615,968 | 780,180 | 4,755,104 | 5.45 |

The 65,537-dimension tensor is **refused**, and refusing it costs 131,507 B —
essentially the input slice — against the 1.7 MB accepting it cost before.

Two controls, each restored afterwards: removing `MAX_RANK` makes the test fail,
and dropping the factor to 9 makes it fail on the metadata case, naming it.

**A fourth round found two more holes in the same derivation, and both are
fixed.**

*serde accepted a second encoding.* The derived `Deserialize` for a struct
accepts a **positional array** as well as an object, and `deny_unknown_fields`
does not change that: `{"0":["U8",[0],[0,0]]}` was accepted at **22.9 serialized
bytes per entry** against the object form's 55. That halves the minimum-cost
term the whole derivation rests on, and the review's counterexamples — 4,097 and
8,193 array-form tensors — beat the bound because of it. `RawEntry` is
hand-written now with only `visit_map`, and `deserialize_map` rather than
`deserialize_struct`, so an array is a typed error; duplicate and unknown fields
stay refused. The safetensors format specifies an object, so accepting a second
encoding bought nothing and cost the bound.

*The slopes averaged over the capacity boundary.* A `Vec` that has just doubled
holds its old allocation beside the new one, so the peak per entry is worst just
past a power of two — which is why the review's counts were 4,097 and 8,193, and
why two-point slopes measured too low. The regression now includes both
constructs at `2^k + 1` for k in {6, 8, 10, 12, 13}. Re-measured across every
such boundary from 64 to 8,192, the worst ratio is **11.69** (metadata at 65),
and at scale **10.54**; the factor of 16 stands with margin.

*And the metadata limit was enforced too late.* `MAX_METADATA_ENTRIES` was
checked after `next_value` had built the whole map, so a 100,000-entry map was
allocated in full and then rejected — stating a limit while paying for its
violation. It is counted inside a `Metadata` visitor now, like `MAX_RANK`.

**A fifth round found the same class once more, on a rejection path.** The
duplicate `__metadata__` check ran in `Header::parse`, after the whole header
had been deserialized, so a header repeating that key accumulated one entry per
declaration and only then failed — spending the memory on its way to an error.
Reproduced at **1,204,315 B against an admitted 1,189,104**. It is refused in
the top-level visitor now, before the value is read and before the entry is
appended: the excess over the input buffer is **454 B at both 4,097 and 8,193
declarations**, flat rather than growing with the abuse. Its control restores
the late check and fails with the original number.

**A resource bound has to cover the rejection paths too**, and an `is_err()`
assertion cannot see the difference — the regression measures the refusal.

The duplicate *tensor name* check stays in `parse`, deliberately: a repeated
name costs a full tensor entry each time, so it is bounded by the ordinary
tensor ratio the derivation already covers, and detecting it earlier would need
a second set of every name — raising the name-byte ratio to pay for a case that
is already paid for.

| Boundary case | Serialized | Peak | Admitted | Ratio |
|---|---:|---:|---:|---:|
| 65 metadata entries | 663 | 7,750 | 18,800 | 11.69 |
| 4,097 tensors | 246,599 | 2,142,205 | 3,953,776 | 8.69 |
| 4,097 metadata entries | 36,951 | 389,478 | 599,408 | 10.54 |
| 8,193 tensors | 496,455 | 4,287,421 | 7,951,472 | 8.64 |
| 8,193 metadata entries | 73,815 | 777,702 | 1,189,232 | 10.54 |

The real shards' headers are 17–64 KB, so their estimated peaks are under
800 KB against the 8 MiB default.

**P2 — duplicate JSON keys bypassed validation.** Deserializing into
`BTreeMap<String, Value>` collapsed duplicates before any check ran: two tensors
sharing a name became one, and `"dtype":"FP8","dtype":"U8"` parsed as `U8`, so
an unsupported dtype could be smuggled past the check that exists to refuse it.
Both reproduced. The header now deserializes through a visitor that preserves
declaration order, refuses a repeated tensor name and a repeated `__metadata__`,
and builds each entry **directly** into its struct so serde's own duplicate-field
rejection applies. `deny_unknown_fields` refuses a field the schema does not
define. Six cases plus a control.

**P2 — the "every code in every lane" claim was not met.** The INT8 test used
three row shifts over four lanes, reaching 768 of 1,024 code/lane pairs, and its
tracker counted codes alone so the gap was invisible; INT4 reached 80 of 128.
Both now use one row per lane offset and a **two-dimensional** tracker asserting
`256 x 4` and `16 x 8` pairs exactly.

**Documentation contradiction.** The support matrix still carried a
"compressed-tensors integer import — **not implemented**" row beside the new
passing one, and the README still said nothing imports checkpoint data. Both are
corrected, and both now say plainly that an import is not an execution.

### Owner acceptance, 2026-09-12

The owner accepted task 0018 after five rounds of independent review found no
remaining blocking correctness or architecture issue. The final round verified
that a repeated `__metadata__` declaration is refused before the second value is
read, measuring **454 bytes beyond the input buffer** at 4,097, 8,193 and 16,385
declarations, and that duplicate tensor-name rejection stays inside the admitted
bound. All five original findings and every header-budget correction are closed.

Independent validation reproduced 689 host tests + 9 doctests including the
real-artifact imports, host clippy, formatting, specification and diff checks,
and clean-archive architecture checks at 74 rejecting + 21 accepted fixtures.
The reviewer did not rerun the GPU, quality or performance gates; the GPU
figures in this record are from this agent's runs and are regression evidence
for already-accepted device behaviour, since this task adds none.

**This acceptance closes task 0018 only.** It states, in the owner's terms:
**M3 and M1.5 remain open, no checkpoint executes, and source lane-order
verification remains outstanding.** An import is not an execution, and a
successful import is not a quality claim.

### Deletion

Nothing was deleted or superseded. `moxie-storage::Artifact`'s refusal to read
an affine tensor still stands: that path is the canonical manifest's, and this
task imports from a **source** container, which is a different thing and
deliberately shares none of its machinery.

### Remaining blockers and next bounded task

M3 is **not** closed. Importing a tensor is not executing one: the W8A16 and
W4A16 shared paths, the bounded inspector/repacker with atomic publish, the
canonical manifest write, and the AutoRound/AutoGPTQ packing adapter are all
still outstanding, and all four are M3's. M1.5 is **not** closed: the Gemma 4
artifact still cannot run.

The next bounded task is the shared **W8A16 execution path**, which is what
turns a canonical INT8 tensor into a result — with the bounded reference
dequantization as the correctness oracle and explicitly not as the claimed fast
path, per document 03.
