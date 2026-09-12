# Task 0016 — M1.5 Gemma operation gap and reduced dense graph

Status: **proposed**. This contract precedes implementation.

## Identity and authority

- Owner: implementation agent; independent code review and owner acceptance follow.
- Writable `/home/rodrigo/Developer/moxie`, `main`, clean base `218b96f`.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; preserve its unrelated untracked
  files. Nothing under `/models` or `/fast/models` is read, copied or converted.
- Opens roadmap M1.5 deliverable 5, building on accepted tasks
  0003/0004/0012/0013/0014/0015. R04/R06/R19/R20/R21/R24.
- Authority: AGENTS, documents 01–09, the owner-gate register, TASK/HANDOVER and
  MODEL-BRINGUP forms, and the
  [Gemma 4 bring-up record](../models/gemma4.md) whose inventory this contract
  consumes. Equation references are the frozen legacy sources cited there:
  `src/models/gemma4/gemma4_ops.cpp`, `src/models/gemma4/gemma4_runtime.cpp`,
  `src/platform/numerics.cpp`, `kernels/cuda/detail/backend_kernels.cuh:905` and
  the constant table `include/strata/models/common/model_adapter.hpp:100`.
- O1–O7 stay open. No catalog claim, no quality claim, no conversion, no network
  exposure, no driver change, no checkpoint read.

## Bounded deliverable

One concrete outcome: **six missing operation parameters become explicit shared
semantic operations with independent oracles, and a reduced Gemma-4-like dense
text graph — composed only of those shared operations, over synthetic BF16
weights — generates tokens through the accepted task 0015 service.**

Nothing here executes, reads or supports the Gemma 4 checkpoint. The reduced
graph is a contract fixture. Its output is not model output and may not be
described as Gemma support anywhere, including commit messages and the support
matrix.

### The three separated concerns

1. **Delivered by this task.** The operation gap and the reduced synthetic
   Gemma-like graph, on synthetic BF16 weights, within the host-reference
   profile.
2. **Recorded, not delivered.** The actual artifact's identity, tensor-role
   mapping and equations are in [the bring-up record](../models/gemma4.md).
   This task adds no importer and reads no shard.
3. **Blocked, not claimed.** Every language-model linear in the artifact is
   compressed-tensors INT8 `pack-quantized`. The importer, packed-layout reader,
   canonical repack and W8A16 execution path are **M3**. This task may not add a
   private loader, a dequantization fallback or a model-owned decode path.

### Owners

- `moxie-graph` owns the operation descriptors, their partition rules and their
  state effects. It is the sole owner of the new parameters.
- `moxie-oracles` owns every independent reference and its FP64 transcription.
- `moxie-interp` owns evaluation for both the dense-reference and paged-history
  sources, through its existing single dispatcher.
- `moxie-state` keeps sole ownership of transactions and physical pages. This
  task changes the KV row **width** derivation only; it adds no second store.
- `moxie-models-gemma4` is a new crate containing metadata, tensor roles and
  graph composition **and nothing else**. `MODEL_ALLOWED` already restricts a
  `moxie-models-` crate to `moxie-types`, `moxie-graph` and `moxie-model-api`;
  that restriction is the deliverable, not an obstacle to route around.
- `moxie-cli` remains a composition root that names a diagnostic graph and
  formats events.

### Allowed and forbidden

Allowed: the crates above, the new model crate, architecture fixtures, tests, an
ADR and the living records.

Explicit non-goals and forbidden shortcuts:

- No checkpoint file read, safetensors reader, compressed-tensors importer,
  INT8 unpacking or dequantization. M3 owns all of it.
- No tokenizer, chat template, text stop sequence or EOS handling. M8 owns them.
- No vision tower, image preprocessing or multimodal mask group. M11 owns them;
  the graph must **refuse** an image-token input rather than ignore its mask
  exemption.
- No device kernel for a new operation. Kernel selection must return a typed
  `UnsupportedKernel` for an unqualified operation, never fall back silently.
- No heterogeneous per-layer KV geometry and no sliding-window ring eviction —
  see the reduction below. M4 owns them.
- No 32,768-token claim, no performance claim, no second memory/state/sampler
  owner, no relaxation of an existing numerical gate or tolerance.

### Existing consumers and the second-consumer proof

The task-0015 synthetic diagnostic graph is retained **unchanged in behaviour**
and becomes the independent second consumer of every new parameter, by carrying
the opposite value:

| Parameter | Gemma-like graph | Retained synthetic graph |
|---|---|---|
| Activation | `GeGlu` | `SwiGlu` |
| RoPE pairing | half-split | interleaved |
| RoPE frequency denominator | full head dimension | rotary dimension |
| Rotary dimension | partial on global layers, full on sliding layers | full |
| Attention score scale | `1.0` | `1/sqrt(head_dim)` |
| Key/value heads | fewer than query heads (GQA) | equal to query heads (MHA) |
| Embedding output scale | `bf16(sqrt(hidden))` | `1.0` |
| Logit softcap | `30.0` | none |
| Residual scale | per-layer, MLP residual only | `1.0` on both residuals |

That table is the proof obligation: each new field must be shown to be
**load-bearing**, not decorative. A regression must assert that substituting the
retained graph's value into the Gemma-like graph changes its logits.

Each foundational operation must additionally be consumed at two distinct
shapes, as M1's exit gate requires.

### Temporary paths and expiry

The reduced graph and its CLI selector are explicit diagnostic fixtures. They
expire when M3 supplies the integer importer and a real artifact graph can be
admitted; the bring-up record's integration-proof section is where that
replacement is recorded. Nothing in this task may outlive that as a substitute
for real integration.

## Contract before implementation

### The six operation parameters

Each is an explicit field with **no default**. Every existing construction site
must state the value it already uses, and the existing gates must prove the
results are bit-identical afterwards. A defaulted parameter is a silent
numerical difference between two checkpoints, which is exactly what document 02
forbids for epsilon and what R06 forbids for family mathematics.

1. **`OpParams::GeGlu { width }`**, a new operation beside `SwiGlu`.
   `y_i = bf16(bf16(gelu_tanh(g_i)) · u_i)` with
   `gelu_tanh(v) = 0.5·v·(1 + tanh(0.7978845608028654·(v + 0.044715·v³)))`.
   Source `gemma4_ops.cpp:70` and `numerics.cpp:86`. Partition rule
   `ColumnShardable`, state effect none, matching `SwiGlu`.

2. **`OpParams::Rope` gains an explicit pairing layout and frequency
   denominator.** For `RopeLayout::HalfSplit` with `half = head_dim/2`, angles
   `j` in `0..rotary_dim/2` and `θ_j = pos · base^(−2j / frequency_dim)`:

   ```text
   y[j]        = x[j]·cos(θ_j) − x[half+j]·sin(θ_j)
   y[half+j]   = x[half+j]·cos(θ_j) + x[j]·sin(θ_j)
   y[i]        = x[i]  elsewhere
   ```

   `RopeLayout::Interleaved` retains task 0003's existing `(2j, 2j+1)` form
   exactly. `frequency_dim` is the denominator and may exceed `rotary_dim`;
   Gemma's global layers use `rotary_dim = head_dim/4`, `frequency_dim =
   head_dim`. Source `gemma4_ops.cpp:22`. Angles stay FP64 and narrow once, per
   task 0003. Validate `rotary_dim ≤ head_dim`, `rotary_dim` even,
   `frequency_dim` nonzero, `base > 1`.

3. **`OpParams::Attention` gains an explicit `scale` and `kv_heads`.**
   Scores are `scale · Σ q·k`, FP32 dot, FP32 max-subtracted softmax, FP32
   accumulation, masked keys removed from the sum rather than biased. Query head
   `h` reads key/value head `h · kv_heads / heads`, requiring
   `kv_heads ≥ 1`, `kv_heads ≤ heads` and `heads % kv_heads == 0`. `scale` must
   be finite and positive. Source: `gemma4_runtime.cpp:857` for `scale = 1.0`.
   The existing attention error bound in `moxie-oracles::attention` is retained
   and must be restated with `scale` appearing where the current `1/sqrt(d)`
   does; it may not be weakened.

4. **`OpParams::Embedding` gains an output scale.** `e = bf16(row_t · s)` where
   `s` is a stated BF16 value; Gemma uses `bf16(sqrt(hidden))`. Source
   `gemma4_runtime.cpp:699`.

5. **`OpParams::VocabProjection` gains an optional logit softcap.** When present,
   `y = bf16(bf16(tanh(bf16(bf16(x)/c))) · c)` applied elementwise after the
   projection, `c` finite and positive. Source
   `backend_kernels.cuh:905` and `gemma4_runtime.cpp:1301`. Absent means no cap;
   it is not a cap of zero or infinity.

6. **`OpParams::Residual` gains a scale.** `y = bf16(bf16(a + b) · s)`. Gemma
   applies a per-layer checkpoint scalar on the **MLP** residual
   (`gemma4_runtime.cpp:1230`) and `s = 1` on the attention residual
   (`:1186`). The asymmetry is the fixture; a single global scale is wrong.

### State schema consequence

`KvGeometry.kv_heads` already exists and the paged reference currently requires
it to equal the attention operation's `heads`. That equality becomes
`geometry.kv_heads == op.kv_heads`, so a stored KV row is
`kv_heads · head_dim` wide while the query row stays `heads · head_dim`. Per-layer
page bytes, the admitted backing, the page table and every allocation regression
follow from that single width change and must be recomputed, not adjusted by
hand. The task 0015 control envelope formula
`W = 1,048,576 + 64·S + 4096·N + 256·C·(D+16)·(L+1)` is retained verbatim; `S`
and `N` grow with the new nodes and `D` is now `kv_heads · head_dim`. Checked
arithmetic throughout; no unchecked multiplication of a head count.

### The reduced graph, and exactly what is reduced

The reduced graph preserves, at small size, every text property in the bring-up
record's inventory that the host-reference profile can express:

- mixed layer types, with sliding-window visibility on the majority and causal
  visibility on every `n`-th layer, following the artifact's
  `(layer + 1) % n == 0` global predicate;
- per-layer-type RoPE: a small theta with full rotation on sliding layers, a
  large theta with partial rotation on global layers, both half-split;
- grouped-query attention with a score scale of 1.0;
- pre-attention, post-attention, pre-feedforward and post-feedforward RMSNorm,
  with per-head query and key normalization and a **unit-gain** value
  normalization supplied as an explicit weight tensor;
- global layers with no value projection, taking `V` from the key projection
  **before** key normalization and rotation, so `K` and `V` differ after the
  layer;
- GeGLU feed-forward;
- a per-layer scalar on the MLP residual only;
- embedding output scale, tied embedding/output weights, and a logit softcap.

It is reduced in exactly these respects, which must be stated in the record and
in the CLI's own admission diagnostics:

| Reduced | Why, and who owns the real case |
|---|---|
| Dimensions far below the artifact's | Task 0015's host-reference limits: context ≤ 256, ≤ 65,536 elements per tensor, ≤ 8 attention layers |
| **Uniform** `kv_heads` and `head_dim` across layers | The artifact uses 16×256 local and 4×512 global. Heterogeneous per-layer KV geometry is a paged-state schema change owned by **M4** |
| Sliding layers store full history rather than a ring | Window eviction and page/ring/tail boundaries are **M4** |
| Synthetic BF16 weights | The artifact is INT8 `pack-quantized`; the importer is **M3** |
| Text tokens only, no image token ID accepted | Vision and the multimodal mask exemption are **M11** |
| Sliding window is a small constant, not 1024 | It must be smaller than the admitted context for the mask to be exercised at all |

Two distinct geometries are required, differing in query/KV head ratio, head
dimension, layer count, global-layer stride and sliding window, so no constant is
accidentally load-bearing.

### Execution, cancellation and failure

Nothing in the execution contract changes. The graph runs through the accepted
task 0015 service: immutable program bound to a sequence, chunked prefill over
actual requested chunks preserving absolute positions, one token per draw
committed before emission, output provenance checked at the materialized prefix,
whole-transaction abort on any failed step, and complete release on finish,
cancellation, failure or drop. Cancellation is checked at the same boundaries,
including the new operations. No CUDA work and no asynchronous lease is
introduced.

### Oracles and predeclared thresholds

Every new operation gets an independent FP64 transcription in
`moxie-oracles`, registered in the `OracleRegistry`, and a source-linked fixture
citing the legacy line it transcribes. No new tolerance is invented: BF16
boundaries keep task 0003's exact-rounding requirement, and FP32 attention keeps
its existing data-dependent error bound with `scale` substituted. The reduced
graph's dense-versus-paged comparison is **bit-for-bit**, as in task 0015.

`gelu_tanh` needs a stated accumulation order because `v³` cancels: the
transcription evaluates `v + 0.044715·v³` in FP64 and the fixture must include
operands where FP32 and FP64 disagree, so the declared bound is tested rather
than assumed.

## Acceptance

- Per-operation oracle agreement for all six parameters against their FP64
  transcriptions, at two distinct shapes each, including the boundary cases:
  `rotary_dim` of 0 and `head_dim`, `frequency_dim ≠ rotary_dim`,
  `kv_heads` of 1 and `heads`, a softcap that saturates, a residual scale that
  is not a power of two, and a `gelu_tanh` argument in the cancelling range.
- The load-bearing regression: substituting each retained-graph value into the
  Gemma-like graph changes its logits. A parameter that can be swapped without
  changing the result is a failed acceptance, not a passing one.
- Dense-reference versus paged-history logits bit-for-bit for both geometries,
  whole and chunked prefill, partial final chunks, multiple pages and decode.
- Generation through the service and CLI for both geometries: one and multiple
  tokens, greedy and temperature, fixed-seed service/CLI agreement, invalid
  input, busy, cancellation then a second generation, exactly one terminal event.
- Allocation: counting allocator peak below the recomputed admitted envelope,
  repeated generations retain no growth, every tier charge and reservation
  returns to baseline, and admission plus post-admission allocation failure both
  clean up completely. Preserve the existing negative controls.
- All prior consumers still pass unchanged: `G-INTERP-BF16`, `G-PAGED-HOST`,
  `G-PAGED-ALLOCATION`, `G-SAMPLING-HOST`, `G-SAMPLING-ALLOCATION`,
  `G-GENERATION-HOST`, `G-GENERATION-ALLOC`, and the 32,768-row storage
  regressions after the KV row-width change.
- Architecture: new positive fixture for `moxie-models-gemma4` under
  `MODEL_ALLOWED`, and new rejecting fixtures for a model crate importing
  `moxie-state`, `moxie-memory`, `moxie-engine`, `moxie-cuda`, `moxie-sampling`
  or `moxie-interp`, for a shared crate importing the model crate, and for a
  forbidden construct in model source. Validate the tracked tree from a clean
  `git archive`, as task 0015 established.
- Full workspace host and device-feature test lanes, both clippy lanes, `fmt`,
  `spec-check`, `git diff --check`, and the real GPU regression on all three
  UUIDs — expected unchanged, since no device behaviour is added. A changed GPU
  result is a blocker, not an update.
- Support matrix: add `G-GEMMA-REDUCED` and record it as a **synthetic reduced
  graph over synthetic BF16 weights**, with the reduction table above quoted in
  the limit column. The checkpoint column stays `none`. Update the bring-up
  record's integration-proof section to name what is and is not proven.
- Update the ADR set with the parameter-explicitness decision and its rationale,
  and write the handover.
- No new topology, sanitizer, quality or paired performance campaign. Those
  remain unmeasured and must be reported as unmeasured.

### Exact conditions requiring owner direction or task rejection

Stop and report, rather than proceeding, if any of these occur:

- A required Gemma equation cannot be resolved from the pinned legacy source and
  the artifact configuration together. Do not substitute a standard transformer
  form; document 09 §B forbids exactly that.
- The reduced graph would need to read a checkpoint, unpack INT8, or otherwise
  depend on M3.
- Heterogeneous per-layer KV geometry or window eviction turns out to be
  unavoidable for a correct reduced graph. That is an M4 dependency and a
  re-scope, not something to implement here.
- Any change would give the model crate execution ownership, add a second
  resource or transaction owner, introduce an unbounded scratch allocation, or
  require weakening a numerical gate.
- The work would depend on an unresolved owner ruling. O1, O2 and O5 in
  particular are open, and none of them may be resolved by inference from an
  artifact being present on disk.

## Result, filled after work

- Changed shared owners and consumers; source commit:
- Commands and result IDs; passed / failed / skipped separately:
- Measured effect and uncertainty:
- Deleted/replaced paths:
- Remaining blockers and next bounded task:
