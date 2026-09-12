# Task 0016 — M1.5 Gemma operation gap and reduced dense graph

Status: **accepted by the owner, 2026-09-12.** Contract `1199267` precedes
implementation `c7dd153`; review corrections `4e7ad56` and the guard-isolation
test improvement follow. M1.5 remains active — this task does not close it.

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

Status on completion: **accepted by the owner on 2026-09-12**, after independent
re-review found no remaining blocking correctness or architecture issue within
the task's declared synthetic scope. Contract `1199267` precedes implementation
`c7dd153`; the review corrections follow it and are recorded in their own
section below, with the acceptance record at the end.
[ADR 0012](../decisions/adr/0012-explicit-family-operation-parameters.md) records
the parameter-explicitness decision;
[ADR 0013](../decisions/adr/0013-one-model-crate-with-family-modules.md) records
the model-crate packaging decision the owner directed during implementation.

### Two deviations from the contract, both widenings

1. **Seven parameters, not six.** The contract listed six, on this record's own
   inventory of the bring-up document, which marked per-head Q/K normalization
   "none (composition)". That was wrong: composing a per-head norm needs
   `Concat`/`Split`, which have no `OpParams` and no registered oracle, so the
   grouping had to become a seventh parameter — `RmsNorm.group`. The bring-up
   record's row is corrected in place with the reason, rather than quietly
   updated. The new parameter carries the same obligations as the other six: an
   FP64 oracle, boundary cases, a device-path refusal, and a load-bearing test.
2. **The diagnostic graph-value cap moved from 128 to 384.** A six-layer
   Gemma-like graph is roughly 210 values, and task 0015's cap would have
   limited the reduced graph to three layers — too few to place a global layer
   among sliding ones the way the artifact does. The cap is an explicit
   host-reference diagnostic bound, not a context, precision or quality gate;
   raising it grows the reserved control envelope by at most 1 MiB
   (`4096 * (384 - 128)`), and the envelope formula itself is unchanged. No
   other task 0015 limit moved: context and chunk stay at 256, tensors at
   65,536 elements, rank at 4 and attention layers at 8.

3. **One `moxie-models` crate with a `gemma4` module, not `moxie-models-gemma4`.**
   The contract named a per-family crate because that is what M0's `arch-check`
   prefix assumed. The owner objected to the ceremony during implementation and
   was right: the boundary is the dependency list, which a module cannot widen,
   so one crate enforces the same three rules with a file and a `pub mod` line
   per family. `arch-check` now recognises both the unprefixed name and the
   retained `moxie-models-*` prefix, with three new fixtures so the change
   cannot silently exempt the real crate.
   [ADR 0013](../decisions/adr/0013-one-model-crate-with-family-modules.md).

Nothing else in the contract was relaxed. No numerical tolerance changed, no
owner gate was resolved, no checkpoint was read and no support claim is made.

### Changed shared owners and consumers

- **`moxie-graph`** owns the seven parameters, `RopeLayout`, and
  `reciprocal_sqrt_scale` for callers that want the conventional attention
  factor. `OpParams::GeGlu` joins the catalogue; `Op::GeGlu` already existed.
  `check_params` rejects a non-divisible head grouping, a zero frequency
  denominator, a half-split layout on an odd head, a group that does not divide
  the width, and any scale, cap or epsilon that is not finite and positive.
  `check_shapes` now derives the key and value operand widths from `kv_heads`
  rather than `heads`; the two coincided while every graph was multi-head, which
  is why the single-width rule went unnoticed.
- **`moxie-oracles`** owns `gelu_tanh`, `geglu_row`, `softcap`,
  `rms_norm_row_grouped`, `residual_row_scaled`, `embedding_row_scaled`, and the
  extended `rope_head`/`attend_multi_head`. Two descriptors —
  `rope::Rotation` and `attention::Heads` — group the parameters that belong to
  one decision, which is also what keeps the argument counts honest. `bf16_round`
  becomes public because GeGLU's gate rounding is part of its equation rather
  than storage; a new test checks it against `moxie-format` over every BF16
  pattern and every midpoint, which is what makes the reimplementation
  defensible rather than merely convenient.
- **`moxie-interp`** dispatches the new parameters through the same single
  dispatcher. Its paged path now compares `geometry.kv_heads` against the
  operation's `kv_heads`.
- **`moxie-engine`** derives the page geometry and the per-row reserve from
  `kv_heads * head_dim`. Under grouped-query attention that is narrower than the
  query width, so a reserve computed the old way would have over-admitted. Its
  uniformity check covers `(heads, kv_heads, head_dim)` and refuses a graph whose
  layers disagree.
- **`moxie-plan::selected`** refuses a scaled residual and a grouped norm with
  `UnsupportedKernel`, naming the offending value. No qualified device kernel
  applies either, and dropping the factor silently would be a numerical change
  wearing a fallback's clothes.
- **`moxie-models`** is new: `gemma4` holds the config, the tensor roles, the
  artifact's geometry as a separate non-runnable type, and graph composition.
  Three workspace dependencies, no allocation, no file read.
- **`moxie-cli`** gains `gemma`, a composition root that fills the declared roles
  with a synthetic pattern and a literal unit gain for the value norm, plus
  `--shape gemma-a|gemma-b`. It prints `event=reduced ... model_support=false`
  with the reduction list before anything executes.
- **`xtask`** recognises `moxie-models` as a model crate alongside the retained
  `moxie-models-*` prefix, and lets `moxie-cli` join `xtask` as a composition
  root permitted to import it.

### Commands and result IDs

| Gate / exact command | Result |
|---|---|
| `cargo test --workspace --locked --offline` | 652 tests, zero failed, zero ignored — 602 before this task plus 50 new |
| `cargo test -p moxie-cli --locked --offline --test gemma` | 14 tests, zero failed |
| `cargo test -p moxie-cli --locked --offline --test allocation -- --nocapture` | 1 test, zero failed; the three Gemma cases are in its printed table below |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | passed |
| `cargo fmt --all -- --check`; `git diff --check` | passed |
| `cargo xtask spec-check` | passed, all ten specification digests unchanged |
| `cargo xtask arch-check` | 73 rejecting + 21 accepted fixtures, 12 rules — three new fixtures for the unprefixed model crate name |
| Same `cargo test` with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda` | 659 tests + 12 doctests, zero failed/ignored |
| Same clippy command with the device feature list | passed |
| `cargo xtask-cuda test-gpu` | **39/39 real GPU cases, 0 failed, 0 skipped**; sm_86 and sm_120 both qualified |

The GPU result is unchanged from task 0015's, which is the expected outcome: this
task adds no device kernel, and the two new operations that could reach one are
explicit `UnsupportedKernel` refusals. A changed GPU result would have been a
blocker rather than an update.

`arch-check` run against the working tree also reports the four pre-existing
findings from the retained task 0014 probe crate under `results/`, exactly as
task 0015 recorded. Product validation uses a clean `git archive`.

### Retained evidence

Raw logs are retained outside git under `results/task0016/` through independent
review. Their SHA-256 values:

```text
host.log            e6d3dd9d1ce756153c5321d2d8d2c3dbe9034e0082c41a3cc2c4ec7449f6bb46
device.log          ab5ec722d3a35142c2d5c3e2860ec0d7596b8d0353dd51ef7ce79fd56b864db5
clippy.log          4cf86278c1e6d659943a46a3d7e3da429ac15be90f0f17742e24ef85228e7d3e
device-clippy.log   fb954d94f68bf842a50a2ce65de5bc86bf389e2ab1da68cc970f4086634dd5ba
gpu.log             debfa6b0a6bd2f53a86b1953c22122ff20c92509c7ee84977c8426c55bbea1ea
arch-clean.log      9ab0daa781e72184a9d2e1be4f7aa7bc529dc231a7e6d35d688fa14258871984
spec.log            7ee9b3fc6c1a5e5a612b078e860e08a5d6cb468f3fb849798dbf3c6df79207e9
gemma.log           b4f9d9aa96a2680a59bbf37a0db907e336b6f6867b8e432d6d1876859df24a6e
allocation.log      7b5df8c0ef89f00fa9cc7bf1155c66752e4195a5eece26da61fc3ad9431a6ba2
gemma-shards.sha256 36756b01120982483f08d4dd49912ba2ea74b39b65b1d3e4d1d4df60325ff86e
```

`gpu.log` and `spec.log` hash identically to task 0015's, which is the intended
result: neither the real GPU behaviour nor the specification digests changed.
`gemma-shards.sha256` is the artifact shard verification, retained because the
checkpoint itself stays outside git.

`arch-clean.log` is `arch-check` run against a copy of the tree without
`results/`, so the four retained task 0014 probe findings do not appear. It is
the product architecture result: 73 rejecting, 21 accepted, 12 rules.

The review corrections were validated again in full, retained under the same
directory with a `correction-` prefix. The pre-correction logs are kept rather
than overwritten, because they are the evidence of what the defects produced.

```text
correction-host.log            31cbb368711584fa0d67bd1d0fa5197a034f57af72d4246e0d4066e0470409a8
correction-device.log          c3ece1b79e2c7851330ecc9e7e9742f7bc8381a0890e82bd0a4355f41cb259bd
correction-clippy.log          a5f4c585ee974ca44916ac30a98bbc189e067a7e0a6bc6d2e8d6bc525be724af
correction-device-clippy.log   cb2794ab28b489f4bed6af23725861c9926dd03a72e38987daf83d288a8197e1
correction-gpu.log             debfa6b0a6bd2f53a86b1953c22122ff20c92509c7ee84977c8426c55bbea1ea
correction-arch-clean.log      51d23b6dd59bbeb000e5dae854365f76d184dfff35cd22f264683be4065272fc
correction-spec.log            7ee9b3fc6c1a5e5a612b078e860e08a5d6cb468f3fb849798dbf3c6df79207e9
correction-gemma.log           8dab7799f7108cbc8280cd94ff2973f94c85ca082ffaa1b51402ab6e076f10c4
correction-allocation.log      5dbd5cd1616ce963723f1b62796abcadf8869b4d67e9f4893be140e5c5ba2811
```

`correction-gpu.log` and `correction-spec.log` again hash identically to task
0015's, which is the intended result for both.

The acceptance commit's guard-isolation change was validated again; the GPU and
device lanes were not repeated for it, because it changes only a test in
`moxie-models`, which no device lane exercises.

```text
acceptance-host.log       6506c001b4d4eaeb229e589b00bbfd0d2c4c745c751b2ef0ba6e773849bd1345
acceptance-models.log     b794fa058436ae138f63868bec2afedc403ffdd1378591ead9e58440102ca332
acceptance-clippy.log     c92508a29848820f033cd2ea015c85dde0953c88b07cc967ce34b7c08f610549
acceptance-arch-clean.log 357e888e81eb78b7298771c5030a2b5d8a4d1abe65d394efb0392d3ff9febe9e
```

### Measured effect and uncertainty

The allocation harness prints, for each shape, the peak requested heap above the
ledger baseline and the admitted generation bytes:

| Shape | Layers | Query/KV heads | Head dim | Prompt / chunk | Peak above baseline | Admitted |
|---|---:|---|---:|---|---:|---:|
| gemma-a | 6 | 4 / 2 | 16 | 37 / 13 | 575,334 | 10,746,366 |
| gemma-a | 6 | 4 / 2 | 16 | 251 / 65 | 1,700,117 | 42,513,398 |
| gemma-b | 3 | 6 / 2 | 8 | 255 / 255 | 1,619,698 | 31,606,670 |

These are the conservative reference envelope's numbers, not a measured minimal
allocation plan and not a performance result. The `gemma-b` figures are lower
than first recorded because the review correction changed its grouping from
`6/3` to `6/2`: one fewer key/value head per layer is one fewer stored lane per
row, which the reserve follows directly. That the admitted bytes track the
key/value head count rather than the query head count is itself the evidence
that the row-width change landed where it should. The three task 0015 shapes retain
their previously recorded figures unchanged.

Sixty-four repeated generations on `gemma-a` retain no growth above the first
run's settled heap, and every tier charge returns to zero.

**Uncertainty and what is not measured:** no prefill/decode timing, no
throughput, no quality, no topology, no sanitizer lane, no context above 256 and
no checkpoint. The reduced graph's agreement with Gemma 4 is agreement with the
pinned legacy reference's equations at small dimensions over invented weights;
it is not evidence about the released model's outputs.

### A test that earned its place

The first reduced geometry used `head_dim` 8 with the artifact's quarter rotary
factor. A quarter of eight is a single rotated pair, whose inverse frequency is
`base^0` — so the global RoPE base cancelled entirely and the geometry silently
stopped testing it. `every_gemma_parameter_is_load_bearing` failed on exactly
that assertion. The geometries now use 16 and 8, the smallest widths at which
their partial rotary factors leave more than one angle. Had the acceptance
condition been "the oracle agrees with its transcription" rather than "the
parameter changes the result", this would have passed review as covered.

### Deleted and replaced paths

Nothing was deleted. No numerical gate, oracle or sampler stage was weakened, and
no legacy source was removed. `moxie-cli`'s `Options.shape_b: bool` became a
four-valued `Shape`; the two synthetic fixtures it selected are unchanged in
behaviour and are now the independent second consumer of every new parameter.
The unprefixed-model-crate rule replaces nothing: the `moxie-models-*` prefix is
retained and still tested.

### Independent review corrections

An independent review of `218b96f..c7dd153` requested corrections before
acceptance, reporting three reproducible defects and one acceptance-test gap.
**All four were confirmed by reproducing the reported numbers before changing
anything, and all four are fixed.** The review's architectural assessment —
shared ownership, the model module composing semantic operations, explicit
device refusals, and the disclosed synthetic-only limitation — is unchanged.

**1. [P1] The scaled residual dropped a required BF16 boundary.** The contract
and `gemma4_runtime.cpp:1230` both specify `bf16(bf16(a + b) * scale)`; the
implementation evaluated `(a + b) * scale` in FP64 and left one rounding to the
caller, on the reasoning that fewer roundings are more accurate. More accurate
is not the contract. Reproduced exactly as reported: with `a = 1`,
`b = 2^-8` and `scale = 0.875` the sum is precisely halfway between two BF16
values, so the declared boundary gives **0.875** and the implementation gave
**0.87890625** — an error that then entered every later layer. `residual_row`
at `scale == 1.0` is untouched and still returns the exact FP32 sum its own
contract promises. Two regressions: the reported probe, and the declared
two-boundary sequence over operands whose sums are not BF16-exact.

**2. [P1] The logit softcap dropped its final BF16 boundary.** The pinned
kernel's last line is `values[index] = bf16_round(value * softcap)`; the
implementation left that multiply unrounded, expecting the caller to round. The
caller does not round this one — logits are stored FP32 so the sampler sees the
pre-truncation distribution — so the boundary was lost, not deferred.
Reproduced exactly as reported: `x = 12.5`, `cap = 30` gave **11.8359375**
where the source gives **11.8125**. Being able to represent more precision is
not a reason to produce it. Two regressions: the reported probe, and an
assertion that every softcap result is a BF16 value across caps and magnitudes.

The review also identified why this escaped: `the_softcap_is_the_pinned_four_step_sequence`
transcribed the **implementation's** step sequence rather than the source's, so
it agreed with the bug. It is rewritten from the source. This is the failure
mode document 07 warns about, and dense-versus-paged agreement could never have
caught either defect, because both paths share the same arithmetic.

**3. [P2] Public graph composition could panic before shape validation.**
`TextConfig::check` deliberately delegates dimension validation to
`GraphBuilder`, but `heads * head_dim` and `kv_heads * head_dim` are computed
before construction and never reach it as multiplications it could check.
Reproduced with `heads = 2^63`, `head_dim = 16`. A checked `width` helper now
covers those two products and `vocab * hidden` and `intermediate * hidden`,
returning a typed `InvalidRequest` and additionally rejecting an extent past
`u32::MAX`. A regression drives five overflow configurations through the public
entry point and requires refusal at construction or composition — never a wrap
and never a panic.

**4. [P2] The "every parameter" acceptance claim was incomplete.** The test's
name overstated its coverage: four substitutions were missing, and one geometry
pair did not vary what the contract said it would.

| Gap reported | Correction |
|---|---|
| `frequency_dim = rotary_dim` never substituted | `Swap::RotaryWidthDenominator`; only global layers rotate partially, so only they change |
| SwiGLU never substituted for GeGLU | `Swap::SwiGlu`; identical shapes, different gate transform |
| Whole-row never substituted for grouped normalization | `per_head_normalization_is_load_bearing`, with its own graph |
| Softcap substituted a huge cap, not `None` | `Swap::NoSoftcap`; the large-cap case is retained beside it, since the contract distinguishes them |
| Both geometries had ratio 2 (`4/2`, `6/3`) | shape B is now `6/2`, so the two group differently |

The grouped-norm substitution needed its own fixture rather than a parameter
swap, because setting `group` to 1 also changes how wide the gain must be — a
rebuilt Gemma graph would have differed in two places at once. The new fixture
holds the gain at all ones under both groupings, so the reduction is the only
difference. Its first version was vacuous: every embedding row followed the same
lane pattern, so every row normalized to the same vector and both groupings gave
identical logits. The rows now vary within each group and between tokens.

### Gates after the corrections

| Gate | Result |
|---|---|
| `cargo test --workspace --locked --offline` | **657 passed**, 0 failed (652 before the corrections, 5 new regressions) |
| `cargo test -p moxie-cli --test gemma` | 15 passed |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | passed |
| `cargo fmt --all -- --check`; `git diff --check` | passed |
| `cargo xtask spec-check` | passed, ten digests unchanged |
| clean-archive `arch-check` | 73 rejecting + 21 accepted, 12 rules |
| Device workspace lane | **664 passed + 12 doctests**, 0 failed |
| Device clippy | passed |
| `cargo xtask-cuda test-gpu` | **39/39**, 0 failed, 0 skipped; sm_86 and sm_120 qualified |

The reduced graph's logits changed, which is the point: they were wrong before.
No test expectation was adjusted to accommodate the new values — the parity,
cancellation and allocation tests compare the implementation against itself or
against a bound, and the two fixed equations are pinned against the source
rather than against the implementation.

### Independent re-review and owner acceptance — 2026-09-12

The owner accepted task 0016 and directed "Document and Push" after independent
re-review confirmed all four findings were addressed and found **no remaining
blocking correctness or architecture issue within the declared synthetic scope**.
The reviewer re-ran the original numerical and overflow probes against the
corrected libraries: the residual reproduces 0.875 with both BF16 boundaries
preserved, the softcap reproduces 11.8125 including the final rounding before
FP32 storage, and the original overflowing configuration returns an error
without panicking.

Independent verification passed the full host workspace suite, both clippy
lanes, formatting, specification and diff checks, clean-archive architecture
checks (73 rejecting / 21 accepted) and 39/39 real GPU cases across all three
cards. The reviewer did not repeat the full device-feature suite or the
checkpoint shard hashing; those remain separately recorded implementation
evidence.

**One non-blocking test improvement was reported and is fixed here rather than
deferred.** The overflow regression's `kv_heads` case also overflowed `heads`,
so the query-width guard rejected it first and the key/value guard was never
exercised. Each case now names the guard it must reach and asserts the error
came from that guard, so a configuration refused by an earlier one fails the
test instead of passing it. The `kv_heads` case perturbs only `kv_heads`, which
genuinely reaches the multiplication because `TextConfig::check` does not
require `kv_heads <= heads` — that belongs to `OpParams::check_params`, which
runs after these products are formed. A negative control confirms the
assertion bites: replacing the checked `kv_heads * head_dim` with a wrapping
multiply makes the case fail with a different error rather than pass.

**Acceptance is limited to task 0016's reduced synthetic graph, and does not
close M1.5.** Actual checkpoint execution, heterogeneous per-layer state
geometry and model-quality validation all remain outstanding. No owner gate was
resolved; O1–O7 remain open.

### Remaining blockers and next bounded task

1. **M3 blocks the artifact.** Every language-model linear in
   `cyankiwi/gemma-4-31B-it-AWQ-8bit` at
   `34ca187d836de874b2c7e3edf48f439b9f583772` is compressed-tensors INT8
   `pack-quantized`, group 32, symmetric, four codes per `int32` along the input
   axis, BF16 scales. The importer, packed-layout reader, canonical repack and
   W8A16 execution path are all M3, and nothing here substitutes for them.
2. **M4 blocks the real graph's shape.** The artifact's sliding layers are 16
   heads of 256 and its global layers 4 of 512; one `KvGeometry` cannot hold
   both, and sliding layers want ring eviction at the window rather than full
   retention.
3. **M11 blocks vision.** The artifact is `image-text-to-text` and its mask rule
   has a multimodal group exemption. The reduced graph is text-only and refuses
   nothing about images because it never sees one — a real text-only bring-up
   must refuse image tokens explicitly.
4. O1, O2 and O5 remain open. A downloaded artifact is not a catalog decision, no
   quality statement exists, and nothing may be converted.

The next bounded task is M1.5's remaining half or M4's state schema, at the
owner's direction. The reduced graph proves the contracts M1.5 asks it to prove;
it does not prove integration, and this record does not claim it does.
