# Model bring-up contract — Gemma 4

One record per family, per [the placement contract](../README.md). Two variants
are inventoried here: the dense **31B-IT** below, and the routed **26B-A4B-IT**
in [its own section](#the-26b-a4b-moe-variant) at the end. They are the same
family — same global predicate, same layer-type asymmetry, same softcap and
window — and neither is executed.

Status: **inventory 2026-09-12; reduced synthetic graph implemented by
[task 0016](../tasks/0016-m1-gemma-reduced-graph.md), given per-layer key/value
geometry and window reclamation by
[task 0017](../tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md).**
The checkpoint is still not imported and not executed. This record exists because roadmap M1.5 requires a
Gemma text graph and document 09 §B requires the mathematical inventory *before*
implementation. It resolves no owner gate and makes no support claim.

## Identity — 31B dense

| Property | Observed value |
|---|---|
| Family | Gemma 4, text tower of `Gemma4ForConditionalGeneration` |
| Local artifact | `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` (read-only input) |
| Source repository | `cyankiwi/gemma-4-31B-it-AWQ-8bit` |
| Immutable revision | `34ca187d836de874b2c7e3edf48f439b9f583772` |
| Base model declared by the artifact | `google/gemma-4-31B-it` |
| License declared in the artifact README | `apache-2.0`, link `https://ai.google.dev/gemma/docs/gemma_4_license` |
| Reference implementation declared | `transformers 5.5.0.dev0`, `model_type` `gemma4` / `gemma4_text` |
| Canonical artifact profile | **not canonical yet**: compressed-tensors `pack-quantized` INT8 |
| Quantizer | `compressed-tensors 0.14.1.a20260326`, `quantization_status: compressed` |
| Tokenizer | `tokenizer.json`, `tokenizers` BPE backend, 262,144 vocabulary entries + 24 added tokens |
| Chat template | `chat_template.jinja`, 16,448 bytes, with tool-call and thinking channels |
| Trained positional range | `max_position_embeddings` 262,144 — **not** an admissible context here (R19) |
| O1 catalog decision | OPEN. Presence on disk is not catalog membership |
| O2 quality decision | OPEN. No paired quality evidence exists |
| O5 storage/conversion authorization | OPEN. Nothing may be copied, converted or requantized |

The same revision was already recorded remotely in
[quantization-candidates.md](../evidence/quantization-candidates.md); the local
`config.json` hashes to `f9f7b7c592c98a99018843aa242b05cbd19f98a423ce43241d3ef208ec444b76`,
which is byte-identical to the hash recorded there on 2026-09-07. File hashes and
shard completeness are in
[checkpoint-inventory.md](../evidence/checkpoint-inventory.md).

### Declared geometry, read from `config.json`

| Field | Value |
|---|---|
| `hidden_size` / `num_hidden_layers` | 5376 / 60 |
| `intermediate_size` | 21,504 |
| `num_attention_heads` | 32 |
| Local layers: `num_key_value_heads` / `head_dim` | 16 / 256 |
| Global layers: `num_global_key_value_heads` / `global_head_dim` | 4 / 512 |
| `layer_types` | 50 `sliding_attention`, 10 `full_attention`; the global layers are exactly indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 |
| `sliding_window` | 1024 |
| `rope_parameters.sliding_attention` | `rope_type` `default`, `rope_theta` 10,000 |
| `rope_parameters.full_attention` | `rope_type` `proportional`, `rope_theta` 1,000,000, `partial_rotary_factor` 0.25 |
| `rms_norm_eps` | 1e-6 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `final_logit_softcapping` | 30.0 |
| `attention_k_eq_v` | `true` |
| `tie_word_embeddings` | `true` (there is no `lm_head` tensor) |
| `vocab_size` | 262,144 |
| `enable_moe_block` / `num_experts` / `top_k_experts` | `false` / `null` / `null` — this 31B variant is **dense** |
| `use_double_wide_mlp`, `num_kv_shared_layers`, `hidden_size_per_layer_input` | `false`, 0, 0 |
| `use_bidirectional_attention` | `"vision"` |
| Vision tower | 27 layers, hidden 1152, patch 16, 280 soft tokens per image, BF16 and wholly excluded from quantization |

Every one of these text values matches the frozen legacy constant table
`kGemma4ExecutionContract` at `include/strata/models/common/model_adapter.hpp:100`,
including the global-layer predicate `(layer + 1) % 6 == 0` at `:107`. That
agreement is what makes the pinned legacy source usable as the equation
reference for this artifact.

## Mathematical inventory — 31B dense

Source citations are paths relative to the frozen legacy root
`/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`,
followed by line number, per document 08's citation convention.

| Component | Source equation / code / line | Common op and options | Gap task if missing | Oracle fixture |
|---|---|---|---|---|
| Token embedding | `src/models/gemma4/gemma4_runtime.cpp:699` — `e = bf16(row · bf16(sqrt(hidden_size)))` | `Embedding` **plus a checkpoint-defined output scale**; today `OpParams::Embedding` has no scale field | **gap** — task 0016 | new; exact BF16 rounding of a known row |
| Output head | `gemma4_runtime.cpp:1290` then `matmul_softcap`; cap applied as `bf16(bf16(tanh(bf16(bf16(x)/c))) · c)`, `kernels/cuda/detail/backend_kernels.cuh:905` | `VocabProjection` **plus logit softcap 30.0**; tied to the embedding matrix | **gap** — task 0016 | new; FP64 transcription of the four-step cap |
| RMSNorm | `src/models/gemma4/gemma4_ops.cpp:11` over `src/platform/numerics.cpp:10`; `y = x · rsqrt(mean(x²) + eps) · w`, FP64 sum, then a BF16 rounding of the result. The gain is applied **as stored**, with no `1 + w` rebias | `RmsNorm { hidden, eps }` — already present, semantics already match | none | existing `moxie-oracles::norm::rms_norm_row` |
| Q/K normalization | `gemma4_runtime.cpp:798` and `:810` — per head, over `head_dim`, with the layer's `q_norm` / `k_norm` gains, before RoPE | `RmsNorm` **plus an explicit group count**; one normalization per head over `hidden / group` lanes, sharing one gain of that width | **gap** — task 0016. This row first read "none (composition)", on the assumption that per-head slicing could be composed from `Concat`/`Split`. It cannot: those operations have no `OpParams` and no oracle, so the grouping had to become a parameter | new; grouped against ungrouped on a row whose groups differ in magnitude |
| V normalization | `gemma4_runtime.cpp:818` — RMSNorm with an **all-ones gain** and the same epsilon, applied to V and never to K | the same grouped `RmsNorm`, with a unit gain tensor | as above; the unit gain is an explicit weight, not an implicit special case | new |
| RoPE | `gemma4_ops.cpp:22` — pairing is **half-split** `(j, j + head_dim/2)`, and the inverse frequency is `theta^(−2j / head_dim)` using the **full** head dimension even when only `partial_rotary_factor · head_dim / 2` angles are rotated | `Rope { heads, head_dim, rotary_dim, base }` exists but implements **interleaved** `(2j, 2j+1)` pairing and divides by `rotary_dim` | **gap** — task 0016 must add an explicit pairing layout and an explicit frequency denominator | new; both conventions pinned against an FP64 transcription |
| GeGLU | `gemma4_ops.cpp:70` — `bf16(bf16(gelu_tanh(gate)) · up)`, with `gelu_tanh` at `src/platform/numerics.cpp:86` | `GeGlu` — **absent**; only `SwiGlu` exists | **gap** — task 0016 | new; `gelu_tanh` against its FP64 transcription plus the BF16 boundary |
| Attention | `gemma4_runtime.cpp:857` — `request.scale = 1.0F`, i.e. **no** `1/sqrt(head_dim)` factor; FP32 dot, FP32 softmax, FP32 accumulation; output rounded to BF16 at `:900` | `Attention { heads, head_dim, visibility, layer }` — hardcodes `1/sqrt(head_dim)` in `moxie-oracles::attention::attend_multi_head` and assumes `kv_heads == heads` | **gap** — task 0016 must add an explicit score scale and a GQA key/value head count | extend the existing attention oracle; keep the existing error bound |
| Masks | `gemma4_ops.cpp:86` — `k ≤ q` for global layers; `k ≤ q ∧ q − k < 1024` for sliding layers | `Visibility::Causal` and `Visibility::SlidingWindow { window }` — already bit-identical to the legacy predicate for the text-only case | none | existing `moxie-oracles::mask` |
| Multimodal mask group exemption | `gemma4_ops.cpp:88` — tokens sharing a non-negative image group attend bidirectionally | not modeled | **deferred**, M11 vision; text-only bring-up must refuse image inputs rather than silently drop the exemption | none |
| Attention residual | `gemma4_runtime.cpp:1186` — `h = bf16(h + post_attention_norm(attn))`, **no** layer scalar | `Residual` | none | existing |
| MLP residual | `gemma4_runtime.cpp:1230` — `h = bf16(bf16(h + post_ffn_norm(mlp)) · layer_scalar)`, a per-layer BF16 scalar read from the checkpoint | `Residual` **plus a per-layer scale**, applied only on this residual | **gap** — task 0016 | new; the asymmetry between the two residuals is the fixture |
| Global-layer K/V sharing | `gemma4_runtime.cpp:779` — when `v_proj` is absent, `V := K` is taken **before** q/k norm and RoPE, so `K = rope(rmsnorm_{k_norm}(Kp))` while `V = rmsnorm_1(Kp)` | graph composition: one projection feeding two normalizations | none (composition), but it must be composition, not a runtime branch | new; a fixture proving K and V differ after the layer |
| Routing / experts | not applicable | — | — | this variant is dense; `enable_moe_block` is `false` |
| Recurrent / convolution / index / compression state | not applicable | — | — | — |
| Native proposal heads | absent from the tensor index | — | — | — |
| Image modality | `src/models/gemma4/gemma4_image.cpp:1`, vision RoPE at `gemma4_ops.cpp:52` | not modeled | **deferred**, M11 | existing legacy fixtures `tests/test_gemma4_image.cpp:1` |

Two entries deserve explicit warning, because both are places where a "standard
transformer" guess would produce plausible text and wrong logits:

1. **The attention score scale is 1.0, not `1/sqrt(head_dim)`.** The shared oracle
   currently bakes the reciprocal square root in. Any Gemma graph that reuses it
   unchanged is numerically wrong at every layer.
2. **RoPE pairing is half-split and the partial-rotary denominator is the full
   head dimension.** The shared oracle uses interleaved pairing and divides by
   `rotary_dim`. For the global layers, where `partial_rotary_factor` is 0.25,
   the two conventions disagree on both which elements rotate and by how much.

Neither is a defect in the existing shared code — task 0003 pinned one convention
against its own fixtures. They are missing *parameters*, and R06 is the standing
instruction not to erase such differences.

## Import and resources — 31B dense

### Logical tensor-role mapping

2008 tensors across seven shards. Text tower, per layer `L` in `0..60`:

| Source name | Logical role | Storage | Logical shape |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | `embedding` (tied output head) | BF16 | `[262144, 5376]` |
| `model.language_model.layers.L.input_layernorm.weight` | `attn_norm[L]` | BF16 | `[5376]` |
| `model.language_model.layers.L.self_attn.q_proj.{weight_packed,weight_scale,weight_shape}` | `q_proj[L]` | INT8 packed | local `[8192, 5376]`, global `[16384, 5376]` |
| `model.language_model.layers.L.self_attn.k_proj.{…}` | `k_proj[L]` | INT8 packed | local `[4096, 5376]`, global `[2048, 5376]` |
| `model.language_model.layers.L.self_attn.v_proj.{…}` | `v_proj[L]` | INT8 packed | local `[4096, 5376]`; **absent on all ten global layers** |
| `model.language_model.layers.L.self_attn.o_proj.{…}` | `o_proj[L]` | INT8 packed | local `[5376, 8192]`, global `[5376, 16384]` |
| `model.language_model.layers.L.self_attn.q_norm.weight` | `q_norm[L]` | BF16 | local `[256]`, global `[512]` |
| `model.language_model.layers.L.self_attn.k_norm.weight` | `k_norm[L]` | BF16 | local `[256]`, global `[512]` |
| `model.language_model.layers.L.post_attention_layernorm.weight` | `attn_out_norm[L]` | BF16 | `[5376]` |
| `model.language_model.layers.L.pre_feedforward_layernorm.weight` | `ffn_norm[L]` | BF16 | `[5376]` |
| `model.language_model.layers.L.mlp.{gate,up}_proj.{…}` | `ffn_gate[L]`, `ffn_up[L]` | INT8 packed | `[21504, 5376]` |
| `model.language_model.layers.L.mlp.down_proj.{…}` | `ffn_down[L]` | INT8 packed | `[5376, 21504]` |
| `model.language_model.layers.L.post_feedforward_layernorm.weight` | `ffn_out_norm[L]` | BF16 | `[5376]` |
| `model.language_model.layers.L.layer_scalar` | `mlp_residual_scale[L]` | BF16 | `[1]` |
| `model.language_model.norm.weight` | `final_norm` | BF16 | `[5376]` |
| `model.embed_vision.embedding_projection.weight`, `model.vision_tower.*` | vision, **out of text-bring-up scope** | BF16 | — |

The absence of `v_proj` on exactly the ten global layers is a structural fact of
this artifact, not an incomplete download: the index lists 50 `v_proj` triples,
and the ten layers missing them are exactly the `full_attention` layers. It is the
serialized form of `attention_k_eq_v`.

### Exact packing parameters

- `format` `pack-quantized`, `num_bits` 8, `type` `int`, `symmetric` `true`,
  `strategy` `group`, `group_size` 32, `actorder` `null`, `zp_dtype` `null`,
  `scale_dtype` `null`, `observer` `mse`.
- `weight_packed` is `I32` with the logical input dimension divided by 4:
  four INT8 codes per `int32` along the **input** axis.
- `weight_scale` is `BF16`, shaped `[out_features, in_features / 32]`.
- `weight_shape` is `I64[2]` holding the logical `[out_features, in_features]`
  — verified for `q_proj` `(8192, 5376)`, `o_proj` `(5376, 8192)` and
  `down_proj` `(5376, 21504)`.
- `quantization_config.ignore` lists 192 modules: the whole vision tower, the
  vision embedding projection, and every norm. Embeddings are absent from the
  quantized tensor set entirely and remain BF16.

Declared `scale_dtype: null` is exactly the case
[quantization-candidates.md](../evidence/quantization-candidates.md) warns about:
the tensor header, not the config, established that the scales are BF16.

### Resources, state and partitions

- State schema per token, at the declared BF16 cache precision: local layers
  `2 · 16 · 256 · 2 B` = 16,384 B; global layers `2 · 4 · 512 · 2 B` = 8,192 B.
  Whole model per token: `50 · 16,384 + 10 · 8,192` = 901,120 B. A **local**
  layer's history is bounded by the 1,024-token window; a global layer's is not.
  At 32,768 admitted tokens the global layers alone need
  `10 · 8,192 · 32,768` = 2.68 GB and the windowed layers a further
  `50 · 16,384 · 1,024` = 0.84 GB. This is an arithmetic projection from the
  declared schema, not a measured admission.
- Weights on disk are 35,089,877,112 B of tensor payload against 62.6 GiB of
  aggregate device memory across three unequal GPUs. Host-backed weights are the
  ordinary path, not a degraded mode.
- Legal partitions are **not determined**. `OpParams::Attention` returns
  `PartitionRule::NotDetermined` deliberately, and M5 owns it. The
  32/16 and 32/4 query-to-KV head ratios are the relevant constraint when it is.
- Rollback method: the accepted task 0013 paged transactions, extended by
  [task 0017](../tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md)
  with per-layer retention. A sliding layer's ring boundaries are now tested,
  and a rollback whose target window has been reclaimed is refused rather than
  served: it needs re-prefill. The device ring, host-backed page streaming and
  the rest of M4 are still outstanding.

## Integration proof — 31B dense

**Nothing in this repository executes this checkpoint.** What exists after
[task 0016](../tasks/0016-m1-gemma-reduced-graph.md) is a reduced synthetic
graph with this family's shape of mathematics, over weights the composition root
invents. Its output is not model output.

- **Adapter contains metadata/graph only: yes.** `moxie_models::gemma4` depends
  on `moxie-types`, `moxie-graph` and `moxie-model-api` and nothing else; it
  allocates nothing, reads nothing and launches nothing. Enforced by
  `arch-check`, including fixtures for the unprefixed crate name
  ([ADR 0013](../decisions/adr/0013-one-model-crate-with-family-modules.md)).
- **New shared ops and second-consumer tests: done for the text path.** Seven
  operation parameters became explicit
  ([ADR 0012](../decisions/adr/0012-explicit-family-operation-parameters.md)),
  each with an FP64 oracle, and each proven load-bearing by substituting the
  conventional value and requiring the logits to change. The retained task 0015
  synthetic graph is the second consumer, carrying the opposite value for every
  one of them.
- **No private runtime/cache/transfer/sampler/branch evaluator: yes.** The
  reduced graph runs through the accepted task 0015 service, task 0013 paged
  transactions and the task 0014 sampler.
- Reference quality, actual-context prefill/decode and continuation: **not
  started**, and not possible without M3.
- GPU precision and distributed tests: **not applicable.** No new device kernel;
  a scaled residual and a grouped norm are explicit `UnsupportedKernel`
  refusals on the qualified device path.
- Common sampling, future entropy and speculation capability results: **not
  started.** M9 and M10.
- HTTP/CLI/template/tokenizer/modality parity: **not started.** The diagnostic
  CLI takes token IDs; M8 owns the surfaces.
- Support matrix updated: yes, gate `G-GEMMA-REDUCED`, with the reduction
  recorded in its limit column.

### What the reduced graph preserves and what it drops

Preserved: the `(layer + 1) % 6 == 0` global predicate, per-layer-type RoPE
(theta and partial rotation), grouped-query attention at a score scale of
exactly 1.0, per-head query and key normalization with a unit-gain value
normalization, global layers taking `V` from the key projection before its norm
and rotation, GeGLU, four RMSNorms per layer, a per-layer scalar on the MLP
residual only, the embedding output scale, tied embeddings and the logit softcap.

Task 0017 removed two of the four reductions. The reduced graph now composes
sliding and global layers at **their own** key/value head counts and head
dimensions, and its store keeps only what each layer's window can see, refusing
anything below it. The 31B artifact's own 16x256 versus 4x512 asymmetry is
expressible; what is still missing to run it is weights, not shape.

Still dropped, each with its owner: INT8 import and execution (**M3**); the
vision tower and the multimodal mask exemption (**M11**); real weights and real
dimensions (**M3**, then a real admission profile). The running CLI prints this
list before it prints anything else.

## Blockers — 31B dense

1. **Execution of this artifact is M3-blocked.** Every language-model linear is
   INT8 `pack-quantized`.
   [Task 0018](../tasks/0018-m3-compressed-tensors-int8-importer.md) closed two
   of the four gaps: there is now a compressed-tensors importer and a
   packed-layout reader, and three of this artifact's real modules import to
   canonical affine form. **No canonical repack and no W8A16 execution path
   exist**, so nothing runs, and an import is not support. M3 owns both. This
   record still does not authorize a private loader, a dequantization fallback
   or a model-owned decode path as a way around that.
2. **The text graph mathematics are recoverable, and the shared gaps are now
   closed for the text path.** Task 0016 made seven operation parameters
   explicit — the six this inventory first listed, plus the grouped norm the
   Q/K row above corrects. None of them depended on M3, and all of them are
   exercised by the reduced graph over synthetic BF16 weights.
3. **Vision is out of scope** and must be refused explicitly rather than ignored:
   the artifact is `image-text-to-text`, its mask rule has a multimodal group
   exemption, and dropping that exemption silently would change text results for
   any prompt containing image tokens.
4. **O1, O2 and O5 remain open.** No catalog membership, no quality statement, no
   conversion.

## Bring-up cost — 31B dense

Not started. Inventory reading on 2026-09-12 cost one session and produced no
code. Record engineering hours, files/lines by ownership, new versus reused
shared operations, initial untuned performance and the O7 review when the family
is actually brought up.


## The 26B-A4B MoE variant

Status: **inventory 2026-09-12; routed graph composed by
[task 0019](../tasks/0019-m2-routed-expert-semantics.md) over synthetic weights,
accepted 2026-09-12; its expert bytes demand-read through the residency
authority by [task 0020](../tasks/0020-m2-weight-residency-authority.md),
2026-09-12.** The checkpoint is **not imported and not executed** — task 0020
reads bounded ranges and computes nothing with them. This record
exists because roadmap M2 needs a real BF16 MoE and document 09 §B requires the
mathematical inventory before implementation. It resolves no owner gate and
makes no support claim.

### Identity

| Property | Observed value |
|---|---|
| Family | Gemma 4, text tower of `Gemma4ForConditionalGeneration`, **routed** |
| Local artifact | `/fast/models/google/gemma-4-26B-A4B-it` (read-only input) |
| Immutable revision | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| `config.json` sha256 | `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3` |
| `model.safetensors.index.json` sha256 | `907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7` |
| Shard completeness | **complete.** For each shard `8 + header + payload_end` equals the file size exactly, and the two payload ends sum to the index's `total_size` of 51,611,872,412 B |
| Base model declared | `google/gemma-4-26B-A4B` |
| License declared in the artifact README | `apache-2.0`, link `https://ai.google.dev/gemma/docs/gemma_4_license` |
| Pipeline declared | `image-text-to-text` |
| Reference implementation declared | `transformers 5.5.0.dev0`, `model_type` `gemma4` / `gemma4_text` |
| Canonical artifact profile | **BF16, unquantized.** There is no `quantization_config` |
| Tokenizer | `tokenizer.json`, 262,144 vocabulary entries; `chat_template.jinja`, 18,683 bytes |
| Trained positional range | `max_position_embeddings` 262,144 — **not** an admissible context here (R19) |
| O1 catalog decision | designated as M2's BF16 MoE by the owner on 2026-09-12; the **catalog gate itself stays OPEN** |
| O2 quality decision | OPEN. No paired quality evidence exists |
| O5 storage/conversion authorization | OPEN. Nothing may be copied, converted or requantized |

**No download was made for this task.** The artifact was already on disk when
the owner designated it.

### Declared geometry, read from `config.json`

| Field | Value | Same as the 31B? |
|---|---|---|
| `hidden_size` / `num_hidden_layers` | 2,816 / 30 | no (5,376 / 60) |
| `intermediate_size` (dense MLP) | 2,112 | no (21,504) |
| `num_attention_heads` | 16 | no (32) |
| Local: `num_key_value_heads` / `head_dim` | 8 / 256 | head dim yes, count no |
| Global: `num_global_key_value_heads` / `global_head_dim` | 2 / 512 | head dim yes, count no |
| `layer_types` | 25 sliding, 5 full at indices 5, 11, 17, 23, 29 | **yes — the same `(layer + 1) % 6 == 0` predicate** |
| `sliding_window` | 1,024 | yes |
| `rope_parameters.sliding_attention` | `default`, theta 10,000 | yes |
| `rope_parameters.full_attention` | `proportional`, theta 1,000,000, `partial_rotary_factor` 0.25 | yes |
| `rms_norm_eps` / `hidden_activation` | 1e-6 / `gelu_pytorch_tanh` | yes |
| `final_logit_softcapping` / `attention_k_eq_v` / `tie_word_embeddings` | 30.0 / true / true | yes |
| `vocab_size` | 262,144 | yes |
| `enable_moe_block` / `num_experts` / `top_k_experts` / `moe_intermediate_size` | **true / 128 / 8 / 704** | **no — the 31B is dense** |
| Vision tower | 27 layers, hidden 1,152, patch 16, 280 soft tokens per image | yes |

That agreement is what lets task 0016's seven operation parameters and task
0017's per-layer paged geometry transfer to this variant instead of being
rebuilt. `_compute_proportional_rope_parameters` in the pinned
`transformers/modeling_rope_utils.py:187` confirms task 0016's reading of the
global layers: `rope_angles = partial_rotary_factor · head_dim // 2`, the
exponent denominator is the **full** head dimension, and the unrotated tail is
zero-padded, which is an identity rotation. No new RoPE gap.

### Mathematical inventory — the routed delta

Only the rows that differ from the dense variant are listed; everything else is
[the 31B inventory](#mathematical-inventory--31b-dense) unchanged.

**The reference is not the legacy tree.** The frozen legacy snapshot has **no
Gemma 4 MoE** — its `gemma4` adapter is the dense path, and a search for
`per_expert_scale`, `router.scale`, `enable_moe_block` and `gate_up_proj` across
it returns only unrelated GLM-5.2 routing text. The pinned reference for the
rows below is the released `transformers` source, read and **never executed**:
`transformers/models/gemma4/modeling_gemma4.py`, classes `Gemma4TextRouter`,
`Gemma4TextExperts` and `Gemma4TextDecoderLayer`. Three copies were compared —
5.5.3 twice and 5.15 once — and the mathematics is identical in all three. One
difference is recorded rather than averaged: **5.15 computes the router softmax
in FP32 and says so ("fp32 for numerical stability"); 5.5.3 computes it in the
input dtype.** The artifact declares 5.5.0.dev0. Task 0019 pins FP32, which is
also what every other reference in `moxie-oracles` does; if that choice ever
changes a *selection* rather than coefficient precision, it becomes an O2
question rather than an implementation detail.

| Component | Source equation | Common op and options | Gap task | Oracle fixture |
|---|---|---|---|---|
| Router score transform | `Gemma4TextRouter.forward` — scale-free RMSNorm, then `⊙ router.scale`, then `· hidden^(-1/2)`, then the `[E, H]` projection, **with a BF16 boundary after each** | `Route` with `eps` and an explicit `input_scale` | **gap** — task 0019 | FP64 transcription carrying all four boundaries, plus a fixture where dropping them selects different experts |
| Selection and renormalization | softmax over **all** experts, `topk`, then `w /= Σw` | `Route`, top-k with the shared **lower-id** tie rule | **gap** — task 0019 | exact selection plus a counted coefficient bound |
| Per-expert coefficient scale | `top_k_weights * per_expert_scale[top_k_index]`, applied **after** renormalization | `Route { per_expert_scale: true }` | **gap** — task 0019 | the coefficients do **not** sum to one; a negative scale is legal |
| Expert feed-forward | `linear(x, gate_up_proj[e]).chunk(2, -1)`, `act(gate) * up`, `linear(·, down_proj[e])`, the first two rounding to BF16 | `ExpertMlp` over the **fused** `[E, 2I, H]` / `[E, H, I]` tensors, `activation = GeGlu` | **gap** — task 0019 | per-expert FP64 with those boundaries, a cancelling fixture bounded against `Σ|terms|`, and a proof that one expert cannot read another's slice |
| Combination | `index_add_` over `expert_hit`, which is expert-major | `Combine { order: AscendingExpertId }` | **gap** — task 0019 | both orders, on a fixture where FP32 addition is not associative |
| Shared expert | the dense `mlp`, normalized by `post_feedforward_layernorm_1`, added to the routed branch's `post_feedforward_layernorm_2` output | **graph composition**, not a routing parameter: it takes no routing coefficient | none (composition) | a substitution test on each of the three norms |
| Router input | the **un-normalized** post-attention residual, not `pre_feedforward_layernorm(r)` | composition; the router is the block's only consumer of the raw residual | none (composition) | a graph test asserting the `Route` node's producer is the residual |
| Expert input | `pre_feedforward_layernorm_2(r)` — its **own** gain tensor, not the dense branch's | composition | none (composition) | binding the dense gain into the routed slot must change the logits |

The block, as composed:

```text
r  = post-attention residual
m  = mlp(pre_feedforward_layernorm(r))            the shared expert
h1 = post_feedforward_layernorm_1(m)
h2 = post_feedforward_layernorm_2(
       Combine(Route(r), ExpertMlp(pre_feedforward_layernorm_2(r), Route(r))))
r' = (r + post_feedforward_layernorm(h1 + h2)) · layer_scalar
```

### Logical tensor-role mapping — the routed tensors

1,013 tensors across two shards; 657 belong to the language model. Per layer
`L` in `0..30`, **in addition to** every dense role in the 31B table (all of
which are present, including the whole `mlp`):

| Source name | Logical role | Storage | Logical shape |
|---|---|---|---|
| `model.language_model.layers.L.router.scale` | `router_scale[L]` | BF16 | `[2816]` |
| `model.language_model.layers.L.router.proj.weight` | `router_proj[L]` | BF16 | `[128, 2816]` |
| `model.language_model.layers.L.router.per_expert_scale` | `router_per_expert_scale[L]` | BF16 | `[128]` |
| `model.language_model.layers.L.experts.gate_up_proj` | `experts_gate_up[L]` | BF16 | `[128, 1408, 2816]` |
| `model.language_model.layers.L.experts.down_proj` | `experts_down[L]` | BF16 | `[128, 2816, 704]` |
| `model.language_model.layers.L.pre_feedforward_layernorm_2.weight` | `ffn_norm_2[L]` | BF16 | `[2816]` |
| `model.language_model.layers.L.post_feedforward_layernorm_1.weight` | `ffn_out_norm_1[L]` | BF16 | `[2816]` |
| `model.language_model.layers.L.post_feedforward_layernorm_2.weight` | `ffn_out_norm_2[L]` | BF16 | `[2816]` |

Three structural facts, each read from the index rather than assumed:

1. **The experts are fused per layer.** One `experts.gate_up_proj` and one
   `experts.down_proj` hold all 128, not 128 tensors each. `1408` is `2 · 704`:
   the gate block followed by the up block along the output axis, which is what
   `chunk(2, dim=-1)` means in the pinned reference.
2. **Every layer carries a dense `mlp` beside its routed experts.** The census
   is 30 of each across 30 layers. It is a shared expert, not an alternative.
3. **The MoE block adds three norms per layer**, not one. The routed and dense
   branches each have their own input and output normalization.

And, as in the 31B, exactly the five global layers have **no**
`self_attn.v_proj.weight` — 25 of 30 layers carry one. That is the serialized
form of `attention_k_eq_v`, not an incomplete download.

### Resources, state and partitions

Byte arithmetic from the declared geometry. **A projection, not a measured
admission.**

| Quantity | Bytes | |
|---|---:|---|
| One expert, one layer | 11,894,784 | 11.34 MiB |
| All 128 experts, one layer | 1,522,532,352 | 1.42 GiB |
| All experts, 30 layers | 45,675,970,560 | **88.5%** of the artifact |
| Dense shared expert, 30 layers | 1,070,530,560 | |
| Embedding (tied, also the output head) | 1,476,395,008 | |
| Router, one layer | 726,784 | |
| Whole artifact, declared tensor payload | 51,611,872,412 | 48.07 GiB |
| Top-k 8 over 30 layers, one token, **no reuse** | 2,854,748,160 | |

Against this machine: the largest single GPU is 24 GiB (25.77 GB) and the
aggregate is 63.9 GiB (68.6 GB) across three unequal cards. **The artifact fits
aggregate VRAM and no single device.** M2 item 4's "intentionally restricted
memory budget smaller than its working weights" is therefore satisfiable by
construction rather than by hoping.

The last row is the one document 03 warns about: it is an upper bound that
assumes no two rows of a batch share an expert. The union of experts a row batch
actually demands is what has to be resident, and task 0019's
`a_real_router_overlaps_routes_and_the_union_is_smaller_than_rows_times_k`
measures that union on routes a real router produced rather than on hand-written
ones.

State per token, at BF16 cache precision: sliding layers `2 · 8 · 256 · 2 B` =
8,192 B; global layers `2 · 2 · 512 · 2 B` = 4,096 B. Whole model per token:
`25 · 8,192 + 5 · 4,096` = 225,280 B, a quarter of the 31B's. A sliding layer's
history is bounded by the 1,024-token window; a global layer's is not.

Legal partitions are **not determined** for the routed operations. `Route` is
`Replicated` by requirement — two ranks that broke a tie differently would
disagree about which expert a row needs, which is a residency divergence as well
as a numerical one. `ExpertMlp` and `Combine` are `NotDetermined` and fail
closed; expert partitioning is **M5**.

### Integration proof

**Nothing in this repository executes this checkpoint.** What exists after task
0019 is a reduced synthetic graph with this variant's shape of mathematics, over
weights the composition root invents.

- **Adapter contains metadata/graph only: yes.** `moxie_models::gemma4` still
  depends on `moxie-types`, `moxie-graph` and `moxie-model-api` and nothing
  else. A new `arch-check` fixture, `models-crate-reaches-memory`, rejects a
  model crate that reaches for the residency authority — the edge a routed
  adapter is most tempted to add.
- **New shared operations and second-consumer tests: done.** `Route`,
  `ExpertMlp` and `Combine` gained parameters, oracles and interpreter support,
  and a second synthetic MoE consumer carries the opposite value for every one
  of them: a different expert count and top-k (`top_k == experts`), SwiGLU
  instead of GeGLU, no shared expert, no per-expert scale, and selection-order
  combination.
- **No private runtime, cache, transfer, sampler or branch evaluator: yes.** The
  routed graph runs through the accepted task 0015 service, task 0013 paged
  transactions and the task 0014 sampler.
- **Residency, expert chunk identity, demand cache and eviction: implemented**
  by [task 0020](../tasks/0020-m2-weight-residency-authority.md), awaiting
  review. `moxie_memory::residency` is the one production weight-residency
  owner. **This artifact's own expert bytes have been read through it**: nine
  distinct experts of layer 0, demanded from routes shaped like a top-k-8 batch,
  **107,053,056 B** read once each against a cache holding four, every range
  verified against an independent read of the same file. That is
  `9 × 11,894,784`, and the union is nine rather than the `3 × 8 = 24` a
  no-overlap bound would charge.

  **Reading is not executing, and this is not model support.** Nothing computed
  with those bytes; no routed layer ran; the selected BF16 chain still refuses
  `Route`, `ExpertMlp` and `Combine`. What made it possible is
  `Shard::read_tensor_range`: the experts are fused, so serving one through the
  whole-tensor reader would have read 1,522,532,352 B to use 11,894,784.
- CPU expert fallback and grouped GPU candidate plans: **not started** — M2
  item 3.
- Reference quality, actual-context prefill/decode: **not started**, and O2.
- GPU and distributed: **not applicable.** No device kernel; the selected BF16
  chain refuses all three routed operations as `UnsupportedKernel`, asserted by
  a test rather than assumed from a catch-all.
- Support matrix updated: gates `G-MOE-ROUTING-HOST`, and `G-RESIDENCY-HOST` /
  `G-RESIDENCY-DEVICE` at task 0020.

### Blockers

1. **Execution of this artifact needs a residency authority it does not have.**
   51.6 GB against a 24 GiB largest device. Task 0020 owns that, and until then
   the routed graph runs only at reduced scale over synthetic weights.
2. **No quality claim of any kind.** Nothing here was compared against the
   released model; that is **O2**, and it is what would also settle the
   5.5.3-versus-5.15 router-softmax dtype question.
3. **One declared numerical deviation from the reference.** The pinned source
   narrows each expert's weighted contribution to BF16 before accumulating; this
   interpreter accumulates the `top_k` terms in FP32 and rounds once at the node
   boundary, as every other operation in `moxie-oracles` does. The reduction
   **order** is pinned regardless. It is **not** covered by the FP32
   `metric::bound` — an earlier draft claimed it was, and a three-expert
   counterexample disproves it — so the honest size is one BF16 ulp of the
   running sum per term, which under cancellation is of the order of the largest
   term. Declared in
   [the task contract](../tasks/0019-m2-routed-expert-semantics.md); O2's to
   close. **Every other boundary the reference has is now implemented**, after
   an independent review found the router's and the expert's missing and a probe
   showed that omitting them changes which experts a row selects.
4. **Vision and audio are out of scope** — **M11**. The artifact is
   `image-text-to-text` and declares both towers.
5. **O1, O2 and O5 remain open.** The owner designated this artifact for M2's
   residency work; that is not catalog membership, a quality statement or a
   conversion authorization.

### Bring-up cost

Inventory and the routed graph cost one session. No checkpoint byte was read
beyond `config.json`, the safetensors headers and the tensor index; nothing was
copied, converted, deleted or downloaded.
