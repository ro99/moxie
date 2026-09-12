# Model bring-up contract — Gemma 4 31B-IT

Status: **inventory only, 2026-09-12.** Nothing here is implemented, imported or
executed. This record exists because roadmap M1.5 requires a Gemma text graph and
document 09 §B requires the mathematical inventory *before* implementation. It
resolves no owner gate and makes no support claim.

## Identity

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

## Mathematical inventory

Source citations are paths relative to the frozen legacy root
`/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`,
followed by line number, per document 08's citation convention.

| Component | Source equation / code / line | Common op and options | Gap task if missing | Oracle fixture |
|---|---|---|---|---|
| Token embedding | `src/models/gemma4/gemma4_runtime.cpp:699` — `e = bf16(row · bf16(sqrt(hidden_size)))` | `Embedding` **plus a checkpoint-defined output scale**; today `OpParams::Embedding` has no scale field | **gap** — task 0016 | new; exact BF16 rounding of a known row |
| Output head | `gemma4_runtime.cpp:1290` then `matmul_softcap`; cap applied as `bf16(bf16(tanh(bf16(bf16(x)/c))) · c)`, `kernels/cuda/detail/backend_kernels.cuh:905` | `VocabProjection` **plus logit softcap 30.0**; tied to the embedding matrix | **gap** — task 0016 | new; FP64 transcription of the four-step cap |
| RMSNorm | `src/models/gemma4/gemma4_ops.cpp:11` over `src/platform/numerics.cpp:10`; `y = x · rsqrt(mean(x²) + eps) · w`, FP64 sum, then a BF16 rounding of the result. The gain is applied **as stored**, with no `1 + w` rebias | `RmsNorm { hidden, eps }` — already present, semantics already match | none | existing `moxie-oracles::norm::rms_norm_row` |
| Q/K normalization | `gemma4_runtime.cpp:798` and `:810` — per head, over `head_dim`, with the layer's `q_norm` / `k_norm` gains, before RoPE | `RmsNorm` at head width | none (composition) | existing |
| V normalization | `gemma4_runtime.cpp:818` — RMSNorm with an **all-ones gain** and the same epsilon, applied to V and never to K | `RmsNorm` with a unit gain tensor | none (composition); the unit gain must be an explicit tensor, not an implicit special case | existing |
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

## Import and resources

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
- Rollback method: the accepted task 0013 paged transactions. A sliding-window
  layer additionally needs the ring-eviction boundary tested, which M4 owns.

## Integration proof

**None. Nothing in this repository executes this or any checkpoint.**

- Adapter changes contain metadata/graph only: not started.
- New shared ops and second-consumer tests: contracted in
  [task 0016](../tasks/0016-m1-gemma-reduced-graph.md), not implemented.
- No private runtime/cache/transfer/sampler/branch evaluator: not started.
- Reference quality, actual-context prefill/decode and continuation: not started.
- CPU/GPU precision, state, cancellation and distributed tests: not started.
- Common sampling, future entropy and speculation capability results: not started.
- HTTP/CLI/template/tokenizer/modality parity: not started; M8 owns the surfaces.
- Temporary code deleted; support matrix updated: not applicable yet.

## Blockers

1. **Execution of this artifact is M3-blocked.** Every language-model linear is
   INT8 `pack-quantized`. Moxie has `moxie-format::affine` host decode for INT8
   codes, but no compressed-tensors importer, no packed-layout reader, no
   canonical repack and no W8A16 execution path. M3 owns all four. This record
   does not authorize a private loader, a dequantization fallback or a
   model-owned decode path as a way around that.
2. **The text graph mathematics are recoverable but incomplete in shared code.**
   Six operation parameters are missing, listed above and contracted in task
   0016. They are implementable now, against synthetic BF16 weights, and do not
   depend on M3.
3. **Vision is out of scope** and must be refused explicitly rather than ignored:
   the artifact is `image-text-to-text`, its mask rule has a multimodal group
   exemption, and dropping that exemption silently would change text results for
   any prompt containing image tokens.
4. **O1, O2 and O5 remain open.** No catalog membership, no quality statement, no
   conversion.

## Bring-up cost

Not started. Inventory reading on 2026-09-12 cost one session and produced no
code. Record engineering hours, files/lines by ownership, new versus reused
shared operations, initial untuned performance and the O7 review when the family
is actually brought up.
