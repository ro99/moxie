# Model bring-up contract — Laguna

Status: **metadata, the routed block, an importer that reads its weights, and
one module repacked into a canonical artifact.** Nothing executes this
checkpoint and no quality claim follows from anything in this record. Since
[task 0024](../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md)
(2026-09-13) its tensors **are** read — three of them import to canonical affine
form, matching document 03's canonical equation over the source's bytes bitwise
and rounding exactly to the source's own BF16 arithmetic — and that is a
statement about a reader agreeing with a file, not about a model.
[Task 0025](../tasks/0025-m3-offline-repack-publication.md) then **published**
one of them: `model.layers.1.mlp.experts.0.down_proj` was repacked into a
1,966,080-byte partial manifest-v1 directory under `/tmp`, reopened through the
production reader, and all **3,145,728** of its values reconstructed and checked
against the canonical equation and the source's own BF16 boundary separately.
That is a statement about **bytes surviving a round trip**, which is
[ADR 0018](../decisions/adr/0018-v1-quality-is-bit-identical-repack.md)'s v1
quality definition and is not evidence about what this model produces. Nothing
was written under any checkpoint root, and **nothing executes a canonical INT4
tensor** — no kernel for one exists. Its attention tower is **not composable** from today's
operation catalogue; the two gaps that block it are named below with the
evidence that they may not be guessed.

Authored for [task 0022](../tasks/0022-m2-laguna-metadata-and-second-consumer.md),
roadmap M2 item 4.

## Identity

| Property | Observed value |
|---|---|
| Family | Laguna, `LagunaForCausalLM`, `model_type` `laguna` |
| Local artifact | `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` (read-only input) |
| Immutable revision | `bc59f497520b23759ce61cc5164ca28bcc4f53bc` |
| `config.json` sha256 | `28e19fc9b58eb6d39b6c24ba21e5feb7dede13c01ddc0f93a5719a3a0aec2f1d` |
| `model.safetensors.index.json` sha256 | `30b6cba031274bcde93f7a4c9f4e3c9259fc4534fd0c5a8e158a8a1d6f4ee8b0` |
| `chat_template.jinja` sha256 | `444819b8ad4612870827ac05b9147fe9e3344d3850cae8c2790898fc514099ff` |
| Shard completeness | **complete**, re-verified 2026-09-13. For each of the 15 shards `8 + header + payload_end` equals the file size exactly, and the payload ends sum to the index's `total_size` of **76,813,095,232 B** |
| Tensor count | **140,989** across 15 shards |
| Base model declared | `poolside/Laguna-S-2.1` |
| License declared | `openmdw-1.1` (OpenMDW License Agreement v1.1), in the artifact's own `LICENSE.md` |
| Declared size | 118B total parameters, 8B active per token (artifact README) |
| Reference implementation declared | `transformers` **5.14.1** (`generation_config.json`); the artifact ships its own `configuration_laguna.py` and `modeling_laguna.py` |
| Canonical artifact profile | **compressed-tensors `pack-quantized`, asymmetric INT4, group 32**, `quantization_status` `compressed` |
| Tokenizer | `tokenizer.json`, 100,352 vocabulary entries; `tokenizer_config.json` names a `tokenizers` backend |
| Trained positional range | `max_position_embeddings` 1,048,576 — **not** an admissible context here (R19) |
| Declared draft model | `poolside/Laguna-S-2.1-DFlash`, method `dflash`, 15 speculative tokens. **Not present on this machine and not inspected.** M9's |
| O1 catalog decision | **RESOLVED 2026-09-13** ([ADR 0017](../decisions/adr/0017-v1-catalog-and-no-quantizer.md)) — Laguna is **number 8** of the ten pinned v1 revisions, at the revision above. Task 0022 recorded this row as OPEN because it was, until the owner ruled mid-task |
| O2 quality decision | **RESOLVED 2026-09-13** ([ADR 0018](../decisions/adr/0018-v1-quality-is-bit-identical-repack.md)) — v1's acceptable loss is a **bit-identical repack** `W=(Q-Z)*S`, publisher quality accepted as-is. Task 0024 shows this artifact's tensors repack bit-identically; **no paired output evidence exists and none is claimed** |
| O5 storage/conversion authorization | **RESOLVED 2026-09-13** ([ADR 0020](../decisions/adr/0020-user-managed-storage-and-canonical-materialization.md), wording corrected by [ADR 0021](../decisions/adr/0021-repack-is-a-moxie-program.md)) — user-managed: Moxie reads the canonical file, repacking is a Moxie program the user runs offline (not an external script), and **no agent-initiated bulk download, copy or conversion** may start without a task naming artifact, revision, expected size and retention. Nothing here has written a byte |

**No download was made.** The artifact finished downloading on 2026-09-12,
before this task, and was verified complete then and again now.

**Remote code was read and never executed.** `modeling_laguna.py` and
`configuration_laguna.py` are the pinned reference for the equations below.
Document 03 forbids executing them, not reading them; the alternative is
guessing a convention from a suffix, which document 03 forbids outright.

## Declared geometry, read from `config.json`

Every row below has an executable counterpart in
`moxie_models::laguna::ARTIFACT`, and
`the_declared_geometry_matches_the_artifact_config` compares the two field by
field against the artifact when it is present.

| Field | Value |
|---|---|
| `hidden_size` / `num_hidden_layers` / `head_dim` | 3,072 / 48 / 128 |
| `num_attention_heads_per_layer` | **48** on layers `l % 4 == 0`, **72** on the rest |
| `num_key_value_heads` | 8 — grouped-query attention at 6:1 and 9:1 depending on the layer type |
| `layer_types` | `full_attention` on layers 0, 4, …, 44 (twelve); `sliding_attention` on the other 36 |
| `sliding_window` | 512 |
| `mlp_only_layers` / `decoder_sparse_step` | `[0]` / 1 — layer **0** is dense, layers 1–47 are routed |
| `intermediate_size` (dense MLP) | 12,288 |
| `num_experts` / `num_experts_per_tok` | **256 / 10** |
| `moe_intermediate_size` | 1,024 |
| `shared_expert_intermediate_size` | 1,024 |
| `hidden_act` | `silu` — the gate transform is **SwiGLU**, for the dense MLP, the shared expert and the routed experts alike |
| `norm_topk_prob` | true |
| `moe_routed_scaling_factor` | **2.5** |
| `moe_router_logit_softcapping` | **0.0 — disabled** |
| `moe_apply_router_weight_on_input` | false (the pinned block raises `NotImplementedError` when true) |
| `gating` / `gating_types` | `per-head`, on **all 48 layers** |
| `attention_bias` / `attention_dropout` | false / 0.0 |
| `rms_norm_eps` | 1e-6 |
| `vocab_size` / `tie_word_embeddings` | 100,352 / **false** — `lm_head` is its own tensor |
| `rope_parameters.sliding_attention` | `default`, theta 10,000, `partial_rotary_factor` 1.0 |
| `rope_parameters.full_attention` | **`yarn`**, theta 500,000, `factor` 128.0, `beta_fast` 32.0, `beta_slow` 1.0, `original_max_position_embeddings` 8,192, `attention_factor` 1.4852030263919618, `partial_rotary_factor` 0.5 |
| `output_router_logits` / `router_aux_loss_coef` | false / 0.0 — no auxiliary loss at inference |

**How this family differs from Gemma 4, axis by axis.** This is why it is a
second consumer rather than a second fixture:

| Axis | Gemma 4 (26B-A4B) | Laguna S 2.1 |
|---|---|---|
| Router input | its own scale-free RMSNorm, a trained gain, then `hidden^(-1/2)` | the block's `post_attention_layernorm` output, unchanged |
| Score transform | softmax over all experts | **sigmoid**, per expert |
| Selection score | the score | score **+ `e_score_correction_bias`** |
| Coefficients | renormalised, then a per-expert **scale** | renormalised, no per-expert scale |
| Routed output | unscaled | **× 2.5**, on the combined row |
| Experts / top-k | 128 / 8 | **256 / 10** |
| Expert gate transform | GeGLU (`gelu_pytorch_tanh`) | **SwiGLU** (`silu`) |
| Expert storage | **fused per layer**, two tensors holding all 128 | **one tensor per expert per projection**, 12,032 of each |
| Shared expert | the dense `mlp`, normalised separately from the routed branch | its own module, reading the **same** norm the router does |
| Layer-type predicate | `(l + 1) % 6 == 0` is global | `l % 4 == 0` is full-attention |
| Attention | `attention_k_eq_v`, per-head Q/K norms, score scale 1.0 | Q/K norms, score scale `head_dim^(-1/2)`, **per-head output gating** |
| Weights | BF16, unquantized | asymmetric INT4 group 32, `pack-quantized` |

## Mathematical inventory

Pinned source: the artifact's own `modeling_laguna.py`, read and never executed.
A "gap" row is a path that **cannot be composed today** and is not composed.

| Component | Source equation / code | Common op and options | Gap task if missing | Oracle fixture |
|---|---|---|---|---|
| Embedding | `LagunaModel.forward` — a plain lookup, **no scale** | `Embedding { scale: 1.0 }` | none | `the_routed_block_composes_and_runs` |
| Output head | `lm_head`, untied; no logit softcap declared | `VocabProjection { softcap: None }` | none | the same |
| Norm | `LagunaRMSNorm` — FP32 reduction, `.to(input_dtype)`, then the gain | `RmsNorm { group: 1, eps: 1e-6 }` | none | task 0003's accepted fixtures |
| Dense / shared MLP | `LagunaMLP.forward` — `down(act(gate(x)) * up(x))`, `act = silu` | `Linear` ×3 and `SwiGlu` | none | task 0003's accepted `swiglu_row` |
| Router score transform | `LagunaTopKRouter.forward` — `F.linear(x, W).float()`, then `torch.sigmoid` | `Route { input: Raw, score: Sigmoid }` | **closed by task 0022** | `the_sigmoid_router_matches_its_fp64_transcription_exactly` |
| Selection bias | `scores_for_selection = routing_scores + e_score_correction_bias`, then `topk`; the coefficients are gathered from `routing_scores` | `Route { selection_bias: true }`, a bound `[experts]` operand | **closed by task 0022** | `the_bias_moves_the_selection_and_not_the_coefficients` |
| Renormalisation | `routing_weights /= routing_weights.sum(-1)` when `norm_topk_prob` | `Route`, unconditional — both families declare true | none | task 0019's accepted fixtures |
| Coefficient narrowing | `routing_weights = routing_weights.to(hidden_states.dtype)` — the router's **last** statement, and the model dtype is BF16. `Gemma4TextRouter.forward` has no such cast | `Route { coefficient: Bf16 }` | **closed after review** — it was missing | `the_coefficients_carry_the_routers_own_output_narrowing`, and the FP64 transcription carries it too |
| Expert feed-forward | `LagunaExperts.forward` — `linear(x, gate_up_proj[e]).chunk(2, -1)`, `act(gate) * up`, `linear(·, down_proj[e])` | `ExpertMlp { activation: SwiGlu }` over the fused `[E, 2I, H]` / `[E, H, I]` tensors | none (task 0019) | task 0019's per-expert FP64 fixture |
| Combination | `index_add_` over `expert_hit`, which is expert-major | `Combine { order: AscendingExpertId }` | none (task 0019) | task 0019's both-orders fixture |
| Routed scaling factor | `expert_output = expert_output * self.routed_scaling_factor`, **after** the accumulation and **before** the shared expert is added | `Combine { output_scale: 2.5 }` | **closed by task 0022** | `the_combine_output_scale_multiplies_the_sum_and_not_each_term` |
| Shared expert | `shared_expert_output = self.shared_expert(hidden_states)`, added after the scale, taking no routing coefficient | **graph composition**, not a routing parameter | none | `the_router_and_the_shared_expert_read_the_same_normalization` |
| Router input tensor | both the router and the shared expert read `post_attention_layernorm(r)` — **one** norm, unlike Gemma 4's three | composition | none | the same test |
| Router logit softcap | `tanh(l / c) * c` when `moe_router_logit_softcapping > 0`; this artifact declares **0.0** | **refused**, not carried: `Route` has no softcap parameter | needs an artifact that declares one | `a_nonzero_router_logit_softcap_is_refused` |
| Attention, GQA and Q/K norms | `LagunaAttention.forward` — `q_norm`/`k_norm` per head before RoPE, `scaling = head_dim**-0.5` | `Attention`, `RmsNorm` | — | task 0016's accepted fixtures |
| **Attention output gating** | `gate = F.softplus(self.g_proj(x).float())`, per head, multiplied into the attention output **before** `o_proj` | **none — no shared operation** | **gap: a new shared semantic operation** | none |
| **Yarn rotary scaling** | `rope_type: "yarn"`, delegated to `ROPE_INIT_FUNCTIONS`, which the artifact does **not** ship | **none — `Rope` has no yarn parameterisation** | **gap: extend `Rope` against a pinned source** | none |
| Sliding-layer rotary | `compute_default_rope_parameters` **in the artifact's own file**: `dim = int(head_dim * partial)` and the exponent denominator is that same `dim` | `Rope { layout: HalfSplit, rotary_dim = frequency_dim = 128, base 10_000 }` | none | task 0016's accepted fixtures |
| Attention sinks | `self.sink` exists only when `swa_attention_sink_enabled`; the config does not declare it and the index holds no `sink` tensor | **absent** | none | — |
| Speculation | `generation_config.json` declares a `dflash` draft model | **absent** | M9 | — |
| Vision / audio | none declared | — | — | — |

### The two gaps, and why neither may be guessed

### A boundary that was missing, and how it got past a bitwise gate

The first version of this record omitted the router's coefficient cast. The
transcription written "independently" from the pinned source omitted it too, so
the two agreed — and their agreement proved only that the same reader had made
the same omission twice. An independent review found it, with the smallest
possible reproduction: on the logits `[0, 1]` the source's coefficients are
`[0.59375, 0.40625]` and the unnarrowed ones are `[0.5938455, 0.4061545]`, and
every combined row downstream carries the difference.

It is post-selection, so it routes no row differently. That is exactly why it
survived: the gates this family's routing had were *selection* exactness and
agreement with a transcription that shared the gap. **A transcription is only
independent of the implementation, not of the reader.** The fixture now checks
the narrowed values against `bf16_round` of an independently computed quotient,
so agreeing for the wrong reason is no longer possible.

`Route` says this about **values, not storage**: a route table holds FP32
coefficients either way and `output_role` still declares `F32`, because that
role is what M2's exit gate reconciles against the ledger. Narrowing a value and
narrowing a buffer are different claims.

**Attention output gating.** The equation is pinned in the artifact's own file,
so it is *known*. What does not exist is a shared operation: `softplus` is not
in the activation catalogue, and the per-head broadcast across `head_dim` needs
a shape operation `OpParams` does not yet carry. Document 02's enforced
extension rule wants a source-linked fixture, a shared operation with an oracle,
a backend or declared fallback, a planner capability and a **second consumer**
before a model graph uses it. `gating_types` is `per_head` on all 48 layers, so
this blocks **every** layer.

**Yarn.** `LagunaRotaryEmbedding.__init__` calls
`ROPE_INIT_FUNCTIONS[self.rope_type]` for anything but `default`, and that table
is in `transformers`, not in the artifact. The artifact declares `transformers`
**5.14.1**; the copy installed on this machine is **5.5.3**. One mismatched copy
is not a pinned exporter — the gemma4 record's own practice was to compare three
copies before pinning a rope reading — and `truncate`, a parameter of that
function, is not declared in this config at all, so its default is exactly the
kind of thing that moves between versions. Guessing it would put a wrong angle
on every position of twelve layers.

Both are recorded as `moxie_models::laguna::Gap`, computed from the declared
geometry by `ArtifactGeometry::tower_gaps` rather than written down, so a
configuration without them reports none.

## Import and resources

### Logical tensor-role mapping

Per layer `L` in `0..48`, from the safetensors index:

| Source name | Role | Notes |
|---|---|---|
| `model.embed_tokens.weight` | `embedding` | BF16 `[100352, 3072]` |
| `lm_head.weight` | `lm_head` | BF16, **untied** |
| `model.norm.weight` | `final_norm` | BF16 `[3072]` |
| `model.layers.L.input_layernorm.weight` | `attn_norm` | BF16 `[3072]` |
| `model.layers.L.post_attention_layernorm.weight` | `ffn_norm` | BF16 `[3072]` |
| `model.layers.L.self_attn.{q,k,v,o}_proj.weight*` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | INT4 on 45 layers, **BF16 on layers 0, 46 and 47** |
| `model.layers.L.self_attn.{q,k}_norm.weight` | `q_norm`, `k_norm` | BF16 `[128]` — per head |
| `model.layers.L.self_attn.g_proj.weight` | `attn_gate_proj` | BF16 `[heads, 3072]`, **unquantized on all 48 layers**. Blocked on the gating gap |
| `model.layers.0.mlp.{gate,up,down}_proj.weight` | `ffn_gate`, `ffn_up`, `ffn_down` | BF16; layer 0 only, at intermediate 12,288 |
| `model.layers.L.mlp.gate.weight` | `router_proj` | BF16 `[256, 3072]`, 47 layers |
| `model.layers.L.mlp.experts.e_score_correction_bias` | `router_selection_bias` | BF16 `[256]`, 47 layers. **The pinned model reads it as `mlp.gate.e_score_correction_bias`** via `_checkpoint_conversion_mapping`; the checkpoint stores it under `mlp.experts` |
| `model.layers.L.mlp.shared_expert.{gate,up,down}_proj.weight` | `shared_gate`, `shared_up`, `shared_down` | BF16, **unquantized**, at intermediate 1,024 |
| `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight_packed` etc. | `experts_gate_up`, `experts_down` | INT4 on 45 layers; **BF16 on layers 46 and 47** |

**Two mapping questions this task does not answer**, both recorded rather than
guessed:

1. **Per-expert on disk, fused in the model.** `LagunaExperts` declares
   `gate_up_proj` `[experts, 2·intermediate, hidden]` and `down_proj`
   `[experts, hidden, intermediate]`; the checkpoint stores
   `mlp.experts.{E}.{gate,up,down}_proj` per expert. The artifact's own
   `_checkpoint_conversion_mapping` remaps **only**
   `e_score_correction_bias` and says nothing about this rewrite. The
   concatenation order is *implied* by `chunk(2, dim=-1)` in
   `LagunaExperts.forward`, and implication is not the pinned exporter. Whoever
   writes the importer resolves it, with evidence.
2. **Which layers are unquantized, and why.** `quantization_config.ignore` lists
   1,788 modules: every module of layers **0, 46 and 47**, every `g_proj`, every
   router `gate`, every `shared_expert`, and `lm_head`. That is a quantizer's
   sensitivity choice, not a format rule, and it must be read from the artifact
   rather than assumed for the family.

### Exact packing parameters, for the importer that does not exist yet

`quantization_config.config_groups.group_0.weights`, verbatim:

```text
num_bits 4, type "int", symmetric false, strategy "group", group_size 32,
observer "mse", actorder null, block_structure null, dynamic false,
scale_dtype null, zp_dtype "torch.int8"
format "pack-quantized", quant_method "compressed-tensors"
compressor version "0.1.dev534+gb269f2e"
```

Read from one shard's headers for `model.layers.1.mlp.experts.0`:

| Tensor | Dtype | Shape | Bytes |
|---|---|---|---|
| `gate_proj.weight_packed` | I32 | `[1024, 384]` | 1,572,864 |
| `gate_proj.weight_scale` | BF16 | `[1024, 96]` | 196,608 |
| `gate_proj.weight_zero_point` | I32 | `[128, 96]` | 49,152 |
| `gate_proj.weight_shape` | I64 | `[2]` | 16 |

`384 · 8 = 3,072` input channels, `3,072 / 32 = 96` groups — consistent with
group 32 and eight INT4 codes per word.

**Two facts the existing importer could not absorb. Both are closed by
[task 0024](../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md)
(2026-09-13), which reads this artifact's tensors:**

- **Asymmetric.** `PackQuantizedSpec` refused an asymmetric source through task
  0018, whose accepted scope was symmetric INT8 group 32. It now carries a
  named `ZeroPointSource` instead of a `symmetric` boolean, because the boolean
  said only that *some* zero points exist and the importer has to know **where**.
- **The zero points are packed along the *output* axis.** `weight_packed` is
  `[1024, 3072/8]` — eight codes per word along the **input** axis — while
  `weight_zero_point` is `[1024/8, 96]`, eight per word along the **output**
  axis. Two packing conventions in one tensor group, and the reader had seen
  only the first. The mapping is pinned to `compressed-tensors` 0.17.0's
  `pack_to_int32(..., packed_dim=0)` and **measured** against this artifact's
  own codes, which the symmetric case could never do: a zero-point word's lanes
  are different output channels, so their statistics separate the readings
  ([experiment 0005](../evidence/experiments/0005-asymmetric-int4-zero-point-assignment.md)).

**Reading a tensor is not executing one.** `model.layers.1.mlp.experts.0`'s
three projections import to canonical affine form; every reconstructed value is
bitwise equal to document 03's canonical FP32 `(q - z) * s` over the source's
own bytes, and rounds exactly to the value the source's own reference computes,
which applies a **BF16** boundary because these artifacts' scales are BF16. The
two are different quantities and an independent review found the first version
of this record calling the first one the second. Nothing computes with them:
W4A16 is M3 item 3 and no kernel exists. **No quality claim follows** (**O2**),
and nothing was written under any checkpoint root (**O5**).

**One thing the import turned up about this artifact's own shape.** Its index
holds 140,989 tensors, so one shard's header is about a megabyte serialized and
its admitted peak estimate is 16.5 MB — above `HeaderBudget::DEFAULT`'s 8 MiB,
which is calibrated on Gemma 4's 17–64 KB headers. The default refusing it is
the budget working; any path that opens this artifact has to state a budget.

### Cost, at this artifact's scale

| Quantity | Value |
|---|---|
| One routed expert, as stored (INT4 + scales + zero points + shape) | **5,455,920 B** |
| One routed expert, in BF16 at the same logical shape | **18,874,368 B** |
| One routed layer's 256 experts, **quantized** (45 of them) | **1,396,715,520 B** |
| One routed layer's 256 experts, **BF16** (layers 46 and 47) | **4,831,838,208 B** |
| All 47 routed layers' experts, as stored | **72,515,874,816 B** — **94.41%** of the artifact |
| Whole artifact | 76,813,095,232 B |

**That total is a sum over the layers, not a multiplication.** An earlier
version of this record multiplied the quantized per-layer cost by all 47 routed
layers and reported 65,645,629,440 B / 85.5% — understating the expert working
set by **6,870,245,376 B**, in a document that had already recorded two
paragraphs earlier that layers 46 and 47 keep BF16 experts. An independent
review found it. The arithmetic is executable now:
`ArtifactGeometry::expert_bytes_total` sums `layer_expert_bytes` over the
routed layers, and `the_expert_inventory_matches_the_artifact_headers` checks
every layer against the artifact's own `data_offsets`.

At 76.8 GB the artifact fits neither a 24 GiB device nor the machine's
**measured 62.6 GiB** of aggregate device memory — `cuDeviceTotalMem` reports
24,123 MiB on each 3090 and 15,883 MiB on the 5060 Ti, per
[the hardware inventory](../evidence/hardware-inventory.md) — with room for
activations and state. Executing it needs M2's host-backed residency **and** an
importer that does not exist. An earlier version of this line said "the
aggregate 72 GiB of the three cards", which was neither the nominal 64 GiB nor
the measured figure this repository had already recorded.

### State schema, partitions and admission

Not established. The state schema needs the attention tower, which is blocked;
`sliding_window` 512 on 36 of 48 layers and full attention on 12 is the shape
task 0017's per-layer paged geometry would carry, but nothing here has been
built against it. Legal partitions are M5's. No admission figure for this model
has been computed, because nothing can plan it.

## Integration proof

- **The adapter contains metadata and graph composition and nothing else.**
  `moxie-models`' dependency list is `moxie-types`, `moxie-graph`,
  `moxie-model-api`, and `arch-check`'s "forbidden construct in model source"
  rule passes. No allocation, no file read, no cache, no loop, no kernel.
- **New shared operations, with independent oracles and a second consumer:**
  `Route` gained `RouterInput`, `RouteScore` and a selection-bias operand;
  `Combine` gained `output_scale`. Each has an FP64 transcription in
  `moxie-oracles` written from the pinned source, and each is exercised by
  **two** consumers carrying opposite values — the Gemma-like reduced graph and
  the synthetic MoE fixture, whose routing parameters are the opposite of it on
  every axis.
- **No private runtime, cache, transfer, sampler or branch evaluator.**
  `moxie-memory` gained nothing; `arch-check`'s "a second weight-residency
  owner" rule passes.
- **Reference quality: none, and none is claimed.** O2 is **resolved** for v1 as
  a bit-identical repack with the publisher's own quality accepted as-is
  ([ADR 0018](../decisions/adr/0018-v1-quality-is-bit-identical-repack.md));
  what task 0024 shows is that this artifact's tensors repack bit-identically at
  tensor scale, which is the repack half and not a statement about output.
- **What runs:** a synthetic stack of Laguna-shaped routed blocks, through the
  shared host interpreter, over weights the test invents. It is row-independent,
  which is the stateless form of whole-versus-chunked parity. **It is not
  Laguna and its output is not model output.**
- **What runs on hardware:** a routed layer at the artifact's **declared expert
  shape** — top-k 10, `hidden` 3,072, `moe_intermediate` 1,024 — over twelve
  synthetic experts, on all three GPUs, against a device budget of one eighth of
  what its route demands. It agrees with the CPU candidate on every component.
  Again: synthetic weights, a route the test writes.

## Blockers

1. **The attention tower**, on the two gaps above. Until both are closed by
   their own tasks, no Laguna layer is composable and no state schema exists.
2. ~~**The importer**: asymmetric INT4 group 32 with output-axis-packed zero
   points.~~ **Closed 2026-09-13 by
   [task 0024](../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md), accepted by the
   owner after three rounds of independent review.**
   What replaces it is narrower: **nothing executes a canonical INT4 tensor.**
   The W4A16 path is M3 item 3, and the graph's `TensorRequirement`s stay BF16
   until one exists, because listing INT4 would advertise a path that is not
   there.
3. **The fused/per-expert mapping**, which the artifact's own conversion mapping
   does not cover.
4. ~~**O1**~~ — resolved 2026-09-13; Laguna is number 8 of the v1 catalog.
   Catalog membership is not support: the blockers above are what support needs.
5. **O2** — resolved as repack-only, and the repack half is now shown at tensor
   scale. **No paired output against the released model exists**, which is what
   any statement about this model's quality would need.
6. ~~**O5**~~ — resolved as user-managed. Still no bulk write without a task
   naming artifact, revision, expected size and retention.
7. **Context**: 1,048,576 declared is not an admissible context here, and the
   yarn gap means the long-context positional path is exactly the part that is
   not established.
8. **The `dflash` draft model** is not on this machine and has not been
   inspected. M9's.

## Bring-up cost

One session on 2026-09-13, within [task 0022](../tasks/0022-m2-laguna-metadata-and-second-consumer.md),
plus one within [task 0024](../tasks/0024-m3-asymmetric-int4-pack-quantized-import.md)
the same day for the importer — which is shared work in `moxie-format` that this
family happened to be the occasion for, and which three other local artifacts
consume unchanged.
By ownership: `moxie-models` gained one module; `moxie-graph`, `moxie-oracles`
and `moxie-interp` gained the routed parameters this family needs;
`moxie-memory` gained nothing. Three shared operation **parameters** added, all
with oracles; two shared operations **not** added and recorded as gaps instead.
No performance figure, because nothing that could be benchmarked exists for this
family. O7 review is not due until a family actually executes.
