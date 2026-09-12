# Checkpoint inventory — M0

Local artifact roots for current and future inspection are [`/models` and
`/fast/models`](artifact-roots.md). The root designation is a location contract,
not an assertion that every download is complete or approved; record exact
subdirectories, revisions and hashes in this inventory as they are inspected.

**Historical inventory under the initial NVFP4 design.** Preserve the measured source facts below. The precision options/questions here are superseded by [ADR 0003](../decisions/adr/0003-int4-int8-bf16-weight-family.md): initial execution is INT4/INT8/BF16, and no FP4-to-INT4 bulk conversion has been authorized. See [the new owner candidate metadata](quantization-candidates.md); those remote candidates have not been downloaded by this review. The presence of FP4 checkpoints on disk does not make them the mandatory starting catalog.

Captured 2026-09-07 by reading `config.json`, `hf_quant_config.json` and
safetensors headers directly. No checkpoint was downloaded, converted or
modified. Nothing here is an O1 catalog decision.

## The headline: three checkpoints, none of them the roadmap's starting point

Document 06's provisional bring-up order is "Gemma text for a dense slice;
Laguna for early MoE; Inkling ... GLM-5.2 for MLA; Kimi and GLM-5.3 for
recurrent/hybrid state; DeepSeek for compressed sparse attention". Of the seven
families, **two are present on disk and five are absent**:

| Family | Checkpoint on disk | Note |
|---|---|---|
| Gemma | **absent on 2026-09-07; present since** | see the [2026-09-12 update](#update-2026-09-12--gemma-4-31b-it-int8-is-now-on-disk) |
| Laguna | **absent** | M2's proposed first MoE workload has no artifact |
| Inkling | **absent** | |
| GLM-5.2 | **absent** | M4's proposed MLA case has no artifact |
| Kimi K3 | present, 1.5 T | mixed MXFP4/BF16 — see below |
| GLM-5.3 | present, two variants | 433 G and 178 G |
| DeepSeek | **absent** | |

Document 06 anticipates this: "If actual artifact availability changes that
order, choose an equivalent stress case by ADR." M1 cannot begin against a real
Gemma checkpoint, and M2 cannot begin against Laguna. Both can still begin
against reduced synthetic graphs, which document 06 M1.5 explicitly provides for
— "A reduced synthetic graph proves contracts; actual checkpoint execution, when
available/admitted, proves integration. Never describe synthetic output as model
support."

This is an **O1 question** and it blocks the bring-up order, not just the final
catalog. Batched below with the rest.

## What is present

### 1. Kimi K3 — `/data/kimi-k3`

| Property | Value |
|---|---|
| Size on disk | 1.5 TB, 96 shards |
| `model_type` | `kimi_linear` (`KimiLinearForCausalLM`, wrapped by `KimiK3ForConditionalGeneration`) |
| Layers / hidden | 93 / 7168 |
| Attention | 96 heads; hybrid — `linear_attn_config` lists full-attention layers every 4th, the rest KDA/linear |
| Experts | 896 routed, `moe_intermediate_size` 3072, plus shared experts |
| Max positions | 1,048,576 |
| Vocab | 163,840 |
| License | Moonshot AI, `LICENSE` present |
| Modality | Vision-capable (`kimi_k3_vision_processing.py`, media placeholder token) |

**Precision, verified from safetensors headers, not from the config's claim.**
Routed expert weights are `U8`-packed **MXFP4**: `format: mxfp4-pack-quantized`,
`group_size: 32`, `scale_dtype: torch.uint8` (E8M0). Shared experts, norms,
attention and embeddings are `BF16`/`F32`. A sampled MoE shard holds 5376 `U8`
tensors against 21 `BF16`.

**MXFP4 is not a canonical Moxie family.** Document 01 fixes NVFP4, INT8 and
BF16; document 03 (R13) says the legacy MXFP4 work is "a **copy/lifetime
lesson**, not the new canonical format choice", and that legacy low-bit codecs
are "import/oracle assets, not additional mandatory runtime families". So this
checkpoint cannot be imported into the canonical schema without a decision:

- Convert MXFP4 → NVFP4. This is **double quantisation** of an already 4-bit
  checkpoint. Document 03: "re-quantizing an already low-bit checkpoint can lose
  quality and must be explicitly labeled and evaluated. Never silently
  double-quantize a library of checkpoints." Needs O2 evidence.
- Add MXFP4 as a fourth canonical weight family. An owner-level scope change.
- Obtain a higher-precision original. Document 03 prefers this, and O5 asks
  exactly this question.

**Tokenizer.** No `tokenizer.json`. It is tiktoken-based (`tiktoken.model` plus
`tokenization_kimi.py`). Document 03 forbids a required Python interpreter in the
production server and forbids remote checkpoint code execution, so the `.py`
files are reference material only. Legacy Strata already implemented this in C++
(`src/models/common/tokenizer.cpp`, `tests/test_kimi_k3_tokenizer.cpp`) — R24
reusable asset.

### 2. GLM-5.3 NVFP4 — `/fast/models/incoai/GLM-5.3-NVFP4`

| Property | Value |
|---|---|
| Size on disk | 433 GB |
| `model_type` | `glm_moe_dsa` (`GlmMoeDsaForCausalLM`) |
| Layers / hidden | 78 / 6144 |
| Attention | 64 heads, `head_dim` 192; DSA sparse indexer, `index_topk` 2048, `index_n_heads` 32, per-layer `full`/`shared` indexer pattern |
| Experts | 256 routed, top-8, 1 shared, `moe_intermediate_size` 2048, first 3 layers dense |
| Max positions | 1,048,576 |
| Vocab | 154,880 |
| License | Z.AI, `LICENSE` present |

**Precision.** Genuine NVFP4: `U8` packed codes, `F8_E4M3` block scales,
`F32` global scales, `group_size: 16`. Producer is **ModelOpt 0.45.0**
(`hf_quant_config.json`, `quant_algo: NVFP4`), which is the *multiplying* global
scale convention — R16's distinction from GLM's compressed-tensors *dividing*
convention, and exactly the trap document 03 says to normalise at import.

**It is W4A4, not weight-only.** The config carries
`input_activations: {num_bits: 4, type: float, group_size: 16}` and the tensors
include `input_scale`. Document 03 pins the initial canonical profile as
**weight-only** NVFP4 with BF16 activations, and says "Native W4A4 or FP8
activation paths are separate optional profiles requiring quality approval, not
automatic consequences of choosing NVFP4 weights." So this artifact's activation
scales are outside the initial profile; the weights can still be imported
weight-only, discarding `input_scale`, which is a different numerical contract
from what the publisher validated. That difference must be measured, not assumed
harmless — O2.

`lm_head`, `embed_tokens` and layer 0's MLP are excluded from quantisation,
matching document 03's sensitive-tensor policy.

### 3. GLM-5.3-Flash NVFP4 (abliterated) — `strata/models/glm53f-nvfp4`

| Property | Value |
|---|---|
| Size on disk | 178 GB, 62 shards |
| `model_type` | `glm5_next_text` (`Glm5NextForConditionalGeneration`) |
| Layers / hidden | 45 / 4096 |
| Attention | 64 heads; KDA linear attention (`short_conv_kernel_size` 4, `gate_lower_bound` -5.0) |
| Experts | 288 routed, top-8, 1 shared |
| Max positions | 1,048,576 |
| Vocab | 154,880 |
| License | README front-matter says **MIT**; base model `zai-org/GLM-5.3-Flash` |

**Precision.** `compressed-tensors`, `nvfp4-pack-quantized`, `group_size: 16`,
`weight_global_scale` — the *dividing* convention, the other half of R16. Only
expert `gate/up/down` in layers 3–44 are quantised; the rest is BF16.

Also carries `hc_attn_base`, `hc_attn_scale`, `hc_ffn_base`, `hc_ffn_scale` per
layer — **mHC residual mixing**, which document 02 lists as a required shared
semantic operation.

Two cautions. It is an **abliterated** derivative, not the released reference
model, so it is unsuitable as a quality baseline: document 07 measures against
"the released model", and this is not it. And it lives **inside the read-only
legacy checkout**, which must not be modified; if it is to be used, it should be
referenced in place or copied out, never converted in situ.

## Together, these are a useful accident

The two GLM artifacts are the two *opposing* NVFP4 global-scale conventions
described in R16 — ModelOpt multiplying, compressed-tensors dividing — in the
same architecture family. That makes them an unusually good M3 import-validation
pair, and validating both against one canonical equation is the concrete way to
discharge R16.

## Storage headroom

Aggregate device memory is 62.6 GiB. **Every checkpoint here exceeds it**, by
2.8x for the smallest. Host-backed weights, expert streaming and demand loading
are the ordinary path for this catalog, not a degraded mode.

| Mount | Free | Holds |
|---|---|---|
| `/` | 551 G | — |
| `/fast` | 1.4 T | GLM-5.3-NVFP4 (433 G) |
| `/data` | 286 G | kimi-k3 (1.5 T) |
| `/archive` | 1.3 T | unrelated (spinning disk) |

Converting any of these to a canonical artifact roughly doubles its footprint
until the source can be released. Converting GLM-5.3-NVFP4 (433 G) fits on
`/fast`; converting Kimi K3 (1.5 T) does not fit anywhere as a second copy.
**This is O5** and no conversion may start before it is answered.

## Not established

- Whether any of these revisions is the intended release catalog (**O1**).
- Whether a higher-precision original of Kimi K3 is obtainable (**O5**).
- Checksums. None recorded yet; document 03 requires source checksums in the
  manifest, and a resumable conversion checks them.
- Feasible context per model on this hardware. `max_position_embeddings` is
  1,048,576 for all three; that is the trained range, not an admissible context
  here, and R19 is the standing warning against confusing the two.
- Full license terms. Only the first lines were read; redistribution and
  derivative terms matter before any fixture is exported from these weights
  (document 06 M0.6 requires permission to be verified first).

## Update 2026-09-12 — Gemma 4 31B-IT INT8 is now on disk

Read-only inspection on 2026-09-12 while preparing the M1.5 contract. No file was
copied, converted, modified or executed; no checkpoint-provided code was run.
This changes the artifact facts below and changes **no** owner gate.

### Roots as they stand today

`/models` is empty. `/fast/models` holds three directories: `canada-quant`,
`cyankiwi` and `Intel`. Under `cyankiwi` there are four artifacts —
`gemma-4-31B-it-AWQ-8bit`, `Inkling-Small-AWQ-INT4`, `Muse-Glimmer-30B-AWQ-INT4`
and `Qwen3.8-27B-AWQ-BF16-INT4`. Only the Gemma directory was inspected in
detail; the other three are recorded here as present, nothing more.

Two of the three M0 checkpoints above are **no longer at the paths this document
records**. `/fast/models/incoai/GLM-5.3-NVFP4` and
`strata/models/glm53f-nvfp4` do not exist today; `/data/kimi-k3` still does and
was not re-verified. This agent did not delete anything and does not know where
those two went, so the M0 sections are left standing as the record of what was
measured on 2026-09-07 rather than edited to match today. `/fast` now reports
742 G used of 1.8 T.

### `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit`

| Property | Value |
|---|---|
| Source repository | `cyankiwi/gemma-4-31B-it-AWQ-8bit` |
| Immutable revision | `34ca187d836de874b2c7e3edf48f439b9f583772` |
| Base model declared | `google/gemma-4-31B-it` |
| License in README front matter | `apache-2.0`, Gemma 4 license link |
| Size | 35,089,877,112 B of tensor payload across seven shards |
| Tensors | 2008 — 1188 `BF16`, 410 `I32`, 410 `I64` |
| Text precision | compressed-tensors `pack-quantized`, INT8, symmetric, **group 32** |
| Vision tower | BF16, excluded from quantization by the 192-entry `ignore` list |

**Revision identity is confirmed, not assumed.** The local `config.json` hashes
to `f9f7b7c592c98a99018843aa242b05cbd19f98a423ce43241d3ef208ec444b76`, which is
byte-identical to the hash recorded for this candidate in
[quantization-candidates.md](quantization-candidates.md) on 2026-09-07 from the
public API. The directory's `.cache/huggingface/` download metadata names the
same commit for every file.

**Shard completeness is verified three ways.** The index's `weight_map`
references exactly the seven shard files present, with no missing and no
unreferenced file; each shard's safetensors header accounts for its file size
exactly (`8 + header + payload == size`) and its tensor set equals the index's
entries for that shard; and the sum of tensor payloads equals the index's
declared `total_size` of 35,089,877,112 B.

**Content hashes, computed locally and compared with the publisher's declared
LFS hashes in the artifact's own download metadata. All eight match.**

```text
852af1be0e6c2f40f5f5fbce5f4b0a05b596004086e40fa4cba3bf15cf00b311  model-00001-of-00007.safetensors
ed73e91ffe05617f737f07b002c8144638cf8cee961c84532e98f95657e04b83  model-00002-of-00007.safetensors
69f5c0848ab9f4696120eef02f271998b4a0a152f5e5532e149768efb724527e  model-00003-of-00007.safetensors
c0aaf5ca502a9cdfe1ed6778d126c3de2ce3ec22cb53fd3376810eab5fc0ba99  model-00004-of-00007.safetensors
d5c771d8b9ae4a6752b2b54e654650a873abc4e58472c1d234a65ca90a9ad0ec  model-00005-of-00007.safetensors
c8e09c17d307a8bdff98072f3fb7011815846b782be8b2653fd7499fc52452b1  model-00006-of-00007.safetensors
dc209d902322575b8c7091ae425e6e35c5eadce8af9265ce530259a7cd77cbd3  model-00007-of-00007.safetensors
cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f  tokenizer.json
```

Small metadata files, hashed locally; the publisher declares no LFS hash for
these, so they are recorded as observed rather than as verified against a source:

```text
f9f7b7c592c98a99018843aa242b05cbd19f98a423ce43241d3ef208ec444b76  config.json
d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de  generation_config.json
90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df  tokenizer_config.json
85a08664d16d8f3be4416c92427b3ac10df1024ac566cc0b4bc3bab409393f98  chat_template.jinja
32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c  processor_config.json
d6d47cb0547c6090c4b81aa5fe90d72026c2124b06c53c2d9efeec0e693fd421  model.safetensors.index.json
```

Retention: the artifact stays where it is, read-only. Nothing is copied out.
The full geometry, tensor-role mapping, packing parameters and equation
inventory are in [the Gemma 4 bring-up record](../models/gemma4.md).

### What this does and does not change

It changes the M0 headline above in one respect only: the roadmap's proposed
dense M1 slice now has an artifact on disk. It does **not** make that artifact
executable here. Every language-model linear is INT8 `pack-quantized`, and Moxie
has no compressed-tensors importer, no packed-layout reader and no W8A16
execution path — all four are M3. M1.5 therefore proceeds against a reduced
synthetic graph, which document 06 explicitly provides for, under
[task 0016](../tasks/0016-m1-gemma-reduced-graph.md).

O1 is still open and still blocks the bring-up order and the catalog; a
downloaded artifact is not a catalog decision. O2 is still open and no quality
statement exists. O5 is still open and nothing may be converted or requantized.
Laguna, GLM-5.2 and DeepSeek remain absent under these roots.
