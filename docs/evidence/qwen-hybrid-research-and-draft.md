# Qwen hybrid (Hemmingway-1): research note and task contract draft

Reviewer (Opus), read-only research for the coordinator, 2026-09-26. Everything
below was read on this machine. Nothing was downloaded, executed from the
checkpoint, or written outside `/tmp/claude-1000/`.

Part 1 is the research note. Part 2 lists owner decisions (not decided here).
Part 3 is the paste-ready contract. Part 4 is the reviewer's safety
assessment of "one task".

---

## Part 1. Research note

### 1.1 The checkpoint (`/fast/models/Altworld/Hemmingway-1`)

- **Revision:** `b987f1800da287b3016862948743a253fbf80553` (first line of
  `.cache/huggingface/download/config.json.metadata`).
- **Files:**
  - 12 shards plus `model-mtp.safetensors`;
  - `model.safetensors.index.json` with 866 entries, all present in the shard
    headers;
  - README license `cc-by-nc-4.0`, base `Qwen/Qwen3.8-27B`.
- **`config.json`** (root object, **no `text_config`**):
  - `architectures ["Qwen3_5ForCausalLM"]`, `model_type qwen3_5_text`,
    `dtype bfloat16`, `transformers_version "5.17.0"`;
  - `hidden_size` 5120, `num_hidden_layers` 64, `intermediate_size` 17408,
    `vocab_size` 248320, `rms_norm_eps` 1e-6, `hidden_act silu`,
    `tie_word_embeddings false`, `attention_bias false`,
    `max_position_embeddings` 262144;
  - `layer_types`: 3 × `linear_attention` then `full_attention`, repeated, so
    `full_attention_interval` 4 and full layers `L ≡ 3 (mod 4)`: 16 full,
    48 linear;
  - **Full attention:** `num_attention_heads` 24, `num_key_value_heads` 4,
    `head_dim` 256, `attn_output_gate true`, **`output_gate_type "swish"`**;
  - **RoPE:** `rope_parameters {rope_type default, rope_theta 10000000,
    partial_rotary_factor 0.25, mrope_interleaved true, mrope_section
    [11,11,10]}`;
  - **Linear attention:** `linear_num_key_heads` 16, `linear_num_value_heads`
    48, `linear_key_head_dim` 128, `linear_value_head_dim` 128,
    `linear_conv_kernel_dim` 4, **`mamba_ssm_dtype "float32"`**;
  - **MTP:** `mtp_num_hidden_layers` 1, `mtp_use_dedicated_embeddings false`
    (not run).
- **Tensors: every tensor is BF16**, including `A_log`, `dt_bias` and every
  norm. The task asked whether `A_log`/`conv1d`/`dt_bias`/`norm` are FP32:
  **they are not**. The per-layer patterns (layer `L`, prefix `model.layers.L.`):

  | Name | Shape | Layers |
  |---|---|---|
  | `input_layernorm.weight`, `post_attention_layernorm.weight` | [5120] | all |
  | `mlp.{gate,up}_proj.weight` | [17408, 5120] | all |
  | `mlp.down_proj.weight` | [5120, 17408] | all |
  | `linear_attn.in_proj_qkv.weight` | [10240, 5120] | linear |
  | `linear_attn.in_proj_z.weight` | [6144, 5120] | linear |
  | `linear_attn.in_proj_{a,b}.weight` | [48, 5120] | linear |
  | `linear_attn.conv1d.weight` | [10240, 1, 4] | linear |
  | `linear_attn.{A_log,dt_bias}` | [48] | linear |
  | `linear_attn.norm.weight` | [128] | linear |
  | `linear_attn.out_proj.weight` | [5120, 6144] | linear |
  | `self_attn.q_proj.weight` | [12288, 5120] | full |
  | `self_attn.{k,v}_proj.weight` | [1024, 5120] | full |
  | `self_attn.o_proj.weight` | [5120, 6144] | full |
  | `self_attn.{q,k}_norm.weight` | [256] | full |

  Global: `model.embed_tokens.weight` [248320, 5120], `model.norm.weight`
  [5120], `lm_head.weight` [248320, 5120] (**untied**). `mtp.*` (14 patterns)
  is excluded.
- **Sizes (summed from headers):**
  - text weights 53,791,996,928 B (MTP 849,398,784 B more);
  - a linear-attention layer is 766,546,368 B, a full-attention layer
    744,510,464 B, and the embedding and `lm_head` are 2,542,796,800 B each.
- **State per sequence:**
  - recurrent: 48 layers × 48 heads × 128 × 128 × FP32 = **150,994,944 B**;
  - convolution history: 48 × 10240 × 3 × BF16 = 2,949,120 B;
  - KV (16 full layers): 4 KV heads × 256 × 2 × BF16 = **65,536 B/token**,
    2 GiB at 32,768 tokens.

### 1.2 The exact mathematics

Source: `transformers` 5.5.3 (the installed copy),
`models/qwen3_5/modeling_qwen3_5.py`. Line numbers refer to that file.
`fla` and `causal_conv1d` are **not installed**, so the reference runs the
torch fallbacks (`is_fast_path_available` false, line 205).

**Decoder layer** (731–783). Pre-norm; `h = x + mixer(norm(x))`, then
`h + mlp(norm(h))`. Residual adds in the activation dtype (775, 781).

**`Qwen3_5RMSNorm`** (711–725). **`(1 + w)` gain**: `out = norm(x.float()) *
(1.0 + w.float())`, then one cast (724). The weight initializes to zeros (714).
Moxie's `RmsNorm` oracle multiplies by `w` (`moxie-oracles/src/norm.rs:51`),
so it is **not the same op**.

**MLP** (695–708). `down(silu(gate(x)) * up(x))` in BF16. SwiGLU: the oracle
exists (`moxie-oracles/src/activation.rs:72`) and the interpreter supports it
(`moxie-interp/src/lib.rs:977`), but there is **no device kernel**
(`moxie-kernels/cuda/dense_ops.cu` has `geglu` only, line 210).

**Full attention** (`Qwen3_5Attention`, 620–692):
1. `q_proj` output viewed `[R, 24, 512]` and `torch.chunk(…, 2, dim=-1)` (658).
   Per head, the first 256 values are the **query** and the next 256 the
   **gate**. The layout is **per-head interleaved** in `q_proj`'s rows: rows
   `[512h, 512h+256)` are query, `[512h+256, 512h+512)` gate.
2. `q_norm`, `k_norm`: per-head `Qwen3_5RMSNorm` with the `(1+w)` gain
   (640–641). `v` is not normalized.
3. RoPE via `apply_rotary_pos_emb` (544–579):
   - rotary width = `cos.shape[-1]` = 64;
   - `rotate_half` over the first 64 lanes only, pairing `(j, j+32)` for
     `j < 32`; lanes 64–255 pass through;
   - `inv_freq = θ^(−2j/64)`, `j < 32`, θ = 1e7 (106–137, `dim =
     int(256·0.25)` at 127);
   - `cos`/`sin` are computed in FP32 and **cast to BF16** (`cos.to(x.dtype)`,
     155), so the rotation arithmetic runs in BF16.
   - **Moxie's device RoPE pairs `(j, j + head_dim/2)`**
     (`dense_ops.cu:189-190`, `half = head_dim / 2`), which is Gemma's
     convention. For Qwen that pairs lane 5 with lane 133. **Wrong for Qwen.**
4. **Interleaved mRoPE for text-only positions.** `position_ids` are expanded
   identically across T/H/W (1243–1247). `apply_interleaved_mrope`
   (157–172) only copies frequency slots from H and W into T, and all three
   are equal for text, so **it is the identity**. Text-only Qwen RoPE is
   plain partial rotary with the pairing above. Record this as a checked
   equality (Part 3, test T2), not an assumption.
5. Scale `256^−0.5 = 1/16` (629); GQA 24/4 (repeat_kv, 582).
6. **Output gate:** `attn_output * torch.sigmoid(gate)` (689), in BF16.
   **But the checkpoint declares `output_gate_type "swish"`, a field 5.5.3
   never reads** (see D1).
7. `o_proj` [5120, 6144].

**Gated DeltaNet** (`Qwen3_5GatedDeltaNet`, 356–533), per linear layer, input
`x` [R, 5120] BF16:
1. `qkv = in_proj_qkv(x)` [R, 10240]; `z = in_proj_z(x)` [R, 6144];
   `b = in_proj_b(x)` [R, 48]; `a = in_proj_a(x)` [R, 48]. All BF16
   (448–454).
2. **Short convolution:**
   - depthwise over all 10240 channels, kernel 4, no bias, causal (374–381);
   - prefill: `silu(conv1d(qkv)[:, :, :R])` (474); decode:
     `torch_causal_conv1d_update` (210–225): concatenate the 3-sample state,
     convolve, `silu`, cast;
   - the conv state is the **last 3 pre-convolution `qkv` rows**, BF16, left
     zero-padded at sequence start (463, 219);
   - the output is BF16 and `silu` rounds once more.
3. Split `[2048 | 2048 | 6144]` into q, k, v (477–485). q and k are 16 heads
   × 128, v is 48 × 128.
4. `beta = sigmoid(b)` **in BF16** (491).
   `g = −exp(A_log.float()) · softplus(a.float() + dt_bias)` **in FP32**
   (493); `dt_bias` BF16 is promoted.
5. q and k are `repeat_interleave`d ×3 (495–496): value head `h` reads key
   head `⌊h/3⌋`.
6. `l2norm` (228–231: `x · rsqrt(Σx² + 1e-6)`) on q and k is applied **before**
   the FP32 cast (318–323 recurrent, 245–250 chunked), so it runs **in
   BF16**. q is then scaled by `1/√128` (327, 262).
7. **Recurrence** (`torch_recurrent_gated_delta_rule`, 314–353, FP32 state
   `[48, 128k, 128v]`):
   ```
   S ← S · exp(g_t)                    (344)
   m = Σ_k S[k,:] · k_t[k]             (345)
   δ = (v_t − m) · β_t                 (346)
   S ← S + k_t ⊗ δ                     (347)
   o_t = Σ_k S[k,:] · q_t[k]           (348)
   ```
   **Prefill uses `torch_chunk_gated_delta_rule` (234–311, chunk 64)**, which
   is algebraically equal but rounds differently. The two reference paths
   therefore **do not agree bit for bit with each other**. The output is
   cast to BF16 (352).
8. **Gated norm** `Qwen3_5RMSNormGated` (175–190), per value head over 128:
   - `n = x·rsqrt(mean x² + eps)` in FP32, **cast to BF16**;
   - `bf16(w · n)` (187, BF16 multiply, **plain `w`, not `1+w`**);
   - `· silu(z.float())` in FP32 (188), then cast.
9. `out_proj` [5120, 6144].

The model ends with the final `norm` and an **untied** `lm_head` (1688–1760).
The embedding is unscaled.

### 1.3 Strata

- **No Qwen / Gated DeltaNet in Strata.** `src/models/` has deepseek, gemma4,
  glm52, glm53, inkling, kimi_k3 and laguna.
- **KDA (Kimi/GLM-5.3) is the closest relative.** Same shape: short conv,
  l2norm, scale, delta rule and gated output norm. Differences:
  - KDA's decay is **per key channel**, `exp(−5·sigmoid(exp(A_log)·(f+dt_bias)))`
    (`kernels/cuda/detail/backend_hybrid_recurrence.inc.cuh:156-162`);
    Qwen's is **one scalar per head**, `−exp(A_log)·softplus(a+dt_bias)`;
  - KDA gates the output with `sigmoid`; Qwen uses `silu(z)`;
  - KDA's output norm eps is 1e-5 (188).
- **Exactness choice** (`backend_hybrid_recurrence.inc.cuh:80-199`):
  - a **token-sequential** recurrence on device;
  - one block per head and one thread per value lane holding a
    `head_dim`-long state row (state index `(head·d + lane)·d + index`, 150);
  - every op explicitly rounded (`__fmul_rn`/`__fadd_rn`, 33–39);
  - a **custom `expf` reproducing glibc's table** (11–78), so the device
    matches the host reference exactly without a host round trip;
  - FP32 state;
  - BF16 rounding at the declared boundaries (`glm53_bf16`, 3, 147, 186, 197).
- **Model-level gate** (`docs/kimi-k3-runtime.md:99-112`):
  - chunkwise ≡ token recurrence checked on real tensors (gate 3);
  - against `modeling_kimi_linear.py`, relative L2 ≤ 2.0e-2 and cosine ≥ 0.999,
    stated before the run; measured 0.0061 (prefill), 0.0050 (state), 0.0055
    (decode).
  - Its gate 5 found a **BF16 reference unsound**: a BF16-versus-F32
    reference control disagreed more than the runtime did (113–128).
- **State:** Kimi refuses a non-current position because "the KDA half is
  recurrent and cannot be rewound" (`docs/kimi-k3-runtime.md:170-171`).
  GLM-5.3 keeps independent recurrent states disjoint per request and prices
  them in admission (`docs/models/glm53.md:374-382`).
- **Reuse candidates:**
  - the explicit-rounding style and `glm53_kda_expf`/`glm53_kda_sigmoid`
    (40–78) for the delta-rule and conv kernels;
  - the conv kernel structure (80–107), with the per-channel history shift;
  - the recurrence kernel's thread layout (109–199), changing the decay to
    Qwen's per-head scalar and the output gate to `silu(z)` with Qwen's
    `bf16(w·bf16(n))` boundary;
  - `glm53_swiglu_kernel` (369) for the device SwiGLU gap.

### 1.4 Moxie today

- **`Op::RecurrentUpdate` and `Op::ShortConv` exist only as names.**
  `moxie-graph/src/lib.rs:83-84`; `touches_state` at 120–128.
  **`OpParams` has no variant for either** (`moxie-graph/src/graph.rs:243-494`),
  so no graph can contain them. The interpreter, planner and dense executor
  have no case for them.
- **The oracles are toys.** `moxie-oracles/src/recurrent.rs:19-30` is
  `h ← decay·h + gain·x`, and `ShortConv` (96–130) is a scalar tap sum. The
  module doc says it explicitly does **not** pin any real model's recurrence
  (14–15). **The M6 ledger's line "against the existing host oracles in
  `moxie-oracles/src/recurrent.rs`" (handover, section "M6 exit
  checkpoints") is therefore wrong.** New oracles are needed (D7).
- **Recurrent state:** `moxie-state/src/accumulator.rs` is a **host**
  `BoundedStateStore<K>` with snapshot/replay/rollback over opaque bytes
  (200–520), with markers for recurrent and convolution kinds (30–45). There
  is **no device recurrent state**; `DeviceKvSequence` (`device.rs`) is KV
  only. Document 04, line 37: "Recurrent state cannot generally recover an
  earlier state by decrementing a position counter. Retain bounded prefix
  snapshots or recompute…". Line 63: a partition may not split an
  inseparable recurrent operation.
- **The dense device path handles** Embedding, Linear, RmsNorm (and grouped),
  Rope (HalfSplit only, `dense.rs:1687`), GeGlu, Attention, Residual,
  VocabProjection and the routed ops (`dense.rs` OpParams census;
  `SemanticKernelOp` at `moxie-types/src/capability.rs:48-90`).
  **Missing for Qwen:**
  - SwiGLU (a kernel only);
  - the `(1+w)` RMSNorm;
  - the within-rotary RoPE pairing;
  - the sigmoid output gate;
  - ShortConv;
  - the gated delta rule;
  - the gated norm;
  - the per-head query/gate row split.
- **Model pattern:**
  - `moxie-models/src/gemma4.rs`: `TextConfig`, `Gemma4Text::{reduced,full}`
    and `compose_with_weight_precisions` (about 420–560);
  - `text_config_from_declared` and `gemma4_source_tensor` (task 0104,
    commit bb2f7f9);
  - `moxie-cli/src/gemma.rs::from_checkpoint`;
  - `moxie_format::checkpoint_config::declared_text_fields`, which falls back
    to the root object when there is no `text_config`, as here.
- **Paged attention** serves head_dim 256 (qualified; task 0105 raises it to
  512). GQA 24/4 is a divisor, so it is admitted.

### 1.5 GPU memory and placement

- **VRAM:** 5060 Ti 16,311 MiB; 3090s 24,576 MiB each; total 65,463 MiB
  (`nvidia-smi`).
- **TP2 on the 3090 pair is impossible.** Half the text weights is
  26,895,998,464 B = 25,650 MiB > 24,576 MiB, before KV, NCCL reserve (144 MiB
  per rank), recurrent state or workspaces.
- **PP over all three GPUs fits.** Example, by layer bytes:
  - 5060 Ti: embedding + 16 layers ≈ 2.54 + 12.2 GB ≈ 14.1 GiB;
  - each 3090: 24 layers ≈ 18.3 GB, plus `lm_head` (2.54 GB) on the last
    ≈ 19.4 GiB.

  That leaves room for KV (2 GiB at 32K), recurrent state (151 MB, doubled
  under D4) and workspaces.
- **PP cuts between layers,** so no recurrent op is partitioned (document 04,
  line 63).
- **This depends on:**
  - **persistent PP** (the M6.1 line after 0102; today's PP re-lowers each
    step and uploads weights from host bindings every step,
    `docs/evidence/multi-gpu-lifecycle-before-0102.md`, section "Pipeline";
    at 53.8 GB per step that is not a benchmarkable path);
  - **the Gemma-exit task's loader** (weights from source shards into the
    residency authority, slice 7 route item 5).
- **Alternative:** TP2 over the 3090s plus a solo 5060 Ti stage (M5's
  combined plan). That needs TP partition rules for every new op. Not
  recommended.

---

## Part 2. Owner-level decisions (flagged, not decided)

**D1 — the reference version (blocking).**
- **What:** the checkpoint was exported by transformers 5.17.0 and declares
  `output_gate_type: "swish"`. The base Qwen3.8-27B (v1 entry 7,
  `/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4/config.json`, exporter
  5.8.0.dev0) declares the same. The installed 5.5.3 **never reads** the
  field and applies `sigmoid` (`modeling_qwen3_5.py:689`).
- **Precedent:** this repository already refused a mismatched `transformers`
  copy as a pinned exporter (Laguna yarn; engineering log, "Laguna's attention
  tower is not composable").
- **Options:**
  - (a) authorize a read-only fetch of the pinned 5.17.0
    `models/qwen3_5/modeling_qwen3_5.py` and `configuration_qwen3_5.py` (an
    operational expansion; read, not executed, as for Laguna);
  - (b) accept 5.5.3's `sigmoid` on an empirical discriminator. Run the host
    reference both ways on a fixed prompt, and the owner rules on which one
    produces the model's text;
  - (c) a different checkpoint.
- **Recommendation:** (a), with (b) as the fallback if the file cannot be
  fetched.

**D2 — new shared semantic operations (document 02).** Recommended set, each
with an FP64 oracle and a shape/precision/state contract:
- **`ShortConv`** params `{channels, kernel, activation: Silu}`. The state is
  the `kernel−1` previous inputs.
- **`GatedDeltaRule`** (the `RecurrentUpdate` op) params `{key_heads,
  value_heads, key_dim, value_dim, qk_l2norm_eps, decay: ScalarPerHead}`.
  Inputs `q,k,v,a,b,A_log,dt_bias`, FP32 state.
- **`GatedRmsNorm`** params `{group, eps, gate: Silu}`, for Qwen's
  `bf16(w·bf16(n))·silu(z)`.
- **`SigmoidGate`** (elementwise `x·σ(g)`).
- **`RmsNorm`** gains `gain_offset: f32` (0 or 1). The alternative is a
  derived `(1+w)` weight at load, which changes the stored tensor's values.
- **`RopeLayout::HalfSplitRotary`** (pairs `(j, j+rotary/2)` within the rotary
  lanes).
- The q/gate interleave handled as a **load-time row gather** into two
  weights (`q_proj`, `attn_gate_proj`) rather than a graph Split op. This is
  value-exact, and ADR 0038 puts layout interpretation in the importer.

Each is new vocabulary in document 02's op list, so the owner rules on the
set.

**D3 — numerical gates** (declared before any run, per AGENTS).
Recommendation:
- **Per device op versus the host oracle:** **bitwise** for ShortConv,
  GatedRmsNorm, SigmoidGate and RoPE, where the host reference is
  explicitly rounded (Strata did exactly this for KDA,
  `backend_hybrid_recurrence.inc.cuh:11-78`). For the delta rule: bitwise
  against an explicitly rounded FP32 host reference in the same token order,
  plus an FP64 oracle bound.
- **Model level versus transformers:** Strata's Kimi gate, stated before the
  run: relative L2 ≤ 2e-2 and cosine ≥ 0.999 per layer output and for the
  final recurrent state, and greedy agreement over N tokens (owner sets N).
- **The chunked prefill reference ≠ the recurrent one** (1.2.7), so
  "byte-identical to transformers" is not available.

**D4 — the recurrent-state transaction mechanism** (document 04, line 37).
Recommendation: **double-buffered committed/tentative slots** per layer (a
depth-1 snapshot). A step reads the committed slot and writes the tentative
one; commit flips, abort discards. Cost: 2 × 150,994,944 B + 2 × 2,949,120 B
per sequence. Alternatives are a per-step copy (151 MB per token) or replay
from a bounded snapshot (M9's spec-decode need, not M6's). This is the same
mechanism for prefill, continuation and abort, as AGENTS' "one
state-transaction mechanism" requires.

**D5 — placement and dependency order.**
- PP over three GPUs requires persistent PP (not built) and the Gemma-exit
  loader.
- The owner rules the order: Qwen after both, or earlier stages (host, ops,
  kernels) proceeding in parallel with the dependencies.
- Running on today's per-step PP would upload 53.8 GB of weights per token.

**D6 — catalog and license: record only.** Hemmingway-1 is not in the v1
catalog (as with gemma-4-26B) and is cc-by-nc-4.0. The owner named it for the
M6 exit (handover "M6 exit checkpoints").

**D7 — ledger correction.** "Against the existing host oracles in
`recurrent.rs`" is false (1.4). The oracles are new work.

**D8 — reference dtype.** Run the transformers reference in FP32. Strata's
Kimi control showed a BF16 reference is unsound for layer comparison
(`docs/kimi-k3-runtime.md:113-128`). FP32 of 53.8 GB is about 108 GB of host
RAM, within 251 GB. The BF16 run is for information only.

---

## Part 3. Contract draft (paste into `docs/tasks/01NN-m6-exit-qwen-hybrid.md`)

> Written as one task (pace ruling), in **five gated stages**. Each stage
> ends at a gate the next stage depends on, so a split, if the coordinator
> records one, happens at a stage boundary without rewriting. **Stages A–D
> need D1–D4. Stage E also needs D5's dependencies.**

### Identity and authority
- **M6 exit**, the "Exit, Qwen" line (handover, "Pace ruling").
- Checkpoint `/fast/models/Altworld/Hemmingway-1`, revision
  `b987f1800da287b3016862948743a253fbf80553`, read in place (ADR 0038). The
  MTP head is not run (M9).
- Team: builder Claude Sonnet `builder`; reviewer Claude Opus `reviewer`;
  coordinator `coordinator`. The design is the coordinator's plus D1–D5's
  rulings; on any conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Stage explicit paths
  only. `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Naming rule: identifiers say what they
  do (`gated_delta_rule`, not a task number).

### Facts (cite in the task; do not re-derive)
Part 1.1 and 1.2 verbatim, with their file:line citations.

### Allowed files
- `crates/moxie-graph/src/{graph.rs,lib.rs}` (D2's params and contracts);
- `crates/moxie-oracles/src/{recurrent.rs,norm.rs,rope.rs,activation.rs,lib.rs}`;
- `crates/moxie-interp/src/lib.rs`;
- `crates/moxie-types/src/capability.rs` (the `SemanticKernelOp` variants);
- `crates/moxie-plan/src/selected.rs` (lowering the new ops; state
  requirements);
- `crates/moxie-kernels/{cuda/recurrent_ops.cu (new),cuda/dense_ops.cu,cuda/dense_graph.cu,src/lib.rs,build.rs}`;
- `crates/moxie-state/src/{device.rs,lib.rs}` (device recurrent-state
  authority, D4);
- `crates/moxie-executor/src/{dense.rs,chain.rs}` and one new
  `crates/moxie-executor/src/recurrent_run.rs`;
- `crates/moxie-models/src/{qwen3_5.rs (new),lib.rs}`;
- `crates/moxie-cli/src/{qwen.rs (new),lib.rs}`;
- tests:
  - `crates/moxie-oracles` unit tests;
  - `crates/moxie-interp/tests/reference_graphs.rs`;
  - `crates/moxie-executor/tests/qwen_hybrid_device.rs` (new);
  - `crates/moxie-cli/tests/qwen_checkpoint.rs` (new);
  - `xtask/src/gpu.rs` (new cases);
- `tools/reference/qwen3_5_reference.py` (new; the transformers reference
  runner, the only Python);
- this task's Result.

Stage E adds the files of the persistent-PP and loader tasks it builds on,
through those tasks, not here.

### Numbered changes

**Stage A — config, graph and names (host, no ops yet).**
1. `moxie-models/src/qwen3_5.rs`:
   - `pub struct HybridConfig { hidden, layers, intermediate, vocab, rms_eps,
     full_every: u32, attn: FullAttnGeometry{heads:24,kv_heads:4,head_dim:256,rotary_dim:64,rope_base:1e7},
     linear: LinearAttnGeometry{key_heads:16,value_heads:48,key_dim:128,value_dim:128,conv_kernel:4} }`
     (fields named as given; values come from the checkpoint);
   - `pub fn hybrid_config_from_declared(&BTreeMap<String, DeclaredValue>) -> Result<HybridConfig>`,
     mapping the keys in 1.1;
   - derive `full_every` from `layer_types` exactly as
     `gemma4::text_config_from_declared` derives `global_stride`;
   - **refuse**, naming key and value, unless:
     - `rope_parameters.rope_type == "default"`;
     - `partial_rotary_factor == 0.25`;
     - `mrope_interleaved == true`;
     - `mrope_section` sums to `rotary_dim/2` (= 32);
     - `attn_output_gate == true`;
     - `output_gate_type ==` D1's ruled value;
     - `hidden_act == "silu"`;
     - `attention_bias == false`;
     - `tie_word_embeddings == false`;
     - `mamba_ssm_dtype == "float32"`.
2. `pub fn qwen3_5_source_tensor(role: &TensorRole) -> Option<String>`, the
   names in 1.1. `attn_gate_proj` and `q_proj` both map to
   `self_attn.q_proj.weight`, with a `RowGather::PerHeadHalves { heads: 24,
   head_dim: 256, half: First|Second }` loader instruction returned alongside
   (D2).
3. `moxie-cli/src/qwen.rs::from_checkpoint(dir) -> Result<HybridCheckpointGraph>`:
   the task-0104 pattern, reading the revision from the Hub metadata and
   **refusing if it is absent** (0104 review M3).

**Stage B — operations, oracles, interpreter (host).**
4. `OpParams::{ShortConv, GatedDeltaRule, GatedRmsNorm, SigmoidGate}` and
   `RmsNorm.gain_offset`, `RopeLayout::HalfSplitRotary`, per D2's ruling, each
   with `Op` contract, shape validation and `touches_state` (ShortConv and
   GatedDeltaRule: true).
5. Oracles in `moxie-oracles/src/recurrent.rs`, replacing nothing (the toys
   stay, since their tests pin the replay rule). Add:
   - `short_conv_silu(history, x, taps) -> (y, new_history)`;
   - `gated_delta_step(state, q, k, v, a, b, a_log, dt_bias) -> o` (FP64,
     the equations in 1.2.7);
   - `gated_delta_chunked(…, chunk)`, a transcription of
     `torch_chunk_gated_delta_rule` 234–311, for the equivalence test only;
   - `gated_rms_norm` in `norm.rs` with the 1.2.8 boundaries.
6. The interpreter evaluates all new params, with state threaded through a
   host `BoundedStateStore` per layer and kind (existing, `accumulator.rs`).

**Stage C — device kernels.**
7. `recurrent_ops.cu`, added to `dense_graph.cu`:
   - `moxie_short_conv_silu_v1`;
   - `moxie_gated_delta_rule_v1`: one block per value head, one thread per
     value lane, a sequential token loop, explicit `__fmul_rn`/`__fadd_rn`,
     and the Strata `expf` table reproduction for `exp(g)`, `softplus` and
     sigmoid;
   - `moxie_gated_rms_norm_v1`;
   - `moxie_sigmoid_gate_v1`;
   - `moxie_dense_swiglu_v1` (after `glm53_swiglu_kernel`);
   - the RoPE kernel's `HalfSplitRotary` branch, which leaves the existing
     branch's bytes unchanged.

   Catalogue descriptors are added in `moxie-kernels/src/lib.rs`, one per SM.
8. `xtask/src/gpu.rs` cases, one per kernel. Each checks the kernel against
   the Stage B oracle on SM86 and SM120 under D3's gate, at Hemmingway's
   real per-layer shapes: 10240 channels; 48×128×128 state; 16→48 head
   repeat.

**Stage D — device state and executor.**
9. `moxie-state/src/device.rs`: `DeviceRecurrentState` (D4). Per layer and
   kind it records the committed slot `0|1`, the tentative slot while a
   transaction is open, `commit` flipping, and `abort` discarding. It holds
   no CUDA. `SequenceState` stays the authority.
10. `moxie-executor/src/recurrent_run.rs`: `RecurrentRun`, admitted once per
    linear layer. It holds two FP32 state slots and two BF16 history slots,
    charged to the ledger at admission, in the paged-attention run pattern.
    **Escape inventory:**

    | Resource | Owner | Rule |
    |---|---|---|
    | State and history slots | `RecurrentRun` | Never released by a step. Closed only after an observed drain. |
    | Tentative slot of an unfinished step | `RecurrentRun`, quarantined | A post-submission failure withholds both slots, and `close` refuses until observed. |
    | Pre-launch refusal | — | Touches no slot; the committed slot is unchanged. |
    | Commit | `DeviceRecurrentState` | Flips only after the step's completion is observed. |
    | Abort | `DeviceRecurrentState` | Discards the tentative slot; bytes stay allocated for reuse. |
    | Drop | `RecurrentRun` | Quarantines (no free while completion is unknown). |

11. `dense.rs` dispatch for every new param. A step's graph reads the
    committed slot and writes the tentative slot, whose addresses come from
    `DeviceRecurrentState`.

**Stage E — checkpoint run (after D5's dependencies).**
12. Load Hemmingway-1 through the Gemma-exit loader, with the row gather of
    change 2. Run a PP plan over the three GPUs:
    - stage 0 on the 5060 Ti: embedding + layers 0–15;
    - stage 1 on 3090 #1: layers 16–39;
    - stage 2 on 3090 #2: layers 40–63 + final norm + `lm_head`.

    Identify GPUs by UUID. This is fixed-plan mode.
13. `tools/reference/qwen3_5_reference.py` (transformers 5.5.3 per D1, FP32 per
    D8, text only). It writes per-layer hidden states, final recurrent
    states and logits for a fixed token-id list to
    `results/qwen-reference/` (ignored scratch).

### Tests (exact assertions)
- **T1** (Stage A, `qwen_checkpoint.rs`, `#[ignore]` needing the checkpoint):
  - the config equals the literal geometry of 1.1;
  - every role maps to exactly one index tensor, and every
    `model.layers.*`/`model.embed_tokens`/`model.norm`/`lm_head` tensor is
    bound exactly once (excluding `mtp.*`);
  - each role's graph shape equals its source shape (for the gathered
    `q_proj`/`attn_gate_proj`: `[6144, 5120]` each, from the `[12288, 5120]`
    source);
  - the refusal unit test covers a wrong `output_gate_type`, a wrong
    `partial_rotary_factor` and an irregular `layer_types`.
- **T2** (Stage B, oracle unit tests):
  - `gated_delta_chunked` ≡ the stepwise form in FP64, max relative error
    ≤ 1e-12 over 130 tokens (crossing two chunk boundaries);
  - interleaved mRoPE with T=H=W equals `HalfSplitRotary` bit for bit on
    positions 0..4096;
  - `short_conv_silu` over a stream equals decode-from-saved-history at every
    split point;
  - `gated_rms_norm` reproduces 1.2.8's two BF16 boundaries. The test asserts
    it differs from single-rounding on a constructed case, which shows the
    boundary is load-bearing.
- **T3** (Stage B, `reference_graphs.rs`): a reduced hybrid graph (2 linear +
  1 full layer, hidden 64). Prefill of 7 tokens then 3 decodes equals prefill
  of 10, byte for byte. An aborted decode leaves the next decode byte-identical
  to one never aborted.
- **T4** (Stage C, `test-gpu`): each kernel meets D3's gate on SM86 and SM120.
  The delta-rule case runs 64 tokens from a nonzero saved state.
- **T5** (Stage D, `qwen_hybrid_device.rs`):
  - the reduced hybrid graph on one GPU is byte-identical to the interpreter
    across prefill, 3 decodes, and an injected abort;
  - after 24 steps, ledger charges are constant (0088's rule).
- **T6** (Stage E): against the Stage E reference, per layer and final state,
  D3's model gate; greedy agreement over D3's N tokens; then the exit benchmark
  rows through task 0090's harness.
- **Coverage mutant** (Stage C): replace the delta-rule kernel's `exp(g)`
  with `exp(g/2)`. Only T4's delta-rule case is run, and it must fail.

### Gates
- **Host:** fmt; workspace clippy; xtask CUDA clippy; `cargo test --workspace
  --locked`; arch-check; spec-check.
- **Stage C onward:** `cargo xtask-cuda test-gpu` (kernel sources change).
- **Stage D:** the `qwen_hybrid_device` suite.
- **Stage E:** T6, and the benchmark rows.
- No timing before Stage E.

### Stop conditions
- D1 unresolved, or the ruled reference disagrees with the checkpoint's
  declared fields;
- an existing kernel's bytes change (RoPE HalfSplit, the dense package);
- a new op cannot be expressed without a Qwen-named execution path;
- the delta-rule kernel cannot meet D3 without host round trips;
- a placement stage exceeds a GPU's admitted capacity;
- a file outside the list is needed.

---

## Part 4. Reviewer's assessment of "one task"

This cannot be built safely as one Sonnet task. The reason, for the record
the pace ruling requires:
- it adds **five new shared operations** (D2);
- a **new device state kind** with its own transaction mechanism (D4);
- **six kernels**;
- and a checkpoint run that **depends on two unbuilt pieces** (persistent PP;
  the Gemma-exit loader).

Every one of those is the kind of change M4–M6 tasks spent several review
rounds on (ops: 0053, 0065–0067; state: 0046–0048; kernels: 0037, 0105).
Stage E cannot start until its dependencies land.

The contract is therefore written with stage gates, so the coordinator can
record a split at A+B / C+D / E without rewriting. Stages A–D can proceed
now, once D1–D4 are ruled.
