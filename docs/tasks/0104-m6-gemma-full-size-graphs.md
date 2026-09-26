# Task 0104 — full-size Gemma 4 text graphs from the downloaded checkpoints

Status: **proposed** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`. Host-only. Queued after task 0100.

## Identity and authority

- Task0104, **M6 exit** (checkpoint benchmarks), route item 2. The owner's
  exit checkpoints (2026-09-25) include
  `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` (dense, INT8) and
  `/fast/models/google/gemma-4-26B-A4B-it` (MoE, BF16). Under
  [ADR 0038](../decisions/adr/0038-run-source-checkpoints-directly.md), both
  are read as downloaded.
- `moxie-models` builds only `Gemma4Text::reduced` today. `TextConfig`
  already holds per-layer key/value geometry and the MoE block. What is
  missing:
  - reading the checkpoint's declared numbers;
  - the per-layer `layer_scalar` tensors, which `TextConfig::layer_scalars`
    documents as "a genuine ordering constraint on the importer";
  - INT8 affine precision for linears;
  - the role → source-tensor naming.
- Architecture rules: only `moxie-format` may use `serde_json`
  (`xtask/src/archcheck.rs` allowlist). Model crates do no I/O. The
  composition root (`moxie-cli`) reads files.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**; on a conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** Naming rule applies. No GPU.

## Facts established before writing (coordinator, 2026-09-25)

- Both `config.json` files hold a `text_config`:
  - **Shared keys:** `hidden_size`, `num_hidden_layers`,
    `num_attention_heads`, `num_key_value_heads` (sliding),
    `num_global_key_value_heads`, `head_dim` (sliding), `global_head_dim`,
    `intermediate_size`, `vocab_size`, `sliding_window`, `rms_norm_eps`,
    `final_logit_softcapping`, `max_position_embeddings` and `layer_types`
    (`sliding_attention` or `full_attention` per layer).
  - **Rotary:** `rope_parameters.full_attention` has `rope_theta`
    1,000,000, `partial_rotary_factor` 0.25 and `rope_type` `proportional`.
    `rope_parameters.sliding_attention` has `rope_theta` 10,000 and
    `rope_type` `default`.
  - **MoE:** `enable_moe_block`, plus `num_experts`, `top_k_experts` and
    `moe_intermediate_size` for the MoE model.
  - **Fixed facts:** `attention_k_eq_v` true, `tie_word_embeddings` true,
    `hidden_activation` `gelu_pytorch_tanh`, `num_kv_shared_layers` 0,
    `hidden_size_per_layer_input` 0, `use_double_wide_mlp` false and
    `attention_bias` false.
- 31B quantization: `quantization_config` (compressed-tensors
  `pack-quantized`, INT8, group 32, symmetric), parsed by
  `moxie_format::checkpoint_config::parse`.
- Tensor names (layer `L`, prefix `model.language_model.`):
  - `embed_tokens.weight`, `norm.weight`;
  - `layers.L.input_layernorm.weight`,
    `layers.L.self_attn.{q,k,v,o}_proj.weight` (31B:
    `.weight_packed`/`.weight_scale`/`.weight_shape`),
    `layers.L.self_attn.{q,k}_norm.weight`,
    `layers.L.post_attention_layernorm.weight`,
    `layers.L.pre_feedforward_layernorm.weight`,
    `layers.L.mlp.{gate,up,down}_proj.weight`,
    `layers.L.post_feedforward_layernorm.weight`,
    `layers.L.layer_scalar`;
  - MoE: `layers.L.router.{scale,proj.weight,per_expert_scale}`,
    `layers.L.experts.{gate_up_proj,down_proj}`,
    `layers.L.pre_feedforward_layernorm_2.weight` and
    `layers.L.post_feedforward_layernorm_{1,2}.weight`;
  - global layers (`(L+1) % 6 == 0`) have no `v_proj`.
- `Gemma4Text::reduced` and `compose_with_weight_precisions`
  (`gemma4.rs` about 420–560) already compose these roles.
  `v_norm_unit_gain` has no source tensor: it is all ones, as in the pinned
  runtime.

## Bounded deliverable

- **Outcome:** `moxie_cli::gemma::from_checkpoint(dir)` returns a composed,
  validated **full-size** graph for either checkpoint. It is built from the
  checkpoint's own declared numbers and `layer_scalar` tensors, with INT8
  affine linears for the 31B. It comes with a complete role → source-tensor
  map that covers every text tensor in the index exactly once.
- **Allowed files:**
  - `crates/moxie-format/src/checkpoint_config.rs`;
  - `crates/moxie-models/src/gemma4.rs`;
  - `crates/moxie-cli/src/gemma.rs`, `crates/moxie-cli/Cargo.toml` (only
    if a dependency on `moxie-format` or `moxie-storage` is missing and
    allowed by arch-check);
  - `crates/moxie-cli/tests/` (one new test file);
  - this task's Result.
- **Non-goals:**
  - loading weights (route item 5);
  - running anything;
  - Qwen;
  - vision and audio;
  - changing `reduced`.

## Numbered changes

1. **Declared fields (`moxie-format`).** Add `pub enum DeclaredValue { Bool,
   Int(i64), Float(f64), Str(String), List(Vec<DeclaredValue>), Null }` and
   `pub fn declared_text_fields(config_json: &str) ->
   Result<BTreeMap<String, DeclaredValue>>`:
   - it reads `text_config` if present, else the root object;
   - it flattens nested objects with dotted keys
     (`rope_parameters.full_attention.rope_theta`);
   - it applies no architecture knowledge, and refuses non-object input.
2. **Config from declared fields (`moxie-models`).** Add `pub fn
   text_config_from_declared(fields: &BTreeMap<String, DeclaredValue>,
   layer_scalars: Vec<f32>) -> Result<TextConfig>`:
   - **Mapping:** the keys above map to `TextConfig`, with
     `embedding_scale(hidden)`.
   - **`global_stride`** is derived from `layer_types`. Refuse unless
     `full_attention` appears exactly at layers with `(l+1) % s == 0` for
     one `s`.
   - **`global_partial_rotary`** is `Fraction::QUARTER` when
     `partial_rotary_factor` is 0.25. Refuse any other value.
   - **Fixed facts:** refuse unless `attention_k_eq_v`,
     `tie_word_embeddings`, the activation, `num_kv_shared_layers`,
     `hidden_size_per_layer_input`, `use_double_wide_mlp`,
     `attention_bias` and both `rope_type`s are exactly as listed. The
     refusal names the key and the value.
   - **MoE:** `moe` is `Some` when `enable_moe_block`, with `experts`,
     `top_k`, `moe_intermediate` and `router_input_scale(hidden)`.
   - **Scalars:** `layer_scalars.len()` must equal `layers`.
3. **Full graph (`moxie-models`).** Add `pub fn Gemma4Text::full(config,
   revision) -> Result<Self>`. It is `reduced`'s tensor list with family
   `gemma4-text`, and its `Reduction` reports only `text_only`. The allowed
   precisions are:
   - BF16 for norms, embedding and router tensors;
   - BF16 or INT8/INT4 affine for `q/k/v/o_proj` and `ffn_gate/up/down`.

   `reduced` is unchanged.
4. **Source names (`moxie-models`).** Add `pub fn gemma4_source_tensor(role:
   &TensorRole) -> Option<String>`, the exact names above.
   `v_norm_unit_gain` returns `None` (synthetic ones). Linears return the
   base name without a suffix; the loader picks `.weight` or
   `.weight_packed`.
5. **Composition root (`moxie-cli`).** Add `pub fn from_checkpoint(dir:
   &Path) -> Result<CheckpointGraph>`. It:
   - reads `config.json` and the index;
   - reads every `layer_scalar` (a BF16 scalar) through the existing
     safetensors reader, with header and shape checked;
   - builds the config (change 2) and `Gemma4Text::full`;
   - takes precisions from `checkpoint_config::parse`: the 31B's linears
     not in `ignore` become INT8 affine group 32 symmetric, and everything
     else BF16;
   - composes.

   `CheckpointGraph` holds the composition, the role → source-name map, and
   the revision.
6. **Tests (`crates/moxie-cli/tests/gemma_checkpoints.rs`).** Mark them
   `#[ignore = "needs the checkpoints under /fast/models"]` and run them
   explicitly. For each checkpoint:
   - (a) the derived config equals `ARTIFACT` or `ARTIFACT_A4B` on every
     shared field;
   - (b) every bound role except `v_norm_unit_gain` maps to a tensor in
     the index (with `.weight` or `.weight_packed`). Every text tensor in
     the index is bound exactly once, excluding `vision`, `audio`,
     `embed_vision`, the `.weight_scale`/`.weight_shape` companions and
     `layer_scalar`, which are consumed by change 5;
   - (c) each role's graph shape equals the source logical shape. That is
     the safetensors header for BF16, and the `weight_shape` tensor's value
     for packed INT8;
   - (d) the graph validates, and composition at `rows` = 1 and 8 succeeds.

   Add a host unit test for change 2's refusals: a wrong `rope_type`, an
   irregular `layer_types`, and a wrong scalar count.
7. **Coverage check (one mutant, reverted after; run only test (b)).** Map
   `post_feedforward_layernorm` to `pre_feedforward_layernorm`. Test (b)
   must fail.

## Acceptance

Host gates:
- fmt;
- workspace clippy;
- `cargo test --workspace --locked`;
- arch-check;
- spec-check;
- the ignored checkpoint tests run explicitly and pass.

No GPU and no timing.

## Result, filled after work
