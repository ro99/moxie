# Task 0104 — full-size Gemma 4 text graphs from the downloaded checkpoints

Status: **accepted** (coordinator, 2026-09-26). Implementation `bb2f7f9`
(Claude Sonnet `builder`, taking over luna's uncommitted work). Review R1
by Claude Opus `reviewer` (1 high, 4 medium, 4 low;
[record](../evidence/task-0104-review-round-1.md)). Repair `da08cad`, and
the R2 delta review was clean. Both full-size graphs compose from the
downloaded checkpoints, and every role's shape, precision and
`layer_scalar` is checked against the files.

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
  - `crates/moxie-types/src/` (one new module holding `DeclaredValue`, plus
    its `lib.rs` line). This was amended after luna's DECISION: the type
    lives in `moxie-types`, which both `moxie-format` and `moxie-models`
    already depend on, so no dependency boundary or arch-check allowlist
    changes;
  - `crates/moxie-format/src/checkpoint_config.rs`;
  - `crates/moxie-models/src/gemma4.rs`;
  - `crates/moxie-cli/src/gemma.rs` and `crates/moxie-cli/Cargo.toml`
    (adding `moxie-format` and `moxie-storage`);
  - `xtask/src/archcheck.rs`, only to add `moxie-format` and
    `moxie-storage` to `moxie-cli`'s workspace allowlist, with this comment:
    "the composition root reads a downloaded checkpoint's config and
    metadata tensors to compose its model (ADR 0038)". This was authorized
    by the coordinator, answering luna's DECISION: both are host-only, with
    no device code and no third-party additions, and document 02 makes
    `moxie-cli` the composition root;
  - `crates/moxie-cli/tests/` (one new test file);
  - this task's Result.
- **Non-goals:**
  - loading weights (route item 5);
  - running anything;
  - Qwen;
  - vision and audio;
  - changing `reduced`.

## Numbered changes

1. **Declared fields (`moxie-format`).** Define `pub enum DeclaredValue {
   Bool, Int(i64), Float(f64), Str(String), List(Vec<DeclaredValue>), Null }`
   **in `moxie-types`**, as plain data with no serde. In `moxie-format`, add
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

Status: **implemented** (builder, Claude Sonnet, 2026-09-26), taking over an
in-progress build luna's session left uncommitted after its weekly Codex
quota ran out. Luna's partial work (changes 1, 2 and most of 3/4) reviewed
against the contract and completed; two defects fixed before continuing:
`Gemma4Text::full` did not compile (`Vec<WeightPrecision>` vs a fixed-size
array literal), and the format-crate imports it had staged
(`compressed_tensors::{Granularity, ZeroPointSource}`, the `checkpoint_config`
declaration types) were unused once `checkpoint_config::parse`'s own
validation was relied on instead of re-checking granularity here.

1. `DeclaredValue` in `moxie-types::declared_value`;
   `moxie_format::checkpoint_config::declared_text_fields` (luna).
2. `moxie_models::gemma4::text_config_from_declared`, with the
   `global_stride`/`global_partial_rotary`/fixed-fact refusals and the MoE
   branch (luna).
3. `Gemma4Text::full` (luna; the `Vec` fix above was this builder's).
4. `gemma4_source_tensor` (luna).
5. `moxie_cli::gemma::from_checkpoint` and `CheckpointGraph` (this builder,
   on luna's struct skeleton): reads `config.json` and the safetensors
   index, reads every `layer_scalar` through `Shard`, builds the config and
   the full graph, and maps the checkpoint's declared quantization
   (`bits: 4 | 8`, via `checkpoint_config::parse`'s own `ignored` list) onto
   each linear's precision -- everything else stays the graph's BF16
   default. The revision is read from the local Hub cache metadata beside
   `config.json` (`.cache/huggingface/download/config.json.metadata`),
   falling back to the directory name when that cache is absent.
6. `crates/moxie-cli/tests/gemma_checkpoints.rs` (this builder): one
   `#[ignore]`d test per checkpoint, each independently re-reading the
   index and safetensors headers (not reusing `from_checkpoint`'s own
   reading) to check (a) the derived config against `ARTIFACT`/
   `ARTIFACT_A4B`, (b) every bound role resolves to an index tensor and
   every non-excluded `model.language_model.*` index tensor is bound
   exactly once, (c) each bound role's graph shape equals the source's
   logical shape (the safetensors header, or `weight_shape`'s declared
   value for a packed INT8 tensor), and (d) the derived config composes
   again at `SymbolId(1)` and `SymbolId(8)`. The refusal unit test change 6
   asks for (wrong `rope_type`, irregular `layer_types`, wrong scalar
   count) is luna's existing
   `gemma4::tests::declared_config_maps_and_refuses_unsupported_values`;
   not duplicated.
7. Coverage check performed and reverted: mapped `ffn_out_norm`'s
   `post_feedforward_layernorm` to `pre_feedforward_layernorm` in
   `gemma4_source_tensor`; `dense_checkpoint_composes_a_full_size_graph`
   failed (`... is bound by more than one role`), confirming test (b) is
   load-bearing. Reverted; both checkpoint tests pass again.

Gates, run on `/home/rodrigo/Developer/moxie` against both real checkpoints
under `/fast/models` (host only, no GPU):
- `cargo fmt --check`: clean.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: clean.
- `cargo clippy -p xtask --features cuda --all-targets --locked -- -D warnings`: clean.
- `cargo test --workspace --locked`: passed (default run; the two checkpoint
  tests skip as `ignored`).
- `cargo xtask arch-check`: passed (79 rejected fixtures, 21 accepted, 13
  rules exercised) -- confirms the `moxie-cli` allowlist addition is the
  only workspace-dependency change.
- `cargo xtask spec-check`: passed (10 documents unchanged).
- `cargo test -p moxie-cli --test gemma_checkpoints -- --ignored`: 2 passed,
  against `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` (INT8, dense) and
  `/fast/models/google/gemma-4-26B-A4B-it` (BF16, MoE).

Not claimed: loading any weight, running any token, Qwen, vision/audio, GPU
execution, or any performance number -- all named non-goals or out of this
task's scope. `CheckpointGraph` also does not carry the checkpoint's
declared quantization granularity or zero-point convention ("INT8 affine
group 32 symmetric") -- only the precision. The loader (route item 5)
re-parses `config.json` for those; this task's graph only needs to know
which precision each linear composes at.

### Round 2 (builder, Claude Sonnet, 2026-09-26)

Reviewer's round-1 findings (1 high, 4 medium, 4 low;
[record](../evidence/task-0104-review-round-1.md)), all applied as written:

- **H1** — `gemma_checkpoints.rs`'s per-role loop now derives the expected
  precision from the index itself (`.weight_packed` → INT8, otherwise the
  header's own BF16 dtype, asserted) and checks it against
  `spec.role`. Coverage check performed and reverted: deleting
  `from_checkpoint`'s `precisions.insert(...)` made
  `dense_checkpoint_composes_a_full_size_graph` fail on `q_proj.0`'s
  precision (`Bf16` vs expected `Int8`), as required.
- **M1** — test (d) no longer recomposes under a different `SymbolId`
  (which only renames the same symbolic graph, since `SymbolId` is the
  rows symbol's identity, not a count). It now binds
  `graph.composition.graph.rows_symbol()` to 1 and to 8 in a `SymbolTable`
  and evaluates every value's shape dims with `Dim::eval`, on
  `graph.composition.graph` itself -- the checkpoint's actual INT8
  composition, not a re-derived BF16-only one.
- **M2** — added an independent `layer_scalar` check: each layer's tensor
  is re-read through `Shard` in the test and compared bit-for-bit
  (`to_bits()`) against `graph.config.layer_scalars[layer]`, plus a length
  check against `artifact.layers`.
- **M3** — `checkpoint_revision` now requires the metadata file's first
  line to be exactly 40 lowercase hex characters, and returns
  `Result<String>` instead of falling back to the directory name. Both
  real checkpoints' `.cache/huggingface/download/config.json.metadata`
  satisfy this (`34ca187d…` for the 31B, `4d7ae498…` for the A4B).
- **M4** — `from_checkpoint` now refuses up front (before reading
  `text_config`) when the checkpoint declares any AutoRound 16-bit
  passthrough override, naming the pattern, instead of silently composing
  a passthrough module as INT8.
- **L1** — `flatten_declared`'s array arm removed; the catch-all arm already
  calls `declared_value`, which recurses into an array's own items.
- **L2** — the `layer_types` stride search is now a single `.find`; two
  strides could never both match (their smallest full-attention layers
  disagree), so the old second-iterator uniqueness check was dead.
- **L3** — recorded above, in "Not claimed".
- **L4** — the Status line's attribution corrected to Claude Sonnet.

Reruns: `cargo fmt --check`, workspace clippy, `xtask` clippy with the
`cuda` feature, `cargo xtask arch-check`, `cargo xtask spec-check`,
`cargo test --workspace --locked`, and
`cargo test -p moxie-cli --test gemma_checkpoints -- --ignored` (2 passed,
against both real checkpoints) -- all clean. No GPU gate applies; this
task touches no kernel or device path.
