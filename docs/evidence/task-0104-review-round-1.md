# Task 0104 review, round 1 (reviewer, read-only, /ponytail:ponytail-review)

Source: bb2f7f9. Contract: docs/tasks/0104-m6-gemma-full-size-graphs.md, with both amendments.
Findings: **1 HIGH, 4 MEDIUM, 4 LOW.**

Ran read-only: `cargo test -p moxie-cli --test gemma_checkpoints --locked -- --ignored`, 2 passed.

## What checks out

- **`declared_text_fields`** (`checkpoint_config.rs:36-139`):
  - architecture-free;
  - reads `text_config`, else the root;
  - dotted flattening;
  - refuses non-object input and a non-object `text_config`;
  - refuses a duplicate flattened key and a u64 above `i64::MAX`.
- **`text_config_from_declared`** (`gemma4.rs` diff):
  - every contract key is mapped;
  - `embedding_scale(hidden)` and `router_input_scale(hidden)`;
  - `global_stride` derived from `layer_types`, with irregular patterns refused;
  - `partial_rotary_factor` must be 0.25;
  - all seven fixed facts and both `rope_type`s are refused with key and value named (`declared_error` prints `key` and `{value:?}`);
  - the MoE branch;
  - the scalar count.
- **`Gemma4Text::full`**:
  - family `gemma4-text`;
  - `Reduction { synthetic_weights: false, text_only: true }`;
  - linears allow BF16/INT8/INT4 and everything else BF16 only;
  - `reduced` is unchanged apart from the new `reduction` field.
- **`gemma4_source_tensor`**: every role, MoE included; `v_norm_unit_gain` → `None`; linears without a suffix. Test (b) confirms this against both real indexes.
- **Precision-map key drift is loud:** `weight_label` in `moxie-cli` copies `gemma4.rs`'s private labelling rule. A drifted key still cannot pass silently, because `GraphBuilder::finish` refuses unused precision keys (`moxie-graph/src/graph.rs:1834`).
- **Dependencies:**
  - `Cargo.lock` gains only `moxie-storage` under `moxie-cli`. `moxie-format` was already a dev-dependency, so moving it to a normal dependency adds no lock line. Justified.
  - `archcheck.rs` adds only `moxie-format` and `moxie-storage` to `moxie-cli`, with the mandated comment. No other allowlist change.
- **Refusal unit test:** the host test `declared_config_maps_and_refuses_unsupported_values` covers the wrong `rope_type`, the irregular `layer_types` and the wrong scalar count.

## HIGH

**H1 — nothing checks the composed precision of any role, so change 5's
central claim is untested.**
- **Where:** `crates/moxie-cli/tests/gemma_checkpoints.rs:151-191`.
- **What:** test (c) compares shapes only. Test (b) accepts either `.weight` or
  `.weight_packed`.
- **Failure scenario:** delete `precisions.insert(...)` in `from_checkpoint`
  (`crates/moxie-cli/src/gemma.rs`, the loop body after the `bits` match). The
  31B then composes every linear as BF16, and both tests still pass. The same
  happens if `declaration.quantization` is read as `None`, or if the `ignored`
  match were inverted.
- **Fix:** inside the loop, after `resolved`, derive the expected precision
  from the index itself. A `.weight_packed` name gives
  `WeightPrecision::new(Precision::Int8)`; otherwise the header's
  `entry.dtype`, which must be BF16, gives `Precision::Bf16`. Then
  `assert_eq!(spec.role, ValueRole::Weight(expected), "{key:?} precision")`.
- **Coverage check to record:** the deletion above must now fail.

## MEDIUM

**M1 — test (d) does not test rows 1 and 8.**
- **Where:** `gemma_checkpoints.rs:207-218`.
- **What:** `SymbolId(rows)` is the *identity* of the rows symbol
  (`compose(&oracles, rows: SymbolId)` → `GraphBuilder::new(oracle, rows)`),
  not a row count. So the loop composes the same symbolic graph twice under
  two symbol names. It also re-composes a **BF16-only** `Gemma4Text::full`,
  not the checkpoint's INT8 composition, so it adds nothing to what
  `from_checkpoint` already did.
- **Fix:** delete the recompose. For `rows in [1, 8]`, bind
  `graph.composition.graph.rows_symbol()` to `rows` in a `SymbolTable` and
  evaluate every value spec's dims with `Dim::eval`, asserting `Ok`. This runs
  on `graph.composition.graph`.

**M2 — the `layer_scalar` values are never checked.**
- **Where:** `gemma.rs` (the `layer_scalars` loop); the test never looks at
  them.
- **Failure scenario:** swap `bytes[0]`/`bytes[1]`, or read layer `L`'s
  scalar from the wrong name, and every test still passes. Test (a) compares
  shared fields only.
- **Fix (test):** re-read each
  `model.language_model.layers.{L}.layer_scalar` through `Shard` in the
  test, independently of `from_checkpoint`. Assert
  `graph.config.layer_scalars[L].to_bits()` equals the BF16-widened bits,
  and that the length equals `layers`.

**M3 — `checkpoint_revision` is an unrequested feature with a silent identity fallback.**
- **Where:** `gemma.rs`, `fn checkpoint_revision`, near the end of the new
  code.
- **What:** when `.cache/huggingface/download/config.json.metadata` is absent,
  the directory name becomes the model's `revision`. That is a label that is
  not a revision, recorded in `ModelMetadata`. AGENTS requires exact
  revisions. Both checkpoints do have the file (31B `34ca187d…`, 26B
  `4d7ae498…`), so the fallback only serves inputs where it misleads.
- **Fix:** read the first line and require 40 lowercase hex characters.
  Otherwise return `Error::InvalidArtifact` naming the path. Delete the
  fallback. Record the rule in the Result.

**M4 — `passthrough_patterns` is ignored when assigning precisions.**
- **Where:** `gemma.rs` precision loop; `QuantizationDeclaration.passthrough_patterns`
  at `checkpoint_config.rs:151`.
- **What:** AutoRound regex overrides mark modules as unquantized source
  tensors. `from_checkpoint` would compose them INT8 for an AutoRound Gemma,
  which is a wrong graph rather than a refusal.
- **Fix:** one guard before the loop:
  `if quantization.passthrough_patterns.is_empty()` is false, return
  `InvalidArtifact` naming the first pattern. Matching them is future work
  with no consumer.

## LOW

**L1 — shrink.** `checkpoint_config.rs` `flatten_declared`: its `Array` arm
duplicates `declared_value`'s `Array` arm. Replace both non-object arms with
`value => insert_declared(key, declared_value(value)?, fields)?`, saving 7 lines.

**L2 — dead check.** In `text_config_from_declared`, the `.filter(|_|
strides.next().is_none())` uniqueness test can never fire. Two strides in
`1..=layers` give different full-attention sets: the smallest full layer is
`s-1`. Replace the `strides` pair with `(1..=layers).find(...)`, saving 3 lines.

**L3 — the quantization's group and symmetry are not carried or checked.**
The contract text says "INT8 affine group 32 symmetric". The graph can only
hold the precision, and `CheckpointGraph` drops the `granularity` /
`zero_points` facts. That is acceptable for this task, because the loader
(route item 5) re-parses `config.json`. State it in the Result's "Not
claimed" line rather than implying it.

**L4 — record error.** The Result says "builder, Claude Opus", but the
commit's `Co-Authored-By` is Claude Sonnet 5 and the ledger names Sonnet as
builder.

Net ponytail: −10 lines possible (L1, L2), plus M3 deleting its fallback branch.
