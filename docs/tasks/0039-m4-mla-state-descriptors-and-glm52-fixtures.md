# Task 0039 — M4.2a shared MLA/state descriptor and a GLM-5.2 mathematical oracle

Status: **accepted** (owner, 2026-09-21). Built by Codex `luna`, independently
reviewed and re-reviewed by Codex `sol`; two rounds — round 1 found two
blocking (P1) correctness defects in the RoPE rotation and its test coverage,
one citation defect and one ponytail finding; round 2's repair was
re-reviewed and accepted with no remaining finding. This closes M4.2's first
half; the second half (reference `Op::MlaAttention` plan admission) is
[task 0040](0040-m4-mla-reference-plan-admission.md).

## Identity and authority

- Task0039, first bounded M4.2 task; roadmap deliverable 2 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.2`).
  Builder Codex `luna` (max, `/ponytail:ponytail`); independent reviewer Codex
  `sol` (high, read-only, `/ponytail:ponytail-review`); coordinator Claude
  Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base commit `5f9b6c9`. Confirm `git status` clean and `HEAD` unmoved before
  starting; report if not.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Checkpoint roots `/models` and
  `/fast/models` are read-only inputs (ADR 0020); nothing under them may be
  copied, converted or bulk-read beyond metadata and small tensor headers
  needed to pin shapes/dtypes for this task's synthetic fixtures.
- Requirement repaired: roadmap `M4.2`, first half — "Add shared MLA/state
  descriptors and GLM-5.2 mathematical fixtures." The second half —
  "Implement absorbed/layout-specific paths only after mask, projection and
  rounding validation" — names the explicit stop condition below and is a
  later task's scope, not this one's.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  (MLA is "latent/positional state and projection semantics, not pretend full
  KV with a giant hidden expansion"; "Plan absorption/fusion only when
  algebra and required rounding boundaries permit it"), [document 08](../spec/08-strata-reference-map.md)
  lines 121–125 (GLM-5.2 absorption evidence, out of scope here — cited only
  so the builder does not rediscover it and reach for it early), and
  [document 03](../spec/03-memory-formats-and-cuda.md) ("MLA/state bytes =
  sum of declared layer state schemas, not full-KV formula").
- Required source/test references: `crates/moxie-graph/src/lib.rs`
  (`Op::MlaAttention`, already a refused stub), `crates/moxie-state/src/lib.rs`
  (`StateKind::MlaLatent`, already a refused stub, `RestoreCapability::Truncate`),
  `crates/moxie-plan/src/lib.rs` (`StateRequirement`, `StateEffect`,
  `stateful_resource_plan`), `crates/moxie-oracles/src/attention.rs` and
  `crates/moxie-oracles/src/mask.rs` (the existing FP64 dense-attention oracle
  and `attend_row`, the pattern a new MLA oracle follows), task 0037 and 0038's
  records for how an oracle-backed shape/precision contract was bounded before
  a kernel existed.
- Legacy source (read-only, algebra reference only — no execution, no
  weights read beyond header/shape; corrected 2026-09-21 after review found
  the ranges below off by the query stage): `src/models/glm52/glm52_runtime.cpp:46-49`
  (`kv_lora_rank`, `rope_head_dim`, `nope`/`rope` query split), `:824-845`
  (query down projection, RMSNorm, up projection to `q_b_proj`), `:846-855`
  (RoPE applied to each head's rope slice of the up-projected query), `:857-882`
  (KV down-projection to `kv_lora_rank + rope_head_dim`, RMSNorm on the latent
  half only, RoPE on the rope half only, latent and rope cached separately),
  `:884-892` (cache append), `src/models/glm52/glm52_ops.cpp:85-114`
  (`glm_rope_interleaved_f32`: reads adjacent input pairs `2*index, 2*index+1`
  but writes the rotated halves **split**, to `values[index]` and
  `values[half+index]` — not back to `2*index, 2*index+1`; position zero on
  `[1,2,3,4]` is `[1,3,2,4]`), `:977-1001`
  (decompression back to `nope + rope` keys at read time), `docs/models/glm53.md:236`
  and `kernels/cuda/detail/backend_absorbed_attention.inc.cuh:1,:35` (the
  **absorbed** path — read for context, not ported: this task builds the
  reference path absorption is checked against, not absorption itself).
- Checkpoint evidence grounding the fixture shapes (read-only, metadata only,
  discovered this session — **not** in the roadmap's O1 v1 catalog and no
  catalog/quality claim follows from its presence): `/fast/models/cyankiwi/GLM-5.2-AWQ-INT4`,
  revision commit `6e4f5c19d96e79e705d9918f763e132059dc668e`, `config.json`
  (`architectures: GlmMoeDsaForCausalLM`, `q_lora_rank: 2048`,
  `kv_lora_rank: 512`, `qk_rope_head_dim: 64`, `qk_nope_head_dim: 192`,
  `v_head_dim: 256`, `num_attention_heads: 64`, `rms_norm_eps: 1e-5`) and
  `model.safetensors.index.json` (layer 0 tensor names: `q_a_proj`,
  `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`,
  `kv_b_proj`, `o_proj` — the down-project/RMSNorm/up-project MLA shape,
  confirming the legacy runtime's algebra against real tensor geometry).
  This checkpoint's `self_attn.indexer.*` tensors are DeepSeek-style sparse
  index-selection (`Op::SparseIndexSelect`), a **separate, out-of-scope**
  mechanism fused into this particular revision; do not read, shape or
  fixture the indexer. [checkpoint-inventory.md](../evidence/checkpoint-inventory.md)'s
  "GLM-5.2 absent" line is stale as of this task and must be corrected as
  part of this task's record-keeping (see Result).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim anywhere in
  this task.

## Bounded deliverable

- **One concrete outcome:** (1) a real `StateKind::MlaLatent`/`Op::MlaAttention`
  descriptor — the shape/precision/state contract for MLA's down-projection,
  RoPE-on-rope-slice, latent cache and decompression, replacing today's bare
  refused stub, mirroring how `StateRequirement` already carries KV-paged
  attention's contract — and (2) a source-linked FP64 GLM-5.2 MLA oracle that
  computes the same equations independently, validated on synthetic tensors
  shaped from the real config above (no checkpoint weights read).
- **Sole owning shared component:** `moxie-oracles` for the reference math
  (mirrors `attention.rs`/`mask.rs`'s existing pattern); `moxie-graph` and
  `moxie-state`/`moxie-plan` for the descriptor vocabulary. A model crate
  owns none of this and is not touched by this task — there is no GLM-5.2
  model crate yet, and none is created here.
- **Allowed production and test files/modules:** `crates/moxie-oracles/src/`
  (new `mla.rs` or equivalent, plus its module registration), `crates/moxie-graph/src/lib.rs`
  (`Op::MlaAttention`'s associated params/shape contract only — not new op
  variants), `crates/moxie-state/src/lib.rs` (`StateKind::MlaLatent`'s
  associated geometry/precision fields), `crates/moxie-plan/src/lib.rs` only
  if a descriptor type is shared between plan and state (do not duplicate
  `StateRequirement`-shaped data). Corresponding test modules in each. Do not
  touch `moxie-executor`, `moxie-kernels`, `moxie-cuda`, or any CUDA-feature
  lane — nothing here runs on a GPU.
- **Explicit non-goals and forbidden shortcuts:** no absorbed/fused kernel or
  layout-specific fast path (document 06's own stop condition for this
  roadmap item); no DSA/sparse index selection (`Op::SparseIndexSelect`
  stays a separate, still-refused stub); no device kernel, no CUDA launch, no
  GPU test; no real checkpoint weights read beyond `config.json`/
  `model.safetensors.index.json` metadata already cited above; no GLM-5.2
  model crate, no graph wired to an actual checkpoint; no change to
  `Op::MlaAttention`'s current refusal in `stateful_resource_plan` — this
  task defines the descriptor's *shape*, it does not admit an MLA plan for
  execution (that is the next task, once this oracle exists to validate
  against); no performance claim.
- **Existing consumers and second-consumer/shape proof:** none yet consume
  `Op::MlaAttention`/`StateKind::MlaLatent` beyond the refusal paths already
  in the codebase (cited above) — this task's proof obligation is that the
  new oracle and descriptor agree with each other and with document 04's
  equations, not that a second model consumes them (no second MLA model is
  available or in scope; DeepSeek's MLA, named in document 06 as a later
  family, is evidence this vocabulary is shared rather than GLM-5.2-specific,
  cited, not built against).
- **Temporary paths to delete or bridge expiry:** none — this is additive
  vocabulary, not a bridge.

## Contract before implementation

- **Equations** (DeepSeek-style MLA, confirmed against both the legacy
  runtime and the real checkpoint's tensor names above): query path
  `q_a = x @ W_q_a` (down to `q_lora_rank`), RMSNorm, `q = q_a_norm @ W_q_b`
  (up to `n_heads * (qk_nope_head_dim + qk_rope_head_dim)`), split into
  nope/rope halves, RoPE applied to the rope half only. KV path
  `kv_a = x @ W_kv_a_with_mqa` (down to `kv_lora_rank + qk_rope_head_dim`),
  split into a latent part (RMSNorm'd, cached) and a rope part (RoPE'd,
  cached separately — this is what makes the state **latent**, not full
  KV: the cache holds `kv_lora_rank + qk_rope_head_dim` per token, not
  `n_heads * head_dim`). At attend time, `kv_b_proj` decompresses the cached
  latent back to per-head nope-key and value; the cached rope part is shared
  (MQA-style) across heads. Attention itself is standard scaled dot-product
  causal softmax over the reconstructed nope+rope keys and decompressed
  values, `o = attn_out @ W_o`.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  oracle computes in FP64 throughout, matching `moxie-oracles`'s existing
  attention oracle convention. Descriptor carries `q_lora_rank`,
  `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`,
  `heads`, `rms_norm_eps`, and cache precision (BF16 only, matching the
  existing paged store's constraint — do not silently admit FP16). Use the
  real config's dimensions (`q_lora_rank=2048`, `kv_lora_rank=512`,
  `qk_nope_head_dim=192`, `qk_rope_head_dim=64`, `v_head_dim=256`,
  `heads=64`) as at least one fixture shape; add a second, smaller synthetic
  shape so a shape/consumer test is not a single coincidence.
- **Partition and hardware capabilities:** none — host-only, no device
  binding in this task.
- **Peak memory and transfer dependencies; source/lease lifetime:** none —
  the oracle is a bounded host computation over `Vec<f64>`/`Vec<f32>`
  fixtures, no allocator, no lease.
- **Cancellation, failure and rollback behavior:** the oracle is a pure
  function; typed refusals for malformed geometry (zero rank, mismatched
  head/latent widths, nonfinite input) match the existing oracle module's
  refusal style (`crate::mask`/`crate::attention`'s `Error` usage).
- **Independent oracle; predeclared numerical metrics/thresholds:** this
  *is* the independent oracle for MLA — it has no oracle of its own beyond
  the algebra citation above and the legacy source's line references. Any
  numerical comparison in this task (e.g. checking that RoPE-only-on-rope-
  slice actually leaves the nope slice untouched, or that decompression is
  the inverse shape of compression) is an exact/analytic check, not a
  tolerance — this task produces no kernel to compare against a tolerance
  yet.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Host tests only: the new oracle module has unit tests proving (a) the
  down-projection/RMSNorm/up-projection shape chain is exact for both
  fixture shapes, (b) RoPE is applied only to the rope slice and the nope
  slice is bit-identical to its pre-RoPE value, (c) the cached latent width
  is `kv_lora_rank + qk_rope_head_dim` (not full per-head KV — this is the
  test that the state is actually latent), (d) decompression reconstructs
  per-head nope-key and value at the shape the descriptor declares, (e) the
  resulting attention scores match a direct FP64 dense causal computation
  over the reconstructed keys/values for a small case, and (f) malformed
  geometry (rank/width mismatches) is a typed refusal, not a panic or a
  silent truncation.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass. No GPU, no driver-feature lane — this task adds nothing there.
- Support-matrix entries: none created — MLA remains explicitly unsupported
  for execution; only the oracle/descriptor exist. Do not mark MLA
  "supported" anywhere.
- Deletion and documentation gates: correct `checkpoint-inventory.md`'s
  stale "GLM-5.2 absent" line (dated update, not a silent edit) with the
  commit hash and size above; no other deletion.
- **Exact condition requiring owner direction or task rejection:** if the
  legacy runtime's algebra and the real checkpoint's tensor shapes disagree
  in a way this task cannot resolve locally (e.g. an undocumented extra
  projection), stop and report rather than inventing a resolution. Do not
  begin the absorbed/fused path under any circumstance in this task — that
  requires this oracle to exist and be reviewed first, which is the whole
  reason M4.2's roadmap line is split the way it is.

## Result, filled after work

- **Changed shared owners and consumers; source commit:** uncommitted at
  writeup, base `5f9b6c9`, branch `main`. `crates/moxie-graph/src/lib.rs`
  (+`MlaAttentionDescriptor`: shape/validate methods, no new `OpParams`
  variant, `Op::MlaAttention` plan refusal unchanged), `crates/moxie-state/src/lib.rs`
  (+`MlaLatentDescriptor`: `cache_width = kv_lora_rank + qk_rope_head_dim`,
  BF16-only), `crates/moxie-oracles/src/lib.rs` (module registration),
  `crates/moxie-oracles/src/mla.rs` (new, the FP64 GLM-5.2 MLA oracle:
  `rotate`, `project`, `decompress`, dense causal attention over
  reconstructed keys/values), `docs/evidence/checkpoint-inventory.md`
  (GLM-5.2 presence correction). No consumer exists yet by design — this
  task's proof is internal (oracle vs. cited legacy source), not a second
  model.
- **Commands and result IDs; passed / failed / skipped separately:**
  **Passed** (round 2, final): `cargo test -p moxie-oracles --lib mla::tests --locked`
  4/4; `cargo test --workspace --locked` full pass; host clippy (targeted
  package and full workspace) `-D warnings`; CUDA-feature workspace clippy;
  `arch-check` (79 rejected fixtures, 21 accepted, 13 rules); `spec-check`
  (10 docs); `cargo fmt --all -- --check`; `git diff --check`. **Failed:**
  none surviving — round 1's two P1 defects (rotate output placement
  disagreeing with `glm_rope_interleaved_f32`; a RoPE acceptance test that
  could not have caught it) are repaired and re-reviewed. **Skipped:** GPU,
  driver-feature and model-graph lanes — out of scope by the task contract,
  not owed here.
- **Measured effect and uncertainty:** none — no timing, no performance
  claim (O6/O7 open). The oracle's correctness is the deliverable; verified
  by independent review tracing production `rotate` against
  `glm52_ops.cpp:85-114` operation-for-operation and confirming the new
  fixture would have failed the pre-repair implementation.
- **Deleted/replaced paths:** `MlaProjection`'s public `query_before_rope`
  and `cached_rope_before_rope` fields, removed per the reviewer's ponytail
  finding; the pre-RoPE reference needed by tests is now computed locally
  inside the test module.
- **Remaining blockers and next bounded task:** none for this task's own
  scope. M4.2's second half — admitting `Op::MlaAttention` through
  `moxie-plan`/`moxie-state` as a reference (unabsorbed) execution path, and
  only after that the absorbed/layout-specific fast path — is a successor
  task, not opened here; the roadmap's own text names that ordering. The
  DSA/sparse-indexer half of the GLM-5.2 checkpoint this task read metadata
  from remains completely untouched and unscoped.

Owner acceptance is requested for this scope only: the descriptor and oracle
above, not MLA execution or GLM-5.2 model support, neither of which this task
claims.
