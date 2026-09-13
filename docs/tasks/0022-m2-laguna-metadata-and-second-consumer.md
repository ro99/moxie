# Task 0022 — M2 item 4: Laguna metadata, a second routed consumer, and a budget smaller than its weights

Status: proposed.

Roadmap **M2 item 4**, in full: "Add Laguna model metadata/graph only. Add a
second synthetic MoE consumer with different expert count, activation, top-k,
shapes and route distribution. Exercise an intentionally restricted memory
budget smaller than its working weights."

It does **not** close M2. M2's exit also needs byte/cost traces reconciled with
the resource ledger across a whole working set, which is item 5's remainder and
a later task.

## Why this task exists in this shape

Task 0019 built the routed mathematics against one family's parameters and one
synthetic counter-example. Task 0021 built the execution path against one
family's shapes. Both are correct and both are narrow in the same way the fourth
review of task 0021 named: **a gate only fires on inputs something actually
hands it.** Every routed fixture in the workspace today comes from a router that
softmaxes, normalises its own input, carries a per-expert coefficient scale and
selects 8 of 128 or 3 of 3.

Laguna is the first artifact on this machine whose router disagrees with every
one of those, and it disagrees in ways that are *pinned in a local source file*
rather than guessed. That is what makes it a second consumer rather than a second
fixture.

## Identity and authority

- Task ID 0022, milestone **M2 item 4**. Owner review required for acceptance;
  independent review required before that, per the standing practice on tasks
  0019–0021.
- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`, base
  commit `551b7cd`, working tree clean at authoring time. This contract is
  committed **before** implementation; the implementation is the commits between
  it and the handover that accompanies it.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, re-verified at authoring time; its
  untracked `.pi/` and `tests/p2p/` remain untouched. **The frozen legacy tree
  has no Laguna adapter**, so it is not a reference for anything below.
- Local checkpoint roots `/models` and `/fast/models` stay read-only inputs.
  Nothing under either may be copied, converted, deleted, downloaded or
  modified. Bounded header reads only.
- Requirements repaired: roadmap M2 item 4; document 02's enforced extension
  rule steps 1–5 for the routed operations; document 03's affine-integer v1
  descriptor as the *recording* contract for Laguna's metadata.
- Required reading, done before this contract was written: documents 02, 03 and
  06 (M2); [ADR 0003](../decisions/adr/0003-int4-int8-bf16-weight-family.md);
  [the gemma4 bring-up record](../models/gemma4.md) as the template a second
  family's record follows; [task 0018](0018-m3-compressed-tensors-int8-importer.md)
  for what compressed-tensors metadata is already readable; tasks 0019–0021 for
  the routed contract and the interface this consumer must go through.
- Owner gates: **O1 (catalog), O2 (quality) and O5 (storage/conversion) remain
  open and this task resolves none of them.** `hy3-w4a16-mtp` is in scope per
  AGENTS.md, which is what authorises inspecting the artifact; it is not
  approval of the model, its quality or any conversion.

## What this task is not

- **It is not Laguna support.** Nothing here reads a Laguna tensor payload,
  imports one, or executes one. A graph that nothing runs on real weights is not
  support, and saying so is a stop condition in the task 0021 handover.
- **It is not an importer.** Laguna's linears are asymmetric INT4 group 32, and
  [`PackQuantizedSpec`](../../crates/moxie-format/src/compressed_tensors.rs)
  refuses asymmetric sources today. Extending it is M3's, and this task must not
  do it as a convenience.
- **It is not the Laguna attention tower.** See "The two gaps that block the
  tower" below: two shared operations do not exist, and inventing either from a
  suffix is forbidden.
- **It is not a second residency owner, a second cache, or a model-owned
  execution path.** `moxie-memory` gains nothing.
- **No quality claim of any kind follows from anything here.**

## What is interpreted, and what is deliberately not

The artifact is `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`, revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`. Its completeness was re-verified at
authoring time: for each of the 15 shards `8 + header + payload_end` equals the
file size exactly, and the payload ends sum to the index's `total_size` of
76,813,095,232 B.

**Interpreted** — every field below is read from `config.json`, the safetensors
index, or bounded shard headers, and each has a named authority:

| Field | Value | Authority |
|---|---|---|
| `hidden_size` / `num_hidden_layers` / `head_dim` | 3,072 / 48 / 128 | `config.json`, confirmed by tensor extents |
| `num_attention_heads_per_layer` | 48 on layers `l % 4 == 0`, else 72 | `config.json`; `q_proj` extents are 6,144 and 9,216 |
| `num_key_value_heads` | 8 | `config.json`; `k_proj`/`v_proj` extents are 1,024 |
| `layer_types` | `full_attention` on `l % 4 == 0`, else `sliding_attention`; `sliding_window` 512 | `config.json`, and `LagunaAttention.__init__` reads exactly this |
| `intermediate_size` (dense MLP) | 12,288 | `config.json`; layer 0 `mlp.gate_proj` is `[12288, 3072]` |
| `mlp_only_layers` / `decoder_sparse_step` | `[0]` / 1 — layer 0 dense, layers 1–47 routed | `LagunaDecoderLayer.__init__`, and `mlp_layer_types` agrees |
| `num_experts` / `num_experts_per_tok` | **256 / 10** | `config.json`; the index holds 256 experts on each of 47 layers |
| `moe_intermediate_size` | 1,024 | `config.json`; expert `gate_proj` logical shape is `[1024, 3072]` |
| `shared_expert_intermediate_size` | 1,024 | `config.json`; `shared_expert.gate_proj` is `[1024, 3072]` |
| `hidden_act` | `silu` — so the expert gate transform is **SwiGLU** | `config.json`; `LagunaMLP`/`LagunaExperts` both use `ACT2FN[config.hidden_act]` |
| `norm_topk_prob` | true | `config.json`; `LagunaTopKRouter.forward` |
| `moe_routed_scaling_factor` | **2.5** | `config.json`; `LagunaSparseMoeBlock.forward` |
| `moe_router_logit_softcapping` | **0.0 — disabled** | `config.json`; the branch in `LagunaTopKRouter.forward` is `> 0.0` |
| `moe_apply_router_weight_on_input` | false | `config.json`; the pinned block raises `NotImplementedError` when true |
| `rms_norm_eps` | 1e-6 | `config.json` |
| `vocab_size` / `tie_word_embeddings` | 100,352 / **false** — a separate `lm_head` | `config.json`; `lm_head.weight` is its own tensor |
| `max_position_embeddings` | 1,048,576 — **not** an admissible context here (R19) | `config.json` |
| `rope_parameters.sliding_attention` | `default`, theta 10,000, `partial_rotary_factor` 1.0 | `config.json`; `compute_default_rope_parameters` is in the artifact's own file |
| Per-expert disk bytes | **5,455,920 B** (3 × [packed 1,572,864 + scales 196,608 + zero points 49,152 + shape 16]) | shard header `data_offsets` |
| One routed layer's 256 experts | **1,396,715,520 B** quantized; **4,831,838,208 B** on layers 46 and 47, whose experts the quantizer left in BF16. All 47 routed layers: **72,515,874,816 B**, **94.41%** of the artifact — a sum over the layers, not a multiplication | the same |

**Not interpreted, and each is a stop rather than a guess:**

1. **The yarn RoPE ramp on `full_attention` layers.** `config.json` declares
   `rope_type: "yarn"`, `factor` 128.0, `beta_fast` 32.0, `beta_slow` 1.0,
   `original_max_position_embeddings` 8,192, `attention_factor`
   1.4852030263919618, `partial_rotary_factor` 0.5, `rope_theta` 500,000.0. The
   artifact's own `modeling_laguna.py` implements only
   `compute_default_rope_parameters` and delegates every other type to
   `ROPE_INIT_FUNCTIONS`, which is **not in the artifact**. The locally
   installed `transformers` is 5.5.3; the artifact declares **5.14.1**. One
   mismatched copy is not a pinned exporter, and `truncate` — a parameter of
   that function — is not declared in the config at all, so its default is the
   kind of thing that moves between versions. **Refused until a matching pinned
   source establishes it.**
2. **Attention output gating.** `LagunaAttention.forward` computes
   `softplus(g_proj(x))` in FP32, per head, and multiplies it into the attention
   output **before** `o_proj`. The equation is pinned in the artifact's own
   file, so it is known; what does not exist is a shared operation for it. It is
   a gap task, not something to fold into `Attention`'s parameters silently.
3. **The mapping from the checkpoint's per-expert tensors to
   `LagunaExperts`' fused `gate_up_proj`.** The pinned model declares
   `gate_up_proj` `[experts, 2·intermediate, hidden]` and `down_proj`
   `[experts, hidden, intermediate]`; the checkpoint stores
   `mlp.experts.{e}.{gate,up,down}_proj` per expert. The artifact's
   `_checkpoint_conversion_mapping` remaps **only** `e_score_correction_bias`
   and says nothing about this. The concatenation order is *implied* by
   `chunk(2, dim=-1)` in `LagunaExperts.forward`, and implication is not the
   pinned exporter. **Recorded as an open importer question, resolved by
   whoever writes the importer, not here.**
4. **Every tensor payload.** No Laguna weight bytes are read. Asymmetric INT4
   group 32 has no importer, and `weight_zero_point` is packed along the
   **output** axis (`[1024/8, 96]` against `weight_packed`'s `[1024, 3072/8]`),
   which is a second packing convention the existing importer has never seen.
   Both facts are recorded for M3; neither is acted on.
5. **The tokenizer, chat template, `reasoning_parser`, `tool_call_parser` and
   the declared `dflash` speculative draft model.** M8 and M9 own those.

## The two gaps that block the tower, and why the deliverable is the block

A faithful Laguna decoder layer needs both of the not-interpreted items above:
gating is on **every** layer (`gating_types` is `per_head` 48 times), and yarn is
on the twelve `full_attention` layers. So there is no layer of this model whose
attention is composable from today's operation catalogue, and there is no honest
"reduced" version of it either — dropping the gate drops mathematics, which is
not what `gemma4::Reduction` means.

Composing the tower anyway would require inventing one equation and one version's
behaviour. The contract therefore delivers **the routed block**, which is
composable in full from pinned sources, and names the two gaps as tasks. This is
a narrowing of "metadata/graph" and it is reported here rather than discovered
later: **M2's exit gate does not need Laguna's attention**, and M7's family
coverage does.

## Bounded deliverable

One concrete outcome, in five parts.

### 1. The routed operations grow the parameters Laguna needs, with oracles

Sole owning components: `moxie-graph` (the operation contract), `moxie-oracles`
(the independent host reference), `moxie-interp` (the shared evaluator). No new
crate, no new op variant — `Route` and `Combine` gain parameters, because
document 02 says differing routing "consumes the same dispatch, transfer,
grouped compute, and reduction machinery".

`Route` today hard-codes four choices that are Gemma's, not routing's:

| Choice | Gemma 4 | Laguna | Becomes |
|---|---|---|---|
| Router input | its own scale-free RMSNorm, a bound gain, then `· hidden^(-1/2)` | the block's `post_attention_layernorm` output, unchanged | `input: RouterInput::{Normalized { eps, input_scale }, Raw}` |
| Score transform | softmax over all experts | **sigmoid**, per expert, independent | `score: RouteScore::{Softmax, Sigmoid}` |
| Selection score | the score itself | score **+ `e_score_correction_bias`**, a bound `[experts]` tensor | `selection_bias: bool`, a trailing optional input |
| Coefficient | renormalised, then `· per_expert_scale` | renormalised, **no** per-expert scale | `per_expert_scale`, unchanged |

`Combine` gains `output_scale: f32`, applied to the combined row at the BF16
boundary: `bf16(Σ terms · scale)`. Laguna's `moe_routed_scaling_factor` of 2.5
multiplies the **summed** routed output, before the shared expert is added.
`Residual { scale }` cannot express it — that is `bf16(bf16(a + b) · scale)`,
which scales the shared expert too.

Two parameters are deliberately **not** added, because nothing hands them a
value and an unreachable branch is a stub (task 0021's own finding, twice):

- **Router logit softcapping.** Laguna declares 0.0. `Route` must **refuse** a
  configuration that needs it rather than carry a dead branch, and the Laguna
  definition must refuse to compose when `moe_router_logit_softcapping > 0.0`.
- **`norm_topk_prob = false`.** Both families declare true. Renormalisation
  stays unconditional and the first family that declares false adds the
  parameter with its fixture.

**The hazard this creates, named before it is built.** `per_expert_scale` and
`selection_bias` are both `[experts]` and would both sit at the tail of the
input list, so a swap is **not** caught by shape validation — the exact shape of
the fourth review's swapped-symbol finding. Mitigations, both required:

- the input order is fixed and documented as `[rows, projection, gain?,
  per_expert_scale?, selection_bias?]`, and
- a substitution test binds a fixture that carries **both**, swaps them, and
  requires a different answer. A fixture on which the two agree tests neither.

### 2. Laguna metadata and the routed-block graph

Sole owning component: `moxie-models`, new module `laguna`. Its dependency list
is unchanged and unchangeable: `moxie-types`, `moxie-graph`, `moxie-model-api`.

- `laguna::ARTIFACT`, an executable statement of the declared geometry table
  above, including the fields that **cannot** be composed, so a test can compare
  it against `config.json` on re-inspection and so the record's numbers have a
  counterpart that fails when they drift.
- `laguna::MoeBlock`, a `ModelDefinition` composing one routed block: the
  shared expert (`SwiGlu` at 1,024), the sigmoid router with its selection bias,
  `ExpertMlp` with `SwiGlu` at 256 experts and top-k 10, `Combine` with
  `output_scale` 2.5 and `AscendingExpertId`, and the sum of the two branches.
  Tensor roles are the canonical ones; the fused expert roles carry the open
  mapping question in their documentation rather than resolving it.
- A `Reduction` that states what it is not, in the same shape Gemma's does and
  with the two gaps named: block only, no attention tower, synthetic weights,
  INT4 not imported.
- `compose` **refuses**, with a typed error naming the gap, when asked for a
  full-attention layer, an attention gate, or a nonzero router softcap.

The combination order is `AscendingExpertId` on evidence, not by analogy:
`LagunaExperts.forward` iterates `expert_hit`, which is `nonzero()` over an
expert-major mask, and accumulates with `index_add_` — the same construction the
gemma4 record cites for the same conclusion.

### 3. The second consumer goes through task 0021's interface, not beside it

Sole owning components: the **tests** of `moxie-plan` and `moxie-executor`. No
structural change to either crate: if the second consumer needs one, that is a
finding about the interface and it is reported, not worked around.

The handover is explicit that the extension is cheaper than new point tests and
that the measurement already exists to say whether it helped. So:

- `moxie-plan/tests/expert_plan_matrix.rs` gains a **profile axis**
  `{GemmaLike, LagunaLike}` over its existing product. The Laguna profile
  carries a different expert count, a different top-k, `SwiGlu`, different
  hidden/intermediate widths and a route distribution with a different overlap
  structure. The printed coverage gains the axis.
- `moxie-executor/tests/grouped_transitions.rs` gains the same axis over its
  144 combinations. A different top-k changes the slot arithmetic; a different
  expert count changes the queue's behaviour under a restricted cache.
- **Both sweeps are re-measured by mutation after the extension**, and the
  numbers go in [experiment 0002](../evidence/experiments/0002-expert-plan-sweep-mutations.md)
  beside the existing ones. An extension that catches nothing new is reported as
  catching nothing new.

### 4. A restricted budget, expressed as a ratio

The budget is `ceil(working_set_bytes · ratio)` with a **declared ratio**, never
a constant, and a test asserts `budget < working_set` by construction so the
case cannot quietly stop being restricted when a shape changes.

`working_set_bytes` is the union of the experts the route actually demands times
the per-expert chunk size — document 03's "estimate union of required experts
over a row batch", not `rows · top_k · chunk`.

The declared ratio for this task's cases is **1/8**, chosen so that a plan whose
route demands more than eight experts must evict at least once under any
policy; task 0021's real-artifact case used two experts of ten, and 1/8 is
tighter. The ratio is a test parameter, printed with the case, not a threshold
anything in production reads.

At Laguna's declared expert shape the arithmetic the cases run on is:

```text
one expert, BF16, at Laguna's shape = 3 · 1024 · 3072 · 2 = 18,874,368 B
the same expert as the artifact stores it, asymmetric INT4 group 32 = 5,455,920 B
```

The cases use the **BF16** figure, because M2 item 1 says "Use BF16 initially to
isolate residency correctness from quantization" and because no importer can
produce the other. Both numbers are recorded so that the difference is a fact
rather than a surprise to M3.

### 5. The records

- `docs/models/laguna.md`, a bring-up contract on the shared template: identity,
  the mathematical inventory with a gap task in every row that has one, the
  logical tensor-role mapping, what the quantizer left unquantized, and
  blockers.
- `docs/evidence/support-matrix.md` gains only what passing gate IDs support.
  **No Laguna capability row.**
- `docs/models/README.md`, `docs/tasks/README.md`, `docs/handovers/README.md`
  and AGENTS.md updated to the new state.

### Temporary paths to delete, and bridge expiry

None. Nothing here is a bridge. If the implementation needs one, it is named in
the Result section with the gate that expires it.

## Contract before implementation

### Equations, shapes, precision, rounding

The router, transcribed from `LagunaTopKRouter.forward` and
`LagunaSparseMoeBlock.forward` in the artifact's own `modeling_laguna.py`, read
and **never executed**:

```text
x            = the block's post_attention_layernorm output          [rows, 3072]
logits       = x · Wᵀ                        computed in FP32       [rows, 256]
scores       = sigmoid(logits)                                      [rows, 256]
selection    = scores + e_score_correction_bias                     [rows, 256]
ids          = top_k(selection, 10)          lower id wins a tie
w            = scores[ids]                   the UNBIASED scores    [rows, 10]
w            = w / Σ w                       norm_topk_prob = true
y_slot[r,j]  = down_e( silu(gate_e(x_r)) · up_e(x_r) ) · w[r,j]
y[r]         = (Σ_j y_slot[r,j]) · 2.5       ascending expert id
out[r]       = y[r] + shared_expert(x_r)
```

Three properties of that are load-bearing and each gets its own fixture:

1. **The bias moves selection and not coefficients.** A router that gathered the
   biased score would still produce a well-formed distribution. The fixture is a
   row where the bias changes which experts are selected *and* where using the
   biased score for the coefficients gives a different, plausible answer.
2. **Sigmoid scores do not sum to one before renormalisation, and need not after
   it in the same way softmax does.** Independent per-expert scores are a
   different function, not softmax with a flag. The fixture is a row whose
   sigmoid and softmax selections differ.
3. **`output_scale` multiplies the combined row, not each term.** Scaling each
   term before summing is a different rounding pattern at BF16. The fixture
   distinguishes them.

Precision follows the accepted routed contract without exception: FP32 router
logits, FP64 oracles, BF16 boundaries where the pinned reference has them, and
`round_to_bf16` at every activation boundary the existing operations already
round at. No new rounding boundary is introduced anywhere.

### Independent oracle and predeclared numerical gates

- The oracle is an FP64 transcription in `moxie-oracles::route`, independent of
  the interpreter, extended rather than duplicated: one selection function, one
  tie rule, one renormalisation, as that module already enforces.
- **Gate: bitwise equality** between the interpreter and the oracle on the
  routed block, for both score transforms and both coefficient modes. This is
  the gate tasks 0019 and 0021 already declared; nothing is loosened.
- `route::expert_error_bound` must cover the sigmoid path or be extended to; a
  bound that silently applies a softmax-derived constant to a sigmoid router is
  a wrong bound.
- **No new numerical threshold is declared and none is changed.** If one turns
  out to be needed, that is the stop condition below.

### Resource envelope, transfer dependencies and lifetimes

Unchanged from task 0021, and that is the point: the second consumer must be
admitted, leased, uploaded, evicted and reduced by the same owner with the same
envelope arithmetic. Specifically:

- `moxie-memory::residency` stays the one weight-residency owner. A second cache
  anywhere in this task is a failed task.
- The device budget for the restricted cases comes from `ExpertBudget` as it
  stands; if Laguna's shapes need a new field, that is a finding to report, not
  a field to add quietly.
- Host buffers keep task 0021's NUMA contract: `required` means `mbind` and the
  gate is every page, read back from `/proc/self/numa_maps`. A placement claim
  that has not been read back is not a measurement.

### Cancellation, failure and rollback

Unchanged and re-swept: the run sweep's axes — failure point, cancellation,
close ordering, queue depth — are exercised on the Laguna profile too, with the
residency authority's lease count reconciled against the run's after every
operation, as `GroupedRun::check_invariants` already requires.

### Partition and hardware capabilities

No partition change. One plan targets one device; spreading a layer across the
three cards is M5. The device cases run on all three GPUs (two sm_86, one
sm_120) or the task reports which did not and why.

### Application compatibility and sampler implications

None. No protocol surface, sampler, context target or compatibility matrix
changes. The diagnostic CLI may gain a shape for the second consumer; if it
does, its disclosure line says what the output is not, as every other shape's
does.

## Acceptance

### Gates that must pass, reported separately

| Gate | Requirement |
|---|---|
| `cargo fmt --all -- --check` | passes |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passes |
| Device-lane clippy | passes |
| `cargo test --workspace --locked --offline` | passes, count reported against task 0021's 883 |
| Device-feature workspace tests | passes, count reported against task 0021's 913 |
| `cargo xtask-cuda test-gpu` | passes on sm_86 and sm_120, 0 skipped, count reported against 42 |
| `cargo xtask spec-check` | passes |
| `cargo xtask arch-check` | **zero failures.** Not "zero new failures" |

Failed, skipped and unmeasured are reported separately. A case that skips prints
why.

### The cases this task must add

1. **Router semantics.** Sigmoid versus softmax selection differ on a fixture;
   the selection bias changes the selection; the coefficients come from the
   unbiased score; a swapped `per_expert_scale`/`selection_bias` binding changes
   the answer; `output_scale` applied to the sum differs from applying it per
   term; a nonzero router softcap is **refused**.
2. **The Laguna block composes and runs** through the host interpreter, over
   synthetic weights, with whole-versus-chunked parity — the same property every
   other consumer must satisfy.
3. **`ARTIFACT` matches `config.json`**, field by field, in a test that reads
   the artifact when present and prints `SKIPPED` with a reason when absent.
4. **The refusals fire**: a full-attention layer, an attention gate and a
   nonzero softcap each produce a typed error naming its gap task.
5. **Both sweeps carry the profile axis**, print their coverage, and are
   re-measured by mutation.
6. **The restricted budget**, at the declared ratio, on the host lane and on all
   three GPUs: the run completes, evicts, drains backpressure, agrees between
   candidates, and reconciles leases.
7. **M2 item 5's nine residency cases, task 0021's two sweeps and its device
   cases keep passing unchanged.** Not "keep passing with adjustments".

### Coverage is measured, never asserted

Every claim this task makes about its own tests is a number the test prints or a
mutation battery result, never a sentence in this file. Specifically:

- the two sweeps print their exercised product, including the new axis;
- the mutation battery is re-run over both and the before/after numbers are
  recorded, including "the extension caught nothing new" if that is the result;
- **every regression added for a review finding is substitution-tested** — a
  regression that passes with its check removed asserted the symptom, and task
  0021 found seven of those across two rounds;
- an equivalent mutant is reported as equivalent, not counted as a gap.

### Support matrix and documentation gates

- `docs/models/laguna.md` exists and its integration proof shows the adapter
  contains metadata and graph composition and nothing else.
- The support matrix gains **no** Laguna capability row. It may gain rows for
  the routed-operation parameters, each linked to a passing gate ID.
- AGENTS.md and the three record indexes state the new position without
  overclaiming it.

### The exact condition requiring owner direction or task rejection

Stop and report, rather than proceeding, if any of these occurs:

1. Composing the routed block requires a Laguna metadata field whose meaning is
   not established by a pinned source — including anything under "Not
   interpreted" above being needed after all.
2. The second consumer cannot go through task 0021's interface without a
   structural change to `moxie-plan` or `moxie-executor`. Report the change the
   interface needs; do not make the consumer special.
3. A residency or execution behaviour needs a second owner, a second cache, or a
   model-owned path.
4. A declared numerical gate would have to be loosened for a case to pass.
5. Executing remote model code, reading a Laguna tensor payload, or any write
   under a checkpoint root would be required.
6. The restricted budget cannot be satisfied without an unbounded fallback.

Any of 3, 4, 5 is a failed task rather than a smaller one.

## Result, filled after work

Status: **implemented, 2026-09-13; corrected after three rounds of independent
review, the third of which recommended acceptance with no new blocking
findings; awaiting owner acceptance.** It does not close M2.

### Changed shared owners and consumers

| Crate | What changed |
|---|---|
| `moxie-graph` | `Route` gained `RouterInput`, `RouteScore` and a `selection_bias` operand; `Combine` gained `output_scale`; `RouteOperand` and `OpParams::route_operands` are the single statement of the router's operand list, and the arity and the shape validation both derive from it |
| `moxie-oracles` | `router_logits`, `router_scores`, `select_top_k_biased`, a `combine_row` that carries the scale, and an FP64 transcription that covers **both** families' routers |
| `moxie-interp` | resolves the router's operands from `route_operands` rather than from positions written out again |
| `moxie-kernels` | `combine_rows_bf16` applies the output scale once, to the finished FP32 accumulator, before the single BF16 store |
| `moxie-plan` | `CombineSpec` carries the `Combine` node's two parameters together; `ExpertPlan` carries `output_scale` |
| `moxie-executor` | `reduce` passes the plan's scale; **no structural change** — the second consumer goes through the interface as it stands |
| `moxie-models` | new `laguna` module: `ARTIFACT`, `RoutedBlocks`, `Reduction`, `Gap` and `ArtifactGeometry::tower_gaps` |
| `moxie-memory` | **nothing**, as the contract required |

Base commit `551b7cd`; contract committed at `78c493d` **before** implementation.

### Commands, and what each reported

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy (`--features moxie-executor/driver`) | passed |
| `cargo test --workspace --locked --offline` | **908 passed, 0 failed** (883 at task 0021, measured in an isolated worktree at `551b7cd` rather than quoted) |
| Device-feature workspace tests | **941 passed, 0 failed** (913 at task 0021) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures**, 78 rejected fixtures, 21 accepted, 13 rules |

Nothing failed. Nothing was skipped: the real-artifact cases print `SKIPPED`
with a reason when their artifact is absent, and on this machine none did —
including `the_declared_geometry_matches_the_artifact_config`, which read
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4/config.json` and compared every
declared field against `laguna::ARTIFACT`.

### What executed, and what that is not

**A routed layer at the artifact's declared expert shape ran on all three
GPUs.** Top-k 10 over twelve experts of `3 · 1024 · 3072 · 2 = 18,874,368 B`
each — a working set of **226,492,416 B** — against a device cache of
**28,311,552 B**, one eighth of it. Each device drained backpressure **11**
times, the authority evicted **22** times, and all **6,144** BF16 components
agreed with the CPU candidate on every card:

```text
device 0 GPU-97fe4889-4874-a378-198e-955d2e72c4a3   11 drains, 22 evictions
device 1 GPU-3032cfa3-19df-028f-5ebd-43314911e0b9   11 drains, 22 evictions
device 2 GPU-81fe4578-59b2-37c4-421e-287cdac78704   11 drains, 22 evictions
```

**Those are weight-shaped bytes this task invents at the artifact's declared
shape, not its weights.** No Laguna tensor was read. The artifact's experts are
asymmetric INT4 at group 32, which no importer accepts, and a BF16 expert at
that shape costs 18,874,368 B where the stored one costs 5,455,920 B. **Nothing
about Laguna's output follows from this**, and O2 is untouched.

### The two gaps, reported rather than worked around

A faithful Laguna decoder layer needs two operations the catalogue has not got,
and **both sit on attention**, so **no layer of this model is composable today**:

1. **Attention output gating** — `softplus(g_proj(x))` per head, before
   `o_proj`. Pinned in the artifact's own file, so the equation is known; what
   is missing is a shared operation, an oracle and a second consumer.
   `gating_types` is `per_head` on all 48 layers.
2. **The yarn rotary ramp** on the twelve `full_attention` layers.
   `modeling_laguna.py` implements only `compute_default_rope_parameters` and
   delegates every other `rope_type` to a `transformers` function it does not
   ship. The artifact declares `transformers` **5.14.1**; the copy on this
   machine is **5.5.3**, and `truncate` — a parameter of that function — is not
   in this config at all. One mismatched copy is not a pinned exporter.

So the deliverable is the **routed block**, which is composable in full, and the
narrowing was written into this contract before implementation rather than
discovered afterwards. `ArtifactGeometry::tower_gaps` computes the list from the
declared geometry, so a configuration without them reports none — both outcomes
are reachable and both are tested.

### Decisions worth finding again

**Sigmoid and softmax select the same experts; the bias is what makes the
transform decide anything.** Both transforms are monotone in the logit, so on
their own they differ only in the *coefficients*. Add `e_score_correction_bias`
and they stop agreeing, because a bias shifts a sigmoid score and a softmax
probability by different relative amounts. Two fixtures say exactly that, and
the second exists because the obvious claim — "a different score transform
selects different experts" — is **false** and would have been written down as a
comment otherwise.

**Two parameters were deliberately not added.** Router logit softcapping (the
artifact declares 0.0) and `norm_topk_prob = false` (both families declare true)
have no consumer, and AGENTS.md counts an unreachable branch as a stub. A
configuration that needs the softcap is **refused**, by name, where the gap is
visible.

**A scale on the sum is not a scale on the terms.** `Combine::output_scale`
multiplies the FP32 accumulator once, before the single BF16 store — which is
where `moxie_oracles::route::combine_row` already puts its rounding. The fixture
that distinguishes the two application points is a cancellation at `2^24` with
Laguna's own factor of 2.5: sum-then-scale gives 2.5, scale-then-sum gives 4.0.

**The two `[experts]` operands are the hazard the fourth review of task 0021
described, in a new place.** `per_expert_scale` and `selection_bias` have the
same shape, so a swapped binding passes every shape check.
`OpParams::route_operands` is the one statement of the order — the arity, the
shape validation and the interpreter all read it — and an oracle-backed test
pins what each one *means*, not merely that they differ.

### Coverage, measured rather than asserted

Both sweeps gained a profile axis over their whole product:

- `expert_plan_matrix.rs`: **10,368** combinations, 5,184 per profile, every
  rejection reason exercised, counts printed by the test.
- `grouped_transitions.rs`: **288** combinations, 144 per profile, every failure
  point, both capacity outcomes, and the authority's live-lease count reconciled
  against the run's after **every** operation.

Strength, measured by mutation: **33 of 33 caught, 0 survivors** —
[experiment 0003](../evidence/experiments/0003-task0022-sweep-and-routed-mutations.md).
The first measurement had **three survivors across the two batteries**, and both
defects they found are worth carrying:

1. **A parameter no fixture ever varies is a parameter no test checks.** Two
   mutants that dropped `Combine::output_scale` on the way to the plan survived
   the entire 10,368-combination sweep and every executor test, because every
   fixture in the workspace used 1.0. The sweep now checks it, and a new
   executor regression reduces the same fixture at 2.5 and at 1.0 and requires
   both the oracle agreement and the difference.
2. **A test that compares a thing to itself is not a test.** The
   operand-swap test built the graph both ways and required different answers;
   reversing `route_operands` relabels both the validation and the
   interpretation consistently, so it survived. It is now checked against an
   independent oracle composed with each operand in the role its name says.

**A null result, reported because it is one:** the profile axis caught **no**
mutation that the single-profile sweeps did not already catch. What it did
produce is the fixture pressure — the first routed scale in the workspace that
is not 1 — which is how the `output_scale` gap became visible at all.

### Narrowings, reported rather than quietly dropped

- **The Laguna attention tower is not composed**, for the two gaps above. M2's
  exit gate does not need it; M7's family coverage does.
- **No importer.** Asymmetric INT4 at group 32, with zero points packed along
  the **output** axis while the codes are packed along the input axis, is M3's.
  Both facts are recorded in the bring-up record.
- **The fused/per-expert expert mapping is an open question**, not a decision:
  the pinned model declares fused tensors, the checkpoint stores per-expert
  ones, and the artifact's own `_checkpoint_conversion_mapping` covers only
  `e_score_correction_bias`.
- **The restricted-budget device case declares its amortisation threshold.** At
  Laguna's expert size the *default* 1 MiB/row sends every expert to the CPU —
  the same arithmetic AGENTS.md records for the Gemma artifact, asserted here by
  `the_default_amortisation_threshold_sends_every_laguna_expert_to_the_cpu`
  rather than left to be rediscovered. Measuring the real crossover is M6's.
- **Twelve experts, not 256.** The device case uses twelve of the artifact's 256
  so that the synthetic shard fits a test's disk and time. The *expert* is at
  full declared width; the *layer* is not.
- **The routed-block graph is stateless** and runs through
  `Interpreter::run_stateless`. Whole-versus-chunked parity for it is
  row-independence, which is checked.

### No owner gate was resolved

O1–O7 remain open. No numerical threshold, precision, context target or
compatibility surface changed. The support matrix gained no Laguna capability
row, because no Laguna capability exists.

### Remaining blockers, and the next bounded task

- **M2 is not closed.** Its exit still asks for byte/cost traces reconciled with
  the resource ledger across a **whole** working set rather than one layer, and
  for the real out-of-device-memory working set to execute.
- **No performance claim.** Both lanes are debug builds and there is no baseline
  on this machine.
- **Quality is O2** for both families.
- **Laguna's tower** needs two shared-operation tasks before any state schema,
  admission figure or partition for it exists.

## Independent review, and what it changed

An independent review of `78c493d` and `f2513fa` requested changes before
acceptance and raised **six** findings — one P1 and five P2. **All six were
reproduced, all six are fixed, none is disputed.** The reviewer also reported
independently re-running 908 host tests, 42/42 GPU gates, the eight
grouped-device integration tests, both sweep counts, `arch-check`, `spec-check`
and formatting, and said plainly what they had **not** re-run: the complete
device-feature suite, the clippy lanes and the mutation campaign.

### 1 (P1) — The router's coefficient narrowing was missing

`LagunaTopKRouter.forward` ends with
`routing_weights = routing_weights.to(hidden_states.dtype)`. The model dtype is
BF16; this implementation returned FP32 coefficients. On the logits `[0, 1]` the
source's are `[0.59375, 0.40625]` and the unnarrowed ones are
`[0.5938455, 0.4061545]`, and every combined row downstream carries the
difference.

`Route` gained a `coefficient: RouteCoefficient` parameter, applied as the
router's last step. Gemma 4 passes `Fp32` — `Gemma4TextRouter.forward` has no
such cast — so the two families disagree on it, which is what makes it a
parameter and not a constant.

**It says values, not storage.** A route table holds FP32 either way and
`output_role` still declares `F32`, because that role is what the byte trace
M2's exit gate reconciles against the ledger — the same reason the index
encoding there says `U32` rather than `U64`.

**Why a bitwise gate did not catch it, which is the part worth keeping.** The
FP64 transcription was written from the same pinned source by the same reader,
and it omitted the same cast. The two agreed, and their agreement proved only
that one reader made one omission twice. **A transcription is independent of the
implementation, not of the reader.** The fixture now checks the narrowed values
against `bf16_round` of an independently computed quotient, so agreeing for the
wrong reason is no longer available. The boundary is post-selection, so it
routed no row differently — which is exactly why the *selection is exact* gate
never saw it.

### 2 (P2) — The output-scale rounding justification was false

`combine_row` claimed that an exactly representable scale such as 2.5 "moves the
deviation nowhere". It does not. The reviewer's counterexample, reproduced
exactly: BF16 slots `[1, 0.0031433105]` with unit coefficients give
`bf16(sum · 2.5) = 2.515625` here against `bf16(bf16(sum) · 2.5) = 2.5` in the
reference.

Representability makes the *multiplication* exact and says nothing about the
operand, and the reference's operand has already been through a BF16 buffer. The
claim is corrected, the deviation is declared as a **second** one rather than as
the accumulation deviation restated, and
`an_exactly_representable_scale_still_moves_the_boundary` is the counterexample
as a test so it cannot quietly come back. The FP32 accumulation convention is
unchanged: reopening task 0019's accepted contract is not this task's to do, and
which of the two answers is closer to the released model is O2's question.

### 3 (P2) — The planner accepted a nonfinite output scale

`compile_experts` takes `&OpParams` directly, so `GraphBuilder`'s validation is
not on that path — and every test in `moxie-plan` supplies a node that never
passed through a graph. The reviewer reproduced `Ok(plan)` with `NaN`. The plan
then reserves its envelope and runs every expert before the reduction finally
refuses, which is a refusal after the work rather than before it. `shape_of`
validates it now, and zero and negative still plan, because this is a checkpoint
scalar and not a probability.

### 4 (P2) — A scaled reduction could overflow to infinity and return success

One BF16 slot of `2.0`, a unit coefficient and a **finite** scale of `f32::MAX`
produced BF16 infinity (`0x7f80`) while `combine_rows_bf16` returned `Ok(())`
and `GroupedRun::reduce` reported success. Both operands were finite, so no
earlier check could have caught it. The store checks the narrowed result and
returns `Error::Numerical`, and a representable large scale is still accepted —
the check is a bound, not a refusal of large scales.

### 5 (P2) — Invalid dimensions panicked before the checked arithmetic ran

`2 * moe_intermediate` was evaluated inline and *then* handed to `width`, the
checked helper. An intermediate of `1 << 63` passed `RoutedBlocks::reduced` and
panicked in `compose` in a debug build, through a public constructor that had
already accepted the configuration. The doubling is checked first and the
checked extent is threaded through composition instead of being recomputed.
Checked arithmetic that runs after the overflow is not checked arithmetic.

### 6 (P2) — The expert inventory understated the working set by 6.87 GB

The record multiplied the quantized per-layer cost by all 47 routed layers and
reported **65,645,629,440 B / 85.5%**, in a document that had already recorded
two paragraphs earlier that layers 46 and 47 keep BF16 experts. Measured from
the artifact's own headers: `45 × 1,396,715,520 + 2 × 4,831,838,208 =`
**72,515,874,816 B**, **94.41%**.

The correction is not a better number. `ArtifactGeometry` now carries
`bf16_expert_layers` and `stored_expert_bytes`, `expert_bytes_total` **sums over
the layers instead of multiplying**, and
`the_expert_inventory_matches_the_artifact_headers` reads the index and every
shard header and checks each layer against what `ARTIFACT` declares. The
bring-up record, the task record and AGENTS.md all carry the corrected figure.

The reviewer also corrected the nearby VRAM line: "the aggregate 72 GiB of the
three cards" was neither the nominal 64 GiB nor the **62.6 GiB** this repository
had already measured and recorded in its own hardware inventory. That is
AGENTS.md's "do not carry an unverified placement, affinity or bandwidth claim
forward" applied to a number that did not even need measuring, because the
measurement was three directories away.

### What the six have in common

Four of them — 1, 3, 4 and 5 — are the **same** shape as the four rounds against
task 0021: *a check that exists on one path and not on the neighbouring one*.
`GraphBuilder` validates the scale and `compile_experts` does not. The
oracle narrows the combination's output and the router does not narrow its
coefficients. `expert_row` is guarded against nonfinite results and the scaled
store was not. `width` is checked and its own argument was not.

The other two are a different and more uncomfortable shape: **a record and a
transcription that each contradicted a fact the same document already
contained.** The inventory multiplied past its own exception; the transcription
omitted the boundary its own source shows. Neither is a reasoning error that
more care at the keyboard would have caught, and neither was reachable by the
mutation batteries, which measure a test suite against mutations of the *code*
and say nothing about a number written in prose. The executable inventory is the
structural answer to the first; checking a transcription against an
independently computed quantity rather than against the implementation is the
answer to the second.

### Evidence

All six regressions were put through the substitution battery this repository
requires — remove the check, run only that regression, require it to fail:
**6 of 6 load-bearing**, 0 that asserted a symptom.

| Gate, after the corrections | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy | passed |
| `cargo test --workspace --locked --offline` | **915 passed, 0 failed** (908 before the corrections) |
| Device-feature workspace tests | **948 passed, 0 failed** (941 before the corrections) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped** |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures** |

Both mutation batteries were re-run against the corrected code, with the
review's four new checks added to them: **37 mutations, 0 survivors** — 17 of 18
caught by the planner sweep and 7 of 19 by the run sweep, the rest by named
tests. The reviewer said explicitly that they had not re-run the mutation
campaign; this is that re-run.

## Second independent review

The second round confirmed all six of the first round's corrections and found
**one new P1 — a regression the first round's own fix introduced.** It is
reproduced, fixed, and not disputed. The reviewer again said plainly what they
had not re-run: the complete workspace and the mutation batteries.

### The narrowing allocated, and an allocation failure here aborts the process

`narrow_coefficients` built a second coefficient vector with
`.iter().map(...).collect()`. Every other allocation on this path is fallible,
for a reason this module already recorded once: an allocation failure inside a
generation step must be a typed error the transaction can roll back, not a panic
that takes the rollback, the lease release and the next generation with it. The
reviewer injected a failure during narrowing only and got **SIGABRT**,
`memory allocation of 8 bytes failed`, on the BF16 arm while the FP32 arm
succeeded without allocating.

The fix is stronger than the `try_vec` the rest of the path uses: the function
takes the route **by value**, so it rounds in place and there is nothing to
allocate. `narrowing_a_routes_coefficients_requests_no_heap` counts allocator
calls and requires zero.

**This is `softmax`'s defect, in the same module, for the third time.** Task
0019's record says it plainly — "it used to use plain `collect()`, which was
harmless while routing was an unreached M0 fixture and became a process abort
the moment task 0019 put it under the interpreter". The correction for the first
review's P1 put a new function on that same path and did not carry the rule
across. A fix is a new caller, and a new caller on a path with a discipline has
to satisfy it.

### And the same defect once more, found by asking the review's question of the
### rest of the change

Applying the reviewer's own question to everything else task 0022 added to that
path found one more: `OpParams::route_operands` returned a `Vec`, and the
**interpreter calls it once per `Route` node per step**. Same class, same path,
same consequence, and the review did not find it because it was looking at the
function it had a reproduction for.

The operand list is bounded by construction — five operands, each pushed at most
once — so it is held inline in a `RouteOperands` value and allocates nothing.
`a_routers_operand_list_requests_no_heap` counts allocator calls for the widest
list, the narrowest, and a non-router operation.

Reported here rather than folded silently into the P1's fix, because **the count
matters**: the first round's fix introduced one defect of this class and the
same change already contained another. One reproduction found one of them.

### Evidence

Both regressions live in a new isolated executable with a counting global
allocator, `moxie-oracles/tests/route_allocation.rs`, which is the harness
`moxie-state` and `moxie-memory` already use for the same kind of claim. They
count allocator **calls** rather than injecting a failure, because "requires no
allocation" is the property and a call count states it directly.

Substitution: **3 of 3 load-bearing** — restoring the `collect()` fails the
narrowing test and the whole-router test, and a single spurious `Vec` inside
`route_operands` fails the operand test.

| Gate, after the second round | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy | passed |
| `cargo test --workspace --locked --offline` | **918 passed, 0 failed** (915 after the first round) |
| Device-feature workspace tests | **951 passed, 0 failed** (948 after the first round) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped** |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures** |

Both mutation batteries were re-run again, with a mutation added for each new
check: **39 mutations, 0 survivors** — 17 of 18 by the planner sweep, 7 of 21 by the run
sweep, the rest by named tests.

## Third independent review

The third round confirmed both production allocation fixes and found **one P2 in
the regressions themselves**. Reproduced, fixed, not disputed.

### The allocation regressions raced on a process-wide counter

`libtest` runs each test on its own thread in parallel, and all three tests read
the same `AtomicUsize`. Another test's allocations fall between a test's two
snapshots and implicate allocation-free code. Reproduced on unchanged `a14f8e5`:

```text
default execution   14 failures in 100 runs
serial execution     0 failures in  50 runs
```

The counter is thread-local now, so the isolation is a property of the harness
rather than of how it is invoked. `--test-threads=1` was **not** the fix and the
reviewer was right to rule it out: it would have removed the flake and left the
workspace gate everyone actually runs unreliable, which is strictly worse than a
flake that announces itself. Two details are load-bearing: the cell is
`const`-initialised, because a lazily initialised thread-local would allocate
*inside the allocator*; and it is read with `try_with`, so an allocation during
thread-local destruction counts as nothing instead of panicking.

After the fix: **0 failures in 200 default runs.**

### And the substitutions were repeated, because a single run could not tell the
### difference

The reviewer's second point is the sharper one: with a racy counter, a
substitution that "failed" might have failed from unrelated allocation noise
rather than from the mutation, so the *previous* round's 3-of-3 was not
trustworthy evidence even though it was the right answer.

The battery now builds each mutant once and runs its test **25 times**, and runs
the clean tree 25 times as well, requiring all 25 to fail and all 25 to pass
respectively:

```text
clean   narrowing-allocates-again                25/25 pass
clean   narrowing-allocates-again-whole-router   25/25 pass
clean   operand-list-allocates-again             25/25 pass
mutant  narrowing-allocates-again                25/25 fail
mutant  narrowing-allocates-again-whole-router   25/25 fail
mutant  operand-list-allocates-again             25/25 fail
```

**A substitution result from a nondeterministic test is not evidence, whichever
way it came out.** That belongs beside the rule this repository already has — a
regression is load-bearing when a substitution says so — because it is the
precondition that rule quietly assumes.

### Evidence

| Gate, after the third round | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy | passed |
| `cargo test --workspace --locked --offline` | **918 passed, 0 failed** (unchanged; the fix is to a test's harness, not its count) |
| Device-feature workspace tests | **951 passed, 0 failed** (unchanged) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped** |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures** |
| `route_allocation` under default parallelism | **200 runs, 0 failures** |

Both mutation batteries re-run: **39 mutations, 0 survivors** — 17 of 18 by the
planner sweep, 7 of 21 by the run sweep, the rest by named tests.

### Outcome

The third round reported **no new blocking findings** and recommended acceptance
within this task's documented routed-block scope, having independently verified
200 default-parallelism runs with zero failures, 25/25 detection for each of the
three allocation checks against deliberately added allocations, 301 affected
host tests, and the architecture, specification and formatting gates. It stated
what it had not re-run for a harness-only correction: the 39-mutation campaign
and the GPU gates.

**That recommendation is not acceptance, and it is not M2.** The reviewer said
so in the same breath and it is worth repeating here: it accepts this task's
scope, not Laguna model support and not M2's exit. The whole-working-set byte
and cost trace reconciled with the resource ledger is still outstanding, and it
is task 0023.

### What the three rounds were each about, because the progression is the point

Round one found defects in the work: a missing BF16 boundary, a false
justification, two validation gaps, an overflow, and an inventory that
contradicted its own document. Round two found a defect in **round one's fix**.
Round three found that the **evidence for round two's fix was not evidence**,
because the test it rested on was nondeterministic.

Each round moved further from the code and closer to the claims made about it,
and the last one is the only one that changed a rule this repository states
rather than a line it contains. A task whose review rounds converge on its
evidence rather than on its behaviour is a task whose behaviour is probably
right; that is worth knowing, and it is not something any single round could
have reported.
