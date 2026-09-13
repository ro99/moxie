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
| One sparse layer's 256 experts | **1,396,715,520 B**; all 47 sparse layers **65,645,629,440 B**, 85.5% of the artifact | the same |

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

*Empty until the work is done. Passed, failed and skipped stay separate, and a
number that was not measured is reported as not measured.*
