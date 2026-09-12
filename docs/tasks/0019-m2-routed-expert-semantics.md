# Task 0019 — M2 shared routed-expert semantics

Status: **implemented at the commit this record accompanies; awaiting
independent review and owner acceptance.** Contract written and committed at
`b63d931` before implementation, per the working rule that produced tasks
0013–0018. See [Result](#result-filled-after-work).

## Identity and authority

- Task ID / milestone / owner: 0019 / **M2 item 1, mathematics only** /
  implementation agent; acceptance belongs to the owner. **This task does not
  close M2 and establishes no residency capability.**
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `8dc9e77`
  (task 0017 acceptance record). Working tree clean at authoring; no initial
  dirty paths.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and
  `tests/p2p/` are preserved and are not source evidence.
- Assigned by [the M1 closure handover](../handovers/2026-09-12-m1-closure-to-m2.md),
  which names task 0019 as "M2's first bounded slice, in roadmap order: shared
  routing semantics and expert residency against the designated BF16 MoE".
- Required documents read: AGENTS.md, README, reference documents 01–09, the
  owner-gate register, `docs/README.md`, the TASK/ADR/HANDOVER/MODEL-BRINGUP
  templates, [ADR 0009](../decisions/adr/0009-engram-conditional-memory.md),
  [ADR 0012](../decisions/adr/0012-explicit-family-operation-parameters.md),
  [ADR 0013](../decisions/adr/0013-one-model-crate-with-family-modules.md), the
  [gemma4 bring-up record](../models/gemma4.md) and tasks 0016/0017/0018.
  Normative for this task: document 02's "Semantic operations" (the
  `Route -> Dispatch -> ExpertMlp -> Combine` row and the `RouteSpec` sketch),
  document 03's "Shared weight-residency lifecycle" and MoE admission
  paragraph, document 06 M2 items 1 and 4, and document 09 §§ B and D.
- Owner gates: **O1–O7 remain open.** None blocks a host-only, synthetic-weight,
  BF16 slice that reads metadata and copies nothing. Stop before any checkpoint
  execution, import, conversion, quality claim or device kernel.

### Why this task is narrower than the handover's sentence

The handover names three owners for task 0019 — `moxie-graph` for routing,
`moxie-engine` for row dispatch and combination, `moxie-memory` for the
residency authority. That is M2 items 1 **and** 2, and `docs/tasks/README.md`
requires one bounded assignment to produce one primitive. This contract takes
**item 1's mathematics** and names **task 0020** for item 2's residency
authority, for a reason that is an ordering fact rather than a convenience: the
residency authority is admitted against the union of experts a row batch
demands, and that union is a *result* of the routing equation. Writing the
authority first would mean reserving bytes for a demand set defined by code that
does not exist yet.

Nothing here weakens the handover's stop conditions; they are restated below and
apply to task 0020 unchanged.

## The designated artifact, inspected

`/fast/models/google/gemma-4-26B-A4B-it`, read-only, **nothing copied,
converted, deleted or executed**. Verified 2026-09-12:

| Property | Observed |
|---|---|
| Revision (HF download metadata) | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| `config.json` sha256 | `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3` |
| `model.safetensors.index.json` sha256 | `907826a6e46ff454272bd6db1fee629d5531a2303be22986d825a0871d7dc7a7` |
| Completeness | **complete.** Both shards: `8 + header + payload_end` equals the file size exactly, and the two payload ends sum to the index's `total_size` of 51,611,872,412 B |
| Declared license | `apache-2.0`, link `https://ai.google.dev/gemma/docs/gemma_4_license`; base model `google/gemma-4-26B-A4B` |
| Precision | **BF16 throughout**; there is no `quantization_config` |
| Reference implementation | `transformers 5.5.0.dev0`, `model_type` `gemma4` / `gemma4_text` |

Text geometry from `config.json.text_config`: hidden 2,816; 30 layers; 16 query
heads; sliding 8 x 256, global 2 x 512; `attention_k_eq_v` true; `layer_types`
25 sliding and 5 full at indices 5, 11, 17, 23, 29 — the same
`(layer + 1) % 6 == 0` predicate task 0016 pinned; `sliding_window` 1,024;
`rms_norm_eps` 1e-6; `hidden_activation` `gelu_pytorch_tanh`;
`final_logit_softcapping` 30.0; `tie_word_embeddings` true; vocab 262,144;
`max_position_embeddings` 262,144 (**not** an admissible context, R19);
sliding RoPE theta 10,000 `default`, global theta 1,000,000 `proportional` with
`partial_rotary_factor` 0.25.

New for this task, and absent from the 31B: `enable_moe_block` true,
`num_experts` 128, `top_k_experts` 8, `moe_intermediate_size` 704, alongside the
dense `intermediate_size` 2,112.

Per-layer tensors, read from the safetensors headers (BF16, all 30 layers):

| Name | Shape | Role |
|---|---|---|
| `router.scale` | `[2816]` | per-channel gain on the router's normalized input |
| `router.proj.weight` | `[128, 2816]` | router projection, `E x H`, no bias |
| `router.per_expert_scale` | `[128]` | per-expert coefficient scale |
| `experts.gate_up_proj` | `[128, 1408, 2816]` | **all 128 experts fused**, `E x 2I x H` |
| `experts.down_proj` | `[128, 2816, 704]` | **all 128 experts fused**, `E x H x I` |
| `mlp.{gate,up}_proj.weight` | `[2112, 2816]` | the dense shared expert |
| `mlp.down_proj.weight` | `[2816, 2112]` | the dense shared expert |
| `pre_feedforward_layernorm_2.weight` | `[2816]` | norm on the routed branch's input |
| `post_feedforward_layernorm_1.weight` | `[2816]` | norm on the dense branch's output |
| `post_feedforward_layernorm_2.weight` | `[2816]` | norm on the routed branch's output |

The census over all 30 layers is exact: 30 of each of the above, 30
`layer_scalar`, and **25** `self_attn.v_proj.weight` — the five global layers
carry none, which is the serialized form of `attention_k_eq_v`, exactly as the
31B's ten global layers do. Two facts the handover named are confirmed here and
a third is added: experts are fused per layer; every layer carries a dense `mlp`
beside its routed experts; and **the MoE block adds three norms per layer**, not
one, so the routed and dense branches each have their own input and output
normalization.

### The reference the equations come from

The frozen legacy tree has **no Gemma 4 MoE**: its `gemma4` adapter is the dense
31B path, and `grep` for `per_expert_scale`, `router.scale`, `enable_moe_block`
and `gate_up_proj` across it returns only unrelated GLM-5.2 routing text and a
timing counter. So the pinned reference for this task is the released
`transformers` implementation, read as source and **never executed**:

- `transformers/models/gemma4/modeling_gemma4.py`, class `Gemma4TextRouter`,
  `Gemma4TextExperts` and `Gemma4TextDecoderLayer.forward`.
- Three independent copies were compared: `transformers` 5.5.3 under
  `~/miniconda3`, another 5.5.3 under `~/Developer/heretic/.venv`, and 5.15
  under the legacy tree's `experiments/tools/`. The router and expert
  mathematics are identical in all three. **One difference exists and is
  recorded rather than averaged:** 5.15 computes the router softmax with
  `dtype=torch.float32` and comments "fp32 for numerical stability"; 5.5.3
  computes it in the input dtype. The artifact declares 5.5.0.dev0.
- `transformers/modeling_rope_utils.py:187` `_compute_proportional_rope_parameters`
  confirms task 0016's global-layer reading — `rope_angles = partial · head_dim // 2`
  with the exponent denominator the **full** head dimension, and a zero-padded
  tail that is an identity rotation. No new RoPE gap.

## Bounded deliverable

**One concrete outcome:** the shared operation catalogue gains routed-expert
mathematics — router score transformation with its tie rule, selection,
renormalization and per-expert coefficient scale; grouped compute over a fused
expert tensor; and a combination whose reduction order is a stated parameter —
each with an independent FP64 transcription and a registered oracle. Two
unrelated graphs consume them: a reduced Gemma-4-MoE-like text graph whose
routed branch sits beside a dense shared expert, and a synthetic MoE with a
different expert count, top-k, activation, shapes and route distribution.

**Sole owning shared component:** `moxie-graph` owns the operation parameters;
`moxie-oracles` owns every equation and its FP64 transcription; `moxie-interp`
dispatches to them. `moxie-models::gemma4` gains metadata and graph composition
**only**.

**Allowed production files:**

- `crates/moxie-graph/src/graph.rs` — `OpParams::Route`, `OpParams::ExpertMlp`,
  `OpParams::Combine`; the `ValueRole` a route table carries; partition rules
  and state effects for the three.
- `crates/moxie-graph/src/lib.rs` — oracle registration for the three.
- `crates/moxie-oracles/src/route.rs` — the router transform, the fused-expert
  grouped compute, and the ordered combination. The existing `route_row`,
  `dispatch`, `required_experts` and `combine` keep their current behaviour and
  their current callers.
- `crates/moxie-interp/src/tensor.rs`, `crates/moxie-interp/src/lib.rs` — the
  route-table value and the three evaluation arms.
- `crates/moxie-models/src/gemma4.rs` — MoE configuration fields, the
  artifact's declared MoE geometry as a constant, the routed-branch tensor
  roles, and the layer composition.
- `crates/moxie-cli` — the second synthetic consumer and the disclosure surface.
- Their tests and manifests, the architecture allowlist and negative fixtures,
  `Cargo.lock`, and tracked task/evidence/model/handover records.

**Explicit non-goals and forbidden shortcuts.** No residency authority, no
expert cache, no chunk identity, no demand or prefetch class, no eviction, no
placement, no reservation — **all of that is task 0020**, and a simulated
placement presented as a reservation is a failed task, not a partial one. No CPU
expert fallback and no grouped GPU candidate plan (M2 item 3). No CUDA and no
device kernel; the selected BF16 device chain must **refuse** these operations
as `UnsupportedKernel` rather than acquire a routed path. No checkpoint import,
no dequantization, no conversion, no download, no write of any kind under
`/models` or `/fast/models`. No vision and no audio tower — M11. No expert
parallelism or partition decision — M5, and the partition rule fails closed
until then. No second weight-residency owner anywhere. No unbounded queue and no
demand path at all, because there is no demand path in this task. **No
synthetic graph may be described as model support**, and a routed synthetic
graph is not Gemma 4 MoE support: the reduction list stays and the CLI keeps
printing it first. No quality claim — that is O2's, and it needs paired output
against the released model.

**Existing consumers and second-shape proof.** Every current consumer of
`OpParams` must keep passing unchanged: the dense reduced Gemma graph, the task
0015 synthetic graph, the paged and sampling gates, and the selected device
chain's refusals. The new operations get two independent consumers with
deliberately opposite parameters:

| | Gemma-4-MoE-like reduced graph | Second synthetic MoE |
|---|---|---|
| Experts / top-k | 128-shaped rule at reduced count, top-k > 1 | a different count, a different top-k, including `top_k == experts` |
| Activation | GeGLU (`gelu_pytorch_tanh`) | SwiGLU |
| Shared expert | present, dense, combined outside routing | absent |
| Per-expert coefficient scale | present | absent (unit) |
| Combination order | ascending expert id | route order (descending score) |
| Route distribution | overlapping across rows | disjoint, plus an all-rows-to-one-expert case |

## Contract before implementation

### Equations

Written for one row. `H` is hidden width, `E` expert count, `K` top-k, `I` the
routed intermediate width. `s` is `router.scale`, `W_r` is
`router.proj.weight`, `c` is `router.per_expert_scale`, `GU[e]` and `D[e]` are
expert `e`'s slices of the fused tensors.

**Router** (`Gemma4TextRouter.forward`):

```text
n  = x · (mean(x²) + ε)^(−1/2)                  scale-free RMS norm, ε = rms_norm_eps
t  = n ⊙ s · H^(−1/2)
z  = W_r · t                                     [E], no bias
p  = softmax(z)                                  over all E experts
(idx, w) = top_k(p, K)                           descending p; ties → lower expert id
w  = w / Σ w                                     renormalised over the selected K
w  = w ⊙ c[idx]                                  per-expert scale, applied last
```

Two properties are load-bearing and must be tested as such. `H^(−1/2)` is
`hidden_size**-0.5` in the reference and is **not** a norm epsilon or an
attention scale; it multiplies the router input only. And the final coefficients
**do not sum to one** — the per-expert scale is applied *after* renormalization
and is never renormalized away. A combine that renormalizes again would erase a
trained parameter.

**Tie rule.** `torch.topk`'s behaviour on equal values is not a contract. The
shared rule is the one `moxie-oracles::route` already pins for the same reason
document 05 pins it for sampler ties: **the lower expert id wins**, so two ranks
cannot route one row to two different experts.

**Expert** (`Gemma4TextExperts.forward`), for each selected `e`:

```text
gu   = GU[e] · x₂                                [2I], GU[e] is [2I, H]
gate = gu[0 .. I]                                the first block
up   = gu[I .. 2I]                               the second block
h    = act(gate) ⊙ up                            act = gelu_tanh for this family
y_e  = D[e] · h                                  [H], D[e] is [H, I]
```

The gate/up split is `chunk(2, dim=-1)` on the projection output, so the fused
tensor's output axis is the **gate block followed by the up block**, not
interleaved pairs. That is a logical-order statement read from the pinned
reference; physical disk layout is the importer's problem (document 03) and this
task does not read tensor bytes.

**Combine**:

```text
y = Σ_j  w[j] · y_{idx[j]}                       over the K selected slots
```

with the summation order an explicit parameter. The pinned reference iterates
`expert_hit`, which is `nonzero()` over an expert-major mask, so its reduction
order is **ascending expert id** — not the row's selection order. Floating-point
addition is not associative, so this is a semantic parameter rather than a
scheduling detail, and both orders are implemented and tested.

**Layer composition** (`Gemma4TextDecoderLayer.forward` with
`enable_moe_block`), where `r` is the post-attention residual stream:

```text
m  = mlp( pre_ffn_norm(r) )                      dense shared expert, width 2112
h₁ = post_ffn_norm_1(m)
route = Route(r)                                 ← r, NOT pre_ffn_norm(r)
h₂ = post_ffn_norm_2( Combine(route, ExpertMlp(pre_ffn_norm_2(r), route)) )
u  = h₁ + h₂
r' = ( r + post_ffn_norm(u) ) · layer_scalar
```

The router consumes the **un-normalized** residual while the experts consume
`pre_feedforward_layernorm_2` of it. Feeding the router the normalized stream
would be an ordinary-looking graph that routes every row on the wrong vector.
The dense branch is a shared expert **outside** routing: its output is added
after both branches have been normalized, and it takes no routing coefficient.
`layer_scalar` still applies to the whole layer output, as task 0016 pinned.

### Shapes, precision, accumulation and rounding

- `Route` inputs `[rows, H]` activation, `[H]` weight, `[E, H]` weight, `[E]`
  weight; output a route table of `[rows, K]`, carrying **integer** expert ids
  and **activation** coefficients as two role-separated fields. Document 02:
  indices "are not quantized weights or floating activations", and a route table
  that typed its ids as floats would be exactly that error.
- `ExpertMlp` inputs `[rows, H]`, the route table, `[E, 2I, H]`, `[E, H, I]`;
  output `[rows · K, H]`, slot-major, slot `j` of row `r` at index `r · K + j`.
  Emitting per-slot outputs rather than a combined row is what keeps `Combine` a
  separately testable operation, which document 09 §D requires of anything that
  could have been fused.
- `Combine` inputs the route table and `[rows · K, H]`; output `[rows, H]`.
- Reductions are FP32, sequential ascending, as everywhere else in this
  crate. **Every BF16 boundary the reference has is part of the equation**, not
  a storage detail left to the node output:

  ```text
  router:  t  = bf16(bf16(bf16(x / rms(x)) ⊙ s) · H^(−1/2))
           z  = bf16(Σ t·W_r)            a BF16 linear's output
           p  = softmax_fp32(z)          5.15's stated convention
  expert:  gu = bf16(Σ x·GU[e])
           h  = bf16( act(gate) ⊙ up )
           y  = Σ h·D[e]                 rounded by the node output
  ```

  **This corrects the contract's first draft**, which kept the router and expert
  chains in FP32 and declared only the combination's deviation. An independent
  review showed that was not merely imprecise: a probe over 200,000 random BF16
  rows found the FP32 chain selecting **different experts** from the
  boundary-preserving one on about one row in 270, with distinct logits and no
  tie involved. Selection decides which expert weights have to be resident, so
  the boundaries are a residency contract as much as a numerical one, and
  `the_routers_bf16_boundaries_decide_which_experts_are_selected` pins a fixture
  where dropping them changes the answer.

  The **gate transform's own** internal boundary is each activation's accepted
  contract rather than something this task imposes: `geglu_row` rounds
  `gelu_tanh(gate)` because `gemma4_ops.cpp:70` does, and `swiglu_row`
  evaluates in FP64 and rounds once because task 0003's contract says so after a
  review found an intermediate underflow producing a 100% error. Overriding
  either from here would rewrite an accepted contract on no source.
- **One declared deviation from the reference remains, and its size is stated
  correctly.** The reference narrows each expert's weighted contribution to BF16
  *before* accumulating (`current_hidden_states.to(dtype)` then `index_add_`).
  This reference accumulates the `K` terms in FP32 and rounds once at the node
  boundary. That difference is **not** bounded by `metric::bound`, which is built
  from FP32 unit roundoff — the contract's first draft claimed it was, and the
  review's counterexample disproves it: coefficients `[1,1,1]` over outputs
  `[256, 1, -256]` give 1 in FP32 and **0** under BF16 accumulation, against a
  claimed bound of about `9.2e-5`. The honest statement is one BF16 ulp of the
  running sum per term, which under cancellation is of the order of the largest
  term rather than of the result.
  `combine_reference_deviation_is_not_covered_by_the_fp32_bound` is that
  counterexample as a test. The reduction *order* is pinned regardless, because
  order changes the result at any precision. Whether the deviation matters to
  output quality is **O2**.
- No new tolerance is invented. Selection is checked **exactly** — ids and their
  order are integers and must match the FP64 transcription bit for bit.
  Coefficients and expert outputs are checked against `moxie-oracles::metric`
  bounds assembled from the counted rounding steps of each equation, in the same
  style as the existing attention bound. Every such bound is stated against
  **`Σ|terms|`**, never against `|y|`: `linear::linear_row_scale` gives the
  reason in its own words, and the review found this task's expert-output test
  scaling by `max(|y|, 1)` — which would have accepted a cancelling fixture that
  is wrong by its whole magnitude.
  `an_experts_down_projection_is_bounded_when_its_terms_cancel` is that case.

### Partition and hardware capabilities

- `Route` is `PartitionRule::Replicated`. Every rank must compute the same route
  from the same row; a sharded router that reduced across ranks could produce
  different selections under different reduction orders, and two ranks that
  disagree about which expert a row needs is a residency bug as well as a
  numerical one.
- `ExpertMlp` and `Combine` are `PartitionRule::NotDetermined`, failing closed.
  Expert partitioning is M5 and this task does not pre-empt it.
- No device capability is claimed. `moxie-plan`'s selected BF16 chain refuses
  every operation outside its qualified package, and these three stay outside
  it; a test asserts the refusal rather than assuming the catch-all covers it.

### Peak memory, transfer dependencies, lease lifetime

None are introduced. This task allocates host tensors inside the existing
interpreter and its existing counted-allocator envelope; there is no device
buffer, no upload, no event and no lease. The routed intermediate values
(`[rows · K, H]`) are the one new peak term and are charged through the existing
admission path like any other activation.

The arithmetic that **task 0020** will need is recorded now, from the artifact's
own declared geometry, as a projection and not a measurement:

| Quantity | Bytes |
|---|---|
| One expert, one layer (`2·704·2816·2 + 2816·704·2`) | 11,894,784 (11.34 MiB) |
| All 128 experts, one layer | 1,522,532,352 (1.42 GiB) |
| All experts, 30 layers | 45,675,970,560 — **88.5%** of the artifact's 51.6 GB |
| Dense shared expert, 30 layers | 1,070,530,560 |
| Top-k 8 over 30 layers, one token, no reuse | 2,854,748,160 |

Against 24 GiB (25.77 GB) on the largest single device and 63.9 GiB (68.6 GB)
aggregate: the artifact fits aggregate VRAM and **no single device**, and M2
item 4's intentionally restricted budget makes it oversized by construction.
Document 03's warning applies directly to the last row — the union of experts a
row batch demands is what must be resident, and multiplying active experts by
batch rows overstates it whenever routes overlap.

### Cancellation, failure and rollback

The three operations are pure: they touch no sequence state, so
`Op::touches_state` stays false for all three and a cancelled step has nothing
to roll back. The interpreter's existing per-operation cancellation boundary
covers them.

**Every allocation on the routing path is fallible.** That is a requirement, not
a style note: an allocation failure inside a generation step must become a typed
error the transaction can roll back. The contract's first implementation missed
it — `softmax` used plain `collect()` and the selection used a stable sort,
whose scratch buffer allocates infallibly — because routing was an unreached M0
fixture when that code was written and became an execution path here. Both are
fixed, the sort is now `sort_unstable_by` with the tie rule written into the
comparator, and the routed step has failure-injection coverage asserting a typed
terminal event, a released charge and a successful retry. Typed failures, never a silent clamp: an empty expert set, a
`top_k` of zero or greater than `E`, a non-finite router logit or scale, a
selected mass that is zero or non-finite, an expert id outside `0..E`, a fused
tensor whose extent does not equal `E · 2I · H` or `E · H · I`, and a slot
tensor whose row count is not `rows · K` are each an `InvalidRequest`,
`InvalidArtifact` or `Numerical` error at the point of detection.

### Independent oracle and predeclared metrics

Every new equation gets an FP64 transcription in the test module, written from
the pinned reference rather than from the implementation, and each operation is
registered in the `OracleRegistry` so that an unregistered operation still
cannot be built into a graph. Reported per document 07: max, RMS and p99 error,
not a maximum alone.

Predeclared, before any code:

1. Selected expert ids and their order: **exact** agreement, including every tie
   fixture.
2. Router coefficients: within `metric::bound` for the counted chain
   (norm over `H`, the two scalings, the `H`-term projection, the softmax, the
   `K`-term renormalization, the per-expert multiply).
3. Expert outputs: within `metric::bound` for the `H`-term and `I`-term
   reductions plus the activation's own declared bound.
4. Combination: within `metric::bound(K, Σ|terms|)`, with the summation order
   fixed by the parameter under test.
5. The load-bearing regression: substituting the conventional value for each new
   parameter — a unit `router.scale`, a unit `per_expert_scale`, `H^(−1/2)`
   replaced by 1, the router fed the normalized stream instead of the residual,
   the shared expert dropped, the gate/up blocks swapped, and the combination
   order flipped — **must change the graph's logits**. A parameter that can be
   swapped without changing the result is a failed acceptance.

### Application compatibility and sampler implications

None. No protocol, template, tokenizer, sampler or CLI contract changes; the CLI
gains a routed fixture and keeps its existing disclosure. Logits still come from
the same tied vocabulary projection with the same softcap.

## Acceptance

- Per-operation oracle agreement against the FP64 transcriptions at two distinct
  shapes each, including the boundary cases: `top_k == 1`, `top_k == E`, an
  exact tie spanning the `k` boundary, an expert that receives no rows, an
  expert that receives every row, nonuniform row counts, a repeated route within
  one batch, and a route whose union is strictly smaller than `rows · K`.
- The load-bearing substitution suite above, all seven cases.
- Generation through the shared service and the CLI for the routed reduced
  graph and the second synthetic MoE: one and multiple tokens, whole versus
  chunked prefill agreeing bit for bit, greedy and temperature, fixed-seed
  service/CLI agreement, cancellation then a second generation, exactly one
  terminal event.
- Allocation: counting-allocator peak below the recomputed admitted envelope
  with the routed intermediate included, repeated generations retain no growth,
  every tier charge returns to baseline.
- The device-path refusal: the selected BF16 chain returns `UnsupportedKernel`
  for `Route`, `ExpertMlp` and `Combine`, asserted explicitly.
- Architecture: `xtask arch-check` passes with `moxie-models` still depending on
  nothing but `moxie-types`, `moxie-graph` and `moxie-model-api`, plus a new
  negative fixture proving a model crate that reaches for a residency or cache
  symbol is rejected.
- All prior gates still pass unchanged: `G-INTERP-BF16`, `G-PAGED-HOST`,
  `G-PAGED-ALLOCATION`, `G-KV-RETENTION`, `G-WINDOW-ALLOCATION`,
  `G-SAMPLING-HOST`, `G-SAMPLING-ALLOCATION`, `G-GENERATION-HOST`,
  `G-GENERATION-ALLOC`, `G-GEMMA-REDUCED`, `G-HOST-ARCH`.
- Support matrix: one new gate `G-MOE-ROUTING-HOST`, whose limit column says in
  its own words that it is synthetic, host-only, unrouted to any real weight,
  and **not Gemma 4 MoE support**.
- Documentation: this contract's Result section filled in with passed, failed
  and skipped kept separate; the [gemma4 bring-up record](../models/gemma4.md)
  extended with the MoE variant's inventory, tensor-role mapping and blockers;
  a handover naming task 0020.

**Exact condition requiring owner direction or task rejection.** If the routing
mathematics cannot be expressed through shared operations without either a
model-owned execution path or a second residency owner, stop and report rather
than widening `moxie-models`. If the 5.5.3-versus-5.15 softmax-dtype difference
turns out to change *selection* rather than only coefficient precision on any
fixture, stop: that is a reference ambiguity about the artifact's own behaviour
and it belongs to the owner with O2, not to a default chosen here.

## Result, filled after work

Status: **implemented and corrected after independent review; awaiting
re-review and owner acceptance.** Contract committed at `b63d931` before
implementation `2608b5e`; the review corrections follow it.

### Independent review, and what it changed

A review of `b63d931` / `2608b5e` reported five issues. **All five were
reproduced and all five are fixed**; none was disputed. They are listed here
rather than quietly folded in, because three of them are corrections to claims
this record previously made.

| # | Finding | Verified how | Fix |
|---|---|---|---|
| 1 | The routing path allocated infallibly (`softmax`'s `collect()`, and the stable sort's scratch), so an allocation failure inside a routed step **aborted the process** instead of returning a rollback-able error | the reviewer injected a 12-byte failure and got exit 134; the code confirms it | `softmax` and the selection are fallibly allocated; the sort is `sort_unstable_by` with the tie rule in the comparator, which allocates nothing and states the rule instead of relying on stability. Failure injection added to `G-GENERATION-ALLOC` on the routed shape: typed terminal event, released charge, successful retry |
| 2 | The router and expert chains dropped **BF16 boundaries the reference has**, and the FP64 transcription dropped them too, so the agreement proved nothing | reproduced independently: over 200,000 random BF16 rows at hidden 8 / 4 experts, the FP32 chain selects **different experts** from the boundary-preserving one on roughly one row in 270, with all logits distinct | every boundary implemented and transcribed; `the_routers_bf16_boundaries_decide_which_experts_are_selected` pins a fixture where the two chains select `[3, 1]` and `[1, 3]` |
| 3 | The stated bound on the remaining combination deviation was **false** — `metric::bound` is FP32 unit roundoff and cannot cover BF16 accumulation | the reviewer's counterexample recomputed: `[1,1,1]` over `[256, 1, -256]` gives 1 here and 0 under BF16 accumulation, against a claimed bound of `9.2e-5` | the claim is replaced with the correct statement, and `combine_reference_deviation_is_not_covered_by_the_fp32_bound` makes the counterexample executable |
| 4 | The expert-output test scaled its bound by `max(\|y\|, 1)`, which is invalid under cancellation and contradicts `linear_row_scale`'s own documented rule | the shape of the counterexample checks out; `linear.rs` says so in its own words | the bound is now stated against `Σ\|terms\|` propagated through both projections, and `an_experts_down_projection_is_bounded_when_its_terms_cancel` adds the cancelling fixture the patterned weights never produced |
| 5 | `ValueRole::Route` declared `IndexEncoding::U64` while `RouteTable` stores `u32`, so the resource plan charged 12 bytes an entry for 8 | read directly | `IndexEncoding::U32` added with its own byte width; `a_routes_declared_index_encoding_matches_what_it_stores` asserts the declared encoding equals the stored width |

The review also corrected this task's **milestone attribution**: grouped GPU
expert execution is **M2 item 3**, not M5/M6. Fixed here, in the handover and in
the support matrix.

Findings 2 and 3 are the substantive ones. Both were places where this record
**claimed more than it had**: "one declared deviation" was three, and the one
that was declared was declared with a bound that does not hold. The lesson is
narrower than "add boundaries" — a transcription that omits the same boundary as
the implementation agrees with it perfectly and proves nothing, so a boundary
needs a fixture that *fails* when it is dropped, which is what the two new
regressions are.

### Changed shared owners and consumers

| Owner | Change |
|---|---|
| `moxie-graph` | `OpParams::Route`, `OpParams::ExpertMlp`, `OpParams::Combine`; the `ExpertActivation` and `CombineOrder` descriptors; `ValueRole::Route { index, coefficient }` and `OpParams::output_role`, so a node's output role is no longer assumed to be an activation |
| `moxie-oracles` | `route.rs` gains `RouterSpec`, `ExpertSpec`, `router_input_row`, `router_probabilities`, `select_top_k`, `apply_per_expert_scale`, `router_route_row`, `expert_row`, `combine_order`, `combine_row` and `combine_scale`; `Op::ExpertMlp` joins the registry. The M0 `route_row`, `dispatch`, `required_experts` and `combine` keep their behaviour, with `select_top_k` factored out so there is one tie rule |
| `moxie-interp` | `Value::Route(RouteTable)` and the three evaluation arms; both binding validators refuse an externally supplied route |
| `moxie-plan` | charges a route entry as its id **plus** its coefficient; refuses a route-role external input; the selected BF16 chain refuses routed operands and operations |
| `moxie-models::gemma4` | `MoeGeometry`, `router_input_scale`, `ARTIFACT_A4B`, the eight routed tensor roles and `route_layer`. Metadata and graph composition only; the crate's dependencies are unchanged |
| `moxie-cli` | `gemma::Shape::C` (routed Gemma-like) and `fixture::build_routed` (the second synthetic MoE), both reachable from the diagnostic surface |
| `xtask` | negative fixture `models-crate-reaches-memory` |

### Commands and results

All re-run after the review corrections, on 2026-09-12. The counts include the
five regressions the review produced.

| Gate | Command | Result |
|---|---|---|
| Format | `cargo fmt --all -- --check` | **passed**, empty diff |
| Clippy, host lane | `cargo clippy --workspace --all-targets --locked -- -D warnings` | **passed**, no warnings |
| Host tests | `cargo test --workspace --locked --offline` | **720 passed + 9 doctests, 0 failed** |
| Device-feature tests | the same with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda` | **736 passed + 12 doctests, 0 failed** |
| Real GPU | `cargo xtask-cuda test-gpu` | **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `G-MOE-ROUTING-HOST` | the four commands in the support matrix | **passed**: 167 `moxie-oracles`, 31 `moxie-interp::reference_graphs`, 14 `moxie-models`, 21 `moxie-cli::gemma` |
| `G-GENERATION-ALLOC` | `cargo test -p moxie-cli --test allocation -- --nocapture` | **passed**, now including the routed shape |

Logs are under `results/task0019/` (untracked, per the placement contract).

**Failed: none. Skipped or unmeasured, kept separate:**

- **`cargo xtask arch-check` reports 4 failures, all pre-existing and none from
  this task.** They are the untracked review crate
  `results/task0014-independent-review-2026-09-11/probes/`, which `arch-check`
  discovers because it walks the tree and `results/` is gitignored rather than
  excluded from the walk. The identical four lines are already recorded in
  `results/task0015/arch-local.log`, so they predate this work. Every rule and
  every fixture passes, including the new one:
  `PASS fixture models-crate-reaches-memory rejected: forbidden dependency ::
  [dependencies] moxie-models -> moxie-memory`. **Nothing under `results/` was
  deleted to make the gate green** — that is a review artifact and removing it
  is not this task's call.
- **No performance measurement.** None is claimed, and none is due: this task
  adds no device path.
- **No quality measurement.** O2, and it needs the released model.

### Measured effect and uncertainty

- Admitted reserve and observed peak on the routed geometry, from
  `G-GENERATION-ALLOC`'s own output: prompt 37 / chunk 13 peak 786,198 B against
  13,277,846 B admitted; prompt 251 / chunk 65 peak 2,056,589 B against
  49,188,646 B admitted. The dense shape A at the same prompts reserves
  11,146,262 B and 43,662,502 B, so the routed intermediate is visible in the
  reserve and the peak stays inside it.
- Numerical agreement is stated against counted `metric::bound` chains, never an
  invented tolerance. Selection agrees with the FP64 transcription **exactly**;
  coefficients, expert outputs and combinations agree within their bounds, with
  max, RMS and p99 reported.
- **One uncertainty is structural and cannot be closed here.** The pinned
  reference narrows each expert's weighted contribution to BF16 before
  accumulating; this interpreter accumulates in FP32 and rounds once at the node
  boundary. The reduction *order* is pinned either way. Which is closer to the
  released model is O2's question.
- **A second is a version difference, not a choice.** `transformers` 5.15
  computes the router softmax in FP32; 5.5.3 uses the input dtype; the artifact
  declares 5.5.0.dev0. FP32 is pinned. On every fixture here the difference is
  coefficient precision, not selection — but "every fixture here" is synthetic,
  and a real checkpoint could sit on a near-tie.

### Deleted or replaced paths

None. Nothing was superseded: the M0 routing fixtures keep their behaviour and
their callers, and `select_top_k` was factored **out of** `route_row` rather
than duplicating its tie rule.

### Remaining blockers and the next bounded task

1. **No residency capability exists.** The designated artifact is 51.6 GB
   against a 24 GiB largest device, and nothing here makes it run. **Task 0020**
   is M2 item 2: connect the real residency authority to storage reads, host
   cache, upload readiness, leases, eviction, error recovery and demand/prefetch
   classes, with expert chunk identity `(artifact, tensor/expert, logical range,
   format version)` and an enforceable reservation replacing any simulated
   placement. Its stop conditions are this contract's, unchanged — in particular
   a second weight-residency owner, a cache class in an adapter, a deadlockable
   demand path, an unbounded queue and any bulk write.
2. **M2 item 3** — CPU expert fallback and GPU grouped candidate plans under one
   interface, with bounded queues and NUMA-aware host placement — follows 0020.
3. **M2 item 4's Laguna checkpoint finished downloading during this task and is
   now verified complete.** `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` held 8
   of 15 shards when this contract was written; at the end of the session all 15
   are present and every one satisfies `8 + header + payload_end == file size`,
   with the payload ends summing to the index's `total_size` of 76,813,095,232 B.
   That completeness check is the only thing done to it: **no metadata was
   interpreted, no tensor was read, and `configuration_laguna.py` and
   `modeling_laguna.py` are remote code document 03 forbids executing.** Laguna
   metadata and graph are M2 item 4 and belong to a later bounded task, after
   the residency work 0020 owns.
4. **No device routed execution.** The three operations refuse on the selected
   chain and `ExpertMlp`/`Combine` fail closed for partitioning. **Grouped GPU expert execution is M2 item 3**, not M5: the roadmap asks there
for "CPU expert fallback and GPU grouped candidate plans under one interface".
**M5** owns expert *partitioning* across ranks and **M6** the shared performance
work.
5. **O1, O2, O4–O7 remain open.** No gate was resolved by this task.
