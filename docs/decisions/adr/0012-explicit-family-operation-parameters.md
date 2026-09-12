# ADR 0012 — Family mathematics is explicit operation parameters, not defaults

- ID / date / author / status: 0012, 2026-09-12, implementation agent; **accepted with task 0016**.
- Classification: measured implementation choice.
- Scope and owning shared component: `moxie-graph`'s `OpParams` and the
  `moxie-oracles` references they are validated against.
- Supersedes / superseded by: neither. Extends the operation catalogue task 0003
  established.

## Problem and mechanism

Task 0003 built the operation catalogue against one synthetic graph, so several
of its mathematical choices were written as constants rather than parameters:

| Constant | Where | What it assumed |
|---|---|---|
| `1.0 / (head_dim as f32).sqrt()` | `moxie-oracles::attention` | every family scales scores by the reciprocal square root |
| `kv_heads == heads` | the same function, implicitly | no grouped-query attention |
| `(2j, 2j+1)` pairing | `moxie-oracles::rope` | one rotary pairing convention |
| `base^(−2j / rotary_dim)` | the same | the denominator is the rotated width |
| no embedding scale, no logit cap, no residual scale, whole-row norm | various | one family's composition |

Every one of these is wrong for Gemma 4, whose pinned reference
(`/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`)
passes `scale = 1.0F` at `src/models/gemma4/gemma4_runtime.cpp:857`, pairs halves
at `src/models/gemma4/gemma4_ops.cpp:22` while dividing by the **full** head
dimension, normalizes queries and keys per head at `gemma4_runtime.cpp:798`,
scales embeddings at `:699`, scales only its MLP residual at `:1230`, and caps
logits at `kernels/cuda/detail/backend_kernels.cuh:905`.

None of these differences would have produced an error. Each produces plausible
text and wrong logits — which is exactly the failure R06 describes and document
09 §B forbids: "do not replace unfamiliar math with a standard transformer
guess."

## Options examined

**1. Defaulted parameters** — add the fields with `Default` implementations
carrying today's values. Cheapest migration: no existing construction site
changes. Rejected. Document 02 already settled this shape of question for
epsilon: "a defaulted epsilon is a silent numerical difference between two
checkpoints that declared different ones." A default is invisible at the call
site, so the one place a reader would look to find out what a graph does is the
one place that does not say.

**2. A per-family options struct threaded through the interpreter** — one
`FamilyProfile` consulted by each operation. Rejected: it moves family
mathematics out of the graph and into an ambient parameter, so the graph stops
being a complete description of what executes, and two nodes of the same
operation could no longer differ. Gemma needs exactly that: its sliding and
global layers disagree about theta and rotary width, and its two residuals
disagree about the scale.

**3. Explicit required fields** — chosen. Every construction site states the
value it uses. The migration cost is real and bounded: thirty sites across nine
files, each now saying what it previously implied.

## Decision and authority

Seven parameters become explicit fields with **no default**:

1. `Embedding.scale` — the factor on every looked-up row.
2. `VocabProjection.softcap` — `Option<f32>`; `None` is no cap, because no cap
   magnitude means "uncapped".
3. `Residual.scale` — per residual, not per layer.
4. `RmsNorm.group` — how many independent normalizations a row is divided into.
5. `Rope.layout` — `Interleaved` or `HalfSplit`.
6. `Rope.frequency_dim` — the inverse-frequency denominator, separate from
   `rotary_dim` because partial rotation makes them differ.
7. `Attention.scale` and `Attention.kv_heads` — the score factor and the
   grouped-query mapping.

`moxie_graph::reciprocal_sqrt_scale` supplies the conventional attention scale
for callers that want it, so the usual choice stays one call and the unusual one
stays visible. `moxie_oracles::rope::Rotation::interleaved` and
`moxie_oracles::attention::Heads::multi` do the same for the references.

This is an implementation choice within the existing common-API contract:
document 02 already required "RoPE/other positional operations with
checkpoint-defined scaling, partial dimensions, and position semantics",
"GQA/MQA head mappings" and GeGLU. It resolves no owner gate, changes no
precision, context, cache policy or product scope, and adds no new numerical
tolerance.

`Op::GeGlu` gains an `OpParams` variant and a registered oracle. Its gate term's
BF16 rounding is part of the equation rather than storage, because the pinned
source rounds there; the contrast with SwiGLU's single rounding on the product is
the reason document 02 keeps the two activations separate.

## Evidence and acceptance

Each parameter has an FP64 transcription in `moxie-oracles` and a fixture citing
the pinned line it transcribes. Beyond agreement, the acceptance condition is a
**difference**: `every_gemma_parameter_is_load_bearing` and
`the_attention_scale_and_rope_layout_are_load_bearing` in
`crates/moxie-cli/tests/gemma.rs` substitute the conventional value into the
reduced Gemma graph and require the logits to change. A parameter that can be
swapped without changing anything is a failed acceptance.

That test already earned its place. The first reduced geometry used
`head_dim` 8 with a quarter rotary factor, which rotates a single pair whose
inverse frequency is `base^0` — so the global RoPE base cancelled and the
geometry silently stopped testing it. The test failed, and the geometries now use
16 and 8, the smallest widths at which the base matters.

Every existing consumer keeps its behaviour bit for bit: the host workspace
passed 593 tests + 9 doctests before this change and the same 593 after, with the
new tests counted separately.

Two device-kernel selections now refuse rather than silently dropping a factor:
a `Residual` whose scale is not 1.0, and an `RmsNorm` whose group is not 1. There
is no qualified kernel for either, and `moxie-plan::selected` returns
`UnsupportedKernel` with the offending value named.

## Enforcement and removal

Absence of a `Default` implementation is the enforcement: a new construction site
does not compile until it states every field. `OpParams::check_params` rejects a
non-divisible head grouping, a zero frequency denominator, a half-split layout on
an odd head, and any scale, cap or epsilon that is not finite and positive.

Nothing here expires. When a device kernel is qualified for a scaled residual, a
grouped norm or GeGLU, its refusal in `moxie-plan::selected` is replaced by a
selection — the refusal is a missing kernel, not a temporary exception.
