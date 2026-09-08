# Task 0003 — M1, part 1: the BF16 host reference interpreter

Status: **active**. Proposed 2026-09-07 after the [M0 correction
task](0002-m0-review-and-integer-transition.md) closed F1–F6 and four review passes; started the
same day with those gates green (`arch-check` 19 rejected + 1 accepted fixtures, `spec-check` 10
documents, 222 host unit tests + 1 doctest, `test-gpu` 15 cases with both architectures qualified).

**The contract below was written and committed before any implementation code**, which is document
07's rule: "the implementation task must specify its absolute/relative or normalized error metric,
independently generated reference, relevant scales and threshold before optimization ... agents must
not choose a threshold after seeing a failing candidate." The commit that fills in the contract
contains no `.rs` change; the implementation follows in a separate commit.

This is the **smallest** M1 slice. It is not "the M1 vertical slice": document 06 M1 also wants a
manifest reader, a rank-owned CUDA execution path, paged state and a generation service. Those are
separate tasks that consume this one. Attempting them together is how a slice becomes a campaign.

## Identity and authority

- Task ID / milestone / owner: 0003 / M1 / implementation agent (Claude), owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `34ac8e6`
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Findings repaired: none. This is new capability, not a correction.
- Required documents: 02 (common API, operation contracts), 03 (BF16 v1, accumulation), 04
  (attention descriptor), 07 (correctness ladder, error metrics), 09 §B; AGENTS.md; ADR 0003.
- Owner gates: **none resolved, and none needed.** O1 (catalog), O2 (quality), O4 (intrinsic
  low-bit state) and O5 (storage) are all OPEN and none of them gates this work: there is no
  checkpoint, no quantization, no cache dtype choice and no bulk write. **Stop and ask** if the work
  appears to need one — that is the signal that the scope grew.

## Bounded deliverable

One outcome: **a host reference interpreter that executes a small BF16 graph, operation by
operation, against registered oracles, and produces logits for a fixed synthetic model.**

- Sole owning shared component: a new `moxie-interp` crate, plus the operation contracts it needs in
  `moxie-graph`. `moxie-oracles` gains the reference implementations for the operations below.
- Operations, and only these: `Embedding`, `Linear`, `RmsNorm`, `SwiGlu`, `Rope`, `Attention`
  (full causal, single head group, exact), `Residual`, `VocabProjection`.
- The graph gains what F6 left out and M1 needs: **edges, shapes and state effects**. An operation
  node names its inputs; shapes are checked with the existing symbolic `Dim`; a state-touching node
  declares its transaction.
- Non-goals, and forbidden shortcuts: no CUDA, no kernel, no checkpoint, no manifest reader, no
  quantized weights, no paging, no sampler integration, no service, no model crate. No `Custom`
  operation variant, ever. No operation without a registered oracle. No fast path — this is the
  thing other paths are compared against, and a "reference" that has been optimised is not one.
- Second-consumer proof: document 06 M1 requires "at least two distinct shapes consume every
  foundational linear/attention operation". Two synthetic graphs with different hidden sizes, head
  counts and vocabulary sizes, one of them with a non-divisible dimension that must be rejected or
  padded explicitly.
- Temporary paths: none. Nothing here is a bridge.

## Contract, fixed before implementation

Notation: `u = 2^-24` is the FP32 unit roundoff. `γ(n) = n·u / (1 − n·u)` is the standard bound for
`n` chained FP32 roundings (Higham, *Accuracy and Stability of Numerical Algorithms*, §3.1). Every
threshold below is `γ` of a **counted** number of rounding steps — derived from the equation, not
selected. `round_bf16` is `moxie_format::bf16::f32_to_bf16_bits`, round-to-nearest-even, already
pinned and exhaustively tested.

### Storage and compute

Every tensor the interpreter holds is **BF16-valued**: stored as `f32`, but every element is exactly
a representable BF16 number, so widening to FP32 is exact and free of surprise. The interpreter
asserts this invariant on construction and after every rounding boundary — a "BF16 tensor" holding
an unrepresentable value would make every downstream error bound a fiction.

Compute is FP32. `AccumulationPolicy::Bf16InF32Acc` throughout. Reductions run in FP32 in a **fixed
sequential ascending order**; that order is part of the contract, because a reference whose result
depends on reassociation cannot be the thing a kernel is compared against. Document 07 permits a
kernel to reassociate later — with measured error and quality evidence — and this is the baseline it
will have to produce that evidence against.

### Rounding boundaries

Where FP32 becomes BF16 is a **semantic decision**, not an implementation detail, because a fused
kernel must round in the same places or it computes something else. Per operation:

| Operation | FP32 internally | Rounded to BF16 |
|---|---|---|
| `Embedding` | — (a copy) | never; the stored value passes through unchanged |
| `Linear` | products, the whole reduction, the bias add | **once**, on the final sum |
| `RmsNorm` | squares, the mean, `+eps`, `rsqrt`, the gain multiply | **once**, after the gain |
| `SwiGlu` | `silu(gate)` and the product with `up` | **once**, on the product |
| `Rope` | `sin`/`cos`, both products, the sum | **once per output element** |
| `Residual` | the add | **once** |
| `Attention` | scores, scale, softmax, the weighted value sum | **once**, on the output vector |
| `VocabProjection` | products, the whole reduction | **never — logits stay FP32** |

The last row is deliberate and is the only place the pattern breaks. Document 05 has the sampler
compute "the shared pre-truncation log normalizer from the modified, legal logits", and exact
speculative verification consumes that distribution. Rounding logits to BF16 quantises the
distribution before any of that runs, so the vocabulary projection's output dtype is
`ActivationPrecision::F32` and the graph refuses to make it anything else.

Intermediate rounding **inside** an operation is forbidden. A `Linear` that rounded per tile would
be a different operation with a different error bound, and this table is what says so.

### Equations, shapes and per-operation contracts

`R` = rows, `H` = hidden, `V` = vocabulary, `F` = feed-forward width, `Hd` = head dim, `Nh` = heads.
Shapes are `Dim` expressions and are checked with the existing symbolic evaluator; a non-exact
division is `DimError::NotDivisible` and fails graph construction, never truncates.

**`Embedding`** — `y[r, :] = W[tokens[r], :]`, `W: [V, H]`, `y: [R, H]`.
Token ids are an integer role, not a quantised weight (document 02: "Token IDs, positions,
page/group/sparse indices and masks also need integer/boolean descriptor roles"). An id outside
`0..V` is `InvalidRequest`, never a wrap.
*Error: **exact**, bit-identical. No arithmetic happens.*
*Partition: `Replicated`. Sharding the vocabulary axis is an M5 question and is not claimed here.*
*State effect: `None`.*

**`Linear`** — `y[r, o] = Σ_{k=0}^{K-1} x[r, k]·W[o, k] (+ b[o])`, `x: [R, K]`, `W: [O, K]`,
`b: [O]`, `y: [R, O]`.
*Error: `|ŷ − y_f64| ≤ γ(K + 1) · Σ_k |x[r,k]·W[o,k]|` before the BF16 rounding, where `K + 1`
counts `K` products-and-adds plus the bias add. Exact on inputs whose products and partial sums are
all FP32-representable.*
*Partition: `ColumnShardable` on `O` — each rank owns a slice of output channels and the bias is
applied exactly once, on the rank that owns it (document 04: "Bias and residual terms are applied
exactly once, not on every partial output before an unintended sum").*
*State effect: `None`.*

**`RmsNorm`** — `y[r, i] = x[r, i] · g[i] / sqrt( (1/H)·Σ_j x[r,j]² + ε )`, `g: [H]`, `ε` explicit
and stored in the node, never defaulted.
Distinct from `LayerNorm`: **no mean subtraction**. A test asserts the two differ on an input with
non-zero mean, so collapsing them is a failing test rather than a review comment.
*Error: `|ŷ − y_f64| ≤ γ(H + 4) · |y_f64|` — `H` accumulation steps for the sum of squares, plus the
divide by `H`, the `+ε`, the `sqrt` and the gain multiply.*
*Partition: `Replicated`. The reduction is over the full hidden axis, so a row-sharded input needs a
global reduction; declaring `Replicated` says the shared form is what M5 must extend, not that
sharding is impossible.*
*State effect: `None`.*

**`SwiGlu`** — `y[r, i] = silu(gate[r, i]) · up[r, i]`, `silu(v) = v / (1 + e^{−v})`.
`gate` and `up` are **separate inputs**. Document 02: "A gate/up tensor's logical order is not its
physical interleaved disk layout" — the graph takes two edges and any interleaving is an importer's
problem, not this operation's.
Distinct from `GeGlu` and from Kimi's bounded `SituGlu` (R06). Neither is implemented; requesting
one is `UnsupportedKernel`, not a silent substitution.
*Error: `|ŷ − y_f64| ≤ γ(4) · |y_f64|` — four rounding steps: `e^{−v}`, `1 +`, the divide, the
product. `e^{−v}` is bounded at 1 ulp by the platform's `expf`; the other three at 0.5 ulp each, so
4 full ulps bounds the composition.*
*Partition: `ColumnShardable` — elementwise on the feed-forward axis.*
*State effect: `None`.*

**`Rope`** — for `j` in `0..rotary_dim/2`, with `θ_j = pos · base^(−2j / rotary_dim)`:

```text
y[2j]     = x[2j]·cos(θ_j) − x[2j+1]·sin(θ_j)
y[2j+1]   = x[2j]·sin(θ_j) + x[2j+1]·cos(θ_j)
y[i]      = x[i]                                for i >= rotary_dim
```

`rotary_dim` must be even and `<= Hd`; both are checked at graph construction. `pos` is the
**absolute position in the sequence**, not the index within a chunk — R21 records a one-page fast
path that got this wrong, and the mask fixtures already encode the same distinction.
*Error: `|ŷ − y_f64| ≤ γ(4) · (|x[2j]| + |x[2j+1]|)` — `sin` and `cos` at 1 ulp each, two products
and one add. Absolute rather than relative because the sum can cancel.*
*Partition: `ColumnShardable` on the head axis; a shard must carry whole `(2j, 2j+1)` pairs, and a
split that would separate a pair is `NotDivisible` at construction.*
*State effect: `None`.*

**`Attention`** — full causal, one head group (MHA), exact:

```text
s[q,k] = (Q[q,:] · K[k,:]) / sqrt(Hd)      for k visible to q
p[q,:] = softmax_FP32(s[q,:])              over visible k only
o[q,:] = Σ_k p[q,k] · V[k,:]
```

Visibility is `moxie_oracles::mask::Visibility::Causal` over **absolute** positions, and a masked
key is **removed from the sum**, not given a large negative bias — the existing fixture asserts a
masked position contributes exactly zero, and a `-inf` bias only approximates that. A query with no
visible key is `Numerical`, never a uniform draw.
*Error: `|ô − o_f64| ≤ γ(2K + Hd + 3) · Σ_k |p[q,k]·V[k,i]|`, where `K` is the number of visible
keys: `Hd` steps for each dot product, `K` exponentials, `K` accumulation steps for the denominator,
the max subtraction, the divide, and the weighted sum.*
*Partition: `NotDetermined`. Head ownership, KV replication for GQA and the output reduction are
document 04's M5 work; declaring it undetermined makes TP lowering fail closed rather than silently
produce a rank-local answer.*
*State effect: **`Appends`**. See below.*

**`Residual`** — `y = a + b`, elementwise, shapes identical.
*Error: **exact** when `a + b` is BF16-representable; otherwise a single rounding, `γ(1)`.*
*Partition: `Replicated`; the residual is added exactly once.*
*State effect: `None`.*

**`VocabProjection`** — `logits[r, t] = Σ_k h[r, k]·W[t, k]`, `W: [V, H]`, output **FP32**.
*Error: as `Linear` with no bias, `γ(H)`, and no final rounding at all.*
*Partition: `ColumnShardable` on the vocabulary axis, with the caveat that document 04 requires
"sharded vocabulary normalization/sampling" to have correct global semantics — sharding the
projection does not by itself make the softmax correct, and M5 owns that.*
*State effect: `None`.*

### State, transactions and cancellation

`Attention` is the only state-touching operation here. Its contract:

- K and V for the rows being executed are appended to a per-layer host KV store at their **absolute
  positions**, and the branch's `executed` counter advances by the number of rows.
- Appends are **staged**, not written through. A step builds its appends in a scratch buffer and
  commits them to the store only when every operation in the graph has succeeded. This is the
  M1-scale form of "commit only accepted state": there is no partial commit to roll back from,
  because a failed or cancelled step never wrote.
- On success the step records its forward result with `SequenceState::record_logits`, so the
  provenance rules from task 0002 apply unchanged: the handle is minted by the sequence, carries its
  branch and prefix lineage, and a later rollback invalidates it.
- **Cancellation** (R08): the interpreter checks a cancellation token at every operation boundary and
  returns `Error::Cancelled { at }` naming the operation. The staged appends are dropped, the
  `executed` counter is untouched, the accepted prefix is untouched, and no result is recorded. A
  cancelled step leaves the state exactly as it found it, which the acceptance tests check by
  running a full step afterwards and comparing against a never-cancelled run.
- `KvPages` is `RestoreCapability::Truncate`, so a rollback drops the tail. No `Explicit` component
  is in this slice's schema, and no restore evidence is therefore required — stated so that a later
  reader does not think the evidence machinery was skipped.

### The oracle relationship, stated plainly

The task asks not to pretend to two independent implementations, so: **the interpreter dispatches to
`moxie-oracles`; it does not reimplement the mathematics.** There is one implementation of each
equation, in the oracle crate, and the interpreter is a graph walker that calls it. Claiming
otherwise would be theatre.

The independence lives in the **tests**, which compute expected values from the equations written
out in the test body — in FP64, separately, from the specification text above — and compare. That is
the "independently generated reference" document 07 asks for, and it is where a transcription error
in the oracle would show up. Tests that merely re-run the oracle and compare to itself are worthless
and are not counted as evidence anywhere in this task.

The registry is the enforcement: the interpreter refuses to execute a node whose `OracleId` is not
registered, so an operation cannot be added without a reference. An acceptance test asserts this
against a deliberately unregistered operation.

### Error reporting

Every numerical test reports **max, RMS and p99** absolute and normalized error, per document 07
("Include max, RMS and high-percentile errors"), and asserts against the `γ` bound declared for that
operation. A test that reports only a max is not sufficient. The helper lives in
`moxie_oracles::metric` so the same numbers are produced everywhere, including by the CUDA lane when
it arrives.

## Acceptance

- `cargo xtask arch-check` passes with `moxie-interp` declared in the ownership table.
- `cargo test --workspace` passes on the host lane, with no CUDA feature enabled.
- Per-operation exactness or error metric, declared in this file before implementation, met by
  every case. Edge shapes, masks, ties and non-finite behaviour covered per document 07's
  correctness ladder.
- Whole-versus-chunked prefill parity on a small fixture, using the mask fixtures already in
  `moxie-oracles::mask`.
- One-token and multi-token generation through the interpreter, on both synthetic graphs.
- A cancelled generation followed by a second generation that produces the same result as an
  uncancelled one.
- An operation with no registered oracle is refused, and a test asserts it.
- Support-matrix rows: add a "BF16 host reference interpreter" row with its gate ID. Do **not**
  touch any row mentioning a checkpoint, a kernel or a context length.
- Stop condition: if the slice appears to need a checkpoint, a CUDA kernel, a real tokenizer or an
  owner gate, stop and report. Each of those is a different task.

## Why this shape

The corrections in task 0002 make this the next thing that fits. The oracle registry exists but has
no consumer, so nothing yet proves an unregistered operation is actually refused in practice. The
state crate's counters and provenance rules exist but no execution advances them. The mask and
routing fixtures exist but nothing consumes them. An interpreter is the smallest thing that turns
all three from declarations into enforced behaviour — which is exactly the transition the review
found M0 had not yet made.

It also comes before any kernel on purpose. Document 07: "Missing numerical contracts block that
primitive's optimized gate." Writing the CUDA path first would mean choosing tolerances after seeing
what the kernel produces.
