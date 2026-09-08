# Task 0003 — M1, part 1: the BF16 host reference interpreter

Status: **accepted 2026-09-08** at `0908f4d`, after four review passes and their corrections.

The acceptance is of **this bounded task only** — the BF16 host reference interpreter and its
numerical contracts. It is explicitly *not* acceptance of the M1 milestone, nor of any checkpoint,
GPU-performance or model-quality claim. The reviewer's closing note is carried into
[task 0004](0004-m1-state-transactions.md): M1.4 should make publication transactional, keeping the
regressions below as acceptance tests. Proposed 2026-09-07 after the [M0 correction
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
| `SwiGlu` | the sigmoid, `silu(gate)` and the product with `up`, **in FP64** | **once**, narrowing the product |
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

**`SwiGlu`** — `y[r, i] = silu(gate[r, i]) · up[r, i]`, `silu(v) = v · σ(v)`.

σ is evaluated in the **overflow-free** form, choosing by sign so the exponent is
never positive:

```text
σ(v) = 1 / (1 + e^{−v})        for v >= 0
σ(v) = e^{v} / (1 + e^{v})     for v <  0
```

The naive single expression overflows: `e^{88}` already exceeds `f32::MAX`, so
for `v` around −88 the denominator becomes `+inf` and `silu` collapses to exactly
zero. This is the form the pinned legacy source uses
(`src/platform/numerics.cpp:44`), and document 08 says to read those references
before inventing a replacement.
`gate` and `up` are **separate inputs**. Document 02: "A gate/up tensor's logical order is not its
physical interleaved disk layout" — the graph takes two edges and any interleaving is an importer's
problem, not this operation's.
Distinct from `GeGlu` and from Kimi's bounded `SituGlu` (R06). Neither is implemented; requesting
one is `UnsupportedKernel`, not a silent substitution.
*Error: `|ŷ − y_f64| ≤ γ(2) · |y_f64|` while the **output** is a normal FP32 number; an absolute
floor of `f32::MIN_POSITIVE` where the output itself is subnormal.*

*The whole expression is evaluated in FP64 and narrowed once, so the only FP32 rounding is that
narrowing — which is why the bound is `γ(2)` rather than the `γ(4)` an FP32 chain would need. This is
not a refinement; it is a correction. An FP32 intermediate **cannot represent this operation's own
output range**: `σ(−104)` is about `1e−45`, at the bottom of the subnormals, so `silu(−104)` flushes
to zero, while `silu(−104) · 1e30` is `−7.1e−14` and perfectly normal. The fifth review found that
100% error on an ordinary output. Rounding the intermediate was also an extra boundary the table
above never declared: SwiGLU's single rounding is on the product.*
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
*Error — **revised 2026-09-08; the first version was disproven**. See "The attention bound was
wrong" below. The sound statement is two terms, and it is data-dependent:*

```text
Δs   = γ(Hd) · max_k ( Σ_i |Q[q,i]·K[k,i]| ) / sqrt(Hd)      // score error
|ô − o_f64| ≤ (e^{2·Δs} − 1) · max_k |V[k,i]|                // through the softmax
            + γ(K + 2) · Σ_k |p[q,k]·V[k,i]|                 // the weighted sum
            + (2K + Hd + 3) · η                              // gradual underflow
```

*where `η = 2^-150` is half the smallest FP32 subnormal. A relative model says nothing once results
leave the normal range: the fifth review measured an absolute error 51 times the two relative terms
with values at `2^-133`. The additive term is around `1e-44` on ordinary data, which is to say
nothing, and is the whole bound where the data is tiny.*

*Implemented as `moxie_oracles::attention::attention_error_bound`, so the contract and the tests use
one expression rather than two that can drift apart.*
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

### The attention bound was wrong, and why the corrected one has a different shape

The first version of this contract said `γ(2K + Hd + 3) · Σ|p·V|`, derived by counting rounding
steps as everywhere else. **It is invalid, and the fourth review disproved it with a counterexample
built from BF16-representable inputs.** The counterexample is preserved as
`a_cancelling_score_widens_the_bound_because_the_error_is_real`:

```text
K[0] = [2^24, 1, -2^24, 0]     Q = [1, 1, 1, 1]     V[0] = [1, 0, 0, 0]
K[1] = [2^24, 0, -2^24, 0]                          V[1] = [0, 1, 0, 0]

exact scores (FP64): [0.5, 0.0]      ->  softmax [0.6225, 0.3775]
computed  (FP32):    [0.0, 0.0]      ->  softmax [0.5,    0.5   ]

normalized error 0.32, against a declared bound of 6.6e-7
```

FP32 cannot represent `2^24 + 1`, so the first key's dot product cancels to exactly zero and the two
scores become equal. The softmax then returns a uniform distribution where the true one is not.

Counting operations cannot bound this, because the score error does not *add* to the result — it
passes through an **exponential**. Multiplying a step count by `Σ|p·V|` is the right shape for the
weighted sum and the wrong shape for the softmax, and no constant makes it right.

The corrected bound says what actually holds. If every score carries absolute error at most `Δs`,
then every ratio `p̂_j/p_j` lies in `[e^{−2Δs}, e^{2Δs}]`, so `‖p̂ − p‖₁ ≤ e^{2Δs} − 1`; multiply by
the largest value component and add the ordinary sequential-sum bound for the weighted sum itself.
`Δs` comes from the dot product's own conditioning, which is where the cancellation shows up.

Two consequences, both stated rather than discovered later:

- **Attention's FP32 accuracy is not a constant.** It depends on how well conditioned `Q·K` is. A
  CUDA kernel qualified against this contract must be qualified on data whose conditioning is
  stated, and document 07's requirement to "stress cancellation and near-zero outputs" is precisely
  the regime where the first term dominates.
- **The bound is honest about being weak there.** On the counterexample it exceeds 0.1, and a test
  asserts that; on well-conditioned data a second test asserts it stays below 1e-5. A bound that was
  merely enlarged until the counterexample fit would fail the second test.

The other operations' bounds are unchanged in *shape*. `Linear`'s `γ(K+1)·Σ|x·w|` is the standard
Higham result and survives cancellation because it is normalized by term magnitude rather than by
`|y|`; the norm's sum of squares cannot cancel; RoPE's is stated absolutely for the same reason.
`moxie_oracles::metric::bound(n, scale)` is the underflow-aware form — `γ(n)·scale + n·η` — and is
what any bound over data that can be subnormal should use.

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
- **Publication cannot half-succeed, and the reason is checked rather than argued.** `state.execute`
  cannot be taken back -- there is no `unexecute`, and the restore path puts back the *cache and
  nothing else*. So the rule is not "undo on failure"; it is that **nothing after `execute` may be
  able to fail**. The cache's ownership, its layer coverage, the counter's headroom, and finally the
  cache's post-append length are all established while the state is still untouched; the KV appends,
  being undoable, go first and are truncated back on error. Two review passes found holes in an
  earlier version that argued this from the shape of the code instead of checking it, so the last
  precondition is now an explicit comparison rather than an inference.
- **A graph with no attention is refused.** It touches no sequence state, so it has nothing to
  publish and would not advance `executed`. That is a coherent operation with different publication
  rules, not this one; document 06 M1.4's transaction API is where it would belong.
- `KvPages` is `RestoreCapability::Truncate`, so a rollback drops the tail. No `Explicit` component
  is in this slice's schema, and no restore evidence is therefore required — stated so that a later
  reader does not think the evidence machinery was skipped.
- **The physical cache carries the same identity as the state.** `KvCache` is bound to a
  `(sequence, branch, prefix lineage)` and the interpreter checks it before reading a byte. Without
  that, `moxie-state`'s provenance rules guard a counter while the bytes come from anywhere: the
  fourth review substituted one sequence's cache into another's execution, and the step succeeded,
  returned the wrong history's answer, and reported valid logits. The identity reuses
  `PrefixLineage` rather than inventing a second scheme, so "another sequence", "another branch" and
  "the prefix that used to be here" are one comparison.

### Operand validation at execution

Declared shapes, roles and precisions are checked against the values actually supplied, not only
between operations at build time:

- **Precision.** A tensor whose precision differs from its declared role is refused. The fourth
  review bound an FP32 tensor holding a BF16-unrepresentable value to a BF16-declared input and
  execution accepted it — a checked constructor guarantees nothing when the caller can pick a
  different one.
- **Finiteness.** A non-finite bound value is `InvalidArtifact` before anything runs, and a
  non-finite node output is `Numerical` naming the operation that produced it. Document 05's rule
  that "NaN logits ... produce typed errors" is worth nothing if the NaN is admitted at the boundary
  and only noticed after the state has advanced.
- **Positions.** Every position-consuming operation must read the **same** binding, enforced when
  the graph is built. Validating one operand and trusting the rest let a graph whose RoPE read
  position 0 and whose attention read 999 execute at prefix 1.

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


---

## Result

Implemented on `main`, base commit `388d85c` (the contract commit above). The contract was fixed
first and not touched afterwards: no threshold in it was adjusted once a test ran.

### Commands

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | **PASS** |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | **PASS** |
| `cargo test --workspace --locked --offline` | **PASS**, 290 unit/integration + 1 doctest |
| `cargo xtask arch-check` | **PASS**, `moxie-interp` declared; 19 rejected + 1 accepted fixtures |
| `cargo xtask spec-check` | **PASS**, 10 documents, digests unchanged |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | **PASS**, 298 + 2 doctests |
| `cargo xtask-cuda test-gpu` | **PASS**, unchanged; this slice touches no CUDA |

Host counts by crate: `moxie-oracles` 104, `moxie-format` 54, `xtask` 30, `moxie-state` 28,
`moxie-types` 22, `moxie-interp` 10 + 16 integration, `moxie-graph` 11, `moxie-cuda` 9,
`moxie-model-api` 5, `moxie-kernels` 1. Before this task: 222 + 1.

### What was built

- **`moxie-graph` gained the graph.** `ValueId`, `NodeId`, `TensorSpec`, `ValueRole`, `OpParams`,
  `Node`, `Graph`, `GraphBuilder`. Roles are three distinct types -- `Weight`, `Activation`,
  `Index` -- so document 02's "token IDs ... are not quantized weights or floating activations" is a
  type error rather than a convention. `Visibility` moved here from `moxie-oracles`, because a mask
  rule is part of the attention contract rather than of the reference that evaluates it.
- **`moxie-oracles` gained six references**: `linear` (linear, embedding, vocabulary projection),
  `norm` (RMS, plus the LayerNorm it must not be confused with), `activation` (SwiGLU),
  `rope`, `attention` (multi-head over an absolute-position history), `residual`, and `metric`
  (γ and the max/RMS/p99 summary document 07 requires).
- **`moxie-interp` is new**: `HostTensor` with the BF16-valued invariant checked rather than
  assumed, `KvCache`, `Cancel`, and the `Interpreter` that walks a graph.

### The contract's rules, as enforced behaviour

| Rule | Where it now bites |
|---|---|
| An operation with no registered oracle cannot be lowered | `GraphBuilder::finish` refuses; the interpreter is never reached |
| Logits stay FP32 | `OpParams::output_precision`, asserted against a fixture where some logits are provably not BF16-representable |
| Positions are absolute (R21) | the interpreter checks them against the branch's own `executed` frontier and refuses a chunk-local index |
| Cancellation leaves state untouched (R08) | KV appends are staged and committed only on success; tested at **every** node depth |
| Declared shapes | bound against the actual data at run time, with `rows` bound from the step |
| Rounding boundaries | one `HostTensor::round_to_bf16`, called exactly where the table says |
| TP fails closed on attention | `PartitionRule::NotDetermined`, asserted per node |

### Numerical results

Every operation's reference is checked against an FP64 transcription of the equation, written
separately in the test body, and reported as max/RMS/p99 against its declared γ bound. All within
bound; nothing was retuned. The `linear` gate over the fixture's own `w_gate` (12 -> 20) reports
through `println!` so the numbers are in the test output rather than only in an assertion.

Two tests exist to stop a bound being met by accident: `a_cancelling_dot_product_is_measured_against_term_magnitude`
constructs a dot product whose terms are ~1e12 and whose result is ~0, and
`the_summation_order_is_part_of_the_contract` shows forward and reverse summation genuinely disagree
on an ill-conditioned input -- so the declared order is doing work rather than describing a
coincidence.

### Acceptance, item by item

| Requirement | Evidence |
|---|---|
| `arch-check` passes with `moxie-interp` in the ownership table | `arch-check` PASS; the checker rejected the crate until it was declared |
| `cargo test --workspace` on the host lane, no CUDA | PASS, 291 |
| Per-operation error metric met | every oracle test, against its γ bound |
| Whole-versus-chunked prefill parity | `whole_and_chunked_prefill_agree`, at **every** chunk width 1..6, logits **and** KV cache compared |
| One-token and multi-token generation, both graphs | `both_graphs_produce_logits_for_one_token_and_for_many` |
| Cancel, then a clean generation matching an uncancelled one | `a_cancelled_step_leaves_the_state_exactly_as_it_found_it`, cancelling at every node depth |
| An unregistered operation is refused | `an_operation_with_no_registered_oracle_cannot_be_built_into_a_graph` |
| Two distinct shapes consume every operation | graphs A (h8/2 heads/ffn16/vocab16) and B (h12/3 heads/ffn20/vocab7) |
| Non-divisible dimension rejected explicitly | `shape_and_divisibility_errors_are_refused_at_construction`: odd `rotary_dim` is `DimError::NotDivisible`, a head geometry that does not match its input is refused |
| Support-matrix row added | `G-INTERP-BF16`; no checkpoint, kernel or context row touched |

### What this does not establish

Stated plainly, because a reference interpreter that produces logits looks more like an engine than
it is:

- **No model.** Two synthetic graph fixtures with pseudo-random weights. No checkpoint has been
  imported, no tokenizer exists, and nothing here is evidence about any of the ten candidates.
  Document 06 M1.5: "Never describe synthetic output as model support."
- **No kernel, no device.** Nothing in this slice touches CUDA. The GPU lane is unchanged and its
  result is carried forward, not re-earned.
- **No performance meaning.** This is deliberately the slow path. It allocates per row and reduces
  in a fixed order so that a kernel has something exact to be compared against.
- **No sampler.** `moxie-oracles::sampler` exists and is not wired in; greedy selection in the state
  test is a state-machine step, not sampling.
- **One head group, full causal, one layer per KV store.** GQA/MQA head mapping, sliding windows in
  a graph, sinks and biases, MLA, and model-defined sparse selection are document 04's later work
  and are absent rather than approximated.
- **No paging, no memory authority, no service.** Their *initial* implementations are M1.3 and
  M1.4, not M2/M4 — document 06 M1.3 asks for "rank-owned CUDA context, event-backed leases,
  resource ledger, basic allocator" and M1.4 for "appendable paged state plus transaction API ...
  a generation service and minimal diagnostic CLI". M2 and M4 deepen them. An earlier draft of this
  record deferred them a milestone too far; the next-task list below is the correct reading.

### Next

The M1 slice continues, and each of these is its own bounded task rather than an extension of this
one:

1. **Manifest reader for tiny BF16 artifacts** (document 06 M1.2) -- bounded tensor reads and
   validation, feeding the same graph. This is where `moxie-format`'s storage half begins, and where
   the owner-designated [artifact roots](../evidence/artifact-roots.md) first matter.
2. **Rank-owned CUDA execution of one layer chain** (M1.3) -- against *this* interpreter, with the γ
   bounds above as the comparison points. Document 07: the numerical contract exists before the
   optimized gate, which is now true.
3. **Appendable paged state plus the transaction API** (M1.4) -- replacing `KvCache`'s dense `Vec`
   with the real thing, and giving `SequenceState`'s restore evidence something that can actually
   prove a restoration happened, which the third review flagged as owed once buffers arrive.
4. **The generation service and diagnostic CLI** (M1.4) -- the first place a sampler is wired in.

Recommended order is 1, 3, 2, 4: the manifest and the state API are what the CUDA path needs to be
compared *at*, and doing the kernel before the paged state would mean writing it twice.


---

## Review corrections, 2026-09-08

A review of `0f803e1` found seven issues that the passing tests did not catch, two of them
mathematical. All seven are reproduced and closed. **The two numerical findings were disproven
contracts, not tolerances that needed room**, and both are corrected by revising the analysis with
the counterexample preserved as a test.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 290 + 1 doctest | **303 + 1 doctest** |
| device lane | 298 + 2 doctests | **311 + 2 doctests** |
| `cargo fmt`, `clippy -D warnings`, `arch-check`, `spec-check` | PASS | **PASS** |
| `cargo xtask-cuda test-gpu` | PASS | **PASS**, unchanged |

### R1 — the attention error bound was invalid · closed

Reproduced with BF16-representable inputs: normalized error **0.32** against a declared bound of
**6.6e-7**. Counting rounding steps cannot bound an error that passes through a softmax, because it
passes through an exponential rather than being added to the result. The corrected bound has a
different *shape*, not a bigger constant — see
"[The attention bound was wrong](#the-attention-bound-was-wrong-and-why-the-corrected-one-has-a-different-shape)"
above for the counterexample and the derivation.

It is implemented once, as `moxie_oracles::attention::attention_error_bound`, so the contract and the
tests cannot drift apart. Three tests: the counterexample (asserting the old bound was wrong by
orders of magnitude, and that the new one exceeds 0.1 there), a well-conditioned case asserting the
bound stays below 1e-5, and the general comparison. A bound merely enlarged to fit the counterexample
would fail the second.

The review also noted the old test normalized by `Σ|V|` where the contract said `Σ|p·V|` — a weaker
metric than the one declared. The shared function removes that gap by construction.

### R2 — SwiGLU collapsed to zero for representable inputs · closed

Reproduced: gate `−90` with a BF16-rounded `up` near `1e30` returned exactly `0`; the reference is
`−7.3765e−8`. `1/(1 + e^{−v})` overflows for `v` around −88, so the denominator became `+inf`.

Fixed with the sign-dependent sigmoid, which is **the form the pinned legacy source already uses**
(`src/platform/numerics.cpp:44`). Document 08 says to read those references before inventing a
replacement, and this is a case where not doing so cost a correct answer. The contract now also
states the subnormal limit explicitly: the relative bound holds while the result is normal, and below
that an absolute floor of `f32::MIN_POSITIVE` applies.

### R3 — a cache from another sequence was accepted · closed

Reproduced: substituting a cache holding a different token prefix succeeded, returned the other
history's answer, and `next_logits_valid` was true. `KvCache` had no identity, so every provenance
rule in `moxie-state` was guarding a counter while the bytes came from anywhere.

`KvCache` is now bound to a `(sequence, branch, prefix lineage)` and `check_owner` runs before the
interpreter reads a byte. The identity reuses `PrefixLineage` rather than inventing a second scheme,
so one comparison distinguishes another sequence, another branch, a cache that lags or leads the
frontier, and a cache holding a prefix that has since been replaced. Four tests, one per case.

Full paging remains a later task, as the review allowed.

### R4 — bindings bypassed the declared precision · closed

Reproduced: an FP32 tensor holding a BF16-unrepresentable value satisfied a BF16-declared input.
Binding validation compared shapes and never dtypes, so every error bound downstream rested on an
invariant nothing checked — a checked constructor guarantees nothing when the caller can choose a
different one. Role and precision are now compared at binding, in both directions (a tensor where an
index is declared, and the reverse).

### R5 — only the first position operand was validated · closed

Reproduced: a graph whose RoPE read position `0` and whose attention read `999` executed at prefix 1.

Fixed structurally rather than by validating each operand: `GraphBuilder` requires every
position-consuming node to read the **same** binding, so there is one vector and the frontier check
covers all of it. A graph built the other way is refused at construction.

### R6 — non-finite results were committed · closed

Reproduced: a BF16-tagged vocabulary weight containing NaN produced a successful step with NaN
logits, advanced the counters, and made `next_logits_valid` true. NaN is BF16-representable, so the
storage invariant does not catch it.

Non-finite bound values are now `InvalidArtifact` before anything runs, and a non-finite node output
is `Numerical` naming the operation that produced it — attributing it to the operation rather than
leaving a mysterious logit.

### R7 — malformed KV geometry panicked · closed

Reproduced: a one-element key attended with head dimension two ran the slice off the end.
`KvHistory::append` checks key and value widths against each other, which is not the same as checking
them against the geometry they are later read under. `head_slice` now validates and returns
`InvalidArtifact`; document 02 requires typed errors at this boundary, and a panic is not one.

### Record correction

The earlier "no paging, no memory authority, no service — M2 and M4 own those" was wrong about the
milestone. Document 06 puts their *initial* implementations in **M1.3 and M1.4**; M2 and M4 deepen
them. Corrected above, and the next-task list was already the right reading.

### What this changes about using these bounds to qualify CUDA work

The attention contract is now data-dependent and says so. A kernel qualified against it must be
qualified on data whose conditioning is stated, and the cancellation regime document 07 asks to
stress is exactly where the bound is weak — correctly, because FP32 attention genuinely is. That is
a more useful acceptance contract than the constant it replaces, but it is a different one, and any
future kernel gate has to be written against this version.


---

## Second review corrections, 2026-09-08

A second review of `663f1d1` found four further gaps. All four are reproduced and closed. Two are
numerical contracts that were still incomplete over their stated input domain, and the review was
right that a bound which fails anywhere in that domain is not usable for CUDA qualification.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 303 + 1 doctest | **309 + 1 doctest** |
| device lane | 311 + 2 doctests | **317 + 2 doctests** |
| `fmt`, `clippy -D warnings`, `arch-check`, `spec-check`, no-driver host build | PASS | **PASS** |
| `cargo xtask-cuda test-gpu`, hidden-SM120 gate | PASS / exit 1 | **PASS / exit 1** |

### S1 — stale cache contents could be re-stamped as current · closed

Reproduced: save a cache at prefix 2, rewrite position 1 in the state, then call the saved cache's
`rollback_to(state, branch, 2)`. The truncation changed nothing, `resync` overwrote the lineage from
the state, and the stale bytes then passed `check_owner` and carried a continuation to valid logits.

The defect was a single "current" stamp that anything could reassign. A cache now records the lineage
of **every prefix at the moment it reached that length**, `stamps[n]`, and:

- `commit` is `pub(crate)` and only the interpreter calls it, after a step succeeded and the state
  advanced — the stamp for a prefix is written when the bytes for it are;
- `rollback_to` **checks before mutating**, comparing the stamp it recorded for the target prefix
  against what the state says now, and refuses when those positions were rewritten;
- `resync` is gone. There is no public way to change a cache's identity without adding the contents
  that justify it.

Making `resync` private alone would have left the `rollback_to` path, as the review said.

### S2 — SwiGLU's underflow contract still failed for normal outputs · closed

Reproduced: gate `−104` with `up` near `1e30` returned `0`; the reference is `−7.08791e−14`. The
sigmoid underflows in FP32 while the product is comfortably normal, so neither the relative bound nor
the subnormal floor applied — the floor was about the *intermediate*, and the contract is about the
*output*.

The operation is now evaluated in FP64 and narrowed once, which is also the more faithful reading of
the rounding table: SwiGLU's single declared rounding is on the product, and rounding the
intermediate was a boundary the contract never named. The bound tightens to `γ(2)` as a consequence,
with the absolute floor now correctly conditioned on the output being subnormal rather than the
intermediate. A test sweeps 512 gates with large `up` values and asserts the bound on every normal
output, failing if fewer than 100 of them are normal so the fixture cannot go vacuous.

### S3 — operand precision was checked against the declaration, not the contract · closed

Reproduced: declaring an input FP32 let it into a RoPE node whose `OpContract` permits only BF16
activations; construction and execution both succeeded. Agreement between a binding and its
declaration is not agreement with the operation that consumes it.

`GraphBuilder::node` now checks each operand's declared precision against the node's contract, by
role — weights against `contract.weights`, activations against `contract.activations`, indices
skipped. An operation that genuinely supports FP32 will list it; contradicting the contract silently
is what is refused.

### S4 — the attention bound omitted gradual underflow · closed

Reproduced: zero queries and keys, three visible positions, values `[2^-133, 0, 0]`. Measured
absolute error `4.671e-46` against a bound of `9.123e-48` — 51 times too small. Numerically tiny, and
still a bound that did not hold over its stated domain.

Both relative terms shrink toward zero with the data, so neither can carry a bound where the results
are subnormal. The additive `(2K + Hd + 3) · η` term does, and `metric::bound(n, scale)` makes the
same correction available to every other bound. A test asserts the fixture still demonstrates the gap
— that the relative terms alone are smaller than the measured error — so it cannot quietly stop being
a regression test.

### On using these bounds for CUDA qualification

Both numerical contracts are now stated over their whole input domain rather than over the
well-behaved part of it. Two properties a kernel gate should carry forward:

- **Attention is data-dependent.** The bound is weak where `Q·K` cancels, correctly. A kernel
  qualified against it must state its data's conditioning.
- **Both have an additive floor.** Relative bounds are silent in the subnormal range, and any gate
  written as "relative error below X" will be wrong there for the same reason these two were.


---

## Third review correction, 2026-09-08

One P1 remained: **a failed commit could leave both the sequence state and the KV contents
modified.**

Reproduced as reported. The interpreter wrote the staged appends and advanced `executed` before
calling the fallible `kv.commit`, so a graph writing one set of layers against a cache with a
different set failed *after* both had moved:

```text
invalid artifact: the cache holds 0 position(s) but the branch has executed 1
  executed:        0 -> 1
  cache lengths:   [0, 1]
  check_owner:     rejects, so there is no clean retry
```

That contradicted the interpreter's own promise that a failure leaves state untouched — the promise
the cancellation tests check at every node depth, which held only because cancellation returns before
the publication block.

Two corrections, and the first is the one that matters:

- **The mismatch is settled before anything is written.** Whether a cache covers a graph is a
  property of the two of them, not something to discover mid-commit. `run` compares
  `Graph::attention_layers()` against the cache's layer count at entry, and `GraphBuilder::finish`
  additionally requires attention layers to be numbered densely from zero — a graph using layer 1 and
  not layer 0 would leave a cache layer permanently empty, so its length could never agree with the
  frontier again. The review was right that this combination has to be rejected before mutation even
  if the slice does not support it.
- **The publication order no longer allows a half-step.** Every precondition is checked first,
  including the counter's headroom, which is the last thing that could make the irreversible
  `execute` fail. The undoable mutation (the KV appends) goes first and is truncated back on any
  error; `execute` follows it; and the two calls after `execute` have preconditions the earlier
  checks already established — `commit` because every layer received exactly `rows` appends, so the
  cache length now equals `executed`, and `record_logits` because its prefix *is* `executed`. Each
  still propagates its error and restores the cache rather than being unwrapped: if that reasoning
  is ever wrong, a failed step is a better outcome than a panic or a corrupted one.

The regression test asserts what the review asked for: unchanged counters, unchanged cache contents,
no retained result after the failure, and a clean successful execution afterwards. A second test
covers the graph-construction half.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 309 + 1 doctest | **311 + 1 doctest** |
| device lane | 317 + 2 doctests | **319 + 2 doctests** |
| `fmt`, `clippy -D warnings`, `arch-check`, `spec-check`, no-driver host build | PASS | **PASS** |
| `cargo xtask-cuda test-gpu`, hidden-SM120 gate | PASS / exit 1 | **PASS / exit 1** |


---

## Fourth review correction, 2026-09-08

The layer-mismatch path was closed, but **the same atomicity defect was still reachable through a
graph with no attention nodes**. Reproduced as reported, with a valid
`RoPE → VocabProjection` graph and a zero-layer cache:

```text
invalid artifact: the cache holds 0 position(s) but the branch has executed 1
  executed after failure: 1
  check_owner:            rejects
```

The coverage check compared counts, and zero equals zero, so it passed. My previous note that
post-`execute` failures "restore the step" was inaccurate and the review was right to say so: the
handler restores the **cache only**. There is no `unexecute`.

Two corrections. The second is the one that removes the class rather than the instance:

- **A stateless graph is refused at entry.** A step publishes sequence state; a graph that touches
  none has nothing to publish, and advancing `executed` past a cache that can never hold anything is
  incoherent. Attention-free execution is a reasonable thing to want — it simply does not advance the
  frontier, which makes it a different operation with different publication rules. Refused here
  rather than half-supported.
- **The last precondition is now checked, not inferred.** The previous version argued that `commit`
  could not fail because "every layer received exactly `rows` appends" — an argument that is
  *vacuously true* when there are no layers, which is exactly how this got through. After the appends
  and while the state is still untouched, the interpreter now compares the cache's length and
  coherence against the prefix the counter is about to reach, and restores the cache and returns if
  they disagree. Whatever else is wrong, `execute` is not reached with a cache that cannot satisfy
  `commit`.

The remaining `?` after `execute` are propagated rather than unwrapped, and the code says plainly
what that means: a failed step beats a panic if the reasoning above is ever wrong, but the state may
then be advanced and restoring the cache does not change that. That is a limitation of `moxie-state`
having no rollback for `executed`, and it is M1.4's transaction API that should remove it — at which
point the two separately-argued "leaves state untouched" paths, cancellation and publication, become
one structural guarantee.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 311 + 1 doctest | **312 + 1 doctest** |
| device lane | 319 + 2 doctests | **320 + 2 doctests** |
| `fmt`, `clippy -D warnings`, `arch-check`, `spec-check`, no-driver host build | PASS | **PASS** |
| `cargo xtask-cuda test-gpu`, hidden-SM120 gate | PASS / exit 1 | **PASS / exit 1** |
