# Task 0003 — M1, part 1: the BF16 host reference interpreter

Status: **proposed**. Proposed 2026-09-07, after the [M0 correction
task](0002-m0-review-and-integer-transition.md) closed F1–F6. Do not start it before that record's
gates are green in the reader's own checkout: `cargo xtask arch-check`, `cargo xtask spec-check`,
`cargo test --workspace`, and `cargo xtask-cuda test-gpu` on real hardware.

This is the **smallest** M1 slice. It is not "the M1 vertical slice": document 06 M1 also wants a
manifest reader, a rank-owned CUDA execution path, paged state and a generation service. Those are
separate tasks that consume this one. Attempting them together is how a slice becomes a campaign.

## Identity and authority

- Task ID / milestone / owner: 0003 / M1 / **to be filled by the implementing agent**
- Writable root: `/home/rodrigo/Developer/moxie`, branch and base commit **to be filled**
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

## Contract before implementation

Write this section fully **before** any code. Document 07: "the implementation task must specify its
absolute/relative or normalized error metric, independently generated reference, relevant scales and
threshold before optimization. ... agents must not choose a threshold after seeing a failing
candidate."

- Equations for each operation, in terms document 02 fixes. `RmsNorm` and `LayerNorm` stay distinct.
  `SwiGlu` is not `SituGlu`.
- Shapes, precision and accumulation per operation: BF16 storage and activations,
  `AccumulationPolicy::Bf16InF32Acc` throughout, FP32 for the norm reduction and the attention
  softmax. Use `WeightPrecision` / `ActivationPrecision` — the role types exist for this.
- The interpreter computes in FP32 internally and rounds to BF16 at declared boundaries. **Which
  boundaries is a semantic decision, not an implementation detail**: state it per operation, because
  it is what a fused kernel will later have to match.
- State effects: `Attention` appends to KV state through `moxie-state`, using the four counters and
  the logit provenance rules that task 0002 established. A step that produces logits records them.
- Partition: `PartitionRule` per operation, or `NotDetermined` with a reason. TP is M5; declaring
  the rule now is what makes M5 a lowering rather than a rewrite.
- Cancellation and failure: a cancelled step releases its state transaction at a safe boundary and
  leaves the accepted prefix untouched (R08). No partial commit.
- Oracle: each operation's reference goes in `moxie-oracles` and is registered there. The
  interpreter is a *consumer* of the registry, not a second implementation. Where the interpreter
  and the oracle would be the same code, say so explicitly rather than pretending to two
  independent implementations — and make the *test* the independent one, computing the expected
  value from the equation in the test body.

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
