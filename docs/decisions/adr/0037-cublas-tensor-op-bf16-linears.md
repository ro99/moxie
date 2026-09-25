# ADR 0037 — BF16 linears may run on cuBLAS tensor-op; selection projections keep the declared order

- **ID / date / author / status:** 0037 / 2026-09-25 / recorded by the coordinator on the owner's ruling / **accepted**
- **Classification:** **owner requirement.** It adopts a third-party library and extends a numerical gate, and AGENTS.md reserves both to the owner. The owner's ruling (2026-09-25) answered two questions with "yes to both".
- **Scope and owning shared component:**
  - BF16 dense `Linear`, `LinearSplit` and `LinearPartial` in the dense step (`moxie-executor/src/dense.rs`);
  - the cuBLAS binding in `moxie-cuda`;
  - the descriptor in `moxie-kernels`;
  - selection in `moxie-plan`.

  It does not change the affine INT8/INT4 kernel (already tensor-core, ADR 0028), `Route`, `VocabProjection` or attention.
- **Supersedes / amends:** it extends [ADR 0028](0028-quantized-reduction-numerical-gate.md)'s two-clause gate to BF16 linears on this path. [ADR 0036](0036-tp-reductions-are-exact-by-declared-order.md) is unchanged and still binding.

## Problem and mechanism

Task 0095 measured Moxie's BF16 dense linear
(`moxie_dense_linear_split_v1`, one thread per output, sequential FP32 sum) at
0.055 TFLOP/s on an RTX 3090 and 0.109 on the RTX 5060 Ti. That is about
three orders of magnitude below the hardware. The kernel is slow because it
adds in exactly the declared sequential order, which is what makes it
bit-identical to the host reference. Tensor-core matrix multiply adds in the
hardware's order, so it cannot be bit-identical to that reference.

## What Strata did (the evidence the ruling rests on)

- Its main matrix-multiply paths ran on cuBLAS in tensor-op mode
  (`strata/kernels/cuda/detail/backend_core.inc.cuh` 162–175:
  `cublasSetMathMode(…, CUBLAS_TENSOR_OP_MATH)`), on the vendored Marlin
  tensor-core kernel (`GemmaMarlin` route), and on its own FP8 tensor-core
  page kernels.
- It kept the exact sequential FP32 order only where last-bit noise changes
  a decision. The DeepSeek indexer's scores choose the rows attention reads,
  and "the tensor-core path … would reassociate the accumulation and could
  change which rows are selected near the score threshold — a semantic
  change, not a rounding one" (`strata/docs/dsv4-rank-local-architecture.md`
  891–897).
- Document 03 (line 151) sets the reuse order "existing Strata proven kernel
  → compatible pinned upstream kernel/library → new shared kernel", and
  names cuBLASLt as the resident GEMM baseline.

## Options examined

- **An exact tiled kernel of our own.** It stays bit-identical to the host,
  and is the slowest option.
- **Our own tensor-core kernel.** No new dependency; slower than cuBLAS
  unless heavily tuned.
- **cuBLAS tensor-op (chosen).** The fastest and most mature option, and
  Strata's choice on this hardware. It is also the least new kernel code.

## Decision

1. **BF16 `Linear`, `LinearSplit` and `LinearPartial` may execute with
   `cublasGemmEx`**: BF16 A and B, compute type `CUBLAS_COMPUTE_32F`, and
   the default algorithm. The workspace is plan-owned, admitted and fixed.
   Such a descriptor declares `AccumulationPolicy::Bf16InF32AccUnordered`
   (FP32 accumulator, hardware order).
2. **Its numerical gate against the ordered host oracle** is ADR 0028's two
   clauses per output element: 2 BF16 ULP at the oracle's magnitude, or
   `2^-8 · Σ|x·W|`.
3. **ADR 0036 still holds.** A TP partial (`LinearPartial`, FP32 output) and
   the single-device `LinearSplit` reference both compute each declared
   block with **identical cuBLAS call parameters**, meaning the same m, n,
   k, leading dimensions, types, algorithm and math mode. The reference
   therefore packs each block contiguously, as a TP rank holds it; a
   strided view of the full matrix is a different call (sol's design
   review, task 0096). The blocks are combined in FP32 in
   the declared order and rounded once. TP stays bit-identical to single-GPU
   on GPUs of one architecture and SM count, which is cuBLAS's documented
   reproducibility condition.
4. **Selection projections keep the declared order.** `Route` (router logits
   that choose experts) and any future sparse-index scoring are separate
   operations. They never select an unordered descriptor. `VocabProjection`
   also stays on its ordered kernel, because its logits feed sampling.
   Moving it needs its own ruling.
5. **The choice is made per plan by the catalogue** that the composition
   root injects. A catalogue holds either the ordered or the unordered BF16
   linear descriptors, never both for one operation. The descriptor id is
   part of plan identity, as ADR 0036's split count is. End-to-end tests
   that assert bit-exact logits keep the ordered catalogue. The unordered
   path is validated per operation (clause 2) and for TP identity
   (clause 3).
6. **Dependency.** `moxie-cuda` links `libcublas` from `CUDA_HOME` behind a
   `cublas` feature. The library's SHA-256 at build time is the descriptor's
   image identity. No other crate links it.

## Consequences and costs

- BF16 generated text on the unordered path can differ from the ordered
  path in the last bits, and so, eventually, in a greedy token (Strata
  measured such a flip at token 7 of 32 for a different reassociated path,
  `strata/docs/models/glm53.md` 98). That is the price of speed the owner
  accepted for BF16 linears. It is why selection projections are excluded.
- The cuBLAS handle and its workspace are device memory the ledger must
  admit (document 03's single memory authority).
- Which path is the default for users is decided with M6's exit defaults.

## Enforcement and removal

- Selection refuses a catalogue holding both orderings for one operation.
- `Route` and `VocabProjection` match only ordered descriptors.
- A test on the 3090 pair keeps TP and single-GPU bit-identical on the
  unordered path.
- Revisit if cuBLAS reproducibility is broken on a supported GPU pair, or if
  the owner rules on `VocabProjection`.
