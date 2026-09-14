# ADR 0028 — A quantized reduction's numerical gate has two clauses

- **ID / date / author / status:** 0028 / 2026-09-14 / recorded by the implementing agent on the owner's ruling / **accepted**
- **Classification:** **owner requirement.** A numerical gate was widened after a result was measured, which AGENTS.md reserves to the owner; the measurement that prompted it is below, and the agent asked before writing any acceptance test around either form.
- **Scope and owning shared component:** the acceptance criterion for any shared kernel that reduces over a quantized weight against a host oracle — [task 0028](../../tasks/0028-m3-shared-w4a16-w8a16-execution.md)'s dense W4A16/W8A16 linear today, and the quantized expert path next. It changes no operation contract, no precision, no rounding boundary and no layout.
- **Supersedes:** the single-clause threshold written in task 0028's contract before implementation. Nothing else.

## Problem and mechanism

Task 0028's contract predeclared, before implementation: for each output
element, `|y_kernel − y_oracle|` ≤ **2 ULP of BF16 at the oracle's magnitude**.
The oracle reconstructs the whole weight on the host and multiplies in ascending
order with FP32 accumulation; the kernel dequantizes into 16x16 tensor-core
tiles and accumulates in whatever order the hardware's MMA uses. The two differ
only in the **order of the additions** — the products are exact, because a BF16
times a BF16 has at most 16 significand bits and FP32 holds 24.

Measured at 33 by 1,024 by 3,072 with uniformly drawn codes and zero points,
**37 of 101,376 elements missed the threshold, worst 6 ULP**. The worst
element:

| Quantity | Value |
|---|---|
| oracle result | 2.21729279e-5 |
| kernel result | 2.28881836e-5 |
| `Σ|x_k · W_k|` over its reduction | 44.8413914 |
| result / term sum | 4.94e-7 |
| kernel error / term sum | **1.6e-8** |

The kernel's error is **1.6e-8 of the reduction's own scale**, below one FP32
epsilon (1.19e-7). What failed is not the kernel: the result cancelled to six
orders of magnitude below the terms that produced it, so "one ULP of the
result" became a quantity smaller than any reordered FP32 sum can control. **No
implementation of this operation can satisfy the single-clause form in
general**, because the clause's denominator collapses on a cancelling output.

## Options examined

| Option | Semantics | What it asserts | Risk |
|---|---|---|---|
| **Keep 2 ULP as written** | unchanged | a real bound on well-conditioned outputs | unsatisfiable on cancelling outputs by any implementation; task stops at a gate no kernel can pass |
| **Replace with `2^-8 · Σ|x·W|` only** | one rule | the reduction's own resolution | stops being a statement about the output's precision where the output *is* well conditioned; a blanket relative tolerance |
| **Both clauses, either sufficient** (chosen) | 2 ULP at the result, **or** `2^-8 · Σ|x·W|` | the strict bound wherever it is meaningful; the reduction's own resolution where the output has cancelled below what BF16 can express | a second clause could become a silent blanket tolerance if nothing measures how often it fires |

`2^-8` is not a tuned constant: BF16 carries eight significand bits, so
`Σ|x·W| / 256` is the smallest difference a BF16 *output of that reduction*
could express at all. An element the second clause covers is one whose BF16
value carries no information about the difference being measured.

## Decision and authority

**Owner ruling, 2026-09-14.** An output element passes when

```text
|y_kernel − y_oracle| ≤ 2 · ULP_bf16(y_oracle)
        or  |y_kernel − y_oracle| ≤ 2^-8 · Σ|x_k · W_k|
```

Both bounds are computed and both are reported by every test that applies it.
Narrowing or widening either clause again is the **owner's** call and not a
task's, exactly as the original threshold was.

This resolves no O-gate. **O2 is untouched**: agreement with a host decoder over
synthetic activations is not evidence about model output, and O6/O7 are
untouched because nothing here is timed.

## Evidence and acceptance

- The measurement above, taken before the ruling was requested and before any
  acceptance test was written around either form.
- **The second clause fires nowhere in task 0028's fixtures.** Every element of
  its five synthetic cases and of the real published module passes the first
  clause alone, worst **2.000 ULP**. The clause is therefore exercised by a
  direct test — `the_cancellation_clause_covers_a_cancelled_element_and_nothing_else`
  in `crates/moxie-executor/tests/affine_linear_device.rs` — which drives it
  with the numbers in the table above, asserts that the fixture still misses the
  first clause, and asserts that the **same difference on a well-conditioned
  reduction is still refused**.
- Rejected: searching for a fixture seed that happens to exceed the first
  clause. That is fitting the evidence to the guard, and this repository's log
  already carries the shape.

## Enforcement and removal

- The two clauses live in the comparison helpers of
  `crates/moxie-executor/tests/affine_linear_device.rs`,
  `crates/moxie-executor/tests/affine_linear_real_module.rs` and the
  `affine_linear_w4a16_w8a16` case in `xtask/src/gpu.rs`. A kernel that reduces
  over a quantized weight uses this criterion; one that does not reduce has no
  business citing it.
- **Re-evaluation trigger:** a fixture in which the second clause covers a
  non-trivial share of elements. That would mean either a real numerical
  regression or a fixture whose outputs are systematically cancelling, and both
  are worth reporting rather than absorbing. Every test using the criterion
  reports the count for that reason.
- **Removal:** if a future kernel reproduces the oracle's reduction order
  exactly, it is bitwise equal and needs neither clause. The criterion exists
  for reordered reductions and should not outlive them.
