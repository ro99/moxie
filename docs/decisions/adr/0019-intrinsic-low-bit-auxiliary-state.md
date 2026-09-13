# ADR 0019 — Intrinsic low-bit auxiliary state is mathematics, not cache compression

- ID / date / author / status: 0019 / 2026-09-13 / owner-directed, recorded by implementation agent / adopted.
- Classification: owner requirement (cache policy exception). It narrows the "cache >=16 bits" product gate to its intended scope.
- Scope and owning shared component: `moxie-graph` / `moxie-model-api` (shared semantic operations, oracles), `moxie-state` / `moxie-memory` (state schemas, residency), `moxie-kernels` (layout-specific paths). No model crate gains a private cache.
- Supersedes / superseded by: refines document 01 product gate "No weights below four bits; cache no lower than 16 bits" and document 03 § BF16/KV and document 04 § MLA/state. See [O4](../owner-gates.md#o4--intrinsic-low-bit-auxiliary-state).

## Problem and mechanism

Product documents fix weight family to INT4/INT8/BF16 and cache/state to >=16 bits (BF16/FP16/FP32). This prevents generic KV/cache compression as a capacity shortcut. Separately, several families define a low-bit auxiliary representation as part of their released mathematics — compressed/sparse MLA state, sparse index selection, absorbed layouts, Engram table dtypes/layouts (ADR 0009) — where the low-bit encoding is not an optimization but the equation (e.g., index bit-width, hash-table dtype). Document 01 O4 asks whether the cache floor forbids even that.

Without a ruling, enabling such a representation is blocked: the default is physical cache >=16 bits and reporting infeasibility/fidelity failure rather than taking a silent exception.

## Options examined

**A. Forbid even intrinsic (status quo default).** Every low-bit representation below 16 bits is forbidden, even when the released model defines it. Rejected by owner ruling: it would make correct execution of families that define such state impossible, not merely expensive.

**B. Allow intrinsic when it is mathematics (selected).** A model-defined low-bit auxiliary representation is allowed when it is part of the released mathematics and is carried as a shared semantic operation with equation, oracle, shape/precision/state/partition contract, and second-consumer or synthetic coverage per document 02's enforced extension rule. Generic KV/cache compression below 16 bits (reducing a BF16/FP16 cache to INT8/FP8/FP4 to save capacity) remains forbidden and has no path.

**C. Per-family deferral.** Decide at each bring-up with evidence. Rejected as standing policy: the principle is now fixed; per-family evidence still required, but the gate is no longer OPEN.

## Decision and authority

1. **Intrinsic low-bit auxiliary state is allowed when it is mathematics.** Examples in scope: MLA latent/positional layouts, absorbed output/query paths that require a specific index bit-width, sparse/compressed state where the index or table dtype is defined by the released model, Engram hash-table dtype/layout/version and collision semantics (per ADR 0009). Each requires a shared operation, independent oracle, and capability/state tests before any family graph consumes it.

2. **Generic cache compression stays forbidden.** Reducing physical KV, recurrent, or attention state below 16 bits as a capacity optimization is not allowed and has no implementation path. "Intrinsic" is not a loophole for it.

3. **Ownership unchanged.** Allowed low-bit state is still owned by shared crates, never by `moxie-models-*`. A model crate that implements a private low-bit cache fails `arch-check`.

Authority: owner ruling on O4 (2026-09-13, verbatim). This ADR records it; O4 is RESOLVED.

## Evidence and acceptance

- O4 ruling verbatim in `owner-gates.md`.
- Families that will use this: DeepSeek (compressed/sparse state, MLA, mHC), GLM-5.2/5.3 (MLA, sparse index, absorbed), Engram conditional memory (post-start, ADR 0009). None is implemented; this ADR unblocks their bring-up, not their claim.
- Acceptance: O4 → RESOLVED. Future bring-ups that need intrinsic low-bit state must cite this ADR and bring equation-linked fixtures, oracle, and affected-consumer reruns.

## Enforcement and removal

- `arch-check` must reject model-owned low-bit caches/allocators; shared kernels dispatch by capability/shape, never model name.
- Generic cache <16-bit remains a permanent product rule. Re-evaluate only if the owner explicitly rescinds this ruling through a new O4 ruling and ADR.
