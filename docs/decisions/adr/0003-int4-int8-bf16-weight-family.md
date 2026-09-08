# ADR 0003 — INT4 / INT8 / BF16, shared affine-integer weights

- Date: 2026-09-07.
- Status: adopted design direction; implementation pending.
- Authority: owner explicitly replaces the earlier NVFP4 preference with INT4, INT8 and BF16 and requests the specification changes. The detailed affine representation is the design recommendation implementing that direction.
- Supersedes: document 03's NVFP4 canonical family and symmetric-only, per-channel INT8 v1 profile. Historical measurements remain historical evidence, not current requirements.

## Decision

Use one versioned affine-integer schema, with 4-bit and 8-bit weight profiles, plus BF16 plain weights. Keep 16-bit activations (BF16 preferred; FP16 where the source contract requires qualification) and FP32 accumulation. No FP4/FP8 runtime family in the initial release. No W4A4 or W8A8 activation quantization in the initial scope. Cache remains at least 16 bits.

W4A16 specifies operand widths, not a file layout or a quantization algorithm. AWQ and AutoRound describe quantization methods; compressed-tensors and AutoGPTQ-style packing describe serialization. Import them into the same mathematical weight contract; never create an AWQ model runtime and an AutoRound model runtime.

Preserve symmetry/zero points, group size, scale values and logical column identity when repacking. The requested checkpoint metadata includes symmetric group-128 INT4, asymmetric group-32 INT4 and symmetric group-32 INT8. A symmetric-only schema or per-channel-only INT8 schema cannot represent these without changing values. See the pinned [metadata evidence](../../evidence/quantization-candidates.md) and revised [document 03](../../spec/03-memory-formats-and-cuda.md).

One format does not mean one hard-coded group size. A small, closed parameter set belongs in a shared descriptor. It does not authorize arbitrary future formats or per-model codecs. Prefer a value-preserving import/repack over requantization. A requested checkpoint name is a candidate, not an approved release catalog or download authorization.

## Hardware rationale and limits

Ampere lacks native FP4/FP8 tensor-core arithmetic. INT4/INT8 packed weights with fused dequantization and BF16/FP16 tensor-core multiplication are a practical initial target with established kernel techniques. **W4A16 does not directly use an INT4-times-INT4 MMA instruction**, which requires low-bit activations too. INT4 weight-only is not automatically faster than every software NVFP4 path; measure the actual shared kernels and transfer bottleneck. This is a scope/ecosystem decision, not an unmeasured throughput claim.

The common graph, memory authority, expert streaming, Flash attention, TP/PP, sequence transactions, speculation, future entropy, HTTP and CLI boundaries remain unchanged. Weight descriptors/importers/kernel qualification and their tests change. No second engine or second rewrite is needed.

## Required implementation changes

Follow the [M0 review and correction task](../../tasks/0002-m0-review-and-integer-transition.md). Replace the active NVFP4 precision variant/codec with the shared integer representation; preserve the old experiment and Git history. Do not merely rename E2M1 to INT4. Expand INT8 to the same affine/grouped semantics, including valid -128 codes; do not mistake a chosen quantizer's symmetric clipping range for a universal decoder rule.

The owner has resolved precision-family preference, not O2 quality thresholds, O1 catalog/order, O4 intrinsic low-bit state or O5 bulk storage/download authority. Those gates remain open at their actual dependent work. Do not ask again whether NVFP4 should be the initial runtime family.

## Acceptance

Exhaustive signed integer decoding, signed/unsigned rebias equivalence, zero points, group-32/128/per-channel boundaries, scale-dtype preservation, activation-order permutations, partial groups, checked lengths/overflow and bounded preparation. Prove source-to-canonical reconstructed weights are unchanged under the declared source arithmetic when an import is called lossless. Test independent synthetic consumers and actual licensed metadata/weight samples before claiming checkpoint support. Qualify dense and grouped MoE kernels separately on SM86 and SM120, at decode and prefill row counts.
