# 0001 — Both NVFP4 global-scale conventions, confirmed against real weights

Date: 2026-09-07. Milestone: M0. Status: **accepted**; the finding stands, the code it used is
retired.

**Retirement note, 2026-09-07.** [ADR 0003](../../decisions/adr/0003-int4-int8-bf16-weight-family.md)
replaced NVFP4 with the INT4/INT8/BF16 affine-integer family, and
`crates/moxie-format/src/nvfp4.rs` — the E2M1 and E4M3FN decoders this experiment used — was removed
from the active API by [task 0002](../../tasks/0002-m0-review-and-integer-transition.md). It is in
git at commit `84273b0e4b41bb04d1b374f6f46f89557bba4a59`, path
`crates/moxie-format/src/nvfp4.rs`, with its tests.

The finding below is **not** retired. R16 is about reading a stored scalar in the direction its
exporter meant, and the integer importers face the same question with their own zero-point and scale
conventions: document 03 requires GPTQ-style stored zero offsets and packed axis order to be
"decoded according to the pinned exporter, never guessed from a suffix". The lesson transfers even
though the format did not.

## Hypothesis

R16 states that "GLM's documented NVFP4 decoding divides by a global scale;
Inkling's ModelOpt path multiplies", and warns that "both say FP4" is not enough
to establish compatibility. Both conventions are present on this machine, in the
same architecture family. If the direction is read backwards, every weight in the
tensor is off by the square of the global scale — silently, with no format error.

## Method

Decoded one real expert weight tensor from each checkpoint using the canonical
equation from document 03, with an E2M1 table and an E4M3FN decoder written from
the format definition. Applied the stored global scale both ways and compared the
resulting weight distributions. Plausible neural-network weights have a standard
deviation on the order of 0.01–0.1; a wrong direction is off by orders of
magnitude, which makes the two hypotheses trivially separable without needing a
reference dequantization.

Tensors read directly from safetensors, read-only, no conversion.

## Result

| Checkpoint | Producer | Tensor | Stored global | As multiply | As divide |
|---|---|---|---|---|---|
| `/fast/models/incoai/GLM-5.3-NVFP4` | ModelOpt 0.45.0 | `layers.10.mlp.experts.100.gate_proj.weight` | 3.57855e-05 | **std 0.01772**, absmax 0.0962 | std 1.38e7, absmax 7.5e7 |
| `strata/models/glm53f-nvfp4` | compressed-tensors | `layers.15.mlp.experts.28.down_proj.weight_packed` | 26240 | std 1.26e7, absmax 7.1e7 | **std 0.01827**, absmax 0.1024 |

**ModelOpt multiplies. compressed-tensors divides.** R16 is confirmed on real
data, not inferred from documentation.

Corroborating detail: `1 / 26240 = 3.81e-05`, against ModelOpt's stored
`3.58e-05`. Once normalised, the two checkpoints carry global scales of the same
magnitude, and the two decoded tensors have nearly identical statistics
(std 0.0177 versus 0.0183). Two different conventions, one underlying quantity.

## Also found: the tensor names differ too

| Role | ModelOpt | compressed-tensors |
|---|---|---|
| Packed codes | `<proj>.weight` | `<proj>.weight_packed` |
| Block scales | `<proj>.weight_scale` | `<proj>.weight_scale` |
| Global scale | `<proj>.weight_scale_2` | `<proj>.weight_global_scale` |

An importer keyed on `weight` finds nothing in a compressed-tensors checkpoint,
and one keyed on `weight_scale_2` silently finds no global scale. Document 03
requires the importer to "absorb source naming and packing differences so runtime
code does not"; this is the concrete list for these two.

## What this establishes, and what it does not

Establishes: the direction of each convention, on these two checkpoints, with the
tensor names each uses. `moxie-format::nvfp4::canonical_tensor_scale` implements
the normalisation and its test asserts that a stored 4.0-as-divisor and a stored
0.25-as-multiplier produce the same canonical multiplier.

Does **not** establish: that the full dequantized tile matches a reference
implementation. Document 03 requires validating "complete dequantized tiles and
model outputs", and a distribution check is weaker than that. The absmax values
(0.096 and 0.102) are consistent with correct block-scale application but do not
prove the group-to-column mapping is right; a transposed or mis-strided block
scale would still produce plausible statistics. That is M3 work.

Also not established: whether the low/high nibble order used here matches these
producers. The unit test pins the order document 03 specifies; agreement with
these files was not separately checked, and a swapped order would also yield a
plausible distribution. **M3 must check nibble order against a reference
dequantization, not against statistics.**

## Reproduction

Read-only, a few seconds, no GPU. Decode any packed expert tensor from either
checkpoint with the equation in `moxie-format::nvfp4`, applying the stored global
scale both ways, and compare standard deviations.
