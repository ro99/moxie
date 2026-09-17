# ADR 0030 — preserve signed affine scales exactly

- **ID / date / status:** 0030 / 2026-09-17 / accepted, owner ruling.
- **Scope:** shared canonical affine scale validation, import, publication and execution.
- **Supersedes:** document03's positive-only affine scale restriction. No other scale contract changes.

## Evidence and decision

Task0034's read-only GLM AutoRound sample at catalog revision
`5eee1846f0321058ed73745f9aa16f2aaf0fc0a0` contains a negative F16 scale:
`-0.005203247` in the first sampled down-projection row. The current reader
rejects it before conversion. Positive-only validation conflicts with preserving
this released source's scale value and encoding.

The owner explicitly ruled: **“Preserve signed scales exactly (recommended)”**
on 2026-09-17 after being shown this evidence and the proposed contract change.
Canonical affine scales are therefore **finite and nonzero, of either sign**.
Preserve every source bit in F16/BF16/F32; do not take absolute values, flip
codes, requantize or round the scale to another encoding. Reject either signed
zero, NaN and infinity. An explicit future zero-block normalization needs its
own source contract; this ruling does not authorize it.

`W=(Q-Z)*S`, wide integer subtraction, logical column identity, BF16 operand
rounding and FP32 accumulation remain unchanged. ADR0028's numerical acceptance
gate is unchanged. Signed scales add no precision family, serialization or
model-specific execution path. This is value-preserving source support, not
model-output or performance evidence.

## Validation

Exercise both signs in the three scalar encodings through validation and the
bounded scale writer; retain zero/nonfinite refusals. Re-run bounded pinned
AutoRound source samples and affected host tests. Execution qualifications must
exercise signed scales before claiming them. Task0034/task0035 carry results.
