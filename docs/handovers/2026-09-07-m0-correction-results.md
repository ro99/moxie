# Handover — M0 corrections and the integer weight family, implemented

Date: 2026-09-07. From: implementation agent (Claude). Base commit:
`84273b0e4b41bb04d1b374f6f46f89557bba4a59`, branch `main`.

The assignment is in [the previous handover](2026-09-07-claude-m0-corrections.md); the findings it
refers to are [task 0002](../tasks/0002-m0-review-and-integer-transition.md), which now also carries
the full results. This file is the short version: what changed, what to check first, and what is
deliberately still missing.

## State

M0's correction gates F1–F6 are closed. The M0 scaffold was kept; nothing was restarted. No
checkpoint was downloaded, converted or executed. No model runtime, no FP4/FP8/W4A4/W8A8 path, no
cache below 16 bits, no throughput claim.

**M0 is not "closed" by fiat.** Its exit gate now reads honestly:

| M0 exit gate | Status |
|---|---|
| Reproducible workspace build | met — 191 host tests, clippy clean, fmt clean |
| Runnable CUDA-free host/architecture CI | met — verified with no toolkit or driver reachable |
| One real CUDA launch on each installed architecture | met — 15 cases, sm_86 and sm_120 qualified |
| Dependency violations demonstrably rejected, including target-specific/renamed/custom-build-path | met — 13 negative fixtures, each asserting its own rule |
| Reference manifest, required tiny fixtures and owner-gate register committed | met — `moxie-oracles`, `docs/evidence/` |
| No production quality or speed claim | met — none made |

## Run these first

```text
cargo xtask spec-check     # do you actually have the specification?
cargo xtask arch-check     # 13 negative fixtures, 6 rules
cargo test --workspace     # 191, no CUDA needed
cargo xtask-cuda test-gpu  # 15 cases on real hardware; needs the toolkit and the cards
```

`cargo xtask` is the host lane and `cargo xtask-cuda` is the device lane. They are different builds
on purpose, and running the host checks through the device build leaves the host lane's
independence untested.

`spec-check` is new and matters most in a fresh clone: `docs/spec/01-09` is kept local and untracked
at the owner's direction, so a clone does **not** have the normative documents. Nothing previously
noticed. It reports presence and digests; it does not fetch, copy or generate anything, and the
publication policy is unchanged.

## What changed, in one line each

- **Host lane** — `moxie-cuda/driver`, `moxie-kernels/fatbin` and `xtask/cuda` are default-off; the
  `CUresult` classification moved to a driver-free `moxie_cuda::status`.
- **arch-check** — resolves package *identity*, reads target-specific tables, follows workspace
  inheritance, honours `build = "..."` at any path, and each fixture declares the rule it proves.
- **CUDA images** — `Module::load` takes a `TrustedImage` (constructed `unsafe`, with the trust
  obligation written down) or a `&CStr` of PTX; context guards and retain-failure cleanup fixed.
- **State** — four counters per branch, execution may legitimately run ahead of acceptance, and
  logit validity is a retained `(branch, prefix, generation)` result rather than counter equality.
- **Precision** — `Precision::Nvfp4` replaced by `Int4`/`Int8`; role types `WeightPrecision`,
  `ActivationPrecision`, `CachePrecision` make W4A4 unrepresentable rather than merely discouraged.
- **Format** — `affine.rs` implements document 03's `(q - z) * s` for both widths; `nvfp4.rs`
  deleted; `int8.rs` refactored into the shared decoder plus a `quantize.rs` policy that owns the
  symmetric clipping choice.
- **Fixtures** — new `moxie-oracles` crate: routing, masks, recurrence, sampler distributions,
  tokenizer/template and protocol frames, 60 tests.
- **GPU lane** — a stream/event completion smoke, a non-PTX rejection case, recorded nvcc/host
  compiler/image identity, and a gate that **fails** when a required architecture has no passing
  device.

## Read these before assuming anything

- Three bugs were found by the new tests and fixed: an inherited `path` resolved against the wrong
  root (which made every workspace dependency look like a violation), an FP16 subnormal decoder that
  halved every subnormal scale, and an asymmetric quantizer that clipped before applying the zero
  point and collapsed a positive-valued group onto the lowest code. Each has a regression test.
- `moxie-graph`'s oracle registry records *where* an independent implementation lives. That is
  weaker than executing it, and the module says so. Task 0003 is where it becomes callable.
- The sampler fixture implements the legality mask, top-k, top-p, min-p and temperature. Penalties,
  DRY, n-gram ban, logit bias, typical-p, XTC and future entropy are **not implemented**, and
  `Pipeline` lists them as absent rather than carrying inert fields.
- Every module that pins a partial contract states what it does not pin. Take those lists literally.

## Still open, unchanged by this work

**O1** (release catalog and bring-up order), **O2** (acceptable quality loss), **O4** (intrinsic
low-bit auxiliary state) and **O5** (storage and download authorization) are all OPEN. Nothing here
selects a catalog, approves a quality loss, enables low-bit state or writes bulk storage. The
precision-family question is resolved by ADR 0003 and is not to be reopened.

The ten [candidate checkpoints](../evidence/quantization-candidates.md) have inspected metadata and
nothing more. The host decoder accepting synthetic tensors of the right shape is not evidence that
any of them loads: their tensor headers, zero-point packing, `g_idx` maps and exclusion lists are
unread. The M3 importer tasks are where that changes.

No CI runner executes the device lane. Its results are hand-run, and recorded with the hardware they
were measured on.

## Next

[Task 0003](../tasks/0003-m1-bf16-reference-interpreter.md) — the BF16 host reference interpreter.
It is deliberately smaller than document 06's M1: the manifest reader, the rank-owned CUDA path,
paged state and the generation service are separate tasks that consume it. Its contract section must
be filled in, with error metrics declared, **before** any code is written.
