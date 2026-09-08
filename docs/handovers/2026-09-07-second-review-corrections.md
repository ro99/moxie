# Handover — second-review corrections

Date: 2026-09-07. From: implementation agent (Claude). Base commit: `88fd983`, branch `main`.

The first correction pass is in [its own handover](2026-09-07-m0-correction-results.md). A second
review reproduced those results and then found six contracts that still accepted an invalid state.
Its verdict — "substantially better, but I would not close M0 yet" — was right. This pass closes
those six. Full detail, with the reproductions, is in
[task 0002](../tasks/0002-m0-review-and-integer-transition.md#second-review-corrections-2026-09-07).

## State

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | PASS |
| `cargo test --workspace --locked` | PASS, **210** tests (was 191) |
| `cargo xtask arch-check` | PASS, **15** negative fixtures (was 13), 6 rules |
| `cargo xtask spec-check` | PASS, 10 documents, digests unchanged |
| host build with no CUDA toolkit or driver reachable | PASS, 210 tests; `ldd` shows no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | PASS, **218** tests + 1 doctest |
| `cargo xtask-cuda test-gpu` | PASS, 15 cases, `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | exit 1, as intended |

Every one of the six findings was **reproduced first**, then fixed, then covered by a regression
test that fails against the previous behaviour.

## What changed

- **Retained results have an identity.** A `(branch, prefix, generation)` triple was not one: a
  handle from sequence A was accepted by sequence B, and a handle for a discarded suffix became
  valid again after re-executing to the same length. There is now a process-unique `SequenceId`, an
  opaque `ResultId` with no public constructor and a ledger of issued results, and a `PrefixLineage`
  that changes when a suffix is replaced and does not change for prefixes the rollback did not
  touch.
- **Restore evidence names what it completed.** A snapshot taken at prefix 4 no longer satisfies a
  rollback to prefix 6. `RestoreMethod::{Snapshot { of_prefix }, Replay { from, to }}` must finish
  *at* the target, on this sequence, under this generation.
- **arch-check classifies imports structurally.** `use std::{fs};` contains the text `std::fs`
  nowhere and used to pass. `use` declarations are now flattened into fully-qualified paths, so
  groups, renames and globs all land on one rule. Production sources come from declared targets, so
  `[lib] path = "tests/production.rs"` is scanned rather than skipped by directory name.
- **PTX has a boundary.** `PtxSource` requires UTF-8 text that does not begin with an image magic
  and carries a `.version` directive, so the driver's format sniffing cannot route the text variant
  to the binary-image parser.
- **Publication is monotonic.** A rollback below already-released output is refused. Usage counts
  committed completion tokens, matching document 05; `emitted` is a delivery counter and `withheld()`
  is the gap a stop string opens.
- **Two reference paths could return successful nonfinite results.** The sampler divided before
  subtracting the maximum, so a tiny temperature gave `Ok([NaN, NaN])`; affine reconstruction
  returned `Ok([inf])` for a `f32::MAX` scale. Both are fixed and both now have a companion test
  proving the new check does not fire on realistic inputs.
- Two smaller ones: `OracleRegistry::register` replaced the entry it was meant to protect before
  returning its error, and the recorded host compiler was nvcc's assumed default rather than the one
  passed to it.

## Read this before assuming anything

- `PrefixLineage` is a **lineage, not a content digest**. It answers "was this suffix replaced",
  not "are these the same tokens". Document 04's prefix-reuse key — checkpoint, tokenizer/template,
  configuration and token IDs — is a separate identity that composes with it and is still not
  implemented. Do not read a matching lineage as a licence to reuse a prefix across sequences.
- The import classifier handles `use` declarations. It does not follow re-export chains and does not
  analyse arbitrary expressions; the token layer that catches an inline `std::fs::read(p)` is still
  there and is deliberately small.
- `PtxSource`'s `.version` check is **necessary, not sufficient**. The driver decides whether the
  text compiles; the check only guarantees it will be read as text.
- The affine overflow check fires on the reconstruction, not on the scale. A scale can be finite and
  positive and still produce an infinite product.

## Still open, unchanged

**O1, O2, O4 and O5** remain OPEN. No importer exists, no kernel implements W4A16 or W8A16, no
checkpoint has been imported, no inference has run, and no throughput has been measured. The device
lane still has no CI runner.

## Next

[Task 0003](../tasks/0003-m1-bf16-reference-interpreter.md) — the BF16 host reference interpreter —
stands. It now inherits state and provenance contracts that were tightened rather than the ones the
second review warned against carrying forward. Its contract section must still be filled in, with
error metrics declared, before any code is written.
