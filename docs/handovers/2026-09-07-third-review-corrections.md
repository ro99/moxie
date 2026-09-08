# Handover — third-review corrections

Date: 2026-09-07. From: implementation agent (Claude). Base commit: `27bc1cf`, branch `main`.

Third in a chain: [M0 corrections](2026-09-07-m0-correction-results.md) →
[second review](2026-09-07-second-review-corrections.md) → this one. The third review confirmed the
second pass's gates independently, accepted most of it, and found three more places where an
invariant could be violated by an operation no test covered. Full detail is in
[task 0002](../tasks/0002-m0-review-and-integer-transition.md#third-review-corrections-2026-09-07).

## State

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | PASS |
| `cargo test --workspace --locked` | PASS, **218** (was 210) |
| `cargo xtask arch-check` | PASS, **17** fixtures (was 15), 6 rules |
| `cargo xtask spec-check` | PASS, 10 documents, digests unchanged |
| host build with no CUDA toolkit or driver reachable | PASS, 218; `ldd` shows no `libcuda` |
| device lane | PASS, **226** + 1 doctest |
| `cargo xtask-cuda test-gpu` | PASS, 15 cases, both architectures qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | exit 1, as intended |
| `Cargo.lock` packages added | **zero**, despite taking a parser dependency |

## What changed

- **`SequenceState` is no longer `Clone`.** A derive copied the sequence id, the result counter, the
  lineages and the result ledger into a second independently mutable authority, which then minted
  identical `ResultId`s. Every identity rule was intact and simply bypassed. The regression test is
  a `compile_fail` doctest, because the guarantee is a property of the type.
- **Restoration evidence carries branch and prefix lineage.** A root snapshot satisfied a child's
  rollback, and evidence survived the replacement of the very suffix it described — component,
  sequence, generation and prefix length were all equal in both. `Restore`'s fields are now private
  and it is minted only by `SequenceState::restore_evidence`.
- **Model source is parsed with `syn`.** The fourth distinct evasion of the same rules
  (`use std::{ /* c */ fs };` and `#[path = "../tests/reader.rs"] mod reader;`) made it clear the
  answer was not another special case. Imports come from parsed `UseTree`s, inline paths from the
  token stream, `extern "C"` from a foreign-module item, `#[cfg(test)]` from a parsed attribute, and
  production sources from following `mod` declarations out of the Cargo targets. See
  [ADR 0004](../decisions/adr/0004-parse-model-source-with-syn.md).

## Read this before assuming anything

The limits are written into the code and into ADR 0004 rather than left to be rediscovered:

- **Restore evidence is a binding, not a deed.** It proves the evidence names this sequence, branch,
  prefix version and generation, and that nothing moved underneath it. It cannot prove the
  restoration work happened; a caller that mints evidence and does nothing still passes. The buffers
  that would make it checkable belong to the memory authority, which does not exist yet.
- **`PrefixLineage` is a lineage, not a content digest.** It answers "was this suffix replaced", not
  "are these the same tokens". Document 04's prefix-reuse key is a separate identity, still absent.
- **The parser reads one crate's own source.** No re-export chains, no `macro_rules!` expansion, no
  `cfg` evaluation beyond `test`. `include!` is refused rather than followed, because a model crate
  cannot have a build script and included source leaves the module tree the traversal depends on.
- **`arch-check` gained three dependencies.** `syn` (pinned `=3.0.3`), `quote`, `proc-macro2`, all in
  `xtask` only. They were already in the lock graph via `toml`, so no package was added to the
  build. `arch-check` rejected the change until its own allowlist named them.

## Still open, unchanged

**O1, O2, O4 and O5** remain OPEN. No importer, no W4A16/W8A16 kernel, no checkpoint imported, no
inference, no throughput measured, no CI runner for the device lane.

## Next

[Task 0003](../tasks/0003-m1-bf16-reference-interpreter.md) — the BF16 host reference interpreter.
The state and provenance contracts it inherits have now survived three reviews. Its contract section
must still be filled in, with error metrics declared, before any code is written.

The review's closing diagnosis is worth carrying into that task: *tests covered the reported
examples, but not every operation that could violate the invariant.* When the interpreter adds
operations to these types, the question to ask of each is which invariant it can break, not which
example it resembles.
