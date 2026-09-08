# Handover — fourth-review corrections

Date: 2026-09-07. From: implementation agent (Claude). Base commit: `e401179`, branch `main`.

Fourth in a chain: [M0 corrections](2026-09-07-m0-correction-results.md) →
[second](2026-09-07-second-review-corrections.md) → [third](2026-09-07-third-review-corrections.md)
→ this one. The fourth review accepted the state work (T1, T2) and the move to `syn`, and found that
the architecture-enforcement correction was **not finished**: the parser was in place but the
traversal was not complete. One P1, reproduced and closed. Detail in
[task 0002](../tasks/0002-m0-review-and-integer-transition.md#fourth-review-corrections-2026-09-07).

## State

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | PASS |
| `cargo test --workspace --locked` | PASS, **222 unit + 1 doctest** (was 217 + 1) |
| `cargo xtask arch-check` | PASS, **19 rejected + 1 accepted** fixtures (was 17 + 0), 6 rules |
| `cargo xtask spec-check` | PASS, 10 documents, digests unchanged |
| host build with no CUDA toolkit or driver reachable | PASS, 222 + 1; `ldd` shows no `libcuda` |
| device lane | PASS, **230 unit + 2 doctests** |
| `cargo xtask-cuda test-gpu` | PASS, 15 cases, both architectures qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | exit 1, as intended |
| packages added to `Cargo.lock` | **zero** |

Counts now name unit tests and doctests separately, because the previous handover's bare totals were
ambiguous. The two doctests are the `compile_fail` guarantees — `DeviceBuffer` cannot outlive its
context (driver feature only, hence one on the host lane) and `SequenceState` cannot be cloned.

## What changed

**Function bodies bypassed the structural rules.** Two model libraries compiled and passed with zero
violations:

```rust
pub fn read_weights(p: &str) -> std::io::Result<Vec<u8>> {
    use std::{fs};                       // never reached the import classifier
    fs::read(p)
}

pub fn read_weights(p: &str) -> std::io::Result<Vec<u8>> {
    include!("../tests/read_expr.rs")    // not an `Item::Macro`, so no include rule
}
```

The first `syn` version matched on module-level items and fell back to a token scan for everything
else — the same "handle the reported shape" mistake, one level down. It is now a
`syn::visit::Visit` implementation that walks items, statements, expressions, nested blocks, closure
bodies and `impl`/`trait` members, and **no rule in it asks where a construct appears**. Imports,
`include!`, `extern "C"`, module declarations, inline paths and the `cfg(test)` exemption all fire
wherever they occur.

The one remaining token scan is over macro *arguments*, which are unparsed tokens by definition.

**Positive fixtures now exist.** `xtask/fixtures/arch-check-accepted/` holds crates that must be
**accepted**, and `arch-check` fails if either fixture directory is empty. A checker with only
negative fixtures is satisfied by rejecting everything; one with only positive fixtures by rejecting
nothing. The first accepted fixture sits deliberately next to the forbidden shapes — function-local
grouped and renamed imports of *allowed* modules, a `#[cfg(test)]` harness that really does use
`std::fs` and `std::thread`, a nested inline module, a file module — so the deeper traversal cannot
be tightened into something that flags ordinary code without a fixture failing.

## Read this before assuming anything

- **The parser's limits are unchanged and are in [ADR 0004](../decisions/adr/0004-parse-model-source-with-syn.md):**
  no re-export chains, no `macro_rules!` expansion, no `cfg` evaluation beyond `test`. `include!` is
  refused rather than followed, because a model crate cannot have a build script and included source
  leaves the module tree the traversal depends on.
- **Restore evidence is still a binding, not a deed.** The fourth review accepted that distinction
  for M0 and attached an obligation to it: *actual snapshot/replay execution must enforce the latter
  when buffers and transactions arrive.* That belongs to the memory authority, not to this crate, and
  whoever builds it should read it as a requirement rather than a note.
- **`PrefixLineage` is a lineage, not a content digest.** Document 04's token-content prefix key is
  a separate identity, still absent.

## Still open, unchanged

**O1, O2, O4 and O5** remain OPEN. No importer, no W4A16/W8A16 kernel, no checkpoint imported, no
inference, no throughput measured, no CI runner for the device lane.

## Next

[Task 0003](../tasks/0003-m1-bf16-reference-interpreter.md) — the BF16 host reference interpreter.

Four reviews have now found the same failure shape in four different places: a rule that was correct
for the construct it was written against, and silent for the same construct written somewhere else.
The habit worth carrying into task 0003 is the one the third review named — ask of each new operation
*which invariant it can break*, not which example it resembles — and the one this pass adds: when a
rule is about a language construct, apply it wherever that construct is legal, not where it was first
seen.
