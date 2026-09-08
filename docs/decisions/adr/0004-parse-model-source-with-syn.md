# ADR 0004 — Parse model source with `syn`, not with a text scanner

- Date: 2026-09-07.
- Status: adopted and implemented.
- Authority: implementation decision. Document 02 requires `arch-check` to check "actual module
  imports, model build scripts/FFI, dynamic registration edges, and generated source", and the M0
  review's F1 said explicitly: "Use Rust parsing/compiler-assisted checks for imports/FFI/cfg where
  needed; do not keep growing a hand-written substring parser into a language parser."
- Amends: the "third-party crates: one" statement in
  [toolchain.md](../../evidence/toolchain.md), which now reads four in `xtask` and still zero in
  every production crate.

## Decision

`xtask` parses model-crate source with `syn`, and takes three direct third-party dependencies to do
it: `syn` (pinned `=3.0.3`), `quote` and `proc-macro2`.

These are **tooling** dependencies of the composition root. No production crate gains a dependency;
`arch-check` continues to enforce an empty third-party allowlist for every shared crate and for model
crates. The `arch-check` allowlist names all four (`toml` included) explicitly, and it rejected this
change until they were added — which is the enforcement working.

## Why

Three successive reviews found the text scanner accepting model crates that do exactly what document
09 §B denies. Each time the fix was a special case, and each time the next review found another
spelling:

| Accepted with zero violations | Why the scanner missed it |
|---|---|
| `use std::fs;` under `#[cfg(test)] mod tests {}` followed by production code | the stripper kept only text before the first test module |
| `use std::{fs};` then `fs::read(p)` | the text `std::fs` appears nowhere |
| `use std::{ /* checkpoint files */ fs };` | the hand-written parser kept the comment inside the path |
| `#[path = "../tests/reader.rs"] pub mod reader;` | production discovery worked by directory *name* |

The pattern is the point. Every one of these is ordinary Rust that compiles; none is malformed or
adversarial in any interesting sense. A scanner that works on characters has to re-derive lexing —
comments, strings, raw strings, char literals versus lifetimes — before it can answer any question
about structure, and that re-derivation is where each defect lived.

`syn` answers the structural questions directly:

- **Imports.** A `UseTree` is a tree. Groups, renames and globs are distinct node types, and a
  comment is not a node at all, so it cannot leak into a path.
- **`extern "C"`.** `Item::ForeignMod`, rather than a lowercase substring that a string literal
  could trip.
- **Test gating.** `#[cfg(test)]` is a parsed attribute on the item it gates. This replaced a
  hand-written brace matcher that had to skip comments and literals to find where the attributed
  item ended.
- **Module reachability.** `mod name;` and `#[path = "..."] mod name;` are items, so production
  source can be found by *following the module tree from the Cargo targets* rather than by guessing
  from directory names. Whether a file is production is a fact about the crate's structure, not
  about its parent directory's spelling.

Inline paths written without an import (`std::fs::read(p)`) are recovered from the token stream, so
they too are immune to comments and string literals.

## Cost, and why it is small

`syn`, `quote` and `proc-macro2` were **already in this workspace's lock graph**, pulled in by
`toml`'s `serde_derive`. Making them direct dependencies of `xtask` added **zero packages** to the
build; `Cargo.lock` gained no `name =` entry. `syn` is pinned to an exact version for the same reason
the CUDA toolkit is: it decides what the enforcement sees.

Document 03 warns "do not adopt a large dependency only from its README". This is not that. `syn` is
the compiler front end's own grammar as a library, it is what every procedural macro in the ecosystem
parses with, and the alternative was to keep writing one.

## Limits, stated

- It parses **one crate's own source**. Re-export chains are not followed: `pub use std::fs as f;`
  in a dependency, re-exported onward, is not traced. The dependency rules are the barrier there,
  and they work on resolved package identity.
- `macro_rules!` expansion is not performed, so a path constructed inside a macro body is seen only
  as the tokens written.
- `include!` is treated as a violation rather than followed. A model crate may not carry a build
  script, so there is no generated source for it to include; what `include!` does do is introduce
  code that a reader will not find by following `mod` declarations, which is the property this
  traversal depends on.
- `cfg` other than `test` is not evaluated. Everything not gated to `cfg(test)` is treated as
  production, which is the conservative direction.

These are recorded so that the next review does not have to rediscover them, and so that none of
them is mistaken for an oversight.
