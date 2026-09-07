# ADR 0002 — Rust edition 2024

- ID / date / author / status: 0002 / 2026-09-07 / M0 / accepted
- Classification: **owner requirement** (directed during M0 setup)
- Scope and owning shared component: workspace
- Supersedes / superseded by: —

## Problem and mechanism

The workspace was first created on edition 2021. The owner directed edition 2024.

## Decision and authority

Edition 2024, resolver 3, `rust-version = "1.97"`. Set once in
`[workspace.package]`; every crate inherits it with `edition.workspace = true`.

## Consequence that matters for this codebase

Edition 2024 makes `unsafe_op_in_unsafe_fn` an error rather than a lint. Inside an
`unsafe fn`, unsafe operations need their own `unsafe` block, so each one carries
its own justification instead of inheriting a blanket exemption from the function
signature. Extern blocks must also be written `unsafe extern "C"`.

For a project whose FFI boundary document 02 requires to be *audited*, this is
the edition doing the work the policy asks for. `moxie-cuda` is written to it,
and the workspace additionally sets `unsafe_op_in_unsafe_fn = "deny"` and
`clippy::undocumented_unsafe_blocks = "deny"` so the requirement holds even if a
future edition relaxes it.

## Evidence and acceptance

Workspace builds, 69 tests pass, `cargo clippy --workspace --all-targets` is
clean with `-D warnings`, and `cargo xtask test-gpu` passes on all three devices.

## Enforcement and removal

`[workspace.package] edition` is the single source. A crate that sets its own
edition is a review finding; `arch-check` does not currently detect it, which is
a known gap in the checker rather than an accepted practice.
