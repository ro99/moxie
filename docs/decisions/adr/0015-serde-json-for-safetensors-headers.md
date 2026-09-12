# ADR 0015 — Parse safetensors headers with `serde_json`

- ID / date / author / status: 0015 / 2026-09-12 / implementation agent /
  **proposed with [task 0018](../../tasks/0018-m3-compressed-tensors-int8-importer.md)**.
- Classification: measured implementation choice. It resolves no owner gate and
  changes no numerical, precision, context or compatibility contract.
- Scope and owning shared component: `moxie-format`'s safetensors reader. The
  dependency is `moxie-format`'s alone; `moxie-storage` supplies bytes and does
  not gain it.
- Supersedes / superseded by: nothing. It follows the precedent of
  [ADR 0004](0004-parse-model-source-with-syn.md), which took `syn` rather than
  hand-rolling a Rust parser, and [ADR 0005](0005-toml-manifest-with-separate-chunks.md),
  which already puts `serde` in this crate.

## Problem and mechanism

A safetensors file is a `u64` little-endian header length, that many bytes of
**JSON**, then the tensor payload. Reading one requires a JSON parser. This
repository has `serde` and `toml` in `moxie-format` already, and no JSON parser
anywhere.

The input is untrusted: it comes from a downloaded checkpoint, and document 03
requires that readers "validate all integer arithmetic and reject overlap,
truncation, NaN scales, incompatible dimensions, and unknown required features"
with "no remote checkpoint code execution; parse supported metadata safely."
The parser is the first thing that touches those bytes.

## Options examined

**A. Hand-write a bounded JSON subset parser.** No new dependency, and the
accepted surface is exactly the safetensors schema, which is a genuine security
argument: anything outside `{"name": {"dtype", "shape", "data_offsets"}}` plus
`__metadata__` could be refused by construction. Against it: JSON's escape,
number and nesting rules are where hand-written parsers go wrong, this one would
be load-bearing on untrusted input from the first checkpoint onward, and it
would need its own fuzz-shaped test surface to earn the trust a maintained
parser already has. Rejected — the risk is concentrated in exactly the code this
project would be writing from scratch.

**B. `serde_json`.** The most-reviewed JSON implementation in the ecosystem,
already paired with the `serde` this crate depends on, resolves offline from the
existing local registry (verified before this ADR: `serde_json` 1.0.151 plus
`memchr` and `itoa` compile into `moxie-format` under `--offline`). It parses;
it does not execute, fetch or allocate unboundedly on a bounded input slice.
**Selected.**

**C. The `safetensors` crate.** Would supply the whole container reader, not
just JSON. Rejected on ownership grounds rather than quality: the validation
rules document 03 states — exact `end - begin` against `product(shape) *
dtype_bytes`, overlap, payload coverage, checked arithmetic everywhere, a typed
`Error` per failure — are the contract this repository is responsible for, and
delegating them would leave a claim the tests could not make. The format is
small enough that owning the validation and borrowing only the JSON parse is the
right split.

## Decision and authority

`moxie-format` takes `serde_json`. It is used **only** to turn the header byte
range into a JSON value; every structural rule, every bound and every arithmetic
check is this repository's code and is tested against document 03's list.

The header length is read and bounded **before** the parser sees anything, so a
hostile `u64` cannot cause a large allocation ahead of validation.

This is an ordinary technical choice inside `moxie-format`'s existing ownership.
It needs no owner approval and takes none.

## Evidence and acceptance

Task 0018's acceptance carries the gates: a truncated, overlapping, mis-sized
and unknown-dtype header each rejected with a typed error; a header declaring an
implausibly large tensor returning `CapacityExceeded` rather than aborting; and
the seven real shards of the inventoried Gemma 4 artifact parsing to the tensor
sets and byte totals the inventory recorded.

Limitations, stated rather than discovered later. `serde_json` builds an owned
value for the header, so header size is a real bound and is charged as one; a
checkpoint with a pathological header is refused rather than streamed. The
dependency is host-only and appears in no device build.

## Enforcement and removal

`moxie-format` must still not touch the filesystem — the existing architecture
rule covers that and `moxie-storage` supplies the bytes. The dependency is
declared in `moxie-format` alone, and `arch-check`'s undeclared-crate rule keeps
it from spreading silently.

There is no expiry and no temporary path: this is a permanent dependency for a
permanent format. Re-evaluate if the header ever needs to be read incrementally
rather than as one bounded slice, which would be a streaming-parser argument
rather than a reason to hand-roll one.
