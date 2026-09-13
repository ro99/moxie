# ADR 0023 — The canonical affine chunk payload, and the repack journal beside it

- ID / date / author / status: 0023 / 2026-09-13 / engineer lead (implementation agent) / adopted technical ruling; implemented by [task 0025](../../tasks/0025-m3-offline-repack-publication.md).
- Classification: roadmap default. Two encodings manifest v1 deliberately left to M3, chosen here before the code that writes them. Not an owner gate, not a performance choice, not a quality claim.
- Scope and owning shared component: `moxie-format` (canonical encoding, I/O-free) and the new `moxie-storage-write` (the private journal it writes). No manifest **schema** field is added, removed or reinterpreted.
- Supersedes / superseded by: completes [ADR 0005](0005-toml-manifest-with-separate-chunks.md)'s deferral of affine payload bytes and [ADR 0022](0022-user-programs-and-canonical-write-authority.md)'s write boundary. Bound by [ADR 0003](0003-int4-int8-bf16-weight-family.md) and [ADR 0018](0018-v1-quality-is-bit-identical-repack.md): this decides *where the bytes go*, never what they are.

## Problem and mechanism

Manifest v1 describes an affine tensor — width, group rule, scale dtype,
zero-point mode, logical shape — and reserves one `(chunk, offset, length)`
byte range with one SHA-256 for it. It does **not** say how the three payloads
that make up an affine tensor are arranged inside that range.
`crates/moxie-format/src/manifest.rs` says so in a comment at the length check:

> NOTE: no `product(shape) * element_size == length` check for affine tensors.
> Their chunk payload (codes, scales, zero points) has no single element size,
> and its exact layout is M3's reader contract.

M3 item 1 is the first task that has to write one. Two encodings must exist
before its code does:

1. **The canonical affine payload.** One byte range holding packed codes, the
   scale table and, when the tensor is asymmetric, the zero-point table.
2. **A restart journal.** Document 03 requires conversion to be "offline,
   restartable": "resume checks source hashes and conversion version". Manifest
   v1 has no field for a half-finished run, and adding one would put a
   *resumable* state inside the schema whose whole job is to describe a
   *published* artifact.

## Options examined

### Affine payload arrangement

1. **Three manifest rows per tensor** — codes, scales and zero points as
   separate `[[tensors]]` entries with a naming convention. Rejected: it makes
   `role` carry structure, it lets a manifest describe a tensor whose scale row
   is missing, and manifest v1 already fixes one descriptor per logical tensor.
   It would also change identity: three rows hash differently from one.
2. **Interleaved per row or per group** — codes and their scales adjacent.
   Rejected: it optimizes for one prepared kernel layout inside the *canonical*
   form, which document 03 forbids in those words ("Canonical format is not a
   single opaque tensor layout optimized for one kernel"). A prepared layout is
   a separate artifact with its own identity and may interleave freely.
3. **Three contiguous sections, codes then scales then zero points**
   (selected). One range, one checksum, one manifest row; section extents are a
   pure function of the descriptor the manifest already carries.

### Journal placement

1. **A manifest field.** Rejected: it overloads canonical completeness. A
   reader would have to distinguish "partial because the selection is a subset"
   from "partial because a run stopped", and the second is not a property of a
   published artifact at all.
2. **A sidecar in the published directory.** Rejected: it survives publication
   and becomes a file the production reader must learn to ignore.
3. **A versioned, line-oriented journal inside the private staging directory**
   (selected). It never enters the published artifact, the production reader
   never sees it, and its version is its own.

## Decision and authority

### The canonical affine payload

For a tensor whose validated descriptor is `D`, the byte range
`(chunk, offset, length)` holds exactly three sections, contiguous, in this
order, with **no interior padding**:

| Section | Bytes | Content |
|---|---|---|
| codes | `D.code_bytes()` | `out_features` rows of `width.row_stride(in_features)` bytes, logical row-major. INT4: low nibble is the earlier logical input element. Each row starts on a byte boundary; a row's tail nibble is padding and is written as zero. |
| scales | `D.group_entries() * D.scale_dtype.bytes()` | Row-major `(output channel, group)`, little-endian, **in the source's own scale dtype**. |
| zero points | `D.group_entries() * 2`, or 0 when symmetric | Row-major `(output channel, group)`, signed little-endian `i16`. |

`length` is their sum, and the tensor's `sha256` covers the whole range. A
symmetric tensor has no third section: zero is implicit, exactly as document 03
states, and a symmetric tensor carrying a zero-point payload is a contradiction
rather than a redundancy.

Section extents come from the descriptor alone, so a reader that has parsed the
manifest knows where each section starts without reading a byte, which is what
keeps inspection bounded. Nothing else may be inferred from `length`: a length
that disagrees with the descriptor's arithmetic is a rejection, and is now
checked rather than tolerated.

**The cost, stated.** No interior alignment means an f32 scale section can begin
at an offset that is not a multiple of four. Every reader here decodes scalars
with `from_le_bytes` over byte slices, so no load is misaligned; a future
consumer that wants to reinterpret the section as `&[f32]` in place must either
copy or gain an alignment rule, and that rule belongs to prepared layouts. The
alternative — padding between sections — would make the same byte range depend
on a padding convention that manifest v1 cannot express, and would make the
expected size of a repack an arithmetic nobody can check by hand.

### The repack journal

The journal lives in the run's private staging directory, is deleted with it,
and is never part of a published artifact. It is **append-only, one TOML
document per line**:

```
version = 1
plan = { plan_digest = "<64 hex>", converter = "<version>", ... }
unit = { tensor = "...", section = "codes", rows = [0, 64], chunk = "...", offset = 0, len = 4096, sha256 = "<64 hex>" }
```

Each line parses independently with the TOML parser this repository already
builds with, so there is no hand-rolled parser (ADR 0004's lesson) and a torn
final line — a crash mid-append — is discarded by the missing newline rather
than corrupting what precedes it. A `unit` line may be written only **after**
its payload bytes are durable, and a journal entry is never evidence its
payload is correct: resume rehashes the staged bytes and compares them with the
`sha256` the entry carries.

The `plan` line binds the run: source content hashes, converter/importer
version, selection and output plan. Resume with any of them different is a
refusal, not a merge.

## Evidence and acceptance

Task 0025 is where this is measured, and none of it is claimed here:

- The payload layout is exercised by tiny fixtures over every signed INT4/INT8
  value, asymmetric zeros, every supported scale dtype, group boundaries and
  tails, compared against **independently built expected bytes** rather than by
  writing with the encoder and reading with the decoder.
- The real-module proof repacks
  `model.layers.1.mlp.experts.0.down_proj` of `Laguna-S-2.1-AWQ-INT4` and
  reconstructs all 3,145,728 values from the reopened artifact. Its expected
  canonical size — 1,572,864 + 196,608 + 196,608 = **1,966,080 B** — is this
  layout's arithmetic, and is header-derived until that run measures it.
- The journal is exercised by publication-state × failure-point ×
  cancellation × restart enumeration, including a torn final line and a
  journal entry whose payload was corrupted after it was written.

**Limitations.** This decides bytes, not execution: nothing here makes a
canonical affine tensor executable, which is M3 item 3. It is not a quality
claim: a bit-identical repack is ADR 0018's v1 quality definition and says
nothing about model output. And it fixes no prepared layout — those are
separate artifacts with their own identity and may arrange the same values any
way a kernel needs.

## Enforcement and removal

- The encoder and the decoder are **one codec** in `moxie-format::affine`, and
  the streaming converter and the whole-tensor `import` share their
  implementation rather than agreeing by inspection.
- Manifest validation now checks an affine tensor's `length` against the
  descriptor's section arithmetic. An artifact written before this rule existed
  cannot exist: no repack has ever published one, and the reader refused to read
  affine tensors until this task.
- The journal version is its own integer, checked before its fields are read.
  A future journal is refused by version, exactly as a future manifest is.
- Rollback: this is additive. Removing it would mean no canonical affine
  artifact could be written at all.
