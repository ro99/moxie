# ADR 0025 — Canonical artifact v2: safetensors shards and a Moxie manifest

- ID / date / author / status: 0025 / 2026-09-14 / engineering agent, under the owner's packaging ruling / adopted; implemented by [task 0026](../../tasks/0026-m3-canonical-safetensors-publication.md).
- Classification: the container is an **owner ruling** ([ADR 0022](0022-user-programs-and-canonical-write-authority.md), amended 2026-09-14). Everything below is the versioned schema that ruling requires to be fixed **before implementation**: component naming, physical shapes, shard mapping, checksum scope and migration.
- Scope and owning shared component: `moxie-format` (schema, safetensors header construction, canonical component mapping), `moxie-storage` (reading a published artifact), `moxie-repack` (writing one).
- Supersedes / superseded by: fixes what ADR 0022's amendment left open. Amends [ADR 0005](0005-toml-manifest-with-separate-chunks.md) — raw chunk files are no longer what a repack writes — and [ADR 0023](0023-canonical-affine-payload-and-repack-journal.md), whose single contiguous byte range becomes separate physical tensors. **The affine mathematics is untouched**: `W = (Q - Z) * S`, the packing, the grouping, the scale dtype and the zero-point convention are exactly as document 03 and ADR 0023 state them.

## Decision

A published canonical artifact is a **directory**:

```
<artifact>/
  manifest.toml                        Moxie manifest v2 (TOML)
  model-00001-of-00002.safetensors     conforming safetensors shards
  model-00002-of-00002.safetensors
```

Each shard is a **conforming safetensors file** — eight-byte little-endian
header length, JSON header, payload — and must open with the reference
implementation. Renaming a raw chunk does not satisfy this.

### Component naming and physical shapes

Safetensors stores physical tensors; the manifest says what they mean. One
logical tensor becomes one, two or three physical tensors:

| Logical precision | Component | Safetensors name | dtype | Physical shape |
|---|---|---|---|---|
| `bf16-v1` | weights | `<role>` | `BF16` | the logical shape |
| `affine-int4-v1` | codes | `<role>.codes` | `U8` | `[out_features, ceil(in_features / 2)]` |
| `affine-int8-v1` | codes | `<role>.codes` | `I8` | `[out_features, in_features]` |
| `affine-*` | scales | `<role>.scales` | `F16` / `BF16` / `F32`, the source's own | `[out_features, groups]` |
| `affine-*`, asymmetric only | zero points | `<role>.zero_points` | `I16` | `[out_features, groups]` |

`groups` is `groups_per_row` from the descriptor: one for per-channel, else
`ceil(in_features / group_size)`.

**The bytes are the ones ADR 0023 already defines**, split across three tensors
instead of concatenated into one range. INT4 codes keep the low nibble as the
earlier logical input element, rows stay byte-aligned, which is why the physical
width is `ceil(in/2)` and why the dtype is `U8`: a nibble pair is not a number
and calling it `I8` would invite a reader to treat it as one. INT8 codes **are**
numbers, and `I8` is what they are. A symmetric tensor has no zero-point
component at all — implicit zero has no payload, and a component of zeros would
be a different claim.

### Shard mapping

Components are assigned to shards in manifest order, each shard filled to at
most the admitted shard-size budget. **A component never spans shards.** Names
follow the convention the ecosystem already reads,
`model-<n>-of-<total>.safetensors`, zero-padded to five digits, with the total
known from the plan before anything is written. One shard is allowed and named
the same way.

Each component's payload starts on an **eight-byte boundary** within the
shard's data section. Safetensors requires no alignment; this costs at most
seven bytes per component and keeps a future mapping consumer from copying. The
padding bytes are zero and belong to no component.

### Checksum scope

One SHA-256 per **component payload** — exactly the bytes of that component's
`data_offsets` range — recorded in the manifest beside the component.

Not per shard file: a shard's bytes include a JSON header whose exact spelling
is a writer's choice, so a file digest would change with layout while saying
nothing about the values. Per component, the digest is over the numbers, and a
reader can verify one tensor without reading the rest — which is the property
the whole bounded-read design exists for. The source artifact's own whole-file
digests stay in `source.files`, where "this file" is the claim being made.

### `__metadata__` is a courtesy, not an authority

Each shard carries `__metadata__` with `moxie.schema = "canonical-v2"` and the
artifact identity. **No reader in this repository may depend on it**, and
nothing in it is validated as meaning. The manifest is the single authority for
roles, grouping, precision, provenance, checksums, exclusions and completeness.
Two sources of truth for the same fact is how they drift apart.

### Migration

- Manifest **v1 artifacts keep their meaning and their reader**. Nothing is
  silently reinterpreted or converted, and the v1 reader refuses a v2 artifact
  by version as it always has.
- The repacker **writes v2 only**. The v1 writer is deleted rather than kept
  behind a flag.
- The transitional v1 read path expires at **M11 item 4**, or earlier if a task
  establishes that no v1 artifact exists outside tests. It is small, tested and
  already written; deleting it today would only remove evidence.

## Evidence and acceptance

Task 0026 carries the gates. The ones this schema exists to make possible:

- Every published shard is opened by the **reference safetensors
  implementation** (`safetensors` 0.7.0 for Python, present on this machine) and
  its tensors compared with what Moxie's reader returns — bytes, dtypes and
  shapes. A format decision whose evidence is only our own reader has not been
  checked against the format.
- Every canonical component is compared against an **independent source
  oracle**, as task 0025 already does for the values.
- Checksum and numerical validation stay ours: safetensors permits NaN and Inf
  and enforces none of document 03's scale rules.

## What this does not authorize

No Hub upload, no bulk conversion, no source mutation, no download, no model
support and no performance claim. A safetensors file is not an executable model,
and nothing here changes which owner gates are open.
