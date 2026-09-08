# Task 0005 — M1.2: canonical manifest v1 and bounded tensor reads

Status: **contract proposed**, 2026-09-08, after [task 0004](0004-m1-state-transactions.md) closed
the publication-atomicity finding.

**This contract is committed before any implementation code**, as in tasks 0003 and 0004. That
commit contains no `.rs` change. Nothing below is to be adjusted once a test has run.

## Identity and authority

- Task ID / milestone / owner: 0005 / M1.2 / implementation agent (Claude), owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `1166779`
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Required documents: 03 (canonical artifact, manifest v1, the affine descriptor), 02 (the
  `moxie-format` / `moxie-storage` ownership row), 06 (M1.2), 07 (what may be version-controlled),
  08 (the legacy reader references), AGENTS.md.
- Owner gates: **none needed, and one deliberately avoided.** O5 blocks "writing large converted
  artifacts, unapproved additional downloads, and any bulk copy or requantization". This task
  **reads only**, and the only artifacts it reads are ones its own tests write into a temporary
  directory. It downloads nothing, converts nothing and writes nothing under `/models` or
  `/fast/models`. O1 and O2 are untouched: no catalog entry, no quality claim.
  **Stop and ask** if the work appears to need any of them.

## Why this next

M1.2 is the smallest remaining M1 item that unblocks the rest. M1.3 (rank-owned CUDA context and a
device-resident layer chain) and M1.5 (a real Gemma text graph) both need weights that came from
somewhere; today `moxie-interp` is fed by test fixtures constructed in memory. Until there is a
validated way to name, locate, bound and check a tensor on disk, every later task either invents its
own or hard-codes arrays.

It is also the item where getting the boundary wrong is most expensive later. Document 06 states the
constraint directly: "Keep architecture-specific metadata parsing separate from shared storage." A
storage layer that learns what a `gemma` config field means is how the legacy repository ended up
with seven readers; document 08 §167 records that the split was written down and then not enforced.

## Bounded deliverable

One outcome: **a canonical BF16 artifact on disk can be opened, fully validated, and read tensor by
tensor within a declared byte budget — and every documented rejection rule actually rejects.**

- Sole owning shared components: `moxie-format` (the schema and its validation, pure, no I/O) and a
  new `moxie-storage` (the bounded reader, the only crate here that touches the filesystem).
- Existing consumers: none yet. `moxie-interp` is **not** rewired to load from disk in this task;
  that is M1.5's integration and would drag a model graph in with it.
- Non-goals, each because it is a different task: no conversion or repacking tool (M3.1, and
  O5-gated); no affine-integer *decode from disk* (M3, the importers — the manifest must be able to
  *describe* an affine tensor and the reader must refuse to read one, which is not the same thing);
  no safetensors/GGUF/HF importer; no `mmap`; no device memory, no `moxie-memory` reservation, no
  CUDA; no tokenizer or chat template implementation, only their recorded identity; no checkpoint,
  and no read of anything under `/models`.
- Deletion plan: `xtask::speccheck::sha256_hex` is moved, not copied. `xtask` consumes the shared
  one, and `spec-check`'s published-vector test moves with it. If two SHA-256 implementations exist
  when this task is done, the task is not done.

## Contract, fixed before implementation

### Where the boundary sits

| Crate | Owns | May not |
|---|---|---|
| `moxie-format` | manifest v1 schema, parsing from a `&str`, every validation rule below, `sha256` | open a file, know a path exists, name a model family |
| `moxie-storage` | opening an artifact directory, bounded chunk reads, checksum verification while reading | interpret architecture metadata, decode weights, choose what to keep resident |

`moxie-format` stays I/O-free so its validation is testable on a string and cannot become
"validated because the file parsed". `arch-check` gains the rule; it must reject `moxie-storage`
naming a model family and reject `moxie-format` importing `std::fs`.

### Manifest v1, encoding

TOML, one `manifest.toml` per artifact directory, tensor payloads in sibling chunk files. Recorded
as [ADR 0005](../decisions/adr/0005-toml-manifest-with-separate-chunks.md) in the same commit as
this contract, and **confirmed by the fifth review**, which recommended `toml` + `serde` explicitly
allowlisted for `moxie-format` over another hand-written parser. Human-auditable, diffable, and reviewable without running the engine — which document
03's offline-inspection workflow requires — and it adds **zero** packages to `Cargo.lock`.

### Manifest v1, required fields

Document 03: "schema version, source model/revision/checksums/license, tokenizer and template
identities, architecture metadata, logical tensor roles/shapes, per-tensor precision, chunk
offsets/lengths/checksums, alignment, endianness, logical ordering, exact scale convention,
quantizer/calibration provenance, excluded tensors, and completeness status." All of them are
required to be *present*; absence is a rejection, not a default.

- `schema_version` — exactly `1`. Any other value is refused by version, before any other field is
  looked at.
- `required_features: [String]` — the "unknown required features" mechanism. Every entry must be in
  the reader's own closed set or the artifact is refused **by name**. This is what makes a future
  field safe to add.
- `source` — model id, revision, per-source-file checksums, license id. Recorded, never fetched.
- `tokenizer` / `template` — identity only: a name, a version, a digest. No implementation.
- `architecture` — a `name`, a `version`, and `metadata`, an **opaque** value tree. `moxie-format`
  validates that it parses and hashes it into artifact identity; it does not look inside. There is
  no accessor that returns a typed field of it, only the whole tree, and the type is named so that a
  later reader knows the omission is deliberate.
- `endianness` — little only; big is refused rather than byte-swapped, because nothing here has ever
  been tested on one.
- `tensors[]` — `role` (a logical name, unique), `shape`, `precision` (`bf16-v1`, `affine-int4-v1`,
  `affine-int8-v1`), `chunk`, `offset`, `length`, `sha256`, `alignment`, `logical_order`. Affine
  tensors additionally carry the closed descriptor document 03 already fixes: group rule, scale
  dtype, zero-point mode, and the group-index map when the source needs one — this task *validates*
  those fields and refuses to *read* the tensor, with an error that says M3.
- `excluded[]` — tensors the source had and this artifact deliberately does not, each with a reason.
  An empty list is a claim, so it must be written explicitly.
- `completeness` — `complete` or `partial { missing: [...] }`. Document 03: "Partial output is not a
  loadable model." A partial artifact opens for inspection and refuses every read.

### Validation, and what each rule stops

Every rule below gets a test with a hand-built malformed manifest. All integer arithmetic is
checked; `as` conversions that could truncate are not used in the validator.

| Rule | Rejects |
|---|---|
| `offset + length` computed with `checked_add` | an offset/length pair that wraps to something in range |
| `offset + length <= chunk file length` | truncation — a manifest describing more bytes than exist |
| no two tensors' byte ranges intersect, per chunk | overlap, including the total-containment case a pairwise "starts inside" test misses |
| `offset % alignment == 0`, alignment a power of two | an unaligned read presented as valid |
| `product(shape) * element_size == length` | a shape and a byte count that disagree; the product itself is checked for overflow |
| every `role` unique; every `chunk` referenced exists | a manifest whose second entry silently wins |
| `schema_version == 1`; every `required_features` entry known | an artifact from a future writer being read by a reader that would ignore the parts it does not know |
| scale dtype in {F16, BF16, F32}; group size in {32, 128}; zero-point mode consistent with the profile | document 03's closed descriptor, enforced at the manifest rather than at decode |
| `completeness = partial` | any read at all |

NaN scales are document 03's rule and belong here, but the scale *values* live in the chunk, not in
the manifest. They are checked where they are read: in M3's affine reader. This task records that as
a stated gap rather than claiming the manifest covers it.

### Opening is bounded too

The fifth review's point about task 0005: budgeting payload reads alone does not bound *opening*.
Three limits. Only the first is enforceable before deserialization; the other two are properties of
the parsed value and are checked while traversing it, which the table says explicitly because the
distinction is what bounds each stage:

| Limit | Value | Enforced | Why |
|---|---|---|---|
| `manifest.toml` file size | 4 MiB | **before parsing** — a capped reader that errors at the limit rather than reading the file and then measuring it | It bounds the parse itself. A manifest is kilobytes; four orders of magnitude of headroom constrains no real artifact. |
| tensor entries | 1,048,576 | **after parsing, before validation** — the first thing checked on the deserialized value | The file-size cap already bounds this indirectly; the explicit cap is what makes the bound on later `O(n)` and `O(n log n)` validation passes stated rather than incidental. |
| architecture-metadata nesting depth and node count | 64 deep, 65,536 nodes | **during traversal** — depth is checked as the tree is walked, and the walk stops at the limit | The opaque tree is the one field with no schema, so it is the one that can be adversarially shaped, and it is walked twice: once to validate, once to hash into artifact identity. |

`toml` is a non-recursive parser, so the depth limit is a bound on *our* traversal and hashing of
the value, not a stack-overflow guard for the parser. Stated so it is not mistaken for one.

### Chunk paths are confined to the artifact directory

A manifest names its chunk files, and a manifest is data from wherever the artifact came from.
Every chunk reference must be a **single path component**: no separator, no `.` or `..`, not
absolute, no prefix or root component. That is checked on the string, before any path is joined,
because a check performed after joining is a check on a path that has already escaped.

After opening, the resolved file is confirmed to be a **regular file** whose canonical path is still
inside the canonicalized artifact directory. That is what rejects a symlink pointing outside, which
no amount of string checking can catch. Both checks exist because either alone is insufficient: the
string check stops traversal without touching the filesystem, and the canonical check stops links.

There is a race between the check and the read that this does not close. It is recorded rather than
papered over: the artifact directory is assumed not to be mutated by another writer during an open,
which is the same assumption document 03's atomic-publish workflow already makes.

### Bounded reads

```rust
fn open(dir: &Path) -> Result<Artifact>;                       // manifest only; no payload read
fn read_tensor(&self, role: &str, into: &mut [u8]) -> Result<usize>;
fn read_budget(&self) -> ByteBudget;
```

- `open` reads and validates `manifest.toml` and **stats** the chunk files. It does not read a
  payload byte, so opening a 400 GB artifact costs the manifest.
- `read_tensor` writes into a caller-supplied buffer. If the buffer is smaller than the tensor, it
  is an error naming both sizes — never a partial read reported as success.
- Internally, a chunk is read in fixed-size slices, and the reader's own scratch allocation is
  capped by `ByteBudget`. The test asserts this with a counting reader rather than trusting the
  code: **peak reader-owned allocation must not exceed the budget**, for a tensor many times the
  budget's size.
- The tensor's SHA-256 is computed **while** reading and compared at the end. A mismatch is an
  error, and the destination buffer's contents are explicitly not to be trusted — stated in the
  type's documentation, because "it returned an error but also wrote something" is how a checksum
  gets skipped in practice.
- BF16 payloads: every 16-bit element is validated as a finite BF16 value on the way through, using
  `moxie-format`'s existing rule, so a NaN weight is caught at load rather than at the first matmul.
  Task 0003 already made non-finite weights a typed error inside the interpreter; this makes them a
  typed error one layer earlier, where the provenance is still known.

### Fixtures

- One **committed** `manifest.toml`, text, in `crates/moxie-format/fixtures/`, that a human can read
  to see what the schema is. Document 07 permits "small redistributable fixtures".
- The **payload** artifact is generated by a test helper into a temporary directory, deterministically
  from a seed. No binary blob enters git, and the generator is itself the demonstration that the
  format can be written as well as read.
- Malformed manifests are built in the tests as strings, one per rule, so each rejection names the
  rule it proves.

### Error metrics

None. This task performs no arithmetic on model values beyond validating BF16 representability,
which is exact. The numerical contracts of tasks 0003 and 0004 are unchanged.

## Acceptance

- `cargo xtask arch-check` (with `moxie-storage` declared, and the two new boundary rules
  exercised by fixtures), `spec-check`, `fmt`, `clippy -D warnings`, and the full host lane pass.
  The device lane and `test-gpu` are unchanged and carried forward; this task touches no CUDA.
- A round trip: generate a tiny BF16 artifact with at least two tensors in two chunks, open it, read
  every tensor, and compare **byte for byte** against what was written.
- One test per rejection rule in the table above, each asserting the error names the rule.
- One test per opening limit, and one per path rule: a separator, `..`, an absolute path, and a
  symlink out of the directory, each refused. The symlink test skips with a reported reason where
  symlinks cannot be created, rather than passing.
- The budget test: a tensor at least 8x the read budget is read correctly, and peak reader-owned
  allocation never exceeds the budget.
- A corrupted payload byte is caught by the checksum, and the error says which tensor.
- An affine-profile tensor validates and then refuses to read, with an error naming M3.
- A `partial` artifact opens and refuses every read.
- `sha256_hex` exists exactly once in the workspace, in `moxie-format`, and `spec-check` still passes
  using it.
- Support matrix: add a row for the manifest reader. It must say host-only, no checkpoint, and must
  not imply that anything can now load a model.
- Stop condition: if this appears to need a real checkpoint, a downloaded artifact, an importer for a
  foreign format, `mmap`, device memory or a model family's config schema, stop and report. Each is a
  different task, and the first two need O5.

## Result, filled after work

*(to be completed)*
