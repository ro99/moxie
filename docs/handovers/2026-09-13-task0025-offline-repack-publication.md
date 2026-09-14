# Handover — task 0025 implemented: the offline repacker, and what an enumeration found in it

**Task 0025 is implemented and not reviewed.** Everything here is a claim until
a review reproduces it. It is **M3 item 1's** bounded offline repack and
canonical publication: `moxie-repack` inspects an explicit selection, converts it
in bounded work units, resumes an interrupted run, validates the result through
the production reader and publishes a manifest-v1 directory with one rename.

**It does not close M3 item 1**, whose catalog and completeness integration is
untouched, and it establishes **no execution and no quality claim**: nothing in
this repository executes a canonical INT4 tensor, which is M3 item 3 and remains
the largest gap in the milestone. A bit-identical repack is
[ADR 0018](../decisions/adr/0018-v1-quality-is-bit-identical-repack.md)'s v1
quality definition — a statement about bytes, not about what a model produces.

## Workspace identity

- Writable repository `/home/rodrigo/Developer/moxie`, branch `main`.
- Implementation base `44a94c4`, clean tree. Contract `74e0709` (a concurrent
  agent's), and [ADR 0023](../decisions/adr/0023-canonical-affine-payload-and-repack-journal.md)
  with the re-recorded base at `aaeff60` — **both before any implementation
  commit**, which is what the contract asks for.
- Read-only legacy reference `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, **unmodified**. Its checkpoint I/O
  (`include/strata/platform/checkpoint_io.hpp`, `src/platform/checkpoint_io.cpp`)
  was read, as the contract asks, and it is a **reader only**: `O_RDONLY`
  shards, bounded `pread` with `EINTR` retried without advancing, premature EOF
  as an error. There is no legacy writer to migrate or delete — ADR 0022 says so
  and this checked it — and the two properties it has are already `moxie-storage`'s.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was written, converted, deleted or modified.** One
  shard was opened read-only: 5,369,738,904 B hashed and 1,818,624 B of declared
  tensor ranges read.

## Completed facts

See [the task record](../tasks/0025-m3-offline-repack-publication.md#result-filled-after-work)
for the full account, including every gate's numbers. In short:

- One new crate — `moxie-repack`, the program ADR 0021 assigns the offline
  workflow to, whose `write` module owns canonical publication — plus four new
  shared modules in `moxie-format`:
  the canonical payload codec, the restart journal's schema, the selection
  schema, and manifest v1's **writer**.
- The importer is now written in terms of a streaming `PackQuantizedPlan`, so a
  whole-tensor import and a tiled repack execute the same arithmetic; the two
  decoders `import` carried inline are deleted rather than left beside it.
- Manifest validation no longer accepts any length for an affine tensor: ADR
  0023 fixes the payload layout, so the length is the descriptor's own
  arithmetic.
- The writer was briefly its own crate, as ADR 0022 specified. The owner
  rejected that on review and
  [ADR 0024](../decisions/adr/0024-one-storage-crate-and-a-write-module.md)
  folds it into the program: one consumer by design, and the boundary had forced
  a second copy of the reader's `pread`. `moxie-storage` gains two public
  bounded-read primitives instead, and the `arch-check` rule written for the
  crate edge is deleted along with its eight fixtures — nothing outside the
  program can name a module that is inside it.
- Every gate passes with nothing failed, ignored or skipped: **1,014 host
  tests**, 1,049 device-feature tests, three clippy lanes, `spec-check`, and
  `arch-check` at 79 rejected fixtures, 21 accepted and 13 rules.
- The mutation battery measures **29 of 29 mutants caught and 4 of 4 expected
  survivors held**. Its first run caught 24 of 28 and left four survivors and
  two skips; three tests and two anchors closed five of those, and the sixth is
  an equivalent mutant whose argument is written out
  ([experiment 0006](../evidence/experiments/0006-repack-publication-mutations.md)).
- One real module round-trips: **3,145,728 values** checked against document
  03's canonical FP32 equation and against the source's own BF16 boundary
  separately, the boundary moving 777,575 of them. The whole-file digest this
  run computed over the 5.37 GB shard equals the one the hub recorded at
  download.

## Decisions

- **[ADR 0023](../decisions/adr/0023-canonical-affine-payload-and-repack-journal.md)**
  fixes the two encodings this task needed, **before** the code that writes
  them: the canonical affine payload is codes, then scales, then zero points,
  contiguous, with no interior padding and one checksum over the whole range;
  the restart journal is a private, versioned, line-oriented file in the
  destination, one TOML document per line so that a torn append loses its own
  line and nothing before it.
- **The destination directory is the staging area.** Chunk files are written
  under their final names; what makes the artifact unreadable is that
  `manifest.toml` does not exist yet, because the production reader opens the
  manifest first. Publication is one rename of a privately named manifest that
  has already been validated **through that same reader**. Payload bytes never
  move after they are durable.
- **Full-file source digests.** `source.files.sha256` means "this file", so the
  repacker hashes the whole file and records exactly that. A digest of the
  ranges a run happened to read, in a field that says "this file", would be a
  false checksum that a later whole-artifact claim would inherit. The cost is
  real — 30 seconds for one 5.37 GB shard — and it is the honest cost.
- **Taking over an interrupted run is explicit.** A lock file outlives a crash,
  and distinguishing a crashed run from a live one needs process liveness, which
  means this machine's telemetry, which ADR 0006 gives to one crate. So the
  choice is the user's (`--take-over-interrupted-run`), the limitation is
  documented, and nothing guesses.
- **Two dev-profile optimizations**, recorded because they change how long a
  gate takes and nothing else: `[profile.dev.package.moxie-format]` and
  `[profile.dev.package."*"]` at `opt-level = 2`. The manifest cap fixture went
  from 99 s to 15 s and the real-artifact lane from 7m31s to 30 s; the published
  artifact's identity is the same either way.

## Remaining hypotheses and blockers

- **The task is not reviewed and not accepted.** Everything above is a claim.
- **Nothing executes a canonical INT4 tensor.** M3 item 3 — shared W4A16/W8A16
  dense and expert paths for SM86, SM120 qualified separately — is untouched and
  is the largest remaining gap in the milestone. Publishing an artifact changes
  nothing about that; it changes what item 3 can execute *from*.
- **The per-unit `source_sha256` in the journal is recorded provenance, not a
  check.** The mutation battery's independence control says so out loud: making
  it a constant is not caught, because what rejects a changed source is the run
  binding, which carries every source file's **whole-file** digest and is
  recomputed on every start. A future caller that bound more cheaply would need
  this field to become a check.
- **Source discovery does not exist.** Cross-shard *resolution* works — a
  module's four tensors may live in four different shards, and a fixture
  publishes one that does — but the shard each tensor lives in is named by the
  selection a user wrote. Nothing reads a `model.safetensors.index.json`.
- **Completeness is the selection's claim.** A selection declares `partial` or
  `complete` and the repacker publishes what it was told. Nothing cross-checks a
  `complete` claim against a model definition, because nothing here knows what a
  complete model is; that is M3 item 1's catalog integration, and it is not
  built.
- **`inspect` reports time as unmeasured.** It has no supplied or measured I/O
  rate, and inventing one would be inventing the number a user plans with.
- **M3 item 2's remainder is open**: group-128 symmetric INT4, whose real content
  is what `actorder: "static"` means for logical column identity. Both local
  candidates declare it, and document 03 forbids ignoring a permutation.
- **Laguna's attention tower is still not composable** and no Laguna capability
  row exists. Repacking one of its expert modules does not change either fact.

## Next task

**M3 item 3 — shared W4A16/W8A16 dense and expert paths.**

- **Owning component**: the shared kernel and executor path, with the canonical
  descriptor as its input. No model-specific execution, and no adapter-local
  dequantization.
- **Required reading**: document 03's precision and preparation sections
  (bounded preparation, no repeated full-tensor dequantization, charge
  scale/zero-point/index metadata in every budget), document 04 for the shapes,
  [ADR 0003](../decisions/adr/0003-int4-int8-bf16-weight-family.md), tasks 0018,
  0021, 0023, 0024 and this one, and the Strata offset-packed INT4/INT8 paths the
  source map names.
- **Bounds document 03 sets before it starts**: weight-only paths with BF16
  preferred and FP32 accumulation, **not** INT4xINT4 or INT8xINT8 MMA; SM86
  first with SM120 qualified **separately**; "bounded reference dequantization is
  not an acceptable final fast path by assertion" — shared bounded
  dequantization plus a BF16 GEMM is the correctness fallback, not a result.
  **Dense GEMM qualification does not qualify routed or grouped MoE.**
- **What it now has that it did not**: a canonical artifact on disk that the
  production reader opens, rather than a fixture built inside a test. The
  Laguna module this task published is one such input, and `moxie-repack` can
  produce more without an agent starting a bulk conversion — the selection is
  the user's.
- **Stop condition**: stop at the gate, with evidence, if a kernel cannot meet
  the declared numerical contract on real codes, if SM120 cannot be qualified
  separately, or if qualification would need a checkpoint operation no task has
  authorized. Do not widen a tolerance, do not claim a fast path from a
  reference dequantization, and do not let a dense result stand in for a routed
  one.
