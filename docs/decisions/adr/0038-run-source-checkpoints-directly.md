# ADR 0038 — Moxie runs downloaded checkpoints directly; conversion is not required

- **ID / date / author / status:** 0038 / 2026-09-25 / recorded by the coordinator on the owner's ruling / **accepted**
- **Classification:** **owner requirement.** The owner (2026-09-25): "I want to be able to just run whatever I download … It is just too stupid to have to convert them." Then, on the coordinator's recommendation: "yes, switch, write the ADR".
- **Scope and owning shared component:**
  - the model loading path: the importers in `moxie-format`, and the chunk sources in `moxie-executor/src/residency.rs`;
  - the status of `moxie-repack`;
  - the "canonical artifact" as a runtime input.

  No kernel, precision or numerical contract changes.
- **Supersedes, in part:**
  - [ADR 0020](0020-user-managed-storage-and-canonical-materialization.md): canonical materialization as the thing Moxie runs;
  - [ADR 0021](0021-repack-is-a-moxie-program.md), [ADR 0022](0022-user-programs-and-canonical-write-authority.md) and [ADR 0027](0027-repacking-is-provisional-pending-measured-inference-benefit.md): repack as the route to a runnable model.

  Their storage-authorization, write-ownership and exact-preservation rules still stand. Document 06's M3.1 sentence "Moxie only reads the canonical" and M5.1's "sharded weights and state use the canonical format" are superseded as input requirements. Their text is amended when the specification is next revised; until then this ADR governs.

## Problem and mechanism

Moxie's plan was: download a checkpoint, run `moxie-repack` to write a
second, "canonical" copy, then run that copy. For the M6 exit benchmark,
that meant a 33 GB second copy of `gemma-4-31B-it-AWQ-8bit` under `/models`,
made on 2026-09-25. ADR 0027 had already found that no measured inference
benefit justifies the step, made repacking provisional, and required the
loading interface to keep accepting a source checkpoint. The owner now rules
the step out as a requirement.

## What already exists

- `ShardSource` (`moxie-executor/src/residency.rs` about 48) reads weight
  chunks straight from a checkpoint's own safetensors shards into the
  residency authority. Real-checkpoint GPU tests already use it
  (`affine_linear_real_module.rs`, `residency_reads.rs`).
- The layout readers live in `moxie-format`: `compressed_tensors.rs`,
  `gptq.rs`, `affine.rs`, `bf16.rs` and `checkpoint_config.rs`.
  `moxie-repack` runs them offline and writes the result. Loading runs the
  same readers at load time, per chunk, and uploads the result.
- No code runs a whole checkpoint yet, so no runtime path depends on the
  canonical files.

## Options examined

- **Keep conversion mandatory.** Rejected by the owner: it means a second
  copy on disk and an extra step for every model, with no measured benefit
  (ADR 0027).
- **Load source checkpoints directly (chosen).** The source is read in place,
  and the layout is reinterpreted chunk by chunk as weights go to the device.
- **Both, with canonical as an optional cache.** Kept possible, not built:
  `moxie-repack` remains an optional tool, and a canonical artifact may still
  be loaded where one exists. Nothing requires it.

## Decision

1. **The input Moxie runs is the checkpoint as downloaded:** its
   `config.json`, `model.safetensors.index.json` and safetensors shards,
   read in place and never modified (ADR 0020's read-only roots).
2. **Source layouts are interpreted at load time** by the shared importers
   in `moxie-format`, one chunk at a time. The result goes to the residency
   authority. Nothing is written to disk. Exact preservation (ADR 0018) still
   holds: the loaded values are the values the source encodes.
3. **A layout Moxie cannot interpret exactly is refused at load**, naming
   the tensor and the reason, just as the converter refuses today (for
   example GPTQ with a group activation order). Nothing is guessed.
4. **Validation moves to load time.** File sizes, safetensors headers, the
   index and shape checks run on load. Content hashes may be cached per file
   (path, size and modification time) so that a model is not re-hashed on
   every start. A cache miss re-hashes, and nothing is trusted without a
   check.
5. **`moxie-repack` becomes optional.** It stays as a tool, and no user is
   required to run it. Broadening it (ADR 0027's deferred continuations) is
   not planned. It is retired if nothing needs it by M11.
6. **Existing canonical artifacts are not needed.** The owner decides
   whether to delete `/models/gemma-4-31B-it-AWQ-8bit-moxie` (33 GB) and its
   plan file. No agent deletes them.

## Consequences and costs

- **Every load pays a CPU reinterpretation cost**, where conversion paid it
  once. For INT8 `pack-quantized` this is mostly bit unpacking. It is
  expected to be small next to reading the bytes from disk, but it is
  unmeasured.
- Every supported source format needs a load-time reader: BF16 safetensors,
  compressed-tensors (AWQ-style INT4/INT8) and GPTQ. AutoRound is still
  missing, as it was before.
- M6's slice 7 item 5 becomes "Gemma's weights from the downloaded
  checkpoint into residency". Item 1 (the canonical artifact) no longer
  gates anything.

## Enforcement and removal

- No runtime path may require a canonical artifact to run a model. A task
  that makes one do so fails review against this ADR.
- Revisit only if a measured inference benefit of a prepared layout appears
  (ADR 0027's test). Then a prepared layout returns as an optional cache,
  never a requirement.
