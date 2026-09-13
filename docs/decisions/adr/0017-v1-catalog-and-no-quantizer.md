# ADR 0017 — v1 catalog and no in-Moxie quantizer

- ID / date / author / status: 0017 / 2026-09-13 / owner-directed, recorded by implementation agent / adopted.
- Classification: owner requirement (catalog membership, order, vision/draft inclusion, and quantizer prohibition). No owner gate is inferred beyond what the owner ruled verbatim.
- Scope and owning shared component: `moxie-format`, `moxie-storage` (importer/converter), `moxie-graph`/`moxie-model-api` (graph composition for listed families), `docs/spec/03-memory-formats-and-cuda.md` (canonical artifact, import priority), `docs/spec/06-implementation-roadmap.md` (M3). Concrete model crates own nothing.
- Supersedes / superseded by: amends [ADR 0003](0003-int4-int8-bf16-weight-family.md) and document 03 § "Import priority and quality boundaries" type-2 path (precision conversion/requantization) — that path is removed from Moxie's scope. Nothing else. See owner-gate [O1](../owner-gates.md#o1--initial-release-catalog).

## Problem and mechanism

Document 01 O1 asks which exact checkpoint revisions are the initial release catalog, in what order, including image-capable and native speculative-head variants. `quantization-candidates.md` had inspected ten candidates with pinned `config.json` revisions but no catalog commitment; their presence did not authorize bulk conversion, quality gating, or migration claims. Document 03 defined canonical import as three distinct operations — value-preserving repack, precision conversion/requantization, and runtime preparation — and left M3 to implement a pinned quantizer/calibration profile for the second. ADR 0003 adopted the INT4/INT8/BF16 family and noted the second operation as a future capability.

The owner has now ruled (2026-09-13, verbatim in O1):

> v1 catalog is those ten, in this order: Gemma, Glimmer, GLM, DeepSeek, Qwen, Laguna, Inkling, Hy3. Image-capable and native speculative-head (MTP/`dflash`) variants are in v1. Requantization is not allowed. Never. Moxie will only repack; any quantization will be done using an external program.

A future model (DeepSeek V4.1-Flash W4A16 AutoRound) was separately named as intent: no quantized artifact exists yet, quantization will be external, Moxie will ingest the result as a new source revision.

Mechanism: without a pinned catalog, every bulk write (35–178 GB per entry, see M0 storage evidence and `checkpoint-inventory.md`) and every quality gate (O2) lacked a subject; without a quantizer ban, M3 could grow unbounded calibration/quantizer code to adjudicate the MXFP4/NVFP4 vs INT4 quality tradeoffs that the roadmap explicitly defers.

## Options examined

**A. Catalog as inspected metadata, no order (status quo).** Keep the ten as investigation candidates with no release order. Rejected: it preserves inspection discipline but leaves O1 OPEN, so no bulk conversion (O5) or quality (O2) work per entry can be authorized and the provisional M1–M11 bring-up order remains the only sequencing.

**B. Catalog as ten revisions in ruled order, with Moxie quantizer for future artifacts (original doc 03).** Pin the ten plus a pinned Moxie quantizer for BF16→INT4 conversions (e.g., V4.1-Flash). Rejected by owner ruling: "Never" — no quantizer/calibration code in Moxie, ever. Keeping the type-2 path would create a second quality-adjudication responsibility inside Moxie that the owner explicitly places outside it.

**C. Catalog as ten revisions in ruled order, repack-only, vision/draft in v1, future quantization external (selected).** The ten become v1; every import is value-preserving repack (`W=(Q-Z)*S`, codes/scales/Z/group preserved bit-exact) plus manifest and chunking; runtime preparation stays; any future quantization happens externally and enters Moxie as a new source revision with its own provenance and gate evidence. This matches the owner's verbatim ruling and removes the unbounded scope in B while unblocking O5/O2 per entry in order.

## Decision and authority

1. **v1 catalog is ten exact revisions, in this order.** GLM and Qwen each cover two revisions the candidate list distinguishes; the ruled family order expands to:

| # | Candidate | Revision | Family | Precision | Declared variant |
|---|---|---|---|---|---|
| 1 | `cyankiwi/gemma-4-31B-it-AWQ-8bit` | `34ca187d836de874b2c7e3edf48f439b9f583772` | gemma4 | INT8 symmetric g32 | image-text-to-text (vision tower present, M11 — in v1) |
| 2 | `cyankiwi/Muse-Glimmer-30B-AWQ-INT4` | `cba01edf73e0f0f4f013615cc01281ea04e79f85` | muse_glimmer | INT4 asymmetric g32 | — |
| 3 | `canada-quant/glm-5.3-w4a16-mtp` | `1c86622dfd7ecca80909ff1524ac1b3618b8da6f` | glm5_next | INT4 symmetric g128 | carries MTP auxiliary (in v1) |
| 4 | `Intel/GLM-5.3-Flash-W4A16-AutoRound` | `5eee1846f0321058ed73745f9aa16f2aaf0fc0a0` | glm5_next | INT4 symmetric g128 AutoRound | carries MTP (in v1) |
| 5 | `Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound` | `c838af0996ae78d27a0184d3da9772f46eb34e25` | deepseek_v4 | INT4 symmetric g128 AutoRound | — |
| 6 | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` | `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` | qwen4_exp | INT4 symmetric g128 AutoRound | — |
| 7 | `cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4` | `6fcdc07bfd6e872632f79680e786c5d17fcafaee` | qwen3_5 | INT4 asymmetric g32 | declares BF16/INT4 mixed |
| 8 | `cyankiwi/Laguna-S-2.1-AWQ-INT4` | `bc59f497520b23759ce61cc5164ca28bcc4f53bc` | laguna | INT4 asymmetric g32 | declares `dflash` draft (M9, not on disk — in v1 as a gap until present) |
| 9 | `cyankiwi/Inkling-Small-AWQ-INT4` | `599a903386348a364fe30ab0a67dcc61ff9e8008` | inkling_mm_model | INT4 asymmetric g32 | — |
| 10 | `canada-quant/hy3-w4a16-mtp` | `49228b990c704e4efd67ac420a8e3d5272f820c0` | hy_v3 | INT4 symmetric g128 | carries MTP (in v1) |

Order is release priority for bulk conversion, quality gating, and migration claims — not merely file-system order. `/fast/models/google/gemma-4-26B-A4B-it` (M2's BF16 workhorse) is **not** in v1.

2. **Image-capable and native speculative-head variants are in v1** where declared (Gemma-4-31B vision, GLM/hy3 MTP, Laguna `dflash`). Their presence does not claim support: vision graph is M11, speculation verifier is M9 — both remain blocked until their gaps close; the catalog ruling makes them v1 obligations rather than deferred items.

3. **Moxie never quantizes.** No quantizer, calibration, or precision-conversion code lives in Moxie, ever. Document 03's type-2 path ("Precision conversion/requantization: changes scales/group partitions/floating dtype or quantized values. Requires separately identified artifact and O2") is removed from Moxie's scope. The converter is `inspect → estimate disk/RAM/time → O5 authorize → repack bounded chunks → validate → atomically publish manifest`. Partial output is not loadable. A converter that changes values is out of scope and must be rejected in review.

4. **Future quantization is external.** Any future model requiring quantization (notably DeepSeek V4.1-Flash W4A16 AutoRound, for which no quantized artifact exists) will be quantized outside Moxie. Moxie ingests its output only as a new source revision with its own `source_checkpoint` + `quantizer`/`calibration` provenance, manifest entry, and O2/O5 gates — indistinguishable from a publisher artifact. DeepSeek V4.1-Flash is **named future**, not v1, pending that external artifact.

Authority: owner ruling on O1 (verbaitm). This ADR records and implements it; it resolves no other gate.

## Evidence and acceptance

- Sources: `docs/evidence/quantization-candidates.md` (ten pinned `config.json` SHAs), `docs/evidence/checkpoint-inventory.md`, `docs/models/gemma4.md`, `docs/models/laguna.md`, `docs/spec/01-product-and-decisions.md` O1, `docs/spec/03-memory-formats-and-cuda.md` affine schema, legacy frozen references for family geometry.
- Acceptance: O1 moves to RESOLVED with the ten revisions and order above; `docs/decisions/owner-gates.md` updated; this ADR committed alongside. Future task contracts for M3/M7 must cite this ADR and name the exact source revision + expected size/retention before any bulk write. A task that adds quantizer/calibration code fails review by this ADR.
- Quality remains O2-gated: "repack" still needs paired released→canonical checks (notably lane order within packed words, which sits inside one group and cannot be verified by group tests alone) before a support claim.

## Enforcement and removal

- `cargo xtask arch-check` and code review must reject quantizer/calibration crates, dependencies, or code paths in Moxie. `moxie-format`/`moxie-storage` importer tests must distinguish `value-preserving-repack` from `precision-conversion`; the second is unreachable.
- Document 03's type-2 prose remains as historical documentation of what external tooling does, with a note pointing to this ADR; it is not deleted to preserve provenance.
- Spec digests unchanged — no reference document file edited; amendment is via ADR per placement contract. Re-evaluate only if the owner explicitly rescinds the "never" prohibition through a new O1 ruling and ADR; no task may infer it.
