# ADR 0018 — v1 quality is bit-identical repack

- ID / date / author / status: 0018 / 2026-09-13 / owner-directed, recorded by implementation agent / adopted.
- Classification: owner requirement (acceptable loss). It records what the owner accepts for v1 and what evidence closes it; it does not set a numerical tolerance beyond bit identity.
- Scope and owning shared component: `moxie-format` / `moxie-storage` (canonical repack, manifest, chunking, paired verification), `docs/spec/03-memory-formats-and-cuda.md` § affine schema, `docs/spec/07-validation-and-performance.md` (quality ladder). No model crate owns quality.
- Supersedes / superseded by: refines [ADR 0003](0003-int4-int8-bf16-weight-family.md) and [ADR 0017](0017-v1-catalog-and-no-quantizer.md) for v1 only. Type-2 precision conversion (quantization) is out of scope per 0017; this ADR defines what remains. See [O2](../owner-gates.md#o2--acceptable-quality-loss-for-the-selected-integer-artifacts).

## Problem and mechanism

Document 07 separates quality into conversion delta (released → canonical) and execution delta (canonical → engine). Before 0017, Moxie could in principle quantize BF16→INT4 and would need perplexity/task/conversation tolerances. After 0017, Moxie never quantizes: all ten v1 entries are publisher-quantized and Moxie only repacks (`W=(Q-Z)*S`, codes/scales/Z/group preserved bit-exact per doc 03). The question becomes: what acceptance closes O2 for a repack-only v1, and what stays as future work for externally quantized artifacts?

Without a ruling, no entry can move from "repacked" to "supported" in the support matrix — `support-matrix.md` currently marks every v1 candidate "import is not support" — because the quality gate lacks a subject.

## Options examined

**A. Full task-quality gate for v1 (perplexity, MMLU, conversations).** Require paired perplexity/task/conversation evidence for each of the ten vs its BF16 parent before closing O2. Rejected for v1: after 0017 the publisher's quantization quality is not Moxie's code; imposing a task bar here would make Moxie gate the publisher's model choice rather than its own preservation.

**B. No quality gate at all.** Close O2 with no evidence. Rejected: repack still has correctness hazards (scale dtype, group map, zero-point axis, and notably Gemma-4-31B lane order within a packed word — four INT8 codes per I32 word, all inside one group-32 group, so group tests cannot catch a byte-order swap). A claim without that check would repeat the "lane order cited from pinned reader, not verified" risk recorded in `gemma4.md`.

**C. Bit-identical repack plus paired-logits spot check where layout is ambiguous (selected).** For v1, acceptable = bit-exact preservation under the declared arithmetic. Evidence is source-arithmetic comparison plus one paired-logit verification per ambiguous layout (Gemma-31B as the sole v1 example). Publisher's own quality is accepted as-is for v1; no perplexity/task/conversation tolerance is imposed. Future external quantizations (e.g., DeepSeek V4.1-Flash) get a new O2 package as a new source revision.

## Decision and authority

1. **v1 acceptance for O2:** a v1 revision is acceptable when its canonical repack is bit-identical to the source under `W=(Q-Z)*S` — same codes, same zero points, same group-32/128 mapping (including `g_idx`/`actorder` where present), same scale values with source `scale_dtype` preserved (BF16/FP16/FP32 as read from tensor headers, not `null` in config), and same logical column identity (including Laguna's output-axis-packed zero points vs input-axis-packed codes). Manifest + chunk checksums + atomic publish required (doc 03).

2. **Required verification:** exhaustive signed-decode tables (all 16 INT4 codes, all 256 INT8 including -128), rebias equivalence, group boundary/partial-group, activation-order, and — for every v1 artifact where packing is ambiguous inside a group — a paired-logit spot check vs the released checkpoint through the canonical path. For v1 this is Gemma-4-31B lane order; other v1 entries have analogous checks (e.g., Qwen/Laguna asymmetric axis).

3. **Publisher quality for v1 is accepted as-is.** No perplexity, task, or representative-conversation delta is imposed on Moxie for these ten. The support matrix may mark "supported" once (1) and (2) pass plus ledger/admission/coverage gates; quality is not a separate v1 blocker.

4. **Future external quantization:** any new source that Moxie ingests after external quantization (DeepSeek V4.1-Flash and any later) is a new artifact with new provenance. Its O2 reopens as: paired logits + perplexity on held-out + task suite + representative conversations vs its BF16 parent, plus the same repack checks above for the new INT4 bytes. That package is authorized under O2/O5 at that time, per ADR 0017.

Authority: owner ruling on O2 (2026-09-13, verbatim). This ADR records it; O2 is RESOLVED for v1 on this basis.

## Evidence and acceptance

- v1 catalog: ten revisions pinned in [ADR 0017](0017-v1-catalog-and-no-quantizer.md) and [O1](../owner-gates.md#o1--initial-release-catalog).
- Repack equation and descriptor fields: `docs/spec/03-memory-formats-and-cuda.md` affine schema; `moxie-format::affine` and `moxie-format::manifest`.
- Historical acceptance: [task 0018](../tasks/0018-m3-compressed-tensors-int8-importer.md) imported three Gemma-4-31B modules to canonical form in memory (no manifest yet, no execution) — import, not support. Its lane-order caveat is the evidence that (2) is required.
- Acceptance: O2 → RESOLVED for v1; tasks closing repack must cite this ADR and produce the paired-logit record where applicable. No quality benchmark is required for v1 "supported" beyond (1)+(2).

## Enforcement and removal

- Review must reject any "supported" claim for a v1 revision without bit-identity evidence and, where applicable, the paired-logit check. `cargo xtask quality` remains NOT IMPLEMENTED for v1 and is not required to close O2 under this ruling; it becomes required for future external quantizations.
- This ADR is permanent for v1. Re-evaluate only if the owner narrows the repack guarantee or adds a task-quality bar through a new O2 ruling and ADR.
