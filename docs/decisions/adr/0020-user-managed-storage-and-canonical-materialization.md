# ADR 0020 — User-managed storage and canonical materialization

- ID / date / author / status: 0020 / 2026-09-13 / owner-directed, recorded by implementation agent / adopted; **wording corrected in part by [ADR 0021](0021-repack-is-a-moxie-program.md) (2026-09-13): repack is a Moxie program, not an external script. The 0020 text below is preserved; where it says "external script/process" for repack, read ADR 0021.**
- Classification: owner requirement (storage authorization). It does not change precision, cache, or catalog membership.
- Scope and owning shared component: `moxie-format` / `moxie-storage` (manifest, canonical chunk reading), docs `artifact-roots.md`, `checkpoint-inventory.md`. No engine crate gains bulk-write authority.
- Supersedes / superseded by: refines O5 portion of [ADR 0017](0017-v1-catalog-and-no-quantizer.md). Corrected in part by ADR 0021 on the repack-"external" wording only; substance (user-run, per-task naming, roots, ten revisions) unchanged. Document 03 converter prose remains repack-only; this ADR records *who* runs it. See [O5](../owner-gates.md#o5--storage-and-conversion-authorization).

## Problem and mechanism

Moxie's canonical artifact for each v1 revision is ~35–178 GB. Materializing ten entries is ~1 TB. Document 03 describes the converter as offline, restartable, atomically publishing a manifest. O5 asks which paths, how much disk/time, which source revisions, and whether a higher-precision original exists before any bulk write. Without a ruling, agents cannot write large converted artifacts, and tasks must name exact artifact/revision/size/retention.

After [ADR 0017](0017-v1-catalog-and-no-quantizer.md), v1 is ten pinned revisions, repack-only, no in-Moxie quantizer. The remaining O5 question is operational: who materializes the canonicals and where, and what budget protects the benchmark machine (`/fast` 1.4 T free, `/` 551 G, `/archive` 1.3 T).

## Options examined

**A. Agent-managed bulk conversion.** Agents run `cargo xtask` converter on the benchmark machine, writing canonicals under `/fast/models`. Rejected as default: it would let any task fill the disk and the rewrite request is not blanket permission for terabytes. Requires explicit per-task authorization under O5.

**B. User-managed repack, Moxie points at canonical (selected).** Repack is done via an external script/process, pointing to the chosen directory under the designated roots (`/models`, `/fast/models`). Moxie only points to the repacked file (manifest + chunks) and validates it. No agent-initiated bulk download/copy/convert without a task naming exact artifact, revision, expected size and retention.

**C. No canonical materialization at all.** Keep repack in-memory only (task 0018 style). Rejected for v1: it leaves the ten v1 entries inspected but never runnable; O5 would stay OPEN and M3 could never close.

## Decision and authority

1. **Designated roots remain `/models` and `/fast/models`** (`artifact-roots.md`). Source and canonical may live under either; they are not assumed to be one filesystem.

2. **v1 canonicals are authorized to be materialized there.** The ten revisions pinned in [ADR 0017](0017-v1-catalog-and-no-quantizer.md) may be repacked to canonical form and published under those roots. Expected sizes are the source payloads (e.g., Gemma-31B 35 GB, Laguna 76 GB, GLM-5.3 178 GB recorded in `quantization-candidates.md` and `checkpoint-inventory.md`); canonical size is comparable plus manifest/alignment. No additional download beyond those ten is authorized by this ruling.

3. **Repack is user-managed, offline, and out of agent scope by default.** An agent must not perform bulk download/copy/convert (including repack) without a task contract that names exact artifact, source revision, expected size/time, and retention. The engine's responsibility is to read and validate the published canonical (checksums, chunk identity, scale dtype, group map), not to produce it.

4. **Higher-precision original:** for v1's ten, none — they are already INT4/INT8 and Moxie only repacks. For future externally quantized artifacts (e.g., V4.1-Flash), the owner's BF16 original exists outside Moxie and the new source revision carries its own provenance.

Authority: owner ruling on O5 (2026-09-13, verbatim). This ADR records it; O5 is RESOLVED for v1 on this basis. M2 sequencing sub-questions (BF16 MoE via `gemma-4-26B-A4B-it`) remain resolved as recorded 2026-09-12.

## Evidence and acceptance

- O5 ruling verbatim in `owner-gates.md`.
- Roots: `docs/evidence/artifact-roots.md`; free space from `hardware-inventory.md` and `checkpoint-inventory.md`.
- Historical: M2 sequencing evidence and four questions answered 2026-09-12 via the BF16 MoE already on disk; "Nothing was downloaded, copied or converted."
- Acceptance: O5 → RESOLVED for v1. Tasks that need a canonical must cite the exact revision and expected size; agents that would write bulk data without that citation fail review by this ADR.

## Enforcement and removal

- Review must reject agent-initiated bulk writes without per-task O5 naming. `arch-check` already enforces no bulk checkpoint conversion/download without scoped authorization.
- This ADR is permanent for v1. Re-evaluate only if the owner authorizes agent-managed bulk conversion or a new catalog entry with different storage requirements through a new O5 ruling and ADR.
