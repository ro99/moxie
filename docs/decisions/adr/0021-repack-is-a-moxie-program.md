# ADR 0021 — Repack is a Moxie program, not an external script

- ID / date / author / status: 0021 / 2026-09-13 / owner-directed correction, recorded by implementation agent / adopted.
- Classification: owner requirement (correction to how O5 was recorded). It changes no catalog, precision, quality, or storage-authorization substance.
- Scope and owning shared component: `moxie-format` / `moxie-storage` (repack logic, manifest, chunk writing/reading); O5 gate text; the active `AGENTS.md` paragraph; the `laguna.md` O5 row. It names **no binary or crate placement** — that is open for the engineer lead (see the companion handover).
- Supersedes / superseded by: corrects in part [ADR 0020](0020-user-managed-storage-and-canonical-materialization.md) option B wording ("external script/process"), the `AGENTS.md` active-assignment paraphrase, the `owner-gates.md` O5 paraphrase, and the `laguna.md` O5 row. It does **not** change O5's substance (user-managed, user-run, per-task naming, designated roots, ten authorized revisions), O1 ([ADR 0017](0017-v1-catalog-and-no-quantizer.md): quantization is external, never in Moxie), or O2 ([ADR 0018](0018-v1-quality-is-bit-identical-repack.md): repack-only quality).

Placement continuation, 2026-09-13: [ADR 0022](0022-user-programs-and-canonical-write-authority.md) now resolves the delegated engineer-lead decision. The open-placement statements below record the state when this correction was adopted.

## Problem and mechanism

Two distinct "external" meanings were conflated into one wrong sentence:

- O1's "any quantization will be done using an external program" is correct and stays. Quantization (precision conversion, calibration, changing values) never lives in Moxie per ADR 0017.
- O5's repack was then also recorded as "an external script the user runs" (ADR 0020 option B, `AGENTS.md`, `owner-gates.md` O5 paraphrase, `laguna.md` O5 row). That is wrong. Repack is the value-preserving path document 03 defines and ADR 0017 keeps in scope: `inspect → estimate disk/RAM/time → O5 authorize → repack bounded chunks → validate → atomically publish manifest`, preserving `W=(Q-Z)*S` bit-exact (codes, zero points, group mapping, scale values and dtype, logical column identity) plus manifest and chunking. Its logic is owned by `moxie-format` / `moxie-storage` — ADR 0017 already scopes them as owner of "importer/converter", and document 02 gives that row "conversion tools". It is built under M3 item 1 and packaged under M11 item 4.

An out-of-repo repack would duplicate the affine equation outside review and outside `arch-check`: the R25 drift pattern (operative logic living outside the governed tree). The engine's half of O5 stays as written — the engine reads and validates the published canonical; it does not produce it — but the producer is a Moxie program, user-run, offline.

Terminology note: the owner names the canonical output ".mox files" as shorthand. Manifest v1 ([ADR 0005](0005-toml-manifest-with-separate-chunks.md)) currently defines a directory (`manifest.toml` + chunk files), not a single `.mox` file. File packaging is part of the user-surface gap the engineer lead resolves; this ADR invents no `.mox` container spec.

## Options examined

**A. Leave the wording.** Rejected: agents will either build around a missing tool or duplicate the affine equation out of repo to fill the hole, and M3's exit ("lossless claims have source-oracle evidence") cannot close without the program that publishes what the evidence is about.

**B. Repack genuinely out-of-repo.** Rejected: duplicates the canonical equation without review, tests, or `arch-check`; two owners of one semantic.

**C. Repack is a Moxie program, user-run, offline (selected).** Logic versioned in-repo next to the reader it must round-trip with; full-size runs executed by the user under O5 authorization. Binary/crate placement deliberately undecided here — for the engineer lead.

## Decision and authority

1. **Repack is a Moxie program** that lets the user convert the pinned source formats on the plan into the canonical format. It runs offline, user-executed, one authorized entry at a time.
2. **The word "external" is struck for repack.** What O5 keeps from "user-managed" is who runs full size and under what authorization: user-run; no agent-initiated bulk download, copy, or conversion without a task naming exact artifact, revision, expected size and retention; designated roots `/models` and `/fast/models`; the ten v1 revisions authorized for materialization. "External to the repo" was never the ruling and is not recorded here.
3. **Quantization stays external** per O1/ADR 0017. Nothing here admits quantizer or calibration code into Moxie.
4. **Placement is open.** No binary or crate (repack tool, apps, CLI, or otherwise) is named or implied by this ADR. The engineer lead decides; no task may decide it by inference.
5. Authority: owner direction, 2026-09-13 session ("repack is done via an external script: this is plain wrong ... Repack is a Moxie program"). O5 moves with a correction appended and its original 2026-09-13 verbatim preserved for provenance, not paraphrased away.

## Evidence and acceptance

- Owner direction in-session 2026-09-13, implemented as this ADR plus: `AGENTS.md` active paragraph corrected; ADR 0020 marked superseded-in-part (text preserved, link added); `owner-gates.md` O5 correction appended with verbatim preserved; `laguna.md` O5 row corrected ("Nothing here has written a byte" unchanged).
- Historical records that already contain the stale sentence (task 0023's result section, the 2026-09-13 task 0023 handover, task 0024's committed contract which states nothing below is adjusted once a test has run) are **left as written** and noted here, per the standing rule that rewriting a record to match a later ruling erases what was true when it was written. They are superseded on this point by this ADR, not edited.
- Acceptance: a search for the stale repack-"external" phrasing returns only the preserved-verbatim/provenance spots named here plus this ADR's own quotations.

## Enforcement and removal

- Review must reject new "repack is external" wording and any out-of-repo repack duplicating the affine equation.
- Review must reject any task that decides the repack binary/crate home by inference; that awaits the engineer lead's user-surface ruling (companion handover).
- Re-evaluate only through a new owner ruling and ADR; no task may infer a change.
