# Task 0036 — reconcile and close M3.1 / M3.2 / M3.3

Status: active; closure contract, 2026-09-18.

## Identity and authority

- Task0036, M3 final closure; repository owner reviews and accepts.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `c7ff2bb`.
  `coordinator.md` is an unrelated carried edit and remains outside this task.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; checkpoint roots remain read-only.
- Roadmap M3 items 1/4 map to M3.1, item 2 to M3.2, and items 3/5 to M3.3.
  Tasks0024–0035 and the 2026-09-16 recovery handover are the evidence ledger.
- O1–O5 are resolved. O6/O7 remain open and block performance claims, not this
  correctness closure. Experiment0007 remains pending and repack retention stays
  provisional.

## Bounded deliverable

Produce one frozen, reviewable M3 candidate in which canonical publication,
the pinned integer importers, and shared dense/expert execution meet every M3
exit clause. Repair only concrete closure findings. Run the final publication
and dense mutation batteries on that source identity, reconcile task and support
records, and obtain the owner's code review and milestone decision.

Shared owners remain `moxie-format`, `moxie-storage`, `moxie-repack`,
`moxie-memory`, `moxie-plan`, `moxie-kernels`, and `moxie-executor`. Model crates
remain declarative consumers. No new cache, allocator, serialization-specific
runtime, quantizer, bulk conversion, download, performance claim, or checkpoint
token generation belongs here.

## Contract before closure

- Canonical values remain `W=(Q-Z)*S`; finite nonzero signed source scale bits
  are preserved under ADR0030. ADR0028's two-clause numerical gate is unchanged.
- INT4/INT8 group32/128 and applicable per-channel cases preserve zero points,
  tails, source scale dtype, and optional logical-column maps. Unsupported fused
  exporter interleaves refuse by name rather than flattening or guessing.
- Dense and grouped experts decode only bounded tiles. Codes, scales, zeros and
  maps are admitted and lease-retained through completion or quarantine; no
  full expanded weight exists.
- Publication stays restartable, checksummed, provenance-bound and atomically
  visible. No mutation expectation is weakened to manufacture a pass.
- Synthetic operation/graph evidence is reported as such. Nothing here proves
  checkpoint model output, token generation, quality, or speed.

## Acceptance

1. M3.1: full T0006 on the frozen tree runs all 66 substitutions, catches every
   declared mutant, holds all expected controls, and reports zero unstable,
   invalid, broken, or skipped cases. Reference safetensors and bounded
   publication suites pass. Experiment0007 remains explicitly pending.
2. M3.2: compressed-tensors static/group variants and GPTQ/AutoRound width,
   zero-offset, signed-scale, map, override/MTP and named interleave-refusal
   regressions pass. Bounded pinned samples retain source identity and values.
3. M3.3: dense and grouped W4A16/W8A16 cases, including maps, both shared graph
   consumers, admission, cancellation and failure lifetimes pass on both SM86
   GPUs and SM120. Full T0028 catches all declared mutants and holds controls.
4. Fmt, host/driver clippy, architecture/spec checks, mutation self-test,
   affected host suites and relevant complete GPU suites pass. Failures,
   skips and unmeasured gates remain separate.
5. README, AGENTS, tasks0031/0034/0035/0036, experiment0006 and the support
   matrix agree with the frozen evidence. The owner reviews the committed diff;
   only the owner declares M3 accepted.

Stop for owner direction if review requires a tolerance change, repack retention
decision, bulk artifact write, new canonical precision, or acceptance of an
unmet clause. Ordinary defects and missing regressions are engineering work.

## Result, filled after work

Implementation candidate and focused gates are in progress. Final source
identity, battery totals, complete gate table, review disposition, and owner
decision will be recorded here without converting partial runs into evidence.
