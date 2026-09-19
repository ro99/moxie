# Task 0036 — reconcile and close M3.1 / M3.2 / M3.3

Status: closure candidate complete; owner review and M3 decision pending,
2026-09-19.

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

The implementation candidate is `0672d24`, comprising `85daa05` (`Execute
mapped dense weights and close importer coverage gaps`) plus the review repair
`0672d24` (`Version the mapped affine launch ABI`). It adds the last known
execution/importer coverage without changing the affine equation or ADR0028
gate:

- dense CUDA execution consumes an admitted `u32` group map per logical input
  column; map bytes participate in residency, extent, device and completion
  lifetime checks, and a missing map refuses before enqueue;
- the mapped INT4/group32/F16 dense tail case passes on both SM86 GPUs and SM120
  with all 240 outputs per GPU inside ADR0028 (worst zero ULP in this case);
- generated GPTQ plans combine a noncontiguous map with multiple passthrough
  override patterns, including MTP, and preserve both BF16 overrides;
- both pinned read-only AutoRound configurations have their group declaration
  and model-specific override families checked; and
- a fused leading expert axis refuses by name instead of being flattened or
  interpreted as an exporter-specific interleave.

Focused evidence at that identity:

| Gate | Result | Evidence |
|---|---|---|
| Dense affine unit tests | 11 passed | `cargo test -p moxie-executor affine_linear --lib` |
| Mapped dense device case | SM120 and both SM86 passed | `results/m3-recovery/mapped-dense-device.log` |
| Complete grouped-device suite | 11 passed on `0672d24` | `results/m3-recovery/m3-final-grouped-device.log` |
| Compressed-tensors interleave refusal | passed | affected `moxie-format` suite |
| GPTQ map x override publication | passed | affected `moxie-repack` suite |
| Pinned AutoRound samples/configs | 43,008 values bitwise plus config checks | `results/m3-recovery/autoround-samples.log` |
| Affected host suites | passed | `results/m3-recovery/m3-final-affected-host.log` |
| Driver clippy | passed on `85daa05`; repeated after ABI repair | `results/m3-recovery/m3-final-driver-clippy.log`, direct `cargo clippy` run |
| Architecture/specification checks | 79 rejected, 21 accepted, 13 rules; 10 documents | `results/m3-recovery/m3-final-arch.log`, `m3-final-spec.log` |
| Mutation driver self-test | 111/111 | direct `cargo xtask mutation-check --self-test` run |
| Final T0006 | 63/63 caught, 3/3 controls held, zero other verdicts | `results/m3-recovery/t0006-final-0672d24.log` |
| Final T0028 | 24/24 caught, 1/1 control held, zero other verdicts | `results/m3-recovery/t0028-final-0672d24.log` |

Read-only review of `85daa05` found no P0 or P1 code defect and judged the M3
candidate sound. It identified the stale support rows, two comments and the
unchanged ABI version after the CUDA signature gained its map parameter. The
support rows are reconciled here; `0672d24` scopes the comments and moves both
the symbol and declared ABI from v1 to v2. The six-case dense device suite then
passed again on SM120 and both SM86 GPUs, and driver clippy remained clean.

The decisive batteries passed from a clean detached worktree at exactly
`0672d24`. T0006 caught all 63 mutants and held all three controls, with all 12
lanes stable three times before and after. T0028 caught all 24 mutants and held
its one control, with all six lanes stable three times before and after. Both
reported zero survivor, unstable, invalid, broken or skipped verdicts, exited
zero, restored a clean tree and left HEAD unchanged. Their log hashes are
`b5e61d449ad87661f230b6b3147b5d4045a3d584d3f86a680722dde3e719fc1c` and
`adb5a19f543026d9197b446419e0f28b8e8cd427fcd06e25ddb4d73596a3ce9b`,
respectively. The five earlier publication survivors are caught in the final
T0006, and both mapped-execution mutations are caught in the final T0028.

Acceptance clauses 1 through 4 now have committed implementation and measured
evidence, and the records named by clause 5 agree with that evidence. The
candidate makes no model-output, quality, token-generation or performance
claim, and experiment0007 remains pending. Clause 5's repository-owner review
and the owner's M3 decision remain.
