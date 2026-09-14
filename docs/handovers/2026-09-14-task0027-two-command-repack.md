# Handover — task 0027: a two-command repack, and repacking made provisional

**Deferred as unfinished, by owner decision on 2026-09-14.** Reviewed four
times, not accepted, and explicitly **not a prerequisite for quantized
execution** — no further converter review cycle gates anything else.

What works is the two-command path itself: a user points at a checkpoint
directory, gets a TOML plan, and passes that plan to `repack`. No hand-authored
tensor entries, no packing parameters, no five mandatory budget flags.

What is **unmet** is the whole-model workflow. The two largest checkpoints on
this machine are refused, and refusing early with the arithmetic is more honest
than the plan-then-fail it replaced but is not the same as delivering the
workflow.

```console
$ moxie-repack plan   --source-root /fast/models/ORG/MODEL --out-plan ./model.plan.toml
$ moxie-repack repack --plan ./model.plan.toml --out ./model-moxie
```

**Read [ADR 0027](../decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)
before building on any of this.** Repacking is **provisional**. The owner's
requirement is that it brings worthwhile inference performance improvements;
no such measurement exists and none is runnable until shared execution does.
Canonical layout, safetensors packaging, compact plans, byte correctness and
importer simplicity are **not** evidence of faster inference, and nothing in
this handover should be read as such.

## Workspace identity

- `/home/rodrigo/Developer/moxie`, branch `main`, built on `19073a1`.
- `/fast/models` is a **read-only** input. `plan` was run read-only against
  every present root; the plans were written to a scratch directory. Nothing
  under `/fast/models` was written, and **no checkpoint was converted**.

## What works, measured

- The two-command path, end to end, on fixtures: plan, repack, and verification
  through the production reader. Fourteen integration tests in
  `crates/moxie-repack/tests/two_command.rs`.
- Budgets are derived from the checkpoint and reported, and any explicit flag
  overrides **that field** without discarding the others.
- A plan binds its checkpoint by content — `config.json`, the index, and the
  index's tensor count — and `repack` refuses a plan whose checkpoint has moved.
- `plan` never writes inside the source root: not the plan, not the staging file
  it is renamed from, and not through a symbolic link pointed at either.

## The capability limit this round exposed

**The two largest local checkpoints cannot be converted at any budget.** Every
work unit appends one resume-journal record, a unit is cut inside one canonical
component and never across two, and the journal cap is 16 MiB. So the minimum
journal is one record per component:

| Root | Selected | Components | Minimum journal | Outcome |
|---|---:|---:|---:|---|
| `Laguna-S-2.1-AWQ-INT4` | 36,769 | 106,249 | 61,522,267 B | **refused** |
| `Inkling-Small-AWQ-INT4` | 31,688 | 93,448 | 55,138,416 B | **refused** |
| `Muse-Glimmer-30B-AWQ-INT4` | 406 | — | — | planned, `partial` |
| `Qwen3.8-27B-AWQ-BF16-INT4` | 1,199 | — | — | planned, complete |
| `gemma-4-31B-it-AWQ-8bit` | 1,188 | — | — | planned, complete |
| `google/gemma-4-26B-A4B-it` | 1,013 | — | — | planned, complete |

This is not a regression — `repack` always refused them, through the identical
check on the resolved plan. What changed is **when** the user is told, and with
what. The ceiling is roughly 28,000 components, about 9,000 quantized modules.

**The ceiling belongs to this program's restart journal**, not to safetensors,
the canonical format, or anything about inference: one record per canonical
component, a 16 MiB cap on a journal a resume must read back, and a work unit
cut inside one component and never across two. All three are `moxie-repack`
choices and all three are changeable. Whoever picks this up should revisit the
cap rather than design around it.

The task record's coverage matrix has been corrected: those two roots are no
longer listed as plannable.

## Review history

Nine findings across two rounds on `19073a1`, all corrected, all with a
regression watched to fail before its fix:

- **Round 1 (six).** A split module planned and dropped; a symlinked staging
  path writing into the checkpoint; a stale plan publishing an old subset as
  complete; one budget override discarding the rest; a relative source root that
  did not travel; and a planner that could reject its own plan.
- **Round 2 (three).** A plan bound to a checkpoint it did not describe — the
  binding digests were read a second time, after discovery, and review changed
  the index in that window. The sizing rule raising the scratch against a floor
  it could not move and landing on a tile the reader refuses. And planning still
  writing inside the source root, because the comment said never and nothing
  enforced it.

The first report of round 1 claimed six regressions. There were **five**. That
is recorded in the task, because miscounting one's own evidence is the failure
this task keeps producing.

Two structural changes came out of round 2 and are the parts worth keeping:

- `SourceBinding` no longer **has** digest fields. They come from the
  `Discovery` the plan is written beside, so no caller can bind a different
  moment than the one it describes.
- The planner sizes work units through `StagingEstimate::journal_bound_for`,
  which is the writer's own arithmetic. A planner with a second estimate of the
  writer's rule is a planner that can emit a plan the writer rejects.

## Verification

Measured on this tree, after every correction:

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`: clean.
- `cargo xtask spec-check`: 10 documents present and unchanged.
- `cargo xtask arch-check`: 79 rejected fixtures, 21 accepted, 13 rules.
- `cargo xtask mutation-check --self-test`: 81 of 81 cases correct (10 verdict,
  5 selector, 66 anchor).
- `cargo test --workspace --locked --offline`: **1,076 passed, 0 failed, 0
  ignored**.
- `cargo test -p moxie-executor --features driver --locked --offline`: **104
  passed, 0 failed, 0 ignored**.

**The full mutation battery has not been run on this tree.** It is about four
hours, and the tree has changed in every one of the last four review rounds, so
a battery started before review converges describes a tree that no longer
exists. The owner asked for it against `19073a1`; that commit's code is no
longer the code. It should run once, on whatever tree the reviewer stops
finding changes in, before acceptance.

One host test was also fixed here rather than in scope: `budget.rs` failed with
`attempt to subtract with overflow`. Its own header says two tests measuring one
global allocator race and neither number means anything — and the file had two
such tests, running concurrently. Both measurements now take a mutex. There is
no regression test: the fix removes the concurrency rather than detecting it.

## Remaining hypotheses and blockers

- **Acceptance is blocked.** Four review rounds, nine findings on this commit
  alone. The shape of round 2 — a fix applied to a copy of the data instead of
  the data — is a reason to expect a fifth round, not to assume none.
- **Nothing executes a canonical tensor.** M3 item 3, untouched. It is also what
  makes [experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md)
  runnable, which is what decides whether any of this is retained.
- **Named continuations, deliberately not started:** AutoRound import,
  `actorder: static`, F16 passthrough, the DeepSeek V4 Flash / V4.1 Flash
  revisions, and the journal ceiling above.

## Next task

**M3 shared W4A16/W8A16 execution.** The owner's 2026-09-14 decision makes this
the next work directly: accepting this converter is **not** a precondition. It
is still not a licence to close M3 or begin unlimited runtime work. Document 03 bounds it: weight-only paths, BF16 preferred
with FP32 accumulation, SM86 first and SM120 qualified separately, and bounded
reference dequantization is not an acceptable final fast path by assertion.
