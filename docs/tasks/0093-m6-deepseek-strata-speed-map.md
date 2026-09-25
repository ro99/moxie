# Task 0093 — map every Strata DeepSeek speed technique onto Moxie

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`. Documentation only: no code, no GPU runs.

## Identity and authority

- Owner ruling (2026-09-25): Moxie is Strata's successor, and Strata's main
  effort and success was DeepSeek (`Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound`,
  v1 catalog entry 5). Moxie must eventually show DeepSeek speed at least
  equal to Strata's. The comparison itself cannot run in M6 (the DeepSeek
  adapter, compressed sparse attention, mHC and Engram arrive in M7). What M6
  does now: list every technique that makes Strata's DeepSeek fast, and say
  where each one lives in Moxie's generic design, so a gap is found now, not
  in M7.
- M6 connection: roadmap M6 "shared performance paths"; this map tells slice
  6, M6.5 and the M6 closure which shared paths DeepSeek will need.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Strata checkout
  `/home/rodrigo/Developer/strata` is **read-only**: no build, no run, no
  write, no `git` command that changes it. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.** No GPU use.

## Sources to read (Strata, all of them)

1. `docs/models/deepseek.md`, `docs/deepseek-v4-runtime.md`,
   `docs/dsv4-rank-local-architecture.md`, `docs/flash-attention.md`.
2. `src/models/deepseek/*.cpp`, `src/models/deepseek/detail/`,
   `include/strata/models/deepseek/`, and every CUDA kernel file those
   include or launch (follow the includes; list the kernel files you read).
3. Every file in `experiments/docs/experiments/` whose name or content is
   about DeepSeek/DSv4 (about 128 by name). Accepted and **rejected**
   experiments both count: a rejected one is a row with outcome "rejected".
4. `git log` of Strata (read-only) for DeepSeek performance commits, to find
   techniques the docs do not describe.

Moxie side, to decide where each technique lives:
`docs/spec/02-architecture-and-common-api.md`,
`03-memory-formats-and-cuda.md`, `04-attention-parallelism-and-speculation.md`,
`06-implementation-roadmap.md` (M6–M12), `08-strata-reference-map.md`,
`strata-arch-diagnosis.md`, the accepted task contracts in `docs/tasks/`, and
the code under `crates/`.

## Deliverable

One new file `docs/evidence/deepseek-strata-speed-map.md`, plus this task's
Result. Nothing else changes.

### Section 1 — Strata baseline (recorded figures only)

A table of Strata's best **recorded** DeepSeek numbers: prefill tok/s and
decode tok/s per prompt length and topology, with the Strata commit or doc
line, the GPUs used, the command/flags, and whether it is a median of
repeated runs or a single screen (`docs/models/deepseek.md` about 191–212 is
the starting point). Copy figures exactly; do not re-run anything. Say which
figure is Strata's accepted production result.

### Section 2 — the technique map

One row per technique. A technique is anything that made Strata's DeepSeek
faster or cheaper in memory, or was tried and rejected. Columns:

| Column | Content |
|---|---|
| # | row number |
| Technique | one line, plain words |
| Phase | prefill, decode, both, load |
| Strata evidence | file:line and/or experiment file; the measured effect as recorded (numbers exactly, with unit); "not measured" if none |
| Outcome in Strata | accepted, opt-in, rejected |
| Generic or DeepSeek-only | whether the technique depends on DeepSeek's equations (sparse attention, mHC, Engram, routing) or is a general engine technique |
| Moxie home | exactly one of: **present** (name the Moxie file/type/task that provides it), **planned** (name the roadmap item, e.g. M6.3, M7 DeepSeek row), **gap** (no place in Moxie's spec or code) |
| Note | at most two lines: what differs, or what a gap would need |

Rules:
- Every row cites evidence you read. No row from memory or inference.
- A technique that Moxie's policy forbids (for example an FP8/FP4 cache under
  the ≥16-bit cache rule, document 03) is **gap** with the rule cited, not
  "planned".
- Do not propose designs. A gap row's note names what is missing, not how to
  build it.
- Group rows under headings: weights and experts (residency, tiers, host
  MoE), attention and KV, multi-GPU (rank-local, TP, transfers), kernels and
  launch (fusion, graphs, streams), prefill specifics, decode specifics,
  memory and admission, other.

### Section 3 — summary

Counts of present / planned / gap, the list of **gap** rows by number, and
the list of techniques whose recorded effect was largest (top ten by the
recorded number, with the number). No opinion on whether Moxie will be fast.

## Acceptance

- The file exists with all three sections; every row has a citation that the
  reviewer can open.
- `cargo xtask spec-check` passes (docs links).
- Only the two files above are committed; the Strata checkout is unchanged
  (`git -C /home/rodrigo/Developer/strata status --short` identical before
  and after; state both in the Result).

## Stop conditions

- A source needs a build or a run to read a number: stop, record "not
  recorded" in the row, continue. Do not run Strata.
- More than 150 rows: stop and send `DECISION` with the grouping you propose.

## Result, filled after work
