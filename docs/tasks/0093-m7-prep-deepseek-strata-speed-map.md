# Task 0093 — M7 preparation: map every Strata DeepSeek speed technique onto Moxie

Status: **active** (coordinator, 2026-09-25). Builder Claude Opus `builder`
(owner direction, 2026-09-25); reviewer Codex `sol`. Documentation only: no code, no GPU runs.

## Identity and authority

- Owner ruling (2026-09-25): Moxie is Strata's successor, and Strata's main
  effort and success was DeepSeek (`Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound`,
  v1 catalog entry 5). Moxie must eventually show DeepSeek speed at least
  equal to Strata's. The comparison itself cannot run in M6 (the DeepSeek
  adapter, compressed sparse attention, mHC and Engram arrive in M7). What
  this task does now: list every technique that makes Strata's DeepSeek fast, and say
  where each one lives in Moxie's generic design, so a gap is found now, not
  in M7.
- **Milestone: M7 preparation** (owner, 2026-09-25), run ahead of time and
  in parallel. It is not part of M6's exit, and M6 does not wait for it or
  reorder around it. Its rows feed M7's DeepSeek work; gap rows are
  registered for M7.
- Builder: Claude Opus `builder`, in its own Herdr tab (owner direction).
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
| Moxie home | exactly one of (amended by the coordinator after review round 2, 2026-09-25): **present** (the cited Moxie code does what the technique does, by the same or an equivalent mechanism with the same effect; name the file/type/task); **partial** (Moxie has part of it; the Note names the missing part); **not needed** (Moxie's design removes the cost the technique addressed, e.g. KV already device-resident so no transfer to optimize; cite the code); **planned** (a named roadmap item covers this exact technique, not merely its area); **gap** (none of the above, including a rule that forbids it) |
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

Builder Claude Opus `builder`, 2026-09-25. Deliverable:
[deepseek-strata-speed-map.md](../evidence/deepseek-strata-speed-map.md).

- **Row count and grouping.** The first pass found about 280 raw candidate
  rows, about 200 after merging duplicates. The builder sent `DECISION`. The
  coordinator's `ANSWER` approved the grouping: one row per technique; one
  row per tuning campaign, with the accepted gain first and each rejected
  variant cited by `file:line`; distinct accepted techniques split; and
  techniques tied to FP4/FP8 weights or a KV cache below 16 bits made gap
  rows, with the rule cited and what they bought in the Note. The evidence
  file's introduction records this rule. **Result: 117 rows** under the
  contract's eight headings.
- **Sources read.**
  - **Experiments:** all 133 DeepSeek experiment records, 0006–0195,
    including the non-`dsv4`-named DeepSeek ones.
  - **Docs:** the four Strata docs.
  - **Code:** the 16 headers, the 21 source and `detail/` files, and 7
    kernel files. They are listed in the evidence file.
  - **History:** about 235 Strata commits, read with `git log` and
    `git show` only.
  - **Verification:** the builder re-opened every Section 1 figure, every
    top-ten figure and every Moxie line cited.
  - **Discrepancies:** three disagreements between Strata sources are
    recorded, not resolved.
- **Baseline.** Strata's accepted production result is 26.231 prefill tok/s at
  1,925 prompt tokens (median of three) and 8.627 decode tok/s (median), on two
  RTX 3090s with rank-local TP2 (`docs/models/deepseek.md:191-201`).
- **Homes (after review round 1).** 28 present, 46 planned, 43 gap. The
  gap rows fall into five groups:
  - **13 forced by representation rules:** rows 6, 7, 15, 18, 36, 45, 46, 77,
    78, 79, 80, 81, 114.
  - **Forced by ADR 0036:** row 70.
  - **Forced by the single-user rule:** row 76.
  - **Rejected in Strata:** rows 10, 30, 69, 74, 104.
  - **Host-attention tuning:** rows 89, 90, 91.
  - **Accepted in Strata with no place in Moxie (20):** rows 4, 5, 24, 31,
    32, 33, 34, 35, 40, 43, 49, 61, 67, 71, 82, 87, 92, 94, 107, 110. Most
    of these are host expert execution and loading.
- **Checks.**
  - `cargo xtask spec-check` passed: 10 documents present and unchanged.
  - The evidence file's relative links resolve.
  - `git -C /home/rodrigo/Developer/strata status --short` was identical
    before and after: `?? .pi/` and `?? tests/p2p/`. HEAD stayed
    `2dc566e`.
  - Nothing was built or run, and no GPU was used.
- **Committed.** Only this contract and the evidence file. The carried
  `.gitignore`, `docs/evidence/specification-version.md`, ADRs 0034/0035, and
  another agent's uncommitted `crates/moxie-plan` edits were left unstaged. One
  row cites `crates/moxie-plan/src/tensor_parallel.rs:498` at HEAD `60fc509`,
  because that file has uncommitted edits in the shared tree.

**Review rounds.**
- **Round 1** (Codex `sol`, source `080fedc`) found 7 high, 5 medium and 2
  low findings.
  - **Every high finding was one class:** a row marked present where Moxie
    had only a related mechanism. These were rows 24, 31, 40, 61, 67, 75,
    93, 94 and 107.
  - **The medium findings:**
    - Rows 69, 71 and 110 were also related mechanisms counted as present.
    - Row 87 misread two GPUs' figures as a before/after pair.
    - Row 78 used a slowdown as its gain, and that figure was also in the top
      ten.
    - Row 115's gap was contradicted by
      `docs/evidence/benchmarks/schema.md:57-61`.
  - **The low findings:** row 83 mapped to M6.3, not M6.1; rows 45 and 46
    cite ADR 0019 as already settled.
  - **All were fixed.**
    - Row 78's gain is now 21.67 s → 4.02 s. It left the top ten, and row
      60's recorded "228.7x fewer" physical page bytes entered it.
    - Row 87 is now gap.
    - Row 115 is now present.
- **Re-audit.** The repair then re-audited every remaining present row
  against the Moxie code. **11 more present rows changed**: 13, 37 and 86 to
  planned; 30, 43, 49, 74, 89, 90, 91 and 92 to gap. Each Note says what
  differs.
- **Net effect of round 2.** Present fell from 50 to 28, planned went from 42
  to 46, and gap went from 25 to 43.

