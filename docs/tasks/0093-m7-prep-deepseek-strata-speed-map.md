# Task 0093 — M7 preparation: map every Strata DeepSeek speed technique onto Moxie

Status: **accepted** (coordinator, 2026-09-25) after three sol review rounds
and a coordinator-verified fourth repair. Builder Claude Opus `builder`
(owner direction, 2026-09-25); reviewer Codex `sol`. Commits `080fedc`,
`b6d4bb8`, `d58a039`, `746ce0e`.
- Final homes for 117 techniques: present 11, partial 26, not needed 7,
  planned 24, gap 49. The first pass had 50 present; every review finding
  was a related mechanism counted as the technique itself. Part of that was
  the contract's three-home scheme, amended to five homes in `a349acd`. Documentation only: no code, no GPU runs.

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
- **Homes (after review round 3, five homes).** 11 present, 26 partial, 7 not
  needed, 24 planned, 49 gap.
  - **Partial rows:** 1, 8, 9, 13, 16, 23, 26, 28, 31, 37, 39, 40, 44, 67,
    68, 71, 75, 93, 95, 96, 97, 100, 107, 115, 116, 117.
  - **Not needed:** 41, 48, 61, 89, 90, 91, 102.
  - **Gap, forced by representation rules (13):** 6, 7, 15, 18, 36, 45, 46,
    77, 78, 79, 80, 81, 114.
  - **Gap, forced by ADR 0036:** 70.
  - **Gap, forced by the single-user rule:** 76.
  - **Gap, rejected in Strata:** 10, 22, 30, 58, 69, 74, 86, 103, 104.
  - **Gap, accepted in Strata with no place in Moxie (25):** 4, 5, 17, 20,
    24, 32, 33, 34, 35, 43, 49, 50, 55, 60, 64, 66, 82, 83, 87, 92, 94, 98,
    105, 106, 110.
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
- **Round 2** (Codex `sol`, source `b6d4bb8`) found 9 high and 6 medium
  findings. Most were rows where Moxie has part of a technique, or where its
  design removes the need; neither could be expressed with three homes.
  - **Contract amendment.** The coordinator added the `partial` and
    `not needed` homes (`a349acd`).
  - **Findings resolved:**
    - Rows 1, 8, 9, 39, 44, 68, 95, 100, 96 and 97 are now partial, each
      naming its missing part.
    - Rows 41 and 48 are now not needed.
    - Row 28 is partial: contiguous ranges, not round-robin.
    - Row 115 is partial: fields are recorded, but there is no
      characterization.
    - Row 117 is partial: there is no host expert timing.
    - Rows 13 and 37 were planned only by area, so they are now partial.
  - **Full re-examination.** Every row was re-checked under the five homes.
    Present rows must pass "same or equivalent mechanism, same effect".
    Planned rows must name a roadmap item covering the exact technique.
    Besides the findings, this moved:
    - rows 16, 23, 26, 31, 40, 67, 71, 75, 93, 107 and 110 to partial;
    - rows 61, 89, 90, 91 and 102 to not needed;
    - row 17 to gap (M6.1 does not name split-K shaping).
  - **Changed in round 2:** 34 rows.
- **Round 3** (Codex `sol`, source `d58a039`, the final review round) found
  2 high and 2 medium findings. All were home changes; no number changed.
  - **Row 116:** now partial. No server entry point enforces
    `CUDA_DEVICE_ORDER=PCI_BUS_ID`; `.cargo/config.toml:16` covers only
    Cargo-launched processes.
  - **Rows 50, 55, 58, 60, 64, 66, 98, 105, 106:** now gap. Roadmap line 120
    names DeepSeek's semantic families, not these scheduling, storage or
    kernel algorithms.
  - **Row 110:** now gap. Neither named part exists; `chain.rs:220-223`
    charges a different allocation.
  - **Rows 20, 22, 83, 86, 103:** now gap. M6.2 and M6.3 name only the area,
    and each Note keeps that relation.
  - **Resulting counts:** 11 present, 26 partial, 7 not needed, 24 planned,
    49 gap.
  - **Closing the loop.** The coordinator closed the review loop after round
    3 and verified this repair directly.

