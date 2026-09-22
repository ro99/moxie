# Task 0051 — M4 closure: 100k/200k/1m admission and positional-capability report

Status: proposed

## Identity and authority

- Task0051, the second and final milestone-closing task for M4. Not a new
  roadmap sub-item. Builder Codex `luna` (max, `/ponytail:ponytail`);
  independent reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Repository owner
  accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `5270780` (task 0050's acceptance commit). Confirm `git status`
  clean and `HEAD` unmoved before starting; report if not. The tree may
  carry unrelated dirty work from a separate session (an ADR touching the
  roadmap file, observed 2026-09-21) — preserve it, do not stage or revert
  it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- **Why this task exists, stated precisely:** M4's exit gate
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md)) has
  **two** sentences, and task 0050 answered only the first. The second:
  "**For 100k/200k/1m publish admission and positional-capability results
  separately from measured performance; blocked tiers are not
  supported-by-assertion.**" Nothing built across tasks 0037–0050 has
  published anything about these tiers — not even a "blocked" statement.
  Silence is not the reporting the gate requires. This task closes that
  gap. It cites R04, R18, R19, R21
  ([08-strata-reference-map.md](../spec/08-strata-reference-map.md)):
  R19 ("width is not actual context... model size or nominal kernel width
  does not establish long-context efficiency") is the specific discipline
  this task must not violate — a report that inflates a computed bound
  into a claim of working support would repeat exactly the mistake R19
  documents.
- **This is expected, in advance, to conclude some or all of these tiers
  are blocked — that is a legitimate, complete result, not a failure to
  avoid.** `moxie_memory::HostBackedPlan::MAX_STAGED_BLOCKS` is currently
  `3` (`crates/moxie-memory/src/report.rs:84`) — the only qualified
  host-backed streaming bound, proven at that exact value by tasks
  0041–0043. With the 32K gate's own geometry (`page_tokens: 256`), that
  bound admits at most 768 rows beyond whatever fits device-resident —
  nowhere close to bridging real single-GPU VRAM capacity
  (`docs/evidence/hardware-inventory.md`: aggregate **62.6 GiB** across
  three GPUs, no TP/multi-GPU partitioning exists yet — M5, unstarted — so
  a single active generation is bound by one GPU's VRAM, not the
  aggregate) up to 100k, 200k or 1m tokens for any realistic per-token KV
  byte cost. If the arithmetic below confirms this, the correct Result is
  "100k/200k/1m are blocked, by this exact margin, pending N-block
  generalization beyond the qualified N=3" — not a strained attempt to
  make a larger tier appear reachable.
- Required documents: M4's full exit-gate text (quoted above), R04/R18/R19/
  R21 (`docs/spec/08-strata-reference-map.md:35,119,125,137`).
- Required source reading: `docs/evidence/hardware-inventory.md` (actual
  measured VRAM, not a spec sheet number); `crates/moxie-memory/src/report.rs`
  (`HostBackedPlan::MAX_STAGED_BLOCKS`); `xtask/src/gpu.rs`'s
  `paged_attention_32k` (the geometry and per-token byte accounting this
  report's arithmetic must use, not a newly invented one — `kv_heads: 2,
  head_dim: 128, page_tokens: 256`, BF16); `crates/moxie-oracles/src/mask.rs`
  (the positional/masking arithmetic — `attend_row`, `page_of` — the
  functions this task's positional-capability check exercises at large
  position values).
- O1–O5 resolved; O6/O7 open — this task **is** the admission/positional
  report the exit gate itself asks for; it must not additionally smuggle
  in a performance claim while doing so ("separately from measured
  performance" is the gate's own instruction, not this task's paraphrase).

## Bounded deliverable

- **One concrete outcome, two parts:**
  1. **Admission arithmetic**, published as a table: using the already-
     qualified 32K gate's exact per-row byte cost (`kv_heads *
     head_dim * 2 (K+V) * 2 bytes (BF16) * layers` — state the layer count
     assumption explicitly, since the 32K gate tests one layer; scale
     honestly, do not silently assume one layer generalizes for free),
     compute: (a) how many actual tokens fit device-resident-only within
     one GPU's actual measured VRAM headroom (not total VRAM — leave
     realistic headroom for weights/activations/workspace, name the
     assumption), (b) how many additional tokens the qualified N=3
     host-backed streaming bound adds on top of that, (c) whether 100k,
     200k and 1m are each reached, partially reached, or blocked by that
     arithmetic, with the exact numeric margin. This is arithmetic over
     already-published facts (hardware inventory, task 0042/0043's
     qualified bound) — it does not require new device execution to
     produce, but its inputs must be the *actual* measured/qualified
     numbers, not assumed ones, and must be checked against source.
  2. **Positional-capability check**: exercise the existing masking/RoPE-
     adjacent oracle arithmetic (`moxie-oracles`'s existing functions —
     do not add new positional math) at position values representative of
     100k, 200k and 1m (not necessarily inside a full context of that
     size — the check is whether the *arithmetic itself* stays correct
     and finite at those position magnitudes, which is testable without
     admitting a context that large), against the same FP64 oracle
     discipline every prior task used. Report pass/fail per tier
     separately from the admission table — the gate asks for them
     reported separately because they can fail independently (a position
     value could be arithmetically fine while the byte budget is still
     blocked, or vice versa in principle).
- **Sole owning shared component:** none new — this task adds no
  production capability. Its only "component" is the published report
  itself (a new evidence document, per
  [docs/README.md](../README.md)'s placement contract — likely under
  `docs/evidence/`) plus, if genuinely needed, host-only oracle tests
  proving the positional arithmetic at large values (in
  `crates/moxie-oracles`'s existing test modules, not a new module,
  unless the builder judges a new module clearer — say why if so).
- **Allowed production and test files/modules:** a new evidence document
  under `docs/evidence/`; test additions to existing `moxie-oracles` test
  modules only if needed to produce the positional-capability evidence
  (no new production functions — the existing masking/positional
  arithmetic is what's being tested, not extended). Do not touch
  `moxie-memory`, `moxie-state`, `moxie-executor`, kernels, or `xtask`'s
  GPU gates — this task reports on the existing qualified bounds, it does
  not change them.
- **Explicit non-goals and forbidden shortcuts:** no new streaming
  generalization (raising `MAX_STAGED_BLOCKS` beyond 3 is explicitly a
  separate, unopened follow-up task, not this one — if the arithmetic
  shows 100k/200k/1m are blocked, name that as the next bounded task
  rather than trying to close the gap here); no actual 100k/200k/1m-token
  device execution (this task is a report, not a new capacity-qualifying
  gate — if the owner later wants an *executed* 100k-token gate, that is
  its own future task with its own real hardware evidence); no
  performance measurement of any kind; no rounding a "blocked" result up
  to "supported with caveats" — R19's exact warning.
- **Existing consumers and second-consumer/shape proof:** not applicable —
  this is a report, not a mechanism.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** the admission arithmetic is stated in full above; use it
  exactly, show the working, do not hide intermediate numbers.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  BF16 throughout, matching every prior M4 gate.
- **Partition and hardware capabilities:** single-GPU bound is the
  correct assumption today (no TP/PP exists — M5 is unstarted); state
  this explicitly as the reason multi-GPU aggregation is not used in the
  admission arithmetic.
- **Peak memory and transfer dependencies; source/lease lifetime:** not
  applicable — no new allocation.
- **Cancellation, failure and rollback behavior:** not applicable.
- **Independent oracle; predeclared numerical metrics/thresholds:** the
  positional-capability check's oracle is the same FP64 discipline every
  mask/attention oracle in this repo already uses; state the finiteness/
  correctness criterion before running the check, not after seeing a
  result.
- **Application compatibility and sampler implications:** none.

## Acceptance

- The admission table is published with its exact assumptions named
  (layer count, VRAM headroom reserved for non-KV use, per-row byte
  cost) and cites its source facts (`hardware-inventory.md`,
  `MAX_STAGED_BLOCKS`) rather than restating them from memory. Each of
  100k/200k/1m is marked reached, partially reached, or blocked, with the
  numeric margin shown, not asserted.
- The positional-capability check runs the existing oracle arithmetic at
  position values representative of each tier and reports pass/fail per
  tier, separately from the admission table, with the FP64 comparison
  shown.
- `cargo test --workspace`, both clippy lanes, `arch-check`, `spec-check`
  pass (only if test additions were made; if this task is pure arithmetic
  plus oracle checks with no code change beyond tests, say so plainly).
- Support-matrix and M4 ledger entries: record the published tiers exactly
  as reached/partial/blocked — no "supported" language for a blocked or
  partially-reached tier.
- Deletion and documentation gates: the new evidence document is the
  deliverable; link it from the M4 ledger.
- **Exact condition requiring owner direction or task rejection:** if the
  admission arithmetic or positional check cannot be produced from
  already-published facts and existing oracle functions without new
  device execution or new production math, stop and report exactly what
  additional evidence-gathering (e.g. a real device probe) would be
  needed, rather than estimating it.

## Result, filled after work

- Changed shared owners and consumers; source commit:
- Commands and result IDs; passed / failed / skipped separately:
- Measured effect and uncertainty:
- Deleted/replaced paths:
- Remaining blockers and next bounded task:

Do not fill acceptance with "M4 is done" or "long context supported" for any
tier this task finds blocked or partially reached. This task's Result,
together with task 0050's, is the complete evidence package for the owner's
M4 exit-gate decision — neither task declares the milestone closed on its
own.
