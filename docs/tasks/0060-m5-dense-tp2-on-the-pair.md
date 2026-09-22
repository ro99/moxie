# Task 0060 — the reduced dense Gemma graph runs TP2 on the 3090 pair

Status: **proposed**.

## Identity and authority

- Task0060, second half of M5 plan slice 2. Builder Codex `luna` (max,
  `/ponytail:ponytail`); reviewer Codex `sol` (read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus `coordinator`.
  Accepted by the coordinator under the owner's auto-mode delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `9b27041` (task 0059
  acceptance). Preserve the unrelated carried work
  (`docs/evidence/specification-version.md` and ADRs 0034 and 0035).
- Requirements:
  - Roadmap M5.2: "TP2 on the 3090 pair: column/row linears, head/KV
    ownership, global … vocabulary operations, collective ordering and
    failure handling. Compare to single-rank reference."
  - [ADR 0036](../decisions/adr/0036-tp-reductions-are-exact-by-declared-order.md):
    exact reductions; the single-rank path runs the same declared order.
  - Document 04: a failed rank invalidates the in-flight transaction.
  - The M5 ledger's slice 2 obligations: per-rank device KV (M5.1-f), a
    whole rank step as one transaction, and an order-sensitive fixture.
- O6/O7 are open: no timing.

## Facts established before writing (coordinator, 2026-09-22)

**Moxie:**
- Task 0058's `moxie_plan::lower_tensor_parallel` produces
  `Local { nodes, join }` stages, where join is a Gather or an FP32 Reduce,
  together with per-rank params, row ranges and input slices. The per-rank
  **stage subgraph** is built only by the test harness today
  (`moxie-cli/tests/tensor_parallel.rs` `stage_graph`). A device consumer
  needs that construction in production code.
- Task 0059's `SelectedReservedPlan::execute_dense`
  (`moxie-executor/src/dense.rs`) runs one selected graph for one step on
  one GPU, with paged attention and admitted host uploads.
- Task 0057's `RankGroup` (`moxie-executor/src/tensor_parallel.rs`) runs a
  sequenced, peer-checked BF16 all-gather on the 3090 pair. Failures are
  typed and bounded, and a missed drain withholds every range. It has **no
  reduce**, and there is no FP32-output partial `Linear` kernel.
  `moxie_bf16_linear_v1` rounds to BF16.
- `moxie-state` has `SequenceState` transactions with `begin` / `abort` /
  `commit_prefix`.

**Strata** (read-only):
- `docs/dsv4-rank-local-architecture.md`, "Failure and rollback": on failure
  "both rank chains abort; every token-local KV … mutation is truncated; all
  outputs are withheld and zeroed — no partial token is ever published; the
  failed command is drained by a single owner; … Post-failure reuse is exact,
  and is tested." This is the model for rank-step atomicity.
- "Collectives": the data reduce is an FP32 sum of rank partials, with one
  BF16 rounding after.

## Bounded deliverable

- **Outcome:** the reduced dense Gemma 4 graph (unchanged model code),
  lowered by task 0058 for R = 2, executes across the two 3090s:
  - prefill, then decode;
  - per-rank stage graphs on each GPU;
  - task 0057's gather for the head output and the vocabulary;
  - a new exact FP32 reduce for `down_proj` and `o_proj`.

  Logits must be **bit-identical** to the same graph on one 3090 running the
  declared split S = 2 (ADR 0036).
- **New pieces:**
  - An FP32-output partial `Linear`, which is `moxie_bf16_linear_v1`'s loop
    without the final rounding.
  - An exact reduce collective on `RankGroup`: exchange the FP32 partials by
    peer copy, add them in ascending rank order in FP32, round to BF16 once.
    It keeps 0057's sequence and shape agreement, bounded settle, and
    drain-or-withhold.
  - The single-GPU device path honours a declared split S, so it can serve as
    the reference.
  - Production construction of the per-rank stage graph (in `moxie-plan`, or
    justify another owner). The harness helper is then deleted or made to
    call it.
- **Rank-step atomicity, Strata's model:** each rank's step runs inside one
  `SequenceState` transaction covering every layer's KV append. If any rank
  or collective fails, both ranks abort, no logits are published, and the
  group stays reusable. The next clean step then matches the reference
  exactly.
- **Order-sensitive fixture:** choose weights or inputs for which S = 2 and
  S = 1 produce *different* logits bits. Assert that difference once, so the
  bit-identity test can detect a wrong combine order.
- **Non-goals:** no thread-per-rank (slice 3); no status collective
  (slice 3); no 5060 Ti; no MLA or routed ops; no timing; no model edits.

## Phase 1 — design proposal before code

Send a `DECISION` report of 40 lines or fewer covering:

1. **Stage-graph construction:** its owner, and how it relates to the
   harness helper.
2. **How a TP step drives the per-rank stage graphs on the device**
   (`execute_dense` per stage, or another entry), and where the gather and
   reduce sit between them.
3. **The reduce collective:** buffers, order, rounding, and reuse of
   task 0057's settle and withhold paths.
4. **Atomicity:** how one transaction per rank spans the stage runs, and
   what abort does.
5. **The order-sensitive fixture.**
6. **Whether this fits one task,** with files and any new crate edge.

## Acceptance

- Host, driver and GPU lanes pass: fmt, clippy, workspace tests,
  `arch-check`, `spec-check`, and the GPU tests with the correct features on
  the 3090 pair (UUIDs in the output).
- **Bit-identity:** TP2 logits equal the one-3090 S = 2 reference, over
  prefill plus decode, on the order-sensitive fixture.
- **Failure:** inject one rank failure and one collective mismatch
  mid-step. Both ranks abort, KV is unchanged, no logits are published, and
  the next clean step is bit-identical.
- **Mutations,** each run and restored:
  - Reduce in descending order.
  - Round a partial before the reduce.
  - Reference ignores S.
  - Abort only the failing rank.
  - One rank keeps its KV append after a failure.
- One test per invariant. Stop conditions: a lease, ledger or `SequenceState`
  semantic change; bit-identity failing for a reason that is not a defect;
  a model edit.

## Result, filled after work

- Design decision (phase 1) and coordinator answer:
- Changed owners and consumers; source commit:
- Commands, GPU UUIDs; passed / failed / skipped:
- Mutation results and restoration:
- Remaining obligations:
