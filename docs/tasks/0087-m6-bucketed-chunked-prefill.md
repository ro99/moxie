# Task 0087 — chunked prefill through a fixed, reused set of bucket plans

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0087, M6 slice 4, roadmap **M6.2** "shape-bucket chunked prefill for
  later pages and tails … all graph pools and workspaces admitted". Document
  04: "Test position zero AND later chunks, partial widths, short tails, and
  continuations." Document 02: plan cache keys include the workload bucket.
- **Placement decision (coordinator, 2026-09-25):** the bucket policy is a
  pure function in `moxie-plan`; the step loop stays in the test, as the
  timing harness does. Wiring device generation into `moxie-engine` needs a
  new dependency edge beyond ADR 0011's host-reference profile and is put to
  the owner separately; this task does not need it.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.

## Facts established before writing (coordinator, 2026-09-25)

- A dense plan is lowered for an exact `ResourceWorkload.rows`; today every
  prompt length gets a fresh plan. A plan is reusable across steps (tasks
  0079, 0085) and its RoPE tables, module and captured graphs belong to it.
- `dense_gemma_device.rs` has `stage_bindings` (about 157), `admit_runs`,
  `geometry`, `commit_paged_state` (executor), `host_step` (about 520, the
  host reference for a run of tokens at given positions) and `assert_logits`.
- Continuation prefill (rows > 1 at a later position) is already correct on
  the device path (M4 task 0050 compared whole, chunked and multi-turn).

## Bounded deliverable

- **Outcome:** a bucket policy splits any prompt length into chunks drawn
  from a fixed bucket set; one plan per bucket is admitted once and reused
  for every chunk of that size, across two different prompts; device logits
  match the host reference.
- **Allowed files:** `crates/moxie-plan/src/selected.rs` (or a new
  `crates/moxie-plan/src/bucket.rs` plus its `lib.rs` export, if
  `selected.rs` has no natural place), `crates/moxie-executor/tests/
  dense_gemma_device.rs`, this task's Result.
- **Non-goals:** `moxie-engine` changes; padding rows; picking bucket sets
  from cost; TP/PP.

## Numbered changes

1. **Policy** in `moxie-plan`: `pub fn prefill_chunks(rows: u64, buckets:
   &[u64]) -> Result<Vec<u64>, Error>`. `buckets` must be non-empty, strictly
   ascending, contain `1`, and every entry nonzero (typed refusal
   `invalid("buckets", …)` otherwise); `rows` must be nonzero. Greedy: take
   the largest bucket not exceeding what remains, repeat. The result sums to
   `rows` exactly. Fallible allocation as the crate does elsewhere.
   **One unit test**, table-driven: `(13, [1,2,4,8]) → [8,4,1]`,
   `(8, [1,2,4,8]) → [8]`, `(3, [1,4]) → [1,1,1]`, and the refusals (empty,
   unsorted, missing 1, zero rows).
2. **GPU test** `bucketed_prefill_reuses_plans_and_matches_host` in
   `dense_gemma_device.rs`, on every GPU, Shape A, buckets `[1, 2, 4, 8]`:
   - lower and admit **one plan per bucket** up front (`rows = bucket`);
     enable segment capture on each (task 0086's `set_segment_capture(true,
     &mut ledger)`);
   - **prompt 1** of 13 tokens: for each chunk from `prefill_chunks(13, …)`,
     begin a transaction, execute the chunk's bucket plan at its absolute
     positions, finish, commit; keep the last chunk's logits; then one decode
     step with the `1` plan;
   - **prompt 2** of 7 tokens (chunks `[4, 2, 1]`) on a fresh
     `DeviceKvSequence` and fresh runs, **reusing the same plans** (their
     captured graphs replay);
   - compare each prompt's last prefill row and its decode row with
     `host_step` over the whole prompt, using `assert_logits`;
   - at the end close every plan and run and assert
     `ledger.outstanding().is_empty()`.

## Contract before implementation

- **Semantics:** chunking changes only which rows each step computes; each
  row attends to exactly the history the whole prompt would give it.
- **Resources:** all bucket plans are admitted simultaneously (their sum is
  charged, including `GraphPools`), which is the fixed set document 04 asks
  for.
- **Failure:** unchanged per step.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`,
`dense_tp2_device`, `cargo xtask-cuda test-gpu` (69/69).

**Coverage check (one mutant, reverted after):** in the GPU test, pass each
chunk's positions starting at 0 instead of its absolute offset; the test must
fail.

**Stop conditions:** a logit comparison fails the existing gate (send
`DECISION` with the worst error; do not change the gate); a change conflicts
with the code; a file outside the allowed list is needed.

## Result, filled after work

(pending)
