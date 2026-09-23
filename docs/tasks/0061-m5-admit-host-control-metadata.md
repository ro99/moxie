# Task 0061 — admit host control metadata and dedupe the test event gate

Status: **proposed**.

## Identity and authority

- Task0061, first half of M5 plan slice 3 ("Robust rank execution").
  Builder Codex `luna` (max, `/ponytail:ponytail`); reviewer Codex `sol`
  (read-only, `/ponytail:ponytail-review`); coordinator Claude Opus
  `coordinator`. Accepted by the coordinator under the owner's auto-mode
  delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `084258b`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035).
- Requirements:
  - AGENTS.md: "One real memory authority admits all
    persistent/transient/branch/draft resources."
  - Task 0059 round-3 review (sol): three host control-metadata allocations
    live outside every host admission request. The M5 ledger carries them as
    a slice 3 obligation.
  - Task 0057 review: about 75 lines of `cuEventRecord` test-gate setup are
    duplicated.
- O6/O7 are open: no timing.

## Facts established before writing (coordinator, 2026-09-23)

- **Unadmitted host metadata:**
  - `PagedAttentionRun.page_table: Vec<u32>`
    (`moxie-executor/src/paged_attention.rs` field around line 1344, built
    around line 1750 and replaced at 2146 and 2272). It is sized at admit but
    not charged in `resource_request` (around line 4710).
  - `moxie-state` `DeviceKvSequence::page_view_for` (`src/device.rs` around
    line 624) returns a `PageView` whose `table` is a freshly allocated
    `Vec<u32>`. `placements_for` (around line 544) allocates a
    `Vec<PagePlacement>`. Both run while dense K/V staging is live.
- **Duplicated gate:** `crates/moxie-executor/tests/device_arena.rs` (lines
  about 20–60) and `tests/tensor_parallel_device.rs` (lines about 33–105)
  each define the same `BLOCK_NEXT`/`RELEASE`/`TIMED_OUT` state and an
  interposed `extern "C" fn cuEventRecord`. An interposed symbol must exist
  in every test binary that uses it, so the fix is one shared source file
  (for example `tests/support/event_gate.rs`) that each test includes with
  `mod`. It is not a library.
- **Precedent:** task 0059 moved the page-table **upload** buffer into
  admitted workspace and reused it in place. The same approach, preferring
  in-place reuse of admitted buffers over new allocations, applies here.

## Bounded deliverable

- **Outcome:** every host allocation on the paged-attention and device-KV
  paths is either inside an admitted request or eliminated. Prefer, in this
  order:
  1. Write into a buffer that is already admitted and reused, as task 0059
     did.
  2. Borrow an existing structure instead of cloning it.
  3. Add checked bytes to the request.

  The admitted envelope must equal peak host use on these paths.
- **The test gate:** extract it into one shared test-support source file
  used by both test files, with no change in behaviour.
- **Non-goals:**
  - no thread-per-rank or status collective (task 0062);
  - no change to paged-attention numerics or quarantine semantics;
  - no new test file beyond the support module;
  - no timing.

## Acceptance

- Host lanes pass: `fmt`, workspace `clippy`, driver-only `clippy`
  (`-p moxie-executor --features driver`), workspace tests, `arch-check` and
  `spec-check`.
- GPU lanes pass on the listed GPUs:
  - `paged_attention_device` (`driver,paged-attention-binding,paged-attention-test-hooks`);
  - `dense_gemma_device`, `dense_tp2_device` and `tensor_parallel_device`
    (`driver,paged-attention-binding`);
  - `device_arena` (`driver`);
  - the full `cargo xtask test-gpu`, which stays at 63/63.
- **Proof of admission:** extend an existing admission assertion; do not add
  a new test. At peak on the paged path, charged host bytes must be greater
  than or equal to the live metadata bytes. State in Result how that peak is
  measured or bounded.
- **Mutations,** each run and restored:
  - Drop one of the new charges from the request.
  - Re-clone the page view.

  Each must fail an existing test, or the Result must explain why no test
  can observe it.
- **Result:** one test per invariant, plus a short **Review map** listing each
  changed file with its lines added and removed, its purpose and the clause
  it serves.
- **Stop condition:** a lease, ledger or `SequenceState` semantic change.

## Result, filled after work

- Changed owners; source commit:
- Commands, GPU UUIDs; passed / failed / skipped:
- Mutation results and restoration:
- Review map:
- Remaining obligations:
