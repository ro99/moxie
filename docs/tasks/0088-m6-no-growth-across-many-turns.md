# Task 0088 — no resource growth across many turns

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0088, M6 slice 6, roadmap **M6.4**: "Validate no host-cache or
  prepared-layout growth across many turns."
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

- Everything a plan keeps across steps is now owned by it: module (0079),
  RoPE tables per step (0077/0083, released with the lease), captured graphs
  and their `GraphPools` reservation (0085/0086). Paged KV is admitted up
  front for `max_tokens` by `DeviceKvSequence` and the runs.
- Task 0087 adds a reused bucket-plan set and `prefill_chunks`.
- `Ledger::committed(scope, tier)`, `scope_committed(scope)` and
  `outstanding()` report charges; `RankContext::memory_info()` reports device
  free bytes; `/proc/self/statm` gives the process's resident pages.

## Bounded deliverable

- **Outcome:** one GPU test drives 24 turns of one conversation through a
  fixed plan set and proves that, after the first two turns, nothing charged,
  nothing uncharged on the device, and nothing material on the host grows.
- **Allowed files:** `crates/moxie-executor/tests/dense_gemma_device.rs`,
  this task's Result. Production code only if a growth is found (then stop
  first: see below).
- **Non-goals:** fixing any growth found (that is its own task); timing.

## The test

`many_turns_hold_every_resource_steady`, on one 3090
(`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`), Shape A, the task 0087 bucket
set `[1, 2, 4, 8]` with segment capture enabled, and `DeviceKvSequence`
geometry with `max_tokens` large enough for 24 turns of 5 prompt tokens and 2
decode tokens (at least 168; round up to a page multiple). One sequence for
all turns. Each turn: prefill 5 new tokens at the next absolute positions
through `prefill_chunks(5, buckets)` (`[4, 1]`), commit each chunk; then 2
decode steps with the `1` plan, each committed. After **every** turn record:

1. `ledger.scope_committed` for the device scope and for `Scope::Host`, and
   `ledger.outstanding_count()`;
2. device free bytes from `memory_info()`;
3. resident bytes from `/proc/self/statm` (second field × page size).

Assert that for every turn from the third on: (1) equals turn 2's values
exactly; (2) equals turn 2's exactly; (3) is at most turn 2's plus 4 MiB (the
declared allocator-slack allowance; print every turn's value). Close
everything at the end and assert the ledger is empty. Print one summary line
per turn.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`.

**Coverage check (one mutant, reverted after):** in the test, admit one
extra 4 KiB host buffer through the ledger on turn 10 and keep it; the test
must fail on check (1).

**Stop conditions:** any check fails on the unmutated code: stop and send
`DECISION` with every turn's recorded values and which quantity grew. Do not
fix production code in this task.

## Result, filled after work

(pending)
