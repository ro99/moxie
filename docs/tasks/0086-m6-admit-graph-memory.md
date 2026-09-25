# Task 0086 — captured graphs are admitted in `GraphPools`

Status: **accepted** (coordinator, 2026-09-25), sol's review round R1 clean.
Implementation `278c27e`.
- Measured bound ([graph-memory.md](../evidence/graph-memory.md)): up to 7 KiB
  per captured kernel node (3090) and 104 KiB per graph (5060 Ti), exact
  repeats; charged as 8 KiB per kernel plus 128 KiB per segment in
  `GraphPools`. The 0-byte reading in task 0085 was below `cuMemGetInfo`'s
  granularity at fixture scale. Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0086, M6 slice 4, roadmap **M6.2** "all graph pools and workspaces
  admitted". AGENTS.md: "One real memory authority admits all persistent/
  transient/branch/draft resources." Task 0085's capture is opt-in precisely
  because its graph memory is not charged.
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

- `moxie-types` `DeviceTier::GraphPools` ("memory pools pinned by a captured
  graph") exists and is unused; document 03 lists graph pools as tracked
  device memory.
- `SelectedReservedPlan::set_segment_capture(&mut self, enabled: bool)`
  (`chain.rs` about 325) has three callers, all in `dense_gemma_device.rs`
  (about 385, 449, 2784).
- The plan's arena consumes its one ledger `Reservation`
  (`DeviceArena::create_partitioned`, `chain.rs` about 233), so graph memory
  needs a separate reservation: `ledger.admit(&PlanRequest)` with one
  `BufferRequest::new(label, Scope::Device(uuid), Tier::Device(GraphPools),
  bytes, span)`, released with `ledger.release(reservation)`.
- `RankContext::memory_info()` (`moxie-cuda/src/driver.rs` about 472) returns
  `(free, total)` through the live context.
- Task 0085 measured no `cuMemGetInfo` change across the fixture's capture
  step, which bounds nothing: a real plan captures hundreds of kernel nodes.

## Bounded deliverable

- **Outcome:** enabling capture on a plan admits a `GraphPools` reservation
  sized by a measured, enforced bound from the plan's captured kernel and
  segment counts; disabling capture or closing the plan releases it; a GPU
  case enforces the bound on all three GPUs.
- **Allowed files:** `crates/moxie-executor/src/chain.rs`,
  `crates/moxie-executor/src/dense.rs` (only if the counting helper belongs
  there), `crates/moxie-executor/tests/dense_gemma_device.rs`,
  `xtask/src/gpu.rs`, new `docs/evidence/graph-memory.md`, this task's Result.
- **Non-goals:** making capture the default (that is decided with the exit
  benchmarks); full-step capture; any change to how graphs are captured.

## Numbered changes

1. **Measurement case** `graph_memory_within_bound` in `xtask/src/gpu.rs`,
   beside `graph_capture_replay`, on every GPU: (a) synchronize, read
   `memory_info`; capture **one** graph of 16,384 axpy launches, instantiate,
   synchronize, read again → `node_delta`; (b) the same for **256** graphs of
   one axpy launch each → `graph_delta`. Keep all graphs alive until both
   readings are taken. Assert `node_delta ≤ 16_384 × B + F` and `graph_delta ≤
   256 × (B + F)`, and print both deltas.
2. **The bound, chosen by rule, not by hand.** Run the case once with `B` and
   `F` set to `u64::MAX / 2^20` (the assertions cannot fail) to obtain the
   deltas on all three GPUs. Then set, in `chain.rs`:
   `CAPTURED_KERNEL_BOUND_BYTES = max(256, next_power_of_two(max over GPUs of
   ceil(node_delta / 16_384)))` and `CAPTURED_GRAPH_BOUND_BYTES = max(4096,
   next_power_of_two(max over GPUs of ceil(graph_delta / 256)))`, and use
   them as `B` and `F` in the case. Record every raw delta, GPU UUID and the
   derivation in `docs/evidence/graph-memory.md`; the constants' doc comments
   cite that file.
3. **Counts.** A private function on the candidate's selected nodes returning
   `(kernels, segments)`: a node is **eager** if its descriptor operation is
   `PagedAttention` or `CombineHostJoin`; `kernels` = sum of
   `descriptor.symbols.len()` over non-eager nodes; `segments` = number of
   maximal runs of consecutive non-eager nodes. It must equal what task
   0085's `close_segment` produces; the dense test asserts
   `captured.len() == segments` after a capture step (expose a
   `pub fn captured_segments(&self) -> usize`).
4. **Admission.** `set_segment_capture(&mut self, enabled: bool, ledger:
   &mut Ledger) -> Result<()>`. Enabling (from disabled): after the existing
   eligibility refusal, admit `kernels × B + segments × F` (checked) as one
   `GraphPools` buffer for the plan's device, store the `Reservation` in a
   new field `graph_reservation: Option<Reservation>`; a ledger rejection is
   a typed refusal and capture stays disabled. Disabling: clear `captured`,
   then release the reservation (a refused release keeps it and returns the
   error). The ledger must be the plan's own (`self.ledger`). Expose `pub fn
   graph_pool_bytes(&self) -> u64` (0 when disabled).
5. **Close.** `close` clears `captured` and releases `graph_reservation`
   before the arena, restoring it on a refused release as it does for ranges.
6. **Test.** Update the three callers. In `run_prefill_decode`'s capture
   step, read `memory_info` before and after and assert the drop is at most
   `plan.graph_pool_bytes()`; assert `captured_segments()` equals the count
   from change 3. `ledger.outstanding().is_empty()` at the end already covers
   release.

## Contract before implementation

- **Resources:** every graph a plan captures is charged before capture; the
  charge is an upper bound enforced by measurement on each GPU class.
- **Lifetimes:** the reservation lives exactly as long as capture is enabled
  on the plan, and is released only after the graphs are cleared.
- **Failure:** a rejected admission leaves capture disabled and nothing
  charged.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`,
`dense_tp2_device`, `cargo xtask-cuda test-gpu` (69/69: the new case on each
GPU).

**Coverage check (one mutant, reverted after):** in the measurement case, use
`B = 0` (drop the per-kernel term); it must fail on at least one GPU.
If it passes everywhere, stop and report the deltas (the bound would then be
untested by its own case).

**Stop conditions:** a change conflicts with the code; a measured delta is
not reproducible within a factor of two between two runs (report both); a
file outside the allowed list is needed.

## Result, filled after work

Implemented `graph_memory_within_bound` and admitted each plan's captured
kernel nodes and segments through one `GraphPools` reservation. Disabling
capture and closing the plan clear graphs before releasing the reservation;
the dense device fixture checks both its measured memory drop and captured
segment count.

Two unconstrained all-GPU runs reproduced every node and graph delta exactly.
The stated rule derives `CAPTURED_KERNEL_BOUND_BYTES = 8,192` and
`CAPTURED_GRAPH_BOUND_BYTES = 131,072`; raw free-memory readings and the
derivation are in [`graph-memory.md`](../evidence/graph-memory.md). The B=0
mutant was caught on all three GPUs by node deltas of 41,943,040 bytes on the
5060 Ti and 117,440,512 bytes on each 3090, against a 131,072-byte allowance;
the mutant was reverted.

Host gates passed: formatting, workspace clippy, executor driver-feature
clippy, workspace tests, architecture check and spec check. With
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, GPU gates passed: `dense_gemma_device` (8
passed, 1 ignored), `dense_tp2_device` (1 passed), and `cargo xtask-cuda
test-gpu` (69/69; all three devices passed, SM86 and SM120 qualified).
