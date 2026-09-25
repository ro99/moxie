# Task 0079 — load the dense kernel module once per admitted plan

Status: **active** (coordinator, 2026-09-24). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0079, M6 slice 1 (sync-free single-GPU dense step), roadmap **M6.1**.
  Chosen by [task 0078](0078-m6-dense-step-timing.md)'s profile: module load
  and unload are 24% of per-step CUDA driver API time (117
  `cuModuleLoadData` for 111 steps).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. On a conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels or strings.
- O6 is open: no speed claim.

## Facts established before writing (coordinator, 2026-09-24)

- `dense.rs` `execute_dense_stage` (about 163–185) builds the symbol list from
  `self.candidate().nodes()`, then `Module::load(ctx,
  ModuleImage::Binary(DENSE_GRAPH_FATBIN)).and_then(|m| m.resolve_all(&symbols))`
  **on every step**, and stores the `ResolvedModule` in
  `DenseOperation::package`. The operation drops it (unloading the module)
  when the lease retires.
- The only launch use is `dense.rs` about 1300–1310
  (`operation.package.launch_async(symbol_index, …)`).
- `SelectedReservedPlan` (`chain.rs` about 122) lives across steps: `finish`
  returns it, and it is dropped only by `close` (about 490) or withheld by a
  refused close or a lost lease.
- The symbol list depends only on the candidate, which the plan owns and
  `execute_dense_stage` already checks (`candidate().matches(…)`, the
  catalogue digest, `ctx.uuid() == capability.uuid`).
- Task 0075's lesson: a module must never be unloaded while a kernel from it
  may run. Keeping it inside the plan gives it exactly the plan's lifetime,
  and a lost lease withholds the plan.
- `ChainOperation` (`chain.rs` about 131, the M1 BF16 chain) also loads per
  step. **It is out of scope.**

## Bounded deliverable

- **Outcome:** the dense module is loaded and resolved on the first dense
  step of an admitted plan, then reused by every later step of that plan, and
  unloaded when the plan is dropped.
- **Allowed files:** `crates/moxie-executor/src/chain.rs`,
  `crates/moxie-executor/src/dense.rs`, this task's Result, and
  `docs/evidence/dense-step-timing.md` (append a section only).
- **Non-goals:** `ChainOperation`; any other module user; the per-operation
  `settle` (deferred in the ledger); launch-count reduction (M6.2).

## Numbered changes

1. **`chain.rs`.** Add `package: Option<ResolvedModule<'ctx>>` to
   `SelectedReservedPlan`, with a doc comment: "The dense kernel module,
   loaded by the first dense step and dropped with the plan, so it is never
   unloaded while a step's kernels may still run." Initialise `None` wherever
   the struct is constructed. `close` needs no change (the module drops with
   the plan after its arena closes).
2. **`dense.rs`, load once.** In `execute_dense_stage`, replace the
   per-step load: if `self.package` is `None`, build the symbol list and
   load/resolve exactly as now (same `SAFETY` comment, same `reject(self,
   bindings, error)` on failure), then store it in `self.package`. If it is
   `Some`, skip the load.
3. **`dense.rs`, use it.** Delete `DenseOperation::package`. The launch site
   takes the module from the operation's plan (`operation.plan.as_ref()
   .expect("dense operation retains plan").package.as_ref()`), returning the
   file's `invalid(…)` if it is `None`. Keep the attribution unchanged.
4. In `dense_gemma_device.rs`'s existing `run_prefill_decode` execute closure,
   after the first `finish()` and existing `launch_order` check, abort that
   transaction and begin a new one. Re-execute the same plan
   (`result.plan`) with `result.returned_inputs` and the same `host_experts`,
   finish it, then assert the first and second outputs are bit-identical with
   the UUID in the failure message. Commit the new transaction and close the
   second result's plan as before. This is the existing test helper; no other
   test or production change.

## Contract before implementation

- **Semantics:** unchanged; outputs **bit-identical**.
- **Resources:** one module per live dense plan instead of one per step.
  Module memory is not charged to the ledger today; this task does not change
  that.
- **Lifetimes:** the module lives exactly as long as the plan, including a
  plan withheld by a lost lease or a refused close.
- **Failure:** a failed first load refuses the step before any launch and
  returns the plan with `package` still `None`; the next step retries.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy
(`driver,paged-attention-binding,paged-attention-test-hooks`); `cargo test
--workspace --locked`; `cargo xtask arch-check`; `cargo xtask spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), full suites:
`dense_gemma_device`, `dense_tp2_device`, `cargo xtask-cuda test-gpu` (63/63,
SM86 and SM120).

**Evidence:** append a "Module per plan" section to
`docs/evidence/dense-step-timing.md`: three unprofiled `dense_step_timing`
runs at the candidate (same command as 0078), and one `nsys` run giving the
`cuModuleLoadData` and `cuModuleUnload` call counts (expected: a handful, not
about one per step). Compare with the 0078 candidate rows; no speed claim.

No mutant: a stale module across plans is impossible by construction (the
field belongs to the plan), and the `nsys` counts prove the cache is used.

**Stop conditions:** outputs not bit-identical; a change conflicts with the
code; a file outside the allowed list is needed.

## Result, filled after work

Implemented the three requested changes: `SelectedReservedPlan` now owns the
optional dense module; the first dense step loads and resolves it; and launch
uses the module through the plan retained by the operation lease. Initial
load/resolve failure returns the plan with the field still empty. No other
module user changed.

Host gates passed: fmt, workspace and driver-feature clippy, workspace tests,
`arch-check` (79 rejected, 21 accepted) and `spec-check` (10 documents). GPU
gates passed: dense Gemma (5 passed, timing test ignored), dense TP2 (1 passed)
and `cargo xtask-cuda test-gpu` (63/63, SM86 and SM120). Three candidate timing
runs and one Nsight profile completed; the profile recorded 8
`cuModuleLoadData` and 8 `cuModuleUnload` calls, compared with 117 each in task
0078. The samples and profiler artifact hashes are appended in [the timing
evidence](../evidence/dense-step-timing.md#module-per-plan). They remain
fixture-scale, O6-open evidence, not a performance claim.

R1: Added the missing output-checked second execution of the same admitted plan after abort; this exercises the cached module and compares the result bit-for-bit. Only the existing `run_prefill_decode` helper changed in the GPU test file; no production code or other test changed. Round-2 gates passed: fmt, workspace and driver-feature clippy, workspace tests, dense Gemma (5 passed, timing test ignored), and dense TP2 (1 passed).

The checkout started at requested base `855e0cc`. During the assignment,
`main` advanced through coordinator commit `102e1c8`, a handover-only M6 ledger
update; it was preserved. The code commit `3aea3a4` therefore has parent
`102e1c8`; that commit did not change the code baseline.
