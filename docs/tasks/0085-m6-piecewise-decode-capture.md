# Task 0085 — a reused dense plan captures and replays its segments

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0085, M6 slice 4, roadmap **M6.2** "stable decode … graph capture with
  piecewise fallback". Piecewise capture as in the pinned vLLM design (source
  map): paged attention stays eager; the kernel runs between attention nodes
  are captured once per plan and replayed.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code, stop and send `DECISION`; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.
- O6 is open: no speed claim.

## Facts established before writing (coordinator, 2026-09-25)

- After tasks 0079–0084, a reused `SelectedReservedPlan` keeps one module
  (`package`), one arena whose ranges never move, bound weights, and fixed
  RoPE-table offsets. Every kernel of `enqueue_dense` (`dense.rs` about 350)
  goes through the one helper `launch(lease, symbol_index, stream, grid,
  block, params, selected, symbol)`, whose arguments are plan addresses and
  plan-fixed scalars (`rows` is the plan's). So a run of kernels between
  eager work is identical on every step of the same plan.
- Eager work in a step: `upload_sources` and `upload_rope_tables` (before the
  loop), the paged-attention arm (`execute_attention`: device append, page
  table publish, attend, each on the run's own module), and the host-expert
  join (`enqueue_host_join`, which synchronizes the stream).
- Host-side checks inside arms (for example the Embedding token-range check)
  read host sources and launch nothing.
- `moxie-cuda` (task 0084): `Stream::begin_capture`, `Stream::end_capture ->
  CapturedGraph<'ctx>`, unsafe `CapturedGraph::launch(&self, stream)`. A
  captured graph executes nothing until launched; its exec must not be
  dropped while a launch may run, and it names the module's functions, so it
  must drop before the module.

## Bounded deliverable

- **Outcome:** a single-GPU dense plan with capture enabled runs its first
  step by capturing each kernel run between eager work into a graph and
  launching it, and runs every later step by launching those graphs instead
  of the kernels; outputs are bit-identical to eager execution. Capture is
  **off by default** until graph memory is admitted (next task).
- **Allowed files:** `crates/moxie-executor/src/{chain.rs,dense.rs}`,
  `crates/moxie-executor/tests/dense_gemma_device.rs`, this task's Result,
  `docs/evidence/dense-step-timing.md` (append a section).
- **Non-goals:** full-step capture including attention; plans with host
  expert joins, linear orders, combine orders or expert ownership (TP, PP
  stages); admitting graph memory; making capture the default.

## Numbered changes

1. **Plan state (`chain.rs`).** Add to `SelectedReservedPlan`, **declared
   before `package`** so graphs drop before the module:
   `captured: Vec<CapturedGraph<'ctx>>` and `capture_enabled: bool` (empty /
   false at admission). Add `pub fn set_segment_capture(&mut self, enabled:
   bool) -> Result<()>`: refuse (typed `invalid("capture", …)`) unless the
   candidate is dense and has no host expert joins, linear orders, combine
   orders or expert ownership; on any call, clear `captured`.
2. **Step mode (`dense.rs`).** At the start of `enqueue_dense`, pick the mode:
   `Eager` if not enabled; `Capture` if enabled and `captured` is empty;
   `Replay` otherwise. Keep it in `DenseOperation` with `segment: usize` and
   `open: bool`.
3. **`launch`.** `Eager`: unchanged. `Capture`: if not `open`,
   `stream.begin_capture()?` and set `open`; then launch as now. `Replay`: if
   not `open`, launch `captured[segment]` (refuse if absent) and set `open`;
   do **not** launch the kernel. The symbol-identity check (task 0080 R1)
   runs in every mode.
4. **`close_segment(lease, stream)`.** If `open`: in `Capture`, `end_capture`,
   push the graph into the plan's `captured` (its index must equal
   `segment`), then launch it; in `Replay`, nothing. Then `segment += 1`,
   `open = false`. Call it at the top of the attention arm, before the
   host-expert join (unreachable while capture is enabled, but closes the
   segment anyway), and after the node loop.
5. **Replay completeness.** After the loop, in `Replay`, refuse unless
   `segment == captured.len()`.
6. **Failure.** If `enqueue_dense` fails while a capture is open, end the
   capture and discard its result before returning the error, and clear the
   plan's `captured` whenever a `Capture`-mode step fails, so a later step
   recaptures from scratch. The lease handling of the error is unchanged.
7. **Test.** In `run_prefill_decode`'s execute closure, **when the plan's
   candidate has empty linear orders, combine orders, expert ownership and
   host expert joins** (amended by the coordinator, 2026-09-25, on the
   builder's `DECISION`: a routed reference candidate has an empty
   `host_experts` slice but carries ordering metadata; for every other
   candidate, assert `set_segment_capture(true)` returns the typed `capture`
   refusal): after the existing replay step (task 0079 R1) and its assertion,
   abort that transaction, `set_segment_capture(true)`, run the step again
   (captures), assert its output equals the first; abort, run again
   (replays), assert equal; commit that transaction. Keep the existing
   commit/close flow otherwise.
8. **Timing.** In `dense_step_timing`, after the decode loop, enable capture
   on the decode plan, run one untimed capture step (abort it), then run the
   same `W + R` loop and print a third line with `phase=decode-captured`.

## Contract before implementation

- **Semantics:** the same kernels with the same arguments, in the same stream
  order; outputs **bit-identical** to eager.
- **Resources:** graph execs hold driver memory the ledger does not charge.
  That is why capture stays opt-in; the evidence records the size.
- **Lifetimes:** graphs live in the plan, drop before its module, and are
  launched only on the plan's own step stream while the plan is held by the
  step's lease.
- **Failure:** a capture-mode failure leaves no half-captured graph behind.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`
(capture and replay on all three GPUs), `dense_tp2_device`, `cargo xtask-cuda
test-gpu` (66/66).

**Coverage check (two mutants, each reverted):** (a) in `Replay`, launch the
kernel as well as the graph (double application must fail the output
comparison); (b) in `Replay`, skip launching the graph. The test must fail
each.

**Evidence:** append a "Piecewise capture" section to
`docs/evidence/dense-step-timing.md`: three timing runs (including the
`decode-captured` line); one `nsys` run with `cuLaunchKernel` and
`cuGraphLaunch` counts; the device free-memory difference (`cuMemGetInfo`, or
the existing ledger measurement helper) before and after the capture step, as
the graph-memory figure the admission task needs.

**Stop conditions:** outputs not bit-identical; a change conflicts with the
code; any kernel inside a segment turns out to take a per-step scalar that is
not plan-fixed (report which); a file outside the allowed list is needed.

## Result, filled after work

R1: Sync failure retains graphs with the lost lease and returns the original error.

Implemented the opt-in segment-capture state and eager/capture/replay modes
in the selected dense plan. Captured graphs live before the module in plan
field order, launch only through the plan's stream, and are cleared after a
failed capture step. Candidate metadata gates capture; candidates with any
linear order, combine order, expert ownership or host join return a typed
`capture` refusal. The device test compares eager, captured and replayed
outputs on eligible candidates, and checks that refusal on all others.

The two requested replay mutants were each caught by the output check and
reverted. Host gates passed: formatting, workspace clippy, executor
driver-feature clippy, workspace tests, architecture check and spec check.
GPU gates passed with `CUDA_DEVICE_ORDER=PCI_BUS_ID`: `dense_gemma_device`
(8 passed, 1 ignored), `dense_tp2_device` (1 passed), and
`cargo xtask-cuda test-gpu` (66/66 on SM86 and SM120). Outputs remained
bit-identical.

The three RTX 3090 timing runs, Nsight driver-call counts and reported
capture-step free-memory deltas are in
[`dense-step-timing.md`](../evidence/dense-step-timing.md#piecewise-capture).
The measured free-memory delta was 0 bytes in all four samples. These
fixture-scale timings do not close O6 or make a performance claim.
