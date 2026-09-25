# Task 0084 — `moxie-cuda` stream capture, instantiate and launch

Status: **accepted** (coordinator, 2026-09-25), after sol's review round R1.
Implementation `bcbab23`; R1 repair `b65e677`.
- R1 (MEDIUM, coverage): the invalidation check captured nothing and ignored
  `synchronize`'s result, so an empty graph could explain the failure. The
  case now captures a launch, asserts `synchronize` fails, `end_capture`
  fails, and ordinary work then runs correctly. Verified by the coordinator. Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0084, M6 slice 4, roadmap **M6.2** "stable decode … graph capture".
  `moxie-cuda` has no graph API; task 0085 (a reused decode plan captures its
  segments between attention nodes) needs one. Strata never used CUDA graphs;
  the pinned vLLM graph-mode design (source map, document 08) is the reference
  for piecewise capture, not for this binding.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code or the installed CUDA headers, stop and send
  `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.

## Facts established before writing (coordinator, 2026-09-25)

- `moxie-cuda/src/ffi.rs` declares the driver API by hand (for example
  `cuStreamSynchronize`, `cuEventRecord`, `cuModuleLoadData` about 111–124);
  `status.rs` classifies each API's errors by name; `driver.rs` wraps them
  (`Stream` about 695 with `new`, `synchronize`, `wait_event`; `Module` with a
  `Drop` that makes its context current and unloads).
- `/usr/local/cuda/include/cuda.h` maps `cuStreamBeginCapture` to
  `cuStreamBeginCapture_v2` and exposes `cuGraphLaunch`.
- `xtask/src/gpu.rs` `stream_event` (about 545) is the per-GPU smoke case
  pattern: acquire a context, load the smoke module, run, check, return an
  `Outcome`; `test-gpu` currently runs 21 cases on each of three GPUs (63).

## Bounded deliverable

- **Outcome:** a stream can capture its enqueued work into an instantiated,
  owned graph that later launches on a stream of the same device; one GPU
  smoke case proves replay equals eager execution on all three GPUs.
- **Allowed files:** `crates/moxie-cuda/src/{ffi.rs,status.rs,driver.rs,lib.rs}`,
  `xtask/src/gpu.rs`, this task's Result.
- **Non-goals:** any executor use (task 0085); graph update APIs; memory
  nodes; conditional nodes; capture of anything but kernels and copies.

## Numbered changes

1. **FFI.** Opaque `CUgraph` and `CUgraphExec` pointer types, and:
   `cuStreamBeginCapture_v2(stream, mode: c_uint)`,
   `cuStreamEndCapture(stream, *mut CUgraph)`,
   `cuGraphInstantiateWithFlags(*mut CUgraphExec, CUgraph, flags: u64)`,
   `cuGraphLaunch(CUgraphExec, CUstream)`, `cuGraphExecDestroy(CUgraphExec)`,
   `cuGraphDestroy(CUgraph)`. Capture mode constant: thread-local (`1`), so a
   rank thread's capture is not invalidated by another thread's API calls.
   Verify each signature against the installed `cuda.h`; any mismatch is a
   `DECISION`.
2. **Status.** Classify the new APIs' errors as the file does for its
   neighbours.
3. **`CapturedGraph<'ctx>`** in `driver.rs`: holds `CUgraphExec` and `&'ctx
   RankContext`. `Drop` makes the context current and calls
   `cuGraphExecDestroy`, ignoring the result, like `Module`. Doc: "The owner
   must not drop a graph while a launch of it may still run; whoever holds
   the kernels' buffers holds the graph with them."
4. **`Stream::begin_capture(&self) -> Result<()>`** (thread-local mode) and
   **`Stream::end_capture(&self) -> Result<CapturedGraph<'ctx>>`**:
   `cuStreamEndCapture` → graph; `cuGraphInstantiateWithFlags(…, 0)` → exec;
   `cuGraphDestroy(graph)` whatever instantiation returned; return the
   wrapper. If `cuStreamEndCapture` fails (for example, capture invalidated
   by an illegal call), destroy any returned graph and return the error: the
   stream is no longer capturing either way.
5. **`CapturedGraph::launch(&self, stream: &Stream<'ctx>) -> Result<()>`**,
   `unsafe`: refuse a stream of another device; `SAFETY` contract: every
   buffer a captured kernel or copy names is live and not concurrently
   written until a completion recorded after this launch is observed.
6. **Smoke case** `graph_capture_replay` in `xtask/src/gpu.rs`, beside
   `stream_event`, registered the same way: allocate `x` and `y`, upload;
   `begin_capture`; enqueue two axpy launches (`y = 2x + y`, twice) with
   `launch_async`; `end_capture`; launch the graph **twice**, synchronize, and
   check `y` equals the host computation of four axpy applications. Also check
   that `end_capture` on a stream whose capture enqueued a
   `stream.synchronize()` returns an error and that the stream then accepts
   ordinary work (the invalidation path of change 4).

## Contract before implementation

- **Semantics:** captured work executes exactly the enqueued launches and
  copies, in order, each time the graph launches.
- **Resources:** a graph exec holds driver memory outside the ledger; task
  0085 decides its admission. This task only binds the API.
- **Lifetimes:** as change 3 and 5.
- **Failure:** a failed capture ends capture and returns a typed error.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gate:** `cargo xtask-cuda test-gpu` passes 66/66 (the new case on each
of three GPUs, SM86 and SM120).

**Coverage check (one mutant, reverted after):** launch the graph once
instead of twice; the case must fail.

**Stop conditions:** a signature differs from the installed header; a change
conflicts with the code; a file outside the allowed list is needed.

## Result, filled after work

Verified the new FFI declarations against `/usr/local/cuda/include/cuda.h`.
The header maps `cuGraphInstantiate` to `cuGraphInstantiateWithFlags`; its
`unsigned long long` flags parameter matches `u64`. The capture-mode enum is
represented by the requested `c_uint`, with thread-local mode set to 1.

Added the graph and executable-graph handles, capture/instantiate/launch/
destroy FFI, and typed classification for CUDA stream-capture errors. Added
`CapturedGraph<'ctx>` with context-bound cleanup, thread-local
`Stream::begin_capture`, `Stream::end_capture` cleanup on failed capture and
instantiation, and unsafe same-device graph launch with the required buffer
lifetime contract. Exported the wrapper.

Added the `graph_capture_replay` GPU case beside `stream_event`: it captures
two async AXPY launches, replays them twice, verifies the result against four
host AXPYs, then enqueues an AXPY during a second capture and verifies that
`stream.synchronize()` and `end_capture()` both return errors. It then verifies
an ordinary AXPY on the recovered stream against the host result.

The launch-once mutant failed only `graph_capture_replay` on all three GPUs:
63 passed and 3 failed, each with output 5 instead of 9. The mutant was
reverted. Host gates passed: `cargo fmt --all -- --check`, workspace clippy,
driver-feature executor clippy, `cargo test --workspace --locked`,
`cargo xtask arch-check`, and `cargo xtask spec-check`. With
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `cargo xtask-cuda test-gpu` passed 66/66 across
the three GPUs and qualified SM86 and SM120.

R1: Added a captured AXPY node before synchronizing. On all three GPUs,
`stream.synchronize()` and `end_capture()` returned errors, and the subsequent
ordinary AXPY matched the host result. Round-2 gates passed: fmt, workspace
clippy, `cargo test --workspace --locked`, and `cargo xtask-cuda test-gpu`
(66/66).
