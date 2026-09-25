# Task 0089 — measure pinned transfers and copy/compute overlap on this machine

Status: **proposed** (coordinator, 2026-09-25); opens when task 0088 is
accepted. Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0089, M6 slice 5, roadmap **M6.3** "measured transfer overlap, bounded
  read-ahead … compare no-prefetch/no-overlap baselines". Document 03:
  "Benchmark direct mapped/pageable transfers, persistent pinned regions, and
  bounded pinned bounce buffers on this machine. Pinning is an option, not a
  universal speed claim." Source map R12: measure actual links; pinning is not
  automatically faster. Strata allocates pinned memory in many back ends but
  records no bound or bandwidth for it, so there is nothing to adopt.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code or the installed CUDA header, stop and send
  `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.
- O6 is open: the figures are measurements for later decisions, not a speed
  claim.

## Facts established before writing (coordinator, 2026-09-25)

- `moxie-cuda` has no pinned allocation (`cuMemHostAlloc` is not declared in
  `ffi.rs`).
- `moxie-executor/src/topology_probe.rs` (M5.5) measures pageable host links,
  granted peer links, simultaneous traffic and device memory, with its own
  measurement-only `DeviceBuffer::alloc` buffers outside the ledger. Its doc
  says: "`ponytail:` pinned and mapped copies stay unmeasured until a pinned
  path exists (M6.3)". `xtask probe` runs it; results are in
  `docs/evidence/topology-costs.md`.
- `TopologyCosts`/`LinkCost` (`moxie-plan/src/costs.rs`) are constructed by
  other code (plan comparison and its tests). **This task does not change
  them.**

## Bounded deliverable

- **Outcome:** measured, per GPU and direction, pinned bulk bandwidth next to
  the existing pageable figure, how long the host is blocked issuing an async
  copy from pageable and from pinned memory, and whether a pinned
  host-to-device copy overlaps a compute kernel on another stream; recorded
  in the topology evidence.
- **Allowed files:** `crates/moxie-cuda/src/{ffi.rs,status.rs,driver.rs,lib.rs}`,
  `crates/moxie-executor/src/topology_probe.rs` (and its `lib.rs` export),
  `xtask/src/probe.rs`, `docs/evidence/topology-costs.md` (append a section),
  this task's Result.
- **Non-goals:** any execution path using pinned memory; ledger admission of
  pinned memory (next task, with this evidence); mapped memory; changing
  `TopologyCosts`.

## Numbered changes

1. **`moxie-cuda` pinned buffer.** Declare `cuMemHostAlloc(pp: *mut *mut
   c_void, bytesize: usize, flags: c_uint)` and `cuMemFreeHost(p: *mut
   c_void)` (verify against `cuda.h`), classify them in `status.rs`, and add
   `PinnedHostBuffer<'ctx>` (pointer, length, `&'ctx RankContext`) with
   `alloc(ctx, bytes)` (flags 0; zero-length refused), `as_slice`,
   `as_mut_slice`, and `Drop` that makes the context current and frees. Its
   doc states that dropping it while an async copy may read it is the owner's
   error, as for any host source.
2. **`probe_pinned`** in `topology_probe.rs`: `pub fn probe_pinned(ordinals:
   &[u32], config: ProbeConfig) -> Result<Vec<PinnedLink>>` with `pub struct
   PinnedLink { device: DeviceUuid, direction: Direction (host-to-device or
   device-to-host), pageable_gbps: f64, pinned_gbps: f64,
   pageable_issue_us: f64, pinned_issue_us: f64, overlap: f64 }` (reuse the
   file's `Direction` if it is public enough; otherwise make it `pub`). For
   each GPU, isolated (one GPU at a time), `config.reps` repetitions, median:
   - `pageable_gbps` / `pinned_gbps`: bulk copy timed with events, as the
     existing isolated measurement does;
   - `*_issue_us`: host wall time spent **inside** the async copy call
     (from before the call to its return), bulk size;
   - `overlap` (host-to-device only; `0.0` for device-to-host): on stream A
     a pinned bulk host-to-device copy, on stream B a compute kernel sized to
     take roughly as long as that copy alone (the smoke axpy looped over a
     device buffer; calibrate the loop count from one timed run). Measure
     `copy_alone`, `kernel_alone` and `both` wall times with events;
     `overlap = (copy_alone + kernel_alone - both) / min(copy_alone,
     kernel_alone)`, clamped to `[0, 1]` (1 = fully hidden, 0 = serialized).
   Remove the `ponytail:` note from `probe_topology`'s doc (pinned is now
   measured; mapped stays unmeasured, say so in one line).
3. **`xtask probe`** prints the pinned table after the existing output (same
   style).
4. **Evidence.** Run `xtask probe` twice; append "Pinned host transfers" to
   `docs/evidence/topology-costs.md`: command, GPU UUIDs, driver, both runs'
   tables, and one paragraph stating only what the numbers show (which
   direction/GPU gains, whether the issue call blocks, whether overlap holds).

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): `cargo xtask-cuda test-gpu`
(69/69, unchanged); the two probe runs.

**Reproducibility:** each figure in run 2 within 25% of run 1, or the
evidence says which did not and by how much.

**Stop conditions:** a header mismatch; a change conflicts with the code;
`probe_pinned` cannot be implemented without changing `TopologyCosts`; a
file outside the allowed list is needed.

## Result, filled after work

(pending)
