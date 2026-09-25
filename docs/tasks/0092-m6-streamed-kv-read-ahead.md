# Task 0092 — bounded read-ahead for host-backed KV streaming

Status: **active** (coordinator, 2026-09-25). Revised after sol's design
review (2 high, 3 medium, all adopted; see "Design review" below) before any
implementation (engineering log, 2026-09-25). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0092, M6 slice 5, roadmap **M6.3** "measured transfer overlap, bounded
  read-ahead and preparation … Compare no-prefetch/no-overlap baselines and
  record wasted work." M4.3 (tasks 0041–0043) deliberately staged one host
  page at a time ("the caller can fold the result and drop the page before
  obtaining the next page"); task 0042's review caught a read-ahead violation
  in its first API. Read-ahead is therefore a new, explicit API, not a
  relaxation of `stage_next`.
- Task 0089 measured on this machine: a pageable async copy blocks the host
  for the whole transfer; a pinned one returns in about 2.5 µs and overlaps
  compute (0.98–1.0). Overlap therefore needs pinned staging.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. On any conflict with the code,
  stop and send `DECISION`; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.
- O6 is open: the timing is evidence, not a speed claim.

## Facts established before writing (coordinator, 2026-09-25)

- `PagedAttentionRun::start_n_block` (`paged_attention.rs` about 3890)
  settles the resident partial and returns an `NBlockStream` (about 1258:
  `run`, `stream`, `launch`, `next_base`, `remaining_rows`,
  `remaining_blocks`, `host_to_device_bytes`).
- `NBlockStream::stage_next(keys: Vec<u8>, values: Vec<u8>)` (about 4286):
  validates the page, holds the `Vec`s in `run.held` as
  `RefusedSource::Stream`, `stage_stream` copies them pageable→device into the
  run's **one** staging page (`staged_keys`, `staged_values`, `staged_table`),
  `launch_partial`, then `read_partials` synchronizes and reads the FP32
  partials back; the caller merges with `moxie_oracles::online_softmax`.
- `Staging::HostBacked { max_staged_blocks }` admits one staging page and one
  partial output; `max_staged_blocks ≤ HostBackedPlan::MAX_STAGED_BLOCKS`
  (3, `moxie-memory/src/report.rs` about 84).
- `moxie-cuda` has `PinnedHostBuffer` (task 0089) and `Stream::wait_event`;
  the ledger supports per-tier caps (`CapacitySnapshot::with_tier_cap`).

## Bounded deliverable

- **Outcome:** a host-backed N-block stream can hold **one** block of
  read-ahead: while block `i`'s partial kernel runs, block `i + 1` is already
  in a pinned bounce page and being copied to a second device staging page on
  a copy stream. Results are bit-identical to `stage_next`; a paired
  measurement compares both; unused prefetched blocks are counted.
- **Allowed files:** `crates/moxie-cuda/src/{ffi.rs,status.rs,driver.rs}`
  (checked pinned free; host-function launch for the test gate),
  `crates/moxie-executor/src/paged_attention.rs`,
  `crates/moxie-memory/src/report.rs` only if the staging byte request lives
  there, `crates/moxie-executor/tests/paged_attention_device.rs` (one test and
  one ignored timing test), new `docs/evidence/kv-read-ahead.md`, this task's
  Result.
- **Non-goals:** raising `MAX_STAGED_BLOCKS`; depth > 1; changing
  `stage_next`; dense-step integration; pinning the host KV store itself.

## Numbered changes

1. **Admission.** A new staging variant `Staging::HostBackedReadAhead {
   max_staged_blocks }` admits **two** device staging pages (keys, values,
   table each) and, in the ledger's `Host(Pinned)` tier, **two** pinned
   bounce pages (keys + values + table entry each). Pinned admission requires
   the host snapshot to declare a `Host(Pinned)` cap; without one it is a
   typed refusal (`invalid("pinned", "no pinned cap is declared")`). The
   pinned pages are allocated after the charge; an allocation failure
   releases the charge.
2. **API.** On `NBlockStream`, `pub fn prefetch(&mut self, keys: Vec<u8>,
   values: Vec<u8>) -> Result<(), PagedRunRefused>` and `pub fn fold_next(&mut
   self) -> Result<Vec<DevicePartial>, PagedRunRefused>`, available only when
   the run was admitted with the read-ahead variant. Order: the caller calls
   `prefetch` for block 0, then repeatedly `prefetch(i + 1)` (while blocks
   remain) and `fold_next()` for block `i`. At most **two** blocks are ever
   outstanding (one being folded, one prefetched); a third `prefetch` before
   a `fold_next` is refused before touching anything. **A stream uses exactly
   one protocol:** `stage_next` refuses a run admitted with the read-ahead
   variant before enqueuing anything, and `prefetch`/`fold_next` refuse the
   other variants; `stage_next`'s behaviour for existing variants is
   unchanged.
3. **`prefetch`.** Validate the page exactly as `stage_next` does for its
   position. Copy the caller's `keys`/`values` synchronously (CPU memcpy)
   into the free pinned bounce page; device work never references caller
   memory. **Keep the original `Vec`s until the first device-copy enqueue
   succeeds and return them in `PagedRunRefused.source` on every earlier
   refusal;** after that enqueue, drop them (the run retains the pinned
   source). On the run's copy stream (a second `Stream` created at
   admission), enqueue the pinned→device copies into the free device staging
   page and record a per-page `copied` event. No wait on an earlier kernel is
   needed: `fold_next` synchronizes before returning (change 4), so a page is
   free whenever `prefetch` may target it.
4. **`fold_next`.** On the compute stream, `wait_event(copied[page])`, launch
   the partial kernel on that page (existing `launch_partial`, with its table
   entry), then `read_partials` (synchronizes the compute stream; after it the
   page's pinned bounce and device staging are free). Advance `next_base`,
   `remaining_*`, `host_to_device_bytes` exactly as `stage_next` does.
   `fold_next` has no caller vectors to return.
5. **Wasted work.** Add `pub fn prefetched_unused(&self) -> u64` on the run:
   blocks prefetched and never folded (for example, a stream dropped after a
   prefetch), counted when the stream ends.
6. **Checked pinned free (`moxie-cuda`).** `PinnedHostBuffer::free(self) ->
   Result<(), (Self, Error)>`: makes the context current, calls
   `cuMemFreeHost`, and on failure returns the buffer with the error (the
   buffer is not freed, and `Drop` of the returned value keeps today's
   behaviour). The run uses it on close; a failed free keeps the pages and
   the charge in the returned refusal.
7. **Deterministic test gate (`moxie-cuda`, test hooks only).** Add
   `unsafe fn Stream::launch_host_func(&self, f: unsafe extern "C" fn(*mut
   c_void), data: *mut c_void) -> Result<()>` (`cuLaunchHostFunc`, checked
   against `cuda.h`). Under the `paged-attention-test-hooks` feature, a run
   flag makes `prefetch` enqueue, on the copy stream **before** its copies, a
   host function that blocks until a static release flag is set (bounded by a
   10 s deadline, as `tests/support/event_gate.rs` does). The test releases it
   after `fold_next` has been called from another thread, or after a short
   sleep in a spawned thread; with the `copied` wait present the output is
   still bit-identical.

## Escape inventory (what keeps each resource valid on every path)

| Resource | Owner | Last device use | Kept valid by |
|---|---|---|---|
| Caller's `keys`/`values` `Vec`s | caller | none | copied to pinned first; kept and returned on every refusal before the first device-copy enqueue; dropped after it |
| Pinned bounce page | run | its pinned→device copy | reused only after `fold_next` has synchronized the compute stream that waited on its `copied` event |
| Device staging page | run | the partial kernel that reads it | `fold_next` synchronizes before returning; `prefetch` can only target a page whose last kernel has completed |
| Query, resident KV pages, page table, partial output | run | the resident and staged partial kernels | unchanged from `stage_next`: owned by the run; synchronous readback after each kernel |
| `copied` events | run | the compute stream's wait | owned by the run; replaced only after the `fold_next` that waited on it has synchronized |
| Copy stream | run | last prefetch copy | dropped only after a drain (below) |
| CUDA module | run | every partial kernel | task 0082 R1 rule: never unloaded while work may run |
| Pinned charge in the ledger | run | — | released only after a **checked** free of both pinned pages succeeds (change 6) |

Exit paths, each required:

- **Refusal before the first device-copy enqueue** (validation, third
  prefetch, wrong variant or protocol, event creation): the caller's `Vec`s
  come back in `PagedRunRefused.source`; nothing else changes.
- **Refusal after an enqueue** (copy, wait, launch failure): the run is
  quarantined; every run-owned resource stays owned.
- **An event that cannot be recorded or observed after a submission:** the
  run is quarantined. `close` refuses unless a **context-wide drain**
  (`RankContext` synchronize) has succeeded; `Drop` attempts that drain and,
  if it fails, withholds (forgets) the pinned pages, device ranges, module and
  copy stream, and keeps the ledger charge.
- **`NBlockStream` dropped mid-stream** (prefetched, not folded): the run
  keeps the unobserved `copied` event as pending work that its next entry
  point observes first (task 0082 rule), and `prefetched_unused` is
  incremented.
- **Run close / drop:** as the event rule above; otherwise `close` observes
  all pending work, frees the pinned pages with the checked free, then
  releases their ledger charge.

## Design review (sol, 2026-09-25; adopted by the coordinator)

1. HIGH: mixing `stage_next` with `prefetch` could race a staging page →
   one protocol per stream (change 2).
2. HIGH: a failed event record leaves submitted work unobservable →
   quarantine and context-wide drain before close; `Drop` withholds on drain
   failure; query/resident/module added to the inventory.
3. MEDIUM: `PinnedHostBuffer::drop` ignores `cuMemFreeHost` failure → checked
   free (change 6); charge kept until it succeeds.
4. MEDIUM: caller `Vec`s dropped before a fallible pre-enqueue step → kept
   until the first device-copy enqueue succeeds (change 3).
5. MEDIUM: the overwrite wait was redundant under the synchronous
   `fold_next`, so its mutant could not fail → removed; the mutant targets the
   compute stream's wait on `copied`, with a deterministic gate (change 7).

## Tests

- **One GPU test** in `paged_attention_device.rs`, on every GPU: a history of
  one resident page plus three staged pages; run the query through
  `stage_next` on one run and through `prefetch`/`fold_next` on a read-ahead
  run; the merged outputs are **bit-identical**; `prefetched_unused() == 0`;
  then a second stream that prefetches block 0 and 1, folds block 0 and is
  dropped: `prefetched_unused() == 1`, and the run still closes with the
  ledger empty.
- **One ignored timing test**: the same history on one 3090, `W = 5`, `R =
  30`, median per-query time for `stage_next` versus read-ahead, plus the
  the copy-versus-kernel overlap if measurable from events.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `paged_attention_device`,
`dense_gemma_device`, `cargo xtask-cuda test-gpu` (69/69).

**Evidence:** `docs/evidence/kv-read-ahead.md`: the timing test twice;
one paragraph stating only what it shows. With only three staged blocks the
gain may be small; report it as measured.

**Coverage check (one mutant, reverted after):** with the change 7 gate held
during `fold_next`, remove the compute stream's `wait_event(copied[page])`;
the bit-identity test must fail (the kernel reads the page before its copy
lands). If it passes, stop and report.

**Stop conditions:** any path in the escape inventory cannot be implemented
as written; a change conflicts with the code; a file outside the allowed list
is needed.

## Result, filled after work

Implemented the two-page pinned and device staging path, the separate
`prefetch`/`fold_next` protocol, checked pinned frees, copied-event waits,
quarantine/drain behavior, unused-prefetch accounting, and the deterministic
test hook. The every-GPU test confirmed bit-identical merged outputs and
successful cleanup after a mid-stream drop. The wait-removal mutant failed
the output comparison with the gate held; the wait was restored.

Host gates passed: fmt, workspace clippy, executor driver-feature clippy,
workspace tests, `arch-check`, and `spec-check`. GPU gates passed with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`: `paged_attention_device` (8 passed, 1
ignored), `dense_gemma_device` (11 passed, 2 ignored), and
`cargo xtask-cuda test-gpu` (69/69 across SM86 and SM120).

Recorded two timing runs in `docs/evidence/kv-read-ahead.md`. On the target
RTX 3090, `stage_next` medians were 348.022 and 346.940 µs; read-ahead medians
were 346.241 and 349.557 µs. The timing difference changed direction between
runs, and overlap was not measurable from the available event intervals.
