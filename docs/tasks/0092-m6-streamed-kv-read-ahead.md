# Task 0092 — bounded read-ahead for host-backed KV streaming

Status: **proposed** (coordinator, 2026-09-25); the contract goes to sol for a
design review of its escape inventory before implementation (engineering
log, 2026-09-25). Builder Codex `luna`; reviewer Codex `sol`.

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
- **Allowed files:** `crates/moxie-executor/src/paged_attention.rs`,
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
   a `fold_next` is refused before touching anything.
3. **`prefetch`.** Validate the page exactly as `stage_next` does for its
   position. **Copy the caller's `keys`/`values` synchronously (CPU memcpy)
   into the free pinned bounce page, then drop or return the `Vec`s** — no
   caller memory is ever referenced by device work. Then, on the run's copy
   stream (a second `Stream` created at admission), make it wait on the event
   of the last kernel that read the target device staging page, enqueue the
   pinned→device copies, and record a per-page `copied` event.
4. **`fold_next`.** On the compute stream, `wait_event(copied[page])`, launch
   the partial kernel on that page (existing `launch_partial`, with its table
   entry), record a per-page `consumed` event, then `read_partials`
   (synchronizes the compute stream; after it, the page's pinned bounce and
   device staging are free). Advance `next_base`, `remaining_*`,
   `host_to_device_bytes` exactly as `stage_next` does.
5. **Wasted work.** Add `pub fn prefetched_unused(&self) -> u64` on the run:
   blocks prefetched and never folded (for example, a stream dropped after a
   prefetch), counted when the stream ends.

## Escape inventory (what keeps each resource valid on every path)

| Resource | Owner | Last device use | Kept valid by |
|---|---|---|---|
| Caller's `keys`/`values` `Vec`s | caller | none | copied to pinned before any enqueue; never referenced by device work |
| Pinned bounce page | run | its pinned→device copy | reused only after its `copied` event is observed (by `fold_next`'s synchronize or by `observe_pending`) |
| Device staging page | run | the partial kernel that reads it | the copy stream waits on its `consumed` event before overwriting it |
| `copied` / `consumed` events | run | the stream wait / host observation | owned by the run; replaced only after observed |
| Copy stream | run | last prefetch copy | dropped only after the run observes all its events |
| Partial output buffer | run | `read_partials` | unchanged (synchronous readback) |

Exit paths, each required:

- **Refusal before enqueue** (validation, third prefetch, wrong variant): the
  caller's `Vec`s come back in `PagedRunRefused.source`; nothing changes.
- **Refusal after enqueue** (copy, wait, launch or record failure): the run is
  quarantined; the pinned pages and staging stay owned by the run; the
  caller's `Vec`s were already copied, so `source` is `None` only if they
  were not returned before the failure (state which in code).
- **`NBlockStream` dropped mid-stream** (prefetched, not folded): the pending
  `copied` event is left in the run (`pending` from task 0082, or an
  equivalent per-page event the run observes first on its next use), and
  `prefetched_unused` is incremented. Nothing is freed while a copy may run.
- **Run close / drop:** `close` observes every event first (task 0082 rule:
  every entry observes pending work); `Drop` with unobserved work follows
  task 0082 R1 (never unloads the module or frees pinned pages while work may
  run: forget them if observation fails).
- **Ledger:** the pinned charge is released only after the pinned pages are
  freed, which is only after every event naming them is observed.

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
  `copied`-to-`consumed` overlap if measurable from events.

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

**Coverage check (one mutant, reverted after):** skip the copy stream's wait
on `consumed` before overwriting a staging page; the bit-identity test must
fail on at least one GPU. If it passes everywhere, stop and report (the
fixture cannot see the race; the coordinator will decide).

**Stop conditions:** any path in the escape inventory cannot be implemented
as written; a change conflicts with the code; a file outside the allowed list
is needed.

## Result, filled after work

(pending)
