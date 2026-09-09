# Task 0010 — M1.3: basic admitted device arena

Status: **contract proposed**, 2026-09-09, after
[task 0009](0009-m1-event-backed-leases.md) was independently reviewed and accepted for its
transient-upload slice.

**This contract is committed before any implementation code**, as in tasks 0003–0009. That
contract commit contains no `.rs` change. The contract below is fixed before tests run; corrections
or deviations belong in the Result section rather than rewritten acceptance criteria.

## Identity and authority

- Task ID / milestone / owner: 0010 / M1.3 / implementation agent, owner review pending.
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `7c36658`, initially
  clean. Local commits are allowed by the assignment; no push is authorized.
- Read-only legacy root: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its existing untracked `.pi/` and `tests/p2p/`
  paths are untouched.
- Requirements repaired: document 02's allocation-generation, bounds and event-retained reuse
  contract; document 03's one real arena/accounting authority; M1.3's basic allocator; R03's five
  private caches and string-classified arena exhaustion; R07's asynchronous lifetime; R08's
  no-next-token release; R11's bounded arena; and R14's warning that an optional shared arena is
  not integration.
- Required living records: tasks 0006–0009 and their final handovers; ADR 0006 for sensor
  separation. Actual legacy sources read include `backend_model_kernels.cuh`'s `WeightArena`,
  `backend_core.inc.cuh`'s `reserve_weight_arena`/upload path,
  `test_cuda_backend.cpp`'s bounded-arena case, the five cache sources in R03, and Kimi's arena and
  arena tests in R11.
- Owner gates: **none needed**. No checkpoint, conversion, quality result, benchmark, public API or
  operational change is involved. Stop if O1–O7 appears necessary.

## Bounded deliverable

- **One concrete outcome:** one real device allocation, made only after ledger admission, is a
  bounded arena that suballocates aligned live ranges. A range used by asynchronous work cannot be
  returned or reused until its recorded event completes. Closing the arena frees the physical
  allocation before releasing the parent reservation.
- **Shared owners:** `moxie-memory` owns allocator metadata, range identity/generation, bounds,
  free-space coalescing and ownership labels. `moxie-executor` binds one such allocator to one
  reservation, rank context, CUDA allocation and the existing completion-state mechanism.
  `moxie-cuda` supplies only checked offset copies and checked final free; it knows no ledger or
  allocator policy.
- **Allowed files:** `crates/moxie-memory/**`, `crates/moxie-executor/**`, the audited
  `crates/moxie-cuda/src/driver.rs` boundary, `xtask/src/gpu.rs`, manifests/lockfile,
  `docs/evidence/support-matrix.md`, this task and its handover. Architecture fixtures may change
  only if an ownership edge actually changes.
- **Non-goals:** no cache key, hit/miss or victim policy; no residency lifecycle, demand/prefetch,
  host arena, pinned memory, mapping or disk I/O; no admitted plan lowering; no kernel or layer
  chain; no multi-stream/rank fan-in; no model/checkpoint; no performance claim or tuning.
- **Forbidden shortcuts:** no allocation before admission; no caller-supplied raw pointer; no
  per-range CUDA allocation fallback; no release in `Drop`; no reuse after cancellation but before
  completion; no retry of an ambiguous failed free; no parsing error strings for exhaustion; no
  caller-chosen UUID that can disagree with the rank context.
- **Consumers:** transient staging proves a range can be filled and read back; persistent ownership
  proves the same physical range transfers between two accountable engine owners without CUDA
  reallocation or ledger release. These are synthetic consumers, not model support.
- **Deletion/expiry:** no production allocator exists to delete. Task 0009's one-allocation upload
  remains the conservative transient path until a later integration task migrates it; this task
  must not silently replace its source-return contract.

## Contract before implementation

### Metadata arena

`moxie-memory::Arena` is pure, contains no pointer and uses checked `u64` arithmetic. Creation
requires a nonempty label, nonzero capacity and power-of-two base alignment. It begins with one free
range and exposes exact occupancy:

```text
capacity, live bytes, free bytes, largest free range, free-range count,
and every outstanding allocation (id, generation, owner, offset, logical bytes,
reserved/aligned bytes, alignment)
```

`allocate(bytes, alignment, owner)` is deterministic address-ordered first fit. Bytes and owner are
nonempty; alignment is a nonzero power of two no greater than the arena's base alignment. The
allocator aligns the start, charges all consumed bytes including alignment padding, splits the free
range with checked arithmetic, and refuses atomically with typed occupancy if no single range fits.
It never silently falls back. `release(allocation)` consumes a non-Clone handle, validates arena,
allocation ID and generation, then coalesces adjacent ranges. A dropped handle releases nothing and
remains visible. Reusing an offset increments its generation, so a stale `(offset, generation)`
cannot identify new storage.

`transfer(allocation, new_owner)` consumes and returns the same allocation identity after changing
its accountable owner. It neither frees nor allocates bytes. Empty owners and foreign/stale handles
are refused without changing either arena. This is ownership transfer, not model residency policy.

### Real device arena

`moxie-executor::DeviceArena` consumes one outstanding `Reservation` and binds it to exactly one
device UUID and device tier already charged by that reservation. This task permits one physical
arena per reservation; mutually exclusive per-tier peaks are not materialized concurrently by
pretending their sum is reserved. Requested arena capacity must fit both the reservation's device
scope charge and the selected tier charge before `cuMemAlloc` is called. Zero, unaligned and host
tier requests are refused with the reservation returned.

The rank context supplies UUID and allocation. The device buffer and reservation remain owned by
the arena. `close` is explicit and succeeds only with no outstanding ranges: checked CUDA free
first, ledger release second. Wrong-ledger close performs no side effect. Failed free quarantines
the arena and reservation and is never retried by `Drop`. Dropping an open arena withholds both the
physical allocation and charge; it does not invoke the synchronizing `DeviceBuffer` fallback.

Every device range carries arena ID, allocation ID, generation, owner, byte bounds and a strong
borrow of the arena allocation. No public operation accepts an independently supplied device
pointer or UUID. Checked offset addition proves a copy cannot escape the range.

### Event-retained use and reuse

The lease state implementation used by task 0009 is factored, not copied, so reservation leases and
arena-range operation leases have the same `Live -> InFlight/Cancelled/Lost -> retired` rules and
loss persistence. An operation lease owns its range and any host source while work is pending.
Submission validates range/stream/event device identity before the copy, enqueues exactly one
bounded H2D copy, records exactly one event, and only then reports `InFlight`. Any failure after a
copy may have been submitted marks the lease lost and withholds the range.

Retirement is nonblocking. Before completion it returns the lease intact. After completion it
returns the same allocation handle and source to the caller; the caller may retain/transfer the
range or explicitly release it to the arena. Cancellation retires intent only. A turn ending with
no next token can sweep operation leases; completed ranges return, held ranges remain named and
owned. Multi-stream fan-in is a later task and cannot be simulated by recording an event on only
one of several streams.

### Precision, layout and application effects

There is no model-value arithmetic, numerical tolerance, sampler, state or protocol effect.
Offsets and sizes are exact integers. This arena establishes byte bounds only; tensor logical dtype,
shape and `LayoutId` remain future handle work. The CUDA allocation's documented alignment is
treated only as the arena base alignment; each subrange additionally satisfies its declared
alignment.

## Acceptance

- **Host allocator:** exact-fit, aligned split, padding charge, deterministic first fit, exhaustion,
  fragmentation versus total free, both-neighbor coalescing, zero/overflow/invalid alignment,
  dropped-handle visibility, foreign/stale/double-release refusal, generation change on offset
  reuse, transfer identity/accounting, and close refusal while ranges exist. At least two allocation
  shapes and two ownership labels.
- **Host lifetime:** shared state-machine tests prove retire-before-completion, cancellation,
  persistent loss, dropped in-flight withholding, completed turn sweep and no-next-token second
  sweep for arena ranges. A bite mutation that skips completion must fail a deterministic host test.
- **Driver fault boundary:** interposed real CUDA symbols prove over-budget/invalid creation makes
  zero allocation calls; physical allocation failure returns the reservation; copy/record failure
  quarantines the range; failed final free retains the physical arena and ledger charge and is not
  retried; successful close performs one free before charge release and no explicit context-wide
  synchronize.
- **Real GPU, every visible UUID and both architectures:** create one admitted arena, allocate at
  least three differently sized/aligned ranges, upload/read back distinct bytes, refuse allocation
  when every fitting range is leased, observe a pre-completion reuse refusal, complete/cancel/sweep,
  transfer a persistent range between owners without another CUDA allocation, coalesce and reuse
  the full arena with a higher generation, then close with driver free memory and ledger outstanding
  state reconciled. No benchmark number is recorded.
- Run and report separately: format; host and full device-feature clippy with `-D warnings`; full
  locked/offline host and device-feature suites; `arch-check`; `spec-check`; isolated no-driver
  build/test/`ldd`; allocator driver-fault test; `test-gpu` on both architectures; restricted
  visibility negative qualification; and `capacity` in normal and reordered visibility.
- Update the support matrix with a narrowly worded device-arena row. Oversized streaming remains
  **not implemented**; a basic allocator is not residency, eviction or execution.
- Stop and report if the implementation requires a second physical arena per reservation,
  multi-stream fan-in, host storage, eviction, plan lowering, kernel execution, a checkpoint or a
  performance assertion.

## Result, filled after work

Pending implementation and independent review.
