# Task 0010 — M1.3: basic admitted device arena

Status: **implemented; independent review pending**, 2026-09-09, after
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

Contract committed separately as `0bab91b`, with no Rust change. Implementation
adds pure range metadata in `moxie-memory`, one admitted physical arena in
`moxie-executor`, and checked offset copies in `moxie-cuda`. Reservation leases
and range operation leases share one completion/loss implementation. Existing
transient upload consumers remain covered by the full device suite.

The basic arena retains at most one host upload source at a time, charging its
full Vec capacity against the parent reservation's host Pageable tier. A source
returns to the caller at retirement; its range remains allocated until explicit
release. This restriction bounds source overlap until a later task introduces
multiple concurrent sources and event fan-in. Arena creation additionally rejects
reservations charging another device UUID before allocation. Other unmaterialized
charges remain reserved until the one arena closes; no second consumer can obtain
the consumed reservation.

Final audit corrected right-neighbor coalescing when a live separator prevents a
left merge. The fragmentation test now covers that release order. It also found
that a returned failed-close arena must refuse new allocations: quarantine now
blocks allocation as well as repeated close, with a real-driver regression.

### Validation

- Passed: locked/offline host and full device-feature workspace tests; focused
  allocator and executor tests; format; both host and full device-feature clippy
  with `-D warnings`; architecture checks (52 rejected, 14 accepted fixtures,
  12 rules); unchanged-reference checks (10 documents).
- Host totals: 532 unit/integration tests and 8 doctests; the same totals passed
  in the isolated no-driver lane. The final focused memory run includes the
  additional coalescing release order; the focused executor run includes dropped
  in-flight range withholding.
- Full device-feature totals: 542 unit/integration tests and 10 doctests.
- Passed: real-device arena and allocation/copy/record/free fault tests on all
  three UUIDs. `test-gpu` passes 33 cases across sm_86 and sm_120. The real arena
  test observes early retirement refusal, distinct offset readbacks, cancellation,
  persistent owner transfer, generation change, and physical/ledger reconciliation.
- Passed: isolated host suite with CUDA tools excluded and explicit plain xtask
  build; `ldd` contains no `libcuda`.
- Passed negative qualification: `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda
  test-gpu` exits 1 with sm_120 explicitly unqualified despite all visible sm_86
  cases passing.
- Passed: `cargo xtask-cuda capacity` with normal visibility and with
  `CUDA_VISIBLE_DEVICES=2,1,0`; host and all three UUIDs measured and admitted.
- Negative experiments: bypassing operation completion made the deterministic
  cancellation/reuse test fail (exit 101); the mutation was reverted. An initial
  overflow fixture produced a capacity refusal instead of arithmetic overflow;
  the fixture was corrected to exercise an actual checked-add overflow. Initial
  formatting differences were corrected. No acceptance threshold was loosened.
- Skipped/unmeasured: model quality, context execution, topology communication,
  paired inference performance, residency/eviction and kernels; these are outside
  this bounded task. No checkpoint was accessed or transformed.

No production path was deleted. Task 0009's conservative upload path expires only
after a later integration task passes its replacement gates. Ambiguous failures
deliberately retain allocations and charges until process teardown; no recovery
or performance claim is made. Independent review remains required before accepting
this slice or starting its dependent implementation.

Real-device identities: RTX 5060 Ti
`GPU-97fe4889-4874-a378-198e-955d2e72c4a3`, RTX 3090
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, RTX 3090
`GPU-81fe4578-59b2-37c4-421e-287cdac78704`. Local verification logs are under
`/tmp/moxie-task0010-final/`; the reproducible commands and outcomes above are
the tracked evidence, since temporary logs are not a durable dependency.

### Independent-review corrections — 2026-09-09

The [independent review](../handovers/2026-09-09-task0010-independent-review.md)
requested R1/R2 changes against `4037fcc`. Its findings are preserved unchanged;
the original passing tests did not establish bounded generation metadata or
controlled pending-work sweeps on every device. Acceptance remains pending
independent re-review; the fixed contract above is unchanged.

R1: removed the per-offset generation history. The existing checked arena-wide
allocation sequence now supplies generations; exhaustion refuses atomically
before free-list mutation. Reusing any offset strictly increases generation.
`arena_history` uses an independent counting global allocator in a single-test
executable: after warming the fixed live-set free-list capacity, 100,000 distinct
offset cycles with at most two live ranges retain **zero additional requested heap
bytes**. The arena ends completely free. The original review measured 3,429,296
bytes of retained history. This is a metadata resource result, not RSS or inference
performance. The atomic-refusal test also covers exhausted generation identity.
The reviewer's unchanged standalone probe, rebuilt against the corrected library,
reports **944 bytes** from initial arena creation (fixed free-list/tree warm-up
capacity), versus its original 3,429,296 bytes. The regression warms that same
fixed live-set capacity before checking that further history retains zero bytes.

R2: replaced the timing-dependent aggregate flag with a test-only real CUDA
stream gate on each UUID. Symbol interposition enqueues `cuLaunchHostFunc` just
before the actual completion event; the callback calls no CUDA API, is released
by an unwind-safe guard, and has a ten-second failure timeout. Timeout fails the
test, never qualifies an early completion. Cancellation and the first
`OperationTurn` sweep must return exactly one held actual range/source and no
retired resource. Once the gate opens, synchronization and the second sweep
return the resource without a next token. Existing readback, transfer, explicit
release, full-arena coalescing/reuse and ledger/free-memory reconciliation remain.
All three UUIDs printed a separate controlled-pending/cancel/two-sweep pass.

Production ownership/FFI APIs, CUDA kernels and reference documents are unchanged
by these corrections. The test gate remains only in the integration executable;
it creates no production injection switch or fabricated event result.

Correction validation: **passed** full locked/offline host (533 tests + 8
doctests) and device-feature workspace suites (543 tests + 10 doctests), the
focused memory suite including generation exhaustion, all 33 GPU harness cases,
format, host/device clippy with warnings denied, architecture (52 negative/14
positive fixtures, 12 rules) and all 10 unchanged-reference checks. Local logs:
`/tmp/moxie-task0010-corrections/`. **Failed:** no final correction gate.
**Not rerun:** isolated no-driver build/ldd and capacity/visibility probes; their
earlier task/review results remain recorded separately. **Unmeasured:** model,
context, quality and paired performance gates remain outside this correction.
