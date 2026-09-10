# Task 0009 — M1.3: event-backed leases

Status: **accepted for the event-retained transient-upload slice at `9b24eeb` after independent
review recorded at `7c36658`**, 2026-09-09. [Task 0008](0008-m1-measured-host-capacity.md) was
already accepted for the measured host-capacity slice.

**This contract is committed before any implementation code**, as in tasks 0003–0008. That commit
contains no `.rs` change. Nothing below is to be adjusted once a test has run.

## Identity and authority

- Task ID / milestone / owner: 0009 / M1.3 / implementation agent; bounded-slice acceptance is
  recorded above
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `b6b4de4`, tree clean
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Required documents: 02 (ownership table — `moxie-memory` owns leases, `moxie-executor`
  owns rank-local execution/events, `moxie-cuda` owns contexts/streams/events and may not
  name a `CapacitySnapshot`), 03 (event-retained source leases, lease-until-completion
  launches, retirement event-driven), 04 (leases released at turn boundaries and
  cancellation, R08), 06 (M1.3), 08 (R07, R08), 09 (playbooks A and E), AGENTS.md,
  and tasks [0006](0006-m1-resource-ledger-and-admission.md),
  [0007](0007-m1-rank-context-and-measured-capacity.md) and
  [0008](0008-m1-measured-host-capacity.md) for the ledger, the rank context with
  its `Stream`/`Event`/`DeviceBuffer` primitives, and the measured capacities
  a lease will spend.
- Owner gates: **none needed.** No checkpoint is read, downloaded or converted; nothing
  is published, benchmarked, or exposed. O5 is untouched and O6 is not reached. **Stop and
  ask** if the work appears to need any gate.

## Why this next

The ledger admits byte envelopes and the sensors measure both tiers, but nothing spends
what they measure: `DeviceBuffer` retires the blunt way (context synchronise in `Drop`)
and `copy_from_host_async` is `unsafe` precisely because no lease discharges the
source-lifetime obligation (R07). Task 0008's result named this half explicitly, in
order: event-backed leases, the basic allocator, the admitted execution plan and the
device-resident layer chain. This task is the first of those, and the first that will
charge *real* bytes rather than declared ones.

## Bounded deliverable

- **One concrete outcome:** a lease over real device bytes whose retirement is driven
  by a recorded CUDA event: acquire → use on a stream → record → retire only after
  completion. `Drop` alone never frees in-flight memory; reuse waits for completion
  across the streams and ranks the lease touched, including cancelled work.
- **Sole owning component:** a new `moxie-executor` crate — the row document 02's
  ownership table already names (rank-local execution, events, transfers). It is not
  a widening: `moxie-memory` documents that it holds no lease over real bytes and
  waits on no event, and an arch-check rule forbids `moxie-cuda` from naming ledger
  types, so neither existing crate can own the binding. The allowlist gains the row
  the table already promises.
- **Allowed production files:** `crates/moxie-executor/**` (new), `xtask/src/archcheck.rs`
  (allowlist + one dependency-direction assertion), `xtask/fixtures/**`,
  plus manifests and `Cargo.lock`.
- **Explicit non-goals and forbidden shortcuts:** no allocator or arena (next task — a
  lease covers one live range, it does not suballocate); no admitted execution plan or
  layer chain; no change to `DeviceBuffer`'s synchronise-in-`Drop` (it stays the safe
  fallback, and must not be "optimised" by deleting the synchronise); no weakening of
  the `moxie-cuda` → `moxie-memory` refusal; no `unsafe` added outside the audited
  driver boundary; and no figure from this task presented as a performance result.
- **Existing consumers and second-consumer proof:** the ledger's `Reservation` (byte
  accounting) and the rank context's `Stream`/`Event`/`DeviceBuffer` (mechanism). The
  proof is one async host-to-device upload and one stream-ordered use retired by the
  same event on a real GPU, plus a host-tested retirement state machine both consumers
  share.
- **Deletion plan:** none, and that is stated rather than invented. No event-backed
  lease exists today; `DeviceBuffer` stays.

## Contract, fixed before implementation

### Where the boundary sits

`moxie-memory` counts bytes and admits envelopes. `moxie-cuda` owns contexts, streams,
events and allocations, and knows no placement. `moxie-executor` binds them: it takes
a `Reservation` and a stream-ordered operation, records the completion event, and
releases both sides only when the event reports complete. The composition root wires
sensors to the ledger to leases, as it already wires sensors to the ledger.

### What a lease guarantees

```text
acquire(reservation, stream) -> Lease
use(lease, operation)        // enqueue ordered after prior work on stream
retire(lease)                // Ok only after the recorded event is complete
release_turn()               // sweep: retire every completed lease; report the rest
```

- An upload leases its source bytes through the completion event (R07): the source
  must outlive the copy, and only the event knows when that stops being true. Scratch
  reuse before completion is an error, not a race.
- A launch leases its inputs, outputs and workspace until completion. Retirement is
  event-driven; `Drop` alone must not free in-flight memory. A dropped lease stays
  charged and visible, exactly like a dropped `Reservation` — R08's leak was
  invisible, not large.
- Reuse waits for all dependent streams and ranks, including cancelled work. A
  cancelled lease is still in flight until its event completes or the context is
  known lost; cancellation retires the *intent*, never the bytes.
- Turn boundaries sweep: `release_turn` retires everything complete and names what
  is still held, so a turn that ends with no next token cannot leak the way
  experiment 0125's KV leases did (R08).
### Failure and cancellation

- `Event` query returning not-ready is a state, never an error
  (`Event::is_complete` already types it so).
- A failed or lost context fails closed: affected leases report `DeviceLost` and are
  withheld from reuse, following the rank-claim teardown precedent from task 0007.
- A `retire` before completion is refused with the lease's identity and stream, not
  a silent wait: blocking is the caller's explicit `synchronize`, never hidden
  inside release.


### Reading is real; the state machine is host-tested

Completion is observed from the driver on a real GPU. The retirement state machine
(live → completed → retired, plus cancelled and lost) is pure over an injected
completion source, so every transition — including retire-before-complete refusal
and turn-sweep reporting — is tested without a GPU. The GPU tests prove the
mechanism end to end, not the transitions.

## Error metrics

None. Leases move byte identity and event state; every quantity is an exact handle
or a counted byte from the ledger. The numerical contracts of tasks 0003, 0004, 0006
and 0007 are unchanged.

## Acceptance

- Host lane: the retirement state machine over an injected completion source —
  acquire/use/record/retire ordering, retire-before-complete refused with identity,
  dropped lease stays charged and visible, turn sweep retires the completed and
  names the held, cancellation retires intent but withholds bytes until completion
  or loss, lost context withholds from reuse. One test per rule above.
- Device lane (`test-gpu`, both architectures): one async H2D upload whose source is
  reused only after the recorded event completes; one stream-ordered use retired by
  the same event; a bite check deleting the event wait fails at least the reuse
  test. No new kernel, no benchmark.
- R08 regressions: a turn ending with no next token releases everything completable
  and reports the rest; a cancelled lease is not reusable before completion. Shaped
  after experiment 0125 and `test_deepseek_kv_cache` / `test_deepseek_runtime`.
- `cargo xtask arch-check` passes with `moxie-executor` declared and its
  `moxie-cuda` + `moxie-memory` edges permitted, each exercised by the crate itself
  compiling and by a rejecting fixture proving a model crate still cannot reach
  execution primitives.
- `fmt`; `clippy -D warnings` on **both** lanes; the full host suite; `spec-check`;
  the no-driver lane with `xtask` rebuilt before `ldd`; the device lane; `test-gpu`
  with both architectures qualified; and the `CUDA_VISIBLE_DEVICES=1,2` negative
  check still exiting 1.
- Support matrix: the oversized-host/disk streaming row stays **not implemented**
  (an allocator, residency lifecycle and eviction are still missing); it must not
  be claimed on the strength of leases alone.
- Stop condition: stop and report if this appears to need an allocator, an arena,
  a plan, a layer chain, a checkpoint, or a performance claim. Each is a different
  task.

## Result, filled after work

Implemented 2026-09-09 on branch `main`, on top of the contract commit for this
task. The contract above is unchanged; only this section is filled in.

What was built:

- `moxie-executor` (new, depends on `moxie-types`, `moxie-memory`, `moxie-cuda`;
  `driver` feature forwards to `moxie-cuda/driver` so the host lane never links
  `libcuda`): `Lease` over an owned `Reservation` bound to one `Completion`
  source, `retire` that queries and hands the lease back on refusal, `Turn`
  with `release_turn` reporting retired vs held, `ManualCompletion` test double,
  and `Completion for moxie_cuda::Event` plus `Lease::use_on` behind `driver`.
- `arch-check`: `moxie-executor` row (`moxie-types`, `moxie-memory`,
  `moxie-cuda`) and the `xtask` composition-root edge, with rejecting fixture
  `model-reaches-executor` proving a model crate cannot reach execution.
- `xtask/cuda` gains `dep:moxie-executor` + `moxie-executor/driver`; `test-gpu`
  gains `event_backed_lease` (two leased async uploads, one retired directly,
  one via a turn sweep, sources reused only after retirement, bytes verified).
- The device lane feature list gains `moxie-executor/driver`, recorded in the
  support matrix.

Bite checks, each reverted to green:

- Deleting the completion query in `retire` fails five host tests (refusal,
  reuse-after-refusal, both-sided release, turn naming, wait-is-no-backdoor).
- Deleting `lease_a.synchronize()` in the GPU case fails `event_backed_lease`
  on two of three devices with still-`InFlight` refusal — the third won the
  race, which is itself the point: refusal is observed on hardware, and the
  deterministic refusal proof lives in the host tests.

Deviations recorded, none silent:

- `Turn::synchronize` and `Lease::synchronize` are small additions beyond the
  contract's four named operations: retirement never blocks, so the caller's
  explicit wait needed a typed home, and the turn sweep needed one for the
  same reason. Both are pure observation, never release.
- `LostInfo` is boxed so the context-loss record does not inflate every live
  lease (`result_large_err`); the public API is unchanged by it.

Gates (this machine, 2026-09-09; nothing loosened):

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `clippy -D warnings`, host and device features | PASS (device lane with `moxie-executor/driver`) |
| `cargo test --workspace --locked --offline` | PASS, 521 unit/integration + 6 doctests (was 515 + 6; +6 drop-withhold, fit, sweep-resource rules) |
| `cargo xtask arch-check` | PASS, 52 rejected + 14 accepted fixtures, 12 rules, unchanged |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane, `xtask` rebuilt before `ldd` | PASS, 521 + 6, no `libcuda` |
| device lane | PASS, 529 + 8 (was 523 + 8) |
| `cargo xtask-cuda test-gpu` | PASS, 30 cases, `sm_86` and `sm_120` qualified (was 27; +`lease_quarantine_on_cross_context_record`) |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, as intended (`sm_120` unqualified) |
| `cargo xtask-cuda capacity` | PASS, the host and 3 devices, every scope to zero |

Nothing failed. Nothing was skipped. `Cargo.lock` gains only the
`moxie-executor` package and the `xtask` edge: no new third-party packages.
No checkpoint was read, downloaded or converted; ~~nothing was allocated, mapped
or reclaimed~~ (incorrect: the GPU tests allocate and free real upload buffers);
no swap entered any budget, and no performance number was produced. The stop condition was not triggered.

Remaining blockers and next bounded task: the spending half continues — the
basic allocator (suballocating admitted envelopes into live ranges), the
admitted execution plan, and the device-resident layer chain.

## Review corrections, 2026-09-09 (two rounds)

Review of the committed tree found ownership and recovery gaps, all reproduced
by the reviewer — first five against `f7d1f6b`, then three deeper ones against
`361049e` on submission, destruction, and accounting lifetimes. The contract
above is preserved as written; what follows corrects the implementation toward
its guarantees.

**Leases own their resources.** `Lease<C, R>` carries the operation resource
alongside the reservation: `retain` moves it in exactly once (enforced by
type, only from `Lease<C, ()>`), and retirement settles it through
`SettledResource` — transient `()` and host bytes to themselves, `Upload` to
its source with the device allocation freed at the observed completion, so no
live allocation outlives its charge. A `compile_fail` doctest proves a
retained source has no early-mutation alias. `acquire` takes the ledger, binds
the admitted scope charges read from it, and returns `AcquireRefused`
(reservation included) on an empty label or a foreign reservation.

**Submission is one operation.** `Upload::prepare` allocates without
enqueueing; `Lease::submit` checks the admitted budget (`check_fit`, pure and
host-tested), runs the async copy, records the event, tracks it, and retains
— with nothing submitted untracked at any step. Admission precedes allocation.
A 4,096-byte upload against a 512-byte reservation is refused with
`CapacityExceeded` before any driver call.

**Drops withhold, sweeps return.** A dropped lease with a tracked completion —
in flight, cancelled, or lost — deliberately withholds its retained resource
instead of freeing it early; the reservation still drops into its charged and
visible state. Only provably unsubmitted (Live, untracked) drops normally.
Destructor-counting tests cover all four shapes. `HeldLease` carries the lease
back, and sweeps return settled resources: sweep → complete → sweep-again
releases with no next token, including the cancelled variant.

**Failed records quarantine; loss persists.** `use_on` and `submit` take the
context for failure attribution; a failed record marks the lease `Lost`
(submission may already have happened) and refuses. Bare events can no longer
reach tracking: `track` is now `track_manual`, gated by the `TrackableManual`
marker the doubles implement and `Event` does not. Query and synchronize
errors of kind `DeviceLost` transition to `Lost`; later observations,
including a racing `Ok(true)`, never reopen it (`ScriptedCompletion` plays
the race; transient failures stay usable). Deterministic hardware proof:
`lease_quarantine_on_cross_context_record` records across two live contexts
and asserts `Lost`, `device_lost`, and still-charged bytes on every device.

Bite checks, each reverted to green:

- Deleting the completion query in `retire` fails seven host tests.
- Deleting `lease_a.synchronize()` failed `event_backed_lease` on two of three
  devices, then three of three after the first corrections — still-`InFlight`
  refusal in every failing case. Under the final flow the bite no longer
  bites: readback is a synchronous copy before retirement, so it observes
  completion first — the refusal proof moved to the host suite (deterministic)
  and the quarantine case (on hardware). A bite that stops biting when the
  design removes the hazard is the mechanism working, not coverage lost.

Gates for the corrections (this machine, 2026-09-09; nothing loosened) are
recorded in the lane table below, updated in place.

## Takeover contract — admission and checked teardown (2026-09-09)

Codex takes over the interrupted Muse implementation at the owner's request, on
`main` at `d525d07`. Muse stopped on provider rate limiting with partial edits in
`moxie-cuda/src/driver.rs` and `moxie-executor/src/lease.rs`. Those edits are retained
and completed, including the pre-settlement ledger identity check.

The bounded correction remains task 0009, not the allocator: one transient upload
per owned reservation. The executor validates the owning context's UUID, device
`TransferStaging` and host `Pageable` charges, plus both scope peaks, before CUDA
allocation. The host input is caller-owned before entry and after successful
retirement; while retained, its full Vec capacity is charged. A preparation refusal
returns the original lease and source. There is no public unadmitted `Upload`
constructor, caller-selected upload scope, or separate caller-enqueued `use_on`.
Preparation allocates only; submission validates stream/event identity and owns
copy plus event recording. Any failure after calling the copy quarantines the
lease. Pre-submission validation failure remains recoverable.

Retirement first validates ledger ownership and observes completion, then performs
checked allocation cleanup, then releases the reservation. Cleanup failure returns
the retained resource, quarantines it and preserves the charge. The CUDA checked
free has an explicit unsafe ordering contract and does not add a context-wide
synchronize. The old DeviceBuffer destructor remains the conservative fallback for
other consumers. Persistent allocation transfer remains the allocator's next task.

Acceptance adds host tests for both tiers/scopes, oversized host capacity, failed
settlement and wrong-ledger ordering; real-driver symbol-interposition regressions
for allocation-before-refusal, copy/record errors and failed free; plus all existing
host, architecture, specification, no-driver and real-device gates. The actual FFI
branches must be exercised, not only their policy doubles. Superseded upload entry
points are deleted; original contract text and failed experiments remain visible.


### Takeover results — complete, local only

All three remaining findings are fixed. Muse's checked-free work is completed with
an **unsafe ordering contract**, and its ledger identity check now runs before any
retirement side effect. The executor's private upload owner discharges that contract.
The infallible settlement API is replaced with a resource-returning refusal; the
ledger charge ends only after cleanup succeeds. Sources are returned to their caller,
not to an unaccounted engine cache. No bare upload constructor, caller-supplied GPU
scope, exposed device buffer or separate `use_on` path remains.

`driver_faults` exercises the actual FFI branches on all three UUIDs, without changing
the driver or adding production injection hooks. The test executable interposes and
forwards CUDA symbols, injecting errors only in its own process. Assertions confirm
that the interceptors run on the happy path. Over-budget, absent-scope, wrong-tier
and oversized-Vec-capacity preparations make **zero allocation calls**. Lost and
non-loss copy/record/free errors quarantine, retain the original charge and do not
retry free on drop. A successful free occurs once, with **zero explicit context
synchronization calls during retirement**. CUDA allocator latency itself is not
measured or promised. Fault tests deliberately withhold six small allocations and
sources per device; these remain named until process teardown.

The cross-context GPU case now rejects a foreign event *before* copying and allows
safe abandonment; it is named `lease_rejects_foreign_completion`. Real post-copy
record failures are covered by the fault test. This replaces the earlier artificial
`use_on` quarantine probe, not its failure coverage.

| Gate rerun on final implementation | Result |
|---|---|
| fmt | PASS |
| clippy `--workspace --all-targets --locked --offline -- -D warnings`, host and full device features | PASS both |
| `cargo test --workspace --locked --offline` | PASS, 523 unit/integration + 7 doctests |
| Same with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda` | PASS, 532 unit/integration + 9 doctests, including real-driver fault injection |
| arch-check / spec-check | PASS, 52 rejected + 14 accepted / 12 rules; 10 unchanged documents |
| Isolated no-driver target, `CUDA_HOME=/nonexistent NVCC=/nonexistent`, sanitized PATH, unset LD_LIBRARY_PATH | PASS, 523 + 7; explicit xtask build then ldd, no libcuda |
| `cargo xtask-cuda test-gpu` | PASS, 30 cases, all 3 devices, sm_86 and sm_120 qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | Expected exit 1, sm_120 unqualified |
| `cargo xtask-cuda capacity`, normal and reversed visibility | PASS, host and 3 devices |

Counts above distinguish integration tests from doctests from Cargo's output; the
older records classified one doctest as a unit/integration test. No required gate
was skipped or carried forward. Quality, model execution and performance remain
unmeasured and outside this slice.

Negative checks, with all mutations restored:

- Removing the pre-allocation budget check makes the driver regression accept an
  oversized upload and fail (exit 101).
- Removing copy-error quarantine leaves the lease Live instead of Lost and fails
  the driver regression (exit 101).
- Releasing the ledger before settlement loses the charge on failed free and fails
  the driver regression (exit 101).
- The retained-source doctest, with `compile_fail` removed, fails with **E0382**.
  Removing the retaining move makes the control compile. Its source is now mutable,
  so an unrelated immutable-variable error cannot satisfy the test.
- An initial device-suite compilation failed while adding the interceptor-liveness
  assertion (missing local `copies`); corrected before the successful full rerun.

The next bounded task is the basic allocator, as detailed in the
[handover](../handovers/2026-09-09-task0009-checked-upload-retirement.md). No push is
performed by this takeover.

Raw logs are in `/tmp/moxie-task0009-takeover/` on this machine, retained for the
owner's review (temporary, not a permanent artifact store). Regression sources and
commands are version-controlled so the evidence can be regenerated. Log hashes:

| Log | SHA-256 |
|---|---|
| `host-tests.log` | `e574b9b54efec550bfecb140b68d9d81f19e2052fc7deaf6d0a83cc2d66b15f3` |
| `device-tests.log` | `d9fe2403c748e57b84c20263d7c7ce2b86a830923793c9e63b9f03dc4579a26e` |
| `no-driver-tests.log` | `bd184035923c79559d6aa336a9423d07c0e5fca6ac7f52142db1b20a9efbdba6` |
| `no-driver-ldd.log` | `84ba98abde130c3b3ae400c88b7a566da10bf1b17b937ef16997104d087187be` |
| `gpu.log` | `a3822473f6581cb58a58a43accb46bb32888b817b748870bd2f4187a69cba306` |
| `gpu-negative.log` | `cac359779414b398572463f3d68a236d7506665c545dbd28798ad94690243429` |
| `capacity.log` | `57454ba0c89a7dbfa453b3526319f73f99ab836e58a14657edc783e355ba3f09` |
| `capacity-reversed.log` | `9f024105f2dc0d2c3d1b951944fa963eaab7feefe36b7cd688f4dca8ad592b89` |
| `bite-admission.log` | `4fdb331eaaab583bb6cba3a4466d9b32a99972b3668e51a7bb462536cabbd89d` |
| `bite-copy_quarantine.log` | `4974942b7820a0c651d121ebd7903280b4baf92cce38869f5e0d389b31e5800f` |
| `bite-settlement_order.log` | `98c4415c4c6d28af6c181778cef7ea6c3bfb76334445d7d7e156a807487dec72` |
