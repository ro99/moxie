# Task 0009 — M1.3: event-backed leases

Status: **contract proposed**, 2026-09-09, after [task 0008](0008-m1-measured-host-capacity.md)
was accepted for the measured host-capacity slice.

**This contract is committed before any implementation code**, as in tasks 0003–0008. That commit
contains no `.rs` change. Nothing below is to be adjusted once a test has run.

## Identity and authority

- Task ID / milestone / owner: 0009 / M1.3 / implementation agent, owner review pending
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

- Event query returning not-ready is a state, never an error (`Event::is_complete`
  already types it so).
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

(Empty until implementation lands.)
