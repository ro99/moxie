# Handover — task 0009 checked upload retirement

## Workspace identity

Writable repository: `/home/rodrigo/Developer/moxie`, `main`, based on `d525d07`.
This handover accompanies the local correction commit; it is not authorization to
push. The owner asked Codex to take over Muse's interrupted work. Herdr pane
`wB:pF` showed provider `429 FreeUsageLimitError` after ten retries, and was idle.
Muse's two partial edits (CUDA checked free and executor settlement/ledger binding)
were completed, not discarded. All accompanying changes belong to task 0009.

Read-only legacy: `/home/rodrigo/Developer/strata` at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, unchanged. R07 source-retention and
R08 no-next-token cleanup remain the source lessons. No checkpoints were used.

## Completed facts

See the final takeover results and gate table in
[task 0009](../tasks/0009-m1-event-backed-leases.md). The shared executor now checks
both tiers/scopes before allocation, derives GPU identity from its owning context,
retains the source's entire capacity, and refuses foreign streams/events before
copying. Copy and recording failures quarantine. Settlement checks the actual CUDA
free before releasing the reservation; a failed free returns the retained resource,
keeps every charge, and cannot be retried by dropping the refusal.

Deleted public paths: `Upload::prepare`, `Upload::stage`, caller-selected upload
scope, `Upload::buffer`, and `Lease::use_on`. The replacement is
`Lease<Event>::prepare_upload` followed by `submit`; preparation refusals carry the
original lease and source. Explicit synchronous readback goes through the lease.
The generic host retirement machine and its returned turn sweeps remain shared.

The real-driver regression is version-controlled at
`crates/moxie-executor/tests/driver_faults.rs`. Its test executable interposes CUDA
allocation/copy/record/free only in that process, forwarding normal calls to the
real driver. It checks that refusal precedes allocation, including oversized Vec
capacity, and injects lost and non-loss errors at the actual copy/record/free
branches on each visible UUID. Quarantines intentionally retain six 512-byte device
allocations and their sources per device until process teardown. These are tests,
not injected production switches or measurements of inference performance.

## Decisions

No new owner ruling or reference-document amendment. The takeover contract is in
task 0009. A source starts and ends as caller-owned input/output; its full Vec
capacity is charged while owned by the engine. A transient device allocation must
be freed before its reservation ends. Persistent ownership transfer remains the
next task; the source being returned is not a persistent engine cache.

Checked free has an unsafe ordering contract in `moxie-cuda`, discharged by the
executor's private upload ownership and observed completion. Other CUDA consumers
keep their existing conservative destructor. Retirement adds no explicit
context-wide synchronization; this is not a latency guarantee about the CUDA
allocator itself, and no performance claim is made.

## Remaining hypotheses and blockers

No allocator, persistent residency transfer, kernel-chain leases or multi-stream
fan-in is claimed by this slice. Device loss and ambiguous cleanup deliberately
withhold resources; process restart remains the recovery boundary. No quality,
model-throughput or actual-context execution result exists.

Negative experiments are preserved in the task: deleting admission, deleting copy
quarantine, and releasing before settlement each fail the real-driver regression.
The source-move doctest fails with E0382 without `compile_fail`, and compiles when
the retaining move is removed. An initial device-suite compile failed while adding
a test-interceptor liveness assertion (missing local `copies`); it was corrected
before the successful full device run. No test expectation or gate was weakened.

## Next task

One bounded basic allocator, owned by `moxie-memory`/`moxie-executor` with audited
CUDA primitives: suballocate an admitted envelope into live ranges while preserving
physical allocation charges across per-operation completion. Read document 02's
buffer generation/ownership contract, document 03 and R03/R07/R08/R11 before its
contract. Require host lifetime/reuse/cancellation tests and real-device consumers,
including allocator failure, all ranges leased, and transfer of a persistent
allocation between accountable owners. Do not extend this into model execution,
residency policy, plan lowering or performance tuning in the same task.
