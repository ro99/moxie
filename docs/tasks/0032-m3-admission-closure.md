# Task 0032 — complete the shared admission failure contract

Status: implemented and measured; independent review finding repaired and mutation-tested, 2026-09-16. Owner milestone acceptance remains separate.

## Identity and authority

Owner assignment: "Put M3 back on track. Do whatever codes changes are necessary
so we can close M3.1/M3.2/M3.3." This authorizes corrective implementation; no
numerical, quality, storage or acceptance requirement is waived. Root
`/home/rodrigo/Developer/moxie`, main, base `4ccb3b9`. Existing publication and
coordinator changes are preserved. Legacy at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
and checkpoint roots remain read-only. Read task 0029, specs 02/03/07/08/09,
request/ledger/arena implementations and allocation-failure harnesses.

## Bounded deliverable

Finish task 0029's admission/rejection/relocation/release guarantee in
`moxie-memory`, `moxie-executor`, affected consumers and tests. Restore the CUDA
test build. No new engine, allocator authority, precision profile or tolerance.
Allowed files: those crates, affected Rust callers, xtask mutation anchors and
these task/evidence records. No checkpoint conversion or performance claim.

## Contract before implementation

- Allocation failure returns typed capacity refusal; never a default/corrupt
  success. Every refusal preserves logical ledger/arena state.
- Reserve arena release capacity at allocation time: successful release must
  allocate nothing even under sustained pressure. Transfer prepares owned labels
  before changing either owner.
- Rejection reports, legal alternatives and host relocation use fallible
  collections and clones; no hidden Box/map allocation in the error path.
- Restore the ability to supply non-static labels through fallible APIs; keep
  zero-allocation static/owned construction where useful. No acceptance of the
  narrowing as an intentional limitation.
- The shared device-core abort requires a fallible ownership mechanism. Its
  design must retain device ranges/events and allow unique cleanup without a
  second allocation. No abort waiver; private mechanism and independent audit
  before acceptance. Record the owner-reserved design disposition separately.
- All temporaries unwind on refusal; device allocations remain owned until
  observed completion. Shared memory/executor own the change; BF16 and affine
  consumers both inherit it.

## Acceptance

1. Exhaustive host allocation-position sweeps for successful requests, malformed
   requests, rejection and host relocation; first non-firing call matches the
   unarmed outcome; no loop-bound exhaustion. Compare all ledger counters and
   arena state outside the failing window.
2. Sustained-allocation-refusal release succeeds after fragmentation; failed
   transfer retains the original handle/state. Dynamic label lifetime regression.
3. Full affine admission sweep through the actual call, including after core
   ownership allocation, on all three GPUs; each injected failure leaves no
   outstanding ledger charge or device allocation. First non-firing admission
   succeeds and has the same resource request/descriptor as baseline.
4. Affected memory/executor/driver tests, host/driver/xtask-CUDA compile/lint,
   architecture/spec checks and actual GPU suite pass. Required T0028/T0029
   mutations are re-anchored where needed and run against the final implementation.
5. Records distinguish measured scope and remaining owner acceptance. No claim
   of quantized MoE or checkpoint execution follows from admission closure.

## Result

Pending. Initial recovery restored an interrupted T0006 mutant from the driver's
parked original using `cargo xtask mutation-check --self-test`: 109/109 passed.
No mutation/test process was alive before restoration. The inherited mutant had
disabled the resume payload checksum comparison in `write/run.rs`; restoration
returned that file to HEAD without removing the publication test changes.


### Implementation and measured gates — 2026-09-16

Rejections are inline rather than boxed; alternative/relocation bookkeeping
and label copies use fallible allocations. Arena allocation reserves the space
needed for every outstanding return, so fragmented release allocates nothing.
Transfer prepares its label before changing ownership. `PlanRequest::new`
accepts borrowed dynamic text again; buffer/reserve constructors also offer
fallible borrowed-label entry points.

A private `Shared<T>` prototype allocates one refcounted entry through
`Vec::try_reserve`, preserving lease retention and unique cleanup. It has no
weak/raw/COW API and cannot cross threads. Construction precedes physical device
allocation; failed physical cleanup retains the same core in permanent quarantine. Owner design approval was
requested explicitly under task 0029's reserved unsafe-design decision;
independent audit and acceptance remain pending.

Measured on this working tree:

- Host allocation-refusal suite: 8 passed, including all 59 rejection/relocation
  positions and fragmented allocation-free release. Full memory suite passed.
- Actual affine admission: all 48 host allocation positions refused cleanly on
  each of GPU-97fe4889-4874-a378-198e-955d2e72c4a3,
  GPU-3032cfa3-19df-028f-5ebd-43314911e0b9 and
  GPU-81fe4578-59b2-37c4-421e-287cdac78704. The first non-firing call matched
  the unarmed descriptor/resource baseline. No live device allocation or ledger
  charge remained after an injected refusal. The old reconstructed prefix was
  removed.
- Workspace host and driver checks passed; combined CUDA/driver all-target
  clippy passed with warnings denied. Restoring this lane exposed a preexisting
  test-module ordering lint in xtask, repaired by moving the test module last.
- Shared-owner unit tests and device_arena, driver_faults, affine_linear_device,
  affine_linear_real_module and grouped_device consumers passed.
- Real `xtask-cuda test-gpu`: 45 passed, 0 failed, 0 skipped; both SM86 and SM120
  qualified for the exercised cases. No timing or model-output claim.
- Architecture: 79 negative and 21 positive fixtures, 13 rules passed. Spec
  check: 10 unchanged documents. Mutation self-test: 109/109 passed.

Logs during execution: `/tmp/moxie-0032-{host-sweeps,device-sweep,memory,clippy,
shared,consumers,gpu,mutation-self}.log`. Final mutation results and durable
handover remain to be recorded; these interim logs are not acceptance by the
owner. An accidental `mutation-check --help` invocation started baseline-only
work (the driver ignores that option); it was stopped, followed by self-test
recovery. No substitution result from that invocation is counted.


### Owner disposition and T0029 — 2026-09-16

Owner explicitly approved: **“Use the fallible shared handle.”** The reserved
unsafe-design choice from task 0029 is resolved in favor of the private handle,
subject to the requested review. This does not constitute acceptance of the
implementation or waive its independent audit.

`cargo xtask mutation-check --battery 0029`: **4/4 caught**, zero survivors,
unstable controls or skips; both baseline lanes passed three times before and
after, and every deciding lane passed its three mutation repetitions. Exit 0.
The existing anchors remained unique; no mutation was weakened or re-anchored.
T0028 is in progress, so its outcome is not included here.


### Independent review and repair

The read-only reviewer found no unsafe shared-ownership or production atomicity
fault. It found that the new rejection sweep compared only the flattened error
kind; an ordinary rejection has that same kind and could hide a swallowed
allocator failure. The sweep now requires the exact `AdmitError::Invalid` value
with pageable-host attribution. It covers host capacities 1,048,576 / 512 / 129
bytes (one byte reserved headroom): **59 / 59 / 47 positions**, respectively,
including full movement, a split and no room. All eight host tests pass.

Added `rejection-swallows-an-alternative-allocation-failure` to T0029. The final
battery caught **5/5**, including that reproduction; zero survivors, instability
or skips. Self-test is now **110/110** (95 anchors). T0028 completed **23/23
caught**, **1/1 expected control held**, zero survivors, unstable/invalid/broken
controls or skips. Both batteries repeated clean baselines before/after and
mutant deciding lanes three times. T0028's run preceded the review's test-only
strengthening and arena comment correction; its production implementation is
unchanged. The new T0029 run measures the strengthened sweep.

A failed physical free is permanently quarantined, not retried; corrected the
comment/prose that suggested otherwise. The reviewer proposed constrained-host
coverage; those two additional cases are included above. A follow-up importer
audit failed at the agent usage limit; it produced no importer evidence and is
not counted as a review.

Remaining M3 work: task0031's full publication battery, importer continuations,
and shared quantized-expert execution and graph consumers. No acceptance of
those follows from this admission repair.
