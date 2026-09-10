# Task 0011 — M1.3: admitted graph/resource plan

Status: **review corrections complete; independent re-review pending**, 2026-09-10, after task
0010's basic device arena was accepted by the owner at `0cc61c7` following independent re-review.

**This contract is committed before implementation code.** Its acceptance criteria are fixed before
tests run. Later corrections and deviations belong in the Result section; they do not rewrite the
gate that was attempted.

## Identity and authority

- Task ID / milestone / owner: 0011 / M1.3 / implementation agent; independent review required
  before acceptance.
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `9e14eeb`, initially
  clean and equal to `origin/main`. Local commits are allowed by the continuing assignment; no push
  is authorized.
- Read-only legacy root: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its existing untracked `.pi/` and `tests/p2p/`
  paths remain untouched.
- Requirements repaired: document 02's pure-plan / atomic-admission boundary, graph identity,
  tensor-handle bounds/layout/generation contract, and rejection before execution; document 03's
  peak-live-set accounting and one real memory authority; document 04's row/position distinction
  and device-resident intermediate requirement; M1.3's admitted execution-plan prerequisite; R03's
  model-local reserve/cache decisions; R07's retained asynchronous operands; R08's no-next-token
  cleanup; R11's bounded arenas; and R14's requirement that consumers inherit the arena rather than
  optionally bypass it.
- Required living records: tasks 0003, 0004 and 0006–0010, their accepted correction handovers,
  ADRs 0003/0006, the owner-gate register, and the support matrix.
- Actual legacy sources read at the frozen commit: the five R03 cache implementations;
  `include/strata/device/cuda_backend.hpp`'s arena/deferred-upload boundary;
  `src/models/inkling/inkling_checkpoint.cpp` mapped-source lifetime;
  experiments 0125/0175/0176 and their source tests; R11's `CheckpointShardSet`, Kimi host arena and
  tests. Their lesson is ownership and boundedness, not an API to copy.
- Owner gates: **none needed**. This task uses synthetic graphs and bytes, reads no checkpoint,
  changes no public product surface, makes no quality or performance claim, and changes no
  operational setting. Stop if O1–O7 becomes necessary.

## Bounded deliverable

- **One concrete outcome:** a validated, immutable graph and a concrete single-device row bucket
  lower to a pure resource plan whose logical tensors have checked shapes, byte sizes, liveness,
  physical slots and layouts. The executor atomically admits that plan, materializes its activation
  arena through task 0010, and returns a graph-bound `ReservedPlan`. Every possible refusal occurs
  before semantic execution and either returns the candidate or leaves an explicitly quarantined,
  still-charged resource.
- **Shared owners:** a new `moxie-plan` crate owns pure graph/resource lowering and plan identity.
  `moxie-executor` owns admission and physical binding. `moxie-memory` remains the only byte
  authority and arena allocator. `moxie-graph` owns semantic tensor declarations and immutable
  graph identity. `moxie-types` owns only descriptor identities/layout vocabulary needed below both
  plan and executor.
- **Allowed production files:** `crates/moxie-plan/**` (new); narrowly scoped graph identity and
  layout/type additions in `moxie-graph` and `moxie-types`; plan binding in `moxie-executor/**`;
  manifests/lockfile; `xtask/src/archcheck.rs` and architecture fixtures for the new declared crate
  edges; the task, support matrix and handover. `moxie-memory` production changes are allowed only
  for a missing read-only reservation query demonstrably required by binding; admission arithmetic
  and arena policy are not reimplemented.
- **Non-goals:** no model/checkpoint, storage read, weight upload or residency policy; no paged state;
  no semantic CUDA kernel, module load or launch; no operation-capability selection; no sampler,
  generation loop, prefill/decode execution, multi-device plan, collective, multi-stream fan-in,
  CUDA graph capture, eviction, benchmark or performance assertion.
- **Forbidden shortcuts:** no caller-supplied raw pointer, allocation generation, byte count or
  graph ID that can disagree with the graph; no model-name dispatch; no device ordinal as identity;
  no `Drop` release; no sum of all logical intermediates when non-overlapping liveness permits reuse;
  no physical slot shared by overlapping values; no uncharged alignment/fragmentation; no caller
  boolean claiming a kernel exists; and no method named `execute` while qualified kernels are absent.
- **Consumers:** two synthetic stateless dense graphs with different row/hidden widths and different
  liveness shapes lower through the same path. One has a fan-out/fan-in residual that keeps an early
  value live; the other permits a slot to be reused. The real-device binding test runs the same
  resource-plan machinery on every visible UUID. These are plan consumers, not model support.
- **Deletion/expiry:** no old plan implementation exists. Task 0010's conservative whole-reservation
  upload remains until an execution integration replaces it. Test-only host bindings and synthetic
  graph builders stay test-only. The explicit "execution unavailable without qualified kernels"
  boundary expires in the next device-layer-chain task, which must add real capability selection and
  execution rather than weaken this plan's validation.

## Contract before implementation

### Immutable graph identity

`GraphBuilder::finish` assigns a nonzero process-unique `GraphId` after every existing structural,
shape, precision, oracle and state check succeeds. Failed construction consumes no identity.
Identity allocation uses checked monotonic arithmetic and refuses exhaustion; wrapping may not make
an old graph current again. `Graph::clone` denotes the same immutable graph and preserves identity.
There is no public constructor for a chosen ID.

The resource plan retains the graph ID and a structural binding summary: value count, node count,
row symbol, output value and ordered node/value descriptors. Admission accepts the graph itself and
checks the summary again. Matching a bare ID is insufficient evidence if memory corruption or a
future deserializer supplies contradictory structure.

### Workload bucket and supported boundary

The pure input is:

```text
ResourceWorkload {
  phase: Prefill | Decode | Verify | Entropy,
  rows: nonzero u64,
  visible_tokens: nonzero u64,
  branch_rows: nonzero u64,
  output: graph output,
  device: DeviceUuid,
}
```

Rows bind the graph's row symbol. Visible tokens remain separate and may not be inferred from rows;
this is R19's distinction made structural. In this task `branch_rows` must equal `rows` for prefill
and decode, while verify/entropy are refused as unsupported because their state/resource contracts
are not implemented. A decode bucket requires `rows == 1`; prefill permits more than one. The
workload identifies a device by UUID only.

This slice lowers only stateless graphs whose existing operation contracts have
`StateEffect::None`. A graph containing attention or another state effect is refused with a typed
unsupported result before allocating or admitting anything. This is not a fallback and not a claim
that attention is optional; task 0004's transaction API and the later paged-state task must be bound
before such a graph can execute.

### Logical values, bytes and layouts

Every graph value receives one of three plan bindings:

```text
ExternalInput { value, concrete_shape, logical_role }
ExternalWeight { value, concrete_shape, logical_role, required_bytes }
ArenaTensor { value, concrete_shape, logical_role, layout, slot, byte_offset, bytes }
```

Input and weight extents are derived by evaluating the graph's `TensorSpec` with the workload's row
binding. No caller supplies an extent. Element widths are role-specific: BF16/F16 are two bytes,
F32 four bytes, and index widths remain an explicit descriptor rather than using a floating or
weight precision predicate. This task has no device index arena; per-step index inputs remain
external.

The only materialized layout in this slice is versioned contiguous logical row-major activation
layout. It is a closed descriptor, not an arbitrary integer. A prepared weight layout is never
invented here. Shape product, element bytes, alignment and offset addition use checked arithmetic;
zero extents and address-space overflow are typed refusals.

Node outputs are born at their producer stage and remain live through their last consumer. The graph
output remains live through a terminal-output stage. Values with no legal producer/consumer chain
are refused rather than assigned storage. Node order from the validated graph is the execution-stage
order; this task does not reorder mathematics.

### Deterministic physical slot plan

The pure lowerer assigns arena slots by deterministic address-ordered first fit over liveness
intervals. A slot may be reused only when the earlier value's last stage is strictly before the
later value's first stage. The slot's bytes and alignment are the maximum required by every value
assigned to it. Offsets are then assigned in slot-identity order with checked alignment.

`activation_arena_bytes` includes every slot's padding and is the exact capacity later passed to
`DeviceArena`. It is also the exact `DeviceTier::Activations` charge placed in the resource envelope
for the complete plan lifetime. Logical value intervals remain in diagnostics but are not charged a
second time. The candidate separately charges each external weight's canonical byte requirement to
`DeviceTier::PackedResidentWeights`; this reserves the full declared envelope but does not claim the
weights are resident or assign them a device pointer.

The candidate records zero kernel workspace because no kernel has been selected. It cannot later be
executed by interpreting zero as evidence that every kernel needs none. The next task must replace
that absence with registry-backed kernel choices and their workspace before adding execution.

### Admission and physical binding

`ReservedPlan::admit(candidate, graph, ledger, rank_context)` performs, in order:

1. validate graph summary and UUID against the candidate and rank context;
2. atomically admit the candidate's complete `PlanRequest` through the existing ledger;
3. create exactly one task-0010 `DeviceArena` for the activation charge;
4. allocate one range per physical slot with the planned bytes/alignment and a plan/value owner;
5. construct read-only tensor handles from the returned range identity, generation, bounds, closed
   layout and logical shape.

No semantic operation can run between these steps. A tensor handle exposes metadata and a bounded
range borrow, never a raw pointer. Two values assigned to the same slot have distinct logical
handles but the same physical allocation identity; the plan's stage/liveness check is what prevents
simultaneous use. Overlapping values must have distinct allocation identities.

Validation failure before ledger admission returns the unchanged candidate and makes zero CUDA
allocation calls. Ledger rejection returns the candidate plus the existing full admission report.
Arena creation failure releases the returned reservation against its ledger before returning unless
the driver outcome is ambiguous; an ambiguous failure stays charged and is reported as quarantined.
Slot-allocation failure releases every earlier range, closes the arena (checked free before ledger
release), and returns the candidate. Cleanup refusal returns a quarantined plan resource and never
claims atomic rollback.

`ReservedPlan::close` consumes the plan, releases every range once, then closes the arena. Wrong
ledger, live borrowed use, failed range release or failed CUDA free returns the complete still-owned
plan/quarantine and performs no later release. `Drop` withholds the arena/reservation, as tasks 0009
and 0010 require.

### Refusal before execution

The admitted resource plan has no `execute` method. `validate_execution_request` may check only that
a request names the same graph, workload bucket, device, selected output and concrete input shapes.
It always returns `UnsupportedKernel` for semantic execution until a subsequent task binds every
node to a qualified kernel and adds workspace to this same reservation. It may not run the host
interpreter as a hidden device fallback and may not dispatch the smoke AXPY probe as a semantic op.

Cancellation before execution closes normally because no event is in flight. The next layer-chain
task must use task 0009/0010 operation leases for every launched input/output/workspace and prove
cancelled in-flight work remains retained through event completion.

### Numerical, state and application effects

No model-value arithmetic occurs, so there is no floating-point tolerance. Exact checks cover shapes,
bytes, stages, identities, layouts, offsets, allocation generations and ledger charges. State is
unchanged because stateful graphs are refused. Sampling, token publication, API and CLI behavior are
unchanged. No synthetic graph result is a quality or model-support result.

## Acceptance

- **Pure host lowering:** exact concrete shapes/bytes for BF16 and F32 node outputs; two distinct
  shapes; fan-out liveness; non-overlapping reuse; no overlapping reuse; output lifetime; alignment
  padding; deterministic slot/offset assignment; row/visible-token separation; decode-row rule;
  unsupported verify/entropy/stateful graphs; zero/overflow/bad output/device cases; and graph clone
  identity versus different-graph refusal.
- **Graph identity bite checks:** IDs are nonzero and distinct for separately finished graphs; failed
  `finish` consumes no ID; a test-only near-exhaustion seam refuses rather than wraps; removing the
  structural-summary comparison makes a deterministic host test fail.
- **Admission/binding host checks:** exact `PlanRequest` stages, activation arena charge and weight
  charge; ledger rejection returns the unchanged candidate and zero outstanding reservation;
  mismatched graph/UUID/rank is refused before admission; successful metadata binding maps every
  arena tensor to its planned slot identity/generation/bounds/layout; execution validation fails
  with `UnsupportedKernel` and cannot call a backend.
- **Driver fault boundary:** interposed CUDA symbols prove validation/ledger refusal causes zero
  allocation calls; arena allocation failure returns the candidate with no ledger charge; slot
  failure unwinds prior ranges and physical allocation; failed final free retains the complete plan
  and charge and is not retried by `Drop`; successful close frees once before ledger release.
- **Real GPU, every visible UUID and both architectures:** lower the same synthetic graph for the
  measured device UUID, admit and bind it, verify planned value/slot identities and aliased versus
  non-aliased liveness, refuse a graph and UUID mismatch before another allocation, reject execution
  as unsupported, then close with driver free memory and ledger outstanding state reconciled. No
  kernel launches and no benchmark number.
- **Architecture:** add `moxie-plan` to the checked dependency graph with production dependencies only
  on `moxie-types` and `moxie-graph`. `moxie-executor` may consume it. Add rejecting fixtures proving
  `moxie-plan -> moxie-memory` and `moxie-plan -> moxie-cuda` are forbidden, and a model crate cannot
  reach `moxie-plan`. Existing 12 rules and all fixtures still pass.
- Run and report separately: format; host and full device-feature clippy with `-D warnings`; full
  locked/offline host and device-feature suites; focused graph/plan/executor tests; `arch-check`;
  `spec-check`; isolated no-toolkit/no-driver build, test and `ldd`; driver-fault tests; `test-gpu`
  on both architectures; restricted-visibility negative qualification; `capacity` in normal and
  reordered visibility. A GPU lane with no launch is still required because physical allocation,
  unwind and free changed.
- Update the support matrix with a narrowly worded graph/resource-plan row. It must state that
  semantic kernel execution, weights, state, generation and model support remain unavailable.
- Stop and report if this requires a second physical arena per reservation, storage or checkpoint
  access, a weight-residency owner, an actual semantic kernel/launch, state mutation, multi-stream or
  multi-device fan-in, model code, a numerical threshold, or a performance claim.

## Result, filled after work

- Changed shared owners and consumers; source commits: contract `96f20fe`; implementation
  `a95fa3a`; final forced slot-unwind evidence `1e87bb4`. `moxie-graph` now owns checked immutable
  graph identity and exact structural summaries. New pure crate `moxie-plan` derives concrete
  workload shapes, BF16/F32 bytes, liveness and deterministic physical slots. `moxie-executor`
  converts the candidate to the shared ledger envelope, materializes one task-0010 activation
  arena, returns bounded tensor metadata, and refuses semantic execution without kernels.
- Passed: format; host and full device-feature clippy with `-D warnings`; locked/offline host suite
  (547 unit/integration tests plus 8 doctests); locked/offline full device-feature suite (559 plus
  10 doctests); focused graph/plan/executor suites; architecture check (55 rejecting fixtures, 14
  accepted fixtures, 12 rules); and unchanged-reference check (10 documents).
- Passed device evidence: resource-plan admission/binding/close and forced slot-exhaustion unwind on
  every visible UUID; interposed allocation/free failures on every UUID; aggregate `test-gpu` 33/33
  across `sm_86` and `sm_120`; normal and `CUDA_VISIBLE_DEVICES=2,1,0` capacity probes. The
  restricted `CUDA_VISIBLE_DEVICES=1,2` qualification exited 1 as required, with `sm_120`
  explicitly unqualified. The isolated no-toolkit/no-driver suite passed 547 plus 8, and an
  explicitly rebuilt `xtask` had no `libcuda` dependency in `ldd`.
- Failed during development, then corrected before the final gates: the first host clippy run found
  one unused feature-gated import and one index loop; the first device clippy run found oversized
  inline refusal variants; and the first forced-unwind test compile needed a non-`Debug` assertion
  rewritten. No final validation command failed and no acceptance threshold changed.
- Measured effect and uncertainty: each real-device plan test allocated and freed one small
  activation arena, observed aliasing only for non-overlapping values, rejected graph/UUID mismatch
  before another allocation, and reconciled driver free memory with an empty ledger. This is exact
  resource/lifetime evidence only. Model quality, actual context, topology communication, semantic
  CUDA arithmetic and paired prefill/decode performance were skipped as outside this task and
  remain unmeasured. No checkpoint was accessed or transformed.
- Deleted/replaced paths: the unused freely constructible `moxie-types::GraphId` was replaced by
  `moxie-graph::GraphId`, which only successful graph construction can issue. No execution,
  allocator, cache or legacy path was deleted; the legacy checkout was unchanged.
- Remaining blockers and next bounded task: independent review is required before acceptance.
  After acceptance, write a separate M1.3 task for registry-backed semantic capability selection,
  exact workspace inclusion in this reservation, and one real device layer chain using operation
  leases. Stop before model/checkpoint loading, residency/eviction policy, paged state, generation,
  multi-device execution, quality claims or performance claims.

## Independent-review corrections — 2026-09-10

- Source commit `78d90be` closes all three contract violations and the requested coverage gap.
  Lowering now rejects a weight-role value declared as a per-step graph input, validates checked
  element products for every external operand, and validates the role-specific byte extent for
  external activations. Index inputs validate their element extent without inventing a byte width
  absent from the graph contract. A regression reproduces both the disguised BF16 weight and the
  overflowing live external operand; the executor's ledger test admits the valid counterpart and
  observes the exact 128-byte packed-weight charge.
- `TensorHandle` descriptor fields are private and exposed only through read-only accessors. The
  driver-feature doctest proves an external caller cannot rewrite a returned handle's byte bounds,
  and the real-device binding test reads every descriptor field through the public accessors while
  retaining the checked allocation identity.
- The slot-failure test now enters the same admission implementation as `ReservedPlan::admit`,
  injects failure on the second planned slot, and checks the returned unchanged candidate, typed
  capacity error, empty ledger and exact device-memory reconciliation on all three UUIDs. The
  public method is a thin call into this tested implementation with the ordinary arena allocator.
- Final validation passed: format; host and full device-feature clippy with `-D warnings`; the
  locked/offline host workspace (548 unit/integration tests plus 8 doctests); the locked/offline
  full device-feature workspace (560 plus 11 doctests); focused plan and executor suites;
  architecture check (55 rejecting fixtures, 14 accepted fixtures, 12 rules); specification check
  (10 unchanged reference documents); GPU qualification (33/33 across `sm_86` and `sm_120`);
  normal and reordered capacity; and the restricted two-3090 negative qualification, which exited
  1 with `sm_120` explicitly unqualified. A fresh no-toolkit/no-driver target passed 548 plus 8,
  and its explicitly rebuilt `xtask` had no `libcuda` dependency in `ldd`.
- Failed during correction, then fixed before the final runs: the first injected allocation used
  `u64::MAX`, which correctly failed alignment validation before reaching the intended capacity
  branch; a finite over-capacity request now exercises that branch. An earlier driver attempt in a
  restricted execution environment could not see CUDA; the fully authorized rerun exercised all
  three GPUs. No acceptance threshold or test was weakened.
- Skipped and unmeasured remain unchanged: no checkpoint or model was accessed; semantic CUDA
  arithmetic, actual-context inference, state, quality, topology communication and paired
  prefill/decode performance remain outside this task. Independent re-review is still required
  before task acceptance.
