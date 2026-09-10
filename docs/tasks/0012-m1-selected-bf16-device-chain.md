# Task 0012 — M1.3: selected BF16 device layer chain

Status: **contract proposed**, 2026-09-10, after the owner accepted
[task 0011](0011-m1-admitted-graph-resource-plan.md) at `64899c2` following independent re-review.

**This contract is committed before implementation code.** Its numerical and resource gates are
fixed before a semantic kernel is written or run. Later corrections and deviations belong in the
Result section; they do not rewrite the gate that was attempted.

## Identity and authority

- Task ID / milestone / owner-reviewer: 0012 / M1.3 / implementation agent; independent review is
  required before acceptance.
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `4ed80be`, initially
  clean and eight commits ahead of `origin/main`. The owner authorized committing and pushing this
  contract together with task 0011's acceptance record. No implementation has begun in this
  documentation assignment.
- Read-only legacy root: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its existing untracked `.pi/` and `tests/p2p/`
  paths remain untouched.
- Requirements advanced: document 02's shape/layout/hardware kernel dispatch, complete plan
  envelope and event-retained launch operands; document 03's separately charged kernel workspace,
  BF16/FP32 accumulation boundary and qualified NVIDIA matrix; document 04's device-resident
  intermediates; document 07's independent per-primitive numerical gates; roadmap M1.3's first
  device-resident layer chain; R05's host-span/device-round-trip boundary; R07's asynchronous
  source and operand lifetime; R08's no-next-token cleanup; and R14's requirement that the admitted
  allocator be used by construction.
- Required living records: tasks 0003, 0006 and 0009–0011 and their final correction/acceptance
  records; ADRs 0001, 0003 and 0006; the owner-gate register; support matrix; hardware, toolchain
  and topology evidence.
- Actual frozen legacy sources read before this contract: R05's
  `kernels/cuda/detail/backend_matmul.inc.cuh:476`, `:552` and `:912` host staging, BF16 routes and
  output rounding; `kernels/cuda/detail/backend_kernels.cuh:701`, `:946` and `:1533` BF16 matvec,
  RMSNorm and residual kernels; `include/strata/device/cuda_backend.hpp:309` and `:1183`
  device-buffer/host-span and deferred-upload boundaries; and `tests/test_cuda_backend.cpp:1150`
  and `:1175` bounded workspace and plain-BF16 matmul cases. They are numerical, launch and
  lifetime references, not an execution API to wrap. Current independent oracles are
  `moxie-oracles::{linear,norm,residual,metric}` and the graph-walking consumer is `moxie-interp`.
- Owner gates: **none needed**. The task uses small synthetic BF16 tensors, performs no checkpoint
  read or conversion, changes no public application surface and makes no quality or performance
  claim. Stop if O1–O7 becomes necessary.

## Bounded deliverable

- **One concrete outcome:** two synthetic stateless BF16 graphs lower through a shared immutable
  kernel catalogue to fully admitted single-device plans. One plan-owned physical allocation holds
  exact tier-separated weight, activation and selected-kernel workspace regions. The executor
  uploads checked operands, launches `Linear -> RmsNorm -> Residual` on one stream without an
  intermediate D2H transfer or synchronization, records one completion event, and returns the
  final BF16 output only after that event. The event-retained operation lease owns every range and
  host source that the chain may still touch.
- **Shared owners:** `moxie-types` owns only closed kernel/layout identity vocabulary shared below
  planning and execution. `moxie-kernels` owns the immutable semantic kernel descriptors, fatbin,
  symbols and CUDA implementations. `moxie-plan` owns pure deterministic selection and exact
  weight/activation/workspace lowering. `moxie-executor` owns physical binding, checked launches,
  completion and cleanup. `moxie-memory` remains the sole byte/admission and range-metadata
  authority; `moxie-cuda` remains the typed driver boundary.
- **Allowed production files:** narrow descriptor/layout additions in `moxie-types`; selection and
  resource-plan extensions in `moxie-plan`; semantic descriptors, build integration and CUDA
  source in `moxie-kernels`; typed asynchronous stream launch support in `moxie-cuda`; plan
  binding/execution/lifetime support in `moxie-executor`; manifests/lockfile; `xtask` GPU and
  architecture checks/fixtures; this task, its handover and the support matrix. `moxie-memory`
  production changes are allowed only for a missing read-only reservation/range query or the
  minimum tier-region validation needed by the one existing arena authority. Oracle/interpreter
  production equations do not change.
- **Non-goals:** no model or checkpoint; no canonical-reader integration; no reusable weight
  cache, chunk identity, eviction, demand/prefetch, prepared layout or cross-plan residency; no
  attention, paged/recurrent state, generation, sampler, service or CLI; no INT4/INT8; no fusion,
  tensor-core optimization, CUDA graph, multiple streams, multi-device work or collective; no
  benchmark, throughput/default-selection or model-quality claim.
- **Forbidden shortcuts:** no model-name dispatch, caller-set availability boolean, arbitrary
  symbol/function pointer, caller-supplied raw device pointer, unvalidated byte count, allocation
  outside admission, hidden `cudaMalloc` in a kernel path, intermediate host fallback, blocking
  launch between graph nodes, release through `Drop`, or reuse after cancellation before observed
  completion. Compiling an SM image is not qualification. A host-interpreter fallback must remain
  explicit and cannot make a required device plan pass.
- **Consumers:** a decode bucket with `rows=1, hidden=8` and a prefill bucket with
  `rows=5, hidden=17` use the same three semantic kernel implementations. The odd hidden width
  exercises tails and the two row counts change the RMS workspace extent. A constructed
  representable fixture and a cancelling/general fixture are independent numerical consumers.
  Every visible GPU UUID executes both graph shapes. These are synthetic operation consumers, not
  model or context support.
- **Deletion/expiry:** task 0011's unconditional `UnsupportedKernel` execution result is replaced
  only for fully selected and bound graphs; every other graph still fails closed. Smoke kernels
  remain toolchain probes. Plan-bound immutable BF16 weights are a deliberately simple all-resident
  bridge for this small graph and must be absorbed by the shared chunk/residency lifecycle in M2;
  it may not grow cache lookup, eviction or artifact policy. No legacy path is deleted.

## Contract before implementation

### Closed semantic kernel catalogue and pure selection

A kernel descriptor is immutable data, not a callback. Its identity includes a stable kernel ID
and ABI version, semantic operation, operand roles and precisions, accumulation/rounding profile,
accepted physical layouts, checked shape constraints, hardware capability, exact workspace
expression, fatbin content hash and ordered symbol set. A descriptor cannot mention a model,
checkpoint or device ordinal. Qualification evidence identifies `sm_86` and `sm_120` separately;
the device is selected by UUID and its measured capability supplies the SM key.

The only accepted descriptors in this task are:

| Semantic op | Required profile/layout | Shape boundary | Workspace |
|---|---|---|---|
| `Linear` | BF16 input/weight/output, FP32 accumulation, one final BF16 RNE, contiguous row-major, no bias | `1 <= rows <= 64`, `1 <= K,O <= 1024`, checked products/address extents | zero |
| `RmsNorm` | BF16 input/gain/output, FP32 sequential reduction, explicit positive finite epsilon, one final BF16 RNE, contiguous row-major | `1 <= rows <= 64`, `1 <= H <= 1024`, checked products/address extents | exactly `rows * sizeof(f32)` logical bytes for row sums, in one 256-byte-aligned slot |
| `Residual` | two BF16 inputs and BF16 output, FP32 add, one final BF16 RNE, contiguous row-major | nonzero equal shapes within the admitted address extent | zero |

`moxie-plan::lower` consumes a read-only catalogue snapshot plus the graph, workload and measured
device capability. It checks every node and selects by semantic op, roles, precision,
accumulation, layout, concrete shape and SM. Selection is deterministic. Duplicate indistinguishable
descriptors are an invalid catalogue; an unsupported node/shape/SM returns `UnsupportedKernel`
before a `PlanId`, ledger request, module load or allocation exists. No best-effort partial plan is
emitted.

The candidate retains the selected descriptor identity for every node and a catalogue digest.
Graph identity/signature, workload, device UUID/capability, kernel IDs/ABI, image hash, layouts and
workspace expressions all participate in plan identity and execution validation. Changing any of
them makes an old candidate inadmissible rather than silently dispatching a new implementation.
`moxie-plan` remains pure and does not import `moxie-kernels`, `moxie-memory` or `moxie-cuda`; the
catalogue is injected as closed descriptors.

Before the first semantic launch, the executor loads the selected trusted image and resolves the
complete ordered symbol set for all three nodes. A missing image/symbol or ABI mismatch returns the
still-owned plan and records zero semantic launches. Catalogue membership is evidence of a
candidate implementation; only the required real-GPU gates below qualify it.

### Graph, bindings and semantic order

Both consumers build this exact unfused graph with `H` equal to 8 or 17:

```text
x: [rows,H] BF16                 W: [H,H] BF16
              \                 /
               Linear(bias=false) -> h: [rows,H] BF16
                                      |
gain: [H] BF16 ---------------- RmsNorm(eps) -> n: [rows,H] BF16
x ---------------------------------------------------\
                                                      Residual -> y: [rows,H] BF16
```

The representable graph uses values whose linear partial sums, RMS denominator and residual are
exactly representable and uses the already established `eps=3.5` RMS construction where
applicable. The general graph uses `eps=1e-5`, mixed signs, cancellation and non-power-of-two
tails. Node order is graph order. No fusion or reordering is permitted in this task, because each
node's BF16 rounding is a semantic boundary. `x` remains live through the residual; the planner
must not reuse its range for `h` or `n`.

The decode workload has `rows=branch_rows=1`. The prefill workload has
`rows=branch_rows=5`; a test sets `visible_tokens=32_768` solely to prove that visible history does
not alter stateless shape or selection. It is not an actual-context or long-context claim. Verify,
entropy and every stateful graph remain unsupported.

### Exact admitted bytes and one physical arena

Pure lowering assigns each external input, immutable weight, activation and workspace an exact
logical extent, tier, stage span, alignment and physical range. External BF16 bytes are derived
from graph roles/shapes and remain caller-independent. Weight ranges live for the full plan and are
never aliased. Per-step inputs begin before node zero and live through their last consumer. Node
outputs keep task 0011's strict liveness rule. Workspace lives from RMS reduction launch through
RMS apply completion and cannot overlap an activation that is live then.

The plan has three separately diagnosed, 256-byte-aligned physical regions:

- BF16 `W` and `gain`, charged once to `DeviceTier::PackedResidentWeights`;
- external `x` plus graph activations, charged once to `DeviceTier::Activations` using the exact
  physical-slot peak rather than the sum of reusable logical values;
- the selected RMS row-sum slot, charged once to `DeviceTier::KernelWorkspace`.

Every region includes its internal alignment padding. Region sizes are multiples of 256, so the
combined physical size is their checked sum with no uncharged cross-region gap. The ledger request
contains the three tier charges and their real stage spans; device-scope peak equals the bytes that
overlap. Admission occurs once, then exactly one existing plan-owned `DeviceArena` allocation is
created for the combined physical size and partitioned according to the candidate. There is no
per-tier or per-node CUDA allocation and no second ledger reservation. A forced allocation or
range failure unwinds as task 0011 requires; ambiguous cleanup remains charged and quarantined.

The RMS descriptor reports `rows * 4` logical workspace bytes with checked multiplication. Its
physical slot is `align_up(rows * 4, 256)`, so both values are visible in diagnostics and exact tests.
Replacing it with zero, charging it as an activation, omitting padding, summing mutually exclusive
workspace, or allocating it after admission must make a deterministic test fail. Unsupported or
overflowing workspace refuses during pure lowering.

### Checked binding, launch and event-retained ownership

An external binding names a plan value and supplies owned bytes only. The plan derives the expected
role, shape, encoding, extent, range, UUID and layout. BF16 inputs/weights are little-endian,
exact-length and finite when decoded. A weight range becomes immutable after its first successful
binding. Rebinding an input is allowed only when no execution lease is in flight. Wrong/missing/
duplicate values, mutable weights, foreign UUIDs and mismatched shapes/layouts fail before a copy or
launch.

Uploads, three semantic nodes and the final completion event use one rank-owned stream. Upload
sources are moved into the same retained execution resource and stream ordering makes them ready
before their first consumer. `Linear` writes `h`; `RmsNorm` reads `h` and its admitted row-sum
workspace and writes `n`; `Residual` reads the still-live `x` and `n` and writes `y`. No graph node
causes D2H, stream synchronization, allocation or module lookup. After all launches enqueue, the
executor records one event and returns an in-flight plan operation lease built on task 0009/0010's
existing lifecycle, not a second completion state machine.

The in-flight lease owns the complete reserved plan, all arena ranges, uploaded host sources and
the event until completion is observed. It is impossible to close the plan, mutate/rebind an input,
reuse a workspace/output slot or free the arena while that lease exists. Cancellation changes intent
but releases nothing early. A turn-boundary sweep with no next token returns the whole held lease
before completion and the whole ready plan/sources after completion. `Drop` withholds in-flight or
lost resources.

Validation, catalogue and binding failures return the unchanged candidate/plan with no semantic
launch. A launch failure after an earlier launch may have executed work; it returns a held plan and
does not launch later nodes. Event-record failure is ambiguous and marks the plan lost. Event query,
synchronization, final D2H or physical-free failure keeps ownership and charges visible. CUDA
asynchronous failures are attributed to this chain/event with the selected node/kernel IDs. A
successful completion permits one bounded final-output readback, returns the ready plan and retained
sources, and then permits explicit close: ranges first, checked physical free second, ledger release
last.

### Independent numerical contract fixed before kernels

Inputs to every comparison are finite BF16 values decoded exactly to FP32. Let `u32 = 2^-24`,
`gamma(n) = n*u32/(1-n*u32)`, `ub = 2^-8` (BF16 unit roundoff), and
`eta_b = 2^-134` (half the smallest positive BF16 subnormal). Independent tests transcribe each
equation in FP64 and apply the graph's BF16 round-to-nearest-even only at the declared node output.
They do not call the CUDA implementation or reuse its reduction.

For each decoded device output `got` and FP64 equation value `want`, the predeclared absolute bounds
before the next node are:

```text
Linear:   |got-want| <= gamma(H+1) * sum_k |x_k * W_ok| + ub*|want| + eta_b
RmsNorm:  |got-want| <= (gamma(H+4) + ub) * |want| + eta_b
Residual: |got-want| <= gamma(1) * (|a| + |b|) + ub*|want| + eta_b
```

Linear accumulation is sequential ascending `k` with separate correctly rounded FP32 multiply and
add; FMA/reassociation and intermediate BF16 rounding are outside this descriptor. RMS sum of
squares is sequential ascending hidden index in FP32, stored once per row in the admitted FP32
workspace, then divided, square-rooted, scaled and rounded once to BF16. Residual adds once in FP32
and rounds once. Zero-bound cases require exact equality. Every operation reports count, max, RMS
and p99 absolute error plus error normalized by the element's bound; normalized max must be at most
1.0.

The constructed representable graph must match the interpreter bit-for-bit at every node and at
the final output. The general integrated graph compares final BF16 outputs to the interpreter with
maximum BF16 ULP distance at most one and reports max/RMS/p99 ULP distance; the per-node bound tests
remain the authority for accepting any difference. Fixtures cover linear cancellation and near-zero
results, RMS zero/small/mixed magnitudes, residual cancellation, odd width 17, rows 1 and 5, and the
largest admitted dimensions through bounded synthetic allocation. A nonfinite external value is
refused before launch; any nonfinite final output is a typed `Numerical` failure after safe event
completion and does not become a successful result.

This is execution-delta evidence for three synthetic BF16 operations only. It is not checkpoint,
conversion, model-quality or throughput evidence. A failing bound cannot be loosened after a run;
the kernel or its descriptor must be corrected or the task rejected.

### State, application and observability effects

The graph is stateless and executes outside `SequenceState`; transaction counters, KV, sampler
history and token publication do not change. No generation/API/CLI surface is added. Diagnostics
report graph/plan/catalogue identities, selected kernel IDs, device UUID/SM, exact logical and
physical bytes by tier, node launch order, completion/cancellation state and final numerical
summary. Timing may be printed only as a diagnostic and cannot be cited as performance.

## Acceptance

- **Pure host selection:** exact selection for both graphs on synthetic SM86 and SM120
  capabilities; deterministic catalogue digest and node order; separate kernel identity for every
  node; refusals for missing/duplicate descriptor, bias, dtype/role/layout/shape/SM mismatch,
  zero/overflow workspace, altered graph signature, catalogue digest and UUID. Removing any one
  selection-key comparison makes a deterministic test fail.
- **Exact resource plan:** byte-for-byte expected `PlanRequest` and combined-arena layout for rows
  1/H8 and rows 5/H17, including BF16 input/weights, strict `x` lifetime, activation reuse only
  after last use, logical RMS workspace of 4 and 20 bytes, each with a 256-byte physical slot, all
  padding and exact three-tier/device-scope peaks. Ledger refusal makes zero allocation calls;
  allocation/range faults unwind or quarantine with the candidate and charge intact.
- **Binding and launch host/fault tests:** exact-length/finiteness/immutability and missing/foreign
  binding refusals; all symbols resolved before launch; zero semantic launches on preflight
  refusal; fixed Linear/RMS-reduce/RMS-apply/Residual launch order; no allocation after admission;
  launch, event-record/query/synchronize, readback and final-free failures return the owned plan in
  the declared live/lost/quarantined state. Compile-fail checks prevent rewriting descriptors or
  extracting raw pointers.
- **Independent numerical gates:** both shapes and all edge classes meet the fixed per-operation
  equations above with max/RMS/p99 reports; the constructed graph is bit-identical; the general
  graph is within one BF16 ULP at final output and every node remains inside its equation-derived
  bound. Reversed/reassociated linear accumulation, omitted BF16 boundaries, LayerNorm substitution,
  residual-twice and zero workspace are biting negative fixtures.
- **Real GPU on every visible UUID and both architectures:** both graphs select, admit, bind and
  execute on RTX 3090 UUIDs `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and
  `GPU-81fe4578-59b2-37c4-421e-287cdac78704`, and RTX 5060 Ti UUID
  `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`. Driver-call census proves exactly one plan payload
  allocation, no intermediate D2H/synchronization, one final event and one final-output D2H. A
  second execution reuses immutable weights without another weight upload. Final close reconciles
  free device memory and an empty ledger.
- **In-flight cancellation on every UUID:** a test-only CUDA stream gate holds the real chain before
  its completion event. Cancellation and the first no-next-token sweep retain the exact plan,
  operands, workspace and sources; after releasing the gate, synchronization and the second sweep
  return them. The watchdog is ten seconds and timeout fails closed.
- **CUDA validation:** aggregate and per-architecture GPU qualification pass; restricted visibility
  fails when a required architecture is absent. Run Compute Sanitizer memcheck on the reduced H8
  and H17 chain on at least one SM86 and the SM120 device; report racecheck/initcheck separately if
  unsupported or skipped. No GPU lane may convert zero visible devices or zero launches into pass.
- **Architecture:** `moxie-plan` keeps production dependencies only on `moxie-types` and
  `moxie-graph`; add/retain rejecting fixtures for planner-to-kernels/CUDA/memory, model-to-plan/
  kernels/executor, kernels-to-plan/executor/model, and shared model-name dispatch. If
  `moxie-executor -> moxie-kernels` is added for the selected package, record it as the sole new
  allowed production edge and add a clean accepted fixture. Existing rules and fixtures pass.
- Run and report separately: `cargo fmt --all -- --check`; `git diff --check`; host and full
  device-feature workspace clippy with `-D warnings`; locked/offline host and device-feature
  suites; focused types/graph/plan/kernels/executor/oracle/interpreter tests; architecture and
  specification checks; isolated no-toolkit/no-driver build/test/`ldd`; driver-fault suites;
  aggregate/per-architecture/restricted-visibility GPU qualification; normal and reordered
  capacity probes; and the sanitizer cases above. Passed, failed, skipped and unmeasured lanes are
  separate.
- Update the support matrix with a narrowly worded `G-BF16-DEVICE-CHAIN` gate only after all
  required host and real-device checks pass. It must say exactly which synthetic operations/shapes
  are qualified and that checkpoint/model execution, attention/state, actual context, generation,
  quality and performance remain unavailable or unmeasured.
- Stop and report if completion requires a checkpoint/model, storage read, a reusable residency or
  eviction owner, a second allocator/admission/completion mechanism, state mutation, attention,
  multi-stream or multi-device fan-in, INT4/INT8, a numerical threshold change, performance claim,
  operational setting change or owner decision. The smallest next task after acceptance is M1.4's
  appendable paged-state allocation bound to the existing transaction mechanism. Generation and
  model integration remain later bounded tasks.

## Result, filled after work

- Changed shared owners and consumers; source commit:
- Commands and result IDs; passed / failed / skipped separately:
- Measured effect and uncertainty:
- Deleted/replaced paths:
- Remaining blockers and next bounded task:
