# Task 0012 — M1.3: selected BF16 device layer chain

Status: **implementation complete at `6305f9d`; owner-review corrections at `eaf8846` and `1138a2a`;
awaiting owner re-review/acceptance**, 2026-09-10, after the owner accepted [task
0011](0011-m1-admitted-graph-resource-plan.md) at `64899c2` following independent re-review.

**Milestone closure assignment.** This is the final planned M1.3 task. The next implementation
agent working on Moxie owns task 0012 through implementation, required evidence, independent review
corrections and acceptance. Review corrections stay in this task; do not create task 0013 or
another preparatory M1.3 slice for work already required below. Once this task is accepted, record
M1.3 complete and make the next bounded implementation assignment M1.4 appendable paged state bound
to the accepted transaction mechanism in
[task 0004](0004-m1-state-transactions.md). Only a demonstrated defect in already accepted M1.3
scope or a defined stop condition can interrupt that handoff.

**This contract is committed before implementation code.** Its numerical and resource gates are
fixed before a semantic kernel is written or run. Later corrections and deviations belong in the
Result section; they do not rewrite the gate that was attempted.

## Identity and authority

- Task ID / milestone / owner-reviewer: 0012 / M1.3 / implementation agent; independent review is
  required before acceptance.
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`. Contract authoring began from
  `4ed80be`; task 0011 acceptance and this contract were committed and pushed at `9ab7a86`. No
  implementation had begun when the contract was committed. The implementation agent must confirm
  the current branch, writable root and dirty state before work.
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

Passing every required gate above closes M1.3. Do not add another M1.3 task merely to package,
refactor or re-review this implementation: include necessary fixes in task 0012 and repeat the
affected gates. After acceptance, update this status, `AGENTS.md`, the root and task indexes, the
support matrix and the active handover to say **M1.3 complete; M1.4 active**, then define the
appendable paged-state assignment. M1.4's later sampler, service and CLI work may remain separately
bounded inside M1.4.

## Result, filled after work

### Second owner-review corrections — `1138a2a`

The two additional findings stay within task 0012. M1.3 remains open pending owner re-review and
acceptance. The original three corrections at `eaf8846` remain in place.

- Fatal final-readback errors now persist their attributed `DeviceLost` status through the existing
  lifecycle before returning the lease. Shared synchronization refuses an already lost lifecycle
  before consulting its completion source. On all three UUIDs, injected CUDA status 700 at readback
  leaves the lease `Lost`; retry performs no further synchronization/readback, retirement refuses,
  turn sweeping returns the held lease, and dropping the held report neither frees the allocation
  nor releases its reservation. The existing nonfatal status-1 retry still succeeds and closes.
- RMS retains the declared sequential FP32 arithmetic and fixed numerical bounds. It explicitly
  refuses unqualified underflow: a nonzero square, mean, scaling product or normalization result
  that rounds to zero or a FP32 subnormal propagates a numerical marker to final readback. Exact
  zero operands remain valid. This conservative refusal can reject values whose eventual answer
  would happen to satisfy the bound; no universal finite-input RMS support is claimed.
- Both owner reproductions are now primitive and integrated regressions: input `2^-75`, gain 1,
  epsilon `2^-149`; and input `2^-80`, gain `2^-70`, epsilon `2^-126`. Each returns typed
  `Numerical` on all three UUIDs. A further primitive case (input `2^-60`, gain `2^-90`, epsilon
  `2^-126`) isolates scaling underflow while keeping the square reduction normal. Numerical
  refusal permits completed resource recovery and preserves the two immutable weight bindings.
- Passed: full locked/offline host workspace (556 tests plus 8 doctests) and device-feature
  workspace suite; focused driver fault harness;
  xtask tests; host and device workspace clippy with warnings denied; host doctests; formatting,
  architecture (60 rejecting / 15 accepted fixtures) and specification checks. Aggregate GPU
  qualification and both SM86/SM120 profiles each passed 39/39. Normal numerical maxima remain
  Linear `0.386110`, RMSNorm `0.960743`, Residual `0.996094`, with integrated H8/H17 ULP zero.
  Compute Sanitizer memcheck passed on SM120 and SM86, and SM120 racecheck/initcheck passed,
  including integrated underflow cases. No sanitizer lane was skipped.
- Negative evidence preserved: the owner's earlier probes measured normalized errors `57.49`
  and `255.95`, and fatal readback incorrectly reopened after clearing status 700. During correction,
  clippy found redundant slice borrows and an unguarded driver-only helper in the host build;
  both were corrected before the final passing checks. No numerical threshold changed.
- Not repeated this round: isolated no-toolkit/no-driver build, restricted-visibility failure and
  capacity reordering probes; their prior evidence remains recorded below. Model execution, actual
  context, quality and paired performance remain unavailable or unmeasured.

### Initial implementation and first owner-review corrections

- Changed shared owners and consumers; implementation commit: `6305f9d`; owner-review correction
  commit: `eaf8846`. `moxie-types` now owns the closed
  semantic-kernel descriptor/catalogue vocabulary. `moxie-kernels` owns an immutable SM86/SM120
  BF16 package and the Linear, RMS-reduce/apply and Residual CUDA symbols. `moxie-plan` owns exact
  topology validation, deterministic descriptor selection and weight/activation/workspace
  lowering. `moxie-executor` binds that selected candidate to one admitted partitioned arena and
  one event-retained operation. `moxie-cuda` resolves a complete symbol set before launch.
  `xtask` consumes both required graph shapes and the largest admitted primitive shape.
- Recorded scope deviation for owner review: `moxie-interp` was not named in the allowed-production
  file list, but the every-node interpreter gate could not be represented through its stateful
  public `run`. The implementation adds `run_stateless`/`StatelessTrace`, reusing the existing
  private dispatcher and oracle calls without changing an equation. A focused test proves graph
  order, every-node values and missing-binding refusal. This is an API/trace addition, not a second
  oracle. The owner review considered this stateless trace addition reasonable to accept; it is not
  one of the remaining acceptance blockers.
- Corrected all three defects reproduced by the owner review without changing a numerical gate.
  Invalid FP32 RMS intermediates now become a nonfinite device marker that survives the residual and
  is reported as a typed numerical failure after completion. The host request now reserves the
  final output for its real overlap with retained upload sources and explicitly hands the owned
  output buffer to the caller. Completed sweep recovery uses the same immutable-weight settlement
  as normal completion, returns only per-step inputs, rejects rebinding and supports input-only
  relaunch.
- Passed host/build evidence: format and diff checks; host and device-feature workspace clippy with
  warnings denied; a fresh locked/offline no-toolkit/no-driver all-target suite (556 unit/integration
  tests) plus focused interpreter doctests (3); the full locked/offline device-feature all-target
  suite (572 unit/integration tests); focused planner, interpreter, executor and xtask tests;
  architecture check (60 rejecting fixtures, 15 accepted fixtures, 12 rules); and specification
  check (10 unchanged reference documents). The fresh host `xtask` linked only `libgcc_s`, `libc`
  and the ELF loader in `ldd`, with no `libcuda`.
- Passed device evidence: `cargo xtask-cuda test-gpu`, `--profile sm86` and `--profile sm120` each
  reported 39 passed, 0 failed and 0 skipped/unmeasured across UUIDs
  `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`,
  `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and
  `GPU-81fe4578-59b2-37c4-421e-287cdac78704`; both SM architectures qualified. Normal and reordered
  capacity probes passed. Restricting visibility to one SM86 while requiring SM120 exited nonzero,
  reported the absent architecture unqualified and separately reported one foreign-completion
  case skipped because only one device was visible; that negative run is not counted as a passing
  qualification lane.
- Passed affected correction evidence at `eaf8846`: the full locked/offline device-feature
  all-target suite; focused executor driver faults; host and device-feature workspace clippy;
  formatting, diff, architecture and specification checks; aggregate and both per-architecture GPU
  qualification runs. Every GPU run reported 39 passed, 0 failed and 0 skipped/unmeasured. New
  primitive and selected-chain H8 overflow regressions return typed `Numerical` on every UUID.
  The exact H8 request now has a 176 B host peak (160 B retained sources plus 16 B final output), and
  a ledger with exactly 160 usable host bytes rejects before allocation. Cancellation followed by
  observed completion recovers two bound immutable weights and one input; weight rebinding is
  rejected before copy/launch, and the input-only reuse succeeds.
- Numerical result: selected H8/rows1 and H17/rows5 chains were bit-identical to the interpreter on
  every GPU (final max/RMS/p99 ULP all zero). The representable primitive graph matched the
  interpreter at Linear, RMSNorm and Residual boundaries. Across H8, H17 and H1024/rows64, worst
  normalized errors were Linear `0.386110`, RMSNorm `0.960743` and Residual `0.996094`, all within
  the fixed bound. Negative fixtures feed reversed accumulation, an omitted BF16 boundary,
  LayerNorm substitution and residual-twice results through the same acceptance predicate and
  observe rejection; nonfinite metrics fail closed.
- Resource/lifetime result: rows1/H8 admits 512 B weights, 768 B activations and 256 B workspace in
  one 1536 B allocation; rows5/H17 admits 1024 B, 768 B and 256 B in one 2048 B allocation. The RMS
  logical workspace is respectively 4 B and 20 B. Interposed driver census on every UUID observes
  one allocation, three first-use uploads, four launches, one event, no intermediate readback or
  synchronization, and one final readback; the second execution uploads only `x`. Binding,
  selection, range-unwind, launch, event create/record/query/sync, final-readback and free failures
  retain or quarantine the declared owner. Cancellation before event completion withholds the
  plan/ranges/sources through the first no-next-token sweep; after completion, the sweep settles
  uploaded weights into the recovered plan and returns the per-step input.
- CUDA validation passed: Compute Sanitizer memcheck reported zero errors on SM120 and SM86 for the
  reduced H8/H17 chain plus the overflow regression; SM120 racecheck reported zero hazards and
  initcheck reported zero errors. No sanitizer lane was skipped.
- Failed during development and corrected before the final evidence: early numerical negatives
  compared only pre-BF16 arithmetic; the final fixtures use biting stored outputs and the real gate.
  Early integrated oracle evidence observed only the final value; the stateless trace now exposes
  every node. Exact graph edges, external-binding membership and descriptor workspace expressions
  were tightened. Completion attribution now wraps the event before the lifecycle can persist
  driver loss, and exact request charges/spans plus forced range unwind are inspected. Intermediate
  compile/clippy/assertion failures were corrected without changing a numerical threshold. During
  the owner-review correction, the first aggregate rerun exposed that primitive nonfiniteness was
  checked only after Linear (36 passed, 3 failed); the common primitive readback gate now checks
  Linear, RMSNorm and Residual, after which all 39 cases passed.
- Deleted/replaced paths: selected graphs replace task 0011's unconditional
  `UnsupportedKernel` result only for the exact admitted chain. No allocator, transaction, cache,
  model, checkpoint or legacy path was deleted; the legacy checkout was unchanged.
- Skipped/unmeasured outside the bounded claim: no checkpoint or model was read; attention, paged
  state, actual-context inference, sampling, generation, model quality, topology communication and
  paired performance remain unimplemented or unmeasured. `visible_tokens=32_768` is selection
  metadata only and is not a long-context result.
- Remaining blocker and next bounded task: owner re-review/acceptance is still required. Do not mark
  M1.3 complete or create task 0013 from this implementation record. After owner acceptance, update
  the active records and define M1.4 appendable paged state bound to task 0004's transaction
  mechanism.
