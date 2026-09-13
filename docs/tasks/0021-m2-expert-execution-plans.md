# Task 0021 — M2 expert execution plans: one interface over a CPU and a GPU candidate

Status: **implemented and corrected after two rounds of independent review;
awaiting a further review and owner acceptance.** The contract above was written
and committed at `cdda4f4` before any implementation, per the working rule that
produced tasks 0013–0020. The two rounds found **fifteen** issues — twelve P1
and three P2 — and **all fifteen were reproduced and fixed; none was disputed.**
**Two of the second round's five were the other half of the first round's own
findings**, so the first round's record that "all ten are closed" was premature
and is corrected below. See [Result](#result-filled-after-work),
[Independent review](#independent-review-and-what-it-changed) and
[Second independent review](#second-independent-review).

**This task does not close M2.** It delivers roadmap M2 **item 3** only. Item 4's
Laguna metadata, its second synthetic MoE consumer and M2's exit gate — "a real
out-of-device-memory working set executes without OOM or hidden allocations,
matches the reference, and produces byte/cost traces reconciled with the resource
ledger" — are outstanding after it, and the scope boundary is restated under
[What this task is not](#what-this-task-is-not).

## Identity and authority

- Task ID / milestone / owner: 0021 / **M2 item 3, expert execution plans** /
  implementation agent; acceptance belongs to the owner.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `b087b92`
  (the owner's support-matrix rows for the task-0020 narrowings, one commit above
  the task 0020 acceptance record `8b230ab`). Working tree clean at authoring; no
  initial dirty paths.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and
  `tests/p2p/` are preserved and are not source evidence.
- Assigned by
  [the task 0020 handover](../handovers/2026-09-12-task0020-weight-residency-authority.md#next-task),
  whose "Next task" section names the owning components, the required reading,
  the five terms this contract must state before implementation, the stop
  conditions and the added tests. Every one of them is carried below and is
  cross-referenced where it is answered.
- Requirement / finding IDs: **R02** (no simulated placement), **R06** (routing
  differences are not erased behind a common default), **R07/R08** (source and
  lease lifetime across asynchronous work), **R12** (this machine's links are not
  equivalent and a pinned buffer is not automatically faster), **R14/R15**.
- Required documents read: AGENTS.md, README, reference documents 01–09, the
  owner-gate register, `docs/README.md`, the TASK/ADR/HANDOVER templates.
  Normative for this task: document 03's **"Transfer and CPU/GPU policy"** in
  full; document 02's **"Planning contract"** (`compile` pure with respect to
  live resources, `admit` atomic, `execute` may not evade the reservation), its
  crate-ownership table and its **"Buffer and asynchronous lifetime contract"**;
  document 06 **M2 items 3, 4 and 5**; document 01's `auto`/`required` rule; and
  the accepted results of tasks [0012](0012-m1-selected-bf16-device-chain.md),
  [0019](0019-m2-routed-expert-semantics.md) and
  [0020](0020-m2-weight-residency-authority.md).
- Owner gates: **O1–O7 remain open.** None blocks this task. It reads bounded
  ranges of a read-only local artifact and **writes nothing** under `/models` or
  `/fast/models`; tasks 0018–0020 already established that boundary. Stop before
  any conversion, requantization, download, bulk write (O5) or quality claim
  (O2).

### What this task is not

It does **not** execute a checkpoint and makes **no model-support claim**. It
executes one routed expert block over real expert weights and **synthetic
activations**, which proves residency and execution machinery and proves nothing
about output quality — quality is O2 and needs paired output against the released
model. It adds no device `Route` kernel (routing stays host-side and is an
*input* to an expert plan), no expert partitioning across devices (**M5**), no
quantized expert path (**M3**), and no measured cost model (**M6**). It changes
nothing in `moxie-memory`: a second weight-residency owner is a failed task.

## The five terms the handover requires stated before implementation

### 1. The operand layout, and how a residency lease becomes it

One expert of one layer is **two residency chunks and nothing else**, exactly as
task 0020 made them:

| Operand | Chunk | Logical shape | Bytes (designated artifact, layer 0) |
|---|---|---|---:|
| `gate_up` | role `…experts.gate_up_proj`, `TensorSlot::expert(role, e)`, range `[e·2IH·2, +2IH·2)` | BF16 row-major `[2I, H]` | 7,929,856 |
| `down` | role `…experts.down_proj`, `TensorSlot::expert(role, e)`, range `[e·HI·2, +HI·2)` | BF16 row-major `[H, I]` | 3,964,928 |

with `H = 2816`, `I = 704`. Within `gate_up` the **gate block is rows `[0, I)`
and the up block is rows `[I, 2I)`** — `chunk(2, dim=-1)` in the pinned
reference, not interleaved pairs. This is task 0019's declared layout and is
restated here because a kernel that reads it interleaved produces a plausible
wrong answer.

A lease becomes an operand by exactly one call, and the two candidates use
different ones:

- **Host candidate:** `ResidencyAuthority::chunk_bytes(&lease) -> &[u8]`. The CPU
  kernel consumes those BF16 bytes **in place**. It never materializes an expert
  as FP32; see term 5's tile budget.
- **Device candidate:** `ResidencyAuthority::device_range(&lease) -> (offset,
  len)`, plus the one real allocation the executor already holds for that scope
  through `DeviceBacking`. The executor adds the base; `moxie-memory` still holds
  no pointer.

No other path to weight bytes exists in this task. A kernel that took a
`Vec<u8>` of weights assembled anywhere else would be a second cache with extra
steps.

### 2. How a plan chooses, and what it reports about the alternative it rejected

The choice is **per expert group** — expert-major, ascending expert id — and it
is deterministic, conservative and structural. For each group:

```text
reuse_rows      = number of (row, slot) pairs routed to this expert
transfer_bytes  = 0 if the snapshot says both chunks are already resident on the
                  target device, else the two chunks' bytes
bytes_per_row   = ceil(transfer_bytes / reuse_rows)
```

`bytes_per_row` is document 03's sentence as arithmetic: "GPU grouped expert
execution is favored where row reuse amortizes transfer". The device candidate is

- **admissible** when the snapshot's device expert cache can hold the incoming
  chunks after its declared evictable bytes, *and* the device arena has room for
  this group's activation and workspace bytes; and
- **preferred** when admissible and `bytes_per_row <= policy.max_transfer_bytes_per_row`.

Otherwise the host candidate is taken if it is admissible against the host
workspace budget. If neither is admissible the plan **refuses** with
`CapacityExceeded` and a report naming both reasons.

`policy.max_transfer_bytes_per_row` is a **declared parameter, not a measured
crossover.** There is no measured CPU expert throughput and no measured device
grouped throughput on this machine, so a number presented as a crossover would be
a fabrication. What "conservative deterministic scheduling" can honestly mean at
M2 is: the same input produces the same decision every time, the parameter is
visible, and the rejected alternative is reported with the numbers that decided
it. A measured crossover is M6's, and the plan carries the fields that
measurement will fill.

Every group carries a `GroupDecision` naming the chosen placement, the rejected
placement, the reason code, and the four numbers above. `StrategyControl` from
document 01 governs each candidate: `Off` removes it from consideration, `Auto`
reports the selection, and `Required` **errors** when that candidate is
unsupported or inadmissible rather than silently taking the other one.

### 3. The bounded queue's capacity and its refusal

`moxie_executor::grouped::OrderQueue` has a capacity fixed at construction from
the plan and **never grows and never blocks**. Capacity is
`min(policy.max_inflight_orders, group_count)` and is *admitted*: each queue slot
owns one intermediate tile —
`max_group_rows · I · 4` bytes of `DeviceTier::KernelWorkspace` for a device
group and `policy.cpu_tile_rows · I · 4` bytes of `HostTier::CpuWorkspace` for a
host group — so the queue's depth is part of the reserved envelope rather than a
number chosen at runtime.

`push` on a full queue returns `Error::CapacityExceeded` naming the tier and the
capacity. It is not a wait, not a grow and not a drop: the caller drains
completed orders and pushes again. A queue that blocked would reintroduce exactly
the waiting path task 0020 removed from `acquire`, and the review that found a
cycle between the demand counter and the prefetch gate is the reason this is
stated as a mechanism rather than a preference.

### 4. NUMA placement on this machine

Topology is **measured, never assumed**. `moxie-host` — the one crate ADR 0006
permits to read machine telemetry — reads, under an injectable root:

- `/sys/devices/system/node/online` for the node set,
- `/sys/devices/system/node/node<N>/cpulist` for each node's CPUs,
- `/sys/devices/system/node/node<N>/meminfo` for its memory, and
- `/sys/bus/pci/devices/<bus id>/numa_node` for a device's node, keyed by the
  `pci_bus_id` that `DeviceCapability` already carries.

On this machine that yields node 0 = CPUs `0-13,28-41` and the 5060 Ti
(`0000:03:00.0`), node 1 = CPUs `14-27,42-55` and the 3090 pair
(`0000:82:00.0`, `0000:83:00.0`) — which reproduces
[the hardware inventory](../evidence/hardware-inventory.md) rather than trusting
it. A device whose `numa_node` is `-1` yields `None`, and the plan then places
host buffers `Unspecified` and says so; it does not default to node 0.

**Placement rule:** every host buffer this task owns — the CPU expert workspace,
the host slot buffer and any transfer staging — is placed on the **node local to
the plan's target device**, because those bytes are either read by that device's
DMA engine or written by a CPU that must then feed it.

**Placement mechanism:** the executor binds the calling thread's affinity to the
node's CPU set with `sched_setaffinity`, allocates and **first-touches** the
buffers while bound, and restores the previous mask on drop of a guard. This is
Linux's first-touch policy used deliberately rather than `mbind`, and it needs
one small audited FFI block; it lives behind a `numa` feature so the crate keeps
`forbid(unsafe_code)` without it, and `HostPlacementControl::Required` errors
when the feature is absent rather than proceeding unplaced.

**Placement evidence:** the resulting pages' node is **read back** from
`/proc/self/numa_maps` through `moxie-host` and asserted. A NUMA claim that is
asserted rather than measured is the exact mistake
[AGENTS.md](../../AGENTS.md) records three times against task 0020's coverage
claims.

**Stated limitation:** the residency authority's own host cache is task 0020's
buffer, allocated before this plan exists, and this task does **not** place it.
It is the largest host buffer in the system, so "NUMA-aware host placement" is
true of this task's buffers and not yet of that one. The next bounded task is
named in the result section.

### 5. The deterministic reduction of partial outputs

Partial outputs are **placed, not accumulated.** Every group writes its results
into distinct slots of one slot-major `[rows · top_k, H]` BF16 buffer, slot `j`
of row `r` at index `r · top_k + j`. Nothing sums into a shared accumulator, so
no sum can depend on which candidate finished first.

The reduction then runs once, per row, over `top_k` terms in an order the
**plan** computed as data: `moxie_plan` turns the graph's `CombineOrder` into an
explicit slot permutation per row, and the kernel reduces in that given order. A
permutation in the plan is the honest form of task 0019's decision that the
order "is a parameter rather than a scheduling detail because floating-point
addition is not associative"; it also keeps `moxie-kernels` free of the graph
vocabulary, which its allowlist row forbids.

The device candidate's slot outputs are copied back into the same host slot
buffer at the same indices before the reduction, so a mixed plan and an
all-device plan reduce identical bytes in an identical order.

## Bounded deliverable

**One concrete outcome:** one planner capability — an `ExpertPlan` that expresses
a routed layer's expert work as CPU and GPU candidates under one interface, with
an exact resource envelope, a bounded queue, NUMA-placed host buffers and a
declared reduction order — together with the executor that performs it and the
two kernels it dispatches to.

**Sole owning shared components, and what each gains:**

- `moxie-plan` — **owns the plan.** Pure: it allocates nothing, opens nothing and
  queries no device. New module `src/expert.rs`.
- `moxie-executor` — **owns execution.** The bounded queue, the NUMA binding
  guard, the dispatch of each group to its kernel, the slot buffer and the
  reduction call. New module `src/grouped.rs`, new `numa` feature.
- `moxie-kernels` — **owns both kernels.** New `src/cpu_expert.rs` (a tiled BF16
  host expert kernel and the host reduction) and new `cuda/expert_mlp.cu`
  (`moxie_bf16_expert_mlp_v1`), plus its catalogue descriptor.
- `moxie-host` — **owns the topology reading.** New NUMA reader; ADR 0006 already
  makes this the only crate that may.
- `moxie-types` — gains the shared descriptors the others need to agree on:
  `NumaTopology`, `NumaNodeId`, `SemanticKernelOp::ExpertMlp` and its
  `GateTransform` dispatch key.

**`moxie-memory` gains nothing. `moxie-models` gains nothing.** No new crate.

**Allowed production files:** the five listed above, plus
`crates/moxie-kernels/build.rs` and `crates/moxie-kernels/Cargo.toml` for the new
`.cu` file, `xtask/src/archcheck.rs` for one allowlist row (`moxie-plan` does not
change; `moxie-executor` does not change; **no new workspace edge is required**),
and the manifests of the crates above.

**Explicit non-goals and forbidden shortcuts:** no cache anywhere but
`moxie_memory::residency`; no model-owned execution path; no unbounded queue; no
CPU path that materializes a whole expert, let alone the model, as FP32; no
device kernel without the unfused oracle it is compared against; no simulated
placement presented as a reservation; no bulk write; no quality claim; no
performance claim; no measured-crossover claim for the policy parameter.

**Existing consumers and second-consumer proof:** the plan is exercised by two
consumers with **opposite** shapes — a low-reuse decode batch (few rows, 8
distinct experts each, `bytes_per_row` large) that selects the host candidate,
and a high-reuse prefill batch (many rows over few experts) that selects the
device candidate — plus a third that mixes both in one plan. Task 0019's second
synthetic MoE consumer, with a different expert count, top-k and activation, is
run through the same interface.

**Temporary paths to delete or bridge expiry:** none. Nothing here is a bridge.

## Contract before implementation

### Equations, shapes, precision, rounding

The mathematics is task 0019's, unchanged, and this task adds none. For row `x`
(`[H]` BF16) and expert `e`:

```text
gu = bf16( Σ_k x_k · GU[e][·,k] )          FP32 sequential ascending k
h  = bf16( act(gu[0..I]) · gu[I..2I] )     act per ExpertActivation, task 0019
y  = Σ_i h_i · D[e][·,i]                   FP32 sequential ascending i
slot = bf16(y)                             the node output boundary
row  = bf16( Σ_{j in order} w_j · slot_j ) FP32, order from the plan
```

`GeGlu` rounds `gelu_tanh(gate)` to BF16 before the product (the pinned Gemma 4
source does); `SwiGlu` evaluates in FP64 and rounds once. Neither boundary is
this task's to move.

Accumulation is `AccumulationPolicy::Bf16InF32Acc`; rounding profile
`RoundingProfile::FinalBf16Rne`; layout `TensorLayout::ContiguousRowMajorV1`.

### Independent oracle and predeclared numerical gates

The oracle is `moxie_oracles::route::{expert_row, combine_row, combine_order}`,
accepted in task 0019, with its unfused stages (`linear::linear_row`,
`activation::{geglu_row, swiglu_row}`) separately callable — which is what makes
the device kernel legal under this task's own stop conditions.

Two gates, declared now:

1. **The CPU kernel is bit-exact.** Every operation in it is FP32/FP64 in a fixed
   order, so `cpu_expert` must equal `expert_row` followed by the BF16 node
   boundary **bitwise**, on every fixture. Anything less is a defect, not a
   tolerance.
2. **The device kernel is bit-exact on the qualification fixtures, and bounded in
   general.** Its only inexact step is the gate transform's transcendental:
   CUDA's device `tanh`/`exp` in FP64 may differ from the host libm's by under an
   ulp. Two claims follow and they are different claims. The *tested* one is
   **bitwise equality of the final BF16 output** over the qualification fixtures,
   reported with the exact number of elements compared. The *contractual* one is
   task 0019's `route::expert_error_bound`, which covers the case a sub-ulp FP64
   difference straddles a BF16 tie point. **No fixture in this task constructs
   that case**, so it is untested rather than shown absent, and the result
   section must say so in those words.

### Resource envelope, transfer dependencies and lifetimes

`compile` produces an `ExpertEnvelope` with exact byte counts per tier and scope:
`DeviceTier::{Activations, KernelWorkspace, TransferStaging}` and
`HostTier::CpuWorkspace`, plus the residency bytes each chosen group will demand,
which are **reported, not reserved here** — the residency authority admits those
against its own cap and is the only thing that may.

`admit` reserves the envelope atomically against the ledger and returns a
`ReservedExpertPlan`. `execute` allocates nothing outside it; a test asserts the
ledger's charge before a single byte is allocated, which is the assertion that
caught task 0020's placement simulator on its first run.

A weight lease is held from before the group's kernel is dispatched until after
its completion is observed — `Lease::retire` semantics, not `Drop` — because a
device kernel leases its inputs until completion (document 02) and an evicted
expert under a running kernel is R07 with the sign flipped.

### Cancellation, failure and rollback

Cancelling a plan mid-execution: queued orders are discarded, in-flight orders
are **withheld** until their completion or loss is observed, every residency
lease is released exactly once, the slot buffer is dropped without being read,
and the ledger returns to the charge it had before `admit`. A cancelled device
group whose upload was already enqueued follows task 0020's
`Outcome::SubmissionUnknown` path: the bytes are quarantined, not reused.

A failed group fails the plan. There is no partial answer: a row reduced over
`top_k - 1` slots is a wrong answer, not a degraded one.

### Partition and hardware capabilities

`Route` remains `Replicated` by requirement. `ExpertMlp` and `Combine` keep
`PartitionRule::NotDetermined` and continue to fail closed for partitioning:
sharding experts is **M5**, and this task must not quietly decide it. The device
kernel is qualified per `sm` exactly as task 0012's are: `sm_86` and `sm_120`
separately, on real hardware, by UUID.

### Application compatibility and sampler implications

None. This task adds no protocol surface, no sampler behaviour and no CLI flag
beyond what the diagnostic CLI already reports about a plan.

## Acceptance

Every command below is run and reported **separately** as passed, failed or
skipped. A skip prints its reason.

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --locked -- -D warnings`, and the
  device-lane clippy
- `cargo test --workspace --locked --offline` (823 tests pass at base `8b230ab`;
  the new total is reported, not rounded)
- device-feature workspace tests
- `cargo xtask-cuda test-gpu` — 39 pass at base; new device cases are added to it
- `cargo xtask spec-check`
- `cargo xtask arch-check` — **zero failures**, which is the standard task 0020
  established. A standing failure count is not carried forward.

### The cases this task must add

Named by the handover, plus the ones its own mechanism requires:

1. **Neither candidate fits.** A plan whose device cache cannot hold one expert
   and whose host workspace budget cannot hold one tile refuses with
   `CapacityExceeded` naming both reasons.
2. **`Required` is not a suggestion.** `StrategyControl::Required` on the device
   candidate errors when that candidate is inadmissible, instead of taking the
   host one.
3. **A cancelled grouped execution.** Leases released exactly once, ledger
   restored, and a regression that fails if a lease is released twice.
4. **The reduction order is asserted where it matters.** A fixture whose slot
   values make FP32 addition non-associative — so `AscendingExpertId` and
   `SelectionOrder` give different BF16 rows — with one slot computed by the host
   candidate and another by the device candidate, asserting the declared order's
   result and asserting the other order's result **differs**.
5. **The restricted budget of M2 item 4.** A memory budget deliberately smaller
   than the working weights, executing to a correct answer with eviction running
   throughout, with the residency statistics reported.
6. **The bounded queue refuses.** Pushing past capacity returns
   `CapacityExceeded`; draining makes room; the queue never grows.
7. **NUMA placement is measured.** Host buffers planned for node `N` have their
   pages read back from `numa_maps` as node `N`, and the case where the topology
   reports no node is exercised under an injected root.
8. **Two opposite consumers and one mixed plan**, as named under the bounded
   deliverable.
9. **Real expert weights execute.** One layer's routed expert block of
   `/fast/models/google/gemma-4-26B-A4B-it` over **synthetic activations**,
   demand-loaded through the residency authority against a device budget smaller
   than the layer's experts, matching the host oracle. It prints `SKIPPED` with
   its reason when the artifact is absent. **This is not model support and no
   quality claim follows from it.**

M2 item 5's nine residency cases and all 800 combinations of
`residency_transitions.rs` must keep passing, unchanged.

### Coverage is measured, never asserted

AGENTS.md records three task-0020 claims that were wrong in the same way: a
property of the tests was asserted instead of measured. This task's plan-choice
matrix is a product — candidate control (3) × device admissibility (2) × host
admissibility (2) × reuse above/below the policy (2) × already-resident (2) — and
it is swept, not sampled, with the sweep **printing what it exercised**. The
sweep's strength is reported as a mutation count against deliberate mutations of
the chooser, and an equivalent mutant is reported as equivalent rather than
counted as a gap.

### Support matrix and documentation gates

`docs/evidence/support-matrix.md` gains the grouped expert kernel's `sm`
qualification and nothing broader. `docs/models/gemma4.md` records what actually
executed and, in the same sentence, what that does not establish.
A handover names the next bounded task. AGENTS.md's active-assignment section is
updated only to state what is true.

### The exact condition requiring owner direction or task rejection

If the device grouped kernel cannot reproduce the oracle within the declared
gate, **stop and report the measured deviation**; do not widen the tolerance, do
not change the accepted rounding boundaries of task 0019, and do not present the
loose `expert_error_bound` as a pass where bitwise equality was the declared
gate. If satisfying M2 item 3 appears to require a second cache, a waiting queue,
a model-owned execution path or a bulk write, stop at that gate and report it.

## Result, filled after work

### Changed shared owners and consumers

| Crate | What it gained |
|---|---|
| `moxie-types` | `NumaTopology` / `NumaNode` / `NumaNodeId` / `HostPlacement`; `GateTransform`; `SemanticKernelOp::ExpertMlp(GateTransform)`; `KernelOperand::RouteIndex`; `WorkspaceExpression::RowsTimesIntermediateF32` |
| `moxie-host` | `numa`: the sysfs topology reading and the `numa_maps` read-back |
| `moxie-plan` | `expert`: the plan, the chooser, the envelope, the reduction permutation, kernel selection |
| `moxie-kernels` | `cpu_expert` (host expert kernel and host reduction); `cuda/expert_mlp.cu`; `expert_mlp_catalogue` |
| `moxie-executor` | `grouped` (bounded queue, placed host buffers, the run); `grouped_device` (arena, launches, readback); the `numa` feature |
| `xtask` | one `test-gpu` case, `grouped_expert_mlp` |

**`moxie-memory` gained nothing and `moxie-models` gained nothing**, as the
contract required. No new crate, and one new workspace edge (`xtask` -> 
`moxie-oracles`, test-lane only).

### Commands, and what each reported

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy (`--features moxie-executor/driver`) | passed |
| `cargo test --workspace --locked --offline` | **882 passed, 0 failed** (823 at task 0020; 871 before the reviews' corrections) |
| Device-feature workspace tests | **910 passed, 0 failed** (843 at task 0020; 895 before the corrections) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified (39 at task 0020) |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures**, 78 rejected fixtures, 21 accepted, 13 rules |

Nothing failed. Nothing was skipped: the real-artifact cases print `SKIPPED`
with a reason when `/fast/models/google/gemma-4-26B-A4B-it` is absent, and **on
this machine they were not skipped**.

### Measured effect, and its uncertainty

**The CPU kernel is bitwise equal to the oracle**, for both gate transforms,
over every fixture, with the tile width proven numerically inert across six
tile sizes.

**The device kernel is bitwise equal to the oracle**: 36,864 BF16 components
across all three GPUs and both gate transforms, plus 45,056 more against the CPU
candidate on real weights. Reaching that needed `__fmul_rn` / `__fadd_rn` and
`__dmul_rn` / `__dadd_rn` throughout — nvcc contracts `a * b + c` into an FMA by
default, which is a different rounding pattern and therefore a different answer.
**The declared uncertainty stands and is untested**: CUDA's device `tanh`/`exp`
in FP64 may differ from the host libm's by under an ulp, and no fixture here
constructs a case where such a difference straddles a BF16 tie point. That case
is covered contractually by `route::expert_error_bound` and is **not shown
absent**.

**A real layer executed.** 118,947,840 B of the designated artifact's layer 0 —
ten distinct experts across two rows of top-k 8 — demand-loaded through the
residency authority into a device cache holding **two** of them, executed behind
a queue four deep, with 28 evictions and 8 backpressure drains, agreeing with
the CPU candidate on all 45,056 BF16 slot components. **This is not model
support**: the activations are synthetic and the route is written by the test,
so no quality claim follows (**O2**). Nothing under any checkpoint root was
written, copied, converted or downloaded.

**NUMA placement is measured, and the measurement changed the design three
times.** Writing zero to first-touch a known-zero buffer can be deleted by the
compiler. First touch does not place pages the allocator has already faulted —
glibc raises its mmap threshold after a large mapped block is freed and then
serves from the brk heap, and a 32 MiB "placed" buffer had **3,317 of 8,192
pages on the wrong node**. And even with fresh pages and the thread bound, the
default policy prefers local and then falls back rather than reclaiming: node 1
on this machine has **334 MB** free against node 0's **5.1 GB**, and **4,471 of
6,144 pages** landed on node 0. So `required` means `mbind`, and the gate is
every page: **6,144 of 6,144** on the planned node, read back from `numa_maps`.
With `auto` it is a preference, and the test prints what it got.

**No performance claim.** `GroupedStats` records groups, slots, backpressure
drains and leases because document 03 requires them recorded. There is no
baseline on this machine, both lanes are debug builds, and the one wall-clock
pair printed by the real-artifact case (device 57 ms, host 2.8 s including
118 MB of disk reads, one run each) is labelled as not a benchmark where it is
printed. None of these numbers is a measurement of anything but itself.

### Coverage, measured rather than asserted

The sweep prints what it covered: **5,184 combinations** over both controls,
both plan-wide admissibilities, the per-expert cache, three amortisation levels,
residency, reduction order, topology and three kernel-catalogue states — 1,348
planned (224 all-device, 1,120 all-host, 4 mixed) and 3,836 refused, with every
rejection reason exercised at least once.

Its strength is **measured**: 16 deliberate mutations of the chooser, **16
caught by the sweep**, 0 survivors — after two rounds of strengthening that the
first measurement forced (13 of 16, two survivors). Both survivors were real
gaps and one was a real defect in the product code. The battery, the two
lessons and what the number does *not* establish are in
[experiment 0002](../evidence/experiments/0002-expert-plan-sweep-mutations.md).

### Decisions worth finding again

**The choice is arithmetic, and its threshold is declared.**
`bytes_per_row = transfer_bytes / reuse_rows` is document 03's "row reuse
amortizes transfer" as a quantity. The threshold it meets is a policy parameter
reported with every decision and is **not** a measured crossover; measuring one
is M6's. A consequence worth stating: at the designated artifact's 11,894,784-byte
experts and a two-row decode batch, the declared **default sends every expert to
the CPU candidate**, and a test asserts that rather than leaving it to be
rediscovered.

**Partial outputs are placed, not accumulated.** Every group writes distinct
slots; the reduction runs once, per row, over a permutation the plan computed.
Nothing sums into a shared accumulator, so no sum can depend on which candidate
finished first. The device readback is per-slot for the same reason: the device
slot buffer never held the host groups' results.

**Backpressure is what makes a small budget work.** An acquire the authority
refuses while the queue holds work drains and retries; with an empty queue it is
propagated. Without that, a bounded queue deeper than the cache would be a
refusal rather than a schedule.

**An unreachable branch is a stub.** Mutation testing found a "the device
candidate is `required` and this group fell back" branch that no test could
reach, because `required` on one candidate excludes the other for the whole
plan. It was deleted, not tested.

### Narrowings, decided during implementation and reported rather than dropped

- **The reduction runs on the host, always.** Device slots are copied back and
  reduced there, so a mixed plan and an all-device plan reduce identical bytes in
  an identical order. A device `Combine` kernel would have to reproduce that
  order exactly and is not needed to establish it; it is M6's.
- **`Route` stays host-side.** The route table is an *input* to an expert plan.
  A device `Route` kernel is not in this task and the selected BF16 chain still
  refuses it.
- **One device per plan.** `Placement::Device` names the plan's own device.
  Spreading a layer's experts across the three cards is expert partitioning,
  which is **M5**, and this task must not decide it quietly.
- **The residency authority's host cache is not NUMA-placed.** It is task 0020's
  buffer, allocated before any plan exists, and it is the largest host buffer in
  the system. "NUMA-aware host placement" is true of this task's buffers and not
  yet of that one.
- **The device workspace bound uses `rows`, not the largest group.** The largest
  group is not known until the assignment exists and the assignment must not
  depend on the envelope, so the charge over-reserves by construction.

### No owner gate was resolved

O1–O7 remain open. No numerical threshold, precision, context target or
compatibility surface changed. No quality claim is made and none follows.

### Remaining blockers, and the next bounded task

- **M2 is not closed.** Item 4 — Laguna metadata and graph, a second synthetic
  MoE consumer through this interface, and the restricted-budget case at its
  scale — is outstanding, and M2's exit asks for byte/cost traces reconciled
  with the resource ledger across a *whole* working set rather than one layer.
- **One layer is not a model.** Nothing composes a routed layer into a graph
  that generates a token, and no quality claim follows from a layer that matches
  its own reference on synthetic input.
- **The declared policy has no measured crossover.** M6 owns that, and the plan
  already carries the fields a measurement would fill.
- **Expert partitioning is M5** and `PartitionRule::NotDetermined` still fails
  closed.
- The next bounded task is named in
  [the handover](../handovers/2026-09-12-task0021-expert-execution-plans.md).


## Independent review, and what it changed

The review ran against the eleven commits through `758dc66` and reported **ten**
findings with seven reproducers. Every reproducer was run before anything was
changed and every one reproduced; the two findings established by source
inspection were confirmed in the source. **None was disputed.** The repository
was left unchanged by the reviewer.

### The shape they shared

Nine of the ten are one sentence: **a check that exists on one path was missing
on the neighbouring one.** The upload path validated the backing a lease is
resolved through; the launch path did not. The planner computed an envelope for
admission; it checked feasibility against a different, smaller one. The run
could be cancelled; it could not fail. The host candidate's workspace was
charged; the reduction's accumulator, which every plan needs, was not.

That is worth stating plainly because it is not the same failure mode as task
0020's, and the method that caught task 0020's would not have caught these. A
transition sweep enumerates a state machine's product; these were **parallel
paths that were never compared to each other**. What finds them is asking, of
every check, "what else reaches this resource?" — and the tenth finding is the
one that shows why: `ExpertGroup`'s indices were public `Vec` fields, so the
answer was "anything at all".

### Finding by finding

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | P1 | A CUDA failure after an enqueue released the weight leases and dropped the index vectors, though a copy or launch might still be reading them | A launch reports `submission_unknown`; the run **withholds** that group's leases and **quarantines** its host buffers, which are then never unmapped. The index staging comes from admitted storage that outlives the copy |
| 2 | P1 | A lease's offset was resolved inside whatever `DeviceResidency` the caller passed. **Reproduced on a real GPU**: authority A's leases through authority B's backing returned success and a different answer | The launch path checks the backing's authority and device before resolving any address, exactly as `perform_upload` has since task 0020 |
| 3 | P1 | `ExpertGroup`'s row and slot vectors were public and mutable, and `run_group` checked neither their lengths nor their bounds before launching | The fields are private with slice accessors, and the attachment checks assignment count, every row and every slot against **its own** extents before anything is enqueued |
| 4 | P1 | A failed group left the run usable. **Reproduced**: the group failed, `reduce()` then succeeded over an unwritten slot buffer, and the next `step()` returned `Done` with zero groups run | `RunState::Failed` ends the run: no reduction, no progress, no reload. Readiness is also checked at the acquire, where its cause is |
| 5 | P1 | Three buffers were allocated outside the envelope: the reduction's accumulator, the per-launch index staging, and a host tile on a GPU-only plan that charged none | All three are charged and allocated once. `HostBuffers::allocated_bytes` exists so a test asserts **allocation equals charge** |
| 6 | P1 | `detach_reservation` separated the charge from the live host buffers. **Reproduced**: release the detached reservation, then write the still-live activation buffer | The reservation never leaves the run. The device attachment is created inside it and owned by it; one `close` gives everything back |
| 7 | P1 | Selection ignored operand roles, output precision, accumulation, rounding and ABI. **Reproduced**: a descriptor with ABI 999, no inputs and an FP32 output was selected for this BF16 executor | Every axis is matched, and each is proven load-bearing by substitution |
| 8 | P2 | `required` placement returned an ordinary unbound report. **Reproduced**: admitted with `page_policy: "heap"` | It refuses admission and releases the reservation |
| 9 | P2 | Feasibility used a smaller envelope than admission reserved. **Reproduced**: a 640-byte device budget admitted a plan needing 1,280; a zero host-buffer budget still produced a 256-byte host envelope | Feasibility is computed from the admitted envelope — aligned, staging included — and the host buffers and reduction accumulator are plan-level preconditions |
| 10 | P2 | No activation-ready state. **Reproduced**: an unloaded run produced a confident all-zero answer, and activations could be replaced between steps | Execution requires a successful load, and activations freeze once a group has run |

Two of the fixes found further defects of their own, both in the same place and
neither reported: a device attachment that failed **after** an arena had taken
and released the reservation left the run holding host buffers against a charge
of zero — so the buffers are now given back at that moment — and a refused
upload never reached the ticket's own authority, which left a placement
`Uploading` forever and a cache that could not close.

### Evidence

Twelve regressions, one per finding plus the two follow-ons, **each proven
load-bearing by substitution**: remove the check and exactly its own test fails.
Three of them did not bite on the first attempt and the substitution battery is
what said so — they asserted the symptom rather than the check — so one now
asserts the failure comes from the acquire, one uses a fixture where only the
row bound can fire, and one pins the device budget at the exact byte.

The gates after the corrections: **880 host tests**, **906 device-feature
tests**, **42/42 real GPU cases** on both architectures, both clippy lanes,
`spec-check`, and `arch-check` with zero failures. The planner mutation battery
is **16 of 16** after one stale mutation site was re-pointed.

**This round's closing claim was premature.** Two of the second round's five
findings are the other half of findings 1 and 2 above: the launch path reported
an unknown submission and the *activation upload* path did not, and the buffers
were quarantined while the *charge* was still released. Saying "all ten are
closed" was a claim about the fixes rather than a measurement of them, which is
the same error AGENTS.md already records three times over test coverage.

## Second independent review

The second round ran against the corrections and reported **five** findings,
three P1 and two P2, with five reproducers. Every one reproduced before anything
was changed; **none was disputed.**

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | P1 | `close` ignored withheld leases and quarantined buffers. **Reproduced**: unknown launch → cancel → close released 640 host bytes whose buffers stay allocated forever. The attachment had no such state either | `close` refuses while anything is withheld and names it; dropping the refused run leaves the charge outstanding and visible. The attachment quarantines its own ranges on the same condition |
| 2 | P1 | Activation uploads returned ordinary errors after an enqueue, and a failed reload left the previous load's `Loaded` state. **Reproduced with a lane double**: the reload failed and execution then used the rejected input | The upload reports its submission state, an unknown one quarantines the source, and **any** failed load is terminal |
| 3 | P1 | Neither lease's `scope()` was checked. **Reproduced on two real GPUs**: one authority's 3090 leases resolved inside its 5060 Ti backing and produced another expert's output | Both lease scopes are checked against the attachment's device before any offset is resolved |
| 4 | P2 | A refused close consumed the lane and discarded the arena its refusal handed back. **Reproduced**: close against the wrong ledger, then the correct one — "already been closed", reservation still outstanding | Ledger identity is checked before anything is touched, the lane closes by borrow, and a refused arena is put back |
| 5 | P2 | Any acquire failure was counted as backpressure when the queue held work. **Reproduced**: a one-shot `InvalidArtifact` read became one drain, retried into a success, and never surfaced | Backpressure is `CapacityExceeded` and nothing else; every other failure ends the run |

### What these had in common, and it is not what the first round's had

The first round's findings were **parallel paths never compared to each other**.
These are **the same path, one step later**. Quarantine was set correctly at the
moment of failure and then ignored by the next call. A failure was made terminal
for a *group* and not for a *load*. A backing was checked and the lease inside it
was not. The reviewer's own summary is the sharpest statement of it: "checking
quarantine immediately after failure missed what `cancel → close` did next."

That is an argument for the runtime transition sweep this task does not yet
have, and the handover carries it to task 0022 as such rather than as a
suggestion.

### Evidence

Six more regressions. The substitution battery is now **eighteen checks, all
eighteen load-bearing** — and it earned that again: four survived on the first
attempt, three because the mutation was aimed at the wrong one of two identical
lines and one because the check it removed was genuinely redundant. **That last
line was deleted**: a state reset before the activation upload that no test could
reach, because the failure path already moves the state.

**One check is not covered and is not claimed to be.** The attachment-level
quarantine in `DeviceExperts::close` sits behind the run-level one, which *is*
covered by a lane double; reaching it directly needs a CUDA fault after an
enqueue, which this task does not inject. It is defence in depth, reported as
untested rather than counted.

Gates after this round: **882 host tests**, **910 device-feature tests**,
**42/42 real GPU cases**, both clippy lanes, `spec-check`, `arch-check` with zero
failures, and the real-artifact execution still agreeing with the host candidate
on all 45,056 components.