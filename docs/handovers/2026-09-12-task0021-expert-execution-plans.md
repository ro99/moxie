# Handover — task 0021 implemented; task 0022 is M2 item 4

**Task 0021 is implemented and awaits independent review and owner acceptance.**
It delivers roadmap **M2 item 3** — "CPU expert fallback and GPU grouped
candidate plans under one interface, with bounded queues and NUMA-aware host
placement" — and nothing beyond it. **It does not close M2.**

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Contract `cdda4f4`, written and committed before implementation; the
  implementation is the commits between it and the one this handover
  accompanies. Base before both was `b087b92`, the owner's support-matrix rows for the
  task-0020 narrowings, one commit above the task 0020 acceptance record
  `8b230ab`.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and
  `tests/p2p/` remain untouched.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was copied, converted, deleted, downloaded or
  modified.** Bounded ranges of one shard of one artifact were read, and — for
  the first time — **computed with**.

## Completed facts

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy | passed |
| `cargo test --workspace --locked --offline` | **880 passed, 0 failed** (823 at task 0020) |
| Device-feature workspace tests | **906 passed, 0 failed** (843 at task 0020) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures** |

Nothing failed and nothing was skipped. The two real-artifact cases print
`SKIPPED` with a reason when the designated artifact is absent; on this machine
they did not.

**A routed expert block of a real checkpoint executed.** 118,947,840 B of
`/fast/models/google/gemma-4-26B-A4B-it` layer 0 — ten distinct experts across
two rows of top-k 8 — demand-loaded through the residency authority into a
device cache holding **two** experts, executed by the grouped kernel behind a
four-deep queue, with 28 evictions and 8 backpressure drains, and agreeing with
the CPU candidate on all 45,056 BF16 slot components.

**That is not model support and no quality claim follows from it.** The
activations are synthetic and the route is written by the test. Quality is O2
and needs paired output against the released model. One layer is not a model:
nothing composes a routed layer into a graph that generates a token.

## Decisions

**The choice between candidates is arithmetic, not a cost model.**
`bytes_per_row = transfer_bytes / reuse_rows` is document 03's "row reuse
amortizes transfer" as a quantity, and the threshold it is compared against is a
**declared policy parameter, not a measured crossover**. There is no measured
host expert throughput and no measured device grouped throughput on this
machine, so a number presented as a crossover would be a fabrication. What
"conservative deterministic scheduling" can honestly mean at M2 is that the same
input yields the same decision, the parameter is visible, and the rejected
alternative is reported with the numbers that decided it.

**At this artifact's scale the declared default sends every expert to the CPU.**
One expert is 11,894,784 B and a two-row batch shares six of ten experts, so the
best available reuse is 5,947,392 B per row against a declared default of 1 MiB.
A test asserts that rather than leaving it to be rediscovered. Whether that is
the *right* decision is a measurement, and the measurement is M6's.

**Partial outputs are placed, not accumulated.** Every group writes distinct
slots of one slot-major buffer and the reduction runs once, per row, over a
permutation the plan computed from `CombineOrder`. Nothing sums into a shared
accumulator, so no sum can depend on which candidate finished first. The device
readback is per-slot for the same reason: the device slot buffer never held the
host groups' results, and copying it whole would overwrite them.

**Backpressure is what makes a budget smaller than the working set executable.**
An acquire the residency authority refuses while the queue holds work drains and
retries; with an empty queue the refusal is real and is propagated. A queue that
waited instead would reintroduce exactly the blocking path task 0020 removed
from `acquire`.

### NUMA placement took three corrections, and every one came from a measurement

This is the part of the task most worth carrying forward, because each step
looked finished before it was measured.

1. **Writing zero to first-touch a known-zero buffer can be deleted.** The
   pages were then faulted later, by whoever wrote first.
2. **First touch does not place pages the allocator has already faulted.**
   glibc raises its dynamic mmap threshold after a large mapped block is freed
   and then serves the next large request from the brk heap. A 32 MiB "placed"
   buffer came back with **3,317 of 8,192 pages on the wrong node**, faulted
   minutes earlier by an unrelated allocation. These buffers now own their
   mappings.
3. **Binding a thread does not bind its pages.** Even with fresh mappings and
   the thread on the right node, the default policy prefers local and then falls
   back rather than reclaiming. Node 1 on this machine has **334 MB** free
   against node 0's **5.1 GB**, both holding ~125 GB of page cache, and
   **4,471 of 6,144 pages** landed on node 0.

So `required` means `mbind` and the gate is every page — **6,144 of 6,144** on
the planned node, read back from `/proc/self/numa_maps` — while `auto` is a
preference whose result the test prints rather than asserts. **A placement claim
that has not been read back is not a measurement.**

### The device kernel is bitwise equal, and one uncertainty is untested

36,864 BF16 components across all three GPUs and both gate transforms, plus
45,056 more against the CPU candidate on real weights. Reaching that needed
`__fmul_rn`/`__fadd_rn` and `__dmul_rn`/`__dadd_rn` throughout: **nvcc contracts
`a * b + c` into an FMA by default**, which is a different rounding pattern and
therefore a different answer from the oracle.

The declared uncertainty stands: CUDA's device `tanh`/`exp` in FP64 may differ
from the host libm's by under an ulp, and **no fixture here constructs a case
where such a difference straddles a BF16 tie point**. That case is covered
contractually by `route::expert_error_bound` and is **not shown absent**.

### Mutation testing found two gaps and one dead branch

The sweep's strength is measured, not asserted: 16 deliberate mutations of the
chooser, and the first measurement was **13 of 16 with two survivors**.

- **A refusal's reason is part of its contract.** The sweep asserted only *that*
  a case was refused, so a mutant reporting `CapacityExceeded` where a
  `required` candidate's own reason belongs survived. The same defect was in the
  product code: a plan reported 11,894,784 B "exceeding" 47,579,136 B available,
  because the other candidate had been excluded by a control rather than by
  capacity.
- **A fixture on which two behaviours agree tests neither.** Every row of the
  sweep's route selected its experts in ascending id order, so the two
  `CombineOrder` values produce the same permutation and a planner that ignores
  the parameter is indistinguishable from one that honours it.
- **An unreachable branch is a stub.** A mutation of "the device candidate is
  `required` and this group fell back" changed nothing, because `required` on
  one candidate excludes the other for the whole plan. It was deleted.

Now **16 of 16, 0 survivors**. The battery and what the number does *not*
establish are in
[experiment 0002](../evidence/experiments/0002-expert-plan-sweep-mutations.md).

### Narrowings, reported rather than quietly dropped

The reduction always runs on the host; `Route` stays host-side and the selected
BF16 chain still refuses it; one plan targets one device, because spreading a
layer across the three cards is **M5**; the residency authority's own host cache
is **not** NUMA-placed, and it is the largest host buffer in the system; and the
device workspace charge uses `rows` rather than the largest group, because the
largest group is not known until the assignment exists.

**No owner gate was resolved.** O1–O7 remain open. No numerical threshold,
precision, context target or compatibility surface changed.

## Remaining hypotheses and blockers

- **M2 is not closed.** Item 4 is outstanding, and M2's exit asks for byte/cost
  traces reconciled with the resource ledger across a **whole** working set
  rather than one layer.
- **No performance claim.** The counts `GroupedStats` records exist because
  document 03 requires them recorded. There is no baseline on this machine, both
  lanes are debug builds, and the one wall-clock pair the real-artifact case
  prints is labelled as not a benchmark where it is printed.
- **The declared threshold is not a crossover.** M6 owns measuring one; the plan
  already carries the fields that measurement would fill.
- **Quality is O2** and needs paired output against the released model.
- **Laguna** remains inspected only: `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`
  is verified complete, **no metadata has been interpreted and no tensor read**,
  and its `configuration_laguna.py` and `modeling_laguna.py` are remote code
  document 03 forbids executing.
- **Vision and audio** are M11; the artifact declares both towers.

## Next task

Task 0022 is **M2 item 4**: "Add Laguna model metadata/graph only. Add a second
synthetic MoE consumer with different expert count, activation, top-k, shapes
and route distribution. Exercise an intentionally restricted memory budget
smaller than its working weights."

- Owning components: `moxie-models` for the Laguna **metadata and graph only**
  — no execution loop, no allocator, no CUDA, per AGENTS.md's ownership rule;
  `moxie-plan` and `moxie-executor` gain nothing structural, because the second
  consumer must go through the interface task 0021 built rather than around it.
  **`moxie-memory` gains nothing**: a second weight-residency owner is still a
  failed task.
- Required reading before the contract: document 03's affine-integer v1
  descriptor in full (Laguna is `w4a16` and its metadata is INT4, so the *graph*
  may be written before any importer exists but the tensor roles must be the
  canonical ones); document 02's model-ownership row and the enforced extension
  rule; M2 items 4 and 5; ADR 0003, ADR 0012 and ADR 0013; task 0018's
  compressed-tensors importer for what metadata is already readable; and
  [the gemma4 bring-up record](../models/gemma4.md) as the template a second
  family's record follows.
- The contract must state, before implementation: exactly which Laguna metadata
  is interpreted and which is **not**, given that its `configuration_laguna.py`
  and `modeling_laguna.py` are remote code that may not be executed and that
  guessing a convention from a suffix is forbidden; what the second synthetic
  MoE consumer differs in, axis by axis, and which of task 0019's and 0021's
  parameters it therefore exercises that the Gemma-like one does not; and the
  restricted budget, expressed as a ratio of working weights rather than a
  constant.
- **Stop conditions:** executing remote model code; interpreting a Laguna
  metadata field whose meaning is not established by the pinned exporter;
  claiming Laguna support from a graph that nothing runs; a second
  weight-residency owner; a model-owned execution path; any bulk write (**O5**);
  and any quality claim (**O2**).
- M2 item 5's nine residency cases, task 0021's sweep and its device cases must
  keep passing unchanged.

**Three habits from this task should be applied there rather than rediscovered.**

First, **a property of the machine is measured or it is not known**: three
plausible NUMA mechanisms in a row were wrong, and only the read-back showed it.

Second, **measure the tests, then fix what the measurement finds**: the mutation
battery found a product defect and a fixture that could not distinguish two
answers, and both were invisible to every other check. Three of the review's
twelve regressions also failed their own substitution on the first attempt,
because they asserted the symptom rather than the check.

Third — and this is the review's lesson rather than mine — **ask of every check
what else reaches the resource it guards.** Nine of the ten findings were a check
present on one path and absent on the neighbour. A sweep over a state machine's
product does not find those; comparing parallel paths does. The tenth is why:
`ExpertGroup`'s launch indices were public `Vec` fields, so the answer to "what
else reaches this?" was "anything at all".
