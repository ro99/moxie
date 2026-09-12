# Handover — task 0020 implemented; task 0021 is M2's grouped expert execution

**Task 0020 is implemented, corrected after independent review, and awaiting
owner acceptance.** The review found **nine** issues; all nine were reproduced
and fixed, and none was disputed. It delivers M2 item 2 only. M2 items 3–5 are
outstanding and the next of them is specified under [Next task](#next-task).

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Contract `d6e9170`; implementation is the commit this handover accompanies.
  Base before both was `8ee8fd0` (task 0019 acceptance record).
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and
  `tests/p2p/` remain untouched.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was copied, converted, deleted or modified, and no
  download was started.** This task did read tensor *payload* bytes for the
  first time — bounded ranges of two tensors in one shard of one artifact, named
  below — and wrote nothing.

## Completed facts

**There is now exactly one production weight-residency owner.**
`moxie_memory::residency` decides what is resident, where, and at whose cost. It
opens no file and touches no device: it issues work orders and
`moxie-executor` performs them, the same split that already pairs `Arena`'s pure
ranges with one real allocation. See
[task 0020](../tasks/0020-m2-weight-residency-authority.md) for the contract,
the terms it fixed before implementation, and the filled-in result.

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy | passed |
| `cargo test --workspace --locked --offline` | **806 passed, 0 failed** (736 at task 0019) |
| Device-feature workspace tests | **826 passed, 0 failed** |
| `cargo xtask-cuda test-gpu` | **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | every rule and every fixture, including the new `a second weight-residency owner` and its three fixtures. **4 pre-existing failures remain**, all from the untracked review crate under `results/task0014-independent-review-2026-09-11/probes/`; the identical four lines are in `results/task0015/arch-local.log` |

Nothing failed. One case is skipped by construction and reports itself: the
real-artifact read prints `SKIPPED` with its reason when
`/fast/models/google/gemma-4-26B-A4B-it` is absent. **On this machine it was not
skipped.**

## Decisions

**The authority never waits, and that is the mechanism rather than a property of
the code.** Document 03 requires admission to "prohibit deadlock when all
evictable entries are leased"; `acquire` has no waiting path, so it cannot take
part in one. When nothing is displaceable it returns `CapacityExceeded` with a
report whose `leased_bytes` equals the cap, and the authority stays usable — on
the host lane and, in `residency_device`, against real device memory.

**Eviction has two phases, and the second is not an afterthought.** Byte
accounting says whether the cache *holds* enough; the arena says whether the free
bytes are in one piece. Document 03 counts `allocator_fragmentation` as real
capacity for exactly that reason, so when the arena refuses bytes the accounting
admitted, the authority keeps evicting in the same deterministic order rather
than reporting a full cache that is not full. A test asserts both numbers.

**A defect this work found in itself, and the stop condition it was.** The first
version gave each device an `Arena` and a capacity and admitted **neither** from
the ledger. Byte accounting was internally consistent and the API was confident,
and it was a placement simulator — R02 in miniature, and one of this task's own
stop conditions in the contract's words, "simulated placement presented as a
reservation". The device test caught it on its first run by asserting the
ledger's charge before allocating anything. Every declared device cache is now
admitted as **one** plan before the host bytes exist, a refusal anywhere in
`open` leaves the ledger as it found it, and two host-lane regressions fail if
either property comes back.

**A self-review pass before hand-off found three more defects**, all in the
least-travelled corner of the lifecycle: a device acquire arriving while another
ticket is already reading the same chunk to the host. Nothing exercised that
path, which is why they survived the first round of tests, and it now has four
of its own. `Retiring` was a **dead state** — eviction only ever chooses
unleased placements, so nothing entered it — and is now reachable through an
explicit `retire(scope, chunk)`, document 03's "`release` retires only after all
consumers complete". Expiring a ticket that had *joined* another's read **freed
the range that read was writing into**, because cleanup treated every ticket as
the owner of its source. And a cancelled upload that then completed **released
its source pin twice**, quietly making a chunk another consumer held evictable.
The last two have regressions proven load-bearing by substitution: reintroducing
each defect fails exactly its own test and no other.

**The independent review found nine more, and the most useful of them corrected
a claim rather than a line of code.** This task had argued that a nonblocking
`acquire` establishes deadlock freedom. It does not: the review found a cycle
between the demand counter and the prefetch gate, which `acquire` never touches
— a device demand waiting on a queued prediction waited for a read that
`next_prefetch` refused to release while demand was outstanding. Priority now
propagates *through* the dependency, and the property is tested rather than
argued.

Four findings were P1. One was a panic (`no entry found for key`) when a host
reader was cancelled while a device acquire waited on it. One was measured on a
real 3090 as **8 MiB of allocation live against a 4 MiB reservation**, with the
charge released by `close` while both allocations stayed readable — R02 with the
sign flipped. One was a **3.4× control-memory undercount**: 166,361 bytes of
retained heap against a 49,216-byte envelope, because a chunk identity is two
heap strings and both the index and the placement held a copy. One was a host
allocation that `Drop` returned to the allocator while a copy might still be
reading it by address.

The task record carries the finding-by-finding table, including the single point
where I resolved a finding differently from the probe's assertion and why. Every
finding has a regression, and the measured ones are measured rather than
asserted.

**Three narrowings, decided during implementation and reported rather than
quietly dropped.** `Artifact::read_tensor_range` was **not** added: a canonical
manifest carries a whole-tensor checksum a ranged read cannot verify, so a
canonical ranged read would skip the integrity check the format exists to
provide, and per-chunk checksums are M3's. The `Preparing` / `Prepared*` states
are **not** implemented: nothing prepares a layout, and an unreachable state is a
stub — `PreparedId` exists so the addition is a state rather than a redesign. The
prefetch class has **no predictor**: the class, its bounded queue, its ordering
behind demand and its one-way eviction rule are implemented and tested, and what
to prefetch is supplied explicitly, because document 03 permits a smarter policy
"only with replayable route traces and measured benefit".

**Tensor payload bytes of a designated checkpoint were read for the first
time.** Nine distinct experts of `/fast/models/google/gemma-4-26B-A4B-it`
layer 0, demanded from routes shaped like a top-k-8 batch of three rows, against
a cache holding four so eviction actually ran: **107,053,056 B**, once each,
every served range verified against an independent read of the same file. That
is `9 × 11,894,784` exactly, and the union is nine rather than the `3 × 8 = 24` a
no-overlap bound would charge. **Nothing was computed with those bytes, no
checkpoint executed, and nothing was written. Reading is not executing, and a
demand-loaded expert is not model support.**

**No owner gate was resolved.** O1–O7 remain open. No numerical threshold,
precision, context target or compatibility surface changed. No quality claim is
made and none follows.

## Remaining hypotheses and blockers

- **M2 is not closed and this task does not close it.** Its exit requires "a real
  out-of-device-memory working set [that] executes without OOM or hidden
  allocations, matches the reference, and produces byte/cost traces reconciled
  with the resource ledger". The residency half and the ledger reconciliation
  exist; **nothing executes a routed layer**, because that is item 3.
- **No device routed execution.** The selected BF16 chain still refuses `Route`,
  `ExpertMlp` and `Combine` as `UnsupportedKernel`, asserted by a test. `Route`
  is `Replicated` by requirement; `ExpertMlp` and `Combine` fail closed for
  partitioning, and expert partitioning is **M5**.
- **Deadlock freedom is a tested property, not an argument.** The absence of a
  waiting path in `acquire` is necessary and **not sufficient**; the review
  proved that by finding a cycle elsewhere. Task 0021 adds queues of its own,
  and the same caution applies to them.
- **No performance claim.** `ResidencyStats` records reads, uploads, hits,
  misses, evictions, wasted prefetch bytes and evictions of demand data, because
  document 03 requires them to be recorded. There is no baseline on this machine
  to compare them against, so **none of them is a measurement of anything but
  itself**.
- **Quality is O2** and needs paired output against the released model.
- **Vision and audio** are M11; the artifact declares both towers.
- **Laguna** remains inspected only: `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`
  is verified complete, and **no metadata has been interpreted and no tensor
  read**. Its `configuration_laguna.py` and `modeling_laguna.py` are remote code
  document 03 forbids executing. M2 item 4 is unblocked on availability, not on
  inspection.

## Next task

Task 0021 is **M2 item 3**: "CPU expert fallback and GPU grouped candidate plans
under one interface, with bounded queues and NUMA-aware host placement. Begin
with conservative deterministic scheduling."

- Owning components: `moxie-plan` for the candidate plans (it is pure and may not
  allocate or do I/O), `moxie-executor` for the grouped execution, `moxie-kernels`
  for the device path. `moxie-memory` gains **nothing**: the residency authority
  is complete for this purpose and a second cache is still a failed task.
  `moxie-models` gains nothing.
- Required reading before the contract: document 03's "Transfer and CPU/GPU
  policy" in full — the shared CPU expert path is "a planner candidate for
  low-reuse, disk/PCIe-constrained decode", grouped GPU execution "is favored
  where row reuse amortizes transfer", a plan "may combine them and reduce
  partial outputs deterministically", and CPU kernels "operate on bounded tiles
  of canonical packed weights; do not materialize the entire model as BF16";
  document 02's planning contract (`compile` is pure with respect to live
  resources, `admit` reserves atomically, `execute` may not evade the
  reservation); M2 items 3 and 5; and task 0019's `ExpertMlp` and `Combine`
  parameters, which are the mathematics the plans must produce.
- The contract must state, before implementation: which operand layout a grouped
  expert kernel consumes and how a residency lease becomes that operand; how a
  plan chooses between the CPU and GPU candidates and what it reports about the
  alternative it rejected; the bounded queue's capacity and its refusal; NUMA
  placement on this machine, where ordinal 0 is the 5060 Ti on node 0 and the
  3090 pair is on node 1; and the deterministic reduction of partial outputs when
  a plan combines both.
- **Stop conditions:** a second weight-residency owner or any cache in the
  planner; a model-owned execution path; an unbounded queue; a CPU path that
  materializes the whole model as BF16; a device kernel without an unfused
  oracle; simulated placement presented as a reservation; any bulk write (O5);
  and any quality claim (O2).
- M2 item 5's residency cases are done and must keep passing. Item 3 adds its
  own: a plan that cannot fit either candidate, a cancelled grouped execution, a
  partial-output reduction whose order is asserted on a fixture where FP32
  addition is not associative, and the restricted-budget case item 4 names.
