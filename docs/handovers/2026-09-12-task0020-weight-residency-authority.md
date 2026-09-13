# Handover — task 0020 implemented; task 0021 is M2's grouped expert execution

**Task 0020 was accepted by the owner on 2026-09-12**, after seven rounds of
independent review; the seventh reported no new blocking findings and recommended
acceptance within the declared M2 item 2 scope. The seven rounds found
**twenty-seven** issues; all twenty-seven were reproduced and fixed, and none was
disputed. **The last two rounds found no residency defect at all** — both
findings were in the architecture check.

**The acceptance closes task 0020 only.** M2 items 3–5 are outstanding and the
next of them is specified under [Next task](#next-task). Three rounds
criticised method or a coverage claim rather than code, and each was right: the
fourth showed by mutation testing that the sweep did not establish its claim, and
the fifth showed that even after that fix, 120 of its cases never applied the
ending they were named for. It delivers M2 item 2 only. M2 items 3–5 are
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
| `cargo test --workspace --locked --offline` | **823 passed, 0 failed** (736 at task 0019) |
| Device-feature workspace tests | **843 passed, 0 failed** |
| `cargo xtask-cuda test-gpu` | **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures** — 78 rejected fixtures, 21 accepted, 13 rules. The "4 pre-existing failures" carried since task 0014 are gone and were never real; see the note below |

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

**A second review round found seven more, every one P1, and three of them were
consequences of the first round's own fixes.** A closed authority still issued
backings — **4,194,304 bytes allocated on a real GPU against a released
reservation**. A backing accepted another authority's upload and **overwrote
still-leased bytes**. A cancelled device read left a ticketless `Reading`
placement and the next acquire panicked. Promotion still deadlocked through a
dependency that already existed, because it promoted the ticket it was given and
stopped. A refused device admission stranded the prediction it had already
promoted. The control envelope still undercounted — **8,274 bytes against 2,368
admitted** at one pending placement, because the charge had no fixed part and was
measured only against settled placements. And `close` surrendered the entitlement
before `try_free`, which leaves the allocation live when it fails.

One of those corrections undid a fix rather than extending it: my first attempt
at the ticketless-`Reading` bug moved held placements to `Retiring`, which would
have let a *failed* placement serve its uninitialised range. The distinction is
whether the bytes are real, not who holds them.

**The review also caught a regression I introduced.** Copying the first round's
probe crate into `results/` took `arch-check` from four failures to nine — and
the four it already carried came from a task-0014 review crate parked the same
way. Every task since 0014 has reported them as "pre-existing". None was ever a
real violation: `docs/README.md` declares `results/` and `artifacts/` ignored
scratch, and the crate walk read them anyway. It now skips both, with a unit test
pinning that a crate under `crates/` is still found. **`arch-check` passes with
zero failures for the first time in six tasks.**

**A third round found four more, two of them panics, and one of them was my own
round-two fix.** A failed device-owned read left the host acquires that had
joined it attached to a dead ticket. The prefetch queue could issue a ticket that
was waiting on another's read, reaching an `unreachable!`. Promotion of an
already-*issued* prediction undercounted demand and cleared somebody else's slot.
And the scratch exclusion I had just added skipped any directory named `results`
or `artifacts` **at any depth**, which hid a declared model crate under
`crates/results/model` — with a forbidden `std::fs::read` in it — from every
rule. Narrowing a check to remove noise was right; narrowing it by name at any
depth made a real rule unenforceable.

**The most useful thing in the third round was not a finding.** It was the
observation that these cases "continue to expose gaps between individually
passing regressions". Every one of the twenty defects had the same shape: a
transition that was individually reasonable left the structure inconsistent in a
combination nobody had written a test for, and the damage surfaced one or two
operations later. Point regressions caught each case and missed the next, because
the space is a product and the tests were points in it.

So there is now `ResidencyAuthority::check_invariants`, which states the
structural invariants once and can be run after any operation, and
`residency_transitions.rs`, which enumerates the product — destination × urgency
× joiner × ending × surviving lease, **200 combinations** — and checks after
*every* step. **It found a defect none of the three rounds had reached** on its
first run: a device prefetch whose host read a host prefetch had joined, ending
in a deadline expiry. Task 0021 adds queues and plans of its own; the same method
should be applied to them rather than rediscovered.

**A fourth round found three more, and the third of them was aimed at my own
answer to the third round.** Releasing an upload's last source pin did not
finish retirement, leaving a placement `Retiring` with zero leases — charged,
unservable and unevictable, holding a one-chunk cache shut, **with
`check_invariants` passing throughout**. Arch-check still skipped production
code: membership was a literal string test, so a forbidden second `ExpertCache`
under `results/storage`, reached through an allowed path dependency, produced
zero violations, and `./results/model` was invisible because its first segment is
`"."`. Exclusion is now by reachability — normalized members, globs expanded, and
a transitive walk of production path dependencies.

**And the transition sweep did not prove what I said it proved.** The
demonstration was a mutation: discard every promoted work order, and all 200
combinations still passed, while a named regression caught it in one. The harness
discovered work through `ticket_of` and completed it directly, so it was testing
whether the authority can be *poked* into consistency, not whether the scheduler
hands the work out. It is now a faithful executor — it performs only orders it
was given and fails when a ticket is left in flight that nobody was told to
perform — with two axes added because mutation testing showed their absence
(cache pressure with a lease held; retirement while a transfer is outstanding),
and an exact pin-balance invariant.

It is now **800 combinations**, and it catches **nine of ten** deliberate
mutations of the authority; the tenth is caught by its named regression. That
number is in the task record, because "the sweep is strong" is a claim and the
battery is the evidence. **The sweep complements the named tests; it does not
replace them** — which is the honest version of what I claimed in round three.

**A fifth round found two more, and both were about claims this handover already
made.** Arch-check still missed production code: the crate walk followed only
dependency entries with a direct `path`, so `moxie-storage = { workspace = true }`
— whose path lives in `[workspace.dependencies]` — resolved to nothing, and a
forbidden second `ExpertCache` under `results/storage` produced zero violations.
It now resolves inheritance through the *same* `effective_spec` the edge checker
uses, extracted so there is one copy; a second implementation of dependency
resolution is a second set of its bugs, and that is now the second time this
exact check has been wrong.

**And the sweep still counted cases whose ending never happened** — 120 of them,
while the record said every ending applies. The early prefetch drain completed
work before the outcome was injected, and an escape hatch excused the rest.
Orders are now received without being performed until the outcome is injected,
the escape hatch is gone, and the sweep **prints its coverage**: 800 combinations,
800 applying their ending, 0 short-circuited. Re-measured against eleven
mutations it catches ten; the eleventh is caught by its named regression, and one
further mutation was identified as an equivalent mutant rather than counted as a
gap.

**Three coverage claims of mine have now been wrong in the same way**: I asserted
a property of the tests instead of measuring it. That is the lesson worth
carrying into task 0021, and it is in AGENTS.md rather than only here.

**A seventh round found one more, and it was the same check wrong a fourth
time** — this one the round-five lesson applied incompletely. Round five said "a
second implementation of dependency resolution is a second set of its bugs", and
I extracted `effective_spec` so *inheritance* had one reading, while leaving my
own duplicate of the **table enumeration** one function away. It read
`[dependencies]`; the real one reads `[dependencies]`, `[build-dependencies]` and
both target variants. A crate reachable only through a build dependency — which
this file already calls production, "how generated code and kernel compilation
get in" — was invisible to every rule. There is now one enumeration,
`production_dependency_sections`, the duplicate is deleted, and a test asserts
the two consumers agree on which tables count.

**A sixth round found one before it, the same check wrong a third time.**
Member globs other than a trailing `/*` were taken literally, so
`members = ["results/mo*"]` resolved to a directory that does not exist, the real
`results/model` was "unreachable", and a `moxie-models` crate with a forbidden
`std::fs::read` was invisible to every rule.

Four rounds, four routes into the same hole: a directory *name* test (round 3),
a literal membership string test missing `{ workspace = true }` (round 5), a
partial glob expansion (round 6), and a second dependency-table enumeration
(round 7). Each was a cheaper approximation of "is this crate part of the build?"
than the question deserves, and each failed **open** — quietly excluding real
code rather than including scratch.

It is no longer an approximation that has to be right. `expand_member` handles
`*` and `?` per segment exactly and returns `None` for anything it cannot; `None`
means reachability is unknown, and unknown reachability excludes nothing and
checks everything. A wrong guess now costs a probe crate in the report instead of
a production crate vanishing from the rules. **If task 0021 needs to narrow a
check, narrow it so that being wrong is loud — and ask one question once.** Those
four rounds were four places answering one question separately; deduplicating
three of them while leaving the fourth is exactly how the fourth was found.

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
