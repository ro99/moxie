# Handover — task 0023 implemented; M2's exit is one clause from complete

**Task 0023 is implemented and awaits independent review and owner acceptance.**
It delivers M2's exit clause on traces in the shape its contract declared before
implementation: **a whole working set, out of device memory, with byte and cost
traces reconciled against the resource ledger as named equalities.**

**It does not close M2.** Closing M2 is the owner's, and the milestone's exit has
other clauses that other tasks discharged; what this task adds is the one nothing
had done.

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Contract `840b0e3`, written and committed **before** implementation; the
  implementation is the commits between it and the one this handover
  accompanies. Base before both was `aeac114`.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, re-verified; its untracked `.pi/`
  and `tests/p2p/` remain untouched. R02 is the source lesson this task is
  against and the frozen tree was not otherwise a reference for anything here.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was copied, converted, deleted, downloaded or
  modified.** One artifact's two shards were read: their headers, and
  **45,675,970,560 B of expert tensor payload per GPU pass**, through the
  accepted bounded ranged reader.

## Completed facts

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy (`--features moxie-executor/driver`) | passed |
| `cargo test --workspace --locked --offline` | **923 passed, 0 failed** (918 at `aeac114`) |
| Device-feature workspace tests | **958 passed, 0 failed** (951 at `aeac114`) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | zero failures, 79 rejected fixtures, 21 accepted, 13 rules |

Baselines were **re-measured at `aeac114`** rather than quoted: **918** host
tests and **951** device-feature tests. Task 0022's own record says why: its
first attempt at the same comparison was taken on a half-built tree and read a
number that was wrong.

**Mutation measurement: 30 of 30 caught, 0 survivors**
([experiment 0004](../evidence/experiments/0004-task0023-whole-working-set-trace.md)).
The first measurement was **27 of 30**, and every one of the three survivors was
a defect rather than an opinion about test strength: a missing equality, an
unchecked counter, and a property with no fixture at all.

### What executed

**Every routed layer of the designated artifact, twice over, at two row shapes,
on all three GPUs.**

`whole-set-decode` — 30 layers, two rows of top-k 8, a distinct route per layer,
unions of twelve or thirteen experts, **4,543,807,488 B** demanded. Run in a
roomy cache and a tight one on each card. **Every layer's device slot buffer is
bitwise equal to the CPU candidate's over the same bytes, on every card.** The
roomy configuration produced an exact prediction on all 30 layers and the tight
one on none — 292 backpressure drains and 1,517 evictions per card — and the test
asserts both, so neither branch can quietly stop being reached.

`whole-set-full` — 30 layers, sixteen rows whose routes partition the expert set,
so each layer's union is all 128 experts: **45,675,970,560 B demanded and
admitted per card**, the artifact's entire routed expert payload, 88.5% of it and
**1.77 times** the largest card on this machine, through a device cache of
**190,316,544 B** — sixteen of one layer's 128 experts, one 240th of the set it
serves. 15,286 evictions per card, no OOM, and **10,813,440** BF16 slot
components bitwise equal across the three cards, which is also sm_86 against
sm_120.

**Those are the artifact's own bytes at their real offsets. The activations are
synthetic and the routes are written by the test** — `whole-set-full`'s is
*constructed* so that its union is the whole expert set, which no prompt
produced. Nothing here is model support, **no quality claim follows (O2)**, and
**no route-distribution claim follows either**.

**No performance claim.** Both lanes are debug builds, there is no baseline on
this machine, and the trace schema has **no duration field at any point**.
Elapsed times are printed beside the runs and labelled as what they are.

## Decisions

**A "whole working set" is a row batch's union over every layer it passes
through, and this task runs two of them rather than choosing.** Document 03
defines the quantity as the "union of required experts over a row batch"; one
layer's union is what task 0021 ran. A decode-shaped batch demands about 3.6 GB,
which *fits* every card here and is out-of-device-memory only because the cache
is capped — true, and weaker. A batch whose union is the whole expert set is
42.5 GiB against a 24 GiB card and needs no ratio argument to be oversized. Both
run so that neither carries the other's claim.

**"Reconciled" is thirteen named equalities, and every one of them can fail.**
Four tie the ledger to what is held, four tie the run's own record to the
authority's bytes, four are the planner's prediction, and one pins the schema.
The name is what a failure reports. A violation battery mutates one number of a
trace that reconciles and requires the equality that number belongs to be the one
reported: 21 mutations over all 13, including three against the lower-bound
branch, which fails on being *below* a bound rather than on differing from it.

**A prediction is an equality or a declared lower bound, and the planner says
which.** A layer whose whole live set fits the displaceable cache admits every
chunk once, so its prediction is an equality and one extra admitted byte is a
defect. A layer that does not can lose a chunk to eviction between a backpressure
refusal and the retry that follows, and admit it twice; what it can state exactly
is a lower bound. Both branches have an acceptance case, because an unreachable
branch is a stub.

**The trace counts nothing.** Every number is read from the residency authority,
the ledger, the plan or the run's own record of its own actions; a `LayerTrace`
is the **difference of two snapshots** of those. The `arch-check` rule that
rejects a second weight-residency owner now also rejects a second *byte* owner —
a crate outside `moxie-memory` that defines a `ByteFlow`, a `ScopeAccount` or a
byte counter of its own — with a fixture that is refused.

**What the ledger charges is compared by identity, not by arithmetic.** The
authority and the run name their reservations; the trace asks the *ledger* what
those cost and requires the total charged to equal it. Nothing recomputes a cap,
a control charge or an envelope, because a second arithmetic for the same bytes
is how two owners come to disagree while both look right.

### Three quantities the accounting did not have

The reconciliation could not be written without them, and each is a real gap
rather than a renaming:

1. **A device upload whose host source is already resident.** It saves a read of
   the chunk's whole length and was counted as **nothing** — neither hit nor
   miss nor any byte counter — so host reads and host admissions could not be
   compared at all.
2. **Joining a transfer already in flight**, as distinct from finding the bytes
   already there. Both are "no bytes moved" and they are different facts; an
   identity over them needs the two separated.
3. **Per-scope accounting.** `ResidencyStats` summed three cards into one
   `bytes_uploaded`. That is AGENTS.md's forbidden "their memory is one
   allocation" assumption written as arithmetic, and no per-tier reconciliation
   can be assembled from it. It is still there, and is now required to **equal**
   the sum over scopes — which is what stops the new accounting from becoming a
   second, divergent tally.

### Two defects this task's own tests found in this task's own work

**A byte count per expert cannot say where an expert is.** The snapshot the
planner is compiled against began as a set of resident experts, then as a byte
count per expert. Both are wrong for the same reason: an expert is more than one
chunk, the authority admits and evicts chunks *individually*, and the device can
hold one of an expert's chunks while the host holds the other. Whether an upload
can copy from the host instead of reading is a question about *which* chunk is
where, and no total can answer it. It is `ResidentChunks` now — a reading, per
chunk — and the planner still never learns what a role is.

**Exactness that ignored what a layer was counting on.** The first rule asked
only whether a layer's *admissions* fit the cache. A layer that predicted hits on
bytes its own admissions then evicted called its prediction exact and read them
again. The condition is coexistence: what it admits **and** what it expects to
find. The sweep found it; reading the rule did not.

## Remaining hypotheses and blockers

- **M2 is not closed, and closing it is the owner's.** What this task discharges
  is the trace clause and the whole-working-set execution behind it.
- **Nothing generates a token from either designated artifact.** What ran is each
  layer's routed expert **block**. Attention, the norms and the shared expert are
  not in this path, and nothing composes them into a graph. The 88.5% of the
  artifact that has now moved through the engine is exactly the 88.5% that is
  experts.
- **`whole-set-full` is not compared against the CPU candidate**, by declared
  exclusion with the arithmetic attached: 2.28e10 multiply-adds is about nine
  minutes of debug-build host compute per pass, and it would repeat a comparison
  `whole-set-decode` makes on every one of the 30 layers. What it is checked
  against instead is the three cards' bitwise agreement with each other.
- **Backpressure does not occur in `whole-set-full`**, and the count is printed
  rather than asserted: a queue four deep pins four of the sixteen experts that
  cache holds, so the authority never has to refuse. That is a property of the
  queue against the cache, not of the working set.
- **No performance claim, no measured crossover.** The amortisation threshold is
  still a declared policy parameter; at this artifact's expert size the default
  sends every expert to the CPU, and both device cases declare their own
  threshold and say so. M6 owns measuring one.
- **Quality is O2** for both families. O1 and O5 remain open.
- The README's "current state" section still describes M1.4/M1.5. It is stale and
  was stale before this task; AGENTS.md is the entry point and is current.

## Next task

To be written after this task's review. The obvious candidates, in the roadmap's
own order, are M3's importer for Laguna's asymmetric INT4 at group 32 with zero
points packed along the output axis, and the two named gap tasks Laguna's
attention tower needs — `softplus` output gating and the yarn rotary ramp —
neither of which may be guessed.
