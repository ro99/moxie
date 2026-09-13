# Handover — task 0023 implemented and corrected; M2's exit is one clause from complete

**Task 0023 is implemented and corrected after two rounds of independent review
— nine findings, three P1, all reproduced, all fixed, none disputed — and awaits
owner acceptance.**
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
| CUDA-lane clippy (`--features cuda`) | passed — **a third lane, declared by this task**; it was failing on two lints standing since task 0021 |
| `cargo test --workspace --locked --offline` | **930 passed, 0 failed** (918 at `aeac114`) |
| Device-feature workspace tests | **962 passed, 0 failed** (951 at `aeac114`) |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | zero failures, 79 rejected fixtures, 21 accepted, 13 rules |

Baselines were **re-measured at `aeac114`** rather than quoted: **918** host
tests and **951** device-feature tests. Task 0022's own record says why: its
first attempt at the same comparison was taken on a half-built tree and read a
number that was wrong.

**Mutation measurement: 39 of 39 caught, 0 survivors**, over six rounds
([experiment 0004](../evidence/experiments/0004-task0023-whole-working-set-trace.md)).
The first was 27 of 30; the third, after the first review's six fixes, was 34 of
36; the fifth, after the second review's three, was 38 of 39.
**Every survivor in every round was a defect or a missing fixture**, and two of
them were checks added in response to an earlier finding that nothing had yet
violated — a check added because something was found is not itself checked until
something violates it.

### What executed

**Every routed layer of the designated artifact, twice over, at two row shapes,
on all three GPUs.**

`whole-set-decode` — 30 layers, two rows of top-k 8, a distinct route per layer,
unions of twelve or thirteen experts, **4,543,807,488 B** demanded. Run in a
roomy cache and a tight one on each card. **Every layer's device slot buffer is
bitwise equal to the CPU candidate's over the same bytes, on every card.** The
roomy configuration produced an exact prediction on all 30 layers with **zero
evictions**, and the tight one an exact prediction on none with 1,517 evictions
and 292 backpressure drains; the test asserts both, so neither branch can quietly
stop being reached. Roomy is sized for the **whole step** rather than one layer,
which is the review's second finding: a prediction is an equality only when
nothing has to be evicted, and a one-layer cache evicts the previous layer to
admit this one.

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
produced. Nothing here is model support, **no quality claim follows**, and **no
route-distribution claim follows either**.

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

### The second round: three more, and one of them was my own test

1. **P1.** `close` still aborted: `reservation_ids()` returned a `Vec` built with
   infallible `push`, and the sixth injected failure took the process down.
   **Reserving a destination says nothing about a temporary the callee builds.**
   It returns a fixed array now.
2. **P1.** Reconciliation **allocated to report that it could not allocate** — a
   failed `try_reserve` turned into a `Discrepancy` by `format!`. The detail is a
   `Cow<'static, str>` and the allocation-failure path borrows.
3. **`Exact` was unsound under fragmentation**, the third wrong version of that
   rule: it compared totals, and admission needs a **contiguous** range. A 4,608
   B cache holding 3,840 B in three 256 B holes predicted 768 B of reads exactly
   and read 1,536.

**The first is half mine, and it is the one to carry.** The regression I wrote
for the previous round's P1 looped six times calling `while_failing(1, ...)`, so
**every iteration failed the first allocation** and the loop index only changed
the assertion message. An axis that was exercised and never varied — in a test
written *because* of a review, for a defect a review had just found. It sweeps
every position now, and the count of positions is measured rather than assumed.

**And a term no fixture varies**: the padding allowance the third fix introduced
is inert wherever chunks are whole alignment units, which is everywhere here, so
it has a pure-planner test of its own rather than a comfortable assumption.

### The first round, and what its six findings have in common

**Four of the six are one sentence: a comparison is worth nothing when both its
sides can come from the wrong place, or when neither is the quantity it names.**

1. **P1.** One injected allocation failure before a snapshot gave **SIGABRT**.
   That is task 0019's rule for the **fifth** time in this workspace, on a path
   added after the previous four. Every collection the trace builds is reserved
   fallibly now, none is a `BTreeMap` — which has no fallible insert and would
   abort however carefully the rest reserved — and the owners hand their numbers
   out through visitors that allocate nothing.
2. **The exact prediction was unsound**, for the third time and each time for the
   same reason: it reasoned about *this plan's share* of a cache, and eviction
   does not. A warm chunk this layer needed was older in LRU than an unrelated
   cached chunk, so the layer's own admissions evicted exactly the chunk it
   predicted a hit on: **3,840 B read against 3,072 B predicted exactly**.
   Exactness now means **nothing has to be evicted**, which is about the whole
   cache.
3. **An omitted layer reconciled.** Totals were derived from the records supplied
   and then re-summed from the same records; three layers ran, the middle one was
   dropped, and **35 checks passed** while 11,520 B of reads went unmentioned. A
   step needs its own boundary, and it has one.
4. **A different, empty ledger reconciled** a completed run with no charges at
   all. Which ledger is checked before what it says.
5. **The launch counter counted groups, not launches** — the device path submits
   one kernel per symbol — so an equality added *because* mutation testing found
   the counter unchecked was then written against the wrong quantity. Adding a
   check is not the same as checking the right thing.
6. **The allocation gate could not see a leak**, because it counted calls and a
   leak is one call whose bytes never come back. It measures live bytes now and
   **performs the leak substitution itself**.

**And a lane nobody had run.** Two clippy lints in `xtask/src/gpu.rs`, untouched
by this task and standing since task 0021, are fixed here rather than recorded as
pre-existing. They stood because the declared gates name a host clippy lane and a
`--features moxie-executor/driver` one, and **neither compiles `xtask`'s CUDA
code**. `cargo clippy --workspace --all-targets --features cuda` is a third lane
and is now one of this task's gates.

**"Reconciled" is seventeen named equalities, and every one of them can fail.**
Five tie the ledger to what is held — including **which** ledger, and whether
every reservation this step can name is in it. Five tie the run's own record to
the authority's bytes. Four are the planner's prediction, one pins the schema,
and two are the step's own boundary: its totals must be its layers', and its
layers must account for every byte that entered a cache inside it.
The name is what a failure reports. A violation battery mutates one number of a
trace that reconciles and requires the equality that number belongs to be the one
reported: **22 mutations over all seventeen**, plus three against the lower-bound
branch — which fails on being *below* a bound rather than on differing from it —
and three against a layer that did not finish.

**A prediction is an equality or a declared lower bound, and the planner says
which.** A layer that can be run **without evicting anything** admits every chunk
once and keeps every chunk it predicted a hit on, so its prediction is an
equality and one extra admitted byte is a defect. A layer that cannot can lose a
chunk between a backpressure refusal and the retry that follows, or lose a warm
chunk to its own admissions, and read it again; what it can state exactly is a
lower bound. The condition is about the **whole cache**, not this plan's share of
it — that was the review's second finding, and it was the third time this rule
had been written too narrowly. Both branches have an acceptance case, because an
unreachable branch is a stub.

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
- **The owner resolved O1–O5 on 2026-09-13, mid-task** (`1e927ef`, ADRs
  0017–0020), and it is worth reading before the next task rather than
  rediscovering. The three that touch this work: the designated artifact here,
  `gemma-4-26B-A4B-it`, is **explicitly not in the v1 catalog** and "remains M2's
  BF16 workhorse", so this whole working set is an engineering fixture at real
  scale; v1's acceptable quality loss is a **bit-identical repack** with the
  publisher's own quality accepted as-is; and storage/conversion is
  user-managed, with no agent bulk write without a task naming artifact,
  revision, size and retention — which this task did not need, having written
  nothing. **Laguna is number 8 of the ten**, at the revision task 0022
  recorded. **O6 and O7 stay open**, which is why every timing here is a
  diagnostic.
- Older records — AGENTS.md's earlier paragraphs and tasks 0018–0022 — still
  describe O1, O2 and O5 as open. Reconciling them is the owner's: rewriting
  another task's record to match a later ruling would erase what was true when it
  was written.
- The README's "current state" section still describes M1.4/M1.5. It is stale and
  was stale before this task; AGENTS.md is the entry point and is current.

## Next task

To be written after this task's review. The obvious candidates, in the roadmap's
own order, are M3's importer for Laguna's asymmetric INT4 at group 32 with zero
points packed along the output axis, and the two named gap tasks Laguna's
attention tower needs — `softplus` output gating and the yarn rotary ramp —
neither of which may be guessed.
