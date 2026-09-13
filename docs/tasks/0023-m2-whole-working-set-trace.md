# Task 0023 — M2 item 5's remainder and M2's exit: a whole working set's byte and cost trace, reconciled with the ledger

Status: proposed.

Roadmap **M2's exit**, in full: "real out-of-device-memory working set executes
without OOM or hidden allocations, matches the reference, and produces byte/cost
traces reconciled with the resource ledger. Exactly one production
weight-residency owner; no cache class in adapters. Demonstrate demand failure
cannot deadlock."

Four of those clauses are already discharged and are **not** reopened here: the
single residency owner and the deadlock demonstration are task 0020's, with an
`arch-check` rule rejecting a second owner; "no cache class in adapters" is the
same rule; and the reference match for one routed layer is task 0021's bitwise
gate against task 0019's FP64 oracle on all three GPUs.

What has never been done is the clause this task owns: **a whole working set**,
rather than one layer, executing out of device memory and producing byte and cost
traces that **reconcile with the resource ledger**. Task 0021 ran one layer of
the designated artifact. Task 0022 ran one layer at a second artifact's declared
shape. Thirty layers have never run, no trace type exists in this workspace, and
nothing has ever checked the engine's byte accounting against itself across a
whole step.

## Why this task exists in this shape

Three numbers in this repository are, today, *assumed* to agree and have never
been compared:

1. What the planner predicted it would move (`ExpertEnvelope`).
2. What the residency authority actually moved (`ResidencyStats`).
3. What the ledger charged for it (`Ledger::committed`).

Each is measured by its own tests and each is correct in its own terms. That is
precisely the configuration AGENTS.md already records failing four times over:
**a check that exists on one path and not on the neighbouring one.** Nothing in
the workspace asks whether the three describe the same bytes, and the
preparatory reading for this contract found that in one identified case they
cannot:

- `ExpertEnvelope::residency_demand_bytes` counts **only device-candidate
  groups that are not already resident**. A host-candidate group acquires its
  chunk through the same authority, into `Scope::Host`, and is predicted by
  nothing. A plan that sends every expert to the CPU — which
  [task 0021 measured as this artifact's default-policy outcome](0021-m2-expert-execution-plans.md)
  — predicts **zero** bytes and moves 118,947,840 of them.
- `ResidencyAuthority::resolve_host_source` reuses an already-resident host
  placement as an upload's source. That is a hit on the disk-read path: it
  saves a read of the chunk's whole length. **It is counted as nothing** —
  neither `hits` nor `misses` nor any byte counter — so host reads and host
  admissions cannot be reconciled against each other today.

Neither is a defect that a stronger assertion would have found, and neither is
reachable by mutating the code: they are *missing* quantities, not wrong ones.
They are found by asking what an equality between two independently maintained
numbers would require, which is what this task builds.

## Identity and authority

- Task ID 0023, milestone **M2 item 5's remainder and M2's exit gate**. Owner
  review required for acceptance; independent review required before that, per
  the standing practice on tasks 0019–0022.
- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`, base
  commit `aeac114`, working tree clean at authoring time. This contract is
  committed **before** implementation; the implementation is the commits between
  it and the handover that accompanies it.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, re-verified at authoring time; its
  untracked `.pi/` and `tests/p2p/` remain untouched. R02 is the source lesson —
  legacy's `ResidencyManager` "modelled accesses and bytes for a simulator while
  the model runtimes allocated for themselves", so the thing that knew the
  budget was not the thing that spent it. **A trace that cannot be reconciled is
  that failure with better formatting**, which is why the deliverable is an
  equality and not a report.
- Local checkpoint roots `/models` and `/fast/models` stay read-only inputs.
  Nothing under either may be copied, converted, deleted, downloaded or
  modified. This task reads **tensor payloads** of one artifact — bounded ranged
  reads through the accepted path, exactly as tasks 0020 and 0021 do.
- Requirements repaired: roadmap M2's exit; document 03's resource-ledger and
  admission section, in particular "an admission report must show physical
  capacity, already committed resources, reserved peak, and remaining headroom
  for every tier" and "counters must reconcile with transfer/allocation events
  and distinguish logical from physical bytes" (document 07).
- Required reading, done before this contract was written: document 03's
  resource-ledger, admission and weight-residency-lifecycle sections in full;
  document 07's benchmark matrix, cost-attribution and evidence-retention
  sections; the M2 exit paragraph of document 06;
  [the benchmark/result schema](../evidence/benchmarks/schema.md), whose
  `counters` block this trace is a declared subset of; tasks
  [0020](0020-m2-weight-residency-authority.md) and
  [0021](0021-m2-expert-execution-plans.md) for the accounting a trace must
  reconcile *against*; and
  [the gemma4 bring-up record](../models/gemma4.md#the-26b-a4b-moe-variant) for
  the designated artifact's geometry.
- Owner gates: **O1 (catalog), O2 (quality), O5 (storage/conversion) and O6
  (performance defaults) remain open and this task resolves none of them.**

## What a "whole working set" is, and why this one

Document 03 defines the quantity: "For MoE, estimate **union of required experts
over a row batch**, cache hits, misses, and available reuse. Do not multiply
active experts by batch rows when routes overlap." A working set is therefore a
property of a **row batch over the layers that batch passes through**, not of a
checkpoint. One layer's union is what task 0021 ran; the whole set is every
routed layer's union, in layer order, through **one** authority and **one**
ledger.

The designated artifact — `/fast/models/google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` — has **30 layers, every one of them
routed**, 128 experts at top-k 8, hidden 2,816 and `moe_intermediate` 704. One
expert is **11,894,784 B** in BF16; one layer's 128 experts are
**1,522,532,352 B**; all thirty layers' experts are **45,675,970,560 B**, which
is **88.5%** of the artifact and more than the **24 GiB** of the largest card on
this machine by a factor of 1.77.

Two row batches are run, and the pair is the deliverable rather than either one
alone. They are named cases, not settings; the decode batch is run in two cache
configurations, for the reason section 3 gives:

| Case | Row batch | Union per layer | Bytes demanded | What it is evidence for |
|---|---|---|---|---|
| **`whole-set-decode`**, in a roomy and a tight cache | 2 rows, top-k 8, a distinct route per layer | 8–16 experts, measured and recorded per layer | Σ over the layers, ≈3.6 GB | The reference clause: every layer's device answer is compared to the CPU candidate over the same bytes |
| **`whole-set-full`** | 16 rows whose routes partition the expert set | **exactly 128** | **45,675,970,560 B** | The out-of-device-memory clause: the artifact's entire routed weight payload streams through a cache one eighth of a single layer |

**Why `whole-set-full` is the honest unit for "out of device memory".** Its
demand set is not a restricted budget's artefact and needs no ratio argument: it
is 42.5 GiB of real expert weights against a 24 GiB card and a 16 GiB card. A
decode-shaped batch demands ~3.6 GB, which *fits* every card here, and is
out-of-device-memory only because the cache is deliberately capped — a true
statement, and a weaker one. Both are run so that neither has to carry the other's
claim.

**What is synthetic, stated before the results are.** The activations are
generated by the test and **the routes are written by the test**, in both cases.
`whole-set-full`'s route is chosen so that the union is the whole expert set —
row `r` selects experts `8r..8r+8` — which is a deliberate construction and not
a distribution any prompt produced. The **bytes are the artifact's own**, read
from its shards at their real offsets. Nothing here is model support, nothing
here is a quality claim, and **no route distribution claim follows** — a real
prefill batch's union is an M6 measurement, not this task's.

**What is not run, and why — with the arithmetic, not an adjective.**
`whole-set-full` is **not** compared against the CPU candidate. Its 3,840
row-expert products are 2.28 × 10¹⁰ multiply-adds; the existing real-artifact
case measured the debug host candidate at 20 such products in 2.76 s, so one
whole-set-full reference pass is **≈9 minutes** of debug-build scalar host
compute, which would roughly quintuple the device lane's total runtime. What it
would buy is a repeat of a comparison already made: the per-group kernel is gated
**bitwise** against task 0019's FP64 oracle for both gate transforms on all three
GPUs (task 0021), and against the host candidate over this artifact's real bytes
on **every one of the 30 layers** by `whole-set-decode`. What `whole-set-full`
adds numerically — a 16-row, 128-expert composition with one row per group — is
gated instead by **bitwise equality of all three cards' slot buffers to each
other**, which costs nothing and also checks sm_86 against sm_120. This is a
declared exclusion with a named substitute, not an omission, and it is written
here before the run rather than after it.

## Bounded deliverable

**One concrete outcome:** a byte-and-cost trace, produced by running a whole
working set of the designated artifact on real hardware, that **reconciles as a
set of named equalities** against the residency authority's own accounting and
the resource ledger's own charges — with every equality individually violable and
individually tested.

**Sole owning shared components**, and what each gains:

| Crate | What it owns here | What it may not gain |
|---|---|---|
| `moxie-memory` | The byte trace's **source**: per-scope accounting inside `ResidencyAuthority`, and the conservation identities in `check_invariants` | A second counter for anything it already counts; any knowledge of layers, plans or models |
| `moxie-plan` | The **prediction**: an envelope that predicts every tier the trace reconciles, including the host demand it does not predict today | Live-resource access; it stays pure |
| `moxie-executor` | The run's **cost record**, and the trace **assembly and reconciliation** — document 02: "`executor` schedules actual effects and updates measured costs" | A byte counter of its own for anything the authority counts |
| `moxie-models`, `moxie-engine` | **Nothing.** No model-owned path may appear | — |

**Explicit non-goals and forbidden shortcuts:**

- **The trace is not a benchmark and carries no wall clock.** Document 07 splits
  profiling mode from benchmark mode; this is neither. No field of the trace
  schema is a duration, and the acceptance gates contain no timing comparison.
  Elapsed times may be *printed* beside a run, labelled as debug-build diagnostics
  exactly as tasks 0021 and 0022 print them, and may not enter the record as a
  result.
- **A field that was not measured says so.** The trace's schema is a declared
  subset of the result schema's `counters` block; every field of that block that
  this task does not measure is present as `unmeasured` and **never as zero**.
  `d2h_bytes: 0` and `d2h_bytes: unmeasured` are different claims.
- **No estimate may enter a trace.** Every number is read from the authority,
  the ledger, the plan or the run. A quantity that must be derived is derived by
  arithmetic over those numbers and is labelled as derived.
- **No second accounting owner.** The trace type holds snapshots of other
  owners' counters and differences between snapshots. It observes no event and
  increments no counter of its own. An `arch-check` rule is extended to say so.
- Not an importer, not a graph, not a model. No attention, no tokens, no
  generation loop: a routed block per layer is what exists and what runs.
- **No performance claim, no quality claim (O2), no bulk write (O5), no
  execution of remote model code.**

**Existing consumers that must keep passing unchanged:** every test task 0022
left green, including both sweeps at their extended products (10,368 and 288
combinations), the restricted-budget device case on all three cards, and the
allocation-substitution battery at 25 repetitions in both directions.

## Contract before implementation

### 1. The byte trace's source: per-scope accounting

`ResidencyStats` is process-wide today: one `bytes_uploaded` for three devices,
one `bytes_read`, one `hits`. A trace that reconciles per tier cannot be built
from it, and summing three devices into one counter is the "their memory is one
allocation" assumption AGENTS.md forbids.

`ResidencyAuthority` gains **per-scope** accounting, and this is the one place
these bytes are counted. Cumulative counters, plus levels computed from the
placements when the account is read — a level that is derivable is derived, so
there is nothing to keep in sync:

```text
ScopeAccount {
    // what was asked of this scope
    requested_bytes          every acquire's chunk length, whatever it resolved to
    hit_bytes                resolved by a ready placement already here
    coalesced_bytes          resolved by joining a transfer already in flight
    refused_bytes            resolved by a refusal

    // what was placed here, split by where the admission came from
    admitted_bytes
    direct_admitted_bytes    admitted for an acquire naming this scope
    as_source_admitted_bytes admitted as the host source of a device upload
    source_created_bytes     a device admission that had to create a host source
    source_reuse_bytes       a device admission whose host source was already here

    // what moved
    read_bytes               an observed read completed (host scopes)
    read_abandoned_bytes     a read that failed or was cancelled
    uploaded_bytes           an observed upload completed (device scopes)
    upload_abandoned_bytes   an upload that failed or was cancelled

    // what left
    evicted_bytes
    retired_bytes
    discarded_bytes

    // levels, computed at read time
    resident_bytes           == cache.committed; includes quarantined
    quarantined_bytes
    in_flight_bytes          Reading or Uploading
    peak_resident_bytes      the one maintained level: a high-water mark
}
```

`ResidencyStats` stays exactly as it is, and is required to equal the sum over
scopes of the corresponding account fields. That equality is itself one of the
checked identities: it is what stops the new accounting from becoming a second,
divergent tally of the same events.

**Three quantities that do not exist today and are the reason this is a task
rather than a report**: `source_reuse_bytes` (see "Why this task exists");
`coalesced_bytes`, which separates "this acquire moved nothing because the bytes
were here" from "this acquire moved nothing because someone else is already
moving them"; and `as_source_admitted_bytes`, because a device acquire admits
**two** placements and today nothing distinguishes the host half of a device
upload from a host-candidate group's own chunk.

### 2. The conservation identities

These hold **after every operation**, not only at the end, and are checked inside
`ResidencyAuthority::check_invariants` so that task 0020's 800-combination
transition sweep exercises them over its whole product without a new harness:

| Name | Identity |
|---|---|
| `request-is-hit-coalesced-admitted-or-refused` | per scope: `requested == hit + coalesced + admitted + refused` |
| `admission-has-one-origin` | host scope: `admitted == direct_admitted + as_source_admitted`; device scope: `admitted == source_created + source_reuse` |
| `a-source-is-admitted-where-it-lives` | `Σ device scopes' source_created == host.as_source_admitted` |
| `admitted-is-resident-or-gone` | per scope: `admitted == resident + evicted + retired + discarded` |
| `host-admission-is-one-read` | host scope: `admitted == read + read_abandoned + reading_level` |
| `device-admission-is-one-upload` | device scope: `admitted == uploaded + upload_abandoned + pending_level` |
| `stats-are-the-sum-of-scopes` | `ResidencyStats::bytes_read == Σ host read`, `bytes_uploaded == Σ device uploaded`, `evicted_bytes == Σ evicted` |
| `resident-equals-placements` | `resident == Σ placements == cache.committed == arena.live_bytes` (this one exists; it is restated so the set is complete) |

Every identity is an **equality**. None is a bound, a tolerance or a comparison
of two approximations. Where a quantity is genuinely in flight it appears as a
named level (`reading_level`, `pending_level`) rather than as slack, and a
quarantined placement is **inside** `resident` rather than beside it — charged is
charged, and counting it twice to make a sum work would be the arithmetic error
this task exists to make impossible.

### 3. The prediction: what the envelope must predict, and where it honestly cannot

`ExpertEnvelope` gains what it does not predict today, and `ExpertBudget` gains
the snapshot field needed to predict it:

- `ExpertBudget::host_resident_experts` — experts whose chunks the snapshot found
  already resident in the **host** cache. The device field exists; its host
  counterpart does not, so host reads are unpredictable today.
- `ExpertEnvelope::predicted` — a per-scope prediction using the same field names
  the trace reconciles against: `device_upload_bytes` (what
  `residency_demand_bytes` means today, kept and documented as that),
  `host_read_bytes`, `host_source_reuse_bytes`, `device_hit_bytes`,
  `host_hit_bytes`.

`residency_demand_bytes` is **kept**, with its meaning stated in its
documentation rather than implied by its name, because task 0021's admission path
uses it and a rename is not this task's business.

**A prediction is exact or it is declared not to be, and the planner knows
which.** If a layer's whole union fits the displaceable cache, no admission can
evict anything, every chunk is admitted at most once, and the prediction is an
**equality**. If it does not fit, a group whose acquire is refused under
backpressure retries after the queue drains, and by then its partner chunk may
have been evicted and must be admitted a second time. The planner has both
numbers — the union and `ExpertBudget::displaceable()` — so `predicted` carries
`Exactness::{Exact, LowerBound}` computed from them, with the reason.

This is not a tolerance and it is not slack. In the `Exact` case the trace
asserts equality and a single extra admitted byte fails it. In the `LowerBound`
case the trace asserts the declared relation **and** an exact cross-owner
equality that holds in both (`requests-are-attempts`, below), so no case is left
checking only an inequality. **Both branches are reached by a named acceptance
case**, because an unreachable branch is a stub — task 0021's fourth review found
one of its sweep's advertised axes unreachable, and this contract is written to
make that failure impossible to repeat here.

### 4. The trace and its schema

```text
StepTrace {
    schema_version: u32          // this record's own version, checked in a test
    artifact: ArtifactId
    case: &str                   // "whole-set-decode" | "whole-set-full"
    rows: u64
    layers: Vec<LayerTrace>
    totals: StepTotals           // = Σ layers, and that is a checked equality
    unmeasured: &[&str]          // named fields this case did not measure
}

LayerTrace {
    layer: u32
    predicted: EnvelopePrediction         // from moxie-plan
    scopes: Vec<(Scope, ScopeAccountDelta)> // from moxie-memory, before/after
    cost: GroupedStats                    // from moxie-executor
    ledger: Vec<(Scope, Tier, u64)>       // from moxie-memory's Ledger
}
```

A `ScopeAccountDelta` is the difference of two snapshots of the authority's own
account, taken around one layer. **It is a difference, not a tally**: the trace
never sees an acquire. This is legitimate only because one interactive
generation is a product gate — a second concurrent consumer of the same authority
would make a delta unattributable, and the contract says so where the type is
defined.

**Granularity is per layer, per scope, per tier.** Not per chunk: a 3,840-row
chunk table is a raw trace, and document 07 puts raw traces outside git with a
content hash. `whole-set-full` writes its per-chunk detail to `results/task0023/`
with its sha256 recorded in the evidence record; the per-layer trace is what the
record carries.

### 5. What "reconciled with the resource ledger" means

M2's exit asks that the traces reconcile **with the resource ledger**. That is
the first group below and it is exact in every case. The second group ties the
run's own record to the same bytes, and the third is the planner's prediction,
exact or declared as a bound per section 3. Each equality is named, and the name
is what a failure reports, together with the layer, the scope and both sides.

**Ledger reconciliation — exact, every case, at every layer boundary and at step
end:**

| Name | Equality |
|---|---|
| `ledger-charges-what-is-held` | for every scope and tier: `Ledger::committed(scope, tier)` equals the authority's cache reservation for that tier plus the live run's envelope for that tier — **there is no third charger**, and the equality is what says so |
| `cache-cap-is-the-reservation` | every scope's `resident_bytes ≤ cap_bytes`, and `cap_bytes` is what the ledger admitted, not a number held beside it |
| `nothing-outstanding` | at step end: `Ledger::outstanding().is_empty()`, every account's `resident_bytes == 0`, and `admitted == retired + evicted + discarded` |
| `step-is-the-sum-of-layers` | `totals == Σ layers`, field by field |

**Run reconciliation — exact, every case:**

| Name | Equality |
|---|---|
| `requests-are-attempts` | `Σ scopes' requested_bytes` for the layer `== gate_up_attempts · gate_up_bytes + down_attempts · down_bytes`, where the attempt counts are the run's record of **its own actions** and the byte lengths are the plan's. It catches an acquire issued against the wrong scope, a retry that was not counted, and a chunk length that disagrees with the plan |
| `run-ran-the-plan` | `cost.groups_run == plan.groups().len()` and `host_groups + device_groups == groups_run` |
| `slots-are-written-once` | `cost.slots_written == rows · top_k` |
| `leases-balance` | `cost.leases_acquired == cost.leases_released`, and the authority's `live_lease_count() == 0` at layer end |

**Prediction reconciliation — exact where the planner declared `Exact`, the
declared relation where it declared `LowerBound`:**

| Name | Relation |
|---|---|
| `predicted-uploads-are-uploaded` | `predicted.device_upload_bytes` vs `Σ device scopes' admitted_bytes` |
| `predicted-reads-are-read` | `predicted.host_read_bytes` vs host `read_bytes` |
| `predicted-hits-are-hit` | `predicted.device_hit_bytes` and `predicted.host_hit_bytes` vs the accounts' `hit_bytes` |
| `predicted-source-reuse-is-reused` | `predicted.host_source_reuse_bytes` vs device `source_reuse_bytes` |

**Reconciled means these hold exactly, in the sense each row states.** A trace
that is merely *printed* is not reconciled; a `reconcile()` that can only succeed
is a stub. Every equality gets a fixture that violates it and a test that names
it, and the mutation battery measures that deleting the check makes that fixture
pass.

### 6. Cancellation, failure and rollback

The trace is produced by the run that already has these contracts and adds no
new ones. What this task adds is that they are **visible in the trace**:

- A cancelled layer's trace records the bytes its cancellation gave back
  (`discarded_bytes`) and the bytes it withheld (`quarantined_bytes`), and the
  conservation identity still holds. A trace is produced for a run that failed:
  "no trace" and "a trace showing the failure" are different outcomes and the
  first is not acceptable.
- `StepTrace::reconcile` returns a typed `Discrepancy` naming the equality, the
  layer, the scope and both sides. It never panics: an accounting mismatch inside
  a generation step must be reportable, not fatal — the rule task 0020
  established for the same reason.

### 7. Hidden allocations

M2's exit says "without OOM or hidden allocations". Task 0022's thread-local
allocation counter is the instrument. This task asserts, on the **host** lane
over a synthetic whole set: the per-layer steady state allocates a bounded,
**declared** number of times, and the trace's own assembly allocates only its
declared per-layer record. A number is asserted, not an absence — the counter
prints what it saw.

## Acceptance

### Gates

| Gate | Requirement |
|---|---|
| `cargo fmt --all -- --check` | passes |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passes |
| Device-lane clippy (`--features moxie-executor/driver`) | passes |
| `cargo test --workspace --locked --offline` | passes; count reported against the measured baseline at `aeac114` |
| Device-feature workspace tests | passes; count reported against the same baseline |
| `cargo xtask-cuda test-gpu` | passes, zero skipped; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passes |
| `cargo xtask arch-check` | **zero failures**, with the extended rule's negative fixture rejected |

Baselines are **re-measured in an isolated worktree at `aeac114`**, not quoted
from the previous handover. Task 0022's own record says why: the first attempt at
that comparison was taken on a half-built tree and read a number that was wrong.

### The whole-working-set cases, on real hardware

| Case | Device cache | Host cache | Required outcome |
|---|---|---|---|
| `whole-set-decode/roomy` | ≥ the layer union | ≥ the layer union | Planner declares **`Exact`**; every prediction equality holds as an equality; zero evictions, zero backpressure drains — the branch where a prediction is a prediction |
| `whole-set-decode/tight` | `ceil(layer union / 4)` | `ceil(layer union / 2)` | Planner declares **`LowerBound`**; evictions and backpressure drains both non-zero; `requests-are-attempts` and every ledger and conservation identity still exact |
| both decode configurations | | | Every one of the 30 layers' device slot buffers **bitwise equal** to the host candidate's over the same bytes, and equal to each other — the same plan on the same bytes may not depend on how much cache it was given |
| `whole-set-full` | `ceil(1,522,532,352 / 8)` = 190,316,544 B | `ceil(1,522,532,352 / 4)` = 380,633,088 B | **45,675,970,560 B** demanded and admitted; all ledger, run and conservation identities exact; no OOM; all three cards' slot buffers bitwise equal to each other; the per-layer allocation count is the declared one |

All three run on **all three GPUs** and are **skipped loudly** — never silently — when
the artifact is absent. `whole-set-full`'s device cache holds **16 of 128**
experts of a single layer and **one 240th** of the set it serves.

### Host-lane coverage, and how its strength is measured

- A **reconciliation sweep** over synthetic fixtures, enumerating the product of:
  candidate mix × resident-set (empty / device-resident / host-resident /
  both) × cache ratio × failure point (none / read / upload) × cancellation ×
  layer count. The invariant set is checked after **every** operation, the
  product is **printed** by the test rather than asserted in prose, and the sweep
  performs only work the scheduler handed out — task 0020's two rules, both
  learned the hard way.
- **One violating fixture per equality**, nine in total, each asserting the
  `Discrepancy` names that equality.
- **Mutation measurement, reported as a number**: a mutation battery over the new
  accounting and reconciliation, with survivors named and equivalent mutants
  reported as equivalent rather than counted as gaps. A survivor is fixed before
  acceptance, or recorded as an equivalent mutant with the argument.
- **Substitution testing of every new regression**, repeated **25 times in both
  directions**, because task 0022's third round established that a substitution
  over a flaky test is a coin flip recorded as a measurement.
- **A parameter's coverage is claimed only with the fixture that varies it**
  named. For this task the parameters at risk are: the resident sets (both of
  them), the candidate mix, and the cache ratio. `whole-set-full` and
  `whole-set-decode` differ in row count and union size, so those two vary as
  well.

### Documentation gates

- [The support matrix](../evidence/support-matrix.md) gains the two cases with
  their gate IDs, and says **tested / unmeasured / unsupported** rather than
  implying support.
- An experiment record under `docs/evidence/experiments/` carries the mutation
  measurement and any null result.
- [The gemma4 bring-up record](../models/gemma4.md) records what executed, in the
  same shape as task 0021's entry, and repeats that **no quality claim follows**.
- A handover states M2's exit position clause by clause: what this task closes,
  and what M2 still does not have.
- This contract's Result section is filled in, including failures and anything
  skipped, reported separately.

### The exact condition requiring owner direction or task rejection

- **If any reconciliation equality cannot be made to hold**, stop at that
  equality with the trace that shows it, and report the smallest decision needed.
  Do not weaken an equality into a bound, do not add slack, and do not remove a
  term to make both sides agree — the disagreement is the finding.
- If the whole-set run cannot complete on this machine for a **resource** reason
  (not a defect), stop with the admission report that refused it. Document 03
  requires a refusal to carry a breakdown and legal alternatives; that refusal is
  a result, and shrinking the working set to get a pass is not.
- If M2's exit requires anything this contract has not named, say so in the
  handover rather than expanding scope silently.

## Result, filled after work

To be completed.
