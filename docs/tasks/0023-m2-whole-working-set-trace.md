# Task 0023 — M2 item 5's remainder and M2's exit: a whole working set's byte and cost trace, reconciled with the ledger

Status: implemented; awaiting independent review and owner acceptance.

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
- Owner gates **at authoring time**: O1, O2, O5 and O6 were open and this task
  resolves none of them. **The owner resolved O1–O5 on 2026-09-13 while this task
  was being implemented** (`1e927ef`, ADRs 0017–0020); what that changes for this
  task is recorded in the Result section, and it changes nothing this task did.

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
- **No performance claim, no quality claim, no bulk write, no execution of
  remote model code.** (Written against O2 and O5 as open gates; both were
  resolved during implementation and neither ruling makes any of these
  permissible — see the Result section.)

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

Status: **implemented 2026-09-13; awaiting independent review and owner
acceptance.**

### Changed shared owners and consumers

| Crate | What it gained |
|---|---|
| `moxie-memory` | `ByteFlow` and `ScopeAccount`: per-scope byte accounting inside `ResidencyAuthority`, and eight conservation identities inside `check_invariants`. `ResidencyStats` is unchanged and is now **required to equal** the sum over scopes. `reservation_ids()` names what the authority charged, so a trace can ask the ledger what it cost instead of recomputing it |
| `moxie-plan` | `ResidentChunks` (a per-chunk reading of both caches), `EnvelopePrediction` and `Exactness` on `ExpertEnvelope`, and `host_cache_cap_bytes` / `host_cache_leased_bytes` on `ExpertBudget`. `residency_demand_bytes` is unchanged and is now documented as what it is |
| `moxie-executor` | `trace`: `LayerSnapshot`, `LayerTrace`, `StepTrace`, `Reconciled` and `Discrepancy`. `GroupedStats` gained `acquires_issued` and `launches` — counts of the run's own actions, never bytes — and `GroupedRun::reservation_id` |
| `xtask` | The `a second weight-residency owner` rule rejects a second **byte** owner too, with a rejected fixture |
| `moxie-models`, `moxie-engine` | **nothing**, as the contract required |

### Commands and results

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| Device-lane clippy (`--features moxie-executor/driver`) | passed |
| CUDA-lane clippy (`--features cuda`, which compiles `xtask`'s device code) | passed — **a third lane, declared by this task**, and it was failing on two lints standing since task 0021 |
| `cargo test --workspace --locked --offline` | **930 passed, 0 failed**, against a re-measured baseline of **918** at `aeac114` |
| Device-feature workspace tests | **965 passed, 0 failed**, against a re-measured baseline of **951** at `aeac114` |
| `cargo xtask-cuda test-gpu` | **42 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | passed, 10 documents |
| `cargo xtask arch-check` | **zero failures**, 79 rejected fixtures, 21 accepted, 13 rules |

Nothing failed. Nothing was skipped except the two device cases when the
artifact is absent, which print `SKIPPED` with the reason and do not run here —
the artifact **is** present on this machine and both ran.

### The two whole-working-set cases, on real hardware

| Case | Demanded | Result |
|---|---|---|
| `whole-set-decode`, roomy and tight, each on all three GPUs | **4,543,807,488 B** over 30 layers | All 30 layers **bitwise equal** to the CPU candidate over the same bytes, on every card and in both configurations. Roomy: exact prediction on all 30 layers, 0 backpressure drains, 1,487 evictions. Tight: exact on none, 292 drains, 1,517 evictions. 3,828 equality checks across the three cards |
| `whole-set-full`, on all three GPUs | **45,675,970,560 B** over 30 layers | Every byte demanded **and admitted** on each card through a 190,316,544 B device cache — sixteen of one layer's 128 experts, one 240th of the set. 15,286 evictions per card, no OOM. **10,813,440** BF16 slot components bitwise equal across the three cards, which is sm_86 against sm_120 |

The CPU candidate for the decode case ran **once** — 83.28 s for 30 layers in a
debug build — because its answer does not depend on which card the device
candidate used. Device passes were 1.99–2.21 s each and `whole-set-full` 20.24 s,
20.63 s and 22.02 s. **None of those is a benchmark**: debug builds, one run
each, no baseline, and no duration appears anywhere in the trace.

### Host-lane coverage

- The **trace sweep** enumerates 64 combinations and traces **192** layers,
  checking **4,900** equalities of which **862** are the declared lower-bound
  form -- 192 layers rather than 128, because a warm pass is a layer execution
  and the step's boundary counts its bytes whether the test does or not. 66
  layers produced an exact prediction, 126 a bound, and 16 did not finish
  — each of those traced as incomplete, with its prediction skipped and counted
  as skipped. The product and the census are **printed** by the test.
- The **violation battery** mutates one number of a trace that reconciles and
  requires the equality that number belongs to to be the one reported: **22
  mutations over all seventeen equalities**, plus 3 against the lower-bound branch
  and 3 against a layer that did not finish -- **28** in all. Four equalities also
  have a dedicated end-to-end regression built from the review's own
  reproducers: a foreign ledger, an omitted layer, a warm chunk evicted by its
  own layer, and a charge nobody can name.
- The **planner sweep** gained a host-residency axis and then, after the review,
  a `crowded` axis -- whether the caches already hold bytes this route does not
  name, which is the condition its second finding turned on. 10,368 combinations
  became **124,416**, and every plan's prediction is checked against a statement
  of the rule written independently of the planner's.
- The **transition sweep** exercises the new conservation identities over its
  whole 800-combination product without a new harness, because they live in
  `check_invariants`.
- The **allocation gate** is a number: **235** heap requests per traced layer,
  identical from layer 2 to layer 8, so a step's heap is bounded by one layer's
  record however long the step runs. Assembling an 8-layer `StepTrace` costs 6.
  **And live bytes**, after the review: at most **348 B** retained per layer
  against a declared 4 KiB bound, with the gate performing its own substitution —
  a deliberate 1 MiB per-layer leak retains 1,048,924 B and is rejected while the
  call count stays flat at 225, which is why the byte measurement exists.

### Measured effect and uncertainty

- **Mutation measurement: 41 of 41 caught, 0 survivors**, over seven rounds
  ([experiment 0004](../evidence/experiments/0004-task0023-whole-working-set-trace.md)).
  The first was 27 of 30; the third, after the first review's six fixes, was 34 of
  36; the fifth, after the second review's three, was 38 of 39. **Every survivor in every round was a defect or a missing fixture**, never
  an opinion about test strength, and two of them were checks added in response
  to an earlier finding that nothing had yet violated: a check added because
  something was found is not itself checked until something violates it.
- **A null result:** the sweep's failing-read axis caught no mutation the other
  axes did not. What it produced is the incomplete-layer branch of the
  reconciliation and the three fixtures that branch needed.
- **No performance claim, no quality claim, no route-distribution claim.** The
  amortisation threshold remains a declared policy parameter; both device cases
  raise it and say so.

### Two defects this task's own tests found in this task's own work

**A byte count per expert cannot say where an expert is.** The planner's
residency snapshot began as a set of experts, then as a byte count per expert.
Both are wrong for the same reason: an expert is more than one chunk, the
authority admits and evicts chunks individually, and the device can hold one of
an expert's chunks while the host holds the other. Whether an upload can copy
from the host instead of reading is a question about **which** chunk is where.
`ResidentChunks` is a per-chunk reading, and the planner still never learns what
a role is.

**Exactness that ignored what a layer was counting on.** The first rule asked
only whether a layer's admissions fit the displaceable cache. A layer that
predicted hits on bytes its own admissions then evicted called itself exact and
read them again. The condition is coexistence — what it admits **and** what it
expects to find — and the sweep found it where reading the rule did not.

### Where the implementation differs from this contract, and why

A contract written before implementation is a prediction too. Six things above
are not what was built, and each is recorded rather than quietly reconciled.

1. **`ScopeAccount` gained `unreported_bytes` and `source_rollback_bytes`, and
   the transfer identity uses the first instead of a `withheld` term.** A
   withheld transfer (R07) has reported **nothing**, so counting it as an
   outcome makes one term mean two things; it is a level, and a placement counts
   exactly once — through the outcome its transfer reported, through the level if
   nothing has reported, or through `unfinished_bytes` if it left first. The
   rollback term is the one way the two scopes' counts of a device acquire
   legitimately differ: the host admitted the source, the device's admission was
   then refused, and only the host ever saw the bytes.
2. **Identity A counts `as_source_admitted` on the asked side.** A host cache
   whose only traffic is other scopes' upload sources has a request count of zero
   and a real, charged working set; the identity as the contract wrote it was
   false for exactly that cache.
3. **`predicted-hits-are-hit` is two-sided in the bound case.** Eviction can turn
   a predicted hit into an admission and a backpressure retry can produce a hit
   nobody predicted, so a one-sided bound is unsound in *both* directions. What
   is sound is `hit + evicted >= predicted` and `hit <= predicted + retried`,
   and both terms are quantities the trace already records.
4. **Exactness is one flag, not one per side.** `BoundOn` still names the cache
   that could not hold the layer, and it is diagnostic: a device chunk admitted
   twice asks the host for its source twice, so a bound on either cache is a
   bound on both predictions. The first version treated them independently and
   the sweep found it.
5. **`ExpertBudget::host_resident_experts` does not exist.** It became one
   `resident: ResidentChunks` covering both caches, per chunk, for the reason in
   "Two defects" above: no per-expert total can say *which* chunk is where.
6. **The per-chunk raw trace is not written to `results/`.** The contract
   proposed it with a content hash. Nothing consumes it, and the authority keeps
   no per-chunk history to write — by design, since the trace counts nothing. An
   unread file with a hash beside it is not evidence, so it was dropped rather
   than produced; `ResidencyAuthority::outstanding` still names every live
   placement for diagnosis.

**The count of equalities is fourteen, not the nine this contract's acceptance
section names.** That line is an inconsistency in the contract itself: section 5
above already lists thirteen in three groups, and the acceptance section's "nine
in total" is left over from an earlier draft of it. What the requirement meant is
unchanged and is met — **one violating fixture per equality, whatever the count
turns out to be** — and the count is printed by the test rather than asserted in
prose, which is the only reason the discrepancy is visible at all.

Two additions arrived from the mutation measurement rather than from the
contract: a fourteenth equality (`launches-match-device-groups`), and the
incomplete-layer branch — a layer that did not finish is traced, reconciled and
counted as incomplete, which the contract asked for and the first implementation
dropped.

### The independent review, and the six findings it reproduced

**Six findings, one P1, all reproduced, all fixed, none disputed.** Four of them
are one sentence: **a check that compares two numbers is worth nothing when both
can be read from the wrong place, or when neither is the number it names.** The
review's own reproducers ran here before anything was changed, and every one of
them is a regression now.

| # | Finding | What it measured | Fix |
|---|---|---|---|
| 1 | **P1 — a trace that cannot allocate aborts the process.** One injected failure before `LayerSnapshot::take` | **SIGABRT**, `memory allocation of 864 bytes failed` | Every collection the trace builds is `try_reserve`d; none is a `BTreeMap`, which has no fallible insert; the owners hand their numbers out through visitors that allocate nothing; `new` takes owned strings, because converting a `&str` inside it would allocate infallibly. `LayerSnapshot` and `StepSnapshot` are **no longer `Clone`** |
| 2 | **The "exact" prediction was unsound.** A warm chunk this layer needs, older in LRU than an unrelated cached chunk | **3,840 B read against 3,072 B predicted exactly** | A prediction is exact only when **nothing has to be evicted**: what the cache already holds plus what this layer admits fits the cap. `ExpertBudget` carries the whole cache's resident bytes, not this plan's share of it |
| 3 | **Reconciliation accepted an omitted layer.** Totals derived from the records supplied, then re-summed from the same records | 3 layers ran and read **11,520 B**; dropping the middle one **passed 35 checks** | `StepSnapshot` taken before the first layer gives the step its own boundary, and `step-covers-every-byte` requires the layers to account for every byte that **entered** a cache inside it |
| 4 | **An unrelated empty ledger passed.** Reservations looked up in the supplied ledger, with no identity check | A completed run **reconciled with no recorded charges at all** | `ledger-is-the-runs-own` compares the ledger the trace read against the one the run was admitted to, and `every-reservation-is-charged` requires every reservation this step can name to be present before any charge is compared |
| 5 | **The launch counter counted groups, not launches.** The device path submits one kernel per **symbol** — a projection and a reduction | The equality validated a quantity wrong by a factor of two, and a group that failed between the two symbols reported zero | The lane reports what it submitted, `LaunchRefused` carries the count of a group that failed part way, and the equality is against the descriptor's own symbol count |
| 6 | **The allocation gate could not see a leak.** It counted allocator calls, not bytes | A deliberate **1 MiB per layer** leak passed at 236 calls per layer | The gate measures live bytes too, declares what a layer may retain (its own trace record), and **performs the leak substitution itself**: 1,048,924 B retained per layer is rejected while the call count stays flat at 225 |

**What the six have in common is worth more than any one of them.** Findings 3
and 4 are the same defect in two places: a comparison whose two sides come from
the same place cannot fail. Finding 2 is the third time this task's exactness
rule was wrong, and each time for the same reason — it reasoned about *this
plan's share* of a cache, and eviction does not. Finding 5 is a check that was
added **because** mutation testing found the counter unchecked, and was then
written against the wrong quantity: adding a check is not the same as checking
the right thing. Finding 1 is the allocation-failure discipline for the **fifth**
time in this workspace, on a path added after the previous four.

**A lane nobody had run.** The review also found two clippy lints in
`xtask/src/gpu.rs`, untouched by this task and last changed at `94d6cfb` (task
0021). They are real and they are **fixed here** rather than recorded as
pre-existing: AGENTS.md says a standing failure is how a real one gets missed.
The reason they stood is that the declared gates name two clippy lanes, host and
`--features moxie-executor/driver`, and **neither compiles `xtask`'s CUDA code**.
`cargo clippy --workspace --all-targets --features cuda` is a third lane and is
now one of this task's gates.

### The second review round: three more, and one of them was my own test

**Three findings, two P1, all reproduced, all fixed, none disputed.**

| # | Finding | What it measured | Fix |
|---|---|---|---|
| 1 | **P1 — `close` still aborted.** `reservation_ids()` returned a `Vec`, built with infallible `push` | **SIGABRT**, 32 bytes, on the **sixth** allocation of `close` | It returns a fixed `[Option<ReservationId>; 2]`, and `accounted_charges` keeps its `named`/`found` sets in fixed arrays. **Reserving a destination says nothing about a temporary the callee builds** |
| 2 | **P1 — reconciliation allocated to report that it could not allocate.** A failed `try_reserve` was turned into a `Discrepancy` by `format!` | **SIGABRT**, 82 bytes | `Discrepancy::detail` is a `Cow<'static, str>`; the allocation-failure path carries a borrowed message. Everything a caller acts on — the check's name, both sides, the layer, the scope — was already beside it |
| 3 | **`Exact` was unsound under fragmentation.** The rule compared totals; admission needs a **contiguous** range | A 4,608 B cache holding 3,840 B in three 256 B holes predicted **768 B of reads exactly** and read **1,536 B** | The rule asks whether **one free run** can hold what the layer admits, padding included. `ExpertBudget` carries the largest contiguous free range, the alignment, and how many chunks an expert arrives in |

**The first finding is half mine.** The regression I wrote for the previous
round's P1 called `while_failing(1, ...)` six times in a loop whose index only
changed the assertion message: **every iteration failed the first allocation**.
An axis that was exercised and never varied — the exact failure AGENTS.md records
from task 0021's fourth review, in a test I wrote *because* of a review. It
sweeps every allocation position now, and the position count is measured rather
than assumed, so a position that stops being reached shows up as a changed
number.

**The third is the third time that rule has been wrong**, and the three are a
sequence worth keeping: it asked whether this plan's admissions fit, then whether
its admissions and its hits fit, then whether everything resident and its
admissions fit the cap. Each is a quantity that is not the one admission asks
for. What admission asks for is a contiguous range, and each wrong version was
found by a counterexample rather than by reading the rule.

**A term no fixture varies.** The padding allowance the third fix introduced is
**inert in every other fixture**, because every chunk in this workspace is a
whole number of alignment units. It has its own pure-planner test —
`exactness_reserves_the_alignment_a_chunk_can_waste` — for exactly the reason
AGENTS.md gives: a parameter no fixture varies is a parameter no test checks.

### The third review round: one P1, and it was my own reasoning

**One finding, P1, reproduced and fixed.** An **ordinary** discrepancy still
allocated: `Check::eq` called a closure that built a `String` with `format!` and
converted it into a `Cow`. A ledger-identity mismatch with one allocation failing
was **SIGABRT, 114 bytes**.

**The reasoning that stopped short is recorded here because it was mine.** The
previous round's fix made the path that *handles* an allocation failure
allocation-free and deliberately left the ordinary mismatch paths formatting,
on the argument that a mismatch is not an out-of-memory context. That argument is
wrong in one line: **under memory pressure a mismatch is as likely as an
allocation failure**, and a diagnostic that cannot be built is a process that
cannot report anything at all. The field has now been walked down three times —
`String`, then `Cow<'static, str>`, then `&'static str` — and each step was a
review finding.

`Discrepancy::detail` is a `&'static str`. Every one of the thirty-two
construction sites carries a static description of what its equality means; the
numbers are the structured fields beside it, and `Display` composes them, so
whether *rendering* allocates is the caller's decision rather than a step's.
Anything richer — which tiers, which flows, which counts — is in the `StepTrace`
the caller already holds.

**The test the previous round's fix needed and did not have**: the OOM sweep
reconciled only a *valid* trace, so it never constructed a diagnostic at all.
Every one of the **seventeen** equalities is now violated in turn with **every**
allocation refused, and each must report the same check it reports with memory
available. Two mutations measure it: one makes an ordinary discrepancy build its
prose, one makes the out-of-memory report build its own.

### The owner resolved O1–O5 during this task

`1e927ef`, `71bd099` and `7a1f655` landed between this contract and this
implementation. Read after the fact, and reconciled here rather than left to
contradict the paragraphs above:

- **O1 is resolved**, and the designated artifact of this task,
  `/fast/models/google/gemma-4-26B-A4B-it`, is **explicitly not in the v1
  catalog** — it "remains M2's BF16 workhorse". So the whole working set this
  task ran is an engineering fixture at real scale, and was never going to be a
  release claim. The other family this workspace has inspected,
  `cyankiwi/Laguna-S-2.1-AWQ-INT4` at the revision task 0022 recorded, **is**
  number 8 of the ten.
- **O2 is resolved**: for v1, acceptable loss is a **bit-identical repack**, and
  the publisher's own quality is accepted as-is. Nothing here claims otherwise.
  What this task's caveat means under the new ruling is unchanged and its
  citation is not: computing a routed block over **synthetic activations and a
  written route** says nothing about any model's output, catalogued or not. It
  is not a quality claim, and it is not evidence for one.
- **O5 is resolved**: storage and conversion are user-managed, repack is an
  external script, and no agent-initiated bulk write may happen without a task
  naming the artifact, revision, size and retention. **This task wrote nothing**
  under either checkpoint root, so it complies with the ruling as it complied
  with the gate.
- **O3 and O4** are resolved and touch nothing here. **O6 and O7 remain open**,
  which is what keeps every timing in this record a diagnostic.
- M2's deliverables and exit text in document 06 are **unchanged** by
  `7a1f655`; this task's target is intact.

Older records — AGENTS.md's earlier paragraphs and tasks 0018–0022 — still
describe O1, O2 and O5 as open. Reconciling those is the owner's, not this
task's: rewriting another task's record to match a later ruling would erase what
was true when it was written.

### Deleted or replaced paths

None. `ExpertBudget::resident_experts` became `ExpertBudget::resident` with a
richer type, and every call site moved with it; no path was left behind as a
bridge.

### Remaining blockers and the next bounded task

In [the handover](../handovers/2026-09-13-task0023-whole-working-set-trace.md).
The short form: **M2 is not closed and closing it is the owner's**; nothing
generates a token from either designated artifact; `whole-set-full` is not
compared against the CPU candidate, by declared exclusion with the arithmetic
attached; and the owner's O1–O5 rulings of 2026-09-13 arrived mid-task and are
reconciled in the Result section and the handover.
