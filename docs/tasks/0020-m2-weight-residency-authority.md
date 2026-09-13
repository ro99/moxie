# Task 0020 — M2 weight-residency authority

Status: **accepted by the owner on 2026-09-12**, after seven rounds of
independent review, the seventh reporting no new blocking findings and
recommending acceptance "within its declared M2 item 2 scope".

**The acceptance closes this residency assignment only. It does not close M2**,
whose exit still needs grouped expert execution, CPU fallback and a real working
set that executes — none of which exists. Nothing here executes a checkpoint.

Contract written and committed at `d6e9170` before
implementation, per the working rule that produced tasks 0013–0019. The seven
rounds found **twenty-seven** issues in total; all twenty-seven were reproduced
and fixed, and none was disputed. Three rounds criticised method or a coverage
claim rather than code, and each was right. **The last two rounds found no
residency defect at all**; both findings were in the architecture check, which
[has now been wrong four times in the same place](#the-check-that-was-wrong-four-times).
See
[the sweep and what it is worth](#the-sweep-and-what-it-is-worth). Also see
[Result](#result-filled-after-work),
[Independent review](#independent-review-and-what-it-changed),
[Second independent review](#second-independent-review) and
[Third independent review](#third-independent-review).

## Identity and authority

- Task ID / milestone / owner: 0020 / **M2 item 2, the residency authority** /
  implementation agent; acceptance belongs to the owner. **This task does not
  close M2**: its exit also needs item 3's CPU fallback and grouped GPU plans,
  item 4's Laguna metadata and restricted budget, and a real working set
  actually executing.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `8ee8fd0`
  (task 0019 acceptance record). Working tree clean at authoring; no initial
  dirty paths.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and
  `tests/p2p/` are preserved and are not source evidence.
- Assigned by
  [the task 0019 handover](../handovers/2026-09-12-task0019-routed-expert-semantics.md),
  whose "Next task" section names owning component, required reading, the terms
  this contract must state before implementation, the stop conditions and the
  tests. Every one of them is carried below.
- Requirement / finding IDs repaired: **R02** (the simulator that modelled bytes
  while the model runtimes spent them), **R03**, **R07**, **R08**, **R10–R15**
  — M2's declared reference set.
- Required documents read: AGENTS.md, README, reference documents 01–09, the
  owner-gate register, `docs/README.md`, the TASK/ADR/HANDOVER templates.
  Normative for this task: document 03's **"Shared weight-residency lifecycle"**
  in full and its **MoE admission paragraph** under "Resource ledger and
  admission"; document 02's crate-ownership table (`moxie-memory` = "real
  resource authority, reservations, leases, arenas, tier movement, admission",
  forbidden "independent caches in consumers") and its **"Buffer and
  asynchronous lifetime contract"**; document 06 **M2 items 2 and 5**;
  [ADR 0009](../decisions/adr/0009-engram-conditional-memory.md) for
  conditional-memory tables as *one residency class under the same authority*;
  and the accepted results of tasks
  [0006](0006-m1-resource-ledger-and-admission.md),
  [0009](0009-m1-event-backed-leases.md), [0010](0010-m1-basic-device-arena.md),
  [0011](0011-m1-admitted-graph-resource-plan.md) and
  [0019](0019-m2-routed-expert-semantics.md).
- Owner gates: **O1–O7 remain open.** None blocks this task, which reads bytes
  and writes none. Stop before any conversion, requantization, download, bulk
  write, quality claim or checkpoint execution. O5 governs writing; **reading a
  bounded byte range of a read-only local artifact is not writing**, and tasks
  0018 and 0019 already established that boundary.

### What "connect the real residency authority" means here, and what it does not

Roadmap M2 item 2 is one sentence with eight nouns in it: storage reads, host
cache, CUDA upload readiness, leases, eviction, error recovery, demand and
prefetch classes. This task delivers all eight as one authority. It deliberately
does **not** deliver the two things that would let the designated artifact
generate a token — grouped expert execution (item 3) and a device routed kernel
(item 3/M6) — because neither can be built against a cache that does not exist,
and because a residency authority that is only exercised by the thing that needs
it cannot be tested at the boundaries M2 item 5 names. The consequence is stated
plainly in the acceptance rules: **nothing in this task executes a checkpoint,
and a demand-loaded expert is not model support.**

## The demand this authority must serve

Task 0019 produced the routing equation, so the union of experts a row batch
demands is now a computable quantity rather than an estimate. That ordering was
the reason task 0019 was narrowed, and it is the input here.

Byte arithmetic for the designated artifact,
`/fast/models/google/gemma-4-26B-A4B-it` (read-only; **nothing copied,
converted, deleted or executed**), re-read from the safetensors headers on
2026-09-12 and agreeing with
[the bring-up record](../models/gemma4.md#resources-state-and-partitions):

| Quantity | Bytes |
|---|---:|
| `layers.0.experts.gate_up_proj`, `[128, 1408, 2816]` BF16 | 1,015,021,568 |
| `layers.0.experts.down_proj`, `[128, 2816, 704]` BF16 | 507,510,784 |
| **One expert, one layer** (its slice of both) | **11,894,784** |
| All 128 experts, one layer | 1,522,532,352 |
| All experts, 30 layers | 45,675,970,560 — **88.5%** of the artifact |
| Top-k 8 over 30 layers, one token, **no reuse** | 2,854,748,160 |

Two facts decide this task's shape.

**The experts are fused, so residency needs a ranged read.** Expert `e`'s
`gate_up_proj` slice is the contiguous byte range
`[e · 7,929,856, (e+1) · 7,929,856)` inside a 1,015,021,568-byte tensor, and its
`down_proj` slice is `[e · 3,964,928, (e+1) · 3,964,928)` inside a 507,510,784-byte
one. `moxie_storage::Shard::read_tensor` reads a *whole* tensor into a
caller-sized buffer, so demand-loading one expert through it would read 1.42 GiB
to use 11.34 MiB. Bounded ranged reads are therefore part of this task, and they
are the thing document 02 means when it gives `moxie-storage` "bounded
reads/mappings" and gives this crate the cache.

**The largest single device is 24 GiB and one layer's experts are 1.42 GiB.**
An intentionally restricted budget is satisfiable by construction, and
`committed + incoming > cap` is the ordinary case, not the edge case.

## Bounded deliverable

**One concrete outcome:** one production weight-residency owner —
`moxie_memory::residency` — that admits, reads, uploads, leases, evicts and
accounts for immutable weight chunks against a ledger-admitted envelope, with a
driver that performs its work orders as real bounded storage reads and real
uploads, and an admission report that names the incoming chunk's size before any
victim is chosen.

**Sole owning shared component:** `moxie-memory`. It owns chunk identity, the
state machine, coalescing, lease accounting, class policy, eviction choice and
the report. It opens no file, allocates no device memory and waits on no event —
the same split that already pairs `Arena`'s pure ranges with one real allocation
in `moxie-executor`. `moxie-storage` supplies bounded reads. `moxie-executor`
supplies the driver that performs work orders and binds device readiness to
completion events. **`moxie-models` gains nothing at all.**

**Allowed production files:**

- `crates/moxie-memory/src/residency.rs` — new. `ChunkId`, `PreparedId`, the
  entry state machine, `ResidencyAuthority`, `ResidencyLease`, `WorkOrder`,
  `UseClass`, `ResidencyReport`, `ResidencyStats`, eviction.
- `crates/moxie-memory/src/host.rs` — one tier-parameterized constructor, so a
  weight cache is charged to a weight-cache host tier instead of borrowing
  `StateSpill`. The two existing constructors are expressed through it and keep
  their current tiers and behaviour.
- `crates/moxie-memory/src/lib.rs` — module and re-exports.
- `crates/moxie-storage/src/lib.rs` — `Shard::read_tensor_range` and
  `Artifact::read_tensor_range`: one bounded byte range of one named tensor,
  through the existing budget-capped pump, with the existing confinement and
  truncation rules and no new mapping.
- `crates/moxie-executor/src/residency.rs` — new. The driver: it asks the
  authority for work, performs reads through `moxie-storage` and uploads through
  the existing lease/event machinery, and reports outcomes back. **It holds no
  cache, chooses no victim and keeps no policy.**
- `crates/moxie-executor/Cargo.toml` — the one new production edge,
  `moxie-storage`, with the allowlist row and its justification.
- `crates/moxie-engine/src/residency.rs` — new, small. Turns a batch's routes
  into an ordered, deduplicated chunk demand against a caller-supplied role
  table. It names no model family; the role strings arrive as data.
- `xtask/src/archcheck.rs` — the new `second-residency-owner` rule, the widened
  executor row, and negative fixtures.
- Their tests and manifests, `Cargo.lock`, and the tracked task, support-matrix,
  model and handover records.

**Explicit non-goals and forbidden shortcuts.**

- **No second weight-residency owner**, anywhere, including a "small" cache in
  the driver, the engine or a test harness promoted to production.
- **No cache class in a model adapter.** `moxie-models` keeps its three
  dependencies.
- **No simulated placement presented as a reservation.** Every resident byte is
  inside an envelope the ledger admitted, and every device range is inside an
  arena the executor backs with one real allocation. A number in a struct that
  no allocation corresponds to is a failed task, not a partial one.
- **No unbounded queue.** The prefetch queue has a declared capacity and refuses
  past it. The demand path never queues at all: it either acquires or is refused.
- **No blocking demand path.** `acquire` never waits, so it cannot deadlock when
  every evictable entry is leased; it returns a typed refusal with the report.
- **No bulk write, no conversion, no requantization, no download.** Nothing is
  written under `/models` or `/fast/models`. That is O5's.
- **No quality claim.** That is O2's and needs paired output against the released
  model.
- **No CPU expert fallback and no grouped GPU candidate plan** — M2 item 3.
- **No expert partitioning** — M5. The routed operations' partition rules stay
  as task 0019 left them and keep failing closed.
- **No device routed kernel.** The selected BF16 chain keeps refusing `Route`,
  `ExpertMlp` and `Combine` as `UnsupportedKernel`, and this task's device work
  is transfer and readiness only.
- **No route prediction policy.** The prefetch *class* exists and is exercised
  from an explicitly supplied list. A predictor is a later, measured decision
  (document 03: "add smarter policies only with replayable route traces and
  measured benefit"), and inventing one here would be the unmeasured policy that
  paragraph forbids.
- **No Engram mathematics.** ADR 0009 requires conditional-memory tables to be
  one residency class under this authority and **never zero-resident**. This task
  delivers exactly that: the class and its resident floor, so a later task has no
  reason to build a second cache. No hash addressing, no gating, no conv, no
  table — and no Engram capability is claimed.
- **A demand-loaded expert is not model support.** The real-artifact evidence
  below reads bytes through the authority and computes nothing with them.

**Existing consumers and second-shape proof.** Every accepted gate keeps passing
unchanged. The authority gets three independent consumers with deliberately
different shapes:

| | Gemma-4-MoE-shaped | Second synthetic MoE | Dense spine |
|---|---|---|---|
| Chunk shape | fused-expert slices, two roles per expert | a different expert count and chunk size | whole tensors, no expert index |
| Class | `Expert` | `Expert` | `DenseSpine` |
| Demand | union of a real router's routes, overlapping across rows | disjoint routes, plus all-rows-to-one-expert | every chunk every step |
| Reuse | high across steps | none across steps | total |

and a fourth, `ConditionalTable`, which exists to hold the resident floor and is
demanded from a fixed row list rather than from a route.

## Contract before implementation

### Identity

Document 03 fixes both identities and this task uses them verbatim.

```text
ChunkId    = (artifact, tensor/expert, logical range, format version)
PreparedId = (chunk, device capability, kernel layout version)
```

- `ArtifactId` is an opaque validated non-empty identity string. The memory
  authority may not learn what a path is (`arch-check` rule
  `memory touches the filesystem`) and may not branch on a model name (rule
  `memory branches on a model name`). For a canonical artifact the identity is
  `moxie_storage::Artifact::identity()`; for the designated artifact, which has
  no canonical manifest, it is the index digest recorded in the bring-up record.
- `TensorSlot { role: RoleName, expert: Option<u32> }` is document 03's
  "tensor/expert" in one type. `expert: None` is a whole tensor; `Some(e)` is
  expert `e`'s slice of a fused one.
- `LogicalRange { offset_bytes, len_bytes }` is the range **inside that tensor**,
  not a file offset. Translating it to a file offset is the driver's job, and
  the authority must not be able to express a file offset.
- `format_version: u32` participates in identity, so a chunk imported under a
  changed scale convention is a different chunk and cannot be served from cache.
  Document 02 states that requirement for plan cache keys; it is the same
  requirement here.

`ChunkId` is `Clone + Ord + Hash`, ordered so eviction can break ties
deterministically. `PreparedId` adds `CapabilityKey` (SM major/minor plus an
opaque kernel-layout tag) and `layout_version`, and is carried but not yet
produced by any preparer: this task uploads canonical bytes, and a prepared
layout is M3's repacker. The type exists so the identity is not retrofitted
later, and a test asserts that two prepared ids differing only in
`layout_version` are different keys.

### The state machine, including failure and cancellation

```text
Absent ──acquire──> Reading ──read ok──> HostReady ──acquire(device)──> Uploading
                       │                    │                              │
                       │ read err           │ evict (unleased)             │ upload ok
                       v                    v                              v
                    Absent               Retiring ──all leases retired──> Absent
                                                                          ^
   Uploading ──upload err (observed)──> HostReady                         │
   Uploading ──upload state unknown──> Quarantined ──loss observed────────┘
   Reading|Uploading ──cancel──> Quarantined (charged) ──completion observed──> settle
   HostReady ──prepare──> Preparing ──> PreparedHostReady | PreparedDeviceReady
```

Rules that are tested rather than asserted:

- **Charge follows the state, not the intent.** An entry in `Reading`,
  `Uploading` or `Quarantined` is charged for its bytes. Document 02 forbids
  `Drop` alone from freeing an in-flight resource and R08 is the cost of the
  opposite habit, so a cancelled in-flight entry stays charged and stays visible
  in `outstanding()` until its completion or loss is *observed*. It is not
  evictable in that state, and it is not silently reused.
- **A failed read releases everything it charged and leaves `Absent`.** Every
  coalesced waiter receives the same typed error. Retry is a fresh `acquire`,
  never an internal retry loop: an invisible retry is how a storage fault becomes
  a latency mystery.
- **A failed upload whose submission state is *observed* returns the entry to
  `HostReady`** and releases the device range, so the host bytes are not
  re-read. A failure whose submission state is **unknown** goes to `Quarantined`
  and withholds the device range — the same rule `LeaseState::Lost` already
  applies to execution leases in `moxie-executor`.
- **`Retiring` is not `Absent`.** Bytes are released when the last lease retires,
  not when eviction is decided.

### Coalescing

`acquire` on a chunk already in `Reading` or `Uploading` issues a **lease and the
existing ticket**, and **no second work order**. The test is a count, not an
inspection: `N` acquires of one absent chunk produce exactly one `WorkOrder`,
exactly one `read_tensor_range` call, exactly `len_bytes` bytes read, and `N`
leases that all become ready from one `complete_read`. A refused or failed
completion resolves every coalesced waiter identically.

### `acquire` and readiness

```text
acquire(chunk, destination, deadline, use_class, turn) -> Result<Acquired>

Acquired::Ready   { lease }                        the chunk is resident now
Acquired::Pending { lease, ticket, work: Option }  work == None means coalesced
```

- `destination` is a `Scope`: `Host` for a host-cache residency, `Device(uuid)`
  for a device one. A device acquire of an absent chunk produces the read first
  and the upload second; the ticket covers both, so a consumer has one readiness
  dependency rather than two.
- `deadline` is a monotonic tick supplied by the caller. The authority reads no
  clock — a clock in the accounting core would make every test a race. Deadlines
  order the demand queue (earliest first, then sequence, so it is total), and
  `expire(now)` fails every pending ticket whose deadline has passed with a typed
  error, releasing its charge. A deadline is never met by dropping correctness.
- `use_class = (urgency, content)`:
  - `Urgency::{Demand, Prefetch}`. Demand outranks prediction: a prefetch is
    never chosen over a demand for queue order, and is always evicted first.
  - `Content::{DenseSpine, Expert, ConditionalTable}`. Document 03 requires dense
    spine tensors to be *charged against expert and context capacity* rather than
    excused from it, so `DenseSpine` shares one cap. `ConditionalTable` carries
    ADR 0009's floor: eviction may not take the class below
    `conditional_floor_bytes`, which is ≥ 1 when the class is in use, which is
    what "never zero-resident" means mechanically.
- `turn` scopes the lease. `end_turn(turn)` retires every lease taken in it and
  reports them. This is R08 stated as a mechanism: the legacy leak was a lease
  released on "next token" that leaked whenever a turn ended and no next token
  came, so a turn ending with no next token must release everything, and a test
  asserts zero live leases and a baseline charge after exactly that sequence.
- `ResidencyLease` is **not `Clone`**, must be released explicitly, and dropping
  one leaves it charged and visible in `outstanding()` — the discipline
  `Reservation`, `Allocation` and `Lease` already share, for the reason each of
  their doc comments gives.

### The eviction rule

Deterministic demand LRU plus a bounded prefetch class, which is document 03's
declared starting policy, and nothing smarter:

1. Compute `need = committed_bytes + incoming_bytes` against `cap_bytes`
   **before choosing any victim**, and record `incoming_bytes` in the report.
2. Candidates are entries in the destination scope with **zero live leases**,
   in `HostReady` or `DeviceReady` only. In-flight and quarantined entries are
   never candidates.
3. Order candidates by `(class_rank, last_used_tick, chunk_id)`, where
   `class_rank` puts prefetch-admitted-and-never-demanded entries first and
   `chunk_id` makes the choice independent of map iteration order.
4. Refuse to take `ConditionalTable` below its floor.
5. Evict in that order until `need <= cap`. If candidates are exhausted and it
   still does not fit, **refuse immediately** with `CapacityExceeded` and the
   report. The authority never waits, so the deadlock M2's exit forbids cannot
   be expressed.
6. A prefetch admission that would evict a demand entry is refused instead —
   document 03: speculative prefetch is "bounded and evictable before useful
   demand data", which is a one-way rule.

### The admission report

Produced on **both** outcomes, because a successful admission that quietly
evicted four experts is the thing an operator needs to see:

```text
ResidencyReport {
  scope, tier, cap_bytes,
  committed_bytes,          // resident + in-flight, before this request
  incoming_bytes,           // THIS chunk, named before any eviction
  leased_bytes,             // present and unevictable
  evictable_bytes,
  would_evict: [ChunkId],   // the exact victims, in the exact order
  remaining_headroom_bytes,
}
```

`incoming_bytes` is document 03's "include incoming expert size before
evicting/allocating" as a field rather than a log line. The refusal path also
carries `LegalAlternative`-shaped guidance reusing the ledger's existing
vocabulary, so a refusal says what would fit rather than only that nothing does.

### Shapes, precision, accumulation and rounding

None. **This task moves bytes and computes nothing.** It performs no arithmetic
on tensor contents, so it has no accumulation order, no rounding boundary and no
numerical oracle — and claiming one would be the manufactured completion
AGENTS.md forbids. Its oracle is byte identity (below). The one arithmetic
contract it does have is capacity arithmetic, which is exact `u64` throughout,
checked for overflow at every add and multiply, with a refusal rather than a
wrap — the rule tasks 0006 and 0011 already established.

### Partition and hardware capabilities

Residency is per `Scope` and a `Scope` is a device UUID or the host, never an
ordinal. A chunk resident on one device is not resident on another, and the
authority holds one independent cap per scope: AGENTS.md forbids assuming the
three devices' memory is one allocation, and a single global cap would be exactly
that assumption. Expert *partitioning* across ranks stays M5 and is not expressed
here.

### Peak memory, transfer dependencies, lease lifetime

- The host cache is one `HostBuffer` of `cap_bytes` admitted from the ledger,
  suballocated by one `Arena`. Per-chunk host bytes are therefore inside an
  admitted envelope by construction, and the peak is the cap, declared once.
- Each device cache is one `Arena` whose capacity the executor backs with one
  real allocation, charged to `Tier::Device(DeviceTier::ExpertCache)`.
- Transfer staging is charged to `Tier::Device(DeviceTier::TransferStaging)`
  through the existing `check_upload_fit`, which already validates the complete
  transient upload footprint before allocation.
- An upload leases its source host bytes through the completion event. Document
  02: "A scratch source cannot be overwritten merely because the enqueue function
  returned." So a host entry with an outstanding upload is not evictable, and a
  test drives exactly that ordering.
- The authority's own control metadata is bounded and charged: the entry table's
  capacity is declared at construction from `max_entries`, admitted as
  `HostTier::Pageable`, and `acquire` on a resident chunk allocates **zero**
  bytes on the heap. That last one is measured with the counting allocator, the
  way tasks 0013–0015 measured theirs, not asserted.

### Cancellation, failure and rollback

- `cancel(ticket)` retires the intent. In-flight bytes stay charged until
  completion or loss is observed; then the entry settles to `Absent` (read) or
  to `HostReady`/`Quarantined` (upload). Repeated cancellation of the same ticket
  is idempotent and typed, not a panic.
- Cancelling the last waiter on a coalesced ticket cancels the work; cancelling
  one of several does not.
- `end_turn` on a turn with outstanding in-flight tickets releases the leases and
  leaves the tickets to settle — it never frees bytes a transfer may still touch.
- Every failure is a typed `moxie_types::Error`, never a string match:
  `CapacityExceeded` for refusals, `InvalidArtifact` for truncation and checksum
  failures, `Cancelled` for cancellation and expiry, `DeviceLost` for an unknown
  submission state.
- Rollback is exact: after any failure or cancellation sequence, `committed_bytes`
  equals the sum of resident and settled-in-flight entries, the arena's occupancy
  agrees with it, and the ledger's charge agrees with both. A test asserts all
  three against each other rather than each against a constant.

### Independent oracle and predeclared metrics

The oracle is **byte identity against an independent reader**: every chunk the
authority serves must equal, byte for byte, the same range read directly through
`moxie_storage` without the cache — and, for the real artifact, against the
tensor's own bytes sliced in the test. A cache whose hits differ from its misses
is the defect this checks for, and it is checked on hits, on post-eviction
reloads and on coalesced waiters.

Predeclared, and failing if not met:

| Metric | Threshold |
|---|---|
| Served bytes vs. independent read | **exact equality**, every case |
| Reads issued for `N` coalesced acquires | exactly **1** |
| Bytes read for a demand-loaded expert of the real artifact | exactly **11,894,784** per expert-layer, not the 1,522,532,352 of its fused tensors |
| Heap allocations on a resident-chunk `acquire` | exactly **0** |
| Retained heap growth after 10,000 acquire/release/evict cycles | **0** |
| `committed_bytes` vs. arena occupancy vs. ledger charge | **equal**, after every case |
| Charge after `end_turn` with no next token | **baseline** |
| Demand refusal when all entries are leased | typed and **immediate**; no wait |
| Victim sequence for a fixed access trace | **exactly** the predeclared list |
| Prefetch entries evicted before any demand entry | always |
| `ConditionalTable` resident bytes | never below the declared floor |

Reported but not thresholded, because no baseline exists to compare them against
and inventing one would be an unsupported performance claim: read and upload
byte counts, hit and miss counts per class, wasted prefetch bytes, and evictions
of demand data. Document 03 requires these to be *recorded*; it does not license
a performance claim from them, and none is made.

### Application compatibility and sampler implications

None. No protocol surface, sampler, precision, context target or capability
matrix entry changes. The generation service's behaviour is unchanged, and the
CLI gains no flag.

## Acceptance

**M2 item 5's list, in full** — it is the milestone's own test list and the
handover carries it forward verbatim:

1. **Full cache** — an admission against a cache with no free bytes evicts in the
   predeclared order and serves correct bytes afterwards.
2. **All entries leased** — a demand against a full cache whose every entry is
   leased is refused immediately with a report showing `leased_bytes == cap_bytes`
   and `evictable_bytes == 0`; releasing one lease and retrying then succeeds.
   This is M2's "demonstrate demand failure cannot deadlock".
3. **Incoming largest expert** — the incoming chunk is larger than any single
   resident one and larger than the free space; the report names its size before
   any victim is chosen, and an incoming chunk larger than `cap_bytes` itself is
   refused without evicting anything at all.
4. **Failure mid-read** — a source that fails after `k` of `n` slices leaves
   `Absent`, releases every byte, fails every coalesced waiter identically, and
   leaves the authority usable.
5. **Failure mid-upload** — observed failure returns to `HostReady` without
   re-reading; unknown submission state quarantines and withholds the range.
6. **Repeated cancellation** — the same ticket cancelled twice, a cancelled
   ticket completed afterwards, and 1,000 cancel/restart cycles with no retained
   growth and no charge left behind.
7. **No-next-token cleanup (R08)** — a turn that ends without a next token
   releases every lease and returns the charge to baseline.
8. **Repeated and missing expert routes** — a route union with one expert
   repeated across rows demands it once; a route naming an expert with no chunk
   is a typed refusal naming the chunk, not a panic and not a silent skip.
9. **Nonuniform row counts** — expert batches of different sizes produce the same
   union and the same demand set as their uniform equivalent.

**Plus:**

- The oracle: byte identity on hits, on reloads after eviction, and on coalesced
  waiters, against an independent read.
- The allocation gates: zero heap allocations on a resident acquire, zero
  retained growth over 10,000 cycles, exact reconciliation of authority, arena
  and ledger after every case.
- **The real-artifact read**, host lane, reported separately and **skipped with
  its reason recorded when the artifact is absent**: a demand union from a real
  router's routes over `/fast/models/google/gemma-4-26B-A4B-it` layer 0, against
  a cache cap deliberately smaller than the union so eviction actually runs,
  reading each expert's 11,894,784 bytes and no more, verified against an
  independent slice of the same tensors. **It computes nothing with the bytes,
  executes no graph, and writes nothing.**
- Architecture: `xtask arch-check` passes, with a **new rule**
  `second-residency-owner` — no production crate other than `moxie-memory` may
  *define* a type whose name matches the residency-cache vocabulary — and
  negative fixtures for a model crate, an engine crate and the driver each
  declaring a cache of their own. `moxie-memory` stays I/O-free and model-free;
  `moxie-models` keeps its three dependencies; the one widened row
  (`moxie-executor` gaining `moxie-storage`) is justified in the allowlist and
  its own fixture proves the reverse edge is still refused.
- Device lane, on real hardware: the upload path with real events on every UUID
  — readiness before use, completed-cancellation recovery, and an injected copy
  fault reaching the typed quarantine path. Reported separately from the host
  lane, and reported as unmeasured if the hardware is unavailable.
- All prior gates unchanged: `G-INTERP-BF16`, `G-PAGED-HOST`,
  `G-PAGED-ALLOCATION`, `G-KV-RETENTION`, `G-WINDOW-ALLOCATION`,
  `G-SAMPLING-HOST`, `G-SAMPLING-ALLOCATION`, `G-GENERATION-HOST`,
  `G-GENERATION-ALLOC`, `G-GEMMA-REDUCED`, `G-MOE-ROUTING-HOST`, `G-CT-IMPORT`,
  `G-ARENA-HISTORY`, `G-ARENA-PENDING`, `G-RESOURCE-PLAN`, `G-LEASE-FAULTS`,
  `G-HOST-ARCH`, `G-HOST-SPEC`, `G-HOST-NODRIVER`.
- Support matrix: one new host gate `G-RESIDENCY-HOST` and one new device gate
  `G-RESIDENCY-DEVICE`, whose limit columns say in their own words that no
  checkpoint executes, that a demand-loaded expert is not model support, and
  that M2's exit is not met.
- Documentation: this contract's Result section filled in with passed, failed and
  skipped kept **separate**; the [gemma4 bring-up record](../models/gemma4.md)
  updated where it currently says residency is "not started"; a handover naming
  the next bounded task.

**Exact condition requiring owner direction or task rejection.**

- If serving a bounded range of the designated artifact requires **writing**
  anything under a checkpoint root — a canonical manifest, an index, a prepared
  layout, a cache file — **stop**. That is O5, and no local inference may take
  it.
- If the authority cannot be built without a second cache in the driver, the
  engine or a model adapter, **stop and report** rather than widening
  `moxie-models` or accepting two owners. M2's exit says "exactly one" and this
  task may not spend that.
- If a bounded ranged read cannot be done without mapping a whole shard, **stop
  and report**: `Shard`'s doc comment pins the reason nothing is mapped ("a file
  that changed under a mapping would make a validated header a lie"), and
  reversing that is an ADR, not a task decision.
- If the device upload path cannot bind readiness to a completion event without
  relaxing the buffer-lifetime contract, **stop**. R07 and R08 are the two
  failures that rule exists to prevent.

## Result, filled after work

Status: **implemented; not accepted.** Acceptance belongs to the owner, and this
task does **not** close M2.

### Changed shared owners and consumers

| Component | What changed |
|---|---|
| `moxie-memory` | **New `residency` module**: `ChunkId`/`PreparedId`, the placement state machine, coalescing, the bounded lease table, class policy, two-phase eviction, `ResidencyReport`, `ResidencyStats`. `HostBuffer` gains `allocate_in`, a tier-parameterized constructor, so a weight cache is charged to `host.pageable` instead of borrowing `StateSpill` |
| `moxie-storage` | `Shard::read_tensor_range`: one bounded byte range of one named tensor, through the existing budget-capped pump, checked against the validated header |
| `moxie-executor` | **New `residency` module**: the `ChunkSource` trait, `ShardSource`, `perform_read`, `drain_reads`, and under `driver` the `DeviceResidency` that backs one scope's cache with one real allocation and binds readiness to a real event. One new production dependency, `moxie-storage` |
| `moxie-engine` | **New `residency` module**: `expert_demand` and `demand_bytes`, the route-union-to-chunk arithmetic |
| `xtask` | New rule `a second weight-residency owner`, matching type **definitions**; three negative fixtures; the widened executor allowlist row |
| `moxie-models` | **Nothing.** It still depends on `moxie-types`, `moxie-graph` and `moxie-model-api` and nothing else |

### Commands and results

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | **passed** |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | **passed** |
| Device-lane clippy (`moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda`) | **passed** |
| `cargo test --workspace --locked --offline` | **823 passed, 0 failed** (736 before this task) |
| Device-feature workspace tests | **843 passed, 0 failed** |
| `cargo xtask-cuda test-gpu` | **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask spec-check` | **passed**, 10 documents |
| `cargo xtask arch-check` | **passes with zero failures**: 78 rejected fixtures, 21 accepted, 13 rules, including the new `a second weight-residency owner` with its three fixtures. The "4 pre-existing failures" every task since 0014 has carried are **gone, and were never real** — see [the arch-check noise floor](#the-arch-check-noise-floor) |

Raw logs: `results/task0020/` (untracked, per `docs/README.md`).

**Nothing failed. Nothing was skipped except one case, named below.**

### The nine cases M2 item 5 names

| Case | Where | Result |
|---|---|---|
| Full cache | `a_full_cache_evicts_in_the_predeclared_order_and_still_serves_correct_bytes` | passed; the victim list is asserted exactly |
| All entries leased | `a_demand_against_a_cache_whose_every_entry_is_leased_is_refused_immediately`, and again on real hardware in `residency_device` | passed; `leased_bytes == cap_bytes`, `evictable_bytes == 0`, immediate typed refusal, usable afterwards |
| Incoming largest expert | `the_incoming_largest_expert_is_counted_before_anything_is_evicted`, `a_chunk_larger_than_the_whole_cache_is_refused_without_evicting_anything` | passed |
| Failure mid-read | `a_failed_read_releases_every_byte_and_fails_every_waiter_identically`, `a_read_that_fails_part_way_serves_nobody_and_leaves_the_cache_usable` | passed; the injected source half-fills the buffer and nobody is served those bytes |
| Failure mid-upload | `an_observed_upload_failure_keeps_the_host_bytes_and_frees_the_device_range`, `an_upload_whose_submission_state_is_unknown_withholds_both_ends` | passed; observed failure retries without re-reading, unknown submission quarantines both ends |
| Repeated cancellation | `cancelling_an_in_flight_read_keeps_its_bytes_charged_until_the_outcome_lands`, `a_thousand_cancel_and_restart_cycles_leave_no_charge_and_no_growth` | passed |
| No-next-token cleanup (R08) | `a_turn_that_ends_with_no_next_token_releases_everything_r08`, `a_turn_ending_over_an_in_flight_transfer_does_not_free_its_bytes` | passed |
| Repeated and missing expert routes | `overlapping_routes_demand_each_expert_once`, `every_row_routing_to_one_expert_demands_one_expert`, `a_route_outside_the_expert_count_is_refused_by_name`, `a_chunk_the_source_cannot_name_is_refused_rather_than_guessed` | passed |
| Nonuniform row counts | `nonuniform_row_counts_produce_the_same_union_as_their_uniform_equivalent` | passed |

### Measured effect and uncertainty

**Predeclared metrics, all met:**

| Metric | Threshold | Measured |
|---|---|---|
| Served bytes vs. an independent read | exact equality | **exact**, on hits, on post-eviction reloads and for coalesced waiters |
| Reads for `N` coalesced acquires | 1 | **1**; a source permitting exactly one read proves it by failing otherwise |
| Bytes read per real expert-layer | 11,894,784 | **11,894,784**, against 1,522,532,352 for its two fused tensors |
| Heap allocations on a resident acquire | 0 | **0** over 1,000 hits, and **0** on release |
| Retained heap after 10,000 cycles | 0 | **0**, for demand/evict and for failed reads |
| Authority vs. arena vs. ledger | equal | **equal** after every case |
| Charge after `end_turn` with no next token | baseline | **baseline** |
| Demand refusal when all entries leased | immediate, typed | **immediate**; there is no waiting path in the API. **A nonblocking `acquire` is not by itself deadlock freedom** — the review found a cycle between the demand counter and the prefetch gate that `acquire` never touches. That cycle is fixed and tested; the absence of a waiting path remains necessary, not sufficient |
| Victim sequence for a fixed trace | the predeclared list | **exact** |
| Prefetch evicted before demand | always | **always**; a prefetch that would displace demand data is refused instead |
| `ConditionalTable` resident bytes | never below the floor | **never** |

**The real artifact, read through the authority.** Host lane, not skipped:
nine distinct experts of `/fast/models/google/gemma-4-26B-A4B-it` layer 0,
demanded from routes shaped like a top-k-8 batch of three rows, against a cache
holding four. **107,053,056 B read**, once each, every served range verified
against an independent read of the same file.
`9 · 11,894,784 = 107,053,056` exactly, and the union is nine rather than the
`3 · 8 = 24` a no-overlap bound would charge. **Nothing was computed with those
bytes. No checkpoint executed. Nothing was written under any checkpoint root,
and no download was made.**

**The device lane, on all three cards.** `GPU-97fe4889` (5060 Ti),
`GPU-3032cfa3` and `GPU-81fe4578` (3090s): 3,145,728 B uploaded into a
2,097,152 B *admitted* expert cache on each, one eviction each, every range
verified by reading it back off the card, and the all-leased refusal
demonstrated against real device memory.

**Uncertainty, stated rather than hidden.** No latency, bandwidth or
throughput number is reported. Document 03 asks for hit latency and end-to-end
impact to be *recorded*; there is no baseline on this machine to compare them
against, and a number without one would be the unsupported performance claim
AGENTS.md forbids. `ResidencyStats` carries the counters; **no performance claim
follows from them.**

### Independent review, and what it changed

**Nine findings. All nine reproduced against this tree; none disputed.** Four
were P1 and one of those was a panic. The review's probes are preserved at
`results/task0020/independent-review-2026-09-12/` (untracked).

| # | Finding | Reproduced as | Fix |
|---|---|---|---|
| 1 | **Device allocations were not bound to their reservation.** `DeviceResidency::create` took a shared borrow and allocated, so two caches could exist against one reservation; `close` then released the charge while both stayed live and readable | **8 MiB live on a real 3090 against a 4 MiB charge**, and readback still worked after `close` | A `DeviceBacking` entitlement, one per scope, claimed before the allocation. `close` refuses while one is out; `return_backing` refuses while placements name ranges inside it, and hands it back so a retry is possible |
| 2 | **Unknown upload completion did not retain the physical host allocation.** `close` refuses over in-flight work, but a *drop* freed the `Vec` a copy may still be reading by address | code inspection, then measured | `HostBuffer::withhold` plus a `Drop` that abandons the pages when anything is in flight. The charge stays visible — document 02's rule, and the same trade `moxie-executor` makes for a lost context |
| 3 | **Cancelling a host reader destroyed a device waiter's source**, then panicked | `no entry found for key` | `drop_if_unheld`: a cancellation retires the *intent*, never bytes a surviving consumer still holds |
| 4 | **Prefetch-to-demand promotion stranded work, and could deadlock.** Promotion de-queued the ticket but returned `Coalesced`, so nobody owned the order; and a device demand waiting on a queued prefetch waited for a read `next_prefetch` refused to release | two probes | Promotion returns the order it de-queued, and priority propagates **through** the dependency: a demand blocking on a queued prediction promotes it and is handed its read |
| 5 | **A released prefetch upload could be issued twice** — returned by `complete_read` and still in the queue | both orders observed for one ticket | A still-queued ticket keeps its upload in the queue; a prediction's upload does not jump the demand queue, and is issued once from there |
| 6 | **Expiring blocked demand left a stale demand count**, blocking the queue forever | prefetch never released after the blocked ticket expired | The counter follows ticket **membership**, not issuance, on both sides |
| 7 | **The control envelope undercounted the real heap** — a flat 512 B per placement | **166,361 B retained against a 49,216 B envelope** | The identity is stored once behind an `Arc` both holders share; its length is declared by `max_identity_bytes` and **enforced on every acquire**; the fixed part is now a measured 1,024 B (measured at 655.7 / 634.1 / 631.3 B for 64 / 256 / 1,024 placements). A gate measures heap against the envelope at the declared bound |
| 8 | **Device admission bypassed `max_placements`** — one check, two placements created | `max_placements(1)` produced two | The check reserves every placement the acquire will create |
| 9 | **Conditional tables were accepted with a zero floor**, making every entry an eviction candidate | accepted | Refused, naming ADR 0009's "never zero-resident" |

**One point of the review I resolved differently, and it is worth stating.**
Finding 5's probe asserts the released upload comes back from `complete_read`.
It does not: it stays in the prefetch queue and `next_prefetch` issues it. The
defect — *the same upload issued twice* — is fixed either way, and a test asserts
the exactly-once invariant directly. Returning it from `complete_read` would let
a prediction's copy jump ahead of demand, which is the one-way rule document 03
states. This is a disagreement about one line of the probe, not about the finding.

The review's closing point is the one worth carrying forward: **"A nonblocking
`acquire` does not establish deadlock freedom."** That was this task's own claim
about its own mechanism, and finding 4 disproved it — the cycle was not in
`acquire` at all but between the demand counter and the prefetch gate. The
claim is corrected below and the property is now tested rather than argued.

Every finding has a regression. Findings 3, 4a, 4b, 5, 6, 8, 9 and the identity
bound are in `crates/moxie-memory/tests/residency.rs`; findings 2 and 7 are
measured in `residency_allocation.rs`; finding 1 is checked on the host lane and
on all three cards in `residency_device.rs`.

### Second independent review

**Seven findings, all P1. All seven reproduced against `5be984f`; none
disputed.** Probes preserved at
`results/task0020/independent-review-2026-09-12-round2/` (untracked).

| # | Finding | Reproduced as | Fix |
|---|---|---|---|
| 1 | **A closed authority still allocated.** `claim_backing` answered from the retained configuration without asking whether the reservation still existed | **4,194,304 B allocated on a real GPU with zero ledger charge** | A `closed` flag set by `close`; `acquire`, `claim_backing`, `retire`, `settle_quarantined`, `complete_read`, `complete_upload`, `next_prefetch` and a second `close` all refuse. A released reservation is not a budget |
| 2 | **A backing accepted another authority's upload.** `perform_upload` checked scope and capacity but never bound the authority to the entitlement | authority B's upload through A's backing **overwrote A's still-leased bytes and marked B ready** | `DeviceBacking::authority()`, checked before either endpoint is touched; `read_back` refuses a closed residency too |
| 3 | **A cancelled device read left a ticketless `Reading` placement**, and the next acquire panicked | **`an in-flight placement owns a ticket`** | The placement is discarded: no upload ran, so it holds nothing. Plus a typed refusal instead of that `expect`, because an invariant breach inside a generation step must be reportable, not fatal |
| 4 | **Promotion still deadlocked through an existing dependency.** It promoted the ticket named and returned early for `BlockedOnRead`, leaving the chain's root queued | demand blocked the gate that had to release the read it waited for | `promote_chain` walks the whole dependency, depth-bounded against a cycle |
| 5 | **Failed device admission lost a promoted read order.** Promotion ran while resolving the source, before the placement was known admissible | the original prediction stranded: off the queue, counted as demand, owned by nobody | Promotion is recorded during resolution and **committed only after admission succeeds** |
| 6 | **The envelope still undercounted small and pending configurations.** The charge was per placement only, and measured against settled placements | **8,274 B retained against 2,368 B admitted** at one pending placement | `AUTHORITY_CONTROL_BYTES` for the fixed part (measured ≈ 7,061 B) and `PLACEMENT_CONTROL_BYTES` raised to 2,048 (a pending placement also owns a live ticket; measured ≈ 1,478 B beyond the identity). The gate now sweeps 1, 2, 4, 16 and 64 **pending** placements |
| 7 | **Failed physical teardown returned the entitlement too early.** `try_free` leaves the allocation live on failure | code inspection | `check_returnable` asks, `try_free` frees, and only then is the entitlement surrendered. A failure at any step keeps it |

**Finding 3 changed a fix from the first round rather than adding to it.** My
first attempt moved held placements to `Retiring`, which would have let a
*failed* placement serve its uninitialised range to the leases still holding it.
The distinction is not who holds a placement but whether its bytes are real: a
read that completed leaves valid bytes others may want and stays; a read that
failed or was cancelled before its data arrived leaves nothing, and is discarded
even under live leases so its holders get a typed error. That is what every
failure test already asserted, and the wrong fix would have broken two of them.

#### The arch-check noise floor

The review also caught something I caused. Copying the first round's probe crate
into `results/` took `arch-check` from four failures to **nine**, and the four it
already had came from a task-0014 review crate parked the same way. Every task
since 0014 has reported "4 pre-existing failures" in its acceptance evidence.

None of them was ever a real violation. `docs/README.md` declares `/results/` and
`/artifacts/` ignored scratch — nothing there is in the workspace or built by
anything — and `find_manifests` walked them anyway. It now skips both, with a
unit test pinning that a crate under `crates/` is still discovered.
**`arch-check` passes with zero failures for the first time.** A standing noise
floor in a pass/fail gate is how a real failure gets read as one of the usual
ones, and this one stood for six tasks.

### Third independent review

**Four findings, three P1. All four reproduced; none disputed.** Two were
panics. Probes at `results/task0020/independent-review-2026-09-12-round3/`.

| # | Finding | Reproduced as | Fix |
|---|---|---|---|
| 1 | **A failed device-owned read left host joiners on a dead ticket.** A host acquire can join a device-owned read and take a lease on the same placement; the failure path released the pin but kept the placement because that lease was live | **panic at `expect("live ticket")`** | A failed read discards its source **whoever else holds it**: the bytes never arrived, so a surviving lease resolves to a typed error, exactly as every other failed read's waiters do |
| 2 | **The prefetch queue could issue a blocked dependency.** Deadline order picked a ticket waiting on another's read | **`unreachable!`: "a blocked ticket has no work of its own"** | `next_prefetch` resolves each candidate to its chain root and issues that; `work_order_for` returns `Option` instead of an `unreachable!` — an unreachable branch on a path a generation step takes is a crash waiting for the right queue order |
| 3 | **The scratch exclusion hid declared production crates.** It skipped any directory named `results` or `artifacts` **at any depth** | a declared model crate with a forbidden `std::fs::read` under `crates/results/model` produced **zero** violations | The skip applies only at the workspace root — `docs/README.md` writes `/results/` and `/artifacts/` with leading slashes — and never to a directory the workspace declares a member under |
| 4 | **Promoting an already-issued prefetch undercounted demand.** The counter tracked the queue transition, not the urgency transition | an unrelated demand's slot was cleared and the gate opened early | Promotion adds a slot exactly when the ticket did not already answer "non-queued demand" |

Finding 3 is the sharper one, because it was my own fix from round two. Narrowing
a check to remove noise is right; narrowing it *by directory name at any depth*
made a real rule unenforceable, and the noise I was removing had itself hidden
nothing only by luck.

### The answer to the pattern

The third round closed with a criticism of method, not another case: *"These
cases continue to expose gaps between individually passing regressions."* That is
exactly right. Every one of the twenty findings had the same shape — a transition
that was individually reasonable left the structure inconsistent in a combination
nobody had hand-written a test for, and the damage surfaced one or two operations
later, usually as a panic. Point regressions caught each case and missed the next,
because the space is a product and my tests were points in it.

Two things now exist that did not:

- **`ResidencyAuthority::check_invariants`**, which states the structural
  invariants **once**: no in-flight state without a live ticket, every ticket's
  placement and source and dependency alive, the prefetch queue and the `queued`
  flag agreeing, `outstanding_demand` equal to the number of tickets that answer
  its description, per-scope committed bytes equal to placements equal to arena
  occupancy, the conditional-memory total, and the lease table's count. It is
  public because it is a debugging tool, not a test fixture.
- **`crates/moxie-memory/tests/residency_transitions.rs`**, which enumerates the
  product — destination × urgency × joiner × ending × surviving lease, **200
  combinations** — drives each one, and calls `check_invariants` after *every*
  operation rather than asserting a hand-picked consequence at the end.

**The sweep immediately found a defect none of the three rounds had reached**: a
device prefetch whose host read a host prefetch had joined, ending in a deadline
expiry, left a `Reading` placement with a stale ticket. Same class as finding 1,
different path, and nothing but the product search would have reached it.

What the sweep proves is narrow and worth stating exactly: no reachable sequence
in that space leaves the authority structurally inconsistent, panics, or fails to
drain. It does **not** prove the policy is right — the named tests do that, and
they stay.

### Fourth independent review

**Three findings; all reproduced, none disputed.** Probes at
`results/task0020/independent-review-2026-09-12-round4/`.

| # | Finding | Reproduced as | Fix |
|---|---|---|---|
| 1 | **Releasing an upload's last source pin did not finish retirement.** Retire a host chunk while an upload holds the last pin, then complete the upload | the placement stayed `Retiring` with **zero leases** — charged, unservable, unevictable, holding a one-chunk cache shut. **`check_invariants` passed throughout** | `settle_source` finalises retirement exactly as `unpin` does; an internal pin is no different from a consumer's. And the checker now rejects `Retiring` with no holder, because that state is not retiring, it is stranded |
| 2 | **Arch-check still skipped production code.** Membership was tested against literal `workspace.members` strings | a forbidden second `ExpertCache` in `results/storage`, reached through an allowed `moxie-executor → moxie-storage` **path dependency**, produced **zero violations**; `./results/model` was skipped because its first segment is `"."` | Exclusion is now by **reachability**: normalized member paths, globs expanded, and a transitive walk of production path dependencies. A manifest in root scratch is skipped only when nothing the workspace builds refers to it |
| 3 | **The sweep could conceal stranded work**, so it did not establish its drain claim | mutating `promote_ticket` to discard every order it produced: **all 200 combinations passed**, while a named regression caught it | The harness is now a faithful executor — see below |

### The sweep and what it is worth

Finding 3 is the one that matters, and it was aimed at the thing I built in
round three and described as the answer to the pattern. It was not, and the
demonstration was a mutation test: discard every promoted work order and the
sweep still passed, because the harness discovered work through `ticket_of` and
completed it directly. It was testing whether the authority *can be poked* into
a consistent state, not whether the scheduler ever hands the work out.

What changed:

- The harness now tracks **only the orders it was actually given** — from an
  acquire, from a completion's follow-ons, from `next_prefetch` — performs
  nothing else, and **fails when a ticket is left in flight that it never
  received an order for**.
- Every ending must demonstrably apply; a branch that silently did nothing was a
  duplicate of `Completed` wearing another name. **This was only half done at
  round four** — an escape hatch still excused 120 cases, which round five
  found; it is now required and measured.
- Two axes were added because mutation testing showed their absence: **cache
  pressure** with a lease held (eviction never ran at all before — a mutation
  making eviction ignore leases survived), and **retirement while a transfer is
  outstanding**, with the chunk warmed to `HostReady` first so an upload's pin
  can be the last holder.
- `check_invariants` gained an **exact pin balance**: a placement's lease count
  equals its consumer leases plus one pin per ticket copying out of it plus one
  per quarantined placement withholding it. Stated as an identity rather than a
  bound, it catches a pin released twice or never released — a leaked pin makes
  a chunk permanently unevictable and is invisible until a cache stops admitting.

**800 combinations**, every step checked. And rather than assert that it is
strong, here is what it measures against a battery of ten deliberate mutations
of the authority:

| Mutation | Caught by the sweep |
|---|---|
| Promoted work order discarded | yes |
| Source pin never released | yes |
| Source pin released twice | yes |
| Demand counter never decremented | yes |
| Cancelled read keeps a held placement | yes |
| Eviction ignores leases | yes |
| Retirement never finalises | yes |
| `unpin` skips retiring finalisation | yes |
| Failed read spares its source | yes |
| `next_prefetch` ignores dependencies | **no** — caught by its named regression instead |

Nine of ten, with the tenth covered by
`the_prefetch_queue_issues_a_dependency_before_the_ticket_waiting_on_it`. That
last one is worth keeping visible: the sweep is a complement to the named tests,
not a replacement, and saying so is the honest version of the claim I made in
round three.

### Fifth independent review

**Two findings; both reproduced, neither disputed.** Both were about things this
record already claimed.

| # | Finding | Reproduced as | Fix |
|---|---|---|---|
| 1 | **Arch-check missed inherited workspace path dependencies.** The crate walk followed only entries with a direct `path`, so `{ workspace = true }` — whose path lives in `[workspace.dependencies]` — resolved to nothing | a forbidden second `ExpertCache` under `results/storage`, reached by `moxie-storage = { workspace = true }`, produced **zero violations** | The walk now resolves inheritance through **the same `effective_spec` the edge checker uses**, extracted so there is one copy. A second implementation of dependency resolution is a second set of its bugs, and this was the proof |
| 2 | **The sweep counted cases whose ending never happened.** The early prefetch drain completed work before the outcome was injected, and an escape hatch then excused the ending for doing nothing | **120 cases** — 40 `Failed`, 40 `SubmissionUnknown`, 40 `Expire` — while the record claimed every ending applies | Orders are now **received without being performed** until the outcome is injected, the escape hatch is gone, and the sweep **reports its coverage** instead of asserting it |

**Finding 2 is the third time a coverage claim of mine has been wrong**, and the
pattern in all three is the same: I asserted a property of the tests rather than
measuring it. The sweep now prints what it exercised —

```text
800 transition combinations, every step invariant-checked:
800 applied their ending, 0 were short-circuited
```

— and the warm preamble that used to turn 100 host cases into cache hits is now
applied only where it creates the scenario it exists for. The claim and the
measurement are the same statement.

Re-measured mutation battery, after the round-5 changes:

| Mutation | Caught by the sweep |
|---|---|
| Promoted work order discarded | yes |
| Source pin never released | yes |
| Source pin released twice | yes |
| Demand counter never decremented | yes |
| Cancelled read keeps a held placement | yes |
| Eviction ignores leases | yes |
| Retirement never finalises | yes |
| `unpin` skips retiring finalisation | yes |
| Failed read spares its source | yes |
| Quarantine loses its source | yes |
| `next_prefetch` ignores dependencies | **no** — caught by its named regression |

Ten of eleven, the eleventh covered by
`the_prefetch_queue_issues_a_dependency_before_the_ticket_waiting_on_it`. One
further mutation was written and then discarded as an **equivalent mutant**:
removing the `upload_source` re-assignment on the quarantine path changes
nothing, because `admit_new` has already set it. Saying so is part of reporting
a battery honestly — an equivalent mutant surviving is not a coverage gap, and
counting it as one would inflate the number in the other direction.

### Sixth independent review

**One finding, reproduced, not disputed. No residency defect this round.**
Probe at `results/task0020/independent-review-2026-09-12-round6-arch/`.

**Valid workspace globs could still hide production crates.** The member
expansion handled only patterns ending in `/*`; everything else was taken
literally. `members = ["results/mo*"]` therefore resolved to a directory that
does not exist, so the real `results/model` — holding a `moxie-models` crate with
a forbidden `std::fs::read` — was *unreachable*, so it was excluded, so every
rule stopped applying to it. Cargo builds that crate; `arch-check` reported zero
violations.

The reviewer also noted that the comment I had written there was simply false:
it claimed literal treatment "errs toward checking more rather than less". It
errs the other way, which is the whole defect.

#### The check that was wrong four times

This is the same check, wrong a third time, each time by a different route:

| Round | What hid a crate |
|---|---|
| 3 | skipping any directory *named* `results`/`artifacts` at any depth |
| 5 | membership tested against literal `workspace.members` strings, missing `{ workspace = true }` inheritance |
| 6 | member globs other than a trailing `/*` taken literally |
| 7 | the crate walk read only ordinary dependency tables, missing **build dependencies** |

All four share one root: I kept computing "is this crate part of the build?"
with a cheaper approximation than the question deserves, and every approximation
failed *open* — quietly excluding real code rather than including scratch.

Round 7 is the sharpest of them, because it is the round-5 lesson applied
incompletely. Round 5 said "a second implementation of dependency resolution is a
second set of its bugs", and I extracted `effective_spec` so *inheritance* had one
reading — while leaving my own duplicate of the **table enumeration** one function
away, reading `[dependencies]` where the real one reads `[dependencies]`,
`[build-dependencies]` and both target variants. A build dependency is how
generated code and kernel compilation get in; this file already said so, in a
comment I had read.

So the answer is no longer an approximation that must be correct. `expand_member`
handles `*` and `?` per path segment exactly, and returns `None` for anything it
cannot — `**`, a character class, a brace set. `None` propagates: reachability is
**unknown**, and an unknown reachability excludes nothing and checks everything.
Guessing wrong now costs a probe crate appearing in the report; before, it cost a
production crate disappearing from every rule.

So there is now **one** enumeration, `production_dependency_sections`, and both
consumers use it; the duplicate is deleted. The tests pin the *property* rather
than the instance — `reachability_and_edge_checking_read_the_same_tables`
asserts the two consumers agree on which tables count, with build dependencies in
and dev dependencies out — alongside the build-dependency scenario, glob
expansion, fail-closed behaviour over three unexpandable patterns, and the
segment matcher over eleven cases. **All five earlier rounds' architecture probes
still pass.**

### Seventh independent review

**One finding, reproduced, not disputed. No residency defect, and no runtime code
changed.** Probe at
`results/task0020/independent-review-2026-09-12-round7-arch/`.

**Reachability excluded build dependencies that the rules classify as
production.** The crate walk enumerated `[dependencies]` and
`[target.<cfg>.dependencies]`; `production_deps` — the function the rules
actually use — enumerates those *and* `[build-dependencies]` and
`[target.<cfg>.build-dependencies]`. A forbidden second `ExpertCache` under
`results/storage`, reached through a build dependency, compiled under
`cargo check` and produced zero violations.

It is folded into the table above, because it is the same failure a fourth time.

### Acceptance

**Accepted 2026-09-12.** The seventh review closed with "no new blocking
findings in `db6b613`. I recommend accepting task 0020 within its declared M2
item 2 scope", verifying all three architecture probes, 53 xtask tests,
arch-check and spec-check, and noting that no runtime code had changed since the
GPU verification of earlier rounds. The owner directed the acceptance be
recorded.

What the acceptance covers, in the review's own framing: **this residency
assignment, not M2.** Grouped expert execution, CPU fallback and checkpoint
execution remain outstanding, and the next bounded task is 0021.

Seven rounds found **twenty-seven** issues. All twenty-seven were reproduced and
fixed; **none was disputed**. Four of them were in this task's own enforcement
tooling rather than in the authority, and three were criticisms of method or of a
coverage claim rather than of code — each correct. The last two rounds found no
residency defect at all.

### What was **not** delivered, and why

Three deviations from this contract, each a narrowing decided during
implementation rather than an omission:

1. **`Artifact::read_tensor_range` was not added.** The contract named it; only
   `Shard::read_tensor_range` exists. A canonical artifact's manifest carries a
   **whole-tensor** checksum, which a ranged read cannot verify, so a canonical
   ranged read would silently skip the integrity check the canonical format
   exists to provide. `Shard` never had a checksum to skip. Per-chunk checksums
   are M3's repacker work, and the canonical ranged read belongs with them.
2. **The `Preparing` / `PreparedHostReady` / `PreparedDeviceReady` states are
   not implemented.** Nothing prepares a layout — that is M3's repacker — and an
   unreachable state is a stub, which AGENTS.md refuses to count as a delivered
   feature. `PreparedId` exists, with a test proving two ids differing only in
   `layout_version` are different keys, so adding the states later is a state
   rather than a redesign.
3. **The prefetch class has no predictor.** By design and as the contract said:
   the class, its bounded queue, its ordering behind demand and its one-way
   eviction rule are all implemented and tested, and the list of what to
   prefetch is supplied explicitly. Document 03 permits a smarter policy "only
   with replayable route traces and measured benefit", and neither exists.

### A defect this work found in itself

The first version of the authority gave each device an `Arena` and a capacity
and **admitted neither from the ledger**. Byte accounting was internally
consistent, the API was confident, and it was a placement simulator — R02 in
miniature, and one of this task's own stop conditions ("simulated placement
presented as a reservation"). The device-lane test caught it on its first run,
by asserting the ledger's `device.expert_cache` charge before allocating
anything. It is fixed: every declared device cache is admitted as **one** plan
before the host bytes are allocated, a refusal anywhere in `open` leaves the
ledger exactly as it found it, and two host-lane regressions now fail if either
property comes back —
`every_device_cache_is_reserved_in_the_ledger_before_a_byte_of_it_exists` and
`a_device_cache_the_ledger_cannot_admit_leaves_no_host_charge_either`.

**A self-review pass before hand-off found three more**, all in the lifecycle's
least-travelled corner — a device acquire that arrives while another ticket is
already reading the same chunk to the host. Nothing exercised that path, which is
why the defects survived the first round of tests, and it now has four of its own.

1. **`Retiring` was a dead state.** Eviction only ever chooses unleased
   placements, so nothing ever entered it — the same stub problem this record
   refuses for `Preparing`. It is now reachable through an explicit
   `retire(scope, chunk)`, which is the `release` half of document 03's pair
   ("`release` retires only after all consumers complete"): live leases keep
   reading, new acquires are refused, the placement is not an eviction
   candidate, and the bytes come back when the last consumer does. Freeing them
   when the decision was made would be the use-after-free document 02 forbids.
2. **Expiring a ticket that had *joined* another ticket's read destroyed that
   read's destination.** Cleanup treated every ticket as the owner of its
   source, so it freed an arena range a read was still writing into. A ticket
   now records whether it created its source, and only the owner may drop it.
3. **A cancelled upload that then completed released its source pin twice**, so
   a host chunk another consumer was holding silently became evictable
   underneath them. Pin settlement is now one function called exactly once on
   every terminal transition, and the invariant it enforces is written where it
   lives.

Both of the last two have regressions proven load-bearing by substitution:
reintroducing each defect fails exactly the test written for it and no other.

Two further findings, both from tests whose first expectation was wrong and
whose second is the real mechanism:

- **Eviction needs two phases.** Byte accounting says whether the cache *holds*
  enough; the arena says whether the free bytes are in one piece. When they
  disagree that is fragmentation, which document 03 counts as real capacity, so
  the authority keeps evicting in the same deterministic order rather than
  reporting a full cache that is not full. The test asserts both numbers: two
  victims satisfy the bytes, four satisfy the contiguity.
- **The hit path allocated.** Keying the index by `(Scope, ChunkId)` meant every
  hit cloned two `String`s to ask whether a chunk was resident, and the lease
  map allocated a node per acquire. The index is now nested by scope and the
  lease table is a pre-admitted slab; the allocation gate measures zero.

### Deleted or replaced paths

None. No temporary path was created and none was retired. `HostBuffer`'s two
existing constructors keep their tiers and their behaviour; the new
`allocate_in` is expressed through the same private function.

### Remaining blockers and the next bounded task

- **M2 is not closed.** Its exit needs "a real out-of-device-memory working set
  [that] executes without OOM or hidden allocations, matches the reference, and
  produces byte/cost traces reconciled with the resource ledger". This task
  delivers the residency half and the ledger reconciliation; **nothing executes**
  a routed layer on a device, because that is M2 item 3 — CPU expert fallback
  and GPU grouped candidate plans under one interface — and the selected BF16
  chain still refuses `Route`, `ExpertMlp` and `Combine` as `UnsupportedKernel`.
- **M2 item 4** is untouched: Laguna metadata and graph only, plus the
  intentionally restricted budget against a second synthetic MoE.
- **Quality is O2** and needs paired output against the released model. Nothing
  here is a quality claim.
- **No owner gate was resolved.** O1–O7 remain open. No numerical threshold,
  precision, context target or compatibility surface changed.
- **Next task: 0021**, M2 item 3. Its contract is sketched in the accompanying
  handover.
