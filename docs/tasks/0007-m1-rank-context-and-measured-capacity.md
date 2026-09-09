# Task 0007 — M1.3: rank-owned CUDA context and measured device capacity

Status: **contract proposed**, 2026-09-08, after [task 0006](0006-m1-resource-ledger-and-admission.md)
was accepted for the accounting and admission slice.

**This contract is committed before any implementation code**, as in tasks 0003–0006. That commit
contains no `.rs` change. Nothing below is to be adjusted once a test has run.

## Identity and authority

- Task ID / milestone / owner: 0007 / M1.3 / implementation agent (Claude), owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `4bbe659`, tree clean
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Required documents: 03 (resource ledger and admission; transfer and CPU/GPU policy), 02 (the
  ownership graph and the buffer and asynchronous lifetime contract), 01 (rank-owned contexts,
  one rank execution thread per GPU), 06 (M1.3), 07 (the two lanes, and what a measurement may be
  claimed to mean), 08 (R02, R07, R08, R12), AGENTS.md, and
  [task 0006](0006-m1-resource-ledger-and-admission.md) for what the ledger already guarantees.
- Owner gates: **none needed.** Nothing is read from `/models` or `/fast/models`, nothing is
  downloaded, converted or written, so O5 is untouched; no catalog or quality claim, so O1 and O2 are
  untouched. **O6 is not reached**: this task measures *capacity*, which is how many bytes a device
  has and how many are free. That is not a throughput, a latency or a regression, and no number it
  produces may be presented as one. **Stop and ask** if the work appears to need any gate.

## Why this next

Task 0006 built the authority that admits bytes and left it unable to learn anything about this
machine: every capacity figure in the workspace is written by a test. M1.3's next item is the
rank-owned CUDA context, and the context is the thing that can ask a device what it has. Doing them
together closes the smallest honest loop — a real device, a real reading, a real admission decision —
without allocating anything through it.

It is also the first task where R02's failure has a second half to avoid. Legacy's residency manager
modelled bytes for a simulator; task 0006 could still be described that way, because nothing it
admits corresponds to a device. After this task it does.

## Bounded deliverable

- **One concrete outcome:** one rank owns one device's context, identified by its UUID, and that
  context produces a measured `CapacitySnapshot` that the ledger admits a declared plan against, end
  to end, on this machine's three GPUs. Nothing is allocated through the context.
- **Sole owning components:** `moxie-cuda` owns the context and the measurement; `moxie-memory` owns
  the rule that turns a measurement into a snapshot; `moxie-types` owns the descriptor that passes
  between them, because document 02's graph puts `memory` above `cuda` and neither may import the
  other's higher layer. The composition root — the `xtask` device command — wires them.
- **Allowed production files:** `crates/moxie-cuda/src/**`, `crates/moxie-types/src/**`,
  `crates/moxie-memory/src/snapshot.rs`, `xtask/src/gpu.rs` and `xtask/src/main.rs`, plus manifests
  and `xtask/fixtures/**`.
- **Explicit non-goals and forbidden shortcuts:** no allocator and no allocation charged to the
  ledger; no lease, no event retention over real bytes, no eviction; no kernel launch and no module
  load beyond what already exists; no host memory measurement (see below); no plan compilation; no
  checkpoint, importer or model metadata; and no number from this task presented as a performance
  result.
- **Host measurement is deliberately excluded.** Document 03 requires host admission to reserve
  headroom "from measured available memory", and reading `/proc/meminfo` is filesystem access that
  `moxie-memory` is forbidden and `moxie-storage` has no business owning. Which component may
  measure the host is a real ownership question and it gets its own task. Until then the host
  snapshot stays caller-supplied, and no claim is made that host admission is measured.
- **Existing consumers and second-consumer proof:** the ledger from task 0006 is the consumer, and
  the proof is that three physically different devices — two `sm_86` and one `sm_120`, on two NUMA
  nodes, with unequal memory — each produce their own snapshot and are admitted against
  independently in one ledger.
- **Deletion plan, part of the deliverable:** `DeviceCapability::uuid` is a `String` and
  `moxie_cuda::status::format_uuid` builds it. Both go: the field becomes `DeviceUuid`, and the one
  spelling of a device UUID in the workspace is `DeviceUuid`'s parse and `Display`. Two ways to write
  a device's identity means the task is not done — this is task 0005's `sha256_hex` gate again.

## Contract, fixed before implementation

### Rank ownership

`RankContext` replaces `DeviceContext` as the public type. One rank owns one device and one device
answers to one rank: acquiring a context for a device that already has one is a typed error naming
the holding rank, and so is acquiring a second device for a rank that already holds one. The
registration is process-wide and keyed by `DeviceUuid`, released when the context drops, so a later
rank can take the device.

The context stays bound to the thread that created it. It is already `!Send` and `!Sync` through its
raw handles; the contract makes that deliberate rather than incidental, with a `compile_fail`
doctest, verified with `compile_fail` removed.

Document 01 asks for "one rank execution thread per GPU" and for unsafe CUDA state to be isolated
behind rank-owned contexts. A second context for one device is exactly the shared mutable state that
isolation is supposed to remove, and today nothing prevents it.

### The measurement

```text
MeasuredDevice {
  uuid, ordinal_label, name, sm, total_bytes, free_bytes, multiprocessor_count, pci_bus_id
}
```

Lives in `moxie-types`. `RankContext::measure()` fills it from the driver **after** the context
exists, because retaining a primary context itself costs device memory, and a snapshot taken before
it would over-promise.

`free_bytes` is what the driver reports at that instant, including memory other processes hold. It
is a reading, not a reservation and not a promise: re-measuring may return something different, and
nothing in this task may treat one reading as a property of the device. `ordinal_label` is
diagnostic only — every decision, record and map key uses `uuid` (AGENTS.md).

### From measurement to snapshot

`CapacitySnapshot::from_measurement(&MeasuredDevice, extra_reserve_bytes)` in `moxie-memory`, pure
and host-testable:

```text
scope            = Scope::Device(measurement.uuid)
physical_bytes   = measurement.total_bytes
system_headroom  = (total - free) + extra_reserve      (checked; each step)
admissible       = free - extra_reserve
```

The headroom therefore carries two different things and says so: what is already gone on this card,
and what this engine chooses to leave alone. `extra_reserve_bytes` greater than `free_bytes` is an
error, not a zero budget. `total < free` from a driver that misreports is an error, not a wrap.

### What is still not true afterwards

Nothing is allocated. The ledger admits against a real number and still owns no memory. The
support-matrix row must say exactly that, and must not be readable as "the engine allocates on the
GPU".

## Acceptance

- Host lane: `from_measurement` has one test per rule above, on synthetic `MeasuredDevice` values,
  with no driver present. `fmt`, `clippy -D warnings`, the full host suite, `spec-check` and the
  no-driver lane pass.
- `arch-check` passes with the ownership graph unchanged, plus a **new rejecting fixture** proving
  `moxie-cuda` taking a dependency on `moxie-memory` is refused. The wiring belongs to the
  composition root, and the checker should say so rather than the reviewer.
- Device lane, on this machine, re-run and reported as new measurements — not carried forward:
  - a second context for a device already held is refused, naming the holding rank; the refusal
    leaves the first context usable;
  - a rank that already holds a device is refused a second one;
  - dropping a context releases the device, and a later rank acquires it;
  - `measure()` after allocating a device buffer reports less free memory than before it, proving
    the reading is live rather than a constant;
  - all three GPUs are measured in one process, produce three distinct UUID scopes, and are admitted
    against independently in one ledger;
  - the same three snapshots are produced with `CUDA_VISIBLE_DEVICES` reordered, and each UUID keeps
    its own total memory. An ordinal that moves must not move a capacity with it.
- A `cargo xtask-cuda` command measures every visible device, builds a ledger from the measurements,
  admits a small declared plan and prints the admission report. Its output is evidence for the
  support matrix and for the record, and it is the end-to-end proof that the two halves meet.
- `cargo xtask-cuda test-gpu` passes with both architectures qualified, and the
  `CUDA_VISIBLE_DEVICES=1,2` negative check still exits 1.
- `format_uuid` no longer exists, and `DeviceUuid` is the only way a device identity is written.
- Support matrix: one row for measured device capacity. It must say that nothing is allocated, that a
  reading is not a reservation, and that host capacity is still unmeasured.
- Stop condition: if this appears to need an allocator, a lease, an event over real bytes, an
  eviction decision, a kernel launch, host memory measurement, or a checkpoint, stop and report.
  Each is a different task.

## Result, filled after work

Implemented 2026-09-08 on branch `main`, on top of the contract commit `689f58f`.
The contract above is unchanged; only this section is filled in.

What was built:

- `moxie-types`: `MeasuredDevice` (uuid, ordinal *label*, name, sm, bus id, SM
  count, total and free bytes) and `DeviceCapability::uuid` typed as
  `DeviceUuid`. The descriptor sits at the bottom of the graph because the crate
  that takes the reading and the crate that turns it into a budget are on
  opposite sides of `memory` -> `cuda`.
- `moxie-cuda`: `RankContext` replaces `DeviceContext`. `acquire(rank, ordinal)`
  registers the claim in a process-wide map keyed by `DeviceUuid`, refuses a
  device another rank holds and a second device for a rank that has one, and
  releases the claim on drop. A failed attach removes the claim rather than
  stranding the card. `measure()` reads through the live context.
- `moxie-memory`: `CapacitySnapshot::measured`, pure and host-tested. The
  headroom carries what is already gone on the card *and* what this engine
  chooses to leave alone, and says which is which.
- `xtask capacity`: the composition root. It measures every visible device,
  builds a ledger from the measurements, admits a declared plan against each
  device, prints the breakdown, releases, and fails if a byte is still held.
- Deletion done: `format_uuid` no longer exists. `DeviceUuid`'s `Display` and
  `parse` are the only way a device identity is written; the published byte
  vector moved to `moxie-cuda::status`'s tests, next to the code that receives
  those bytes from the driver.

Measured on this machine (readings, not reservations, and not performance):

| Device | UUID | total | free | admissible after a 64 MiB reserve |
|---|---|---|---|---|
| RTX 5060 Ti, `sm_120`, ordinal 0 | `GPU-97fe4889-...-2e72c4a3` | 15,886 MiB | 15,747 MiB | 15,683 MiB |
| RTX 3090, `sm_86`, ordinal 1 | `GPU-3032cfa3-...-4911e0b9` | 24,123 MiB | 23,858 MiB | 23,794 MiB |
| RTX 3090, `sm_86`, ordinal 2 | `GPU-81fe4578-...-dac78704` | 24,123 MiB | 23,858 MiB | 23,794 MiB |

Each admitted a 3,920 MiB weights + 1,960 MiB KV plan and released it in full.

Deviations recorded, none silent:

- The 64 MiB engine reserve in `xtask capacity` is a **placeholder constant**,
  named `ENGINE_RESERVE_BYTES` and documented as one. Document 03 requires the
  legacy 48 MiB and two-largest-linears constants to become derived
  reservations, and this is neither: it exists so the command demonstrates a
  non-zero engine reserve reaching the ledger. Nothing in a production crate
  contains it.
- One **pre-existing defect found and fixed**: the device lane had never been run
  under `clippy -D warnings`, and two `unsafe` blocks in `xtask/src/gpu.rs`
  failed `undocumented_unsafe_blocks` -- one comment said "SAFETY of both async
  copies" without the colon the lint looks for, and one had a statement between
  the comment and its block. Both are in this task's allowed files. Confirmed
  pre-existing by running the lint at `4bbe659` before any change: two findings,
  the same two. The device lane is now clippy-clean and should stay in the gate
  list.
- `xtask` gains `moxie-memory` as a dependency, which the arch-check allowlist
  now permits for `xtask` only. A new rejecting fixture proves the same edge is
  refused for `moxie-cuda`.

Gates (this machine, 2026-09-08):

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | PASS |
| the same clippy **with the device features** | PASS (new to the gate list; it had never been run) |
| `cargo test --workspace --locked --offline` | PASS, 464 unit/integration + 6 doctests (was 460 + 6) |
| `cargo xtask arch-check` | PASS, 46 rejected + 12 accepted fixtures, 10 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane | PASS, 464 + 6; `ldd target/debug/xtask` shows no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | PASS, 472 + 8 |
| `cargo xtask-cuda test-gpu` | PASS, **21 cases** (7 per device, up from 5), `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, `UNQUALIFIED sm_120`, as intended |
| `cargo xtask-cuda capacity` | PASS, 3 devices measured and admitted against |
| `CUDA_VISIBLE_DEVICES=2,1,0 cargo xtask-cuda capacity` | PASS, and each UUID kept its own memory total |

Nothing failed. Nothing was skipped.

The reordering result is the one worth reading twice. With the visible set
reversed, ordinal 0 is `GPU-81fe4578` at 24,123 MiB and ordinal 2 is the 5060 Ti
at 15,886 MiB -- the ordinals moved and no capacity moved with them. That is
AGENTS.md's warning turned into a passing case rather than a comment.

The two new device cases:

- `rank_context_is_exclusive`: a second rank is refused a held device and the
  refusal names both the device and the holder; the holder is still usable
  afterwards; one rank cannot take a second device; dropping releases the claim
  and rank 7 acquires the same card.
- `measurement_is_live`: allocating 64 MiB on the device makes the next reading
  report less free memory, and total memory does not move. Without this, the
  number reaching the ledger could be a constant and every admission decision
  made from it would be fiction.

No checkpoint was read, downloaded or converted. Nothing was allocated through
the ledger, and no performance number was produced: every figure here is a byte
count of capacity. The stop condition was not triggered.

Remaining blockers and next bounded task: **host capacity is still unmeasured**,
and which component may read `/proc/meminfo` is the open ownership question that
task's contract has to answer first. After it, M1.3 continues with event-backed
leases and the basic allocator -- the first things that will charge real bytes to
this ledger.
