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

To be filled in after implementation, keeping passed, failed and skipped separate.
