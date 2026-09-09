# Task 0008 — M1.3: measured host capacity

Status: **contract proposed**, 2026-09-09, after [task 0007](0007-m1-rank-context-and-measured-capacity.md)
was accepted for the rank-owned context and measured device-capacity slice.

**This contract is committed before any implementation code**, as in tasks 0003–0007. That commit
contains no `.rs` change. Nothing below is to be adjusted once a test has run.

## Identity and authority

- Task ID / milestone / owner: 0008 / M1.3 / implementation agent (Claude), owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `e8c5e4a`, tree clean
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Required documents: 03 (host admission, the 251 GB warning, page-cache pressure, the swap rule),
  02 (the ownership table, and resolving shared descriptor types into lower-level crates), 06 (M1.3),
  07, 08 (R11), AGENTS.md, [ADR 0006](../decisions/adr/0006-host-telemetry-owner.md), and tasks
  [0006](0006-m1-resource-ledger-and-admission.md) and
  [0007](0007-m1-rank-context-and-measured-capacity.md) for what the ledger and the device sensor
  already guarantee.
- Owner gates: **none needed.** Reading `/proc/meminfo` and the cgroup files under `/sys/fs/cgroup`
  is read-only introspection of the machine the engine already runs on: no privilege, no mutation,
  no network, no driver or system change, and nothing written anywhere. It is not the "material
  operational expansion" AGENTS.md gates. O5 is untouched — no checkpoint is read, downloaded or
  converted. **O6 is not reached**: a byte count of host capacity is not a throughput, a latency or
  a regression, and nothing here may be presented as one. **Stop and ask** if the work appears to
  need any gate.

## Why this next

The ledger admits three measured devices and a host budget nobody measured. Document 03 requires
host admission to reserve headroom "from measured available memory" and warns, in as many words,
against treating all 251 GB as an expert cache — and today the host snapshot is whatever a caller
says it is.

M2's entire workload is an oversized MoE whose working set lives on the host. Admitting that against
a fabricated host budget would be R02 on the other tier: an authority that models a resource it does
not own. The device half stopped being a model at task 0007; this is the other half.

Which component may take the measurement is settled in [ADR 0006](../decisions/adr/0006-host-telemetry-owner.md)
and summarised below. It is an implementation decision, not an owner gate.

## Bounded deliverable

- **One concrete outcome:** this machine's host memory is measured, becomes a `CapacitySnapshot`,
  and is admitted against in one ledger alongside the three device snapshots, end to end through
  `cargo xtask-cuda capacity`. Nothing is allocated, mapped or reclaimed.
- **Sole owning components:** a new `moxie-host` owns machine telemetry (ADR 0006); `moxie-types`
  gains the `MeasuredHost` descriptor beside `MeasuredDevice`; `moxie-memory` gains the rule that
  turns that descriptor into a budget; `xtask` wires them, as it already does for devices.
- **Allowed production files:** `crates/moxie-host/**` (new), `crates/moxie-types/src/capability.rs`
  and `src/lib.rs`, `crates/moxie-memory/src/snapshot.rs`, `xtask/src/archcheck.rs`,
  `xtask/src/capacity.rs`, plus manifests, `Cargo.lock` and `xtask/fixtures/**`.
- **Explicit non-goals and forbidden shortcuts:** no allocation, mapping, `mmap` or `madvise`; no
  page-cache eviction and no pressure-driven replan (document 03 asks for both and both are M2); no
  swap counted as budget under any circumstance; no NUMA-aware placement (M2); no periodic or
  background re-measurement; no checkpoint, importer or model metadata; no change to the ledger's
  admission rules; and no figure from this task presented as a performance result.
- **Existing consumers and second-consumer proof:** the ledger from task 0006 and the device sensor
  from task 0007. The proof is one ledger holding three device snapshots and one host snapshot, and
  a plan that places weights and KV on a device *and* state spill and CPU workspace on the host,
  admitted and released as one envelope.
- **Deletion plan:** none, and that is stated rather than invented. Nothing reads host telemetry
  today, so nothing is superseded. `CapacitySnapshot::new` stays: a caller-supplied host snapshot is
  what tests use, and after this task it is no longer what the composition root uses.

## Contract, fixed before implementation

### Where the boundary sits

Per ADR 0006: `moxie-host` is the only crate that may name a `/proc` or `/sys` path, and
`moxie-memory` may not depend on it. The ledger never probes — task 0006 documented
`Ledger::preview` as pure with respect to live resources, and a path from the ledger to a sensor
would make that a promise rather than a structure.

### What is read, and what each number means

From `/proc/meminfo`: `MemTotal`, `MemAvailable`, `MemFree`, `Buffers`, `Cached`, `SwapTotal`,
`SwapFree`. Values are in kB in the file and are converted to bytes exactly, checked.

**`MemAvailable` is the budget input, not `MemFree`.** It is the kernel's own estimate of what a new
allocation can obtain without swapping, and it already accounts for reclaimable page cache. On this
machine the two differ by about 5 GiB, and on a machine with a warm cache they differ by far more.

**A missing `MemAvailable` is an error, not a computed fallback.** Kernels before 3.14 lack it and
the substitute formula is an implementation detail that has changed. Guessing it would be a private
model of the kernel's reclaim behaviour presented as a measurement, which is the shape AGENTS.md
forbids.

### The cgroup v2 effective limit

A process may be held far below `MemTotal` by a cgroup limit, and admitting against the machine
would then over-promise by orders of magnitude. The effective limit is the **minimum of `memory.max`
over the process's own cgroup and every ancestor**, read from `/proc/self/cgroup` and
`/sys/fs/cgroup`; `max` means no limit at that level. Where a limit applies:

```text
effective_total     = min(MemTotal, limit)
effective_available = min(MemAvailable, limit - current)     (saturating at zero)
```

Absence is not an error: a cgroup v1 machine, an unmounted hierarchy or an unreadable file means the
machine view governs, and the descriptor records which view was used so a report can say so. On the
benchmark machine every level reads `max` today, so the limited paths are proven by fixtures.

### Swap is never in the budget

`swap_total` and `swap_free` are read and reported so that a reader can see the exclusion was
deliberate, and they enter no arithmetic. Document 03: do not rely on uncontrolled swap as an
invisible fourth execution tier.

### The descriptor

```text
MeasuredHost {
  total_bytes, available_bytes            // effective, after any cgroup limit
  machine_total_bytes, machine_available_bytes   // before it, for the report
  free_bytes, buffers_bytes, cached_bytes
  swap_total_bytes, swap_free_bytes
  limit: Machine | Cgroup { path, limit_bytes, current_bytes }
}
```

In `moxie-types`, beside `MeasuredDevice`, for the reason ADR 0006 gives. Everything except
`total_bytes` and `available_bytes` is a diagnostic: reported, never spent.

Like a device reading, this is **capacity at an instant, not a reservation**. Another process can
take memory a moment later and this descriptor will not know. Document 03's answer is a
pressure-driven replan, which is M2's; here it is a documented limit.

### Reading is injectable

`MeasuredHost::read()` reads under `/`. `MeasuredHost::read_under(root)` takes a directory, so every
rule above is tested against committed fixture trees rather than against whatever this machine
happens to look like. A limit that does not exist here is still tested.

### From measurement to snapshot

`CapacitySnapshot::measured_host(&MeasuredHost, extra_reserve_bytes)` in `moxie-memory`:

```text
scope           = Scope::Host
physical_bytes  = total_bytes
system_headroom = (total_bytes - available_bytes) + extra_reserve
admissible      = available_bytes - extra_reserve
```

A reserve larger than `available_bytes` is an error. `available_bytes` greater than `total_bytes` is
an error, not a wrap. The existing rule stands and is load-bearing here: a host snapshot whose total
headroom is zero is refused, so nothing can declare a machine with nothing in use and no reserve.

## Error metrics

None. Every quantity is an exact integer byte count read from a file, and every conversion and sum
is checked. The numerical contracts of tasks 0003, 0004, 0006 and 0007 are unchanged.

## Acceptance

- Host lane, against committed fixture trees: this machine's shape; a cgroup limit that binds; a
  limit on an **ancestor** rather than the leaf; `memory.max = max` at every level; no cgroup mount
  at all; a machine with large swap, which changes no budget figure; a missing `MemAvailable`,
  refused; and a malformed unit, a non-numeric value and a truncated file, each refused with the
  field named.
- One test per `measured_host` rule above, including the reserve and ordering errors.
- `cargo xtask arch-check` passes with `moxie-host` declared and **two new rules** — machine
  telemetry outside `moxie-host`, and `moxie-memory` depending on `moxie-host` — each exercised by a
  rejecting fixture and by a clean accepted one.
- `cargo xtask-cuda capacity` measures the host and all three devices into one ledger and admits a
  plan spanning both tiers: weights and KV on a device, state spill and CPU workspace on the host.
  It releases everything and fails if a byte is still held.
- This machine's real `/proc/meminfo` is read once and its figures recorded in the result, with the
  cgroup view named.
- `fmt`; `clippy -D warnings` on **both** lanes; the full host suite; `spec-check`; the no-driver
  lane with `xtask` rebuilt before `ldd`; the device lane; `test-gpu` with both architectures
  qualified; and the `CUDA_VISIBLE_DEVICES=1,2` negative check still exiting 1.
- Support matrix: the host capacity row moves from unmeasured to measured. It must say that nothing
  is allocated, that swap is excluded from the budget on purpose, and that page-cache pressure is
  reported rather than managed.
- Stop condition: stop and report if this appears to need an allocator, a mapping, `madvise`,
  page-cache eviction, a pressure-driven replan, swap counted as budget, NUMA placement, or a
  checkpoint. Each is a different task.

## Result, filled after work

To be filled in after implementation, keeping passed, failed and skipped separate.
