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
  between them. Document 02 permits `memory` -> `cuda` and forbids the reverse, so the constraint is
  one-directional: `moxie-cuda` may not name a `CapacitySnapshot`. The descriptor goes to the bottom
  of the graph so `cuda` never reaches upward, and the composition root — the `xtask` device
  command — does the wiring.
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
  `DeviceUuid`. The descriptor sits at the bottom of the graph because
  `moxie-cuda` may not name a `CapacitySnapshot`: document 02 permits
  `memory` -> `cuda` and forbids the reverse.
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

### Review corrections, 2026-09-08 (commit `15d136c` not accepted)

Two findings, both reproduced before any fix.

- **P2 — the rank claim was released before the CUDA teardown finished.**
  `Drop` removed the registry entry and *then* called
  `cuDevicePrimaryCtxRelease_v2`, so another thread could acquire the card while
  the previous rank's primary-context reference was still outstanding. Two live
  primary contexts on one device is precisely the state rank ownership exists to
  prevent, and it can also defeat the reset that happens when the last reference
  goes. The reviewer demonstrated it by pausing the old release through an
  interposed symbol.

  The bookkeeping moved to a new `moxie_cuda::claims` module, compiled in **both**
  lanes and touching no driver symbol, whose `release_with(uuid, rank, teardown)`
  runs the teardown **while the claim is still held** and only then removes it.
  The mutex is deliberately not held across the teardown -- the claim's presence
  is what excludes another rank, not the lock -- so a teardown that consults the
  registry cannot deadlock.

  Teardown now **fails closed**: a release that errors leaves the device marked
  `TeardownFailed`, and a later `acquire` is refused with the reason and the
  previous holder's rank rather than being handed a card whose context is in an
  unknown state. Withholding a device is recoverable by restarting the process;
  handing out a half-released one is not.

  Regressions: `the_device_stays_held_until_teardown_has_finished` and
  `a_device_whose_teardown_failed_is_not_handed_out`, both **host-lane** tests
  that need no GPU, because the ordering window cannot be observed from outside
  without pausing a CUDA call. Both failed against the old order before the fix.
  Plus a device case, `concurrent_handoff_is_exclusive`, which proves the same
  property across real threads on real hardware where the timing is not ours to
  choose. The reviewer's own `drop_race` binary now reports `false` and exits 0.

- **P2 — the buffer-lifetime `compile_fail` doctest passed for the wrong reason.**
  Renaming `DeviceContext` to `RankContext` left the example calling
  `RankContext::new(0)`, which does not exist, so it failed with `E0599` instead
  of the lifetime error it exists to pin. My mistake and my process failure: I
  ran the prescribed bite check on the doctest I *added* and not on the one the
  rename touched. The example now calls `RankContext::acquire(RankId(0), 0)`,
  and every `compile_fail` doctest in the crate was rebuilt without
  `compile_fail`: `E0597 ctx does not live long enough` for the buffer lifetime,
  `E0277 *mut c_void cannot be sent between threads safely` for the thread
  binding. The rule this earns: **after a rename, re-run the bite check on every
  compile_fail doctest, not only on new ones.**

Two documentation corrections the reviewer also asked for:

- Document 02 **permits** `moxie-memory` -> `moxie-cuda` and forbids only the
  reverse. Four places said "neither may import the other", which is wrong. The
  real constraint is one-directional -- `moxie-cuda` may not name a
  `CapacitySnapshot` -- and that is why the descriptor sits in `moxie-types`.
- Host telemetry ownership is a **normal technical choice for the implementing
  agent**, resolvable from document 02's ownership table, not an owner gate. The
  handover said "answer it in the contract" but framed it as an open question;
  it now says plainly that the agent decides it and does not ask the owner.

**A third defect, found while re-running the gates and not reported by anyone:**
the `G-HOST-NODRIVER` `ldd` step had no explicit build. `cargo test --workspace`
builds test harnesses, not the plain `xtask` binary, so `ldd target/debug/xtask`
read whatever the last build left in the shared `target/`. Running the device
lane first this time made it report `libcuda` present in a host-lane run, from a
binary the host lane had not produced. Every earlier result happened to be taken
with the host build last, so the conclusion held and the method did not. The
procedure now builds `xtask` explicitly before `ldd`, in
[toolchain.md](../evidence/toolchain.md) and in the support matrix.

Re-verified after the corrections:

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `clippy -D warnings`, host and device features | PASS |
| `cargo test --workspace --locked --offline` | PASS, 468 unit/integration + 6 doctests |
| `cargo xtask arch-check` | PASS, 46 rejected + 12 accepted fixtures, 10 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane, with the corrected `ldd` procedure | PASS, 468 + 6, no `libcuda` |
| device lane | PASS, 476 + 8 |
| `cargo xtask-cuda test-gpu` | PASS, 24 cases, both architectures qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, as intended |
| `cargo xtask-cuda capacity` | PASS, 3 devices |
| the reviewer's `drop_race`, `lifetime_corrected`, `lifetime_stale`, `thread_bound` | all give the expected result |

### Review corrections, round 2 (2026-09-08, commit `a8969e6` not accepted)

One finding, and it is round 1's finding in the error path round 1 did not
route through the new mechanism.

- **P2 — the failed-attach cleanup bypassed the fail-closed teardown.** When
  `cuDevicePrimaryCtxRetain` succeeded and `cuCtxSetCurrent` then failed,
  `attach` released the reference, **ignored the result**, and `acquire`
  unconditionally called `claims::abandon`. If that cleanup release failed, the
  next rank was handed a device whose context reference was still outstanding.
  The comment claiming "nothing was retained" was true of two paths out of three
  and wrong about the one that mattered.

  A failed attach now has three distinguishable outcomes, carried by a private
  `AttachError`:

  | Outcome | Claim |
  |---|---|
  | failed before the retain | `ClaimUntouched` — the caller drops it, so a transient error does not strand the card |
  | retained, cleanup release succeeded | `ClaimResolved` — already removed by `release_with` |
  | retained, cleanup release failed | `ClaimResolved` — already marked `TeardownFailed` by `release_with` |

  The retained-context cleanup goes through **the same `claims::release_with`**
  that `Drop` uses, so there is one fail-closed path rather than two spellings of
  one. And `claims::abandon` is now hardened independently: it removes only a
  `Held` claim and never erases a `TeardownFailed` one, so a caller that is wrong
  about whether anything was retained cannot hand out a quarantined device.

  Regressions, both host-lane: `an_abandon_cannot_erase_a_failed_teardown`
  (failed before the fix) and
  `a_failed_attach_that_cleaned_up_leaves_the_device_available`, which mirror the
  two retained outcomes at the level where the policy lives. Bite check: making
  `abandon` unconditional again fails the first.

  **Stated limit:** the FFI branch itself — retain succeeds, `cuCtxSetCurrent`
  fails — is not reachable from an in-repo test without symbol interposition. The
  policy it delegates to is host-tested; the branch is exercised by the
  reviewer's `attach_cleanup` binary, which now reports `false`. That is a gap in
  this repository's own coverage and is recorded rather than papered over.

Re-verified after round 2:

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `clippy -D warnings`, host and device features | PASS |
| `cargo test --workspace --locked --offline` | PASS, 470 unit/integration + 6 doctests |
| `cargo xtask arch-check` | PASS, 46 rejected + 12 accepted fixtures, 10 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane, `xtask` rebuilt before `ldd` | PASS, 470 + 6, no `libcuda` |
| device lane | PASS, 478 + 8 |
| `cargo xtask-cuda test-gpu` | PASS, 24 cases, both architectures qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, as intended |
| `cargo xtask-cuda capacity` | PASS, 3 devices |
| the reviewer's `attach_cleanup` and `drop_race` | both report `false` and exit 0 |

The pattern across both rounds, worth carrying: **every exit from a function that
retained a resource is a teardown, including the error paths, and each one needs
the same ordering and the same fail-closed rule.** Round 1 fixed `Drop` and left
the error path; round 2 found it there.
