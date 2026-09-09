# Handover — M1.3 has measured device capacity; host capacity is the open question

Written 2026-09-08 at the end of the session that implemented task 0007.

## Workspace identity

- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, clean at the commit this record
  ships in. Everything through task 0006's acceptance is on `origin/main` at `4bbe659`.
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
  Never write there. Document 08 is the map into it.
- **Do not push without the owner saying so.** They authorised the push of `4bbe659` explicitly and
  for that state. Commit locally, report, wait.
- Toolchain: rustc/cargo 1.97.1 pinned, CUDA 13.0 (nvcc V13.0.88).
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced in `.cargo/config.toml`.

The three GPUs, by the identity that matters:

| UUID | Card | SM | Ordinal today | Bus | Total |
|---|---|---|---|---|---|
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | RTX 5060 Ti | `sm_120` | 0 | `0000:03:00.0` | 15,886 MiB |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | RTX 3090 | `sm_86` | 1 | `0000:82:00.0` | 24,123 MiB |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | RTX 3090 | `sm_86` | 2 | `0000:83:00.0` | 24,123 MiB |

"Ordinal today" is exactly as unstable as it sounds. `cargo xtask-cuda capacity` run with
`CUDA_VISIBLE_DEVICES=2,1,0` gives every card a different ordinal and moves no capacity with it.

## Completed facts

**Task 0006 (M1.3 part 1, the resource ledger) was accepted at `b6ad977`** for the accounting and
admission slice only, after five review rounds. Its record is
[docs/tasks/0006-m1-resource-ledger-and-admission.md](../tasks/0006-m1-resource-ledger-and-admission.md).

**Task 0007 (M1.3 part 2, rank-owned context and measured device capacity)** was contracted at
`689f58f` — a commit with no `.rs` change — and implemented on top of it. Owner review is pending.
Its record is [docs/tasks/0007-m1-rank-context-and-measured-capacity.md](../tasks/0007-m1-rank-context-and-measured-capacity.md).

What exists that did not before:

- `RankContext`: one rank, one GPU, enforced by a process-wide claim keyed by `DeviceUuid`. A second
  rank is refused a held device, one rank cannot hold two, and the claim is released on drop.
- `MeasuredDevice` in `moxie-types` and `RankContext::measure()`: a device's own account of itself,
  read through the live context so the context's own cost is already gone from `free_bytes`.
- `CapacitySnapshot::measured`: the rule that turns a reading into a budget, pure and host-tested.
- `cargo xtask-cuda capacity`: the composition root, measuring three devices, admitting a plan
  against each, and failing if a byte is still held afterwards.
- `format_uuid` is gone. `DeviceUuid` is the only way a device identity is written.

Gates, all run at the implementation state on this machine:

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | PASS |
| the same **with device features** | PASS — new to the gate list, see below |
| `cargo test --workspace --locked --offline` | PASS, 464 unit/integration + 6 doctests |
| `cargo xtask arch-check` | PASS, 46 rejected + 12 accepted fixtures, 10 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane | PASS, 464 + 6; `ldd target/debug/xtask` shows no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | PASS, 472 + 8 |
| `cargo xtask-cuda test-gpu` | PASS, 21 cases, `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, `UNQUALIFIED sm_120`, as intended |
| `cargo xtask-cuda capacity` | PASS, 3 devices |
| `CUDA_VISIBLE_DEVICES=2,1,0 cargo xtask-cuda capacity` | PASS, identity travelled with the UUID |

Nothing failed. Nothing was skipped. Nothing was allocated through the ledger, and no performance
number was produced anywhere in this repository.

## Decisions

- `MeasuredDevice` lives in `moxie-types`. Document 02's graph puts `memory` above `cuda`, so the
  crate that takes a reading and the crate that turns it into a budget cannot import each other; the
  descriptor goes to the bottom and the composition root does the wiring. A new arch-check fixture
  refuses a `moxie-cuda` -> `moxie-memory` dependency, so the next agent finds this out from the
  checker rather than from a reviewer. No ADR: this amends no reference document.
- Rank ownership is a **registry**, not a convention. The driver will hand out a second context for
  one card, and that is the shared mutable state document 01 asks to isolate.
- A measurement is a reading. Every doc comment, the support-matrix row and the command's own output
  say so, because the difference between "this card has 23 GiB free" and "this engine has 23 GiB"
  is the whole of R02.
- The 64 MiB reserve in `xtask capacity` is a placeholder constant and is named one. It is not a
  derived reservation and no production crate contains it.

## Remaining hypotheses and blockers

- **Host capacity is unmeasured, and it is blocked on an ownership question, not on effort.**
  Document 03 requires host admission to reserve headroom from *measured* available memory. Reading
  `/proc/meminfo` is filesystem access: `moxie-memory` is forbidden it by an arch-check rule,
  `moxie-storage` owns the filesystem but has no business owning host telemetry, and `moxie-cuda` is
  the wrong layer entirely. Answer that before writing the code, in the next task's contract.
- **A reading can go stale between the snapshot and the admission.** Document 03's answer is a
  replan on a memory-pressure event, which is M2's. Today it is a documented limit and nothing
  detects it.
- The device lane had **never been run under `clippy -D warnings`** until this task, and it had two
  real findings. Assume the same of any lane that is not in the gate table: absence of a result is
  not a pass. `G-DEVICE-CLIPPY` is now a gate.
- Owner gates O1-O7 remain OPEN. None blocked this task.

## Next task

**M1.3 part 3: measured host capacity — but the contract answers the ownership question first.**

- **Before any code**: name the component that may read host memory telemetry, and say why it is not
  `moxie-memory` (forbidden the filesystem) or `moxie-storage` (owns artifacts, not the machine).
  A small `moxie-host` crate at the bottom of the graph and a new arch-check rule confining it is one
  answer; the composition root reading it and passing a `MeasuredHost` descriptor is another. Pick
  one in the contract, with the reason.
- **One outcome**: a host `CapacitySnapshot` built from measured available memory with a declared
  OS-and-application reserve, admitted alongside the three device snapshots in one ledger.
- **Required reading**: document 03 (host admission, the 251 GB warning, mapped resident pages),
  document 02's ownership table, R11, AGENTS.md, and tasks 0006 and 0007.
- **Watch for**: `MemAvailable` is not `MemFree`, and neither is what a process may spend. Page cache
  counts as available and is not free. Do not treat all 251 GB as an expert cache — document 03 says
  so in those words, and `CapacitySnapshot::new` already refuses a zero host reserve.
- **Stop condition**: stop and report if it appears to need an allocator, a lease, a mapping, a
  checkpoint, or a change to the ledger's admission rules.

### Working rules this project holds agents to

1. **Write the contract first, in a commit with no `.rs` change**, and do not retune a threshold
   after a test runs. Tasks 0003-0007 all did this.
2. **Reproduce every reported defect before fixing it**, and keep the reproduction as a named
   regression test.
3. **Verify a test bites.** Mutate the code it constrains and watch it fail; if it does not, say so
   rather than claiming a check it does not have.
4. **Report passed, failed and skipped separately, with real counts.** An unmeasured lane is never a
   pass.
5. Preserve negative results and disproven designs rather than quietly widening a bound.
6. **Advice must be recomputed, not attributed, and two bounds are not a plan.** Task 0006's review
   found six defects across five rounds, and five were that shape.
