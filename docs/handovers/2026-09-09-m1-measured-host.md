# Handover — M1.3's accounting half is complete; what spends it is next

Written 2026-09-09 at the end of the session that implemented task 0008.

## Workspace identity

- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, clean at the commit this record
  ships in. Everything through task 0007's acceptance is on `origin/main` at `e8c5e4a`.
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
  Never write there. Document 08 is the map into it.
- **Do not push without the owner saying so.** They have authorised two pushes, each explicitly and
  for one state. Commit locally, report, wait.
- Toolchain: rustc/cargo 1.97.1 pinned, CUDA 13.0 (nvcc V13.0.88).
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced in `.cargo/config.toml`.
- The machine, measured rather than assumed: 264,005,080 kB total (251.8 GiB), `MemAvailable`
  ~244.9 GiB, 2 GiB of swap that enters no budget, cgroup v2 mounted with `memory.max` = `max` at
  every level of this session's chain. The three GPUs and their UUIDs are in
  [the previous handover](2026-09-08-m1-measured-capacity.md).

## Completed facts

**Tasks 0006 and 0007 are accepted** (`b6ad977`, `68d573a`), for the accounting slice and the
device-sensor slice respectively — neither for M1.3 as a whole.

**Task 0008 (M1.3 part 3, measured host capacity)** was contracted at `00fe7df` alongside
[ADR 0006](../decisions/adr/0006-host-telemetry-owner.md) — a commit with no `.rs` change — and
implemented on top of it. Owner review is pending. Its record is
[docs/tasks/0008-m1-measured-host-capacity.md](../tasks/0008-m1-measured-host-capacity.md).

What exists that did not before:

- `moxie-host`: the one component that may read machine telemetry. `MemAvailable` rather than
  `MemFree`; the **smallest** cgroup v2 `memory.max` over the process's chain, not the leaf's; swap
  read, reported and excluded from every budget; an injectable root so every rule is tested against
  committed fixture trees rather than against whatever this machine happens to be.
- `MeasuredHost` / `HostLimit` in `moxie-types`, and `CapacitySnapshot::measured_host` in
  `moxie-memory`.
- Two `arch-check` rules pinning ADR 0006: nothing outside `moxie-host` may name a `/proc` or `/sys`
  path, and `moxie-memory` may not depend on `moxie-host`.
- `cargo xtask-cuda capacity` measures the host and all three GPUs into one ledger and admits a plan
  that spans both tiers.

Gates, all run at the implementation state on this machine:

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `clippy -D warnings`, host and device features | PASS |
| `cargo test --workspace --locked --offline` | PASS, 485 unit/integration + 6 doctests |
| `cargo xtask arch-check` | PASS, 49 rejected + 13 accepted fixtures, 12 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane, `xtask` rebuilt before `ldd` | PASS, 485 + 6, no `libcuda` |
| device lane | PASS, 493 + 8 |
| `cargo xtask-cuda test-gpu` | PASS, 24 cases, both architectures qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, as intended |
| `cargo xtask-cuda capacity` | PASS, the host and 3 devices |

Nothing failed. Nothing was skipped. Nothing was allocated, mapped or reclaimed.

## Decisions

- [**ADR 0006**](../decisions/adr/0006-host-telemetry-owner.md): host telemetry gets its own crate.
  Not `moxie-memory` (the ledger never probes — task 0006 documents `preview` as pure with respect
  to live resources), not `moxie-storage` (owns artifacts, not the machine), not the composition
  root (the parser is shared semantics and would be copied at M8). Two machine checks enforce it.
- `MemAvailable`, never `MemFree`. A missing `MemAvailable` is an **error**, not a computed
  fallback: the substitute formula is a reclaim implementation detail that has changed, and guessing
  it would be a private model of the kernel presented as a measurement.
- The cgroup limit is the minimum over the whole chain. A limit on `user.slice` binds as hard as one
  on the leaf.
- Swap is read and reported and enters no arithmetic anywhere.

## Remaining hypotheses and blockers

- **Nothing spends what is now measured.** Both tiers have real budgets and the engine still owns no
  memory. That gap is the rest of M1.3, and it is where the accounting stops being theory.
- **A reading goes stale.** Another process can take host or device memory a moment after a
  snapshot. Document 03's answer is a pressure-driven replan; that is M2's and nothing detects it
  today.
- **On this machine no cgroup limit applies**, so the limited paths are proven by fixtures alone.
  That is the right way round — the fixtures test what the machine cannot — but it means a
  containerised deployment exercises code no hardware here has run.
- The CUDA attach-failure branch still has no in-repository regression (task 0007's accepted
  limitation).
- Owner gates O1–O7 remain OPEN. None blocked this task.

## Next task

**M1.3 part 4: event-backed leases and a basic device allocator that charges the ledger.**

- **One outcome**: a device allocation is admitted by the ledger, held under a lease that retires on
  a CUDA event rather than on `Drop`, and released back to the ledger — with the ledger's committed
  figure and the driver's own free-memory reading agreeing at every step.
- **Required reading**: document 02's buffer and asynchronous lifetime contract in full, document 03
  on leases and the residency lifecycle, R07 (deferred uploads need source ownership), R08 (the
  turn-boundary lease leak), and tasks 0006–0008.
- **The shape to expect**: `DeviceBuffer` today discharges its obligation bluntly, by synchronising
  the context before freeing, and its own doc comment says the real engine replaces that with an
  event-retained lease and that it "must not be optimised by simply deleting the synchronise". That
  replacement is this task.
- **Watch for**: R08 is a lease released on "next token" that leaked when the turn ended. Task 0006's
  `Reservation` already refuses to be freed by `Drop` for the same reason; the device lease must
  match it, and the two must be one mechanism rather than two that agree by convention.
- **Stop condition**: stop and report if it appears to need an eviction policy, a residency state
  machine, a host cache, a kernel that outlives its launch, a checkpoint, or a change to the
  ledger's admission rules.

### Working rules this project holds agents to

1. **Write the contract first, in a commit with no `.rs` change.** Tasks 0003–0008 all did.
2. **Reproduce every reported defect before fixing it**, and keep the reproduction as a named test.
3. **Verify a test bites** — and when a mutation fails *nothing*, that is a finding about coverage,
   not a formality. Task 0008's cgroup rule had two implementations that agreed on every fixture
   until a new one told them apart.
4. **Report passed, failed and skipped separately, with real counts.** An unmeasured lane is never a
   pass, and a green result whose method was never examined is not evidence.
5. **After a rename, re-run the bite check on every `compile_fail` doctest.**
6. **Every exit from a function that retained a resource is a teardown, error paths included.**
7. **Advice must be recomputed, not attributed, and two bounds are not a plan.**
8. Preserve negative results and disproven designs rather than quietly widening a bound.
