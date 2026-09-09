# Handover — M1.2 manifest reader and M1.3 ledger done pending review; rank-owned CUDA context next

Written 2026-09-08 at the end of the session that implemented task 0006.

## Workspace identity

- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, clean at the commit this record
  ships in. The task 0005 acceptance state is `712271c`, which is also `origin/main`.
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
  Never write there. Document 08 is the map into it.
- **Do not push without the owner saying so.** The owner stated this in the session that produced
  task 0004; nothing has withdrawn it. Commit locally, report, wait.
- Toolchain: rustc/cargo 1.97.1 pinned, CUDA 13.0 (nvcc V13.0.88). Ordinal 0 is the RTX 5060 Ti
  (sm_120, NUMA 0); ordinals 1-2 are the RTX 3090 pair (sm_86, NUMA 1). The UUIDs recorded by this
  session's `test-gpu` run are `GPU-3032cfa3-...` and `GPU-81fe4578-...` for the 3090 pair.
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced in `.cargo/config.toml`.

## Completed facts

**Task 0005 (M1.2, canonical manifest v1 and bounded tensor reads)** is implemented and committed at
`712271c`, after seven review rounds and one implementation takeover. Owner acceptance is still
pending for it; treat "gates pass" as gates passing, not as acceptance.

**Task 0006 (M1.3 part 1, the resource ledger and admission)** was contracted at `305c765` — a commit
with no `.rs` change, as tasks 0003, 0004 and 0005 did — implemented at `a486930`, and then
corrected against five rounds of review findings — four, then two, then one area in two halves, then one, then one. Its record,
[docs/tasks/0006-m1-resource-ledger-and-admission.md](../tasks/0006-m1-resource-ledger-and-admission.md),
carries the contract, the result, six recorded deviations, the bite checks and all five
review-correction rounds. It was **accepted at `b6ad977`** for the accounting and admission slice
only, explicitly not for M1.3 as a whole, and everything through it is on `origin/main`.

The reviewer's acceptance added evidence worth carrying: an independent exhaustive oracle over 6,048
small resource configurations found a feasible split behind every one of the 1,731 host-fallback
suggestions the ledger made. It also recorded a limit of the accepted behaviour -- because the
relocation greedy is conservative, **an absent suggestion does not prove that no host-backed plan
exists**, and no caller may read it that way.

One of those findings was a **contract** defect, not only an implementation one: the contract
excluded resident mapped pages from the host budget. It is struck in place rather than rewritten, so
the error stays legible. Resident pages are charged; a mapping's virtual extent is the separate,
uncharged quantity, declared per buffer.

What exists that did not before:

- `moxie-types::tier`: the closed tier vocabulary — document 03's fifteen device tiers and six host
  tiers, `Scope`, `ScopeKind`, an exhaustive `ALL`. Every tier's bytes are charged; the one tier
  that also carries an *uncharged* quantity is `host.mapped_resident`, whose virtual extent is
  reported beside its resident pages.
- `moxie-types::ids::DeviceUuid`: a GPU's identity as 16 bytes, parsed strictly from the canonical
  `GPU-` form. It keys maps; `DeviceId` still cannot, and that asymmetry is the point.
- `moxie-memory`: `CapacitySnapshot`, `PlanRequest` with liveness spans over ordered stages,
  derived reserves computed from the request's own buffers, and the `Ledger` — peak overlapping live
  set, atomic admission, typed refusal with a per-tier breakdown, explicit release. A refusal's
  alternatives are decided by **recomputing** the peaks with the relevant buffers at zero, not by
  attributing bytes to the reported peak.
- `arch-check`: two new rules for the memory boundary, the two crate-specific checks generalised into
  tables, and the 343-chain generated test now exercising four crate boundaries per chain.

Gates on this machine. The host lanes were re-run after the second review round; the device lanes
were measured at `a486930` and are **carried forward**, because the two correction rounds since
touched `moxie-types`'s UUID parser and `moxie-memory` only, and changed no device code, kernel or
FFI path:

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | PASS |
| `cargo test --workspace --locked --offline` | PASS, 460 unit/integration + 6 doctests |
| `cargo xtask arch-check` | PASS, 45 rejected + 12 accepted fixtures, 10 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane | PASS, 460 + 6; `ldd target/debug/xtask` shows no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` (carried forward from `a486930`) | PASS, 450 + 7 |
| `cargo xtask-cuda test-gpu` (carried forward from `a486930`) | PASS, 15 cases, `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` (carried forward from `a486930`) | **exit 1**, `UNQUALIFIED sm_120`, as intended |

Nothing failed. Nothing was skipped. No checkpoint was read, downloaded or converted, no byte was
allocated or copied, and no capacity was measured.

Support matrix: one new row for the ledger, updated host counts, an updated arch-check line, and the
M2 streaming row corrected — it used to say "no memory authority exists", and now says which half
exists and which does not.

## Decisions

- The tier vocabulary lives in `moxie-types`, not in `moxie-memory`. Document 02 requires shared
  descriptor types to be resolved into lower-level crates, and the typed error already had to name a
  tier. No ADR: this amends no reference document.
- `Error::CapacityExceeded` carries `Option<Tier>`. `None` means the driver reported an
  out-of-memory and nothing has attributed it yet — `moxie-cuda::status::classify` sees a CUDA code
  and has no tier. The ledger always names one.
- A scope carries two counters. Per-tier commitments answer to the per-tier caps; the scope
  commitment is the sum of the admitted plans' **scope peaks**, because inside one plan tiers that
  are live at different stages never coexist, while separate plans share no timeline. Summing the
  per-tier commitments for the scope figure was the first implementation and was wrong; the record
  keeps that negative result.
- Release is explicit and hands the reservation back when it refuses. Dropping a reservation leaves
  it charged and visible in `outstanding()`. Document 02 forbids `Drop` alone from freeing an
  in-flight resource, and R08 is the cost of the opposite habit.
- Owner gates O1–O7 are all still OPEN. None blocked this task and none was touched. No performance
  number has been measured anywhere in this repository; do not produce one from arithmetic.

## Remaining hypotheses and blockers

- **The ledger cannot measure anything.** It is given a `CapacitySnapshot`. Nothing in the workspace
  produces one from a real device or from host memory, so no admission decision here has been made
  against this machine's actual capacity. Saying otherwise would be the exact substitution the
  support matrix exists to prevent.
- **M1.3 is one part done of five.** The rank-owned CUDA context, event-backed leases, the basic
  allocator, the admitted execution plan and the device-resident layer chain are all outstanding.
  M1.4 is also still open beyond task 0004: appendable paged state, sampler integration, the
  generation service and the diagnostic CLI.
- The recurring failure shape in this workspace is still **an identity that exists to be unique,
  handed out by a derive**. It has happened twice (`SequenceState`, `KvCache`). `Reservation` is the
  third identity-bearing type and carries `compile_fail` doctests for both halves of it; check any
  new one against the same pattern before adding it.
- A second shape is now worth naming: **an error path that consumes the only handle to a live
  resource.** `Ledger::release` originally did, and it would have leaked exactly the way R08 did.
- A third, from the review: **a diagnostic that names a knob without checking that turning it would
  move the number that failed.** Five of the six findings across two rounds were that shape — an
  alternative offered for a buffer not live at the binding stage, then for a buffer live at a *tied*
  peak that survives its removal, then for one whose derived reserve a tie holds up anyway; a host
  fallback into memory the same request had taken, then one checked against an unrelated tier's cap;
  and a tier excluded from a budget it physically occupies. The lesson the second round added is
  sharper than the first: **attribution is not an answer, recomputation is.** Any advice the engine
  gives a caller must be tested by computing what happens if they follow it. Round 3 added the
  companion rule: **a label is not a resource.** `BindingConstraint::tier` on a scope failure names
  the largest contributor so a person knows where to look; deciding anything from it -- what can
  move, where it would go -- reads a diagnostic as a fact about the world. Round 4 showed the first
  rule had been applied unevenly: the host fallback was still answered from one stage's capacity
  after the scaling alternatives had moved to recomputation, and it gave a different answer when the
  two tied stages were swapped. When a rule like "recompute, do not attribute" is adopted, apply it
  to **every** place that gives advice in the same pass, not only the one the finding named. Round 5
  closed the series with the sharpest form of it: **two bounds are not a plan.** An upper bound on
  what the host could accept and an upper bound on what the device could shed can each be met by a
  different, incompatible move. Advice about a change has to name one concrete change and check that
  same one everywhere it lands.

## Next task

**M1.3 part 2: the rank-owned CUDA context, and the first measured capacity snapshot.**

- **One outcome**: one rank owns one device context, identified by UUID, and can produce a
  `CapacitySnapshot` for that device from a real driver query — so the ledger stops being fed only
  by tests. Nothing is allocated through it yet.
- **Owning components**: `moxie-cuda` for the context and the query; `moxie-memory` only if the
  snapshot's shape has to change, which would be a contract change, not a convenience.
- **Required reading**: document 03 (resource ledger and admission; transfer and CPU/GPU policy),
  document 02 (the buffer and asynchronous lifetime contract, and the `moxie-cuda`/`moxie-memory`
  rows), document 06 M1.3, document 07 on device lanes, R02, R07, R08, R12, AGENTS.md, and
  [task 0006](../tasks/0006-m1-resource-ledger-and-admission.md) for what the ledger already
  guarantees.
- **Write the contract first**, in a commit with no `.rs` change. Do not retune a threshold after a
  test has run.
- **Stop condition**: stop and report if it appears to need an allocator, an eviction policy, a
  lease over real bytes, a kernel launch that holds memory across a boundary, a checkpoint, or a
  change to the ledger's admission rules. Each is a different task, and a checkpoint needs O5.
- **Gate reminder**: a device behavioural change needs the device lane and `test-gpu` re-run, and
  "build passed" is not evidence for one (document 09 §E).

### Working rules this project holds agents to

1. **Write the contract first, in a commit with no `.rs` change**, and do not retune a threshold
   after a test runs. Tasks 0003–0006 all did this.
2. **Reproduce every reported defect before fixing it**, and keep the reproduction as a named
   regression test.
3. **Verify a test bites.** A `compile_fail` doctest is checked with `compile_fail` removed; a new
   rule is checked against a deliberate mutation of the code it constrains.
4. **Report passed, failed and skipped separately, with real counts.** An unmeasured architecture is
   never a pass.
5. Preserve negative results and disproven designs rather than quietly widening a bound.
