# Task 0006 — M1.3: resource ledger and admission

Status: **contract proposed**, 2026-09-08, after [task 0005](0005-m1-canonical-manifest-and-bounded-reads.md)
delivered the bounded reader.

**This contract is committed before any implementation code**, as in tasks 0003, 0004 and 0005. That
commit contains no `.rs` change. Nothing below is to be adjusted once a test has run.

## Identity and authority

- Task ID / milestone / owner: 0006 / M1.3 / implementation agent (Claude), owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `712271c`, tree clean
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Required documents: 03 (resource ledger and admission; the residency lifecycle this ledger will
  later serve), 02 (the `moxie-memory` ownership row, the planning contract, the buffer and
  asynchronous lifetime contract), 06 (M1.3), 07 (what may be claimed), 08 (R02, R03, R07, R08, R10,
  R11, R14, R15), AGENTS.md.
- Owner gates: **none needed.** Nothing is read, downloaded, converted or written outside the
  repository, so O5 is untouched. No quality or catalog claim is made, so O1 and O2 are untouched.
  **O6 is deliberately avoided:** this task produces no performance number and no measurement of any
  kind. All arithmetic here is exact integer accounting over byte counts the caller declares.
  **Stop and ask** if the work appears to need any gate.

## Why this next

Every remaining M1.3 part — the rank-owned CUDA context, event-backed leases, the basic allocator,
the admitted execution plan, the device-resident layer chain — charges bytes against something. The
ledger is the thing they charge. Writing it first means the allocator is built against an authority
that already exists, which is the R14 lesson stated as an ordering: a shared allocator adapters may
omit is insufficient, so the authority must precede its consumers rather than be retrofitted.

It is also the M1.3 part that needs no device. R02 is the failure being repaired: legacy's
`ResidencyManager` modelled accesses and bytes for a simulator and was never the live owner. This
task builds the accounting core and nothing else, so the next task can connect it to real memory
instead of writing a second one.

## Bounded deliverable

- **One concrete outcome:** a declared plan's complete resource envelope is admitted atomically
  against a caller-supplied per-scope, per-tier capacity snapshot, using the **peak overlapping live
  set** rather than a sum, or refused with a full per-tier breakdown and only the alternatives the
  request actually makes legal. Nothing in the workspace can charge a byte anywhere else.
- **Sole owning component:** a new crate `moxie-memory`, depending only on `moxie-types`.
  `moxie-types` gains the closed `Tier` descriptor, because document 02 requires shared descriptor
  types to be resolved into lower-level crates and the typed error already names a tier.
- **Allowed production files:** `crates/moxie-memory/**` (new); `crates/moxie-types/src/tier.rs`,
  `src/lib.rs`, `src/error.rs`; `crates/moxie-cuda/src/status.rs` and `src/driver.rs` as consumers of
  the changed error field; `xtask/src/archcheck.rs` and its fixtures; `Cargo.toml`, `Cargo.lock`.
- **Explicit non-goals and forbidden shortcuts:** no byte is allocated, mapped or freed; no CUDA
  call; no filesystem access; no probe of real device or host capacity; no eviction policy or victim
  choice; no residency state machine; no lease over an actual buffer and no event; no plan
  compilation or kernel selection; no model metadata or model name; no `Drop`-based release; and no
  silent shrink, clamp or retry of any request.
- **Existing consumers and second-consumer proof:** two synthetic workloads with different shapes and
  different tier profiles — a dense single-device layer chain, and an MoE-shaped workload spanning
  two devices of unequal capacity plus host spill and an incoming-expert reserve. Both must admit,
  and both must reject at their own binding tier.
- **Deletion plan, part of the deliverable:** `Error::CapacityExceeded` currently carries
  `tier: &'static str`, and `moxie-cuda` writes the literal `"device"` into it. Both go: the field
  becomes the closed `Tier`. Two spellings of a tier in the workspace means the task is not done.

## Contract, fixed before implementation

### Where the boundary sits

`moxie-types` owns the closed `Tier` and `Scope` descriptors and the typed error. `moxie-memory`
owns the ledger: the capacity snapshot, the commitments, the peak computation, admission, the
reservation's lifetime, and the rejection with its alternatives. No other crate computes a
reservation, and no consumer keeps a private counter.

The ledger never probes. It is given a capacity snapshot and told what a plan wants; measuring the
real device or host figure belongs to the next task, which owns the CUDA context.

### Scope and device identity

`Scope::Host` and `Scope::Device(DeviceUuid)`. `DeviceUuid` is a 16-byte value parsed from and
rendered back to the canonical `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` form. Parsing is over
bytes: a malformed identifier is a typed error, never a panic, whatever it contains. AGENTS.md: ordinals
are diagnostics only and evidence identifies a GPU by UUID. An ordinal may be attached as a
**label** for reporting; it is never identity and there is no lookup by it.

### Tiers, exactly document 03's lists

```text
DeviceTier: PackedResidentWeights, ExpertCache, KvStatePages, RecurrentState, Activations,
            Logits, KernelWorkspace, CollectiveBuffers, GraphPools, TransferStaging,
            SpeculativeTargetState, SpeculativeDraftState, EntropyBranches,
            AllocatorFragmentation, SafetyHeadroom
HostTier:   Pageable, Pinned, MappedResident, CpuWorkspace, StateSpill, ConversionReadBuffers
```

`Tier` is `Device(DeviceTier) | Host(HostTier)`, with a stable `name()` and an exhaustive `ALL`. A
device tier requested in the host scope, or a host tier in a device scope, is `InvalidRequest`, not a
coerced default.

~~`HostTier::MappedResident` is reported in its own row and **does not consume the committed host
budget**: document 03 says mapped virtual bytes do not equal committed host RAM, and that neither is
free. It is charged against its own declared per-tier cap and shown separately, so a large mapping
is visible without being counted twice.~~

**Corrected by review, 2026-09-08** (finding 1; struck text above is what the contract said, kept so
the error is legible). Resident pages of a mapping are physical RAM and are charged against the host
budget like every other byte. What is not charged is the mapping's **virtual extent**, which a
buffer declares separately through `BufferRequest::virtual_extent` and which is reported beside the
resident figure. Only a `MappedResident` buffer may declare one, and it may not be smaller than the
resident set. The struck rule did not prevent double counting; it permitted under-counting, and let
1,600 B of resident pages into a 900 B host budget. Document 03 requires resident mapping pressure
to be reconciled with the host budget, not excused from it.

### Capacity snapshot

```text
CapacitySnapshot { scope, physical_bytes, system_headroom_bytes, per_tier_cap: map<Tier, u64> }
admissible_bytes = physical_bytes - system_headroom_bytes    (checked; underflow is an error)
```

A **host** snapshot with `system_headroom_bytes == 0` is refused. Document 03: host admission
reserves operating-system and application headroom from measured available memory and must not treat
all 251 GB as an expert cache. Refusing the zero is how that is enforced rather than assumed.
A per-tier cap is optional; absent means "bounded only by the scope".

### The request

```text
PlanRequest {
  stages:           [StageLabel]      ordered, at least one
  buffers:          [BufferRequest]
  derived_reserves: [DerivedReserve]
}
BufferRequest  { label, scope, tier, bytes, first_stage..=last_stage, scales_with: Option<Scaling> }
Scaling        = Context | Branches
DerivedReserve { scope, tier, rule, first_stage..=last_stage }
ReserveRule    = LargestBufferOfTier | NLargestBuffersOfTier(n)
```

`ReserveRule` exists because of R03: Inkling's worst-case incoming-expert reserve and GLM-5.3's
two-largest-linears reserve must become **derived** reservations computed from the request's own
buffers, not constants copied between models. There is therefore no API to supply a literal reserve
byte count. `NLargestBuffersOfTier(n)` with `n` greater than the number of buffers in that tier is an
error, not a silent clamp; `n == 0` is an error.

### Peak overlapping live set

For each `(scope, tier)`, and for each stage `s`:

```text
live(scope, tier, s) = sum of bytes of buffers of that scope and tier live at s
                     + bytes of derived reserves of that scope and tier live at s
tier_peak(scope, tier) = max over s of live(scope, tier, s)
scope_peak(scope)      = max over s of ( sum over tiers of live(scope, tier, s) )
```

Every tier is in that sum, including `MappedResident`, per the correction above.

The scope figure is the stage-wise total, **not** the sum of the per-tier peaks, which would
over-reserve mutually exclusive buffers — document 03 requires the peak overlapping live set. The
two peaks may occur at different stages, so the report names the stage of each. Transition
coexistence is expressed by intervals that overlap at the barrier stage, which is what document 03
means by admission reflecting temporary coexistence.

Every sum is checked. Overflow is a typed error, never a wrap.

### Admission is atomic

`admit(request) -> Result<Reservation, Rejection>` reserves the complete envelope or reserves
nothing. Document 02: `admit(candidate)` atomically reserves the candidate's complete resource
envelope. On rejection every counter in the ledger, in every scope and tier, is byte-for-byte what it
was before — including the tiers that would have fit.

### Reservation lifetime

`Reservation` is neither `Clone` nor `Copy`, and carries a process-unique `ReservationId` together
with the `LedgerId` it came from. `release(reservation)` consumes it by value, so a double release
cannot be written. Releasing against a different ledger is `InvalidRequest` and changes no counter in
either ledger.

Release is **explicit**. Document 02 forbids `Drop` alone from freeing in-flight resources, and R08
is the concrete failure: a lease released on "next token" leaks when the turn ends and no next token
arrives. `outstanding()` therefore lists open reservations by label, so a leak is observable rather
than inferred, and a regression test reproduces the turn that ends with no next token.

### Rejection content

```text
Rejection {
  report:          AdmissionReport   every scope and tier: capacity, already committed,
                                     this request's peak, remaining headroom, and the peak's stage
  binding:         [(Scope, Tier)]   every constraint that failed, not only the first
  shortfall_bytes: u64               for the worst binding constraint
  alternatives:    [LegalAlternative]
}
LegalAlternative = LowerContext | FewerBranches | OtherWeightPrecisionOrArtifact
                 | DifferentTopology | HostBackedExecution
```

An alternative is listed only when the request itself makes it legal:

| Alternative | Listed only when |
|---|---|
| `LowerContext` | some buffer **contributing to a failed constraint** declared `scales_with = Context` |
| `FewerBranches` | some buffer contributing to a failed constraint declared `scales_with = Branches` |
| `OtherWeightPrecisionOrArtifact` | the binding tier is `PackedResidentWeights` or `ExpertCache` |
| `DifferentTopology` | more than one device scope is declared |
| `HostBackedExecution` | the binding scope is a device, no host constraint binds, and host headroom **after this request's own host demand** covers the shortfall |

**Corrected by review, 2026-09-08** (findings 2 and 3). A buffer *contributes to* a failed
constraint when it is live at the stage where that constraint's peak occurred, in the same scope,
and -- for a tier cap -- in the same tier; a derived reserve live at that stage contributes through
the buffers whose sizes it was computed from. The first two rows previously accepted any scaling
buffer anywhere in a failing scope, which offered `LowerContext` for a KV buffer that was not live
at the binding stage and so could not move the peak however much of it was removed. The
`HostBackedExecution` row previously subtracted only earlier commitments from host headroom, so it
offered a fallback into memory the same request had already spoken for; it now also subtracts this
request's host peak, refuses when the host itself binds, and respects a declared cap on
`StateSpill` or `CpuWorkspace` when one exists.

A generic menu of five is worse than nothing, because it invites the caller to try a change that
cannot help. The ledger **never applies** an alternative: document 03 forbids automatically
shortening context, quantizing the cache or silently lowering weights, so a rejection returns no
reservation and mutates nothing.

`Rejection` converts into `Error::CapacityExceeded` naming the worst binding tier, for callers that
only need the variant.

### Error metrics

None. This task performs no arithmetic on model values. Every quantity is an exact integer byte
count supplied by the caller, and every sum is checked for overflow. The numerical contracts of tasks
0003 and 0004 are unchanged.

## Acceptance

- `cargo xtask arch-check` (with `moxie-memory` declared, and two new rules — memory touches the
  filesystem, memory branches on a model name — each exercised by a rejecting fixture and by a clean
  positive fixture), `spec-check`, `fmt`, `clippy -D warnings`, and the full host lane pass. The
  no-driver lane passes.
- The device lane and `cargo xtask-cuda test-gpu` are **re-run**, not carried forward: this task
  changes `moxie-cuda`'s status mapping when the tier field becomes typed.
- Peak, not sum: two buffers in different tiers live in disjoint stages admit inside a budget their
  sum exceeds; the same two overlapping do not. A barrier stage where prefill and decode buffers are
  both live is the peak.
- A per-tier cap binds independently of the scope total, and the report names the stage at which each
  peak occurs.
- Atomicity: a request whose early tiers fit and whose last tier does not leaves the whole ledger
  identical, compared counter by counter.
- One test per rejection rule and one per alternative-legality row above, including the negative: an
  over-budget request with no branch bytes does not offer `FewerBranches`.
- Derived reserves are computed from the request: changing the largest buffer changes the reserve;
  `NLargestBuffersOfTier` with `n` above the buffer count, or `n == 0`, is an error.
- A host snapshot with zero system headroom is refused. A large `MappedResident` figure is reported
  and does not consume the committed host budget.
- Device identity is the UUID: two devices whose ordinal labels are swapped keep their own
  commitments.
- Release is explicit and single: `compile_fail` doctests for `Reservation: Clone` and for use after
  release, each verified to fail for the intended reason with `compile_fail` removed; cross-ledger
  release refused with no counter changed; the R08 no-next-token regression test.
- Overflow is an error, not a wrap, on every sum that a caller can drive.
- Both synthetic consumers admit, and both reject at their own binding tier.
- `Tier` exists exactly once in the workspace, as a closed enum with an exhaustive `ALL` and distinct
  names; no `&'static str` tier remains anywhere.
- Support matrix: add a row for the ledger. It must say host-only, that it allocates nothing, that it
  measures nothing, and that it loads no model.
- Stop condition: if this appears to need a real device or host capacity query, a CUDA call, an
  allocator, an eviction decision, a lease over an actual buffer, an event, or a model's config
  schema, stop and report. Each is a different task.

## Result, filled after work

Implemented 2026-09-08 on branch `main`, on top of the contract commit `305c765`.
The contract above is unchanged; only this section is filled in.

What was built:

- `moxie-types` gains `tier` (the closed `DeviceTier`/`HostTier`/`Tier`/`Scope`/
  `ScopeKind` descriptors, exactly document 03's two lists, with an exhaustive
  `ALL`, stable names, the scope-kind rule and `charges_scope_budget`) and
  `DeviceUuid` in `ids` (16 bytes, strict canonical `GPU-...` parse and render).
- `moxie-memory` (new, depends only on `moxie-types`): `CapacitySnapshot`
  (caller-measured capacity, refused headroom rules, per-tier caps),
  `PlanRequest` / `BufferRequest` / `DerivedReserve` / `StageSpan` / `Scaling`
  (declaration and its validation), the `Ledger` (peak-overlapping-live-set
  evaluation, atomic `admit`, `preview`, explicit `release`, `outstanding`), and
  the report types (`AdmissionReport`, `BindingConstraint`, `LegalAlternative`,
  `Rejection`, with a readable `Display` for each).
- `arch-check`: `moxie-memory` declared (workspace `moxie-types`, no third
  party); the two crate-specific checks generalised into `IO_FREE_CRATES` and
  `MODEL_FREE_CRATES` tables so a second boundary is a row rather than a second
  copy; two new rules (`memory touches the filesystem`, `memory branches on a
  model name`), each with a rejecting fixture and a clean accepted one. The
  existing 343-chain generated test now checks all four crate boundaries per
  chain instead of two.
- Deletion done: `Error::CapacityExceeded` no longer carries a `&'static str`
  tier, and `moxie-cuda`'s `"device"` literal is gone. There is one spelling of
  a tier in the workspace.
- `Cargo.lock` gains exactly one entry, the new workspace crate, and zero
  third-party packages: `moxie-memory` depends on `moxie-types` alone.

Deviations recorded, none silent:

- The contract said the error's tier field "becomes the closed `Tier`". It became
  `Option<Tier>`. `moxie-cuda::status::classify` sees a CUDA result code and
  nothing else, so it has no tier to name; `None` states that the attribution is
  missing, where picking any variant would be a fabrication the report would
  then repeat. The ledger always names one. The deletion the contract asked for
  is complete either way.
- `Rejection.binding` is a `Vec<BindingConstraint>` rather than a
  `Vec<(Scope, Tier)>`. Each constraint still carries its scope and tier, and
  adds which ceiling it hit, the needed and available bytes, and the peak stage.
  The pair alone cannot distinguish a tier cap from the scope budget, and the
  contract's requirement -- every failing constraint, not only the first -- is
  what the type had to serve.
- No ordinal label is attached to a device scope. The contract permitted one
  ("may"); no API in this crate takes an ordinal, so a field that exists only to
  be ignored would be worse than its absence. Identity is the UUID and nothing
  else.
- A scope keeps **two** counters, not one: per-tier commitments, which the tier
  caps constrain, and a scope commitment, which is the sum of the admitted
  plans' *scope peaks*. The first version summed the per-tier commitments for
  the scope figure, which contradicted the peak-not-sum rule admission checks
  with. The two differ because tiers live at different stages inside one plan,
  while separate plans share no timeline and so their peaks add.
- `Ledger::release` returns the reservation inside `ReleaseRefused` when it
  refuses, rather than consuming it. Consuming a reservation on the error path
  destroys the only handle to bytes that are still charged, which is an
  R08-shaped leak manufactured by the error path.
- `ReserveRule::LargestBufferOfTier` over a tier with no buffers is an error, on
  the same reasoning the contract gave for `NLargestBuffersOfTier`: a reserve
  derived from nothing is a zero pretending to be a reservation.

Gates (this machine, 2026-09-08):

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | PASS |
| `cargo test --workspace --locked --offline` | PASS, 442 unit/integration + 6 doctests (was 402 + 4) |
| `cargo xtask arch-check` | PASS, 45 rejected + 12 accepted fixtures, 10 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane | PASS, 442 + 6; `ldd target/debug/xtask` shows no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | PASS, 450 + 7 |
| `cargo xtask-cuda test-gpu` | PASS, 15 cases, `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, `UNQUALIFIED sm_120`, as intended |

Nothing failed. Nothing was skipped. The device gates were re-run rather than
carried forward, because this task changed the tier field of the error
`moxie-cuda`'s status mapping constructs.

`moxie-memory` itself: 25 acceptance tests, 8 unit tests, 2 compile-fail
doctests.

Bite checks, each reverted to green:

- Both `compile_fail` doctests were rebuilt with `compile_fail` removed and fail
  for the intended reasons: `E0599 no method named clone`, and `E0382 use of
  moved value` on the second release.
- Replacing the stage maximum with a sum over stages fails four tests, including
  both synthetic consumers and the barrier-coexistence test.
- Offering `LowerContext` unconditionally fails
  `lower_context_is_offered_only_when_a_binding_buffer_says_it_scales_with_context`,
  which is the test that stops a generic five-item menu.

Negative result worth keeping: the first implementation derived a scope's
committed total by summing its per-tier commitments. That is not the peak
overlapping live set, and
`the_peak_is_the_stage_maximum_and_not_the_sum` caught it before any of the
consumer tests were written. The fix is the two-counter design above; the
tempting alternative -- charging the sum and calling it conservative -- would
have made the ledger refuse plans that fit, which is the same class of error as
admitting plans that do not.

No checkpoint was read, downloaded or converted; no byte was allocated, mapped
or copied; no capacity was measured. The stop condition was not triggered: this
needed no device query, no CUDA call, no allocator, no eviction decision, no
lease over a real buffer and no model config.

Remaining blockers and next bounded task: M1.3 continues with the rank-owned
CUDA context, which is the first thing that can *measure* a capacity snapshot
and the first consumer that charges real bytes to this ledger. Event-backed
leases, the basic allocator and the admitted execution plan follow it. The
ledger's measuring half does not exist and must not be claimed.

### Review corrections, 2026-09-08 (commit `a486930` not accepted)

The reviewer returned four findings against the accepted-scope implementation,
each with a reproduction. All four reproduced inside this repository before any
fix, and each reproduction is kept as a named regression test. One finding is
also a **contract** defect and the contract above is corrected in place, struck
rather than rewritten, at the reviewer's direction.

- **P1 — resident mapped pages escaped the physical host budget.** The contract
  excluded `HostTier::MappedResident` from the scope budget on the grounds that
  mapped virtual bytes are not committed RAM. That conflated a mapping's virtual
  extent with its resident pages: the resident pages are physical RAM. With a
  900 B host budget the ledger admitted 800 B pinned **plus** 800 B resident.
  Every tier is now charged, `Tier::charges_scope_budget` is gone, and the
  uncharged quantity is the mapping's virtual extent, declared per buffer with
  `BufferRequest::virtual_extent`, reported in `TierReport::virtual_extent_bytes`
  and binding nothing. Only a `MappedResident` buffer may declare one, and it may
  not be smaller than the resident set. Regressions:
  `resident_mapped_pages_share_the_physical_host_budget`,
  `a_mappings_virtual_extent_is_reported_and_charged_to_nothing`,
  `a_virtual_extent_is_refused_where_it_would_be_meaningless_or_impossible`, and
  `mapped_resident_is_the_only_tier_with_a_virtual_extent` in `moxie-types`.
  The test that enforced the old rule is deleted, not weakened.

- **P2 — host-backed execution was offered into memory the same request had
  already taken.** The headroom calculation subtracted earlier commitments but
  not this request's own host demand, so a plan needing 900 B of CPU workspace
  from a 900 B host budget was still told it could fall back to the host. The
  calculation now also subtracts the request's host scope peak, declines
  entirely when any host constraint is among the binding ones, and honours a
  declared cap on `StateSpill` or `CpuWorkspace` when one exists. Regression:
  `host_backed_execution_accounts_for_this_requests_own_host_working_set`.

- **P2 — context and branch alternatives ignored when the buffer was live.** Any
  scaling buffer in a failing scope qualified, including one live only in a
  stage that did not bind. A 200 B fixed prefill workspace over a 100 B budget
  was answered with "lower the requested context" for a 50 B decode-stage KV
  buffer, when removing all of it changes the 200 B peak by nothing. A buffer now
  contributes to a constraint only when it is live at that constraint's peak
  stage, in the same scope, and -- for a tier cap -- the same tier; a derived
  reserve live at that stage contributes through the buffers whose sizes it was
  computed from, so shrinking the largest expert is correctly credited with
  shrinking the incoming-expert reserve. Regressions:
  `a_context_alternative_must_be_able_to_move_the_binding_peak` (which also
  asserts the positive case, so it tests the liveness rule rather than the
  alternative being unreachable) and
  `a_branch_alternative_must_be_able_to_move_the_binding_peak`.

- **P2 — a malformed multi-byte UUID panicked.** `DeviceUuid::parse` checked a
  length in bytes and then sliced the `&str`, which cuts through a multi-byte
  character: `"GPU-0000000é-..."` panicked with "byte index 8 is not a char
  boundary" instead of returning the typed error every other malformed
  identifier returns. Parsing is now over `&[u8]` throughout, with an explicit
  lower-case-only hex digit helper. Regression:
  `a_malformed_multibyte_uuid_is_refused_rather_than_panicking`, covering a
  multi-byte character at the start, middle and end of the identifier and a
  two-character CJK group.

Bite checks for the corrections, each reverted to green: dropping the liveness
condition fails both alternative regressions; not subtracting the request's own
host peak fails the host regression; re-excluding `MappedResident` from the
scope total fails both mapping regressions. The reviewer's own four tests, run
unmodified from outside the repository, pass.

Re-verified after the corrections: `fmt` PASS; `clippy -D warnings` PASS;
`cargo test --workspace --locked --offline` PASS, 448 unit/integration + 6
doctests (was 442 + 6: five net new tests in `moxie-memory`, one in
`moxie-types`); `arch-check` PASS (45 rejected + 12 accepted, 10 rules);
`spec-check` PASS (10 documents); no-driver lane PASS (448 + 6, no `libcuda`). The device
lane and `test-gpu` were re-run for the previous round and are carried forward
here: these corrections touch `moxie-types`'s UUID parser and `moxie-memory`
only, and no device code, kernel or FFI path changed.
