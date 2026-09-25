# Task 0091 — a dense plan set shares one resident copy of the weights

Status: **active** (coordinator, 2026-09-25); design option (B) from the
ledger, assigned before the owner's morning review (redirectable). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0091, M6 slice 6, roadmap **M6.4** "eliminating duplicate resident
  representations". Ledger finding (2026-09-25): every `SelectedReservedPlan`
  admits and uploads its own weights, so a bucket set of four plans holds four
  weight copies. This is on slice 7's critical path (no real checkpoint fits
  four times).
- AGENTS.md: "One real memory authority admits all … resources"; M2's
  "exactly one production weight-residency owner" is
  `moxie_memory::residency` (task 0020). The shared copy therefore lives in
  that authority's device cache, and plans only borrow addresses of leases.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. It is asynchronous-ownership
  work; follow it exactly. On any conflict with the code, stop and send
  `DECISION`; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals. Arch-check forbids definitions whose names contain
  "weightcache", "residencymap", "chunktable" and similar: this task defines
  no cache.

## Facts established before writing (coordinator, 2026-09-25)

- Residency flow (`moxie-executor/tests/residency_device.rs` about 94–190):
  `ResidencyAuthority::open(&mut ledger, &ResidencyRequest::new(label,
  host_cap).device(uuid, cap))`; `DeviceResidency::create(&ctx, &mut
  authority)`; `authority.acquire(AcquireRequest { chunk, destination, now,
  deadline, class, turn })` → `Acquired::Pending { lease, work, .. }`;
  `drain_reads(&mut authority, &mut source, work)` → upload orders;
  `device.perform_upload(&mut authority, &stream, &upload)`;
  `authority.device_range(&lease)` → `(offset, len)` inside the device cache;
  `authority.release(lease)` returns a lease; `ChunkId::new(artifact,
  TensorSlot::tensor(role), LogicalRange::new(offset, len), version)` names
  a dense tensor.
- The device cache is charged to `DeviceTier::ExpertCache`, hard-coded in
  `moxie-memory/src/residency.rs` (about 1413 and 1503). Dense weights held
  there would be mislabelled; document 03 tracks packed resident weights
  separately.
- A plan's weight operands are addressed through `range_for_value` on the
  plan's own `Weights` region (`chain.rs`), bound once by `bound_weights`,
  charged as `weight_region_bytes` (`PackedResidentWeights`).
- Every dense kernel receives weights only as device addresses (`address(…)`
  in `dense.rs`); the affine arm adds section offsets to the weight's address.

## Bounded deliverable

- **Outcome:** a `DensePlanSet` holds the leases of one resident weight copy
  and every bucket plan admitted against it; plans are admitted without a
  weights region, step through the set only, and cannot outlive the leases; a
  four-bucket set charges the weights once.
- **Allowed files:** `crates/moxie-memory/src/residency.rs` (and its
  `lib.rs` export only if needed), `crates/moxie-executor/src/chain.rs`,
  `crates/moxie-executor/src/dense.rs`, new
  `crates/moxie-executor/src/dense_set.rs` plus its `lib.rs` module line,
  `crates/moxie-executor/tests/dense_gemma_device.rs` (one new test), this
  task's Result.
- **Non-goals:** migrating existing tests or the timing harnesses to the set;
  affine formatted weights in the set (refuse a candidate with a non-empty
  `weight_formats()`); TP/PP; host experts; eviction of dense weights.

## Numbered changes

1. **Residency tier (`moxie-memory`).** Record a tier per device cache:
   `ResidencyRequest::device` keeps charging `ExpertCache`; add `pub fn
   device_weights(self, uuid, cap_bytes) -> Self` that charges
   `DeviceTier::PackedResidentWeights`. Use the recorded tier at both charge
   sites (about 1413, 1503) and anywhere the report names the tier. Nothing
   else in the authority changes.
2. **Plan admission against resident weights (`chain.rs`).**
   `pub(crate) fn admit_with_resident_weights(candidate, graph, capability,
   catalogue, ledger, ctx, weights: BTreeMap<ValueId, u64>) -> Result<Self,
   SelectedAdmitRefused<'ctx>>`: identical to `admit` except that the request
   and arena **omit the weights region**, and `weights` (value → device
   address) is stored in a new field `resident_weights`. Refuse (typed
   `invalid("weights", …)`) unless its keys equal `graph.weights()` exactly.
   `admit` stores an empty map.
3. **Addressing (`chain.rs`, `dense.rs`).** Add one plan method
   `pub(crate) fn value_address(&self, value) -> Result<u64>` that returns the
   resident address for a resident weight and the range's device address
   otherwise; route `dense.rs`'s `address(…)` through it. If any code path
   needs a weight's `DeviceRange` itself (not just its address), stop and
   send `DECISION` naming it.
4. **Bindings (`chain.rs`).** For a plan with resident weights, a binding for
   a weight value is refused (`invalid("bindings", "this plan's weights are
   resident")`), and a missing weight binding is not an error.
5. **`DensePlanSet<'r, 'ctx>` (`dense_set.rs`).** Fields: `residency: &'r
   DeviceResidency<'ctx>`, `leases: BTreeMap<ValueId, ResidencyLease>`,
   `plans: BTreeMap<u64, SelectedReservedPlan<'ctx>>` keyed by rows,
   `lost: Option<OperationLease<SelectedCompletion<'ctx>, DenseOperation<'ctx>>>`,
   and the ledger id.
   - `pub fn admit(ledger, ctx, graph, capability, catalogue, residency,
     authority: &ResidencyAuthority, leases: BTreeMap<ValueId,
     ResidencyLease>, candidates: Vec<SelectedPlanCandidate>) -> Result<Self,
     DensePlanSetRefused<'r, 'ctx>>`. Validate before admitting anything: the
     lease keys equal `graph.weights()`; every lease's scope is the
     context's device; every `authority.device_range(lease)` length equals
     every candidate's planned weight `logical_bytes`; candidates have
     distinct `workload().rows` and none has `weight_formats()`. Address =
     `residency` base device address + offset (add a `pub(crate)` base
     accessor to `DeviceResidency` if none exists). Admit each candidate with
     change 2; on any refusal, close the plans already admitted and return
     **the leases unreleased** inside the refusal.
   - `pub fn step(&mut self, rows: u64, step: DenseSetStep<'_, 'ctx>) ->
     Result<DenseSetOutput>` where `DenseSetStep` has the fields of
     `DenseGraphStep` minus `graph`/`host_experts` plus `authority:
     &ResidencyAuthority`, and `DenseSetOutput { output, returned_inputs,
     launch_order }`. Refuse if `lost` is `Some`. Revalidate every lease with
     `authority.device_range` (unchanged offset and length, else refuse before
     launch). Take the plan out, `execute_dense`, `finish`, put the plan back.
     A pre-launch refusal puts the returned plan back. A post-submission
     refusal or a failed `finish` stores the held lease in `lost` (the set is
     then poisoned) and returns the error.
   - `pub fn set_segment_capture(&mut self, rows, enabled, ledger)`.
   - `pub fn close(self, ledger, authority: &mut ResidencyAuthority) ->
     Result<(), DenseSetCloseRefused<'r, 'ctx>>`: refuse (returning `self`)
     while `lost` is `Some`; close every plan (a refused close returns the
     set with that plan restored); then release every lease to `authority`
     (a refused release keeps the rest in the returned set).
6. **Test** `bucket_plans_share_one_resident_weight_copy` on the 3090
   `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, Shape A, buckets
   `[1, 2, 4, 8]`: a ledger from `measured_ledger`; a residency authority
   with `device_weights(uuid, total)` where `total` is the sum of the planned
   weight `physical_bytes`; `DeviceResidency`; a test `ChunkSource` serving
   each weight's bytes (the same encoding `stage_bindings` uses) by role name;
   acquire, drain, upload every weight; build the set; run task 0087's
   prompt-13 chunks plus one decode through `set.step`, with capture enabled,
   and compare with `host_step` using `assert_logits`. Assert
   `ledger.committed(scope, Tier::Device(PackedResidentWeights)) == total`
   (one copy, not four) while the set is open. Close the set, close the
   device residency and the authority as the residency tests do, and assert
   `ledger.outstanding().is_empty()`.

## Contract before implementation

- **Semantics:** unchanged per step; outputs match the host reference as the
  per-plan path does.
- **Resources:** the weights are charged once, in `PackedResidentWeights`,
  through the residency authority; each plan charges only activations,
  workspace and (with capture) graph pools.
- **Lifetimes:** no plan or captured graph can run without the set, the set
  holds the leases, and a step never leaves the set in flight. A poisoned set
  keeps its lost operation and its leases and refuses to close.
- **Failure:** every refusal before launch leaves the set usable.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`,
`residency_device`, `dense_tp2_device`, `cargo xtask-cuda test-gpu` (69/69).

**Coverage check (one mutant, reverted after):** in `DensePlanSet::admit`,
use offset `0` for every weight's address; the new test must fail.

**Stop conditions:** a change conflicts with the code; a weight's
`DeviceRange` (not its address) is needed somewhere; arch-check flags a
definition; a file outside the allowed list is needed.

## Result, filled after work

Implemented tier-aware device residency and `DensePlanSet`: resident-weight
plans omit their own weight region, resolve weight addresses through the
authority-backed leases, reject duplicate weight bindings, and retain plans
and leases across refusals. Added
`bucket_plans_share_one_resident_weight_copy` for Shape A buckets `[1, 2, 4,
8]` on the specified 3090; captured prefill chunks and decode matched
`host_step` at 0.000 BF16 ULP, with `PackedResidentWeights` charged once.

The offset-zero mutant failed before launch at lease-range revalidation, then
was reverted. Host gates passed: fmt, workspace and driver-feature clippy,
workspace tests, `arch-check`, and `spec-check`. GPU gates passed with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`: `dense_gemma_device` (11 passed, 2 ignored),
`residency_device` (1 passed), `dense_tp2_device` (1 passed), and
`cargo xtask-cuda test-gpu` (69/69).
