# Task 0014 — M1.4 transactional sampler history and base distributions

Status: **implementation and validation complete; owner review pending**
(2026-09-11). Contract `4054dd7` preceded implementation. Task 0013 is accepted;
M1.4 remains active.

## Identity and authority

- Owner: implementation agent; independent review precedes owner acceptance.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, clean base
  `d1f6bf0a272006c3fe5fe51996503238a5d1cd6a`.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Preserve its existing untracked
  `.pi/` and `tests/p2p/`; these are not source evidence.
- Repairs the missing sampler/history portion of roadmap M1.4, R08/R20/R22.
  Extends accepted [task 0004](0004-m1-state-transactions.md) transactions and
  [task 0013](0013-m1-appendable-paged-state.md) physical paging.
- Normative contracts: documents 01–09, especially 02 ownership, 03 admission,
  04 frontiers/restore/RNG, 05 `moxie-v1`, and 07 distribution/state validation;
  TASK/HANDOVER templates and the owner-gate register.
- Frozen source: `include/strata/engine/sampling.hpp` (`SamplingHistory`, generated
  windows, options); `src/engine/sampling.cpp:16` (uniform/Gumbel draw), `:95`
  (window counts), `:110` (penalty ordering), `:245` (pipeline), `:469`
  (validation); `tests/test_sampling.cpp` (greedy, seeded draws, raw logprobs,
  generated-token windows). Source equations are migration evidence; the old
  all-banned fallback and unstable filtered ties are not contracts to preserve.
- Current implementation references: `moxie-oracles::sampler`,
  `SequenceState::{begin,commit_prefix,abort,restore_evidence,rollback_to}`,
  `PagedSequence`, and `moxie-memory::HostBuffer`. Document 08's pinned upstream
  references supply ownership context; no upstream runtime or kernel is adopted.
- O1–O7 remain open. No decision is needed for this synthetic host slice. Stop
  before dependent public-surface, checkpoint, quality or performance claims.

## Bounded deliverable

One admitted host sampling session exposes normalized greedy/temperature
distributions and generated-token history whose publication and restoration
participate in the existing sequence transaction. A synthetic consumer combines
it with actual paged rows, including a committed token pending forward execution.

- `moxie-sampling` owns pure distribution equations, history/count semantics and
  versioned RNG. It consumes bounded storage views and has no state, cache,
  CUDA, model, executor or allocator dependency.
- `moxie-state` owns the composition with its existing transaction journal and
  paged facade. It may consume the pure sampling types; sampling never reaches
  back into state. `moxie-memory` remains the physical allocation/admission owner.
  There is no second transaction ID, independently resolvable history journal,
  model callback or production generation loop.
- Allowed files: new sampling crate; state, memory and types for the minimum
  composition/resource vocabulary; synthetic integration tests; independent
  oracle tests; workspace manifests/lockfile; architecture allowlist/fixtures;
  tracked task, evidence and handover records. Existing interpreter production
  equations, CUDA/kernels, concrete models and application code are out of scope.
- First capability: one root sequence and one generation, fixed vocabulary and
  history capacities, explicit temperature and seed. Preserve ordered generated
  tokens and occurrence counts, with a read-only last-N-token view (N=0 means all).
  Prompt tokens never enter generated-token counts.
- This slice enables no penalties, filters beyond explicit legality, DRY, n-gram
  bans, XTC, entropy, speculative proposer/verifier, grammar, raw-logprob reporting,
  HTTP or CLI. Unsupported configurations must be unrepresentable or explicitly
  refused. Existing oracle filter coverage stays available as reference coverage.
- Consumers: independent vocabularies of 3 and 7 tokens, different prompt lengths
  and paging geometries; pure distribution tests and the actual paged composition.
  No checkpoint or attention-execution claim follows from synthetic logits/rows.
- Expiry: the synthetic producer stays a test fixture. Later service integration
  consumes this shared API. Keep the independent sampling oracle; do not call it
  from production or copy a complete second pipeline into state.

## Contract before implementation

### Distribution and drawing

Input is a contiguous host FP32 logit vector of declared nonzero vocabulary V,
optional exact-length legality mask, and finite temperature T in [0,10], matching
the legacy request validation bound. An explicit seed is required by this internal
API; no application default is selected. Token IDs are ascending indices, checked
to fit u32. The existing oracle's wider mathematical temperature range is not a
request-validation promise.

Reject NaN and positive infinity as `Numerical`, including in explicitly masked
input. Negative infinity means zero support. Empty vocabulary, malformed shape,
invalid temperature or token ID is `InvalidRequest`; no surviving finite legal
logit is `Numerical`. Neither error may draw RNG or publish history.

For T=0, return exactly one-hot argmax over legal finite scores, with the lowest
token ID winning ties. For T>0, promote logits and temperature to FP64, subtract
the maximum before dividing, and compute
`w_i = exp((l_i - max(l))/T)`, `p_i = w_i / sum(w)` in ascending token order.
Masked weights are exactly zero; underflow to zero is permitted. FP64 subtraction
prevents opposite finite FP32 extrema overflowing before scaling. Expose the
normalized FP64 distribution separately from drawing, through a borrow of admitted
workspace. Pure distribution queries mutate no sequence/history/RNG state.

Draw by inverse CDF in ascending token order: for uniform u in [0,1), choose the
first positive-mass token whose cumulative mass exceeds `u * sum(p)`. Rounding
at the final boundary may select only the last positive-mass token. Zero-mass
tokens can never be selected, including u=0. Greedy consumes no random words.

Use the document 04 candidate **Philox4x32-10**, profile
`moxie-philox4x32-10-cdf-v1`. Key is the low/high u32 halves of the u64 seed;
counter is `[step_low, step_high, domain, 0]`. Step is the generated-token position
within this generation, including the tentative prefix being sampled. It is not
prompt length, executed-row count, transaction ID or number of distribution queries.
Reserve domain 0 for target draw, 1 proposal, 2 acceptance, 3 stochastic filtering;
only target drawing is enabled here. Do not infer statistical independence merely
from differing sample outputs; test the exact distinct counter assignments.

Each round multiplies c0 by `0xD2511F53` and c2 by `0xCD9E8D57`, obtaining high/low
u32 halves h0/l0 and h1/l1. The next counter is
`[h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0]`. Between rounds add `0x9E3779B9` and
`0xBB67AE85` to the key halves with u32 wrapping; execute ten rounds. Combine
output words 0 and 1 as `(word0 << 32) | word1`; set
`u = (word64 >> 11) * 2^-53`. Golden integer vectors must be computed independently
of production code and retained with their derivation. Counter overflow refuses;
it never silently reuses a position. This is a versioned engine RNG, not security
randomness. Seeded text compatibility with legacy MT19937/Gumbel-max is not claimed.
Repeatability is within the same declared numerical/RNG execution profile.

### History, frontiers and the existing transaction

Generated history stores token IDs in order, their logical sequence positions,
and an exact whole-generation count table. A tentative view may include staged
tokens; the committed view exposes only accepted generated transitions. Prompt
append/materialization and repeated distribution queries change neither history
nor the next target-draw position. A selected token becomes history only through
the shared transaction publication API; drawing alone is not publication.

The state facade owns and validates the existing `StateTransactionId` before
changing any sampling participant. Extend its existing journal with bounded
participant marks/undo information; use suffix tokens to undo count increments
and restore the generated step. No whole-history/count-table clone per begin or
draw. Foreign, unknown and resolved IDs mutate nothing. Active-session APIs cannot
bypass history by accepting an anonymous generated count or mutating a raw state
handle. Existing state-only consumers retain their accepted semantics.

`commit_prefix(txn,n)` still means n additional accepted transitions and retains
executed work. It publishes exactly the matching staged generated prefix, never
unaccepted candidates. Partial acceptance discards unpublished sampler suffix
effects and restores the next draw position to the accepted generation prefix;
the existing explicit resolved rollback truncates unwanted physical rows. Test
the intermediate executed/accepted distinction rather than redefining task 0004.
`commit_prefix(txn,0)` can materialize a pending token without counting it twice.

A committed sampled token may lead executed KV by one pending token. Sampling
must not synthesize that token's KV row or valid next logits. Synthetic producers
provide explicit logits for a checked logical prefix; this is storage/provenance
testing, not a production model-output path. Reusing a distribution for a changed
prefix must fail; copying numerical probabilities does not confer state authority.

Abort restores generated IDs, counts, draw position, all frontiers, lineage and
visible physical rows to their pre-begin values. The real paged append failure
path must restore sampling even when paging aborts internally. Commit validation
failure remains abortable. Cancellation is checked at each mutation/publication
boundary; test faults after sampler mutation as well as after physical row copy.
No external emission is permitted while the transaction is open.

Keep `StateKind::SamplerHistory` classified as `RestoreCapability::Explicit`.
Resolved rollback must perform real count/history restoration before producing
sequence/prefix/lineage-bound restore evidence; decrementing a counter or changing
the schema to `Truncate` is insufficient. Refuse unsupported forks and restore
requests. A second generation after explicit close must start with empty history.

### Resources and lifetimes

Implementation layout fixed before allocator work: append `16*H + 16*V` state
bytes and `8*V` CPU workspace bytes to the existing physical KV allocation. Each
history entry is little-endian `(position:u64, token:u32, padding:u32)`; two u64
count tables hold tentative and committed counts. Workspace stores FP64 bits in
little-endian bytes, accessed without aligned pointer casts. One allocation and
one ledger reservation cover KV plus sampler state (StateSpill), probabilities
(CpuWorkspace), and the existing checked control expression (Pageable), extended
for the enlarged facade/journal and second schema entry. No additional heap-backed
sampler metadata is necessary. H must be positive and no larger than KV capacity.

Admit all simultaneous sampler storage with the existing ledger before allocation:
bounded generated token/position storage, count table, probability workspace and
participant undo metadata. Publish an exact checked byte expression before writing
the allocator path, including padding and coexistence with task 0013's backing and
control reserve. Count each physical allocation once. Storage uses the existing
memory authority; pure sampling receives slices, not a private allocator.

V and maximum history H are fixed at construction. No allocation inside successful
distribution/draw/history append, no O(H) flatten/clone per token, and no retained
growth across repeated aborts. Any existing per-transaction allocation remains
bounded and charged. Capacity refusal preserves precision and requested history.
Post-admission allocation failure is `CapacityExceeded` with the correct tier and
requested bytes and releases every reservation allocated by the failed constructor.

Logits/masks are synchronous borrows; a distribution view cannot outlive or overlap
mutation of its backing. No CUDA/transfer/event work is introduced. Explicit close
releases physical storage before charges, even with unfinished work and no next
token. Wrong-ledger close preserves the owner for retry; accidental drop keeps the
existing visible-charge policy. No sampler-specific allocation authority is added.

## Acceptance

- Independent FP64 analytic reference over every logit vector in {-2,0,2}^V for
  V=1..5, temperatures {0,0.25,1,2,10}, and every legality mask. Greedy bits/support
  and errors are exact; positive-temperature max absolute probability error and
  normalization error are <=1e-12. Include FP32 extrema, subnormal positive T,
  one survivor, all masked, ties, NaN and both infinities. Compare the existing
  FP32 oracle on ordinary fixtures with max absolute error <=1e-6; retain its tests.
- Independent Philox integer vectors, word ordering and domain/counter boundary
  checks. Exact CDF intervals for tiny rational distributions, including zero mass
  and endpoints. Test seed replay, abort/retry, prompt-length independence and
  repeated distribution queries. Do not rely only on matching sampled text.
- Supplement with 100,000 draws per distribution for uniform V=3 and V=7 and
  p=(1/2,1/4,1/4), seed 33377335. Retain all 13 bin counts. For each bin require
  absolute frequency error <=sqrt(log(2*13/1e-6)/(2*100000)); this predeclares a
  Hoeffding/union false-alarm bound under the uniform-draw model. Statistical
  evidence supplements the exact algorithm and CDF tests.
- Real paged composition: whole/chunked prompt, generated-only windows/counts,
  pending token materialization, partial acceptance then resolved rollback,
  repeated abort/replay, foreign/resolved IDs, stale prefix views, failed commit,
  cancellation at every mutation and close followed by a second session. Compare
  exact history/counts/RNG coordinates/frontiers/lineage/rows to independent replay.
- Counting allocator: at least 32,768 actual generated-history entries, zero
  successful per-token allocations, 10,000 abort/retry cycles with no retained
  growth, exact admitted-byte reconciliation and zero outstanding charges after
  close. This is history/storage evidence, not 32K attention. Inject failure into
  each new post-admission allocation and assert complete cleanup and error fields.
- Mutation controls must detect missing count rollback, prompt contamination,
  wrong CDF boundary and a foreign transaction mutating history. Restore each
  deliberate mutation and preserve failing-control conclusions.
- Run `cargo test --workspace --locked --offline`, `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`,
  `cargo xtask arch-check`, `cargo xtask spec-check`, and `git diff --check`.
  Add accepted pure-sampling/composition fixtures and rejecting sampling-to-state,
  sampling-to-CUDA, sampling-to-memory and model-to-sampling ownership fixtures.
- Run workspace tests and clippy with
  `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda`
  as affected-consumer checks. Real GPU regression `cargo xtask-cuda test-gpu`
  qualifies existing behavior on the two 3090 UUIDs and 5060 Ti UUID with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`; it is not GPU sampler qualification. No new
  topology, sanitizer, checkpoint-quality or paired performance lane is required
  for this host slice; record those as unmeasured, with no performance/default claim.
- Update the support matrix with the exact bounded sampler/history capability,
  retain omitted processors and service/CLI as outstanding, and produce a bounded
  handover with commands, source identities and hashed raw evidence retained through
  review and M1 closure. Independent review and owner acceptance remain required.
- Stop for an ownership cycle, inability to compose with the existing journal,
  unbounded history copying/allocation, missing independent RNG oracle, a proposed
  numerical-gate relaxation or dependency on an open owner gate. Report the smallest
  required decision. Do not introduce a model runtime or claim M1.4 complete.

## Result, filled after work

Implemented at `23a7a40` after contract `4054dd7`. [ADR 0008](../decisions/adr/0008-transactional-base-sampling.md)
records the shared owners, physical layout and numerical/RNG choices. Implementation
source identity and final gate results are recorded below for independent review.

### Changed owners and behavior

- `moxie-sampling`: pure FP64 greedy/temperature distribution, explicit legality,
  lowest-ID ties, Philox4x32-10 target draws, generated token/position storage,
  tentative/committed count tables and bounded window views. It imports only
  `moxie-types` in production; `moxie-oracles` remains a dev-only independent oracle.
- `PagedSequence::with_sampling`: binds that history to the existing
  `SequenceState` journal using one length mark and the existing transaction ID.
  No second journal or mutable raw-state escape is introduced. Distribution
  preparation holds an exclusive sequence borrow through staging; a compile-fail
  test proves abort cannot invalidate a prepared sample while it remains usable.
- Commit publishes frontiers and the accepted sampler prefix while the original
  journal is open, then resolves it with zero additional accepted transitions.
  Cancellable commit observes every participant boundary; abort restores both
  count tables, token/position entries, lineage/frontiers and physical rows.
  Partial acceptance preserves executed work until explicit resolved rollback.
  That rollback rebuilds counts before minting existing explicit replay evidence.
- `HostBuffer::allocate_with_workspace`: one physical pool and reservation with
  StateSpill, CpuWorkspace and Pageable charges. CPU workspace does not advertise
  context scaling. A mixed state/workspace allocation failure names all requested
  bytes and `tier: None`; a workspace-only failure names CpuWorkspace. Lineage
  failure names Pageable, retaining task 0013's correction and regression.

The one-generation host facade accepts synthetic logits for a checked materialized
logical prefix. It does not certify a model-produced logits handle or execute
attention. A sampled token can be committed while pending execution; materializing
it with zero additional acceptance does not duplicate history. Anonymous generated
acceptance, prompt growth under generated history, unsupported forks and stale/
foreign transaction identities are refused.

### Independent numerical and RNG evidence

Exhaustive {-2,0,2}^V logits for V=1..5, every mask and temperatures
{0,0.25,1,2,10} pass the frozen gates: exact greedy/support/errors, max absolute
FP64 probability and normalization error <=1e-12. Ordinary fixtures agree with the
existing FP32 oracle within 1e-6. Additional cases cover finite FP32 extrema,
subnormal positive temperatures and vocabularies 32,768/100,000/131,072.

The initial plain FP64 sum refused a valid 100,000-token uniform distribution.
The retained failing test demonstrates that problem; ascending-order compensated
reductions now pass the same 1e-12 bound. No threshold was loosened. Three external
Philox known-answer vectors, exact counter/domain assignment and word-to-uniform
mapping pass. The source pin is recorded in ADR 0008 and the test itself.

Fixed seed 33377335, 100,000 draws per distribution, epsilon
`0.009239482424894198` from the predeclared 13-bin bound:

| Expected probabilities | Observed counts |
|---|---|
| (1/3,1/3,1/3) | 33123, 33361, 33516 |
| (1/7) repeated 7 times | 14104, 14227, 14397, 14238, 14404, 14304, 14326 |
| (1/2,1/4,1/4) | 49913, 24970, 25117 |

Every bin passes. Exact inverse-CDF interval tests supplement these statistical
results; sampled-text agreement is not the distribution oracle.

### Physical resource evidence

The counting-allocator executable stores **32,768 generated entries and 32,769 KV
rows**, vocabulary 7, H=32,769, two BF16 layers, K dimension 2/V dimension 1,
page width 127 and KV capacity 32,770. Extra history/count storage is **524,416 B**,
workspace **56 B**, complete physical backing **921,260 B**, and complete
admission including control **1,187,358 B**. Distribution, draw, history and KV
append allocate zero heap. Ten thousand abort/retry cycles retain zero growth;
close returns the requested heap delta and every ledger charge to zero.

Faults targeted at the combined backing and lineage requests both return exact
`CapacityExceeded` fields and zero outstanding/tier charges. There are no new
separate sampler heap allocations. Existing sequence/ledger bookkeeping remains
under the accepted control/headroom contract. This measures requested heap and
stored history, not RSS, long-context attention or product performance.

### Failed controls and environment failures

Removing count undo, changing CDF `>` to `>=`, and allowing foreign-ID history
mutation each made the required regression fail (exit 101). The prompt-contamination
mutation initially passed an end-of-transaction check because zero-accept commit
removed it; the test now checks tentative history immediately after prompt append,
and the identical mutation fails. Both attempts are retained. All deliberate code
mutations were removed before final validation.

The first device workspace attempt failed on CUDA error 803; another test then
observed the poisoned test mutex. `nvidia-smi` independently reported a driver/library
mismatch: loaded module 610.43.02, CUDA/NVML libraries 610.57.04. Aggregate GPU
enumeration exited 2 for that same environment issue. The owner repaired the driver;
all three UUIDs now report 610.57.04. No driver or system configuration was changed
by this task. Final post-repair results are recorded separately below. A focused
Cargo invocation selecting packages without the named device features was rejected
before compilation; it is not a test pass and is retained as command-error evidence.

### Final validation and retention

| Gate | Exact command / final result |
|---|---|
| Host workspace | `cargo test --workspace --locked --offline`: **584 tests + 9 doctests**, zero failed/ignored |
| Device-feature workspace | `cargo test --workspace --features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda --locked --offline`: **600 tests + 12 doctests**, zero failed/ignored |
| Focused sampler/state/memory | `cargo test -p moxie-sampling -p moxie-state -p moxie-memory --locked --offline -- --nocapture`: passed, including all retained bins and allocation metrics |
| Host clippy | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`: passed |
| Device clippy | Same workspace/all-targets command with the device feature list above: passed |
| Architecture | `cargo xtask arch-check`: **65 rejecting + 18 accepted fixtures**, 12 rules |
| GPU regression | `cargo xtask-cuda test-gpu`: **39 passed, zero failed/skipped**, SM86/SM120 qualified after driver repair |
| Format / specification | `cargo fmt --all -- --check`, `cargo xtask spec-check`, `git diff --check`: passed; all 10 normative documents unchanged |

GPU identities: 5060 Ti `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`; 3090
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`; 3090
`GPU-81fe4578-59b2-37c4-421e-287cdac78704`. PCI_BUS_ID ordering was applied by
Cargo configuration (and explicitly to standalone driver diagnostics). Driver
610.57.04 is the final observed environment. No new CUDA sampler, topology,
sanitizer, model-quality or paired performance result is claimed.

Raw logs
are retained outside git in `/home/rodrigo/Developer/moxie/results/task0014/`, through
owner review and M1 closure. `SHA256SUMS` records every retained log, including
negative experiments and the resolved environment failures.

The final log manifest SHA-256 is
`a924caff370c9240f6383ef7513b65b853d095d95f22f0ce25c0c64cfb361f8a`;
`sha256sum --check --status SHA256SUMS` passes. Final full lanes are `review-host.log`,
`review-device.log`, `review-gpu.log`, `review-clippy.log`,
`review-device-clippy.log`, `review-arch.log`, `review-spec.log` and
`review-fmt.log`; numerical/bin/allocation output is in `final-focused.log`.
Earlier runs remain labeled separately, including the initial GPU environment
failures and the rejected focused feature command.

Build identities at `23a7a40`: CUDA xtask
`target/debug/deps/xtask-ab778877d540f843`, SHA-256
`1e15adec592366b510f990ae61954cb7ccffd7b56bf252e550da10d700d9b8cd`;
host xtask `target/debug/deps/xtask-1b2a551c0290ffef`, SHA-256
`5fb0463d416c9130cd8d2fa21575716b9362c805074d409caa0ad12af23cf45a`.
Build outputs may expire on clean; source, log hashes and tracked conclusions remain.

### Deletion and remaining scope

The first implementation's plain reductions were replaced by compensated sums;
there is no competing production sampler or generation loop to retire. The dense
interpreter and sampling oracle remain mathematical references. No legacy path,
processor, public surface or accepted test was removed.

This implements the bounded sampler/history slice. Owner review remains required;
M1.4 stays active. After acceptance, define the shared generation-service and minimal
diagnostic-CLI integration task. Device attention, full sampler processors, model
execution, quality, topology and paired prefill/decode performance retain their
separate gates; none is claimed by this host sampler result.
