# Task 0014 — M1.4 transactional sampler history and base distributions

Status: **active contract; implementation not started** (2026-09-11).
This contract must be committed before implementation. Task 0013 is accepted;
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

Contract authoring only. No implementation, sampler qualification or new GPU/
performance result is claimed. Implementation commits must follow this contract.
