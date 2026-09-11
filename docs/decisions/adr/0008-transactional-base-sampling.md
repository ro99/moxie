# ADR 0008 — Base sampling as a shared transaction participant

- Date / author / status: 2026-09-11 / implementation agent / implemented, owner
  acceptance of task 0014 pending.
- Classification: roadmap defaults for ownership and counter RNG; implementation
  choices for physical composition, numerical reduction and draw mapping.
- Scope: `moxie-sampling`, `moxie-state` and the existing host allocation authority.
- Supersedes: none. No normative reference document is amended.

## Problem and mechanism

[Task 0014](../../tasks/0014-m1-transactional-sampler-history.md), contract `4054dd7`,
requires generated-token history to share task 0004's transaction mechanism and
task 0013's physical paging. A separate mutable sampler next to a public state
handle would permit acceptance, cancellation or append failure to update only one
participant. The existing oracle offers mathematical distributions but has no
admitted production storage or RNG/history integration.

Document 02 gives pure sampling ownership to `moxie-sampling`; document 04
requires explicit restoration of accumulated sampler state and a versioned
counter generator. The frozen legacy sampler uses generated-token spans/counts
and MT19937/Gumbel-max. Document 01 explicitly permits new seeded sequences with
versioned repeatability; legacy exact sampled text is not the migration gate.

## Options examined

- Cloning token history and count tables at every transaction would make begin
  proportional to generation length and create repeated allocation pressure.
- An independently resolved history journal would duplicate transaction ownership
  and leave paging's internal append-abort path unable to restore sampling.
- Giving sampling a paged-state or memory dependency would violate pure transform
  ownership and allow a second allocation/cache access path.
- A single physical host buffer with separate state/workspace charges preserves
  admission and lifetime authority. Pure sampling can operate on borrowed byte
  regions without alignment casts, unsafe code or a memory dependency.

These are ownership/resource choices, with no measured prefill/decode speed or
checkpoint-quality comparison. No GPU sampler, distributed vocabulary or model
generation is implemented in this slice.

## Decision and authority

`PagedSequence::with_sampling` constructs one fixed host pool. Its existing journal
stores one sampler length mark, using the same `StateTransactionId`. The shared
sampling component maintains ordered `(position, token)` entries and separate
tentative/committed count tables. Abort undoes suffix count increments; resolved
rollback actually rebuilds retained counts and then mints the existing explicit
replay evidence. `StateKind::SamplerHistory` is still `Explicit`.

Commit keeps the journal open while accepting sequence transitions and publishing
the matching history prefix, then closes it with zero additional transitions.
`commit_prefix_cancellable` observes cancellation before mutation, after frontier
publication and after history publication; either intermediate state can still
abort. Once the journal is closed, the caller may emit accepted output through
the existing separate emission operation.

The prepared sample holds the exclusive sequence borrow through distribution
access, draw and staging. A copied numerical distribution cannot publish state.
The input logits remain explicitly supplied synthetic prefix data: the facade
checks materialized/logical position, but does not claim model-output provenance.

The exact extra physical layout is `16*H + 16*V` StateSpill bytes and `8*V`
CpuWorkspace bytes, coallocated after the paged KV bytes. Control charges use the
existing checked facade/journal/node-size expression plus the second schema entry.
`HostBuffer::allocate_with_workspace` keeps one reservation and one allocation.
A failed mixed-tier physical allocation reports `CapacityExceeded { tier: None }`
with the full requested allocation bytes; `None` means multiple charged tiers.
Lineage allocation failure continues to name Pageable and its exact bytes.

The base profile accepts temperature [0,10], FP32 logits, explicit seed and an
optional legality mask. It computes FP64 normalized probabilities, with
ascending-order compensated reductions for weights, validation and CDF prefixes.
Ordinary accumulation on a 100,000-token uniform fixture exceeded the fixed
1e-12 normalization threshold. Compensation fixes that case; the threshold was
not relaxed. Greedy chooses the smallest ID on ties and uses no random words.

`moxie-philox4x32-10-cdf-v1` uses the seed as key and generated position/domain
as counter, with inverse-CDF draws from the top 53 bits of the first two words.
Proposal, acceptance and filtering domains are reserved, not enabled processors.
Three external Philox known-answer vectors are from Random123 `tests/kat_vectors`
at `9545ff6413f258be2f04c1d319d99aaef7521150`; only factual oracle values are retained,
not an upstream runtime/dependency. The algorithm is implemented from task 0014's
fixed round equations. Repeatability is within this numerical/RNG execution profile.

## Evidence and acceptance

The task result records complete commands, counts and retained hashes. Tests cover
exhaustive tiny logits/masks, external RNG vectors, analytic CDF intervals, the
predeclared statistical bins, two paged geometries, every mutation/commit fault
boundary, exact accumulated-count restoration, and compile-time stale-borrow refusal.
The allocation gate stores 32,768 generated entries plus 32,769 KV rows using
1,187,358 admitted bytes, with zero token-path allocations and no retained growth
after 10,000 aborts. Both physical-buffer and lineage allocation failures unwind.

The tentative prompt-contamination mutation initially survived an end-of-transaction
assertion because zero-accept commit removed it. The test now checks history directly
after prompt publication; the same mutation fails. Retain both controls. Missing
count undo, wrong CDF boundary and foreign-ID mutation controls also fail as intended.

## Enforcement and removal

Architecture fixtures accept state consuming pure sampling and reject sampling
depending on state, memory or CUDA, and models depending on sampling. The oracle
remains independent; production never imports it. No legacy source is modified.
No compatibility surface or feature is removed. Service/CLI, device attention,
common history processors and multiple generations per session require later
bounded integration. Re-evaluate the fixed layout/profile when those contracts
need growth, COW, stochastic processors or distributed normalization; do not add
another sampler or transaction authority.
