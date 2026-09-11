# Handover — task 0014 sampler history accepted

## Workspace identity

Writable root `/home/rodrigo/Developer/moxie`, branch `main`; task contract
`4054dd7` follows task 0013 acceptance `d1f6bf0`; implementation is `23a7a40`.
All implementation changes in this
handover belong to task 0014. No unrelated dirty paths were present. Implementation
and final evidence are committed before handing over to the owner's reviewer.

Read-only legacy `/home/rodrigo/Developer/strata` remains at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, with its pre-existing untracked `.pi/`
and `tests/p2p/` preserved. No checkpoint was read or changed.

## Completed facts

[Task 0014](../tasks/0014-m1-transactional-sampler-history.md) implements an admitted
host base sampler: FP64 greedy/temperature distributions with legality, smallest-ID
ties, versioned Philox inverse-CDF drawing, and generated-token history/counts.
`moxie-sampling` is pure; `moxie-state::PagedSequence` composes history with the
existing transaction journal and physical pages; `moxie-memory` owns the single
allocation/reservation. See [ADR 0008](../decisions/adr/0008-transactional-base-sampling.md).

Acceptance tests cover the fixed exhaustive numerical/statistical gates, published
RNG vectors, two paged geometries, pending tokens, partial acceptance and explicit
count replay, wrong transaction IDs, cancellation after each participant mutation
and compile-time refusal of a stale prepared borrow. The allocation test stores
32,768 generated history entries and 32,769 physical rows with zero token-path
allocations, no retained growth after 10,000 aborts and complete close/fault cleanup.

Negative controls catch lost count undo, CDF endpoint mistakes, tentative prompt
contamination and foreign-ID mutation. A 100,000-token vocabulary exposed rounding
error in the first plain sum; compensated FP64 reductions satisfy the unchanged
tolerance. Both the initial failing experiment and restored passing evidence remain.

Final validation passes 584 host tests + 9 doctests, 600 device-feature tests +
12 doctests, both clippy lanes, 65 rejecting/18 accepted architecture fixtures,
formatting/specification and 39/39 aggregate real GPU regression cases. The first
GPU attempts failed because loaded driver and user-space libraries differed; the
owner repaired that environment before final GPU validation. Full commands,
UUIDs and retained hashes are in the task result.

## Decisions

Task 0013 remains accepted. The owner accepted task 0014 on 2026-09-11 after
independent review of `23a7a40` / `1e6f253`. The reviewer repeated all final gates
and passed 272 additional state combinations, finding no code blocker. The exact
layout paragraph is now explicitly identified as recorded after implementation;
the tracked record does not prove it preceded allocator work. The task's acceptance
section records this auditability gap and the independent evidence manifest hash.
**M1.4 remains active.** O1–O7 are unchanged. No numerical
threshold, public default or legacy compatibility requirement was relaxed.

One generation, fixed vocabulary/history capacity and one root host paged pool are
the qualified boundary. CPU workspace is charged separately. Input logits are
synthetic data at a checked prefix; model-result provenance, attention, service/CLI,
full history processors, speculation and entropy integration remain future work.

## Remaining hypotheses and blockers

Independent review is complete. No product throughput, checkpoint quality,
actual-context attention, GPU sampling or distributed-vocabulary claim is supported.
The prepared distribution's exclusive borrow is the publication authority; copied
probabilities do not confer it. Future engine integration must bind real output
provenance while preserving this single state/history transaction mechanism.

## Next task

Define one bounded shared generation-service/minimal diagnostic-CLI integration
assignment; no successor task contract has yet been authored.
Read documents 02/04/05/07, the accepted interpreter and state contracts, and the
source-linked application fixtures first. Do not create a client-private token loop
or sampler, claim M1.4 complete from this slice, or loosen a gate to bypass a review
finding. Stop at any dependent O1–O7 ruling and present the smallest needed decision.
