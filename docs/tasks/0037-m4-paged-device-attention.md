# Task 0037 — M4.1 common paged device attention at actual 32K context

Status: active contract before implementation, authorized by the owner on
2026-09-19.

## Identity and authority

- Task0037, first bounded M4 task; repository owner reviews and accepts.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base closure
  record `54b52e3`. `coordinator.md` is unrelated carried work and remains
  outside this task.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; checkpoint roots remain read-only.
- Requirement: roadmap M4 item1 and the 32,768-actual-context part of its exit;
  source-map lessons R04, R19 and R21. Read specs02/03/04/06/07/08/09, existing
  `moxie-oracles` attention/mask contracts, graph attention descriptors,
  task0013 paged state and task0017 sliding retention before editing.
- Pinned candidates to audit before adopting code: FlashAttention commit
  `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` and FlashInfer commit
  `91bda04c66f7cb851e1ab3b78b9fecea644b9844`. Legacy attention is evidence and
  a source of regressions, not the architecture to copy.
- O1–O5 remain resolved. O6/O7 remain open, so this task proves correctness,
  bounded state and hardware qualification without making a speed claim.

## Bounded deliverable

Implement one common BF16 paged-device attention path for the existing semantic
`Attention` operation. It must append K/V into persistent admitted device state
and execute whole or chunked prefill plus single-row decode over full causal and
sliding visibility, MHA and GQA, partial chunks and page tails. The same shared
API and state owner must run on both SM86 GPUs and SM120. At least one gate must
hold and attend to **32,768 actual rows**, then append and decode the next row.

`moxie-state` remains the transaction/state authority, `moxie-memory` the
admission authority, `moxie-kernels` the device implementation owner, and
`moxie-executor` the binding/launch owner. Model crates may provide semantic
parameters but cannot own pages, CUDA code or execution loops.

This task does not implement MLA, sparse/index attention, host-backed page
streaming, COW forks, prefix reuse, recurrent/convolution state, distributed
attention, a checkpoint-backed block, tuning or a performance claim. Those are
later M4/M5 slices. No fixed short-history fallback may be presented as 32K
support, and no host attention loop may hide behind a model adapter.

Allowed changes are the shared graph/type descriptors only where the current
contract is insufficient, paged state/admission, kernel catalogue/CUDA image,
executor binding, independent oracles/tests, architecture rules, xtask gates and
living records. Any adopted upstream source must have its pinned revision,
license, modified files and supported hardware/shapes recorded before copying.

## Contract before implementation

- Positions are absolute. Append is dense and ordered. A causal query at
  position `p` sees allowed keys through `p`, including the just-appended row;
  sliding visibility follows `moxie_graph::Visibility` exactly. Masked keys are
  excluded rather than represented by an arbitrary finite bias.
- Q/K/V and cache payloads are BF16 for this slice. Head counts, KV head counts,
  head dimension and score scale are explicit semantic parameters; query heads
  map to GQA KV heads by the existing divisible grouping contract. FP16 cache is
  unsupported by this task rather than silently treated as BF16.
- Scores, stable maximum subtraction, exponentials, online-softmax partials and
  weighted-value accumulation use the declared FP32 path. Output narrows only
  at the graph's BF16 boundary. Reordering is allowed only inside the existing
  data-dependent `moxie_oracles::attention_error_bound`, including its
  cancellation and gradual-underflow terms; changing that bound requires owner
  direction before implementation continues.
- The existing bound helper currently derives `1/sqrt(head_dim)` internally
  while the semantic operation accepts an explicit score scale. Before device
  code, parameterize the helper by the declared scale and add cases for scale
  1.0 and reciprocal-square-root scale. This repairs the helper's input without
  changing or widening its error formula.
- Logical pages map to checked physical page identities. Page bytes, page table,
  persistent KV, launch workspace and any staging are admitted before allocation
  by the existing authorities. Capacity and arithmetic overflow refuse with a
  typed error; allocation failure must not abort.
- Enqueued work retains page, activation and workspace leases until a completion
  event is observed. Cancellation before commit leaves the committed frontier
  and every prior byte unchanged, and releases or quarantines transient device
  resources through the existing mechanism. A failed append never publishes a
  partial row.
- Whole and chunked prefill, page layouts and repeated decode are alternative
  executions of the same semantic operation. The independent host oracle must
  build visibility and reference values without reading candidate page tables
  or kernel output as its expected result.

## Acceptance

1. Host tests cover full/sliding visibility, MHA/GQA, explicit scale, page sizes
   and boundaries, empty/partial/final pages, invalid head ratios, position gaps,
   overflow, capacity refusal and exact append/abort/rollback byte preservation.
2. Device tests pass on both SM86 UUIDs and SM120 for head dimensions 64 and 128,
   MHA and at least one nontrivial GQA ratio, full and sliding masks, one-row
   decode, multi-row prefill, uneven chunks and page tails. Each component is
   checked against an independent FP64 equation under the predeclared attention
   error bound and reports max, RMS and high-percentile error.
3. A full gate materializes **32,768 actual BF16 K/V rows**, attends over the
   declared visible history, appends row 32,768 and produces the next decode
   result. It compares whole versus chunked construction and verifies that
   admitted capacity, actual rows and visible rows are reported separately.
4. Admission-failure sweeps cover every allocation before enqueue; cancellation
   and injected launch/synchronization failure preserve the committed frontier,
   release or quarantine every lease, and allow a clean retry. No hidden host
   allocation or transfer grows with decode steps beyond the declared page and
   diagnostic bounds.
5. Architecture checks prove that model crates cannot own the path and that no
   second state/admission authority was added. Fmt, host and driver clippy,
   affected workspace suites, specification checks and a task mutation battery
   pass; failed, skipped and unsupported shapes are reported separately.
6. Update the support matrix with exact shapes, devices, state bytes and visible
   history. Preserve negative upstream/legacy results. Remove any superseded
   device-attention bridge only after all replacement gates pass.

Stop for owner direction before changing the numerical bound, cache precision,
canonical state ownership, importing code with unresolved license/provenance,
starting a bulk checkpoint operation, or claiming performance. A hardware/shape
that cannot pass is a named refusal, not permission to narrow the 32K exit gate.

## Result, filled after work

No implementation or result is claimed by this contract.
