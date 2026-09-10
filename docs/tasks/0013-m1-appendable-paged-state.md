# Task 0013 — M1.4 appendable paged state

Status: **active**. Contract recorded before implementation; owner review pending.

## Identity and authority

- Milestone/owner: M1.4 / implementation agent; acceptance belongs to the owner.
- Writable root `/home/rodrigo/Developer/moxie`, `main`, base `b0f06fc`; initially clean.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and `tests/p2p/`
  are preserved and are not source evidence.
- Repairs M1.4's missing physical paging, R02/R08/R20. Extends accepted task 0004,
  without reopening M1.3.
- Read contracts: documents 01–09, TASK/HANDOVER templates, owner-gate register,
  task 0004 and the M1.3 closure handover; specification digests verified.
- Sources inspected at the frozen commit: `glm53_sequence.hpp:26` and
  `glm53_sequence.cpp:11` (paged row addressing, append/truncate, immutable shared
  prefixes); `gemma4_runtime.cpp:174` (KV marks); `kimi_k3_kv_cache.cpp:206`
  (recurrent state cannot truncate). Pinned vLLM architecture overview at
  `51da0ca66c8065619c79e35dff97aa99aeaf5644` supplies ownership context, not a
  multi-user scheduler or adopted code. No upstream code is copied.
- O1–O7 remain open. None blocks this synthetic, host, >=16-bit storage slice.
  Stop before dependent model, quantization, storage or product performance claims.

## Bounded deliverable

One appendable, fixed-capacity host paged KV store physically bound to task 0004's
`SequenceState` transactions. `moxie-state` owns page layout and state effects;
`moxie-memory` admits and owns its host backing allocation. The paged facade owns
one sequence and exposes only an immutable counter/provenance view, so callers
cannot advance its executed frontier independently of its rows.

Allowed files: state and memory crates, their tests/manifests, architecture
allowlist/negative fixtures, Cargo.lock, and tracked task/evidence/handover records.
No model, CUDA, kernel, planner or application implementation changes.

Initial capability: one root branch, uniform per-layer KV geometry, fixed page
capacity admitted before use. The page table and complete padded pool coexist in
one allocation. Append activates pages within that envelope; abort/truncation
return the suffix to that pool. Pool growth, COW forks/prefix sharing, sliding
reclamation, device attention, recurrent state, samplers, service and CLI remain
subsequent work. Requests for unsupported capabilities must fail explicitly.

Consumers: synthetic multi-layer append/prefill/continuation producers with at
least two head/dimension/page geometries, checked against independent flat bytes.
Existing interpreter/selected-device-chain consumers must keep passing. The dense
`moxie-interp::KvCache` remains explicitly a host mathematical reference; this task
does not create a second interpreter or route production attention through it.
Its reference-only role expires for production state at attention integration;
the mathematical oracle remains. No temporary production bridge is introduced.

## Contract before implementation

- Geometry: positive layers, KV heads, key/value dimensions, page tokens and
  maximum tokens; checked products and ceil division. BF16/FP16/FP32 encodings only.
- Input: one complete position across every layer, with exact encoded K/V byte
  widths; position must equal the executed frontier and transaction must be open
  on this sequence. Rows are copied bit-for-bit; no arithmetic or rounding.
- Physical layout: page table of byte offsets, followed by fixed pages. Within
  each page: layer-major K rows then V rows, each row head-major. Final page
  padding is charged and inaccessible as logical context. Borrowed row views
  cannot outlive storage or overlap a mutable append/release.
- Resource envelope: padded KV payload + page table, plus a conservative bounded
  reserve for existing sequence lineage/journal metadata. No allocation per row
  or history flattening. Existing ledger admits the whole envelope atomically;
  allocation failure unwinds the reservation. Host allocation/allocator overhead
  is covered by host headroom, as with existing resource consumers.
- No CUDA or transfers. Source slices need live only through the synchronous
  append. No host pointer is exported for asynchronous use; a future device
  integration must use accepted event-retained leases.
- `begin`, `commit_prefix` and `abort` use task 0004's actual transaction mechanism,
  not another transaction counter or clone-based journal. As accepted there,
  commit accepts n additional tokens and retains executed work; n=0 materializes
  pending state. Partial verification resolves then explicitly rolls back to the
  accepted prefix, truncating physical pages and invalidating suffix provenance.
- Known-transaction append failure/cancellation aborts the complete transaction,
  restoring counters, lineage and all visible rows. Invalid/foreign/resolved
  transaction IDs mutate nothing. Destructive rollback/publication while open
  follows task 0004's refusal rules. Transaction identities must not cross sequences.
- Capacity refusal never shrinks context or changes precision. A failed commit
  remains abortable. Explicit close releases backing and reservation even with
  unfinished work and no next token. Wrong-ledger close returns the live owner for
  retry; accidental drop leaves a visible ledger charge, matching existing policy.
- Oracle threshold: exact equality of encoded bytes, counters, lineage and ledger
  totals. No new floating tolerance. Application/sampler behavior is unchanged.

## Acceptance

- Full host workspace tests; fmt, clippy with warnings denied, architecture and
  specification checks. Add a negative fixture for the new state→memory edge's
  boundary; shared crates must still reject concrete model ownership.
- Page-boundary appends and uneven tails; whole versus chunked publication and
  later continuation; BF16, FP16, FP32 bits; repeated abort/retry and rollback;
  cancellation injected through the real append path at every publication boundary.
- Wrong sequence/transaction/position/shape, duplicate begin, resolved transaction,
  insufficient ledger capacity, arithmetic overflow, low-bit precision and unsupported
  forks. Failed release retains authority. Second sequence after close succeeds.
- 32,768 actual stored rows with small synthetic widths: storage capacity evidence
  only, explicitly not actual-context attention or model support. Reconcile padded
  bytes/table/control reserve with ledger admission and release; no per-row growth.
- Run device-feature workspace and aggregate real GPU regression qualification on
  the two 3090 UUIDs and 5060 Ti UUID with PCI_BUS_ID ordering. No new CUDA behavior:
  topology, sanitizer, model-quality and paired performance gates are unmeasured
  for this slice; no default strategy or speed claim is made.
- Update support matrix and bounded handover with exact commands, failures and
  remaining M1.4 work. Do not mark M1.4 complete or task accepted without owner review.

## Result, filled after work

Pending implementation and validation.
