# Task 0013 — M1.4 appendable paged state

Status: **accepted by the owner, 2026-09-10; M1.4 remains active**.
Contract committed at `a80ff2c` before implementation `c14e32a`; correction
`c9a4b33`; independent review and retained evidence through `0f39cf5`.

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

Implementation: `c14e32a` (2026-09-10). No model code, kernels or device execution
path changed. The shared owners are:

- `moxie-memory::HostBuffer`: admits padded backing and the control reserve in
  one ledger envelope, allocates synchronously, and frees before releasing the
  reservation. Wrong-ledger release preserves the authority. Allocation failure
  after admission unwinds the charge; accidental drop remains visible.
- `moxie-state::PagedSequence`: owns one root sequence, checked geometry and
  page table. It delegates transaction/frontier/lineage changes to `SequenceState`;
  there is no second journal or transaction counter. Append copies complete K/V
  rows, and failure aborts prior work plus partial copies through the same path.
- `SequenceState::begin`: IDs are now process-unique, with checked exhaustion.
  Previously two separate sequences issued ID 1 and could resolve each other's
  first journal. Existing consumers inherit this correction. Three eagerly built
  error messages became lazy so successful row append allocates no heap.

The fixed pool is reserved at construction. Truncation deactivates/zeros the
suffix; those bytes remain charged and available inside the admitted pool until
close. It does not release pages to the global ledger individually or grow the
pool. `commit_prefix(n)` retains task 0004's accepted semantics: n additional
accepted tokens, executed work retained. Partial verification then calls the
existing rollback at the resolved boundary. No sampler/verifier is implemented.

**Storage evidence:** the counting-allocator executable stored 32,768 actual rows
with two BF16 layers, one KV head, K dimension 2/V dimension 1, 127 tokens/page and
maximum 32,769 tokens. Backing is **396,788 B**, including **2,072 B** of page table;
control reserve is **265,937 B**, for **662,725 B** total admitted. There were zero
allocations inside append, zero retained growth after 10,000 abort/retry cycles,
and zero remaining allocation/ledger delta after close. This is a requested-heap
test, not RSS or a performance measurement. Existing journal allocation is per
transaction, bounded by the reserve; no payload/lineage allocation is per row.
The fixed control bound was checked against the pinned Rust 1.97.1
`library/alloc/src/collections/btree/node.rs` (B=6, 11 entries, 12 edges) and the
counted live-transaction footprint. Allocator and authority bookkeeping use the
existing host headroom contract.

Additional source/test checks: frozen `tests/test_glm53_manifest.cpp:106` exercises
paged rows/COW; `tests/test_kimi_k3_kv_cache.cpp:90` refuses recurrent truncation.
The new slice carries forward append/truncate and the recurrent refusal boundary,
without claiming COW support. The earlier searched filename
`tests/test_glm53_sequence.cpp` does not exist at the frozen commit.

### Verification

| Gate | Exact command / result |
|---|---|
| Host | `cargo test --workspace --locked --offline`: **570 unit/integration tests + 8 doctests passed**, no failures/ignored tests after correction `c9a4b33` |
| Focused storage/state | `cargo test -p moxie-state -p moxie-memory --locked --offline`: passed, including 12 paging/identity/allocation tests and two new host-buffer tests |
| Actual stored rows / allocation | `cargo test -p moxie-state --test paged_allocation --locked --offline -- --nocapture`: passed with the figures above |
| Architecture | `cargo xtask arch-check`: **61 rejecting + 16 accepted fixtures**, 12 rules; new state→memory accepted fixture and state→CUDA rejecting fixture |
| Format / lint | `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`; `git diff --check`: passed |
| Isolated host build | Fresh `CARGO_TARGET_DIR=target/task0013-host-isolated`, `CUDA_HOME=/nonexistent NVCC=/nonexistent PATH=/home/rodrigo/.cargo/bin:/usr/bin:/bin cargo build -p xtask --locked --offline`: passed; `ldd` reports no libcuda. Full host tests were run in the normal host target, not repeated in this isolated target. |
| Device-feature workspace | `cargo test --workspace --features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda --locked --offline`: **586 tests + 11 doctests passed**, zero failed/ignored after correction `c9a4b33`. Device clippy with the same feature list and `--all-targets -- -D warnings` also passed. |
| Real GPU regression | `cargo xtask-cuda test-gpu`: **39 passed, zero failed/skipped**, SM86 and SM120 qualified on the UUIDs below. PCI_BUS_ID ordering supplied by the Cargo configuration. |
| Specification | `cargo xtask spec-check`: all 10 normative documents present and unchanged. |

| Hardware | UUID |
|---|---|
| RTX 5060 Ti / SM120 | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` |
| RTX 3090 / SM86 | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` |
| RTX 3090 / SM86 | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` |

These GPU runs are regression evidence for accepted device behavior; they do not
turn the host paged store into device state. Separate per-SM/restricted-visibility
qualification and Compute Sanitizer were not repeated for this host-only change.

### First review correction, 2026-09-10

Independent review found one P2 issue and no paging corruption, transaction
isolation defect or competing resource owner. A valid 1,000-token geometry whose
8,008-byte lineage allocation failed returned `InvalidRequest("lineage")`. The
geometry was valid; the failed allocation was the admitted pageable control
resource, so callers could not distinguish exhaustion from malformed input.

Correction `c9a4b33` returns `CapacityExceeded` naming
`HostTier::Pageable`, the exact requested lineage bytes and zero allocator
availability, consistent with `HostBuffer`'s post-admission allocation failures.
The constructor still drops the partial `SequenceState`, releases the backing
through the admitting ledger, and only then returns the error.

`paged_allocation_failure` is a separate executable with a one-shot global
allocator fault aimed only at the 8,008-byte lineage request. It asserts the
exact error variant and fields, that the intended allocation failed once, and
that outstanding reservations, total host charge, `StateSpill` and `Pageable`
charges are all zero. It passes normally; returning the old `InvalidRequest`
would fail its exact error assertion. No contract, numerical threshold or
accepted test was weakened.

**Deliberately failing controls:** forcing both sequences' first transaction IDs
to 1 makes `transaction_ids_cannot_resolve_another_sequences_journal` fail (exit
101). Removing physical truncation from abort makes the real-path cancellation
test fail on backing bytes (exit 101). Both mutations were restored and the
focused suite passed again. These are test-sensitivity evidence, not unresolved
product failures. No acceptance test or numerical threshold was weakened.

**Unmeasured/skipped:** no new topology, sanitizer, checkpoint-quality or paired
prefill/decode benchmark. No CUDA code changed. No long-context attention,
checkpoint, COW branch, service or sampler support is established. O1–O7 remain
open and M1.4 remains active.

**Deletion:** removed the per-sequence transaction counter and eager error-message
allocation from successful state append. No production bridge or competing state
transaction mechanism was introduced. The existing dense interpreter cache stays
as the mathematical reference; production attention must consume shared paged
state when it is integrated.

### Owner acceptance, 2026-09-10

The owner accepted task 0013 after independent review through `0f39cf5`. The review
found no remaining code-review blockers, paging corruption, transaction-isolation
defect or competing resource owner. It confirmed that correction `c9a4b33` reports
the exact 8,008-byte pageable lineage allocation failure as `CapacityExceeded` and
releases the outstanding reservation and both tier charges before returning.

The reviewer ran the allocation regression against the previous implementation and
observed the intended failure specifically on `invalid_request` versus
`capacity_exceeded`; the corrected implementation passes. State/memory tests, the
regression with device features, formatting, host clippy, architecture and
specification checks passed independently. The retained full-workspace and aggregate
GPU logs and hashes matched; those full runs were not repeated during final review.

Task 0013 is closed. M1.4 remains active. The next bounded assignment is sampler
history and deterministic greedy/temperature distribution, with history and rollback
using task 0004's accepted transaction mechanism. Service/CLI and device attention
remain later scope.

### Raw evidence retention

Logs are at `/home/rodrigo/Developer/moxie/results/task0013/`, outside git. Retain
through task review and M1 closure, then retain the tracked conclusions/hashes.
`SHA256SUMS` lists every log's digest; after the first review correction its own
SHA-256 is
`c257a105312bc4b18309158444b570d964bc30731aa5e68542ebeb16fd45ae9b`.
The manifest covers `host`, `focused`, `allocation`, `arch`, `clippy`, `spec`,
`isolated-host`, `device`, `device-clippy`, `gpu`, `negative-identity` and
`negative-abort` `.log` files, plus correction host/device/clippy/architecture/
specification/GPU logs and `correction-allocation-failure.log`.

Build identities after correction `c9a4b33`: CUDA xtask
`target/debug/deps/xtask-f6925ac3b8aa5987`, SHA-256
`485f6af5132396e7fb63877e2c6836d6c975838a91dd2355acd9aa6425190096`;
host xtask `target/debug/deps/xtask-1b2a551c0290ffef`, SHA-256
`2a107fba9639d95dfcd368f94acaa4118e110d462ce4d31c637b181dc54cf74a`.
Build outputs are regenerable and may expire on clean; the source, task contract,
log hashes and result record are the retained evidence.
