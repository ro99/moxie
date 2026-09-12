# ADR 0011 — Shared host-reference generation service

Status: accepted with task 0015 on 2026-09-11; M1.4 complete.

## Context and decision

M1.4 needs a shared generation service and diagnostic CLI after task 0014's
accepted paging/history/sampler composition. The qualified device chain has no
attention operation. The existing dense interpreter is a mathematical reference
with an independent persistent cache, unsuitable as the service's state owner.

Extend the interpreter dispatcher with a read-only history source backed by
`PagedSequence`. Temporary dense histories are bounded oracle scratch; all
persistent KV bytes remain in the existing memory-owned pool. `run_paged` uses
the caller's existing transaction, aborts it on failure and leaves successful
work open for the engine's commit. It does not call the older `KvCache` journal.

The paged facade retains at most one logits identity using `SequenceState`'s
existing registry. Its control expression gains the exact conservative node
bound committed in task 0015. The private interpreter output pairs immutable
logits with that handle; staging validates current identity and materialization.
Before the next forward, the engine clears old output and delegates invalidation
to the existing state generation mechanism. No parallel output identity scheme.

`moxie-engine` owns prefill/decode and a separate `service` module owns the typed
request/event facade. Document 02 explicitly permits fewer physical crates while
ownership is preserved. `moxie-cli` builds synthetic mathematical inputs and
formats service events; it cannot depend on state, sampler, interpreter or device
executor in production. It may depend on graph/oracle registration, the engine's
input types, host telemetry and memory admission for composition. Architecture
fixtures enforce these edges, including model/interpreter-to-engine refusal.

The only execution profile is explicitly `host-reference`, context <=256 and
bounded graph/tensor dimensions. This is a diagnostic choice, not an `auto`
fallback, production CPU backend, 32K attention claim or checkpoint integration.
The exact conservative scratch expression is in contract `2fbacec`. It reserves
CpuWorkspace without allocating an unused mirror. The prompt is admitted physical
storage; paged state retains its own existing reservation. Partial admission
failure releases earlier reservations. Borrowed graph/weight inputs are immutable
composition-root fixtures; building those inputs precedes generation admission.

Correction `5990f2a` makes that immutability an enforced paged-execution authority:
one graph and borrowed weight set claims an empty sequence before physical execution,
and only that authority can extend its KV history. Admission evaluates every distinct
row count the request will actually execute, including a partial prefill tail and
one-row decode. Reference payload and arithmetic scratch copies use fallible reserve
and report `CpuWorkspace` capacity exhaustion, so a post-admission forward failure
aborts the shared transaction and reaches the service's existing cleanup/restart path.
Correction `7a08adf` places transaction validation before those fallible copies and
encloses preparation plus interpretation in one abort-on-error path. A failure on a
later call in the same transaction therefore restores earlier tentative rows,
frontiers and logits before returning.

Pull delivery provides bounded backpressure: no event queue and no work until
the next pull. Only committed tokens are returned. Commit is the linearization
point; cancellation arriving after commit is observed on the next pull and cannot
retract that event. Finish, failure, cancellation and service drop close all
generation-owned storage before returning charges. Exclusive ledger borrowing
preserves release authority. The existing ledger retains fixed zero-valued tier
accounting nodes until its own drop; tests distinguish those from generation leaks.

## Alternatives and expiry

Using the old persistent dense cache would create a competing state participant.
Implementing fresh attention equations in the engine would bypass the accepted
oracle. Copying the GPU chain into a generator would imply an unqualified attention
path. None is adopted. The dense interpreter remains the independent reference;
the shared dispatcher is reused for both sources.

This slice adds no HTTP, tokenizer/text stop/EOS, chat, processor, checkpoint or
CUDA attention promise. The token-ID diagnostic CLI refuses unknown options.
Its deterministic `--cancel-after` exercises cancellation; interactive Ctrl-C
handling and text/protocol contracts remain M8. M1.5 consumes the common graph
integration; M4 adds paged device attention; M8 replaces diagnostics as the sole
CLI mode. O1–O7 remain open. The owner accepted task 0015 after final independent
review through `3f13784`; this decision closes M1.4 only within the explicit
synthetic host-reference boundary. M1.5 model integration remains separate.
