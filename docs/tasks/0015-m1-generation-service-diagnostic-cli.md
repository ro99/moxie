# Task 0015 — M1.4 shared generation service and diagnostic CLI

Status: **active**, 2026-09-11. Commit this contract before implementation.

## Identity and authority

- Owner: implementation agent; independent code review and owner acceptance follow.
- Writable `/home/rodrigo/Developer/moxie`, `main`, clean base `c0f61a7`.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; preserve its unrelated untracked files.
- Completes the remaining service/diagnostic-CLI deliverable of roadmap M1.4,
  building on accepted tasks 0003/0004/0012/0013/0014. R08/R20/R21/R22/R24.
- Authority: AGENTS, documents 01–09, owner-gate register, TASK/HANDOVER forms.
  Legacy `include/strata/app/openai_protocol.hpp`,
  `src/app/openai_protocol.cpp:343`, `docs/cli.md`, `docs/server.md`,
  `src/engine/sampling.cpp` and `tests/test_sampling.cpp` supply request, usage,
  cancellation and sampler context. No legacy generation loop is ported.
- O1–O7 stay open. No checkpoint/catalog/default/compatibility removal, network
  exposure, driver change, model-quality or product-performance conclusion.

## Bounded deliverable

One typed request/event service runs token-ID diagnostics through shared host
graph interpretation, admitted paged state and the accepted sampler. A minimal
CLI only parses, invokes the service and formats events. Two different synthetic
graph shapes exercise the same route. The execution profile is explicitly
`host-reference`; there is no automatic CPU fallback or synthetic model claim.

`moxie-engine` owns prefill/decode and the service facade (separate modules in the
same crate, allowed by document 02). `moxie-interp` owns graph evaluation and reads
the existing physical pages; it must reuse its operation dispatcher/oracles.
`moxie-state` owns the only transactions and output-provenance registry;
`moxie-memory` admits all new generation-owned backing and bounded interpreter
scratch. The CLI is a composition root for diagnostic graph metadata, never an
execution callback. No model-owned loop, second persistent KV cache or sampler.

Allowed: these shared crates, diagnostic CLI, manifests, architecture fixtures,
tests, ADR and living records. No device kernels/attention implementation,
checkpoint loading, tokenizer/chat/HTTP, full processors, speculation or entropy.
The old dense interpreter remains the independent migration/reference consumer;
the new service may not use its persistent dense cache. Diagnostic fixtures remain
explicit fixtures and expire as the sole CLI mode when M1.5/M8 add real integration.

## Contract before implementation

Requests name immutable graph/bindings, token and absolute-position input IDs,
prompt token IDs, maximum new tokens, chunk size, temperature and seed. Validate
nonempty prompt, vocabulary IDs, positive lengths, checked total context, exact
graph inputs/weights and sampler bounds before execution. Host-reference limits:
context <=256, graph values <=128, each bound/node tensor <=65,536 elements,
rank <=4, attention layers <=8 and equal BF16 MHA geometry across layers.
These are explicit diagnostic limits, not reduced production context targets.
Unsupported shapes/precision return typed refusal. No new mathematics: BF16
boundaries and FP32 attention/logits retain task 0003's equations and tolerances.

Prefill executes the actual requested chunks against paged history, preserving
absolute positions. Each forward uses the existing transaction; only fully
computed layer rows and logits are published. A private interpreter output binds
immutable logits to the existing sequence-issued handle. Sampling checks that
handle at the current materialized prefix. Output validity must fail for a
foreign sequence, replaced prefix, pending token or changed configuration. Retain
at most one forward result; clearing it is refused during an open transaction.

Generate exactly one token at a time through the task 0014 sampler, commit before
emission, and materialize the pending token before the next draw. A pull event
interface supplies admission/profile diagnostics, prefill progress, committed
token IDs, and a terminal event with usage and finish/cancellation/failure.
No unbounded queue or background execution. One service permits one active
generation; a second start returns a typed busy result. Token usage counts only
prompt and committed generated tokens. Max-length finish does not execute an
unneeded next token. Token-ID diagnostics do not advertise text stops or EOS.

Cancellation is checked before every graph operation and publication boundary,
as well as before prefill/decode. Abort the original journal on every failed step,
including a step partially writing physical rows or history. A committed event
cannot be retracted. Finish, cancellation, errors and service drop release all
generation reservations without needing another token; a second generation starts
empty. Service holds the admitting ledger exclusively, so release cannot target
a foreign ledger. No CUDA work or asynchronous leases are introduced.

Resource expression fixed here before allocator work: retain task 0014's paged
backing/control reserve, adding one conservative BTree node for the one live
logits result: `11*(size_of<(ResultId,LogitsHandle)> + size_of<usize>) +
16*size_of<usize>`. Copy prompt into `4*P` admitted state bytes. Reserve
`W = 1,048,576 + 64*S + 4096*N + 256*C*(D+16)*(L+1)` bytes of CpuWorkspace
control envelope for the allocating reference interpreter, where S is the sum
of graph value element counts evaluated at maximum chunk rows (including bound
weights), N is graph value count, C context capacity, D per-layer KV row width,
and L layer count. Checked arithmetic throughout. This deliberately conservative
envelope covers cloned bindings/values, staged rows, temporary dense oracle
history/head slices, logits and collection capacities; no extra persistent cache.
Fixed service/control metadata is included in the 1 MiB base. Borrowed immutable
graph/weight inputs are caller fixtures; their live storage is also conservatively
covered by W, but construction occurs before service admission. Generation-owned
payload is admitted before allocation. Count physical storage once; control
reservation does not allocate an unused W-byte mirror. A counting allocator must
check the bound at worst supported representative shapes and every release path.

## Acceptance

- Compare paged interpreter logits bit-for-bit with the existing dense reference
  for two graph shapes, whole/chunked prefill, partial final chunks, multiple
  pages and decode. Independent primitive oracle gates remain unchanged.
- Service and CLI fixed-seed outputs agree. Greedy and temperature requests,
  one/multiple tokens, invalid inputs, busy, pending output provenance and
  cancellation then a second generation. Terminal event exactly once; no events
  after termination. Disconnect/drop releases charges.
- Inject deterministic cancellation at operation/publication boundaries, including
  after writes; verify frontiers/history/rows and no escaped tentative tokens.
  Counting allocator: live heap below admitted envelope, repeated generations
  do not retain growth, all tier charges/reservations return to baseline. Exercise
  admission and post-admission allocation refusal. Preserve negative controls.
- Host and device-feature workspace tests/clippy, fmt, architecture, spec checks;
  real GPU regression on all three UUIDs (existing behavior, no GPU service claim).
  Architecture adds positive engine/CLI composition and negative CLI-to-state/
  sampler/executor, model-to-engine and interpreter-to-engine fixtures. Retained
  independent probe crates are evidence, not product dependencies: validate the
  tracked tree separately if they trigger the existing architecture scan.
- No new topology, sanitizer, checkpoint-quality or paired performance campaign:
  unmeasured, no throughput/default claim. 32K paging/history results remain storage
  evidence; the host-reference diagnostic context cap is explicitly 256.
- Update support matrix, ADR and handover; retain raw logs/hashes through review
  and M1 closure. Commit implementation and evidence for independent review.
- Stop for a second resource/transaction owner, opaque model execution callback,
  unbounded scratch, unverified output provenance, numerical-gate weakening or
  dependency on an unresolved owner ruling. M1.4 closure requires owner acceptance;
  M1.5 model graph/integration and M4 device attention remain separate.

## Result, filled after work

Pending implementation and verification.
