# Task 0015 — M1.4 shared generation service and diagnostic CLI

Status: **implementation and validation complete; independent review and owner
acceptance pending**, 2026-09-11. Contract `2fbacec` preceded implementation `f7cff56`.

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

Implementation `f7cff56`, with architecture-fixture correction `3c346e6`.
[ADR 0011](../decisions/adr/0011-host-reference-generation-service.md) records the
ownership and reference-profile decisions. No task requirement or numerical gate
was relaxed. M1.4 remains active pending review and owner acceptance.

### Shared owners and behavior

- `moxie-engine`: validated immutable graph/input descriptors, exact bounded
  reference scratch admission, prompt ownership, chunked prefill and one-token
  decode through existing paging/history transactions. Its `service` module
  provides typed busy/refusal, pull events and cleanup before terminal delivery
  and on drop. No HTTP or client-private inference path.
- `moxie-interp`: one operation dispatcher for dense-reference and paged-history
  sources. Ephemeral oracle history reads the existing BF16 pages; all physical
  appends and output recording use the caller's transaction. `PagedOutput` keeps
  its logits private and checks the existing live result identity before sampling.
- `moxie-state`: one bounded live logits registry entry, a corresponding control
  node reserve, read-only transaction validation and existing generation invalidation.
  Abort restores the same rows, frontiers, lineage and sampler state.
- `moxie-cli`: synthetic graph composition, bounded argument validation, host
  measurement and event presentation. Admission diagnostics name the explicit
  host-reference profile, sampler, temperature, seed, context and reserved bytes.
  Unknown/duplicate fields are refused. No text tokenizer or model execution claim.

### Validation and resource evidence

Runtime source `f7cff56`; `3c346e6` only renames the undeclared-owner fixture so
its name remains outside the now-extended allowlist. Full runtime lanes therefore
retain `f7cff56` identities; final architecture checks use `3c346e6`.

| Gate / exact command | Result |
|---|---|
| `cargo test --workspace --locked --offline` | 591 unit/integration tests + 9 doctests, zero failed/ignored |
| Same command with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda` | 607 tests + 12 doctests, zero failed/ignored |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | passed |
| Same clippy command with the device feature list above | passed |
| `cargo fmt --all -- --check`; `cargo xtask spec-check`; `git diff --check` | passed; all ten specification digests unchanged |
| `cargo xtask-cuda test-gpu` | 39/39 real GPU cases passed; no skipped cases |
| `cargo xtask arch-check` on a clean archive of `3c346e6` | 71 rejecting + 20 accepted fixtures, 12 rules |
| `cargo test -p moxie-cli --locked --offline --test generation` | six tests; two shapes, exact dense/paged prefill and decode logits, whole/chunked/tail/page parity, seeded CLI/service agreement, provenance, busy/refusal, failure/disconnect and cancellation |
| `cargo test -p moxie-cli --locked --offline --test allocation -- --nocapture` | one isolated counting/failing allocator test; peak bound, 1,000 cancellation/restart cycles without retained growth, complete admission/allocation failure cleanup |

The two primary shapes are heads/dimension/vocabulary/layers `(2,4,16,1)` and
`(3,4,7,2)`. Prompt length 19 uses chunk sizes 1/3/7/8/19/32 and temperatures
0/0.25/1/10 in service parity tests; paged/dense interpreter comparisons use
1/3/7/19 plus pending-token decode. Cancellation tests traverse all 117 observed
service boundaries, verify committed-event usage and restart after every stop.
Direct forward cancellation checks existing row bytes, frontiers, lineage and
absence of an open transaction or live result. Replaying to the same numerical
prefix does not authorize reuse of an aborted output.

| Heads × dimension / layers | Prompt / chunk / max output | Peak requested heap above baseline | Admitted generation bytes |
|---|---|---:|---:|
| 2 × 4 / 1 | 37 / 13 / 1 | 24,591 | 1,617,382 |
| 3 × 4 / 2 | 251 / 65 / 1 | 139,317 | 6,946,822 |
| 4 × 64 / 8 | 255 / 255 / 1 | 14,557,322 | 280,786,774 |

The last case requests capacity 256, materializes 255 prompt rows and commits one
pending output token. It does not execute 32K attention. The conservative reference
envelope is not a measured minimal allocation plan or performance result. The
reference interpreter still allocates per step; zero per-token allocations are
claimed only by the accepted paging/history storage gates, not this service.
The immutable fixture inputs exist before admission; their construction is not a
checkpoint-loading path. The existing ledger retains bounded zero-valued tier
bookkeeping until ledger drop; generation allocations and charges are released.

Injected physical failures for a valid 193-prompt/3-output request target the
772-byte prompt payload, 1,576-byte lineage and 19,256-byte combined paged/sampler
buffer. They return the exact `CapacityExceeded` fields and leave no outstanding
reservation or StateSpill/CpuWorkspace/Pageable charge. An inadmissible 1 KiB budget
also leaves no charges. Existing 32,768-row/history allocation regressions pass
with the additional live-result control reserve.

### Failures, corrections and retained evidence

- Deliberately omitting paged-forward abort makes the transaction/row restoration
  test fail. Deliberately bypassing output-identity validation makes the same-prefix
  replay test accept a stale result and fail. Both mutations were restored.
- The first allocation harness compared a live ledger to its pre-first-admission
  heap baseline and caught its retained zero-valued tier map. The corrected harness
  verifies full cleanup after ledger drop and separately verifies no generation
  growth after initializing those fixed bookkeeping nodes.
- An initial size-only fault at 148 bytes hit an infallible allocation instead of
  the intended physical prompt payload and aborted the test process. The qualified
  failure geometry uses distinct larger extents and checks exact typed error fields.
  Both failed harness runs are retained; they are not passing cleanup evidence.
- Self-review found that a shared static default cancellation flag could be
  mutated through its reference. The implementation uses an object-local default
  flag; a regression proves unrelated cancellation tokens remain unaffected.
- The architecture fixture for an undeclared owner previously used `moxie-engine`.
  Registering the real engine made that old fixture pass unexpectedly. `3c346e6`
  renames it `moxie-unregistered-owner`, preserving its dependency and expected rule.
- Local architecture scans also find four pre-existing findings from the retained
  task 0014 independent probe crate under `results/`. That evidence is unchanged.
  Final product architecture validation uses `git archive 3c346e6` and
  `cargo run --manifest-path <archive>/xtask/Cargo.toml --locked --offline -- arch-check`.
  No architecture exemption or allowlist relaxation was made for ignored evidence.

Raw logs are retained outside git in `results/task0015/` through independent review
and M1 closure. `host-final.log`, `device.log`, `gpu.log`, `clippy.log`,
`device-clippy.log`, `arch-tracked-final.log`, `allocation-source-final.log` and
`generation-qualified.log` are the principal final evidence. Earlier logs retain
negative controls, harness failures and pre-correction architecture results.
The final manifest hash is recorded below after all evidence is sealed.

Final evidence SHA-256 values:

```text
host-final.log              c4d63e57d6523e904dbaf4bb960ff967efb03a9824e46715378d296df9f562d7
device.log                  816a5a74419fa6e53d1fd6bf00b02da29aa870caef5cd7af6a7527cd772c0015
gpu.log                     debfa6b0a6bd2f53a86b1953c22122ff20c92509c7ee84977c8426c55bbea1ea
clippy.log                  7c61d86dda4036b10abccf536dd8cd48e9b98b0740521b1dfcf721387f206e59
device-clippy.log           9221fa4e91748acfc97bc6a9c9021a3987247cc59765d6875e39da669c34deaa
arch-tracked-final.log      6b66f678e9126e57071371c6b4b1ea330311ff8521a6ee05cb0b6b9067506e0b
allocation-source-final.log d3128155c3c3fcefbc85b7346b6ffdfa4e0c9f618ccf903c05b3279440f87d49
generation-qualified.log    509bf7dbf5d6a043a3bddba04b2c0b871c6df355d6354b67e3b11e02448e09d8
```

The logs are retained outside git under `results/task0015/`; these hashes identify
the exact local review inputs without making transient build output part of the
living record.

### Deletion, remaining scope and next task

No legacy source, accepted oracle, sampler stage or public compatibility surface
was removed. No duplicate generation/cache path was introduced. The old dense
interpreter remains a reference consumer, sharing arithmetic with the new path.

This implements the remaining bounded M1.4 service/CLI deliverable. Independent
review and owner acceptance precede recording milestone closure. M1.5 graph/model
integration and M4 device attention remain separate; M8 retains production HTTP,
chat CLI, tokenizer/text stop/EOS and full sampler processors. No new topology,
sanitizer, checkpoint-quality or paired prefill/decode performance measurement was
required or claimed. O1–O7 remain open.
