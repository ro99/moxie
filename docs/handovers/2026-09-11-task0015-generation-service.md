# Handover — task 0015 generation service accepted

Superseded for continuation by the
[M1.4 closure to M1.5 handover](2026-09-11-m1.4-closure-to-m1.5.md). The task 0015
acceptance and evidence below remain historical records.

## Workspace identity

Writable `/home/rodrigo/Developer/moxie`, `main`, clean task base `c0f61a7`.
Contract `2fbacec` precedes implementation `f7cff56`; final evidence is a separate
documentation commit. Independent-review correction `5990f2a` makes generation
execution fail closed; correction `7a08adf` closes the remaining direct-API
transaction rollback gap. No unrelated source changes were present or overwritten.
Read-only legacy `/home/rodrigo/Developer/strata` remains at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, with `.pi/` and `tests/p2p/` untouched.

## Completed facts

[Task 0015](../tasks/0015-m1-generation-service-diagnostic-cli.md) implements the
remaining M1.4 service/diagnostic-CLI deliverable. `moxie-engine` orchestrates one
generation; its separate `service` module accepts typed requests and returns a
pull stream with diagnostics, prefill progress, committed token IDs and terminal
usage/status. Busy requests are refused. There is no queue or client token loop.

The host interpreter now reads physical paged history through its existing oracle
dispatcher. It writes all layers and records immutable logits under the caller's
existing state transaction. The paged facade uses the existing result registry,
retaining at most one handle; stale/foreign/pending outputs cannot stage tokens.
No persistent dense cache, duplicate journal or new sampling equations enter the
service. `moxie-memory` owns prompt/paging allocations and admits conservative
reference scratch. Finish, failure, cancellation and service drop release storage
before charges without requiring a next token. See [ADR 0011](../decisions/adr/0011-host-reference-generation-service.md).

Two synthetic shapes use the same service and match the older dense interpreter
bit-for-bit for prefill/decode. Whole/chunked/tail/page cases and seeded CLI/service
parity pass. Cancellation at all 117 service boundaries permits restart; direct
forward cancellation also checks retained rows, frontiers and lineage. Negative
controls detect omitted abort and reuse of an aborted output at the same numerical
prefix. Allocation tests cover the diagnostic context limit, 1,000 cancellation/
restart cycles, admission refusal and three targeted physical allocation failures.
Full gate commands, counts, failure history and retained hashes are in the task.

Independent review found three reproducible gaps. Correction `5990f2a` converts
the 262,144-byte weight-payload clone and subsequent reference arithmetic scratch
to fallible `CpuWorkspace` allocations; failure aborts, closes all reservations and
allows a new request. Admission now validates each distinct first/tail/decode row
count before charging. A `PagedExecution` holds immutable graph/weight borrows and
the sequence-issued execution authority, so a second configuration cannot extend
existing KV. Regressions cover all three behaviors.

Independent re-review found one remaining boundary: weight/input preparation in
`PagedExecution::run` still preceded the interpreter's abort handler. Correction
`7a08adf` validates transaction authority first, then runs configuration validation,
fallible preparation and interpretation under one abort guard. Its direct allocator
regression first publishes one tentative row and logits, fails the next 262,144-byte
weight copy in the same transaction, and verifies zero rows, baseline frontiers, no
live logits or journal, and a successful two-row retry. Final gates pass with 593+9
host and 609+12 device-feature tests, 39/39 GPU cases, both clippy lanes, formatting,
specification, and 71 rejecting/20 accepted architecture fixtures.

Final independent review through `3f13784` found no remaining blocking or actionable
defect. It independently passed the same host, device-feature, GPU, architecture,
clippy, formatting and specification gates, and additionally verified that foreign
transaction IDs leave local work untouched. The owner accepted task 0015 and closed
M1.4 on 2026-09-11. The evidence manifest and exact accepted boundary are recorded
in the task result.

## Decisions and limits

The profile is explicitly `host-reference`: requested context <=256, graph values
<=128, per-tensor elements <=65,536, rank <=4, uniform BF16 MHA with <=8 layers.
The service allocates bounded reference scratch per step; it does not claim zero
token-path allocations. The widest measured case materializes 255 prompt rows and
commits one pending sampled token. The conservative workspace reservation exceeds
measured peak; this is a correctness diagnostic, not a performance optimization.

No released model, checkpoint, tokenizer/text stop/EOS, HTTP, GPU attention/sampling,
full sampler processors, speculation or entropy integration is claimed. The CLI's
`--cancel-after` is diagnostic cancellation, not M8 interactive signal handling.
32K paging/history evidence remains storage-only. No new checkpoint-quality,
topology, sanitizer or paired throughput result is claimed. O1–O7 are unchanged.

The retained independent task 0014 probe crate still makes a local architecture
scan report four undeclared/dependency findings. It is preserved unchanged. The
tracked implementation is checked in a clean `git archive`; no allowlist exception
was introduced for ignored evidence.

## Next task

Task 0015 is closed. Continue from the
[M1.4 closure handover](2026-09-11-m1.4-closure-to-m1.5.md): author the first bounded
M1.5 model graph/integration contract only after exact local artifact inspection.
Device attention remains M4 scope; do not substitute synthetic diagnostic output
for actual checkpoint integration.
