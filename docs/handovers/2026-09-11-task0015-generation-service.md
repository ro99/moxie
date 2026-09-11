# Handover — task 0015 generation service ready for review

## Workspace identity

Writable `/home/rodrigo/Developer/moxie`, `main`, clean task base `c0f61a7`.
Contract `2fbacec` precedes implementation `f7cff56`; final evidence is a separate
documentation commit. No unrelated source changes were present or overwritten.
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

Independently review `f7cff56` against `2fbacec`, ADR 0011 and the task's retained
gates. Corrections stay in task 0015. **M1.4 remains active until owner acceptance.**
After acceptance, record M1.4 closure and define the bounded M1.5 model graph/
integration assignment, inspecting exact local artifact revisions before any model
claim and stopping at dependent owner gates. Device attention remains M4 scope;
do not substitute synthetic diagnostic output for actual checkpoint integration.
