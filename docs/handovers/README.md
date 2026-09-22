# Handovers

## Active handover

[M4 closed; M5 opened](2026-09-22-m4-closure-to-m5.md) is the current
continuation. M4 is accepted and complete (owner, 2026-09-22; tasks
0037–0052; see the [M4 closure ledger](2026-09-20-m4.2-mla-descriptors.md)).
Owner instruction, 2026-09-22: "open M5 for luna to work." The next bounded
deliverable is [task 0053](../tasks/0053-m5-attention-partition-semantics.md):
define `Attention`/`MlaAttention`'s partition rule — head ownership, GQA
KV-head replication, output reduction — replacing today's fail-closed
`NotDetermined`, the first slice of M5.1.

## Historical handovers

- [M3 accepted; begin M4](2026-09-19-m3-closure-to-m4.md)
- [M4 closure ledger (M4.2–M4 acceptance)](2026-09-20-m4.2-mla-descriptors.md)
- [M3 recovery direction](2026-09-16-m3-recovery-direction.md)
- [Task0030 repack correctness](2026-09-16-task0030-repack-correctness.md)
- [Task0029 fallible admission](2026-09-15-task0029-fallible-admission.md)
- [Task0028 shared quantized execution](2026-09-14-task0028-shared-quantized-execution.md)
- [Task0027 two-command repack](2026-09-14-task0027-two-command-repack.md)
- [Task0026 safetensors publication](2026-09-14-task0026-safetensors-publication.md)
- [Task0025 offline repack publication](2026-09-13-task0025-offline-repack-publication.md)
- [Task0024 asymmetric INT4 import](2026-09-13-task0024-asymmetric-int4-import.md)
- [Task0023 whole-working-set trace](2026-09-13-task0023-whole-working-set-trace.md)
- [M1 closure to M2](2026-09-12-m1-closure-to-m2.md)

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md). Name the writable root,
branch/commit, dirty files and legacy snapshot. Keep facts, hypotheses and
decisions separate, record failed/skipped work, and end with one bounded next
task.
