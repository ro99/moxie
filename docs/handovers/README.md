# Handovers

## Active handover

[Task 0018, compressed-tensors pack-quantized import](2026-09-12-task0018-compressed-tensors-import.md)
is the current continuation. **M1.4 complete; M1.5 active; M3 and M4 open.**
The owner **accepted task 0018 on 2026-09-12** within its import-only scope.
Task 0017, per-layer paged geometry and window reclamation, has had its review
corrections completed and pushed; the owner directed the next task rather than
stating a separate acceptance, so it is recorded as awaiting one.

**Nothing executes a checkpoint**: the W8A16 path, the repacker and the manifest
write are M3's, and vision is M11.

This entry was stale until task 0017: it still named the M1.4 closure handover
after [the M1.5 Gemma operation gap](2026-09-12-m1.5-gemma-operation-gap.md) had
become the active one. Update it with the handover, not afterwards.

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md).

Name the writable root, branch/commit, dirty files and the legacy snapshot. Keep facts, hypotheses
and decisions in separate sections, name what failed as well as what worked, and end with one
bounded next task.
