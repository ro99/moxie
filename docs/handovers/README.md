# Handovers

## Active handover

[Task 0017, per-layer key/value geometry and window reclamation](2026-09-12-task0017-per-layer-kv-retention.md)
is the current continuation. **M1.4 complete; M1.5 active; M4 open.** The owner
accepted task 0016 on 2026-09-12, opening M1.5 with a reduced synthetic Gemma
graph, and selected M4's state schema as task 0017; that work is implemented and
awaiting owner review. Checkpoint execution (M3), device attention and the rest
of M4, and vision (M11) retain their gates.

This entry was stale until task 0017: it still named the M1.4 closure handover
after [the M1.5 Gemma operation gap](2026-09-12-m1.5-gemma-operation-gap.md) had
become the active one. Update it with the handover, not afterwards.

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md).

Name the writable root, branch/commit, dirty files and the legacy snapshot. Keep facts, hypotheses
and decisions in separate sections, name what failed as well as what worked, and end with one
bounded next task.
