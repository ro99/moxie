# Handovers

## Active handover

[M1 complete (M1.5 closed); M2 active](2026-09-12-m1-closure-to-m2.md) is the
current continuation. The owner closed M1.5 on 2026-09-12 on the reduced Gemma
graph plus the accepted importer, with the remaining blockers recorded: the
Gemma 4 artifact is 32.7 GiB against a 24 GiB largest GPU, so executing it needs
M3 plus M2 or M5.

**Nothing executes a checkpoint.** M2's sequencing waits on a batched O1/O5
question about whether a BF16 MoE may be downloaded; no download may be made
before it is answered.

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md).

Name the writable root, branch/commit, dirty files and the legacy snapshot. Keep facts, hypotheses
and decisions in separate sections, name what failed as well as what worked, and end with one
bounded next task.
