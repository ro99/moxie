# Handovers

## Active handover

[Task 0020 implemented; task 0021 is M2 item 3](2026-09-12-task0020-weight-residency-authority.md)
is the current continuation. M2 item 2's residency authority exists and this
artifact's own expert bytes have been read through it; **nothing executes a
routed layer**, which is item 3.

The milestone context below remains in force.

[M1 complete (M1.5 closed); M2 active](2026-09-12-m1-closure-to-m2.md) is the
current continuation. The owner closed M1.5 on 2026-09-12 on the reduced Gemma
graph plus the accepted importer, with the remaining blockers recorded: the
Gemma 4 artifact is 32.7 GiB against a 24 GiB largest GPU, so executing it needs
M3 plus M2 or M5.

**M2 proceeds in roadmap order.** The owner designated
`/fast/models/google/gemma-4-26B-A4B-it` — BF16, 128 experts at top-k 8, already
on disk — so no download was needed and none was made. **Nothing executes a
checkpoint yet.**

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md).

Name the writable root, branch/commit, dirty files and the legacy snapshot. Keep facts, hypotheses
and decisions in separate sections, name what failed as well as what worked, and end with one
bounded next task.
