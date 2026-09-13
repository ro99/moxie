# Handovers

## Active handover

[Task 0022 implemented; M2 item 5's remainder is next](2026-09-13-task0022-laguna-and-second-consumer.md)
is the current continuation. M2 item 4 — Laguna's metadata and graph, a second
synthetic MoE consumer, and a restricted budget smaller than its working weights
— is implemented, corrected after three rounds of independent review (eight
findings, two P1, all reproduced and fixed, none disputed), and awaits a further
review and owner acceptance. The
deliverable is Laguna's **routed block**: its attention tower is not composable
from today's operation catalogue, `softplus` output gating and the yarn rotary
ramp are both named gaps, and that narrowing was written into the contract
before implementation rather than discovered during it.

[Task 0021 implemented; task 0022 is M2 item 4](2026-09-12-task0021-expert-execution-plans.md)
precedes it. M2 item 3 — CPU expert fallback and GPU grouped
candidate plans under one interface, with bounded queues and NUMA-aware host
placement — is implemented and awaits independent review and owner acceptance.
**One layer's routed expert block of the designated artifact has now been
executed**, over synthetic activations and a route the test writes: that is not
model support and no quality claim follows from it.

[Task 0020 accepted](2026-09-12-task0020-weight-residency-authority.md) precedes
it: M2 item 2's residency authority, accepted on 2026-09-12 after seven rounds
of independent review.

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
