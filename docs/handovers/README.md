# Handovers

## Active handover

[Task 0024 implemented: asymmetric INT4 import](2026-09-13-task0024-asymmetric-int4-import.md)
is the current continuation. M3 item 2's asymmetric half — compressed-tensors
`pack-quantized` **asymmetric INT4 at group 32**, whose zero points are packed
along the **output** axis while its codes are packed along the input axis — is
implemented on 2026-09-13 and **has had no independent review and is not
accepted**. Six modules of two real artifacts import with 112,640 reconstructed
values bitwise equal to the source's own arithmetic, and the zero-point lane
assignment is **measured** against the artifacts' own codes rather than taken
from the pinned library. **Import is not execution**: nothing consumes a
canonical INT4 tensor, and W4A16 is M3 item 3.

[Task 0023 accepted; M3 is authorized](2026-09-13-task0023-whole-working-set-trace.md)
precedes it, and is M2's last task. M2 item 5's remainder — byte and cost traces
reconciled with the resource ledger across a **whole working set** rather than
one layer — was accepted by the owner on 2026-09-13, after three rounds of
independent review (ten findings, four P1, all reproduced and fixed, none
disputed). **The owner authorized M3 on 2026-09-13**, so the next task comes
from M3 — canonical INT4/INT8/BF16 with quality separation. Its two owner gates
are already answered: O2 is repack-only (ADR 0018) and O5 is user-managed
storage and conversion (ADR 0020), so **no agent-initiated bulk download, copy
or conversion** may start without a task naming artifact, revision, expected
size and retention.

[Task 0022 accepted](2026-09-13-task0022-laguna-and-second-consumer.md) precedes
it. M2 item 4 — Laguna's metadata and graph, a second
synthetic MoE consumer, and a restricted budget smaller than its working weights
— was accepted by the owner on 2026-09-13, after three rounds of independent
review (eight findings, two P1, all reproduced and fixed, none disputed; the
third recommended acceptance with no new blocking findings). The
deliverable is Laguna's **routed block**: its attention tower is not composable
from today's operation catalogue, `softplus` output gating and the yarn rotary
ramp are both named gaps, and that narrowing was written into the contract
before implementation rather than discovered during it.

[Task 0021 accepted](2026-09-12-task0021-expert-execution-plans.md)
precedes it. M2 item 3 — CPU expert fallback and GPU grouped
candidate plans under one interface, with bounded queues and NUMA-aware host
placement — was accepted by the owner on 2026-09-13, after four rounds of
independent review.
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

**M2's five items are all accepted** — items 1 and 2 on 2026-09-12, items 3, 4
and 5 on 2026-09-13 — **and the owner authorized M3 on 2026-09-13.** M2 proceeded
in roadmap order and M3 does the same. The owner designated
`/fast/models/google/gemma-4-26B-A4B-it` — BF16, 128 experts at top-k 8, already
on disk — so no download was needed and none was made. **Nothing executes a
checkpoint yet.**

One file per bounded continuation, `YYYY-MM-DD-slug.md`. Use
[HANDOVER.md](../spec/templates/HANDOVER.md).

Name the writable root, branch/commit, dirty files and the legacy snapshot. Keep facts, hypotheses
and decisions in separate sections, name what failed as well as what worked, and end with one
bounded next task.
