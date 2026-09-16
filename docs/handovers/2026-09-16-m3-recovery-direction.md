# Handover — recover the M3 completion sequence

## Workspace identity

- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, inspected at
  `4ccb3b9` on 2026-09-16. Recheck before assigning work.
- Existing dirty work: `coordinator.md` is the replacement coordination workflow;
  `crates/moxie-repack/tests/publication.rs` and
  `docs/tasks/0031-m3-battery-closure.md` are the previous builder's unfinished
  package. Preserve and inspect all three. Do not discard or rebuild them.
- Legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e` stays read-only, as do `/models`
  and `/fast/models`.
- Herdr discovery found a fresh idle OMP session in `wB:p1P`, displaying
  **Muse Spark 1.3 Free**, and the diagnosing Codex session in `wB:p1M`.
  Previous builder/reviewer panels no longer exist. Rediscover before control.
- This is a recovery assignment, not a completion report. No new product tests
  were run by the diagnosing agent. Recheck background jobs and mutation markers
  before starting any battery; agent idleness alone proves neither is absent.

## Completed facts

The previous builder reported host 1,109 and device-feature 1,154 passing tests
for the publication regression candidate; the figures are in task 0031. The
previous reviewer reported publication 22/22 and fault enumeration 48/48,
self-test 109/109, and no product change in `run.rs` or `fault.rs`. The targeted
`published-validation-skipped` mutation was caught. These are inherited results,
not reruns by this session, and do not establish a full battery pass.

Task 0031 still owes full T0006 and T0028 plus record reconciliation. Task 0030
has contradictory current summaries despite its later review/handover. The
`xtask` CUDA build fails at `gpu.rs`'s borrowed label; the driver-feature suite
does not compile that lane. Tasks 0028/0029 have unresolved admission coverage.
No quantized expert path or checkpoint-backed model execution has been shown.

## Decisions

Owner instruction, 2026-09-16:

> instruct him to complete M3.1/M3.2/M3.3. I trust in your judgement about how to better execute this.
>
> Claude is out of credits, so we will use OMP as the builder (Muse Spark 1.3 Free).

Use the new `coordinator.md`. Coordinator manages a separate OMP/Muse builder
and independent Codex reviewer (`gpt-5.6-sol`, high). No Claude. Verify the actual
builder model after launch; do not silently substitute a different model.

The instruction authorizes pursuing completion and exercising engineering
judgment. It is not acceptance of unfinished work, a tolerance change, an abort
waiver, bulk conversion authority, or a ruling that repacking is retained.
Keep existing owner-reserved decisions explicit and batch concrete proposals.

Use the roadmap's five original items in the closure ledger. For the user's
three labels, show this mapping explicitly: M3.1 covers publication plus the
value-preservation obligation (items 1/4); M3.2 covers importers (item 2); M3.3
covers dense/expert execution plus independent-graph integration (items 3/5).
Do not let this shorthand hide any exit clause.

## Remaining hypotheses and blockers

- Publication validation appears sound; the regression repairs missing coverage.
  Amend task 0031's mistaken production-before/after premise with the evidence
  and coordinator authority. Require the production corruption case to pass and
  the validation-removal mutant to fail. Do not manufacture a product defect,
  weaken the gate or reclassify the mutant as an expected survivor.
- Restoring the CUDA helper through a fallible owned-label conversion is a
  concrete repair to evaluate separately from accepting the public API narrowing.
  Scope and test it; do not claim that repairing the helper settles compatibility.
- Task 0029 needs a complete inventory and sweeps of admission, rejection,
  relocation and release, plus the shared-core allocation problem. The `Rc`
  ownership/unsafe decision and label narrowing are explicitly owner-reserved.
  Prepare a small design comparison and recommendation. Do not treat trust in
  coordination as permission to accept aborts or silently narrow the API.
- ADR 0027 and the owner's existing task 0027 deferral remain in force. Close
  bounded publication correctness without reopening whole-model converter polish.
  Experiment 0007 remains pending until its actual model-execution trigger;
  it must not block building the execution it needs. Do not claim full M3.1
  acceptance if the remaining requirement still needs an owner disposition.
- Remaining importer continuations stay visible: group-128 symmetric INT4,
  AutoRound/AutoGPTQ packing, activation-order maps, expert interleaving,
  applicable scale/activation dtypes and unquantized/MTP overrides. Completion
  now needs explicit contracts for them; no bulk source acquisition is authorized.

## Next task and continuation

1. Inspect the existing publication diff, reconcile task 0031's premise and
   current ledger, and establish a stable candidate. Create the team, assign
   ownership, and start the full T0006 battery with log, job handle, exit status
   and source identity. Freeze its tree. Collect completion automatically.
   Address actual failures with focused regressions and independent review.
   Finish T0028 as task 0031 requires, serially, preserving its source identity;
   later shared-code changes require fresh affected regression evidence.
2. Close the bounded publication package and present its precise acceptance
   scope. Repair the CUDA consumer and close the admission obligation through a
   separate bounded contract, including the owner decisions above. Preserve the
   publication-before-expert sequencing ruling; obtain its release through the
   required correctness/acceptance disposition, not by silently skipping it.
3. Complete dense execution qualification and quantized expert execution using
   supported formats, the existing residency authority and shared operations.
   Cover real weights, constrained residency, cancellation/failure and independent
   numerical comparison on the declared hardware. No full-tensor dequantization
   shortcut or model-private runtime.
4. Complete the remaining M3 importer contracts and extend the value-preservation
   and two-independent-graph execution matrix. Refine ordering by actual shared
   dependencies. Do not make every deferred converter feature a prerequisite for
   an expert path that can use already-supported weights.
5. Reconcile all five roadmap items and exit clauses with current evidence;
   obtain owner-reserved acceptance, commit/push owned work and hand over a
   fetchable state. Any unfinished or deferred obligation remains explicit.

Use this handover as the initial milestone ledger location and add the concise
live table required by `coordinator.md`, linking task-level details. First
report: actual team, first assignment, actual running job or precise blocker,
and automatic next action. Continue through the authorized queue; do not stop
at a plan or after asking a worker to fill one cell. Do not promise an elapsed
M3 completion time from the duration of a single battery.
