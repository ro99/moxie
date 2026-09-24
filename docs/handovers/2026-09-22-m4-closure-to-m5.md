# Handover — M4 closed, M5 opened (coordinator working record)

Status: **M5 open.** M4 is accepted and complete
([M4 closure ledger](2026-09-20-m4.2-mla-descriptors.md); AGENTS.md,
2026-09-22). Owner instruction, 2026-09-22: "open M5 for luna to work." This
is the coordinator's running ledger for M5 (coordinator.md §2).

## Workspace identity

- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base
  `178c10f` (Accept task 0052 — clean at ledger open, 2026-09-22).
- Unrelated carried dirty work, preserve and do not include in M5 commits:
  `docs/evidence/specification-version.md` (modified) and
  `docs/decisions/adr/0034-tokenizers-crate-named-at-m8.md` (untracked) — a
  separate session's M8.4 tokenizer-crate ADR amendment, already named and
  excused in task 0052's own identity section. Not this coordinator's to
  stage or revert.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. `/models` and `/fast/models`
  read-only checkpoint roots (ADR 0020).
- Team per coordinator.md's team table: builder Codex `luna`
  (`gpt-5.6-luna`, max effort, `/ponytail:ponytail`), independent reviewer
  Codex `sol` (`gpt-5.6-sol`, high effort, read-only,
  `/ponytail:ponytail-review`), coordinator Claude Opus
  (`/ponytail:ponytail-audit` at milestone end). No Herdr agents were running
  at ledger open; this session starts luna fresh in its own tab.

## Completed facts

M4's five roadmap sub-items and both closure tasks (0050, 0051) are accepted;
see the M4 ledger linked above for the full evidence table. Task 0052
(accepted, 2026-09-22) additionally unified the three hand-duplicated
bounded/explicit-restore state stores the M4 milestone-end audit found
(`accumulator.rs`/`convolution.rs`/`sparse_index.rs` in `crates/moxie-state`)
into one generic store, preserving every original test's assertions; two
real regressions surfaced and were repaired across two review rounds
(implicit lost `Clone`/`Copy` derive bounds; an unsealed generic trait
reachable from outside the crate). Nothing in this repository generates a
token from a checkpoint; M5 does not change that.

## Decisions

Owner instruction on 2026-09-22: "open M5 for luna to work." This authorizes
M5 (multi-GPU through common partitioning) and the first bounded task within
it, following the same pattern as the 2026-09-19 M3→M4 authorization. No
other M5 scope beyond the first task below is authorized by this instruction;
later M5 items get their own bounded tasks as each dependency clears.

Owner direction, 2026-09-22 ("go auto-mode until I say differently … keep
advancing … with the goal of completing M5"): until the owner revokes it, the
coordinator accepts reviewed tasks and opens the next slice without waiting
for owner approval. Each acceptance record says "coordinator, under the
owner's auto-mode delegation" rather than "owner". Owner-reserved matters not
settled by Strata or an existing ADR still go to the owner, per AGENTS.md.

## Milestone obligations (roadmap M5, original item numbers)

M5.1 is split into explicit obligations (coordinator.md §2). A post-acceptance
coordinator review of task 0053 (2026-09-22) found that the earlier statement
"attention was the only implemented-op gap M5.1 still has to close" was
wrong. Rows M5.1-c and M5.1-d record what it missed.

| Item | Outcome | State | Evidence |
|---|---|---|---|
| M5.1-a | Attention and MLA op-level partition rule | **Accepted** (owner, 2026-09-22) | [0053](../tasks/0053-m5-attention-partition-semantics.md) |
| M5.1-b | Column-sharded weights through the canonical format and residency authority | **Accepted** (owner, 2026-09-22) | [0054](../tasks/0054-m5-weight-shard-addressing.md) |
| M5.1-c | Head-aligned partition across the head chain | **Accepted** (in M5.2) | [0056](../tasks/0056-m5-host-tp-attention-lowering.md), [0058](../tasks/0058-m5-dense-tp-host-lowering.md): host TP bit-identical to the unsplit graph |
| M5.1-d | Per-weight MLA partition | **Accepted** 2026-09-23 | [0067](../tasks/0067-m5-mla-head-partition.md): bit-identical at R = 2 and 4 |
| M5.1-e | Trim task 0053's redundant tests | **Accepted** (owner, 2026-09-22) | [0055](../tasks/0055-m5-trim-task-0053-tests.md) |
| M5.1-f | Per-rank KV state through the residency authority | **Accepted** | [0060](../tasks/0060-m5-dense-tp2-on-the-pair.md): per-rank device KV, a whole rank step as one transaction, two-phase commit |
| M5.2 | TP2 on the 3090 pair | **Accepted** | [0057](../tasks/0057-m5-tp2-rank-group-and-gather.md), [0060](../tasks/0060-m5-dense-tp2-on-the-pair.md), [0061](../tasks/0061-m5-admit-host-control-metadata.md), [0062](../tasks/0062-m5-thread-per-rank-execution.md) (thread per rank; status rendezvous; rank-failure and collective-mismatch injection), [0063](../tasks/0063-m5-admit-sequence-state-growth.md) |
| M5.3 | PP with uneven stages, microbatch-aware prefill, TP2 + TP1 stage groups, capacity/latency report | **Accepted** 2026-09-24 | [0069](../tasks/0069-m5-host-pipeline-lowering.md), [0070](../tasks/0070-m5-solo-rank-worker.md), [0071](../tasks/0071-m5-pipeline-on-three-gpus.md), [0072](../tasks/0072-m5-combined-tp2-tp1-plan.md): at fixture scale, capacity improves and latency worsens about 5× |
| M5.4 | Bounded expert-owner partitioning: duplicate destinations, host experts, route unions | **Accepted** 2026-09-24 | [0064](../tasks/0064-m5-host-expert-owner-lowering.md), [0065](../tasks/0065-m5-routed-gemma-single-gpu.md), [0066](../tasks/0066-m5-expert-owner-tp2-on-the-pair.md), [0068](../tasks/0068-m5-host-owned-experts.md) |
| M5.5 | Topology cost probes and a deterministic plan comparison tool | **Accepted** 2026-09-24 | [0073](../tasks/0073-m5-topology-cost-probes.md), [0074](../tasks/0074-m5-plan-comparison-tool.md); `docs/evidence/topology-costs.md`, `docs/evidence/plan-comparison.md` |

Rows marked "Accepted" without "(owner)" were accepted by the coordinator
under the owner's auto-mode delegation. The history of each row is in its
task records and in the sections below.

Owner direction, 2026-09-23: a second builder starts slice 4 in parallel
with task 0062. MLA partitioning and M5.5 (topology probes and plan
comparison) are **not** deferred; they stay in M5. Slice 4 began with
[task 0064](../tasks/0064-m5-host-expert-owner-lowering.md), **accepted**
2026-09-23: a host expert-owner lowering of the routed block, bit-identical at
2 and 4 ranks, covering duplicate-owner, cross-owner and empty-owner routes.
[Task 0065](../tasks/0065-m5-routed-gemma-single-gpu.md), **accepted** 2026-09-23: the routed Gemma graph runs on one GPU (Route, ExpertMlp and Combine device kernels), matching the host on all three GPUs. It also fixed a latent softcap defect in task 0059's vocabulary kernel. [Task 0066](../tasks/0066-m5-expert-owner-tp2-on-the-pair.md), **accepted** 2026-09-23 (routed TP2 on the pair bit-identical; expert-owner resource and mid-step cancellation tests; the planner cross-checks every routed edge and sidecar pair): expert-owner TP2 on the pair, with resource and mid-step cancellation tests. Remaining slice-4 plan (coordinator, 2026-09-23; order swapped after the owner chose sequential work with luna): [task 0067](../tasks/0067-m5-mla-head-partition.md), MLA per-weight head partition on the host (M5.1-d), **accepted** 2026-09-23 (M5.1's last implemented op; bit-identical at R = 2 and 4); [task 0068](../tasks/0068-m5-host-owned-experts.md), host-owned experts (M5.4 "host experts"), **accepted** 2026-09-24: one GPU plus the host, with synchronous staging only, bit-identical to the grouped reference on all three GPUs. **Slice 4 is closed.** Ledger for the milestone-end ponytail audit: task 0068's size duplication, listed in its status block. That is one task more than first planned, because host experts were split out of task 0066. A scaled combine (Laguna 2.5) under expert-owner TP is **M7 work** (M7's Laguna row: "router softcap/scaling" and per-family topology legality). No M5 clause requires it, and task 0064 refuses it, fail-closed, until then. Corrected 2026-09-23: task 0064's amendment had carried it to a later slice-4 task. Earlier status: still open in slice 4: device execution of the expert-partitioned plan (the
exit gate's resource and cancellation tests), host-owned experts, a scaled
combine (Laguna 2.5), and MLA per-weight partitioning.

**GPU reservation (owner, 2026-09-23):** the owner is using the GPUs. No
agent runs GPU work (no `--features driver` device tests, no `xtask-cuda
test-gpu`, no CUDA benchmark) until the owner releases them. Builders and
the reviewer run host gates only. A task's GPU gates stay pending, and the
task is recorded as "code-complete, GPU gates pending" rather than
accepted. The coordinator batches the pending GPU runs and asks the owner to
release the GPUs, naming the devices, commands and expected duration.

**GPUs released (owner, 2026-09-23):** the reservation above is lifted.
GPU gates run again inside each task. Only one agent uses the GPUs at a time.

**Slice 5 plan (coordinator, 2026-09-24):**
- [task 0069](../tasks/0069-m5-host-pipeline-lowering.md): host pipeline lowering, with microbatched prefill in wavefront order; **accepted** 2026-09-24. Dense and routed Gemma through uneven 3-stage pipelines are bit-identical to the unsplit graph;
- **Re-scoped 2026-09-24 (coordinator), from three tasks to four.** Document 01 requires one execution thread per GPU, and the only worker is the pair-specific TP worker (task 0062). Generalizing it would reopen accepted failure-propagation code. So [task 0070](../tasks/0070-m5-solo-rank-worker.md) adds a single-GPU rank worker (**accepted** 2026-09-24 in 352332f; its size duplication with `dense_tp_workers.rs` goes to the milestone-end audit), and the rest shift by one: 0071 runs the pipeline on the three GPUs, and 0072 is the combined plan.
- [task 0071](../tasks/0071-m5-pipeline-on-three-gpus.md) (was 0070), **accepted** 2026-09-24: pipeline on the three GPUs. Dense Shape A and routed Shape C run on uneven stages (3090, 3090, 5060 Ti), within host tolerance; the 3090 pair is bit-identical to one 3090. All-or-nothing commit; stage-failure, cancellation and prepare-refusal tests; the lowering is re-derived and compared. Its size goes to the milestone-end audit. Every stage boundary is a declared, reported host-staged handoff: the single-GPU worker returns the stage output to the host. Revised 2026-09-24 from "the pair uses a peer handoff", because no M5 clause requires a peer PP handoff. A peer copy on the 3090 boundary is a performance option for M5.5's plan comparison. With resource and cancellation tests;
- [task 0072](../tasks/0072-m5-combined-tp2-tp1-plan.md) (was 0071), **accepted** 2026-09-24 (the TP2 pair runs layers 0–2 and the 5060 Ti runs layers 3–5 plus the head, within host tolerance; all-or-nothing commit; four fault and retry cases; the M5.3 report says capacity improves and latency worsens, about 5×, at fixture scale; the test size goes to the milestone-end audit; **slice 5 is closed**): the combined TP2 (3090 pair) + TP1 (5060 Ti) plan, with correctness, resource and cancellation tests, and the capacity and latency report.

Strata has no microbatch overlap: it uses a contiguous pipeline schedule on PCIe, and its wavefront design is documented as not implemented. The ordering rule therefore comes from document 04.

Slice 6 plan (coordinator, 2026-09-24):
- [task 0073](../tasks/0073-m5-topology-cost-probes.md), **accepted** 2026-09-24 (evidence in `docs/evidence/topology-costs.md`; the size goes to the milestone-end audit): topology cost probes. A pure `TopologyCosts` record in `moxie-plan`; a probe in `moxie-executor` measuring small-copy latency, bulk bandwidth and bandwidth under simultaneous traffic for host↔device and peer links, plus on-device memory bandwidth; `cargo xtask-cuda probe --costs`; and a measured evidence file. Strata has no bandwidth probe (only a boolean peer matrix). Pinned memory is left to M6.3, because no M5 path uses it.
- [task 0074](../tasks/0074-m5-plan-comparison-tool.md), **accepted** 2026-09-24 (evidence in `docs/evidence/plan-comparison.md`: the winner changes with the workload; the size goes to the milestone-end audit): the deterministic plan comparison tool. `xtask`'s arch-check allowlist gains `moxie-models`, as an optional dependency under `cuda` (a coordinator-approved edit). The model-import rule already names `xtask` a composition root. It ranks the candidates (single GPU, TP2, PP, TP2 + TP1, host experts) by the joint user workload, from the measured costs and each plan's bytes, and reports rejected alternatives with their reasons.
- The milestone-end `/ponytail:ponytail-audit`. The carried items: task 0068's size duplication, task 0069's dead check, the size overruns of tasks 0070–0074, and a sweep of task-numbered names in code (for example `xtask/src/gpu.rs`'s `task_0012_negative_fixtures`, and the `taskNNNN` labels and fixture ids in about 25 test files).
- The milestone-end `/ponytail:ponytail-audit` ran on 2026-09-24 and found about −330 to −430 lines, and no dependencies to remove. The owner chose to run a cleanup task before the package. [Task 0075](../tasks/0075-m5-audit-cleanup.md), **accepted** 2026-09-24. Change 6 (test deduplication) was skipped, because it moved the assertions and saved no lines. Change 8 fixed a latent TP-worker run-close defect. It covers:
  - one byte-sizing helper;
  - the plan comparison's decode total as a loop;
  - `Graph::producer`;
  - one paged-writer assembly;
  - the dead check in `lower_pipeline`;
  - the test duplication in `dense_gemma_device`;
  - removing task numbers from code names.
  Left out on purpose: `HostJoinExtents` (it is the consumer-side kernel check). The difference in run-close behaviour between the solo and TP workers went to review, which confirmed it as a defect; change 8 fixes it.
- The M5 acceptance package.

## Route to M5 closure (owner-agreed plan, 2026-09-22)

Accepted so far: tasks 0053–0057. This covers the attention partition rule,
column-shard weight ranges, the test trim, the host TP lowering of attention,
and a TP2 rank group with an all-gather on the 3090 pair. M5.1 is **not
closed**. Its leftovers are routed into the slices below. A slice may take
more than one task; the order holds.

| Order | Slice | Delivers | Closes |
|---|---|---|---|
| 1 | Dense TP on the host | Extend the task 0056 lowering to the MLP (`gate`/`up` column split, local GLU, input-axis-split `down_proj` with exact FP32 reduction per ADR 0036) and to the vocabulary projection (split and gather). Input-axis weights addressed as strided slices with group-alignment checks, as in Strata's `Dsv4RankShardDescriptor`. Correct or retire the partition labels the lowering does not use (`Rope`'s is inaccurate). Reduced Gemma 4 bit-identical to one rank. | M5.1 for dense ops; M5.2 input-axis split and vocabulary |
| 2 | Dense TP2 on the 3090 pair | Device RoPE with per-head norm (port of Strata `gemma4_norm_rope_kernel`, keyed by layout) and GLU kernels; FP32 partial-sum `Linear` with an exact all-reduce; slice 1's lowering wired to the device; per-rank device KV (M5.1-f); a whole rank step as one transaction. Reduced Gemma bit-identical to one GPU. **The fixture must be sensitive to summation order** (declared S ≠ 1 must change bits against S = 1), because task 0058's fixture is not, and so cannot catch a wrong combine order end to end. | M5.2 core; M5.1 state clause |
| 3 | Robust rank execution | Host control metadata admitted: `PagedAttentionRun.page_table` and `moxie-state` `DeviceKvSequence::page_view_for`/`placements_for` `Vec`s are allocated outside the host request (found in task 0059's round-3 review; they predate it, from task 0038). One thread per GPU (document 01); failure propagation with Strata's status collective, where a failing rank still enters every collective; rank-failure and mismatch injection at graph level; remove task 0057's duplicated test gate. | M5 exit fault injection |
| 4 | MoE and MLA partitioning | Expert ownership with dispatch, transport and combine in the declared expert order; duplicate destinations; host experts. MLA per-weight partition. Host first, then the pair. | M5.4; M5.1's last ops; M5 exit expert-partitioned plan |
| 5 | PP and combined TP + PP | Uneven stages; microbatch-aware prefill; a TP2 stage on the 3090s plus a TP1 stage on the 5060 Ti, with an explicit, reported host-staged handoff (no cross-socket peer access); report the capacity and latency effect. Strata's precedent: `src/engine/placement.cpp` assigns contiguous, byte-aware layer blocks per GPU (`assign_contiguous_blocks`, capacity-proportional cuts) and reports `cross_device_activation_hops` (corrected by the coordinator, 2026-09-23; an earlier version of this row said Strata never did PP). | M5.3; M5 exit combined plan |
| 6 | Probes, plan comparison, close | Topology cost probes; a deterministic plan-ranking tool; milestone-end `/ponytail:ponytail-audit`; acceptance package. | M5.5; M5 closure |

Slices 4 and 5 are independent and may swap. Rows M5.1-c/d/f and the M5.2
carried obligations above map into slices 1–4.

Lesson applied to every contract written from here on: read the storage layout,
group axis and graph wiring a contract depends on before writing its acceptance
clauses. Task 0053's shape-proof clause and task 0054's two amendments all came
from skipping that step.

## M5 acceptance package (coordinator, 2026-09-24): ready for the owner

Candidate: `main` at the commit that records this package (code at
`ad09f9b`). Every exit clause below was re-read on its own, against its
evidence.

| Exit clause (roadmap M5) | Evidence | Verdict |
|---|---|---|
| The same dense graph executes on a single GPU without model edits | [0059](../tasks/0059-m5-single-gpu-dense-gemma-device.md): reduced dense Gemma on both 3090s and the 5060 Ti within the task 0012 gate | met |
| … dense TP | [0060](../tasks/0060-m5-dense-tp2-on-the-pair.md), [0062](../tasks/0062-m5-thread-per-rank-execution.md): TP2 on the pair, bit-identical to a one-3090 reference under the same declared split | met |
| … dense PP | [0071](../tasks/0071-m5-pipeline-on-three-gpus.md): three stages on 3090, 3090 and 5060 Ti, within host tolerance; the 3090 pair bit-identical to one 3090 | met |
| The same MoE graph on a single GPU | [0065](../tasks/0065-m5-routed-gemma-single-gpu.md): routed Gemma on all three GPUs | met |
| … MoE TP (expert-partitioned) | [0066](../tasks/0066-m5-expert-owner-tp2-on-the-pair.md): expert-owner TP2, bit-identical | met |
| … MoE PP | [0071](../tasks/0071-m5-pipeline-on-three-gpus.md): routed Shape C on three stages | met |
| A combined TP/PP plan has correctness, resource and cancellation tests | [0072](../tasks/0072-m5-combined-tp2-tp1-plan.md): TP2 pair plus TP1 5060 Ti; four fault and retry cases (including cancellation); frontiers and stats unchanged; reservations returned | met |
| An expert-partitioned plan has correctness, resource and cancellation tests | [0066](../tasks/0066-m5-expert-owner-tp2-on-the-pair.md): expert-owner resource test and mid-step cancellation | met |
| Rank failure and collective mismatch are injected safely in the harness | [0062](../tasks/0062-m5-thread-per-rank-execution.md): rank failure mid-step, collective mismatch and stalls; lost is sticky; every rank drains | met |
| No TP3 speed guarantee | No TP3 plan is claimed. The 0072 report and `plan-comparison.md` state fixture-scale estimates only, with no speed claim | met |
| No hidden peer-to-host fallback | `moxie-cuda` refuses a peer copy without a grant ("refusing a host-staged copy"). Every host-staged transfer is declared: the pipeline handoffs and their bytes (0071, 0072), and host-expert staging (0068). The probe records a peer link only where CUDA grants it (0073) | met |
| M5.5 probes and a deterministic plan comparison, ranked by the joint workload | [0073](../tasks/0073-m5-topology-cost-probes.md), [0074](../tasks/0074-m5-plan-comparison-tool.md): the winner changes with prompt length; the output is byte-identical across runs | met |

Final gates at the candidate (task 0075): format, both clippy lanes, `cargo
test --workspace`, arch-check (79/21/13) and spec-check pass. On the GPUs,
`dense_gemma_device` is 6/6, `dense_tp2_device` passes and `cargo xtask-cuda
test-gpu` is 63/63 (SM86 and SM120).

**Milestone-end audit** (`/ponytail:ponytail-audit`, 2026-09-24): about −330
to −430 lines were found. [Task 0075](../tasks/0075-m5-audit-cleanup.md)
applied the behaviour-preserving subset (about −95 lines) and fixed one latent
defect the audit exposed: the TP worker dropped a paged-attention run whose
close was refused, unloading its CUDA module while a kernel might still run.
Still open, and not blocking the milestone:
- size overruns against the estimates of tasks 0070–0074 (`rank_worker.rs`,
  the executor's `pipeline.rs`, `topology_probe.rs`, `compare.rs`, and the
  device-test growth);
- the `dense_gemma_device` step-runner duplication (skipped, because it moved
  assertions).

**Scope and exclusions.**
- Every device result is on reduced fixtures. No checkpoint-backed model runs
  or generates a token in M5.
- There is no pipeline microbatch overlap: dispatch is sequential in
  wavefront order, and overlap is M6.3.
- There are no pinned transfers (M6.3) and no compute probe, so the plan
  comparison is weight-traffic and transfer bound, and prefill is
  underestimated. Automatic plan selection is M6.5.
- A TP stage may only come first in a pipeline, and a pipeline with a TP
  stage takes one microbatch.
- The Laguna scaled combine under expert-owner TP is M7 work, refused
  fail-closed until then.

**For the owner:**
1. **Accept M5**, within this scope. Only the owner can.
2. **Driver pin.** `docs/evidence/topology-p2p.md` pins the patched open
   module at 610.43.02. The machine now runs a locally built 610.57.04
   (2026-09-11), with peer access still granted. The evidence file should be
   corrected, and the pin re-stated.
3. The board's slices T2–T8 are in `review`, waiting for you to mark them
   `done`.

## Remaining hypotheses and blockers

None blocking. `RankId`/`RankContext` exist as a single-rank-per-device
abstraction; whether TP2 needs a rank-group type is M5.2's question. Document
04's TP section (lines 49–57) is the normative source for partition legality.

## Next task

None inside M5. The acceptance package above is ready for the owner. The first M6 task is opened only on the owner's authorization.

## Slice history (earlier next-task notes, kept for provenance)

Slice 1 ([task 0058](../tasks/0058-m5-dense-tp-host-lowering.md)) is **accepted**: the dense layer and vocabulary lowered on the host, with exact reductions under ADR 0036 and strided weight addressing. Slice 2 is split in two: [task 0059](../tasks/0059-m5-single-gpu-dense-gemma-device.md) runs the reduced dense Gemma graph on one GPU (the missing device kernels; RoPE with a host-computed angle table, following Strata), and task 0060 splits it across the pair, bit-identical to one GPU. **Task 0059 accepted**: the reduced dense Gemma graph runs on one GPU (both 3090s and the 5060 Ti) within the task 0012 gate, observed at 0 ULP on this fixture. Full GPU suite 63/63. [Task 0060](../tasks/0060-m5-dense-tp2-on-the-pair.md) is **accepted (2026-09-23), closing slice 2**. The reduced dense Gemma graph runs TP2 on the 3090 pair, bit-identical to a one-3090 reference that uses the same declared split; the fixture is sensitive to summation order both between and within blocks. Commit is two-phase (`DeviceKvSequence::prepare_commit`/`apply_commit`). One settle routine drains both ranks with a deadline, following a written reusable-versus-lost rule. Every failure site is inventoried in the task Result. Slice 3 (robust rank execution) is split: [task 0061](../tasks/0061-m5-admit-host-control-metadata.md) (**accepted** 2026-09-23) admits host control metadata and dedupes the test event gate; task 0063 (**accepted** 2026-09-23) admits `SequenceState` transaction-map and prefix-lineage growth on the step (split from task 0061 by amendment on 2026-09-23, following `PagedSequence`'s reserve-and-charge precedent); task 0062 then adds thread-per-rank execution. **Task 0062 accepted (2026-09-23), closing slice 3.** Each 3090 rank runs on its own thread (document 01); peer reads use an owned, acknowledged `PeerRead`; every collective is a status rendezvous with a deadline, where lost is sticky; and the single-thread path of task 0060 is deleted. **Owner direction, 2026-09-23: task 0062 continues and is slice 3's last task.** Anything its reviews surface that the M5 exit gate does not require goes to the ledger rather than into slice 3 (coordinator.md, "Own the pace and the scope"). Task 0062 adds failure propagation (Strata drove both ranks from one host thread through grouped collectives, but document 01 requires one rank execution thread per GPU). Owner direction, 2026-09-22: luna went in circles on failing tests, so the task was handed to the Claude Opus session `builder` (rescue), continuing from luna's uncommitted work (a backup patch was taken first). Luna is stood down for this task.
