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

## Milestone obligations (roadmap M5, original item numbers)

| Item | Outcome | State | Owner | Dependency | Evidence | Next action |
|---|---|---|---|---|---|---|
| M5.1 | Legal partition semantics for all implemented ops; sharded weights and state use the canonical format and common residency authority | **Active** | luna (build) / sol (review) / coordinator | none technical | [task 0053](../tasks/0053-m5-attention-partition-semantics.md) opened, 2026-09-22 | luna to start under `/ponytail:ponytail` |
| M5.2 | TP2 on the 3090 pair: column/row linears, head/KV ownership, global routing/vocabulary ops, collective ordering and failure handling | Not started | unassigned | M5.1 (partition semantics must exist before lowering to them) | none | queue after task 0053 accepted |
| M5.3 | PP with uneven stages, microbatch-aware prefill, mixed TP+PP stage groups | Not started | unassigned | M5.2 (needs a working TP lowering to compose a stage with) | none | queue after M5.2 |
| M5.4 | Bounded expert-owner partitioning: shared dispatch/transport/reduction, duplicate destinations, host experts, route unions | Not started | unassigned | M5.1 for `ExpertMlp`/`Combine`'s own partition rule (currently `NotDetermined` by the same code path task 0053 touches, deliberately out of task 0053's scope — see its non-goals) | none | not queued; needs its own task once M5.1's expert-owner semantics are written |
| M5.5 | Topology cost probes and a deterministic plan comparison tool | Not started | unassigned | M5.2/M5.3/M5.4 (needs real candidate plans to compare) | none | not queued |

M5.1 is the only implemented-op gap the codebase itself already flags:
`crates/moxie-graph/src/graph.rs`'s `OpParams::partition_rule()` assigns
`PartitionRule::NotDetermined` to `Attention`/`MlaAttention` with a comment
naming exactly this as "document 04's M5 work," and separately to
`ExpertMlp`/`Combine` naming that as M5.4's own expert-partitioning decision.
Task 0053 is scoped to the first of those two, matching the roadmap's own
"complete the shared contract before its adapters" ordering — M5.2's TP2
lowering needs attention's partition rule defined before it can lower
anything through it.

## Remaining hypotheses and blockers

None blocking task 0053. `RankId`/`RankContext` (`crates/moxie-types`,
`crates/moxie-executor`) already exist as a single-rank-per-device
abstraction from earlier milestones' resource-ledger work; whether that
abstraction extends to a rank *group* (TP2's two collaborating ranks) or
needs a new type is an open design question task 0053 does not have to
answer — it defines partition legality, not the group/collective machinery
that lowers to it (M5.2's scope). Document 04's TP section (`04-attention-
parallelism-and-speculation.md:49-57`) is the normative source for what
"legal" means here: row/column linear partition semantics (already decided,
task 0003), attention head ownership, GQA KV replication when fewer KV heads
than ranks, output reduction, and bias/residual applied exactly once.

## Next task

[Task 0053](../tasks/0053-m5-attention-partition-semantics.md): define
`Attention`/`MlaAttention`'s partition rule — head ownership across ranks,
GQA KV-head replication where ranks exceed KV heads, and where the output
reduction legally happens — replacing today's fail-closed `NotDetermined`.
Semantics only; no TP2 execution, no collective/NCCL plumbing, no rank-group
type, no `ExpertMlp`/`Combine` (M5.4). Start luna in a new Herdr tab under
`/ponytail:ponytail`; assign with the return protocol in coordinator.md §6.
