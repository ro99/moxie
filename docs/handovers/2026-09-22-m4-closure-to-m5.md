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

M5.1 is split into explicit obligations (coordinator.md §2). A post-acceptance
coordinator review of task 0053 (2026-09-22) found that the earlier statement
"attention was the only implemented-op gap M5.1 still has to close" was
wrong. Rows M5.1-c and M5.1-d record what it missed.

| Item | Outcome | State | Owner | Dependency | Evidence | Next action |
|---|---|---|---|---|---|---|
| M5.1-a | Attention/MLA op-level partition rule | **Accepted** (owner, 2026-09-22) | closed | none | [Task 0053](../tasks/0053-m5-attention-partition-semantics.md): `Attention` → `HeadShardable{GqaReplicateWhenOversubscribed, ConcatenateHeads}`, `MlaAttention` → `HeadShardable{SharedLatentReplicated, GlobalReduction}`. The semantics are correct. Most of its new test lines check fixture literals, re-test descriptor methods already pinned in `moxie-graph`, or check the fixture's own wiring (see M5.1-e) | none |
| M5.1-b | Column-sharded weights addressed through the canonical format and residency authority (`LogicalRange` per component) | **Accepted** (owner, 2026-09-22) | closed | none | [Task 0054](../tasks/0054-m5-weight-shard-addressing.md), amended twice before any code: input-axis (row) shards are strided in row-major `[out, in]`, and quantization groups lie along `k`, so a column shard never splits a group. Three review rounds: a rank-dependent overflow verdict (fixed by validating the full extent first), hand-derived component widths (now `affine_components`), then redundant re-validation removed | none |
| M5.1-c | Head-aligned partition: `Rope` rotates pairs within a head (`HalfSplit` pairs j and j + head_dim/2), and the Q/K/V `Linear`s produce heads, yet both are plain `ColumnShardable` with no head-boundary requirement. Under GQA oversubscription (8 query heads, 2 KV heads, 4 ranks), an even column split of `k_proj` cuts each KV head in half | **Queued** | luna / sol / coordinator | M5.1-b accepted (both touch partition consumers; serialize) | coordinator review of task 0053, 2026-09-22. The rules come from task 0003 and were not visited by task 0053 | open as a task after M5.1-b and M5.1-e; the contract must first check how the graph ties a projection to the attention op that consumes it |
| M5.1-d | Per-weight partition inside `MlaAttention`: `q_a_proj`/`kv_a_proj` replicated, `q_b_proj`/`kv_b_proj` split by head, `o_proj` split along its input axis. One `PartitionRule` per op cannot express this, so M5.1-b's addressing cannot serve MLA weights | **Queued** | luna / sol / coordinator | M5.1-c (same head-alignment vocabulary); `o_proj`'s input-axis split also needs M5.2's strided addressing | coordinator review of task 0053, 2026-09-22 | fold into the M5.1-c task if the same mechanism covers it; otherwise its own task |
| M5.1-e | Trim task 0053's redundant tests to one assertion per invariant (owner instruction, 2026-09-19) | **Queued** | luna / sol / coordinator | M5.1-b accepted (luna is busy; same test files) | Keep: the contract-table enum rows and the single `node.contract.partition` assertion on the real MLA node in `mla_reference.rs`. Remove: the GQA-literal block and the `cache_width`/`more_query_heads` block in `reference_graphs.rs`, the fixture-wiring assertions in `mla_reference.rs`, and the new `HeadShardable` line in `moxie-graph`'s `partition_semantics_fail_closed_until_defined` | next task for luna after M5.1-b; small, test-only |
| M5.1-f | State (KV) shard addressing through the residency authority | Not started | unassigned | M5.1-c (which KV heads a rank owns) | named in task 0054's non-goals | after M5.1-c |
| M5.2 | TP2 on the 3090 pair: column/row linears, head/KV ownership, global routing/vocabulary ops, collective ordering and failure handling | Not started | unassigned | M5.1 | none | Carries from M5.1: reject non-divisible head/rank combinations (review-task-0053-round-1); input-axis (row-sharded) weight addressing and quantization-group alignment (task 0054 amendments), via a strided-range primitive or an ADR for a pre-sharded layout |
| M5.3 | PP with uneven stages, microbatch-aware prefill, mixed TP+PP stage groups | Not started | unassigned | M5.2 | none | after M5.2 |
| M5.4 | Bounded expert-owner partitioning: shared dispatch/transport/reduction, duplicate destinations, host experts, route unions | Not started | unassigned | `ExpertMlp`/`Combine` partition rule (still `NotDetermined`) | none | its own task once expert-owner semantics are written |
| M5.5 | Topology cost probes and a deterministic plan comparison tool | Not started | unassigned | M5.2/M5.3/M5.4 | none | not queued |

## Work queue (critical path)

1. Task 0054 (M5.1-b): accepted.
2. M5.1-e test trim (task 0055): small and test-only, cleared before the same files change again.
3. M5.1-c head alignment, with M5.1-d if one mechanism covers both. This is the
   substantive remaining M5.1 work; M5.2 cannot lower a real attention block
   without it.
4. M5.1-f state addressing, then M5.2.

Lesson applied to every contract written from here on: read the storage layout,
group axis and graph wiring a contract depends on before writing its acceptance
clauses. Task 0053's shape-proof clause and task 0054's two amendments all came
from skipping that step.

## Remaining hypotheses and blockers

None blocking. `RankId`/`RankContext` exist as a single-rank-per-device
abstraction; whether TP2 needs a rank-group type is M5.2's question. Document
04's TP section (lines 49–57) is the normative source for partition legality.

## Next task

Task 0054 is active. When it is accepted, the M5.1-e test trim goes to luna, then
M5.1-c. Each gets a full contract under `docs/tasks/` when it opens.
