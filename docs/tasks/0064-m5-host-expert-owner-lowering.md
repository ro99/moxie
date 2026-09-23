# Task 0064 — host expert-owner partitioning of the routed MoE block

Status: **accepted** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Started by Codex `luna2` (paused, then closed); completed by
Codex `luna` from the coordinator's numbered design with three amendments.
Sol accepted on round 1 with no findings. The coordinator re-ran `fmt`,
workspace `clippy`, `arch-check`, `spec-check` and the host TP tests.

**Amendment, 2026-09-23 (coordinator).**
- *Old premise:* change 4 adds `scale: f32` to `Join::Reduce`.
- *Evidence (builder):* that breaks exhaustive destructuring in
  `moxie-executor/src/dense_tp.rs`, which task 0062 owns.
- *Replacement:* `Join::Reduce` is unchanged. The lowering refuses, with a
  typed error, any routed `Combine` whose `output_scale != 1.0`, and the
  refusal table gains that row. Gemma uses 1.0.
- *Carried to a later slice-4 task,* after task 0062 lands: a scaled combine
  (Laguna 2.5) applied once after the cross-rank sum.
- *Consequence:* the "`output_scale` per partial" mutation is dropped.
- *Authority:* coordinator.

## Identity and authority

- Task0064, M5 plan slice 4, first task. It runs in parallel with task 0062
  (owner, 2026-09-23: "Second builder … YES").
- Builder: a second Codex session named `luna2` (`gpt-6-luna`, max,
  `/ponytail:ponytail`). Reviewer: Codex `sol` (read-only,
  `/ponytail:ponytail-review`). Coordinator: Claude Opus `coordinator`.
  Accepted by the coordinator under the owner's auto-mode delegation.
- The **design below is the coordinator's**. The builder implements the
  numbered changes exactly and reports any conflict with the code as a
  `DECISION`, rather than redesigning (coordinator.md, "Match the assignment
  to the worker").
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `7e902b3`.
- **Parallel work.** Task 0062 is changing `moxie-executor` and `moxie-cuda`
  on the same tree. Touch **only** the files listed under "Allowed files".
  Never stage, edit or revert anything else. Preserve the carried
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035.
- **Exit-gate clause served:** M5 exit, "an expert-partitioned plan [has]
  correctness, resource and cancellation tests". Also roadmap M5.4
  ("bounded expert-owner partitioning … duplicate destinations … varying
  route unions") and M5.1 (`ExpertMlp`/`Combine` partition semantics).
- [ADR 0036](../decisions/adr/0036-tp-reductions-are-exact-by-declared-order.md)
  applies: an exact reduction in declared order, bit-identity, no tolerance.

## Facts (coordinator, 2026-09-23)

- **The routed block** (`moxie-models/src/gemma4.rs`, around lines 940–1010):
  `Route` produces the route table. `ExpertMlp { hidden, intermediate,
  experts, top_k, activation }` takes inputs
  `(routed_input, route, gate_up, down)` and returns `[rows*top_k, hidden]`
  slot rows, each rounded to BF16. `Combine { hidden, top_k, order:
  AscendingExpertId, output_scale }` takes `(route, slots)`.
- **Weight layout is expert-first:** `experts_gate_up` and `experts_down`
  have shape `[experts, …]`, so a contiguous expert range is a contiguous
  outer-axis range.
- **The combine arithmetic** (`moxie-oracles/src/route.rs`, `combine_row`,
  around line 857): FP32 `acc` starts at zero; each slot in `combine_order`
  (ascending expert id) adds `w * v`; then `output_scale` is applied once;
  the interpreter rounds to BF16 once (`moxie-interp/src/lib.rs`, around
  line 1227).
- **Partition rules today:** `Route` is `Replicated`, and
  `ExpertMlp`/`Combine` are `NotDetermined` (`graph.rs`, `partition_rule`).
  Task 0056's lowering refuses routed ops.
- **Task 0058 precedent:** `LinearReductionOrder { blocks, slice }` is a
  node-keyed sidecar (`moxie-graph/src/lib.rs`, around line 494), read by an
  additive interpreter entry point. `lower_tensor_parallel` emits
  `Stage::Local { nodes, join }` with join `Gather` or `Reduce` (FP32
  partials, ascending rank order, one BF16 rounding). The host test harness
  is `moxie-cli/tests/tensor_parallel.rs`.
- **Shape C** (the routed reduced Gemma) has 5 experts, which does not divide
  by 2 or 4. The test uses a custom `TextConfig` through
  `moxie_cli::gemma::build_with_config`, as task 0060 did.
- **Strata** split each DeepSeek expert's intermediate dimension across ranks
  (expert TP), not by ownership. Moxie's spec (document 04 "Expert
  partitioning"; M5.4) requires **expert-owner** partitioning. Only Strata's
  exact FP32 partial reduction carries over.

## The design: numbered changes to implement exactly

1. **`crates/moxie-graph/src/lib.rs`:** add, next to
   `LinearReductionOrder`:
   ```rust
   /// Declared order for one `Combine` node split across expert owners.
   pub struct CombineReductionOrder { pub groups: u32, pub owned: Option<u32> }
   ```
   - `groups` = the number of contiguous, equal expert-owner groups. Group
     `g` owns experts `[g*E/groups, (g+1)*E/groups)`.
   - `owned: None` is the reference: every group summed.
   - `owned: Some(g)` is rank `g`'s partial.

   Add a second sidecar, `ExpertOwnership { groups: u32, owned: u32 }`, for
   `ExpertMlp`. Derive `Debug`, `Clone`, `Copy`, `PartialEq` and `Eq`, as the
   existing sidecar does.
2. **`crates/moxie-oracles/src/route.rs`:** add
   `combine_row_ordered(experts, weights, slots, width, order, output_scale,
   experts_total, groups, owned) -> Result<Vec<f32>>`.
   - Per group `g`: FP32 `acc_g` starts at 0.0. For each slot, in
     `combine_order` (ascending expert id), whose expert is in group `g`, add
     `w * v`. Slots of other groups are **skipped, never added as zero**.
   - `owned = Some(g)`: return `acc_g` unscaled and unrounded, as FP32.
   - `owned = None`: `total = acc_0`, then `total += acc_g` for
     g = 1..groups, element-wise in ascending `g`; multiply by `output_scale`
     once; return (the caller rounds).
   - Refuse with a typed error: `groups == 0`; `experts_total % groups != 0`;
     `owned >= groups`.
   - `groups == 1` must equal today's `combine_row` bit for bit. Assert that
     in one unit test.
3. **`crates/moxie-interp/src/lib.rs`:** extend task 0058's additive ordered
   entry point with two more node-keyed maps: combine orders and expert
   ownership. Absent entries keep today's behaviour exactly.
   - **`ExpertMlp` with `ExpertOwnership { groups, owned }`:** the local
     `gate_up` and `down` hold only the owned experts, as the outer-axis
     slice. Compute a slot only for a selected expert `e` in the owned range,
     using local index `e - first`. Leave non-owned slots as BF16 zeros in
     the output (they are never read; see 4).
   - **`Combine` with `CombineReductionOrder { owned: Some(g) }`:** return
     `combine_row_ordered(..., Some(g))` as **FP32** (no rounding), exactly
     as task 0058's partial `Linear` returns FP32.
   - **`Combine` with `owned: None`:** `combine_row_ordered(..., None)`, then
     the usual single BF16 rounding.
4. **`crates/moxie-plan/src/tensor_parallel.rs`:** extend
   `lower_tensor_parallel`:
   - (Amended 2026-09-23.) `Route`, its `ExpertMlp` and the consuming
     `Combine` together become
     `Stage::Local { nodes: [route, expert_mlp, combine], join: Reduce }`.
     Every rank runs `Route` with its unchanged, full params on the
     replicated input and router weights, so every rank reaches the same
     selection, as document 04 requires. The route table therefore never
     crosses a stage boundary, and the existing refusal of an externally
     supplied route table is unchanged. Router weights are whole on every
     rank (no `RankPart.rows` entry).
   - Per rank:
     - per rank: `ExpertOwnership { groups: R, owned: r }` and
       `CombineReductionOrder { groups: R, owned: Some(r) }`;
     - `RankPart.rows` for `experts_gate_up` and `experts_down`: the outer
       range `[r*E/R, (r+1)*E/R)`;
     - the reference (unsplit) plan carries
       `CombineReductionOrder { groups: R, owned: None }` for that
       `Combine`.
   - `Join::Reduce` is unchanged (see the amendment). Refuse, with a typed
     error, any routed `Combine` whose `output_scale != 1.0`.
   - Refuse with a typed error: `experts % R != 0`; an `ExpertMlp` whose
     slots feed anything other than exactly one `Combine`; any local-stage
     value escaping, under task 0058's generic boundary check.
   - The dense shared expert and every other op keep their existing rules
     (task 0058).
   - Update `partition_rule()` in `crates/moxie-graph/src/graph.rs`: give
     `ExpertMlp` and `Combine` a truthful new label (for example
     `ExpertOwnerShardable`) consumed by the lowering. `Route` stays
     `Replicated`.
5. **`crates/moxie-cli/tests/tensor_parallel.rs`:** the existing harness
   gains the routed case. Add no new test file.
   - Config: the Shape C routed geometry through `build_with_config` with
     **8 experts, top_k 2, vocab 12**, everything else as Shape C.
     (Amended 2026-09-23: Shape C's vocab of 11 does not divide by R, and
     task 0058 correctly refuses that; the dense TP fixture already uses 12.)
   - R ∈ {2, 4}; a multi-row prefill plus decode. The logits must be
     bit-identical to the reference running declared `groups = R`.
   - The fixture must include, and the test must assert that the route
     produced:
     - one row whose top-2 experts belong to the **same** owner (duplicate
       destination);
     - one row whose experts belong to **different** owners;
     - at R=4, one owner selected by **no** row (a rank with an empty
       partial).

     If the random route does not produce these, set the router weights or
     input deliberately; do not search randomly.
   - Add rows to the existing refusal table for `experts % R != 0` and for a
     slots-escape graph.

## Allowed files

`crates/moxie-graph/src/lib.rs`, `crates/moxie-graph/src/graph.rs`,
`crates/moxie-oracles/src/route.rs`, `crates/moxie-interp/src/lib.rs`,
`crates/moxie-plan/src/tensor_parallel.rs`, `crates/moxie-plan/src/lib.rs`
(exports only), `crates/moxie-interp/src/paged.rs` (the call-site arity),
the two existing `ExpertMlp`/`Combine` partition-label expectations in
`crates/moxie-interp/tests/reference_graphs.rs` (around line 2325) and
`crates/moxie-plan/src/selected.rs` (around line 1869), updated from
`NotDetermined` to `ExpertOwnerShardable` (amended 2026-09-23),
`crates/moxie-cli/tests/tensor_parallel.rs`, and this task
file's Result. **Not** `moxie-executor` or `moxie-cuda`, which task 0062
owns.

## Non-goals

No device execution (the next slice-4 task); no host-owned experts
(`M5.4 host experts`, next task); no uneven ownership; no MLA; no timing; no
model edits.

## Acceptance

- `cargo fmt --all -- --check`, workspace `clippy -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` and
  `cargo xtask spec-check` all pass.
- **Parallel-work note:** if a workspace gate fails **only** in files task
  0062 owns, report it; do not fix it.
- **Bit-identity** at R=2 and R=4, with the three route cases asserted.
- The `groups == 1` equivalence unit test (change 2).
- **Mutations,** each run and restored:
  - Add non-owned slots as zeros instead of skipping them.
  - Combine the partials in descending group order (at R=4).
  - The reference ignores `groups` (runs 1).

  Each must fail a test. At R=2 descending equals ascending (a+b == b+a),
  so run that mutation at R=4.
- One test per invariant. The Result lists per-file lines added and removed,
  and the mutation results.
- **Stop:** send `DECISION` if a numbered change conflicts with the code;
  do not improvise.

## Result, filled after work

- Source identity: the synthetic Shape C graph built by
  `moxie_cli::gemma::build_with_config`, with experts 8, top_k 2 and vocab 12;
  all other Shape C fields stay unchanged. The test sets the first routed
  projection to select cross-owner and duplicate-owner rows, and uses a unit
  router gain. Reference and split runs use the same fixture weights and token
  rows. This is not checkpoint or model support.
- The local stage spans Route through Combine, including Gemma's unchanged
  routed-input RMSNorm between Route and ExpertMlp; `Stage` represents a
  contiguous node range.
- Review map (added / removed lines):

  | File | + | - |
  |---|---:|---:|
  | `crates/moxie-cli/tests/tensor_parallel.rs` | 221 | 43 |
  | `crates/moxie-graph/src/graph.rs` | 5 | 4 |
  | `crates/moxie-graph/src/lib.rs` | 20 | 0 |
  | `crates/moxie-interp/src/lib.rs` | 215 | 32 |
  | `crates/moxie-interp/src/paged.rs` | 3 | 0 |
  | `crates/moxie-interp/tests/reference_graphs.rs` | 1 | 1 |
  | `crates/moxie-oracles/src/route.rs` | 266 | 0 |
  | `crates/moxie-plan/src/selected.rs` | 1 | 1 |
  | `crates/moxie-plan/src/tensor_parallel.rs` | 195 | 8 |
  | `docs/tasks/0064-m5-host-expert-owner-lowering.md` | 54 | 4 |
  | `docs/evidence/specification-version.md` (carried) | 1 | 1 |

  `docs/evidence/specification-version.md` and carried ADRs 0034 and 0035 were
  preserved; the existing specification-version delta was not part of this
  task's edits.
- Commands; passed / failed / skipped: `cargo fmt --all -- --check`, workspace
  clippy with `-D warnings`, `cargo test --workspace --locked`,
  `cargo xtask arch-check` (79 rejected fixtures, 21 accepted, 13 rules), and
  `cargo xtask spec-check` (10 documents) passed. Also passed
  `cargo test -p moxie-cli --test tensor_parallel --locked` and
  `cargo test -p moxie-oracles --locked`. No acceptance gate was skipped.
  Clippy first identified
  the specified nine-argument oracle and the interpreter's expanded private
  evaluator; the API was kept as contracted with narrow argument-count allows,
  and the workspace clippy rerun passed.
- Route cases observed in the five-row prefill (row indexes are zero-based):
  - Both rank counts observed `[[4, 1], [4, 1], [0, 1], [0, 1], [4, 1]]`.
    At R=2, row 0 selected experts `[4, 1]` owned by `[1, 0]` (cross-owner);
    row 2 selected `[0, 1]` owned by `[0, 0]` (duplicate destination).
  - R=4: row 0 selected `[4, 1]` owned by `[2, 0]`; row 2 selected `[0, 1]`
    owned by `[0, 0]`. Only owners 0 and 2 were selected, so owners 1 and 3
    had empty partials. Both rank counts were bit-identical to their declared
    reference at prefill and decode.
- Mutation results and restoration:
  - Adding non-owned slots as zero contributions failed the ordered-combine
    test (`NaN` instead of `8.0`); restored skip behavior.
  - Descending group accumulation at R=4 failed (`0.0` instead of `4.0`);
    restored ascending group order.
  - Making the reference run one group failed (`5.0` instead of `4.0`);
    restored the declared group reduction.
- Remaining obligations: none for task 0064. Device execution, host-owned
  experts and uneven ownership remain outside this task's scope.
