# Task 0066 — the expert-partitioned routed Gemma runs on the 3090 pair

Status: **accepted** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Built by Codex `luna` from the coordinator's design, with two
amendments: the boundary `Dim` evaluation, and mutation 2 moved to a kernel
case because zero-filled slots mask it end to end. Sol returned REVISE in
round 1 (sidecar cross-check; route proof on S = 2; resource-test cut), then
ACCEPT in round 2. The coordinator re-ran `fmt`, workspace and driver
`clippy`, `arch-check`, `spec-check`, the `moxie-plan` tests,
`dense_tp2_device` (dense and routed TP2 on the pair), `dense_gemma_device`
and the combine kernel test.

**Amendment, 2026-09-23 (coordinator), after the builder's DECISION.**
- *Observation:* `dense_tp_workers.rs::value_bytes` resolves only
  `Dim::Const` and `Dim::Symbol`. `ExpertMlp`'s output extent
  `Mul(Symbol(rows), Const(top_k))` fails in `boundary_bytes` before any
  routed stage runs. Separately, the `dense_tp2_device` gate command omits
  `paged-attention-test-hooks`, which that target needs to compile.
- *Replacement:*
  - Change 6c (below): `value_bytes` evaluates the whole `Dim` expression.
  - The `dense_tp2_device` command gains the `paged-attention-test-hooks`
    feature.
- *Authority:* coordinator.

**Amendment 2, 2026-09-23 (coordinator), after the builder's STOP.**
- *Observation:* mutation 2 (the partial combine sums every group) survived
  the routed TP2 gate.
- *Cause:* rank-local `ExpertMlp` writes exact zeros into non-owned slots.
  Adding `w · (+0)` terms changes no bits, so end to end the owned-group
  predicate is masked by the zero-fill. The predicate is still the kernel's
  contract: the oracle skips non-owned slots and never adds them as zero.
- *Replacement:* change 8b below. Mutation 2 is now expected to fail that
  case, not the TP2 gate.
- *Authority:* coordinator.

## Identity and authority

- Task0066, M5 plan slice 4, third task. It follows task 0064 (host
  expert-owner lowering) and task 0065 (routed Gemma on one GPU), both
  accepted.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `6d8ef08`.
  Preserve the carried `.gitignore`, `docs/evidence/specification-version.md`
  and ADRs 0034 and 0035. Never stage, edit or revert them.
- **GPUs are free** (owner, 2026-09-23). Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder is the only agent using the
  GPUs during this task.
- **Exit-gate clauses served:**
  - M5 exit: "same … MoE graph definitions execute single GPU, **TP** … without
    model edits";
  - M5 exit: "an expert-partitioned plan [has] **correctness, resource and
    cancellation** tests";
  - roadmap M5.4: "duplicate destinations … varying route unions".

  This task also adds the generic mid-step cancellation to the TP workers.
  Slice 5's combined TP/PP plan needs the same cancellation.

## Facts established before writing (coordinator, 2026-09-23)

- **Task 0064's lowering** (`moxie-plan/src/tensor_parallel.rs`, around
  lines 736–800) makes `[Route … ExpertMlp … Combine]` one
  `Stage::Local { join: Reduce }`.
  - Each rank's `ExpertMlp` gets `experts = E/R`, with the outer-axis
    slice `[r·E/R, (r+1)·E/R)` of `experts_gate_up` and `experts_down`
    (`RankPart.rows`).
  - Each rank carries `ExpertOwnership { groups: R, owned: r }` and
    `CombineReductionOrder { groups: R, owned: Some(r) }`.
  - The reference carries `CombineReductionOrder { groups: R, owned: None }`.
  - `Route` keeps `experts = E` and runs whole on every rank.
  - `build_stage_graph` copies both sidecars into `StageGraph.combine_orders`
    and `StageGraph.expert_ownership`, keyed by local node id.
- **The host arithmetic** is `combine_row_ordered` (`moxie-oracles/src/route.rs`,
  around lines 916–1040). Group `g` holds experts `[g·per, (g+1)·per)`.
  - Each group's FP32 partial starts at zero and adds `w·v` in ascending
    `(id, slot)`.
  - Reference (`owned: None`): `total = acc_0`, then `total += acc_g` in
    ascending `g`, then `× output_scale` if the scale is not 1, then one BF16
    rounding by the interpreter.
  - A rank (`owned: Some(g)`): `acc_g`, FP32, unscaled and unrounded.
- **The existing reduce** `moxie_tp_reduce_f32_v1` computes
  `bf16(rank0 + rank1)`. That is exactly the reference `total` for R = 2 with
  `output_scale == 1`. Task 0064 already refuses any other scale.
- **The device lowering** (`selected.rs`):
  - `lower_selected_ordered` takes only linear orders.
  - `check_routed_edges` (task 0065) requires the `Route`'s `experts` to equal
    the `ExpertMlp`'s. On a rank that is `E` versus `E/R`, so the check must
    become ownership-aware.
  - `partial_outputs` makes a partial's output FP32. `LinearPartial` is the
    precedent for a new partial operation.
- **The TP workers** (`moxie-executor/src/dense_tp_workers.rs`):
  - `run_stage` (around line 1732) calls `lower_selected_ordered` with
    `stage.linear_orders` only.
  - `execute_dense` (around line 481) loops over `lowering.stages` inside an
    `execution` closure. An error returned there takes the same recovery path
    as the test's `Fault::Rank`.
  - There is **no mid-step cancellation**. `Fault::Drop` only abandons a
    finished, uncommitted step.
- **The TP2 device test** (`moxie-executor/tests/dense_tp2_device.rs`):
  - `order_sensitive_fixture` builds a dense Shape A variant with heads 8 and
    vocab 12.
  - `tp2_dense_worker_gate` runs, in order: reference steps, worker prefill
    and decode, then the fault matrix (`Rank`, `Collective`, the recoverable
    faults, `Stall`), checking `workers.stats()` for leaks.
  - `stage_weight_value` slices `rows` as `[rows, cols]`. For a rank-3
    expert weight it must keep the trailing dims.
- **The router crafting** from task 0064
  (`moxie-cli/tests/tensor_parallel.rs`, around line 395):
  - `router_proj` is set to zeros except `[0] = 1`, `[4·width] = -1`;
  - `router_scale` is set to ones.

  With 8 experts this produces same-owner and cross-owner rows.
- **Strata**: expert TP only, split along each expert's intermediate axis
  (task 0064's facts). Only the exact ordered FP32 reduction carries over,
  per ADR 0036.

## Bounded deliverable

- The routed fixture (8 experts, top-2, vocab 12) runs TP2 on the 3090 pair
  through the persistent rank workers. The ranks own experts 0–3 and 4–7.
  Prefill and decode are **bit-identical** to one GPU running the declared
  `groups = 2` reference.
- The whole existing fault matrix runs on the routed fixture too, plus a new
  mid-step cancellation. Worker stats are unchanged after every fault.
- A **resource** assertion: each rank's stage admits exactly half the
  expert weight bytes.
- **Non-goals:**
  - no host-owned experts (the next slice-4 task);
  - no scaled combine: Laguna's 2.5 stays refused by task 0064, and it goes
    to the ledger;
  - no R = 4 on the device (there are only two 3090s);
  - no MLA;
  - no performance work;
  - no model-crate edits;
  - no change to `grouped.rs` or `grouped_device.rs`.

## Numbered changes

1. **`crates/moxie-types/src/capability.rs`:** add
   `SemanticKernelOp::CombinePartial` ("combine_partial"), with a doc line:
   "one expert-owner group's FP32 combine partial, unscaled and unrounded".
   Fix the exhaustive matches the compiler reports, including the
   `descriptor()` test helper in `selected.rs` (last `Vec::new()` arm).

2. **`crates/moxie-kernels/cuda/routed_ops.cu`:**
   - a. Add `unsigned long long first_expert, unsigned long long
     local_experts` as the **last two parameters** of
     `moxie_dense_expert_project_gelu_v1` and `moxie_dense_expert_down_v1`.
     Let `id = ids[slot]`.
     - If `id < first_expert || id >= first_expert + local_experts`: the
       project kernel `return`s, and the down kernel writes
       `__float2bfloat16_rn(0.0F)` to its slot component and `return`s.
     - Otherwise both use `id - first_expert` where they now use
       `ids[slot]` to offset the weights.
     - On one GPU the executor passes `first_expert = 0` and
       `local_experts = experts`, so task 0065's behaviour is unchanged.
   - b. Add `unsigned long long experts_per_group, unsigned long long groups`
     as the last two parameters of `moxie_dense_combine_v1`.
     - For `g` in `0..groups`: `partial = 0`, then visit the row's slots in
       ascending `(id, slot)` as today, skipping any with
       `id / experts_per_group != g`, and add `w·v`.
     - For `g == 0`, `total = partial`; otherwise `total = total + partial`.
     - Then the existing scale and BF16 store, applied to `total`.
     - With `groups == 1` and `experts_per_group == experts`, the result is
       bit-identical to today's.
   - c. Add `moxie_dense_combine_partial_v1(const unsigned int* ids, const
     float* coefficients, const bf16* slots, float* out, u64 rows, u64 top_k,
     u64 hidden, u64 experts_per_group, u64 owned)`.
     - It is the same visit as (b), but only for group `owned`.
     - It writes the FP32 `acc` with no scale and no rounding.
   - Same `__fadd_rn`/`__fmul_rn` discipline throughout.

3. **`crates/moxie-kernels/src/lib.rs`:**
   - Add `DENSE_COMBINE_PARTIAL`.
   - In `dense_graph_catalogue()`, per SM, add `dense-combine-partial-v1-{sm}`:
     - `operation: CombinePartial`;
     - `inputs: [RouteIndex, bf16]`;
     - `output: F32`;
     - `rounding: Unrounded`;
     - `workspace: Zero`;
     - `symbols: [DENSE_COMBINE_PARTIAL]`;
     - everything else as `dense-combine-v1`.

4. **`crates/moxie-plan/src/selected.rs`:**
   - a. `lower_selected_ordered` gains two parameters after `orders`:
     `combine_orders: &BTreeMap<NodeId, CombineReductionOrder>` and
     `expert_ownership: &BTreeMap<NodeId, ExpertOwnership>`. Validate them
     beside the linear orders, each with a typed `invalid(...)`:
     - every combine-order key names a `Combine`, and every ownership key
       names an `ExpertMlp`;
     - `groups > 0`;
     - `owned < groups` (`owned` is an `Option` for combine orders).

     Pass both maps to `lower_dense_mode`. `lower_dense` passes empty maps.
   - b. `lower_dense_mode` gains the two maps. Operation selection: a
     `Combine` whose order has `owned: Some(_)` becomes
     `SemanticKernelOp::CombinePartial`; every other `Combine` stays
     `Combine`. In the descriptor filter, add `CombinePartial` wherever
     `LinearPartial` appears (the F32 output and the `Unrounded` rounding).
     Add `CombinePartial` outputs to `partial_outputs`, the set that makes a
     value FP32.
   - c. `check_routed_edges` takes `&BTreeMap<NodeId, ExpertOwnership>`. The
     `ExpertMlp` rule becomes `route.experts == mlp.experts × groups`, where
     `groups` is that `ExpertMlp`'s ownership `groups`, or 1 when it has none.
     `top_k` must still be equal. The `Combine` rules are unchanged.
     `routed_edges_must_agree_on_the_route` must still pass unchanged.
   - d. `SelectedPlanCandidate` gains `combine_orders` and `expert_ownership`
     fields, each with a getter like `linear_orders()`. The two other struct
     literals (around lines 419 and 607) set them to `BTreeMap::new()`.
   - Import `CombineReductionOrder` and `ExpertOwnership` from wherever
     `LinearReductionOrder` is imported.

5. **`crates/moxie-executor/src/dense.rs`, `enqueue_dense`:**
   - `ExpertMlp` arm: if `candidate.expert_ownership().get(&node.id)` is
     `Some(o)`, pass `first_expert = o.owned × experts` (checked) and
     `local_experts = experts`, where `experts` is this node's param.
     Otherwise pass `0` and `experts`.
   - `Combine` arm: find the producer `ExpertMlp` of `node.inputs[1]` with
     `graph.nodes().iter().find(|n| n.output == value)`. Set:
     - `experts_total = mlp.experts × ownership_groups`, where
       `ownership_groups` is that `ExpertMlp`'s ownership `groups`, or 1;
     - `groups = combine_orders().get(&node.id).map_or(1, |o| o.groups)`;
     - `experts_per_group = experts_total / groups`, refusing a remainder
       with `invalid("combine", …)`.

     Branch on `selected.descriptor.operation`:
     - `Combine`: the existing launch, plus the two new arguments.
     - `CombinePartial`: launch `DENSE_COMBINE_PARTIAL` with
       `owned = order.owned.unwrap()` (the planner guarantees it), labelled
       `"combine-partial"`.

6. **`crates/moxie-executor/src/dense_tp_workers.rs`:**
   - a. `run_stage` passes `&stage.combine_orders` and
     `&stage.expert_ownership` to `lower_selected_ordered`.
   - b. **Mid-step cancellation.** `execute_dense` gains a last parameter,
     `cancel: &std::sync::atomic::AtomicBool`. As the first statement of
     each iteration of `for declared in &lowering.stages` (inside the
     `execution` closure):
     ```rust
     if cancel.load(Ordering::Acquire) {
         return Err(Error::Cancelled { at: "tensor-parallel stage" });
     }
     ```
     Add nothing else. The error takes the existing recovery path. Doc
     comment: the check sits at stage boundaries, which are the safe
     boundaries where both ranks have drained.

   - c. `value_bytes`: bind the graph's rows symbol once, with
     `let mut table = SymbolTable::new(); table.bind(graph.rows_symbol(), rows);`.
     Replace the `match dim` with
     `dim.eval(&table).map_err(|_| invalid("boundary", "unresolved dimension"))?`.
     `Dim::eval` (`moxie-types/src/dim.rs`) already checks overflow. Any
     unbound symbol still refuses, as before.

7. **`crates/moxie-executor/tests/dense_tp2_device.rs`:**
   - a. **Fixture:** `order_sensitive_fixture` becomes
     `order_sensitive_fixture(routed: bool)`.
     - When `routed` is true, before building, set `config.moe` to
       `Shape::C.config().moe` with `experts = 8`. After building, apply task
       0064's router values (`router_proj` zeros except `[0] = 1` and
       `[4·width] = -1`; `router_scale` ones) to **every** layer's router
       weights. Task 0064's test set only the first match.
     - The existing linear crafting runs in both cases.
     - `fixture()` gains the same `routed` parameter.
   - b. **Weights:** `stage_weight_value`'s `rows` branch keeps the
     trailing dims: shape `[rows.end - rows.start]` followed by
     `whole.shape()[1..]`, using the tensor's existing shape accessor.
   - c. **References:** `host_logits` and `reference_step` also take
     `&BTreeMap<NodeId, CombineReductionOrder>` (the lowering's
     `combine_orders`).
     - `host_logits` calls `run_with_partition_orders`, with empty expert
       ownership.
     - `reference_step` passes the combine orders and an empty ownership map
       to `lower_selected_ordered`.
     - `one_block_orders` gets `&BTreeMap::new()` for combine orders.
   - d. **The gate:** `tp2_dense_worker_gate()` becomes
     `tp2_worker_gate(routed: bool)`. The `#[test]` calls it with `false`,
     then with `true`. The routed run executes the **whole** existing
     sequence unchanged.
   - e. **Routes:** in the routed run only, assert that the host prefill
     routes (read the interpreter's route values as task 0064's test does,
     or recompute them with `route::router_route_row`) contain a
     same-owner row and a cross-owner row. Assert too that the decode step
     `[6]`/`[5]` routes to a single owner, so one rank's partial is empty.
     If the decode row does not, change the decode token deterministically
     to one that does, and say which in the Result. Do not search randomly.
   - f. **Cancellation:** add `Fault::Cancel`. In `tp_worker_step`, pass an
     `AtomicBool` to `execute_dense`. For `Fault::Cancel`, set it inside the
     `bindings` closure at the same trigger as `Fault::Rank`. In the gate,
     immediately after the `Fault::Rank` recovery step, add:
     - a `Fault::Cancel` step that must return `Error::Cancelled { .. }`;
     - `workers.stats()` equal to its value before the step;
     - then a clean step equal to one more reference step, computed in the
       reference block like the others.

     Every other `execute_dense` call site passes a `false` flag.
   - g. **Resource:** in the routed run, for each rank, lower that rank's
     routed stage graph with `lower_selected_ordered` (host-side, as
     `run_stage` does). For the stage's `experts_gate_up` and `experts_down`
     weight values, assert that `candidate.value(id).logical_bytes` is
     exactly half the same weight's bytes in the reference candidate.

8. **Extend `combine_kernel_sums_in_ascending_expert_order`** (`dense.rs`
   `mod tests`) with a second case in the same test, not a new test. The
   case is `groups = 2`, `experts_per_group = 2`:
   - `ids = [3, 0, 2]`, `coefficients = [1, 1, 1]`;
   - `slots = [2^-30, -1.0, 1.0]`, so experts 3, 0 and 2 have outputs
     `2^-30`, `-1` and `1`.

   Hand-derived results, to put in the comment:
   - **Ungrouped:** `-1 + 1 = 0`, then `+ 2^-30`, giving `2^-30`.
   - **Grouped:** group 0 is `-1`; group 1 is `1 + 2^-30 = 1` (absorbed); the
     total is `0`.

   Assert `0.0` for `groups = 2`. The existing case passes
   `experts_per_group = 4`, `groups = 1`.

8b. **The same test gains a third case: the partial kernel.** Launch
    `DENSE_COMBINE_PARTIAL` with the same `ids = [3, 0, 2]`,
    `coefficients = [1, 1, 1]`, `slots = [2^-30, -1.0, 1.0]`,
    `experts_per_group = 2` and `owned = 0`.
    - Group 0 holds only expert 0, so the result is **exactly `-1.0`**
      (FP32 bits).
    - With the owned predicate removed, the kernel sums every slot in
      ascending id: `-1 + 1 + 2^-30 = 2^-30`.

    Put both derivations in the comment. Load the module with both symbols,
    or a second module; follow the existing case's mechanics.

**Review round 1 fixes (sol REVISE, 2026-09-23; coordinator's design):**

9. **`selected.rs`, `check_routed_edges`:** also take `&BTreeMap<NodeId,
   CombineReductionOrder>`, and pass it from `lower_dense_mode`. For each
   `Combine`, let `order` be its combine order (if any) and `own` be its
   producer `ExpertMlp`'s ownership (if any). Refuse with the existing
   `UnsupportedKernel { operation: "route", .. }` unless one of these holds:
   - no `order` and no `own`;
   - `order = { groups, owned: None }`, no `own`, and `mlp.experts % groups
     == 0` (the single-device reference);
   - `order = { groups, owned: Some(g) }` and `own = { groups: same,
     owned: g }` (a rank).

   Add **one** assertion to the existing `routed_edges_must_agree_on_the_route`
   test, not a new test. Take a well-formed route / 4-expert `ExpertMlp` /
   `Combine` graph (build one if the existing fixture does not have one),
   with `ExpertOwnership { groups: 2, owned: 0 }` and
   `CombineReductionOrder { groups: 2, owned: Some(1) }`. `lower_selected_ordered`
   must refuse it. This is sol's repro.
10. **`dense_tp2_device.rs`, route proof on the declared orders:**
    - `prefix_activation` gains `linear_orders: &BTreeMap<NodeId,
      LinearReductionOrder>`. The prefix stage is built with `part = None`
      from `0..end`, so its local node ids equal the original ones. Call
      `run_with_partition_orders` with the entries whose `id.0 < end`, and
      empty combine and ownership maps.
    - `host_route_ids` passes `&lowering.linear_orders`.
    - The fixture-crafting caller passes `&BTreeMap::new()`, which is
      unchanged behaviour.
    - The same-owner, cross-owner and empty-owner-decode assertions then
      inspect the routes the S = 2 reference actually takes. If one no
      longer holds, STOP and report; do not re-craft.
11. **`dense_tp2_device.rs`, resource check (sol's cut):** delete the
    unsplit `StageGraph` construction and the Combine node-id remapping.
    Take the reference expert-weight `logical_bytes` from
    `lower_selected_ordered` on the full fixture graph at the original
    weight ids, with the lowering's combine orders. Keep the per-rank stage
    candidates and the exact half-bytes assertion.

## Allowed files

- `crates/moxie-types/src/capability.rs`
- `crates/moxie-kernels/cuda/routed_ops.cu`
- `crates/moxie-kernels/src/lib.rs`
- `crates/moxie-plan/src/selected.rs`
- `crates/moxie-executor/src/dense.rs`
- `crates/moxie-executor/src/dense_tp_workers.rs`
- `crates/moxie-executor/tests/dense_tp2_device.rs`
- `crates/moxie-executor/tests/dense_gemma_device.rs`: only if a call site
  needs the new parameters.
- This task's Result.
- Other files only for a compiler-reported exhaustive match. Name each one
  in the Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**GPU gates** (run by the builder, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`):
- `cargo test -p moxie-executor --features
  driver,paged-attention-binding,paged-attention-test-hooks --test dense_tp2_device`. Both the dense and the routed runs must pass,
  with the pair's UUIDs in the output.
- `cargo test -p moxie-executor --features driver,paged-attention-binding
  --test dense_gemma_device`. Task 0065's five cases must still pass.
- `cargo test -p moxie-executor --features driver,paged-attention-binding
  --lib combine_kernel_sums_in_ascending_expert_order`.
- `cargo xtask-cuda test-gpu`, as a no-regression check (the fixed list;
  63/63).

**Mutations**, each applied, run, shown failing and restored. Each names the
check that catches it; the fixture was chosen so that it does.
1. The expert kernels test `id < local_experts` instead of the
   `[first_expert, first_expert + local_experts)` range. Rank 1 then drops
   experts 4–7, and the routed TP2 prefill (cross-owner rows) fails.
2. `moxie_dense_combine_partial_v1` sums every group, ignoring `owned`.
   Change 8b's partial case fails (Amendment 2: the TP2 gate cannot
   see it, because non-owned slots are exact zeros).
3. `moxie_dense_combine_v1` ignores `groups` (it sums ungrouped). Change 8's
   `groups = 2` case fails.
4. The cancellation check in `execute_dense` is removed. The
   `Fault::Cancel` step then does not return `Cancelled`.

**Stop conditions:**
- The routed TP2 output is not bit-identical to the reference. Report which
  step and which logit; do not loosen anything.
- A mutation survives. Report it and the check you expected to catch it.
- Execution needs a model edit, or a change to lease or ledger semantics.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- **Complete after Review round 1 fixes.** No commit was made. The coordinator's
  amendments and carried files are preserved.
- The coordinator-authorized `value_bytes` change binds the graph rows symbol
  once in a `SymbolTable` and evaluates each complete `Dim`, mapping evaluation
  failures to the prescribed boundary error.
- `check_routed_edges` cross-checks each Combine order with its producer
  ExpertMlp ownership. The existing `routed_edges_must_agree_on_the_route`
  test now includes sol's well-formed route-8 / local-expert-4 repro and
  asserts typed refusal for ownership `{ groups: 2, owned: 0 }` versus combine
  order `{ groups: 2, owned: Some(1) }`.
- Route proofs evaluate prefix activations with the lowering's declared S=2
  linear orders. The dense TP2 gate confirmed same-owner and cross-owner
  prefill routes and a single-owner decode route; decode uses token 9 at
  position 5 because the host oracle showed token 6 routes cross-owner there.
  The resource check now compares each rank's stage candidate against a
  full-fixture reference candidate using original weight ids and no unsplit
  StageGraph or node-id remap. Both expert weight charges are exactly half.
- Routed TP2 prefill and decode were bit-identical on UUIDs
  `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and
  `GPU-81fe4578-59b2-37c4-421e-287cdac78704`. Cancellation/recovery and sticky
  stall checks passed. The combine-order device test covers selection-order
  cancellation, grouped accumulation and the owned FP32 partial, whose exact
  result is `-1.0` for ids `[3,0,2]`, coefficients `[1,1,1]`, slots
  `[2^-30,-1,1]`, `experts_per_group=2`, `owned=0`.
- All four mutations were applied, observed failing at their intended check,
  and restored: (1) replacing the expert range guard with `expert >=
  local_experts` failed routed prefill bit identity; (2) removing the partial
  owned-group predicate failed the partial case with FP32 `2^-30` bits
  (`813694976`) instead of `-1.0` bits (`3212836864`); (3) making combine
  process one ungrouped ascending traversal failed the `groups=2` case with
  BF16 `2^-30` (`12416`) instead of zero; (4) deleting the stage cancellation
  check made the `Fault::Cancel` call return logits instead of `Cancelled`.
- Host gates passed: formatting, workspace clippy, driver-feature executor
  clippy, `cargo test --workspace --locked`, `cargo xtask arch-check` (79
  rejected and 21 accepted fixtures), and `cargo xtask spec-check` (10
  documents). The focused `cargo test -p moxie-plan --locked` passed (22 unit
  tests and 11 expert-plan matrix tests).
- Review-round GPU gates passed with `CUDA_DEVICE_ORDER=PCI_BUS_ID`:
  `dense_tp2_device` (dense and routed runs on the 3090 pair) and
  `combine_kernel_sums_in_ascending_expert_order`. The prior round's
  `dense_gemma_device` (five cases on all three GPUs) and full CUDA suite
  (63/63, SM86 and SM120 qualified) remain recorded; they were not rerun for
  this review round.
- Review map (`+/-` lines from the current diff):
  - `crates/moxie-types/src/capability.rs`: +3/-0.
  - `crates/moxie-kernels/cuda/routed_ops.cu`: +70/-8.
  - `crates/moxie-kernels/src/lib.rs`: +20/-4.
  - `crates/moxie-plan/src/selected.rs`: +175/-21.
  - `crates/moxie-executor/src/dense.rs`: +195/-25.
  - `crates/moxie-executor/src/dense_tp_workers.rs`: +17/-8.
  - `crates/moxie-executor/tests/dense_tp2_device.rs`: +367/-29.
  - `docs/tasks/0066-m5-expert-owner-tp2-on-the-pair.md`: +137/-5, including
    coordinator amendments and this Result.
