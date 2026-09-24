# Task 0068 — host-owned experts beside one GPU

Status: **open** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0068, M5 plan slice 4, the last task. It follows task 0066 (the
  expert-owner device path) and task 0067 (MLA).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`. The owner chose sequential work
  with luna.
- The **design below is the coordinator's**, including every lifetime and
  cross-check rule. Implement the numbered changes exactly. If one conflicts
  with the code, stop and send a `DECISION` report; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `3f864be` (task 0067 accepted). Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035.
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs during this task.
- **Clauses served:**
  - roadmap M5.4: "bounded expert-owner partitioning using shared
    dispatch/transport/reduction. Test … **host experts**";
  - M5 exit: "no … hidden peer-to-host fallback". The host step is
    declared in the plan and visible in the launch order.

## Facts established before writing (coordinator, 2026-09-23)

- **Strata precedent** (`docs/dsv4-rank-local-architecture.md`, line 79 and
  "Asynchronous host-staging lifetimes"):
  - The GPU computes the router, the routed experts run on the host, and
    the result returns through staging.
  - Its hardest defect (Experiment 0091) was an asynchronous
    host-to-device copy whose source died too early. Strata's rule: "any
    command that submits an asynchronous H2D and returns *without
    synchronizing* must own a private pinned staging slot". Only a fully
    synchronous path may share one buffer.
  - This task uses **only synchronous copies**. There is no asynchronous
    host source anywhere, so there is nothing to retain.
- **The CPU expert kernel.** `moxie_kernels::cpu_expert::expert_group_bf16`
  (`cpu_expert.rs`, around line 152) is bitwise-equal to
  `route::expert_row` (task 0019's gate).
  - It takes one expert's `gate_up`/`down` BF16 bytes and an
    `ExpertAssignment { rows, slots }`.
  - It writes BF16 slots into a slot-major `[any, hidden]` buffer.
  - Its FP32 workspace is `ExpertShape::workspace_f32(tiling)` floats.
- **The device pieces from tasks 0065 and 0066:**
  - `moxie_dense_combine_partial_v1` gives a group's FP32 partial.
  - `moxie_tp_reduce_f32_v1(a, b, out, n)` gives `bf16(a + b)`.
  - The grouped reference (`CombineReductionOrder { groups: 2, owned: None
    }`) is `bf16(acc_0 + acc_1)`. That equals `reduce(device partial of
    group 0, host partial of group 1)` exactly.
- **Synchronous copies.**
  - `moxie_cuda` has a synchronous device-to-host `copy_to_host_at`
    (`driver.rs`, around line 1307), and `DeviceRange::copy_to_host` wraps it
    (`arena.rs`, around line 610).
  - Host-to-device is synchronous only whole-buffer (`copy_from_host(&mut
    self)`, around line 926), so this task adds an offset form.
  - The dense executor already makes a synchronous mid-graph readback, for
    attention K/V (`dense.rs`, `execute_attention`, around line 1145),
    charged through `host_workspace_bytes`.
- **`build_stage_graph`** (`tensor_parallel.rs`, around lines 184–208)
  rewrites every `Attention` and `MlaAttention` layer to **0**. A
  whole-graph stage with several attention nodes would therefore fail
  validation.
- **Routing coverage.** Task 0064's router crafting (`router_proj` zeros
  except `[0] = 1`, `[4·width] = -1`; `router_scale` ones; 8 experts) gives
  rows routed to experts {0, 1} (device only) and rows routed to {4, 1}
  (device plus host).

## Bounded deliverable

- On **one** GPU, with the routed fixture (8 experts, top-2):
  - experts 0–3 run on the device;
  - experts 4–7 run on the host, and their weights are **never uploaded**;
  - prefill and decode are **bit-identical** to the same GPU running the
    declared `groups = 2` reference with every expert on the device.
- The device weight charge for the expert tensors is exactly half. The
  host step is charged through `host_workspace_bytes`.
- The host step appears as `"combine-host-join"` in the launch order.
- **Non-goals:**
  - no host experts under TP (on the pair);
  - no other split than 2 groups with the host owning group 1;
  - no asynchronous or overlapped host work;
  - no scaled combine (it stays refused);
  - no affine or quantized host experts;
  - no performance claim;
  - no model-crate edits.

## Numbered changes

1. **`crates/moxie-plan/src/tensor_parallel.rs`, `build_stage_graph`:**
   renumber the state layers densely instead of setting them to 0.
   - For each `Attention` or `MlaAttention` node, in node order:
     `local_layer = state_layers.len() as u32` **before** inserting. Then
     `state_layers.insert(local id, original)`, and set the param's layer
     to `local_layer`.
   - A stage with one attention node still gets layer 0, so current
     behaviour is unchanged. Task 0067's MLA check compares against the
     source **before** this rewrite; keep that order.
   - For the whole Gemma graph, whose layers are 0..L in node order, the
     numbering is the identity.

2. **New `crates/moxie-plan/src/host_experts.rs`**, exported from
   `lib.rs`:
   ```rust
   /// Two expert-owner groups: group 0 on the device, group 1 on the host.
   pub struct HostExpertJoin { first_host_expert: u32, host_experts: u32 }
   pub struct HostExpertLowering {
       /// Applied to the whole graph with `build_stage_graph`.
       device: RankPart,
       /// The full-device reference's declaration: `{ groups: 2, owned: None }`.
       reference_orders: BTreeMap<NodeId, CombineReductionOrder>,
       /// Keyed by the routed `Combine` node.
       joins: BTreeMap<NodeId, HostExpertJoin>,
   }
   pub fn lower_host_experts(graph: &Graph) -> Result<HostExpertLowering, TensorParallelRefused>
   ```
   **The fields of `HostExpertLowering` and `HostExpertJoin` are private**
   (read through getters). The only constructor is `lower_host_experts`,
   so a hand-edited, malformed lowering cannot exist. This is the lesson
   from task 0067's three review rounds: derive, do not trust a declaration.
   For every `ExpertMlp` node:
   - It must satisfy the **same routed-block checks** as task 0064's expert
     pass, with the same refusals: a `Route` producer; exactly one consumer,
     which is the adjacent `Combine` consuming the same route;
     `output_scale == 1` (`ScaledCombine`); `experts % 2 == 0`
     (`Dimension`).
   - Let `h = experts / 2`. The device part gets:
     - `params`: that `ExpertMlp` with `experts = h`;
     - `rows[gate_up] = rows[down] = 0..h`;
     - `expert_ownership = { groups: 2, owned: 0 }`;
     - `combine_orders[combine] = { groups: 2, owned: Some(0) }`.
   - The join gets `{ first_host_expert: h, host_experts: h }`, and the
     reference orders get `{ groups: 2, owned: None }`.
   - Refuse a graph with no `ExpertMlp` using `TensorParallelRefused::Op`
     on its first node.

3. **`crates/moxie-types/src/capability.rs`:**
   - Add `SemanticKernelOp::CombineHostJoin` ("combine_host_join"), with a
     doc line: "group 0's device partial plus group 1's host partial,
     reduced in group order and rounded once".
   - Add `WorkspaceExpression::RowsTimesHiddenTimesTwoF32`, which
     `evaluate` returns as `None` (it needs the width).
   - Fix the exhaustive matches the compiler reports.

4. **`crates/moxie-kernels/src/lib.rs`**, in `dense_graph_catalogue()`, per
   SM, add `dense-combine-host-join-v1-{sm}`:
   - `operation: CombineHostJoin`;
   - `inputs: [RouteIndex, bf16]`;
   - `output: BF16`;
   - `rounding: FinalBf16Rne`;
   - `workspace: RowsTimesHiddenTimesTwoF32`;
   - `symbols: [DENSE_COMBINE_PARTIAL, TP_REDUCE_F32]`, in that order.

   No CUDA change.

5. **`crates/moxie-plan/src/selected.rs`:**
   - a. Add `pub fn lower_selected_host_experts(graph, workload, capability,
     catalogue, lowering: &HostExpertLowering)`. It checks the UUID like
     `lower_selected_ordered`. It validates that every join key names a
     `Combine` whose device combine order is `{ groups: 2, owned: Some(0) }`,
     and whose producer `ExpertMlp` has
     `experts == join.host_experts == join.first_host_expert`. Anything
     else is a typed `invalid(...)`. Then it calls `lower_dense_mode` with
     `require_complete_graph = true`, the device part's combine orders and
     ownership, and the joins.
   - b. `lower_dense_mode` gains `joins: &BTreeMap<NodeId, HostExpertJoin>`.
     Every other caller passes an empty map. A `Combine` with a join
     selects `CombineHostJoin`, taking precedence over `CombinePartial`. Its
     output stays BF16 and is **not** in `partial_outputs`. The descriptor
     filter needs no change for it: BF16 output, `FinalBf16Rne`.
   - c. `dense_workspace` for `CombineHostJoin`:
     - device bytes: `2 · rows · hidden · 4`;
     - host bytes: `rows·hidden·2` (routed input) + `rows·top_k·8` (route)
       + `rows·top_k·hidden·2` (host slots) + `rows·hidden·4` (host partial)
       + `4 · (intermediate + 2·intermediate)` (CPU workspace, with
       `ExpertTiling::lanes(intermediate)`).

     `intermediate` comes from the producer `ExpertMlp`, found in `graph`.
     Use checked arithmetic throughout.
   - d. `SelectedPlanCandidate` gains `host_expert_joins` and its getter.
     The other two struct literals set an empty map.

6. **`crates/moxie-cuda/src/driver.rs`:** add
   `pub fn copy_from_host_at(&self, offset: usize, src: &[u8]) -> Result<()>`.
   It mirrors `copy_to_host_at` (the same range checks) and calls the
   **synchronous** `cuMemcpyHtoD_v2(ptr + offset, …)`. The SAFETY comment
   is the one on `copy_from_host`: the synchronous form returns only once
   the copy has completed.
   **`crates/moxie-executor/src/arena.rs`:** add
   `pub(crate) fn copy_from_host(&self, source: &[u8]) -> Result<()>` on
   `DeviceRange`, mirroring `copy_to_host`.

7. **`crates/moxie-executor/src/dense.rs`:**
   - a. Add
     `pub struct HostExpertWeights<'a> { pub combine: NodeId, pub gate_up: &'a [u8], pub down: &'a [u8] }`,
     holding the host group's BF16 bytes, expert-first. `DenseGraphStep`
     gains `pub host_experts: &'step [HostExpertWeights<'step>]`.
   - b. **At step entry,** before any launch, validate the host weights:
     - each candidate join has exactly one entry;
     - no entry lacks a join;
     - each entry's lengths equal `host_experts · 2·I·H · 2` (gate/up) and
       `host_experts · H·I · 2` (down).

     Anything else is `invalid("host_experts", …)`.
   - c. **The `Combine` arm when the descriptor is `CombineHostJoin`,** in
     this exact order:
     1. Launch `DENSE_COMBINE_PARTIAL` (symbol `base`) with
        `owned = 0` and `experts_per_group = experts_total / 2`, into
        workspace bytes `[0, rows·H·4)` (**A**).
     2. `stream.synchronize()?`.
     3. Copy the route range (`rows·top_k·8` bytes) and the producer
        `ExpertMlp`'s `inputs[0]` range (`rows·H·2` bytes) to host `Vec`s,
        allocated with `try_reserve_exact`, as `execute_attention` does.
     4. For each host expert `e` in ascending order, collect the assignment
        `rows[i] = r`, `slots[i] = r·top_k + j` for every `(r, j)` with
        `ids[r·top_k + j] == e`. If it is non-empty, call
        `cpu_expert::expert_group_bf16` with expert `e - first`'s byte
        slices, `GateTransform::GeluTanh`,
        `ExpertTiling::lanes(intermediate)` and one reused workspace. It
        writes into a zeroed `rows·top_k·H` BF16 slot buffer.
     5. **Host partial,** FP32, for each row `r`: `acc = 0.0f32`. Visit the
        slots in ascending `(id, slot)` order (the same selection-pass
        rule as the kernel), keeping only `id ≥ first`, and add
        `acc += w · v`, where `v` is the slot's BF16 value widened.
        Plain Rust `f32` `*`/`+` (no `mul_add`).
     6. Upload the partial with the **synchronous**
        `DeviceRange::copy_from_host` to workspace bytes
        `[rows·H·4, 2·rows·H·4)` (**B**). Take a sub-range with the
        existing range-slicing helper, or add an offset parameter if none
        exists; say which in the Result.
     7. Launch `TP_REDUCE_F32(A, B, output, rows·H)` (symbol `base + 1`).
     8. `push_launch(lease, "combine-host-join")`.

     Any error returns through the existing post-submission path.
   - Update every `DenseGraphStep` struct literal with `host_experts: &[]`:
     `dense_tp_workers.rs::run_stage`, `dense_gemma_device.rs` (three) and
     `dense_tp2_device.rs` (one).

8. **`crates/moxie-executor/tests/dense_gemma_device.rs`:** add **one**
   test, `routed_gemma_host_experts_match_the_grouped_reference_on_every_gpu`,
   reusing the file's existing helpers.
   - **Fixture:** the `Shape::C` config with `moe.experts = 8`, plus task
     0064's router crafting on every layer.
   - **Lowering:** `lower_host_experts(&fixture.graph)`, then
     `build_stage_graph(&graph, Some(&lowering.device), 0..n, Some(output))`
     for the device graph. Slice its weights by `StageWeight.rows` along
     the outer axis, keeping the trailing dims. The host weights are
     experts `4..8` of each layer's `experts_gate_up` and `experts_down`.
   - **Per GPU:**
     - the reference runs the full graph with
       `lower_selected_ordered(…, &lowering.reference_orders, &BTreeMap::new())`;
     - the host run uses `lower_selected_host_experts`;
     - both do prefill (5 rows) then one decode row, with the same paged
       runs, **and must be bit-identical**.
   - Assert that the launch order contains `"combine-host-join"` once per
     routed layer.
   - **Resource:** the host candidate's expert-weight `logical_bytes` are
     exactly half the reference's.
   - **Refusal:** one assertion that an empty `host_experts` slice is
     refused with `InvalidRequest` before any launch.

## Allowed files

- `crates/moxie-plan/src/tensor_parallel.rs` (change 1 only)
- `crates/moxie-plan/src/host_experts.rs` (new)
- `crates/moxie-plan/src/lib.rs` (exports)
- `crates/moxie-plan/src/selected.rs`
- `crates/moxie-types/src/capability.rs`
- `crates/moxie-kernels/src/lib.rs`
- `crates/moxie-cuda/src/driver.rs`
- `crates/moxie-executor/src/arena.rs`
- `crates/moxie-executor/src/dense.rs`
- `crates/moxie-executor/src/dense_tp_workers.rs` (struct literal only)
- `crates/moxie-executor/tests/dense_gemma_device.rs`
- `crates/moxie-executor/tests/dense_tp2_device.rs` (struct literal only)
- This task's Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):
- `cargo test -p moxie-executor --features driver,paged-attention-binding
  --test dense_gemma_device`, on all three GPUs, with the new test plus
  task 0065's five cases.
- `cargo test -p moxie-executor --features
  driver,paged-attention-binding,paged-attention-test-hooks --test
  dense_tp2_device`. Change 1 must not disturb TP2.
- `cargo xtask-cuda test-gpu`, as a no-regression check.

**Mutations**, each applied, run, shown failing and restored. The router
crafting routes some rows to expert 4, which is host-owned, so each is
observable.
1. Step 4 passes expert `e`'s slice without subtracting `first`. The host
   then reads the wrong expert or is refused, and the new test fails.
2. Step 7 passes **A** as both reduce inputs. The host partial is then
   dropped, and the rows routed to expert 4 fail.

**Stop conditions:**
- The host run is not bit-identical. Report the step, GPU, row and element.
- A mutation survives.
- Any copy would need to be asynchronous.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Pending.
