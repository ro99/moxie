# Task 0065 — the routed Gemma 4 graph runs on one GPU

Status: **open** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Builder Codex `luna`; reviewer Codex `sol`.

**GPUs released (owner, 2026-09-23).** The reservation below is lifted. The
builder now runs the GPU gates too: the end-to-end test and the mutations.
Only one agent uses the GPUs at a time; during this task that agent is the
builder. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

~~GPU reservation (owner, 2026-09-23): host and compile-only gates only.~~

## Identity and authority

- Task0065, M5 plan slice 4, second task, after task 0064 (host expert-owner
  lowering, accepted).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `6994645`.
  Preserve the carried `.gitignore`, `docs/evidence/specification-version.md`
  and ADRs 0034 and 0035. Never stage, edit or revert them.
- **Exit-gate clause served:** M5 exit, "same dense and **MoE** graph
  definitions execute single GPU, TP and PP without model edits". Today the
  routed Gemma graph (`Shape::C`) cannot run on a GPU at all: the dense
  device package has no `Route`, `ExpertMlp` or `Combine`. Task 0066
  (expert-owner TP2 on the pair) needs this single-GPU path as its
  reference.

## Facts established before writing (coordinator, 2026-09-23)

- `moxie-plan/src/selected.rs`:
  - `dense_semantic` (around line 941) maps `OpParams` to
    `SemanticKernelOp`. It has no arm for `Route`, `ExpertMlp` or `Combine`,
    so they fall through to "outside the reduced dense device package".
  - `dense_operands` (around line 964) maps a non-index input through
    `operand()`, and `operand()` refuses `ValueRole::Route`.
  - `lower_dense_mode` (around line 652) matches exactly one descriptor per
    node. It expects `output == node.contract.output`, except for
    `LinearPartial` (F32). It expects rounding `FinalBf16Rne`, except for
    `VocabProjection` and `LinearPartial` (`Unrounded`). A `Route` node's
    contract output is BF16 (the default `output_precision`), but its value
    is a route table: `u32` ids plus FP32 coefficients.
- `moxie-plan/src/lib.rs` (around line 808) already sizes a route value at
  8 bytes per entry (`u32` id + `f32` coefficient), with shape
  `[rows, top_k]`. The arena therefore already charges the right bytes. This
  task fixes the device layout inside that range (change 5).
- The host reference, which is the interpreter (`moxie-interp/src/lib.rs`,
  around lines 1181–1440), calls:
  - **Route:** `route::router_route_row` (`moxie-oracles/src/route.rs`,
    around line 504). It runs `router_input_row` (line 310), then
    `router_logits` (BF16-rounded `linear_row`), then `softmax` (line 45,
    Rust `f32::exp`), then `select_top_k_biased` (descending probability,
    **ties to the lower id**, then renormalized by the selected mass summed
    in selection order), then `apply_per_expert_scale`. Gemma's coefficient
    is `Fp32`, so nothing is narrowed.
  - **ExpertMlp:** `route::expert_row` per slot, slot-major (`r * top_k +
    j`), rounded to BF16. The grouped kernels in `cuda/expert_mlp.cu`
    (`moxie_expert_lanes_v1`, `moxie_gelu_tanh_v1` and the down loop) were
    qualified against this oracle in tasks 0019/0021. Reuse that
    arithmetic; do not rewrite it.
  - **Combine:** `route::combine_row` with `AscendingExpertId`. It is an FP32
    sum `acc += w * v` over slots in ascending expert id, then
    `acc *= output_scale` only when the scale is not 1.0, then rounded to
    BF16.
- `cuda/dense_graph.cu` builds the dense image by `#include`-ing whole
  `.cu` files (`bf16_chain.cu`, `paged_attention.cu`, `dense_ops.cu`).
- `moxie-executor/tests/dense_gemma_device.rs`,
  `reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu`
  (line 297), already compares device logits with the interpreter for
  prefill and decode on every GPU, over `for shape in [Shape::A,
  Shape::B]`. Shape C is the routed variant of shape A (5 experts, top-2,
  `moe_intermediate` 6).
- **Strata** (read-only): routed experts run on the host. Only the router
  logits are computed on the device and staged to the host
  (`docs/dsv4-rank-local-architecture.md` line 79,
  `backend_moe.inc.cuh`). Strata therefore has no device top-k or combine to
  follow. Moxie keeps the whole routed block on the device, so the M5 TP
  and PP stages need no host round trip inside a layer. Where host and
  device math could differ (the `exp` in the softmax), this follows the
  task 0059 precedent: evaluate in FP64 on the device and round once, and
  accept a possible last-ulp difference inside the existing gate.

## Bounded deliverable

- The routed Gemma 4 graph (`Shape::C`), built by the unchanged model code,
  lowers through the selected dense package and executes on **one** GPU.
  `Route`, `ExpertMlp` and `Combine` run as device kernels. Nothing routed
  runs on the host.
- Supported parameters are exactly those Gemma uses. Anything else is a
  typed `UnsupportedKernel` refusal at lowering:
  - `Route`: `input: Normalized`, `score: Softmax`,
    `per_expert_scale: true`, `selection_bias: false`,
    `coefficient: Fp32`;
  - `ExpertMlp`: `activation: GeGlu`;
  - `Combine`: `order: AscendingExpertId`, any finite `output_scale`. The
    kernel takes the scale, so task 0066 gets Laguna's 2.5 for free.
- **Non-goals:**
  - no TP or expert-owner execution (task 0066);
  - no host-owned experts;
  - no Laguna router (raw input, sigmoid, bias, BF16 coefficients);
  - no SwiGLU experts;
  - no quantized expert weights;
  - no performance work;
  - no model-crate edits;
  - no change to the M2 grouped path (`grouped.rs`, `grouped_device.rs`,
    the `expert_mlp` catalogue).

## Numbered changes

1. **`crates/moxie-types/src/capability.rs`:** add
   `SemanticKernelOp::Route` ("route") and `SemanticKernelOp::Combine`
   ("combine"), each with a one-line doc comment, and extend `name()`. Fix
   every exhaustive match the compiler reports. One known match is the test
   helper `descriptor()` in `moxie-plan/src/selected.rs` (around line 1338):
   add both variants to its last `=> Vec::new()` arm.

2. **New `crates/moxie-kernels/cuda/routed_ops.cu`, `#include`d by
   `cuda/dense_graph.cu` after `#include "expert_mlp.cu"`** (add that
   include too, so the dense image carries the qualified expert helpers). Add
   `cuda/routed_ops.cu` to the `rerun-if-changed` list in
   `crates/moxie-kernels/build.rs`. Every FP32 operation uses `__fadd_rn`,
   `__fmul_rn`, `__fdiv_rn` and `__fsqrt_rn`, exactly as `expert_mlp.cu`
   does, so nvcc cannot contract into an FMA. Four kernels:

   a. `moxie_dense_route_v1(const bf16* x, const bf16* proj, const bf16*
      gain, const bf16* per_expert, unsigned int* ids, float* coefficients,
      u64 rows, u64 hidden, u64 experts, u64 top_k, float eps, float
      input_scale)`. One thread per row. It follows the oracle step for step:
      - `sum` of `v*v` ascending from `0`; `mean = sum / (float)hidden`;
        `denom = sqrt(mean + eps)`.
      - `t_k = bf16(bf16(bf16(v_k / denom) * g_k) * input_scale)`.
      - `logit_e = bf16(Σ_k t_k * proj[e*hidden + k])`, accumulated
        ascending from 0.
      - `max` over the logits with `fmaxf`, starting at `-INFINITY`.
      - `exp_e = (float)exp((double)(logit_e - max))`, the subtraction in
        FP32.
      - `total = Σ_e exp_e`, ascending; `p_e = exp_e / total`.
      - `top_k` selection passes. Pass `j` scans `e` ascending and keeps the
        best candidate strictly after the previous pick in the order
        "higher `p`, then lower `e`". This reproduces the oracle's sort,
        including ties to the lower id. Write `ids[r*top_k + j]`.
      - `mass = Σ_j p_{ids[j]}` in selection order;
        `coefficients[r*top_k + j] = (p_{ids[j]} / mass) * per_expert[ids[j]]`.

      Recompute `t` and the logits in each pass rather than storing them;
      mark that with a `ponytail:` comment (O(top_k · experts · hidden) per
      row; a declared `rows × experts` FP32 workspace can replace it when
      routing cost matters, which is M6).
   b. `moxie_dense_expert_project_gelu_v1(const bf16* x, const unsigned
      int* ids, const bf16* gate_up, float* activated, u64 assignments, u64
      top_k, u64 hidden, u64 intermediate)`. It is
      `moxie_bf16_expert_project_gelu_v1`'s body with two differences:
      - `x_row = x + (slot / top_k) * hidden`;
      - the expert's weights start at
        `gate_up + ids[slot] * 2 * intermediate * hidden`.

      Call `moxie_expert_lanes_v1` and `moxie_gelu_tanh_v1`; do not copy
      them.
   c. `moxie_dense_expert_down_v1(const float* activated, const unsigned
      int* ids, const bf16* down, bf16* slots, u64 assignments, u64 hidden,
      u64 intermediate)`. It is `moxie_bf16_expert_down_v1`'s loop with two
      differences:
      - `row = down + ids[slot] * hidden * intermediate + component *
        intermediate`;
      - it writes `slots[slot * hidden + component]`.
   d. `moxie_dense_combine_v1(const unsigned int* ids, const float*
      coefficients, const bf16* slots, bf16* out, u64 rows, u64 top_k, u64
      hidden, float output_scale)`. One thread per `(row, component)`. It
      visits the row's `top_k` slots in ascending `(id, slot)` order, found
      by a selection pass per step (no array). Then:
      - `acc = acc + w * v`;
      - `if (output_scale != 1.0f) acc = acc * output_scale`;
      - `out = bf16(acc)`.

3. **`crates/moxie-kernels/src/lib.rs`:**
   - Add the constants `DENSE_ROUTE`, `DENSE_EXPERT_PROJECT_GELU`,
     `DENSE_EXPERT_DOWN` and `DENSE_COMBINE` for the four symbols.
   - In `dense_graph_catalogue()`, per SM, add three descriptors. Each uses
     `ContiguousRowMajorV1`, `Bf16InF32Acc`, the existing `shape` bounds,
     `DENSE_GRAPH_ABI` and the dense image hash.

   | Descriptor id | `operation` | `inputs` | `output` | `rounding` | `workspace` | `symbols` |
   |---|---|---|---|---|---|---|
   | `dense-route-v1-{sm}` | `Route` | `[bf16, weight, weight, weight]` | F32 | `Unrounded` | `Zero` | `[DENSE_ROUTE]` |
   | `dense-expert-mlp-gelu-v1-{sm}` | `ExpertMlp(GateTransform::GeluTanh)` | `[bf16, RouteIndex, weight, weight]` | BF16 | `FinalBf16Rne` | `RowsTimesIntermediateF32` | `[DENSE_EXPERT_PROJECT_GELU, DENSE_EXPERT_DOWN]`, in that order |
   | `dense-combine-v1-{sm}` | `Combine` | `[RouteIndex, bf16]` | BF16 | `FinalBf16Rne` | `Zero` | `[DENSE_COMBINE]` |

   Extend the `images` module's `use` list.

4. **`crates/moxie-plan/src/selected.rs`:**
   - `dense_semantic`: add three arms. A `Route` with exactly the supported
     parameters maps to `SemanticKernelOp::Route`. `ExpertMlp {
     activation: GeGlu, .. }` maps to `ExpertMlp(GateTransform::GeluTanh)`.
     `Combine { order: AscendingExpertId, .. }` maps to `Combine`. Any
     other `Route`, `ExpertMlp` or `Combine` returns `UnsupportedKernel`
     with a detail naming the unsupported parameter.
   - `dense_operands`: map an input whose role is `ValueRole::Route { .. }`
     to `KernelOperand::RouteIndex`, in the same `match role` that handles
     `ValueRole::Index`. Leave `operand()` itself unchanged: the BF16 chain
     still refuses routes.
   - `dense_shape`: return `(hidden, experts)` for `Route`,
     `(hidden, intermediate)` for `ExpertMlp` and `(hidden, hidden)` for
     `Combine`.
   - `dense_workspace`: for `ExpertMlp(_)`, return
     `(RowsTimesIntermediateF32, rows * top_k * intermediate * 4, 0)`, with
     checked multiplication and the same `invalid("workspace", …)` error
     style.
   - `lower_dense_mode`'s descriptor filter: add
     `SemanticKernelOp::Route` beside `LinearPartial` in the F32-output
     condition, and beside `VocabProjection | LinearPartial` in the
     `Unrounded` condition. Change nothing else in the filter.

5. **The route value's device layout**, stated once as a doc comment on
   the executor's `Route` arm (change 6): the route value's arena range
   holds `rows * top_k` `u32` ids at offset 0, then `rows * top_k` `f32`
   coefficients at byte offset `rows * top_k * 4`. The planner already
   charges `rows * top_k * 8` bytes, and `ExpertMlp` and `Combine` read
   both halves from that one range.

6. **`crates/moxie-executor/src/dense.rs`, `enqueue_dense`:** add three
   match arms before the final `_ =>`. Copy the style of the existing arms:
   `address(...)`, `let mut launch_* = …`, a `params` array, `launch(...)`
   with the symbol index `base` (and `base + 1` for the second symbol),
   then `push_launch`.
   - `OpParams::Route { hidden, experts, top_k, input:
     RouterInput::Normalized { eps, input_scale }, .. }`:
     - inputs `x = node.inputs[0]`, `proj = [1]`, `gain = [2]`,
       `per_expert = [3]`;
     - `ids = address(node.output)`,
       `coefficients = ids + rows * top_k * 4` (checked);
     - grid `rows.div_ceil(64)`, block 64;
     - label `"route"`.

     Any other `Route` returns `invalid("route", …)`: lowering has already
     refused it.
   - `OpParams::ExpertMlp { hidden, intermediate, top_k, .. }`:
     - `assignments = rows * top_k`;
     - the route is `node.inputs[1]`, and its ids are at the route
       address;
     - `activated = workspace_address(...)`;
     - project launch over `assignments * intermediate` threads (`base`);
     - down launch over `assignments * hidden` threads (`base + 1`);
     - block 256;
     - label `"expert-mlp"`.
   - `OpParams::Combine { hidden, top_k, output_scale, .. }`:
     - the route is `node.inputs[0]`, the slots are `node.inputs[1]`;
     - `coefficients = ids + rows * top_k * 4`;
     - `rows * hidden` threads, block 256;
     - label `"combine"`.

7. **`crates/moxie-executor/tests/dense_gemma_device.rs`:** change
   `for shape in [Shape::A, Shape::B]` in
   `reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu` to
   `[Shape::A, Shape::B, Shape::C]`, and update the test's doc comment or
   message to say "dense and routed". Nothing else in that test changes;
   it already builds the fixture through `moxie_cli::gemma::build(shape)`.
   This is the end-to-end GPU gate (**pending**).

8. **Stale expectations:** if a host test asserts the dense catalogue's
   descriptor count, or that `Route`/`ExpertMlp`/`Combine` are refused by
   the dense package, update that expectation and name it in the Result. Do
   not add a new test for it.

9. **One host lowering test**, in
   `crates/moxie-executor/tests/dense_gemma_device.rs`:
   `routed_gemma_lowers_through_the_dense_package`. It must not touch CUDA:
   - Build `Shape::C` through `moxie_cli::gemma::build`.
   - Construct an SM 8.6 `DeviceCapability` literal, as the host tests in
     `crates/moxie-executor/tests/allocation_refusal.rs` do.
   - Call `moxie_plan::lower_selected` with the rows used by the
     end-to-end test's prefill (5) and `moxie_kernels::dense_graph_catalogue()`.
   - Assert that the candidate's nodes include one `Route`, one
     `ExpertMlp(GeluTanh)` and one `Combine` descriptor per routed layer.
   - Assert that the planned route value's `logical_bytes == rows * top_k * 8`.

   Run only this test, filtered:
   `cargo test -p moxie-executor --features driver,paged-attention-binding
   --test dense_gemma_device routed_gemma_lowers_through_the_dense_package
   -- --exact`. **Never run that test binary unfiltered while the GPUs are
   reserved.**

## Allowed files

- `crates/moxie-types/src/capability.rs`
- `crates/moxie-kernels/cuda/routed_ops.cu` (new)
- `crates/moxie-kernels/cuda/dense_graph.cu`
- `crates/moxie-kernels/build.rs`
- `crates/moxie-kernels/src/lib.rs`
- `crates/moxie-plan/src/selected.rs`
- `crates/moxie-executor/src/dense.rs`
- `crates/moxie-executor/tests/dense_gemma_device.rs`
- Other files only for a compiler-reported exhaustive match (change 1) or
  a stale expectation (change 8). Name each one in the Result.

## Acceptance

**Host gates (run now):**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`. This compiles the CUDA image through nvcc; it does not open a
  device.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.
- Change 9's filtered host test.

**GPU gates (the builder runs them; GPUs released 2026-09-23):**
- Change 7's end-to-end test on both 3090s and the 5060 Ti. Shape C
  prefill plus decode must be within the task 0012 gate (bit-exact on
  exactly representable values, otherwise at most one BF16 ulp). Shapes A
  and B stay green.
- Mutations, each applied, run and restored:
  - `Combine` in selection order instead of ascending id;
  - route ties broken to the higher id;
  - the per-expert scale dropped;
  - `ExpertMlp` indexing weights by slot `j` instead of `ids[slot]`.

  Record them in the Result as exact one-line patches, so the GPU batch
  can apply them without redesign.
- The full `cargo xtask test-gpu` stays green (last known: 63/63).

**Stop conditions:**
- The end-to-end test selects different experts from the host, or misses
  the gate for a reason that is not a defect. Report it; do not loosen the
  gate.
- Execution needs a model edit, or a change to lease or ledger semantics.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Implemented changes 1–6: the dense catalogue now selects Route,
  ExpertMlp(GeGlu) and Combine; lowering admits only Gemma's declared route,
  expert and combine parameters; the CUDA image contains the routed kernels
  and the already-qualified expert helpers; the executor binds the shared
  route-value range and admitted expert workspace. Change 7 adds Shape C to
  the existing prefill/decode comparison. Change 9 adds the exact host-only
  lowering assertion for five rows, one descriptor of each routed operation
  per routed layer, and `rows * top_k * 8` route bytes.
- Change 8: no stale dense-catalogue count or refusal expectation was found.
  `selected_chain_refuses_routed_operations` remains valid: it exercises the
  separate reduced BF16 chain catalogue, not the dense graph package.
- Host gates passed: `cargo fmt --all -- --check`; workspace clippy with
  `-D warnings`; driver/paged-attention clippy with `-D warnings` (nvcc image
  compiled; no device opened); `cargo test --workspace --locked`; `cargo xtask
  arch-check` (79 rejected fixtures, 21 accepted, 13 rules); `cargo xtask
  spec-check` (10 specification documents unchanged); and the required exact
  filtered test (`1 passed`, `2 filtered out`).
- GPU gate pending: the Shape A/B/C prefill/decode test on both 3090s and the
  5060 Ti, the four runtime mutations below, and `cargo xtask-cuda test-gpu`.
  They were not run because the owner's GPU reservation forbids device work.

### Mutation patches for the GPU batch

These are unapplied, one-line substitutions; apply and restore each in the
GPU window.

1. Combine by selection slot: `const unsigned int expert = row_ids[slot];` → `const unsigned int expert = static_cast<unsigned int>(slot);`
2. Break equal route probabilities toward the higher id: `|| (probability == best_probability && expert < best_expert)) {` → `|| (probability == best_probability && expert > best_expert)) {`
3. Drop per-expert scaling: `__fdiv_rn(probability, mass), __bfloat162float(per_expert[expert]));` → `__fdiv_rn(probability, mass), 1.0F);`
4. Use slot `j` for the gate/up expert weights: `gate_up + static_cast<unsigned long long>(ids[slot]) * 2 * intermediate * hidden;` → `gate_up + static_cast<unsigned long long>(slot % top_k) * 2 * intermediate * hidden;`

### Review map

The code changes are confined to the allowed implementation and test files;
this task record contains the Result. The carried `.gitignore`,
`specification-version.md`, and ADRs 0034/0035 remain untouched.

| File | Change |
|---|---|
| `crates/moxie-types/src/capability.rs` | +6 / −0; add Route and Combine semantic operation identities. |
| `crates/moxie-kernels/cuda/routed_ops.cu` | +214 / −0; route, expert project/down, and combine kernels. |
| `crates/moxie-kernels/cuda/dense_graph.cu` | +2 / −0; include expert helpers and routed kernels. |
| `crates/moxie-kernels/build.rs` | +1 / −0; track the new source. |
| `crates/moxie-kernels/src/lib.rs` | +59 / −6; routed symbols and per-SM descriptors. |
| `crates/moxie-plan/src/selected.rs` | +79 / −3; supported lowering, route operands/shapes, and checked workspace charge. |
| `crates/moxie-executor/src/dense.rs` | +194 / −0; bind routed operations and document route-range layout. |
| `crates/moxie-executor/tests/dense_gemma_device.rs` | +91 / −6; extend the pending device gate and add the host lowering test. |
| `docs/tasks/0065-m5-routed-gemma-single-gpu.md` | +50 / −1; record results and unapplied GPU mutation patches. |

No model code, lease/ledger semantics, or files outside this set were changed;
no commit was created.
