# Task 0067 — MLA per-weight head partition on the host

Status: **accepted** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Built by Codex `luna` from the coordinator's design. Sol
reviewed it over four rounds, all on one finding class: the rank-local MLA
declaration was not anchored to its source. That tripped the three-round
tripwire. After round 3, the coordinator replaced the per-field checks
with one derive-and-compare rule (change 9), and round 4 was ACCEPT with a
whole-class audit. The coordinator re-ran `fmt`, workspace and driver
`clippy`, `arch-check`, `spec-check`, the host TP tests and the MLA oracle
tests. There was no GPU work (host-only).

## Identity and authority

- Task0067, M5 plan slice 4, fourth task. The owner chose sequential work
  with luna (2026-09-23). MLA was moved ahead of host-owned experts (now
  task 0068) because it is host-only and fully precedented.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `00ceb69`.
  Preserve the carried `.gitignore`, `docs/evidence/specification-version.md`
  and ADRs 0034 and 0035. Never stage, edit or revert them.
- **No GPU work.** This task is host-only; MLA has no device kernel.
- **Clauses served:**
  - M5.1: "define legal partition semantics for **all implemented ops**".
    `MlaAttention` is the last implemented op that TP lowering still refuses
    (ledger row M5.1-d: `q_a`/`kv_a` replicated, `q_b`/`kv_b` split by head,
    `o_proj` split along its input axis).
  - ADR 0036: the split is exact by declared order.

## Facts established before writing (coordinator, 2026-09-23)

- **The op.** `OpParams::MlaAttention { descriptor }` is one fused node.
  - Its inputs are `[hidden, positions, q_a_proj, q_a_layernorm, q_b_proj,
    kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj]`
    (`moxie-interp/tests/mla_reference.rs` around line 155; the interpreter
    reads `input(2..=8)` in that order, `moxie-interp/src/lib.rs` around line
    1093).
  - Its partition rule is already `HeadShardable { SharedLatentReplicated,
    GlobalReduction }` (task 0053).
- **The oracle** (`moxie-oracles/src/mla.rs`) is FP64 throughout.
  - `project` (around line 258) produces the shared latent from
    `q_a`/`kv_a`, and the per-head query from `q_b_proj`, whose head `h` owns
    rows `[h·(nope+rope), (h+1)·(nope+rope))`.
  - `decompress` (around line 326) reads `kv_b_proj` per head, at
    `base = head · (nope+v) · kv_lora`. Head `h` therefore owns rows
    `[h·(nope+v), (h+1)·(nope+v))` of the `[heads·(nope+v), kv_lora]` matrix.
  - `attend` (around line 409) computes each head independently, then
    `matvec(o_proj, hidden, heads·v, head_output)`. That is
    `sum = 0.0f64; sum += w·x` in ascending column order (around line 168).
  - The interpreter narrows the output `as f32`, then rounds it to BF16 once.
- **Consequence:** a rank that owns `H/R` whole heads can run the unchanged
  oracle, given:
  - a descriptor with `heads = H/R`;
  - its `q_b_proj` and `kv_b_proj` row ranges;
  - its `o_proj` input-axis column slice `[r·(H/R)·v, (r+1)·(H/R)·v)`.

  Its result is its `o_proj` partial. The shared latent (`q_a`, `kv_a`, the
  MLA cache) is replicated on every rank.
- **TP lowering** (`moxie-plan/src/tensor_parallel.rs`):
  - The first node loop (around line 408) refuses `MlaAttention`.
  - Spans are built pass by pass (attention chains, MLP, experts,
    output linears, vocabulary).
  - `RankPart.rows` gives an outer-axis weight range.
  - `RankPart.slices` gives an input-axis slice, but it is keyed by the
    node's **activation** input `inputs[0]` and applied to that node's
    weights (`build_stage_graph`, around line 130). MLA's `inputs[0]`
    (`hidden`) is replicated, so `o_proj` needs a per-weight slice keyed by
    the weight value itself.
  - `build_stage_graph` rewrites an `Attention` node's `layer` to 0 and
    records it in `state_layers`; MLA needs the same.
- **The sidecar.** `CombineReductionOrder { groups, owned: Option<u32> }` is
  a node-keyed "ordered reduction across owner groups" declaration, and the
  interpreter reads it by node id. MLA reuses it for its head groups. No new
  type is added.
- **The host harness** (`moxie-cli/tests/tensor_parallel.rs`):
  - `stage_graph` already slices weights by `rows` and by `slice` (rank-2).
  - `bind` handles reads.
  - `reduce_rows` sums FP32 partials in ascending rank order, then rounds to
    BF16 once.
  - The MLA cache is `KvCache::for_mla_branch` with
    `SequenceState::new([StateKind::MlaLatent])`, as `mla_reference.rs`
    uses it (around line 255).
- **Strata:** GLM/DeepSeek MLA ran rank-local. The only rule that carries
  over is ADR 0036: FP32 partials, combined in declared order, rounded once.

## Bounded deliverable

- `lower_tensor_parallel` lowers an `MlaAttention` node at R = 2 and R = 4
  as one `Stage::Local { join: Reduce }`, and refuses `heads % R != 0`.
- A host test runs a heads-4 MLA graph through prefill (3 rows) and decode
  (1 row). The split result must be **bit-identical** to the unsplit graph
  running the declared `groups = R` order.
- **Non-goals:**
  - no device MLA;
  - no MLA inside a larger model graph (there is no MLA model fixture);
  - no KV-cache sharding (the latent stays replicated, per task 0053's
    rule);
  - no performance work;
  - no model-crate edits.

## Numbered changes

1. **`crates/moxie-graph/src/lib.rs`:** extend `CombineReductionOrder`'s doc
   comment with one sentence: it also declares an `MlaAttention` node's
   `o_proj` reduction across head groups. There is no code change in this
   file.

2. **`crates/moxie-oracles/src/mla.rs`:** add
   `pub fn o_proj_grouped(o_proj: &[f64], hidden: usize, width: usize,
   head_output: &[f64], groups: usize) -> Result<Vec<f32>>`.
   - Refuse, with a typed `invalid(...)`, `groups == 0` or
     `width % groups != 0`, and use the same length and finiteness checks
     as `matvec`.
   - For each output row `i`, for `g` in `0..groups`:
     `s = 0.0f64; s += o[i][k] · h[k]` for `k` in the group's columns
     `[g·width/groups, (g+1)·width/groups)`, ascending. Then
     `p = s as f32`.
   - `total = p` for `g == 0`; otherwise `total += p`, in FP32.
   - Return the totals (FP32, not rounded to BF16).
   - Add **one** unit test: with `groups == 1`, it equals
     `matvec(...).iter().map(|v| *v as f32)` bit for bit.

3. **`crates/moxie-interp/src/lib.rs`, the `MlaAttention` arm:** if
   `combine_orders.get(&node.id)` is `Some(order)`, replace the
   `result.output` narrowing with a call to
   `mla::o_proj_grouped(&o_proj, hidden, descriptor.heads·v_head_dim,
   &result.head_output, groups)`, where `groups` depends on the order:
   - **`owned: Some(_)`** (a rank): `groups = 1`. Return the values as
     **FP32**, not rounded:
     `Value::Float(HostTensor::f32(out, shape)?)`, exactly as the `Combine`
     partial does.
   - **`owned: None`** (the reference): `groups = order.groups`, then the
     usual `HostTensor::round_to_bf16`.

   With no order, today's path is unchanged.

4. **`crates/moxie-plan/src/tensor_parallel.rs`:**
   - a. `RankPart` gains `weight_slices: BTreeMap<ValueId,
     LinearInputSlice>`, doc: "an input-axis slice keyed by the weight
     itself, for fused operations whose activation input is replicated".
     `Default` still derives.
   - b. `build_stage_graph`: when `input` is a weight and
     `p.weight_slices` has it, use that slice. Otherwise keep the existing
     lookup. The existing rank-2 check and `spec.shape[1]` rewrite apply
     unchanged. Also rewrite an `MlaAttention` descriptor's `layer` to 0 and
     record it in `state_layers`, exactly as the `Attention` branch does.
   - c. The first node loop: `OpParams::MlaAttention { .. } => continue`,
     instead of the refusal.
   - d. A new span pass, placed after the vocabulary pass and in the same
     style. For each `MlaAttention` node:
     - refuse `heads % R != 0` with `TensorParallelRefused::Heads { layer,
       heads, ranks }`;
     - refuse an already-claimed node with `HeadChain`;
     - claim the node and insert
       `Span { end: index + 1, join: Join::Reduce { output: node.output } }`;
     - record it for (e).
   - e. In the part-assembly section, beside the expert loop, set
     `hp = heads/R`, `qh = nope+rope`, `kvh = nope+v` and `W = heads·v`.
     - The reference `combine_orders` gets
       `{ groups: R, owned: None }` for the node.
     - Each rank `r` gets:
       - `params`: the same `MlaAttention` with `descriptor.heads = hp`;
       - `rows[q_b_proj] = r·hp·qh .. (r+1)·hp·qh`;
       - `rows[kv_b_proj] = r·hp·kvh .. (r+1)·hp·kvh`;
       - `weight_slices[o_proj] = LinearInputSlice { first: r·hp·v, width:
         hp·v, full_width: W }`;
       - `combine_orders` gets `{ groups: R, owned: Some(r) }`.
     - `q_a_proj`, `q_a_layernorm`, `kv_a_proj_with_mqa` and
       `kv_a_layernorm` stay whole.

5. **`crates/moxie-cli/tests/tensor_parallel.rs`:** add **one** test,
   `mla_heads_split_is_bit_identical_at_prefill_and_decode`, plus one
   helper, `mla_graph(heads) -> (Graph, Bindings<Value>)`.
   - **The graph** follows `mla_reference.rs`'s `build()`, with `hidden = 8`,
     `q_lora = 4`, `kv_lora = 4`, `nope = 2`, `rope = 2`, `v = 2`, `heads`
     as given, and `layer = 0`. Weight shapes follow from those:
     - `q_a [4, 8]`, `q_a_layernorm [4]`;
     - `q_b [heads·4, 4]`;
     - `kv_a [6, 8]`, `kv_a_layernorm [4]`;
     - `kv_b [heads·4, 4]`;
     - `o [8, heads·2]`.

     Use deterministic, non-repeating values (copy the `values(len, seed)`
     helper), and all-ones layernorms.
   - **The test,** for R in {2, 4} with `heads = 4`:
     - `lower_tensor_parallel` returns exactly one stage,
       `Local { 0..1, Reduce }`.
     - **Reference:** `run_with_partition_orders` on the full graph with
       `lowering.combine_orders`, a fresh `MlaLatent` state and an MLA cache.
       Prefill 3 rows at positions `[0, 1, 2]`, then decode 1 row at `[3]`.
       Append the prompt to the state exactly as `mla_reference.rs` does.
     - **Split:** per rank, `stage_graph(&graph, &weights, 0..1,
       Some(&part), Some(output))`, its own state and MLA cache, and
       `run_with_partition_orders` with the stage's sidecars. Then
       `reduce_rows` over the ranks' FP32 partials.
     - Assert `bits(split) == bits(reference)` for prefill and for decode.
   - **Refusal:** add a row to `unsupported_partitions_are_refused` for
     `mla_graph(2)` at R = 4, expecting `TensorParallelRefused::Heads`.

**Review round 1 fixes (sol REVISE, 2026-09-23; coordinator's design):**

6. **Interpreter, `MlaAttention` arm:** before using an order, validate it,
   refusing with `Error::InvalidRequest { field: "mla_order", .. }`:
   - `groups > 0`;
   - for `owned: None`, `descriptor.heads % groups == 0`, so a head is
     never split across groups (sol's heads = 3, groups = 2 repro);
   - for `owned: Some(g)`, `g < groups`.

   Also merge the duplicate BF16 match arms (sol's LOW finding).
7. **`build_stage_graph`: owner-to-slice agreement.** For an `MlaAttention`
   node whose rank-local combine order is `{ groups, owned: Some(g) }`,
   refuse, with a typed `invalid("stage", …)`, unless all of these hold,
   where `hp` is the local `descriptor.heads`:
   - `hp · groups` equals the head count implied by the `o_proj` slice's
     `full_width / v_head_dim`;
   - `weight_slices[o_proj].first == g · hp · v`;
   - `rows[q_b_proj].start == g · hp · (nope + rope)`;
   - `rows[kv_b_proj].start == g · hp · (nope + v)`.

   Add **one** assertion to the new MLA test. Take the R = 2 lowering,
   change rank 0's MLA combine order to `owned: Some(1)`, and
   `build_stage_graph` must refuse it. This is sol's second repro. Add
   **one** refusal assertion for the heads = 3, `groups = 2` reference
   order, by running the interpreter on `mla_graph(3)` with that order.

**Review round 2 fix (sol REVISE, 2026-09-23; coordinator's design):**

8. **`build_stage_graph`: anchor change 7 to the source graph, not to the
   rank's own declaration.** Let `source` be the original node's
   `MlaAttention` descriptor (`node.params`, before the part override), and
   `hp` the local `descriptor.heads`. Refuse, as in change 7, unless all of
   these hold:
   - `hp · groups == source.heads`;
   - the `o_proj` slice has `full_width == source.heads · v`,
     `width == hp · v` and `first == g · hp · v`;
   - `rows[q_b_proj] == g·hp·(nope+rope) .. (g+1)·hp·(nope+rope)`, as the
     full range, not just the start;
   - `rows[kv_b_proj] == g·hp·(nope+v) .. (g+1)·hp·(nope+v)`, as the full
     range.

   This replaces change 7's `full_width`-based count. Add **one**
   assertion to the same MLA test for sol's coherent-truncation repro
   (`heads = 4`, R = 2, rank 0 truncated to one head): `build_stage_graph`
   must refuse it.

**Review round 3: one rule replaces the per-field checks (coordinator,
2026-09-23).** Rounds 1–3 each found another MLA declaration field that
was not checked: first the order, then a self-consistent truncation, then
shape-preserving descriptor fields and a missing order. That is the
three-round tripwire. The fix is to stop checking fields and instead
**derive, then compare for equality**.

9. **`tensor_parallel.rs`: one derivation, used twice.**
   - a. Add `fn mla_rank(node: &Node, groups: u32, owned: u32) ->
     Result<MlaRank>`, returning
     `MlaRank { params: OpParams, q_b_rows: Range<u64>, kv_b_rows: Range<u64>, o_slice: LinearInputSlice }`.
     It is computed from the **source** node only: the same descriptor with
     `heads = heads/groups`, and change 4e's ranges and slice. Refuse
     `heads % groups != 0` or `owned >= groups`.
   - b. The lowering (change 4e) builds each rank's MLA entries by calling
     `mla_rank`, with no second copy of the arithmetic.
   - c. `build_stage_graph`, for every `MlaAttention` node when `part` is
     `Some(p)`:
     - require `p.combine_orders[node] == { groups, owned: Some(g) }`.
       A missing order or `owned: None` is refused (sol's finding 2);
     - require `p.params[node]`, `p.rows[q_b_proj]`, `p.rows[kv_b_proj]` and
       `p.weight_slices[o_proj]` to **equal** `mla_rank(source, groups, g)`
       exactly. Every descriptor field is then covered, `rms_norm_eps`,
       `rope_base`, `rope_layout` and `visibility` included (sol's finding
       1);
     - require no `rows` or `weight_slices` entry for `q_a_proj`,
       `q_a_layernorm`, `kv_a_proj_with_mqa` or `kv_a_layernorm`.

     Anything else is a typed `invalid("stage", …)`. The layer rewrite
     happens **after** the comparison, and `state_layers` keeps the source
     layer.
   - d. **Delete** change 7's and change 8's per-field checks. They are
     subsumed.
   - e. **Tests:** keep the existing three repro assertions (they must still
     be refused, now by the equality). Add two more to the same MLA test:
     rank 0 with `rms_norm_eps` changed, and rank 0 with its MLA order
     removed. The interpreter's change-6 validation stays: it is the
     unsplit reference's guard.

## Allowed files

- `crates/moxie-graph/src/lib.rs` (doc only)
- `crates/moxie-oracles/src/mla.rs`
- `crates/moxie-interp/src/lib.rs`
- `crates/moxie-plan/src/tensor_parallel.rs`
- `crates/moxie-cli/tests/tensor_parallel.rs`
- This task's Result.
- Other files only for a compiler-reported `RankPart` struct literal. Name
  each one in the Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`. This is compile-only, because `RankPart` is shared with the
  device workers.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**Mutations**, each applied, run, shown failing and restored. Both are
observable: the weights are non-repeating, so a wrong head or column
selection changes the output.
1. `weight_slices[o_proj].first = 0` for every rank. At R ≥ 2, rank 1
   multiplies its heads by rank 0's columns, and the new test fails.
2. `rows[q_b_proj]` is rank 0's range for every rank. Every rank then
   computes rank 0's queries, and the new test fails.

**Stop conditions:**
- The split is not bit-identical. Report which step, R and element; do not
  loosen anything.
- A mutation survives. Report it.
- The oracle's head layout differs from the facts above.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Implemented ordered MLA `o_proj` group accumulation, whole-head lowering at
  R=2 and R=4, the weight-keyed input slice, local descriptor/state-layer
  rewrite, the heads-divisibility refusal, and the host prefill/decode test.
  The oracle head-major layouts match the established facts. No other
  `RankPart` struct literals needed updates.
- `mla_heads_split_is_bit_identical_at_prefill_and_decode` passes for prefill
  (3 rows) and decode (1 row) at both rank counts. The reference uses the
  lowering's `groups = R` order; each split uses its rank sidecars and fresh
  per-rank MLA state/cache, then reduces FP32 partials in rank order.
- Host gates passed: `cargo fmt --all -- --check`; workspace clippy with
  `-D warnings`; executor clippy with driver/paged-attention features
  (compile-only); `cargo test --workspace --locked`; `cargo xtask arch-check`
  (79 rejected fixtures, 21 accepted, 13 rules); and `cargo xtask spec-check`
  (10 documents).
- Both required mutations were applied, failed the focused test at 2-rank
  prefill, and were restored: rank 1 using `o_proj` slice 0, and every rank
  using rank 0's `q_b_proj` rows.
- No GPU work was run; this is the host-only task. Changes are limited to the
  five allowed source/test files and this Result. The carried `.gitignore`,
  specification version, and ADRs 0034/0035 were left untouched.
- Review round 1 fix 6 validates MLA combine orders before execution: groups
  must be nonzero, unowned head groups must divide the descriptor's head count,
  and an owned group must be in range. The duplicate BF16 output arms are
  merged.
- Review round 1 fix 7 makes `build_stage_graph` refuse an MLA owner whose
  `o_proj` slice or `q_b_proj`/`kv_b_proj` row starts disagree with that
  owner's head range, using checked arithmetic. The existing MLA test now
  asserts typed refusals for both the owner/slice mismatch and heads=3 with
  groups=2; its R=2/R=4 prefill and decode bit-identity checks still pass.
- Review round 1 host gates passed: `cargo fmt --all -- --check`, workspace
  clippy, executor clippy with driver/paged-attention features (compile-only),
  `cargo test --workspace --locked`, `cargo xtask arch-check` (79 rejected,
  21 accepted, 13 rules), and `cargo xtask spec-check` (10 documents). The
  focused MLA test passed. No GPU work was run.
- Review round 2 anchors every owner check to the source node descriptor:
  source head count and dimensions determine the expected full `o_proj` width,
  local width and offset, plus the complete `q_b_proj` and `kv_b_proj` row
  ranges. The same MLA test truncates rank 0 to a self-consistent one-head
  partition and asserts `build_stage_graph` refuses it.
- Review round 2 host gates passed: formatting, workspace clippy, executor
  driver-feature clippy (compile-only), the full locked workspace tests,
  `arch-check` (79 rejected, 21 accepted, 13 rules), and `spec-check` (10
  documents). The focused MLA test passed; no GPU work was run.
- Review round 3 adds `mla_rank`, the single checked derivation of rank-local
  MLA parameters, q_b/kv_b ranges and the o_proj slice. Lowering uses that
  result directly. With a rank part, `build_stage_graph` requires an owned
  order and exact equality with the derived entries, rejects q_a/kv_a/norm
  row or weight-slice metadata, then rewrites the layer while retaining the
  source layer in `state_layers`.
- The existing MLA test keeps its three refusal repros and adds the changed
  `rms_norm_eps` and removed-order repros. All five refusals and the R=2/R=4
  prefill/decode bit-identity checks pass.
- Review round 3 host gates passed: formatting, workspace clippy, executor
  driver-feature clippy (compile-only), full locked workspace tests,
  `arch-check` (79 rejected, 21 accepted, 13 rules), and `spec-check` (10
  documents). No GPU work was run.
