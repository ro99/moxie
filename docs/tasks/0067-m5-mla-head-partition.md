# Task 0067 — MLA per-weight head partition on the host

Status: **open** (coordinator, 2026-09-23, under the owner's auto-mode
delegation). Builder Codex `luna`; reviewer Codex `sol`.

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

- Pending.
