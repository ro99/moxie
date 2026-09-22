# Task 0058 — dense TP on the host: the MLP and the vocabulary projection

Status: **accepted** (coordinator, 2026-09-22, under the owner's auto-mode
delegation). Built by Codex `luna`; reviewed by Codex `sol` over three
rounds, round 3 ACCEPT with no findings. The coordinator re-ran the focused
tests, `arch-check` and `spec-check`.

## Identity and authority

- Task0058, M5 plan slice 1 ("Dense TP on the host"; see the [M5
  ledger](../handovers/2026-09-22-m4-closure-to-m5.md)). Builder Codex `luna`
  (max, `/ponytail:ponytail`); reviewer Codex `sol` (read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus `coordinator`. Owner
  accepts.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `687c389`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035).
- Requirements:
  - Roadmap M5.1: legal partition semantics for all implemented dense ops;
    weights split along the input axis addressed through the canonical format.
  - Roadmap M5.2: "column/row linears … global … vocabulary operations …
    Compare to single-rank reference."
  - [ADR 0036](../decisions/adr/0036-tp-reductions-are-exact-by-declared-order.md):
    TP reductions are exact. FP32 partials per shard are combined in FP32 in
    a declared order and rounded to BF16 once, and the single-rank path runs
    the same order.
  - Document 04, lines 49–57: bias and residual are applied exactly once;
    non-divisible dimensions are refused explicitly.
- O6/O7 are open: no timing.

## Facts established before writing (coordinator, 2026-09-22)

**Strata** (read-only, `/home/rodrigo/Developer/strata`):
- `src/models/deepseek/deepseek_rank_local_weights.cpp` lines 234–236 and
  351–355: `w1`/`w3` (gate/up) are `ContiguousRows`, split by output; `w2`
  (down) and the attention output's second projection are `StridedColumns`,
  split by input. This is the standard layout.
- `src/models/deepseek/deepseek_rank_shard.cpp` `build_slices`: a
  strided-column shard is one slice per output row, holding the rank's
  contiguous run of columns. Scale tensors follow the same pattern. Refusals
  (lines 430–440): an input width that the world size does not divide, or a
  shard that splits a quantization block or group along the input width.
- `docs/dsv4-rank-local-architecture.md`, "Collectives": data partials cross
  ranks as FP32 (`ncclSum`, `ncclFloat`), one BF16 rounding after the combine.
  The output head is split and its logits **all-gathered**, not reduced.

**Moxie:**
- `moxie-oracles/src/linear.rs`: task 0003 fixed `Linear`'s declared order as
  **sequential ascending `k`**, FP32 accumulator, one BF16 rounding by the
  caller. `VocabProjection` does not round. An input-axis split changes that
  declared order. ADR 0036 authorizes the change, but the declared split must
  live in a contract the host oracle and the interpreter honour. Today it
  lives nowhere.
- `OpParams::Linear { in_features, out_features, bias }` is constructed at 31
  sites. The M5 exit requires "without model edits".
- Task 0056's lowering (`moxie-plan/src/tensor_parallel.rs`) has
  `Stage::Replicated` and `Stage::HeadLocal { gather }`, and runs everything
  outside the attention head chain replicated. Its harness is
  `moxie-cli/tests/tensor_parallel.rs`.
- Task 0054's `moxie_executor::shard_weight_ranges` refuses `RowShardable`
  because the shard is strided.
- The partition labels are inaccurate. `Rope` claims `ColumnShardable` but
  is legal only at head boundaries. The lowering reads labels only to refuse
  `NotDetermined`.

## Bounded deliverable

- **Outcome:** extend the task 0056 lowering to the rest of the dense layer
  and to the vocabulary projection. The reduced Gemma 4 graph, on the host at
  R ∈ {2, 4}, over prefill plus decode, must be **bit-identical** to its
  single-rank reference.
  - **MLP:** `gate`/`up` split by output columns; the GLU computed locally
    on each rank's columns; `down_proj` split along the input axis. Each rank
    produces an FP32 partial; the partials are combined in ascending rank
    order in FP32 and rounded to BF16 once, per ADR 0036.
  - **Vocabulary projection:** split by output columns and gathered.
  - **`o_proj`:** extend it to the input-axis split with the same exact
    reduction, if that falls out of the same mechanism. If not, say so and
    keep the task 0056 gather.
- **Declared split order:** the single-rank reference for a plan with split S
  must run the ADR 0036 order: S contiguous `k` blocks, each summed ascending
  from zero in FP32, blocks combined ascending in FP32, rounded once. The
  default S = 1 is exactly today's order, and every existing test keeps its
  bits. The split must come from the plan, not from the model.
- **Strided weight addressing:** add input-axis addressing next to task
  0054's column ranges, following Strata's per-row slices and refusals
  (non-divisible input width; a split that cuts an `AffineDescriptor` group
  along `k`). Keep the representation compact rather than a `Vec` of every
  row, if that is cheaper.
- **Labels:** every `partition_rule()` label must be true or narrower than
  the truth. Correct `Rope`'s. A label the lowering contradicts is a defect.
  If a label has no reader and no truthful value, propose removing it; do not
  invent one.
- **Refused:** a biased input-axis `Linear` (bias must be applied once;
  supporting it is not needed here); `Embedding` stays replicated
  (vocabulary-parallel embedding needs its own reduction and is out of
  scope); MLA and routed ops remain refused (slice 4).
- **Non-goals:** no GPU (slice 2), no model-crate edits, no new collective
  graph ops, no timing.

## Phase 1 — design proposal before code

Send a `DECISION` report of 40 lines or fewer covering:

1. **Where the declared split lives.** Consider an `OpParams::Linear` field
   (31 sites; how do models avoid edits?), a plan-level annotation the
   interpreter reads, or another way. How does the host oracle stay an
   independent statement of the order, rather than a copy of the TP code
   path?
2. **Stage representation for a reduction.** How a rank-local stage ends in
   FP32 partials plus a combine, next to task 0056's gather.
3. **The strided-address representation** and where it lives relative to
   `shard_weight_ranges`.
4. **The label decision.**
5. **Files, and any new crate edge.**

The coordinator answers before implementation starts. If a contract fact is
wrong, say so in that report.

## Contract

- **Equations:** unchanged, except that the declared summation order of an
  input-axis-split `Linear` becomes ADR 0036's order, and the reference runs
  the same order.
- **State:** unchanged from task 0056; per-rank KV caches.
- **Oracle:** the single-rank reference running the declared split,
  bit-for-bit. Also: S = 1 must reproduce today's reduced Gemma logits
  exactly, so the declared-order change breaks no existing result.

## Acceptance

- Host lanes pass: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` and
  `cargo xtask spec-check`.
- The bit-identity test extends task 0056's test: R ∈ {2, 4}, prefill and
  decode, the MLP and vocabulary splits active.
- One refusal table: add the new refusals to task 0056's.
- **Mutations,** each run and restored:
  - Combine the partials in descending order.
  - Round a partial to BF16 before the combine.
  - The reference ignores the declared split (runs S = 1).
  - A strided slice uses the wrong row stride.
  - The GLU runs on gathered rather than local columns.

  Each must fail a test.
- One test per invariant. Task 0056's harness grows; it is not duplicated.
- **Stop conditions:**
  - The declared split cannot live anywhere without model edits.
  - S = 1 changes any existing bits.
  - Bit-identity fails for a reason that is not a defect.

## Result, filled after work

- Design decision (phase 1) and coordinator answer: the declared split is a
  node-keyed `moxie-graph::LinearReductionOrder` sidecar, so the 31 model-side
  `OpParams::Linear` constructors remain unchanged. The interpreter gets
  additive sidecar-aware entry points while `run()` remains the S=1 path;
  `moxie-oracles::linear` states the ordered arithmetic independently. The
  coordinator approved `Stage::Local { nodes, join }`, with `Gather` for the
  head output and F32 vocabulary output and `Reduce` for `down_proj` and
  `o_proj`; biased row-parallel linears are refused. The coordinator also
  approved compact `LinearInputSlice` and per-component strided views beside
  `shard_weight_ranges`, with canonical affine geometry and group validation,
  and corrected labels: `RmsNorm` group 1 is `Replicated`, grouped `RmsNorm`
  is `HeadAligned` like `Rope`, and `Linear` remains `ColumnShardable`.
- Changed owners and consumers; source commit: `moxie-graph` owns the
  partition labels and sidecar value types; `moxie-plan` owns rank-local
  stages, joins, slices and declarations; `moxie-interp` consumes the
  declarations; `moxie-oracles` owns the independent ordered linear
  reference; `moxie-executor` owns canonical BF16/affine strided addressing;
  and the existing `moxie-cli` tensor-parallel harness is the consumer
  fixture. `reference_graphs.rs` covers the additive interpreter contract.
  No model crate or Cargo dependency changed; no source commit was made.
- Commands; passed / failed / skipped: passed `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` (79 rejected
  and 21 accepted fixtures; 13 rules exercised), `cargo xtask spec-check` (10
  documents), and `git diff --check`. No required acceptance gate failed or
  was skipped. GPU/device execution and timing remain intentionally skipped
  as non-goals. Round 3 also passed the focused tensor-parallel and affine
  group-refusal tests before the full host-gate rerun.
- Mutation results and restoration: descending partial combination failed
  `moxie-oracles::linear::tests::declared_block_order_is_load_bearing`; early
  BF16 rounding failed
  `moxie-oracles::linear::tests::ordered_partials_remain_fp32_until_the_combine`;
  forcing the reference to S=1 failed
  `moxie-interp/tests/reference_graphs.rs::a_declared_linear_split_reaches_the_host_interpreter`;
  adding one byte to the row stride failed
  `moxie-executor::affine_linear::tests::row_sharding_exposes_one_compact_run_per_affine_component`;
  and making GLU use the global rather than local width failed
  `moxie-cli/tests/tensor_parallel.rs::dense_tp_is_bit_identical_to_the_unsplit_graph_at_prefill_and_decode`
  at graph construction. Each mutation was restored and the gates were rerun.
  S=1 preserves the existing reduced-Gemma logits: `linear_row` delegates to
  the ordered oracle with one block, and the unchanged
  `moxie-cli/tests/gemma.rs` tests `both_reduced_geometries_generate_through_the_shared_service`,
  `paged_and_dense_logits_are_bit_identical_across_pages_and_decode`, and
  `the_cli_selects_the_reduced_shapes_and_agrees_with_the_service` all pass.
- Review trail (round 2, sol): the affine row-shard path now rejects
  non-contiguous, non-monotone activation-order group maps globally before
  checking rank boundaries; `row_sharding_refuses_a_split_inside_an_affine_group`
  covers both ranks of the alternating Int8 repro. MLP gate/up/GLU outputs
  now reject consumers outside their local four-node stage; the refusal table
  covers an extra Linear consuming the gate. The TP fixture now compares the
  split result only with its declared S=2/4 reference, and the one-use
  `JoinOutput` trait is inlined. Sol's other findings remain closed: oracle
  order, S=1 compatibility, truthful labels, no model edits or crate edges,
  and contiguous-group offsets were rechecked.
- Review trail (round 3, sol): one generic local-stage boundary check now
  covers attention, MLP and vocabulary stages; any non-join value read by a
  node in another stage or used as `graph.output()` is refused. The
  pattern-specific attention and MLP escape checks were removed, and the
  refusal table adds the graph-output gate repro. The redundant
  `validate_group_slice` helper and call were deleted; affine group ownership
  remains checked globally by `validate_group_ownership`.
- Remaining obligations: the current fixture is insensitive to summation order
  (S=2 bits equal S=1), so the end-to-end test cannot detect a wrong combine
  order; the device slice needs an order-sensitive fixture. The device TP
  slice, MLA and routed-op lowering, and vocabulary-parallel embedding remain
  out of this task. No performance claim is made.
