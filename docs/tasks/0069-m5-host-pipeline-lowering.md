# Task 0069 — host pipeline lowering with microbatched prefill

Status: **open** (coordinator, 2026-09-24, under the owner's auto-mode
delegation). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0069, M5 plan **slice 5**, first task. Slice 5 is three tasks:
  - **0069:** host pipeline lowering and microbatched prefill;
  - **0070:** pipeline execution on the three GPUs;
  - **0071:** the combined TP2 + TP1 plan and its report.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `73e7e5c`.
  Preserve the carried `.gitignore`, `docs/evidence/specification-version.md`
  and ADRs 0034 and 0035. **Stage explicit paths only**; never `git add -A`
  or `commit -a`.
- **Host-only. No GPU work.**
- **Clauses served:**
  - M5 exit: "same dense and MoE graph definitions execute … **PP** without
    model edits". This is the host proof; task 0070 is the device proof.
  - roadmap M5.3: "PP with uneven stages, microbatch-aware prefill".
  - document 04: "Prefill microbatches can overlap stage work only while
    causal/state dependencies are preserved; test later-token visibility
    and ordering across stages".

## Facts established before writing (coordinator, 2026-09-24)

- **Pipeline parallelism changes no arithmetic.** Every node runs whole, on
  exactly one stage. A pipelined result is therefore **bit-identical** to
  the unsplit graph, with no declared order. The risks are state ownership
  (each stage owns its layers' KV), handoffs, and microbatch ordering.
- **Strata** (`src/engine/placement.cpp`, `assign_contiguous_blocks`) cuts
  contiguous layer blocks and reports `cross_device_activation_hops`. It
  has no microbatch overlap: its GLM-5.3 doc says PCIe systems "use a
  contiguous pipeline schedule", and its wavefront design is documented as
  not implemented. The ordering rule here therefore comes from document 04,
  not from Strata.
- **`build_stage_graph`** (`moxie-plan/src/tensor_parallel.rs`) already
  handles pipeline stages.
  - With `part = None` it rebuilds `nodes` as a standalone graph. It
    records external `reads`, and `produces`.
  - Since task 0068 it renumbers a stage's attention layers **densely from
    0**, so a stage owning layers 1–3 has local layers 0–2.
- **A stateful interpreter run returns only the graph output**
  (`StepOutput.logits`). A pipeline stage therefore hands off exactly **one**
  activation: its output. `tokens` and `positions` are graph inputs that
  every stage reads directly, so they are not handoffs.
- **A route table cannot be a stage input.** `GraphBuilder::finish` refuses
  a route-valued step input. So a cut between `Route` and its `ExpertMlp`
  must be a typed refusal before any graph is built.
- **The host harness** (`moxie-cli/tests/tensor_parallel.rs`) provides the
  pieces to reuse:
  - `fixture()`: the dense Shape A variant, 6 layers, vocab 12;
  - `stage_graph`, `bind` and `bits`;
  - `unsplit_logits`: the whole-graph reference over `STEPS = [0..5, 5..6,
    6..7]`;
  - the stateful stage-run pattern: `state.append_prompt(ROOT, rows)`, then
    `interpreter.run(…)`.
- **Gemma weight names** are `{name}.{layer}` (`gemma4.rs`, `weight()`,
  around line 850). The first node of layer `L` is therefore the lowest node
  index consuming a weight whose name ends in `.L`.

## Bounded deliverable

- `lower_pipeline(graph, cuts)` splits a graph at node indices into
  contiguous stages. Each boundary carries exactly one activation. Anything
  else is refused with a typed error.
- `wavefront(stages, microbatches)` returns a production microbatch
  schedule. Its invariant: `(s, m)` comes after `(s-1, m)` and after
  `(s, m-1)`.
- A host test runs the dense and the routed Gemma graphs through 3
  **uneven** stages (layers `{0}`, `{1,2,3}`, `{4,5}`). The prefill is split
  into microbatches `[0..2, 2..5]` in wavefront order, followed by two
  decode steps. The logits must be **bit-identical** to the unsplit graph.
- **Non-goals:**
  - no devices (task 0070);
  - no automatic cut choice (the plan-comparison tool in slice 6 ranks
    candidates);
  - no TP inside a stage (task 0071);
  - no concurrency;
  - no model edits.

## Numbered changes

1. **New `crates/moxie-plan/src/pipeline.rs`**, exported from `lib.rs`:
   ```rust
   pub struct PipelineLowering { stages: Vec<Range<usize>>, handoffs: Vec<ValueId> }
   #[derive(Debug, Clone, PartialEq, Eq)]
   pub enum PipelineRefused {
       /// `cuts` is empty, unsorted, repeated, 0, or >= the node count.
       Cuts,
       /// A boundary is crossed by a value that is not a float activation.
       NonActivation { value: ValueId },
       /// A boundary is crossed by more than one activation.
       Handoffs { boundary: usize, values: Vec<ValueId> },
   }
   pub fn lower_pipeline(graph: &Graph, cuts: &[usize]) -> Result<PipelineLowering, PipelineRefused>
   pub fn wavefront(stages: usize, microbatches: usize) -> Vec<(usize, usize)>
   ```
   - Fields are **private**, read through getters `stages()` and
     `handoffs()`. `lower_pipeline` is the only constructor.
   - The stages are `[0..c0, c0..c1, …, c_last..n]`.
   - For each boundary `b` (between stage `b` and `b+1`), the crossing
     values are the values produced by a node in stages `≤ b` and consumed
     by a node in stages `> b`, or equal to `graph.output()`. Graph inputs
     and weights are excluded.
   - Refuse `NonActivation` for any crossing value whose role is not
     `ValueRole::Activation(_)`.
   - Refuse `Handoffs` unless there is exactly one crossing value, and it
     is the output of the last node of stage `b`. So a value can never skip
     a stage.
   - `handoffs[b]` is that value.
   - `wavefront(S, M)`: for `w` in `0..S+M-1`, for `s` in `0..S` ascending,
     emit `(s, w - s)` when `0 ≤ w - s < M`.
   - Add **one** unit test in `pipeline.rs`: for `S` in 1..=4 and `M` in
     1..=4, every pair appears exactly once, and each `(s, m)` appears
     after `(s-1, m)` and after `(s, m-1)`.
   - Also give `PipelineRefused` `Display` and `std::error::Error` impls,
     as `TensorParallelRefused` has them.

2. **`crates/moxie-cli/tests/tensor_parallel.rs`:** add **one** test,
   `pipeline_stages_are_bit_identical_with_microbatched_prefill`, plus one
   helper, `first_node_of_layer(graph, layer) -> usize` (the rule in the
   Facts).
   - **Fixtures:** the existing `fixture()` (dense), and a routed one. The
     routed one is `gemma::Shape::C.config()` with `moe.experts = 8` and
     `vocab = 12`, built with `build_with_config`, with no router crafting.
   - **Cuts:** `[first_node_of_layer(g, 1), first_node_of_layer(g, 4)]`.
   - **Reference:** `unsplit_logits`. Pass it a lowering-free call, or
     inline the plain `Interpreter::run` form if `unsplit_logits` needs a
     TP lowering; say which in the Result.
   - **Pipeline run:**
     - Each stage is `stage_graph(&g, &weights, range, None, None)` and
       owns a `SequenceState` and a
       `KvCache::for_branch(stage.graph.attention_layers().len(), …)`.
     - **Prefill** uses microbatches `[0..2, 2..5]`. Run the pairs in
       `wavefront(3, 2)` order. Each microbatch has its own value table
       holding its tokens, positions and handoffs. Each stage run does
       `state.append_prompt(ROOT, rows)`, then `interpreter.run`, and stores
       `out.logits` under `handoffs[s]`, or as the step output for the last
       stage.
     - The prefill logits are microbatch 0's rows, then microbatch 1's.
     - **Decode:** steps `5..6` and `6..7`, one microbatch each, through
       the stages in order.
     - Assert `bits == unsplit` for all three steps, for both fixtures.
   - **Refusals:** add rows to `unsupported_partitions_are_refused`:
     - cuts `[]`, `[0]` and unsorted cuts give `Cuts`;
     - a cut at `route_index + 1` of the routed fixture (the route crosses)
       gives `NonActivation`;
     - a cut at the dense fixture's first `Attention` node index (Q, K and
       V all cross) gives `Handoffs`.

## Allowed files

- `crates/moxie-plan/src/pipeline.rs` (new)
- `crates/moxie-plan/src/lib.rs` (exports)
- `crates/moxie-cli/tests/tensor_parallel.rs`
- This task's Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**Mutations**, each applied, run, shown failing and restored:
1. `wavefront` iterates `s` in **descending** order within a wave, **and**
   emits microbatches in descending order: `(s, M-1-(w-s))`. The unit test
   fails on ordering. Show that the host test also fails, or is refused:
   stage 0 would append positions 2..5 before 0..2.
2. `lower_pipeline` accepts two crossing values (drop the `Handoffs`
   check). The `Handoffs` refusal row fails.

**Stop conditions:**
- A Gemma layer boundary is crossed by more than one activation. That
  means the one-handoff rule is wrong for the real graph: report the values
  and do not relax the rule yourself.
- The pipelined logits are not bit-identical. Report the step, fixture and
  element.
- A mutation survives.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Pending.
