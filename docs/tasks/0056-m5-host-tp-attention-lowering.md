# Task 0056 — host-only tensor-parallel lowering of the attention sublayer

Status: **accepted** (owner, 2026-09-22). Built by Claude Opus `builder`; reviewed
by Codex `sol`; round 2 ACCEPT. The coordinator re-ran the focused tests
and `arch-check`.

## Identity and authority

- Task0056, the first M5.2 slice. It replaces the queued M5.1-c/M5.1-d
  labelling tasks (owner, 2026-09-22: "restructure the queue this way").
  Builder Claude Opus session `builder` (`/ponytail:ponytail`; replaced
  Codex `luna`, out of credits, 2026-09-22); reviewer Codex `sol`
  (read-only, `/ponytail:ponytail-review`; a Claude Opus `reviewer` was
  briefly assigned while sol was out of credits and did not review); coordinator
  Claude Opus. Owner accepts.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `8d19da3`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035); do not stage or edit it.
- Requirements: roadmap M5.2, "Compare to single-rank reference on small
  graphs", for column/head ownership. Roadmap M5 exit: "same dense … graph
  definitions execute single GPU, TP and PP without model edits." Document 04
  lines 49–57: head ownership, GQA KV-head replication when ranks exceed KV
  heads, output reduction, bias/residual applied exactly once, and
  "non-divisible dimensions use checked padding or explicit unsupported
  combinations". Document 02: `moxie-plan` lowers a semantic graph and never
  allocates or performs I/O; the executor owns collectives.
- Carried obligation closed here: reject query-head/rank combinations that
  do not divide (review-task-0053-round-1).
- O6/O7 are open: no timing and no performance claim. No owner gate is
  reached.

## Facts established before writing (coordinator, 2026-09-22)

- **Partition rules are stored but never read.** The graph builder copies
  `partition_rule()` into each node contract (`graph.rs`), and only tests
  read it back. This task is their first consumer.
- **Head structure spans a chain of ops, not one.** Gemma 4's attention block
  (`crates/moxie-models/src/gemma4.rs`, around lines 600–720) runs `q/k/v`
  `Linear` → per-head `RmsNorm { group: heads | kv_heads }` (partition rule
  `Replicated`) → `Rope { heads, HalfSplit }` → `Attention { heads, kv_heads }`
  → `o_proj` `Linear`. Global layers have fewer KV heads than sliding layers,
  and no value projection (`the_global_layers_carry_no_value_projection…` in
  `crates/moxie-cli/tests/gemma.rs`).
- **Weights are row-major `[out, in]`.** A head-aligned shard of a projection
  is a contiguous row range. `o_proj`'s input-axis split is strided, and
  task 0054 refuses it.
- **The interpreter runs a whole graph as one step.** `Interpreter::run` takes
  one `Graph` (single `output`), a `SequenceState` branch and a `KvCache` that
  must cover exactly the layers the graph writes, and advances the branch
  frontier. Running rank-local pieces with a gather between them does not
  fit that model as it stands.
- **Precedents to reuse:** `crates/moxie-cli/tests/gemma.rs` already runs a
  reduced Gemma 4 graph on the interpreter (`dense_logits`) and rebuilds a
  graph with transformed op parameters (`rebuild`). `moxie-plan` may depend
  only on `moxie-types` and `moxie-graph` (`arch-check` forbids
  `moxie-format`).

## Bounded deliverable

- **Outcome:** a pure lowering in `moxie-plan`. It takes a graph and a rank
  count and returns, for each rank:
  - the op parameters that rank executes, for example `Attention { heads:
    local, kv_heads: local }`, `RmsNorm { hidden: local, group: local }`, and
    `Rope { heads: local }`;
  - which logical rows of each weight that rank holds: a head-aligned row
    range, or the whole tensor where replicated;
  - where the per-rank results are combined. In this slice that is one
    concatenation of the attention output in global head order (the gather),
    after which the rest of the layer runs replicated.

  A host-only test runs each rank's part on the existing interpreter and
  gathers on the host. Logits must be **bit-identical** to the unsplit graph
  over a multi-row prefill followed by at least one decode step.
- **Head assignment:** query heads are split contiguously, `heads / R` per
  rank.
  - If `kv_heads % R == 0`, KV heads are split contiguously too.
  - Else if `R % kv_heads == 0`, rank `r` holds the single KV head
    `r * kv_heads / R`, replicated across the ranks that share it.
  - Anything else, including `heads % R != 0`, is a typed refusal.

  Rank-local GQA grouping must equal the global grouping: query head `h` uses
  KV head `h / (heads / kv_heads)`.
- **Scope of the graph:** dense reduced Gemma 4 (no routed experts), with at
  least one layer where `kv_heads % R == 0` and one where `R > kv_heads`,
  for `R = 2` and `R = 4`. Everything outside the head chain runs
  replicated. `o_proj` runs replicated after the gather; its input-axis
  split stays with the strided-addressing M5.2 obligation.
- **Refused with typed errors:** `MlaAttention`, `Route`, `ExpertMlp`,
  `Combine`, and any `NotDetermined` node.
- **Owners:** the lowering lives in `moxie-plan`. The host multi-rank runner
  and host gather are test harness code; put them where `moxie-cli`'s
  existing Gemma test already composes models and the interpreter, or justify
  another place. The harness is **temporary**: it is replaced by executor
  collectives in the device TP2 slice. Name that expiry in Result.
- **Non-goals:** no GPU, no NCCL/P2P, no rank-group type in the executor, no
  timing. No MLA (a later slice), no row-parallel `o_proj`, no `moxie-state`
  KV-shard addressing beyond what per-rank caches need. No model-crate edits.
  No new semantic ops for collectives: collectives are a plan property, not
  graph mathematics. No change to task 0054's `shard_weight_ranges`.
  Converting row ranges to canonical bytes is the device slice's job.

## Phase 1 — design proposal before code

Before any code, send a `DECISION` report of about 40 lines or fewer with:

1. **Stage representation.** How the lowering expresses the sequence of
   rank-local work, the gather, and replicated work, and how a
   multi-layer graph repeats it.
2. **Host execution within the interpreter's one-step state transaction.**
   Choose one of the following and say why:
   - an additive interpreter entry point that runs a stage within one
     transaction;
   - per-rank complete graphs in which the gather is realized differently;
   - another way.

   Changing the semantics of `Interpreter::run` or `SequenceState` is out
   of bounds.
3. **Parameter rewriting.** How per-rank params are produced without editing
   the model, and whether the `rebuild` precedent serves.
4. **Per-rank KV caches.** How each rank's `KvCache`/`SequenceState` is
   formed.
5. **The files you expect to touch, and any new crate edge.**

The coordinator answers before implementation starts.

## Contract

- **Equations:** unchanged. Splitting whole heads changes no per-head
  arithmetic, the gather concatenates without summing, and `o_proj` stays
  replicated, so bit-identity is the correct gate. Do not loosen it. If
  results differ, find the cause.
- **State:** each rank appends only its local KV heads. Replicated KV heads
  exist on every rank that needs them.
- **Failure:** a refused lowering creates no partial per-rank artefact. Test
  cancellation only if the chosen host execution adds a new stage boundary
  where cancellation can land.
- **Oracle:** the unsplit graph on the same interpreter. The test also
  asserts that every rank's replicated stages agree.

## Acceptance

- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` and
  `cargo xtask spec-check` pass. Any new crate edge is declared and justified.
- The bit-identity test covers `R ∈ {2, 4}`, both KV cases, prefill and
  decode.
- One refusal test covers non-dividing heads, a non-dividing KV/rank pair,
  and an MLA or routed graph.
- **Mutation proof, run and restored by the builder:**
  - Split a KV head across ranks instead of replicating it.
  - Split query heads non-contiguously, or leave a rank's rewritten
    `Attention.kv_heads` global.
  - Skip rewriting the per-head `RmsNorm` group.

  Each must fail the bit-identity test. Record the results and the
  restoration.
- **Tests:** one test per invariant. Do not re-test graph construction or
  fixture literals.
- **Stop conditions:**
  - The design needs an interpreter or `SequenceState` semantic change.
  - Bit-identity cannot hold for a reason that is not a defect.
  - A model crate would need editing.

  In each case, report instead of working around it.

## Result, filled after work

- **Design decision (phase 1) and coordinator answer:**
  - The approved design runs each stage as its own small graph, with no
    interpreter change. Replicated stages go through the existing
    `Interpreter::run_stateless`, which builds its own empty `SequenceState`
    and zero-layer cache and never touches the caller's. Head-local stages go
    through the unchanged `Interpreter::run`, with one `SequenceState` and one
    one-layer `KvCache` per (rank, attention layer).
  - **The harness does not make a rank step atomic.** One rank step is one
    transaction per layer, so a failure part-way through a step would leave
    earlier layers appended. The coordinator recorded the M5.2 obligation that
    the device slice makes a whole rank step one transaction.
  - Fact corrections, accepted:
    - Shape A has no layer whose KV heads divide over 4 ranks. The test
      therefore uses Shape A through `gemma::build_with_config` with 8 query
      heads, 4 sliding KV heads and 1 global KV head. At both R=2 and R=4 the
      sliding layers split their KV heads and the global layer replicates its
      one.
    - `GraphBuilder::finish` requires attention layers numbered densely from 0,
      so the harness renumbers each stage subgraph's attention to layer 0.
    - `k_proj` is sharded as a KV projection, because on global layers it
      feeds V.
  - One further fact, found during implementation: a global layer's per-head
    K norm has `group == kv_heads == 1`. The chain therefore takes every
    `RmsNorm` between a projection and attention as per-head, not only those
    with `group > 1`, and then checks that `group` equals the axis's head
    count.
- **Changed owners and consumers; source commit:** uncommitted on base
  `1090e80`.
  - `crates/moxie-plan/src/tensor_parallel.rs` (new) holds
    `lower_tensor_parallel`, `Stage`, `RankPart`, `TensorParallelLowering` and
    `TensorParallelRefused`. They are re-exported from
    `crates/moxie-plan/src/lib.rs`, and `moxie-plan`'s dependencies are
    unchanged.
  - `crates/moxie-cli/tests/tensor_parallel.rs` (new) is the temporary
    harness, with two tests.
  - `crates/moxie-cli/Cargo.toml` adds `moxie-plan` as a dev-dependency only;
    `Cargo.lock` records it. `arch-check` does not govern dev-dependencies.
  - The first consumer of `partition_rule()` outside tests is the lowering's
    fail-closed refusal of `NotDetermined`.
- **Commands; passed / failed / skipped:**
  - `cargo fmt --all -- --check` passed.
  - `cargo clippy --workspace --all-targets --locked -- -D warnings` passed.
  - `cargo xtask arch-check` passed.
  - `cargo xtask spec-check` passed.
  - `cargo test --workspace --locked`: passed (exit 0; 108 test binaries, 1232 passed, 0 failed, 0 ignored; round 2 log /tmp/claude-1000/-home-rodrigo-Developer-moxie/75e03800-b1df-4452-b029-94f209bf853d/scratchpad/test-workspace-r2.log).
  - The bit-identity test covers R in {2, 4, 8}: a 5-row prefill, then two
    1-row decodes. R=8 was added in round 2 so that a replicated KV head with
    `kv_heads > 1` is exercised (each of the 4 sliding KV heads over 2 ranks). It compares logit bits and asserts at every replicated stage that
    every rank's values agree bit for bit.
  - The refusal test's table covers 4 query heads over 3 ranks (`Heads`), 3
    global KV heads over 2 ranks (`KvHeads`), the routed Shape C graph (`Op`)
    and a biased query projection (`HeadChain`).
  - Nothing was skipped. There was no GPU work and no timing.
- **Mutation results and restoration.** Each mutation was applied to
  `tensor_parallel.rs`, followed by `cargo test -p moxie-cli --test
  tensor_parallel head_split`, then a restore from a copy verified with
  `cmp`. Every mutation fails the bit-identity test.
  - The shared KV head's rows split across the ranks that share it, instead
    of replicated, fails when the stage subgraph is validated: `linear: input
    1 axis 0 must be 32, got Const(16)`.
  - Query heads assigned out of rank order fail on the logits comparison
    itself: the R=2 logit bits differ.
  - A rank's `Attention.kv_heads` left global fails at subgraph validation:
    `attention: input 1 axis 1 must be 64, got Const(32)`.
  - The per-head `RmsNorm` group not rewritten fails at subgraph validation:
    `rms_norm: input 1 axis 0 must be 8, got Const(16)`.
  - Three of the four are caught by the graph builder's shape validation of
    the rank's stage, before any logits exist. Only the reordering reaches
    the bit comparison, because it is the only one of the four that stays
    shape-consistent.
  - Round 2 (review-task-0056-round-1). F1: the harness now takes each
    head-local stage's output and gather point from `Stage::HeadLocal.gather`
    rather than assuming the stage's last node. Mutating the lowering to
    gather at `nodes[attention].inputs[0]` (the query RoPE output, same
    width) fails: the rank graphs return the RoPE output, the gathered value
    is stored under the RoPE id, and the next stage's read of the real
    attention output panics (`no entry found for key` in `bind`). F2: with
    R=8, mutating the replicated KV head index to `rank % kv_heads` fails the
    R=8 logit bit comparison. Both were restored and checked with `cmp`.
- **Review trail (sol).** Round 1: F1 major, the harness never read
  `Stage::HeadLocal.gather`, so a wrong gather point passed; a coverage gap,
  every replicated KV case had one KV head. Both were repaired in round 2 in
  the harness only, since the production lowering was correct. Round 2:
  **ACCEPT**, with the gather-point failure confirmed load-bearing because
  value ids are unique. Sol also confirmed, by arithmetic, the local-to-global
  GQA mapping in the split and replicated cases, and that head-chain
  traversal admits only `Linear`/`RmsNorm`/`Rope`.
- **Temporary harness and its expiry.** The multi-rank runner, the host
  column gather and the `stage_graph` subgraph helper all live in
  `crates/moxie-cli/tests/tensor_parallel.rs`. They are test-only and are
  marked temporary in the file's header. They expire when executor
  collectives land in the device TP2 slice.
- **Remaining obligations:**
  - MLA lowering (a later slice).
  - Row-parallel `o_proj`, whose strided input-axis addressing M5.2 still
    owes.
  - Device collectives and a rank-group type in the executor.
  - One transaction per whole rank step on device (above).
  - `ExpertMlp`/`Combine` partitioning (M5.4).
  - Converting row ranges to canonical bytes (the device slice).
