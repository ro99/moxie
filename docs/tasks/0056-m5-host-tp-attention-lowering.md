# Task 0056 — host-only tensor-parallel lowering of the attention sublayer

Status: **proposed**.

## Identity and authority

- Task0056, the first M5.2 slice. It replaces the queued M5.1-c/M5.1-d
  labelling tasks (owner, 2026-09-22: "restructure the queue this way").
  Builder Claude Opus session `builder` (`/ponytail:ponytail`; replaced
  Codex `luna`, out of credits, 2026-09-22); reviewer Claude Opus session
  `reviewer` (read-only, `/ponytail:ponytail-review`); coordinator
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

- Design decision (phase 1) and coordinator answer:
- Changed owners and consumers; source commit:
- Commands; passed / failed / skipped:
- Mutation results and restoration:
- Temporary harness and its expiry:
- Remaining obligations (MLA slice, row-parallel `o_proj`, device collectives):
