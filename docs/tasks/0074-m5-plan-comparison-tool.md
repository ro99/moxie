# Task 0074 — a deterministic plan comparison tool

Status: **accepted** (coordinator, 2026-09-24, under the owner's auto-mode
delegation), after sol's review rounds R1 and R2. Builder Codex `luna`;
reviewer Codex `sol`.
- Three amendments came from the builder's DECISIONs: the `HostExperts`
  variant, tuple stage bounds, and oracle parameters. All three were gaps in
  the contract, where API details were not checked against the code before
  writing.
- R1 (HIGH): the KV read was multiplied by rows, which inflated every
  prefill estimate (a single 3090 showed about 36 s at a 32k prompt).
  Fixed, and all four evidence runs were regenerated.
- R2 (LOW): a units label (ms, not s). The coordinator fixed it.
- The owner flagged task numbers in code names. The labels were renamed,
  and a sweep of the older occurrences goes to the milestone-end audit.
- **Size, carried to the milestone-end ponytail audit:** `compare.rs` is 832
  lines, against an estimate of 250–350.

## Identity and authority

- Task0074, M5 plan slice 6, second task (0073 probes, **0074 this**, then
  the milestone-end audit and the acceptance package).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `87bd04d` (task
  0073's acceptance). Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- The tool is pure and needs no GPU. It reads task 0073's cost file.
- **Clauses served:**
  - M5.5: "a **deterministic plan comparison tool**; rank candidates by the
    **joint user workload**, not GPU utilization alone" (the M5 exit gate's
    last open clause, board T1);
  - document 02: "Diagnostics explaining rejected alternatives and
    estimated versus measured transfer/compute costs";
  - document 03: "step lower bound >= max(compute_time,
    required_transfer_time) and may approach their sum when dependencies
    prevent overlap";
  - document 04: "Balance by measured compute, weight traffic and memory,
    not equal layer counts" and "`auto` does not select TP merely because
    cards exist".

## Facts established before writing (coordinator, 2026-09-24)

- **Where it runs.**
  - `moxie-cli` may import models but not `moxie-plan`.
  - `xtask` has `moxie-plan`, the cost reader (`read_costs`, task 0073) and
    the probe, but not `moxie-models`.
  - arch-check's model-import rule already exempts `xtask` and `moxie-cli`
    by name (`archcheck.rs`, "Only the composition roots may"). Only
    `xtask`'s workspace allowlist lacks `moxie-models`.
  - Adding it there, as an optional dependency under the `cuda` feature,
    is the smallest boundary change. It is recorded here as a
    coordinator-approved allowlist edit.
- **Graphs without weights.** `moxie_models::gemma4::Gemma4Text::reduced(config,
  revision)?.compose(&oracles, rows)` builds a validated graph with no
  weight bytes, so a checkpoint-sized configuration costs only metadata.
- **Everything is derived, never declared** (engineering log, tasks
  0065–0067):
  - weight bytes come from each weight's spec (shape × BF16);
  - TP per-rank bytes come from `lower_tensor_parallel` and the per-rank
    stage graphs;
  - legal pipeline cuts come from `lower_pipeline`;
  - KV bytes come from the attention nodes;
  - link and device figures come from the measured `TopologyCosts`.
- **The cost model is weight-traffic and transfer bound.**
  - Batch-one decode reads every resident weight each step.
  - There is **no measured compute probe** yet (M6 owns kernel throughput),
    so prefill is modelled with weight reads plus transfers only, which
    underestimates a compute-bound prefill. This is a stated ceiling, not
    hidden.
  - Host-owned experts have no measured host compute cost, so they are
    listed as rejected with that reason, not ranked.
- The pipeline handoff is host-staged in every M5 plan (tasks 0071 and
  0072). The TP pair's collectives are peer copies.

## Bounded deliverable

- A pure `compare_plans` in `moxie-plan` that:
  - enumerates the legal candidates for a graph on the measured devices;
  - estimates per-device bytes and the prefill and per-token decode times;
  - rejects what does not fit or has no measured path, with reasons;
  - ranks the rest by the joint workload's total time, deterministically.
- `cargo xtask-cuda compare-plans` prints the ranking.
- A measured evidence file.
- **Non-goals:**
  - automatic selection (M6.5);
  - compute probes;
  - executing any candidate;
  - MLA or routed-expert placement costs beyond the listed rejection.

## Numbered changes

1. **New `crates/moxie-plan/src/compare.rs`** (pure; declared and
   re-exported in `lib.rs`):
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub struct UserWorkload { pub prompt_tokens: u64, pub generated_tokens: u64 }
   #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
   pub enum CandidateKind {
       Single { device: DeviceUuid },
       Tp2 { devices: [DeviceUuid; 2] },
       /// One stage per entry; a stage is one device, or two for a TP2 stage.
       /// `(start, end)` node indices, not `Range`, so `Ord` derives (amended after the builder's DECISION).
       Pipeline { stages: Vec<(Vec<DeviceUuid>, (usize, usize))> },
       /// Routed experts on the host beside one device (amended after the builder's DECISION).
       HostExperts { device: DeviceUuid },
   }
   #[derive(Debug, Clone, PartialEq)]
   pub struct Estimate {
       pub device_bytes: Vec<(DeviceUuid, u64)>,   // weights + KV at full context
       pub prefill_ms: f64,
       pub first_decode_ms: f64,                  // at context = prompt
       pub total_ms: f64,                         // prefill + every decode step
   }
   #[derive(Debug, Clone, PartialEq)]
   pub enum Verdict { Ranked { rank: usize, estimate: Estimate }, Rejected { reason: String } }
   pub fn compare_plans(graph: &Graph, oracle: OracleId, oracles: &OracleRegistry,
                        workload: UserWorkload, costs: &TopologyCosts)
       -> Result<Vec<(CandidateKind, Verdict)>>;
   // Amended after the builder's DECISION: the composition root supplies
   // the oracle registry, as `PipelineWorkers::execute` takes it, so stage
   // graphs are built with `build_stage_graph` and a TP2 stage lowers with
   // `lower_tensor_parallel(&stage.graph, 2)`. No new production dependency.
   // A full-graph TP2 refusal (e.g. an odd vocabulary) is a `Rejected` Tp2
   // entry and does not affect the TP2-first pipeline candidates.
   ```
   - **Candidates**, in this order:
     1. `Single` on each device.
     2. `Tp2` on each unordered pair with **both** peer links in `costs`,
        and only if `lower_tensor_parallel(graph, 2)` succeeds. On a
        refusal, one `Rejected` entry carries the refusal text.
     3. `Pipeline` over **every device ordering** of all devices, with one
        stage per device.
     4. `Pipeline` with a TP2 stage first (each peer pair), then one stage
        on each remaining device, in every order of the remaining devices.
   - **Legal cuts** are the node indices `i` with `lower_pipeline(graph,
     &[i])` succeeding.
   - **Balancing:** choose the cuts greedily. Stage `k`'s share of the
     weight bytes is proportional to its devices' summed `usable_bytes`.
     Each cut is the legal cut whose prefix bytes are nearest the running
     target, strictly after the previous cut. Ties go to the lower index.
     Mark the greedy choice with `ponytail:` (no search over cut sets;
     upgrade when stage costs are measured).
   - **Bytes:**
     - weights: the sum over the stage's weights of spec elements × 2
       (BF16);
     - a TP2 rank uses the per-rank stage graphs from
       `lower_tensor_parallel` on that stage's graph;
     - KV per token per attention node: its key and value input widths × 2
       bytes. A windowed layer holds `min(context, window)` rows. A TP2
       rank holds the rank-local widths, and replicated KV heads count on
       both ranks.
   - **Fit:** `device_bytes` = weights + KV at `prompt + generated`. If it
     exceeds a device's `usable_bytes`, the verdict is `Rejected` with
     "needs N bytes on UUID, which has M usable".
   - **Times** (ms; `GB/s` as 1e9 bytes per second; `latency_us`
     included once per copy):
     - per stage, per step: the max over its ranks of
       `(weight_bytes + kv_read_bytes(context)) / memory_gbps` of that
       device;
     - a pipeline handoff: `rows × hidden × 2` bytes, as
       `latency + bytes / bandwidth` over `src → Host`, **plus** the same
       over `Host → dst`, taken sequentially;
     - a TP2 stage: for each collective in its lowering, `rows × width × 4`
       bytes over the peer link, both directions at `concurrent_gbps`, plus
       latency;
     - decode at context `c`: the sum of the stage times (dependent stages
       do not overlap) and all transfers, with `rows = 1`;
     - prefill: the same, with `rows = prompt_tokens` and `kv_read` at the
       prompt;
     - `total = prefill + Σ_{t=0}^{generated−1} decode(prompt + t)`, with
       the sum in closed form (KV reads are linear in context).
   - **Rank:** the feasible candidates by `total_ms` ascending, ties by
     `CandidateKind`'s `Ord`. Rejected candidates follow in candidate order.
     The output is identical for identical inputs.
   - **Routed graph:** if it contains `ExpertMlp` nodes, also emit one
     `Rejected` "host-owned experts: no measured host compute cost (M6)"
     entry per device, as `HostExperts { device }`. These come after the
     other candidates, in device order. Display it as `host-experts+<uuid8>`. The device candidates are still
     costed.
   - Size: about 250–350 lines.

2. **New integration test `crates/moxie-plan/tests/plan_comparison.rs`.**
   - Add `moxie-models` (and `moxie-oracles`, if composition needs it) as
     **dev-dependencies** of `moxie-plan`. arch-check exempts
     dev-dependencies, and document 02 lets integration tests import
     concrete models.
   - Compose small dense Gemma graphs with `Gemma4Text::reduced(...)
     .compose(...)`. Use a hand-written `TopologyCosts`: two 3090-like
     peers at 800 GB/s, one 5060-like device at 380 GB/s, and host links.
   - Three tests, each one invariant:
     - `the_ranking_is_deterministic_and_input_order_free`: two calls, and
       a `TopologyCosts` with shuffled `devices` and `links`, give equal
       output.
     - `the_joint_workload_changes_the_winner`: the same graph and costs,
       with a short context favouring one plan and a long context (its KV
       over one device's `usable_bytes`) that rejects `Single` and ranks a
       multi-device plan first. Size `usable_bytes` in the hand-written
       costs to make this happen at test-sized graphs.
     - `no_tp2_without_both_peer_links`: remove one peer link, and no `Tp2`
       or TP2-stage candidate appears.

3. **`xtask`:**
   - **`xtask/src/archcheck.rs`:** add `"moxie-models"` to `xtask`'s
     workspace allowlist, with a one-line comment: "task 0074: the
     plan-comparison tool composes model graphs without weights".
   - **`xtask/Cargo.toml`:** `moxie-models` as an optional dependency,
     added to the `cuda` feature.
   - **`xtask/src/main.rs`:** add `compare-plans --costs <toml> --model
     <name> --prompt <n> --generate <n>` to the cuda lane. Refuse (as
     `probe` does) without it.
   - **New `xtask/src/compare.rs`:**
     - It builds the graph from one of two named dense `TextConfig`s,
       defined there:
       - `gemma-dense-fits`: about 6 GiB of BF16 weights;
       - `gemma-dense-large`: about 30 GiB, so no single device fits.
     - Both are valid for `TextConfig::check()`.
     - It reads the costs with `read_costs`, calls `compare_plans`, and
       prints a table: rank, candidate (UUIDs shortened to 8 hex digits),
       per-device GiB, prefill ms, first decode ms, total s. The rejected
       candidates follow with their reasons.

4. **New `docs/evidence/plan-comparison.md`:** the output of four runs on
   the committed 0073 costs (the TOML inline in `topology-costs.md`, saved
   to a scratch file):
   - `gemma-dense-fits` at (prompt 512, generate 256) and at (prompt 32768,
     generate 256);
   - `gemma-dense-large` at the same two workloads.

   Add a short paragraph per model saying which candidate wins, and why,
   from the table's numbers. Then add the estimated-against-measured line:
   task 0072 measured one 3090 at about 35 ms and TP2 + TP1 at about 184
   ms per decode. Give this tool's estimate for the task 0072 fixture
   placement, and say plainly that the fixture is launch-bound, not
   weight-bound, which is why the model underestimates it. Close with the
   stated ceilings: no compute term, no pinned transfers, greedy cuts.

## Allowed files

- `crates/moxie-plan/src/compare.rs` (new), `crates/moxie-plan/src/lib.rs`,
  `crates/moxie-plan/tests/plan_comparison.rs` (new), and
  `crates/moxie-plan/Cargo.toml` (dev-dependencies only)
- `xtask/src/compare.rs` (new), `xtask/src/main.rs`, `xtask/src/archcheck.rs`
  (the one allowlist line and its comment), `xtask/Cargo.toml`
- `Cargo.lock` (the workspace-internal edge only; no new package)
- `docs/evidence/plan-comparison.md` (new)
- This task's Result.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check`: it passes with the new edge, and every existing
  fixture still rejects.
- `cargo xtask spec-check`.

**Tool runs** (`cargo xtask-cuda compare-plans …`): the four evidence runs.
Each run twice gives byte-identical output (show `cmp` in the Result).

**Mutations**, each applied, run, shown failing and restored:
1. Ignore KV in the fit check. `the_joint_workload_changes_the_winner`
   fails.
2. Generate `Tp2` for any pair. `no_tp2_without_both_peer_links` fails.
3. Break ties by the input device order instead of `CandidateKind`.
   `the_ranking_is_deterministic_and_input_order_free` fails. Build its
   costs so that two candidates tie exactly (two identical devices), so the
   mutation is observable.

**Stop conditions:**
- `lower_pipeline` finds no legal cut on a composed Gemma graph.
- `lower_tensor_parallel` refuses the dense Gemma graph. Report it; the
  `Tp2` candidates then appear as rejected, and the tool still runs.
- A configuration of the stated size fails `TextConfig::check()`, or
  composing it allocates weight bytes.
- A numbered change conflicts with the code: send a `DECISION` report.

## Result, filled after work

- Implemented deterministic single, TP2, pipeline and routed-host-rejection
  candidates. Stage and rank bytes still come from validated
  `build_stage_graph` results; TP2-first pipeline lowering remains
  stage-local, including when the full graph's odd vocabulary refuses TP2.

### R1 fixes

- Stage compute now uses `(weight_bytes + kv_read_bytes(context)) / memory_rate`
  once per step. Prompt rows scale only handoffs and collectives. The full
  context fit is explicitly a **resident-bytes fit** for weights plus KV; it
  excludes activation and peer scratch, so it is not full execution admission.
- Legal cuts are exactly the internal indices where
  `lower_pipeline(graph, &[i])` succeeds. Stage ranges are derived once from
  `CandidateKind`; there is no second candidate-stage copy or dummy `0..0`
  range. Greedy cut selection uses a direct iterator minimum. Decode summation
  now handles a stage's one or two rank lines directly, splitting at sliding
  window caps and the pair's arithmetic crossing.
- `compare.rs` is now **832 lines**, down from the previous 964-line draft.
  This is still above the 700-line aim; the final size includes candidate
  enumeration, checked capacity balancing, graph/rank metering, and the
  measured link calculations.
- All four tool outputs were regenerated twice with
  `CARGO_PROFILE_DEV_OPT_LEVEL=2 CUDA_DEVICE_ORDER=PCI_BUS_ID`; all four
  `cmp` checks passed. The short large-fixture output also matched the earlier
  unoptimized run byte-for-byte. No GPU device was opened.
- Corrected task 0072 Shape A fixed-placement estimate (layers 0–2 on the
  3090 pair; layers 3–5 plus the head on the 5060 Ti), prompt 7 / generate 8:
  prefill **0.108713 ms**, first decode at context 7 **0.106523 ms**, and total
  **0.960903 ms**. Resident bytes remain **33,696 / 33,696 / 74,928**. Task
  0072 measured **35.880 ms** per decode on one 3090 and **184.975 ms** for
  the combined plan. The fixture is launch-bound, not weight-bound; this model
  has no launch or dispatch term.
- All three mutations were applied, failed their focused test, and were
  restored: omitting KV from resident-byte fit failed
  `the_joint_workload_changes_the_winner`; admitting a pair without both peer
  links failed `no_tp2_without_both_peer_links` on the missing path; and using
  input-device order for ties failed
  `the_ranking_is_deterministic_and_input_order_free` with reversed costs.
- Updated evidence results: `gemma-dense-fits` ranks TP2 first at prompt 512
  (**1.680 s**) and one 3090 first at prompt 32,768 (**2.238 s**, versus
  **13.811 s** for TP2). `gemma-dense-large` ranks TP2 first at prompt 512
  (**8.551 s**); at prompt 32,768 the pipeline with the 5060 Ti first ranks
  first (**15.122 s**, versus **69.826 s** for TP2). Single-device candidates
  fail the resident-bytes fit at 32.693 GB short context and 37.106 GB long
  context. Evidence explains that prompt rows scale transfers and collectives,
  and identifies the fit's omitted activation and peer scratch bytes.

### Verification and review map

- Passed `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, and `cargo xtask spec-check` (10
  documents). `cargo xtask arch-check` passed with 79 rejected fixtures, 21
  accepted fixtures, and all 13 rules exercised.
- Review map (lines added / removed relative to HEAD):

  | File | + / − |
  |---|---:|
  | `Cargo.lock` | 3 / 0 |
  | `crates/moxie-plan/Cargo.toml` | 4 / 0 |
  | `crates/moxie-plan/src/compare.rs` | 832 / 0 (final file: 832 lines) |
  | `crates/moxie-plan/src/lib.rs` | 2 / 0 |
  | `crates/moxie-plan/tests/plan_comparison.rs` | 260 / 0 |
  | `docs/evidence/plan-comparison.md` | 119 / 0 |
  | `docs/tasks/0074-m5-plan-comparison-tool.md` | 72 / 1 |
  | `xtask/Cargo.toml` | 2 / 0 |
  | `xtask/src/archcheck.rs` | 2 / 0 |
  | `xtask/src/compare.rs` | 176 / 0 |
  | `xtask/src/main.rs` | 14 / 1 |

  The carried `.gitignore`, `docs/evidence/specification-version.md`, and ADRs
  0034/0035 remain untouched and unstaged. No commit was created.
