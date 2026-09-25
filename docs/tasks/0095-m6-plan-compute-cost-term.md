# Task 0095 — a measured compute term in plan comparison

Status: **accepted** (coordinator, 2026-09-25) after sol's review round R1.
Implementation `e9b975f`. Builder Codex `luna`; reviewer Codex `sol`.
- R1 (medium, the coordinator's contract gap): when the final
  synchronization fails, the probe forgets its module, buffers and events
  instead of freeing them. This is correct: kernel completion is unknown, and
  freeing would risk use after free. The Resources clause is amended below
  with this quarantine exception; the code is unchanged.
- Reading (estimates, not a claim): Moxie's dense linear kernel measures
  0.055 TFLOP/s on each 3090 and 0.109 on the 5060 Ti. With the compute
  term, matrix multiply dominates every estimate, and split phase pairs no
  longer differ from same-placement ones beyond ±0.04 s.

## Identity and authority

- Task0095, M6 slice 7, roadmap **M6.5** "automatic plan selection based on
  measured topology/shape costs". The planner prices every step by memory
  bytes only (`rank_time` in `compare.rs`: weights plus KV read over memory
  bandwidth). It has no compute term, so one 3090 "prefills" 32,768 tokens
  in 8.7 ms ([plan-comparison.md](../evidence/plan-comparison.md)). Task
  0094's split-pair wins are partly an artifact of this, and slice 6's
  device transition is gated on this term (ledger, M6.4).
- The rate must be **Moxie's own kernel on this machine**, not a datasheet
  figure. The dense linear kernel `moxie_dense_linear_split_v1`
  (`moxie-kernels/cuda/dense_ops.cu` about 22–46) is correctness-first: one
  thread per output value, no tiling. Its measured rate is what Moxie
  actually runs.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  `.gitignore`, `docs/evidence/specification-version.md` and ADRs 0034 and
  0035. **Stage explicit paths only.** GPUs are free; always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in identifiers, labels or strings.
- O6 open: estimates and a probe, not a performance claim.

## Facts established before writing (coordinator, 2026-09-25)

- `costs.rs`: `DeviceCost { device, memory_gbps, usable_bytes }`.
  `xtask/src/probe.rs` `write_costs`/`read_costs` serialize it with required
  fields (`required_float`). Its round-trip test is
  `written_costs_read_back_equal`.
- `topology_probe.rs` `probe_topology` measures `memory_gbps` per device
  with events (about 404–427); `ProbeConfig` holds the repetition counts.
- The kernel: `moxie_dense_linear_split_v1(x, weight, output, rows,
  input_width, output_width, blocks)`. It uses one thread per output element,
  weight row-major `[output_width, input_width]`, and BF16 in and out.
  `dense.rs` (about 637) launches it with 256-thread blocks. The fatbin is
  `moxie_kernels::DENSE_GRAPH_FATBIN`.
- `compare.rs` `stage_meter` builds a `RankMeter { device, weight_bytes,
  memory_gbps, kv }`. `rank_time(rank, context)` = bytes / (`memory_gbps` ×
  1e6) ms. `stage_time` takes the slowest rank. The prefill estimate calls
  `stage_time(stage, prompt_tokens)` once; decode sums `stage_time` per
  generated token.
- `OpParams::Linear { in_features, out_features, .. }`, `VocabProjection {
  vocab, hidden, .. }`, `ExpertMlp { hidden, intermediate, experts, top_k,
  .. }`, `Attention { heads, head_dim, visibility, .. }`.

## Bounded deliverable

- **Outcome:** a measured `linear_tflops` per device. The planner prices
  each rank's step as the larger of its memory time and its compute time.
  The committed costs and plan-comparison evidence (including 0094's phase
  pairs) are regenerated with it.
- **Allowed files:** `crates/moxie-plan/src/costs.rs`,
  `crates/moxie-plan/src/compare.rs`,
  `crates/moxie-plan/tests/plan_comparison.rs` (fixture costs gain the field,
  plus one new test), `crates/moxie-executor/src/topology_probe.rs`,
  `xtask/src/probe.rs`, `docs/evidence/topology-costs.md`,
  `docs/evidence/plan-comparison.md`, this task's Result.
- **Non-goals:** a faster kernel; a separate attention rate; automatic
  selection wiring (the next M6.5 task); the device transition.

## Numbered changes

1. **Cost field (`costs.rs`).** `DeviceCost` gains `pub linear_tflops: f64`,
   with the doc comment "median sustained rate of Moxie's dense BF16 linear
   kernel at the probe shape, in TFLOP/s". `sorted_devices` in `compare.rs`
   refuses a non-finite or non-positive value, as it does for
   `memory_gbps`.
2. **TOML (`xtask/src/probe.rs`).** `write_costs` writes `linear_tflops`,
   and `read_costs` requires it with `required_float`. There is no default:
   an old costs file is refused, naming the field. Update the round-trip
   test's values. `append_costs` prints the new column.
3. **Probe (`topology_probe.rs`).** In `probe_topology`, per device, after the
   memory measurement:
   - load the dense-graph fatbin as a module on that context and get
     `moxie_kernels::DENSE_LINEAR_SPLIT`;
   - allocate BF16 `x [512 × 4096]`, `weight [4096 × 4096]` and `output
     [512 × 4096]`, filled with any finite BF16 value (for example 1.0);
   - launch with `blocks = 1`, grid `ceil(512·4096 / 256)`, 256 threads,
     once untimed;
   - time `config.reps` launches with the existing `event_times`;
   - `linear_tflops = median(2·512·4096·4096 / seconds / 1e12)`.

   Record the shape as named constants in the probe. Release the module and
   buffers before the next device.
4. **Compute metering (`compare.rs`).** `RankMeter` gains
   `linear_flops_per_row: u64` and `tflops: f64`. In `stage_meter`, per node:
   - `Linear`: `2 × in_features × out_features`;
   - `VocabProjection`: `2 × vocab × hidden`;
   - `ExpertMlp`: `2 × top_k × (the node's expert weight element count /
     experts)`, computed from the node's weight inputs, so a gated expert
     counts its three matrices;
   - nothing else.

   In the TP meter, each rank meters its own local graph as it already does,
   so the flops follow the partition. `KvRead` gains `head_dim: u64` and
   `query_heads: u64` from the `Attention` params (local values on a TP
   rank).
5. **Time (`compare.rs`).** Replace `rank_time(rank, context)` with
   `rank_time(rank, rows, context)`:
   - `memory_ms` is today's formula at `context`;
   - `flops = rows × linear_flops_per_row + Σ_attention 4 × query_heads ×
     head_dim × keys`;
   - `keys` for rows `context − rows + 1 ..= context`, each seeing `min(its
     position, window)` keys. Compute this sum in closed form (with window
     capping), not a loop;
   - `compute_ms = flops / (tflops × 1e9)`;
   - return `max(memory_ms, compute_ms)`, with a `ponytail:` comment that
     max assumes perfect overlap.

   Prefill uses `rows = prompt_tokens, context = prompt_tokens`, and decode
   `rows = 1`. Everything else in `estimate` is unchanged.
6. **Test** `compute_term_prices_long_prefill` in `plan_comparison.rs`, with
   the fixture:
   - with `linear_tflops` huge (1e9), every estimate equals the memory-only
     value (compute never dominates);
   - with `linear_tflops = 1e-6`, a single-device prefill at prompt 10,000
     equals the closed-form compute time, which the test computes by hand
     from the fixture's dimensions.
7. **Evidence.**
   - Run `cargo xtask-cuda probe --costs <path>` twice. Replace the costs
     block in `docs/evidence/topology-costs.md` with the second run, and add
     one line with both runs' `linear_tflops` per device.
   - Regenerate every table in `docs/evidence/plan-comparison.md`
     (including the phase pairs) with the new costs, each command twice with
     `cmp`.
   - Add one paragraph: does the best plan change at prompt 512 and 32,768,
     and does a split phase pair still beat the best same-placement pair?
     Numbers only; estimates.

## Contract before implementation

- **Semantics:** with compute never dominating, every estimate equals
  today's (the test's first case).
- **Resources:** the probe allocates about 40 MiB per device and frees it;
  failure frees before returning, **except** when a synchronization fails:
  then completion is unknown, and the resources are quarantined (leaked)
  rather than freed (amended after R1).
- **Failure:** a missing or invalid `linear_tflops` is a typed refusal;
  arithmetic overflow is `Err`.

## Acceptance

Host gates: `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`;
`cargo xtask spec-check`. GPU: the probe runs twice on all three devices.

**Coverage check (one mutant, reverted after):** return `memory_ms` only
from `rank_time`; the new test must fail.

## Result, filled after work

Implemented the measured `linear_tflops` cost field, dense BF16 probe, and
per-rank compute estimate. The one-mutant check (memory-only `rank_time`)
failed `compute_term_prices_long_prefill` as required, and the mutant was
reverted. Two probe runs completed on all three GPUs; linear rates differed by
at most 0.33% per device. All four plan-comparison commands ran twice with
matching output. The regenerated estimates show the prompt-512 and prompt-
32,768 winner changes and phase-pair comparison recorded in
`docs/evidence/plan-comparison.md`.

Gates passed: fmt, workspace clippy, executor driver-feature clippy, workspace
tests, `cargo xtask arch-check`, and `cargo xtask spec-check`. The two all-GPU
probe runs passed. The updated evidence contains run 2 costs, both measured
linear rates per device, and the regenerated plan and phase-pair tables.
