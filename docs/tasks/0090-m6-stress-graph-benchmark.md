# Task 0090 — paired benchmark of the dense and two MoE stress graphs (fixture scale)

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0090, M6 slice 7 (benchmark package), first part. M6 exit: "paired
  prefill/decode/quality/memory benchmark for at least the dense and two MoE
  stress graphs, plus actual available checkpoints". This task builds the
  method on the fixture-scale stress graphs; the checkpoint rows come later in
  slice 7, after weights are shared across plans (ledger finding).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.** No `git stash`, no `git worktree`.
- **GPUs are free**; the builder is the only GPU user and the only writer.
  Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.
- **O6 is open.** Fixture-scale figures are evidence of method, not a speed
  claim; the evidence file's first line says so.

## Facts established before writing (coordinator, 2026-09-25)

- `dense_gemma_device.rs` has the fixed-plan timing harness
  `dense_step_timing` (task 0078; phases prefill, decode, decode-captured),
  `host_step` (host reference logits), `bf16_ulp`, and the routed fixtures:
  Shape C (`moxie_cli::gemma::Shape::C`, dense shared and routed experts on
  the device) and its top-3 variant built with
  `moxie_cli::gemma::build_with_config` (`top_k = 3`).
- Plans without host joins, orders or ownership accept segment capture (task
  0085); device-expert routed plans qualify.
- `SelectedReservedPlan` exposes region byte counts and
  `graph_pool_bytes()` (task 0086).

## Bounded deliverable

- **Outcome:** one ignored test prints, for each of three graphs (Shape A
  dense, Shape C routed top-2, Shape C routed top-3) on the 3090
  `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`: prefill and decode step
  latency (eager and captured), admitted device memory, and logit error
  against the host reference; an evidence file records two runs at the
  candidate and one paired run at the base commit named below.
- **Allowed files:** `crates/moxie-executor/tests/dense_gemma_device.rs` (one
  new ignored test appended at the end, reusing helpers; `dense_step_timing`
  may be deleted if the new test covers its Shape A lines exactly, say so),
  new `docs/evidence/m6-stress-benchmark.md`, this task's Result.
- **Non-goals:** production code; checkpoints; TP/PP; making any figure a
  gate.

## The test

`stress_graph_benchmark`, ignored (`"benchmark; run explicitly"`):

1. For each graph: build the fixture; prompt of 5 tokens (as the main test),
   decode 1 row at position 5; lower and admit a prefill plan and a decode
   plan once.
2. **Latency:** `W = 5` warm-ups, `R = 30` measured repetitions, median/min/
   max of `execute_dense` + `finish`, abort after each (as task 0078). Phases:
   `prefill`, `decode`, and `decode-captured` (enable capture on the decode
   plan, one untimed capture step, then the loop).
3. **Memory:** per plan, `weight_region_bytes + activation_region_bytes +
   workspace_region_bytes + graph_pool_bytes()` for the decode plan with
   capture enabled, and the prefill plan's total; also the paged-state bytes
   admitted for the runs.
4. **Quality:** the worst logit error of the committed prefill's last row
   and of one committed decode row against `host_step`, in BF16 ULP at the
   reference magnitude (the unit `assert_logits` uses). Print the worst value;
   do not assert beyond what `assert_logits` already asserts.
5. Print one line per graph and phase:
   `stress-benchmark graph=<a|c-top2|c-top3> phase=<…> median_us=… min_us=…
   max_us=… device_bytes=… worst_ulp=…` (fields that do not apply print
   `-`).

## Measurements

Command: `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test --release -p
moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks
--test dense_gemma_device stress_graph_benchmark -- --ignored --nocapture`.

- Two runs at the candidate commit.
- One **paired** run at base `08a6fdb` (the M5 closure, before any M6 change)
  using the task 0078 procedure: commit the test first, `git checkout
  --detach 08a6fdb`, apply only the new test with `git diff`, run, restore,
  return to `main`, `git status --short` showing only the carried files
  before and after. If the test does not apply or build at the base (APIs it
  uses did not exist yet), run the base with capture phases removed; if it
  still cannot run, record that the base could not be measured and why. Do
  not change production code at the base.

`docs/evidence/m6-stress-benchmark.md`: first line "fixture-scale, O6 open,
not a performance claim"; GPU UUID, driver, commits, command; one table per
run; one paragraph stating only what the numbers show (latency change from
base, memory, error).

## Acceptance

- Host gates: `cargo fmt --all -- --check`; `cargo clippy --workspace
  --all-targets --locked -- -D warnings`; the executor driver-feature clippy;
  `cargo test --workspace --locked`.
- `dense_gemma_device` full suite passes (the new test is ignored there).
- The evidence file exists with every item above; the tree is back on `main`
  with only the carried files dirty.

## Result, filled after work

Implemented the single ignored `stress_graph_benchmark` test at the end of
`dense_gemma_device.rs`; no production code changed. It measures Shape A,
Shape C top-2 and top-3 prefill, eager decode and captured decode on the
specified 3090, including admitted plan regions, graph pools, paged state, and
worst committed logit error against `host_step`.

Two candidate runs completed at `bdd099065dabbfaba8a0e9c9191425ef3d6bfb92`.
The full test-only diff did not apply at base `08a6fdb`, whose dense test file
predates the appended M6 timing harness; the specified fallback with capture
removed built and completed one paired prefill/decode run. The temporary base
test edit was discarded and the tree returned to `main`, with only the
carried files dirty before and after.

Host gates passed: fmt check, workspace clippy, driver-feature executor
clippy, and workspace tests. The full `dense_gemma_device` suite passed (10
passed, 2 ignored). All committed prefill/decode comparisons were 0.000 BF16
ULP. Tables, memory totals, paired latency readings and the base fallback are
recorded in [m6-stress-benchmark.md](../evidence/m6-stress-benchmark.md).
