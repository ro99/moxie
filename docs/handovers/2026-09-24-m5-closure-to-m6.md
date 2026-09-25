# Handover — accepted M5 boundary

## Workspace identity

- **Writable root:** `/home/rodrigo/Developer/moxie`, branch `main`.
  - M5's last code is task 0075 at `ad09f9b`.
  - The acceptance package is in the [M5 ledger](2026-09-22-m4-closure-to-m5.md)
    (commits `e0fde39` and `f252824`, plus the closure commit that adds this
    file).
  - Fetch the pushed branch and recheck HEAD before implementation.
- **Carried owner files, not part of M5 and never staged by agents:**
  `.gitignore`, `docs/evidence/specification-version.md`, and the untracked
  ADRs 0034 and 0035.
- **Read-only inputs:** legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; `/models` and `/fast/models`
  (ADR 0020).
- **Hardware:**
  - 3090 `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and 3090
    `GPU-81fe4578-59b2-37c4-421e-287cdac78704`, which have peer access to
    each other;
  - 5060 Ti `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` (SM120), with no peer
    access.
  - Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
  - The patched P2P open module is pinned in
    [topology-p2p.md](../evidence/topology-p2p.md): 610.57.04, branch
    `p2p-610.57.04-p2p-v2`, commit `e546041`, `srcversion`
    `8C9DB5D610ABCC6245B2266` on both installed kernels.
- **Team at close:** builder Codex `luna`, reviewer Codex `sol` and
  coordinator Claude Opus `coordinator` are idle. No job is running and no GPU
  is reserved.

## Completed facts

The owner accepted M5 on 2026-09-24 ("close m5"). Tasks 0053–0075 are
accepted; the ledger maps each exit clause to its evidence. In short:
- **Partition semantics:** attention, MLA, dense linears, the vocabulary
  projection and routed experts have legal partitions, proven by host
  lowerings bit-identical to the unsplit graph (tasks 0053–0058, 0064, 0067).
- **Devices:**
  - one GPU, dense and routed (tasks 0059, 0065);
  - TP2 on the 3090 pair, with a thread per rank, a status rendezvous and
    rank-failure and mismatch injection (tasks 0060–0063, 0066);
  - host-owned experts (task 0068);
  - a single-GPU rank worker (task 0070);
  - a three-GPU pipeline with microbatched prefill and an all-or-nothing
    commit (tasks 0069, 0071);
  - a combined TP2 + TP1 plan (task 0072).
- **M5.3 report:** at fixture scale, the combined plan improves capacity (the
  per-GPU weight peak falls from 134,544 to 71,472 bytes) and worsens latency
  (decode is about 5× slower).
- **M5.5:** measured topology costs
  ([topology-costs.md](../evidence/topology-costs.md)), and a deterministic
  plan comparison ranked by the joint workload
  ([plan-comparison.md](../evidence/plan-comparison.md)). The winner changes
  with prompt length.
- **Final gates at `ad09f9b`:** format, both clippy lanes,
  `cargo test --workspace`, arch-check (79/21/13) and spec-check pass. On the
  GPUs, `dense_gemma_device` is 6/6, `dense_tp2_device` passes and
  `cargo xtask-cuda test-gpu` is 63/63 (SM86 and SM120).
- **Milestone-end audit:** task 0075 applied the behaviour-preserving cleanup
  and fixed a latent defect: the TP worker dropped a refused paged-attention
  run, unloading its CUDA module under a possibly running kernel.

## Decisions

- **Owner rulings in M5:**
  - auto-mode delegation (2026-09-22);
  - sequential work with luna only (2026-09-23);
  - no task numbers in code names (2026-09-24);
  - cleanup before the acceptance package (2026-09-24);
  - "fix the driver pin. then close m5. then document and push." (2026-09-24).
- **ADR 0036:** FP32 partials are combined in declared order and rounded to
  BF16 once.
- **Coordinator-approved boundary edit:** `xtask` may depend on
  `moxie-models`, the composition-root exemption arch-check already names
  (task 0074).
- **Laguna's scaled combine under expert-owner TP** is M7 work, refused
  fail-closed until then.
- **Every host-staged transfer is declared**, and a peer copy without a
  grant is refused.

## Remaining hypotheses and blockers

- **Not claimed by M5:**
  - any checkpoint-backed run or generated token;
  - pipeline microbatch overlap (M6.3);
  - pinned transfers (M6.3);
  - a compute-cost term: the plan comparison is weight-traffic and transfer
    bound, and underestimates launch-bound and compute-bound work;
  - automatic plan selection (M6.5);
  - any speed guarantee (O6 is open).
- **Pipeline limits:** a TP stage must be first, and a pipeline with a TP
  stage takes one microbatch.
- **Unexplained measurement:** the forward 3090 peer link drops from about
  6.58 GB/s isolated to 2.92 GB/s with both directions busy, while the
  reverse direction holds 5.2 GB/s. It is recorded as unexplained in
  `topology-costs.md`.
- **Maintenance, not blocking:**
  - size overruns against estimate in `rank_worker.rs`, the executor's
    `pipeline.rs`, `topology_probe.rs` and `compare.rs`;
  - the `dense_gemma_device` step-runner duplication (skipped in task 0075,
    because it moved assertions).
- **Lesson (engineering log, tasks 0065–0067):** validate a declaration by
  re-deriving it at the consumer, never field by field. Before writing a
  contract, check every signature it names against the code (tasks 0074 and
  0075).

## M6 closure ledger (coordinator, opened 2026-09-24)

The owner opened M6 on 2026-09-24 ("open M6 for luna to work on"). This
section is the coordinator's running ledger for M6 (coordinator.md §2); the
tsk board thread `m6` mirrors it.

### Milestone obligations (roadmap M6, original item numbers)

| Item | Outcome | State | Owner | Evidence / next action |
|---|---|---|---|---|
| M6.1 | Device-resident layer chains, batched/grouped expert kernels, fused dequant/activation/reduction, output-minimizing transfers | **Active** | luna, slice 1 | [0076](../tasks/0076-m6-device-kv-append.md) **accepted** 2026-09-24 (`1f64a92`, `0f5363f`): no per-layer K/V device→host→device copy; decode host workspace 128 → 64 bytes on `gemma-a`; outputs bit-identical; no timing claim. [0077](../tasks/0077-m6-rope-tables-without-sync.md) **accepted** 2026-09-24 (`a9da4af`): distinct RoPE tables retained per step, no per-node `synchronize`; `gemma-a` host workspace prefill 320 → 480 B, decode 64 → 96 B (both tables now held). Remaining in slice 1: per-operation `settle` in `PagedAttentionRun`, the per-Rope `synchronize`, the routed-expert route readback, per-step module load |
| M6.2 | Shape-bucket chunked prefill; stable decode graph capture with piecewise fallback; graph pools and workspaces admitted | Not started | — | Capture cannot record a host wait, so the deferred per-operation `settle` item is reopened as slice 4's first task, [0082](../tasks/0082-m6-deferred-paged-attention-completion.md) (**accepted** 2026-09-25: one blocking wait per step; [0083](../tasks/0083-m6-rope-tables-uploaded-once.md) **accepted** 2026-09-25: tables uploaded once per step; [0084](../tasks/0084-m6-cuda-graph-capture-wrappers.md) **accepted** 2026-09-25: `moxie-cuda` capture/instantiate/launch, smoke case 66/66; [0085](../tasks/0085-m6-piecewise-decode-capture.md) **accepted** 2026-09-25: opt-in piecewise capture and replay, bit-identical on all three GPUs; replayed decode about 12% faster at fixture scale. [0086](../tasks/0086-m6-admit-graph-memory.md) **active**: admit graph memory in `DeviceTier::GraphPools` by a measured bound before capture can be a default). Strata never used CUDA graphs; the pinned vLLM graph-mode design (source map) is the reference |
| M6.3 | Measured transfer overlap, bounded read-ahead, optional hot-expert tiers and route prediction, against no-overlap baselines | Not started | — | Carries M5's pipeline microbatch overlap and pinned transfers |
| M6.4 | Joint prefill/decode placement and phase transitions; no host-cache or prepared-layout growth across many turns | Not started | — | — |
| M6.5 | Automatic plan selection from measured topology/shape costs; fixed-plan mode | Not started | — | Builds on M5.5's `compare-plans`; needs a compute-cost term |
| Exit | Paired prefill/decode/quality/memory benchmark for the dense and two MoE stress graphs, **plus actual available checkpoints**; defaults justified outside variance | Not started | coordinator | Slice 7; checkpoints stay in M6 (owner, below) |

### Route to M6 closure (proposed, coordinator, 2026-09-24)

| Order | Slice | Delivers | Closes |
|---|---|---|---|
| 1 | Sync-free single-GPU dense step | K/V appended on the device (0076); event-retained instead of settled paged-attention operations; RoPE tables uploaded once per step; one host synchronization per step, at `finish` | M6.1 dense chain; prerequisite of M6.2 capture |
| 2 | Quantized weights in the dense step | **Re-scoped (coordinator, 2026-09-24)** from "device routing and grouped experts": device experts already route on the device (`DENSE_ROUTE`); the only route readback is the host-owned-expert join, whose overlap is M6.3. What the dense step lacks is any quantized weight. [0080](../tasks/0080-m6-affine-linear-admitted-in-dense-plans.md) **accepted** 2026-09-24: plans admit affine INT4/INT8 `Linear` weights through a planner format map, and the dense `launch` checks symbol identity. [0081](../tasks/0081-m6-affine-linear-in-the-dense-step.md) **accepted** 2026-09-25: the single-GPU step runs INT8/INT4 affine linears, 0.000 BF16 ULP against the host reference on all three GPUs. **Slice 2's planned tasks are complete.** Open for slice 7: TP/PP with formats, quantized experts in the graph, and the affine kernel's `max_input` bound. Grouped expert kernels and quantized experts in the graph follow only if the exit benchmarks need them | M6.1 fused dequantization; prerequisite of slice 7's quantized checkpoint benchmarks |
| 3 | Benchmark harness | Fixed-plan paired prefill/decode/memory benchmark with repetitions and variance, base versus candidate | M6 exit method; every later slice measures with it |
| 4 | Prefill buckets and decode capture | Shape buckets, captured decode with piecewise fallback, admitted pools. Planned route (coordinator, 2026-09-25), piecewise first as in the pinned vLLM design: [0082](../tasks/0082-m6-deferred-paged-attention-completion.md) no waits inside the step; [0083](../tasks/0083-m6-rope-tables-uploaded-once.md) RoPE tables uploaded once per step; 0084 `moxie-cuda` capture/instantiate/launch wrappers; 0085 a reused decode plan captures its segments between attention nodes and replays them; then full-step capture (needs device-side step scalars) and prefill buckets | M6.2 |
| 5 | Overlap and read-ahead | Measured overlap against no-overlap baselines, wasted work recorded | M6.3 |
| 6 | Phase placement and many-turn growth | Joint placement; no growth across many turns | M6.4 |
| 7 | Plan selection, benchmark package, close | Automatic and fixed-plan selection; exit benchmarks; milestone-end audit | M6.5; M6 closure |

Slice 3 may move ahead of slice 2 if slice 1 needs timing evidence to choose
between mechanisms. **It did (coordinator, 2026-09-24):** after 0077, slice
1's remaining candidates (per-operation `settle` in `PagedAttentionRun`, the
per-step `Module::load`) are ranked by [task
0078](../tasks/0078-m6-dense-step-timing.md)'s fixed-plan timing and CUDA API
breakdown before either is opened. **0078 accepted** 2026-09-24
([evidence](../evidence/dense-step-timing.md)): module load/unload is 24% of
per-step driver API time and event synchronization 7%, so module caching is
next: [task 0079](../tasks/0079-m6-dense-module-per-plan.md) **accepted** 2026-09-24: module loaded once per plan; fixture step median about 1.6 → 1.15 ms. **Slice 1 is complete** except the deferred per-operation `settle`. **Deferred** (coordinator, 2026-09-24): removing the per-operation
`settle` in `PagedAttentionRun`. Unfinished scope: about 19 event
synchronizations per fixture step. Revisit when decode capture (slice 4)
needs a sync-free step, or when a checkpoint-scale profile shows them
material.

### Coverage gap carried (coordinator, 2026-09-25)

Task 0082's asynchronous-ownership repairs (cross-stream ordering, fork from
a parent with pending writes, dropping a run with pending work) are made by
construction and verified in review; no test separates the defects from the
repairs, because the test support can only gate an event record, not a copy.
Task 0085's failed-drain cleanup after a failed capture step (graphs kept
with the withheld plan) is in the same class: it needs an injected stream
synchronization failure the suite does not have.
Owner: coordinator. Revisit: when capture work (tasks 0084–0085) adds stream
gating to the test support, or at M6 closure at the latest.

### Known bound for slice 7 (coordinator, 2026-09-24)

`gemma-4-31B-it-AWQ-8bit` is INT8, group 32, symmetric, which the affine
format covers. Its `down_proj` has 21,504 inputs, above the affine kernel's
qualified `max_input` of 16,384. Lifting that bound is a qualification task
before the slice 7 checkpoint run.

### Checkpoints stay in M6's exit (owner, 2026-09-24)

The coordinator first proposed deferring the exit gate's checkpoint benchmarks
to M7, then pulling a real checkpoint forward as slice 2. The owner rejected
both: checkpoints are not deferred to M7, and the route keeps roadmap order,
with checkpoint benchmarks in slice 7's exit package. Where a slice's decision
depends on real-scale bytes (likely M6.3's overlap and read-ahead), that
slice measures it. Available inputs, read-only:
`/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` (33 GB) and
`/fast/models/google/gemma-4-26B-A4B-it` (49 GB); `torch 2.10` and
`transformers 5.5.3` are installed as an external reference.

## Next task

[Task 0076](../tasks/0076-m6-device-kv-append.md), assigned to luna on
2026-09-24. The paragraph below is the pre-authorization note, kept for
provenance.

None is authorized. M6 ("shared performance paths and phase balance")
opens only on the owner's instruction. When it does, the natural first bounded
deliverable is M6.1's device-resident layer chain on one GPU, with its own
contract and stop conditions. The carried maintenance items above are its
candidates for a cleanup slice.
