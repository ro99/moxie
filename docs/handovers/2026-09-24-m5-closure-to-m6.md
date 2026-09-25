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
| M6.2 | Shape-bucket chunked prefill; stable decode graph capture with piecewise fallback; graph pools and workspaces admitted | Not started | — | Capture cannot record a host wait, so the deferred per-operation `settle` item is reopened as slice 4's first task, [0082](../tasks/0082-m6-deferred-paged-attention-completion.md) (**accepted** 2026-09-25: one blocking wait per step; [0083](../tasks/0083-m6-rope-tables-uploaded-once.md) **accepted** 2026-09-25: tables uploaded once per step; [0084](../tasks/0084-m6-cuda-graph-capture-wrappers.md) **accepted** 2026-09-25: `moxie-cuda` capture/instantiate/launch, smoke case 66/66; [0085](../tasks/0085-m6-piecewise-decode-capture.md) **accepted** 2026-09-25: opt-in piecewise capture and replay, bit-identical on all three GPUs; replayed decode about 12% faster at fixture scale. [0086](../tasks/0086-m6-admit-graph-memory.md) **accepted** 2026-09-25: captured graphs charged in `GraphPools` by a measured, enforced bound (8 KiB per kernel, 128 KiB per segment). Capture remains opt-in until the exit benchmarks justify a default. Remaining in M6.2: [0087](../tasks/0087-m6-bucketed-chunked-prefill.md) **accepted** 2026-09-25: `prefill_chunks` bucket policy; one admitted plan per bucket reused across chunks and prompts, captured then replayed, matching the host reference on all three GPUs. Remaining: full-step capture. **For the owner:** document 02 gives `moxie-engine` prefill/verify/decode, but arch-check limits it to ADR 0011's host-reference profile; wiring device generation into the engine needs a new dependency edge and an ADR, and is not needed for M6's exit benchmarks). Strata never used CUDA graphs; the pinned vLLM graph-mode design (source map) is the reference |
| M6.3 | Measured transfer overlap, bounded read-ahead, optional hot-expert tiers and route prediction, against no-overlap baselines | **Active** | luna, slice 5 | [0089](../tasks/0089-m6-pinned-transfer-probe.md) **accepted** 2026-09-25: pinning gains little bandwidth, but pageable async copies block the host for the whole transfer while pinned ones return in µs and overlap compute. [0092](../tasks/0092-m6-streamed-kv-read-ahead.md) **accepted** 2026-09-25 ([evidence](../evidence/kv-read-ahead.md)): bounded two-page pinned read-ahead for host-backed KV streaming, bit-identical to `stage_next`, with wasted work reported (`prefetched_unused`). Medians match the no-prefetch baseline within ±3 µs of about 347 µs, and overlap is not shown at fixture scale. Remaining: the optional hot-expert tiers and route prediction, and M5's pipeline microbatch overlap |
| M6.4 | Joint prefill/decode placement and phase transitions; no host-cache or prepared-layout growth across many turns | **Partly met** | luna, slice 6 | [0088](../tasks/0088-m6-no-growth-across-many-turns.md) **accepted** 2026-09-25: over 24 turns, ledger charges, outstanding count and device free bytes are exactly constant (the "no growth" clause, at fixture scale). Duplicate resident weights across plans eliminated by [0091](../tasks/0091-m6-plan-set-shares-resident-weights.md) (accepted 2026-09-25). Joint placement: [0094](../tasks/0094-m6-joint-phase-placement-estimate.md) **accepted** 2026-09-25: `compare_phase_pairs` ranks prefill/decode placement pairs with the KV move priced; at prompt 512 a split pair (one 3090 prefill, TP2 decode) estimates 11% below TP2 throughout. The cost model has no prefill compute term, so that win is partly an artifact. The device KV transition is built regardless (owner ruling below). Owner (2026-09-25): M6's shared paths are designed from the roadmap and spec, not from one family's mechanisms |
| M6.5 | Automatic plan selection from measured topology/shape costs; fixed-plan mode | **Active** | luna, slice 7 | Builds on M5.5's `compare-plans`. [0095](../tasks/0095-m6-plan-compute-cost-term.md) **accepted** 2026-09-25: measured `linear_tflops` (0.055 TFLOP/s on a 3090, 0.109 on the 5060 Ti; Moxie's correctness-first dense linear) and a max(memory, compute) step time. Next: selection wiring and fixed-plan mode |
| Exit | Paired prefill/decode/quality/memory benchmark for the dense and two MoE stress graphs, **plus actual available checkpoints**; defaults justified outside variance | **Method accepted** | coordinator | [0090](../tasks/0090-m6-stress-graph-benchmark.md) **accepted** 2026-09-25 ([evidence](../evidence/m6-stress-benchmark.md)): fixture-scale stress graphs A, C top-2, C top-3, paired against the M5 closure `08a6fdb`; decode 1.77 → 0.97 ms (A) and 2.84 → 1.95 ms (C top-2) eager. Still required: checkpoint rows, variance-justified defaults. Checkpoint rows need shared weights first (finding below) |

### Build the features; measure at the exit (owner, 2026-09-25)

The owner's ruling: M6's pace slowed because the coordinator gated features
behind measurements. That was the coordinator's error. Moxie is a generic
engine for many models, so every feature the roadmap names is built. Whether
it is used depends on the user's hardware and model, not on one fixture or
checkpoint. "Optional" in M6.3 means optional to the user at run time, so the
feature is built opt-in. Measurement belongs to the exit gate (paired
benchmarks, default choices) and to each feature's correctness test, not in
front of the work. One builder works at a time; the Claude Opus `builder`
stays idle unless the owner asks.

Remaining build route, in roadmap order (each a bounded task, luna, sol
reviews):
1. **M6.1** Fast BF16 linears. Owner ruling (2026-09-25,
   [ADR 0037](../decisions/adr/0037-cublas-tensor-op-bf16-linears.md)):
   cuBLAS tensor-op under ADR 0028's gate; selection projections keep the
   declared order. [Task 0096](../tasks/0096-m6-cublas-bf16-linears.md). The
   affine INT8/INT4 kernel is already tensor-core (ADR 0028). Task 0095
   measured the current kernel at 0.055 TFLOP/s on a 3090 and 0.109 on the
   5060 Ti, about 1,000 times below the hardware.
2. **M6.2** Full-step decode capture, attention included, with the piecewise
   path as the fallback.
3. **M6.3** Pipeline microbatch overlap (carried from M5); opt-in hot-expert
   tier; opt-in route prediction feeding the bounded prefetch class.
4. **M6.4** Device KV transition between a prefill and a decode placement.
5. **M6.5** Automatic selection wired into plan choice, and fixed-plan mode.
6. **Slice 7** Checkpoint route items 2–6, the exit benchmarks, default
   choices, the carried failure-path tests, and the milestone-end audit.

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

### Finding: duplicate resident weights across plans (coordinator, 2026-09-25)

Every `SelectedReservedPlan` admits its own `PackedResidentWeights` region and
uploads the weights into it on its first step (`moxie-executor/src/chain.rs`
`bound_weights`; `moxie-plan` `weight_region_bytes` per candidate). Task
0087's reused bucket set of four plans therefore holds four full weight
copies on one GPU. Harmless at fixture scale; impossible for any real
checkpoint, and exactly M6.4's "eliminating duplicate resident
representations". **This is on slice 7's critical path.**

Direction (to be designed as slice 6's next task): a dense plan is admitted
without a weights region, and its weight operands resolve to addresses of
leases held from the one weight-residency owner (`moxie_memory::residency`,
task 0020; M3's `affine_linear.rs` `resolve` already does this for the
standalone linear). The leases must outlive every plan and captured graph
that names their addresses. A new shared-weights object would be a second
residency owner, which AGENTS.md and arch-check's `second-residency-owner`
rule forbid.

**Resolved in code by [task 0091](../tasks/0091-m6-plan-set-shares-resident-weights.md)
(accepted 2026-09-25, option B):** `DensePlanSet` holds the weight leases and
every bucket plan; weights are charged once in `PackedResidentWeights`.
Existing tests and harnesses still use per-plan weights; the checkpoint
benchmarks must use the set.

Design fork (coordinator, 2026-09-25; for the owner's morning review):
- **(A)** a plan records resident weight addresses at admission and every
  step revalidates the leases through a new `DenseGraphStep` field (about
  twenty construction sites in tests). Hard part: an in-flight operation
  (between `execute` and `finish`) does not borrow the weights, so the leases
  could be released under a running step unless a separate in-use count
  refuses it.
- **(B, recommended)** a new executor type owns the weight leases **and**
  every plan admitted against them (the bucket set), and drives steps itself;
  plans cannot leave it, so no plan or captured graph outlives the weights by
  construction, and no existing struct changes.
Either is a new executor API on M6's critical path with asynchronous
ownership at its core, the class coordinator.md assigns to the Opus
`builder` or to a coordinator-designed contract; the coordinator will design
(B) to file level unless the owner directs otherwise.

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

### Slice 7 checkpoint route (coordinator, 2026-09-25; owner decisions marked)

Proposed checkpoint for the exit's "actual available checkpoints" rows:
`/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` (first in the v1 catalog;
compressed-tensors `pack-quantized`, INT8, group 32, symmetric; 60 layers,
hidden 5376, intermediate 21504, vocabulary 262144; 7 shards, 33 GB). It
does not fit one 3090, so it runs as a pipeline over the three GPUs (64 GB
total) or TP2 on the pair (48 GB) plus paged state. Known work before it can
run, each a bounded task:
1. **Canonical artifact (owner decision).** ADR 0020: conversion is
   user-managed, and no agent may start one without a task naming artifact,
   revision, expected size and retention. The M3 importer reads
   `pack-quantized`; the owner either runs `moxie-repack` on this revision or
   authorizes a task that names it.
2. Full-size Gemma graph: `moxie-models` builds `Gemma4Text::reduced` only,
   and its INT8 allowance is explicitly absent (`gemma4.rs` about 405). With
   task 0080 the format is a planner input, so the graph can stay BF16-typed.
3. The affine kernel's qualified `max_input` (16,384) is below this
   checkpoint's `down_proj` input (21,504): a qualification task (next
   section).
4. Plan sets per pipeline stage (task 0091 is single-GPU), with formats.
5. Weights loaded from the canonical artifact into the residency authority
   (`CanonicalSource`/`ShardSource` exist from M2/M3).
6. Quality oracle: `transformers 5.5.3` + `torch 2.10` on the host produce
   reference logits for the same token ids (the reference tokenizer only
   makes inputs). Paired with the fixture-scale method of task 0090.
**Owner decisions:** the checkpoint choice, and item 1.

**Owner ruling (2026-09-25):** the checkpoint is confirmed. The owner
authorized the coordinator to run the conversion under ADR 0020, with this
scope. Artifact `cyankiwi/gemma-4-31B-it-AWQ-8bit`, at revision
`34ca187d836de874b2c7e3edf48f439b9f583772`. Expected size is about 35 GB (the
source payload is 35,089,877,112 B). The conversion writes under `/models`:
plan `/models/gemma-4-31B-it-AWQ-8bit.plan.toml`, and output
`/models/gemma-4-31B-it-AWQ-8bit-moxie`. Retention: until the owner removes
it. Command: `moxie-repack plan`, then `moxie-repack repack`. The plan lists
1,188 selected tensors and 0 skipped. `/` had 67 GB free before the run.
**Published (2026-09-25):** outcome `published`, artifact identity
`f99f370e7b1af3ce2e34f70b21157b11071381dc8a0507e00460daf60739fdcb`,
35,090,087,216 B in 9 shards plus `manifest.toml`, 2,362 units written, not
resumed. The source shard digests are recorded in the run's output. `/`
had 35 GB free (99% used) right after the run. The owner then freed space: 321 GB
free (82% used), and the artifact is unaffected (2026-09-25).

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

### Carried to M7: DeepSeek speed against Strata (owner, 2026-09-25)

This is not part of M6's exit, and M6 does not wait for it. The owner's
ruling: Moxie succeeds Strata, and Strata's main effort and success was
DeepSeek (`Intel/DeepSeek-V4-Flash-0731-W4A16-AutoRound`, v1 entry 5), so
Moxie must show DeepSeek speed at least equal to Strata's. The measured
comparison belongs to M7's DeepSeek work. As M7 preparation, the owner had
[task 0093](../tasks/0093-m7-prep-deepseek-strata-speed-map.md) start now,
run by a separate Claude Opus `builder` and documentation only. It maps
every Strata DeepSeek speed technique onto Moxie as present, planned or gap,
and records Strata's baseline figures (for example 26.231 prefill tok/s at
1,925 tokens and 8.627 decode tok/s, `docs/models/deepseek.md`). Its gap
rows are registered for M7; they do not reorder M6. The M7 roadmap's
DeepSeek row gains the speed comparison when M6 closes.

**Accepted 2026-09-25** ([map](../evidence/deepseek-strata-speed-map.md)):
of 117 Strata DeepSeek techniques, 11 are present in Moxie, 26 partial, 7
not needed, 24 planned and 49 gaps. Of the gaps, 15 are deliberate (formats
Moxie does not admit, the exact-reduction and single-user rules), and 9 were
rejected in Strata. The other 25 were accepted in Strata and have no Moxie
home. The largest cluster is the host expert path: Strata's decode rate is set by
the host DRAM expert read, and it used an all-resident expert arena, NUMA
binding and a worker pool, where Moxie's host expert kernel is
single-threaded. Loading, page bookkeeping, kernels and launch, and admission
make up the rest. M7's DeepSeek work starts from this map.

## Next task

[Task 0076](../tasks/0076-m6-device-kv-append.md), assigned to luna on
2026-09-24. The paragraph below is the pre-authorization note, kept for
provenance.

None is authorized. M6 ("shared performance paths and phase balance")
opens only on the owner's instruction. When it does, the natural first bounded
deliverable is M6.1's device-resident layer chain on one GPU, with its own
contract and stop conditions. The carried maintenance items above are its
candidates for a cleanup slice.
