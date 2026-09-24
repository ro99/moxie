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

## Next task

None is authorized. M6 ("shared performance paths and phase balance")
opens only on the owner's instruction. When it does, the natural first bounded
deliverable is M6.1's device-resident layer chain on one GPU, with its own
contract and stop conditions. The carried maintenance items above are its
candidates for a cleanup slice.
