# Task 0057 — TP2 rank group and ordered all-gather on the 3090 pair

Status: **proposed**.

## Identity and authority

- Task0057, M5.2 slice 2: the first device TP work. Builder Claude Opus
  session `builder` (`/ponytail:ponytail`); reviewer Codex `sol` (read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus `coordinator`. Owner
  accepts.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `41f9b7a`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035).
- Requirements:
  - Roadmap M5.2: "TP2 on the 3090 pair … collective ordering and failure
    handling. Compare to single-rank reference."
  - M5 exit: "Inject rank failure and collective mismatch safely in the
    harness … No TP3 speed guarantee or hidden peer-to-host fallback."
  - Document 04, lines 55–57: rank groups with a monotonically ordered
    collective schedule and consistent tensor sizes and types; timeouts;
    group-wide failure propagation; "Cancellation cannot leave one rank
    waiting forever"; "A failed rank invalidates the in-flight transaction".
  - Document 02: the executor owns collectives; `moxie-cuda` owns the audited
    unsafe FFI.
- O6/O7 are open: no timing and no transport-performance claim. No NCCL: a
  third-party collective library would need its own ADR, and this slice needs
  none.

## Facts established before writing (coordinator, 2026-09-22)

- **Device ops available.** The device BF16 chain supports `Linear`,
  `RmsNorm` and `Residual` (`moxie-plan/src/selected.rs` `semantic()`), plus
  paged attention. There is no device RoPE, so task 0056's attention head
  chain cannot run on the GPU yet. This slice therefore exercises the
  collective on a column-sharded BF16 `Linear`, which can.
- **Bit-identity is the right gate.** `moxie_bf16_linear_v1`
  (`moxie-kernels/cuda/bf16_chain.cu`) computes each output element with a
  fixed ascending loop over `k`, independent of `output_width`. A
  column-sharded linear followed by a concatenating gather therefore equals
  the single-GPU linear bit for bit. No reduction is involved.
- **Peer access.** Peer access is granted only within the 3090 pair
  (`docs/evidence/hardware-inventory.md`, `docs/evidence/topology-p2p.md`;
  `cuDeviceCanAccessPeer`). `moxie-cuda` queries it
  (`driver.rs` around line 133) but has no `cuCtxEnablePeerAccess` or
  `cuMemcpyPeerAsync` binding. Ordinal 0 is the 5060 Ti; identify GPUs by
  UUID, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **First real byte consumer.** Task 0054's
  `moxie_executor::shard_weight_ranges` has never been called with real
  weight bytes. This slice is its first consumer.

## Bounded deliverable

- **Outcome:** in `moxie-executor`, a two-rank group over the 3090 pair and
  one ordered all-gather collective, proven by a column-sharded BF16 `Linear`:
  - Each rank uploads only its `shard_weight_ranges` rows.
  - Each rank runs the existing `moxie_bf16_linear_v1` over its columns.
  - Each rank receives the gathered `[rows, out]` result.
  - The gathered output equals the same linear on one 3090, bit for bit.
- **Rank group:** built from device UUIDs.
  - It refuses a group whose members lack mutual peer access, with a typed
    error. The 5060 Ti with either 3090 must be refused.
  - There is no host-staged fallback. Document 04 forbids a *hidden* one, and
    an explicit one is not needed here.
  - Enabling peer access is part of forming the group.
- **Collective:** each call carries a sequence number and a declared shape and
  dtype, and every rank must agree on them before bytes move. The transfer is
  a direct peer copy, ordered by events on each rank's stream. Buffers stay
  leased until the copy's completion event, following the existing lease
  rules.
- **Failure handling (M5 exit):** tests must inject all three:
  - **Rank failure:** one rank's work returns an error before the collective.
  - **Collective mismatch:** ranks disagree on sequence or shape.
  - **Cancellation between rank launches.**

  Each must end in a typed error on both ranks, bounded in time with no
  indefinite wait, with every lease and allocation released. The group must
  then be able to run a clean collective afterwards.
- **`moxie-cuda`:** add the smallest safe wrappers over
  `cuCtxEnablePeerAccess` and `cuMemcpyPeerAsync` (or the driver's
  equivalents). Errors are typed; there are no string-matched CUDA messages
  and no panic across FFI.
- **Non-goals:** no NCCL; no reduction or all-reduce (row-parallel needs a
  numerical gate, which is an owner decision); no RoPE or attention on the
  device; no wiring of task 0056's lowering to the device; no TP3 or 5060 Ti
  participation; no timing; no model-crate edits.

## Contract

- **Equations:** unchanged. The gather concatenates columns in rank order,
  which is global column order.
- **Memory:** each rank admits its weight shard, its local output and the
  full gathered output through the existing ledger/arena. Nothing is
  allocated outside admission. Peer-copy destinations are leased until their
  event completes.
- **Failure and cancellation:** as above. A failed step leaves no rank
  holding a partially gathered result that a caller can read as valid.
- **Oracle:** the same BF16 linear on one 3090, compared bit for bit. The
  weights are synthetic and deterministic; no checkpoint is read.

## Acceptance

- Host lanes: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` and
  `cargo xtask spec-check`.
- Driver lane: `cargo test -p moxie-executor --features driver` for the new
  tests.
- GPU lane: the real 3090 pair, identified by UUID in the test output.
  - The bit-identity case covers at least two shapes: one where `out` splits
    evenly, and one that `shard_weight_ranges` refuses as non-divisible,
    proving the refusal happens before any device work.
  - The three failure injections.
  - The 5060 Ti pairing refused.
- Mutation proof, run and restored:
  - Swap the gather's rank order.
  - Let one rank skip the sequence check.
  - Drop the lease wait on the peer-copy destination.

  Each must fail a test. If one does not, say which, and explain why no test
  can observe it; do not add a test merely to catch it.
- One test per invariant.
- Stop conditions:
  - Peer access is not actually granted on this machine.
  - The work would need a lease or ledger semantic change.
  - Bit-identity fails for a reason that is not a defect.

## Result, filled after work

- Changed owners and consumers; source commit:
- Commands, GPU UUIDs; passed / failed / skipped:
- Failure-injection results:
- Mutation results and restoration:
- Remaining obligations (reduction and its numerical gate, device RoPE,
  device lowering, rank-step atomicity):
