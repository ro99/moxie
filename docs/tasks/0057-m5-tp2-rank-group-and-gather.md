# Task 0057 — TP2 rank group and ordered all-gather on the 3090 pair

Status: **accepted** (owner, 2026-09-22). Built by Claude Opus `builder`; reviewed
by Codex `sol`; round 2 ACCEPT. The coordinator re-ran the GPU tests (4/4 on
the 3090 pair) and `arch-check`.

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

- **Design decisions (coordinator, 2026-09-22):**
  - **One host thread drives both ranks.** Document 01's one execution
    thread per GPU is **not yet met**. `RankContext`, `DeviceBuffer` and
    `DeviceRange` are `!Send`, and a rank thread would need a cross-thread
    handle to its peer's range. The coordinator recorded thread-per-rank
    execution as a named M5.2 obligation for the device-lowering slice.
  - **The deadline.** Every wait the group makes is a bounded poll of the
    completion event up to the group's deadline. If completion is not
    observed by then, the collective returns `Error::DeviceLost`, naming the
    lease and the deadline, and the ranges stay withheld under the existing
    lease rule. There is no `moxie-types` change.
  - **No hidden host staging** (accepted as a contract requirement).
    `cuMemcpyPeerAsync` silently stages through the host when peer access is
    not enabled. So `moxie-cuda` refuses a peer copy unless the destination
    context has enabled access to the source device.
  - **Drain before release** (review round 1, C1). A collective refused
    before its first copy settles each shard's producer and releases
    everything. Every per-collective event and output range is obtained
    before the first copy. Once copies are enqueued, both destination
    streams read both sources, so nothing is released until both streams'
    drain events are observed within the deadline. If either is not, every
    range is withheld on both ranks: input, weight shard, local output and
    gathered output. The error is `DeviceLost` naming the collective and its
    deadline. This follows Strata's "the failed command is drained by a
    single owner".
  - **A peer grant names the acquisition** (review round 1, M1).
    `cuCtxEnablePeerAccess` grants access to a context, not a device, and a
    context handle can be reused. Each `RankContext` therefore carries a
    process-unique acquisition generation, and a grant records the peer's
    generation. A context reacquired after release is not authorized by the
    old grant.
  - **Sequence and shape are checked separately.** A rank's sequence is
    checked against the group's schedule, and the ranks' shape and precision
    are checked against each other. Neither check covers the other's case,
    which is what lets mutation 2 below be observed at all.
- **Changed owners and consumers; source commit:** uncommitted on base
  `b774456`.
  - `moxie-cuda` (`ffi.rs`, `driver.rs`, `status.rs`) adds bindings for
    `cuCtxEnablePeerAccess`, `cuMemcpyPeerAsync` and `cuStreamWaitEvent`.
    The safe wrappers are:
    - `RankContext::enable_peer_access`: checks `cuDeviceCanAccessPeer`
      first and is idempotent (code 704);
    - `Stream::wait_event`;
    - `DeviceBuffer::copy_from_peer_async_at`: `unsafe` only for the
      lifetime obligation, and guarded by the peer grant.

    Code 217 (`PEER_ACCESS_UNSUPPORTED`) now classifies as
    `Unsupported { capability: "peer_access" }`.
  - `moxie-executor`: `src/tensor_parallel.rs` (new, `driver` feature) holds
    `RankGroup::form`/`all_gather`, `ColumnLinear::load`/`launch`,
    `RankShard::finish`, `Gathered` and `GatherRefused`. In round 2,
    `all_gather`'s failure paths became `refuse_before_copies` (nothing in
    flight) and drain-or-withhold (copies enqueued). Two
    crate-private accessors were added: `OperationLease::completion` (for
    `cuStreamWaitEvent`) and `DeviceRange::copy_from_peer_async_at`. There
    is no lease or ledger semantic change.
  - `shard_weight_ranges` has its first real byte consumer: each rank
    uploads only its rows.
- **Commands, GPU UUIDs; passed / failed / skipped:**
  - The TP2 pair is `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` + `GPU-81fe4578-59b2-37c4-421e-287cdac78704`
    (the 3090s, chosen by mutual `cuDeviceCanAccessPeer`).
  - The refused pairing is `GPU-3032cfa3-…` + `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`
    (the 5060 Ti).
  - `cargo test -p moxie-executor --features driver --test tensor_parallel_device -- --nocapture`:
    4 passed.
    - Bit identity: a 256x1024x2048 linear, split over the pair and
      gathered, equals the same linear on one 3090 on both ranks, bit for
      bit. `out_features = 2047` is refused by `shard_weight_ranges` (`Dim`)
      with zero arena allocations.
    - The injected failures (below).
    - A missed drain: the group deadline is zero, and a test gate (the
      `cuEventRecord` interposer from `device_arena.rs`) holds rank 0's
      stream before its drain event. Both ranks get `device_lost`, and all 4
      ranges stay withheld on each rank.
    - The 5060 Ti pairing is refused. A direct peer copy without a grant is
      refused. After the peer context is dropped and reacquired, a copy
      from the new context is refused even though the old one was granted.
  - `cargo fmt --all -- --check`,
    `cargo clippy --workspace --all-targets --locked -- -D warnings`,
    clippy on `moxie-executor`/`moxie-cuda` with `driver` and `-D warnings`,
    `cargo xtask arch-check` and `cargo xtask spec-check`: all passed.
  - `cargo test --workspace --locked`: round 2: exit 0; 109 test binaries, 1232 passed, 0 failed, 0 ignored (log /tmp/claude-1000/-home-rodrigo-Developer-moxie/75e03800-b1df-4452-b029-94f209bf853d/scratchpad/t57r2-workspace.log).
  - `cargo test -p moxie-executor --features driver --locked`: round 2: exit 0; 21 test binaries, 162 passed, 0 failed, 0 ignored (log /tmp/claude-1000/-home-rodrigo-Developer-moxie/75e03800-b1df-4452-b029-94f209bf853d/scratchpad/t57r2-driver.log); `moxie-cuda --features driver` 21 passed.
  - No timing claim.
- **Failure-injection results** (one test; each case asserts a typed error
  on both ranks, completion within the 10 s deadline, and zero live arena
  ranges on both ranks afterwards):
  - Rank 1's own work fails before the collective: its launch is refused
    by `shard_weight_ranges` (`out = 2047`), and `dim` reaches both ranks.
    Every case asserts completion within a tenth of the deadline, not
    merely within it. Deadline expiry is covered by the missed-drain test.
  - Rank 1 declares the next sequence: `invalid_request` on both.
  - Rank 1 declares another shape: `invalid_request` on both.
  - Cancellation between the rank launches (rank 1's launch refuses):
    `cancelled` on both.

  A clean collective then succeeds on the same group, both arenas close,
  and both ledgers have nothing outstanding.
- **Mutation results and restoration** (each applied, the device test file
  run, the source restored from a copy and checked with `cmp`):
  1. Gather rank order swapped: the bit-identity test fails ("rank 0's
     gather differs from one GPU").
  2. Rank 1 skips the sequence check: the injected-failure test fails, since
     "rank 1 declares the next sequence" returns `Ok`.
  3. The destination lease no longer waits for the copy's event. Round 1:
     the injected-failure test failed in 5 of 5 runs, through `cuMemFree`
     error 719 at arena close. Round 2: the missed-drain test fails
     deterministically in 3 of 3 runs, because the collective returns `Ok`
     while rank 0's stream is held. In one of those runs, the bit-identity
     test also read a wrong value.
  4. The hidden-staging guard is removed: the refusal test fails, because a
     copy from a 3090 into the 5060 Ti without a grant returns `Ok(())`
     (staged by the driver).
  5. Round 2, C1: sources are released without draining the destinations.
     The missed-drain test fails: live ranges are (1, 1), not (4, 4).
  6. Round 2, M1: the grant is looked up by UUID only. The refusal test
     fails, because the copy from the reacquired peer context returns
     `Ok(())`.

  Every mutation above was re-run against the round 2 source, and each
  file was restored and checked with `cmp`.
- **Remaining obligations:**
  - Reduction and its numerical gate (the owner's decision).
  - Device RoPE, and wiring task 0056's lowering to the device.
  - Rank-step atomicity across layers (task 0056).
  - Thread-per-rank execution, with a cross-thread peer-range handle
    (above).
  - A pitched peer copy in place of one copy per row (marked `ponytail:` in
    the source).

- **Review trail (sol).**
  - **Round 1, CHANGES REQUESTED:**
    - **Critical:** source shards were released after only their producer
      event, while the peer's destination stream could still be reading
      them (a use-after-free on a failed or timed-out gather).
    - **Major:** peer grants were recorded by UUID, but CUDA grants access
      per context, so a reacquired context was authorized without being
      granted.
    - **Moderate:** the rank-failure test injected `DeviceLost` directly.
  - **Round 2 fixes:**
    - Drain the destination streams, then release; if the drain misses the
      deadline, withhold every range. This follows Strata's "the failed
      command is drained by a single owner".
    - Grants carry the acquisition generation of the context they were
      given to.
    - The rank-failure test injects a real typed refusal.
  - **Round 2 verdict:** ACCEPT. One non-blocking note: about 75 lines of
    `cuEventRecord` gate setup in `tests/tensor_parallel_device.rs` duplicate
    `tests/device_arena.rs`. That is recorded as a follow-up, not held
    against acceptance.

