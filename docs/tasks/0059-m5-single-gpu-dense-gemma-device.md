# Task 0059 — the reduced dense Gemma 4 graph runs on one GPU

Status: **accepted** (coordinator, 2026-09-22, under the owner's auto-mode
delegation). Built by Codex `luna`; reviewed by Codex `sol` over four
rounds. Round 4 ACCEPT. The coordinator ran the full `cargo xtask test-gpu`:
63 passed, 0 failed, sm_86 and sm_120 qualified.

## Identity and authority

- Task0059, first half of M5 plan slice 2 ("Dense TP2 on the 3090 pair").
  The second half, task 0060, splits this execution across the pair. Builder
  Codex `luna` (max, `/ponytail:ponytail`); reviewer Codex `sol` (read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus `coordinator`.
  Accepted by the coordinator under the owner's auto-mode delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `0622397`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035).
- Requirement: the M5 exit requires "same dense … graph definitions execute
  single GPU, TP and PP without model edits". The dense graph definition in
  use is the reduced Gemma 4 graph (`moxie_cli::gemma::build_with_config`).
  Today it cannot execute on a GPU at all, so TP2 of it (task 0060) has
  nothing to split.
- O6/O7 are open: no timing and no performance claim.

## Facts established before writing (coordinator, 2026-09-22)

**Moxie:**
- The selected device package (`moxie-plan/src/selected.rs`, `semantic()`
  and the params match around lines 580–630) admits only:
  - `Linear` without bias;
  - `RmsNorm` with `group == 1` (a grouped norm is refused: "no qualified
    device kernel normalizes {group} groups");
  - `Residual` with `scale == 1.0`;
  - paged attention (`selected_attention.rs`).

  The reduced Gemma graph also needs `Embedding` (with its scale), grouped
  per-head `RmsNorm`, `Rope` (`HalfSplit`, with partial rotation on global
  layers), `GeGlu` (gelu-tanh), a scaled `Residual` where the model uses
  one, and `VocabProjection` (FP32, unrounded) with the logit softcap.
  Inventory the exact list from the graph; do not trust this one.
- **Declared device-versus-host gate:** task 0012's chain lane in
  `xtask/src/gpu.rs` (around lines 995–1040) compares against the host
  interpreter. It requires bit-exactness on the exactly-representable fixture
  and at most **1 BF16 ULP** otherwise. New kernels use this same gate. It is
  already declared, so there is no new tolerance.
- The host RoPE oracle (`moxie-oracles/src/rope.rs`, around line 118)
  computes each angle in f64 with libm `powf`/`sin`/`cos`, then casts to f32.
  The host GELU (`moxie-oracles/src/activation.rs`, `gelu_tanh`) uses f64
  `tanh`.

**Strata** (read-only):
- `docs/dsv4-rank-local-architecture.md` around line 533: "The rotation angles
  are still evaluated on the host and uploaded: host libm and device
  trigonometry differ in the last ulp." Follow this: device RoPE consumes a
  host-computed cos/sin table built with the oracle's own f64 arithmetic, so
  RoPE can be bit-exact.
- Kernels to read for semantics, not to copy bit for bit:
  `gemma4_norm_rope_kernel` and `gemma4_norm_rope_qkv_rows_kernel`
  (`kernels/cuda/detail/backend_kernels.cuh`, around line 965), and
  `gemma4_geglu_kernel` (around line 1562). Strata's norm sums squares in
  FP64 through a warp shuffle. Moxie's kernels follow Moxie's oracle order,
  as the existing chain kernels do ("fixed ascending reductions").

## Bounded deliverable

- **Outcome:** the reduced dense Gemma 4 graph (no routed experts), built by
  the unchanged model code, lowers through the selected device package and
  executes on **one** GPU:
  - a multi-row prefill, then decode steps, through paged attention;
  - logits within the task 0012 gate against the host interpreter;
  - on a 3090 and on the 5060 Ti (SM86 and SM120).
- **Kernels:** add the missing ones as shared kernels keyed by semantic
  op, shape and layout, never by model name, in `moxie-kernels`, with their
  binding and selected-package admission. Each follows its host oracle's
  declared order. RoPE takes an uploaded angle table. GELU-tanh may differ by
  a last ulp inside the 1-ULP gate; say whether it does.
- **Admission:** the new ops' buffers and workspaces go through the existing
  ledger and arena. Nothing is allocated outside admission.
- **Non-goals:** no TP (task 0060); no FP32-partial linear (task 0060); no
  quantized weights, which stay BF16 as in the reduced graph; no model-crate
  edits; no timing; no MLA or routed ops.

## Phase 1 — design proposal before code

Send a `DECISION` report of 40 lines or fewer covering:

1. **The exact missing-op inventory**, taken from the reduced dense graph.
2. **For each new kernel:** its reduction or evaluation order relative to the
   oracle, and whether it will be exact or within 1 ULP.
3. **RoPE's angle table:** who computes it, where it lives, and how it is
   admitted, following the Strata precedent.
4. **Whether this fits one task.** If not, propose a split, for example
   "elementwise kernels" and then "graph admission and end-to-end".
5. **Files, and any new crate edge.**

## Acceptance

- Host lanes pass: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check` and
  `cargo xtask spec-check`.
- The driver lane and the GPU lane (`cargo xtask test-gpu` or the focused
  equivalent) pass on both 3090s and the 5060 Ti, with UUIDs in the output.
- **End-to-end:** reduced-Gemma logits from the device versus the host
  interpreter, prefill plus decode, within the task 0012 gate on all three
  GPUs.
- **Per-kernel qualification:** one test per new kernel against its oracle,
  unless the end-to-end test already fails on every plausible mutation of
  that kernel. Say which.
- **Mutations:** one per new kernel (for example, a wrong RoPE pair layout, a
  GLU with gate and up swapped, a grouped norm reducing the whole row), each
  run and restored.
- **Stop conditions:**
  - A kernel cannot meet the 1-ULP gate for a reason that is not a defect.
    Report it; do not loosen the gate.
  - Execution needs a model edit.
  - A change to lease or ledger semantics is required.

## Result, filled after work

- Design decision (phase 1) and coordinator answer: one additive task is
  sufficient. The selected dense package now admits the exact reduced Gemma
  inventory: scaled embedding, grouped per-head RMSNorm, HalfSplit RoPE,
  GeGLU, scaled residual, unrounded FP32 vocabulary projection, and the
  existing linear/RMS/residual/paged-attention operations. RoPE angles are
  computed in the executor with the host f64 `powf`/`sin`/`cos` formula and
  uploaded through admitted workspace. The production duplication is
  intentional: the executor does not depend on `moxie-oracles`; one
  dev-only test cross-checks the uploaded table against that oracle. The
  coordinator's answer was to keep one end-to-end gate, add per-kernel tests
  only where it cannot catch a mutation, run prefill and decode on all three
  GPUs, and stop rather than loosen the one-BF16-ULP gate.
- Changed owners and consumers; source commit: `moxie-types` owns the new
  semantic operations, operands, rounding and workspace contracts;
  `moxie-kernels` owns the shared CUDA dense operations, image and catalogue;
  `moxie-plan` owns dense graph admission and arena/workspace accounting;
  `moxie-executor` owns binding, host angle generation, paged-state execution
  and retirement. The test-only edges are `moxie-executor` -> `moxie-cli`,
  `moxie-interp` and `moxie-format`; no model crate was changed. Changed paths
  are `Cargo.lock`, `crates/moxie-executor/Cargo.toml`,
  `crates/moxie-executor/src/{chain.rs,dense.rs,lib.rs}`,
  `crates/moxie-executor/tests/dense_gemma_device.rs`,
  `crates/moxie-kernels/{build.rs,src/lib.rs,cuda/dense_graph.cu,cuda/dense_ops.cu}`,
  `crates/moxie-plan/src/selected.rs` and
  `crates/moxie-types/src/capability.rs`, plus this task Result section.
  Source commit was `fe1d8a9`; no commit was made for this task.
- Commands, GPU UUIDs; passed / failed / skipped: PASS `cargo fmt --all
  -- --check`; PASS `cargo clippy --workspace --all-targets --locked --
  -D warnings`; PASS the feature-enabled executor clippy lane; PASS `cargo
  test --workspace --locked`; PASS `cargo xtask arch-check`; PASS `cargo
  xtask spec-check`; PASS the focused uploaded-angle oracle test. The focused
  GPU equivalent passed prefill and decode for both reduced fixtures on every
  visible GPU, with maximum 0.000 BF16 ULP: `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`
  (SM120, 5060 Ti), `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` (SM86,
  3090), and `GPU-81fe4578-59b2-37c4-421e-287cdac78704` (SM86, 3090).
  No required gate failed. The full `cargo xtask test-gpu` matrix was not
  invoked; its task-approved focused equivalent was run. No timing was taken.
- Mutation results and restoration: the end-to-end gate caught each mutation
  and every source was restored before the final clean run: embedding scale
  set to one (94.5 ULP), grouped RMS reduced over the whole row (2 ULP), RoPE
  half-split pair order swapped (4 ULP), GeGLU gate/up swapped (27 ULP),
  scaled residual changed to subtraction (18 ULP), and vocabulary projection
  column shifted (90.75 ULP). Each failed on the first SM120 prefill as
  expected; the restored final run in
  `reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu` passed on
  all three UUIDs. Device GELU and softcap use `tanhf`; the observed 0 ULP is
  fixture-specific and is not a general bit-identity claim.
- Remaining obligations: task 0060 still owns TP2 device splitting and
  ordered partial-linear combination; MLA and routed operations, quantized
  device weights, vocabulary-parallel embedding, PP and a real checkpoint
  execution remain outside this slice. No performance or general model
  support claim follows from this synthetic reduced-Gemma prefill/decode gate.
- Round-2 repair: every host-to-device source in the dense path is now held by
  `DenseOperation` until its completion is observed. Token and position index
  sources stay in their `OwnedBinding`s; the RoPE angle bytes stay in the
  pending-upload slot through both the asynchronous copy and stream
  synchronization, and remain held by the refused operation on either error.
  `IndexView` decodes retained index bytes without cloning positions or token
  IDs into `Vec<u64>`. The existing GPU test file gained
  `angle_copy_sync_failure_retains_its_host_source`, which injects a failed
  synchronization after the angle copy and proves the exact angle source
  remains retained. The duplicate executor `moxie-format` dev dependency was
  removed. Round-2 changed paths are
  `crates/moxie-executor/src/dense.rs`,
  `crates/moxie-executor/tests/dense_gemma_device.rs`, this Result section,
  and `crates/moxie-executor/Cargo.toml`.
- Round-2 gates: PASS `cargo fmt --all -- --check`; PASS `git diff --check`;
  PASS `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  PASS the feature-enabled executor check/clippy lanes; PASS `cargo
  test --workspace --locked`; PASS the focused angle-oracle unit test; PASS
  `cargo xtask arch-check`; PASS `cargo xtask spec-check`; and PASS
  `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --features
  driver,paged-attention-binding --test dense_gemma_device -- --nocapture`.
  The last command ran both the retention fault test and the existing
  prefill/decode gate on all three UUIDs above. A command using only
  `driver` runs zero tests for this integration target; the required
  `driver,paged-attention-binding` feature set is the reported GPU command.
  No gate failed, no job remains running, and no commit was made.
- The observed maximum of 0 ULP is fixture-specific: device GELU and softcap
  use `tanhf`, so the result is reported only as within the task's one-BF16-ULP
  gate, not as general bit identity. The six round-1 mutations remain the
  end-to-end mutation results above; the round-2 synchronization fault was
  caught by `angle_copy_sync_failure_retains_its_host_source`, and all sources
  were restored to the clean implementation before the final gates.
- Round-3 repair (review finding R2-1): `publish_page_table` now validates
  physical-page uniqueness in place, removing the pages-byte `Vec<bool>`.
  Its little-endian upload bytes are a reusable buffer allocated during
  `PagedAttentionRun::admit` and charged by the shared paged-attention
  `resource_request` as host pageable workspace. The buffer is moved into the
  run's retained source before the asynchronous copy and returned to the
  admitted workspace only after synchronization succeeds; copy or sync failure
  therefore keeps it quarantined. This common request covers both direct
  `PagedAttentionRun` callers and the runs used by the dense path; the dense
  prefill/decode gate asserts that the page-table reservation remains present
  while each phase's K/V workspace is admitted. Round-3 changed paths are
  `crates/moxie-executor/src/paged_attention.rs`,
  `crates/moxie-executor/tests/dense_gemma_device.rs`, and this Result section.
- Round-3 gates: PASS `cargo fmt --all -- --check`; PASS `git diff --check`;
  PASS `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  PASS the feature-enabled executor check/clippy lanes; PASS `cargo
  test --workspace --locked`; PASS `cargo xtask arch-check`; PASS `cargo
  xtask spec-check`; PASS `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p
  moxie-executor --features driver,paged-attention-binding --test
  dense_gemma_device -- --nocapture` on all three UUIDs; PASS
  `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --features
  driver,paged-attention-binding,paged-attention-test-hooks --test
  paged_attention_device -- --nocapture` (7/7); and PASS the focused direct
  admission test with the same features. No gate failed, no job remains
  running, and no commit was made. The carried documentation changes remain
  untouched.
- Round-4 repair (review finding R3-1): page-table uniqueness validation now
  reuses the admitted upload buffer as an O(pages) bitset, then overwrites the
  same buffer with little-endian entries before enqueue; the quadratic prefix
  scan and all new allocation are gone. The statement that nothing is
  allocated outside admission applies to data and upload buffers only. The
  pre-existing control metadata (`PagedAttentionRun.page_table` and
  `moxie-state`'s `DeviceKvSequence::page_view_for`/`placements_for` vectors)
  remains outside this host request and is a separate coordinator-recorded
  obligation; this task does not claim total host peak equals admission.
- Round-4 gates: PASS `cargo fmt --all -- --check`; PASS `git diff --check`;
  PASS `cargo clippy --workspace --all-targets --locked -- -D warnings`;
  PASS the feature-enabled executor clippy lane; PASS
  `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test -p moxie-executor --features
  driver,paged-attention-binding --test dense_gemma_device -- --nocapture`
  on all three GPU UUIDs; and PASS the feature-enabled
  `paged_attention_device` test (7/7); and PASS the focused direct admission
  test with the test hooks. No commit was made.
