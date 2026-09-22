# Task 0059 — the reduced dense Gemma 4 graph runs on one GPU

Status: **proposed**.

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

- Design decision (phase 1) and coordinator answer:
- Changed owners and consumers; source commit:
- Commands, GPU UUIDs; passed / failed / skipped:
- Mutation results and restoration:
- Remaining obligations:
