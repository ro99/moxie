# Task 0081 — the single-GPU dense step executes affine linear weights

Status: **active** (coordinator, 2026-09-24). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0081, M6 slice 2, roadmap **M6.1** "fused dequantization … where
  supported". Second half of [task 0080](0080-m6-affine-linear-admitted-in-dense-plans.md).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. On a conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels or strings.
- Moxie never quantizes (ADR 0017): the fixture writes synthetic codes and
  scales directly. Numerical gates are not loosened (AGENTS.md; ADR 0028).

## Facts established before writing (coordinator, 2026-09-24)

- After task 0080, a candidate from `lower_selected_with_formats` carries
  `weight_formats()`; a formatted `Linear` node's descriptor names the single
  symbol `AFFINE_LINEAR`, which the plan's cached dense module resolves with
  every other selected symbol (task 0079). The weight value's arena range holds
  `WeightFormat::sections(out, in)`: codes, scales, zero points (if any), group
  index (if any), each at a 16-byte-aligned offset.
- The standalone launch (`affine_linear.rs` about 1597–1644) is the ABI to
  copy: 14 parameters in the order `x, codes, scales, zero_points (0 if
  symmetric), group_index (0 if unmapped), output, rows, in_features,
  out_features, row_stride, groups_per_row, code_bits (u32), group_size (u32),
  scale_kind (u32: F16 0, BF16 1, F32 2)`; grid `(ceil(out/16), ceil(rows/16),
  1)`, block `(32, 1, 1)`, no shared memory. `row_stride` is the packed row
  bytes; `groups_per_row = ceil(in / group)`.
- `validate_bindings_except` (`chain.rs` about 800–845) requires
  `binding.bytes.len() == planned.logical_bytes` (satisfied by 0080's sizing)
  **and** rejects any weight whose bytes are not finite BF16 words. Integer
  codes are arbitrary bytes, so that check would refuse a valid affine weight
  at random.
- The dense test's host reference binds float values; `affine_linear_device.rs`
  `oracle` (about 236) reconstructs the weight in FP32 and rounds each element
  to BF16, which is the kernel's contract.

## Bounded deliverable

- **Outcome:** a reduced dense Gemma step whose chosen linears are bound as
  affine INT4/INT8 runs on one GPU, on both 3090s and the 5060 Ti, and matches
  a host reference that binds the reconstructed, BF16-rounded weights.
- **Allowed files:** `crates/moxie-executor/src/{chain.rs,dense.rs}`,
  `crates/moxie-executor/tests/dense_gemma_device.rs`, this task's Result.
- **Non-goals:** TP/PP with formats; quantized experts; raising the affine
  kernel's `max_input` bound; importer or checkpoint work; any timing.

## Numbered changes

1. **`chain.rs`, binding validation.** For a binding whose value is in
   `plan.candidate().weight_formats()`, skip the BF16-finiteness check and
   instead check the scale section (from `sections`): every scale, decoded by
   the format's scale precision, is finite and nonzero (ADR 0030; F16: exponent
   bits not all ones and magnitude bits nonzero; BF16 likewise; F32:
   `is_finite() && != 0.0`). Refuse with the file's `invalid("bindings", …)`
   naming the value. Nothing else changes.
2. **`dense.rs`, launch.** In the `OpParams::Linear` arm, when the node's
   weight (`node.inputs[1]`) is in the candidate's `weight_formats()`, launch
   the affine symbol with the ABI above instead of the BF16 linear. Addresses:
   `x` = input 0, output = node output, and each component = the weight
   range's address plus its section offset. Use checked arithmetic and the
   file's `invalid(…)` for every conversion. Call `launch(…)` with
   `moxie_kernels::AFFINE_LINEAR` as its symbol argument: since task 0080's
   R1 repair, `launch` refuses when the resolved symbol differs from the one
   the arm prepared arguments for. `push_launch(lease, "affine-linear")`. Every other path is unchanged.
3. **Test** (one, in `dense_gemma_device.rs`):
   `affine_linear_weights_match_host_on_every_gpu`. Shape A. Formats on one
   layer's attention projections (INT8, group 32, BF16 scales, symmetric) and
   on its MLP projections (INT4, group 32, F16 scales, zero points). Codes and
   scales come from a deterministic generator (scales of both signs, never
   zero). Build each tensor through `moxie_format`'s affine types; the host
   binding is `reconstruct()` rounded to BF16 per element; the device binding
   is the sections laid out per `WeightFormat::sections` (padding bytes zero).
   Lower with `lower_selected_with_formats` and run the prefill, decode and
   replay steps through `run_prefill_decode` on every GPU; compare logits with
   the existing `assert_logits`. **If the gate fails, stop and send
   `DECISION` with the measured worst error; do not change the gate.**

## Contract before implementation

- **Semantics:** document 03's `W = (Q - Z) * S`; the kernel rounds each
  reconstructed weight to BF16 inside its tile, accumulates in FP32 and rounds
  the output once. The host reference is the same equation.
- **Resources:** unchanged from task 0080's admission; no workspace.
- **Failure:** invalid scales are refused at binding validation, before any
  upload.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`
(including the new test on all three GPUs), `dense_tp2_device`, `cargo
xtask-cuda test-gpu` (63/63, SM86 and SM120).

**Coverage check (two mutants, each reverted):** (a) pass `scale_kind` 1 for
F16 scales; (b) pass `0` for the zero-point address. The new test must fail
each. If either survives, stop and report.

**Stop conditions:** the numerical gate fails; a change conflicts with the
code; a file outside the allowed list is needed.

## Result, filled after work

Implemented all three changes. Both coverage mutants were caught and
reverted: forcing F16 scales to `scale_kind = 1` and omitting the zero-point
address each failed at 3 BF16 ULP on the first GPU. The unmutated prefill,
decode and replay matched the BF16-rounded host reference at 0.000 BF16 ULP on
all three GPUs.

Host gates passed: formatting, workspace and executor driver-feature clippy,
workspace tests, architecture check and spec check. GPU gates passed: dense
Gemma (8 passed, 1 ignored), dense TP2 (1 passed), and CUDA image suite
(63/63 across SM86 and SM120).
