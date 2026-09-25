# Task 0097 — cuBLAS TP partials and a packed split reference, bit-identical

Status: **accepted** (coordinator, 2026-09-25). Builder Codex `luna`;
reviewer Codex `sol`. Candidate `24e470f`, isolation repair `5be7d06`.
- TP2 on the 3090 pair is byte-identical to the packed single-GPU reference
  on cuBLAS (ADR 0036 holds). All three GPUs are within ADR 0028, 1 ULP at
  the decode shape.
- Reviews: an early review of the working tree while the final suite ran
  (3 medium, 1 low; fixed in the same round), then two delta reviews. The
  second found test isolation: success depended on test order. It was fixed
  with a separate test binary; the third delta review was clean.
- The strided-layout mutant survived: on this driver cuBLAS gives the same
  bits packed or strided at every shape tried. Packing is kept, because
  cuBLAS guarantees reproducibility only for identical call parameters. Asynchronous-ownership work: the escape inventory is in change 5.

## Identity and authority

- Task0097, M6 roadmap **M6.1**, the second half of
  [ADR 0037](../decisions/adr/0037-cublas-tensor-op-bf16-linears.md). Task
  0096 (accepted) put single-GPU BF16 `Linear` on cuBLAS. This task moves
  `LinearPartial` (the TP rank partial) and `LinearSplit` (the single-GPU
  reference for TP) to cuBLAS. ADR 0036 still holds: TP2 must stay
  **bit-identical** to the single-GPU reference.
- Clause 3 of ADR 0037, as amended after sol's design review of 0096:
  bit-identity needs **identical cuBLAS call parameters**. A TP rank
  multiplies a compact `[out, bw]` weight half by a compact `[rows, bw]`
  input half (`lda = ldb = bw`). The reference must therefore pack each block
  into the same compact layout and make the same call.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. On a conflict with the code,
  stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  `.gitignore`, `docs/evidence/specification-version.md` and ADRs 0034 and
  0035. **Stage explicit paths only.** GPUs are free; always
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in identifiers, labels or strings.

## Facts established before writing (coordinator, 2026-09-25)

- **TP lowering** (`moxie-plan/src/tensor_parallel.rs` about 1160–1230): a
  row-parallel linear on rank `r` becomes `Linear { in_features: local,
  out_features }`, with its input slice `first = r·local, width = local`,
  and `LinearReductionOrder { blocks: ranks, slice }`. The rank's weight and
  input are compact.
- **0096's work** (`dense.rs`, `chain.rs`, `moxie-cuda/src/blas.rs`,
  `moxie-kernels` `dense_graph_catalogue_unordered`):
  - `Blas::bind` runs per step, and runs before capture;
  - `gemm_bf16` produces BF16 output;
  - the node-to-module symbol map, and `cublas:` descriptor validation;
  - an aligned BLAS workspace placed after the RoPE tables;
  - explicit teardown and quarantine rules.
- `TP_REDUCE_F32` (`moxie_tp_reduce_f32_v1`) adds two FP32 partials in order
  and rounds once to BF16. The TP rank-group reduce and the ordered
  `LinearSplit` kernel both implement this combine.
- `moxie-cuda` has no 2D strided copy (`cuMemcpy2DAsync`) yet
  (`driver.rs` has 1D async copies only).
- `dense_tp2_device.rs` already compares TP2 with a single-GPU reference
  plan built with `lower_selected_ordered(…, &lowering.linear_orders, …)`,
  using `moxie_kernels::dense_graph_catalogue()`.

## Bounded deliverable

- **Outcome:** with the unordered catalogue:
  - a TP rank's `LinearPartial` runs one `cublasGemmEx` with FP32 output;
  - the single-GPU `LinearSplit` (2 blocks) packs each block contiguously,
    runs the same call into an FP32 partial, and combines the two partials
    with `TP_REDUCE_F32`.

  On the 3090 pair, TP2 output is byte-identical to the single-GPU
  reference.
- **Allowed files:**
  - `crates/moxie-cuda/src/{ffi.rs,driver.rs,blas.rs,lib.rs}` (`lib.rs`
    re-export only; amended after sol's early review);
  - `crates/moxie-plan/src/lib.rs` (re-export of the workspace type only;
    amended);
  - `crates/moxie-kernels/src/lib.rs`;
  - `crates/moxie-plan/src/selected.rs`;
  - `crates/moxie-executor/src/{dense.rs,chain.rs}`;
  - a new `crates/moxie-executor/tests/dense_tp2_cublas_device.rs` for
    test (b), in its own process, because `dense_tp2_device`'s terminal
    stall test leaves the pair claimed (sol, review round 2). Revert any
    rename or change in `dense_tp2_device.rs`; move shared helpers only
    by copying the minimum needed;
  - `crates/moxie-executor/tests/dense_gemma_device.rs` (extend 0096's gate
    test to the two new operations);
  - this task's Result.
- **Non-goals:**
  - more than 2 blocks (refuse);
  - `Route`, `VocabProjection`;
  - admission-time weight repacking (per-step packing is the reference's
    cost: it exists to be compared against, and production single-GPU plans
    use `Linear`);
  - timing.

## Numbered changes

1. **Output type (`blas.rs`).** Give `gemm_bf16` a `c_f32: bool`. When set,
   C is `CUDA_R_32F` with `ldc` in FP32 elements. Nothing else about the
   call changes: math mode (with
   `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`), compute type and
   algorithm stay as they are.
2. **2D copy (`ffi.rs`, `driver.rs`).** Add `cuMemcpy2DAsync_v2` and
   `pub unsafe fn copy_2d_async(ctx, dst: u64, dst_pitch, src: u64,
   src_pitch, width_bytes, height, stream)`, a device-to-device copy on a
   stream. It is capturable.
3. **Descriptors (`moxie-kernels`).** In `dense_graph_catalogue_unordered(sm)`,
   also replace the ordered `LinearPartial` and `LinearSplit` descriptors:
   - ids `bf16-linear-partial-cublas-v1-sm_XX` and
     `bf16-linear-split-cublas-v1-sm_XX`;
   - accumulation `Bf16InF32AccUnordered`, ABI 1, image identity
     `cublas_sha256()`, and the same bounds as 0096;
   - symbols: partial `["cublas:gemm_ex_partial"]`; split
     `["cublas:gemm_ex_split", TP_REDUCE_F32]`. The reduce is a real module
     symbol, resolved through 0096's node-to-module map (sol's H3 on 0096).
4. **Selection (`selected.rs`).** Extend 0096's dense-only admission so that
   `Bf16InF32Acc` admits `Bf16InF32AccUnordered` for `LinearPartial` and
   `LinearSplit` too. Chain and paged matching stay exact.
5. **Execution and workspace (`dense.rs`, the workspace layout).**
   - **`LinearPartial`:** `gemm_bf16(m=out, n=rows, k=local, a=weight,
     lda=local, b=input, ldb=local, c=output, ldc=out, c_f32=true)`.
   - **`LinearSplit` with `blocks = 2` and `bw = in/2`:** for `b` in 0, 1,
     in stream order:
     1. `copy_2d_async` weight columns `[b·bw, (b+1)·bw)`. The source is
        the weight's address plus `b·bw·2`, with pitch `in·2`. The
        destination is the packed-weight slot, with pitch `bw·2`. Width is
        `bw·2` bytes; height is `out`.
     2. `copy_2d_async` the input block the same way into the packed-input
        slot (height `rows`).
     3. `gemm_bf16(out, rows, bw, packed_w, bw, packed_x, bw, partial_b,
        out, c_f32=true)`.

     Then launch `TP_REDUCE_F32(partial_0, partial_1 → output)` with the
     launch shape the TP path uses. The packed slots are reused for block 1
     after block 0's GEMM in stream order; no extra synchronization is
     needed on one stream. Blocks other than 2 are refused (`invalid`).
   - **Workspace:** after 0096's BLAS range, add 256-byte-aligned slots,
     each sized by the maximum over the candidate's split nodes, with
     checked arithmetic:
     - packed weight, `out·bw·2`;
     - packed input, `rows·bw·2`;
     - two FP32 partials, `rows·out·4` each.

     Include their end in `workspace_region_bytes`, the arena ranges and the
     ledger request.
   - **Escape inventory:** the slots are ranges inside the plan's arena. No
     new owner and no new handle. They live and die with the plan's
     workspace under 0096's teardown and quarantine rules, which cover every
     exit path (sol's H2 on 0096). A graph captured over a split node holds
     the slot addresses, and it is destroyed before the arena, per 0096's
     order.
6. **Tests.**
   - (a) Extend 0096's `cublas_linear_holds_the_quantized_gate`:
     - `LinearPartial` against the host FP32 partial oracle, then the
       two-clause rule after the ordered combine and rounding;
     - `LinearSplit` (2 blocks) against the ordered host oracle, same rule;
     - same three shapes, all three GPUs.
   - (b) `tp2_unordered_matches_the_packed_split_reference` in
     `dense_tp2_device.rs`, on the 3090 pair:
     - use `dense_graph_catalogue_unordered` for both the TP2 ranks and the
       single-GPU `lower_selected_ordered` reference (with
       `lowering.linear_orders`), on the fixture the existing
       TP-versus-single comparison uses;
     - assert the output **bytes** are equal, **eager only**. Coordinator,
       answering luna's DECISION: `set_segment_capture` refuses candidates
       with linear orders (`chain.rs` about 524–537). Capturing ordered-
       reduction plans (TP ranks, the split reference) moves to the M6.2
       full-step capture task.
7. **Coverage check (one mutant, reverted after; run only test (b) with
   `--exact`).** In `LinearSplit`, call the GEMM on the strided full-width
   view (`lda = ldb = in`, no packing). Test (b) must fail, or the Result
   records that this driver's cuBLAS gives the same bits for both layouts
   at these shapes. In that case, name the shape that would distinguish
   them.

## Acceptance

Host gates:
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- executor driver clippy, with and without `cublas`;
- `cargo test --workspace --locked`;
- `cargo xtask arch-check`;
- `cargo xtask spec-check`.

GPU gates, once, with `cublas` (the feature is additive):
- the full `dense_gemma_device`;
- the full `dense_tp2_device`.

No `test-gpu` (no kernel source changes), and no timing.

**Stop conditions:**
- test (b) fails with packed, identical calls. The cause is then inside
  cuBLAS (different algorithms per device or per call), and ADR 0036 cannot
  hold on this path: stop and report, do not loosen the test;
- a file outside the allowed list is needed.

## Result, filled after work

Implemented cuBLAS FP32 `LinearPartial` and the packed two-block `LinearSplit`
reference, admitted their scratch only for the `cublas:gemm_ex_split`
descriptor, and added the eager TP2 byte comparison.

R1 repairs: split scratch sizing now ignores ordered and non-cuBLAS plans;
the device test checks each raw FP32 partial against its host FP32 oracle
under ADR 0028's two clauses before checking the combined BF16 result; the
two amended public re-exports remain. Removed the permanent strided control
and its output bookkeeping, leaving compact and packed paths.

| Shape `(rows, input, output)` | GPU UUID | Partial/split combined BF16 result: worst ULP / second-clause elements | Linear BF16 result: worst ULP / second-clause elements |
|---|---|---:|---:|
| `(1, 5376, 21504)` | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 1 / 0 | 1 / 0 |
| `(1, 5376, 21504)` | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 1 / 0 | 1 / 0 |
| `(33, 1024, 3072)` | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 5 / 5 | 10 / 1 |
| `(33, 1024, 3072)` | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 4 / 3 | 4 / 1 |
| `(512, 4096, 4096)` | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 80 / 88 | 27976 / 173 |
| `(512, 4096, 4096)` | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 70 / 92 | 28051 / 171 |

The strided-layout mutant survived the fixture comparison: its output bytes
matched the packed reference. The gate shapes also matched. Additional
3090 probes at `(1, 8194, 8192)`, `(1, 4098, 16384)`, `(1, 16386, 4096)`,
`(33, 1026, 8192)` and `(127, 258, 8192)` showed zero differing BF16 outputs
and zero differing FP32 partial bits. No distinguishing shape was found on
this cuBLAS driver; `(127, 258, 8192)` was the largest probe by output count.

Host gates passed: fmt, workspace clippy, executor driver clippy with and
without `cublas`, workspace tests, arch-check (79 rejected / 21 accepted
fixtures, 13 rules), and spec-check (10 documents). The full
`dense_gemma_device` suite passed before R1 and was not rerun; after R1,
`cublas_linear_holds_the_quantized_gate --exact` passed on all three GPUs.
R2 isolation: moved `tp2_unordered_matches_the_packed_split_reference` and
its required helpers into the separate `dense_tp2_cublas_device` integration
test binary, and restored `dense_tp2_device.rs` to its pre-task contents.
The new comparison passed (1/1), and the full `dense_tp2_device` suite passed
(1/1) with Cargo's default test-thread setting. R2 host gates passed again:
fmt, workspace clippy, executor driver clippy with and without `cublas`,
workspace tests, arch-check (79 rejected / 21 accepted fixtures, 13 rules),
and spec-check (10 documents).
