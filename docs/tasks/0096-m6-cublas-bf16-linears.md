# Task 0096 — BF16 dense linears on cuBLAS tensor-op

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`. Asynchronous-ownership work: the escape inventory below is part
of the contract.

## Identity and authority

- Task0096, M6 roadmap **M6.1** (owner ruling, 2026-09-25: build every
  roadmap feature). It implements
  [ADR 0037](../decisions/adr/0037-cublas-tensor-op-bf16-linears.md) (owner
  ruling: cuBLAS for ordinary BF16 linears; selection projections keep the
  declared order). Read the ADR first; it is binding.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. On a conflict with the code,
  stop and send `DECISION`; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  `.gitignore`, `docs/evidence/specification-version.md` and ADRs 0034 and
  0035. **Stage explicit paths only.** GPUs are free; always
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in identifiers, labels or strings.

## Facts established before writing (coordinator, 2026-09-25)

- **Linking.** `libcublas.so.13` (13.1) is in `/usr/local/cuda/lib64`
  (`CUDA_HOME=/usr/local/cuda`). Its only CUDA dependency is
  `libcublasLt.so.13`; it needs no separate `libcudart`. `moxie-cuda/build.rs`
  links `libcuda` only under the `driver` feature.
- **Execution.** `dense.rs` about 623–670: `SemanticKernelOp::Linear`
  launches `BF16_LINEAR`, `LinearPartial` launches `DENSE_LINEAR_PARTIAL`
  (FP32 out) and `LinearSplit` launches `DENSE_LINEAR_SPLIT` with `blocks`
  from `candidate().linear_orders()`. Every launch goes through
  `launch(…)`, which checks the symbol against the plan's module. Weight
  layout is row-major `[out, in]`; activations are row-major `[rows, in]`.
- **Selection.** `selected.rs` (about 254–262, 715–720, 1113–1130) matches
  descriptors on operation, operands, output, `accumulation ==
  node.contract.accumulation`, rounding, layout, SM and shape. It requires
  exactly one match. Linear node contracts carry
  `AccumulationPolicy::Bf16InF32Acc` (`moxie-graph`).
- **Catalogue.** `KernelCatalogue::new` refuses indistinguishable
  descriptors (`capability.rs` about 351). `images::…` in
  `moxie-kernels/src/lib.rs` builds the dense-graph catalogue.
- **Capture.** Piecewise capture (task 0085) records segments on the plan
  stream. cuBLAS calls are capturable when the handle has a user workspace
  (`cublasSetWorkspace`) and a stream (`cublasSetStream`).
- `TP_REDUCE_F32` (`moxie_tp_reduce_f32_v1`) adds two FP32 partials and
  rounds once to BF16.

## Bounded deliverable

- **Outcome:** a dense plan built from the unordered catalogue runs every
  BF16 `Linear`, `LinearSplit` (blocks ≤ 2) and `LinearPartial` through
  `cublasGemmEx`, captured or eager. It holds ADR 0028's rule per element
  against the ordered path. TP2 on the 3090 pair stays bit-identical to the
  single-GPU `LinearSplit` on the unordered path. The ordered path is
  unchanged.
- **Allowed files:**
  - `crates/moxie-cuda/{build.rs,Cargo.toml,src/lib.rs,src/ffi.rs}`, plus a
    new `src/blas.rs`;
  - `crates/moxie-types/src/precision.rs` (the new policy);
  - `crates/moxie-kernels/{build.rs,Cargo.toml,src/lib.rs}` (the unordered
    catalogue);
  - `crates/moxie-plan/src/selected.rs`;
  - `crates/moxie-executor/src/{dense.rs,chain.rs}`;
  - the executor `Cargo.toml` feature line;
  - `crates/moxie-executor/tests/dense_gemma_device.rs` (new tests);
  - `crates/moxie-executor/tests/dense_tp2_device.rs` (one new test);
  - `xtask/src/archcheck.rs`, only if it must name the link;
  - this task's Result.
- **Non-goals:**
  - `Route`, `VocabProjection`, attention, the affine kernel;
  - `chain.rs`'s older BF16 chain path (except where the dense plan admits);
  - `LinearSplit` with more than 2 blocks (refuse);
  - changing any default catalogue;
  - speed claims (O6 open; one timing line is allowed, see change 7).

## Numbered changes

1. **Policy (`moxie-types`).** Add `AccumulationPolicy::Bf16InF32AccUnordered`
   with the doc comment: "16-bit inputs, FP32 accumulator in hardware order
   (tensor-op); held to ADR 0028 against the ordered oracle (ADR 0037)."
   `accumulator()` returns `F32`.
2. **Selection (`selected.rs`).** Add one helper `fn admits(required:
   AccumulationPolicy, declared: AccumulationPolicy, operation) -> bool`:
   - equality as today;
   - additionally, `Bf16InF32Acc` admits `Bf16InF32AccUnordered` **only**
     for `Linear`, `LinearSplit` and `LinearPartial`.

   Use it at the three matching sites. Two matches is today's "expected
   exactly one" refusal. `Route` and `VocabProjection` therefore never match
   an unordered descriptor.
3. **cuBLAS binding (`moxie-cuda`).**
   - Feature `cublas = ["driver"]`. `build.rs`: when it is on, add
     `$CUDA_HOME/lib64` (default `/usr/local/cuda`) to the link search, link
     `dylib=cublas`, and add an rpath to that directory. It panics loudly,
     like the driver branch, if `libcublas.so.13` is absent.
   - `ffi.rs`: `cublasCreate_v2`, `cublasDestroy_v2`, `cublasSetStream_v2`,
     `cublasSetWorkspace_v2`, `cublasSetMathMode`, `cublasGemmEx`, and the
     enum constants they need.
   - `blas.rs`: `pub struct Blas<'ctx>`, created for one `Context` with a
     borrowed `DeviceBuffer` workspace of fixed size `BLAS_WORKSPACE_BYTES =
     32 MiB` and bound to one `Stream`. Make the context current on the
     calling thread for every call, as the module wrappers do.
   - `pub unsafe fn gemm_bf16(&self, transa_t: bool, m, n, k, a: u64, lda, b:
     u64, ldb, c: u64, ldc, c_is_f32: bool) -> Result<()>` with α = 1 and
     β = 0, compute `CUBLAS_COMPUTE_32F`, algorithm `CUBLAS_GEMM_DEFAULT`,
     and math mode `CUBLAS_DEFAULT_MATH`.
   - Status codes map to typed errors through the existing status
     machinery.
   - If cuBLAS does not work with Moxie's explicitly created (non-primary)
     contexts, stop and send `DECISION` with the error.
4. **Descriptors (`moxie-kernels`).** Under a `cublas` feature,
   `pub fn dense_graph_catalogue_unordered(sm) -> KernelCatalogue`. It is
   the dense-graph catalogue with the three BF16 linear descriptors replaced
   by unordered ones:
   - ids `bf16-linear-cublas-v1-sm_XX` (and `…-split-…`, `…-partial-…`);
   - accumulation `Bf16InF32AccUnordered`;
   - workspace `Zero`, because the plan charges the fixed BLAS workspace
     itself (change 5);
   - one symbol each: `cublas:gemm_ex`, `cublas:gemm_ex_split` and
     `cublas:gemm_ex_partial`;
   - `image_sha256` = SHA-256 of `libcublas.so.13`, computed in `build.rs`;
   - shape bounds `max_rows 65_536`, `max_input 65_536`, `max_output
     262_144`.
5. **Execution (`dense.rs`, plan admission).**
   - A plan whose selected nodes include a `cublas:` symbol:
     - excludes those symbols from module symbol resolution;
     - adds `BLAS_WORKSPACE_BYTES` plus, for `LinearSplit`, `2 × rows ×
       out_features × 4` FP32 block partials to its admitted workspace
       region;
     - creates one `Blas` bound to the plan stream and that workspace.
   - `Linear`: `gemm_bf16(transa_t=true, m=out, n=rows, k=in, a=weight,
     lda=in, b=input, ldb=in, c=output, ldc=out, c_is_f32=false)`.
   - `LinearPartial`: the same, with FP32 `c`.
   - `LinearSplit` with `blocks = 2` and block width `bw = in/2`:
     1. for `b` in 0, 1, call `gemm_bf16(true, out, rows, bw, weight + b·bw·2,
        in, input + b·bw·2, in, partial_b, out, true)`;
     2. launch `TP_REDUCE_F32` on the two partials to produce the BF16
        output.

     Block count 1 is `Linear`'s call; more than 2 refuses (`invalid`).
   - Record launch-order labels `linear` exactly as today.
6. **Escape inventory (the `Blas` handle and its workspace).** The owner is
   the plan (`SelectedReservedPlan`).
   - The handle is destroyed only in the plan's close, after the plan's
     stream is observed complete. Declare it before the captured graphs and
     the module in field order, so that graphs drop first.
   - On a failed completion or a close refusal, the plan keeps the handle
     and the workspace (quarantine), as it keeps its module today.
   - On drop without close, the handle is not destroyed while work may be
     pending: use the same `ManuallyDrop` rule the module uses.
   - `Blas` is not `Clone`, not `Send`, and never leaves the plan through a
     public type or refusal.
7. **Tests.**
   - (a) `cublas_linear_holds_the_quantized_gate` in
     `dense_gemma_device.rs`, on the 3090 and the 5060 Ti. For each of
     `Linear`, `LinearSplit` (2 blocks) and `LinearPartial`, at `(rows, in,
     out)` ∈ `{(1, 5376, 21504), (33, 1024, 3072), (512, 4096, 4096)}` with
     random BF16 data from a fixed seed, compare against the host ordered
     oracle with ADR 0028's two clauses. Report the worst ULP and how often
     the second clause fired.
   - (b) Shape A end-to-end with the unordered catalogue, eager and
     captured: logits are finite, and the greedy top token per row equals
     the ordered path's. Report the worst ULP; don't assert it.
   - (c) `dense_tp2_device.rs`: on the 3090 pair, TP2 with the unordered
     catalogue is bit-identical to single-GPU `LinearSplit` with the
     unordered catalogue.
   - (d) One printed timing line of (a)'s 512×4096×4096 on a 3090: median
     of 20, as `cublas-linear tflops=…`. It is recorded in the Result, and
     it is not a gate.
8. **Coverage check (one mutant, reverted after).** In `LinearSplit`, pass
   the whole `in` as `k` for block 0 and skip block 1. Test (c) or (a) must
   fail.

## Acceptance

Host gates:
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- the executor driver-feature clippy, plus the same clippy with `cublas`;
- `cargo test --workspace --locked`;
- `cargo xtask arch-check`;
- `cargo xtask spec-check`.

GPU gates (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):
- the full `dense_gemma_device` and `dense_tp2_device`, with and without
  the `cublas` feature;
- `cargo xtask-cuda test-gpu`.

**Stop conditions:**
- cuBLAS fails on Moxie's contexts;
- capture of a cuBLAS call fails;
- a `LinearSplit` reference and a TP rank select different cuBLAS results
  (the bit-identity test fails and the cause is inside cuBLAS);
- a file outside the allowed list is needed.

## Result, filled after work
