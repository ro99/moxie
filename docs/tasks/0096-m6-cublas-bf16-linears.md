# Task 0096 — BF16 dense linears on cuBLAS tensor-op

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`. Asynchronous-ownership work: the escape inventory below is part
of the contract. **Revised after sol's design review** (4 high, 3 medium, all
adopted) before any implementation. The TP partial, the split reference and
ADR 0036 bit-identity moved to task 0097, because they need packed weight
blocks (sol H4).

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

- **Outcome:** a dense plan lowered from the unordered catalogue runs every
  BF16 `Linear` through `cublasGemmEx`, eager and captured. It holds ADR
  0028's rule per element against the ordered host oracle. The ordered path
  is unchanged. `LinearSplit` and `LinearPartial` stay ordered in this task:
  the unordered catalogue does not replace them, so TP plans keep their
  present kernels until task 0097.
- **Allowed files:**
  - `crates/moxie-cuda/{build.rs,Cargo.toml,src/lib.rs,src/ffi.rs}`, plus a
    new `src/blas.rs`;
  - `crates/moxie-types/src/precision.rs`;
  - `crates/moxie-kernels/{build.rs,Cargo.toml,src/lib.rs}`;
  - `crates/moxie-plan/src/selected.rs`;
  - `crates/moxie-executor/src/{dense.rs,chain.rs}`;
  - the executor `Cargo.toml` feature line;
  - `crates/moxie-executor/tests/dense_gemma_device.rs` (new tests);
  - `xtask/src/archcheck.rs`, only if it must name the link;
  - this task's Result.
- **Non-goals:**
  - `LinearSplit`, `LinearPartial`, TP (task 0097);
  - `Route`, `VocabProjection`, attention, the affine kernel, the older
    chain path;
  - changing any default catalogue;
  - speed claims (one timing line allowed, change 8).

## Numbered changes

1. **Policy (`moxie-types`).** Add `AccumulationPolicy::Bf16InF32AccUnordered`
   with the doc comment: "16-bit inputs, FP32 accumulator in hardware order
   (tensor-op); held to ADR 0028 against the ordered oracle (ADR 0037)."
   `accumulator()` returns `F32`.
2. **Selection (`selected.rs`), dense lowering only (sol M2).**
   - In the dense lowering's matching site only, `Bf16InF32Acc` admits
     `Bf16InF32AccUnordered` for `SemanticKernelOp::Linear`.
   - The chain and paged matching sites stay exact equality.
   - Two matches is today's "expected exactly one" refusal.
   - The unordered descriptor's `abi_version` is `DENSE_GRAPH_ABI` (1).
3. **cuBLAS binding (`moxie-cuda`).**
   - Feature `cublas = ["driver"]`. `build.rs`: when it is on, add
     `$CUDA_HOME/lib64` (default `/usr/local/cuda`) to the link search, link
     `dylib=cublas`, and add an rpath. It panics loudly if
     `libcublas.so.13` is absent.
   - `ffi.rs`: `cublasCreate_v2`, `cublasDestroy_v2`, `cublasSetStream_v2`,
     `cublasSetWorkspace_v2`, `cublasSetMathMode`, `cublasGemmEx`, and their
     constants.
   - `blas.rs`, `pub struct Blas<'ctx>` (sol H1):
     - it owns the handle and records the `&'ctx Context`; it is not
       `Clone` and not `Send`;
     - it holds no borrow of a workspace or stream.
     - `pub unsafe fn bind(&mut self, stream: &Stream<'ctx>, workspace: u64,
       bytes: usize) -> Result<()>` calls `cublasSetStream`, then
       `cublasSetWorkspace` (SetStream resets the user workspace). The
       caller guarantees the range is valid device memory of this context.
     - `pub unsafe fn gemm_bf16(&self, m, n, k, a: u64, lda, b: u64, ldb,
       c: u64, ldc) -> Result<()>` uses `transa = T`, `transb = N`, α = 1,
       β = 0, BF16 A, B and C, compute `CUBLAS_COMPUTE_32F`,
       `CUBLAS_GEMM_DEFAULT` and `CUBLAS_DEFAULT_MATH`.
     - Every call makes the context current, as the module wrappers do.
       Status codes map to typed errors.
     - `pub unsafe fn destroy(self) -> Result<()>` is the only destructor.
       `Drop` does **not** call `cublasDestroy`; an undestroyed handle
       leaks (quarantine), like a live module.
   - If cuBLAS fails on Moxie's explicitly created contexts, stop and send
     `DECISION`.
4. **Descriptor (`moxie-kernels`).** Under a `cublas` feature, add
   `pub fn dense_graph_catalogue_unordered(sm) -> KernelCatalogue`. It is
   `dense_graph_catalogue(sm)` with the one BF16 `Linear` descriptor
   replaced:
   - id `bf16-linear-cublas-v1-sm_XX`;
   - accumulation `Bf16InF32AccUnordered`;
   - `abi_version` 1;
   - workspace `Zero`, because the plan places its own BLAS range (change 6);
   - one symbol, `cublas:gemm_ex`;
   - `image_sha256` = SHA-256 of `libcublas.so.13` from `build.rs`;
   - bounds `max_rows 65_536`, `max_input 65_536`, `max_output 262_144`.
5. **Symbols and identity (`dense.rs`, sol H3).**
   - At admission, build an explicit map from each selected node to its
     module symbol index, for real CUDA symbols only.
   - `launch()` looks up the index through that map, never by position in
     the descriptor list.
   - A `cublas:` node resolves nothing in the module. At execution it
     validates the descriptor's operation, its backend symbol and the
     catalogue's image identity.
   - Extend dense execution's known-catalogue digest check (about 144–146)
     to accept `dense_graph_catalogue_unordered(sm)` for the device's SM,
     keeping the candidate, catalogue and UUID identity checks.
6. **Workspace placement (`selected.rs` workspace layout, sol M3).** When a
   candidate holds a `cublas:` node, place one 256-byte-aligned range of
   `BLAS_WORKSPACE_BYTES = 32 MiB` after the last RoPE table. Use checked
   offsets and include its end in `workspace_region_bytes`, the arena ranges
   and the ledger request. The candidate exposes the range's offset.
7. **Ownership and teardown (`chain.rs`/`dense.rs`, sol H1, H2, M1).**
   - **Creation:** the plan creates its `Blas` lazily, before its first
     dense submission. It calls `bind(stream, workspace range)` before
     every step and before `begin_capture`, never inside a capture. A step
     on a different stream rebinds before any launch.
   - **Teardown, explicit, not by field order:**
     1. launched work is observed complete;
     2. captured graphs are destroyed;
     3. `Blas::destroy`;
     4. the module;
     5. the arena and workspace are released.
   - **Exit paths:**

     | Path | Rule |
     |---|---|
     | Pre-launch refusal | Returns the whole plan unchanged. |
     | Post-submission, capture or event-record failure, or failed completion | Returns the held lease, or withholds the plan. Nothing is torn down. |
     | Close refusal | Keeps every surviving resource in the returned plan. |
     | `Drop` of a plan whose work was not observed | Destroys none of Blas, graphs, module or arena. It leaks them (quarantine). |

     State the `Drop`/`ManuallyDrop` behaviour of each such field in code
     comments, and test it (change 8c).
   - **Capture:** a cuBLAS `Linear` inside a captured segment is allowed.
     Graph-pool admission counts graph nodes, and cuBLAS may emit more than
     one kernel per call. Measure the node count and pool bytes of
     `cublasGemmEx` at each test shape once. Charge a conservative per-call
     bound (a named constant with its measurement in a comment), as task
     0086 did for kernels. A capture failure ends capture, and it
     quarantines if work is unobserved.
8. **Tests (`dense_gemma_device.rs`).**
   - (a) `cublas_linear_holds_the_quantized_gate`, on the 3090 and the 5060
     Ti:
     - at `(rows, in, out)` ∈ `{(1, 5376, 21504), (33, 1024, 3072), (512,
       4096, 4096)}`, with fixed-seed random BF16;
     - apply ADR 0028's two clauses against the ordered host oracle;
     - print the worst ULP and how often the second clause fired.
   - (b) Shape A end-to-end with the unordered catalogue, eager and
     captured:
     - assert finite logits and the same greedy top token per row as the
       ordered path;
     - print the worst ULP.
   - (c) Ownership:
     - a plan dropped after submission without `finish` leaks rather than
       destroys: the handle is still valid, checked by a test hook counter
       of `destroy` calls under `paged-attention-test-hooks`, or the
       nearest existing test-hooks feature;
     - a normal close destroys exactly once;
     - the ledger returns to empty after close.
   - (d) One printed line on a 3090, for (a)'s 512×4096×4096: `cublas-linear
     tflops=…`, the median of 20. Recorded in the Result; not a gate.
9. **Coverage check (one mutant, reverted after).** Bind the workspace
   before `cublasSetStream` instead of after; test (b)'s captured case, or
   (a), must fail or report a cuBLAS error. If it does not, report that the
   order is unobservable here and name the test that would catch it.

## Acceptance

Host gates:
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- the executor driver-feature clippy, plus the same with `cublas`;
- `cargo test --workspace --locked`;
- `cargo xtask arch-check`;
- `cargo xtask spec-check`.

GPU gates (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):
- the full `dense_gemma_device`, with and without `cublas`;
- `dense_tp2_device`;
- `cargo xtask-cuda test-gpu`.

**Stop conditions:**
- cuBLAS fails on Moxie's contexts;
- capture of `cublasGemmEx` fails;
- a file outside the allowed list is needed.

## Design review (sol, 2026-09-25, before implementation)

Adopted: H1 (the Blas handle borrows nothing; bind per step before capture),
H2 (explicit teardown order and quarantine on every path), H3 (map from each
node to its module symbol index; `TP_REDUCE_F32` moves with the split to
0097), H4 (single-GPU split and TP partial are different call shapes, so ADR
0036 identity needs packed blocks: task 0097), M1 (bind before capture;
charge measured cuBLAS graph nodes), M2 (unordered admission in dense
lowering only; ABI 1; digest check extended), M3 (aligned, checked
workspace placement).

## Result, filled after work
