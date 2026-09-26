# Task 0105 — qualify the affine and attention kernels at Gemma 31B's shapes

Status: **accepted** (coordinator, 2026-09-26). Hash capture `debf248`,
implementation `0777d84`, R2 record/comment repair `da308e1`. Built by
Claude Sonnet `builder`; reviewed by Claude Opus `reviewer`: design review
(14 findings, all applied), R1 (0 high, 0 medium, 3 low), R2 delta clean.
- Affine INT8/INT4 linears are qualified at input 21,504 under ADR 0028.
- Paged attention (direct, indirect, partial) is qualified at head_dim 512,
  and the pinned bytes for head dims 256, 200, 128 and 64 are unchanged.
  `test-gpu` 78/78.
- R1 L3 (a test-harness refactor extracting shared launch helpers, about
  −60 lines) was not taken: it forces a full kernel-suite rerun for test
  code only.

## Identity and authority

- Task0105, **M6 exit**, route item 3. `gemma-4-31B-it-AWQ-8bit` needs two
  shapes that no kernel is qualified for:
  - **Affine INT8 linear:** `down_proj` input 21,504, and `gate_proj` /
    `up_proj` output 21,504. The affine descriptors declare `max_input` and
    `max_output` 16,384 (`moxie-kernels/src/lib.rs` about 325–340, 380,
    430, 570).
  - **Paged attention:** global layers have `global_head_dim` 512. The
    kernel's `MOXIE_ATTN_MAX_HEAD_DIM` is 256 (`paged_attention.cu` about
    59–60), mirrored by `PAGED_ATTENTION_MAX_HEAD_DIM` (`lib.rs` about 110)
    in the descriptor and `paged_attention_declares`.
- Both are **declared** bounds, not structural ones. The affine kernel loops
  over 16×16 WMMA tiles along the input axis. The attention cap sizes
  `q_sh` and the per-thread accumulator slots (`MOXIE_ATTN_ACC_SLOTS =
  MAX_HEAD_DIM / THREADS`). Each output dimension's sum is independent of
  the slot count, so smaller heads keep their order of operations.
- Builder: Claude Sonnet `builder` (`/ponytail:ponytail`). Reviewer: Claude
  Opus `reviewer` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**; on a conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  files. **Stage explicit paths only.** GPUs free;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Naming rule applies.

## Bounded deliverable

- **Outcome:**
  - the affine descriptors declare `max_input` and `max_output` 21,504
    (`max_rows` unchanged), qualified under ADR 0028 at Gemma 31B's
    shapes;
  - paged attention (the direct, partial and step-indirect entry points)
    admits `head_dim` ≤ 512, is qualified against the host oracle at 512,
    and still reproduces task 0098's pinned bytes for smaller heads.
- **Allowed files:**
  - `crates/moxie-kernels/{cuda/paged_attention.cu,cuda/affine_linear.cu,src/lib.rs}`;
  - `xtask/src/gpu.rs` (new `test-gpu` cases);
  - `crates/moxie-executor/tests/affine_linear_device.rs` (shapes only);
  - this task's Result.
- **Non-goals:** any change to the arithmetic order of either kernel;
  performance tuning; timing.

## Numbered changes

1. **Affine bounds.**
   - Raise `max_input` and `max_output` to 21,504 in every affine
     descriptor that declares 16,384.
   - Audit the kernel's index arithmetic for 32-bit overflow at `out ×
     in = 21,504 × 5,376` and at `rows × max_output`. Widen to 64-bit where
     needed, and record each site in the Result.
2. **Affine qualification.** Extend `affine_linear_device.rs`'s ADR 0028
   gate test with INT8 group-32 symmetric shapes `(rows, in, out)` ∈
   `{(1, 21504, 5376), (33, 21504, 5376), (1, 5376, 21504), (8, 5376,
   21504)}`, on all three GPUs. Report the worst ULP and the second-clause
   count per shape, as task 0096 did.
3. **Attention bound.** Set `MOXIE_ATTN_MAX_HEAD_DIM` to 512 and
   `PAGED_ATTENTION_MAX_HEAD_DIM` to 512. Update `paged_attention_declares`
   and the catalogue accordingly. Keep every loop over head dimensions
   bounded by the runtime `head_dim` with the slot guard as today. Check
   the shared-memory and register use at 512 on SM86 and SM120 (`nvcc
   --resource-usage` output in the Result), and that the launch still fits
   the 128-thread block.
4. **Attention qualification (`xtask/src/gpu.rs`).** A new case,
   `paged_attention_head_dim_512`: GQA with 32 query heads and 4 KV heads,
   head_dim 512, partial rotary irrelevant (attention only). Cover:
   - a decode row, an 8-row chunk and a sliding window;
   - the direct, step-indirect and partial-plus-merge paths;
   - comparison against the host oracle with the tolerance the existing
     `paged_attention` case uses.

   The existing `paged_attention_indirect` case's pinned pre-refactor
   hashes (head dims 128 and 64) must still match. That is the evidence
   that smaller heads are unchanged.
5. **Coverage check (one mutant, reverted after; run only the new case).**
   Leave `MOXIE_ATTN_ACC_SLOTS` at its old value of 2 while raising the
   admitted head_dim. The new case must fail.

## Acceptance

Host gates:
- fmt;
- workspace clippy;
- xtask CUDA-feature clippy;
- `cargo test --workspace --locked`;
- arch-check;
- spec-check.

GPU gates:
- `cargo xtask-cuda test-gpu`, all cases on SM86 and SM120 (kernel source
  changes);
- the full `dense_gemma_device` once with `cublas`, because the descriptor
  bounds change what selection admits.

No timing.

**Stop conditions:**
- the pinned smaller-head hashes change;
- head_dim 512 does not fit the block's shared memory or registers without
  restructuring the kernel;
- a file outside the allowed list is needed.

## Design review — adopted in full (read the record for exact text)

The exact replacement text for each item is in
[task-0105-design-review.md](../evidence/task-0105-design-review.md). A
summary, in the order to apply them:

- **H1:** raise `max_input` to 21,504 **only** in `affine_linear_catalogue`
  (`lib.rs` about 438) and `dense_graph_catalogue`'s `dense-affine-linear-*`
  (about 575). **Never** in `expert_mlp_catalogue` (about 337 and 387) or
  any other catalogue.
- **M1:** `max_output` stays 65,536 and `max_rows` stays 65,536.
- **M2:** no kernel edit to `affine_linear.cu`. It is removed from the
  allowed files. Copy the 64-bit index audit list into the Result. New stop
  condition: the audit finds a 32-bit index.
- **H3:** new change 3a, **before any kernel edit**. Add the `gqa-256-decode`
  and `gqa-200-chunk` cases to `paged_attention_indirect`, capture their v1
  SHA-256s on the unchanged kernel (SM86 and SM120 must agree), commit
  them, and only then do change 3. New stop condition: a captured hash
  differs between SM86 and SM120. The partial kernel is covered by the
  oracle only; say so in the Result.
- **H2:** rename `mha-256-widest-declared` to `mha-256-two-full-slots`, with
  a literal 256. Add `mha-512-widest-declared`.
- **L2:** `paged_attention_declares`, the catalogue, the host guard and the
  refusal tests already read `PAGED_ATTENTION_MAX_HEAD_DIM`. Do not edit
  them; confirm by grep.
- **H4:** in `affine_linear_device.rs` `run_case`, raise both
  `CapacitySnapshot` totals to `256 << 20`, and add the new rows to
  `CASES`. No other edit to that file.
- **L1:** the fixture is `Grouping::Contiguous { size: 32 }`, `asymmetric:
  false`, `ScaleDtype::Bf16`, `mapped: false`.
- **M3:** add the INT4 row `(1, 21504, 5376)`.
- **M4:** register `paged_attention_head_dim_512` in both the case-name list
  and the dispatch. The mutant is `#define MOXIE_ATTN_ACC_SLOTS 2`, run with
  `cargo xtask-cuda test-gpu --profile sm86`. The verdict must be FAILED.
  Revert, and confirm an empty `git diff`.
- **L3:** run `cuobjdump -res-usage` on `paged_attention.fatbin`, and quote
  REG and SHARED for the three attention symbols.
- **H5:** a new GPU gate, `cargo test -p moxie-executor --features driver
  --test affine_linear_device --locked -- --nocapture`. Quote each new
  case's worst ULP and cancellation count.
- **L4:** new stop condition: a 21,504 case fails ADR 0028's gate. Report the
  element, its terms and its ULP; the gate is the owner's.
- **L5:** the Identity is updated to the Sonnet builder and the Opus
  reviewer.

## Result, filled after work

Status: **implemented** (builder, Claude Sonnet, 2026-09-26). Applied the
design review in full, in its stated order. Root
`/home/rodrigo/Developer/moxie`, host and GPU work on all three visible
GPUs (`GPU-97fe4889…` SM120 5060 Ti, `GPU-3032cfa3…` and `GPU-81fe4578…`
SM86 3090s).

### 3a (H3) — pinned before any kernel edit, committed separately (`debf248`)

Added `gqa-256-decode` (kv_heads 2, head_dim 256, two full accumulator
slots) and `gqa-200-chunk` (kv_heads 2, head_dim 200, a partly-used second
slot) to `paged_attention_indirect`. Captured on the unchanged kernel, all
three GPUs agreed:
- `gqa-256-decode`: `50a65b949de6c90123077955814097e03581a1a99d35db6ca5d3d0b6e8ef559e`
- `gqa-200-chunk`: `c1ac5bf245a6753c2833b2ff703f8edd974c3a17de2762d709d04778c3667673`

The partial kernel has no pinned hash; it is covered only by the FP64
oracle (see change 4 below).

### Change 3 — attention bound

`MOXIE_ATTN_MAX_HEAD_DIM` (`paged_attention.cu`) and
`PAGED_ATTENTION_MAX_HEAD_DIM` (`kernels/src/lib.rs`) raised 256 → 512.
Per L2, no other edit: `paged_attention_declares`, the paged-attention
catalogue descriptor, the host admission guard
(`moxie-executor/src/paged_attention.rs:146`) and the refusal tests
(`paged_attention.rs:995`, `paged_attention_device.rs`'s narrowed-descriptor
test) all already read the constant -- confirmed by grep, no hit outside
those. Every accumulator-slot loop stays guarded by the runtime `head_dim`
(`if (d >= head_dim) continue;`), so widening the constant only grows how
many *unused, guarded* slots a thread can own at a smaller runtime
`head_dim`; the order of operations for an existing shape is unchanged.

### Change 4 (H2, M4) — attention qualification

- `mha-256-widest-declared` renamed `mha-256-two-full-slots`, with a
  literal `head_dim: 256` (no longer tracks the constant).
- New sibling `mha-512-widest-declared`, at `PAGED_ATTENTION_MAX_HEAD_DIM`.
- New case `paged_attention_head_dim_512`, registered in both the
  case-name list and the dispatch (after `paged_attention_indirect`): GQA
  32 query heads over 4 KV heads, head_dim 512. Three shapes -- `decode`
  (rows 1), `chunk-8` (rows 8), `sliding-window` (window 40, rows 5) --
  each checked two ways: the direct kernel's raw output against the FP64
  oracle (`check_attention`, the same bound `paged_attention` uses), and
  the step-indirect kernel's raw output checked byte-identical against
  that same direct output (the same cross-check `paged_attention_indirect`
  makes at smaller heads). The two-block partial-plus-merge path
  (`Staging::TwoBlock`/`attend_two_block`) is checked for the `decode`
  shape only -- `attend_two_block` refuses `rows() != 1`, so decode is the
  only shape the production path actually serves, the same shape
  `paged_attention_host_streaming` already qualifies it at.

`cargo xtask-cuda test-gpu` (all cases, all three GPUs): **78 passed, 0
failed**; sm_86 and sm_120 both QUALIFIED. Printed summary lines:
```
decode direct=oracle-checked(max=4.877e-4) indirect=byte-identical
chunk-8 direct=oracle-checked(max=5.664e-4) indirect=byte-identical
sliding-window direct=oracle-checked(max=9.757e-4) indirect=byte-identical
decode-merge transfer=262148 B max=6.746e-8
```
(SM86 values shown; SM120 was within the same order of magnitude on every
line.)

### Change 5 (M4) — coverage check, performed and reverted

Substituted the literal `#define MOXIE_ATTN_ACC_SLOTS 2` for the derived
`#define MOXIE_ATTN_ACC_SLOTS (MOXIE_ATTN_MAX_HEAD_DIM / MOXIE_ATTN_THREADS)`.
`cargo xtask-cuda test-gpu --profile sm86` (which also exercised the
present SM120 device): both `paged_attention`'s `mha-512-widest-declared`
and `paged_attention_head_dim_512`'s `decode` shape **FAILED** on all
three GPUs (`device 0 against oracle ...`, off by whole tenths, far past
the declared bound) -- the mutant is caught. Reverted; `git diff` on
`paged_attention.cu` afterward showed only change 3's legitimate edit, the
`MOXIE_ATTN_ACC_SLOTS` line itself unchanged (context, not a diff line).
Full suite rerun clean: 78 passed, 0 failed, both architectures QUALIFIED.

### Change 1 (H1, M1, M2) — affine bounds

`max_input` raised 16,384 → 21,504 in exactly two descriptors:
`affine_linear_catalogue` (`lib.rs`, INT4/INT8 × SM86/SM120) and
`dense_graph_catalogue`'s `dense-affine-linear-*`. `max_output` (65,536)
and `max_rows` (65,536) untouched. `expert_mlp_catalogue`'s two descriptors
(BF16 and affine expert MLP) are untouched -- Gemma 31B is dense, and
nothing here qualifies a routed shape at this depth.

No kernel edit (M2). Audit, copied here as required:
- `affine_linear.cu`: every index touching `in_features`/`out_features` is
  `unsigned long long` -- `n0`/`m0` (~59-61), the `k0`/`k`/`m`/`n`/`group`/
  `entry` family through the tile loop (~78-122), and the output write
  `m * out_features + n` (~151-155).
- `affine_decode.cuh` (38 lines total): `moxie_affine_scale_v1`'s `entry`
  parameter and `moxie_affine_code_v1`'s `row_base` parameter are both
  `unsigned long long`, and `entry * 4`/`entry * 2`/`row_base + k`/
  `row_base + (k >> 1)` all operate on that type.
- Host grid: `affine_linear.rs:1628-1635` checks both grid dimensions fit
  `u32` before launch; at `max_rows`/`max_output` 65,536 and a 16-wide
  tile, grid.y and grid.x are both ≤ 4,096, under CUDA's 65,535 limit.

No 32-bit index was found; the stop condition did not fire.

### Change 2 (H4, L1, M3) — affine qualification

`affine_linear_device.rs` `run_case`: both `CapacitySnapshot` totals
raised `64 << 20` → `256 << 20` (the 21,504×5,376 INT8 codes alone are
115,605,504 B). No other edit to that file, per H4.

Five new `Case`s added to `CASES`, all `Grouping::Contiguous { size: 32 }`,
`mapped: false` (L1):
- `g`/`h`: INT8 symmetric, BF16 scales, `(out, in) = (5,376, 21,504)`
  (`down_proj`'s shape), rows 1 and 33.
- `i`/`j`: INT8 symmetric, BF16 scales, `(out, in) = (21,504, 5,376)`
  (`gate_proj`/`up_proj`'s shape), rows 1 and 8.
- `k` (M3): INT4 symmetric, BF16 scales, `(out, in) = (5,376, 21,504)`,
  row 1 -- qualifies the INT4 code path (`k >> 1` in `affine_decode.cuh`)
  at the same depth.

`cargo test -p moxie-executor --features driver --test affine_linear_device
--locked -- --nocapture` (H5): **4 passed, 0 failed**, 572.15 s, on all
three visible devices. Per-case worst ULP and cancellation-clause count
(SM86 3090; SM120 was the same order of magnitude):

| Case | elements | worst ULP | 2nd-clause count |
|---|---|---|---|
| g (INT8, down-shape, row 1) | 5,376 | 9.0 | 4 |
| h (INT8, down-shape, 33 rows) | 177,408 | 15,288.0 | 109 |
| i (INT8, gate-shape, row 1) | 21,504 | 7.0 | 1 |
| j (INT8, gate-shape, 8 rows) | 172,032 | 140.5 | 23 |
| k (INT4, down-shape, row 1) | 5,376 | 7.0 | 2 |

Every element that missed ADR 0028's 2-ULP clause passed the reduction's
own `2^-8 · Σ|x·W|` clause instead (L4's stop condition never fired); `h`,
the deepest reduction (21,504) at the most rows (33), has the largest
misses and the most clause-2 elements, matching the clause's own
rationale. No timing is claimed -- device/BF16-copy byte counts printed
alongside are a memory-footprint assertion (`footprint < dequantized`),
not a speed measurement.

### L3 — resource usage

`cuobjdump -res-usage` on the built `paged_attention.fatbin`:

| Symbol | SM86 REG / SHARED | SM120 REG / SHARED |
|---|---|---|
| `moxie_bf16_paged_attention_v1` (direct) | 40 / 3,608 B | 40 / 4,632 B |
| `moxie_bf16_paged_attention_indirect_v1` | 40 / 3,640 B | 42 / 4,664 B |
| `moxie_bf16_paged_attention_partial_v1` | 42 / 3,608 B | 42 / 4,632 B |

Matches the design review's estimate (3,608 B on SM86) exactly. Well
under any per-block shared-memory or register limit at a 128-thread
block; the 512 stop condition ("does not fit... without restructuring")
did not fire.

### GPU acceptance gate: `dense_gemma_device` with `cublas`

`cargo test -p moxie-executor --features
driver,paged-attention-binding,paged-attention-test-hooks,nccl,cublas
--test dense_gemma_device --locked -- --nocapture`: **18 passed, 0
failed, 2 ignored** (`dense_step_timing`, `stress_graph_benchmark` --
timing/benchmark harnesses, correctly not run here), 188.5 s. The
descriptor-bound change does not change what selection admits for this
suite's shapes.

### Host gates

fmt clean; `cargo clippy --workspace --all-targets --locked -- -D
warnings` clean; `cargo clippy -p xtask --features cuda --all-targets
--locked -- -D warnings` clean; `cargo clippy -p moxie-executor
--all-targets --features
driver,paged-attention-binding,paged-attention-test-hooks,nccl,cublas
--locked -- -D warnings` clean (the executor driver lane, since this task
touches an `#![cfg(feature = "driver")]` test file); `cargo test
--workspace --locked`: **PASS (0 failed)**, rerun to completion on
`0777d84` after this Result was first drafted; `cargo xtask arch-check`
passed (79 rejected / 21 accepted fixtures, 13 rules -- no allowlist change, matching
the Allowed files list); `cargo xtask spec-check` passed (10 documents
unchanged).

### Not claimed

No arithmetic order changed in either kernel (task's own non-goal); no
performance tuning or timing; nothing about a routed (expert-MLP) shape at
21,504 -- `expert_mlp_catalogue` is untouched and Gemma 31B is dense.
