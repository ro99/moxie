# Task 0105 — qualify the affine and attention kernels at Gemma 31B's shapes

Status: **proposed** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`. Queued after task 0100.

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
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
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

## Result, filled after work
