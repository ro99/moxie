# Task 0105 — qualify the affine and attention kernels at Gemma 31B's shapes

Status: **active, revision 2** (coordinator, 2026-09-26). The Opus
reviewer's design review (5 high, 4 medium, 5 low;
[record](../evidence/task-0105-design-review.md)) is **adopted in full and
overrides the numbered changes, allowed files, gates and stop conditions
below wherever they differ**. Builder Claude Sonnet `builder`; reviewer
Claude Opus `reviewer`.

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
