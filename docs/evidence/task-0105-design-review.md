# Task 0105 design review (reviewer, read-only, /ponytail:ponytail-review)

Contract: docs/tasks/0105-m6-kernel-bounds-for-gemma-31b.md at f2cc331.
Findings: 5 HIGH, 4 MEDIUM, 5 LOW.

Verified facts first, so the builder doesn't have to re-derive them:
- **Affine kernel: 16,384 is a declared bound only, and no kernel edit is needed.**
  Every index in `affine_linear.cu` is `unsigned long long`: n0/m0 (58-61), the
  k loop (79), `m*in_features+k` (86), `n*groups_per_row+group` (99, 115),
  `n*row_stride` (122), `m*out_features+n` (155). The same holds in
  `affine_decode.cuh`: `entry*4` / `entry*2` (171, 174) and `row_base+k` (190, 192).
  The host grid is `u32`-checked (`affine_linear.rs:1628-1635`). Grid y is
  rows/16 ≤ 4,096 at max_rows 65,536, under 65,535. The only bound is the
  `descriptor_mismatch` Domain check (`affine_linear.rs:1040-1044`).
- **Attention at 512 fits without restructuring.** Shared memory per block:
  q_sh 2,048 B + score_sh 512 + base_sh 1,024 + reduce_sh 16 + tile_sh 8 =
  3,608 B. ACC_SLOTS becomes 4 register floats. Each output component's sum
  order over t (269-274, 593-598) and the lane-strided dot (188-193) do not
  depend on ACC_SLOTS, so the order argument in the contract holds.
- Checkpoint (`/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit`): head_dim 256
  with 16 KV heads (sliding), global_head_dim 512 with 4 KV heads, 32 query
  heads. `down_proj` scale is **BF16 [5376, 672]**, INT8 group 32 symmetric.
  Global q_proj is out 16,384 and o_proj is in 16,384, so only the MLP needs
  21,504.

## HIGH

**H1 — the contract's line references pull the ExpertMlp descriptors into scope.**
`lib.rs:337-339` (BF16 expert MLP) and `lib.rs:387-389` (affine-expert) are
the descriptors that declare 16,384/16,384. They serve `ExpertMlp` through
`expert_mlp.cu`, and `moxie-plan/src/expert.rs:1906` admits against them. The
change says "every affine descriptor that declares 16,384" and cites "about
325–340, 380". A literal builder would raise the expert bounds, and nothing in
this task qualifies them. That is a declared domain wider than the qualified
one, which is the document 07 gap. Gemma 31B is dense.
*Contract change:* replace change 1's first bullet and the Identity line
reference with:
> Raise `max_input` from 16,384 to 21,504 in exactly these descriptors:
> `affine_linear_catalogue` (`lib.rs` ~438, INT4/INT8 × SM86/SM120) and
> `dense_graph_catalogue`'s `dense-affine-linear-*` (`lib.rs` ~575). Leave
> `max_output` at 65,536 and `max_rows` at 65,536. Do not touch
> `expert_mlp_catalogue` (`lib.rs` ~337 and ~387) or any other catalogue.

**H2 — raising the constant silently turns the only 256 case into a 512 case.**
`xtask/src/gpu.rs:3286-3302`, `mha-256-widest-declared`, sets
`head_dim: moxie_kernels::PAGED_ATTENTION_MAX_HEAD_DIM`. At 512 it stops
testing 256, which is Gemma's sliding width and the only two-full-slot case.
Its label then lies.
*Contract change:* add to change 4:
> In `paged_attention`, change `mha-256-widest-declared` to the literal
> `head_dim: 256` and `reciprocal_sqrt_scale(256)`, and rename it
> `mha-256-two-full-slots`. Add a sibling case `mha-512-widest-declared`
> that uses `PAGED_ATTENTION_MAX_HEAD_DIM`, with the same other fields.

**H3 — the pinned hashes do not cover any multi-slot head dim, so "smaller heads
unchanged" has no evidence at 256.**
`paged_attention_indirect`'s `BASELINE_SHA256` (`gpu.rs:3592-3605`) pins
128/128/64 only, and every one of those is ACC_SLOTS slot 0 only. Raising
MAX_HEAD_DIM changes the unroll from 2 to 4 slots. The shapes most exposed to
a codegen change are 256 (two full slots) and 200 (partial second slot), and
neither is pinned. The partial kernel has no hash at all.
*Contract change:* insert a new change 3a **before** change 3:
> 3a. At the current HEAD, before any kernel edit, add two cases to
> `paged_attention_indirect`'s case list and `BASELINE_SHA256`:
> `gqa-256-decode` (kv_heads 2, head_dim 256, page_tokens 32, pages 4,
> heads 8, scale `reciprocal_sqrt_scale(256)`, window 0, rows 1, history
> 100, first_position 99) and `gqa-200-chunk` (kv_heads 2, head_dim 200,
> page_tokens 16, pages 7, heads 6, scale 1.0, window 0, rows 8, history
> 100, first_position 92). Run `cargo xtask-cuda test-gpu` once on the
> unchanged kernel to capture both v1 SHA-256s. The capture is valid only
> if SM86 and SM120 print the same hash, as the existing three do. Commit
> the captured hashes. Then apply change 3.
> The partial kernel is covered by the oracle only; state that in the Result.

Add a stop condition: "a captured 256/200 hash differs between SM86 and SM120."

**H4 — "shapes only" cannot pass: the test ledger is 64 MiB and the weight is ~123 MB.**
`affine_linear_device.rs` builds `CapacitySnapshot::new(Scope::Host, 64 << 20, …)`
and the same for the device, at about line 555. INT8 21,504×5,376 codes are
115,605,504 B, plus 7,225,344 B of BF16 scales. `ResidencyAuthority::open`
refuses.
*Contract change:* replace "(shapes only)" in Allowed files, and extend change 2:
> In `run_case`, raise both `CapacitySnapshot` totals from `64 << 20` to
> `256 << 20`. Add the new rows to `CASES`. No other edit to that file.

**H5 — acceptance never runs the extended affine test.**
`affine_linear_device.rs` is `#![cfg(feature = "driver")]` (line 26). Neither
`cargo test --workspace` nor `test-gpu` compiles or runs it.
*Contract change:* add to GPU gates:
> `cargo test -p moxie-executor --features driver --test affine_linear_device
> --locked -- --nocapture` (runs every visible device); quote each new
> case's printed worst ULP and cancellation count in the Result.

## MEDIUM

**M1 — the outcome would lower `max_output` from 65,536 to 21,504.**
The outcome says "the affine descriptors declare `max_input` and `max_output`
21,504". The linear descriptors already declare `max_output` 65,536
(`lib.rs:440`, `577`). *Contract change:* in the outcome, "raise `max_input` to
21,504 in the affine **Linear** descriptors (`max_output` stays 65,536,
`max_rows` unchanged)".

**M2 — change 1's "widen to 64-bit where needed" is a no-op that invites a kernel edit.**
Any edit to `affine_linear.cu` rebuilds both the affine fatbin and the dense
graph fatbin (`dense_graph.cu` `#include`s it) with no reason. *Contract
change:* remove `cuda/affine_linear.cu` from Allowed files. Replace the audit
bullet with:
> No kernel edit: every index is `unsigned long long` (affine_linear.cu
> 58-61, 79-122, 151-155; affine_decode.cuh 169-195), the grid is
> `u32`-checked at `affine_linear.rs:1628-1635`. Copy this list into the Result.

Add a stop condition: "the audit finds a 32-bit index."

**M3 — the INT4 descriptors are raised but only INT8 is qualified.**
The affine loop sets both widths from one literal. Add one row:
> `(rows, in, out) = (1, 21504, 5376)`, INT4 group-32 symmetric, BF16 scales,
so the INT4 code path (`k >> 1` at `affine_decode.cuh:192`) is qualified at
the new depth too.

**M4 — change 4 misses the case registration, and change 5's run command does not exist.**
`test-gpu` has no case filter, and every case is listed twice: in the name
list (`gpu.rs:84-106`) and in the `results.push(case(...))` dispatch
(~`gpu.rs:253-280`). *Contract change:* in change 4, "register
`paged_attention_head_dim_512` in both the case-name list and the dispatch,
after `paged_attention_indirect`". Replace change 5 with:
> Exact substitution: `#define MOXIE_ATTN_ACC_SLOTS (MOXIE_ATTN_MAX_HEAD_DIM /
> MOXIE_ATTN_THREADS)` → `#define MOXIE_ATTN_ACC_SLOTS 2`. Run
> `cargo xtask-cuda test-gpu --profile sm86` once; the verdict read is
> `paged_attention_head_dim_512` = FAILED (and `mha-512-widest-declared`
> if H2 is adopted). Revert and confirm `git diff` on the file is empty.

## LOW

**L1 — the fixture is unspecified beyond "group-32 symmetric".** Specify
`Grouping::Contiguous { size: 32 }`, `asymmetric: false`,
`ScaleDtype::Bf16`, `mapped: false`, which is the checkpoint's own encoding
(`down_proj.weight_scale` BF16 [5376, 672]).

**L2 — "update `paged_attention_declares` and the catalogue" needs no edit.**
Both read `PAGED_ATTENTION_MAX_HEAD_DIM` (`lib.rs:175-176`, `495-496`). So
do the host guard (`paged_attention.rs:146`) and the refusal tests
(`paged_attention.rs:995`, `paged_attention_device.rs:466`). Say "no edit;
confirm by grep" so the builder does not add a second statement of the bound.

**L3 — no command is given for the resource-usage check.** `build.rs` drives
nvcc, so give the command:
`cuobjdump -res-usage $(find target -name paged_attention.fatbin | head -1)`.
It reports both sm_86 and sm_120 entries. Quote REG and SHARED for the three
attention symbols.

**L4 — there is no stop condition for the numerical gate.** Add: "a 21,504 case
fails ADR 0028's two-clause gate → stop and report the element, its terms
and ULP; the gate is the owner's."

**L5 — record error: the header names the Codex luna/sol team.** Commit
f2cc331 switched to Sonnet `builder` / Opus `reviewer`. Update the Identity
bullet.

Note, not a finding: the test rebuilds the tensor and the FP32 oracle per
device, at opt-level 0. That is about 4.7e9 MACs per device across the new
rows. Expect minutes; no restructure is asked for.

Ponytail: the contract shrinks. It loses one allowed file (affine_linear.cu)
and one no-op edit step (L2). Net: −1 file, −2 steps, +2 small cases.
