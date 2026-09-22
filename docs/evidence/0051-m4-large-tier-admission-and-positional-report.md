# M4 large-tier admission and positional report

Status: **evidence delivered; owner acceptance pending** (task 0051, 2026-09-22).

This is an arithmetic and host-oracle report. It adds no device capability, does
not execute a 100k/200k/1m-token history, and contains no performance result.
The labels below mean arithmetic reachability under the stated assumptions, not
model support. R19's distinction between nominal width and actual context still
applies.

## Published inputs and explicit assumptions

- The 32K gate uses `kv_heads = 2`, `head_dim = 128`, `page_tokens = 256`,
  BF16 K/V, and one layer (`xtask/src/gpu.rs`, `paged_attention_32k`). Its KV
  payload cost is therefore
  `2 * 128 * 2 * 2 = 1,024 B/row/layer`.
- Hardware inventory reports 15,883 MiB on the RTX 5060 Ti and 24,123 MiB on
  each RTX 3090 ([hardware-inventory.md](hardware-inventory.md), device table).
  The 62.6 GiB aggregate is not pooled: the current engine has no TP/PP, so a
  generation is bounded by one GPU.
- This report reserves **2,048 MiB per GPU** for weights, activations, page
  tables, workspace, allocator fragmentation and other non-KV use. That reserve
  is an explicit accounting assumption, not a measured free-space claim.
- `HostBackedPlan::MAX_STAGED_BLOCKS = 3`
  ([`report.rs`](../../crates/moxie-memory/src/report.rs)) and the qualified
  page width is 256 rows. The accepted host-backed executor admits exactly
  one resident page plus at most three staged pages:
  `256 + 3 * 256 = 1,024` total rows. This is an alternative qualified
  geometry, not 768 rows added to the larger resident-only envelope; it does
  not allocate three simultaneous staging buffers. The plan refuses
  `resident_rows != page_tokens` ([`lib.rs`](../../crates/moxie-plan/src/lib.rs),
  lines 661–665), and the executor derives the staged count from one
  `page_tokens` resident page ([`paged_attention.rs`](../../crates/moxie-executor/src/paged_attention.rs),
  lines 331–350 and 3554–3556).
- `L = 1` is the only layer count directly exercised by the 32K gate. For
  sensitivity, the second table also shows `L = 78`, the
  `num_hidden_layers` value read from the local GLM-5.2 `config.json` metadata
  on 2026-09-22. No checkpoint weights were read, and this row is not a GLM
  execution claim: the actual checkpoint's MLA geometry is not the synthetic
  32K geometry used here.

For layer count `L`:

```text
row_bytes       = 1,024 * L
usable_bytes    = (measured_mib - 2,048) * 1,048,576
resident_rows   = floor(usable_bytes / row_bytes)
    qualified_host_backed_rows = 256 * (1 + 3) = 1,024
```

These are KV payload rows. The 32K gate's admitted arena is 33,915,648 B for
33,554,432 B of one-layer KV payload, so the larger-tier rows below are
optimistic about fixed allocation overhead; no unqualified extrapolation of
that overhead is hidden in the result.

## Admission arithmetic

| GPU | measured VRAM | usable after reserve | `L` | KV row cost | resident-only rows | qualified host-backed total |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5060 Ti (`sm_120`) | 15,883 MiB | 13,835 MiB | 1 | 1,024 B | 14,167,040 | 1,024 rows |
| RTX 3090 (`sm_86`) | 24,123 MiB | 22,075 MiB | 1 | 1,024 B | 22,604,800 | 1,024 rows |
| RTX 5060 Ti (`sm_120`) | 15,883 MiB | 13,835 MiB | 78 | 79,872 B | 181,628 | 1,024 rows |
| RTX 3090 (`sm_86`) | 24,123 MiB | 22,075 MiB | 78 | 79,872 B | 289,805 | 1,024 rows |

The directly qualified one-layer fixture reaches all three numeric tiers by
payload arithmetic, but that is not a model-level support statement. The
78-layer sensitivity row is the honest model-scale warning:

| Requested tier | 5060 Ti result | 3090 result | all-current-device result |
|---|---|---|---|
| 100,000 | **reached** resident-only, margin +81,628 rows | **reached** resident-only, margin +189,805 | **reached arithmetically** under `L=78`; no execution/support claim |
| 200,000 | **blocked**, resident-only is short by 18,372 rows | **reached** resident-only, margin +89,805 | **partially reached** across hardware; blocked on the 5060 Ti, so not portable across the qualified set |
| 1,000,000 | **blocked**, resident-only is short by 818,372 rows | **blocked**, resident-only is short by 710,195 rows | **blocked** |

The qualified host-backed path is exactly 1,024 total rows, not an extension
of either resident-only budget above. It is smaller than both resident-only
envelopes in this accounting scenario, so it adds nothing to their tier
margins. Reaching a blocked tier through host staging would require a
separately qualified geometry/generalization and, for any actual model, the
model's real per-layer KV geometry and all-layer resource plan. It is not
authorized or implemented by M4.

## Positional capability, independently of admission

The host-only test
`rope::tests::large_tier_positions_match_fp64_and_absolute_masking` exercises
the existing `rope_head`, `chunk_mask`, `page_of`, and multi-head path through
`attend_row`; it adds no positional math. Before running it, the criterion was:

1. every RoPE output is finite;
2. its normalized comparison with the independent FP64 transcription is within
   `gamma(4) = 2.384186359449949e-7`; and
3. causal/sliding absolute-position masking and page indexing select the exact
   expected rows.

The test uses the last position of each requested tier (`99,999`, `199,999`,
`999,999`) and checks the sliding-window result independently (`[3.0, 0.0]`)
as well as the mask endpoints and `page_of` result.

| Tier | position | normalized max | RMS | p99 | positional result |
|---|---:|---:|---:|---:|---|
| 100k | 99,999 | 5.882e-8 | 2.149e-8 | 5.882e-8 | **PASS** |
| 200k | 199,999 | 6.213e-8 | 2.415e-8 | 6.213e-8 | **PASS** |
| 1m | 999,999 | 6.072e-8 | 1.904e-8 | 6.072e-8 | **PASS** |

Positional arithmetic passing does not change any admission result above. No
performance measurement or large-tier device execution was performed.

## Commands and result

- `cargo test -p moxie-oracles --lib rope::tests::large_tier_positions_match_fp64_and_absolute_masking --locked -- --nocapture` — **passed**, one test; all three positional rows passed.
- `cargo test --workspace --locked --offline` — **passed**, including all workspace and doc tests.
- `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` — **passed**.
- `cargo clippy --workspace --all-targets --locked --offline --features moxie-executor/driver -- -D warnings` — **passed**.
- `cargo xtask arch-check` — **passed**, 79 rejected and 21 accepted fixtures across 13 rules.
- `cargo xtask spec-check` — **passed**, all 10 normative documents unchanged.
- `cargo fmt --all -- --check` and `git diff --check` — **passed**.
- No GPU command was required or run; this report contains no device or performance measurement.

## Scope and next task

No production paths, streaming limits, kernels, or positional functions were
changed. The evidence supports the exit-gate report only. An executed 100k tier,
N-block generalization beyond 3, or model-level MLA capacity requires a new
bounded task with real hardware evidence.
