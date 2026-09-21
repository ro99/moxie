# 0008 — Paged attention and its final state binding mutation battery

Date: 2026-09-20. Tasks: 0037 and 0038. Status: **PASSED**.

## Question

Can the common BF16 paged-attention kernel, its numerical bound, or the final
state/executor binding be wrong while the acceptance gates still pass?

## Method

`cargo xtask mutation-check --battery 0037` applies one exact source
substitution at a time. Its four lanes are the `moxie-state` device tests, the
attention-oracle tests, the executor's paged-attention binding tests, and
`cargo xtask-cuda test-gpu --profile sm120`.

The unmodified lanes passed three times before the mutations and three times
after restoration. Each deciding lane then reproduced its failure three times
with the mutation installed. The harness reported no skipped anchor and its
own `--self-test` passed 122 of 122 cases.

## Result

| Substitution | Deciding lane |
|---|---|
| Publish a multi-layer frontier after one layer | state |
| Keep the old retention watermark at commit | state |
| Do not poison after partial page-view publication | state |
| Skip the authority's page-view publication | binding |
| Alias selected output to the query slot | device |
| Launch selected attention at base zero after reclamation | device |
| Compute bound weights at scale 1 instead of the declared scale | oracle |
| Leak 1 MiB on every decode append | binding |
| Ignore the declared scale in the CUDA kernel | device |
| Ignore the page table in the CUDA kernel | binding |
| Do not rescale the running online-softmax partial | device |

**11 of 11 mutants were caught.** There were zero survivors, unstable
verdicts, invalid controls, broken controls, or skipped mutations.

## Scope

This measures whether the named correctness and lifecycle gates notice these
faults. It is not exhaustive proof, a timing result, or model-output evidence.
O6 and O7 remain open.
