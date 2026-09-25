fixture-scale, O6 open, not a performance claim

# M6 stress graph benchmark

The benchmark measures fixture-scale `Shape A`, `Shape C` top-2, and `Shape C`
top-3 on the NVIDIA GeForce RTX 3090, UUID
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, driver `610.57.04`, with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. The candidate test commit is
`bdd099065dabbfaba8a0e9c9191425ef3d6bfb92`; the paired base is `08a6fdb`.

Candidate command, run twice:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test --release -p moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks --test dense_gemma_device stress_graph_benchmark -- --ignored --nocapture
~~~

The test uses five warm-ups and 30 measured executions per phase. It times
`execute_dense` plus `finish`; transaction aborts are outside the timed span.
`device_bytes` is the selected plan's three admitted regions, plus graph-pool
reservation for `decode-captured`. `paged_state_bytes` is the ledger's
`KvStatePages` commitment for the admitted paged runs. The candidate's
`worst_ulp` values compare the last committed prefill row and committed
captured decode row against `host_step`; the eager decode timing line has no
committed output and prints `-`.

## Candidate run 1

| Graph | Phase | Median (µs) | Min (µs) | Max (µs) | Device bytes | Graph pool bytes | Paged state bytes | Worst ULP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| a | prefill | 1021.909 | 1016.790 | 1089.011 | 92,416 | - | 19,968 | 0.000 |
| a | decode | 968.910 | 964.820 | 1057.040 | 88,064 | - | 19,968 | - |
| a | decode-captured | 856.400 | 853.598 | 933.394 | 2,177,024 | 2,088,960 | 19,968 | 0.000 |
| c-top2 | prefill | 2027.691 | 2023.591 | 2102.702 | 129,280 | - | 19,968 | 0.000 |
| c-top2 | decode | 1953.756 | 1938.252 | 2021.912 | 124,928 | - | 19,968 | - |
| c-top2 | decode-captured | 1778.617 | 1756.927 | 1808.115 | 2,754,560 | 2,629,632 | 19,968 | 0.000 |
| c-top3 | prefill | 2334.419 | 2310.718 | 2343.839 | 129,536 | - | 19,968 | 0.000 |
| c-top3 | decode | 2209.170 | 2186.186 | 2234.450 | 124,928 | - | 19,968 | - |
| c-top3 | decode-captured | 2039.841 | 2025.233 | 2113.466 | 2,754,560 | 2,629,632 | 19,968 | 0.000 |

## Candidate run 2

| Graph | Phase | Median (µs) | Min (µs) | Max (µs) | Device bytes | Graph pool bytes | Paged state bytes | Worst ULP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| a | prefill | 1052.897 | 1042.605 | 1058.926 | 92,416 | - | 19,968 | 0.000 |
| a | decode | 990.460 | 977.495 | 1052.550 | 88,064 | - | 19,968 | - |
| a | decode-captured | 878.328 | 860.028 | 909.566 | 2,177,024 | 2,088,960 | 19,968 | 0.000 |
| c-top2 | prefill | 2053.144 | 2047.601 | 2087.302 | 129,280 | - | 19,968 | 0.000 |
| c-top2 | decode | 1975.986 | 1956.423 | 2010.581 | 124,928 | - | 19,968 | - |
| c-top2 | decode-captured | 1800.535 | 1777.679 | 1845.302 | 2,754,560 | 2,629,632 | 19,968 | 0.000 |
| c-top3 | prefill | 2356.574 | 2338.596 | 2417.940 | 129,536 | - | 19,968 | 0.000 |
| c-top3 | decode | 2240.040 | 2208.542 | 2291.400 | 124,928 | - | 19,968 | - |
| c-top3 | decode-captured | 2056.709 | 2036.028 | 2083.501 | 2,754,560 | 2,629,632 | 19,968 | 0.000 |

## Paired base run

The full test-only diff did not apply at `08a6fdb`: that checkout predates the
appended M6 timing harness used as the patch context. Following the task
fallback, the same test was appended temporarily with the capture phase
removed; the eager prefill/decode phases built and ran. No production files
were changed. The temporary test edit was discarded before returning to
`main`.

Command:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID cargo test --release -p moxie-executor --features driver,paged-attention-binding,paged-attention-test-hooks --test dense_gemma_device stress_graph_benchmark -- --ignored --nocapture
~~~

| Graph | Phase | Median (µs) | Min (µs) | Max (µs) | Device bytes | Graph pool bytes | Paged state bytes | Worst ULP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| a | prefill | 1876.929 | 1770.129 | 1964.731 | 91,904 | - | 19,968 | 0.000 |
| a | decode | 1766.142 | 1717.133 | 1861.150 | 87,552 | - | 19,968 | 0.000 |
| c-top2 | prefill | 2915.977 | 2895.888 | 3074.432 | 128,768 | - | 19,968 | 0.000 |
| c-top2 | decode | 2835.400 | 2825.104 | 2965.431 | 124,416 | - | 19,968 | 0.000 |
| c-top3 | prefill | 3215.086 | 3202.739 | 3305.346 | 128,768 | - | 19,968 | 0.000 |
| c-top3 | decode | 3091.567 | 3079.972 | 3172.244 | 124,416 | - | 19,968 | 0.000 |

Across the six eager phases shared with the base, the mean of the two
candidate medians was 27.05–44.73% lower than the single base median, depending
on graph and phase; the candidate-only captured decode medians were 856.400
and 878.328 µs for Shape A, 1778.617 and 1800.535 µs for top-2, and 2039.841
and 2056.709 µs for top-3. Candidate plan regions were 92,416/88,064 bytes
(Shape A prefill/decode), 129,280/124,928 bytes (top-2), and 129,536/124,928
bytes (top-3); captured decode graph pools were 2,088,960 bytes for Shape A
and 2,629,632 bytes for both routed graphs, with 19,968 bytes of paged state
for each case. The worst committed output error was 0.000 BF16 ULP for every
graph at prefill and decode; these fixture-scale readings do not establish
checkpoint behavior or a general performance result.

Raw outputs remain outside Git at `/tmp/task-0090-candidate-run-1.txt`,
`/tmp/task-0090-candidate-run-2.txt`, and `/tmp/task-0090-base-run.txt`.
