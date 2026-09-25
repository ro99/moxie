# Plan comparison estimates

These are deterministic estimates using the task 0095 costs in
`docs/evidence/topology-costs.md`; comparison itself opened no CUDA device. The
four unique commands below were each run twice and `cmp` confirmed byte-identical
output. The fit check is a resident-bytes fit for weights plus KV at full
context. It excludes activation and peer scratch buffers, so it is not full
execution admission.

## gemma-dense-fits

At prompt 512, the single 5060 Ti ranks first at 44.679 s. At prompt 32,768,
TP2 on the 3090 pair ranks first at 2,597.250 s. All candidates fit the
resident-byte check.

### Prompt 512, generate 256

```text
gemma-dense-fits: prompt=512, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | single+97fe4889 | 97fe4889=5.845 | 29540.983 | 58.656 | 44.679 | |
| 2 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.250, 81fe4578=2.250, 97fe4889=1.589 | 29545.183 | 59.808 | 44.978 | |
| 3 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.045, 81fe4578=3.045 | 29551.050 | 60.174 | 45.077 | |
| 4 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.175, 81fe4578=2.203, 97fe4889=1.589 | 50781.772 | 100.905 | 76.828 | |
| 5 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.203, 81fe4578=2.175, 97fe4889=1.589 | 50782.096 | 100.905 | 76.828 | |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.203, 97fe4889=1.467 | 51388.107 | 102.086 | 77.736 | |
| 7 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.203, 81fe4578=2.297, 97fe4889=1.467 | 51388.934 | 102.087 | 77.738 | |
| 8 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.175, 97fe4889=1.495 | 51854.226 | 102.996 | 78.435 | |
| 9 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.175, 81fe4578=2.297, 97fe4889=1.495 | 51854.730 | 102.997 | 78.436 | |
| 10 | single+3032cfa3 | 3032cfa3=5.845 | 58694.740 | 116.542 | 88.773 | |
| 11 | single+81fe4578 | 81fe4578=5.845 | 58718.013 | 116.588 | 88.808 | |
```

### Prompt 32768, generate 256

```text
gemma-dense-fits: prompt=32768, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.424, 81fe4578=3.424 | 2574028.179 | 90.633 | 2597.250 | |
| 2 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.536, 81fe4578=2.536, 97fe4889=1.776 | 2574706.188 | 90.313 | 2597.846 | |
| 3 | single+97fe4889 | 97fe4889=6.603 | 2577711.406 | 89.303 | 2600.593 | |
| 4 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.461, 81fe4578=2.489, 97fe4889=1.776 | 4450231.831 | 154.380 | 4489.787 | |
| 5 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.489, 81fe4578=2.461, 97fe4889=1.776 | 4450252.558 | 154.380 | 4489.808 | |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.489, 97fe4889=1.654 | 4489037.497 | 155.560 | 4528.895 | |
| 7 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.489, 81fe4578=2.583, 97fe4889=1.654 | 4489090.433 | 155.562 | 4528.948 | |
| 8 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.461, 97fe4889=1.682 | 4518869.136 | 156.470 | 4558.960 | |
| 9 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.461, 81fe4578=2.583, 97fe4889=1.682 | 4518901.345 | 156.471 | 4558.992 | |
| 10 | single+3032cfa3 | 3032cfa3=6.603 | 5121633.974 | 177.436 | 5167.097 | |
| 11 | single+81fe4578 | 81fe4578=6.603 | 5123664.735 | 177.507 | 5169.145 | |
```

## gemma-dense-large

At prompt 512, the two-3090 stage followed by the 5060 Ti ranks first at
234.349 s. At prompt 32,768, TP2 on the 3090 pair ranks first at 13,641.825 s.
Single-device plans are refused by the resident-byte fit. These figures exclude
activation and peer scratch space.

### Prompt 512, generate 256

```text
gemma-dense-large: prompt=512, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | pipeline+3032cfa3+81fe4578@0..3948/97fe4889@3948..5259 | 3032cfa3=11.510, 81fe4578=11.510, 97fe4889=7.675 | 153920.444 | 311.653 | 234.349 | |
| 2 | tp2+3032cfa3+81fe4578 | 3032cfa3=15.348, 81fe4578=15.348 | 153930.641 | 313.625 | 234.863 | |
| 3 | pipeline+81fe4578@0..1965/3032cfa3@1965..3948/97fe4889@3948..5259 | 3032cfa3=11.443, 81fe4578=11.452, 97fe4889=7.675 | 267562.844 | 531.473 | 404.750 | |
| 4 | pipeline+3032cfa3@0..1965/81fe4578@1965..3948/97fe4889@3948..5259 | 3032cfa3=11.452, 81fe4578=11.443, 97fe4889=7.675 | 267563.245 | 531.473 | 404.750 | |
| 5 | pipeline+81fe4578@0..1965/97fe4889@1965..3274/3032cfa3@3274..5259 | 3032cfa3=11.565, 81fe4578=11.452, 97fe4889=7.553 | 268169.179 | 532.654 | 405.658 | |
| 6 | pipeline+3032cfa3@0..1965/97fe4889@1965..3274/81fe4578@3274..5259 | 3032cfa3=11.452, 81fe4578=11.565, 97fe4889=7.553 | 268170.084 | 532.655 | 405.660 | |
| 7 | pipeline+97fe4889@0..1291/81fe4578@1291..3274/3032cfa3@3274..5259 | 3032cfa3=11.565, 81fe4578=11.443, 97fe4889=7.562 | 268732.425 | 533.772 | 406.510 | |
| 8 | pipeline+97fe4889@0..1291/3032cfa3@1291..3274/81fe4578@3274..5259 | 3032cfa3=11.443, 81fe4578=11.565, 97fe4889=7.562 | 268732.928 | 533.773 | 406.511 | |
| — | single+3032cfa3 | | | | | resident-byte fit needs 32693381888 bytes on GPU-3032cfa3-19df-028f-5ebd-43314911e0b9, which has 25017319424 usable |
| — | single+81fe4578 | | | | | resident-byte fit needs 32693381888 bytes on GPU-81fe4578-59b2-37c4-421e-287cdac78704, which has 25017319424 usable |
| — | single+97fe4889 | | | | | resident-byte fit needs 32693381888 bytes on GPU-97fe4889-4874-a378-198e-955d2e72c4a3, which has 16512122880 usable |
```

### Prompt 32768, generate 256

```text
gemma-dense-large: prompt=32768, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | tp2+3032cfa3+81fe4578 | 3032cfa3=17.403, 81fe4578=17.403 | 13519227.364 | 478.479 | 13641.825 | |
| 2 | pipeline+3032cfa3+81fe4578@0..3948/97fe4889@3948..5259 | 3032cfa3=13.052, 81fe4578=13.052, 97fe4889=8.701 | 13524369.073 | 476.761 | 13646.527 | |
| 3 | pipeline+81fe4578@0..1965/3032cfa3@1965..3948/97fe4889@3948..5259 | 3032cfa3=13.016, 81fe4578=12.963, 97fe4889=8.701 | 23550063.659 | 820.238 | 23760.232 | |
| 4 | pipeline+3032cfa3@0..1965/81fe4578@1965..3948/97fe4889@3948..5259 | 3032cfa3=12.963, 81fe4578=13.016, 97fe4889=8.701 | 23550120.575 | 820.240 | 23760.290 | |
| 5 | pipeline+81fe4578@0..1965/97fe4889@1965..3274/3032cfa3@3274..5259 | 3032cfa3=13.138, 81fe4578=12.963, 97fe4889=8.579 | 23588869.325 | 821.419 | 23799.340 | |
| 6 | pipeline+3032cfa3@0..1965/97fe4889@1965..3274/81fe4578@3274..5259 | 3032cfa3=12.963, 81fe4578=13.138, 97fe4889=8.579 | 23588958.450 | 821.422 | 23799.430 | |
| 7 | pipeline+97fe4889@0..1291/81fe4578@1291..3274/3032cfa3@3274..5259 | 3032cfa3=13.138, 81fe4578=13.016, 97fe4889=8.526 | 23664071.010 | 824.927 | 23875.442 | |
| 8 | pipeline+97fe4889@0..1291/3032cfa3@1291..3274/81fe4578@3274..5259 | 3032cfa3=13.016, 81fe4578=13.138, 97fe4889=8.526 | 23664103.219 | 824.928 | 23875.475 | |
| — | single+3032cfa3 | | | | | resident-byte fit needs 37106313984 bytes on GPU-3032cfa3-19df-028f-5ebd-43314911e0b9, which has 25017319424 usable |
| — | single+81fe4578 | | | | | resident-byte fit needs 37106313984 bytes on GPU-81fe4578-59b2-37c4-421e-287cdac78704, which has 25017319424 usable |
| — | single+97fe4889 | | | | | resident-byte fit needs 37106313984 bytes on GPU-97fe4889-4874-a378-198e-955d2e72c4a3, which has 16512122880 usable |
```

## Comparison with the earlier memory-only estimate

Task 0072 measured a median 35.880 ms per decode on one 3090 and 184.975 ms for
the combined TP2+TP1 path (eight committed decode steps). The previous Shape A
memory-only estimate was 0.108713 ms prefill, 0.106523 ms first decode at
context 7 and 0.960903 ms total for prompt 7 / generate 8, with 33,696 resident
bytes on each 3090 and 74,928 bytes on the 5060 Ti. Those estimates predate the
compute term and are retained as a historical baseline; this task did not
re-estimate that Shape A case.

The current term uses one measured dense linear rate at a single shape for
Linear, VocabProjection and ExpertMlp operations, estimates conventional
attention FLOPs separately, and assumes perfect memory/compute overlap. Host
transfers use pageable-link costs, and pipeline cuts still use the greedy
capacity-weighted rule. These remain estimates, not performance claims.

## Phase pairs

The costs TOML was extracted from the run 2 block in
`docs/evidence/topology-costs.md` to `/tmp/task0095-topology-run2.toml`. Each
command below was run twice; `cmp` confirmed identical output for both runs.

### Prompt 512, generate 256

```text
$ CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda compare-plans --costs /tmp/task0095-topology-run2.toml --model gemma-dense-fits --prompt 512 --generate 256
gemma-dense-fits: prompt=512, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | single+97fe4889 | 97fe4889=5.845 | 29540.983 | 58.656 | 44.679 | |
| 2 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.250, 81fe4578=2.250, 97fe4889=1.589 | 29545.183 | 59.808 | 44.978 | |
| 3 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.045, 81fe4578=3.045 | 29551.050 | 60.174 | 45.077 | |
| 4 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.175, 81fe4578=2.203, 97fe4889=1.589 | 50781.772 | 100.905 | 76.828 | |
| 5 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.203, 81fe4578=2.175, 97fe4889=1.589 | 50782.096 | 100.905 | 76.828 | |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.203, 97fe4889=1.467 | 51388.107 | 102.086 | 77.736 | |
| 7 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.203, 81fe4578=2.297, 97fe4889=1.467 | 51388.934 | 102.087 | 77.738 | |
| 8 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.175, 97fe4889=1.495 | 51854.226 | 102.996 | 78.435 | |
| 9 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.175, 81fe4578=2.297, 97fe4889=1.495 | 51854.730 | 102.997 | 78.436 | |
| 10 | single+3032cfa3 | 3032cfa3=5.845 | 58694.740 | 116.542 | 88.773 | |
| 11 | single+81fe4578 | 81fe4578=5.845 | 58718.013 | 116.588 | 88.808 | |
phase pairs
| Rank | Prefill candidate | Decode candidate | Transition MB | Via host | Prefill ms | First decode ms | Total s |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | single+97fe4889 | single+97fe4889 | 0.000 | no | 29540.983 | 58.656 | 44.679 |
| 2 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | single+97fe4889 | 39.846 | yes | 29545.183 | 58.656 | 44.697 |
| 3 | tp2+3032cfa3+81fe4578 | single+97fe4889 | 52.429 | yes | 29551.050 | 58.656 | 44.707 |
| 4 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 0.000 | no | 29545.183 | 59.808 | 44.978 |
| 5 | single+97fe4889 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 39.846 | yes | 29540.983 | 59.808 | 44.986 |
| 6 | tp2+3032cfa3+81fe4578 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 12.583 | yes | 29551.050 | 59.808 | 44.988 |
| 7 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 12.583 | yes | 29545.183 | 60.174 | 45.075 |
| 8 | tp2+3032cfa3+81fe4578 | tp2+3032cfa3+81fe4578 | 0.000 | no | 29551.050 | 60.174 | 45.077 |
| 9 | single+97fe4889 | tp2+3032cfa3+81fe4578 | 52.429 | yes | 29540.983 | 60.174 | 45.083 |
| 10 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 19.923 | no | 29545.183 | 100.905 | 55.595 |
best same-placement pair: rank 1 (single+97fe4889)
```

### Prompt 32768, generate 256

```text
$ CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda compare-plans --costs /tmp/task0095-topology-run2.toml --model gemma-dense-fits --prompt 32768 --generate 256
gemma-dense-fits: prompt=32768, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.424, 81fe4578=3.424 | 2574028.179 | 90.633 | 2597.250 | |
| 2 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.536, 81fe4578=2.536, 97fe4889=1.776 | 2574706.188 | 90.313 | 2597.846 | |
| 3 | single+97fe4889 | 97fe4889=6.603 | 2577711.406 | 89.303 | 2600.593 | |
| 4 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.461, 81fe4578=2.489, 97fe4889=1.776 | 4450231.831 | 154.380 | 4489.787 | |
| 5 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.489, 81fe4578=2.461, 97fe4889=1.776 | 4450252.558 | 154.380 | 4489.808 | |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.489, 97fe4889=1.654 | 4489037.497 | 155.560 | 4528.895 | |
| 7 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.489, 81fe4578=2.583, 97fe4889=1.654 | 4489090.433 | 155.562 | 4528.948 | |
| 8 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.461, 97fe4889=1.682 | 4518869.136 | 156.470 | 4558.960 | |
| 9 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.461, 81fe4578=2.583, 97fe4889=1.682 | 4518901.345 | 156.471 | 4558.992 | |
| 10 | single+3032cfa3 | 3032cfa3=6.603 | 5121633.974 | 177.436 | 5167.097 | |
| 11 | single+81fe4578 | 81fe4578=6.603 | 5123664.735 | 177.507 | 5169.145 | |
phase pairs
| Rank | Prefill candidate | Decode candidate | Transition MB | Via host | Prefill ms | First decode ms | Total s |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | tp2+3032cfa3+81fe4578 | single+97fe4889 | 889.192 | yes | 2574028.179 | 89.303 | 2597.210 |
| 2 | tp2+3032cfa3+81fe4578 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 218.104 | yes | 2574028.179 | 90.313 | 2597.242 |
| 3 | tp2+3032cfa3+81fe4578 | tp2+3032cfa3+81fe4578 | 0.000 | no | 2574028.179 | 90.633 | 2597.250 |
| 4 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | single+97fe4889 | 671.089 | yes | 2574706.188 | 89.303 | 2597.814 |
| 5 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 0.000 | no | 2574706.188 | 90.313 | 2597.846 |
| 6 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 218.104 | yes | 2574706.188 | 90.633 | 2597.994 |
| 7 | single+97fe4889 | single+97fe4889 | 0.000 | no | 2577711.406 | 89.303 | 2600.593 |
| 8 | single+97fe4889 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 671.089 | yes | 2577711.406 | 90.313 | 2601.056 |
| 9 | single+97fe4889 | tp2+3032cfa3+81fe4578 | 889.192 | yes | 2577711.406 | 90.633 | 2601.205 |
| 10 | tp2+3032cfa3+81fe4578 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 553.648 | yes | 2574028.179 | 154.380 | 2613.715 |
best same-placement pair: rank 3 (tp2+3032cfa3+81fe4578)
```

With the compute term, the best `gemma-dense-fits` plan changes from TP2 to
one 5060 Ti at prompt 512 (44.679 s), and from one 3090 to TP2 at prompt 32,768
(2,597.250 s). For `gemma-dense-large`, it changes from TP2 to the two-3090
stage plus the 5060 Ti at prompt 512 (234.349 s), and from the 5060 Ti first
pipeline to TP2 at prompt 32,768 (13,641.825 s). At prompt 512 the best split
`gemma-dense-fits` phase pair takes 44.697 s versus 44.679 s for the best
same-placement pair, 0.018 s slower; at prompt 32,768 the split pair takes
2,597.210 s versus 2,597.250 s, 0.040 s faster. These are estimates only.
