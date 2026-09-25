# Plan comparison estimates

These are pure estimates using the committed task 0073 topology costs in
`docs/evidence/topology-costs.md`; no CUDA device was opened. Each of the four
commands was run twice and `cmp` confirmed byte-identical output. The fit check
is a resident-bytes fit for weights plus KV at full context. It excludes
activation and peer scratch buffers, so it is not full execution admission.

## gemma-dense-fits

At prompt 512, TP2 on the 3090 pair ranks first at 1.680 s, ahead of one 3090
at 1.977 s. At prompt 32,768, one 3090 ranks first at 2.238 s while TP2 takes
13.811 s. The model reads weights and KV once per step; at long prompt the
TP2 collective and pipeline handoff payloads scale with rows, making the
single-device plan faster. All candidates fit the resident-byte check.

### Prompt 512, generate 256

```text
gemma-dense-fits: prompt=512, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.045, 81fe4578=3.045 | 195.892 | 5.788 | 1.680 | |
| 2 | single+81fe4578 | 81fe4578=5.845 | 7.676 | 7.676 | 1.977 | |
| 3 | single+3032cfa3 | 3032cfa3=5.845 | 7.724 | 7.724 | 1.989 | |
| 4 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.250, 81fe4578=2.250, 97fe4889=1.589 | 143.980 | 8.735 | 2.384 | |
| 5 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.203, 81fe4578=2.297, 97fe4889=1.467 | 11.367 | 10.063 | 2.593 | |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.203, 97fe4889=1.467 | 11.419 | 10.064 | 2.593 | |
| 7 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.175, 81fe4578=2.297, 97fe4889=1.495 | 11.456 | 10.105 | 2.603 | |
| 8 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.175, 97fe4889=1.495 | 11.455 | 10.106 | 2.604 | |
| 9 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.203, 81fe4578=2.175, 97fe4889=1.589 | 11.631 | 10.247 | 2.640 | |
| 10 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.175, 81fe4578=2.203, 97fe4889=1.589 | 11.685 | 10.248 | 2.640 | |
| 11 | single+97fe4889 | 97fe4889=5.845 | 16.297 | 16.297 | 4.197 | |
```

### Prompt 32768, generate 256

```text
gemma-dense-fits: prompt=32768, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | single+81fe4578 | 81fe4578=6.603 | 8.704 | 8.704 | 2.238 | |
| 2 | single+3032cfa3 | 3032cfa3=6.603 | 8.758 | 8.758 | 2.252 | |
| 3 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.489, 81fe4578=2.583, 97fe4889=1.654 | 95.032 | 11.376 | 3.008 | |
| 4 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.489, 97fe4889=1.654 | 98.285 | 11.378 | 3.012 | |
| 5 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.461, 81fe4578=2.583, 97fe4889=1.682 | 98.103 | 11.418 | 3.022 | |
| 6 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.461, 97fe4889=1.682 | 97.948 | 11.419 | 3.022 | |
| 7 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.489, 81fe4578=2.461, 97fe4889=1.776 | 100.319 | 11.561 | 3.061 | |
| 8 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.461, 81fe4578=2.489, 97fe4889=1.776 | 103.726 | 11.561 | 3.064 | |
| 9 | single+97fe4889 | 97fe4889=6.603 | 18.479 | 18.479 | 4.751 | |
| 10 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.536, 81fe4578=2.536, 97fe4889=1.776 | 8682.023 | 9.661 | 11.156 | |
| 11 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.424, 81fe4578=3.424 | 12196.396 | 6.305 | 13.811 | |
```

## gemma-dense-large

At prompt 512, TP2 on the 3090 pair ranks first at 8.551 s, ahead of the
three-device pipeline at 11.920 s. At prompt 32,768, the pipeline with the
5060 Ti first ranks first at 15.122 s; TP2 takes 69.826 s because its
row-scaled collective cost dominates this long prefill. Single-device plans
are rejected by the resident-byte fit: weights plus KV require 32.693 GB at
the short context and 37.106 GB at the long context, exceeding usable memory
on every device. These figures do not include activation or peer scratch
space.

### Prompt 512, generate 256

```text
gemma-dense-large: prompt=512, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | tp2+3032cfa3+81fe4578 | 3032cfa3=15.348, 81fe4578=15.348 | 988.962 | 29.496 | 8.551 | |
| 2 | pipeline+3032cfa3+81fe4578@0..3948/97fe4889@3948..5259 | 3032cfa3=11.510, 81fe4578=11.510, 97fe4889=7.675 | 756.334 | 43.530 | 11.920 | |
| 3 | pipeline+3032cfa3@0..1965/97fe4889@1965..3274/81fe4578@3274..5259 | 3032cfa3=11.452, 81fe4578=11.565, 97fe4889=7.553 | 52.729 | 51.425 | 13.245 | |
| 4 | pipeline+81fe4578@0..1965/97fe4889@1965..3274/3032cfa3@3274..5259 | 3032cfa3=11.565, 81fe4578=11.452, 97fe4889=7.553 | 52.782 | 51.426 | 13.246 | |
| 5 | pipeline+97fe4889@0..1291/3032cfa3@1291..3274/81fe4578@3274..5259 | 3032cfa3=11.443, 81fe4578=11.565, 97fe4889=7.562 | 52.791 | 51.439 | 13.249 | |
| 6 | pipeline+97fe4889@0..1291/81fe4578@1291..3274/3032cfa3@3274..5259 | 3032cfa3=11.565, 81fe4578=11.443, 97fe4889=7.562 | 52.790 | 51.441 | 13.249 | |
| 7 | pipeline+3032cfa3@0..1965/81fe4578@1965..3948/97fe4889@3948..5259 | 3032cfa3=11.452, 81fe4578=11.443, 97fe4889=7.675 | 52.994 | 51.610 | 13.293 | |
| 8 | pipeline+81fe4578@0..1965/3032cfa3@1965..3948/97fe4889@3948..5259 | 3032cfa3=11.443, 81fe4578=11.452, 97fe4889=7.675 | 53.047 | 51.610 | 13.293 | |
| — | single+3032cfa3 | | | | | resident-byte fit needs 32693381888 bytes on GPU-3032cfa3-19df-028f-5ebd-43314911e0b9, which has 25017319424 usable |
| — | single+81fe4578 | | | | | resident-byte fit needs 32693381888 bytes on GPU-81fe4578-59b2-37c4-421e-287cdac78704, which has 25017319424 usable |
| — | single+97fe4889 | | | | | resident-byte fit needs 32693381888 bytes on GPU-97fe4889-4874-a378-198e-955d2e72c4a3, which has 16512122880 usable |
```

### Prompt 32768, generate 256

```text
gemma-dense-large: prompt=32768, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | pipeline+97fe4889@0..1291/3032cfa3@1291..3274/81fe4578@3274..5259 | 3032cfa3=13.016, 81fe4578=13.138, 97fe4889=8.526 | 145.168 | 58.484 | 15.122 | |
| 2 | pipeline+97fe4889@0..1291/81fe4578@1291..3274/3032cfa3@3274..5259 | 3032cfa3=13.138, 81fe4578=13.016, 97fe4889=8.526 | 145.014 | 58.485 | 15.122 | |
| 3 | pipeline+3032cfa3@0..1965/97fe4889@1965..3274/81fe4578@3274..5259 | 3032cfa3=12.963, 81fe4578=13.138, 97fe4889=8.579 | 142.215 | 58.560 | 15.138 | |
| 4 | pipeline+81fe4578@0..1965/97fe4889@1965..3274/3032cfa3@3274..5259 | 3032cfa3=13.138, 81fe4578=12.963, 97fe4889=8.579 | 145.469 | 58.562 | 15.142 | |
| 5 | pipeline+3032cfa3@0..1965/81fe4578@1965..3948/97fe4889@3948..5259 | 3032cfa3=12.963, 81fe4578=13.016, 97fe4889=8.701 | 147.503 | 58.744 | 15.191 | |
| 6 | pipeline+81fe4578@0..1965/3032cfa3@1965..3948/97fe4889@3948..5259 | 3032cfa3=13.016, 81fe4578=12.963, 97fe4889=8.701 | 150.910 | 58.745 | 15.194 | |
| 7 | pipeline+3032cfa3+81fe4578@0..3948/97fe4889@3948..5259 | 3032cfa3=13.052, 81fe4578=13.052, 97fe4889=8.701 | 45755.868 | 48.578 | 58.195 | |
| 8 | tp2+3032cfa3+81fe4578 | 3032cfa3=17.403, 81fe4578=17.403 | 61556.398 | 32.295 | 69.826 | |
| — | single+3032cfa3 | | | | | resident-byte fit needs 37106313984 bytes on GPU-3032cfa3-19df-028f-5ebd-43314911e0b9, which has 25017319424 usable |
| — | single+81fe4578 | | | | | resident-byte fit needs 37106313984 bytes on GPU-81fe4578-59b2-37c4-421e-287cdac78704, which has 25017319424 usable |
| — | single+97fe4889 | | | | | resident-byte fit needs 37106313984 bytes on GPU-97fe4889-4874-a378-198e-955d2e72c4a3, which has 16512122880 usable |
```

## Estimated against measured

Task 0072 measured a median 35.880 ms per decode on one 3090 and 184.975 ms
for the combined TP2+TP1 path (eight committed decode steps). For that same
Shape A placement—layers 0–2 on the 3090 pair and layers 3–5 plus the head on
the 5060 Ti—the corrected estimate is **0.108713 ms** prefill,
**0.106523 ms** for first decode at context 7, and **0.960903 ms** total for
prompt 7 / generate 8. The resident bytes remain 33,696 on each 3090 and
74,928 on the 5060 Ti. The fixture is
launch-bound, not weight-bound; the estimator has no kernel-launch or runtime
dispatch term, which is why it underestimates the measured decode time.

Ceilings: there is no compute term, host transfers use pageable rather than
pinned-copy measurements, and pipeline cuts use the greedy capacity-weighted
rule rather than a search over cut sets. These are estimates, not performance
claims.

## Phase pairs

The costs TOML was extracted from the committed code block in
`docs/evidence/topology-costs.md` to `/tmp/task0094-topology-costs.toml`.
Each command below was run twice; `cmp` reported identical output for both
run pairs.

### Prompt 512, generate 256

```text
$ cargo xtask-cuda compare-plans --costs /tmp/task0094-topology-costs.toml --model gemma-dense-fits --prompt 512 --generate 256
gemma-dense-fits: prompt=512, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.045, 81fe4578=3.045 | 195.892 | 5.788 | 1.680 | |
| 2 | single+81fe4578 | 81fe4578=5.845 | 7.676 | 7.676 | 1.977 | |
| 3 | single+3032cfa3 | 3032cfa3=5.845 | 7.724 | 7.724 | 1.989 | |
| 4 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.250, 81fe4578=2.250, 97fe4889=1.589 | 143.980 | 8.735 | 2.384 | |
| 5 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.203, 81fe4578=2.297, 97fe4889=1.467 | 11.367 | 10.063 | 2.593 | |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.203, 97fe4889=1.467 | 11.419 | 10.064 | 2.593 | |
| 7 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.175, 81fe4578=2.297, 97fe4889=1.495 | 11.456 | 10.105 | 2.603 | |
| 8 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.297, 81fe4578=2.175, 97fe4889=1.495 | 11.455 | 10.106 | 2.604 | |
| 9 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.203, 81fe4578=2.175, 97fe4889=1.589 | 11.631 | 10.247 | 2.640 | |
| 10 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.175, 81fe4578=2.203, 97fe4889=1.589 | 11.685 | 10.248 | 2.640 | |
| 11 | single+97fe4889 | 97fe4889=5.845 | 16.297 | 16.297 | 4.197 | |
phase pairs
| Rank | Prefill candidate | Decode candidate | Transition MB | Via host | Prefill ms | First decode ms | Total s |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | single+3032cfa3 | tp2+3032cfa3+81fe4578 | 26.214 | no | 7.724 | 5.788 | 1.496 |
| 2 | single+81fe4578 | tp2+3032cfa3+81fe4578 | 26.214 | no | 7.676 | 5.788 | 1.497 |
| 3 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | tp2+3032cfa3+81fe4578 | 32.506 | yes | 11.367 | 5.788 | 1.502 |
| 4 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | tp2+3032cfa3+81fe4578 | 32.506 | yes | 11.419 | 5.788 | 1.503 |
| 5 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | tp2+3032cfa3+81fe4578 | 32.506 | yes | 11.455 | 5.788 | 1.503 |
| 6 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | tp2+3032cfa3+81fe4578 | 32.506 | yes | 11.456 | 5.788 | 1.503 |
| 7 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 32.506 | yes | 11.631 | 5.788 | 1.503 |
| 8 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 32.506 | yes | 11.685 | 5.788 | 1.503 |
| 9 | single+97fe4889 | tp2+3032cfa3+81fe4578 | 52.429 | yes | 16.297 | 5.788 | 1.516 |
| 10 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 12.583 | yes | 143.980 | 5.788 | 1.632 |
best same-placement pair: rank 11 (tp2+3032cfa3+81fe4578)
```

### Prompt 32768, generate 256

```text
$ cargo xtask-cuda compare-plans --costs /tmp/task0094-topology-costs.toml --model gemma-dense-fits --prompt 32768 --generate 256
gemma-dense-fits: prompt=32768, generate=256
| Rank | Candidate | Per-device GiB | Prefill ms | First decode ms | Total s | Rejection |
|---:|---|---|---:|---:|---:|---|
| 1 | single+81fe4578 | 81fe4578=6.603 | 8.704 | 8.704 | 2.238 | |
| 2 | single+3032cfa3 | 3032cfa3=6.603 | 8.758 | 8.758 | 2.252 | |
| 3 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | 3032cfa3=2.489, 81fe4578=2.583, 97fe4889=1.654 | 95.032 | 11.376 | 3.008 | |
| 4 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.489, 97fe4889=1.654 | 98.285 | 11.378 | 3.012 | |
| 5 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | 3032cfa3=2.461, 81fe4578=2.583, 97fe4889=1.682 | 98.103 | 11.418 | 3.022 | |
| 6 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | 3032cfa3=2.583, 81fe4578=2.461, 97fe4889=1.682 | 97.948 | 11.419 | 3.022 | |
| 7 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | 3032cfa3=2.489, 81fe4578=2.461, 97fe4889=1.776 | 100.319 | 11.561 | 3.061 | |
| 8 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | 3032cfa3=2.461, 81fe4578=2.489, 97fe4889=1.776 | 103.726 | 11.561 | 3.064 | |
| 9 | single+97fe4889 | 97fe4889=6.603 | 18.479 | 18.479 | 4.751 | |
| 10 | pipeline+3032cfa3+81fe4578@0..748/97fe4889@748..995 | 3032cfa3=2.536, 81fe4578=2.536, 97fe4889=1.776 | 8682.023 | 9.661 | 11.156 | |
| 11 | tp2+3032cfa3+81fe4578 | 3032cfa3=3.424, 81fe4578=3.424 | 12196.396 | 6.305 | 13.811 | |
phase pairs
| Rank | Prefill candidate | Decode candidate | Transition MB | Via host | Prefill ms | First decode ms | Total s |
|---:|---|---|---:|---|---:|---:|---:|
| 1 | single+3032cfa3 | tp2+3032cfa3+81fe4578 | 444.596 | no | 8.758 | 6.305 | 1.691 |
| 2 | single+81fe4578 | tp2+3032cfa3+81fe4578 | 444.596 | no | 8.704 | 6.305 | 1.709 |
| 3 | pipeline+3032cfa3@0..371/97fe4889@371..616/81fe4578@616..995 | tp2+3032cfa3+81fe4578 | 553.648 | yes | 95.032 | 6.305 | 1.833 |
| 4 | pipeline+97fe4889@0..239/81fe4578@239..616/3032cfa3@616..995 | tp2+3032cfa3+81fe4578 | 553.648 | yes | 97.948 | 6.305 | 1.836 |
| 5 | pipeline+97fe4889@0..239/3032cfa3@239..616/81fe4578@616..995 | tp2+3032cfa3+81fe4578 | 553.648 | yes | 98.103 | 6.305 | 1.837 |
| 6 | pipeline+81fe4578@0..371/97fe4889@371..616/3032cfa3@616..995 | tp2+3032cfa3+81fe4578 | 553.648 | yes | 98.285 | 6.305 | 1.837 |
| 7 | pipeline+3032cfa3@0..371/81fe4578@371..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 553.648 | yes | 100.319 | 6.305 | 1.839 |
| 8 | pipeline+81fe4578@0..371/3032cfa3@371..748/97fe4889@748..995 | tp2+3032cfa3+81fe4578 | 553.648 | yes | 103.726 | 6.305 | 1.842 |
| 9 | single+97fe4889 | tp2+3032cfa3+81fe4578 | 889.192 | yes | 18.479 | 6.305 | 1.903 |
| 10 | single+81fe4578 | single+81fe4578 | 0.000 | no | 8.704 | 8.704 | 2.238 |
best same-placement pair: rank 10 (single+81fe4578)
```

For prompt 512 the rank-1 split pair estimates 1.496 s versus 1.680 s for the
best same-placement pair at rank 11, a 0.184 s reduction; for prompt 32768 the
rank-1 split pair estimates 1.691 s versus 2.238 s for the best same-placement
pair at rank 10, a 0.547 s reduction. These are one-turn estimates from
committed topology costs, not execution measurements; the next turn's
transfer back to the prefill placement is outside this model.
