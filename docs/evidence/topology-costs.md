# Measured topology costs

Measured on **2026-09-25** with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, NVIDIA driver
`610.57.04`. The default probe configuration uses 16 KiB small copies, 64 MiB
bulk copies and five samples per figure. Host transfers use synchronous
pageable copies. Peer links appear only when the destination context is granted
access to the source device. `usable_bytes` is the driver's free-byte reading
after context acquisition and before probe buffers were allocated. Device-memory
bandwidth counts one read and one write per byte. `cargo xtask-cuda probe
--costs` ran twice on all three devices; this block uses run 2.

The dense BF16 linear rate uses `moxie_dense_linear_split_v1` from the dense
graph fatbin with BF16 `x [512, 4096]`, `weight [4096, 4096]` and
`output [512, 4096]`. The buffers contain finite zero values. Each device gets
one untimed launch followed by five event-timed launches; the reported rate is
the median of `2 * 512 * 4096 * 4096 / seconds / 1e12`. Per-device rates from
both runs:

| Device UUID | Run 1 TFLOP/s | Run 2 TFLOP/s |
|---|---:|---:|
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 0.109458067 | 0.109209710 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 0.054985993 | 0.054965098 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 0.054764684 | 0.054943313 |

The two linear-rate runs differ by at most 0.33% per device. These
are rates for Moxie's correctness-first kernel at this shape, not GPU peak
rates or a performance claim.

| Ordinal | Device | UUID | Peer-access row |
|---:|---|---|---|
| 0 | RTX 5060 Ti | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | no direct peer links |
| 1 | RTX 3090 | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | can access ordinal 2 |
| 2 | RTX 3090 | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | can access ordinal 1 |

| Kind | From | To | Small latency (us) | Isolated (GB/s) | Concurrent (GB/s) | Linear TFLOP/s | Usable bytes |
|---|---|---|---:|---:|---:|---:|---:|
| Device memory | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | same device | — | 385.896 | — | 0.109 | 16,512,122,880 |
| Device memory | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | same device | — | 794.376 | — | 0.055 | 25,017,319,424 |
| Device memory | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | same device | — | 799.220 | — | 0.055 | 25,017,319,424 |
| Link | host | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 9.180 | 5.692 | 1.938 | — | — |
| Link | host | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 8.514 | 6.542 | 2.661 | — | — |
| Link | host | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 9.178 | 6.568 | 5.590 | — | — |
| Link | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | host | 13.367 | 5.263 | 5.284 | — | — |
| Link | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 9.984 | 6.584 | 2.924 | — | — |
| Link | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | host | 13.190 | 5.528 | 12.128 | — | — |
| Link | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 10.240 | 5.201 | 5.205 | — | — |
| Link | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | host | 9.829 | 7.072 | 6.977 | — | — |

Observations from run 2:

- For `3032cfa3 → 81fe4578`, isolated direct peer bulk was **6.584 GB/s**
  and concurrent peer was **2.924 GB/s**. The sequential host path implies
  about **2.92 GB/s**, from D2H 5.263 GB/s and H2D 6.542 GB/s. In reverse,
  peer was **5.201 GB/s** versus about **2.80 GB/s** staged from D2H 5.528
  GB/s and H2D 5.692 GB/s.
- On the 5060 Ti, concurrent pageable H2D was **5.590 GB/s** versus **6.568
  GB/s** isolated, 14.9% lower. This is a machine and load measurement, not a
  fixed ratio.
- The measured host-to-5060 small copy was **16 KiB** and took **9.178 us**.
  The combined decode handoff in task 0072 is 240 bytes, so this probe extent
  is about 68 times larger; 9.178 us is not a 240-byte latency estimate.

Forward-peer repeatability (`GPU-3032cfa3` → `GPU-81fe4578`): isolated and
concurrent bulk figures were **6.585871 / 2.924931 GB/s** in run 1 and
**6.583886 / 2.923626 GB/s** in run 2.

The TOML written by run 2:

```toml
[[device]]
linear_tflops = 0.10920970990145071
memory_gbps = 385.8960307930676
usable_bytes = 16512122880
uuid = "GPU-97fe4889-4874-a378-198e-955d2e72c4a3"

[[device]]
linear_tflops = 0.05496509829925133
memory_gbps = 794.3757338566783
usable_bytes = 25017319424
uuid = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"

[[device]]
linear_tflops = 0.05494331292357321
memory_gbps = 799.2195252935354
usable_bytes = 25017319424
uuid = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"

[[link]]
bandwidth_gbps = 5.6915420353912864
concurrent_gbps = 1.9378956170706572
from = "host"
latency_us = 9.18
to = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"

[[link]]
bandwidth_gbps = 6.542413367611961
concurrent_gbps = 2.660640906157238
from = "host"
latency_us = 8.514
to = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"

[[link]]
bandwidth_gbps = 6.567847225627232
concurrent_gbps = 5.590088241757125
from = "host"
latency_us = 9.177999999999999
to = "GPU-97fe4889-4874-a378-198e-955d2e72c4a3"

[[link]]
bandwidth_gbps = 5.262697757390938
concurrent_gbps = 5.283974460121577
from = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"
latency_us = 13.366999999999999
to = "host"

[[link]]
bandwidth_gbps = 6.583885946540923
concurrent_gbps = 2.923626069723224
from = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"
latency_us = 9.983999654650688
to = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"

[[link]]
bandwidth_gbps = 5.528357469266063
concurrent_gbps = 12.128220466623608
from = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"
latency_us = 13.190000000000001
to = "host"

[[link]]
bandwidth_gbps = 5.200856968830295
concurrent_gbps = 5.205400944169026
from = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"
latency_us = 10.239999741315842
to = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"

[[link]]
bandwidth_gbps = 7.072129352921351
concurrent_gbps = 6.977076521142273
from = "GPU-97fe4889-4874-a378-198e-955d2e72c4a3"
latency_us = 9.829
to = "host"
```

## Pinned host transfers

Measured on **2026-09-25** with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, NVIDIA driver
`610.57.04`, using `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo xtask-cuda probe`
twice. Each row is the median of five event-timed 64 MiB samples per figure.
Issue time is host wall time around the async copy call. The overlap value is
for a pinned H2D copy and a calibrated smoke AXPY loop on separate streams;
D2H overlap is reported as zero because it was not measured.

Devices: `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` (RTX 5060 Ti),
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` (RTX 3090), and
`GPU-81fe4578-59b2-37c4-421e-287cdac78704` (RTX 3090).

Run 1:

| Device UUID | Direction | Pageable GB/s | Pinned GB/s | Pageable issue us | Pinned issue us | Overlap |
|---|---|---:|---:|---:|---:|---:|
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | host-to-device | 6.573 | 6.706 | 10047.768 | 2.411 | 0.980 |
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | device-to-host | 7.078 | 7.146 | 9474.839 | 2.304 | 0.000 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | host-to-device | 5.719 | 5.906 | 11478.332 | 2.569 | 0.998 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | device-to-host | 5.571 | 3.533 | 12019.610 | 2.762 | 0.000 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | host-to-device | 6.649 | 7.401 | 9929.823 | 2.834 | 1.000 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | device-to-host | 5.623 | 3.594 | 12356.976 | 2.495 | 0.000 |

Run 2:

| Device UUID | Direction | Pageable GB/s | Pinned GB/s | Pageable issue us | Pinned issue us | Overlap |
|---|---|---:|---:|---:|---:|---:|
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | host-to-device | 6.546 | 6.722 | 10057.616 | 2.793 | 0.979 |
| `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | device-to-host | 7.078 | 7.149 | 9481.528 | 2.384 | 0.000 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | host-to-device | 5.742 | 5.904 | 11455.804 | 2.732 | 0.999 |
| `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | device-to-host | 5.601 | 3.517 | 11971.980 | 2.479 | 0.000 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | host-to-device | 6.592 | 7.395 | 10015.204 | 2.685 | 1.000 |
| `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | device-to-host | 5.606 | 3.598 | 11952.012 | 2.567 | 0.000 |

Every nonzero figure in run 2 is within 25% of run 1; the largest relative
change is the 5060 Ti's H2D pinned issue time, **2.411 to 2.793 us (15.8%)**.
The values show higher pinned H2D bandwidth on all three GPUs (6.706 vs 6.573,
5.906 vs 5.719, and 7.401 vs 6.649 GB/s), while pinned D2H is slightly higher
on the 5060 Ti and lower on both 3090s (3.533 vs 5.571 and 3.594 vs 5.623
GB/s). Pageable issue calls take about **9.5–12.4 ms**; pinned calls take
about **2.3–2.8 us**. The H2D overlap values range from **0.979 to 1.000** in
both runs.

**Uncontrolled (coordinator, after review):** the probe controls neither the
pinned buffer's NUMA node nor the thread's CPU affinity. The 3090s sit on
NUMA node 1 and the 5060 Ti on node 0 (AGENTS.md), so the lower pinned D2H
rate on both 3090s may be a cross-node placement effect; these runs do not
establish that pinned D2H is inherently slower on a 3090. The practical
reading for staging design: pageable async copies block the host for the
whole transfer, pinned ones return in microseconds and overlap compute; the
bandwidth gain from pinning is small.
