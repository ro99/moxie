# Measured topology costs

Measured on **2026-09-24** with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, NVIDIA driver
`610.57.04`. The command used the default probe configuration: 16 KiB small
copies, 64 MiB bulk copies, five samples per figure. Host transfers used
synchronous pageable copies. Peer links appear only when the destination
context is granted access to the source device. `usable_bytes` is the driver’s
free-byte reading after context acquisition and before probe buffers were
allocated. Device-memory bandwidth counts one read and one write per byte.

| Ordinal | Device | UUID | Peer-access row |
|---:|---|---|---|
| 0 | RTX 5060 Ti | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | no direct peer links |
| 1 | RTX 3090 | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | can access ordinal 2 |
| 2 | RTX 3090 | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | can access ordinal 1 |

| Kind | From | To | Small latency (us) | Isolated (GB/s) | Concurrent (GB/s) | Usable bytes |
|---|---|---|---:|---:|---:|---:|
| Device memory | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | same device | — | 383.462 | — | 16,512,122,880 |
| Device memory | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | same device | — | 809.086 | — | 25,017,319,424 |
| Device memory | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | same device | — | 814.112 | — | 25,017,319,424 |
| Link | host | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 9.196 | 5.734 | 4.534 | — |
| Link | host | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 8.765 | 6.711 | 7.313 | — |
| Link | host | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 9.025 | 6.586 | 5.053 | — |
| Link | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | host | 13.181 | 5.531 | 3.586 | — |
| Link | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 9.216 | 6.584 | 2.925 | — |
| Link | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | host | 13.320 | 5.566 | 2.808 | — |
| Link | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 9.312 | 5.202 | 5.205 | — |
| Link | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | host | 9.532 | 7.060 | 6.879 | — |

Observations from the primary run:

- For `3032cfa3 → 81fe4578`, isolated direct peer bulk was **6.584 GB/s** and
  concurrent peer was **2.925 GB/s**. The sequential host path implies about
  **3.03 GB/s**, from D2H 5.531 GB/s and H2D 6.711 GB/s. In reverse, peer was
  **5.202 GB/s** versus about **2.82 GB/s** staged from D2H 5.566 GB/s and
  H2D 5.734 GB/s.
- On the 5060 Ti, concurrent pageable H2D was **5.053 GB/s** versus **6.586
  GB/s** isolated, 23.3% lower. This is a machine and load measurement, not a
  fixed ratio.
- The measured host-to-5060 small copy was **16 KiB** and took **9.025 us**.
  The combined decode handoff in task 0072 is 240 bytes, so this probe extent
  is about 68 times larger; 9.025 us is not a 240-byte latency estimate.

Forward-peer stability check (`GPU-3032cfa3` → `GPU-81fe4578`): isolated and
concurrent bulk figures were **6.583886 / 2.924547 GB/s** in the primary run,
**6.583886 / 2.924808 GB/s** in repeat 1, and **6.585354 / 2.923887 GB/s** in
repeat 2. The isolated value is about **2.25×** the concurrent value in all
three runs, exceeding the 2× check. This is consistent with bidirectional contention (the concurrent phase runs both directions of the pair at once); the one-sided magnitude (the reverse direction stays about 5.2 GB/s) is unexplained.

The TOML written by the primary run:

```toml
[[device]]
memory_gbps = 383.4616954950226
usable_bytes = 16512122880
uuid = "GPU-97fe4889-4874-a378-198e-955d2e72c4a3"

[[device]]
memory_gbps = 809.0864365584532
usable_bytes = 25017319424
uuid = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"

[[device]]
memory_gbps = 814.1117831772806
usable_bytes = 25017319424
uuid = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"

[[link]]
bandwidth_gbps = 5.733936077535134
concurrent_gbps = 4.533633734156757
from = "host"
latency_us = 9.196
to = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"

[[link]]
bandwidth_gbps = 6.710539465109654
concurrent_gbps = 7.31294268924916
from = "host"
latency_us = 8.765
to = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"

[[link]]
bandwidth_gbps = 6.586107974154069
concurrent_gbps = 5.053083803699995
from = "host"
latency_us = 9.025
to = "GPU-97fe4889-4874-a378-198e-955d2e72c4a3"

[[link]]
bandwidth_gbps = 5.530575344506862
concurrent_gbps = 3.5855128297120786
from = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"
latency_us = 13.181
to = "host"

[[link]]
bandwidth_gbps = 6.583885946540923
concurrent_gbps = 2.924547334485995
from = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"
latency_us = 9.216000325977802
to = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"

[[link]]
bandwidth_gbps = 5.566225519307523
concurrent_gbps = 2.8083857952948805
from = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"
latency_us = 13.32
to = "host"

[[link]]
bandwidth_gbps = 5.201747364256623
concurrent_gbps = 5.205414036246958
from = "GPU-81fe4578-59b2-37c4-421e-287cdac78704"
latency_us = 9.312000125646591
to = "GPU-3032cfa3-19df-028f-5ebd-43314911e0b9"

[[link]]
bandwidth_gbps = 7.060340048818275
concurrent_gbps = 6.879155355401879
from = "GPU-97fe4889-4874-a378-198e-955d2e72c4a3"
latency_us = 9.532
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
