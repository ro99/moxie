# Hardware inventory — M0

Captured 2026-09-07 on the target machine. Device identity, capabilities and peer
access come from `cargo xtask probe`, which calls the CUDA driver API directly;
CPU, memory and storage from the OS. Bandwidth figures are from the bounded probe
described at the bottom and are **not** a planner cost model.

## Devices

Enumerated under `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Evidence identifies a GPU by
UUID; the ordinal is a convenience that is only stable because of that setting.

| Ordinal | Device | SM | UUID | Driver-reported VRAM | SMs | PCI bus | NUMA |
|---|---|---|---|---|---|---|---|
| 0 | RTX 5060 Ti | `sm_120` | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` | 15883 MiB | 36 | `0000:03:00.0` | 0 |
| 1 | RTX 3090 | `sm_86` | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` | 24123 MiB | 82 | `0000:82:00.0` | 1 |
| 2 | RTX 3090 | `sm_86` | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` | 24123 MiB | 82 | `0000:83:00.0` | 1 |

`cuDeviceTotalMem` reports less than the marketing capacity (15883 of 16384 MiB;
24123 of 24576 MiB). Document 01 requires planning "against separately measured
usable VRAM", so admission uses these numbers, and even they are before context
reservation, display use and allocator fragmentation.

Aggregate device memory is **62.6 GiB**. Every checkpoint currently on disk
exceeds it (see [checkpoint-inventory.md](checkpoint-inventory.md)), so
host-backed execution is the normal case here, not a fallback.

## PCIe link width — the two 3090s are not equivalent

| Ordinal | Bus | Negotiated width | Max width | Max gen |
|---|---|---|---|---|
| 0 | `03:00.0` | **x8** | x16 | 3 |
| 1 | `82:00.0` | **x8** | x16 | 3 |
| 2 | `83:00.0` | x16 | x16 | 3 |

This reproduces R12 exactly: two links at x8, one at x16, against an earlier
assumption that all were x16. It is not a detail. Device 1 and device 2 are the
same model of card with half the host bandwidth between them, which makes a
symmetric TP2 plan across the pair asymmetric in practice.

(`pcie.link.gen.current` idles at 1–2 and rises to 3 under traffic; the measured
rates below are consistent with Gen3.)

## Peer access — narrower than `nvidia-smi` suggests

`cuDeviceCanAccessPeer`, which is what actually governs whether the engine can
issue a peer transfer:

| from \ to | 0 | 1 | 2 |
|---|---|---|---|
| **0** (5060 Ti) | — | no | no |
| **1** (3090) | no | — | **yes** |
| **2** (3090) | no | **yes** | — |

`nvidia-smi topo -p2p rwnap` reports `OK` for all nine pairs, including both
5060 Ti paths. **The CUDA API disagrees, and the CUDA API is the authority**:
peer access is usable only within the 3090 pair, which are `PHB` peers on NUMA
node 1. The 5060 Ti is on NUMA node 0, reachable only as `SYS` across the socket
interconnect, and cross-socket peer access is not available.

Treat `nvidia-smi topo -p2p` as a statement about the topology's capability, not
about what CUDA will grant. Any capability probe in the engine must call
`cuDeviceCanAccessPeer` per ordered pair. See
[topology-p2p.md](topology-p2p.md) for the measured behaviour of the pair that
does have it, and why having it is not the same as wanting it.

## Host-to-device bandwidth

64 MiB pageable source, synchronous copy, 5 repetitions after 1 warm-up, one
device at a time.

| Ordinal | Width | Median GB/s | Min | Max |
|---|---|---|---|---|
| 0 | x8 | 6.37 | 6.35 | 6.41 |
| 1 | x8 | 5.83 | 5.81 | 5.86 |
| 2 | x16 | **11.50** | 11.48 | 11.57 |

The ~2x split tracks the link widths. NUMA placement of the pageable source was
not controlled, so link width is *consistent with* the difference rather than
proven to be its sole cause. Document 03 requires pinned versus pageable and
simultaneous-transfer behaviour to be measured before a planner uses them; this
probe does neither, and R12's warning stands — a pinned buffer measured *slower*
than pageable on this machine for one warm-expert workload.

## Host

| Property | Value |
|---|---|
| CPU | 2 x Intel Xeon E5-2680 v4, 14 cores / 28 threads each, 56 threads total |
| SIMD | AVX2, FMA, F16C, BMI2. **No AVX-512** (Broadwell) |
| RAM | 251 GiB total, ~246 GiB available |
| NUMA | 2 nodes, ~128 GiB each; node distance 21 (remote) versus 10 (local) |
| Swap | 1 GiB — effectively none |

No AVX-512 constrains the CPU expert path in document 03. CPU kernels operating
on "bounded tiles of canonical packed weights" get AVX2 at best, which changes
the crossover against a GPU grouped-expert plan.

Host RAM is split across two NUMA nodes, and the GPUs are split with it: the
3090 pair on node 1, the 5060 Ti on node 0. A 251 GiB host expert cache is not
one uniform tier, and document 03's requirement to reserve OS and application
headroom applies on top.

## Storage

| Mount | Device | Type | Size | Free |
|---|---|---|---|---|
| `/` | nvme0n1 (XPG GAMMIX S70 BLADE) | NVMe SSD | 1.9 T | 551 G |
| `/fast` | nvme1n1 (CT2000E100SSD8) | NVMe SSD | 1.8 T | 1.4 T |
| `/data` | sda (Netac SSD 2TB) | SATA SSD | 1.8 T | 286 G |
| `/archive` | sdb (ST2000DM008) | **spinning disk** | 1.8 T | 1.3 T |

`nvidia-fs` reports `broken`, so GPUDirect Storage is unavailable. Sustained read
bandwidth per mount is **not yet measured** — it is required before any
disk-spilling claim for the 1.5 TB checkpoint, where document 03 warns that disk,
host memory bus and PCIe can be three different bottlenecks.

## Not measured

Named here so none of it is mistaken for covered: pinned versus pageable
transfer, simultaneous transfers competing on one root complex, device-to-host
and peer bandwidth, per-mount sustained read, behaviour under thermal or power
load, and host pinned-memory limits. These belong to the M0 follow-up task and to
document 03's cost probes, not to this inventory.
