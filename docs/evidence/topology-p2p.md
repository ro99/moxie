# PCIe peer-to-peer: enabled, and not a speedup by itself

Status: **verified enabled on this machine, 2026-09-07.** Performance figures below are **inherited,
not re-measured in this repository** — see Provenance. Nothing here is a benchmark result for Moxie.

## Verified here

`nvidia-smi topo -p2p rwnap` reports `OK` for every pair, in all five modes (read, write, nvlink,
atomics, prop):

```
      GPU0  GPU1  GPU2
GPU0   X    OK    OK
GPU1  OK     X    OK
GPU2  OK    OK     X
```

Including GPU0 (5060 Ti, NUMA 0) to the 3090 pair (NUMA 1) — that path is `SYS`, crossing the socket
interconnect, and it still reports P2P-capable.

## Why this is not the stock configuration

P2P over PCIe is not available on stock GeForce drivers. It is present here because the NVIDIA **open**
kernel module is built from patched source:

- Driver `610.43.02`, `NVIDIA UNIX Open Kernel Module`, locally built (`rodrigo@ubuntu2`, 2026-09-07).
- Patch source per prior project notes: `aikitoria/open-gpu-kernel-modules`, branch `610.43.02-p2p`,
  commit `14de73d818f98eba82f753132bfed8f6ed6314b7`.
- Both installed kernels carry the same module `srcversion` `5133AA53FEA92ECFF8E5016`
  (`6.8.0-100-generic`, currently running, and `6.8.0-138-generic`), so a reboot into either keeps P2P.

**This is a fragile dependency and M0 must pin it.** A driver package upgrade, a new kernel, or a DKMS
rebuild from stock source removes P2P silently — no error, just a changed `topo -p2p` matrix and a
plan whose cost model is now wrong. `dkms status` already reports "Differences between built and
installed modules" for both kernels, which is the expected bookkeeping for a manually installed
patched module but also the thing that a routine `dkms autoinstall` would resolve in the wrong
direction.

Two consequences for the engine, not just for the build:

1. The capability probe in document 03 must read P2P availability **at runtime per pair**, and the
   planner must treat it as a discovered capability, never a constant. A plan cached from a run with
   P2P is invalid on a boot without it — plan cache keys include capabilities (document 02).
2. `nvidia-fs/2.26.6` currently reports `broken`, so GPUDirect Storage is unavailable. Relevant to the
   terabyte/disk-spilling path in document 03; not required by any current milestone.

## Inherited measurements — do not treat as current

Measured during the legacy Strata project on `0000:82:00.0` <-> `0000:83:00.0` (the 3090 pair, `PHB`,
cross-root-complex; bus 82 is Gen3 **x8**, bus 83 is x16 — the R12 correction). Re-measure before any
planner cost model depends on these numbers.

| Metric | Host-staged | Direct P2P |
|---|---|---|
| GPU–GPU latency | 15.33 us | **1.37 us** |
| Unidirectional | 4.08 / 3.98 GB/s | 6.59 / 5.22 GB/s |
| Bidirectional | 5.72 GB/s | 5.85 GB/s (flat) |

**The result that matters: enabling P2P does not make collectives faster here.** NCCL refuses P2P
across `PHB` unless `NCCL_P2P_LEVEL=SYS`. Forced on, the all-reduce crossover is around **512 KB**:

- Below it, P2P is 3–20x faster and far more consistent (~20 us flat, versus 130–450 us of jitter).
- Above it, P2P is **2.6–4x slower** (at 128 MB: 80 ms staged versus 283 ms P2P).

Mechanism: NCCL's P2P path issues device-side loads/stores through the peer mapping, which is
inefficient across a root complex. `cudaMemcpyPeer` uses the copy engine with large DMA bursts and
does beat staging. So peer **DMA** is fast here; peer **kernel access** is not. The two must be costed
separately.

## What this obliges

- **M5.** "P2P is enabled" is not an argument for TP2 on the 3090 pair. Document 04 already says `auto`
  must not select TP merely because cards exist; this is the concrete reason. Rank the TP2 candidate on
  measured collective time at the real message size for this workload, and record which transport
  each measurement used.
- **M0.** Pin driver version, module source commit, and kernel(s) in the dependency manifest. Record
  the `topo -p2p` matrix as a startup capability probe result, not as a documented constant.
- Never set `NCCL_P2P_LEVEL=SYS` globally. It is a per-message-size decision, and above ~512 KB it is
  a large regression.

## Provenance

- Verified in this repository 2026-09-07: the `topo -p2p` matrix, driver version, open-module identity,
  build timestamps, `srcversion` equality across both installed kernels, `nvidia-fs` state.
- Inherited from legacy Strata project notes and **not** re-run here: every figure in the measurements
  table, the NCCL crossover, and the patch commit identity. Their original context limitations apply;
  document 08's rule holds — historical measurements are not predictions.
