# Evidence

- `topology-p2p.md` — PCIe P2P is enabled on all pairs via a **patched open kernel module**, not the
  stock driver. Read it before assuming anything about multi-GPU transport: P2P cuts peer latency ~11x
  but makes NCCL bulk collectives 2.6–4x *slower* above roughly 512 KB.
- `support-matrix.md` — capability claims, each linked to a passing gate ID. Use
  [SUPPORT-MATRIX.md](../spec/templates/SUPPORT-MATRIX.md). Initial status of every row is
  NOT IMPLEMENTED, never supported.
- `benchmarks/` — benchmark manifests and machine-readable result summaries. A manifest pins
  checkpoint/artifact checksums, the actual tokenized prompt, committed and requested context,
  sampler, topology, hardware UUIDs, versions and executable hash.
- `experiments/` — accepted and **rejected** experiment conclusions, each with hypothesis, mechanism,
  paired prefill/decode baselines, actual context, and exact scope of the result. Rejected results
  are kept so a later agent does not repeat the campaign (R10, R15, R27).

Large raw traces and profiler output stay outside git. Cite them by content hash, access location and
retention policy.
