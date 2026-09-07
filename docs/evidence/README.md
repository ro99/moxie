# Evidence

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
