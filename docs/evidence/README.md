# Evidence

- `topology-p2p.md` — patched-driver topology and measurements. The CUDA peer-access API allows the 3090 pair, not every pair despite permissive nvidia-smi output. Read the actual measured pair/message size before generalizing latency or collective results.
- `quantization-candidates.md` — pinned public configurations for the owner's ten INT4/INT8 candidates; metadata evidence only, no weights or quality claims.
- `../tasks/0002-m0-review-and-integer-transition.md` — M0 review, reproduced checks, architecture
  bypasses and correction gates, followed by the results of two correction passes and the exact
  commands behind them. The second pass's findings are the ones to read first: each was reproduced
  before being fixed, and each says what the fix does **not** establish.
- `support-matrix.md` — capability claims and the gate IDs behind them. Every row that mentions a
  checkpoint, a kernel or a context length is NOT IMPLEMENTED or unmeasured, and says so.
- `toolchain.md` — pinned toolchain, the recorded build identity (nvcc, host compiler, fatbin
  digests), and the host/device lane split.
- `specification-version.md` — digests of the ten normative reference documents. They are kept local
  and untracked, so this is how a fresh clone learns whether it has them. `cargo xtask spec-check`
  verifies it; it reads and hashes only.
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
