# Benchmark manifest and result schema — M0

Document 07 requires a manifest per case and a machine-readable result. This is
the schema. **No benchmark has been run**, and this file makes no measurement
claim; `cargo xtask bench` is not implemented and arrives with M6.

## Why a manifest at all

R19 is the reason a manifest is not bureaucracy. A "prefill width 32,768"
experiment was run with a 619-token prompt; the number that looked like context
was a scheduler's maximum row width. A manifest that records requested context,
admitted context and **actual tokenized prompt length** as three separate fields
makes that mistake visible in the record rather than in a conclusion.

## Manifest

```json
{
  "case_id": "glm53-int4-32k-decode",
  "created": "2026-09-07",
  "artifact": {
    "path": "...",
    "checksum": "sha256:...",
    "precision_profile": "affine-int4-v1",
    "quantizer": "<pinned-source-method-and-version>",
    "serialization": "<pinned-compressed-tensors-or-autoround-packing>",
    "normalization": "value-preserving-repack | precision-conversion",
    "integer_contract": {
      "bits": 4,
      "group_size": 128,
      "zero_point_mode": "implicit-zero",
      "scale_dtype": "<from-tensor-header>",
      "group_index_hash": null
    },
    "source_checkpoint": "...",
    "source_checksum": "sha256:..."
  },
  "execution": { "profile": "w4a16", "activation_dtype": "bf16", "accumulator_dtype": "f32" },
  "graph": { "model_graph_version": "...", "state_schema_version": "..." },
  "workload": {
    "prompt_file": "...",
    "prompt_tokens_actual": 32768,
    "committed_starting_context": 0,
    "requested_context": 32768,
    "admitted_context": 32768,
    "output_tokens": 128,
    "phase": "first_prefill | later_chunk | uneven_tail | prefix_hit | continuation | decode"
  },
  "sampler": { "profile": "greedy", "seed": 0, "future_entropy": "off" },
  "speculation": { "mode": "off | auto | required", "proposer": null },
  "cache": { "dtype": "bf16" },
  "topology": {
    "devices": ["GPU-...", "GPU-..."],
    "plan": "single | tp2 | pp2 | tp2+pp1 | expert_partition",
    "peer_transport": "none | p2p | host_staged"
  },
  "environment": {
    "executable_hash": "sha256:...",
    "rust": "1.97.1", "cuda": "13.0.88", "driver": "610.43.02",
    "clocks_locked": false, "power_limit_w": null, "thermal_state": "..."
  },
  "protocol": { "warmups": 1, "repetitions": 5, "isolated": true }
}
```

Three separate context fields, deliberately. `requested_context` is what was
asked for, `admitted_context` what the planner reserved, `prompt_tokens_actual`
what the model actually attended over. A result where the third is far below the
first two is not a long-context result, whatever the other two say.

`devices` are UUIDs, never ordinals (document 07).

## Result

```json
{
  "case_id": "...", "manifest_checksum": "sha256:...",
  "run": { "started": "...", "host": "...", "repetitions_completed": 5 },
  "timing": {
    "time_to_first_token_ms": { "median": 0.0, "p10": 0.0, "p90": 0.0 },
    "prefill_tokens_per_s": { "median": 0.0, "min": 0.0, "max": 0.0 },
    "decode_tokens_per_s": { "median": 0.0, "min": 0.0, "max": 0.0 },
    "total_turn_ms": { "median": 0.0 },
    "load_and_prepare_ms": 0.0,
    "prefill_includes": ["tokenization", "file_reads", "plan_compilation"],
    "end_to_end_vs_phase_only": "both reported separately"
  },
  "counters": {
    "expert_rows": 0, "unique_experts": 0,
    "cache_demand_hits": 0, "cache_demand_misses": 0,
    "prefetch_hits": 0, "prefetch_wasted_bytes": 0,
    "disk_bytes": 0, "disk_read_ms": 0.0,
    "h2d_bytes": 0, "d2h_bytes": 0, "p2p_bytes": 0, "collective_bytes": 0,
    "attention_state_bytes_scanned": 0,
    "launch_count": 0, "sync_wait_ms": 0.0,
    "peak_device_bytes": 0, "peak_host_bytes": 0,
    "speculative_wasted_tokens": 0, "accepted_prefix_histogram": []
  },
  "outcome": "passed | failed | unmeasured | unsupported",
  "notes": "..."
}
```

## Rules that are easy to break

- **Decode throughput excludes rejected proposals from its numerator but includes
  their time.** A speculative result that counts draft tokens is not a decode
  rate.
- **Usage counts committed tokens only** — not draft proposals, rejected
  verification rows or entropy branches.
- **Repetition count is pre-registered.** Document 07: "do not choose repetition
  count after a favorable result." For expensive cases, pre-register fewer and
  explain the uncertainty *before* running.
- **Both phases, always.** A prefill win with an unreported decode regression is
  not a result. Document 07's default-selection gate needs the pair.
- **Isolated.** No competing agent jobs on the box. This machine has run
  concurrent agents before; a benchmark run must not.
- **Teacher-forced runs are diagnostics.** They isolate performance by fixing
  routes, and they hide changed route distributions. The headline interactive
  number is free-running.

## Baseline status

Document 06 M0.7 asks for legacy baselines "where available". **None recorded.**
The legacy build directories (`build-ci/`, `build-clean-ci/`) exist in the legacy
checkout with binaries of unverified provenance, and the two checkpoints usable
with them are large. Running a legacy baseline would need a confirmed binary
identity, an unloaded machine and owner time; per doc 06 it is therefore
explicitly **unmeasured**, not zero and not assumed.

Establish source-identified baselines for representative resident and streamed integer artifacts at 32,768 actual tokens before changing performance defaults. Pair prefill and decode; no single GLM/NVFP4 comparison is the only relevant baseline. Original-model quality and quantized-source import parity are separate from runtime performance. Real-model runs need the applicable storage/time authorization and an isolated machine.
