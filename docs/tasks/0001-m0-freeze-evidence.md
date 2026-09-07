# Task 0001 — M0: freeze evidence, decisions and build boundaries

Status: **accepted**

## Identity and authority

- Task ID / milestone / owner: 0001 / M0 / implementation agent, owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Required documents: 01–09, AGENTS.md
- Owner gates resolved before start: none. O1–O7 all OPEN.

## Bounded deliverable

Document 06 M0.1–M0.7 and its exit gate. Explicitly **not** in scope: any
model-specific runtime, any checkpoint conversion or download, any performance
claim.

## Result

### M0.1 — repository and legacy inventory · **done**

Moxie root verified writable, `main`, clean. Legacy checkout verified at the
frozen commit, **not modified**. Its working tree carries two untracked
directories, left alone: `.pi/worktrees/` and `tests/p2p/p2p_probe.cu` (24 KB,
the probe behind the inherited P2P measurements — a reusable asset). Ignored
build trees `build-ci/` and `build-clean-ci/` contain binaries of unverified
provenance; document 08 warns that ignored experiment records may not exist in a
clone, and no conclusion here depends on them.

The specification pack is under `docs/spec/` (kept local, per owner direction)
with living records tracked; see `docs/README.md`.

### M0.2 — hardware, topology, bandwidth · **done, with named gaps**

[hardware-inventory.md](../evidence/hardware-inventory.md). Device identity, SM,
UUID, usable VRAM, SM counts, PCI bus, peer access and H2D bandwidth captured by
`cargo xtask probe`, which calls the driver API directly. CPU, NUMA, RAM and
storage from the OS.

Two findings that change planning assumptions:

- **Peer access is narrower than `nvidia-smi` reports.** `cuDeviceCanAccessPeer`
  grants it only between the two 3090s. `nvidia-smi topo -p2p rwnap` says `OK`
  for all nine pairs. The CUDA API governs. [topology-p2p.md](../evidence/topology-p2p.md)
  was corrected.
- **The two 3090s do not have equal links.** Bus 82 negotiated x8, bus 83 x16,
  measured 5.83 versus 11.50 GB/s H2D. R12 reproduced independently.

Not measured, named as gaps: pinned versus pageable, simultaneous transfers,
D2H and peer bandwidth, per-mount sustained read, host pinned-memory limits,
behaviour under thermal/power load.

### M0.3 — checkpoints, licences, feasible context · **done; blocks on O1/O5**

[checkpoint-inventory.md](../evidence/checkpoint-inventory.md). **Five of the
seven roadmap families have no artifact on this machine** — Gemma, Laguna,
Inkling, GLM-5.2, DeepSeek. M1's proposed dense slice (Gemma) and M2's proposed
first MoE (Laguna) therefore have no checkpoint. Both can still proceed against
reduced synthetic graphs, which M1.5 provides for.

Present: Kimi K3 (1.5 TB, **MXFP4** routed experts — not a canonical family),
GLM-5.3-NVFP4 (433 GB, ModelOpt, **W4A4** not weight-only), GLM-5.3-Flash NVFP4
(178 GB, compressed-tensors, abliterated derivative, inside the legacy checkout).

Aggregate device memory is 62.6 GiB; every checkpoint exceeds it. Host-backed
execution is the ordinary path for this catalog.

O1–O7 register created: [owner-gates.md](../decisions/owner-gates.md).

### M0.4 — toolchain and CUDA linkage · **done**

[toolchain.md](../evidence/toolchain.md), [ADR 0001](../decisions/adr/0001-hand-written-cuda-ffi.md),
[ADR 0002](../decisions/adr/0002-rust-edition-2024.md).

Rust 1.97.1 / edition 2024 / resolver 3 pinned. CUDA 13.0 pinned, targets
`sm_86` and `sm_120`. `moxie-cuda` is hand-written driver-API FFI; error
propagation is mapped by numeric code and unit-tested without a device. One
third-party crate in the whole workspace (`toml`, `xtask` only). No Python in the
build or product.

### M0.5 — workspace, arch-check, negative fixtures, API drafts · **done**

Seven crates plus `xtask`. `cargo xtask arch-check` enforces document 02's
ownership table: dependency direction, no shared crate importing a concrete
model, no build script under a model crate, no forbidden constructs in model
source. **Five negative fixtures, all rejected** — `inverted-dependency`,
`model-depends-on-cuda`, `model-has-build-script`, `model-has-ffi`,
`shared-imports-model`.

API drafts, each with tests that assert refusal rather than placeholder success:

- `moxie-types` — typed errors, checked symbolic `Dim` (overflow and non-exact
  division are errors), precision rules (no weight < 4 bits, no cache < 16 bits),
  `off`/`auto`/`required`, `ChunkId` and `LayoutId` as distinct types (R17).
- `moxie-graph` — closed operation catalogue; no `Custom` variant, by design.
  An operation without a host oracle cannot be lowered.
- `moxie-state` — the two-frontier rule from document 04: committed history
  versus materialized state, with rejection-at-every-depth tested, and recurrent
  state marked non-truncatable (R20).
- `moxie-model-api` — `ModelDefinition` as data and graph composition; no
  execute/forward/sample method exists.
- `moxie-format` — see M0.6.

Host CI lane written (`.github/workflows/ci.yml`). **Unverified**: it needs
`nvcc` for `moxie-kernels` and no runner is configured. Recorded in
[toolchain.md](../evidence/toolchain.md) as a known gap, not as passing.

### M0.6 — source-linked oracle corpus · **partly done**

Done, with exhaustive tables:

- **BF16** round-to-nearest-even. 28 tests, including every one of the 65,536
  bf16 patterns round-tripping, exact ties going to even, and a NaN that
  truncation would turn into an infinity.
- **NVFP4** E2M1 (all 16 codes, derived from the format rather than copied from
  the table) and E4M3FN (all 256 codes; exactly two are NaN). The full
  reconstruction equation, nibble order, incomplete-group padding, and rejection
  of NaN/negative/non-finite scales.
- **INT8 v1** symmetric weight-only, `-128` rejected, ties to even.
- **Scale conventions.** [Experiment 0001](../evidence/experiments/0001-nvfp4-scale-conventions.md)
  confirms R16 against real weights: ModelOpt multiplies, compressed-tensors
  divides, verified on both checkpoints.

Not done: routed-expert, attention-mask, recurrent-update, sampling-distribution,
tokenizer/template and protocol-frame fixtures. These need either a graph that
does not exist yet or export permission from a licensed checkpoint, which
document 06 M0.6 requires to be verified first. **Carried to the next task.**

### M0.7 — benchmark schema and legacy baselines · **schema done, baselines unmeasured**

[schema.md](../evidence/benchmarks/schema.md). Manifest separates
`requested_context`, `admitted_context` and `prompt_tokens_actual`, which is R19
encoded in the record format.

Legacy baselines: **explicitly unmeasured**. Needs a verified binary identity,
an unloaded machine and O5.

## Exit gate

| Gate | Status |
|---|---|
| Reproducible workspace build | **met** — 69 tests, clippy clean under `-D warnings`, fmt clean |
| One real CUDA launch on each installed architecture | **met** — `sm_86` (devices 1, 2) and `sm_120` (device 0), all cases pass |
| Dependency violations demonstrably rejected | **met** — 5 negative fixtures, each rejected with the expected rule |
| Reference manifest and owner-gate register committed | **met** — `docs/evidence/`, `docs/decisions/owner-gates.md` |
| No production quality or speed claim | **met** — none made |

## Owner questions, batched

Per AGENTS.md these are asked together, before dependent work.

1. **O1, bring-up order.** Five of seven families have no checkpoint. Proceed
   with synthetic graphs for M1/M2 as document 06 M1.5 allows, or acquire a
   Gemma/Laguna checkpoint first? (Acquisition also touches O5.)
2. **O5, Kimi K3 and MXFP4.** Its routed experts are MXFP4, which is not a
   canonical family. Add MXFP4 as an import format producing canonical NVFP4
   (double quantisation — needs O2 evidence), add it as a fourth canonical
   family, or seek a higher-precision original?
3. **O5, storage.** Converting GLM-5.3-NVFP4 (433 G) fits on `/fast`. Converting
   Kimi K3 (1.5 T) does not fit anywhere as a second copy. What disk budget and
   which paths are authorised?
4. **O2, W4A4.** GLM-5.3-NVFP4 ships activation scales. Importing weight-only
   discards them, giving a different numerical contract from the one its
   publisher validated. Acceptable pending measurement, or is the W4A4 profile
   wanted?
5. **Quality baseline.** The GLM-5.3-Flash artifact is an *abliterated*
   derivative, so it cannot serve as a document 07 quality reference. Is a
   released reference model available for any family here?
6. **CI.** Is a self-hosted runner available, or should the host lane be made to
   run without `nvcc`?

## Next bounded task

**Task 0002 — M1 vertical slice, part 1: reference interpreter and BF16
operations.** Owning component `moxie-graph` plus a new host reference
interpreter. Deliverable: `Linear`, `RmsNorm`, `SwiGlu`, `Rope` and a small exact
attention, each with an independent FP32/FP64 oracle and a declared error metric
*before* any optimisation, per document 07. No CUDA, no checkpoint.

Stop condition: if it needs a decision from the batch above, stop and report
rather than assuming an answer.
