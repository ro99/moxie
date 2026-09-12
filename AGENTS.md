# Moxie — mandatory agent entry point

This repository builds one NVIDIA inference engine for one interactive user, including models larger than VRAM and large context. It does not build independent engines per model.

## Active assignment

**M1.4 complete; M1.5 active.** The owner accepted [task 0012](docs/tasks/0012-m1-selected-bf16-device-chain.md)
on 2026-09-10 after independent verification of implementation `6305f9d` and corrections
`eaf8846` / `1138a2a` (reviewed record `ce7548d`). The owner accepted
[task 0013](docs/tasks/0013-m1-appendable-paged-state.md) on 2026-09-10 after
independent review through `0f39cf5`; its contract `a80ff2c` preceded implementation
`c14e32a`, and correction `c9a4b33` fixed the sole reported issue.

The owner accepted [task 0014](docs/tasks/0014-m1-transactional-sampler-history.md),
M1.4 sampler history and deterministic greedy/temperature distribution, implemented
after contract `4054dd7`, on 2026-09-11 following independent review of `23a7a40`
and `1e6f253`. The review's layout-chronology documentation finding is qualified
in the task record.

The owner accepted [task 0015](docs/tasks/0015-m1-generation-service-diagnostic-cli.md),
the final M1.4 generation-service/minimal diagnostic-CLI integration, on 2026-09-11.
Contract
`2fbacec` precedes implementation `f7cff56`. It uses the accepted shared interpreter,
paged transactions, memory authority and sampler, with an explicit host-reference
profile. Correction `5990f2a` addresses the independent review's allocation-failure,
executed-row-shape and immutable paged-program findings. Re-review found one
remaining direct-API rollback gap; correction `7a08adf` moves all paged preparation
under the transaction abort guard. Final independent review through `3f13784` found
no remaining blocker. This acceptance closes M1.4 within its synthetic
host-reference bounds.

The owner accepted [task 0016](docs/tasks/0016-m1-gemma-reduced-graph.md),
which opens M1.5, on 2026-09-12 after independent re-review found no remaining
blocking correctness or architecture issue in its declared synthetic scope.
Contract `1199267` follows the artifact-inspection evidence `1e1867c` and
precedes implementation `c7dd153`; corrections `4e7ad56` address the review's
two P1 numerical boundary omissions — a dropped BF16 rounding in the scaled
residual and in the logit softcap — plus an unchecked extent product and an
incomplete acceptance claim. A following commit isolates the key/value overflow
guard the re-review found untested.

**This acceptance does not close M1.5.** It covers the reduced synthetic graph
only. It makes seven family-mathematics parameters explicit
([ADR 0012](docs/decisions/adr/0012-explicit-family-operation-parameters.md)) and
adds `moxie-models` with a `gemma4` module — one crate with a module per family,
at the owner's direction during implementation
([ADR 0013](docs/decisions/adr/0013-one-model-crate-with-family-modules.md)).

The owner selected **M4's paged state schema** as
[task 0017](docs/tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md) from
that handover's three candidates. It is **implemented and awaiting owner
review**: per-layer key/value geometry and per-layer retention, one ring per
layer, reclamation by overwrite, and bounded tentative-undo headroom
([ADR 0014](docs/decisions/adr/0014-bounded-tentative-undo-headroom.md)). Its
numerical claim is exact: reclaiming outside a layer's window changes no output
bit. **It closes neither M4 nor M1.5** — device paged attention, COW forks,
host-backed page streaming and recurrent state are still M4's, and the artifact
still cannot execute. It does clear two of the reduced graph's four reductions:
sliding and global layers now compose at their own widths and sliding layers
keep only what they can see. Contract `b748536` precedes implementation.

[Task 0018](docs/tasks/0018-m3-compressed-tensors-int8-importer.md) is
**implemented and awaiting owner review**: a bounded safetensors reader and a
compressed-tensors `pack-quantized` importer that turns the Gemma 4 artifact's
real INT8 weights into the accepted canonical `AffineTensor`, verified on three
of its modules read-only. `moxie-format` takes `serde_json` for the header parse
([ADR 0015](docs/decisions/adr/0015-serde-json-for-safetensors-headers.md)).
Contract `db9529e` precedes implementation.

**It still executes no checkpoint, and an import is not support.** There is no
W8A16 path, no repacker and no canonical manifest write; all three are M3's. The
lane order inside a packed word is taken from the pinned reader and **cannot**
be verified against this artifact, so no quality claim follows from a successful
import — that needs paired output against the released model, which is O2. The
reduced graph remains a synthetic contract fixture. M11 owns vision. See the
[active handover](docs/handovers/2026-09-12-task0018-compressed-tensors-import.md),
whose next bounded task is the shared W8A16 execution path.

## Read before editing

Read [README.md](README.md), [product boundaries](docs/spec/01-product-and-decisions.md), [common API](docs/spec/02-architecture-and-common-api.md), [agent playbooks](docs/spec/09-agent-playbooks.md), and the assigned milestone in [the roadmap](docs/spec/06-implementation-roadmap.md). Then read the relevant memory, attention/state, sampling/application and validation contracts linked there, and their actual legacy source references. These documents are normative; this short file is their discovery entry point.

Confirm writable repository root/branch/dirty state and read-only legacy snapshot before work. Preserve unrelated changes. Root policies apply to all subdirectories; local instructions may narrow but not silently weaken them. Keep shared policies, ADRs, task contracts and handovers version-controlled.

Reference documents 01–09 are kept local and untracked; every living record is tracked. [docs/README.md](docs/README.md) is the placement contract: which directory each record type belongs in and which template it uses. Open owner gates are in [the owner-gate register](docs/decisions/owner-gates.md), not only in document 01. Amend a reference document only through an ADR, never by editing it to match what was built.

## Non-negotiable ownership

- Model definitions contain metadata, tensor roles and semantic graph composition. No model-owned execution loop, allocator/cache, CUDA/transfer code, scheduling, KV/branch lifecycle, sampling, speculation or collectives.
- Missing model mathematics becomes a shared semantic operation with an independent oracle and shape/precision/state/partition contract. Then the model consumes it. No opaque whole-model custom op or arbitrary backend callback.
- One real memory authority admits all persistent/transient/branch/draft resources; one state-transaction mechanism handles prefill, continuation, entropy and speculation. Event-retained leases govern asynchronous buffer reuse.
- Shared engine crates cannot import concrete models. Composition roots inject the registry. Kernel dispatch is by semantic capability/shape/layout/hardware, not model name. Architecture checks and affected-consumer tests are required.
- Rust owns the application/control plane. Reusing audited C++/CUDA kernels behind the common ABI is encouraged. Wrapping seven legacy generators is not a Rust rewrite of the engine.

## Product gates

- NVIDIA only; benchmark the two 3090s plus 5060 Ti and 251 GB host RAM. Never assume their memory is one allocation or their links equal.
- Local checkpoint roots on the benchmark machine are `/models` and `/fast/models` (see [artifact-roots](docs/evidence/artifact-roots.md)). Inspect exact revisions and shard completeness there before a model task; these roots are read-only inputs, not proof of catalog approval or quality, and do not authorize copying, conversion, deletion or requantization.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` always, to request PCI-bus ordering within the visible device set; it is set in `.cargo/config.toml` for cargo-launched processes and must be set explicitly for anything else. Ordinals are diagnostics only — plans, manifests and results identify a GPU by UUID. On this machine ordinal 0 is the 5060 Ti on NUMA node 0, and ordinals 1–2 are the 3090 pair on NUMA node 1; do not assume ordinal 0 is a 3090 or that the three links are equivalent.
- INT4, INT8 and BF16 are the initial canonical precision family; NVFP4/FP8 are deferred. Follow [ADR 0003](docs/decisions/adr/0003-int4-int8-bf16-weight-family.md). W4A16 is an execution profile, AWQ/AutoRound are methods, and serialization belongs in shared importers. Preserve group size, zero points, scale dtype and logical column identity; no method-specific model runtime. No sub-four-bit weights or optional sub-16-bit cache. Intrinsic low-bit state ambiguity requires O4 resolution.
- Large actual context: 32,768 minimum useful target; larger tiers need positional, admission and execution evidence. Good prefill AND decode are simultaneous goals. Do not shrink context or hide one phase to pass.
- Quality is against the released model, not self bit-reproducibility. Numerical/quality gates must be declared before changing them. Never loosen a test to make an optimization pass.
- One active interactive generation; OpenAI-compatible API and CLI share the engine/sampler. Flash-style attention, TP, PP, speculation, common samplers and future entropy have shared contracts and truthful capability matrices.
- Do not silently drop samplers/features, lower precision/context, or run an unbounded fallback. `auto` reports selection; `required` errors when unsupported or inadmissible.

## Work and completion

Use [TASK.md](docs/spec/templates/TASK.md) for a bounded assignment; use the model and handover templates when applicable. Read [the source map](docs/spec/08-strata-reference-map.md) and pinned upstream references before inventing replacements. Extend existing shared ownership instead of copying a cache/runtime with small model differences.

Every task names the mechanism fixed, shared owner, consumers, source/oracle, resource and cancellation contract, acceptance tests and deletion/expiry plan. Run architecture/host tests plus relevant real GPU, topology, state, quality and paired performance gates. Report failed/skipped/unmeasured separately. No required feature is complete because it has a stub or flag.

Preserve negative experiments and remove superseded rewrite paths after replacement gates pass. Do not delete the legacy checkout. No bulk checkpoint conversion/download, system-driver change, network exposure or other material operational expansion without scoped authorization.

Open owner decisions O1–O7 are in document 01. Ask in a batch before dependent conclusions or work. Resolve normal technical choices through source, tests and measured ADRs; do not ask the owner to design kernels. A task cannot override an owner requirement through local inference.

If a requirement cannot be met, stop at its defined gate and report evidence plus the smallest decision needed. Do not manufacture completion with a private model path, hidden regression, test weakening or unsupported performance claim.
