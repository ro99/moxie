# Moxie — mandatory agent entry point

This repository builds one NVIDIA inference engine for one interactive user, including models larger than VRAM and large context. It does not build independent engines per model.

## What this file is

The policy entry point, and the only always-loaded document in this repository.
Every line costs on every turn, so it holds **what binds the work**: current
assignment, ownership rules, product gates, completion rules, and pointers to
the records that carry detail.

Narrative, measurements, review findings and lessons go in
[the engineering log](docs/engineering-log.md). The authoritative account of a
task is its own record in [docs/tasks/](docs/tasks/), its
[handover](docs/handovers/) and its
[experiment](docs/evidence/experiments/). Keep this file short enough that it
reads as instructions.

## Active assignment

**M3 is authorized** (owner, 2026-09-13). M1 is complete, M2's five items are
all accepted, and M2's formal closure statement remains the owner's to make.

| | |
|---|---|
| Accepted | [Task 0024](docs/tasks/0024-m3-asymmetric-int4-pack-quantized-import.md) — M3 item 2's asymmetric INT4 import (owner, 2026-09-13, after three review rounds). **Importer-only**: no execution, no quality claim. |
| Implemented, unreviewed | [Task 0025](docs/tasks/0025-m3-offline-repack-publication.md) — M3 item 1's offline repack, and [task 0026](docs/tasks/0026-m3-canonical-safetensors-publication.md) — the same publication as **safetensors shards** under the owner's packaging ruling. Corrected after **two** independent reviews: ten findings, then fourteen, all fixed with a regression each. **Bytes only**, and neither closes M3 item 1. |
| Implemented, reviewed three times | [Task 0028](docs/tasks/0028-m3-shared-w4a16-w8a16-execution.md) — M3 item 3's **dense** W4A16/W8A16 path. One shared kernel, four catalogue identities, no whole-tensor dequantization; the republished Laguna module executes on all three GPUs within the predeclared threshold. Seven findings, then **three of those seven still open**, then **two of those three still open** — each round found the repair had closed the path the reproduction used and left the path beside it. All are now fixed with a mutation each. One gap is **named and unmeasured**, not fixed: allocation failure inside `moxie-memory`'s shared `PlanRequest` still aborts, which is a shared-owner change and its own task. **Dense only, synthetic activations, no timing.** |
| Next | M3 item 3's **expert** half — quantized MoE. Nothing routed executes a quantized weight, so no quantized checkpoint runs as a model. |
| Continuation | [The task 0028 handover](docs/handovers/2026-09-14-task0028-shared-quantized-execution.md); [the task 0026 handover](docs/handovers/2026-09-14-task0026-safetensors-publication.md) carries task 0025's forward |

**Nothing in this repository generates a token from a checkpoint.** Tensors have
been imported, expert bytes made resident, real expert weights computed with,
one module repacked into a canonical artifact, and — since task 0028 — that
module's canonical INT4 weight multiplied on all three GPUs. **Every one of
those ran over synthetic activations and routes written by their tests.** One
projection of one layer is not a model: nothing composes a block and nothing
generates a token. No output-quality claim follows from any of it, and none may
be made without paired output against the released model. Say what ran; never
call it model support.

## Owner gates

O1–O5 are resolved; **O6 and O7 are open**, which is why no timing in this
repository is a performance claim. The register is
[owner-gates.md](docs/decisions/owner-gates.md); the rulings that bound current
work:

- **Moxie never quantizes** ([ADR 0017](docs/decisions/adr/0017-v1-catalog-and-no-quantizer.md)). Quantization happens externally and enters as a new source revision. The v1 catalog is ten pinned revisions; `gemma-4-26B-A4B-it` is **not** in it and is M2's BF16 workhorse.
- **v1 quality is a bit-identical repack** `W=(Q-Z)*S` ([ADR 0018](docs/decisions/adr/0018-v1-quality-is-bit-identical-repack.md)), publisher quality accepted as-is. A repack is not evidence about model output.
- **Storage and conversion are user-managed** ([ADR 0020](docs/decisions/adr/0020-user-managed-storage-and-canonical-materialization.md)). **No agent-initiated bulk download, copy or conversion** may start without a task naming artifact, revision, expected size and retention. `/models` and `/fast/models` are read-only inputs.
- **Repack is a Moxie program the user runs offline**, not an external script ([ADR 0021](docs/decisions/adr/0021-repack-is-a-moxie-program.md)); `moxie-repack` is its placement ([ADR 0022](docs/decisions/adr/0022-user-programs-and-canonical-write-authority.md)), and the writer is a **module of that program** ([ADR 0024](docs/decisions/adr/0024-one-storage-crate-and-a-write-module.md)): nothing else can name it, so no rule is needed. A new crate needs several consumers and something distinct to own; one consumer plus a rule to maintain is a module.
- **Offline repacking is provisional** ([ADR 0027](docs/decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)). The owner's requirement is **measured inference benefit**; none exists and **experiment 0007's trigger has not fired**: it needs shared execution that can run a *model* and enough checkpoint infrastructure to load both sides honestly, and task 0028 delivers one dense projection. Never cite layout, packaging, plans or byte-exactness as evidence of speed. Retention is decided by [experiment 0007](docs/evidence/experiments/0007-offline-versus-load-time-preparation.md). Scope is bounded to task 0027; AutoRound, `actorder: static`, F16 passthrough and DeepSeek revisions are **named continuations**, not prerequisites.
- **A quantized linear's numerical gate has two clauses** (owner, 2026-09-14). An output element passes at 2 ULP of BF16 at the oracle's magnitude **or** within `2^-8 · Σ|x·W|`, the reduction's own resolution, because no reordered FP32 reduction can meet the first clause on an output that has cancelled. Narrowing or widening it again is the owner's call, not a task's. [ADR 0028](docs/decisions/adr/0028-quantized-reduction-numerical-gate.md) records it and the measurement that prompted it.
- **A published artifact is safetensors shards plus a TOML manifest** (owner, 2026-09-14; [ADR 0022](docs/decisions/adr/0022-user-programs-and-canonical-write-authority.md) amended, schema in [ADR 0025](docs/decisions/adr/0025-canonical-safetensors-schema.md)). No custom container. Each shard must open with the **reference implementation**; the manifest, not `__metadata__`, defines what the tensors mean. The affine mathematics of [ADR 0023](docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md) is unchanged, and its journal stays private and out of the artifact.

Ask about an open gate in a batch, before dependent conclusions or irreversible
work. A task cannot resolve an owner gate through local inference.

## Read before editing

Read [README.md](README.md), [product boundaries](docs/spec/01-product-and-decisions.md), [common API](docs/spec/02-architecture-and-common-api.md), [agent playbooks](docs/spec/09-agent-playbooks.md), and the assigned milestone in [the roadmap](docs/spec/06-implementation-roadmap.md). Then read the relevant memory, attention/state, sampling/application and validation contracts linked there, and their actual legacy source references. These documents are normative; this short file is their discovery entry point. Read [the engineering log](docs/engineering-log.md) once when starting on this repository, and again whenever a review finds something in your work — most findings there are a shape that has appeared before.

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
