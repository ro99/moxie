# Moxie

Moxie is a Rust rewrite of the Strata inference engine. It replaces the pattern of model-specific
inference engines with one scarce-memory execution engine and declarative model definitions. Rust
improves ownership and integration; the essential architectural changes do not depend on Rust.

Prepared for the owner's September 2026 requirements. Canonical weight families are **NVFP4, INT8,
and BF16**. NVIDIA only. One interactive user. OpenAI-compatible HTTP API and CLI chat. Large
context, good prefill and decode, and models larger than VRAM are simultaneous requirements.

This repository currently contains the specification and staged implementation roadmap. It is not an
implementation and makes no claim of achieved performance.

## Repository layout

| Path | Contents | Tracked |
|---|---|---|
| `AGENTS.md` | Single shared policy entry point for implementation agents | yes |
| `CLAUDE.md` | Pointer to `AGENTS.md`; no divergent second policy | yes |
| `docs/spec/` | Normative implementation pack: documents 01–09, the architectural diagnosis, and task templates | **no — gitignored** |

`docs/spec/` is deliberately excluded from version control at the owner's direction, so it does not
reach a remote. Consequence to be aware of: the tracked entry points link into an untracked
directory, and a fresh clone gets pointers with no targets. Documents 07 and 09 require operative
contracts, ADRs, decision registers and handovers to be version-controlled; satisfying that will
need either a private remote for `docs/spec/` or a separate tracked location for the decision
register and ADRs. Raise it before M0's decision register is created.

## Start here

Read these documents in order before implementation:

1. [Product boundaries and decisions](docs/spec/01-product-and-decisions.md): authority, defaults, unresolved release gates.
2. [Architecture and common API](docs/spec/02-architecture-and-common-api.md): ownership, semantic operations, dependency enforcement.
3. [Memory, formats, and CUDA](docs/spec/03-memory-formats-and-cuda.md): the core scarce-memory engine and precision contract.
4. [Attention, parallelism, and speculation](docs/spec/04-attention-parallelism-and-speculation.md): execution strategies and state correctness.
5. [Sampling, API, and CLI](docs/spec/05-sampling-api-and-cli.md): user-visible behavior, including future entropy.
6. [Implementation roadmap](docs/spec/06-implementation-roadmap.md): ordered work packages and exit gates.
7. [Validation and performance](docs/spec/07-validation-and-performance.md): what constitutes evidence of completion.
8. [Strata reference map](docs/spec/08-strata-reference-map.md): frozen source references, assets, regressions, and upstream references.
9. [Agent playbooks](docs/spec/09-agent-playbooks.md), [AGENTS.md](AGENTS.md), and [task templates](docs/spec/templates/TASK.md): how to keep implementation aligned.

[The existing architectural diagnosis](docs/spec/strata-arch-diagnosis.md) is background evidence, not a substitute for these implementation contracts. Its historical policy quotations describe the old project; they are not instructions to preserve its policy mistakes.

## Naming

Crate names in document 02 and the sampling contract identifier use the `moxie-` prefix
(`moxie-engine`, `moxie-memory`, `moxie-v1`, and so on). "Strata" in these documents always refers to
the legacy C++ project, never to anything built here. Source citations, file paths and quoted
historical policy retain their original names.

## Development setup

The legacy repository is a read-only reference. Source citations in the reference map are paths
relative to the legacy root, resolved against commit `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.

- Legacy checkout: `/home/rodrigo/Developer/strata` (verified at that commit, 2026-09-06).
- Do not modify, reset or clean the legacy tree, and do not depend on its dirty worktree or deleted handover.

Toolchain and hardware are pinned and inventoried by M0. A preliminary check on this machine found
rustc/cargo 1.97.1, CUDA 13.0 (nvcc V13.0.88), and three GPUs — RTX 5060 Ti 16 GB as device 0, two
RTX 3090 24 GB as devices 1 and 2. This is a sanity check, not the M0 inventory; M0 must capture
UUIDs, SM capabilities, peer access, PCIe topology under load, NUMA, host/pinned limits and measured
sustained bandwidth.

## First assignment to an implementation agent

> Read AGENTS.md and documents 01–09. Execute M0 only from the roadmap. Inventory the actual hardware, checkpoints, licenses, storage, legacy behavior, and pinned build dependencies. Create the Rust workspace skeleton, architecture checks, decision register, benchmark manifest, and tiny source-linked numerical fixtures. Do not begin a model-specific runtime. Do not convert the full checkpoint library or promise throughput. Report every unresolved gate and the exact M1 task proposed next. Preserve the legacy checkout.

Subsequent assignments use [TASK.md](docs/spec/templates/TASK.md), never "do everything necessary to make model X fast." Each task names a shared owner, a bounded deliverable, its consumers, tests, and stop conditions.

## Meaning of "first-class"

A feature has a shared contract, planner integration, resource accounting, capability reporting, observability, and correctness tests. It is not an optional private implementation in one adapter. It need not be enabled for every workload: for example, tensor parallelism or speculative decoding may lose on this PCIe topology. `auto` may choose a measured alternative; `required` must return an actionable error rather than silently disabling the requested feature.

Final release does not mean every combination is fast or every checkpoint is supported. It means the declared support matrix is tested, the required strategies are implemented through common paths, and unsupported combinations are explicit. No skipped GPU test, tiny-context benchmark, or unmeasured fallback is a passing result.
