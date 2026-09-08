# Moxie

Moxie is a Rust rewrite of the Strata inference engine. It replaces the pattern of model-specific
inference engines with one scarce-memory execution engine and declarative model definitions. Rust
improves ownership and integration; the essential architectural changes do not depend on Rust.

Prepared for the owner's September 2026 requirements. Canonical weight families are **INT4, INT8,
and BF16**, using the shared affine-integer contract in [ADR 0003](docs/decisions/adr/0003-int4-int8-bf16-weight-family.md). NVIDIA only. One interactive user. OpenAI-compatible HTTP API and CLI chat. Large
context, good prefill and decode, and models larger than VRAM are simultaneous requirements.

This repository contains an M0 Rust/CUDA scaffold, tests, specification and roadmap; **it is not yet
an inference engine.** Nothing here loads, imports or executes a model. The
[M0 review and correction task](docs/tasks/0002-m0-review-and-integer-transition.md) records what
passes, with the exact commands, and what remains missing; the
[support matrix](docs/evidence/support-matrix.md) is the list of claims and the gate IDs behind
them. No model-throughput claim has been established, and no checkpoint has been imported.

## Repository layout

| Path | Contents | Tracked |
|---|---|---|
| `AGENTS.md` | Single shared policy entry point for implementation agents | yes |
| `CLAUDE.md` | Pointer to `AGENTS.md`; no divergent second policy | yes |
| `docs/README.md` | Placement contract: what is tracked, what is not, and which template each record uses | yes |
| `docs/spec/01-09`, `docs/spec/strata-arch-diagnosis.md` | Reference documents: the normative specification | **no — kept local** |
| `docs/spec/templates/` | Forms used to author living records | yes |
| `docs/decisions/`, `docs/tasks/`, `docs/handovers/`, `docs/models/`, `docs/evidence/` | Living records: owner-gate register, ADRs, task contracts, handovers, bring-up contracts, support matrix, benchmark manifests, experiment conclusions | yes |
| `crates/` | Shared engine crates. `moxie-types`, `moxie-graph`, `moxie-model-api`, `moxie-format`, `moxie-oracles`, `moxie-state`, `moxie-cuda`, `moxie-kernels` | yes |
| `xtask/` | Command index and its architecture-check fixtures | yes |

The reference documents are kept local at the owner's direction and do not reach the remote. Everything
implementation actually decides, assigns, measures and hands over **is** tracked, which is what
documents 07 and 09 require and what R25 identifies as a cause of legacy drift. See
[docs/README.md](docs/README.md) for the full contract. A fresh clone therefore has the living records
but not the specification; the pack has to be copied in separately.

Run `cargo xtask spec-check` in a new checkout before doing anything else. It reports whether the ten
normative documents are present and whether they match
[their recorded digests](docs/evidence/specification-version.md). It reads and hashes only — it does
not fetch, copy or generate a document. Implementing against inferred contracts because the pack was
missing is R25 repeating itself.

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

## Commands

Two lanes, two builds, two aliases. They are separate on purpose: document 07 requires host CI to
run "without a checkpoint, NVIDIA driver library or CUDA toolkit", and running the host checks
through the device build would leave that independence untested.

| Command | Needs | What it does |
|---|---|---|
| `cargo xtask arch-check` | nothing | Dependency direction, model ownership, and the negative fixtures that prove each rule fires |
| `cargo xtask spec-check [--update]` | nothing | The normative specification is present and unmodified |
| `cargo xtask index` | nothing | The command contracts and their required lanes |
| `cargo test --workspace` | nothing | Host semantics, formats, state, sampling, protocol |
| `cargo xtask-cuda test-gpu [--profile sm_NN]` | CUDA 13.0 + the cards | Real launches on every visible device; **fails** when a required architecture has no passing device |
| `cargo xtask-cuda probe [--out <path>]` | CUDA 13.0 + the cards | Bounded hardware and topology inventory |

`cargo xtask` builds with no CUDA feature: it links no driver and runs no `nvcc`. `cargo xtask-cuda`
adds `--features cuda`, which compiles the fatbins and links `libcuda`. A device command invoked
from the host build fails loudly rather than reporting nothing and exiting zero — a lane that did not
run is unmeasured, never passing.

`test-topology`, `quality`, `bench` and `support-matrix --verify` are named in the index and are
**not implemented**; they arrive with the milestones that define them.

### GPU device ordering

**`CUDA_DEVICE_ORDER=PCI_BUS_ID` is required.** CUDA's default is `FASTEST_FIRST`, which reorders
devices by an opaque heuristic that can change with the driver; `PCI_BUS_ID` requests PCI-bus ordering among the visible CUDA devices. It is not a permanent identity guarantee: visibility filters can renumber devices, and NVML/nvidia-smi identity must be reconciled by UUID/PCI bus. It is set in
[`.cargo/config.toml`](.cargo/config.toml) for everything cargo launches; a binary started outside
cargo needs it in its own environment. Ordinals are for humans and diagnostics — anything recorded as
evidence identifies a GPU by UUID, per document 07.

### Preliminary hardware check

Not the M0 inventory. M0 must still capture SM capabilities, peer access, PCIe topology under load,
NUMA, host/pinned-memory limits and measured sustained bandwidth. rustc/cargo 1.97.1, CUDA 13.0
(nvcc V13.0.88).

| Ordinal | Device | VRAM | PCI bus | NUMA |
|---|---|---|---|---|
| 0 | RTX 5060 Ti | 16311 MiB | `0000:03:00.0` | 0 |
| 1 | RTX 3090 | 24576 MiB | `0000:82:00.0` | 1 |
| 2 | RTX 3090 | 24576 MiB | `0000:83:00.0` | 1 |

The 3090 pair are `PHB` peers on NUMA node 1; the 5060 Ti is on NUMA node 0 and reachable only as
`SYS`, across the socket interconnect. The patched driver reports permissive nvidia-smi topology, but the measured CUDA peer-access API permits only the 3090 pair; enabling P2P is not a speedup on its own — read
[docs/evidence/topology-p2p.md](docs/evidence/topology-p2p.md) before costing any multi-GPU plan. Relevant to M5: the TP2 candidate on the 3090 pair and any
heterogeneous stage involving the 5060 Ti are not comparable placements, and document 03 requires the
link widths and simultaneous-transfer behavior to be measured rather than assumed (R12).

## Current state and the next assignment

M0's deliverables are in, and the corrections its
[review](docs/tasks/0002-m0-review-and-integer-transition.md) required are implemented: the host lane
runs with no CUDA toolkit or driver, architecture enforcement resolves package identity rather than
manifest keys, the CUDA image boundary states its trust obligation, sequence state distinguishes
accepted history from published usage from tentative execution and requires retained logit
provenance, and the canonical weight family is the shared INT4/INT8/BF16 affine-integer contract.
A second review then found six contracts that still accepted an invalid state -- a retained result
that did not identify its sequence or its prefix's contents, an earlier snapshot accepted as
restoration to a later prefix, grouped imports and a library declared under `tests/` slipping past
architecture enforcement, an unbounded PTX entry point, a rollback that unpublished released output,
and two reference paths returning successful nonfinite results. Those are closed too. The
A third review found three more — a derived `Clone` that handed the new sequence identity to a
second authority, restoration evidence with no branch or prefix-version identity, and source
enforcement still missing ordinary Rust syntax — and those are closed too. The
A fourth found that the architecture checker's new parser was not traversing function bodies, and
that is closed too. The [first](docs/handovers/2026-09-07-m0-correction-results.md),
[second](docs/handovers/2026-09-07-second-review-corrections.md),
[third](docs/handovers/2026-09-07-third-review-corrections.md) and
[fourth](docs/handovers/2026-09-07-fourth-review-corrections.md) handovers are the short versions.

The next bounded task is [task 0003](docs/tasks/0003-m1-bf16-reference-interpreter.md), the BF16 host
reference interpreter. It is deliberately smaller than document 06's whole M1: the manifest reader,
the rank-owned CUDA path, paged state and the generation service are separate tasks that consume it.

Assignments use [TASK.md](docs/spec/templates/TASK.md), never "do everything necessary to make model
X fast." Each task names a shared owner, a bounded deliverable, its consumers, tests, and stop
conditions.

The original M0 assignment, kept for the record:

> Read AGENTS.md and documents 01–09. Execute M0 only from the roadmap. Inventory the actual hardware, checkpoints, licenses, storage, legacy behavior, and pinned build dependencies. Create the Rust workspace skeleton, architecture checks, decision register, benchmark manifest, and tiny source-linked numerical fixtures. Do not begin a model-specific runtime. Do not convert the full checkpoint library or promise throughput. Report every unresolved gate and the exact M1 task proposed next. Preserve the legacy checkout.

## Meaning of "first-class"

A feature has a shared contract, planner integration, resource accounting, capability reporting, observability, and correctness tests. It is not an optional private implementation in one adapter. It need not be enabled for every workload: for example, tensor parallelism or speculative decoding may lose on this PCIe topology. `auto` may choose a measured alternative; `required` must return an actionable error rather than silently disabling the requested feature.

Final release does not mean every combination is fast or every checkpoint is supported. It means the declared support matrix is tested, the required strategies are implemented through common paths, and unsupported combinations are explicit. No skipped GPU test, tiny-context benchmark, or unmeasured fallback is a passing result.
