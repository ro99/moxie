# Moxie

Moxie is a Rust rewrite of the Strata inference engine. It replaces the pattern of model-specific
inference engines with one scarce-memory execution engine and declarative model definitions. Rust
improves ownership and integration; the essential architectural changes do not depend on Rust.

Prepared for the owner's September 2026 requirements. Canonical weight families are **INT4, INT8,
and BF16**, using the shared affine-integer contract in [ADR 0003](docs/decisions/adr/0003-int4-int8-bf16-weight-family.md). NVIDIA only. One interactive user. OpenAI-compatible HTTP API and CLI chat. Large
context, good prefill and decode, and models larger than VRAM are simultaneous requirements.

This repository contains shared Rust/CUDA execution components and a bounded
host-reference generation service; **it does not yet execute a checkpoint-backed
model.** The diagnostic CLI emits synthetic token IDs. **Nothing here executes a
released model**, and no model-throughput or model-quality claim has been
established.

[Task 0018](docs/tasks/0018-m3-compressed-tensors-int8-importer.md) imports
compressed-tensors `pack-quantized` INT8 weights -- including real tensors from a
local Gemma 4 artifact -- into canonical affine form. **An import is not
support**: nothing runs those weights, because the W8A16 execution path, the
repacker and the canonical manifest write are all still M3's, and the reduced
Gemma graph remains a synthetic fixture over invented weights.

The
[M0 review and correction task](docs/tasks/0002-m0-review-and-integer-transition.md) records what
passes, with the exact commands, and what remains missing; the
[support matrix](docs/evidence/support-matrix.md) is the list of claims and the gate IDs behind
them.
Downloaded checkpoints for later, explicitly scoped tasks are found under the owner-designated
local roots [`/models` and `/fast/models`](docs/evidence/artifact-roots.md); their presence does not
mean that Moxie has imported or supports them.

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
| `crates/moxie-storage` | The one crate that touches the filesystem for checkpoints: bounded reads, chunk validation, and the bounded-read primitives the repacker writes through | yes |
| `crates/moxie-repack/` | `moxie-repack`: the offline inspector, repacker and verifier (ADR 0021/0022), whose `write` module owns canonical publication (ADR 0024) | yes |
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
| `cargo clippy --workspace --all-targets -- -D warnings` | nothing | The host lane's lints |
| `cargo clippy --workspace --all-targets --features moxie-executor/driver -- -D warnings` | CUDA 13.0 | The executor's device code |
| `cargo clippy --workspace --all-targets --features cuda -- -D warnings` | CUDA 13.0 | **`xtask`'s own device code.** A third lane, declared by task 0023: the first two do not compile it, and two lints stood in it unnoticed from task 0021 until an independent review ran this |
| `cargo run -p moxie-cli -- diagnostic --shape b --prompt 0,1,2,3 --max-new 4 --chunk 3 --temperature 1 --seed 42` | host telemetry | Synthetic token-ID generation through the shared host-reference service; no checkpoint or GPU attention |
| `cargo xtask-cuda test-gpu [--profile sm_NN]` | CUDA 13.0 + the cards | Real launches on every visible device; **fails** when a required architecture has no passing device |
| `cargo xtask-cuda test-bf16-chain` | CUDA 13.0 + one card | Reduced H8/H17 selected semantic chain used by Compute Sanitizer |
| `cargo xtask-cuda probe [--out <path>]` | CUDA 13.0 + the cards | Bounded hardware and topology inventory |
| `moxie-repack inspect --selection <file> --source-root <dir> <budgets>` | a source checkpoint | Headers only: what a selection is, what it would cost, and what is refused. Writes nothing |
| `moxie-repack repack --selection <file> --source-root <dir> --out <dir> <budgets>` | a source checkpoint | Converts a selection in bounded units, resumes an interrupted run, validates through the production reader, publishes with one rename |
| `moxie-repack verify --artifact <dir> --scratch-bytes <n>` | a published artifact | Re-reads every payload through the production reader and checks it against its recorded checksum |

`cargo xtask` builds with no CUDA feature: it links no driver and runs no `nvcc`. `cargo xtask-cuda`
adds `--features cuda`, which compiles the fatbins and links `libcuda`. A device command invoked
from the host build fails loudly rather than reporting nothing and exiting zero — a lane that did not
run is unmeasured, never passing.

`test-topology`, `quality`, `bench` and `support-matrix --verify` are named in the index and are
**not implemented**; they arrive with the milestones that define them.

### The offline repacker

`moxie-repack` is the program [ADR 0021](docs/decisions/adr/0021-repack-is-a-moxie-program.md) and
[ADR 0022](docs/decisions/adr/0022-user-programs-and-canonical-write-authority.md) place canonical
write authority in, and task 0025 is what built it. Five budgets are required on every invocation —
`--total-bytes`, `--header-bytes`, `--scratch-bytes`, `--chunk-file-bytes` and `--disk-bytes` — and
there is no default for any of them: how much of a user's machine a tool may spend is not a question
a tool should answer on the user's behalf.

What it converts is a **selection**: a small TOML document naming each tensor, its role, its source
shard and, for a quantized module, its packing parameters. Nothing is discovered, expanded from a
pattern, or inferred from a name, which is what keeps ADR 0020's "no agent-initiated bulk
conversion" a property of the tool rather than a promise about how it is invoked.

Measured capabilities, and nothing else: BF16 passthrough, and compressed-tensors `pack-quantized`
INT4/INT8 at group 32, group 128 and per-channel, symmetric or with zero points packed along the
output axis. AutoGPTQ/AutoRound packing, activation-order maps, tokenizers, fused expert role
mapping and whole-catalog conversion are **refused by name**. Exit status distinguishes the
outcomes: 0 published or verified, 1 a command-line error, 2 refused or failed (leaving a resumable
destination), 3 cancelled, and 4 **published, durability unconfirmed** — the artifact exists and the
confirming `fsync` did not report success, which is neither a failure nor a success.

A repack is a statement about bytes. It is
[ADR 0018](docs/decisions/adr/0018-v1-quality-is-bit-identical-repack.md)'s v1 quality definition
and says nothing about what a model produces; a selection of part of a model publishes a **partial**
artifact, which the reader opens for inspection and refuses to load.

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

**M3 is authorized** (owner, 2026-09-13). M1 is complete and M2's five items are all accepted; M2's
formal closure statement remains the owner's to make. The list below is the historical record of
how the earlier milestones closed and is not the current assignment — [AGENTS.md](AGENTS.md) holds
that, and is the only file that does.

The most recent completed work is [task 0024](docs/tasks/0024-m3-asymmetric-int4-pack-quantized-import.md),
M3 item 2's asymmetric INT4 importer, accepted on 2026-09-13 after three rounds of independent
review, and [task 0025](docs/tasks/0025-m3-offline-repack-publication.md), M3 item 1's offline
repack and canonical publication — the `moxie-repack` program documented above. Neither establishes
model execution or output quality: **nothing in this repository executes a quantized weight**, which
is M3 item 3 and the largest remaining gap in the milestone.

**M1.4 complete; M1.5 active.** The owner accepted
[task 0012](docs/tasks/0012-m1-selected-bf16-device-chain.md) on 2026-09-10 after independent
verification of the selected BF16 device chain and all five review corrections through `1138a2a`.
The accepted scope includes explicit refusal of unqualified RMS underflow; it establishes no
model execution, actual-context support or performance claim.

[Task 0013](docs/tasks/0013-m1-appendable-paged-state.md) is accepted after independent
review through `0f39cf5`. Its contract `a80ff2c` preceded implementation `c14e32a`;
correction `c9a4b33` classifies lineage allocation exhaustion as capacity failure.
Tests cover physical byte preservation, admission, rollback, cancellation and 32,768
actual stored rows; this establishes no attention, model or long-context execution claim.

The owner accepted
[task 0014](docs/tasks/0014-m1-transactional-sampler-history.md): sampler history and
deterministic greedy/temperature distribution, with history and rollback using
task 0004's accepted transactions. Implementation follows contract `4054dd7` and
was independently reviewed at `23a7a40` / `1e6f253` and accepted on 2026-09-11.
The task record qualifies the review's layout-chronology documentation finding.

[Task 0015](docs/tasks/0015-m1-generation-service-diagnostic-cli.md) implements the
remaining M1.4 service/diagnostic-CLI slice at `f7cff56`, after contract `2fbacec`.
Correction `5990f2a` makes forward payload allocation failures typed and recoverable,
validates every row count a request executes, and binds paged history to one immutable
program configuration. Correction `7a08adf` also rolls back the whole open paged
transaction when fallible input/weight preparation fails before interpretation.
The service executes synthetic graphs through the shared host interpreter, physical
paged state and sampler; it commits token events and releases resources on finish,
cancellation, failure or disconnect. Its explicit `host-reference` profile is
limited to 256 requested context tokens, with bounded graph/scratch sizes.
It has no tokenizer, text stop/EOS, HTTP or checkpoint support. Use `--help` for the
diagnostic CLI and `--cancel-after N` for deterministic cancellation testing.
Final independent review through `3f13784` found no remaining blocker, and the owner
accepted task 0015 on 2026-09-11. This closes M1.4 within the explicit synthetic
host-reference scope; **M1.5 is active**. Model/checkpoint integration and M4 device
attention remain separate work. The
[active handover](docs/handovers/2026-09-11-m1.4-closure-to-m1.5.md) records the
accepted boundary and next contract.

Assignments use [TASK.md](docs/spec/templates/TASK.md), never "do everything necessary to make model
X fast." Each task names a shared owner, a bounded deliverable, its consumers, tests, and stop
conditions.

## Meaning of "first-class"

A feature has a shared contract, planner integration, resource accounting, capability reporting, observability, and correctness tests. It is not an optional private implementation in one adapter. It need not be enabled for every workload: for example, tensor parallelism or speculative decoding may lose on this PCIe topology. `auto` may choose a measured alternative; `required` must return an actionable error rather than silently disabling the requested feature.

Final release does not mean every combination is fast or every checkpoint is supported. It means the declared support matrix is tested, the required strategies are implemented through common paths, and unsupported combinations are explicit. No skipped GPU test, tiny-context benchmark, or unmeasured fallback is a passing result.
