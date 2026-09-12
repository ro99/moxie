# Moxie — mandatory agent entry point

This repository builds one NVIDIA inference engine for one interactive user, including models larger than VRAM and large context. It does not build independent engines per model.

## Active assignment

**M1 complete (M1.5 closed 2026-09-12); M2 active, item 1 accepted, item 2
implemented and corrected after six rounds of independent review.** See the
[task 0020 handover](docs/handovers/2026-09-12-task0020-weight-residency-authority.md)
for the current continuation, and the
[closure handover](docs/handovers/2026-09-12-m1-closure-to-m2.md), which carries
M1's exit evidence gate by gate.

**M2 item 1's routed-expert mathematics is accepted**
([task 0019](docs/tasks/0019-m2-routed-expert-semantics.md), 2026-09-12, after
three rounds of independent review): `Route`, `ExpertMlp` and `Combine` as
shared operations with FP64 oracles, the pinned BF16 boundaries that decide
which experts a row selects, and two consumers carrying opposite routing
parameters.

**M2 item 2's residency authority is implemented and corrected after six
rounds of independent review** — twenty-six findings, all reproduced, all fixed,
none disputed
([task 0020](docs/tasks/0020-m2-weight-residency-authority.md), 2026-09-12):
`moxie_memory::residency` is the **one** production weight-residency owner, with
document 03's chunk identity, its lifecycle and failure transitions, coalescing,
event-bound device uploads whose allocation is tied to its reservation,
deterministic demand LRU with a bounded prefetch class, ADR 0009's
conditional-memory floor, and an admission report naming the incoming chunk
before any victim is chosen. An `arch-check` rule rejects a second owner. All
nine of M2 item 5's cases pass, on the host lane and on all three GPUs, alongside
a regression for each review finding. One of those findings corrected a claim
rather than a defect: **a nonblocking acquire is necessary but not sufficient for
deadlock freedom**, and the cycles it missed were between the demand counter and
the prefetch gate, and along promotion's dependency chain.

**`arch-check` now passes with zero failures.** The four it reported as
"pre-existing" from task 0014 onward were review probe crates parked under
`results/`, which [docs/README.md](docs/README.md) declares ignored scratch; the
crate walk now excludes root scratch **by reachability, computed exactly**, and
**fails closed** — checking everything — when reachability cannot be computed. Do
not carry a standing failure count forward as background noise; that is how a
real one gets missed. And when you narrow a check, **narrow it so that being
wrong is loud**: three review rounds found three different ways a cheaper
approximation of "is this crate part of the build?" silently hid production code
from every rule.

**A state machine's tests should enumerate its product, not sample it — and the
sweep itself needs evidence.** Twenty-three review findings against task 0020
shared one shape: a transition that was individually reasonable left the
structure inconsistent in a combination nobody had written a test for.
`ResidencyAuthority::check_invariants` states the invariants once and
`residency_transitions.rs` sweeps 800 combinations calling it after every
operation. Two things are required of such a sweep, both learned the hard way:
its harness must perform **only** work the scheduler actually handed out — the
first version completed work it discovered, and a mutation that discarded every
promoted order passed all 200 combinations — and its strength must be
**measured by mutation testing** rather than asserted.

**State coverage as a number the test prints, never as a sentence in a record.**
Three claims about task 0020's test strength were wrong in the same way: the
property was asserted rather than measured. The sweep now prints what it
exercised, and an equivalent mutant is reported as such instead of being counted
as a gap. Apply all of this to M2 item 3's queues rather than rediscovering it. **Task 0021 is next**: M2 item 3's CPU expert fallback and GPU
grouped candidate plans under one interface, specified in
[the task 0020 handover](docs/handovers/2026-09-12-task0020-weight-residency-authority.md)
and not yet authored.

M1's accepted work: shared semantic tensors/graph and the bounded host
interpreter, sequence transactions, canonical manifest and bounded reads, the
resource ledger and admission, event-backed leases, the device arena, the
admitted graph resource plan, the selected BF16 device chain, appendable paged
state, transactional sampler history, the generation service and diagnostic CLI
(tasks 0003–0015), the reduced Gemma graph
([0016](docs/tasks/0016-m1-gemma-reduced-graph.md)), per-layer paged geometry and
window reclamation
([0017](docs/tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md),
accepted 2026-09-12), and the compressed-tensors importer
([0018](docs/tasks/0018-m3-compressed-tensors-int8-importer.md), accepted
2026-09-12 within its import-only scope).

**M1.5 closed on the reduced graph, and the arithmetic is why.** Roadmap M1 item
5 qualifies its second clause as "actual checkpoint execution, **when
available/admitted**". The Gemma 4 artifact is **32.7 GiB** of tensor payload
against a **24 GiB** largest single GPU, so it fits on no device here: executing
it needs M3's W8A16 path *and* either M5's TP2 or M2's host-backed residency.

**Nothing executes a checkpoint.** The reduced graph is a synthetic contract
fixture over invented weights and may not be described as model support; the
importer produces canonical tensors that nothing runs, and its packed-word lane
order is cited from the pinned reader rather than verified, so **no quality
claim follows from a successful import** — that needs paired output against the
released model, which is O2. Task 0020 demand-read 107,053,056 B of the
designated artifact's real expert weights through the residency authority and
verified every range against an independent read; **reading is not executing**,
nothing computed with those bytes, and a demand-loaded expert is not model
support. M4 still owns device paged attention, COW forks and page streaming. M11
owns vision.

**M2 is active and proceeds in roadmap order.** The owner designated
`/fast/models/google/gemma-4-26B-A4B-it` as M2's BF16 MoE on 2026-09-12: BF16,
unquantized, **128 experts at top-k 8**, 51.6 GB across 1,013 tensors, already
on disk — **no download is required, and none was made.** It is the same
`gemma4` family M1 already gated, with the same `(layer + 1) % 6` global
predicate, `attention_k_eq_v`, local 8x256 versus global 2x512 key/value
geometry, softcap 30.0 and 1,024-token window, so the accepted text-tower
mathematics and task 0017's per-layer paged geometry transfer rather than being
rebuilt. Its experts are **fused per layer** — one `experts.gate_up_proj` and
one `experts.down_proj` holding all 128 — and **every layer carries a dense
`mlp` beside the routed experts**, so routing semantics must model a shared
expert explicitly. At 51.6 GB it fits aggregate VRAM but no single 24 GiB
device, and M2 item 4's restricted budget makes it oversized by construction.
Task 0019 composed its routed block over synthetic weights at reduced scale and
recorded the artifact's inventory in
[the bring-up record](docs/models/gemma4.md#the-26b-a4b-moe-variant). Task 0020
made its expert bytes resident on demand — the experts are fused per layer, so
that needed the bounded ranged read it added — but **nothing runs it**, which is
what M2 item 3 exists to make possible.

`hy3-w4a16-mtp` is in scope. The Laguna checkpoint at
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` **finished downloading and was
verified complete on 2026-09-12** — all 15 shards, each satisfying
`8 + header + payload_end == file size`, payload ends summing to the index's
`total_size` of 76,813,095,232 B. That check is all that has been done to it:
**no metadata has been interpreted and no tensor read**, and its
`configuration_laguna.py` and `modeling_laguna.py` are remote code that document
03 forbids executing. M2 item 4 is unblocked on availability, not on inspection.
None of the three is approved for quality, conversion or requantization: O1's
catalog and O5's storage questions stay open.

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
