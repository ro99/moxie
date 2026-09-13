# Moxie — mandatory agent entry point

This repository builds one NVIDIA inference engine for one interactive user, including models larger than VRAM and large context. It does not build independent engines per model.

## Active assignment

**M1 complete (M1.5 closed 2026-09-12); M2 active, items 1 and 2 accepted and
item 3 awaiting review.** See the
[task 0021 handover](docs/handovers/2026-09-12-task0021-expert-execution-plans.md)
for the current continuation, and the
[closure handover](docs/handovers/2026-09-12-m1-closure-to-m2.md), which carries
M1's exit evidence gate by gate.

**M2 item 1's routed-expert mathematics is accepted**
([task 0019](docs/tasks/0019-m2-routed-expert-semantics.md), 2026-09-12, after
three rounds of independent review): `Route`, `ExpertMlp` and `Combine` as
shared operations with FP64 oracles, the pinned BF16 boundaries that decide
which experts a row selects, and two consumers carrying opposite routing
parameters.

**M2 item 2's residency authority is accepted** (2026-09-12, after seven rounds
of independent review — twenty-seven findings, all reproduced, all fixed, none
disputed; the seventh recommended acceptance with no new blocking findings)
([task 0020](docs/tasks/0020-m2-weight-residency-authority.md)):
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
the prefetch gate, and along promotion's dependency chain. **The acceptance
closes task 0020 only, not M2**, whose exit also needs item 4 and traces
reconciled with the ledger across a whole working set.

**`arch-check` now passes with zero failures.** The four it reported as
"pre-existing" from task 0014 onward were review probe crates parked under
`results/`, which [docs/README.md](docs/README.md) declares ignored scratch; the
crate walk now excludes root scratch **by reachability, computed exactly**, and
**fails closed** — checking everything — when reachability cannot be computed. Do
not carry a standing failure count forward as background noise; that is how a
real one gets missed. And when you narrow a check, **narrow it so that being
wrong is loud**: four review rounds found four different ways a cheaper
approximation of "is this crate part of the build?" silently hid production code
from every rule. **Ask one question once.** Those four were four places
answering it separately — a directory-name test, a membership-string test, a
partial glob expander, a second dependency-table enumeration — and deduplicating
three of them while leaving the fourth is how the fourth was found.

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
property was asserted rather than measured. The sweep prints what it exercised,
and an equivalent mutant is reported as such instead of being counted as a gap.

**M2 item 3 is implemented, corrected after two rounds of independent review,
and awaits a further one**
([task 0021](docs/tasks/0021-m2-expert-execution-plans.md), 2026-09-12; the two
rounds found **fifteen** issues, twelve P1, all reproduced and fixed, none
disputed): CPU
expert fallback and GPU grouped candidate plans under one interface, with an
admitted envelope, a bounded queue that refuses rather than waits, NUMA-placed
host buffers, and a reduction over a permutation the plan computes so partial
outputs are **placed, not accumulated**. The grouped kernel is **bitwise** equal
to task 0019's oracle on all three GPUs for both gate transforms — which needed
`__fmul_rn`/`__fadd_rn` and `__dmul_rn`/`__dadd_rn` throughout, because nvcc
contracts into an FMA by default and that is a different answer. **One layer's
routed expert block of the designated artifact executed**: 118,947,840 B of
layer 0 demand-loaded into a device cache holding **two** of ten experts, 28
evictions, 8 backpressure drains, agreeing with the CPU candidate on 45,056 BF16
components. **The activations are synthetic and the route is written by the
test, so this is not model support and no quality claim follows.** M2's exit
still needs item 4 and traces reconciled with the ledger across a whole working
set.

**A property of this machine is measured or it is not known.** Three plausible
NUMA mechanisms in a row were wrong and only the read-back showed it: a zero
store the compiler may delete, first touch on pages the allocator had already
faulted elsewhere (**3,317 of 8,192** on the wrong node), and the default policy
falling back rather than reclaiming, because node 1 has **334 MB** free against
node 0's **5.1 GB** (**4,471 of 6,144** on the wrong node). `required` now means
`mbind` and the gate is every page. Do not carry an unverified placement,
affinity or bandwidth claim forward.

**Ask of every check what else reaches the resource it guards, and what the next
call does with what it set.** These are two questions and task 0021's two review
rounds are one each.

Nine of the ten first-round findings were one sentence: **a check that existed on one path
and was missing on the neighbouring one.** The upload path validated the backing
a lease is resolved through and the launch path did not — authority A's leases
driven through authority B's backing returned a confident, different answer on a
real GPU. The planner computed one envelope for admission and checked
feasibility against a smaller one. A run could be cancelled but not fail, so a
failed group was followed by a successful reduction over an unwritten buffer.
This is **not** the failure mode task 0020's transition sweep exists for: a
sweep enumerates one state machine's product, and these were parallel paths
never compared to each other. The tenth finding is why it matters that the
question be asked at all — `ExpertGroup`'s launch indices were public `Vec`
fields, so the answer was "anything".

The second round was **the same path, one step later**: quarantine set correctly
at the moment of failure and ignored by `close`, which then released the charge
for buffers that can never be freed; a failure made terminal for a *group* and
not for a *load*; a backing checked and the lease inside it not. Two of its five
were the other half of the first round's own, so **"all ten are closed" was a
claim about fixes rather than a measurement of them** — the same error AGENTS.md
already records three times over test coverage. The answer is the method task
0020 established and task 0021 applied only to its planner: **enumerate the
run's product too** — candidate × failure point × cancellation × close ordering,
with an invariant after every call. Task 0022 is to build it.

**Measure the tests, then fix what the measurement finds.** Task 0021's sweep
started at 13 of 16 mutations with two survivors. One survivor was a **product
defect** — a refusal reporting `CapacityExceeded` where a `required` candidate's
own reason belonged — and the other was a fixture whose every row was already
ascending, so the two reduction orders agreed and a planner ignoring the
parameter passed. It is 16 of 16 now. A fixture on which two behaviours agree
tests neither, and an unreachable branch is a stub: one was found and deleted
the same way. The same battery applied to the review's twelve regressions found
**three that asserted the symptom rather than the check** and would have passed
with the check removed; the second round's six added four more, three of them
aimed at the wrong one of two identical lines. It is eighteen of eighteen now. A
regression is not load-bearing until a substitution says so — and when a
substitution says a check is redundant, **delete the check**: one line went that
way, a state reset the failure path already performed.

**Task 0022 is next**: M2 item 4's Laguna metadata and graph, a second synthetic
MoE consumer through task 0021's interface, and the restricted budget at its
scale, specified in
[the task 0021 handover](docs/handovers/2026-09-12-task0021-expert-execution-plans.md)
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
released model, which is O2. Task 0021 went one step further than task 0020's
read and **computed** with 118,947,840 B of the designated artifact's real
expert weights — but over **synthetic activations and a route its own test
writes**, so it establishes machinery and nothing about output. **One layer is
not a model**: nothing composes a routed layer into a graph that generates a
token, and the whole 51.6 GB working set has not run. M4 still owns device paged
attention, COW forks and page streaming. M11 owns vision.

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
that needed the bounded ranged read it added — and task 0021 ran one layer's
expert block from them. A fact that came out of that: its experts are
**11,894,784 B each**, so at a two-row decode batch the best reuse available is
5,947,392 B per row, and against the declared default amortisation threshold of
1 MiB per row **every expert goes to the CPU candidate**. Whether that is the
right decision is a measurement, and measuring it is M6's.

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
