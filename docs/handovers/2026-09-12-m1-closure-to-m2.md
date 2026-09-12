# Handover — M1 complete (M1.5 closed); M2 active

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Base `e864444` (task 0018 acceptance record). This closure record follows it
  and the accepted tasks it names; no source file changes with it.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and
  `tests/p2p/` remain untouched.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  Nothing under either was copied, converted, deleted or modified while closing
  M1, and no download was made.

## Completed facts

**The owner closed M1.5, and with it M1, on 2026-09-12**, on the reduced Gemma
graph plus the accepted importer, with the remaining blockers recorded rather
than hidden. **M2 is active.**

M1's exit criteria are met and each is gated:

| M1 exit requirement | Evidence |
|---|---|
| One-token and multi-token generation through the shared service | `G-GENERATION-HOST`, `G-GEMMA-REDUCED` |
| Whole-versus-chunked prefill parity on small fixtures | `G-GENERATION-HOST`, `G-GEMMA-REDUCED` (chunks 1, 2, 3, 5, 8, 17, 64) |
| Cancellation then a second generation | `G-GENERATION-ALLOC` (1,000 cancellation/restart cycles) |
| Complete allocation accounting | `G-GENERATION-ALLOC`, `G-PAGED-ALLOCATION`, `G-WINDOW-ALLOCATION` |
| No direct CUDA or model-runtime dependencies | `G-HOST-ARCH`, 74 rejecting + 21 accepted fixtures, 12 rules |
| At least two distinct shapes per foundational operation | Two reduced Gemma geometries with opposite layer-type asymmetries, plus two synthetic graphs |

The accepted M1 work: shared semantic tensors/graph and the bounded host
interpreter (0003), sequence transactions (0004), canonical manifest and bounded
reads (0005), resource ledger and admission (0006–0008), event-backed leases
(0009), device arena (0010), admitted graph resource plan (0011), selected BF16
device chain (0012), appendable paged state (0013), transactional sampler
history (0014), generation service and diagnostic CLI (0015), the reduced Gemma
graph (0016), per-layer paged geometry and window reclamation (0017), and the
compressed-tensors importer (0018).

## Decisions

**M1.5's second clause closes as written, not as wished.** Roadmap M1 item 5
asks for "a small Gemma-like dense test graph **and** an actual supported Gemma
text graph", qualifying the second as "actual checkpoint execution, **when
available/admitted**". The reduced graph exists and is gated. Checkpoint
execution is **not** available or admitted on this machine, and the arithmetic
is the reason rather than an opinion:

- The Gemma 4 artifact is **32.7 GiB** of tensor payload.
- The largest single GPU here is **24 GiB** (3090); aggregate is 63.9 GiB across
  three unequal cards.

So the artifact does not fit on any one device. Executing it needs **M3**'s
W8A16 path for the weights *and* either **M5**'s TP2 across the 3090 pair or
**M2**'s host-backed residency for the placement. That is two milestones away at
minimum, which is why holding M1.5 open would have tracked nothing actionable.
The conditional in the roadmap's own wording is what this closure rests on.

**No owner gate was resolved.** O1–O7 remain open. No numerical threshold,
precision, context target or compatibility surface changed in closing M1.

**Nothing here is model support.** The reduced graph is a synthetic contract
fixture over invented weights; the importer produces canonical tensors that
nothing executes. Both records say so at their point of use, and the CLI prints
its reduction list before any output.

## Remaining hypotheses and blockers

**What M1 did not deliver, with its owner:**

- Gemma 4 checkpoint execution — **M3** (W8A16 path, role binding) plus **M2**
  or **M5** (placement). Recorded in [the bring-up record](../models/gemma4.md).
- The packed-word **lane order** is taken from the pinned reader and cannot be
  verified against the artifact, because all four lanes fall inside one scale
  group. Closing it needs paired output against the released model, which is
  **O2**. **A successful import is not a quality claim.**
- Vision — **M11**. The artifact is `image-text-to-text`.
- Device paged attention, COW forks, host-backed page streaming, recurrent and
  index state — the rest of **M4**. Task 0017 delivered the state schema those
  need, not the paths.

**M2 entry is not blocked; M2's exit is, and the owner is being asked about it.**
Routing semantics, the residency authority, CPU expert fallback and a synthetic
BF16 MoE consumer can all begin now. M2's exit — "real out-of-device-memory
working set executes without OOM or hidden allocations" — needs a real oversized
MoE, and every MoE on this machine is INT4 or MXFP4 while M2 says "use BF16
initially to isolate residency correctness from quantization":

| Artifact | Experts / top-k | On disk | Precision | vs 63.9 GiB VRAM |
|---|---|---:|---|---:|
| `canada-quant/glm-5.3-w4a16-mtp` | 288 / 8 | 178 GB | W4A16 | 2.6x |
| `Intel/GLM-5.3-Flash-W4A16-AutoRound` | 288 / 8 | 170 GB | W4A16 | 2.5x |
| `canada-quant/hy3-w4a16-mtp` | 192 / 8 | 161 GB | W4A16 | 2.4x |
| `cyankiwi/Inkling-Small-AWQ-INT4` | 256 / 6 | 152 GB | INT4 | 2.2x |
| `/data/kimi-k3` | — | 1.5 TB | MXFP4 | 23x |

A BF16 copy of any of these families is **541–652 GB of routed experts alone**,
computed from their own declared geometry. `/fast` has 1,073 GB free and
`/archive` 1,262 GB, so one would fit and would consume most of it. That is an
**O1** catalog question and an **O5** storage question together, and it is put to
the owner below rather than decided here.

**Also outstanding for M2:** item 4 asks for Laguna metadata, and **no Laguna
checkpoint exists on this machine**. O1's M0 evidence still stands for that
family.

## Next task

Blocked on the owner's answer to the batched O1/O5 question below. Once it is
answered, task 0019 is one of:

1. **M3's shared W8A16 execution path**, if the answer is to use the local
   integer artifacts. Owning `moxie-kernels` for the kernel and `moxie-executor`
   for dispatch, with `AffineTensor::reconstruct` plus the accepted BF16 linear
   as the **correctness oracle** — document 03: "bounded reference
   dequantization is not an acceptable final fast path by assertion". Declare
   the error bound before writing the kernel. Gates: real GPU on both
   architectures, the declared shape matrix, bounded prepared-layout accounting
   charging scale metadata as well as weight bits.
2. **M2's shared routing and residency mechanism on a synthetic BF16 MoE**, if
   the answer is to proceed in roadmap order and meet the real-artifact exit
   later. Owning `moxie-engine` for routing and `moxie-memory` for residency,
   with an intentionally restricted budget smaller than the working weights.

Stop conditions carried forward: a second resource or transaction owner, a
model-owned or method-specific execution path, a dequantization fallback
presented as a fast path, a weakened numerical gate, a speed claim without a
paired measurement, a quality claim without O2, or any bulk write, which remains
O5's. **No synthetic graph and no successful import may be described as model
support.**
