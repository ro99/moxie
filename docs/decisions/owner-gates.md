# Owner-gate register — O1–O7

Seeded from reference document 01. Every gate is `OPEN` until the owner answers it. An agent may not
resolve one by inference, and may not let a local task contract override one.

Work on shared infrastructure and small fixtures proceeds while a gate is open. Stop at the gate's
blocked claim, present evidence and a recommendation, and **ask related questions in a batch** before
any conclusion or irreversible work that depends on them.

M0 attached measured evidence to O1, O2 and O5; see
[task 0001](../tasks/0001-m0-freeze-evidence.md) for the batched questions.

The [M0 correction task](../tasks/0002-m0-review-and-integer-transition.md), implemented 2026-09-07,
changed **none** of these gates. It closed enforcement, safety, state and precision-representation
findings; it selected no catalog, approved no quality loss, enabled no low-bit state and wrote no
bulk storage. One question from task 0001's batch is answered and withdrawn: question 6 asked
whether a self-hosted runner was available or the host lane should be made to run without `nvcc`.
That was a technical choice, not an owner decision, and the host lane now runs without it. The
remaining five questions stand.

Record an answer by changing `Status` to `RESOLVED <date>`, writing the ruling verbatim under the
gate, and linking the ADR that implements it. Do not paraphrase the owner's words into a stronger
claim than they made.

---

## O1 — Initial release catalog

**Status:** OPEN

**New owner candidate list:** [ten inspected checkpoint configurations](../evidence/quantization-candidates.md). This updates investigation priorities but does not finalize release membership/order or authorize downloads. Model metadata includes new families/variants; common integer packing does not prove their graph mathematics are already supported.

**M0 evidence (2026-09-07).** Five of the seven roadmap families have no
checkpoint on this machine: Gemma, Laguna, Inkling, GLM-5.2, DeepSeek. Present:
Kimi K3, GLM-5.3-NVFP4, GLM-5.3-Flash-NVFP4. This blocks the *bring-up order*,
not only the final catalog. See [checkpoint-inventory](../evidence/checkpoint-inventory.md).

**M2 sequencing evidence (2026-09-12), asked with O5 below.** M2 item 1 says
"Use BF16 initially to isolate residency correctness from quantization" and its
exit needs a real out-of-device-memory MoE working set. **Every MoE on this
machine is INT4 or MXFP4**, and no Laguna checkpoint exists at all:

| Artifact | Experts / top-k | On disk | Precision | vs 63.9 GiB VRAM |
|---|---|---:|---|---:|
| `canada-quant/glm-5.3-w4a16-mtp` | 288 / 8 | 178 GB | W4A16 | 2.6x |
| `Intel/GLM-5.3-Flash-W4A16-AutoRound` | 288 / 8 | 170 GB | W4A16 | 2.5x |
| `canada-quant/hy3-w4a16-mtp` | 192 / 8 | 161 GB | W4A16 | 2.4x |
| `cyankiwi/Inkling-Small-AWQ-INT4` | 256 / 6 | 152 GB | INT4 | 2.2x |
| `/data/kimi-k3` | — | 1.5 TB | MXFP4 | 23x |

Which exact checkpoint revisions are the initial release catalog, and in what order? Include
image-capable and native speculative-head variants.

**And, for M2 specifically: is a BF16 MoE in the catalog at all, or does M2's
residency work proceed against the local integer artifacts once M3 lands?**

**Owner ruling, 2026-09-12 — partial.** The BF16 MoE for M2 is
`/fast/models/google/gemma-4-26B-A4B-it`. `hy3-w4a16-mtp` **is in scope**.
A Laguna checkpoint, `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`, **is being
downloaded** and was incomplete when this was written.

This answers M2's sequencing and **no download is required**: the artifact is
already local. It does **not** finalize the release catalog or its order, which
is what this gate still blocks. Nothing here approves quality, conversion or
requantization for any of the three.

**Blocks:** final catalog commitment, bulk conversion, and declaring all legacy behavior migrated.
The provisional bring-up order in M1–M11 is for architectural stress coverage only, not an approved
catalog.

## O2 — Acceptable quality loss for the selected integer artifacts

**Status:** OPEN

**M0 evidence.** GLM-5.3-NVFP4 is **W4A4**: it ships `input_scale` activation
scales. Document 03's initial canonical profile is weight-only with BF16
activations, so importing it weight-only discards a part of what its publisher
validated. Separately, converting Kimi K3's MXFP4 experts to NVFP4 would be
double quantisation. Both need paired evidence under this gate.

**Current direction (2026-09-07):** the owner replaces NVFP4 with INT4/INT8/BF16; [ADR 0003](adr/0003-int4-int8-bf16-weight-family.md) records the change. The preceding M0 NVFP4/W4A4 evidence is historical, not a request to implement those profiles now. Existing FP4-to-INT4 conversions are not authorized by this change.

**Implementation note (2026-09-07):** the integer representation is implemented in
`moxie-format::affine` and the NVFP4 codec is retired from the active API. That resolves the
*family*, not this gate. No paired logits, perplexity, task results or conversation failures exist,
because no checkpoint has been imported and nothing has been quantized. A passing host decode test
is not quality evidence and must not be cited as any part of an answer here.

What numerical/task-quality loss is acceptable for the selected INT4/INT8 artifacts versus the released checkpoint? Requires
paired logits, perplexity, task results, and representative conversation failures.

**Blocks:** selecting the production quantizer/profile, and quality release. Proposed engineering
tolerances are not owner approval.

## O3 — Legacy surface compatibility

**Status:** OPEN

Which legacy HTTP/CLI fields, flags, presets, templates, and response extensions must be
byte-compatible? Which may be deprecated?

**Blocks:** removing any existing surface. Default until answered: preserve documented behavior or
provide an explicit compatibility alias. The earlier isolated answer "No" is not to be interpreted as
a ruling.

## O4 — Intrinsic low-bit auxiliary state

**Status:** OPEN

Does the cache restriction (nothing below 16 bits) also forbid a model's intrinsic low-bit
auxiliary index/cache representation, when that representation is part of the released mathematics?

**Blocks:** enabling such a representation. Default until answered: physical cache >= 16 bits,
preserving required rounding in the values. Report infeasibility or fidelity failure rather than
taking a silent exception.

## O5 — Storage and conversion authorization

**Status:** OPEN

**M0 evidence.** Free space: `/` 551 G, `/fast` 1.4 T, `/data` 286 G,
`/archive` 1.3 T (spinning). Converting GLM-5.3-NVFP4 (433 G) fits on `/fast`;
converting Kimi K3 (1.5 T) does not fit anywhere as a second copy. No
higher-precision original of Kimi K3 is known to be available.

The owner has designated `/models` and `/fast/models` as local checkpoint roots;
see [artifact-roots](../evidence/artifact-roots.md). This records where agents
should look, not which revisions are approved or whether a download is complete.

The location question is answered for inspection: owner-directed downloads may be placed under
`/models` or `/fast/models`. The remaining questions are the disk-space budget, which source
revisions are approved, conversion time, retention and whether a higher-precision original is
available.

**M2 sequencing question (2026-09-12).** M2's exit needs a real oversized MoE
in BF16, and none is present. A BF16 copy of any local MoE family, computed from
its own declared geometry, is **541–652 GB of routed experts alone**:

| Family | Routed experts in BF16 | Local INT4 copy |
|---|---:|---:|
| GLM-5.3, 288 experts | 652 GB | 178 GB |
| hy3, 192 experts | 580 GB | 161 GB |
| Inkling-Small, 256 experts | 541 GB | 152 GB |

Free space today: `/fast` **1,073 GB**, `/archive` **1,262 GB**, `/data` 286 GB,
`/` 106 GB. One such download fits on `/fast` or `/archive` and would consume
most of it.

The four questions the agent needs answered, and will not decide:

1. **May a BF16 MoE be downloaded for M2 at all?** If yes, which family and
   revision — this is O1 as much as O5.
2. If the answer is a **smaller** BF16 MoE instead, that is a legitimate option:
   anything above roughly 64 GB is already out-of-device on this hardware, so a
   90–120 GB BF16 MoE would satisfy M2's exit at a fifth of the storage.
3. If the answer is **no download**, does M2's real-artifact exit wait for M3's
   integer path, or is it met on a synthetic BF16 MoE with the real-artifact
   half deferred and recorded as unmet?
4. `hy3-w4a16-mtp` is present locally and appears in no tracked record. Is it in
   scope, and what is it?

**Answered 2026-09-12, and no download was needed.** Questions 1–3 are resolved
by `/fast/models/google/gemma-4-26B-A4B-it`, which was already on disk: BF16,
unquantized, 128 experts at top-k 8, 51.6 GB across 1,013 tensors. Question 4 is
answered yes.

**Nothing was downloaded, copied or converted by this agent**, before or after
the ruling. The storage questions this gate governs — disk budget, approved
revisions, conversion time, retention, and whether a higher-precision original
exists — remain open for every other artifact.

**Blocks:** writing large converted artifacts, unapproved additional downloads, and any bulk copy or
requantization. A task must still name the exact artifact, source revision, expected size and
retention policy before an agent mutates storage. The rewrite request is not blanket permission for
terabytes of new data. **It also now blocks M2's sequencing**, per the four
questions above.

## O6 — Performance limits and acceptable regression

**Status:** OPEN

After measurement, which model/context cases have hard latency/throughput limits, and how much
regression is acceptable in exchange for a quality or capacity gain?

**Blocks:** product performance sign-off. The directional aspirations in document 01 stay reported as
aspirations; they are not to be converted into invented universal thresholds.

## O7 — Bring-up effort target

**Status:** OPEN

What concrete effort/time target defines successful future model bring-up?

**Blocks:** claiming "easy enough" and the M12 verdict. Until answered: require no duplicate
execution ownership, and report actual measured effort.
