# ADR 0027 — Offline repacking is provisional until it shows measured inference benefit

- **ID / date / author / status:** 0027 / 2026-09-14 / recorded by the implementing agent on owner direction relayed through Herdr / **accepted**
- **Classification:** **owner requirement** (the justification clause), plus an **engineering sequencing recommendation** adopted through the owner's request to instruct. The two are marked separately below and must not be merged.
- **Scope and owning shared component:** `moxie-repack` and the canonical artifact contract. No engine or kernel contract changes here.
- **Supersedes:** in part, the justification and mandatory framing of
  [ADR 0021](0021-repack-is-a-moxie-program.md),
  [ADR 0022](0022-user-programs-and-canonical-write-authority.md) and
  [ADR 0026](0026-generated-plans-and-automatic-budgets.md). Their historical text stands; what changes is why the step exists and whether every user must take it.

## Problem and mechanism

Offline repacking was justified to the owner on the strength of an architectural
argument: a single canonical layout simplifies the runtime, removes per-format
branching from the execution path, and makes importers cheaper to reason about.
The owner challenged that pitch. The agent that made it has acknowledged
presenting potential benefits too confidently and understating implementation
cost.

**The owner's requirement, explicitly:** repacking is acceptable **if it brings
worthwhile inference performance improvements**. Runtime simplification *alone*
does not justify a mandatory conversion step, its conversion cost, or a second
copy of every model on the user's disk.

**No measured inference benefit currently exists.** None can exist yet: there is
no checkpoint execution in this repository, so no prefill or decode comparison
is runnable at all. The following must therefore **never** be recorded, cited or
implied as evidence of faster inference:

* a canonical tensor layout, or its internal tidiness;
* safetensors-plus-manifest packaging, or reference-reader conformance;
* compact generated plans, or any planning census;
* byte-exact repacking, checksums, or importer simplicity.

Each of those is evidence about **bytes and structure**. None is evidence about
**speed**.

## Options examined

| Option | Semantics | Memory / disk | Prefill & decode | Complexity | Migration cost |
|---|---|---|---|---|---|
| **Offline preparation** (today's repack) | Exact represented weights preserved; publication validated through the production reader | A second full copy on disk; bounded RAM during conversion | **Unmeasured.** Hypothesis only | One layout in the engine; a user-visible conversion step | Retiring the mandatory step later means removing a user-facing requirement, with compatibility and retention handled explicitly. **Artifacts a user already owns are theirs**; nothing is deleted without authorization |
| **Load-time preparation** (direct source loading, bounded adaptation once at load) | Same represented weights; adaptation performed once when the model is loaded | *Expected*, subject to implementation: no second copy, with an adaptation peak at startup whose size is not yet known | **Unmeasured.** Plausibly identical sustained throughput if the same kernels see the same layout | Adaptation logic inside the loader | *Expected* to be low, since no user-visible artifact format is introduced — not a no-cost guarantee until a loader exists to price |
| **No preparation** (kernels read source layouts directly) | Same weights | *Expected*: no second copy and no adaptation peak, subject to what the kernels require | **Unmeasured.** May foreclose layout-dependent kernels | Per-format branching in or near the execution path | *Expected* to be low, on the same reasoning |

These are recorded as **candidate mechanisms, not findings**. Nothing here
measures any of the three. The prefill-and-decode column is what the future gate
exists to fill in; the memory, storage and migration columns are **expected
properties of designs that do not exist yet**, and are stated as expectations
rather than as facts about code anyone has written.

## Decision and authority

1. **Repacking is provisional.** It is an experimental artifact and input path
   for developing and evaluating shared execution. It is **not** an established
   permanent prerequisite for all Moxie users, and no record may describe it as
   one.
2. **Bounded current scope** — finish, then stop:
   * task 0027's two-command flow **for the formats already implemented**:
     generated plan, `repack` consuming that plan without repeating expert
     flags, safe automatic budgets with advanced overrides;
   * the existing completeness, source-binding, resource, cancellation and
     publication guarantees, unchanged;
   * user documentation and meaningful integration tests;
   * **all outstanding review corrections from tasks 0025 and 0026, finished and
     measured.** Provisional status is not a waiver for correctness.
3. **Stop broadening the converter** once that scope is met. Deferred as
   **named continuations, not silent removals**: AutoRound import, `actorder:
   static` support, F16 passthrough, and establishing DeepSeek V4 Flash /
   V4.1 Flash revisions. The owner's eventual target models stay recorded in the
   coverage matrix; deferral is not a catalog reduction.
4. **Normal whole-model mode refuses incomplete conversion.** Partial
   experiments require an explicit opt-in flag.
5. **Sequencing recommendation** (engineering, not an owner requirement): after
   this bounded task is validated and handed over for review, the next
   implementation priority is **M3 shared W4A16/W8A16 execution**, together with
   the checkpoint infrastructure later comparisons need. This does **not** close
   M3, and it does not authorize unlimited runtime work.
6. **Keep the door open both ways.** Source descriptors are preserved and the
   future public loading interface must remain capable of accepting a **source
   checkpoint**. A canonical *internal* tensor representation may still prove
   useful; that would not establish a mandatory *user-visible offline
   conversion*. No second engine and no full direct loader is to be built now
   merely to enable the eventual comparison.

**Unchanged by this ADR:** the no-quantizer rule (ADR 0017), exact preservation
of represented weights (ADR 0018), safetensors-plus-manifest packaging when
publishing (ADR 0025), write ownership (ADR 0022/0024), and the read-only
authorization for source roots (ADR 0020). This revises **why** offline
materialization exists and whether it is mandatory. It is **not** permission for
engine code to publish artifacts.

## Evidence and acceptance

The acceptance criterion is deferred to a tracked experiment, because it cannot
be run today:
[experiment 0007 — offline-prepared versus direct-source loading](../../evidence/experiments/0007-offline-versus-load-time-preparation.md).

That contract carries the activation trigger, the paired-measurement
methodology, the cost accounting, and the three possible outcomes. Two rules
from it belong here because they bind this decision:

* **"Worthwhile" needs a predeclared criterion and owner acceptance**, recorded
  *before* results are seen. No percentage threshold may be invented afterwards.
* **Sunk cost is not acceptance evidence.** The amount of work already spent on
  repacking has no bearing on whether it is retained.

Nothing in this ADR establishes a speed claim, resolves **O6** or **O7**, or
authorizes bulk conversion or downloads.

## Enforcement and removal

* Records describing repacking must state its provisional status and the absence
  of runnable-model and performance evidence. The README, the repack guide, the
  support matrix and the task records carry this.
* The coverage matrix distinguishes **discovered/planned**,
  **repacked/validated**, **executed** and **performance-measured**. The
  planning census establishes only the first.
* **Revisit trigger:** experiment 0007 becoming runnable — shared execution plus
  enough real checkpoint infrastructure to measure prefill and decode honestly.
* **Retirement trigger:** outcome (2) or (3) of that experiment — a startup-only
  or internal-simplicity benefit, or no worthwhile benefit. Either **retires the
  mandatory user-visible step** and simplifies toward direct loading. Retiring
  it is a change to what Moxie *requires*, not a cleanup of what users have:
  compatibility for artifacts already published and any retention policy are
  decided explicitly at that point, and **artifacts a user owns are not deleted
  without authorization**. The negative evidence is preserved, and whether an
  optional tool is still worth keeping is reassessed separately.
