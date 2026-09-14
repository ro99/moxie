# 0007 — Does offline preparation buy inference performance? **Pending; not runnable.**

Date opened: 2026-09-14. Milestone: gates M3 item 1's retention.
Status: **PENDING — the trigger has not fired. Nothing here has been measured.**
Decision it serves: [ADR 0027](../../decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md).

## Why this exists

The owner's requirement for offline repacking is that it **brings worthwhile
inference performance improvements**. Runtime simplification alone does not
justify a mandatory conversion step, its cost, or a second copy of every model.

That requirement cannot be tested today: this repository cannot execute a
checkpoint, so no prefill or decode number exists for either side. This contract
is written **now, before any result is visible**, so that the criterion is
predeclared rather than fitted to whatever the first measurement happens to
show.

**The absence of evidence is not evidence.** Nothing may infer success from the
fact that this experiment has not run.

## Activation trigger

All of the following, together:

1. Shared execution exists — the M3 W4A16/W8A16 paths can run a model.
2. Enough real checkpoint infrastructure exists to load and run **both** sides
   honestly.
3. The comparison is subject to the existing **O6** and **O7** gates, which this
   experiment does not resolve.

Until all three hold, this record stays `PENDING` and no partial number from it
may be cited.

## What is compared

**Offline-prepared layout** against **a reasonable direct-source loading path**,
in the **same engine**, with the **same shared kernels** and the same execution
policies wherever applicable.

Fairness rules, stated in advance:

* Direct loading **may** perform necessary bounded adaptation **once** at load
  time. It must not be handicapped by repeatedly redoing work, nor by an
  artificial slow path introduced for the comparison.
* If a layout difference enables a faster kernel, the experiment must **identify
  the mechanism** — which kernel, which property of the layout — and then ask
  whether **load-time preparation can obtain the same benefit** within the same
  resource limits. "The prepared path was faster" is not a finding until that
  question is answered.

## What is measured

Paired prefill **and** decode, at:

* the same exact model revision;
* the same hardware, topology and device UUIDs;
* the same precision, context length and workload shape;
* the same placement and resource limits;
* **including the intended larger-than-VRAM / expert-streaming case**, which is
  the case the canonical layout was argued for.

Reported separately, never merged:

| Quantity | Why separate |
|---|---|
| **Warm and cold** conditions | A page-cache-warm number is a different claim |
| **Startup** versus **sustained** inference | A startup-only difference is outcome (2) below, not (1) |
| RAM and VRAM peaks | A faster path that needs more memory has a cost |
| Transfer and storage bytes | Streaming cost is part of the comparison |
| Conversion wall-clock time | The preparation step's own price |
| Extra disk footprint | A second copy of every model is the user's cost |

Also required, from existing project policy:

* **Byte-preservation evidence** for the prepared artifact; and
* the **released-model output/quality checks** the project requires.
  Byte-identical weight repacking does **not** on its own certify identical
  generated output, and this experiment may not treat it as doing so.

**Kernel and transfer microbenchmarks may guide design but cannot close this
gate.** It is an end-to-end comparison or it is not this experiment.

## The acceptance criterion

**Not yet set, deliberately.** "Worthwhile inference improvement" requires a
**predeclared numeric criterion and owner acceptance**, recorded here when the
experiment becomes runnable and **before** results are seen. Inventing a
percentage after seeing the numbers is how a negative result becomes a positive
one.

Today's bounded implementation is **not blocked** on that number.

## Outcomes, declared in advance

1. **Worthwhile inference benefit that justifies the preparation cost.**
   Propose retaining the offline step, with the evidence attached and the cost
   columns filled in.
2. **Startup-only, or internal-simplicity, benefit.** This **does not satisfy**
   the owner's stated justification. The mandatory step is not retained on this
   basis.
3. **No worthwhile benefit.** **Retire** the mandatory user-visible step and
   simplify toward direct loading. Retiring the requirement is not deleting
   anyone's data: compatibility for artifacts already published and any
   retention policy are decided explicitly at that point, and **artifacts a user
   owns are not deleted without authorization**. **Preserve the negative
   evidence** — document 07 requires rejected results to be kept with their
   mechanism and exact scope — and reassess separately whether an optional tool
   is worth retaining.

**Sunk cost is not acceptance evidence.** How much work went into repacking has
no bearing on which outcome is recorded.

## What this experiment does not do

It does not establish any speed claim today, does not resolve O6 or O7, and does
not authorize bulk conversion or downloads.
