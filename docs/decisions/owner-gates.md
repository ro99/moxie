# Owner-gate register — O1–O7

Seeded from reference document 01. Every gate is `OPEN` until the owner answers it. An agent may not
resolve one by inference, and may not let a local task contract override one.

Work on shared infrastructure and small fixtures proceeds while a gate is open. Stop at the gate's
blocked claim, present evidence and a recommendation, and **ask related questions in a batch** before
any conclusion or irreversible work that depends on them.

Record an answer by changing `Status` to `RESOLVED <date>`, writing the ruling verbatim under the
gate, and linking the ADR that implements it. Do not paraphrase the owner's words into a stronger
claim than they made.

---

## O1 — Initial release catalog

**Status:** OPEN

Which exact checkpoint revisions are the initial release catalog, and in what order? Include
image-capable and native speculative-head variants.

**Blocks:** final catalog commitment, bulk conversion, and declaring all legacy behavior migrated.
The provisional bring-up order in M1–M11 is for architectural stress coverage only, not an approved
catalog.

## O2 — Acceptable NVFP4 quality loss

**Status:** OPEN

What numerical/task-quality loss is acceptable for NVFP4 versus the released checkpoint? Requires
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

Which storage paths, disk-space budget, source checkpoints, and conversion time may be used? Is a
higher-precision original available?

**Blocks:** writing large converted artifacts, and downloading checkpoints. The rewrite request is
not blanket permission for terabytes of new data.

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
