# ADR 0029 — New crate or new module: the packaging rule

- ID / date / author / status: 0029 / 2026-09-15 / implementation agent / adopted; consolidation directed in-session the same day.
- Classification: roadmap default. It writes down the packaging rule already implied by document 02 and ADRs 0006, 0013 and 0024; it resolves no owner gate and changes no numerical, precision, context, concurrency or catalog contract.
- Scope and owning shared component: workspace packaging; document 09 (gains §G); `xtask arch-check` where a new boundary lands.
- Supersedes / superseded by: nothing. Interprets document 02's "fewer physical crates ... no empty abstraction crates" without changing its ownership table.

## Problem and mechanism

Nineteen crates plus `xtask` read as explosion, and the question arrived as `moxie-interp` versus `moxie-host`. The count maps to document 02 ownership rows, but the rule agents were expected to apply lived in four places: document 02 §Ownership ("fewer physical crates if ... machine-checked between modules; no empty abstraction crates"), ADR 0006 §Cost (a crate must own real content plus a confinement rule no other crate could carry), ADR 0013 (crate-per-family rejected: the module boundary enforces the same dependency list), and ADR 0024's general rule (one consumer plus a maintained rule is a module). Each new boundary was re-argued from taste.

The worked example that prompted this: `moxie-host` is the machine sensor — reads `/proc`/`/sys` under an injectable root, emits `MeasuredHost`, depends on `moxie-types` alone; consumers `moxie-cli`, `moxie-repack`, `xtask`. `moxie-interp` is reference execution — walks a validated `Graph`, dispatches to `moxie-oracles`, advances `moxie-state`; deps `types`/`graph`/`format`/`oracles`/`state`; production consumer `moxie-engine`. A merge puts the sensor behind `graph`+`oracles`+`state` and pulls machine-telemetry reading into every graph execution, demoting `TELEMETRY_OUTSIDE_HOST` and `MEMORY_PROBES_THE_MACHINE` from dependency properties to coding promises — the shape ADR 0006 rejected three times. Different rows, opposite constraints: the split is the mechanism, not the ceremony.

## Options examined

**1. Merge by consumer count — one production consumer implies a module.** Rejected: it condemns `moxie-plan` (pure `compile` beside the executor's effects), `moxie-cuda`/`moxie-kernels` (audited-unsafe ABI boundary), `moxie-oracles` (must stay free of `state`/`format` so test binaries reach mathematics without execution) and `moxie-sampling` (pure transforms beside the state's allocation authority). Count alone is not the rule; each of these carries a confinement property a module cannot supply.

**2. Split by distinction — a crate per family, per writer, per future program.** Rejected twice with evidence: ADR 0013 measured four files of ceremony per model buying no enforcement the module boundary does not already provide, and ADR 0024 showed the writer split duplicating `read_exact_at` for one permanent consumer. Distinction without a boundary reason is ceremony.

**3. Two-condition rule with a pre-proposal checklist (selected).** Keeps every current split that carries confinement, deletes the next `moxie-storage-write`, and moves the argument from taste to named properties.

## Decision and authority

A new crate needs **both**:

1. A distinct sole responsibility with real content — a parser, ledger, oracle corpus, ABI boundary. A name is not content.
2. A boundary reason a module cannot supply: two or more production consumers that must share it, **or** a confinement property expressible as a machine check — dependency exclusion, purity, an unsafety boundary, single-reader telemetry, model-isolation direction.

Corollaries, each precedented:

- One consumer plus a rule someone has to maintain is a module. Structural confinement — the code is not in their dependency tree — is stronger than the rule it replaces and costs nothing to maintain (ADR 0024).
- Refuse the fold when the consumer would inherit dependencies it must not have (interp/oracles/state out of the host sensor; state/format out of the oracles; effects out of the plan; unsafe out of the executor and the engine).
- Refuse the split when the module boundary already enforces the same dependency list (models: one crate, a module per family). The escape hatch stays: a family needing a dependency the others must not have takes its own `moxie-models-*` crate under the same list (ADR 0013).
- No placeholder crates. Each program arrives with its first real command (ADR 0022, program inventory).

Checklist before proposing a crate: the document 02 owner row; the production consumers (dev-dependencies do not count); the `arch-check` rule plus its rejecting fixture; what the consumer would wrongly inherit if folded, or what the split would duplicate if divided.

This is a roadmap-level default inside existing documents 01/02 authority. It needs no owner approval and takes none: no owner gate is touched. It amends document 09 by adding §G; document 02's table is unchanged. A future crate meeting the checklist cites this ADR and ships its rule with fixtures; a split failing it lands as a module.

## Evidence and acceptance

- Precedents with teeth: ADR 0006 (two machine-checked rules, both with rejecting fixtures), ADR 0013 (three fixtures; escape-hatch fixtures unchanged and still passing), ADR 0024 (crate deleted, duplicate `pread` deleted, rule with its fixtures removed because there is no edge left to police).
- Current workspace against the rule: all nineteen crates plus `xtask` inventoried in the README table fixed alongside; the eight single-production-consumer crates (`interp`, `plan`, `cuda`, `kernels`, `oracles`, `sampling`, `engine`, `models`) each name their confinement property in 09 §G. `moxie-executor` is the execution owner whose engine integration is pending M3 work, not a packaging question.
- Acceptance: 09 §G present; `spec-check` manifest regenerated (only document 09's digest moves); `arch-check` unchanged — no dependency edge moved, so its rule/fixture counts must not move.

## Enforcement and removal

- Reviewers reject a new crate that does not carry the §G checklist, and reject a new module that duplicates an owner's vocabulary — a second reader, writer, cache or telemetry parser — without showing why the fold refusal applies.
- No expiry and no temporary path. Re-evaluate only if document 02's ownership table is revised through an ADR; that revision supersedes the affected §G example, not the rule.
