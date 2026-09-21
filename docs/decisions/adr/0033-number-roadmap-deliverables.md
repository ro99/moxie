# ADR 0033 — number the roadmap's deliverable bullets as M{n}.{i}

- **Date / status:** 2026-09-20 / decided and applied.
- **Classification:** editorial amendment to `docs/spec/06-implementation-roadmap.md`,
  under the rule in `docs/README.md`: "a reference document is amended only
  through an ADR that says what changed and why."
- **Owner:** whoever tracks milestone progress in `AGENTS.md`.
- **Re-evaluated by:** never, unless a milestone's deliverable list is itself
  restructured (items added, removed or reordered).

## What changed

Every milestone from M0 through M11 lists its deliverables as a plain numbered
list under `Deliver:` — `1.`, `2.`, `3.` … Each bullet is now prefixed with its
milestone, `M{n}.{i}.` — `M4.1.`, `M4.2.`, `M4.3.` … No bullet's text changed;
this is a rename of the enumeration, not a content edit. `M7` (a family table,
not a sequential list) and `M12` (two prose paragraphs, no list) keep their
existing shape.

## Why

`AGENTS.md`'s milestone table has used `M3.1`/`M3.2`/`M3.3` as tracking labels
since M3 closed, and tonight added `M4.1`–`M4.5` for the same reason: the owner
needs to see progress at the granularity of one deliverable, not just one
milestone, especially on a milestone (M4) that spans months and several tasks.

Those labels already corresponded to the roadmap's own numbered bullets —
`M3.1` was always "M3's deliverable 1" — but the correspondence was never
written down at its source. An agent tracking a task against the roadmap had to
count bullets under a heading and infer which one a task record meant. The
owner asked directly whether the `.N` suffix was meant to number these bullets;
it was, and this ADR makes that literal instead of inferred.

## What this does not change

No deliverable's requirement text changed. No milestone gained or lost a
deliverable. `Exit:` paragraphs are untouched. This does not retroactively
re-litigate which tasks satisfied which already-accepted deliverable (M0–M3);
it only gives the bullets a name that matches how they were already being
referenced.

## Enforcement

`cargo xtask spec-check --update` was run after this edit and the new digest is
committed in `docs/evidence/specification-version.md`. A future edit that
changes a deliverable's actual scope still needs its own ADR; renumbering an
added or removed bullet under an existing milestone does not need a second ADR
for the renumbering itself, only for the scope change that caused it.
