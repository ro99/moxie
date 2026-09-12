# ADR 0013 — One model crate with a module per family

- ID / date / author / status: 0013, 2026-09-12, implementation agent; **accepted with task 0016 on 2026-09-12**.
- Classification: measured implementation choice.
- Scope and owning shared component: `crates/moxie-models`, and `arch-check`'s
  model-crate rules in `xtask/src/archcheck.rs`.
- Supersedes / superseded by: neither. Narrows the packaging assumption M0's
  architecture checker was written with.

## Problem and mechanism

M0's `arch-check` identified a concrete model adapter by the crate-name prefix
`moxie-models-`, so every family implied its own crate. Adding the first real
family made the cost concrete: a manifest, a workspace member, a
`[workspace.dependencies]` entry, an `arch-check` allowlist row and a build unit,
per model, for seven planned families.

The owner's objection during implementation was that this is ceremony, and it is
correct. It is worth being precise about what the ceremony was protecting.

The boundary document 02 draws is a **dependency list**: a model definition may
depend on `moxie-types`, `moxie-graph` and `moxie-model-api` and nothing else,
and document 09 §B lists the code it may not contain. That boundary is enforced
per crate, and a module cannot import what its crate does not depend on. So a
single crate holding every family enforces exactly the same rule as a crate per
family. The prefix was never the mechanism; it was how the checker found the
crates to apply the mechanism to.

## Options examined

**1. A crate per family** — the M0 assumption. Rejected: four files of ceremony
per model, buying no enforcement a module boundary inside one crate does not
already provide. Its one genuine advantage is per-family dependency granularity,
which option 3 keeps.

**2. One crate, families behind features** — `moxie-models` with a
`cargo` feature per family so a build can exclude one. Rejected as premature:
nothing in this product loads models selectively, the whole crate is metadata and
graph composition with no heavy dependencies, and feature-gated modules would
make the architecture fixtures depend on which features were enabled.

**3. One crate, a module per family, with the prefix retained as an escape
hatch** — chosen.

## Decision and authority

Concrete model definitions live in `crates/moxie-models`, one module per family,
holding the same three workspace dependencies a `moxie-models-*` crate would.
Adding a family is a file and a `pub mod` line.

`arch-check` identifies a model crate as `moxie-models` **or** any crate whose
name starts with `moxie-models-`. The prefix is retained deliberately: a family
that needs a dependency the others must not have — a narrowly approved metadata
parser, which document 02 permits case by case — may still take its own crate and
is held to the same list. That stays available without being the default.

`moxie-cli` joins `xtask` as a composition root permitted to depend on the model
crate. Document 02: "Only the composition root/registry and integration tests
may" import concrete models. No other production crate may, and a fixture proves
it.

This is a packaging decision. It changes no dependency rule, no forbidden
construct, no ownership boundary and no owner gate.

## Evidence and acceptance

Three architecture fixtures cover the change, because a rule that stopped
matching the real crate would be worse than the ceremony it replaced:

- `models-crate-reaches-state` — a crate named exactly `moxie-models` depending
  on `moxie-state` must be rejected for `forbidden dependency`. Without this,
  renaming the crate would have silently exempted every concrete model.
- `shared-imports-the-models-crate` — a shared crate depending on
  `moxie-models` must be rejected for `shared crate imports a concrete model`.
- `models-crate-with-family-modules` — one crate, two family modules, three
  allowed dependencies, accepted.

`arch-check` passes 73 rejecting and 21 accepted fixtures across 12 rules,
against a clean archive of the tracked tree.

The existing `moxie-models-*` fixtures are unchanged and still pass, so the
escape hatch is tested rather than merely described.

## Enforcement and removal

`is_model_crate` in `xtask/src/archcheck.rs` is the single place the two names
are recognised, and the three fixtures above fail if it stops recognising either.
The forbidden-construct scan — which is deliberately broad enough to fire on a
comment mentioning the vendor toolkit — applies to every file in the crate, so a
family module cannot reach further than its siblings.

Nothing here expires. If a family ever needs its own crate, it takes one under
the retained prefix; that is a decision about one family's dependencies, not a
reversal of this one.
