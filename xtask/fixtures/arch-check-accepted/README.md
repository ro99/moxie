# Positive fixtures — crates that must be **accepted**

`arch-check/` holds crates that must be rejected. This directory holds the other half: model crates
that do only what document 02 permits, and which the checker must therefore pass with zero
violations.

Both halves are needed. A checker with only negative fixtures is satisfied by `return one_violation`,
which rejects the entire workspace; a checker with only positive ones is satisfied by returning
nothing. The fourth M0 review asked for these specifically, alongside the negative cases for
function-local imports.

Keep them minimal and realistic. A fixture here is a claim that this exact shape of model crate is
allowed, so widening one is the same kind of change as widening the allowlist.
