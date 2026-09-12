# ADR 0014 — A windowed layer reserves bounded tentative-undo headroom, and refuses beyond it

- ID / date / author / status: 0014 / 2026-09-12 / implementation agent /
  **proposed with [task 0017](../../tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md)**.
- Classification: measured implementation choice, inside an accepted owner.
  It resolves no owner gate and changes no numerical, precision, context or
  compatibility contract.
- Scope and owning shared component: `moxie-state`'s paged host KV store. It
  binds `moxie-engine`'s admission, because the headroom must be admitted
  before use.
- Supersedes / superseded by: nothing. It carries forward task 0013's single
  transaction owner and task 0004's transaction mechanism.

## Problem and mechanism

A sliding-window layer must stop retaining history it can never see again, or
the window buys no memory. Reclamation and rollback then collide: `abort` and
`rollback_to` restore a committed prefix, and the rows that prefix needs may
already have been overwritten by the very appends being undone.

The pinned legacy runtime resolves this and gets it wrong.
`src/models/gemma4/gemma4_runtime.cpp:914` evicts from the front of a sliding
layer and advances `cache.start`. `:1385` marks a rewind point by saving
**exactly one** evicted row — `first_key` and `first_value`, and only when the
cache is exactly `sliding_window` rows long. `:1407` reinserts that single row
whenever `cache.start` has moved at all, by any amount.

That is correct only for its single caller, `future_entropy`, which forwards one
candidate token at a time. For any deeper rewind it restores one stale row and
then reports success, which is document 04's "silently change results" —
the failure mode is a wrong answer, not an error. Nothing in the legacy type
records how far back a rewind is actually recoverable.

Document 04 states the requirement directly: "Sliding-window reclamation must
preserve state needed for allowed regeneration/branches; if history has been
released, recompute or report that the requested operation needs re-prefill.
Do not silently change results."

## Options examined

**A. Retain everything; window only masks.** What task 0016's reduced graph
does today, recorded as `Reduction::sliding_layers_retain_full_history`.
Semantically exact and trivially rollback-safe. Memory is the whole context per
layer, so it defeats the window: at the artifact's 1,024-token window and
262,144-token context a sliding layer would hold 256 times what it can see.
Rejected as the thing this task exists to remove.

**B. Snapshot the evicted rows into a side buffer on demand.** Exact for any
rewind depth. The buffer's size is the number of rows evicted while a
transaction is open, which is not known at admission time; a prefill chunk can
evict a whole window. An unbounded or lazily grown snapshot is a second store
with its own allocation, which the ownership rules forbid. Rejected.

**C. Bounded headroom in the ring, with an explicit refusal beyond it.** Give a
windowed layer `capacity = window + tentative_rows` physical rows, admitted up
front in the one envelope. Never let a caller read above the window, so the
headroom is invisible undo space. Refuse — before writing any byte — an append
that would make one transaction longer than `tentative_rows`. Refuse
`rollback_to` a prefix whose window has been reclaimed. **Selected.**

**D. Evict only at page granularity, with no explicit bound.** Reclaiming a page
only when all its rows are outside the window gives free headroom of up to
`page_tokens − 1` rows. It is cheaper to state but the headroom is an accident
of the page size rather than a declared property, and a transaction longer than
one page still corrupts silently. Rejected as A's problem at a smaller scale.
Page-granular reclamation falls out of C's ring anyway.

## Decision and authority

A windowed layer's physical capacity is `ceil((window + tentative_rows) /
page_tokens) · page_tokens`, admitted with the rest of the envelope. Rows above
the declared window are never readable, so the headroom cannot be observed as
history.

With `base` the executed frontier when a transaction began, an append is refused
before writing when it would make `rows − base > tentative_rows`. Given
`capacity >= window + tentative_rows`, every row in `[base − window, base)`
satisfies `r + capacity >= rows_max`, so `abort` restores the entire readable
window without a snapshot and cannot fail for capacity reasons.

`rollback_to(p)` is refused when `p.saturating_sub(window) < evicted`, where
`evicted = high_water.saturating_sub(capacity)` and `high_water` is the greatest
executed frontier ever reached and never decreases. The refusal names re-prefill.

This is an ordinary technical choice inside `moxie-state`'s existing ownership.
It needs no owner approval, and it takes none: no gate is resolved, no precision
or context target moves, and `tentative_rows` is a declared admission input, not
a hidden default.

## Evidence and acceptance

The oracle is independent of the store: `moxie-oracles`' dense `KvHistory` gains
an absolute base position and evicts by position, while the store evicts by ring
overwrite. The predeclared criterion is **exact equality** — evicting outside the
declared window changes no output bit, so a windowed run and a full-retention run
of the same graph produce byte-identical logits and tokens. No tolerance is
introduced or relaxed.

Task 0017's acceptance carries the gates: parity across page boundaries and the
ring wrap, abort exactness with the ring full at every publication boundary, the
two reclamation refusals each with a negative control, and an allocation
reconciliation reporting the envelope with and without windowing at the same
context.

Limitations, stated rather than discovered later. `tentative_rows` bounds one
transaction, so a prefill chunk larger than it is refused and the caller must
chunk smaller or admit more headroom; the engine sets it from the request's
chunk width for exactly that reason. Reclamation is per layer and per sequence:
there is one root branch and no COW, so no shared page can be reclaimed out from
under another branch — when forks arrive, page ownership must be established
before this rule extends to them. Nothing here applies to recurrent or index
state, which cannot truncate at all.

## Enforcement and removal

The refusals are typed errors with their own variants, asserted by tests with
negative controls, not comments. `capacity >= window + tentative_rows` is
checked at construction and is the invariant the abort argument rests on.
`high_water` is monotone by construction and is exercised by a rollback test
that reaches past a reclaimed window.

There is no temporary path and no expiry: option A's full retention is deleted,
not flagged off. Re-evaluate when COW forks arrive (a shared page needs an
owner before it can be reclaimed), when device-resident paged state arrives
(the ring becomes a device kernel's addressing contract), or if a real workload
shows `tentative_rows` forcing chunk widths that cost more than the headroom
saves — that would be a measured argument for option B with an admitted bound,
not for silent undo.
