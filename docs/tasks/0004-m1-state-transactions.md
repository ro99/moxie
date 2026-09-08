# Task 0004 — M1.4, part 1: sequence state transactions

Status: **active**. Proposed and started 2026-09-08, after
[task 0003](0003-m1-bf16-reference-interpreter.md) was accepted at `0908f4d`.

**The contract below was written and committed before any implementation code**, as in task 0003.
That commit contains no `.rs` change.

This is one part of document 06's M1.4, not all of it. M1.4 also wants sampler history, deterministic
distribution tests, a generation service and a diagnostic CLI. Those are separate tasks that consume
this one; combining them is how a slice becomes a campaign.

## Identity and authority

- Task ID / milestone / owner: 0004 / M1.4 / implementation agent (Claude), owner review pending
- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, base commit `2ddc014`
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`
- Findings repaired: the limitation recorded at the end of task 0003 and named by its reviewer —
  publication is not transactional, and "leaves state untouched" rests on two separately-argued
  paths rather than one mechanism.
- Required documents: 02 (the `SequenceState` sketch), 04 (transactions, the two frontiers, R08,
  R20), 07 (state correctness ladder); AGENTS.md.
- Owner gates: **none needed.** O1, O2, O4 and O5 are OPEN and none of them gates this: no
  checkpoint, no quantization, no cache dtype choice, no storage write. **Stop and ask** if the work
  appears to need one.

## Why this before the manifest reader

Task 0003 shipped with an acknowledged hole. `moxie-state` has no way to undo `executed`, so a
failure after that point cannot be unwound. Four review passes reached that same defect from four
directions — a layer mismatch, a zero-layer cache, a stale cache, an unchecked precondition — and
each was closed by adding a check *before* the irreversible step. That works, and it is the wrong
shape: it makes correctness depend on having enumerated every way the remaining calls could fail.

The reviewer's closing note is the requirement: make publication transactional, and keep the
existing regressions as acceptance tests. This closes the class rather than the next instance of it.

## Bounded deliverable

One outcome: **a step's publication is a single transaction that either commits entirely or leaves
the sequence and its cache exactly as they were.**

- Sole owning shared component: `moxie-state`, plus the publication path in `moxie-interp`.
- Non-goals: no paging (M4), no copy-on-write fork implementation, no recurrent snapshot/replay
  machinery beyond the evidence rules that already exist, no sampler, no service, no CLI, no CUDA,
  no checkpoint.
- Existing consumers: `moxie-interp` is the only one. Its 25 acceptance tests must keep passing
  unchanged in intent — where they assert "nothing moved", they must now be asserting it of the
  transaction rather than of a hand-checked precondition.
- Temporary paths to delete: the pre-`execute` precondition checks in `Interpreter::run` that exist
  only because publication could not be undone. The layer-coverage and stateless-graph checks
  **stay** — those are statements about whether a graph and a cache belong together, which is a
  different question from atomicity and is worth answering early regardless.

## Contract, fixed before implementation

### The transaction model

Document 02 sketches it and document 04 defines the semantics:

> `begin` creates tentative state; `fork` creates copy-on-write branch state; `commit_prefix(n)`
> publishes exactly n accepted token transitions; `abort` restores the committed prefix.

```rust
fn begin(&mut self, branch: BranchId) -> Result<StateTransactionId>;
fn commit_prefix(&mut self, txn: StateTransactionId, accept: u64) -> Result<()>;
fn abort(&mut self, txn: StateTransactionId) -> Result<()>;
```

- **`begin`** opens a transaction on one branch and records an undo journal. At most one may be open
  per branch; a second `begin` is `InvalidRequest`, because two overlapping journals cannot both be
  the truth about what to restore.
- Inside a transaction, `execute` and `record_logits` do tentative work. This is document 04's
  "a speculative branch may materialize unaccepted candidates beyond the committed prefix" — the
  transaction is what makes that safe rather than merely permitted.
- **`commit_prefix(txn, n)`** accepts `n` tokens into history and closes the transaction, keeping
  the executed state. `n = 0` is the ordinary decode case: the interpreter executes, and acceptance
  is the caller's decision after sampling.
- **`abort`** restores the branch to exactly its state at `begin` and closes the transaction.

### What "exactly" means

The undo is a **journal, not a copy**. Every mutation inside a transaction is monotone — counters
only rise, the lineage vector only grows, results are only added — so the journal is small and the
restoration is exact rather than approximate:

| Recorded at `begin` | Restored on `abort` |
|---|---|
| the four frontier counters | assigned back |
| `lineage.len()` and `epoch` | vector truncated, epoch assigned back |
| the branch's retained `LogitsHandle` | assigned back |
| `next_result` | every result minted during the transaction is removed from the live set |

A full clone of the branch would also be exact, but its lineage vector is `O(context)`, so cloning
per step would make a sequence `O(n²)`. The journal is `O(1)` to record and `O(changed)` to apply.
Stated because the cheap-looking option is the wrong one here and a later reader should not
"simplify" it back.

**`abort` is infallible** except for an unknown transaction id. Restoration is assignment and
truncation; nothing in it can run out of anything. This is what lets the interpreter abort on a
failure path without a second failure to handle.

### What the transaction does not cover, and why

- **The KV cache is not inside `moxie-state`.** Ownership forbids it: `moxie-state` owns "paged
  sequence state, forks, transactions, rollback and prefix reuse" and must not reach into a
  consumer's buffers. `KvCache` gets its own `begin`/`abort` with the same journal shape — the
  recorded per-layer lengths and stamp count — and `moxie-interp` is the composition point that
  opens and resolves both together. One mechanism with two participants, which is the honest
  structure; a single mechanism owning both would put device buffers under the state crate.
- **A dropped, unresolved transaction leaves the branch mutated but locked.** Rust's `Drop` cannot
  reach the `SequenceState` that owns the journal, so an unresolved transaction cannot auto-abort.
  Instead it is *detectable*: the branch refuses a further `begin`, and `SequenceState::open_transactions()`
  reports it. That is a leak the tests assert on rather than a silent corruption, and the closure-
  scoped alternative is rejected because it would force the interpreter's borrows into a shape that
  makes the cache participant awkward.
- **`fork` remains what it is.** Copy-on-write branch state is M4; `fork` still creates a branch with
  its own identity and no inherited result, and transactions are per branch.

### Cancellation

R08: "Memory leases are released at turn boundaries and cancellation even if no next token arrives."
Cancellation becomes an abort rather than an early return that happens to precede the mutations. The
existing test that cancels at every node depth keeps its assertions and gains meaning: it is now
checking the abort path rather than checking that nothing had started yet.

### Error metrics

None. This task changes no arithmetic. The numerical contracts in task 0003 are unchanged, and any
test here that touches them is asserting they are *unchanged*, not re-deriving them.

## Acceptance

- `cargo xtask arch-check`, `spec-check`, `fmt`, `clippy -D warnings`, and the full host lane pass.
  The device lane and `test-gpu` are unchanged and carried forward; this task touches no CUDA.
- **Failure injection at every publication step.** A test drives the interpreter with a fault
  injected at each point publication can fail, and asserts after each that the frontiers, the
  retained result, the live result set, the lineage, and every KV layer are identical to before.
  This is the acceptance test the previous four review passes each found a hole in, generalised.
- Cancellation at every node depth still leaves state identical, now through `abort`.
- A second `begin` on a branch with an open transaction is refused; `commit_prefix` and `abort` with
  an unknown id are refused; an unresolved transaction is visible.
- `abort` restores `executed` — the thing that was impossible before, asserted directly.
- Every task 0003 acceptance test passes unchanged in intent.
- Support matrix: update `G-INTERP-BF16`'s row and add nothing that mentions a checkpoint, a kernel
  or a context length.
- Stop condition: if this appears to need paging, a real fork, a sampler or a checkpoint, stop and
  report. Each is a different task.

## Result, filled after work

*(to be completed)*
