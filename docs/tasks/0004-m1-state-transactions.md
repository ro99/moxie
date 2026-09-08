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

## Result

Implemented on `main`, base commit `42e1e9c` (the contract commit above, which contains no `.rs`
change). Nothing in the contract was adjusted once a test ran.

### Commands

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | **PASS** |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | **PASS** |
| `cargo test --workspace --locked --offline` | **PASS**, 325 unit/integration + 1 doctest (was 312 + 1) |
| `cargo xtask arch-check` | **PASS**, 19 rejected + 1 accepted fixtures, 6 rules |
| `cargo xtask spec-check` | **PASS**, 10 documents, digests unchanged |
| no-driver host lane, `CUDA_HOME=/nonexistent NVCC=/nonexistent`, no CUDA on `PATH` | **PASS**, 325 + 1; `ldd target/debug/xtask` reports no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | **PASS**, 333 + 2 doctests |
| `cargo xtask-cuda test-gpu` | **PASS**, 15 cases, `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, `UNQUALIFIED sm_120`, as intended |

Host counts by crate: `moxie-oracles` 114, `moxie-format` 54, `moxie-state` 38, `xtask` 30,
`moxie-interp` 14 + 27 integration, `moxie-types` 22, `moxie-graph` 11, `moxie-cuda` 9,
`moxie-model-api` 5, `moxie-kernels` 1.

### What was built

**`moxie-state` gained the transaction.** `begin` records a `Journal` — the four frontier counters,
`lineage.len()`, the epoch, the retained `LogitsHandle`, and `next_result` — and returns a
`StateTransactionId`. `commit_prefix(txn, n)` accepts `n` tokens and closes it; `abort(txn)` restores
and closes it. Both refuse an unknown or already-resolved id. `open_transactions()` reports what is
open, so an unresolved transaction is a visible leak rather than silent drift.

One thing the contract's table understates, found while writing the tests: truncating the lineage to
its recorded length is not sufficient, because `rollback_to` inside a transaction *shortens* the
vector, and a later append then overwrites entries the journal assumed were untouched. `begin`
therefore also records `lineage_saved_from`/`lineage_saved`, the suffix that any in-transaction
rollback discarded, and `abort` puts that suffix back before truncating. This is still `O(changed)`
rather than a clone — nothing is copied unless something was discarded — and
`abort_restores_a_lineage_that_was_rolled_back_twice_inside_the_transaction` is the case that
required it.

**`KvCache` gained the matching journal.** `begin()` returns a `CacheJournal` holding the per-layer
lengths and the stamp count; `abort(&journal)` is infallible and truncates back to them. `commit`
stays `pub(crate)` — no caller outside the crate may re-stamp a cache — and now carries the check
the interpreter used to do before `execute`: every layer must hold exactly the executed prefix, both
halves of it, because `len()` reads layer 0 alone and a ragged cache would otherwise be stamped as
current on the strength of one layer.

**`moxie-interp` composes the two.** `Interpreter::run` opens `kv.begin()` and `state.begin(branch)`,
calls the new `publish`, and on any error calls `state.abort(txn)` then `kv.abort(&journal)` — neither
of which can fail. `publish` does the appends, `state.execute`, `kv.commit` and `state.record_logits`
in the order the work happens, not in an order chosen to put the irreversible step last.

### Temporary paths deleted, as the contract required

The pre-`execute` precondition checks that existed only because publication could not be undone are
gone: the `expected_prefix` overflow check, the manual `truncate_layers` on each append failure, and
the "check the cache length before `execute` because `commit` must not fail" block. The comment
block arguing that nothing after `execute` could fail is gone with them. What stays, because it is a
statement about whether a graph and a cache belong together rather than about atomicity: the
layer-count check and the stateless-graph refusal, both still before anything is written.

### Acceptance, item by item

| Requirement | Evidence |
|---|---|
| `arch-check`, `spec-check`, `fmt`, `clippy -D warnings`, full host lane | all PASS, above |
| Device lane and `test-gpu` unchanged and carried forward | PASS; this task touches no CUDA |
| Failure injection at every publication step | `a_partly_published_step_aborts_to_exactly_where_it_started` drives a step's mutations one at a time and aborts after each, comparing the four counters, the retained result, the live set, the lineage at every prefix and every KV layer; `abort_restores_after_every_prefix_of_a_step_s_mutations` does the same at the state level |
| Cancellation at every node depth still leaves state identical, through `abort` | `a_cancelled_step_leaves_the_state_exactly_as_it_found_it`, unchanged in intent, plus `a_successful_step_leaves_no_transaction_open` which also asserts a cancelled step leaves none |
| A second `begin` is refused | `a_branch_has_at_most_one_open_transaction` |
| Unknown / resolved id refused | `an_unknown_or_resolved_transaction_is_refused` |
| An unresolved transaction is visible | `an_unresolved_transaction_is_visible_rather_than_silent` |
| `abort` restores `executed` | `abort_restores_every_counter_including_executed`, asserted directly |
| Every task 0003 acceptance test passes unchanged in intent | 27 integration tests pass; the two added are new, none were weakened |
| Support-matrix row updated | `G-INTERP-BF16` recounted; one capability row added for transactional publication. No checkpoint, kernel or context row touched |

Further tests beyond the list: `commit_prefix_keeps_the_work_and_publishes_what_it_is_told_to`,
`abort_puts_back_the_epoch_so_a_later_lineage_is_unchanged`,
`a_failed_commit_leaves_the_transaction_open_to_abort`,
`a_branch_with_an_open_transaction_cannot_be_discarded`, and
`a_cache_whose_layers_disagree_cannot_be_stamped_as_current`.

### What this does not establish

- **This is not a fork.** `fork` still creates a branch with its own identity and no inherited
  state. Copy-on-write branch state, paging and prefix reuse are M4 and untouched.
- **Nothing here executes a model.** The interpreter is still a host BF16 reference over synthetic
  graphs. Atomic publication makes a step safe to fail; it does not make anything faster, and it
  does not make the reference an engine.
- **The rest of M1.4 is not done.** Sampler history, deterministic distribution tests, the
  generation service and the diagnostic CLI are separate tasks that consume this one.
- **A dropped transaction still leaks.** By design, and asserted: `Drop` cannot reach the owning
  `SequenceState`, so an unresolved transaction locks its branch and is reported by
  `open_transactions()` rather than auto-aborting.
