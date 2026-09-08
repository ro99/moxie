# Task 0004 — M1.4, part 1: sequence state transactions

Status: **accepted 2026-09-08** at `3aaf259`, after three review passes and their corrections.

The acceptance is of **this bounded slice** -- host-side transactional publication over sequence
state and the KV cache -- and explicitly **not of M1.4**. Appendable paged state, sampler
integration, the generation service and the diagnostic CLI remain outstanding. The reviewer said so
in those words; it is recorded here so a later reader cannot mistake this record for a milestone.

Accepted with CUDA gates not rerun by the reviewer. They were rerun here at `3aaf259`
(`test-gpu` PASS, hidden-SM120 exit 1); the review's independent lanes were fmt, clippy,
`arch-check`, `spec-check`, whitespace and the 332 + 4 host tests.

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
- Non-goals: no paged allocator, no copy-on-write fork implementation, no recurrent snapshot/replay
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
- **Paging is not deferred to M4.** M1.4 asks for "appendable paged state **plus** transaction API",
  so the initial paged implementation is an M1 requirement; M4 adds paged *device* attention,
  host-backed page streaming and the growth-admission path on top of it. This task deliberately
  narrows to the transaction half, which is the part the task-0003 finding is about. The paged
  allocator is the next M1.4 task and is not being re-scoped to M4 by being excluded here.

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
- Stop condition: if this appears to need the paged allocator, a real fork, a sampler or a
  checkpoint, stop and
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

Host counts by crate, after the seventh review's correction: `moxie-oracles` 114, `moxie-format` 54,
`moxie-state` 39, `xtask` 30, `moxie-interp` 19 + 27 integration, `moxie-types` 22, `moxie-graph` 11,
`moxie-cuda` 9, `moxie-model-api` 5, `moxie-kernels` 1; 4 doctests, 3 of them `compile_fail`.

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
| Failure injection at every publication step | `a_step_cancelled_at_any_publication_boundary_aborts_to_exactly_where_it_started` drives the real `Interpreter::run` and injects a fault after each of publication's four mutations, comparing the four counters, the retained result, the live set, the lineage at every prefix, every KV layer **and the cache stamps**; `abort_restores_after_every_prefix_of_a_step_s_mutations` does the same at the state level. **Rewritten after the fifth review**, which was right that the first version hand-performed a subset of the mutations and hand-called `abort`, so it never exercised `run`'s error handler |
| Cancellation at every node depth still leaves state identical, through `abort` | `a_cancelled_step_leaves_the_state_exactly_as_it_found_it`, unchanged in intent, plus `a_successful_step_leaves_no_transaction_open`. **Corrected after the fifth review**: cancellation during node evaluation happens before `begin`, so until that review it satisfied the rule by never having started. `publish` now checks the token after each of its four mutations, so cancellation genuinely aborts |
| A second `begin` is refused | `a_branch_has_at_most_one_open_transaction` |
| Unknown / resolved id refused | `an_unknown_or_resolved_transaction_is_refused` |
| An unresolved transaction is visible | `an_unresolved_transaction_is_visible_rather_than_silent` |
| `abort` restores `executed` | `abort_restores_every_counter_including_executed`, asserted directly |
| Every task 0003 acceptance test passes unchanged in intent | 27 integration tests pass; the two added are new, none were weakened |
| Support-matrix row updated | `G-INTERP-BF16` recounted; one capability row added for transactional publication. No checkpoint, kernel or context row touched |

Further tests beyond the list: `commit_prefix_keeps_the_work_and_publishes_what_it_is_told_to`,
`a_transaction_is_append_only`, `a_failed_commit_leaves_the_transaction_open_to_abort`,
`a_branch_with_an_open_transaction_cannot_be_discarded`, and
`a_cache_whose_layers_disagree_cannot_be_stamped_as_current`.

### What this does not establish

- **This is not a fork, and it is not the paged allocator.** `fork` still creates a branch with its
  own identity and no inherited state. Copy-on-write branch state and prefix reuse are M4;
  **appendable paged state is M1.4 and still outstanding** -- excluded from this narrower task, not
  reassigned to a later milestone.
- **Nothing here executes a model.** The interpreter is still a host BF16 reference over synthetic
  graphs. Atomic publication makes a step safe to fail; it does not make anything faster, and it
  does not make the reference an engine.
- **The rest of M1.4 is not done.** Sampler history, deterministic distribution tests, the
  generation service and the diagnostic CLI are separate tasks that consume this one.
- **A dropped transaction still leaks.** By design, and asserted: `Drop` cannot reach the owning
  `SequenceState`, so an unresolved transaction locks its branch and is reported by
  `open_transactions()` rather than auto-aborting.

---

## Fifth review corrections, 2026-09-08

A fifth review of `1166779` reproduced **five transaction-correctness defects and one acceptance-test
gap**, and was right about all six. Every one was reproduced here before being changed; the
reproductions are now permanent regression tests rather than scratch files.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 325 + 1 doctest | **328 + 1 doctest** (4 tests added, 1 removed) |
| device lane | 333 + 2 doctests | **336 + 2 doctests** |
| `cargo test -p moxie-interp` | 14 unit + 27 acceptance | **15 unit + 27 acceptance** |
| `fmt`, `clippy -D warnings`, `arch-check`, `spec-check`, no-driver host build | PASS | **PASS** |
| `cargo xtask-cuda test-gpu`, hidden-SM120 gate | PASS / exit 1 | **PASS / exit 1** |

### The shape all five share

Four of the five are the same mistake in different places: **`abort` was written as "put the
recorded values back", when what it has to be is "leave the branch as if the transaction had never
run".** Those differ exactly where a mutation touched something the journal does not describe --
another branch's results, a counter that is not the branch's, output that has left the process, or
content that a destructive operation removed. Restoring a field is not the same as undoing an
effect, and each defect is one place where that gap was visible.

### R1 -- aborting one branch deleted another branch's committed results · closed

Reproduced: begin on root and on a child, execute and `commit_prefix` the child, abort the root; the
child's committed handle stopped validating. `abort` filtered the whole sequence's `live` table by
result id, and the child's ids are also above the root journal's mark.

The predicate now names both halves -- `id >= results_from` **and** `branch == journal.branch`.
`aborting_one_branch_leaves_another_branch_s_results_alone`.

### R2 -- aborted result identities were reissued · closed

Reproduced: abort a transaction that minted a handle, retry the same prefix, and the new handle was
**equal in every field** to the discarded one; `restore_logits` accepted the dead handle again.
`abort` was restoring `next_result`.

`next_result` is now recorded as `results_from` and used only to *identify* what to remove. It is
never restored, so identities are not recycled and an aborted result stays dead.
`an_aborted_result_identity_is_never_reissued`.

### R3 and R5 -- a transaction is now append-only · closed

R3: a rollback inside a transaction discarded live results, and the journal saved discarded lineage
but not those; aborting left counters at 2 and the retained result gone. R5: `KvCache::abort` only
truncates, so it cannot put back rows a rollback removed -- abort left the sequence at prefix 2 and
the cache at 1.

Both are the same operation, and the review offered the resolution for the cache half: journal the
discarded content, or refuse the destructive mutation while a transaction is open. **Refusing** is
what is implemented, for both participants, because it is exact rather than approximately exact, it
is cheap, and document 04's mechanism for keeping part of a transaction's work is `commit_prefix(n)`
rather than a rollback inside it. `rollback_to` is refused on `SequenceState` and on `KvCache` while
a transaction is open, and `invalidate_generation` is refused outright because it clears every
branch's retained result and no journal records what it removed.

The journal is smaller as a result: `lineage_saved`/`lineage_saved_from` are gone, and the test that
existed only to exercise them is replaced by `a_transaction_is_append_only` and
`a_cache_transaction_is_append_only`, which assert the refusal and that both operations work again
once the transaction resolves.

### R4 -- abort could retract emitted tokens · closed

Reproduced: `emit` inside a transaction moved `emitted` 0 -> 1, and `abort` moved it back. A counter
can be restored; text the client already has cannot. The previous test
`abort_restores_every_counter_including_executed` asserted this behaviour, so it was not merely
untested -- it was enshrined.

`emit` is refused while a transaction is open on the branch. The test now asserts the refusal and
that `emitted` comes back **because it never moved**, which is a different guarantee from being
rewound. `tentative_work_cannot_be_released_to_the_client`.

### R6 -- the failure injection was not through the real path · closed

The review was right on both counts. The old test opened both journals by hand, performed a subset
of the mutations by hand, called `abort` by hand, and never touched cache stamping -- so it checked
the *participants* and not `Interpreter::run`'s error handler. And the record's claim that
cancellation now exercises abort was inaccurate: cancellation was checked only during node
evaluation, before `state.begin`.

Both are fixed by the same change. `publish` checks the cancellation token after each of its four
mutations, so cancellation is a genuine mid-publication failure. The new test derives the graph's
boundary count rather than hardcoding it, then drives the real `run` cancelled at each publication
boundary in turn and asserts the four counters, the retained result, the live set, the lineage at
every prefix, every KV layer and `check_owner` (the stamps) are unchanged, that no transaction is
left open, and that the sequence still runs cleanly afterwards.

Cancellation is the injected fault because it is the only one left: once the preconditions moved
inside the transaction, no malformed input can make publication fail halfway, so a test that fed bad
input would be refused at entry and never reach the handler it claims to check. That is stated in
the test rather than left for the next reviewer to work out.

The test was checked against two deliberate mutations -- dropping `kv.abort` and dropping
`state.abort` from `run`'s error arm -- and it fails on each, at a different assertion.

### Milestone wording, corrected

The record described paging as M4. M1.4 asks for "appendable paged state **plus** transaction API",
so the initial paged implementation is an M1 requirement; M4 adds paged *device* attention,
host-backed page streaming and growth admission on top of it. This task narrows to the transaction
half deliberately. The support matrix now says the same.

---

## Sixth review corrections, 2026-09-08

A sixth review of `1eca710` accepted the five transaction fixes and the rewritten failure-injection
test, and found **one remaining blocker**: `CacheJournal` carried no identity. It was reproduced and
closed.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 328 + 1 doctest | **330 + 3 doctests** |
| device lane | 336 + 2 doctests | **338 + 4 doctests** |
| `fmt`, `clippy -D warnings`, `arch-check`, `spec-check`, no-driver host build | PASS | **PASS** |
| `cargo xtask-cuda test-gpu`, hidden-SM120 gate | PASS / exit 1 | **PASS / exit 1** |

### A journal was data; it is now an authority · closed

`CacheJournal` was `Clone` and held only lengths and a stamp count. `commit` ignored its contents and
`abort` took it by reference, so nothing tied a journal to the cache or the transaction it came from.
Both consequences reproduce:

- **A resolved journal deleted committed rows.** Take a journal from an empty cache, commit it, then
  append two rows and execute. Applying the saved journal afterwards truncated the cache to zero
  while the sequence stayed at prefix 2 — the exact divergence the transaction exists to prevent,
  produced by the mechanism meant to prevent it.
- **A stale journal resolved a newer transaction.** Open a real transaction, then `commit` the old
  journal: `in_transaction` cleared, and `begin` succeeded again while the real transaction was
  still outstanding. The append-only rule from the fifth review was unlocked by the same move.

A journal is now an authority to undo **one transaction on one cache**, and three things enforce it,
none of them sufficient alone — which is the review's point:

| | Enforces |
|---|---|
| `CacheJournal` carries a process-unique `CacheId` | a journal from another cache is refused |
| it carries the transaction number, and `next_txn` is monotone | a journal for a resolved or superseded transaction is refused; a number is never reissued |
| it is not `Clone`, and `commit`/`abort` take it **by value** | it cannot be duplicated or applied twice |

`check_journal` runs **before any mutation**, so a rejected journal changes nothing; that is
asserted rather than argued. `commit` and `abort` now return `Result`, and `Interpreter::run`
`expect`s on both because it opened the journal on that cache three lines earlier — the same shape
as the existing `state.abort(txn).expect(...)`.

### What is tested, and how

- `a_journal_from_another_cache_is_refused_before_it_mutates_anything` — the runtime case, asserting
  the contents are untouched and that the cache's own transaction is still open and still
  resolvable afterwards.
- `a_journal_cannot_resolve_a_transaction_that_is_not_open` — reaches the third arm through the
  private constructor, because resolving consumes the journal and safe code cannot get there.
- Two `compile_fail` doctests for the cases the type system now rules out: cloning a journal, and
  applying one twice. Both were checked with `compile_fail` removed, and each fails with exactly
  one error — `E0599 no method named clone` and `E0382 use of moved value` — rather than passing for
  an unrelated reason.

### Task 0005 wording, corrected

The review was right that the contract claimed all three opening limits are enforced before
deserialization. Only the manifest's file size is; the tensor-entry cap and the
architecture-metadata depth and node counts are necessarily checked while traversing the parsed
value. The table now says which is which.

---

## Seventh review corrections, 2026-09-08

A seventh review of `6b7f5ad` accepted the journal-identity work and found **one remaining blocker**:
the journal could no longer be duplicated, but the cache holding its authority still could.

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | 330 + 3 doctests | **332 + 4 doctests** |
| device lane | 338 + 4 doctests | **341 + 4 doctests** |
| `fmt`, `clippy -D warnings`, `arch-check`, `spec-check`, no-driver host build | PASS | **PASS** |
| `cargo xtask-cuda test-gpu`, hidden-SM120 gate | PASS / exit 1 | **PASS / exit 1** |

### `KvCache` was `Clone`, so `CacheId` was not unique · closed

Reproduced exactly as the review described: clone an empty cache A into B, execute two tokens
through B, open the first transaction on each -- the numbers match because the counter was copied
too -- and apply A's journal to B. B accepted it and truncated its committed rows to nothing while
the sequence stayed at prefix 2, and B's own journal then failed because its transaction had been
resolved by a cache that was not it.

The sixth review's fix put a process-unique identity on the journal. A derive then handed the same
identity to a second object, which is the same failure the third M0 review found on
`SequenceState` -- and the reason that type carries a `compile_fail` doctest. An identity that
exists to be unique cannot be duplicated by a derive.

`Clone` is removed. `PartialEq` goes with it: two caches holding identical bytes are legitimately
different caches, so the comparison a test wants is `contents()`, and leaving `==` in place invites
a comparison that is now always false.

### Copying contents is still a real need, so it is explicit

Two tests genuinely need a saved cache -- one of them is the fifth review's own
`a_stale_cache_cannot_be_certified_by_rolling_it_back`, which cannot demonstrate that stale bytes
resist laundering without saving stale bytes. `KvCache::snapshot` is the operation:

- copies what the cache **holds** -- the layers and the owner's sequence, branch and per-prefix
  stamps;
- mints a **fresh `CacheId`** and its own transaction counter, so no journal is ever valid for both;
- is **refused while a transaction is open**, because a snapshot taken mid-transaction would copy
  tentative rows and outlive the abort meant to remove them.

The third call site did not need a copy at all -- it compared whole caches when it meant contents --
and now takes `contents().to_vec()`.

### Tested

- `a_snapshot_is_a_different_cache_and_does_not_share_journals` -- the reproduction, asserting the
  ids differ, the foreign journal is refused, nothing is truncated, and the cache's own journal
  still resolves afterwards.
- `a_snapshot_is_refused_while_a_transaction_is_open`.
- A third `compile_fail` doctest, for `KvCache` not being `Clone`. All three were re-checked with
  `compile_fail` removed, and each fails with exactly one error -- `E0599` for the two `clone` calls
  and `E0382` for the double application -- rather than passing for an unrelated reason.

Also swept the two crates for other derived `Clone` on a type that carries identity or authority.
The remaining ones are `Branch`, `Journal` and `CacheOwner`, all private, none of which escapes the
type that owns it; `Journal` is cloned only to read its fields while `self.open` still holds the
canonical entry, and resolution is the `remove` rather than the clone.
