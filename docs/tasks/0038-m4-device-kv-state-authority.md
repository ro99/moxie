# Task 0038 — M4.1b the state authority owns the device pages, and the graph reaches them

**Status: accepted and closed** (owner, 2026-09-20). Opened 2026-09-19 after
task 0037's kernel and binding were qualified and twice reviewed. Both bounded
deliverables landed, and the two items moved from task 0037 (the mutation
battery and the measured-allocator lane) are met: 11 of 11 mutations caught,
32 decode steps bounded and a deliberate leak caught. The lifecycle,
selected-plan and 32K device gates pass on both SM86 GPUs and SM120. This
closes M4.1 (roadmap deliverable 1 of 5); M4.2 through M4.5 remain unstarted.

## Identity and authority

- Task0038, second bounded M4 task; repository owner reviews and accepts.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `0fb2128`.
  `coordinator.md` is unrelated carried work and stays outside this task.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; checkpoint roots remain read-only.
- Requirement: the two halves task 0037 left open, named by its record and by
  [its handover](../handovers/2026-09-19-paged-attention-to-state-binding.md).
  Document 04's attention contract — "attention consumes device tensor handles
  and page-table/state handles", host paths "explicit separate implementations,
  not compulsory staging interfaces" — is the specification for the second half.
- Read before editing: `moxie-state`'s `lib.rs` (journal, `StateKind`,
  `RestoreCapability`, transactions, lineage) and `paged.rs` (geometry,
  `Retention`, ring reclamation, `tentative_rows`), ADR 0014, tasks 0013, 0017
  and 0037, `moxie-executor::paged_attention` in full, and `moxie-plan`'s
  `stateful_resource_plan` refusal.
- O1–O5 resolved; O6/O7 open, so nothing here may be timed or called fast.

## What the reading already settled

Two facts were established before this contract was written, and they are what
make it bounded rather than exploratory.

**The host store's ring *is* a page table.** `PagedSequence::ranges` computes
`page = (row / page_tokens) % pages`, so logical page `L` lives at physical page
`L % pages`. The device path's `page_table[L] = L % pages` is the same mapping
written down, not a second scheme. The two reclamation strategies the handover
called an open question are one strategy at two granularities.

**A row-granular window cannot be expressed by a page-granular base.**
`retained_start` is `max(high_water − capacity, rows − window)`: the second term
is row-granular and the kernel already handles it exactly, through
`Visibility::SlidingWindow` on absolute positions. The first term is the ring's
eviction boundary and is *not* page-aligned — at capacity 64 and high water 100,
rows 36..99 are live, and physical page 2 then holds rows 96..99 in its first
four slots and rows 36..47 in its last twelve. A partially overwritten page has
no `history_base` that describes it.

So the device store evicts **whole pages**: a page leaves the retained range as
soon as any of its rows would be overwritten. That costs up to `page_tokens − 1`
rows of capacity and is why the admitted row count must round up. It is a state
decision, which is why it belongs to the state authority and not to a launch.

## Bounded deliverable

Two halves. **Neither closes this task alone**, and the record must say which
landed if only one does.

**1. `moxie-state` owns the device pages.** The crate that owns transactions,
frontiers and retention owns them for device-resident KV as it does for host KV.
It does not own the bytes: it owns the *decisions* and publishes them as data —
where a logical row is placed, what the committed frontier is, which rows are
retained, what a page table contains and when a page leaves the retained range.
`moxie-executor` performs those decisions and owns the allocation and the
launch. No trait object, no callback into the executor, no second frontier:
`PagedAttentionRun::committed_rows` stops being its own counter and starts being
what the authority says.

> **Amended by the owner, 2026-09-20.** The line above — "no trait object, no
> callback into the executor" — required a data-only boundary: the authority
> publishes decisions as data, and the executor reads them. The implemented
> design is a callback, `&mut dyn PagedKvWriter`, called by `moxie-state`. The
> owner's ruling: **the callback stays; this line is amended, not the code.**
> A data-only boundary cannot enforce that publication follows an observed
> write — nothing in "the authority writes a decision, the executor later
> reads it" requires the executor to have performed the write before the
> authority publishes, which is the one thing this whole mechanism exists to
> guarantee. What the callback buys instead: `append` is the only public way
> to move the frontier, `stage` and the publish step are private, and there is
> no value a caller can hold that makes publication happen — the only way to
> reach it is to be one of `writers` and return `Ok`. What it does **not**
> buy, unchanged from a data-only boundary: a writer that returns `Ok` before
> its copy has completed is lying, and the authority has no way to check that
> from outside. This was the owner's decision, not an agent's, and it settles
> the shape; the later lifecycle and visibility findings are closed below.

**2. `OpParams::Attention` reaches it from the graph.** `moxie-plan` refuses
every stateful graph today (`stateful_resource_plan`). That refusal is replaced
by an admitted plan for the attention node — not deleted — and the executor binds
query and output as device handles it already owns rather than host `Vec<u8>`.
The host-staging entry point survives only as document 04's "explicit separate
implementation", used by gates, or not at all.

Out of scope and unsupported by name: MLA, sparse/index attention, host-backed
page streaming, COW forks, prefix reuse, recurrent/convolution state, multi-GPU
partitioning, tensor cores, tuning and any performance claim. FP16 cache remains
unsupported rather than silently BF16.

## Contract before implementation

- The device store's admitted capacity is `window + tentative_rows` rounded **up
  to whole pages, plus one page**, so that evicting a whole page never drops a
  row the window still admits. The three numbers — admitted capacity, committed
  rows, retained rows — stay separately reported, as task 0037's gate already
  requires.
- Placement is `page_table[logical] = logical % pages` and `slot = position %
  page_tokens`, the host ring written as a mapping. A test must show the host
  store and the device placement agree on the physical slot of every row across
  a wrap, because two implementations of one mapping is what this task exists to
  avoid.
- The committed frontier advances only inside a committed transaction. An
  aborted transaction leaves the frontier, the retained range and every prior
  byte unchanged; `RestoreCapability::Truncate` means a truncation to a prefix
  leaves exactly what was there before those positions were written.
- A launch's `history_base` comes from the authority's retained range and is
  always a whole number of pages. `history_rows` is the committed frontier minus
  that base. No launch may declare more than the authority reports committed.
- Enqueued work retains its leases until a completion event is observed, exactly
  as task 0037 established; the frontier moves after that event and not before.
- Lowering keeps kernel selection by capability, shape and layout. A model crate
  supplies semantic parameters and owns no pages, no CUDA and no loop.

## Acceptance

1. Host tests: placement agrees with the host store's ring at every row across a
   wrap; whole-page eviction never drops a row the window admits; capacity
   rounding is exact; the frontier, retained range and capacity are three
   numbers; abort and truncation restore exactly; a launch built from the
   authority never exceeds it.
2. Device tests on both SM86 UUIDs and SM120: append through a state
   transaction, attend, abort, truncate and re-append, with a **reclaimed base**
   — `history_base > 0` after a wrap — which task 0037 could not reach. Every
   component still checked against the FP64 oracle under the predeclared bound.
3. The 32,768-row gate still passes, now driven through the state authority, and
   still reports the three counters separately.
4. A graph containing `OpParams::Attention` lowers, admits and executes on
   device handles, with the previous refusal preserved as a negative fixture for
   the cases still unsupported.
5. Architecture checks prove no second state or admission authority appeared and
   that model crates cannot reach the path. Fmt, both clippy lanes, affected
   suites, `arch-check`, `spec-check` pass; failed, skipped and unsupported are
   reported separately.
6. Support matrix updated with what executes, on which devices, at which
   context, and with what remains unsupported. Task 0037's record is updated to
   point at whichever of its open items this closes.
7. **Moved from task 0037's own scope, 2026-09-20 (owner):** the task mutation
   battery. Task 0037 deliberately did not write one against a state binding
   that did not yet exist; this task's own binding is what it must now measure.
8. **Moved from task 0037's own scope, 2026-09-20 (owner):** host allocator
   behaviour across decode steps, measured through the measured-allocator lane
   rather than the "admission does not grow" assertion task 0037 used as a
   placeholder. Task 0037 closed without this measurement on the understanding
   that it belongs here.

Stop for owner direction before changing the numerical bound, the cache
precision, the canonical state ownership, or before any operation that writes
under a checkpoint root. Stop and report rather than inventing a second
retention rule if the host ring and whole-page device eviction cannot be
expressed as one contract.

## Progress — 2026-09-19, half one: the authority owns the pages

Placement, retention, frontiers, transactions, abort and truncation are decided
by the state authority. Commit republishes a changed page view before finalizing
the state transition and poisons the sequence if a multi-layer publication
partially succeeds. Raw page mutation is confined to the
`paged-attention-test-hooks` feature; the production binding exposes only the
authority-driven append and commit operations.

### What moved

- `moxie_state::device::DeviceKvSequence` is the authority for a
  device-resident sequence: transactions through the same `SequenceState` the
  host store uses, retention through the same `Retention`, geometry through the
  same `KvGeometry`. It owns no bytes. It answers where a row goes
  (`placements`), what the mapping is (`page_table`), what is still held
  (`retained`), what is history (`committed_rows`) and what a truncation or an
  abort leaves.
- `moxie_types::PagePlacement` is the vocabulary between them. It lives in the
  crate both already use. The executor's production state binding has an
  optional `moxie-state` dependency; the test-hook feature extends that binding
  with raw kernel qualification, and model crates cannot reach either path.
- `PagedAttentionRun::append` became `write_rows`, taking placements instead of
  computing `position / page_tokens` itself. Its `committed_rows` became
  `written_rows` and is documented as what it always physically was — a
  high-water mark over observed copies, not a frontier. The two checks are now
  separate and neither subsumes the other: the authority refuses a launch
  reading rows it never committed, the run refuses one reading bytes it never
  wrote.
- Publishing a page table more than once is now legal. The old rule — "the
  mapping cannot change once rows are committed" — is one only a cache that
  never reclaims can keep, and a ring's retained range slides by construction.

### The two facts the contract rested on, now tested

- **The ring is a page table.** `the_device_authority_places_rows_where_this_store_does`
  reads `PagedSequence`'s **own** page-table bytes for the physical page a
  placement names and asserts the byte offset matches what `ranges` returns, for
  every row of every layer. Not a restatement of the formula: the host store's
  table is the oracle.
- **Whole-page eviction needs the extra page.**
  `whole_page_eviction_never_drops_a_row_the_window_admits` sweeps windows 1, 7,
  8, 9, 24 and 31 against page widths 4, 8 and 16 over 300 appends each, and
  asserts at every frontier that the retained base is at or below the oldest row
  the window still admits. That is what the rounding buys, checked rather than
  argued.

### Evidence

- `cargo test -p moxie-state`: 57 lib tests, every integration suite, 0 failed.
  Twelve of them are this module's, including abort, truncation, the ADR 0014
  headroom refusal, reclaimed placement and the malformed-geometry refusals.
- `cargo test -p moxie-executor --features driver --test paged_attention_device`:
  7 cases including the new `a_wrapped_ring_answers_exactly_as_an_unwrapped_one`
  — **the `history_base > 0` case task 0037 could not reach**. A four-page ring
  wraps (retained base 48 of 100 committed rows) and an eight-page store does
  not; both hold the same rows at the same absolute positions, and the decode at
  position 99 under a 40-row window is **byte-identical** between them. An
  equality, not a tolerance: a tolerance would accept a kernel that read the
  wrong page and happened to land close.
- `cargo xtask-cuda test-gpu`: 51 cases, 51 passed, 0 failed, 0 skipped, both
  architectures qualified. `paged_attention_32k` is now driven **through the
  authority** — it places every row, publishes the mapping and is the only thing
  that says what is committed — and reports the same error summaries as before
  the change: max 3.037e-5 at 32,768 visible rows, 2.953e-5 after the append,
  identical on all three GPUs.

## Progress — 2026-09-19, half two: the graph lowers, the handles exist

The selected planner now chooses paged attention and the executor launches it
from the selected plan's admitted query and output slots.

### The planner no longer refuses attention

`moxie-plan`'s `stateful_resource_plan` refusal is **narrowed, not removed**.
Appending to paged KV on an `Op::Attention` node lowers; every other state
effect still has no admission contract here and still says so, and the branch is
fail-closed — `OpParams::state_effect` returns `Appends` for `Attention` and
`None` for everything else today, so an operation that gains an effect is
refused until someone writes its contract.

What lowering now produces is `StateRequirement`, one per attention node: the
layer, the head geometry, the declared scale, the visibility rule, and the two
row counts kept separate — what this step appends and what it attends over. KV
pages are **not** in the activation arena: they are persistent, they outlive
every step, and `moxie-state` admits them against its own retention. So the plan
*reports* them and a caller checks the authority holds a layer of that shape.
A plan whose arena was the whole truth about its memory would be wrong for every
attention graph, which is why this is a field rather than an omission.

### Attention consumes device handles

`PagedAttentionRun::attend_into` takes a device query range and a device output
range and copies no host bytes. That is document 04's contract — "attention
consumes device tensor handles and page-table/state handles", with host paths as
"explicit separate implementations, not compulsory staging interfaces" — and the
host-staged `attend` is now explicitly that separate implementation, sharing one
precondition check and one launch with it.

`device_handles_and_host_staging_give_the_same_bytes` asserts they are one
operation: the same launch, staged through the run's own ranges and read from a
caller's arena ranges, produces **byte-identical** output. A separate
implementation that computed something else would satisfy the words and not the
contract.

### Still out of scope

MLA, streaming, COW, prefix reuse, FP16 cache, tensor cores and timing remain
outside this task.

## Progress — 2026-09-19, the second review's four blockers

Four correctness fixes, a precision gap, a resource gap and two pieces of
evidence, numbered below. All are closed, including the resource gap: it took
a second attempt (item 6).

1. **Placements, the page table and the launch were not bound to each other.**
   A caller could write rows through one permutation, publish another valid
   permutation, and launch against rows that were never written there — each
   check passed because each looked at one of the three alone. A table now
   carries the `history_base` its logical page zero names, so it is a view of
   absolute rows; a write is checked against that view; a republication must
   agree with the view it replaces wherever both describe a page that holds
   written rows; and a launch must name the base the view describes.
2. **Placement was not tied to a transaction or the frontier.** `placements`
   took any position up to the admitted context, so an empty sequence could
   place row 100, write it, publish one row, and leave the authority reporting
   one row committed while the performer's high-water mark sat at 101 — with
   both checks passing and row zero never written. Positions are the
   authority's now: `stage(txn, rows)` reserves them at the frontier, one batch
   at a time, and placements and publication both speak only of that batch.
3. **A windowed abort left the retained base advanced.** The base was derived
   from the published frontier, so a tentative append moved it and an abort left
   it moved: at window 24 and headroom 8, a sequence at row 100 retained
   64..100, staged eight rows, aborted, and retained 72..100 for ever after —
   eight rows the window still admitted, lost to a transaction that was rolled
   back. The base is now derived from the **committed** frontier plus the
   admitted headroom, so the worst a transaction could do is priced in before it
   starts and nothing it does can move it.
4. **Truncation below a retained base was accepted.** The host store refuses the
   equivalent rollback because those rows are not there; so does this now, with
   `Error::Reclaimed` naming the layer and the base.
5. **Cache precision was unvalidated, and unreported.** `DeviceKvSequence::new`
   accepted any precision while everything below read two bytes as BF16 — and
   FP16 has the same width and a different meaning. Non-BF16 is now
   `Unsupported`, a zero window and zero headroom are refused as the host
   geometry refuses them, and lowering now validates all three attention
   operand roles: query must be an activation and BF16 — the only precision
   the selected kernel serves — and key and value precision must match each
   other. `StateRequirement` carries that shared key/value cache precision so
   a binder can reconcile graph, kernel and authority on one encoding.
6. **The direct-device path was charged for staging it never uses.** The first
   attempt only shrank the arena byte counter (`Extents::per_step`) while
   `resource_request` still unconditionally requested query, output and
   host-readback buffers for both staging modes, so the ledger could still
   refuse a direct-device plan for capacity it never touches; a review caught
   that the passing test proved the byte count, not the charge. `resource_request`
   now requests those three buffers only for `Staging::Host`, and the ledger
   test checks the missing-host-scope distinction rather than the arena bytes.

**The missing lifecycle evidence now exists.** `paged_attention_state_lifecycle`
runs append, attend, a real authority-driven tentative append and abort,
truncate and re-append after the ring has wrapped. It then lowers, admits and
executes an attention graph through the plan's own device slots against that
same reclaimed state. The case passes on the SM120 and both SM86 devices and
checks every answer against the FP64 oracle. The retained base is 80, so this
is the `history_base > 0` case acceptance 2 required.

Separately, `abort_truncate_and_reappend_hold_on_device`'s truncation assertion
`sequence.committed_rows() == 8` (after truncating from 12) did not hold under
the implementation this was written against: `committed_rows` reported the
physical write high-water mark, which a truncate does not move, not the
accepted/committed frontier the method's own documentation promised. **Closed**
as of the 2026-09-20 regression run below: the facts are now split three ways —
`published_rows()` (materialized including tentative), `committed_rows()` (from
`SequenceState`'s own accepted count, moved only by `commit`'s `accept`
argument and bounded to what the transaction published), and a private
`committed_high_water` used only by `retained()`. `commit` also now refuses an
`accept` past what the transaction published, closing the separate overflow
this record had not previously named.

**The numerical bound is closed.** It rejects nonfinite operands and
nonpositive scales, derives its FP64 softmax weights from the supplied query,
keys and scale, and refuses a nonfinite result. A caller can no longer supply a
different distribution or make a NaN comparison read as a pass.

**The writer callback.** `moxie_types::WriteReceipt` — a per-layer digest a
caller could construct and hand back, with a public constructor and no
sequence/layer/device/run identity in it — is **deleted**.
`moxie_types::PagedKvWriter` replaces it, implemented by `PagedKvWriterAdapter`
and called from `DeviceKvSequence::append`. `&mut dyn PagedKvWriter` called by
`moxie-state` **is** the trait object and the callback into the executor that
"Bounded deliverable" above originally ruled out by name — see the owner's
amendment there, 2026-09-20: the callback stays, the contract line is amended.
The lifecycle findings are closed. `PageView` is moved into the writer, commit
publishes every changed view before finalization, and a partial multi-layer
publication poisons the sequence. `PagedKvWriterAdapter` is private. Per the
owner's ruling, raw mutation exists only behind `paged-attention-test-hooks`;
the production `paged-attention-binding` feature does not expose it. A
pre-enqueue refusal returns the key/value allocations unchanged, while an
unknown completion keeps them quarantined.

## Earlier regression evidence (superseded)

Measured fact, dated 2026-09-20, against commits `b1e9381..6984c17`. This is
historical regression evidence: it says the 51 then-existing GPU
cases and the host/driver suites still pass through the committed
`append`/`PagedKvWriter` path — page table published from the authority's own
`PageView` on `append`, not a caller's copy of its arithmetic. A later review
found it does not cover the uncommitted commit-to-page-table lifecycle change
above, and it cannot establish invariants the public API still lets a caller
bypass — passing tests over a path a caller can also reach directly are not
evidence that the authority is exclusive. It did not satisfy acceptance 2;
the closure run in the result below supersedes it. It carries no timing or
performance claim (O6/O7 remain open; nothing here was timed).

- Host: `cargo test --workspace`, 1185 passed, 0 failed.
- Driver: `cargo test -p moxie-executor --lib --tests --features driver`, 160
  passed, 0 failed.
- GPU: `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo run -p xtask --features cuda --
  test-gpu`, 51 passed, 0 failed, 0 skipped/unmeasured, on all three devices.
  QUALIFIED sm_86 (both 3090s, devices 1 and 2) and sm_120 (5060 Ti, device 0).
  `paged_attention` and `paged_attention_32k` pass on every device. 32k-decode
  on sm_86: max 3.037e-5, rms 5.625e-6, p99 1.519e-5 — identical to the figure
  recorded before the state binding was rewritten; 32k-append-decode max
  2.953e-5; 32k-sliding max 1.092e-4.

## Result, filled after work

Acceptance 1–6 are implemented and awaiting owner acceptance, not yet
independently reviewed to closure. `cargo test --workspace` passed with no
failures; the executor's driver/test-hook lane passed 158 tests with no
failures. Both clippy lanes, `arch-check` and `spec-check` pass. The final GPU
lane passed 54 cases with 0 failed and 0 skipped/unmeasured on the SM120 and
both SM86 GPUs. It includes the combined reclaimed-base lifecycle and
selected-plan execution; the 32,768-row figures remain unchanged. No timing or
performance claim is made (O6/O7 remain open).

Acceptance 7 is met by the final binding's T0037 battery: 11 of 11 substitutions
were caught, with zero survivors, unstable verdicts, invalid controls or
skipped anchors. The four unmodified lanes passed three times before and after
the run; [experiment 0008](../evidence/experiments/0008-paged-attention-mutations.md)
records the exact substitutions and deciding lanes.

Acceptance 8 is met by the existing 32-step decode case under its own
thread-local counting allocator. It measured at most 9 allocation calls, a
256 B transient peak, -256 B net within one operation (the consumed staging
inputs exceed the returned output), and 1,000 B maximum live growth including
the retained output and bounded lineage/page-table metadata. The declared
metadata bound is 544 B. The battery's deliberate 1 MiB-per-step leak is caught
by this lane.

All eight acceptance criteria are implemented and evidenced above. **Accepted,
2026-09-20 (owner).** This closes task 0038 and M4.1.
