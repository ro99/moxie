# Task 0038 — M4.1b the state authority owns the device pages, and the graph reaches them

Status: active, half implemented. Opened 2026-09-19 after task 0037's kernel and
binding were qualified and twice reviewed. The state authority owns the device
pages; the graph does not yet reach them. See "Progress" below.

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

Stop for owner direction before changing the numerical bound, the cache
precision, the canonical state ownership, or before any operation that writes
under a checkpoint root. Stop and report rather than inventing a second
retention rule if the host ring and whole-page device eviction cannot be
expressed as one contract.

## Progress — 2026-09-19, half one: the authority owns the pages

Deliverable 1 landed. **Deliverable 2 — lowering `OpParams::Attention` and
binding device handles — is not started**, so this task stays open and nothing
below claims otherwise.

### What moved

- `moxie_state::device::DeviceKvSequence` is the authority for a
  device-resident sequence: transactions through the same `SequenceState` the
  host store uses, retention through the same `Retention`, geometry through the
  same `KvGeometry`. It owns no bytes. It answers where a row goes
  (`placements`), what the mapping is (`page_table`), what is still held
  (`retained`), what is history (`committed_rows`) and what a truncation or an
  abort leaves.
- `moxie_types::PagePlacement` is the vocabulary between them. It lives in the
  crate both already depend on, so the executor gained **no** dependency on
  `moxie-state`: retention and frontiers stay out of the crate that launches
  kernels, and `arch-check` is unchanged.
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

### Still open

Deliverable 2 in full: `moxie-plan` still refuses every stateful graph, no
`OpParams::Attention` node lowers to this binding, and query and output still
cross the host boundary. Acceptance 4 is untouched; 1, 2, 3 and 5 are met for
the half that landed. Task 0037's remaining items are unchanged except its
reclaimed-base gap, which this closes.

## Result, filled after work

No completion is claimed: half of the bounded deliverable is not started.
