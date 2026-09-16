# Task 0029 — an admission path that refuses instead of aborting

Status: **implemented, reviewed three times, not accepted.** Contract written
before implementation and kept as written; the Result section at the end records what
it did and did not deliver, and which clauses are still open. The header used to
say `proposed` while the Result said `implemented`, which is a document
disagreeing with itself.

## Identity and authority

- **Task ID / milestone**: 0029. Not a milestone deliverable of its own: it is
  the shared half of task 0028's open allocation finding, split out because it
  changes code with its own consumers and does not belong inside a correction
  round.
- **Writable root / base**: `/home/rodrigo/Developer/moxie`, branch `main`.
- **Requirement repaired**: AGENTS.md's product gate that an allocation failure
  is a **typed error**, not an abort — the rule tasks 0019, 0021, 0022, 0023,
  0024 and 0028 have each broken on a path added after the previous fix.
- **Owner gates**: none opened or touched. No performance claim follows.

## The measurement this starts from

`moxie_memory::PlanRequest::new`, built exactly as
`moxie_executor::affine_linear::resource_request` builds it, with a one-shot
failing allocator armed at position 4:

```text
memory allocation of 5 bytes failed
signal: 6, SIGABRT
```

Independent review reached it through `AffineLinearRun::admit`.

## Scope: the whole admission call graph, not one module

**An earlier draft of this contract named `moxie-memory`'s `request` module as
its sole owner and then required a sweep of every allocation in
`AffineLinearRun::admit`. Those two statements are not compatible**, and review
said so: repairing `PlanRequest` only moves the first abort further down the
same call graph. The scope is therefore the graph, and the inventory below is
part of the contract rather than a note.

| Crate / module | Site | What allocates infallibly |
|---|---|---|
| `moxie-memory::request` | `PlanRequest::new` | the stage `collect()`, the `BTreeSet` built to check distinctness, three `format!` refusals |
| `moxie-memory::request` | `PlanRequest::buffer` / `reserve` | `Vec::push` on the request's lists, and the `format!` refusals around them |
| `moxie-memory::request` | `BufferRequest::new` | `label.into()`. **The one that changes a signature**: it returns `Self`, it has 115 call sites, and a label that cannot be allocated has nowhere to go |
| `moxie-memory::ledger` | `Ledger::admit` | `committed.entry().or_insert()`, then `outstanding.insert(..)` and the `label().to_string()` inside it — **in that order**. The counters are charged *before* the record that names them is allocated, so an abort or a refusal between the two leaves capacity charged against a reservation that does not exist |
| `moxie-memory::arena` | `Arena::new` | `vec![FreeRange { .. }]` |
| `moxie-memory::arena` | `Arena::allocate` | `owner.clone()` and `live.insert`, **after the free list has already been mutated** — so an abort there is not just a crash, it is a crash with the arena half-updated |
| `moxie-memory::ledger` | `Ledger::outstanding` | a label and two vectors cloned **per reservation**. Reached from admission, not only from reporting: `DeviceArena::create_partitioned` calls it to find the reservation it was handed. An earlier draft of this row named `Arena::outstanding`, which admission never calls; review corrected it. `Ledger::for_each_outstanding` already exists as the allocation-free form, so the repair may be a call-site change rather than a new one |
| `moxie-executor::arena` | `create_partitioned` | `BTreeSet::insert` while validating the tier list |
| `moxie-executor::arena` | `create_partitioned` | `Rc::new(ArenaCore)`, **after `DeviceBuffer::alloc` has succeeded** — an abort here leaves a live device allocation with no owner |
| `moxie-cuda::driver` | `Module::function` | `CString::new(name)`, reached from `resolve_all` for every symbol |

**The inventory is a starting point, not the acceptance.** It was assembled by
reading, and reading is how the last four rounds of this finding were each
declared closed while still open. The sweep in acceptance (1) is what decides
whether anything remains.

## Contract before implementation

- A failed allocation produces `Error::CapacityExceeded` and **no** partially
  built value: a request that lost a label is not a request with an empty
  label. Review caught exactly that shape in task 0028 — a fallible clone that
  returned `Ok("")` on a failed reservation, producing a corrupted descriptor
  rather than a refusal.
- No allocation on a **successful** admission may be infallible. The success
  path is the one with no refusal to fall back to, which is why task 0028's
  selection was repaired there before anywhere else.
- **Nothing may be left half-changed by a refusal.** `Arena::allocate` mutates
  its free list before it allocates the record; a refusal after that point has
  to put the free list back, and the sweep has to check it.
- Distinctness of stage labels and of tiers must not need a `BTreeSet`: both
  lists are small, and an O(n²) scan over them allocates nothing.

## Acceptance

1. A sweep over **every** allocation position of `AffineLinearRun::admit` on
   real hardware, with a per-thread one-shot failing allocator, requiring at
   each firing position a typed refusal — never a value built from a
   reservation that failed — and requiring the first non-firing position to
   return a run equal to the unarmed one. Exhausting the loop bound is a
   failure. This is the shape task 0028's review required after a weaker sweep
   accepted a corrupted success.
2. After a refusal at **any** position of (1): the ledger has nothing
   outstanding and no device allocation is live.
3. **Two sweeps below `admit`, because (1) and (2) cannot see inside it.**
   Review established both gaps:
   - **`Arena::allocate`, with the arena retained.** `admit` owns its arena and
     drops it on refusal, so a top-level sweep cannot tell a rolled-back free
     list from a discarded one. This sweep keeps the arena and compares its
     exact state — its whole `Debug` rendering, which is free-range offsets and
     order, live records, owners and generations, rather than the aggregate
     `ArenaOccupancy` an earlier version compared — against what it was before
     the armed call. Without it, a mutation that returns a refusal *without* undoing
     the free-list mutation survives while (1) still passes.
   - **`Ledger::admit`, comparing every counter.** It charges `committed` per
     tier and `committed_scope` per scope **before** allocating the record that
     names them, so "nothing outstanding" can pass while capacity stays charged.
     This sweep compares every scope and tier counter before and after each
     refusal, not just the outstanding list.
4. The same sweep over `PlanRequest::new` plus `buffer` in the host lane, which
   needs no GPU and is where most of the inventory lives.
5. Four mutations in the `T0029` battery: a `try_reserve` whose failure yields a
   default value instead of a refusal; one infallible growth restored; a refusal
   that returns without undoing the free-list mutation; and a refusal that
   returns without undoing the committed counters.
6. Every existing consumer still compiles and passes: the BF16 chain, the
   expert plans, the residency authority and the whole device lane.
7. Task 0028's **three** unmeasured repairs — its reserved range list, its
   fallibly built symbol list, and `Module::resolve_all`'s fallible growth — are
   covered by (1) and recorded as measured there. An earlier draft of this
   clause said two and omitted `resolve_all`, which task 0028's own record has
   right; review caught the disagreement.

## Exact condition requiring owner direction

- If making `BufferRequest::new` fallible forces a change to what a *caller*
  can express — a label that must now be static, say — that is an API narrowing
  across 115 sites and the owner should see the shape before it lands.
- If the sweep shows the abort surface is materially larger than the inventory
  above, report the measurement before expanding the work.

## Result, filled after work

Status: **implemented, reviewed three times, not accepted.** The reviews found
five P1 issues, then six, then two more, all repaired below. **Acceptance
clauses 1, 2 and 7 remain open**, and two further items are decisions rather
than work. Clause 3 — the two sweeps below `admit` — **is** met; an earlier
version of this line said 1, 3 and 7, which had it backwards on both: the direct
sweeps satisfy 3, while 2 cannot be established until refusals *after* `Rc::new`
and the device allocation are swept. An earlier version of this record said one line was open; it was
five, and then six.

### What changed

A new `moxie-memory::fallible` module is the vocabulary the rest of this
depends on: a `format!` that refuses, a `push` that reserves first, a
`with_capacity` and a string copy that return `CapacityExceeded` — and a
**`Map`**, which is the part worth explaining. `BTreeMap::insert` allocates a
node through `handle_alloc_error` and there is no fallible form of it, so every
map on this path was an abort waiting for memory pressure, one of them sitting
between charging a ledger counter and recording what had been charged. `Map` is
a `Vec` of pairs in key order, searched by binary search, with
`try_reserve_one` and `insert` deliberately **separate** so a caller can take
the room it needs before doing the work that cannot fail.

The labels are the other structural change. Every label was a `String` behind
`impl Into<String>`, which means the *constructor* allocated:
`BufferRequest::new("activations", ..)` copied fourteen bytes onto the heap, on
the admission path, infallibly. They are now `Cow<'static, str>`, which borrows
a literal and moves an owned string — so the constructors allocate nothing and
not one of the 115 existing call sites changed.

**That is not the same as compatibility**, and an earlier version of this
paragraph used it to argue the API had not narrowed. It had: a
`Cow<'static, str>` will not accept a borrowed `&str` with a caller's lifetime,
and `HostBuffer` had to copy its label explicitly — which is the proof, in the
diff. This is the contract's owner-direction clause, and it is open.

| Site | Repair |
|---|---|
| `PlanRequest::new` | stages grown one reserved element at a time; distinctness by a linear scan rather than a `BTreeSet`; refusal prose composed fallibly |
| `PlanRequest::buffer` / `reserve` | reserved pushes, fallible prose |
| `PlanRequest::scopes` | returns `Result<Vec<Scope>>` — it built a `BTreeSet` by `collect`, and it is called first thing inside `evaluate` |
| `Ledger::admit` | **reordered**: the label, every tier entry and the reservation's slot are obtained first; the counters are added last and cannot fail |
| `Ledger::charges_of`, `scope_charges_of` | new allocation-free accessors. `DeviceArena::create_partitioned` called `outstanding()` — which clones a label and two vectors *per reservation* — to find the one reservation it already had the id of |
| `Ledger::evaluate`, `peaks` | reserved pushes, fallible rows, `fallible::Map` in place of four `BTreeMap`s and a `BTreeSet` |
| `Arena::new` | no `vec!` |
| `Arena::allocate` | the record's slot is reserved **before** the free list moves, so a refusal is byte-for-byte atomic |
| `moxie-executor::arena` | tier distinctness without a `BTreeSet`; the accessors above instead of `outstanding()` |
| `moxie-cuda::Module::function` | `CString::new` grown through `try_reserve` |
| `moxie-executor::plan`, `::chain` | both request builders compose labels and stage lists fallibly |

### Acceptance, against the contract

1. **Host sweeps, all three passing** (`crates/moxie-memory/tests/allocation_refusal.rs`,
   an isolated executable with a per-thread one-shot failing allocator):
   building a request, admitting against a ledger, and allocating inside an
   arena. Each requires a typed refusal at every firing position, a value equal
   to the unarmed one at the first non-firing position, and treats an exhausted
   loop bound as a failure.
2. **The counters clause holds.** A refused admission leaves every tier and
   scope counter exactly where it was — which is a different check from
   "nothing outstanding", and the reason the contract asked for it.
3. **The free-list clause holds.** A refused arena allocation leaves the arena's
   **whole `Debug` rendering** identical — free-range offsets and order, live
   records, owners, generations. An earlier version compared `ArenaOccupancy`,
   which is byte and range *counts* and cannot see any of those; review said so. `admit` owns its arena and drops it on refusal, so this had to be
   swept directly; a top-level sweep cannot tell a rolled-back free list from a
   discarded one.
4. **The device sweep runs** (`every_allocation_in_a_quantized_admission_refuses_rather_than_aborting`):
   **at least 28 of admission's 41 allocation positions** in a real
   `AffineLinearRun::admit` on hardware, every one refusing with
   `CapacityExceeded` and every one leaving the ledger with nothing
   outstanding.

   **"At least", and the word is doing work.** The count is a *reconstruction*
   of admission's pre-`Rc` preparation — the same calls, in the same order, with
   the same primitives — and a reconstruction is evidence about the
   reconstruction. It does not prove position 28 is the last before `Rc::new`.
   Two things would make it exact and neither is built: a production boundary
   hook immediately before `Rc::new` that a counting pass reads, or a
   subprocess-per-position sweep, where an abort is observable because the child
   dies. The sweep also never reaches a non-firing position, so it does not
   establish the contract's equality-with-the-unarmed-run clause. **Clause 1 is
   open**, and so is clause 2, which needs the positions after the device
   allocation.
5. **The `T0029` battery**, run to completion: **4 of 4 caught**, 0 survivors,
   tree clean afterwards. Four mutations over a host and a device lane: a
   failed reservation that yields a value, an infallible growth restored, a
   refusal after the free list has moved, and a refusal after the counters have
   moved. The last two are the shapes no top-level sweep can see.

### Second review: six more

The first repair round fixed five P1s and left three half-done. What it got
wrong is one shape worth naming: **a repair that satisfies the test rather than
the property.**

| Finding | Repair |
|---|---|
| The device sweep still stopped early — it counted `resource_request` + `Ledger::admit` and stopped, skipping the arena label and the metadata arena's own free list, both reachable before `Rc::new` | The counted prefix mirrors `admit`'s order through all four, using production's own primitives. The test also counts admission alone — an earlier version counted `close()` too, which reaches `Arena::release` — and reports **at least 28 of 41** positions, on a named GPU |
| The ledger rollback mutant charged through `get_mut(..).expect(..)` before the production loop installed the rows, so on a fresh ledger it panicked before mutating anything — a crash again | The substitution installs the rows itself, charges, reserves, and un-charges on success, so it fails for the property it is named for |
| `Ledger::admit` reserved the outstanding slot and then installed missing tier entries **one at a time**, so a later failure left earlier rows behind — reachable, because the affine request touches Host and Device | Counted per scope, reserved per scope, then inserted; after the reservation loop no insertion can fail. The sweep's ledger now has **two scopes**, which is what makes the partial state reachable at all |
| The arena test compared `ArenaOccupancy` — byte and range *counts*, not offsets, order, live records, owners or generations | It compares the arena's full `Debug` rendering, built outside the armed window |
| The RAII disarm went into the host harness only | Both harnesses have it |
| `try_label` was made `#[doc(hidden)] pub` so the sweep could call it — real public API added for a test | Deleted. Production and the sweep both use `moxie_memory::fallible::text`, which was already public and is the same function; the duplicate is gone rather than exported |
| `Drop` could remove a parked original while its marker survived, leaving the next run knowing it must recover with nothing to recover from | The marker is removed first and the parked copy only if that succeeded, on both the ordinary and the emergency path |

### Third review: the recovery state machine, and a harness with no tests

| Finding | Repair |
|---|---|
| Two booleans could not express three outcomes: a marker removal that **succeeded** and a parked-original removal that **failed** left the guard claiming both files were still there, retrying the removed marker and never retrying the copy | A three-step `Stage` — `Mutated`, `Restored`, `MarkerCleared`, `Clean` — advanced one confirmed step at a time. `disarm` and `Drop` run the same steps and stop at the first that fails, so a retry resumes where it stopped |
| **The harness had no tests at all**, after three rounds of harness bugs. The 109-case self-test covers verdicts, selectors and anchors — not restoration | Five regressions in `xtask`, injecting failure at each step by putting a directory where a file must be written or removed: the ordinary path, a failed rewrite, a failed marker removal, a failed parked removal, and a retry that resumes. Mutating the last step to ignore its failure fails exactly two of them |
| `try_label` was exposed as `#[doc(hidden)] pub` for the sweep | Deleted; production and the sweep both call `moxie_memory::fallible::text`, which was already public |
| `ledger.release` sat inside the prefix-counting window | Outside it. It allocates nothing today, and counting it would make the reconstruction stop being attributable the moment that changed |
| Two stale comments — `request.rs` still promised an empty detail, the Result still named the deleted `try_label` | Both corrected |
| A failed diagnostic allocation returned an empty `InvalidRequest` | It returns `CapacityExceeded`, which the contract already said. Treating that as an open decision contradicted the contract rather than interpreting it |

Two operational repairs came from the same round: the mutation harness clears
its **parked original** after restoring rather than only the marker, and every
lane runs under a timeout.

### Review of the first implementation: five P1 findings

The first pass claimed one open line. It had five, and the first is the one
that matters most: **the `Cow` removed the allocation from the constructor, not
from the clone.** `AffineLinearRun::admit` builds *owned* labels with its own
fallible formatter, and `Arena::allocate` cloned one **after** the free list had
moved — so the real quantized path could still abort with a live device arena
and a mutated free list. The sweep missed it because its fixture passed the
literal `"first"`, which is `Cow::Borrowed` and clones a pointer. The same
mistake sat in the admission report's stage cloning, which the BF16 plan and
chain builders feed with owned stages.

| Finding | Repair |
|---|---|
| Owned labels cloned after mutation | `fallible::clone_label` copies an owned label fallibly, and `Arena::allocate` does it **before** the free list moves. The arena sweep now passes an owned label, built outside the armed window so it measures the arena rather than the test |
| The device sweep measured its prefix on a **warmed, reused** ledger while every iteration used a fresh one | Measured on a fresh ledger. The count went from 21 to 24 — exactly the undercount the finding predicted, and it rose again to 28 in the round after |
| Zero-valued tier entries inserted **before** the reservation's slot was reserved | The slot is taken first. A failure there no longer leaves the ledger's map carrying rows it did not have |
| The rollback mutants deleted the reservation, so a later `insert` panicked on zero capacity — a crash, not a refusal-after-mutation | Both mutants **swap the order** instead: the state moves first and the fallible reservation follows, so the mutant returns a refusal with the change already made |
| A mutation lane could hang with the tree still mutated | Every lane runs under a 600-second `timeout`; a lane that does not finish is a lane that failed. The sweep harness disarms through an RAII guard, so a panicking `body` cannot leak the trap onto the next test |

The wedge was not hypothetical: review found `T0029` stuck for over twenty-five
minutes on a futex with the mutant still in the source file. That is the failure
this repository already has a shape for — *a tool that measures the tree it is
editing* — and it happened again because the mutation modelled a panic.

The battery now runs to completion: **4 of 4 caught**, 0 survivors, tree clean
afterwards.

### What is still open

Three of the contract's clauses are **not** met, and two of the gaps below are
decisions rather than work.

1. **Capacity-rejection, invalid and host-relocation paths are unswept**
   (clause 1, partly). A request that *fits* never enters `Ledger::alternatives`,
   so the sweeps here miss its `BTreeSet`/`BTreeMap`/`collect`/`vec!`, the
   `Box::new(Rejection)` on the rejection path, and infallible diagnostic
   construction in the arena. This is the repository's own documented
   **valid-only sweep** shape, and naming it is not the same as fixing it: each
   of those paths needs its own sweep.
2. **`Arena::release` calls infallible `Vec::insert`** on the free list after
   removing the live record, so an unwind can still abort under continuing
   pressure even when the original failure was returned correctly.
3. **`CapacityExceeded { tier: None }` is now used for host metadata
   allocations**, while the shared contract reserves `None` for unattributed
   driver allocations. That needs reconciling before more callers copy it.
4. **The API narrowing did happen, and it is the owner's call.**
   `Label = Cow<'static, str>` no longer accepts a borrowed `&str` with a
   caller's lifetime — `HostBuffer` had to copy its label explicitly, which is
   the proof. The first version of this record argued from "no existing call
   site changed", and that does not establish compatibility; review was right.
   This is the condition the contract says requires owner direction.

### Gates, as run

On this tree, GPU `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` (ordinal 0) for the
device lane:

| Command | Result |
|---|---|
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets` | clean |
| `cargo clippy --workspace --all-targets --features moxie-executor/driver` | clean |
| `cargo xtask spec-check` | 10 documents present and unchanged |
| `cargo xtask arch-check` | 79 rejected fixtures, 21 accepted, 13 rules |
| `cargo xtask mutation-check --self-test` | 109 of 109 over 94 anchors |
| `cargo test --workspace --locked --offline` | **1,106 passed, 0 failed, 0 ignored, 0 skipped** |
| `cargo test --workspace --features moxie-executor/driver --locked --offline` | **1,151 passed, 0 failed, 0 ignored, 0 skipped**, 101 suites |
| `cargo xtask mutation-check --battery 0029` | **4 of 4 caught**, 0 survivors, 0 unstable, tree clean and no parked original afterwards |
| the device sweep | at least **28 of admission's 41** positions refuse, on `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` |

Not run: `cargo xtask-cuda test-gpu`, and the `T0006`/`T0028` batteries. This
task changed `moxie-memory`, `moxie-executor` and `moxie-cuda`, so **`T0028` is
now stale on this tree** and is a re-run someone owes before task 0028 is
accepted.

### The one thing this task could not do

**`Rc::new` has no fallible form in stable Rust**, and
`DeviceArena::create_partitioned` needs one for the arena's shared core.

What was fixed is the *consequence*: the `Rc` is now allocated **before** the
device allocation rather than after it, so a failure can no longer leave live
device memory with no owner and no path to a free. What remains is that the
failure is an abort.

That bounds the device sweep. The bound is a **reconstruction** of admission's
pre-`Rc` preparation — `resource_request`, `Ledger::admit`, the arena label
through the same `moxie_memory::fallible::text` production calls, and
`Arena::new` — in `admit`'s order, and
the sweep asserts every armed position fired, so a divergence between the count
and the call is a failure rather than a quietly short loop. An earlier version
of this paragraph said only the request and the ledger were counted, which was
true of an earlier version of the test and skipped two reachable positions.

**It is still a reconstruction**, and that is the honest limit: it does not
prove position 28 is the last before `Rc::new`. A production boundary hook, or a
subprocess-per-position sweep where an abort is observable because the child
dies, would. Neither is built.

**Everything at or after `Rc::new` is unswept, and named as such** in the test
and here. Task 0028's three repairs inside `admit` sit after that line and are
therefore still **fixed but unmeasured**; acceptance clause 7 is *not* met, and
neither are 1 and 2.

The options, none of which should be chosen inside this task:

* a hand-rolled reference-counted handle over a `Vec`-allocated block, which
  buys full fallibility for roughly sixty lines of `unsafe` in a crate that has
  kept its unsafe surface small and audited;
* removing the sharing, so the core is owned by the arena and borrowed by its
  ranges — a lifetime change across the executor;
* accepting the abort, on the argument that a host allocator that cannot serve
  a two-word block has already failed in ways no refusal path improves.

This is the contract's "exact condition requiring owner direction": it is a
change to what the repository is willing to spend `unsafe` on.
