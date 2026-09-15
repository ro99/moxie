# Task 0029 — an admission path that refuses instead of aborting

Status: **proposed**; contract written before implementation.

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
     exact state — free list and live map — against what it was before the
     armed call. Without it, a mutation that returns a refusal *without* undoing
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

*(empty — no implementation has started)*
