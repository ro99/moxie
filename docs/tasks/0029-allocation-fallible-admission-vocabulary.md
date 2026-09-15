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
| `moxie-memory::ledger` | `Ledger::admit` | `committed.entry().or_insert()`, `outstanding.insert(..)` and the `label().to_string()` inside it |
| `moxie-memory::arena` | `Arena::new` | `vec![FreeRange { .. }]` |
| `moxie-memory::arena` | `Arena::allocate` | `owner.clone()` and `live.insert`, **after the free list has already been mutated** — so an abort there is not just a crash, it is a crash with the arena half-updated |
| `moxie-memory::arena` | `Arena::outstanding` | `owner.clone()` per record. Reporting rather than admission, and on the path a refusal takes to describe itself |
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
2. After a refusal at **any** position: the ledger has nothing outstanding, the
   arena's free list is what it was, and no device allocation is live. The
   `Rc::new` site is the reason this clause exists.
3. The same sweep over `PlanRequest::new` plus `buffer` in the host lane, which
   needs no GPU and is where most of the inventory lives.
4. Three mutations in the `T0029` battery: a `try_reserve` whose failure yields
   a default value instead of a refusal, one infallible growth restored, and a
   refusal that returns without undoing the free-list mutation.
5. Every existing consumer still compiles and passes: the BF16 chain, the
   expert plans, the residency authority and the whole device lane.
6. Task 0028's two unmeasured repairs — its reserved range list and its
   fallibly built symbol list — are covered by (1) and recorded as measured
   there.

## Exact condition requiring owner direction

- If making `BufferRequest::new` fallible forces a change to what a *caller*
  can express — a label that must now be static, say — that is an API narrowing
  across 115 sites and the owner should see the shape before it lands.
- If the sweep shows the abort surface is materially larger than the inventory
  above, report the measurement before expanding the work.

## Result, filled after work

*(empty — no implementation has started)*
