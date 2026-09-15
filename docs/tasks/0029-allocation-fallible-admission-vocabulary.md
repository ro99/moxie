# Task 0029 — an admission vocabulary that refuses instead of aborting

Status: **proposed**; contract written before implementation.

## Identity and authority

- **Task ID / milestone**: 0029. Not a milestone deliverable of its own: it is
  the shared-owner half of task 0028's third review finding, split out because
  it changes a vocabulary with its own consumers and does not belong inside a
  correction round.
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

Independent review reached it through `AffineLinearRun::admit`. Task 0028's own
allocations on that path are fallible; this one is not, and it is reached first.

## Bounded deliverable

- **One concrete outcome**: constructing and admitting a plan request returns a
  typed `CapacityExceeded` when the allocator refuses, on every allocation
  position, for the affine linear, the BF16 chain and the expert paths.
- **Sole owning shared component**: `moxie-memory`'s `request` module.
- **The surface, named**, because the point is that it is a vocabulary and not
  one function:
  - `PlanRequest::new` — the stage `collect()`, the `BTreeSet` built to check
    distinctness, and three `format!` refusals.
  - `PlanRequest::buffer` / `reserve` — `Vec::push` on the request's lists, and
    the `format!` refusals around them.
  - `BufferRequest::new` — `label.into()`. This is the one that changes a
    **signature**: it returns `Self` today, it has 115 call sites, and a label
    that cannot be allocated has nowhere to go. Whether it takes `&'static str`,
    returns `Result`, or stores a borrowed label is the design question this
    task exists to answer.
  - `Ledger::admit`, for whatever it allocates between accepting a request and
    returning a reservation.
- **Non-goals**: any change to what is admitted, to the ledger's accounting, or
  to any tier policy. This is about how a refusal is *reported*, not about what
  is refused.

## Contract before implementation

- A failed allocation produces `Error::CapacityExceeded` and **no** partially
  built request: a request that lost a label is not a request with an empty
  label. Review caught exactly that shape in task 0028 — a fallible clone that
  returned `Ok("")` on a failed reservation, producing a corrupted descriptor
  rather than a refusal.
- No allocation on a **successful** admission may be infallible. The success
  path is the one with no refusal to fall back to, which is why task 0028's
  selection was repaired there before anywhere else.
- Distinctness of stage labels must not need a `BTreeSet`: stage lists are
  small and an O(n²) scan over them allocates nothing.

## Acceptance

1. A sweep over **every** allocation position of `PlanRequest::new` plus
   `buffer`, in an isolated executable with a per-thread one-shot failing
   allocator, requiring at each firing position a typed refusal — never a value
   built from a reservation that failed — and requiring the first non-firing
   position to return a request equal to the unarmed one. The loop bound being
   exhausted is a failure. This is the shape task 0028's review required after
   a weaker sweep accepted a corrupted success.
2. The same sweep over `AffineLinearRun::admit` on real hardware, which is the
   test task 0028 could not write. It must also show the ledger has nothing
   outstanding after a refusal at any position.
3. Two mutations in a battery: a `try_reserve` whose failure yields a default
   value instead of a refusal, and one infallible growth restored.
4. Every existing consumer still compiles and passes: the BF16 chain, the
   expert plans, the residency authority and the whole device lane.

## Exact condition requiring owner direction

- If making `BufferRequest::new` fallible forces a change to what a *caller*
  can express — a label that must now be static, say — that is an API narrowing
  across 115 sites and the owner should see the shape before it lands.

## Result, filled after work

*(empty — no implementation has started)*
