# Handover — task 0029: admission refuses instead of aborting

**Implemented, reviewed three times, not accepted.** Three acceptance clauses
are open, two items are owner decisions, and one battery is stale. None of that
is a surprise to the record: [task 0029](../tasks/0029-allocation-fallible-admission-vocabulary.md)
carries the detail and this handover is the short form.

## Workspace identity

- `/home/rodrigo/Developer/moxie`, branch `main`.
- The device lane ran on `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` (ordinal 0).
- `/models` and `/fast/models` untouched. No checkpoint was read or written.

## What is true now

An allocation failure on the admission path **refuses** rather than aborting,
across `moxie-memory`'s request, ledger and arena modules, `moxie-executor`'s
arena and plan builders, and `moxie-cuda`'s symbol lookup. Two orderings that
made a refusal worse than a crash are fixed: the ledger charged its counters
before allocating the record that named them, and the arena mutated its free
list before allocating the record describing it.

The vocabulary that made it possible is `moxie_memory::fallible` — a refusing
`format!`, a reserving `push`, and a sorted-`Vec` `Map` whose `try_reserve_one`
and `insert` are **separate**, so a caller takes the room it needs before doing
the work that cannot fail. Labels are `Cow<'static, str>`, which is why 115 call
sites did not change.

## What is open, precisely

| Clause | Why |
|---|---|
| **1** | The sweep covers at least 28 of admission's 41 allocation positions. The rest are at or after `Rc::new`, which aborts and cannot be swept past. The 28 is a *reconstruction* of the pre-`Rc` prefix, not a proof of where that line falls |
| **2** | Needs the positions after the device allocation, which clause 1 cannot reach |
| **7** | Task 0028's three repairs inside `admit` sit after `Rc::new` and stay **fixed but unmeasured** |

Also open, and not clauses:

- **Rejection, invalid and host-relocation paths have no sweep.** A request that
  fits never enters `Ledger::alternatives`, so its `BTreeSet`/`collect`/`vec!`,
  the `Box::new(Rejection)`, and the arena's infallible diagnostic construction
  are untested. This is the repository's own **valid-only sweep** shape.
- **`Arena::release` calls infallible `Vec::insert`** after removing the live
  record, so an unwind can abort under continuing pressure.
- **`T0028` is stale on this tree** — this task changed three crates it covers.
  Someone owes a re-run before task 0028 is accepted.

## Two owner decisions

1. **`Rc::new` has no fallible form in stable Rust**, and the arena's shared
   core needs one. What is fixed is the consequence: it is allocated *before*
   the device allocation, so a failure can no longer orphan live device memory.
   The options are a hand-rolled refcount over a `Vec`-allocated block (about
   sixty lines of `unsafe` in a crate that has kept that surface small), a
   lifetime change removing the sharing, or accepting the abort.
2. **`Label = Cow<'static, str>` narrowed the API.** It does not accept a
   borrowed `&str` with a caller's lifetime; `HostBuffer` had to copy its label
   explicitly. The contract says an API narrowing is the owner's call. An
   earlier version of the record argued from "no call site changed", which does
   not establish compatibility.

Also unreconciled: `CapacityExceeded { tier: None }` is now used for host
metadata allocations, while the shared contract reserves `None` for
unattributed driver allocations.

## What the four review rounds are worth reading for

They are in the task record, and the engineering log carries the two shapes:
**a repair that satisfies the test rather than the property** (four rounds of
it), and **a harness with no tests of its own** (the mutation guard, three
defects in three rounds, now with five injected-failure regressions). The second
is the one to act on first if this codebase grows more tooling.

## Gates

fmt; both clippy lanes; spec-check (10 documents); arch-check (79 rejected
fixtures, 21 accepted, 13 rules); `mutation-check --self-test` 109 of 109 over
94 anchors; **1,106** host tests and **1,151** device-feature tests across 101
suites, nothing failed, ignored or skipped; `T0029` **4 of 4 caught**, 0
survivors, tree clean and no parked artifacts afterwards.

Not run: `cargo xtask-cuda test-gpu`, `T0006`, `T0028`.

## Next

The open clauses above, then M3 item 3's **expert** half — quantized MoE.
Nothing routed executes a quantized weight, so no quantized checkpoint runs as a
model.
