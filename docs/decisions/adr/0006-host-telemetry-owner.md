# ADR 0006 — Host memory telemetry gets its own crate

- Date: 2026-09-09.
- Status: proposed with [task 0008](../../tasks/0008-m1-measured-host-capacity.md); not yet
  implemented.
- Authority: **implementation decision.** Document 03 fixes that host admission reserves headroom
  "from measured available memory"; it does not say which component takes the measurement, and
  document 02's ownership table has no row for machine telemetry at all. That absence is why this is
  an ADR and not a silent addition.
- Amends: nothing. It adds a crate document 02 does not name, and two `arch-check` rules that confine
  it.

## Decision

A new crate **`moxie-host`** is the single owner of reading this machine's memory telemetry
(`/proc/meminfo`, and the cgroup v2 files under `/sys/fs/cgroup`). It depends on `moxie-types` and
nothing else, and it produces a plain `MeasuredHost` descriptor declared in `moxie-types`, beside
`MeasuredDevice`.

Two machine-checked rules pin the ownership:

- **No production source outside `moxie-host` may name a `/proc` or `/sys` path.** Telemetry has one
  reader.
- **`moxie-memory` may not depend on `moxie-host`.** The ledger never probes.

This mirrors the shape [task 0007](../../tasks/0007-m1-rank-context-and-measured-capacity.md)
established and the owner's reviewer accepted, one tier over:

```text
moxie-cuda  --(MeasuredDevice)-->  moxie-types  <--(MeasuredHost)--  moxie-host
                                        |
                                   moxie-memory turns a descriptor into a budget
                                        |
                        the composition root wires sensors to the ledger
```

## Options examined

**1. `moxie-memory` reads it directly.** Rejected twice over. An `arch-check` rule from task 0006
already forbids that crate the filesystem, and the deeper reason is task 0006's accepted contract:
the ledger *never probes*, and `Ledger::preview` is documented as pure with respect to live
resources. A crate that can open `/proc` during admission makes that a promise about how the code is
written rather than a property of what it can reach. R02 is the cost of an authority that models
resources instead of owning them; an authority that measures them mid-decision is a different way to
lose the same property.

**2. `moxie-storage` reads it.** Rejected. It owns *artifacts*, and its accepted boundary rule is
that it reads bytes and never interprets what they mean. Machine telemetry is a different subject
with a different reason to change, and folding it in would also make the artifact reader a
dependency of admission, which nothing needs.

**3. The composition root reads and parses it.** Rejected, though it is the smallest change. The
parser is not glue: `MemAvailable` versus `MemFree`, the cgroup v2 effective limit over a hierarchy,
and the rule that swap is never budget are all shared semantics with a right and a wrong answer.
Document 09's completion checklist asks whether "a generic responsibility acquired a second owner
anywhere"; leaving this in `xtask` guarantees a second copy the first time the service needs a host
budget. A composition root should *call* a sensor, not be one.

**4. A new `moxie-host` crate.** Selected. It gives telemetry one owner, at the bottom of the graph
where both the ledger's descriptor type and the composition root can reach the result, and it makes
the confinement expressible as a machine check rather than a review habit.

## Cost

A crate that document 02's table does not name. The table describes responsibilities rather than a
closed crate list, and it explicitly allows starting with fewer physical crates when the rules are
machine-checked between modules — but it also forbids creating "empty abstraction crates for
appearance", so the burden is to show this one is not that. It owns a parser with real content, a
cgroup hierarchy walk, an injectable filesystem root that makes all of it testable without touching
`/proc`, and a confinement rule no other crate could carry.

The alternative cost, had this gone to the composition root, is one duplicated parser at M8 and a
silent second owner of a shared semantic in the meantime.

## Evidence and acceptance

Task 0008's acceptance gates. In particular: the parser is exercised against committed fixture trees
rather than against this machine, so a cgroup limit, an ancestor limit, a missing `MemAvailable` and
a malformed field are all tested where none of them exist here; and the two `arch-check` rules each
get a rejecting fixture and an accepted counterpart, because a rule that has never rejected anything
is not evidence.

## Enforcement and removal

The two `arch-check` rules are the enforcement. There is no temporary bridge and nothing to delete:
no component reads host telemetry today.

Re-evaluate when the service composition root exists (M8), which is the first consumer other than
`xtask`. If document 02 is ever revised to place machine telemetry elsewhere, that revision
supersedes this ADR.
