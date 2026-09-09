# Handover — task 0010 basic device arena

## Workspace identity

Writable root `/home/rodrigo/Developer/moxie`, branch `main`, implementation based
on contract commit `0bab91b` (parent `7c36658`). All accompanying changes belong to
task 0010. Local commits only; no push. Read-only legacy remains at
`/home/rodrigo/Developer/strata`, commit
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its existing `.pi/` and `tests/p2p/`
untracked paths remain untouched. No checkpoint inputs or generated model binaries.

## Completed facts

[Task 0010](../tasks/0010-m1-basic-device-arena.md) records the fixed contract,
implementation, tests and limitations. `moxie-memory` owns aligned first-fit range
metadata, identities/generations, transfer and coalescing. `moxie-executor` owns
one admitted physical allocation, the parent reservation and event-retained uses.
Both lease families use the same completion/loss state machine. Checked final
free precedes charge release; ambiguous cleanup cannot allocate or retry free.

Host, real-device, fault, architecture and isolated no-driver gates are recorded
in the task. The support matrix adds the narrow arena capability. No production
path was deleted; task 0009's whole-allocation upload remains available.

## Decisions

No owner ruling, ADR or reference amendment. One arena owns its reservation and
one GPU UUID. One retained host source and one recorded completion per use are
the explicit basic-slice limits. Persistent transfer changes range ownership
without changing physical storage or ending the reservation. Process teardown
remains the recovery boundary for quarantined resources.

## Remaining hypotheses and blockers

Independent review is pending. No admitted execution plan, tensor layout binding,
kernel chain, residency/eviction, host arena, multi-stream fan-in, model quality,
actual context or inference performance is established. Source budgets apply
while the engine owns the source; returned sources are caller-owned.

## Next task

First independently review task 0010 against its contract, including coalescing,
quarantine, reservation ownership and asynchronous source retention. After its
acceptance, write a separate bounded contract for an admitted execution plan in
the shared executor. Read documents 02/03, M1.3 in document 06, document 07 and
the relevant actual R03/R07/R08/R11/R14 sources. Establish graph/resource-plan
binding and refusal before execution, with host oracles and real-device lifetime
tests. Scope allowed files and cancellation explicitly before code. Stop before
model loading, residency policy, kernel-chain optimization or performance claims.
