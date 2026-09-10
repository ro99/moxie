# Handover — task 0011 review corrections

## Workspace identity

The correction implementation is commit `78d90be` in
`/home/rodrigo/Developer/moxie`, branch `main`, based on the independently
reviewed task range `9e14eeb..34055b8`. This handover, the task result amendment
and support-matrix update are the only later documentation changes. No push is
authorized or performed. Legacy remains read-only at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, with its existing untracked `.pi/`
and `tests/p2p/` untouched. No checkpoint or model artifact was accessed.

## Completed facts

All three contract violations and the coverage gap in the independent review
are corrected:

- `moxie-plan` rejects any `ValueRole::Weight` declared through
  `GraphBuilder::input`, so it cannot become an uncharged `ExternalInput`.
  External activations receive checked element-count and byte-extent
  validation before their binding is emitted. External indices receive checked
  element-count validation without assigning an index width absent from the
  graph contract. The regression reproduces the disguised 8x8 BF16 weight and
  overflowing live external shapes. A valid declared 8x8 BF16 weight passes
  lowering and is then admitted to the shared ledger as exactly 128 bytes of
  `PackedResidentWeights`.
- Every `TensorHandle` descriptor field is private. Public read-only accessors
  expose value, shape, role, layout, offset and bytes, while allocation identity
  remains tied to the borrowed checked range. A compile-fail doctest verifies
  that a downstream caller cannot rewrite the byte bound; the device test reads
  and checks every descriptor through the public API.
- `ReservedPlan::admit` is a thin wrapper over one admission implementation.
  The integrated fault test invokes that implementation, fails the second
  planned slot, and proves the capacity refusal returns the unchanged candidate,
  releases the earlier slot and physical arena, empties the ledger and restores
  exact free device memory. It passes independently on all three visible UUIDs.

Passed final validation:

- `cargo fmt --all -- --check`.
- Host and full `xtask/cuda` workspace clippy with all targets, locked/offline
  dependencies and `-D warnings`.
- Locked/offline host workspace: 548 unit/integration tests and 8 doctests.
- Locked/offline full `xtask/cuda` workspace: 560 unit/integration tests and 11
  doctests.
- Focused `moxie-plan` and host/driver `moxie-executor` suites, including
  resource binding and fault tests on every GPU.
- Architecture check: 55 rejecting fixtures, 14 accepted fixtures and 12
  rules. Specification check: all 10 reference documents unchanged.
- `cargo xtask-cuda test-gpu`: 33/33 cases, with `sm_86` and `sm_120` both
  qualified. With `CUDA_VISIBLE_DEVICES=1,2`, 22/22 visible-device cases passed
  and the command exited 1 as required because `sm_120` was unqualified.
- Capacity passed in normal visibility and with
  `CUDA_VISIBLE_DEVICES=2,1,0`; budgets remained keyed by UUID.
- A fresh target with `CUDA_HOME=/nonexistent`, `NVCC=/nonexistent`, CUDA
  removed from `PATH` and `LD_LIBRARY_PATH` unset passed the 548 host tests and
  8 doctests. Its explicitly rebuilt `xtask` had no `libcuda` entry in `ldd`.

The GPUs exercised were RTX 5060 Ti
`GPU-97fe4889-4874-a378-198e-955d2e72c4a3`, RTX 3090
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` and RTX 3090
`GPU-81fe4578-59b2-37c4-421e-287cdac78704`.

Failed during correction, then fixed before the final validation: the first
fault injection requested `u64::MAX`, which correctly failed checked alignment
before reaching the intended capacity branch. The final test requests a finite
amount larger than the admitted activation arena. An earlier driver run in a
restricted execution environment saw no CUDA devices; after the user enabled
the drivers and full authorization, every device lane above passed. No
acceptance threshold or existing assertion was relaxed.

Skipped and unmeasured: model quality, checkpoint import, actual-context
inference, semantic CUDA arithmetic, state mutation, topology communication,
and paired prefill/decode performance remain outside this resource-only task.

## Decisions

No ADR, owner ruling, reference-document amendment or product setting changed.
Rejecting weight-role per-step inputs uses the explicit option allowed by the
review and preserves `GraphBuilder::weight` as the path whose bytes enter the
complete resource envelope. Index byte width remains unspecified rather than
being inferred from a floating or weight precision. The allocation fault seam
is private and changes no public API or production allocation policy.

## Remaining hypotheses and blockers

Independent re-review remains the only acceptance blocker for task 0011. This
correction establishes resource completeness and immutable checked descriptors;
it does not add resident weights, kernel capability selection, workspace,
semantic execution or state. Task 0011 and M1 as a whole must not be represented
as model execution.

## Next task

Re-review `34055b8..78d90be` against the original findings and the unchanged
task contract. If accepted, write the next bounded M1.3 task for registry-backed
semantic capability selection, exact kernel workspace inclusion in this same
reservation, and one real device layer chain using task 0009/0010 operation
leases. Stop before checkpoint loading, residency/eviction policy, paged state,
generation, multi-device execution, quality claims or performance claims.
