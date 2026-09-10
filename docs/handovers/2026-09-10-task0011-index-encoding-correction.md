# Handover — task 0011 index-encoding correction

## Workspace identity

The second review correction is source commit `64899c2` in
`/home/rodrigo/Developer/moxie`, branch `main`, following the reviewed range
`34055b8..8b2fbd2`. This handover, the appended task result and support-matrix
update are the only later documentation changes. No push is authorized or
performed. Legacy remains read-only at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its pre-existing untracked `.pi/`
and `tests/p2p/` remain untouched. No checkpoint or model artifact was accessed.

## Completed facts

The remaining P2 finding is closed. `moxie-graph` now owns an explicit
`IndexEncoding` descriptor. Its one supported encoding is `U64`, matching the
shared host interpreter's existing `Value::Index(Vec<u64>)`. `ValueRole::Index`
must carry that encoding, so graph construction cannot omit the width and no
floating activation or quantized-weight precision is used to infer it.

`moxie-plan::ExternalInput` retains the encoded role and a checked
`required_bytes` extent. Pure lowering evaluates every external shape, checks
its element product, multiplies by the declared role width with checked
arithmetic, and emits no binding if either calculation overflows. The new
embedding regression proves three `U64` token indices retain a 24-byte extent.
The same graph at `2^62` rows now returns `invalid_request` with a byte-count
overflow before a `PlanCandidate` exists.

All previous correction results remain intact. This change touches no executor,
ledger, arena, CUDA, lifetime or cleanup implementation.

Passed validation:

- `cargo fmt --all -- --check` and `git diff --check`.
- Host and `--features xtask/cuda` workspace clippy, all targets,
  locked/offline, with warnings denied.
- Locked/offline host workspace: 549 unit/integration tests and 8 doctests.
- Locked/offline full device-feature workspace: 561 unit/integration tests and
  11 doctests, including integrated unwind, binding and fault coverage on all
  three GPUs.
- Focused `moxie-graph`, `moxie-plan` and `moxie-interp` suites.
- Architecture check: 55 rejecting fixtures, 14 accepted fixtures and 12
  rules. Specification check: all 10 reference documents unchanged.
- `cargo xtask-cuda test-gpu`: 33/33, with `sm_86` and `sm_120` qualified.
  Restricted two-3090 visibility passed its 22 visible cases and exited 1 as
  required because no `sm_120` device was visible.
- Capacity passed in normal and reversed visibility with identity and budgets
  keyed by GPU UUID.
- A fresh no-toolkit/no-driver target passed 549 host tests and 8 doctests. Its
  explicitly rebuilt `xtask` had no `libcuda` dependency in `ldd`.

No unexpected validation command failed. The restricted-visibility exit was the
required negative result. Skipped and unmeasured remain model quality,
checkpoint import, actual-context inference, semantic CUDA arithmetic, state,
topology communication and paired prefill/decode performance.

## Decisions

No ADR, owner ruling, reference-document amendment, threshold or product
setting changed. `U64` is declared because it is the index representation the
shared semantic consumer actually implements. A future integer encoding must be
added through the shared graph and kernel capability contracts with its own
value validation; it cannot be inferred locally by a planner or backend.

The earlier correction handover is preserved as the record reviewed at
`8b2fbd2`. Its claim that index width was absent is superseded by this correction
rather than silently edited.

## Remaining hypotheses and blockers

Independent re-review remains the task 0011 acceptance gate. No known concrete
finding remains from either review. This resource-plan slice still has no
resident weight owner, kernel selection, workspace, semantic execution or
state, and it is not model support.

## Next task

Re-review `8b2fbd2..64899c2` for explicit index encoding retention and checked
byte overflow. If task 0011 is accepted, write the bounded M1.3 assignment for
shared semantic kernel selection, exact workspace admission and one
device-resident layer chain using event-retained operands. Stop before model or
checkpoint integration, residency/eviction policy, paged state, generation,
multi-device execution, quality claims or performance claims.
