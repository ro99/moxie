# Handover — task 0010 correction review

## Workspace identity

Re-review covers `4037fcc..0cc61c76f7312e0d827bd8d3cc789b671c787efa`
in `/home/rodrigo/Developer/moxie`, branch `main`, initially clean and writable.
Only this new handover is added by the reviewer. Production code, the fixed task
contract and the original independent review remain unchanged. No commit or push.
Legacy remains read-only at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, with its
existing untracked `.pi/` and `tests/p2p/` untouched. No checkpoint was accessed.

## Completed facts

The corrections close both findings from the
[original review](2026-09-09-task0010-independent-review.md).
No new actionable defect was found in the correction diff. **Accept task 0010's
bounded basic-arena slice at `0cc61c7`; both required corrections are closed.**

**R1 closed:** allocation identity now supplies an arena-wide monotonically
increasing generation. The checked increment precedes free-list mutation and
fails closed at exhaustion. An offset's reuse cannot reset its generation, and
no map of historical offsets remains. The independent review's unchanged
100,000-cycle standalone probe, rebuilt against this commit, reports **944 bytes**
of initial retained metadata rather than 3,429,296 bytes. The new single-test
counting-allocator executable checks zero further retained requested heap after
warm-up, bounded to two live ranges while visiting 100,000 offsets. These are
heap-allocation measurements, not RSS or inference performance.

**R2 closed:** each visible UUID independently executes a real CUDA upload and
records its event behind a controlled stream callback. The first sweep must
withhold the cancelled range/source and return no retired resource. After the
guard releases the stream, synchronization and the second sweep return the real
resource without another token. Explicit release, coalescing and generation-aware
reuse remain covered. The test no longer depends on racing the upload or on a
single aggregate observation across devices. The callback is test-only, calls
no CUDA API, and has a ten-second watchdog whose expiry fails validation.

## Validation

Passed independently during this re-review:

- `cargo test --workspace --locked --offline`: 533 unit/integration tests and
  8 doctests.
- `cargo test --workspace --features xtask/cuda --locked --offline`:
  543 unit/integration tests and 10 doctests, including driver fault coverage.
- `cargo test -p moxie-memory --locked --offline --test arena_history -- --nocapture`:
  zero-growth regression passes.
- `cargo test -p moxie-executor --features driver --locked --offline --test
  device_arena -- --nocapture`: controlled cancellation and both sweeps pass on
  each UUID listed below, with the existing arena assertions also passing.
- `cargo xtask-cuda test-gpu`: 33 cases, no failures or skips, `sm_86` and
  `sm_120` qualified.
- `cargo xtask arch-check`: 52 negative/14 positive fixtures, 12 rules.
- `cargo xtask spec-check`: all ten normative documents unchanged.
- `cargo fmt --all -- --check`; host and `--features xtask/cuda` clippy with
  `--workspace --all-targets --locked --offline -- -D warnings`.
- The original standalone counting-allocator probe rebuilt against this commit:
  944 bytes retained from the initial baseline, zero outstanding allocations.

Controlled sweep identities:

- RTX 5060 Ti: `GPU-97fe4889-4874-a378-198e-955d2e72c4a3`.
- RTX 3090: `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`.
- RTX 3090: `GPU-81fe4578-59b2-37c4-421e-287cdac78704`.

No validation command failed. Not rerun: isolated toolkit/driver-hidden host
build, `ldd`, capacity/reordered/restricted-visibility probes, mutation tests,
Compute Sanitizer, Miri or native sanitizers. Earlier evidence for those lanes
is separate. No model, actual-context, quality, topology-performance or paired
inference benchmark was measured. Local logs use
`/tmp/moxie-task0010-rereview-*.log`; this handover records the commands and
outcomes without depending on temporary-log retention.

## Decisions

No ADR, owner ruling, acceptance relaxation or reference amendment. The
corrections stay within the original shared allocator and test-harness scope.
The original review remains a preserved negative result against `4037fcc`;
this follow-up records its resolution against `0cc61c7`.

## Remaining hypotheses and blockers

No remaining blocker was identified for these two corrections. This assessment
covers task 0010's basic admitted arena, not all of M1.3 or M1. Admission-bound
plan lowering, tensor layout binding, the device-resident layer chain, residency,
multi-stream fan-in, generation service and the other milestone exit gates
remain separate work.

## Next task

Validation passed. The next assignment is the separately bounded
admitted-execution-plan contract described in the
[implementation handover](2026-09-09-task0010-basic-device-arena.md#next-task).
Preserve the shared memory and lifecycle authorities, read the linked normative
contracts and actual sources, and declare graph/resource-plan binding, refusal
and cancellation tests before implementation. Do not expand this review into
model loading, residency policy, kernel optimization or performance claims.
