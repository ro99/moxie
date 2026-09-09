# Handover — task 0010 independent review

## Workspace identity

Reviewed implementation `4037fcc9341b386bb31dfda180852609f06290e2` against the
separately committed contract `0bab91b`, in `/home/rodrigo/Developer/moxie`,
branch `main`, initially clean. The review changes only this handover; it does
not fix production code, change acceptance criteria, or amend reference documents.
The read-only legacy checkout remains at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, with its existing untracked `.pi/`
and `tests/p2p/` untouched. No checkpoint was accessed.

## Completed facts

**Recommendation: request changes before accepting task 0010.** The ownership
architecture fits the bounded M1.3 assignment, but one resource-growth defect
and one acceptance-evidence gap remain.

The metadata allocator is shared, contains no CUDA/model policy, and uses
non-Clone release authority. The executor consumes the parent reservation before
allocating, checks the selected device scope and tier, retains source capacity,
and returns ranges only after the common lifecycle permits retirement. Explicit
close frees device storage before releasing its charge; ambiguous failed cleanup
withholds both and refuses retries. The task correctly leaves residency, kernels,
admitted execution plans, model support, and the remaining M1 exit gates open.
No model-specific execution path or changed numerical contract was found.

### R1 — P2: generation history retains unaccounted host memory

Location: `crates/moxie-memory/src/arena.rs:310–333`, especially the insertion at
line 333 and the `generations` map at line 174.

Every previously unseen allocation offset creates a permanent BTreeMap entry.
Release removes the live record and coalesces space, but never reclaims generation
history. Thus a long-lived arena's host metadata grows with historical offsets,
even when its maximum simultaneous live allocation count stays at two. This is
normal successful operation, not an intentional failed-operation quarantine.

Independent counting-global-allocator probe against the committed library:

```rust
let mut arena = Arena::new("probe", 1_048_576, 256).unwrap();
for offset in 1..=100_000 {
    let prefix = arena.allocate(offset, 1, "prefix").unwrap();
    let tail = arena.allocate(1, 1, "tail").unwrap();
    assert_eq!(tail.offset(), offset);
    arena.release(tail).unwrap();
    arena.release(prefix).unwrap();
}
```

Net retained requested heap grew **3,429,296 bytes** after arena creation.
At the end, occupancy reports capacity/free/largest-free all 1,048,576 bytes,
one free range, zero live bytes and zero live allocations. The probe measures
live requested heap, not RSS or inference performance. It holds no payload and
uses at most two live handles. Temporary probe source/binary:
`/tmp/moxie-task0010-generation-probe.rs` and
`/tmp/moxie-task0010-generation-probe`; the essential reproducer is above.

The device binding admits payload and upload-source bytes but neither bounds nor
charges this growing history. Device capacity provides only an impractically
large eventual bound: alignment 1 permits an entry per historical byte offset.
This conflicts with document 03's bounded resource authority and the project's
scarce-host-memory/long-session requirements.

Correction: use an arena-wide checked monotonically increasing generation (the
existing allocation sequence can supply uniqueness), or another explicitly
bounded scheme. Preserve stale-handle rejection and higher generation on offset
reuse. Add a repeated variable-offset allocate/release test proving retained
metadata does not grow with history for a fixed live-set bound. Do not simply
clear history and reset generations, which would violate the identity contract.

### R2 — P2: the real-device lifetime gate is weaker than the fixed contract

Location: `crates/moxie-executor/tests/device_arena.rs:101–109,164–167`.

The contract requires pre-completion reuse refusal and complete/cancel/sweep on
**every visible UUID**. The test instead shares one `observed_in_flight` flag
across the entire device loop and accepts an immediately retired first upload
on any individual GPU. Its final assertion requires only one device to have
exposed pending work. It also relies on winning a timing race against a pageable
8 MiB copy, so a valid run where every copy completes quickly fails the suite.

Neither real-device arena consumer exercises `OperationTurn` sweeping. The
`xtask` arena case synchronizes through readback before retirement, and the host
sweep test uses a scalar resource. Those are useful checks, but they do not
establish the full per-UUID real-range gate specified in task 0010.

Correction: add a bounded, controlled pending-work harness on each UUID, prove
that cancellation and a first sweep retain the actual range/source, complete
the work, then sweep again without a next token and explicitly release/reuse
the range. Keep the existing readback, coalescing, transfer and accounting tests.
Report these per-device results separately; do not weaken the committed contract
to the current aggregate assertion.

## Validation

Passed during this review:

- `cargo test --workspace --locked --offline`: 532 unit/integration tests,
  8 doctests.
- `cargo test --workspace --features xtask/cuda --locked --offline`:
  542 unit/integration tests, 10 doctests. This feature enables the CUDA driver,
  kernel fatbins and executor driver together. Includes real arena and driver
  fault tests across all three GPUs.
- `cargo xtask-cuda test-gpu`: 33 cases, no failures or skips, both `sm_86`
  and `sm_120` qualified.
- `cargo xtask arch-check`: 52 negative and 14 positive fixtures, 12 rules.
- `cargo xtask spec-check`: all ten reference documents unchanged.
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`,
  and the same command with `--features xtask/cuda`.
- `cargo xtask-cuda capacity`, normal visibility and
  `CUDA_VISIBLE_DEVICES=2,1,0`.
- Restricted qualification, `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu`,
  exits 1 with `sm_120` explicitly unqualified, as required.

The independent generation probe reproduced R1. Existing suites pass despite
both findings. One initial review command used the nonexistent
`moxie-kernels/cuda` feature and failed dependency resolution before testing;
the corrected full device-feature command above passed. This was a review
invocation error, not a repository defect.

Not rerun: an isolated driver/toolkit-hidden host lane, `ldd`, Compute Sanitizer,
Miri/native sanitizers, or the author's completion-bypass mutation experiment.
No model quality, actual-context execution, topology communication benchmark or
paired inference performance was measured; these remain outside the slice.
Local logs use `/tmp/moxie-task0010-review-*.log`; the commands, outcomes and
reproducer in this handover are the durable review record.

## Decisions

No owner ruling or ADR is needed for these corrections. No change to precision,
context, quality, model catalog, storage authorization or performance policy is
proposed. The findings concern the existing task's resource and validation
contracts. Passing current tests does not close the missing tests above.

## Remaining hypotheses and blockers

R1 and R2 remain open. No device use-after-free was established by this review;
the generation finding concerns retained host metadata, and the second finding
concerns missing and timing-dependent validation. The task is a basic allocator
slice, not completion of M1 or all of M1.3.

## Next task

Correct these two findings in `moxie-memory` and the executor's test harness,
with only the necessary shared test support. Read task 0010 and documents 02/03/07;
preserve R07/R08 source-lifetime and no-next-token guarantees. Re-run affected
consumers, host/device suites and architecture checks; record the independent
resource-growth regression and per-UUID lifetime evidence. Seek independent
re-review before accepting the slice and starting dependent plan implementation.
