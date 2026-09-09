# Handover — task 0009 independent review

## Workspace identity

Review covered `d525d07..9b24eeb` in `/home/rodrigo/Developer/moxie`, branch
`main`. The repository was clean before review and remains unchanged apart from
this handover. The read-only legacy checkout `/home/rodrigo/Developer/strata`
remains at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its pre-existing
untracked `.pi/` and `tests/p2p/` entries were left untouched.

## Completed facts

No blocking finding or actionable defect was identified. The bounded M1.3
transient-upload slice can advance. Preparation checks admitted device and host
scope/tier charges, including `Vec` capacity, before allocation. Submission owns
the copy and event record, rejects foreign stream/event identity before copying,
and quarantines errors after a copy may have been submitted. Retirement checks
ledger identity and observed completion, performs checked CUDA cleanup before
releasing the reservation, and preserves the resource and charge on cleanup
failure. Submitted, cancelled, and lost drops withhold resources; turn sweeps
return held leases for a later sweep. Returned sources remain caller-owned.

The public API has no unadmitted upload constructor, caller-selected upload
scope, raw upload buffer exposure, or separate caller-enqueued use path. The
ordinary synchronizing `DeviceBuffer` destructor remains the fallback for other
consumers.

## Validation

The following checks passed during the review:

- `cargo fmt --all -- --check`.
- Host clippy and full device-feature clippy with `-D warnings`.
- Host workspace tests: 523 unit/integration tests and 7 doctests.
- Device-feature workspace tests: 532 unit/integration tests and 9 doctests.
- `cargo xtask arch-check`: 52 rejected and 14 accepted fixtures across 12 rules.
- `cargo xtask spec-check`: all 10 normative documents unchanged.
- `cargo xtask-cuda test-gpu`: 30 cases, all three GPUs, `sm_86` and `sm_120`.
- `cargo xtask-cuda capacity`, including `CUDA_VISIBLE_DEVICES=2,1,0`.
- Restricted visibility (`CUDA_VISIBLE_DEVICES=1,2`) exited 1 as required.
- An explicitly rebuilt host `xtask` binary had no `libcuda` dependency in `ldd`.
- An independent real-driver probe covered allocation refusal/source return,
  foreign streams, cancellation, wrong-ledger ordering, query/wait/readback
  failures, failed cleanup, and no Drop retry on quarantine for all three UUIDs.

The test-only `driver_faults` interposition forwards successful calls to the real
driver and injects only selected error codes in its own process. It proves the
error branches, not actual hardware loss or performance. The independent probe
was likewise a boundary probe, not a performance or model-execution result.

## Remaining hypotheses and blockers

No unexpected check failed. Compute Sanitizer, Miri, native sanitizers, a fully
driver/toolkit-hidden lane, and mutation/bite checks were not rerun by this
review. No checkpoint, model execution, quality, topology benchmark, or paired
performance measurement was used. Those are outside task 0009 and remain
unmeasured. The allocator, persistent residency transfer, multi-stream fan-in,
admitted plan lowering, and device-resident layer chain remain future work.

## Next task

Proceed to the separately bounded basic allocator task under the shared
`moxie-memory`/`moxie-executor` ownership contract. Preserve event-retained
resource lifetime, cancellation, and ledger accounting while adding suballocation;
do not expand it into residency policy, model execution, or performance tuning.
