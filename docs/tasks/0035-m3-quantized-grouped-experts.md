# Task 0035 — canonical integer weights in shared grouped experts

Status: implemented and measured; owner review pending, 2026-09-19.

## Identity and authority

Owner's M3 recovery assignment; builder on main at 46f0876, writable Moxie
root. Preserve task0034 importer work and coordinator.md. Read-only legacy
strata at 2dc566eb8e440fff4837ac75ca1dad1b20c2264e. Specs02/03/06/08/09,
task0019 semantic oracle, task0021 grouped execution and task0028 affine
linear govern this extension. O6/O7 remain open: no performance claim.

## Bounded deliverable

Extend the existing grouped-expert planner, host kernel, CUDA kernel and
lease-based executor to consume canonical W4A16/W8A16 weights. Preserve one
residency authority and the existing routing, slot reduction and cancellation
mechanism. Shared GeGLU and SwiGLU graphs are the two consumers. No model-owned
runtime, weight-wide dequantization, separate cache or new crate.

## Contract before implementation

Each expert has gate/up [2I,H] and down [H,I] affine matrices. Decode W=(Q-Z)*S
in FP32 directly from packed codes and metadata, then narrow each operand to
BF16 as in the shared dense affine contract. Existing BF16 activation,
gate/up projection, gate transform and final output boundaries remain task0019's.
Each reduction is ascending input order; CPU uses its existing bounded tile and
GPU its admitted intermediate workspace. Group maps preserve original logical
column identity. Codes, scales, zeros and map bytes must all count in residency
and staging before an acquire or launch. No hidden expanded weight allocation.

Borrowed packed views cannot outlive source leases. Completion events retain
leases; cancellation drains or quarantines under the existing mechanism. Rejected
metadata must fail before a partial output is visible. Both GPU architectures
remain distinct catalogue qualifications. Task0028/ADR0028's reduction gate is
unchanged; no new end-to-end quality claim follows from synthetic routes.

## Acceptance

Independent source equations and task0019 gate oracle; full signed code ranges,
asymmetric zeros, scale encodings, group32/128, tails and maps. Compare two gate
transforms, decode/prefill shapes and route/slot permutations, host versus all
three GPUs. Malformed lengths/maps, admission failure, cancellation and lease
release regression. Architecture, affected consumers and clippy required.
No timing/quality/model support claim. Remove superseded branches only after
replacement tests pass. Temporary fixtures are test-owned and deleted.

## Result

Implemented canonical artifact binding and mapped grouped execution; owner review
remains.
Shared host packed views, CUDA scalar decoders, grouped planner byte extents,
lease-based executor and catalogue now accept INT4/INT8 and mixed BF16 integer
projection pairs. The CUDA affine decoder is shared with dense execution and
uses alignment-safe scale loads for packed sections. There is no expanded weight
copy or second residency owner. Existing BF16 entry points remain wrappers.

The full synthetic grouped path passed on both3090s and5060Ti: 294,912 slot and
reduced-output BF16 components match the independent task0019 oracle exactly,
across both gates, host/device placement, INT4/group32/F16,
INT8/group128/F32, INT8/group32/BF16, asymmetric/symmetric and signed scales,
including both directions of mixed BF16/integer projection pairs. Logs:
`results/m3-recovery/quantized-expert-device.log`. Host-only packed tests add odd
tails, explicit group maps, multiple tile widths and invalid late route rejection
before any output write. Shared dense signed-scale execution also passed all
three GPUs under unchanged ADR0028.

Gemma4 and Laguna graph definitions now accept explicit role-to-precision data
through one shared GraphBuilder contract; both compose INT4 gate/up and INT8 down
expert operands. Unknown roles refuse. This is graph composition plus separate
shared operation execution, not execution of either complete model graph.

CanonicalSource now binds complete canonical artifacts to explicit expert role
pairs, verifies projection shapes and formats, and streams checked payloads into
the residency authority's destination. ADR0031 charges optional group maps in
the same weight lease; no separate allocation or map cache exists.

Published synthetic GPTQ fixtures execute the expert and combine operations
extracted from reduced Gemma and Laguna graphs: H128/I67 GeGLU and H35/I65 SwiGLU,
INT4/F16 gate-up and INT8/F32 down, signed scales and mapped group32 columns.
Both host and device placements pass on all three GPUs: 46,944 BF16 components
match the independent oracle bitwise. Cancellation after a group releases all
leases and refuses reduction in every placement/device case. Insufficient host/device
admission refuses with a typed ledger rejection and leaves no reservations.
Evidence: `results/m3-recovery/mapped-expert-admission-cancellation.log`. Wrong shapes,
foreign identities, legacy cache versions, nonzero offsets and omitted map
budgets refuse before touching destination bytes. Complete graph execution is
not implemented by this test.

Affected host suites, driver clippy, architecture and specification checks pass.
The broad host run first found an obsolete CLI diagnostic assertion; it now
checks the exact unsupported auto-gptq format, and repack/storage suites pass.
Still required: owner review and the task0036 closure report.
Per-channel grouped formats and heterogeneous formats within one projection
remain explicitly unsupported: `CanonicalSource` returns the typed capability
names `per-channel grouped weights` and `heterogeneous expert formats` before
residency or launch. No milestone closure, checkpoint token, model-output or
performance claim.

The complete grouped-device suite then ran all 11 tests: 10 passed, while the
kernel-image refusal fixture failed before attachment because it selected the
first catalogue entry instead of preserving capability selection. The fixture
now corrupts image identities across the otherwise unchanged catalogue; its
exact rerun passes and still verifies that attachment refuses the foreign image
and returns every reservation. Evidence: `current-grouped-device-full.log`
(retained failed run) and `grouped-attachment-refusal.log` under
`results/m3-recovery/`. All 11 tests have passing results; this is not a claim
that the original full invocation passed.

The complete suite was run again after final review on candidate `0672d24`:
**11 passed, zero failed/ignored/filtered**. It includes the repaired attachment
fixture, the 294,912-component quantized matrix, the 46,944-component mapped
published fixtures, both consumers, all three GPUs, cancellation, admission and
artifact-binding refusals. Evidence:
`results/m3-recovery/m3-final-grouped-device.log`.
