# Task 0025 — M3 item 1: bounded offline repack and canonical publication

Status: **proposed, contract only; no implementation or acceptance claimed**.

## Identity and authority

- Task ID 0025; M3 item 1. Engineering assignment: shared offline publication
  slice. Reviewer/acceptance: repository owner under the existing review process.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`. **Implementation
  base re-recorded on 2026-09-13 as `44a94c4` with a clean tree**, which is the
  third recording this contract has asked for and the one implementation
  actually started from: it was authored at
  `645e75902ea6bd201f60ac78628178ddc734d985` while task 0024's corrections were
  still in flight, re-recorded at `adf47cb` when they landed, and the entry
  point was rewritten at `44a94c4` afterwards. ADRs 0021/0022 and this contract
  are committed at `74e0709`; [ADR 0023](../decisions/adr/0023-canonical-affine-payload-and-repack-journal.md),
  which fixes the canonical affine payload layout and the journal this contract
  asks for, is committed with this re-recording and **before** any
  implementation commit.
- Legacy root `/home/rodrigo/Developer/strata`, read-only at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; checkpoint roots `/models` and
  `/fast/models` remain read-only to the implementing agent.
- Requirements: R11 bounded checkpoint I/O, R16 explicit packing, R25 one
  version-controlled owner; document 03 offline restart/validate/publish contract.
- Read documents 01/02/03/06/07/08/09; [ADR 0005](../decisions/adr/0005-toml-manifest-with-separate-chunks.md),
  [ADR 0018](../decisions/adr/0018-v1-quality-is-bit-identical-repack.md),
  [ADR 0021](../decisions/adr/0021-repack-is-a-moxie-program.md),
  [ADR 0022](../decisions/adr/0022-user-programs-and-canonical-write-authority.md),
  [the originating handover](../handovers/2026-09-13-repack-user-surface-gap.md),
  tasks 0005/0018/0024, `moxie-format::{affine,manifest,compressed_tensors}`,
  `moxie-storage`, its import/round-trip tests and `xtask/src/archcheck.rs`.
  Read pinned compressed-tensors files/hashes in task 0024 before extending
  bounded import; inspect frozen Strata checkpoint I/O sources listed by ADR 0022.
- O1–O5 already resolved. No full artifact materialization, source copy or download
  is authorized. O6/O7 remain open; timings here cannot establish speed targets.
- Commit this contract before implementation or acceptance tests. Placement is
  settled by ADR 0022. **Task 0024 is accepted by the owner on 2026-09-13**,
  after three rounds of independent review, so the dependency this contract
  recorded as unmet is met: its importer contract and corrections are the
  accepted asymmetric lane. Its acceptance is **importer-only** — no W4A16
  execution, no model-output quality, no cross-shard resolution — so consuming
  it here inherits that boundary rather than widening it.

## Bounded deliverable

One outcome: the real **`moxie-repack`** executable inspects an explicit selection,
repackages it in bounded work units, resumes an interrupted run, validates it
through the production reader and publishes a manifest-v1 directory atomically.
Commands are `inspect`, `repack`, and `verify`; there is no unchecked publish.

The sole canonical write-I/O owner is `moxie-storage-write`, the storage write
half selected in ADR 0022. Reuse the I/O-free format importer/serializer and
`moxie-storage` reader. The binary handles arguments/reporting; it cannot contain
a second decoder, hash implementation, manifest validator or filesystem writer.

Allowed files: new `crates/moxie-repack/` and `crates/moxie-storage-write/`;
shared format codec/manifest modules and storage index/bounded-reader modules as
needed; their tests; workspace Cargo metadata/lockfile; `xtask` architecture
rules/fixtures and command index; README, this task's result, handover and
support-matrix evidence. Dependency additions follow ADR 0022. No generation,
model graph, CUDA, sampler, cache or kernel changes.

The first source set is BF16 safetensors and the existing compressed-tensors
INT8/symmetric and INT4/asymmetric profiles exercised by tasks 0018/0024. Refuse
unsupported metadata explicitly; this task does not add AutoGPTQ/AutoRound,
activation-order source interpretation, a quantizer, fused expert role mapping,
a tokenizer, full catalog conversion, or single-file packaging. Preserve any
already-supported canonical group map; never silently discard one on import.
The output may be a selected module fixture and must say so. A completed
selection is not a complete model. Model completeness and run completion are
separate fields/identities; partial artifacts remain unsuitable for generation.

Consumers: published bytes opened by the existing canonical reader; the CLI
`verify` path independently uses that reader. Two tiny shapes and mixed BF16,
INT8 and asymmetric INT4 fixtures exercise the shared path. Include a synthetic
split-shard module; task 0024 measured that companions need not share a shard.
Delete test-created directories after verification; retain only code, small
redistributable synthetic fixtures and compact evidence. No duplicate temporary
repacker is allowed.

## Contract before implementation

### Values, identity and layout

Preserve document 03's `W[o,k] = (Q[o,k] - Z[o,group(k)]) * decode_scale(S)`.
Signed code ranges, little-endian packing, i16 zero points, scale dtype/bits,
logical columns, grouping, tails and source provenance are unchanged. Compare
canonical reconstruction in FP32 bitwise and the result at the source's declared
rounding boundary separately. Never call those two quantities identical before
rounding. BF16 payload bits are preserved exactly. No execution profile changes.

Manifest-v1 encoding must round-trip through the existing parser/validator and
preserve all identity fields, opaque architecture metadata, exclusions and
completeness. If current schema lacks resume metadata, keep a versioned private
journal separate; do not overload canonical completeness or silently weaken
reader validation. A required schema change needs its own ADR before code.

Source resolution uses validated index entries and each companion's own shard;
reject missing/duplicate/mismatched entries, unsafe paths, overflow, overlap,
truncation and inconsistent declared shapes. No remote checkpoint code executes.
Keep the source handles and range identities bound throughout each read/hash;
recheck source identity on resume. A revision string, filename or mtime alone
is not checksum evidence. Hash the selected source ranges and consumed metadata,
record that scope honestly, and require full-file checksums before any eventual
whole-artifact claim that names them. Reject source changes between inspection,
resume and publish; document the immutable-source assumption and its limits.

### Budgets and inspection

Require explicit bounded header/index, payload scratch, output chunk and disk
budgets; checked byte arithmetic includes codes, scales, zero points, journal,
manifest and simultaneous staging. Use `moxie-memory` admission for offline
allocations without creating a residency cache. Queue depth is one initially;
no checkpoint-sized collection or whole-tensor expansion. Import and hashing
must stream bounded tiles while preserving input/output packing axes. Reserve
fallibly; failure diagnostics must remain constructible under allocation failure.

First acceptance profile: at most **128 MiB** total admitted dynamic working
memory, including metadata; at most **1 MiB** payload scratch; output payload
chunk files at most **4 MiB** for these proof fixtures. Manifest v1 assigns a
tensor to one chunk: stream multiple bounded work units into that file, rather
than inventing tensor segmentation or splitting one tensor across manifest rows.
The chunk-file limit is an explicit disk-plan parameter, not a RAM allocation;
a larger selected tensor must require a larger admitted file limit.
Open a source header with an explicit **64 MiB** header budget, not a raised
`HeaderBudget::DEFAULT`. Metadata and payload coexist within the total bound.
A tensor larger than scratch must succeed; metadata larger than its cap must
refuse before payload work. Finalize the executable budget formula and allocation
sites in the implementation record before running its gates, without raising
these ceilings to pass.

`inspect` has no output-directory side effects. Report source/selection identity,
supported/refused profiles, canonical bytes including metadata, staging disk
peak, admitted RAM peak, and read/write byte counts. Time is either an estimate
with an explicit supplied/measured rate and scope or **unmeasured**, never an
invented rate. Header-only inspection cannot report verified payload checksums.
`repack` revalidates its plan and refuses insufficient budget before creating
payloads. A later disk-full condition remains a typed failure, not a promise
that a free-space estimate prevents it.

### Publication, resume, cancellation and failure

Use a private staging area in the destination filesystem, exclusive run
ownership, immutable uniquely named chunk files and a versioned journal. Refuse
an existing published destination; no overwrite or in-place model update in this
slice. Confine every created path to the explicit destination, reject source/
destination overlap and symlink escapes, and never delete unrelated files.

A durable work unit is written and checksummed before its journal entry can be
committed. Resume binds source content, importer/converter version, selection,
canonical schema/layout and output plan; rehash reusable completed units and
reject mismatches. Interrupted units are discarded/recomputed within the bound.
No journal entry may serve as evidence its payload is correct without checking.

Validate all output through the production reader before exposing the final
manifest. Sync payloads, journal as needed, and the temporary manifest before
the final atomic manifest publication on the same filesystem; sync its directory
for the stated Linux durability guarantee. No loadable final manifest may exist
before every referenced chunk is complete. If the publication syscall succeeds
but the following durability confirmation fails, report **published, durability
unconfirmed** and reconcile on restart; never delete a possibly published output
as though it were an uncommitted attempt. Unsupported publication/filesystem
semantics refuse explicitly. Readers see no final artifact or a fully validated
selection, never a half-written manifest.

Cancellation is checked between bounded reads/writes and before publish; return
resources and leave only a documented resumable private state. Cancellation
observed after the publish boundary reports the committed outcome. Read/write,
hash, allocation, sync, rename, disk-full and restart failures exercise this same
state machine. Do not release accounting for retained buffers/files prematurely.

### Real-module proof and storage authorization

Read-only source: `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`, revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`, module
`model.layers.1.mlp.experts.0.down_proj`. Its four entries are in
`model-00001-of-00015.safetensors`. Header inspection on 2026-09-13 measured:

| Source suffix | Dtype / physical shape | Payload bytes |
|---|---|---:|
| `weight_packed` | I32 `[3072,128]` | 1,572,864 |
| `weight_scale` | BF16 `[3072,32]` | 196,608 |
| `weight_shape` | I64 `[2]` | 16 |
| `weight_zero_point` | I32 `[384,32]` | 49,152 |

Total selected source payload **1,818,640 B**. Expected logical `[3072,1024]`
canonical payload: 1,572,864 code bytes + 196,608 BF16 scale bytes + 196,608 i16
zero-point bytes = **1,966,080 B**, excluding manifest/alignment/journal. These
are header-derived expectations, not converted output measurements; verify
`weight_shape`, revision and selected content hashes in the real run.

Destination: one private test-created directory under `/tmp/moxie-task0025-*`,
never under either checkpoint root; cap total task scratch disk at **64 MiB**
including interrupted attempts. Retention: remove created payloads on completion,
including failure cleanup; retain compact hashes, byte counts and logs in ignored
`results/task0025/` plus a tracked result summary. Crash tests must track their
own temporary directories for subsequent cleanup. Stop if these limits or the
selected module cannot satisfy the proof; do not expand to a shard/full model.

No GPU, distributed, prefill/decode, sampling or actual-context effect. Such
measurements are not applicable to this offline task. No model output or full
revision O2 acceptance claim follows from a module round-trip.

## Acceptance

1. `cargo xtask spec-check`, `cargo xtask arch-check`, `cargo test --workspace`,
   `cargo fmt --all -- --check`, and host clippy with warnings denied. Run both
   declared CUDA clippy lanes if shared changes affect their compilation;
   no GPU execution gate is inferred from this CPU-only publication task.
2. Implement ADR 0022's named write-authority rule and accepted/negative fixtures,
   including renamed, transitive, target/optional and build dependency paths.
   Measure substitutions that remove each protection; a fixture rejected for
   an unrelated rule is not evidence this rule works.
3. CLI integration tests invoke the real binary: read-only inspection, estimate
   refusal, bounded repack, successful production-reader `verify`, explicit
   unsupported/missing source, source/destination collision and existing-output
   refusal. Test nonzero status and structured distinction between failed,
   cancelled, partial, verified and published/durability-unconfirmed outcomes.
4. Tiny fixtures cover every signed INT4/INT8 value, asymmetric zeros, all source
   scale dtypes already supported, group boundaries/tails, mixed BF16 and split
   shards. Compare exact metadata/payloads to independently built expected bytes,
   not merely writer and reader agreeing with each other. Preserve existing
   importer/reader regression lanes. Reopened bytes must match source arithmetic.
5. Enumerate publication state × failure point × cancellation × restart, check
   invariants after every operation and print measured coverage. Inject failures
   at every allocation and durable I/O boundary, including after publication;
   corrupt journal/source/output separately. A valid preexisting published
   artifact must remain unchanged. Exercise actual process interruption/restart
   in addition to in-process fault injection.
6. Mutation-measure the gates for omitted checksum/source/version checks,
   premature manifest publish, false journal completion, missing sync/error
   handling, budget bypass and cancellation ordering. Report caught, surviving
   and equivalent substitutions separately; fix survivors or establish
   equivalence. Fixtures must check each axis's effect, not just count visits.
7. Measure peak live allocations and disk use, compare with admission/estimates,
   and stream a synthetic tensor larger than scratch in multiple work units;
   use multiple tensors to exercise multiple manifest chunks.
   Repeated cancellation/resume must not grow retained allocations or disk
   without accounting. Time is diagnostic; no paired performance claim.
8. Execute the exact real-module temporary-directory proof above, compare **all
   3,145,728 reconstructed values** under the canonical equation bitwise and
   source BF16 rounding separately, and verify every reopened payload/checksum.
   Missing source is a reported blocked real lane, not acceptance. This is not
   model support or full-artifact source-oracle proof.
9. Update command documentation and support matrix only for measured offline
   capabilities. Record remaining M3 item 1 catalog/completeness integration,
   M3 item 2 source formats, and M11 packaging work explicitly. Delete scratch
   outputs and any superseded rewrite-owned path; preserve legacy and sources.

Stop for missing source-oracle evidence, incompatible schema/layout, an exceeded
proof budget, or a requirement to mutate checkpoint roots. Report the smallest
bounded follow-up; no owner gate, quality threshold or format requirement may
be silently widened to make this task pass.

## Result, filled after work

Not implemented. This contract was authored with ADR 0022 to resolve the assigned
handover's placement blocker. No task-0025 test, mutation score, repack output,
model support or acceptance result is claimed. Record the committed contract
identity and implementation baseline here before filling measured results.
