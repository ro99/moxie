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
half selected in ADR 0022. **Superseded after implementation by
[ADR 0024](../decisions/adr/0024-one-storage-crate-and-a-write-module.md)**, on
the owner's ruling: the writer is a module of `moxie-repack`, not a crate. The
contract is left as it was written; the result section records what changed and
why. Reuse the I/O-free format importer/serializer and
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

Status: **implemented; not reviewed and not accepted.** Everything below is a
claim until a review reproduces it.

### Identity

- Contract committed at `74e0709` by a concurrent agent; base re-recorded and
  [ADR 0023](../decisions/adr/0023-canonical-affine-payload-and-repack-journal.md)
  committed at `aaeff60`, **before** any implementation commit. Implementation
  base `44a94c4`, clean tree.
- Legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`, **read-only and unmodified**. The
  checkpoint I/O ADR 0022 names was inspected as the contract asks:
  `include/strata/platform/checkpoint_io.hpp` (40 lines) and
  `src/platform/checkpoint_io.cpp` (109 lines). It is a **reader and nothing
  else** — shards opened `O_RDONLY`, bounded positional `pread` with `EINTR`
  retried without advancing, premature EOF reported as an error rather than a
  short read, and no writer, journal, publication or rename anywhere in it.
  That confirms ADR 0022's "there is no legacy writer to migrate or delete"
  rather than repeating it, and the two properties it does have are already
  what `moxie-storage`'s pump does. Nothing was ported and nothing was deleted.
- **Nothing under `/models` or `/fast/models` was written, converted, deleted or
  modified.** One shard was opened read-only and read through: 5,369,738,904
  bytes of `model-00001-of-00015.safetensors`, hashed, plus 1,818,624 bytes of
  declared tensor ranges converted.

### What was built

Two new crates, four new modules in the shared codec and three new reader entry
points, with the boundary ADR 0022 drew:

| Component | What it owns |
|---|---|
| `moxie-format::payload` | The canonical affine payload: three contiguous sections, one range, one checksum. The **only** place those bytes are laid out. |
| `moxie-format::journal` | The restart journal's schema: one TOML document per line, parsed from a `&str`, I/O-free. |
| `moxie-format::selection` | What a user asked to be repacked, as text. No discovery, no patterns, no inference. |
| `moxie-format::manifest::encode` | The writer half of manifest v1. It re-parses its own output and compares artifact identity before returning. |
| `moxie-format::compressed_tensors::PackQuantizedPlan` | The streaming importer. `import` is written in terms of it, so a whole-tensor import and a tiled repack are one codec. |
| `moxie-storage::Artifact::{stream_tensor, verify_tensor, open_unpublished}` | Verification without decoding, and validating a manifest that is not published yet. |
| `moxie-storage::{read_range, read_text_capped}` | The bounded-read primitives the writer needs, public, in the crate that owns file I/O. |
| `moxie-repack` | Arguments, the offline workflow, reporting, and its `write` module: confined output creation, bounded chunk writing, the journal's file half, atomic publication. No second decoder, hash, validator or read path. |

The destination directory **is** the staging area: chunk files are written under
their final names and what makes the artifact unreadable is that `manifest.toml`
does not exist yet. Publication is one rename of a privately named, already
validated manifest. Payload bytes never move after they are durable, and the
failure case needs no second copy of the disk.

### Gates

Run at the implementation tree, each result recorded separately; the raw log
is in the ignored `results/task0025/gates.log`.

| Gate | Command | Result |
|---|---|---|
| Formatting | `cargo fmt --all -- --check` | **passed** |
| Host lints | `cargo clippy --workspace --all-targets --locked -- -D warnings` | **passed** |
| Device-lane lints | `... --features moxie-executor/driver` | **passed** |
| `xtask` device lints | `... --features cuda` | **passed** |
| Specification | `cargo xtask spec-check` | **passed**, 10 documents present and unchanged |
| Architecture | `cargo xtask arch-check` | **passed**: 79 rejected fixtures, 21 accepted, 13 rules. The task added a write-authority rule and eight fixtures; [ADR 0024](../decisions/adr/0024-one-storage-crate-and-a-write-module.md) removed all of them with the crate they policed |
| Host tests | `cargo test --workspace --locked --offline` | **1,097 passed, 0 failed, 0 ignored, 0 skipped** (task 0024 recorded 947; 1,014 before the review's corrections, +83 for the seven regressions they needed) |
| Device-feature tests | `cargo test --workspace --locked --offline --features moxie-executor/driver` | **1,057 passed, 0 failed, 0 ignored** |

**Nothing failed and nothing was skipped.** The real-artifact lane ran rather
than skipping: its source is present on this machine, and it prints `SKIP` with
a reason when it is not. No GPU execution gate is claimed — this task executes
nothing on a device, and the device-feature lane is here to show that shared
changes did not break that build, not as evidence about kernels.

The two clippy lanes that need CUDA were run because this task changes
`moxie-format` and `moxie-storage`, which both compile in them.

### What the suites measure

| Suite | What it establishes |
|---|---|
| `moxie-format` unit and `manifest_v1` | The payload codec against **bytes written out by hand**, the journal's torn-tail and version rules, the selection's refusals, and manifest v1 round-tripping through its own writer — including awkward strings, the opaque architecture tree and partial completeness |
| `the_streaming_converter_and_the_whole_tensor_import_agree` | One codec, not two: every tile size that lands on and off a word, a group and the padded tail produces the same bytes as a whole-tensor import |
| `moxie-repack::publication` | 18 tests, including **35 enumerated failure cases** — every visit to every named durable boundary — each checked for no readable artifact before the rename, a resumable destination after, and a byte-identical republish; plus cancellation at the publication boundary, a corrupted staged unit, bytes past the journal, a torn journal line, a rebound plan, a stale lock and an oversized unit |
| `moxie-repack::round_trip` | Every signed INT4 and INT8 code, asymmetric zero points, all three scale encodings, group and axis tails, symmetric and asymmetric modules, split shards, mixed BF16, several chunk files, and a tensor 32 times the scratch |
| `moxie-repack::cli` | The real binary: exit statuses, refusals, an actual `SIGABRT` restart, a changed source refused, budgets required, inspection writing nothing |
| `moxie-repack::budget` | **208,461 B peak live heap** against a 134,217,728 B admission while publishing 2,107,392 B through a 65,536 B scratch, every admitted byte returned, and resume peaks flat across four attempts (58,183 / 57,560 / 57,560 / 57,560 B) |
| `moxie-repack::workflow` | The review's own cases, at the workflow boundary: a source changed after hashing, malformed entry dtypes, a symmetric selection over an asymmetric source, a scratch too small for a zero-point word, a total budget below the working set, and **an injected failure at every named boundary leaving the ledger empty** |
| `moxie-repack::real_module` | The Laguna module below |

### Mutation measurement

**29 of 29 mutants caught, 4 of 4 expected survivors held, 0 survivors, 0
unstable, 0 invalid controls, 0 skipped**, every verdict repeated three times in
both directions, driver committed and its verdict rule self-tested at 15 of 15
cases ([experiment 0006](../evidence/experiments/0006-repack-publication-mutations.md),
[its driver](../../xtask/src/mutationcheck.rs)).

The first run of that battery caught 24 of 28 and left **four survivors and two
skips**, and all six were worth having: an ignored payload-write error survived
every gate because the resume machinery repaired it, the disk-budget rule had no
case of its own because the chunk-file rule fired first in the one that was
supposed to cover it, and the cancellation check between validation and the
rename — the last point at which stopping is free — was covered by nothing.
Three tests closed those. The fourth survivor is declared **equivalent** with
its argument written out: the plan already refuses a selection above the disk
budget and no unit can exceed its planned range, so the runtime counter cannot
fire. The two skips were the battery failing to measure — one anchor reformatted
by `cargo fmt`, one matching two places in one file — and a mutation that does
not apply is not evidence.

**Three of those measurements are now gone with what they measured.** The
battery carried two mutations of the write-authority `arch-check` rule and one
independence control proving the rule was not the dependency allowlist in
disguise — the control added `moxie-storage-write` to `moxie-engine`'s allowlist
and required the battery to stay green. All three held when they ran. ADR 0024
then deleted the rule, because a module inside the only program that writes
cannot be reached by anything else, and there is no longer a rule for a
substitution to weaken. The remaining battery is 26 mutants and 3 expected
survivors, and **it has not been re-run since the merge**.

### What the real module measured

`model.layers.1.mlp.experts.0.down_proj` of `Laguna-S-2.1-AWQ-INT4` revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`, repacked into a temporary directory
under `/tmp` and reopened through the production reader:

| Quantity | Value |
|---|---|
| Canonical payload | **1,966,080 B** — 1,572,864 code + 196,608 BF16 scale + 196,608 i16 zero-point bytes, exactly the contract's header-derived expectation |
| Source payload converted | 1,818,624 B in three tensors, plus the 16-byte `weight_shape` read as metadata (the contract's 1,818,640 B counts that sixteen) |
| Work units | 5, each at most 1 MiB, against a 1 MiB admitted payload scratch |
| Values reconstructed and checked | **3,145,728**, every one of them |
| Canonical FP32 agreement | bitwise, all 3,145,728 |
| Source BF16-boundary agreement | exact, all 3,145,728; the boundary **moves 777,575** of them, and the gate requires that count to be nonzero |
| Code range | `[-8, 7]` — the whole INT4 range appears |
| Zero-point range | `[-6, 5]` — the asymmetric lane is exercised, not a symmetric tensor in disguise |
| Whole-file digest | `279766e8604281c8dc41130f837793174a16068072eca47769b96855b4348748`, **equal to the digest the hub recorded at download** in `.cache/huggingface/download/…metadata` |
| Directory on disk | 1,967,394 B, against the task's 64 MiB scratch cap |
| Wall clock | ~30 s, nearly all of it hashing the 5.37 GB source shard |

The two comparisons are kept apart on purpose. Document 03 fixes canonical
reconstruction at FP32; the pinned `_dequantize` casts to `scale.dtype` — BF16
for this artifact — before subtracting and multiplying, so the source's own
value is a BF16 number. Task 0024's review found the first version of that test
calling one by the other's name, and this record does not repeat it.

### What was found while building it

The publication enumeration — every visit to every named durable boundary,
failed in turn — found three defects that reading the code had not:

1. **A discarded unit whose bytes were already in the tensor's checksum.** A
   resumed unit that fails its own hash is discarded and recomputed; the first
   version folded its bytes into the tensor's running hash *before* comparing,
   so the published checksum covered bytes that were then overwritten, and every
   later check agreed with it. The running hash is cloned and committed only on
   acceptance now.
2. **A journal interrupted before its plan line stranded the destination.** The
   file existing was read as proof that a run existed. A journal with no plan
   line records no run, and `begin` starts over.
3. **A directory holding only this program's own lock file was refused as
   somebody else's data.** The check now names what it found and distinguishes
   its own private files from a user's.

A fourth came from measured coverage rather than from a failure:
`chunk-read-back` is only visited on a resume, so an enumeration derived from a
clean run had **zero** cases for it. The gate prints its coverage table and
refuses to pass with more than one unreached boundary, which is how the gap
announced itself; it has a test of its own now.

### The crate split, and why it is gone

The first implementation put canonical write I/O in its own crate,
`moxie-storage-write`, as ADR 0022 specified and this contract required. **The
owner rejected it on review, and was right.** It had one consumer by design, and
the boundary forced the writer to grow its own `read_exact_at`, byte-range pump
and capped text read — the first character-identical to the reader's, because
the reader's was private. Two copies of "read bytes at an offset, bounded" in one
workspace is the duplication this repository exists to refuse, and the split is
what created it.

[ADR 0024](../decisions/adr/0024-one-storage-crate-and-a-write-module.md) folds
the writer into `moxie-repack::write`, makes the two primitives public on
`moxie-storage`, and deletes the duplicate. Confinement got **stronger**: no
crate outside the program can name the writer, so the `arch-check` rule and its
eight fixtures were removed rather than maintained. The general rule it
establishes: a crate needs several consumers *and* something distinct to own —
one consumer plus a rule someone has to maintain is a module.

### The independent review, and what it found

A review of the six commits made **ten findings, five of them P1**. All ten are
reproduced, all ten are fixed, none is disputed.

| # | Finding | Fix |
|---|---|---|
| 1 P1 | A private file replaced by a symlink was followed: `File::create` truncated the target and validation rejected the escape *after the damage*. Publishing beneath the source root also succeeded. | One `open_confined` helper for every write — component name, `symlink_metadata` refusal, and `O_NOFOLLOW` to close the gap between the check and the open. Destination/source overlap refused both ways, including a destination that does not exist yet. |
| 2 P1 | The cross-shard resolver applied none of `source_entries`' checks: `F32` codes, an `F64` shape, and a symmetric selection over a source carrying zero points all published. | The rules are `validate_source_entries` now, and both resolvers call it. The zero-point entry passed in is the one the **source** has, which is what lets the disagreement be seen. |
| 3 P1 | A source changed after its digest was taken published under the old digest. | Every source is hashed again after conversion and before anything is exposed. A second full pass over every source file; the real-module lane went from 31 to 61 seconds. |
| 4 P1 | A torn journal tail was never truncated, so the next record appended onto the fragment and the journal became unparseable for good. | The tear is truncated and synced before anything appends — **and the run's own handle is repositioned**, which the regression for this finding is what caught: it was still positioned past the repaired end, writing a hole of NULs. |
| 5 P1 | Headers, selections, plans and manifests allocated outside the ledger; `units_of` materialized every unit; the fault recorder retained every visit. `--total-bytes 2048` published, and inspection accepted zero. | Metadata is charged at the caps the format crate declares, units are produced by an iterator, the recorder is a fixed counter per boundary, and `Budgets::validate` refuses a total that cannot hold the working set. |
| 6 P2 | The disk budget covered payload only: 5,005 bytes published against 4,096. | The plan checks payload **plus** a bound on journal and staged manifest. |
| 7 P2 | An injected chunk-write failure left all three reservations outstanding. | The workflow is wrapped: every path releases the buffers and abandons the run, which stays resumable. Measured across every named boundary. |
| 8 P2 | `--scratch-bytes 8` exited 101 — the planner forced a word-sized zero-point block into a four-byte tile. | The minimum tile is computed during planning and refused with the number, before output exists. |
| 9 P2 | Whole-file hashing, recovery and validation read for minutes without consulting cancellation. | All three check between slices, units and tensors. |
| 10 P2 | Only the destination was synced, not the parents of directories the run created. | Every created directory's parent is synced. |

`cargo fmt --all -- --check` was also failing at the reviewed head, in four
files. Seven regressions were added for the fixes that had none: the symlink
escape, repeated interruption during recovery, the dtype refusals, the
symmetric-over-asymmetric refusal, the source-changed refusal, ledger cleanup at
every boundary, and the scratch and total-budget refusals.

### Explicit limits of this task

- **Nothing executes a canonical INT4 tensor.** Publishing one changes nothing
  about that: W4A16/W8A16 is M3 item 3, no kernel exists, and this task adds no
  capability row for execution.
- **A repack is bytes.** ADR 0018 makes a bit-identical repack the v1 quality
  definition; it is not evidence about model output, and no paired output
  against any released model exists (**O2**).
- **A completed selection is not a complete model.** One module publishes a
  *partial* artifact, the reader opens it for inspection and refuses every
  tensor read, and `inspect` says so in as many words.
- **Source discovery does not exist.** A selection names which shard holds each
  tensor; nothing reads a `model.safetensors.index.json`, expands a pattern or
  infers placement. Cross-shard *resolution* works and is tested; cross-shard
  *discovery* is not built.
- **Liveness is not detectable here.** A lock file outlives a crashed run, and
  telling a crashed run from a live one needs process liveness — which means
  this machine's own telemetry, and ADR 0006 gives that to one crate. Taking
  over an interrupted run is therefore an explicit flag, not a guess.
- **AutoGPTQ/AutoRound packing, activation-order maps, a quantizer, fused expert
  role mapping, tokenizers, whole-catalog conversion and single-file `.mox`
  packaging** are all refused by name. M3 item 2's remainder and M11 item 4.
- **No GPU, distributed, prefill/decode, sampling or context effect.** None is
  applicable to an offline CPU task, and no timing here is a performance claim
  (**O6**, **O7**).

### Next bounded task

**M3 item 3 — shared W4A16/W8A16 dense and expert paths for SM86, with SM120
qualified separately.** It is the largest remaining gap in the milestone and now
has something to execute from: a canonical artifact a reader can open, rather
than a fixture built in a test. Document 03 bounds it before it starts —
weight-only paths with BF16 preferred and FP32 accumulation, **not** INT4xINT4
or INT8xINT8 MMA, and "bounded reference dequantization is not an acceptable
final fast path by assertion". M3 item 2's remainder (group-128 symmetric INT4,
whose real content is what `actorder: "static"` means for logical column
identity) is smaller and also open.
