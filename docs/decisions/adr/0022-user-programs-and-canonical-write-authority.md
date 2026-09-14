# ADR 0022 — User programs and canonical write authority

- ID / date / author / status: 0022 / 2026-09-13; amended 2026-09-14 / engineer lead (implementation agent), packaging amendment directed by the repository owner / adopted; safetensors publication implementation pending.
- Classification: program placement was a roadmap default selected under the owner's explicit delegation. The 2026-09-14 safetensors packaging amendment is an **owner ruling**, replacing the requested custom `.mox` container. Neither is a measured performance claim.
- Scope and owning shared component: application composition roots; `moxie-format` and the storage boundary; `xtask` architecture enforcement.
- Supersedes / superseded by: resolves the placement gap in [ADR 0021](0021-repack-is-a-moxie-program.md) and [the assigned handover](../../handovers/2026-09-13-repack-user-surface-gap.md). Amends document 02's storage row by splitting write I/O into a separate crate, document 05's surface inventory, document 07's command placement, and document 06's M3/M11 packaging distinction **by reference only**. ADRs 0017–0021 retain their owner-gate substance.

**Amendment precedence.** [ADR 0024](0024-one-storage-crate-and-a-write-module.md)
supersedes the original crate split and write-authority rule below: the writer
is `moxie-repack::write`. Those passages remain as the original placement record.
The 2026-09-14 packaging decision below supersedes this ADR's former M11
container deferral, ADR 0024's reference to that deferral, and
[ADR 0005](0005-toml-manifest-with-separate-chunks.md)'s raw-chunk container
choice for future published artifacts. It also supersedes
[ADR 0023](0023-canonical-affine-payload-and-repack-journal.md)'s single affine
byte-range layout where the new safetensors mapping requires separate physical
tensors. Existing manifest-v1 artifacts retain their original interpretation;
the affine mathematical contract and private-journal boundary remain unchanged.

## Problem and mechanism

M3 item 1 requires an offline inspector/repacker but names no executable. The
existing `moxie-cli` package declares binary `moxie`, which currently offers
synthetic generation diagnostics. `xtask` is the development command index;
quality, bench and support verification are still unimplemented entries. Neither
is a canonical writer. `moxie-storage` is currently read-only and is reachable
from execution code. Putting a writer behind a feature in that crate would let
Cargo feature unification expose write APIs in a combined workspace build.

The owner explicitly assigned placement to the engineer lead. This ruling makes
that choice without changing who authorizes or runs full-size repacks. It does
not authorize any bulk operation or network service launch. `publish = false`
controls crates.io publication; it does not prevent distributing compiled
programs in M11.

## Options examined

1. Put conversion in `moxie` alongside chat, or in the HTTP service. This would
   join canonical write authority to generation application dependencies and
   complicate auditing offline-only behavior. Rejected.
2. Make `xtask` the only user repacker. It would require a source checkout and
   Cargo for the released offline workflow and couple user commands to CI.
   Rejected; development smoke tests may invoke the actual program.
3. Dedicated offline program with a physically separate storage writer
   (selected). Reuses the existing codec and reader, isolates dependencies, and
   permits standalone packaging. Costs two small crates when real implementation
   lands, not empty scaffolding now.

These options change no weight equation, precision, execution hardware, context,
prefill or decode behavior. No performance difference has been measured. No new
third-party library or license is selected by this ruling. Existing reader and
format APIs remain; there is no legacy writer to migrate or delete.

## Decision and authority

### Program inventory

Names below are reserved placements, **not claims that commands exist**. Each
program is an ordinary user process, with no elevated privileges. Writable
reports, histories and logs are distinct from canonical checkpoint publication.

| Surface | Binary / crate and role | Build milestone | Package milestone | Authority |
|---|---|---|---|---|
| Local/remote chat and script generation | `moxie` / existing `moxie-cli`; client of the common generation service | M1 diagnostic exists; production chat M8 item 3 | M11 item 4 | Read canonicals through the engine; write explicitly configured history/output; no canonical mutation |
| HTTP service | `moxie-server` / `crates/moxie-server`; protocol composition root over the same service and sampler | M8 item 2 | M11 item 4 | Read canonicals; load/unload are residency/session operations, never repack/delete; loopback default and remote-bind requirements from document 01 |
| Source/canonical inspection, estimate, repack, checksum verification | `moxie-repack` / `crates/moxie-repack`; offline entry point | M3 item 1, beginning with task 0025; source coverage grows with M3 item 2 | M11 item 4 | `inspect` and `verify` read only; `repack` alone invokes canonical write authority, with explicit destination and budget; no inference, downloads or quantization |
| Quality evidence | `moxie-ops quality` / `crates/moxie-ops`; evidence client of shared format/service APIs | M3 repack evidence; future external-quantization gates only when a new O2 package requires them, per ADR 0018 | M11 item 4 | Read source/canonical/reference; write reports outside checkpoint roots; no canonical mutation |
| Paired performance benchmark | `moxie-ops bench` / same crate; shared generation-service client | M6; topology inputs from M5 | M11 item 4 | Execute admitted workloads; write benchmark reports; no canonical mutation |
| Support-matrix verification | `moxie-ops support-matrix --verify` / same crate | M7 | M11 item 4 | Read claims and evidence, report missing/failed/skipped gates; never promote a claim merely because import passed |
| Operator diagnostics | `moxie-ops probe`, `capacity`, `diagnostic` / same crate; host telemetry and admitted service diagnostics | Existing development probes M0–M2; standalone operator interface M6, recovery diagnostics M11 | M11 item 4 | Read machine/artifact state and run explicitly selected admitted probes; write reports; no canonical mutation or system changes |
| Developer validation | `xtask` / existing crate; arch/spec checks, host/device/topology gates and smoke harnesses | Existing M0 onward; topology M5 | Source-development interface, not a required installed user binary | Build/test and scratch evidence only; no production canonical writer dependency |

Do not instantiate the future crates as placeholders. Build each when its first
real command lands. `moxie-ops` has a CUDA-free artifact/evidence lane and an
explicit device lane; missing hardware or missing implementation cannot return a
passing result. Its quality command must distinguish repack preservation from
execution delta and must not restore the v1 task-quality requirement ADR 0018
removed. Ambiguous-layout paired-logit evidence remains required where applicable.

The existing `cargo xtask quality`, `bench`, `support-matrix --verify` command
contracts remain development wrappers when implemented: they invoke the same
operator command logic, never duplicate it. Existing `xtask` probes remain until
the M6 move has parity tests, then become wrappers. M1's `moxie diagnostic` stays
until a separately recorded M8 compatibility/deletion decision; no rename today.

### Repack ownership and write boundary

- `moxie-format` remains I/O-free: source interpretation, affine normalization,
  canonical schema/encoding/validation and checksums. Exactly one affine codec.
- `moxie-storage` retains source index/shard resolution and bounded canonical
  reads. It must not depend on the writer or expose mutation via a feature.
- New `moxie-storage-write` is the **write half of the existing shared storage
  owner**, not another reader, importer or residency authority. It owns confined
  output creation, bounded chunk writing, resume journal and atomic publication.
  Its allowed direct dependencies are `moxie-types`, `moxie-format`,
  `moxie-storage`, and `moxie-memory` for admitted offline resources. It owns no
  CUDA, model interpretation, scheduling or quantizer. A new third-party
  dependency still needs an explicit ADR justification.
- `moxie-repack` owns argument parsing, offline workflow and reporting. Its
  allowed direct dependencies are those four shared crates plus
  `moxie-storage-write` and `moxie-host` for disk/RAM observations. Family role
  mapping, when needed, requires an explicit model-API integration contract; the
  first task operates on declared tensor selections and invents no family map.
- The user may invoke `moxie-repack repack` to an authorized destination under
  `/models` or `/fast/models` according to O5. This is not an OS permission grant.
  Agents continue to treat those roots as read-only; a tooling task is not bulk
  run authorization. No auto-discovery that starts a conversion, overwrite of
  source files, background conversion during model load, or HTTP publish route.
- `verify` validates a published canonical using the production reader and
  hashes payloads; it does not repair or publish. Publication is the final
  validated step of `repack`, with no separate unchecked `publish` command.

### Packaging: safetensors shards and a Moxie manifest

**Owner decision, 2026-09-14:** use safetensors for published tensor files,
including repacked models intended for the Hugging Face Hub. Drop the custom
`.mox` container requirement. This is a format decision now, not a choice
deferred to M11.

A published canonical artifact is a directory containing **one or more
`.safetensors` shards plus a small, versioned Moxie manifest**. Retain TOML for
the manifest. A single tensor file is allowed when appropriate; a whole model
need not fit one file. No custom `.mox` binary container or archive wrapper is
required. Renaming a raw chunk to `.safetensors` does not satisfy this decision:
each shard must conform to the safetensors format and open with its reference
implementation.

The reason is an established data-only tensor format, ecosystem tooling, and a
smaller custom container-maintenance burden. The
[safetensors documentation](https://huggingface.co/docs/safetensors/index)
describes tensor loading without pickle's arbitrary-code execution and supports
selective tensor reads. The
[format specification](https://github.com/huggingface/safetensors#format)
defines typed, shaped tensor ranges and string-valued optional metadata.
This does not establish that the existing TOML/raw-chunk format executes code,
nor does changing containers fix filesystem, allocation or publication defects.

Safetensors stores physical tensors; the Moxie manifest defines their canonical
meaning. Preserve logical roles and shapes, grouping and logical column
identity, packing/version, scale dtype, zero-point mode, provenance, checksums,
exclusions, and completeness. Packed INT4 codes may use `U8` storage, INT8 codes
`I8`, scales their original FP16/BF16/FP32 dtype, and zero points `I16`, with
BF16 weights stored directly. The precise component naming, physical shapes,
shard mapping and checksum scope must be fixed in a versioned schema before
implementation. No regrouping, requantization, scale rounding or change to
`W=(Q-Z)*S` is authorized. A safetensors file alone does not define those
semantics or make the artifact executable by Transformers or another engine.

The implementation belongs to **M3 item 1**, through a bounded follow-up task
with a schema/migration decision before code. M11 item 4 retains program
distribution, installation and migration documentation; it no longer selects a
custom tensor container. Task 0025's existing manifest-v1/raw-chunk results stay
evidence for that implementation only, and do not establish safetensors output.
Its ongoing correctness corrections remain necessary.

The follow-up must establish bounded header and payload access, admitted RAM
and total staging disk, source identity checks, restart and cancellation,
confined writes, and durable publication. Write and validate shards before
atomically publishing the manifest; a private restart journal never becomes
part of the published artifact. Retain checksum and numerical validation:
safetensors permits NaN/Inf values and does not enforce Moxie's scale rules.
Test shards with the reference safetensors reader as well as the production
Moxie reader, and compare all canonical components against independent source
oracles. Record how existing manifest-v1 artifacts are read or explicitly
refused, and the expiry of any transitional path, without silently
reinterpreting or converting them.

This decision authorizes no Hub upload, bulk conversion, source mutation,
download, model-support claim or performance claim. Container adoption does
not resolve the open owner gates.

## Evidence and acceptance

Inspected at workspace base `645e75902ea6bd201f60ac78628178ddc734d985` plus the
existing dirty changes: `Cargo.toml`, `crates/moxie-cli/Cargo.toml`,
`crates/moxie-storage/src/lib.rs`, `xtask/src/main.rs`, `xtask/src/archcheck.rs`,
ADRs 0005/0018/0021, and documents 01/02/03/05/06/07/08/09. Frozen Strata
`include/strata/platform/checkpoint_io.hpp` and `src/platform/checkpoint_io.cpp`
at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e` confirm bounded read-only descriptors
as reuse evidence, not a source for a writer or application placement.

[Task 0025](../../tasks/0025-m3-offline-repack-publication.md) is the first
implementation contract. It must supply writer/reader round-trip, bounded
inspection and restart, atomic publication, actual CLI tests and write-boundary
negative fixtures. This ADR closes the **placement decision**, not those gates,
M3, task 0024 acceptance, or any model-support claim.

## Enforcement and removal

Task 0025 must implement a named `arch-check` rule, **canonical write authority**:
only `moxie-repack` may reach `moxie-storage-write` in the production dependency
graph (besides the writer itself). Chat, HTTP, operator tools, developer command
roots, shared engine crates and model crates must not reach it, directly or
transitively. Resolve aliases, workspace inheritance, optional/target/build
dependencies and source reachability using the existing graph resolver; fail
closed. No second dependency-table parser or scratch-path heuristic.

Test-only writer harnesses may depend on it through dev-dependencies and write
only private temporary directories. Negative fixtures must bind failures to the
new rule and cover a renamed direct dependency, an indirect dependency, a build
edge and a target/optional edge. An accepted fixture proves the legitimate
repacker works. No feature in a runtime-reachable crate may re-export the writer.

Dependency enforcement is not a filesystem sandbox: source review and the
existing source checks must also reject a second canonical writer made with
raw filesystem calls outside the write owner. History/log/report output does
not grant checkpoint mutation authority. M11 runtime tests must prove chat/HTTP
load and diagnostics leave source/canonical files unchanged. Until task 0025
lands, this new rule is a declared gate, **not an implemented check**.

Rollback preserves ADR 0005's reader and directory artifacts; remove only
unreleased writer/app code if its gates fail. Revisit placement through an ADR
if a demonstrated boundary problem requires it. No external repack duplicate,
legacy deletion, or unbounded compatibility bridge is authorized.
