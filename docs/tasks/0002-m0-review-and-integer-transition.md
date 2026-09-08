# Task 0002 — M0 review, corrections, and integer precision transition

Status: **implemented 2026-09-07**, then corrected again after a second review. F1-F6 results are in
[Correction results](#correction-results-2026-09-07); the second review's findings and their results
are in [Second-review corrections](#second-review-corrections-2026-09-07). Review performed 2026-09-07 against Moxie `84273b0e4b41bb04d1b374f6f46f89557bba4a59`, with a clean worktree before documentation edits. The reviewer changed documentation only; the implementation that follows the review is recorded at the end of this file.

## Verdict

**Good M0 scaffold; retain it, but do not treat M0 as fully closed or build M1 on the current state/safety contracts.** The separation into shared crates, closed operation catalog, typed errors, context-borrowing buffers, independent numerical fixtures, explicit unsupported paths and real three-GPU smoke tests are useful work. There are no model-local engines to undo. Synchronous probe-only CUDA execution is reasonable at M0 when its temporary role is explicit; it is not the later event/stream ownership solution.

Several claimed guarantees are stronger than the code enforces. Correct them now through one bounded follow-up, rather than a second rewrite or an open-ended model campaign.

## Reproduced checks

All builds used an isolated target directory `/tmp/moxie-review-ybM1NQ/target`; the source checkout was not modified by tests. Commands use `cargo run -p xtask` because the advertised alias is missing.

| Check | Result |
|---|---|
| `cargo test --workspace --locked --offline` | **PASS: 69 unit tests + 1 compile-fail doctest**. Format crate has 28 total tests, not 28 BF16-only tests. |
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | PASS |
| `cargo run -p xtask --locked --offline -- arch-check` | PASS for current workspace and **8** supplied negative fixtures |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID <isolated-target>/debug/xtask test-gpu` | PASS: AXPY, BF16 and architecture-mismatch expectations on SM120 and both SM86 cards; 9 cases |
| `cargo xtask index` | **FAIL: no such command `xtask`** |
| Three additional architecture-negative probes | **FAIL: all accepted with zero violations**, detailed below |

GPU testing initially returned `cuInit` code 100 inside the sandbox. The same bounded test was rerun with approved device access and passed. This is an environment-access difference, not evidence the actual machine lacks GPUs. No inference, long-context, quality or performance benchmark was run.

## Findings ranked by correction priority

### F1 — P1: architectural enforcement has demonstrated bypasses

Sources at reviewed commit: `xtask/src/archcheck.rs:237` (dependency collection), `:293` (build.rs-only detection), `:331` (top-level dependency keys); existing negative runner accepts any violation rather than checking the expected rule at `:186`.

Using the unchanged checker source in an isolated harness, all three model-crate manifests returned `violations=0`:

```toml
# Case A: real production dependency hidden in a platform table
[target.'cfg(unix)'.dependencies]
moxie-cuda = { path = ".../crates/moxie-cuda" }

# Case B: forbidden package under an allowed dependency alias
[dependencies]
moxie-types = { package = "moxie-cuda", path = ".../crates/moxie-cuda" }

# Case C: custom build-script path, with a real codegen.rs file
[package]
name = "moxie-models-test"
version = "0.0.0"
build = "codegen.rs"
```

The cases were separate complete manifests with minimal source; they did not execute a forbidden build script or launch device work. The checker examined manifest keys instead of resolved package identity, omitted target-specific tables, and only looked for a file literally named `build.rs`.

**Fix:** use Cargo metadata/resolved package IDs plus complete manifest/target analysis without executing untrusted model build scripts. Account for workspace inheritance, renames, path/package identity, target dependencies and configured source/build paths. Use Rust parsing/compiler-assisted checks for imports/FFI/cfg where needed; do not keep growing a hand-written substring parser into a language parser. Add these three negative fixtures and assert the specific expected rule. Keep dev harness allowances narrow. This is enforcement work, not a need for more model code.

### F2 — P1: state frontiers encode the wrong verification/logit assumptions

Sources: `crates/moxie-state/src/lib.rs:104` forbids materialization beyond published/committed tokens; `:133` equates equal counters with valid logits; `:138` truncates counters; `:206` tests rejection by first committing every proposal.

Speculation and entropy branches legitimately execute **unaccepted** tokens. They need tentative branch state, not early publication. The current type could represent a committed view only, but its global comments/tests claim a verification protocol it cannot represent. Separate prompt/accepted history, emitted completion usage and tentative branch execution explicitly.

Counter equality is insufficient for logit validity: `Frontiers::new().next_logits_valid()` is true before any forward pass, and rolling back 20 -> 12 reports validity even though no logits at prefix 12 were retained or recomputed. Recurrent/index state also cannot be restored by changing counters.

A separate program linked against the unchanged state crate reproduced: `empty_state_next_logits_valid=true`, `rollback_without_restoring_logits_valid=true`, and rejection of tentative materialization beyond the accepted prefix. These are counter-only observations, not a test of production KV/recurrent rollback (which does not exist yet).

**Fix:** branch/transaction-qualified frontiers and explicit output provenance/readiness. Only accept tokens after verification. A bonus/mismatch can be committed but pending execution. Logits need a matching retained `(branch, prefix, graph/state generation)` result or recomputation. Restore state according to a component's snapshot/replay/append-only capability; a blanket `SparseIndex` or history-kind boolean must not promise truncation for mutable compression/histogram state. Tests: empty state invalid; tentative execution without publication; rejection every depth; rollback loses logits unless restored; pending bonus; synthetic recurrent replay and entropy branch leaves parent unchanged. Full production paging can still wait for later milestones.

### F3 — P1: host CI is not executable on its declared runner

Sources: `.github/workflows/ci.yml:34`; `xtask/Cargo.toml:11`; `crates/moxie-kernels/build.rs:28`; `crates/moxie-cuda/build.rs:20`; acknowledged in `docs/evidence/toolchain.md`.

Even `arch-check` builds xtask's unconditional CUDA/kernel dependencies. The workflow runs on stock `ubuntu-latest` without installing nvcc or libcuda; workspace tests include those dependencies too. This is not just an unavailable GPU test: the required **host enforcement lane itself** cannot run there. Unit tests of CUDA error mapping still link and call driver functions.

**Fix:** separate pure tooling/host features from opt-in CUDA codegen/link/run. Make host fmt/arch-check/clippy/tests runnable with no driver, toolkit or checkpoint; retain separate real CUDA build/run gates and never count them passed by a host-only run. Extract pure error classification if testing it without a driver. Establish a clean host build test. The choice of implementation mechanism is technical work, not a reason to ask the owner whether host CI should function.

### F4 — P1: safe module loading accepts an unbounded raw image pointer

Source: `crates/moxie-cuda/src/lib.rs:466` accepts arbitrary safe `&[u8]`, but `:471` passes only its pointer to `cuModuleLoadData`, with no length. The safety comment establishes lifetime, not a valid image or PTX termination.

The C API expects a valid supported binary image or NUL-terminated PTX. An arbitrary empty/truncated/nonterminated Rust slice does not satisfy that contract; Rust slice bounds cannot protect an external parser receiving no length. The review deliberately did **not** run a malformed-image crash probe.

**Fix:** narrow the raw loader to an explicitly unsafe trusted-image contract, then expose safe typed loading for compiler-produced validated embedded images and CStr PTX as appropriate. Do not claim generic untrusted binary validation unless actually implemented; executable images need a trust boundary too. Audit context-current guards for module operations and failure cleanup after retaining a primary context. M0 does not need a complete general-purpose CUDA object loader.

### F5 — P2, blocks formal closure: required M0 fixtures and evidence are deferred

Source: `docs/tasks/0001-m0-freeze-evidence.md` marks accepted while M0.6 says routed experts, masks, recurrence, sampler distributions, tokenizer/template and protocol fixtures are not done. Its reason—that these require an executor or checkpoint permission—is not sufficient for tiny synthetic equation/mask/distribution/protocol fixtures. Streams/events are only declarations; no event smoke is exercised, per ADR 0001.

**Fix:** finish the promised minimal corpus (or get an explicit narrower milestone amendment), including a bounded stream/event completion smoke. Synthetic source-linked tests need no large checkpoint or full inference graph. Preserve genuine unavailable real-checkpoint/quality measurements as unmeasured. Update the task's current acceptance status and next step, counts 5 -> 8 negative fixtures, and stale README statements after correction. Add the missing Cargo alias or standardize every command on its real spelling; a fresh clone must execute the documented command.

Also distinguish compiled architectures from actually qualified capabilities. A required GPU profile must fail its acceptance gate when the required architecture/cases are absent or all skipped; the present `test-gpu` can return success for all-skipped devices and only notes missing compiled architectures (`xtask/src/gpu.rs:126`). Do not let a future CI wrapper interpret that exit status as qualification. Record/verify actual nvcc and host-compiler versions and image hashes: a freely overridable NVCC path alone does not pin a toolchain. The build currently emits SASS-only code flags despite its PTX comment (`crates/moxie-kernels/build.rs:65`); correct the claim, not the architecture-mismatch test by accident.

### F6 — P2, resolve before M1 lowering: graph contracts are declarations, not validated capabilities

Sources: `crates/moxie-graph/src/lib.rs:119` exposes `has_host_oracle: bool`; `:130` accepts it as proof; `:143` checks every input with `is_legal_weight`; `crates/moxie-model-api/src/lib.rs:66` default validation returns success.

The closed enum is a useful M0 draft, but a caller can assert an oracle exists and request low-bit activations by passing a legal weight dtype. Output/shape/state/partition checks are not implemented. The default model validator cannot substantiate consistency by itself. Do not mistake these placeholders for a validated graph or add all M1 behavior just to satisfy M0 naming.

**Fix:** label remaining scaffolding accurately, then introduce registered oracle/kernel identities and role-specific weight/activation/cache/accumulator contracts before lowering real operations. Unregistered or incompatible contracts fail closed. Graph construction must carry edges/shapes/state effects, not just a list of operation names. This evolution is M1 work after the M0 safety/state/CI corrections, not permission to write a model-local executor.

## Precision transition: scope and implementation assignment

Authority: owner request, [ADR 0003](../decisions/adr/0003-int4-int8-bf16-weight-family.md), revised documents 01–04/06–09 and [candidate evidence](../evidence/quantization-candidates.md).

Required code changes, **not performed by this documentation review**:

This record is the review/closure umbrella. Execute it as separate bounded subtasks: host tooling/CI and alias; architecture enforcement; CUDA image safety; state frontier/output contract; integer codec/types; missing M0 evidence and closure. Each has its own acceptance results and shared owner. Do not combine them into a model-port campaign or declare the umbrella complete after only one subtask.

1. Replace `Precision::Nvfp4` as an active format with a typed affine-integer descriptor for INT4/INT8 plus BF16. Keep weight storage distinct from activation/cache/accumulator dtype. Preserve cache >=16 bits; no FP8/W4A4/W8A8 scope expansion.
2. Replace `moxie-format/src/nvfp4.rs` as the active canonical codec. Do not mechanically rename E2M1 values: signed INT4 codes are different mathematics. Remove the active NVFP4 exports/tests/capability examples after integer replacement gates pass; Git and historical experiment records preserve the work. A retained legacy inspector must be clearly non-production/deferred.
3. Refactor `int8.rs` to consume the same affine/group/scale contract, including group 32, zero-point handling and valid -128 input. Preserve useful BF16 rounding and general error/dimension/identity tests. Quantizer clipping preference does not constrain all imported codes.
4. Implement exhaustive small INT4/INT8 host reconstruction tests, unsigned rebias equivalence, zero-point extremes, group boundaries/tails, scale dtype, overflow/nonfinite checks and group-index mapping. Initial canonical host decoder is shared between dense/MoE consumers; no new model crate needed.
5. Finish missing M0 fixtures and enforcement/safety fixes F1–F5. Introduce the role-specific type direction from F6, without pretending the full graph/kernel implementation is part of M0.
6. Write bounded importer tasks for M3: compressed-tensors integer first, AutoRound/AutoGPTQ packing second. Inspect the pinned metadata/index/header and exporter version before decoding bytes. Preserve quantization exclusions and proposal heads. Do not download/convert full checkpoints or implement all ten model families now.
7. Record actual results, fix active claims, and update the support matrix/owner gates. The precision-family preference is resolved; catalog/order, quality, intrinsic cache semantics and storage authority are not.

## Acceptance and stopping conditions

Writable implementation root is Moxie only. Legacy Strata is read-only reference. Task owner/reviewer and branch/base commit must be filled before implementation. No whole-model runtime, throughput claim, driver change or checkpoint download belongs to this task.

Exit: F1–F5 regression cases pass; host commands work without CUDA; GPU smoke still passes on both architectures; common integer host contract and tests replace the active NVFP4 assumptions; remaining M1 scaffolding is accurately labeled; required M0 corpus complete or formally re-scoped with explicit authority. Report all failed/skipped/unmeasured lanes. Then propose the small BF16 graph/interpreter M1 slice. Do not declare all candidate checkpoints supported from codec tests.

## Scope caveats, not blanket findings

- The specification is ignored locally by recorded owner direction. This review does not reverse that direction or edit `.gitignore`. Record a specification version/digest and check its required presence during setup so a fresh clone cannot silently proceed without normative files; ask before changing publication policy.
- Hand-written CUDA FFI is not inherently disallowed. Its safety obligations must be correct; a small narrow wrapper can be retained.
- CUDA graph capture, event-retained production caches, full TP/PP, actual model quality and 32K inference are later-milestone work. Their absence alone is not an M0 failure.
- New integer requirements do not make the earlier NVFP4 implementation disobedient: it followed the earlier design. The change is now explicitly authorized and localized.


---

# Correction results, 2026-09-07

Implemented against `84273b0e4b41bb04d1b374f6f46f89557bba4a59` on branch `main`, writable root
`/home/rodrigo/Developer/moxie`. The legacy checkout at `/home/rodrigo/Developer/strata` was not
read from or written to by this work. No checkpoint was downloaded, converted or executed; no
driver or system configuration was changed; no throughput was measured or claimed.

## Commands and results

Host lane, `cargo` 1.97.1 / `rustc` 1.97.1:

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | **PASS** |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | **PASS** |
| `cargo test --workspace --locked --offline` | **PASS**, 191 tests |
| `cargo xtask arch-check` | **PASS**, workspace clean; 13 negative fixtures rejected for their declared rule; all 6 rules exercised |
| `cargo xtask spec-check` | **PASS**, 10 normative documents present and matching their recorded digests |
| host build with `CUDA_HOME=/nonexistent NVCC=/nonexistent`, no CUDA on `PATH`, `LD_LIBRARY_PATH` unset | **PASS**, 191 tests; `ldd target/debug/xtask` reports no `libcuda` |

Device lane, on this machine's three GPUs:

| Command | Result |
|---|---|
| `cargo test --workspace --locked --offline --features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | **PASS**, 197 unit tests + 1 compile-fail doctest |
| `cargo xtask-cuda test-gpu` | **PASS**, 15 cases, 0 failed, 0 skipped; `QUALIFIED sm_86`, `QUALIFIED sm_120` |
| `cargo xtask-cuda test-gpu --profile sm86` | **PASS** |
| `cargo xtask-cuda test-gpu --profile sm120` | **PASS** |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **FAILS with exit 1**, as intended: `UNQUALIFIED sm_120` |
| `cargo xtask-cuda probe` | **PASS**, topology and H2D unchanged from M0.2 |

Test counts by crate, host lane (191 unit tests, 0 doctests): `moxie-types` 22, `moxie-format` 52,
`moxie-graph` 11, `moxie-model-api` 5, `moxie-oracles` 60, `moxie-state` 16, `moxie-cuda` 9,
`moxie-kernels` 1, `xtask` 15.

The device lane (197 unit tests + 1 doctest) adds 2 in `moxie-cuda` (the image-header boundary
cases), 4 in `moxie-kernels` (image presence, compiled-versus-targeted architectures, build identity,
capability) and the `DeviceBuffer` compile-fail doctest, which only exists when the driver feature
compiles the type. The other seven crates are identical in both lanes: the feature split does not
change what they test.

For comparison, the reviewed commit had 69 unit tests + 1 doctest.

### Not run, and not claimed

| Lane | Status |
|---|---|
| `cargo xtask test-topology` | **not implemented** (M5) |
| `cargo xtask quality` | **not implemented** (M3); also needs a released reference model, which O1/O2 gate |
| `cargo xtask bench` | **not implemented** (M6) |
| Any inference, any context length | **unmeasured**; nothing here executes a model |
| Any checkpoint import | **unmeasured**; the ten candidates' tensor bytes remain uninspected |
| Device lane in CI | **not run**; no self-hosted runner exists. The workflow does not contain it and does not claim it. |

## F1 — architecture enforcement · closed

`xtask/src/archcheck.rs`. Dependency edges are now resolved to a package **identity** rather than
trusted by the key they are written under: the alias, an explicit `package = "..."` rename, and the
`[package] name` of a `path` target's own manifest are all collected, and every one of them must be
permitted. Every production dependency table is read, including `[target.<cfg>.dependencies]` and
`[target.<cfg>.build-dependencies]`; `dev-dependencies` stay exempt in every table, which is the
narrow allowance document 02 grants. Workspace inheritance is followed to `[workspace.dependencies]`,
with paths resolved against the **workspace root** rather than the member. Build scripts come from
`[package] build` in all its forms (absent with autodetect, `false`, a path, a list). Model source
scanning covers the whole crate directory except `target/`, `tests/` and `benches/`, so a
`[lib] path` pointing elsewhere is no longer invisible. An edge whose identity cannot be established
fails closed.

The negative-fixture runner no longer accepts "any violation": each fixture declares
`[package.metadata.moxie-arch-check] expect-rule`, and a fixture rejected for a different reason is a
failure. Thirteen fixtures, up from five; the three added from this review are the ones it
reproduced, plus two that give the remaining rules a fixture:

| Fixture | Rule it proves |
|---|---|
| `model-hides-dependency-in-target-table` | forbidden dependency (review case A) |
| `model-renames-forbidden-dependency` | forbidden dependency (review case B) |
| `model-has-custom-build-script` | model crate has a build script (review case C) |
| `undeclared-crate` | undeclared crate |
| `unresolvable-path-dependency` | unresolvable dependency |

Twelve unit tests in `archcheck.rs` cover the manifest analysis directly, so the rules are regression
tested without the fixture runner. `cargo metadata` was deliberately not adopted: it wants a lock
file, a registry and a writable directory, none of which belong in a check that runs offline over an
untrusted tree. Nothing executes a build script.

Found while implementing: resolving an inherited `path` against the member directory instead of the
workspace root made every `foo.workspace = true` in the real workspace unresolvable, and the
fail-closed rule reported all ten as violations. Fixed, with a regression test.

## F2 — state frontiers and output provenance · closed

`crates/moxie-state/src/lib.rs`, rewritten. Four counters per branch replace two:

- `prompt` -- prompt tokens inside the accepted prefix;
- `accepted` -- the accepted logical prefix, prompt included;
- `emitted` -- completion tokens published to the user and counted in usage;
- `executed` -- tokens whose forward pass has run.

`executed` may exceed `accepted` (tentative speculative or entropy execution) and `accepted` may
exceed `executed` (a bonus token pending execution). Neither is corruption, and the old type
forbade the first, which is why its rejection test had to commit every proposal before rejecting it.

Logit validity is a **retained result**, `LogitsHandle { branch, prefix, generation }`, not counter
equality. `Frontiers` has no logit method at all: the retained handle lives on `SequenceState`,
where the branch and generation are known. `record_logits` refuses a prefix nothing executed;
`rollback_to` discards the handle; `restore_logits` accepts a saved one only for the right branch
and the current generation; `invalidate_generation` stales every handle.

`StateKind::restore_capability()` replaces `is_truncatable()`. `SparseIndex` and `SamplerHistory`
moved to `Explicit` alongside the recurrent kinds: a mutable compressed index and accumulated
penalty counters are not restored by shortening anything, however history-like the name is.
`rollback_to` takes restore evidence and refuses to proceed when an `Explicit` component is
uncovered or its snapshot/replay is *after* the target prefix.

16 tests, including the ones the review named: empty state invalid; tentative execution without
publication; rejection at every depth 0..=4; rollback loses logits unless restored; pending bonus
token; a synthetic recurrent replay whose arithmetic shows the earlier state is unrecoverable; and
an entropy branch that leaves its parent's counters and retained result untouched.

Production paging, COW page tables and real snapshots remain M1/M2/M4 and are labelled as absent.

## F3 — host CI without CUDA · closed

Three default-off features -- `moxie-cuda/driver`, `moxie-kernels/fatbin`, `xtask/cuda` -- and two
cargo aliases, `cargo xtask` and `cargo xtask-cuda`. The `CUresult` classification and UUID
formatting moved to `moxie_cuda::status`, which declares no `extern "C"` and emits no link
directive, so the error-mapping tests run with no driver present. Details and the verification
command are in [toolchain.md](../evidence/toolchain.md).

The separation is bounded so it cannot become a way to skip the device lane: a device command run
from the host build exits 2 and names the command that would work; `moxie-cuda/driver` panics at
build time when `libcuda.so.1` is absent rather than producing an unlinkable build; and the CI
workflow asserts that the host `xtask` links no CUDA driver.

The missing alias is added. `cargo xtask index` lists both lanes and which commands need which.

## F4 — CUDA image loading · closed

`crates/moxie-cuda/src/driver.rs`. `Module::load` now takes a `ModuleImage`, which is either a
`TrustedImage` or a `&CStr` of PTX. Both satisfy `cuModuleLoadData`'s contract by construction: the
`CStr` is NUL-terminated as the C API requires for PTX, and `TrustedImage::from_build_output` is
`unsafe`, with the obligation stated -- a complete, unmodified cubin or fatbin from the pinned
build, not a run-time file. The old `&[u8]` form is gone; `Module::load_raw` remains for a raw
pointer and is `unsafe`.

A fatbin/ELF magic check rejects an empty or obviously-wrong buffer with a typed error before the
driver's parser sees it. It is documented as a **sanity check, not validation**, and nothing claims
this accepts untrusted input. `CUDA_ERROR_INVALID_IMAGE` (200) and `INVALID_SOURCE` (300) were added
to the module-error classification, so a rejected image is `UnsupportedKernel` rather than the
numerical catch-all.

Context guards audited: `Module::function` now makes the owning context current before
`cuModuleGetFunction`, which it did not. `DeviceContext::new` releases the primary-context reference
when `cuCtxSetCurrent` fails; it previously leaked it, keeping the context and its memory alive for
the life of the process.

The composition root is where the trust assertion lives: `xtask/src/gpu.rs` wraps
`moxie_kernels::SMOKE_FATBIN` with a written justification, because it is the thing that knows those
bytes are this build's `include_bytes!` output.

A `non_ptx_text_rejected` GPU case exercises the PTX path against a real driver. As in the review, no
malformed-image crash probe was run.

## F5 — required fixtures, evidence and claims · closed

**New crate `moxie-oracles`**, 60 tests, holding the M0.6 fixtures task 0001 deferred. The stated
reason for deferring them -- that they needed an executor or checkpoint permission -- was wrong;
none of these needs either.

| Module | What it pins |
|---|---|
| `route` | top-k selection with the lowest-id tie rule, renormalisation over the selected experts, dispatch grouping, the expert union being smaller than rows x k when routes overlap, and per-row coefficients surviving dispatch and combine |
| `mask` | causal and sliding-window visibility, chunked-versus-whole prefill parity at every width, a later chunk not being the first chunk shifted (R21), short tails, page boundaries, and an exact softmax attention where a masked position contributes exactly zero |
| `recurrent` | replay-from-saved-prefix equivalence for every (saved, target) pair, the non-invertibility that makes R20 true, and a short convolution whose state is a bounded window |
| `sampler` | legality mask, shared pre-truncation normalizer, top-k/top-p/min-p against *that* distribution, temperature last, argmax tie rule, and typed failures for empty vocabulary, NaN, `+inf` and an all-banned set |
| `template` | a byte-level tokenizer, multi-byte round trip, a split character failing to decode, user text never colliding with a special token, deterministic role delimiters, and prefix identity including tokenizer and template versions |
| `protocol` | legal SSE frame order, finish reasons, usage arithmetic excluding drafts/rejections/entropy branches, and a stop string held across chunk and character boundaries |

Each module's header states what it does **not** pin. The registry function registers only the six
operations this crate actually implements a reference for, and a test asserts that nothing else is
registered.

**Stream/event smoke**: implemented in `moxie-cuda` (`Stream`, `Event`, async copies with the R07
lifetime obligation stated as an `unsafe` contract) and exercised as a `test-gpu` case on all three
devices. ADR 0001's "declarations only" note is now discharged.

**Compiled versus qualified**: `moxie_kernels::TARGET_ARCHS` (declared) is separate from
`compiled_sm()` (what nvcc emitted, `fatbin` feature only), and neither is qualification, which is a
passing `test-gpu` case on real hardware. `test-gpu` now fails when a required architecture has no
passing device -- demonstrated by hiding the SM120 card and observing exit 1.

**Toolchain identity**: `nvcc --version` is checked against the pinned `release 13.0, V13.0.88` at
build time; the host compiler version and both fatbin SHA-256 digests are recorded and printed by
`test-gpu`. The SASS-only claim is corrected in the comment, not in the flags -- embedding PTX would
have quietly disarmed the architecture-mismatch assertion.

**Specification presence**: `cargo xtask spec-check` verifies the ten normative documents exist and
match [their recorded digests](../evidence/specification-version.md). It reads and hashes only; it
does not fetch, copy or generate a document, and the owner-directed publication policy is unchanged.

## F6 — graph and model contracts · direction introduced, honestly labelled

`has_host_oracle: bool` is gone. `OpContract` names an `OracleId` that must be present in an
`OracleRegistry` the contract cannot write to; an empty registry, or one holding an oracle for a
different operation, refuses lowering. `OracleEvidence` records where the independent implementation
and its exhaustive test live. This is **weaker than executing the oracle**, and the module says so:
M1 replaces it with a callable reference once there is an interpreter to call it from. What it
already prevents is a contract certifying itself.

Precision roles are separate types in `moxie-types`: `WeightPrecision`, `ActivationPrecision` and
`CachePrecision`, each constructible only through its own rule. `ActivationPrecision::new(Int4)` is
an error, so the F6 case -- requesting low-bit activations by naming a legal weight dtype -- no
longer compiles into existence. `ExecutionProfile` pairs them for W4A16/W8A16/BF16 and declares
`needs_weight_dequantization()`, so no kernel can read W4A16 as an INT4xINT4 MMA.

`PartitionRule::NotDetermined` is the default and fails closed for TP lowering. `StateEffect` must
be declared for a state-touching operation.

`ModelDefinition::validate()` with its `Ok(())` default is removed. Admission is
`moxie_model_api::admit(model, oracles)`, owned by the shared side, and it fails when the registry
has no oracle for an operation the model needs.

Still absent and labelled as absent: edges, shapes, layouts, workspace derivation, and any lowering.
The crate header lists them rather than implying they exist.

## Precision transition · implemented

**`Precision::Nvfp4` is gone**, replaced by `Int4` and `Int8` as signed two's-complement weight
codes. `crates/moxie-format/src/nvfp4.rs` is deleted from the active API; its E2M1/E4M3FN tables and
the reconstruction it pinned are in git at `84273b0`, and the finding it produced is preserved in
[experiment 0001](../evidence/experiments/0001-nvfp4-scale-conventions.md).

**`affine.rs` is not a renamed NVFP4 decoder.** It implements document 03's equation,
`W[o,k] = (Q[o,k] - Z[o,group(k)]) * decode_scale(S[o,group(k)])`, with a subtractive zero point, a
descriptor-supplied group size, and a group index map. Its header states why signed integer codes
are different mathematics from E2M1.

**`int8.rs` is refactored, not kept.** Its decoder is gone; the INT8 profile is the same affine
decoder at a different width, and it accepts `-128`. The symmetric `[-127, 127]` clipping moved to
`quantize.rs` as a declared property of a named quantizer, with a test asserting that the
quantizer's choice does not constrain the decoder. Its ties-to-even, saturation, round-trip and
error tests are preserved there; `bf16.rs` is untouched.

**`scale.rs`** preserves the source scalar encoding as a closed FP16/BF16/FP32 tag with the raw
payload retained, so a repacker can write the same bytes back. An exhaustive FP16 decoder covers all
65,536 patterns against the format definition. A bug was found and fixed while writing it: the
subnormal normalisation loop halved every subnormal scale.

Acceptance items from ADR 0003, each with a test: exhaustive INT4 (16 codes) and INT8 (256 codes)
decoding; signed/unsigned rebias equivalence over both code spaces; zero points at both extremes,
unclipped; group 32, group 128 and per-output-channel boundaries; scale-dtype preservation, shown by
an FP16 scale that BF16 cannot represent; activation-order permutations honoured and inconsistent
maps rejected; partial final groups and odd INT4 rows; checked lengths and overflow; bounded
preparation into a caller's one-row buffer; and a **value-preserving repack** proving canonical
reconstruction equals the source's own arithmetic bit for bit.

Group sizes are a closed set of {32, 128}; widening it is an ADR. No FP4/FP8 path, no W4A4/W8A8, no
cache below 16 bits, no method-specific runtime, and no checkpoint was downloaded or converted.

## What remains open

- **O1, O2, O4, O5** are untouched by this work and remain OPEN. Nothing here selects a catalog,
  approves a quality loss, enables intrinsic low-bit state or writes bulk storage.
- **M3 importers** are not written. The ten candidates have inspected metadata and nothing more;
  their tensor headers, zero-point packing, `g_idx` maps and exclusion lists are unread.
- **No kernel** implements W4A16 or W8A16. The host decoder is a reference, not a fast path.
- **The device lane has no CI runner.** Its results are hand-run and recorded with the hardware they
  were measured on.


---

# Second-review corrections, 2026-09-07

A second review of the F1-F6 work independently reproduced the passing checks and then found six
places where a contract still accepted an invalid state. **Every finding was reproduced here before
being fixed**, and each now has a regression test that fails against the previous behaviour. The
reviewer's summary -- "substantially better, but I would not close M0 yet" -- was correct.

## Commands and results after the second pass

| Command | Before | After |
|---|---|---|
| `cargo test --workspace --locked --offline` | PASS, 191 | **PASS, 210** |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | PASS, 197 + 1 doctest | **PASS, 218 + 1 doctest** |
| `cargo xtask arch-check` | PASS, 13 fixtures | **PASS, 15 fixtures**, 6 rules |
| `cargo xtask spec-check` | PASS, 10 documents | **PASS**, digests unchanged |
| `cargo fmt --all -- --check`, `cargo clippy ... -D warnings` | PASS | **PASS** |
| host build with `CUDA_HOME=/nonexistent NVCC=/nonexistent`, no CUDA on `PATH` | PASS, 191 | **PASS, 210**; `ldd` shows no `libcuda` |
| `cargo xtask-cuda test-gpu` | PASS, 15 cases | **PASS, 15 cases**, both architectures qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | exit 1 | **exit 1**, as intended |

Per-crate host counts: `moxie-types` 22, `moxie-format` 54, `moxie-graph` 11, `moxie-model-api` 5,
`moxie-oracles` 62, `moxie-state` 25, `moxie-cuda` 9, `moxie-kernels` 1, `xtask` 21.

Nothing in the "not run / unmeasured" table above changed. No checkpoint was imported, no inference
ran, no throughput was measured, and no owner gate moved.

## S1 - retained results did not identify a sequence or a prefix's contents · closed

Reproduced, both halves:

```text
handle from sequence A, presented to sequence B  -> restore ok, next_logits_valid = true
handle for a discarded suffix, after rollback and
  re-execution to the same length                -> restore ok
```

A `(branch, prefix, generation)` triple is not an identity. Every fresh sequence starts at the same
root branch and generation zero, and a prefix *length* says nothing about which tokens occupy those
positions. The reviewer also noted, correctly, that the rollback test built its "saved" handle out
of struct literals, so it tested the assertion rather than provenance.

Three changes in `crates/moxie-state/src/lib.rs`:

- **`SequenceId`**, allocated from a process counter rather than supplied by a caller who could
  duplicate it. Two `SequenceState` values never share one.
- **`ResultId` and an issued-result ledger.** `LogitsHandle`'s fields are now private with no public
  constructor, so only `record_logits` mints one; the sequence keeps the set it has issued and not
  invalidated, and a handle that is not in it -- fabricated, or belonging to a discarded branch --
  is refused. The rollback test now has to record a real result and keep it.
- **`PrefixLineage`.** A chain value over the positions of a prefix, where each position carries the
  epoch it was written in. A rollback bumps the branch's epoch, so positions re-executed afterwards
  contribute differently from the ones they replaced. Prefixes at or before the rollback target keep
  their lineage, which is the other half of the requirement: replacing a suffix must not invalidate
  results for prefixes it did not touch.

It is a *lineage*, not a content digest, and the type says so. Document 04's prefix-reuse key --
checkpoint, tokenizer/template, configuration and token IDs -- is a separate identity that composes
with this one and is still not implemented.

Tests: `a_result_from_another_sequence_is_refused`, `a_result_for_a_replaced_suffix_does_not_come_back`,
`a_result_at_an_unchanged_prefix_survives_a_rollback`, `a_handle_cannot_be_fabricated`,
`a_fork_does_not_inherit_the_parents_retained_result`.

## S2 - an earlier snapshot was accepted as restoration to a later prefix · closed

Reproduced: a snapshot `taken_at: 4` satisfied a rollback to prefix 6, and the execution frontier
advanced to 6 over state that stopped at 4.

`Restore` is now a struct carrying the component, the sequence, the generation and a
`RestoreMethod`. The method names what the restoration **completed**, not merely what was available:

```rust
RestoreMethod::Snapshot { of_prefix }   // an exact snapshot *of* the target
RestoreMethod::Replay   { from, to }    // recomputed from `from`, finished at `to`
```

`rollback_to` requires `completed_prefix() == target` for every `Explicit` component, plus matching
sequence and generation, a coherent replay (`from <= to`), exactly one entry per component, no entry
for a component outside the schema, and no entry for a truncatable one. A refused rollback changes
nothing.

Tests: `an_earlier_source_is_not_a_completed_restoration`,
`restore_evidence_must_name_this_sequence_and_generation`,
`every_explicit_kind_in_the_schema_must_be_covered_exactly_once`,
`a_restore_for_a_component_outside_the_schema_is_refused`.

## S3 - architecture enforcement still missed ordinary production code · closed

Reproduced, using two model crates that compile:

```rust
use std::{fs};                                   // contains the text `std::fs` nowhere
pub fn read_weights(p: &str) -> std::io::Result<Vec<u8>> { fs::read(p) }
```

```toml
[lib]
path = "tests/production.rs"                     # the whole library, in a skipped directory
```

Both returned zero violations. The dependency-identity work from F1 was sound; the *source* layer
was still a substring blacklist over a directory walk that exempted by name.

Two structural changes in `xtask/src/archcheck.rs`:

- **Imports are classified structurally.** `use_paths` flattens `use` declarations into
  fully-qualified paths -- nested groups, renames, globs, leading `::`, `crate`/`self`/`super` -- so
  `use std::fs;`, `use std::{fs};`, `use std::fs as f;`, `use std::{io, fs::{read, write}};` and
  `use std::*;` all land on the same rule. `MODEL_FORBIDDEN_PATHS` holds prefixes; `path_reaches`
  decides whether an import reaches one, including a glob at or above it. This is a parser for one
  construct with a small regular grammar, which is the alternative to the substring list the review
  said not to keep extending. `MODEL_FORBIDDEN_TOKENS` keeps only the residue that is not an import:
  an `extern "C"` block, and a fully-qualified call written inline without a `use`.
- **Production sources come from declared targets.** `declared_target_paths` reads `[lib]`, `[[bin]]`
  and `[[example]]` paths; `production_sources` withdraws the `tests`/`benches` exemption from any
  directory that actually holds one, and includes a declared path even when it sits outside the
  crate directory. The exemption document 02 grants is for the dev harness, and a declared library
  is not one.

Two new negative fixtures, `model-imports-through-a-group` and
`model-declares-its-library-under-tests`, are the reviewer's two crates. Six new unit tests cover
the parser directly, including `an_innocent_import_is_not_flagged` and
`a_use_inside_a_comment_or_string_is_not_an_import` -- a checker that flags everything is not
enforcement either.

Not claimed: this classifies *imports*, not arbitrary expressions, and it does not follow
re-export chains. Both are stated in the module.

## S4 - the safe PTX entry point had no boundary · closed

`ModuleImage::Ptx(&CStr)` passed its pointer to the same auto-detecting `cuModuleLoadData`. A `CStr`
guarantees termination, not that the bytes are text, and the Rust enum label is never passed to
CUDA -- so a NUL-terminated buffer beginning with ELF or fatbin magic reached the binary-image
parser through what looked like the text path.

New `PtxSource` type in `crates/moxie-cuda/src/driver.rs`. Constructing one requires the bytes to be
valid UTF-8, to **not** begin with any image magic the driver recognises, and to carry the
`.version` directive every PTX module opens with. `ModuleImage::Ptx` now takes one, so the driver's
format sniffing cannot route this variant to the binary parser.

The `.version` check is documented as **necessary, not sufficient**: it does not make the text valid
PTX. The `non_ptx_text_rejected` GPU case now asserts both halves -- text without `.version` and
text beginning with ELF magic are refused before any driver call, and text that *is* shaped like a
PTX module but does not compile comes back from the real driver as `UnsupportedKernel`. It passes on
all three devices. As before, no malformed-image crash probe was run.

## S5 - publication and usage accounting · closed

Reproduced: accepting and releasing one completion token, then rolling back, silently reset the
released count to zero. The client already had that text.

- `rollback_to` now **refuses** a target below `prompt + emitted`, naming the prefix through which
  output has been released. Regeneration is a new response, not a rollback.
- The contract mismatch the reviewer identified is resolved in favour of the specification:
  `Frontiers::completion()` -- committed completion tokens -- is what usage counts, matching document
  05 and `moxie-oracles::protocol`. `emitted` is a *delivery* counter, documented as such, and
  `withheld()` is the gap a stop string opens between them.

Tests: `published_output_cannot_be_unpublished_by_a_rollback`,
`usage_counts_committed_completion_tokens_not_released_ones`.

## S6 - reference paths returned successful nonfinite results · closed

Both reproduced:

```text
sampler:  logits [8, 9], temperature f32::MIN_POSITIVE  -> Ok([NaN, NaN])
affine:   code 7, scale f32::MAX                        -> Ok([inf])
```

- **Sampler.** The division happened before the stabilising subtraction, so `9.0 / f32::MIN_POSITIVE`
  overflowed to `+inf` and `inf - inf` gave `NaN`. `normalize_with_temperature` now subtracts the
  maximum first, bounding every numerator at zero, and `check_distribution` refuses to return
  anything that is not finite, non-negative and normalised. A vanishing temperature now agrees with
  greedy; a huge one flattens toward uniform.
- **Affine.** A scale can be finite and positive and still produce an infinite product.
  `reconstruct_row_into` now reports `InvalidArtifact` naming the position, the codes and the scale.
  A companion test confirms the check does not fire on realistic scales.

Tests: `an_extreme_temperature_stays_a_distribution`,
`a_wide_logit_range_does_not_overflow_at_a_small_temperature`,
`a_reconstruction_that_overflows_is_an_invalid_artifact`,
`realistic_scales_reconstruct_without_tripping_the_overflow_check`.

## Smaller corrections

- **`OracleRegistry::register` mutated on rejection.** It used `insert(...).is_some()`, which wrote
  the new evidence and *then* reported the duplicate -- so the registration it was meant to protect
  was silently replaced. Now it checks first. Test:
  `a_rejected_registration_leaves_the_registry_unchanged`.
- **The recorded host compiler was a guess.** `moxie-kernels/build.rs` read `CUDAHOSTCXX` to describe
  the compiler and never passed the selection to nvcc, so `HOST_COMPILER_VERSION` described nvcc's
  assumed default rather than what ran. It now passes `-ccbin` explicitly and adds
  `rerun-if-env-changed` for `CUDAHOSTCXX` and `NVCC_CCBIN`. The fatbin digests are unchanged, which
  confirms `c++` was already the compiler in use -- it is now recorded as a fact rather than an
  assumption.

## What the second pass did not change

The reviewer's own list of what was right stands: the shared affine INT4/INT8 decoder, preserved
source scale encodings, the full INT8 decoding range, role-specific precision types, the host/CUDA
feature separation, negative-fixture rule assertions and real architecture qualification.

O1, O2, O4 and O5 remain **OPEN**. No importer exists. No kernel implements W4A16 or W8A16. The
device lane still has no CI runner. [Task 0003](0003-m1-bf16-reference-interpreter.md) stands as the
next bounded task, and now inherits state and provenance contracts that were tightened rather than
the ones the second review warned against carrying forward.
