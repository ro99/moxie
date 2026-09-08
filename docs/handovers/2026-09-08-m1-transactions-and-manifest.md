# Handover — M1.4 transactions closed pending review; M1.2 manifest reader next

Written 2026-09-08 at the end of the session that produced task 0004 and task 0005's contract.

## Workspace identity

- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, **clean**, `HEAD` = `3aaf259`.
- Read-only legacy root: `/home/rodrigo/Developer/strata` @ `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
  Never write there. Document 08 is the map into it; read the pinned references before inventing a
  replacement.
- Task 0004 was **accepted at `3aaf259`** and everything through it is on `origin/main`.
- **Do not push without the owner saying so.** The owner stated this explicitly mid-session. Commit
  locally, report, wait for the instruction.

- Toolchain: rustc/cargo 1.97.1 pinned, CUDA 13.0 (nvcc V13.0.88). Devices: ordinal 0 is the
  RTX 5060 Ti (sm_120, NUMA 0), ordinals 1-2 are the RTX 3090 pair (sm_86, NUMA 1).
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is forced in `.cargo/config.toml`.

## Completed facts

Task 0004 (M1.4 part 1, sequence-state transactions) is **accepted at `3aaf259`** -- the bounded
host slice only, not M1.4. Its record is [docs/tasks/0004-m1-state-transactions.md](../tasks/0004-m1-state-transactions.md),
which carries the contract, the result, and three review-correction sections.

What exists that did not before:

- `moxie-state`: `begin` / `commit_prefix(n)` / `abort` / `open_transactions`, journal-based, with
  a transaction that is **append-only** — `rollback_to` and `emit` are refused while one is open on
  the branch, and `invalidate_generation` is refused outright.
- `moxie-interp`: `KvCache::begin` / `commit` / `abort` with a `CacheJournal` bound to a
  process-unique `CacheId` **and** a single-use transaction number; `KvCache::snapshot` replaces the
  removed `Clone`; `KvCache::stamp` (was `commit`) is still `pub(crate)`.
- `Interpreter::run` opens both participants and resolves both the same way. `publish` checks the
  cancellation token after each of its four mutations, so cancellation is a genuine abort.

Gates, all rerun at `3aaf259` on this machine:

| Lane | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | PASS |
| `cargo test --workspace --locked --offline` | PASS, 332 unit/integration + 4 doctests |
| `cargo xtask arch-check` | PASS, 19 rejected + 1 accepted fixtures, 6 rules |
| `cargo xtask spec-check` | PASS, 10 documents |
| no-driver host lane (`CUDA_HOME=/nonexistent NVCC=/nonexistent`, no CUDA on `PATH`) | PASS, 332 + 4; `ldd target/debug/xtask` shows no `libcuda` |
| device lane, `--features moxie-cuda/driver,moxie-kernels/fatbin,xtask/cuda` | PASS, 341 + 4 |
| `cargo xtask-cuda test-gpu` | PASS, 15 cases, `sm_86` and `sm_120` qualified |
| `CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu` | **exit 1**, `UNQUALIFIED sm_120`, as intended |

Nothing is failing. Nothing was skipped. No checkpoint was read, downloaded or converted.

Support matrix updated: `G-HOST-TEST`, `G-HOST-NODRIVER` and `G-INTERP-BF16` counts, plus one
capability row for transactional publication.

## Decisions

- [ADR 0005](../decisions/adr/0005-toml-manifest-with-separate-chunks.md) — canonical manifest v1 is
  TOML in an artifact directory, payloads in sibling chunk files. **Proposed, not implemented.** It
  takes `toml` + `serde` as the first third-party dependencies in a production crate
  (`moxie-format`), with a per-crate allowlist. The reviewer endorsed this over another hand-written
  parser; the owner has not ruled on it. If the owner rejects it, the ADR's "Why" section lists what
  the alternative costs.
- A transaction is **append-only** on both participants rather than journaling discarded content.
  The reasoning is in `Journal`'s doc comment and in task 0004's fifth-review section: document 04's
  mechanism for keeping part of a transaction's work is `commit_prefix(n)`, so a destructive
  rollback inside a transaction has no use case, and refusing it is exact where journaling is
  approximately exact.
- **Paging is an outstanding M1.4 requirement, not M4.** M1.4 asks for "appendable paged state plus
  transaction API"; M4 adds paged *device* attention, host-backed page streaming and growth
  admission. Task 0004 narrowed to the transaction half deliberately. Do not restate paging as M4.

## Remaining hypotheses and blockers

- **Task 0004 is accepted; M1.4 is not.** Appendable paged state, sampler integration, the generation
  service and the diagnostic CLI are all outstanding. Seven review passes ran on this slice, and the
  last three each found exactly one defect, each a consequence of the previous fix — identity added
  to the journal, then the cache handing that identity out by derive. Do not treat "gates pass" as
  acceptance on the next task either.
- The one recurring failure shape in this crate pair: **an identity that exists to be unique, handed
  out by a derive.** It has happened twice (`SequenceState`, `KvCache`). Both now carry
  `compile_fail` doctests. Check any new identity-bearing type against that pattern before adding it.
- Owner gates O1, O2, O4, O5, O6, O7 are all OPEN. None blocks the next task. O5 blocks any bulk
  copy, download, conversion or write of a large artifact — task 0005 is written to stay clear of it.
- No performance number has been measured anywhere in this repository. Do not produce one from
  arithmetic.

## Next task

**Implement [task 0005](../tasks/0005-m1-canonical-manifest-and-bounded-reads.md), M1.2: canonical
manifest v1 and bounded tensor reads.** The contract is already written and committed (`42c478a`)
and **must not be edited once a test has run** — that is the working rule this project uses, from
document 07. Read it in full before writing code; what follows is orientation, not a substitute.

- **One outcome**: a canonical BF16 artifact on disk can be opened, fully validated, and read tensor
  by tensor within a declared byte budget, and every documented rejection rule actually rejects.
- **Owning components**: `moxie-format` gains the manifest schema and its validation, and stays
  I/O-free; a new `moxie-storage` crate gains the bounded reader and is the only crate here that
  touches the filesystem. Both must be declared in `arch-check`'s ownership table, and `arch-check`
  needs the two new boundary rules with fixtures.
- **Required reading**: document 03 (canonical artifact, manifest v1, the closed affine descriptor),
  document 02's `moxie-format`/`moxie-storage` ownership row, document 06 M1.2, document 07 on what
  may be version-controlled, AGENTS.md, ADR 0003 and ADR 0005.
- **Deletion plan, part of the deliverable**: `xtask::speccheck::sha256_hex` moves into
  `moxie-format`; `xtask` consumes the shared one and the published-vector test moves with it. Two
  SHA-256 implementations in the workspace means the task is not done.
- **Stop condition**: stop and report if this appears to need a real checkpoint, a downloaded
  artifact, an importer for a foreign format, `mmap`, device memory, or a model family's config
  schema. The first two need O5.

### Working rules this project holds agents to

1. **Write the contract first, in a commit with no `.rs` change**, and do not retune a threshold
   after a test runs. Tasks 0003, 0004 and 0005 all did this.
2. **Reproduce every reported defect before fixing it**, and keep the reproduction as a named
   regression test. Scratch reproduction files are deleted; the test that replaces one names the
   review that found it.
3. **Verify a test bites.** A `compile_fail` doctest is checked with `compile_fail` removed, to
   confirm it fails for the intended reason. A new error-handling test is checked against a
   deliberate mutation of the handler.
4. **Report passed, failed and skipped separately, with real counts.** An unmeasured architecture is
   never a pass.
5. Preserve negative results and disproven contracts rather than quietly widening a bound.
