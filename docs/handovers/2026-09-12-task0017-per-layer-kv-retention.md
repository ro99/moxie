# Handover — task 0017, per-layer key/value geometry and window reclamation

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Base `27dc09f` (task 0016 acceptance). Contract and ADR 0014 at `b748536`,
  then the implementation commit this handover accompanies. The working tree was
  clean at the start and carries no unrelated change.
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and
  `tests/p2p/` are untouched. Citations resolve against the frozen commit's
  paths, **not** against the copies in that checkout's own `.pi/worktrees/`.
- No file under `/models` or `/fast/models` was read, copied, converted,
  deleted or modified by this task. No checkpoint is involved at all.
- Raw logs are in the session scratchpad, not in git; the figures and commands
  are transcribed into
  [the task record](../tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md),
  which is the retained evidence.

## Completed facts

The owner selected this task on 2026-09-12 from the three candidates the
[previous handover](2026-09-12-m1.5-gemma-operation-gap.md) listed. The contract
was authored and committed **before** implementation, as required.

**Independent review requested changes and found three defects plus an
acceptance-test gap. All four were reproduced before any change and all four are
fixed**; the task record carries each one. A second round confirmed the three
fixes and found the closing test still short of the contract — it never
fault-injected `commit_checked`, the one operation that advances the accepted
frontier and publishes committed sampler history — so that branch was added too.
It found no new implementation defect. The review found no numerical-parity
or ring-addressing defect. The most instructive is the P1: cloning `KvGeometry`
in the interpreter put an **infallible** heap allocation inside an open
transaction, so exhaustion there aborted the process instead of returning
`CapacityExceeded` and rolling back. Per-layer geometry made a previously `Copy`
type own a vector, and every existing `.clone()` of it silently became an
allocation. **When a `Copy` type grows a heap field, audit its clones for the
paths that must not allocate.**

The paged host KV store now admits a per-layer key/value geometry and a
per-layer retention rule. Pages belong to one layer; each layer has its own ring
and reclamation *is* the ring overwrite, so a sliding layer costs nothing per
token to reclaim and never grows past its window plus a bounded undo headroom.
The engine's uniform-geometry refusal is deleted, the interpreter reads a
layer's retained range with its absolute base, and the Gemma reduced graph
composes its sliding and global layers at their own widths.

**The numerical claim is exact equality**: reclaiming outside a layer's window
changes no output bit, checked against the full-retention dense reference over
two shapes and five page/chunk combinations, with the test asserting that
reclamation actually occurred. No tolerance was added and none was relaxed.

| Lane | Command | Result |
|---|---|---|
| Host workspace | `cargo test --workspace --locked --offline` | **666 + 9 doctests passed**, 0 failed, 0 ignored |
| Retention | `cargo test -p moxie-state --test paged_window --locked --offline` | **10 passed** |
| Wrapped-ring aborts | `cargo test -p moxie-state --lib --locked --offline faults_at_every_boundary` | **1 passed**, nine boundaries across append, sample staging and commit |
| Storage | `cargo test -p moxie-state --test paged_allocation --test paged_window_allocation --locked --offline -- --nocapture` | **2 passed** |
| Gemma integration | `cargo test -p moxie-cli --test gemma --locked --offline` | **16 passed** |
| Allocation | `cargo test -p moxie-cli --test allocation --locked --offline -- --nocapture` | **1 passed**, six shapes inside their envelopes |
| Clippy | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | passed |
| Format / diff | `cargo fmt --all -- --check`; `git diff --check` | passed |
| Specification | `cargo xtask spec-check` | passed, 10 digests unchanged |
| Architecture | `cargo xtask arch-check` on a clean `git archive` | **73 rejecting + 21 accepted**, 12 rules, unchanged |
| Device workspace | the host command with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda` | **682 + 12 doctests passed**, 0 failed |
| Device clippy | the clippy command with the same features | passed |
| Real GPU | `cargo xtask-cuda test-gpu` | **39/39**, 0 failed, 0 skipped; sm_86 and sm_120 qualified |

`arch-check` against the working tree also reports the four pre-existing
findings from the retained task 0014 probe crate under `results/`, exactly as
tasks 0015 and 0016 recorded. Product validation uses the clean archive.

The GPU result is unchanged from task 0016's, which is the expected outcome:
**no device behaviour is added**, and it is regression evidence for accepted
work rather than evidence for this task's mechanism.

**Storage evidence, not a context claim.** Two layers at 32,768 actual stored
rows cost 398,860 B of admitted backing with full retention and **206,360 B**
with one layer windowed at 1,024 — zero allocations inside `append` either way.
Task 0013's 396,788 B for the same geometry moved to 398,860 B because the page
table now has one entry per `(layer, page)`; the pools are unchanged.

Support matrix gains `G-KV-RETENTION` and `G-WINDOW-ALLOCATION`, a row for
per-layer geometry and reclamation, and an amended Gemma reduced-graph row that
now lists **two** reductions instead of four.

## Decisions

- [ADR 0014](../decisions/adr/0014-bounded-tentative-undo-headroom.md): a
  windowed layer reserves bounded tentative-undo headroom and refuses beyond it.
  Appending more rows in one transaction than the admitted headroom is refused
  before any byte is written, which is what keeps `abort` exactly restorative
  without a snapshot; `rollback_to` a prefix whose window has been reclaimed is
  refused and names re-prefill.
- The pinned legacy rewind is **not** carried forward. `gemma4_runtime.cpp:1385`
  saves exactly one evicted row and `:1407` reinserts it for any rewind depth,
  which is correct only for its single-token `future_entropy` caller and
  silently wrong otherwise. Document 04 requires reporting re-prefill instead.
- `Error::Reclaimed` is a new typed variant rather than an `InvalidRequest`.
  The request is well formed and was legal earlier; a caller has to be able to
  tell "you asked for the impossible" from "that history is gone".
- No owner gate was resolved. O1–O7 remain open. No numerical threshold,
  precision, context target or compatibility surface changed.
- One diagnostic parameter is new and declared rather than defaulted:
  `KvGeometry::tentative_rows`, which the generation service sets from the
  request's prefill chunk.

## Remaining hypotheses and blockers

**M4 is not closed by this.** Device paged attention, COW forks and prefix
sharing, host-backed page streaming with the online-softmax merge,
recurrent/convolution/index state snapshot and replay, and MLA all remain. This
task delivered the state schema those need, not the paths themselves.

**M1.5 is not closed either, and shape is no longer what blocks it.** The
Gemma 4 artifact still cannot execute: every language-model linear is
compressed-tensors INT8 `pack-quantized`, group 32, symmetric, four codes per
`int32` along the input axis with BF16 scales, and the importer, packed-layout
reader, canonical repack and W8A16 path are all M3. Vision is M11.

**A finding worth not rediscovering.** A sliding window is *shift invariant* —
the mask reads only `q - k` — so the retention frontier cannot be validated by
comparing outputs. Shortening it by one row still passes the numerical parity
test, because the interpreter reads a layer's history before appending the
chunk's own rows and there is exactly one row of slack at that boundary.
Shortening by two fails. The frontier is pinned by an exact assertion instead,
and both the slack and the reason are documented in the code. **A parity test is
not a frontier test.**

Rejected during implementation, with the reason: snapshotting evicted rows into
a side buffer. It is exact for any rewind depth, but its size is the number of
rows evicted while a transaction is open, which is not known at admission time,
and a lazily grown buffer would be a second store. ADR 0014 records it as
option B and the measured argument that would revive it.

Also worth knowing, from the review corrections: the sequence holds the caller's
layer vector for life, so what it retains is that vector's *capacity*, not its
length -- charging the length left megabytes outside the ledger when a caller
passed an over-reserved vector. Any structure the memory authority takes
ownership of has to be charged for what it retains, or normalized first.

And: two counted-allocation tests cannot share one test executable, because the counter is a global allocator and they race. Each gets
its own binary, as task 0013 already did. And a counted test must print its
report **after** its live-heap assertion — captured harness output retains a
buffer the counter sees as a leak.

## Next task

Author task 0018 as **M3's compressed-tensors INT8 importer**, which is now the
single thing standing between this engine and executing the inventoried Gemma 4
artifact's text tower.

- Owning component `moxie-format`, with `moxie-storage` reading bytes.
- Required reading: document 03's byte-level affine-integer descriptor,
  `moxie-format::affine`'s existing INT8 decode and its exhaustive code tests,
  the packing parameters in [the bring-up record](../models/gemma4.md), the
  hashes in [the inventory](../evidence/checkpoint-inventory.md), and the pinned
  `src/platform/compressed_tensors.cpp`.
- It must produce canonical tensors, not a second runtime decoder, and it must
  not become a model-owned loader or a dequantization fallback.
- **O5 is open and forbids bulk writes.** The task reads the artifact's bytes,
  which is already authorized inspection, and stops at canonical in-memory
  tensors. It may not copy, convert, requantize or publish anything to disk.
  If the work reaches a point where it needs to, stop there and report.
- Oracle: the published tensor values, decoded independently of the reader under
  test. Exhaustive signed code coverage including `-128`, group tails,
  zero-point and scale dtype, exactly as document 03 requires. Declare the
  numerical criterion before writing the decoder, not after.

Stop conditions carried forward from tasks 0016 and 0017: a private loader, a
dequantization fallback, a model-owned execution path, a second resource or
transaction owner, a weakened numerical gate, or a dependency on an unresolved
owner ruling. **No synthetic graph may be described as model support.**
