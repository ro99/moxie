# Task 0042 — M4.3b generalize host-backed streaming to N blocks

Status: **accepted** (owner, 2026-09-21). Built by Codex `luna`,
independently reviewed and re-reviewed by Codex `sol`; two rounds — round 1
found a blocking (P1) scope violation (the production N-block staging API
took every future block up front, which is read-ahead by construction, not
merely a test-harness choice, and the task's contract explicitly excludes
it) and round 2's repair — a lazy, one-block-at-a-time
`NBlockStream::stage_next` surface — was re-reviewed and accepted with a
full call-sequence trace confirming block k+1 is never read before block k
has settled. This closes M4.3's core mechanism (bounded N-block streaming,
exact merge, honest transfer diagnostics, legal-plan-or-rejection
admission); it does not wire host-backed streaming into `moxie-plan`'s
graph admission the way task 0040 did for MLA — that remains open.

## Identity and authority

- Task0042, second bounded M4.3 task; roadmap deliverable 3 of 5
  ([06-implementation-roadmap.md](../spec/06-implementation-roadmap.md) `M4.3`).
  Builder Codex `luna` (max, `/ponytail:ponytail`); independent reviewer
  Codex `sol` (high, read-only, `/ponytail:ponytail-review`); coordinator
  Claude Opus. Repository owner accepts.
- Writable repository root `/home/rodrigo/Developer/moxie`, branch `main`,
  base `6317e03` (task 0041's acceptance commit). Confirm `git status` clean
  and `HEAD` unmoved before starting; report if not. The tree may carry
  unrelated dirty work from a separate session (an ADR touching the roadmap
  file, observed 2026-09-21) — preserve it, do not stage or revert it.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`.
- Requirement repaired: task 0041 proved the mechanism for **exactly** one
  resident block plus one staged block —
  `PagedAttentionLaunch::two_block_stream` (`crates/moxie-executor/src/paged_attention.rs:296`)
  caps `staged_rows` at one device page by construction, and
  `PagedAttentionRun::attend_two_block` (`:2980`) refuses anything else.
  Real declared context will routinely need more than one staged block. This
  task generalizes the *mechanism* task 0041 already validated to an
  arbitrary bounded count of staged blocks, reusing one bounded staging
  buffer sequentially — it does not add speed.
- **Explicitly out of scope, by roadmap boundary, not this task's
  judgment:** transfer/compute overlap and prefetch/read-ahead are
  document 06's own **M6.3** ("Measured transfer overlap, bounded
  read-ahead and preparation... Compare no-prefetch/no-overlap baselines and
  record wasted work"), a later milestone gated behind M5. O6 (performance)
  remains an open owner gate — nothing in this task may be timed, and
  "sequential, one buffer reused per block" is the correctness baseline
  M6.3 will later compare against, not something this task should try to
  beat.
- Required documents: [document 04](../spec/04-attention-parallelism-and-speculation.md)
  line 23 (the merge equations — unchanged from task 0041, this task only
  changes how many blocks fold into them).
- Required source reading: [task 0041](0041-m4-host-backed-page-streaming.md)
  in full (the two-block mechanism this task generalizes — read its Result
  section for the exact API surface and the two review rounds' findings,
  especially the numerical-gate defect: any new comparison this task adds
  must be a real asserted gate against a representation-consistent bound,
  not measured-and-printed). `crates/moxie-oracles/src/online_softmax.rs`
  (`Partial::merge` already folds an arbitrary chain — `attend_row_blocked`
  is the existing N-block *oracle* reference; this task's device path
  catches up to what the oracle already proves, it does not extend the
  oracle). `crates/moxie-state/src/paged.rs` `PagedSequence::read_block`
  (`:1094`, already a general `(first, rows)` reader — reusable per block
  as-is). `crates/moxie-memory/src/report.rs` `HostBackedPlan` (currently one
  `NonZeroU64` staging bound sized for one block — this task's admission
  extension point).
- O1–O5 resolved; O6/O7 open — no timing, no performance claim, no framing
  that a bounded staging loop is "efficient."

## Bounded deliverable

- **One concrete outcome:** a query attends correctly over a declared
  history split into one resident block plus **N** staged blocks (N bounded
  and admitted, at minimum proven for N=3 on real hardware — enough to
  distinguish "generalized" from "task 0041 plus one more special case"),
  each staged block read from the host paged store into **one reused
  bounded device staging buffer** (sequential: stage, launch the partial
  kernel, fold into the running merge, discard, repeat — never N
  simultaneous staging buffers), producing the same answer as the existing
  single-shot whole-history device path within `attention_error_bound`,
  exactly as task 0041 already validates for N=1.
- **Sole owning shared component:** unchanged from task 0041 —
  `moxie-state` for the host source, `moxie-executor` for staging and launch,
  `moxie-memory` for admission. This task extends those three, it does not
  add a fourth.
- **Allowed production and test files/modules:** the same files task 0041
  touched (`crates/moxie-executor/src/paged_attention.rs`,
  `crates/moxie-kernels/cuda/paged_attention.cu` only if the additive
  partial-kernel entry point itself needs a signature change — prefer
  reusing it unchanged, called N times, over modifying it),
  `crates/moxie-memory/src/ledger.rs` and `report.rs`
  (`HostBackedPlan` generalized to N blocks), `crates/moxie-state/src/paged.rs`
  if `read_block` needs a small extension, plus corresponding tests. Do not
  touch `online_softmax.rs`'s algebra.
- **Explicit non-goals and forbidden shortcuts:** no prefetch, read-ahead or
  overlap of any kind (M6.3, stated above — do not smuggle it in as
  "obviously better while I'm here"); no unbounded N (admission must name a
  maximum and refuse or produce a legal plan, never an open-ended loop); no
  MLA streaming; no COW forks or prefix reuse; no recurrent/index state; no
  tensor cores; no performance claim, no "faster than task 0041" framing —
  this is a correctness generalization, and a sequential N-block loop is
  expected to cost more bytes/time than fewer blocks, which is fine to
  state and not something to explain away.
- **Existing consumers and second-consumer/shape proof:** task 0041's
  exact two-block case (N=1 staged block) must still pass unchanged — this
  task's own regression proof that generalizing did not silently narrow or
  alter the N=1 path.
- **Temporary paths to delete or bridge expiry:** none.

## Contract before implementation

- **Equations:** unchanged from task 0041 and `online_softmax.rs` — an
  N-fold `Partial::merge` chain, exactly as `attend_row_blocked` already
  proves in FP64. No new algebra.
- **Shapes, precision, accumulation/rounding, logical/physical layout:**
  same as task 0041 — BF16 cache/activations, FP64 merge, RNE narrowing to
  BF16 before any cross-path comparison (task 0041's round-1 review finding
  applies identically here: do not compare an unnarrowed FP64 value against
  a narrowed BF16 one, and do not add tolerance slack to paper over a
  representation mismatch).
- **Partition and hardware capabilities:** both SM86 GPUs and SM120 — this
  is real device work, same bar as task 0041.
- **Peak memory and transfer dependencies; source/lease lifetime:** exactly
  one staging buffer's worth of device memory, reused N times — this is the
  property that makes it "bounded" per document 04's own text ("bounds
  working memory but still transfers/scans required history"). Report total
  bytes transferred across all N staged blocks, honestly, and per-block if
  that is cheap to add.
- **Cancellation, failure and rollback behavior:** an injected staging
  fault on any one of the N blocks must refuse cleanly without corrupting
  the running merge state or leaking the buffer — generalize task 0041's
  single-fault test to at least a mid-sequence failure (not only the first
  or last block).
- **Independent oracle; predeclared numerical metrics/thresholds:**
  `attention_error_bound`, applied per lane, representation-consistent —
  exactly task 0041's repaired gate, extended to N blocks, not a new
  tolerance.
- **Application compatibility and sampler implications:** none.

## Acceptance

- Device tests on both SM86 UUIDs and SM120: N=3 (minimum) staged blocks
  plus one resident block match the single-shot whole-history path within
  `attention_error_bound`, asserted per lane, no BF16-representation slack.
  Task 0041's own N=1 case still passes unchanged. A mid-sequence injected
  staging fault refuses cleanly, buffer state uncorrupted. Peak device
  memory for staging does not grow with N (one buffer, reused) — verify
  this explicitly, not just assert it in a comment.
- Admission (host test): a request needing more staged blocks than the
  admitted maximum produces a clear typed rejection, not an unbounded loop
  or a silent context reduction.
- `cargo xtask-cuda test-gpu` passes on all three devices; host suites,
  both clippy lanes, CUDA-feature clippy, `arch-check`, `spec-check`, fmt
  all pass.
- Support-matrix entries: streamed-page attention's scope note is updated
  from "exactly two blocks" to "N bounded staged blocks, sequential, no
  overlap" — do not write "host-backed streaming supported" or make a
  context-size product claim.
- Deletion and documentation gates: none.
- **Exact condition requiring owner direction or task rejection:** if
  generalizing to N blocks reveals that `attend_two_block`'s existing
  kernel entry point cannot be called N times without a contract change
  that risks task 0041's own accepted N=1 gates, stop and report rather
  than modifying an already-qualified path silently. If admission's bounded
  maximum for N cannot be derived from existing resource-ledger scopes
  without inventing a new capacity concept, stop and report.

## Result, filled after work

- Changed shared owners and consumers; source commit: base `6317e03` (working
  tree intentionally uncommitted). `moxie-executor` now derives and validates
  a bounded N-block launch, returns an incremental `NBlockStream`, and reuses
  its one staged page/partial-output set for every block; the caller supplies
  one page to `stage_next`, folds the returned partial, and only then reads the
  next page. The existing `TwoBlock` path remains intact. The existing
  separately-qualified partial kernel ABI is called N times, unchanged.
  `moxie-memory` extends `HostBackedPlan` with its admitted maximum of three
  staged blocks and a request declaration that rejects larger N values before
  admission. `xtask` folds the raw resident/staged results through the shared
  `Partial::merge` chain and checks the RNE-narrowed cross-path result.
- Commands and result IDs; passed / failed / skipped separately: targeted
  admission tests passed 44/44; `moxie-executor` library tests passed 33/33;
  `cargo test --workspace --locked` passed; host clippy, driver-feature
  clippy and CUDA-feature clippy passed with `-D warnings`; `cargo xtask
  arch-check`, `cargo xtask spec-check` and `cargo fmt --all -- --check`
  passed; `cargo xtask-cuda test-gpu` passed 60/60 on both SM86 UUIDs and
  SM120, with zero failed and zero skipped/unmeasured cases. No required gate
  failed or was skipped.
- Measured effect and uncertainty: task 0041's N=1 case remained passing on
  all three devices. N=3 (one resident plus three staged BF16 pages) transferred
  `12,300 B` total, `4,100 B` per block, and matched the existing single-shot
  path with maximum asserted pairwise BF16 difference `0` under each lane's
  unchanged `attention_error_bound`. The N=3 staging arena was `10,752 B`,
  equal to the one-block staging shape, proving reuse rather than N device
  buffers. The raw streamed-vs-FP64-oracle maxima were
  `3.2830878210488024e-8` on the fixture; the cross-path comparison first
  RNE-narrows the FP64 merge to BF16 and adds no tolerance slack. A fault after
  the first staged block was refused with its in-flight source retained; the
  fault harness also reads pages incrementally and does not retain future
  blocks.
  Inputs are synthetic; no timing, prefetch/overlap, checkpoint or model claim
  follows.
- Deleted/replaced paths: none. The task-0041 single-shot and N=1 streaming
  paths, kernel and ABI remain; no oracle algebra or CUDA kernel was rewritten.
- Remaining blockers and next bounded task: none within this contract once the
  full host/architecture/device gate set is recorded. Prefetch/read-ahead and
  transfer/compute overlap remain M6.3 work; larger context/product claims,
  MLA streaming, COW and prefix reuse remain separate tasks.

Do not fill acceptance with "streaming works" or "long context supported."
Prefetch/overlap (M6.3) and any long-context product claim remain separate,
later work even after this task is accepted.
