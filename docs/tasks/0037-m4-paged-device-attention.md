# Task 0037 — M4.1 common paged device attention at actual 32K context

**Status: accepted and closed** (owner, 2026-09-20). Authorized by the owner on
2026-09-19. All six acceptance items are met or explicitly reassigned: the
kernel, its binding, the 32,768-row gate and the admission-failure/
injected-fault sweeps are evidenced on all three GPUs; the state authority and
semantic path this task scoped landed as task 0038's work, which is that
task's to accept, not this one's to wait on. This task's own two remaining
items — the mutation battery and host allocator growth across decode steps —
are moved to task 0038's acceptance 7 and 8 rather than left as this task's
unfinished work; see "Closed, or moved to task 0038" below for the item-by-item
accounting.

## Identity and authority

- Task0037, first bounded M4 task; repository owner reviews and accepts.
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base closure
  record `54b52e3`. `coordinator.md` is unrelated carried work and remains
  outside this task.
- Read-only legacy root `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; checkpoint roots remain read-only.
- Requirement: roadmap M4 item1 and the 32,768-actual-context part of its exit;
  source-map lessons R04, R19 and R21. Read specs02/03/04/06/07/08/09, existing
  `moxie-oracles` attention/mask contracts, graph attention descriptors,
  task0013 paged state and task0017 sliding retention before editing.
- Pinned candidates to audit before adopting code: FlashAttention commit
  `ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820` and FlashInfer commit
  `91bda04c66f7cb851e1ab3b78b9fecea644b9844`. Legacy attention is evidence and
  a source of regressions, not the architecture to copy.
- O1–O5 remain resolved. O6/O7 remain open, so this task proves correctness,
  bounded state and hardware qualification without making a speed claim.

## Bounded deliverable

Implement one common BF16 paged-device attention path for the existing semantic
`Attention` operation. It must append K/V into persistent admitted device state
and execute whole or chunked prefill plus single-row decode over full causal and
sliding visibility, MHA and GQA, partial chunks and page tails. The same shared
API and state owner must run on both SM86 GPUs and SM120. At least one gate must
hold and attend to **32,768 actual rows**, then append and decode the next row.

`moxie-state` remains the transaction/state authority, `moxie-memory` the
admission authority, `moxie-kernels` the device implementation owner, and
`moxie-executor` the binding/launch owner. Model crates may provide semantic
parameters but cannot own pages, CUDA code or execution loops.

This task does not implement MLA, sparse/index attention, host-backed page
streaming, COW forks, prefix reuse, recurrent/convolution state, distributed
attention, a checkpoint-backed block, tuning or a performance claim. Those are
later M4/M5 slices. No fixed short-history fallback may be presented as 32K
support, and no host attention loop may hide behind a model adapter.

Allowed changes are the shared graph/type descriptors only where the current
contract is insufficient, paged state/admission, kernel catalogue/CUDA image,
executor binding, independent oracles/tests, architecture rules, xtask gates and
living records. Any adopted upstream source must have its pinned revision,
license, modified files and supported hardware/shapes recorded before copying.

## Contract before implementation

- Positions are absolute. Append is dense and ordered. A causal query at
  position `p` sees allowed keys through `p`, including the just-appended row;
  sliding visibility follows `moxie_graph::Visibility` exactly. Masked keys are
  excluded rather than represented by an arbitrary finite bias.
- Q/K/V and cache payloads are BF16 for this slice. Head counts, KV head counts,
  head dimension and score scale are explicit semantic parameters; query heads
  map to GQA KV heads by the existing divisible grouping contract. FP16 cache is
  unsupported by this task rather than silently treated as BF16.
- Scores, stable maximum subtraction, exponentials, online-softmax partials and
  weighted-value accumulation use the declared FP32 path. Output narrows only
  at the graph's BF16 boundary. Reordering is allowed only inside the existing
  data-dependent `moxie_oracles::attention_error_bound`, including its
  cancellation and gradual-underflow terms; changing that bound requires owner
  direction before implementation continues.
- The existing bound helper currently derives `1/sqrt(head_dim)` internally
  while the semantic operation accepts an explicit score scale. Before device
  code, parameterize the helper by the declared scale and add cases for scale
  1.0 and reciprocal-square-root scale. This repairs the helper's input without
  changing or widening its error formula.
- Logical pages map to checked physical page identities. Page bytes, page table,
  persistent KV, launch workspace and any staging are admitted before allocation
  by the existing authorities. Capacity and arithmetic overflow refuse with a
  typed error; allocation failure must not abort.
- Enqueued work retains page, activation and workspace leases until a completion
  event is observed. Cancellation before commit leaves the committed frontier
  and every prior byte unchanged, and releases or quarantines transient device
  resources through the existing mechanism. A failed append never publishes a
  partial row.
- Whole and chunked prefill, page layouts and repeated decode are alternative
  executions of the same semantic operation. The independent host oracle must
  build visibility and reference values without reading candidate page tables
  or kernel output as its expected result.

## Acceptance

1. Host tests cover full/sliding visibility, MHA/GQA, explicit scale, page sizes
   and boundaries, empty/partial/final pages, invalid head ratios, position gaps,
   overflow, capacity refusal and exact append/abort/rollback byte preservation.
2. Device tests pass on both SM86 UUIDs and SM120 for head dimensions 64 and 128,
   MHA and at least one nontrivial GQA ratio, full and sliding masks, one-row
   decode, multi-row prefill, uneven chunks and page tails. Each component is
   checked against an independent FP64 equation under the predeclared attention
   error bound and reports max, RMS and high-percentile error.
3. A full gate materializes **32,768 actual BF16 K/V rows**, attends over the
   declared visible history, appends row 32,768 and produces the next decode
   result. It compares whole versus chunked construction and verifies that
   admitted capacity, actual rows and visible rows are reported separately.
4. Admission-failure sweeps cover every allocation before enqueue; cancellation
   and injected launch/synchronization failure preserve the committed frontier,
   release or quarantine every lease, and allow a clean retry. No hidden host
   allocation or transfer grows with decode steps beyond the declared page and
   diagnostic bounds.
5. Architecture checks prove that model crates cannot own the path and that no
   second state/admission authority was added. Fmt, host and driver clippy,
   affected workspace suites, specification checks and a task mutation battery
   pass; failed, skipped and unsupported shapes are reported separately.
6. Update the support matrix with exact shapes, devices, state bytes and visible
   history. Preserve negative upstream/legacy results. Remove any superseded
   device-attention bridge only after all replacement gates pass.

Stop for owner direction before changing the numerical bound, cache precision,
canonical state ownership, importing code with unresolved license/provenance,
starting a bulk checkpoint operation, or claiming performance. A hardware/shape
that cannot pass is a named refusal, not permission to narrow the 32K exit gate.

## Progress — 2026-09-19, host mathematics before device code

Two contract prerequisites, both host-only. **No device code, no kernel, no
state or admission change, and no part of acceptance 2–6 is claimed.**

- `moxie_oracles::attention::attention_error_bound` now takes the layer's
  **declared** score scale instead of deriving `1/sqrt(head_dim)`, as the
  contract requires before device code. The formula is untouched; only its input
  changed, and the magnitude is taken so no caller can drive the bound negative.
  The repair is not cosmetic:
  `the_derived_scale_was_not_a_bound_for_a_layer_that_declares_its_own` holds a
  128-dimension fixture with a declared scale of 1.0 where the measured error is
  1.94e-3, the declared-scale bound is 1.57e-2 and the **derived** value is
  1.38e-3 — smaller than the error it claimed to bound. A Gemma-style layer
  normalizes queries and keys per head and attends with a scale of exactly 1.0,
  so that was the realistic case, not a constructed one.
  `the_bound_tracks_the_declared_scale_on_ordinary_data` covers scale 1.0 and
  reciprocal-square-root scale on well-conditioned data, where the bound stays
  under 1e-5. The FP64 reference in these fixtures now uses the declared f32
  value widened, not an idealized `1/sqrt(head_dim)` no operation declared.
- New `moxie_oracles::online_softmax`: the partial/merge algebra a Flash-style
  paged kernel runs on, which `mask.rs` had listed as owed mathematics. FP64
  throughout, so it states the algebra a kernel may reorder into; the distance
  to an FP32 kernel stays bounded by `attention_error_bound` under ADR 0028 and
  nothing here widens it. Eight fixtures pin: every block width from one row to
  wider than the history reproduces the whole-history softmax (causal and
  sliding); a fully masked block contributes nothing and produces no `NaN`
  (`−∞ − (−∞)`, the merge's sharpest edge, reached whenever a window has moved
  past a page); block order and tree reduction do not change the answer; the
  running maximum is what keeps scores that overflow `exp` finite; a query with
  no visible key anywhere is a typed refusal, not a uniform draw; and every
  score is checked for finiteness rather than the maximum alone, because
  `f64::max` ignores a `NaN` operand and would have let one reach the
  denominator unexplained.

Commands, all on the host lane at `01da0de` plus these changes: `cargo fmt --all
--check` clean; `cargo clippy -p moxie-oracles --all-targets` zero warnings;
`cargo test -p moxie-oracles` 195 passed, 0 failed, 0 skipped; `cargo test
--workspace` all suites passed, 0 failed, 0 skipped (5m26s); `cargo xtask
arch-check` 79 rejected fixtures, 21 accepted, 13 rules; `cargo xtask spec-check`
10 documents unchanged. No GPU lane was run, because nothing device-side changed:
that is unmeasured, not passing.

## Progress — 2026-09-19, the audit, the kernel and the 32,768-row gate

### Pinned-source audit and its decision

[ADR 0032](../decisions/adr/0032-first-paged-attention-kernel-is-written-here.md)
records it. FlashAttention at `ce088ab9` is BSD-3, admits `cc_major >= 8` at
runtime and gencodes `sm_120` on CUDA ≥ 12.8, but its FA2 sources are the
`*_sm80.cu` generation whose own unsupported-arch message declares **sm80–sm90**,
its host API is a PyTorch extension (`torch/python.h`, `at::Tensor`,
`TORCH_CHECK`) rather than a C ABI, and its paged path requires
`page_block_size % 256 == 0`. FlashInfer at `91bda04c` is Apache-2.0, lists SM 8.6
and SM 12.0, and has a genuinely torch-free core, but its `paged_kv_t` fixes a
page-table layout that task 0037 assigns to `moxie-state`, and its scheduling and
workspace ownership live in the Python/JIT layer. **No upstream source was
copied.** The kernel is written here, against the common ABI, and FlashInfer is
named as the candidate to revisit when a slice is authorized to make a
performance claim.

### What now runs

- `moxie_bf16_paged_attention_v1` (`crates/moxie-kernels/cuda/paged_attention.cu`):
  one symbol for whole prefill, a prefill chunk and a single decode row, and one
  path for MHA and GQA. Absolute positions throughout, masked keys excluded
  rather than biased, FP32 scores/maximum/exponentials/partials with one BF16
  output boundary, `expf` rather than `__expf`, and a 128-key online-softmax tile
  deliberately independent of the page width. Compiled SASS-only for sm_86 and
  sm_120, in its own fatbin with its own digest.
- `SemanticKernelOp::PagedAttention` and `KernelOperand::PageIndex` in
  `moxie-types`; two catalogue identities, one per SM, naming one symbol and
  `WorkspaceExpression::Zero` — there is no materialized score matrix, which is
  what lets a 32,768-row history be attended over without a buffer that grows
  with it.
- `moxie_executor::paged_attention`: the checked launch contract (geometry, page
  mapping, absolute-position visibility, the append-before-attend rule, typed
  capacity and overflow refusals) and, behind `driver`, `PagedAttentionRun` —
  admission of the pages, table, query and output through `moxie-memory`, a
  validated page table published once, dense appends that advance the committed
  frontier **only** after a completion event, launch, and refusals that either
  hand the source back or retain it under quarantine. `DeviceRange` gained a
  checked partial write, which is what an append to one page needs.
- `moxie_oracles::online_softmax` and the score-scale repair to
  `attention_error_bound`, from the earlier progress entry, plus
  `attention_error_bounds_at`: the same bound for a whole row, asserted bitwise
  equal to the per-component entry point, because the per-component one
  recomputes `Δs` per lane and costs four billion operations per head at 32K.

### Evidence

`cargo xtask-cuda test-gpu`: **51 cases, 51 passed, 0 failed, 0 skipped**; both
required architectures qualified on real devices — GPU-3032cfa3 and GPU-81fe4578
(RTX 3090, sm_86) and GPU-97fe4889 (RTX 5060 Ti, sm_120).

- `paged_attention`: head dimensions 64 and 128, MHA and 4:1 GQA, full causal and
  sliding (window 20) visibility, a declared scale of exactly 1.0 as well as the
  conventional one, whole and chunked prefill compared **bit for bit**, one-row
  decode, page widths 8/16/32 with tails, and a **reversed** page table. Every
  component checked against `online_softmax` in FP64, cut into 37-key blocks the
  kernel never uses, under `attention_error_bound` at the declared scale plus the
  one declared BF16 boundary. Worst case measured: max 1.953e-3, RMS 7.524e-4,
  p99 1.913e-3 over 1,280 components (sliding, scale 1.0); the grouped 128-wide
  decode is max 4.875e-4 over 1,024.
- `paged_attention_32k`: **32,768 actual BF16 rows materialized**, in 33,915,648 B
  of admitted state. Whole-append and five-chunk construction produce
  **bit-identical** decode results at row 32,767; a 24-row prefill chunk and 24
  single-row decodes agree bit for bit; row 32,768 is appended and decoded. Against
  FP64 at 32,768 visible rows: max 3.037e-5, RMS 5.625e-6, p99 1.519e-5, and after
  the append max 2.953e-5 — **identical on all three GPUs**. Admitted capacity
  (33,024), committed rows (32,769) and visible rows (32,769, or 4,096 under a
  window) are reported separately, and the windowed answer is required to differ
  from the full-history one.
- `cargo test -p moxie-executor --features driver --test paged_attention_device`:
  6 cases, all passed. Admission that cannot fit refuses **before allocating**,
  carries the ledger's own rejection and strands nothing; a descriptor selection
  would never have chosen — narrowed shape bounds, another package's symbol, an
  INT4 cache operand, an indivisible head ratio — is refused at admission; a page
  mapping that names a physical page twice, names one that does not exist, is
  empty, is longer than the admitted pages, or changes under committed rows is
  refused; four kinds of malformed append each hand the rows back, leave the
  frontier at 20 and leave every committed byte identical; a stream from another
  device is refused for both append and attend; and 32 append-and-decode steps
  grow neither the arena nor the ledger while producing a different answer every
  step.
- `cargo test -p moxie-executor --features driver --test driver_faults`: 5 cases,
  all passed, including the new
  `a_paged_attention_failure_keeps_its_query_and_its_frontier`. With the event
  record made to fail after a real copy, the append refuses, **retains** the
  rows, leaves the frontier at four and quarantines the run — which then refuses
  to close or to read its own pages. With `cuLaunchKernel` made to fail, the
  attend refuses, retains the query, leaves the frontier unmoved and quarantines.
- Host: `cargo test --workspace` **106** suites passed, 0 failed, 0 skipped. The
  count was 105 before this task's `paged_attention_device.rs` added a test
  binary; it is `#![cfg(feature = "driver")]`, so on the host lane it builds and
  reports zero tests.
  `cargo test -p moxie-executor --features driver` all suites passed.
  `cargo fmt --all --check`, `cargo clippy` on the host and driver lanes with zero
  warnings, `cargo xtask arch-check` (79 rejected fixtures, 21 accepted, 13 rules)
  and `cargo xtask spec-check` (10 documents) all pass.

### Two pre-existing failures found and repaired

Neither is task 0037's code, both blocked its gates, and both are recorded in
[the engineering log](../engineering-log.md).

1. **The GPU lane was red at HEAD.** `xtask`'s hand-written `affine_linear` case
   never gained the `group_index` operand when task 0035 versioned that ABI to v2,
   so the output pointer was bound to the kernel's map parameter; the kernel read
   group identities out of its own output and the illegal access poisoned the
   process context, failing 34 cases across all three devices. `moxie-executor`'s
   own device tests use the production binding and passed throughout, which is how
   it survived M3 closure. Fixed by passing the absent map as the null operand it
   is.
2. **`grouped_device`'s negative fixture selected the wrong descriptor.** It found
   its GeGLU descriptor by operation and SM alone; task 0035's `affine-expert-*`
   entries share both and sort first in an id-sorted catalogue, so the planner was
   handed a quantized descriptor for a BF16 plan and correctly refused it — before
   the fixture had mutated anything. Fixed by selecting on the projection symbol.

### Review of 2026-09-19 and what it changed

An independent review of the three commits above found four closure-blocking
implementation defects and several record gaps. All are repaired here except
the first, which is scope rather than a defect and is now stated as such.

1. **Not the common semantic path.** The binding is driven by tests and `xtask`
   and nothing else: `moxie-plan` refuses every stateful graph
   (`stateful_resource_plan`), no `OpParams::Attention` node lowers to this, and
   `attend` takes host query bytes and returns host output bytes where document
   04 requires attention to consume device tensor and page-table handles with
   host paths as "explicit separate implementations, not compulsory staging
   interfaces". R04 itself is respected — the history is **not** re-uploaded per
   step — but the query and output cross the bus every launch. This is now named
   in the module header, the support matrix and the next task, which was too
   narrow: planner lowering and device handles belong in it beside the state
   binding.
2. **Partial descriptor identity, repaired.** `descriptor_mismatch` checked
   operation, ABI, operands, output, shape bounds and symbol; admission then
   loaded this build's fatbin unconditionally, so a descriptor declaring a
   different layout, accumulation policy, rounding profile, workspace or image
   would have been executed by code declaring something else. That is exactly
   the defect task 0021's review found one package over. Admission now asks
   `moxie_kernels::paged_attention_declares`, an **allocation-free** predicate
   that compares every field the catalogue sets — rebuilding the catalogue to
   answer would allocate a `Vec`, two `String`s and a `format!` on the path that
   must refuse under memory pressure. `the_package_predicate_and_the_catalogue_agree`
   pins the predicate to the catalogue over ten mutations, and the binding's own
   fixture covers seven more with a passing control. `layout` and `rounding` are
   compared and **cannot be mutated**: `TensorLayout` and `RoundingProfile` each
   have one variant today, which is recorded in both fixtures rather than
   claimed as coverage.
3. **Infallible allocations on refusal paths, repaired.** The refusal joined the
   key and value vectors with `Vec::append` to hand them back — a reallocation on
   the path whose purpose is to report a refusal — and the page-table refusal
   re-encoded its entries with `collect()`. Both are gone: `RefusedSource` hands
   each source back in the allocation it arrived in, and the run holds the two
   append vectors unjoined.
4. **ABI widths discovered after enqueue, repaired.** A sliding window wider than
   the kernel's `u32` passed `check`, and `window()` only failed inside
   `enqueue_attend` — after the query copy had been submitted — turning a
   knowable refusal into an unknown submission, a quarantined run and a withheld
   source. Every width this ABI takes is now refused by `PagedAttentionLaunch`'s
   constructor, and the remaining conversions happen in `abi_scalars` before any
   copy. The launch's fields are private with `new`/`at`/`over` constructors, so
   a checked launch cannot be edited afterwards, and `allows` no longer adds
   positions that could overflow.
5. **A declared envelope wider than the qualified one, closed.** The catalogue
   advertises head dimensions through 256 while only 64 and 128 had run. The
   device gate now covers **256**, the widest declared, **100**, which the
   32-lane loop cannot divide, and **200**, which cannot be divided *and* leaves
   the second accumulator slot partly used — on all three GPUs. A first attempt
   used 96 and called it unaligned; 96 is 3x32, so it tested a non-power-of-two
   width and nothing else. The number is now spelled out in the case comment.
6. **A wrong BF16 spacing in the gate, repaired.** `bf16_ulp` returned `2^-149`
   for every exponent field at or below seven instead of BF16's own `2^-133`
   floor — sixteen binades too small. It could only reject a correct kernel at
   subnormal magnitudes, never accept a wrong one. There were **three** copies of
   the arithmetic: this lane's helper, the affine case's inline version and
   `affine_linear_device.rs`. All three are corrected.
7. **The grid limit, after the second review.** `check` proved the head count
   fits a `u32`; CUDA's grid `y` stops at 65,535. A launch with 65,536 heads
   therefore admitted and would have failed inside `cuLaunchKernel` *after* the
   query copy — the same predictable-after-enqueue shape the ABI repair was
   for. `DeviceCapability` now carries the device's **queried** maximum grid
   dimensions, admission refuses a geometry this device cannot launch before
   charging anything, `attend` re-applies it, and the device fixture proves both
   the refusal and that one head fewer admits.
8. **Fallible diagnostics throughout, in two rounds.** Every refusal in this
   module composed its prose with `format!`, which aborts rather than refusing
   when an allocation fails. Removing `format!` was not enough and the third
   review caught it: `invalid(field, "a literal")` still called `detail.into()`,
   and converting a `&str` to a `String` allocates infallibly, so the shortest
   refusals — no query row, an empty history, a zero-row append — were the ones
   still able to abort. Every refusal in the module now goes through the fallible
   sink whether or not its prose interpolates anything.

   The sweeps were strengthened with it. They now cover the fixed-literal paths
   as well as the interpolated ones, and they assert what the affine sweeps
   assert: that at least one position actually failed an allocation, and that the
   sweep ran **past the end** of the call rather than exhausting its bound. The
   control is measured rather than assumed — with `invalid` restored to
   `detail.into()`, the test binary dies at `memory allocation of 26 bytes
   failed`, `signal: 6, SIGABRT`, which is what a regression for an abort has to
   be able to do.

   Admission itself still inherits the `PlanRequest` abort that task 0029 is the
   bounded fix for; that is pre-existing, outside this module, and is not claimed
   to be repaired here.

Record gaps repaired: ADR 0032 said "no kernel exists yet" and now records the
implementation and, as document 07 requires, **the legacy negative result** —
the frozen `backend_flash_attention` is qualified for exactly SM86 and SM120 yet
appends the packed keys and values to every upload in FP32 and can size a score
scratch at `rows × heads × history`, while `bf16_kv_attention` keeps a persistent
BF16 cache but serves one query row with FP32 host staging. The support matrix
said common paged attention was implemented two rows above saying paged
attention was not; the second row is now the streaming and tensor-core work it
meant. The module header claimed to hold no pages or frontier while doing both;
it now says they are provisional executor mechanics pending the state binding.

### Closed, or moved to task 0038

All six acceptance items are accounted for. 1 is met for the launch contract
and the oracles, with the state-authority piece implemented under task 0038.
2 and 3 are met. 4 is met for the injected-fault sweeps; the
host-allocator-growth measurement moved to task 0038's acceptance 8. 5 is met
for fmt, clippy, architecture, specification and the affected suites; the
mutation battery moved to task 0038's acceptance 7. 6 is met: the support
matrix is updated and this record points at task 0038 below. Nothing here is
left as this task's own unfinished work.

- ~~**The state authority binding.** `moxie-state` does not own these pages yet.
  There is no transaction, branch, retention or truncation behind them, and the
  committed frontier `PagedAttentionRun` reports is a fact about copied bytes
  rather than a journal entry.~~ This was the largest remaining piece of
  acceptance 1 and 4, and the record below is what was open at this task's own
  close. See the next bullet for what closed it.
- **The semantic path is closed by task 0038's closure candidate.**
  `lower_selected` selects `OpParams::Attention`, admission owns separate query
  and output slots, and the executor launches from those device ranges against
  authority-owned state. The host-staged path remains a separate test path.
- **The state binding is closed by task 0038's closure candidate.**
  `DeviceKvSequence` decides placement, retention, frontiers and lifecycle;
  commit publishes mapping changes before finalization. Raw mutation is
  test-feature-only. The 32,768-row gate and the reclaimed-base lifecycle run
  through this binding on both SM86 GPUs and SM120. Task 0038 still awaits
  owner acceptance.
- ~~**A reclaimed base is host-checked only.**~~ **Closed by task 0038**:
  `a_wrapped_ring_answers_exactly_as_an_unwrapped_one` attends after the ring
  has wrapped, at a retained base of 48, and gets byte-identical results to a
  store that never reclaimed. The paragraph below is kept for the record of what
  was open at this task's own close. Every device case *in this task* ran with
  `history_base = 0`. The kernel takes the base and masks on absolute positions,
  and the launch contract refuses a base that is not a whole number of pages, but
  no device gate has actually attended over a history whose first logical row is
  above zero. That is the shape a sliding layer reaches once it reclaims, so it
  belongs in the state-binding task's gates rather than in prose.
- ~~**The task mutation battery** (acceptance 5) is not written and not run.~~
  **Moved to task 0038's acceptance 7, 2026-09-20 (owner):** a battery measures
  whether a task's gates can fail, and this task's gates were never finished on
  their own — writing one against a state binding that did not exist would
  have measured substitutions in code the next task replaces. It runs against
  task 0038's binding instead, as T0006 and T0028's closure candidates did.
  **Completed there:** 11 of 11 substitutions caught, with no survivors,
  instability or skipped anchors; [experiment 0008](../evidence/experiments/0008-paged-attention-mutations.md)
  records the battery.
- ~~**Host-side growth** is checked only as "admission does not grow".~~
  **Moved to task 0038's acceptance 8, 2026-09-20 (owner):** 32 append and
  decode steps leave the arena bytes and the ledger's charged buffers
  unchanged, and each step's answer differs from the last so the appends are
  demonstrably read — but host allocator behaviour across decode steps needs
  the measured-allocator lane rather than that assertion, and task 0038 is
  where it will run. **Completed there:** the 32-step lane measured at most 9
  calls, a 256 B transient peak and 1,000 B maximum live growth, and its
  deliberate 1 MiB-per-step leak mutation is caught.
- MLA, host-backed streaming, COW forks, prefix reuse, FP16 cache, tensor cores
  and any performance claim remain out of scope and unsupported by name.

## Result, filled after work

**Accepted, 2026-09-20 (owner).** Acceptance 2, 3 and the injected-fault half
of 4 are met and evidenced above. Acceptance 1's remaining piece and the
semantic path named in 4 were implemented as task 0038's own work, which that
task's own record and acceptance now carry — this task does not restate or
wait on 0038's acceptance to close, because task 0038's contract always named
these as its deliverable, not a debt this task owed. The two items that were
genuinely this task's own and unfinished — the mutation battery (acceptance 5)
and host allocator growth across decode steps (the remaining half of
acceptance 4) — are moved to task 0038's acceptance 7 and 8, dated 2026-09-20,
rather than closed here by assertion. Nothing in this task's own scope is
open.
