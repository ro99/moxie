# Handover — paged attention runs; its state owner does not exist yet

Status: **fulfilled by task 0038's closure candidate**. The historical
"remaining" section below describes the boundary at task 0037's handoff; task
0038 now owns the device state, reaches it from selected attention plans, and
has completed the moved allocator and mutation gates. Its task record and
[experiment 0008](../evidence/experiments/0008-paged-attention-mutations.md) are
the current evidence.

## Workspace identity

- Writable root `/home/rodrigo/Developer/moxie`, branch `main`. This work is
  four commits: `92a1ea4` (the oracle preparation), `a15d0af` (the kernel, the
  binding and the 32,768-row gate), `11ae103` (the refusal gates) and `7164175`
  (the first review's findings), plus the second review's corrective commit on
  top. Fetch and recheck `HEAD` before implementing.
- `coordinator.md` is unrelated carried work, still dirty, still outside every
  commit here. Preserve it.
- Legacy `/home/rodrigo/Developer/strata` remains read-only at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. `/models` and `/fast/models` are
  read-only inputs under ADR 0020. Nothing was written under either.
- Device identities used by every gate below: GPU-3032cfa3 and GPU-81fe4578
  (RTX 3090, sm_86), GPU-97fe4889 (RTX 5060 Ti, sm_120). `nvcc` 13.0.88, the
  pinned toolkit.

## Completed facts

[Task 0037](../tasks/0037-m4-paged-device-attention.md) carries the full evidence
table. In short:

**Passed.** `cargo xtask-cuda test-gpu`: 51 cases, 51 passed, 0 failed, 0
skipped, both architectures qualified. `paged_attention` covers head dimensions
64, 100, 128, 200 and 256 — the widest the catalogue declares, so its shape
domain is qualified at the boundary rather than inside it, with 100 and 200 the
widths the 32-lane loop cannot divide — MHA and grouped 4:1 and 3:1, causal and sliding visibility, a declared scale of
1.0 and the conventional one, whole versus chunked prefill compared bit for bit,
one-row decode, page tails and a reversed page table, all against the FP64 oracle
under `attention_error_bound`. `paged_attention_32k` materializes **32,768 actual
BF16 rows** in 33,915,648 B of admitted state, decodes at row 32,767 with whole
and chunked construction bit-identical, appends row 32,768 and decodes again
(max error 3.0e-5 against FP64, identical on all three GPUs), and reports
admitted capacity, committed rows and visible rows separately. Host:
`cargo test --workspace` 106 suites, 0 failed (105 before this task added a
driver-only test binary, which builds and reports zero tests on the host lane); `cargo test -p moxie-executor
--features driver` every suite passed, including six new refusal cases and one
new injected-fault case; both clippy lanes, `arch-check` and `spec-check` clean.

**Failed.** Nothing, at the end. Two things *were* failing at `HEAD` before this
work, both pre-existing, both repaired here and both recorded in [the engineering
log](../engineering-log.md): `test-gpu`'s hand-written `affine_linear` case had
never gained the `group_index` operand when task 0035 versioned that ABI, which
bound the output pointer to the kernel's map parameter and poisoned the process
context for 34 cases; and `grouped_device`'s negative fixture selected its
descriptor by operation and SM alone, which since task 0035 picks a quantized
entry for a BF16 plan.

**Skipped / unmeasured.** The task mutation battery was not written or run, on
purpose (see below). No timing was taken anywhere.

**Reviewed twice.** The first independent review found four
closure-blocking defects — partial descriptor identity at the binding boundary,
infallible allocations on refusal paths, ABI widths discovered after enqueue, and
a declared shape envelope wider than the qualified one — plus a wrong BF16
spacing in the gate and three stale records. All are repaired, and task 0037's
record carries them one by one. The review's first finding is scope rather than a
defect and became the next task below. The second review found three more —
diagnostics that could abort instead of refusing, a CUDA grid-`y` limit still
discovered after the query copy, and a head dimension whose "unaligned" claim
was false — plus two overstated records. All are repaired in the corrective
commit; `DeviceCapability` now carries the device's queried maximum grid
dimensions, and `moxie-kernels` exposes an allocation-free package-identity
predicate so admission need not rebuild a catalogue to check membership.

## Decisions

- [ADR 0032](../decisions/adr/0032-first-paged-attention-kernel-is-written-here.md):
  the first paged attention kernel is written here and **no** FlashAttention or
  FlashInfer source enters the tree. The audit's facts — FA2's sm80–sm90
  declaration and PyTorch-extension host API, its 256-row page constraint, and
  FlashInfer's torch-free core but fixed `paged_kv_t` page-table layout — are in
  the ADR with file and line citations. FlashInfer is named as the candidate to
  revisit when a slice may make a performance claim. Adoption later needs no new
  ADR, only what task 0037 already requires of adopted source.
- `attention_error_bound` now takes the layer's **declared** score scale.
  ADR 0028's clauses and the formula are unchanged; only the input is. The
  counterexample fixture shows the derived value was not a bound at all for a
  layer declaring scale 1.0.
- `moxie-plan` re-exports `Visibility` and `reciprocal_sqrt_scale`, so the
  executor speaks the graph's vocabulary without a new dependency edge. No
  `arch-check` table changed.
- A launch's `history_base` must be a whole number of pages. A partially
  reclaimed page has no stable slot for its rows, which is the state a page table
  exists to avoid.

## Remaining hypotheses and blockers

- **`moxie-state` does not own these pages.** This is the blocker, not a
  refinement. `PagedAttentionRun` admits device pages through `moxie-memory` and
  tracks a committed frontier that is a fact about copied bytes: there is no
  transaction, no branch, no retention rule, no truncation and no lineage behind
  it. Everything in `moxie-state`'s journal — `StateKind::KvPages`,
  `RestoreCapability::Truncate`, the tentative-undo headroom of ADR 0014, task
  0017's per-layer windows — applies to the **host** paged store and to nothing
  on the device.
- **What is uncertain and what was checked.** The host store (`moxie-state::paged`)
  reclaims a windowed layer by **ring overwrite**; the device path reclaims by
  whole pages through a page table. Those are two different physical schemes for
  one logical retention rule, and which one the device store should use is the
  open design question. Evidence already gathered: the kernel masks on absolute
  positions and takes `history_base`, so it does not care which scheme is chosen;
  the launch contract already refuses a base that is not page-aligned; and the
  page table is validated for aliasing, which is what makes whole-page
  reclamation safe. No device gate has yet run with `history_base > 0`.
- **Rejected while building this.** Holding retention or frontier policy in the
  executor was rejected as the second state authority task 0037 forbids by name;
  `PagedAttentionRun` therefore has no retention rule at all and refuses launches
  whose declared history exceeds what it has observed committed. Adopting
  FlashInfer's `paged_kv_t` now was rejected for the same reason — it would settle
  a layout that `moxie-state` has not written yet.
- **The mutation battery is deliberately deferred.** A battery measures whether a
  task's gates can fail. Task 0037's gates are not finished, and substitutions
  written against a binding that does not exist would measure code the next task
  replaces. T0006 and T0028 were both run at closure candidates; T0037 belongs
  there too.
- No performance claim exists or may be made: O6 and O7 are open, the kernel uses
  no tensor cores, and nothing in any lane is timed.

## Next task

**Make paged attention the engine's attention: bind its state to `moxie-state`
and reach it from the graph.**

A review of this work established that the binding is not yet on the execution
path at all — `moxie-plan` refuses every stateful graph, no `OpParams::Attention`
node lowers to it, and `attend` stages the query and output through the host
where document 04 requires device tensor and page-table handles. The state
binding below is necessary and not sufficient; both halves are this task.

- **Owning component:** `moxie-state` for the schema, frontier, retention and
  transaction; `moxie-memory` for admission; `moxie-executor` keeps the
  allocation and the launch and gains no policy. A model crate owns none of it.
- **Required reading before editing:** `crates/moxie-state/src/lib.rs` (journal,
  `StateKind`, `RestoreCapability`, transactions) and `src/paged.rs` (geometry,
  `Retention`, ring reclamation, tentative headroom), ADR 0014, tasks 0013 and
  0017, `crates/moxie-executor/src/paged_attention.rs` in full, and this
  handover's uncertainty above.
- **Second deliverable, equal in weight:** `OpParams::Attention` lowers through
  `moxie-plan` to this binding, the stateful-graph refusal is replaced by an
  admitted plan rather than removed, and query and output are device handles the
  executor already owns. The host-staging entry point stays only as what document
  04 calls an explicit separate reference path, if it stays at all.
- **Deliverable:** one sequence transaction that appends rows to *device* pages
  and publishes them, so that the committed frontier a launch attends over is the
  journal's and not the run's own counter; aborting leaves the committed frontier
  and every prior byte unchanged; truncation to a prefix is exact; and a windowed
  layer reclaims whole pages, raising `history_base`, without moving a retained
  row's slot.
- **Oracle and tests:** `moxie_oracles::attention::KvHistory` already reclaims by
  absolute position over a dense `Vec`, and task 0017 established that the host
  store and that history must agree bit for bit. The device store must join that
  comparison. Add a device gate that attends with `history_base > 0` after a
  windowed reclamation — the one shape no current gate reaches.
- **Resource gates:** the 32,768-row case must still pass, with the admitted
  capacity, the committed frontier and the visible history reported separately;
  admission must cover every byte before allocation; no decode step may grow the
  arena or the ledger.
- **Stop condition:** stop and ask the owner before changing the numerical bound,
  the cache precision, the canonical state ownership, or before any operation that
  writes under a checkpoint root. Stop and report rather than inventing a second
  retention rule if the host store's ring scheme and whole-page device
  reclamation cannot be expressed as one contract — that disagreement is a design
  decision with an ADR, not something to settle inside a task.
