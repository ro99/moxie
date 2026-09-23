# Task 0061 — admit host control metadata and dedupe the test event gate

Status: **ready for review**.

**Amendment, 2026-09-23 (coordinator, coordinator.md §3).** Old premise:
"every host allocation on the paged-attention and device-KV paths" was
bounded paged-run metadata. Evidence, from the round-3 review by sol:
`DeviceKvSequence::begin_for`/`publish_for` call `SequenceState::begin`/
`execute`, which insert transaction-map nodes and grow `PrefixLineage` on
every token (`moxie-state/src/lib.rs:509-513, 717, 950, 983`), and
`commit_prefix` can extend it too. In the 32K gate the root lineage reaches
32K entries and the forked child grows beyond its charged clone. That is
`SequenceState`'s own growth accounting, and bounding it needs a reserved
lineage capacity with a typed refusal, which is a `SequenceState` semantic
change and this task's stop condition. Replacement criterion: task 0061
covers the paged-run table, state views, placements, append, commit,
partial-readback and fork allocations. `SequenceState` transaction and
lineage growth moves to task 0063 (slice 3), following the `PagedSequence`
precedent (`moxie-state/src/paged.rs:538-548`, transaction-map bound at
`:276-278`). The obligation stays open in the M5 ledger until task 0063 is
accepted. Authority: coordinator.

## Identity and authority

- Task0061, first half of M5 plan slice 3 ("Robust rank execution").
  Builder Codex `luna` (max, `/ponytail:ponytail`); reviewer Codex `sol`
  (read-only, `/ponytail:ponytail-review`); coordinator Claude Opus
  `coordinator`. Accepted by the coordinator under the owner's auto-mode
  delegation.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `084258b`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`
  and ADRs 0034 and 0035).
- Requirements:
  - AGENTS.md: "One real memory authority admits all
    persistent/transient/branch/draft resources."
  - Task 0059 round-3 review (sol): three host control-metadata allocations
    live outside every host admission request. The M5 ledger carries them as
    a slice 3 obligation.
  - Task 0057 review: about 75 lines of `cuEventRecord` test-gate setup are
    duplicated.
- O6/O7 are open: no timing.

## Facts established before writing (coordinator, 2026-09-23)

- **Unadmitted host metadata:**
  - `PagedAttentionRun.page_table: Vec<u32>`
    (`moxie-executor/src/paged_attention.rs` field around line 1344, built
    around line 1750 and replaced at 2146 and 2272). It is sized at admit but
    not charged in `resource_request` (around line 4710).
  - `moxie-state` `DeviceKvSequence::page_view_for` (`src/device.rs` around
    line 624) returns a `PageView` whose `table` is a freshly allocated
    `Vec<u32>`. `placements_for` (around line 544) allocates a
    `Vec<PagePlacement>`. Both run while dense K/V staging is live.
- **Duplicated gate:** `crates/moxie-executor/tests/device_arena.rs` (lines
  about 20–60) and `tests/tensor_parallel_device.rs` (lines about 33–105)
  each define the same `BLOCK_NEXT`/`RELEASE`/`TIMED_OUT` state and an
  interposed `extern "C" fn cuEventRecord`. An interposed symbol must exist
  in every test binary that uses it, so the fix is one shared source file
  (for example `tests/support/event_gate.rs`) that each test includes with
  `mod`. It is not a library.
- **Precedent:** task 0059 moved the page-table **upload** buffer into
  admitted workspace and reused it in place. The same approach, preferring
  in-place reuse of admitted buffers over new allocations, applies here.

## Bounded deliverable

- **Outcome:** every host allocation on the paged-attention and device-KV
  paths is either inside an admitted request or eliminated. Prefer, in this
  order:
  1. Write into a buffer that is already admitted and reused, as task 0059
     did.
  2. Borrow an existing structure instead of cloning it.
  3. Add checked bytes to the request.

  The admitted envelope must equal peak host use on these paths.
- **The test gate:** extract it into one shared test-support source file
  used by both test files, with no change in behaviour.
- **Non-goals:**
  - no thread-per-rank or status collective (task 0062);
  - no change to paged-attention numerics or quarantine semantics;
  - no new test file beyond the support module;
  - no timing.

## Acceptance

- Host lanes pass: `fmt`, workspace `clippy`, driver-only `clippy`
  (`-p moxie-executor --features driver`), workspace tests, `arch-check` and
  `spec-check`.
- GPU lanes pass on the listed GPUs:
  - `paged_attention_device` (`driver,paged-attention-binding,paged-attention-test-hooks`);
  - `dense_gemma_device`, `dense_tp2_device` and `tensor_parallel_device`
    (`driver,paged-attention-binding`);
  - `device_arena` (`driver`);
  - the full `cargo xtask test-gpu`, which stays at 63/63.
- **Proof of admission:** extend an existing admission assertion; do not add
  a new test. At peak on the paged path, charged host bytes must be greater
  than or equal to the live metadata bytes. State in Result how that peak is
  measured or bounded.
- **Mutations,** each run and restored:
  - Drop one of the new charges from the request.
  - Re-clone the page view.

  Each must fail an existing test, or the Result must explain why no test
  can observe it.
- **Result:** one test per invariant, plus a short **Review map** listing each
  changed file with its lines added and removed, its purpose and the clause
  it serves.
- **Stop condition:** a lease, ledger or `SequenceState` semantic change.

## Result, filled after work

- **Changed owners; source commit:** `PagedAttentionRun` keeps its admitted
  `page_table` capacity across publications. Both root and branch executor
  append wrappers reject `count > max_rows` with `InvalidRequest { field:
  "rows", .. }`, before calling `DeviceKvSequence`, so the placement charge
  is a true bound and the source rows are returned on refusal. `page_view_for`
  and `placements_for` remain owned by `moxie-state`; their allocations are
  charged by the run's request. Fork-child runs carry a separate logical
  lineage-entry capacity, independent of query rows and physical pages;
  `fork_paged_layer` checks that bound before calling the state authority. The
  child request charges the lineage clone, per-layer vectors and branch-map
  metadata. Commit update/adaptor/reference vectors and partial-stream host
  readback results are also charged by phase. Pair commit's two-element
  `PreparedCommit` vector is now a stack array. The
  `cuEventRecord` test gate remains one shared source. No lease, ledger
  algorithm, or `SequenceState` semantic change was made. No builder commit;
  round 1 began at `09fb1dd`, with coordinator-only commits `ed45551` and
  `c6a08c2`; rounds 2 and 3 began at `c6a08c2`. The carried
  `docs/evidence/specification-version.md` and ADRs 0034/0035 remain preserved.
- **Admission proof:** `page_table_host` and `page_view_host` now reuse
  `page_table_upload_bytes()` (P1); all byte arithmetic is checked. At append
  phase 1, the request charges the upload and persistent run table plus one
  page view (`pages * size_of::<u32>()`) and placements
  (`(ceil(max_rows / page_tokens) + 1) * size_of::<PagePlacement>()`). The
  preflight makes `max_rows` bound the placement vector. At commit phase 4,
  it charges another maximum page view and, per layer,
  `size_of::<(usize, PageView)>() + size_of::<PagedKvWriterAdapter>() +
  size_of::<&mut dyn PagedKvWriter>()`; these are the state updates, adapter,
  and writer-reference vector slots. The two view charges occupy distinct
  phases. Partial-stream readback at phase 3 charges `2*S + W + N*O`, where
  `P = max_rows * heads`, `S = P * size_of::<f32>()`, `W = P * head_dim *
  size_of::<f32>()`, `O = P * size_of::<DevicePartial>() + W`, and
  `N = 1 + max_staged_blocks`. This covers the three raw byte readbacks, the
  in-progress nested result, and every bounded result that may remain live
  while later blocks are read. The direct-admission assertion checks the
  exact maximum of append and commit host peaks. The 32K later-turn gate
  compares the fork request's incremental host peak against the fork formula,
  then checks the actual reservation matches that request. The dense Gemma
  gate checks that append metadata is charged through selected execution and
  prefill workspace is released before decode admission.
- **Mechanical allocation inventory:** `rg` searched the full
  `paged_attention.rs` and `moxie-state/src/device.rs`, the executor
  commit-wrapper range, and the `SequenceState::fork` block in `lib.rs` for
  `Vec::with_capacity`, `with_capacity`, `try_reserve`, `vec![]`, `.collect()`,
  `.to_vec()`, `.clone()`, and `Box::new`. No `Box::new` hits occur. Production
  hits:

  | Classification | Hits and accounting |
  |---|---|
  | **(a) charged** | `paged_attention.rs`: `try_zeroed` (99) call sites charge page-table upload (1844) as `page_table_upload_bytes()`, ordinary attention output readback (3379) as `extents.query`, and partial readback (3222–24) by `2*S + W + N*O`; page-table `try_reserve_exact` (1818) is `pages * size_of::<u32>()`; partial result outer/nested reserves (3245, 3265) are `P * size_of::<DevicePartial>()` and `P * head_dim * size_of::<f32>()`. `moxie-state/src/device.rs`: `placements_for` (603) uses the preflight-bounded `(ceil(max_rows / page_tokens) + 1) * size_of::<PagePlacement>()`; `page_view_for` (679) and commit `updates` (1052) use per-layer `pages * size_of::<u32>()` and the commit formula above. Executor commit `with_capacity` sites (4525, 4537–38, 4568, 4578) reserve one adapter and writer-reference slot per layer; both slots and the state update slot are charged at commit phase 4. Fork admission uses `L = fork_at + 1` lineage entries, checked before state mutation. It charges `L * size_of::<PrefixLineage>()` for `SequenceState::fork`'s `to_vec()` (lib.rs 1344) and `layers * (size_of::<u64>() + size_of::<bool>())` for `retained_floor` and `completed_layers` (device.rs 1252, 1272). It also charges a full BTreeMap internal-node upper bound for each branch-map insertion at device.rs 1266 and lib.rs 1348. For `M(K,V) = align_up(14 * size_of::<usize>() + 11 * (size_of::<K>() + size_of::<V>()), max_align(K,V,usize))`, the fork charge is the checked sum `L * size_of::<PrefixLineage>() + layers * (size_of::<u64>() + size_of::<bool>()) + M(BranchId, Branch) + M(BranchId, DeviceBranchStorage)`. |
  | **(b) eliminated** | The run mapping uses `clear` plus `extend_from_slice` after admission instead of replacing `page_table`; the page-view clone mutation remains absent. Pair commit's growable `prepared` vector is a two-slot stack array. Commit adapters' empty key/value `Vec::new()` values have zero capacity and allocate nothing. |
  | **(c) out of scope** | `paged_attention.rs` admission scaffolding: the 2/3-entry partition-region vector (1660), staging-sized range holder (1703), 3–10 allocation descriptor vector (1721), and descriptor-sized symbol vector (1759) exist only while building/loading one admitted run, before a production step; these remain ledger follow-ups. `read_rows`'s `try_zeroed` (2712) is diagnostic/test readback only, never a production step. `moxie-state/src/device.rs` sequence construction hits (250, 308, 324, 315–26) happen before a device run request exists; SequenceState construction scaffolding in lib.rs (605, 609–19) likewise predates a run request. Stand-alone `DeviceKvSequence` callers using their own writer have no paged-run reservation to attach append/commit/fork charges to; that consumer path remains a ledger follow-up. Executor GPU/admission helpers (5281, 5379, 5441, 5509, 5515, 5533, 5579) and state unit-test fixture/assertion allocations (1413, 1428, 1527, 1534, 1621, 1636, 1651, 1868–69, 2081, 2087) are test-only. |

  The request uses distinct append and commit phases so their vectors are not
  summed as if simultaneously live. Fork metadata is live throughout the
  fork-child run. The coordinator should
  carry the `(c)` production items above as ledger obligations.
- **Commands, GPU UUIDs; passed / failed / skipped:** PASS `cargo fmt --all
  -- --check`; PASS `cargo clippy --workspace --all-targets --locked --
  -D warnings`; PASS driver-only clippy (`cargo clippy -p moxie-executor
  --all-targets --features driver --locked -- -D warnings`); PASS
  `cargo test --workspace --locked`; PASS `cargo xtask arch-check` (79
  rejected fixtures, 21 accepted, 13 rules); PASS `cargo xtask spec-check` (10
  documents unchanged); PASS `git diff --check`; PASS the direct-admission
  unit assertion.

  All five focused GPU lanes passed: `paged_attention_device` 7/7;
  `dense_gemma_device` 2/2 on all three GPUs; `dense_tp2_device` 1/1 on the
  3090 pair; `tensor_parallel_device` 4/4 on the 3090 pair; and `device_arena`
  1/1 on all three GPUs. The final `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo
  xtask-cuda test-gpu` passed 63/63, 0 failed, 0 skipped/unmeasured. Both
  required architectures qualified: SM120 5060 Ti
  (`GPU-97fe4889-4874-a378-198e-955d2e72c4a3`), and both SM86 3090s
  (`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`,
  `GPU-81fe4578-59b2-37c4-421e-287cdac78704`). The first full-matrix attempt
  exposed 15 test-harness appends larger than their run admission; those
  harness admissions were raised to each scenario's existing append maximum.
  The second attempt had three N=3 arena-comparison failures because its
  one-block control still admitted one row; that bound was aligned. The final
  matrix rerun passed. No timing was taken.
- **Mutation results and restoration:** Round 1's placement-charge deletion
  failed the existing direct-admission assertion (`24` charged versus `88`
  required) and was restored. The page-view re-clone mutation passed dense
  Gemma on all three GPUs; as sol confirmed, that residual test-observability
  limit is acceptable. The new oversized-append assertion calls
  `append_paged_layer` with a state that could hold 17 rows but a run admitted
  for `max_rows=1`; it observes the typed `rows` refusal at count 17 before the
  writer runs. The 32K assertion compares the fork request's incremental host
  peak against the calculated lineage/vector/map charge and checks the actual
  reservation against the preview; removing the fork buffer charge makes that
  assertion fail. The shared gate remains behaviour-identical.
- **Review map:**

  | File | Added / removed lines | Purpose and task clause |
  |---|---:|---|
  | `crates/moxie-executor/src/paged_attention.rs` | 410 / 52 | Reuse page-table storage; bound append and fork lineage before state allocation; admit append, fork, commit, and partial-readback peaks; eliminate pair-commit vector. B1, B2, P1. |
  | `crates/moxie-state/src/device.rs` | 25 / 0 | Calculate checked fork vector and device branch-map upper bounds. B2. |
  | `crates/moxie-state/src/lib.rs` | 36 / 0 | Calculate checked SequenceState lineage-clone and branch-map upper bounds. B2. |
  | `crates/moxie-executor/tests/dense_gemma_device.rs` | 32 / 14 | Prove append metadata is admitted in selected execution and reservation release remains exact. Admission proof. |
  | `crates/moxie-executor/tests/device_arena.rs` | 4 / 77 | Include the common event-gate module. Shared test-gate acceptance. |
  | `crates/moxie-executor/tests/tensor_parallel_device.rs` | 3 / 75 | Include the common event-gate module. Shared test-gate acceptance. |
  | `crates/moxie-executor/tests/support/event_gate.rs` | 81 / 0 | Hold the single `cuEventRecord` interposer and unchanged controls. Shared test-gate acceptance. |
  | `crates/moxie-executor/tests/paged_attention_device.rs` | 4 / 6 | Admit existing multi-row append fixtures to their tested append bounds. B1 behavior compatibility. |
  | `xtask/src/gpu.rs` | 83 / 23 | Admit explicit lineage capacity for 32K, all-device COW, windowed and fault forks; assert the 32K request and reservation charge. B2; keep append bounds. |
  | `docs/tasks/0061-m5-admit-host-control-metadata.md` | 112 / 6 | Record round-3 fork admission, mechanical inventory, gate evidence, and review map. Result requirement. |
- **Remaining obligations:** the production `(c)` allocation inventory above
  is reported for the coordinator's ledger. In particular, admission
  scaffolding, state construction, and stand-alone `DeviceKvSequence` callers
  remain outside the paged-run reservation. The
  scanned runtime append, commit, and partial-readback allocations in the
  selected paged executor path are admitted or eliminated. No timing was
  taken.
