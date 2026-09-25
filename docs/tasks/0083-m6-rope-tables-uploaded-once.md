# Task 0083 — RoPE tables at fixed workspace offsets, uploaded once per step

Status: **proposed** (coordinator, 2026-09-25); opens when task 0082 is
accepted. Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0083, M6 slice 4, roadmap **M6.2**. Piecewise decode capture (the
  pinned vLLM graph-mode design: attention eager, the kernel chains between
  attention nodes captured) needs every captured segment to be step-invariant.
  A RoPE table is uploaded **in the middle of each layer** into the one shared
  workspace, so it would split every layer's segment in two. This task moves
  all table uploads to the start of the step.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. On a conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings or
  numeric literals.
- O6 is open: no speed claim.

## Facts established before writing (coordinator, 2026-09-25)

- `dense.rs` RoPE arm (task 0077): per RoPE node, look up or compute the table
  keyed by `(rotary_dim, frequency_dim, base.to_bits())` in
  `DenseOperation::rope_tables`, then `copy_from_host_async` it into the plan's
  **one** workspace range at offset 0 and launch with that address. RMS nodes
  also use the workspace at offset 0; stream order keeps them apart.
- Planner (`selected.rs` `lower_dense_mode` loop): `workspace_logical_bytes =
  max` over nodes of `dense_workspace`'s device bytes; the RoPE arm's device
  bytes are its table bytes. Host bytes: sum of distinct tables plus the peak
  of the rest (task 0077). The workspace is one `(StorageRegion::Workspace, 0)`
  range; admission charges `workspace_region_bytes()`.
- Every RoPE node of a step reads the step's one positions value, so all
  tables can be computed at the start of the step.

## Bounded deliverable

- **Outcome:** each distinct RoPE table has a fixed offset in the workspace,
  past the shared part; all of a step's tables are uploaded once, right after
  the step's source uploads; RoPE launches read their table at its offset; no
  host copy happens between the first and last kernel of a step except inside
  the attention and host-expert paths.
- **Allowed files:** `crates/moxie-plan/src/selected.rs`,
  `crates/moxie-executor/src/dense.rs`, and any test whose assertion encodes
  the old workspace byte count (change the number only; name it). This task's
  Result; `docs/evidence/dense-step-timing.md` (append a section).
- **Non-goals:** computing angles on the device; graph capture; pinned host
  memory; any change to the host-byte accounting from task 0077.

## Numbered changes

1. **Planner layout.** In `lower_dense_mode`, RoPE nodes stop contributing
   to the `max`. Collect distinct keys in a `BTreeMap<(u64, u64, u32), u64>`
   (key → table bytes). Shared part = `align_up(max over non-RoPE nodes)`;
   then each key in map order gets offset `shared + sum of align_up(previous
   tables)`; `workspace_logical_bytes = shared + sum of align_up(tables)`
   (checked). Store `rope_table_offsets: BTreeMap<(u64, u64, u32), u64>` in
   `SelectedPlanCandidate` (empty for every other lowering) with a `pub fn
   rope_table_offsets(&self)` accessor. `SelectedNode.workspace_logical_bytes`
   for a RoPE node is unchanged.
2. **Upload once.** In `enqueue_dense`, right after `upload_sources`, for each
   key of the candidate's `rope_table_offsets` in order: compute the table
   (same `angle_table` call, same fallible reservation into `rope_tables`),
   then `copy_from_host_async_at(offset, table, stream)` into the workspace
   range. A key present in the plan but with no RoPE node in the graph is a
   planner bug; refuse with `invalid(…)`.
3. **Launch.** The RoPE arm no longer computes or uploads; it looks up its
   key's offset and passes `workspace_address + offset` as the angle address.
   A missing key refuses with `invalid(…)`.
4. No new test: every dense GPU test runs RoPE with both keys (sliding and
   global) and compares outputs; a wrong offset or a table uploaded for the
   wrong key fails them (task 0077's mutant proved this fixture separates the
   two tables).

## Contract before implementation

- **Semantics:** unchanged; outputs **bit-identical**.
- **Resources:** device workspace grows from `max(shared, largest table)` to
  `shared + sum(distinct tables)`, each aligned; admitted through the
  existing workspace charge. Host bytes unchanged.
- **Lifetimes:** unchanged from task 0077; the tables stay in the operation
  until the lease retires.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`,
`dense_tp2_device`, `cargo xtask-cuda test-gpu` (63/63, SM86 and SM120).

**Coverage check (one mutant, reverted after):** give every key offset
`shared` (the first table's). `dense_gemma_device` must fail.

**Evidence:** append a section to `docs/evidence/dense-step-timing.md`: three
timing runs and the `nsys` `cuMemcpyHtoDAsync_v2` count per step, compared
with the previous section; the `gemma-a` decode plan's workspace bytes before
and after.

**Stop conditions:** outputs not bit-identical; a change conflicts with the
code; a file outside the allowed list is needed.

## Result, filled after work

(pending)
