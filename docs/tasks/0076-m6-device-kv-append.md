# Task 0076 — dense step appends K/V on the device

Status: **active** (coordinator, 2026-09-24). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0076, M6 slice 1 (sync-free single-GPU dense step), roadmap **M6.1**:
  "device-resident layer chains … and output-minimizing transfers".
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. If one conflicts with the code, stop and send a `DECISION` report;
  do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free.** Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The builder
  is the only agent using the GPUs.
- **Naming rule** (the owner's): no task numbers in code identifiers, labels,
  fixture ids or strings.
- Owner gates: O6 (performance) is open, so nothing here is a speed claim.

## Facts established before writing (coordinator, 2026-09-24; each checked)

- Every attention layer of a dense device step copies its K and V
  **device → host → device**. `dense.rs` (about 1183–1215) allocates two host
  `Vec`s, reads the K and V value ranges with the blocking
  `DeviceRange::copy_to_host`, then passes them to
  `paged_attention::device::append_paged_layer`, whose writer uploads them
  into the pages with `copy_from_host_async_at` (`enqueue_writes`, about
  2672–2707).
- The planner admits that staging: `moxie-plan/src/selected.rs`
  `dense_workspace`, `SemanticKernelOp::PagedAttention` arm (about
  1396–1409), returns `rows * kv_heads * head_dim * 4` host bytes.
- A device-to-device copy between two ranges on one device already exists:
  `DeviceRange::copy_from_device_async_at(within, source, source_within,
  bytes, stream)` (`arena.rs` about 486), used by the branch copy
  (`paged_attention.rs` `copy_branch_from`, about 2359).
- `PagedKvWriterAdapter` (`paged_attention.rs` about 4295) already carries
  one alternative source (`source: Option<&PagedAttentionRun>`, for
  branches).
- Consumers of the dense step, all through `enqueue_dense`: the single-GPU
  step (`execute_dense`), the TP2 workers (`dense_tp_workers.rs`), the solo
  rank worker and the pipeline (`execute_dense_stage`).
- `selected_attention.rs` also calls `append_paged_layer`, but its K/V are
  genuine host inputs. **It is unchanged.**

## Bounded deliverable

- **Outcome:** a dense device step writes each layer's K/V rows from the
  plan's arena ranges straight into the layer's pages, on the step stream,
  with no host copy and no admitted host staging.
- **Owner:** `moxie-executor` (`paged_attention.rs`, `dense.rs`);
  `moxie-plan` (`selected.rs`) for the admitted bytes.
- **Allowed files:** `crates/moxie-executor/src/paged_attention.rs`,
  `crates/moxie-executor/src/dense.rs`, `crates/moxie-plan/src/selected.rs`,
  and any test file whose assertion encodes the old host-workspace number
  (change the expected number only; say which in the Result). This task
  record's Result.
- **Non-goals** (later tasks in slice 1; do not start them):
  - removing the per-write, per-page-table-publish and per-attend `settle`
    (event synchronization) in `PagedAttentionRun`. **Keep `settle` after
    the device write exactly as the host write does.**
  - the RoPE angle-table upload and its `stream.synchronize()`;
  - the routed-expert route readback;
  - any module-load caching.

## Numbered changes

1. **`paged_attention.rs`, share the placement checks.** Move the checks in
   `PagedAttentionRun::write_rows` that run before `self.held = …` (quarantine,
   `same_device`, empty placements, `row_bytes`, the per-placement loop, the
   `want` computation) into one private method on `PagedAttentionRun` that
   returns `(row_bytes, rows, position_end)` as `Result`. `write_rows` calls
   it and keeps its give-back behaviour: on `Err(error)` it returns
   `give_back(error, keys, values)`. The key/value length check stays in
   `write_rows` (it is host-specific). No message changes.
2. **`paged_attention.rs`, add `write_rows_from_device`.**
   `pub(crate) fn write_rows_from_device(&mut self, stream: &Stream<'ctx>,
   placements: &[PagePlacement], keys: &DeviceRange<'ctx>, values:
   &DeviceRange<'ctx>) -> Result<()>`:
   - run the shared checks from change 1;
   - refuse (without touching the device) when `keys.bytes() < rows *
     row_bytes` or `values.bytes() < rows * row_bytes`;
   - for each placement, exactly as `enqueue_writes` computes `within` and
     the source offset `done * row_bytes`, enqueue
     `copy_from_device_async_at(within, source, done * row_bytes, placement.rows
     * row_bytes, stream)` into `self.keys` and `self.values`;
   - on a copy error: `self.quarantined = true`, return `self.attribute(error)`;
   - then `self.settle(Ok(()), stream)?` and, last,
     `self.written = self.written.max(position_end)`, as `write_rows` does.
   - `self.held` is not used: the source ranges belong to the caller's
     operation lease, which outlives the settle.
3. **`paged_attention.rs`, device rows in the writer adapter.** Add a field
   `device_rows: Option<(&'run DeviceRange<'ctx>, &'run DeviceRange<'ctx>)>`
   to `PagedKvWriterAdapter` (`None` in `new` and `for_branch`) and a
   constructor `from_device(layer, run, stream, keys, values)`. In
   `write_layer`, after `self.publish(layer, view)?`: when `device_rows` is
   `Some`, call `self.run.write_rows_from_device(self.stream, placements,
   keys, values)` and return its result; otherwise the existing host path.
4. **`paged_attention.rs`, the entry point.** Next to `append_paged_layer`,
   add `pub(crate) fn append_paged_layer_from_device(state: &mut
   DeviceKvSequence, txn, layer: usize, count: u64, run: &mut
   PagedAttentionRun<'ctx>, stream: &Stream<'ctx>, keys: &DeviceRange<'ctx>,
   values: &DeviceRange<'ctx>) -> Result<()>`: `run.check_append_rows(count)?`,
   then `state.append_layer(txn, layer, count, &mut
   PagedKvWriterAdapter::from_device(layer, run, stream, keys, values))`.
   Same `cfg` as `append_paged_layer`.
5. **`dense.rs`.** In the attention function (about 1177–1216), delete the
   `payload_bytes` computation, both host `Vec`s and both `copy_to_host`
   calls. Keep the `layer_count` check. Replace the `append_paged_layer`
   call with `append_paged_layer_from_device(state, transaction, layer as
   usize, rows, run, stream, key_range, value_range)?`, where `key_range`
   and `value_range` are the existing `address_range(lease.resource(),
   node.inputs[1|2])?` borrows. Drop the `PagedKvRows` import if unused.
6. **`selected.rs`.** In `dense_workspace`, the `PagedAttention` arm returns
   `(WorkspaceExpression::Zero, 0, 0)`. Remove the now-unused `kv_heads` /
   `head_dim` destructure. If the arm then matches another arm's result,
   merge them only if that is a one-line change.

**Tests:** no new test. The invariant ("appended rows are the rows the step
computed, at the placements the authority gave") is already failed by
`dense_gemma_device`'s decode-after-prefill and continuation cases, which read
history back through the pages.

## Contract before implementation

- **Semantics:** unchanged. Bytes written to the pages are identical to the
  host path's; outputs must be **bit-identical** to the base commit.
- **Resources:** each dense plan's admitted host workspace falls by the
  attention staging term (the Rope angle table remains). No new device
  allocation.
- **Lifetimes:** the source ranges are the dense operation's arena ranges,
  retained by its lease through the step's completion event; the device
  write is additionally settled before `write_rows_from_device` returns.
- **Failure:** a refused check leaves the device untouched and the run
  usable; a failed enqueue or settle quarantines the run, as the host write
  does. The dense step then fails through its existing error path; the
  transaction stays the caller's to abort.

## Acceptance

**Host gates:**
- `cargo fmt --all -- --check`.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`.
- `cargo clippy -p moxie-executor --all-targets --features
  driver,paged-attention-binding,paged-attention-test-hooks --locked -- -D
  warnings`.
- `cargo test --workspace --locked`.
- `cargo xtask arch-check` and `cargo xtask spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), full suites:
- `dense_gemma_device` (6/6);
- `dense_tp2_device`;
- `cargo xtask-cuda test-gpu` (63/63, SM86 and SM120).

**Coverage check (one mutant, reverted after):** in `write_rows_from_device`
pass source offset `0` instead of `done * row_bytes`. Run `dense_gemma_device`;
it must fail. Record which test caught it. If every test passes, **stop and
report**: the fixture never writes more than one placement, and the coverage
gap is the coordinator's to resolve.

**Evidence in the Result:** the decode plan's `host_workspace_bytes` before
and after, for one `dense_gemma_device` fixture; net lines changed.

**Stop conditions:**
- any output is not bit-identical to the base commit;
- a change in this list conflicts with the code (send `DECISION`);
- the change needs a file outside the allowed list.

## Result, filled after work

**Complete** (builder Codex `luna`, 2026-09-24; base `6b8239c`). Dense K/V rows now copy from the selected operation's arena ranges directly into the authority's page placements. The device write settles before advancing `written`; dense attention no longer allocates or fills host K/V vectors, and the planner reports no attention K/V host workspace. Host-backed `append_paged_layer` remains unchanged.

- **Workspace:** `reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu`, `gemma-a` decode candidate: `host_workspace_bytes` **128 → 64 bytes**. The old maximum was the two BF16 K/V row staging term; after removal, the remaining maximum is the local-layer RoPE angle table. This does not claim a speedup.
- **Mutation:** changing the device source offset to `0` failed as required. Both `reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu` and `pipeline_runs_dense_and_routed_gemma_on_three_gpus` caught the wrong rows by their logit comparisons. The mutation was reverted to `done * row_bytes` before the clean gates.
- **Host gates:** `cargo fmt --all -- --check`, both listed clippy commands, `cargo test --workspace --locked`, `cargo xtask arch-check` (79 rejected and 21 accepted fixtures), and `cargo xtask spec-check` (10 documents) passed.
- **GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): `dense_gemma_device` passed 6/6, `dense_tp2_device` passed, and `cargo xtask-cuda test-gpu` passed 63/63 with no skips across SM86 and SM120. Dense outputs remained bit-identical.
- **Files and lines:** no tests added or changed. Production Rust diff: 170 insertions, 153 deletions, net **+17 lines** across the three allowed source files.
- **Reference:** persistent device KV follows R04 and Strata's BF16 KV request contract (`include/strata/device/cuda_backend.hpp:304-307`); no performance result is claimed.
