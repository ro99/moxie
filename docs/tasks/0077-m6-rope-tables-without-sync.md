# Task 0077 — dense step uploads RoPE tables without synchronizing

Status: **active** (coordinator, 2026-09-24). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0077, M6 slice 1 (sync-free single-GPU dense step), roadmap **M6.1**
  "device-resident layer chains". Follows [task 0076](0076-m6-device-kv-append.md).
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
- O6 is open: nothing here is a speed claim.

## Facts established before writing (coordinator, 2026-09-24; each checked)

- `dense.rs` `enqueue_dense`, `OpParams::Rope { layout: HalfSplit }` arm
  (about 583–640): for **every** RoPE node it computes `angle_table(positions,
  rotary_dim, frequency_dim, base)` on the host, parks it in
  `DenseOperation::pending_host_upload`, enqueues `copy_from_host_async` into
  the plan's one workspace range, then calls **`stream.synchronize()`** so the
  single slot can be reused by the next RoPE node.
- Every RoPE node in a step reads the same positions value (`positions_value`,
  about 379). A table therefore depends only on `(rotary_dim, frequency_dim,
  base)`. Reduced Gemma has two such keys (sliding and global layers; the
  bases are 10,000 and 1,000,000).
- The device workspace range is shared with other nodes (RMS uses it). Stream
  order already serializes "copy table, launch RoPE, later node reuses
  workspace". Only the **host** slot reuse needs the synchronize.
- `DenseOperation::sources` already shows the pattern this task adopts: host
  sources held by the operation until the lease retires after the completion
  event.
- The planner (`moxie-plan/src/selected.rs`, the loop at about 1020–1029)
  takes `host_workspace_bytes = max(host_bytes)` over nodes; the RoPE arm of
  `dense_workspace` returns the table size as host bytes.
- `DensePlanRunRefused::retained_host_upload_bytes` (about 78) exists only
  for `dense_gemma_device.rs`'s
  `angle_copy_sync_failure_retains_its_host_source` (about 1601), which
  injects a failure into `cuStreamSynchronize` through the interposer and
  `FailNextStreamSynchronize` (about 41–78). No other test uses them.

## Bounded deliverable

- **Outcome:** a dense step computes each distinct RoPE table once, keeps it
  in the operation until the step retires, and uploads it without any host
  synchronization. The planner admits exactly those retained bytes.
- **Owner:** `moxie-executor` (`dense.rs`); `moxie-plan` (`selected.rs`).
- **Allowed files:** `crates/moxie-executor/src/dense.rs`,
  `crates/moxie-plan/src/selected.rs`,
  `crates/moxie-executor/tests/dense_gemma_device.rs` (change 4 only), and
  any test file whose assertion encodes the old host-workspace number (change
  the number only; name it in the Result). This task's Result.
- **Non-goals:** the `PagedAttentionRun` `settle` calls; the routed-expert
  route readback; module-load caching; moving RoPE angles onto the device.

## Numbered changes

1. **`dense.rs`, retain tables.** Replace `DenseOperation::pending_host_upload:
   Option<Vec<u8>>` with `rope_tables: Vec<((u64, u64, u32), Vec<u8>)>`, keyed
   by `(rotary_dim, frequency_dim, base.to_bits())`. Doc comment: every angle
   table uploaded by the step stays here until the lease retires after the
   completion event, like `sources`. Initialise with `Vec::new()`.
2. **`dense.rs`, the RoPE arm.** Look up the key in `rope_tables`. If absent:
   compute `angle_table(...)`, `try_reserve(1)` on `rope_tables` (map failure
   to the file's existing capacity error), push. Then enqueue
   `workspace.copy_from_host_async(<the retained table>, stream)` exactly as
   now, keeping its `SAFETY` comment updated ("retained by the operation until
   the lease retires"). **Delete** the `stream.synchronize()?` and the
   `pending_host_upload = None` line and their comment. The launch is
   unchanged.
3. **`dense.rs`, remove the accessor.** Delete
   `DensePlanRunRefused::retained_host_upload_bytes`.
4. **`dense_gemma_device.rs`.** Delete
   `angle_copy_sync_failure_retains_its_host_source`, and the
   `cuStreamSynchronize` interposer, `FAIL_NEXT_STREAM_SYNC`,
   `FailNextStreamSynchronize` and any import left unused. Its invariant
   ("a host source outlives its asynchronous copy") now holds by
   construction: the tables live exactly as long as `sources`.
5. **`selected.rs`, admitted bytes.** In the node loop, a `Rope` node's host
   bytes are **added once per distinct key** `(rotary_dim, frequency_dim,
   base.to_bits())` (a local `BTreeSet`) into a separate `rope_host_bytes`
   with `checked_add`; other nodes keep `max`. After the loop,
   `host_workspace_bytes = host_workspace_bytes.checked_add(rope_host_bytes)`
   (overflow → the file's `invalid("host_workspace", …)`). Sum, not max,
   because the retained tables coexist with every later node's host bytes.

No new test (see change 4).

## Contract before implementation

- **Semantics:** unchanged; each RoPE launch reads the same table bytes as
  before. Outputs must be **bit-identical**.
- **Resources:** host bytes rise from one table to the sum of distinct
  tables, and the planner admits exactly that. Device workspace unchanged.
- **Lifetimes:** tables are owned by `DenseOperation` and released only when
  the lease retires after the completion event, or are withheld with a lost
  lease, as `sources` are.
- **Failure:** a table computation or reservation failure refuses before
  that node's copy is enqueued, through the existing error path.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; `cargo clippy -p moxie-executor
--all-targets --features driver,paged-attention-binding,paged-attention-test-hooks
--locked -- -D warnings`; `cargo test --workspace --locked`; `cargo xtask
arch-check`; `cargo xtask spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`), full suites:
`dense_gemma_device` (now 5 tests), `dense_tp2_device`, `cargo xtask-cuda
test-gpu` (63/63, SM86 and SM120).

**Coverage check (one mutant, reverted after):** make the lookup always reuse
the first table in `rope_tables` when it is non-empty. `dense_gemma_device`
must fail; record which test caught it. If nothing fails, **stop and report**.

**Evidence in the Result:** the `gemma-a` decode plan's `host_workspace_bytes`
(64 after task 0076) and prefill plan's, before and after; the count of
`stream.synchronize()` calls left in `dense.rs` and where; net lines.

**Stop conditions:** any output not bit-identical; a change conflicts with
the code; a file outside the allowed list is needed.

## Result, filled after work

**Complete** (builder Codex `luna`, 2026-09-24; base `97a7281`). The dense
operation now caches each distinct RoPE angle table for the operation lease and
queues each workspace upload on the execution stream without synchronizing.
The planner sums those retained table bytes by `(rotary_dim, frequency_dim,
base.to_bits())`, then adds the peak non-RoPE host workspace. Device workspace
and numerical behavior are unchanged.

- **Workspace:** `gemma-a` prefill (5 rows) `host_workspace_bytes` changed
  **320 → 480 B**: one 320 B sliding table plus one 160 B global table. Decode
  (1 row) changed **64 → 96 B**. Before this task the planner took the larger
  table only (320 B prefill, 64 B decode); the decode baseline is task 0076's
  64 B. Query and key nodes share each layer's table, leaving two distinct keys.
- **Mutation:** forcing reuse of the first cached table for every later key
  failed as required. `reduced_dense_gemma_prefill_and_decode_match_host_on_every_gpu`
  and `pipeline_runs_dense_and_routed_gemma_on_three_gpus` caught a 2 BF16 ULP
  logit difference. The mutation was reverted before the clean gates.
- **Synchronizations:** the RoPE arm no longer calls `stream.synchronize()`.
  Six calls remain in `dense.rs`: one production call in the host expert join,
  after its partial combine and before route/input readbacks, and five in
  `combine_kernel_sums_in_ascending_expert_order`'s device unit test.
- **Host gates:** fmt, workspace clippy, executor clippy with driver and
  `paged-attention-binding,paged-attention-test-hooks` features, workspace
  tests, `cargo xtask arch-check` (79 rejected and 21 accepted fixtures), and
  `cargo xtask spec-check` (10 documents) passed.
- **GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): `dense_gemma_device` passed
  5/5 with bit-identical output on both SM86 GPUs and SM120;
  `dense_tp2_device` passed; and `cargo xtask-cuda test-gpu` passed 63/63 with
  no skips across SM86 and SM120. No speed claim is made.
- **Files and lines:** no tests were added. The obsolete sync-failure test and
  interposer were removed. The three Rust files changed by this task have 46
  insertions and 154 deletions, net **−108 lines** (production sources net
  +14; removed device test and hook −122).
