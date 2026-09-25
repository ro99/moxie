# Task 0098 — step-indirect paged attention and KV append kernels

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`. Kernel-only; the executor wiring is task 0099 (outline below,
reviewed together).

## Identity and authority

- Task0098, M6 roadmap **M6.2** "stable decode/verification graph capture
  with piecewise fallback". The owner ruled (2026-09-25) to build every roadmap
  feature. Piecewise capture (task 0085) leaves attention eager, because
  three things change every step:
  1. the KV append is a device-to-device copy whose destination (physical
     page, slot) the host computes;
  2. the attention kernel takes `rows`, `first_position`, `history_base` and
     `history_rows` **by value**, so a captured launch would freeze them;
  3. the page table is uploaded each step.

  Item 3 already lands at a fixed per-run device address (`self.table`).
  This task removes items 1 and 2 at the kernel level. Values that change
  each step are read from device memory at a fixed address. This is the
  pinned vLLM graph-mode design (source map).
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**; on a conflict, send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  `.gitignore`, `docs/evidence/specification-version.md` and ADRs 0034 and
  0035. **Stage explicit paths only.** GPUs free;
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in identifiers, labels or strings.

## Facts established before writing (coordinator, 2026-09-25)

- `moxie-kernels/cuda/paged_attention.cu`:
  - `moxie_bf16_paged_attention_v1(query, key_pages, value_pages,
    page_table, output, rows, first_position, history_base, history_rows,
    heads, kv_heads, head_dim, page_tokens, window, scale)`;
  - its grid is `(rows, heads)` with `PAGED_ATTENTION_THREADS` (128);
  - the partial kernel `moxie_bf16_paged_attention_partial_v1` is separate.
- Constants in `moxie-kernels/src/lib.rs` about 90–110. The paged-attention
  catalogue is built at about 466–510, and `paged_attention_declares` about
  150–180 pins the descriptor fields.
- The KV append is `write_rows_from_device` (`paged_attention.rs` about
  2878). It runs one async D2D copy per placement per payload, to
  `physical_page · page_bytes + slot · row_bytes` in the run's key and value
  page ranges.
- `xtask/src/gpu.rs` lists the `test-gpu` cases (about 98–103, 246–261).

## Bounded deliverable

- **Outcome:** two new kernels, qualified bit-identical against the present
  path on SM86 and SM120:
  - an indirect attention kernel that reads the four per-step scalars from
    a device array;
  - a KV append kernel that writes each row to byte offsets read from a
    device array.

  Existing kernels and descriptors are unchanged. No executor code uses the
  new kernels yet.
- **Allowed files:**
  - `crates/moxie-kernels/cuda/paged_attention.cu`;
  - `crates/moxie-kernels/src/lib.rs` (symbol constants only, plus an
    optional catalogue entry, see change 3);
  - `xtask/src/gpu.rs` (one new `test-gpu` case);
  - this task's Result.
- **Non-goals:** the executor, capture, the partial kernel, any change to
  `moxie_bf16_paged_attention_v1`'s behaviour or symbol.

## Numbered changes

1. **One body, two entry points.** Move the body of
   `moxie_bf16_paged_attention_v1` into a `__device__ __forceinline__`
   function taking the same parameters.
   - `moxie_bf16_paged_attention_v1` calls it with its by-value arguments,
     so its observable behaviour is unchanged.
   - Add `extern "C" __global__ void
     moxie_bf16_paged_attention_indirect_v1(query, key_pages, value_pages,
     page_table, output, const unsigned long long* __restrict__ step,
     heads, kv_heads, head_dim, page_tokens, window, scale)`, where
     `step[0..4] = {rows, first_position, history_base, history_rows}`. It
     loads the four values once per block (into registers or shared memory)
     and calls the same body.
   - The grid stays `(max_rows, heads)`. Blocks with `row >= step[0]` return
     at once, so one captured grid serves any `rows ≤ max_rows`.
2. **Indirect append.** Add `extern "C" __global__ void
   moxie_kv_append_indirect_v1(const unsigned char* __restrict__ keys, const
   unsigned char* __restrict__ values, unsigned char* __restrict__
   key_pages, unsigned char* __restrict__ value_pages, const unsigned long
   long* __restrict__ step, const unsigned long long* __restrict__ offsets,
   unsigned long long row_bytes)`:
   - `step[0]` is `rows`;
   - `offsets[2·r]` and `offsets[2·r + 1]` are the key and value
     destination byte offsets of row `r`;
   - the grid is `(max_rows)` with 128 threads, and blocks with `r >=
     step[0]` return;
   - each block copies row `r`'s `row_bytes` of keys and values with 16-byte
     vector loads when the addresses and length are 16-aligned, else byte
     loads;
   - the output is byte-identical to the D2D copies.
3. **Symbols (`lib.rs`).**
   - Add `PAGED_ATTENTION_INDIRECT` and `KV_APPEND_INDIRECT` constants.
   - Do **not** add them to the existing paged-attention catalogue or
     change `paged_attention_declares`; task 0099 adds the descriptor it
     selects.
   - The fatbin SHA changes because the source changed. Update anything
     that pins it in the same crate, if a test fails for that reason only.
4. **Qualification (`xtask/src/gpu.rs`, new case
   `paged_attention_indirect`).** Reuse the setup of the existing
   `paged_attention` case: its geometry, pages and table. For each of
   decode (`rows = 1`), a prefill chunk (`rows = 8`, `first_position > 0`)
   and a sliding window:
   - (a) run `v1` with by-value scalars and `indirect_v1` with the same
     scalars in a device `step` array, and assert the output **bytes** are
     equal;
   - (b) launch `indirect_v1` with a grid of `max_rows = 16` and `step[0] =
     rows`, and assert the same bytes, plus untouched output rows beyond
     `rows` (fill the output with a sentinel beforehand);
   - (c) append `rows` K and V rows with the existing D2D path into one
     page set, and with `kv_append_indirect_v1` into a second page set at
     the same offsets. Assert both page sets are byte-equal, including a
     misaligned fixture (for example `kv_heads = 1`, `head_dim = 70`,
     `row_bytes = 140`). Assert that the fixture really is not 16-byte
     aligned (sol M4).
   - (d) **Against the pre-refactor kernel (sol M3).** Before changing the
     `.cu`, run today's `v1` on (a)'s geometries and record the output
     bytes' SHA-256 in the case as constants, or embed the pre-refactor
     image. After the refactor, both `v1` and `indirect_v1` must reproduce
     those exact bytes. Comparing the two rebuilt entry points only with
     each other is circular.
5. **Coverage check (one mutant, reverted after; run only the new case).**
   In `indirect_v1`, read `first_position` from `step[2]` instead of
   `step[1]`. Case (a) must fail.

## Acceptance

Host gates:
- `cargo fmt --all -- --check`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`;
- `cargo test --workspace --locked`;
- `cargo xtask arch-check`;
- `cargo xtask spec-check`.

GPU: `cargo xtask-cuda test-gpu` (this task changes kernel source, so the
suite is required), all cases passing on SM86 and SM120. No dense suites:
no executor code changes.

## Design review (sol, 2026-09-25, before implementation)

- **0098:** M3 and M4 adopted into change 4.
- **0099's outline:** H1–H4, M1, M2, M5 and L1 are recorded below. They
  bind task 0099's contract when it is written:
  - **H1:** a graph names specific runs' key, value and table addresses. A
    fresh run (new prompt) recaptures, unless the ABI is extended to take
    run pointers indirectly too. Captured runs, their arenas and the module
    live until every graph naming them is destroyed.
  - **H2:** a prepare/apply split through `DeviceKvSequence`. Before
    capture or replay, validate the placements and page view, upload the
    tables and fill the mirror. Inside the graph, only the indirect append
    and attention. After submission, advance the bookkeeping. Never capture
    the present publish/D2D path.
  - **H3:** every indirect offset and extent is checked on the host, with
    the existing placement, range and launch checks, before any enqueue.
    Resources are kept until completion is observed. Bookkeeping updates
    only after tracked submission. Quarantine on uncertain enqueue.
  - **H4:** every replay also uploads the token and position bindings and
    the RoPE angle tables, before the one graph launch.
  - **M1:** the table extent is the run's admitted `geometry.pages`, not
    `visible_tokens / page_tokens`.
  - **M2:** the full-step graph-pool bound is computed and admitted before
    capture.
  - **M5:** the captured identity includes every launch pointer and fixed
    scalar, and every replay is checked against it.
  - **L1:** the dense lease's completion gates the mirror rewrite; no
    separate event.

## Outline of task 0099 (for the design review, not for implementation)

- **Step-state buffer.** Each dense plan whose candidate holds attention
  owns one admitted device buffer. Per attention layer it holds the four
  scalars, plus `2 × max_rows` append offsets, in one contiguous `u64`
  array. Its address is fixed for the plan's life, and it is charged in the
  workspace region. A pinned host mirror (task 0089's `PinnedHostBuffer`,
  admitted as `Host(Pinned)`) is filled on the host per step. It is then
  copied host-to-device in one async copy on the plan stream, **before**
  the graph launch, outside any capture.
- **Page tables.** Each run's `self.table` upload moves before the graph
  launch, outside capture. The table range is sized for the plan's admitted
  `visible_tokens`, so its address never changes.
- **Full-step mode.** A plan with capture enabled and no host joins,
  orders or ownership captures its entire decode step, attention included,
  as one graph: indirect append, then indirect attention, per layer. A
  replay step runs the step-state copy and the table uploads, then one
  graph launch.
- **Fallback.** A plan that cannot use full-step capture keeps task 0085's
  piecewise capture. That covers prefill buckets with host-backed
  streaming, host joins, and ordered reductions until task 0100.
- **Invalidation.** Any change in a captured plan constant (page geometry,
  window, admitted rows, the table address) refuses replay and recaptures.
  State that exceeds the admitted bucket is refused, as today.
- **Ownership.** The step-state buffer and pinned mirror follow task 0096's
  teardown and quarantine rules. The mirror is not rewritten while a
  previous step's copy may still read it: the plan waits on the previous
  step's copy event before rewriting.

## Result, filled after work
