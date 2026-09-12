# Task 0017 — M4 per-layer key/value geometry and window eviction

Status: **proposed**.

## Identity and authority

- Task ID / milestone / owner: 0017 / **M4 state schema, taken early because it
  blocks M1.5's real Gemma text graph** / implementation agent; acceptance
  belongs to the owner. **This task does not close M4 and does not close M1.5.**
- Writable root `/home/rodrigo/Developer/moxie`, branch `main`, base `27dc09f`
  (task 0016 acceptance). Working tree clean at authoring; no initial dirty paths.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Its untracked `.pi/` and
  `tests/p2p/` are preserved and are not source evidence.
- Selected by the owner on 2026-09-12 from the three candidates in
  [the active handover](../handovers/2026-09-12-m1.5-gemma-operation-gap.md).
- Repairs the first of the two blockers that task 0016 recorded against the
  Gemma reduced graph: `Reduction::uniform_kv_geometry` and
  `Reduction::sliding_layers_retain_full_history`. R04/R19/R20/R21.
- Required documents read: AGENTS.md, README, 01–09, the owner-gate register,
  `docs/README.md`, the TASK/ADR/HANDOVER templates, task 0013's contract and
  result, and the task 0016 record and handover. Document 04's
  "Attention contract and large context" and "Sequence transactions" sections
  are the normative text for this task.
- Sources inspected at the frozen legacy commit, all in
  `src/models/gemma4/gemma4_runtime.cpp`: `:174` (`LayerKv` — per-layer
  `capacity_rows`/`start`/`cached_rows`), `:384` (per-layer capacity is
  `global ? maximum_context_tokens : min(maximum_context_tokens,
  sliding_window)`), `:914` (host eviction — erase from the front, advance
  `start`), `:1066` and `:1124` (the device ring — `start` advances when the
  ring is full), `:1385`/`:1407` (`mark_kv`/`rewind_kv`).
- Owner gates: O1–O7 remain open. None blocks this synthetic, host, >=16-bit
  slice. Stop before any checkpoint, quantization, device-attention or product
  performance claim.

## Bounded deliverable

**One concrete outcome:** the paged host KV store admits a *per-layer* key/value
geometry and a *per-layer* retention rule, physically reclaims the history a
sliding layer can no longer see, and refuses — explicitly, never silently — any
read or rollback that would need a reclaimed row. The engine stops refusing a
mixed-geometry graph, and the Gemma reduced graph consumes both.

**Sole owning shared component:** `moxie-state` owns the page schema, the
retention frontier and the reclamation rule. `moxie-memory` remains the sole
admitting authority for its backing bytes. No second store, no second
transaction owner, no per-layer allocator.

**Allowed production files:**

- `crates/moxie-state/src/paged.rs` (schema, layout, addressing, retention).
- `crates/moxie-oracles/src/attention.rs` (`KvHistory` gains an explicit base
  position and an eviction operation; `attend_multi_head` maps history index to
  absolute position through it).
- `crates/moxie-interp/src/paged.rs` (`read_history` reads a layer's retained
  range; per-layer geometry validation).
- `crates/moxie-engine/src/lib.rs` (per-layer admission; removal of the uniform
  attention-geometry refusal).
- `crates/moxie-models/src/gemma4.rs` (per-layer-type key/value geometry in
  `TextConfig` and `compose`; two `Reduction` flags cleared).
- `crates/moxie-cli` diagnostic surface where it prints the reduction list.
- Their tests and manifests, architecture allowlist/negative fixtures,
  `Cargo.lock`, and tracked task/ADR/evidence/handover records.

**Explicit non-goals and forbidden shortcuts.** No device attention, no CUDA,
no kernel. No checkpoint, importer, quantized weight or dequantization
fallback — the artifact stays M3. No vision — M11. No COW fork or prefix
sharing; `fork` keeps refusing. No recurrent, MLA, convolution or index state.
No growth of the admitted pool and no second resource owner. No page sharing
between layers and no per-row allocation. No model-owned execution path. **No
synthetic graph may be described as model support**, and clearing two reduction
flags does not make the Gemma reduced graph Gemma support — the remaining two
flags and the explicit "not implemented" support-matrix row stay.

**Existing consumers and second-shape proof.** All current consumers of
`KvGeometry` must keep passing under `Retention::All`, which is bit-identical to
today's behaviour. The new schema gets two independent consumers: a synthetic
multi-layer graph whose layers disagree in key/value head count, head dimension
*and* retention, and the Gemma reduced graph at two geometries. The dense
`moxie-interp::KvCache` stays full-retention and is the parity oracle; it does
not acquire eviction.

**Temporary paths / expiry.** None introduced. The uniform-geometry refusal in
`moxie-engine` and the two cleared `Reduction` fields' `true` values are
deleted, not flagged off.

## Contract before implementation

### Schema

```
Retention = All | Window { window: usize }      // window > 0
LayerKv   = { kv_heads, key_dim, value_dim, retention }   // all dims > 0
KvGeometry = { layers: Vec<LayerKv>, precision, page_tokens, max_tokens,
               tentative_rows }
```

`KvGeometry` stops being `Copy`; `PagedSequence::geometry()` returns a shared
borrow. The layer vector is allocated once at construction and charged to the
control reserve. `tentative_rows >= 1` is the bounded undo headroom defined
below. Precision stays `>= 16` bits (`is_legal_cache`), unchanged: O4 is open
and this task does not touch it.

### Physical layout

Pages are **per layer**; they are no longer shared across layers, because layers
no longer agree on row width or capacity.

- `element = precision.bits()/8`;
  `key_bytes(L) = kv_heads(L)·key_dim(L)·element`,
  `value_bytes(L) = kv_heads(L)·value_dim(L)·element`.
- `needed(L) = max_tokens` for `All`;
  `min(max_tokens, window(L) + tentative_rows)` for `Window`.
- `pages(L) = ceil(needed(L) / page_tokens)`;
  `capacity(L) = pages(L)·page_tokens`.
- `page_bytes(L) = page_tokens·(key_bytes(L) + value_bytes(L))`.
- Within one page of layer `L`: `page_tokens` K rows, then `page_tokens` V rows,
  each row head-major. This is task 0013's within-page order, now per layer
  instead of layer-major inside a shared page.
- The page table is one `u64` byte offset per `(layer, page)` in layer-major
  order, `sum_L pages(L)` entries, followed by the pages themselves. Every
  product and sum is checked; `isize::MAX` bounds are kept.

### Addressing and the retention frontier

Row `r` of layer `L` occupies page `(r / page_tokens) mod pages(L)`, local index
`r mod page_tokens`. For an `All` layer `capacity(L) >= max_tokens`, so the
modulus never wraps and addressing is byte-identical to task 0013's.

The store keeps `rows` (the executed frontier, as today) and one monotone
`high_water` = the greatest `rows` ever reached. Neither truncation nor rollback
lowers `high_water`.

- `evicted(L) = high_water.saturating_sub(capacity(L))` — the first row index
  whose physical slot may have been overwritten.
- **Readable range of layer `L` = `[ max( rows − window(L), evicted(L) ), rows )`**
  with `rows − window(L)` saturating and `window(All) = ∞`. Capacity beyond the
  window is undo headroom and is **never readable**, so a caller can never
  observe a row that is only accidentally still present.
- `row(L, p)` refuses `p >= rows` as "not yet executed" and `p` below the
  readable start as **reclaimed**, with a distinct, matchable error naming the
  layer, the requested position, the readable start and that re-prefill is
  required. The two refusals are distinguishable; conflating them is the defect
  this bullet exists to prevent.

Because `capacity(L) >= window(L) + tentative_rows`, every position that layer
`L`'s declared `Visibility::SlidingWindow { window }` admits is always readable.
A store whose `window(L)` disagrees with its graph's `Visibility` is refused at
binding, not tolerated.

### Eviction, and why abort stays exact

Eviction is the ring overwrite itself; there is no separate reclamation pass, no
memmove and no page free. The admitted envelope is fixed at construction, as in
task 0013.

An append inside an open transaction is refused, **before any byte is written**,
when it would make `rows − base > tentative_rows`, where `base` is the executed
frontier recorded when that transaction began. The refusal is explicit and names
`tentative_rows`.

That rule is exactly what makes `abort` restorative. After truncation to `base`,
the rows the window can see are `[base − window(L), base)`. Row `r` is intact iff
`r + capacity(L) >= rows_max`. With `rows_max − base <= tentative_rows` and
`capacity(L) >= window(L) + tentative_rows`,
`base − window(L) >= rows_max − capacity(L)`, so every readable row survives.
`abort` therefore never fails for capacity reasons and never needs a snapshot.

`rollback_to(p)` may reach further back than one transaction and **is** refused
when `p.saturating_sub(window(L)) < evicted(L)` for any layer — the target's
window has been reclaimed and the operation needs re-prefill. Document 04:
"if history has been released, recompute or report that the requested operation
needs re-prefill. Do not silently change results."

**The pinned legacy rewind is not carried forward and is the reason this rule is
explicit.** `gemma4_runtime.cpp:1385` saves exactly one evicted row
(`first_key`/`first_value`) and `:1407` reinserts it whenever `start` moved at
all. That is correct only for the single-token `future_entropy` lookahead that
calls it, and silently restores the wrong history for any deeper rewind. The
bounded headroom plus an explicit refusal replaces it. See ADR 0014.

Truncation keeps zeroing the discarded suffix's slots. A zeroed slot can alias a
row below the readable start; that row was already overwritten by the row being
zeroed, so nothing readable is lost.

### Oracle

`moxie-oracles::attention::KvHistory` gains an explicit base absolute position:
`try_with_base(base, capacity)`, `base()`, and `evict_before(position)` which
drops leading entries and raises the base. `append` requires
`position == base + len`; `truncate(prefix)` requires `prefix >= base`;
`attend_multi_head` maps history index `k` to absolute position `base + k` and
requires `base <= position < base + len`. `new()` and `try_with_capacity` keep
base 0, so every existing caller is unchanged.

This makes the oracle an *independent* check on the store's retention rather
than a transcription of it: the oracle evicts by absolute position from its own
dense history, the store evicts by ring overwrite, and the two are compared.

**Predeclared numerical criterion: exact equality.** Evicting any row outside a
layer's declared window must change no output bit. Windowed and full-retention
runs of the same graph, same weights, same tokens must produce byte-identical
logits and byte-identical sampled tokens. No new tolerance is introduced and no
existing tolerance is relaxed. The attention error bound is untouched.

### Resource envelope

Admitted atomically through the existing ledger, as one envelope, before use:
the per-layer padded page pools, the page table, the sampler history and
workspace where present, and the control reserve — which now additionally covers
the layer vector, a bounded `Vec<LayerKv>` of exactly `layers` entries. No
allocation inside `append`, as task 0013 proved and its allocation test must
keep proving. Allocation failure after admission unwinds the charge and releases
through the admitting ledger, unchanged.

Windowed layers must measurably shrink the admitted envelope versus full
retention at the same context; the allocation test reports both.

### Cancellation, failure and rollback

Unchanged in mechanism: one transaction owner, `SequenceState`. A failed or
cancelled append aborts the whole transaction through the same path, restores
counters, lineage and every readable row. Invalid, foreign or resolved
transaction IDs mutate nothing. Capacity and retention refusals never shrink
context, never lower precision and never drop a sampler.

### Partition, hardware, application

No partition semantics change; paged rows stay rank-local host bytes. No CUDA,
no transfer, no lease, no device behaviour. The sampler, its history, the
OpenAI-compatible surface and every declared capability are unchanged. Nothing
here is a performance claim.

## Acceptance

- Full host workspace tests; `cargo fmt --all -- --check`; clippy with
  `-D warnings`; `cargo xtask arch-check`; `cargo xtask spec-check`;
  `git diff --check`. Architecture fixtures extended for any new crate edge;
  shared crates must still reject concrete model ownership.
- **Parity (the central gate):** a windowed run and a full-retention run of the
  same graph produce byte-identical logits and tokens, at more than one window,
  more than one page size, and across page-boundary and ring-wrap positions.
- **Mixed geometry:** a graph whose layers disagree in `kv_heads`, `head_dim`
  and retention executes end to end through the generation service; per-layer
  row widths are checked against independently computed flat bytes.
- **Reclamation refusals:** reading a reclaimed row, and `rollback_to` a prefix
  whose window was reclaimed, each fail with their own distinguishable error;
  a negative control confirms each assertion bites. Reading a not-yet-executed
  row keeps its existing distinct error.
- **Abort exactness:** with the ring full, at every publication boundary and
  every layer, abort restores every readable row, all counters and all lineage.
  An append that would exceed `tentative_rows` is refused before writing, and
  the sequence stays usable.
- **Ring boundaries:** first row of a page, last row of a page, the wrap row,
  and an uneven tail, at both `All` and `Window` layers in the same sequence.
- Whole-versus-chunked prefill parity and later continuation on a windowed
  graph; cancellation then a second generation.
- Allocation accounting: per-layer padded bytes, page table and control reserve
  reconciled with ledger admission and release; zero allocations inside append;
  zero residual charge after close. Report the envelope with and without
  windowing at the same context.
- 32,768 actual stored rows on a windowed layer, showing the envelope bounded by
  the window rather than by the context. **Storage capacity evidence only** —
  explicitly not actual-context attention and not model support.
- Device-feature workspace build/tests and `cargo xtask-cuda test-gpu`
  regression on the two 3090 UUIDs and the 5060 Ti UUID with `PCI_BUS_ID`
  ordering. **No device behaviour is added**, so an unchanged GPU result is the
  expected outcome and is not evidence for this task's mechanism.
- Report failed, skipped and unmeasured separately. Topology, sanitizer,
  checkpoint-quality and paired prefill/decode performance gates are **not
  measured** by this task.
- Support matrix: update the Gemma reduced-graph row to drop the two cleared
  reductions and keep the remaining two; keep the explicit **not implemented**
  row for Gemma 4 checkpoint execution; add a gate ID for per-layer geometry and
  window reclamation. Add ADR 0014 and a bounded handover.
- Do not mark M4 complete, M1.5 complete, or this task accepted without owner
  review.

**Exact condition requiring owner direction or rejection.** Stop and report if
the work would need: a second KV store or a second transaction owner; a
model-owned execution or eviction path; growth of the admitted pool; a snapshot
whose size is not bounded before admission; a weakened numerical gate or a new
tolerance; sub-16-bit state (O4); any checkpoint byte, conversion or download
(O5); or any owner ruling that is still open.

## Result, filled after work

Not started.
