# Task 0017 — M4 per-layer key/value geometry and window eviction

Status: **implemented, awaiting owner review**. Contract and
[ADR 0014](../decisions/adr/0014-bounded-tentative-undo-headroom.md) committed at
`b748536`, before implementation.

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

Implementation follows contract `b748536`. No CUDA, kernel, checkpoint,
importer or device execution path changed. The shared owners are:

- **`moxie-state::paged`** owns the schema. `KvGeometry` is now a per-layer
  `Vec<LayerKv>` plus the shared `precision`, `page_tokens`, `max_tokens` and
  `tentative_rows`; each `LayerKv` carries its own `kv_heads`, `key_dim`,
  `value_dim` and `Retention`. It is no longer `Copy`, and `geometry()` returns
  a borrow. `KvGeometry::uniform` builds the all-`Retention::All` shape every
  previous consumer had, which is byte-identical to the old behaviour.
- Pages belong to **one layer**. Each layer gets its own ring of
  `ceil(min(window + tentative_rows, max_tokens) / page_tokens)` pages; row `r`
  of layer `L` is page `(r / page_tokens) mod pages(L)`. Reclamation *is* the
  ring overwrite -- there is no eviction pass, no memmove and no page free, and
  appending costs the same whether the ring has wrapped or not.
- `high_water`, monotone, records the greatest frontier ever reached, so a
  rollback can tell whether its target's window survived. `retained_range` and
  `row` refuse below it.
- `moxie-types::Error::Reclaimed { layer, position, retained_from }` is a new
  variant, not a reuse of `InvalidRequest`. `kind()` is `"reclaimed"` and it is
  not retryable. `moxie-executor`'s deliberately exhaustive `attribute_error`
  passes it through rather than wrapping it, because its structured fields are
  the attribution and no free-text field could carry them.
- **`moxie-oracles::attention::KvHistory`** gained an absolute base position and
  `evict_before`, and `attend_multi_head` masks on `base + index`. `new()` and
  `try_with_capacity` keep base 0, so every existing caller is unchanged.
- **`moxie-interp::paged`** reads a layer's retained range with its base, and
  validates each attention node's geometry **and visibility** against that
  layer's pages. A store whose retention disagrees with its mask is refused.
- **`moxie-engine`** builds per-layer geometry from the graph's attention nodes,
  rejects a duplicate or absent layer index by position, and sets
  `tentative_rows` from the request's prefill chunk. Its uniform-geometry
  refusal is **deleted**.
- **`moxie-models::gemma4`** carries `local_kv_heads`/`local_head_dim` and
  `global_kv_heads`/`global_head_dim`; `layer_geometry(layer)` is the single
  place the layer-type branch resolves. `compose` uses each layer's own query
  and key/value widths. `Reduction` lost `uniform_kv_geometry` and
  `sliding_layers_retain_full_history`; the CLI's disclosure line lost the two
  labels with them and a test asserts they are gone rather than merely absent.

Both reduced CLI shapes now differ between layer types, in opposite directions
so neither is load-bearing by accident: shape A is 2 heads of 16 sliding and 1
of 32 global; shape B is 2 of 8 sliding and 3 of 12 global.

### The numerical claim, and what proves it

**Reclaiming outside a layer's window changes no output bit.**
`reclaiming_a_sliding_layers_window_changes_no_logit_bit` runs both reduced
shapes against the full-retention dense `KvCache` reference at five
page-size/chunk combinations, comparing raw FP32 bit patterns, and asserts that
reclamation actually happened on every sliding layer and on no global one -- a
windowed layer that never reached capacity would make the test pass while
proving nothing. No tolerance was introduced and none was relaxed.

The oracle is independent rather than a transcription: `KvHistory` reclaims by
absolute position over a dense `Vec`, the store reclaims by ring overwrite, and
`evicting_outside_the_window_changes_no_output_bit` pins the oracle's own half
over four windows and every query position.

**A finding worth carrying forward.** A sliding window is *shift invariant*:
renumber the retained rows and the query by the same amount and the answer is
identical, because the mask only reads `q - k`. So the retention frontier
**cannot** be validated by comparing outputs, and a store one row short still
answers every query correctly -- the interpreter reads a layer's history before
appending the chunk's own rows, so there is exactly one row of slack at the read
boundary. Shortening the frontier by one passed the numerical parity test; by
two it failed. The frontier is therefore pinned by an exact assertion in
`paged_window::check`, not by parity, and both facts are recorded in the code.

That one row of slack is deliberate and documented: `rows - window` retains the
row a query at `rows - 1` would need, so the most recently executed position can
be re-executed without re-prefilling.

### Storage evidence

Not a context claim and not model support: no attention runs in these, and
32,768 *stored rows* is storage capacity, not attention over 32,768 tokens.

| Case | Admitted backing | Page table | Control reserve |
|---|---|---|---|
| Two layers, full retention, 32,768 rows | **398,860 B** | 4,144 B | 267,129 B |
| Same, one layer windowed at 1,024 | **206,360 B** | 2,144 B | 267,129 B |

Zero allocations inside `append` in both, including while reclaiming; zero
retained growth after 10,000 abort/retry cycles; zero allocation and ledger
delta after close. Task 0013 recorded 396,788 B of backing for the same
geometry; the difference is the page table, which now has one entry per
`(layer, page)` rather than one shared entry per page, because a page belongs to
one layer. The pools themselves are unchanged.

### Verification

| Gate | Exact command / result |
|---|---|
| Host workspace | `cargo test --workspace --locked --offline`: **663 unit/integration + 9 doctests passed**, 0 failed, 0 ignored |
| Retention | `cargo test -p moxie-state --test paged_window --locked --offline`: **9 passed** |
| Storage / allocation | `cargo test -p moxie-state --test paged_allocation --test paged_window_allocation --locked --offline -- --nocapture`: **2 passed**, figures above |
| Gemma integration | `cargo test -p moxie-cli --test gemma --locked --offline`: **16 passed** |
| Allocation | `cargo test -p moxie-cli --test allocation --locked --offline -- --nocapture`: **1 passed**; six shapes within their admitted envelopes |
| Host clippy | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings`: passed |
| Format / diff | `cargo fmt --all -- --check`; `git diff --check`: passed |
| Specification | `cargo xtask spec-check`: passed, 10 documents unchanged |
| Architecture | `cargo xtask arch-check` on a clean `git archive` of the tree: **73 rejecting + 21 accepted fixtures, 12 rules**, unchanged from task 0016. No new crate edge was introduced, so no new fixture was needed |
| Device workspace | the host command with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda`: **679 + 12 doctests passed**, 0 failed |
| Device clippy | the clippy command with the same features: passed |
| Real GPU | `cargo xtask-cuda test-gpu`: **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |

| Hardware | UUID |
|---|---|
| RTX 5060 Ti / SM120 | `GPU-97fe4889-4874-a378-198e-955d2e72c4a3` |
| RTX 3090 / SM86 | `GPU-3032cfa3-19df-028f-5ebd-43314911e0b9` |
| RTX 3090 / SM86 | `GPU-81fe4578-59b2-37c4-421e-287cdac78704` |

**The GPU result is unchanged from tasks 0015 and 0016, which is the expected
outcome: this task adds no device behaviour.** It is regression evidence for
already-accepted device work and is not evidence for this task's mechanism.

The device lane earned its keep anyway. `moxie-executor::attribute_error` is a
deliberately exhaustive match on `Error` behind the `driver` feature, so adding
`Reclaimed` broke the device build while the host lane stayed green. That is the
match doing its job; the arm was added rather than a wildcard.

**Unmeasured / not run:** no topology, Compute Sanitizer, checkpoint-quality or
paired prefill/decode benchmark. No CUDA source changed. No device paged
attention, COW fork, host-backed page streaming, recurrent or index state, and
no checkpoint, importer or quantized weight. O1-O7 remain open; none was
resolved or relied on.

### Deliberately failing controls

Each mutation was applied, observed to fail, and reverted.

- Retention frontier at `window - 2`: `reclaiming_a_sliding_layers_window_changes_no_logit_bit`
  fails on the logit bit comparison. At `window - 1` it **passes**, for the
  shift-invariance reason above; that is why the exact frontier is asserted
  separately.
- `without_the_headroom_bound_an_abort_would_lose_readable_rows` and
  `a_rollback_whose_window_survives_is_served_and_one_that_does_not_is_refused`
  each carry their own in-test control: a full-retention sequence rolls back to
  the same prefix successfully, so the refusal is about reclamation rather than
  about the frontier.
- `a_window_at_or_above_the_context_retains_everything_and_costs_the_same`
  is the control for the envelope claim: windowing wider than the context must
  cost exactly what full retention costs.

### Deletion

Removed: `moxie-engine`'s uniform-attention-geometry refusal; `Reduction`'s
`uniform_kv_geometry` and `sliding_layers_retain_full_history` fields and their
two CLI disclosure labels; `moxie-state`'s single shared page pool and its
uniform width fields. Nothing was flagged off and no compatibility shim remains.

The pinned legacy single-row rewind (`gemma4_runtime.cpp:1385`/`:1407`) was
**not** carried forward; ADR 0014 records why, and the bounded headroom plus two
explicit refusals replaces it.

### Remaining blockers and next task

M4 is **not** closed: device paged attention, COW forks and prefix sharing,
host-backed page streaming with online-softmax merge, recurrent/index state
snapshot and replay, and MLA all remain. M1.5 is **not** closed: the Gemma 4
artifact still cannot be executed, because every language-model linear is
compressed-tensors INT8 `pack-quantized` and its importer is M3, and it is
`image-text-to-text` whose vision tower is M11.

The next bounded task is the handover's candidate 2, M3's compressed-tensors
INT8 importer, now that shape is no longer the blocker.
