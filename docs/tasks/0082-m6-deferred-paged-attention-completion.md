# Task 0082 — the dense step's paged-attention work completes with the step

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer Codex `sol`.

## Identity and authority

- Task0082, M6 slice 4 (prefill buckets and decode capture), roadmap
  **M6.2** "stable decode … graph capture". A capture cannot record a host
  synchronization, so the dense step must first stop waiting inside itself.
  This is the ledger's deferred "per-operation `settle`" item; its revisit
  trigger ("when decode capture needs a sync-free step") has fired.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. It changes when asynchronous work
  is observed, so follow it exactly. On any conflict with the code, stop and
  send `DECISION`; do not redesign.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **GPUs are free**; the builder is the only GPU user. Always set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Naming rule:** no task numbers in code identifiers, labels, strings **or
  numeric literals** (a `RankId(81_000 + …)` was caught in tasks 0075 and 0081).
- O6 is open: no speed claim.

## Facts established before writing (coordinator, 2026-09-24)

- `PagedAttentionRun::settle` (`paged_attention.rs` about 4002) records an
  event and **synchronizes** it. In a dense step each attention layer calls it
  up to three times: after the page-table publish (`publish_page_table`, about
  2076; the host source is the run's own `page_table_upload` buffer, held in
  `self.held` as `RefusedSource::PageTableUpload` across the copy), after the
  device-sourced row write (`write_rows_from_device`, about 2620; no host
  source), and after the launch in `attend_into` (about 2807; the query and
  output ranges are the dense plan's, retained by the dense operation lease
  until its completion event).
- Task 0078's profile: about 19 `cuEventSynchronize` per fixture step.
- Every run operation in a dense step is enqueued on the step's one stream,
  and `finish()` synchronizes the dense completion event recorded **after**
  all of them on that stream.
- `Drop for PagedAttentionRun` (about 4252) forgets `held` host sources only
  when `quarantined`.
- Callers that inspect run idleness: `dense_tp_workers.rs` about 1560 (`any(|run|
  !run.is_idle())` at step entry) and about 2105 (`reclaim_drained` in
  cleanup).
- `selected_attention.rs` (standalone attention) calls `attend_into` and
  keeps settled behaviour. **It is unchanged.**

## Bounded deliverable

- **Outcome:** in a dense step, a run's publish, device write and attend
  record their completion instead of waiting for it; the run observes that
  completion the next time anything needs it. A single-GPU dense step then
  has one blocking wait per step, at `finish`; each later deferred publish
  observes the previous step's event, which `finish` already completed, so
  that call does not wait. Those observations sit on the eager attention path,
  outside any future captured segment.
- **Allowed files:** `crates/moxie-executor/src/paged_attention.rs`,
  `crates/moxie-executor/src/dense.rs`,
  `crates/moxie-executor/src/dense_tp_workers.rs`, this task's Result,
  `docs/evidence/dense-step-timing.md` (append a section).
- **Non-goals:** graph capture itself; device-side page-table updates; the
  host-expert join's synchronization; `selected_attention.rs`; any other
  caller's behaviour.

## Numbered changes (all in `paged_attention.rs` unless stated)

1. **State.** Add `pending: Option<Event<'ctx>>` to `PagedAttentionRun`
   (`None` at admission), doc: "The event recorded after this run's last
   deferred operation. Until it is observed, `held` may still be read by the
   device, and `written` counts rows whose copies are enqueued rather than
   observed."
2. **`defer`.** `fn defer(&mut self, launched: Result<()>, stream) ->
   Result<()>`: identical to `settle` except it stores the recorded event in
   `self.pending` instead of synchronizing it. (On any error: quarantine and
   attribute, as `settle` does.)
3. **`observe_pending`.** `pub(crate) fn observe_pending(&mut self) ->
   Result<()>`: if `pending` is `Some`, synchronize it; on error quarantine,
   attribute and return the error (leave `held` held); on success take
   `pending`, and if `held` is `PageTableUpload`, move it back into
   `page_table_upload`.
4. **Every entry point observes first.** At the top of every `pub` /
   `pub(crate)` method of `PagedAttentionRun` that enqueues device work, reads
   device memory or releases resources (including `publish_page_table`,
   `write_rows`, `attend`, `attend_into`, `copy_branch_from`, `read_rows`,
   the streaming entry points, `reclaim_drained` and `close`), call
   `self.observe_pending()?` (convert to that method's refusal type the way its
   existing quarantine refusal is built). **Exception:** `write_rows_from_device`
   and the two deferred entries of change 5 do not observe first.
   **Amended (coordinator, 2026-09-25, on the builder's `DECISION`):** an
   entry point that takes `&self` (for example `read_rows`, an evidence
   readback whose callers only use settled paths) keeps `&self` and instead
   refuses with `invalid("run", "this run has unobserved device work")` while
   `pending` is `Some`.
5. **Deferred mode for the dense step.** Add `pub(crate) fn
   publish_page_table_deferred(…)` and `pub(crate) fn attend_into_deferred(…)`
   with the same signatures as the settled versions. Implement by giving the
   existing bodies a private `defer: bool` parameter that selects `defer` or
   `settle` at their final step; the public names keep `defer = false`.
   `publish_page_table_deferred` still calls `observe_pending` first (it needs
   `page_table_upload`, which the previous step's publish may hold; after
   `finish` that event is already complete). On success the deferred publish
   leaves `held = PageTableUpload` in place (change 3 returns it), and
   updates `page_table`/`page_table_base` as now. `attend_into_deferred`
   returns the ranges immediately; its doc states the caller's obligation:
   "the caller retains `query` and `output` until it observes a completion
   recorded after this call on the same stream".
6. **`write_rows_from_device`** ends with `self.defer(Ok(()), stream)?` instead
   of `settle`. Update the `written` doc comment per change 1.
7. **Writer adapter.** `PagedKvWriterAdapter`, when constructed by
   `from_device`, publishes with `publish_page_table_deferred`; every other
   adapter keeps the settled publish.
8. **`Drop`.** Treat `pending.is_some()` like `quarantined`: forget `held`
   host sources instead of dropping them.
9. **`dense.rs`.** `execute_attention` calls `attend_into_deferred`.
10. **`dense_tp_workers.rs`.** Before the idleness check at about 1560, call
    `observe_pending()` on each run; handle an error exactly as that site
    handles a run that is not idle. `is_idle` also requires `pending.is_none()`.

No new test: every GPU test drives prefill, decode, replay (task 0079 R1),
commit and close through these paths, and a missing observation that returned
a buffer early would show up as corrupted page tables in their output
comparisons.

## R1 repairs (coordinator, 2026-09-25, from sol's review; all were contract gaps)

11. **Order, don't wait (H1).** Add `fn order_after_pending(&self, stream) ->
    Result<()>`: if `pending` is `Some`, `stream.wait_event(pending)` (a device
    wait, no host synchronization). Every deferred entry point
    (`write_rows_from_device`, `publish_page_table_deferred`,
    `attend_into_deferred`) calls it **before enqueuing anything**. `defer`
    then replaces `pending` safely: the new event is ordered after the old one
    on either stream. Doc the invariant on `pending`.
12. **Fork source (H2).** `copy_branch_from` calls `stream.wait_event` on the
    **source** run's `pending` (if `Some`) before its first copy.
13. **Drop (H3, and the older quarantined case, task 0075's class).** In
    `Drop`, if `pending` is `Some`, synchronize it; on success continue as a
    normal drop. If that fails, or the run is quarantined, the run's
    `ResolvedModule` is **never unloaded**: hold it as
    `ManuallyDrop<ResolvedModule<'ctx>>` and drop it explicitly only on the
    normal path. `held` host sources keep their existing forget rule.
14. **Record (M1, the coordinator's wording).** The Outcome above over-claimed
    "one blocking synchronization". Correct statement: one blocking wait per
    step, at `finish`; each later deferred publish observes the previous
    step's event, which `finish` already completed, so that call does not
    wait. Those observations sit on the eager attention path, outside any
    future captured segment. No code change.
15. **Comment (L1).** `written_rows`' doc: rows whose copies are enqueued on
    this run's stream order, not observed.

**Coverage gap, recorded not waived:** no test separates H1–H3 from their
repairs: that needs a stream gate *before* a copy, and the test support only
gates an event record. The ledger carries it with a revisit trigger.

## Contract before implementation

- **Semantics:** unchanged; outputs **bit-identical**.
- **Ordering:** all deferred work of a step is on the step's stream, so the
  dense completion event (and each run's `pending` event) orders after it.
- **Lifetimes:** the only host source a deferred operation holds is the run's
  own `page_table_upload`, returned by `observe_pending`. Device ranges passed
  to `attend_into_deferred` are the dense lease's, retained until its
  completion.
- **Failure:** an error from a deferred event surfaces at the next
  observation and quarantines the run, as a settled failure does now; the
  dense step's own completion error is unchanged.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy;
`cargo test --workspace --locked`; `cargo xtask arch-check`; `cargo xtask
spec-check`.

**GPU gates** (`CUDA_DEVICE_ORDER=PCI_BUS_ID`): full `dense_gemma_device`,
`dense_tp2_device`, `paged_attention_device`, `cargo xtask-cuda test-gpu`
(63/63, SM86 and SM120).

**Evidence:** append a "Deferred paged-attention completion" section to
`docs/evidence/dense-step-timing.md`: three `dense_step_timing` runs and one
`nsys` run giving `cuEventSynchronize` and `cuEventRecord` counts, compared
with task 0079's section. No speed claim.

**Stop conditions:** outputs not bit-identical; a change conflicts with the
code; a method's refusal type cannot carry the observation error without
changing its public signature; a file outside the allowed list is needed.

## Result, filled after work

Implemented all ten changes. Existing output checks passed. Host gates passed:
formatting, workspace clippy, executor driver-feature clippy, workspace tests,
architecture check, and spec check. GPU gates passed: dense Gemma (8 passed,
1 ignored), dense TP2 (1 passed), paged attention (7 passed), and the CUDA
image matrix (63/63 across SM86 and SM120).

Appended the three-run timing and Nsight evidence. On the target RTX 3090,
prefill medians were 1042.890, 1053.008, and 1045.807 µs; decode medians were
999.911, 1007.000, and 993.226 µs. The 111-step profile recorded 777
`cuEventSynchronize` calls and 2109 `cuEventRecord` calls. This is fixture
evidence only; O6 remains open and no performance claim is made.

### R1

Applied changes 11–15: deferred operations and branch-source copies are
ordered after pending events; Drop observes pending work and preserves the
resolved module on quarantine or synchronization failure; the outcome and
`written_rows` comments now match the implemented ordering. Existing dense,
TP2 and paged-attention output checks passed with bit-identical results.

Host gates passed: formatting, workspace clippy, executor driver-feature
clippy, and workspace tests. GPU gates passed: dense Gemma (8 passed,
1 ignored), dense TP2 (1 passed), paged attention (7 passed), and the CUDA
image matrix (63/63 across SM86 and SM120).
