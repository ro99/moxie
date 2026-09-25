# Task 0094 — rank prefill/decode placement pairs jointly

Status: **active** (coordinator, 2026-09-25). Builder Codex `luna`; reviewer
Codex `sol`.

## Identity and authority

- Task0094, M6 slice 6, roadmap **M6.4** "joint prefill/decode placement and
  phase transitions". This is the first of two tasks. This one is the
  planner's decision: a prefill placement and a decode placement chosen
  together, with the cost of moving KV between them. The second executes a
  transition on the device, and is opened only if this estimate shows a
  split pair winning on measured costs.
- Why this is generic, not tied to one model: `compare_plans` today gives
  both phases one placement. In
  [plan-comparison.md](../evidence/plan-comparison.md), prompt 512 ranks TP2
  first on total time (prefill 195.9 ms, decode 5.8 ms per token), while one
  3090 prefills in 7.7 ms. Prefill on one card and decode on TP2 could beat
  both, and the planner cannot express that pair today.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement it exactly. On a
  conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`. Preserve the carried
  `.gitignore`, `docs/evidence/specification-version.md` and ADRs 0034 and
  0035. **Stage explicit paths only.** No GPU use: the planner is host-only,
  and the evidence uses the committed costs.
- **Naming rule:** no task numbers in identifiers, labels or strings.
- O6 open: estimates, not measurements, and not a performance claim.

## Facts established before writing (coordinator, 2026-09-25)

- `compare.rs`: `compare_plans` enumerates the candidates (`Single`, `Tp2`,
  `Pipeline`, and `HostExperts`, which is always rejected). It then calls
  `estimate(…)` per candidate. That returns
  `Estimate { device_bytes, prefill_ms, first_decode_ms, total_ms }`, where
  `total_ms = prefill_ms + total decode`. Ranking is by `total_ms`, then by
  kind order.
- KV metering: `stage_meter` builds one `KvRead { bytes_per_row, window }`
  per `Attention` node, where `bytes_per_row = 2 × (K width + V width)`
  (BF16). `kv_storage(kv, context)` applies the window.
  `tensor_parallel_meter` meters each rank's local graph.
- TP KV head ownership (`tensor_parallel.rs` about 1021–1030): for `r`
  ranks, when `kv_heads % r == 0` rank `i` owns heads
  `i·kv_heads/r .. +kv_heads/r`. Otherwise (`r % kv_heads == 0`) it owns the
  one head `i·kv_heads/r`, and the heads are replicated.
- `TopologyCosts::link(from, to)` gives a directed `LinkCost` with
  `latency_us` and `bandwidth_gbps`; host links are `Endpoint::Host`.

## Bounded deliverable

- **Outcome:** `compare_phase_pairs` ranks every ordered pair (prefill
  placement, decode placement) of estimable candidates. The pair's cost
  includes the KV transition, and its fit check covers both placements'
  resident bytes. Same-placement pairs reproduce `compare_plans` exactly.
  `compare-plans` prints the pair ranking, and an evidence file records it
  for the committed costs.
- **Allowed files:** `crates/moxie-plan/src/compare.rs`,
  `crates/moxie-plan/src/tensor_parallel.rs` (only the helper in change 1),
  `crates/moxie-plan/src/lib.rs` (exports), `crates/moxie-plan/tests/plan_comparison.rs`
  (one new test), `xtask/src/compare.rs`,
  `docs/evidence/plan-comparison.md` (a new section), this task's Result.
- **Non-goals:** executing a transition; multi-turn workloads (see the
  ceiling in change 3); exact weight-slice identity across placements;
  changing `compare_plans`' output or ranking.

## Numbered changes

1. **KV head helper (`tensor_parallel.rs`).** Factor the ownership rule above
   into `pub(crate) fn kv_head_range(kv_heads: u64, ranks: u64, rank: u64) ->
   Range<u64>`, and use it at the existing site (behaviour unchanged).
2. **KV attribution (`compare.rs`).** `KvRead` gains `layer: u32` (from the
   `Attention` params) and `heads: Range<u64>` (the global KV heads this
   rank stores). `stage_meter` sets `0..kv_heads` from the node params. In
   `tensor_parallel_meter`, entries metered in a `Stage::Local` for rank `i`
   get `kv_head_range(global_kv_heads, 2, i)`, where the global head count is
   read from the stage graph's attention node for that layer. `Replicated`
   entries keep the full range. `bytes_per_row` stays the rank's local
   value, so bytes per head per row = `bytes_per_row / heads.len()`.
   Existing estimates must not change (the existing tests pin them).
3. **Pairs (`compare.rs`).** Factor `compare_plans`' enumeration and
   estimation into a private function that also returns each estimable
   candidate's rank meters (per device: `weight_bytes` and its `KvRead`s).
   `compare_plans` keeps its exact output. Then add:
   ```rust
   pub struct PhasePair { pub prefill: CandidateKind, pub decode: CandidateKind }
   pub struct PairEstimate {
       pub device_bytes: Vec<(DeviceUuid, u64)>,
       pub prefill_ms: f64,
       pub transition_bytes: u64,
       pub transition_ms: f64,
       pub transition_via_host: bool,
       pub first_decode_ms: f64,
       pub total_ms: f64,
   }
   pub enum PairVerdict { Ranked { rank: usize, estimate: PairEstimate }, Rejected { reason: String } }
   pub fn compare_phase_pairs(graph, oracle, oracles, workload, costs)
       -> Result<Vec<(PhasePair, PairVerdict)>>
   ```
   For every ordered pair (P, D) of estimable candidates, P == D included:
   - **Same placement (P == D):** `device_bytes`, `prefill_ms`,
     `first_decode_ms` and `total_ms` are P's `Estimate` values exactly, and
     the transition is 0 bytes, 0 ms, not via host.
   - **Transition bytes:** for each device `d` and layer `L` in D's meters,
     every head D stores on `d` for `L` that P does not store on `d` for `L`
     moves. Its bytes are `bytes per head per row × visible rows`, where
     visible = `prompt_tokens`, capped by the layer's window. The source is
     the lowest-ordered device (by `DeviceUuid`) that stores that head for
     `L` in P.
   - **Transition time:** group the moved bytes by (source, destination).
     With a direct device link: `latency_us/1000 + bytes/(bandwidth_gbps ×
     1e6)` ms. Without one, go through the host: the source→host plus
     host→destination link terms, and set `transition_via_host`. If a needed
     host link is missing, reject the pair, naming it. Sum the groups
     serially (`ponytail:` serial transfers; overlap them when a transition
     executes).
   - **Fit (P ≠ D):** per device, P's weight bytes + D's weight bytes + P's
     KV at `prompt_tokens` + D's KV at full context. This is conservative:
     it counts a weight slice held in both placements twice, and both KV
     layouts at once, as during the move. If this exceeds `usable_bytes`,
     reject the pair with the same message shape as `estimate`.
   - **Times (P ≠ D):** `prefill_ms = P.prefill_ms`; `first_decode_ms =
     D.first_decode_ms`; `total_ms = P.prefill_ms + transition_ms +
     (D.total_ms − D.prefill_ms)`.
   - Rank by `total_ms`, then (prefill kind, decode kind). Append rejected
     pairs after, in enumeration order, as `compare_plans` does.
   - Doc comment on `compare_phase_pairs`: one turn only. A multi-turn pair
     also moves the new history back to the prefill placement each turn,
     which is not estimated (`ponytail:` ceiling named there).
4. **Test** `split_phase_pairs_price_the_kv_move` in `plan_comparison.rs`,
   with the existing fixture, `costs()` and workload prompt 8 / generate 2:
   - every same-placement pair's estimate equals `compare_plans`' estimate
     for that candidate;
   - `Single(a) → Single(b)` moves exactly 3,072 bytes (6 layers × 2 KV
     heads × 8 head dim × K and V × 2 bytes × 8 rows) through the a↔b link
     (not via host), and `Single(a) → Single(c)` moves the same bytes via
     the host;
   - output is identical when `costs.devices` and `costs.links` are
     reversed.
5. **`xtask/src/compare.rs`:** after the existing table, print a second table
   `phase pairs` with the ten best ranked pairs. Columns: rank, prefill
   candidate, decode candidate, transition MB, via host, prefill ms, first
   decode ms, total s. Then print one line naming the best same-placement
   pair's rank.
6. **Evidence:** in `docs/evidence/plan-comparison.md`, add a section
   "Phase pairs" with the command output for `gemma-dense-fits` at prompt
   512/generate 256 and prompt 32,768/generate 256, using the committed
   costs, each run twice and compared with `cmp`. Add one paragraph saying
   whether a split pair outranks the best same-placement pair, and by how
   much, as an estimate.

## Contract before implementation

- **Semantics:** `compare_plans` output is unchanged byte for byte (the
  existing tests and the committed evidence tables).
- **Determinism:** pair output is independent of the input order of devices
  and links.
- **Failure:** arithmetic overflow is an `Err`, and a pair that cannot move
  or fit is `Rejected` with a reason. Nothing panics on missing links.

## Acceptance

Host gates: `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; `cargo test --workspace --locked`;
`cargo xtask arch-check`; `cargo xtask spec-check`. The evidence commands
are run twice with identical output.

**Coverage check (one mutant, reverted after):** make the transition count
every D head as moved (ignore P's ownership); the new test must fail.

## Result, filled after work

Implemented KV head ownership metering and deterministic `compare_phase_pairs`
ranking with two-placement fit and KV transition costs. `compare_plans` keeps
its existing estimates and output. The CLI appends the phase-pair table and
best same-placement rank; `split_phase_pairs_price_the_kv_move` checks exact
same-placement values, the 3,072-byte direct and host-routed cases, partial KV
reuse, and input-order independence.

The ownership mutant failed the partial-reuse assertion and was reverted.
Host gates passed: fmt, workspace clippy and tests, `arch-check`, and
`spec-check`. Each requested `gemma-dense-fits` command (prompt 512 and
prompt 32768, generate 256) was run twice with `cmp` confirming identical
output. The rank-1 split estimate beats the best same-placement estimate by
0.184 s at prompt 512 and 0.547 s at prompt 32768. These are estimates only;
no GPU was used.
