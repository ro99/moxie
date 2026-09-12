# Handover — task 0019 implemented; task 0020 is M2's residency authority

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Contract `b63d931`; implementation is the commit this handover accompanies.
  Base before both was `8dc9e77` (task 0017 acceptance record).
- Read-only legacy reference: `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and
  `tests/p2p/` remain untouched.
- Local checkpoint roots `/models` and `/fast/models` remain read-only inputs.
  **Nothing under either was copied, converted, deleted or modified, and no
  download was started.** Reads were `config.json`, safetensors headers and
  tensor indexes only.

## Completed facts

**M2 item 1's mathematics is implemented and gated.** Routed-expert semantics
are shared operations with FP64 oracles, an interpreter, two independent
consumers and a device-path refusal. See
[task 0019](../tasks/0019-m2-routed-expert-semantics.md) for the contract, the
equations and the filled-in result, and
[the bring-up record](../models/gemma4.md#the-26b-a4b-moe-variant) for the
artifact inventory.

Measured, at the implementation commit:

| Gate | Result |
|---|---|
| `cargo fmt --all -- --check` | passed |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | passed |
| `cargo test --workspace --locked --offline` | **716 passed + 9 doctests, 0 failed** |
| Device-feature workspace tests | **732 passed + 12 doctests, 0 failed** |
| `cargo xtask-cuda test-gpu` | **39 passed, 0 failed, 0 skipped**; sm_86 and sm_120 qualified |
| `cargo xtask arch-check` | every rule and every fixture passes, including the new `models-crate-reaches-memory`. **4 pre-existing failures remain**, all from the untracked review crate under `results/task0014-independent-review-2026-09-11/probes/`; the identical four lines are already in `results/task0015/arch-local.log` |

## Decisions

**Task 0019 was narrowed to item 1's mathematics, and task 0020 owns item 2.**
The M1 closure handover named routing *and* residency for 0019. The contract
narrowed it for an ordering reason rather than a convenience: the residency
authority admits against the union of experts a row batch demands, and that union
is a result of the routing equation. The narrowing, its justification and the
unchanged stop conditions are in the contract's own words.

**The pinned reference for the routed mathematics is released `transformers`
source, read and never executed**, because the frozen legacy tree has no Gemma 4
MoE at all. Three copies were compared and agree. The one difference between them
— 5.15's FP32 router softmax against 5.5.3's input-dtype softmax — is recorded
rather than averaged, and FP32 is pinned.

**One numerical deviation from that reference is declared rather than
discovered.** The reference narrows each expert's weighted contribution to BF16
before accumulating; this interpreter accumulates the `top_k` terms in FP32 and
rounds once at the node boundary, as every other operation in `moxie-oracles`
does. The reduction **order** is pinned either way — ascending expert id for this
family, because `Gemma4TextExperts.forward` iterates an expert-major mask. Which
is closer to the released model is **O2**.

**No owner gate was resolved.** O1–O7 remain open. No numerical threshold,
precision, context target or compatibility surface changed.

**Nothing here is model support.** Both routed graphs are synthetic fixtures over
invented weights at reduced scale. The CLI prints its reduction list first.

## Remaining hypotheses and blockers

- **No residency capability exists.** The designated artifact is 51.6 GB of
  BF16 against a 24 GiB largest device and a 63.9 GiB aggregate; 88.5% of it is
  routed experts. Nothing in this repository makes it run.
- **Device routed kernels and expert partitioning** are M5/M6. The selected BF16
  chain refuses `Route`, `ExpertMlp` and `Combine`; `ExpertMlp` and `Combine`
  fail closed for partitioning, while `Route` is `Replicated` by requirement.
- **Quality** is O2 and needs paired output against the released model.
- **Vision and audio** are M11; the artifact declares both towers.
- **The Laguna checkpoint finished downloading and is now verified complete.**
  `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`: all 15 shards present, each
  satisfying `8 + header + payload_end == file size`, payload ends summing to the
  index's `total_size` of 76,813,095,232 B. **That check is all that was done.**
  No metadata was interpreted and no tensor was read; `configuration_laguna.py`
  and `modeling_laguna.py` are remote code document 03 forbids executing. M2
  item 4 is now unblocked on availability, not on inspection.

## Next task

Task 0020 is **M2 item 2**: connect the real residency authority to the routed
demand the shared operations now produce.

- Owning component: `moxie-memory`, extending the accepted task 0006–0010
  ledger, admission, event-backed leases and device arena. **Exactly one
  production weight-residency owner.** `moxie-storage` supplies bounded reads;
  `moxie-models` gains nothing.
- Required reading before the contract: document 03's "Shared weight-residency
  lifecycle" and its MoE admission paragraph, document 02's memory-authority
  boundary and buffer/asynchronous lifetime contract, M2 items 2 and 5,
  [ADR 0009](../decisions/adr/0009-engram-conditional-memory.md) for Engram as
  one residency class under the same authority, and tasks 0006–0010's results.
- The contract must state, before implementation: the chunk identity
  `(artifact, tensor/expert, logical range, format version)` and its separate
  prepared-layout identity; the state machine including its failure and
  cancellation transitions; how concurrent requests for one chunk coalesce; the
  `acquire(chunk, destination, deadline, use_class)` lease and its readiness
  dependency; the eviction rule; and the admission report that includes the
  **incoming** expert's size before anything is evicted.
- **Stop conditions**, carried forward unchanged: a second weight-residency
  owner; a cache class in a model adapter; simulated placement presented as a
  reservation; a demand path that can deadlock when every evictable entry is
  leased; an unbounded queue; any bulk write, which remains O5's; and any quality
  claim, which remains O2's.
- M2 item 5's tests belong with it: full cache, all entries leased,
  incoming-largest-expert, failure mid-read and mid-upload, repeated
  cancellation, no-next-token cleanup (R08), repeated and missing expert routes,
  and nonuniform row counts. The routing side of the last two already has
  fixtures; the residency side does not.
