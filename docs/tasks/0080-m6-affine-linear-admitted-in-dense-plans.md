# Task 0080 — dense plans admit quantized (affine) linear weights

Status: **proposed** (coordinator, 2026-09-24). Builder Codex `luna`;
reviewer Codex `sol`.

## Identity and authority

- Task0080, M6 slice 2, roadmap **M6.1** "fused dequantization … where
  supported". First half of a two-task pair: this task is the planner,
  catalogue and image; the device step that launches it is the next task.
- Builder: Codex `luna` (`gpt-6-luna`, max, `/ponytail:ponytail`).
  Reviewer: Codex `sol` (read-only, `/ponytail:ponytail-review`).
  Coordinator: Claude Opus `coordinator`.
- The **design below is the coordinator's**. Implement the numbered changes
  exactly. On a conflict with the code, stop and send `DECISION`.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base: the commit that
  opens this task. Preserve the carried `.gitignore`,
  `docs/evidence/specification-version.md` and ADRs 0034 and 0035. **Stage
  explicit paths only.**
- **Naming rule:** no task numbers in code identifiers, labels or strings.
- Owner rulings that bind: Moxie never quantizes (ADR 0017), so fixtures
  write synthetic integer codes directly and nothing here derives codes from
  a float weight; the two-clause numerical gate (ADR 0028) is unchanged.

## Why, and the design decision (coordinator, 2026-09-24)

- Every v1 checkpoint except `gemma-4-26B-A4B` is quantized (the first,
  `gemma-4-31B-it-AWQ-8bit`, is INT8, group 32, symmetric). The M6 exit's
  checkpoint benchmarks need the dense graph step to run such weights. Today
  it cannot: `moxie-plan/src/lib.rs` `tensor_bytes` refuses integer weights
  ("integer weight bytes require affine scale/zero-point metadata absent from
  the graph"), and the dense catalogue has BF16 linears only. M3's
  `moxie_affine_linear_v2` exists, but only as a standalone executor.
- **The format is a planner input, not a graph property.** Document 02 keeps
  "weight storage" separate from the semantic roles. The same Gemma graph must
  bind a BF16 checkpoint and an AWQ checkpoint. So the model graph keeps
  `ValueRole::Weight(Bf16)` (the logical, real-valued weight), and the lowering
  takes a sidecar map `ValueId → WeightFormat`, exactly as it already takes
  `LinearReductionOrder`, `CombineReductionOrder` and `ExpertOwnership` maps.
  The host reference binds the reconstructed weight, rounded to BF16, which is
  the kernel's own oracle (`affine_linear_device.rs` `oracle`, about 236).
- The descriptor document 03 defines (width, grouping, scale dtype, zero-point
  mode, group map) is already `moxie-plan`'s `ExpertWeightFormat::Affine`. It is
  renamed to `WeightFormat` and reused, not duplicated.

## Facts established before writing (each checked)

- `ExpertWeightFormat` (`moxie-plan/src/expert.rs` about 671) has 43 uses in
  `expert.rs`, `moxie-executor/src/{grouped_device.rs,grouped.rs,residency.rs}`
  and `moxie-executor/tests/grouped_device.rs`. `payload_bytes` packs codes
  then scale+zero metadata contiguously; `bytes` appends a `u32` map per input
  when `mapped`.
- `lower_dense_mode` (`selected.rs` about 903) is private, with three callers
  in the file (about 549, 622, 783). The node loop builds `roles` with
  `dense_operands` and matches descriptors on operation, roles, output,
  accumulation, rounding, layout, SM and shape bounds. Weight values are placed
  in the loop at about 1075–1092 with `logical_bytes: weight.required_bytes`
  from the base plan.
- `SelectedReservedPlan::admit` (`moxie-executor/src/chain.rs` about 171)
  charges `candidate.weight_region_bytes()` etc., which come from those
  planned values, not from the base plan's totals.
- `dense_graph.cu` is a list of `#include`s of already-qualified sources;
  `affine_linear.cu` is built separately as `affine_linear.fatbin`
  (`moxie-kernels/build.rs` about 126). `dense_graph_catalogue` (`lib.rs`
  about 513) has one BF16 `Linear` descriptor per SM. `affine_linear_catalogue`
  (about 411) has Int4/Int8 `Linear` descriptors: shape bounds `max_rows
  65_536, max_input 16_384, max_output 65_536`, `WorkspaceExpression::Zero`,
  symbol `AFFINE_LINEAR`.

## Bounded deliverable

- **Outcome:** `lower_selected_with_formats` lowers a complete dense graph in
  which chosen `Linear` weights are bound as affine INT4/INT8; each such node
  selects an affine descriptor from the dense catalogue, each such weight is
  sized by its format, and the dense image contains the affine symbol.
- **Allowed files:** `crates/moxie-plan/src/{expert.rs,selected.rs,lib.rs}`
  (lib only for re-exports), `crates/moxie-executor/src/{grouped_device.rs,
  grouped.rs,residency.rs}` and `crates/moxie-executor/tests/grouped_device.rs`
  (rename only), `crates/moxie-kernels/cuda/dense_graph.cu`,
  `crates/moxie-kernels/src/lib.rs`, this task's Result.
- **Non-goals:** executing an affine node on the device (next task); TP/PP
  lowerings with formats (`lower_selected_ordered` and the pipeline reject a
  non-empty format map by not taking one); quantized experts in the dense
  graph; raising the affine kernel's `max_input` bound; any importer or model
  builder change.

## Numbered changes

1. **Rename.** `ExpertWeightFormat` → `WeightFormat` everywhere (mechanical;
   no behaviour change). Doc comment: "How a weight value is stored: BF16, or
   the document 03 affine descriptor. Shared by expert and dense linear
   plans."
2. **Sections.** Add to `WeightFormat`:
   `pub fn sections(self, outputs: u64, inputs: u64) -> Option<AffineSections>`,
   `None` for `Bf16` or invalid geometry, with
   `pub struct AffineSections { pub codes: (u64, u64), pub scales: (u64, u64),
   pub zero_points: Option<(u64, u64)>, pub group_index: Option<(u64, u64)>,
   pub bytes: u64 }` (each pair is `(offset, len)`). Order: codes, scales, zero
   points (when `zeros`), group index (when `mapped`, `inputs * 4`). **Each
   section starts at an offset aligned up to 16 bytes**; `bytes` is the end of
   the last section. Lengths: codes `outputs * ceil(inputs * bits / 8)`
   (per-row stride as `payload_bytes` computes it), scales `outputs *
   ceil(inputs / group) * scale_bytes`, zero points `outputs * ceil(inputs /
   group) * 2`. Checked arithmetic throughout. `payload_bytes` and `bytes` are
   unchanged (experts keep their layout).
3. **Catalogue and image.** In `dense_graph.cu`, add `#include
   "affine_linear.cu"`. In `dense_graph_catalogue`, per SM add two
   descriptors (`Int4`, `Int8`): id `dense-affine-linear-<profile>-v1-<sm>`,
   `abi_version: DENSE_GRAPH_ABI`, `operation: Linear`, inputs `[bf16
   activation, Weight(width)]`, output BF16, `Bf16InF32Acc`, `FinalBf16Rne`,
   `ContiguousRowMajorV1`, **the affine catalogue's shape bounds**, workspace
   `Zero`, symbol `AFFINE_LINEAR`. If the include causes a duplicate symbol or
   macro clash, stop and send `DECISION`.
4. **Lowering entry.** `pub fn lower_selected_with_formats(graph, workload,
   capability, catalogue, formats: &BTreeMap<ValueId, WeightFormat>) ->
   Result<SelectedPlanCandidate, Error>`: same device check as `lower_dense`,
   then `lower_dense_mode(…, true, empty orders/combines/ownership/joins,
   formats)`. Add the `formats` parameter to `lower_dense_mode`; the three
   existing callers pass `&BTreeMap::new()`.
5. **Validation, before the node loop** (typed `invalid("weight_formats", …)`
   refusals): every key is one of `graph.weights()`; its role is
   `Weight(Bf16)`; the format is `Affine` (a `Bf16` entry is refused as
   meaningless); the weight is consumed by exactly one node, which is an
   `OpParams::Linear { bias: false, .. }` as its input 1; `format.sections(
   out_features, in_features)` is `Some`.
6. **Selection.** In the node loop, for a `Linear` whose weight has a format,
   replace the weight operand in `roles` with
   `KernelOperand::Weight(WeightPrecision::expect(format.precision()))` before
   matching. Nothing else in the match changes.
7. **Sizing.** Where weight values are placed (about 1075–1092), a weight with
   a format uses `sections(out, in).bytes` as `logical_bytes` (physical aligned
   as now). The `PlannedValue.role` stays the graph's role.
8. **Exposure.** Store the map in `SelectedPlanCandidate`
   (`weight_formats: BTreeMap<ValueId, WeightFormat>`, empty for every other
   lowering) with `pub fn weight_formats(&self) -> &BTreeMap<ValueId,
   WeightFormat>`.
9. **Tests (two, in `selected.rs`'s existing test module, using the reduced
   Gemma fixture its tests already use):**
   - *Admission:* formats on every attention and MLP projection weight of one
     layer (INT8 group 32 BF16 scales symmetric on some, INT4 group 32 F16
     scales with zeros on others). Assert each formatted node selected an
     affine descriptor (`symbols == [AFFINE_LINEAR]`), each formatted weight's
     `logical_bytes == sections(..).bytes`, every other node's descriptor and
     every other value's bytes equal `lower_selected`'s for the same graph.
   - *Refusal:* one test, table-driven over: a format on the embedding weight;
     a `Bf16` format; a group size of 64. Each must be a typed refusal naming
     `weight_formats`.

## Contract before implementation

- **Semantics:** unchanged for every existing caller (empty map). For a
  formatted weight the equation is document 03's, `W = (Q - Z) * S`, rounded
  to BF16 per tile, FP32 accumulation, one final BF16 rounding (the affine
  kernel's own contract).
- **Resources:** a formatted weight is charged its section bytes, not BF16
  bytes. No workspace.
- **Failure:** refusals are typed and happen at lowering, before admission.

## Acceptance

**Host gates:** `cargo fmt --all -- --check`; `cargo clippy --workspace
--all-targets --locked -- -D warnings`; the executor driver-feature clippy
(`driver,paged-attention-binding,paged-attention-test-hooks`); `cargo test
--workspace --locked`; `cargo xtask arch-check`; `cargo xtask spec-check`.

**GPU gates** (the dense image changed; `CUDA_DEVICE_ORDER=PCI_BUS_ID`):
`dense_gemma_device`, `dense_tp2_device`, `cargo xtask-cuda test-gpu` (63/63,
SM86 and SM120).

**Coverage check (one mutant, reverted after):** in change 7, keep BF16
sizing for formatted weights. The admission test must fail.

**Stop conditions:** a change conflicts with the code; the include clashes;
any existing output changes; a file outside the allowed list is needed.

## Result, filled after work

(pending)
