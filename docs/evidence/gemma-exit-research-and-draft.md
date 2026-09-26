# Gemma exit: research note and task contract draft

Reviewer (Opus), read-only, 2026-09-26. The byte figures below are summed
from the checkpoints' own safetensors headers on this machine.

- Part 1: research.
- Part 2: owner questions, in plain words.
- Part 3: engineering choices already made, with reasons.
- Part 4: the paste-ready contract.
- Part 5: whether it is safe as one task.

---

## Part 1. Research

### 1.1 The two checkpoints

**`/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit`** (revision `34ca187d…`)
- Text tensors 33,938,390,040 B, non-text (vision) 1,151,487,072 B, 60 layers.
- A sliding layer is 508,914,802 B; a global layer (`(L+1)%6==0`, 10 of
  them) is 567,406,690 B.
- The embedding `[262144, 5376]` BF16 is 2,818,572,288 B and is tied, so the
  vocabulary projection reads it again.
- **Linears are INT8 packed as I32** (`.weight_packed [out, in/4]`), with
  **BF16 scales** `.weight_scale [out, in/32]` and `.weight_shape` I64[2].
  All scales together are 1,830,420,480 B. Norms and `layer_scalar` are BF16.
- Sliding attention: q 8192 = 32×256, k/v 4096 = 16×256. Global attention:
  q 16384 = 32×512, k 2048 = 4×512, **no `v_proj`**. MLP: 21504 × 5376.

**`/fast/models/google/gemma-4-26B-A4B-it`** (revision `4d7ae498…`)
- Text 50,466,283,580 B, 30 layers, all BF16.
- **Routed experts are 45,675,970,560 B (90.5%)**. Per layer:
  `experts.gate_up_proj [128, 1408, 2816]` and `experts.down_proj
  [128, 2816, 704]` (1.52 GB per layer).
- The dense "spine" (everything else, embedding included) is about 4.79 GB.
- A layer is 1,628,189,954 B (sliding) or 1,657,026,818 B (global). The
  embedding `[262144, 2816]` is 1,476,395,008 B, tied.
- Sliding k/v 2048 = 8×256; global k 1024 = 2×512, no `v_proj`.

### 1.2 Placement: what fits where

**VRAM** (`nvidia-smi`): 5060 Ti 16,311 MiB; each 3090 24,576 MiB.

**KV per token** (K and V both stored), at a stated context of **32,768
tokens**. Sliding layers keep window 1,024 plus one page.
- 31B: global 8 KiB per token per layer, sliding 16 KiB. That is about
  2.5 GiB (global) + 0.85 GiB (sliding) in total.
- 26B: global 4 KiB, sliding 8 KiB, about 0.64 + 0.21 GiB.

**Other overhead:** the NCCL reserve (144 MiB per rank,
`dense_tp_workers.rs:40`) applies to TP only.

| Plan | 31B, per GPU (weights + KV + tied-table copies) | 26B, per GPU | Verdict |
|---|---|---|---|
| One GPU | 33.9 GB | 50.5 GB | neither fits |
| TP2 on the 3090s | 18.43 GiB + 0.14 NCCL + 1.67 KV ≈ **20.2 GiB** | expert-owner TP2: experts 21.27 GiB + spine 4.46 ≈ **25.7 GiB > 24** | 31B fits; 26B does not |
| PP over the 2 3090s | 30 + 30 layers: ≈ 17.6 + 17.6 GiB plus the table on both ends | 46.99 GiB of layers + 2 × 1.38 table > 48 | 31B fits; 26B does not |
| **PP over all 3** | 16 / 22 / 22 layers: **11.05 / 11.94 / 14.57 GiB** | 8 / 11 / 11 layers: **13.72 / 17.06 / 18.43 GiB** (cap 15.93 / 24 / 24) | both fit |

**The 26B can only run as a three-GPU pipeline** (or with experts on the
host; see 1.3). The 31B can run as TP2, PP2 or PP3.

### 1.3 What is missing between the accepted work (+0102) and a runnable step

1. **Weight formats in resident plan sets.**
   - `DensePlanSet::admit` refuses any candidate with `weight_formats`
     (`crates/moxie-executor/src/dense_set.rs:175`). Task 0102 copies that
     refusal (`docs/tasks/0102-…:99,176`).
   - The planner already sizes a formatted weight as its packed section bytes
     (`crates/moxie-plan/src/selected.rs:1433-1450`, `logical_bytes =
     formatted_weight_bytes[..]`). The resident-length check
     (`dense_set.rs:205`, `len != planned.logical_bytes`) therefore already
     holds for a formatted lease.
   - The gap is the refusal plus a proof that the affine launch reads a
     resident address exactly as it reads an uploaded one.
2. **A loader from the downloaded shards.**
   - `ShardSource` (`crates/moxie-executor/src/residency.rs:55-117`) serves
     a tensor's **raw** byte range. That is right for BF16 but wrong for
     INT8: the plan wants the canonical affine layout (codes section, then
     scales, per `WeightFormat::sections`, `moxie-plan/src/expert.rs:672-700`).
   - The row converters exist:
     `PackQuantizedPlan::convert_code_rows` / `convert_scale_rows`
     (`crates/moxie-format/src/compressed_tensors.rs:597-680`, plan built by
     `PackQuantizedPlan::new`, 420).
   - `v_norm_unit_gain` has no source tensor and must be synthesized as BF16
     ones (task 0104, `gemma4_source_tensor` → `None`).
   - The residency pattern to follow is acquire → `drain_reads` →
     `perform_upload` (`crates/moxie-executor/tests/dense_gemma_device.rs:5811-5842`).
3. **Persistent PP.**
   - Today's pipeline re-lowers every stage each step, uploads weights from
     host bindings on every step, and hands activations over as host bytes
     (`docs/evidence/multi-gpu-lifecycle-before-0102.md`, "Pipeline").
   - At 34–50 GB per step this is not a benchmarkable path.
   - Task 0102 builds persistent **TP** only and lists PP as "a following task"
     (`docs/tasks/0102-…:303`).
   - **No PP task is contracted yet.** The pace table puts persistent PP in the
     M6.1 line "after 0102".
4. **26B experts on device at their real shape.**
   - The dense graph's routed ops (`crates/moxie-kernels/cuda/routed_ops.cu:41`
     route, `:126` project-gelu, `:152` down, `:179` combine) are qualified
     only on reduced fixtures (Shape C, task 0090 evidence).
   - M2 ran the real layer through the separate grouped path (task 0021).
   - Nothing has run 128 experts, top-8, hidden 2816, expert width 704
     through the **dense** routed ops.
   - The alternative is host experts beside the GPU (task 0068). That is a
     single-threaded host kernel reading about 2.85 GB of experts per decode
     token (8 × 11.9 MB × 30 layers), which is not viable for a benchmark.
5. **TP2 for the 31B** would additionally need affine TP slicing:
   - column-parallel: slice code and scale rows;
   - row-parallel: slice each row's codes and its scale groups at `in/2`
     (21504/2 = 10752 = 336 groups of 32; 8192/2 and 16384/2 also align).

   Not needed if the 31B runs as a pipeline (Part 3).

### 1.4 The host quality reference

- **Installed:** transformers 5.5.3 (has `models/gemma4`), torch 2.10.0
  (CPU). Both checkpoints declare exporter 5.5.0.dev0.
- **`compressed-tensors` is not installed.** transformers therefore cannot
  load the 31B's INT8 weights as is (see O1).
- **RAM:** 251 GB total, 243 available.
  - FP32 reference: 31B ≈ 31e9 parameters × 4 ≈ 124 GB; 26B ≈ 25e9 × 4 ≈
    101 GB. One at a time fits.
  - BF16 would halve that, but Strata's Kimi control showed a BF16 reference
    is unsound for comparison (`strata/docs/kimi-k3-runtime.md:113-128`).
    Recommend FP32, the same choice the owner approved for Qwen.
- Token ids come from the checkpoint's own tokenizer, which the ledger allows
  for making inputs (handover, slice 7 item 6).

### 1.5 The benchmark rows the exit needs

Roadmap M6 exit (`docs/spec/06-implementation-roadmap.md:106`): "paired
prefill/decode/quality/memory benchmark for at least the dense and two MoE
stress graphs, plus actual available checkpoints. Results outside variance
justify defaults…".

Task 0090's harness (`docs/evidence/m6-stress-benchmark.md`):
- `stress_graph_benchmark` in `dense_gemma_device.rs`, `--release`,
  5 warm-ups, 30 timed runs per phase;
- median/min/max, device bytes, graph pool, paged state and worst ULP;
- paired against the M5 closure commit.

**No base commit can run a checkpoint** (there is no loader before this
task), so checkpoint rows pair **option on versus option off** on the same
commit (O3).

### 1.6 Strata

Strata ran Gemma 4 31B from a different, MXFP4 checkpoint on one 3090. It
measured 555 tok/s prefill and 29.7 tok/s decode at 348 tokens
(`strata/docs/models/gemma4.md:131-138`). Useful context, but not a pair:
the precision and checkpoint differ.

---

## Part 2. Questions for the owner (plain words)

**O1. How should the reference program read the 31B's 8-bit weights?**
The 31B stores its weights in a compressed 8-bit form. The installed
reference program (transformers) can only read it with one more Python
package, `compressed-tensors`, which is not installed. Two ways:
- (a) install that package, at a pinned version, in the reference
  environment only;
- (b) our reference script unpacks the 8-bit weights itself, using the
  published formula (weight = code × scale).

*Recommendation: (a).* It keeps the reference independent of anything we
wrote. Choose (b) if you prefer not to install packages.

**O2. What quality bar must each Gemma pass?**
Proposed, the same bar you approved for Qwen:
- the model's output scores match the reference within 2% overall
  difference, with 0.999 similarity, at every position of the test prompt;
- the first 32 generated tokens are identical to the reference's.

*Recommendation: yes.*

**O3. What do we compare against, to justify defaults?**
No earlier Moxie version can run these checkpoints, so there is no "before".
Proposal: run each checkpoint with each speed option switched on and then
off, on the same code, and keep an option as the default only where it wins
by more than the run-to-run variation. The options are graph capture on/off
and the fast matrix-multiply library on/off.

*Recommendation: yes.*

**O4. Order of work.**
The 26B only fits split across all three GPUs in sequence. Moxie's
split-in-sequence mode still re-sends every weight to the GPUs on every
token. Making it keep the weights on the GPUs is the next multi-GPU task
after 0102, and it has no contract yet.

The Gemma exit task has to wait for it, so the order would be: 0102, then
"keep the pipeline's weights on the GPUs", then the Gemma exit.

*Please confirm that order.*

---

## Part 3. Engineering choices made here (not owner questions)

- **Both checkpoints run as pipelines.** Only the 26B needs one, but one
  mechanism for both means no TP slicing of 8-bit weights (1.3, item 5).
  - 31B: **two stages on the 3090 pair**, layers 0–29 and 30–59, because it
    fits there and the 5060 Ti would add a slower hop.
  - 26B: **three stages**: 5060 Ti layers 0–7, 3090 #1 layers 8–18, 3090 #2
    layers 19–29.
  - GPUs are identified by UUID. This is fixed-plan mode.
- **Context for memory accounting: 32,768.** Benchmark prompts: 512 and
  8,192 tokens, then 64 decode tokens. The long prompt satisfies the exit's
  "no regression hidden by a shorter prompt".
- **Timing repetitions:** 3 warm-ups and 10 timed runs per phase (the
  checkpoint scale makes 30 impractical). Variance is reported as min/max and
  the spread of the medians over 2 runs, as in 0090.
- **Quality is checked on final logits only.** Per-layer dumps would need a
  new debug surface in the executor.

---

## Part 4. Contract draft (paste into `docs/tasks/01NN-m6-exit-gemma-checkpoints.md`)

### Identity and authority
- **M6 exit**, "Exit, Gemma" (handover, "Pace ruling").
- Checkpoints, read in place (ADR 0038):
  - `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` @ `34ca187d836de874b2c7e3edf48f439b9f583772`;
  - `/fast/models/google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
- **Depends on:**
  - task 0104 (accepted: `moxie_cli::gemma::from_checkpoint`);
  - task 0105 (accepted: affine 21,504, attention head_dim 512);
  - task 0102 (persistent TP; the pattern only);
  - **the persistent-PP task** (O4), whose API this task consumes.
- Team: builder Claude Sonnet; reviewer Claude Opus. The design is the
  coordinator's; on a conflict, send `DECISION`.
- Root and branch as usual. Stage explicit paths only.
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Naming rule applies.

### Facts
Part 1.1–1.3 verbatim, with their citations.

### Allowed files
- `crates/moxie-executor/src/dense_set.rs` (drop the formats refusal);
- `crates/moxie-executor/src/residency.rs` (add `CheckpointSource`);
- `crates/moxie-cli/src/gemma.rs` (a loader entry point beside
  `from_checkpoint`);
- `xtask/src/gpu.rs` (one routed-ops case);
- tests:
  - `crates/moxie-executor/tests/dense_gemma_device.rs` (one resident-affine
    test);
  - `crates/moxie-executor/tests/gemma_checkpoint_device.rs` (new, ignored);
- `tools/reference/gemma4_reference.py` (new);
- `docs/evidence/m6-checkpoint-benchmark.md` (new);
- this task's Result.

The persistent-PP API is used, not edited. Needing to edit it is a stop.

### Numbered changes

1. **Resident formatted weights.**
   - Delete `|| !candidate.weight_formats().is_empty()` from
     `dense_set.rs:175`, and adjust its refusal text to "resident plan sets
     require matching dense candidates without host expert joins".
   - Test (in `dense_gemma_device.rs`, driver lane): Shape A with its q/k/v/o
     and MLP linears formatted INT8 group 32, as task 0081's affine test
     builds them. Run once through `DensePlanSet` with the formatted section
     bytes resident (uploaded via `drain_reads` from an in-memory source) and
     once through an ordinary uploaded plan. Logits are byte-identical on
     every visible GPU.
2. **`CheckpointSource`** (`residency.rs`), implementing `ChunkSource`.
   - Built from opened `Shard`s and a map `role → Source`, where
     `enum Source { Plain { shard, tensor }, Affine { plan: PackQuantizedPlan, shard_codes, codes, shard_scales, scales }, Ones { bytes } }`.
   - `read_chunk` accepts only whole-tensor chunks (offset 0, length equal to
     the declared size). Anything else is `InvalidRequest`.
   - `Plain`: `read_tensor_range`.
   - `Affine`: fill `into` with the exact `WeightFormat::sections` layout.
     Codes come from `convert_code_rows(0..out)`, reading the source rows
     through `Shard::read_tensor_range` in 4 MiB row blocks into one reused
     scratch buffer. Scales come from `convert_scale_rows(0..out)` at the
     `sections.scales` offset. The padding between sections is zero.
   - `Ones`: BF16 `0x3F80` repeated.
3. **Loader entry** `moxie_cli::gemma::load_stage_weights(dir, graph:
   &CheckpointGraph, stage_values: &[ValueId], ctx, stream, ledger) ->
   Result<(ResidencyAuthority, DeviceResidency, BTreeMap<ValueId, ResidencyLease>)>`.
   - The chunk role is the graph weight label (the `weight_label` rule in
     `gemma.rs`), and the source comes from `role_to_source_tensor`:
     - a name ending `.weight_packed` becomes `Affine`, with
       `PackQuantizedPlan::new` from the checkpoint's `quantization` and the
       `.weight_shape` tensor;
     - a `None` source becomes `Ones`;
     - everything else becomes `Plain`.
   - `ResidencyRequest::new("gemma checkpoint weights", 4 GiB).device_weights(uuid, Σ planned physical_bytes)`.
   - Then the acquire → `drain_reads` → `perform_upload` loop of
     `dense_gemma_device.rs:5811-5842`, with `AcquireRequest { now: 0,
     deadline: u64::MAX, class: UseClass::demand(Content::DenseSpine), turn:
     TurnId::new(1) }`.
   - On any error, release what was acquired in reverse order and close in
     0102's H5 order: leases, `end_turn`, `retire_all`, `DeviceResidency::close`,
     `ResidencyAuthority::close`. Return the error.
4. **Routed ops at the 26B shape** (`xtask/src/gpu.rs`, new case
   `dense_routed_ops_gemma_26b_shape`).
   - Route (128 experts, top-8, the 26B router math), ExpertMlp (GELU,
     `gate_up [128, 1408, 2816]`, `down [128, 2816, 704]`) and Combine for 1
     and 8 rows, against the host oracle under the gate the reduced
     routed-op cases already use.
   - Run on SM86 and SM120, and register it in both case lists.
5. **Pipeline runs** (`gemma_checkpoint_device.rs`, `#[ignore]`).
   - For each checkpoint: `from_checkpoint`, cut at Part 3's layer
     boundaries, `load_stage_weights` per stage, then admit the
     persistent-PP stage plan sets.
   - Run a 512-token prompt and 32 greedy decode steps.
   - Assert O2's bar against the reference files from change 6.
   - Print per-GPU ledger commitments (weights, KV, workspaces, graph pools)
     for the memory rows.
6. **Reference** `tools/reference/gemma4_reference.py`:
   - transformers 5.5.3, torch CPU, **FP32**, text only;
   - the 31B's weights via O1's ruling;
   - reads a token-id file, writes the logits for every prompt position and
     the 32 greedy tokens to `results/gemma-reference/<checkpoint>/`
     (ignored scratch), with the transformers/torch versions and the
     checkpoint revision in a header.
7. **Benchmark rows** (`gemma_checkpoint_device.rs`,
   `checkpoint_benchmark`, `--release`, `#[ignore]`).
   - Prompts 512 and 8,192, then 64 decode steps.
   - 3 warm-ups and 10 timed runs per phase, 2 independent runs.
   - Options paired per O3: graph capture on/off where the PP path supports
     it, and the cuBLAS BF16 catalogue on/off.
   - Write `docs/evidence/m6-checkpoint-benchmark.md` in 0090's table format,
     plus per-GPU memory and the quality result.
   - Defaults change only where an option wins by more than the spread of
     the medians.

### Escape inventory (loader and resident sets)

| Resource | Owner | Success | Refusal or error | Close |
|---|---|---|---|---|
| Host staging chunk | residency authority | released after upload completes (in-flight is not evictable) | released by `retire_all` | before authority close |
| Device weight lease | the stage's plan set | held for the run | released in reverse on a load error | after every plan closes (0102 H5 order) |
| `DeviceResidency` backing | loader, then the stage owner | persistent | closed after the leases | before `ResidencyAuthority::close` |
| Opened `Shard`s | `CheckpointSource` | dropped after load | dropped | n/a (host only) |

### Tests (exact assertions)
- **Change 1:** byte-identical logits, resident versus uploaded, on every
  GPU.
- **Change 4:** each output within the existing routed-op gate, at both row
  counts, on SM86 and SM120.
- **Change 5:**
  - O2's bar, per checkpoint;
  - every stage's ledger outstanding set is constant across the 32 decode
    steps (0088's rule).
- **Mutant** (run only the 31B quality test): in `CheckpointSource`'s
  `Affine` branch, write the scales at offset 0 instead of `sections.scales`.
  The quality test must fail.

### Gates
- **Host:** fmt; workspace clippy; xtask CUDA clippy; executor clippy with
  `driver,paged-attention-binding,nccl,cublas` and without;
  `cargo test --workspace --locked`; arch-check; spec-check.
- **GPU:**
  - `cargo xtask-cuda test-gpu`, because change 4 is a new case;
  - `dense_gemma_device` with the driver feature (change 1);
  - both ignored checkpoint tests and the benchmark, run explicitly.

### Stop conditions
- The persistent-PP API does not let a stage plan set take externally loaded
  leases.
- A stage exceeds its GPU's admitted capacity at 32,768 context.
- The routed ops fail at the 26B shape, or need a kernel change.
- O1 is unresolved.
- The reference and Moxie disagree beyond O2.
- A file outside the list is needed.

---

## Part 5. One task or a split?

**One task, no split,** once persistent PP exists. What is left here is:
- deleting one refusal;
- one source type;
- one loader function;
- one qualification case;
- a reference script;
- two ignored tests.

Each has a proven pattern in the repository. The real risk is the
dependency: this task cannot start until the persistent-PP task lands (O4).
If the coordinator wants Gemma rows sooner, the 31B alone could run as TP2
through 0102's rank chain. That needs affine TP slicing (1.3, item 5) and
still leaves the 26B waiting, so it is not recommended.
