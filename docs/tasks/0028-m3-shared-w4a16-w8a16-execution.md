# Task 0028 — M3 item 3: shared W4A16 / W8A16 execution

Status: **proposed**; contract written before implementation.

## Identity and authority

- **Task ID / milestone / owner**: 0028, M3 item 3 of
  [the roadmap](../spec/06-implementation-roadmap.md): *"Implement shared
  W4A16/W8A16 dense and expert paths for SM86; qualify SM120 separately.
  Preserve BF16 and explicitly qualified FP16 activation/scale semantics.
  Bounded reference dequantization is not an acceptable final fast path by
  assertion."* Owner decision of 2026-09-14 makes this the next work and states
  that accepting task 0027 is **not** a precondition.
- **Writable root / base**: `/home/rodrigo/Developer/moxie`, branch `main`,
  base `ed8ac11`. No dirty paths at start.
- **Read-only**: `/models` and `/fast/models`. This task reads checkpoints; it
  converts nothing and writes nothing into either root.
- **Owner gates**: O2 (bit-identical repack) and O5 (user-managed storage) are
  resolved and unchanged by this work. **O6/O7 (performance) remain open and
  are not touched**: this task establishes *correctness and coverage*, not
  speed, and no result from it may be recorded as a performance claim.
- **It is also what makes [experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md)
  runnable**, which is what decides whether repacking is retained at all
  ([ADR 0027](../decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)).
  Running that experiment is **not** in this task's scope; making it possible is
  a consequence, not a deliverable.

## Bounded deliverable

- **One concrete outcome**: a dense linear whose weight is a canonical affine
  INT4 or INT8 tensor executes on a real SM86 device, through the shared
  operation path, and matches an independent CPU oracle within a predeclared
  threshold.
- **Sole owning shared component**: `moxie-kernels` owns the device image and
  its catalogue descriptors. `moxie-executor` owns binding and launch.
  `moxie-types` already owns the precision and descriptor vocabulary
  (`ExecutionProfile::W4A16` / `W8A16`, `KernelOperand::Weight`) and is
  **not** expected to change.
- **Allowed files**: `crates/moxie-kernels/cuda/*.cu`, `crates/moxie-kernels/src/`,
  `crates/moxie-executor/src/chain.rs`, their tests, and the support matrix.
- **Non-goals, explicitly**: expert/MoE quantized paths (a following task);
  SM120 qualification (separate, below); FP16 activations; W4A4/W8A8; any
  attention change; any performance tuning or claim.
- **Forbidden shortcut, named because the roadmap names it**: dequantizing a
  whole weight to BF16 on device and calling the existing BF16 kernel is
  acceptable **only** as a declared correctness stepping stone, recorded as
  *not the fast path*, and it does not satisfy this task. The delivered path
  unpacks and dequantizes into 16-bit tensor-core tiles inside the kernel, with
  no whole-tensor BF16 copy resident.
- **Second consumer**: the same kernel must serve a group-32 asymmetric INT4
  tensor and a group-128 symmetric INT8 tensor — two different group rules and
  two different zero-point sections — or it is one checkpoint's kernel wearing
  a shared name.

## Contract before implementation

- **Equation**: `W = (Q − Z) · S` per [ADR 0023][adr23]'s canonical layout, then
  `y = x · Wᵀ`. `Q` is the packed code, `Z` the i16 zero point of the code's
  group (0 when the tensor is symmetric and has no zero-point component), `S`
  the group's scale.
- **Components, and their shapes**: a canonical affine tensor is three device
  buffers, not one — `codes` (`U8`, `out × ceil(in/2)` for INT4; `I8`,
  `out × in` for INT8), `scales` (`out × groups`), and `zero_points` (`I16`,
  `out × groups`, **absent** when symmetric). The INT4 low nibble is the
  **earlier** input column. Group rules are contiguous-32, contiguous-128 or
  per-channel.
- **Precision and accumulation**: BF16 activations, FP32 accumulation
  (`AccumulationPolicy::Bf16InF32Acc`), `RoundingProfile::FinalBf16Rne` at the
  output boundary. Scales are read in the source's own encoding and converted
  once per group, not per element.
- **Hardware**: SM86 first. SM120 is **qualified separately** and a passing
  SM86 run is not evidence for it; the support matrix gets one row per
  architecture actually exercised.
- **Memory and lifetime**: the three components are admitted through the
  existing ledger and residency authority (task 0020) as one logical weight
  with three ranges. No component is readable until its copy is **observed**.
  Peak device memory must not include a dequantized copy of the weight.
- **Cancellation and failure**: unchanged from the selected chain — a refused
  admission is a typed refusal before any allocation, and a cancelled launch
  retires its lease and returns every charge.
- **Independent oracle**: `moxie_format::affine::Affine::reconstruct` on the
  host, which is task 0024's exhaustively tested decoder, followed by a CPU
  BF16 matmul with FP32 accumulation. The oracle and the kernel share **no**
  code path: the oracle reconstructs the whole weight and multiplies; the
  kernel never materializes one.
- **Predeclared threshold**: for each output element, `|y_kernel − y_oracle|`
  ≤ 2 ULP of BF16 at the oracle's magnitude, and the **exact** integer codes
  and group indices the kernel used are checked separately against the decoder
  on a small shape, so a numerical near-miss cannot hide a wrong group map.
  A threshold widened after seeing a result is a changed gate and needs the
  owner (AGENTS.md product gates).

## Acceptance

1. **Host tests**: descriptor selection — a catalogue entry for `W4A16` is
   chosen for an INT4 weight and refused for a BF16 one, with the refusal
   naming the mismatch; the group map is computed correctly for contiguous-32,
   contiguous-128 and per-channel on shapes with a non-multiple tail.
2. **Device tests, real hardware, all three UUIDs**: the kernel's output
   matches the oracle within the threshold above, on (a) a group-32 asymmetric
   INT4 tensor with zero points spanning both signs, (b) a group-128 symmetric
   INT8 tensor with no zero-point component. Both shapes include a row count
   that is not a multiple of the tile.
3. **Whole-integer-range coverage**: every INT4 code 0..=15 and a signed INT8
   range appear in at least one tested tensor, with negative and asymmetric
   zero points — M3's exit gate requires this and a random fixture does not
   guarantee it.
4. **No dequantized weight resident**: the measured peak device allocation for
   the launch is below the size a BF16 copy of the weight would need. This is
   a memory-boundedness assertion, **not** a speed measurement.
5. **A real module, read-only**: one module of an already-published canonical
   artifact executes. No bulk conversion, no download, no write under
   `/fast/models`.
6. **Support matrix**: one new gate ID per architecture actually exercised,
   each linked to the command that passed. An architecture that was not run is
   recorded as unmeasured, never as supported.
7. **Gates**: fmt, clippy, spec-check, arch-check, the host suite and the
   device-feature lane all pass; the mutation battery runs **once** on the
   final tree before acceptance, with its result recorded.

## Exact condition requiring owner direction

- If the only path that meets the numerical threshold on SM86 is a whole-tensor
  device dequantization, **stop and report**. That is the shortcut the roadmap
  forbids by assertion, and substituting it silently would be the same failure
  this milestone has already produced twice: a claim checked against a copy of
  itself.
- If SM120 needs a different kernel rather than a recompile, that is a separate
  task, not an expansion of this one.

## Result, filled after work

*(empty — no implementation has started)*

[adr23]: ../decisions/adr/0023-canonical-affine-payload-and-repack-journal.md
