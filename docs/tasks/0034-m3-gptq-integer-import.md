# Task 0034 — shared GPTQ integer serialization import

Status: active; contract before implementation, 2026-09-16.

## Identity and authority

Owner assignment: implement M3 recovery; root `/home/rodrigo/Developer/moxie`,
main after task0032 commit16e5760 and publication checkpoint356f965. Static CT task0033 was committed as46f0876; preserve it. Legacy and model roots read-only.
Normative specs03/06/08/09; existing affine, compressed_tensors, selection,
repack work units, source ownership and canonical publication govern this work.

Pinned source AutoRound0.15.0 tag object4235489c9bc040ce51f49342b5fd8c5424784abc,
commit7269a3cd89afbb516d5a25a254ab1faea9bb1aec. Inspected small source files under
`/tmp/moxie-autoround-0.15.0` (no checkpoint code executed): export/utils.py
selects torch/qlinear_torch_zp.py for auto_gptq. Packer stores zero-minus-one;
reader adds one before subtraction. qlinear_torch.py's direct-zero layout is a
different serialization and must never be substituted because names resemble it.

## Bounded deliverable

Add an I/O-free, fallible shared format importer for GPTQ-v1 integer layout,
then wire it into the existing bounded source/repack path, selection and config
reporting. Existing affine canonical format, ledger and publication writer remain
sole owners. No quantizer, runtime method branch, whole-weight FP16 expansion or
new storage container. Preserve all source scale encodings and optional group
maps. Both dense and expert module shapes are consumers; graph execution is
separate from importer acceptance.

## Contract before implementation

- qweight I32 `[ceil(K/pack), O]`, low-first input lanes; qzeros I32
  `[groups, ceil(O/pack)]`, low-first output lanes; scales `[groups,O]`.
- Decode q as unsigned-minus-bias. Decode stored zero as unsigned-plus-one-minus-
  bias into i16 (never clip at signed code endpoints). Preserve source scale bits
  while transposing metadata to output-major canonical order.
- Optional I32 g_idx `[K]` maps original logical columns to scale groups; validate
  exact length/nonnegative/in-range and preserve it. Missing map means contiguous
  grouping only when the source declaration agrees. Refuse unsupported remaps.
- Support INT4/INT8 group32/128 and header-specified F16/BF16/F32 scales within
  existing affine contract. Checked dimension/byte arithmetic before allocation.
- Conversion tiles bound source+destination live payload, no full tensor decode.
  Cancel between existing units; source identity and checksummed resume remain
  unchanged. Every new allocation/error path in the format importer is fallible.
- Independent byte-equation oracle and pinned source packing, exact canonical
  FP32 values; separately describe exporter scale/output rounding. No tolerance
  changes, model-output claim or performance claim.

## Acceptance

Full code ranges, plus-one zero extremes, scale transpose/dtype, tails, malformed
shapes/length/maps, bounded row tiles and allocation refusal. Synthetic publication
round-trip with source oracle, restart/budget regression, two distinct dense/MoE
row shapes. Read-only bounded real samples from Intel GLM revision
5eee1846f0321058ed73745f9aa16f2aaf0fc0a0 and Qwen revision
4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897 only after confirming catalog identity;
no bulk conversion. Record source hashes and actual scale dtype. No passing
metadata-only declaration is counted as a fixture.

Affected format/repack host tests, architecture/spec and clippy; no CUDA claim
from importer tests. Update truthful capability/user docs. Independent review
before acceptance. Stop for an owner decision if preserving a source needs a
new canonical precision, weaker quality gate or bulk artifact write. Temporary
synthetic fixtures deleted by tests. Source files kept as bounded reference
material until evidence pins hashes; no new runtime dependency on Python.

## Result

Implemented, validation in progress; independent review pending.

Shared GPTQ-v1 decoder and bounded publisher preserve I32 packed inputs,
zero-minus-one semantics, source scale bits, tails and explicit g_idx. Generated
plans recognize pinned AutoRound0.15.0 and its 16-bit passthrough overrides;
unsupported overrides or regex syntax refuse. The offline program uses the
bounded Rust regex matcher (one 1MiB compiled matcher at a time), allowed only
on the repacker dependency edge. Group-map metadata and expanded i16 zeros are
included in automatic budgets. Existing F16 *unquantized weights* remain refused;
F16 scales are supported. Discovery assumes the pinned exporter's aligned input
extent; hand-written GPTQ selections carry explicit logical [O,K] for tails.

Two synthetic publication cases pass, including tiny gathers, both widths,
noncontiguous group maps and source-to-canonical byte equality. The importer
allocation sweep refuses each measured allocation, including map validation,
without aborting. Full format/repack host suites passed before signed-scale
integration; affected tests are being rerun after that integration.

Read-only pinned samples exposed negative F16 scales in GLM. The owner approved
preserving them exactly; ADR0030 changes only the affine scale sign restriction.
Afterward **43,008 canonical FP32 values match the source equation bitwise**:
GLM [4096,2048], outputs0..8 and4088..4096; Qwen [2560,640], outputs0..8 and2552..2560.
Both use group128 F16 scales and no g_idx on the sampled module. Source revisions
are checked against the download metadata and catalog pins. No checkpoint code
is executed and no model artifact is written. Sample/log hashes are in
`results/m3-recovery/autoround-samples.log`; header hashes:
GLM dc338195ea26db754f308f54a6d05e9490c01a03c6972fa74e531b133f887ffb;
Qwen c8a3adef4e064142c043ac15eb5684bbca6aa8d382ac92f13e060160add2ca16.
This is bounded affine-value evidence, not whole-artifact, model-output or
performance qualification.
