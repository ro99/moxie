# Task 0033 — pinned static activation-order import

Status: accepted by the owner as part of M3 closure, 2026-09-19; independent
review finding repaired.

## Identity and authority

Owner assignment: implement the M3 recovery. Root `/home/rodrigo/Developer/moxie`,
main at `356f965`; coordinator.md remains an unrelated carried edit. Legacy
`/home/rodrigo/Developer/strata` stays read-only at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Task0031's publication battery runs
in isolated `/tmp/moxie-m3-publication-battery` at the same commit; nothing in
that worktree may be edited while it runs.

Sources: compressed-tensors 0.17.0 installed under
`/home/rodrigo/.cache/uv/archive-v0/uqs9z2Tvizx6-8cq0I0rd/compressed_tensors`,
`quantization/quant_args.py` ActivationOrdering, pack_quantized/base.py
compression_param_names/decompress, lifecycle/forward_helpers.py. Static aliases
weight ordering: calibration changes, saved logical columns/grouping do not.
Group/dynamic ordering instead requires weight_g_idx and must remain refused
until that payload is preserved. Read existing format/repacker source and
spec03/06/08/09; preserve all numerical/storage gates.

## Bounded deliverable

Shared config reader accepts measured static/weight ordering as contiguous.
Shared import entry resolution and offline repack refuse an unexpected g_idx
rather than silently dropping it. No method-specific runtime, new quantizer,
checkpoint conversion, full-model support or performance claim. AutoRound and
mapped grouping are separate continuations, not completed by this change.

## Contract before implementation

`W[o,k]=(Q[o,k]-Z[o,floor(k/g)])*S[o,floor(k/g)]`; preserve source BF16/F16/F32
scale bytes and logical column order. Static/weight alias selection must produce
identical declarations and bytes to no activation ordering. Unknown, malformed,
and group/dynamic declarations refuse explicitly. A source carrying a map cannot
use the unmapped importer even through a manually authored selection or split
shards. All new diagnostics/name construction on format entry paths are fallible.
No new live state, device work or transfer. Existing source ownership,
cancellation, budgets and publication transaction remain authoritative.

## Acceptance

- Alias/config tests; negative unknown/group/dynamic values; map-bearing source
  refusal through both same-shard and cross-shard paths.
- Symmetric group128 fixture with distinct scales and all integer codes through
  existing shared importer and repacker; exact source equation, no tolerance.
- Read-only real evidence from canada-quant/hy3 revision
  `49228b990c704e4efd67ac420a8e3d5272f820c0`: two bounded module/row samples,
  record headers and hashes. No materialized checkpoint or bulk copy. Local GLM
  metadata currently says revision `80a8729...`, unlike catalog `1c86622...`;
  it cannot establish the pinned catalog fixture and is not used for that claim.
- Affected format/import/repack tests, architecture/spec checks, clippy.
- Document support accurately; owner direction if source semantics disagree or
  any gate would need weakening. Tiny temporary fixtures delete on test exit.

## Result

Pending. Read-only inspection found all five hy3 shard files and no weight_g_idx
in its tensor index. Config SHA256
`8f5f45c43eb2243af93718347626ba1fb4db7abd466c30321b290adcfcce35aa`.
File existence alone is not a full payload-integrity check.


### Measured implementation

Config now accepts `static` and `weight` aliases. Format entry resolution refuses
serialized weight_g_idx; repack checks every shard declared for the module, and
normal discovery rejects map-bearing indexes rather than reporting a map as an
unrelated skipped tensor. Hand selections have only the source shards they name;
no claim is made that an omitted, undeclared file was inspected.

Pinned `quant_args.py` SHA256:
`c606a17f3ee6bf4e820041186c816437cb8a19cf8a1a360d0408083e37377fbd`.
HY3 `down_proj` rows0/4095 and `gate_proj` rows0/1535: **11,264 canonical FP32
values exact**. Source scales stayed BF16; no source-arithmetic or model-output
claim. All indexed shard headers parsed with file-length consistency; source
header budget is explicitly128MiB, payload reads capped64KiB. The original
8MiB default refused an80,755,712-byte estimated header peak and was not
silently bypassed. No whole payload digest was computed. Sample/header digests
are in `/tmp/moxie-0033-repack-review.log` pending durable evidence consolidation.

Format suite passed; repack workflow21, library failure workflow7 and real
sample1 passed. Full CUDA/driver all-target clippy passed with warnings denied.
Architecture79negative/21positive/13rules and ten unchanged spec documents pass.
Initial new tests failed because the generic BF16 weight reader correctly refused
an affine tensor and because the header needed a larger explicit budget; they
now use the checksummed canonical stream and the declared header budget.
Independent review's source-set finding is repaired, and the owner accepted the
task through M3 closure. The later task0034 implements AutoRound/GPTQ maps and
task0035 carries them through shared grouped execution; those continuations do
not change this task's contiguous static-order result.


### Review correction

Independent review found a map beside a separately selected BF16 tensor escaped
the per-module companion scan. Added that reproduction, now refused. Resolution
now scans **all selection-declared shards once**, borrowing header names and
matching selected contiguous modules; it does not repeatedly reopen every shard
for every module. The added source-set scan uses the existing header budget and
identity checks. Final workflow tests: **8 passed**; two-command21 and real1
passed again, with clippy clean.

Evidence boundary: generated discovery inspects the authoritative index and
refuses any declared weight_g_idx. A direct/manual partial selection inspects its
explicit source set, including maps beside other selected tensors; it cannot
establish absence in an omitted file. A future index-aware manual path must
admit its index parse memory rather than silently loading a second full index.
This limitation remains explicit; task0033 does not claim general mapped import.
