# ADR 0009 — Engram conditional memory as a post-start roadmap item (parity, not priority)

- ID / date / author / status: 0009 / 2026-09-11 / owner-directed, recorded by
  implementation agent / roadmap placement accepted, implementation pending.
- Classification: owner requirement (Engram belongs on the roadmap); roadmap
  default (where it sits and when it is implemented). No owner gate is resolved.
- Scope and owning shared component: `moxie-memory` (residency), shared semantic
  catalog (`moxie-graph` / `moxie-model-api`), later `moxie-plan` / `moxie-executor` /
  `moxie-kernels`. Concrete model crates own nothing.
- Supersedes / superseded by: none. Amends `docs/spec/06-implementation-roadmap.md` §§ M2/M7 by owner direction; digest regenerated alongside. Spec files are local-only, so the portable record is this ADR.

## Problem and mechanism

Post-start architecture. The normative pack predates DeepSeek's Engram paper
(`2601.07372`, Jan 2026), demo repo (`deepseek-ai/Engram`), Qwen 3.8 Flash Next
and DeepSeek-V4.1-Flash (552B backbone + 196B Engram, Sep 2026). Moxie has no
Engram plan. The owner directs (2026-09-11) that Engram be placed on the roadmap
as one fundamental stone among others — same standing as TP2 or Flash attention,
implemented smoothly at the appropriate moment, not rushed and not prioritized
above them.

Mechanism, from primary sources: suffix $N$-gram ($N=2..3$) of compressed token
IDs → deterministic multi-head prime-modulus hash → static embedding rows →
per-branch context gate against live hidden state → short depthwise causal conv →
residual add before attention/MoE. Deterministic addressing permits
prefetch/overlap of table rows from slower tiers. Paper reports <3% overhead for
a 100B table offloaded to host on its setup; demo mocks attention/MoE/mHC and is
not a production stack.

r/LocalLLaMA threads `1w0198r` / `1wcdati` (read 2026-09-11 via browser; direct
fetch 403s) confirm both the fit and the confusion: the top post refutes
980B-on-SSD 1T-local ("lookup is dumb": last 2–3 tokens only, 4-grams dilute,
500B tables wasteful; real win is smaller models spending active params on
reasoning), while replies assert unmeasured "SSD so fine" / "native precision at
no cost" beside counter-reports (200GB engram safetensors not fitting a 256GB
rig). Nothing there is a measured Moxie gate.

## Options examined

- Model-private $N$-gram tables/caches in `moxie-models-*` or a client. Rejected:
  duplicates residency/state ownership against documents 02/03/09.
- Immediate V4.1-Flash bring-up. Rejected now: O1/O2/O5 `OPEN`, FP4 KV conflicts
  with the INT4/INT8/BF16 family and the >=16-bit cache default (O4), ~510GB
  checkpoint with no bulk authorization. A paper delta is not a quality gate.
- Selected: shared conditional-memory primitive at parity with TP2/Flash
  attention, built through the enforced extension rule when its milestone arrives.
  Proportional cost, noearlier. Estimates only; prefill/decode, BW and quality
  effects are unmeasured on this machine.

## Decision and authority

Effective roadmap delta (applied by this ADR to the local spec copy §§ M2/M7; digest regenerated alongside):

- M2 scope gains one residency class: NVMe-backed deterministic-prefetch tables
  under the single memory authority — canonical chunk identity, demand outranks
  prefetch, event-retained leases, admitted HBM/DRAM footprint plus staging, Zipf
  -aware caching, pressure-driven replan or admission rejection. "Loaded FROM
  NVMe" means NVMe-backed with a small admitted footprint, never zero VRAM/RAM:
  active rows, projections, gating/conv state and transfer staging are budgeted
  every step.
- M7 DeepSeek row gains Engram as listed required math alongside compressed/sparse
  state, routing, mHC/residual mixing: hash addressing, table dtype/layout
  version, collision semantics, gated fusion and short-conv effects become shared
  semantic-operation contracts with oracle, shape/precision/state/partition terms
  before any family graph consumes them. Tokenizer compression stays a shared
  tokenizer/template concern, not a model-private normalizer.
- Ordering: Engram work follows the same gates as its neighbors — synthetic
  fixtures and residency/overlap measurement (M2/M6 lens: SSD random-row IOPS,
  PCIe contention with expert/KV traffic, early-placement overlap-window tension)
  before checkpoint-backed bring-up (M7, O1/O5-gated). No priority over TP2,
  Flash paths, or the active M1.4 slice.

Authority: owner placement requirement; roadmap ordering is a default. O1–O7
unchanged. No precision, cache, concurrency or catalog commitment is made here.

## Evidence and acceptance

Migration evidence (not gates): Engram paper, `engram_demo_v1.py` data flow,
V4.1-Flash API news + HF card (196B Engram, 890 B/tok KV, CED/CSA2 context),
both reddit threads as excitement/correction record. Future bounded task must
bring: equation/shape fixtures, independent oracle, admission/lease/cancellation
tests, second shape consumer, affected-consumer reruns, paired prefill/decode +
quality/memory reporting per doc 07. Paper <3% and thread anecdotes are never
cited as Moxie measurements.

## Enforcement and removal

`xtask arch-check` must reject model-owned $N$-gram tables, caches, allocators
and private prefetch loops; shared kernels dispatch by capability/shape, never
model name. Temporary bridges expire at their named milestone per doc 09 playbook
A. Re-evaluate if the extension rule cannot express the hash/gating math, if
overlap measurements reject NVMe backing on this topology, or if an owner gate
ruling changes catalog/precision/storage scope.
