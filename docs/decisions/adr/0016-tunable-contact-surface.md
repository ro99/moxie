# ADR 0016 — Tunable contact surface, after llama.cpp server flags

- ID / date / author / status: 0016 / 2026-09-12 / implementation agent / proposed.
- Classification: roadmap default. It clarifies product philosophy already
  implied by documents 01, 02 and 05; it resolves no owner gate and changes no
  numerical, precision, context, concurrency or catalog contract.
- Scope and owning shared component: `moxie-service` / `moxie-protocol` and the
  CLI + HTTP apps as thin clients of one generation service. Planner/sampler
  capability reporting belongs to `moxie-plan` / `moxie-sampling`; this ADR only
  fixes the surface rule they report through.
- Supersedes / superseded by: nothing. It constrains how M8/M11 satisfy
  [O3](../owner-gates.md#o3--legacy-surface-compatibility).

## Problem and mechanism

The owner pointed at llama.cpp's server README
(`tools/server/README.md`, ~2,100 lines): every behavior that matters is a
user-visible flag with documented default, precedence (CLI over
`LLAMA_ARG_*`), and per-request override — threads/CPU affinity, ctx/batch/
ubatch/keep, RoPE/YaRN scaling, KV offload, K/V cache types, load/lazy modes,
device selection, tensor overrides, cpu-moe/gpu-layers/split-mode/tensor-split/
main-gpu/fit, LoRA/control-vector, model sourcing, logging, full sampler
ordering plus seed/temp/top-k/p/min-p/typical/penalties/DRY/XTC/logit-bias/
grammar/JSON-schema, server host/port/timeout/slots/cache/metrics/auth/CORS/
chat-template/reasoning, and the whole `spec-draft`/`spec-ngram` speculative
family. There are no silent constants; each flag is a tunable contact surface.

Question asked: are Moxie's plans aligned with that philosophy? Answer found on
2026-09-12 against specs 01, 02, 05, 06 and the O1–O7 register: the philosophy
is present in contracts but absent in code, and several llama categories have
no named Moxie counterpart. Without a recorded decision, M8's "HTTP endpoint/
field matrix" and M11's "O3 parity report" could be judged prose rather than
surface. This ADR pins the rule so the gap is tracked, not drifted around.

## Options examined

**A. Implicit defaults (status quo).** Keep the existing contract language
(`auto` reports, `required` errors, presets + overrides, capability matrices)
and let each M8 task invent flags as needed. Rejected: it preserves the
philosophy in principle but guarantees an ad-hoc surface — exactly the drift
R25 attributes to the legacy project. No matrix, no precedence rule, no
startup-vs-per-request scoping.

**B. Full llama.cpp parity, including multi-user serving.** Adopt every llama
flag family verbatim: `-np/--parallel`, continuous batching, unified KV,
slots, embeddings/rerank/pooling, multimodal, tools/MCP. Rejected on product
grounds, not effort: document 01 fixes one interactive user, `n=1`, busy
`409`, no indefinite queue. Multi-tenant scheduling contradicts the admitted
single-generation resource envelope and the ledger in document 02. Parity here
would be scope violation, and document 01 forbids silently re-scoping through
a task.

**C. Explicit tunable-surface rule with stated single-user divergence
(selected).** Every behavior-affecting choice is a documented tunable with a
default, a scope (startup vs per-request), and `auto`/`required` semantics;
anything unsupported is an actionable error or a declared matrix gap, never a
silent constant or ignored field. Single-user remains the product boundary, so
the multi-user scheduler family is the one deliberate non-parity, recorded
below rather than discovered later.

## Decision and authority

1. No behavior-affecting constant without a contact surface. Any default that
   changes tokens, placement, memory, latency or quality (shapes excluded —
   symbolic dims, not magic maxima, per document 02) must be: documented with
   its default, settable at its owning layer, visible in startup/request
   diagnostics, and covered by the support/capability matrix. This restates
   document 05's "each active stage is shown" and document 02's "diagnostics
   explaining rejected alternatives" as a universal rule.
2. Presets supply defaults; explicit user values override them; `auto` reports
   what it selected; `required` returns an actionable error rather than
   silently disabling. No hidden "balanced" filters, no quiet fallback. Per
   documents 01 and 05 and the README "Meaning of first-class".
3. Unknown fields that could change semantics are structured errors, not
   ignored input; harmless client metadata is explicitly allowlisted. Per
   document 05's HTTP scope. CLI and HTTP reach the same engine configuration
   for equivalent requests (document 05 CLI + document 06 M8 exit); transcript
   tests enforce it.
4. Deliberate non-parity, recorded once: no multi-user parallel slots,
   continuous batching across users, or per-model concurrency switches. Busy
   generation returns the documented retryable status. Branch batches,
   speculative verification and prefill microbatches serve the one user only.
5. This ADR decides no catalog, quality, storage, performance or surface-removal
   question. O3 stays OPEN: which legacy fields/flags/presets/templates must be
   byte-compatible vs aliased vs deprecated still needs the owner's ruling, and
   the M8 flag-surface matrix below is the evidence that ruling will act on.

This is a roadmap-level default inside existing document 01/02/05 authority.
It needs no owner approval and takes none.

## Evidence and acceptance

Reviewed 2026-09-12: `docs/spec/01-product-and-decisions.md` (boundaries,
delegated defaults, O1–O7), `02-architecture-and-common-api.md` (planning
contract, plan-cache keys, auto/fixed-plan), `05-sampling-api-and-cli.md`
(pipeline, required knobs, HTTP/CLI scope), `06-implementation-roadmap.md`
(M8/M11 exits), `docs/decisions/owner-gates.md` O3, and llama.cpp
`tools/server/README.md` Common / Sampling / Server-specific / speculative
tables (first ~700 of 2,138 lines; flag categories enumerated, not every row
quoted).

Aligned today: sampler knob list, ordered pipeline with diagnostics,
`auto`/`required` discipline, plan-cache identity, bounded-request and
auth/CORS rules, CLI+HTTP same-service requirement.

Known gaps (no named Moxie counterpart; each becomes a matrix row in M8, not a
silent default): CPU threading/affinity/priority/poll/NUMA; logical vs
physical batch and `keep`; RoPE/YaRN override family; KV-offload toggles;
per-K/V cache types (Moxie pins only cache ≥16 bits); load/lazy
(mmap/mlock/dio) modes; device list, tensor overrides, cpu-moe/n-cpu-moe/ffn,
gpu-layers/split-mode/tensor-split/main-gpu/fit-target/fit-ctx; LoRA/
control-vector; HF/docker/URL model sourcing; log verbosity/prefix/jsonl;
speculative draft/ngram families; server host/port/timeout/threads-http/
SSE-ping/prompt-cache/metrics/props/slots/auth/SSL/chat-template/reasoning/
embeddings/rerank/multimodal/tools/MCP; `LLAMA_ARG_*`-style env precedence.
Current code proves the distance: `moxie-cli diagnostic` exposes only
`--shape/--prompt/--max-new/--chunk/--temperature/--seed/--cancel-after`.

Acceptance: M8 delivers the flag-surface matrix (every tunable: name, scope,
default, `auto`/`required` behavior, request override, diagnostics key,
matrix entry) plus the O3 alias/deprecate report; M11 blocks release if a
required behavior is reachable only through an undocumented constant. Until
then this ADR's claim is "philosophy recorded, surface unbuilt" — no
completion is manufactured from it.

## Enforcement and removal

- M8 tasks cite this ADR when adding or refusing a tunable; reviewers reject
  undocumented behavior-affecting defaults and ignored semantic fields.
- Transcript tests (document 05/06 M8 exit) cover CLI/HTTP equivalence under a
  fixed deterministic profile; SSE/no-stream equivalence; cancel/second-turn
  recovery.
- No temporary path, no expiry: this is a permanent product rule. Re-evaluate
  only if document 01's single-user boundary changes, which is an owner
  decision, not an implementation inference — at that point the deliberate
  non-parity in point 4 is re-opened through O3/O6, not edited away here.
