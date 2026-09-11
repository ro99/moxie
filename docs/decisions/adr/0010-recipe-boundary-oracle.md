# ADR 0010 — deepseek-recipe as application-boundary oracle (post-start)

- ID / date / author / status: 0010 / 2026-09-11 / owner-directed, recorded by
  implementation agent / adoption posture accepted, integration pending.
- Classification: owner requirement (record the opportunity); roadmap default
  (oracle-now, dependency-later posture and M8/M11 placement). No owner gate is
  resolved.
- Scope and owning shared component: `moxie-service` / `moxie-protocol` and apps
  (future consumers only). Shared engine crates (`moxie-engine`, `moxie-memory`,
  `moxie-state`, `moxie-sampling`, `moxie-plan`, `moxie-executor`, kernels) are
  explicitly out of scope: the upstream artifact is DeepSeek-family-specific and
  must never enter family-neutral code.
- Supersedes / superseded by: none. Amends `docs/spec/06-implementation-roadmap.md`
  §§ M8/M11 by owner direction; digest regenerated alongside. Spec files are
  local-only, so the portable record is this ADR.

## Problem and mechanism

Post-start upstream artifact. DeepSeek's `deepseek-recipe` (MIT, v0.1.0,
checkout `../deepseek-recipe` at `8cadfed`, Rust workspace + Python bindings)
converts Messages / Chat Completions / Responses requests into one
`Conversation` IR, renders V4/V4.1 prompts (special tokens, DSML tool markup,
thinking delimiters, effort template, tool-result ordering), and incrementally
parses model output back into protocol events (reasoning/tool/JSON-fence/stop
state machine with incomplete-marker retention). Inference, sampling, usage
accounting and transport are the caller's. Image fetching/preprocessing is
trait-based with byte budgets, but its defaults pull OpenCV native and reqwest,
and its own docs warn the default fetcher does no SSRF filtering. Its README
defers logprobs, JSON-schema enforcement, `n > 1`, conversation storage and
server tool execution to the caller.

That seam is exactly Moxie's doc-02 application boundary, and the artifact
covers the DeepSeek rows of M8's protocol/template/stream matrix and M11's O3
parity work — with gaps (logprobs, real grammar enforcement, busy/usage policy)
that remain Moxie's by contract.

## Options examined

- Test-oracle and fixture reference now, no dependency. Zero supply-chain or
  build cost; golden V4.1 renderings and conversion/validation bounds feed O3
  transcript and equivalence tests. Selected as the immediate posture.
- Narrow crate dependency later (`deepseek-recipe{,-core,-encoding}` without
  image defaults), decided by the M8 task: service/protocol layer only, DeepSeek
  family only, behind Moxie's own protocol types, pinned with an arch-check
  allowlist entry. Deferred, not rejected.
- Port where Moxie must own the behavior: constrained generation (grammar state
  joins speculation rollback), logprob reporting, usage/cancellation. An external
  crate cannot satisfy these.
- Never: OpenCV/reqwest image defaults in the service path, Python bindings (no
  production Python), mock example servers (no auth, loopback-only by their own
  warning), or any use inside shared engine crates or model definitions.

Upstream drift, family coverage (six other families need their own renderers)
and Moxie's dependency discipline (M0: no large dep from its README alone) argue
against anything broader. Estimates only; no measured integration exists.

## Decision and authority

Effective roadmap delta (applied by this ADR to the local spec copy §§ M8/M11;
digest regenerated alongside):

- M8 item 2 names `deepseek-recipe` conversion/validation and V4/V4.1
  rendering/stream fixtures as the oracle for the DeepSeek rows of the
  endpoint/field matrix; a crate dependency is an explicit M8 service-layer
  decision, not a default.
- M11 item 2 names its V4/V4.1 rendering fixtures as O3-parity reference and
  excludes its OpenCV/reqwest image defaults; shared media preprocessing still
  goes through common operations.

Authority: owner recording requirement; posture and ordering are defaults.
O1–O7 unchanged. No catalog, precision, concurrency or dependency commitment.

## Evidence and acceptance

Migration evidence (not gates): repo README scope lists, `request/`,
`protocol/validation.rs`, `encoding/v4/{mod,dsv41}.rs`,
`stream/state_machine.rs`, `core/conversation.rs`, `image/lib.rs` limits and
trait seams, `docs/development.md` checks and SSRF warning. A future M8 task
must bring vendored fixtures with pinned upstream revision, affected-consumer
reruns, and any allowlist entry; protocol conformance is then proven by Moxie's
own transcript/equivalence gates per doc 07, never by upstream's test suite.

## Enforcement and removal

`xtask arch-check` must reject `deepseek-recipe*` imports from shared engine
crates and model crates; only the service/protocol composition side may propose
an allowlist entry at M8. Re-evaluate on upstream API drift, license change, or
a second family needing the same treatment; prefer extending the oracle corpus
over widening the dependency.
