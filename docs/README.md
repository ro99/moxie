# Documentation placement contract

Two categories, with different version-control treatment.

**Reference documents** — the normative specification. They describe contracts that were decided
before implementation and change rarely. Kept local, **not tracked**, at the owner's direction.

**Living records** — what implementation actually decides, assigns, measures and hands over. These
accumulate as work proceeds and are the operative constraints on later tasks. **Tracked**, because
documents 07 and 09 require them to be, and because R25 attributes part of legacy Strata's drift to
operative instructions living only in an ignored directory.

| Path | Contents | Template | Tracked |
|---|---|---|---|
| `spec/01-09`, `spec/strata-arch-diagnosis.md` | Reference documents | — | no |
| `spec/templates/` | Forms used to author living records | — | yes |
| `decisions/owner-gates.md` | The O1–O7 owner-gate register: question, status, what it blocks, the answer once given | — | yes |
| `decisions/adr/` | Architecture decision records, `NNNN-slug.md` | [ADR.md](spec/templates/ADR.md) | yes |
| `tasks/` | Task contracts, `NNNN-slug.md`, including their filled-in results | [TASK.md](spec/templates/TASK.md) | yes |
| `handovers/` | Bounded continuations between agents/sessions | [HANDOVER.md](spec/templates/HANDOVER.md) | yes |
| `models/` | Model bring-up contracts, one per family | [MODEL-BRINGUP.md](spec/templates/MODEL-BRINGUP.md) | yes |
| `evidence/support-matrix.md` | Capability claims linked to passing gate IDs | [SUPPORT-MATRIX.md](spec/templates/SUPPORT-MATRIX.md) | yes |
| `evidence/benchmarks/` | Benchmark manifests and result summaries | — | yes |
| `evidence/experiments/` | Accepted **and rejected** experiment conclusions | — | yes |

## Rules

Large raw traces, checkpoints and profiler output stay outside git; record a content hash, access
location and retention policy in the record that cites them. `/results/` and `/artifacts/` are
ignored for exactly this.

A record is not complete because a task finished. Doc 09 §E is the completion checklist; doc 07
requires that rejected results be preserved with their mechanism and exact scope, so a later agent
does not repeat the campaign.

A reference document is amended only through an ADR that says what changed and why. Do not edit
`spec/01-09` to match what was built.

Because `spec/` is untracked, a fresh clone has the living records but not the specification. Anyone
setting up a new checkout needs the pack copied in separately.
