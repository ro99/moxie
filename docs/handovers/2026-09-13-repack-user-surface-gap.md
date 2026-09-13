# Handover — repack correction done, user-surface gap open for engineer lead

Owner direction, 2026-09-13 session: "repack is done via an external script — this is plain wrong, and must be fixed right now. Repack is a Moxie program that allow the users to take the formats that are on the plan and convert then to our canonical format (.mox files)." Placement (repack tool, apps, CLI, or otherwise) is explicitly **not** decided here — it is the engineer lead's call. Second owner direction same session: scan for gaps about the user application surface; if none, the answer is clear, if any, write them clearly for the lead.

## Workspace identity

- Writable root: `/home/rodrigo/Developer/moxie`, branch `main`, commit `645e759`.
- Dirty at writing: task 0024's unaccepted implementation (see its handover) plus this correction: `M AGENTS.md`, `M docs/decisions/owner-gates.md`, `M docs/decisions/adr/0020-user-managed-storage-and-canonical-materialization.md`, `M docs/models/laguna.md`, `?? docs/decisions/adr/0021-repack-is-a-moxie-program.md`, `?? docs/handovers/2026-09-13-repack-user-surface-gap.md` (this file).
- Read-only legacy root: `/home/rodrigo/Developer/strata` at `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`. Untouched.
- Checkpoint roots `/models`, `/fast/models`: read-only inputs. Nothing written under either.

## Completed facts

Correction applied (tracked living records only; no `spec/01-09` file touched):

- New [ADR 0021](../decisions/adr/0021-repack-is-a-moxie-program.md): repack is a Moxie program, user-run, offline. Strikes the word "external" for repack. Changes nothing else: O5 substance stays (user-run, one authorized entry at a time, per-task naming of artifact + revision + size + retention, roots `/models` + `/fast/models`, ten authorized revisions); quantization stays external (O1/ADR 0017); repack-only quality stays (O2/ADR 0018). Placement explicitly left open. Notes the `.mox` terminology gap (owner shorthand vs manifest-v1 directory) without inventing a container spec.
- `AGENTS.md` active assignment paragraph: now reads "repacking is a Moxie program the user runs offline — not an external script" (ADRs 0020–0021).
- `docs/decisions/owner-gates.md` O5: original 2026-09-13 verbatim preserved for provenance, correction appended pointing to ADR 0021.
- ADR 0020 header: marked "wording corrected in part by ADR 0021"; body text preserved per the ADR rule (a superseded ADR stays in place with a link, it is not deleted).
- `docs/models/laguna.md` O5 row (living record): corrected, "Nothing here has written a byte" unchanged.

Deliberately **not** edited, with reason:

- `docs/tasks/0024-*.md` line 60–62 ("user-run external script"): committed contract stating nothing below is adjusted once a test has run. Immutable.
- `docs/tasks/0023-*.md` result section, `docs/handovers/2026-09-13-task0023-*.md`: historical records; the standing rule is that rewriting a record to match a later ruling erases what was true when written. Superseded on this point by ADR 0021, listed there.
- No code changed, so no test lane re-runs as proof of work. `spec-check` unaffected (no reference document touched).

## Decisions

- [ADR 0021](../decisions/adr/0021-repack-is-a-moxie-program.md) (owner-directed correction, 2026-09-13).
- Owner rulings restated, not changed: O1 catalog + never-quantize (ADR 0017), O2 repack-only (ADR 0018), O5 user-managed + per-task naming + roots + ten revisions (ADR 0020, wording corrected by 0021).
- Review rules added by 0021: reject new "repack is external" wording; reject any out-of-repo repack duplicating the affine equation; reject any task deciding the repack binary/crate home by inference.

## Remaining hypotheses and blockers — the user-surface gap (for the lead)

**Settled:** logic home (`moxie-format` / `moxie-storage` own "conversion tools" per doc 02 and ADR 0017), logic timing (M3 item 1 builds the inspector/repacker; M11 item 4 packages the offline workflow), who runs full size (user, offline, one authorized entry at a time), what repack preserves (`W=(Q-Z)*S` bit-exact + manifest + chunking).

**Gap: no user-surface inventory names the repack program or when each user-facing binary lands.** Evidence:

1. Doc 05 defines exactly two user surfaces, both generation: HTTP and CLI chat over one service (`GenerationRequest -> stream<GenerationEvent>`). No inspect / repack / verify / publish surface.
2. Doc 07's command contracts list `arch-check`, `test`, `test-gpu`, `test-topology`, `quality`, `bench`, `support-matrix --verify`. No repack slot. `xtask/src/main.rs` USAGE matches — no repack command.
3. M3 item 1 orders the inspector/repacker built; M11 item 4 orders the offline workflow packaged. Neither names a binary. M3 item 2 has two tasks (0018, 0024, both in-memory-only by contract); M3 item 1 has zero tasks.
4. Every workspace crate is `publish = false` (`Cargo.toml`); everything is WIP, so no shipped-binary story exists yet — expected at this stage, but the next M3 task will hit the same wall 0018/0024 hit ("writes no bytes, item 1 owns publication") with item 1 owning no binary.
5. `.mox` terminology: owner shorthand says ".mox files"; manifest v1 (ADR 0005) defines a directory (`manifest.toml` + chunk files), not a single-file container. Packaging undecided.
6. Prior assistant suggestions in-session (repack core in format/storage, entry in a new operator binary, dev smoke in xtask, chat stays chat-only) are **suggestions only, superseded**: the lead decides placement. This handover and ADR 0021 name no binary.

## Next task — engineer lead

One bounded deliverable: a user-surface ruling, recorded as an ADR (or ADR set) amending via reference (do not edit `spec/01-09` directly):

1. Inventory of user-facing programs (chat, HTTP service, repack/inspect/verify, quality, bench, support-matrix verify, operator diagnostics) — what each is, what privilege each holds (notably who may write canonicals under the checkpoint roots).
2. Home of each (binary/crate) and the milestone that builds vs packages it (M3 item 1 vs M11 item 4 for repack).
3. `.mox` packaging: directory as-is, single-file container, or alias — or explicitly deferred with a named milestone.
4. `arch-check` write-authority rule to match (which binary may write canonicals; chat/HTTP never gain write deps).
5. First repack task contract (M3 item 1) unblocked: writer + atomic publish + reader round-trip, inspector (estimate/validate) + restartable bounded repacker, proof on tiny + one real module to a temp dir, nothing under `/models`.

Stop condition: no task decides placement by inference before this ruling lands; tasks citing repack cite ADR 0021 and this handover.
