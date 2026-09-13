# Handover — user-surface ruling to the first repack implementation

## Workspace identity

Writable repository `/home/rodrigo/Developer/moxie`, branch `main`, base
`645e75902ea6bd201f60ac78628178ddc734d985`. No commit or branch change was made.
The initial tree was already dirty: task 0024 importer/review work across shared
Rust crates and tests, `xtask/src/gpu.rs`, AGENTS, task/evidence/model records,
and the ADR 0021 repack correction. Those changes were preserved.

This assignment adds ADR 0022, task 0025 and this handover; adds discovery
paragraphs to `AGENTS.md` and `docs/tasks/README.md`; and appends a placement
continuation near the top of the existing untracked ADR 0021. It changes no
production source or reference document. Git's total diff includes prior work
and must not be attributed to this assignment.

Read-only legacy root `/home/rodrigo/Developer/strata` verified at
`2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; source viewed with `git show`.
`/models` and `/fast/models` remain read-only. Only a Laguna index/shard header
was inspected for the new contract; no new conversion was run. Workspace tests
may exercise their existing read-only real-artifact lanes.

## Completed facts

The [assigned handover](2026-09-13-repack-user-surface-gap.md) requested a bounded
engineer-lead ruling, not an implementation. All five decision deliverables are
recorded in [ADR 0022](../decisions/adr/0022-user-programs-and-canonical-write-authority.md)
and [task 0025](../tasks/0025-m3-offline-repack-publication.md): program inventory,
crate/milestone placement, packaging disposition, write-authority rule contract,
and the first bounded publication task.

Validation, 2026-09-13 (all commands exited zero):

- `cargo xtask spec-check`: all ten reference documents present and unchanged.
- `cargo xtask arch-check`: zero workspace violations; 79 negative and 21
  positive fixtures; 13 existing rules exercised. The new write rule is pending.
- `cargo test --workspace`: passed on the current dirty workspace.
- `cargo fmt --all -- --check` and
  `cargo clippy --workspace --all-targets -- -D warnings`: passed.
- `git diff --check`: passed. A local link walk checked all 17 links in ADRs
  0021/0022, task 0025 and this handover; all targets exist. Source/canonical byte
  totals and logical value count were recomputed from the recorded shapes.

These check the current dirty workspace, not an implementation of task 0025.
Failed commands: none. CUDA compile/device, topology, model quality and paired
performance lanes were not run: this assignment changes documentation only.
No new architecture-rule pass or model claim follows. No support-matrix claim
changed, and no old path was deleted.

## Decisions

ADR 0022 adopts the delegated technical placement: `moxie-repack`, with
`moxie-storage-write` as the isolated shared write-I/O owner; runtime storage
stays read-only. It also names `moxie-server` and `moxie-ops`, preserving the
existing `moxie` chat binary and `xtask` development contracts. Future programs
are built when their real commands land, not scaffolded now.

M3 retains manifest-v1 directories; a `.mox` directory suffix is optional and
not a container signature. M11 item 4 owns the explicit final single-file vs
directory packaging ADR and migration evidence. No owner gate is reopened or
implicitly resolved. Task 0024 remains unaccepted.

## Remaining hypotheses and blockers

The writer, inspector/repacker executable, restart protocol and new
`canonical write authority` architecture rule are **not implemented**. Task
0025's asymmetric acceptance depends on acceptance of task 0024. Full catalog
materialization, unresolved source formats and model-role/completeness integration
remain subsequent M3 work. One module cannot establish a complete model or
satisfy any required ambiguous-layout paired-logit evidence.

The real-module contract fixes Laguna revision, module, expected source payload
1,818,640 B, canonical payload 1,966,080 B, a 128 MiB admitted memory ceiling and
64 MiB temporary-disk ceiling. Sizes were derived from actual header shapes and
checked arithmetically; the payload and content hashes must be checked during
implementation. These are expectations, not observed repack results.

## Next task

Activate and commit [task 0025](../tasks/0025-m3-offline-repack-publication.md)
before implementation or its acceptance tests, recording the current base and
precise dirty ownership. Implement its bounded writer/reader/CLI slice and rule
fixtures under ADRs 0021–0022. Test tiny sources plus exactly the pinned module
under `/tmp`; remove created outputs and retain compact evidence. Stop at the
contract's schema, source-oracle, budget or bulk-write gates. Do not start a
whole-artifact conversion or implement the future server/operator surfaces as
part of this task.
