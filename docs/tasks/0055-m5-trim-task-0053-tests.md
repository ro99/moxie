# Task 0055 — trim task 0053's redundant partition tests

Status: **accepted** (owner, 2026-09-22). Built by Codex `luna`; reviewed by Codex
`sol`, round 1 ACCEPT.

## Identity and authority

- Task0055, M5 ledger item M5.1-e. Builder Codex `luna` (max,
  `/ponytail:ponytail`); reviewer Codex `sol` (high, read-only,
  `/ponytail:ponytail-review`); coordinator Claude Opus. Owner accepts.
- Root `/home/rodrigo/Developer/moxie`, branch `main`, base `c268d4c`.
  Preserve the unrelated carried work (`docs/evidence/specification-version.md`,
  `docs/decisions/adr/0034-*.md`, `docs/decisions/adr/0035-*.md`). Do not
  stage or edit it.
- Requirement: the owner's standing instruction (AGENTS.md, 2026-09-19): one
  test per invariant, at the layer that owns it; no test of a fixture's own
  literals and no second test of something an existing test already fails on.
  The post-acceptance review of task 0053 (engineering log, 2026-09-22) found
  that most of its new assertions could not fail through a production change.
  `OpParams::partition_rule()` returns a constant per op and ignores geometry.
- No owner gate is involved. Test-only; no production code changes.

## Bounded deliverable

Keep the assertions that can catch a production mutation, and delete the
rest.

**Keep:**
- `crates/moxie-interp/tests/reference_graphs.rs`, `the_contract_table_is_what_the_code_says`:
  the `Attention` and `MlaAttention` table rows and their `assert_eq!` on
  `partition_rule()`, `state_effect()` and `output_precision()`. The final
  loop over the built graph's nodes asserting `check_partitionable().is_ok()`
  also stays; it predates task 0053.
- `crates/moxie-interp/tests/mla_reference.rs`,
  `mla_plan_and_interpreter_read_latent_cache_across_prefill_and_decode`:
  the single `assert_eq!(node.contract.partition, HeadShardable { SharedLatentReplicated, GlobalReduction })`
  on the real graph node. It proves the builder copies the rule into the node
  contract.

**Delete:**
1. `reference_graphs.rs`: the whole `match params { OpParams::Attention {..} => …, OpParams::MlaAttention {..} => …, _ => {} }`
   block inside the table loop (currently about lines 820–874). The GQA arm
   checks only the test's literals (`HYPOTHETICAL_RANKS`, `heads`,
   `kv_heads`). The MLA arm re-tests `cache_width`/`decompressed_kv_width`,
   which `moxie-graph`'s `mla_descriptor_declares_latent_not_full_kv_geometry`
   already pins exactly.
2. `reference_graphs.rs`: the `assert!(params.partition_rule().is_partitionable(), "{op} partitionability")`
   in the same loop. The `assert_eq!` on the exact rule, one line above,
   already fails on every mutation this one catches.
3. `reference_graphs.rs`: the comments on the new table rows that describe a
   "hypothetical four-rank lowering" and the MLA cache. They document
   properties the table no longer asserts. The table rows themselves may keep
   their task-0053 geometry values (`heads: 8`, `kv_heads: 2`, the MLA
   descriptor); they are ordinary valid inputs.
4. `mla_reference.rs`: every assertion added by task 0053 except the
   `node.contract.partition` one. That covers `node.inputs.get(8)`,
   `graph.name(o_proj)`, the `role_weight()` role check, `node.output == graph.output()`,
   the no-downstream-`Linear` scan and the `o_proj` shape check. They assert
   the wiring this same fixture built. Then remove the `o_proj: ValueId`
   field that task 0053 added to `Fixture`, and its initialiser, since nothing
   reads them. Reduce the `descriptor` extraction to what the kept assertion
   needs.
5. `crates/moxie-graph/src/lib.rs`, `partition_semantics_fail_closed_until_defined`:
   the `HeadShardable` block task 0053 added. `is_partitionable` is
   `!matches!(self, NotDetermined)`, and the existing `RowShardable` line
   already proves a determined rule passes.

Remove imports that become unused. Change nothing else.

- **Allowed files:** exactly the three test locations above.
- **Non-goals:** no production change and no new test. Do not touch the
  routing-contract test or task 0054's `affine_linear` tests. Do not change
  the `Rope`/Q/K/V head-alignment question; that is M5.1-c.

## Contract before implementation

Equations, shapes, precision, partition, memory, cancellation and
application behaviour are all unchanged; this task edits tests only.
Oracle: the mutation check below.

## Acceptance

- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`,
  `cargo test --workspace --locked`, `cargo xtask arch-check`,
  `cargo xtask spec-check` and `git diff --check` pass.
- Mutation check, run and restored by the builder and recorded in Result.
  Temporarily change `OpParams::partition_rule()` in
  `crates/moxie-graph/src/graph.rs` in three ways, one at a time:
  - `Attention` → `NotDetermined`: the contract-table row must fail.
  - `MlaAttention`'s `output` → `ConcatenateHeads`: the contract-table row
    and the `mla_reference` node assertion must fail.
  - `Attention`'s `kv` → `SharedLatentReplicated`: the contract-table row
    must fail.
  Restore `graph.rs` and confirm `git diff crates/moxie-graph/src/graph.rs` is
  empty before reporting.
- Report the net line change for each file.
- **Stop condition:** if a kept assertion does not fail under its mutation,
  stop and report rather than adding a test.

## Result, filled after work

- Changed files; source commit: test-only, on base `50bfee2`.
  `crates/moxie-interp/tests/reference_graphs.rs` (-66),
  `crates/moxie-interp/tests/mla_reference.rs` (-39),
  `crates/moxie-graph/src/lib.rs` (-5); 110 lines deleted, none added.
- Commands: `cargo fmt --all -- --check`, `cargo clippy --workspace
  --all-targets --locked -- -D warnings`, `cargo test --workspace --locked`,
  `cargo xtask arch-check`, `cargo xtask spec-check`, `git diff --check`, all
  passed (builder). The reviewer re-ran `moxie-graph --lib` (19/19) and
  `moxie-interp` `reference_graphs` + `mla_reference` (33/33). None failed or
  were skipped.
- Mutation results: each was applied to `OpParams::partition_rule()` and then
  restored. `Attention` → `NotDetermined`: the contract-table row failed.
  `MlaAttention` output → `ConcatenateHeads`: the contract-table row and the
  `mla_reference` node assertion both failed. `Attention` kv →
  `SharedLatentReplicated`: the contract-table row failed. `git diff
  crates/moxie-graph/src/graph.rs` was empty afterwards; the reviewer and the
  coordinator both confirmed this.
- Review: sol, round 1, ACCEPT. The diff is exactly the contract's named
  deletions. Removing the direct `MlaAttention` pattern match loses no
  coverage: the fixture has one node, `attention_layers() == [0]` requires
  that node to be an attention op, the kept `SharedLatent`/`GlobalReduction`
  assertion distinguishes MLA, and the interpreter execution depends on
  MLA's nine-input semantics. No further redundancy was found. The reviewer's
  account hit its usage limit after the verdict was written and before the
  Herdr notification was sent; the verdict and evidence are in its pane
  (`wC:pY`), read by the coordinator on 2026-09-22.
- Remaining blockers: none.
