# ADR 0024 — The canonical writer is a module of the program that writes, not a crate

- ID / date / author / status: 0024 / 2026-09-13 / repository owner, implemented by the engineering agent / adopted; implemented in the same change.
- Classification: **owner ruling.** The owner rejected the crate split on review of the implementation.
- Scope and owning shared component: `moxie-repack` (gains a `write` module), `moxie-storage` (gains two public bounded-read functions), `xtask` architecture rules.
- Supersedes / superseded by: **amends [ADR 0022](0022-user-programs-and-canonical-write-authority.md)**, which created `moxie-storage-write` as a separate crate. Everything else in ADR 0022 — the program inventory, the read/write boundary, `moxie-format` staying I/O-free, packaging deferred to M11 — stands unchanged.

## Problem

ADR 0022 put canonical write I/O in its own crate so that "who may write a
checkpoint" would be a dependency edge a machine could check. Task 0025
implemented that. On review the owner rejected it, and the implementation
itself had already produced the evidence:

- **One consumer, by design.** `moxie-storage-write` was used by `moxie-repack`
  and nothing else, and ADR 0022 says nothing else may ever write. A crate whose
  consumer set is one program, permanently, is a module.
- **The boundary forced duplication.** The writer must read — it rehashes staged
  units on resume and validates through the production reader — and
  `moxie-storage`'s bounded-read helpers are private. So the writer grew its own
  `read_exact_at`, byte-range pump and capped text read. The first was
  character-identical to `OpenChunk::read_at`. Two copies of "read bytes at an
  offset, bounded" in one workspace is the duplication this repository exists to
  refuse, and the crate split is what created it.
- **The stated reason argued against a *feature*, not for a *crate*.** ADR 0022's
  argument was that a `write` feature on `moxie-storage` would leak through Cargo
  feature unification. True, and irrelevant to the alternative actually
  available.

## Decision

The writer is `moxie-repack::write`, a module of the one program that publishes.

- `moxie-storage` keeps its job and gains two public functions that were already
  its job: `read_range` (one exact byte range of a file, in scratch-sized
  slices) and `read_text_capped` (a whole small file, refused above a cap before
  it is read). The writer calls those. The duplicate `pread` is deleted.
- **Confinement is now structural rather than enforced.** No crate outside
  `moxie-repack` can name the writer, because it is not in their dependency
  trees. That is stronger than the rule it replaces and costs nothing to
  maintain.
- The `arch-check` rule `canonical write authority outside the offline repacker`
  is **removed**, along with its six negative fixtures and two accepted ones.
  They policed a dependency edge that no longer exists. `arch-check` returns to
  13 rules, 79 rejected and 21 accepted fixtures.
- No process that only reads a checkpoint links publication code any more.

## Why not fold it into `moxie-storage` instead

That was the other candidate, and it is the better answer *if* a second writer
ever appears — a service that publishes, or an ops tool that materializes
artifacts. Nothing in the roadmap does. Until something does, the writer belongs
with its only consumer, and moving it later is a file move plus one arch rule.

## Enforcement and removal

- `moxie-storage-write` is deleted, not deprecated. No compatibility shim.
- The mutation battery's three write-authority substitutions are removed with
  the rule: there is no longer a rule to weaken, and the module boundary cannot
  be weakened by an edit to one file.
- **The general rule this establishes**, and the reason it is written down: a
  new crate needs a shared consumer set *and* something distinct enough to own.
  One consumer plus a rule someone has to maintain is a module.
