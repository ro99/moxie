# ADR 0005 — Canonical manifest v1 is TOML; payloads are separate chunk files

- Date: 2026-09-08.
- Status: proposed with [task 0005](../../tasks/0005-m1-canonical-manifest-and-bounded-reads.md);
  not yet implemented.
- Authority: implementation decision. Document 03 fixes *what* manifest v1 must contain and that
  readers must validate it; it does not fix the encoding, so this records the choice and its cost.
- Amends: nothing. It adds a fourth `arch-check` allowlist entry (`toml`, `serde`) for
  `moxie-format`, which is the first third-party dependency in a production crate — see "Cost".

## Decision

A canonical artifact is a **directory**: one `manifest.toml`, plus one or more opaque chunk files
holding tensor payloads. The manifest is TOML, parsed with `toml` and `serde` — the same two crates
`xtask` already builds with.

Tensor bytes never appear in the manifest. The manifest holds, per tensor, the chunk it lives in,
its byte offset, its length, its SHA-256 and its alignment.

## Why a text manifest and a separate payload

Document 03 requires an "offline artifact inspection/conversion workflow" and states that
"conversion is an offline, restartable tool: inspect -> estimate disk/RAM/time -> obtain required
authorization -> convert bounded chunks -> validate -> atomically publish manifest." Three
properties follow, and a text manifest beside opaque chunks has all three:

- **Inspection costs the manifest, not the artifact.** Opening a 400 GB artifact to answer "what
  precision are the expert tensors" must not read 400 GB. A separate manifest makes that structural
  rather than a promise about lazy code.
- **Atomic publish is a rename.** Chunks are written, then validated, then a single small file is
  renamed into place. A single-file format with an embedded header makes the publish step a rewrite
  of the thing being published.
- **A human can review it, and so can `git diff`.** A malformed-manifest test fixture is a string in
  a test file. A reviewer can check that a rejection rule is testing what it claims without decoding
  anything. That property is why the fixtures in this repository are text wherever they can be.

A binary header would be smaller and marginally faster to parse. Neither matters: the manifest is
read once per artifact open and is kilobytes.

## Why TOML rather than JSON

`toml` is already in `Cargo.lock`, added for `xtask`, so this adds **zero** packages. TOML also has
comments, which a JSON manifest cannot carry, and an artifact whose provenance and exclusion
decisions cannot be annotated is one where those decisions live only in a commit message.

The counterargument — that JSON is what checkpoint ecosystems emit — does not apply. This is the
*canonical* manifest, written by our own converter. Foreign metadata (a HuggingFace `config.json`,
a safetensors header) is read by importers in M3, which will parse whatever their source emits and
normalize into this. Choosing this format to resemble a source format would be choosing it for the
one job it does not have.

## Cost, stated plainly

`moxie-format` gains two third-party dependencies, `toml` and `serde`. Until now every production
crate had zero and `arch-check` enforced an empty allowlist for them; ADR 0004's dependencies went
into `xtask`, the composition root, and did not cross this line. This does.

It is worth it only because the alternative is worse in a specific way: the alternative is a
hand-written parser for a subset of some text format, and ADR 0004 is a four-review record of what
hand-written parsers of structured text cost in this repository. The mitigations:

- The allowlist stays **per crate**, not global. `moxie-format` is allowed `toml` and `serde`; every
  other production crate keeps an empty allowlist, and `arch-check` must have a fixture proving a
  second crate taking `serde` is rejected.
- `moxie-format` remains I/O-free. `serde` deserializes from a `&str` that `moxie-storage` supplies.
- Deserialization is the *first* step, not the validation. Every rule in task 0005's table runs on
  the deserialized value, in our own code. Nothing is trusted because it parsed.
- `#[serde(deny_unknown_fields)]` throughout, so an unrecognized key is a rejection rather than a
  silent drop. Combined with `required_features`, an artifact written by a newer converter fails
  loudly in two independent ways.

## Limits

- **The architecture metadata is opaque here by construction.** It deserializes into an
  uninterpreted value tree. `moxie-format` hashes it into artifact identity and never reads a field
  of it. Interpreting it is a `moxie-models-*` job, and `arch-check` enforces that this crate cannot
  name a model family.
- **This decides the manifest, not the prepared layouts.** Document 03's "bounded, versioned
  prepared-layout caches" are a separate artifact with their own identity, and nothing here commits
  to how they are stored.
- **Chunk granularity is not decided.** One chunk per tensor and one chunk for the whole model are
  both valid under this ADR. The converter that has to estimate disk and resume from a partial write
  is the task that should decide it, with measurements.
