# ADR 0026 — A checkpoint plan is generated and compact, and budgets have defaults

> **Qualified by [ADR 0027](0027-repacking-is-provisional-pending-measured-inference-benefit.md) (2026-09-14).**
> The plan format and automatic budgets below stand. They make a **provisional**
> step usable; they are not evidence that the step should exist. The text below
> stands as written.


Status: **accepted** (owner direction, 2026-09-14, relayed through Herdr)
Supersedes: the "every budget is required on the command line" clause of
[ADR 0021](0021-repack-is-a-moxie-program.md) and the hand-authored selection
assumed by [ADR 0022](0022-user-programs-and-canonical-write-authority.md).

## The decision

1. A user points `moxie-repack plan` at a checkpoint and gets a **plan**: a
   compact TOML document, generated from the checkpoint's own declarations,
   which `moxie-repack repack --plan` consumes without hand editing.
2. The plan carries its **source binding** and its **resolved options**, so the
   second command needs neither the source path nor any budget.
3. Budgets acquire **automatic defaults**, derived from the plan. Explicit flags
   remain and override them, with stated precedence.
4. Planning never starts a conversion, and never writes inside the source root.

## Why the selection stays, and why nobody writes one

[ADR 0020](0020-user-managed-storage-and-canonical-materialization.md) requires
a conversion to be **explicit**: nothing discovered, expanded from a pattern or
inferred from a filename. That is a property of the artifact's provenance, and
it is worth keeping. It never followed that a person should type the file —
explicit and handwritten are different things, and conflating them made the
program usable only by its author.

Every value in a selection is already in the checkpoint: the packing parameters
in `quantization_config`, which modules are unquantized in its `ignore` list,
which shard holds which tensor in `model.safetensors.index.json`. A generated
plan is *more* explicit than a hand-written one, because it is complete.

## Why the representation changed, with the measurement

The v1 selection is one TOML stanza per tensor. For
`cyankiwi/Laguna-S-2.1-AWQ-INT4` — **34,740 quantized modules and 2,029 BF16
tensors, 140,989 index entries across 15 shards** — that document is
**15,210,830 bytes**, against a 4 MiB parse cap. The feature does not work in
that representation.

The first guess was that dropping the repeated per-module file fields would
shrink it roughly fourfold. **Measured, it does not.** Byte census of the module
stanzas (14,810,544 bytes total):

| Part | Bytes | Share |
|---|---:|---:|
| the four `weight_* = "shard"` lines | 7,156,440 | 48.3% |
| `role = "…"` | 1,968,462 | 13.3% |
| `module = "…"` | 1,794,762 | 12.1% |
| `zero_points = "…"` | 1,250,640 | 8.4% |
| `[[tensor]]` / `[tensor.files]` | 903,240 | 6.1% |
| `kind`, `width`, `group` | 1,737,000 | 11.7% |

Removing the file fields entirely leaves ~8.05 MB — still over the cap. The
bytes are dominated by **repeating what every module shares** and by repeating
each module's name twice. So the plan hoists what is common and states each
module once:

* a `shards` table, so a shard is named once and referenced by index;
* a `[weights]` block carrying `kind`, `width`, `group` and `zero_points` once;
* one entry per module: `["module.name", shard_index]`.

Measured on the same checkpoint, exactly **one** module of 34,740 has its four
tensors in different shards, so the common case is one index and the exception
is spelled out in a `[[weights.split]]` entry.

Projected size for Laguna: **~1.8 MB**, inside the cap with room. The cap still
rises — to 64 MiB — because `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` indexes
224,280 tensors and a format should admit the models it targets.

**A larger cap is not a bound.** What bounds the work is the *entry count*,
which a v2 plan states in its own header and which the expansion is admitted
against. Text length stopped being the proxy for peak memory when the document
stopped being one stanza per tensor.

## Why budgets default

Five mandatory numbers with no defaults was justified as "how much of a user's
machine a tool may spend is not a question the tool should answer". That reads
well and is wrong in practice: the user cannot answer it either, because the
right values follow from the plan — the largest shard header, the largest
component, the payload total and the unit count that keeps the journal readable.
The program knows all four; the user knows none of them.

So the defaults are **derived and reported**, not hidden. A plan records what it
resolved, the run prints it, and any flag overrides it. The admission, the
cancellation and the disk accounting the reviews established are unchanged:
what changes is who computes the numbers, not whether they are enforced.

## What planning may not do

* **It never converts.** `plan` writes one document and exits.
* **It never writes inside the source root.** The default output is the current
  directory, and an existing plan is not overwritten without `--force`; the
  write is atomic.
* **It never guesses a format.** A checkpoint whose declarations this repository
  has not measured is refused with what it declared and what is missing, not
  partially converted. See the coverage matrix in task 0027.
* **It never fabricates provenance.** Where a revision and per-file digests are
  recorded by the downloader they are carried; where they are absent that is
  stated as absent, under the contract in task 0027, and the source is bound by
  content instead.
