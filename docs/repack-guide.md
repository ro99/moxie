# Using `moxie-repack`

`moxie-repack` converts quantized checkpoints you already have into Moxie's
**canonical artifact** format, offline. This page is the user guide: how to run
it, what it refuses and why.

> ### Status: experimental
>
> **This step is provisional, and there is no performance evidence for it.**
> Moxie cannot yet execute a checkpoint, so nothing here has been shown to make
> inference faster — or slower. The artifacts it produces are not runnable by
> Moxie today.
>
> The reason the step exists is a hypothesis: that a single prepared layout
> helps the runtime. Whether that is worth a conversion step and a second copy
> of your model is decided by a measurement that
> [cannot be run yet](evidence/experiments/0007-offline-versus-load-time-preparation.md),
> and the step is retired if the answer is no
> ([ADR 0027](decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)).
>
> Use it to experiment with canonical artifacts. Do not adopt it expecting
> speed.

For why it exists at all, see
[ADR 0021](decisions/adr/0021-repack-is-a-moxie-program.md) (it is a program you
run, not a script someone pastes) and
[ADR 0022](decisions/adr/0022-user-programs-and-canonical-write-authority.md)
(where canonical write authority lives).

---

## What it does, in one paragraph

You write a small TOML file — a **selection** — naming exactly which tensors to
convert and where they come from. `moxie-repack` reads those tensors from a
checkpoint directory, converts them into Moxie's canonical layout, and publishes
a directory containing a `manifest.toml` and one or more
`model-NNNNN-of-NNNNN.safetensors` shards. It never scans, never expands a
wildcard, never guesses from a filename, and never writes inside the checkpoint
it read.

## What it is *not*

- **Not faster inference, as far as anyone knows.** No prefill or decode
  comparison exists. Byte-exact repacking, conformant packaging and compact
  plans are facts about bytes and structure, not about speed.

- **Not a model converter.** The output has no `config.json`, no tokenizer, no
  `model.safetensors.index.json`. Transformers cannot load it.
- **Not an execution path.** Nothing here runs a quantized tensor. That is M3
  item 3 and no kernel exists yet.
- **Not a quality claim.** A repack is bit-identical by definition
  ([ADR 0018](decisions/adr/0018-v1-quality-is-bit-identical-repack.md)): the
  values it publishes are the values the source encoded. It says nothing about
  how the model behaves.
- **Not a downloader.** It reads a directory that is already on your disk.

---

## Quick start — two commands

```console
$ moxie-repack plan --source-root /fast/models/ORG/MODEL --out-plan ./model.plan.toml
$ moxie-repack repack --plan ./model.plan.toml --out ./model-moxie
```

That is the whole normal path. No TOML to write, no packing parameters to look
up, no memory or disk numbers to guess.

### What `plan` does

It reads the checkpoint's own `config.json` and `model.safetensors.index.json`,
works out which tensors exist and how they are packed, and writes a plan you can
read. It converts nothing and **never writes inside the checkpoint**.

```console
$ moxie-repack plan --source-root /fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4
architecture: Qwen3_5ForConditionalGeneration
shards-read: 6
modules: 256
bf16-tensors: 943
selected: 1199
skipped: 0
selection-bytes: 96960
outcome: planned
selection: ./Qwen3.8-27B-AWQ-BF16-INT4.plan.toml
quantization: 4-bit, Group { size: 32 }, PackedAlongOutput

Read it, edit it if you want, then:
  moxie-repack repack --plan ./Qwen3.8-27B-AWQ-BF16-INT4.plan.toml --out <dir>
```

### A limit worth knowing before you start

**There is a ceiling on how many tensors one conversion can cover**, and the
largest checkpoints are above it. A conversion records every work unit in a
resume journal, a unit is cut inside one component of one tensor and never
across two, and the journal has a hard 16 MiB cap because a resume has to read
it back. So the smallest journal a conversion can write is one record per
component — around **28,000 components**, roughly 9,000 quantized modules.

`plan` checks this first and refuses with the arithmetic:

```console
$ moxie-repack plan --source-root /fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4
outcome: refused
moxie-repack: invalid artifact: this checkpoint's 36769 selected tensor(s) hold
106249 component(s), and a conversion writes at least one resume-journal record
for each: 61522267 byte(s), above the 16777216 byte cap a resume can read back.
```

No budget changes this. It is a limit of **this program's restart journal** —
one record per component, a 16 MiB cap on a journal a resume must read back, and
a work unit cut inside one component — and not a limit of safetensors, of the
canonical format, or of anything about inference. All three are changeable;
raising the cap is open work, not a setting.

**So the whole-model workflow is not delivered for the largest checkpoints.**
Converting one in parts needs a hand-written selection (below). This is recorded
as unfinished rather than worked around; see
[task 0027](tasks/0027-m3-generated-plans-and-automatic-budgets.md).

The plan records where the checkpoint is and which resource settings it chose,
so the second command needs neither again.

### What `repack` does

It converts in bounded work units, resumes if interrupted, validates the result
through the same reader the engine uses, and publishes with a single rename. It
prints the settings it used:

```console
$ moxie-repack repack --plan ./model.plan.toml --out ./model-moxie
settings: total=673239040 header=16777216 scratch=67108864 shard-max=4294967296 disk=33524528284
progress: hashed source model-00001-of-00006.safetensors: ...
progress: unit 1 of 'model.layers.0.mlp.down_proj.weight.codes': ...
outcome: published
```

### If it refuses

Refusals name what was found and what to do. The common ones:

| Message contains | What it means |
|---|---|
| `quant_method 'auto-round'` | This packing is not implemented yet. The plan is refused rather than partially guessed |
| `actorder "group"` / `"dynamic"`, or `weight_g_idx` | This source needs a column-to-group map that the contiguous importer cannot discard. `static` / `weight` ordering is supported because it preserves the saved contiguous grouping |
| `this plan is partial` | Some tensors are not covered. `--allow-partial` converts the covered subset **deliberately**; without it, a partial conversion is refused because it is not a model |
| `already exists` | A plan is not silently replaced. `--force`, or name another path |
| `bound to a different plan` | The destination belongs to a different run. Use a fresh `--out` |
| `changed while this run was reading it` | A source file moved mid-run; nothing is published |

### If it is interrupted

Run the same command again with `--take-over-interrupted-run`. The resume
re-reads and re-hashes every byte the journal claims before reusing it, and
refuses outright if the journal belongs to a different plan. The flag is
explicit because this program cannot tell a crashed run from one another process
is still working on.

## Advanced: writing a selection by hand

Everything below is **optional**. A hand-written version 1 selection is still
accepted, and every resource setting can be given explicitly. You need this only
to convert a subset deliberately, or to override what `plan` chose.

Precedence is explicit: **a command-line flag wins over the plan's recorded
setting, which wins over the automatic default.**

| Flag | Bounds |
|---|---|
| `--total-bytes` | All dynamic working memory, including parsed metadata |
| `--header-bytes` | Peak heap for one source header |
| `--scratch-bytes` | Payload scratch; the work-unit size, which drives how many journal records a run writes |
| `--chunk-file-bytes` | The largest output shard |
| `--disk-bytes` | Everything the destination holds at its peak, including two journals during compaction |

### The selection file, field by field

One document, no includes, no globs. Every field is required.

| Section | Field | Meaning |
|---|---|---|
| top | `version` | Selection schema version. Currently `1` |
| `[source]` | `model`, `revision`, `license` | What you are converting, recorded into the manifest as provenance |
| `[tokenizer]`, `[template]` | `name`, `version`, `digest` | Identity of the tokenizer and chat template this artifact is for. `"none"` / `"not-selected"` when the selection is not a whole model |
| `[architecture]` | `name`, `version`, `metadata` | An opaque tree. No shared crate interprets it; it travels with the artifact |
| `[provenance]` | `scale_convention`, `quantizer`, `calibration` | How the source was quantized, recorded rather than inferred |
| `[completeness]` | `status`, `missing` | `"complete"` or `"partial"`. A partial artifact opens for inspection and **refuses every tensor read** — a subset is not a loadable model |
| `[[tensor]]` | `role` | The name the tensor has in the published artifact |
| | `kind` | `"bf16"` or `"pack-quantized"` |
| | `name`, `file` | For `bf16`: the source tensor and the shard holding it |
| | `module`, `width`, `group`, `zero_points`, `[tensor.files]` | For `pack-quantized`: the module prefix, `"int4"`/`"int8"`, the group size, `"symmetric"` or `"packed-along-output"`, and which shard holds each of the four source tensors |

`role` and `name` are separate on purpose: the output name and the source name
need not match, and a module's four tensors need not share a shard — Qwen3.8-27B
splits every one of its 256 modules.

---

### The budgets

Five, all required, no defaults. How much of your machine a tool may spend is
not a question the tool should answer for you.

| Flag | Bounds | Choosing it |
|---|---|---|
| `--total-bytes` | All dynamic working memory, including parsed metadata | Must cover the tiles, the parsed selection, the plan and — on a resume — the journal records. Start at 512 MiB |
| `--header-bytes` | Peak heap for **one** source header | A large checkpoint's shard header can be tens of megabytes. 64 MiB is usually ample |
| `--scratch-bytes` | Payload scratch: one source tile plus one canonical tile | The work-unit size. Larger means fewer, bigger units and a much smaller journal |
| `--chunk-file-bytes` | The largest output shard | A component is never split across shards, so this must exceed your largest single component |
| `--disk-bytes` | Everything the destination holds at its peak | Payload, the staged manifest, and **two** journals — compaction writes a replacement beside the original before renaming over it |

Sizes take a plain byte count or a `KiB`/`MiB`/`GiB` suffix.

**If a budget is too small the run refuses and says which one and by how much.**
That is the intended way to find the right numbers — `inspect` reports the same
refusals without creating anything.

A note on `--scratch-bytes`: it is the strongest lever you have **for one large
tensor**. Every work unit writes one journal record, so a very small scratch on a
large tensor produces a very large journal, and a journal has its own hard cap
because a resume has to read it back.

It is not a lever on tensor **count**. A unit never spans two components, so a
checkpoint with more components than the cap allows is refused whatever the
scratch is — see the ceiling above.

---

### Outcomes and exit codes

| Code | `outcome:` | Meaning |
|---|---|---|
| 0 | `published` / `inspected` / `verified` | It did what you asked |
| 1 | — | The command line was wrong |
| 2 | `refused` / `failed` | Nothing was published. A repack leaves a **resumable** destination |
| 3 | `cancelled` | You stopped it before the publication boundary. Nothing was published; the destination resumes |
| 4 | `published-durability-unconfirmed` | The artifact **exists**, and this program could not confirm it survives a power cut |

Four is deliberately neither success nor failure. Deleting the artifact would be
deleting a possibly-published output; calling it success would claim a
durability nothing observed.

---

### Interrupting and resuming

A repack writes in bounded work units and records each one in a private journal
inside the destination. If it is interrupted — cancelled, crashed, killed — the
destination keeps three private files (`.moxie-repack-journal`,
`.moxie-repack-lock`, `.moxie-repack-manifest`) and no artifact.

To continue it:

```console
$ moxie-repack repack ... --take-over-interrupted-run
```

The flag is explicit because this program cannot tell an interrupted run from
one another process is still working on — that needs process liveness, which is
a different crate's job. Continuing is your decision to make.

A resume **re-reads and re-hashes** every byte the journal claims before reusing
it. A journal entry is never evidence its payload is correct. It also refuses
outright if the journal was written for a different plan, a different selection,
or different source bytes: a resume binds all three, and a difference is a
refusal rather than a merge.

---

### What is supported

| Source | Status |
|---|---|
| BF16 tensors, copied through unchanged | supported |
| compressed-tensors `pack-quantized` INT4 and INT8 | supported at group 32, group 128 and per-channel |
| Symmetric, or zero points packed along the output axis | both supported |
| Scales in F16, BF16 or F32 | supported; the source's own dtype is preserved |
| A module whose four tensors live in different shards | supported |
| AutoGPTQ / AutoRound packing | **not supported** |
| Activation-order (`g_idx`) permutations | **not supported.** A permutation is never ignored — it is refused |
| Discovering tensors not named in the selection | **not supported, by design** |

---

### When something is refused

Refusals name what they refused and why. The common ones:

| Message contains | What it means |
|---|---|
| `cannot hold this run` | `--total-bytes` is below the tiles plus metadata. The message states the minimum |
| `journal byte(s)` | The plan would write a journal larger than a resume can read back. If it names a *tensor*, raise `--scratch-bytes`; if it names a **component count**, no budget helps — the checkpoint is above the ceiling described above |
| `above the admitted disk budget` | `--disk-bytes` does not cover payload plus staging. The message states the total |
| `bound to a different plan` | This destination belongs to a different run. Use a fresh `--out`, or the selection it was started with |
| `changed while this run was reading it` | A source file moved under the run. Nothing is published, because the digest would describe bytes nobody has |
| `already holds a published manifest` | This slice never overwrites a published artifact |
| `is not a regular file` / `has N names` / `is a symbolic link` | A private file in the destination is not one this run created |
| `resolves outside` | A path escaped the source root or the destination |

---

### What a published artifact contains

```
laguna-canonical/
├── manifest.toml
└── model-00001-of-00001.safetensors
```

The shards are conforming safetensors files: the reference implementation opens
them, and the acceptance gate for this format is that it does
(`cargo xtask reference-check --artifact <dir>`). The schema — which physical
tensors a logical one becomes, their dtypes and shapes, and what is checksummed
— is fixed by
[ADR 0025](decisions/adr/0025-canonical-safetensors-schema.md):

| Logical tensor | Becomes |
|---|---|
| BF16 | one `BF16` tensor, named for the role |
| INT4 affine | `<role>.codes` `U8` `[out, ceil(in/2)]`, `<role>.scales` in the source's dtype, `<role>.zero_points` `I16` when asymmetric |
| INT8 affine | `<role>.codes` `I8` `[out, in]`, plus scales and zero points as above |

`manifest.toml` is the authority on what those tensors mean. The shards'
`__metadata__` is a courtesy for other tools and nothing here depends on it.
