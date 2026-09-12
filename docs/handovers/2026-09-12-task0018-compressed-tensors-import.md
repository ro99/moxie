# Handover — task 0018, compressed-tensors pack-quantized import

## Workspace identity

- Writable repository: `/home/rodrigo/Developer/moxie`, branch `main`.
- Base `22f5733` (task 0017, corrected through two review rounds and pushed).
  Contract and ADR 0015 at `db9529e`, then the implementation commit this
  handover accompanies. The working tree was clean at the start.
- Read-only legacy `/home/rodrigo/Developer/strata` at
  `2dc566eb8e440fff4837ac75ca1dad1b20c2264e`; its untracked `.pi/` and
  `tests/p2p/` are untouched. Citations resolve against the frozen commit's
  paths, not against that checkout's `.pi/worktrees/` copies.
- `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit` was **opened and read**.
  Nothing under `/models` or `/fast/models` was created, modified, deleted,
  converted or downloaded. **O5 was not exercised** — see below.
- Raw logs are in the session scratchpad, not in git; the figures and commands
  are transcribed into
  [the task record](../tasks/0018-m3-compressed-tensors-int8-importer.md).

## Completed facts

The contract was authored and committed **before** implementation, and the
packing was verified against the artifact before the contract was authored —
document 03 requires inspecting "pinned config, tensor index and bounded tensor
headers before an importer claim".

**Independent review requested changes and found five defects plus a
documentation contradiction. All were reproduced before any change and all are
fixed**; the task record carries each one. The two P1s are worth carrying
forward: `import` took byte slices, so a transposed packed shape with the right
byte count imported silently wrong weights; and the importer's own reservations
were fallible while the `pack_row` it called was not, so a row-sized allocation
failure aborted the process. **Fallible allocation is a property of a whole
path, not of the function you happened to write.**

Three new owners: `moxie-format::safetensors` for the container,
`moxie-format::compressed_tensors` for the `pack-quantized` decode, and
`moxie-storage::Shard` for bounded positioned reads. The decode produces the
**existing** canonical `AffineTensor`; the reconstruction equation, the
descriptor and every prior expectation are unchanged. There is no second
decoder and no runtime decode path.

| Lane | Command | Result |
|---|---|---|
| Host workspace | `cargo test --workspace --locked --offline` | **689 + 9 doctests passed**, 0 failed, 0 ignored |
| Importer | `cargo test -p moxie-format --locked --offline` | **102 passed** |
| Import allocation | `cargo test -p moxie-format --test import_allocation --locked --offline -- --nocapture` | **1 passed**; 3 allocations at 4, 64 and 512 rows, against 7/67/515 for its control |
| Real artifact | `cargo test -p moxie-storage --test gemma4_import --locked --offline -- --nocapture` | **3 passed** |
| Host clippy | `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` | passed |
| Format / diff | `cargo fmt --all -- --check`; `git diff --check` | passed |
| Specification | `cargo xtask spec-check` | passed, 10 digests unchanged |
| Architecture | `cargo xtask arch-check` | **74 rejecting + 21 accepted**, 12 rules |
| Device workspace | the host command with `--features moxie-cuda/driver,moxie-kernels/fatbin,moxie-executor/driver,xtask/cuda` | **705 + 12 doctests passed**, 0 failed |
| Header cost | `cargo test -p moxie-storage --test header_budget --locked --offline -- --nocapture` | **1 passed**; measured peak heap within the admitted estimate at five header shapes |
| Device clippy | the clippy command with the same features | passed |
| Real GPU | `cargo xtask-cuda test-gpu` | **39/39**, 0 failed, 0 skipped; sm_86 and sm_120 qualified |

`arch-check` against the working tree also reports the four pre-existing
findings from the retained task 0014 probe crate under `results/`, as tasks 0015
through 0017 recorded.

The GPU result is unchanged, which is expected: **no device behaviour is added.**

**Real-artifact evidence, re-derived rather than restated.** Seven shards parse,
each covering its payload exactly; 2008 tensors — 1188 `BF16`, 410 `I32`, 410
`I64`; 35,089,877,112 B of payload. Three modules import to canonical form
(`q_proj` `(8192, 5376)`, `o_proj` `(5376, 8192)`, `down_proj`
`(5376, 21504)`), reading 216,416,304 B, with codes spanning the full
`[-128, 127]` and every sampled reconstructed value finite. Those tests **skip
with a message** when the artifact is absent, so a fresh clone stays green.

Support matrix gains `G-CT-IMPORT` and a row that says **import**, with
execution still **not implemented**.

## Decisions

- [ADR 0015](../decisions/adr/0015-serde-json-for-safetensors-headers.md):
  `moxie-format` takes `serde_json` for the header parse, and **only** for the
  parse — every structural rule, bound and arithmetic check stays in this
  repository. Hand-rolling a JSON parser for untrusted checkpoint input was
  rejected; so was the `safetensors` crate, on ownership grounds rather than
  quality, because document 03's validation rules are this repository's to make
  and test.
- **Asymmetric pack-quantized is a typed `Unsupported`, not a guess.** The
  pinned reader rejects it, no local artifact is one, and document 03 forbids
  inferring a zero offset "from a suffix". The canonical descriptor already
  carries `ZeroPoints::PerGroup`: what is missing is a verified *source*
  contract, not a canonical capability.
- No owner gate was resolved. O1–O7 remain open. **O5 was respected as the
  contract stated it**: the work reads bytes under the designated inspection
  root and writes nothing, anywhere. The moment it would need to persist a
  canonical tensor, manifest or prepared layout, it stops — that is the bulk
  write O5 governs and it is not authorized.
- No numerical threshold, precision, context target or compatibility surface
  changed. There is no tolerance here to declare: the importer performs no
  arithmetic on scales and only a rebias on codes, so the criterion is exact
  equality against an independent oracle.

## Remaining hypotheses and blockers

**M3 is not closed, and an import is not an execution.** Still outstanding, all
M3's: the shared W8A16 and W4A16 paths, the bounded inspector/repacker with
restartability and atomic publish, the canonical manifest write, and the pinned
AutoRound/AutoGPTQ packing adapter. **M1.5 is not closed**: the Gemma 4 artifact
still cannot run, and vision is M11.

**The declared risk this task could not close, stated so it is not forgotten.**
All `32 / bits` lanes of a packed word fall inside one scale group, so the
artifact's own data **cannot** distinguish the lane order within a word: a
reversed order reconstructs a different but equally plausible weight, and every
test here would still pass. The order is taken from the pinned reader
(`compressed_tensors.cpp:286`) and is not inferred. Closing it needs paired
output against the released model, which is **O2** evidence and arrives with the
first execution path. **Do not let a successful import become a quality claim.**

What was checked and is not a risk: the code bias. A byte histogram of one real
packed row is dense in `64..191`, mean raw byte **127.4**, so the codes are
biased-unsigned and `raw - 128` is right; two's-complement bytes would have been
edge-heavy. The full `[-128, 127]` range appears in real data, including `-128`,
which document 03 forbids a decoder from rejecting.

Worth knowing, from two rounds of review corrections on the same finding.
`ByteBudget` caps the reader's *payload slicing*, and a header is a different
resource -- read whole because it must be parsed whole, and partly retained for
the shard's life. Giving it its own `HeaderBudget` was the first fix and was not
enough: that budget capped the **serialized length**, while parsing costs
several times that in entry lists, key strings, shape vectors, temporary spans
and the retained maps. A 54,899-byte bound admitted a 353,105-byte peak.

**A budget on an input is not a budget on what that input costs.** But
denominating it in peak heap was only half the lesson: a third round defeated
that bound too, with many tiny `__metadata__` entries and one 65,537-dimension
tensor. **A factor fitted to sampled header shapes is not a bound.** Any shape
you did not sample can beat it, and twice it did.

The bound is derived now. Peak is linear in how a header splits its bytes among
constructs, so `peak <= serialized x max(per-construct ratio)`, and the work is
enumerating constructs rather than sampling shapes. Measured marginals: tensor
entry 5.4, metadata entry 9.5, shape dimension 12.0, name byte 2.0. The shape
dimension is both the largest and **unbounded per tensor**, which is why no
multiplier survived it -- it gets `MAX_RANK = 8`, refused while parsing, which
brings it to 2.8. The factor is then 16 against a largest remaining ratio of
9.5. The regression measures **each construct at its own worst shape**, and its
two controls -- removing the rank limit, and dropping the factor to 9 -- each
fail.

**A derivation is only as good as the grammar it assumes.** A fourth round broke
this one twice more, and both are worth carrying forward:

- serde's **derived** `Deserialize` for a struct accepts a positional array as
  well as an object, and `deny_unknown_fields` does not stop it. A tensor entry
  written `["U8",[0],[0,0]]` cost 22.9 serialized bytes against the object
  form's 55, so the minimum-cost term was wrong by half. Hand-write the visitor
  and use `deserialize_map`, not `deserialize_struct`, when the parse cost is
  load-bearing.
- Marginal costs measured two points apart **average over collection-capacity
  boundaries**. A `Vec` that has just doubled holds its old allocation beside
  the new one, so the worst per-entry peak is just past a power of two. Measure
  at `2^k + 1`, which is where the review's counterexamples sat.

And: a limit checked after `next_value` has built the collection is not a limit.
`MAX_METADATA_ENTRIES` was enforced only after a 100,000-entry map had been
allocated in full. Count inside the visitor.

And: deserializing untrusted JSON into a map collapses duplicate keys before any
validation runs, so a header could declare an unsupported dtype and overwrite it
with a supported one. Parse straight into the target struct and keep the
top-level entries in declaration order.

Also: `arch-check` **rejected** the `serde_json` edge before it was
declared, which is the allowlist working rather than an obstacle. A new
`shared-takes-serde-json` fixture keeps a second crate from taking it, so the
header contract keeps one owner.

## Next task

Author task 0019 as the shared **W8A16 execution path**, which is what turns a
canonical INT8 tensor into a result and is the last thing between this engine
and the inventoried artifact's text tower.

- Owning component `moxie-kernels` for the kernel and `moxie-executor` for
  dispatch, with `moxie-format`'s `AffineTensor` as the input and
  `moxie-graph`'s `Linear` as the consumer. **No model crate changes.**
- Required reading: document 03's "Execution profiles and Ampere" — these are
  **weight-only** paths, "not INT4xINT4 or INT8xINT8 MMA" — document 07's
  qualification rules, the accepted task 0012 device chain, and Strata's
  offset-packed INT8 path as a numerical and streaming reference.
- Oracle: `AffineTensor::reconstruct` followed by the accepted BF16 linear.
  Document 03 is explicit that "bounded reference dequantization is not an
  acceptable final fast path by assertion" — it is the **correctness oracle**,
  and a kernel is qualified against it on the declared hardware and shape
  matrix, with the error bound declared before the kernel is written.
- Gates: real GPU on both architectures, the declared shape matrix, and
  bounded prepared-layout accounting. Charge scale metadata as well as weight
  bits, per document 03.
- **Stop conditions.** Stop at: widening streamed weights to BF16 on the host or
  PCIe path to make a kernel convenient; a prepared layout that retains both the
  canonical and expanded copies; a speed claim without a paired measurement; a
  quality claim without O2; a model-owned or method-specific execution path; or
  any bulk write, which remains O5's. **An import that runs is still not model
  support until quality is measured against the released model.**
