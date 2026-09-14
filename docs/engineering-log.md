# Engineering log — lessons carried forward

**This is a log, not policy.** Nothing here grants permission or overrides a
contract. [AGENTS.md](../AGENTS.md) is the policy entry point, the
[reference documents](spec/) are normative, and the authoritative account of any
task is its own record in [tasks/](tasks/), [handovers/](handovers/) or
[evidence/experiments/](evidence/experiments/). What this file holds is the
distilled version: what went wrong, what the fix was, and the sentence worth
remembering the next time the same shape appears.

It accumulated inside AGENTS.md's "Active assignment" section through M1 and M2
until that section was 692 of the file's 735 lines, and an entry point that is
94% narrative stops being read as instructions. Moved here on 2026-09-13,
verbatim, so nothing hard-won was lost in the move. Prune it like any other
record: an entry whose lesson has been absorbed into a check, a rule or a
reference document has done its job and can go.

## How to use it

Read it once when starting on this repository, and again when a review finds
something in your work — most findings here are a shape that has appeared
before. Do **not** read it as a checklist before every task; the gates that must
run are in AGENTS.md and the task template.

## The recurring shapes

Each links into the entries below.

| Shape | Where it keeps appearing |
|---|---|
| An allocation that aborts instead of returning a typed error | tasks 0019, 0021, 0022, 0023, 0024 — six times, each on a path added after the previous fix |
| A gate that never sees the input it exists for | task 0021's descriptor, task 0023's valid-only sweep, task 0024's import-only sweep, task 0025's never-visited boundary |
| A recovery that only knows the states it imagined | task 0025's empty journal and its own leftover lock file |
| An undo that reverses the decision but not the state it already changed | task 0025's running checksum |
| A boundary invented before it had two sides | task 0025's `moxie-storage-write`, which forced a second copy of the reader's `pread` |
| A format assumption nobody asked the format about | task 0026's eight-byte alignment, refused by the reference reader |
| A new boundary that hides the boundary beside it | task 0026's shard-header pass, which absorbed every payload fault |
| A test measured by assertion rather than by mutation | tasks 0020, 0021, 0022, 0023, 0024 |
| A record contradicting a fact it already contains | task 0022's expert inventory, task 0022's VRAM figure, task 0024's sign claim |
| A comparison whose two sides are not independent | task 0022's FP64 transcription, task 0024's FP32-versus-BF16 arithmetic |
| A fix that is not guarded by the test written for it | task 0023's fake axis, task 0024's helper-only regression |

## Entries, newest first


**A new durable boundary can make an old one untestable (2026-09-14).** The
safetensors writer added a pass that writes and syncs every shard's header
before the first payload unit. It reused the boundaries already there --
`chunk-create`, `chunk-write`, `chunk-sync` -- because they are the same three
syscalls. The mutation battery said what that cost: `chunk-sync-skipped` and
`write-errors-are-swallowed`, both caught before, now **survived**. Every fault
injected for a payload unit fired at a header first, so the tests kept failing
for a reason that had nothing to do with the code under test.

Nothing about the product was wrong. The *measurement* was, and it had been
quietly wrong since the header pass was added. The fix is naming:
`shard-create`, `shard-header-write` and `shard-header-sync` are their own
boundaries, and `write_at` takes the boundary it is writing at as a parameter,
with the reason in a comment beside it.

The same run found a third: `cancellation-not-checked-before-publish` survived
because an earlier correction had added a cancellation check *inside* the
validation loop, and the existing test's "cancel from the second question
onwards" closure now stopped there and never reached the final gate before the
rename. A test that cancels at the last boundary -- and asserts how many
boundaries there are -- is what makes that gate observable again.

**The lesson: a boundary is only tested while it is the only thing that can
fail there.** Two boundaries sharing one name is not a naming preference, it is
a test that measures whichever one comes first. The battery is what turned an
invisible regression in the gates into three lines of output.

**The container changed, and the reference implementation decided the schema
(2026-09-14).** The owner replaced the custom `.mox` container with safetensors
shards plus a TOML manifest, and asked that the schema be fixed before the code.
[ADR 0025](decisions/adr/0025-canonical-safetensors-schema.md) fixed it:
component naming, physical shapes, shard mapping, checksum scope, migration.

**The first version of that schema was wrong, and measuring found it in one
minute.** It aligned each component's payload to eight bytes, for the benefit of
a future mapping consumer. Before writing a single shard, a four-line script
asked the reference implementation what it thought:

```text
gap=False: LOADED
gap=True:  SafetensorError: invalid offset for tensor `w.scales`
```

Every aligned shard would have been refused by the one reader the owner's ruling
names. Reading the reference source afterwards confirmed the rule --
`Metadata::validate` requires `s != start` to fail -- plus two more the writer
already satisfied: declared length equals shape times dtype size, and the file
must end exactly at the last tensor. `validate()` sorts by offset first, so
header key order is free.

The lesson is not "read the spec". It is that **a format decision's evidence has
to come from the format's own implementation, and it is available before the
code that depends on it.** The cost of asking was a scratch file; the cost of
not asking would have been a schema, a writer, a reader and a task's worth of
tests built on a file nothing else can open.

That is also why the acceptance gate for this work is
[a script that opens every published shard with the reference reader](evidence/experiments/drivers/0007-reference-reader.py)
and compares each component's dtype, shape and checksum against the manifest,
rather than our reader agreeing with our writer.

**Task 0025 — the offline repacker — is implemented and not reviewed
(2026-09-13).** `moxie-repack` inspects a selection, converts it in bounded work
units, resumes an interrupted run, validates through the production reader and
publishes a manifest-v1 directory with one rename; its `write` module is the
write authority (ADR 0022 put that in its own crate, ADR 0024 folded it back --
see below), and
[ADR 0023](decisions/adr/0023-canonical-affine-payload-and-repack-journal.md)
fixes the two encodings it needed — the canonical affine payload and the restart
journal — **before** the code that writes them. One Laguna module round-trips
with all **3,145,728** values checked against document 03's canonical FP32
equation and against the source's own BF16 boundary separately. **Nothing
executes a canonical INT4 tensor**; that is M3 item 3 and it is untouched.

Four lessons, and three of them came from the same test.

**The enumeration is the test.** Every durable boundary in the publication state
machine is *named*, a clean run measures how many times each is visited, and the
gate fails every visit to every boundary in turn — 35 cases — checking the same
three invariants each time: no readable artifact before the publication rename, a
resumable destination after the failure, and a restarted run publishing
**byte-identical** output with the same artifact identity. Writing it found three
defects that no amount of reading the code had:

1. **An undo that reversed the decision and not the state.** A resumed unit
   whose staged bytes fail their checksum is discarded and recomputed — and the
   first version had already folded those bytes into the *tensor's* running hash
   before comparing. The tensor then published a checksum computed over bytes
   that had been overwritten, and every later check agreed with it, because they
   all came from the same wrong hash. The fix is to clone the running hash,
   commit it only on acceptance, and it is two lines. Finding it needed a test
   that corrupted a staged byte and then demanded the *published* artifact match
   an uninterrupted run's, byte for byte.
2. **A recovery that only knew the states it had imagined.** The journal file
   existing was treated as proof that a run existed. A crash between creating it
   and writing its plan line therefore left a destination that could never be
   used again: no plan to bind a resume to, and a file whose presence refused a
   fresh start. A journal with no plan line records no run, and starting over is
   the only reading that does not strand the directory.
3. **The same shape once more, one layer out.** "Refuse a directory this run did
   not create" was checked by asking whether the destination was empty — and a
   run that crashed after taking its lock leaves exactly one file: its own lock.
   The check now names what it found, and a directory holding only this
   program's own private files is one of its own interrupted runs.

**A boundary a clean run never reaches is a boundary the enumeration does not
cover, and only measured coverage says so.** `chunk-read-back` is visited only on
a resume, so the enumeration derived from a clean run had zero cases for it. The
gate prints its coverage table and asserts that at most one boundary is
unreached, which is how that gap announced itself; the missing case is now a test
of its own. This is task 0021's descriptor and task 0023's valid-only sweep
again: the sweep covers the paths the scenario that generated it happened to take.

**The regression for a fix is what finds the fix incomplete (2026-09-13).** An
independent review of task 0025 made ten findings, five P1: a symlink escape
that wrote through a private file before validation rejected it, a cross-shard
resolver applying none of the importer's dtype and symmetric/asymmetric checks,
a source that could change after its digest was taken, a torn journal that was
parsed around but never repaired, metadata allocating outside the ledger, a disk
budget covering payload only, reservations leaking on every early return, a
panic on a scratch too small for one zero-point word, minutes of reading with no
cancellation check, and directory entries synced without their parents. All ten
reproduced and fixed.

The part worth remembering: **writing the regression for finding 4 found that
the fix for finding 4 was wrong.** The tear was truncated and synced, and the
run's own journal handle was still positioned at the length the file had when it
was opened -- so the next record landed past the new end, behind a hole of NUL
bytes, and the journal was unparseable in a new way. The fix looked right, the
suite was green, and a test that tore the journal *twice* took four lines to
disprove it. This is task 0024's second-round shape exactly: fixing a finding
and guarding the fix are two jobs, and the second one is where the first one's
mistakes are.

**The crate that should have been a module.** The writer went into
`moxie-storage-write`, its own crate, because ADR 0022 said a dependency edge
would make "who may publish a checkpoint" machine-checkable. The owner rejected
it on sight and the implementation had already proved him right: one consumer by
design, and a boundary that forced the writer to grow its own `read_exact_at` —
character-identical to the reader's private one — plus a byte-range pump and a
capped text read. The fix ([ADR 0024](decisions/adr/0024-one-storage-crate-and-a-write-module.md))
is a module inside the one program that writes, two primitives made public on
`moxie-storage`, and the deletion of a rule, six negative fixtures and two
accepted ones. **Confinement got stronger by deleting the thing that enforced
it**: nothing outside the program can name a module that is inside it. The
lesson for the next crate: a crate needs several consumers *and* something
distinct to own. One consumer plus a rule someone maintains is a module, and
inventing the crate first is what makes the duplication necessary.

**A fixture rejected for an unrelated rule is not evidence the rule under test
works — and the way to show it is to remove the *other* protection.** The
`arch-check` rule written for the write-authority crate edge had six negative
fixtures, five of which the dependency allowlist would also have refused, so the
fixtures alone said nothing about the new rule. The battery therefore carried an
**independence control**: a substitution that widened the allowlist and had to
leave the battery green, because the new rule still rejected those fixtures. It
held. The rule is gone now with the crate it policed, and the technique is the
part worth keeping: when two protections overlap, remove the other one and see
whether the gate still fires.

**A gate nobody wants to run is a gate that stops running.** The manifest cap
fixture parses a document with 1,048,577 tables in it, and took 99 seconds; the
real-artifact lane hashed one 5.37 GB shard at 12 MB/s and took seven and a half
minutes. Both were unoptimized *dependencies* and an unoptimized codec in a dev
build. `[profile.dev.package."*"]` and `[profile.dev.package.moxie-format]` at
`opt-level = 2` cut them to 15 seconds and 30 seconds. Nothing about what runs
changed — the artifact identity is the same either way — and that is the point:
this was never a correctness trade, it was a gate that had quietly become
expensive enough to skip.

**Repack user-surface ruling (2026-09-13):** [ADR 0022](decisions/adr/0022-user-programs-and-canonical-write-authority.md) resolves the engineer-lead placement gap: `moxie-repack` is the offline program; canonical write I/O is isolated in `moxie-storage-write`. M3 retains manifest-v1 directories; final single-file `.mox` packaging is assigned to M11 item 4. [Task 0025](tasks/0025-m3-offline-repack-publication.md) is the proposed first publication contract, not an implementation or acceptance. Its write-authority architecture rule is pending implementation.

**Task 0025 is the next task**, and it comes before M3 item 3 for three reasons
rather than one: repack is roadmap item **1** while the importers tasks 0018 and
0024 delivered are item 2; ADR 0021 assigns repack to item 1 explicitly and
states that M3's exit clause — "lossless (repack) claims have source-oracle
evidence" — **cannot close without the program that publishes what the evidence
is about**; and nothing persists what the importers produce, so a kernel built
first would have nothing to execute from but a test fixture. **M3 item 3, the
W4A16/W8A16 paths, is what follows** and is still the largest gap in the
milestone. The task 0024 handover's own "next task" section named item 3 when it
was written, before this ruling existed; it is corrected in place.

**M1 complete (M1.5 closed 2026-09-12); M2's five items are all accepted — items
1 and 2 on 2026-09-12, items 3, 4 and 5 on 2026-09-13 — and the owner authorized
M3 on 2026-09-13; **M3's first contract, task 0024, is accepted by the owner the
same day** after three rounds of independent review.** M2's formal
closure statement is still the owner's to make and is not claimed here. See the
[task 0024 handover](handovers/2026-09-13-task0024-asymmetric-int4-import.md)
for the current continuation, the
[task 0023 handover](handovers/2026-09-13-task0023-whole-working-set-trace.md)
for M2's last task, and the
[closure handover](handovers/2026-09-12-m1-closure-to-m2.md), which carries
M1's exit evidence gate by gate.

**M3 — canonical INT4/INT8/BF16 with quality separation — is authorized (owner,
2026-09-13)**, and both of its owner gates are already answered, which bounds the
work rather than opening it. **O2** is repack-only for v1 (ADR 0018): a lossless
claim needs source-oracle evidence and **Moxie never quantizes**. **O5** is
user-managed storage and conversion (ADRs 0020–0021): Moxie reads the canonical file,
repacking is a Moxie program the user runs offline — not an external script — and **no agent-initiated bulk
download, copy or conversion may start** without a task naming artifact,
revision, expected size and retention. `/models` and `/fast/models` remain
read-only inputs to an agent. The roadmap's M3 items are in
[06-implementation-roadmap.md](spec/06-implementation-roadmap.md).

**M3's first task is accepted: task 0024, M3 item 2's asymmetric half**
(owner, 2026-09-13, after three rounds of independent review — eight findings,
two P1, all reproduced, all fixed, none disputed; the third recommended
acceptance "within task 0024's declared importer-only M3 item 2 scope"). **The
acceptance closes task 0024 only, not M3 item 2**, which also wants group-128
symmetric INT4 and the AutoRound/AutoGPTQ packing ([task 0024](tasks/0024-m3-asymmetric-int4-pack-quantized-import.md),
2026-09-13, contract committed at `e122de3` before implementation). The importer
reads compressed-tensors `pack-quantized` **asymmetric INT4 at group 32**, whose
`weight_zero_point` is packed along the **output** axis while the codes are
packed along the input axis: **two packing conventions inside one tensor group,
distinguished by nothing in either name.** That is R16 in its exact form, and it
is why the shapes are validated rather than derived from byte counts. Six
modules of `/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4` and
`/fast/models/cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4` import, and **112,640
reconstructed values are checked against two quantities that are not the same
one**: document 03's canonical FP32 equation over the source's own bytes,
bitwise; and the source's own arithmetic **including the BF16 rounding its
reference applies**, which the canonical value rounds to exactly. The second
comparison is the review's, not this task's first instinct — see below. **Import is not
execution**: nothing consumes a canonical INT4 tensor, W4A16 is M3 item 3, and
Laguna's graph still declares BF16 tensor requirements — listing INT4 would
advertise a path that is not there. A bit-identical repack is **ADR 0018's v1
quality definition, not evidence about model output.** Mutation-measured 16 of
16, 0 survivors, first measurement.

**What a symmetric source could never establish, and this one can.** Task 0018
recorded that a *code* word's lane order cannot be checked against an artifact,
because all its lanes fall inside one scale group. A **zero-point** word's lanes
are different **output channels**, whose codes have different statistics, so the
assignment is a measurable property of the file. It was measured — `mean |mean
code − z|` of **0.52** for the pinned reading against **1.48–1.78** for every
alternative the same shape permits, on four tensors across two artifacts, with
the thresholds written into the contract beforehand
([experiment 0005](evidence/experiments/0005-asymmetric-int4-zero-point-assignment.md)).
That mattered rather than being a flourish: both artifacts declare compressor
versions (`0.1.dev534+gb269f2e`, `0.1.dev535+gdc9611a`) that are untagged
development builds and pin nothing, and this repository already refused one
mismatched `transformers` copy as a pinned exporter for yarn. **When a
convention cannot be checked, say so and pin it; when it can, measure it.** The
code word's lane order is still unchecked and stays where task 0018 left it.

**One independent review, five findings, one P1, none disputed — and the P1 is
this workspace's most repeated defect for the sixth time.** Every refusal in
`moxie-format` built its prose with `format!`, so refusing one allocation while
the importer rejected a malformed artifact was `SIGABRT`. **The sweep written to
prevent exactly this swept only imports that succeed, and a refusal has no
allocation positions in it at all.** That is task 0023's own third-round
sentence — "a gate that exercises the happy path under the adverse condition has
tested the adverse condition on the happy path" — reproduced by someone who had
read it that morning. Reading a lesson is not the same as applying it to the
gate you are writing; **ask of every new gate which of its axes the adverse
condition is actually on.** `Error::InvalidArtifact` carries a
`Cow<'static, str>` now so the variant can be built without allocating, and
`moxie-format`'s refusals compose their prose into a `try_reserve`d buffer with a
borrowed fallback. The other three string-carrying variants still allocate, and
that is named as an open question rather than quietly widened into this task.

**Two quantities, one name: the BF16 boundary again, one task after the task
that recorded it.** The pinned compressor's `_dequantize` casts to `scale.dtype`
before subtracting and multiplying, and that dtype is BF16 for these artifacts;
the test computed FP32 and every record called it "the source's own declared
arithmetic". **27,501 of 112,640 sampled values differ**, and neither number is
wrong — document 03 fixes the canonical reconstruction at FP32. What was wrong
was the name. There are two comparisons now, and the count of values the
boundary moves is **asserted nonzero**, because a boundary check on a sample
where the boundary never fires is a check of nothing.

**A filter applied to the population being audited removes exactly the rows the
audit exists to see.** The inventory dropped every module missing one of its four
tensors and then checked that every module had all four; a review built a shard
with a module missing its zero point and the test passed while reporting one
fewer module. Task 0023's omitted layer, in a different file and a different
year's worth of confidence. **Audit the population before you filter it, and
make the filtered subset a separately named thing.**

**The mutation names are not the measurement; the exact substitutions are.**
Experiment 0005 said its driver was "reproduced in the task record" and it was
not, in either commit. The driver is tracked now, at
[`docs/evidence/experiments/drivers/0005-mutations.py`](evidence/experiments/drivers/0005-mutations.py),
and every verdict is repeated three times in both directions because the
contract promised that and the first run did not do it. **21 of 21 caught, 0
survivors** when measured, **24 of 24** after a second review — where the first
battery was 16 of 16 and complete against a suite with **five** holes in it.
**Eight mutations were added after a review, and every one of them is caught by
exactly one lane**; seven of the eight by a check a finding added, the eighth by
a pre-existing test, because an existing lane acquiring a new regression is not
a lane a review created. A battery that is complete against the suite it was
written for says nothing about the suite's holes, and only a finding from
outside can add the case.

**An impossibility claim is a claim, and this one contradicted its own
document.** The record said the statistic could not separate a zero point that is
subtracted from one that is added, because a distribution centred near zero is
symmetric. That holds only for independent quantities, and the correlation
between a group's code mean and its own zero point is the premise of the
measurement **two paragraphs earlier**. Measured: **0.52 against 2.18**. Third
instance of a record contradicting a fact it already contains, and the first of
the three that was an argument rather than arithmetic — which is easier to wave
through, not harder.

**A second review, three findings, one P1 — and all three are first-round
corrections that were narrower than they looked.** This is the sentence to carry
out of task 0024: **fixing a finding and guarding the fix are two jobs**, and a
round that does only the first buys another round.

**A sweep that names one function has tested one function.** The first review's
P1 was refusals that abort; the fix made the refusal prose fallible and the
sweep that proved it calls **only `import`**. `source_entries` — the same
module, the same public surface — still built four lookup names with `format!`,
and eight of its eleven allocation positions aborted, two of them on the
zero-point name this task itself added. Task 0023's "reserving a destination
says nothing about a temporary the callee builds", one layer further out. When a
class of defect is found, ask **which entry points reach that class**, not which
line was named.

**A correct helper cannot detect a row that was removed before it was called.**
The regression written for the inventory's premature filter tested
`missing_companions` directly — and the defect was the filter that ran *before*
it, so restoring the bug left every artifact test passing. The reviewer had
supplied a synthetic shard as the reproduction and I wrote a table test of the
helper instead. **A negative fixture has to enter through the same door the
defect was behind**; the fixture builds a real shard through `Inventory::build`
now and fails on `left: 3, right: 4`.

**A measurement tool that cannot report an invalid measurement is not one.** The
mutation driver appended unstable repeats and failing restored controls to a
`nondet` list **and counted them as caught anyway** — a review drove its `main`
with a control that never passed and read "1 of 1 caught, 0 survivors". The
verdict is a pure function now, only `caught` counts, skips are named, the
process **exits nonzero** unless every mutation is caught, and the rule is
self-tested over eight cases rather than read. **Test the thing that produces
your evidence, not only the thing the evidence is about.**

**The paragraph on task 0022 below says "no Laguna tensor was read". That was
true when it was written.** Three of its tensors have now been read and
imported; nothing computes with them. The older paragraph is left as it was,
because rewriting a record to match later work erases what was true when it was
written — and this sentence exists so that a reader meets the correction before
the stale claim rather than after it. The same applies to that section's "O1's
catalog and O5's storage questions stay open": the owner resolved O1–O5 on
2026-09-13, and [the Laguna bring-up record](models/laguna.md), which is a
living record rather than a task's, carries the rulings.

**Two facts about these artifacts that only reading them produced.** A module's
four tensors **need not share a shard** — Laguna keeps them together for 34,739
of 34,740 modules and Qwen3.8-27B for **none** of its 256, so a caller needs an
index resolver, which is M3 item 1's. And Laguna's **140,989** tensors put a
shard header at about a megabyte serialized, a 16.5 MB admitted peak against
`HeaderBudget::DEFAULT`'s 8 MiB: the default refusing it is the budget working,
the default is unchanged, and any production path that opens this artifact has
to state one.

**M2 item 1's routed-expert mathematics is accepted**
([task 0019](tasks/0019-m2-routed-expert-semantics.md), 2026-09-12, after
three rounds of independent review): `Route`, `ExpertMlp` and `Combine` as
shared operations with FP64 oracles, the pinned BF16 boundaries that decide
which experts a row selects, and two consumers carrying opposite routing
parameters.

**M2 item 2's residency authority is accepted** (2026-09-12, after seven rounds
of independent review — twenty-seven findings, all reproduced, all fixed, none
disputed; the seventh recommended acceptance with no new blocking findings)
([task 0020](tasks/0020-m2-weight-residency-authority.md)):
`moxie_memory::residency` is the **one** production weight-residency owner, with
document 03's chunk identity, its lifecycle and failure transitions, coalescing,
event-bound device uploads whose allocation is tied to its reservation,
deterministic demand LRU with a bounded prefetch class, ADR 0009's
conditional-memory floor, and an admission report naming the incoming chunk
before any victim is chosen. An `arch-check` rule rejects a second owner. All
nine of M2 item 5's cases pass, on the host lane and on all three GPUs, alongside
a regression for each review finding. One of those findings corrected a claim
rather than a defect: **a nonblocking acquire is necessary but not sufficient for
deadlock freedom**, and the cycles it missed were between the demand counter and
the prefetch gate, and along promotion's dependency chain. **The acceptance
closes task 0020 only, not M2**, whose exit also needs traces reconciled with
the ledger across a whole working set.

**`arch-check` now passes with zero failures.** The four it reported as
"pre-existing" from task 0014 onward were review probe crates parked under
`results/`, which [docs/README.md](README.md) declares ignored scratch; the
crate walk now excludes root scratch **by reachability, computed exactly**, and
**fails closed** — checking everything — when reachability cannot be computed. Do
not carry a standing failure count forward as background noise; that is how a
real one gets missed. And when you narrow a check, **narrow it so that being
wrong is loud**: four review rounds found four different ways a cheaper
approximation of "is this crate part of the build?" silently hid production code
from every rule. **Ask one question once.** Those four were four places
answering it separately — a directory-name test, a membership-string test, a
partial glob expander, a second dependency-table enumeration — and deduplicating
three of them while leaving the fourth is how the fourth was found.

**A state machine's tests should enumerate its product, not sample it — and the
sweep itself needs evidence.** Twenty-three review findings against task 0020
shared one shape: a transition that was individually reasonable left the
structure inconsistent in a combination nobody had written a test for.
`ResidencyAuthority::check_invariants` states the invariants once and
`residency_transitions.rs` sweeps 800 combinations calling it after every
operation. Two things are required of such a sweep, both learned the hard way:
its harness must perform **only** work the scheduler actually handed out — the
first version completed work it discovered, and a mutation that discarded every
promoted order passed all 200 combinations — and its strength must be
**measured by mutation testing** rather than asserted.

**State coverage as a number the test prints, never as a sentence in a record.**
Three claims about task 0020's test strength were wrong in the same way: the
property was asserted rather than measured. The sweep prints what it exercised,
and an equivalent mutant is reported as such instead of being counted as a gap.

**M2 item 3 is accepted by the owner on 2026-09-13**, after four rounds of
independent review
([task 0021](tasks/0021-m2-expert-execution-plans.md), implemented
2026-09-12; the four
rounds found **twenty-one** issues, fourteen P1, all reproduced and fixed, none
disputed): CPU
expert fallback and GPU grouped candidate plans under one interface, with an
admitted envelope, a bounded queue that refuses rather than waits, NUMA-placed
host buffers, and a reduction over a permutation the plan computes so partial
outputs are **placed, not accumulated**. The grouped kernel is **bitwise** equal
to task 0019's oracle on all three GPUs for both gate transforms — which needed
`__fmul_rn`/`__fadd_rn` and `__dmul_rn`/`__dadd_rn` throughout, because nvcc
contracts into an FMA by default and that is a different answer. **One layer's
routed expert block of the designated artifact executed**: 118,947,840 B of
layer 0 demand-loaded into a device cache holding **two** of ten experts, 28
evictions, 8 backpressure drains, agreeing with the CPU candidate on 45,056 BF16
components. **The activations are synthetic and the route is written by the
test, so this is not model support and no quality claim follows.** M2's exit
still needs traces reconciled with the ledger across a whole working set.

**A property of this machine is measured or it is not known.** Three plausible
NUMA mechanisms in a row were wrong and only the read-back showed it: a zero
store the compiler may delete, first touch on pages the allocator had already
faulted elsewhere (**3,317 of 8,192** on the wrong node), and the default policy
falling back rather than reclaiming, because node 1 has **334 MB** free against
node 0's **5.1 GB** (**4,471 of 6,144** on the wrong node). `required` now means
`mbind` and the gate is every page. Do not carry an unverified placement,
affinity or bandwidth claim forward.

**Ask of every check what else reaches the resource it guards, and what the next
call does with what it set.** These are two questions and task 0021's first two
review rounds are one each.

Nine of the ten first-round findings were one sentence: **a check that existed on one path
and was missing on the neighbouring one.** The upload path validated the backing
a lease is resolved through and the launch path did not — authority A's leases
driven through authority B's backing returned a confident, different answer on a
real GPU. The planner computed one envelope for admission and checked
feasibility against a smaller one. A run could be cancelled but not fail, so a
failed group was followed by a successful reduction over an unwritten buffer.
This is **not** the failure mode task 0020's transition sweep exists for: a
sweep enumerates one state machine's product, and these were parallel paths
never compared to each other. The tenth finding is why it matters that the
question be asked at all — `ExpertGroup`'s launch indices were public `Vec`
fields, so the answer was "anything".

The second round was **the same path, one step later**: quarantine set correctly
at the moment of failure and ignored by `close`, which then released the charge
for buffers that can never be freed; a failure made terminal for a *group* and
not for a *load*; a backing checked and the lease inside it not. Two of its five
were the other half of the first round's own, so **"all ten are closed" was a
claim about fixes rather than a measurement of them** — the same error AGENTS.md
already records three times over test coverage. The answer is the method task
0020 established and task 0021 had applied only to its planner. The third round said plainly that
deferring that to a later task is not evidence this one is finished, and it was
right: **enumerate the run's product too**. It is built — 144 combinations of
candidate × failure point × cancellation × close ordering × queue depth, an
invariant after every operation, the residency authority's lease count
reconciled against the run's after every one of them, coverage printed, and
**15 of 15** mutations caught, two of which are round two's own P1s. The
fifteenth is round four's: one of the sweep's advertised axes was **unreachable**
— both branches drained the queue before closing, so `close`'s queued-work
refusal could be deleted with all 144 combinations still passing.

**A gate only fires on inputs something actually hands it.** A descriptor
carrying this build's own package hash with the **other activation's** projection
symbol was resolved and launched, and every GPU computed SwiGLU for a GeGLU plan
— 3,959 of 4,096 components wrong. Task 0021's declared gate is bitwise equality
with an FP64 oracle for the activation the **plan** requested, so it *would* have
failed on that; the review measured those wrong components with it. What was
missing was that **no case ever built an inconsistent descriptor**: every plan
came from the built-in catalogue, so descriptor consistency was an assumption the
tests carried rather than a property they checked. A strong gate over a narrow
input space is still a narrow test.

The fix is a binding, not a better gate: a selected descriptor must **be** one
the built-in package declares — operation, ABI, operand roles, precisions,
rounding, layout, shape bounds, SM, workspace and symbols together. Half an
identity check is not a weaker check; it is a check of something else.

**A public function may not assert a safety contract on its caller's behalf, and
may not hand out what it is responsible for retaining.** Round three: the device
attachment took any `&'static [u8]` and passed it to
`TrustedImage::from_build_output`, whose contract is that the bytes are this
build's own fatbin; and its launch and upload entry points took borrowed operands
that only the **run** withholds after an unknown submission. Both are now the
crate's own — the image named at the call site as `chain` already did, the
operands reachable only through their owner. When a check is "the caller must
guarantee", ask who the caller can be.

**Measure the tests, then fix what the measurement finds.** Task 0021's sweep
started at 13 of 16 mutations with two survivors. One survivor was a **product
defect** — a refusal reporting `CapacityExceeded` where a `required` candidate's
own reason belonged — and the other was a fixture whose every row was already
ascending, so the two reduction orders agreed and a planner ignoring the
parameter passed. It is 16 of 16 now. A fixture on which two behaviours agree
tests neither, and an unreachable branch is a stub: one was found and deleted
the same way. The same battery applied to the review's twelve regressions found
**three that asserted the symptom rather than the check** and would have passed
with the check removed; the second round's six added four more, three of them
aimed at the wrong one of two identical lines. It is eighteen of eighteen now. A
regression is not load-bearing until a substitution says so — and when a
substitution says a check is redundant, **delete the check**: one line went that
way, a state reset the failure path already performed. The run sweep took four
rounds of strengthening for the same reason: one equivalent mutant, and three
axes that **recorded being reached without checking what they caused**. Counting
that a failure happened is not the same as checking what it did — and an axis
that is *exercised* is not an axis that is *checked*: a fourth round found one of
this sweep's advertised axes unreachable. **Three separate coverage claims in one
task were softer than they looked, and every one was found by asking what a
mutation would survive rather than by reading the test.**

**M2 item 4 is accepted by the owner on 2026-09-13**, after three rounds of
independent review
([task 0022](tasks/0022-m2-laguna-metadata-and-second-consumer.md),
implemented 2026-09-13; three rounds found **eight** issues, two P1, all
reproduced and fixed, none disputed, and the third recommended acceptance with
no new blocking
findings): Laguna's metadata and its **routed block**, a second routed
consumer driven through task 0021's interface, and a restricted budget expressed
as a **ratio** of what the route demands rather than a constant. A routed layer
at Laguna's declared expert width — 18,874,368 B per expert in BF16 — ran on all
three GPUs against a device cache of one eighth of its 226,492,416 B working
set, with 11 backpressure drains and 22 evictions per card, agreeing with the
CPU candidate on all 6,144 components. **Those are weight-shaped bytes at a real
artifact's declared shape, not its weights**, and no quality claim follows.

**Laguna's attention tower is not composable, and the contract said so before
the work started.** Two operations are missing and neither may be guessed:
`softplus` attention output gating, whose equation is pinned in the artifact's
own file but which has no shared operation; and the **yarn** rotary ramp on its
twelve `full_attention` layers, which `modeling_laguna.py` delegates to a
`transformers` function the artifact does not ship — the artifact declares
5.14.1, the copy installed here is 5.5.3, and `truncate` is not declared at all.
Gating is on all 48 layers, so **no Laguna layer's attention is composable
today**. Both are named gap tasks, computed from the declared geometry rather
than written down. Its weights are asymmetric INT4 at group 32 with zero points
packed along the **output** axis — a second convention the accepted importer has
never seen — so **no Laguna tensor was read** and M3 owns the importer.
[The bring-up record](models/laguna.md) carries the inventory and every
open mapping question.

**A substitution result from a nondeterministic test is not evidence, whichever
way it came out.** Task 0022's two allocation regressions read a process-wide
counter while the harness ran them on parallel threads: **14 failures in 100
runs**, and another test's allocations falling between a measurement's two
snapshots. The rule this repository already has — a regression is load-bearing
when a substitution says so — quietly assumes the substitution is repeatable,
and a flaky test turns it into a coin flip recorded as a measurement. The
battery repeats each substitution **25 times in both directions** now. Fixing
such a test by requiring `--test-threads=1` is not fixing it: it removes the
flake and leaves the gate everyone actually runs unreliable.

**A fix is a new caller, and a new caller on a path with a discipline has to
satisfy it.** The second review's only finding was a defect the **first round's
own correction** introduced: `narrow_coefficients` built its result with
`collect()`, and an allocation failure there aborts the process instead of
returning a typed error the transaction can roll back. That is `softmax`'s
defect, in the same module, for the **third** time — task 0019's record states
the rule and the reason in as many words. A diff review of a fix will not catch
this, because the fix looks like the thing it replaced; the question to ask is
what discipline the *path* has. Asking it of the rest of the same change found a
second instance the review had not reached: `route_operands` returned a `Vec`
and the interpreter calls it once per `Route` node per step. Two of one class in
one change, and one reproduction found one of them.

**A transcription is independent of the implementation, not of the reader.**
The review's P1 was a BF16 boundary this implementation did not have:
`LagunaTopKRouter.forward` ends with
`routing_weights.to(hidden_states.dtype)`, and the coefficients were being
returned in FP32 — `[0.5938455, 0.4061545]` where the source has
`[0.59375, 0.40625]`, on every combined row. It survived a **bitwise** gate,
because the FP64 transcription was written from the same source by the same
reader and omitted the same cast; the two agreed, and their agreement proved
only that one omission had been made twice. The boundary is post-selection, so
"the selection is exact" could never have seen it. Check a transcription against
a quantity computed from the source's stated equation, not against the
implementation it is supposed to be independent of.

**A number in prose is outside every battery this repository has.** Two of the
same review's findings were arithmetic in a record that contradicted a fact the
same record already contained: the expert inventory multiplied a per-layer cost
by all 47 routed layers two paragraphs after recording that two of them cost
something else, understating the working set by **6.87 GB**; and a nearby line
put this machine's aggregate VRAM at "72 GiB" when
[the hardware inventory](evidence/hardware-inventory.md) had already
measured **62.6 GiB**. Mutation testing measures a suite against mutations of the
*code* and cannot reach either. Make a record's arithmetic **executable** — the
expert inventory now sums over the layers and a test checks every one of them
against the artifact's own `data_offsets`.

**A parameter no fixture ever varies is a parameter no test checks.** Two
mutants that dropped `Combine`'s new output scale survived a **10,368**-case
sweep and every executor test, because every routed fixture in the workspace
used a scale of exactly 1.0 until a second family arrived with 2.5. Nothing was
weakly asserted; the assertions were strong over an input space in which the
parameter was constant. That is the fourth review of task 0021's own lesson — a
gate only fires on inputs something hands it — in a new place, and the question
to ask before claiming a parameter is covered is **which fixture varies it**.

**A test that compares two runs of the same code is not a test of what the code
means.** `per_expert_scale` and `selection_bias` are both `[experts]`, so a
swapped binding passes every shape check. The test written for exactly that
hazard built the graph both ways and required different answers — and survived
its own mutation, because reversing the single statement of the operand order
relabels the validation and the interpretation consistently. The two runs were
still different; they were each other's. It is now checked against an oracle
composed with each operand in the role its name says.

**M2 item 5's remainder is accepted** (2026-09-13, at `ad69a0a`, after three
rounds of independent review — ten findings, four P1, all reproduced, all fixed,
none disputed; the third recommended acceptance "within its declared M2
engineering scope")
([task 0023](tasks/0023-m2-whole-working-set-trace.md)): byte
and cost traces reconciled with the resource ledger across a **whole working
set**. **Every routed layer of the designated artifact has executed, on all three
GPUs, at two row shapes.** At a decode-shaped batch all 30 layers agree
**bitwise** with the CPU candidate over the same bytes on every card; at a batch
whose route partitions the expert set, the artifact's **entire routed expert
payload — 45,675,970,560 B, 88.5% of it and 1.77 times the largest card here —
was demanded, admitted and computed with** through a device cache holding one
240th of it, 15,286 evictions per card, no OOM, the three cards bitwise equal to
each other. **The activations are synthetic and the routes are written by the
test**, the full-payload route *constructed* so its union is the whole expert
set, so this is not model support, no quality claim follows and **no
route-distribution claim follows either**. **The acceptance closes task 0023
only, not M2**, which is the owner's separate decision.

**The owner resolved O1–O5 on 2026-09-13** (`1e927ef`, ADRs 0017–0020), while
task 0023 was being implemented. Three of them touch the active work and are
worth reading before the next task rather than rediscovering: the **v1 catalog is
ten pinned revisions**, Laguna is number 8 of them, and `gemma-4-26B-A4B-it` is
**explicitly not in it** — it "remains M2's BF16 workhorse", so every
whole-working-set result above is an engineering fixture at real scale. **v1's
acceptable quality loss is a bit-identical repack**, with the publisher's own
quality accepted as-is, and **Moxie never quantizes**. Storage and conversion are
**user-managed**: no agent-initiated bulk write without a task naming the
artifact, revision, size and retention. **O6 and O7 stay open**, which is what
keeps every timing in these records a diagnostic. The paragraphs below, and tasks
0018–0022, were written while those gates were open and still describe them so;
they are left as they were, because rewriting a record to match a later ruling
erases what was true when it was written.

**Reconciled means a named equality, not a printed report.** Seventeen of them
tie the ledger's charges, the run's own record and the planner's prediction to
the residency authority's per-scope byte flow, and **each has a fixture that
violates it** — a `reconcile()` that can only succeed is a stub. **The trace
counts nothing**: every number is read from an owner and a layer's record is the
difference of two snapshots. The `arch-check` rule against a second residency
owner now rejects a second *byte* owner too.

**A comparison is worth nothing when both its sides can come from the wrong
place, or when neither is the quantity it names.** That is four of the six
findings an independent review made against this task, and it is the sentence to
carry. A trace read from a **different, empty ledger** reconciled a completed run
with no charges at all, because empty equals empty. A step that **omitted an
executed layer** reconciled 35 checks, because its totals were derived from the
records it was handed and then re-summed from the same records — a step needs its
own boundary, taken before its first layer, and now has one. The **launch
equality compared groups** where the device submits one kernel per *symbol*: a
check added *because* mutation testing found the counter unchecked, then written
against the wrong quantity. And the **"exact" prediction was unsound for the
third time**, each time for the same reason — it reasoned about *this plan's
share* of a cache, and eviction does not. A warm chunk a layer needed was older
in LRU than an unrelated cached chunk, so the layer's own admissions evicted the
chunk it had predicted a hit on: **3,840 B read against 3,072 B predicted
exactly**. Exactness now means **nothing has to be evicted**.

**A trace runs inside a generation step, so it may not abort.** The review's P1:
one injected allocation failure before a snapshot gave **SIGABRT**. Task 0019's
rule, for the **fifth** time in this workspace, on a path added after the
previous four. Every collection the trace builds is reserved fallibly, **none is
a `BTreeMap`** — it has no fallible insert, so a map there would abort however
carefully the rest reserved — the owners hand their numbers out through visitors
that allocate nothing, and the snapshots are deliberately **not `Clone`**.

**A gate that counts calls cannot see a leak.** The allocation gate compared
allocator *call counts* between layers and a deliberate **1 MiB per layer** leak
passed it. It measures live bytes now, declares what a layer may retain, and
performs the leak substitution itself — the gate is one somebody has watched
fail.

**Reserving a destination says nothing about a temporary the callee builds.** A
second review round found the trace still aborting on an injected allocation
failure: the *caller* reserved everything it needed and the callee it asked for
reservation identities built a `Vec`. Fixed-size now. And **a report of an
allocation failure may not allocate**: the same round found reconciliation
formatting the message that says it ran out of memory. A `Cow<'static, str>`
detail, borrowed on that path.

**The test I wrote for a review's finding had a fake axis.** Its loop called
`while_failing(1, ...)` six times, so **every iteration failed the first
allocation** and the index only changed the assertion message — task 0021's
fourth-review lesson, in a test written *because* of a review, one round later.
It sweeps every allocation position now and **measures how many there are**
rather than assuming.

**Three wrong versions of one rule, each found by a counterexample and none by
reading it.** Exactness asked whether this plan's admissions fit; then whether
its admissions and its hits fit; then whether everything resident and its
admissions fit the cap. Each is a quantity that is not the one admission asks
for. **Admission asks for a contiguous range**: a 4,608 B cache holding 3,840 B
in three 256 B holes satisfies every one of those inequalities and evicts anyway
— 1,536 B read against 768 predicted exactly. When a rule is wrong three times,
the thing to change is not the inequality but which quantity is being compared.

**A diagnostic that cannot be built is a process that cannot report anything.**
A third review round found an *ordinary* discrepancy still formatting its prose:
a mismatch plus one failed allocation was **SIGABRT, 114 bytes**. The fix before
it had made the path that *handles* an allocation failure safe and left the
ordinary paths alone, on the argument that a mismatch is not an out-of-memory
context. **That argument is wrong: under memory pressure a mismatch is as likely
as an allocation failure.** An error's prose is `&'static str` now and its
numbers are structured fields; `Display` composes them, so whether rendering
allocates is the caller's decision rather than a step's. The field was walked
down three times -- `String`, `Cow`, `&'static str` -- once per review round, and
each step was somebody else finding what the previous one left.

**Test the failure path under the failure.** The sweep written for the previous
P1 reconciled only a **valid** trace, so it never constructed a diagnostic and
could not have caught this. Every equality is violated in turn with every
allocation refused now. A gate that exercises the happy path under the adverse
condition has tested the adverse condition on the happy path.

**A lane nobody runs is a lane that holds failures.** The same review found two
clippy lints standing in `xtask/src/gpu.rs` since task 0021. The declared gates
named a host clippy lane and a `--features moxie-executor/driver` one, and
**neither compiles `xtask`'s CUDA code**. `cargo clippy --workspace
--all-targets --features cuda` is a third lane and is now declared.

**Three quantities the accounting did not have, and could not be reconciled
without.** A device upload whose host source was already resident was counted as
**nothing at all**, so host reads and host admissions could not be compared;
joining a transfer in flight was not distinguished from finding the bytes
already there; and `ResidencyStats` summed three cards into one
`bytes_uploaded`, which is AGENTS.md's forbidden "their memory is one
allocation" assumption written as arithmetic. **None of the three is reachable by
mutating the code**: they are missing quantities, not wrong ones, and what finds
them is asking what an equality between two independently maintained numbers
would require.

**A total cannot say where something is.** The planner's residency snapshot was a
set of experts, then a byte count per expert; both are wrong for the same reason.
An expert is more than one chunk, the authority admits and evicts chunks
individually, and the device can hold one of an expert's chunks while the host
holds the other. Whether an upload can copy from the host instead of reading is a
question about **which** chunk is where. It is a per-chunk reading now, and the
planner still never learns what a role is.

**A prediction is exact or it is a declared bound, and the condition is
coexistence.** A layer whose whole live set fits the displaceable cache admits
every chunk once; one that does not can lose a chunk to eviction between a
backpressure refusal and its retry. The first version of that rule asked only
whether the layer's *admissions* fit, so a layer that predicted hits on bytes its
own admissions then evicted called itself exact and read them again. The sweep
found it; reading the rule did not. **Both branches have an acceptance case**,
because an unreachable branch is a stub.

**The mutation battery found two things no test was checking and one that could
not fail.** Deleting the device launch counter survived everything, because
nothing compared it to anything — it is an equality now. Setting the residency
high-water mark to the current level survived, because nothing ever compared two
readings of it. And deleting the filter that keeps only *this step's* reservations
survived, because in every fixture the only reservations the ledger held were
this step's: "no third charger" was the property with no fixture at all, and a
fixture on which two behaviours agree tests neither. **All three were product or
coverage defects rather than test-strength opinions.**

M1's accepted work: shared semantic tensors/graph and the bounded host
interpreter, sequence transactions, canonical manifest and bounded reads, the
resource ledger and admission, event-backed leases, the device arena, the
admitted graph resource plan, the selected BF16 device chain, appendable paged
state, transactional sampler history, the generation service and diagnostic CLI
(tasks 0003–0015), the reduced Gemma graph
([0016](tasks/0016-m1-gemma-reduced-graph.md)), per-layer paged geometry and
window reclamation
([0017](tasks/0017-m4-per-layer-kv-geometry-and-window-eviction.md),
accepted 2026-09-12), and the compressed-tensors importer
([0018](tasks/0018-m3-compressed-tensors-int8-importer.md), accepted
2026-09-12 within its import-only scope).

**M1.5 closed on the reduced graph, and the arithmetic is why.** Roadmap M1 item
5 qualifies its second clause as "actual checkpoint execution, **when
available/admitted**". The Gemma 4 artifact is **32.7 GiB** of tensor payload
against a **24 GiB** largest single GPU, so it fits on no device here: executing
it needs M3's W8A16 path *and* either M5's TP2 or M2's host-backed residency.

**Nothing executes a checkpoint.** The reduced graph is a synthetic contract
fixture over invented weights and may not be described as model support; the
importer produces canonical tensors that nothing runs, and its packed-word lane
order is cited from the pinned reader rather than verified, so **no quality
claim follows from a successful import** — that needs paired output against the
released model, which is O2. Task 0021 went one step further than task 0020's
read and **computed** with 118,947,840 B of the designated artifact's real
expert weights — but over **synthetic activations and a route its own test
writes**, so it establishes machinery and nothing about output. **One layer is
not a model**: nothing composes a routed layer into a graph that generates a
token, and the whole 51.6 GB working set has not run. Task 0022 went a different
way and not a further one: it ran a routed layer at **Laguna's declared expert
shape** rather than from Laguna's bytes, because its weights are asymmetric INT4
at group 32 and no importer accepts them. A shape is not a checkpoint. M4 still
owns device paged attention, COW forks and page streaming. M11 owns vision.

**M2 is active and proceeds in roadmap order.** The owner designated
`/fast/models/google/gemma-4-26B-A4B-it` as M2's BF16 MoE on 2026-09-12: BF16,
unquantized, **128 experts at top-k 8**, 51.6 GB across 1,013 tensors, already
on disk — **no download is required, and none was made.** It is the same
`gemma4` family M1 already gated, with the same `(layer + 1) % 6` global
predicate, `attention_k_eq_v`, local 8x256 versus global 2x512 key/value
geometry, softcap 30.0 and 1,024-token window, so the accepted text-tower
mathematics and task 0017's per-layer paged geometry transfer rather than being
rebuilt. Its experts are **fused per layer** — one `experts.gate_up_proj` and
one `experts.down_proj` holding all 128 — and **every layer carries a dense
`mlp` beside the routed experts**, so routing semantics must model a shared
expert explicitly. At 51.6 GB it fits aggregate VRAM but no single 24 GiB
device, and M2 item 4's restricted budget makes it oversized by construction.
Task 0019 composed its routed block over synthetic weights at reduced scale and
recorded the artifact's inventory in
[the bring-up record](models/gemma4.md#the-26b-a4b-moe-variant). Task 0020
made its expert bytes resident on demand — the experts are fused per layer, so
that needed the bounded ranged read it added — and task 0021 ran one layer's
expert block from them. A fact that came out of that: its experts are
**11,894,784 B each**, so at a two-row decode batch the best reuse available is
5,947,392 B per row, and against the declared default amortisation threshold of
1 MiB per row **every expert goes to the CPU candidate**. Whether that is the
right decision is a measurement, and measuring it is M6's.

`hy3-w4a16-mtp` is in scope. The Laguna checkpoint at
`/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4`, revision
`bc59f497520b23759ce61cc5164ca28bcc4f53bc`, is **complete** — all 15 shards,
each satisfying `8 + header + payload_end == file size`, payload ends summing to
the index's `total_size` of 76,813,095,232 B, re-verified 2026-09-13. Task 0022
**interpreted its metadata and read no tensor**: 48 layers at hidden 3,072, a
dense layer 0 and 47 routed layers of **256 experts at top-k 10**, SwiGLU at
`moe_intermediate` 1,024 beside a shared expert of the same width, a **sigmoid**
router with an `e_score_correction_bias` that moves selection only, a routed
scaling factor of 2.5, and **94.41%** of the artifact in expert weights —
72,515,874,816 B, summed over the routed layers rather than multiplied, because
the quantizer left layers 46 and 47 in BF16 and a multiplication cannot say so.
Its
`configuration_laguna.py` and `modeling_laguna.py` were **read and never
executed**, which is how document 03's "decoded according to the pinned
exporter, never guessed from a suffix" is satisfied; executing them stays
forbidden. Its `dflash` draft model is not on this machine and is M9's. None of
the three artifacts is approved for quality, conversion or requantization: O1's
catalog and O5's storage questions stay open.

