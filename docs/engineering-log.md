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
| A claim checked against a copy of itself | task 0026's component validation, shape check, source digest and memory bound; task 0038's device test reimplementing the authority's page-table arithmetic instead of reading it from the authority |
| Optimizing the architecture, and calling it a user benefit | offline repacking, built and packaged before its performance hypothesis was tested |
| A tool that measures the tree it is editing | the mutation battery, twice: killed mid-substitution, and left unrunnable for two tasks |
| A test measured by assertion rather than by mutation | tasks 0020, 0021, 0022, 0023, 0024 |
| A record contradicting a fact it already contains | task 0022's expert inventory, task 0022's VRAM figure, task 0024's sign claim, task 0038's page-ownership claim in `AGENTS.md` and the support matrix, task 0038's writer-trait claim going stale in the opposite direction one round later, task 0038's "deliverable 1 landed" claim called premature the very next round |
| A comparison whose two sides are not independent | task 0022's FP64 transcription, task 0024's FP32-versus-BF16 arithmetic |
| A fix that is not guarded by the test written for it | task 0023's fake axis, task 0024's helper-only regression |
| A numerical gate whose denominator can collapse | task 0028's 2-ULP-at-the-result threshold |
| A check attached to a value nobody has to keep | task 0028's public launch fields, trusted descriptor, unchecked lease scope and borrowed operands; task 0038's `WriteReceipt` and the quarantined range `PagedCloseRefused` handed back on the refusing path |
| Work that exists on one disk and nowhere else | 24 commits, four tasks and five review rounds, unpushed for thirty hours |
| A repair that satisfies the test rather than the property | task 0029, four rounds: an owned label the fixture never used, a prefix measured on a warmed ledger, a mutant that panicked before it mutated, an aggregate standing in for a state |
| A harness with no tests of its own | the mutation guard, after three rounds of guard bugs: the self-test covers verdicts, selectors and anchors, and never touched restoration |
| A measurement that shares its instrument with whatever runs beside it | task 0025's budget lane, whose global allocator counted the other test's fixtures for two tasks, and was serialised rather than fixed |

## Entries, newest first

**A measurement that shares its instrument with whatever runs beside it
(2026-09-16).** Task 0025's budget lane is the only gate in this repository that
measures memory rather than asserting about it: a counting global allocator,
peak live heap across a whole repack, against the bytes the ledger admitted.
The instrument is **process-wide**, and the file had two tests in it.

The first version let the two tests reset each other's peak, and that was found
and fixed with a lock. The lock covered the measured call. It did not cover
building a two-mebibyte fixture, opening the sources, reading `LIVE` after the
run, or destroying the scratch directory -- all of which allocate, all of which
ran in parallel with the other test's open window, and every byte of which a
process-wide counter attributes to whichever window is open. So the lane stayed
load-dependent: two baseline runs on an identical clean tree reported "fails"
and "disagrees with itself", and `cargo xtask mutation-check` -- correctly --
refused to build verdicts on either.

What happened next is the part worth keeping. The battery was made to pass
`--test-threads=1` to the lane, with a comment saying whose bug it was and that
the fix belonged in the test file. That is a **workaround recorded honestly and
still a workaround**: it kept the battery runnable, which matters, and it also
meant that for two tasks the only memory gate here was measured under a
condition nothing else in the suite runs under, and nobody had to look at it
again. A gate that needs the harness held a particular way is a gate whose
number nobody can reproduce from the command in the record.

The repair is scope, not locking: a measurement is now a session, taken before
the fixture exists and released after it is destroyed, with every counter read
inside it. The peak window opens after the fixture is built, because what is
being measured is a repack and not the bytes a test wrote to give it something
to repack. `--test-threads=1` is gone from the lane.

Three things came out of measuring the repair rather than asserting it:

* **The negative control is what the gate is worth.** A bound that cannot fail
  is not a measurement. The lane now allocates 32 MiB against a bound of 8, and
  checks the meter sees it. Written first as a buffer *held* for a quarter of a
  second, it proved nothing: a window reports the peak above the live bytes it
  opened with, so a buffer already live when the window opens is part of the
  baseline and moves nothing. Only a burst **inside** a window moves a peak.
* **The counterfactual is a rate, not a yes or no.** With the burst outside a
  session -- exactly where the fixtures used to be built -- the lane failed 5
  of 20 runs, the failure landing in whichever window the burst happened to
  overlap. That is the "disagrees with itself" the battery reported, reproduced
  on demand. With sessions: 20 of 20, and 12 of 12 under 56-way CPU load with a
  concurrent workspace build, every one reporting the same 209,256 B peak.
* **The instrument has a resolution, and it is now named.** Under load the
  control read 33,554,044 B of a 33,554,432 B burst -- 388 B short, because the
  harness freed its own bookkeeping on another thread while the window was
  open. A window is conservative by whatever is freed elsewhere during it. The
  tile bound sits **4.33 orders of magnitude** above that 388 B and the admitted
  total **5.54** -- 8,388,608 / 388 and 134,217,728 / 388, divided rather than
  guessed. The first draft of this entry said "six orders" of both, which is the
  smaller version of the same habit the entry is about: a number written to feel
  like headroom instead of measured.

The control also had to be taught that a loaded machine can lose it two seconds
between reading the clock and checking it: its first version checked the
deadline before the body and failed under load having measured nothing,
reporting a broken meter. A control that can report zero work as a failed
instrument is a control that will be disabled by the next person who sees it.

**A repair that satisfies the test rather than the property (2026-09-15).**
Task 0029 made an admission path refuse instead of aborting. Its contract was
one sentence — an allocation failure is a typed error — and it took **four**
review rounds, none of which found the property unmet by accident. Each round
found a repair that made the test pass without making the statement true:

* labels became `Cow<'static, str>` so the constructor stopped allocating, and
  the *clone* still did — on the path where the label is owned, after the free
  list had moved. The sweep missed it because its fixture passed a literal,
  which is the borrowed half;
* the device sweep's prefix was measured on a warmed, reused ledger while every
  armed iteration used a fresh one, so it swept fewer positions than the call
  made and could not say so;
* the rollback mutants *deleted* a reservation, so a later insert panicked on
  zero capacity: they tested a crash, not a refusal returned after a mutation,
  and one of them wedged the battery for twenty-five minutes with the mutant
  still in the source file;
* the arena's "exact state" comparison compared `ArenaOccupancy` — byte and
  range counts, which cannot see offsets, order, owners or generations.

The common shape is not carelessness, it is **the test being the thing that was
repaired**. A property is a statement about every input; a test is one input. If
the fixture only ever passes a borrowed label, "labels are copied fallibly" is
never under test, and a green run says nothing about it. The question that would
have caught all four is the same one: *what would have to be true for this to
pass while the statement is false?*

**A harness with no tests of its own (2026-09-15).** The mutation guard — the
thing that puts a source file back after a substitution — produced three
separate defects in three rounds: it deleted its own backup after a failed
restore, it could leave a marker with no parked original, and it could not
express "marker cleared, copy pending" at all. It had **no tests**. The
`mutation-check --self-test` runs 109 cases and none of them touched
restoration, because the self-test was built to check verdicts, selectors and
anchors — which is exactly the tooling equivalent of a valid-only sweep.

It has five now, injecting failure at each step by putting a directory where a
file must be written or removed. The lesson is narrower than "test everything":
**the code that runs when something has already gone wrong is the code least
likely to have been exercised**, and a recovery path is entirely made of that
code.

**A branch nobody else can see is not a record (2026-09-15).** `origin/main` sat
at `44a94c4` from 2026-09-13 20:43 while local `main` ran twenty-four commits
ahead — 09-13 20:46 through 09-14 22:21, about thirty hours, 86 files, 29,747
insertions. Tasks 0025, 0026, 0027 and 0028 were completed, reviewed five times
and corrected in that window, and none of it existed anywhere but one disk.

The backlog was not a decision. No commit was held back for a reason; pushing
simply never became a step, because every session ended on the same two
sentences — the gates passed, here is what changed — and neither of them is
`git push`. Each session then inherited a repository that was already behind and
treated that as the normal state.

What it costs is not the risk of losing the disk, which is real but obvious. It
is that the whole apparatus this repository runs on — task contracts, ADRs,
handovers, independent review — is built for other people to read, and for
thirty hours the only way to read any of it was to sit at this machine. An
independent reviewer working from `origin/main` would have been reviewing task
0024. A handover written "so the next agent can pick this up" pointed at commits
that were not anywhere the next agent could fetch.

The rule, and it is a small one: **a task is not handed over until its commits
are on the remote.** A handover names a base commit; if that commit is not
fetchable, the handover names nothing. Check `git status -sb` for an `ahead`
count at the end of a session the same way the gates are checked, and push
before writing the summary rather than after — a summary is a claim about a
state other people can reach.

**A check is only as good as the value it is attached to (2026-09-14).** Task
0028's first review returned seven findings, five P1. Four were in the new
executor code and, listed separately, they look like four different bugs:

* a lease's **device** was never checked, only the authority that issued it and
  the length of its range;
* a launch that could not prove completion kept its arena and **not** the
  operands it was reading;
* the output readback aborted instead of returning a typed capacity error;
* every field of the checked geometry was `pub`, and the public admission entry
  point trusted whatever descriptor it was handed.

Together they are one shape. Each check happened at one moment, to a value that
anyone could change afterwards or that nobody had to hold. `AffineLaunch::derive`
validated geometry and then handed out a struct whose fields could be rewritten;
`select_affine_linear_kernel` matched a descriptor and `admit` accepted a
different one; `component()` validated a lease and the *device* it belonged to
was never part of the validation; `run` checked its operands and then borrowed
them across an asynchronous boundary, where "checked" stops meaning anything.

The repairs are all the same repair: make the value carry its own evidence.
Private fields, so a checked launch cannot be edited. One predicate, applied by
both selection and admission, so a descriptor cannot be checked by one and
trusted by the other. The lease's scope in the check that resolves its address.
And operands taken **by value**, returned on success and kept forever when
completion is unknown — because borrowing an operand across an asynchronous
boundary is not a lifetime, it is a hope.

Three more findings were in `moxie-repack`'s user surface, reached while reading
the same batch: an edited plan publishing a subset as `complete` because the
guard read the plan's own field rather than recomputing coverage; a size cap
applied to the second read of a document instead of the first; and planning
deleting any file at a predictable staging path, including one it had not
created. That last one is worth its own sentence: **a path this program did not
create is not this program's to remove.**

**The lesson to carry:** when you write a check, ask what holds the result of
it. If the answer is "the caller, until it decides otherwise", the check is
documentation.

**A tolerance is only as meaningful as what it divides by (2026-09-14).** Task
0028's contract predeclared, before any implementation existed, that each output
element must land within **2 ULP of BF16 at the oracle's magnitude**. That is a
good rule and it was written at the right time. It is also **unsatisfiable by
any implementation** on an output that has cancelled: the kernel reduces in the
tensor core's order and the oracle reduces in ascending order, and on one
element in a few thousand the true result sits six orders of magnitude below the
sum of the terms that produced it. At that point "one ULP of the result" is a
quantity no reordered FP32 sum controls. Measured: 37 of 101,376 elements missed
it at worst 6 ULP, while the same kernel's error against the reduction's own
term sum was **1.6e-8** — below one FP32 epsilon. The kernel was fine; the
denominator had collapsed.

Three things are worth carrying:

* **The threshold was still right to predeclare.** The failure mode this
  repository keeps producing is a tolerance fitted to a result. Writing it first
  is what made the conversation a ruling instead of an adjustment.
* **Ask before writing the tests, not after.** The measurement existed and the
  acceptance suite did not, so the owner's ruling ([ADR 0028](decisions/adr/0028-quantized-reduction-numerical-gate.md))
  shaped what got asserted rather than being retrofitted onto it.
* **The guard fires nowhere in the fixtures, and that had to be said.** Every
  element of five synthetic cases and of the real module passes the first clause
  alone. The temptation was to hunt for a seed that reached the second clause so
  the suite would "exercise" it — which is fitting the evidence to the guard,
  the same shape from the other end. It is driven directly instead, by a test
  carrying the measured numbers, which also asserts the clause still refuses the
  same difference on a well-conditioned reduction.

**The lesson to carry:** when a gate is a ratio, write down what happens when the
denominator goes to zero — before the first run, not after the first surprise.


**An internal improvement is not a demonstrated user benefit (2026-09-14).**
Offline repacking was sold to the owner on an architectural argument: one
canonical layout, no per-format branching in the execution path, cheaper
importers. All three are true statements about the code. **None of them is a
statement about inference speed**, and the owner said so when he challenged the
pitch.

What had accumulated by then: a canonical schema, a safetensors writer, a
manifest, a journal and restart protocol, a publication state machine, a
mutation battery, four rounds of independent review, and a compact plan format
for 34,740-module checkpoints. Real work, carefully done, and **not one measured
number about prefill or decode** — because this repository cannot execute a
checkpoint at all, so the hypothesis the whole thing rests on has never been
runnable.

The failure is not that the hypothesis might be wrong. It is the **order**: the
tooling grew to four review rounds of polish before anything tested whether the
step it automates is worth taking. A cheaper sequence existed — establish the
comparison first, or at least write the acceptance contract first, so the
criterion could not be fitted to the result later.

What was done about it: [ADR 0027](decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)
makes repacking **provisional** and bounds its scope;
[experiment 0007](evidence/experiments/0007-offline-versus-load-time-preparation.md)
is the acceptance contract, written **before** any result exists, with its three
outcomes and its fairness rules fixed in advance. It says plainly that sunk cost
is not acceptance evidence, which is the trap this shape of mistake sets on the
way out.

**The lesson to carry:** when a design's justification is performance, the first
artifact should be the measurement contract, not the implementation. And an
agent proposing such a design should say which parts are measured and which are
hypothesis — the pitch that started this did not.

**Tooling belongs in the tooling crate, and the port found the rot
(2026-09-14).** The mutation batteries were Python scripts living under
`docs/`. The owner asked why, given that `xtask` exists for exactly this. There
was no good answer: they substitute one string in a tracked file, shell out to
`cargo test`, and compare exit codes, all of which is `std`.

The port to `cargo xtask mutation-check` paid for itself in its first run. The
Rust self-test also checks that **every anchor matches exactly once** -- which
the Python driver only did while running, four hours at a time -- and it
immediately reported that thirteen of experiment 0005's twenty-four anchors had
matched nothing since before `ccfd7fa`. Tasks 0025 and 0026 had rewritten the
importer that battery measured, and nothing noticed because nothing re-ran it.
A mutation that does not apply is not evidence, so that battery was retired
rather than carried as a number nobody can reproduce.

The restore got stronger in the move, too, and for a reason with a scar: a
battery killed mid-substitution left `if false && got != unit.sha256` in this
repository's resume path, and the gates were re-run against it before the
device suite failed a corruption test and gave it away. A signal handler cannot
fix that -- `SIGKILL` catches nobody -- so the original is now parked on disk
**before** the substitution and cleared after it is put back, and the next run
restores whatever a killed one left.

One script stays outside the workspace: the reference `safetensors` reader.
That is not a tooling decision but a dependency one. Linking the reference would
breach the task's no-new-dependency constraint and weaken the claim -- our
reader agreeing with a crate we vendored is a weaker statement than our bytes
being accepted by an implementation that has never heard of us. It is invoked
as `cargo xtask reference-check`, so every gate here is still one command.

**Six ways to check a claim against a copy of itself (2026-09-14).** A second
independent review of the safetensors publication returned fourteen findings,
six of them P1. Listed separately they look unrelated. Together they are one
mistake:

* A component was checked to **exist** in its shard, never to be the dtype, the
  shape and the byte length its descriptor implies. Retyping a published `BF16`
  component to `F16` verified.
* A component list was validated as a **sorted set of kinds** and then streamed
  in the order given. Reversing it verified, and streamed reversed.
* A source file was **hashed by reopening its path** while its tensors were read
  through a cached handle. Replace the file between the two and the artifact
  holds one file's values under another file's digest.
* A memory bound counted **serialized text** -- the selection cap, the manifest
  cap -- while the run held the structures parsed out of it. Five thousand
  tensors peaked at twice the admitted total.
* Staging estimates were **per-record constants** while the records carry names
  of any length. A 1,000-character role overran a 160,000-byte budget.
* A published shard was held to **our own parser**, which reports payload slack,
  while the owner's ruling names the reference implementation, which refuses it.

The fix in every case is the same move: derive the expectation from something
that is not the artifact, and hold the artifact to it. The descriptor derives
the components. The kernel's `(dev, ino)` derives file identity. The selection's
own byte length derives the memory bound. The reference implementation derives
what a conforming shard is.

**And the tests have to be able to fail.** Two of these corrections needed a
second attempt, both caught by writing the regression first: binding a source to
one handle made the run self-consistent while making an atomic replacement
invisible -- strictly worse than before, since the re-hash had been catching it
-- and a cancellation test that keyed on a hard-coded count became
self-fulfilling the moment the code asked one more question. A test that cannot
distinguish the fix from the defect is not evidence either.

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
[`cargo xtask reference-check`](../tools/experiments/0007-reference-reader.py)
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
`tools/experiments/0005-mutations.py` (retired 2026-09-14),
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



## 2026-09-16 — M3 admission recovery (task 0032)

The actual affine admission call now has an allocation-position sweep through
its end, instead of a reconstruction that stopped before Rc allocation. A
private fallibly allocated shared core (explicitly approved by the owner)
preserves range retention; all 48 positions refused cleanly on each of the
three GPUs. Rejection/relocation collections and label copies are fallible;
reserving return capacity during allocation makes fragmented release allocate
nothing. Borrowed dynamic plan labels work again, restoring the CUDA gate.

Independent review found the new rejection sweep accepted the wrong refusal:
ordinary admission rejection and metadata allocation failure both flattened to
`capacity_exceeded`. Checking the exact admission variant and allocator fields
now separates them. The swallowed-error substitution is caught; full, partial
and zero-room relocation cases reach their first non-firing result. Error kinds
are useful API categories, but do not prove which failure path ran.

An interrupted publication mutation had left checksum validation disabled in
the working tree. Driver recovery restored its parked original before any
implementation or measurement. Long mutation campaigns need a stable snapshot
and their own restoration record; a dirty working tree can contain a deliberately
broken experiment, not a builder's proposed change.


## 2026-09-16 — static ordering and the source-set boundary (task 0033)

Pinned compressed-tensors static/weight ordering preserves the saved column
order; group/dynamic ordering requires a map. Treating both as permutations had
unnecessarily refused the static sources. The new static lane matches11,264
sampled canonical values across two pinned HY3 projections.

The first map guard searched only a selected module's companion shards.
Independent review placed its map beside a different selected BF16 tensor and
escaped that guard. The corrected scan checks all declared source shards once,
borrowing names from each header; a nested per-module scan would have reparsed
large headers tens of thousands of times. Indexed discovery checks its index;
a manual partial selection has only its declared source set. A claim about all
checkpoint files needs the index and its admitted parse memory.


## 2026-09-17 — source scales and quantized expert continuation

Task0034's first bounded real AutoRound sample failed the positive-scale rule:
the pinned GLM source stores negative F16 scales. Synthetic positive-only fixtures
could never have shown it. The owner chose exact signed-scale preservation in
ADR0030; 43,008 sampled source values then matched the canonical equation bitwise.
Read actual source values before calling a format integration validated.

Task0035 extends the existing grouped planner, host kernel, device attachment and
residency authority. Packed codes and metadata replace BF16 byte extents; they do
not introduce a second cache or a full dequantized weight copy. Both gate
transforms and mixed projection precisions passed synthetic oracle checks on all
three GPUs. Composition of two graphs is separate from execution of a complete
graph; neither yields a checkpoint token. Mapped GPU operands and artifact binding
remain explicit work, not an implied capability of a successful decoder test.


## 2026-09-17 — mapped artifact execution and survivor regressions

Canonical expert group maps now occupy the tail of the already admitted weight
lease. Both reduced graph consumers execute published synthetic GPTQ operands
on all three GPUs, including odd projection tails and cancellation. Artifact
binding rejects an unbudgeted map before writing the destination. This exercises
shared expert/combine operations, not complete model graphs.

All five T0006 survivors are caught by new targeted regressions with three
baseline/mutant/restored repetitions. The tail-write mutant was initially
misdescribed: it adds duplicate I/O after the journal, rather than removing the
original write. Reading the exact substitution changed the claim and regression.
The full battery remains a separate gate; targeted success cannot close it.

The enlarged expert catalogue exposed a test that chose descriptor zero to
inject a wrong image hash. Descriptor order is not a capability-selection API.
Keep all descriptors and change only image identity when testing attachment
failure, so the test reaches the intended resource-release boundary on either
GPU architecture. The original failure and successful targeted rerun are kept.

## 2026-09-18 — close the activation-order execution gap

Importing a GPTQ group map while the dense shared kernel refused it left one
serialization feature without the dense consumer promised by M3's common
matrix. The map is now a separately admitted weight component whose lease has
the same device and completion lifetime checks as codes, scales and zeros.
Mapped columns choose metadata per element; contiguous columns retain the
existing half-tile path. A missing map refuses before enqueue and returns its
operands. The mapped dense case passes on SM120 and both SM86 devices.

The closure audit also found that “expert interleaving” existed only in a
format comment. The inspected pinned integer artifacts store rank-two modules,
so a leading expert axis now receives a specific refusal instead of being
flattened or guessed. AutoRound coverage now combines a real group map with
multiple 16-bit overrides, including MTP, and checks the override families in
both pinned real configurations.

Read-only review found no mapped-path soundness defect, but caught the other
half of changing a kernel call: adding the group-map pointer changed the launch
ABI while its version and versioned symbol still said v1. The catalogue and
fatbin are built together today, so this could not cross-wire the measured
binary, but an ABI field is a compatibility claim rather than an ornament.
Both advance to v2, and the three-GPU dense suite passes after the rebuild.
Review also caught two comments that still described the old unmapped-only
path. When a restriction is removed, search prose and capability records for
the old refusal alongside the production branch.

## 2026-09-19 — freeze M3 evidence on one source identity

The final publication and dense-execution mutation batteries ran from clean
detached worktrees at the same candidate, `0672d24`. T0006 caught all 63
declared defects and held three controls; T0028 caught all 24 defects and held
one control. Both repeated every clean lane three times before and after,
reported no survivor, unstable, invalid, broken or skipped verdict, and restored
clean trees. Targeted survivor runs helped locate missing tests, but only these
complete final-tree batteries close their gates.

The closure pass also reconciled the living capability records. Early README
and task prose still said the repacker, W8 execution, map consumer and override
paths were missing after those paths existed. A final gate is incomplete while
the entry point sends the next builder toward already-finished work.

The repository owner accepted the complete M3 packet on 2026-09-19 and directed
the project to M4. The accepted boundary remains operation-level integer import,
publication and dense/grouped execution over test inputs. It does not promote a
partial artifact into checkpoint model support, convert correctness evidence
into a performance claim, or fire experiment0007's model-execution trigger.
Task0037 starts M4 with common paged device attention and actual 32,768-row
state rather than a model-local attention loop or a nominal width setting.

## 2026-09-19 — a bound is an input away from not being a bound

Task 0037's first change was supposed to be a tidy-up: `attention_error_bound`
derived `1/sqrt(head_dim)` internally while the operation has carried an
explicit score scale since task 0003. It is not a tidy-up. `Δs` bounds the error
of the **scaled** score, the term passes through an exponential, and a
Gemma-style layer normalizes its queries and keys per head and then declares a
scale of exactly 1.0 — so at head dimension 128 the derived value was 11.3 times
too small. The fixture that now holds the case measures an error of 1.94e-3
against a derived "bound" of 1.38e-3. A kernel qualified against the old helper
could have been accepted while computing something the bound forbade.

The shape worth carrying: a bound is a statement about a computation, and every
parameter of that computation it derives instead of receiving is an assumption
about a caller it has never met. Two earlier reviews disproved this same
function's *formula*; this one was its *input*, and no amount of care about the
formula would have found it.

## 2026-09-19 — the gate that was not being run against the ABI it launched

The GPU lane was failing at HEAD, before M4 touched anything, and had been since
task 0035. `cargo xtask-cuda test-gpu`'s hand-written `affine_linear` case
builds its launch parameter array by hand. When task 0035 versioned the affine
ABI to v2 and added the `group_index` operand, the executor's binding was
updated and this one was not: the array stayed at thirteen entries, so the
**output pointer was bound to the kernel's map parameter**, the kernel read group
identities out of its own output buffer, and the resulting illegal access
poisoned the process context. Every later case, on every later device, failed
with an inherited "illegal memory access" — 34 failures from one wrong index.

Two things hid it. `moxie-executor`'s own device tests use the production
binding, so they passed throughout and the failure looked like a hardware or
environment problem. And the second bug found the same day has the same shape:
`grouped_device`'s negative fixture selected its descriptor by operation and SM
alone, and task 0035 also added `affine-expert-*` descriptors for the same
operation and SM that sort **before** the BF16 one, so the fixture had been
handing the planner a quantized descriptor for a BF16 plan and failing before it
mutated anything.

Both are the same lesson, and it is not "write more tests". A hand-written
duplicate of a production binding is a second implementation of an ABI, and
`find(operation, sm)` is a second implementation of kernel selection. Neither
was updated when the thing it duplicated changed, because nothing links them.
When a task versions an ABI or adds catalogue entries, grep for every place that
builds a parameter array or picks a descriptor by hand — the compiler will not,
and a green executor suite will not tell you either.

## 2026-09-19 — pages are storage, tiles are scheduling

The first paged attention kernel keeps its online-softmax tile (128 keys)
deliberately independent of the page width (16, 32, 256 in the gates). A tile
that matched the page would never cross a page boundary and never leave a tail
partly masked, which are the two cases a paged kernel exists to get right. The
same reasoning put a **reversed** page table in every device case: an identity
mapping produces the right answer even when the table is ignored, so it proves
nothing about the indirection it is there to exercise.

The merge's sharp edge is the empty partial. A fully masked tile has a maximum
of `-inf`, and `exp(-inf - (-inf))` is `NaN`, so the identity of the merge has to
be handled *around* the arithmetic rather than through it. A sliding window
reaches that case as soon as the window has moved past a whole tile — which is
to say, always, in production — and the host fixture and the kernel now pin it
from both sides.

## 2026-09-19 — a checked value has to be one nobody can edit afterwards

Review of the first paged attention slice found four defects that share one
sentence: a check is worth nothing if the thing it checked can change, or if
something else is trusted in its place.

`PagedAttentionLaunch` had public fields and a public `check`. Every caller did
call it, so every test passed — and an instance edited after the check, or built
by a caller that skipped it, would have carried positions the kernel indexes
with. `AffineLaunch` learned this in task 0028 and the lesson did not travel.
The fields are private now, with `new`, `at` and `over` as the only ways in.

The same shape, twice more. The binding checked a descriptor field by field and
then loaded **this build's** fatbin regardless, so a descriptor declaring another
layout, accumulation policy, rounding profile or image would have run on code
that declares something else; `grouped_device` already required whole-catalogue
membership for exactly that reason, and the new module reimplemented the weaker
check instead of the stronger one. And the ABI's `u32` window conversion sat
*after* the query copy, so a refusal that was knowable before any device work
became an unknown submission and a quarantined run.

The fourth was the oldest rule in this workspace: a refusal path that allocates.
Handing an append's rows back joined them with `Vec::append`, which reallocates,
on the path whose entire purpose is to report that something could not be done.

Worth carrying: when a new module sits beside an older one that solved the same
problem, the older one's *strongest* check is the specification, not its shape.
Read what it refuses, not what it computes.

## 2026-09-19 — the second review, and what "checked before enqueue" has to mean

The ABI-width repair moved every `u32` conversion into the launch constructor,
and it was still incomplete: CUDA's grid `y` and `z` stop at **65,535** while
`x` reaches `2^31 - 1`, so a head count that fits a `u32` can still be
unlaunchable. A launch with 65,536 heads admitted, copied its query, and would
have learned the truth from `cuLaunchKernel` — the exact shape the repair was
written to eliminate, one field over.

The fix is not a bigger constant. A launch is a host value and cannot know a
device's limits; a run is bound to one device and can, so `DeviceCapability` now
carries the **queried** maximum grid dimensions beside the peer-access flags it
already discovers rather than assumes, and admission refuses a geometry the
device cannot launch before charging anything for it.

Two smaller lessons from the same round. A membership test that rebuilds a
catalogue allocates a `Vec`, two `String`s per descriptor and a `format!` per id
— on the path that exists to refuse under memory pressure; the identity question
now has an allocation-free predicate pinned to the catalogue by a fixture, which
is the same "two statements of one fact, asserted equal" shape `bf16_round` uses.
And a test comment claimed 96 was not a multiple of the warp width. 96 is 3x32.
The case tested a non-power-of-two width and nothing else for a week, because
the comment was read as the coverage. Numbers in test prose are claims, and
review checked this one by dividing.

## 2026-09-19 — "no `format!`" is not "cannot abort"

The third review round on the same module. Every refusal had been routed through
a `try_reserve` sink and `format!` was gone — and the shortest refusals could
still abort, because `invalid(field, "a launch with no query row")` ends in
`detail.into()`, and converting a `&str` into a `String` allocates infallibly.
The repair had been checked by grepping for `format!`, which is a proxy for the
property rather than the property.

Two things follow. The helper that takes `impl Into<String>` is the hazard, not
the interpolation: a fixed message is not free just because it is constant, and
the only way to be sure is for every path to end in the same fallible sink.
And a sweep that exercises the interpolated refusals is not a sweep of the
module — the literal ones are a different code path and were the ones failing.

The control is what settles it, and it is worth stating that it was run rather
than reasoned about: restoring `detail.into()` kills the test binary at
`memory allocation of 26 bytes failed`, `signal: 6, SIGABRT`. A regression for an
abort that has never been seen to abort is a test of nothing.

## 2026-09-19 — the ring was already a page table

Task 0038's handover called it an open design question: the host KV store
reclaims a windowed layer by ring overwrite, the device path reclaims by whole
pages through a page table, and something had to reconcile them. Reading
`PagedSequence::ranges` settled it in one line — it places row `r` at
`page = (r / page_tokens) % pages`, which *is* a page table, `L → L % pages`.
There was never a second scheme to reconcile, only the same one at two
granularities, and the device path now writes it down rather than inventing it.

What the reading did surface is a real mismatch the handover had not seen. The
ring's eviction boundary is row-granular: at capacity 64 and high water 100,
physical page 2 holds rows 96..99 in its first four slots and rows 36..47 in its
last twelve. No `history_base` describes a page like that, because a page table
addresses whole pages. So a device page has to leave the retained range as soon
as *any* of its rows would be overwritten — which costs up to a page of capacity
and is why the admitted rows round up by one page. That rounding is not slack,
and the sweep that proves it (six windows against three page widths, 300 appends
each, asserting the base never passes the oldest row the window admits) is what
turns "should be enough" into a checked property.

The lesson for the boundary itself: the crate that owns retention cannot own
device bytes, and the crate that owns device bytes must not own retention. What
passes between them is neither — it is a *placement*, so it lives in the
vocabulary crate both already depend on, and the executor gained no new edge at
all. The check the performer keeps is the one about bytes it wrote; the check
the authority keeps is the one about rows it committed. Neither subsumes the
other, and a launch that passes both is reading rows that are both history and
present.

## 2026-09-19 — a plan whose arena is the whole truth about its memory

Lowering attention meant deciding what a resource plan says about KV pages, and
the first instinct — put them in the arena like everything else — is wrong in a
way worth writing down. Activation slots are per-step and reused across nodes by
liveness; KV pages are persistent, they outlive every step, and their size comes
from retention and context rather than from the graph's shapes. Charging them to
the step arena would make every decode appear to allocate a cache it inherited.

So the plan *reports* them. `StateRequirement` is what one attention node needs
from the state authority, and a caller checks the authority holds a layer of
that shape before binding. The property that makes this honest is that the plan
no longer claims its arena is the whole truth about its memory — for an
attention graph it never was, and a plan that silently omitted the pages would
have been read as saying otherwise.

The other half of the day: `attend_into` takes device ranges, which is what
document 04 asked for all along — "host reference paths are explicit separate
implementations, not compulsory staging interfaces". The word that carries the
weight there is *separate*, and the only way to keep two implementations honest
is to assert they compute the same thing. They share one precondition check and
one launch, and a fixture asserts byte equality between them. A separate
implementation that computed something else would satisfy the sentence and break
the contract.

## 2026-09-19 — three checks that each looked at one third of the thing

The review of the device KV binding found the same shape three times, and it is
worth naming because every individual check was correct.

Writes took placements, the page table took any non-aliasing permutation, and a
launch checked the table's *length*. So a caller could write rows through one
mapping, publish another, and attend against rows that were never written there
— and nothing was wrong with any of the three checks. What was missing was that
they were checks of three separate facts that only mean something together. The
repair is one view: a table carries the base its logical page zero names, a
write is checked against that view, a republication must agree with the view it
replaces where both describe written rows, and a launch must name the base the
view describes.

The same shape in the authority: `placements` took a position, `publish` took a
count, and neither knew about the other, so an empty sequence could place row
100, write it, publish one row, and leave the frontier at 1 with a high-water
mark at 101. Positions are the authority's to choose now, and a staged batch is
one value carrying both the positions and the count.

And the subtlest: the retained base was derived from the published frontier, so
a *tentative* append moved it and an abort left it moved. Eight rows the window
still admitted became permanently unreadable because of a transaction that was
rolled back. The base now comes from the committed frontier plus the admitted
undo headroom — the worst a transaction could do, priced in before it starts, so
nothing it does can move it. That is what the headroom was admitted for, and it
took a review reproducing it at window 24 to notice that the code was spending
it twice.

The lesson is not "add a check". All three were the same mistake: an invariant
that spans several operations cannot be enforced by validating each one against
its own arguments. It has to be expressed as a value they all have to speak
about — a staged batch, a published view — so that the states that violate it
cannot be constructed.

## 2026-09-19 — a record that agreed with itself in one sentence and not the next

A third review of task 0038 found `AGENTS.md` and the support matrix each
asserting, within one entry, both that `moxie-state` now owns the device pages
and that it does not: the M4 status line said the binding was missing, one
sentence of the support matrix row said the state authority already places
every row, and a later sentence in that same row said `moxie-state` does not
yet own these pages. Both halves had been true at different points in the same
day and nobody deleted the half that stopped being true when the other landed.

The task record had a second, sharper version of the same shape. It described
`abort_truncate_and_reappend_hold_on_device` and
`a_wrapped_ring_answers_exactly_as_an_unwrapped_one` under a heading reading
"evidence that was missing and now exists," which is true of each test's own
claim and false of acceptance criterion 2's conjunction: append, attend,
abort, truncate and re-append together, with a reclaimed `history_base > 0`,
on both SM86 GPUs and SM120. One test has the transaction shapes at full
retention on one ordinal; the other has a reclaimed base at append/attend only,
also on one ordinal. Two tests each satisfying part of a conjunction is not the
conjunction, and "the evidence now exists" read as if it were. The disposition
this review reached — the record's claim that task 0038 was "one wiring step
from done" is not supported by the current code — was really this same error
one level up: two open items summarized as the smaller of the two.

The standing lesson is the same one three checks each looking at one third of
the thing taught a few entries up, applied to prose instead of code: a
conjunction of conditions cannot be reported as satisfied by citing evidence
for each condition separately, and a claim that was true before a later change
landed has to be deleted, not left beside the sentence that supersedes it.

## 2026-09-19 — a fix that moves ownership into a new return type and hands it straight back out

A second-round review rejected the tree the round above tried to close.
`PagedCloseRefused` was supposed to be the fix for the original
external-range lifetime bug: quarantine an ambiguous query or output range
instead of releasing it, and return the run so the caller cannot reuse
memory that may still be in flight. But `close` `take`s that same range out
of `held_ranges` and puts it on `PagedCloseRefused` even when the run is
quarantined, so a caller holding the refusal can release or reuse exactly
the bytes the refusal exists to withhold. The type changed; the range still
left the building.

`WriteReceipt` was the same shape one layer up. It replaced an unchecked
write with a value a caller had to produce before publication — except the
constructor was public, the digest named no sequence, layer, device or run,
and one honest receipt could be replayed for every layer. A caller never had
to observe anything to hold a valid-looking one. Ownership had moved into a
type; the type did not require having earned it.

The standing lesson: introducing a return type, a token or a quarantine flag
is not a fix by itself. The question to ask of any such type is not "does a
caller have to produce this to proceed" but "can a caller produce this
without doing the thing the type is supposed to prove happened." If the
answer is yes — a public constructor, a value handed back on every path
including the refusing one, a digest with no identity in it — the type is
decoration on the same hole, not a wall across it.

## 2026-09-20 — a record can be stale in both directions on the same page

A third review rejected task 0038's tree again, and this time the docs were
wrong on both sides of the same question at once. The record said the
`PagedKvWriter` trait had no implementor and nothing called it — true when
written, false two commits later, because `PagedKvWriterAdapter` now exists
and `DeviceKvSequence::append` calls it. Two paragraphs over, `AGENTS.md`, the
support matrix and task 0037 all said the state binding was landed —
`moxie-state` "owns" the device pages — which was never quite true and had
gotten less true since: the authority still does not hand its callback a page
view, so the integration test reimplements the retained-base/page/modulo
arithmetic the authority exists to own, and the public `write_rows`/
`publish_page_table` can write a device page with no transaction behind it at
all. One claim hadn't caught up to code that shipped; the other had run ahead
of code that never fully landed. Both are a record answering "is this true"
from memory of an earlier tree instead of from the tree in front of it.

The reimplemented arithmetic is its own instance of an older shape, "a claim
checked against a copy of itself" (see the recurring-shapes table above),
worth naming here because it is also a documentation habit, not only a
test-fixture one: a description that restates what a component should do,
written by someone who did not go back and check what it now does, is the
same mistake in prose.

The standing lesson, stated flat: a status claim has an expiry date the
moment the code it describes can change, and neither direction is safer than
the other. Writing "not yet implemented" into a record is not a conservative
default that ages gracefully — it goes stale exactly as fast as "implemented"
does, and this round is the proof, since both were wrong in the same file at
the same time. The fix is not to hedge harder; it is to check the current
tree before every claim, in both directions, every round.

## 2026-09-20 — task 0038 regression evidence, commits `b1e9381..6984c17`

Host: `cargo test --workspace`, 1185 passed, 0 failed. Driver:
`cargo test -p moxie-executor --lib --tests --features driver`, 160 passed, 0
failed. GPU: `CUDA_DEVICE_ORDER=PCI_BUS_ID cargo run -p xtask --features cuda
-- test-gpu`, 51 passed, 0 failed, 0 skipped/unmeasured, on all three
devices — sm_86 (both 3090s, devices 1 and 2) and sm_120 (5060 Ti, device 0)
QUALIFIED. `paged_attention` and `paged_attention_32k` pass on every device.
32k-decode on sm_86: max 3.037e-5, rms 5.625e-6, p99 1.519e-5, identical to
the figure recorded before the state binding was rewritten; 32k-append-decode
max 2.953e-5; 32k-sliding max 1.092e-4.

This is regression evidence for the existing 51 GPU cases and the host/driver
suites now running through the rewritten `append`/`PagedKvWriter` path, with
the page table published from the authority's own `PageView` rather than a
caller-reimplemented one. It does not satisfy task 0038 acceptance criterion
2, which needs a combined append/attend/abort/truncate/reappend case with a
reclaimed `history_base > 0` on both SM86 GPUs and SM120; that case does not
exist yet and criterion 2 stays open. No timing or performance claim is made
anywhere in this evidence: O6 and O7 are open, and nothing was timed.

## 2026-09-20 — the final task 0038 gate found an undeclared dependency edge

The implementation and its tests compiled, but `arch-check` rejected the new
optional `moxie-executor -> moxie-state` edge. Document 02 already shows that
direction: state owns retention and frontiers; executor performs its page
decisions. The missing piece was the checker declaration, not a reason to move
state ownership into the executor or weaken dependency checking. The allowlist
now names that one edge and still rejects the reverse `moxie-state -> CUDA` and
`moxie-state -> executor` directions.

The same pass removed two smaller closure hazards: `append_layer` no longer has
a caller-reachable `expect` if a completed batch cannot publish, and the public
attention-bound oracle refuses an empty visible history instead of returning a
number for an attention operation that cannot exist.

## 2026-09-20 — task 0037 closes; its remaining scope becomes 0038's own

Task 0037's status line and Result section had drifted from its own body:
the header still called the `moxie-state` binding and the injected-fault
sweeps "open" after the body had already struck the binding through as
closed by task 0038, and after the support matrix already documented the
injected-fault sweeps passing. A record contradicting itself is the same
defect this log has already named once tonight, one level up.

Separately, two items were genuinely still task 0037's own and unfinished:
the task mutation battery (deliberately deferred, since writing one against
a state binding that did not yet exist would measure code the next task
replaces) and host allocator growth across decode steps (checked only as
"admission does not grow," never through the measured-allocator lane).
Neither belongs to task 0037 once task 0038's binding exists to measure
against, so the owner moved both into task 0038's acceptance as items 7 and
8, and accepted task 0037 with nothing left in its own scope open.

Lesson: a task that hands its hardest remaining work to a successor task by
design (task 0038's own contract always named the state binding as its
deliverable, not a debt task 0037 owed) does not need to wait on that
successor's acceptance to close its own scope — but scope actually left
over at the boundary, like a mutation battery deferred for a real reason,
needs to be named as the successor's acceptance item, not silently dropped
between two closed records.

## 2026-09-20 — task 0038's last two gates are measurements

The decode allocation gate could not require an identical call count: the
authority's bounded page table and prefix-lineage vectors grow at capacity
boundaries. It measures the quantities that matter instead — calls, transient
peak, net bytes for each operation, and live bytes at the stable boundary —
against bounds derived from the admitted table, lineage capacity and one
retained output. Over 32 steps it observed 9 calls at worst, a 256 B transient
peak and 1,000 B maximum live growth. A mutation leaking 1 MiB per step fails
the live-byte bound. The measurement also exposed four avoidable calls per
step: the device wrapper was collecting `SequenceState`'s open transactions to
re-check the one ID it already owns. Removing that redundant collection took
the observed maximum from 13 calls to 9.

The final T0037 battery applies 11 substitutions across state publication,
selected-plan wiring, the numerical bound and the CUDA kernel. All 11 are
caught; the four clean lanes pass three times before and after restoration.
[Experiment 0008](evidence/experiments/0008-paged-attention-mutations.md) holds
the exact list. The mutation harness self-test also exposed one stale anchor
left by the earlier shared fallible-allocation refactor; the anchor now mutates
the shared sink actually in use rather than dead implementation text.
