# Task 0028 — M3 item 3: shared W4A16 / W8A16 execution

Status: **implemented, reviewed four times, not accepted, one finding open**
(the shared admission vocabulary aborts; [task 0029](0029-allocation-fallible-admission-vocabulary.md)). Contract written before
implementation; the result below is filled from the runs, not from the plan.

## Identity and authority

- **Task ID / milestone / owner**: 0028, M3 item 3 of
  [the roadmap](../spec/06-implementation-roadmap.md): *"Implement shared
  W4A16/W8A16 dense and expert paths for SM86; qualify SM120 separately.
  Preserve BF16 and explicitly qualified FP16 activation/scale semantics.
  Bounded reference dequantization is not an acceptable final fast path by
  assertion."* Owner decision of 2026-09-14 makes this the next work and states
  that accepting task 0027 is **not** a precondition.
- **Writable root / base**: `/home/rodrigo/Developer/moxie`, branch `main`,
  base `ed8ac11`. No dirty paths at start.
- **Read-only**: `/models` and `/fast/models`. This task reads checkpoints; it
  converts nothing and writes nothing into either root.
- **Owner gates**: O2 (bit-identical repack) and O5 (user-managed storage) are
  resolved and unchanged by this work. **O6/O7 (performance) remain open and
  are not touched**: this task establishes *correctness and coverage*, not
  speed, and no result from it may be recorded as a performance claim.
- **It is also what makes [experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md)
  runnable**, which is what decides whether repacking is retained at all
  ([ADR 0027](../decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)).
  Running that experiment is **not** in this task's scope; making it possible is
  a consequence, not a deliverable.
  **This bullet was wrong and is left as written** — a contract is a record of
  what was promised, not a place to edit history. Experiment 0007's trigger
  needs execution that can run a *model* and both sides loadable honestly; this
  task delivers one dense projection and fires neither. See the result below.

## Bounded deliverable

- **One concrete outcome**: a dense linear whose weight is a canonical affine
  INT4 or INT8 tensor executes on a real SM86 device, through the shared
  operation path, and matches an independent CPU oracle within a predeclared
  threshold.
- **Sole owning shared component**: `moxie-kernels` owns the device image and
  its catalogue descriptors. `moxie-executor` owns binding and launch.
  `moxie-types` already owns the precision and descriptor vocabulary
  (`ExecutionProfile::W4A16` / `W8A16`, `KernelOperand::Weight`) and is
  **not** expected to change.
- **Allowed files**: `crates/moxie-kernels/cuda/*.cu`, `crates/moxie-kernels/src/`,
  `crates/moxie-executor/src/chain.rs`, their tests, and the support matrix.
- **Non-goals, explicitly**: expert/MoE quantized paths (a following task);
  SM120 qualification (separate, below); FP16 activations; W4A4/W8A8; any
  attention change; any performance tuning or claim.
- **Forbidden shortcut, named because the roadmap names it**: dequantizing a
  whole weight to BF16 on device and calling the existing BF16 kernel is
  acceptable **only** as a declared correctness stepping stone, recorded as
  *not the fast path*, and it does not satisfy this task. The delivered path
  unpacks and dequantizes into 16-bit tensor-core tiles inside the kernel, with
  no whole-tensor BF16 copy resident.
- **Second consumer**: the same kernel must serve a group-32 asymmetric INT4
  tensor and a group-128 symmetric INT8 tensor — two different group rules and
  two different zero-point sections — or it is one checkpoint's kernel wearing
  a shared name.

## Contract before implementation

- **Equation**: `W = (Q − Z) · S` per [ADR 0023][adr23]'s canonical layout, then
  `y = x · Wᵀ`. `Q` is the packed code, `Z` the i16 zero point of the code's
  group (0 when the tensor is symmetric and has no zero-point component), `S`
  the group's scale.
- **Components, and their shapes**: a canonical affine tensor is three device
  buffers, not one — `codes` (`U8`, `out × ceil(in/2)` for INT4; `I8`,
  `out × in` for INT8), `scales` (`out × groups`), and `zero_points` (`I16`,
  `out × groups`, **absent** when symmetric). The INT4 low nibble is the
  **earlier** input column. Group rules are contiguous-32, contiguous-128 or
  per-channel.
- **Precision and accumulation**: BF16 activations, FP32 accumulation
  (`AccumulationPolicy::Bf16InF32Acc`), `RoundingProfile::FinalBf16Rne` at the
  output boundary. Scales are read in the source's own encoding and converted
  once per group, not per element.
- **Hardware**: SM86 first. SM120 is **qualified separately** and a passing
  SM86 run is not evidence for it; the support matrix gets one row per
  architecture actually exercised.
- **Memory and lifetime**: the three components are admitted through the
  existing ledger and residency authority (task 0020) as one logical weight
  with three ranges. No component is readable until its copy is **observed**.
  Peak device memory must not include a dequantized copy of the weight.
- **Cancellation and failure**: unchanged from the selected chain — a refused
  admission is a typed refusal before any allocation, and a cancelled launch
  retires its lease and returns every charge.
- **Independent oracle**: `moxie_format::affine::Affine::reconstruct` on the
  host, which is task 0024's exhaustively tested decoder, followed by a CPU
  BF16 matmul with FP32 accumulation. The oracle and the kernel share **no**
  code path: the oracle reconstructs the whole weight and multiplies; the
  kernel never materializes one.
- **Predeclared threshold**: for each output element, `|y_kernel − y_oracle|`
  ≤ 2 ULP of BF16 at the oracle's magnitude, and the **exact** integer codes
  and group indices the kernel used are checked separately against the decoder
  on a small shape, so a numerical near-miss cannot hide a wrong group map.
  A threshold widened after seeing a result is a changed gate and needs the
  owner (AGENTS.md product gates).

## Acceptance

1. **Host tests**: descriptor selection — a catalogue entry for `W4A16` is
   chosen for an INT4 weight and refused for a BF16 one, with the refusal
   naming the mismatch; the group map is computed correctly for contiguous-32,
   contiguous-128 and per-channel on shapes with a non-multiple tail.
2. **Device tests, real hardware, all three UUIDs**: the kernel's output
   matches the oracle within the threshold above, on (a) a group-32 asymmetric
   INT4 tensor with zero points spanning both signs, (b) a group-128 symmetric
   INT8 tensor with no zero-point component. Both shapes include a row count
   that is not a multiple of the tile.
3. **Whole-integer-range coverage**: every INT4 code 0..=15 and a signed INT8
   range appear in at least one tested tensor, with negative and asymmetric
   zero points — M3's exit gate requires this and a random fixture does not
   guarantee it.
4. **No dequantized weight resident**: the measured peak device allocation for
   the launch is below the size a BF16 copy of the weight would need. This is
   a memory-boundedness assertion, **not** a speed measurement.
5. **A real module, read-only**: one module of an already-published canonical
   artifact executes. No bulk conversion, no download, no write under
   `/fast/models`.
6. **Support matrix**: one new gate ID per architecture actually exercised,
   each linked to the command that passed. An architecture that was not run is
   recorded as unmeasured, never as supported.
7. **Gates**: fmt, clippy, spec-check, arch-check, the host suite and the
   device-feature lane all pass; the mutation battery runs **once** on the
   final tree before acceptance, with its result recorded.

## Exact condition requiring owner direction

- If the only path that meets the numerical threshold on SM86 is a whole-tensor
  device dequantization, **stop and report**. That is the shortcut the roadmap
  forbids by assertion, and substituting it silently would be the same failure
  this milestone has already produced twice: a claim checked against a copy of
  itself.
- If SM120 needs a different kernel rather than a recompile, that is a separate
  task, not an expansion of this one.

## Result, filled after work

### Corrections after the first independent review

**Seven findings, five P1. All seven reproduced, all seven fixed, none
disputed.** Four are in this task's code; three are in `moxie-repack`'s user
surface, which the review reached while reading the same unpushed batch.

Every one of the four is the **same mistake**: a check that was made once, to a
value nobody had to keep.

1. **(P1) A lease from another GPU was accepted.** `component()` checked the
   authority that issued a lease and the byte length of its range, and never
   the **device** it is resident on. One authority can hold a cache on every
   GPU, each with its own offsets, so a 5060 Ti launch took 3090 leases,
   resolved their offsets inside the 5060 Ti's allocation and returned a
   confident answer over whatever lived there. Task 0021's review found this
   shape one layer down and I did not carry it across. Every lease's scope is
   checked before its address is resolved, and
   `a_component_resident_on_another_device_is_refused` drives two real GPUs
   under one authority.

2. **(P1) A launch that could not prove completion did not keep what it was
   reading.** `run` took the weight leases and the activation source by
   reference. After a failure between the first copy and an observed
   completion, the caller still owned both and could free the source or release
   the leases while submitted work was in flight; quarantining this run's own
   arena protected none of it. The operands are now taken **by value** and
   either handed back — when the refusal happened before anything was enqueued
   — or kept by the run forever, exactly as it keeps its arena.
   `a_quantized_launch_that_cannot_prove_completion_keeps_its_operands` fails
   `cuEventRecord` for real, on real hardware, and asserts that the refusal
   returns no operands, that `close` refuses, that the charge stays outstanding
   and that **the authority cannot close** because its leases are still live.

3. **(P1) An allocation failure aborted the process.** The output readback used
   `vec![0u8; n]`, and every refusal composed its prose with `format!`. The
   review failed one 32-byte allocation and got `SIGABRT`. This is task 0024's
   finding in a new crate: the readback is `try_zeroed`, and this module's
   refusals compose through a `try_reserve`d sink that degrades to an empty
   detail — the variant and the `&'static str` field are what a caller branches
   on — rather than aborting.

4. **(P1) Checked geometry could be edited after it was checked.** Every
   `AffineLaunch` field was public. The review built a valid launch, set
   `row_stride = 1` and `groups_per_row = 0` on a 64-column INT4 tensor, and
   drove it through selection **and** admission: the component-size checks
   became vacuous. The fields are private now, so that edit does not compile,
   and `admit` — a public entry point that accepts a descriptor — re-applies
   the same `descriptor_serves` predicate selection uses, instead of trusting
   what it was handed.

And three in `moxie-repack`, task 0027's deferred code:

5. **(P1) An edited plan published a subset as `complete`.** The binding check
   asks whether the **checkpoint** changed; an edit to the plan leaves that
   answered "no", and the guard downstream read the plan's own `completeness`
   field, which the same edit leaves untouched. `confirm_binding` now recomputes
   coverage from the document in hand against the tensor count the binding
   carries, the way `Discovery::accounted` counts it.

6. **(P2) The plan's size cap applied to the second read, not the first.**
   `read_to_string` had already made an arbitrarily large file resident before
   `read_selection` checked anything. The first read is capped, and the same
   text is now **parsed** rather than read a second time — a second read is a
   second moment, which is round two of this converter's own review.

7. **(P2) Planning deleted a file it had not created.** Any regular file at the
   predictable staging path was treated as an interrupted plan's leftover and
   removed; the review put unrelated data there and lost it to a successful
   `plan`. A path this program did not create is not its to remove, so it
   refuses and names `--force`, which already means "replace an existing plan".

**Each fix has a regression, and each regression is measured by substitution
rather than by assertion**: six new mutations in the `T0028` battery put the
defect back, and every one of them is a plausible wrong answer or a freed
buffer, never a crash.

**That sentence was wrong when it was written, and the re-review said so.** Six
mutations for seven findings is not one each, and the finding with none was the
allocation-failure one — the finding whose failure mode is a `SIGABRT`, which no
mutation in the table could have caught because no lane failed an allocation.
Three of the seven were also still open. Both are corrected below; the paragraph
above is left as written because a record that quietly repairs its own claims is
worth less than one that shows them being repaired.

### The re-review: three of the seven were not fixed

| Finding | What the first repair missed | What closes it |
|---|---|---|
| **2 (P1)**, operand lifetime | `AffineLinearRun` retained the operands and `close` refused while quarantined — but the type had **no `Drop`**, so dropping the refused run ran `Vec::drop` on the activations an asynchronous `cuMemcpyHtoDAsync` may still be reading. The physical half was worse: `DeviceBuffer::drop` called `cuMemFree_v2` even when the `cuCtxSynchronize` that would prove nothing is reading it **failed**. | A quarantined run's `Drop` forgets its activations and its leases. A buffer whose context cannot be synchronized is **permanently withheld** rather than freed. Leaking device memory at teardown is bounded and visible; freeing pages a live copy is reading is a silent wrong answer somewhere else |
| **3 (P1)**, allocation failure | The fallible sink was real, but `descriptor_serves` evaluated `to_string()` on its arguments **before** the sink ever saw them, and other paths — `attribute`, the selection count refusal, three arena labels — still used `format!`. The test that claimed to prove it used an allocation so large that `Vec` rejects it on layout, which never reaches the allocator at all | Every one of those paths composes through `try_reserve`, and a label that cannot be built is a typed refusal rather than an empty string. A new test binary owns a **per-thread one-shot failing allocator** and arms it around the call |
| **5 (P1)**, edited plan | The repair compared the **count** of planned tensors with the bound `index_tensors`. Review swapped one entry for a different tensor the index already named: same count, same digests, and a `complete` artifact carrying one tensor twice and another not at all | The exact `(tensor name, shard)` pairs are compared against the bound index — missing, extra, duplicated and wrongly-bound are each named separately. A total is a shadow of a set, and two different sets cast the same one |

**Measured by substitution, this time including the abort.** Six mutations were
added to the `T0028` battery for this round, bringing it to **19**. Every one
was run and every one is caught:

| Mutation | Deciding lane |
|---|---|
| `refusal-prose-allocated-before-the-fallible-sink` | `allocation-refusal` |
| `the-fallible-sink-grows-infallibly` | `allocation-refusal` |
| `quarantined-run-releases-its-operands-on-drop` | `w4a16-faults` |
| `device-buffer-freed-under-a-failed-synchronize` | `w4a16-faults` |
| `completeness-compares-a-count-not-a-set` | `two-command` |
| `an-edited-plan-publishes-as-complete` (re-anchored) | `two-command` |

**Two of those six survived on the first run**, and finding out was the whole
point of running the battery rather than asserting. `the-fallible-sink-grows-infallibly`
consumed the test's own one-shot allocation trap before reaching the infallible
path, and `completeness-compares-a-count-not-a-set` left the duplicate guard
standing in front of the code it was mutating — so neither substitution
expressed the defect it was named for. Both were rewritten and both are now
caught. A mutation that cannot fail is the same mistake as a test that cannot
fail, one level up.

### The third review: two of those three were still open

Both in the same shape as the round before — the repair closed the path the
reproduction used and left the path beside it.

| Finding | What the second repair missed | What closes it |
|---|---|---|
| **3 (P1)**, allocation failure | The refusal *formatters* were fixed and the **successful** selection was not: `select_affine_linear_kernel` collected two temporary `Vec`s and cloned the winner, all infallibly — on the one path where a caller has no refusal to fall back to. Review armed a valid sm_86 selection and got `memory allocation of 32 bytes failed`. The regression only exercised an unsupported sm_120, where both vectors stay empty and nothing is allocated before the formatter runs | Selection is one pass over the catalogue holding a reference and two counters, and the descriptor it returns is built by `SemanticKernelDescriptor::try_clone`. The regression **sweeps every allocation position** of a selection that succeeds, walking until the trap stops firing |
| **5 (P1)**, edited plan | The exact `(tensor, shard)` comparison is measured **against the binding**, and `confirm_binding` returned success the moment `[binding]` was absent — which the parser allows, because a hand-written selection has none. Review deleted the block, deleted a tensor, and published a `complete` artifact missing it | `--plan` means this program generated the document, so it now requires a binding and says so by name. `--selection` remains the unbound hand-written route |

Three more mutations, all caught on their first run, bringing the `T0028`
battery to **22**:

| Mutation | Deciding lane |
|---|---|
| `selection-collects-a-temporary-vector` | `allocation-refusal` |
| `the-selected-descriptor-is-cloned-infallibly` | `allocation-refusal` |
| `a-plan-without-a-binding-is-left-unchecked` | `two-command` |

### The fourth review: the sweep accepted a corrupted success, and admission was still wrong

| Finding | What the third repair missed | What closes it |
|---|---|---|
| **3 (P1)**, allocation failure, *again* | The claim that `admit`'s own allocations were fallible was **false**. Two remained in this task's code: `hold.push` grew an unreserved `Vec` after the arena exists, and the symbol list was an infallible `clone().collect()`. `Module::resolve_all` underneath them used `with_capacity` and `to_vec` | Those three are fixed. **The path as a whole is not** — see below |
| **(P2)**, the sweep itself | It counted refusals **globally** and required only `refusals > 0`. Review mutated the fallible clone to return `Ok("")` on a failed reservation — a corrupted descriptor, not a refusal — and it still passed, because other positions supplied the refusals it counted | Every firing position must return `capacity_exceeded`; the first non-firing position must return a descriptor **equal** to the catalogue's; exhausting the loop bound is a failure |

**Running that stronger sweep immediately found a real defect**, which is the
argument for it. `descriptor_serves` composed its refusal prose for every
candidate it rejected — including candidates rejected on the way to a match — so
a *successful* selection allocated prose nobody would read, and a failure there
was discarded. The predicate is now a `Mismatch` value that allocates nothing,
and the prose is built from it only at the point a refusal is returned. One
predicate still, read by both the selection scan and the public
`descriptor_serves`: splitting a check into a fast boolean and a separate
explanation is how the two drift, which is a defect an earlier round already
found here.

Two mutations added, both caught, bringing the battery to **24**:

| Mutation | Deciding lane |
|---|---|
| `a-failed-reservation-yields-a-value-instead-of-a-refusal` | `allocation-refusal` |
| `selection-composes-prose-for-candidates-it-rejects` | `allocation-refusal` |

### The fifth review: the admission path aborts in more places than were named

Two records were wrong and one mutation measured the wrong thing.

**The claim that "`admit`'s own allocations are fallible apart from
`PlanRequest`" was itself incomplete**, and this record made it twice. Review
walked the call graph and found infallible allocation in five more places
across three crates — including `Rc::new(ArenaCore)` **after**
`DeviceBuffer::alloc` has succeeded, where an abort leaves a live device
allocation with no owner, and `Arena::allocate`'s `owner.clone()` **after** the
free list has been mutated, where an abort leaves the arena half-updated.
Repairing `PlanRequest` alone would only have moved the first abort further
down the same graph.

The inventory now lives in [task 0029](0029-allocation-fallible-admission-vocabulary.md),
whose scope was also wrong: it named `moxie-memory`'s `request` module as sole
owner and then required a sweep of all of `admit`, which that scope cannot
satisfy. It is rewritten around the whole call graph.

**`selection-composes-prose-for-candidates-it-rejects` was attributed to the
`w4a16-host` lane**, and the battery did report that — because the substitution
changed *which reason* a refusal names as well as putting the allocation back,
so the host lane caught it on refusal text. It was measuring the wrong half of
its own name. The substitution now builds the error and discards it: every
observable value is identical and only an allocation failure can see it. The
deciding lane is `allocation-refusal`, re-run and confirmed.

### The open finding, and two fixes that are unmeasured

**Allocation failure on the admission path still aborts, in at least six places
across three crates.** Armed at position 4, a plan request built exactly as
`resource_request` builds one dies with `memory allocation of 5 bytes failed`;
the rest of the inventory is in task 0029. This is **failing**, not
unmeasured — it has been measured, and it fails.

An earlier version of this record called it "a named gap" and put an empty
`#[test]` in `allocation_refusal.rs` to say so. That test reported `ok` and
counted toward the passing total, which is worse than saying nothing; review
was right to reject both. It is now a comment where a test cannot pass, and
[task 0029](0029-allocation-fallible-admission-vocabulary.md) is the bounded
task that fixes it — written, with the reproduction it starts from.

**The three admission repairs above are fixed but unmeasured** — the reserved
range list, the fallibly built symbol list, and `resolve_all`'s growth. No
mutation covers them and none is in the battery, because an admission sweep has
to get past `PlanRequest::new` first, and then past five further aborts. They are not recorded as expected survivors:
this battery's `Expect::Survivor` means *independence control* — "nothing should
catch this, and something catching it is a finding" — and using it for "nothing
can reach this yet" would encode an untested fix as an expected one. Task
0029's acceptance includes the sweep that measures them.

**Gates on the corrected tree** (2026-09-15, after the fourth round): fmt; both
clippy lanes; spec-check (10 documents); arch-check (79 rejected fixtures, 21
accepted, 13 rules); the mutation self-test at **105 of 105** cases over 90
anchors; **1,099** host tests and **1,143** device-feature tests across 100
suites, with nothing failed, ignored or skipped. The host total is one lower
than the round before because the empty test that claimed the admission gap was
deleted; a passing test that asserts nothing is not a test.

The full `T0028` battery has **not** been re-run end to end on this tree, and
neither has `cargo xtask-cuda test-gpu`. What has been run is each of the
**eleven** mutations the four rounds added, individually, all caught. That is a
narrower claim than a battery pass and it is the only one these corrections are
entitled to; the 13-of-13 result in the support matrix and the handover
describes **`448c9a2`** — after the first round's corrections in `9051f9a`, not
before any of them — and both now say so. An earlier version of this paragraph
attributed it to `cfc1061`, which carried 8 mutations and none of the first
round's; review corrected it.

**A new lane.** `allocation-refusal` runs a test binary with a per-thread
one-shot failing allocator. A process that aborts cannot be observed from
inside itself, so the regression for an abort is a binary that dies — which is
a lane result, not an assertion. The trap is per **thread**: a process-wide flag
was the first attempt and it landed in an unrelated test's `format!`, which is
`budget.rs`'s concurrency bug in a new place.

**One claim in this record was also wrong and is corrected.** It said task 0028
makes experiment 0007 runnable. It does not: that trigger needs execution that
can run a **model** and both sides loadable honestly, and one dense projection
is neither. The contract bullet that first made the claim is left as written,
with a note — a contract records what was promised.

### The work itself

**A canonical INT4 tensor executes.** One module of a published canonical
artifact — 3,072 by 1,024, group-32 asymmetric INT4, the module task 0025
published and nothing ran — was made resident as three components, multiplied by
activations this repository wrote, and matched an independent host decoder on
every one of the three GPUs. That is the first time anything here computes with
a quantized checkpoint weight.

**It is not model support and no quality claim follows.** The activations are
synthetic, the module is one projection of one layer, nothing composes a block,
and no token is generated. Output quality is **O2** and needs paired output
against the released model. **No timing was taken and none may be quoted**:
O6/O7 are open, both lanes are debug or release test builds, and there is no
baseline on this machine.

### What was built

**One kernel, one symbol.** `crates/moxie-kernels/cuda/affine_linear.cu`
defines `moxie_affine_linear_v1`, and W4A16 and W8A16 are two catalogue
identities over it rather than two implementations. The code width is a
parameter, and so are the group rule, the zero-point section and the scale
encoding — the sharing is structural, and
`the_two_quantized_profiles_are_one_symbol_and_four_identities` fails if a
descriptor ever names a second symbol.

**No whole-tensor dequantization anywhere.** One warp computes one 16x16 output
tile. Per `k` tile it unpacks and dequantizes a 16x16 weight tile **into shared
memory** and multiplies it with a `wmma` BF16 fragment pair accumulating in
FP32. Nothing wider than that tile is ever 16-bit, the catalogue's workspace
expression is `Zero`, and both device tests assert the launch's whole device
footprint against what one BF16 copy of the weight would need — 1,990,656 B
against 6,291,456 B for the real module. The shortcut the roadmap forbids by
name was never needed and is not present.

**The scale of a group is converted once per group.** Group sizes are 32 or 128
and a `k` tile is 16 wide at a 16-aligned offset, so a lane's eight columns
always lie inside one group. That coupling spans two crates that otherwise never
meet, so `every_allowed_group_size_fits_the_tile` holds the host lane to it: if
`moxie_format::affine::ALLOWED_GROUP_SIZES` widens without revisiting the
kernel, the host lane breaks rather than a GPU returning a wrong number.

**Two refusals rather than two approximations.** A tensor carrying an
activation-order group map is refused by name — ADR 0027 makes `actorder:
static` a continuation, and reading a permuted tensor as contiguous is a
plausible wrong answer no tolerance catches. A group size the tile cannot honour
is refused for the same reason.

### Two deviations from this contract, both deliberate

**The executor code is a new module, not `chain.rs`.** The contract listed
`crates/moxie-executor/src/chain.rs` among the allowed files before either shape
was known. `chain.rs` binds task 0012's selected three-node BF16 chain — a fixed
graph with a fixed edge shape and a launch order written node by node — and a
quantized dense linear has a different operand set, three device buffers for one
logical weight. Folding them together would make that file two things whose only
shared property is the word "linear". The work is in the same crate under the
same ownership, in `crates/moxie-executor/src/affine_linear.rs`.

**`moxie-executor` gained a dependency on `moxie-format`.** The launch is
derived from the canonical affine descriptor — width, group rule, zero-point
mode, scale dtype — which is `moxie-format`'s vocabulary and nobody else's.
Restating those four things in the executor would be a second description of one
format. The edge is declared in `arch-check`'s allowlist with that reason, it
carries descriptors only (`moxie-format` is I/O-free by rule, so it can never
become a path to a file), and the crate was already in this one's graph beneath
`moxie-storage`. `every_spelling_of_a_forbidden_import_lands_on_one_rule` and
the allowlist test both assert the directions that stay refused.

### The owner ruling this task needed

**The predeclared threshold could not be met in general, and the reason is the
metric.** The contract fixed `|y_kernel − y_oracle|` ≤ 2 ULP of BF16 at the
oracle's magnitude. Measured at 33 by 1,024 by 3,072 with uniformly drawn codes
and zero points, 37 of 101,376 elements missed it, worst 6 ULP. The worst
element's oracle result is **2.21729279e-5** against a term sum of
**44.8413914** — six orders of magnitude of cancellation — and the kernel's
error against that term sum is **1.6e-8**, below one FP32 epsilon. No reordered
FP32 reduction can land within 2 ULP of a result that has cancelled that far;
the denominator collapses, not the kernel.

**Owner ruling, 2026-09-14: add a cancellation guard.** An element passes if the
difference is within 2 ULP of the oracle's magnitude **or** within
`2^-8 · Σ|x_k · W_k|`, the smallest difference a BF16 output of that reduction
could express at all. Both bounds are computed and both are reported. This is a
widened gate and it was the owner's to widen; it was put to him **before** the
acceptance tests were written around either form.

**Where the second clause actually fires: nowhere in this task's fixtures.**
Every element of all five synthetic cases and of the real module passes the
first clause alone, worst **2.000 ULP**. A guard nothing exercises is a shape
this repository keeps producing, so the guard is driven directly instead, by
`the_cancellation_clause_covers_a_cancelled_element_and_nothing_else` — with the
measured numbers above, asserting that the clause covers the cancelled element,
that it still refuses the same difference on a well-conditioned reduction, and
that the fixture still misses the first clause. Fishing for a seed that happens
to exceed would have been fitting the evidence to the guard.

### Coverage, and why each case is there

| Case | Width | Group | Zero points | Scales | Shape |
|---|---|---|---|---|---|
| a | INT4 | 32 | asymmetric | bf16 | 5 x 100 x 96 |
| b | INT8 | 128 | symmetric | f32 | 3 x 300 x 64 |
| c | INT4 | 128 | symmetric | f16 | 17 x 300 x 48 |
| d | INT8 | per-channel | asymmetric | bf16 | 7 x 100 x 33 |
| e | INT4 | 32 | asymmetric | bf16 | 33 x 1024 x 3072 |
| real | INT4 | 32 | asymmetric | bf16 (source's own) | 3 x 1024 x 3072 |

(a) and (b) are the contract's two. (c) and (d) **cross** the group rule and the
zero-point section against the other width: if either were secretly tied to the
code width, one of those two fails. (e) is the geometry that produced the
measurement above. Every row count — 5, 3, 17, 7, 33, 3 — is a non-multiple of
the 16-wide tile, and 100 and 300 input features leave a short final group.

**Whole-integer-range coverage is asserted from the tensor, not from the
generator.** Every third position of the flattened fixture walks the code range
in order; three is coprime to both 16 and 256, so those positions cycle through
every code of either width spread across rows and columns. The test then reads
the codes back through the decoder and requires all sixteen INT4 codes, INT8's
−128 and 127 with more than 200 distinct values, and zero points spanning both
signs.

### The oracle shares no code path with the kernel

The host side reconstructs the **whole** weight through
`AffineTensor::reconstruct` — task 0024's exhaustively tested decoder — rounds
each value to BF16, and multiplies in ascending order with FP32 accumulation.
The kernel never materializes a whole weight at all. For the real module the
oracle's bytes arrive through `Artifact::stream_tensor`, which verifies each
component against its own checksum, while the device's bytes arrive through the
residency authority's chunk reads: the same published file, read by two code
paths that meet nowhere.

### Memory and lifetime

The weight's three components are resident in the **one** production
weight-residency owner (task 0020), acquired as three chunks of one logical
tensor and reaching the launch as `device_address` of a lease's offset. Nothing
in the new module copies a weight, keeps one, or maps chunk to bytes — the
`a second weight-residency owner` rule still passes. Readiness is the authority's
own event-gated upload, so no component is readable before its copy is observed.
Each component's resident range is checked against the length the descriptor
implies **before** it becomes a pointer: a short component is a refusal, not a
read into the next tensor. The per-step activations and output are a separate
admitted `DeviceArena`; a launch whose completion cannot be established
quarantines the run, and `close` refuses while quarantined.

### Gates

Run on this tree, all three GPUs present:

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets` and the
  same with `--features moxie-executor/driver`: clean.
- `cargo xtask spec-check`: 10 documents present and unchanged.
- `cargo xtask arch-check`: 79 rejected fixtures, 21 accepted, 13 rules, zero
  failures — including the new `moxie-executor -> moxie-format` entry and the
  assertions that the reverse edges stay refused.
- `cargo xtask-cuda test-gpu`: **45 passed, 0 failed, 0 skipped**; `sm_86` and
  `sm_120` both QUALIFIED, with the new `affine_linear_w4a16_w8a16` case passing
  on all three devices.
- `cargo test -p moxie-executor --features driver --test affine_linear_device`:
  five cases on three UUIDs, worst 2.000 ULP, plus the cancellation-clause test.
- `cargo test -p moxie-executor --features driver --test affine_linear_real_module`:
  one module published in 59.5 s (nearly all of it hashing the 5.37 GB source
  shard) and executed on all three UUIDs, 9,216 output elements per device,
  worst 2.000 ULP.
- `cargo test --workspace --locked --offline`: **1,091 passed, 0 failed, 0
  ignored**.
- `cargo test --workspace --features moxie-executor/driver --locked --offline`:
  **1,134 passed, 0 failed, 0 ignored**.
- `cargo xtask mutation-check --battery 0028`, on the corrected tree:
  **13 of 13 mutants caught, 1 of 1 expected survivor held**, 0 unstable, 0
  broken controls, and `git status` clean afterwards. Seven substitutions are
  ways of getting a **plausible wrong answer** rather than a crash — an INT4
  nibble pair read backwards, a zero point never subtracted, every group using
  the first group's scale, BF16 scales decoded as F16, the weight tile loaded
  untransposed, a group map off by one, a permuted tensor read as contiguous —
  because those are exactly the defects a tolerance cannot catch by being
  tight. Six more put the review's findings back: another device's lease
  accepted, admission trusting the descriptor it is handed, an unprovable
  launch handing its operands back, an edited plan publishing as complete, a
  staging path deleted without ownership, and the plan read before its cap
  applies. The independence control removes the resident-component length
  check, which no fixture violates, and it held: the numerical lanes are not
  depending on a bounds check for their answer.

### One gate unmeasured, and one finding handed back

**`cargo xtask mutation-check --battery 0006` is UNMEASURED on this tree, not
passed.** Two things stopped it, in that order.

First it **could not run at all**, and could not have run on the previous tree
either. Its `budget` lane measures peak
live heap through a global allocator; the lane's own lock stops one test
resetting the other's peak but not the other test's live bytes being counted
into it, so the number depends on how loaded the machine is. Two baseline runs
on an identical clean tree reported it as "fails" and as "disagrees with
itself", and the battery refused to build verdicts on either — correctly.

The lane now runs with `--test-threads=1`, which gives the measurement the
isolation it already assumes and weakens no assertion. **The fix belongs in
`crates/moxie-repack/tests/budget.rs`**, whose two tests should not share a
process-wide counter at all; that crate is task 0027's and deferred, so this is
recorded rather than fixed here.

That worked — the battery got past its baseline and into the substitutions —
and then it **ran past a two-hour limit I set on it and was killed
mid-substitution**, leaving `if cancelled() { ... }` replaced by
`let _ = &cancelled;` live in `crates/moxie-repack/src/write/run.rs`. This is
the scar the engineering log records, and the recovery built for it worked
exactly as designed: the next invocation printed *"restored
crates/moxie-repack/src/write/run.rs from an interrupted run before measuring
anything"*, `git status` came back clean, and `git diff` confirmed the
substitution was the only change the kill had left.

It was **not re-run**, and that is a decision rather than an oversight. T0006
measures the repack publication path, which this task did not touch; the battery
that measures *this* task's work is T0028 and it passed. A second two-hour-plus
run risks the same mid-substitution kill for a regression check on unchanged
code. **Recorded as unmeasured**, and a full T0006 run belongs in whatever picks
up `moxie-repack` next — together with the `budget.rs` fix, which would make it
runnable without the serialisation workaround.

### What this does not do

- **No expert or MoE quantized path.** A following task; the grouped expert
  kernel is still BF16-only.
- **No FP16 activations, no W4A4/W8A8, no attention change, no tuning.**
- **No performance claim, and it does not make [experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md)
  runnable.** An earlier version of this record claimed it did; the review was
  right that this contradicts the experiment's own trigger, which needs shared
  execution that can run **a model** and enough checkpoint infrastructure to
  load and run **both sides** honestly. One dense projection is neither. The
  trigger has not fired, no partial number from it may be cited, and repacking
  stays provisional.
- **It does not close M3 item 3 by itself**, and it closes no owner gate.

[adr23]: ../decisions/adr/0023-canonical-affine-payload-and-repack-journal.md
