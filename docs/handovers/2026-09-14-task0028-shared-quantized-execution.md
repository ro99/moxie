# Handover — task 0028: a canonical INT4 tensor executes

**Implemented, not reviewed, not accepted.** M3 item 3's dense half: one shared
kernel multiplies a canonical affine INT4 or INT8 weight on both architectures,
and the module task 0025 published is the first checkpoint weight this
repository has ever computed with.

**Read this before quoting anything from it.** No timing was taken and none may
be quoted — O6 and O7 are open. The activations are synthetic, one projection of
one layer ran, nothing composes a block, and no token was generated: output
quality is O2 and needs paired output against the released model. And this does
**not** justify offline repacking ([ADR 0027](../decisions/adr/0027-repacking-is-provisional-pending-measured-inference-benefit.md)),
and it does **not** make [experiment 0007](../evidence/experiments/0007-offline-versus-load-time-preparation.md)
runnable. That experiment's trigger needs shared execution that can run **a
model** and enough checkpoint infrastructure to load and run **both sides**
honestly; one dense projection is neither. The trigger has not fired.

## Workspace identity

- `/home/rodrigo/Developer/moxie`, branch `main`, built on `714946b`.
- `/fast/models` is a **read-only** input. One module of
  `cyankiwi/Laguna-S-2.1-AWQ-INT4` rev `bc59f497` was republished into a
  temporary directory that the test removes when it ends; nothing under a
  checkpoint root was written and no checkpoint was bulk-converted.

## What exists now

```text
crates/moxie-kernels/cuda/affine_linear.cu        moxie_affine_linear_v1
crates/moxie-kernels/src/lib.rs                   affine_linear_catalogue(), 4 identities
crates/moxie-executor/src/affine_linear.rs        AffineLaunch, selection, AffineLinearRun
crates/moxie-executor/tests/affine_linear_device.rs
crates/moxie-executor/tests/affine_linear_real_module.rs
xtask/src/gpu.rs                                  affine_linear_w4a16_w8a16
```

**One symbol, four catalogue identities.** W4A16 and W8A16 are two entries over
the same code on each of SM86 and SM120; the code width, the group rule, the
zero-point section and the scale encoding are all runtime parameters. A test
fails if any descriptor ever names a second symbol, because that is the moment
"shared path" would stop being true.

**The forbidden shortcut is not present and was never needed.** One warp
computes a 16x16 output tile; per `k` tile it unpacks and dequantizes a 16x16
weight tile into **shared memory** and multiplies it with `wmma` BF16 fragments
accumulating in FP32. The catalogue's workspace expression is `Zero`. The real
module's whole launch holds **1,990,656** device bytes where one BF16 copy of
that weight alone would need **6,291,456** — a memory bound, not a speed.

## The three things worth knowing before extending this

**1. The 16-wide tile and the group set are coupled across two crates.** The
kernel converts one scale per `(row, k tile)`, which is only correct while a
group boundary cannot fall inside a tile. Group sizes 32 and 128 are multiples
of 16, so it holds — and `every_allowed_group_size_fits_the_tile` in
`moxie-executor` fails in the **host** lane if `ALLOWED_GROUP_SIZES` widens
without revisiting the kernel. There is a runtime refusal behind it for the same
reason. Do not widen the closed set casually.

**2. The numerical gate has two clauses, and the owner added the second.** The
contract's predeclared 2 ULP of BF16 at the oracle's magnitude **cannot be met
in general by any reordered FP32 reduction**: on an output that has cancelled,
the denominator collapses. Measured at 33 by 1,024 by 3,072 with uniformly drawn
codes, 37 of 101,376 elements missed it at worst 6 ULP, while the kernel's error
against that reduction's own term sum was 1.6e-8 — below one FP32 epsilon. The
owner ruled on 2026-09-14 to accept an element that is within 2 ULP **or**
within `2^-8 · Σ|x_k · W_k|`.

Two things about that clause matter to whoever touches it. It is a **widened
gate**, so narrowing or widening it again is the owner's call, not a task's. And
**it fires nowhere in this task's fixtures** — every element of all five
synthetic cases and of the real module passes the first clause alone, worst
2.000 ULP — so it is exercised by a direct test with the measured numbers rather
than by hoping a seed reaches it. If you add a case and find the clause firing
often, that is worth reporting, not absorbing.

**3. Two deviations from the task contract, both recorded and both deliberate.**
The executor code is a new module rather than `chain.rs`, because `chain.rs`
binds task 0012's fixed three-node BF16 graph and a quantized linear has a
different operand set. And `moxie-executor` now depends on `moxie-format` for
the canonical descriptor vocabulary; the edge is declared in `arch-check` with
its reason, carries descriptors only, and was already in the graph beneath
`moxie-storage`.

## What is refused rather than approximated

- A tensor carrying an **activation-order group map**. ADR 0027 makes
  `actorder: static` a continuation; reading a permuted tensor as contiguous is
  a plausible wrong answer no tolerance catches.
- A **group size the tile cannot honour**.
- A component whose **resident range is shorter than its descriptor implies** —
  checked before it becomes a pointer, so a short component is a refusal rather
  than a read into the next tensor.
- A symmetric tensor handed a zero-point section, or an asymmetric one handed
  none.

## Verification

Measured on this tree with all three GPUs present:

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, and the
  same with `--features moxie-executor/driver`: clean.
- `cargo xtask spec-check`: 10 documents present and unchanged.
- `cargo xtask arch-check`: 79 rejected fixtures, 21 accepted, 13 rules, zero
  failures.
- `cargo xtask-cuda test-gpu`: **45 passed, 0 failed, 0 skipped**; `sm_86` and
  `sm_120` both QUALIFIED.
- `cargo test -p moxie-executor --features driver --test affine_linear_device`:
  five cases on three UUIDs, worst 2.000 ULP.
- `cargo test -p moxie-executor --features driver --test affine_linear_real_module`:
  the real module published in 59.5 s and executed on three UUIDs, 9,216 output
  elements per device, worst 2.000 ULP.
- `cargo xtask mutation-check --battery 0028`: **7 of 7 mutants caught, 1 of 1
  expected survivor held**, 0 unstable, 0 broken controls, `git status` clean
  afterwards. The substitutions are plausible wrong *answers* — a nibble pair
  read backwards, a zero point never subtracted, one group's scale used for
  every group, BF16 scales decoded as F16, the weight tile loaded untransposed
  — which is why the battery needs device lanes and why a tolerance alone could
  not have caught any of them.
- `cargo test --workspace --locked --offline`: **1,085 passed, 0 failed, 0
  ignored**.
- `cargo test --workspace --features moxie-executor/driver --locked --offline`:
  **1,125 passed, 0 failed, 0 ignored**.
- `cargo xtask mutation-check --battery 0006`: **UNMEASURED, not passed.** See
  below.

## One thing to fix that is not this task's

**`moxie-repack`'s `budget` lane cannot be trusted under load**, and the T0006
mutation battery refused to run because of it. It measures peak live heap
through a global allocator; its lock stops one test resetting the other's peak
but not the other test's live bytes being counted into it. Two baselines on an
identical clean tree called it "fails" and "disagrees with itself"; run alone it
passes ten times out of ten. The lane now runs with `--test-threads=1`, which is
the isolation the measurement already assumes and not a weakened assertion. The
real fix is in `crates/moxie-repack/tests/budget.rs` — those two tests should
not share a process-wide counter — and that crate is task 0027's, deferred.

With the lane serialised the battery reached its substitutions and then **ran
past a two-hour limit and was killed mid-substitution**, leaving a mutation live
in `crates/moxie-repack/src/write/run.rs`. The recovery that exists for exactly
this worked: the next invocation restored the file, cleared the marker and said
so, and `git diff` confirmed the substitution was the only thing the kill left.
The battery was **not re-run** — it measures the repack path, which this task
did not touch, and the battery that measures this task's work passed. T0006 is
therefore **unmeasured on this tree**, and a full run belongs with the
`budget.rs` fix in whoever picks up `moxie-repack`. Budget four hours or more,
and do not run anything else on the machine while it goes.

## What is next, and what is not

**Next in M3 item 3: the quantized expert path.** Nothing here touches MoE — the
grouped expert kernel is still BF16-only, so Laguna and every other quantized
MoE still does not execute as a model. That is the largest remaining gap and it
is a task of its own, not an extension of this one.

**Not next, and worth saying so:** performance. The temptation after a kernel
lands is to tune it and quote a number. O6 and O7 are open, this kernel makes no
occupancy or throughput claim, and the first performance artifact this
milestone needs is experiment 0007's comparison, not a faster tile — and that
experiment cannot start until a *model* runs, which is further away than one
working kernel makes it feel.
