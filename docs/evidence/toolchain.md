# Pinned toolchain and build dependencies — M0

Document 03: "Pin CUDA toolkit, driver minimum, compiler, Rust toolchain, library
commits, build flags and license notices." Document 07 requires a reproducible
build and a recorded executable identity for every benchmark. Changing anything
in this file is an ADR.

## Pinned

| Component | Version | Pinned in |
|---|---|---|
| Rust | 1.97.1 (`8bab26f4f`, 2026-07-14) | `rust-toolchain.toml` |
| Rust edition | 2024 | `Cargo.toml` `[workspace.package]` |
| Cargo resolver | 3 | `Cargo.toml` |
| CUDA toolkit | 13.0, `nvcc` V13.0.88 | `moxie-kernels/build.rs`, **verified at build time** |
| CUDA runtime lib | 13.0.96 | not linked — see below |
| NVIDIA driver | 610.43.02, **open kernel module, patched** | [topology-p2p.md](topology-p2p.md) |
| Kernel | 6.8.0-100-generic (also built for -138) | |
| Host compiler (driver module) | GCC 12.3.0 | |
| Host compiler (nvcc `-ccbin`) | `c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0` | recorded, not pinned |
| Target architectures | `sm_86`, `sm_120` | `moxie-kernels/build.rs` `ARCHS` |

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set in `.cargo/config.toml` for everything
cargo launches. It is a correctness requirement, not a preference: see AGENTS.md.

**The pin is now enforced, not just written down.** `moxie-kernels/build.rs` runs
`nvcc --version` and fails the build unless it reports `release 13.0, V13.0.88`.
Task 0002 F5 was right that "a freely overridable NVCC path alone does not pin a
toolchain": `NVCC=/somewhere/else` previously changed the compiler silently.
Changing the pinned string is an ADR, and the build says so in the panic message.

## Recorded build identity

Document 07 requires a recorded executable/image identity for every benchmark.
`moxie-kernels` now exposes it, and `cargo xtask-cuda test-gpu` prints it at the
top of every run:

| Item | Value |
|---|---|
| `NVCC_VERSION` | `Cuda compilation tools, release 13.0, V13.0.88` |
| `HOST_COMPILER_VERSION` | `c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`, passed to nvcc as `-ccbin` |
| `SMOKE_FATBIN_SHA256` | `014b6a8f1e383c64c970195b663f1d1e2443c5176972d1f2d66a75441ebd7606` |
| `SMOKE_FATBIN_SM86_SHA256` | `e5bfb5aeece31607b5ed962c4ce1f03ebf6a83bd5bb3ca9b08d1771c5f5fd190` |

Hashed with a SHA-256 implementation in `build.rs` itself, checked against the
published FIPS 180-4 vectors at build time. No third-party crate, no shell-out.

The host compiler is **passed** to nvcc with `-ccbin`, not merely described. An
earlier version read `CUDAHOSTCXX` to fill this table in and then let nvcc pick
its own default, so the record was an assumption about the environment. The
digests did not change when the flag was added, which confirms `c++` was already
the compiler in use -- it is now a fact about the build rather than a guess.

Recorded 2026-09-07 on this machine. The digests are of *this* build's images;
they change whenever `smoke.cu`, the architecture list or the toolkit changes,
which is what makes them an identity.

## Both fatbins are SASS only

`-gencode arch=compute_NN,code=sm_NN` embeds cubin for `sm_NN` and no PTX. The
build script previously carried a comment claiming it embedded PTX as well "so a
future architecture can JIT"; it never did. The comment was the error, not the
flags: embedding `code=compute_NN` would let the driver JIT the SM86-only image
onto the SM120 card, and `arch_mismatch_is_typed` would stop asserting anything
while still passing. Corrected in the comment, deliberately not in the flags.

## Dependency decisions

**The CUDA driver API, not the runtime API.** `moxie-cuda` links `libcuda`
(shipped with the display driver) rather than `libcudart` (shipped with the
toolkit). The driver API is what allows a rank to own its context explicitly,
which document 01 requires: "one rank execution thread per GPU ... isolate unsafe
CUDA state behind rank-owned contexts." `build.rs` deliberately does not search
`lib64/stubs`: the stub links successfully and then fails at run time.

**Hand-written FFI, no Rust CUDA wrapper crate.** See
[ADR 0001](../decisions/adr/0001-hand-written-cuda-ffi.md).

**Third-party crates: four, all tooling, all in `xtask`.** `toml` 0.9 parses
manifests; `syn` (pinned `=3.0.3`) with `quote` and `proc-macro2` parse model
crate source structurally for `arch-check`, per
[ADR 0004](../decisions/adr/0004-parse-model-source-with-syn.md). **No production
crate has a third-party dependency**, and `arch-check` enforces an empty
third-party allowlist for every shared crate and every model crate.

All three parsing crates were already in the lock graph via `toml`'s
`serde_derive`, so making them direct dependencies of `xtask` added no package to
the build: `Cargo.lock` gained no `name =` entry. Document 03 warns "do not adopt
a large dependency only from its README"; ADR 0004 records why this is not that,
and what the parser does *not* do.

**No Python anywhere in the build or the product.** Document 03 forbids "a
required Python interpreter in the production server". Python was used during M0
only for read-only checkpoint inspection, and no such script is part of the
build.

## Verified

Both fatbins compile and both architectures execute. `cargo xtask-cuda test-gpu`,
2026-09-07, five cases per device:

| Device | SM | axpy_f32 | bf16 round-trip | arch mismatch typed | stream/event | non-PTX rejected |
|---|---|---|---|---|---|---|
| 0 RTX 5060 Ti | `sm_120` | PASS | PASS | PASS | PASS | PASS |
| 1 RTX 3090 | `sm_86` | PASS | PASS | PASS | PASS | PASS |
| 2 RTX 3090 | `sm_86` | PASS | PASS | PASS | PASS | PASS |

15 passed, 0 failed, 0 skipped. `QUALIFIED sm_86`, `QUALIFIED sm_120`, exit 0.

`bf16 round-trip` compares the device's `__float2bfloat16` against the host
oracle in `moxie-format::bf16` bit for bit, over exact ties, subnormals,
infinities and the zeroes. `arch mismatch typed` loads an intentionally
SM86-only fatbin and asserts it is *refused* on the SM120 device with
`UnsupportedKernel`, and accepted on the SM86 devices.

`stream/event` is the bounded completion smoke ADR 0001 left as a declaration:
two asynchronous copies and a launch on a non-default stream, an event recorded
after them, `is_complete()` true after synchronising, a finite non-negative
elapsed time, and the device result verified against the host oracle. It proves
the mechanism, not overlap, and it is **not** the event-retained lease from R07.

`non-PTX rejected` covers both halves of the PTX boundary. Text with no
`.version` directive, and text beginning with ELF magic, are refused by
`PtxSource` **before any driver call** -- `cuModuleLoadData` sniffs the leading
bytes and picks a parser itself, so without that check a NUL-terminated buffer
holding a binary image reached the image parser through the text path. Text that
*is* shaped like a PTX module but does not compile then reaches the real driver
and comes back as `UnsupportedKernel`. Termination is what the C API requires; it
is not validity, and the case says so.

### The gate fails when an architecture is absent

Task 0002 F5: the previous lane printed a NOTE for a missing architecture and
exited 0, so a CI wrapper could read an all-skipped run as qualification. Every
required architecture must now have **every** case pass on a real device of that
architecture. Demonstrated the same day by hiding the SM120 card:

```text
$ CUDA_VISIBLE_DEVICES=1,2 cargo xtask-cuda test-gpu
10 passed, 0 failed, 0 skipped/unmeasured
QUALIFIED   sm_86: every case passed on a real device
UNQUALIFIED sm_120: no installed device has this architecture
test-gpu FAILED: 0 case(s) failed; 1 required architecture(s) unqualified
$ echo $?
1
```

`--profile sm86` narrows the *required* set to one architecture, which is
document 07's `test-gpu --profile sm86 / sm120` split. It still exercises every
visible device; it only changes what must qualify for the run to pass.

## Closed: the host lane no longer needs the toolkit

Task 0002 F3: even `arch-check` pulled in xtask's unconditional CUDA and kernel
dependencies, so the *host enforcement lane itself* could not run on a stock
runner, and the `moxie-cuda` error-mapping tests linked and called driver
functions.

Two default-off features now split the build:

| Feature | What it adds | Default |
|---|---|---|
| `moxie-cuda/driver` | the `extern "C"` surface, the device types, `-lcuda` | off |
| `moxie-kernels/fatbin` | running `nvcc`, the embedded images, the build identity | off |
| `xtask/cuda` | both of the above, plus `test-gpu` and `probe` | off |

Two cargo aliases, in `.cargo/config.toml`:

```text
cargo xtask <cmd>        host lane: no nvcc, no libcuda, no GPU
cargo xtask-cuda <cmd>   device lane: compiles fatbins, links the driver
```

What is pure without the features: the `CUresult` -> typed error classification
and the UUID formatting, which moved to `moxie_cuda::status` precisely so they
can be tested with no driver present.

Verified on 2026-09-07 by building and testing the whole workspace with
`CUDA_HOME=/nonexistent NVCC=/nonexistent`, an unset `LD_LIBRARY_PATH` and a
`PATH` with no CUDA directory: 222 unit tests and 1 doctest passed. `ldd target/debug/xtask` on the
host build reports no `libcuda`, and the CI workflow asserts that.

**This is not a way to make the GPU lane optional.** Document 07 requires the
separation *and* a distinct mandatory CUDA lane, "so this separation does not
turn unmeasured GPU work into a pass". Three things enforce that:

- a device command run from the host build fails with exit 2 and names the
  command that would work; it does not print a friendly nothing and exit 0;
- `moxie-cuda/driver` panics at build time if `libcuda.so.1` is not found, rather
  than producing an unlinkable build;
- `test-gpu` fails when a required architecture has no passing device (below).

Still true, and still recorded as a gap: **no CI runner executes the device
lane.** There is no self-hosted runner. Its results are the ones in this file,
measured by hand on this machine, and the workflow neither runs it nor claims it.

## Not pinned yet

No upstream kernel source has been adopted, so nothing from document 08's table
(vLLM, FlashAttention, FlashInfer, Marlin) is pinned here. When one is, this file
records the commit, the license and the exact architecture coverage — document 08
requires pinning adopted code, "not moving `main` URLs".

Also unrecorded: the build's own executable hash. Document 07 requires it in a
benchmark manifest; no benchmark has run.
