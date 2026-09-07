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
| CUDA toolkit | 13.0, `nvcc` V13.0.88 | `moxie-kernels/build.rs` via `CUDA_HOME`/`NVCC` |
| CUDA runtime lib | 13.0.96 | not linked — see below |
| NVIDIA driver | 610.43.02, **open kernel module, patched** | [topology-p2p.md](topology-p2p.md) |
| Kernel | 6.8.0-100-generic (also built for -138) | |
| Host compiler | GCC 12.3.0 (used to build the driver module) | |
| Target architectures | `sm_86`, `sm_120` | `moxie-kernels/build.rs` `ARCHS` |

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is set in `.cargo/config.toml` for everything
cargo launches. It is a correctness requirement, not a preference: see AGENTS.md.

## Dependency decisions

**The CUDA driver API, not the runtime API.** `moxie-cuda` links `libcuda`
(shipped with the display driver) rather than `libcudart` (shipped with the
toolkit). The driver API is what allows a rank to own its context explicitly,
which document 01 requires: "one rank execution thread per GPU ... isolate unsafe
CUDA state behind rank-owned contexts." `build.rs` deliberately does not search
`lib64/stubs`: the stub links successfully and then fails at run time.

**Hand-written FFI, no Rust CUDA wrapper crate.** See
[ADR 0001](../decisions/adr/0001-hand-written-cuda-ffi.md).

**Third-party crates: one.** `toml` 0.9, used only by `xtask` to parse manifests
for `arch-check`. No production crate has a third-party dependency. Document 03:
"do not adopt a large dependency only from its README."

**No Python anywhere in the build or the product.** Document 03 forbids "a
required Python interpreter in the production server". Python was used during M0
only for read-only checkpoint inspection, and no such script is part of the
build.

## Verified

Both fatbins compile and both architectures execute. `cargo xtask test-gpu`,
2026-09-07:

| Device | SM | axpy_f32 | bf16 round-trip | arch mismatch typed |
|---|---|---|---|---|
| 0 RTX 5060 Ti | `sm_120` | PASS | PASS | PASS |
| 1 RTX 3090 | `sm_86` | PASS | PASS | PASS |
| 2 RTX 3090 | `sm_86` | PASS | PASS | PASS |

`bf16 round-trip` compares the device's `__float2bfloat16` against the host
oracle in `moxie-format::bf16` bit for bit, over exact ties, subnormals,
infinities and the zeroes. `arch mismatch typed` loads an intentionally
SM86-only fatbin and asserts it is *refused* on the SM120 device with
`UnsupportedKernel`, and accepted on the SM86 devices.

## Known gap: the host CI lane needs the toolkit

`.github/workflows/ci.yml` runs fmt, `arch-check`, clippy and
`cargo test --workspace`. None of these needs a GPU, but `moxie-kernels`'
build script needs `nvcc`, so the workflow cannot run on a stock
`ubuntu-latest` runner as written. It is **unverified**: no runner is configured
and it has never executed.

Options, none chosen yet: install the pinned toolkit in CI; put the kernel build
behind a feature so the pure crates test without it; or run the lane on a
self-hosted machine. Feature-gating is the tempting one and is the one to be
careful with — a default-off kernel build means CI stops noticing when the fatbin
breaks, which is precisely the "skipped lane treated as passing" that document 07
forbids.

## Not pinned yet

No upstream kernel source has been adopted, so nothing from document 08's table
(vLLM, FlashAttention, FlashInfer, Marlin) is pinned here. When one is, this file
records the commit, the license and the exact architecture coverage — document 08
requires pinning adopted code, "not moving `main` URLs".

Also unrecorded: the build's own executable hash. Document 07 requires it in a
benchmark manifest; no benchmark has run.
