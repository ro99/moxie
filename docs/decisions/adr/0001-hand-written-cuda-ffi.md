# ADR 0001 — Hand-written CUDA driver FFI, not a wrapper crate

- ID / date / author / status: 0001 / 2026-09-07 / M0 / accepted
- Classification: **measured implementation choice** (roadmap-delegated; document 01 delegates internal algorithm and dependency choices within its contracts)
- Scope and owning shared component: `moxie-cuda`
- Supersedes / superseded by: —

## Problem and mechanism

Document 06 M0.4 requires evaluating "Rust CUDA wrapper and upstream kernel
linkage with a minimal allocate/copy/event/launch smoke, including error
propagation". Two shapes are available: adopt an existing crate (`cust`,
`cudarc`), or declare the driver API directly.

Document 02 requires the FFI boundary to validate "sizes, alignment, device
ownership, layouts and capability support", with errors "typed
(`CapacityExceeded`, `UnsupportedKernel`, `InvalidArtifact`, `DeviceLost`,
`Cancelled`), not classified by matching CUDA error-message strings". It also
requires that "Rust `Drop` alone must not free in-flight CUDA memory" (R07).
These are contracts about *our* error taxonomy and *our* lifetime rules, not
about CUDA's.

## Options examined

**Adopt a wrapper crate.** Faster to a first launch. But its error type is its
own, so every call site needs a mapping layer anyway — the typed-error
requirement is not satisfied by having *a* Rust error type, it is satisfied by
having *our* variants chosen from CUDA's numeric codes. Its `Drop` semantics are
its own, and R07's requirement that retirement be event-driven is a property we
would have to verify against someone else's buffer type. Document 03 is explicit:
"do not adopt a large dependency only from its README."

**Hand-written declarations.** The M0 surface is ~30 driver functions. Each
signature is checked against `cuda.h` once, and the crate then owns its error
mapping and lifetime rules outright. Cost is the ongoing discipline of checking
signatures when the surface grows, and the risk of binding an unversioned alias.

## Decision and authority

Hand-written `unsafe extern "C"` declarations in `moxie-cuda::ffi`, linking
`libcuda` (driver API), with the safe wrappers in `moxie-cuda` discharging the
driver's contract. This is an implementation choice inside document 02's stated
boundaries; it needs no owner gate.

Two rules the declarations follow:

- Use the versioned symbol (`cuMemAlloc_v2`, not `cuMemAlloc`). Linking the
  unversioned alias silently binds a different ABI on some toolkits.
- Link the real `libcuda`, never the toolkit's `lib64/stubs` copy. The stub links
  and then fails at run time.

## Evidence and acceptance

- The error mapping is unit-tested without a device: OOM maps to
  `CapacityExceeded`; `NO_BINARY_FOR_GPU`, `INVALID_PTX`, `UNSUPPORTED_PTX_VERSION`
  and `NOT_FOUND` map to `UnsupportedKernel`; the six fatal context codes map to
  `DeviceLost`; `NO_DEVICE`/`INVALID_DEVICE` map to `Unsupported`; `NOT_INITIALIZED`
  maps to `InvalidRequest` so a wrapper defect cannot be misreported as absent
  hardware. The variant is selected from the code alone, and a test asserts that.
- `cargo xtask test-gpu` allocates, copies, launches and reads back on all three
  devices, and asserts an architecture mismatch is refused with the right variant.
- `DeviceBuffer::drop` synchronises before freeing, discharging R07 bluntly. This
  is documented in the type as a placeholder for the event-retained lease, with a
  warning against "optimising" it by deleting the synchronise.

Limitations: the surface is the M0 minimum. Streams and events are declared but
not yet wrapped, so nothing asynchronous is exercised yet — which is exactly
where R07's real difficulty lives.

## Enforcement and removal

`arch-check` denies `moxie-cuda` to every crate except the composition root, and
denies FFI declarations inside model crates (proven by the `model-has-ffi`
negative fixture). `clippy::undocumented_unsafe_blocks` is `deny` workspace-wide,
so every `unsafe` block carries a written justification.

Re-evaluate if the driver surface grows past roughly a hundred functions, or if
a wrapper crate appears that already provides event-retained leases. Adopting one
later is a contained change: it replaces `moxie-cuda`'s internals without moving
the ownership boundary.
