# ADR 0031 — admit expert group maps with their weight leases

- **Date / status:** 2026-09-17 / implemented; mapped execution and cancellation validated on all three GPUs, review pending.
- **Classification:** implementation choice under the authorized task0035.
- **Owner:** shared expert plan, residency source and kernel operand binding.

A canonical artifact keeps an optional logical-column group map in its manifest.
Execution needs those bytes on the device as long as it needs the weight. A
separate map cache would duplicate residency policy; an uncharged device upload
would violate the admission contract.

For the shared expert residency representation, one projection lease contains
its canonical codes/scales/zeros followed by its optional map as K little-endian
u32 values. The published artifact is unchanged. This is a transient resident
operand layout, never another file format. CanonicalSource copies the manifest's
validated map into the end of the already admitted destination; it allocates no
map or weight expansion. The planner adds exactly 4K bytes when a map exists.

Canonical expert bindings use residency format_version2, distinct from earlier
raw-source chunk identities. Artifact identity includes manifest metadata, so a
changed map changes the artifact binding as well. Every runtime consumer checks
full extents; map loads are alignment-safe, since packed sections can end at an
odd offset. CPU and GPU validate/use original logical column identity.

The existing two weight leases retain the maps through upload, launch, drain and
quarantine. Existing eviction and cancellation apply to the entire operand;
there is no additional cache, allocator or lifetime mechanism. Task0035 gates
must cover mapped source publication, both gates, both precisions, both device
architectures and failed admission/cancellation before this is qualified.
