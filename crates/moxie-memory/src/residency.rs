//! The one authority for which weight chunks are resident, where, and at whose
//! cost.
//!
//! R02 is the failure this repairs, in its second form. Task 0006 fixed the
//! first: legacy's `ResidencyManager` modelled bytes for a simulator while the
//! model runtimes allocated for themselves. The second form is subtler and is
//! what this module exists to make impossible -- *several* caches, each correct
//! about its own bytes and none correct about the device. Document 02 gives
//! `moxie-memory` the "real resource authority" and forbids "independent caches
//! in consumers" in those words, and document 06's M2 exit requires "exactly one
//! production weight-residency owner; no cache class in adapters".
//!
//! This is that owner. It is also, deliberately, *only* the owner:
//!
//! * It opens no file. `moxie-storage` reads bytes; an `arch-check` rule keeps
//!   `std::fs` and `std::path` out of this crate entirely.
//! * It allocates no device memory and waits on no event. `moxie-executor`
//!   performs the work orders this authority issues and reports their outcome,
//!   the same split that already pairs [`crate::Arena`]'s pure ranges with one
//!   real allocation.
//! * It never blocks. [`ResidencyAuthority::acquire`] either admits or refuses,
//!   so the deadlock document 03 prohibits -- "prohibit deadlock when all
//!   evictable entries are leased" -- cannot be expressed by this API.
//!
//! Four properties are the point of the design:
//!
//! * **Incoming first.** Admission arithmetic is `committed + incoming` against
//!   the cap, computed and reported *before* a victim is chosen. Document 03:
//!   "Include incoming expert size before evicting/allocating."
//! * **Coalescing.** Concurrent acquires of one absent chunk produce exactly one
//!   read. The second acquire gets a lease and the first one's ticket.
//! * **Demand outranks prediction.** A prefetch is queued behind demand,
//!   evicted before demand, and refused outright when admitting it would evict
//!   demand data.
//! * **Explicit release.** A [`ResidencyLease`] is given back by name. Dropping
//!   one leaves it held and visible in [`ResidencyAuthority::outstanding`] --
//!   R08's leak was invisible, not large.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{DeviceTier, DeviceUuid, Error, HostTier, Result, Scope, Tier};

use crate::arena::{Allocation, Arena};
use crate::host::HostBuffer;
use crate::ledger::{Ledger, Reservation};
use crate::request::{BufferRequest, PlanRequest, StageSpan};

/// Alignment every cache range is placed at.
///
/// One value, not a per-chunk parameter: a residency cache holds opaque bytes
/// whose consumer is a copy engine or a kernel operand, and 256 bytes satisfies
/// both without asking the caller to know which. It is also the arena's base
/// alignment, so no allocation can ask for more than the arena guarantees.
pub const CACHE_ALIGNMENT: u64 = 256;

/// Declared control cost of one tracked placement, **excluding its identity**.
///
/// Charged up front as
/// `max_placements * (PLACEMENT_CONTROL_BYTES + max_identity_bytes)`, so the
/// authority's own bookkeeping is admitted rather than assumed free. This
/// constant covers the fixed part: the placement record, the two map nodes that
/// reach it, the shared identity's own header, and the ticket slot.
///
/// The identity is charged **separately and from a declared bound**, because
/// guessing it was a real defect. A flat 512 bytes per placement was measured
/// by an independent review at 166,361 bytes of retained heap against a 49,216
/// byte envelope, because a chunk identity is two heap strings that the index
/// and the placement each held a copy of. Two things changed: the identity is
/// now stored **once** behind an `Arc` that both share, and its length is
/// bounded by [`ResidencyRequest::max_identity_bytes`] and **enforced on every
/// acquire**. An envelope derived from a bound nobody checks is not an
/// envelope, and document 03 is blunt about the class of error: mapped virtual
/// bytes are not committed host RAM, and "neither is free".
///
/// The value is **measured, not chosen**, and it is measured against the
/// *expensive* shape. With the identity charged separately, a settled placement
/// costs 655.7, 634.1 and 631.3 bytes at 64, 256 and 1,024 placements. A
/// **pending** one costs more -- it also owns a live ticket -- and an
/// independent review found the first version of this constant covering only
/// the settled case: one pending placement retained 8,274 bytes against a 2,368
/// byte envelope. Measured across 1, 2, 4 and 64 pending placements, the
/// per-placement cost is about 1,478 bytes beyond the identity, so this is the
/// next power of two above that.
pub const PLACEMENT_CONTROL_BYTES: u64 = 2048;

/// Declared control cost of the authority **itself**, independent of how much
/// it holds.
///
/// The maps, the arena's free list, the lease slab's headers and the labels
/// exist before a single chunk does. Charging only per placement made small
/// caches the worst case rather than the cheapest: at one placement the fixed
/// part *was* the envelope, and the envelope missed it entirely. Measured at
/// about 7,061 bytes; this is the next power of two above it.
pub const AUTHORITY_CONTROL_BYTES: u64 = 8192;

/// Declared control cost of one lease slot, in bytes. Charged like a
/// placement's: the lease table is real memory and a bounded one is still
/// memory.
pub const LEASE_CONTROL_BYTES: u64 = 64;

/// Upper bound on a declared placement or lease count, so the control charge
/// cannot overflow or quietly become the dominant cost.
const MAX_DECLARED_PLACEMENTS: u32 = 1 << 22;

/// The longest artifact identity and tensor role the constructors accept. They
/// are the ceiling [`ResidencyRequest::max_identity_bytes`] is measured against.
const MAX_ARTIFACT_BYTES: usize = 512;
const MAX_ROLE_BYTES: usize = 256;

fn invalid(field: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// An artifact's stable identity, opaque to this crate.
///
/// **Not a path.** This authority may not learn what a filesystem is, so what it
/// holds is an identity string its caller already validated:
/// `moxie_storage::Artifact::identity()` for a canonical artifact, or a recorded
/// digest for a source container that has no manifest. The constructor rejects
/// the two shapes that would let a path be smuggled through anyway.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactId(String);

impl ArtifactId {
    pub fn new(identity: impl Into<String>) -> Result<Self> {
        let identity = identity.into();
        if identity.is_empty() {
            return Err(invalid("artifact", "an artifact identity cannot be empty"));
        }
        if identity.len() > 512 {
            return Err(invalid(
                "artifact",
                "artifact identity longer than 512 bytes",
            ));
        }
        if identity.contains('/') || identity.contains('\\') || identity.contains('\0') {
            return Err(invalid(
                "artifact",
                "an artifact identity is not a path: separators and NUL are refused",
            ));
        }
        Ok(ArtifactId(identity))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A canonical tensor role, or one expert's slice of a fused one.
///
/// Document 03's chunk identity says "tensor/expert" and this is that in one
/// type. `expert: None` is a whole tensor; `Some(e)` is expert `e`. The role is
/// a canonical name supplied as data -- naming a model family here would breach
/// the `memory branches on a model name` rule, and the rule is right: a cache
/// that knows which family it holds will eventually hold two.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorSlot {
    role: String,
    expert: Option<u32>,
}

impl TensorSlot {
    pub fn tensor(role: impl Into<String>) -> Result<Self> {
        Self::build(role, None)
    }

    pub fn expert(role: impl Into<String>, expert: u32) -> Result<Self> {
        Self::build(role, Some(expert))
    }

    fn build(role: impl Into<String>, expert: Option<u32>) -> Result<Self> {
        let role = role.into();
        if role.is_empty() {
            return Err(invalid("role", "a tensor role cannot be empty"));
        }
        if role.len() > MAX_ROLE_BYTES {
            return Err(invalid(
                "role",
                format!("tensor role longer than {MAX_ROLE_BYTES} bytes"),
            ));
        }
        Ok(TensorSlot { role, expert })
    }

    pub fn role(&self) -> &str {
        &self.role
    }

    pub const fn expert_index(&self) -> Option<u32> {
        self.expert
    }
}

/// A byte range **inside one tensor**, never inside a file.
///
/// The distinction is load-bearing. Translating a logical range to a file offset
/// needs the artifact's header, which lives in `moxie-storage`; if this type
/// could express a file offset, the authority would be one refactor away from
/// owning the layout too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalRange {
    offset_bytes: u64,
    len_bytes: u64,
}

impl LogicalRange {
    pub fn new(offset_bytes: u64, len_bytes: u64) -> Result<Self> {
        if len_bytes == 0 {
            return Err(invalid("range", "a chunk range cannot be empty"));
        }
        offset_bytes
            .checked_add(len_bytes)
            .ok_or_else(|| invalid("range", "chunk range end overflows"))?;
        Ok(LogicalRange {
            offset_bytes,
            len_bytes,
        })
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn len_bytes(&self) -> u64 {
        self.len_bytes
    }
}

/// Document 03's immutable canonical chunk identity, exactly:
/// `(artifact, tensor/expert, logical range, format version)`.
///
/// `format_version` participates in identity for the reason document 02 gives
/// for plan cache keys: "changing a checkpoint scale convention must invalidate
/// incompatible packed layouts". A chunk read under one format version can never
/// be served to a consumer asking for another, because they are different
/// chunks.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkId {
    artifact: ArtifactId,
    slot: TensorSlot,
    range: LogicalRange,
    format_version: u32,
}

impl ChunkId {
    pub const fn new(
        artifact: ArtifactId,
        slot: TensorSlot,
        range: LogicalRange,
        format_version: u32,
    ) -> Self {
        ChunkId {
            artifact,
            slot,
            range,
            format_version,
        }
    }

    pub const fn artifact(&self) -> &ArtifactId {
        &self.artifact
    }

    pub const fn slot(&self) -> &TensorSlot {
        &self.slot
    }

    pub const fn range(&self) -> LogicalRange {
        self.range
    }

    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    pub const fn len_bytes(&self) -> u64 {
        self.range.len_bytes
    }

    /// What this identity costs on the heap: its two strings.
    ///
    /// The rest of a `ChunkId` is inline, so this is the whole of what an
    /// authority's control envelope has to cover per placement beyond its fixed
    /// part.
    pub fn identity_bytes(&self) -> u64 {
        (self.artifact.0.len() + self.slot.role.len()) as u64
    }
}

impl core::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}::{}", self.artifact.0, self.slot.role)?;
        if let Some(e) = self.slot.expert {
            write!(f, "[expert {e}]")?;
        }
        write!(
            f,
            "@{}+{}v{}",
            self.range.offset_bytes, self.range.len_bytes, self.format_version
        )
    }
}

/// What a prepared layout was prepared *for*.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CapabilityKey {
    pub sm_major: u32,
    pub sm_minor: u32,
    /// An opaque kernel-layout tag. Not a model name and not a kernel name: a
    /// layout family, so two kernels that consume the same bytes share one
    /// prepared chunk.
    pub layout_tag: String,
}

/// Document 03's separate prepared-layout identity:
/// `(chunk, device capability, kernel layout version)`.
///
/// **Nothing produces one yet.** Preparation is a repacker, and the repacker is
/// M3; this task uploads canonical bytes unchanged. The type exists now so the
/// identity is not retrofitted onto a cache that already has keys, which is the
/// mistake document 03 is describing when it insists the prepared identity is
/// *separate* from the canonical one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PreparedId {
    pub chunk: ChunkId,
    pub capability: CapabilityKey,
    pub layout_version: u32,
}

// ---------------------------------------------------------------------------
// Classes
// ---------------------------------------------------------------------------

/// How urgent a residency request is. Document 03: "Demand work outranks
/// predictions; speculative prefetch is bounded and evictable before useful
/// demand data."
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Urgency {
    /// The step cannot proceed without it.
    Demand,
    /// A prediction. Bounded, queued behind demand, evicted first, and refused
    /// rather than admitted at demand data's expense.
    Prefetch,
}

/// What kind of resident thing this is.
///
/// Three classes under **one** authority and one capacity, which is the whole
/// point. Document 03 requires dense spine tensors to be charged "against expert
/// and context capacity" rather than excused from it, and
/// [ADR 0009](../../../docs/decisions/adr/0009-engram-conditional-memory.md)
/// requires conditional-memory tables to be "one residency class under the same
/// authority", NVMe-backed with an admitted footprint and **never
/// zero-resident**.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Content {
    /// A dense tensor persisted for reuse.
    DenseSpine,
    /// One expert, or one expert's slice of a fused tensor.
    Expert,
    /// A deterministic-prefetch conditional-memory table row set. The class and
    /// its resident floor exist so a later Engram task has no reason to build a
    /// second cache. **No table mathematics is implemented here**, and no Engram
    /// capability is claimed.
    ConditionalTable,
}

/// Document 03's `use_class`, in its two independent dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UseClass {
    pub urgency: Urgency,
    pub content: Content,
}

impl UseClass {
    pub const fn demand(content: Content) -> Self {
        UseClass {
            urgency: Urgency::Demand,
            content,
        }
    }

    pub const fn prefetch(content: Content) -> Self {
        UseClass {
            urgency: Urgency::Prefetch,
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Where one placement stands, in document 03's lifecycle.
///
/// ```text
/// Absent -> Reading -> HostReady -> Uploading -> DeviceReady -> Retiring -> Absent
/// ```
///
/// Preparation (`Preparing -> PreparedHost/DeviceReady`) is **not** in this
/// enum. No preparer exists -- that is M3's repacker -- and an unreachable state
/// is a stub, which document 09 and AGENTS.md both refuse to count as a
/// delivered feature. [`PreparedId`] carries the identity so the addition is a
/// state, not a redesign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChunkState {
    /// A read is outstanding into this placement's range.
    Reading,
    /// Host bytes are valid and usable.
    HostReady,
    /// A copy into this device range is outstanding.
    Uploading,
    /// Device bytes are valid and usable.
    DeviceReady,
    /// Evicted, with leases still live. Bytes are released when the last one
    /// retires, not when eviction was decided.
    Retiring,
    /// Cancelled or lost with submission state unknown. Charged, unusable, and
    /// never a candidate for anything until the outcome is observed.
    Quarantined,
}

impl ChunkState {
    /// Whether the bytes may be read by a consumer.
    pub const fn is_ready(self) -> bool {
        matches!(self, ChunkState::HostReady | ChunkState::DeviceReady)
    }

    /// Whether a transfer may still touch these bytes.
    pub const fn is_in_flight(self) -> bool {
        matches!(
            self,
            ChunkState::Reading | ChunkState::Uploading | ChunkState::Quarantined
        )
    }

    pub const fn name(self) -> &'static str {
        match self {
            ChunkState::Reading => "reading",
            ChunkState::HostReady => "host ready",
            ChunkState::Uploading => "uploading",
            ChunkState::DeviceReady => "device ready",
            ChunkState::Retiring => "retiring",
            ChunkState::Quarantined => "quarantined",
        }
    }
}

// ---------------------------------------------------------------------------
// Identities issued by the authority
// ---------------------------------------------------------------------------

macro_rules! process_unique_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u64);

        impl $name {
            fn next() -> Self {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                $name(NEXT.fetch_add(1, Ordering::Relaxed))
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

process_unique_id!(AuthorityId, "One authority's process-unique identity.");

/// One residency lease's identity: a slot in the authority's bounded lease
/// table, plus the generation that slot was handed out at.
///
/// Not a global counter, because the table is not a map. The lease table is a
/// pre-admitted slab sized at `open`, so taking and giving back a lease costs
/// no allocation at all -- and a resident-chunk `acquire` is the hot path of a
/// routed step, one call per expert per layer per token. A generation makes a
/// stale identity detectable after its slot has been reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidencyLeaseId {
    slot: u32,
    generation: u32,
}

impl ResidencyLeaseId {
    /// One number naming both halves, for a report.
    pub const fn get(self) -> u64 {
        ((self.generation as u64) << 32) | self.slot as u64
    }

    pub const fn slot(self) -> u32 {
        self.slot
    }
}

impl core::fmt::Display for ResidencyLeaseId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lease {}#{}", self.slot, self.generation)
    }
}
process_unique_id!(
    TicketId,
    "One outstanding transfer's process-unique identity. It is the readiness dependency a consumer waits on."
);

/// A caller-supplied turn. Every lease belongs to one, and
/// [`ResidencyAuthority::end_turn`] releases all of them.
///
/// R08 in one mechanism: the legacy leak was a lease released on "next token",
/// which leaked whenever a turn ended and no next token came. A turn that ends
/// releases its leases whether or not anything was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TurnId(u64);

impl TurnId {
    pub const fn new(id: u64) -> Self {
        TurnId(id)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Interned placement identity: `(scope, chunk)`, resolved once so the hit path
/// compares integers instead of cloning strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PlacementKey(u32);

/// The right to read one resident chunk, and the only way to give it back.
///
/// **Deliberately not `Clone`.** A second handle is a second authority to
/// release the same residency, and the workspace has met that failure before:
/// [`crate::Reservation`] and [`crate::Allocation`] carry the same rule for the
/// same reason. Dropping one does **not** release it -- document 02 forbids
/// `Drop` alone from freeing a resource a transfer may still touch, and a held
/// lease that nobody holds is at least visible in
/// [`ResidencyAuthority::outstanding`].
#[derive(Debug)]
#[must_use = "a lease that is never released keeps its chunk pinned; release it explicitly"]
pub struct ResidencyLease {
    id: ResidencyLeaseId,
    authority: AuthorityId,
    placement: PlacementKey,
    scope: Scope,
    turn: TurnId,
    class: UseClass,
}

impl ResidencyLease {
    pub const fn id(&self) -> ResidencyLeaseId {
        self.id
    }

    pub const fn scope(&self) -> Scope {
        self.scope
    }

    pub const fn turn(&self) -> TurnId {
        self.turn
    }

    pub const fn class(&self) -> UseClass {
        self.class
    }
}

/// The right to back **one** scope's cache with **one** real allocation.
///
/// Finding 1 of the independent review is why this type exists. Without it,
/// `DeviceResidency::create` took a shared borrow of the authority and
/// allocated, so a caller could build two 4 MiB caches against one 4 MiB
/// reservation -- measured on a real 3090 as 8 MiB of physical allocation
/// against a 4 MiB charge -- and `close` would then release that charge while
/// both allocations stayed live and readable. That is R02 with the sign
/// flipped: the ledger saying zero while the card holds 8 MiB.
///
/// **Deliberately not `Clone`.** A second entitlement is a second allocation.
/// Dropping one does **not** give it back: the authority keeps counting it, so
/// [`ResidencyAuthority::close`] refuses and the leak is visible rather than
/// silent -- the same rule [`Reservation`] and [`ResidencyLease`] carry.
#[derive(Debug)]
#[must_use = "an unreturned backing keeps its scope claimed; give it back with `return_backing`"]
pub struct DeviceBacking {
    authority: AuthorityId,
    scope: Scope,
    capacity: u64,
}

impl DeviceBacking {
    pub const fn scope(&self) -> Scope {
        self.scope
    }

    /// The authority this entitlement came from.
    ///
    /// A backing is a right to allocate against **one** authority's
    /// reservation. Anything that reaches through it -- an upload, a readback --
    /// must check this first: an independent review drove authority B's upload
    /// through authority A's backing on a real GPU, and it **overwrote A's
    /// still-leased bytes and marked B ready**.
    pub const fn authority(&self) -> AuthorityId {
        self.authority
    }

    /// Exactly the admitted capacity. A backing may not be larger than what was
    /// reserved for it, and may not be smaller either: a short allocation would
    /// make the authority's ranges address memory that does not exist.
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }
}

/// A backing the authority refused to take back, carrying it intact.
#[derive(Debug)]
#[must_use = "the scope is still backed; correct the cause and return it again"]
pub struct ReturnBackingRefused {
    pub backing: DeviceBacking,
    pub error: Error,
}

impl core::fmt::Display for ReturnBackingRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for ReturnBackingRefused {}

/// A release the authority refused, carrying the lease back.
///
/// Consuming the lease on a *failed* release would destroy the only handle to a
/// residency that is still pinned -- the same shape of leak as R08, created by
/// the error path instead of the happy one.
#[derive(Debug)]
#[must_use = "the chunk is still pinned; release the lease against its own authority"]
pub struct ReleaseLeaseRefused {
    pub lease: ResidencyLease,
    pub error: Error,
}

impl core::fmt::Display for ReleaseLeaseRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for ReleaseLeaseRefused {}

// ---------------------------------------------------------------------------
// Work orders
// ---------------------------------------------------------------------------

/// One effect the authority needs performed, addressed to whoever can perform
/// it. The authority itself performs none of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkOrder {
    /// Read `chunk`'s logical range into the host cache at `host_offset`.
    Read {
        ticket: TicketId,
        chunk: ChunkId,
        host_offset: u64,
        len_bytes: u64,
    },
    /// Copy the host cache range into the device cache range on `scope`.
    Upload {
        ticket: TicketId,
        chunk: ChunkId,
        scope: Scope,
        host_offset: u64,
        device_offset: u64,
        len_bytes: u64,
    },
}

impl WorkOrder {
    pub const fn ticket(&self) -> TicketId {
        match self {
            WorkOrder::Read { ticket, .. } | WorkOrder::Upload { ticket, .. } => *ticket,
        }
    }
}

/// What an `acquire` produced.
///
/// The pending arm is much larger than the ready one, and deliberately so: it
/// carries the report that says what was admitted and what it displaced. Boxing
/// that to equalise the arms would put an allocation on the path that is about
/// to perform a disk read, to save a move on the path that is not.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
#[must_use = "an acquire that is dropped leaks its lease"]
pub enum Acquired {
    /// Resident now. Nothing was admitted and nothing was evicted.
    Ready(ResidencyLease),
    /// Not yet resident. `ticket` is the readiness dependency; the bytes must
    /// not be read before it completes.
    Pending {
        lease: ResidencyLease,
        ticket: TicketId,
        work: PendingWork,
        report: ResidencyReport,
    },
}

/// What the caller must do about a pending acquire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingWork {
    /// Perform this, then report the outcome.
    Issued(WorkOrder),
    /// Another acquire already owns the work for this ticket. Wait on it; do
    /// **not** perform a second read.
    Coalesced,
    /// A prefetch, queued behind demand. [`ResidencyAuthority::next_prefetch`]
    /// releases it when no demand work is outstanding.
    Queued,
}

/// How a read or upload ended.
#[derive(Debug)]
pub enum Outcome {
    /// The bytes are valid.
    Completed,
    /// It failed, observably: nothing is in flight against these bytes.
    Failed(Error),
    /// It failed and the submission state is **unknown**: a transfer may still
    /// be touching these bytes. They are quarantined, not reused. This is the
    /// same rule `moxie_executor::LeaseState::Lost` already applies, and R07 is
    /// why both exist.
    SubmissionUnknown(Error),
}

// ---------------------------------------------------------------------------
// Reports and statistics
// ---------------------------------------------------------------------------

/// What admitting one chunk costs, and what it would displace.
///
/// Document 03 requires an admission report to show "physical capacity, already
/// committed resources, reserved peak, and remaining headroom", and to "include
/// incoming expert size before evicting/allocating". `incoming_bytes` is that
/// sentence as a field, and `would_evict` is the list it is measured against,
/// computed before a single victim is actually removed.
///
/// Produced whenever something is **admitted or refused**. A cache hit produces
/// none: it admits nothing, and building a report on the hit path would put a
/// heap allocation in the one path that is measured for having none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyReport {
    pub scope: Scope,
    pub tier: Tier,
    pub cap_bytes: u64,
    /// Resident plus in-flight, **before** this request.
    pub committed_bytes: u64,
    /// This chunk, named before anything is evicted.
    pub incoming_bytes: u64,
    /// Present and pinned by a live lease: unevictable at any price.
    pub leased_bytes: u64,
    /// Present, unpinned and in a settled state.
    pub evictable_bytes: u64,
    /// The exact victims, in the exact order. Empty when nothing was displaced.
    pub would_evict: Vec<ChunkId>,
    /// Free bytes in the arena after those evictions.
    pub free_bytes: u64,
    /// The largest single range available after them. Smaller than `free_bytes`
    /// is fragmentation, which document 03 tracks as real capacity rather than
    /// hiding it.
    pub largest_free_bytes: u64,
    pub remaining_headroom_bytes: u64,
}

impl core::fmt::Display for ResidencyReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} {}: cap {} B, committed {} B, incoming {} B, leased {} B, evictable {} B, \
             evicting {} chunk(s), free {} B (largest {} B), remaining {} B",
            self.scope,
            self.tier.name(),
            self.cap_bytes,
            self.committed_bytes,
            self.incoming_bytes,
            self.leased_bytes,
            self.evictable_bytes,
            self.would_evict.len(),
            self.free_bytes,
            self.largest_free_bytes,
            self.remaining_headroom_bytes,
        )
    }
}

/// A refusal, with the breakdown that explains it.
///
/// Document 03: reject "with a breakdown and legal alternatives", not with a
/// bare error. `leased_bytes == cap_bytes` in this report is the exact shape of
/// the deadlock M2's exit asks to be demonstrated impossible: the request is
/// refused rather than parked.
#[derive(Debug, Clone)]
#[must_use = "a refusal carries the breakdown a caller has to act on"]
pub struct ResidencyRefused {
    pub error: Error,
    pub report: ResidencyReport,
}

impl core::fmt::Display for ResidencyRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.error, self.report)
    }
}

impl std::error::Error for ResidencyRefused {}

impl From<ResidencyRefused> for Error {
    fn from(r: ResidencyRefused) -> Error {
        r.error
    }
}

/// Counters document 03 requires to be recorded: "prediction precision, wasted
/// bytes, evictions of demand data, hit latency, and end-to-end impact".
///
/// The four this authority can observe are here. **No performance claim follows
/// from them** -- there is no baseline to compare against, and inventing one
/// would be the unsupported claim AGENTS.md forbids.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub bytes_uploaded: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    /// Demand-class bytes evicted. A prefetch never contributes to this: it is
    /// refused before it can.
    pub demand_evicted_bytes: u64,
    pub prefetch_admitted_bytes: u64,
    /// Prefetched bytes later demanded: the prediction was right.
    pub prefetch_used_bytes: u64,
    /// Prefetched bytes evicted without ever being demanded: it was wrong.
    pub prefetch_wasted_bytes: u64,
    pub refusals: u64,
    pub read_failures: u64,
    pub upload_failures: u64,
    pub quarantined: u64,
    pub expired: u64,
}

/// One visible placement, for reconciliation and diagnosis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutstandingChunk {
    pub chunk: ChunkId,
    pub scope: Scope,
    pub state: ChunkState,
    pub class: UseClass,
    pub bytes: u64,
    pub leases: u32,
    pub last_used: u64,
}

/// What a turn's end released.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnCleanup {
    pub released_leases: Vec<ResidencyLeaseId>,
    /// Placements that became evictable because this turn ended.
    pub unpinned_chunks: Vec<ChunkId>,
    /// Tickets still in flight. Their bytes stay charged until observed --
    /// ending a turn never frees bytes a transfer may still touch.
    pub still_in_flight: Vec<TicketId>,
}

// ---------------------------------------------------------------------------
// Internal records
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Placement {
    /// Shared with the index entry that reaches it, so one identity costs one
    /// allocation rather than two.
    chunk: Arc<ChunkId>,
    scope: Scope,
    class: UseClass,
    state: ChunkState,
    allocation: Option<Allocation>,
    offset: u64,
    bytes: u64,
    leases: u32,
    last_used: u64,
    /// A prefetch that a demand later asked for. Once true, it is ordered and
    /// accounted as demand data.
    ever_demanded: bool,
    ticket: Option<TicketId>,
    /// For a device placement: where its source bytes live in the host cache,
    /// and the host placement pinning them.
    upload_source: Option<PlacementKey>,
}

impl Placement {
    const fn is_evictable(&self) -> bool {
        self.leases == 0 && self.state.is_ready()
    }

    /// Prefetch data that has never been demanded is displaced first.
    fn class_rank(&self) -> u8 {
        if self.class.urgency == Urgency::Prefetch && !self.ever_demanded {
            0
        } else {
            1
        }
    }
}

/// Which half of a two-stage ticket is outstanding.
///
/// A device acquire of an absent chunk is one readiness dependency covering two
/// transfers: read it to the host, then copy it to the device. Splitting it into
/// two tickets would make every consumer wait twice and would let the host bytes
/// be evicted between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// Filling the host placement named by `source`.
    Read,
    /// Copying host bytes into the device placement.
    Upload,
    /// Waiting for another ticket's read of the same host chunk. No work order
    /// of its own; it is released when that ticket completes.
    BlockedOnRead(TicketId),
}

#[derive(Debug)]
struct Ticket {
    placement: PlacementKey,
    /// The host placement this ticket reads into, when it has one. Equal to
    /// `placement` for a plain host acquire.
    source: PlacementKey,
    stage: Stage,
    urgency: Urgency,
    deadline: u64,
    /// Live leases waiting on this ticket. The last one leaving cancels the
    /// work; one of several leaving does not.
    waiters: u32,
    cancelled: bool,
    /// A queued prefetch has not been handed out yet.
    queued: bool,
    /// True once a work order for the current stage has been given out, so the
    /// same transfer is never issued twice.
    issued: bool,
    /// This ticket created the host source placement, so it is responsible for
    /// dropping it if the read never produced usable bytes.
    ///
    /// A ticket that merely *joined* someone else's read must never drop that
    /// placement: it is another ticket's in-flight destination.
    owns_source: bool,
}

/// One slot of the bounded lease table.
#[derive(Debug, Clone, Copy)]
struct LeaseSlot {
    generation: u32,
    held: Option<LeaseRecord>,
}

#[derive(Debug, Clone, Copy)]
struct LeaseRecord {
    placement: PlacementKey,
    turn: TurnId,
    ticket: Option<TicketId>,
}

#[derive(Debug)]
struct ScopeCache {
    tier: Tier,
    cap_bytes: u64,
    arena: Arena,
    /// Bytes of every placement in this scope, resident and in flight.
    committed: u64,
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// What a residency authority wants, declared before anything is charged.
#[derive(Debug, Clone)]
pub struct ResidencyRequest {
    pub label: String,
    /// The host cache, charged to `host.pageable`.
    pub host_cap_bytes: u64,
    /// Per-device expert caches, charged to `device.expert_cache`, each against
    /// its own device UUID. One cap per device, never one shared number:
    /// AGENTS.md forbids assuming the three cards' memory is one allocation.
    pub device_caps: Vec<(DeviceUuid, u64)>,
    /// Bound on tracked placements. Its control cost is admitted up front.
    pub max_placements: u32,
    /// Bound on one chunk identity's heap: `artifact` plus `role`, in bytes.
    ///
    /// Charged per placement **and enforced on every acquire**, so the control
    /// envelope is a bound rather than a hope. An identity longer than this is
    /// a typed refusal naming both lengths; nothing is truncated and nothing is
    /// admitted unpriced.
    pub max_identity_bytes: u32,
    /// Bound on simultaneously live leases. The table is allocated once at
    /// this size, so acquiring a resident chunk allocates nothing.
    pub max_leases: u32,
    /// Bound on the prefetch queue. An unbounded queue is a stop condition.
    pub prefetch_queue_capacity: u32,
    /// ADR 0009's floor: eviction may never take resident `ConditionalTable`
    /// bytes below this. Zero when the class is unused.
    pub conditional_floor_bytes: u64,
}

impl ResidencyRequest {
    pub fn new(label: impl Into<String>, host_cap_bytes: u64) -> Self {
        ResidencyRequest {
            label: label.into(),
            host_cap_bytes,
            device_caps: Vec::new(),
            max_placements: 4096,
            max_identity_bytes: 256,
            max_leases: 4096,
            prefetch_queue_capacity: 64,
            conditional_floor_bytes: 0,
        }
    }

    pub fn device(mut self, uuid: DeviceUuid, cap_bytes: u64) -> Self {
        self.device_caps.push((uuid, cap_bytes));
        self
    }

    pub const fn max_placements(mut self, n: u32) -> Self {
        self.max_placements = n;
        self
    }

    pub const fn max_leases(mut self, n: u32) -> Self {
        self.max_leases = n;
        self
    }

    pub const fn max_identity_bytes(mut self, n: u32) -> Self {
        self.max_identity_bytes = n;
        self
    }

    pub const fn prefetch_queue_capacity(mut self, n: u32) -> Self {
        self.prefetch_queue_capacity = n;
        self
    }

    pub const fn conditional_floor_bytes(mut self, bytes: u64) -> Self {
        self.conditional_floor_bytes = bytes;
        self
    }
}

/// The one production weight-residency owner.
#[derive(Debug)]
#[must_use = "an authority that is never closed keeps its envelope charged"]
pub struct ResidencyAuthority {
    id: AuthorityId,
    label: String,
    host: HostBuffer,
    /// The device caches' admitted envelope, one reservation covering every
    /// declared device. Held by name; released by name.
    device_envelope: Option<Reservation>,
    caches: BTreeMap<Scope, ScopeCache>,
    /// Identity -> placement, nested by scope.
    ///
    /// Nested rather than keyed by `(Scope, ChunkId)` for one measured reason:
    /// a tuple key cannot be looked up from a borrowed chunk, so every hit
    /// would have to *clone* the identity -- two heap allocations -- just to
    /// ask whether it was already resident. The hit path is the one path this
    /// task predeclares as allocation-free, and a gate measures it.
    index: BTreeMap<Scope, BTreeMap<Arc<ChunkId>, PlacementKey>>,
    placements: BTreeMap<PlacementKey, Placement>,
    next_placement: u32,
    max_placements: u32,
    max_identity_bytes: u32,
    tickets: BTreeMap<TicketId, Ticket>,
    /// Queued prefetch tickets in `(deadline, sequence)` order.
    prefetch_queue: Vec<(u64, u64, TicketId)>,
    prefetch_queue_capacity: u32,
    next_sequence: u64,
    outstanding_demand: u32,
    /// Scopes whose cache has an outstanding physical backing. One at a time,
    /// and `close` refuses while any is out.
    claimed_backings: BTreeMap<Scope, bool>,
    /// Set once `close` succeeds. Everything that could admit, claim or move
    /// bytes refuses afterwards.
    ///
    /// Without it a closed authority still answered `claim_backing` from its
    /// retained configuration, and an independent review allocated **4,194,304
    /// bytes with zero ledger charge** through one. A released reservation is
    /// not a budget.
    closed: bool,
    conditional_floor_bytes: u64,
    conditional_resident: u64,
    /// The bounded lease table: one pre-allocated slot per declared lease, so
    /// acquiring and releasing allocate nothing.
    lease_slots: Vec<LeaseSlot>,
    /// Indices of the free slots. Filled at `open` to its final capacity, so
    /// pushing a released slot back never reallocates.
    free_lease_slots: Vec<u32>,
    live_leases: u32,
    stats: ResidencyStats,
}

impl ResidencyAuthority {
    /// Admit the whole cache envelope, then build. A refusal charges nothing.
    pub fn open(ledger: &mut Ledger, request: &ResidencyRequest) -> Result<Self> {
        if request.label.is_empty() {
            return Err(invalid("label", "a residency authority must be named"));
        }
        if request.host_cap_bytes == 0 {
            return Err(invalid(
                "host_cap_bytes",
                "a residency authority needs a host cache: every device chunk is uploaded from one",
            ));
        }
        if request.max_placements == 0 || request.max_placements > MAX_DECLARED_PLACEMENTS {
            return Err(invalid(
                "max_placements",
                format!("max_placements must be in 1..={MAX_DECLARED_PLACEMENTS}"),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (uuid, cap) in &request.device_caps {
            if !seen.insert(*uuid) {
                return Err(invalid("device_caps", "a device is declared twice"));
            }
            if *cap == 0 {
                return Err(invalid("device_caps", "a device cache cannot be empty"));
            }
        }
        if request.conditional_floor_bytes > request.host_cap_bytes {
            return Err(invalid(
                "conditional_floor_bytes",
                "the conditional-memory floor cannot exceed the host cache",
            ));
        }

        if request.max_leases == 0 || request.max_leases > MAX_DECLARED_PLACEMENTS {
            return Err(invalid(
                "max_leases",
                format!("max_leases must be in 1..={MAX_DECLARED_PLACEMENTS}"),
            ));
        }
        if request.max_identity_bytes == 0
            || request.max_identity_bytes > (MAX_ARTIFACT_BYTES + MAX_ROLE_BYTES) as u32
        {
            return Err(invalid(
                "max_identity_bytes",
                format!(
                    "max_identity_bytes must be in 1..={}",
                    MAX_ARTIFACT_BYTES + MAX_ROLE_BYTES
                ),
            ));
        }
        let control_bytes = u64::from(request.max_placements)
            .checked_mul(PLACEMENT_CONTROL_BYTES + u64::from(request.max_identity_bytes))
            .and_then(|p| {
                u64::from(request.max_leases)
                    .checked_mul(LEASE_CONTROL_BYTES)
                    .and_then(|l| p.checked_add(l))
            })
            .and_then(|c| c.checked_add(AUTHORITY_CONTROL_BYTES))
            .ok_or_else(|| invalid("max_placements", "control charge overflows"))?;
        let host_data = usize::try_from(request.host_cap_bytes).map_err(|_| {
            invalid(
                "host_cap_bytes",
                "host cache exceeds this machine's address space",
            )
        })?;
        let control = usize::try_from(control_bytes)
            .map_err(|_| invalid("max_placements", "control charge exceeds address space"))?;

        // The device caches are admitted **first**, as one plan covering every
        // declared device. Two reasons, both load-bearing:
        //
        // * A cache whose capacity nobody reserved is a placement simulator
        //   with a confident API -- R02 -- and document 06 names it a stop
        //   condition for this task in the words "replace any simulated
        //   placement with enforceable reservations".
        // * One plan rather than one per device, because a partial device
        //   admission is the non-atomic reservation task 0006 removed.
        //
        // Everything after it releases this envelope on the way out, so a
        // refusal anywhere in `open` leaves the ledger exactly as it found it.
        let device_envelope = if request.device_caps.is_empty() {
            None
        } else {
            let mut plan = PlanRequest::new(format!("{} device caches", request.label), ["live"])?;
            for (uuid, cap) in &request.device_caps {
                plan.buffer(BufferRequest::new(
                    format!("expert cache on {uuid}"),
                    Scope::Device(*uuid),
                    Tier::Device(DeviceTier::ExpertCache),
                    *cap,
                    StageSpan::at(0),
                ))?;
            }
            Some(ledger.admit(&plan).map_err(Error::from)?)
        };

        let release_devices = |ledger: &mut Ledger, envelope: Option<Reservation>| {
            if let Some(r) = envelope {
                ledger.release(r).expect("the admitting ledger");
            }
        };

        let host = match HostBuffer::allocate_in(
            ledger,
            &request.label,
            HostTier::Pageable,
            host_data,
            control,
        ) {
            Ok(h) => h,
            Err(e) => {
                release_devices(ledger, device_envelope);
                return Err(e);
            }
        };

        // One allocation each, at their final size, before anything is served.
        let leases = request.max_leases as usize;
        let mut lease_slots = Vec::new();
        let mut free_lease_slots = Vec::new();
        if lease_slots.try_reserve_exact(leases).is_err()
            || free_lease_slots.try_reserve_exact(leases).is_err()
        {
            let mut host = host;
            let _ = host.release(ledger);
            release_devices(ledger, device_envelope);
            return Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: u64::from(request.max_leases) * LEASE_CONTROL_BYTES,
                available_bytes: 0,
            });
        }
        lease_slots.resize(
            leases,
            LeaseSlot {
                generation: 0,
                held: None,
            },
        );
        // Descending, so the first lease handed out is slot 0 and a trace reads
        // in the order it happened.
        free_lease_slots.extend((0..request.max_leases).rev());

        let mut caches = BTreeMap::new();
        caches.insert(
            Scope::Host,
            ScopeCache {
                tier: Tier::Host(HostTier::Pageable),
                cap_bytes: request.host_cap_bytes,
                arena: match Arena::new(
                    format!("{} host cache", request.label),
                    request.host_cap_bytes,
                    CACHE_ALIGNMENT,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        let mut host = host;
                        let _ = host.release(ledger);
                        release_devices(ledger, device_envelope);
                        return Err(e);
                    }
                },
                committed: 0,
            },
        );

        for (uuid, cap) in &request.device_caps {
            match Arena::new(
                format!("{} device cache", request.label),
                *cap,
                CACHE_ALIGNMENT,
            ) {
                Ok(arena) => {
                    caches.insert(
                        Scope::Device(*uuid),
                        ScopeCache {
                            tier: Tier::Device(DeviceTier::ExpertCache),
                            cap_bytes: *cap,
                            arena,
                            committed: 0,
                        },
                    );
                }
                Err(e) => {
                    let mut host = host;
                    let _ = host.release(ledger);
                    release_devices(ledger, device_envelope);
                    return Err(e);
                }
            }
        }

        Ok(Self {
            id: AuthorityId::next(),
            label: request.label.clone(),
            host,
            device_envelope,
            caches,
            index: BTreeMap::new(),
            placements: BTreeMap::new(),
            next_placement: 0,
            max_placements: request.max_placements,
            max_identity_bytes: request.max_identity_bytes,
            tickets: BTreeMap::new(),
            prefetch_queue: Vec::new(),
            prefetch_queue_capacity: request.prefetch_queue_capacity,
            next_sequence: 0,
            outstanding_demand: 0,
            claimed_backings: BTreeMap::new(),
            closed: false,
            conditional_floor_bytes: request.conditional_floor_bytes,
            conditional_resident: 0,
            lease_slots,
            free_lease_slots,
            live_leases: 0,
            stats: ResidencyStats::default(),
        })
    }

    /// How many placements are in a state a transfer may still be touching.
    ///
    /// Non-zero means [`ResidencyAuthority::close`] will refuse, and means a
    /// drop will withhold the host storage rather than free it.
    pub fn in_flight_placements(&self) -> usize {
        self.placements
            .values()
            .filter(|p| p.state.is_in_flight())
            .count()
    }

    /// Release the envelope. Refused while anything is leased or in flight:
    /// closing over live bytes is the leak, not the fix.
    pub fn close(&mut self, ledger: &mut Ledger) -> Result<()> {
        if self.closed {
            return Err(invalid("close", "this authority is already closed"));
        }
        if self.live_leases > 0 {
            return Err(invalid(
                "close",
                format!("{} lease(s) are still live", self.live_leases),
            ));
        }
        if let Some(p) = self.placements.values().find(|p| p.state.is_in_flight()) {
            return Err(invalid(
                "close",
                format!("chunk {} is {}", p.chunk, p.state.name()),
            ));
        }
        if let Some((scope, _)) = self.claimed_backings.iter().find(|(_, out)| **out) {
            // Releasing the reservation now would let the ledger report zero
            // while the card still holds the allocation it paid for.
            return Err(invalid(
                "close",
                format!("{scope}'s cache is still physically backed; return its backing first"),
            ));
        }
        let keys: Vec<PlacementKey> = self.placements.keys().copied().collect();
        for key in keys {
            self.drop_placement(key);
        }
        self.host.release(ledger)?;
        if let Some(envelope) = self.device_envelope.take() {
            ledger.release(envelope).map_err(|refused| refused.error)?;
        }
        // Past this point the reservation is gone, so the configuration this
        // authority still remembers is a description of memory it no longer
        // owns. Everything that could spend against it refuses.
        self.closed = true;
        Ok(())
    }

    /// Claim the right to back one scope's cache with one real allocation.
    ///
    /// Exactly once per scope per authority: a second claim is refused, because
    /// a second allocation would be memory the reservation never covered.
    pub fn claim_backing(&mut self, scope: Scope) -> Result<DeviceBacking> {
        self.check_open("claim_backing")?;
        let capacity = self
            .caches
            .get(&scope)
            .map(|c| c.cap_bytes)
            .ok_or_else(|| invalid("scope", format!("no residency cache is open for {scope}")))?;
        if self.claimed_backings.get(&scope).copied().unwrap_or(false) {
            return Err(invalid(
                "scope",
                format!("{scope}'s cache is already backed; one allocation, one reservation"),
            ));
        }
        self.claimed_backings.insert(scope, true);
        Ok(DeviceBacking {
            authority: self.id,
            scope,
            capacity,
        })
    }

    /// Give a backing back, after its physical allocation is gone.
    ///
    /// Refused while the scope still holds placements: the ranges they name
    /// live inside the allocation this entitlement stands for.
    /// Whether this backing may be given back yet, without taking it.
    ///
    /// The physical free happens **between** this question and
    /// [`ResidencyAuthority::return_backing`], because `try_free` leaves the
    /// allocation live when it fails: an entitlement surrendered before the
    /// memory is actually gone would let a second allocation, or this
    /// authority's own `close`, proceed over memory that still exists.
    pub fn check_returnable(&self, backing: &DeviceBacking) -> Result<()> {
        if backing.authority != self.id {
            return Err(invalid("backing", "backing belongs to another authority"));
        }
        if self.placements.values().any(|p| p.scope == backing.scope) {
            return Err(invalid(
                "backing",
                format!(
                    "{} still holds placements; their ranges live in this allocation",
                    backing.scope
                ),
            ));
        }
        Ok(())
    }

    pub fn return_backing(
        &mut self,
        backing: DeviceBacking,
    ) -> std::result::Result<(), ReturnBackingRefused> {
        // A refusal hands the entitlement back. Consuming it here would destroy
        // the only handle to a claim that is still outstanding -- the same
        // shape of leak as R08, created by the error path instead of the happy
        // one, and the reason `ReleaseRefused` and `ReleaseLeaseRefused` are
        // written this way too.
        let error = if backing.authority != self.id {
            Some(invalid("backing", "backing belongs to another authority"))
        } else if self.placements.values().any(|p| p.scope == backing.scope) {
            Some(invalid(
                "backing",
                format!(
                    "{} still holds placements; their ranges live in this allocation",
                    backing.scope
                ),
            ))
        } else {
            None
        };
        if let Some(error) = error {
            return Err(ReturnBackingRefused { backing, error });
        }
        self.claimed_backings.insert(backing.scope, false);
        Ok(())
    }

    /// Check every structural invariant this authority maintains, and name the
    /// first one that does not hold.
    ///
    /// **Why this is production code and not a test helper.** Three rounds of
    /// independent review found defects in the same shape: a transition that
    /// was individually reasonable left the structure inconsistent in a
    /// combination nobody had written a test for, and the damage surfaced one
    /// or two operations later as a panic. Point regressions caught each case
    /// and missed the next. A checker that states the invariants *once*, and can
    /// be run after any operation, is the thing those tests were each
    /// approximating.
    ///
    /// It is pure, allocates a message only on failure, and is called by the
    /// exhaustive transition sweep after **every** step.
    pub fn check_invariants(&self) -> Result<()> {
        let fail = |detail: String| Err(invalid("invariant", detail));

        for (key, p) in &self.placements {
            // The one that panicked twice: an in-flight state is a promise that
            // a transfer is coming.
            if p.state.is_in_flight() && p.state != ChunkState::Quarantined {
                match p.ticket {
                    None => {
                        return fail(format!("{} is {} with no ticket", p.chunk, p.state.name()));
                    }
                    Some(ticket) if !self.tickets.contains_key(&ticket) => {
                        return fail(format!(
                            "{} is {} naming a ticket that is gone",
                            p.chunk,
                            p.state.name()
                        ));
                    }
                    Some(_) => {}
                }
            }
            if let Some(source) = p.upload_source
                && !self.placements.contains_key(&source)
            {
                return fail(format!("{} names a source that is gone", p.chunk));
            }
            // `Retiring` means "freed when the last holder lets go". With no
            // holder left it is not retiring, it is stranded -- charged,
            // unservable and unevictable. The checker accepted exactly this
            // state while a review found a one-chunk cache held shut by it.
            if p.state == ChunkState::Retiring && p.leases == 0 {
                return fail(format!("{} is retiring with nothing holding it", p.chunk));
            }
            if !self.caches.contains_key(&p.scope) {
                return fail(format!("{} sits in a scope with no cache", p.chunk));
            }
            if self
                .index
                .get(&p.scope)
                .and_then(|m| m.get(p.chunk.as_ref()))
                != Some(key)
            {
                return fail(format!("{} is not indexed at its own key", p.chunk));
            }
        }

        for (id, t) in &self.tickets {
            if !self.placements.contains_key(&t.placement) {
                return fail(format!(
                    "ticket {} names a placement that is gone",
                    id.get()
                ));
            }
            if !self.placements.contains_key(&t.source) {
                return fail(format!("ticket {} names a source that is gone", id.get()));
            }
            if let Stage::BlockedOnRead(dep) = t.stage
                && !self.tickets.contains_key(&dep)
            {
                return fail(format!(
                    "ticket {} waits on a dependency that is gone",
                    id.get()
                ));
            }
            // A ticket that is not waiting must be able to say what it wants.
            if !matches!(t.stage, Stage::BlockedOnRead(_)) && self.work_order_for(*id).is_none() {
                return fail(format!("ticket {} has a stage but no order", id.get()));
            }
            let queued_here = self
                .prefetch_queue
                .iter()
                .filter(|(_, _, q)| q == id)
                .count();
            if t.queued != (queued_here == 1) || queued_here > 1 {
                return fail(format!(
                    "ticket {} says queued={} but appears {queued_here} time(s) in the queue",
                    id.get(),
                    t.queued
                ));
            }
        }
        for (_, _, id) in &self.prefetch_queue {
            if !self.tickets.contains_key(id) {
                return fail(format!("the prefetch queue holds dead ticket {}", id.get()));
            }
        }

        // The counter three findings were about.
        let demand = self
            .tickets
            .values()
            .filter(|t| !t.queued && t.urgency == Urgency::Demand)
            .count() as u32;
        if demand != self.outstanding_demand {
            return fail(format!(
                "outstanding_demand is {} but {demand} ticket(s) answer that description",
                self.outstanding_demand
            ));
        }

        for (scope, cache) in &self.caches {
            let bytes: u64 = self
                .placements
                .values()
                .filter(|p| p.scope == *scope)
                .map(|p| p.bytes)
                .sum();
            if bytes != cache.committed {
                return fail(format!(
                    "{scope} committed {} B against {bytes} B of placements",
                    cache.committed
                ));
            }
            let live = cache.arena.occupancy().live_bytes;
            if bytes != live {
                return fail(format!("{scope} holds {bytes} B against {live} B of arena"));
            }
            if cache.committed > cache.cap_bytes {
                return fail(format!(
                    "{scope} committed {} B over a {} B cap",
                    cache.committed, cache.cap_bytes
                ));
            }
        }

        let conditional: u64 = self
            .placements
            .values()
            .filter(|p| p.class.content == Content::ConditionalTable)
            .map(|p| p.bytes)
            .sum();
        if conditional != self.conditional_resident {
            return fail(format!(
                "conditional_resident is {} against {conditional} B resident",
                self.conditional_resident
            ));
        }

        let occupied = self.lease_slots.iter().filter(|s| s.held.is_some()).count() as u32;
        if occupied != self.live_leases {
            return fail(format!(
                "live_leases is {} against {occupied} occupied slot(s)",
                self.live_leases
            ));
        }
        // A lease **may** outlive its placement, and that is deliberate: a read
        // that fails discards its bytes whoever holds them, and the holders'
        // leases resolve to a typed error rather than to a range that never
        // arrived. So a dangling lease is legal.
        //
        // What must balance exactly is the count. A placement's `leases` is the
        // number of consumer leases naming it **plus** one internal pin for
        // every ticket copying out of it -- document 02's "an upload owns or
        // leases its source bytes through a completion event". Stating it as an
        // identity rather than a bound is what catches a pin released twice or
        // never released at all: a leaked pin makes a chunk permanently
        // unevictable, which is invisible until a cache stops admitting.
        for (key, p) in &self.placements {
            let consumers = self
                .lease_slots
                .iter()
                .filter_map(|s| s.held)
                .filter(|h| h.placement == *key)
                .count() as u32;
            let ticket_pins = self
                .tickets
                .values()
                .filter(|t| t.source == *key && t.source != t.placement)
                .count() as u32;
            // A withheld copy holds its source too. `SubmissionUnknown` takes
            // the ticket away but the copy may still be reading, so the pin
            // outlives it and `settle_quarantined` is what releases it (R07).
            let quarantine_pins = self
                .placements
                .values()
                .filter(|q| q.state == ChunkState::Quarantined && q.upload_source == Some(*key))
                .count() as u32;
            let pins = ticket_pins + quarantine_pins;
            if p.leases != consumers + pins {
                return fail(format!(
                    "{} counts {} lease(s) against {consumers} consumer(s) and {pins} pin(s)",
                    p.chunk, p.leases
                ));
            }
        }
        Ok(())
    }

    /// Whether this authority has been closed. A closed one owns nothing.
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    fn check_open(&self, what: &'static str) -> Result<()> {
        if self.closed {
            return Err(invalid(
                what,
                "this authority is closed; its reservation has been released",
            ));
        }
        Ok(())
    }

    pub const fn id(&self) -> AuthorityId {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn stats(&self) -> ResidencyStats {
        self.stats
    }

    /// Bytes resident or in flight in one scope.
    pub fn committed_bytes(&self, scope: Scope) -> Option<u64> {
        self.caches.get(&scope).map(|c| c.committed)
    }

    pub fn cap_bytes(&self, scope: Scope) -> Option<u64> {
        self.caches.get(&scope).map(|c| c.cap_bytes)
    }

    /// Exact arena occupancy for one scope, for reconciliation against
    /// `committed_bytes` and against the ledger.
    pub fn occupancy(&self, scope: Scope) -> Option<crate::ArenaOccupancy> {
        self.caches.get(&scope).map(|c| c.arena.occupancy())
    }

    /// Every placement the authority holds. A dropped lease's pin shows up
    /// here, which is the failure mode that can be found.
    pub fn outstanding(&self) -> Vec<OutstandingChunk> {
        self.placements
            .values()
            .map(|p| OutstandingChunk {
                chunk: (*p.chunk).clone(),
                scope: p.scope,
                state: p.state,
                class: p.class,
                bytes: p.bytes,
                leases: p.leases,
                last_used: p.last_used,
            })
            .collect()
    }

    pub fn live_lease_count(&self) -> usize {
        self.live_leases as usize
    }

    pub fn in_flight_count(&self) -> usize {
        self.tickets.len()
    }

    pub fn prefetch_queue_len(&self) -> usize {
        self.prefetch_queue.len()
    }

    /// The ticket outstanding against one placement, if any.
    ///
    /// Diagnostic. A driver that lost track of what it was performing can ask;
    /// the transition sweep uses it to settle whatever it finds in flight.
    pub fn ticket_of(&self, scope: Scope, chunk: &ChunkId) -> Option<TicketId> {
        self.index
            .get(&scope)
            .and_then(|m| m.get(chunk))
            .and_then(|k| self.placements.get(k))
            .and_then(|p| p.ticket)
    }

    /// Whether a chunk is ready at a scope right now.
    pub fn state_of(&self, scope: Scope, chunk: &ChunkId) -> Option<ChunkState> {
        self.index
            .get(&scope)
            .and_then(|m| m.get(chunk))
            .and_then(|k| self.placements.get(k))
            .map(|p| p.state)
    }
}

// ---------------------------------------------------------------------------
// Acquisition
// ---------------------------------------------------------------------------

/// One residency request. Document 03 writes
/// `acquire(chunk, destination, deadline, use_class)`; `now` and `turn` are the
/// two additions this implementation makes, each for a stated reason: the
/// authority reads no clock, because a clock in the accounting core makes every
/// test a race, and a lease belongs to a turn, because R08 is a lease that
/// outlived the turn that took it.
#[derive(Debug, Clone, Copy)]
pub struct AcquireRequest<'a> {
    pub chunk: &'a ChunkId,
    pub destination: Scope,
    pub now: u64,
    pub deadline: u64,
    pub class: UseClass,
    pub turn: TurnId,
}

impl ResidencyAuthority {
    /// Admit, coalesce, or refuse -- and **never wait**.
    ///
    /// The absence of a waiting path is the mechanism, not an optimisation.
    /// Document 03 requires admission to "prohibit deadlock when all evictable
    /// entries are leased", and a call that cannot block cannot take part in
    /// one. When nothing can be displaced, this returns `CapacityExceeded` with
    /// the report that explains why, and the authority stays usable.
    ///
    /// The hit path performs exactly one map lookup and **allocates nothing**:
    /// no report is built, because a hit admits nothing and evicts nothing.
    // The refusal carries the breakdown document 03 requires a rejection to
    // show. Boxing it to save a move on the error path would put the caller's
    // only explanation behind an allocation.
    #[allow(clippy::result_large_err)]
    pub fn acquire(
        &mut self,
        request: AcquireRequest<'_>,
    ) -> std::result::Result<Acquired, ResidencyRefused> {
        if self.closed {
            let report = self.report_for(Scope::Host, request.chunk.len_bytes(), &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: invalid(
                    "acquire",
                    "this authority is closed; its reservation has been released",
                ),
                report,
            });
        }
        if !self.caches.contains_key(&request.destination) {
            let report = self.report_for(Scope::Host, request.chunk.len_bytes(), &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: invalid(
                    "destination",
                    format!("no residency cache is open for {}", request.destination),
                ),
                report,
            });
        }
        let identity_bytes = request.chunk.identity_bytes();
        if identity_bytes > u64::from(self.max_identity_bytes) {
            // The envelope was admitted from `max_identity_bytes`. Admitting a
            // longer one would spend control memory nobody reserved, which is
            // the exact defect this bound replaced.
            let report = self.report_for(request.destination, request.chunk.len_bytes(), &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: Error::CapacityExceeded {
                    tier: Some(Tier::Host(HostTier::Pageable)),
                    requested_bytes: identity_bytes,
                    available_bytes: u64::from(self.max_identity_bytes),
                },
                report,
            });
        }
        if request.class.content == Content::ConditionalTable && self.conditional_floor_bytes == 0 {
            // ADR 0009 is explicit: a conditional-memory class is NVMe-backed
            // with an admitted footprint and **never zero-resident**. A class
            // with no floor is one whose every entry is an eviction candidate,
            // which is the zero-resident case wearing the class's name. Refused
            // here rather than left to a caller to remember.
            let report = self.report_for(request.destination, request.chunk.len_bytes(), &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: invalid(
                    "conditional_floor_bytes",
                    "a conditional-memory class needs a nonzero resident floor (ADR 0009: \
                     never zero-resident); this authority declared none",
                ),
                report,
            });
        }
        if self.free_lease_slots.is_empty() {
            // The lease table is bounded and admitted, so exhausting it is a
            // capacity refusal like any other -- never a growth nobody charged
            // for, and never a panic on a path a generation step takes.
            let report = self.report_for(request.destination, request.chunk.len_bytes(), &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: Error::CapacityExceeded {
                    tier: Some(Tier::Host(HostTier::Pageable)),
                    requested_bytes: LEASE_CONTROL_BYTES,
                    available_bytes: 0,
                },
                report,
            });
        }

        if let Some(&key) = self
            .index
            .get(&request.destination)
            .and_then(|m| m.get(request.chunk))
        {
            return self.acquire_existing(key, request);
        }
        self.admit_new(request)
    }

    // The refusal carries the breakdown document 03 requires a rejection to
    // show. Boxing it to save a move on the error path would put the caller's
    // only explanation behind an allocation.
    #[allow(clippy::result_large_err)]
    fn acquire_existing(
        &mut self,
        key: PlacementKey,
        request: AcquireRequest<'_>,
    ) -> std::result::Result<Acquired, ResidencyRefused> {
        let state = self.placements[&key].state;

        if state.is_ready() {
            let p = self.placements.get_mut(&key).expect("indexed placement");
            p.last_used = request.now;
            if request.class.urgency == Urgency::Demand && !p.ever_demanded {
                p.ever_demanded = true;
                if p.class.urgency == Urgency::Prefetch {
                    let bytes = p.bytes;
                    self.stats.prefetch_used_bytes =
                        self.stats.prefetch_used_bytes.saturating_add(bytes);
                }
            }
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(Acquired::Ready(self.issue_lease(
                key,
                request.turn,
                request.class,
                None,
            )));
        }

        if matches!(state, ChunkState::Reading | ChunkState::Uploading) {
            let Some(ticket) = self.placements[&key].ticket else {
                // Unreachable if `retire_or_drop` is used on every terminal
                // path, and a typed refusal rather than a panic even so: an
                // invariant breach inside a generation step must be reportable,
                // not fatal. This exact shape panicked before that rule existed.
                let report = self.report_for(request.destination, request.chunk.len_bytes(), &[]);
                self.stats.refusals = self.stats.refusals.saturating_add(1);
                return Err(ResidencyRefused {
                    error: invalid(
                        "chunk",
                        format!(
                            "chunk {} is {} at {} with no transfer outstanding",
                            request.chunk,
                            state.name(),
                            request.destination
                        ),
                    ),
                    report,
                });
            };
            {
                let p = self.placements.get_mut(&key).expect("indexed placement");
                p.last_used = request.now;
                if request.class.urgency == Urgency::Demand {
                    p.ever_demanded = true;
                }
            }
            let report = self.report_for(request.destination, 0, &[]);
            let lease = self.issue_lease(key, request.turn, request.class, Some(ticket));
            let promote = {
                let t = self.tickets.get_mut(&ticket).expect("live ticket");
                t.waiters = t.waiters.saturating_add(1);
                // A demand arriving on a prediction promotes it: the step is now
                // blocked on bytes a guess asked for first, so it stops being a
                // guess. Demand outranking prediction is about queue order, not
                // about punishing a correct one.
                request.class.urgency == Urgency::Demand && t.urgency == Urgency::Prefetch
            };
            // Promotion takes the ticket off the prefetch queue, so whoever
            // promoted it is the only caller that can still be handed its
            // order. Returning `Coalesced` here would strand the work: the
            // queue no longer holds it and nobody else will ask.
            let work = if promote {
                match self.promote_ticket(ticket, request.deadline) {
                    Some(order) => PendingWork::Issued(order),
                    None => PendingWork::Coalesced,
                }
            } else {
                PendingWork::Coalesced
            };
            self.stats.misses = self.stats.misses.saturating_add(1);
            return Ok(Acquired::Pending {
                lease,
                ticket,
                work,
                report,
            });
        }

        // Quarantined or retiring. The bytes exist but may not be served, and
        // reading into them would race a transfer that may still be running.
        // A refusal, never a silent second copy.
        let report = self.report_for(request.destination, request.chunk.len_bytes(), &[]);
        self.stats.refusals = self.stats.refusals.saturating_add(1);
        Err(ResidencyRefused {
            error: invalid(
                "chunk",
                format!(
                    "chunk {} is {} at {}; it cannot be acquired until that settles",
                    request.chunk,
                    state.name(),
                    request.destination
                ),
            ),
            report,
        })
    }

    // The refusal carries the breakdown document 03 requires a rejection to
    // show. Boxing it to save a move on the error path would put the caller's
    // only explanation behind an allocation.
    #[allow(clippy::result_large_err)]
    fn admit_new(
        &mut self,
        request: AcquireRequest<'_>,
    ) -> std::result::Result<Acquired, ResidencyRefused> {
        let incoming = request.chunk.len_bytes();
        let destination = request.destination;

        // A device acquire of an absent chunk creates **two** placements: the
        // device range and the host source it is copied from. Checking room for
        // one and then making two is how a declared bound stops bounding
        // anything.
        let needed = 1 + usize::from(
            destination.kind() == moxie_types::ScopeKind::Device
                && self
                    .index
                    .get(&Scope::Host)
                    .and_then(|m| m.get(request.chunk))
                    .is_none(),
        );
        if self.placements.len() + needed > self.max_placements as usize {
            let report = self.report_for(destination, incoming, &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: Error::CapacityExceeded {
                    tier: Some(Tier::Host(HostTier::Pageable)),
                    requested_bytes: PLACEMENT_CONTROL_BYTES,
                    available_bytes: 0,
                },
                report,
            });
        }
        if request.class.urgency == Urgency::Prefetch
            && self.prefetch_queue.len() >= self.prefetch_queue_capacity as usize
        {
            // A bounded queue that grows under pressure is an unbounded queue.
            // Refused before anything is admitted, so a refused prediction
            // costs no bytes at all.
            let report = self.report_for(destination, incoming, &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: Error::CapacityExceeded {
                    tier: Some(self.caches[&destination].tier),
                    requested_bytes: incoming,
                    available_bytes: 0,
                },
                report,
            });
        }

        // A device chunk is copied from host bytes, so the host side is settled
        // first. Admitting the device range before its source exists would be
        // the scratch source document 02 forbids.
        let device = destination.kind() == moxie_types::ScopeKind::Device;
        let source = if device {
            Some(self.resolve_host_source(request)?)
        } else {
            None
        };

        let (key, report) = match self.place(destination, request, incoming) {
            Ok(v) => v,
            Err(e) => {
                if let Some(HostSource { key, created, .. }) = source {
                    self.release_source_pin(key, created);
                }
                return Err(e);
            }
        };

        let ticket = TicketId::next();
        let queued = request.class.urgency == Urgency::Prefetch;
        let source_created = source.as_ref().is_some_and(|s| s.created);
        let source_key_opt = source.as_ref().map(|s| s.key);
        // The placement is admitted, so the promotion this acquire depends on
        // is now known to be wanted. Applying it any earlier would strand the
        // prediction when admission failed.
        let promoted = source
            .as_ref()
            .and_then(|s| s.promote)
            .and_then(|dep| self.promote_ticket(dep, request.deadline));
        let (stage, source_key, state) = match source.as_ref() {
            None => (Stage::Read, key, ChunkState::Reading),
            Some(HostSource {
                key: src,
                created: true,
                ..
            }) => (Stage::Read, *src, ChunkState::Reading),
            Some(HostSource {
                key: src,
                pending: None,
                ..
            }) => (Stage::Upload, *src, ChunkState::Uploading),
            Some(HostSource {
                key: src,
                pending: Some(blocking),
                ..
            }) => (Stage::BlockedOnRead(*blocking), *src, ChunkState::Reading),
        };

        {
            let p = self.placements.get_mut(&key).expect("just placed");
            p.ticket = Some(ticket);
            p.upload_source = source_key_opt;
            p.state = state;
        }
        if let Some(src) = source.as_ref().filter(|s| s.created) {
            let p = self
                .placements
                .get_mut(&src.key)
                .expect("just placed source");
            p.ticket = Some(ticket);
            p.state = ChunkState::Reading;
        }

        self.tickets.insert(
            ticket,
            Ticket {
                placement: key,
                source: source_key,
                stage,
                urgency: request.class.urgency,
                deadline: request.deadline,
                waiters: 1,
                cancelled: false,
                queued,
                issued: false,
                owns_source: source_created,
            },
        );

        let work = if queued {
            let sequence = self.next_sequence;
            self.next_sequence += 1;
            self.prefetch_queue
                .push((request.deadline, sequence, ticket));
            self.prefetch_queue.sort_unstable();
            self.stats.prefetch_admitted_bytes =
                self.stats.prefetch_admitted_bytes.saturating_add(incoming);
            PendingWork::Queued
        } else if matches!(stage, Stage::BlockedOnRead(_)) {
            self.outstanding_demand = self.outstanding_demand.saturating_add(1);
            // If resolving the source promoted a queued prediction, that
            // prediction's read is what this acquire is waiting for, and this
            // caller is the only one still able to perform it.
            match promoted {
                Some(order) => PendingWork::Issued(order),
                None => PendingWork::Coalesced,
            }
        } else {
            self.outstanding_demand = self.outstanding_demand.saturating_add(1);
            self.tickets.get_mut(&ticket).expect("just inserted").issued = true;
            match self.work_order_for(ticket) {
                Some(order) => PendingWork::Issued(order),
                // A ticket with no order of its own is waiting on another's
                // read; the caller waits with it.
                None => PendingWork::Coalesced,
            }
        };

        let lease = self.issue_lease(key, request.turn, request.class, Some(ticket));
        self.stats.misses = self.stats.misses.saturating_add(1);
        Ok(Acquired::Pending {
            lease,
            ticket,
            work,
            report,
        })
    }

    /// Resolve, and pin, the host bytes a device acquire will copy from.
    // The refusal carries the breakdown document 03 requires a rejection to
    // show. Boxing it to save a move on the error path would put the caller's
    // only explanation behind an allocation.
    #[allow(clippy::result_large_err)]
    fn resolve_host_source(
        &mut self,
        request: AcquireRequest<'_>,
    ) -> std::result::Result<HostSource, ResidencyRefused> {
        if let Some(&existing) = self
            .index
            .get(&Scope::Host)
            .and_then(|m| m.get(request.chunk))
        {
            let p = self.placements.get_mut(&existing).expect("indexed");
            match p.state {
                ChunkState::HostReady => {
                    // Document 02: "An upload owns or leases its source bytes
                    // through a completion event." The pin is that lease, and
                    // it makes the source unevictable for the copy's lifetime.
                    p.leases = p.leases.saturating_add(1);
                    p.last_used = request.now;
                    Ok(HostSource {
                        key: existing,
                        created: false,
                        pending: None,
                        promote: None,
                    })
                }
                ChunkState::Reading => {
                    let blocking = p.ticket.expect("a reading placement owns a ticket");
                    p.leases = p.leases.saturating_add(1);
                    p.last_used = request.now;
                    // A demand that waits on a *queued prediction* is a
                    // dependency cycle: this acquire is demand, so
                    // `next_prefetch` will refuse to release the read it is
                    // waiting for, and neither ever finishes. Priority has to
                    // propagate through the dependency -- but only once this
                    // acquire is actually admitted.
                    let promote = (request.class.urgency == Urgency::Demand
                        && self.tickets[&blocking].urgency == Urgency::Prefetch)
                        .then_some(blocking);
                    Ok(HostSource {
                        key: existing,
                        created: false,
                        pending: Some(blocking),
                        promote,
                    })
                }
                state => {
                    let report = self.report_for(Scope::Host, request.chunk.len_bytes(), &[]);
                    self.stats.refusals = self.stats.refusals.saturating_add(1);
                    Err(ResidencyRefused {
                        error: invalid(
                            "chunk",
                            format!(
                                "chunk {} is {} on the host; a device acquire needs settled \
                                 host bytes",
                                request.chunk,
                                state.name()
                            ),
                        ),
                        report,
                    })
                }
            }
        } else {
            let host_request = AcquireRequest {
                destination: Scope::Host,
                ..request
            };
            let (key, _) = self.place(Scope::Host, host_request, request.chunk.len_bytes())?;
            self.placements.get_mut(&key).expect("just placed").leases = 1;
            Ok(HostSource {
                key,
                created: true,
                pending: None,
                promote: None,
            })
        }
    }

    /// Undo a source pin taken by an acquire that then failed to be admitted.
    ///
    /// Only the rollback path inside `acquire` uses this: nothing has been
    /// handed out, so a source this acquire created can go immediately.
    fn release_source_pin(&mut self, key: PlacementKey, created: bool) {
        let drop_it = {
            let Some(p) = self.placements.get_mut(&key) else {
                return;
            };
            p.leases = p.leases.saturating_sub(1);
            created && p.leases == 0
        };
        if drop_it {
            self.drop_placement(key);
        }
    }

    /// Make room for `incoming` in `scope`, then record the placement.
    ///
    /// The order is the contract. `committed + incoming` against the cap and
    /// the victim list are both computed **before** a single victim is removed,
    /// which is document 03's "include incoming expert size before
    /// evicting/allocating". The report's free and fragmentation figures are
    /// read after the evictions it names, because those are the real ones.
    ///
    /// Eviction happens in two phases, and the second one is not an afterthought.
    /// Byte accounting says whether the cache *holds* enough; the arena says
    /// whether the free bytes are in one piece. Document 03 tracks
    /// `allocator_fragmentation` as real capacity precisely because those two
    /// answers differ, so when the arena refuses a request the bytes admitted,
    /// this keeps evicting in the same deterministic order rather than reporting
    /// a full cache that is not full.
    #[allow(clippy::result_large_err)]
    fn place(
        &mut self,
        scope: Scope,
        request: AcquireRequest<'_>,
        incoming: u64,
    ) -> std::result::Result<(PlacementKey, ResidencyReport), ResidencyRefused> {
        let cap = self.caches[&scope].cap_bytes;
        if incoming > cap {
            // It would not fit an empty cache, so evicting first would destroy
            // live data to no purpose. Nothing is displaced.
            let report = self.report_for(scope, incoming, &[]);
            self.stats.refusals = self.stats.refusals.saturating_add(1);
            return Err(ResidencyRefused {
                error: Error::CapacityExceeded {
                    tier: Some(self.caches[&scope].tier),
                    requested_bytes: incoming,
                    available_bytes: cap,
                },
                report,
            });
        }

        let mut candidates = self.eviction_candidates(scope, request.class).into_iter();
        let mut conditional = self.conditional_resident;
        let floor = self.conditional_floor_bytes;
        let committed_before = self.caches[&scope].committed;

        // Phase one: enough bytes.
        let mut need = committed_before.saturating_add(incoming);
        let mut freed = 0u64;
        let mut victims: Vec<Candidate> = Vec::new();
        while need > cap {
            match next_allowed(&mut candidates, &mut conditional, floor) {
                Some(c) => {
                    need = need.saturating_sub(c.bytes);
                    freed = freed.saturating_add(c.bytes);
                    victims.push(c);
                }
                None => {
                    let named = victims.iter().map(|c| (*c.chunk).clone()).collect();
                    let report = self.build_report(
                        scope,
                        incoming,
                        committed_before,
                        committed_before.saturating_sub(freed),
                        named,
                    );
                    self.stats.refusals = self.stats.refusals.saturating_add(1);
                    return Err(ResidencyRefused {
                        error: Error::CapacityExceeded {
                            tier: Some(self.caches[&scope].tier),
                            requested_bytes: incoming,
                            available_bytes: cap
                                .saturating_sub(committed_before.saturating_sub(freed)),
                        },
                        report,
                    });
                }
            }
        }

        let mut named: Vec<ChunkId> = victims.iter().map(|c| (*c.chunk).clone()).collect();
        for c in &victims {
            self.evict(c.key);
        }

        // Phase two: enough *contiguous* bytes.
        let allocation = loop {
            let cache = self.caches.get_mut(&scope).expect("checked");
            match cache.arena.allocate(incoming, CACHE_ALIGNMENT, "residency") {
                Ok(a) => break a,
                Err(refused) => match next_allowed(&mut candidates, &mut conditional, floor) {
                    Some(c) => {
                        named.push((*c.chunk).clone());
                        self.evict(c.key);
                    }
                    None => {
                        let report = self.report_after(scope, incoming, committed_before, named);
                        self.stats.refusals = self.stats.refusals.saturating_add(1);
                        return Err(ResidencyRefused {
                            error: refused.error,
                            report,
                        });
                    }
                },
            }
        };

        let offset = allocation.offset();
        // Captured before this placement is charged: the report's headroom is
        // what is left *after* admitting it, computed once, not twice.
        let committed_after_evictions = self.caches[&scope].committed;
        let cache = self.caches.get_mut(&scope).expect("checked");
        cache.committed = cache.committed.saturating_add(incoming);

        let key = PlacementKey(self.next_placement);
        self.next_placement += 1;
        if request.class.content == Content::ConditionalTable {
            self.conditional_resident = self.conditional_resident.saturating_add(incoming);
        }
        // One identity, two holders. The index key and the placement record
        // share it, so a chunk's name costs one allocation rather than two --
        // which is what makes a per-placement control bound derivable at all.
        let identity = Arc::new(request.chunk.clone());
        self.placements.insert(
            key,
            Placement {
                chunk: Arc::clone(&identity),
                scope,
                class: request.class,
                state: ChunkState::Reading,
                allocation: Some(allocation),
                offset,
                bytes: incoming,
                leases: 0,
                last_used: request.now,
                ever_demanded: request.class.urgency == Urgency::Demand,
                ticket: None,
                upload_source: None,
            },
        );
        self.index.entry(scope).or_default().insert(identity, key);
        let report = self.build_report(
            scope,
            incoming,
            committed_before,
            committed_after_evictions,
            named,
        );
        Ok((key, report))
    }

    /// Everything this scope could displace, in the one deterministic order.
    ///
    /// Deterministic demand LRU plus a bounded prefetch class: document 03's
    /// declared starting policy and nothing cleverer, because the same paragraph
    /// says to "add smarter policies only with replayable route traces and
    /// measured benefit", and neither exists yet.
    fn eviction_candidates(&self, scope: Scope, class: UseClass) -> Vec<Candidate> {
        let mut candidates: Vec<(u8, u64, &Placement, PlacementKey)> = self
            .placements
            .iter()
            .filter(|(_, p)| p.scope == scope && p.is_evictable())
            .map(|(k, p)| (p.class_rank(), p.last_used, p, *k))
            .collect();
        // `(class_rank, last_used, chunk)`. The chunk identity is last so the
        // choice depends on nothing but the access history -- not on map
        // iteration order, not on insertion order, not on how many placements
        // happen to share a tick.
        candidates.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.chunk.cmp(&b.2.chunk))
        });
        candidates
            .into_iter()
            // Document 03 makes speculative prefetch "bounded and evictable
            // before useful demand data". That is a one-way rule: a prediction
            // may never cost a step bytes it already holds, so a prefetch's
            // candidate list stops where demand data begins.
            .take_while(|(rank, _, _, _)| class.urgency == Urgency::Demand || *rank == 0)
            .map(|(_, _, p, key)| Candidate {
                key,
                bytes: p.bytes,
                content: p.class.content,
                chunk: p.chunk.clone(),
            })
            .collect()
    }
}

/// One placement eviction may take, resolved before any `&mut self` call.
#[derive(Debug, Clone)]
struct Candidate {
    key: PlacementKey,
    bytes: u64,
    content: Content,
    chunk: Arc<ChunkId>,
}

/// The next candidate the conditional-memory floor permits.
///
/// ADR 0009 requires a conditional-memory class to be "never zero-resident", so
/// a table row is skipped -- not refused -- when displacing it would take the
/// class below its floor. Skipping rather than stopping matters: the next
/// candidate may be an expert, and refusing the whole admission because one
/// protected row came first would make the floor a denial of service.
fn next_allowed(
    candidates: &mut std::vec::IntoIter<Candidate>,
    conditional: &mut u64,
    floor: u64,
) -> Option<Candidate> {
    for candidate in candidates.by_ref() {
        if candidate.content == Content::ConditionalTable {
            let after = conditional.saturating_sub(candidate.bytes);
            if after < floor {
                continue;
            }
            *conditional = after;
        }
        return Some(candidate);
    }
    None
}

/// Where a device acquire's source bytes came from.
///
/// Every variant leaves the source **pinned**, and exactly one terminal
/// transition of the ticket releases that pin. Document 02 is the reason the
/// pin exists at all: "An upload owns or leases its source bytes through a
/// completion event."
#[derive(Debug, Clone)]
struct HostSource {
    key: PlacementKey,
    /// This acquire created the host placement, so it owns the read.
    created: bool,
    /// Another ticket is already reading it; this one waits for that.
    pending: Option<TicketId>,
    /// A queued prediction this acquire will need promoted, recorded but **not
    /// yet applied**.
    ///
    /// Promotion mutates the prefetch queue and the demand counter, and the
    /// device placement this acquire wants may still be refused. Applying it
    /// first and rolling back the placement stranded the original prediction:
    /// off the queue, counted as demand, owned by nobody. So it is applied
    /// after admission succeeds, which is the only point at which it is known
    /// to be wanted.
    promote: Option<TicketId>,
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

impl ResidencyAuthority {
    /// The report for a request that admitted nothing.
    fn report_for(&self, scope: Scope, incoming: u64, victims: &[PlacementKey]) -> ResidencyReport {
        let named = victims
            .iter()
            .filter_map(|k| self.placements.get(k))
            .map(|p| (*p.chunk).clone())
            .collect();
        let freed: u64 = victims
            .iter()
            .filter_map(|k| self.placements.get(k))
            .map(|p| p.bytes)
            .sum();
        let committed = self.caches[&scope].committed;
        self.build_report(
            scope,
            incoming,
            committed,
            committed.saturating_sub(freed),
            named,
        )
    }

    /// The report for a request whose evictions have already happened.
    fn report_after(
        &self,
        scope: Scope,
        incoming: u64,
        committed_before: u64,
        named: Vec<ChunkId>,
    ) -> ResidencyReport {
        self.build_report(
            scope,
            incoming,
            committed_before,
            self.caches[&scope].committed,
            named,
        )
    }

    fn build_report(
        &self,
        scope: Scope,
        incoming: u64,
        committed_before: u64,
        committed_after_evictions: u64,
        would_evict: Vec<ChunkId>,
    ) -> ResidencyReport {
        let cache = &self.caches[&scope];
        let mut leased = 0u64;
        let mut evictable = 0u64;
        for p in self.placements.values().filter(|p| p.scope == scope) {
            if p.is_evictable() {
                evictable = evictable.saturating_add(p.bytes);
            } else {
                leased = leased.saturating_add(p.bytes);
            }
        }
        let occupancy = cache.arena.occupancy();
        ResidencyReport {
            scope,
            tier: cache.tier,
            cap_bytes: cache.cap_bytes,
            committed_bytes: committed_before,
            incoming_bytes: incoming,
            leased_bytes: leased,
            evictable_bytes: evictable,
            would_evict,
            free_bytes: occupancy.free_bytes,
            largest_free_bytes: occupancy.largest_free_bytes,
            remaining_headroom_bytes: cache
                .cap_bytes
                .saturating_sub(committed_after_evictions.saturating_add(incoming)),
        }
    }
}

// ---------------------------------------------------------------------------
// Work, completion and failure
// ---------------------------------------------------------------------------

impl ResidencyAuthority {
    fn issue_lease(
        &mut self,
        placement: PlacementKey,
        turn: TurnId,
        class: UseClass,
        ticket: Option<TicketId>,
    ) -> ResidencyLease {
        let slot = self
            .free_lease_slots
            .pop()
            .expect("a lease slot was reserved before the placement was admitted");
        let p = self
            .placements
            .get_mut(&placement)
            .expect("placement exists");
        p.leases += 1;
        let scope = p.scope;
        let entry = &mut self.lease_slots[slot as usize];
        entry.held = Some(LeaseRecord {
            placement,
            turn,
            ticket,
        });
        let id = ResidencyLeaseId {
            slot,
            generation: entry.generation,
        };
        self.live_leases += 1;
        ResidencyLease {
            id,
            authority: self.id,
            placement,
            scope,
            turn,
            class,
        }
    }

    fn lease_record(&self, id: ResidencyLeaseId) -> Option<LeaseRecord> {
        let slot = self.lease_slots.get(id.slot as usize)?;
        if slot.generation != id.generation {
            // The slot was reused. A stale identity names someone else's lease
            // and must never resolve to it.
            return None;
        }
        slot.held
    }

    /// Take a lease slot back, bumping its generation so the identity that held
    /// it can never be mistaken for the next one.
    fn free_lease(&mut self, id: ResidencyLeaseId) -> Option<LeaseRecord> {
        let slot = self.lease_slots.get_mut(id.slot as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        let held = slot.held.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free_lease_slots.push(id.slot);
        self.live_leases -= 1;
        Some(held)
    }

    /// Turn a prediction into demand, and hand back the order that stops being
    /// the queue's responsibility.
    ///
    /// Returns `Some` exactly when this promotion took an **unissued** ticket
    /// off the prefetch queue: the caller is then the only one who can still be
    /// given that work. `None` means the ticket was already outstanding as
    /// demand, or is waiting on another ticket's read and has no order of its
    /// own.
    fn promote_ticket(&mut self, ticket: TicketId, deadline: u64) -> Option<WorkOrder> {
        self.promote_chain(ticket, deadline, 0)
    }

    /// Promote a ticket **and everything it is waiting for**.
    ///
    /// The dependency is the point. A demand that waits on a prediction which
    /// itself waits on another prediction's read is still blocked by the
    /// prefetch gate unless priority travels the whole chain -- and the gate
    /// will not release anything while that demand is outstanding. Promoting
    /// only the ticket the caller named leaves exactly that deadlock, which an
    /// independent review reproduced twice.
    ///
    /// The returned order belongs to whichever ticket in the chain actually has
    /// work: a blocked ticket has none of its own, so the order comes from its
    /// dependency.
    fn promote_chain(&mut self, ticket: TicketId, deadline: u64, depth: u32) -> Option<WorkOrder> {
        // Chains are one or two links in practice. The bound is a guard against
        // a cycle becoming a stack overflow, never an expected limit.
        const MAX_DEPTH: u32 = 16;
        if depth >= MAX_DEPTH {
            return None;
        }
        let (was_queued, issued, stage, urgency_before) = {
            let t = self.tickets.get_mut(&ticket)?;
            let before = t.urgency;
            t.urgency = Urgency::Demand;
            t.deadline = t.deadline.min(deadline);
            (t.queued, t.issued, t.stage, before)
        };
        // The counter tracks "non-queued demand tickets", so a promotion adds
        // to it exactly when the ticket did not already answer that
        // description. Counting only the *queue* transition missed an
        // already-issued prediction becoming demand: it went uncounted, and
        // `take_ticket` -- which decrements every non-queued demand ticket --
        // then cleared somebody else's slot and let the gate open early.
        let held_slot_before = !was_queued && urgency_before == Urgency::Demand;
        if was_queued {
            self.prefetch_queue.retain(|(_, _, id)| *id != ticket);
            self.tickets.get_mut(&ticket).expect("live ticket").queued = false;
        }
        if !held_slot_before {
            self.outstanding_demand = self.outstanding_demand.saturating_add(1);
        }
        if let Stage::BlockedOnRead(dependency) = stage {
            // This ticket has no work of its own; what it is waiting for does.
            return self.promote_chain(dependency, deadline, depth + 1);
        }
        if issued || !was_queued {
            return None;
        }
        self.tickets.get_mut(&ticket).expect("live ticket").issued = true;
        self.work_order_for(ticket)
    }

    /// The order a ticket's current stage calls for, if it has one.
    ///
    /// `None` for a ticket waiting on another ticket's read: it has no work of
    /// its own. That used to be an `unreachable!`, and a review reached it --
    /// `next_prefetch` selected a blocked ticket by deadline order. An
    /// unreachable branch on a path a generation step takes is a crash waiting
    /// for the right queue order.
    fn work_order_for(&self, ticket: TicketId) -> Option<WorkOrder> {
        let t = self.tickets.get(&ticket)?;
        match t.stage {
            Stage::Read => {
                let src = self.placements.get(&t.source)?;
                Some(WorkOrder::Read {
                    ticket,
                    chunk: (*src.chunk).clone(),
                    host_offset: src.offset,
                    len_bytes: src.bytes,
                })
            }
            Stage::Upload => {
                let src = self.placements.get(&t.source)?;
                let p = self.placements.get(&t.placement)?;
                Some(WorkOrder::Upload {
                    ticket,
                    chunk: (*p.chunk).clone(),
                    scope: p.scope,
                    host_offset: src.offset,
                    device_offset: p.offset,
                    len_bytes: p.bytes,
                })
            }
            Stage::BlockedOnRead(_) => None,
        }
    }

    /// Release the next queued prefetch, if demand is idle.
    ///
    /// Returns `None` while any demand transfer is outstanding. That is
    /// document 03's "demand work outranks predictions" as a mechanism rather
    /// than as a priority number nobody checks.
    pub fn next_prefetch(&mut self) -> Option<WorkOrder> {
        if self.closed || self.outstanding_demand > 0 || self.prefetch_queue.is_empty() {
            return None;
        }
        // Deadline order decides *what to look at first*, not what is
        // executable. A queued ticket that is waiting on another ticket's read
        // has no work of its own, and issuing it by position alone panicked
        // reaching for an order that does not exist -- a device prefetch with
        // the earlier deadline sorted ahead of the very read it depends on.
        //
        // So the queue is scanned in order and each candidate is resolved to
        // the root of its dependency chain: the ticket that actually holds the
        // work. A blocked entry stays queued and is released by
        // `release_blocked_on` when its dependency completes.
        let mut chosen = None;
        for index in 0..self.prefetch_queue.len() {
            let (_, _, ticket) = self.prefetch_queue[index];
            let Some(root) = self.chain_root(ticket) else {
                continue;
            };
            let Some(t) = self.tickets.get(&root) else {
                continue;
            };
            if t.issued || !t.queued {
                continue;
            }
            chosen = Some((
                root,
                self.prefetch_queue.iter().position(|(_, _, q)| *q == root),
            ));
            break;
        }
        let (root, position) = chosen?;
        if let Some(position) = position {
            self.prefetch_queue.remove(position);
        }
        let t = self.tickets.get_mut(&root)?;
        t.queued = false;
        t.issued = true;
        self.work_order_for(root)
    }

    /// Follow `BlockedOnRead` links to the ticket that actually owns the work.
    ///
    /// Depth-bounded for the same reason [`ResidencyAuthority::promote_chain`]
    /// is: a cycle must cost a `None`, never a stack.
    fn chain_root(&self, ticket: TicketId) -> Option<TicketId> {
        let mut current = ticket;
        for _ in 0..16 {
            match self.tickets.get(&current)?.stage {
                Stage::BlockedOnRead(next) => current = next,
                _ => return Some(current),
            }
        }
        None
    }

    /// The host bytes a read must fill. Valid only for a ticket whose current
    /// stage is a read.
    pub fn read_destination(&mut self, ticket: TicketId) -> Result<&mut [u8]> {
        let t = self
            .tickets
            .get(&ticket)
            .ok_or_else(|| invalid("ticket", "no such ticket"))?;
        if t.stage != Stage::Read {
            return Err(invalid("ticket", "this ticket is not reading"));
        }
        let p = &self.placements[&t.source];
        let (offset, bytes) = (p.offset, p.bytes);
        Ok(Self::slice_mut(&mut self.host, offset, bytes))
    }

    /// The host bytes an upload must copy from. Valid only for a ticket whose
    /// current stage is an upload; the source is pinned for its lifetime.
    pub fn upload_source(&self, ticket: TicketId) -> Result<&[u8]> {
        let t = self
            .tickets
            .get(&ticket)
            .ok_or_else(|| invalid("ticket", "no such ticket"))?;
        if t.stage != Stage::Upload {
            return Err(invalid("ticket", "this ticket is not uploading"));
        }
        let p = &self.placements[&t.source];
        Ok(Self::slice(&self.host, p.offset, p.bytes))
    }

    /// The resident host bytes a lease grants. Host scope only: device bytes
    /// have no address this crate is allowed to hold.
    pub fn chunk_bytes(&self, lease: &ResidencyLease) -> Result<&[u8]> {
        self.check_lease(lease)?;
        // A lease can outlive its placement: a failed read gives every byte
        // back, and each of its coalesced waiters is still holding a lease.
        // They must all see the same thing -- no bytes -- rather than the
        // authority indexing a map entry that is gone.
        let p = self.placements.get(&lease.placement).ok_or_else(|| {
            invalid(
                "lease",
                "this chunk is no longer resident; its transfer failed or it was evicted",
            )
        })?;
        if p.scope != Scope::Host {
            return Err(invalid(
                "lease",
                "device bytes are named by range, not by slice",
            ));
        }
        // `Retiring` still serves the leases it already has: retirement stops
        // new acquires, and frees the bytes when the last consumer is done.
        if !matches!(p.state, ChunkState::HostReady | ChunkState::Retiring) {
            return Err(invalid(
                "lease",
                format!("chunk {} is {}, not readable", p.chunk, p.state.name()),
            ));
        }
        Ok(Self::slice(&self.host, p.offset, p.bytes))
    }

    /// A device lease's range inside its scope's cache arena. The executor adds
    /// the one real allocation's base; this crate holds no pointer.
    pub fn device_range(&self, lease: &ResidencyLease) -> Result<(u64, u64)> {
        self.check_lease(lease)?;
        let p = self.placements.get(&lease.placement).ok_or_else(|| {
            invalid(
                "lease",
                "this chunk is no longer resident; its transfer failed or it was evicted",
            )
        })?;
        if p.scope == Scope::Host {
            return Err(invalid("lease", "this lease is a host residency"));
        }
        if !matches!(p.state, ChunkState::DeviceReady | ChunkState::Retiring) {
            return Err(invalid(
                "lease",
                format!("chunk {} is {}, not readable", p.chunk, p.state.name()),
            ));
        }
        Ok((p.offset, p.bytes))
    }

    fn check_lease(&self, lease: &ResidencyLease) -> Result<()> {
        if lease.authority != self.id {
            return Err(invalid("lease", "lease belongs to another authority"));
        }
        if self.lease_record(lease.id).is_none() {
            return Err(invalid("lease", "lease is not live"));
        }
        Ok(())
    }

    fn slice(host: &HostBuffer, offset: u64, bytes: u64) -> &[u8] {
        let start = offset as usize;
        let end = start + bytes as usize;
        &host.bytes()[start..end]
    }

    fn slice_mut(host: &mut HostBuffer, offset: u64, bytes: u64) -> &mut [u8] {
        let start = offset as usize;
        let end = start + bytes as usize;
        &mut host.bytes_mut()[start..end]
    }

    /// Release the pin this ticket holds on its host source, exactly once.
    ///
    /// The invariant every terminal path below depends on: a ticket whose
    /// `source` differs from its `placement` holds **one** pin on that source
    /// from the moment it is resolved until the moment it settles. Releasing it
    /// twice would unpin bytes another consumer is reading; never releasing it
    /// would make a host chunk permanently unevictable, which is the quiet leak
    /// R08 is about.
    ///
    /// A ticket that *created* the source also drops it when the read never
    /// produced usable bytes. A ticket that merely joined someone else's read
    /// never does: that placement is another ticket's in-flight destination.
    fn settle_source(&mut self, source: PlacementKey, placement: PlacementKey, owns: bool) {
        if source == placement {
            return;
        }
        let drop_it = {
            let Some(p) = self.placements.get_mut(&source) else {
                return;
            };
            p.leases = p.leases.saturating_sub(1);
            // Two reasons to free it now, and the second was missing. The
            // source is dead because this ticket owned a read that never
            // delivered -- or it was **retired** while this pin was the last
            // thing holding it, in which case releasing the pin is what
            // finishes the retirement. `unpin` already does that for a
            // consumer's lease; an internal pin is no different, and leaving it
            // out stranded a `Retiring` placement with zero leases: charged,
            // unevictable and holding a one-chunk cache shut forever.
            p.leases == 0
                && (p.state == ChunkState::Retiring || (owns && p.state != ChunkState::HostReady))
        };
        if drop_it {
            self.drop_placement(source);
        }
    }

    /// Take a ticket out of flight, correcting the demand counter.
    fn take_ticket(&mut self, ticket: TicketId) -> Option<Ticket> {
        let t = self.tickets.remove(&ticket)?;
        // The counter follows **membership**, not issuance. A demand ticket
        // that is blocked on someone else's read has no order of its own yet
        // and still holds a demand slot; counting it on one side of its life
        // and not the other is how a settled queue stays blocked forever.
        if !t.queued && t.urgency == Urgency::Demand {
            self.outstanding_demand = self.outstanding_demand.saturating_sub(1);
        }
        if t.queued {
            self.prefetch_queue.retain(|(_, _, id)| *id != ticket);
        }
        if let Some(p) = self.placements.get_mut(&t.placement) {
            p.ticket = None;
        }
        Some(t)
    }

    /// Report how a read ended. Returns the work orders this outcome released:
    /// the ticket's own upload, and any ticket that was waiting for these host
    /// bytes.
    pub fn complete_read(&mut self, ticket: TicketId, outcome: Outcome) -> Result<Vec<WorkOrder>> {
        self.check_open("complete_read")?;
        let t = self
            .tickets
            .get(&ticket)
            .ok_or_else(|| invalid("ticket", "no such ticket"))?;
        if t.stage != Stage::Read {
            return Err(invalid("ticket", "this ticket is not reading"));
        }
        let (source, placement, cancelled, owns, bytes) = (
            t.source,
            t.placement,
            t.cancelled,
            t.owns_source,
            self.placements[&t.source].bytes,
        );

        match outcome {
            Outcome::Completed => {
                self.stats.bytes_read = self.stats.bytes_read.saturating_add(bytes);
                self.placements
                    .get_mut(&source)
                    .expect("source placement")
                    .state = ChunkState::HostReady;

                let mut released = Vec::new();
                if source == placement {
                    // A host acquire: this ticket is done.
                    let t = self.take_ticket(ticket).expect("live ticket");
                    if cancelled {
                        // The intent was retired while the read ran. The bytes
                        // were charged all along -- R08 -- and only now may
                        // they be given back, and only if nobody else is
                        // holding them: a device acquire waiting on this read
                        // still has its source pin.
                        self.drop_if_unheld(t.placement);
                    }
                } else {
                    // A device acquire: the second stage is now legal.
                    if cancelled {
                        let t = self.take_ticket(ticket).expect("live ticket");
                        // The device placement never left `Reading` and no
                        // upload ever ran, so it holds nothing worth keeping --
                        // and it must not stay in-flight without a ticket.
                        self.discard(t.placement);
                        self.settle_source(source, placement, owns);
                    } else {
                        // The read's demand slot carries straight into the
                        // upload's: one ticket, one outstanding transfer, so
                        // the counter is not touched here.
                        let t = self.tickets.get_mut(&ticket).expect("live ticket");
                        t.stage = Stage::Upload;
                        t.issued = true;
                        self.placements
                            .get_mut(&placement)
                            .expect("device placement")
                            .state = ChunkState::Uploading;
                        if let Some(order) = self.work_order_for(ticket) {
                            released.push(order);
                        }
                    }
                }
                released.extend(self.release_blocked_on(ticket, source));
                Ok(released)
            }
            Outcome::Failed(_) => {
                // Everything this read charged is given back, and every waiter
                // sees the same failure. A retry is a fresh acquire: an
                // invisible internal retry is how a storage fault becomes a
                // latency mystery.
                self.stats.read_failures = self.stats.read_failures.saturating_add(1);
                self.fail_blocked_on(ticket);
                let t = self.take_ticket(ticket).expect("live ticket");
                self.discard(t.placement);
                if source != placement {
                    // Release the pin, then discard the source **whoever else
                    // still holds it**. A host acquire can join a device-owned
                    // read and take a lease on that placement; leaving it
                    // because that lease was live left a `Reading` placement
                    // whose ticket had already gone, and the next acquire of
                    // the chunk panicked reaching for it. The bytes never
                    // arrived, so a surviving lease has nothing to read: it
                    // resolves to a typed error, exactly as every other failed
                    // read's waiters do.
                    self.settle_source(source, placement, owns);
                    self.discard(source);
                }
                Ok(Vec::new())
            }
            Outcome::SubmissionUnknown(_) => {
                // The read may still be writing into these bytes. Both ends are
                // withheld, and the source keeps this ticket's pin until
                // `settle_quarantined` releases it.
                self.stats.read_failures = self.stats.read_failures.saturating_add(1);
                self.stats.quarantined = self.stats.quarantined.saturating_add(1);
                self.fail_blocked_on(ticket);
                self.take_ticket(ticket);
                self.quarantine(source);
                if source != placement {
                    self.quarantine(placement);
                    // The device placement remembers its source, so settling it
                    // releases the pin.
                    self.placements
                        .get_mut(&placement)
                        .expect("device placement")
                        .upload_source = Some(source);
                }
                Ok(Vec::new())
            }
        }
    }

    /// Report how an upload ended.
    pub fn complete_upload(&mut self, ticket: TicketId, outcome: Outcome) -> Result<()> {
        self.check_open("complete_upload")?;
        let t = self
            .tickets
            .get(&ticket)
            .ok_or_else(|| invalid("ticket", "no such ticket"))?;
        if t.stage != Stage::Upload {
            return Err(invalid("ticket", "this ticket is not uploading"));
        }
        let (source, placement, cancelled, owns, bytes) = (
            t.source,
            t.placement,
            t.cancelled,
            t.owns_source,
            self.placements[&t.placement].bytes,
        );

        match outcome {
            Outcome::Completed => {
                self.stats.bytes_uploaded = self.stats.bytes_uploaded.saturating_add(bytes);
                self.take_ticket(ticket);
                {
                    let p = self
                        .placements
                        .get_mut(&placement)
                        .expect("device placement");
                    p.state = ChunkState::DeviceReady;
                    // The copy is done, so the source stops being a source. The
                    // host copy becomes an ordinary evictable resident: a cache,
                    // not a mirror that doubles every device byte's cost.
                    p.upload_source = None;
                }
                self.settle_source(source, placement, owns);
                if cancelled {
                    self.drop_if_unheld(placement);
                }
                Ok(())
            }
            Outcome::Failed(_) => {
                // Observed failure: nothing is in flight against these bytes,
                // so the device range goes back and the host bytes stay. A
                // retry uploads again without re-reading.
                self.stats.upload_failures = self.stats.upload_failures.saturating_add(1);
                self.take_ticket(ticket);
                self.discard(placement);
                self.settle_source(source, placement, owns);
                Ok(())
            }
            Outcome::SubmissionUnknown(_) => {
                // R07: the copy may still be running. Both ends are withheld,
                // and the pin stays until `settle_quarantined` releases it.
                self.stats.upload_failures = self.stats.upload_failures.saturating_add(1);
                self.stats.quarantined = self.stats.quarantined.saturating_add(1);
                self.take_ticket(ticket);
                self.quarantine(placement);
                self.quarantine(source);
                Ok(())
            }
        }
    }

    /// Tickets that were waiting for `ticket`'s host read may now upload.
    fn release_blocked_on(&mut self, ticket: TicketId, source: PlacementKey) -> Vec<WorkOrder> {
        let waiting: Vec<TicketId> = self
            .tickets
            .iter()
            .filter(|(_, t)| t.stage == Stage::BlockedOnRead(ticket))
            .map(|(id, _)| *id)
            .collect();
        let mut orders = Vec::new();
        for id in waiting {
            let (cancelled, placement, owns) = {
                let t = self.tickets.get_mut(&id).expect("waiting ticket");
                t.stage = Stage::Upload;
                t.source = source;
                (t.cancelled, t.placement, t.owns_source)
            };
            self.placements
                .get_mut(&placement)
                .expect("device placement")
                .state = ChunkState::Uploading;
            if cancelled {
                self.take_ticket(id);
                self.discard(placement);
                self.settle_source(source, placement, owns);
            } else if self.tickets[&id].queued {
                // Still a prediction. Its read was carried by somebody else's
                // demand, but its *upload* is new work, and handing it back
                // here would both jump the demand queue and hand out an order
                // the queue still owns -- which is how the same upload was
                // returned twice. It waits its turn in `next_prefetch`.
                self.tickets.get_mut(&id).expect("waiting ticket").issued = false;
            } else {
                self.tickets.get_mut(&id).expect("waiting ticket").issued = true;
                if let Some(order) = self.work_order_for(id) {
                    orders.push(order);
                }
            }
        }
        orders
    }

    /// Tickets that were waiting for a read that failed fail with it.
    fn fail_blocked_on(&mut self, ticket: TicketId) {
        let waiting: Vec<TicketId> = self
            .tickets
            .iter()
            .filter(|(_, t)| t.stage == Stage::BlockedOnRead(ticket))
            .map(|(id, _)| *id)
            .collect();
        for id in waiting {
            let t = self.take_ticket(id).expect("waiting ticket");
            self.discard(t.placement);
            // A blocked ticket never owns the source it was waiting for, but it
            // does hold a pin on it. Leaving that pin behind would make the host
            // chunk permanently unevictable.
            self.settle_source(t.source, t.placement, t.owns_source);
        }
    }
}

// ---------------------------------------------------------------------------
// Eviction, release and cancellation
// ---------------------------------------------------------------------------

impl ResidencyAuthority {
    /// Remove a settled, unleased placement.
    fn evict(&mut self, key: PlacementKey) {
        let (class, bytes, ever_demanded) = {
            let p = &self.placements[&key];
            (p.class, p.bytes, p.ever_demanded)
        };
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        self.stats.evicted_bytes = self.stats.evicted_bytes.saturating_add(bytes);
        if class.urgency == Urgency::Prefetch && !ever_demanded {
            self.stats.prefetch_wasted_bytes =
                self.stats.prefetch_wasted_bytes.saturating_add(bytes);
        } else {
            self.stats.demand_evicted_bytes = self.stats.demand_evicted_bytes.saturating_add(bytes);
        }
        self.drop_placement(key);
    }

    /// Withhold a placement whose transfer state is unknown. Charged, indexed
    /// and unusable until the outcome is observed -- document 02's rule that
    /// buffer reuse waits for all dependent streams, "including cancelled work".
    fn quarantine(&mut self, key: PlacementKey) {
        if let Some(p) = self.placements.get_mut(&key) {
            p.state = ChunkState::Quarantined;
            p.ticket = None;
        }
    }

    /// Retire a resident chunk: stop serving it, and free it when its last
    /// consumer is done.
    ///
    /// This is the `release` half of document 03's pair -- "`release` retires
    /// only after all consumers complete" -- and the only way `Retiring` is
    /// entered. Eviction never uses it, because eviction only ever chooses
    /// unleased placements; what needs it is everything that invalidates bytes
    /// somebody is still reading: a superseded format version, an artifact
    /// whose identity changed underneath the cache, or a plan transition at the
    /// explicit barrier document 03 requires ("transition prefill/decode plans
    /// at explicit barriers; avoid retaining redundant centralized and sharded
    /// copies by accident").
    ///
    /// Live leases keep reading. New acquires are refused. The bytes come back
    /// when the last lease does, and not before -- freeing them at the moment
    /// the decision was made is the use-after-free document 02 forbids.
    pub fn retire(&mut self, scope: Scope, chunk: &ChunkId) -> Result<ChunkState> {
        self.check_open("retire")?;
        let key = *self
            .index
            .get(&scope)
            .and_then(|m| m.get(chunk))
            .ok_or_else(|| invalid("chunk", "no such placement"))?;
        let p = &self.placements[&key];
        if !p.state.is_ready() {
            return Err(invalid(
                "chunk",
                format!(
                    "chunk {} is {}; only a settled placement can be retired",
                    p.chunk,
                    p.state.name()
                ),
            ));
        }
        if p.leases == 0 {
            self.drop_placement(key);
            return Ok(ChunkState::Retiring);
        }
        self.placements
            .get_mut(&key)
            .expect("indexed placement")
            .state = ChunkState::Retiring;
        Ok(ChunkState::Retiring)
    }

    /// Retire every settled, unleased placement in one scope.
    ///
    /// What a scope's cache is drained with before its physical backing goes
    /// away, and what document 03's "transition prefill/decode plans at explicit
    /// barriers" needs: the barrier is explicit, so the drain is too. Placements
    /// that are leased or in flight are **left alone** and reported, because
    /// taking bytes from a live consumer is not a transition, it is a fault.
    pub fn retire_all(&mut self, scope: Scope) -> usize {
        let drainable: Vec<PlacementKey> = self
            .placements
            .iter()
            .filter(|(_, p)| p.scope == scope && p.is_evictable())
            .map(|(k, _)| *k)
            .collect();
        for key in &drainable {
            self.drop_placement(*key);
        }
        self.placements
            .values()
            .filter(|p| p.scope == scope)
            .count()
    }

    /// Give a quarantined placement back, once its transfer is known to be
    /// over. The only way out of quarantine, and it is explicit on purpose.
    pub fn settle_quarantined(&mut self, scope: Scope, chunk: &ChunkId) -> Result<()> {
        self.check_open("settle_quarantined")?;
        let key = *self
            .index
            .get(&scope)
            .and_then(|m| m.get(chunk))
            .ok_or_else(|| invalid("chunk", "no such placement"))?;
        let p = &self.placements[&key];
        if p.state != ChunkState::Quarantined {
            return Err(invalid(
                "chunk",
                format!("chunk {} is {}, not quarantined", p.chunk, p.state.name()),
            ));
        }
        let source = p.upload_source;
        if p.leases > 0 {
            return Err(invalid(
                "chunk",
                "a quarantined placement is settled after its leases are released",
            ));
        }
        self.drop_placement(key);
        // A withheld upload still pins the host bytes it was copying from.
        // That pin is the authority's, not a consumer's, and it is what kept
        // the source from being reused while the copy might still have been
        // running. It is released here, when the copy is known to be over --
        // which is why a quarantined pair is settled device end first.
        if let Some(source) = source {
            self.release_source_pin(source, false);
        }
        Ok(())
    }

    /// Drop a placement **only if nothing holds it**.
    ///
    /// Finding 3 of the independent review, in one sentence: a cancellation
    /// retires the *intent*, and it may not take bytes away from a consumer
    /// that still wants them. The reproduction was a host read cancelled by its
    /// own consumer while a device acquire was waiting on it -- the placement
    /// was destroyed, and building the follow-on upload then panicked on a map
    /// entry that was gone.
    ///
    /// A cancelled read that *completed* produced valid bytes. If someone still
    /// holds them, they stay; only the cancelling consumer leaves.
    /// Discard a placement whose transfer will never deliver valid bytes.
    ///
    /// **The invariant it restores:** a placement is never left in an in-flight
    /// state with no ticket. An in-flight state is a promise that a transfer is
    /// coming; when the ticket is gone, so is the promise, and a cancelled
    /// device read that left one in `Reading` made the next acquire of that
    /// chunk panic reaching for the ticket that was not there.
    ///
    /// It drops **even under live leases**, and that is the difference from
    /// [`ResidencyAuthority::drop_if_unheld`]. The distinction is not who holds
    /// the placement but whether the bytes are real:
    ///
    /// * A read that *completed* produced valid bytes. If somebody still wants
    ///   them they stay -- `drop_if_unheld`.
    /// * A read that failed, or was cancelled before its data arrived, produced
    ///   nothing. Keeping it would charge for bytes that will never exist and,
    ///   worse, `Retiring` would let its holders *read* the uninitialised range.
    ///   Its leases resolve to a typed error instead, which is what every
    ///   failure test already asserts.
    fn discard(&mut self, key: PlacementKey) {
        self.drop_placement(key);
    }

    fn drop_if_unheld(&mut self, key: PlacementKey) -> bool {
        if self.placements.get(&key).is_none_or(|p| p.leases > 0) {
            return false;
        }
        self.drop_placement(key);
        true
    }

    /// Release the bytes and the index entry. Bookkeeping only: nothing here
    /// decides that it *should* happen.
    fn drop_placement(&mut self, key: PlacementKey) {
        let Some(mut p) = self.placements.remove(&key) else {
            return;
        };
        if p.class.content == Content::ConditionalTable {
            self.conditional_resident = self.conditional_resident.saturating_sub(p.bytes);
        }
        if let Some(cache) = self.caches.get_mut(&p.scope) {
            cache.committed = cache.committed.saturating_sub(p.bytes);
            if let Some(allocation) = p.allocation.take() {
                cache
                    .arena
                    .release(allocation)
                    .expect("a placement's allocation belongs to its own arena");
            }
        }
        if let Some(m) = self.index.get_mut(&p.scope) {
            m.remove(&p.chunk);
        }
    }

    /// Give a lease back.
    ///
    /// Releasing the **last** waiter on a pending ticket cancels its work:
    /// nobody is waiting for those bytes any more, and finishing the transfer
    /// would spend the link on a result with no consumer. Releasing one of
    /// several does not.
    // The refusal carries the breakdown document 03 requires a rejection to
    // show. Boxing it to save a move on the error path would put the caller's
    // only explanation behind an allocation.
    #[allow(clippy::result_large_err)]
    pub fn release(
        &mut self,
        lease: ResidencyLease,
    ) -> std::result::Result<(), ReleaseLeaseRefused> {
        if lease.authority != self.id {
            return Err(ReleaseLeaseRefused {
                lease,
                error: invalid("lease", "lease belongs to another authority"),
            });
        }
        let Some(record) = self.free_lease(lease.id) else {
            return Err(ReleaseLeaseRefused {
                lease,
                error: invalid("lease", "lease is not live"),
            });
        };
        self.unpin(record.placement, record.ticket);
        Ok(())
    }

    fn unpin(&mut self, placement: PlacementKey, ticket: Option<TicketId>) {
        let mut retiring_done = false;
        if let Some(p) = self.placements.get_mut(&placement) {
            p.leases = p.leases.saturating_sub(1);
            retiring_done = p.leases == 0 && p.state == ChunkState::Retiring;
        }
        if retiring_done {
            self.drop_placement(placement);
            return;
        }
        let Some(ticket) = ticket else { return };
        let Some(t) = self.tickets.get_mut(&ticket) else {
            return;
        };
        t.waiters = t.waiters.saturating_sub(1);
        if t.waiters == 0 {
            t.cancelled = true;
        }
    }

    /// Retire the intent behind a ticket.
    ///
    /// The bytes are **not** freed here. Document 02 forbids `Drop` alone from
    /// freeing memory a transfer may still touch, and document 03's lifecycle
    /// says states have explicit cancellation transitions, not implicit ones.
    /// So this marks, and the eventual completion settles. Idempotent: a second
    /// cancellation of the same ticket is a no-op, not an error and not a panic.
    pub fn cancel(&mut self, ticket: TicketId) -> Result<()> {
        match self.tickets.get_mut(&ticket) {
            Some(t) => {
                t.cancelled = true;
                Ok(())
            }
            // Already settled. Repeated cancellation is a normal shape of a
            // cancelled generation, not a caller error.
            None => Ok(()),
        }
    }

    pub fn is_cancelled(&self, ticket: TicketId) -> Option<bool> {
        self.tickets.get(&ticket).map(|t| t.cancelled)
    }

    /// Fail every pending ticket whose deadline has passed.
    ///
    /// A deadline is met by refusing, never by serving bytes that are not
    /// there: the caller learns its transfer will not arrive in time and can
    /// replan, which is what document 03 means by a pressure-driven replan.
    pub fn expire(&mut self, now: u64) -> Vec<TicketId> {
        let expired: Vec<TicketId> = self
            .tickets
            .iter()
            .filter(|(_, t)| t.deadline < now)
            .map(|(id, _)| *id)
            .collect();
        for id in &expired {
            // Anything that was waiting on this ticket expires with it: a
            // dependent whose dependency will never arrive is not pending, it
            // has failed, and leaving it in the table would strand its bytes.
            self.fail_blocked_on(*id);
            let Some(t) = self.take_ticket(*id) else {
                continue;
            };
            self.stats.expired = self.stats.expired.saturating_add(1);
            if t.issued {
                // A transfer was handed out and may still be running, so both
                // ends are withheld rather than reused, and the source keeps
                // this ticket's pin until `settle_quarantined` releases it.
                self.quarantine(t.placement);
                if t.source != t.placement {
                    self.quarantine(t.source);
                    self.placements
                        .get_mut(&t.placement)
                        .expect("placement")
                        .upload_source = Some(t.source);
                }
                self.stats.quarantined = self.stats.quarantined.saturating_add(1);
            } else {
                // Nothing was ever handed out, so nothing can be touching these
                // bytes, and this ticket's own read will never happen.
                self.discard(t.placement);
                if t.source != t.placement {
                    self.settle_source(t.source, t.placement, t.owns_source);
                    if t.owns_source {
                        // This ticket owned the read, so the source is dead
                        // whoever else joined it -- the same rule a failed read
                        // follows. Leaving it because a joiner's lease was live
                        // left a `Reading` placement with a stale ticket, which
                        // the transition sweep found in a combination no review
                        // round had reached: a device prefetch, a host prefetch
                        // joining its read, and a deadline passing.
                        self.discard(t.source);
                    }
                }
            }
        }
        expired
    }

    /// End a turn: release every lease it took.
    ///
    /// **R08, as a mechanism.** The legacy leak was a lease released on "next
    /// token", which leaked whenever a turn ended and no next token came. A
    /// turn that ends releases its leases whether or not it produced anything,
    /// and a transfer still in flight keeps its bytes charged until it settles
    /// rather than having them freed underneath it.
    pub fn end_turn(&mut self, turn: TurnId) -> TurnCleanup {
        let mine: Vec<ResidencyLeaseId> = self
            .lease_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| {
                s.held.filter(|h| h.turn == turn).map(|_| ResidencyLeaseId {
                    slot: slot as u32,
                    generation: s.generation,
                })
            })
            .collect();
        let mut cleanup = TurnCleanup::default();
        for id in mine {
            let Some(LeaseRecord {
                placement, ticket, ..
            }) = self.free_lease(id)
            else {
                continue;
            };
            let chunk = self.placements.get(&placement).map(|p| (*p.chunk).clone());
            self.unpin(placement, ticket);
            cleanup.released_leases.push(id);
            if let Some(chunk) = chunk
                && self
                    .placements
                    .get(&placement)
                    .is_some_and(|p| p.leases == 0 && p.state.is_ready())
            {
                cleanup.unpinned_chunks.push(chunk);
            }
            if let Some(ticket) = ticket
                && self.tickets.contains_key(&ticket)
                && !cleanup.still_in_flight.contains(&ticket)
            {
                cleanup.still_in_flight.push(ticket);
            }
        }
        cleanup
    }
}

/// Withhold the host storage when anything may still be reading it.
///
/// `close` refuses over in-flight work, but a caller can drop the authority
/// instead, and dropping a `Vec` returns its pages to the allocator. A
/// host-to-device copy reads those pages by address, so document 02's rule
/// applies exactly: "Retirement is event-driven; Rust `Drop` alone must not
/// free in-flight CUDA memory."
///
/// The bytes are therefore abandoned rather than freed, and the ledger keeps
/// showing the charge. That is deliberate: an unrecoverable, *visible* leak is
/// the safe side of this trade, and it is the same one `moxie-executor` makes
/// for a lost context. A caller that wants its memory back closes the authority,
/// which tells it exactly what is still in flight.
impl Drop for ResidencyAuthority {
    fn drop(&mut self) {
        if self.placements.values().any(|p| p.state.is_in_flight()) {
            self.host.withhold();
        }
    }
}
