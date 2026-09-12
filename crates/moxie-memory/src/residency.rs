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

/// Declared control cost of one tracked placement, in bytes.
///
/// Charged up front as `max_placements * PLACEMENT_CONTROL_BYTES` so the
/// authority's own bookkeeping is admitted rather than assumed free. It covers
/// the placement record, its map node, the interned identity and the waiter
/// slot. It is an over-estimate on purpose: an under-estimated control budget
/// is an unadmitted allocation, which is the thing document 03 calls out when
/// it says mapped virtual bytes are not committed host RAM and "neither is
/// free".
pub const PLACEMENT_CONTROL_BYTES: u64 = 512;

/// Declared control cost of one lease slot, in bytes. Charged like a
/// placement's: the lease table is real memory and a bounded one is still
/// memory.
pub const LEASE_CONTROL_BYTES: u64 = 64;

/// Upper bound on a declared placement or lease count, so the control charge
/// cannot overflow or quietly become the dominant cost.
const MAX_DECLARED_PLACEMENTS: u32 = 1 << 22;

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
        if role.len() > 256 {
            return Err(invalid("role", "tensor role longer than 256 bytes"));
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
    chunk: ChunkId,
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
    index: BTreeMap<Scope, BTreeMap<ChunkId, PlacementKey>>,
    placements: BTreeMap<PlacementKey, Placement>,
    next_placement: u32,
    max_placements: u32,
    tickets: BTreeMap<TicketId, Ticket>,
    /// Queued prefetch tickets in `(deadline, sequence)` order.
    prefetch_queue: Vec<(u64, u64, TicketId)>,
    prefetch_queue_capacity: u32,
    next_sequence: u64,
    outstanding_demand: u32,
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
        let control_bytes = u64::from(request.max_placements)
            .checked_mul(PLACEMENT_CONTROL_BYTES)
            .and_then(|p| {
                u64::from(request.max_leases)
                    .checked_mul(LEASE_CONTROL_BYTES)
                    .and_then(|l| p.checked_add(l))
            })
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
            tickets: BTreeMap::new(),
            prefetch_queue: Vec::new(),
            prefetch_queue_capacity: request.prefetch_queue_capacity,
            next_sequence: 0,
            outstanding_demand: 0,
            conditional_floor_bytes: request.conditional_floor_bytes,
            conditional_resident: 0,
            lease_slots,
            free_lease_slots,
            live_leases: 0,
            stats: ResidencyStats::default(),
        })
    }

    /// Release the envelope. Refused while anything is leased or in flight:
    /// closing over live bytes is the leak, not the fix.
    pub fn close(&mut self, ledger: &mut Ledger) -> Result<()> {
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
        let keys: Vec<PlacementKey> = self.placements.keys().copied().collect();
        for key in keys {
            self.drop_placement(key);
        }
        self.host.release(ledger)?;
        if let Some(envelope) = self.device_envelope.take() {
            ledger.release(envelope).map_err(|refused| refused.error)?;
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
                chunk: p.chunk.clone(),
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
            let ticket = self.placements[&key]
                .ticket
                .expect("an in-flight placement owns a ticket");
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
            if promote {
                self.promote_ticket(ticket, request.deadline);
            }
            self.stats.misses = self.stats.misses.saturating_add(1);
            return Ok(Acquired::Pending {
                lease,
                ticket,
                work: PendingWork::Coalesced,
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

        if self.placements.len() >= self.max_placements as usize {
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
        let (stage, source_key, state) = match source {
            None => (Stage::Read, key, ChunkState::Reading),
            Some(HostSource {
                key: src,
                created: true,
                ..
            }) => (Stage::Read, src, ChunkState::Reading),
            Some(HostSource {
                key: src,
                created: false,
                pending: None,
            }) => (Stage::Upload, src, ChunkState::Uploading),
            Some(HostSource {
                key: src,
                created: false,
                pending: Some(blocking),
            }) => (Stage::BlockedOnRead(blocking), src, ChunkState::Reading),
        };

        {
            let p = self.placements.get_mut(&key).expect("just placed");
            p.ticket = Some(ticket);
            p.upload_source = source.map(|s| s.key);
            p.state = state;
        }
        if let Some(src) = source.filter(|s| s.created) {
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
            PendingWork::Coalesced
        } else {
            self.outstanding_demand = self.outstanding_demand.saturating_add(1);
            self.tickets.get_mut(&ticket).expect("just inserted").issued = true;
            PendingWork::Issued(self.work_order_for(ticket))
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
                    })
                }
                ChunkState::Reading => {
                    let blocking = p.ticket.expect("a reading placement owns a ticket");
                    p.leases = p.leases.saturating_add(1);
                    p.last_used = request.now;
                    Ok(HostSource {
                        key: existing,
                        created: false,
                        pending: Some(blocking),
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
            })
        }
    }

    fn release_source_pin(&mut self, key: PlacementKey, created: bool) {
        if let Some(p) = self.placements.get_mut(&key) {
            p.leases = p.leases.saturating_sub(1);
        }
        if created {
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
                    let named = victims.iter().map(|c| c.chunk.clone()).collect();
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

        let mut named: Vec<ChunkId> = victims.iter().map(|c| c.chunk.clone()).collect();
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
                        named.push(c.chunk.clone());
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
        self.placements.insert(
            key,
            Placement {
                chunk: request.chunk.clone(),
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
        self.index
            .entry(scope)
            .or_default()
            .insert(request.chunk.clone(), key);
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
    chunk: ChunkId,
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
#[derive(Debug, Clone, Copy)]
struct HostSource {
    key: PlacementKey,
    /// This acquire created the host placement, so it owns the read.
    created: bool,
    /// Another ticket is already reading it; this one waits for that.
    pending: Option<TicketId>,
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
            .map(|p| p.chunk.clone())
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

    fn promote_ticket(&mut self, ticket: TicketId, deadline: u64) {
        let (was_queued, stage) = {
            let t = self.tickets.get_mut(&ticket).expect("live ticket");
            t.urgency = Urgency::Demand;
            t.deadline = t.deadline.min(deadline);
            (t.queued, t.stage)
        };
        if was_queued {
            self.prefetch_queue.retain(|(_, _, id)| *id != ticket);
            self.tickets.get_mut(&ticket).expect("live ticket").queued = false;
            if !matches!(stage, Stage::BlockedOnRead(_)) {
                self.outstanding_demand = self.outstanding_demand.saturating_add(1);
            }
        }
    }

    fn work_order_for(&self, ticket: TicketId) -> WorkOrder {
        let t = &self.tickets[&ticket];
        match t.stage {
            Stage::Read => {
                let src = &self.placements[&t.source];
                WorkOrder::Read {
                    ticket,
                    chunk: src.chunk.clone(),
                    host_offset: src.offset,
                    len_bytes: src.bytes,
                }
            }
            Stage::Upload => {
                let src = &self.placements[&t.source];
                let p = &self.placements[&t.placement];
                WorkOrder::Upload {
                    ticket,
                    chunk: p.chunk.clone(),
                    scope: p.scope,
                    host_offset: src.offset,
                    device_offset: p.offset,
                    len_bytes: p.bytes,
                }
            }
            Stage::BlockedOnRead(_) => unreachable!("a blocked ticket has no work of its own"),
        }
    }

    /// Release the next queued prefetch, if demand is idle.
    ///
    /// Returns `None` while any demand transfer is outstanding. That is
    /// document 03's "demand work outranks predictions" as a mechanism rather
    /// than as a priority number nobody checks.
    pub fn next_prefetch(&mut self) -> Option<WorkOrder> {
        if self.outstanding_demand > 0 || self.prefetch_queue.is_empty() {
            return None;
        }
        let (_, _, ticket) = self.prefetch_queue.remove(0);
        let t = self.tickets.get_mut(&ticket)?;
        t.queued = false;
        t.issued = true;
        Some(self.work_order_for(ticket))
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
        if p.state != ChunkState::HostReady {
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
        if p.state != ChunkState::DeviceReady {
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

    /// Report how a read ended. Returns the work orders this outcome released:
    /// the ticket's own upload, and any ticket that was waiting for these host
    /// bytes.
    pub fn complete_read(&mut self, ticket: TicketId, outcome: Outcome) -> Result<Vec<WorkOrder>> {
        let t = self
            .tickets
            .get(&ticket)
            .ok_or_else(|| invalid("ticket", "no such ticket"))?;
        if t.stage != Stage::Read {
            return Err(invalid("ticket", "this ticket is not reading"));
        }
        let (source, placement, cancelled, bytes) = (
            t.source,
            t.placement,
            t.cancelled,
            self.placements[&t.source].bytes,
        );
        self.outstanding_demand = self
            .outstanding_demand
            .saturating_sub(u32::from(self.tickets[&ticket].urgency == Urgency::Demand));

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
                    self.retire_ticket(ticket, cancelled);
                } else {
                    // A device acquire: the second stage is now legal.
                    let t = self.tickets.get_mut(&ticket).expect("live ticket");
                    t.stage = Stage::Upload;
                    t.issued = false;
                    self.placements
                        .get_mut(&placement)
                        .expect("device placement")
                        .state = ChunkState::Uploading;
                    if cancelled {
                        self.settle_cancelled(ticket);
                    } else {
                        self.mark_issued(ticket);
                        released.push(self.work_order_for(ticket));
                    }
                }
                released.extend(self.release_blocked_on(ticket, source));
                Ok(released)
            }
            Outcome::Failed(error) => {
                // Everything this read charged is given back, and every waiter
                // sees the same failure. A retry is a fresh acquire: an
                // invisible internal retry is how a storage fault becomes a
                // latency mystery.
                self.stats.read_failures = self.stats.read_failures.saturating_add(1);
                self.fail_blocked_on(ticket, &error);
                self.tickets.remove(&ticket);
                if source != placement {
                    self.release_source_pin(source, true);
                    self.drop_placement(placement);
                } else {
                    self.drop_placement(source);
                }
                Ok(Vec::new())
            }
            Outcome::SubmissionUnknown(error) => {
                self.stats.read_failures = self.stats.read_failures.saturating_add(1);
                self.stats.quarantined = self.stats.quarantined.saturating_add(1);
                self.fail_blocked_on(ticket, &error);
                self.tickets.remove(&ticket);
                self.quarantine(source);
                if source != placement {
                    self.quarantine(placement);
                }
                Ok(Vec::new())
            }
        }
    }

    /// Report how an upload ended.
    pub fn complete_upload(&mut self, ticket: TicketId, outcome: Outcome) -> Result<()> {
        let t = self
            .tickets
            .get(&ticket)
            .ok_or_else(|| invalid("ticket", "no such ticket"))?;
        if t.stage != Stage::Upload {
            return Err(invalid("ticket", "this ticket is not uploading"));
        }
        let (source, placement, cancelled, urgency, bytes) = (
            t.source,
            t.placement,
            t.cancelled,
            t.urgency,
            self.placements[&t.placement].bytes,
        );
        self.outstanding_demand = self
            .outstanding_demand
            .saturating_sub(u32::from(urgency == Urgency::Demand));

        match outcome {
            Outcome::Completed => {
                self.stats.bytes_uploaded = self.stats.bytes_uploaded.saturating_add(bytes);
                self.placements
                    .get_mut(&placement)
                    .expect("device placement")
                    .state = ChunkState::DeviceReady;
                // The copy is done, so the source stops being a source. The
                // host copy becomes an ordinary evictable resident: a cache,
                // not a mirror that doubles every device byte's cost.
                self.placements
                    .get_mut(&placement)
                    .expect("device placement")
                    .upload_source = None;
                if let Some(p) = self.placements.get_mut(&source) {
                    p.leases = p.leases.saturating_sub(1);
                    p.ticket = None;
                }
                self.retire_ticket(ticket, cancelled);
                Ok(())
            }
            Outcome::Failed(_) => {
                // Observed failure: nothing is in flight against these bytes,
                // so the device range goes back and the host bytes stay. A
                // retry uploads again without re-reading.
                self.stats.upload_failures = self.stats.upload_failures.saturating_add(1);
                self.tickets.remove(&ticket);
                if let Some(p) = self.placements.get_mut(&source) {
                    p.leases = p.leases.saturating_sub(1);
                    p.ticket = None;
                }
                self.drop_placement(placement);
                Ok(())
            }
            Outcome::SubmissionUnknown(_) => {
                // R07: the copy may still be running. Both ends are withheld.
                self.stats.upload_failures = self.stats.upload_failures.saturating_add(1);
                self.stats.quarantined = self.stats.quarantined.saturating_add(1);
                self.tickets.remove(&ticket);
                self.quarantine(placement);
                self.quarantine(source);
                Ok(())
            }
        }
    }

    fn mark_issued(&mut self, ticket: TicketId) {
        if let Some(t) = self.tickets.get_mut(&ticket) {
            t.issued = true;
            if t.urgency == Urgency::Demand {
                self.outstanding_demand = self.outstanding_demand.saturating_add(1);
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
            let cancelled = self.tickets[&id].cancelled;
            {
                let t = self.tickets.get_mut(&id).expect("waiting ticket");
                t.stage = Stage::Upload;
                t.source = source;
                t.issued = false;
            }
            let placement = self.tickets[&id].placement;
            self.placements
                .get_mut(&placement)
                .expect("device placement")
                .state = ChunkState::Uploading;
            if cancelled {
                self.settle_cancelled(id);
            } else {
                self.mark_issued(id);
                orders.push(self.work_order_for(id));
            }
        }
        orders
    }

    fn fail_blocked_on(&mut self, ticket: TicketId, _error: &Error) {
        let waiting: Vec<TicketId> = self
            .tickets
            .iter()
            .filter(|(_, t)| t.stage == Stage::BlockedOnRead(ticket))
            .map(|(id, _)| *id)
            .collect();
        for id in waiting {
            let t = self.tickets.remove(&id).expect("waiting ticket");
            if t.urgency == Urgency::Demand {
                self.outstanding_demand = self.outstanding_demand.saturating_sub(1);
            }
            self.drop_placement(t.placement);
        }
    }

    fn retire_ticket(&mut self, ticket: TicketId, cancelled: bool) {
        let t = self.tickets.remove(&ticket).expect("live ticket");
        if let Some(p) = self.placements.get_mut(&t.placement) {
            p.ticket = None;
        }
        if cancelled {
            // The intent was retired while the transfer ran. The bytes were
            // charged all along -- R08 -- and only now may they be given back.
            self.drop_placement(t.placement);
            if t.source != t.placement {
                self.release_source_pin(t.source, false);
            }
        }
    }

    fn settle_cancelled(&mut self, ticket: TicketId) {
        let t = self.tickets.remove(&ticket).expect("live ticket");
        if let Some(p) = self.placements.get_mut(&t.placement) {
            p.ticket = None;
        }
        self.drop_placement(t.placement);
        if t.source != t.placement {
            self.release_source_pin(t.source, false);
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

    /// Give a quarantined placement back, once its transfer is known to be
    /// over. The only way out of quarantine, and it is explicit on purpose.
    pub fn settle_quarantined(&mut self, scope: Scope, chunk: &ChunkId) -> Result<()> {
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
            let t = self.tickets.remove(id).expect("listed ticket");
            self.stats.expired = self.stats.expired.saturating_add(1);
            if t.queued {
                self.prefetch_queue.retain(|(_, _, q)| q != id);
            } else if t.urgency == Urgency::Demand {
                self.outstanding_demand = self.outstanding_demand.saturating_sub(1);
            }
            if t.issued {
                // A transfer was handed out and may still be running.
                self.quarantine(t.placement);
                if t.source != t.placement {
                    self.quarantine(t.source);
                }
                self.stats.quarantined = self.stats.quarantined.saturating_add(1);
            } else {
                self.drop_placement(t.placement);
                if t.source != t.placement {
                    self.release_source_pin(t.source, true);
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
            let chunk = self.placements.get(&placement).map(|p| p.chunk.clone());
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
