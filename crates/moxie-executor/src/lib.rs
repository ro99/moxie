//! Rank-local execution leases: byte accounting bound to observed completion.
//!
//! Document 02 gives this crate rank-local execution, events and transfers. It
//! binds the two halves task 0006–0008 built: the ledger's [`Reservation`] and
//! the rank context's ordered stream work. A lease is acquired against admitted
//! bytes, tracks one completion source, and retires only after observed completion
//! and checked cleanup. The upload source and destination remain owned together.
//!
//! Three properties carry over from the ledger, because the failure modes are
//! the same:
//!
//! * **Event-driven retirement.** [`Lease::retire`] queries the recorded
//!   completion and refuses while work is in flight. Dropping submitted or lost
//!   leases withholds their resources; an unsubmitted resource can drop normally.
//!   Every dropped lease's reservation stays charged and visible in
//!   [`Ledger::outstanding`], exactly like a dropped `Reservation`.
//! * **Refusal returns the handle.** A failed `retire` hands the lease back,
//!   so the error path cannot strand charged bytes the way R08's "next token"
//!   release did when no next token came.
//! * **Withholding wins.** Cancellation retires the *intent*; the bytes stay
//!   held until completion or loss is observed. A lost context withholds
//!   forever rather than advertising memory nothing can recover.
//!
//! This crate also binds one admitted reservation to a physical `DeviceArena`
//! behind the `driver` feature. `moxie-memory` owns its pure range metadata;
//! operation leases reuse the same completion lifecycle described above.
//!
//! This crate now admits and materializes a pure graph resource candidate, but
//! it still does not derive the envelope (the planner's job), choose victims,
//! manage residency, select semantic kernels, execute a graph, or know what a
//! model is.
//!
//! [`Ledger::outstanding`]: moxie_memory::Ledger::outstanding
//! [`Reservation`]: moxie_memory::Reservation

// The `numa` feature adds one audited glibc FFI block (thread affinity) and
// `driver` adds the CUDA one; every other build of this crate forbids unsafe.
#![cfg_attr(not(any(feature = "driver", feature = "numa")), forbid(unsafe_code))]

#[cfg(all(test, feature = "driver"))]
pub(crate) static DRIVER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub mod lease;
pub use lease::{
    AcquireRefused, Completion, Lease, LeaseId, LeaseState, ManualCompletion, RetireRefused,
    Script, ScriptedCompletion, SettledResource, TrackableManual, check_fit, check_upload_fit,
};
pub mod plan;
pub mod turn;
pub use plan::{resource_request, validate_plan_binding};

pub mod residency;
pub use residency::{ChunkSource, ShardSource, drain_reads, perform_read};

pub mod grouped;
pub use grouped::{
    BufferAddresses, ExpertRoles, GroupedAdmitRefused, GroupedCloseRefused, GroupedRun,
    GroupedStats, HostBuffers, OrderQueue, PlacementReport, Progress, QueueFull, QueuedGroup,
};

pub mod arena;
pub use arena::{
    OperationAcquireRefused, OperationHeld, OperationLease, OperationRetireRefused,
    OperationRetired, OperationTurn, OperationTurnReport,
};

#[cfg(feature = "driver")]
pub use plan::{HeldPlanResource, PlanAdmitRefused, PlanCloseRefused, ReservedPlan, TensorHandle};

#[cfg(feature = "driver")]
pub use arena::{
    ArenaCloseRefused, ArenaCreateRefused, ArenaUpload, DeviceArena, DeviceRange,
    PrepareArenaUploadRefused, RangeReleaseRefused, RangeTransferRefused,
};

pub use turn::{HeldLease, RetiredLease, Turn, TurnReport};

#[cfg(feature = "driver")]
mod chain;
#[cfg(feature = "driver")]
pub use chain::{
    ChainOperation, ChainResult, OwnedBinding, SelectedAdmitRefused, SelectedCloseRefused,
    SelectedCompletion, SelectedHeldResource, SelectedLaunchRefused, SelectedReservedPlan,
    selected_resource_request,
};

#[cfg(feature = "driver")]
pub use lease::{PrepareRefused, Upload};
