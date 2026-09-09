//! The resource ledger: one authority for admission and accounting.
//!
//! Document 03 requires the engine to be "a real memory-and-transfer authority
//! under scarcity", and says plainly that "an allocator, LRU utility, or
//! placement simulator alone is not that asset". This crate is the accounting
//! core of that authority and nothing more: it holds no memory, calls no driver,
//! reads no file and measures nothing. It is given a measured capacity snapshot
//! and a declared plan, and it answers whether the plan fits, what it would
//! cost, and -- when it does not fit -- which ceiling binds and what the caller
//! could legally ask for instead.
//!
//! Three properties are the point of the design:
//!
//! * **Peak, not sum.** A plan declares liveness spans over ordered stages, so
//!   two buffers that are never live together cost what the larger costs, and
//!   two that coexist at a prefill-to-decode barrier cost both.
//! * **Atomic.** [`Ledger::admit`] reserves the whole envelope or nothing. A
//!   refusal leaves every counter untouched, including tiers that would have fit.
//! * **Explicit release.** A [`Reservation`] is given back by name. Dropping one
//!   leaves it charged and visible in [`Ledger::outstanding`] -- R08's leak was
//!   invisible, not large.
//!
//! What this crate deliberately does not do: allocate, map, copy, evict, choose
//! a victim, hold a lease over real bytes, wait on an event, or know what a
//! model is.

#![forbid(unsafe_code)]

pub mod ledger;
pub mod report;
pub mod request;
pub mod snapshot;

pub use ledger::{
    AdmitError, Ledger, LedgerId, Outstanding, ReleaseRefused, Reservation, ReservationId,
};
pub use report::{
    AdmissionReport, BindingConstraint, BindingKind, LegalAlternative, Rejection, ScopeReport,
    TierReport,
};
pub use request::{BufferRequest, DerivedReserve, PlanRequest, ReserveRule, Scaling, StageSpan};
pub use snapshot::CapacitySnapshot;
