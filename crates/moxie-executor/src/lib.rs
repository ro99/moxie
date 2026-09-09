//! Rank-local execution leases: byte accounting bound to observed completion.
//!
//! Document 02 gives this crate rank-local execution, events and transfers. It
//! binds the two halves task 0006–0008 built: the ledger's [`Reservation`] and
//! the rank context's ordered stream work. A lease is acquired against admitted
//! bytes, tracks one completion source, and retires only when that source
//! lease-until-completion launch rule, encoded in the API rather than in caller
//! comments.
//!
//! Three properties carry over from the ledger, because the failure modes are
//! the same:
//!
//! * **Event-driven retirement.** [`Lease::retire`] queries the recorded
//!   completion and refuses while work is in flight. `Drop` alone never frees
//!   anything: a dropped lease's reservation stays charged and visible in
//!   [`Ledger::outstanding`], exactly like a dropped `Reservation`.
//! * **Refusal returns the handle.** A failed `retire` hands the lease back,
//!   so the error path cannot strand charged bytes the way R08's "next token"
//!   release did when no next token came.
//! * **Withholding wins.** Cancellation retires the *intent*; the bytes stay
//!   held until completion or loss is observed. A lost context withholds
//!   forever rather than advertising memory nothing can recover.
//!
//! What this crate deliberately does not do: admit envelopes (the ledger's
//! job), allocate or suballocate (the next task's allocator), choose victims,
//! plan execution, or know what a model is.
//!
//! [`Ledger::outstanding`]: moxie_memory::Ledger::outstanding
//! [`Reservation`]: moxie_memory::Reservation

#![forbid(unsafe_code)]

pub mod lease;
pub mod turn;

pub use lease::{Completion, Lease, LeaseId, LeaseState, ManualCompletion, RetireRefused};
pub use turn::{HeldLease, Turn, TurnReport};
