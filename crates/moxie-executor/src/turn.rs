//! Turn-boundary sweeps: retire everything completable, return the rest.
//!
//! R08's leak was a lease released on "next token" when the turn ended and no
//! next token came. A [`Turn`] holds the turn's leases and [`Turn::release_turn`]
//! retires each one: completed leases release, and anything else comes back in
//! the report with its lease attached — naming the hold is not enough, because
//! a named-but-dropped lease can never release once its event completes. The
//! next turn holds it again, with no next token required.

use moxie_memory::Ledger;

use crate::lease::{Completion, Lease, LeaseId, ManualCompletion, RetireRefused};

/// A lease the turn sweep could not retire, with its reason and its handle.
///
/// The lease comes back so a later sweep — after completion is observed, or
/// after cancellation — can still release it. A sweep that kept only the
/// description would strand the reservation the moment the event completes.
#[derive(Debug)]
pub struct HeldLease<C = ManualCompletion, R = ()> {
    /// Which lease is still charged.
    pub id: LeaseId,
    /// The label given at acquisition.
    pub label: String,
    /// Why it stays held: completion unobserved, or the context lost.
    pub reason: String,
    /// The lease itself, still holding its reservation and resources.
    pub lease: Lease<C, R>,
}

/// A lease the turn sweep retired, with its released resource.
#[derive(Debug)]
pub struct RetiredLease<R = ()> {
    /// Which lease retired.
    pub id: LeaseId,
    /// The resource retirement returned: safe to reuse from here.
    pub resource: R,
}
/// What a turn sweep did.
#[derive(Debug)]
pub struct TurnReport<C = ManualCompletion, R = ()> {
    /// Leases retired into the ledger, in sweep order, resources returned.
    pub retired: Vec<RetiredLease<R>>,
    /// Leases still charged, each named with its reason and its handle.
    pub held: Vec<HeldLease<C, R>>,
}
/// The leases acquired during one turn.
///
/// Dropping a turn retires nothing: each lease drops with it and stays charged
/// and visible in the ledger, exactly like a dropped [`Reservation`]. The
/// release path is [`Turn::release_turn`], which returns whatever it could not
/// retire instead of stranding it.
///
/// [`Reservation`]: moxie_memory::Reservation
#[derive(Debug, Default)]
pub struct Turn<C = ManualCompletion, R = ()> {
    label: String,
    leases: Vec<Lease<C, R>>,
}

impl<C, R> TurnReport<C, R> {
    /// Whether every lease the turn held was retired.
    pub fn is_clean(&self) -> bool {
        self.held.is_empty()
    }
}

impl<C: Completion, R> Turn<C, R> {
    /// Hold this turn's leases under `label`. The label must not be empty: an
    /// unnamed report names nothing.
    pub fn new(label: impl Into<String>) -> moxie_types::Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(moxie_types::Error::InvalidRequest {
                field: "label",
                detail: "a turn is labelled, so its sweep report can name it".into(),
            });
        }
        Ok(Turn {
            label,
            leases: Vec::new(),
        })
    }

    /// The label given at creation.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// How many leases this turn holds.
    pub fn len(&self) -> usize {
        self.leases.len()
    }

    /// Whether this turn holds no lease.
    pub fn is_empty(&self) -> bool {
        self.leases.is_empty()
    }

    /// Move a lease into this turn's sweep set.
    pub fn hold(&mut self, lease: Lease<C, R>) {
        self.leases.push(lease);
    }

    /// Block until every held lease's completion is observable. The explicit
    /// wait before [`Turn::release_turn`]: the sweep itself never blocks, so a
    /// missing wait reports held leases rather than stalling inside release.
    pub fn synchronize(&mut self) -> moxie_types::Result<()> {
        for lease in &mut self.leases {
            lease.synchronize()?;
        }
        Ok(())
    }

    /// Retire every held lease into `ledger`. Completed leases release;
    /// anything still in flight, cancelled-but-incomplete, or lost comes back
    /// in the report by identity, with its reason and its handle, and stays
    /// charged. Consumes the turn; hold the returned leases in the next one.
    pub fn release_turn(self, ledger: &mut Ledger) -> TurnReport<C, R> {
        let mut report = TurnReport {
            retired: Vec::new(),
            held: Vec::new(),
        };
        for lease in self.leases {
            let id = lease.id();
            let label = lease.label().to_string();
            match lease.retire(ledger) {
                Ok((id, resource)) => report.retired.push(RetiredLease { id, resource }),
                Err(RetireRefused { lease, error }) => report.held.push(HeldLease {
                    id,
                    label,
                    reason: error.to_string(),
                    lease,
                }),
            }
        }
        report
    }
}
