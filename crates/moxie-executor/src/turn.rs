//! Turn-boundary sweeps: retire everything completable, name the rest.
//!
//! R08's leak was a lease released on "next token" when the turn ended and no
//! next token came. A [`Turn`] holds the turn's leases and [`Turn::release_turn`]
//! retires each one: completed leases release, anything else is reported by
//! identity and stays charged in the ledger — visible, never silently dropped.

use moxie_memory::Ledger;

use crate::lease::{Completion, Lease, LeaseId, ManualCompletion, RetireRefused};

/// A lease the turn sweep could not retire, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldLease {
    /// Which lease is still charged.
    pub id: LeaseId,
    /// The label given at acquisition.
    pub label: String,
    /// Why it stays held: completion unobserved, or the context lost.
    pub reason: String,
}

/// What a turn sweep did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TurnReport {
    /// Leases retired into the ledger, in sweep order.
    pub retired: Vec<LeaseId>,
    /// Leases still charged, each named with its reason.
    pub held: Vec<HeldLease>,
}

impl TurnReport {
    /// Whether every lease the turn held was retired.
    pub fn is_clean(&self) -> bool {
        self.held.is_empty()
    }
}

/// The leases acquired during one turn.
///
/// Dropping a turn retires nothing: each lease drops with it and stays charged
/// and visible in the ledger, exactly like a dropped [`Reservation`]. The only
/// way out is [`Turn::release_turn`], and its report names whatever remains.
///
/// [`Reservation`]: moxie_memory::Reservation
#[derive(Debug, Default)]
pub struct Turn<C = ManualCompletion> {
    label: String,
    leases: Vec<Lease<C>>,
}

impl<C: Completion> Turn<C> {
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
    pub fn hold(&mut self, lease: Lease<C>) {
        self.leases.push(lease);
    }
    /// Block until every held lease's completion is observable. The explicit
    /// wait before [`Turn::release_turn`]: the sweep itself never blocks, so a
    /// missing wait reports held leases rather than stalling inside release.
    pub fn synchronize(&self) -> moxie_types::Result<()> {
        for lease in &self.leases {
            lease.synchronize()?;
        }
        Ok(())
    }

    /// Retire every held lease into `ledger`. Completed leases release;
    /// anything still in flight, cancelled-but-incomplete, or lost is reported
    /// by identity with its reason and stays charged. Consumes the turn.
    pub fn release_turn(self, ledger: &mut Ledger) -> TurnReport {
        let mut report = TurnReport::default();
        for lease in self.leases {
            let id = lease.id();
            let label = lease.label().to_string();
            match lease.retire(ledger) {
                Ok(retired) => report.retired.push(retired),
                Err(RetireRefused { lease, error }) => {
                    // Intentionally dropped: the reservation stays charged and
                    // visible in the ledger, and the report names it here.
                    drop(lease);
                    report.held.push(HeldLease {
                        id,
                        label,
                        reason: error.to_string(),
                    });
                }
            }
        }
        report
    }
}
