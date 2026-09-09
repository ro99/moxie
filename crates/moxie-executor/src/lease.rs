//! One lease: admitted bytes bound to one observed completion.
//!
//! Acquire against a [`Reservation`], track exactly one completion source,
//! retire only when that source reports complete. The two rules that matter:
//!
//! * A failed [`Lease::retire`] hands the lease back. Consuming the lease on a
//!   refusal would strand charged bytes with no handle — the error-path version
//!   of R08.
//! * Dropping a lease releases nothing. The owned reservation drops with it and
//!   stays charged and visible in [`Ledger::outstanding`], which is the failure
//!   mode that can be found.
//!
//! [`Ledger::outstanding`]: moxie_memory::Ledger::outstanding

use std::sync::atomic::{AtomicU64, Ordering};

use moxie_memory::{Ledger, Reservation};
use moxie_types::{Error, Result};

/// Observed completion truth for one recorded operation.
///
/// A CUDA event is one implementation (behind the `driver` feature);
/// [`ManualCompletion`] is the host-testable one. Not-ready is a state the
/// caller observes through [`Lease::retire`], never an error smuggled out of
/// this trait: implementations must map their not-ready signal to `Ok(false)`,
/// as [`Event::is_complete`] already does for code 600.
///
/// [`Event::is_complete`]: moxie_cuda::Event::is_complete
pub trait Completion: core::fmt::Debug {
    /// Whether the recorded operation has completed. Errors only when the
    /// completion source itself is unusable.
    fn query_complete(&self) -> Result<bool>;
    /// Block until completion is observable. The caller's explicit wait:
    /// retirement never blocks, so reaching for this is always a visible
    /// decision at the call site. The default is a no-op for sources with no
    /// waitable handle.
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    /// A human-readable name for reports: which stream, rank or flag this is.
    fn describe(&self) -> String;
}

/// A hand-driven completion source for host tests.
///
/// The retirement machine cannot tell it from a driver event, which is what
/// makes every transition below testable with no GPU present.
#[derive(Debug, Clone, Default)]
pub struct ManualCompletion {
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ManualCompletion {
    /// An uncompleted source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe completion from now on.
    pub fn complete(&self) {
        self.done.store(true, Ordering::Release);
    }

    /// The current observed state, for test assertions.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

impl Completion for ManualCompletion {
    fn query_complete(&self) -> Result<bool> {
        Ok(self.is_done())
    }

    fn describe(&self) -> String {
        format!("manual (done={})", self.is_done())
    }
}

#[cfg(feature = "driver")]
impl Completion for moxie_cuda::Event<'_> {
    fn query_complete(&self) -> Result<bool> {
        self.is_complete()
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }

    fn describe(&self) -> String {
        "cuda event".to_string()
    }
}

/// One lease's process-unique identity, so reports can name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeaseId(u64);

impl LeaseId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        LeaseId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The numeric identity, for reports.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for LeaseId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "lease {}", self.0)
    }
}

/// Where a lease stands. Completion itself is queried, never stored: the event
/// in the driver is the truth, and a cached copy would be a second authority
/// that can disagree with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseState {
    /// Acquired; nothing enqueued yet.
    Live,
    /// An operation is recorded against the completion source.
    InFlight,
    /// Cancellation retired the intent. The bytes stay held until completion
    /// or loss is observed — R08.
    Cancelled,
    /// The context is known lost. Withheld from every future use.
    Lost,
}

/// Which device was lost and why. Boxed in [`Lease`]: it exists only on the
/// context-loss path, so it must not inflate every live lease.
#[derive(Debug, Clone)]
struct LostInfo {
    device: u32,
    detail: String,
}

/// One event-retained lease over an admitted reservation.
///
/// Deliberately not `Clone`: a second handle is a second authority to retire
/// the same bytes. [`Lease::retire`] consumes the lease on success and hands
/// it back on refusal, so neither path duplicates or strands it.
#[derive(Debug)]
#[must_use = "a lease that is never retired stays charged; retire it explicitly"]
pub struct Lease<C = ManualCompletion> {
    id: LeaseId,
    label: String,
    reservation: Option<Reservation>,
    completion: Option<C>,
    state: LeaseState,
    /// Set on [`Lease::mark_lost`]: which device was lost and why.
    lost: Option<Box<LostInfo>>,
}

fn empty_label(field: &'static str) -> Error {
    Error::InvalidRequest {
        field,
        detail: "a lease is labelled, so a report can name it".into(),
    }
}

impl<C: Completion> Lease<C> {
    /// Acquire a lease against an admitted reservation. The reservation moves
    /// in: from here until retirement there is exactly one authority over
    /// these bytes.
    pub fn acquire(reservation: Reservation, label: impl Into<String>) -> Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(empty_label("label"));
        }
        Ok(Lease {
            id: LeaseId::next(),
            label,
            reservation: Some(reservation),
            completion: None,
            state: LeaseState::Live,
            lost: None,
        })
    }

    /// Identity, for reports.
    pub const fn id(&self) -> LeaseId {
        self.id
    }

    /// The label given at acquisition.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The current state. Completion is observed, not stored: use
    /// [`Lease::retire`] to act on it.
    pub const fn state(&self) -> LeaseState {
        self.state
    }

    /// The reservation this lease holds, while it holds one.
    pub fn reservation_id(&self) -> Option<moxie_memory::ReservationId> {
        self.reservation.as_ref().map(Reservation::id)
    }

    /// Track one completion source for the recorded operation. Exactly one:
    /// a second source would leave two authorities over when the bytes are
    /// free. Only from [`LeaseState::Live`].
    pub fn track(&mut self, completion: C) -> Result<()> {
        if self.state != LeaseState::Live {
            return Err(Error::InvalidRequest {
                field: "lease",
                detail: format!(
                    "{} ({}) cannot track a completion while {:?}",
                    self.id, self.label, self.state
                ),
            });
        }
        self.completion = Some(completion);
        self.state = LeaseState::InFlight;
        Ok(())
    }
    /// Block until the tracked completion is observable. The caller's explicit
    /// wait before [`Lease::retire`]: retirement itself never blocks, so a
    /// missing wait fails loudly at retire time rather than stalling inside
    /// release. Untracked leases have nothing to wait for.
    pub fn synchronize(&self) -> Result<()> {
        match self.completion.as_ref() {
            None => Ok(()),
            Some(c) => c.synchronize(),
        }
    }
    /// Cancel the intent behind this lease. Further tracking is refused, but
    /// the bytes stay held: cancellation retires what was *meant*, never what
    /// is in flight. Withholding wins over cancellation — a lost lease stays
    /// lost.
    pub fn cancel(&mut self) {
        if self.state == LeaseState::Live || self.state == LeaseState::InFlight {
            self.state = LeaseState::Cancelled;
        }
    }

    /// Record that the owning context is lost. Nothing on this lease is ever
    /// reusable again; retirement reports the loss rather than freeing into
    /// it. Fail closed, following the rank-claim teardown precedent.
    pub fn mark_lost(&mut self, device: u32, detail: impl Into<String>) {
        self.state = LeaseState::Lost;
        self.lost = Some(Box::new(LostInfo {
            device,
            detail: detail.into(),
        }));
    }

    fn take_reservation(&mut self) -> Reservation {
        self.reservation
            .take()
            .expect("a lease always holds its reservation until retirement")
    }

    /// Retire when the recorded completion is observed complete, releasing the
    /// reservation into `ledger`. Consumes the lease on success; hands it back
    /// with the error on any refusal, so no path strands charged bytes.
    ///
    /// * [`LeaseState::Live`] with no tracked operation holds no in-flight
    ///   work and releases immediately — the abandon path.
    /// * [`LeaseState::InFlight`] and [`LeaseState::Cancelled`] query the
    ///   source: complete releases, incomplete refuses, an unusable source
    ///   returns the lease with that error.
    /// * [`LeaseState::Lost`] reports the loss. Withheld, never released.
    pub fn retire(mut self, ledger: &mut Ledger) -> std::result::Result<LeaseId, RetireRefused<C>> {
        let fail = |lease: Self, error: Error| RetireRefused { lease, error };
        match self.state {
            LeaseState::Lost => {
                let lost = self.lost.clone().unwrap_or(Box::new(LostInfo {
                    device: u32::MAX,
                    detail: "unknown".into(),
                }));
                let id = self.id;
                let label = self.label.clone();
                Err(fail(
                    self,
                    Error::DeviceLost {
                        device: lost.device,
                        detail: format!("{id} ({label}) is withheld: {}", lost.detail),
                    },
                ))
            }
            LeaseState::Live | LeaseState::InFlight | LeaseState::Cancelled => {
                let complete = match self.completion.as_ref() {
                    // Nothing recorded: nothing in flight.
                    None => true,
                    Some(c) => match c.query_complete() {
                        Ok(done) => done,
                        Err(e) => return Err(fail(self, e)),
                    },
                };
                if !complete {
                    let id = self.id;
                    let label = self.label.clone();
                    let state = self.state;
                    return Err(fail(
                        self,
                        Error::InvalidRequest {
                            field: "lease",
                            detail: format!(
                                "{id} ({label}) is still {state:?}: completion not observed"
                            ),
                        },
                    ));
                }
                let id = self.id;
                let reservation = self.take_reservation();
                match ledger.release(reservation) {
                    Ok(()) => Ok(id),
                    Err(refused) => {
                        self.reservation = Some(refused.reservation);
                        Err(fail(self, refused.error))
                    }
                }
            }
        }
    }
}

#[cfg(feature = "driver")]
impl<'ctx> Lease<moxie_cuda::Event<'ctx>> {
    /// Record the lease's completion after the caller's operation on `stream`:
    /// the event is recorded into the stream's ordered work and the lease
    /// tracks it. This is the contract's `use` step — enqueue first, then bind
    /// the lease to what comes after. Only from [`LeaseState::Live`].
    pub fn use_on(
        &mut self,
        stream: &moxie_cuda::Stream<'ctx>,
        event: moxie_cuda::Event<'ctx>,
    ) -> Result<()> {
        if self.state != LeaseState::Live {
            return Err(Error::InvalidRequest {
                field: "lease",
                detail: format!(
                    "{} ({}) cannot bind a stream while {:?}",
                    self.id, self.label, self.state
                ),
            });
        }
        event.record(stream)?;
        self.completion = Some(event);
        self.state = LeaseState::InFlight;
        Ok(())
    }
}

/// A retirement the lease refused, carrying the lease back.
///
/// Consuming the lease on a failed retire would destroy the only handle to
/// bytes that are still charged — the error-path version of R08. The caller
/// observes completion by other means and retries, cancels, or reports.
#[derive(Debug)]
#[must_use = "the lease is still charged; observe completion and retry, or report it"]
pub struct RetireRefused<C = ManualCompletion> {
    /// The lease, still holding its reservation.
    pub lease: Lease<C>,
    /// Why retirement did not happen.
    pub error: Error,
}

impl<C> core::fmt::Display for RetireRefused<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl<C: core::fmt::Debug> std::error::Error for RetireRefused<C> {}
