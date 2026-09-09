//! One lease: admitted bytes bound to one observed completion, retaining
//! the operation's own resources until then.
//!
//! Acquire against a [`Reservation`] in a [`Ledger`], retain the operation's
//! buffers, track exactly one completion source, retire only when that source
//! reports complete. The rules that matter:
//!
//! * The lease owns what it accounts for. A retained source cannot be mutated
//!   or reused early because the caller no longer names it — R07's
//!   source-retention rule, encoded in moves rather than comments. Retirement
//!   hands the resources back with the release.
//! * A failed [`Lease::retire`] hands the lease back, resources included.
//!   Consuming the lease on a refusal would strand charged bytes with no
//!   handle — the error-path version of R08.
//! * Dropping a lease releases nothing. The owned reservation drops with it
//!   and stays charged and visible in [`Ledger::outstanding`], which is the
//!   failure mode that can be found.
//! * Observed device loss persists. Once a completion source reports the
//!   context lost, no later observation — including a racing `Ok(true)` —
//!   reopens the lease.
//!
//! [`Ledger::outstanding`]: moxie_memory::Ledger::outstanding

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use moxie_memory::{Ledger, Reservation};
use moxie_types::{Error, Result, Scope};

/// Observed completion truth for one recorded operation.
///
/// A CUDA event is one implementation (behind the `driver` feature);
/// [`ManualCompletion`] and [`ScriptedCompletion`] are the host-testable ones.
/// Not-ready is a state the caller observes through [`Lease::retire`], never
/// an error smuggled out of this trait: implementations must map their
/// not-ready signal to `Ok(false)`, as [`Event::is_complete`] already does for
/// code 600.
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
    done: Arc<std::sync::atomic::AtomicBool>,
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

/// One scripted observation step for [`ScriptedCompletion`].
#[derive(Debug, Clone)]
pub enum Script {
    /// Report this completion state.
    Ready(bool),
    /// Fail with device loss. The lease persists it: no later step reopens.
    Lost(u32, String),
    /// Fail transiently. The lease stays usable and a later step may complete.
    Broken(String),
}

/// A programmed completion source for host tests: loss-then-complete races,
/// transient failures, and exhaustion, all deterministically.
///
/// A test double, documented as one. Production completions are driver events
/// and manual flags; nothing outside tests should script the truth.
#[derive(Debug, Default)]
pub struct ScriptedCompletion {
    script: Mutex<VecDeque<Script>>,
}

impl ScriptedCompletion {
    /// Play these steps in order, one per query.
    pub fn new(steps: impl IntoIterator<Item = Script>) -> Self {
        ScriptedCompletion {
            script: Mutex::new(steps.into_iter().collect()),
        }
    }

    /// Steps not yet observed.
    pub fn remaining(&self) -> usize {
        self.script.lock().map(|s| s.len()).unwrap_or(0)
    }
}

impl Completion for ScriptedCompletion {
    fn query_complete(&self) -> Result<bool> {
        let step = self
            .script
            .lock()
            .map_err(|_| Error::InvalidRequest {
                field: "completion",
                detail: "scripted completion is unusable".into(),
            })?
            .pop_front();
        match step {
            Some(Script::Ready(done)) => Ok(done),
            Some(Script::Lost(device, detail)) => Err(Error::DeviceLost { device, detail }),
            Some(Script::Broken(detail)) => Err(Error::InvalidRequest {
                field: "completion",
                detail,
            }),
            None => Err(Error::InvalidRequest {
                field: "completion",
                detail: "scripted completion exhausted its steps".into(),
            }),
        }
    }

    fn describe(&self) -> String {
        format!("scripted ({} steps left)", self.remaining())
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
    /// The context is known lost, or recording failed with submission state
    /// unknown. Withheld from every future use.
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
/// `C` is the completion source; `R` is the operation resource retained until
/// retirement — the upload source, the device allocation, or `()` when the
/// lease guards accounting alone. Retirement returns the resource with the
/// release, so premature reuse is a move error, not a rule in a comment:
///
/// ```compile_fail
/// # use moxie_executor::Lease;
/// # fn acquire() -> Lease { todo!() }
/// let lease = acquire();
/// let src = vec![0u8; 4];
/// let lease = lease.retain(src);
/// src.clear(); // moved into the lease: no alias to mutate early
/// ```
///
/// Deliberately not `Clone`: a second handle is a second authority to retire
/// the same bytes. [`Lease::retire`] consumes the lease on success and hands
/// it back on refusal, so neither path duplicates or strands it.
#[derive(Debug)]
#[must_use = "a lease that is never retired stays charged; retire it explicitly"]
pub struct Lease<C = ManualCompletion, R = ()> {
    id: LeaseId,
    label: String,
    reservation: Option<Reservation>,
    /// The admitted scope charges bound at acquisition: what this lease may
    /// spend, and where. Read from the ledger, never declared by the caller.
    admitted: Vec<(Scope, u64)>,
    completion: Option<C>,
    retained: Option<R>,
    state: LeaseState,
    /// Set when loss is observed or recording fails with submission unknown.
    lost: Option<Box<LostInfo>>,
}

/// A resource the lease retains until retirement, and what retirement makes
/// of it. Settlement runs exactly once, on the successful-retire path only:
/// refusal and drop never settle, so a resource is never half-released.
///
/// The bound that keeps accounting honest: a device allocation must stop
/// existing when its usage budget is released, because there is not yet
/// another accounted owner to transfer it to — that transfer arrives with the
/// allocator. Transient `()` and host bytes settle to themselves.
pub trait SettledResource: core::fmt::Debug {
    /// What successful retirement hands back for legal reuse.
    type Settled: core::fmt::Debug;
    /// End the operation: free what must not outlive the budget, return what
    /// is safe to reuse.
    fn settle(self) -> Self::Settled;
}

impl SettledResource for () {
    type Settled = ();
    fn settle(self) -> Self::Settled {}
}

impl SettledResource for Vec<u8> {
    type Settled = Vec<u8>;
    fn settle(self) -> Self::Settled {
        self
    }
}

/// Whether `bytes` in `scope` fit the admitted charges. Pure, so the check
/// the submission path enforces is testable without a driver.
pub fn check_fit(admitted: &[(Scope, u64)], scope: Scope, bytes: u64) -> Result<()> {
    match admitted.iter().find(|(s, _)| *s == scope) {
        None => Err(Error::InvalidRequest {
            field: "lease",
            detail: format!("{scope} has no admitted budget on this lease"),
        }),
        Some((_, budget)) if bytes > *budget => Err(Error::CapacityExceeded {
            tier: None,
            requested_bytes: bytes,
            available_bytes: *budget,
        }),
        Some(_) => Ok(()),
    }
}

impl<C, R> Drop for Lease<C, R> {
    fn drop(&mut self) {
        // A lease with no tracked completion submitted nothing provable: a
        // Live abandon, or a Lost lease quarantined before anything could be
        // enqueued. Everything else may still be in flight — including work
        // submitted before a failed record — so the retained resource is
        // deliberately withheld rather than freed early. The reservation drops
        // normally and stays charged and visible, naming the hold.
        let submitted = self.completion.is_some() || self.state == LeaseState::Lost;
        if submitted {
            std::mem::forget(self.retained.take());
        }
    }
}

fn empty_label(field: &'static str) -> Error {
    Error::InvalidRequest {
        field,
        detail: "a lease is labelled, so a report can name it".into(),
    }
}

/// A refusal to acquire, carrying the reservation back.
///
/// The reservation moves into `acquire` before its ledger binding is known;
/// consuming it on a failed acquisition would strand charged bytes with no
/// handle — the same shape as R08, one step earlier.
#[derive(Debug)]
#[must_use = "the reservation is still charged; acquire again with a valid label and ledger"]
pub struct AcquireRefused {
    /// The reservation, never admitted under a lease.
    pub reservation: Reservation,
    /// Why acquisition did not happen.
    pub error: Error,
}

impl core::fmt::Display for AcquireRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for AcquireRefused {}

impl<C: Completion> Lease<C, ()> {
    /// Acquire a lease against an admitted reservation in `ledger`. The
    /// admitted scope charges are read from the ledger and bound here, so the
    /// lease names real budget rather than a caller declaration. The label is
    /// validated before anything else: a refusal returns the reservation.
    pub fn acquire(
        ledger: &Ledger,
        reservation: Reservation,
        label: impl Into<String>,
    ) -> std::result::Result<Self, AcquireRefused> {
        let label = label.into();
        if label.is_empty() {
            return Err(AcquireRefused {
                reservation,
                error: empty_label("label"),
            });
        }
        let admitted = match ledger
            .outstanding()
            .iter()
            .find(|o| o.id == reservation.id())
        {
            Some(record) => record.scope_charges.clone(),
            None => {
                return Err(AcquireRefused {
                    reservation,
                    error: Error::InvalidRequest {
                        field: "reservation",
                        detail: "reservation is not outstanding in this ledger".into(),
                    },
                });
            }
        };
        Ok(Lease {
            id: LeaseId::next(),
            label,
            reservation: Some(reservation),
            admitted,
            completion: None,
            retained: Some(()),
            state: LeaseState::Live,
            lost: None,
        })
    }

    /// Move the operation resource under this lease. Exactly once, enforced by
    /// type: only a resourceless lease retains, so a second retention cannot
    /// be written and the caller keeps no alias.
    pub fn retain<R>(mut self, resource: R) -> Lease<C, R> {
        // Field by field through `&mut`: `Lease` implements `Drop`, so
        // destructuring moves are illegal. What remains drops as an empty,
        // Live shell — never observed, since this function consumes `self`.
        Lease {
            id: self.id,
            label: std::mem::take(&mut self.label),
            reservation: self.reservation.take(),
            admitted: std::mem::take(&mut self.admitted),
            completion: self.completion.take(),
            retained: Some(resource),
            state: self.state,
            lost: self.lost.take(),
        }
    }
}
impl From<AcquireRefused> for Error {
    /// The reason, without the returned handle. Use the struct itself when
    /// the reservation must survive the error.
    fn from(refused: AcquireRefused) -> Error {
        refused.error
    }
}

impl<C: Completion, R> Lease<C, R> {
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

    /// The retained resource, while this lease holds one. Owner reads (like a
    /// post-completion readback) go through here; ownership returns at
    /// retirement.
    pub fn retained(&self) -> Option<&R> {
        self.retained.as_ref()
    }

    /// Admitted scope charges bound at acquisition: `(scope, bytes)` pairs
    /// read from the ledger, the budget this lease may spend.
    pub fn admitted(&self) -> &[(Scope, u64)] {
        &self.admitted
    }

    /// Admitted bytes across all scopes.
    pub fn admitted_bytes(&self) -> u64 {
        self.admitted.iter().map(|(_, b)| b).sum()
    }

    /// Track one completion source for the recorded operation. Exactly one:
    /// a second source would leave two authorities over when the bytes are
    /// free. Only from [`LeaseState::Live`].
    ///
    /// Host-side and test completions only. Driver events bind through
    /// [`Lease::use_on`], which records before tracking — a bare, unrecorded
    /// event must never reach retirement.
    pub fn track_manual(&mut self, completion: C) -> Result<()>
    where
        C: TrackableManual,
    {
        if self.state != LeaseState::Live {
            return Err(Error::InvalidRequest {
                field: "lease",
                detail: format!(
                    "{} ({}) cannot track a completion while {:?}",
                    self.id, self.label, self.state
                ),
            });
        }
        self.completion = Some(completion.into_tracked());
        self.state = LeaseState::InFlight;
        Ok(())
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

    /// Block until the tracked completion is observable. The caller's explicit
    /// wait before [`Lease::retire`]: retirement itself never blocks, so a
    /// missing wait fails loudly at retire time rather than stalling inside
    /// release. Untracked leases have nothing to wait for. Observed loss
    /// persists, exactly as in retirement.
    pub fn synchronize(&mut self) -> Result<()> {
        let observed = match self.completion.as_ref() {
            None => return Ok(()),
            Some(c) => c.synchronize(),
        };
        if let Err(e) = observed {
            self.persist_loss(e.clone());
            return Err(e);
        }
        Ok(())
    }

    /// Record observed device loss permanently. A racing later observation —
    /// including `Ok(true)` — never reopens the lease.
    fn persist_loss(&mut self, error: Error) {
        if let Error::DeviceLost { device, detail } = error {
            self.state = LeaseState::Lost;
            self.lost = Some(Box::new(LostInfo { device, detail }));
        }
    }

    fn take_reservation(&mut self) -> Reservation {
        self.reservation
            .take()
            .expect("a lease always holds its reservation until retirement")
    }

    fn take_retained(&mut self) -> R::Settled
    where
        R: SettledResource,
    {
        self.retained
            .take()
            .expect("a lease always holds its retained resource until retirement")
            .settle()
    }

    /// Retire when the recorded completion is observed complete, releasing the
    /// reservation into `ledger` and settling the retained resource for legal
    /// reuse. Settlement ends the allocation where the budget ends: a device
    /// buffer stops existing here, so no live allocation outlives its charge.
    /// Consumes the lease on success; hands it back — resources included — on
    /// any refusal, so no path strands charged bytes.
    /// * [`LeaseState::Lost`] reports the loss. Withheld, never released.
    // The refused lease comes back whole by design (R08): its size is the
    // guarantee, not an accident, so boxing it to satisfy the lint would only
    // hide what every refusal carries.
    #[allow(clippy::result_large_err)]
    pub fn retire(
        mut self,
        ledger: &mut Ledger,
    ) -> std::result::Result<(LeaseId, R::Settled), RetireRefused<C, R>>
    where
        R: SettledResource,
    {
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
                        Err(e) => {
                            self.persist_loss(e.clone());
                            return Err(fail(self, e));
                        }
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
                    Ok(()) => {
                        let resource = self.take_retained();
                        Ok((id, resource))
                    }
                    Err(refused) => {
                        self.reservation = Some(refused.reservation);
                        Err(fail(self, refused.error))
                    }
                }
            }
        }
    }
}

/// Marker for completions trackable without a driver record step: the manual
/// and scripted test doubles. Driver events are excluded by construction, so
/// no public path binds a bare, unrecorded event — [`Lease::use_on`] records
/// first.
pub trait TrackableManual: Completion {
    /// Wrap into the tracked slot. Identity for the doubles.
    fn into_tracked(self) -> Self;
}

impl TrackableManual for ManualCompletion {
    fn into_tracked(self) -> Self {
        self
    }
}

impl TrackableManual for ScriptedCompletion {
    fn into_tracked(self) -> Self {
        self
    }
}

#[cfg(feature = "driver")]
mod driver_binding {
    use super::{Completion, Lease, LeaseState, SettledResource};
    use moxie_types::{Error, Result};

    /// A staged host-to-device upload, retained by its lease from submission
    /// through retirement: the device allocation and the source bytes travel
    /// together, so neither can be freed, mutated or reused early.
    ///
    /// Settlement drops the device allocation — the event is complete by then,
    /// allocation never outlives its budget: there is not yet another
    /// accounted owner to transfer it to.
    #[derive(Debug)]
    pub struct Upload<'ctx> {
        buffer: moxie_cuda::DeviceBuffer<'ctx>,
        source: Vec<u8>,
        scope: moxie_types::Scope,
    }

    impl<'ctx> Upload<'ctx> {
        /// Allocate room for `source` without enqueueing anything. The upload
        /// provably has no submitted work until [`Lease::submit`] runs, which
        /// is what makes the pre-submission state droppable without a hold.
        pub fn prepare(
            ctx: &'ctx moxie_cuda::RankContext,
            scope: moxie_types::Scope,
            source: Vec<u8>,
        ) -> Result<Self> {
            if source.is_empty() {
                return Err(Error::InvalidRequest {
                    field: "source",
                    detail: "an upload stages bytes, not an empty source".into(),
                });
            }
            let buffer = moxie_cuda::DeviceBuffer::alloc(ctx, source.len())?;
            Ok(Upload {
                buffer,
                source,
                scope,
            })
        }

        /// Staged byte count.
        pub fn len(&self) -> usize {
            self.source.len()
        }

        /// Whether nothing was staged. Never true from [`Upload::prepare`].
        pub fn is_empty(&self) -> bool {
            self.source.is_empty()
        }

        /// The scope this upload will spend against, checked at submission.
        pub fn scope(&self) -> moxie_types::Scope {
            self.scope
        }

        /// The device allocation under lease.
        pub fn buffer(&self) -> &moxie_cuda::DeviceBuffer<'ctx> {
            &self.buffer
        }

        /// The retained source. Readable throughout; ownership returns at
        /// retirement.
        pub fn source(&self) -> &[u8] {
            &self.source
        }
    }

    impl SettledResource for Upload<'_> {
        type Settled = Vec<u8>;
        fn settle(self) -> Vec<u8> {
            // `buffer` drops here, after observed completion: freeing is safe
            // and the accounting released alongside covers nothing live.
            self.source
        }
    }

    impl<'ctx> Lease<moxie_cuda::Event<'ctx>, Upload<'ctx>> {
        /// Submit the retained upload as one operation: check the admitted
        /// budget, run the async copy, record the completion event, and track
        /// it. Pending state and its actual completion dependency are
        /// established together — there is no step at which submitted work is
        /// untracked, and staging never precedes admission because the lease
        /// only exists after it. Only from [`LeaseState::Live`].
        ///
        /// A failed record quarantines the lease: the copy may already be
        /// submitted, so nothing is provably free and retirement must never
        /// release. Anything earlier fails with everything in place.
        pub fn submit(
            &mut self,
            ctx: &moxie_cuda::RankContext,
            stream: &moxie_cuda::Stream<'ctx>,
            event: moxie_cuda::Event<'ctx>,
        ) -> Result<()> {
            if self.state != LeaseState::Live {
                return Err(Error::InvalidRequest {
                    field: "lease",
                    detail: format!(
                        "{} ({}) cannot submit while {:?}",
                        self.id, self.label, self.state
                    ),
                });
            }
            let (scope, bytes) = match self.retained.as_ref() {
                Some(upload) => (upload.scope(), upload.len() as u64),
                None => {
                    return Err(Error::InvalidRequest {
                        field: "lease",
                        detail: format!(
                            "{} ({}) has no retained upload to submit",
                            self.id, self.label
                        ),
                    });
                }
            };
            super::check_fit(&self.admitted, scope, bytes)?;
            // The upload is retained by this lease until retirement, so source
            // and buffer outlive the enqueue; the buffer holds at least the
            // source length by construction in `Upload::prepare`. The borrow
            // ends before any mutation below.
            let upload = self.retained.as_mut().expect("checked above");
            // SAFETY: bounds checked by construction above; the source lives
            // in the retained upload and the destination holds it.
            unsafe { upload.buffer.copy_from_host_async(&upload.source, stream)? };
            // The copy is submitted; from here only quarantine or tracking
            // follow. Record before anything else so a failed record cannot
            // leave submitted work untracked.
            if let Err(error) = event.record(stream) {
                let device = ctx.ordinal();
                self.state = LeaseState::Lost;
                self.lost = Some(Box::new(super::LostInfo {
                    device,
                    detail: format!("event record failed, submission state unknown: {error}"),
                }));
                return Err(Error::InvalidRequest {
                    field: "lease",
                    detail: format!(
                        "{} ({}) is quarantined: event record failed: {error}",
                        self.id, self.label
                    ),
                });
            }
            self.completion = Some(event);
            self.state = LeaseState::InFlight;
            Ok(())
        }

        /// Record the lease's completion after the caller's own operation on
        /// `stream`: the event is recorded into the stream's ordered work and
        /// the lease tracks it. For operations the caller enqueues itself
        /// (launches); uploads go through [`Lease::submit`], which ties the
        /// stream. Only from [`LeaseState::Live`].
        ///
        /// A failed record quarantines the lease: the operation may already be
        /// submitted, so nothing is provably free and retirement must never
        /// release. The context ordinal names the loss.
        pub fn use_on(
            &mut self,
            ctx: &moxie_cuda::RankContext,
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
            if let Err(e) = event.record(stream) {
                let device = ctx.ordinal();
                self.state = LeaseState::Lost;
                self.lost = Some(Box::new(super::LostInfo {
                    device,
                    detail: format!("event record failed, submission state unknown: {e}"),
                }));
                return Err(Error::InvalidRequest {
                    field: "lease",
                    detail: format!(
                        "{} ({}) is quarantined: event record failed: {e}",
                        self.id, self.label
                    ),
                });
            }
            self.completion = Some(event);
            self.state = LeaseState::InFlight;
            Ok(())
        }
    }

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
}

#[cfg(feature = "driver")]
pub use driver_binding::Upload;

/// A retirement the lease refused, carrying the lease back.
///
/// Consuming the lease on a failed retire would destroy the only handle to
/// bytes that are still charged — the error-path version of R08. The caller
/// observes completion by other means and retries, cancels, or reports.
#[derive(Debug)]
#[must_use = "the lease is still charged; observe completion and retry, or report it"]
pub struct RetireRefused<C = ManualCompletion, R = ()> {
    /// The lease, still holding its reservation and resources.
    pub lease: Lease<C, R>,
    /// Why retirement did not happen.
    pub error: Error,
}

impl<C: core::fmt::Debug, R> core::fmt::Display for RetireRefused<C, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl<C: core::fmt::Debug, R: core::fmt::Debug> std::error::Error for RetireRefused<C, R> {}
