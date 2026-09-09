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
use moxie_types::{DeviceTier, Error, HostTier, Result, Scope, Tier};

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
/// let mut src = vec![0u8; 4];
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
    /// The ledger the reservation was outstanding in. Retirement into any
    /// other ledger is refused before any side effect.
    ledger: moxie_memory::LedgerId,
    /// The admitted scope charges bound at acquisition: what this lease may
    /// spend, and where. Read from the ledger, never declared by the caller.
    admitted: Vec<(Scope, u64)>,
    admitted_tiers: Vec<(Scope, Tier, u64)>,
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
    /// Free the operation's allocation after completion was observed. On failure,
    /// return the resource intact for quarantine; do not retry cleanup in Drop.
    fn settle(self) -> std::result::Result<Self::Settled, (Self, Error)>
    where
        Self: Sized;
}

impl SettledResource for () {
    type Settled = ();
    fn settle(self) -> std::result::Result<(), (Self, Error)> {
        Ok(())
    }
}

impl SettledResource for Vec<u8> {
    type Settled = Vec<u8>;
    fn settle(self) -> std::result::Result<Self::Settled, (Self, Error)> {
        Ok(self)
    }
}

/// Validate the complete transient upload footprint before allocation. The host
/// input is charged by capacity, because unused Vec capacity is still live RAM.
pub fn check_upload_fit(
    scopes: &[(Scope, u64)],
    tiers: &[(Scope, Tier, u64)],
    device: moxie_types::DeviceUuid,
    device_bytes: u64,
    host_bytes: u64,
) -> Result<()> {
    for (scope, tier, bytes) in [
        (
            Scope::Device(device),
            Tier::Device(DeviceTier::TransferStaging),
            device_bytes,
        ),
        (Scope::Host, Tier::Host(HostTier::Pageable), host_bytes),
    ] {
        check_fit(scopes, scope, bytes)?;
        let budget = tiers
            .iter()
            .find(|(s, t, _)| *s == scope && *t == tier)
            .map(|(_, _, b)| *b)
            .unwrap_or(0);
        if bytes > budget {
            return Err(Error::CapacityExceeded {
                tier: Some(tier),
                requested_bytes: bytes,
                available_bytes: budget,
            });
        }
    }
    Ok(())
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
        // Only an untracked, non-lost lease is provably unsubmitted. A lost
        // lease may have copied before recording failed, or failed cleanup
        // without an event. Its retained resource is
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
        let (admitted, admitted_tiers) = match ledger
            .outstanding()
            .iter()
            .find(|o| o.id == reservation.id())
        {
            Some(record) => (record.scope_charges.clone(), record.charges.clone()),
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
            ledger: ledger.id(),
            admitted,
            admitted_tiers,
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
            ledger: self.ledger,
            admitted: std::mem::take(&mut self.admitted),
            admitted_tiers: std::mem::take(&mut self.admitted_tiers),
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
    /// [`Lease::submit`], which records before tracking — a bare, unrecorded
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
        // Wrong ledger is refused before any side effect: nothing has been
        // queried, settled, or released, so the lease comes back whole.
        if ledger.id() != self.ledger {
            let id = self.id;
            return Err(fail(
                self,
                Error::InvalidRequest {
                    field: "reservation",
                    detail: format!("{id} is not outstanding in this ledger"),
                },
            ));
        }
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
                let resource = self.retained.take().expect("lease retains its resource");
                let settled = match resource.settle() {
                    Ok(settled) => settled,
                    Err((resource, error)) => {
                        self.retained = Some(resource);
                        self.mark_lost(u32::MAX, format!("settlement failed: {error}"));
                        return Err(fail(self, error));
                    }
                };
                let id = self.id;
                // The owned, non-Clone reservation and checked ledger identity
                // make this release infallible. Cleanup ran while fully charged.
                ledger
                    .release(self.take_reservation())
                    .expect("validated ledger owns the live reservation");
                Ok((id, settled))
            }
        }
    }
}

/// Marker for completions trackable without a driver record step: the manual
/// and scripted test doubles. Driver events are excluded by construction, so
/// no public path binds a bare, unrecorded event — [`Lease::submit`] records
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
    use moxie_cuda::{DeviceBuffer, Event, RankContext, Stream};
    use moxie_types::{Error, Result};

    /// A transient upload. Only an admitted lease can construct one. It never
    /// exposes the allocation for additional untracked asynchronous work.
    #[derive(Debug)]
    pub struct Upload<'ctx> {
        buffer: DeviceBuffer<'ctx>,
        source: Vec<u8>,
        ctx: &'ctx RankContext,
    }

    impl Upload<'_> {
        pub fn len(&self) -> usize {
            self.source.len()
        }
        pub fn is_empty(&self) -> bool {
            self.source.is_empty()
        }
        pub fn source(&self) -> &[u8] {
            &self.source
        }
    }

    impl SettledResource for Upload<'_> {
        type Settled = Vec<u8>;
        fn settle(mut self) -> std::result::Result<Vec<u8>, (Self, Error)> {
            // SAFETY: Upload has no public constructor or mutable buffer access.
            // Its lease either submitted nothing or observed its sole event;
            // no other operation can use this allocation. Failure preserves it.
            if let Err(error) = unsafe { self.buffer.try_free() } {
                return Err((self, error));
            }
            Ok(self.source)
        }
    }

    /// A preparation refusal keeps the unsubmitted lease and caller's source.
    #[derive(Debug)]
    pub struct PrepareRefused<'ctx> {
        pub lease: Lease<Event<'ctx>>,
        pub source: Vec<u8>,
        pub error: Error,
    }
    impl core::fmt::Display for PrepareRefused<'_> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "{}", self.error)
        }
    }
    impl std::error::Error for PrepareRefused<'_> {}

    impl<'ctx> Lease<Event<'ctx>> {
        /// Validate device and host tier/scope charges before any CUDA allocation.
        /// The source is caller-owned on refusal and again after retirement.
        #[allow(clippy::result_large_err)]
        pub fn prepare_upload(
            self,
            ctx: &'ctx RankContext,
            source: Vec<u8>,
        ) -> std::result::Result<Lease<Event<'ctx>, Upload<'ctx>>, PrepareRefused<'ctx>> {
            let checked = if self.state != LeaseState::Live || source.is_empty() {
                Err(Error::InvalidRequest {
                    field: "upload",
                    detail: "preparation requires a live lease and nonempty source".into(),
                })
            } else {
                super::check_upload_fit(
                    &self.admitted,
                    &self.admitted_tiers,
                    ctx.uuid(),
                    source.len() as u64,
                    source.capacity() as u64,
                )
            };
            if let Err(error) = checked {
                return Err(PrepareRefused {
                    lease: self,
                    source,
                    error,
                });
            }
            let buffer = match DeviceBuffer::alloc(ctx, source.len()) {
                Ok(buffer) => buffer,
                Err(error) => {
                    return Err(PrepareRefused {
                        lease: self,
                        source,
                        error,
                    });
                }
            };
            Ok(self.retain(Upload {
                buffer,
                source,
                ctx,
            }))
        }
    }

    impl<'ctx> Lease<Event<'ctx>, Upload<'ctx>> {
        /// Submit once on the upload's own device. Argument validation precedes
        /// all driver calls. Copy/record failures quarantine because submission
        /// state can no longer be proven absent (including asynchronous errors).
        pub fn submit(&mut self, stream: &Stream<'ctx>, event: Event<'ctx>) -> Result<()> {
            if self.state != LeaseState::Live {
                return Err(Error::InvalidRequest {
                    field: "lease",
                    detail: format!("cannot submit while {:?}", self.state),
                });
            }
            let upload = self.retained.as_mut().expect("lease retains upload");
            let uuid = upload.ctx.uuid();
            if stream.device_uuid() != uuid || event.device_uuid() != uuid {
                return Err(Error::InvalidRequest {
                    field: "upload",
                    detail: "stream and event must belong to the upload's device".into(),
                });
            }
            let device = upload.ctx.ordinal();
            // SAFETY: the source and allocation stay owned by this lease through
            // completion or quarantine. Stream identity and bounds are checked.
            let result = unsafe { upload.buffer.copy_from_host_async(&upload.source, stream) }
                .and_then(|()| event.record(stream));
            if let Err(error) = result {
                self.mark_lost(device, format!("submission failed: {error}"));
                return Err(error);
            }
            self.completion = Some(event);
            self.state = LeaseState::InFlight;
            Ok(())
        }

        /// Explicit synchronous validation readback. It waits on this lease's
        /// event and does not enqueue additional asynchronous consumers.
        pub fn readback(&mut self, destination: &mut [u8]) -> Result<()> {
            if self.state != LeaseState::InFlight && self.state != LeaseState::Cancelled {
                return Err(Error::InvalidRequest {
                    field: "lease",
                    detail: "readback requires a submitted, non-lost upload".into(),
                });
            }
            self.synchronize()?;
            let result = self
                .retained
                .as_ref()
                .expect("lease retains upload")
                .buffer
                .copy_to_host(destination);
            if let Err(error) = &result {
                self.persist_loss(error.clone());
            }
            result
        }
    }

    impl Completion for Event<'_> {
        fn query_complete(&self) -> Result<bool> {
            self.is_complete()
        }
        fn synchronize(&self) -> Result<()> {
            self.synchronize()
        }
        fn describe(&self) -> String {
            format!("cuda event on {}", self.device_uuid())
        }
    }
}

#[cfg(feature = "driver")]
pub use driver_binding::{PrepareRefused, Upload};

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
