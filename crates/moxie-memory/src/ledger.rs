//! The one authority that admits and charges bytes.
//!
//! R02 is the failure this repairs: legacy's `ResidencyManager` modelled
//! accesses and bytes for a simulator while the model runtimes allocated for
//! themselves, so the thing that knew the budget was not the thing that spent
//! it. Here there is one ledger, admission is atomic, and a reservation is
//! released by name rather than by scope exit.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{DeviceTier, Error, HostTier, Result, Scope, ScopeKind, Tier};

use crate::report::{
    AdmissionReport, BindingConstraint, BindingKind, HostBackedPlan, LegalAlternative, Rejection,
    ScopeReport, TierReport,
};
use crate::request::{BufferRequest, PlanRequest, ReserveRule, Scaling};
use crate::snapshot::CapacitySnapshot;

/// One ledger's process-unique identity, so a reservation cannot be released
/// against a different ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LedgerId(u64);

/// One admitted envelope's process-unique identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReservationId(u64);

impl LedgerId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        LedgerId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl ReservationId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        ReservationId(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Proof that an envelope is reserved, and the only way to give it back.
///
/// **Deliberately not `Clone` and not `Copy`.** A second handle to one
/// reservation is a second authority to release it, and the workspace has
/// already met that failure twice in identity-bearing types. Releasing consumes
/// it, so a double release cannot be written:
///
/// ```compile_fail
/// # use moxie_memory::{Ledger, PlanRequest, CapacitySnapshot};
/// # use moxie_types::Scope;
/// let mut l = Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).unwrap()]).unwrap();
/// let r = l.admit(&PlanRequest::new("p", ["decode"]).unwrap()).unwrap();
/// let second = r.try_clone()?; // a second authority over the same bytes
/// drop(second);
/// ```
///
/// ```compile_fail
/// # use moxie_memory::{Ledger, PlanRequest, CapacitySnapshot};
/// # use moxie_types::Scope;
/// let mut l = Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).unwrap()]).unwrap();
/// let r = l.admit(&PlanRequest::new("p", ["decode"]).unwrap()).unwrap();
/// l.release(r).unwrap();
/// l.release(r).unwrap(); // moved into the first release
/// ```
///
/// Dropping one does **not** release it. Document 02 forbids `Drop` alone from
/// freeing an in-flight resource, and R08 is the concrete cost of the opposite
/// habit: a lease released on "next token" leaked whenever a turn ended and no
/// next token came. A dropped reservation stays charged and stays visible in
/// [`Ledger::outstanding`], which is the failure mode that can be found.
#[derive(Debug)]
#[must_use = "a reservation that is never released stays charged; release it explicitly"]
pub struct Reservation {
    id: ReservationId,
    ledger: LedgerId,
}

impl Reservation {
    pub const fn id(&self) -> ReservationId {
        self.id
    }

    pub const fn ledger(&self) -> LedgerId {
        self.ledger
    }
}

/// A release the ledger refused, carrying the reservation back.
///
/// Consuming the reservation on a *failed* release would destroy the only
/// handle to bytes that are still charged -- the same shape of leak as R08,
/// created by the error path instead of by the happy one. So a refusal returns
/// it, and the caller can release it against the ledger it belongs to.
#[derive(Debug)]
#[must_use = "the reservation is still charged; release it against its own ledger"]
pub struct ReleaseRefused {
    pub reservation: Reservation,
    pub error: Error,
}

impl core::fmt::Display for ReleaseRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)
    }
}

/// Why an admission did not produce a reservation.
///
/// The two are not the same outcome and must not be handled the same way: a
/// malformed request is a bug in the caller, while a refusal is an answer about
/// this machine that the caller can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmitError {
    /// The request could not be evaluated at all.
    Invalid(Error),
    /// The request was evaluated and does not fit.
    Rejected(Rejection),
}

impl AdmitError {
    pub fn as_rejection(&self) -> Option<&Rejection> {
        match self {
            AdmitError::Rejected(r) => Some(r),
            AdmitError::Invalid(_) => None,
        }
    }
}

impl core::fmt::Display for AdmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AdmitError::Invalid(e) => write!(f, "invalid plan request: {e}"),
            AdmitError::Rejected(r) => write!(f, "{r}"),
        }
    }
}

impl std::error::Error for AdmitError {}

impl From<Error> for AdmitError {
    fn from(e: Error) -> Self {
        AdmitError::Invalid(e)
    }
}

impl From<AdmitError> for Error {
    fn from(e: AdmitError) -> Error {
        match e {
            AdmitError::Invalid(e) => e,
            AdmitError::Rejected(r) => r.into(),
        }
    }
}

#[derive(Debug, Clone)]
struct ScopeState {
    snapshot: CapacitySnapshot,
    /// Per-tier commitments, against the per-tier caps.
    committed: crate::fallible::Map<Tier, u64>,
    /// The scope's committed total: the sum of each admitted plan's **scope
    /// peak**, not the sum of its tier commitments.
    ///
    /// The two differ, and the difference is the whole peak-not-sum rule. Inside
    /// one plan, tiers that are live at different stages never coexist, so the
    /// scope holds their maximum rather than their total. Across plans there is
    /// no shared timeline -- an admitted plan may be at its own peak whenever
    /// another is at its -- so their peaks add.
    committed_scope: u64,
}

#[derive(Debug, Clone)]
struct OutstandingRecord {
    label: String,
    charges: Vec<(Scope, Tier, u64)>,
    scope_charges: Vec<(Scope, u64)>,
}

/// What an admitted plan holds: its label, its per-tier charges and the scope
/// totals it occupies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outstanding {
    pub id: ReservationId,
    pub label: String,
    pub charges: Vec<(Scope, Tier, u64)>,
    pub scope_charges: Vec<(Scope, u64)>,
}

/// The resource authority. Every byte any consumer holds is charged here.
#[derive(Debug)]
pub struct Ledger {
    id: LedgerId,
    scopes: BTreeMap<Scope, ScopeState>,
    outstanding: crate::fallible::Map<ReservationId, OutstandingRecord>,
}

/// The intermediate result of evaluating a request against the ledger.
struct Evaluation {
    report: AdmissionReport,
    /// The per-tier peaks this request would commit.
    charges: Vec<(Scope, Tier, u64)>,
    /// The scope peaks this request would commit, which are at or below the
    /// totals of the per-tier peaks above.
    scope_charges: Vec<(Scope, u64)>,
    binding: Vec<BindingConstraint>,
    /// The changes this request makes capable of helping. Empty when nothing
    /// binds.
    alternatives: Vec<LegalAlternative>,
}

impl Ledger {
    /// A ledger over one snapshot per scope. A scope declared twice is refused:
    /// two capacities for one device is how a budget quietly doubles.
    pub fn new(snapshots: impl IntoIterator<Item = CapacitySnapshot>) -> Result<Self> {
        let mut scopes = BTreeMap::new();
        for snapshot in snapshots {
            let scope = snapshot.scope();
            if scopes.contains_key(&scope) {
                return Err(Error::InvalidRequest {
                    field: "snapshots",
                    detail: format!("{scope} is declared twice"),
                });
            }
            scopes.insert(
                scope,
                ScopeState {
                    snapshot,
                    committed: crate::fallible::Map::new(),
                    committed_scope: 0,
                },
            );
        }
        Ok(Ledger {
            id: LedgerId::next(),
            scopes,
            outstanding: crate::fallible::Map::new(),
        })
    }

    pub const fn id(&self) -> LedgerId {
        self.id
    }

    pub fn scopes(&self) -> impl Iterator<Item = Scope> + '_ {
        self.scopes.keys().copied()
    }

    pub fn snapshot(&self, scope: Scope) -> Option<&CapacitySnapshot> {
        self.scopes.get(&scope).map(|s| &s.snapshot)
    }

    /// Bytes committed in one tier.
    pub fn committed(&self, scope: Scope, tier: Tier) -> u64 {
        self.scopes
            .get(&scope)
            .and_then(|s| s.committed.get(&tier))
            .copied()
            .unwrap_or(0)
    }

    /// Bytes committed against a scope's budget: the sum of the admitted plans'
    /// scope peaks, which is at or below the sum of this scope's per-tier
    /// commitments.
    pub fn scope_committed(&self, scope: Scope) -> u64 {
        self.scopes
            .get(&scope)
            .map(|s| s.committed_scope)
            .unwrap_or(0)
    }

    /// Every reservation that has been admitted and not released.
    ///
    /// R08: the leak that mattered was invisible, not large. A dropped
    /// `Reservation` still appears here.
    pub fn outstanding(&self) -> Vec<Outstanding> {
        self.outstanding
            .iter()
            .map(|(id, r)| Outstanding {
                id: *id,
                label: r.label.clone(),
                charges: r.charges.clone(),
                scope_charges: r.scope_charges.clone(),
            })
            .collect()
    }

    /// Visit every outstanding reservation's charges, allocating **nothing**.
    ///
    /// [`Ledger::outstanding`] clones a label and two vectors per reservation,
    /// which is right for a diagnostic and wrong for a caller that must not
    /// abort. A trace is such a caller: it runs inside a generation step, and an
    /// allocation failure there has to be a typed error the transaction can roll
    /// back rather than a panic that takes the rollback with it. So the charges
    /// are handed out by reference and the caller keeps what it needs in storage
    /// it reserved.
    pub fn for_each_charge(&self, mut visit: impl FnMut(ReservationId, Scope, Tier, u64)) {
        for (id, record) in &self.outstanding {
            for (scope, tier, bytes) in &record.charges {
                visit(*id, *scope, *tier, *bytes);
            }
        }
    }

    /// One reservation's scope charges, by reference, allocating **nothing**.
    ///
    /// `outstanding()` clones a label and two vectors per reservation, and an
    /// arena calls it only to find the one reservation it was handed -- on the
    /// admission path, where an allocation failure has to be a refusal.
    /// Independent review found that call; this is what it should have been.
    pub fn scope_charges_of(&self, id: ReservationId) -> Option<&[(Scope, u64)]> {
        self.outstanding
            .get(&id)
            .map(|record| record.scope_charges.as_slice())
    }

    /// One reservation's tier charges, by reference, allocating **nothing**.
    pub fn charges_of(&self, id: ReservationId) -> Option<&[(Scope, Tier, u64)]> {
        self.outstanding
            .get(&id)
            .map(|record| record.charges.as_slice())
    }

    /// How many reservations are outstanding, without building them.
    pub fn outstanding_count(&self) -> usize {
        self.outstanding.len()
    }

    /// How many scopes this ledger knows, for a caller reserving storage.
    pub fn scope_count(&self) -> usize {
        self.scopes.len()
    }

    /// The full breakdown for a request, whether or not it would be admitted.
    /// Pure with respect to live resources: it commits nothing (document 02).
    pub fn preview(&self, request: &PlanRequest) -> Result<AdmissionReport> {
        Ok(self.evaluate(request)?.report)
    }

    /// Reserve a plan's complete envelope, or nothing at all.
    ///
    /// Document 02: `admit` "atomically reserves its complete resource
    /// envelope". On refusal every counter in this ledger is what it was --
    /// including the tiers that would have fit, which is the half a
    /// commit-as-you-go loop gets wrong.
    pub fn admit(&mut self, request: &PlanRequest) -> std::result::Result<Reservation, AdmitError> {
        let evaluation = match self.evaluate(request) {
            Ok(e) => e,
            Err(e) => return Err(AdmitError::Invalid(e)),
        };
        if !evaluation.binding.is_empty() {
            let shortfall = evaluation
                .binding
                .iter()
                .map(BindingConstraint::shortfall_bytes)
                .max()
                .unwrap_or(0);
            return Err(AdmitError::Rejected(Rejection {
                report: evaluation.report,
                binding: evaluation.binding,
                shortfall_bytes: shortfall,
                alternatives: evaluation.alternatives,
            }));
        }

        // **Everything that can fail happens before anything is charged.**
        //
        // This used to add every committed counter first and then allocate the
        // record that names them. An allocation failure between the two left
        // capacity charged against a reservation that did not exist, which
        // `outstanding()` cannot see and no release can undo -- independent
        // review named the ordering. So the label, the record's slot and the
        // per-tier map entries are all obtained first; the additions come last
        // and cannot fail.
        let no_room = || AdmitError::Invalid(crate::fallible::no_room());
        let label = crate::fallible::string(request.label()).map_err(AdmitError::Invalid)?;
        let id = ReservationId::next();
        // **Every allocation happens before anything is charged**, and the
        // charging then cannot fail. This used to add the committed counters
        // first and allocate the record that names them second: a failure
        // between the two left capacity charged against a reservation nobody
        // could see and no release could undo. Independent review named the
        // ordering.
        //
        // A tier's entry is created here, zeroed, so the additions below only
        // ever touch an entry that already exists.
        // **The reservation's slot first, then the tier entries.** The entries
        // are a change to the ledger's own map, and taking them before the slot
        // meant a failure to reserve the slot returned a refusal with the map
        // already carrying zeroed rows it did not have before. The logical
        // counters were right and the structure was not, which is the same
        // shape one level down. Independent review found the ordering.
        self.outstanding.try_reserve_one().map_err(|_| no_room())?;
        // **Room for every missing entry, in every scope, before any of them is
        // installed.** Reserving one at a time and inserting as it goes leaves
        // the earlier rows behind when a later reservation fails -- the affine
        // request touches a Host scope and a Device scope, so that is two maps
        // and directly reachable. Independent review found it. Counting first
        // and reserving per scope makes the whole preparation atomic: after the
        // loop below, no insertion can fail.
        {
            // One pass to count what each scope is missing, one to take the
            // room, and only then the insertions.
            let mut wanted: Vec<(Scope, usize)> =
                crate::fallible::with_capacity(evaluation.charges.len())
                    .map_err(AdmitError::Invalid)?;
            for (scope, tier, _) in &evaluation.charges {
                let state = self
                    .scopes
                    .get(scope)
                    .expect("evaluation rejected unknown scopes");
                if state.committed.contains_key(tier) {
                    continue;
                }
                match wanted.iter_mut().find(|(s, _)| s == scope) {
                    Some((_, n)) => *n += 1,
                    None => wanted.push((*scope, 1)),
                }
            }
            for (scope, additional) in &wanted {
                let state = self
                    .scopes
                    .get_mut(scope)
                    .expect("evaluation rejected unknown scopes");
                state
                    .committed
                    .try_reserve(*additional)
                    .map_err(|_| no_room())?;
            }
            for (scope, tier, _) in &evaluation.charges {
                let state = self
                    .scopes
                    .get_mut(scope)
                    .expect("evaluation rejected unknown scopes");
                if !state.committed.contains_key(tier) {
                    state.committed.insert(*tier, 0);
                }
            }
        }

        // From here nothing allocates and nothing can fail: every sum was
        // checked during evaluation, and adding a peak that already fits under
        // a ceiling cannot overflow it.
        for (scope, tier, bytes) in &evaluation.charges {
            let state = self
                .scopes
                .get_mut(scope)
                .expect("evaluation rejected unknown scopes");
            *state
                .committed
                .get_mut(tier)
                .expect("the entry was created above") += *bytes;
        }
        for (scope, bytes) in &evaluation.scope_charges {
            let state = self
                .scopes
                .get_mut(scope)
                .expect("evaluation rejected unknown scopes");
            state.committed_scope += *bytes;
        }
        self.outstanding.insert(
            id,
            OutstandingRecord {
                label,
                charges: evaluation.charges,
                scope_charges: evaluation.scope_charges,
            },
        );
        Ok(Reservation {
            id,
            ledger: self.id,
        })
    }

    /// Give an envelope back. Consumes the reservation on success, so it
    /// happens once; hands it back on refusal, so nothing is stranded.
    pub fn release(&mut self, reservation: Reservation) -> std::result::Result<(), ReleaseRefused> {
        if reservation.ledger != self.id {
            let error = match crate::fallible::text(format_args!(
                "reservation {} belongs to ledger {}, not {}",
                reservation.id.get(),
                reservation.ledger.get(),
                self.id.get()
            )) {
                Ok(detail) => Error::InvalidRequest {
                    field: "reservation",
                    detail,
                },
                Err(error) => error,
            };
            return Err(ReleaseRefused { reservation, error });
        }
        let record = self
            .outstanding
            .remove(&reservation.id)
            .expect("a reservation exists exactly while its record does");
        for (scope, tier, bytes) in record.charges {
            let state = self
                .scopes
                .get_mut(&scope)
                .expect("a charge names a scope this ledger admitted");
            let entry = state
                .committed
                .get_mut(&tier)
                .expect("a charge names a tier this ledger committed");
            *entry = entry
                .checked_sub(bytes)
                .expect("released bytes were charged by this reservation");
        }
        for (scope, bytes) in record.scope_charges {
            let state = self
                .scopes
                .get_mut(&scope)
                .expect("a charge names a scope this ledger admitted");
            state.committed_scope = state
                .committed_scope
                .checked_sub(bytes)
                .expect("released bytes were charged by this reservation");
        }
        Ok(())
    }

    // --- evaluation ----------------------------------------------------------

    fn evaluate(&self, request: &PlanRequest) -> Result<Evaluation> {
        for scope in request.scopes()? {
            if !self.scopes.contains_key(&scope) {
                return Err(Error::InvalidRequest {
                    field: "scope",
                    // Task 0029's contract: an allocation failure is
                    // `CapacityExceeded`, not a different error missing its
                    // data, so a failed `text` propagates rather than degrades.
                    detail: crate::fallible::text(format_args!(
                        "{scope} has no capacity snapshot in this ledger"
                    ))?,
                });
            }
        }

        let base = peaks(request, &|_| false)?;

        // Reserved, then filled. `collect()` reallocates infallibly, and
        // admission may not abort to say a plan does not fit.
        let mut charges: Vec<(Scope, Tier, u64)> = crate::fallible::with_capacity(base.tier.len())?;
        for ((scope, tier), (bytes, _)) in base.tier.iter() {
            if *bytes > 0 {
                charges.push((*scope, *tier, *bytes));
            }
        }
        charges.sort_unstable_by_key(|(scope, tier, _)| (*scope, *tier));

        let mut binding: Vec<BindingConstraint> = Vec::new();
        let mut scope_reports: Vec<ScopeReport> = Vec::new();
        let mut scope_charges: Vec<(Scope, u64)> = Vec::new();

        for (scope, state) in &self.scopes {
            let (scope_peak, scope_peak_stage) = base.scope_peak(*scope);
            let committed_charged = self.scope_committed(*scope);
            let admissible = state.snapshot.admissible_bytes();
            let needed = committed_charged
                .checked_add(scope_peak)
                .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;

            let mut tiers: Vec<TierReport> = Vec::new();
            for tier in Tier::valid_in(scope.kind()) {
                let (peak, peak_stage) = base.tier_peak(*scope, tier);
                let committed = self.committed(*scope, tier);
                let cap = state.snapshot.tier_cap(tier);
                let tier_needed = committed
                    .checked_add(peak)
                    .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
                if let Some(cap) = cap
                    && tier_needed > cap
                {
                    crate::fallible::push(
                        &mut binding,
                        BindingConstraint {
                            scope: *scope,
                            tier,
                            kind: BindingKind::TierCap,
                            needed_bytes: tier_needed,
                            available_bytes: cap,
                            peak_stage,
                        },
                    )?;
                }
                crate::fallible::push(
                    &mut tiers,
                    TierReport {
                        tier,
                        cap_bytes: cap,
                        committed_bytes: committed,
                        request_peak_bytes: peak,
                        peak_stage,
                        remaining_headroom_bytes: cap.map(|c| c.saturating_sub(tier_needed)),
                        virtual_extent_bytes: base
                            .virtual_live
                            .get(&(*scope, tier))
                            .and_then(|row| row.iter().copied().max())
                            .unwrap_or(0),
                    },
                )?;
            }

            if needed > admissible {
                // Name the tier a caller should look at first: the largest
                // contributor at the stage where the scope peaks. Ties break on
                // `Tier::ALL` order, so the answer is deterministic.
                let largest = base
                    .live
                    .iter()
                    .filter(|((s, _), _)| s == scope)
                    .map(|((_, tier), row)| (row[scope_peak_stage as usize], *tier))
                    .max_by(|a, b| {
                        a.0.cmp(&b.0).then_with(|| {
                            // Later in ALL order loses, so the earlier tier wins.
                            position(b.1).cmp(&position(a.1))
                        })
                    })
                    .map(|(_, tier)| tier)
                    .unwrap_or(Tier::Device(DeviceTier::SafetyHeadroom));
                crate::fallible::push(
                    &mut binding,
                    BindingConstraint {
                        scope: *scope,
                        tier: largest,
                        kind: BindingKind::ScopeBudget,
                        needed_bytes: needed,
                        available_bytes: admissible,
                        peak_stage: scope_peak_stage,
                    },
                )?;
            }

            if scope_peak > 0 {
                crate::fallible::push(&mut scope_charges, (*scope, scope_peak))?;
            }

            crate::fallible::push(
                &mut scope_reports,
                ScopeReport {
                    scope: *scope,
                    physical_bytes: state.snapshot.physical_bytes(),
                    system_headroom_bytes: state.snapshot.system_headroom_bytes(),
                    admissible_bytes: admissible,
                    committed_bytes: committed_charged,
                    request_peak_bytes: scope_peak,
                    peak_stage: scope_peak_stage,
                    remaining_headroom_bytes: admissible.saturating_sub(needed),
                    tiers,
                },
            )?;
        }

        let (alternatives, host_backed_plan) = self.alternatives(request, &binding, &base)?;

        Ok(Evaluation {
            report: AdmissionReport {
                // Cloning a `Cow` clones a pointer for a literal stage and
                // the bytes for an owned one; the reserve is what makes the
                // vector's own growth fallible.
                stages: {
                    // `Cow::clone` copies an owned label's bytes infallibly,
                    // and the BF16 plan and chain builders make owned stages.
                    let mut stages = crate::fallible::with_capacity(request.stages().len())?;
                    for stage in request.stages() {
                        stages.push(crate::fallible::clone_label(stage)?);
                    }
                    stages
                },
                scopes: scope_reports,
                host_backed_plan,
            },
            charges,
            scope_charges,
            binding,
            alternatives,
        })
    }

    /// The changes this request makes capable of helping. Never applied.
    ///
    /// "Capable of helping" is the whole bar, and it is answered by recomputing
    /// rather than by attribution. For each declared scaling class the request
    /// uses, the peaks are computed again with every buffer of that class at
    /// zero bytes; the alternative is offered only when some failing constraint
    /// is strictly smaller in that counterfactual.
    ///
    /// Nothing short of recomputation is sound. Two stages can hold the same
    /// total, so a buffer that is live at the reported peak can be removed
    /// entirely without moving the maximum; and a derived reserve takes the
    /// largest buffer of its tier, so shrinking one member of a tie leaves the
    /// reserve exactly where it was. Both look like contributions and neither
    /// is one.
    ///
    /// The bar differs between the two kinds of alternative, deliberately. A
    /// caller chooses *how much* to lower context by, so a strict decrease means
    /// the knob is connected to the failure and is worth naming. Host-backed
    /// execution is a switch: it either clears the constraint or leaves the
    /// caller where they were, so it must be able to close the whole shortfall
    /// before it is offered.
    fn alternatives(
        &self,
        request: &PlanRequest,
        binding: &[BindingConstraint],
        base: &Peaks,
    ) -> Result<(Vec<LegalAlternative>, Option<HostBackedPlan>)> {
        let mut out = Vec::new();
        let host_backed_plan = bounded_host_backed_plan(request, base)?;
        if binding.is_empty() {
            return Ok((Vec::new(), host_backed_plan));
        }

        for (scaling, alternative) in [
            (Scaling::Context, LegalAlternative::LowerContext),
            (Scaling::Branches, LegalAlternative::FewerBranches),
        ] {
            if !request
                .buffers()
                .iter()
                .any(|b| b.scales_with == Some(scaling))
            {
                continue;
            }
            let without = peaks(request, &|b| b.scales_with == Some(scaling))?;
            let helps = binding.iter().any(|c| match c.kind {
                BindingKind::TierCap => {
                    without.tier_peak(c.scope, c.tier).0 < base.tier_peak(c.scope, c.tier).0
                }
                BindingKind::ScopeBudget => {
                    without.scope_peak(c.scope).0 < base.scope_peak(c.scope).0
                }
            });
            if helps {
                crate::fallible::push(&mut out, alternative)?;
            }
        }

        if binding.iter().any(|c| {
            matches!(
                c.tier,
                Tier::Device(DeviceTier::PackedResidentWeights)
                    | Tier::Device(DeviceTier::ExpertCache)
            )
        }) {
            crate::fallible::push(&mut out, LegalAlternative::OtherWeightPrecisionOrArtifact)?;
        }

        if request
            .scopes()?
            .iter()
            .filter(|s| s.kind() == ScopeKind::Device)
            .count()
            > 1
        {
            crate::fallible::push(&mut out, LegalAlternative::DifferentTopology)?;
        }

        // Moving work to the host is offered only when one **concrete**
        // relocation both fits on the host and clears the constraint. Bounding
        // the two sides separately does not do it: an upper bound on what the
        // host could take and an upper bound on what the device could shed can
        // each be satisfied by a different, incompatible move.
        if let Some(host) = self.scopes.get(&Scope::Host)
            && !binding.iter().any(|c| c.scope == Scope::Host)
        {
            let mut moved = crate::fallible::Map::new();
            for scope in request
                .scopes()?
                .into_iter()
                .filter(|s| s.kind() == ScopeKind::Device)
            {
                if !binding.iter().any(|c| c.scope == scope) {
                    continue;
                }
                if let Some(peaks) = self.relocate_to_host(request, base, host, scope)? {
                    moved.try_insert(scope, peaks)?;
                }
            }

            let clears = |c: &BindingConstraint| {
                let Some(after) = moved.get(&c.scope) else {
                    return false;
                };
                // The same relocation, checked on both sides: the host holds
                // what it received, and the device constraint is no longer over
                // its ceiling.
                let host_fits = self
                    .scope_committed(Scope::Host)
                    .saturating_add(after.scope_peak(Scope::Host).0)
                    <= host.snapshot.admissible_bytes()
                    && Tier::valid_in(ScopeKind::Host).all(|t| match host.snapshot.tier_cap(t) {
                        None => true,
                        Some(cap) => {
                            self.committed(Scope::Host, t)
                                .saturating_add(after.tier_peak(Scope::Host, t).0)
                                <= cap
                        }
                    });
                let device_clears = match c.kind {
                    BindingKind::TierCap => {
                        self.committed(c.scope, c.tier)
                            .saturating_add(after.tier_peak(c.scope, c.tier).0)
                            <= c.available_bytes
                    }
                    BindingKind::ScopeBudget => {
                        self.scope_committed(c.scope)
                            .saturating_add(after.scope_peak(c.scope).0)
                            <= c.available_bytes
                    }
                };
                host_fits && device_clears
            };

            if binding
                .iter()
                .any(|c| c.scope.kind() == ScopeKind::Device && clears(c))
            {
                crate::fallible::push(&mut out, LegalAlternative::HostBackedExecution)?;
            }
        }

        out.sort_unstable();
        out.dedup();
        Ok((out, host_backed_plan))
    }

    /// Build one concrete relocation of `scope`'s movable bytes onto the host,
    /// and return the peaks of the plan that results. `None` when nothing moves.
    ///
    /// The relocation is greedy and deterministic: each movable buffer, in
    /// declaration order, gives up as much as its destination tier and the host
    /// budget still have free **at every stage it is live**, so what comes back
    /// is a plan the host can hold rather than a bound on one. The caller then
    /// checks the same plan on both sides. Two upper bounds -- what the host
    /// could take, what the device could shed -- can each be met by a different
    /// move and together prove nothing.
    ///
    /// Two deliberate conservatisms, both of which can only withhold an
    /// alternative and never invent one:
    ///
    /// * A tier that a derived reserve is computed over does not move. Splitting
    ///   such a tier would shrink the reserve on the device while modelling no
    ///   equivalent on the host, and what a host-side cache reserves is a
    ///   residency question this task does not own (R03, R11; residency is M2).
    /// * The greedy takes one pass in declaration order. A different split might
    ///   clear a constraint this one leaves binding.
    fn relocate_to_host(
        &self,
        request: &PlanRequest,
        base: &Peaks,
        host: &ScopeState,
        scope: Scope,
    ) -> Result<Option<Peaks>> {
        let stage_count = request.stages().len();

        // What the host has spare, before anything moves.
        let scope_room = host
            .snapshot
            .admissible_bytes()
            .saturating_sub(self.scope_committed(Scope::Host))
            .saturating_sub(base.scope_peak(Scope::Host).0);
        let room_of = |destination: Tier| match host.snapshot.tier_cap(destination) {
            None => scope_room,
            Some(cap) => cap
                .saturating_sub(self.committed(Scope::Host, destination))
                .saturating_sub(base.tier_peak(Scope::Host, destination).0)
                .min(scope_room),
        };

        let mut free_in = crate::fallible::Map::new();
        let mut free_scope = crate::fallible::with_capacity(stage_count)?;
        free_scope.resize(stage_count, scope_room);
        // The label borrows `request`, which does not live as long as the
        // plan being built, so it is copied fallibly. The stages are `Cow`s
        // already: cloning one is a pointer for a literal.
        let mut relocated = {
            let mut stages = crate::fallible::with_capacity(request.stages().len())?;
            for stage in request.stages() {
                stages.push(crate::fallible::clone_label(stage)?);
            }
            PlanRequest::new(crate::fallible::string(request.label())?, stages)?
        };
        let mut any = false;

        for b in request.buffers() {
            let destination = if b.scope == scope
                && !request
                    .reserves()
                    .iter()
                    .any(|r| r.scope == b.scope && r.tier == b.tier)
            {
                host_destination(b.tier)
            } else {
                None
            };
            let Some(destination) = destination else {
                relocated.buffer(b.try_clone()?)?;
                continue;
            };
            if !free_in.contains_key(&destination) {
                let mut row = crate::fallible::with_capacity(stage_count)?;
                row.resize(stage_count, room_of(destination));
                free_in.try_insert(destination, row)?;
            }
            let free = free_in.get_mut(&destination).expect("inserted destination");
            let mut take = b.bytes;
            for stage in b.live.first..=b.live.last {
                take = take
                    .min(free[stage as usize])
                    .min(free_scope[stage as usize]);
            }
            if take == 0 {
                relocated.buffer(b.try_clone()?)?;
                continue;
            }
            for stage in b.live.first..=b.live.last {
                free[stage as usize] -= take;
                free_scope[stage as usize] -= take;
            }
            any = true;
            if take < b.bytes {
                let mut stays = b.try_clone()?;
                stays.bytes -= take;
                relocated.buffer(stays)?;
            }
            let mut goes = b.try_clone()?;
            goes.label = crate::fallible::text(format_args!("{} (host)", b.label))?.into();
            goes.scope = Scope::Host;
            goes.tier = destination;
            goes.bytes = take;
            goes.scales_with = None;
            relocated.buffer(goes)?;
        }

        for r in request.reserves() {
            relocated.reserve(r.try_clone()?)?;
        }

        if !any {
            return Ok(None);
        }
        Ok(Some(peaks(&relocated, &|_| false)?))
    }
}

/// Extract the bounded device-side envelope for an explicit streamed-page
/// request. The request itself remains the admission authority; this helper
/// only gives callers a typed size to report and to compare with a device cap.
fn bounded_host_backed_plan(request: &PlanRequest, base: &Peaks) -> Result<Option<HostBackedPlan>> {
    let staging = Tier::Device(DeviceTier::TransferStaging);
    let mut bytes = 0u64;
    for scope in request.scopes()? {
        if scope.kind() != ScopeKind::Device {
            continue;
        }
        bytes = bytes
            .checked_add(base.tier_peak(scope, staging).0)
            .ok_or(Error::Dim(moxie_types::DimError::Overflow))?;
    }
    let requested_blocks = request.declared_host_backed_blocks().unwrap_or(1);
    if requested_blocks > HostBackedPlan::MAX_STAGED_BLOCKS {
        return Err(Error::InvalidRequest {
            field: "host_backed_blocks",
            detail: crate::fallible::text(format_args!(
                "{requested_blocks} staged block(s) exceeds the admitted maximum of {}",
                HostBackedPlan::MAX_STAGED_BLOCKS
            ))?,
        });
    }
    if request.declared_host_backed_blocks().is_some() && bytes == 0 {
        return Err(Error::InvalidRequest {
            field: "host_backed_blocks",
            detail: crate::fallible::string(
                "a bounded host-backed plan must declare a TransferStaging buffer",
            )?,
        });
    }
    Ok(HostBackedPlan::with_max_staged_blocks(
        bytes,
        requested_blocks,
    ))
}

/// Where a device tier's bytes would go if its work moved to the host, or
/// `None` when they cannot move at all.
///
/// Sequence state spills; the tensors and scratch of execution need CPU-side
/// workspace to be executed there. Two device tiers have **no** host
/// destination: `SafetyHeadroom` is deliberate slack in that device's memory and
/// `AllocatorFragmentation` is bytes its allocator cannot hand out. Neither is
/// work or state, so neither can be relocated -- offering a host fallback for
/// them would be advice that cannot be followed.
///
/// The mapping is otherwise deliberately coarse: a host weight arena of its own
/// (R11) is a residency question, and residency is M2. When that arrives, this
/// is the function that gains a row.
fn host_destination(tier: Tier) -> Option<Tier> {
    match tier {
        Tier::Device(
            DeviceTier::KvStatePages
            | DeviceTier::RecurrentState
            | DeviceTier::SpeculativeTargetState
            | DeviceTier::SpeculativeDraftState
            | DeviceTier::EntropyBranches,
        ) => Some(Tier::Host(HostTier::StateSpill)),
        Tier::Device(DeviceTier::SafetyHeadroom | DeviceTier::AllocatorFragmentation) => None,
        Tier::Device(_) => Some(Tier::Host(HostTier::CpuWorkspace)),
        // A host tier is already on the host; there is nowhere to move it to.
        Tier::Host(_) => None,
    }
}

/// Per-scope and per-tier peaks over a request's declared stages.
#[derive(Debug)]
struct Peaks {
    /// Bytes live per stage, per scope and tier.
    live: crate::fallible::Map<(Scope, Tier), Vec<u64>>,
    /// Declared virtual extent per stage. Reported, charged to nothing.
    virtual_live: crate::fallible::Map<(Scope, Tier), Vec<u64>>,
    tier: crate::fallible::Map<(Scope, Tier), (u64, u32)>,
    scope: crate::fallible::Map<Scope, (u64, u32)>,
}

impl Peaks {
    fn tier_peak(&self, scope: Scope, tier: Tier) -> (u64, u32) {
        self.tier.get(&(scope, tier)).copied().unwrap_or((0, 0))
    }

    fn scope_peak(&self, scope: Scope) -> (u64, u32) {
        self.scope.get(&scope).copied().unwrap_or((0, 0))
    }
}

/// Compute a request's peaks, with every buffer `zero` selects contributing zero
/// bytes.
///
/// That predicate is what makes an alternative answerable: the counterfactual is
/// the same arithmetic on the same declaration, so a derived reserve is
/// re-derived and a tie between stages is re-resolved rather than assumed away.
/// A zeroed buffer still exists, so a reserve's arity check is unaffected and
/// the counterfactual cannot fail where the base pass succeeded.
fn peaks(request: &PlanRequest, zero: &dyn Fn(&BufferRequest) -> bool) -> Result<Peaks> {
    let stage_count = request.stages().len();
    let overflow = || Error::Dim(moxie_types::DimError::Overflow);
    let bytes_of = |b: &BufferRequest| if zero(b) { 0 } else { b.bytes };

    // **Every row and every entry is reserved before it exists.** `vec!` and
    // `BTreeMap::insert` both abort when the allocator refuses, and this runs
    // inside admission, whose whole job is to answer "does this fit" without
    // dying.
    let zero_row = |count: usize| -> Result<Vec<u64>> {
        let mut row: Vec<u64> = crate::fallible::with_capacity(count)?;
        row.resize(count, 0);
        Ok(row)
    };
    let mut live: crate::fallible::Map<(Scope, Tier), Vec<u64>> = crate::fallible::Map::new();
    let mut virtual_live: crate::fallible::Map<(Scope, Tier), Vec<u64>> =
        crate::fallible::Map::new();
    for b in request.buffers() {
        if !live.contains_key(&(b.scope, b.tier)) {
            live.try_insert((b.scope, b.tier), zero_row(stage_count)?)?;
        }
        let row = live.get_mut(&(b.scope, b.tier)).expect("just inserted");
        for stage in b.live.first..=b.live.last {
            let slot = &mut row[stage as usize];
            *slot = slot.checked_add(bytes_of(b)).ok_or_else(overflow)?;
        }
        if let Some(extent) = b.virtual_bytes {
            if !virtual_live.contains_key(&(b.scope, b.tier)) {
                virtual_live.try_insert((b.scope, b.tier), zero_row(stage_count)?)?;
            }
            let row = virtual_live
                .get_mut(&(b.scope, b.tier))
                .expect("just inserted");
            for stage in b.live.first..=b.live.last {
                let slot = &mut row[stage as usize];
                *slot = slot.checked_add(extent).ok_or_else(overflow)?;
            }
        }
    }

    for r in request.reserves() {
        let mut sizes: Vec<u64> = crate::fallible::with_capacity(request.buffers().len())?;
        for b in request.buffers() {
            if b.scope == r.scope && b.tier == r.tier {
                sizes.push(bytes_of(b));
            }
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let wanted = match r.rule {
            ReserveRule::LargestBufferOfTier => 1,
            ReserveRule::NLargestBuffersOfTier(n) => n as usize,
        };
        if sizes.len() < wanted {
            return Err(Error::InvalidRequest {
                field: "rule",
                detail: crate::fallible::text(format_args!(
                    "{}: {:?} needs {wanted} buffer(s) in {} of {}, and the request declares {}",
                    r.label,
                    r.rule,
                    r.tier.name(),
                    r.scope,
                    sizes.len()
                ))?,
            });
        }
        let mut bytes: u64 = 0;
        for size in &sizes[..wanted] {
            bytes = bytes.checked_add(*size).ok_or_else(overflow)?;
        }
        if !live.contains_key(&(r.scope, r.tier)) {
            live.try_insert((r.scope, r.tier), zero_row(stage_count)?)?;
        }
        let row = live.get_mut(&(r.scope, r.tier)).expect("just inserted");
        for stage in r.live.first..=r.live.last {
            let slot = &mut row[stage as usize];
            *slot = slot.checked_add(bytes).ok_or_else(overflow)?;
        }
    }

    let mut tier: crate::fallible::Map<(Scope, Tier), (u64, u32)> = crate::fallible::Map::new();
    for (key, row) in &live {
        let (mut peak, mut at) = (0u64, 0u32);
        for (stage, bytes) in row.iter().enumerate() {
            if *bytes > peak {
                peak = *bytes;
                at = stage as u32;
            }
        }
        tier.try_insert(*key, (peak, at))?;
    }

    // The scope figure is the stage-wise total, not the sum of the per-tier
    // peaks, which would reserve buffers that are never live together.
    let mut scope: crate::fallible::Map<Scope, (u64, u32)> = crate::fallible::Map::new();
    let mut present: Vec<Scope> = crate::fallible::with_capacity(live.len())?;
    for ((s, _), _) in &live {
        if !present.contains(s) {
            present.push(*s);
        }
    }
    for s in present {
        let (mut peak, mut at) = (0u64, 0u32);
        for stage in 0..stage_count {
            let mut total: u64 = 0;
            for ((scope_of, _), row) in &live {
                if *scope_of == s {
                    total = total.checked_add(row[stage]).ok_or_else(overflow)?;
                }
            }
            if total > peak {
                peak = total;
                at = stage as u32;
            }
        }
        scope.try_insert(s, (peak, at))?;
    }

    Ok(Peaks {
        live,
        virtual_live,
        tier,
        scope,
    })
}

/// A tier's index in `Tier::ALL`, for deterministic tie-breaking.
fn position(tier: Tier) -> usize {
    Tier::ALL
        .iter()
        .position(|t| *t == tier)
        .expect("every tier is in ALL")
}
