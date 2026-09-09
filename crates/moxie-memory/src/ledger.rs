//! The one authority that admits and charges bytes.
//!
//! R02 is the failure this repairs: legacy's `ResidencyManager` modelled
//! accesses and bytes for a simulator while the model runtimes allocated for
//! themselves, so the thing that knew the budget was not the thing that spent
//! it. Here there is one ledger, admission is atomic, and a reservation is
//! released by name rather than by scope exit.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use moxie_types::{DeviceTier, Error, HostTier, Result, Scope, ScopeKind, Tier};

use crate::report::{
    AdmissionReport, BindingConstraint, BindingKind, LegalAlternative, Rejection, ScopeReport,
    TierReport,
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
/// let second = r.clone(); // a second authority over the same bytes
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
    Rejected(Box<Rejection>),
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
            AdmitError::Rejected(r) => (*r).into(),
        }
    }
}

#[derive(Debug, Clone)]
struct ScopeState {
    snapshot: CapacitySnapshot,
    /// Per-tier commitments, against the per-tier caps.
    committed: BTreeMap<Tier, u64>,
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
    outstanding: BTreeMap<ReservationId, OutstandingRecord>,
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
                    committed: BTreeMap::new(),
                    committed_scope: 0,
                },
            );
        }
        Ok(Ledger {
            id: LedgerId::next(),
            scopes,
            outstanding: BTreeMap::new(),
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
            return Err(AdmitError::Rejected(Box::new(Rejection {
                report: evaluation.report,
                binding: evaluation.binding,
                shortfall_bytes: shortfall,
                alternatives: evaluation.alternatives,
            })));
        }

        // Nothing below can fail: every sum was checked during evaluation, and
        // adding a peak that already fits under a ceiling cannot overflow it.
        for (scope, tier, bytes) in &evaluation.charges {
            let state = self
                .scopes
                .get_mut(scope)
                .expect("evaluation rejected unknown scopes");
            *state.committed.entry(*tier).or_insert(0) += *bytes;
        }
        for (scope, bytes) in &evaluation.scope_charges {
            let state = self
                .scopes
                .get_mut(scope)
                .expect("evaluation rejected unknown scopes");
            state.committed_scope += *bytes;
        }
        let id = ReservationId::next();
        self.outstanding.insert(
            id,
            OutstandingRecord {
                label: request.label().to_string(),
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
            let error = Error::InvalidRequest {
                field: "reservation",
                detail: format!(
                    "reservation {} belongs to ledger {}, not {}",
                    reservation.id.get(),
                    reservation.ledger.get(),
                    self.id.get()
                ),
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
        for scope in request.scopes() {
            if !self.scopes.contains_key(&scope) {
                return Err(Error::InvalidRequest {
                    field: "scope",
                    detail: format!("{scope} has no capacity snapshot in this ledger"),
                });
            }
        }

        let base = peaks(request, &|_| false)?;

        let mut charges: Vec<(Scope, Tier, u64)> = base
            .tier
            .iter()
            .filter(|(_, (bytes, _))| *bytes > 0)
            .map(|((scope, tier), (bytes, _))| (*scope, *tier, *bytes))
            .collect();
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
                    binding.push(BindingConstraint {
                        scope: *scope,
                        tier,
                        kind: BindingKind::TierCap,
                        needed_bytes: tier_needed,
                        available_bytes: cap,
                        peak_stage,
                    });
                }
                tiers.push(TierReport {
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
                });
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
                binding.push(BindingConstraint {
                    scope: *scope,
                    tier: largest,
                    kind: BindingKind::ScopeBudget,
                    needed_bytes: needed,
                    available_bytes: admissible,
                    peak_stage: scope_peak_stage,
                });
            }

            if scope_peak > 0 {
                scope_charges.push((*scope, scope_peak));
            }

            scope_reports.push(ScopeReport {
                scope: *scope,
                physical_bytes: state.snapshot.physical_bytes(),
                system_headroom_bytes: state.snapshot.system_headroom_bytes(),
                admissible_bytes: admissible,
                committed_bytes: committed_charged,
                request_peak_bytes: scope_peak,
                peak_stage: scope_peak_stage,
                remaining_headroom_bytes: admissible.saturating_sub(needed),
                tiers,
            });
        }

        let alternatives = self.alternatives(request, &binding, &base)?;

        Ok(Evaluation {
            report: AdmissionReport {
                stages: request.stages().to_vec(),
                scopes: scope_reports,
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
    ) -> Result<Vec<LegalAlternative>> {
        let mut out: BTreeSet<LegalAlternative> = BTreeSet::new();
        if binding.is_empty() {
            return Ok(Vec::new());
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
                out.insert(alternative);
            }
        }

        if binding.iter().any(|c| {
            matches!(
                c.tier,
                Tier::Device(DeviceTier::PackedResidentWeights)
                    | Tier::Device(DeviceTier::ExpertCache)
            )
        }) {
            out.insert(LegalAlternative::OtherWeightPrecisionOrArtifact);
        }

        if request
            .scopes()
            .iter()
            .filter(|s| s.kind() == ScopeKind::Device)
            .count()
            > 1
        {
            out.insert(LegalAlternative::DifferentTopology);
        }

        // Moving work to the host has to clear the constraint on three counts,
        // and each one alone has been wrong: the host must have room after
        // everything this same request already asks of it; the tiers that would
        // receive the bytes must be able to take them; and the relocation must
        // actually lower the failing ceiling, which is a question about the
        // whole timeline rather than about the stage that happened to be
        // reported.
        if let Some(host) = self.scopes.get(&Scope::Host)
            && !binding.iter().any(|c| c.scope == Scope::Host)
        {
            let available = host
                .snapshot
                .admissible_bytes()
                .saturating_sub(self.scope_committed(Scope::Host))
                .saturating_sub(base.scope_peak(Scope::Host).0);

            // One counterfactual per failing scope: everything in it that has a
            // host destination, moved.
            let mut relocated: BTreeMap<Scope, Peaks> = BTreeMap::new();
            for scope in binding
                .iter()
                .filter(|c| c.scope.kind() == ScopeKind::Device)
                .map(|c| c.scope)
                .collect::<BTreeSet<Scope>>()
            {
                relocated.insert(
                    scope,
                    peaks(request, &|b| {
                        b.scope == scope && host_destination(b.tier).is_some()
                    })?,
                );
            }

            let clears = |c: &BindingConstraint| {
                let need = c.shortfall_bytes();
                let Some(without) = relocated.get(&c.scope) else {
                    return false;
                };
                let reduction = match c.kind {
                    BindingKind::TierCap => base
                        .tier_peak(c.scope, c.tier)
                        .0
                        .saturating_sub(without.tier_peak(c.scope, c.tier).0),
                    BindingKind::ScopeBudget => base
                        .scope_peak(c.scope)
                        .0
                        .saturating_sub(without.scope_peak(c.scope).0),
                };
                available >= need
                    && self.host_absorbable(base, host, c) >= need
                    && reduction >= need
            };

            if binding
                .iter()
                .any(|c| c.scope.kind() == ScopeKind::Device && clears(c))
            {
                out.insert(LegalAlternative::HostBackedExecution);
            }
        }

        Ok(out.into_iter().collect())
    }

    /// How many of a failing constraint's bytes could actually be held on the
    /// host, given what can move and where it would go.
    ///
    /// Two things are deliberately not consulted. `BindingConstraint::tier` on a
    /// scope-budget failure is a **diagnostic label** -- the largest contributor
    /// at the peak stage -- so eligibility is decided over every contributor
    /// instead: the largest one may be immovable while a smaller one covers the
    /// whole shortfall. And a tier with no host destination contributes nothing,
    /// however large it is: device safety headroom and allocator fragmentation
    /// are properties of that device's memory, not work or state with somewhere
    /// else to be.
    fn host_absorbable(&self, base: &Peaks, host: &ScopeState, c: &BindingConstraint) -> u64 {
        let mut movable: BTreeMap<Tier, u64> = BTreeMap::new();
        for ((scope, tier), row) in &base.live {
            if *scope != c.scope {
                continue;
            }
            if c.kind == BindingKind::TierCap && *tier != c.tier {
                continue;
            }
            let bytes = row[c.peak_stage as usize];
            if bytes == 0 {
                continue;
            }
            if let Some(destination) = host_destination(*tier) {
                *movable.entry(destination).or_insert(0) += bytes;
            }
        }

        let mut absorbable: u64 = 0;
        for (destination, bytes) in movable {
            let room = match host.snapshot.tier_cap(destination) {
                // An absent cap means the tier is bounded by the scope budget,
                // which the caller checks separately.
                None => bytes,
                Some(cap) => {
                    let used = self.committed(Scope::Host, destination)
                        + base.tier_peak(Scope::Host, destination).0;
                    bytes.min(cap.saturating_sub(used))
                }
            };
            absorbable = absorbable.saturating_add(room);
        }
        absorbable
    }
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
    live: BTreeMap<(Scope, Tier), Vec<u64>>,
    /// Declared virtual extent per stage. Reported, charged to nothing.
    virtual_live: BTreeMap<(Scope, Tier), Vec<u64>>,
    tier: BTreeMap<(Scope, Tier), (u64, u32)>,
    scope: BTreeMap<Scope, (u64, u32)>,
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

    let mut live: BTreeMap<(Scope, Tier), Vec<u64>> = BTreeMap::new();
    let mut virtual_live: BTreeMap<(Scope, Tier), Vec<u64>> = BTreeMap::new();
    for b in request.buffers() {
        let row = live
            .entry((b.scope, b.tier))
            .or_insert_with(|| vec![0; stage_count]);
        for stage in b.live.first..=b.live.last {
            let slot = &mut row[stage as usize];
            *slot = slot.checked_add(bytes_of(b)).ok_or_else(overflow)?;
        }
        if let Some(extent) = b.virtual_bytes {
            let row = virtual_live
                .entry((b.scope, b.tier))
                .or_insert_with(|| vec![0; stage_count]);
            for stage in b.live.first..=b.live.last {
                let slot = &mut row[stage as usize];
                *slot = slot.checked_add(extent).ok_or_else(overflow)?;
            }
        }
    }

    for r in request.reserves() {
        let mut sizes: Vec<u64> = request
            .buffers()
            .iter()
            .filter(|b| b.scope == r.scope && b.tier == r.tier)
            .map(bytes_of)
            .collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let wanted = match r.rule {
            ReserveRule::LargestBufferOfTier => 1,
            ReserveRule::NLargestBuffersOfTier(n) => n as usize,
        };
        if sizes.len() < wanted {
            return Err(Error::InvalidRequest {
                field: "rule",
                detail: format!(
                    "{}: {:?} needs {wanted} buffer(s) in {} of {}, and the request declares {}",
                    r.label,
                    r.rule,
                    r.tier.name(),
                    r.scope,
                    sizes.len()
                ),
            });
        }
        let mut bytes: u64 = 0;
        for size in &sizes[..wanted] {
            bytes = bytes.checked_add(*size).ok_or_else(overflow)?;
        }
        let row = live
            .entry((r.scope, r.tier))
            .or_insert_with(|| vec![0; stage_count]);
        for stage in r.live.first..=r.live.last {
            let slot = &mut row[stage as usize];
            *slot = slot.checked_add(bytes).ok_or_else(overflow)?;
        }
    }

    let mut tier: BTreeMap<(Scope, Tier), (u64, u32)> = BTreeMap::new();
    for (key, row) in &live {
        let (mut peak, mut at) = (0u64, 0u32);
        for (stage, bytes) in row.iter().enumerate() {
            if *bytes > peak {
                peak = *bytes;
                at = stage as u32;
            }
        }
        tier.insert(*key, (peak, at));
    }

    // The scope figure is the stage-wise total, not the sum of the per-tier
    // peaks, which would reserve buffers that are never live together.
    let mut scope: BTreeMap<Scope, (u64, u32)> = BTreeMap::new();
    let present: BTreeSet<Scope> = live.keys().map(|(s, _)| *s).collect();
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
        scope.insert(s, (peak, at));
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
