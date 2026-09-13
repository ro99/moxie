//! The byte and cost trace, and what it means for one to reconcile.
//!
//! Roadmap M2's exit asks that a real out-of-device-memory working set "produces
//! byte/cost traces reconciled with the resource ledger". This module is the
//! second half of that sentence. R02 is why it is an equality rather than a
//! report: legacy's `ResidencyManager` "modelled accesses and bytes for a
//! simulator while the model runtimes allocated for themselves", so the thing
//! that knew the budget was not the thing that spent it -- and a trace that
//! cannot be checked against what was actually charged is that same failure with
//! better formatting.
//!
//! **Nothing here counts a byte.** Every number is read from one of three
//! owners:
//!
//! * `moxie_memory::ResidencyAuthority` -- what was requested, admitted, moved
//!   and evicted, per scope, in [`ByteFlow`].
//! * `moxie_memory::Ledger` -- what is charged, per scope and tier.
//! * `moxie_plan::expert::ExpertPlan` -- what was predicted, and whether the
//!   prediction is an equality.
//! * the run's own [`GroupedStats`] -- counts of actions it took, never bytes.
//!
//! A [`LayerTrace`] is the **difference between two snapshots** of those, taken
//! around one layer. That arithmetic is attributable to one layer only because
//! one active interactive generation is a product gate; a second concurrent
//! consumer of the same authority would make it meaningless, and nothing here
//! can detect that, which is why it is written down.
//!
//! **This is not a benchmark.** There is no duration anywhere in this file and
//! there is no field for one. Document 07 keeps profiling mode and benchmark
//! mode apart; a trace is neither, and a debug build's wall clock is not a cost.

use moxie_memory::{ByteFlow, Ledger, ResidencyAuthority, ScopeAccount};
use moxie_plan::expert::{EnvelopePrediction, Exactness, ExpertPlan};
use moxie_types::{Error, HostTier, Scope, Tier};

use crate::grouped::{GroupedRun, GroupedStats};

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

/// Reserve exactly `len` more, or fail.
///
/// Every collection this module builds goes through here. Task 0019's rule, for
/// the fourth time in this workspace: an allocation failure inside a generation
/// step must be a typed error the transaction can roll back, not a panic that
/// takes the rollback, the lease release and the next generation with it. A
/// review injected one failure immediately before a snapshot and got **SIGABRT**.
fn reserve<T>(out: &mut Vec<T>, len: usize) -> Result<(), Error> {
    out.try_reserve_exact(len)
        .map_err(|_| Error::CapacityExceeded {
            tier: Some(Tier::Host(HostTier::Pageable)),
            requested_bytes: (len * core::mem::size_of::<T>()) as u64,
            available_bytes: 0,
        })
}

/// Find or append a scope's entry in a sorted-by-insertion association list.
///
/// A `Vec`, not a `BTreeMap`: `BTreeMap` has no fallible insert, so a map here
/// would abort on an allocation failure however carefully the rest of this
/// module reserved. The lists are one entry per scope -- four on this machine --
/// so the linear search is not the interesting cost.
fn entry_for<V: Default>(list: &mut Vec<(Scope, V)>, scope: Scope) -> Result<&mut V, Error> {
    if let Some(index) = list.iter().position(|(s, _)| *s == scope) {
        return Ok(&mut list[index].1);
    }
    reserve(list, 1)?;
    list.push((scope, V::default()));
    let last = list.len() - 1;
    Ok(&mut list[last].1)
}

/// The trace schema's version.
///
/// Document 07 requires evidence to have a schema. A record carries this so a
/// later reader can tell which fields it was written with, and a test pins it so
/// the number cannot drift silently.
pub const TRACE_SCHEMA_VERSION: u32 = 1;

/// Fields of document 07's result `counters` block that nothing in this task
/// measures.
///
/// Named, rather than left at zero. `d2h_bytes: 0` and `d2h_bytes: unmeasured`
/// are different claims, and a record that reports the first when it means the
/// second is the kind of quiet untruth AGENTS.md keeps finding in prose.
///
/// The counters that block asks for and that **are** measured are here or in
/// their owner: expert rows and unique experts in [`LayerTrace`], the cache's
/// hits, misses, prefetch waste and eviction *counts* in
/// `moxie_memory::ResidencyStats`, and this trace carries their **bytes** per
/// scope. Every entry below is a quantity nothing in this workspace produces --
/// four of them are durations, and this trace has no duration field at all.
pub const UNMEASURED: &[&str] = &[
    "disk_read_ms",
    "host_memcpy_ms",
    "h2d_ms",
    "d2h_bytes",
    "p2p_bytes",
    "collective_bytes",
    "sync_wait_ms",
    "attention_state_bytes_scanned",
    "speculative_wasted_tokens",
    "accepted_prefix_histogram",
];

/// One scope's position at a layer boundary: what changed, and where it stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeDelta {
    pub scope: Scope,
    pub tier: Tier,
    pub cap_bytes: u64,
    /// This layer's share of the scope's flow.
    pub flow: ByteFlow,
    /// Levels at the end of the layer, which are not differences.
    pub resident_bytes: u64,
    pub quarantined_bytes: u64,
    pub unreported_bytes: u64,
    pub peak_resident_bytes: u64,
}

/// What one layer asked for, moved, cost and was charged.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerTrace {
    pub layer: u32,
    pub predicted: EnvelopePrediction,
    /// The plan's chunk lengths, by role: `[gate_up, down]`. The trace's only
    /// multiplier, and it comes from the plan rather than from a constant.
    pub chunk_bytes: [u64; 2],
    pub groups: u64,
    pub rows: u64,
    pub top_k: u64,
    /// Kernel launches one device group submits: the number of symbols in the
    /// descriptor this plan selected. Zero when the plan selected none.
    ///
    /// A review found the launch count being one per completed group while the
    /// device path submits one per symbol -- a projection and a reduction. The
    /// count of symbols is the plan's own, so the equality compares what the
    /// lane reported against what the descriptor says it takes.
    pub kernels_per_group: u64,
    pub scopes: Vec<ScopeDelta>,
    pub cost: GroupedStats,
    /// Whether this layer ran every group. A layer that failed or was cancelled
    /// still produces a trace -- "no trace" and "a trace showing the failure"
    /// are different outcomes, and the first is not acceptable for a step that
    /// has to account for its bytes either way.
    pub completed: bool,
    /// Leases this run is withholding: their bytes may still be in flight, so
    /// they are neither released nor lost (R07).
    pub withheld_leases: u64,
    /// What the ledger held while this layer's run was live, per scope and tier.
    pub ledger: Vec<(Scope, Tier, u64)>,
    /// The ledger this trace read, and the one the run was admitted against.
    ///
    /// A review handed `close` a **different, empty** ledger: observed and
    /// accounted charges both came back empty and the layer reconciled with no
    /// recorded charges at all. Comparing two numbers is worth nothing if they
    /// can both be read from the wrong place.
    pub ledger_id: u64,
    pub run_ledger_id: u64,
    /// Reservations this trace could name, and how many of them the ledger
    /// actually holds. A named reservation the ledger has never heard of is the
    /// other half of the same finding.
    pub reservations_named: u64,
    pub reservations_found: u64,
    /// What the authority's caps and this plan's envelope together account for,
    /// in the same shape. The `ledger-charges-what-is-held` equality compares
    /// these two lists and nothing else.
    pub accounted: Vec<(Scope, Tier, u64)>,
}

/// A whole working set's trace: every layer, and the totals they sum to.
#[derive(Debug, Clone, PartialEq)]
pub struct StepTrace {
    pub schema_version: u32,
    /// The artifact these bytes came from, by identity rather than by path.
    pub artifact: String,
    /// The case's name, so two configurations of one working set are not
    /// confused for each other.
    pub case: String,
    pub layers: Vec<LayerTrace>,
    /// Field-by-field sum of the layers' flows, per scope.
    pub totals: Vec<(Scope, ByteFlow)>,
    /// Reservations still outstanding when the step ended. Zero, or the step
    /// leaked.
    pub outstanding_reservations: usize,
    /// Each scope's account after everything was retired.
    pub final_accounts: Vec<ScopeAccount>,
    /// What the **authority** moved over the whole step, from the snapshot taken
    /// before the first layer to the end. The outer boundary the layers are
    /// checked against: a byte that entered a cache and belongs to no layer
    /// shows up here and nowhere else.
    pub whole: Vec<(Scope, ByteFlow)>,
    pub unmeasured: &'static [&'static str],
}

/// A check that did not hold, and both sides of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discrepancy {
    /// The equality's name. What a failure is reported as, so "the trace does
    /// not reconcile" is never the whole message.
    pub check: &'static str,
    pub layer: Option<u32>,
    pub scope: Option<Scope>,
    pub left: u64,
    pub right: u64,
    /// What the equality means, in prose.
    ///
    /// **`&'static str`, so that constructing a discrepancy allocates nothing.**
    /// Three review findings walked this field down to here. It was a `String`
    /// built by `format!`, which aborted the process when the thing being
    /// reported *was* an allocation failure; then a `Cow` whose ordinary arm
    /// still formatted, which aborted when an ordinary mismatch happened to
    /// coincide with memory pressure -- **114 bytes**, reported by a review, on a
    /// ledger-identity mismatch.
    ///
    /// The numbers a caller acts on are the fields beside this one, and
    /// [`Discrepancy`]'s `Display` composes them: rendering writes into the
    /// caller's formatter, so whether *that* allocates is the caller's decision
    /// and not a step's. Anything richer -- which tiers, which flows, which
    /// counts -- is in the `StepTrace` the caller already holds.
    pub detail: &'static str,
}

impl core::fmt::Display for Discrepancy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.check)?;
        if let Some(layer) = self.layer {
            write!(f, " at layer {layer}")?;
        }
        if let Some(scope) = self.scope {
            write!(f, " in {scope}")?;
        }
        write!(
            f,
            ": {} against {} -- {}",
            self.left, self.right, self.detail
        )
    }
}

impl std::error::Error for Discrepancy {}

/// What a reconciliation checked, so a pass is a number rather than a silence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Reconciled {
    pub layers: u32,
    pub equalities_checked: u32,
    /// Of those, how many were the declared-lower-bound form rather than an
    /// equality. A run in which this is zero has never exercised the bound, and
    /// a run in which it equals the total has never exercised the equality.
    pub bounds_checked: u32,
    /// Layers whose prediction was not compared, because they did not finish.
    /// Reported rather than silently absent: a step of nothing but failed layers
    /// would otherwise reconcile without checking a single prediction.
    pub predictions_skipped: u32,
}

// ---------------------------------------------------------------------------
// Taking the snapshots
// ---------------------------------------------------------------------------

/// The authority's position before a **step** runs.
///
/// The trace's outer boundary. Without one, a step's totals are a sum of the
/// records it was handed and reconciliation re-sums the same records: a review
/// executed three layers, dropped the middle one, and the trace reconciled 35
/// checks while 3,840 B of reads went unmentioned. The step's own delta is the
/// only thing that can say a layer is missing.
/// **Not `Clone`**, deliberately: cloning it would allocate infallibly, and the
/// one thing this type exists to be is safe to take inside a generation step.
#[derive(Debug, PartialEq)]
pub struct StepSnapshot {
    flows: Vec<(Scope, ByteFlow)>,
}

impl StepSnapshot {
    /// Read every scope's flow. Call this before the step's first layer.
    ///
    /// Fallible, and the storage is reserved before a byte of it is written: an
    /// allocation failure here is a typed error, not a process abort.
    pub fn take(authority: &ResidencyAuthority) -> Result<Self, Error> {
        let mut flows = Vec::new();
        reserve(&mut flows, authority.scope_count())?;
        let mut overflow = None;
        authority.for_each_account(|a| {
            if flows.len() == flows.capacity() {
                // Reserved from `scope_count`, so this cannot happen unless the
                // authority gained a scope between the two calls. Recorded
                // rather than pushed: a push here would allocate.
                overflow = Some(a.scope);
                return;
            }
            flows.push((a.scope, a.flow));
        });
        if let Some(scope) = overflow {
            return Err(invalid(
                "scopes",
                format!("{scope} appeared while its snapshot was being taken"),
            ));
        }
        Ok(StepSnapshot { flows })
    }

    fn flow_of(&self, scope: Scope) -> ByteFlow {
        self.flows
            .iter()
            .find(|(s, _)| *s == scope)
            .map_or_else(ByteFlow::default, |(_, f)| *f)
    }
}

/// The authority's and the ledger's position before a layer runs.
///
/// Held by value, so nothing here borrows the owners while the layer executes.
/// **Not `Clone`**, for the same reason [`StepSnapshot`] is not.
#[derive(Debug, PartialEq)]
pub struct LayerSnapshot {
    layer: u32,
    flows: Vec<(Scope, ByteFlow)>,
}

impl LayerSnapshot {
    /// Read every scope's flow. Call this immediately before the layer runs.
    ///
    /// Fallible, for the reason every allocation on this path is: a review
    /// injected one failure here and got `SIGABRT` where a generation step needs
    /// a typed error it can roll back.
    pub fn take(layer: u32, authority: &ResidencyAuthority) -> Result<Self, Error> {
        let step = StepSnapshot::take(authority)?;
        Ok(LayerSnapshot {
            layer,
            flows: step.flows,
        })
    }

    fn flow_of(&self, scope: Scope) -> ByteFlow {
        self.flows
            .iter()
            .find(|(s, _)| *s == scope)
            .map_or_else(ByteFlow::default, |(_, f)| *f)
    }

    /// Close the layer: difference the flows, read the levels, and record what
    /// the plan predicted, what the run did, and what the ledger holds.
    ///
    /// The ledger is read **while the run's reservation is still live**, which is
    /// the only moment at which "what is charged" and "what this layer needs"
    /// are the same question.
    pub fn close(
        self,
        run: &GroupedRun<'_>,
        authority: &ResidencyAuthority,
        ledger: &Ledger,
    ) -> Result<LayerTrace, Error> {
        let plan = run.plan();
        let cost = run.stats();
        let mut scopes = Vec::new();
        reserve(&mut scopes, authority.scope_count())?;
        let mut overflow = None;
        authority.for_each_account(|account| {
            if scopes.len() == scopes.capacity() {
                overflow = Some(account.scope);
                return;
            }
            let before = self.flow_of(account.scope);
            scopes.push(ScopeDelta {
                scope: account.scope,
                tier: account.tier,
                cap_bytes: account.cap_bytes,
                flow: account.flow.since(&before),
                resident_bytes: account.resident_bytes,
                quarantined_bytes: account.quarantined_bytes,
                unreported_bytes: account.unreported_bytes,
                peak_resident_bytes: account.peak_resident_bytes,
            });
        });
        if let Some(scope) = overflow {
            return Err(invalid(
                "scopes",
                format!("{scope} appeared while this layer was being closed"),
            ));
        }
        let shape = plan.shape();
        // Both charge lists are built in the same order -- scope order, then the
        // tier order `Tier::ALL` fixes -- so they are comparable row by row. Two
        // orderings of the same numbers would make a mismatch report the wrong
        // pair, which is a diagnosis problem rather than a correctness one, and
        // this repository has spent two review rounds on findings of exactly
        // that shape.
        let mut observed: Vec<(Scope, Tier, u64)> = Vec::new();
        let mut scope_list = Vec::new();
        reserve(&mut scope_list, ledger.scope_count())?;
        for scope in ledger.scopes() {
            if scope_list.len() == scope_list.capacity() {
                // Reserved from `scope_count`; growing here would be an
                // unreserved allocation on a step path.
                return Err(invalid(
                    "scopes",
                    "the ledger gained a scope while this layer was being closed".into(),
                ));
            }
            scope_list.push(scope);
        }
        for scope in &scope_list {
            for tier in Tier::valid_in(scope.kind()) {
                let committed = ledger.committed(*scope, tier);
                if committed != 0 {
                    reserve(&mut observed, 1)?;
                    observed.push((*scope, tier, committed));
                }
            }
        }
        let (accounted, named, found) = accounted_charges(authority, run, ledger)?;
        Ok(LayerTrace {
            layer: self.layer,
            predicted: plan.envelope().predicted,
            chunk_bytes: role_chunk_bytes(plan),
            groups: plan.groups().len() as u64,
            kernels_per_group: plan
                .kernel()
                .map_or(0, |descriptor| descriptor.symbols.len() as u64),
            rows: plan.rows(),
            top_k: shape.top_k,
            accounted,
            ledger_id: ledger.id().get(),
            run_ledger_id: run.ledger().get(),
            reservations_named: named,
            reservations_found: found,
            scopes,
            cost,
            completed: run.failure().is_none() && !run.is_cancelled(),
            withheld_leases: run.withheld_leases() as u64,
            ledger: observed,
        })
    }
}

/// The two chunk lengths a plan's roles have, taken from the plan's own shape.
///
/// Deliberately computed here rather than read from the run's roles. It is a
/// second statement of the same arithmetic, and `requests-are-attempts` compares
/// it against the bytes the authority was actually asked for -- so a disagreement
/// between the shape a plan was compiled for and the chunks its run went looking
/// for is a failure with a name rather than a quietly different answer.
fn role_chunk_bytes(plan: &ExpertPlan) -> [u64; 2] {
    let shape = plan.shape();
    [
        2 * shape.intermediate * shape.hidden * 2,
        shape.hidden * shape.intermediate * 2,
    ]
}

/// What the reservations this trace can **name** account for, tier by tier.
///
/// Named by identity -- the authority's own reservations and this run's -- and
/// costed by asking the ledger what they cost. Nothing here recomputes a cap, a
/// control charge or an envelope: a second arithmetic for the same bytes is how
/// two owners come to disagree while both look right. Anything the ledger holds
/// beyond this sum belongs to a reservation nobody in this step can name, which
/// is what "no third charger" means and what the equality finds.
/// What the named reservations charge, how many this step can name, and how
/// many of those the ledger actually holds.
type AccountedCharges = (Vec<(Scope, Tier, u64)>, u64, u64);

fn accounted_charges(
    authority: &ResidencyAuthority,
    run: &GroupedRun<'_>,
    ledger: &Ledger,
) -> Result<AccountedCharges, Error> {
    // Fixed-size, because this runs inside a generation step and every
    // allocation on that path has to be one somebody reserved. A review found
    // the previous version aborting the process on an injected failure inside
    // `reservation_ids`, which built a `Vec`: reserving the destination says
    // nothing about a temporary the callee builds.
    //
    // Three is the count, not a guess: the authority holds at most two -- its
    // host cache's backing and the one envelope covering every device cache --
    // and the run holds one.
    let mut named: [Option<moxie_memory::ReservationId>; 3] = [None; 3];
    let mut count = 0;
    for id in authority.reservation_ids().into_iter().flatten() {
        named[count] = Some(id);
        count += 1;
    }
    named[count] = Some(run.reservation_id());
    count += 1;
    let named = &named[..count];

    let mut found: [Option<moxie_memory::ReservationId>; 3] = [None; 3];
    let mut found_count = 0usize;
    let mut out: Vec<(Scope, Tier, u64)> = Vec::new();
    let mut failure = None;
    ledger.for_each_charge(|id, scope, tier, bytes| {
        if failure.is_some() || !named.contains(&Some(id)) {
            return;
        }
        if !found[..found_count].contains(&Some(id)) {
            if found_count == found.len() {
                failure = Some(invalid(
                    "reservations",
                    "more reservations were charged than this step can name".into(),
                ));
                return;
            }
            found[found_count] = Some(id);
            found_count += 1;
        }
        match out.iter_mut().find(|(s, t, _)| *s == scope && *t == tier) {
            Some(entry) => entry.2 = entry.2.saturating_add(bytes),
            None => {
                if out.try_reserve(1).is_err() {
                    failure = Some(Error::CapacityExceeded {
                        tier: Some(Tier::Host(HostTier::Pageable)),
                        requested_bytes: core::mem::size_of::<(Scope, Tier, u64)>() as u64,
                        available_bytes: 0,
                    });
                    return;
                }
                out.push((scope, tier, bytes));
            }
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    out.retain(|(_, _, bytes)| *bytes != 0);
    // Scope order then `Tier::ALL` order, so this list and the observed one are
    // comparable row by row.
    out.sort_by_key(|(scope, tier, _)| (*scope, tier_rank(*tier)));
    Ok((out, named.len() as u64, found_count as u64))
}

/// A tier's position in `Tier::ALL`, which is the order both charge lists use.
fn tier_rank(tier: Tier) -> usize {
    Tier::ALL
        .iter()
        .position(|t| *t == tier)
        .unwrap_or(usize::MAX)
}

impl StepTrace {
    /// Assemble a step's trace from its layers and the state it ended in.
    ///
    /// Call this **after** every run is closed and the authority has given its
    /// cache envelope back. `nothing-outstanding` asks whether anything is still
    /// charged when the step is over, and a step still holding its own cache is
    /// not over.
    /// `artifact` and `case` are **owned strings the caller already built**, not
    /// `impl Into<String>`: converting a `&str` here would be an infallible
    /// allocation inside this function, which is the one thing it may not have.
    /// Building them is the caller's, where a failure is its own to handle.
    pub fn new(
        artifact: String,
        case: String,
        layers: Vec<LayerTrace>,
        start: &StepSnapshot,
        authority: &ResidencyAuthority,
        ledger: &Ledger,
    ) -> Result<Self, Error> {
        let mut totals: Vec<(Scope, ByteFlow)> = Vec::new();
        for layer in &layers {
            for delta in &layer.scopes {
                let entry = entry_for(&mut totals, delta.scope)?;
                *entry = entry.plus(&delta.flow);
            }
        }
        let mut whole: Vec<(Scope, ByteFlow)> = Vec::new();
        let mut final_accounts: Vec<ScopeAccount> = Vec::new();
        reserve(&mut whole, authority.scope_count())?;
        reserve(&mut final_accounts, authority.scope_count())?;
        let mut overflow = None;
        authority.for_each_account(|a| {
            if whole.len() == whole.capacity() || final_accounts.len() == final_accounts.capacity()
            {
                overflow = Some(a.scope);
                return;
            }
            whole.push((a.scope, a.flow.since(&start.flow_of(a.scope))));
            final_accounts.push(a);
        });
        if let Some(scope) = overflow {
            return Err(invalid(
                "scopes",
                format!("{scope} appeared while this step was being assembled"),
            ));
        }
        Ok(StepTrace {
            schema_version: TRACE_SCHEMA_VERSION,
            artifact,
            case,
            layers,
            totals,
            outstanding_reservations: ledger.outstanding_count(),
            final_accounts,
            whole,
            unmeasured: UNMEASURED,
        })
    }
}

// ---------------------------------------------------------------------------
// Reconciliation
// ---------------------------------------------------------------------------

/// One named equality's two sides.
struct Check<'a> {
    layer: Option<u32>,
    scope: Option<Scope>,
    out: &'a mut Reconciled,
}

impl Check<'_> {
    fn eq(
        &mut self,
        check: &'static str,
        left: u64,
        right: u64,
        detail: &'static str,
    ) -> Result<(), Discrepancy> {
        self.out.equalities_checked += 1;
        if left == right {
            return Ok(());
        }
        Err(Discrepancy {
            check,
            layer: self.layer,
            scope: self.scope,
            left,
            right,
            detail,
        })
    }

    /// The declared-lower-bound form. Counted separately, so a run can be asked
    /// how many of its checks were the weaker claim.
    fn at_least(
        &mut self,
        check: &'static str,
        left: u64,
        right: u64,
        detail: &'static str,
    ) -> Result<(), Discrepancy> {
        self.out.equalities_checked += 1;
        self.out.bounds_checked += 1;
        if left >= right {
            return Ok(());
        }
        Err(Discrepancy {
            check,
            layer: self.layer,
            scope: self.scope,
            left,
            right,
            detail,
        })
    }
}

impl LayerTrace {
    fn scope(&self, scope: Scope) -> Option<&ScopeDelta> {
        self.scopes.iter().find(|d| d.scope == scope)
    }

    fn device_flow(&self) -> ByteFlow {
        self.scopes
            .iter()
            .filter(|d| matches!(d.scope, Scope::Device(_)))
            .fold(ByteFlow::default(), |acc, d| acc.plus(&d.flow))
    }

    /// Every equality this layer must satisfy.
    fn reconcile(&self, out: &mut Reconciled) -> Result<(), Discrepancy> {
        let mut c = Check {
            layer: Some(self.layer),
            scope: None,
            out,
        };

        // --- the ledger --------------------------------------------------
        //
        // Which ledger, before what it says. A review read a completed run's
        // charges out of a different, empty ledger and the layer reconciled:
        // observed and accounted were both empty, and empty equals empty.
        c.eq(
            "ledger-is-the-runs-own",
            self.ledger_id,
            self.run_ledger_id,
            "the ledger this trace read is not the one the run was admitted against",
        )?;
        c.eq(
            "every-reservation-is-charged",
            self.reservations_found,
            self.reservations_named,
            "a reservation this step can name is not in the ledger, so its charge cannot be compared",
        )?;
        c.eq(
            "ledger-charges-what-is-held",
            self.ledger.len() as u64,
            self.accounted.len() as u64,
            "the ledger holds a different number of charged tiers than the authority's caps and this plan's envelope account for",
        )?;
        for ((scope, tier, charged), (a_scope, a_tier, accounted)) in
            self.ledger.iter().zip(self.accounted.iter())
        {
            c.scope = Some(*scope);
            if scope != a_scope || tier != a_tier {
                return Err(Discrepancy {
                    check: "ledger-charges-what-is-held",
                    layer: Some(self.layer),
                    scope: Some(*scope),
                    left: *charged,
                    right: *accounted,
                    detail: "the ledger charges a tier the envelope does not account for",
                });
            }
            c.eq(
            "ledger-charges-what-is-held",
                *charged,
                *accounted,
                "a tier's charge differs from what the authority's caps and this plan's envelope account for -- a difference is a third charger",
            )?;
        }
        c.scope = None;

        for delta in &self.scopes {
            c.scope = Some(delta.scope);
            c.eq(
                "cache-cap-is-the-reservation",
                delta.resident_bytes.min(delta.cap_bytes),
                delta.resident_bytes,
                "a scope holds more than its cap; the first figure is the cap it was clamped to",
            )?;
        }
        c.scope = None;

        // --- the run -----------------------------------------------------
        let requested: u64 = self
            .scopes
            .iter()
            .map(|d| d.flow.requested_bytes)
            .fold(0, u64::saturating_add);
        let attempted = self.cost.acquires_issued[0]
            .saturating_mul(self.chunk_bytes[0])
            .saturating_add(self.cost.acquires_issued[1].saturating_mul(self.chunk_bytes[1]));
        c.eq(
            "requests-are-attempts",
            requested,
            attempted,
            "the bytes the authority was asked for are not the run's own acquire count times the plan's chunk lengths",
        )?;
        // A layer that failed still has to account for its bytes, and every
        // identity above applies to it unchanged. What changes is what its
        // *counts* may be: it ran some prefix of its plan, so the equality below
        // becomes the bound that prefix satisfies. The trace says which it was
        // rather than leaving a reader to infer it from the numbers.
        if self.completed {
            c.eq(
                "run-ran-the-plan",
                self.cost.groups_run,
                self.groups,
                "a completed layer ran a different number of groups than its plan has",
            )?;
        } else {
            c.at_least(
                "run-ran-the-plan",
                self.groups,
                self.cost.groups_run,
                "a failed layer ran more groups than its plan has",
            )?;
        }
        c.eq(
            "run-ran-the-plan",
            self.cost
                .host_groups
                .saturating_add(self.cost.device_groups),
            self.cost.groups_run,
            "the host and device group counts do not sum to the groups run",
        )?;
        // Found by mutation: deleting the launch counter survived the whole
        // sweep, because nothing compared it to anything. Then found by review:
        // the first version of this equality compared it to the *group* count,
        // and the device path submits one launch per **symbol** of the
        // descriptor it selected -- a projection and a reduction. It was
        // validating a quantity that was wrong by a factor of two, which is
        // worse than not checking it.
        let per_group = self
            .cost
            .device_groups
            .saturating_mul(self.kernels_per_group);
        if self.completed {
            c.eq(
            "launches-match-device-groups",
                self.cost.launches,
                per_group,
                "the launches the lane reported are not one per symbol of the descriptor for every device group",
            )?;
        } else {
            // A layer that failed part way through a group submitted what it
            // submitted: at least every completed group's symbols, and at most
            // one further group's. Both sides, because a lane that reported
            // nothing and one that reported a whole extra group are different
            // defects.
            c.at_least(
            "launches-match-device-groups",
                self.cost.launches,
                per_group, "a failed layer submitted fewer launches than its completed device groups account for")?;
            c.at_least(
            "launches-match-device-groups",
                per_group.saturating_add(self.kernels_per_group),
                self.cost.launches, "a failed layer submitted more launches than its completed groups plus the one that failed can account for")?;
        }
        let slots = self.rows.saturating_mul(self.top_k);
        if self.completed {
            c.eq(
                "slots-are-written-once",
                self.cost.slots_written,
                slots,
                "a completed layer wrote a different number of slots than rows times top-k",
            )?;
        } else {
            c.at_least(
                "slots-are-written-once",
                slots,
                self.cost.slots_written,
                "a failed layer wrote more slots than rows times top-k",
            )?;
        }
        // Exact in both cases, and the withheld term is why. A run that failed
        // after an unknown submission keeps its leases rather than releasing
        // bytes a copy may still be reading (R07); a count that compared only
        // taken against given back would call that a leak, and one that ignored
        // the difference would miss a real one.
        c.eq(
            "leases-balance",
            self.cost.leases_acquired,
            self.cost
                .leases_released
                .saturating_add(self.withheld_leases),
            "the leases taken are not the leases given back plus the leases withheld",
        )?;

        // --- the prediction ----------------------------------------------
        //
        // What holds depends on what the planner declared, and the two cases are
        // genuinely different claims rather than one claim with a tolerance.
        //
        // When the layer's whole admitted set fits, every chunk is admitted once
        // and every prediction is an equality.
        //
        // When it does not, two things can happen and they pull in opposite
        // directions: eviction can turn a predicted hit into an admission, and a
        // backpressure retry can find a chunk still resident and hit on it
        // without anyone having predicted a hit. So admissions and reads have a
        // sound **lower** bound -- a chunk that had to be admitted is still
        // admitted at least once -- while hits are bounded on *both* sides by
        // quantities this trace already records: a predicted hit can only be
        // lost to an eviction, and an unpredicted one can only come from a
        // request the plan did not contain.
        if !self.completed {
            // A layer that stopped part way admitted some prefix of what was
            // predicted, and neither direction of the comparison is sound: it
            // may have admitted everything and failed on a launch, or nothing
            // and failed on the first read. Skipped, counted as skipped, and
            // said so -- an inequality that cannot fail is not a check.
            c.out.predictions_skipped += 1;
            return Ok(());
        }
        let device = self.device_flow();
        let host = self.scope(Scope::Host).map(|d| d.flow).unwrap_or_default();
        let p = self.predicted;
        // One flag, not two. `Exactness::LowerBound` names the cache that could
        // not hold the layer, and that is worth reporting, but it does not make
        // the *other* side exact: a device chunk admitted twice asks the host for
        // its source twice, so a bound on either cache is a bound on both
        // predictions. This task got that wrong once and the sweep found it.
        let exact = matches!(p.exactness, Exactness::Exact);
        let device_exact = exact;
        let host_exact = exact;
        let pair_bytes = self.chunk_bytes[0].saturating_add(self.chunk_bytes[1]);
        let device_retried = device
            .requested_bytes
            .saturating_sub(self.cost.device_groups.saturating_mul(pair_bytes));
        let host_retried = host
            .requested_bytes
            .saturating_sub(self.cost.host_groups.saturating_mul(pair_bytes));

        if device_exact {
            c.eq(
                "predicted-uploads-are-uploaded",
                device.admitted_bytes,
                p.device_upload_bytes,
                "device admissions differ from what was predicted exactly",
            )?;
            c.eq(
                "predicted-hits-are-hit",
                device.hit_bytes,
                p.device_hit_bytes,
                "device hits differ from what was predicted exactly",
            )?;
            c.eq(
                "predicted-source-reuse-is-reused",
                device.source_reuse_bytes,
                p.host_source_reuse_bytes,
                "reused host sources differ from what was predicted exactly",
            )?;
        } else {
            c.at_least(
                "predicted-uploads-are-uploaded",
                device.admitted_bytes,
                p.device_upload_bytes,
                "device admissions are below the declared lower bound",
            )?;
            c.at_least(
            "predicted-hits-are-hit",
                device.hit_bytes.saturating_add(device.evicted_bytes),
                p.device_hit_bytes,
                "device hits plus evictions are below what was predicted resident: a predicted hit can only be lost to an eviction",
            )?;
            c.at_least(
            "predicted-hits-are-hit",
                p.device_hit_bytes.saturating_add(device_retried),
                device.hit_bytes,
                "device hits exceed what was predicted plus the retried requests: an unpredicted hit can only come from a retry",
            )?;
            c.at_least(
                "predicted-source-reuse-is-reused",
                device.source_reuse_bytes.saturating_add(host.evicted_bytes),
                p.host_source_reuse_bytes,
                "reused host sources plus host evictions are below what was predicted",
            )?;
        }

        if host_exact {
            c.eq(
                "predicted-reads-are-read",
                host.read_bytes,
                p.host_read_bytes,
                "host reads differ from what was predicted exactly",
            )?;
            c.eq(
                "predicted-hits-are-hit",
                host.hit_bytes,
                p.host_hit_bytes,
                "host hits differ from what was predicted exactly",
            )?;
        } else {
            c.at_least(
                "predicted-reads-are-read",
                host.read_bytes,
                p.host_read_bytes,
                "host reads are below the declared lower bound",
            )?;
            c.at_least(
                "predicted-hits-are-hit",
                host.hit_bytes.saturating_add(host.evicted_bytes),
                p.host_hit_bytes,
                "host hits plus evictions are below what was predicted resident",
            )?;
            c.at_least(
                "predicted-hits-are-hit",
                p.host_hit_bytes.saturating_add(host_retried),
                host.hit_bytes,
                "host hits exceed what was predicted plus the retried requests",
            )?;
        }
        Ok(())
    }
}

impl StepTrace {
    /// Check every equality, layer by layer and then over the step.
    ///
    /// Returns what it checked rather than `()`: a reconciliation that reports
    /// only success cannot be told apart from one that checked nothing, and this
    /// repository has met that shape of claim before.
    pub fn reconcile(&self) -> Result<Reconciled, Discrepancy> {
        let mut out = Reconciled {
            layers: self.layers.len() as u32,
            ..Reconciled::default()
        };
        if self.schema_version != TRACE_SCHEMA_VERSION {
            return Err(Discrepancy {
                check: "schema-version-is-this-one",
                layer: None,
                scope: None,
                left: u64::from(self.schema_version),
                right: u64::from(TRACE_SCHEMA_VERSION),
                detail: "this trace was written against another schema",
            });
        }
        for layer in &self.layers {
            layer.reconcile(&mut out)?;
        }

        // --- the step ------------------------------------------------------
        //
        // Summed **without** `ByteFlow::plus`, which is what `StepTrace::new`
        // used to build `totals`. Recomputing with the same function would make
        // this a comparison of a thing to itself, and this repository has twice
        // found that shape passing over a defect.
        let mut summed: Vec<(Scope, ByteFlow)> = Vec::new();
        for layer in &self.layers {
            for delta in &layer.scopes {
                // **No `format!` here.** This is the path that handles an
                // allocation failure, and a review reproduced it aborting the
                // process while building the message that says so. The detail is
                // static; the numbers a caller acts on are the fields beside it.
                let e = entry_for(&mut summed, delta.scope).map_err(|_| Discrepancy {
                    check: "step-is-the-sum-of-layers",
                    layer: Some(layer.layer),
                    scope: Some(delta.scope),
                    left: 0,
                    right: 0,
                    detail: "the step's totals could not be computed: out of memory",
                })?;
                let f = &delta.flow;
                e.requested_bytes += f.requested_bytes;
                e.hit_bytes += f.hit_bytes;
                e.coalesced_bytes += f.coalesced_bytes;
                e.refused_bytes += f.refused_bytes;
                e.admitted_bytes += f.admitted_bytes;
                e.direct_admitted_bytes += f.direct_admitted_bytes;
                e.as_source_admitted_bytes += f.as_source_admitted_bytes;
                e.source_created_bytes += f.source_created_bytes;
                e.source_reuse_bytes += f.source_reuse_bytes;
                e.source_rollback_bytes += f.source_rollback_bytes;
                e.read_bytes += f.read_bytes;
                e.uploaded_bytes += f.uploaded_bytes;
                e.abandoned_bytes += f.abandoned_bytes;
                e.withheld_bytes += f.withheld_bytes;
                e.unfinished_bytes += f.unfinished_bytes;
                e.evicted_bytes += f.evicted_bytes;
                e.retired_bytes += f.retired_bytes;
                e.discarded_bytes += f.discarded_bytes;
            }
        }
        let mut c = Check {
            layer: None,
            scope: None,
            out: &mut out,
        };
        c.eq(
            "step-is-the-sum-of-layers",
            self.totals.len() as u64,
            summed.len() as u64,
            "the step's totals cover a different set of scopes than its layers",
        )?;
        for (scope, total) in &self.totals {
            c.scope = Some(*scope);
            let want = summed
                .iter()
                .find(|(s, _)| s == scope)
                .map_or_else(ByteFlow::default, |(_, f)| *f);
            if *total != want {
                return Err(Discrepancy {
                    check: "step-is-the-sum-of-layers",
                    layer: None,
                    scope: Some(*scope),
                    left: total.admitted_bytes,
                    right: want.admitted_bytes,
                    detail: "the step's totals are not the sum of its layers' flows",
                });
            }
            c.out.equalities_checked += 1;
        }
        c.scope = None;

        // --- the step's outer boundary -------------------------------------
        //
        // Every byte that **entered** a cache during this step belongs to a
        // layer. Departures need not: a step retires its caches after its last
        // layer, and that is cleanup rather than an omission. Stated as six
        // equalities per scope, because a trace that only re-sums the records it
        // was handed cannot notice one that is missing -- which is exactly what
        // a review demonstrated by dropping the middle layer of three.
        for (scope, whole) in &self.whole {
            c.scope = Some(*scope);
            let mine = summed
                .iter()
                .find(|(s, _)| s == scope)
                .map_or_else(ByteFlow::default, |(_, f)| *f);
            // One static message per flow, so a failure says **which** inflow
            // the layers do not account for without formatting anything: a
            // discrepancy has to be constructible when there is no memory to
            // build prose with.
            for (what, outer, inner) in [
                (
                    "the layers do not account for every byte the authority was asked for",
                    whole.requested_bytes,
                    mine.requested_bytes,
                ),
                (
                    "the layers do not account for every cache hit",
                    whole.hit_bytes,
                    mine.hit_bytes,
                ),
                (
                    "the layers do not account for every coalesced acquire",
                    whole.coalesced_bytes,
                    mine.coalesced_bytes,
                ),
                (
                    "the layers do not account for every byte admitted -- a byte that entered a \
                     cache and belongs to no layer is a layer missing from this trace",
                    whole.admitted_bytes,
                    mine.admitted_bytes,
                ),
                (
                    "the layers do not account for every byte read",
                    whole.read_bytes,
                    mine.read_bytes,
                ),
                (
                    "the layers do not account for every byte uploaded",
                    whole.uploaded_bytes,
                    mine.uploaded_bytes,
                ),
            ] {
                c.eq("step-covers-every-byte", outer, inner, what)?;
            }
        }
        c.scope = None;

        c.eq(
            "nothing-outstanding",
            self.outstanding_reservations as u64,
            0,
            "reservations were still charged when the step ended",
        )?;
        for account in &self.final_accounts {
            c.scope = Some(account.scope);
            c.eq(
                "nothing-outstanding",
                account.resident_bytes,
                0,
                "a scope still holds bytes after the step",
            )?;
            let gone = account
                .flow
                .evicted_bytes
                .saturating_add(account.flow.retired_bytes)
                .saturating_add(account.flow.discarded_bytes);
            c.eq(
                "nothing-outstanding",
                account.flow.admitted_bytes,
                gone,
                "a scope's admissions are not all evicted, retired or discarded",
            )?;
        }
        Ok(out)
    }
}
