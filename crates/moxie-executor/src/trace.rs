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

use std::collections::BTreeMap;

use moxie_memory::{ByteFlow, Ledger, ResidencyAuthority, ScopeAccount};
use moxie_plan::expert::{EnvelopePrediction, Exactness, ExpertPlan};
use moxie_types::{Scope, Tier};

use crate::grouped::{GroupedRun, GroupedStats};

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
    pub detail: String,
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

/// The authority's and the ledger's position before a layer runs.
///
/// Held by value, so nothing here borrows the owners while the layer executes.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSnapshot {
    layer: u32,
    flows: BTreeMap<Scope, ByteFlow>,
}

impl LayerSnapshot {
    /// Read every scope's flow. Call this immediately before the layer runs.
    pub fn take(layer: u32, authority: &ResidencyAuthority) -> Self {
        LayerSnapshot {
            layer,
            flows: authority
                .accounts()
                .into_iter()
                .map(|a| (a.scope, a.flow))
                .collect(),
        }
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
    ) -> LayerTrace {
        let plan = run.plan();
        let cost = run.stats();
        let mut scopes = Vec::new();
        for account in authority.accounts() {
            let before = self.flows.get(&account.scope).copied().unwrap_or_default();
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
        }
        let shape = plan.shape();
        // Collected through the same ordered map the accounted side uses, so the
        // two lists are comparable row by row. Two orderings of the same numbers
        // would make the comparison report the wrong pair on a mismatch, which
        // is a diagnosis problem rather than a correctness one -- and this
        // repository has spent two review rounds on findings that were exactly
        // that.
        let mut observed_map: BTreeMap<(Scope, Tier), u64> = BTreeMap::new();
        for scope in ledger.scopes() {
            for tier in tiers_of(scope) {
                let committed = ledger.committed(scope, tier);
                if committed != 0 {
                    observed_map.insert((scope, tier), committed);
                }
            }
        }
        let observed: Vec<(Scope, Tier, u64)> = observed_map
            .into_iter()
            .map(|((scope, tier), bytes)| (scope, tier, bytes))
            .collect();
        LayerTrace {
            layer: self.layer,
            predicted: plan.envelope().predicted,
            chunk_bytes: role_chunk_bytes(plan),
            groups: plan.groups().len() as u64,
            rows: plan.rows(),
            top_k: shape.top_k,
            accounted: accounted_charges(authority, run, ledger),
            scopes,
            cost,
            completed: run.failure().is_none() && !run.is_cancelled(),
            withheld_leases: run.withheld_leases() as u64,
            ledger: observed,
        }
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

/// **Every** tier valid in a scope, not a subset.
///
/// The subset this started as was a hole with a confident name: a third charger
/// using a tier nobody thought to list would have been missed by both sides of
/// `ledger-charges-what-is-held`, and the check would have passed. `Tier::ALL`
/// has its own completeness test, so a tier added later arrives here on its own.
fn tiers_of(scope: Scope) -> Vec<Tier> {
    Tier::valid_in(scope.kind()).collect()
}

/// What the reservations this trace can **name** account for, tier by tier.
///
/// Named by identity -- the authority's own reservations and this run's -- and
/// costed by asking the ledger what they cost. Nothing here recomputes a cap, a
/// control charge or an envelope: a second arithmetic for the same bytes is how
/// two owners come to disagree while both look right. Anything the ledger holds
/// beyond this sum belongs to a reservation nobody in this step can name, which
/// is what "no third charger" means and what the equality finds.
fn accounted_charges(
    authority: &ResidencyAuthority,
    run: &GroupedRun<'_>,
    ledger: &Ledger,
) -> Vec<(Scope, Tier, u64)> {
    let mut named = authority.reservation_ids();
    named.push(run.reservation_id());
    let mut out: BTreeMap<(Scope, Tier), u64> = BTreeMap::new();
    for outstanding in ledger.outstanding() {
        if !named.contains(&outstanding.id) {
            continue;
        }
        for (scope, tier, bytes) in outstanding.charges {
            let entry = out.entry((scope, tier)).or_default();
            *entry = entry.saturating_add(bytes);
        }
    }
    out.into_iter()
        .filter(|(_, bytes)| *bytes != 0)
        .map(|((scope, tier), bytes)| (scope, tier, bytes))
        .collect()
}

impl StepTrace {
    /// Assemble a step's trace from its layers and the state it ended in.
    ///
    /// Call this **after** every run is closed and the authority has given its
    /// cache envelope back. `nothing-outstanding` asks whether anything is still
    /// charged when the step is over, and a step still holding its own cache is
    /// not over.
    pub fn new(
        artifact: impl Into<String>,
        case: impl Into<String>,
        layers: Vec<LayerTrace>,
        authority: &ResidencyAuthority,
        ledger: &Ledger,
    ) -> Self {
        let mut totals: BTreeMap<Scope, ByteFlow> = BTreeMap::new();
        for layer in &layers {
            for delta in &layer.scopes {
                let entry = totals.entry(delta.scope).or_default();
                *entry = entry.plus(&delta.flow);
            }
        }
        StepTrace {
            schema_version: TRACE_SCHEMA_VERSION,
            artifact: artifact.into(),
            case: case.into(),
            layers,
            totals: totals.into_iter().collect(),
            outstanding_reservations: ledger.outstanding().len(),
            final_accounts: authority.accounts(),
            unmeasured: UNMEASURED,
        }
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
        detail: impl FnOnce() -> String,
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
            detail: detail(),
        })
    }

    /// The declared-lower-bound form. Counted separately, so a run can be asked
    /// how many of its checks were the weaker claim.
    fn at_least(
        &mut self,
        check: &'static str,
        left: u64,
        right: u64,
        detail: impl FnOnce() -> String,
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
            detail: detail(),
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
        c.eq(
            "ledger-charges-what-is-held",
            self.ledger.len() as u64,
            self.accounted.len() as u64,
            || {
                format!(
                    "the ledger holds {} charged tier(s) against {} the authority's caps and this \
                     plan's envelope account for: charged {:?}, accounted {:?}",
                    self.ledger.len(),
                    self.accounted.len(),
                    self.ledger,
                    self.accounted
                )
            },
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
                    detail: format!(
                        "the ledger charges {} {} where the envelope accounts for {} {}",
                        scope,
                        tier.name(),
                        a_scope,
                        a_tier.name()
                    ),
                });
            }
            c.eq("ledger-charges-what-is-held", *charged, *accounted, || {
                format!(
                    "{scope} {}: charged {charged} B, accounted {accounted} B -- a difference is a \
                     third charger",
                    tier.name()
                )
            })?;
        }
        c.scope = None;

        for delta in &self.scopes {
            c.scope = Some(delta.scope);
            c.eq(
                "cache-cap-is-the-reservation",
                delta.resident_bytes.min(delta.cap_bytes),
                delta.resident_bytes,
                || {
                    format!(
                        "{} holds {} B against a {} B cap",
                        delta.scope, delta.resident_bytes, delta.cap_bytes
                    )
                },
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
        c.eq("requests-are-attempts", requested, attempted, || {
            format!(
                "the authority was asked for {requested} B; the run issued {} gate-up and {} down \
                 acquire(s) of {} and {} B",
                self.cost.acquires_issued[0],
                self.cost.acquires_issued[1],
                self.chunk_bytes[0],
                self.chunk_bytes[1]
            )
        })?;
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
                || {
                    format!(
                        "{} group(s) ran against a plan of {}",
                        self.cost.groups_run, self.groups
                    )
                },
            )?;
        } else {
            c.at_least(
                "run-ran-the-plan",
                self.groups,
                self.cost.groups_run,
                || {
                    format!(
                        "a failed layer ran {} group(s) of a plan of {}",
                        self.cost.groups_run, self.groups
                    )
                },
            )?;
        }
        c.eq(
            "run-ran-the-plan",
            self.cost
                .host_groups
                .saturating_add(self.cost.device_groups),
            self.cost.groups_run,
            || {
                format!(
                    "{} host and {} device group(s) against {} run",
                    self.cost.host_groups, self.cost.device_groups, self.cost.groups_run
                )
            },
        )?;
        // Found by mutation: deleting the launch counter survived the whole
        // sweep, because nothing compared it to anything. One launch per device
        // group is the equality it was missing, and it is worth having on its
        // own terms -- document 07 lists launch count among the per-phase
        // counters, and a count nobody checks is a count nobody can trust.
        c.eq(
            "launches-match-device-groups",
            self.cost.launches,
            self.cost.device_groups,
            || {
                format!(
                    "{} launch(es) submitted for {} device group(s)",
                    self.cost.launches, self.cost.device_groups
                )
            },
        )?;
        let slots = self.rows.saturating_mul(self.top_k);
        if self.completed {
            c.eq(
                "slots-are-written-once",
                self.cost.slots_written,
                slots,
                || {
                    format!(
                        "{} slot(s) written against {} rows of top-k {}",
                        self.cost.slots_written, self.rows, self.top_k
                    )
                },
            )?;
        } else {
            c.at_least(
                "slots-are-written-once",
                slots,
                self.cost.slots_written,
                || {
                    format!(
                        "a failed layer wrote {} slot(s) of {slots}",
                        self.cost.slots_written
                    )
                },
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
            || {
                format!(
                    "{} lease(s) taken, {} given back, {} withheld",
                    self.cost.leases_acquired, self.cost.leases_released, self.withheld_leases
                )
            },
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
                || {
                    format!(
                        "device admissions: {} B happened, {} B was predicted exactly",
                        device.admitted_bytes, p.device_upload_bytes
                    )
                },
            )?;
            c.eq(
                "predicted-hits-are-hit",
                device.hit_bytes,
                p.device_hit_bytes,
                || {
                    format!(
                        "device hits: {} B happened, {} B was predicted exactly",
                        device.hit_bytes, p.device_hit_bytes
                    )
                },
            )?;
            c.eq(
                "predicted-source-reuse-is-reused",
                device.source_reuse_bytes,
                p.host_source_reuse_bytes,
                || {
                    format!(
                        "reused host sources: {} B happened, {} B was predicted exactly",
                        device.source_reuse_bytes, p.host_source_reuse_bytes
                    )
                },
            )?;
        } else {
            c.at_least(
                "predicted-uploads-are-uploaded",
                device.admitted_bytes,
                p.device_upload_bytes,
                || {
                    format!(
                        "device admissions: {} B against a declared lower bound of {} B",
                        device.admitted_bytes, p.device_upload_bytes
                    )
                },
            )?;
            c.at_least(
                "predicted-hits-are-hit",
                device.hit_bytes.saturating_add(device.evicted_bytes),
                p.device_hit_bytes,
                || {
                    format!(
                        "device hits: {} B happened and {} B was evicted, against {} B predicted \
                         resident -- a predicted hit can only be lost to an eviction",
                        device.hit_bytes, device.evicted_bytes, p.device_hit_bytes
                    )
                },
            )?;
            c.at_least(
                "predicted-hits-are-hit",
                p.device_hit_bytes.saturating_add(device_retried),
                device.hit_bytes,
                || {
                    format!(
                        "device hits: {} B happened against {} B predicted plus {} B of retried \
                         requests -- an unpredicted hit can only come from a retry",
                        device.hit_bytes, p.device_hit_bytes, device_retried
                    )
                },
            )?;
            c.at_least(
                "predicted-source-reuse-is-reused",
                device.source_reuse_bytes.saturating_add(host.evicted_bytes),
                p.host_source_reuse_bytes,
                || {
                    format!(
                        "reused host sources: {} B happened and {} B was evicted from the host, \
                         against {} B predicted",
                        device.source_reuse_bytes, host.evicted_bytes, p.host_source_reuse_bytes
                    )
                },
            )?;
        }

        if host_exact {
            c.eq(
                "predicted-reads-are-read",
                host.read_bytes,
                p.host_read_bytes,
                || {
                    format!(
                        "host reads: {} B happened, {} B was predicted exactly",
                        host.read_bytes, p.host_read_bytes
                    )
                },
            )?;
            c.eq(
                "predicted-hits-are-hit",
                host.hit_bytes,
                p.host_hit_bytes,
                || {
                    format!(
                        "host hits: {} B happened, {} B was predicted exactly",
                        host.hit_bytes, p.host_hit_bytes
                    )
                },
            )?;
        } else {
            c.at_least(
                "predicted-reads-are-read",
                host.read_bytes,
                p.host_read_bytes,
                || {
                    format!(
                        "host reads: {} B against a declared lower bound of {} B",
                        host.read_bytes, p.host_read_bytes
                    )
                },
            )?;
            c.at_least(
                "predicted-hits-are-hit",
                host.hit_bytes.saturating_add(host.evicted_bytes),
                p.host_hit_bytes,
                || {
                    format!(
                        "host hits: {} B happened and {} B was evicted, against {} B predicted \
                         resident",
                        host.hit_bytes, host.evicted_bytes, p.host_hit_bytes
                    )
                },
            )?;
            c.at_least(
                "predicted-hits-are-hit",
                p.host_hit_bytes.saturating_add(host_retried),
                host.hit_bytes,
                || {
                    format!(
                        "host hits: {} B happened against {} B predicted plus {} B of retried \
                         requests",
                        host.hit_bytes, p.host_hit_bytes, host_retried
                    )
                },
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
                detail: "this trace was written against another schema".into(),
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
        let mut summed: BTreeMap<Scope, ByteFlow> = BTreeMap::new();
        for layer in &self.layers {
            for delta in &layer.scopes {
                let e = summed.entry(delta.scope).or_default();
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
            || "the step's totals cover a different set of scopes than its layers".into(),
        )?;
        for (scope, total) in &self.totals {
            c.scope = Some(*scope);
            let want = summed.get(scope).copied().unwrap_or_default();
            if *total != want {
                return Err(Discrepancy {
                    check: "step-is-the-sum-of-layers",
                    layer: None,
                    scope: Some(*scope),
                    left: total.admitted_bytes,
                    right: want.admitted_bytes,
                    detail: format!("{scope}: totals {total:?} against the layers' sum {want:?}"),
                });
            }
            c.out.equalities_checked += 1;
        }
        c.scope = None;

        c.eq(
            "nothing-outstanding",
            self.outstanding_reservations as u64,
            0,
            || {
                format!(
                    "{} reservation(s) were still charged when the step ended",
                    self.outstanding_reservations
                )
            },
        )?;
        for account in &self.final_accounts {
            c.scope = Some(account.scope);
            c.eq("nothing-outstanding", account.resident_bytes, 0, || {
                format!(
                    "{} still holds {} B after the step",
                    account.scope, account.resident_bytes
                )
            })?;
            let gone = account
                .flow
                .evicted_bytes
                .saturating_add(account.flow.retired_bytes)
                .saturating_add(account.flow.discarded_bytes);
            c.eq(
                "nothing-outstanding",
                account.flow.admitted_bytes,
                gone,
                || {
                    format!(
                        "{} admitted {} B and released {} B",
                        account.scope, account.flow.admitted_bytes, gone
                    )
                },
            )?;
        }
        Ok(out)
    }
}
