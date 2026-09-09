//! The admission report, and what a refusal has to say.
//!
//! Document 03: "An admission report must show physical capacity, already
//! committed resources, reserved peak, and remaining headroom for every tier",
//! and an impossible request is rejected "before generation with a breakdown and
//! legal alternatives". Both halves are types here rather than log lines,
//! because a caller has to be able to act on them.

use core::fmt;

use moxie_types::{Error, Scope, Tier};

/// One tier's row of the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierReport {
    pub tier: Tier,
    /// The declared ceiling for this tier, if the snapshot set one.
    pub cap_bytes: Option<u64>,
    /// Bytes already reserved here by earlier admissions.
    pub committed_bytes: u64,
    /// This request's peak overlapping live set in this tier.
    pub request_peak_bytes: u64,
    /// The stage at which that peak occurs. The first such stage, when several
    /// tie, so the number is deterministic.
    pub peak_stage: u32,
    /// What is left under the cap after this request. `None` when uncapped:
    /// the tier is bounded by the scope instead, and the scope row says by how
    /// much.
    pub remaining_headroom_bytes: Option<u64>,
    /// Whether these bytes count against the scope's committed budget. False
    /// only for `host.mapped_resident` -- mapped virtual bytes are not
    /// committed host RAM (document 03).
    pub charged_to_scope: bool,
}

/// One scope's totals, and every tier valid in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeReport {
    pub scope: Scope,
    pub physical_bytes: u64,
    pub system_headroom_bytes: u64,
    pub admissible_bytes: u64,
    /// Charged bytes already committed here: the sum of the admitted plans'
    /// scope peaks.
    pub committed_bytes: u64,
    /// This request's peak, summed across charged tiers **at one stage** -- not
    /// the sum of the per-tier peaks, which would reserve buffers that are never
    /// live together.
    pub request_peak_bytes: u64,
    pub peak_stage: u32,
    pub remaining_headroom_bytes: u64,
    pub tiers: Vec<TierReport>,
}

/// Capacity, commitment, this request's peak and the headroom left, for every
/// scope and every tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionReport {
    /// The request's stage labels, so a `peak_stage` index reads as a name.
    pub stages: Vec<String>,
    pub scopes: Vec<ScopeReport>,
}

impl AdmissionReport {
    pub fn scope(&self, scope: Scope) -> Option<&ScopeReport> {
        self.scopes.iter().find(|s| s.scope == scope)
    }

    fn stage_name(&self, index: u32) -> &str {
        self.stages
            .get(index as usize)
            .map(String::as_str)
            .unwrap_or("?")
    }
}

impl ScopeReport {
    pub fn tier(&self, tier: Tier) -> Option<&TierReport> {
        self.tiers.iter().find(|t| t.tier == tier)
    }
}

impl fmt::Display for AdmissionReport {
    /// The breakdown a refusal shows a person. Tiers with nothing in them and no
    /// cap are omitted from the text -- they are still in `scopes` for a caller
    /// that wants them, but twenty-one empty rows per device hide the two that
    /// matter.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.scopes {
            writeln!(
                f,
                "{}: physical {} B, headroom {} B, admissible {} B, committed {} B, \
                 request peak {} B at stage {:?}, remaining {} B",
                s.scope,
                s.physical_bytes,
                s.system_headroom_bytes,
                s.admissible_bytes,
                s.committed_bytes,
                s.request_peak_bytes,
                self.stage_name(s.peak_stage),
                s.remaining_headroom_bytes,
            )?;
            for t in &s.tiers {
                if t.cap_bytes.is_none() && t.committed_bytes == 0 && t.request_peak_bytes == 0 {
                    continue;
                }
                write!(
                    f,
                    "  {:<32} committed {} B, peak {} B at stage {:?}",
                    t.tier.name(),
                    t.committed_bytes,
                    t.request_peak_bytes,
                    self.stage_name(t.peak_stage),
                )?;
                match t.cap_bytes {
                    Some(cap) => write!(
                        f,
                        ", cap {cap} B, remaining {} B",
                        t.remaining_headroom_bytes.unwrap_or(0)
                    )?,
                    None => write!(f, ", uncapped")?,
                }
                if !t.charged_to_scope {
                    write!(f, " (not charged to the scope budget)")?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// Which kind of ceiling a failing constraint hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    /// The tier's own declared cap.
    TierCap,
    /// The scope's admissible total. `tier` then names the largest charged
    /// contributor at the peak stage, which is where a caller looks first.
    ScopeBudget,
}

/// One constraint that failed. Every failing constraint is reported, not the
/// first one hit: a caller that lowers context to clear a KV ceiling and then
/// hits an expert-cache ceiling has been told half the truth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingConstraint {
    pub scope: Scope,
    pub tier: Tier,
    pub kind: BindingKind,
    /// Already committed plus this request's peak.
    pub needed_bytes: u64,
    /// The ceiling that was crossed.
    pub available_bytes: u64,
    pub peak_stage: u32,
}

impl BindingConstraint {
    pub fn shortfall_bytes(&self) -> u64 {
        self.needed_bytes.saturating_sub(self.available_bytes)
    }
}

/// A change the caller could legally make. The ledger lists one only when this
/// request makes it capable of helping, and **never applies one**: document 03
/// forbids automatically shortening context, quantizing the cache or silently
/// lowering weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LegalAlternative {
    /// Some binding buffer declared that it scales with context.
    LowerContext,
    /// Some binding buffer declared that it scales with branches.
    FewerBranches,
    /// A binding tier holds weights, so a different precision or artifact would
    /// change its size.
    OtherWeightPrecisionOrArtifact,
    /// The request spans more than one device, so the split could differ.
    DifferentTopology,
    /// The binding scope is a device and the host has room for the shortfall.
    HostBackedExecution,
}

impl LegalAlternative {
    pub const fn name(self) -> &'static str {
        match self {
            LegalAlternative::LowerContext => "lower the requested context",
            LegalAlternative::FewerBranches => "request fewer speculative or entropy branches",
            LegalAlternative::OtherWeightPrecisionOrArtifact => {
                "use another weight precision or artifact"
            }
            LegalAlternative::DifferentTopology => "use a different topology",
            LegalAlternative::HostBackedExecution => "ask explicitly for host-backed execution",
        }
    }
}

impl fmt::Display for LegalAlternative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A refused admission: the whole breakdown, every failing constraint, and the
/// alternatives this request actually makes legal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub report: AdmissionReport,
    pub binding: Vec<BindingConstraint>,
    /// The worst shortfall among the binding constraints.
    pub shortfall_bytes: u64,
    pub alternatives: Vec<LegalAlternative>,
}

impl Rejection {
    /// The constraint with the largest shortfall. Ties keep the **first** such
    /// constraint, and `binding` is built in scope order, then `Tier::ALL`
    /// order, with a scope-budget constraint after that scope's tier caps, so
    /// the answer is deterministic rather than merely stable today.
    pub fn worst(&self) -> &BindingConstraint {
        let mut best: Option<&BindingConstraint> = None;
        for c in &self.binding {
            if best.is_none_or(|b| c.shortfall_bytes() > b.shortfall_bytes()) {
                best = Some(c);
            }
        }
        best.expect("a rejection has at least one binding constraint")
    }
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "admission refused, short by {} B on {} constraint(s):",
            self.shortfall_bytes,
            self.binding.len()
        )?;
        for c in &self.binding {
            writeln!(
                f,
                "  {} {} needs {} B, {} B available ({:?})",
                c.scope,
                c.tier.name(),
                c.needed_bytes,
                c.available_bytes,
                c.kind
            )?;
        }
        writeln!(f, "legal alternatives:")?;
        if self.alternatives.is_empty() {
            writeln!(f, "  none this request makes available")?;
        }
        for a in &self.alternatives {
            writeln!(f, "  {a}")?;
        }
        write!(f, "{}", self.report)
    }
}

impl From<Rejection> for Error {
    /// For a caller that only needs the variant. The richer breakdown is the
    /// `Rejection` itself; this names the worst binding tier and nothing else.
    fn from(r: Rejection) -> Error {
        let worst = r.worst();
        Error::CapacityExceeded {
            tier: Some(worst.tier),
            requested_bytes: worst.needed_bytes,
            available_bytes: worst.available_bytes,
        }
    }
}
