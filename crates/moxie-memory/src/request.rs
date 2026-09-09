//! What a plan wants, declared before anything is charged.
//!
//! A request is a set of buffers with **liveness spans over an ordered list of
//! stages**, not a list of sizes. That is the whole reason admission can reserve
//! the peak overlapping live set instead of a sum: two buffers that are never
//! live at the same time cost what the larger one costs, and two that coexist at
//! a prefill-to-decode barrier cost both, which document 03 calls the temporary
//! coexistence admission must reflect.

use std::collections::BTreeSet;

use moxie_types::{Error, Result, Scope, Tier};

/// An inclusive span of stage indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSpan {
    pub first: u32,
    pub last: u32,
}

impl StageSpan {
    pub const fn inclusive(first: u32, last: u32) -> Self {
        StageSpan { first, last }
    }

    /// A buffer live for exactly one stage.
    pub const fn at(stage: u32) -> Self {
        StageSpan {
            first: stage,
            last: stage,
        }
    }

    pub const fn covers(&self, stage: u32) -> bool {
        self.first <= stage && stage <= self.last
    }
}

/// What a buffer's size follows, when it follows something the caller could
/// legally ask for less of.
///
/// This is the only reason a rejection may offer `LowerContext` or
/// `FewerBranches`: without a declaration, the ledger would be guessing which
/// knob shrinks which bytes, and a suggestion that cannot help is worse than
/// none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// Grows with the requested context length.
    Context,
    /// Grows with the number of speculative or entropy branches.
    Branches,
}

/// One buffer a plan needs, in one scope and tier, live over one span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferRequest {
    pub label: String,
    pub scope: Scope,
    pub tier: Tier,
    /// Physical bytes. For a mapping this is its **resident** set, which is
    /// charged like every other byte.
    pub bytes: u64,
    pub live: StageSpan,
    pub scales_with: Option<Scaling>,
    /// A mapping's virtual extent, when it differs from the resident set.
    /// Reported, never charged, and never a binding constraint: address space is
    /// not memory. Only [`Tier::has_virtual_extent`] tiers may declare one.
    pub virtual_bytes: Option<u64>,
}

impl BufferRequest {
    pub fn new(
        label: impl Into<String>,
        scope: Scope,
        tier: Tier,
        bytes: u64,
        live: StageSpan,
    ) -> Self {
        BufferRequest {
            label: label.into(),
            scope,
            tier,
            bytes,
            live,
            scales_with: None,
            virtual_bytes: None,
        }
    }

    pub fn scaling(mut self, with: Scaling) -> Self {
        self.scales_with = Some(with);
        self
    }

    /// Declare the mapping's virtual extent. It must be at least the resident
    /// set: a resident set larger than its own mapping is a contradiction, not a
    /// conservative estimate.
    pub fn virtual_extent(mut self, bytes: u64) -> Self {
        self.virtual_bytes = Some(bytes);
        self
    }
}

/// How a reserve's size is derived from the request's own buffers.
///
/// R03: Inkling reserved the worst-case incoming expert and GLM-5.3 separately
/// reserved its two largest linears, each as a constant inside one model's
/// runtime. Those become *derived* reservations here -- there is deliberately no
/// way to hand the ledger a literal reserve size, because a constant copied
/// between models is exactly what stopped being true when the shapes changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveRule {
    /// Room for one more item the size of the tier's largest buffer: the
    /// incoming expert that must fit before anything is evicted.
    LargestBufferOfTier,
    /// Room for the `n` largest buffers of the tier.
    NLargestBuffersOfTier(u32),
}

/// A reserve whose bytes the ledger computes, charged like any other bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedReserve {
    pub label: String,
    pub scope: Scope,
    pub tier: Tier,
    pub rule: ReserveRule,
    pub live: StageSpan,
}

impl DerivedReserve {
    pub fn new(
        label: impl Into<String>,
        scope: Scope,
        tier: Tier,
        rule: ReserveRule,
        live: StageSpan,
    ) -> Self {
        DerivedReserve {
            label: label.into(),
            scope,
            tier,
            rule,
            live,
        }
    }
}

/// A complete resource envelope, admitted or refused as one thing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRequest {
    label: String,
    stages: Vec<String>,
    buffers: Vec<BufferRequest>,
    reserves: Vec<DerivedReserve>,
}

impl PlanRequest {
    /// A request over the given ordered stages. Stage labels must be distinct:
    /// a report that names a peak stage is useless if two stages print the same.
    pub fn new<I, S>(label: impl Into<String>, stages: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stages: Vec<String> = stages.into_iter().map(Into::into).collect();
        if stages.is_empty() {
            return Err(Error::InvalidRequest {
                field: "stages",
                detail: "a plan has at least one stage".into(),
            });
        }
        if u32::try_from(stages.len()).is_err() {
            return Err(Error::InvalidRequest {
                field: "stages",
                detail: format!("{} stages does not fit a stage index", stages.len()),
            });
        }
        let distinct: BTreeSet<&String> = stages.iter().collect();
        if distinct.len() != stages.len() {
            return Err(Error::InvalidRequest {
                field: "stages",
                detail: "stage labels must be distinct".into(),
            });
        }
        if let Some(empty) = stages.iter().position(|s| s.is_empty()) {
            return Err(Error::InvalidRequest {
                field: "stages",
                detail: format!("stage {empty} has an empty label"),
            });
        }
        Ok(PlanRequest {
            label: label.into(),
            stages,
            buffers: Vec::new(),
            reserves: Vec::new(),
        })
    }

    pub fn buffer(&mut self, buffer: BufferRequest) -> Result<&mut Self> {
        self.check(buffer.scope, buffer.tier, buffer.live, &buffer.label)?;
        if let Some(virtual_bytes) = buffer.virtual_bytes {
            if !buffer.tier.has_virtual_extent() {
                return Err(Error::InvalidRequest {
                    field: "virtual_bytes",
                    detail: format!(
                        "{}: {} has no virtual extent distinct from its bytes",
                        buffer.label,
                        buffer.tier.name()
                    ),
                });
            }
            if virtual_bytes < buffer.bytes {
                return Err(Error::InvalidRequest {
                    field: "virtual_bytes",
                    detail: format!(
                        "{}: {virtual_bytes} B of address space cannot hold {} B resident",
                        buffer.label, buffer.bytes
                    ),
                });
            }
        }
        self.buffers.push(buffer);
        Ok(self)
    }

    pub fn reserve(&mut self, reserve: DerivedReserve) -> Result<&mut Self> {
        self.check(reserve.scope, reserve.tier, reserve.live, &reserve.label)?;
        if let ReserveRule::NLargestBuffersOfTier(0) = reserve.rule {
            return Err(Error::InvalidRequest {
                field: "rule",
                detail: format!(
                    "{}: a reserve of the 0 largest buffers is not a reserve",
                    reserve.label
                ),
            });
        }
        self.reserves.push(reserve);
        Ok(self)
    }

    fn check(&self, scope: Scope, tier: Tier, live: StageSpan, label: &str) -> Result<()> {
        if label.is_empty() {
            return Err(Error::InvalidRequest {
                field: "label",
                detail: "every buffer and reserve is labelled, so a report can name it".into(),
            });
        }
        if tier.scope_kind() != scope.kind() {
            return Err(Error::InvalidRequest {
                field: "tier",
                detail: format!("{label}: {} is not a tier of {scope}", tier.name()),
            });
        }
        let last_stage = (self.stages.len() - 1) as u32;
        if live.first > live.last {
            return Err(Error::InvalidRequest {
                field: "live",
                detail: format!(
                    "{label}: span {}..={} ends before it starts",
                    live.first, live.last
                ),
            });
        }
        if live.last > last_stage {
            return Err(Error::InvalidRequest {
                field: "live",
                detail: format!(
                    "{label}: span {}..={} leaves the plan's {} stage(s)",
                    live.first,
                    live.last,
                    self.stages.len()
                ),
            });
        }
        Ok(())
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn stages(&self) -> &[String] {
        &self.stages
    }

    pub fn buffers(&self) -> &[BufferRequest] {
        &self.buffers
    }

    pub fn reserves(&self) -> &[DerivedReserve] {
        &self.reserves
    }

    /// Every scope the request touches, in identity order.
    pub fn scopes(&self) -> BTreeSet<Scope> {
        self.buffers
            .iter()
            .map(|b| b.scope)
            .chain(self.reserves.iter().map(|r| r.scope))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_types::{DeviceTier, DeviceUuid, HostTier};

    fn device() -> Scope {
        Scope::Device(DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000001").unwrap())
    }

    fn plan() -> PlanRequest {
        PlanRequest::new("p", ["prefill", "barrier", "decode"]).unwrap()
    }

    #[test]
    fn a_plan_needs_at_least_one_distinctly_labelled_stage() {
        let none: [&str; 0] = [];
        assert_eq!(
            PlanRequest::new("p", none).unwrap_err().kind(),
            "invalid_request"
        );
        assert_eq!(
            PlanRequest::new("p", ["decode", "decode"])
                .unwrap_err()
                .kind(),
            "invalid_request"
        );
        assert_eq!(
            PlanRequest::new("p", ["decode", ""]).unwrap_err().kind(),
            "invalid_request"
        );
    }

    #[test]
    fn a_tier_must_match_its_scope() {
        let mut p = plan();
        let e = p
            .buffer(BufferRequest::new(
                "wrong",
                device(),
                Tier::Host(HostTier::Pinned),
                1,
                StageSpan::at(0),
            ))
            .unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        assert!(e.to_string().contains("is not a tier of"), "{e}");
        assert!(p.buffers().is_empty(), "a refused buffer is not recorded");
    }

    #[test]
    fn a_span_must_lie_inside_the_declared_stages() {
        let mut p = plan();
        for span in [StageSpan::inclusive(2, 1), StageSpan::inclusive(0, 3)] {
            let e = p
                .buffer(BufferRequest::new(
                    "b",
                    device(),
                    Tier::Device(DeviceTier::Activations),
                    1,
                    span,
                ))
                .unwrap_err();
            assert_eq!(e.kind(), "invalid_request");
        }
        assert!(
            p.buffer(BufferRequest::new(
                "b",
                device(),
                Tier::Device(DeviceTier::Activations),
                1,
                StageSpan::inclusive(0, 2),
            ))
            .is_ok()
        );
    }

    #[test]
    fn a_reserve_of_zero_buffers_is_refused_at_declaration() {
        let mut p = plan();
        let e = p
            .reserve(DerivedReserve::new(
                "r",
                device(),
                Tier::Device(DeviceTier::ExpertCache),
                ReserveRule::NLargestBuffersOfTier(0),
                StageSpan::at(0),
            ))
            .unwrap_err();
        assert_eq!(e.kind(), "invalid_request");
        assert!(p.reserves().is_empty());
    }

    #[test]
    fn spans_and_scopes_read_back() {
        let mut p = plan();
        p.buffer(BufferRequest::new(
            "kv",
            device(),
            Tier::Device(DeviceTier::KvStatePages),
            8,
            StageSpan::inclusive(0, 2),
        ))
        .unwrap()
        .buffer(
            BufferRequest::new(
                "spill",
                Scope::Host,
                Tier::Host(HostTier::StateSpill),
                4,
                StageSpan::at(1),
            )
            .scaling(Scaling::Context),
        )
        .unwrap();
        assert_eq!(p.scopes().len(), 2);
        assert!(p.buffers()[0].live.covers(1));
        assert!(!p.buffers()[1].live.covers(2));
        assert_eq!(p.buffers()[1].scales_with, Some(Scaling::Context));
        assert_eq!(p.label(), "p");
    }
}
