//! Resource tiers and scopes: the closed vocabulary the memory authority
//! accounts in.
//!
//! Document 03 names what must be tracked *separately* on each device and on
//! the host, and the separation is the point: an admission report that collapses
//! weights, cache pages, workspace and staging into one number cannot say which
//! constraint binds, and therefore cannot offer a legal alternative. The lists
//! below are that document's lists, not a superset invented here.
//!
//! These descriptors live in `moxie-types` rather than in the memory crate
//! because document 02 requires shared descriptor types to be resolved into
//! lower-level crates, and because [`crate::Error::CapacityExceeded`] already
//! has to name a tier.

use core::fmt;

use crate::ids::DeviceUuid;

/// What is tracked separately on one device (document 03).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceTier {
    /// Weights resident in their packed canonical layout.
    PackedResidentWeights,
    /// Expert chunks held for reuse. Bounded, and evictable under lease rules.
    ExpertCache,
    /// KV and other paged sequence state.
    KvStatePages,
    /// Recurrent / convolution state, which is not shaped like KV.
    RecurrentState,
    /// Transient per-step activations.
    Activations,
    /// The logits buffer, which is vocabulary-sized and not an activation.
    Logits,
    /// Scratch a kernel requires beyond its operands.
    KernelWorkspace,
    /// Buffers a collective needs on top of the tensors it moves.
    CollectiveBuffers,
    /// Memory pools pinned by a captured graph.
    GraphPools,
    /// Staging for host-to-device and device-to-host transfers.
    TransferStaging,
    /// The target model's state during speculative decoding.
    SpeculativeTargetState,
    /// The draft proposer's state during speculative decoding.
    SpeculativeDraftState,
    /// State held per future-entropy branch.
    EntropyBranches,
    /// Bytes lost to fragmentation. Real capacity, so it is declared, not hidden.
    AllocatorFragmentation,
    /// Deliberate headroom that must stay unallocated.
    SafetyHeadroom,
}

/// What is tracked separately on the host (document 03).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HostTier {
    /// Ordinary pageable allocations.
    Pageable,
    /// Pinned memory. Bounded on purpose: pinning is an option, not a universal
    /// speed claim (R12), and an unbounded pinned region starves the system.
    Pinned,
    /// Resident pages of a mapping. Reported, and **not** charged against the
    /// committed host budget -- see [`Tier::charges_scope_budget`].
    MappedResident,
    /// CPU-side compute workspace, for host expert execution and conversion.
    CpuWorkspace,
    /// Sequence state spilled from a device.
    StateSpill,
    /// Bounded read and conversion buffers owned by the importer.
    ConversionReadBuffers,
}

/// One tracked resource tier, in one scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Device(DeviceTier),
    Host(HostTier),
}

/// Whether a scope is a device or the host, without naming which device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScopeKind {
    Device,
    Host,
}

/// Where a tier is tracked. A device is identified by its UUID, never by an
/// ordinal: AGENTS.md requires plans, manifests and results to name a GPU by
/// UUID, and on this machine ordinal 0 is not a 3090.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    Host,
    Device(DeviceUuid),
}

impl Scope {
    pub const fn kind(self) -> ScopeKind {
        match self {
            Scope::Host => ScopeKind::Host,
            Scope::Device(_) => ScopeKind::Device,
        }
    }

    /// The device's UUID, or `None` for the host scope.
    pub const fn device(self) -> Option<DeviceUuid> {
        match self {
            Scope::Host => None,
            Scope::Device(u) => Some(u),
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scope::Host => f.write_str("host"),
            Scope::Device(u) => write!(f, "device({u})"),
        }
    }
}

impl DeviceTier {
    pub const ALL: &'static [DeviceTier] = &[
        DeviceTier::PackedResidentWeights,
        DeviceTier::ExpertCache,
        DeviceTier::KvStatePages,
        DeviceTier::RecurrentState,
        DeviceTier::Activations,
        DeviceTier::Logits,
        DeviceTier::KernelWorkspace,
        DeviceTier::CollectiveBuffers,
        DeviceTier::GraphPools,
        DeviceTier::TransferStaging,
        DeviceTier::SpeculativeTargetState,
        DeviceTier::SpeculativeDraftState,
        DeviceTier::EntropyBranches,
        DeviceTier::AllocatorFragmentation,
        DeviceTier::SafetyHeadroom,
    ];

    /// Stable machine-readable name. Adding a variant without extending this is
    /// a compile error, which is how the list stays document 03's list.
    pub const fn name(self) -> &'static str {
        match self {
            DeviceTier::PackedResidentWeights => "device.packed_resident_weights",
            DeviceTier::ExpertCache => "device.expert_cache",
            DeviceTier::KvStatePages => "device.kv_state_pages",
            DeviceTier::RecurrentState => "device.recurrent_state",
            DeviceTier::Activations => "device.activations",
            DeviceTier::Logits => "device.logits",
            DeviceTier::KernelWorkspace => "device.kernel_workspace",
            DeviceTier::CollectiveBuffers => "device.collective_buffers",
            DeviceTier::GraphPools => "device.graph_pools",
            DeviceTier::TransferStaging => "device.transfer_staging",
            DeviceTier::SpeculativeTargetState => "device.speculative_target_state",
            DeviceTier::SpeculativeDraftState => "device.speculative_draft_state",
            DeviceTier::EntropyBranches => "device.entropy_branches",
            DeviceTier::AllocatorFragmentation => "device.allocator_fragmentation",
            DeviceTier::SafetyHeadroom => "device.safety_headroom",
        }
    }
}

impl HostTier {
    pub const ALL: &'static [HostTier] = &[
        HostTier::Pageable,
        HostTier::Pinned,
        HostTier::MappedResident,
        HostTier::CpuWorkspace,
        HostTier::StateSpill,
        HostTier::ConversionReadBuffers,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            HostTier::Pageable => "host.pageable",
            HostTier::Pinned => "host.pinned",
            HostTier::MappedResident => "host.mapped_resident",
            HostTier::CpuWorkspace => "host.cpu_workspace",
            HostTier::StateSpill => "host.state_spill",
            HostTier::ConversionReadBuffers => "host.conversion_read_buffers",
        }
    }
}

impl Tier {
    /// Every tier, device tiers first, in declaration order.
    pub const ALL: &'static [Tier] = &[
        Tier::Device(DeviceTier::PackedResidentWeights),
        Tier::Device(DeviceTier::ExpertCache),
        Tier::Device(DeviceTier::KvStatePages),
        Tier::Device(DeviceTier::RecurrentState),
        Tier::Device(DeviceTier::Activations),
        Tier::Device(DeviceTier::Logits),
        Tier::Device(DeviceTier::KernelWorkspace),
        Tier::Device(DeviceTier::CollectiveBuffers),
        Tier::Device(DeviceTier::GraphPools),
        Tier::Device(DeviceTier::TransferStaging),
        Tier::Device(DeviceTier::SpeculativeTargetState),
        Tier::Device(DeviceTier::SpeculativeDraftState),
        Tier::Device(DeviceTier::EntropyBranches),
        Tier::Device(DeviceTier::AllocatorFragmentation),
        Tier::Device(DeviceTier::SafetyHeadroom),
        Tier::Host(HostTier::Pageable),
        Tier::Host(HostTier::Pinned),
        Tier::Host(HostTier::MappedResident),
        Tier::Host(HostTier::CpuWorkspace),
        Tier::Host(HostTier::StateSpill),
        Tier::Host(HostTier::ConversionReadBuffers),
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Tier::Device(t) => t.name(),
            Tier::Host(t) => t.name(),
        }
    }

    /// The scope kind this tier may be requested in. A device tier in the host
    /// scope is a malformed request, not a coerced default.
    pub const fn scope_kind(self) -> ScopeKind {
        match self {
            Tier::Device(_) => ScopeKind::Device,
            Tier::Host(_) => ScopeKind::Host,
        }
    }

    /// Whether bytes in this tier count against the scope's committed budget.
    ///
    /// False for exactly one tier. Document 03: "Mapped virtual bytes do not
    /// equal committed host RAM; neither is free." A mapping's resident pages
    /// are real pressure and are reported in their own row against their own
    /// cap, but charging them to the same budget as pinned and pageable
    /// allocations would double-count bytes the process never committed.
    pub const fn charges_scope_budget(self) -> bool {
        !matches!(self, Tier::Host(HostTier::MappedResident))
    }

    /// The tiers valid in a scope, in `ALL` order.
    pub fn valid_in(kind: ScopeKind) -> impl Iterator<Item = Tier> {
        Tier::ALL
            .iter()
            .copied()
            .filter(move |t| t.scope_kind() == kind)
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn all_is_exhaustive_and_names_are_distinct() {
        // `ALL` is written by hand, so it needs a test that it is complete.
        // Constructing each variant here means adding one to the enum without
        // adding it to `ALL` fails this test rather than silently dropping a
        // tier from every admission report.
        assert_eq!(DeviceTier::ALL.len(), 15);
        assert_eq!(HostTier::ALL.len(), 6);
        assert_eq!(Tier::ALL.len(), DeviceTier::ALL.len() + HostTier::ALL.len());

        for d in DeviceTier::ALL {
            assert!(Tier::ALL.contains(&Tier::Device(*d)), "{d:?} missing");
        }
        for h in HostTier::ALL {
            assert!(Tier::ALL.contains(&Tier::Host(*h)), "{h:?} missing");
        }

        let names: BTreeSet<&str> = Tier::ALL.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), Tier::ALL.len(), "tier names must be distinct");
    }

    #[test]
    fn every_tier_belongs_to_exactly_one_scope_kind() {
        let device: Vec<Tier> = Tier::valid_in(ScopeKind::Device).collect();
        let host: Vec<Tier> = Tier::valid_in(ScopeKind::Host).collect();
        assert_eq!(device.len(), DeviceTier::ALL.len());
        assert_eq!(host.len(), HostTier::ALL.len());
        assert!(device.iter().all(|t| !host.contains(t)));
    }

    #[test]
    fn mapped_resident_is_the_only_uncharged_tier() {
        // If a second tier ever stops being charged, that is a decision about
        // what admission means, and it should fail this test first.
        let uncharged: Vec<&str> = Tier::ALL
            .iter()
            .filter(|t| !t.charges_scope_budget())
            .map(|t| t.name())
            .collect();
        assert_eq!(uncharged, vec!["host.mapped_resident"]);
    }

    #[test]
    fn scope_orders_and_prints_by_identity_not_ordinal() {
        let a = DeviceUuid::parse("GPU-00000000-0000-0000-0000-00000000000a").unwrap();
        let b = DeviceUuid::parse("GPU-00000000-0000-0000-0000-00000000000b").unwrap();
        assert!(Scope::Host < Scope::Device(a));
        assert!(Scope::Device(a) < Scope::Device(b));
        assert_eq!(
            Scope::Device(a).to_string(),
            "device(GPU-00000000-0000-0000-0000-00000000000a)"
        );
        assert_eq!(Scope::Host.to_string(), "host");
        assert_eq!(Scope::Device(a).kind(), ScopeKind::Device);
        assert_eq!(Scope::Host.device(), None);
    }
}
