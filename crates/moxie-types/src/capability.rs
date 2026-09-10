//! Hardware and kernel capabilities, and the shared `off` / `auto` / `required`
//! control semantics.
//!
//! Document 04: "`off`, `auto`, and `required` are common control semantics.
//! Report the selected plan and why a requested strategy was rejected." The
//! asymmetry encoded here is the whole point: `auto` may decline a strategy and
//! must say so; `required` must produce an actionable error rather than silently
//! disabling what was asked for.

use core::fmt;

use crate::{
    AccumulationPolicy, ActivationPrecision, TensorLayout, WeightPrecision, ids::DeviceUuid,
};

/// Stable identity of one audited semantic kernel implementation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(pub String);

/// Closed semantic operations that may cross the planning/execution boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SemanticKernelOp {
    Linear,
    RmsNorm,
    Residual,
}

impl SemanticKernelOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::RmsNorm => "rms_norm",
            Self::Residual => "residual",
        }
    }
}

/// Operand role and stored precision accepted by a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelOperand {
    Activation(ActivationPrecision),
    Weight(WeightPrecision),
}

/// The one output-rounding boundary qualified by task 0012.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoundingProfile {
    FinalBf16Rne,
}

/// Hardware identity used for dispatch. Device names and ordinals are absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion {
    pub major: u32,
    pub minor: u32,
}

impl SmVersion {
    pub const SM86: Self = Self { major: 8, minor: 6 };
    pub const SM120: Self = Self {
        major: 12,
        minor: 0,
    };

    pub fn name(self) -> String {
        format!("sm_{}{}", self.major, self.minor)
    }
}

/// Checked shape domain of the initial BF16 semantic package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelShapeBounds {
    pub max_rows: u64,
    pub max_input: u64,
    pub max_output: u64,
}

/// Exact logical workspace expression, evaluated by the pure planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorkspaceExpression {
    Zero,
    RowsTimesF32,
}

impl WorkspaceExpression {
    pub fn evaluate(self, rows: u64) -> Option<u64> {
        match self {
            Self::Zero => Some(0),
            Self::RowsTimesF32 => rows.checked_mul(4),
        }
    }
}

/// Ordered symbols needed to execute one semantic node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelSymbol(pub String);

/// Immutable data selected by the pure planner. It carries no callback, image
/// pointer, model identity or device ordinal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticKernelDescriptor {
    pub id: KernelId,
    pub abi_version: u32,
    pub operation: SemanticKernelOp,
    pub inputs: Vec<KernelOperand>,
    pub output: ActivationPrecision,
    pub accumulation: AccumulationPolicy,
    pub rounding: RoundingProfile,
    pub layout: TensorLayout,
    pub shape: KernelShapeBounds,
    pub sm: SmVersion,
    pub workspace: WorkspaceExpression,
    pub image_sha256: [u8; 32],
    pub symbols: Vec<KernelSymbol>,
}

/// Read-only, closed catalogue injected into planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCatalogue {
    descriptors: Vec<SemanticKernelDescriptor>,
    digest: [u8; 32],
}

impl KernelCatalogue {
    pub fn new(mut descriptors: Vec<SemanticKernelDescriptor>) -> crate::Result<Self> {
        descriptors.sort_by(|a, b| a.id.cmp(&b.id));
        for (index, first) in descriptors.iter().enumerate() {
            for second in &descriptors[index + 1..] {
                if first.id == second.id || indistinguishable(first, second) {
                    return Err(crate::Error::InvalidArtifact {
                        detail: format!(
                            "duplicate or indistinguishable kernel descriptors {} and {}",
                            first.id.0, second.id.0
                        ),
                    });
                }
            }
        }
        let digest = catalogue_digest(&descriptors);
        Ok(Self {
            descriptors,
            digest,
        })
    }

    pub fn descriptors(&self) -> &[SemanticKernelDescriptor] {
        &self.descriptors
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

fn indistinguishable(a: &SemanticKernelDescriptor, b: &SemanticKernelDescriptor) -> bool {
    a.operation == b.operation
        && a.inputs == b.inputs
        && a.output == b.output
        && a.accumulation == b.accumulation
        && a.rounding == b.rounding
        && a.layout == b.layout
        && a.shape == b.shape
        && a.sm == b.sm
        && a.workspace == b.workspace
}

fn catalogue_digest(descriptors: &[SemanticKernelDescriptor]) -> [u8; 32] {
    // Four domain-separated FNV-1a lanes over an explicit Debug rendering.
    // This is an identity digest, not a security boundary; image integrity is
    // separately the build-produced SHA-256 in every descriptor.
    let bytes = format!("{descriptors:?}");
    let mut out = [0u8; 32];
    for lane in 0..4u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ lane.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for byte in bytes.as_bytes() {
            h ^= u64::from(*byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        out[(lane as usize) * 8..(lane as usize + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrategyControl {
    /// Never select this strategy.
    Off,
    /// Select it when it is applicable, admissible and measured to help.
    #[default]
    Auto,
    /// Select it, or fail with a typed error naming why it is unavailable.
    Required,
}

impl StrategyControl {
    /// Whether an unavailable strategy is an error rather than a fallback.
    pub const fn must_error_when_unavailable(self) -> bool {
        matches!(self, StrategyControl::Required)
    }

    /// Whether the planner is permitted to consider this strategy at all.
    pub const fn may_select(self) -> bool {
        !matches!(self, StrategyControl::Off)
    }
}

impl fmt::Display for StrategyControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StrategyControl::Off => "off",
            StrategyControl::Auto => "auto",
            StrategyControl::Required => "required",
        })
    }
}

/// What a device can actually do, from device queries and launch probes -- never
/// inferred from a marketing name (document 03).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapability {
    /// Diagnostic only. Valid inside one process's visible device set and
    /// nowhere else; a plan, a manifest or a result names the UUID (AGENTS.md).
    pub ordinal: u32,
    /// Stable identity. Evidence records this, not the ordinal (document 07).
    pub uuid: DeviceUuid,
    pub name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub total_memory_bytes: u64,
    pub multiprocessor_count: u32,
    /// PCI bus id, so a record can be reconciled with `nvidia-smi` and with the
    /// NUMA topology.
    pub pci_bus_id: String,
    /// Peer access, per peer ordinal. Discovered, never assumed: on this machine
    /// it depends on a patched kernel module that a driver upgrade can remove.
    /// See docs/evidence/topology-p2p.md.
    pub peer_access: Vec<(u32, bool)>,
}

impl DeviceCapability {
    /// `sm_86`, `sm_120`, ... Used as a kernel dispatch key, together with shape
    /// and layout. Never the device name (document 02).
    pub fn sm(&self) -> String {
        format!("sm_{}{}", self.compute_major, self.compute_minor)
    }

    pub fn can_access_peer(&self, peer: u32) -> bool {
        self.peer_access
            .iter()
            .find(|(p, _)| *p == peer)
            .map(|(_, ok)| *ok)
            .unwrap_or(false)
    }
}

/// One device's capacity as the driver reported it at one instant.
///
/// This is a **reading**, not a reservation and not a property of the card.
/// `free_bytes` includes whatever other processes hold, and re-measuring may
/// return something different; document 03 is explicit that an admission report
/// shows "physical capacity, already committed resources, reserved peak, and
/// remaining headroom", and only the first of those is a constant.
///
/// It lives here, at the bottom of the dependency graph, because the crate that
/// takes the reading sits *below* the crate that turns it into a budget:
/// document 02 permits `memory` -> `cuda` and forbids the reverse, so `cuda`
/// cannot name a `CapacitySnapshot`. Putting the descriptor in `types` lets both
/// sides use it without `cuda` reaching upward, and leaves the composition root
/// free to do the wiring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredDevice {
    /// The device's identity. Every decision and every record keys on this.
    pub uuid: DeviceUuid,
    /// The ordinal this reading was taken through, for reconciling a log with
    /// `nvidia-smi`. Never an identity, never a map key.
    pub ordinal_label: u32,
    pub name: String,
    /// `sm_86`, `sm_120`, ... A kernel dispatch key, not a marketing name.
    pub sm: String,
    pub pci_bus_id: String,
    pub multiprocessor_count: u32,
    /// Total device memory.
    pub total_bytes: u64,
    /// Free device memory at the moment of the reading, other processes
    /// included.
    pub free_bytes: u64,
}

/// Which view of host memory a reading was taken through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostLimit {
    /// No cgroup limit applies; the machine's own figures govern.
    Machine,
    /// A cgroup v2 limit binds, from this cgroup or one of its ancestors.
    ///
    /// Total and available headroom minimise independently over the chain and
    /// can bind at different levels: the smallest `memory.max` caps the total,
    /// while the smallest saturating `memory.max - memory.current` caps what
    /// a new allocation can obtain. A sibling scope holding most of an
    /// ancestor slice is the ordinary shape where they diverge, so both
    /// binding points are recorded rather than reading headroom off the
    /// tightest limit.
    Cgroup {
        /// Path of the smallest `memory.max`; binds the total.
        path: String,
        /// The smallest `memory.max` on the chain.
        limit_bytes: u64,
        /// Usage at `path`, diagnostic only; not a budget input.
        current_bytes: u64,
        /// Path of the smallest saturating `memory.max - memory.current`;
        /// binds the available figure. Equals `path` when one level binds both.
        avail_path: String,
        /// The `memory.max` at `avail_path`.
        avail_limit_bytes: u64,
        /// The `memory.current` at `avail_path`.
        avail_current_bytes: u64,
    },
}

/// The host's memory capacity as the kernel reported it at one instant.
///
/// A reading, on the same terms as [`MeasuredDevice`]: another process can take
/// memory a moment later and this will not know. Document 03's answer to that is
/// a pressure-driven replan, which is M2's.
///
/// Only `total_bytes` and `available_bytes` are budget. Everything else is a
/// diagnostic, and two of them are deliberately excluded: **swap is never
/// budget** (document 03 forbids relying on it as an invisible fourth execution
/// tier), and the machine-view pair is kept only so a report can show what a
/// cgroup limit cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredHost {
    /// Effective total, after any cgroup limit.
    pub total_bytes: u64,
    /// Effective available, after any cgroup limit. This is `MemAvailable`, not
    /// `MemFree`: the kernel's own estimate of what a new allocation can obtain
    /// without swapping, which already accounts for reclaimable page cache.
    pub available_bytes: u64,
    /// What the machine reports before any cgroup limit is applied.
    pub machine_total_bytes: u64,
    pub machine_available_bytes: u64,
    /// Completely unused memory. Small on a healthy system, and **not** the
    /// budget input -- most of a busy machine's spendable memory is reclaimable
    /// cache rather than free.
    pub free_bytes: u64,
    pub buffers_bytes: u64,
    /// Page cache. Reported so page-cache pressure is visible; managing it is
    /// M2's (document 03).
    pub cached_bytes: u64,
    /// Reported, and never in any budget.
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
    pub limit: HostLimit,
}

/// A kernel's declared applicability. The planner matches against this rather
/// than against a model name (document 02).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCapability {
    pub operation: &'static str,
    /// Compute capabilities this kernel has been *qualified* on. Document 03:
    /// SM100/Hopper recipes are not automatically compatible with SM120, and
    /// SM120 must be qualified separately from SM86.
    pub qualified_sm: Vec<String>,
    pub workspace_upper_bound_bytes: u64,
}

impl KernelCapability {
    pub fn supports(&self, device: &DeviceCapability) -> bool {
        self.qualified_sm.contains(&device.sm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(major: u32, minor: u32) -> DeviceCapability {
        DeviceCapability {
            ordinal: 0,
            uuid: DeviceUuid::from_bytes([0; 16]),
            name: "test".into(),
            compute_major: major,
            compute_minor: minor,
            total_memory_bytes: 0,
            multiprocessor_count: 0,
            pci_bus_id: "0000:00:00.0".into(),
            peer_access: vec![(1, true), (2, false)],
        }
    }

    #[test]
    fn required_errors_where_auto_falls_back() {
        assert!(StrategyControl::Required.must_error_when_unavailable());
        assert!(!StrategyControl::Auto.must_error_when_unavailable());
        assert!(!StrategyControl::Off.must_error_when_unavailable());

        assert!(StrategyControl::Auto.may_select());
        assert!(StrategyControl::Required.may_select());
        assert!(!StrategyControl::Off.may_select());
    }

    #[test]
    fn default_control_is_auto() {
        assert_eq!(StrategyControl::default(), StrategyControl::Auto);
    }

    #[test]
    fn sm120_is_not_covered_by_an_sm86_qualification() {
        // R17 / document 03: qualification is per compute capability. A kernel
        // proven on the 3090s says nothing about the 5060 Ti.
        let k = KernelCapability {
            operation: "linear_nvfp4",
            qualified_sm: vec!["sm_86".to_string()],
            workspace_upper_bound_bytes: 0,
        };
        assert!(k.supports(&dev(8, 6)));
        assert!(!k.supports(&dev(12, 0)));
    }

    #[test]
    fn peer_access_defaults_to_false_for_unknown_peers() {
        // An unprobed pair is not a usable pair.
        let d = dev(8, 6);
        assert!(d.can_access_peer(1));
        assert!(!d.can_access_peer(2));
        assert!(!d.can_access_peer(99));
    }
}
