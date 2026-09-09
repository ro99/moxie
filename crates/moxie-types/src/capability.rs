//! Hardware and kernel capabilities, and the shared `off` / `auto` / `required`
//! control semantics.
//!
//! Document 04: "`off`, `auto`, and `required` are common control semantics.
//! Report the selected plan and why a requested strategy was rejected." The
//! asymmetry encoded here is the whole point: `auto` may decline a strategy and
//! must say so; `required` must produce an actionable error rather than silently
//! disabling what was asked for.

use core::fmt;

use crate::ids::DeviceUuid;

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
/// takes the reading and the crate that turns it into a budget sit on opposite
/// sides of `memory` -> `cuda` and neither may import the other (document 02).
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
