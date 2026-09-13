//! Measured NUMA topology: which CPUs and which memory are near which device.
//!
//! This is a **reading**, exactly like [`crate::MeasuredDevice`], and it lives
//! here for the same reason: the crate that takes the reading (`moxie-host`,
//! the one ADR 0006 permits to read machine telemetry) sits below the crate that
//! turns it into a placement (`moxie-plan`), and document 02 requires a shared
//! descriptor to be resolved into a lower-level crate rather than reached
//! upward for.
//!
//! Nothing here reads a file or names a sysfs path. It is the parsed result.
//!
//! On the benchmark machine this describes two nodes: node 0 carries the
//! 5060 Ti, node 1 carries the 3090 pair, and the node distance is 21 remote
//! against 10 local. AGENTS.md's warning applies to every field below --
//! ordinal 0 is not a 3090, and the two nodes are not interchangeable.

use core::fmt;

use crate::{Error, Result};

/// One NUMA node's identity. A small integer, but not an ordinal: it is the
/// kernel's own node number and is stable for the life of the boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumaNodeId(u32);

impl NumaNodeId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for NumaNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node{}", self.0)
    }
}

/// One node, as the machine reported it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    pub id: NumaNodeId,
    /// The CPUs of this node, ascending. Never empty: a memory-only node cannot
    /// host a first-touch allocation, so it is rejected rather than represented
    /// as a node a plan could place work on.
    pub cpus: Vec<u32>,
    pub total_bytes: u64,
    pub free_bytes: u64,
}

/// Where a host buffer is placed.
///
/// `Unspecified` is a real answer, not a missing one: a device whose node the
/// machine does not report gets no placement, and the executor then allocates
/// without binding and **says so**. Defaulting it to node 0 would turn an
/// unknown into a confident wrong answer on a machine where node 0 is across
/// the socket from two of the three GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HostPlacement {
    Unspecified,
    Node(NumaNodeId),
}

impl HostPlacement {
    pub const fn node(self) -> Option<NumaNodeId> {
        match self {
            Self::Unspecified => None,
            Self::Node(id) => Some(id),
        }
    }
}

impl fmt::Display for HostPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unspecified => f.write_str("unplaced"),
            Self::Node(id) => write!(f, "{id}"),
        }
    }
}

/// The machine's nodes plus the node of each PCI device that was asked about.
///
/// Devices are keyed by **PCI bus id**, which is what [`crate::DeviceCapability`]
/// already carries and what reconciles a CUDA device with the operating system's
/// view of it. A UUID would not: the kernel has never heard of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaTopology {
    nodes: Vec<NumaNode>,
    /// Ascending by bus id. The value is `None` when the machine reports no node
    /// for that device, which is a distinct fact from the device being absent.
    devices: Vec<(String, Option<NumaNodeId>)>,
}

impl NumaTopology {
    /// Validate and take ownership of one reading.
    ///
    /// Rejected, because each would make a later placement silently wrong: an
    /// empty node set, a duplicate node id, a node with no CPUs, a CPU claimed
    /// by two nodes, a duplicate device entry, and a device whose node is not in
    /// the node set.
    pub fn new(
        mut nodes: Vec<NumaNode>,
        mut devices: Vec<(String, Option<NumaNodeId>)>,
    ) -> Result<Self> {
        if nodes.is_empty() {
            return Err(invalid("nodes", "a machine with no NUMA node".to_string()));
        }
        nodes.sort_by_key(|n| n.id);
        for pair in nodes.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(invalid("nodes", format!("duplicate {}", pair[0].id)));
            }
        }
        let mut seen_cpus: Vec<(u32, NumaNodeId)> = Vec::new();
        for node in &mut nodes {
            if node.cpus.is_empty() {
                return Err(invalid(
                    "cpus",
                    format!("{} has no CPU; nothing can first-touch on it", node.id),
                ));
            }
            node.cpus.sort_unstable();
            for pair in node.cpus.windows(2) {
                if pair[0] == pair[1] {
                    return Err(invalid(
                        "cpus",
                        format!("{} lists CPU {} twice", node.id, pair[0]),
                    ));
                }
            }
            for cpu in &node.cpus {
                if let Some((_, other)) = seen_cpus.iter().find(|(c, _)| c == cpu) {
                    return Err(invalid(
                        "cpus",
                        format!("CPU {cpu} is claimed by {other} and by {}", node.id),
                    ));
                }
                seen_cpus.push((*cpu, node.id));
            }
        }
        devices.sort_by(|a, b| a.0.cmp(&b.0));
        for pair in devices.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(invalid("devices", format!("duplicate {:?}", pair[0].0)));
            }
        }
        for (bus, node) in &devices {
            if let Some(id) = node
                && !nodes.iter().any(|n| n.id == *id)
            {
                return Err(invalid(
                    "devices",
                    format!("{bus:?} names {id}, which is not an online node"),
                ));
            }
        }
        Ok(Self { nodes, devices })
    }

    pub fn nodes(&self) -> &[NumaNode] {
        &self.nodes
    }

    pub fn node(&self, id: NumaNodeId) -> Option<&NumaNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// The CPUs to bind to in order to first-touch on `id`.
    pub fn cpus_of(&self, id: NumaNodeId) -> Option<&[u32]> {
        self.node(id).map(|n| n.cpus.as_slice())
    }

    /// Where a PCI device's memory is local.
    ///
    /// `None` covers both "this device was not in the reading" and "the machine
    /// reports no node for it". Both mean the same thing to a placement -- there
    /// is nothing to be near -- and a plan that distinguished them would be
    /// choosing differently on the strength of a distinction it cannot act on.
    pub fn node_of_pci(&self, bus_id: &str) -> Option<NumaNodeId> {
        self.devices
            .iter()
            .find(|(bus, _)| bus.eq_ignore_ascii_case(bus_id))
            .and_then(|(_, node)| *node)
    }

    /// Placement for a device, which is the plan's actual question.
    pub fn placement_for_pci(&self, bus_id: &str) -> HostPlacement {
        match self.node_of_pci(bus_id) {
            Some(id) => HostPlacement::Node(id),
            None => HostPlacement::Unspecified,
        }
    }

    pub fn devices(&self) -> &[(String, Option<NumaNodeId>)] {
        &self.devices
    }
}

fn invalid(field: &'static str, detail: String) -> Error {
    Error::InvalidRequest { field, detail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u32, cpus: &[u32]) -> NumaNode {
        NumaNode {
            id: NumaNodeId::new(id),
            cpus: cpus.to_vec(),
            total_bytes: 1 << 37,
            free_bytes: 1 << 36,
        }
    }

    fn machine() -> NumaTopology {
        NumaTopology::new(
            vec![node(0, &[0, 1]), node(1, &[2, 3])],
            vec![
                ("0000:03:00.0".into(), Some(NumaNodeId::new(0))),
                ("0000:82:00.0".into(), Some(NumaNodeId::new(1))),
                ("0000:83:00.0".into(), Some(NumaNodeId::new(1))),
            ],
        )
        .expect("valid")
    }

    #[test]
    fn a_device_resolves_to_its_own_node_not_to_ordinal_zero() {
        let t = machine();
        assert_eq!(
            t.placement_for_pci("0000:03:00.0"),
            HostPlacement::Node(NumaNodeId::new(0))
        );
        assert_eq!(
            t.placement_for_pci("0000:82:00.0"),
            HostPlacement::Node(NumaNodeId::new(1))
        );
        assert_eq!(
            t.placement_for_pci("0000:83:00.0"),
            HostPlacement::Node(NumaNodeId::new(1))
        );
    }

    #[test]
    fn an_unknown_or_nodeless_device_is_unplaced_rather_than_node_zero() {
        let t = NumaTopology::new(
            vec![node(0, &[0, 1]), node(1, &[2, 3])],
            vec![("0000:04:00.0".into(), None)],
        )
        .expect("valid");
        assert_eq!(
            t.placement_for_pci("0000:04:00.0"),
            HostPlacement::Unspecified
        );
        assert_eq!(
            t.placement_for_pci("0000:99:00.0"),
            HostPlacement::Unspecified
        );
        assert_eq!(HostPlacement::Unspecified.node(), None);
    }

    #[test]
    fn bus_ids_compare_without_case() {
        assert_eq!(
            machine().node_of_pci("0000:83:00.0".to_uppercase().as_str()),
            Some(NumaNodeId::new(1))
        );
    }

    #[test]
    fn a_node_with_no_cpu_is_refused() {
        let err = NumaTopology::new(vec![node(0, &[])], Vec::new()).unwrap_err();
        assert!(format!("{err}").contains("no CPU"), "{err}");
    }

    #[test]
    fn a_cpu_claimed_twice_is_refused() {
        let err =
            NumaTopology::new(vec![node(0, &[0, 1]), node(1, &[1, 2])], Vec::new()).unwrap_err();
        assert!(format!("{err}").contains("claimed by"), "{err}");
    }

    #[test]
    fn a_device_naming_an_offline_node_is_refused() {
        let err = NumaTopology::new(
            vec![node(0, &[0, 1])],
            vec![("0000:03:00.0".into(), Some(NumaNodeId::new(7)))],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("not an online node"), "{err}");
    }

    #[test]
    fn duplicates_are_refused_on_both_axes() {
        assert!(NumaTopology::new(vec![node(0, &[0]), node(0, &[1])], Vec::new()).is_err());
        assert!(
            NumaTopology::new(
                vec![node(0, &[0])],
                vec![("0000:03:00.0".into(), None), ("0000:03:00.0".into(), None)]
            )
            .is_err()
        );
    }

    #[test]
    fn an_empty_machine_is_refused() {
        assert!(NumaTopology::new(Vec::new(), Vec::new()).is_err());
    }
}
