//! Reading this machine's NUMA topology, and reading back where pages landed.
//!
//! ADR 0006 makes this crate the one reader of machine telemetry, and `arch-check`
//! enforces it, so the sysfs and procfs paths below may exist here and nowhere
//! else. Everything in this module reads and reports: it binds nothing, places
//! nothing and decides nothing. Turning a node into a placement is
//! `moxie-plan`'s, and performing the placement is `moxie-executor`'s.
//!
//! Two different readings live here and they answer two different questions.
//! [`read_topology_under`] answers "which CPUs and which memory are near this
//! device", which a plan needs *before* it allocates. [`node_of_address_under`]
//! answers "where did these pages actually land", which is the only honest way
//! to make a NUMA claim afterwards -- AGENTS.md records three task-0020 claims
//! that were asserted instead of measured, and a placement is exactly the kind
//! of property that looks obviously true and silently is not.
//!
//! As everywhere in this crate, the root is injectable, so a topology this
//! machine does not have is still exercised.

use std::path::Path;

use moxie_types::{NumaNode, NumaNodeId, NumaTopology, Result};

use crate::{invalid, slurp};

/// Read the node set, each node's CPUs and memory, and the node of each PCI
/// device named in `pci_bus_ids`, under `/`.
pub fn read_topology(pci_bus_ids: &[String]) -> Result<NumaTopology> {
    read_topology_under(Path::new("/"), pci_bus_ids)
}

/// Read the topology under `root`.
///
/// A device that is absent, or whose node the kernel reports as `-1`, becomes
/// `(bus, None)`. Both are recorded rather than dropped: a plan that cannot see
/// the difference between "asked and got no answer" and "never asked" cannot
/// report why a buffer went unplaced.
pub fn read_topology_under(root: &Path, pci_bus_ids: &[String]) -> Result<NumaTopology> {
    let node_root = root.join("sys/devices/system/node");
    let online = slurp(&node_root.join("online"), "numa online")?;
    let ids = parse_cpu_list(online.trim(), "numa online")?;
    let mut nodes = Vec::with_capacity(ids.len());
    for id in ids {
        let dir = node_root.join(format!("node{id}"));
        let cpus = parse_cpu_list(
            slurp(&dir.join("cpulist"), "node cpulist")?.trim(),
            "cpulist",
        )?;
        let meminfo = slurp(&dir.join("meminfo"), "node meminfo")?;
        let (total_bytes, free_bytes) = parse_node_meminfo(&meminfo, id)?;
        nodes.push(NumaNode {
            id: NumaNodeId::new(id),
            cpus,
            total_bytes,
            free_bytes,
        });
    }

    let mut devices = Vec::with_capacity(pci_bus_ids.len());
    for bus in pci_bus_ids {
        let path = root.join("sys/bus/pci/devices").join(bus).join("numa_node");
        let node = match std::fs::read_to_string(&path) {
            Ok(text) => {
                let raw = text.trim();
                let value: i64 = raw.parse().map_err(|e| {
                    invalid("numa_node", format!("{}: {raw:?}: {e}", path.display()))
                })?;
                match u32::try_from(value) {
                    Ok(id) => Some(NumaNodeId::new(id)),
                    // `-1` is the kernel's "no node for this device", which is
                    // an answer. Anything else negative is not, and is refused
                    // rather than folded into the same bucket.
                    Err(_) if value == -1 => None,
                    Err(_) => {
                        return Err(invalid(
                            "numa_node",
                            format!("{}: {value} is neither a node nor -1", path.display()),
                        ));
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(invalid("numa_node", format!("{}: {e}", path.display()))),
        };
        devices.push((bus.clone(), node));
    }

    NumaTopology::new(nodes, devices)
}

/// Where one mapping's pages physically are, read back after the fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingNodes {
    /// The mapping's start address, as the kernel reports it.
    pub start: u64,
    /// Resident pages per node, ascending by node. Empty when the kernel
    /// reported no per-node counts for the mapping, which happens for a mapping
    /// with no resident page yet -- an untouched allocation is not placed
    /// anywhere, and reporting a node for it would be an invention.
    pub pages: Vec<(NumaNodeId, u64)>,
}

impl MappingNodes {
    /// The node holding the most pages, and `None` for a tie or an empty
    /// reading. A tie is not a placement failure and not a placement success;
    /// it is a measurement that does not answer the question, and the caller
    /// decides what that means.
    pub fn dominant(&self) -> Option<NumaNodeId> {
        let mut best: Option<(NumaNodeId, u64)> = None;
        let mut tied = false;
        for (id, count) in &self.pages {
            match best {
                Some((_, b)) if *count < b => {}
                Some((_, b)) if *count == b => tied = true,
                _ => {
                    best = Some((*id, *count));
                    tied = false;
                }
            }
        }
        if tied { None } else { best.map(|(id, _)| id) }
    }

    pub fn total_pages(&self) -> u64 {
        self.pages.iter().map(|(_, c)| *c).sum()
    }
}

/// Read this process's own mapping that contains `address`, under `/`.
pub fn node_of_address(address: u64) -> Result<Option<MappingNodes>> {
    node_of_address_under(Path::new("/"), address)
}

/// Read the mapping containing `address` from `numa_maps` under `root`.
///
/// `numa_maps` prints one line per mapping, keyed by its **start** address and
/// with no end, so the containing mapping is the one with the greatest start at
/// or below `address`. That is an ordering fact about the file, which the kernel
/// prints in ascending address order; this function does not assume the order
/// and takes the maximum.
pub fn node_of_address_under(root: &Path, address: u64) -> Result<Option<MappingNodes>> {
    let text = slurp(&root.join("proc/self/numa_maps"), "numa_maps")?;
    let mut best: Option<MappingNodes> = None;
    for line in text.lines() {
        let Some(entry) = parse_numa_maps_line(line)? else {
            continue;
        };
        if entry.start <= address && best.as_ref().is_none_or(|b| entry.start > b.start) {
            best = Some(entry);
        }
    }
    Ok(best)
}

fn parse_numa_maps_line(line: &str) -> Result<Option<MappingNodes>> {
    let mut fields = line.split_ascii_whitespace();
    let Some(addr) = fields.next() else {
        return Ok(None);
    };
    let Ok(start) = u64::from_str_radix(addr, 16) else {
        // Not an address-led line. The file has no such line today; skipping
        // rather than failing keeps a future kernel's extra header from making
        // a placement unverifiable.
        return Ok(None);
    };
    let mut pages = Vec::new();
    for field in fields {
        let Some(rest) = field.strip_prefix('N') else {
            continue;
        };
        let Some((node, count)) = rest.split_once('=') else {
            continue;
        };
        let (Ok(node), Ok(count)) = (node.parse::<u32>(), count.parse::<u64>()) else {
            continue;
        };
        pages.push((NumaNodeId::new(node), count));
    }
    pages.sort_by_key(|(id, _)| *id);
    Ok(Some(MappingNodes { start, pages }))
}

/// `"0-13,28-41"` -> `[0..=13, 28..=41]`, and `""` -> `[]`.
///
/// Every element is validated. A range whose end precedes its start is refused:
/// silently producing an empty node would make a machine look like it has fewer
/// CPUs than it does, and a placement would then be legal and wrong.
fn parse_cpu_list(text: &str, what: &'static str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    if text.is_empty() {
        return Ok(out);
    }
    for piece in text.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            return Err(invalid(what, format!("empty element in {text:?}")));
        }
        let (lo, hi) = match piece.split_once('-') {
            Some((a, b)) => (a, b),
            None => (piece, piece),
        };
        let lo: u32 = lo
            .parse()
            .map_err(|e| invalid(what, format!("{piece:?} in {text:?}: {e}")))?;
        let hi: u32 = hi
            .parse()
            .map_err(|e| invalid(what, format!("{piece:?} in {text:?}: {e}")))?;
        if hi < lo {
            return Err(invalid(what, format!("{piece:?} runs backwards")));
        }
        out.extend(lo..=hi);
    }
    Ok(out)
}

/// `Node 1 MemTotal: 132107112 kB` -> bytes, for the node it claims to be.
///
/// The node number on the line is checked against the directory it was read
/// from. They agree on every kernel; checking costs one comparison and turns a
/// mismatched reading into an error instead of a budget.
fn parse_node_meminfo(text: &str, node: u32) -> Result<(u64, u64)> {
    let mut total = None;
    let mut free = None;
    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        if fields.next() != Some("Node") {
            continue;
        }
        let Some(id) = fields.next().and_then(|v| v.parse::<u32>().ok()) else {
            continue;
        };
        if id != node {
            return Err(invalid(
                "node meminfo",
                format!("node{node}/meminfo reports node {id}"),
            ));
        }
        let Some(key) = fields.next() else { continue };
        let slot = match key {
            "MemTotal:" => &mut total,
            "MemFree:" => &mut free,
            _ => continue,
        };
        let value: u64 = fields
            .next()
            .ok_or_else(|| invalid("node meminfo", format!("{key} has no value")))?
            .parse()
            .map_err(|e| invalid("node meminfo", format!("{key}: {e}")))?;
        let unit = fields.next().unwrap_or("");
        if unit != "kB" {
            return Err(invalid(
                "node meminfo",
                format!("{key} is in {unit:?}, not kB"),
            ));
        }
        *slot = Some(value.checked_mul(1024).ok_or_else(|| {
            invalid("node meminfo", format!("{key} overflows bytes: {value} kB"))
        })?);
    }
    match (total, free) {
        (Some(t), Some(f)) => Ok((t, f)),
        _ => Err(invalid(
            "node meminfo",
            format!("node{node}/meminfo is missing MemTotal or MemFree"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_list_expands_ranges_and_singletons() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11", "t").unwrap(),
            vec![0, 1, 2, 3, 8, 10, 11]
        );
        assert_eq!(parse_cpu_list("", "t").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn a_backwards_or_empty_range_is_refused_rather_than_emptied() {
        assert!(parse_cpu_list("5-3", "t").is_err());
        assert!(parse_cpu_list("0,,2", "t").is_err());
        assert!(parse_cpu_list("0-x", "t").is_err());
    }

    #[test]
    fn node_meminfo_converts_kb_and_checks_the_node_it_claims() {
        let text = "Node 1 MemTotal:  132107112 kB\nNode 1 MemFree:   399564 kB\n";
        assert_eq!(
            parse_node_meminfo(text, 1).unwrap(),
            (132107112 * 1024, 399564 * 1024)
        );
        assert!(parse_node_meminfo(text, 0).is_err());
    }

    #[test]
    fn node_meminfo_refuses_a_unit_it_did_not_expect() {
        let text = "Node 0 MemTotal:  1024 MB\nNode 0 MemFree: 1 kB\n";
        assert!(parse_node_meminfo(text, 0).is_err());
    }

    #[test]
    fn a_numa_maps_line_yields_its_per_node_page_counts() {
        let entry = parse_numa_maps_line(
            "7f0000001000 default anon=64 dirty=64 N0=20 N1=44 kernelpagesize_kB=4",
        )
        .unwrap()
        .expect("an address line");
        assert_eq!(entry.start, 0x7f0000001000);
        assert_eq!(
            entry.pages,
            vec![(NumaNodeId::new(0), 20), (NumaNodeId::new(1), 44)]
        );
        assert_eq!(entry.total_pages(), 64);
        assert_eq!(entry.dominant(), Some(NumaNodeId::new(1)));
    }

    #[test]
    fn a_tie_has_no_dominant_node() {
        let entry = parse_numa_maps_line("7f0000001000 default N0=8 N1=8")
            .unwrap()
            .expect("an address line");
        assert_eq!(entry.dominant(), None);
    }

    #[test]
    fn a_mapping_with_no_resident_page_names_no_node() {
        let entry = parse_numa_maps_line("7f0000001000 default kernelpagesize_kB=4")
            .unwrap()
            .expect("an address line");
        assert!(entry.pages.is_empty());
        assert_eq!(entry.dominant(), None);
        assert_eq!(entry.total_pages(), 0);
    }
}
