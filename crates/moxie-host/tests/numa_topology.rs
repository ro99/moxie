//! Acceptance tests for task 0021's NUMA reading.
//!
//! Two kinds of case, deliberately. Most read a committed fixture tree, so a
//! four-node machine and a device the kernel gives no node are exercised on a
//! machine that has neither. One reads `/` and checks that what this machine
//! says agrees with `docs/evidence/hardware-inventory.md` -- the recorded
//! evidence is then reproduced rather than trusted, and it fails loudly if the
//! machine is re-cabled.

use std::path::{Path, PathBuf};

use moxie_types::{HostPlacement, NumaNodeId};

fn tree(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn buses(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn a_two_node_fixture_yields_cpus_memory_and_device_placement() {
    let t = moxie_host::numa::read_topology_under(
        &tree("numa-two-nodes"),
        &buses(&["0000:03:00.0", "0000:82:00.0", "0000:83:00.0"]),
    )
    .expect("fixture is well formed");

    assert_eq!(t.nodes().len(), 2);
    assert_eq!(
        t.cpus_of(NumaNodeId::new(0)).unwrap(),
        &[0, 1, 2, 3, 8, 9, 10, 11]
    );
    assert_eq!(
        t.cpus_of(NumaNodeId::new(1)).unwrap(),
        &[4, 5, 6, 7, 12, 13, 14, 15]
    );
    assert_eq!(
        t.node(NumaNodeId::new(1)).unwrap().total_bytes,
        132_107_112 * 1024
    );
    assert_eq!(
        t.node(NumaNodeId::new(1)).unwrap().free_bytes,
        399_564 * 1024
    );

    // The shape that matters on this machine: one device on one node and two on
    // the other. A reader that returned the first node for everything would
    // pass a single-node fixture and fail here.
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
fn a_device_the_kernel_gives_no_node_is_unplaced_and_recorded_as_asked_about() {
    let t = moxie_host::numa::read_topology_under(
        &tree("numa-nodeless-device"),
        &buses(&["0000:03:00.0", "0000:04:00.0"]),
    )
    .expect("fixture is well formed");
    assert_eq!(
        t.placement_for_pci("0000:03:00.0"),
        HostPlacement::Unspecified
    );
    // Both are recorded: one was asked about and answered `-1`, one was asked
    // about and is absent. The placement is the same and the record is not.
    assert_eq!(t.devices().len(), 2);
    assert!(t.devices().iter().all(|(_, node)| node.is_none()));
}

#[test]
fn a_numa_node_that_is_neither_a_node_nor_minus_one_is_refused() {
    let e =
        moxie_host::numa::read_topology_under(&tree("numa-bad-node"), &buses(&["0000:03:00.0"]))
            .expect_err("-2 is not an answer");
    assert!(format!("{e}").contains("neither a node nor -1"), "{e}");
}

#[test]
fn four_nodes_are_read_and_the_device_lands_on_the_last_one() {
    let t =
        moxie_host::numa::read_topology_under(&tree("numa-four-nodes"), &buses(&["0000:c1:00.0"]))
            .expect("fixture is well formed");
    assert_eq!(t.nodes().len(), 4);
    assert_eq!(
        t.placement_for_pci("0000:c1:00.0"),
        HostPlacement::Node(NumaNodeId::new(3))
    );
    assert_eq!(t.cpus_of(NumaNodeId::new(3)).unwrap(), &[12, 13, 14, 15]);
}

/// The recorded topology of the benchmark machine, reproduced rather than
/// trusted. `docs/evidence/hardware-inventory.md` records the 5060 Ti on node 0
/// and the 3090 pair on node 1; AGENTS.md repeats it because assuming otherwise
/// is a real mistake here.
#[test]
fn this_machine_matches_the_recorded_hardware_inventory() {
    let root = Path::new("/");
    if !root.join("sys/devices/system/node/online").exists() {
        eprintln!("SKIPPED: this kernel exposes no NUMA node list");
        return;
    }
    let t = moxie_host::numa::read_topology_under(
        root,
        &buses(&["0000:03:00.0", "0000:82:00.0", "0000:83:00.0"]),
    )
    .expect("this machine answers");

    assert_eq!(t.nodes().len(), 2, "the inventory records two nodes");
    for node in t.nodes() {
        assert!(!node.cpus.is_empty());
        assert!(node.total_bytes > 0);
        assert!(node.free_bytes <= node.total_bytes);
    }
    let total: u64 = t.nodes().iter().map(|n| n.total_bytes).sum();
    // ~251 GiB across two nodes, per the inventory. A wide band: this asserts
    // the reading is the machine's, not a figure to plan against.
    assert!(
        (200 << 30..300u64 << 30).contains(&total),
        "two nodes total {total} B"
    );
    let cpus: usize = t.nodes().iter().map(|n| n.cpus.len()).sum();
    assert_eq!(cpus, 56, "2 x 14 cores, 56 threads");

    assert_eq!(
        t.placement_for_pci("0000:03:00.0"),
        HostPlacement::Node(NumaNodeId::new(0)),
        "the 5060 Ti is on node 0"
    );
    assert_eq!(
        t.placement_for_pci("0000:82:00.0"),
        HostPlacement::Node(NumaNodeId::new(1)),
        "the first 3090 is on node 1"
    );
    assert_eq!(
        t.placement_for_pci("0000:83:00.0"),
        HostPlacement::Node(NumaNodeId::new(1)),
        "the second 3090 is on node 1"
    );
}

/// `numa_maps` answers for this process, and an address in a freshly touched
/// allocation resolves to a mapping. This is the read-back the placement
/// evidence depends on; if it stopped working, a placement claim would quietly
/// become unverifiable rather than fail.
#[test]
fn a_touched_allocation_resolves_to_a_mapping_in_numa_maps() {
    let root = Path::new("/");
    if !root.join("proc/self/numa_maps").exists() {
        eprintln!("SKIPPED: this kernel exposes no numa_maps");
        return;
    }
    // Large enough that the allocator maps it on its own rather than carving it
    // out of an existing arena, and touched so it has resident pages at all.
    let mut buffer = vec![0u8; 8 << 20];
    for page in buffer.chunks_mut(4096) {
        page[0] = 1;
    }
    let address = buffer.as_ptr() as u64;
    let mapping = moxie_host::numa::node_of_address_under(root, address)
        .expect("numa_maps is readable")
        .expect("an address inside the process resolves to a mapping");
    assert!(mapping.start <= address);
    assert!(
        mapping.total_pages() > 0,
        "a touched mapping has resident pages: {mapping:?}"
    );
    assert!(mapping.dominant().is_some(), "{mapping:?}");
    std::hint::black_box(&buffer);
}
