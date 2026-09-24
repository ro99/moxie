#![cfg(feature = "driver")]

use moxie_cuda::{device_count, query_device};
use moxie_executor::{ProbeConfig, probe_topology};
use moxie_plan::{Endpoint, TopologyCosts};

#[test]
fn probe_measures_every_direct_link_only() {
    let capabilities = (0..device_count().expect("enumerate CUDA devices"))
        .map(|ordinal| query_device(ordinal).expect("query CUDA device"))
        .collect::<Vec<_>>();
    let ordinals = capabilities
        .iter()
        .map(|capability| capability.ordinal)
        .collect::<Vec<_>>();
    let costs = probe_topology(
        &ordinals,
        ProbeConfig {
            small_bytes: 4 * 1024,
            bulk_bytes: 1024 * 1024,
            reps: 3,
        },
    )
    .expect("measure topology costs");
    assert_costs(&costs, &capabilities);

    println!("| From | To | Latency us | Isolated GB/s | Concurrent GB/s |");
    println!("|---|---|---:|---:|---:|");
    for link in &costs.links {
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.3} |",
            endpoint(link.from),
            endpoint(link.to),
            link.latency_us,
            link.bandwidth_gbps,
            link.concurrent_gbps,
        );
    }
    for device in &costs.devices {
        println!(
            "| `{}` memory | same device | — | {:.3} | — |",
            device.device, device.memory_gbps
        );
    }
}

fn assert_costs(costs: &TopologyCosts, capabilities: &[moxie_types::DeviceCapability]) {
    assert_eq!(costs.devices.len(), capabilities.len());
    for capability in capabilities {
        let device = costs
            .device(capability.uuid)
            .expect("one device cost per visible device");
        assert!(device.memory_gbps.is_finite() && device.memory_gbps > 0.0);
        assert!(device.usable_bytes > 0);
        assert!(
            costs
                .link(Endpoint::Host, Endpoint::Device(capability.uuid))
                .is_some()
        );
        assert!(
            costs
                .link(Endpoint::Device(capability.uuid), Endpoint::Host)
                .is_some()
        );
    }
    for from in capabilities {
        for to in capabilities {
            if from.ordinal == to.ordinal {
                continue;
            }
            // Device(a) -> Device(b) is direct exactly when b may access a.
            assert_eq!(
                costs
                    .link(Endpoint::Device(from.uuid), Endpoint::Device(to.uuid))
                    .is_some(),
                to.can_access_peer(from.ordinal),
                "{} -> {}",
                from.uuid,
                to.uuid
            );
        }
    }
    for link in &costs.links {
        assert!(link.latency_us.is_finite() && link.latency_us > 0.0);
        assert!(link.bandwidth_gbps.is_finite() && link.bandwidth_gbps > 0.0);
        assert!(link.concurrent_gbps.is_finite() && link.concurrent_gbps > 0.0);
    }
}

fn endpoint(endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::Host => "host".into(),
        Endpoint::Device(uuid) => uuid.to_string(),
    }
}
