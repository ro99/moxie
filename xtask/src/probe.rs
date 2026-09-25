//! `cargo xtask probe` -- bounded hardware and topology inventory.
//!
//! Document 06 M0.2: capture device identity, capabilities, peer access and
//! topology "using bounded probes; do not consume all RAM or convert/download
//! models". Nothing here allocates a large fraction of any tier.

use std::{fmt::Write as _, path::Path};

use moxie_cuda::{DeviceBuffer, RankContext, query_device};
use moxie_executor::{Direction, PinnedLink, ProbeConfig, probe_pinned, probe_topology};
use moxie_plan::{DeviceCost, Endpoint, LinkCost, TopologyCosts};
use moxie_types::{DeviceUuid, RankId, Result};

/// Bytes moved per host-to-device timing sample. 64 MiB is large enough to leave
/// per-call overhead behind and small enough to be harmless on a 16 GiB card.
const XFER_BYTES: usize = 64 * 1024 * 1024;
const XFER_REPS: usize = 5;

pub fn run(out: Option<&str>, costs_path: Option<&str>) -> i32 {
    let count = match moxie_cuda::device_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("cannot enumerate devices: {e}");
            return 2;
        }
    };

    let mut md = String::new();
    let _ = writeln!(
        md,
        "| Ordinal | Name | SM | UUID | VRAM MiB | SMs | PCI bus |"
    );
    let _ = writeln!(md, "|---|---|---|---|---|---|---|");

    let mut caps = Vec::new();
    for i in 0..count {
        match query_device(i) {
            Ok(c) => {
                let _ = writeln!(
                    md,
                    "| {} | {} | {} | `{}` | {} | {} | `{}` |",
                    c.ordinal,
                    c.name,
                    c.sm(),
                    c.uuid,
                    c.total_memory_bytes / (1024 * 1024),
                    c.multiprocessor_count,
                    c.pci_bus_id
                );
                caps.push(c);
            }
            Err(e) => {
                eprintln!("device {i}: {e}");
                return 1;
            }
        }
    }

    let _ = writeln!(md, "\n### Peer access (driver `cuDeviceCanAccessPeer`)\n");
    let _ = write!(md, "| from \\ to |");
    for c in &caps {
        let _ = write!(md, " {} |", c.ordinal);
    }
    let _ = writeln!(md);
    let _ = write!(md, "|---|");
    for _ in &caps {
        let _ = write!(md, "---|");
    }
    let _ = writeln!(md);
    for c in &caps {
        let _ = write!(md, "| **{}** |", c.ordinal);
        for peer in &caps {
            if peer.ordinal == c.ordinal {
                let _ = write!(md, " -- |");
            } else if c.can_access_peer(peer.ordinal) {
                let _ = write!(md, " yes |");
            } else {
                let _ = write!(md, " no |");
            }
        }
        let _ = writeln!(md);
    }

    let _ = writeln!(
        md,
        "\n### Host-to-device, pageable source, {} MiB x {} reps\n",
        XFER_BYTES / (1024 * 1024),
        XFER_REPS
    );
    let _ = writeln!(md, "| Ordinal | Median GB/s | Min | Max |");
    let _ = writeln!(md, "|---|---|---|---|");
    for c in &caps {
        match measure_h2d(c.ordinal) {
            Ok((med, lo, hi)) => {
                let _ = writeln!(md, "| {} | {med:.2} | {lo:.2} | {hi:.2} |", c.ordinal);
            }
            Err(e) => {
                let _ = writeln!(md, "| {} | failed: {e} | | |", c.ordinal);
            }
        }
    }

    let _ = writeln!(
        md,
        "\n`--costs <path>` adds measured pageable-link, direct-peer, simultaneous-traffic and device-memory costs for deterministic planner comparisons."
    );

    if let Some(path) = costs_path {
        let ordinals: Vec<_> = caps.iter().map(|capability| capability.ordinal).collect();
        let costs = match probe_topology(&ordinals, ProbeConfig::default()) {
            Ok(costs) => costs,
            Err(error) => {
                eprintln!("topology cost probe failed: {error}");
                return 1;
            }
        };
        append_costs(&mut md, &costs);
        if let Err(error) = write_costs(path, &costs) {
            eprintln!("cannot write topology costs {path}: {error}");
            return 1;
        }
        eprintln!("wrote topology costs {path}");
    }

    let ordinals: Vec<_> = caps.iter().map(|capability| capability.ordinal).collect();
    let pinned = match probe_pinned(&ordinals, ProbeConfig::default()) {
        Ok(pinned) => pinned,
        Err(error) => {
            eprintln!("pinned transfer probe failed: {error}");
            return 1;
        }
    };
    append_pinned(&mut md, &pinned);

    print!("{md}");
    if let Some(path) = out {
        if let Err(e) = std::fs::write(path, &md) {
            eprintln!("cannot write {path}: {e}");
            return 1;
        }
        eprintln!("wrote {path}");
    }
    0
}

fn append_pinned(md: &mut String, links: &[PinnedLink]) {
    let _ = writeln!(
        md,
        "\n### Pinned host transfers, {} MiB x {} reps\n",
        XFER_BYTES / (1024 * 1024),
        XFER_REPS
    );
    let _ = writeln!(
        md,
        "| Device UUID | Direction | Pageable GB/s | Pinned GB/s | Pageable issue us | Pinned issue us | Overlap |"
    );
    let _ = writeln!(md, "|---|---|---:|---:|---:|---:|---:|");
    for link in links {
        let direction = match link.direction {
            Direction::H2d => "host-to-device",
            Direction::D2h => "device-to-host",
        };
        let _ = writeln!(
            md,
            "| `{}` | {direction} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            link.device,
            link.pageable_gbps,
            link.pinned_gbps,
            link.pageable_issue_us,
            link.pinned_issue_us,
            link.overlap,
        );
    }
}

fn append_costs(md: &mut String, costs: &TopologyCosts) {
    let _ = writeln!(md, "\n### Topology costs\n");
    let _ = writeln!(
        md,
        "| Kind | From | To | Small latency (us) | Isolated (GB/s) | Concurrent (GB/s) | Linear TFLOP/s | Usable bytes |"
    );
    let _ = writeln!(md, "|---|---|---|---:|---:|---:|---:|---:|");
    for device in &costs.devices {
        let _ = writeln!(
            md,
            "| Device memory | `{}` | same device | — | {:.3} | — | {:.3} | {} |",
            device.device, device.memory_gbps, device.linear_tflops, device.usable_bytes
        );
    }
    for link in &costs.links {
        let _ = writeln!(
            md,
            "| Link | {} | {} | {:.3} | {:.3} | {:.3} | — | — |",
            endpoint(link.from),
            endpoint(link.to),
            link.latency_us,
            link.bandwidth_gbps,
            link.concurrent_gbps
        );
    }
}

fn endpoint(endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::Host => "host".into(),
        Endpoint::Device(device) => format!("`{device}`"),
    }
}

fn write_costs(path: impl AsRef<Path>, costs: &TopologyCosts) -> Result<()> {
    let mut root = toml::Table::new();
    let mut device_rows = Vec::with_capacity(costs.devices.len());
    for device in &costs.devices {
        let usable_bytes = i64::try_from(device.usable_bytes).map_err(|_| {
            cost_field_error(
                "topology_costs.device.usable_bytes",
                "value exceeds TOML's signed integer range",
            )
        })?;
        device_rows.push(toml::Value::Table(toml::Table::from_iter([
            (
                "uuid".into(),
                toml::Value::String(device.device.to_string()),
            ),
            ("memory_gbps".into(), toml::Value::Float(device.memory_gbps)),
            (
                "linear_tflops".into(),
                toml::Value::Float(device.linear_tflops),
            ),
            ("usable_bytes".into(), toml::Value::Integer(usable_bytes)),
        ])));
    }
    let mut link_rows = Vec::with_capacity(costs.links.len());
    for link in &costs.links {
        link_rows.push(toml::Value::Table(toml::Table::from_iter([
            (
                "from".into(),
                toml::Value::String(endpoint_string(link.from)),
            ),
            ("to".into(), toml::Value::String(endpoint_string(link.to))),
            ("latency_us".into(), toml::Value::Float(link.latency_us)),
            (
                "bandwidth_gbps".into(),
                toml::Value::Float(link.bandwidth_gbps),
            ),
            (
                "concurrent_gbps".into(),
                toml::Value::Float(link.concurrent_gbps),
            ),
        ])));
    }
    root.insert("device".into(), toml::Value::Array(device_rows));
    root.insert("link".into(), toml::Value::Array(link_rows));
    let text = toml::to_string(&root).map_err(|error| invalid_costs(error.to_string()))?;
    std::fs::write(path, text).map_err(|error| invalid_costs(error.to_string()))
}

/// Read a topology-cost TOML file written by the probe.
///
/// Task 0074 consumes this reader when it compares plan candidates.
#[allow(dead_code)]
pub fn read_costs(path: impl AsRef<Path>) -> Result<TopologyCosts> {
    let text = std::fs::read_to_string(path).map_err(|error| invalid_costs(error.to_string()))?;
    let root: toml::Table = text
        .parse()
        .map_err(|error: toml::de::Error| invalid_costs(error.to_string()))?;
    let devices = required_array(&root, "device", "topology_costs.device")?;
    let links = required_array(&root, "link", "topology_costs.link")?;
    Ok(TopologyCosts {
        devices: devices
            .iter()
            .map(|device| {
                let device = device.as_table().ok_or_else(|| {
                    cost_field_error("topology_costs.device", "entry must be a table")
                })?;
                let uuid = required_string(device, "uuid", "topology_costs.device.uuid")?;
                let uuid = DeviceUuid::parse(uuid).map_err(|error| {
                    cost_field_error("topology_costs.device.uuid", error.to_string())
                })?;
                let memory_gbps =
                    required_float(device, "memory_gbps", "topology_costs.device.memory_gbps")?;
                let linear_tflops = required_float(
                    device,
                    "linear_tflops",
                    "topology_costs.device.linear_tflops",
                )?;
                let usable_bytes =
                    required_integer(device, "usable_bytes", "topology_costs.device.usable_bytes")?;
                let usable_bytes = u64::try_from(usable_bytes).map_err(|_| {
                    cost_field_error(
                        "topology_costs.device.usable_bytes",
                        "value must be nonnegative",
                    )
                })?;
                Ok(DeviceCost {
                    device: uuid,
                    memory_gbps,
                    linear_tflops,
                    usable_bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?,
        links: links
            .iter()
            .map(|link| {
                let link = link.as_table().ok_or_else(|| {
                    cost_field_error("topology_costs.link", "entry must be a table")
                })?;
                let from = required_string(link, "from", "topology_costs.link.from")?;
                let to = required_string(link, "to", "topology_costs.link.to")?;
                let latency_us =
                    required_float(link, "latency_us", "topology_costs.link.latency_us")?;
                let bandwidth_gbps =
                    required_float(link, "bandwidth_gbps", "topology_costs.link.bandwidth_gbps")?;
                let concurrent_gbps = required_float(
                    link,
                    "concurrent_gbps",
                    "topology_costs.link.concurrent_gbps",
                )?;
                Ok(LinkCost {
                    from: parse_endpoint(from, "topology_costs.link.from")?,
                    to: parse_endpoint(to, "topology_costs.link.to")?,
                    latency_us,
                    bandwidth_gbps,
                    concurrent_gbps,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

fn required_array<'a>(
    table: &'a toml::Table,
    name: &str,
    field: &'static str,
) -> Result<&'a [toml::Value]> {
    table
        .get(name)
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| cost_field_error(field, "missing or mistyped array"))
}

fn required_string<'a>(table: &'a toml::Table, name: &str, field: &'static str) -> Result<&'a str> {
    table
        .get(name)
        .and_then(toml::Value::as_str)
        .ok_or_else(|| cost_field_error(field, "missing or mistyped string"))
}

fn required_float(table: &toml::Table, name: &str, field: &'static str) -> Result<f64> {
    table
        .get(name)
        .and_then(toml::Value::as_float)
        .ok_or_else(|| cost_field_error(field, "missing or mistyped float"))
}

fn required_integer(table: &toml::Table, name: &str, field: &'static str) -> Result<i64> {
    table
        .get(name)
        .and_then(toml::Value::as_integer)
        .ok_or_else(|| cost_field_error(field, "missing or mistyped integer"))
}

fn endpoint_string(endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::Host => "host".into(),
        Endpoint::Device(device) => device.to_string(),
    }
}

fn parse_endpoint(value: &str, field: &'static str) -> Result<Endpoint> {
    if value == "host" {
        Ok(Endpoint::Host)
    } else {
        DeviceUuid::parse(value)
            .map(Endpoint::Device)
            .map_err(|error| cost_field_error(field, error.to_string()))
    }
}

fn invalid_costs(detail: String) -> moxie_types::Error {
    moxie_types::Error::InvalidRequest {
        field: "topology_costs",
        detail,
    }
}

fn cost_field_error(field: &'static str, detail: impl Into<String>) -> moxie_types::Error {
    moxie_types::Error::InvalidRequest {
        field,
        detail: detail.into(),
    }
}

fn measure_h2d(ordinal: u32) -> Result<(f64, f64, f64)> {
    let ctx = RankContext::acquire(RankId(ordinal), ordinal)?;
    let host = vec![0u8; XFER_BYTES];
    let mut dev = DeviceBuffer::alloc(&ctx, XFER_BYTES)?;

    // One warm-up: the first copy pays context and mapping costs that are not
    // part of the steady-state rate.
    dev.copy_from_host(&host)?;

    let mut rates = Vec::with_capacity(XFER_REPS);
    for _ in 0..XFER_REPS {
        let t = std::time::Instant::now();
        dev.copy_from_host(&host)?;
        let secs = t.elapsed().as_secs_f64();
        rates.push((XFER_BYTES as f64) / secs / 1e9);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((rates[rates.len() / 2], rates[0], rates[rates.len() - 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_costs_read_back_equal() {
        let uuid = DeviceUuid::parse("GPU-00000000-0000-0000-0000-000000000001").unwrap();
        let costs = TopologyCosts {
            devices: vec![DeviceCost {
                device: uuid,
                memory_gbps: 731.25,
                linear_tflops: 17.5,
                usable_bytes: 12_345,
            }],
            links: vec![LinkCost {
                from: Endpoint::Host,
                to: Endpoint::Device(uuid),
                latency_us: 8.5,
                bandwidth_gbps: 12.25,
                concurrent_gbps: 9.75,
            }],
        };
        let path = std::env::temp_dir().join(format!(
            "moxie-topology-costs-{}-{}.toml",
            std::process::id(),
            uuid
        ));
        write_costs(&path, &costs).unwrap();
        assert_eq!(read_costs(&path).unwrap(), costs);
        let text = std::fs::read_to_string(&path).unwrap();
        let legacy = text
            .lines()
            .filter(|line| !line.starts_with("linear_tflops = "))
            .collect::<Vec<_>>()
            .join("\n");
        let legacy_path = path.with_extension("legacy.toml");
        std::fs::write(&legacy_path, legacy).unwrap();
        assert!(matches!(
            read_costs(&legacy_path),
            Err(moxie_types::Error::InvalidRequest {
                field: "topology_costs.device.linear_tflops",
                ..
            })
        ));
        std::fs::remove_file(legacy_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
