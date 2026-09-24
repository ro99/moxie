//! Measured, topology-specific costs consumed by plan comparison.

use moxie_types::DeviceUuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Endpoint {
    Host,
    Device(DeviceUuid),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinkCost {
    pub from: Endpoint,
    pub to: Endpoint,
    /// Median wall time of one small copy, in microseconds.
    pub latency_us: f64,
    /// Median sustained rate of one bulk copy with the link otherwise idle, GB/s.
    pub bandwidth_gbps: f64,
    /// Median rate of the same bulk copy while every link in its class runs at once, GB/s.
    pub concurrent_gbps: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCost {
    pub device: DeviceUuid,
    pub memory_gbps: f64,
    pub usable_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TopologyCosts {
    pub devices: Vec<DeviceCost>,
    pub links: Vec<LinkCost>,
}

impl TopologyCosts {
    pub fn link(&self, from: Endpoint, to: Endpoint) -> Option<&LinkCost> {
        self.links
            .iter()
            .find(|link| link.from == from && link.to == to)
    }

    pub fn device(&self, device: DeviceUuid) -> Option<&DeviceCost> {
        self.devices.iter().find(|cost| cost.device == device)
    }
}
