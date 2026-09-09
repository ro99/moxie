//! A ledger that measures its own capacity. Must be rejected.

pub fn host_physical_bytes() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    text.len() as u64
}
