//! A composition root that measures the host itself. Must be rejected.

pub fn host_available_bytes() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    text.len() as u64
}
