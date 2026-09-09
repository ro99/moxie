//! A storage layer that sizes its own read buffers from the machine. Rejected.

pub fn spare_bytes() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    text.len() as u64
}
