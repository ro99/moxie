//! A storage layer sizing buffers from a relative telemetry path. Rejected.

pub fn spare_bytes(root: &std::path::Path) -> u64 {
    let text = std::fs::read_to_string(root.join("proc/meminfo")).unwrap_or_default();
    text.len() as u64
}
