//! The host sensor reading this machine's telemetry, which is its whole job.

pub fn total_kb() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let limit = std::fs::read_to_string("/sys/fs/cgroup/memory.max").unwrap_or_default();
    (text.len() + limit.len()) as u64
}
