//! A storage layer reading cgroup state through a relative path. Rejected.

pub fn cgroup_bytes(root: &std::path::Path) -> u64 {
    let text =
        std::fs::read_to_string(root.join("sys/fs/cgroup/memory.max")).unwrap_or_default();
    text.len() as u64
}
