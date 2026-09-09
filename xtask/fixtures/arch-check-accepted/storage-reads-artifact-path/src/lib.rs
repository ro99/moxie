//! A storage crate reading its own artifact path, not telemetry. Accepted.

pub fn manifest_len() -> u64 {
    let text = std::fs::read_to_string("artifacts/manifest.toml").unwrap_or_default();
    text.len() as u64
}
