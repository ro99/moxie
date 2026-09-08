//! A format crate that touches the filesystem. Must be rejected.

use std::fs;

pub fn read_manifest(path: &str) -> Vec<u8> {
    let _ = std::path::Path::new(path);
    fs::read(path).unwrap_or_default()
}
