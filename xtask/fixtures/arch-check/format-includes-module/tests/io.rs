//! Module production-dependency of an included file.

use std::fs;

pub fn read_count() -> usize {
    fs::read("/dev/null").map(|b| b.len()).unwrap_or(0)
}
