//! Included production source: reached from the crate root by `include!`, so it
//! is production despite living in a dev-named directory.

use std::fs;

pub fn read_count() -> usize {
    fs::read("/dev/null").map(|b| b.len()).unwrap_or(0)
}
