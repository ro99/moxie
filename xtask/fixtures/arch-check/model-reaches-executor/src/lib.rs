//! A model adapter reaching the execution lease authority. Must be rejected.
//!
//! The dependency alone is the breach: document 02 denies model crates rank-
//! local execution, and the checker refuses the edge without reading this file.

pub fn placeholder() -> u64 {
    0
}
