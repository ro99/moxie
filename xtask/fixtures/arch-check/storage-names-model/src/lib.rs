//! A storage layer that knows what it is storing. Must be rejected.

pub const FAMILY: &str = "gemma";

pub fn chunk_for(family: &str) -> &'static str {
    if family == FAMILY {
        "gemma-chunk.bin"
    } else {
        "other.bin"
    }
}
