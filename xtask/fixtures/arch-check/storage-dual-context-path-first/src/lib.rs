//! A storage layer loading one file through two contexts.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    alias::inner::FAMILY_CHUNK
}
