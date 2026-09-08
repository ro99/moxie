//! A storage crate loading one clean file through two contexts.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    alias::inner::DEFAULT_CHUNK
}
