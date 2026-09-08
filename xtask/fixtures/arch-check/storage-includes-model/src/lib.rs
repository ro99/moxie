//! A storage layer whose model knowledge arrives by include.

include!("family.rs");

pub fn chunk_name() -> &'static str {
    FAMILY_CHUNK
}
