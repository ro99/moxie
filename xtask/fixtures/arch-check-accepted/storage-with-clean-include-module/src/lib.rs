//! A storage crate with a clean include into a clean module.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    store::DEFAULT_CHUNK
}
