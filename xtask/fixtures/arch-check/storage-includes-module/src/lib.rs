//! A storage layer whose model knowledge is an include away from a module away.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    deep::FAMILY_CHUNK
}
