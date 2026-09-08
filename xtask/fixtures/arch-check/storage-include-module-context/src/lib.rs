//! A storage layer whose model knowledge is two module hops past an include.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    outer::inner::FAMILY_CHUNK
}
