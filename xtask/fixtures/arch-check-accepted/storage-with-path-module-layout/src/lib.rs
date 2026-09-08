//! A storage crate with a clean `#[path]` module chain.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    outer::inner::DEFAULT_CHUNK
}
