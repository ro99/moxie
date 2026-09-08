//! A storage layer whose model knowledge hides behind a `#[path]` module.

include!("../tests/entry.rs");

pub fn chunk_name() -> &'static str {
    outer::inner::FAMILY_CHUNK
}
