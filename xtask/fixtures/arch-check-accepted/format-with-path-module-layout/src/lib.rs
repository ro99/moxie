//! A format crate with a clean `#[path]` module chain.

include!("../tests/entry.rs");

pub fn entry_count() -> usize {
    outer::inner::count()
}
