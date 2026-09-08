//! A format crate whose breach hides behind a `#[path]` module.

include!("../tests/entry.rs");

pub fn entry_count() -> usize {
    outer::inner::read_count()
}
