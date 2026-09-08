//! A format crate loading one file through two contexts.

include!("../tests/entry.rs");

pub fn entry_count() -> usize {
    alias::inner::read_count() + outer::inner::read_count()
}
