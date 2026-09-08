//! A format crate loading one clean file through two contexts.

include!("../tests/entry.rs");

pub fn entry_count() -> usize {
    alias::inner::count() + outer::inner::count()
}
