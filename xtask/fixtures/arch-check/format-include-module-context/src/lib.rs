//! A format crate whose breach is two module hops past an include.

include!("../tests/entry.rs");

pub fn entry_count() -> usize {
    outer::inner::read_count()
}
