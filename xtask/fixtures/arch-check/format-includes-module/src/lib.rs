//! A format crate whose breach is an include away from a module away.

include!("../tests/entry.rs");

pub fn entry_count() -> usize {
    io::read_count()
}
