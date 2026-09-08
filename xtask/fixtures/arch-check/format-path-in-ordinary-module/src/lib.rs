//! A format crate whose breach hides behind `#[path]` in a module.

pub mod outer;

pub fn total() -> usize {
    outer::io::read_count()
}
