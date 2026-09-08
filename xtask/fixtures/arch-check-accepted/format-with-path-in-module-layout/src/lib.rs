//! A format crate with a clean `#[path]`-in-module chain.

pub mod outer;

pub fn total() -> usize {
    outer::io::count()
}
