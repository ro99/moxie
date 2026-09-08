//! A format crate with clean nested modules.

pub mod outer;

pub fn total() -> usize {
    outer::inner::count()
}
