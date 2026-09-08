//! A format crate with `#[path]` nested in an inline module.

pub mod outer {
    #[path = "io.rs"]
    pub mod io;
}

pub fn total() -> usize {
    outer::io::read_count()
}
