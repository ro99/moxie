//! A format crate with a clean inline `#[path]` chain.

pub mod outer {
    #[path = "io.rs"]
    pub mod io;
}

pub fn total() -> usize {
    outer::io::count()
}
