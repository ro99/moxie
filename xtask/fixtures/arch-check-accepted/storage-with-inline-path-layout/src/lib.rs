//! A storage crate with a clean inline `#[path]` chain.

pub mod outer {
    #[path = "io.rs"]
    pub mod io;
}

pub fn chunk_name() -> &'static str {
    outer::io::DEFAULT_CHUNK
}
