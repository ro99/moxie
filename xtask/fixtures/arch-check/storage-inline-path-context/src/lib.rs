//! A storage layer with `#[path]` nested in an inline module.

pub mod outer {
    #[path = "io.rs"]
    pub mod io;
}

pub fn chunk_name() -> &'static str {
    outer::io::FAMILY_CHUNK
}
