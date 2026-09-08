//! A storage layer whose model knowledge hides behind `#[path]` in a module.

pub mod outer;

pub fn chunk_name() -> &'static str {
    outer::io::FAMILY_CHUNK
}
