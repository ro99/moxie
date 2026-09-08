//! A storage crate with a clean `#[path]`-in-module chain.

pub mod outer;

pub fn chunk_name() -> &'static str {
    outer::io::DEFAULT_CHUNK
}
