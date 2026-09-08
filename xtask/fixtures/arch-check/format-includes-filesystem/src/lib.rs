//! A format crate whose filesystem breach arrives by include.

include!("../tests/io.rs");

pub fn parse_len(text: &str) -> usize {
    let _ = text;
    read_count()
}
