// Ordinary declaration first, `#[path]` declaration second.

pub mod outer;
#[path = "outer.rs"]
pub mod alias;
