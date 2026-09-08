// `#[path]` declaration first, ordinary declaration second.

#[path = "outer.rs"]
pub mod alias;
pub mod outer;
