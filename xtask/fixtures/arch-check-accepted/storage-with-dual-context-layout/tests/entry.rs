// Two declarations, one shared backing file.

pub mod outer;
#[path = "outer.rs"]
pub mod alias;
