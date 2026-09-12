//! An engine that keeps its own idea of what is resident.
//!
//! Computing the demand set is legal and is what `moxie_engine::residency`
//! does. Remembering the answer is not: two things that both believe they know
//! what is resident is R02 with the roles swapped, and the disagreement shows up
//! as an eviction of bytes a step is about to read.

/// The definition that is refused. Naming `moxie_memory`'s types would be fine;
/// declaring a cache is not.
#[derive(Debug, Default)]
pub struct ExpertCache {
    resident: Vec<u32>,
}

impl ExpertCache {
    pub fn hit(&self, expert: u32) -> bool {
        self.resident.contains(&expert)
    }
}
