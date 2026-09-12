//! A model adapter that decides which expert should go.
//!
//! It imports nothing and allocates nothing, which is exactly why the manifest
//! rule cannot see it. What it does is choose a victim, and document 03 gives
//! that to the one authority so the choice can account for reload bytes, reuse
//! estimate, expert size, NUMA placement and phase -- none of which a model
//! knows.

/// The definition that is refused.
#[derive(Debug, Default)]
pub struct EvictionPolicy {
    pub keep_hot: u32,
}

impl EvictionPolicy {
    pub fn victim(&self, resident: &[u32]) -> Option<u32> {
        resident.iter().copied().min()
    }
}
