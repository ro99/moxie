//! The stage maximum, which is not the sum.

pub fn peak(per_stage: &[u64]) -> u64 {
    per_stage.iter().copied().max().unwrap_or(0)
}
