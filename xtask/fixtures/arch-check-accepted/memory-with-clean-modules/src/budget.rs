//! Pure arithmetic over byte counts the caller supplied.

pub mod peak;

pub fn remaining(physical: u64, committed: u64) -> u64 {
    physical.saturating_sub(committed)
}
