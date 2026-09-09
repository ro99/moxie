//! A ledger crate with a clean nested module.

pub mod budget;

pub fn headroom(physical: u64, committed: u64) -> u64 {
    budget::remaining(physical, committed)
}
