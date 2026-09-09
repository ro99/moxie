//! A reserve that depends on which model is running. Must be rejected.

pub fn incoming_reserve_bytes(family: &str) -> u64 {
    if family == "inkling" {
        48 * 1024 * 1024
    } else {
        0
    }
}
