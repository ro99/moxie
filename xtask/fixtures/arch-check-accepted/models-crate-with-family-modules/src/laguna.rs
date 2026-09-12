//! A second family beside the first, sharing the crate and its boundary. The
//! two cannot reach anything the crate does not depend on, which is the whole
//! reason they may share it.
pub fn compose() -> u32 {
    48
}
