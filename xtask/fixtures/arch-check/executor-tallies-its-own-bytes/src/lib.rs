//! An executor that counts the bytes it moves.
//!
//! `moxie_executor::trace` may *read* `moxie_memory::ByteFlow` and difference
//! two snapshots of it -- that is arithmetic over the owner's numbers. Declaring
//! a flow of its own is a second tally of the same events, and the two will
//! disagree exactly when something is wrong, which is the moment a trace is
//! supposed to be trustworthy.

/// The definition that is refused.
#[derive(Debug, Default)]
pub struct ByteFlow {
    admitted_bytes: u64,
}

impl ByteFlow {
    pub fn admitted(&mut self, bytes: u64) {
        self.admitted_bytes += bytes;
    }
}
