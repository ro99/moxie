//! A driver that remembers what it read.
//!
//! `moxie_executor::residency` may perform a read order and report its outcome.
//! The moment it keeps the bytes, the authority's accounting is no longer the
//! whole truth about host memory -- and the ledger is the thing document 03
//! requires to be the whole truth.

/// The definition that is refused.
#[derive(Debug, Default)]
pub struct ChunkCache {
    bytes: Vec<u8>,
}

impl ChunkCache {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}
