//! Opaque identifiers.
//!
//! These are newtypes rather than bare integers so that a chunk id cannot be
//! passed where a layout id is expected. Document 03 requires canonical chunk
//! identity and prepared-layout identity to be *separate*: a buffer packed for
//! one kernel must not be consumed by another under a different interpretation.

macro_rules! opaque_id {
    ($(#[$m:meta])* $name:ident, $inner:ty) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub $inner);

        impl $name {
            pub const fn get(self) -> $inner { self.0 }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

opaque_id!(
    /// A canonical artifact: one converted checkpoint at one precision profile.
    ArtifactId, u64);
opaque_id!(
    /// A logical tensor role within an artifact.
    TensorId, u64);
opaque_id!(
    /// Immutable canonical chunk identity: (artifact, tensor/expert, logical
    /// range, format version). Document 03.
    ChunkId, u64);
opaque_id!(
    /// Prepared-layout identity: (chunk, device capability, kernel layout
    /// version). Deliberately *not* interchangeable with `ChunkId` -- see R17.
    LayoutId, u64);
opaque_id!(
    /// A compiled semantic graph.
    GraphId, u64);
opaque_id!(
    /// A copy-on-write branch of sequence state.
    BranchId, u64);
opaque_id!(
    /// An open state transaction. Committed only on `commit_prefix`.
    StateTransactionId, u64);
opaque_id!(
    /// A CUDA device ordinal. Diagnostic only -- evidence identifies a GPU by
    /// UUID (document 07). Valid only under CUDA_DEVICE_ORDER=PCI_BUS_ID.
    DeviceId, u32);
opaque_id!(
    /// An execution rank. One rank owns one device context (document 01).
    RankId, u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_and_layout_ids_are_distinct_types() {
        // This is the R17 lesson expressed in the type system: the same integer
        // is a different thing depending on which identity it names.
        let chunk = ChunkId(7);
        let layout = LayoutId(7);
        assert_eq!(chunk.get(), layout.get());
        assert_eq!(chunk.to_string(), "ChunkId(7)");
        assert_eq!(layout.to_string(), "LayoutId(7)");
        // `chunk == layout` does not compile, which is the point.
    }
}
