//! Attention visibility: which keys a query position may attend to.
//!
//! This is a *descriptor*, not a computation, so it lives with the operation
//! contracts rather than with the reference implementation that consumes it.
//! Document 04 lists "causal/window rules" among the attention descriptor's
//! fields, alongside head geometry and cache schema.
//!
//! `moxie-oracles` re-exports it, so the fixtures that already pin its behaviour
//! keep working and keep being the place its semantics are tested.

/// Which keys a query position may attend to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// Every key at or before the query position.
    Causal,
    /// The `window` most recent keys, inclusive of the query position.
    /// A window of 1 sees only the query's own position.
    SlidingWindow { window: u64 },
}

impl Visibility {
    /// Whether query position `q` may attend to key position `k`.
    ///
    /// Positions are absolute over the whole sequence, not indices into a chunk.
    /// R21 is about exactly this confusion: a one-page fast path that indexes
    /// within a chunk is right at position zero and wrong everywhere after it.
    pub const fn allows(self, q: u64, k: u64) -> bool {
        if k > q {
            return false;
        }
        match self {
            Visibility::Causal => true,
            Visibility::SlidingWindow { window } => q - k < window,
        }
    }
}
