//! Sequence state: schema, transactions, and the two-frontier rule.
//!
//! Document 02: this crate owns "Paged sequence state, forks, transactions,
//! rollback and prefix reuse".
//!
//! M0 scope: the *two-frontier* invariant from document 04, expressed as types
//! and tested. Everything else -- paging, COW, recurrent snapshots -- lands in
//! M1/M2/M4. The frontier rule is here first because document 06 warns that M9
//! "must not retrofit incompatible cache ownership", and because getting it
//! wrong is the off-by-one that silently corrupts a cache.

#![forbid(unsafe_code)]

use moxie_types::{Error, Result};

/// The kinds of state a schema can declare (document 04).
///
/// Listed so that a rollback test can enumerate them: "A rejection rolls back
/// every state kind, including grammar/history/recurrent state."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StateKind {
    KvPages,
    MlaLatent,
    SparseIndex,
    RecurrentAccumulator,
    ConvolutionHistory,
    PositionCounter,
    SamplerHistory,
    ConstraintState,
}

impl StateKind {
    /// Whether an earlier state can be recovered by truncating a counter.
    ///
    /// Paged KV can drop a tentative suffix. Recurrent state cannot: R20 records
    /// Kimi's documented inability to recover recurrent state by decrementing a
    /// position. These need a bounded snapshot or a replay from a saved prefix,
    /// and the cost has to be accounted for.
    pub const fn is_truncatable(self) -> bool {
        match self {
            StateKind::KvPages
            | StateKind::MlaLatent
            | StateKind::SparseIndex
            | StateKind::PositionCounter
            | StateKind::SamplerHistory => true,
            StateKind::RecurrentAccumulator
            | StateKind::ConvolutionHistory
            | StateKind::ConstraintState => false,
        }
    }
}

/// The two frontiers of a sequence (document 04).
///
/// "Track two frontiers explicitly: committed token history and materialized
/// forward/state position. Sampling a token does not itself execute that token's
/// forward pass."
///
/// A bonus or mismatch token can therefore be *committed* to history while its
/// KV/recurrent state does not yet exist. Conflating the two is the off-by-one
/// that corrupts a cache while the generated text still looks plausible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Frontiers {
    /// Tokens published to the user and counted in usage.
    committed: u64,
    /// Tokens whose forward pass has actually run and whose state exists.
    materialized: u64,
}

impl Frontiers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn committed(self) -> u64 {
        self.committed
    }

    pub fn materialized(self) -> u64 {
        self.materialized
    }

    /// Tokens committed but not yet executed. Before continuation or branching
    /// these must be materialized, or an equivalent saved forward result used.
    pub fn pending_suffix(self) -> u64 {
        self.committed - self.materialized
    }

    /// Commit `n` tokens to history without executing them.
    pub fn commit(&mut self, n: u64) -> Result<()> {
        self.committed = self
            .committed
            .checked_add(n)
            .ok_or_else(|| Error::InvalidRequest {
                field: "commit",
                detail: "token counter overflow".into(),
            })?;
        Ok(())
    }

    /// Record that `n` more tokens have had their forward pass executed.
    ///
    /// Materializing past the committed frontier is a defect: it would mean
    /// state exists for a token that was never committed, which is exactly the
    /// corruption this type exists to prevent.
    pub fn materialize(&mut self, n: u64) -> Result<()> {
        let next = self
            .materialized
            .checked_add(n)
            .ok_or_else(|| Error::InvalidRequest {
                field: "materialize",
                detail: "token counter overflow".into(),
            })?;
        if next > self.committed {
            return Err(Error::InvalidRequest {
                field: "materialize",
                detail: format!(
                    "would materialize {next} tokens past a committed frontier of {}",
                    self.committed
                ),
            });
        }
        self.materialized = next;
        Ok(())
    }

    /// Whether the next-token logits are valid right now.
    ///
    /// Document 04: a full prefix hit "still produces valid next-token logits or
    /// recomputes the required boundary step". With a pending suffix, they are
    /// not valid: the last committed token has not been through the model.
    pub fn next_logits_valid(self) -> bool {
        self.pending_suffix() == 0
    }

    /// Roll back to a committed prefix, as `abort` does.
    pub fn rollback_to(&mut self, prefix: u64) -> Result<()> {
        if prefix > self.committed {
            return Err(Error::InvalidRequest {
                field: "prefix",
                detail: format!(
                    "cannot roll back to {prefix}, only {} committed",
                    self.committed
                ),
            });
        }
        self.committed = prefix;
        // State cannot exist beyond the committed frontier after a rollback.
        self.materialized = self.materialized.min(prefix);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sampled_token_is_committed_before_it_is_executed() {
        // The speculative-decoding case from document 04: a bonus token is
        // committed to history, but its forward pass has not run.
        let mut f = Frontiers::new();
        f.commit(10).unwrap();
        f.materialize(10).unwrap();
        assert!(f.next_logits_valid());

        f.commit(1).unwrap(); // bonus token, not yet executed
        assert_eq!(f.pending_suffix(), 1);
        assert!(
            !f.next_logits_valid(),
            "logits after an unexecuted token must not be treated as valid"
        );

        f.materialize(1).unwrap();
        assert!(f.next_logits_valid());
    }

    #[test]
    fn state_cannot_run_ahead_of_history() {
        let mut f = Frontiers::new();
        f.commit(4).unwrap();
        assert!(f.materialize(5).is_err());
        assert!(f.materialize(4).is_ok());
    }

    #[test]
    fn rollback_drops_state_beyond_the_prefix() {
        let mut f = Frontiers::new();
        f.commit(20).unwrap();
        f.materialize(20).unwrap();
        f.rollback_to(12).unwrap();
        assert_eq!(f.committed(), 12);
        assert_eq!(f.materialized(), 12);
        assert!(f.next_logits_valid());
    }

    #[test]
    fn rollback_past_the_committed_frontier_is_refused() {
        let mut f = Frontiers::new();
        f.commit(3).unwrap();
        assert!(f.rollback_to(9).is_err());
    }

    #[test]
    fn rejection_at_every_depth_leaves_consistent_frontiers() {
        // Verification proposed 4 tokens; each rejection depth must land on a
        // state where the frontiers agree and logits are valid.
        for accepted in 0..=4u64 {
            let mut f = Frontiers::new();
            f.commit(100).unwrap();
            f.materialize(100).unwrap();
            f.commit(4).unwrap();
            f.materialize(4).unwrap();
            f.rollback_to(100 + accepted).unwrap();
            assert_eq!(f.committed(), 100 + accepted);
            assert_eq!(f.materialized(), 100 + accepted);
            assert!(f.next_logits_valid());
        }
    }

    #[test]
    fn recurrent_state_is_not_truncatable() {
        // R20: Kimi documents that recurrent state cannot be recovered by
        // decrementing a position counter. A rollback plan that assumes it can
        // is silently wrong.
        assert!(!StateKind::RecurrentAccumulator.is_truncatable());
        assert!(!StateKind::ConvolutionHistory.is_truncatable());
        assert!(StateKind::KvPages.is_truncatable());
    }
}
