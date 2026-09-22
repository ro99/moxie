//! Prefix-reuse identity and admission.
//!
//! The caller owns checkpoint, tokenizer/template, configuration and token
//! identity. This module carries those opaque identifiers and asks
//! [`SequenceState`] whether the requested prefix can reuse its current
//! retained forward result. It does not compare token content or mutate state.

use crate::{BranchId, LogitsHandle, SequenceState, StateGeneration};
use moxie_types::Result;

/// Opaque caller identity for a reusable prefix.
///
/// `generation` is the state authority's freshness guard. The other fields
/// stay opaque here: the caller that owns checkpoint and tokenizer identity
/// compares them before presenting this key to state. Token content is not
/// hashed or compared by `moxie-state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixReuseKey {
    checkpoint: u64,
    tokenizer_template: u64,
    configuration: u64,
    token_prefix: u64,
    generation: StateGeneration,
}

impl PrefixReuseKey {
    /// Build a key from caller-owned opaque identities and the sequence's
    /// current generation.
    pub const fn new(
        checkpoint: u64,
        tokenizer_template: u64,
        configuration: u64,
        token_prefix: u64,
        generation: StateGeneration,
    ) -> Self {
        Self {
            checkpoint,
            tokenizer_template,
            configuration,
            token_prefix,
            generation,
        }
    }

    pub const fn checkpoint(self) -> u64 {
        self.checkpoint
    }

    pub const fn tokenizer_template(self) -> u64 {
        self.tokenizer_template
    }

    pub const fn configuration(self) -> u64 {
        self.configuration
    }

    pub const fn token_prefix(self) -> u64 {
        self.token_prefix
    }

    pub const fn generation(self) -> StateGeneration {
        self.generation
    }

    /// Decide whether `claimed_prefix` can be reused on `branch`.
    ///
    /// A full hit returns the already validated handle; the caller may pass it
    /// to [`SequenceState::restore_logits`]. A boundary is the largest
    /// logically materialized position at or before the claim. It does not
    /// certify physical KV retention; a physical owner must check its retained
    /// range separately. Zero means the caller must recompute from the
    /// beginning. No branch state is changed.
    pub fn decide(
        self,
        sequence: &SequenceState,
        branch: BranchId,
        claimed_prefix: u64,
    ) -> Result<PrefixReuseDecision> {
        let current_generation = sequence.generation();
        if self.generation != current_generation {
            return Ok(PrefixReuseDecision::Refused(
                PrefixReuseRefusal::ConfigMismatch {
                    key_generation: self.generation,
                    current_generation,
                },
            ));
        }

        let frontiers = sequence.frontiers(branch)?;
        if claimed_prefix > frontiers.accepted {
            return Ok(PrefixReuseDecision::Refused(
                PrefixReuseRefusal::ClaimExceedsCommitted {
                    claimed: claimed_prefix,
                    committed: frontiers.accepted,
                },
            ));
        }

        let boundary = claimed_prefix.min(frontiers.executed);
        let retained = sequence.retained_logits(branch)?;
        if let Some(handle) = retained
            && handle.prefix() == claimed_prefix
            && (claimed_prefix < frontiers.accepted || sequence.next_logits_valid(branch))
        {
            return Ok(PrefixReuseDecision::FullHit { handle });
        }

        Ok(PrefixReuseDecision::BoundaryRecompute { from: boundary })
    }
}

/// The pure result of a prefix-reuse admission check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixReuseDecision {
    /// The exact claimed prefix has a live handle on this branch.
    FullHit { handle: LogitsHandle },
    /// Recompute from this logically materialized position. Physical KV
    /// retention is a separate owner check; zero means a full re-prefill.
    BoundaryRecompute { from: u64 },
    /// The request cannot be admitted against this sequence.
    Refused(PrefixReuseRefusal),
}

/// The two distinct reasons a prefix-reuse claim is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixReuseRefusal {
    /// The key was minted for an older execution configuration.
    ConfigMismatch {
        key_generation: StateGeneration,
        current_generation: StateGeneration,
    },
    /// The caller claimed tokens that are not committed on this branch.
    ClaimExceedsCommitted { claimed: u64, committed: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ROOT, StateKind};

    fn key(sequence: &SequenceState) -> PrefixReuseKey {
        PrefixReuseKey::new(11, 22, 33, 44, sequence.generation())
    }

    fn committed(sequence: &mut SequenceState, prompt: u64, completion: u64) {
        sequence.append_prompt(ROOT, prompt).unwrap();
        sequence.accept(ROOT, completion).unwrap();
        sequence.execute(ROOT, prompt + completion).unwrap();
    }

    #[test]
    fn full_hit_returns_the_live_handle_and_rejects_replaced_lineage() {
        // Document 04: a full hit needs an exact saved result, not equal
        // counters. The decision itself is pure; the returned handle remains
        // usable by the existing restore API.
        let mut sequence = SequenceState::new([StateKind::KvPages]);
        committed(&mut sequence, 4, 4);
        let handle = sequence.record_logits(ROOT, 8).unwrap();
        let old_lineage = handle.lineage();
        let before = (
            sequence.frontiers(ROOT).unwrap(),
            sequence.retained_logits(ROOT).unwrap(),
            sequence.live_results(),
        );

        assert_eq!(
            key(&sequence).decide(&sequence, ROOT, 8).unwrap(),
            PrefixReuseDecision::FullHit { handle }
        );
        assert_eq!(
            (
                sequence.frontiers(ROOT).unwrap(),
                sequence.retained_logits(ROOT).unwrap(),
                sequence.live_results(),
            ),
            before,
            "deciding does not restore or otherwise mutate state"
        );
        sequence.restore_logits(ROOT, handle).unwrap();

        // Replace the suffix and execute to the same numeric prefix. The old
        // handle was discarded, so matching counters cannot produce a hit.
        sequence.rollback_to(ROOT, 4, &[]).unwrap();
        sequence.accept(ROOT, 4).unwrap();
        sequence.execute(ROOT, 4).unwrap();
        assert_eq!(sequence.frontiers(ROOT).unwrap().accepted, 8);
        assert_ne!(sequence.lineage_at(ROOT, 8).unwrap(), Some(old_lineage));
        assert!(sequence.restore_logits(ROOT, handle).is_err());
        assert_eq!(
            key(&sequence).decide(&sequence, ROOT, 8).unwrap(),
            PrefixReuseDecision::BoundaryRecompute { from: 8 }
        );
    }

    #[test]
    fn boundary_recompute_uses_logical_materialization_not_a_logits_handle() {
        // Document 04: committed history and materialized forward state are
        // distinct. The branch has committed ten tokens but only materialized
        // six, so both sides of min(claimed, executed) must be load-bearing.
        let mut sequence = SequenceState::new([StateKind::KvPages]);
        sequence.append_prompt(ROOT, 4).unwrap();
        sequence.accept(ROOT, 6).unwrap();
        sequence.execute(ROOT, 6).unwrap();
        let handle = sequence.record_logits(ROOT, 6).unwrap();
        assert!(!sequence.next_logits_valid(ROOT));

        assert_eq!(
            key(&sequence).decide(&sequence, ROOT, 10).unwrap(),
            PrefixReuseDecision::BoundaryRecompute { from: 6 }
        );
        assert_eq!(
            key(&sequence).decide(&sequence, ROOT, 5).unwrap(),
            PrefixReuseDecision::BoundaryRecompute { from: 5 }
        );
        assert_eq!(
            sequence.lineage_at(ROOT, 6).unwrap(),
            Some(handle.lineage())
        );
        let frontiers = sequence.frontiers(ROOT).unwrap();
        assert_eq!(frontiers.accepted, 10);
        assert_eq!(frontiers.executed, 6);

        // A stale or absent logits cache does not erase materialized state.
        let mut without_handle = SequenceState::new([StateKind::KvPages]);
        committed(&mut without_handle, 4, 6);
        assert_eq!(without_handle.retained_logits(ROOT).unwrap(), None);
        assert_eq!(
            key(&without_handle)
                .decide(&without_handle, ROOT, 10)
                .unwrap(),
            PrefixReuseDecision::BoundaryRecompute { from: 10 }
        );
    }

    #[test]
    fn a_stale_generation_is_a_typed_config_refusal() {
        // Document 04: changing graph/precision/position configuration
        // invalidates saved outputs even when the claimed prefix is plausible.
        let mut sequence = SequenceState::new([StateKind::KvPages]);
        committed(&mut sequence, 4, 4);
        let key = key(&sequence);
        let key_generation = key.generation();
        sequence.invalidate_generation().unwrap();
        let current_generation = sequence.generation();

        assert_eq!(
            key.decide(&sequence, ROOT, 8).unwrap(),
            PrefixReuseDecision::Refused(PrefixReuseRefusal::ConfigMismatch {
                key_generation,
                current_generation,
            })
        );
        assert_ne!(key_generation, current_generation);
    }

    #[test]
    fn a_claim_past_committed_history_is_a_distinct_refusal() {
        // Document 04: a caller may not silently clip a claimed prefix to the
        // actual committed frontier and call that a hit.
        let mut sequence = SequenceState::new([StateKind::KvPages]);
        committed(&mut sequence, 4, 4);
        let key = key(&sequence);

        assert_eq!(
            key.decide(&sequence, ROOT, 9).unwrap(),
            PrefixReuseDecision::Refused(PrefixReuseRefusal::ClaimExceedsCommitted {
                claimed: 9,
                committed: 8,
            })
        );
    }
}
