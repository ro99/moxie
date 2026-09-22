//! Thin public naming for the generic bounded explicit-restore store.
//!
//! The implementation lives in the accumulator module; this alias preserves
//! the sparse-index API and its StateKind selection.

use crate::accumulator::{
    BoundedReplaySource, BoundedSnapshot, BoundedStateStore, SparseIndexStoreKind,
};
#[cfg(test)]
use crate::{ROOT, RestoreMethod};
#[cfg(test)]
use moxie_types::{HostTier, Result};

pub type SparseIndex = BoundedStateStore<SparseIndexStoreKind>;
pub type SparseIndexSnapshot = BoundedSnapshot<SparseIndexStoreKind>;
pub type SparseIndexReplaySource<'a> = BoundedReplaySource<'a, SparseIndexStoreKind>;

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_memory::{CapacitySnapshot, Ledger};
    use moxie_types::{Error, Scope, Tier};
    use std::cmp::Ordering;

    const K: usize = 2;
    const CANDIDATE_BYTES: usize = 8;
    const INDEX_BYTES: usize = K * CANDIDATE_BYTES;
    const EMPTY_POSITION: u32 = u32::MAX;
    const EMPTY_INDEX: [u8; INDEX_BYTES] = [
        0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
    ];

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Candidate {
        score: u32,
        position: u32,
    }

    impl Candidate {
        const fn empty() -> Self {
            Self {
                score: 0,
                position: EMPTY_POSITION,
            }
        }
    }

    fn ledger() -> Ledger {
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).unwrap()]).unwrap()
    }

    fn candidate_bytes(candidate: Candidate) -> [u8; CANDIDATE_BYTES] {
        let mut bytes = [0; CANDIDATE_BYTES];
        bytes[..4].copy_from_slice(&candidate.score.to_le_bytes());
        bytes[4..].copy_from_slice(&candidate.position.to_le_bytes());
        bytes
    }

    fn read_candidate(bytes: &[u8]) -> Candidate {
        Candidate {
            score: u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            position: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        }
    }

    fn write_candidate(bytes: &mut [u8], candidate: Candidate) {
        bytes[..4].copy_from_slice(&candidate.score.to_le_bytes());
        bytes[4..8].copy_from_slice(&candidate.position.to_le_bytes());
    }

    fn decode_index(bytes: &[u8]) -> [Option<Candidate>; K] {
        let mut entries = [None; K];
        for (slot, entry) in entries.iter_mut().enumerate() {
            let candidate = read_candidate(&bytes[slot * CANDIDATE_BYTES..][..CANDIDATE_BYTES]);
            if candidate.position != EMPTY_POSITION {
                *entry = Some(candidate);
            }
        }
        entries
    }

    /// A deterministic bounded top-K step used only by these host fixtures.
    /// Full slots keep the higher scores and are serialized canonically.
    fn top_k_step(input: &[u8], index: &mut [u8]) -> Result<()> {
        if input.len() != CANDIDATE_BYTES {
            return Err(Error::InvalidRequest {
                field: "candidate",
                detail: format!(
                    "synthetic sparse-index candidate needs {CANDIDATE_BYTES} bytes, got {}",
                    input.len()
                ),
            });
        }
        if index.len() != INDEX_BYTES {
            return Err(Error::InvalidRequest {
                field: "index",
                detail: format!(
                    "synthetic sparse index needs {INDEX_BYTES} bytes, got {}",
                    index.len()
                ),
            });
        }
        let candidate = read_candidate(input);
        if candidate.position == EMPTY_POSITION {
            return Err(Error::InvalidRequest {
                field: "candidate",
                detail: "synthetic sparse-index candidate uses the empty position marker".into(),
            });
        }

        let mut entries = [Candidate::empty(); K];
        for (slot, entry) in entries.iter_mut().enumerate() {
            *entry = read_candidate(&index[slot * CANDIDATE_BYTES..][..CANDIDATE_BYTES]);
        }
        if let Some(slot) = entries
            .iter()
            .position(|entry| entry.position == EMPTY_POSITION)
        {
            entries[slot] = candidate;
        } else {
            let minimum = (0..K)
                .min_by(|left, right| {
                    entries[*left]
                        .score
                        .cmp(&entries[*right].score)
                        .then_with(|| entries[*right].position.cmp(&entries[*left].position))
                })
                .expect("the synthetic index has at least one slot");
            if candidate.score <= entries[minimum].score {
                return Ok(());
            }
            entries[minimum] = candidate;
        }
        entries.sort_by(|left, right| {
            match (
                left.position == EMPTY_POSITION,
                right.position == EMPTY_POSITION,
            ) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => right
                    .score
                    .cmp(&left.score)
                    .then_with(|| left.position.cmp(&right.position)),
            }
        });
        for (slot, entry) in entries.into_iter().enumerate() {
            write_candidate(
                &mut index[slot * CANDIDATE_BYTES..][..CANDIDATE_BYTES],
                entry,
            );
        }
        Ok(())
    }

    fn advance_candidate(index: &mut SparseIndex, ledger: &mut Ledger, candidate: Candidate) {
        let input = candidate_bytes(candidate);
        index.advance(ledger, &input, top_k_step).unwrap();
    }

    fn encode_inputs(candidates: &[Candidate]) -> Vec<[u8; CANDIDATE_BYTES]> {
        candidates.iter().copied().map(candidate_bytes).collect()
    }

    fn input_refs(inputs: &[[u8; CANDIDATE_BYTES]]) -> Vec<&[u8]> {
        inputs.iter().map(AsRef::as_ref).collect()
    }

    fn run_independently(
        mut index: [u8; INDEX_BYTES],
        candidates: &[Candidate],
    ) -> [u8; INDEX_BYTES] {
        for candidate in candidates {
            let input = candidate_bytes(*candidate);
            top_k_step(&input, &mut index).unwrap();
        }
        index
    }

    #[test]
    fn value_dependent_eviction_is_ambiguous_from_step_count_and_current_bytes() {
        let mut owner = ledger();
        let mut with_eviction = SparseIndex::new(&mut owner, &EMPTY_INDEX, 4).unwrap();
        let mut without_eviction = SparseIndex::new(&mut owner, &EMPTY_INDEX, 4).unwrap();

        for candidate in [
            Candidate {
                score: 5,
                position: 0,
            },
            Candidate {
                score: 3,
                position: 1,
            },
        ] {
            advance_candidate(&mut with_eviction, &mut owner, candidate);
        }
        let full_before_eviction = with_eviction.bytes().to_vec();
        advance_candidate(
            &mut with_eviction,
            &mut owner,
            Candidate {
                score: 8,
                position: 2,
            },
        );
        let after_eviction = with_eviction.bytes().to_vec();
        assert_ne!(after_eviction, full_before_eviction);
        assert_eq!(
            decode_index(&after_eviction),
            [
                Some(Candidate {
                    score: 8,
                    position: 2,
                }),
                Some(Candidate {
                    score: 5,
                    position: 0,
                }),
            ]
        );
        // The later low-scoring step does not evict; the current bytes alone
        // retain no record that candidate (3, 1) was ever present.
        advance_candidate(
            &mut with_eviction,
            &mut owner,
            Candidate {
                score: 1,
                position: 3,
            },
        );
        assert_eq!(with_eviction.bytes(), after_eviction);

        for candidate in [
            Candidate {
                score: 5,
                position: 0,
            },
            Candidate {
                score: 8,
                position: 2,
            },
        ] {
            advance_candidate(&mut without_eviction, &mut owner, candidate);
        }
        let without_eviction_before_noops = without_eviction.bytes().to_vec();
        advance_candidate(
            &mut without_eviction,
            &mut owner,
            Candidate {
                score: 1,
                position: 3,
            },
        );
        assert_eq!(without_eviction.bytes(), without_eviction_before_noops);
        advance_candidate(
            &mut without_eviction,
            &mut owner,
            Candidate {
                score: 2,
                position: 4,
            },
        );
        assert_eq!(without_eviction.bytes(), without_eviction_before_noops);

        // Both histories have four steps and the same physical present, while
        // the first had one real eviction and the second had none.
        assert_eq!(with_eviction.prefix(), 4);
        assert_eq!(without_eviction.prefix(), 4);
        assert_eq!(with_eviction.bytes(), without_eviction.bytes());

        with_eviction.release(&mut owner).unwrap();
        without_eviction.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn truncation_and_prefix_bound_are_typed_and_accounted() {
        let mut constrained = ledger();
        assert!(matches!(
            SparseIndex::new(&mut constrained, &EMPTY_INDEX, 200_000),
            Err(Error::CapacityExceeded { .. })
        ));
        assert!(constrained.outstanding().is_empty());

        let mut owner = ledger();
        let mut index = SparseIndex::new(&mut owner, &EMPTY_INDEX, 2).unwrap();
        advance_candidate(
            &mut index,
            &mut owner,
            Candidate {
                score: 5,
                position: 0,
            },
        );
        advance_candidate(
            &mut index,
            &mut owner,
            Candidate {
                score: 8,
                position: 1,
            },
        );
        let before = index.bytes().to_vec();

        assert!(matches!(
            index.advance(
                &mut owner,
                &candidate_bytes(Candidate {
                    score: 9,
                    position: 2,
                }),
                top_k_step,
            ),
            Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                ..
            })
        ));
        assert!(matches!(
            index.truncate(0),
            Err(Error::Unsupported {
                capability: "sparse_index_truncate",
                ..
            })
        ));
        assert_eq!(index.bytes(), before);
        assert_eq!(index.prefix(), 2);

        index.release(&mut owner).unwrap();
        assert!(index.sequence().is_err());
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn snapshot_and_replay_restore_exact_selection_against_an_independent_run() {
        let mut owner = ledger();
        let candidates = [
            Candidate {
                score: 5,
                position: 0,
            },
            Candidate {
                score: 3,
                position: 1,
            },
            Candidate {
                score: 8,
                position: 2,
            },
            Candidate {
                score: 1,
                position: 3,
            },
            Candidate {
                score: 9,
                position: 4,
            },
        ];
        let mut index = SparseIndex::new(&mut owner, &EMPTY_INDEX, 5).unwrap();
        for candidate in &candidates[..2] {
            advance_candidate(&mut index, &mut owner, *candidate);
        }
        let mut snapshot = index.snapshot(&mut owner).unwrap();
        let at_snapshot = snapshot.bytes().to_vec();
        assert_eq!(
            at_snapshot,
            run_independently(EMPTY_INDEX, &candidates[..2]).to_vec()
        );

        for candidate in &candidates[2..] {
            advance_candidate(&mut index, &mut owner, *candidate);
        }
        let at_end = index.bytes().to_vec();
        assert_eq!(snapshot.bytes(), at_snapshot);
        assert_eq!(at_end, run_independently(EMPTY_INDEX, &candidates).to_vec());

        let encoded_tail = encode_inputs(&candidates[2..]);
        let replayed_forward = index
            .replay(
                &mut owner,
                SparseIndexReplaySource::Snapshot(&snapshot),
                5,
                &input_refs(&encoded_tail),
                top_k_step,
            )
            .unwrap();
        assert_eq!(index.bytes(), at_end);
        assert_eq!(
            replayed_forward.method(),
            RestoreMethod::Replay { from: 2, to: 5 }
        );

        let encoded_prefix = encode_inputs(&candidates[..2]);
        let replayed_back = index
            .replay(
                &mut owner,
                SparseIndexReplaySource::Start,
                2,
                &input_refs(&encoded_prefix),
                top_k_step,
            )
            .unwrap();
        assert_eq!(index.bytes(), at_snapshot);
        assert_eq!(
            index.bytes(),
            run_independently(EMPTY_INDEX, &candidates[..2]).as_slice()
        );
        index.rollback_to(2, replayed_back).unwrap();
        assert_eq!(
            index.sequence().unwrap().frontiers(ROOT).unwrap().accepted,
            2
        );

        snapshot.release(&mut owner).unwrap();
        index.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn snapshot_release_refuses_wrong_ledger_without_consuming_the_snapshot() {
        let mut owner = ledger();
        let mut wrong = ledger();
        let mut index = SparseIndex::new(&mut owner, &EMPTY_INDEX, 1).unwrap();
        let mut snapshot = index.snapshot(&mut owner).unwrap();
        let bytes = snapshot.bytes().to_vec();
        let charge_before = owner.scope_committed(Scope::Host);

        assert!(matches!(
            snapshot.release(&mut wrong),
            Err(Error::InvalidRequest {
                field: "ledger",
                ..
            })
        ));
        assert_eq!(snapshot.bytes(), bytes);
        assert_eq!(owner.scope_committed(Scope::Host), charge_before);
        snapshot.release(&mut owner).unwrap();
        index.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
        assert!(wrong.outstanding().is_empty());
    }

    #[test]
    fn failed_replay_preserves_selection_frontier_and_charge() {
        let mut owner = ledger();
        let candidates = [
            Candidate {
                score: 5,
                position: 0,
            },
            Candidate {
                score: 3,
                position: 1,
            },
            Candidate {
                score: 8,
                position: 2,
            },
            Candidate {
                score: 9,
                position: 3,
            },
        ];
        let mut index = SparseIndex::new(&mut owner, &EMPTY_INDEX, 4).unwrap();
        for candidate in &candidates {
            advance_candidate(&mut index, &mut owner, *candidate);
        }
        let before = index.bytes().to_vec();
        let failed_scratch = run_independently(EMPTY_INDEX, &candidates[..3]).to_vec();
        assert_ne!(before, failed_scratch);
        let frontier_before = index.sequence().unwrap().frontiers(ROOT).unwrap();
        let charge_before = owner.scope_committed(Scope::Host);
        let encoded = encode_inputs(&candidates);
        let mut calls = 0;
        let error = index
            .replay(
                &mut owner,
                SparseIndexReplaySource::Start,
                4,
                &input_refs(&encoded),
                |input, state| {
                    top_k_step(input, state)?;
                    calls += 1;
                    if calls == 3 {
                        assert_eq!(state, failed_scratch);
                        Err(Error::Cancelled {
                            at: "sparse index replay",
                        })
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap_err();
        assert_eq!(
            error,
            Error::Cancelled {
                at: "sparse index replay"
            }
        );
        assert_eq!(calls, 3);
        assert_eq!(index.bytes(), before);
        assert_eq!(index.prefix(), 4);
        assert_eq!(
            index.sequence().unwrap().frontiers(ROOT).unwrap(),
            frontier_before
        );
        assert_eq!(owner.scope_committed(Scope::Host), charge_before);
        index.release(&mut owner).unwrap();
    }
}
