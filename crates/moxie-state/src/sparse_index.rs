//! Physical host storage for explicitly restorable maintained sparse indexes.
//!
//! The store is deliberately model-independent. It owns one fixed-width
//! bounded selection state and asks its caller to supply the deterministic
//! candidate-selection step. [`SequenceState`] remains the authority for
//! branch identity, prefix lineage and rollback evidence; this module only
//! makes that evidence correspond to actual bytes.

use moxie_memory::{HostBuffer, Ledger};
use moxie_types::{BranchId, Error, HostTier, Result, StateTransactionId, Tier};

use crate::{
    Branch, Journal, PrefixLineage, ROOT, Restore, RestoreMethod, SequenceState, StateKind,
};

/// A physical snapshot of one maintained sparse-index state.
#[derive(Debug)]
pub struct SparseIndexSnapshot {
    prefix: u64,
    evidence: Restore,
    bytes: HostBuffer,
}

impl SparseIndexSnapshot {
    /// The accepted prefix represented by this snapshot.
    pub const fn prefix(&self) -> u64 {
        self.prefix
    }

    /// The logical restore evidence stamped when the snapshot was taken.
    pub const fn evidence(&self) -> Restore {
        self.evidence
    }

    /// The exact bounded selection state captured at [`Self::prefix`].
    pub fn bytes(&self) -> &[u8] {
        self.bytes.bytes()
    }

    /// Release this snapshot's admitted host bytes.
    ///
    /// A refused release leaves the snapshot and its release authority intact,
    /// so the caller can retry with the owning ledger.
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        self.bytes.release(ledger)
    }
}

/// Where a sparse-index replay starts.
#[derive(Debug, Clone, Copy)]
pub enum SparseIndexReplaySource<'a> {
    /// Re-run from the initial selection state at prefix zero.
    Start,
    /// Re-run from an exact snapshot owned by this index's sequence.
    Snapshot(&'a SparseIndexSnapshot),
}

/// A host-backed, model-independent bounded sparse-index state.
///
/// The current selection and the prefix-zero source are separate admitted host
/// buffers. Replay uses a third, short-lived admitted buffer, so a failed step
/// can mutate only scratch bytes. The current selection is replaced only after
/// the whole replay and the existing [`SequenceState::restore_evidence`] check
/// have succeeded.
#[derive(Debug)]
pub struct SparseIndex {
    sequence: Option<SequenceState>,
    start: HostBuffer,
    current: HostBuffer,
    prefix: u64,
    max_prefix: u64,
}

impl SparseIndex {
    /// Create a sparse index with fixed-width initial bytes and a bounded
    /// accepted prefix. Every supplied step must preserve that byte width.
    pub fn new(ledger: &mut Ledger, initial: &[u8], max_prefix: u64) -> Result<Self> {
        if initial.is_empty() {
            return Err(Error::InvalidRequest {
                field: "initial",
                detail: "sparse index state must contain at least one byte".into(),
            });
        }
        let lineage_capacity = max_prefix
            .checked_add(1)
            .ok_or(moxie_types::DimError::Overflow)?;
        let lineage_capacity =
            usize::try_from(lineage_capacity).map_err(|_| moxie_types::DimError::Overflow)?;
        let lineage_bytes = lineage_capacity
            .checked_mul(core::mem::size_of::<PrefixLineage>())
            .ok_or(moxie_types::DimError::Overflow)?;
        let lineage_bytes_u64 =
            u64::try_from(lineage_bytes).map_err(|_| moxie_types::DimError::Overflow)?;
        let control_bytes = core::mem::size_of::<Self>()
            .checked_add(core::mem::size_of::<StateKind>())
            .and_then(|bytes| bytes.checked_add(lineage_bytes))
            .and_then(|bytes| {
                bytes.checked_add(btree_node_bound(core::mem::size_of::<(BranchId, Branch)>()))
            })
            .and_then(|bytes| {
                bytes.checked_add(btree_node_bound(core::mem::size_of::<(
                    StateTransactionId,
                    Journal,
                )>()))
            })
            .ok_or(moxie_types::DimError::Overflow)?;
        if control_bytes > isize::MAX as usize {
            return Err(moxie_types::DimError::Overflow.into());
        }

        let mut current =
            HostBuffer::allocate(ledger, "sparse index state", initial.len(), control_bytes)?;
        let mut start = match HostBuffer::allocate(ledger, "sparse index start", initial.len(), 0) {
            Ok(buffer) => buffer,
            Err(error) => {
                current
                    .release(ledger)
                    .expect("the admitting ledger owns current sparse-index bytes");
                return Err(error);
            }
        };
        current.bytes_mut().copy_from_slice(initial);
        start.bytes_mut().copy_from_slice(initial);

        let mut sequence = SequenceState::new([StateKind::SparseIndex]);
        let lineage = &mut sequence
            .branches
            .get_mut(&ROOT)
            .expect("root exists")
            .lineage;
        if lineage
            .try_reserve_exact(lineage_capacity - lineage.len())
            .is_err()
            || lineage.capacity() != lineage_capacity
        {
            drop(sequence);
            start
                .release(ledger)
                .expect("the admitting ledger owns start sparse-index bytes");
            current
                .release(ledger)
                .expect("the admitting ledger owns current sparse-index bytes");
            return Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: lineage_bytes_u64,
                available_bytes: 0,
            });
        }

        Ok(Self {
            sequence: Some(sequence),
            start,
            current,
            prefix: 0,
            max_prefix,
        })
    }

    /// The logical state authority used for evidence and rollback.
    pub fn sequence(&self) -> Result<&SequenceState> {
        self.sequence_ref()
    }

    /// The current physical prefix.
    pub const fn prefix(&self) -> u64 {
        self.prefix
    }

    /// The exact bounded selection state currently held by the physical store.
    pub fn bytes(&self) -> &[u8] {
        self.current.bytes()
    }

    /// Advance one accepted candidate using caller-supplied deterministic
    /// selection algebra.
    ///
    /// The callback receives the candidate payload and a private copy of the
    /// current bytes. A callback failure, including one after it has partially
    /// mutated that copy, leaves physical and logical state unchanged.
    pub fn advance<F>(&mut self, ledger: &mut Ledger, input: &[u8], mut step: F) -> Result<()>
    where
        F: FnMut(&[u8], &mut [u8]) -> Result<()>,
    {
        self.require_open()?;
        self.require_aligned()?;
        if self.prefix >= self.max_prefix {
            return Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                requested_bytes: core::mem::size_of::<PrefixLineage>() as u64,
                available_bytes: 0,
            });
        }
        let next_prefix = self.next_prefix()?;
        let mut scratch =
            HostBuffer::allocate(ledger, "sparse index step", self.current.bytes().len(), 0)?;
        scratch.bytes_mut().copy_from_slice(self.current.bytes());
        if let Err(error) = step(input, scratch.bytes_mut()) {
            scratch
                .release(ledger)
                .expect("the admitting ledger owns sparse-index step scratch");
            return Err(error);
        }

        let sequence = self.sequence_mut()?;
        let transaction = match sequence.begin(ROOT) {
            Ok(transaction) => transaction,
            Err(error) => {
                scratch
                    .release(ledger)
                    .expect("the admitting ledger owns sparse-index step scratch");
                return Err(error);
            }
        };
        if let Err(error) = sequence.execute(ROOT, 1) {
            sequence
                .abort(transaction)
                .expect("the new sparse-index transaction is open");
            scratch
                .release(ledger)
                .expect("the admitting ledger owns sparse-index step scratch");
            return Err(error);
        }
        if let Err(error) = sequence.commit_prefix(transaction, 1) {
            sequence
                .abort(transaction)
                .expect("the failed sparse-index commit remains abortable");
            scratch
                .release(ledger)
                .expect("the admitting ledger owns sparse-index step scratch");
            return Err(error);
        }
        self.current.bytes_mut().copy_from_slice(scratch.bytes());
        self.prefix = next_prefix;
        scratch
            .release(ledger)
            .expect("the admitting ledger owns sparse-index step scratch");
        Ok(())
    }

    /// Capture the current selection and stamp existing snapshot evidence.
    pub fn snapshot(&self, ledger: &mut Ledger) -> Result<SparseIndexSnapshot> {
        self.require_open()?;
        self.require_aligned()?;
        let evidence = self.sequence_ref()?.restore_evidence(
            StateKind::SparseIndex,
            ROOT,
            RestoreMethod::Snapshot {
                of_prefix: self.prefix,
            },
        )?;
        let mut bytes = HostBuffer::allocate(
            ledger,
            "sparse index snapshot",
            self.current.bytes().len(),
            core::mem::size_of::<SparseIndexSnapshot>(),
        )?;
        bytes.bytes_mut().copy_from_slice(self.current.bytes());
        Ok(SparseIndexSnapshot {
            prefix: self.prefix,
            evidence,
            bytes,
        })
    }

    /// Reconstruct a target prefix by replaying supplied candidates.
    pub fn replay<F>(
        &mut self,
        ledger: &mut Ledger,
        source: SparseIndexReplaySource<'_>,
        target: u64,
        inputs: &[&[u8]],
        mut step: F,
    ) -> Result<Restore>
    where
        F: FnMut(&[u8], &mut [u8]) -> Result<()>,
    {
        self.require_open()?;
        let (from, source_bytes) = match source {
            SparseIndexReplaySource::Start => (0, self.start.bytes()),
            SparseIndexReplaySource::Snapshot(snapshot) => {
                let sequence = self.sequence_ref()?;
                if snapshot.evidence.sequence() != sequence.id()
                    || snapshot.evidence.branch() != ROOT
                    || snapshot.evidence.generation() != sequence.generation()
                    || snapshot.evidence.kind() != StateKind::SparseIndex
                    || snapshot.evidence.method()
                        != (RestoreMethod::Snapshot {
                            of_prefix: snapshot.prefix,
                        })
                    || snapshot.bytes().len() != self.current.bytes().len()
                    || sequence.lineage_at(ROOT, snapshot.prefix)?
                        != Some(snapshot.evidence.lineage())
                {
                    return Err(Error::InvalidRequest {
                        field: "snapshot",
                        detail: "snapshot does not belong to this sparse index's current lineage"
                            .into(),
                    });
                }
                (snapshot.prefix, snapshot.bytes())
            }
        };
        if target < from {
            return Err(Error::InvalidRequest {
                field: "target",
                detail: format!("replay target {target} precedes source prefix {from}"),
            });
        }
        let steps = target
            .checked_sub(from)
            .expect("target was checked not to precede source");
        let expected_inputs = usize::try_from(steps).map_err(|_| Error::InvalidRequest {
            field: "inputs",
            detail: "replay step count does not fit host indexing".into(),
        })?;
        if inputs.len() != expected_inputs {
            return Err(Error::InvalidRequest {
                field: "inputs",
                detail: format!(
                    "replay from {from} to {target} needs {expected_inputs} input(s), got {}",
                    inputs.len()
                ),
            });
        }
        let _ = self
            .sequence_ref()?
            .lineage_at(ROOT, target)?
            .ok_or_else(|| Error::InvalidRequest {
                field: "target",
                detail: format!("prefix {target} is not occupied in the logical sequence"),
            })?;

        let mut scratch =
            HostBuffer::allocate(ledger, "sparse index replay", self.current.bytes().len(), 0)?;
        scratch.bytes_mut().copy_from_slice(source_bytes);
        for input in inputs {
            if let Err(error) = step(input, scratch.bytes_mut()) {
                scratch
                    .release(ledger)
                    .expect("the admitting ledger owns sparse-index replay scratch");
                return Err(error);
            }
        }

        let evidence = match self.sequence_ref()?.restore_evidence(
            StateKind::SparseIndex,
            ROOT,
            RestoreMethod::Replay { from, to: target },
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                scratch
                    .release(ledger)
                    .expect("the admitting ledger owns sparse-index replay scratch");
                return Err(error);
            }
        };
        self.current.bytes_mut().copy_from_slice(scratch.bytes());
        self.prefix = target;
        scratch
            .release(ledger)
            .expect("the admitting ledger owns sparse-index replay scratch");
        Ok(evidence)
    }

    /// Commit a successful replay in the logical state authority.
    pub fn rollback_to(&mut self, target: u64, evidence: Restore) -> Result<()> {
        self.require_open()?;
        if self.prefix != target {
            return Err(Error::InvalidRequest {
                field: "target",
                detail: format!(
                    "physical sparse index is at {}, not replay target {target}",
                    self.prefix
                ),
            });
        }
        self.sequence_mut()?.rollback_to(ROOT, target, &[evidence])
    }

    /// A maintained selection cannot be restored by dropping a prefix.
    pub fn truncate(&mut self, _target: u64) -> Result<()> {
        Err(Error::Unsupported {
            capability: "sparse_index_truncate",
            reason: "maintained selection requires an explicit snapshot or replay".into(),
        })
    }

    /// Release the current and prefix-zero host allocations and their logical
    /// control state.
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        self.current.release(ledger)?;
        self.start.release(ledger)?;
        self.sequence.take();
        Ok(())
    }

    fn next_prefix(&self) -> Result<u64> {
        self.prefix.checked_add(1).ok_or(Error::InvalidRequest {
            field: "prefix",
            detail: "sparse index prefix counter overflow".into(),
        })
    }

    fn require_open(&self) -> Result<()> {
        if self.sequence.is_none()
            || self.current.reservation_id().is_none()
            || self.start.reservation_id().is_none()
        {
            return Err(Error::InvalidRequest {
                field: "state",
                detail: "sparse index has been released".into(),
            });
        }
        Ok(())
    }

    fn require_aligned(&self) -> Result<()> {
        let frontiers = self.sequence_ref()?.frontiers(ROOT)?;
        if frontiers.accepted != self.prefix || frontiers.executed != self.prefix {
            return Err(Error::InvalidRequest {
                field: "state",
                detail: format!(
                    "physical prefix {} does not match logical accepted/executed {}/{}",
                    self.prefix, frontiers.accepted, frontiers.executed
                ),
            });
        }
        Ok(())
    }

    fn sequence_ref(&self) -> Result<&SequenceState> {
        self.sequence.as_ref().ok_or(Error::InvalidRequest {
            field: "state",
            detail: "sparse index has been released".into(),
        })
    }

    fn sequence_mut(&mut self) -> Result<&mut SequenceState> {
        self.sequence.as_mut().ok_or(Error::InvalidRequest {
            field: "state",
            detail: "sparse index has been released".into(),
        })
    }
}

fn btree_node_bound(entry: usize) -> usize {
    // Match the bounded control reserve used by PagedSequence for one node.
    11 * (entry + core::mem::size_of::<usize>()) + 16 * core::mem::size_of::<usize>()
}

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
