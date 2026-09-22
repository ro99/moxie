//! Physical host storage for explicitly restorable accumulated state.
//!
//! The store is deliberately model-independent.  It owns opaque bytes and
//! asks its caller to supply the deterministic step function that knows how
//! those bytes are updated.  [`SequenceState`] remains the authority for
//! branch identity, prefix lineage and rollback evidence; this module only
//! makes that evidence correspond to actual bytes.

use core::marker::PhantomData;

use moxie_memory::{HostBuffer, Ledger};
use moxie_types::{BranchId, Error, HostTier, Result, StateTransactionId, Tier};

use crate::{
    Branch, Journal, PrefixLineage, ROOT, Restore, RestoreMethod, SequenceState, StateKind,
};

mod sealed {
    pub trait StateStoreKind {}
}

/// The three explicit-restore stores differ only in the state kind and their
/// diagnostic vocabulary. The zero-sized marker keeps that choice out of the
/// admitted control-byte size while the implementation remains one type.
pub trait StateStoreKind: sealed::StateStoreKind {
    const KIND: StateKind;
}

#[derive(Clone, Copy, Debug)]
pub struct RecurrentStoreKind;

impl sealed::StateStoreKind for RecurrentStoreKind {}

impl StateStoreKind for RecurrentStoreKind {
    const KIND: StateKind = StateKind::RecurrentAccumulator;
}

#[derive(Clone, Copy, Debug)]
pub struct ConvolutionStoreKind;

impl sealed::StateStoreKind for ConvolutionStoreKind {}

impl StateStoreKind for ConvolutionStoreKind {
    const KIND: StateKind = StateKind::ConvolutionHistory;
}

#[derive(Clone, Copy, Debug)]
pub struct SparseIndexStoreKind;

impl sealed::StateStoreKind for SparseIndexStoreKind {}

impl StateStoreKind for SparseIndexStoreKind {
    const KIND: StateKind = StateKind::SparseIndex;
}

#[derive(Clone, Copy)]
struct StoreText {
    initial_field: &'static str,
    initial_detail: &'static str,
    state_label: &'static str,
    start_label: &'static str,
    current_ownership: &'static str,
    start_ownership: &'static str,
    step_label: &'static str,
    step_ownership: &'static str,
    begin_ownership: &'static str,
    commit_ownership: &'static str,
    snapshot_label: &'static str,
    snapshot_mismatch: &'static str,
    replay_label: &'static str,
    replay_ownership: &'static str,
    rollback_label: &'static str,
    prefix_overflow: &'static str,
    released: &'static str,
    truncate_capability: &'static str,
    truncate_reason: &'static str,
}

fn store_text(kind: StateKind) -> StoreText {
    match kind {
        StateKind::RecurrentAccumulator => StoreText {
            initial_field: "initial",
            initial_detail: "recurrent accumulator state must contain at least one byte",
            state_label: "recurrent accumulator state",
            start_label: "recurrent accumulator start",
            current_ownership: "the admitting ledger owns current accumulator bytes",
            start_ownership: "the admitting ledger owns start accumulator bytes",
            step_label: "recurrent accumulator step",
            step_ownership: "the admitting ledger owns step scratch",
            begin_ownership: "the new accumulator transaction is open",
            commit_ownership: "the failed accumulator commit remains abortable",
            snapshot_label: "recurrent accumulator snapshot",
            snapshot_mismatch: "snapshot does not belong to this accumulator's current lineage",
            replay_label: "recurrent accumulator replay",
            replay_ownership: "the admitting ledger owns replay scratch",
            rollback_label: "physical accumulator",
            prefix_overflow: "recurrent accumulator prefix counter overflow",
            released: "recurrent accumulator has been released",
            truncate_capability: "recurrent_accumulator_truncate",
            truncate_reason: "accumulated state requires an explicit snapshot or replay",
        },
        StateKind::ConvolutionHistory => StoreText {
            initial_field: "initial_window",
            initial_detail: "convolution history must retain at least one byte",
            state_label: "convolution history state",
            start_label: "convolution history start",
            current_ownership: "the admitting ledger owns current convolution bytes",
            start_ownership: "the admitting ledger owns start convolution bytes",
            step_label: "convolution history step",
            step_ownership: "the admitting ledger owns convolution step scratch",
            begin_ownership: "the new convolution transaction is open",
            commit_ownership: "the failed convolution commit remains abortable",
            snapshot_label: "convolution history snapshot",
            snapshot_mismatch: "snapshot does not belong to this convolution history's current lineage",
            replay_label: "convolution history replay",
            replay_ownership: "the admitting ledger owns convolution replay scratch",
            rollback_label: "physical convolution history",
            prefix_overflow: "convolution history prefix counter overflow",
            released: "convolution history has been released",
            truncate_capability: "convolution_history_truncate",
            truncate_reason: "raw window history requires an explicit snapshot or replay",
        },
        StateKind::SparseIndex => StoreText {
            initial_field: "initial",
            initial_detail: "sparse index state must contain at least one byte",
            state_label: "sparse index state",
            start_label: "sparse index start",
            current_ownership: "the admitting ledger owns current sparse-index bytes",
            start_ownership: "the admitting ledger owns start sparse-index bytes",
            step_label: "sparse index step",
            step_ownership: "the admitting ledger owns sparse-index step scratch",
            begin_ownership: "the new sparse-index transaction is open",
            commit_ownership: "the failed sparse-index commit remains abortable",
            snapshot_label: "sparse index snapshot",
            snapshot_mismatch: "snapshot does not belong to this sparse index's current lineage",
            replay_label: "sparse index replay",
            replay_ownership: "the admitting ledger owns sparse-index replay scratch",
            rollback_label: "physical sparse index",
            prefix_overflow: "sparse index prefix counter overflow",
            released: "sparse index has been released",
            truncate_capability: "sparse_index_truncate",
            truncate_reason: "maintained selection requires an explicit snapshot or replay",
        },
        _ => unreachable!("unsupported bounded state-store kind"),
    }
}

/// A physical snapshot of one bounded explicit-restore state prefix.
///
/// The payload has its own ledger reservation and must be explicitly released,
/// just like the other host-backed state owned by this crate.  Dropping it
/// without releasing it intentionally leaves the charge visible to the ledger.
#[derive(Debug)]
pub struct BoundedSnapshot<K: StateStoreKind> {
    prefix: u64,
    evidence: Restore,
    bytes: HostBuffer,
    marker: PhantomData<K>,
}

impl<K: StateStoreKind> BoundedSnapshot<K> {
    /// The accepted prefix represented by this snapshot.
    pub const fn prefix(&self) -> u64 {
        self.prefix
    }

    /// The logical restore evidence stamped when the snapshot was taken.
    pub const fn evidence(&self) -> Restore {
        self.evidence
    }

    /// The exact opaque bytes captured at [`Self::prefix`].
    pub fn bytes(&self) -> &[u8] {
        self.bytes.bytes()
    }

    /// Release this snapshot's admitted host bytes.
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        self.bytes.release(ledger)
    }
}

/// Where a bounded explicit-restore replay starts.
#[derive(Debug, Clone, Copy)]
pub enum BoundedReplaySource<'a, K: StateStoreKind> {
    /// Re-run from the store's initial bytes at prefix zero.
    Start,
    /// Re-run from an exact snapshot owned by this store's sequence.
    Snapshot(&'a BoundedSnapshot<K>),
}

/// A host-backed, model-independent bounded explicit-restore state store.
///
/// The current state and the prefix-zero source are separate admitted host
/// buffers.  Replay uses a third, short-lived admitted buffer, so a failed
/// step can mutate only scratch bytes.  The current state is replaced only
/// after the whole replay and the existing [`SequenceState::restore_evidence`]
/// check have succeeded.
#[derive(Debug)]
pub struct BoundedStateStore<K: StateStoreKind> {
    sequence: Option<SequenceState>,
    start: HostBuffer,
    current: HostBuffer,
    prefix: u64,
    max_prefix: u64,
    marker: PhantomData<K>,
}

impl<K: StateStoreKind> BoundedStateStore<K> {
    /// Create a bounded store whose opaque state has a fixed byte width and
    /// whose accepted prefix is bounded by `max_prefix`.
    pub fn new(ledger: &mut Ledger, initial: &[u8], max_prefix: u64) -> Result<Self> {
        let text = store_text(K::KIND);
        if initial.is_empty() {
            return Err(Error::InvalidRequest {
                field: text.initial_field,
                detail: text.initial_detail.into(),
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
            HostBuffer::allocate(ledger, text.state_label, initial.len(), control_bytes)?;
        let mut start = match HostBuffer::allocate(ledger, text.start_label, initial.len(), 0) {
            Ok(buffer) => buffer,
            Err(error) => {
                current.release(ledger).expect(text.current_ownership);
                return Err(error);
            }
        };
        current.bytes_mut().copy_from_slice(initial);
        start.bytes_mut().copy_from_slice(initial);
        let mut sequence = SequenceState::new([K::KIND]);
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
            start.release(ledger).expect(text.start_ownership);
            current.release(ledger).expect(text.current_ownership);
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
            marker: PhantomData,
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

    /// The exact opaque bytes currently held by the physical store.
    pub fn bytes(&self) -> &[u8] {
        self.current.bytes()
    }

    /// Advance one accepted step using caller-supplied deterministic algebra.
    ///
    /// The callback receives the input token/step payload and a private copy
    /// of the current bytes.  A callback failure, including one after it has
    /// partially mutated that copy, leaves both physical and logical state
    /// unchanged.
    pub fn advance<F>(&mut self, ledger: &mut Ledger, input: &[u8], mut step: F) -> Result<()>
    where
        F: FnMut(&[u8], &mut [u8]) -> Result<()>,
    {
        let text = store_text(K::KIND);
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
            HostBuffer::allocate(ledger, text.step_label, self.current.bytes().len(), 0)?;
        scratch.bytes_mut().copy_from_slice(self.current.bytes());
        if let Err(error) = step(input, scratch.bytes_mut()) {
            scratch.release(ledger).expect(text.step_ownership);
            return Err(error);
        }

        let sequence = self.sequence_mut()?;
        let transaction = match sequence.begin(ROOT) {
            Ok(transaction) => transaction,
            Err(error) => {
                scratch.release(ledger).expect(text.step_ownership);
                return Err(error);
            }
        };
        if let Err(error) = sequence.execute(ROOT, 1) {
            sequence.abort(transaction).expect(text.begin_ownership);
            scratch.release(ledger).expect(text.step_ownership);
            return Err(error);
        }
        if let Err(error) = sequence.commit_prefix(transaction, 1) {
            sequence.abort(transaction).expect(text.commit_ownership);
            scratch.release(ledger).expect(text.step_ownership);
            return Err(error);
        }
        self.current.bytes_mut().copy_from_slice(scratch.bytes());
        self.prefix = next_prefix;
        scratch.release(ledger).expect(text.step_ownership);
        Ok(())
    }

    /// Capture the current bytes and stamp existing snapshot evidence.
    pub fn snapshot(&self, ledger: &mut Ledger) -> Result<BoundedSnapshot<K>> {
        let text = store_text(K::KIND);
        self.require_open()?;
        self.require_aligned()?;
        let evidence = self.sequence_ref()?.restore_evidence(
            K::KIND,
            ROOT,
            RestoreMethod::Snapshot {
                of_prefix: self.prefix,
            },
        )?;
        let mut bytes = HostBuffer::allocate(
            ledger,
            text.snapshot_label,
            self.current.bytes().len(),
            core::mem::size_of::<BoundedSnapshot<K>>(),
        )?;
        bytes.bytes_mut().copy_from_slice(self.current.bytes());
        Ok(BoundedSnapshot {
            prefix: self.prefix,
            evidence,
            bytes,
            marker: PhantomData,
        })
    }

    /// Reconstruct a target prefix by replaying the supplied inputs.
    ///
    /// The target must already be occupied in the logical sequence.  The
    /// returned evidence can be passed to [`Self::rollback_to`] to commit the
    /// same target in the logical state.  Until then, the physical prefix is
    /// intentionally allowed to differ from the logical frontier, and methods
    /// that would append or snapshot refuse that mismatch.
    pub fn replay<F>(
        &mut self,
        ledger: &mut Ledger,
        source: BoundedReplaySource<'_, K>,
        target: u64,
        inputs: &[&[u8]],
        mut step: F,
    ) -> Result<Restore>
    where
        F: FnMut(&[u8], &mut [u8]) -> Result<()>,
    {
        let text = store_text(K::KIND);
        self.require_open()?;
        let (from, source_bytes) = match source {
            BoundedReplaySource::Start => (0, self.start.bytes()),
            BoundedReplaySource::Snapshot(snapshot) => {
                let sequence = self.sequence_ref()?;
                if snapshot.evidence.sequence() != sequence.id()
                    || snapshot.evidence.branch() != ROOT
                    || snapshot.evidence.generation() != sequence.generation()
                    || snapshot.evidence.kind() != K::KIND
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
                        detail: text.snapshot_mismatch.into(),
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
        // Evidence must name an actually occupied prefix and the current
        // lineage.  This check is done before allocating or mutating scratch.
        let _ = self
            .sequence_ref()?
            .lineage_at(ROOT, target)?
            .ok_or_else(|| Error::InvalidRequest {
                field: "target",
                detail: format!("prefix {target} is not occupied in the logical sequence"),
            })?;

        let mut scratch =
            HostBuffer::allocate(ledger, text.replay_label, self.current.bytes().len(), 0)?;
        scratch.bytes_mut().copy_from_slice(source_bytes);
        for input in inputs {
            if let Err(error) = step(input, scratch.bytes_mut()) {
                scratch.release(ledger).expect(text.replay_ownership);
                return Err(error);
            }
        }

        let evidence = match self.sequence_ref()?.restore_evidence(
            K::KIND,
            ROOT,
            RestoreMethod::Replay { from, to: target },
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                scratch.release(ledger).expect(text.replay_ownership);
                return Err(error);
            }
        };
        self.current.bytes_mut().copy_from_slice(scratch.bytes());
        self.prefix = target;
        scratch.release(ledger).expect(text.replay_ownership);
        Ok(evidence)
    }

    /// Commit a successful replay in the logical state authority.
    ///
    /// All validation, including identity, lineage, generation and explicit
    /// coverage, is delegated to [`SequenceState::rollback_to`].
    pub fn rollback_to(&mut self, target: u64, evidence: Restore) -> Result<()> {
        let text = store_text(K::KIND);
        self.require_open()?;
        if self.prefix != target {
            return Err(Error::InvalidRequest {
                field: "target",
                detail: format!(
                    "{} is at {}, not replay target {target}",
                    text.rollback_label, self.prefix
                ),
            });
        }
        self.sequence_mut()?.rollback_to(ROOT, target, &[evidence])
    }

    /// Explicit-restore state cannot be restored by dropping a prefix.
    pub fn truncate(&mut self, _target: u64) -> Result<()> {
        let text = store_text(K::KIND);
        Err(Error::Unsupported {
            capability: text.truncate_capability,
            reason: text.truncate_reason.into(),
        })
    }

    /// Release the current and prefix-zero host allocations.
    pub fn release(&mut self, ledger: &mut Ledger) -> Result<()> {
        self.current.release(ledger)?;
        self.start.release(ledger)?;
        self.sequence.take();
        Ok(())
    }

    fn next_prefix(&self) -> Result<u64> {
        self.prefix.checked_add(1).ok_or(Error::InvalidRequest {
            field: "prefix",
            detail: store_text(K::KIND).prefix_overflow.into(),
        })
    }

    fn require_open(&self) -> Result<()> {
        if self.sequence.is_none()
            || self.current.reservation_id().is_none()
            || self.start.reservation_id().is_none()
        {
            return Err(Error::InvalidRequest {
                field: "state",
                detail: store_text(K::KIND).released.into(),
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
            detail: store_text(K::KIND).released.into(),
        })
    }

    fn sequence_mut(&mut self) -> Result<&mut SequenceState> {
        self.sequence.as_mut().ok_or(Error::InvalidRequest {
            field: "state",
            detail: store_text(K::KIND).released.into(),
        })
    }
}

fn btree_node_bound(entry: usize) -> usize {
    // Match the bounded control reserve used by PagedSequence for one node.
    11 * (entry + core::mem::size_of::<usize>()) + 16 * core::mem::size_of::<usize>()
}

pub type RecurrentAccumulator = BoundedStateStore<RecurrentStoreKind>;
pub type RecurrentSnapshot = BoundedSnapshot<RecurrentStoreKind>;
pub type ReplaySource<'a> = BoundedReplaySource<'a, RecurrentStoreKind>;

const _: fn() = || {
    fn assert_clone_copy<T: Clone + Copy>() {}
    assert_clone_copy::<ReplaySource<'static>>();
    assert_clone_copy::<crate::convolution::ConvolutionReplaySource<'static>>();
    assert_clone_copy::<crate::sparse_index::SparseIndexReplaySource<'static>>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_memory::{CapacitySnapshot, Ledger};
    use moxie_types::{Error, HostTier, Scope, Tier};

    const WIDTH: usize = 32;

    fn ledger() -> Ledger {
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).unwrap()]).unwrap()
    }

    fn initial() -> [u8; WIDTH] {
        let mut state = [0u8; WIDTH];
        for (lane, bytes) in state.chunks_exact_mut(8).enumerate() {
            bytes.copy_from_slice(&((lane as u64) + 1).to_le_bytes());
        }
        state
    }

    // A decay-and-mix accumulator, intentionally not a model equation.  Every
    // prior lane is changed before a new contribution is folded into the last
    // lane, so dropping the newest raw lane cannot recover the preceding state.
    fn synthetic_step(input: &[u8], state: &mut [u8]) -> Result<()> {
        let token = input.first().copied().ok_or(Error::InvalidRequest {
            field: "input",
            detail: "synthetic step needs one byte".into(),
        })? as u64;
        for lane in state.chunks_exact_mut(8) {
            let value = u64::from_le_bytes(lane.try_into().expect("eight-byte lane"));
            let next = value.wrapping_mul(3).wrapping_add(token).rotate_left(7);
            lane.copy_from_slice(&next.to_le_bytes());
        }
        let first = u64::from_le_bytes(state[..8].try_into().expect("first lane"));
        let last = first ^ token.wrapping_mul(0x9e37_79b9);
        state[WIDTH - 8..].copy_from_slice(&last.to_le_bytes());
        Ok(())
    }

    fn run_independently(mut state: Vec<u8>, inputs: &[u8]) -> Vec<u8> {
        for input in inputs {
            synthetic_step(&[*input], &mut state).unwrap();
        }
        state
    }

    fn input_refs(inputs: &[u8]) -> Vec<&[u8]> {
        inputs.iter().map(std::slice::from_ref).collect()
    }

    #[test]
    fn truncation_is_a_typed_refusal_and_does_not_change_bytes() {
        let mut owner = ledger();
        let initial = initial();
        let mut accumulator = RecurrentAccumulator::new(&mut owner, &initial, 1).unwrap();
        accumulator
            .advance(&mut owner, &[5], synthetic_step)
            .unwrap();
        let before = accumulator.bytes().to_vec();

        assert!(matches!(
            accumulator.advance(&mut owner, &[9], synthetic_step),
            Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                ..
            })
        ));

        assert!(matches!(
            accumulator.truncate(0),
            Err(Error::Unsupported {
                capability: "recurrent_accumulator_truncate",
                ..
            })
        ));
        assert_eq!(accumulator.bytes(), before);
        assert_eq!(accumulator.prefix(), 1);
        accumulator.release(&mut owner).unwrap();
        assert_eq!(
            owner.committed(Scope::Host, Tier::Host(moxie_types::HostTier::StateSpill)),
            0
        );
        assert!(owner.outstanding().is_empty());
        assert!(accumulator.sequence().is_err());
    }

    #[test]
    fn snapshot_and_replay_restore_exact_bytes_and_use_existing_evidence() {
        let mut owner = ledger();
        let initial = initial();
        let inputs = [3u8, 7, 11, 19, 23];
        let mut accumulator = RecurrentAccumulator::new(&mut owner, &initial, 5).unwrap();
        for input in &inputs[..2] {
            accumulator
                .advance(&mut owner, &[*input], synthetic_step)
                .unwrap();
        }
        let mut snapshot = accumulator.snapshot(&mut owner).unwrap();
        let at_snapshot = snapshot.bytes().to_vec();
        assert_eq!(
            at_snapshot,
            run_independently(initial.to_vec(), &inputs[..2])
        );
        assert_eq!(
            snapshot.evidence().method(),
            RestoreMethod::Snapshot { of_prefix: 2 }
        );

        for input in &inputs[2..] {
            accumulator
                .advance(&mut owner, &[*input], synthetic_step)
                .unwrap();
        }
        assert_eq!(snapshot.bytes(), at_snapshot);
        let after = accumulator.bytes().to_vec();
        let forward_inputs = input_refs(&inputs[2..]);
        let replayed_forward = accumulator
            .replay(
                &mut owner,
                ReplaySource::Snapshot(&snapshot),
                5,
                &forward_inputs,
                synthetic_step,
            )
            .unwrap();
        assert_eq!(accumulator.bytes(), after);
        assert_eq!(
            replayed_forward.method(),
            RestoreMethod::Replay { from: 2, to: 5 }
        );

        let backward_inputs = input_refs(&inputs[..2]);
        let replayed_back = accumulator
            .replay(
                &mut owner,
                ReplaySource::Start,
                2,
                &backward_inputs,
                synthetic_step,
            )
            .unwrap();
        assert_eq!(accumulator.bytes(), at_snapshot);
        assert_eq!(
            accumulator.bytes(),
            run_independently(initial.to_vec(), &inputs[..2])
        );
        assert_eq!(
            replayed_back.method(),
            RestoreMethod::Replay { from: 0, to: 2 }
        );
        accumulator.rollback_to(2, replayed_back).unwrap();
        assert_eq!(
            accumulator
                .sequence()
                .unwrap()
                .frontiers(ROOT)
                .unwrap()
                .accepted,
            2
        );
        snapshot.release(&mut owner).unwrap();
        accumulator.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn snapshot_release_refuses_wrong_ledger_without_consuming_snapshot() {
        let mut owner = ledger();
        let mut wrong = ledger();
        let initial = initial();
        let mut accumulator = RecurrentAccumulator::new(&mut owner, &initial, 1).unwrap();
        let mut snapshot = accumulator.snapshot(&mut owner).unwrap();
        let bytes = snapshot.bytes().to_vec();

        assert!(matches!(
            snapshot.release(&mut wrong),
            Err(Error::InvalidRequest {
                field: "ledger",
                ..
            })
        ));
        assert_eq!(snapshot.bytes(), bytes);
        snapshot.release(&mut owner).unwrap();
        accumulator.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
        assert!(wrong.outstanding().is_empty());
    }

    #[test]
    fn decay_and_mix_cannot_be_recovered_by_dropping_the_newest_raw_lane() {
        let mut owner = ledger();
        let initial = initial();
        let mut accumulator = RecurrentAccumulator::new(&mut owner, &initial, 2).unwrap();
        accumulator
            .advance(&mut owner, &[13], synthetic_step)
            .unwrap();
        let at_n = accumulator.bytes().to_vec();
        accumulator
            .advance(&mut owner, &[29], synthetic_step)
            .unwrap();
        let at_n_plus_one = accumulator.bytes().to_vec();
        let naive_truncation = &at_n_plus_one[..at_n_plus_one.len() - 8];
        assert_ne!(naive_truncation, &at_n[..at_n.len() - 8]);
        assert_ne!(at_n_plus_one, at_n);
        accumulator.release(&mut owner).unwrap();
    }

    #[test]
    fn failed_replay_keeps_the_pre_attempt_bytes_and_frontier() {
        let mut owner = ledger();
        let initial = initial();
        let inputs = [2u8, 5, 8, 13, 21];
        let mut accumulator = RecurrentAccumulator::new(&mut owner, &initial, 5).unwrap();
        for input in &inputs {
            accumulator
                .advance(&mut owner, &[*input], synthetic_step)
                .unwrap();
        }
        let before = accumulator.bytes().to_vec();
        let frontier_before = accumulator.sequence().unwrap().frontiers(ROOT).unwrap();
        let charge_before = owner.scope_committed(Scope::Host);
        let mut calls = 0;
        let replay_inputs = input_refs(&inputs);
        let error = accumulator
            .replay(
                &mut owner,
                ReplaySource::Start,
                5,
                &replay_inputs,
                |input, state| {
                    synthetic_step(input, state)?;
                    calls += 1;
                    if calls == 3 {
                        Err(Error::Cancelled {
                            at: "recurrent replay",
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
                at: "recurrent replay"
            }
        );
        assert_eq!(calls, 3);
        assert_eq!(accumulator.bytes(), before);
        assert_eq!(accumulator.prefix(), 5);
        assert_eq!(
            accumulator.sequence().unwrap().frontiers(ROOT).unwrap(),
            frontier_before
        );
        assert_eq!(owner.scope_committed(Scope::Host), charge_before);
        accumulator.release(&mut owner).unwrap();
    }
}
