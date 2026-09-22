//! Physical host storage for explicitly restorable raw convolution history.
//!
//! The store is deliberately model-independent. It owns one fixed-width raw
//! sliding window and asks its caller to supply the step that shifts a new
//! value into that window. [`SequenceState`] remains the authority for branch
//! identity, prefix lineage and rollback evidence; this module only makes that
//! evidence correspond to actual bytes.

use moxie_memory::{HostBuffer, Ledger};
use moxie_types::{BranchId, Error, HostTier, Result, StateTransactionId, Tier};

use crate::{
    Branch, Journal, PrefixLineage, ROOT, Restore, RestoreMethod, SequenceState, StateKind,
};

/// A physical snapshot of one raw convolution-history window.
#[derive(Debug)]
pub struct ConvolutionHistorySnapshot {
    prefix: u64,
    evidence: Restore,
    bytes: HostBuffer,
}

impl ConvolutionHistorySnapshot {
    /// The accepted prefix represented by this snapshot.
    pub const fn prefix(&self) -> u64 {
        self.prefix
    }

    /// The logical restore evidence stamped when the snapshot was taken.
    pub const fn evidence(&self) -> Restore {
        self.evidence
    }

    /// The exact raw window captured at [`Self::prefix`].
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

/// Where a convolution-history replay starts.
#[derive(Debug, Clone, Copy)]
pub enum ConvolutionReplaySource<'a> {
    /// Re-run from the initial raw window at prefix zero.
    Start,
    /// Re-run from an exact snapshot owned by this history's sequence.
    Snapshot(&'a ConvolutionHistorySnapshot),
}

/// A host-backed, model-independent raw sliding-window history.
///
/// The current window and the prefix-zero source are separate admitted host
/// buffers. Replay uses a third, short-lived admitted buffer, so a failed step
/// can mutate only scratch bytes. The current window is replaced only after the
/// whole replay and the existing [`SequenceState::restore_evidence`] check have
/// succeeded.
#[derive(Debug)]
pub struct ConvolutionHistory {
    sequence: Option<SequenceState>,
    start: HostBuffer,
    current: HostBuffer,
    prefix: u64,
    max_prefix: u64,
}

impl ConvolutionHistory {
    /// Create a history with a fixed-width initial raw window and a bounded
    /// accepted prefix. Every supplied step must preserve that byte width.
    pub fn new(ledger: &mut Ledger, initial_window: &[u8], max_prefix: u64) -> Result<Self> {
        if initial_window.is_empty() {
            return Err(Error::InvalidRequest {
                field: "initial_window",
                detail: "convolution history must retain at least one byte".into(),
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

        let mut current = HostBuffer::allocate(
            ledger,
            "convolution history state",
            initial_window.len(),
            control_bytes,
        )?;
        let mut start = match HostBuffer::allocate(
            ledger,
            "convolution history start",
            initial_window.len(),
            0,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                current
                    .release(ledger)
                    .expect("the admitting ledger owns current convolution bytes");
                return Err(error);
            }
        };
        current.bytes_mut().copy_from_slice(initial_window);
        start.bytes_mut().copy_from_slice(initial_window);

        let mut sequence = SequenceState::new([StateKind::ConvolutionHistory]);
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
                .expect("the admitting ledger owns start convolution bytes");
            current
                .release(ledger)
                .expect("the admitting ledger owns current convolution bytes");
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

    /// The exact raw window currently held by the physical store.
    pub fn bytes(&self) -> &[u8] {
        self.current.bytes()
    }

    /// Shift one caller-defined raw value into the window.
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
        let mut scratch = HostBuffer::allocate(
            ledger,
            "convolution history step",
            self.current.bytes().len(),
            0,
        )?;
        scratch.bytes_mut().copy_from_slice(self.current.bytes());
        if let Err(error) = step(input, scratch.bytes_mut()) {
            scratch
                .release(ledger)
                .expect("the admitting ledger owns convolution step scratch");
            return Err(error);
        }

        let sequence = self.sequence_mut()?;
        let transaction = match sequence.begin(ROOT) {
            Ok(transaction) => transaction,
            Err(error) => {
                scratch
                    .release(ledger)
                    .expect("the admitting ledger owns convolution step scratch");
                return Err(error);
            }
        };
        if let Err(error) = sequence.execute(ROOT, 1) {
            sequence
                .abort(transaction)
                .expect("the new convolution transaction is open");
            scratch
                .release(ledger)
                .expect("the admitting ledger owns convolution step scratch");
            return Err(error);
        }
        if let Err(error) = sequence.commit_prefix(transaction, 1) {
            sequence
                .abort(transaction)
                .expect("the failed convolution commit remains abortable");
            scratch
                .release(ledger)
                .expect("the admitting ledger owns convolution step scratch");
            return Err(error);
        }
        self.current.bytes_mut().copy_from_slice(scratch.bytes());
        self.prefix = next_prefix;
        scratch
            .release(ledger)
            .expect("the admitting ledger owns convolution step scratch");
        Ok(())
    }

    /// Capture the current raw window and stamp existing snapshot evidence.
    pub fn snapshot(&self, ledger: &mut Ledger) -> Result<ConvolutionHistorySnapshot> {
        self.require_open()?;
        self.require_aligned()?;
        let evidence = self.sequence_ref()?.restore_evidence(
            StateKind::ConvolutionHistory,
            ROOT,
            RestoreMethod::Snapshot {
                of_prefix: self.prefix,
            },
        )?;
        let mut bytes = HostBuffer::allocate(
            ledger,
            "convolution history snapshot",
            self.current.bytes().len(),
            core::mem::size_of::<ConvolutionHistorySnapshot>(),
        )?;
        bytes.bytes_mut().copy_from_slice(self.current.bytes());
        Ok(ConvolutionHistorySnapshot {
            prefix: self.prefix,
            evidence,
            bytes,
        })
    }

    /// Reconstruct a target prefix by replaying the supplied raw-window steps.
    pub fn replay<F>(
        &mut self,
        ledger: &mut Ledger,
        source: ConvolutionReplaySource<'_>,
        target: u64,
        inputs: &[&[u8]],
        mut step: F,
    ) -> Result<Restore>
    where
        F: FnMut(&[u8], &mut [u8]) -> Result<()>,
    {
        self.require_open()?;
        let (from, source_bytes) = match source {
            ConvolutionReplaySource::Start => (0, self.start.bytes()),
            ConvolutionReplaySource::Snapshot(snapshot) => {
                let sequence = self.sequence_ref()?;
                if snapshot.evidence.sequence() != sequence.id()
                    || snapshot.evidence.branch() != ROOT
                    || snapshot.evidence.generation() != sequence.generation()
                    || snapshot.evidence.kind() != StateKind::ConvolutionHistory
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
                        detail:
                            "snapshot does not belong to this convolution history's current lineage"
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

        let mut scratch = HostBuffer::allocate(
            ledger,
            "convolution history replay",
            self.current.bytes().len(),
            0,
        )?;
        scratch.bytes_mut().copy_from_slice(source_bytes);
        for input in inputs {
            if let Err(error) = step(input, scratch.bytes_mut()) {
                scratch
                    .release(ledger)
                    .expect("the admitting ledger owns convolution replay scratch");
                return Err(error);
            }
        }

        let evidence = match self.sequence_ref()?.restore_evidence(
            StateKind::ConvolutionHistory,
            ROOT,
            RestoreMethod::Replay { from, to: target },
        ) {
            Ok(evidence) => evidence,
            Err(error) => {
                scratch
                    .release(ledger)
                    .expect("the admitting ledger owns convolution replay scratch");
                return Err(error);
            }
        };
        self.current.bytes_mut().copy_from_slice(scratch.bytes());
        self.prefix = target;
        scratch
            .release(ledger)
            .expect("the admitting ledger owns convolution replay scratch");
        Ok(evidence)
    }

    /// Commit a successful replay in the logical state authority.
    pub fn rollback_to(&mut self, target: u64, evidence: Restore) -> Result<()> {
        self.require_open()?;
        if self.prefix != target {
            return Err(Error::InvalidRequest {
                field: "target",
                detail: format!(
                    "physical convolution history is at {}, not replay target {target}",
                    self.prefix
                ),
            });
        }
        self.sequence_mut()?.rollback_to(ROOT, target, &[evidence])
    }

    /// A raw history window cannot be restored by dropping a prefix.
    pub fn truncate(&mut self, _target: u64) -> Result<()> {
        Err(Error::Unsupported {
            capability: "convolution_history_truncate",
            reason: "raw window history requires an explicit snapshot or replay".into(),
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
            detail: "convolution history prefix counter overflow".into(),
        })
    }

    fn require_open(&self) -> Result<()> {
        if self.sequence.is_none()
            || self.current.reservation_id().is_none()
            || self.start.reservation_id().is_none()
        {
            return Err(Error::InvalidRequest {
                field: "state",
                detail: "convolution history has been released".into(),
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
            detail: "convolution history has been released".into(),
        })
    }

    fn sequence_mut(&mut self) -> Result<&mut SequenceState> {
        self.sequence.as_mut().ok_or(Error::InvalidRequest {
            field: "state",
            detail: "convolution history has been released".into(),
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
    use moxie_types::{Error, HostTier, Scope, Tier};

    const WINDOW: [u8; 3] = [0, 0, 0];

    fn ledger() -> Ledger {
        Ledger::new([CapacitySnapshot::new(Scope::Host, 1 << 20, 1 << 10).unwrap()]).unwrap()
    }

    /// A raw sliding history: no value is mixed or transformed; each step
    /// shifts the old bytes and stores the one new value at the right edge.
    fn shift_window(input: &[u8], window: &mut [u8]) -> Result<()> {
        let value = input.first().copied().ok_or(Error::InvalidRequest {
            field: "input",
            detail: "synthetic window step needs one byte".into(),
        })?;
        if window.is_empty() {
            return Err(Error::InvalidRequest {
                field: "window",
                detail: "synthetic window must not be empty".into(),
            });
        }
        window.rotate_left(1);
        *window.last_mut().expect("nonempty window") = value;
        Ok(())
    }

    fn run_independently(mut window: Vec<u8>, inputs: &[u8]) -> Vec<u8> {
        for input in inputs {
            shift_window(&[*input], &mut window).unwrap();
        }
        window
    }

    fn input_refs(inputs: &[u8]) -> Vec<&[u8]> {
        inputs.iter().map(std::slice::from_ref).collect()
    }

    #[test]
    fn window_loss_is_geometric_and_current_bytes_are_ambiguous() {
        let mut owner = ledger();
        let mut first = ConvolutionHistory::new(&mut owner, &WINDOW, 4).unwrap();
        let mut second = ConvolutionHistory::new(&mut owner, &WINDOW, 4).unwrap();
        for input in [10u8, 20, 30] {
            first.advance(&mut owner, &[input], shift_window).unwrap();
        }
        for input in [99u8, 20, 30] {
            second.advance(&mut owner, &[input], shift_window).unwrap();
        }
        let first_before_loss = first.bytes().to_vec();
        let second_before_loss = second.bytes().to_vec();
        assert_ne!(first_before_loss, second_before_loss);

        first.advance(&mut owner, &[40], shift_window).unwrap();
        second.advance(&mut owner, &[40], shift_window).unwrap();
        assert_eq!(first.bytes(), &[20, 30, 40]);
        assert_eq!(first.bytes(), second.bytes());
        assert!(!first.bytes().contains(&10));
        assert!(!second.bytes().contains(&99));

        first.release(&mut owner).unwrap();
        second.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn truncation_and_prefix_bound_are_typed_and_leave_the_window_unchanged() {
        let mut owner = ledger();
        let mut history = ConvolutionHistory::new(&mut owner, &WINDOW, 2).unwrap();
        history.advance(&mut owner, &[7], shift_window).unwrap();
        history.advance(&mut owner, &[8], shift_window).unwrap();
        let before = history.bytes().to_vec();

        assert!(matches!(
            history.advance(&mut owner, &[9], shift_window),
            Err(Error::CapacityExceeded {
                tier: Some(Tier::Host(HostTier::Pageable)),
                ..
            })
        ));
        assert!(matches!(
            history.truncate(0),
            Err(Error::Unsupported {
                capability: "convolution_history_truncate",
                ..
            })
        ));
        assert_eq!(history.bytes(), before);
        assert_eq!(history.prefix(), 2);

        history.release(&mut owner).unwrap();
        assert!(history.sequence().is_err());
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn snapshot_and_replay_restore_exact_windows_against_an_independent_run() {
        let mut owner = ledger();
        let inputs = [10u8, 20, 30, 40, 50];
        let mut history = ConvolutionHistory::new(&mut owner, &WINDOW, 5).unwrap();
        for input in &inputs[..2] {
            history
                .advance(&mut owner, &[*input], shift_window)
                .unwrap();
        }
        let mut snapshot = history.snapshot(&mut owner).unwrap();
        let at_snapshot = snapshot.bytes().to_vec();
        assert_eq!(
            at_snapshot,
            run_independently(WINDOW.to_vec(), &inputs[..2])
        );

        for input in &inputs[2..] {
            history
                .advance(&mut owner, &[*input], shift_window)
                .unwrap();
        }
        let at_end = history.bytes().to_vec();
        assert_eq!(snapshot.bytes(), at_snapshot);
        assert_eq!(at_end, run_independently(WINDOW.to_vec(), &inputs));
        let replayed_forward = history
            .replay(
                &mut owner,
                ConvolutionReplaySource::Snapshot(&snapshot),
                5,
                &input_refs(&inputs[2..]),
                shift_window,
            )
            .unwrap();
        assert_eq!(history.bytes(), at_end);
        assert_eq!(
            replayed_forward.method(),
            RestoreMethod::Replay { from: 2, to: 5 }
        );

        let replayed_back = history
            .replay(
                &mut owner,
                ConvolutionReplaySource::Start,
                2,
                &input_refs(&inputs[..2]),
                shift_window,
            )
            .unwrap();
        assert_eq!(history.bytes(), at_snapshot);
        assert_eq!(
            history.bytes(),
            run_independently(WINDOW.to_vec(), &inputs[..2])
        );
        history.rollback_to(2, replayed_back).unwrap();
        assert_eq!(
            history
                .sequence()
                .unwrap()
                .frontiers(ROOT)
                .unwrap()
                .accepted,
            2
        );

        snapshot.release(&mut owner).unwrap();
        history.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
    }

    #[test]
    fn snapshot_release_refuses_wrong_ledger_without_consuming_the_snapshot() {
        let mut owner = ledger();
        let mut wrong = ledger();
        let mut history = ConvolutionHistory::new(&mut owner, &WINDOW, 1).unwrap();
        let mut snapshot = history.snapshot(&mut owner).unwrap();
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
        history.release(&mut owner).unwrap();
        assert!(owner.outstanding().is_empty());
        assert!(wrong.outstanding().is_empty());
    }

    #[test]
    fn failed_replay_preserves_window_frontier_and_charge() {
        let mut owner = ledger();
        let inputs = [1u8, 2, 3, 4];
        let mut history = ConvolutionHistory::new(&mut owner, &WINDOW, 4).unwrap();
        for input in &inputs {
            history
                .advance(&mut owner, &[*input], shift_window)
                .unwrap();
        }
        let before = history.bytes().to_vec();
        let frontier_before = history.sequence().unwrap().frontiers(ROOT).unwrap();
        let charge_before = owner.scope_committed(Scope::Host);
        let mut calls = 0;
        let error = history
            .replay(
                &mut owner,
                ConvolutionReplaySource::Start,
                4,
                &input_refs(&inputs),
                |input, window| {
                    shift_window(input, window)?;
                    calls += 1;
                    if calls == 3 {
                        Err(Error::Cancelled {
                            at: "convolution history replay",
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
                at: "convolution history replay"
            }
        );
        assert_eq!(calls, 3);
        assert_eq!(history.bytes(), before);
        assert_eq!(history.prefix(), 4);
        assert_eq!(
            history.sequence().unwrap().frontiers(ROOT).unwrap(),
            frontier_before
        );
        assert_eq!(owner.scope_committed(Scope::Host), charge_before);
        history.release(&mut owner).unwrap();
    }
}
