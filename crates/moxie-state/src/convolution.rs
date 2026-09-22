//! Thin public naming for the generic bounded explicit-restore store.
//!
//! The implementation lives in the accumulator module; this alias preserves
//! the convolution-history API and its StateKind selection.

use crate::accumulator::{
    BoundedReplaySource, BoundedSnapshot, BoundedStateStore, ConvolutionStoreKind,
};
#[cfg(test)]
use crate::{ROOT, RestoreMethod};
#[cfg(test)]
use moxie_types::Result;

pub type ConvolutionHistory = BoundedStateStore<ConvolutionStoreKind>;
pub type ConvolutionHistorySnapshot = BoundedSnapshot<ConvolutionStoreKind>;
pub type ConvolutionReplaySource<'a> = BoundedReplaySource<'a, ConvolutionStoreKind>;

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
