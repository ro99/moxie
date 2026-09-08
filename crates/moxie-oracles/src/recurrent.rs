//! Recurrent updates and convolution history.
//!
//! R20 and document 04: "Recurrent state cannot generally recover an earlier
//! state by decrementing a position counter. Retain bounded prefix snapshots or
//! recompute the accepted prefix from a saved state."
//!
//! `moxie-state` encodes that as a rule about rollback. This module is the
//! arithmetic the rule is about: a small, explicit recurrence and a short
//! convolution, with the replay equivalence document 07 requires ("compare the
//! committed target and draft after every accept/reject path to replay from a
//! clean saved prefix").
//!
//! Explicitly **not** pinned here: any real model's recurrence (Kimi's and
//! GLM-5.3's have their own gating and normalisation), selective/gated state
//! space forms, or the cost model for snapshot spacing.

use moxie_types::{Error, Result};

/// A first-order gated recurrence: `h <- decay * h + gain * x`.
///
/// Deliberately the simplest form with the property that matters: `decay` is
/// applied *before* the input is mixed in, so the map is order-dependent and,
/// for `decay` other than one, not invertible in a way a counter decrement could
/// exploit.
#[derive(Debug, Clone, PartialEq)]
pub struct Recurrence {
    pub decay: f32,
    pub gain: f32,
}

impl Recurrence {
    /// One step. `state` and `x` have the same width.
    pub fn step(&self, state: &mut [f32], x: &[f32]) -> Result<()> {
        if state.len() != x.len() {
            return Err(Error::InvalidRequest {
                field: "recurrent_state",
                detail: format!(
                    "state width {} against input width {}",
                    state.len(),
                    x.len()
                ),
            });
        }
        for (h, v) in state.iter_mut().zip(x.iter()) {
            *h = self.decay * *h + self.gain * v;
        }
        Ok(())
    }

    /// Run `inputs` from `initial`, returning the state after each step.
    ///
    /// The returned vector has `inputs.len() + 1` entries: index `n` is the
    /// state after consuming `n` inputs, so it is indexed by *prefix length*,
    /// which is the same thing `moxie-state` counts.
    pub fn trace(&self, initial: &[f32], inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut state = initial.to_vec();
        let mut out = vec![state.clone()];
        for x in inputs {
            self.step(&mut state, x)?;
            out.push(state.clone());
        }
        Ok(out)
    }

    /// Replay from a saved state at prefix `from` up to prefix `to`.
    ///
    /// This is the operation a rollback pays for. `inputs` is the whole token
    /// stream; only `from..to` is consumed.
    pub fn replay(
        &self,
        saved: &[f32],
        inputs: &[Vec<f32>],
        from: usize,
        to: usize,
    ) -> Result<Vec<f32>> {
        if from > to || to > inputs.len() {
            return Err(Error::InvalidRequest {
                field: "replay",
                detail: format!("{from}..{to} outside 0..={}", inputs.len()),
            });
        }
        let mut state = saved.to_vec();
        for x in &inputs[from..to] {
            self.step(&mut state, x)?;
        }
        Ok(state)
    }
}

/// A causal short convolution over the last `history.len()` positions.
///
/// Document 02 makes `ShortConv` its own operation. Its state is a window of
/// recent inputs, so an earlier state is recovered by *refilling the window*
/// from the token stream, not by decrementing anything.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortConv {
    /// Taps, most recent first: `y = sum(taps[i] * x[t - i])`.
    pub taps: Vec<f32>,
}

impl ShortConv {
    pub fn window(&self) -> usize {
        self.taps.len()
    }

    /// Output at position `t` of `xs`, with implicit zeros before position 0.
    pub fn at(&self, xs: &[f32], t: usize) -> Result<f32> {
        if self.taps.is_empty() {
            return Err(Error::InvalidRequest {
                field: "taps",
                detail: "a short convolution needs at least one tap".into(),
            });
        }
        if t >= xs.len() {
            return Err(Error::InvalidRequest {
                field: "position",
                detail: format!("position {t} outside {} input(s)", xs.len()),
            });
        }
        Ok(self
            .taps
            .iter()
            .enumerate()
            .map(|(i, tap)| if i <= t { tap * xs[t - i] } else { 0.0 })
            .sum())
    }

    /// The history window a decode step at position `t` needs.
    pub fn history_for(&self, xs: &[f32], t: usize) -> Vec<f32> {
        let start = t.saturating_sub(self.window() - 1);
        xs[start..=t.min(xs.len().saturating_sub(1))].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(n: usize) -> Vec<Vec<f32>> {
        (0..n).map(|i| vec![i as f32, -(i as f32) * 0.5]).collect()
    }

    #[test]
    fn a_replay_from_a_saved_prefix_reproduces_the_state_exactly() {
        // The equivalence a rollback depends on. Bit-exact, not approximate:
        // the same operations in the same order on the same values.
        let r = Recurrence {
            decay: 0.9,
            gain: 0.25,
        };
        let xs = stream(12);
        let trace = r.trace(&[0.0, 0.0], &xs).unwrap();

        for saved_at in 0..=12usize {
            for target in saved_at..=12 {
                let replayed = r.replay(&trace[saved_at], &xs, saved_at, target).unwrap();
                assert_eq!(replayed, trace[target], "saved {saved_at} -> {target}");
            }
        }
    }

    #[test]
    fn an_earlier_state_is_not_recoverable_from_a_later_one() {
        // R20 as arithmetic. Nothing about the state at prefix 10 lets you
        // reconstruct the state at prefix 8 -- the inputs are mixed in and the
        // decay has already been applied twice.
        let r = Recurrence {
            decay: 0.5,
            gain: 1.0,
        };
        let xs = stream(10);
        let trace = r.trace(&[0.0, 0.0], &xs).unwrap();
        assert_ne!(trace[8], trace[10]);

        // "Undo" by dividing out the decay: the naive inverse, which is what a
        // counter decrement amounts to. It does not land on the earlier state.
        let mut naive = trace[10].clone();
        for h in naive.iter_mut() {
            *h /= 0.5 * 0.5;
        }
        assert_ne!(naive, trace[8]);
    }

    #[test]
    fn the_recurrence_is_order_dependent() {
        let r = Recurrence {
            decay: 0.7,
            gain: 1.0,
        };
        let forward = r.trace(&[0.0], &[vec![1.0], vec![2.0]]).unwrap();
        let reversed = r.trace(&[0.0], &[vec![2.0], vec![1.0]]).unwrap();
        assert_ne!(forward.last(), reversed.last());
    }

    #[test]
    fn a_width_mismatch_is_a_typed_error() {
        let r = Recurrence {
            decay: 1.0,
            gain: 1.0,
        };
        let mut s = vec![0f32; 3];
        assert!(r.step(&mut s, &[1.0, 2.0]).is_err());
        assert!(r.replay(&[0.0], &stream(3), 2, 1).is_err());
        assert!(r.replay(&[0.0, 0.0], &stream(3), 0, 9).is_err());
    }

    #[test]
    fn short_convolution_is_causal_and_zero_padded_at_the_start() {
        let c = ShortConv {
            taps: vec![1.0, 2.0, 3.0],
        };
        let xs = [1.0f32, 10.0, 100.0, 1000.0];
        // t=0 sees only x0.
        assert_eq!(c.at(&xs, 0).unwrap(), 1.0);
        // t=1 sees x1 and x0.
        assert_eq!(c.at(&xs, 1).unwrap(), 10.0 + 2.0);
        // t=2 is the first fully populated window.
        assert_eq!(c.at(&xs, 2).unwrap(), 100.0 + 20.0 + 3.0);
        assert_eq!(c.at(&xs, 3).unwrap(), 1000.0 + 200.0 + 30.0);
        assert!(c.at(&xs, 4).is_err());
        assert!(ShortConv { taps: vec![] }.at(&xs, 0).is_err());
    }

    #[test]
    fn a_decode_step_needs_only_the_window_not_the_whole_history() {
        // The bounded-state property: a short convolution's state is the last
        // `window` inputs, so rolling it back means refilling that window --
        // cheap, but not free, and definitely not a counter decrement.
        let c = ShortConv {
            taps: vec![1.0, 2.0, 3.0],
        };
        let xs: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let h = c.history_for(&xs, 15);
        assert_eq!(h, vec![13.0, 14.0, 15.0]);
        assert_eq!(h.len(), c.window());

        // Recomputing from the window alone matches the full-history answer.
        let from_window = c.at(&h, h.len() - 1).unwrap();
        assert_eq!(from_window, c.at(&xs, 15).unwrap());

        // Near the start the window is shorter, which is the padded case.
        assert_eq!(c.history_for(&xs, 1), vec![0.0, 1.0]);
        assert_eq!(c.history_for(&xs, 0), vec![0.0]);
    }
}
