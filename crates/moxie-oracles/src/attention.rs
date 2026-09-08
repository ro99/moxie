//! Multi-head causal attention over an absolute-position KV history.
//!
//! The single-head, single-row core is [`crate::mask::attend_row`], which the
//! mask fixtures already pin: masked positions contribute **exactly zero**, an
//! empty visible set is a typed failure rather than a uniform draw, and the
//! weights sum to one over the visible set. This module is the head and history
//! bookkeeping around it.
//!
//! Task 0003's slice is deliberately narrow: full causal, one head group (MHA),
//! exact. GQA head mapping, sliding windows in the graph, sinks, biases, MLA and
//! model-defined sparse selection are document 04's later work, and none of them
//! is approximated here.

use moxie_types::{Error, Result};

use crate::mask::{Visibility, attend_row};

/// One layer's key/value history, indexed by **absolute** sequence position.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct KvHistory {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Append the key and value for the next position.
    ///
    /// Positions are dense and appended in order; a gap would mean a row was
    /// executed without its state being written, which is the corruption the
    /// state crate's counters exist to make visible.
    pub fn append(&mut self, position: u64, key: Vec<f32>, value: Vec<f32>) -> Result<()> {
        if position != self.keys.len() as u64 {
            return Err(Error::InvalidRequest {
                field: "position",
                detail: format!(
                    "appending position {position} to a history of {} entries leaves a gap",
                    self.keys.len()
                ),
            });
        }
        if key.len() != value.len() {
            return Err(Error::InvalidArtifact {
                detail: format!("key has {} elements, value {}", key.len(), value.len()),
            });
        }
        self.keys.push(key);
        self.values.push(value);
        Ok(())
    }

    /// Drop everything at or after `prefix`.
    ///
    /// `StateKind::KvPages` is `RestoreCapability::Truncate`, and this is what
    /// that means physically: the tail is discarded, and what remains is exactly
    /// what was there before those positions were written.
    pub fn truncate(&mut self, prefix: u64) {
        self.keys.truncate(prefix as usize);
        self.values.truncate(prefix as usize);
    }

    fn head_slice(row: &[f32], head: usize, head_dim: usize) -> &[f32] {
        &row[head * head_dim..(head + 1) * head_dim]
    }
}

/// Attend one query row against the history, for every head.
///
/// `query` is `heads · head_dim` long. `position` is the query's absolute
/// position, and the history must already contain it -- the caller appends this
/// row's key and value before attending, because a causal query attends to
/// itself.
///
/// Scores are scaled by `1/sqrt(head_dim)`, the softmax runs in FP32 with the
/// maximum subtracted, and masked keys are removed from the sum rather than
/// biased by a large negative number.
pub fn attend_multi_head(
    query: &[f32],
    history: &KvHistory,
    position: u64,
    heads: usize,
    head_dim: usize,
    visibility: Visibility,
) -> Result<Vec<f32>> {
    if query.len() != heads * head_dim {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "query has {} elements, expected {heads}x{head_dim}",
                query.len()
            ),
        });
    }
    if head_dim == 0 {
        return Err(Error::InvalidRequest {
            field: "head_dim",
            detail: "zero head dimension".into(),
        });
    }
    if (position as usize) >= history.len() {
        return Err(Error::InvalidRequest {
            field: "position",
            detail: format!(
                "position {position} has no key in a history of {}; append before attending",
                history.len()
            ),
        });
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let allowed: Vec<bool> = (0..history.len())
        .map(|k| visibility.allows(position, k as u64))
        .collect();

    let mut out = Vec::with_capacity(query.len());
    for h in 0..heads {
        let q = KvHistory::head_slice(query, h, head_dim);
        let keys: Vec<Vec<f32>> = history
            .keys
            .iter()
            .map(|k| KvHistory::head_slice(k, h, head_dim).to_vec())
            .collect();
        let values: Vec<Vec<f32>> = history
            .values
            .iter()
            .map(|v| KvHistory::head_slice(v, h, head_dim).to_vec())
            .collect();
        out.extend(attend_row(q, &keys, &values, &allowed, scale)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    fn history_of(rows: &[Vec<f32>]) -> KvHistory {
        let mut h = KvHistory::new();
        for (i, r) in rows.iter().enumerate() {
            h.append(i as u64, r.clone(), r.iter().map(|v| v * 0.5).collect())
                .unwrap();
        }
        h
    }

    /// The equation, transcribed separately in FP64 for one head.
    fn attend_head_f64(
        q: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        allowed: &[bool],
        scale: f32,
    ) -> Vec<f64> {
        let visible: Vec<usize> = (0..keys.len()).filter(|i| allowed[*i]).collect();
        let scores: Vec<f64> = visible
            .iter()
            .map(|i| {
                let dot: f64 = q
                    .iter()
                    .zip(&keys[*i])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                dot * scale as f64
            })
            .collect();
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f64 = exps.iter().sum();
        let dim = values[visible[0]].len();
        (0..dim)
            .map(|d| {
                visible
                    .iter()
                    .enumerate()
                    .map(|(n, i)| exps[n] / denom * values[*i][d] as f64)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn attention_matches_the_equation_within_its_declared_bound() {
        let head_dim = 16usize;
        let n = 24usize;
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..head_dim)
                    .map(|d| (((i * 13 + d * 7) % 29) as f32 - 14.0) / 9.0)
                    .collect()
            })
            .collect();
        let history = history_of(&rows);
        let q: Vec<f32> = (0..head_dim).map(|d| (d as f32 - 8.0) / 6.0).collect();
        let pos = (n - 1) as u64;

        let got = attend_multi_head(&q, &history, pos, 1, head_dim, Visibility::Causal).unwrap();

        let allowed: Vec<bool> = (0..n)
            .map(|k| Visibility::Causal.allows(pos, k as u64))
            .collect();
        let keys: Vec<Vec<f32>> = history.keys.clone();
        let values: Vec<Vec<f32>> = history.values.clone();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let want = attend_head_f64(&q, &keys, &values, &allowed, scale);

        // gamma(2K + Hd + 3): Hd per dot product, K exponentials, K accumulation
        // steps, the max subtraction, the divide and the weighted sum.
        let k = n as u64;
        let bound = gamma(2 * k + head_dim as u64 + 3);
        let scale_v: Vec<f64> = (0..head_dim)
            .map(|d| values.iter().map(|v| (v[d] as f64).abs()).sum())
            .collect();
        let s = ErrorSummary::normalized(&got, &want, &scale_v);
        assert_eq!(s.count, head_dim);
        assert!(s.within(bound), "{s} exceeded {bound:.3e}");
    }

    #[test]
    fn a_causal_query_attends_to_itself_and_not_to_the_future() {
        // Two positions with very different values: the query at position 0 must
        // see only position 0, whatever comes later.
        let rows = vec![vec![1.0f32, 0.0], vec![0.0f32, 1000.0]];
        let history = history_of(&rows);
        let q = vec![1.0f32, 0.0];

        let at_zero = attend_multi_head(&q, &history, 0, 1, 2, Visibility::Causal).unwrap();
        // Position 0's value is its key halved: [0.5, 0.0].
        assert!((at_zero[0] - 0.5).abs() < 1e-6, "{at_zero:?}");
        assert!(at_zero[1].abs() < 1e-6, "the future did not leak in");

        let at_one = attend_multi_head(&q, &history, 1, 1, 2, Visibility::Causal).unwrap();
        assert!(at_one[1] > 0.0, "position 1 does see position 1");
    }

    #[test]
    fn heads_are_independent() {
        // Head 1 is fed values that head 0 must not see. If the head slicing
        // were wrong, this is where it would show.
        let head_dim = 2usize;
        let rows = vec![vec![1.0f32, 0.0, 0.0, 1.0]];
        let history = history_of(&rows);
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        let out = attend_multi_head(&q, &history, 0, 2, head_dim, Visibility::Causal).unwrap();
        assert_eq!(out.len(), 4);
        // With one visible position the output is that position's value.
        assert_eq!(out, vec![0.5, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn a_sliding_window_drops_history_the_window_excludes() {
        let rows: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, 1.0]).collect();
        let history = history_of(&rows);
        let q = vec![0.0f32, 1.0];
        let full = attend_multi_head(&q, &history, 4, 1, 2, Visibility::Causal).unwrap();
        let windowed = attend_multi_head(
            &q,
            &history,
            4,
            1,
            2,
            Visibility::SlidingWindow { window: 2 },
        )
        .unwrap();
        assert_ne!(full, windowed);
    }

    #[test]
    fn appending_out_of_order_or_ragged_is_refused() {
        let mut h = KvHistory::new();
        assert!(h.append(1, vec![1.0], vec![1.0]).is_err(), "gap");
        h.append(0, vec![1.0], vec![1.0]).unwrap();
        assert!(h.append(0, vec![1.0], vec![1.0]).is_err(), "duplicate");
        assert!(h.append(1, vec![1.0], vec![1.0, 2.0]).is_err(), "ragged");
    }

    #[test]
    fn truncation_restores_the_earlier_history_exactly() {
        // KvPages is RestoreCapability::Truncate, and this is what that means:
        // the tail is dropped and what remains is bit-identical to before.
        let rows: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, -(i as f32)]).collect();
        let short = history_of(&rows[..4]);
        let mut long = history_of(&rows);
        long.truncate(4);
        assert_eq!(long, short);
    }

    #[test]
    fn attending_to_a_position_with_no_key_is_refused() {
        let history = history_of(&[vec![1.0f32, 2.0]]);
        let q = vec![1.0f32, 2.0];
        assert!(attend_multi_head(&q, &history, 1, 1, 2, Visibility::Causal).is_err());
        assert!(attend_multi_head(&q, &KvHistory::new(), 0, 1, 2, Visibility::Causal).is_err());
        assert!(attend_multi_head(&[1.0], &history, 0, 1, 2, Visibility::Causal).is_err());
    }
}
