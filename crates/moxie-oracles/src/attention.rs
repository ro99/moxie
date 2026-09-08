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

    /// One head's lanes, checked rather than sliced blind.
    ///
    /// The fourth review appended a one-element key and attended with head
    /// dimension two: the slice ran off the end and panicked. A stored row that
    /// does not match the attention geometry is an artifact error, and a panic
    /// is not one -- document 02 requires typed errors at this boundary.
    fn head_slice(row: &[f32], head: usize, head_dim: usize, heads: usize) -> Result<&[f32]> {
        if row.len() != heads * head_dim {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "a stored row of {} element(s) does not match {heads} head(s) of \
                     dimension {head_dim}",
                    row.len()
                ),
            });
        }
        Ok(&row[head * head_dim..(head + 1) * head_dim])
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
        let q = KvHistory::head_slice(query, h, head_dim, heads)?;
        let keys: Vec<Vec<f32>> = history
            .keys
            .iter()
            .map(|k| KvHistory::head_slice(k, h, head_dim, heads).map(<[f32]>::to_vec))
            .collect::<Result<_>>()?;
        let values: Vec<Vec<f32>> = history
            .values
            .iter()
            .map(|v| KvHistory::head_slice(v, h, head_dim, heads).map(<[f32]>::to_vec))
            .collect::<Result<_>>()?;
        out.extend(attend_row(q, &keys, &values, &allowed, scale)?);
    }
    Ok(out)
}

/// The error bound for one attention output, per task 0003's **revised**
/// analysis.
///
/// The first version of this contract counted rounding steps and multiplied by
/// `Σ|p·V|`, which is wrong, and the fourth review disproved it: with keys
/// containing `2^24` and `1` the score dot product cancels catastrophically, two
/// scores that differ by 0.5 in FP64 both evaluate to 0 in FP32, the softmax
/// returns a uniform distribution instead of `[0.62, 0.38]`, and the normalized
/// error is 0.32 against a declared bound of 6.6e-7.
///
/// Counting operations cannot bound this, because the error does not pass
/// through the softmax additively -- it passes through an **exponential**. The
/// sound statement has two terms:
///
/// ```text
/// Δs   = γ(Hd) · max_k ( Σ_i |Q_i · K_ki| ) · scale        // score error
/// |ô_i − o_i| ≤ (e^{2·Δs} − 1) · max_k |V_ki|              // via the softmax
///             + γ(K + 2) · Σ_k |p_k · V_ki|                // the weighted sum
/// ```
///
/// The first term is the perturbation of the softmax weights: if every score
/// carries absolute error at most `Δs`, then every ratio `p̂_j/p_j` lies in
/// `[e^{−2Δs}, e^{2Δs}]`, so `‖p̂ − p‖₁ ≤ e^{2Δs} − 1`. The second is the ordinary
/// sequential-sum bound over the `K` visible keys plus the max subtraction and
/// the divide.
///
/// **This bound is data-dependent and it does not shrink to a constant.** When
/// `Q·K` is well conditioned, `Δs` is tiny and the bound is a few ulps; when it
/// cancels, `Δs` is large relative to the scores and the bound is correspondingly
/// weak -- which is not a defect in the bound but a true statement about FP32
/// attention. A kernel qualified against this must be qualified on data whose
/// conditioning is stated, and document 07's requirement to "stress cancellation
/// and near-zero outputs" is exactly the case where this term dominates.
pub fn attention_error_bound(
    query_head: &[f32],
    visible_keys: &[&[f32]],
    visible_values: &[&[f32]],
    weights: &[f64],
    component: usize,
) -> f64 {
    use crate::metric::gamma;
    let head_dim = query_head.len();
    let scale = 1.0 / (head_dim as f64).sqrt();

    let delta_s = visible_keys
        .iter()
        .map(|k| {
            let abs_sum: f64 = query_head
                .iter()
                .zip(k.iter())
                .map(|(q, kk)| (*q as f64 * *kk as f64).abs())
                .sum();
            gamma(head_dim as u64) * abs_sum * scale
        })
        .fold(0f64, f64::max);

    let max_v = visible_values
        .iter()
        .map(|v| (v[component] as f64).abs())
        .fold(0f64, f64::max);
    let weighted: f64 = weights
        .iter()
        .zip(visible_values.iter())
        .map(|(p, v)| (p * v[component] as f64).abs())
        .sum();

    let softmax_term = (2.0 * delta_s).exp_m1() * max_v;
    softmax_term + gamma(visible_keys.len() as u64 + 2) * weighted
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

    /// The FP64 reference plus the pieces the bound needs: the exact softmax
    /// weights and the visible key/value slices.
    fn reference(
        q: &[f32],
        history: &KvHistory,
        pos: u64,
        vis: Visibility,
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let visible: Vec<usize> = (0..history.len())
            .filter(|k| vis.allows(pos, *k as u64))
            .collect();
        let scale = 1.0 / (q.len() as f64).sqrt();
        let scores: Vec<f64> = visible
            .iter()
            .map(|i| {
                let dot: f64 = q
                    .iter()
                    .zip(history.keys[*i].iter())
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                dot * scale
            })
            .collect();
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f64 = exps.iter().sum();
        let weights: Vec<f64> = exps.iter().map(|e| e / denom).collect();
        let dim = q.len();
        let out: Vec<f64> = (0..dim)
            .map(|d| {
                visible
                    .iter()
                    .enumerate()
                    .map(|(n, i)| weights[n] * history.values[*i][d] as f64)
                    .sum()
            })
            .collect();
        (out, weights, visible)
    }

    /// Check one attention result against the revised, data-dependent bound.
    fn assert_within_bound(
        q: &[f32],
        history: &KvHistory,
        pos: u64,
        vis: Visibility,
        label: &str,
    ) -> f64 {
        let got = attend_multi_head(q, history, pos, 1, q.len(), vis).unwrap();
        let (want, weights, visible) = reference(q, history, pos, vis);
        let keys: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.keys[*i].as_slice())
            .collect();
        let values: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.values[*i].as_slice())
            .collect();

        let mut worst = 0f64;
        for d in 0..q.len() {
            let bound = attention_error_bound(q, &keys, &values, &weights, d);
            let err = (got[d] as f64 - want[d]).abs();
            assert!(
                err <= bound,
                "{label} component {d}: error {err:.4e} exceeded bound {bound:.4e} \
                 (got {}, want {want:?})",
                got[d]
            );
            worst = worst.max(err);
        }
        worst
    }

    #[test]
    fn attention_matches_the_equation_within_its_revised_bound() {
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

        for pos in [0u64, 1, 7, (n - 1) as u64] {
            assert_within_bound(&q, &history, pos, Visibility::Causal, &format!("pos {pos}"));
        }

        // On well-conditioned data the bound really is tight -- a few ulps, not
        // a licence. If this ever loosens, the bound has stopped saying anything.
        let (want, weights, visible) = reference(&q, &history, (n - 1) as u64, Visibility::Causal);
        let keys: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.keys[*i].as_slice())
            .collect();
        let values: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.values[*i].as_slice())
            .collect();
        let bound = attention_error_bound(&q, &keys, &values, &weights, 0);
        assert!(
            bound < 1e-4,
            "on benign data the bound should be tiny, got {bound:.3e}"
        );
        let got = attend_multi_head(
            &q,
            &history,
            (n - 1) as u64,
            1,
            head_dim,
            Visibility::Causal,
        )
        .unwrap();
        let s = ErrorSummary::absolute(&got, &want);
        println!("attention[{head_dim}x{n}] {s} bound={bound:.3e}");
    }

    #[test]
    fn a_cancelling_score_widens_the_bound_because_the_error_is_real() {
        // The fourth review's counterexample, preserved. Keys holding 2^24 and 1
        // make the score dot product cancel catastrophically: two scores that
        // differ by 0.5 in FP64 both evaluate to exactly 0 in FP32, so the
        // softmax returns a uniform distribution instead of [0.62, 0.38].
        //
        // The old contract counted rounding steps and declared gamma(2K+Hd+3),
        // about 6.6e-7. The actual normalized error is ~0.32. That is not a
        // tolerance that needed enlarging; it is a bound of the wrong *form*,
        // because the score error passes through an exponential rather than
        // being added to the result.
        let hd = 4usize;
        let big = 16_777_216.0f32; // 2^24; f32 cannot represent 2^24 + 1
        let mut history = KvHistory::new();
        history
            .append(0, vec![big, 1.0, -big, 0.0], vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        history
            .append(1, vec![big, 0.0, -big, 0.0], vec![0.0, 1.0, 0.0, 0.0])
            .unwrap();
        let q = vec![1.0f32; hd];

        let got = attend_multi_head(&q, &history, 1, 1, hd, Visibility::Causal).unwrap();
        let (want, weights, visible) = reference(&q, &history, 1, Visibility::Causal);

        // The disagreement is large and real: FP32 genuinely computes a
        // different distribution here.
        assert!((got[0] - 0.5).abs() < 1e-6, "FP32 saw a tie: {got:?}");
        assert!((want[0] - 0.6224593).abs() < 1e-6, "FP64 did not: {want:?}");
        let old_bound = gamma(2 * 2 + hd as u64 + 3);
        assert!(
            (got[0] as f64 - want[0]).abs() > 1000.0 * old_bound,
            "the old bound was not merely tight, it was wrong by orders of magnitude"
        );

        // The revised bound covers it, because the score-error term grows with
        // the magnitude of the terms being cancelled.
        let keys: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.keys[*i].as_slice())
            .collect();
        let values: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.values[*i].as_slice())
            .collect();
        for d in 0..2 {
            let bound = attention_error_bound(&q, &keys, &values, &weights, d);
            let err = (got[d] as f64 - want[d]).abs();
            assert!(err <= bound, "component {d}: {err:.4e} > {bound:.4e}");
        }

        // And it is honest about being weak here rather than pretending.
        let bound = attention_error_bound(&q, &keys, &values, &weights, 0);
        assert!(
            bound > 0.1,
            "on data this ill-conditioned the bound must say so, got {bound:.3e}"
        );
    }

    #[test]
    fn the_bound_is_tight_when_the_scores_are_well_conditioned() {
        // The other half: the score-error term must vanish for ordinary data, or
        // the bound would be useless as a kernel acceptance gate.
        let hd = 8usize;
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|i| (0..hd).map(|d| ((i + d) % 5) as f32 * 0.25 - 0.5).collect())
            .collect();
        let history = history_of(&rows);
        let q: Vec<f32> = (0..hd).map(|d| (d % 3) as f32 * 0.5 - 0.5).collect();
        let (_, weights, visible) = reference(&q, &history, 5, Visibility::Causal);
        let keys: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.keys[*i].as_slice())
            .collect();
        let values: Vec<&[f32]> = visible
            .iter()
            .map(|i| history.values[*i].as_slice())
            .collect();
        for d in 0..hd {
            let bound = attention_error_bound(&q, &keys, &values, &weights, d);
            assert!(bound < 1e-5, "component {d} bound {bound:.3e} is not tight");
        }
        assert_within_bound(&q, &history, 5, Visibility::Causal, "benign");
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
    fn a_stored_row_that_does_not_match_the_head_geometry_is_a_typed_error() {
        // Fourth review, reproduced: a one-element key attended with head
        // dimension two ran the slice off the end and panicked. `KvHistory`
        // checks key and value widths agree with each other, which is not the
        // same as agreeing with the geometry they are read under.
        let mut h = KvHistory::new();
        h.append(0, vec![1.0], vec![1.0]).unwrap();
        let e = attend_multi_head(&[1.0, 2.0], &h, 0, 1, 2, Visibility::Causal).unwrap_err();
        assert_eq!(e.kind(), "invalid_artifact");
        assert!(e.to_string().contains("does not match"), "{e}");

        // The query itself is checked first, and a multi-head geometry over a
        // single-head row is refused too.
        let mut h = KvHistory::new();
        h.append(0, vec![1.0, 2.0], vec![3.0, 4.0]).unwrap();
        assert!(attend_multi_head(&[1.0, 2.0], &h, 0, 2, 2, Visibility::Causal).is_err());
        assert!(attend_multi_head(&[1.0, 2.0], &h, 0, 1, 2, Visibility::Causal).is_ok());
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
