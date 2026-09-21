//! The online-softmax partial and merge a Flash-style attention kernel runs on.
//!
//! [`crate::mask::attend_row`] is the exact single-row attention: every visible
//! key at once, one maximum, one denominator. A kernel that attends over paged
//! state cannot do that, because it sees one block of keys at a time and must
//! produce a running answer that is correct after every block. It carries a
//! partial state instead -- a running maximum, a running denominator and a
//! running weighted sum of values -- and rescales it whenever a later block
//! raises the maximum. `mask.rs` named this as owed mathematics: "the online-
//! softmax merge for streamed pages ... deserves its own fixture when the
//! streaming path exists". Task 0037 is that path.
//!
//! What this module pins: that the merge is exactly the whole-history softmax
//! however the blocks are cut, that a fully masked block contributes nothing
//! and produces no `NaN`, that block order does not matter, that the maximum
//! subtraction really is what keeps large scores finite, and that a query with
//! no visible key anywhere is a typed failure rather than a uniform draw.
//!
//! What it does **not** pin: the FP32 arithmetic a device kernel actually uses.
//! This module is FP64 throughout, like every other reference here, so it states
//! the *algebra* a kernel may reorder into. The distance between that algebra
//! and an FP32 kernel is bounded by
//! [`crate::attention::attention_error_bound`], which is the only tolerance a
//! candidate is qualified against; ADR 0028 owns its two clauses. Nothing here
//! widens it. Also not pinned here: page tables, physical residency, admission
//! and lease lifetime, which are `moxie-state` and `moxie-memory` contracts, and
//! the transfer diagnostics of host-backed streaming, which is later M4 work.

use moxie_types::{Error, Result};

/// A running softmax over the blocks of keys seen so far.
///
/// The three fields are the whole state a Flash-style kernel keeps per query
/// row: `max` is the largest scaled score seen, `sum` is `Σ exp(s_k − max)` over
/// the visible keys seen, and `weighted` is `Σ exp(s_k − max) · V_k`. The
/// quotient of the last two is the answer, and it is correct after every block
/// rather than only at the end.
///
/// `sum` is zero **exactly** when no visible key has been accumulated: a single
/// visible key contributes `exp(max − max) = 1`, so any non-empty partial has
/// `sum ≥ 1`. That is what makes emptiness testable without inspecting `max`,
/// which is `−∞` for an empty partial and would turn every arithmetic path that
/// touched it into a `NaN`.
#[derive(Debug, Clone, PartialEq)]
pub struct Partial {
    /// The largest scaled score in this partial, or negative infinity when empty.
    pub max: f64,
    /// The denominator after subtracting [`Self::max`].
    pub sum: f64,
    /// The unnormalized weighted value sum after subtracting [`Self::max`].
    ///
    /// Device partial producers use the same representation before the host
    /// calls [`Partial::merge`]. The algebra remains private to the methods;
    /// these fields only make an already-computed partial transportable.
    pub weighted: Vec<f64>,
}

impl Partial {
    /// The identity of [`Partial::merge`]: no key, no weight, no value.
    ///
    /// `value_dim` may be zero, which is what a kernel holds before it has seen
    /// a block wide enough to tell it the value width. Merging such a partial
    /// adopts the other side's width rather than refusing it.
    pub fn empty(value_dim: usize) -> Result<Self> {
        let mut weighted = crate::try_vec(value_dim)?;
        weighted.resize(value_dim, 0.0);
        Ok(Self {
            max: f64::NEG_INFINITY,
            sum: 0.0,
            weighted,
        })
    }

    /// Accumulate one block of keys and values under its own mask.
    ///
    /// `keys`, `values` and `allowed` are the block's rows -- a page, a tile, a
    /// tail, whatever the caller is iterating -- and `allowed[i]` is the
    /// visibility decision already made on **absolute** positions by
    /// `moxie_graph::Visibility`. This function never decides visibility itself:
    /// R21 is the family of bugs where a block-local index is used as a
    /// position, and the way to not have them is to not have the information
    /// here.
    ///
    /// A block none of whose rows are visible returns the empty partial. That is
    /// the ordinary case for a sliding window, not an error: every page below
    /// the window is fully masked and a kernel walks over it.
    pub fn block(
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
        allowed: &[bool],
        scale: f32,
    ) -> Result<Self> {
        if keys.len() != values.len() || keys.len() != allowed.len() {
            return Err(Error::InvalidRequest {
                field: "attention",
                detail: format!(
                    "{} key(s), {} value(s), {} mask entries",
                    keys.len(),
                    values.len(),
                    allowed.len()
                ),
            });
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::InvalidRequest {
                field: "scale",
                detail: format!("score scale must be finite and positive, got {scale}"),
            });
        }
        let mut visible = crate::try_vec(keys.len())?;
        visible.extend((0..keys.len()).filter(|i| allowed[*i]));
        let Some(&first) = visible.first() else {
            // The width is still knowable from a masked row, and a caller that
            // merges this partial gets a dimension check rather than silence.
            let dim = values.first().map_or(0, |v| v.len());
            return Self::empty(dim);
        };
        let value_dim = values[first].len();

        let mut scores = crate::try_vec(visible.len())?;
        for i in &visible {
            if keys[*i].len() != query.len() {
                return Err(Error::InvalidRequest {
                    field: "key",
                    detail: format!("key {i} has dimension {}", keys[*i].len()),
                });
            }
            let dot: f64 = query
                .iter()
                .zip(keys[*i].iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let score = dot * scale as f64;
            // Every score, not just the largest. An infinite or NaN score is a
            // broken input rather than a distribution with one certain outcome,
            // and `NaN` in particular cannot be caught by inspecting the maximum
            // afterwards: `f64::max` *ignores* a `NaN` operand, so the fold
            // below would return an ordinary finite maximum and the `NaN` would
            // reappear in the denominator with nothing to explain it.
            if !score.is_finite() {
                return Err(Error::Numerical {
                    detail: format!("score for key {i} is {score}"),
                });
            }
            scores.push(score);
        }
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut weighted = crate::try_vec(value_dim)?;
        weighted.resize(value_dim, 0.0);
        let mut sum = 0.0f64;
        for (n, i) in visible.iter().enumerate() {
            if values[*i].len() != value_dim {
                return Err(Error::InvalidRequest {
                    field: "value",
                    detail: format!("value {i} has dimension {}", values[*i].len()),
                });
            }
            let w = (scores[n] - max).exp();
            sum += w;
            for (o, v) in weighted.iter_mut().zip(values[*i].iter()) {
                *o += w * *v as f64;
            }
        }
        Ok(Self { max, sum, weighted })
    }

    /// Whether this partial has accumulated no visible key.
    pub fn is_empty(&self) -> bool {
        self.sum == 0.0
    }

    /// The running maximum, `−∞` when empty.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// The running denominator, `Σ exp(s_k − max)`.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Fold another block's partial into this one.
    ///
    /// Both sides are rescaled to the larger maximum before they are added:
    /// with `m = max(mₐ, m_b)`, the combined state is
    /// `sum = sumₐ·e^{mₐ−m} + sum_b·e^{m_b−m}` and the same for each value
    /// component. Both factors are at most one, so neither side can overflow,
    /// and the side that is far below the new maximum underflows to zero --
    /// which is the correct answer for a key whose score is `exp(−700)` times
    /// smaller than another's.
    ///
    /// An empty partial is the identity and is returned around, rather than
    /// through, the arithmetic: its maximum is `−∞`, and `−∞ − (−∞)` is `NaN`.
    /// That is the single most common way to write this function wrong, and
    /// `a_fully_masked_block_contributes_nothing_and_never_a_nan` is the
    /// regression.
    pub fn merge(&self, other: &Self) -> Result<Self> {
        if other.is_empty() {
            return self.clone_fallibly();
        }
        if self.is_empty() {
            return other.clone_fallibly();
        }
        if self.weighted.len() != other.weighted.len() {
            return Err(Error::InvalidRequest {
                field: "value_dim",
                detail: format!(
                    "merging partials of value width {} and {}",
                    self.weighted.len(),
                    other.weighted.len()
                ),
            });
        }
        let max = self.max.max(other.max);
        let left = (self.max - max).exp();
        let right = (other.max - max).exp();
        let mut weighted = crate::try_vec(self.weighted.len())?;
        weighted.extend(
            self.weighted
                .iter()
                .zip(other.weighted.iter())
                .map(|(a, b)| a * left + b * right),
        );
        Ok(Self {
            max,
            sum: self.sum * left + other.sum * right,
            weighted,
        })
    }

    /// The attention output, `Σ p_k · V_k` with `p` summing to one.
    ///
    /// A partial that accumulated nothing is [`Error::Numerical`], for the
    /// reason `attend_row` refuses an empty visible set: there is no
    /// distribution over no outcomes, and a uniform draw over the whole cache is
    /// the wrong answer rather than a graceful one.
    pub fn finish(&self) -> Result<Vec<f64>> {
        if self.is_empty() {
            return Err(Error::Numerical {
                detail: "no visible key for this query position".into(),
            });
        }
        let mut out = crate::try_vec(self.weighted.len())?;
        out.extend(self.weighted.iter().map(|w| w / self.sum));
        Ok(out)
    }

    fn clone_fallibly(&self) -> Result<Self> {
        Ok(Self {
            max: self.max,
            sum: self.sum,
            weighted: crate::try_clone_slice(&self.weighted)?,
        })
    }
}

/// Attend one query row over a history cut into blocks of `block` rows.
///
/// The whole point of the module in one function: the caller supplies the same
/// keys, values and mask [`crate::mask::attend_row`] would get, and gets the
/// same answer computed the way a paged kernel computes it. `block` is the page
/// or tile width; the final block may be short, which is the page tail.
pub fn attend_row_blocked(
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
    allowed: &[bool],
    scale: f32,
    block: usize,
) -> Result<Vec<f64>> {
    if block == 0 {
        return Err(Error::InvalidRequest {
            field: "block",
            detail: "zero-row attention block".into(),
        });
    }
    if keys.len() != values.len() || keys.len() != allowed.len() {
        return Err(Error::InvalidRequest {
            field: "attention",
            detail: format!(
                "{} key(s), {} value(s), {} mask entries",
                keys.len(),
                values.len(),
                allowed.len()
            ),
        });
    }
    let mut state = Partial::empty(0)?;
    for start in (0..keys.len()).step_by(block) {
        let end = (start + block).min(keys.len());
        let part = Partial::block(
            query,
            &keys[start..end],
            &values[start..end],
            &allowed[start..end],
            scale,
        )?;
        state = state.merge(&part)?;
    }
    state.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::{Visibility, attend_row};

    /// Deterministic, mildly ill-conditioned rows: the products cancel enough to
    /// be interesting and stay in the normal range.
    fn rows(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| (((i * 7 + d * 11) % 23) as f32 - 11.0) / 5.0)
                    .collect()
            })
            .collect()
    }

    fn views(v: &[Vec<f32>]) -> Vec<&[f32]> {
        v.iter().map(|r| r.as_slice()).collect()
    }

    /// The exact whole-history reference in FP64, with no blocking at all.
    fn whole(
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
        allowed: &[bool],
        scale: f32,
    ) -> Vec<f64> {
        let visible: Vec<usize> = (0..keys.len()).filter(|i| allowed[*i]).collect();
        let scores: Vec<f64> = visible
            .iter()
            .map(|i| {
                let dot: f64 = query
                    .iter()
                    .zip(keys[*i].iter())
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum();
                dot * scale as f64
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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

    fn assert_close(got: &[f64], want: &[f64], tolerance: f64, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: width");
        for (d, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let scale = w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tolerance * scale,
                "{label} component {d}: {g:.17e} vs {w:.17e}"
            );
        }
    }

    #[test]
    fn every_block_width_reproduces_the_whole_history_softmax() {
        // Page widths that divide the history, widths that leave a tail, a width
        // of one row and a width larger than the whole history. A kernel meets
        // all four: a full page, a partial final page, a single decode row, and
        // a history shorter than one page.
        let dim = 8usize;
        let n = 37usize;
        let keys = rows(n, dim);
        let values: Vec<Vec<f32>> = rows(n, dim)
            .iter()
            .map(|r| r.iter().map(|v| v * 0.75 - 0.25).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 - 4.0) / 3.0).collect();
        let k = views(&keys);
        let v = views(&values);

        for visibility in [Visibility::Causal, Visibility::SlidingWindow { window: 12 }] {
            let position = (n - 1) as u64;
            let allowed: Vec<bool> = (0..n)
                .map(|i| visibility.allows(position, i as u64))
                .collect();
            let want = whole(&query, &k, &v, &allowed, 1.0);
            for block in [1usize, 2, 8, 16, 32, 37, 64] {
                let got = attend_row_blocked(&query, &k, &v, &allowed, 1.0, block).unwrap();
                assert_close(&got, &want, 1e-14, &format!("{visibility:?} block {block}"));
            }
        }
    }

    #[test]
    fn the_blocked_result_is_the_exact_row_attention_to_fp32_resolution() {
        // Against the *other* oracle, not a second copy of this one. `attend_row`
        // is FP32 and this is FP64, so they agree to FP32's resolution rather
        // than bitwise -- which is the whole reason a kernel is qualified
        // against `attention_error_bound` and not against equality.
        let dim = 6usize;
        let n = 20usize;
        let keys = rows(n, dim);
        let values = rows(n, dim);
        let query: Vec<f32> = (0..dim).map(|d| (d % 4) as f32 * 0.5 - 0.75).collect();
        let allowed: Vec<bool> = (0..n)
            .map(|i| Visibility::Causal.allows((n - 1) as u64, i as u64))
            .collect();
        let scale = moxie_graph::reciprocal_sqrt_scale(dim as u64);

        let exact = attend_row(&query, &keys, &values, &allowed, scale).unwrap();
        let blocked =
            attend_row_blocked(&query, &views(&keys), &views(&values), &allowed, scale, 7).unwrap();
        for (d, (a, b)) in exact.iter().zip(blocked.iter()).enumerate() {
            assert!(
                (*a as f64 - b).abs() <= 1e-6 * b.abs().max(1.0),
                "component {d}: fp32 row {a} vs fp64 blocked {b}"
            );
        }
    }

    #[test]
    fn a_fully_masked_block_contributes_nothing_and_never_a_nan() {
        // The sliding-window case, and the merge's sharpest edge: the first
        // blocks are entirely below the window, so their partial is empty and
        // its maximum is -inf. Rescaling by exp(-inf - m) is zero, but
        // exp(-inf - -inf) is NaN, so an implementation that merges empty
        // partials through the arithmetic instead of around it poisons the
        // whole row the moment the window has moved past a page.
        let dim = 4usize;
        let n = 24usize;
        let keys = rows(n, dim);
        let values = rows(n, dim);
        let query = vec![0.5f32, -0.25, 1.0, 0.125];
        let window = Visibility::SlidingWindow { window: 5 };
        let position = (n - 1) as u64;
        let allowed: Vec<bool> = (0..n).map(|i| window.allows(position, i as u64)).collect();
        assert!(
            !allowed[..8].iter().any(|a| *a),
            "the fixture needs whole masked pages at the front"
        );

        let got =
            attend_row_blocked(&query, &views(&keys), &views(&values), &allowed, 1.0, 8).unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "got {got:?}");
        assert_close(
            &got,
            &whole(&query, &views(&keys), &views(&values), &allowed, 1.0),
            1e-14,
            "sliding",
        );

        // And the empty partial is genuinely empty, both ways round.
        let empty = Partial::empty(dim).unwrap();
        let block = Partial::block(
            &query,
            &views(&keys)[..8],
            &views(&values)[..8],
            &allowed[..8],
            1.0,
        )
        .unwrap();
        assert!(
            block.is_empty(),
            "a fully masked block accumulated something"
        );
        assert_eq!(block.max(), f64::NEG_INFINITY);
        let full = Partial::block(
            &query,
            &views(&keys)[19..],
            &views(&values)[19..],
            &allowed[19..],
            1.0,
        )
        .unwrap();
        assert!(!full.is_empty() && full.sum() >= 1.0, "sum {}", full.sum());
        assert_eq!(empty.merge(&full).unwrap(), full);
        assert_eq!(full.merge(&empty).unwrap(), full);
        assert!(empty.merge(&empty).unwrap().is_empty());
    }

    #[test]
    fn block_order_does_not_change_the_answer() {
        // A kernel may walk pages in whatever order its page table hands them
        // over, and a later slice may merge partials from several workers. The
        // merge has to be order-independent to FP64 resolution for that to be
        // legal, so it is asserted rather than assumed.
        let dim = 5usize;
        let n = 18usize;
        let keys = rows(n, dim);
        let values = rows(n, dim);
        let query: Vec<f32> = (0..dim).map(|d| 0.3 * d as f32 - 0.6).collect();
        let allowed = vec![true; n];
        let k = views(&keys);
        let v = views(&values);

        let parts: Vec<Partial> = (0..n)
            .step_by(4)
            .map(|s| {
                let e = (s + 4).min(n);
                Partial::block(&query, &k[s..e], &v[s..e], &allowed[s..e], 0.5).unwrap()
            })
            .collect();

        let mut forward = Partial::empty(dim).unwrap();
        for p in &parts {
            forward = forward.merge(p).unwrap();
        }
        let mut backward = Partial::empty(dim).unwrap();
        for p in parts.iter().rev() {
            backward = backward.merge(p).unwrap();
        }
        // Pairwise, the way a tree reduction would.
        let mut tree: Vec<Partial> = parts;
        while tree.len() > 1 {
            let mut next = Vec::new();
            for pair in tree.chunks(2) {
                next.push(match pair {
                    [a, b] => a.merge(b).unwrap(),
                    [a] => a.clone(),
                    _ => unreachable!(),
                });
            }
            tree = next;
        }

        let want = whole(&query, &k, &v, &allowed, 0.5);
        assert_close(&forward.finish().unwrap(), &want, 1e-14, "forward");
        assert_close(&backward.finish().unwrap(), &want, 1e-14, "backward");
        assert_close(&tree[0].finish().unwrap(), &want, 1e-14, "tree");
    }

    #[test]
    fn the_running_maximum_is_what_keeps_large_scores_finite() {
        // Scores near 750 overflow `exp` in FP64 outright. The point of the
        // running maximum is that they never reach it; the point of *rescaling*
        // is that a late block raising the maximum does not overflow either.
        // Without the subtraction this test produces inf/inf = NaN, which is
        // exactly how a kernel that "works on small fixtures" fails at real
        // score magnitudes.
        let keys: Vec<Vec<f32>> = vec![
            vec![800.0, 0.0],
            vec![1200.0, 0.0], // a later, larger maximum
            vec![100.0, 0.0],  // and something far below it
        ];
        let values: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let query = vec![1.0f32, 1.0];
        let allowed = vec![true; 3];
        assert!(
            (800.0f64).exp().is_infinite(),
            "the fixture must really overflow"
        );

        let got =
            attend_row_blocked(&query, &views(&keys), &views(&values), &allowed, 1.0, 1).unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "got {got:?}");
        // 1200 dominates 800 by e^400: the answer is the second value.
        assert_close(&got, &[0.0, 1.0], 1e-12, "dominated");

        // One block rather than three: same answer, so the rescaling path and
        // the single-maximum path agree where both are defined.
        let one =
            attend_row_blocked(&query, &views(&keys), &views(&values), &allowed, 1.0, 3).unwrap();
        assert_close(&one, &got, 1e-15, "one block");
    }

    #[test]
    fn a_query_with_no_visible_key_anywhere_is_a_typed_failure() {
        // Document 05's rule, and the same refusal `attend_row` makes: no
        // visible key is not a uniform distribution over the cache.
        let dim = 3usize;
        let keys = rows(4, dim);
        let values = rows(4, dim);
        let query = vec![1.0f32; dim];
        let allowed = vec![false; 4];
        let err = attend_row_blocked(&query, &views(&keys), &views(&values), &allowed, 1.0, 2)
            .unwrap_err();
        assert!(matches!(err, Error::Numerical { .. }), "{err:?}");
        assert!(Partial::empty(dim).unwrap().finish().is_err());
    }

    #[test]
    fn malformed_blocks_and_widths_are_typed_errors() {
        let dim = 3usize;
        let keys = rows(4, dim);
        let values = rows(4, dim);
        let k = views(&keys);
        let v = views(&values);
        let query = vec![1.0f32; dim];
        let allowed = vec![true; 4];

        assert!(matches!(
            attend_row_blocked(&query, &k, &v, &allowed, 1.0, 0).unwrap_err(),
            Error::InvalidRequest { field: "block", .. }
        ));
        assert!(matches!(
            attend_row_blocked(&query, &k[..3], &v, &allowed, 1.0, 2).unwrap_err(),
            Error::InvalidRequest {
                field: "attention",
                ..
            }
        ));
        // The graph and kernel contract require a strictly positive scale, not
        // merely a nonzero one: zero, a negative value, and non-finite values
        // are all refused.
        for scale in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
            assert!(
                matches!(
                    Partial::block(&query, &k, &v, &allowed, scale).unwrap_err(),
                    Error::InvalidRequest { field: "scale", .. }
                ),
                "scale {scale} was accepted"
            );
        }
        // A key of the wrong width is caught rather than silently truncated.
        let short = vec![1.0f32, 2.0];
        let mixed: Vec<&[f32]> = vec![short.as_slice(), k[1], k[2], k[3]];
        assert!(matches!(
            Partial::block(&query, &mixed, &v, &allowed, 1.0).unwrap_err(),
            Error::InvalidRequest { field: "key", .. }
        ));
        // Two partials of different value width cannot be merged.
        let wide = rows(2, 5);
        let wide_values = views(&wide);
        let narrow = Partial::block(&query, &k[..2], &v[..2], &allowed[..2], 1.0).unwrap();
        let other = Partial::block(&query, &k[..2], &wide_values, &allowed[..2], 1.0).unwrap();
        assert!(matches!(
            narrow.merge(&other).unwrap_err(),
            Error::InvalidRequest {
                field: "value_dim",
                ..
            }
        ));
    }

    #[test]
    fn a_nonfinite_score_is_refused_rather_than_propagated() {
        // An infinite score is not "one certain key"; it is a broken input, and
        // the softmax of it is NaN in every component that follows.
        let keys: Vec<Vec<f32>> = vec![vec![f32::INFINITY, 0.0], vec![1.0, 1.0]];
        let values: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let query = vec![1.0f32, 1.0];
        let allowed = vec![true; 2];
        let err =
            Partial::block(&query, &views(&keys), &views(&values), &allowed, 1.0).unwrap_err();
        assert!(matches!(err, Error::Numerical { .. }), "{err:?}");

        // The NaN half, and the reason every score is checked rather than the
        // maximum: `f64::max` ignores a NaN operand, so a fold over these scores
        // reports the finite one and the NaN survives into the denominator.
        let finite = vec![vec![1.0f32, 1.0], vec![2.0, 2.0]];
        let query = vec![f32::NAN, 1.0];
        assert_eq!(
            f64::NEG_INFINITY.max(f64::NAN).max(4.0),
            4.0,
            "this is the trap being guarded"
        );
        let err =
            Partial::block(&query, &views(&finite), &views(&values), &allowed, 1.0).unwrap_err();
        assert!(matches!(err, Error::Numerical { .. }), "{err:?}");

        // A nonfinite score behind a masked row is not an input to anything, so
        // it is not an error either.
        let masked = Partial::block(
            &[1.0f32, 1.0],
            &views(&keys),
            &views(&values),
            &[false, true],
            1.0,
        )
        .unwrap();
        assert!(!masked.is_empty() && masked.max().is_finite());
    }
}
