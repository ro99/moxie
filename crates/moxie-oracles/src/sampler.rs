//! Sampling distributions.
//!
//! Document 05 pins one ordered pipeline. This module implements the stages M0
//! needs a fixture for, in that order, and returns an explicit normalized
//! distribution rather than a token: "Provide explicit normalized
//! `SamplingDistribution` access independently of drawing. Exact speculative
//! verification consumes this distribution."
//!
//! Pinned here:
//!
//! * the hard legality mask, and the rule that a banned token stays banned;
//! * the shared pre-truncation log normalizer, and top-k / top-p / min-p
//!   computed against **that** distribution rather than an intermediate one;
//! * temperature last, with zero meaning argmax;
//! * the deterministic tie rule: lowest token id;
//! * typed failures for an empty vocabulary, NaN logits, `+inf`, and an
//!   all-banned candidate set.
//!
//! Explicitly **not** implemented here, and not to be read as supported:
//! presence/frequency/repetition penalties, DRY, no-repeat n-gram, logit bias,
//! typical-p, XTC and future entropy. Document 05 specifies all of them and M8
//! integrates them; each needs its own fixtures, and listing a stage is not
//! implementing it. `Pipeline` names the missing stages in a comment so that a
//! reader cannot mistake this subset for the whole contract.

use moxie_types::{Error, Result};

/// A normalized distribution over the vocabulary.
///
/// `probs` sums to one over the surviving candidates and is exactly zero for
/// every filtered or banned token -- not "very small", so a verifier computing
/// `max(p - q, 0)` gets exact zeros where it should.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingDistribution {
    pub probs: Vec<f32>,
}

impl SamplingDistribution {
    /// The token a greedy draw would select: highest probability, lowest id on a
    /// tie. Document 05: "Deterministic score ties choose the smallest token ID
    /// in the new contract; do not depend on an unstable sort."
    pub fn argmax(&self) -> u32 {
        let mut best = 0usize;
        for i in 1..self.probs.len() {
            if self.probs[i] > self.probs[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Tokens with non-zero probability, ascending.
    pub fn support(&self) -> Vec<u32> {
        self.probs
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// The stages this module implements, in document 05's order.
#[derive(Debug, Clone, Default)]
pub struct Pipeline {
    /// Tokens banned outright: EOS before min-length, grammar-illegal tokens,
    /// explicit bans. Applied before any truncation stage, and a banned token
    /// "remains banned throughout later stages".
    pub banned: Vec<u32>,
    /// Keep at most this many candidates. `None` disables the stage.
    pub top_k: Option<usize>,
    /// Smallest descending prefix whose mass reaches this. `None` disables it.
    pub top_p: Option<f32>,
    /// Drop candidates below `min_p * max_p`. `None` disables it.
    pub min_p: Option<f32>,
    /// Applied last. Zero selects the highest final score.
    pub temperature: f32,
    // Not implemented, and deliberately absent rather than present-and-ignored:
    //   presence / frequency / repetition penalties and their windows
    //   DRY, no-repeat n-gram, logit bias
    //   typical-p, XTC
    //   future entropy
    // Document 05 defines each; adding a field here without the mathematics
    // would be exactly the "silently drop a sampler" failure AGENTS.md forbids.
}

impl Pipeline {
    /// Greedy: no truncation, temperature zero.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Plain temperature sampling with no filters.
    pub fn temperature(t: f32) -> Self {
        Self {
            temperature: t,
            ..Self::default()
        }
    }

    /// Run the pipeline and return the normalized distribution.
    pub fn distribution(&self, logits: &[f32]) -> Result<SamplingDistribution> {
        if logits.is_empty() {
            return Err(Error::InvalidRequest {
                field: "logits",
                detail: "empty vocabulary".into(),
            });
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(Error::InvalidRequest {
                field: "temperature",
                detail: format!(
                    "temperature must be finite and >= 0, got {}",
                    self.temperature
                ),
            });
        }
        for (i, l) in logits.iter().enumerate() {
            if l.is_nan() {
                return Err(Error::Numerical {
                    detail: format!("logit {i} is NaN"),
                });
            }
            if *l == f32::INFINITY {
                // Document 05: "unexpected +inf is a numerical failure, not
                // silently a uniform draw." -inf is a legitimate mask.
                return Err(Error::Numerical {
                    detail: format!("logit {i} is +inf"),
                });
            }
        }
        for b in &self.banned {
            if *b as usize >= logits.len() {
                return Err(Error::InvalidRequest {
                    field: "banned",
                    detail: format!("token {b} outside a vocabulary of {}", logits.len()),
                });
            }
        }

        // Stage 1: hard legality. A banned token is removed from the candidate
        // set here and can never come back, whatever a later stage does.
        let mut legal: Vec<bool> = logits.iter().map(|l| *l != f32::NEG_INFINITY).collect();
        for b in &self.banned {
            legal[*b as usize] = false;
        }
        if !legal.iter().any(|v| *v) {
            return Err(Error::Numerical {
                detail: "every candidate token is banned or masked".into(),
            });
        }

        // Stage 2: the shared pre-truncation distribution. Document 05: top-p,
        // min-p and typical-p all measure against *this* one, not against an
        // accidentally renormalised intermediate subset.
        let base = normalize(logits, &legal);

        // Stage 3: top-k, then top-p, then min-p, intersecting the survivors.
        let mut keep = legal.clone();
        let order = descending_by_prob(&base, &keep);

        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(Error::InvalidRequest {
                    field: "top_k",
                    detail: "top_k 0 would ban every token".into(),
                });
            }
            for id in order.iter().skip(k) {
                keep[*id as usize] = false;
            }
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) || !p.is_finite() {
                return Err(Error::InvalidRequest {
                    field: "top_p",
                    detail: format!("top_p must be in [0, 1], got {p}"),
                });
            }
            // The smallest descending prefix whose mass reaches `p`. The token
            // that crosses the threshold is included; at least one always is.
            let mut mass = 0f32;
            let mut reached = false;
            for id in &order {
                if !keep[*id as usize] {
                    continue;
                }
                if reached {
                    keep[*id as usize] = false;
                } else {
                    mass += base[*id as usize];
                    if mass >= p {
                        reached = true;
                    }
                }
            }
        }
        if let Some(m) = self.min_p {
            if !(0.0..=1.0).contains(&m) || !m.is_finite() {
                return Err(Error::InvalidRequest {
                    field: "min_p",
                    detail: format!("min_p must be in [0, 1], got {m}"),
                });
            }
            // Compared against the maximum of the *same* pre-truncation
            // distribution, per document 05.
            let max_p = base
                .iter()
                .enumerate()
                .filter(|(i, _)| legal[*i])
                .map(|(_, p)| *p)
                .fold(0f32, f32::max);
            let floor = m * max_p;
            for (i, p) in base.iter().enumerate() {
                if keep[i] && *p < floor {
                    keep[i] = false;
                }
            }
        }
        if !keep.iter().any(|v| *v) {
            return Err(Error::Numerical {
                detail: "every candidate was filtered out".into(),
            });
        }

        // Stage 4: temperature, last. Zero means argmax over the final scores,
        // with the lowest id winning a tie.
        if self.temperature == 0.0 {
            let mut best: Option<usize> = None;
            for (i, l) in logits.iter().enumerate() {
                if !keep[i] {
                    continue;
                }
                match best {
                    Some(b) if logits[b] >= *l => {}
                    _ => best = Some(i),
                }
            }
            let mut probs = vec![0f32; logits.len()];
            probs[best.expect("at least one survivor")] = 1.0;
            return Ok(SamplingDistribution { probs });
        }

        let probs = normalize_with_temperature(logits, &keep, self.temperature);
        check_distribution(&probs)?;
        Ok(SamplingDistribution { probs })
    }
}

/// Softmax over the entries `keep` allows; exactly zero elsewhere.
fn normalize(logits: &[f32], keep: &[bool]) -> Vec<f32> {
    normalize_with_temperature(logits, keep, 1.0)
}

/// Softmax of `logits / temperature`, shifting **before** the division.
///
/// The order matters and the second M0 review found it wrong here. Dividing
/// first overflows for a small temperature -- `9.0 / f32::MIN_POSITIVE` is
/// `+inf` -- and the stabilising `l - max` subtraction then evaluates
/// `inf - inf`, so the whole distribution came back as `NaN` with an `Ok`.
/// Subtracting the maximum first bounds every numerator at zero, so the division
/// produces `0` or a large negative number and `exp` produces `1` or `0`.
fn normalize_with_temperature(logits: &[f32], keep: &[bool], temperature: f32) -> Vec<f32> {
    let max = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, l)| {
            if keep[i] {
                ((l - max) / temperature).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Refuse to return a distribution that is not one.
///
/// Document 05: "Empty vocabulary, NaN logits and all-illegal candidate sets
/// produce typed errors." A silently non-finite or unnormalised result is the
/// same class of failure arriving by a different route, and this reference is
/// what speculative verification will consume -- `max(p - q, 0)` over a `NaN`
/// is not a recoverable situation further downstream.
fn check_distribution(probs: &[f32]) -> Result<()> {
    for (i, p) in probs.iter().enumerate() {
        if !p.is_finite() || *p < 0.0 {
            return Err(Error::Numerical {
                detail: format!("probability {i} is {p}"),
            });
        }
    }
    let sum: f32 = probs.iter().sum();
    if !(0.999..=1.001).contains(&sum) {
        return Err(Error::Numerical {
            detail: format!("distribution sums to {sum}, not one"),
        });
    }
    Ok(())
}

/// Candidate ids in descending probability, lowest id first on a tie.
fn descending_by_prob(probs: &[f32], keep: &[bool]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..probs.len() as u32)
        .filter(|i| keep[*i as usize])
        .collect();
    // `sort_by` is stable and `ids` starts ascending, so equal probabilities
    // keep the lower id first.
    ids.sort_by(|a, b| {
        probs[*b as usize]
            .partial_cmp(&probs[*a as usize])
            .expect("probabilities are finite")
    });
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(d: &SamplingDistribution) -> f32 {
        d.probs.iter().sum()
    }

    #[test]
    fn a_distribution_is_normalized_and_zero_outside_its_support() {
        let d = Pipeline::temperature(1.0)
            .distribution(&[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert!((sum(&d) - 1.0).abs() < 1e-6);
        assert_eq!(d.support(), vec![0, 1, 2, 3]);
        assert!(d.probs[3] > d.probs[0]);
    }

    #[test]
    fn temperature_zero_is_argmax_with_the_lowest_id_winning_a_tie() {
        let d = Pipeline::greedy()
            .distribution(&[1.0, 5.0, 5.0, 2.0])
            .unwrap();
        assert_eq!(d.argmax(), 1);
        assert_eq!(d.probs, vec![0.0, 1.0, 0.0, 0.0]);

        // A four-way tie: token 0, deterministically, every time.
        for _ in 0..16 {
            let d = Pipeline::greedy().distribution(&[2.0; 4]).unwrap();
            assert_eq!(d.argmax(), 0);
            assert_eq!(d.probs[0], 1.0);
        }
    }

    #[test]
    fn temperature_scales_the_final_scores() {
        let logits = [0.0f32, 1.0];
        let cold = Pipeline::temperature(0.5).distribution(&logits).unwrap();
        let hot = Pipeline::temperature(2.0).distribution(&logits).unwrap();
        // Lower temperature concentrates mass on the leader.
        assert!(cold.probs[1] > hot.probs[1]);
        assert!((sum(&cold) - 1.0).abs() < 1e-6);
        assert!((sum(&hot) - 1.0).abs() < 1e-6);
        // Both agree with the analytic softmax.
        let want = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((cold.probs[1] - want).abs() < 1e-6);
    }

    #[test]
    fn top_k_keeps_exactly_k_candidates_and_renormalises() {
        let logits = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let p = Pipeline {
            top_k: Some(2),
            temperature: 1.0,
            ..Pipeline::default()
        };
        let d = p.distribution(&logits).unwrap();
        assert_eq!(d.support(), vec![3, 4]);
        assert!((sum(&d) - 1.0).abs() < 1e-6);
        assert_eq!(d.probs[0], 0.0, "a filtered token is exactly zero");

        // k at and beyond the vocabulary keeps everything.
        let all = Pipeline {
            top_k: Some(9),
            temperature: 1.0,
            ..Pipeline::default()
        }
        .distribution(&logits)
        .unwrap();
        assert_eq!(all.support().len(), 5);
        assert!(
            Pipeline {
                top_k: Some(0),
                temperature: 1.0,
                ..Pipeline::default()
            }
            .distribution(&logits)
            .is_err()
        );
    }

    #[test]
    fn top_p_takes_the_smallest_prefix_that_reaches_the_mass() {
        // Analytic fixture: probabilities 0.5, 0.25, 0.125, 0.125 by
        // construction, so the prefix boundaries are exact.
        let logits: Vec<f32> = [0.5f32, 0.25, 0.125, 0.125]
            .iter()
            .map(|p| p.ln())
            .collect();
        let dist = |p: f32| {
            Pipeline {
                top_p: Some(p),
                temperature: 1.0,
                ..Pipeline::default()
            }
            .distribution(&logits)
            .unwrap()
        };
        assert_eq!(dist(0.5).support(), vec![0]);
        assert_eq!(dist(0.4).support(), vec![0]);
        assert_eq!(dist(0.6).support(), vec![0, 1]);
        assert_eq!(dist(0.75).support(), vec![0, 1]);
        assert_eq!(dist(0.8).support(), vec![0, 1, 2]);
        assert_eq!(dist(1.0).support(), vec![0, 1, 2, 3]);
        // Even a mass of zero leaves one survivor rather than banning the field.
        assert_eq!(dist(0.0).support(), vec![0]);
    }

    #[test]
    fn min_p_compares_against_the_pre_truncation_maximum() {
        // The detail document 05 calls out: min-p uses "the maximum probability
        // from that same distribution", not the maximum of whatever survived an
        // earlier stage. Probabilities are 0.5, 0.25, 0.125, 0.125.
        let logits: Vec<f32> = [0.5f32, 0.25, 0.125, 0.125]
            .iter()
            .map(|p| p.ln())
            .collect();
        let d = Pipeline {
            min_p: Some(0.5), // floor = 0.5 * 0.5 = 0.25
            temperature: 1.0,
            ..Pipeline::default()
        }
        .distribution(&logits)
        .unwrap();
        assert_eq!(d.support(), vec![0, 1]);

        // Combined with top-k: the floor is still measured against the full
        // distribution's maximum, so shrinking the candidate set first does not
        // move it.
        let combined = Pipeline {
            top_k: Some(3),
            min_p: Some(0.5),
            temperature: 1.0,
            ..Pipeline::default()
        }
        .distribution(&logits)
        .unwrap();
        assert_eq!(combined.support(), vec![0, 1]);
    }

    #[test]
    fn a_banned_token_stays_banned_through_every_later_stage() {
        // Document 05: "hard-banned tokens remain banned throughout later
        // stages." Banning the leader must not merely reorder it back in.
        let logits = [5.0f32, 1.0, 0.0];
        let d = Pipeline {
            banned: vec![0],
            top_k: Some(3),
            top_p: Some(1.0),
            min_p: Some(0.0),
            temperature: 1.0,
        }
        .distribution(&logits)
        .unwrap();
        assert_eq!(d.probs[0], 0.0);
        assert_eq!(d.support(), vec![1, 2]);
        assert!((sum(&d) - 1.0).abs() < 1e-6);

        // Greedy must not select it either.
        let g = Pipeline {
            banned: vec![0],
            ..Pipeline::greedy()
        }
        .distribution(&logits)
        .unwrap();
        assert_eq!(g.argmax(), 1);
    }

    #[test]
    fn a_negative_infinity_mask_is_honoured_and_a_positive_one_is_a_failure() {
        // -inf is how a legality mask arrives; +inf is a defect upstream.
        let d = Pipeline::temperature(1.0)
            .distribution(&[f32::NEG_INFINITY, 1.0, 2.0])
            .unwrap();
        assert_eq!(d.probs[0], 0.0);
        assert_eq!(d.support(), vec![1, 2]);

        assert_eq!(
            Pipeline::temperature(1.0)
                .distribution(&[f32::INFINITY, 1.0])
                .unwrap_err()
                .kind(),
            "numerical"
        );
    }

    #[test]
    fn an_all_banned_candidate_set_is_a_typed_failure_not_an_argmax() {
        // The legacy fallback document 05 forbids inheriting: "Do not inherit
        // the old fallback that can select an argmax after all tokens were
        // banned."
        let e = Pipeline {
            banned: vec![0, 1, 2],
            ..Pipeline::greedy()
        }
        .distribution(&[1.0, 2.0, 3.0])
        .unwrap_err();
        assert_eq!(e.kind(), "numerical");

        let all_masked = Pipeline::greedy()
            .distribution(&[f32::NEG_INFINITY; 3])
            .unwrap_err();
        assert_eq!(all_masked.kind(), "numerical");
    }

    #[test]
    fn malformed_requests_are_request_errors_not_numerical_ones() {
        // Document 05 keeps the two apart: a bad request is the caller's, a
        // numerical failure is execution's.
        assert_eq!(
            Pipeline::greedy().distribution(&[]).unwrap_err().kind(),
            "invalid_request"
        );
        for bad in [-1.0f32, f32::NAN, f32::INFINITY] {
            assert_eq!(
                Pipeline::temperature(bad)
                    .distribution(&[1.0])
                    .unwrap_err()
                    .kind(),
                "invalid_request",
                "temperature {bad}"
            );
        }
        for bad in [-0.1f32, 1.1, f32::NAN] {
            assert!(
                Pipeline {
                    top_p: Some(bad),
                    temperature: 1.0,
                    ..Pipeline::default()
                }
                .distribution(&[1.0, 2.0])
                .is_err()
            );
            assert!(
                Pipeline {
                    min_p: Some(bad),
                    temperature: 1.0,
                    ..Pipeline::default()
                }
                .distribution(&[1.0, 2.0])
                .is_err()
            );
        }
        assert!(
            Pipeline {
                banned: vec![9],
                ..Pipeline::greedy()
            }
            .distribution(&[1.0, 2.0])
            .is_err()
        );
        assert_eq!(
            Pipeline::greedy()
                .distribution(&[1.0, f32::NAN])
                .unwrap_err()
                .kind(),
            "numerical"
        );
    }

    #[test]
    fn an_extreme_temperature_stays_a_distribution() {
        // Second review, reproduced: `temperature = f32::MIN_POSITIVE` returned
        // `Ok([NaN, NaN])`, because the division overflowed before the
        // stabilising subtraction could bound it.
        let logits = [8.0f32, 9.0];
        for t in [
            f32::MIN_POSITIVE,
            1e-30,
            1e-6,
            0.5,
            1.0,
            1e6,
            1e30,
            f32::MAX,
        ] {
            let d = Pipeline::temperature(t).distribution(&logits).unwrap();
            assert!(
                d.probs.iter().all(|p| p.is_finite()),
                "temperature {t} produced {:?}",
                d.probs
            );
            let s: f32 = d.probs.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "temperature {t} summed to {s}");
        }

        // A vanishing temperature concentrates on the leader, like greedy does.
        let cold = Pipeline::temperature(f32::MIN_POSITIVE)
            .distribution(&logits)
            .unwrap();
        assert_eq!(cold.probs, vec![0.0, 1.0]);
        assert_eq!(
            cold.argmax(),
            Pipeline::greedy().distribution(&logits).unwrap().argmax()
        );

        // A huge temperature flattens toward uniform rather than overflowing.
        let hot = Pipeline::temperature(f32::MAX)
            .distribution(&logits)
            .unwrap();
        assert!((hot.probs[0] - 0.5).abs() < 1e-5, "{:?}", hot.probs);
    }

    #[test]
    fn a_wide_logit_range_does_not_overflow_at_a_small_temperature() {
        // The same failure at the scale a real vocabulary reaches.
        let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 1e3).collect();
        let d = Pipeline::temperature(1e-20).distribution(&logits).unwrap();
        assert!(d.probs.iter().all(|p| p.is_finite()));
        assert_eq!(d.support(), vec![63]);
    }

    #[test]
    fn exhaustive_tiny_vocabulary_stays_normalized_under_every_filter_combination() {
        // Document 07: "Use exhaustive tiny vocabularies for every pipeline
        // stage and combinations where order matters."
        let logits = [3.0f32, 3.0, 1.0, 0.0, -2.0];
        for k in [None, Some(1), Some(2), Some(5)] {
            for p in [None, Some(0.1f32), Some(0.5), Some(1.0)] {
                for m in [None, Some(0.0f32), Some(0.3), Some(1.0)] {
                    for t in [0.0f32, 0.5, 1.0, 4.0] {
                        let pipe = Pipeline {
                            banned: vec![4],
                            top_k: k,
                            top_p: p,
                            min_p: m,
                            temperature: t,
                        };
                        let d = pipe.distribution(&logits).unwrap();
                        let s: f32 = d.probs.iter().sum();
                        assert!((s - 1.0).abs() < 1e-5, "{pipe:?} summed to {s}");
                        assert_eq!(d.probs[4], 0.0, "{pipe:?} un-banned token 4");
                        assert!(!d.support().is_empty());
                        // The leader ties at ids 0 and 1; 0 must always survive.
                        assert!(d.probs[0] > 0.0, "{pipe:?} dropped the tied leader");
                    }
                }
            }
        }
    }
}
