//! RMS normalisation, and the LayerNorm it must not be confused with.
//!
//! Document 02: "RMSNorm and LayerNorm as distinct operations; explicit epsilon,
//! axes, pre/post scaling, and accumulation." They are separate here for the
//! reason document 02 gives -- GLM-5.3 uses LayerNorm where its siblings use
//! RMSNorm, and treating them as one parameterised norm loses the mean
//! subtraction. A test asserts they differ on an input with non-zero mean, so
//! collapsing them fails rather than merely looking wrong in review.
//!
//! Only RMSNorm is an implemented graph operation in task 0003's slice.
//! `layer_norm` is here as the thing it is *not*: it has no registered oracle
//! and no `OpParams`, so it cannot be built into a graph.

use moxie_types::{Error, Result};

/// `y[i] = x[i] · g[i] / sqrt( mean(x²) + ε )`, FP32 throughout, unrounded.
///
/// The reduction is sequential ascending, as everywhere in this crate. `eps` is
/// a required argument, never defaulted: two checkpoints that declare different
/// epsilons are two different functions.
pub fn rms_norm_row(x: &[f32], gain: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.is_empty() {
        return Err(Error::InvalidRequest {
            field: "rms_norm",
            detail: "a norm over zero features".into(),
        });
    }
    if gain.len() != x.len() {
        return Err(Error::InvalidArtifact {
            detail: format!("gain has {} elements for {} features", gain.len(), x.len()),
        });
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidRequest {
            field: "eps",
            detail: format!("epsilon must be finite and positive, got {eps}"),
        });
    }
    let mut sum = 0f32;
    for v in x {
        sum += v * v;
    }
    let mean = sum / x.len() as f32;
    let denom = (mean + eps).sqrt();
    if !denom.is_finite() || denom == 0.0 {
        return Err(Error::Numerical {
            detail: format!("rms denominator is {denom}"),
        });
    }
    let mut out = crate::try_vec(x.len())?;
    out.extend(x.iter().zip(gain).map(|(v, g)| v * g / denom));
    Ok(out)
}

/// [`rms_norm_row`] applied to each of `group` contiguous lanes of a row.
///
/// The row is `group` blocks of `x.len() / group` elements; each block is
/// normalized against **its own** mean square and multiplied by the same gain,
/// which is one block wide. `group == 1` is [`rms_norm_row`] exactly.
///
/// This is how per-head query and key normalization works
/// (`src/models/gemma4/gemma4_runtime.cpp:798`, which loops heads and norms
/// each one). Normalizing the whole concatenated row instead would let one
/// head's magnitude set every other head's scale factor, which is a different
/// function that happens to have the same shape.
pub fn rms_norm_row_grouped(x: &[f32], gain: &[f32], group: usize, eps: f32) -> Result<Vec<f32>> {
    if group == 0 {
        return Err(Error::InvalidRequest {
            field: "group",
            detail: "a norm over zero groups".into(),
        });
    }
    if x.is_empty() || !x.len().is_multiple_of(group) {
        return Err(Error::InvalidArtifact {
            detail: format!("{} features do not divide into {group} group(s)", x.len()),
        });
    }
    let width = x.len() / group;
    let mut out = crate::try_vec(x.len())?;
    for g in 0..group {
        out.extend(rms_norm_row(&x[g * width..(g + 1) * width], gain, eps)?);
    }
    Ok(out)
}

/// `y[i] = (x[i] − mean(x)) · g[i] / sqrt(var(x) + ε) + b[i]`.
///
/// **Not** an operation in task 0003's slice, and deliberately not registered.
/// It exists so that the difference from `rms_norm_row` is executable evidence
/// rather than a claim in a comment.
pub fn layer_norm_row(x: &[f32], gain: &[f32], bias: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.is_empty() || gain.len() != x.len() || bias.len() != x.len() {
        return Err(Error::InvalidArtifact {
            detail: "layer norm shapes disagree".into(),
        });
    }
    let n = x.len() as f32;
    let mut mean = 0f32;
    for v in x {
        mean += v;
    }
    mean /= n;
    let mut var = 0f32;
    for v in x {
        var += (v - mean) * (v - mean);
    }
    var /= n;
    let denom = (var + eps).sqrt();
    Ok(x.iter()
        .zip(gain)
        .zip(bias)
        .map(|((v, g), b)| (v - mean) * g / denom + b)
        .collect())
}

/// The scale an `rms_norm_row` bound is stated against: the output magnitude.
pub fn rms_norm_row_scale(want: &[f64]) -> Vec<f64> {
    want.iter().map(|v| v.abs()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    /// The equation from task 0003, transcribed separately in FP64.
    fn rms_norm_row_f64(x: &[f32], gain: &[f32], eps: f32) -> Vec<f64> {
        let n = x.len() as f64;
        let mut sum = 0f64;
        for v in x {
            sum += (*v as f64) * (*v as f64);
        }
        let denom = (sum / n + eps as f64).sqrt();
        x.iter()
            .zip(gain)
            .map(|(v, g)| (*v as f64) * (*g as f64) / denom)
            .collect()
    }

    #[test]
    fn rms_norm_matches_the_equation_within_gamma_h_plus_four() {
        // H accumulation steps, plus the divide by H, the +eps, the sqrt and
        // the gain multiply: gamma(H + 4).
        let h = 512usize;
        let x: Vec<f32> = (0..h)
            .map(|i| ((i * 41 % 97) as f32 - 48.0) / 11.0)
            .collect();
        let gain: Vec<f32> = (0..h).map(|i| 1.0 + (i % 7) as f32 * 0.05).collect();
        let eps = 1e-5f32;

        let got = rms_norm_row(&x, &gain, eps).unwrap();
        let want = rms_norm_row_f64(&x, &gain, eps);
        let s = ErrorSummary::normalized(&got, &want, &rms_norm_row_scale(&want));
        let bound = gamma(h as u64 + 4);
        assert_eq!(s.count, h);
        assert!(
            s.within(bound),
            "{s} exceeded gamma({}) = {bound:.3e}",
            h + 4
        );
    }

    #[test]
    fn rms_norm_is_exact_on_a_constructed_representable_case() {
        // x = [3, 4], mean square = 12.5; with eps chosen so the denominator is
        // exactly 4, every step is representable.
        let x = [3.0f32, 4.0];
        let gain = [1.0f32, 1.0];
        let eps = 3.5f32; // 12.5 + 3.5 = 16, sqrt = 4
        let got = rms_norm_row(&x, &gain, eps).unwrap();
        assert_eq!(got, vec![0.75, 1.0]);
    }

    #[test]
    fn rms_norm_is_not_layer_norm() {
        // The distinction document 02 requires, as an executable check: on an
        // input with non-zero mean the two disagree, because only one subtracts
        // it. Collapsing them into "a parameterised norm" fails here.
        let x = [10.0f32, 11.0, 12.0, 13.0];
        let gain = [1.0f32; 4];
        let bias = [0.0f32; 4];
        let eps = 1e-5f32;

        let rms = rms_norm_row(&x, &gain, eps).unwrap();
        let ln = layer_norm_row(&x, &gain, &bias, eps).unwrap();
        assert_ne!(rms, ln);

        // Concretely: LayerNorm centres, so its outputs sum to ~0. RMSNorm does
        // not, so its outputs keep the input's sign and scale.
        let ln_sum: f32 = ln.iter().sum();
        assert!(ln_sum.abs() < 1e-4, "layer norm centres: {ln_sum}");
        assert!(rms.iter().all(|v| *v > 0.0), "rms norm does not: {rms:?}");

        // On a zero-mean input they coincide, which is exactly why a test on the
        // wrong fixture would not have caught a substitution.
        let centred = [-1.5f32, -0.5, 0.5, 1.5];
        let r = rms_norm_row(&centred, &gain, eps).unwrap();
        let l = layer_norm_row(&centred, &gain, &bias, eps).unwrap();
        for (a, b) in r.iter().zip(&l) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn epsilon_is_required_and_changes_the_result() {
        let x = [1.0f32, 1.0];
        let gain = [1.0f32, 1.0];
        let a = rms_norm_row(&x, &gain, 1e-6).unwrap();
        let b = rms_norm_row(&x, &gain, 1.0).unwrap();
        assert_ne!(a, b, "epsilon is part of the function, not a nicety");

        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(rms_norm_row(&x, &gain, bad).is_err(), "eps {bad}");
        }
    }

    #[test]
    fn an_all_zero_row_is_defined_by_epsilon_rather_than_dividing_by_zero() {
        // The case a missing epsilon turns into NaN.
        let x = [0.0f32; 8];
        let gain = [1.0f32; 8];
        let got = rms_norm_row(&x, &gain, 1e-5).unwrap();
        assert!(got.iter().all(|v| *v == 0.0), "{got:?}");
    }

    #[test]
    fn shape_disagreements_are_typed_errors() {
        assert!(rms_norm_row(&[], &[], 1e-5).is_err());
        assert!(rms_norm_row(&[1.0, 2.0], &[1.0], 1e-5).is_err());
        assert!(layer_norm_row(&[1.0], &[1.0], &[], 1e-5).is_err());
    }

    #[test]
    fn a_grouped_norm_is_each_group_normalized_on_its_own() {
        // Two groups of three with very different magnitudes. A whole-row norm
        // would let the large group set the small group's scale factor.
        let x = [1.0f32, 2.0, 3.0, 100.0, 200.0, 300.0];
        let gain = [1.0f32, 1.0, 1.0];
        let grouped = rms_norm_row_grouped(&x, &gain, 2, 1e-6).unwrap();
        let first = rms_norm_row(&x[..3], &gain, 1e-6).unwrap();
        let second = rms_norm_row(&x[3..], &gain, 1e-6).unwrap();
        assert_eq!(grouped[..3], first[..]);
        assert_eq!(grouped[3..], second[..]);
        // The two groups are scalar multiples of each other in the input and
        // therefore equal after their own norms -- which a whole-row norm
        // would not produce.
        for i in 0..3 {
            assert!((grouped[i] - grouped[i + 3]).abs() < 1e-5);
        }
        let whole = rms_norm_row(&x, &[1.0; 6], 1e-6).unwrap();
        assert_ne!(&grouped[..], &whole[..]);
    }

    #[test]
    fn one_group_is_the_ungrouped_norm_exactly() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 3.0).collect();
        let gain: Vec<f32> = (0..32).map(|i| 1.0 + (i as f32) / 64.0).collect();
        assert_eq!(
            rms_norm_row_grouped(&x, &gain, 1, 1e-5).unwrap(),
            rms_norm_row(&x, &gain, 1e-5).unwrap()
        );
    }

    #[test]
    fn the_gain_is_one_group_wide_and_shared() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        // A gain as wide as the whole row is the wrong length for two groups.
        assert!(rms_norm_row_grouped(&x, &[1.0; 4], 2, 1e-6).is_err());
        assert!(rms_norm_row_grouped(&x, &[1.0; 2], 2, 1e-6).is_ok());
        // Indivisible and zero group counts are typed errors.
        assert!(rms_norm_row_grouped(&x, &[1.0; 2], 3, 1e-6).is_err());
        assert!(rms_norm_row_grouped(&x, &[1.0; 2], 0, 1e-6).is_err());
    }
}
