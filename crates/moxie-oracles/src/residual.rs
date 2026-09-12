//! Residual combination.
//!
//! Trivial arithmetic with a non-trivial rule attached: document 04 requires
//! that "bias and residual terms are applied exactly once, not on every partial
//! output before an unintended sum". That is a partitioning rule rather than an
//! equation, and it is why this is its own operation with its own
//! `PartitionRule::Replicated` rather than an anonymous `+` inside another node.

use moxie_types::{Error, Result};

/// `y[i] = a[i] + b[i]`, FP32, unrounded.
pub fn residual_row(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    residual_row_scaled(a, b, 1.0)
}

/// `y[i] = (a[i] + b[i]) · scale`, the sum in FP64 and narrowed once.
///
/// The scale is per-residual because Gemma 4 applies a checkpoint scalar to its
/// MLP residual and leaves its attention residual alone
/// (`src/models/gemma4/gemma4_runtime.cpp:1230` against `:1186`). A single
/// per-layer factor would be wrong on half the residuals in that graph.
///
/// The sum is evaluated in FP64 before the multiply so the scale cannot rescue
/// or destroy an FP32 intermediate that the unscaled sum would have overflowed:
/// the declared boundary is one rounding on the scaled result, and two
/// roundings would be a different contract.
pub fn residual_row_scaled(a: &[f32], b: &[f32], scale: f32) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(Error::InvalidArtifact {
            detail: format!("operands have {} and {} elements", a.len(), b.len()),
        });
    }
    if a.is_empty() {
        return Err(Error::InvalidRequest {
            field: "residual",
            detail: "a residual over zero features".into(),
        });
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(Error::InvalidRequest {
            field: "residual_scale",
            detail: format!("residual scale must be finite and positive, got {scale}"),
        });
    }
    let mut out = crate::try_vec(a.len())?;
    if scale == 1.0 {
        // Exactly the previous contract: one FP32 addition, so a sum that was
        // bit-exact before this parameter existed still is.
        out.extend(a.iter().zip(b).map(|(x, y)| x + y));
    } else {
        let s = scale as f64;
        out.extend(
            a.iter()
                .zip(b)
                .map(|(x, y)| ((*x as f64 + *y as f64) * s) as f32),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    #[test]
    fn a_residual_is_exact_when_the_sum_is_representable() {
        let a = [1.0f32, 0.5, -2.0, 0.0];
        let b = [2.0f32, 0.25, 2.0, -0.0];
        assert_eq!(residual_row(&a, &b).unwrap(), vec![3.0, 0.75, 0.0, 0.0]);
    }

    #[test]
    fn a_residual_is_one_rounding_otherwise() {
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) / 3.0).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32) / 7.0 + 1e-8).collect();
        let got = residual_row(&a, &b).unwrap();
        let want: Vec<f64> = (0..n).map(|i| a[i] as f64 + b[i] as f64).collect();
        let scale: Vec<f64> = (0..n)
            .map(|i| (a[i] as f64).abs() + (b[i] as f64).abs())
            .collect();
        let s = ErrorSummary::normalized(&got, &want, &scale);
        assert!(s.within(gamma(1)), "{s} exceeded gamma(1)");
    }

    #[test]
    fn adding_a_residual_twice_is_not_the_same_as_adding_it_once() {
        // Document 04's rule as an assertion. A partitioned linear that applied
        // the residual on each rank's partial output before the reduction would
        // produce this.
        let x = [1.0f32, 2.0];
        let r = [10.0f32, 20.0];
        let once = residual_row(&x, &r).unwrap();
        let twice = residual_row(&once, &r).unwrap();
        assert_ne!(once, twice);
    }

    #[test]
    fn shape_disagreements_are_typed_errors() {
        assert!(residual_row(&[1.0], &[1.0, 2.0]).is_err());
        assert!(residual_row(&[], &[]).is_err());
    }

    #[test]
    fn a_unit_scale_is_the_unscaled_residual_bit_for_bit() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32) / 7.0 - 9.0).collect();
        let b: Vec<f32> = (0..128).map(|i| -(i as f32) / 11.0 + 1e-7).collect();
        assert_eq!(
            residual_row_scaled(&a, &b, 1.0).unwrap(),
            residual_row(&a, &b).unwrap()
        );
    }

    #[test]
    fn the_scale_applies_to_the_sum_not_to_each_operand() {
        // Gemma's MLP residual is `(h + f) * s`, not `h + f * s`. The two agree
        // only when `h` is zero, which is why the fixture uses a nonzero one.
        let (a, b, s) = ([4.0f32], [1.0f32], 0.5f32);
        assert_eq!(residual_row_scaled(&a, &b, s).unwrap(), vec![2.5]);
        assert_ne!(residual_row_scaled(&a, &b, s).unwrap(), vec![4.5]);
    }

    #[test]
    fn the_two_residuals_in_a_gemma_layer_do_not_share_a_factor() {
        // The attention residual is unscaled and the MLP residual is not.
        // Applying one factor to both changes the stream at every layer.
        let stream = [1.0f32, -2.0, 0.5];
        let attn = [0.25f32, 0.5, -0.75];
        let mlp = [2.0f32, 1.0, -1.0];
        let scalar = 0.875f32;
        let correct = {
            let after_attn = residual_row_scaled(&stream, &attn, 1.0).unwrap();
            residual_row_scaled(&after_attn, &mlp, scalar).unwrap()
        };
        let both_scaled = {
            let after_attn = residual_row_scaled(&stream, &attn, scalar).unwrap();
            residual_row_scaled(&after_attn, &mlp, scalar).unwrap()
        };
        assert_ne!(correct, both_scaled);
    }

    #[test]
    fn a_scale_must_be_finite_and_positive() {
        for s in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(residual_row_scaled(&[1.0], &[1.0], s).is_err(), "{s}");
        }
    }
}
