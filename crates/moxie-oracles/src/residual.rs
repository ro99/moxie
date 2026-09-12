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

/// `y[i] = bf16(a[i] + b[i]) · scale`.
///
/// The scale is per-residual because Gemma 4 applies a checkpoint scalar to its
/// MLP residual and leaves its attention residual alone
/// (`src/models/gemma4/gemma4_runtime.cpp:1230` against `:1186`). A single
/// per-layer factor would be wrong on half the residuals in that graph.
///
/// **The sum passes through BF16 before the multiply, and that boundary is part
/// of the equation.** The pinned source is
/// `bf16_round_f32(bf16_round_f32(h + n) * scalar)`: two roundings, not one.
/// The first version of this function evaluated `(a + b) * scale` in FP64 and
/// left the single rounding to the caller, reasoning that fewer roundings is
/// more accurate. More accurate is not the contract. With `a = 1`,
/// `b = 2^-8` and `scale = 0.875` the sum is exactly halfway between two BF16
/// values, so the declared boundary rounds it to 1.0 and the result is 0.875,
/// while carrying the extra precision gives 0.87890625 — a difference that then
/// enters every later layer.
///
/// The caller still rounds the result, which is the second boundary; at
/// `scale == 1.0` the two formulations agree, and that path is kept unrounded
/// so [`residual_row`] remains the exact FP32 sum its own contract promises.
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
        // One FP32 addition, so a sum that was bit-exact before this parameter
        // existed still is. Rounding here would be equivalent -- the caller
        // rounds anyway -- but `residual_row` promises the unrounded sum.
        out.extend(a.iter().zip(b).map(|(x, y)| x + y));
    } else {
        out.extend(
            a.iter()
                .zip(b)
                .map(|(x, y)| crate::bf16_round(x + y) * scale),
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
    fn the_sum_is_rounded_to_bf16_before_the_scale_applies() {
        // Independent review finding, reproduced. `1 + 2^-8` is exactly halfway
        // between two BF16 values, so the declared first boundary resolves it
        // to 1.0 and the scaled result is 0.875. Carrying FP64 precision
        // through the multiply instead gives 0.87890625, which is a different
        // residual stream for every later layer.
        let (a, b, scale) = ([1.0f32], [0.00390625f32], 0.875f32);
        let got = residual_row_scaled(&a, &b, scale).unwrap();
        assert_eq!(
            got,
            vec![0.875],
            "the intermediate BF16 boundary is missing"
        );
        let unrounded = ((a[0] as f64 + b[0] as f64) * scale as f64) as f32;
        assert_eq!(crate::bf16_round(unrounded), 0.87890625);
        assert_ne!(got[0], crate::bf16_round(unrounded));
    }

    #[test]
    fn the_declared_two_boundary_sequence_holds_across_probes() {
        // The equation transcribed at the call site, over operands whose sums
        // are not BF16-exact. `bf16(bf16(a + b) * s)` is what the caller sees
        // after its own rounding, which is the boundary this crate does not own.
        for (a, b, s) in [
            (1.0f32, 0.00390625f32, 0.875f32),
            (2.0, 0.015625, 0.625),
            (-3.5, 0.0078125, 1.75),
            (0.125, 0.00048828125, 0.5),
        ] {
            let want = crate::bf16_round(crate::bf16_round(a + b) * s);
            let got = crate::bf16_round(residual_row_scaled(&[a], &[b], s).unwrap()[0]);
            assert_eq!(got, want, "a={a} b={b} s={s}");
        }
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
        // 5.0 is BF16-exact, so the intermediate boundary is not what this
        // fixture is measuring.
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
