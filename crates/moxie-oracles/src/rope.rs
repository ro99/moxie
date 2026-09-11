//! Rotary positional embedding.
//!
//! Document 02: "RoPE/other positional operations with checkpoint-defined
//! scaling, partial dimensions, and position semantics." All three appear here
//! as parameters rather than constants: the base is supplied, the rotary
//! dimension may be smaller than the head dimension, and the position is the
//! **absolute** position in the sequence.
//!
//! That last one is R21 and R19 in one line. A chunked prefill that passed the
//! index within its chunk would be right for the first chunk and wrong for every
//! one after it, and the resulting model would produce plausible text.

use moxie_types::{Error, Result};

/// Rotate one head's vector in place-style, returning a new vector.
///
/// For `j` in `0..rotary_dim/2`, with `θ_j = pos · base^(−2j / rotary_dim)`:
///
/// ```text
/// y[2j]   = x[2j]·cos(θ_j) − x[2j+1]·sin(θ_j)
/// y[2j+1] = x[2j]·sin(θ_j) + x[2j+1]·cos(θ_j)
/// y[i]    = x[i]                                for i >= rotary_dim
/// ```
///
/// Angles are computed in FP64 and narrowed once, because `pos · base^(−2j/d)`
/// for a large position loses its low bits in FP32 long before the rotation
/// does. The multiply-add that follows is FP32, which is what the `γ(4)` bound
/// in task 0003 counts.
pub fn rope_head(x: &[f32], pos: u64, base: f32, rotary_dim: usize) -> Result<Vec<f32>> {
    if rotary_dim > x.len() {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "rotary_dim {rotary_dim} exceeds the head dimension {}",
                x.len()
            ),
        });
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(Error::InvalidArtifact {
            detail: format!("rotary_dim {rotary_dim} is odd; rotation is over pairs"),
        });
    }
    if !(base.is_finite() && base > 1.0) {
        return Err(Error::InvalidRequest {
            field: "rope_base",
            detail: format!("base must be finite and > 1, got {base}"),
        });
    }
    let mut out = crate::try_clone_slice(x)?;
    let half = rotary_dim / 2;
    for j in 0..half {
        let inv_freq = (base as f64).powf(-2.0 * j as f64 / rotary_dim as f64);
        let theta = pos as f64 * inv_freq;
        let (sin, cos) = (theta.sin() as f32, theta.cos() as f32);
        let a = x[2 * j];
        let b = x[2 * j + 1];
        out[2 * j] = a * cos - b * sin;
        out[2 * j + 1] = a * sin + b * cos;
    }
    Ok(out)
}

/// Rotate every head of one row. `x` is `heads · head_dim` long.
pub fn rope_row(
    x: &[f32],
    pos: u64,
    base: f32,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Vec<f32>> {
    if x.len() != heads * head_dim {
        return Err(Error::InvalidArtifact {
            detail: format!("row has {} elements, expected {heads}x{head_dim}", x.len()),
        });
    }
    let mut out = crate::try_vec(x.len())?;
    for h in 0..heads {
        let head = &x[h * head_dim..(h + 1) * head_dim];
        out.extend(rope_head(head, pos, base, rotary_dim)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    /// The equation, transcribed separately in FP64.
    fn rope_head_f64(x: &[f32], pos: u64, base: f32, rotary_dim: usize) -> Vec<f64> {
        let mut out: Vec<f64> = x.iter().map(|v| *v as f64).collect();
        for j in 0..rotary_dim / 2 {
            let theta = pos as f64 * (base as f64).powf(-2.0 * j as f64 / rotary_dim as f64);
            let (sin, cos) = (theta.sin(), theta.cos());
            let a = x[2 * j] as f64;
            let b = x[2 * j + 1] as f64;
            out[2 * j] = a * cos - b * sin;
            out[2 * j + 1] = a * sin + b * cos;
        }
        out
    }

    #[test]
    fn rope_matches_the_equation_within_gamma_four() {
        let d = 64usize;
        let x: Vec<f32> = (0..d)
            .map(|i| ((i * 17 % 53) as f32 - 26.0) / 5.0)
            .collect();
        let mut worst = ErrorSummary::absolute(&[], &[]);
        for pos in [0u64, 1, 7, 128, 4095] {
            let got = rope_head(&x, pos, 10_000.0, d).unwrap();
            let want = rope_head_f64(&x, pos, 10_000.0, d);
            // Absolute, scaled by the magnitudes that enter each output --
            // the rotation can cancel, so a relative-to-|y| bound would be
            // meaningless exactly where it matters.
            let scale: Vec<f64> = (0..d)
                .map(|i| {
                    let pair = i / 2 * 2;
                    (x[pair] as f64).abs() + (x[pair + 1] as f64).abs()
                })
                .collect();
            let s = ErrorSummary::normalized(&got, &want, &scale);
            assert!(s.within(gamma(4)), "pos {pos}: {s} exceeded gamma(4)");
            if s.max > worst.max {
                worst = s;
            }
        }
        assert!(worst.count > 0);
    }

    #[test]
    fn position_zero_is_the_identity() {
        // theta = 0 for every j, so cos = 1 and sin = 0 exactly.
        let x: Vec<f32> = (0..16).map(|i| i as f32 - 8.0).collect();
        assert_eq!(rope_head(&x, 0, 10_000.0, 16).unwrap(), x);
    }

    #[test]
    fn a_later_position_is_not_the_first_position() {
        // R21 as an assertion: if an implementation passed the index within a
        // chunk instead of the absolute position, this is what would silently
        // hold.
        let x: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let at_zero = rope_head(&x, 0, 10_000.0, 8).unwrap();
        let at_five = rope_head(&x, 5, 10_000.0, 8).unwrap();
        assert_ne!(at_zero, at_five);
    }

    #[test]
    fn rotation_preserves_the_norm_of_each_pair() {
        // The defining property: each 2-vector is rotated, not scaled.
        let x = [3.0f32, 4.0, -1.0, 2.0];
        for pos in [0u64, 1, 3, 1000] {
            let y = rope_head(&x, pos, 10_000.0, 4).unwrap();
            for j in 0..2 {
                let before = (x[2 * j] as f64).hypot(x[2 * j + 1] as f64);
                let after = (y[2 * j] as f64).hypot(y[2 * j + 1] as f64);
                assert!(
                    (before - after).abs() < 1e-5 * before,
                    "pair {j} at pos {pos}: {before} -> {after}"
                );
            }
        }
    }

    #[test]
    fn a_partial_rotary_dimension_leaves_the_tail_untouched() {
        // Document 02's "partial dimensions". The unrotated tail must be
        // bit-identical, not approximately equal.
        let x: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let y = rope_head(&x, 9, 10_000.0, 4).unwrap();
        assert_eq!(&y[4..], &x[4..]);
        assert_ne!(&y[..4], &x[..4]);
    }

    #[test]
    fn the_base_is_a_parameter_and_changes_the_rotation() {
        let x = [1.0f32, 0.0, 1.0, 0.0];
        let a = rope_head(&x, 3, 10_000.0, 4).unwrap();
        let b = rope_head(&x, 3, 1_000_000.0, 4).unwrap();
        assert_ne!(a, b, "a checkpoint-defined base is part of the function");
    }

    #[test]
    fn odd_or_oversized_rotary_dimensions_are_refused() {
        let x = [1.0f32; 8];
        assert!(rope_head(&x, 1, 10_000.0, 3).is_err(), "odd");
        assert!(
            rope_head(&x, 1, 10_000.0, 9).is_err(),
            "larger than the head"
        );
        assert!(rope_head(&x, 1, 0.5, 4).is_err(), "base <= 1");
        assert!(rope_head(&x, 1, f32::NAN, 4).is_err());
        assert!(
            rope_row(&[1.0; 7], 1, 10_000.0, 2, 4, 4).is_err(),
            "ragged row"
        );
    }

    #[test]
    fn every_head_of_a_row_rotates_by_the_same_position() {
        let heads = 3usize;
        let head_dim = 4usize;
        let x: Vec<f32> = (0..heads * head_dim)
            .map(|i| (i % 4) as f32 + 1.0)
            .collect();
        let row = rope_row(&x, 6, 10_000.0, heads, head_dim, head_dim).unwrap();
        let one = rope_head(&x[..head_dim], 6, 10_000.0, head_dim).unwrap();
        // Heads 0 and 1 hold the same values here, so their rotations agree.
        assert_eq!(&row[..head_dim], &one[..]);
        assert_eq!(&row[..head_dim], &row[head_dim..2 * head_dim]);
    }
}
