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

use moxie_graph::RopeLayout;
use moxie_types::{Error, Result};

/// How one head rotates: everything except the vector and the position.
///
/// Grouped rather than passed as four more arguments because they are one
/// decision. `rotary_dim` and `frequency_dim` are also easy to transpose at a
/// call site, and a transposed pair is a wrong angle rather than a type error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation {
    pub base: f32,
    /// How many of a head's elements rotate.
    pub rotary_dim: usize,
    /// The denominator of the inverse-frequency exponent.
    pub frequency_dim: usize,
    pub layout: RopeLayout,
}

impl Rotation {
    /// Task 0003's convention: interleaved pairs, with the rotated width as
    /// the denominator.
    pub fn interleaved(base: f32, rotary_dim: usize) -> Self {
        Self {
            base,
            rotary_dim,
            frequency_dim: rotary_dim,
            layout: RopeLayout::Interleaved,
        }
    }
}

/// Rotate one head's vector in place-style, returning a new vector.
///
/// For `j` in `0..rotary_dim/2`, with `θ_j = pos · base^(−2j / frequency_dim)`,
/// the rotated pair `(p, q)` is chosen by `layout`:
///
/// ```text
/// Interleaved: (p, q) = (2j, 2j+1)
/// HalfSplit:   (p, q) = (j,  j + x.len()/2)
///
/// y[p] = x[p]·cos(θ_j) − x[q]·sin(θ_j)
/// y[q] = x[q]·cos(θ_j) + x[p]·sin(θ_j)
/// y[i] = x[i]                            for every unrotated i
/// ```
///
/// **Both parameters exist because both conventions are in released
/// checkpoints.** Task 0003 pinned the interleaved form with
/// `frequency_dim == rotary_dim`; Gemma 4 pairs halves and, on its global
/// layers, divides by the full head dimension while rotating only a quarter of
/// it (`src/models/gemma4/gemma4_ops.cpp:22`). Under partial rotation the two
/// disagree about which elements move *and* about every angle, so neither can
/// be the other's default.
///
/// Angles are computed in FP64 and narrowed once, because `pos · base^(−2j/d)`
/// for a large position loses its low bits in FP32 long before the rotation
/// does. The multiply-add that follows is FP32, which is what the `γ(4)` bound
/// in task 0003 counts.
pub fn rope_head(x: &[f32], pos: u64, rotation: Rotation) -> Result<Vec<f32>> {
    let Rotation {
        base,
        rotary_dim,
        frequency_dim,
        layout,
    } = rotation;
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
    if frequency_dim == 0 {
        return Err(Error::InvalidRequest {
            field: "frequency_dim",
            detail: "the inverse-frequency denominator cannot be zero".into(),
        });
    }
    // Half-split pairing reads `x[j + len/2]`, so an odd-width head has an
    // element with no partner. Interleaved pairing never reaches outside the
    // rotated prefix, which is why this is checked per layout.
    if layout == RopeLayout::HalfSplit && !x.len().is_multiple_of(2) {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "half-split rotation needs an even head dimension, got {}",
                x.len()
            ),
        });
    }
    if !(base.is_finite() && base > 1.0) {
        return Err(Error::InvalidRequest {
            field: "rope_base",
            detail: format!("base must be finite and > 1, got {base}"),
        });
    }
    let mut out = crate::try_clone_slice(x)?;
    let split = x.len() / 2;
    for j in 0..rotary_dim / 2 {
        let inv_freq = (base as f64).powf(-2.0 * j as f64 / frequency_dim as f64);
        let theta = pos as f64 * inv_freq;
        let (sin, cos) = (theta.sin() as f32, theta.cos() as f32);
        let (p, q) = match layout {
            RopeLayout::Interleaved => (2 * j, 2 * j + 1),
            RopeLayout::HalfSplit => (j, j + split),
        };
        let (a, b) = (x[p], x[q]);
        out[p] = a * cos - b * sin;
        out[q] = b * cos + a * sin;
    }
    Ok(out)
}

/// Rotate every head of one row. `x` is `heads · head_dim` long.
pub fn rope_row(
    x: &[f32],
    pos: u64,
    heads: usize,
    head_dim: usize,
    rotation: Rotation,
) -> Result<Vec<f32>> {
    if x.len() != heads * head_dim {
        return Err(Error::InvalidArtifact {
            detail: format!("row has {} elements, expected {heads}x{head_dim}", x.len()),
        });
    }
    let mut out = crate::try_vec(x.len())?;
    for h in 0..heads {
        let head = &x[h * head_dim..(h + 1) * head_dim];
        out.extend(rope_head(head, pos, rotation)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{ErrorSummary, gamma};

    /// The interleaved form with the rotated width as the denominator:
    /// exactly the signature and behaviour task 0003 pinned, so every fixture
    /// below still asserts the same fact after the layout and the
    /// inverse-frequency denominator became parameters.
    fn rope_interleaved(x: &[f32], pos: u64, base: f32, rotary_dim: usize) -> Result<Vec<f32>> {
        rope_head(x, pos, Rotation::interleaved(base, rotary_dim))
    }

    fn rope_row_interleaved(
        x: &[f32],
        pos: u64,
        base: f32,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Vec<f32>> {
        rope_row(
            x,
            pos,
            heads,
            head_dim,
            Rotation::interleaved(base, rotary_dim),
        )
    }

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
            let got = rope_interleaved(&x, pos, 10_000.0, d).unwrap();
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
        assert_eq!(rope_interleaved(&x, 0, 10_000.0, 16).unwrap(), x);
    }

    #[test]
    fn a_later_position_is_not_the_first_position() {
        // R21 as an assertion: if an implementation passed the index within a
        // chunk instead of the absolute position, this is what would silently
        // hold.
        let x: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let at_zero = rope_interleaved(&x, 0, 10_000.0, 8).unwrap();
        let at_five = rope_interleaved(&x, 5, 10_000.0, 8).unwrap();
        assert_ne!(at_zero, at_five);
    }

    #[test]
    fn rotation_preserves_the_norm_of_each_pair() {
        // The defining property: each 2-vector is rotated, not scaled.
        let x = [3.0f32, 4.0, -1.0, 2.0];
        for pos in [0u64, 1, 3, 1000] {
            let y = rope_interleaved(&x, pos, 10_000.0, 4).unwrap();
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
        let y = rope_interleaved(&x, 9, 10_000.0, 4).unwrap();
        assert_eq!(&y[4..], &x[4..]);
        assert_ne!(&y[..4], &x[..4]);
    }

    #[test]
    fn the_base_is_a_parameter_and_changes_the_rotation() {
        let x = [1.0f32, 0.0, 1.0, 0.0];
        let a = rope_interleaved(&x, 3, 10_000.0, 4).unwrap();
        let b = rope_interleaved(&x, 3, 1_000_000.0, 4).unwrap();
        assert_ne!(a, b, "a checkpoint-defined base is part of the function");
    }

    #[test]
    fn odd_or_oversized_rotary_dimensions_are_refused() {
        let x = [1.0f32; 8];
        assert!(rope_interleaved(&x, 1, 10_000.0, 3).is_err(), "odd");
        assert!(
            rope_interleaved(&x, 1, 10_000.0, 9).is_err(),
            "larger than the head"
        );
        assert!(rope_interleaved(&x, 1, 0.5, 4).is_err(), "base <= 1");
        assert!(rope_interleaved(&x, 1, f32::NAN, 4).is_err());
        assert!(
            rope_row_interleaved(&[1.0; 7], 1, 10_000.0, 2, 4, 4).is_err(),
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
        let row = rope_row_interleaved(&x, 6, 10_000.0, heads, head_dim, head_dim).unwrap();
        let one = rope_interleaved(&x[..head_dim], 6, 10_000.0, head_dim).unwrap();
        // Heads 0 and 1 hold the same values here, so their rotations agree.
        assert_eq!(&row[..head_dim], &one[..]);
        assert_eq!(&row[..head_dim], &row[head_dim..2 * head_dim]);
    }

    /// The half-split form, transcribed separately in FP64 with the full head
    /// dimension as the inverse-frequency denominator.
    fn rope_half_split_f64(x: &[f32], pos: u64, base: f32, rotary_dim: usize) -> Vec<f64> {
        let mut out: Vec<f64> = x.iter().map(|v| *v as f64).collect();
        let half = x.len() / 2;
        for j in 0..rotary_dim / 2 {
            let theta = pos as f64 * (base as f64).powf(-2.0 * j as f64 / x.len() as f64);
            let (sin, cos) = (theta.sin(), theta.cos());
            let (a, b) = (x[j] as f64, x[j + half] as f64);
            out[j] = a * cos - b * sin;
            out[j + half] = b * cos + a * sin;
        }
        out
    }

    #[test]
    fn half_split_matches_the_equation_within_gamma_four() {
        for (d, rotary) in [(64usize, 64usize), (64, 16), (8, 2), (32, 8)] {
            let x: Vec<f32> = (0..d)
                .map(|i| ((i * 11 % 41) as f32 - 20.0) / 7.0)
                .collect();
            for pos in [0u64, 1, 7, 4096, 262_143] {
                let got = rope_head(
                    &x,
                    pos,
                    Rotation {
                        base: 10_000.0,
                        rotary_dim: rotary,
                        frequency_dim: d,
                        layout: RopeLayout::HalfSplit,
                    },
                )
                .unwrap();
                let want = rope_half_split_f64(&x, pos, 10_000.0, rotary);
                let scale: Vec<f64> = (0..d)
                    .map(|_| x.iter().map(|v| v.abs() as f64).fold(0.0, f64::max))
                    .collect();
                let s = ErrorSummary::normalized(&got, &want, &scale);
                let bound = gamma(4);
                assert!(s.within(bound), "d={d} rotary={rotary} pos={pos}: {s}");
            }
        }
    }

    #[test]
    fn the_two_layouts_are_different_functions() {
        // R06 as an assertion. Both are legitimate; substituting one for the
        // other rotates different elements and produces plausible nonsense.
        let d = 16usize;
        let x: Vec<f32> = (0..d).map(|i| (i as f32) - 8.0).collect();
        let inter = rope_head(
            &x,
            3,
            Rotation {
                base: 10_000.0,
                rotary_dim: d,
                frequency_dim: d,
                layout: RopeLayout::Interleaved,
            },
        )
        .unwrap();
        let split = rope_head(
            &x,
            3,
            Rotation {
                base: 10_000.0,
                rotary_dim: d,
                frequency_dim: d,
                layout: RopeLayout::HalfSplit,
            },
        )
        .unwrap();
        assert_ne!(inter, split);
        // At position zero every angle is zero, so both are the identity and
        // the difference above is the rotation, not the pairing bookkeeping.
        assert_eq!(
            rope_head(
                &x,
                0,
                Rotation {
                    base: 10_000.0,
                    rotary_dim: d,
                    frequency_dim: d,
                    layout: RopeLayout::Interleaved
                }
            )
            .unwrap(),
            x
        );
        assert_eq!(
            rope_head(
                &x,
                0,
                Rotation {
                    base: 10_000.0,
                    rotary_dim: d,
                    frequency_dim: d,
                    layout: RopeLayout::HalfSplit
                }
            )
            .unwrap(),
            x
        );
    }

    #[test]
    fn the_frequency_denominator_is_not_the_rotated_width() {
        // Gemma 4's global layers rotate a quarter of the head and still
        // divide by the whole of it. Collapsing the two parameters changes
        // every angle but the first, which this pins.
        let d = 32usize;
        let rotary = 8usize;
        let x: Vec<f32> = (0..d).map(|i| ((i % 7) as f32 - 3.0) / 2.0).collect();
        let full = rope_head(
            &x,
            11,
            Rotation {
                base: 1e6,
                rotary_dim: rotary,
                frequency_dim: d,
                layout: RopeLayout::HalfSplit,
            },
        )
        .unwrap();
        let collapsed = rope_head(
            &x,
            11,
            Rotation {
                base: 1e6,
                rotary_dim: rotary,
                frequency_dim: rotary,
                layout: RopeLayout::HalfSplit,
            },
        )
        .unwrap();
        assert_ne!(full, collapsed);
        // j = 0 has exponent zero either way, so that pair must agree exactly
        // -- the divergence is in the later angles, not in a global shift.
        assert_eq!(full[0], collapsed[0]);
        assert_eq!(full[d / 2], collapsed[d / 2]);
        assert_ne!(full[1], collapsed[1]);
    }

    #[test]
    fn unrotated_elements_pass_through_untouched() {
        let d = 16usize;
        let rotary = 4usize;
        let x: Vec<f32> = (0..d).map(|i| (i as f32) + 0.5).collect();
        let y = rope_head(
            &x,
            5,
            Rotation {
                base: 10_000.0,
                rotary_dim: rotary,
                frequency_dim: d,
                layout: RopeLayout::HalfSplit,
            },
        )
        .unwrap();
        // Half-split touches j in 0..2 and j+8 in 8..10.
        for i in 0..d {
            let touched = i < rotary / 2 || (d / 2..d / 2 + rotary / 2).contains(&i);
            if !touched {
                assert_eq!(y[i], x[i], "element {i} moved and should not have");
            }
        }
        let z = rope_head(
            &x,
            5,
            Rotation {
                base: 10_000.0,
                rotary_dim: rotary,
                frequency_dim: d,
                layout: RopeLayout::Interleaved,
            },
        )
        .unwrap();
        for i in rotary..d {
            assert_eq!(z[i], x[i], "element {i} moved and should not have");
        }
    }

    #[test]
    fn half_split_refuses_an_odd_head_and_a_zero_denominator() {
        let x = [1.0f32, 2.0, 3.0];
        assert!(
            rope_head(
                &x,
                1,
                Rotation {
                    base: 10_000.0,
                    rotary_dim: 2,
                    frequency_dim: 3,
                    layout: RopeLayout::HalfSplit
                }
            )
            .is_err()
        );
        // Interleaved does not read across the midpoint, so an odd head is
        // fine there; the check belongs to the layout, not to the operation.
        assert!(
            rope_head(
                &x,
                1,
                Rotation {
                    base: 10_000.0,
                    rotary_dim: 2,
                    frequency_dim: 3,
                    layout: RopeLayout::Interleaved
                }
            )
            .is_ok()
        );
        assert!(
            rope_head(
                &x,
                1,
                Rotation {
                    base: 10_000.0,
                    rotary_dim: 2,
                    frequency_dim: 0,
                    layout: RopeLayout::Interleaved
                }
            )
            .is_err()
        );
    }

    #[test]
    fn a_rotation_preserves_each_pair_norm() {
        // A rotation is orthogonal on each pair whichever elements it pairs.
        let d = 16usize;
        let x: Vec<f32> = (0..d).map(|i| (i * 5 % 9) as f32 - 4.0).collect();
        for layout in [RopeLayout::Interleaved, RopeLayout::HalfSplit] {
            let y = rope_head(
                &x,
                17,
                Rotation {
                    base: 10_000.0,
                    rotary_dim: d,
                    frequency_dim: d,
                    layout,
                },
            )
            .unwrap();
            let before: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
            let after: f64 = y.iter().map(|v| (*v as f64).powi(2)).sum();
            assert!(
                (after - before).abs() <= before * 1e-5,
                "{layout:?}: {before} -> {after}"
            );
        }
    }
}
