//! Quantizer policy: turning float weights into affine integer codes.
//!
//! Kept deliberately separate from [`crate::affine`], because the M0 review's
//! F-list turns on the distinction. The old `int8.rs` folded a quantizer's
//! choices into the *decoder*: it rejected code `-128` outright, on the grounds
//! that a symmetric range must be symmetric. That is a correct statement about
//! one quantizer and a wrong statement about the format. Document 03:
//!
//! > A quantizer may choose to emit a symmetric subset such as [-127,127]; that
//! > does not authorize a generic INT8 decoder to reject -128 from a valid
//! > source artifact.
//!
//! So the clipping range lives here, as a declared parameter of a named
//! quantizer, and the decoder accepts every representable code.
//!
//! Scope: this is the tie-breaking and clipping contract, used to build test
//! fixtures and, later, to quantize from a high-precision original under a
//! pinned calibration profile. Document 03 is explicit that "for existing usable
//! INT4/INT8 artifacts prefer repacking over another quantization pass", and
//! that any real quantization is a precision conversion needing O2 evidence.

use moxie_types::{Error, Result};

use crate::affine::IntWidth;

/// Round half to even. The pinned rounding rule (document 03).
///
/// `f32::round` rounds half *away from zero*, which biases a tensor's magnitude
/// upward at exactly the values a uniform quantizer hits most often.
pub fn round_ties_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// A quantizer with an explicitly declared code range.
///
/// `clip` is the quantizer's choice, not the format's. `Symmetric` for INT8
/// clips to `[-127, 127]` so that saturation is symmetric about zero; a source
/// artifact quantized by someone else may still contain `-128`, and the decoder
/// reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantizer {
    width: IntWidth,
    lo: i32,
    hi: i32,
}

impl Quantizer {
    /// Symmetric clipping: the range is mirrored about zero, so the most
    /// negative representable code is excluded.
    ///
    /// The bias this avoids: with `[-128, 127]` and no zero point, every
    /// saturated weight in a group can only saturate negative, and the group's
    /// mean shifts.
    pub const fn symmetric(width: IntWidth) -> Self {
        // The format's most negative code (-8 / -128) is deliberately dropped:
        // it has no positive counterpart. `full_range` keeps it.
        let (_representable_low, hi) = width.code_range();
        Self { width, lo: -hi, hi }
    }

    /// The full representable range, for a quantizer that wants it and has
    /// accounted for the asymmetry (typically because it also emits a zero
    /// point).
    pub const fn full_range(width: IntWidth) -> Self {
        let (lo, hi) = width.code_range();
        Self { width, lo, hi }
    }

    pub const fn range(self) -> (i32, i32) {
        (self.lo, self.hi)
    }

    pub const fn width(self) -> IntWidth {
        self.width
    }

    /// Quantize one value against a group scale and zero point.
    ///
    /// `q = clip(round_ties_even(value / scale) + zero_point)`.
    pub fn quantize(self, value: f32, scale: f32, zero_point: i32) -> Result<i32> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::InvalidArtifact {
                detail: format!("scale must be positive and finite, got {scale}"),
            });
        }
        if !value.is_finite() {
            return Err(Error::InvalidArtifact {
                detail: format!("cannot quantize non-finite value {value}"),
            });
        }
        let q = round_ties_even(value / scale);
        // Bound the quotient in f32 before casting: a tiny scale makes it exceed
        // i32, and `as i32` on an out-of-range float saturates without saying so.
        // The bound is the integer range, **not** the code range: an asymmetric
        // quantizer's quotient legitimately sits outside the codes, and the zero
        // point is what brings it back. Clipping before the offset collapses a
        // whole positive-valued group onto the lowest code.
        const BOUND: f32 = (i32::MAX / 2) as f32;
        let clamped = q.clamp(-BOUND, BOUND) as i32;
        Ok(clamped.saturating_add(zero_point).clamp(self.lo, self.hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::{AffineDescriptor, AffineTensor, Grouping, IntWidth, ZeroPoints, pack_row};
    use crate::scale::{ScaleDtype, ScaleValues};

    #[test]
    fn round_trip_is_exact_on_grid_points() {
        // Preserved from the INT8 v1 tests: every code the symmetric quantizer
        // can emit survives dequantize -> quantize.
        let q = Quantizer::symmetric(IntWidth::Int8);
        for code in -127i32..=127 {
            let v = code as f32 * 0.5;
            assert_eq!(q.quantize(v, 0.5, 0).unwrap(), code);
        }
    }

    #[test]
    fn saturation_is_symmetric() {
        let q8 = Quantizer::symmetric(IntWidth::Int8);
        assert_eq!(q8.quantize(1e9, 1.0, 0).unwrap(), 127);
        assert_eq!(q8.quantize(-1e9, 1.0, 0).unwrap(), -127);
        assert_eq!(q8.range(), (-127, 127));

        let q4 = Quantizer::symmetric(IntWidth::Int4);
        assert_eq!(q4.quantize(1e9, 1.0, 0).unwrap(), 7);
        assert_eq!(q4.quantize(-1e9, 1.0, 0).unwrap(), -7);
        assert_eq!(q4.range(), (-7, 7));
    }

    #[test]
    fn ties_go_to_even() {
        let q = Quantizer::symmetric(IntWidth::Int8);
        assert_eq!(q.quantize(0.5, 1.0, 0).unwrap(), 0);
        assert_eq!(q.quantize(1.5, 1.0, 0).unwrap(), 2);
        assert_eq!(q.quantize(2.5, 1.0, 0).unwrap(), 2);
        assert_eq!(q.quantize(-0.5, 1.0, 0).unwrap(), 0);
        assert_eq!(q.quantize(-1.5, 1.0, 0).unwrap(), -2);
        assert_eq!(q.quantize(-2.5, 1.0, 0).unwrap(), -2);
    }

    #[test]
    fn bad_scales_and_values_are_rejected() {
        let q = Quantizer::symmetric(IntWidth::Int8);
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(q.quantize(1.0, bad, 0).is_err(), "scale {bad}");
        }
        assert!(q.quantize(f32::NAN, 1.0, 0).is_err());
        assert!(q.quantize(f32::INFINITY, 1.0, 0).is_err());
    }

    #[test]
    fn a_tiny_scale_saturates_instead_of_wrapping() {
        // `value / scale` overflows i32 long before it overflows f32. A bare
        // `as i32` would give a saturated but unclamped intermediate on some
        // paths; the range must hold regardless.
        let q = Quantizer::symmetric(IntWidth::Int8);
        assert_eq!(q.quantize(1.0, f32::MIN_POSITIVE, 0).unwrap(), 127);
        assert_eq!(q.quantize(-1.0, f32::MIN_POSITIVE, 0).unwrap(), -127);
    }

    #[test]
    fn the_quantizers_clipping_choice_does_not_constrain_the_decoder() {
        // The heart of the correction. The symmetric quantizer never emits
        // -128; the decoder must still read it from a source that does.
        let q = Quantizer::symmetric(IntWidth::Int8);
        assert_ne!(q.quantize(-1e9, 1.0, 0).unwrap(), -128);
        assert_eq!(q.range().0, -127);

        let t = AffineTensor::new(
            AffineDescriptor {
                width: IntWidth::Int8,
                out_features: 1,
                in_features: 2,
                grouping: Grouping::PerOutputChannel,
                group_index: None,
                scale_dtype: ScaleDtype::F32,
            },
            pack_row(IntWidth::Int8, &[-128, 127]).unwrap(),
            ScaleValues::F32(vec![1.0]),
            ZeroPoints::Symmetric,
        )
        .unwrap();
        assert_eq!(t.reconstruct().unwrap(), vec![-128.0, 127.0]);

        // And a quantizer that has accounted for the asymmetry may use it.
        let full = Quantizer::full_range(IntWidth::Int8);
        assert_eq!(full.quantize(-1e9, 1.0, 0).unwrap(), -128);
        assert_eq!(full.range(), (-128, 127));
    }

    #[test]
    fn an_asymmetric_quantizer_places_the_zero_point_and_round_trips() {
        // A group of positive-only values: an asymmetric quantizer with a zero
        // point uses the whole code range where a symmetric one would waste
        // half of it. This is why zero points are preserved rather than removed
        // (document 03: "Do not requantize a usable AWQ artifact merely to
        // remove its zero points").
        let values: Vec<f32> = (0..32).map(|k| 10.0 + k as f32 * 0.5).collect();
        let (min, max) = (values[0], values[31]);
        let (lo, hi) = IntWidth::Int4.code_range();
        let scale = (max - min) / (hi - lo) as f32;
        let zero = lo - round_ties_even(min / scale) as i32;

        let q = Quantizer::full_range(IntWidth::Int4);
        let codes: Vec<i32> = values
            .iter()
            .map(|v| q.quantize(*v, scale, zero).unwrap())
            .collect();
        assert!(codes.iter().all(|c| (lo..=hi).contains(c)));
        assert_eq!(*codes.first().unwrap(), lo);
        assert_eq!(*codes.last().unwrap(), hi);

        let t = AffineTensor::new(
            AffineDescriptor {
                width: IntWidth::Int4,
                out_features: 1,
                in_features: 32,
                grouping: Grouping::Contiguous { size: 32 },
                group_index: None,
                scale_dtype: ScaleDtype::F32,
            },
            pack_row(IntWidth::Int4, &codes).unwrap(),
            ScaleValues::F32(vec![scale]),
            ZeroPoints::PerGroup(vec![zero as i16]),
        )
        .unwrap();
        let back = t.reconstruct().unwrap();
        for (v, w) in values.iter().zip(back.iter()) {
            assert!((v - w).abs() <= scale, "{v} vs {w}, scale {scale}");
        }
    }

    #[test]
    fn round_ties_even_matches_the_definition_over_a_dense_grid() {
        for i in -400i32..=400 {
            let x = i as f32 / 4.0; // hits .0, .25, .5, .75
            let r = round_ties_even(x);
            assert_eq!(r, r.trunc(), "{x} did not round to an integer");
            assert!((r - x).abs() <= 0.5);
            if (x.fract()).abs() == 0.5 {
                assert_eq!(r % 2.0, 0.0, "{x} is a tie and must land on an even value");
            }
        }
    }
}
