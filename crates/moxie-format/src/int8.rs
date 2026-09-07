//! INT8 v1: signed symmetric, weight-only.
//!
//! Document 03 pins the initial canonical profile: "signed symmetric weight-only
//! INT8, per-output-channel positive BF16 scale, no zero point;
//! `W[o,i] = int8(code[o,i]) * scale[o]`. Define code range `[-127,127]`; reject
//! `-128` for this profile."
//!
//! Rejecting -128 is deliberate. A symmetric range must be symmetric: allowing
//! -128 without a matching +128 biases every saturated weight negative.

use moxie_types::{Error, Result};

/// Quantize with ties-to-even and saturation, per document 03.
pub fn quantize(value: f32, scale: f32) -> Result<i8> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::InvalidArtifact {
            detail: format!("INT8 scale must be positive and finite, got {scale}"),
        });
    }
    if !value.is_finite() {
        return Err(Error::InvalidArtifact {
            detail: format!("cannot quantize non-finite value {value}"),
        });
    }
    let q = round_ties_even(value / scale);
    Ok(q.clamp(-127.0, 127.0) as i8)
}

pub fn dequantize(code: i8, scale: f32) -> Result<f32> {
    if code == -128 {
        return Err(Error::InvalidArtifact {
            detail: "code -128 is outside the symmetric [-127, 127] range of INT8 v1".into(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::InvalidArtifact {
            detail: format!("INT8 scale must be positive and finite, got {scale}"),
        });
    }
    Ok(code as f32 * scale)
}

fn round_ties_even(x: f32) -> f32 {
    let r = x.round(); // rounds half away from zero
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_exact_on_grid_points() {
        for code in -127i8..=127 {
            let v = dequantize(code, 0.5).unwrap();
            assert_eq!(quantize(v, 0.5).unwrap(), code);
        }
    }

    #[test]
    fn minus_128_is_rejected_not_silently_accepted() {
        // The asymmetry this prevents: -128 * scale has no positive counterpart.
        assert!(dequantize(-128, 1.0).is_err());
        assert!(dequantize(-127, 1.0).is_ok());
    }

    #[test]
    fn saturation_is_symmetric() {
        assert_eq!(quantize(1e9, 1.0).unwrap(), 127);
        assert_eq!(quantize(-1e9, 1.0).unwrap(), -127);
    }

    #[test]
    fn ties_go_to_even() {
        assert_eq!(quantize(0.5, 1.0).unwrap(), 0);
        assert_eq!(quantize(1.5, 1.0).unwrap(), 2);
        assert_eq!(quantize(2.5, 1.0).unwrap(), 2);
        assert_eq!(quantize(-0.5, 1.0).unwrap(), 0);
        assert_eq!(quantize(-1.5, 1.0).unwrap(), -2);
        assert_eq!(quantize(-2.5, 1.0).unwrap(), -2);
    }

    #[test]
    fn bad_scales_and_values_are_rejected() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(quantize(1.0, bad).is_err());
            assert!(dequantize(1, bad).is_err());
        }
        assert!(quantize(f32::NAN, 1.0).is_err());
        assert!(quantize(f32::INFINITY, 1.0).is_err());
    }
}
