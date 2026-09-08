//! Scale scalars, in the encoding the source actually used.
//!
//! Document 03's affine descriptor: "Preserve source numerical values and source
//! scalar encoding: a closed FP16/BF16/FP32 scale-dtype tag per tensor ... Read
//! dtype from tensor headers, not the model's general dtype field."
//!
//! Three encodings is not format proliferation. Rounding an FP16 scale to BF16
//! loses seven mantissa bits on a value that multiplies every weight in its
//! group, and doing it silently would turn a value-preserving repack into a
//! quantization change -- which document 03 classifies separately and which
//! needs O2 evidence. The candidate evidence is explicit that `scale_dtype` is
//! *null* in the inspected configurations, so the dtype comes from the tensor
//! header and this tag records what was found there.

use moxie_types::{Error, Result};

use crate::bf16::bf16_bits_to_f32;

/// The closed set of scale scalar encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleDtype {
    F16,
    Bf16,
    F32,
}

impl ScaleDtype {
    pub const fn name(self) -> &'static str {
        match self {
            ScaleDtype::F16 => "f16",
            ScaleDtype::Bf16 => "bf16",
            ScaleDtype::F32 => "f32",
        }
    }

    pub const fn bytes(self) -> usize {
        match self {
            ScaleDtype::F16 | ScaleDtype::Bf16 => 2,
            ScaleDtype::F32 => 4,
        }
    }

    pub const ALL: &'static [ScaleDtype] = &[ScaleDtype::F16, ScaleDtype::Bf16, ScaleDtype::F32];
}

/// Exact IEEE-754 binary16 -> f32.
///
/// Every f16 is representable in f32, so this is lossless in both the value and
/// the sign of zero. Subnormals are handled explicitly rather than by a
/// bit-shift shortcut, which is where hand-rolled versions usually go wrong.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    let out = match exp {
        // Zero or subnormal.
        0 => {
            if mant == 0 {
                sign << 31
            } else {
                // Normalise: shift the mantissa up until the implicit bit is
                // set, decrementing the exponent once per shift. Starting the
                // counter at -1 instead of 0 halves every subnormal scale.
                let mut e: i32 = 0;
                let mut m = mant;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03FF;
                let f32_exp = (e + 127 - 14) as u32;
                (sign << 31) | (f32_exp << 23) | (m << 13)
            }
        }
        // Infinity or NaN.
        0x1F => (sign << 31) | 0x7F80_0000 | (mant << 13),
        // Normal: rebias 15 -> 127.
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

/// The scale values of one tensor, kept in their source encoding.
///
/// The raw payload is retained rather than a decoded `Vec<f32>` so that a
/// repacker can write the same bytes back out and prove nothing changed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleValues {
    F16(Vec<u16>),
    Bf16(Vec<u16>),
    F32(Vec<f32>),
}

impl ScaleValues {
    pub fn dtype(&self) -> ScaleDtype {
        match self {
            ScaleValues::F16(_) => ScaleDtype::F16,
            ScaleValues::Bf16(_) => ScaleDtype::Bf16,
            ScaleValues::F32(_) => ScaleDtype::F32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ScaleValues::F16(v) | ScaleValues::Bf16(v) => v.len(),
            ScaleValues::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decode scale `i` to f32. The reconstruction oracle evaluates the scale
    /// multiplication in FP32 (document 03), so decoding here is exact for all
    /// three encodings.
    pub fn get(&self, i: usize) -> Option<f32> {
        match self {
            ScaleValues::F16(v) => v.get(i).copied().map(f16_bits_to_f32),
            ScaleValues::Bf16(v) => v.get(i).copied().map(bf16_bits_to_f32),
            ScaleValues::F32(v) => v.get(i).copied(),
        }
    }

    /// Reject anything a kernel must never see.
    ///
    /// Document 03: "Positive finite scales; reject nonfinite/zero/negative
    /// values unless an explicit source zero-block convention is normalized
    /// losslessly to valid canonical values." No such convention is pinned for
    /// an integer source, so zero is rejected here: an all-zero group is
    /// expressible with zero *codes*, and a zero scale would instead erase a
    /// group silently while looking like a valid tensor.
    pub fn validate(&self) -> Result<()> {
        for i in 0..self.len() {
            let v = self.get(i).expect("index below len");
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "scale[{i}] is {v} ({}); scales must be positive and finite",
                        self.dtype().name()
                    ),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::f32_to_bf16_bits;

    #[test]
    fn f16_landmark_values_decode_exactly() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert!(f16_bits_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0x3800), 0.5);
        // Largest finite f16.
        assert_eq!(f16_bits_to_f32(0x7BFF), 65504.0);
        // Smallest positive normal and smallest positive subnormal.
        assert_eq!(f16_bits_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_bits_to_f32(0x7C00).is_infinite());
        assert!(f16_bits_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn every_f16_subnormal_decodes_to_the_defined_value() {
        // Subnormals are 2^-24 * mantissa. Getting the normalisation loop wrong
        // is a silent factor-of-two error on the smallest scales in a tensor.
        for m in 1u16..0x0400 {
            assert_eq!(
                f16_bits_to_f32(m),
                2f32.powi(-24) * m as f32,
                "subnormal 0x{m:04x}"
            );
            assert_eq!(f16_bits_to_f32(m | 0x8000), -(2f32.powi(-24) * m as f32));
        }
    }

    #[test]
    fn every_f16_pattern_decodes_to_a_value_derived_from_the_format() {
        // Derived from the definition rather than a table, so the decoder cannot
        // validate itself.
        for bits in 0u16..=u16::MAX {
            let got = f16_bits_to_f32(bits);
            let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let exp = ((bits >> 10) & 0x1F) as i32;
            let mant = (bits & 0x03FF) as f32;
            if exp == 0x1F {
                assert!(got.is_infinite() || got.is_nan(), "0x{bits:04x}");
                continue;
            }
            let want = if exp == 0 {
                sign * 2f32.powi(-14) * (mant / 1024.0)
            } else {
                sign * 2f32.powi(exp - 15) * (1.0 + mant / 1024.0)
            };
            assert_eq!(got, want, "0x{bits:04x}");
        }
    }

    #[test]
    fn the_three_encodings_are_preserved_not_normalised() {
        // The failure this guards: an FP16 scale rounded to BF16 during import.
        // 1.0009765625 is exactly representable in FP16 and not in BF16, so the
        // two decodings differ -- silently changing every weight in that group.
        let f16_only = f16_bits_to_f32(0x3C01);
        assert_eq!(f16_only, 1.0 + 1.0 / 1024.0);
        let through_bf16 = bf16_bits_to_f32(f32_to_bf16_bits(f16_only));
        assert_ne!(
            f16_only, through_bf16,
            "rounding an FP16 scale to BF16 is a value change, not a re-encoding"
        );

        let s = ScaleValues::F16(vec![0x3C01]);
        assert_eq!(s.dtype(), ScaleDtype::F16);
        assert_eq!(s.get(0), Some(f16_only));
    }

    #[test]
    fn nonfinite_zero_and_negative_scales_are_rejected() {
        assert!(ScaleValues::F32(vec![1.0, 2.0]).validate().is_ok());
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                ScaleValues::F32(vec![1.0, bad]).validate().is_err(),
                "{bad} should be rejected"
            );
        }
        assert!(ScaleValues::F16(vec![0x0000]).validate().is_err()); // +0
        assert!(ScaleValues::F16(vec![0x7C00]).validate().is_err()); // inf
        assert!(ScaleValues::F16(vec![0xBC00]).validate().is_err()); // -1
        assert!(ScaleValues::Bf16(vec![0x0000]).validate().is_err());
        assert!(ScaleValues::Bf16(vec![0x3F80]).validate().is_ok()); // 1.0
    }

    #[test]
    fn scale_dtype_sizes_match_their_encodings() {
        assert_eq!(ScaleDtype::F16.bytes(), 2);
        assert_eq!(ScaleDtype::Bf16.bytes(), 2);
        assert_eq!(ScaleDtype::F32.bytes(), 4);
        let mut names: Vec<_> = ScaleDtype::ALL.iter().map(|d| d.name()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(n, names.len());
    }
}
