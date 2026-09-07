//! BF16 conversion, as a pinned numerical contract.
//!
//! Document 03: "BF16 little-endian weights with specified round-to-nearest-even
//! conversion from FP32."
//!
//! Truncation is the common shortcut and it is wrong: it biases every weight
//! toward zero. The GPU lane compares this host implementation against the
//! device's `__float2bfloat16` bit for bit.

/// Round-to-nearest-even f32 -> bf16, returning the raw 16-bit pattern.
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        // Preserve the sign and force the quiet bit, as the hardware does.
        // Propagating a payload that happens to truncate to zero would turn a
        // NaN into an infinity.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round to nearest, ties to even: add half an ulp, plus one more when the
    // retained low bit is already 1, so exact halves round to the even value.
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
    (rounded >> 16) as u16
}

/// Exact bf16 -> f32. Every bf16 is representable in f32, so this is lossless.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Truncating conversion, provided *only* as the thing to compare against.
///
/// Never use it to build an artifact. It exists so a test can demonstrate that
/// the two differ, and so a reviewer can see the difference is real.
pub fn f32_to_bf16_bits_truncating(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values_round_trip() {
        for v in [0.0f32, -0.0, 1.0, -1.0, 2.0, 0.5, -256.0] {
            let b = f32_to_bf16_bits(v);
            assert_eq!(bf16_bits_to_f32(b), v, "{v} did not round-trip");
        }
    }

    #[test]
    fn ties_go_to_even() {
        // 0x3F80_8000 is exactly halfway between bf16 0x3F80 (1.0) and 0x3F81.
        // Ties-to-even must pick 0x3F80, whose low bit is 0.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 0x3F81_8000 is halfway between 0x3F81 and 0x3F82; the even one is
        // 0x3F82. A "round half up" implementation agrees here but not above,
        // and a truncating one disagrees with both.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
    }

    #[test]
    fn just_below_a_tie_rounds_down() {
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_7FFF)), 0x3F80);
    }

    #[test]
    fn just_above_a_tie_rounds_up() {
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
    }

    #[test]
    fn rounding_differs_from_truncation_and_is_not_biased_to_zero() {
        // The reason the contract is pinned. Truncation always moves toward
        // zero; over a whole tensor that is a systematic magnitude loss.
        let v = f32::from_bits(0x3F80_C000); // clearly above the tie
        assert_eq!(f32_to_bf16_bits(v), 0x3F81);
        assert_eq!(f32_to_bf16_bits_truncating(v), 0x3F80);

        let mut differ = 0;
        let mut truncation_lost_magnitude = 0;
        for i in 0..2000u32 {
            let x = f32::from_bits(0x3F80_0000 + i * 37);
            let r = bf16_bits_to_f32(f32_to_bf16_bits(x));
            let t = bf16_bits_to_f32(f32_to_bf16_bits_truncating(x));
            if r != t {
                differ += 1;
                assert!(t.abs() <= r.abs());
                truncation_lost_magnitude += 1;
            }
        }
        assert!(differ > 100, "expected many disagreements, saw {differ}");
        assert_eq!(differ, truncation_lost_magnitude);
    }

    #[test]
    fn infinities_survive() {
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xFF80);
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::INFINITY)).is_infinite());
    }

    #[test]
    fn nan_stays_nan_and_does_not_become_infinity() {
        // The failure this guards: a NaN whose mantissa lives entirely in the
        // discarded low 16 bits truncates to the infinity pattern.
        let sneaky = f32::from_bits(0x7F80_0001);
        assert!(sneaky.is_nan());
        assert_eq!(f32_to_bf16_bits_truncating(sneaky), 0x7F80); // infinity!
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(sneaky)).is_nan());

        let neg = f32::from_bits(0xFF80_0001);
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(neg)).is_nan());
    }

    #[test]
    fn rounding_up_at_the_top_of_the_range_reaches_infinity_not_a_wrap() {
        // f32::MAX rounds up past the largest bf16. The result must be infinity,
        // never a wrapped small number. `wrapping_add` in the implementation is
        // what makes this land on the infinity pattern rather than panic.
        let b = f32_to_bf16_bits(f32::MAX);
        assert_eq!(b, 0x7F80);
        assert!(bf16_bits_to_f32(b).is_infinite());
    }

    #[test]
    fn subnormal_and_tiny_values_do_not_panic() {
        for v in [f32::MIN_POSITIVE, 1e-38, -1e-38, f32::from_bits(1)] {
            let _ = f32_to_bf16_bits(v);
        }
    }

    #[test]
    fn every_bf16_pattern_round_trips_through_f32() {
        // Exhaustive over the whole 16-bit space: bf16 -> f32 -> bf16 is the
        // identity for every non-NaN pattern.
        for bits in 0u16..=u16::MAX {
            let f = bf16_bits_to_f32(bits);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_bf16_bits(f), bits, "pattern 0x{bits:04x}");
        }
    }
}
