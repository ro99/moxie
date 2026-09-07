//! NVFP4 v1: the canonical low-bit weight family.
//!
//! Document 03 pins the reconstruction unambiguously:
//!
//! ```text
//! W[row, col] = E2M1(code[row, col])
//!             * decode_E4M3FN(block_scale[row, floor(col / 16)])
//!             * tensor_scale_F32
//! ```
//!
//! "Groups are 16 consecutive logical input-channel values within one output
//! row. Low nibble is the earlier logical element, high nibble the next."
//!
//! The decode tables are exhaustive and pinned. Document 07 requires "exhaustive
//! low-bit decode tables" as artifact-layer evidence, because a single wrong
//! entry is a silent accuracy loss that no kernel tolerance will catch.

use moxie_types::{Error, Result};

/// Logical values per group. Fixed by the format, not a tuning parameter.
pub const GROUP_SIZE: usize = 16;

/// E2M1 decode: the complete 4-bit code space.
///
/// Sign in bit 3, 2-bit exponent (bias 1), 1-bit mantissa. Code 0b0000 is +0.0
/// and 0b1000 is -0.0; both decode to zero but are distinct encodings.
pub const E2M1_TABLE: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, // sign 0
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, // sign 1
];

/// Decode one E2M1 code. Only the low four bits are meaningful.
pub fn decode_e2m1(code: u8) -> f32 {
    E2M1_TABLE[(code & 0x0F) as usize]
}

/// Decode an E4M3FN block scale.
///
/// "FN" is finite-and-NaN: there are no infinities, and `0x7F`/`0xFF` are NaN.
/// Exponent bias is 7; exponent 0 is subnormal.
///
/// Returns `None` for the two NaN encodings. Document 03: "Reject negative
/// scales and reserved/NaN encodings rather than letting them reach a kernel."
pub fn decode_e4m3fn(code: u8) -> Option<f32> {
    if code == 0x7F || code == 0xFF {
        return None; // NaN
    }
    let sign = if code & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((code >> 3) & 0x0F) as i32;
    let mant = (code & 0x07) as f32;
    let v = if exp == 0 {
        // Subnormal: 2^-6 * (m / 8)
        2f32.powi(-6) * (mant / 8.0)
    } else {
        2f32.powi(exp - 7) * (1.0 + mant / 8.0)
    };
    Some(sign * v)
}

/// How a source checkpoint applies its global scale.
///
/// R16: "GLM's documented NVFP4 decoding divides by a global scale; Inkling's
/// ModelOpt path multiplies and has an interleaved expert layout." Both
/// conventions are present in the checkpoints on this machine, in the same
/// architecture family, so this is not hypothetical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalScaleConvention {
    /// ModelOpt: the stored value is already the multiplier.
    Multiply,
    /// compressed-tensors: the stored value is a divisor.
    Divide,
}

/// Normalise a source global scale into the canonical multiplier.
///
/// Document 03: "Normalize division into the canonical multiplier during
/// import." After this, exactly one equation runs at execution time.
pub fn canonical_tensor_scale(stored: f32, convention: GlobalScaleConvention) -> Result<f32> {
    if !stored.is_finite() {
        return Err(Error::InvalidArtifact {
            detail: format!("global scale is not finite: {stored}"),
        });
    }
    // Document 03: "Manifest records positive finite global scale."
    if stored <= 0.0 {
        return Err(Error::InvalidArtifact {
            detail: format!("global scale must be positive and finite, got {stored}"),
        });
    }
    Ok(match convention {
        GlobalScaleConvention::Multiply => stored,
        GlobalScaleConvention::Divide => 1.0 / stored,
    })
}

/// Validate a block scale before it can reach a kernel.
///
/// Document 03: "Block scales are finite and nonnegative; zero represents an
/// all-zero block."
pub fn validate_block_scale(code: u8) -> Result<f32> {
    match decode_e4m3fn(code) {
        None => Err(Error::InvalidArtifact {
            detail: format!("block scale 0x{code:02x} is a NaN encoding"),
        }),
        Some(v) if v < 0.0 => Err(Error::InvalidArtifact {
            detail: format!("block scale 0x{code:02x} decodes to {v}, which is negative"),
        }),
        Some(v) => Ok(v),
    }
}

/// Unpack one row of packed codes into logical values.
///
/// `packed` holds two codes per byte; the **low nibble is the earlier logical
/// element**. `block_scales` holds one E4M3FN code per group of 16.
///
/// `logical_len` is the row's logical width; document 03 requires incomplete
/// groups to be zero-padded *outside* the logical shape, so trailing padding is
/// dropped here rather than surfacing as extra values.
pub fn dequantize_row(
    packed: &[u8],
    block_scales: &[u8],
    tensor_scale: f32,
    logical_len: usize,
) -> Result<Vec<f32>> {
    let groups = logical_len.div_ceil(GROUP_SIZE);
    if block_scales.len() < groups {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "row needs {groups} block scales for {logical_len} values, got {}",
                block_scales.len()
            ),
        });
    }
    let needed_bytes = (groups * GROUP_SIZE).div_ceil(2);
    if packed.len() < needed_bytes {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "row needs {needed_bytes} packed bytes for {logical_len} values, got {}",
                packed.len()
            ),
        });
    }
    if !tensor_scale.is_finite() || tensor_scale <= 0.0 {
        return Err(Error::InvalidArtifact {
            detail: format!("tensor scale must be positive and finite, got {tensor_scale}"),
        });
    }

    let mut out = Vec::with_capacity(logical_len);
    for i in 0..logical_len {
        let byte = packed[i / 2];
        // Low nibble first: element 0 is the low nibble of byte 0.
        let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let scale = validate_block_scale(block_scales[i / GROUP_SIZE])?;
        out.push(decode_e2m1(code) * scale * tensor_scale);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_table_is_exhaustive_and_correct() {
        // Derived from the format definition rather than copied from the table,
        // so the table cannot validate itself.
        for code in 0u8..16 {
            let sign = if code & 0x8 != 0 { -1.0f32 } else { 1.0 };
            let exp = ((code >> 1) & 0x3) as i32;
            let mant = (code & 0x1) as f32;
            let want = if exp == 0 {
                sign * 2f32.powi(-1) * mant // subnormal: 0 or 0.5
            } else {
                sign * 2f32.powi(exp - 1) * (1.0 + mant / 2.0)
            };
            assert_eq!(decode_e2m1(code), want, "code {code:#06b}");
        }
    }

    #[test]
    fn e2m1_spans_the_documented_magnitudes() {
        let mags: Vec<f32> = (0u8..8).map(decode_e2m1).collect();
        assert_eq!(mags, vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        // Negative zero is a distinct encoding that decodes to zero.
        assert_eq!(decode_e2m1(0b1000), 0.0);
        assert!(decode_e2m1(0b1000).is_sign_negative());
    }

    #[test]
    fn e2m1_ignores_high_bits() {
        for code in 0u8..16 {
            assert_eq!(decode_e2m1(code), decode_e2m1(code | 0xF0));
        }
    }

    #[test]
    fn e4m3fn_decodes_every_code_and_only_two_are_nan() {
        let mut nans = 0;
        for code in 0u8..=255 {
            match decode_e4m3fn(code) {
                None => nans += 1,
                Some(v) => assert!(v.is_finite(), "0x{code:02x} decoded to {v}"),
            }
        }
        // FN: no infinities, exactly two NaN encodings.
        assert_eq!(nans, 2);
        assert!(decode_e4m3fn(0x7F).is_none());
        assert!(decode_e4m3fn(0xFF).is_none());
    }

    #[test]
    fn e4m3fn_landmark_values() {
        assert_eq!(decode_e4m3fn(0x00), Some(0.0));
        assert_eq!(decode_e4m3fn(0x38), Some(1.0)); // exp 7, mant 0
        assert_eq!(decode_e4m3fn(0xB8), Some(-1.0));
        assert_eq!(decode_e4m3fn(0x7E), Some(448.0)); // largest finite
        assert_eq!(decode_e4m3fn(0x01), Some(2f32.powi(-9))); // smallest subnormal
    }

    #[test]
    fn nan_and_negative_block_scales_are_rejected_before_a_kernel_sees_them() {
        assert!(validate_block_scale(0x7F).is_err());
        assert!(validate_block_scale(0xFF).is_err());
        assert!(validate_block_scale(0xB8).is_err()); // -1.0
        assert_eq!(validate_block_scale(0x38).unwrap(), 1.0);
        // Zero is legal: document 03 says it represents an all-zero block.
        assert_eq!(validate_block_scale(0x00).unwrap(), 0.0);
    }

    #[test]
    fn the_two_source_conventions_normalise_to_the_same_multiplier() {
        // This is R16 as an executable check. A checkpoint that stores 4.0 as a
        // divisor and one that stores 0.25 as a multiplier mean the same thing;
        // getting it backwards scales every weight by 16x.
        let from_divide = canonical_tensor_scale(4.0, GlobalScaleConvention::Divide).unwrap();
        let from_multiply = canonical_tensor_scale(0.25, GlobalScaleConvention::Multiply).unwrap();
        assert_eq!(from_divide, from_multiply);
        assert_eq!(from_divide, 0.25);

        // And they are genuinely different readings of the same stored number.
        let a = canonical_tensor_scale(4.0, GlobalScaleConvention::Divide).unwrap();
        let b = canonical_tensor_scale(4.0, GlobalScaleConvention::Multiply).unwrap();
        assert_ne!(a, b);
        assert_eq!(b / a, 16.0);
    }

    #[test]
    fn non_positive_or_non_finite_global_scales_are_rejected() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                canonical_tensor_scale(bad, GlobalScaleConvention::Multiply).is_err(),
                "{bad} should be rejected"
            );
            assert!(canonical_tensor_scale(bad, GlobalScaleConvention::Divide).is_err());
        }
    }

    #[test]
    fn nibble_order_is_low_first() {
        // One group, values 1.0 then 2.0. Codes: 1.0 = 0b0010, 2.0 = 0b0100.
        // Low nibble carries the earlier element, so the byte is 0x42.
        let mut packed = vec![0u8; 8];
        packed[0] = 0x42;
        let scales = [0x38u8]; // 1.0
        let out = dequantize_row(&packed, &scales, 1.0, 2).unwrap();
        assert_eq!(out, vec![1.0, 2.0]);

        // Reading it the other way round would give 2.0 then 1.0 -- a silent
        // transposition of every pair in the tensor.
        assert_ne!(out, vec![2.0, 1.0]);
    }

    #[test]
    fn the_full_reconstruction_equation_holds() {
        // Two groups of 16, second group scaled 2x, tensor scale 0.25.
        let mut packed = vec![0u8; 16];
        for b in packed.iter_mut() {
            *b = 0x22; // both nibbles = 1.0
        }
        let scales = [0x38u8, 0x40u8]; // 1.0 and 2.0
        let out = dequantize_row(&packed, &scales, 0.25, 32).unwrap();
        assert_eq!(out.len(), 32);
        for v in &out[..16] {
            assert_eq!(*v, 1.0 * 1.0 * 0.25);
        }
        for v in &out[16..] {
            assert_eq!(*v, 1.0 * 2.0 * 0.25);
        }
    }

    #[test]
    fn an_incomplete_group_is_padded_outside_the_logical_shape() {
        // 20 logical values need 2 groups (32 slots) and 16 packed bytes. The
        // 12 padding slots must not appear in the output.
        let packed = vec![0x22u8; 16];
        let scales = [0x38u8, 0x38u8];
        let out = dequantize_row(&packed, &scales, 1.0, 20).unwrap();
        assert_eq!(out.len(), 20);
    }

    #[test]
    fn truncated_inputs_are_rejected_rather_than_read_past_the_end() {
        let packed = vec![0x22u8; 4];
        let scales = [0x38u8];
        assert!(dequantize_row(&packed, &scales, 1.0, 32).is_err());
        assert!(dequantize_row(&packed, &[], 1.0, 8).is_err());
        assert!(dequantize_row(&packed, &scales, 0.0, 8).is_err());
        assert!(dequantize_row(&packed, &scales, f32::NAN, 8).is_err());
    }

    #[test]
    fn a_nan_scale_inside_a_row_fails_the_whole_row() {
        let packed = vec![0x22u8; 16];
        let scales = [0x38u8, 0x7Fu8]; // second group is NaN
        assert!(dequantize_row(&packed, &scales, 1.0, 32).is_err());
        // The first group alone still works, proving the failure is the scale.
        assert!(dequantize_row(&packed, &scales, 1.0, 16).is_ok());
    }
}
