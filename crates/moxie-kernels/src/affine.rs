//! Borrowed canonical affine operands for shared host kernels. No allocation,
//! conversion buffer, placement policy or ownership of the source bytes.
use moxie_types::{Error, Precision, Result};

#[derive(Debug, Clone, Copy)]
pub struct AffineWeight<'a> {
    payload: &'a [u8],
    outputs: usize,
    inputs: usize,
    bits: usize,
    group: usize,
    groups: usize,
    scale_dtype: Precision,
    scale_offset: usize,
    zero_offset: usize,
    per_group_zeros: bool,
    map: Option<&'a [u32]>,
}

fn invalid() -> Error {
    Error::InvalidArtifact {
        detail: "invalid canonical affine host operand".into(),
    }
}

impl<'a> AffineWeight<'a> {
    /// A single canonical codes/scales/optional-i16-zeros payload. The caller
    /// retains its lease for the entire lifetime of this view.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        payload: &'a [u8],
        outputs: usize,
        inputs: usize,
        width: Precision,
        group: usize,
        scale_dtype: Precision,
        per_group_zeros: bool,
        map: Option<&'a [u32]>,
    ) -> Result<Self> {
        if outputs == 0
            || inputs == 0
            || !matches!(group, 32 | 128)
            || !matches!(width, Precision::Int4 | Precision::Int8)
            || !matches!(
                scale_dtype,
                Precision::F16 | Precision::Bf16 | Precision::F32
            )
        {
            return Err(invalid());
        }
        let bits = width.bits() as usize;
        let groups = inputs.div_ceil(group);
        let entries = outputs.checked_mul(groups).ok_or_else(invalid)?;
        let scale_offset = outputs
            .checked_mul(inputs.div_ceil(8 / bits))
            .ok_or_else(invalid)?;
        let zero_offset = entries
            .checked_mul(scale_dtype.bits() as usize / 8)
            .and_then(|n| n.checked_add(scale_offset))
            .ok_or_else(invalid)?;
        let len = if per_group_zeros {
            entries
                .checked_mul(2)
                .and_then(|n| zero_offset.checked_add(n))
                .ok_or_else(invalid)?
        } else {
            zero_offset
        };
        if payload.len() != len
            || map.is_some_and(|m| m.len() != inputs || m.iter().any(|&g| g as usize >= groups))
        {
            return Err(invalid());
        }
        if map.is_some_and(|m| (0..groups).any(|g| !m.contains(&(g as u32)))) {
            return Err(invalid());
        }
        let view = Self {
            payload,
            outputs,
            inputs,
            bits,
            group,
            groups,
            scale_dtype,
            scale_offset,
            zero_offset,
            per_group_zeros,
            map,
        };
        for entry in 0..entries {
            let scale = view.scale(entry);
            if !scale.is_finite() || scale == 0.0 {
                return Err(invalid());
            }
        }
        Ok(view)
    }

    pub fn shape(self) -> (usize, usize) {
        (self.outputs, self.inputs)
    }

    fn scale(self, index: usize) -> f32 {
        let at = self.scale_offset + index * (self.scale_dtype.bits() as usize / 8);
        if self.scale_dtype == Precision::F32 {
            return f32::from_le_bytes(
                self.payload[at..at + 4]
                    .try_into()
                    .expect("validated scale"),
            );
        }
        let bits = u16::from_le_bytes(
            self.payload[at..at + 2]
                .try_into()
                .expect("validated scale"),
        );
        if self.scale_dtype == Precision::Bf16 {
            return f32::from_bits(u32::from(bits) << 16);
        }
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 31;
        let mantissa = u32::from(bits & 1023);
        if exponent == 0 {
            let value = mantissa as f32 * 2f32.powi(-24);
            return if sign == 0 { value } else { -value };
        }
        let exponent = if exponent == 31 {
            255
        } else {
            u32::from(exponent) + 112
        };
        f32::from_bits(sign | exponent << 23 | mantissa << 13)
    }

    /// Checked shape at kernel entry makes every inner-loop index in range.
    pub(crate) fn value(self, row: usize, column: usize) -> f32 {
        let stride = self.inputs.div_ceil(8 / self.bits);
        let code = if self.bits == 4 {
            let nibble = (self.payload[row * stride + column / 2] >> ((column % 2) * 4)) & 15;
            ((nibble as i8) << 4 >> 4) as i32
        } else {
            self.payload[row * stride + column] as i8 as i32
        };
        let group = self.map.map_or(column / self.group, |m| m[column] as usize);
        let entry = row * self.groups + group;
        let zero = if self.per_group_zeros {
            let at = self.zero_offset + entry * 2;
            i16::from_le_bytes(self.payload[at..at + 2].try_into().expect("validated zero")) as i32
        } else {
            0
        };
        (code - zero) as f32 * self.scale(entry)
    }
}
