//! The canonical affine chunk payload: three contiguous sections, one range.
//!
//! Manifest v1 reserves one `(chunk, offset, length)` range per tensor and one
//! SHA-256 over it. [ADR 0023] fixes what is inside that range for an affine
//! tensor: packed codes, then the scale table, then -- when the tensor is
//! asymmetric -- the zero-point table, contiguous and with no interior padding.
//! Every section extent is a function of the validated descriptor, so a reader
//! that has parsed the manifest knows where each section begins without reading
//! a byte.
//!
//! This module is the **only** place those bytes are laid out. The whole-tensor
//! encoder below and the streaming converter in [`crate::compressed_tensors`]
//! write through the same three section serializers, so "the repacker and the
//! reader agree" is a property of the code rather than a claim about two
//! implementations.
//!
//! I/O-free, like the rest of this crate: it fills and reads caller-supplied
//! byte slices and never learns that a file exists.
//!
//! [ADR 0023]: ../../../docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md

use core::ops::Range;

use moxie_types::{Error, Result};

use crate::affine::{AffineDescriptor, AffineTensor, ZeroPoints};
use crate::scale::{ScaleDtype, ScaleValues};

/// This module's refusals, composed **fallibly** -- see [`crate::invalid_fmt`].
fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed canonical payload (detail unavailable: out of memory)",
        detail,
    )
}

fn invalid_static(detail: &'static str) -> Error {
    crate::invalid_static(detail)
}

/// Canonical bytes per zero point: signed little-endian `i16`, wider than
/// either code range on purpose (document 03).
pub const ZERO_POINT_BYTES: usize = 2;

/// Whether a tensor's payload carries the third section.
///
/// A named pair rather than a `bool`, because "has zero points" and "is
/// asymmetric" are the same fact stated two ways and the manifest, the
/// descriptor and the payload each state it. Naming it keeps a disagreement
/// between them a refusal rather than a silent zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPointSection {
    /// Symmetric: implicit zero, **no payload at all**.
    Absent,
    /// One signed `i16` per `(output channel, group)`, row-major.
    PerGroup,
}

impl ZeroPointSection {
    /// The section a set of canonical zero points implies.
    pub fn of(zero_points: &ZeroPoints) -> Self {
        match zero_points {
            ZeroPoints::Symmetric => ZeroPointSection::Absent,
            ZeroPoints::PerGroup(_) => ZeroPointSection::PerGroup,
        }
    }
}

/// Where each section of one tensor's payload sits, relative to the start of
/// its manifest byte range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadExtents {
    codes: Range<usize>,
    scales: Range<usize>,
    zero_points: Option<Range<usize>>,
}

impl PayloadExtents {
    pub fn codes(&self) -> Range<usize> {
        self.codes.clone()
    }

    pub fn scales(&self) -> Range<usize> {
        self.scales.clone()
    }

    /// `None` for a symmetric tensor, which has no third section.
    pub fn zero_points(&self) -> Option<Range<usize>> {
        self.zero_points.clone()
    }

    /// Total payload length: the manifest's `length` for this tensor.
    pub fn total(&self) -> usize {
        match &self.zero_points {
            Some(z) => z.end,
            None => self.scales.end,
        }
    }
}

/// Section extents for a descriptor. The descriptor is validated first: every
/// number below is a product of fields whose bounds `validate` has checked.
pub fn extents(desc: &AffineDescriptor, zero_points: ZeroPointSection) -> Result<PayloadExtents> {
    desc.validate()?;
    let code_bytes = desc.code_bytes()?;
    let entries = desc.group_entries()?;
    let scale_bytes = entries
        .checked_mul(desc.scale_dtype.bytes())
        .ok_or_else(|| invalid_static("scale section length overflows usize"))?;
    let scales_end = code_bytes
        .checked_add(scale_bytes)
        .ok_or_else(|| invalid_static("payload length overflows usize"))?;
    let zero_points = match zero_points {
        ZeroPointSection::Absent => None,
        ZeroPointSection::PerGroup => {
            let zp_bytes = entries
                .checked_mul(ZERO_POINT_BYTES)
                .ok_or_else(|| invalid_static("zero-point section length overflows usize"))?;
            let end = scales_end
                .checked_add(zp_bytes)
                .ok_or_else(|| invalid_static("payload length overflows usize"))?;
            Some(scales_end..end)
        }
    };
    Ok(PayloadExtents {
        codes: 0..code_bytes,
        scales: code_bytes..scales_end,
        zero_points,
    })
}

/// The payload length a descriptor implies, as `u64` for manifest arithmetic.
pub fn length_of(desc: &AffineDescriptor, zero_points: ZeroPointSection) -> Result<u64> {
    let total = extents(desc, zero_points)?.total();
    u64::try_from(total).map_err(|_| invalid_static("payload length does not fit in u64"))
}

// --- the three section serializers -------------------------------------------
//
// Every canonical byte in this repository is written by one of these three
// functions, whether it comes from a whole imported tensor or from one bounded
// tile of a streaming repack.

/// Copy one block of packed codes.
///
/// Codes are already canonical bytes by the time they exist -- `pack_row_into`
/// produced them -- so this is a length-checked copy and exists so that the
/// code section has a named writer beside the other two.
pub fn write_code_block(codes: &[u8], out: &mut [u8]) -> Result<()> {
    if out.len() != codes.len() {
        return Err(invalid(format_args!(
            "code block is {} byte(s) but the destination holds {}",
            codes.len(),
            out.len()
        )));
    }
    out.copy_from_slice(codes);
    Ok(())
}

/// Write one block of scales from **source bytes already in the canonical
/// dtype**, validating every scalar on the way through.
///
/// This is the streaming repacker's path: a `pack-quantized` source serializes
/// its scale table row-major in its own dtype, which is the canonical order and
/// the canonical dtype, so the bytes are preserved exactly -- ADR 0018's
/// bit-identical repack, at the level of the individual scalar. What is *not*
/// preserved is a scalar a kernel must never see: a non-finite or non-positive
/// scale is refused here rather than copied.
pub fn write_scale_block(dtype: ScaleDtype, source_le: &[u8], out: &mut [u8]) -> Result<()> {
    let width = dtype.bytes();
    if !source_le.len().is_multiple_of(width) {
        return Err(invalid(format_args!(
            "a {} scale block of {} byte(s) is not a whole number of scalars",
            dtype.name(),
            source_le.len()
        )));
    }
    if out.len() != source_le.len() {
        return Err(invalid(format_args!(
            "scale block is {} byte(s) but the destination holds {}",
            source_le.len(),
            out.len()
        )));
    }
    for (i, raw) in source_le.chunks_exact(width).enumerate() {
        let v = decode_scalar(dtype, raw);
        if !v.is_finite() || v <= 0.0 {
            return Err(invalid(format_args!(
                "scale[{i}] of this block is {v} ({}); scales must be positive and finite",
                dtype.name()
            )));
        }
    }
    out.copy_from_slice(source_le);
    Ok(())
}

/// Write one block of scales from decoded canonical values.
///
/// The whole-tensor encoder's path. It produces the same bytes
/// [`write_scale_block`] copies, and `payload_round_trips_through_both_writers`
/// is what keeps that true.
pub fn write_scale_values(scales: &ScaleValues, out: &mut [u8]) -> Result<()> {
    scales.validate()?;
    let width = scales.dtype().bytes();
    let need = scales
        .len()
        .checked_mul(width)
        .ok_or_else(|| invalid_static("scale section length overflows usize"))?;
    if out.len() != need {
        return Err(invalid(format_args!(
            "{} {} scale(s) need {need} byte(s), the destination holds {}",
            scales.len(),
            scales.dtype().name(),
            out.len()
        )));
    }
    match scales {
        ScaleValues::F16(v) | ScaleValues::Bf16(v) => {
            for (bits, slot) in v.iter().zip(out.chunks_exact_mut(2)) {
                slot.copy_from_slice(&bits.to_le_bytes());
            }
        }
        ScaleValues::F32(v) => {
            for (value, slot) in v.iter().zip(out.chunks_exact_mut(4)) {
                slot.copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(())
}

/// Write **one** canonical zero point: signed little-endian `i16`.
///
/// The single-value form exists because the streaming converter decodes one
/// zero point at a time out of a source word and must not need an `i16`
/// scratch buffer to reach the serializer. Both writers below go through this,
/// so there is one definition of what a canonical zero point looks like on
/// disk.
pub fn write_zero_point(value: i16, out: &mut [u8]) -> Result<()> {
    if out.len() != ZERO_POINT_BYTES {
        return Err(invalid(format_args!(
            "a canonical zero point is {ZERO_POINT_BYTES} byte(s), the destination holds {}",
            out.len()
        )));
    }
    out.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// Write one block of canonical zero points: signed little-endian `i16`.
pub fn write_zero_point_block(values: &[i16], out: &mut [u8]) -> Result<()> {
    let need = values
        .len()
        .checked_mul(ZERO_POINT_BYTES)
        .ok_or_else(|| invalid_static("zero-point block length overflows usize"))?;
    if out.len() != need {
        return Err(invalid(format_args!(
            "{} zero point(s) need {need} byte(s), the destination holds {}",
            values.len(),
            out.len()
        )));
    }
    for (z, slot) in values.iter().zip(out.chunks_exact_mut(ZERO_POINT_BYTES)) {
        write_zero_point(*z, slot)?;
    }
    Ok(())
}

fn decode_scalar(dtype: ScaleDtype, raw: &[u8]) -> f32 {
    match dtype {
        ScaleDtype::Bf16 => crate::bf16::bf16_bits_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
        ScaleDtype::F16 => crate::scale::f16_bits_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
        ScaleDtype::F32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
    }
}

// --- whole-tensor encode and decode ------------------------------------------

/// Encode a whole canonical tensor into its payload range.
pub fn encode(tensor: &AffineTensor, out: &mut [u8]) -> Result<()> {
    let section = ZeroPointSection::of(tensor.zero_points());
    let ext = extents(tensor.descriptor(), section)?;
    if out.len() != ext.total() {
        return Err(invalid(format_args!(
            "this tensor's payload is {} byte(s), the destination holds {}",
            ext.total(),
            out.len()
        )));
    }
    write_code_block(tensor.codes(), &mut out[ext.codes()])?;
    write_scale_values(tensor.scales(), &mut out[ext.scales()])?;
    match (ext.zero_points(), tensor.zero_points()) {
        (None, ZeroPoints::Symmetric) => {}
        (Some(range), ZeroPoints::PerGroup(values)) => {
            write_zero_point_block(values, &mut out[range])?;
        }
        // `extents` derived the section from these same zero points, so this
        // is unreachable by construction; it is a refusal rather than a panic
        // because "unreachable by construction" has been wrong before.
        _ => return Err(invalid_static("zero-point section and payload disagree")),
    }
    Ok(())
}

/// Decode a payload range back into a canonical tensor.
///
/// The inverse of [`encode`], and the reader half of ADR 0023. It validates the
/// length against the descriptor's own arithmetic first: a range whose length
/// disagrees with the descriptor is a rejection, never a prefix read.
pub fn decode(
    desc: AffineDescriptor,
    zero_points: ZeroPointSection,
    bytes: &[u8],
) -> Result<AffineTensor> {
    let ext = extents(&desc, zero_points)?;
    if bytes.len() != ext.total() {
        return Err(invalid(format_args!(
            "payload is {} byte(s); this descriptor requires exactly {}",
            bytes.len(),
            ext.total()
        )));
    }
    let mut codes = crate::try_vec::<u8>(ext.codes().len())?;
    codes.extend_from_slice(&bytes[ext.codes()]);

    let entries = desc.group_entries()?;
    let scales = decode_scale_section(desc.scale_dtype, &bytes[ext.scales()], entries)?;

    let zeros = match ext.zero_points() {
        None => ZeroPoints::Symmetric,
        Some(range) => {
            let mut values = crate::try_vec::<i16>(entries)?;
            for raw in bytes[range].chunks_exact(ZERO_POINT_BYTES) {
                values.push(i16::from_le_bytes([raw[0], raw[1]]));
            }
            ZeroPoints::PerGroup(values)
        }
    };
    AffineTensor::new(desc, codes, scales, zeros)
}

/// Decode the scale section into canonical values, fallibly sized.
pub fn decode_scale_section(
    dtype: ScaleDtype,
    bytes: &[u8],
    entries: usize,
) -> Result<ScaleValues> {
    let need = entries
        .checked_mul(dtype.bytes())
        .ok_or_else(|| invalid_static("scale section length overflows usize"))?;
    if bytes.len() != need {
        return Err(invalid(format_args!(
            "scale section is {} byte(s); {entries} {} scale(s) need {need}",
            bytes.len(),
            dtype.name()
        )));
    }
    Ok(match dtype {
        ScaleDtype::Bf16 | ScaleDtype::F16 => {
            let mut v = crate::try_vec::<u16>(entries)?;
            for raw in bytes.chunks_exact(2) {
                v.push(u16::from_le_bytes([raw[0], raw[1]]));
            }
            match dtype {
                ScaleDtype::Bf16 => ScaleValues::Bf16(v),
                _ => ScaleValues::F16(v),
            }
        }
        ScaleDtype::F32 => {
            let mut v = crate::try_vec::<f32>(entries)?;
            for raw in bytes.chunks_exact(4) {
                v.push(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]));
            }
            ScaleValues::F32(v)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::{Grouping, IntWidth, pack_row};

    fn desc(width: IntWidth, out_features: usize, in_features: usize) -> AffineDescriptor {
        AffineDescriptor {
            width,
            out_features,
            in_features,
            grouping: Grouping::Contiguous { size: 32 },
            group_index: None,
            scale_dtype: ScaleDtype::Bf16,
        }
    }

    fn tensor(width: IntWidth, rows: usize, columns: usize, asymmetric: bool) -> AffineTensor {
        let d = desc(width, rows, columns);
        let entries = d.group_entries().unwrap();
        let mut codes = Vec::new();
        for o in 0..rows {
            let row: Vec<i32> = (0..columns)
                .map(|k| {
                    let (lo, hi) = width.code_range();
                    lo + ((o + k) as i32 % (hi - lo + 1))
                })
                .collect();
            codes.extend_from_slice(&pack_row(width, &row).unwrap());
        }
        let scales = ScaleValues::Bf16(vec![crate::bf16::f32_to_bf16_bits(0.5); entries]);
        let zeros = if asymmetric {
            ZeroPoints::PerGroup((0..entries).map(|i| (i as i16 % 9) - 4).collect())
        } else {
            ZeroPoints::Symmetric
        };
        AffineTensor::new(d, codes, scales, zeros).unwrap()
    }

    /// The arithmetic ADR 0023 states, checked against a hand-computed size
    /// rather than against the function that computes it.
    #[test]
    fn section_extents_are_the_documented_arithmetic() {
        // 4 rows x 64 columns, INT4, group 32, BF16 scales, asymmetric:
        // codes 4*32 = 128, groups 4*2 = 8 entries -> scales 16, zeros 16.
        let d = desc(IntWidth::Int4, 4, 64);
        let e = extents(&d, ZeroPointSection::PerGroup).unwrap();
        assert_eq!(e.codes(), 0..128);
        assert_eq!(e.scales(), 128..144);
        assert_eq!(e.zero_points(), Some(144..160));
        assert_eq!(e.total(), 160);
        // Symmetric drops the third section entirely: no reserved zeros.
        let e = extents(&d, ZeroPointSection::Absent).unwrap();
        assert_eq!(e.total(), 144);
        assert_eq!(e.zero_points(), None);
    }

    #[test]
    fn encode_then_decode_is_the_same_tensor() {
        for width in [IntWidth::Int4, IntWidth::Int8] {
            for asymmetric in [false, true] {
                // 66 columns: a partial group and, at INT4, a tail nibble.
                let t = tensor(width, 5, 66, asymmetric);
                let section = ZeroPointSection::of(t.zero_points());
                let len = extents(t.descriptor(), section).unwrap().total();
                let mut bytes = vec![0u8; len];
                encode(&t, &mut bytes).unwrap();
                let back = decode(t.descriptor().clone(), section, &bytes).unwrap();
                assert_eq!(back, t, "{width:?} asymmetric={asymmetric}");
            }
        }
    }

    /// The two scale writers must produce identical bytes: one copies source
    /// bytes through a validation, the other serializes decoded values.
    #[test]
    fn both_scale_writers_produce_the_same_bytes() {
        for dtype in ScaleDtype::ALL.iter().copied() {
            let values: Vec<f32> = vec![0.5, 1.0, 2.0, 0.25];
            let decoded = match dtype {
                ScaleDtype::Bf16 => ScaleValues::Bf16(
                    values
                        .iter()
                        .map(|v| crate::bf16::f32_to_bf16_bits(*v))
                        .collect(),
                ),
                // 0.5/1.0/2.0/0.25 are exact in f16; the bit patterns are the
                // format's, written out rather than converted by code under
                // test.
                ScaleDtype::F16 => ScaleValues::F16(vec![0x3800, 0x3C00, 0x4000, 0x3400]),
                ScaleDtype::F32 => ScaleValues::F32(values.clone()),
            };
            let mut from_values = vec![0u8; decoded.len() * dtype.bytes()];
            write_scale_values(&decoded, &mut from_values).unwrap();
            let mut from_source = vec![0u8; from_values.len()];
            write_scale_block(dtype, &from_values, &mut from_source).unwrap();
            assert_eq!(from_values, from_source, "{}", dtype.name());
            // And both decode back to the values they came from.
            let back = decode_scale_section(dtype, &from_source, decoded.len()).unwrap();
            assert_eq!(back, decoded, "{}", dtype.name());
        }
    }

    #[test]
    fn a_nonfinite_or_zero_scale_is_refused_by_both_writers() {
        for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            let values = ScaleValues::F32(vec![1.0, bad]);
            let mut out = vec![0u8; 8];
            assert!(write_scale_values(&values, &mut out).is_err(), "{bad}");
            let mut raw = Vec::new();
            raw.extend_from_slice(&1.0f32.to_le_bytes());
            raw.extend_from_slice(&bad.to_le_bytes());
            let mut out = vec![0u8; 8];
            assert!(
                write_scale_block(ScaleDtype::F32, &raw, &mut out).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_payload_whose_length_disagrees_with_the_descriptor_is_refused() {
        let t = tensor(IntWidth::Int4, 4, 64, true);
        let section = ZeroPointSection::of(t.zero_points());
        let len = extents(t.descriptor(), section).unwrap().total();
        let mut bytes = vec![0u8; len];
        encode(&t, &mut bytes).unwrap();
        // One byte short, and one byte long: a prefix read would accept the
        // first and a trailing-garbage read the second.
        for wrong in [len - 1, len + 1] {
            let mut resized = bytes.clone();
            resized.resize(wrong, 0);
            let e = decode(t.descriptor().clone(), section, &resized).unwrap_err();
            assert!(e.to_string().contains("requires exactly"), "{e}");
        }
        // A symmetric descriptor over an asymmetric payload has the wrong
        // length too, which is the mismatch that would otherwise read scales
        // as zero points.
        let e = decode(t.descriptor().clone(), ZeroPointSection::Absent, &bytes).unwrap_err();
        assert!(e.to_string().contains("requires exactly"), "{e}");
    }

    /// Expected bytes built by hand, not by the encoder: the one comparison
    /// that can catch the encoder and the decoder agreeing on the wrong thing.
    #[test]
    fn one_tiny_tensor_encodes_to_bytes_written_out_by_hand() {
        // INT4, 2 rows x 4 columns, per-channel grouping, F32 scales,
        // asymmetric. Codes: row 0 is [-8, -1, 0, 7], row 1 is [1, 2, 3, 4].
        let d = AffineDescriptor {
            width: IntWidth::Int4,
            out_features: 2,
            in_features: 4,
            grouping: Grouping::PerOutputChannel,
            group_index: None,
            scale_dtype: ScaleDtype::F32,
        };
        let mut codes = Vec::new();
        codes.extend_from_slice(&pack_row(IntWidth::Int4, &[-8, -1, 0, 7]).unwrap());
        codes.extend_from_slice(&pack_row(IntWidth::Int4, &[1, 2, 3, 4]).unwrap());
        let t = AffineTensor::new(
            d,
            codes,
            ScaleValues::F32(vec![0.5, 0.25]),
            ZeroPoints::PerGroup(vec![-3, 6]),
        )
        .unwrap();
        let mut got = vec![0u8; 2 * 2 + 2 * 4 + 2 * 2];
        encode(&t, &mut got).unwrap();
        let want: Vec<u8> = vec![
            // codes: low nibble is the earlier column. Row 0: (-8, -1) ->
            // 0x8 | 0xF<<4 = 0xF8; (0, 7) -> 0x0 | 0x7<<4 = 0x70.
            0xF8, 0x70, // Row 1: (1, 2) -> 0x21; (3, 4) -> 0x43.
            0x21, 0x43, // scales, f32 little-endian: 0.5, 0.25.
            0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x80,
            0x3E, // zero points, i16 little-endian: -3, 6.
            0xFD, 0xFF, 0x06, 0x00,
        ];
        assert_eq!(got, want);
    }
}
