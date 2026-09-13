//! compressed-tensors `pack-quantized` import into canonical affine form.
//!
//! This is an **importer**, not a runtime. Document 03 orders it first among
//! M3's serializations, and its whole job is to absorb one source packing so
//! that nothing downstream has to know the source existed: the output is the
//! [`AffineTensor`](crate::affine::AffineTensor) that the accepted `affine`
//! module already reconstructs, with the same equation and the same oracle.
//!
//! There is deliberately no place here to hang a method-specific decode path.
//! AWQ, AutoRound and GPTQ are quantization *methods*; this is a serialization,
//! and document 03's rule is that they all import into one descriptor.
//!
//! ## The code packing
//!
//! Pinned to `src/platform/compressed_tensors.cpp:186` and `:286` at the frozen
//! legacy commit, which is the authority for every constant below:
//!
//! ```text
//! values_per_word = 32 / bits                (4 for INT8, 8 for INT4)
//! packed_columns  = ceil(in_features / values_per_word)
//! word_index      = o * packed_columns + k / values_per_word
//! lane            = k % values_per_word
//! raw             = (word >> (lane * bits)) & ((1 << bits) - 1)
//! q               = raw - (1 << (bits - 1))
//! ```
//!
//! The last line is document 03's rebias -- "INT8 unsigned u in [0,255]
//! analogously uses q=u-128 ... This is rebiasing, not requantization" -- and it
//! calls the accepted [`crate::affine::rebias_code`] rather than repeating it.
//!
//! **Why biased-unsigned rather than two's complement.** Confirmed against the
//! artifact before this module was written: a byte histogram of one packed row
//! of `model.language_model.layers.12.self_attn.q_proj.weight_packed` in the
//! Gemma 4 shard `model-00002-of-00007.safetensors` is dense in `64..191` and
//! sparse at both ends, mean raw byte **127.4**. Two's-complement bytes from a
//! symmetric quantizer would be edge-heavy instead.
//!
//! **What the artifact cannot establish, stated because no local check closes
//! it.** All `values_per_word` lanes of a word fall inside one scale group, so
//! the artifact's own data cannot distinguish the lane order: reversing it
//! reconstructs a different but equally plausible weight. The order above comes
//! from the pinned reader and is not inferred. Closing it needs paired output
//! against the released model, which is O2 evidence and M3's later kernel work.
//! Nothing here may be read as a quality claim.
//!
//! ## The zero-point packing, which is a **second** convention
//!
//! Task 0024. An asymmetric source serializes a fourth tensor,
//! `weight_zero_point`, and packs it along the **output** axis while
//! `weight_packed` packs along the input axis. Two packing conventions inside
//! one tensor group, distinguished by nothing in either name -- which is R16
//! ("format names hide incompatible scale/layout conventions") in its exact
//! form, and why the shapes are validated rather than derived from byte counts.
//!
//! Pinned to the `compressed-tensors` library, **version 0.17.0**, resolved on
//! the benchmark machine at
//! `/home/rodrigo/.cache/uv/archive-v0/uqs9z2Tvizx6-8cq0I0rd/compressed_tensors`:
//!
//! | File | sha256 |
//! |---|---|
//! | `compressors/pack_quantized/helpers.py` | `8619308666eba5e8a442d1647c34b6b5f9716b13ed2051ed17fb1ea683b0db72` |
//! | `compressors/pack_quantized/base.py` | `a6a532a0b2ae19b7ebfb425d73776653ee8d117b3f69778f05e44e67175cfc9d` |
//! | `quantization/lifecycle/forward_helpers.py` | `8b3399fda143cc249c231e291d938c987a7da4793eeae35b3136b3390bbde6c8` |
//!
//! `PackedQuantizationCompressor.compress` writes
//! `pack_to_int32(zero_point, num_bits, packed_dim=0)` when the scheme is
//! asymmetric and its strategy is `GROUP` or `CHANNEL`; `decompress` unpacks it
//! into `(*original_shape[:-1], scale.shape[-1])`, which is
//! `[out_features, groups]`. `pack_to_int32` with `packed_dim=0` transposes,
//! packs along what is then the column axis, and transposes back;
//! `unpack_from_int32` inverts it with
//! `unpacked[l::pack_factor, :] = value >> (bits * l)`. One mapping follows:
//!
//! ```text
//! zp_rows    = ceil(out_features / values_per_word)
//! word_index = (o / values_per_word) * groups + g
//! lane       = o % values_per_word
//! raw        = (word >> (lane * bits)) & ((1 << bits) - 1)
//! z          = raw - (1 << (bits - 1))
//! ```
//!
//! and the same library's `_dequantize` is `(x_q - zero_point) * scale`, which
//! is document 03's equation with document 03's sign. A zero point *added*
//! rather than subtracted would reconstruct a different tensor with no shape
//! disagreeing, so the sign is named here rather than inferred from a round
//! trip through this crate's own encoder.
//!
//! **What the artifacts do establish, and the symmetric case could not.** A
//! zero-point word's lanes are *different output channels*, whose code
//! statistics differ -- unlike a code word's lanes, which all fall inside one
//! scale group. So this lane assignment is measurable against the artifacts'
//! own bytes, and it was measured rather than assumed: see
//! `docs/evidence/experiments/0005-asymmetric-int4-zero-point-assignment.md`.
//! That matters because the four local asymmetric artifacts declare compressor
//! versions (`0.1.dev534+gb269f2e`, `0.1.dev535+gdc9611a`) that are **not** the
//! locally resolved 0.17.0. The measurement is corroboration of a reading, not
//! a quality claim; quality is O2's and needs paired output.

use crate::affine::{
    AffineDescriptor, AffineTensor, Grouping, IntWidth, ZeroPoints, rebias_code, rebias_zero_point,
};
use crate::safetensors::{Dtype, Header, TensorEntry};
use crate::scale::{ScaleDtype, ScaleValues};
use moxie_types::{Error, Result};

/// This module's refusals, composed **fallibly**.
///
/// Takes `fmt::Arguments` rather than a `String`, so the message is never
/// built by an allocation that can abort: [`crate::invalid_fmt`] grows its
/// buffer through `try_reserve` and falls back to a borrowed static detail.
/// Task 0024's review reached `SIGABRT` here by refusing one allocation while
/// this module refused a malformed artifact.
fn invalid(detail: core::fmt::Arguments<'_>) -> Error {
    crate::invalid_fmt(
        "a malformed compressed-tensors pack-quantized source (detail unavailable: out of memory)",
        detail,
    )
}

/// A refusal whose whole message is static: allocation-free, always available.
#[allow(dead_code)]
fn invalid_static(detail: &'static str) -> Error {
    crate::invalid_static(detail)
}

/// How a source divides the input axis for scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// One scale per output channel.
    Channel,
    /// Contiguous groups of `size` input channels.
    Group { size: u32 },
}

/// How a source serializes its zero points.
///
/// A named convention rather than a `symmetric: bool`, because the boolean said
/// only that *some* zero points exist and this module has to know **where**.
/// Document 03: "GPTQ-style stored zero offsets ..., packed integer axis order,
/// interleaving and activation-order maps must be decoded according to the
/// pinned exporter, never guessed from a suffix."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroPointSource {
    /// Implicit zero, no payload: `W = q * s`.
    Symmetric,
    /// A `weight_zero_point` tensor of `I32` words, `values_per_word` zero
    /// points per word packed along the **output** axis, shaped
    /// `[ceil(out_features / values_per_word), groups]`.
    PackedAlongOutput,
}

/// The `quantization_config` fields this importer consumes.
///
/// Named rather than inferred: a caller reads these from the artifact's config
/// and passes them in; this module does not go looking for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackQuantizedSpec {
    pub width: IntWidth,
    pub granularity: Granularity,
    /// The source's zero-point serialization, from its `symmetric` flag and the
    /// pinned compressor's rule for where the payload goes.
    pub zero_points: ZeroPointSource,
}

impl PackQuantizedSpec {
    /// Values packed into each 32-bit word.
    pub const fn values_per_word(&self) -> usize {
        32 / self.width.bits() as usize
    }
}

/// The `weight_zero_point` payload and the shape its header declared.
#[derive(Debug, Clone, Copy)]
pub struct PackedZeroPoints<'a> {
    /// `weight_zero_point`, Safetensors `I32`.
    pub payload: &'a [u8],
    /// Its declared shape, `[ceil(out_features / values_per_word), groups]`.
    pub shape: &'a [u64],
}

/// The tensors a `pack-quantized` weight is serialized as.
///
/// Each payload is accompanied by its **declared shape**, because byte counts
/// alone do not establish the axes. Independent review imported a logical
/// `[2, 64]` from a packed tensor declared `[16, 2]` instead of `[2, 16]` and a
/// scale declared `[1, 4]` instead of `[2, 2]`: the byte counts matched, so
/// nothing noticed that the axes were transposed and the weights were silently
/// wrong. The shapes are checked in [`import`] against the logical shape, which
/// `weight_shape` is the authority for.
#[derive(Debug, Clone, Copy)]
pub struct SourceTensors<'a> {
    /// `weight_packed`, Safetensors `I32`.
    pub packed: &'a [u8],
    /// `weight_packed`'s declared shape, `[out_features, packed_columns]`.
    pub packed_shape: &'a [u64],
    /// `weight_scale`, whose dtype is read from **its own header entry**.
    pub scale: &'a [u8],
    /// `weight_scale`'s declared shape.
    pub scale_shape: &'a [u64],
    pub scale_dtype: ScaleDtype,
    /// `weight_zero_point`, present exactly when the spec declares an
    /// asymmetric source. A payload without a declaration, or a declaration
    /// without a payload, is refused: those are two independently obtained
    /// facts about one tensor and a disagreement means one of them came from
    /// the wrong place.
    pub zero_point: Option<PackedZeroPoints<'a>>,
    /// Logical `[out_features, in_features]`, from the `weight_shape` payload.
    pub logical: (usize, usize),
}

/// Decode a `weight_shape` payload: exactly two little-endian `I64` values.
pub fn decode_weight_shape(payload: &[u8]) -> Result<(usize, usize)> {
    if payload.len() != 16 {
        return Err(invalid(format_args!(
            "weight_shape must hold exactly two I64 values, got {} byte(s)",
            payload.len()
        )));
    }
    let read = |at: usize| i64::from_le_bytes(payload[at..at + 8].try_into().expect("eight bytes"));
    let (rows, columns) = (read(0), read(8));
    if rows <= 0 || columns <= 0 {
        return Err(invalid(format_args!(
            "weight_shape dimensions must be positive, got [{rows}, {columns}]"
        )));
    }
    let to_usize = |v: i64| {
        usize::try_from(v)
            .map_err(|_| invalid_static("weight_shape dimension does not fit this platform"))
    };
    Ok((to_usize(rows)?, to_usize(columns)?))
}

/// The safetensors dtype a scale tensor must carry, mapped to the canonical tag.
///
/// Document 03: "Read dtype from tensor headers, not the model's general dtype
/// field." The Gemma 4 artifact is exactly why -- it declares
/// `scale_dtype: null` while its headers say BF16.
pub fn scale_dtype_of(dtype: Dtype) -> Result<ScaleDtype> {
    Ok(match dtype {
        Dtype::Bf16 => ScaleDtype::Bf16,
        Dtype::F16 => ScaleDtype::F16,
        Dtype::F32 => ScaleDtype::F32,
        other => {
            return Err(invalid(format_args!(
                "weight_scale is {}; canonical scales are BF16, F16 or F32",
                other.name()
            )));
        }
    })
}

/// The header entries one `pack-quantized` module is serialized as.
#[derive(Debug)]
pub struct SourceEntries<'a> {
    pub packed: &'a TensorEntry,
    pub scale: &'a TensorEntry,
    pub shape: &'a TensorEntry,
    /// Present exactly when `zero_points` is [`ZeroPointSource::PackedAlongOutput`].
    pub zero_point: Option<&'a TensorEntry>,
}

/// Pull the payload ranges for one module out of a parsed header.
///
/// Validates dtypes, so a caller reads exactly the byte ranges a module has and
/// nothing else. Naming is the source's: `<module>.weight_packed` and so on.
///
/// The declared zero-point serialization and what the file actually carries are
/// **required to agree**. A module that serializes `weight_zero_point` while
/// the config says symmetric is not a symmetric module with a spare tensor; it
/// is a disagreement between two independently read facts, and reading one of
/// them and ignoring the other is how a whole tensor comes out shifted.
pub fn source_entries<'a>(
    header: &'a Header,
    module: &str,
    zero_points: ZeroPointSource,
) -> Result<SourceEntries<'a>> {
    // Every name is built fallibly: `format!` aborts, and this helper resolves
    // four of them before it reaches a refusal.
    let packed = header.get(&crate::join_name(module, "weight_packed")?)?;
    let scale = header.get(&crate::join_name(module, "weight_scale")?)?;
    let shape = header.get(&crate::join_name(module, "weight_shape")?)?;
    if packed.dtype != Dtype::I32 {
        return Err(invalid(format_args!(
            "{module}.weight_packed is {}; pack-quantized requires I32",
            packed.dtype.name()
        )));
    }
    // `as_slice()` rather than `vec![2]`: the comparison needed no allocation
    // and the one it made was infallible.
    if shape.dtype != Dtype::I64 || shape.shape.as_slice() != [2] {
        return Err(invalid(format_args!(
            "{module}.weight_shape must be I64[2], got {} {:?}",
            shape.dtype.name(),
            shape.shape
        )));
    }
    let name = crate::join_name(module, "weight_zero_point")?;
    let present = header.tensors().get(&name);
    let zero_point = match (zero_points, present) {
        (ZeroPointSource::Symmetric, None) => None,
        (ZeroPointSource::PackedAlongOutput, Some(e)) => {
            if e.dtype != Dtype::I32 {
                return Err(invalid(format_args!(
                    "{name} is {}; a packed zero point is I32",
                    e.dtype.name()
                )));
            }
            Some(e)
        }
        (ZeroPointSource::Symmetric, Some(_)) => {
            return Err(invalid(format_args!(
                "{module} is declared symmetric but serializes {name}; the config and the \
                 tensor index disagree about this module's zero points"
            )));
        }
        (ZeroPointSource::PackedAlongOutput, None) => {
            return Err(invalid(format_args!(
                "{module} is declared asymmetric but has no {name}"
            )));
        }
    };
    Ok(SourceEntries {
        packed,
        scale,
        shape,
        zero_point,
    })
}

/// Import one `pack-quantized` tensor into canonical affine form.
///
/// Every length is checked against the logical shape from `weight_shape`, which
/// is the authority; the packed, scale and zero-point shapes are validated
/// against it rather than used to derive it.
pub fn import(spec: &PackQuantizedSpec, src: SourceTensors<'_>) -> Result<AffineTensor> {
    // The spec and the payload are two independently obtained facts about one
    // tensor. Neither is allowed to be read while the other is ignored.
    match (spec.zero_points, src.zero_point.is_some()) {
        (ZeroPointSource::Symmetric, false) | (ZeroPointSource::PackedAlongOutput, true) => {}
        (ZeroPointSource::Symmetric, true) => {
            return Err(invalid_static(
                "a symmetric pack-quantized source carries no weight_zero_point, but one was \
                 supplied",
            ));
        }
        (ZeroPointSource::PackedAlongOutput, false) => {
            return Err(invalid_static(
                "an asymmetric pack-quantized source requires its weight_zero_point payload",
            ));
        }
    }

    let (out_features, in_features) = src.logical;
    if out_features == 0 || in_features == 0 {
        return Err(invalid(format_args!(
            "empty logical shape {out_features}x{in_features}"
        )));
    }
    let per_word = spec.values_per_word();
    let packed_columns = in_features.div_ceil(per_word);
    // The declared axes, before the byte counts. A transposed or otherwise
    // incompatible shape can have exactly the right number of bytes.
    let want_packed = [out_features as u64, packed_columns as u64];
    if src.packed_shape != want_packed {
        return Err(invalid(format_args!(
            "weight_packed is declared {:?}; {out_features}x{in_features} at {} bit(s) \
             requires {want_packed:?}",
            src.packed_shape,
            spec.width.bits()
        )));
    }
    let need_words = packed_columns
        .checked_mul(out_features)
        .ok_or_else(|| invalid_static("packed word count overflows usize"))?;
    let need_packed = need_words
        .checked_mul(4)
        .ok_or_else(|| invalid_static("packed byte count overflows usize"))?;
    if src.packed.len() != need_packed {
        return Err(invalid(format_args!(
            "weight_packed is {} byte(s); {out_features}x{in_features} at {} bit(s) needs \
             {need_packed} ({out_features}x{packed_columns} I32 words)",
            src.packed.len(),
            spec.width.bits()
        )));
    }

    let grouping = match spec.granularity {
        Granularity::Channel => Grouping::PerOutputChannel,
        Granularity::Group { size } => Grouping::Contiguous { size },
    };
    let desc = AffineDescriptor {
        width: spec.width,
        out_features,
        in_features,
        grouping,
        // Contiguous. `actorder` is null in every inspected asymmetric
        // artifact; a source that carried a permutation would have to supply
        // the map here, and document 03 forbids ignoring one.
        group_index: None,
        scale_dtype: src.scale_dtype,
    };
    desc.validate()?;

    let entries = desc.group_entries()?;
    let groups_per_row = desc.groups_per_row()?;
    // Per-channel sources serialize the scale as `[out, 1]`; some writers emit
    // `[out]`. Both are accepted, and nothing else is: a scale whose rows do
    // not match the output channels is a different tensor, not a reshape.
    let scale_ok = src.scale_shape == [out_features as u64, groups_per_row as u64]
        || (groups_per_row == 1 && src.scale_shape == [out_features as u64]);
    if !scale_ok {
        return Err(invalid(format_args!(
            "weight_scale is declared {:?}; {out_features} output channel(s) with \
             {groups_per_row} group(s) each requires [{out_features}, {groups_per_row}]",
            src.scale_shape
        )));
    }
    let need_scale = entries
        .checked_mul(src.scale_dtype.bytes())
        .ok_or_else(|| invalid_static("scale byte count overflows usize"))?;
    if src.scale.len() != need_scale {
        return Err(invalid(format_args!(
            "weight_scale is {} byte(s); {entries} {} scale(s) need {need_scale}",
            src.scale.len(),
            src.scale_dtype.name()
        )));
    }

    // The zero points, before anything reads a code: a refusal here must not
    // come after a whole tensor has been unpacked.
    let zero_points = match src.zero_point {
        None => ZeroPoints::Symmetric,
        Some(zp) => decode_zero_points(spec, zp, out_features, groups_per_row, entries)?,
    };

    // Unpack into canonical packed codes: logical row-major, each row
    // byte-aligned, which is what `AffineTensor` stores.
    let mut codes = crate::try_vec::<i32>(in_features)?;
    let code_bytes = desc.code_bytes()?;
    let mut out = crate::try_vec::<u8>(code_bytes)?;
    out.resize(code_bytes, 0);
    let bits = spec.width.bits();
    let mask: u32 = (1u32 << bits) - 1;
    for o in 0..out_features {
        codes.clear();
        for k in 0..in_features {
            let word_index = o * packed_columns + k / per_word;
            let at = word_index * 4;
            let word = u32::from_le_bytes(
                src.packed[at..at + 4]
                    .try_into()
                    .expect("four bytes, length checked above"),
            );
            let lane = k % per_word;
            let raw = (word >> (lane as u32 * bits)) & mask;
            codes.push(rebias_code(spec.width, raw)?);
        }
        // Into the already reserved destination: no per-row allocation, so
        // there is no infallible allocation anywhere on this path.
        let stride = spec.width.row_stride(in_features);
        crate::affine::pack_row_into(spec.width, &codes, &mut out[o * stride..(o + 1) * stride])?;
    }

    let scales = decode_scales(src.scale, src.scale_dtype, entries)?;
    AffineTensor::new(desc, out, scales, zero_points)
}

/// Decode `weight_zero_point` into canonical `(output channel, group)` order.
///
/// The packing is the pinned compressor's `packed_dim=0`: word row
/// `o / values_per_word`, lane `o % values_per_word`. Padding lanes above
/// `out_features` exist whenever the output axis is not a multiple of
/// `values_per_word`, and are **never read** -- the loop is over logical output
/// channels, so what a writer padded with is not this reader's business.
fn decode_zero_points(
    spec: &PackQuantizedSpec,
    zp: PackedZeroPoints<'_>,
    out_features: usize,
    groups_per_row: usize,
    entries: usize,
) -> Result<ZeroPoints> {
    let per_word = spec.values_per_word();
    let zp_rows = out_features.div_ceil(per_word);
    let want = [zp_rows as u64, groups_per_row as u64];
    // Exactly this shape. The scale rule also accepts a one-dimensional
    // `[out]`, because a writer was seen to emit it; no writer has been seen to
    // emit a one-dimensional zero point, and accepting a shape nothing produces
    // is a branch no fixture can justify.
    if zp.shape != want {
        return Err(invalid(format_args!(
            "weight_zero_point is declared {:?}; {out_features} output channel(s) at {} bit(s) \
             with {groups_per_row} group(s) each requires {want:?} -- zero points are packed \
             along the output axis, unlike the codes",
            zp.shape,
            spec.width.bits()
        )));
    }
    let need = zp_rows
        .checked_mul(groups_per_row)
        .and_then(|w| w.checked_mul(4))
        .ok_or_else(|| invalid_static("zero-point byte count overflows usize"))?;
    if zp.payload.len() != need {
        return Err(invalid(format_args!(
            "weight_zero_point is {} byte(s); {zp_rows}x{groups_per_row} I32 words need {need}",
            zp.payload.len()
        )));
    }

    let bits = spec.width.bits();
    let mask: u32 = (1u32 << bits) - 1;
    let mut values = crate::try_vec::<i16>(entries)?;
    for o in 0..out_features {
        let row = o / per_word;
        let lane = (o % per_word) as u32;
        for g in 0..groups_per_row {
            let at = (row * groups_per_row + g) * 4;
            let word = u32::from_le_bytes(
                zp.payload[at..at + 4]
                    .try_into()
                    .expect("four bytes, length checked above"),
            );
            let raw = (word >> (lane * bits)) & mask;
            values.push(rebias_zero_point(spec.width, raw)?);
        }
    }
    Ok(ZeroPoints::PerGroup(values))
}

fn decode_scales(bytes: &[u8], dtype: ScaleDtype, entries: usize) -> Result<ScaleValues> {
    Ok(match dtype {
        ScaleDtype::Bf16 | ScaleDtype::F16 => {
            let mut v = crate::try_vec::<u16>(entries)?;
            v.extend(
                bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]])),
            );
            match dtype {
                ScaleDtype::Bf16 => ScaleValues::Bf16(v),
                _ => ScaleValues::F16(v),
            }
        }
        ScaleDtype::F32 => {
            let mut v = crate::try_vec::<f32>(entries)?;
            v.extend(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
            ScaleValues::F32(v)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The independent oracle: document 03's equation and the pinned decode,
    /// written out over the *source* bytes without touching `import` or
    /// `AffineTensor`. A test that reconstructed through the code under test
    /// would be checking it against itself.
    fn oracle(spec: &PackQuantizedSpec, src: SourceTensors<'_>, o: usize, k: usize) -> f32 {
        let (_, in_features) = src.logical;
        let per_word = 32 / spec.width.bits() as usize;
        let packed_columns = in_features.div_ceil(per_word);
        let bits = spec.width.bits();
        let word_index = o * packed_columns + k / per_word;
        let word = u32::from_le_bytes(
            src.packed[word_index * 4..word_index * 4 + 4]
                .try_into()
                .unwrap(),
        );
        let lane = (k % per_word) as u32;
        let raw = (word >> (lane * bits)) & ((1u32 << bits) - 1);
        let q = raw as i32 - (1i32 << (bits - 1));
        let groups = match spec.granularity {
            Granularity::Channel => 1,
            Granularity::Group { size } => in_features.div_ceil(size as usize),
        };
        let group = match spec.granularity {
            Granularity::Channel => 0,
            Granularity::Group { size } => k / size as usize,
        };
        let index = o * groups + group;
        let scale = match src.scale_dtype {
            ScaleDtype::Bf16 => crate::bf16::bf16_bits_to_f32(u16::from_le_bytes(
                src.scale[index * 2..index * 2 + 2].try_into().unwrap(),
            )),
            ScaleDtype::F16 => crate::scale::f16_bits_to_f32(u16::from_le_bytes(
                src.scale[index * 2..index * 2 + 2].try_into().unwrap(),
            )),
            ScaleDtype::F32 => {
                f32::from_le_bytes(src.scale[index * 4..index * 4 + 4].try_into().unwrap())
            }
        };
        // The zero point, decoded from the source payload along the **output**
        // axis -- the pinned `unpack_from_int32(packed_dim=0)`, written here
        // over the raw bytes so that this stays independent of `import`.
        let z = match src.zero_point {
            None => 0i32,
            Some(zp) => {
                let row = o / per_word;
                let at = (row * groups + group) * 4;
                let word = u32::from_le_bytes(zp.payload[at..at + 4].try_into().unwrap());
                let lane = (o % per_word) as u32;
                let raw = (word >> (lane * bits)) & ((1u32 << bits) - 1);
                raw as i32 - (1i32 << (bits - 1))
            }
        };
        (q - z) as f32 * scale
    }

    /// Pack biased-unsigned codes the way the source writer does.
    fn pack(
        width: IntWidth,
        rows: usize,
        columns: usize,
        code: impl Fn(usize, usize) -> i32,
    ) -> Vec<u8> {
        let bits = width.bits();
        let per_word = 32 / bits as usize;
        let packed_columns = columns.div_ceil(per_word);
        let mut out = vec![0u8; rows * packed_columns * 4];
        for o in 0..rows {
            for k in 0..columns {
                let raw = (code(o, k) + (1i32 << (bits - 1))) as u32;
                let word_index = o * packed_columns + k / per_word;
                let lane = (k % per_word) as u32;
                let at = word_index * 4;
                let mut word = u32::from_le_bytes(out[at..at + 4].try_into().unwrap());
                word |= raw << (lane * bits);
                out[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        out
    }

    fn packed_shape(width: IntWidth, rows: usize, columns: usize) -> Vec<u64> {
        let per_word = 32 / width.bits() as usize;
        vec![rows as u64, columns.div_ceil(per_word) as u64]
    }

    fn scale_shape(granularity: Granularity, rows: usize, columns: usize) -> Vec<u64> {
        let groups = match granularity {
            Granularity::Channel => 1,
            Granularity::Group { size } => columns.div_ceil(size as usize),
        };
        vec![rows as u64, groups as u64]
    }

    fn bf16_scales(entries: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for i in 0..entries {
            // Distinct, positive, exactly representable in BF16.
            let v = 1.0f32 + (i % 7) as f32 * 0.25;
            out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        out
    }

    /// **Every INT8 code in every lane**, against the oracle.
    ///
    /// Independent review found the earlier version short of the claim it made:
    /// three row shifts over four lanes reached 768 of the 1,024 code/lane
    /// pairs, and the tracker counted codes alone, so the gap was invisible.
    /// The row count is now the lane count, and the tracker is two-dimensional.
    #[test]
    fn all_256_int8_codes_round_trip_in_every_lane() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::Symmetric,
        };
        let per_word = spec.values_per_word();
        // One row per lane offset: row `o` puts code `c` in lane
        // `(c - o) mod per_word`, so the rows together cover every pair.
        let (rows, columns) = (per_word, 256usize);
        let packed = pack(IntWidth::Int8, rows, columns, |o, k| {
            ((k + o) % 256) as i32 - 128
        });
        let scale = bf16_scales(rows * columns.div_ceil(32));
        let src = SourceTensors {
            packed: &packed,
            packed_shape: &packed_shape(IntWidth::Int8, rows, columns),
            scale: &scale,
            scale_shape: &scale_shape(Granularity::Group { size: 32 }, rows, columns),
            scale_dtype: ScaleDtype::Bf16,
            zero_point: None,
            logical: (rows, columns),
        };
        let tensor = import(&spec, src).unwrap();
        let mut seen = vec![[false; 4]; 256];
        for o in 0..rows {
            let row = tensor.reconstruct_row(o).unwrap();
            for (k, value) in row.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    oracle(&spec, src, o, k).to_bits(),
                    "row {o} column {k}"
                );
                seen[(tensor.code(o, k).unwrap() + 128) as usize][k % per_word] = true;
            }
        }
        let covered: usize = seen.iter().flatten().filter(|s| **s).count();
        assert_eq!(
            covered,
            256 * per_word,
            "every one of the {} INT8 code/lane pairs must be exercised",
            256 * per_word
        );
    }

    /// **Every INT4 code in every one of the eight lanes**, with the same
    /// two-dimensional coverage assertion.
    #[test]
    fn all_16_int4_codes_round_trip_in_every_lane() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int4,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::Symmetric,
        };
        let per_word = spec.values_per_word();
        let (rows, columns) = (per_word, 128usize);
        let packed = pack(IntWidth::Int4, rows, columns, |o, k| {
            ((k + o) % 16) as i32 - 8
        });
        let scale = bf16_scales(rows * columns.div_ceil(32));
        let src = SourceTensors {
            packed: &packed,
            packed_shape: &packed_shape(IntWidth::Int4, rows, columns),
            scale: &scale,
            scale_shape: &scale_shape(Granularity::Group { size: 32 }, rows, columns),
            scale_dtype: ScaleDtype::Bf16,
            zero_point: None,
            logical: (rows, columns),
        };
        let tensor = import(&spec, src).unwrap();
        let mut seen = [[false; 8]; 16];
        for o in 0..rows {
            let row = tensor.reconstruct_row(o).unwrap();
            for (k, value) in row.iter().enumerate() {
                assert_eq!(value.to_bits(), oracle(&spec, src, o, k).to_bits());
                seen[(tensor.code(o, k).unwrap() + 8) as usize][k % per_word] = true;
            }
        }
        let covered: usize = seen.iter().flatten().filter(|s| **s).count();
        assert_eq!(
            covered,
            16 * per_word,
            "every one of the {} INT4 code/lane pairs must be exercised",
            16 * per_word
        );
    }

    /// Group tails, per-channel granularity, and all three scale encodings.
    #[test]
    fn tails_granularities_and_every_scale_encoding_agree_with_the_oracle() {
        for width in [IntWidth::Int4, IntWidth::Int8] {
            for granularity in [
                Granularity::Channel,
                Granularity::Group { size: 32 },
                Granularity::Group { size: 128 },
            ] {
                // Not a multiple of 32, 128, 4 or 8: the tail is partial for
                // both the groups and the packed words.
                for columns in [1usize, 33, 130, 200] {
                    for dtype in ScaleDtype::ALL.iter().copied() {
                        let rows = 2;
                        let spec = PackQuantizedSpec {
                            width,
                            granularity,
                            zero_points: ZeroPointSource::Symmetric,
                        };
                        let (lo, hi) = width.code_range();
                        let span = (hi - lo + 1) as usize;
                        let packed = pack(width, rows, columns, |o, k| {
                            lo + ((k * 3 + o) % span) as i32
                        });
                        let groups = match granularity {
                            Granularity::Channel => 1,
                            Granularity::Group { size } => columns.div_ceil(size as usize),
                        };
                        let entries = rows * groups;
                        let scale: Vec<u8> = match dtype {
                            ScaleDtype::Bf16 => bf16_scales(entries),
                            ScaleDtype::F16 => (0..entries)
                                .flat_map(|i| {
                                    // 1.0, 1.5, 2.0 ... exact in binary16.
                                    let bits: u16 = 0x3C00 + (i % 5) as u16 * 0x0200;
                                    bits.to_le_bytes()
                                })
                                .collect(),
                            ScaleDtype::F32 => (0..entries)
                                .flat_map(|i| (1.0f32 + i as f32 * 0.125).to_le_bytes())
                                .collect(),
                        };
                        let src = SourceTensors {
                            packed: &packed,
                            packed_shape: &packed_shape(width, rows, columns),
                            scale: &scale,
                            scale_shape: &scale_shape(granularity, rows, columns),
                            scale_dtype: dtype,
                            zero_point: None,
                            logical: (rows, columns),
                        };
                        let tensor = import(&spec, src).unwrap();
                        for o in 0..rows {
                            let row = tensor.reconstruct_row(o).unwrap();
                            assert_eq!(row.len(), columns);
                            for (k, value) in row.iter().enumerate() {
                                assert_eq!(
                                    value.to_bits(),
                                    oracle(&spec, src, o, k).to_bits(),
                                    "{width:?} {granularity:?} {columns} cols {dtype:?} at ({o},{k})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_mis_sized_or_mis_declared_or_bad_scale_source_is_refused() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::Symmetric,
        };
        let packed = pack(IntWidth::Int8, 2, 64, |_, _| 1);
        let scale = bf16_scales(4);
        let ps = packed_shape(IntWidth::Int8, 2, 64);
        let ss = scale_shape(Granularity::Group { size: 32 }, 2, 64);
        let good = SourceTensors {
            packed: &packed,
            packed_shape: &ps,
            scale: &scale,
            scale_shape: &ss,
            scale_dtype: ScaleDtype::Bf16,
            zero_point: None,
            logical: (2, 64),
        };
        assert!(import(&spec, good).is_ok());

        // A spec that declares zero points with no payload to read.
        let asymmetric = PackQuantizedSpec {
            zero_points: ZeroPointSource::PackedAlongOutput,
            ..spec
        };
        assert!(matches!(
            import(&asymmetric, good),
            Err(Error::InvalidArtifact { .. })
        ));

        // Packed payload too short and too long for the logical shape.
        for logical in [(2usize, 65usize), (2, 60), (3, 64), (0, 64), (2, 0)] {
            assert!(
                import(&spec, SourceTensors { logical, ..good }).is_err(),
                "{logical:?} must be refused"
            );
        }
        // Scale payload of the wrong length.
        let short = bf16_scales(3);
        assert!(
            import(
                &spec,
                SourceTensors {
                    scale: &short,
                    ..good
                }
            )
            .is_err()
        );
        // Nonfinite, zero and negative scales each rejected by the canonical
        // validator, reached through the importer.
        for bits in [0x7F80u16, 0x0000, 0xBF80] {
            let bad: Vec<u8> = (0..4).flat_map(|_| bits.to_le_bytes()).collect();
            assert!(
                import(
                    &spec,
                    SourceTensors {
                        scale: &bad,
                        ..good
                    }
                )
                .is_err(),
                "scale bits {bits:#06x} must be refused"
            );
        }
    }

    /// Byte counts do not establish axes. Every one of these has exactly the
    /// right number of bytes and the wrong shape.
    #[test]
    fn declared_shapes_that_are_byte_compatible_but_axis_incompatible_are_refused() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::Symmetric,
        };
        let (rows, columns) = (2usize, 64usize);
        let packed = pack(IntWidth::Int8, rows, columns, |_, k| (k % 200) as i32 - 100);
        let scale = bf16_scales(rows * columns / 32);
        let ps = packed_shape(IntWidth::Int8, rows, columns); // [2, 16]
        let ss = scale_shape(Granularity::Group { size: 32 }, rows, columns); // [2, 2]
        let good = SourceTensors {
            packed: &packed,
            packed_shape: &ps,
            scale: &scale,
            scale_shape: &ss,
            scale_dtype: ScaleDtype::Bf16,
            zero_point: None,
            logical: (rows, columns),
        };
        assert!(import(&spec, good).is_ok(), "the control must import");

        // Transposed packed axes, same 128 bytes.
        for bad in [&[16u64, 2][..], &[32, 1][..], &[2, 16, 1][..], &[32][..]] {
            assert!(
                import(
                    &spec,
                    SourceTensors {
                        packed_shape: bad,
                        ..good
                    }
                )
                .is_err(),
                "packed shape {bad:?} must be refused"
            );
        }
        // Transposed or regrouped scale axes, same four values.
        for bad in [&[1u64, 4][..], &[4, 1][..], &[2, 2, 1][..], &[2][..]] {
            assert!(
                import(
                    &spec,
                    SourceTensors {
                        scale_shape: bad,
                        ..good
                    }
                )
                .is_err(),
                "scale shape {bad:?} must be refused"
            );
        }
        // Per-channel sources may serialize the scale as [out] or [out, 1].
        let channel = PackQuantizedSpec {
            granularity: Granularity::Channel,
            ..spec
        };
        let one = bf16_scales(rows);
        for shape in [&[2u64, 1][..], &[2][..]] {
            assert!(
                import(
                    &channel,
                    SourceTensors {
                        scale: &one,
                        scale_shape: shape,
                        ..good
                    }
                )
                .is_ok(),
                "per-channel scale shape {shape:?} must be accepted"
            );
        }
    }

    #[test]
    fn weight_shape_is_two_positive_little_endian_i64_values() {
        let mut ok = 8192i64.to_le_bytes().to_vec();
        ok.extend_from_slice(&5376i64.to_le_bytes());
        assert_eq!(decode_weight_shape(&ok).unwrap(), (8192, 5376));
        assert!(decode_weight_shape(&ok[..15]).is_err());
        let mut zero = 0i64.to_le_bytes().to_vec();
        zero.extend_from_slice(&5376i64.to_le_bytes());
        assert!(decode_weight_shape(&zero).is_err());
        let mut negative = (-1i64).to_le_bytes().to_vec();
        negative.extend_from_slice(&5376i64.to_le_bytes());
        assert!(decode_weight_shape(&negative).is_err());
    }

    /// The scale dtype comes from the tensor header, never from the config.
    #[test]
    fn the_scale_dtype_is_read_from_the_tensor_header() {
        assert_eq!(scale_dtype_of(Dtype::Bf16).unwrap(), ScaleDtype::Bf16);
        assert_eq!(scale_dtype_of(Dtype::F16).unwrap(), ScaleDtype::F16);
        assert_eq!(scale_dtype_of(Dtype::F32).unwrap(), ScaleDtype::F32);
        for other in [Dtype::I32, Dtype::U8, Dtype::I64, Dtype::F64] {
            assert!(scale_dtype_of(other).is_err());
        }
    }

    /// A header-declared size larger than this machine can allocate must be a
    /// typed `CapacityExceeded`, never the infallible allocator's abort.
    ///
    /// The sizes an importer allocates come from an artifact's own header, so
    /// this is the one place a corrupt or hostile file reaches the allocator.
    /// The refusal happens before the allocation, because the payload length is
    /// checked against the logical shape first.
    #[test]
    fn an_implausibly_large_logical_shape_is_refused_before_it_allocates() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::Symmetric,
        };
        let packed = pack(IntWidth::Int8, 1, 4, |_, _| 0);
        let scale = bf16_scales(1);
        // A logical shape that would need exabytes. The payload-length check
        // rejects it against the four bytes actually present, so nothing is
        // ever asked of the allocator.
        for logical in [(usize::MAX, 32usize), (1, usize::MAX), (1 << 40, 1 << 40)] {
            let e = import(
                &spec,
                SourceTensors {
                    packed: &packed,
                    packed_shape: &[1, 1],
                    scale: &scale,
                    scale_shape: &[1, 1],
                    scale_dtype: ScaleDtype::Bf16,
                    zero_point: None,
                    logical,
                },
            )
            .unwrap_err();
            assert!(
                matches!(
                    e,
                    Error::InvalidArtifact { .. } | Error::CapacityExceeded { .. }
                ),
                "{logical:?} gave {e}"
            );
        }
    }

    // --- asymmetric sources: zero points packed along the output axis -------

    /// Pack zero points the way the pinned library's `compress` path does.
    ///
    /// Deliberately **procedural**, not the closed-form index [`import`] uses:
    /// transpose, pad the packed axis, pack lanes along it, transpose back --
    /// `pack_to_int32(value, bits, packed_dim=0)` step by step. Two expressions
    /// of one convention, so a wrong index in the importer cannot agree with a
    /// wrong index in the fixture. Task 0022's lesson: a transcription is
    /// independent of the implementation, not of the reader, and writing the
    /// fixture with the reader's own formula would prove only that the formula
    /// equals itself.
    fn pack_zero_points(width: IntWidth, zeros: &[Vec<i16>], groups: usize) -> Vec<u8> {
        let bits = width.bits();
        let per_word = 32 / bits as usize;
        let out_features = zeros.len();
        // 1. transpose to [groups][out_features], biased to unsigned.
        let mut transposed = vec![vec![0u32; out_features]; groups];
        for (o, row) in zeros.iter().enumerate() {
            assert_eq!(row.len(), groups);
            for (g, z) in row.iter().enumerate() {
                transposed[g][o] = (*z as i32 + width.bias()) as u32;
            }
        }
        // 2. pad the column axis to a whole number of words, with zeros.
        let padded = out_features.div_ceil(per_word) * per_word;
        for row in &mut transposed {
            row.resize(padded, 0);
        }
        // 3. pack `per_word` columns into each word, lane l at bit l * bits.
        let words = padded / per_word;
        let mut packed = vec![vec![0u32; words]; groups];
        for (g, row) in transposed.iter().enumerate() {
            for w in 0..words {
                let mut word = 0u32;
                for l in 0..per_word {
                    word |= row[w * per_word + l] << (l as u32 * bits);
                }
                packed[g][w] = word;
            }
        }
        // 4. transpose back to [words][groups] and serialize little-endian.
        let mut out = Vec::with_capacity(words * groups * 4);
        for w in 0..words {
            for column in &packed {
                out.extend_from_slice(&column[w].to_le_bytes());
            }
        }
        out
    }

    fn zero_point_shape(width: IntWidth, rows: usize, groups: usize) -> Vec<u64> {
        let per_word = 32 / width.bits() as usize;
        vec![rows.div_ceil(per_word) as u64, groups as u64]
    }

    /// **Every INT4 code against every INT4 zero point**, in every lane of the
    /// code word *and* every lane of the zero-point word.
    ///
    /// The two lanes run along different axes -- codes along the input axis,
    /// zero points along the output axis -- so the fixture varies both and the
    /// coverage tracker is three-dimensional. 16 codes x 16 zero points is 256
    /// pairs, and document 03's "full representable code ranges are valid
    /// decoder inputs" means all of them, negative zero points included.
    #[test]
    fn all_256_int4_code_and_zero_point_pairs_reconstruct_exactly() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int4,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::PackedAlongOutput,
        };
        let per_word = spec.values_per_word();
        // 16 output channels covers every zero-point lane twice over; 512
        // columns is 16 groups, and the code varies with both k and o.
        let (rows, columns) = (16usize, 512usize);
        let groups = columns / 32;
        let packed = pack(IntWidth::Int4, rows, columns, |o, k| {
            ((k + o) % 16) as i32 - 8
        });
        let zeros: Vec<Vec<i16>> = (0..rows)
            .map(|o| (0..groups).map(|g| ((o + g) % 16) as i16 - 8).collect())
            .collect();
        let zp = pack_zero_points(IntWidth::Int4, &zeros, groups);
        let zp_shape = zero_point_shape(IntWidth::Int4, rows, groups);
        let scale = bf16_scales(rows * groups);
        let src = SourceTensors {
            packed: &packed,
            packed_shape: &packed_shape(IntWidth::Int4, rows, columns),
            scale: &scale,
            scale_shape: &scale_shape(Granularity::Group { size: 32 }, rows, columns),
            scale_dtype: ScaleDtype::Bf16,
            zero_point: Some(PackedZeroPoints {
                payload: &zp,
                shape: &zp_shape,
            }),
            logical: (rows, columns),
        };
        let tensor = import(&spec, src).unwrap();

        // The zero points the importer recovered are the ones the fixture put
        // in, value by value -- the check the oracle cannot make, because the
        // oracle reads the same payload.
        let ZeroPoints::PerGroup(recovered) = tensor.zero_points() else {
            panic!("an asymmetric import must carry per-group zero points");
        };
        for (o, row) in zeros.iter().enumerate() {
            for (g, z) in row.iter().enumerate() {
                assert_eq!(
                    recovered[o * groups + g],
                    *z,
                    "zero point at output {o} group {g}"
                );
            }
        }

        let mut seen = vec![[[false; 8]; 16]; 16];
        for (o, row_zeros) in zeros.iter().enumerate() {
            let row = tensor.reconstruct_row(o).unwrap();
            for (k, value) in row.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    oracle(&spec, src, o, k).to_bits(),
                    "row {o} column {k}"
                );
                let code = tensor.code(o, k).unwrap();
                let z = row_zeros[k / 32];
                seen[(code + 8) as usize][(z + 8) as usize][k % per_word] = true;
            }
        }
        let pairs: usize = seen
            .iter()
            .flatten()
            .filter(|lanes| lanes.iter().any(|s| *s))
            .count();
        assert_eq!(
            pairs, 256,
            "every one of the 256 code/zero-point pairs must be exercised"
        );
        // And the lanes of one zero-point word carry different values, so a
        // lane mix-up cannot pass by coincidence. Checked on the fixture rather
        // than assumed: a word whose eight lanes agreed would make this whole
        // test blind to the mapping it exists to pin.
        for g in 0..groups {
            let word: Vec<i16> = zeros[..per_word].iter().map(|row| row[g]).collect();
            assert!(
                word.iter().any(|z| *z != word[0]),
                "zero-point word (row 0, group {g}) has identical lanes {word:?}"
            );
        }
    }

    /// INT8 asymmetric: four zero points per word instead of eight, over the
    /// full signed code range.
    #[test]
    fn int8_zero_points_pack_four_to_a_word_and_reconstruct_exactly() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::PackedAlongOutput,
        };
        let (rows, columns) = (12usize, 128usize);
        let groups = columns / 32;
        let packed = pack(IntWidth::Int8, rows, columns, |o, k| {
            ((k * 5 + o * 37) % 256) as i32 - 128
        });
        let zeros: Vec<Vec<i16>> = (0..rows)
            .map(|o| {
                (0..groups)
                    .map(|g| ((o * 19 + g * 7) % 256) as i16 - 128)
                    .collect()
            })
            .collect();
        let zp = pack_zero_points(IntWidth::Int8, &zeros, groups);
        let zp_shape = zero_point_shape(IntWidth::Int8, rows, groups);
        let scale = bf16_scales(rows * groups);
        let src = SourceTensors {
            packed: &packed,
            packed_shape: &packed_shape(IntWidth::Int8, rows, columns),
            scale: &scale,
            scale_shape: &scale_shape(Granularity::Group { size: 32 }, rows, columns),
            scale_dtype: ScaleDtype::Bf16,
            zero_point: Some(PackedZeroPoints {
                payload: &zp,
                shape: &zp_shape,
            }),
            logical: (rows, columns),
        };
        let tensor = import(&spec, src).unwrap();
        let ZeroPoints::PerGroup(recovered) = tensor.zero_points() else {
            panic!("asymmetric");
        };
        for o in 0..rows {
            for g in 0..groups {
                assert_eq!(recovered[o * groups + g], zeros[o][g], "({o},{g})");
            }
        }
        for o in 0..rows {
            for (k, value) in tensor.reconstruct_row(o).unwrap().iter().enumerate() {
                assert_eq!(value.to_bits(), oracle(&spec, src, o, k).to_bits());
            }
        }
    }

    /// The two axes are different axes, and a fixture proves it.
    ///
    /// Zero points constant along the input axis and varying along the output
    /// axis; codes constant along the output axis and varying along the input
    /// axis. A reader that packed the zero points along the input axis -- the
    /// convention the *codes* use, and the only one this module knew before
    /// task 0024 -- reconstructs a different tensor here, because it would read
    /// one word's eight lanes as eight groups of one channel instead of one
    /// group of eight channels.
    #[test]
    fn the_zero_point_axis_is_the_output_axis_and_not_the_input_axis() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int4,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::PackedAlongOutput,
        };
        let (rows, columns) = (8usize, 256usize);
        let groups = columns / 32;
        let packed = pack(IntWidth::Int4, rows, columns, |_, k| (k % 16) as i32 - 8);
        // z depends only on the output channel: z[o] = o - 8, which is the
        // whole negative half plus zero.
        let zeros: Vec<Vec<i16>> = (0..rows).map(|o| vec![o as i16 - 8; groups]).collect();
        let zp = pack_zero_points(IntWidth::Int4, &zeros, groups);
        let zp_shape = zero_point_shape(IntWidth::Int4, rows, groups);
        let scale: Vec<u8> = (0..rows * groups)
            .flat_map(|_| ((1.0f32.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let src = SourceTensors {
            packed: &packed,
            packed_shape: &packed_shape(IntWidth::Int4, rows, columns),
            scale: &scale,
            scale_shape: &scale_shape(Granularity::Group { size: 32 }, rows, columns),
            scale_dtype: ScaleDtype::Bf16,
            zero_point: Some(PackedZeroPoints {
                payload: &zp,
                shape: &zp_shape,
            }),
            logical: (rows, columns),
        };
        let tensor = import(&spec, src).unwrap();
        // With unit scales the reconstruction is exactly `q - z`, and `q` does
        // not depend on the row, so each row is the previous one minus one.
        for o in 0..rows {
            let row = tensor.reconstruct_row(o).unwrap();
            for (k, value) in row.iter().enumerate() {
                let q = (k % 16) as f32 - 8.0;
                assert_eq!(*value, q - (o as f32 - 8.0), "({o},{k})");
            }
        }
        // And every row is distinct, so a reader that assigned one zero point
        // to the whole tensor -- or assigned them along the wrong axis -- would
        // have to produce a repeated row somewhere.
        for o in 1..rows {
            assert_ne!(
                tensor.reconstruct_row(o).unwrap(),
                tensor.reconstruct_row(o - 1).unwrap()
            );
        }
    }

    /// The output axis has a tail, and what a writer padded it with is not read.
    ///
    /// `pack_to_int32` pads the packed axis to a whole word. The padding lanes
    /// are above the logical output channels, so the importer must never touch
    /// them: the test fills them with two different patterns and requires the
    /// same tensor both times. Asserting "the padding is zero" instead would be
    /// a claim about a writer, not a property of this reader.
    #[test]
    fn output_axis_padding_is_never_read() {
        for width in [IntWidth::Int4, IntWidth::Int8] {
            let per_word = 32 / width.bits() as usize;
            // Deliberately not a multiple of `per_word`, for either width.
            let (rows, columns) = (13usize, 64usize);
            assert!(!rows.is_multiple_of(per_word));
            let groups = columns / 32;
            let spec = PackQuantizedSpec {
                width,
                granularity: Granularity::Group { size: 32 },
                zero_points: ZeroPointSource::PackedAlongOutput,
            };
            let (lo, hi) = width.code_range();
            let packed = pack(width, rows, columns, |o, k| {
                lo + ((k + o) % (hi - lo + 1) as usize) as i32
            });
            let zeros: Vec<Vec<i16>> = (0..rows)
                .map(|o| {
                    (0..groups)
                        .map(|g| (lo + ((o + g) as i32 % 3)) as i16)
                        .collect()
                })
                .collect();
            let scale = bf16_scales(rows * groups);
            let zp_shape = zero_point_shape(width, rows, groups);
            let mut imported = Vec::new();
            for fill in [0u32, u32::MAX] {
                let mut zp = pack_zero_points(width, &zeros, groups);
                // Overwrite the padding lanes of the last word row.
                let words = rows.div_ceil(per_word);
                let used_lanes = rows - (words - 1) * per_word;
                let mask_kept: u32 = if used_lanes * width.bits() as usize >= 32 {
                    u32::MAX
                } else {
                    (1u32 << (used_lanes as u32 * width.bits())) - 1
                };
                for g in 0..groups {
                    let at = ((words - 1) * groups + g) * 4;
                    let word = u32::from_le_bytes(zp[at..at + 4].try_into().unwrap());
                    let patched = (word & mask_kept) | (fill & !mask_kept);
                    zp[at..at + 4].copy_from_slice(&patched.to_le_bytes());
                }
                let src = SourceTensors {
                    packed: &packed,
                    packed_shape: &packed_shape(width, rows, columns),
                    scale: &scale,
                    scale_shape: &scale_shape(Granularity::Group { size: 32 }, rows, columns),
                    scale_dtype: ScaleDtype::Bf16,
                    zero_point: Some(PackedZeroPoints {
                        payload: &zp,
                        shape: &zp_shape,
                    }),
                    logical: (rows, columns),
                };
                let tensor = import(&spec, src).unwrap();
                for o in 0..rows {
                    for (k, value) in tensor.reconstruct_row(o).unwrap().iter().enumerate() {
                        assert_eq!(
                            value.to_bits(),
                            oracle(&spec, src, o, k).to_bits(),
                            "{width:?} fill {fill:#x} at ({o},{k})"
                        );
                    }
                }
                imported.push(tensor);
            }
            assert_eq!(
                imported[0], imported[1],
                "{width:?} padding changed the tensor"
            );
            // The fixture has to actually have padding, or it tests nothing.
            assert!(
                mask_has_padding(width, rows),
                "{width:?} fixture has no tail"
            );
        }
    }

    fn mask_has_padding(width: IntWidth, rows: usize) -> bool {
        let per_word = 32 / width.bits() as usize;
        !rows.is_multiple_of(per_word)
    }

    /// Asymmetric sources across every granularity, width, tail and scale
    /// encoding, against the oracle.
    #[test]
    fn asymmetric_tails_granularities_and_scale_encodings_agree_with_the_oracle() {
        for width in [IntWidth::Int4, IntWidth::Int8] {
            for granularity in [
                Granularity::Channel,
                Granularity::Group { size: 32 },
                Granularity::Group { size: 128 },
            ] {
                for columns in [1usize, 33, 130, 200] {
                    for dtype in ScaleDtype::ALL.iter().copied() {
                        // Not a multiple of 4 or 8: the output axis has a tail
                        // for both widths.
                        let rows = 7usize;
                        let spec = PackQuantizedSpec {
                            width,
                            granularity,
                            zero_points: ZeroPointSource::PackedAlongOutput,
                        };
                        let (lo, hi) = width.code_range();
                        let span = (hi - lo + 1) as usize;
                        let packed = pack(width, rows, columns, |o, k| {
                            lo + ((k * 3 + o) % span) as i32
                        });
                        let groups = match granularity {
                            Granularity::Channel => 1,
                            Granularity::Group { size } => columns.div_ceil(size as usize),
                        };
                        let entries = rows * groups;
                        let scale: Vec<u8> = match dtype {
                            ScaleDtype::Bf16 => bf16_scales(entries),
                            ScaleDtype::F16 => (0..entries)
                                .flat_map(|i| {
                                    let bits: u16 = 0x3C00 + (i % 5) as u16 * 0x0200;
                                    bits.to_le_bytes()
                                })
                                .collect(),
                            ScaleDtype::F32 => (0..entries)
                                .flat_map(|i| (1.0f32 + i as f32 * 0.125).to_le_bytes())
                                .collect(),
                        };
                        let zeros: Vec<Vec<i16>> = (0..rows)
                            .map(|o| {
                                (0..groups)
                                    .map(|g| (lo + ((o * 3 + g * 5) % span) as i32) as i16)
                                    .collect()
                            })
                            .collect();
                        let zp = pack_zero_points(width, &zeros, groups);
                        let zp_shape = zero_point_shape(width, rows, groups);
                        let src = SourceTensors {
                            packed: &packed,
                            packed_shape: &packed_shape(width, rows, columns),
                            scale: &scale,
                            scale_shape: &scale_shape(granularity, rows, columns),
                            scale_dtype: dtype,
                            zero_point: Some(PackedZeroPoints {
                                payload: &zp,
                                shape: &zp_shape,
                            }),
                            logical: (rows, columns),
                        };
                        let tensor = import(&spec, src).unwrap();
                        let ZeroPoints::PerGroup(recovered) = tensor.zero_points() else {
                            panic!("asymmetric");
                        };
                        assert_eq!(recovered.len(), entries);
                        for o in 0..rows {
                            let row = tensor.reconstruct_row(o).unwrap();
                            assert_eq!(row.len(), columns);
                            for (k, value) in row.iter().enumerate() {
                                assert_eq!(
                                    value.to_bits(),
                                    oracle(&spec, src, o, k).to_bits(),
                                    "{width:?} {granularity:?} {columns} cols {dtype:?} at ({o},{k})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// A zero-point tensor whose declared shape is byte-compatible and axis-
    /// incompatible is refused -- including the **unpacked** shape, which is
    /// the mistake a reader makes when it forgets that this tensor is packed at
    /// all.
    #[test]
    fn a_mis_declared_or_mis_sized_zero_point_is_refused() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int4,
            granularity: Granularity::Group { size: 32 },
            zero_points: ZeroPointSource::PackedAlongOutput,
        };
        let (rows, columns) = (16usize, 128usize);
        let groups = columns / 32;
        let packed = pack(IntWidth::Int4, rows, columns, |_, k| (k % 16) as i32 - 8);
        let scale = bf16_scales(rows * groups);
        let zeros: Vec<Vec<i16>> = (0..rows).map(|_| vec![1i16; groups]).collect();
        let zp = pack_zero_points(IntWidth::Int4, &zeros, groups);
        let zp_shape = zero_point_shape(IntWidth::Int4, rows, groups);
        assert_eq!(zp_shape, vec![2, 4]);
        let ps = packed_shape(IntWidth::Int4, rows, columns);
        let ss = scale_shape(Granularity::Group { size: 32 }, rows, columns);
        let good = SourceTensors {
            packed: &packed,
            packed_shape: &ps,
            scale: &scale,
            scale_shape: &ss,
            scale_dtype: ScaleDtype::Bf16,
            zero_point: Some(PackedZeroPoints {
                payload: &zp,
                shape: &zp_shape,
            }),
            logical: (rows, columns),
        };
        assert!(import(&spec, good).is_ok(), "the control must import");

        // Transposed, unpacked, flattened and over-ranked declarations, each
        // with the right number of bytes for *something*.
        for bad in [
            &[4u64, 2][..], // transposed
            &[16, 4][..],   // the unpacked [out, groups]
            &[8][..],       // flattened
            &[2, 4, 1][..], // three axes
            &[2, 2][..],    // the scale's group count, halved
        ] {
            assert!(
                import(
                    &spec,
                    SourceTensors {
                        zero_point: Some(PackedZeroPoints {
                            payload: &zp,
                            shape: bad,
                        }),
                        ..good
                    }
                )
                .is_err(),
                "zero-point shape {bad:?} must be refused"
            );
        }
        // Right shape, wrong payload length.
        for len in [zp.len() - 4, zp.len() + 4] {
            let mut wrong = zp.clone();
            wrong.resize(len, 0);
            assert!(
                import(
                    &spec,
                    SourceTensors {
                        zero_point: Some(PackedZeroPoints {
                            payload: &wrong,
                            shape: &zp_shape,
                        }),
                        ..good
                    }
                )
                .is_err(),
                "zero-point payload of {len} byte(s) must be refused"
            );
        }
        // A symmetric declaration that carries a payload anyway.
        let symmetric = PackQuantizedSpec {
            zero_points: ZeroPointSource::Symmetric,
            ..spec
        };
        assert!(matches!(
            import(&symmetric, good),
            Err(Error::InvalidArtifact { .. })
        ));
    }

    /// The header entries and the declared serialization must agree.
    #[test]
    fn a_module_whose_index_disagrees_with_its_declared_zero_points_is_refused() {
        let asym = header_with(&[
            ("m.weight_packed", "I32", &[2, 4]),
            ("m.weight_scale", "BF16", &[2, 1]),
            ("m.weight_shape", "I64", &[2]),
            ("m.weight_zero_point", "I32", &[1, 1]),
        ]);
        let sym = header_with(&[
            ("m.weight_packed", "I32", &[2, 4]),
            ("m.weight_scale", "BF16", &[2, 1]),
            ("m.weight_shape", "I64", &[2]),
        ]);
        // Each read the way it is serialized: accepted.
        let e = source_entries(&asym, "m", ZeroPointSource::PackedAlongOutput).unwrap();
        assert!(e.zero_point.is_some());
        let e = source_entries(&sym, "m", ZeroPointSource::Symmetric).unwrap();
        assert!(e.zero_point.is_none());
        // Each read the other way: refused, in both directions.
        assert!(source_entries(&asym, "m", ZeroPointSource::Symmetric).is_err());
        assert!(source_entries(&sym, "m", ZeroPointSource::PackedAlongOutput).is_err());
        // A zero point of the wrong dtype.
        let wrong = header_with(&[
            ("m.weight_packed", "I32", &[2, 4]),
            ("m.weight_scale", "BF16", &[2, 1]),
            ("m.weight_shape", "I64", &[2]),
            ("m.weight_zero_point", "I8", &[1, 1]),
        ]);
        assert!(source_entries(&wrong, "m", ZeroPointSource::PackedAlongOutput).is_err());
    }

    /// A minimal safetensors header over tensors whose payloads are all zero.
    fn header_with(tensors: &[(&str, &str, &[u64])]) -> Header {
        let bytes_of = |dtype: &str, shape: &[u64]| -> u64 {
            let width = match dtype {
                "I32" | "F32" => 4u64,
                "I64" | "F64" => 8,
                "BF16" | "F16" | "I16" => 2,
                _ => 1,
            };
            shape.iter().product::<u64>() * width
        };
        let mut json = String::from("{");
        let mut at = 0u64;
        for (i, (name, dtype, shape)) in tensors.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let end = at + bytes_of(dtype, shape);
            json.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\
                 \"data_offsets\":[{at},{end}]}}"
            ));
            at = end;
        }
        json.push('}');
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(json.as_bytes());
        let file_len = bytes.len() as u64 + at;
        Header::parse(&bytes, file_len).expect("well-formed header")
    }
}
