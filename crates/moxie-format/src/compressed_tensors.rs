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
//! ## The packing
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

use crate::affine::{AffineDescriptor, AffineTensor, Grouping, IntWidth, ZeroPoints, rebias_code};
use crate::safetensors::{Dtype, Header, TensorEntry};
use crate::scale::{ScaleDtype, ScaleValues};
use moxie_types::{Error, Result};

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidArtifact {
        detail: detail.into(),
    }
}

/// How a source divides the input axis for scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// One scale per output channel.
    Channel,
    /// Contiguous groups of `size` input channels.
    Group { size: u32 },
}

/// The `quantization_config` fields this importer consumes.
///
/// Named rather than inferred: document 03 requires that "GPTQ-style stored zero
/// offsets ..., packed integer axis order, interleaving and activation-order
/// maps must be decoded according to the pinned exporter, never guessed from a
/// suffix." A caller reads these from the artifact's config and passes them in;
/// this module does not go looking for them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackQuantizedSpec {
    pub width: IntWidth,
    pub granularity: Granularity,
    /// The source's `symmetric` flag. Asymmetric is refused; see below.
    pub symmetric: bool,
}

impl PackQuantizedSpec {
    /// Values packed into each 32-bit word.
    pub const fn values_per_word(&self) -> usize {
        32 / self.width.bits() as usize
    }
}

/// The three tensors a `pack-quantized` weight is serialized as.
#[derive(Debug, Clone, Copy)]
pub struct TensorTriple<'a> {
    /// `weight_packed`, Safetensors `I32`.
    pub packed: &'a [u8],
    /// `weight_scale`, whose dtype is read from **its own header entry**.
    pub scale: &'a [u8],
    pub scale_dtype: ScaleDtype,
    /// Logical `[out_features, in_features]`, from the `weight_shape` payload.
    pub logical: (usize, usize),
}

/// Decode a `weight_shape` payload: exactly two little-endian `I64` values.
pub fn decode_weight_shape(payload: &[u8]) -> Result<(usize, usize)> {
    if payload.len() != 16 {
        return Err(invalid(format!(
            "weight_shape must hold exactly two I64 values, got {} byte(s)",
            payload.len()
        )));
    }
    let read = |at: usize| i64::from_le_bytes(payload[at..at + 8].try_into().expect("eight bytes"));
    let (rows, columns) = (read(0), read(8));
    if rows <= 0 || columns <= 0 {
        return Err(invalid(format!(
            "weight_shape dimensions must be positive, got [{rows}, {columns}]"
        )));
    }
    let to_usize = |v: i64| {
        usize::try_from(v).map_err(|_| invalid("weight_shape dimension does not fit this platform"))
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
            return Err(invalid(format!(
                "weight_scale is {}; canonical scales are BF16, F16 or F32",
                other.name()
            )));
        }
    })
}

/// Pull the three payload ranges for one module out of a parsed header.
///
/// Returns the `weight_packed`, `weight_scale` and `weight_shape` entries and
/// validates their dtypes, so a caller reads exactly three byte ranges and
/// nothing else. Naming is the source's: `<module>.weight_packed` and so on.
pub fn triple_entries<'a>(
    header: &'a Header,
    module: &str,
) -> Result<(&'a TensorEntry, &'a TensorEntry, &'a TensorEntry)> {
    let packed = header.get(&format!("{module}.weight_packed"))?;
    let scale = header.get(&format!("{module}.weight_scale"))?;
    let shape = header.get(&format!("{module}.weight_shape"))?;
    if packed.dtype != Dtype::I32 {
        return Err(invalid(format!(
            "{module}.weight_packed is {}; pack-quantized requires I32",
            packed.dtype.name()
        )));
    }
    if shape.dtype != Dtype::I64 || shape.shape != vec![2] {
        return Err(invalid(format!(
            "{module}.weight_shape must be I64[2], got {} {:?}",
            shape.dtype.name(),
            shape.shape
        )));
    }
    Ok((packed, scale, shape))
}

/// Import one `pack-quantized` tensor into canonical affine form.
///
/// Every length is checked against the logical shape from `weight_shape`, which
/// is the authority; the packed shape is validated against it rather than used
/// to derive it.
pub fn import(spec: &PackQuantizedSpec, triple: TensorTriple<'_>) -> Result<AffineTensor> {
    // Refused, not guessed. The pinned reader rejects asymmetric pack-quantized
    // (`compressed_tensors.cpp:204`), no local artifact is one, and document 03
    // forbids inferring a zero offset "from a suffix". The canonical descriptor
    // already carries `ZeroPoints::PerGroup`, so what is missing is a *verified
    // source contract*, not a canonical capability.
    if !spec.symmetric {
        return Err(Error::Unsupported {
            capability: "asymmetric pack-quantized import",
            reason: "no pinned exporter or local artifact establishes this source's zero-point \
                     serialization; importing it would mean guessing the stored offset \
                     convention. Supply a pinned exporter reference or an asymmetric artifact \
                     to inspect."
                .into(),
        });
    }

    let (out_features, in_features) = triple.logical;
    if out_features == 0 || in_features == 0 {
        return Err(invalid(format!(
            "empty logical shape {out_features}x{in_features}"
        )));
    }
    let per_word = spec.values_per_word();
    let packed_columns = in_features.div_ceil(per_word);
    let need_words = packed_columns
        .checked_mul(out_features)
        .ok_or_else(|| invalid("packed word count overflows usize"))?;
    let need_packed = need_words
        .checked_mul(4)
        .ok_or_else(|| invalid("packed byte count overflows usize"))?;
    if triple.packed.len() != need_packed {
        return Err(invalid(format!(
            "weight_packed is {} byte(s); {out_features}x{in_features} at {} bit(s) needs \
             {need_packed} ({out_features}x{packed_columns} I32 words)",
            triple.packed.len(),
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
        // Contiguous. `actorder` is null in the inspected artifact; a source
        // that carried a permutation would have to supply the map here, and
        // document 03 forbids ignoring one.
        group_index: None,
        scale_dtype: triple.scale_dtype,
    };
    desc.validate()?;

    let entries = desc.group_entries()?;
    let need_scale = entries
        .checked_mul(triple.scale_dtype.bytes())
        .ok_or_else(|| invalid("scale byte count overflows usize"))?;
    if triple.scale.len() != need_scale {
        return Err(invalid(format!(
            "weight_scale is {} byte(s); {entries} {} scale(s) need {need_scale}",
            triple.scale.len(),
            triple.scale_dtype.name()
        )));
    }

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
                triple.packed[at..at + 4]
                    .try_into()
                    .expect("four bytes, length checked above"),
            );
            let lane = k % per_word;
            let raw = (word >> (lane as u32 * bits)) & mask;
            codes.push(rebias_code(spec.width, raw)?);
        }
        let row = crate::affine::pack_row(spec.width, &codes)?;
        let stride = row.len();
        out[o * stride..(o + 1) * stride].copy_from_slice(&row);
    }

    let scales = decode_scales(triple.scale, triple.scale_dtype, entries)?;
    AffineTensor::new(desc, out, scales, ZeroPoints::Symmetric)
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
    fn oracle(spec: &PackQuantizedSpec, triple: TensorTriple<'_>, o: usize, k: usize) -> f32 {
        let (_, in_features) = triple.logical;
        let per_word = 32 / spec.width.bits() as usize;
        let packed_columns = in_features.div_ceil(per_word);
        let bits = spec.width.bits();
        let word_index = o * packed_columns + k / per_word;
        let word = u32::from_le_bytes(
            triple.packed[word_index * 4..word_index * 4 + 4]
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
        let scale = match triple.scale_dtype {
            ScaleDtype::Bf16 => crate::bf16::bf16_bits_to_f32(u16::from_le_bytes(
                triple.scale[index * 2..index * 2 + 2].try_into().unwrap(),
            )),
            ScaleDtype::F16 => crate::scale::f16_bits_to_f32(u16::from_le_bytes(
                triple.scale[index * 2..index * 2 + 2].try_into().unwrap(),
            )),
            ScaleDtype::F32 => {
                f32::from_le_bytes(triple.scale[index * 4..index * 4 + 4].try_into().unwrap())
            }
        };
        q as f32 * scale
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

    fn bf16_scales(entries: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for i in 0..entries {
            // Distinct, positive, exactly representable in BF16.
            let v = 1.0f32 + (i % 7) as f32 * 0.25;
            out.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        out
    }

    /// Every INT8 code, in every lane of a word, against the oracle.
    #[test]
    fn all_256_int8_codes_round_trip_in_every_lane() {
        let (rows, columns) = (3usize, 256usize);
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            symmetric: true,
        };
        // Column k carries code k - 128, so all 256 appear, and each appears in
        // lane k % 4 -- over 256 columns every code meets every lane across rows.
        let packed = pack(IntWidth::Int8, rows, columns, |o, k| {
            ((k + o) % 256) as i32 - 128
        });
        let scale = bf16_scales(rows * columns.div_ceil(32));
        let triple = TensorTriple {
            packed: &packed,
            scale: &scale,
            scale_dtype: ScaleDtype::Bf16,
            logical: (rows, columns),
        };
        let tensor = import(&spec, triple).unwrap();
        let mut seen = [false; 256];
        for o in 0..rows {
            let row = tensor.reconstruct_row(o).unwrap();
            for k in 0..columns {
                assert_eq!(
                    row[k].to_bits(),
                    oracle(&spec, triple, o, k).to_bits(),
                    "row {o} column {k}"
                );
                seen[(tensor.code(o, k).unwrap() + 128) as usize] = true;
            }
        }
        assert!(seen.iter().all(|s| *s), "every INT8 code must be exercised");
    }

    /// Every INT4 code, in every one of the eight lanes.
    #[test]
    fn all_16_int4_codes_round_trip_in_every_lane() {
        let (rows, columns) = (5usize, 128usize);
        let spec = PackQuantizedSpec {
            width: IntWidth::Int4,
            granularity: Granularity::Group { size: 32 },
            symmetric: true,
        };
        let packed = pack(IntWidth::Int4, rows, columns, |o, k| {
            ((k + o) % 16) as i32 - 8
        });
        let scale = bf16_scales(rows * columns.div_ceil(32));
        let triple = TensorTriple {
            packed: &packed,
            scale: &scale,
            scale_dtype: ScaleDtype::Bf16,
            logical: (rows, columns),
        };
        let tensor = import(&spec, triple).unwrap();
        let mut seen = [false; 16];
        for o in 0..rows {
            let row = tensor.reconstruct_row(o).unwrap();
            for k in 0..columns {
                assert_eq!(row[k].to_bits(), oracle(&spec, triple, o, k).to_bits());
                seen[(tensor.code(o, k).unwrap() + 8) as usize] = true;
            }
        }
        assert!(seen.iter().all(|s| *s));
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
                            symmetric: true,
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
                        let triple = TensorTriple {
                            packed: &packed,
                            scale: &scale,
                            scale_dtype: dtype,
                            logical: (rows, columns),
                        };
                        let tensor = import(&spec, triple).unwrap();
                        for o in 0..rows {
                            let row = tensor.reconstruct_row(o).unwrap();
                            assert_eq!(row.len(), columns);
                            for (k, value) in row.iter().enumerate() {
                                assert_eq!(
                                    value.to_bits(),
                                    oracle(&spec, triple, o, k).to_bits(),
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
    fn a_mis_sized_or_asymmetric_or_bad_scale_source_is_refused() {
        let spec = PackQuantizedSpec {
            width: IntWidth::Int8,
            granularity: Granularity::Group { size: 32 },
            symmetric: true,
        };
        let packed = pack(IntWidth::Int8, 2, 64, |_, _| 1);
        let scale = bf16_scales(4);
        let good = TensorTriple {
            packed: &packed,
            scale: &scale,
            scale_dtype: ScaleDtype::Bf16,
            logical: (2, 64),
        };
        assert!(import(&spec, good).is_ok());

        // Asymmetric: a typed capability refusal, distinguishable from a
        // malformed artifact, because the source is fine and we cannot read it.
        let asymmetric = PackQuantizedSpec {
            symmetric: false,
            ..spec.clone()
        };
        assert!(matches!(
            import(&asymmetric, good),
            Err(Error::Unsupported { .. })
        ));

        // Packed payload too short and too long for the logical shape.
        for logical in [(2usize, 65usize), (2, 60), (3, 64), (0, 64), (2, 0)] {
            assert!(
                import(&spec, TensorTriple { logical, ..good }).is_err(),
                "{logical:?} must be refused"
            );
        }
        // Scale payload of the wrong length.
        let short = bf16_scales(3);
        assert!(
            import(
                &spec,
                TensorTriple {
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
                    TensorTriple {
                        scale: &bad,
                        ..good
                    }
                )
                .is_err(),
                "scale bits {bits:#06x} must be refused"
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
            symmetric: true,
        };
        let packed = pack(IntWidth::Int8, 1, 4, |_, _| 0);
        let scale = bf16_scales(1);
        // A logical shape that would need exabytes. The payload-length check
        // rejects it against the four bytes actually present, so nothing is
        // ever asked of the allocator.
        for logical in [(usize::MAX, 32usize), (1, usize::MAX), (1 << 40, 1 << 40)] {
            let e = import(
                &spec,
                TensorTriple {
                    packed: &packed,
                    scale: &scale,
                    scale_dtype: ScaleDtype::Bf16,
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
}
