//! Affine integer v1: one schema, INT4 and INT8 profiles.
//!
//! Document 03, as amended by [ADR 0003], pins the reconstruction:
//!
//! ```text
//! W[o,k] = (Q[o,k] - Z[o,group(k)]) * decode_scale(S[o,group(k)])
//! ```
//!
//! "Subtraction is performed in a sufficiently wide integer type; scale
//! multiplication is evaluated in FP32 in the host reconstruction oracle."
//!
//! [ADR 0003]: ../../../docs/decisions/adr/0003-int4-int8-bf16-weight-family.md
//!
//! ## This is not a renamed NVFP4 decoder
//!
//! The module it replaces decoded E2M1 floats scaled by an E4M3FN block scale
//! and a positive tensor scale. Signed integer codes are different mathematics:
//! there is a subtractive zero point, the code space is uniform rather than
//! exponentially spaced, the group size is a descriptor field rather than a
//! constant 16, and the group boundaries can be non-contiguous when the source
//! carries an activation-order permutation. ADR 0003: "Do not mechanically
//! rename E2M1 values: signed INT4 codes are different mathematics."
//!
//! ## One algorithm, not one packing per method
//!
//! AWQ, AutoRound and GPTQ are quantization *methods*; compressed-tensors and
//! AutoGPTQ-style packing are *serializations*. Symmetric and asymmetric weights
//! both satisfy `(q - z) * s`, with `z` implicit in the symmetric case. Every
//! one of them imports into the descriptor below. There is deliberately no place
//! here to hang a method-specific decode path, because that is how seven engines
//! grow back.

use moxie_types::{Error, Precision, Result};

use crate::scale::{ScaleDtype, ScaleValues};

/// Integer code width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntWidth {
    Int4,
    Int8,
}

impl IntWidth {
    /// Inclusive signed two's-complement code range. **The full range is valid
    /// decoder input.** A quantizer may choose to emit only a symmetric subset
    /// such as `[-127, 127]`; document 03: "that does not authorize a generic
    /// INT8 decoder to reject -128 from a valid source artifact."
    pub const fn code_range(self) -> (i32, i32) {
        match self {
            IntWidth::Int4 => (-8, 7),
            IntWidth::Int8 => (-128, 127),
        }
    }

    /// The offset between an unsigned source code and the canonical signed one:
    /// `q = u - bias`, `z = zu - bias`. Exact rebiasing, not requantization.
    pub const fn bias(self) -> i32 {
        match self {
            IntWidth::Int4 => 8,
            IntWidth::Int8 => 128,
        }
    }

    /// Largest unsigned source code: 15 for INT4, 255 for INT8.
    pub const fn max_unsigned(self) -> u32 {
        match self {
            IntWidth::Int4 => 15,
            IntWidth::Int8 => 255,
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            IntWidth::Int4 => 4,
            IntWidth::Int8 => 8,
        }
    }

    /// Codes stored per byte. INT4 packs two, low nibble first.
    pub const fn codes_per_byte(self) -> usize {
        match self {
            IntWidth::Int4 => 2,
            IntWidth::Int8 => 1,
        }
    }

    /// The stored-element precision this width corresponds to.
    pub const fn precision(self) -> Precision {
        match self {
            IntWidth::Int4 => Precision::Int4,
            IntWidth::Int8 => Precision::Int8,
        }
    }

    pub const fn profile(self) -> &'static str {
        match self {
            IntWidth::Int4 => "affine-int4-v1",
            IntWidth::Int8 => "affine-int8-v1",
        }
    }

    /// Bytes one logical row occupies. Rows are byte-aligned, so an odd INT4
    /// row is padded by one nibble that is never a logical value.
    pub const fn row_stride(self, in_features: usize) -> usize {
        in_features.div_ceil(self.codes_per_byte())
    }

    pub const ALL: &'static [IntWidth] = &[IntWidth::Int4, IntWidth::Int8];
}

/// The closed set of contiguous group sizes.
///
/// Document 03: "Contiguous input-channel groups of 32 or 128". Both appear in
/// the pinned candidate evidence -- symmetric group-128 INT4, asymmetric
/// group-32 INT4 and symmetric group-32 INT8. "One format does not mean one
/// hard-coded group size. A small, closed parameter set belongs in a shared
/// descriptor. It does not authorize arbitrary future formats."
pub const ALLOWED_GROUP_SIZES: &[u32] = &[32, 128];

/// How the input-channel axis is divided into groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Grouping {
    /// Contiguous groups of `size` logical input channels.
    Contiguous { size: u32 },
    /// One group spanning the whole logical row. Represented explicitly rather
    /// than as "group size equals in_features", so per-channel is a stated
    /// choice and never an accident of a divisibility check.
    PerOutputChannel,
}

/// Whether the tensor carries zero points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroPoints {
    /// Implicit zero, no payload. `W = q * s`.
    Symmetric,
    /// One signed value per `(output channel, group)`, row-major.
    ///
    /// Stored as `i16`, wider than either code range on purpose. Document 03:
    /// this permits "exact rebias from signed/unsigned source codes without
    /// artificially clipping the zero point to the code range. This is metadata,
    /// not 16-bit weights."
    PerGroup(Vec<i16>),
}

impl ZeroPoints {
    pub fn is_symmetric(&self) -> bool {
        matches!(self, ZeroPoints::Symmetric)
    }

    fn get(&self, index: usize) -> i32 {
        match self {
            ZeroPoints::Symmetric => 0,
            ZeroPoints::PerGroup(v) => v[index] as i32,
        }
    }
}

/// The closed v1 descriptor from document 03.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineDescriptor {
    pub width: IntWidth,
    /// Output channels. The leading expert axis, when present, is handled by
    /// storing one descriptor per expert slice: the mathematics is identical and
    /// a grouped MoE kernel is qualified separately from a dense one anyway.
    pub out_features: usize,
    /// Logical input channels, before any padding.
    pub in_features: usize,
    pub grouping: Grouping,
    /// Per-input-column group index, when the source's activation ordering makes
    /// grouping non-contiguous after logical columns are restored.
    ///
    /// Document 03: "Never ignore g_idx or a permutation." `None` means the
    /// default contiguous mapping, which is a claim about the source, not a
    /// default to fall back on when the map is inconvenient.
    pub group_index: Option<Vec<u32>>,
    pub scale_dtype: ScaleDtype,
}

impl AffineDescriptor {
    /// Number of groups along one row.
    pub fn groups_per_row(&self) -> Result<usize> {
        match self.grouping {
            Grouping::PerOutputChannel => Ok(1),
            Grouping::Contiguous { size } => {
                if size == 0 {
                    return Err(Error::InvalidArtifact {
                        detail: "group size 0".into(),
                    });
                }
                Ok(self.in_features.div_ceil(size as usize))
            }
        }
    }

    /// Total scale/zero-point entries: one per `(output channel, group)`.
    pub fn group_entries(&self) -> Result<usize> {
        let per_row = self.groups_per_row()?;
        self.out_features
            .checked_mul(per_row)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!(
                    "group table size overflows: {} rows x {per_row} groups",
                    self.out_features
                ),
            })
    }

    /// Bytes of packed codes the whole tensor needs.
    pub fn code_bytes(&self) -> Result<usize> {
        self.width
            .row_stride(self.in_features)
            .checked_mul(self.out_features)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: "packed code length overflows usize".into(),
            })
    }

    /// The group a logical input column belongs to.
    pub fn group_of(&self, k: usize) -> Result<usize> {
        if k >= self.in_features {
            return Err(Error::InvalidArtifact {
                detail: format!("column {k} is outside {} input features", self.in_features),
            });
        }
        match &self.group_index {
            Some(map) => Ok(map[k] as usize),
            None => match self.grouping {
                Grouping::PerOutputChannel => Ok(0),
                Grouping::Contiguous { size } => Ok(k / size as usize),
            },
        }
    }

    /// Validate everything that does not need the payload.
    pub fn validate(&self) -> Result<()> {
        if self.out_features == 0 || self.in_features == 0 {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "empty logical shape {}x{}",
                    self.out_features, self.in_features
                ),
            });
        }
        if let Grouping::Contiguous { size } = self.grouping
            && !ALLOWED_GROUP_SIZES.contains(&size)
        {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "group size {size} is outside the closed set {ALLOWED_GROUP_SIZES:?}; \
                     widening it is an ADR, not an import decision"
                ),
            });
        }
        // Overflow checks before anything indexes with these.
        let groups = self.groups_per_row()?;
        self.group_entries()?;
        self.code_bytes()?;

        if let Some(map) = &self.group_index {
            if map.len() != self.in_features {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "group index map has {} entries for {} input features",
                        map.len(),
                        self.in_features
                    ),
                });
            }
            let mut used = vec![0usize; groups];
            for (k, g) in map.iter().enumerate() {
                let g = *g as usize;
                if g >= groups {
                    return Err(Error::InvalidArtifact {
                        detail: format!(
                            "group index map sends column {k} to group {g}, but there are \
                             only {groups} groups"
                        ),
                    });
                }
                used[g] += 1;
            }
            // Consistency: "Group IDs, lengths and associated scales must be
            // consistent." An unused group means a scale nothing reads, which is
            // how a misread permutation hides.
            if let Some(g) = used.iter().position(|n| *n == 0) {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "group {g} has a scale but no input column maps to it; \
                         the activation-order map and the group table disagree"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Stable identity for artifact and prepared-layout keys.
    ///
    /// Document 03: "Profile/version, bit width, group rule/index-map hash,
    /// scale dtype, zero-point mode, logical shape and source provenance
    /// participate in artifact/prepared-layout identity." Source provenance is
    /// added by the manifest, which owns it; everything intrinsic to the
    /// descriptor is here.
    pub fn identity(&self, zero_points: &ZeroPoints) -> String {
        let group = match self.grouping {
            Grouping::PerOutputChannel => "per-out".to_string(),
            Grouping::Contiguous { size } => format!("g{size}"),
        };
        let map = match &self.group_index {
            None => "contiguous".to_string(),
            Some(m) => format!("gidx:{:016x}", fnv1a64_u32(m)),
        };
        let zp = if zero_points.is_symmetric() {
            "sym"
        } else {
            "asym"
        };
        format!(
            "{}/{}x{}/{group}/{map}/s:{}/{zp}",
            self.width.profile(),
            self.out_features,
            self.in_features,
            self.scale_dtype.name()
        )
    }
}

/// FNV-1a over a `u32` sequence. Small, dependency-free, and only ever used as a
/// cache/identity key -- never as an integrity checksum, which the manifest owns.
fn fnv1a64_u32(v: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// A canonical affine-integer weight tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineTensor {
    desc: AffineDescriptor,
    /// Packed codes, logical row-major, each row byte-aligned.
    codes: Vec<u8>,
    /// One scale per `(output channel, group)`.
    scales: ScaleValues,
    zero_points: ZeroPoints,
}

impl AffineTensor {
    /// Validate a tensor against its descriptor.
    ///
    /// Every length is checked before anything indexes it, and the scale dtype
    /// declared by the descriptor must be the one the payload actually carries:
    /// a mismatch means the header was read from the wrong place.
    pub fn new(
        desc: AffineDescriptor,
        codes: Vec<u8>,
        scales: ScaleValues,
        zero_points: ZeroPoints,
    ) -> Result<Self> {
        desc.validate()?;

        let need_bytes = desc.code_bytes()?;
        if codes.len() != need_bytes {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{} needs {need_bytes} packed byte(s) for {}x{}, got {}",
                    desc.width.profile(),
                    desc.out_features,
                    desc.in_features,
                    codes.len()
                ),
            });
        }

        let entries = desc.group_entries()?;
        if scales.len() != entries {
            return Err(Error::InvalidArtifact {
                detail: format!("expected {entries} scale(s), got {}", scales.len()),
            });
        }
        if scales.dtype() != desc.scale_dtype {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "descriptor declares {} scales but the payload is {}",
                    desc.scale_dtype.name(),
                    scales.dtype().name()
                ),
            });
        }
        scales.validate()?;

        if let ZeroPoints::PerGroup(z) = &zero_points
            && z.len() != entries
        {
            return Err(Error::InvalidArtifact {
                detail: format!("expected {entries} zero point(s), got {}", z.len()),
            });
        }

        Ok(Self {
            desc,
            codes,
            scales,
            zero_points,
        })
    }

    pub fn descriptor(&self) -> &AffineDescriptor {
        &self.desc
    }

    pub fn zero_points(&self) -> &ZeroPoints {
        &self.zero_points
    }

    pub fn scales(&self) -> &ScaleValues {
        &self.scales
    }

    pub fn identity(&self) -> String {
        self.desc.identity(&self.zero_points)
    }

    /// The signed code at `(o, k)`, sign-extended from its stored width.
    ///
    /// INT4: "low nibble is the earlier logical input element, high nibble the
    /// next". Reading the two the other way round transposes every adjacent pair
    /// in the tensor, which no accuracy tolerance would catch.
    pub fn code(&self, o: usize, k: usize) -> Result<i32> {
        if o >= self.desc.out_features || k >= self.desc.in_features {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "({o},{k}) is outside {}x{}",
                    self.desc.out_features, self.desc.in_features
                ),
            });
        }
        let stride = self.desc.width.row_stride(self.desc.in_features);
        Ok(match self.desc.width {
            IntWidth::Int8 => self.codes[o * stride + k] as i8 as i32,
            IntWidth::Int4 => {
                let byte = self.codes[o * stride + k / 2];
                let nibble = if k.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                sign_extend_4(nibble)
            }
        })
    }

    /// Bytes one reconstructed row occupies as f32.
    ///
    /// Document 03 requires bounded preparation: "no repeated full-tensor
    /// dequantization on every token", and dequantization tiles "must remain
    /// bounded and accounted". A caller that reconstructs row by row into a
    /// reused buffer needs exactly this many bytes, whatever the tensor's size.
    pub fn prepared_row_bytes(&self) -> usize {
        self.desc.in_features * core::mem::size_of::<f32>()
    }

    /// Reconstruct one output row into a caller-provided buffer.
    ///
    /// The bounded form: no allocation, and the working set is one row.
    pub fn reconstruct_row_into(&self, o: usize, out: &mut [f32]) -> Result<()> {
        if o >= self.desc.out_features {
            return Err(Error::InvalidArtifact {
                detail: format!("row {o} is outside {} rows", self.desc.out_features),
            });
        }
        if out.len() != self.desc.in_features {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "destination holds {} values for a {}-wide row",
                    out.len(),
                    self.desc.in_features
                ),
            });
        }
        let per_row = self.desc.groups_per_row()?;
        for (k, slot) in out.iter_mut().enumerate() {
            let g = self.desc.group_of(k)?;
            let entry = o * per_row + g;
            let scale = self
                .scales
                .get(entry)
                .ok_or_else(|| Error::InvalidArtifact {
                    detail: format!("no scale for row {o} group {g}"),
                })?;
            let z = self.zero_points.get(entry);
            let q = self.code(o, k)?;
            // Subtraction in i32: wide enough that no (code, zero point) pair in
            // either profile can overflow, which is what document 03 means by "a
            // sufficiently wide integer type". The multiplication is FP32.
            let w = (q - z) as f32 * scale;
            // A scale can be finite, positive and still large enough that the
            // product is not. The second M0 review found this returning
            // `Ok([inf])` for a `f32::MAX` scale: a reconstruction that
            // overflows is an invalid artifact, not a weight. Readers "reject
            // ... NaN scales, incompatible dimensions" (document 03), and an
            // infinite reconstructed weight belongs in the same list -- it would
            // reach a kernel and poison a whole row's accumulation.
            if !w.is_finite() {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "reconstruction overflows at ({o},{k}): ({q} - {z}) * {scale} = {w}"
                    ),
                });
            }
            *slot = w;
        }
        Ok(())
    }

    /// Reconstruct one output row.
    pub fn reconstruct_row(&self, o: usize) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.desc.in_features];
        self.reconstruct_row_into(o, &mut out)?;
        Ok(out)
    }

    /// Reconstruct the whole tensor, row-major.
    ///
    /// The oracle for tests and for import verification. Runtime paths use the
    /// bounded row form; document 03 forbids materialising a whole model as
    /// BF16 to make a kernel convenient.
    pub fn reconstruct(&self) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.desc.out_features * self.desc.in_features];
        for o in 0..self.desc.out_features {
            let start = o * self.desc.in_features;
            let end = start + self.desc.in_features;
            self.reconstruct_row_into(o, &mut out[start..end])?;
        }
        Ok(out)
    }
}

/// Sign-extend a 4-bit two's-complement nibble to `i32`.
pub const fn sign_extend_4(nibble: u8) -> i32 {
    let n = (nibble & 0x0F) as i32;
    if n >= 8 { n - 16 } else { n }
}

/// Pack signed codes for one row, low nibble first for INT4.
///
/// Provided so importers and tests build the canonical layout the same way. A
/// code outside the width's range is an error rather than a silent truncation.
pub fn pack_row(width: IntWidth, codes: &[i32]) -> Result<Vec<u8>> {
    let (lo, hi) = width.code_range();
    for (k, c) in codes.iter().enumerate() {
        if *c < lo || *c > hi {
            return Err(Error::InvalidArtifact {
                detail: format!("code {c} at column {k} is outside [{lo}, {hi}]"),
            });
        }
    }
    Ok(match width {
        IntWidth::Int8 => codes.iter().map(|c| *c as i8 as u8).collect(),
        IntWidth::Int4 => {
            let mut out = vec![0u8; codes.len().div_ceil(2)];
            for (k, c) in codes.iter().enumerate() {
                let nib = (*c as i8 as u8) & 0x0F;
                if k.is_multiple_of(2) {
                    out[k / 2] |= nib;
                } else {
                    out[k / 2] |= nib << 4;
                }
            }
            out
        }
    })
}

/// Rebias an unsigned source code to the canonical signed one: `q = u - bias`.
///
/// Document 03: "INT4 unsigned source u in [0,15] with zero point zu can be
/// normalized exactly to q=u-8 and z=zu-8; INT8 unsigned u in [0,255]
/// analogously uses q=u-128 and z=zu-128. This is rebiasing, not
/// requantization."
pub fn rebias_code(width: IntWidth, unsigned: u32) -> Result<i32> {
    if unsigned > width.max_unsigned() {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "unsigned code {unsigned} exceeds {} for {}",
                width.max_unsigned(),
                width.profile()
            ),
        });
    }
    Ok(unsigned as i32 - width.bias())
}

/// Rebias an unsigned source zero point the same way.
///
/// The result is deliberately not clipped to the code range: an asymmetric
/// source may legitimately place the zero point at either end, and clipping it
/// would change every reconstructed value in the group.
pub fn rebias_zero_point(width: IntWidth, unsigned: u32) -> Result<i16> {
    let z = rebias_code(width, unsigned)?;
    i16::try_from(z).map_err(|_| Error::InvalidArtifact {
        detail: format!("rebiased zero point {z} does not fit in i16"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16::f32_to_bf16_bits;

    fn desc(
        width: IntWidth,
        out_features: usize,
        in_features: usize,
        grouping: Grouping,
    ) -> AffineDescriptor {
        AffineDescriptor {
            width,
            out_features,
            in_features,
            grouping,
            group_index: None,
            scale_dtype: ScaleDtype::F32,
        }
    }

    /// Build a one-row tensor from explicit signed codes.
    fn row_tensor(
        width: IntWidth,
        codes: &[i32],
        grouping: Grouping,
        scales: Vec<f32>,
        zeros: ZeroPoints,
    ) -> AffineTensor {
        let d = desc(width, 1, codes.len(), grouping);
        AffineTensor::new(
            d,
            pack_row(width, codes).unwrap(),
            ScaleValues::F32(scales),
            zeros,
        )
        .unwrap()
    }

    // --- exhaustive decoding -------------------------------------------------

    #[test]
    fn every_int4_code_decodes_to_its_twos_complement_value() {
        // All 16 codes, derived from the definition rather than a table.
        let codes: Vec<i32> = (-8..=7).collect();
        let t = row_tensor(
            IntWidth::Int4,
            &codes,
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::Symmetric,
        );
        for (k, want) in codes.iter().enumerate() {
            assert_eq!(t.code(0, k).unwrap(), *want, "column {k}");
        }
        assert_eq!(
            t.reconstruct().unwrap(),
            codes.iter().map(|c| *c as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_int8_code_decodes_including_minus_128() {
        // The INT8 v1 draft rejected -128 outright. Document 03: a quantizer's
        // symmetric clipping preference "does not authorize a generic INT8
        // decoder to reject -128 from a valid source artifact."
        let codes: Vec<i32> = (-128..=127).collect();
        let t = row_tensor(
            IntWidth::Int8,
            &codes,
            Grouping::PerOutputChannel,
            vec![0.5],
            ZeroPoints::Symmetric,
        );
        for (k, want) in codes.iter().enumerate() {
            assert_eq!(t.code(0, k).unwrap(), *want, "column {k}");
        }
        let w = t.reconstruct().unwrap();
        assert_eq!(w[0], -64.0, "-128 * 0.5");
        assert_eq!(w[255], 63.5);
        assert_eq!(w.len(), 256);
    }

    #[test]
    fn sign_extension_is_exhaustive_over_the_nibble_space() {
        for n in 0u8..16 {
            let want = if n >= 8 { n as i32 - 16 } else { n as i32 };
            assert_eq!(sign_extend_4(n), want, "nibble {n:#06b}");
            // High bits of the byte are ignored.
            assert_eq!(sign_extend_4(n | 0xF0), want);
        }
    }

    #[test]
    fn int4_nibble_order_is_low_first() {
        // Codes 1 then 2 pack to 0x21. Reading them the other way round would
        // transpose every adjacent pair in the tensor.
        let packed = pack_row(IntWidth::Int4, &[1, 2]).unwrap();
        assert_eq!(packed, vec![0x21]);
        let t = row_tensor(
            IntWidth::Int4,
            &[1, 2],
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::Symmetric,
        );
        assert_eq!(t.reconstruct().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn negative_codes_pack_and_unpack_through_the_nibble() {
        let packed = pack_row(IntWidth::Int4, &[-8, -1]).unwrap();
        assert_eq!(packed, vec![0xF8]); // low nibble 0x8 = -8, high 0xF = -1
        let t = row_tensor(
            IntWidth::Int4,
            &[-8, -1],
            Grouping::PerOutputChannel,
            vec![2.0],
            ZeroPoints::Symmetric,
        );
        assert_eq!(t.reconstruct().unwrap(), vec![-16.0, -2.0]);
    }

    // --- rebias equivalence --------------------------------------------------

    #[test]
    fn signed_and_unsigned_source_encodings_reconstruct_identically() {
        // The rebias identity, exhaustively for INT4 and over the whole INT8
        // space: (u - zu) == ((u - bias) - (zu - bias)) for every pair. If this
        // ever fails, an asymmetric import is silently shifted.
        for width in IntWidth::ALL {
            let max = width.max_unsigned();
            let step = if max > 15 { 7 } else { 1 };
            let mut u = 0;
            while u <= max {
                let mut zu = 0;
                while zu <= max {
                    let unsigned_difference = u as i32 - zu as i32;
                    let q = rebias_code(*width, u).unwrap();
                    let z = rebias_zero_point(*width, zu).unwrap() as i32;
                    assert_eq!(q - z, unsigned_difference, "{width:?} u={u} zu={zu}");
                    zu += step;
                }
                u += step;
            }
        }
    }

    #[test]
    fn an_out_of_range_unsigned_code_is_rejected() {
        assert!(rebias_code(IntWidth::Int4, 16).is_err());
        assert!(rebias_code(IntWidth::Int4, 15).is_ok());
        assert!(rebias_code(IntWidth::Int8, 256).is_err());
        assert!(rebias_code(IntWidth::Int8, 255).is_ok());
    }

    #[test]
    fn a_zero_point_at_either_extreme_survives_rebiasing_unclipped() {
        // The failure this guards: clipping a rebiased zero point back into the
        // code range. For INT4, zu=0 rebiases to -8 and zu=15 to +7; both are
        // inside the range, but the *reconstruction* must use them as offsets,
        // not as codes.
        let t = row_tensor(
            IntWidth::Int4,
            &[7, -8],
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::PerGroup(vec![rebias_zero_point(IntWidth::Int4, 0).unwrap()]),
        );
        // z = -8, so 7 - (-8) = 15 and -8 - (-8) = 0.
        assert_eq!(t.reconstruct().unwrap(), vec![15.0, 0.0]);

        let t = row_tensor(
            IntWidth::Int4,
            &[7, -8],
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::PerGroup(vec![rebias_zero_point(IntWidth::Int4, 15).unwrap()]),
        );
        // z = +7, so 7 - 7 = 0 and -8 - 7 = -15.
        assert_eq!(t.reconstruct().unwrap(), vec![0.0, -15.0]);
    }

    #[test]
    fn the_int8_zero_point_extremes_reach_the_full_signed_span() {
        let t = row_tensor(
            IntWidth::Int8,
            &[127, -128],
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::PerGroup(vec![-128]),
        );
        assert_eq!(t.reconstruct().unwrap(), vec![255.0, 0.0]);
        let t = row_tensor(
            IntWidth::Int8,
            &[127, -128],
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::PerGroup(vec![127]),
        );
        assert_eq!(t.reconstruct().unwrap(), vec![0.0, -255.0]);
    }

    // --- grouping ------------------------------------------------------------

    #[test]
    fn group_boundaries_select_the_right_scale_at_32_and_128() {
        for size in [32u32, 128u32] {
            let n = size as usize * 3;
            let codes = vec![1i32; n];
            let scales: Vec<f32> = (1..=3).map(|g| g as f32).collect();
            let t = row_tensor(
                IntWidth::Int8,
                &codes,
                Grouping::Contiguous { size },
                scales,
                ZeroPoints::Symmetric,
            );
            let w = t.reconstruct().unwrap();
            for (k, v) in w.iter().enumerate() {
                let want = (k / size as usize + 1) as f32;
                assert_eq!(*v, want, "size {size}, column {k}");
            }
            // The boundary itself: the last column of one group and the first of
            // the next must differ.
            assert_ne!(w[size as usize - 1], w[size as usize]);
        }
    }

    #[test]
    fn a_partial_final_group_is_valid_and_uses_its_own_scale() {
        // 70 columns at group 32: groups of 32, 32 and a 6-wide tail. Document
        // 03: "Validate lengths with checked arithmetic; mask padded columns in
        // compute and omit them from logical values."
        let n = 70;
        let d = desc(IntWidth::Int4, 1, n, Grouping::Contiguous { size: 32 });
        assert_eq!(d.groups_per_row().unwrap(), 3);
        let t = row_tensor(
            IntWidth::Int4,
            &vec![1i32; n],
            Grouping::Contiguous { size: 32 },
            vec![1.0, 10.0, 100.0],
            ZeroPoints::Symmetric,
        );
        let w = t.reconstruct().unwrap();
        assert_eq!(w.len(), n, "the padding nibble is not a logical value");
        assert_eq!(w[31], 1.0);
        assert_eq!(w[32], 10.0);
        assert_eq!(w[64], 100.0);
        assert_eq!(w[69], 100.0);
    }

    #[test]
    fn an_odd_int4_row_is_byte_aligned_and_its_padding_nibble_is_not_a_value() {
        let n = 5;
        let t = row_tensor(
            IntWidth::Int4,
            &[1, 2, 3, 4, 5],
            Grouping::PerOutputChannel,
            vec![1.0],
            ZeroPoints::Symmetric,
        );
        assert_eq!(t.descriptor().code_bytes().unwrap(), 3);
        assert_eq!(t.reconstruct().unwrap().len(), n);
        assert!(t.code(0, 5).is_err(), "column 5 is outside the logical row");
    }

    #[test]
    fn per_output_channel_is_one_group_over_the_row() {
        let d = desc(IntWidth::Int8, 4, 100, Grouping::PerOutputChannel);
        assert_eq!(d.groups_per_row().unwrap(), 1);
        assert_eq!(d.group_entries().unwrap(), 4);
        for k in 0..100 {
            assert_eq!(d.group_of(k).unwrap(), 0);
        }
    }

    #[test]
    fn a_group_size_outside_the_closed_set_is_refused() {
        for bad in [16u32, 64, 1, 0, 129] {
            let d = desc(IntWidth::Int4, 1, 256, Grouping::Contiguous { size: bad });
            assert!(d.validate().is_err(), "group size {bad} should be refused");
        }
        for good in ALLOWED_GROUP_SIZES {
            let d = desc(IntWidth::Int4, 1, 256, Grouping::Contiguous { size: *good });
            assert!(d.validate().is_ok(), "group size {good}");
        }
    }

    #[test]
    fn each_row_has_its_own_scales() {
        // 2 rows x 64 columns at group 32 => 4 scale entries, row-major.
        let d = desc(IntWidth::Int8, 2, 64, Grouping::Contiguous { size: 32 });
        let mut codes = Vec::new();
        codes.extend(pack_row(IntWidth::Int8, &vec![1i32; 64]).unwrap());
        codes.extend(pack_row(IntWidth::Int8, &vec![1i32; 64]).unwrap());
        let t = AffineTensor::new(
            d,
            codes,
            ScaleValues::F32(vec![1.0, 2.0, 3.0, 4.0]),
            ZeroPoints::Symmetric,
        )
        .unwrap();
        assert_eq!(t.reconstruct_row(0).unwrap()[0], 1.0);
        assert_eq!(t.reconstruct_row(0).unwrap()[32], 2.0);
        assert_eq!(t.reconstruct_row(1).unwrap()[0], 3.0);
        assert_eq!(t.reconstruct_row(1).unwrap()[32], 4.0);
    }

    // --- activation order ----------------------------------------------------

    #[test]
    fn an_activation_order_permutation_selects_scales_by_the_map() {
        // Document 03: "If source activation-order metadata makes grouping
        // noncontiguous after restoring logical columns, retain a validated
        // per-input-column group-index map ... Never ignore g_idx."
        //
        // Four columns, two groups, interleaved: columns 0 and 2 belong to group
        // 0, columns 1 and 3 to group 1. Ignoring the map would use the
        // contiguous split and give every value the wrong scale.
        let mut d = desc(IntWidth::Int8, 1, 4, Grouping::Contiguous { size: 32 });
        d.in_features = 4;
        d.grouping = Grouping::Contiguous { size: 32 };
        d.group_index = Some(vec![0, 1, 0, 1]);
        // With in_features 4 and group size 32 there would be one group, so the
        // map's two groups must be refused as inconsistent...
        assert!(d.validate().is_err());

        // ... and with a shape where two groups exist, the map is honoured.
        let mut d = desc(IntWidth::Int8, 1, 64, Grouping::Contiguous { size: 32 });
        let mut map = vec![0u32; 64];
        for (k, g) in map.iter_mut().enumerate() {
            *g = (k % 2) as u32;
        }
        d.group_index = Some(map);
        let t = AffineTensor::new(
            d,
            pack_row(IntWidth::Int8, &vec![1i32; 64]).unwrap(),
            ScaleValues::F32(vec![1.0, 100.0]),
            ZeroPoints::Symmetric,
        )
        .unwrap();
        let w = t.reconstruct().unwrap();
        for (k, v) in w.iter().enumerate() {
            let want = if k % 2 == 0 { 1.0 } else { 100.0 };
            assert_eq!(*v, want, "column {k}");
        }
        // The contiguous reading would have given the first 32 columns scale 1.0
        // and the rest 100.0.
        assert_ne!(w[1], w[0]);
    }

    #[test]
    fn an_inconsistent_group_index_map_is_rejected() {
        let mut d = desc(IntWidth::Int8, 1, 64, Grouping::Contiguous { size: 32 });

        // Wrong length.
        d.group_index = Some(vec![0; 63]);
        assert!(d.validate().is_err());

        // Out of range group id.
        d.group_index = Some(vec![2; 64]);
        assert!(d.validate().is_err());

        // A group with a scale that no column maps to: the map and the table
        // disagree, and the extra scale is the symptom.
        d.group_index = Some(vec![0; 64]);
        let e = d.validate().unwrap_err();
        assert!(e.to_string().contains("group 1"), "{e}");
    }

    // --- scale dtype ---------------------------------------------------------

    #[test]
    fn the_source_scale_dtype_is_preserved_through_reconstruction() {
        // The same numeric intent in three encodings. FP16 0x3C01 is
        // 1 + 1/1024, which BF16 cannot represent; if import rounded it to BF16
        // the reconstruction would differ.
        let codes = [1i32, 1];
        let f16 = AffineTensor::new(
            AffineDescriptor {
                scale_dtype: ScaleDtype::F16,
                ..desc(IntWidth::Int8, 1, 2, Grouping::PerOutputChannel)
            },
            pack_row(IntWidth::Int8, &codes).unwrap(),
            ScaleValues::F16(vec![0x3C01]),
            ZeroPoints::Symmetric,
        )
        .unwrap();
        assert_eq!(f16.reconstruct().unwrap()[0], 1.0 + 1.0 / 1024.0);

        let as_bf16 = AffineTensor::new(
            AffineDescriptor {
                scale_dtype: ScaleDtype::Bf16,
                ..desc(IntWidth::Int8, 1, 2, Grouping::PerOutputChannel)
            },
            pack_row(IntWidth::Int8, &codes).unwrap(),
            ScaleValues::Bf16(vec![f32_to_bf16_bits(1.0 + 1.0 / 1024.0)]),
            ZeroPoints::Symmetric,
        )
        .unwrap();
        assert_ne!(
            as_bf16.reconstruct().unwrap()[0],
            f16.reconstruct().unwrap()[0],
            "rounding the scale to BF16 changes every weight in the group"
        );
    }

    #[test]
    fn a_scale_payload_that_disagrees_with_the_descriptor_is_rejected() {
        let d = AffineDescriptor {
            scale_dtype: ScaleDtype::F16,
            ..desc(IntWidth::Int8, 1, 2, Grouping::PerOutputChannel)
        };
        let e = AffineTensor::new(
            d,
            pack_row(IntWidth::Int8, &[1, 1]).unwrap(),
            ScaleValues::F32(vec![1.0]),
            ZeroPoints::Symmetric,
        )
        .unwrap_err();
        assert_eq!(e.kind(), "invalid_artifact");
    }

    // --- lengths, overflow, identity ----------------------------------------

    #[test]
    fn truncated_payloads_are_rejected_rather_than_read_past_the_end() {
        let d = desc(IntWidth::Int8, 2, 64, Grouping::Contiguous { size: 32 });
        // Too few code bytes.
        assert!(
            AffineTensor::new(
                d.clone(),
                vec![0u8; 100],
                ScaleValues::F32(vec![1.0; 4]),
                ZeroPoints::Symmetric
            )
            .is_err()
        );
        // Too few scales.
        assert!(
            AffineTensor::new(
                d.clone(),
                vec![0u8; 128],
                ScaleValues::F32(vec![1.0; 3]),
                ZeroPoints::Symmetric
            )
            .is_err()
        );
        // Too few zero points.
        assert!(
            AffineTensor::new(
                d.clone(),
                vec![0u8; 128],
                ScaleValues::F32(vec![1.0; 4]),
                ZeroPoints::PerGroup(vec![0; 3])
            )
            .is_err()
        );
        // Exactly right.
        assert!(
            AffineTensor::new(
                d,
                vec![0u8; 128],
                ScaleValues::F32(vec![1.0; 4]),
                ZeroPoints::PerGroup(vec![0; 4])
            )
            .is_ok()
        );
    }

    #[test]
    fn shape_arithmetic_that_would_overflow_is_an_error_not_a_wrap() {
        let d = desc(
            IntWidth::Int8,
            usize::MAX / 2,
            usize::MAX / 2,
            Grouping::PerOutputChannel,
        );
        assert!(d.code_bytes().is_err() || d.group_entries().is_err());
        assert!(d.validate().is_err());

        let empty = desc(IntWidth::Int4, 0, 16, Grouping::PerOutputChannel);
        assert!(empty.validate().is_err());
    }

    #[test]
    fn a_reconstruction_that_overflows_is_an_invalid_artifact() {
        // Second review, reproduced: a `f32::MAX` scale passes scale validation
        // -- it is finite and positive -- and then produced `Ok([inf])`.
        let t = row_tensor(
            IntWidth::Int4,
            &[7, 7],
            Grouping::PerOutputChannel,
            vec![f32::MAX],
            ZeroPoints::Symmetric,
        );
        let e = t.reconstruct().unwrap_err();
        assert_eq!(e.kind(), "invalid_artifact");
        assert!(e.to_string().contains("overflow"), "{e}");

        // The bounded row form must refuse it too, not write a partial row.
        let mut buf = vec![0f32; 2];
        assert!(t.reconstruct_row_into(0, &mut buf).is_err());

        // A code of zero cannot overflow whatever the scale, and must still work.
        let zeroed = row_tensor(
            IntWidth::Int4,
            &[0, 0],
            Grouping::PerOutputChannel,
            vec![f32::MAX],
            ZeroPoints::Symmetric,
        );
        assert_eq!(zeroed.reconstruct().unwrap(), vec![0.0, 0.0]);

        // The zero point participates: a large scale with a large offset
        // overflows where the code alone would not.
        let offset = row_tensor(
            IntWidth::Int8,
            &[0],
            Grouping::PerOutputChannel,
            vec![f32::MAX / 2.0],
            ZeroPoints::PerGroup(vec![-100]),
        );
        assert!(offset.reconstruct().is_err());
    }

    #[test]
    fn realistic_scales_reconstruct_without_tripping_the_overflow_check() {
        // The check must not fire on anything a real artifact contains. Weights
        // are order 0.01-0.1 and INT4 group scales are correspondingly small.
        let cols = 128;
        let t = row_tensor(
            IntWidth::Int4,
            &(0..cols).map(|k| (k as i32 % 15) - 7).collect::<Vec<_>>(),
            Grouping::Contiguous { size: 128 },
            vec![0.0134],
            ZeroPoints::Symmetric,
        );
        let w = t.reconstruct().unwrap();
        assert_eq!(w.len(), cols);
        assert!(w.iter().all(|v| v.abs() < 0.2));
    }

    #[test]
    fn a_nonfinite_or_zero_scale_never_reaches_a_kernel() {
        let d = desc(IntWidth::Int8, 1, 2, Grouping::PerOutputChannel);
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                AffineTensor::new(
                    d.clone(),
                    pack_row(IntWidth::Int8, &[1, 1]).unwrap(),
                    ScaleValues::F32(vec![bad]),
                    ZeroPoints::Symmetric
                )
                .is_err(),
                "scale {bad}"
            );
        }
    }

    #[test]
    fn a_code_outside_the_width_cannot_be_packed() {
        assert!(pack_row(IntWidth::Int4, &[8]).is_err());
        assert!(pack_row(IntWidth::Int4, &[-9]).is_err());
        assert!(pack_row(IntWidth::Int4, &[7, -8]).is_ok());
        assert!(pack_row(IntWidth::Int8, &[128]).is_err());
        assert!(pack_row(IntWidth::Int8, &[-129]).is_err());
        assert!(pack_row(IntWidth::Int8, &[127, -128]).is_ok());
    }

    #[test]
    fn identity_separates_every_field_document_03_lists() {
        let base = desc(IntWidth::Int4, 8, 256, Grouping::Contiguous { size: 128 });
        let id = base.identity(&ZeroPoints::Symmetric);

        let mut w8 = base.clone();
        w8.width = IntWidth::Int8;
        assert_ne!(id, w8.identity(&ZeroPoints::Symmetric));

        let mut g32 = base.clone();
        g32.grouping = Grouping::Contiguous { size: 32 };
        assert_ne!(id, g32.identity(&ZeroPoints::Symmetric));

        let mut f16 = base.clone();
        f16.scale_dtype = ScaleDtype::F16;
        assert_ne!(id, f16.identity(&ZeroPoints::Symmetric));

        let mut shape = base.clone();
        shape.out_features = 16;
        assert_ne!(id, shape.identity(&ZeroPoints::Symmetric));

        assert_ne!(id, base.identity(&ZeroPoints::PerGroup(vec![0; 16])));

        // Two different permutations are two different layouts.
        let mut a = base.clone();
        a.group_index = Some((0..256).map(|k| (k / 128) as u32).collect());
        let mut b = base.clone();
        b.group_index = Some((0..256).map(|k| (k % 2) as u32).collect());
        assert_ne!(
            a.identity(&ZeroPoints::Symmetric),
            b.identity(&ZeroPoints::Symmetric)
        );
        assert_ne!(id, a.identity(&ZeroPoints::Symmetric));
    }

    // --- bounded preparation -------------------------------------------------

    #[test]
    fn preparation_is_bounded_to_one_row() {
        // Document 03: dequantization tiles "must remain bounded and accounted"
        // and there must be "no repeated full-tensor dequantization on every
        // token". A caller can reconstruct any row into one reused buffer.
        let rows = 64;
        let cols = 128;
        let mut codes = Vec::new();
        for o in 0..rows {
            codes.extend(pack_row(IntWidth::Int4, &vec![(o % 8) as i32; cols]).unwrap());
        }
        let t = AffineTensor::new(
            desc(
                IntWidth::Int4,
                rows,
                cols,
                Grouping::Contiguous { size: 128 },
            ),
            codes,
            ScaleValues::F32(vec![1.0; rows]),
            ZeroPoints::Symmetric,
        )
        .unwrap();

        assert_eq!(t.prepared_row_bytes(), cols * 4);
        let mut buf = vec![0f32; cols];
        for o in 0..rows {
            t.reconstruct_row_into(o, &mut buf).unwrap();
            assert!(buf.iter().all(|v| *v == (o % 8) as f32), "row {o}");
        }
        // A wrongly sized destination is refused rather than partially written.
        let mut short = vec![0f32; cols - 1];
        assert!(t.reconstruct_row_into(0, &mut short).is_err());
        assert!(t.reconstruct_row_into(rows, &mut buf).is_err());
    }

    // --- value-preserving repack --------------------------------------------

    #[test]
    fn a_source_repack_reconstructs_bit_identical_weights() {
        // ADR 0003: "Prove source-to-canonical reconstructed weights are
        // unchanged under the declared source arithmetic when an import is
        // called lossless."
        //
        // The source here is the shape the pinned candidate evidence describes:
        // asymmetric INT4, group 32, unsigned nibble codes with an unsigned zero
        // point per group, FP16 scales. Its declared arithmetic is
        // `(u - zu) * s`, evaluated in FP32.
        let cols = 64usize;
        let groups = cols / 32;
        let source_codes: Vec<u32> = (0..cols).map(|k| (k * 7 % 16) as u32).collect();
        let source_zeros: Vec<u32> = vec![3, 11];
        let source_scale_bits: Vec<u16> = vec![0x3C01, 0x3800]; // 1+1/1024, 0.5

        // The source's own reconstruction, computed independently of the
        // canonical decoder.
        let source_weights: Vec<f32> = (0..cols)
            .map(|k| {
                let g = k / 32;
                let s = crate::scale::f16_bits_to_f32(source_scale_bits[g]);
                (source_codes[k] as i32 - source_zeros[g] as i32) as f32 * s
            })
            .collect();

        // The repack: rebias codes and zero points, keep the scale *bytes*.
        let canonical_codes: Vec<i32> = source_codes
            .iter()
            .map(|u| rebias_code(IntWidth::Int4, *u).unwrap())
            .collect();
        let canonical_zeros: Vec<i16> = source_zeros
            .iter()
            .map(|zu| rebias_zero_point(IntWidth::Int4, *zu).unwrap())
            .collect();

        let t = AffineTensor::new(
            AffineDescriptor {
                width: IntWidth::Int4,
                out_features: 1,
                in_features: cols,
                grouping: Grouping::Contiguous { size: 32 },
                group_index: None,
                scale_dtype: ScaleDtype::F16,
            },
            pack_row(IntWidth::Int4, &canonical_codes).unwrap(),
            ScaleValues::F16(source_scale_bits.clone()),
            ZeroPoints::PerGroup(canonical_zeros),
        )
        .unwrap();

        assert_eq!(
            t.reconstruct().unwrap(),
            source_weights,
            "the repack changed reconstructed values, so it is not lossless"
        );
        assert_eq!(groups, 2);
        assert_eq!(
            t.scales(),
            &ScaleValues::F16(source_scale_bits),
            "the scale payload must be the source's own bytes"
        );
    }

    #[test]
    fn a_symmetric_source_needs_no_zero_point_payload() {
        // The canada-quant / Intel shape: symmetric INT4, group 128, signed
        // codes. `(q - 0) * s` and `q * s` must agree exactly.
        let cols = 128usize;
        let codes: Vec<i32> = (0..cols).map(|k| ((k % 16) as i32) - 8).collect();
        let scale = 0.125f32;

        let symmetric = row_tensor(
            IntWidth::Int4,
            &codes,
            Grouping::Contiguous { size: 128 },
            vec![scale],
            ZeroPoints::Symmetric,
        );
        let explicit_zero = row_tensor(
            IntWidth::Int4,
            &codes,
            Grouping::Contiguous { size: 128 },
            vec![scale],
            ZeroPoints::PerGroup(vec![0]),
        );
        assert_eq!(
            symmetric.reconstruct().unwrap(),
            explicit_zero.reconstruct().unwrap()
        );
        assert_eq!(
            symmetric.reconstruct().unwrap(),
            codes.iter().map(|q| *q as f32 * scale).collect::<Vec<_>>()
        );
        // ... and they are different artifacts, because the payload differs.
        assert_ne!(symmetric.identity(), explicit_zero.identity());
    }

    #[test]
    fn the_int4_and_int8_profiles_share_one_algorithm() {
        // The same logical weights, expressed at both widths with matching
        // scales, reconstruct to the same values. That is what "one schema, two
        // profiles" has to mean.
        let cols = 32usize;
        let values: Vec<f32> = (0..cols).map(|k| (k as f32) - 16.0).collect();

        // INT4 codes span [-8, 7] with scale 2; INT8 codes span [-16, 15] with
        // scale 1. Both describe the same weights, at different resolutions.
        let int4 = row_tensor(
            IntWidth::Int4,
            &(0..cols).map(|k| (k as i32 - 16) / 2).collect::<Vec<_>>(),
            Grouping::Contiguous { size: 32 },
            vec![2.0],
            ZeroPoints::Symmetric,
        );
        let int8 = row_tensor(
            IntWidth::Int8,
            &(0..cols).map(|k| k as i32 - 16).collect::<Vec<_>>(),
            Grouping::Contiguous { size: 32 },
            vec![1.0],
            ZeroPoints::Symmetric,
        );
        assert_eq!(int8.reconstruct().unwrap(), values);
        // INT4 at half the resolution: even values match exactly.
        let w4 = int4.reconstruct().unwrap();
        for k in (0..cols).step_by(2) {
            assert_eq!(w4[k], values[k], "column {k}");
        }
        assert_eq!(int4.descriptor().width.precision(), Precision::Int4);
        assert_eq!(int8.descriptor().width.precision(), Precision::Int8);
        assert_eq!(int4.descriptor().width.profile(), "affine-int4-v1");
        assert_eq!(int8.descriptor().width.profile(), "affine-int8-v1");
    }
}
