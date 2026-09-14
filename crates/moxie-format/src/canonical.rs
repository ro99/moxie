//! Canonical artifact v2: which physical tensors a logical one becomes.
//!
//! [ADR 0025] fixes the mapping and this module is it, so that the writer, the
//! reader and every test name the same components, shapes and dtypes. Nothing
//! here opens a file or decides where a component lives; it says what a
//! component **is**.
//!
//! The mathematics does not move. These are the bytes [ADR 0023] already
//! defines -- the same packed nibbles, the same scale table in the source's own
//! dtype, the same `i16` zero points -- carried as separate physical tensors
//! instead of three sections of one range.
//!
//! [ADR 0023]: ../../../docs/decisions/adr/0023-canonical-affine-payload-and-repack-journal.md
//! [ADR 0025]: ../../../docs/decisions/adr/0025-canonical-safetensors-schema.md

use moxie_types::Result;

use crate::affine::{AffineDescriptor, IntWidth};
use crate::payload::ZeroPointSection;
use crate::safetensors::Dtype;
use crate::scale::ScaleDtype;

/// The three things a logical tensor can be made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComponentKind {
    /// A BF16 tensor, carried whole and unchanged.
    Weights,
    /// Packed integer codes.
    Codes,
    /// The scale table, in the source's own encoding.
    Scales,
    /// The zero-point table. Absent for a symmetric tensor: implicit zero has
    /// no payload, and a component of zeros would be a different claim.
    ZeroPoints,
}

impl ComponentKind {
    pub const fn name(self) -> &'static str {
        match self {
            ComponentKind::Weights => "weights",
            ComponentKind::Codes => "codes",
            ComponentKind::Scales => "scales",
            ComponentKind::ZeroPoints => "zero_points",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "weights" => ComponentKind::Weights,
            "codes" => ComponentKind::Codes,
            "scales" => ComponentKind::Scales,
            "zero_points" => ComponentKind::ZeroPoints,
            _ => return None,
        })
    }

    /// The safetensors tensor name for this component of `role`.
    ///
    /// A BF16 tensor keeps the role itself, so an ecosystem reader sees the
    /// name the model uses. The affine components are suffixed, exactly as the
    /// sources this repacker reads suffix theirs.
    pub fn tensor_name(self, role: &str) -> String {
        match self {
            ComponentKind::Weights => role.to_string(),
            other => format!("{role}.{}", other.name()),
        }
    }
}

/// One physical tensor of a canonical artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub kind: ComponentKind,
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub len: u64,
}

/// The components a BF16 tensor becomes: one, carrying its own bytes.
pub fn bf16_components(role: &str, shape: &[u64]) -> Result<Vec<Component>> {
    let elements: u64 = shape.iter().try_fold(1u64, |a, d| {
        a.checked_mul(*d)
            .ok_or_else(|| crate::invalid_static("shape product overflows"))
    })?;
    let len = elements
        .checked_mul(2)
        .ok_or_else(|| crate::invalid_static("BF16 byte count overflows"))?;
    Ok(vec![Component {
        kind: ComponentKind::Weights,
        name: ComponentKind::Weights.tensor_name(role),
        dtype: Dtype::Bf16,
        shape: shape.to_vec(),
        len,
    }])
}

/// The components an affine tensor becomes: codes, scales, and zero points
/// when it has them.
///
/// INT4 codes are `U8` because a nibble pair is not a number -- calling it `I8`
/// would invite a reader to treat it as one -- and their physical width is
/// `ceil(in_features / 2)`, which is the row stride the packing already has.
/// INT8 codes **are** numbers, so they are `I8` at the logical width.
pub fn affine_components(
    role: &str,
    desc: &AffineDescriptor,
    zero_points: ZeroPointSection,
) -> Result<Vec<Component>> {
    desc.validate()?;
    let out = desc.out_features as u64;
    let groups = desc.groups_per_row()? as u64;
    let (code_dtype, code_width) = match desc.width {
        IntWidth::Int4 => (Dtype::U8, desc.in_features.div_ceil(2) as u64),
        IntWidth::Int8 => (Dtype::I8, desc.in_features as u64),
    };
    let scale_dtype = match desc.scale_dtype {
        ScaleDtype::F16 => Dtype::F16,
        ScaleDtype::Bf16 => Dtype::Bf16,
        ScaleDtype::F32 => Dtype::F32,
    };
    let mut components = vec![
        Component {
            kind: ComponentKind::Codes,
            name: ComponentKind::Codes.tensor_name(role),
            dtype: code_dtype,
            shape: vec![out, code_width],
            len: out * code_width,
        },
        Component {
            kind: ComponentKind::Scales,
            name: ComponentKind::Scales.tensor_name(role),
            dtype: scale_dtype,
            shape: vec![out, groups],
            len: out * groups * desc.scale_dtype.bytes() as u64,
        },
    ];
    if zero_points == ZeroPointSection::PerGroup {
        components.push(Component {
            kind: ComponentKind::ZeroPoints,
            name: ComponentKind::ZeroPoints.tensor_name(role),
            dtype: Dtype::I16,
            shape: vec![out, groups],
            len: out * groups * crate::payload::ZERO_POINT_BYTES as u64,
        });
    }
    Ok(components)
}

/// The `__metadata__` a shard carries.
///
/// A courtesy for ecosystem tools and **not an authority**: no reader here may
/// depend on it, and nothing in it is validated as meaning. The manifest is the
/// single source of truth for what these tensors are.
pub const SCHEMA_TAG_KEY: &str = "moxie.schema";
pub const SCHEMA_TAG: &str = "canonical-v2";
pub const IDENTITY_KEY: &str = "moxie.artifact";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Grouping;

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

    /// The mapping ADR 0025 fixes, written out here rather than recomputed.
    #[test]
    fn the_component_mapping_is_the_one_the_schema_states() {
        // INT4 [4, 66]: codes are U8 [4, 33] -- the row stride the packing
        // already has -- and three groups per row at group 32.
        let c = affine_components(
            "w",
            &desc(IntWidth::Int4, 4, 66),
            ZeroPointSection::PerGroup,
        )
        .unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].name, "w.codes");
        assert_eq!(
            (c[0].dtype, c[0].shape.clone(), c[0].len),
            (Dtype::U8, vec![4, 33], 132)
        );
        assert_eq!(c[1].name, "w.scales");
        assert_eq!(
            (c[1].dtype, c[1].shape.clone(), c[1].len),
            (Dtype::Bf16, vec![4, 3], 24)
        );
        assert_eq!(c[2].name, "w.zero_points");
        assert_eq!(
            (c[2].dtype, c[2].shape.clone(), c[2].len),
            (Dtype::I16, vec![4, 3], 24)
        );

        // INT8 codes are I8 at the logical width, because they are numbers.
        let c =
            affine_components("w", &desc(IntWidth::Int8, 2, 64), ZeroPointSection::Absent).unwrap();
        assert_eq!(c.len(), 2, "a symmetric tensor has no zero-point component");
        assert_eq!(
            (c[0].dtype, c[0].shape.clone(), c[0].len),
            (Dtype::I8, vec![2, 64], 128)
        );

        // A BF16 tensor keeps its own name, so an ecosystem reader sees the
        // name the model uses.
        let c = bf16_components("model.norm.weight", &[8]).unwrap();
        assert_eq!(c[0].name, "model.norm.weight");
        assert_eq!((c[0].dtype, c[0].len), (Dtype::Bf16, 16));
    }

    #[test]
    fn component_kinds_round_trip_through_their_names() {
        for kind in [
            ComponentKind::Weights,
            ComponentKind::Codes,
            ComponentKind::Scales,
            ComponentKind::ZeroPoints,
        ] {
            assert_eq!(ComponentKind::parse(kind.name()), Some(kind));
        }
        assert_eq!(ComponentKind::parse("payload"), None);
    }
}
