//! Host tensors, and the BF16 invariant they carry.
//!
//! Task 0003: "Every tensor the interpreter holds is BF16-valued: stored as
//! `f32`, but every element is exactly a representable BF16 number, so widening
//! to FP32 is exact." That invariant is what makes the error bounds mean
//! anything -- a "BF16 tensor" holding a value BF16 cannot represent would put
//! an unaccounted rounding somewhere upstream of every bound in the contract.
//!
//! So it is checked on construction, not assumed. The only way to get a BF16
//! tensor is to present values that already satisfy it, or to round.

use moxie_format::bf16::{bf16_bits_to_f32, f32_to_bf16_bits};
use moxie_types::{ActivationPrecision, Error, Precision, Result};

/// A dense host tensor, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    precision: Precision,
}

/// Round one value to BF16 and back, the pinned round-to-nearest-even contract.
pub fn to_bf16(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(v))
}

/// Whether `v` is exactly representable in BF16.
pub fn is_bf16_valued(v: f32) -> bool {
    if v.is_nan() {
        // Any NaN is acceptable; its payload is not part of the contract.
        return true;
    }
    to_bf16(v) == v
}

impl HostTensor {
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: crate::try_clone_slice(&self.data)?,
            shape: crate::try_clone_slice(&self.shape)?,
            precision: self.precision,
        })
    }
    fn checked(data: Vec<f32>, shape: Vec<usize>, precision: Precision) -> Result<Self> {
        let want: usize = shape.iter().product();
        if data.len() != want {
            return Err(Error::InvalidArtifact {
                detail: format!("{} values for shape {shape:?}", data.len()),
            });
        }
        if shape.is_empty() {
            return Err(Error::InvalidArtifact {
                detail: "a rank-zero tensor".into(),
            });
        }
        Ok(Self {
            data,
            shape,
            precision,
        })
    }

    /// A BF16-valued tensor. Every element must already be representable.
    pub fn bf16(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        if let Some(i) = data.iter().position(|v| !is_bf16_valued(*v)) {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "element {i} is {} which is not representable in BF16; \
                     round it explicitly rather than storing it in a BF16 tensor",
                    data[i]
                ),
            });
        }
        Self::checked(data, shape, Precision::Bf16)
    }

    /// Round FP32 values to BF16 and build the tensor.
    ///
    /// This is the *only* rounding boundary. Task 0003's table says where each
    /// operation crosses it, and every crossing goes through here so that a
    /// second rounding rule cannot appear somewhere else.
    pub fn round_to_bf16(mut data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        for value in &mut data {
            *value = to_bf16(*value);
        }
        Self::checked(data, shape, Precision::Bf16)
    }

    /// An FP32 tensor. Used only where the contract says not to round: the
    /// vocabulary projection's logits.
    pub fn f32(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::checked(data, shape, Precision::F32)
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn precision(&self) -> Precision {
        self.precision
    }
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1..].iter().product()
    }

    /// One row of a rank-2 tensor.
    pub fn row(&self, r: usize) -> Result<&[f32]> {
        if self.shape.len() != 2 {
            return Err(Error::InvalidArtifact {
                detail: format!("row() on a rank-{} tensor", self.shape.len()),
            });
        }
        if r >= self.rows() {
            return Err(Error::InvalidRequest {
                field: "row",
                detail: format!("row {r} of {}", self.rows()),
            });
        }
        let c = self.cols();
        Ok(&self.data[r * c..(r + 1) * c])
    }

    /// Whether this tensor satisfies the role it is being used as.
    pub fn matches(&self, p: ActivationPrecision) -> bool {
        self.precision == p.get()
    }
}

/// A value flowing through the graph: a float tensor, or integer indices.
///
/// Document 02: "Token IDs, positions, page/group/sparse indices and masks also
/// need integer/boolean descriptor roles; they are not quantized weights or
/// floating activations." Keeping them a different variant means an index cannot
/// be handed to an operation expecting activations by accident.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Float(HostTensor),
    Index(Vec<u64>),
}

impl Value {
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(match self {
            Self::Float(tensor) => Self::Float(tensor.try_clone()?),
            Self::Index(indices) => Self::Index(crate::try_clone_slice(indices)?),
        })
    }
    pub fn as_float(&self) -> Result<&HostTensor> {
        match self {
            Value::Float(t) => Ok(t),
            Value::Index(_) => Err(Error::InvalidArtifact {
                detail: "expected a float tensor, got an index vector".into(),
            }),
        }
    }

    pub fn as_index(&self) -> Result<&[u64]> {
        match self {
            Value::Index(v) => Ok(v),
            Value::Float(_) => Err(Error::InvalidArtifact {
                detail: "expected an index vector, got a float tensor".into(),
            }),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Value::Float(t) => t.rows(),
            Value::Index(v) => v.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bf16_tensor_refuses_values_it_cannot_represent() {
        // The invariant every error bound rests on. 1 + 1/1024 needs more
        // mantissa bits than BF16 has.
        let unrepresentable = 1.0f32 + 1.0 / 1024.0;
        assert!(!is_bf16_valued(unrepresentable));
        assert!(HostTensor::bf16(vec![unrepresentable], vec![1]).is_err());

        // Rounding first is the supported way to get there.
        let t = HostTensor::round_to_bf16(vec![unrepresentable], vec![1]).unwrap();
        assert_eq!(t.data()[0], to_bf16(unrepresentable));
        assert!(is_bf16_valued(t.data()[0]));
    }

    #[test]
    fn representable_values_pass_through_unchanged() {
        let exact = vec![0.0f32, -0.0, 1.0, -2.0, 0.5, 256.0, f32::INFINITY];
        let t = HostTensor::bf16(exact.clone(), vec![7]).unwrap();
        assert_eq!(t.data(), &exact[..]);
        // ... and rounding them is the identity.
        let r = HostTensor::round_to_bf16(exact.clone(), vec![7]).unwrap();
        assert_eq!(r.data(), &exact[..]);
    }

    #[test]
    fn rounding_is_the_pinned_round_to_nearest_even_contract() {
        // Delegated to moxie-format, which tests it exhaustively over all 65,536
        // patterns. This asserts the delegation, not the contract.
        assert_eq!(
            to_bf16(f32::from_bits(0x3F80_8000)),
            f32::from_bits(0x3F80_0000)
        );
        assert_eq!(
            to_bf16(f32::from_bits(0x3F81_8000)),
            f32::from_bits(0x3F82_0000)
        );
    }

    #[test]
    fn shape_and_data_must_agree() {
        assert!(HostTensor::bf16(vec![1.0; 6], vec![2, 3]).is_ok());
        assert!(HostTensor::bf16(vec![1.0; 5], vec![2, 3]).is_err());
        assert!(HostTensor::bf16(vec![], vec![]).is_err());
    }

    #[test]
    fn rows_are_sliced_by_shape_not_guessed() {
        let t = HostTensor::bf16((0..6).map(|i| i as f32).collect(), vec![2, 3]).unwrap();
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 3);
        assert_eq!(t.row(0).unwrap(), &[0.0, 1.0, 2.0]);
        assert_eq!(t.row(1).unwrap(), &[3.0, 4.0, 5.0]);
        assert!(t.row(2).is_err());
        let flat = HostTensor::bf16(vec![1.0; 3], vec![3]).unwrap();
        assert!(flat.row(0).is_err(), "row() needs a rank-2 tensor");
    }

    #[test]
    fn an_index_is_not_a_tensor_and_a_tensor_is_not_an_index() {
        let t = Value::Float(HostTensor::bf16(vec![1.0], vec![1]).unwrap());
        let i = Value::Index(vec![7]);
        assert!(t.as_float().is_ok());
        assert!(t.as_index().is_err());
        assert!(i.as_index().is_ok());
        assert!(i.as_float().is_err());
    }

    #[test]
    fn logits_are_allowed_to_be_unrounded_fp32() {
        // The one place the contract does not round. A value BF16 cannot hold
        // is legal here and would be rejected by `bf16`.
        let v = 1.0f32 + 1.0 / 1024.0;
        let t = HostTensor::f32(vec![v], vec![1]).unwrap();
        assert_eq!(t.precision(), Precision::F32);
        assert_eq!(t.data()[0], v);
    }
}
