//! Precision families, role-specific contracts, and accumulation policy.
//!
//! Documents 01 and 03, as amended by [ADR 0003]: the canonical weight families
//! are **INT4, INT8 and BF16**. NVFP4/FP8 are deferred and have no active
//! runtime representation here. `Precision` is the set of *stored element
//! encodings*; which of them a given role may use is a separate question, and
//! the role types below answer it.
//!
//! [ADR 0003]: ../../../docs/decisions/adr/0003-int4-int8-bf16-weight-family.md
//!
//! ## Why roles are types
//!
//! The M0 review's F6 found `OpContract` validating every input with
//! `is_legal_weight`, which meant a caller could request INT4 *activations* by
//! passing a legal weight dtype. Weight storage, activation dtype, cache dtype
//! and accumulator dtype are four different policies with four different rules,
//! and document 03 is explicit that "cache dtypes are not controlled by a
//! weight-quantization switch". Making them one enum is what let them merge.

use core::fmt;

use crate::error::{Error, Result};

/// A stored element encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Precision {
    /// Signed two's-complement 4-bit weight code, `[-8, 7]`. Carries no scale by
    /// itself: the affine descriptor in `moxie-format` supplies group size,
    /// scale and zero point.
    Int4,
    /// Signed two's-complement 8-bit weight code, `[-128, 127]`.
    Int8,
    Bf16,
    F16,
    F32,
}

impl Precision {
    /// Bits per stored element, excluding scales, zero points and index maps.
    ///
    /// Document 03 requires that metadata be charged too: an INT4 tensor with
    /// group-32 FP16 scales and i16 zero points is not 4 bits per weight in any
    /// budget. This function is the element width, not the storage cost.
    pub const fn bits(self) -> u32 {
        match self {
            Precision::Int4 => 4,
            Precision::Int8 => 8,
            Precision::Bf16 | Precision::F16 => 16,
            Precision::F32 => 32,
        }
    }

    /// Whether this encoding is an integer weight code rather than a float.
    pub const fn is_integer(self) -> bool {
        matches!(self, Precision::Int4 | Precision::Int8)
    }

    /// Document 01: "No weights below four bits."
    pub const fn is_legal_weight(self) -> bool {
        self.bits() >= 4
    }

    /// Document 01: "cache no lower than 16 bits", and it must be a float
    /// encoding. Weight packing and cache precision are separate policies -- a
    /// weight-quantization switch must not reach the cache dtype (document 03).
    pub const fn is_legal_cache(self) -> bool {
        !self.is_integer() && self.bits() >= 16
    }

    /// Document 03: initial execution profiles are weight-only W4A16/W8A16.
    /// "Native W4A4/W8A8 changes activation quantization and is outside initial
    /// scope", so an integer encoding is not a legal activation.
    pub const fn is_legal_activation(self) -> bool {
        !self.is_integer() && self.bits() >= 16
    }

    pub const fn name(self) -> &'static str {
        match self {
            Precision::Int4 => "int4",
            Precision::Int8 => "int8",
            Precision::Bf16 => "bf16",
            Precision::F16 => "f16",
            Precision::F32 => "f32",
        }
    }

    /// Every encoding, so a test can enumerate the space exhaustively.
    pub const ALL: &'static [Precision] = &[
        Precision::Int4,
        Precision::Int8,
        Precision::Bf16,
        Precision::F16,
        Precision::F32,
    ];
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Build one role newtype: a `Precision` that passed that role's rule.
///
/// Constructing one is the only way to get it, so a value of this type *is* the
/// evidence that the rule was checked. Passing a weight dtype where an
/// activation dtype is expected stops compiling.
macro_rules! role_precision {
    ($(#[$m:meta])* $name:ident, $check:ident, $role:literal, $why:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(Precision);

        impl $name {
            pub const fn new(p: Precision) -> Result<Self> {
                if p.$check() {
                    Ok(Self(p))
                } else {
                    Err(Error::Unsupported {
                        capability: $role,
                        reason: String::new(),
                    })
                }
            }

            /// Panicking constructor for constants that are legal by inspection.
            ///
            /// # Panics
            /// If `p` is not legal for this role.
            pub const fn expect(p: Precision) -> Self {
                assert!(p.$check(), $why);
                Self(p)
            }

            pub const fn get(self) -> Precision {
                self.0
            }

            pub const fn bits(self) -> u32 {
                self.0.bits()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.0.name())
            }
        }
    };
}

role_precision!(
    /// How a weight tensor is stored. INT4/INT8 codes, or plain BF16/FP16/FP32.
    WeightPrecision,
    is_legal_weight,
    "weight_precision",
    "document 01 forbids weights below four bits"
);

role_precision!(
    /// The dtype activations are computed in. Never an integer code at initial
    /// scope: W4A16/W8A16 are weight-only (document 03).
    ActivationPrecision,
    is_legal_activation,
    "activation_precision",
    "W4A4/W8A8 activation quantization is outside initial scope"
);

role_precision!(
    /// The dtype of KV and persistent recurrent/attention state.
    ///
    /// Document 01: "cache no lower than 16 bits". A model's *intrinsic* low-bit
    /// auxiliary representation is a separate question, gated on O4, and must
    /// not be enabled by widening this type.
    CachePrecision,
    is_legal_cache,
    "cache_precision",
    "document 01 forbids a cache below sixteen bits"
);

/// How a linear or expert operation accumulates.
///
/// Document 03: "BF16 activations with FP32 accumulation are the baseline linear
/// contract; model-specific sensitive outputs may request FP32." This is part of
/// an operation's semantic contract, not a backend detail to be chosen freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccumulationPolicy {
    /// 16-bit inputs, FP32 accumulator. The default linear contract.
    Bf16InF32Acc,
    /// FP32 throughout, for sensitive tensors and model-required state.
    F32,
}

impl AccumulationPolicy {
    pub const fn accumulator(self) -> Precision {
        match self {
            AccumulationPolicy::Bf16InF32Acc | AccumulationPolicy::F32 => Precision::F32,
        }
    }
}

/// A weight/activation execution profile.
///
/// `W4A16` names operand widths. It is **not** a file layout, a quantization
/// algorithm, or an INT4xINT4 tensor-core instruction: ADR 0003 and document 03
/// are explicit that "W4A16 does not directly use an INT4-times-INT4 MMA
/// instruction, which requires low-bit activations too". The integer codes are
/// unpacked and dequantized into 16-bit tensor-core tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionProfile {
    pub weight: WeightPrecision,
    pub activation: ActivationPrecision,
    pub accumulation: AccumulationPolicy,
}

impl ExecutionProfile {
    /// The initial profiles from document 03: W4A16, W8A16 and plain BF16, all
    /// with FP32 accumulation and BF16 activations.
    pub const W4A16: Self = Self {
        weight: WeightPrecision::expect(Precision::Int4),
        activation: ActivationPrecision::expect(Precision::Bf16),
        accumulation: AccumulationPolicy::Bf16InF32Acc,
    };
    pub const W8A16: Self = Self {
        weight: WeightPrecision::expect(Precision::Int8),
        activation: ActivationPrecision::expect(Precision::Bf16),
        accumulation: AccumulationPolicy::Bf16InF32Acc,
    };
    pub const BF16: Self = Self {
        weight: WeightPrecision::expect(Precision::Bf16),
        activation: ActivationPrecision::expect(Precision::Bf16),
        accumulation: AccumulationPolicy::Bf16InF32Acc,
    };

    pub const fn name(self) -> &'static str {
        match (self.weight.get(), self.activation.get()) {
            (Precision::Int4, Precision::Bf16) => "w4a16-bf16",
            (Precision::Int8, Precision::Bf16) => "w8a16-bf16",
            (Precision::Int4, Precision::F16) => "w4a16-f16",
            (Precision::Int8, Precision::F16) => "w8a16-f16",
            (Precision::Bf16, _) => "bf16",
            _ => "custom",
        }
    }

    /// Whether the weights need integer unpacking before the tensor cores see
    /// them. True for W4A16/W8A16 and false for a plain float profile.
    pub const fn needs_weight_dequantization(self) -> bool {
        self.weight.get().is_integer()
    }
}

impl fmt::Display for ExecutionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_weight_below_four_bits() {
        for p in Precision::ALL {
            assert!(p.is_legal_weight(), "{p} must be a legal weight precision");
            assert!(p.bits() >= 4);
            assert!(WeightPrecision::new(*p).is_ok());
        }
    }

    #[test]
    fn low_bit_cache_is_rejected() {
        // Document 01: no optional sub-16-bit cache. A model's *intrinsic*
        // low-bit auxiliary state is a separate question, gated on O4, and must
        // not be enabled by reusing this predicate.
        assert!(CachePrecision::new(Precision::Int4).is_err());
        assert!(CachePrecision::new(Precision::Int8).is_err());
        for p in [Precision::Bf16, Precision::F16, Precision::F32] {
            assert!(CachePrecision::new(p).is_ok(), "{p}");
        }
    }

    #[test]
    fn integer_activations_are_rejected_because_w4a4_is_out_of_scope() {
        // This is F6 in one assertion. Under the old single predicate, INT4 was
        // "a legal weight precision", and an operation contract that validated
        // its inputs with that predicate accepted INT4 activations.
        assert!(ActivationPrecision::new(Precision::Int4).is_err());
        assert!(ActivationPrecision::new(Precision::Int8).is_err());
        assert!(ActivationPrecision::new(Precision::Bf16).is_ok());
        assert!(ActivationPrecision::new(Precision::F16).is_ok());
        assert!(ActivationPrecision::new(Precision::F32).is_ok());

        // ... while the same encodings remain legal weights.
        assert!(WeightPrecision::new(Precision::Int4).is_ok());
        assert!(WeightPrecision::new(Precision::Int8).is_ok());
    }

    #[test]
    fn the_role_types_do_not_interconvert() {
        // A weight dtype cannot be passed where a cache dtype is expected: the
        // types differ, so `let _: CachePrecision = w;` does not compile. What
        // is checkable at run time is that the underlying encoding survives.
        let w = WeightPrecision::new(Precision::Int4).unwrap();
        assert_eq!(w.get(), Precision::Int4);
        assert_eq!(w.bits(), 4);
        assert_eq!(w.to_string(), "int4");
    }

    #[test]
    fn baseline_linear_accumulates_in_f32() {
        assert_eq!(
            AccumulationPolicy::Bf16InF32Acc.accumulator(),
            Precision::F32
        );
        assert_eq!(AccumulationPolicy::F32.accumulator(), Precision::F32);
    }

    #[test]
    fn initial_profiles_are_weight_only_with_sixteen_bit_activations() {
        for p in [
            ExecutionProfile::W4A16,
            ExecutionProfile::W8A16,
            ExecutionProfile::BF16,
        ] {
            assert!(
                !p.activation.get().is_integer(),
                "{p} would be an activation-quantized profile"
            );
            assert_eq!(p.activation.bits(), 16);
            assert_eq!(p.accumulation.accumulator(), Precision::F32);
        }
        assert_eq!(ExecutionProfile::W4A16.to_string(), "w4a16-bf16");
        assert_eq!(ExecutionProfile::W8A16.to_string(), "w8a16-bf16");
    }

    #[test]
    fn an_integer_profile_declares_that_it_must_dequantize() {
        // ADR 0003: W4A16 is not an INT4xINT4 MMA. The weights are unpacked into
        // 16-bit tiles, and the profile says so rather than leaving a kernel to
        // assume otherwise.
        assert!(ExecutionProfile::W4A16.needs_weight_dequantization());
        assert!(ExecutionProfile::W8A16.needs_weight_dequantization());
        assert!(!ExecutionProfile::BF16.needs_weight_dequantization());
    }

    #[test]
    fn no_nvfp4_encoding_remains_in_the_active_family() {
        // ADR 0003 replaces NVFP4 with the integer family. The check that it is
        // really gone is that nothing in `ALL` is a 4-bit *float*: every 4-bit
        // encoding here is an integer code.
        for p in Precision::ALL {
            if p.bits() == 4 {
                assert!(p.is_integer(), "{p} is a 4-bit non-integer encoding");
            }
        }
        assert_eq!(
            Precision::ALL.iter().filter(|p| p.bits() == 4).count(),
            1,
            "exactly one 4-bit encoding: int4"
        );
    }
}
