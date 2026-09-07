//! Precision families and accumulation policy.
//!
//! Document 01 fixes the canonical families: NVFP4 preferred, INT8 and BF16.
//! Document 03 adds the hard rules that this module encodes: no weight below
//! four bits, and no cache below sixteen bits.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// Canonical low-bit weight family. E2M1 codes, E4M3FN block scale over 16
    /// consecutive logical input-channel values, positive finite tensor scale.
    Nvfp4,
    /// Signed symmetric weight-only INT8, per-output-channel BF16 scale, no zero
    /// point, code range [-127, 127].
    Int8,
    Bf16,
    F16,
    F32,
}

impl Precision {
    /// Bits per stored weight element, excluding block/tensor scales.
    pub const fn weight_bits(self) -> u32 {
        match self {
            Precision::Nvfp4 => 4,
            Precision::Int8 => 8,
            Precision::Bf16 | Precision::F16 => 16,
            Precision::F32 => 32,
        }
    }

    /// Document 01: "No weights below four bits."
    pub const fn is_legal_weight(self) -> bool {
        self.weight_bits() >= 4
    }

    /// Document 01: "cache no lower than 16 bits". Weight packing and cache
    /// precision are separate policies -- a weight-quantization switch must not
    /// reach the cache dtype (document 03).
    pub const fn is_legal_cache(self) -> bool {
        match self {
            Precision::Bf16 | Precision::F16 | Precision::F32 => true,
            Precision::Nvfp4 | Precision::Int8 => false,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Precision::Nvfp4 => "nvfp4",
            Precision::Int8 => "int8",
            Precision::Bf16 => "bf16",
            Precision::F16 => "f16",
            Precision::F32 => "f32",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How a linear or expert operation accumulates.
///
/// Document 03: "BF16 activations with FP32 accumulation are the baseline linear
/// contract; model-specific sensitive outputs may request FP32." This is part of
/// an operation's semantic contract, not a backend detail to be chosen freely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccumulationPolicy {
    /// BF16 inputs, FP32 accumulator. The default linear contract.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_weight_below_four_bits() {
        for p in [
            Precision::Nvfp4,
            Precision::Int8,
            Precision::Bf16,
            Precision::F16,
            Precision::F32,
        ] {
            assert!(p.is_legal_weight(), "{p} must be a legal weight precision");
            assert!(p.weight_bits() >= 4);
        }
    }

    #[test]
    fn low_bit_cache_is_rejected() {
        // Document 01: no optional sub-16-bit cache. A model's *intrinsic*
        // low-bit auxiliary state is a separate question, gated on O4, and must
        // not be enabled by reusing this predicate.
        assert!(!Precision::Nvfp4.is_legal_cache());
        assert!(!Precision::Int8.is_legal_cache());
        assert!(Precision::Bf16.is_legal_cache());
        assert!(Precision::F16.is_legal_cache());
        assert!(Precision::F32.is_legal_cache());
    }

    #[test]
    fn baseline_linear_accumulates_in_f32() {
        assert_eq!(
            AccumulationPolicy::Bf16InF32Acc.accumulator(),
            Precision::F32
        );
    }
}
