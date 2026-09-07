//! Typed semantic graph.
//!
//! Document 02: this crate owns "Model metadata, tensor roles, typed semantic
//! graph, shape/precision/state rules" and must not touch CUDA handles, file
//! I/O, residency policy or sampling loops.
//!
//! M0 scope: the operation catalogue as a **closed enum**, and the rule that an
//! operation without a declared contract cannot be lowered. Operations gain
//! their equations, oracles and kernels in the milestones that need them; this
//! draft exists so that adding one is a visible, typed change rather than a new
//! string appearing in a match arm.

#![forbid(unsafe_code)]

use moxie_types::{Dim, Error, Precision, Result};

/// The semantic operation catalogue from document 02.
///
/// Deliberately closed. Document 02 forbids "an opaque whole-model custom op or
/// arbitrary backend callback", so there is no `Custom(Box<dyn ...>)` variant and
/// there must never be one: a model that needs new mathematics extends this enum
/// through the enforced extension rule, with an oracle and a second consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Op {
    Embedding,
    Linear,
    ExpertLinear,
    VocabProjection,
    Residual,
    Concat,
    Split,
    RmsNorm,
    /// Distinct from `RmsNorm`. Document 02 requires them to be separate
    /// operations: GLM-5.3 uses LayerNorm where its siblings use RMSNorm, and
    /// treating them as one parameterised norm loses the mean subtraction.
    LayerNorm,
    Rope,
    SwiGlu,
    GeGlu,
    /// Kimi's bounded SiTU-GLU. R06: "explicitly bounded in both gate and up
    /// terms, not ordinary SwiGLU."
    SituGlu,
    Route,
    Dispatch,
    ExpertMlp,
    Combine,
    Attention,
    /// MLA has "latent/positional state and projection semantics, not pretend
    /// full KV with a giant hidden expansion" (document 04).
    MlaAttention,
    /// Model-defined sparse selection, e.g. GLM-5.3's DSA indexer.
    SparseIndexSelect,
    RecurrentUpdate,
    ShortConv,
    /// mHC-style residual mixing (document 02).
    ResidualMix,
}

impl Op {
    pub const fn name(self) -> &'static str {
        match self {
            Op::Embedding => "embedding",
            Op::Linear => "linear",
            Op::ExpertLinear => "expert_linear",
            Op::VocabProjection => "vocab_projection",
            Op::Residual => "residual",
            Op::Concat => "concat",
            Op::Split => "split",
            Op::RmsNorm => "rms_norm",
            Op::LayerNorm => "layer_norm",
            Op::Rope => "rope",
            Op::SwiGlu => "swiglu",
            Op::GeGlu => "geglu",
            Op::SituGlu => "situ_glu",
            Op::Route => "route",
            Op::Dispatch => "dispatch",
            Op::ExpertMlp => "expert_mlp",
            Op::Combine => "combine",
            Op::Attention => "attention",
            Op::MlaAttention => "mla_attention",
            Op::SparseIndexSelect => "sparse_index_select",
            Op::RecurrentUpdate => "recurrent_update",
            Op::ShortConv => "short_conv",
            Op::ResidualMix => "residual_mix",
        }
    }

    /// Whether this operation reads or writes sequence state. State-touching
    /// operations need a transaction and cannot be freely reordered or replayed.
    pub const fn touches_state(self) -> bool {
        matches!(
            self,
            Op::Attention
                | Op::MlaAttention
                | Op::SparseIndexSelect
                | Op::RecurrentUpdate
                | Op::ShortConv
        )
    }
}

/// What an operation must declare before it can be planned.
///
/// Document 02: "For every operation define: equations, input/output shapes,
/// valid dtypes, layout independence, tie behavior, rounding boundaries that are
/// semantically required, state effects, partition legality, workspace upper
/// bound, host oracle, and capability tests."
///
/// This draft carries the machine-checkable subset. `has_host_oracle` is the field that
/// makes the rule bite: without one there is nothing to validate a kernel
/// against, so lowering is refused.
#[derive(Debug, Clone)]
pub struct OpContract {
    pub op: Op,
    pub input_dtypes: Vec<Precision>,
    pub output_dtype: Precision,
    pub workspace_upper_bound: Dim,
    pub has_host_oracle: bool,
    pub partition_legal: bool,
}

impl OpContract {
    /// Refuse to lower an operation that cannot be validated.
    ///
    /// Document 07: "Missing numerical contracts block that primitive's
    /// optimized gate; agents must not choose a threshold after seeing a failing
    /// candidate."
    pub fn check_lowerable(&self) -> Result<()> {
        if !self.has_host_oracle {
            return Err(Error::UnsupportedKernel {
                operation: self.op.name(),
                detail: "no host oracle: nothing would validate a kernel against".into(),
            });
        }
        if self.input_dtypes.is_empty() {
            return Err(Error::UnsupportedKernel {
                operation: self.op.name(),
                detail: "no declared input dtypes".into(),
            });
        }
        for d in &self.input_dtypes {
            if !d.is_legal_weight() {
                return Err(Error::InvalidArtifact {
                    detail: format!("{} is not a legal weight precision", d),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(op: Op) -> OpContract {
        OpContract {
            op,
            input_dtypes: vec![Precision::Bf16],
            output_dtype: Precision::Bf16,
            workspace_upper_bound: Dim::constant(0),
            has_host_oracle: true,
            partition_legal: true,
        }
    }

    #[test]
    fn an_operation_without_an_oracle_cannot_be_lowered() {
        let mut c = contract(Op::Linear);
        assert!(c.check_lowerable().is_ok());
        c.has_host_oracle = false;
        let e = c.check_lowerable().unwrap_err();
        assert_eq!(e.kind(), "unsupported_kernel");
    }

    #[test]
    fn norms_are_distinct_operations() {
        // Document 02 requires RMSNorm and LayerNorm to be separate operations;
        // collapsing them loses LayerNorm's mean subtraction.
        assert_ne!(Op::RmsNorm, Op::LayerNorm);
        assert_ne!(Op::RmsNorm.name(), Op::LayerNorm.name());
    }

    #[test]
    fn glu_variants_are_distinct() {
        // R06: Kimi's SiTU-GLU is bounded in both terms; substituting SwiGLU
        // would be numerically wrong, not merely slower.
        let all = [Op::SwiGlu, Op::GeGlu, Op::SituGlu];
        let mut names: Vec<_> = all.iter().map(|o| o.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn state_touching_operations_are_identified() {
        assert!(Op::Attention.touches_state());
        assert!(Op::RecurrentUpdate.touches_state());
        assert!(Op::ShortConv.touches_state());
        assert!(Op::SparseIndexSelect.touches_state());
        assert!(!Op::Linear.touches_state());
        assert!(!Op::RmsNorm.touches_state());
    }

    #[test]
    fn operation_names_are_unique() {
        let all = [
            Op::Embedding,
            Op::Linear,
            Op::ExpertLinear,
            Op::VocabProjection,
            Op::Residual,
            Op::Concat,
            Op::Split,
            Op::RmsNorm,
            Op::LayerNorm,
            Op::Rope,
            Op::SwiGlu,
            Op::GeGlu,
            Op::SituGlu,
            Op::Route,
            Op::Dispatch,
            Op::ExpertMlp,
            Op::Combine,
            Op::Attention,
            Op::MlaAttention,
            Op::SparseIndexSelect,
            Op::RecurrentUpdate,
            Op::ShortConv,
            Op::ResidualMix,
        ];
        let mut names: Vec<_> = all.iter().map(|o| o.name()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(n, names.len(), "duplicate operation name");
    }
}
