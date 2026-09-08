//! Typed semantic graph.
//!
//! Document 02: this crate owns "Model metadata, tensor roles, typed semantic
//! graph, shape/precision/state rules" and must not touch CUDA handles, file
//! I/O, residency policy or sampling loops.
//!
//! ## Scope, stated accurately
//!
//! This is **not a validated graph**. It is the operation catalogue as a closed
//! enum, plus the precision/oracle/partition rules that must hold before an
//! operation can be lowered. Present and enforced:
//!
//! * a closed `Op` catalogue with no escape hatch;
//! * role-separated precisions, so an operation cannot request low-bit
//!   activations by naming a legal weight dtype;
//! * an oracle *registry* the contract cannot write to;
//! * a partition rule that fails closed when undetermined.
//!
//! Deliberately absent, and not to be mistaken for present: edges, shapes,
//! layouts, workspace derivation, state-effect checking against a real schema,
//! and any lowering. The M0 review's F6 found the earlier draft describing
//! itself as more than it was -- `has_host_oracle: bool` was set by the caller
//! and accepted as proof, and every input dtype was checked with
//! `is_legal_weight`, which admits INT4 activations. Those two are fixed here.
//! The rest is M1 work and is labelled as missing rather than stubbed.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use moxie_types::{AccumulationPolicy, ActivationPrecision, Dim, Error, Result, WeightPrecision};

/// The semantic operation catalogue from document 02.
///
/// Deliberately closed. Document 02 forbids "an opaque whole-model custom op or
/// arbitrary backend callback", so there is no `Custom(Box<dyn ...>)` variant and
/// there must never be one: a model that needs new mathematics extends this enum
/// through the enforced extension rule, with an oracle and a second consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    pub const ALL: &'static [Op] = &[
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
}

/// Identity of an independent host reference implementation.
///
/// Document 07: an operation's optimized gate is blocked without a numerical
/// contract, and the reference must be "independently generated". A boolean on
/// the contract could not express that, because the thing being validated was
/// setting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OracleId(pub &'static str);

/// What backs a registered oracle.
///
/// M0 records *where* the independent implementation and its exhaustive test
/// live, so the claim is checkable by a reader and by review. It is honestly
/// weaker than executing the oracle: M1 replaces `evidence` with a callable
/// reference implementation once there is an interpreter to call it from. What
/// it already prevents is the F6 defect -- a contract asserting its own oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OracleEvidence {
    /// Where the independent implementation lives, e.g. `moxie_format::affine`.
    pub implementation: &'static str,
    /// The test module that exercises it exhaustively.
    pub test_module: &'static str,
}

/// The set of oracles that actually exist.
///
/// Owned by the composition root, never by an operation contract. An operation
/// whose `OracleId` is absent cannot be lowered, and no amount of editing the
/// contract changes that.
#[derive(Debug, Clone, Default)]
pub struct OracleRegistry {
    entries: BTreeMap<(Op, OracleId), OracleEvidence>,
}

impl OracleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an independent host reference for one operation.
    ///
    /// A rejected registration leaves the registry **unchanged**. The first
    /// version used `insert(...).is_some()`, which wrote the new evidence and
    /// then reported failure -- so a duplicate registration silently replaced
    /// the entry it was supposed to protect, and a caller that ignored the error
    /// ended up validating against whichever implementation registered last.
    pub fn register(&mut self, op: Op, id: OracleId, evidence: OracleEvidence) -> Result<()> {
        if self.entries.contains_key(&(op, id)) {
            return Err(Error::InvalidRequest {
                field: "oracle",
                detail: format!("{} already has an oracle named {}", op.name(), id.0),
            });
        }
        self.entries.insert((op, id), evidence);
        Ok(())
    }

    pub fn evidence(&self, op: Op, id: OracleId) -> Option<OracleEvidence> {
        self.entries.get(&(op, id)).copied()
    }

    /// Whether any independent reference exists for this operation at all.
    ///
    /// Used at model admission, where the question is "does this operation have
    /// mathematics somebody else can check" rather than "is this particular
    /// contract's named oracle present".
    pub fn has_any_oracle(&self, op: Op) -> bool {
        self.entries.keys().any(|(o, _)| *o == op)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// How an operation may be split across ranks.
///
/// `NotDetermined` is the default and fails closed: document 04 requires legal
/// partition semantics to be *defined* per operation, and "non-divisible
/// dimensions use checked padding or explicit unsupported combinations".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartitionRule {
    /// Nobody has worked out how to shard this yet. Single-rank only.
    #[default]
    NotDetermined,
    /// Shardable along the output-channel axis, concatenating results.
    ColumnShardable,
    /// Shardable along the input-channel axis, with a global reduction.
    RowShardable,
    /// Replicated on every rank; bias and residual terms applied exactly once.
    Replicated,
}

impl PartitionRule {
    pub const fn is_partitionable(self) -> bool {
        !matches!(self, PartitionRule::NotDetermined)
    }
}

/// What an operation does to sequence state.
///
/// Named here rather than imported: `moxie-graph` may not depend on
/// `moxie-state` (the ownership table puts both above `moxie-types` and beside
/// each other). The state crate owns the schema and the restore rules; this is
/// only the graph-side declaration that a transaction is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StateEffect {
    #[default]
    None,
    /// Reads existing state; safe to replay.
    Reads,
    /// Appends to state; needs a transaction and a rollback path.
    Appends,
    /// Mutates state in place; needs a snapshot or replay to roll back.
    Mutates,
}

/// What an operation must declare before it can be planned.
///
/// Document 02: "For every operation define: equations, input/output shapes,
/// valid dtypes, layout independence, tie behavior, rounding boundaries that are
/// semantically required, state effects, partition legality, workspace upper
/// bound, host oracle, and capability tests."
///
/// This carries the machine-checkable subset. Shapes, layout independence, tie
/// behaviour and rounding boundaries are **not** here yet; `check_lowerable`
/// says so rather than passing silently.
#[derive(Debug, Clone)]
pub struct OpContract {
    pub op: Op,
    /// Precisions the weight operands may be stored in. Empty for an operation
    /// with no weights (a norm's gain is a weight; a residual add has none).
    pub weights: Vec<WeightPrecision>,
    /// Precisions the activation operands may arrive in.
    pub activations: Vec<ActivationPrecision>,
    pub output: ActivationPrecision,
    pub accumulation: AccumulationPolicy,
    pub workspace_upper_bound: Dim,
    /// The independent host reference this operation is validated against. It
    /// must be present in the registry; naming one here proves nothing.
    pub oracle: OracleId,
    pub partition: PartitionRule,
    pub state_effect: StateEffect,
}

impl OpContract {
    /// Refuse to lower an operation that cannot be validated.
    ///
    /// Document 07: "Missing numerical contracts block that primitive's
    /// optimized gate; agents must not choose a threshold after seeing a failing
    /// candidate."
    pub fn check_lowerable(&self, oracles: &OracleRegistry) -> Result<()> {
        if oracles.evidence(self.op, self.oracle).is_none() {
            return Err(Error::UnsupportedKernel {
                operation: self.op.name(),
                detail: format!(
                    "oracle {:?} is not registered for {}: nothing independent would \
                     validate a kernel against",
                    self.oracle.0,
                    self.op.name()
                ),
            });
        }
        if self.activations.is_empty() {
            return Err(Error::UnsupportedKernel {
                operation: self.op.name(),
                detail: "no declared activation dtypes".into(),
            });
        }
        if self.op.touches_state() && self.state_effect == StateEffect::None {
            return Err(Error::UnsupportedKernel {
                operation: self.op.name(),
                detail: "a state-touching operation must declare its state effect".into(),
            });
        }
        Ok(())
    }

    /// Refuse to partition an operation whose partition semantics are undefined.
    pub fn check_partitionable(&self) -> Result<()> {
        if !self.partition.is_partitionable() {
            return Err(Error::Unsupported {
                capability: "tensor_parallel",
                reason: format!(
                    "{} has no defined partition semantics; document 04 requires them \
                     per operation before TP lowering",
                    self.op.name()
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_types::Precision;

    const TEST_ORACLE: OracleId = OracleId("test");

    fn registry(op: Op) -> OracleRegistry {
        let mut r = OracleRegistry::new();
        r.register(
            op,
            TEST_ORACLE,
            OracleEvidence {
                implementation: "test",
                test_module: "test",
            },
        )
        .unwrap();
        r
    }

    fn contract(op: Op) -> OpContract {
        OpContract {
            op,
            weights: vec![WeightPrecision::new(Precision::Bf16).unwrap()],
            activations: vec![ActivationPrecision::new(Precision::Bf16).unwrap()],
            output: ActivationPrecision::new(Precision::Bf16).unwrap(),
            accumulation: AccumulationPolicy::Bf16InF32Acc,
            workspace_upper_bound: Dim::constant(0),
            oracle: TEST_ORACLE,
            partition: PartitionRule::ColumnShardable,
            state_effect: StateEffect::None,
        }
    }

    #[test]
    fn an_operation_whose_oracle_is_not_registered_cannot_be_lowered() {
        // F6: the old field was `has_host_oracle: bool`, set by the caller, and
        // accepted as proof. Now the claim lives in a registry the contract
        // cannot write to.
        let c = contract(Op::Linear);
        assert!(c.check_lowerable(&registry(Op::Linear)).is_ok());

        // An empty registry refuses it, however the contract is written.
        let e = c.check_lowerable(&OracleRegistry::new()).unwrap_err();
        assert_eq!(e.kind(), "unsupported_kernel");

        // So does a registry that has an oracle for a *different* operation.
        assert!(c.check_lowerable(&registry(Op::RmsNorm)).is_err());
    }

    #[test]
    fn an_oracle_registered_under_another_name_does_not_count() {
        let mut r = OracleRegistry::new();
        r.register(
            Op::Linear,
            OracleId("something_else"),
            OracleEvidence {
                implementation: "x",
                test_module: "y",
            },
        )
        .unwrap();
        assert!(contract(Op::Linear).check_lowerable(&r).is_err());
    }

    #[test]
    fn a_rejected_registration_leaves_the_registry_unchanged() {
        // Second review: the duplicate check returned an error *after* the
        // replacement had already happened, so the evidence changed anyway.
        let mut r = registry(Op::Linear);
        let before = r.evidence(Op::Linear, TEST_ORACLE).unwrap();

        assert!(
            r.register(
                Op::Linear,
                TEST_ORACLE,
                OracleEvidence {
                    implementation: "a_different_implementation",
                    test_module: "a_different_test",
                }
            )
            .is_err()
        );
        assert_eq!(r.len(), 1);
        assert_eq!(
            r.evidence(Op::Linear, TEST_ORACLE).unwrap(),
            before,
            "the rejected registration overwrote the entry it was meant to protect"
        );
    }

    #[test]
    fn low_bit_activations_cannot_be_requested_through_a_weight_dtype() {
        // The other half of F6. Under the old contract, `input_dtypes` was
        // validated with `is_legal_weight`, so INT4 activations passed. Now the
        // activation list is typed, and `ActivationPrecision::new` is the only
        // way to build an element of it.
        assert!(ActivationPrecision::new(Precision::Int4).is_err());
        assert!(ActivationPrecision::new(Precision::Int8).is_err());

        // A W4A16 contract: INT4 weights, BF16 activations. The two roles are
        // separate fields with separate rules.
        let mut c = contract(Op::Linear);
        c.weights = vec![WeightPrecision::new(Precision::Int4).unwrap()];
        assert!(c.check_lowerable(&registry(Op::Linear)).is_ok());
        assert_eq!(c.activations[0].get(), Precision::Bf16);
    }

    #[test]
    fn an_operation_with_no_declared_activations_cannot_be_lowered() {
        let mut c = contract(Op::Linear);
        c.activations.clear();
        assert!(c.check_lowerable(&registry(Op::Linear)).is_err());
    }

    #[test]
    fn a_state_touching_operation_must_declare_its_effect() {
        let mut c = contract(Op::Attention);
        assert!(
            c.check_lowerable(&registry(Op::Attention)).is_err(),
            "state_effect defaults to None, which an attention cannot mean"
        );
        c.state_effect = StateEffect::Appends;
        assert!(c.check_lowerable(&registry(Op::Attention)).is_ok());
    }

    #[test]
    fn partition_semantics_fail_closed_until_defined() {
        let mut c = contract(Op::Linear);
        c.partition = PartitionRule::NotDetermined;
        let e = c.check_partitionable().unwrap_err();
        assert_eq!(e.kind(), "unsupported");
        // ... but an undetermined partition does not block single-rank lowering.
        assert!(c.check_lowerable(&registry(Op::Linear)).is_ok());

        c.partition = PartitionRule::RowShardable;
        assert!(c.check_partitionable().is_ok());
        assert_eq!(PartitionRule::default(), PartitionRule::NotDetermined);
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
        let mut names: Vec<_> = Op::ALL.iter().map(|o| o.name()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(n, names.len(), "duplicate operation name");
    }
}
