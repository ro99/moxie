//! What a model definition is allowed to be.
//!
//! Document 02: "A model definition is data and graph construction, not a
//! coroutine with access to the engine." There is deliberately no method
//! equivalent to "execute arbitrary model callback with the backend".
//!
//! M0 scope: the trait shape and the tensor-role vocabulary, so that the
//! ownership boundary exists in the type system before any adapter is written.

#![forbid(unsafe_code)]

use moxie_graph::{Op, OracleRegistry};
use moxie_types::{Error, Result, WeightPrecision};

/// A logical tensor role. Importers map source names onto these; runtime code
/// never sees a checkpoint's naming (document 03).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorRole {
    pub name: String,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
}

/// What a model requires of an artifact, without naming any file.
#[derive(Debug, Clone)]
pub struct TensorRequirement {
    pub role: TensorRole,
    /// Storage precisions this tensor may legally use.
    ///
    /// Typed as [`WeightPrecision`] rather than a bare precision: document 03
    /// allows sensitive tensors (router, norms, biases) to stay BF16/F32 while
    /// the rest of the family is low-bit, and that is a *weight-storage*
    /// decision. It must not become a way to state an activation or cache dtype.
    pub allowed: Vec<WeightPrecision>,
    pub required: bool,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub family: String,
    pub revision: String,
    /// The longest position the released model was trained for.
    ///
    /// R19: width, admitted context and actual visible tokens are three
    /// different numbers, and this is only the first. It is not a supported
    /// context claim, and nothing may read it as one.
    pub max_trained_position: u64,
    pub vocab_size: u32,
}

/// The operations a model's graph uses, declared up front.
///
/// This is what lets the planner reject a model before execution when an
/// operation has no qualified kernel, instead of discovering it mid-forward.
#[derive(Debug, Clone)]
pub struct GraphRequirements {
    pub ops: Vec<Op>,
}

/// A model definition: metadata, tensor roles and graph composition. Nothing else.
///
/// Note what is absent, per document 09 §B: no allocation, no launch, no stream
/// or event, no file read, no cache or eviction, no KV page allocation, no
/// prefill/decode/sampling loop, no collective. `arch-check` enforces the same
/// boundary at the crate level; this trait is the shape that makes obeying it
/// natural.
pub trait ModelDefinition: core::fmt::Debug + Send + Sync {
    fn metadata(&self) -> &ModelMetadata;
    fn tensors(&self) -> &[TensorRequirement];
    fn graph_requirements(&self) -> GraphRequirements;
}

/// Admission checks a registry runs before a model is accepted.
///
/// These are **not** a method on the trait. The earlier draft had
/// `fn validate(&self) -> Result<()> { Ok(()) }`, which the M0 review flagged:
/// a default implementation returning success means every adapter is
/// self-certifying, and the one thing being checked is the thing writing the
/// check. Admission belongs to the registry, which is shared and which holds
/// the oracle registry the model does not.
///
/// An adapter may still add its own internal consistency assertions -- as tests
/// in its own crate, where they are evidence rather than a bypass.
pub fn admit(model: &dyn ModelDefinition, oracles: &OracleRegistry) -> Result<()> {
    let meta = model.metadata();
    if meta.family.is_empty() || meta.revision.is_empty() {
        return Err(Error::InvalidArtifact {
            detail: "a model definition must name its family and revision".into(),
        });
    }
    if meta.vocab_size == 0 {
        return Err(Error::InvalidArtifact {
            detail: format!("{} declares an empty vocabulary", meta.family),
        });
    }

    let mut seen = Vec::new();
    for t in model.tensors() {
        if t.allowed.is_empty() {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{}: tensor role {:?} allows no storage precision",
                    meta.family, t.role.name
                ),
            });
        }
        if seen.contains(&&t.role) {
            return Err(Error::InvalidArtifact {
                detail: format!("{}: duplicate tensor role {:?}", meta.family, t.role.name),
            });
        }
        seen.push(&t.role);
    }

    let req = model.graph_requirements();
    if req.ops.is_empty() {
        return Err(Error::InvalidArtifact {
            detail: format!("{} declares no operations", meta.family),
        });
    }
    // Every operation the model needs must have an independent host reference
    // somewhere in the shared registry. A model cannot supply its own.
    for op in &req.ops {
        if !oracles.has_any_oracle(*op) {
            return Err(Error::UnsupportedKernel {
                operation: op.name(),
                detail: format!(
                    "{} requires {}, which has no registered host oracle",
                    meta.family,
                    op.name()
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_graph::{OracleEvidence, OracleId};
    use moxie_types::Precision;

    #[derive(Debug)]
    struct Tiny {
        tensors: Vec<TensorRequirement>,
        ops: Vec<Op>,
    }

    impl Default for Tiny {
        fn default() -> Self {
            Self {
                tensors: vec![TensorRequirement {
                    role: TensorRole {
                        name: "embedding".into(),
                        layer: None,
                        expert: None,
                    },
                    allowed: vec![WeightPrecision::new(Precision::Bf16).unwrap()],
                    required: true,
                }],
                ops: vec![Op::Embedding, Op::Linear, Op::RmsNorm],
            }
        }
    }

    impl ModelDefinition for Tiny {
        fn metadata(&self) -> &ModelMetadata {
            static M: std::sync::OnceLock<ModelMetadata> = std::sync::OnceLock::new();
            M.get_or_init(|| ModelMetadata {
                family: "tiny".into(),
                revision: "test".into(),
                max_trained_position: 4096,
                vocab_size: 32,
            })
        }
        fn tensors(&self) -> &[TensorRequirement] {
            &self.tensors
        }
        fn graph_requirements(&self) -> GraphRequirements {
            GraphRequirements {
                ops: self.ops.clone(),
            }
        }
    }

    fn oracles_for(ops: &[Op]) -> OracleRegistry {
        let mut r = OracleRegistry::new();
        for op in ops {
            r.register(
                *op,
                OracleId("host_reference"),
                OracleEvidence {
                    implementation: "test",
                    test_module: "test",
                },
            )
            .unwrap();
        }
        r
    }

    #[test]
    fn a_definition_is_data_not_a_generator() {
        let m = Tiny::default();
        assert_eq!(m.metadata().family, "tiny");
        assert!(m.graph_requirements().ops.contains(&Op::Linear));
        // There is no `execute`, `forward`, `sample` or `backend` method to call.
        // If one is ever added, this comment is the place the review starts.
    }

    #[test]
    fn a_model_cannot_certify_itself() {
        // The F6 correction: admission is the registry's, and it fails when the
        // shared registry has no oracle for an operation the model needs. The
        // model has no way to change that answer.
        let m = Tiny::default();
        assert!(admit(&m, &OracleRegistry::new()).is_err());
        assert!(admit(&m, &oracles_for(&[Op::Embedding])).is_err());
        assert!(admit(&m, &oracles_for(&m.ops)).is_ok());
    }

    #[test]
    fn an_incoherent_definition_is_refused() {
        let full = oracles_for(&[Op::Embedding, Op::Linear, Op::RmsNorm]);

        let mut no_ops = Tiny::default();
        no_ops.ops.clear();
        assert!(admit(&no_ops, &full).is_err());

        let mut no_precision = Tiny::default();
        no_precision.tensors[0].allowed.clear();
        assert!(admit(&no_precision, &full).is_err());

        let mut duplicate = Tiny::default();
        let first = duplicate.tensors[0].clone();
        duplicate.tensors.push(first);
        assert!(admit(&duplicate, &full).is_err());
    }

    #[test]
    fn a_tensor_role_states_storage_precision_only() {
        // `allowed` is a list of weight-storage precisions. There is no way to
        // write an activation or cache dtype into it, because the element type
        // is `WeightPrecision` and its constructor enforces the weight rule.
        let m = Tiny::default();
        assert_eq!(m.tensors()[0].allowed[0].get(), Precision::Bf16);
        // Document 03 allows a sensitive tensor to stay BF16 while the family is
        // low-bit; both are expressible.
        assert!(WeightPrecision::new(Precision::Int4).is_ok());
    }

    #[test]
    fn trained_position_is_metadata_not_a_supported_context_claim() {
        // R19: width, admitted context and actual visible tokens are different
        // things. This field is only the first of the three.
        assert_eq!(Tiny::default().metadata().max_trained_position, 4096);
    }
}
