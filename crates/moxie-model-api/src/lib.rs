//! What a model definition is allowed to be.
//!
//! Document 02: "A model definition is data and graph construction, not a
//! coroutine with access to the engine." There is deliberately no method
//! equivalent to "execute arbitrary model callback with the backend".
//!
//! M0 scope: the trait shape and the tensor-role vocabulary, so that the
//! ownership boundary exists in the type system before any adapter is written.

#![forbid(unsafe_code)]

use moxie_graph::Op;
use moxie_types::{Precision, Result};

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
    /// Precisions this tensor may legally be stored in. Document 03 allows
    /// sensitive tensors (router, norms, biases) to stay BF16/F32 while the rest
    /// of the family is low-bit.
    pub allowed: Vec<Precision>,
    pub required: bool,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub family: String,
    pub revision: String,
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

    /// Validate that this definition is internally consistent. Called by the
    /// registry before a model is admitted.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Tiny;

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
            &[]
        }
        fn graph_requirements(&self) -> GraphRequirements {
            GraphRequirements {
                ops: vec![Op::Embedding, Op::Linear, Op::RmsNorm],
            }
        }
    }

    #[test]
    fn a_definition_is_data_not_a_generator() {
        let m = Tiny;
        assert_eq!(m.metadata().family, "tiny");
        assert!(m.graph_requirements().ops.contains(&Op::Linear));
        assert!(m.validate().is_ok());
        // There is no `execute`, `forward`, `sample` or `backend` method to call.
        // If one is ever added, this comment is the place the review starts.
    }

    #[test]
    fn trained_position_is_metadata_not_a_supported_context_claim() {
        // R19: width, admitted context and actual visible tokens are different
        // things. This field is only the first of the three.
        assert_eq!(Tiny.metadata().max_trained_position, 4096);
    }
}
