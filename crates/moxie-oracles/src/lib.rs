//! Independent host reference implementations, and the tiny source-linked
//! fixtures that pin them.
//!
//! Document 06 M0.6: "Freeze a tiny source-linked oracle corpus: INT4/INT8
//! signed decoding, scales/zero points/group maps, BF16 rounding, routed
//! experts, attention masks, recurrent updates, sampling distributions,
//! tokenizer/templates and protocol frames."
//!
//! The first three live in `moxie-format`, next to the format they pin. The rest
//! live here, because they are *operation* mathematics rather than artifact
//! mathematics, and because document 02 requires every shared semantic operation
//! to have "an independent oracle" that is not the implementation being checked.
//!
//! ## What this crate is and is not
//!
//! It is a set of small, explicit, obviously-correct reference implementations
//! with exhaustive or enumerable tests. Task 0001 deferred these on the grounds
//! that they needed an executor or a licensed checkpoint; the M0 review rejected
//! that reasoning for tiny synthetic equations, mask patterns, distributions and
//! protocol frames, and it was right -- nothing here needs either.
//!
//! It is **not** an interpreter, an executor, or a fast path. Nothing here
//! allocates device memory, reads a file, or knows a model's name. The M1 slice
//! builds the interpreter that consumes these as its comparison points.
//!
//! Coverage is deliberately partial and says so: each module's header lists what
//! it pins and what it explicitly does not, so a later reader does not mistake a
//! bounded fixture for a finished contract.

#![forbid(unsafe_code)]

pub mod activation;
pub mod attention;
pub mod linear;
pub mod mask;
pub mod metric;
pub mod norm;
pub mod protocol;
pub mod recurrent;
pub mod residual;
pub mod rope;
pub mod route;
pub mod sampler;
pub mod template;

use moxie_graph::{Op, OracleEvidence, OracleId, OracleRegistry};
use moxie_types::Result;

/// The one oracle name this crate registers under.
pub const HOST_REFERENCE: OracleId = OracleId("moxie_oracles::host_reference");

/// Register every operation this crate actually supplies a reference for.
///
/// The composition root calls this. An operation absent from the list has no
/// registered oracle and therefore cannot be lowered -- which is the point of
/// the registry, and why this function returns a short list rather than looping
/// over `Op::ALL`.
pub fn register(registry: &mut OracleRegistry) -> Result<()> {
    let entries: &[(Op, &'static str, &'static str)] = &[
        // Task 0003's slice.
        (Op::Embedding, "moxie_oracles::linear", "linear::tests"),
        (Op::Linear, "moxie_oracles::linear", "linear::tests"),
        (
            Op::VocabProjection,
            "moxie_oracles::linear",
            "linear::tests",
        ),
        (Op::RmsNorm, "moxie_oracles::norm", "norm::tests"),
        (Op::SwiGlu, "moxie_oracles::activation", "activation::tests"),
        (Op::Rope, "moxie_oracles::rope", "rope::tests"),
        (
            Op::Attention,
            "moxie_oracles::attention",
            "attention::tests",
        ),
        (Op::Residual, "moxie_oracles::residual", "residual::tests"),
        // Fixtures from M0, whose operations have references but no interpreter
        // consumer yet.
        (Op::Route, "moxie_oracles::route", "route::tests"),
        (Op::Dispatch, "moxie_oracles::route", "route::tests"),
        (Op::Combine, "moxie_oracles::route", "route::tests"),
        (
            Op::RecurrentUpdate,
            "moxie_oracles::recurrent",
            "recurrent::tests",
        ),
        (
            Op::ShortConv,
            "moxie_oracles::recurrent",
            "recurrent::tests",
        ),
    ];
    for (op, implementation, test_module) in entries {
        registry.register(
            *op,
            HOST_REFERENCE,
            OracleEvidence {
                implementation,
                test_module,
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_covers_exactly_what_is_implemented() {
        let mut r = OracleRegistry::new();
        register(&mut r).unwrap();

        for op in [
            Op::Embedding,
            Op::Linear,
            Op::VocabProjection,
            Op::RmsNorm,
            Op::SwiGlu,
            Op::Rope,
            Op::Attention,
            Op::Residual,
            Op::Route,
            Op::Dispatch,
            Op::Combine,
            Op::RecurrentUpdate,
            Op::ShortConv,
        ] {
            assert!(r.has_any_oracle(op), "{} should be registered", op.name());
        }
        // Nothing is registered for operations this crate does not implement.
        // An over-broad registration would let an unvalidated operation lower.
        for op in [
            Op::MlaAttention,
            Op::SparseIndexSelect,
            // R06: bounded in both terms, and not SwiGlu. Registering SwiGlu's
            // reference for it would be the substitution R06 warns against.
            Op::SituGlu,
            Op::GeGlu,
            Op::ResidualMix,
            Op::LayerNorm,
            Op::ExpertLinear,
            Op::ExpertMlp,
        ] {
            assert!(
                !r.has_any_oracle(op),
                "{} has no reference here and must not be registered",
                op.name()
            );
        }
    }
}
