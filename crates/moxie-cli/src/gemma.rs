//! Composition root for the reduced Gemma-4-like diagnostic graph.
//!
//! The model module names roles and builds the graph; this module invents the
//! bytes. That split is the point: nothing here is model mathematics, and
//! nothing in `moxie_models::gemma4` allocates or reads a file.
//!
//! **These weights are synthetic.** They are a deterministic pattern, not
//! Gemma 4's parameters, and the tokens this graph emits are not model output.
//! Executing the real artifact needs M3's INT8 importer; see
//! `docs/models/gemma4.md`.

use crate::fixture::Fixture;
use moxie_engine::{HostTensor, Value};
use moxie_graph::{Bindings, OracleRegistry};
use moxie_models::gemma4::{Fraction, Gemma4Text, Reduction, TextConfig, embedding_scale};
use moxie_types::{Dim, Error, Result, SymbolId};

/// Which reduced geometry to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Six layers on the artifact's own 1-in-6 global stride, so five sliding
    /// layers precede one full-causal layer. 4 query heads over 2 key/value
    /// heads.
    A,
    /// Three layers on a 1-in-3 stride, with a different head ratio, head
    /// dimension, window and residual-stream width. Nothing the two geometries
    /// share is load-bearing by accident.
    B,
}

impl Shape {
    pub fn config(self) -> TextConfig {
        match self {
            Shape::A => TextConfig {
                hidden: 24,
                layers: 6,
                heads: 4,
                kv_heads: 2,
                // Sixteen, not eight. A quarter of an eight-wide head is a
                // single rotated pair, whose inverse frequency is `base^0` --
                // so the global theta would cancel and the geometry would
                // silently stop testing it. Sixteen gives the global layers two
                // angles, which is the smallest size at which the base matters.
                head_dim: 16,
                intermediate: 16,
                vocab: 11,
                global_stride: 6,
                sliding_window: 3,
                rms_eps: 1e-6,
                sliding_rope_theta: 10_000.0,
                global_rope_theta: 1_000_000.0,
                global_partial_rotary: Fraction::QUARTER,
                final_logit_softcap: 30.0,
                layer_scalars: vec![1.0, 0.75, 1.25, 0.5, 1.5, 0.875],
                embedding_scale: embedding_scale(24),
                max_trained_position: 256,
            },
            Shape::B => TextConfig {
                hidden: 12,
                layers: 3,
                heads: 6,
                kv_heads: 3,
                // Half of eight is four: two angles again, for the same reason.
                head_dim: 8,
                intermediate: 20,
                vocab: 7,
                global_stride: 3,
                sliding_window: 5,
                rms_eps: 1e-5,
                sliding_rope_theta: 500.0,
                global_rope_theta: 250_000.0,
                // A half rather than a quarter, so the rotary fraction is not
                // the same constant in both geometries.
                global_partial_rotary: Fraction {
                    numerator: 1,
                    denominator: 2,
                },
                final_logit_softcap: 4.0,
                layer_scalars: vec![0.625, 1.75, 1.0],
                embedding_scale: embedding_scale(12),
                max_trained_position: 256,
            },
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Shape::A => "gemma-a",
            Shape::B => "gemma-b",
        }
    }
}

/// One line naming what the reduced graph is not, for the diagnostic surface.
///
/// Printed before anything executes. Document 06: "Never describe synthetic
/// output as model support" -- so the surface that produces the output says so
/// itself rather than relying on a reader having found the documentation.
pub fn reduction_line(r: Reduction) -> String {
    let mut parts = Vec::new();
    if r.synthetic_weights {
        parts.push("synthetic-bf16-weights");
    }
    if r.uniform_kv_geometry {
        parts.push("uniform-kv-geometry");
    }
    if r.sliding_layers_retain_full_history {
        parts.push("no-window-eviction");
    }
    if r.text_only {
        parts.push("text-only");
    }
    parts.join(",")
}

/// Build the reduced graph and fill every declared role.
pub fn build(shape: Shape) -> Result<Fixture> {
    build_with_config(shape.config())
}

pub fn build_with_config(config: TextConfig) -> Result<Fixture> {
    let model = Gemma4Text::reduced(config, "synthetic-reduced")?;
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles)?;
    // The registry admission the shared API defines, run before the graph is
    // used rather than after: a model that declares an operation with no
    // reference must not reach an interpreter.
    moxie_model_api::admit(&model, &oracles)?;
    let composed = model.compose(&oracles, SymbolId(0))?;

    let mut weights = Bindings::new();
    for (index, bound) in composed.weights.iter().enumerate() {
        let spec = composed
            .graph
            .spec(bound.value)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("role {} has no graph value", bound.role.name),
            })?;
        let extent: Vec<usize> = spec
            .shape
            .iter()
            .map(|d| match d {
                Dim::Const(n) => Ok(*n as usize),
                other => Err(Error::InvalidArtifact {
                    detail: format!(
                        "weight {} has a non-constant extent {other:?}",
                        bound.role.name
                    ),
                }),
            })
            .collect::<Result<_>>()?;
        let count: usize = extent.iter().product();
        // The unit gain the value normalization is defined against. A pattern
        // here would silently make V's norm a scaling too, and the difference
        // from `k_norm` -- which does carry a gain -- is exactly what the
        // fixture is for.
        let data: Vec<f32> = if bound.role.name == "v_norm_unit_gain" {
            vec![1.0; count]
        } else {
            (0..count)
                .map(|i| {
                    let mixed = (i * 13 + index * 7 + 3) % 31;
                    (mixed as f32 - 15.0) / 32.0
                })
                .collect()
        };
        weights.set(bound.value, Value::Float(HostTensor::bf16(data, extent)?));
    }

    Ok(Fixture {
        graph: composed.graph,
        weights,
        tokens: composed.tokens,
        positions: composed.positions,
    })
}
